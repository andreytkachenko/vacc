//! NVDEC AV1 decoder using vacc-parser (Av1Parser).
//!
//! Mirrors the structure of [`crate::vp9::NvdecVp9Decoder`] and the DPB logic
//! of the Vulkan AV1 decoder (`crates/vacc-vulkan/src/av1.rs`):
//!
//! - IVF packets are OBU-walked; the SPS (OBU type 1) is parsed once, and each
//!   Frame OBU (type 6) / show_existing FrameHeader OBU (type 3) is parsed with
//!   [`Av1Parser::parse_frame_header`].
//! - The DPB tracks frame-buffer -> DPB-slot (surface) mapping with FIFO slot
//!   allocation. `refresh_frame_flags` bit `i` refreshes frame buffer `i`
//!   (the parser's convention, matching the C++ `UpdateFramePointers`).
//! - Each real frame's [`CUVIDPICPARAMS`] is built by
//!   [`build_cuvid_av1_picparams`] and submitted to `cuvidDecodePicture`. The
//!   bitstream passed to cuvid is the **tile data only** (Frame OBU payload
//!   minus the frame header); `frame_offset` carries the `order_hint`.
//! - `show_existing_frame` commands re-display the referenced frame buffer's
//!   surface (no new decode).
//!
//! Field mapping (verified against the cuvid-parser baseline dump):
//! - `ref_frame_map[fb]` = DPB slot of frame buffer `fb` (255 if unmapped).
//! - `ref_frame[i].index` = DPB slot of `ref_frame_idx[i]` (255 if unmapped);
//!   `.width/.height` = that frame buffer's coded dims.
//! - `primary_ref_frame` = DPB slot of the primary reference (255 if absent).

use std::collections::VecDeque;
use std::os::raw::{c_int, c_uint};
use std::sync::Mutex;

use vacc_core::{
    codec::VideoCodec,
    decoder::{Decoder, DecoderInfo},
    format::{ChromaSubsampling, ComponentBitDepth, VideoFormat},
    frame::{DecodedFrame, FieldFlags, PixelData, PixelPlane},
    picture::Av1Sps,
    session::Extent2D,
};
use vacc_parser::av1::{Av1FrameHeader, Av1Parser};
use vacc_parser::av1_dpb::{Av1Dpb, AV1_NUM_FRAME_BUFFERS};
use vacc_parser::{DetectedVideoFormat, VideoParser};

use crate::{
    device::{
        cu_ctx_set_current, cu_ctx_synchronize, cu_mem_free_host, cu_mem_host_alloc, cu_memcpy_2d,
        get_funcs, init_nvdec, query_decoder_caps, CUDA_MEMCPY2D, CU_MEMORYTYPE_DEVICE,
        CU_MEMORYTYPE_HOST,
    },
    error::{NvdecError, NvdecResult},
    ffi::{
        cudaVideoChromaFormat, cudaVideoCodec, cudaVideoDeinterlaceMode, cudaVideoSurfaceFormat,
        CUdeviceptr, CUvideodecoder, CUDA_SUCCESS, CUVIDAV1GLOBALMOTION, CUVIDAV1PICPARAMS,
        CUVIDAV1REFFRAME, CUVIDDECODECREATEINFO, CUVIDPICPARAMS, CUVIDPROCPARAMS, CUVIDRECT,
    },
};

/// Number of decode surfaces / DPB slots (matches the cuvid parser baseline).
const NUM_SURFACES: u32 = 16;

/// IVF header size in bytes (packets start at offset 32).
const IVF_HEADER_SIZE: usize = 32;

/// Zero-filled tail padding appended to the bitstream host buffer.
///
/// The AV1 decoder's bit reader can read SIMD chunks that extend past
/// `nBitstreamDataLen`, so the tail must be valid (zero) memory, not
/// uninitialized.
const BITSTREAM_PADDING: usize = 4096;

// ============================================================================
// OBU extraction
// ============================================================================

/// A Frame OBU (type 6) or show_existing FrameHeader OBU (type 3) payload.
struct Av1FrameObu {
    /// The OBU payload (frame header + tile data for Frame OBUs; frame header
    /// only for show_existing FrameHeader OBUs).
    payload: Vec<u8>,
}

/// Find the Sequence Header OBU (type 1) payload in a packet, if present.
fn find_sps_obu(packet: &[u8]) -> Option<Vec<u8>> {
    let mut pos = 0;
    while pos < packet.len().saturating_sub(1) {
        let first = packet[pos];
        let obu_type = (first >> 3) & 0x0F;
        let ext = (first >> 2) & 1;
        let has_size = (first >> 1) & 1 != 0;
        let header_size = 1 + ext as usize;
        if has_size && pos + header_size < packet.len() {
            let mut size: usize = 0;
            let mut shift = 0;
            let mut size_pos = pos + header_size;
            loop {
                if size_pos >= packet.len() {
                    break;
                }
                let b = packet[size_pos];
                size |= ((b & 0x7F) as usize) << shift;
                shift += 7;
                size_pos += 1;
                if b & 0x80 == 0 {
                    break;
                }
            }
            if obu_type == 1 {
                let payload_start = size_pos;
                let payload_end = (payload_start + size).min(packet.len());
                return Some(packet[payload_start..payload_end].to_vec());
            }
            let next = size_pos + size;
            pos = if next > pos { next } else { size_pos + 1 };
        } else {
            pos += header_size.max(1);
        }
    }
    None
}

/// Extract all Frame OBUs (type 6) and show_existing FrameHeader OBUs (type 3)
/// from a packet, in order.
///
/// A type-3 FrameHeader OBU is extracted only when its payload signals
/// `show_existing_frame = 1` (MSB of the first payload byte). Those carry no
/// tile data — the decode loop re-displays the referenced frame buffer instead
/// of issuing a GPU decode. Redundant frame headers (type 3 with
/// `show_existing_frame = 0`) are skipped: the corresponding Frame OBU is
/// decoded instead (C++ behavior).
fn extract_frame_obus(packet: &[u8]) -> Vec<Av1FrameObu> {
    let mut obus = Vec::new();
    let mut pos = 0;
    while pos < packet.len().saturating_sub(1) {
        let first = packet[pos];
        let obu_type = (first >> 3) & 0x0F;
        let ext = (first >> 2) & 1;
        let has_size = (first >> 1) & 1 != 0;
        let header_size = 1 + ext as usize;
        if has_size && pos + header_size < packet.len() {
            let mut size: usize = 0;
            let mut shift = 0;
            let mut size_pos = pos + header_size;
            loop {
                if size_pos >= packet.len() {
                    break;
                }
                let b = packet[size_pos];
                size |= ((b & 0x7F) as usize) << shift;
                shift += 7;
                size_pos += 1;
                if b & 0x80 == 0 {
                    break;
                }
            }
            let is_frame = obu_type == 6;
            let is_show_existing = obu_type == 3
                && size > 0
                && size_pos < packet.len()
                && (packet[size_pos] & 0x80) != 0;
            if is_frame || is_show_existing {
                let payload_start = size_pos;
                let payload_end = (payload_start + size).min(packet.len());
                obus.push(Av1FrameObu {
                    payload: packet[payload_start..payload_end].to_vec(),
                });
            }
            let next = size_pos + size;
            pos = if next > pos { next } else { size_pos + 1 };
        } else {
            pos += header_size.max(1);
        }
    }
    obus
}

// ============================================================================
// CUVIDPICPARAMS construction
// ============================================================================

/// Build a complete [`CUVIDPICPARAMS`] for one AV1 frame from parser output +
/// the common DPB state.
///
/// `fh` is the [`Av1FrameHeader`] (its `ref_frame_idx` is already fully
/// resolved by the parser via the common DPB), `sps` the [`Av1Sps`], `dpb`
/// the running common [`Av1Dpb`], and `bitstream_ptr`/`bitstream_len` point at
/// the tile data (Frame OBU payload minus the frame header). The pointer must
/// stay valid for the duration of the subsequent `cuvidDecodePicture` call.
///
/// This allocates the output slot and computes the reference mapping from the
/// **pre-decode** DPB state. It does NOT apply this frame's refresh — the
/// caller does that after the decode (see [`Av1Dpb::commit_decoded`]).
pub fn build_cuvid_av1_picparams(
    fh: &Av1FrameHeader,
    sps: &Av1Sps,
    dpb: &mut Av1Dpb,
    bitstream_ptr: *const u8,
    bitstream_len: u32,
    ts_90k: u64,
    slice_offsets: *const c_uint,
) -> CUVIDPICPARAMS {
    let is_key = fh.frame_type == 0;
    let is_intra_only = fh.frame_type == 2;

    // 1. Output slot: key frame / first frame -> slot 0 + reset DPB; else FIFO.
    let output_slot = if is_key || dpb.decoded_frames() == 0 {
        dpb.reset_for_keyframe();
        0
    } else {
        dpb.allocate_output_slot()
    };

    // 2. Effective ref_frame_idx (reference name -> frame buffer index). The
    // parser already resolved short signaling via the common DPB.
    let effective_ref_frame_idx: [i32; 7] = std::array::from_fn(|i| fh.ref_frame_idx[i] as i32);

    // 3. AV1-specific parameters.
    let mut av1 = unsafe { std::mem::zeroed::<CUVIDAV1PICPARAMS>() };
    av1.width = fh.frame_width;
    av1.height = fh.frame_height;
    av1.frame_offset = fh.order_hint;
    // decodePicIdx must be the OUTPUT SURFACE index (== CurrPicIdx), NOT the
    // decode-order counter. Verified against the NVIDIA cuvid-parser baseline:
    // once the surface pool wraps, decodePicIdx tracks CurrPicIdx (e.g. 5,4,6,3),
    // never exceeding num_surfaces-1. Using the decode counter here produced an
    // out-of-range index (>= num_surfaces) and cuvidDecodePicture error 719.
    av1.decodePicIdx = output_slot as c_int;

    // Sequence header bitfields.
    let bit_depth_minus8 = if sps.twelve_bit {
        4
    } else if sps.high_bitdepth {
        2
    } else {
        0
    };
    let mut seq_flags: u32 = 0;
    seq_flags |= sps.profile as u32 & 0x7;
    seq_flags |= (sps.use_128x128_superblock as u32 & 1) << 3;
    seq_flags |= (sps.subsampling_x as u32 & 1) << 4;
    seq_flags |= (sps.subsampling_y as u32 & 1) << 5;
    seq_flags |= (sps.mono_chrome as u32 & 1) << 6;
    seq_flags |= (bit_depth_minus8 & 0xF) << 7;
    seq_flags |= (sps.enable_filter_intra as u32 & 1) << 11;
    seq_flags |= (sps.enable_intra_edge_filter as u32 & 1) << 12;
    seq_flags |= (sps.enable_interintra_compound as u32 & 1) << 13;
    seq_flags |= (sps.enable_masked_compound as u32 & 1) << 14;
    seq_flags |= (sps.enable_dual_filter as u32 & 1) << 15;
    seq_flags |= (sps.enable_order_hint as u32 & 1) << 16;
    seq_flags |= (sps.order_hint_bits_minus1 as u32 & 0x7) << 17;
    seq_flags |= (sps.enable_jnt_motion as u32 & 1) << 20;
    seq_flags |= (sps.enable_superres as u32 & 1) << 21;
    seq_flags |= (sps.enable_cdef as u32 & 1) << 22;
    seq_flags |= (sps.enable_restoration as u32 & 1) << 23;
    // The FGS (film grain synthesis) flag is intentionally left clear,
    // matching the NVIDIA cuvid-parser baseline (enable_fgs=0): the decoder
    // does not apply film grain, so the flag must stay 0.
    av1.seq_flags = seq_flags;

    // Frame header bitfields.
    let mut frame_flags: u32 = 0;
    frame_flags |= fh.frame_type as u32 & 0x3;
    frame_flags |= (fh.show_frame as u32 & 1) << 2;
    frame_flags |= (fh.disable_cdf_update as u32 & 1) << 3;
    frame_flags |= (fh.allow_screen_content_tools as u32 & 1) << 4;
    frame_flags |= (fh.force_integer_mv as u32 & 1) << 5;
    frame_flags |= (fh.coded_denom as u32 & 0x7) << 6;
    frame_flags |= (fh.allow_intrabc as u32 & 1) << 9;
    frame_flags |= (fh.allow_high_precision_mv as u32 & 1) << 10;
    frame_flags |= (fh.interpolation_filter as u32 & 0x7) << 11;
    // Bit 14 is `switchable_motion_mode` (is_motion_mode_switchable), NOT
    // is_filter_switchable. The switchable interpolation filter is already
    // signalled by interpolation_filter == 4 (SWITCHABLE) in bits 11-13.
    // Verified against the NVIDIA cuvid-parser baseline: bit 14 is 1 only for
    // frames with is_motion_mode_switchable (e.g. DECODE 1/17), independent of
    // is_filter_switchable.
    frame_flags |= (fh.is_motion_mode_switchable as u32 & 1) << 14;
    frame_flags |= (fh.use_ref_frame_mvs as u32 & 1) << 15;
    frame_flags |= (fh.disable_frame_end_update_cdf as u32 & 1) << 16;
    frame_flags |= (fh.delta_q_present as u32 & 1) << 17;
    frame_flags |= (fh.delta_q_res as u32 & 0x3) << 18;
    frame_flags |= (fh.using_qmatrix as u32 & 1) << 20;
    frame_flags |= (fh.coded_lossless as u32 & 1) << 21;
    frame_flags |= (fh.use_superres as u32 & 1) << 22;
    frame_flags |= (fh.tx_mode as u32 & 0x3) << 23;
    frame_flags |= (fh.reference_select as u32 & 1) << 25;
    frame_flags |= (fh.allow_warped_motion as u32 & 1) << 26;
    frame_flags |= (fh.reduced_tx_set as u32 & 1) << 27;
    frame_flags |= (fh.skip_mode_present as u32 & 1) << 28;
    av1.frame_flags = frame_flags;

    // Tiling.
    av1.tile_info = (fh.tile_cols & 0xFF)
        | ((fh.tile_rows & 0xFF) << 8)
        | ((fh.context_update_tile_id & 0xFFFF) << 16);
    // cuvid expects the tile width/height in superblocks (the actual count),
    // whereas the parser stores `*_in_sbs_minus_1` (count minus 1). Verified
    // against the NVIDIA cuvid-parser baseline: 1920x1080 (64px superblocks,
    // single tile) -> baseline tile_widths[0]=30 / tile_heights[0]=17 vs the
    // parser's 29 / 16. Only the used tile slots get the +1; the rest stay 0
    // (the baseline leaves unused slots zeroed).
    let num_cols = (fh.tile_cols as usize).min(64);
    let num_rows = (fh.tile_rows as usize).min(64);
    for i in 0..64 {
        av1.tile_widths[i] = if i < num_cols {
            fh.tile_width_in_sbs_minus_1[i].saturating_add(1)
        } else {
            0
        };
        av1.tile_heights[i] = if i < num_rows {
            fh.tile_height_in_sbs_minus_1[i].saturating_add(1)
        } else {
            0
        };
    }

    // CDEF. The parser's `cdef_damping` is already the bitstream's
    // `cdef_damping_minus_3` (2-bit value), so pass it through unchanged.
    // (Verified against the NVIDIA cuvid-parser baseline: inter frames carry
    // cdef_damping_minus_3=1, which the previous `saturating_sub(3)` clamped
    // to 0, corrupting every CDEF-filtered block.)
    let cdef_damping_minus_3 = fh.cdef_damping;
    av1.cdef_flags = (cdef_damping_minus_3 & 0x3) | ((fh.cdef_bits & 0x3) << 2);
    // Zero CDEF units not re-coded this frame (match the NVIDIA cuvid-parser
    // baseline: it only carries the refreshed units, zeroing the rest). With
    // this, picparams are byte-identical to the baseline.
    let recoded = 1usize << (fh.cdef_bits & 0x3);
    for i in 0..8 {
        if i < recoded {
            av1.cdef_y_strength[i] =
                (fh.cdef_y_pri_strength[i] & 0xF) | ((fh.cdef_y_sec_strength[i] & 0xF) << 4);
            av1.cdef_uv_strength[i] =
                (fh.cdef_uv_pri_strength[i] & 0xF) | ((fh.cdef_uv_sec_strength[i] & 0xF) << 4);
        } else {
            av1.cdef_y_strength[i] = 0;
            av1.cdef_uv_strength[i] = 0;
        }
    }

    // Skip mode.
    av1.skip_mode_frames = (fh.skip_mode_frame[0] & 0xF) | ((fh.skip_mode_frame[1] & 0xF) << 4);

    // Quantization.
    av1.base_qindex = fh.base_q_index;
    av1.qp_y_dc_delta_q = fh.delta_q_y_dc;
    av1.qp_u_dc_delta_q = fh.delta_q_u_dc;
    av1.qp_v_dc_delta_q = fh.delta_q_v_dc;
    av1.qp_u_ac_delta_q = fh.delta_q_u_ac;
    av1.qp_v_ac_delta_q = fh.delta_q_v_ac;
    av1.qm_y = fh.qm_y;
    av1.qm_u = fh.qm_u;
    av1.qm_v = fh.qm_v;

    // Segmentation.
    let mut segmentation_flags: u8 = 0;
    segmentation_flags |= fh.segmentation_enabled as u8 & 1;
    segmentation_flags |= (fh.segmentation_update_map as u8 & 1) << 1;
    segmentation_flags |= (fh.segmentation_update_data as u8 & 1) << 2;
    segmentation_flags |= (fh.segmentation_temporal_update as u8 & 1) << 3;
    av1.segmentation_flags = segmentation_flags;
    av1.segmentation_feature_mask = fh.segment_feature_enabled;
    for i in 0..8 {
        for j in 0..8 {
            av1.segmentation_feature_data[i * 8 + j] = fh.segment_feature_data[i][j];
        }
    }

    // Loop filter.
    av1.loop_filter_level[0] = fh.loop_filter_level[0];
    av1.loop_filter_level[1] = fh.loop_filter_level[1];
    av1.loop_filter_level_u = fh.loop_filter_level_uv[0];
    av1.loop_filter_level_v = fh.loop_filter_level_uv[1];
    av1.loop_filter_sharpness = fh.loop_filter_sharpness;
    av1.loop_filter_ref_deltas = fh.loop_filter_ref_deltas;
    av1.loop_filter_mode_deltas = fh.loop_filter_mode_deltas;
    let mut loop_filter_flags: u8 = 0;
    loop_filter_flags |= fh.loop_filter_delta_enabled as u8 & 1;
    loop_filter_flags |= (fh.loop_filter_delta_update as u8 & 1) << 1;
    loop_filter_flags |= (fh.delta_lf_present as u8 & 1) << 2;
    loop_filter_flags |= (fh.delta_lf_res & 0x3) << 3;
    loop_filter_flags |= (fh.delta_lf_multi as u8 & 1) << 5;
    av1.loop_filter_flags = loop_filter_flags;

    // Loop restoration. The parser stores the spec codes directly
    // (0: 32px, 1: 64px, 2: 128px, 3: 256px) — the same numbering
    // cuviddec.h documents for `lr_unit_size`.
    for i in 0..3 {
        av1.lr_unit_size[i] = fh.loop_restoration_size[i] as u8;
        av1.lr_type[i] = fh.loop_restoration_type[i];
    }

    // Reference mapping (from the pre-decode DPB state). The common DPB
    // reports `-1` for an empty frame buffer; cuvid's sentinel is 255.
    for fb in 0..AV1_NUM_FRAME_BUFFERS {
        av1.ref_frame_map[fb] = dpb.slot_of_frame_buffer(fb) as u8;
    }
    for (i, &eff) in effective_ref_frame_idx.iter().enumerate() {
        let fb = eff as usize;
        let slot = if fb < AV1_NUM_FRAME_BUFFERS {
            dpb.slot_of_frame_buffer(fb)
        } else {
            -1
        };
        if slot >= 0 {
            let (w, h) = dpb.frame_buffer_dims(fb);
            av1.ref_frame[i] = CUVIDAV1REFFRAME {
                width: w,
                height: h,
                index: slot as u8,
                reserved24Bits: [0; 3],
            };
        } else {
            av1.ref_frame[i] = CUVIDAV1REFFRAME {
                width: 0,
                height: 0,
                index: 255,
                reserved24Bits: [0; 3],
            };
        }
    }

    // Primary reference frame: DPB slot of the primary reference (255 if the
    // primary ref name is absent (7) or this is a key frame).
    let pr = fh.primary_ref_frame as usize;
    if is_key || pr >= 7 {
        av1.primary_ref_frame = 255;
    } else {
        let fb = effective_ref_frame_idx[pr] as usize;
        let slot = if fb < AV1_NUM_FRAME_BUFFERS {
            dpb.slot_of_frame_buffer(fb)
        } else {
            -1
        };
        av1.primary_ref_frame = if slot >= 0 { slot as u8 } else { 255 };
    }

    // Global motion. The cuvid parser emits the identity matrix for INVALID
    // (type 0) and IDENTITY (type 1) global motion; the parser leaves the
    // params all-zero for those, so substitute the 16.16 identity matrix.
    for i in 0..7 {
        let gm_type = fh.global_motion_type[i];
        let gm_params = if gm_type <= 1 {
            [0i32, 0, 65536, 0, 0, 65536]
        } else {
            fh.global_motion_params[i]
        };
        av1.global_motion[i] = CUVIDAV1GLOBALMOTION {
            flags: (gm_type & 0x3) << 1,
            reserved24Bits: [0; 3],
            wmmat: gm_params,
        };
    }

    // Film grain (only apply_grain is signalled by the parser).
    av1.film_grain_flags = fh.apply_grain as u16 & 1;

    // 4. Common CUVIDPICPARAMS.
    let mut params = unsafe { std::mem::zeroed::<CUVIDPICPARAMS>() };
    params.PicWidthInMbs = (fh.frame_width / 16) as c_int;
    params.FrameHeightInMbs = (fh.frame_height / 16) as c_int;
    params.CurrPicIdx = output_slot as c_int;
    params.field_pic_flag = 0;
    params.bottom_field_flag = 0;
    params.second_field = 0;
    params.nBitstreamDataLen = bitstream_len;
    params.pBitstreamData = bitstream_ptr;
    params.nNumSlices = 1;
    params.pSliceDataOffsets = slice_offsets;
    params.ref_pic_flag = (fh.refresh_frame_flags != 0) as c_int;
    params.intra_pic_flag = (is_key || is_intra_only) as c_int;
    // Display timestamp on the 90 kHz clock in Reserved[0]: NVIDIA's own
    // cuvid parser sets it from the packet timestamp, and the NVDEC AV1
    // decoder consumes it. Leaving it zero makes inter-frame reconstruction
    // diverge (small error that propagates through the reference chain).
    let mut reserved = [0u32; 30];
    reserved[0] = (ts_90k & 0xFFFF_FFFF) as u32;
    params.Reserved = reserved;
    params.CodecSpecific.av1 = av1;

    params
}

// ============================================================================
// NvdecAv1Decoder
// ============================================================================

/// NVDEC AV1 decoder using vacc-parser.
///
/// Not `Send`/`Sync`; use from a single thread. The CUDA context must be set
/// current before decode methods.
///
/// Driven by the [`Av1Parser`] (which owns the common AV1 DPB for surface
/// management): IVF packets (or a raw single frame) are OBU-walked, parsed,
/// and each frame's [`CUVIDPICPARAMS`] is built by
/// [`build_cuvid_av1_picparams`] and submitted to `cuvidDecodePicture`.
/// Displayed frames are extracted in display order (NV12 -> planar YUV420P).
pub struct NvdecAv1Decoder {
    parser: Av1Parser,
    decoder: Mutex<CUvideodecoder>,
    info: Mutex<DecoderInfo>,
    pending_frames: Mutex<VecDeque<DecodedFrame>>,
    /// Decode-order frame count (every real picture submitted to the decoder).
    frame_count: Mutex<u32>,
    /// Display-order count; used as the `frame_index`/`poc` of output frames.
    display_count: u32,
    /// (left, top, right, bottom) crop region within the coded surface.
    display_area: Mutex<(i32, i32, i32, i32)>,
    initialized: Mutex<bool>,
    /// Parsed sequence header (available after the first SPS OBU).
    sps: Mutex<Option<Av1Sps>>,
    /// (width, height) of the last decoder configuration.
    prev_coded_size: Mutex<(u32, u32)>,
    pending_data: Vec<u8>,
    parsed_offset: usize,
    /// True if the input is an IVF container (packets start at offset 32).
    is_ivf: bool,
    /// Cached pinned host buffer for frame extraction.
    pinned_cache: Mutex<Option<(*mut std::ffi::c_void, usize)>>,
    /// Ring of cached pinned (page-locked) host buffers for the bitstream.
    ///
    /// `cuvidDecodePicture` may read the bitstream from this host pointer
    /// asynchronously (after the call returns), so each decode's data must
    /// survive until that decode completes. We round-robin through a ring of
    /// pinned buffers so an in-flight decode never sees its bitstream
    /// overwritten by a later one (NVIDIA's own parser stages each frame in
    /// its own buffer for exactly this reason). Buffers are allocated with
    /// zero-filled padding beyond the data: the AV1 decoder's bit reader can
    /// read in SIMD chunks that extend past `nBitstreamDataLen`, so the tail
    /// must be valid (zero) memory, not uninitialized.
    bitstream_ring: Mutex<(Vec<(*mut std::ffi::c_void, usize)>, u32)>,
    /// Per-decode slice-offset storage: `[0, bitstream_len, 0, ...]`.
    ///
    /// The NVDEC AV1 front end reads `nNumSlices + 1` entries from
    /// `pSliceDataOffsets` — the last one is the terminating offset (the C
    /// parser fills `[0, total_len]`) — even though cuviddec.h documents only
    /// `nNumSlices` entries. Pointing at smaller storage makes the driver
    /// read adjacent memory, which corrupts the decode (the driver reports
    /// `cuvidDecodeStatus_Error` and pixels come out wrong toward the end of
    /// the scan).
    slice_offsets: [u32; 64],
    /// If set (via `NVDEC_DUMP_PARAMS`), dump the exact [`CUVIDPICPARAMS`]
    /// submitted for each picture (DECODE order) to this path.
    dump_params_path: Option<std::path::PathBuf>,
    dump_params_count: u32,
    /// IVF timebase (rate_num, rate_den); converts packet pts to the 90 kHz
    /// clock NVDEC expects in `CUVIDPICPARAMS.Reserved[0]`.
    ivf_timebase: (u32, u32),
}

impl NvdecAv1Decoder {
    /// Create a new NVDEC AV1 decoder and begin decoding the input data.
    ///
    /// `data` is either an IVF container (magic `DKIF`, packets at offset 32)
    /// or a raw single AV1 frame.
    pub fn new(data: Vec<u8>) -> NvdecResult<Self> {
        init_nvdec()?;

        let is_ivf = data.len() >= IVF_HEADER_SIZE && &data[0..4] == b"DKIF";
        // IVF header (FFmpeg/canonical layout): fourcc @8, width @12, height
        // @14, time_base.den @16, time_base.num @20. There is no reserved
        // field before the fourcc.
        let ivf_timebase = if is_ivf && data.len() >= 24 {
            let rate_den = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
            let rate_num = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
            (rate_num, rate_den)
        } else {
            (0, 1)
        };

        let mut decoder = Self {
            parser: Av1Parser::new(),
            decoder: Mutex::new(std::ptr::null_mut()),
            info: Mutex::new(DecoderInfo {
                backend: "nvdec".to_string(),
                codec: VideoCodec::DecodeAv1,
                coded_size: Extent2D {
                    width: 0,
                    height: 0,
                },
                display_size: Extent2D {
                    width: 0,
                    height: 0,
                },
                chroma_subsampling: ChromaSubsampling::_420,
                luma_bit_depth: ComponentBitDepth::Bit8,
                chroma_bit_depth: ComponentBitDepth::Bit8,
                profile_idc: None,
                dpb_slots: NUM_SURFACES,
            }),
            pending_frames: Mutex::new(VecDeque::new()),
            frame_count: Mutex::new(0),
            display_count: 0,
            display_area: Mutex::new((0, 0, 0, 0)),
            initialized: Mutex::new(false),
            sps: Mutex::new(None),
            prev_coded_size: Mutex::new((0, 0)),
            pending_data: data,
            parsed_offset: if is_ivf { IVF_HEADER_SIZE } else { 0 },
            is_ivf,
            ivf_timebase,
            pinned_cache: Mutex::new(None),
            bitstream_ring: Mutex::new((Vec::new(), 0)),
            slice_offsets: [0; 64],
            dump_params_path: std::env::var("NVDEC_DUMP_PARAMS")
                .ok()
                .map(std::path::PathBuf::from),
            dump_params_count: 0,
        };

        decoder.init_parser_format()?;
        decoder.parser.set_dpb_slots(NUM_SURFACES);
        decoder.parse_and_decode()?;

        let initialized = *decoder.initialized.lock().unwrap();
        if !initialized {
            return Err(NvdecError::DecoderCreationFailed(
                "Parser did not initialize decoder - no AV1 frame found".into(),
            ));
        }

        Ok(decoder)
    }

    /// Initialize the parser with the AV1 format (required before parsing).
    fn init_parser_format(&mut self) -> NvdecResult<()> {
        self.parser
            .init(&DetectedVideoFormat::new(VideoCodec::DecodeAv1))
            .map_err(|e| NvdecError::DecodeFailed(format!("parser init: {}", e)))
    }

    /// Parse pending data and decode any available frames.
    fn parse_and_decode(&mut self) -> NvdecResult<()> {
        if self.is_ivf {
            loop {
                // Need at least 12 bytes for the packet header (4 size + 8 pts).
                if self.parsed_offset + 12 > self.pending_data.len() {
                    break;
                }
                let size = u32::from_le_bytes(
                    self.pending_data[self.parsed_offset..self.parsed_offset + 4]
                        .try_into()
                        .unwrap(),
                ) as usize;
                if size == 0 || self.parsed_offset + 12 + size > self.pending_data.len() {
                    break;
                }
                let pts = u64::from_le_bytes(
                    self.pending_data[self.parsed_offset + 4..self.parsed_offset + 12]
                        .try_into()
                        .unwrap(),
                );
                let payload =
                    &self.pending_data[self.parsed_offset + 12..self.parsed_offset + 12 + size];
                // Parse the SPS (type 1 OBU) once, from whichever packet carries it.
                if self.sps.lock().unwrap().is_none() {
                    if let Some(sps_payload) = find_sps_obu(payload) {
                        match self.parser.parse_sequence_header_obu(&sps_payload) {
                            Ok(s) => {
                                eprintln!(
                                    "[AV1-DBG] SPS: profile={} maxw-1={} maxh-1={} ohb-1={} 128x128={} sub={}/{} mono={} highbit={} 12bit={} cdef={} restoration={} superres={}",
                                    s.profile, s.max_frame_width_minus_1, s.max_frame_height_minus_1,
                                    s.order_hint_bits_minus1, s.use_128x128_superblock,
                                    s.subsampling_x, s.subsampling_y, s.mono_chrome,
                                    s.high_bitdepth, s.twelve_bit, s.enable_cdef,
                                    s.enable_restoration, s.enable_superres
                                );
                                *self.sps.lock().unwrap() = Some(s);
                            }
                            Err(e) => {
                                return Err(NvdecError::DecodeFailed(format!(
                                    "parse_sequence_header_obu: {}",
                                    e
                                )));
                            }
                        }
                    }
                }
                for obu in extract_frame_obus(payload) {
                    self.process_frame(&obu.payload, pts)?;
                }
                self.parsed_offset += 12 + size;
            }
        } else {
            // Raw single-frame: process the whole buffer once, then mark consumed.
            if self.parsed_offset == 0 && !self.pending_data.is_empty() {
                let data = self.pending_data.clone();
                self.process_frame(&data, 0)?;
                self.parsed_offset = self.pending_data.len();
            }
        }
        Ok(())
    }

    /// Stage `src` into the cached pinned (page-locked) host buffer and return
    /// a pointer to it.
    ///
    /// The buffer is grown on demand and kept alive across frames; it is only
    /// reallocated when a larger frame arrives (by which point the previous
    /// decode has completed, so the old buffer is not in use).
    fn bitstream_buffer(&mut self, src: &[u8]) -> NvdecResult<*const u8> {
        const RING_SIZE: usize = 8;
        let size = src.len();
        let need = size + BITSTREAM_PADDING;
        let mut ring = self.bitstream_ring.lock().unwrap();
        if ring.0.len() < RING_SIZE {
            ring.0.push((std::ptr::null_mut(), 0));
        }
        let slot = (ring.1 % RING_SIZE as u32) as usize;
        ring.1 += 1;
        let (p, cap) = &mut ring.0[slot];
        if *cap < need {
            if !p.is_null() {
                let _ = unsafe { cu_mem_free_host(*p) };
            }
            let np = cu_mem_host_alloc(need)?;
            *p = np;
            *cap = need;
        }
        let p = (*p).cast::<u8>();
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), p, size);
            std::ptr::write_bytes(p.add(size), 0x00, BITSTREAM_PADDING);
        }
        Ok(p as *const u8)
    }

    /// Parse and decode one AV1 frame OBU payload. `packet_pts` is the IVF
    /// packet timestamp (in IVF timebase ticks) carrying this frame.
    fn process_frame(&mut self, payload: &[u8], packet_pts: u64) -> NvdecResult<()> {
        let sps = match self.sps.lock().unwrap().clone() {
            Some(s) => s,
            // No SPS yet (e.g. first packet had no type-1 OBU); skip this OBU.
            None => return Ok(()),
        };

        let fh = self
            .parser
            .parse_frame_header(payload, &sps)
            .map_err(|e| NvdecError::DecodeFailed(format!("parse_frame_header: {}", e)))?;

        // show_existing_frame: no decode, no DPB change — re-display an
        // already-decoded surface (always displayed).
        if fh.show_existing_frame {
            let surface = self
                .parser
                .dpb()
                .slot_of_frame_buffer(fh.frame_to_show_map_idx as usize);
            if surface >= 0 {
                if let Some(frame) = self.extract_frame(surface) {
                    self.pending_frames.lock().unwrap().push_back(frame);
                }
                self.display_count += 1;
            }
            return Ok(());
        }

        // Real frame: create the decoder on the first frame; reconfigure on a
        // coded-size change.
        let initialized = *self.initialized.lock().unwrap();
        if !initialized {
            self.create_decoder(&fh, &sps)?;
        } else {
            let (prev_w, prev_h) = *self.prev_coded_size.lock().unwrap();
            if fh.frame_width != prev_w || fh.frame_height != prev_h {
                self.recreate_decoder(&fh, &sps)?;
            }
        }

        // The bitstream passed to cuvid is the tile data only (Frame OBU
        // payload minus the frame header). Stage it into the cached pinned
        // (page-locked) host buffer: `cuvidDecodePicture` CPU-memcpys the
        // bitstream from this host pointer (a device pointer segfaults).
        let hdr = (fh.frame_header_size as usize).min(payload.len());
        let tile_len = payload.len() - hdr;
        let tile_ptr = self.bitstream_buffer(&payload[hdr..])?;

        // Convert the packet pts to the 90 kHz clock (same result as the
        // NVIDIA cuvid-parser baseline: raw ticks scaled by the IVF timebase
        // num/den onto the 90 kHz clock).
        let (rate_num, rate_den) = self.ivf_timebase;
        let ts_90k = if rate_den > 0 {
            packet_pts.saturating_mul(90_000u64 * rate_num as u64) / rate_den as u64
        } else {
            0
        };

        // Build the picparams (allocates the output slot; references from the
        // pre-decode common DPB state).
        // The AV1 front end reads nNumSlices+1 slice offsets (terminator
        // included) — fill [0, bitstream_len] before every decode.
        self.slice_offsets[0] = 0;
        self.slice_offsets[1] = tile_len as u32;
        let params = build_cuvid_av1_picparams(
            &fh,
            &sps,
            self.parser.dpb_mut(),
            tile_ptr,
            tile_len as u32,
            ts_90k,
            self.slice_offsets.as_ptr().cast::<c_uint>(),
        );

        if let Some(dump_path) = &self.dump_params_path {
            dump_cuvid_av1_picparams(dump_path, self.dump_params_count, &params);
            self.dump_params_count += 1;
        }

        let decoder_handle = {
            let d = self.decoder.lock().unwrap();
            if d.is_null() {
                return Err(NvdecError::DecoderCreationFailed(
                    "decoder not created".into(),
                ));
            }
            *d
        };
        let funcs = get_funcs()?;
        let _ = cu_ctx_set_current();

        let procparams = crate::ffi::default_procparams();
        let result = unsafe { (funcs.decode_picture)(decoder_handle, &params, &procparams) };
        if result != CUDA_SUCCESS {
            return Err(NvdecError::DecodeFailed(format!(
                "cuvidDecodePicture failed: {}",
                result
            )));
        }

        cu_ctx_synchronize()?;

        // Commit this frame's refresh to the common DPB (now that the decode
        // is submitted).
        self.parser.dpb_mut().commit_decoded(
            params.CurrPicIdx as u32,
            &fh,
            sps.order_hint_bits_minus1 as u32,
        );

        if fh.show_frame {
            if let Some(frame) = self.extract_frame(params.CurrPicIdx) {
                self.pending_frames.lock().unwrap().push_back(frame);
            }
            self.display_count += 1;
        }
        // else: decoded for references only, nothing displayed.

        {
            let mut count = self.frame_count.lock().unwrap();
            *count += 1;
        }

        Ok(())
    }

    /// Create the NVDEC decoder from the first frame's parameters.
    fn create_decoder(&mut self, first: &Av1FrameHeader, sps: &Av1Sps) -> NvdecResult<()> {
        let w = first.frame_width;
        let h = first.frame_height;
        let bit_depth: u8 = if sps.twelve_bit {
            12
        } else if sps.high_bitdepth {
            10
        } else {
            8
        };

        // NVDEC extraction is 4:2:0 only. Reject other subsamplings early
        // with a clear message instead of silently mis-decoding (a 4:2:2
        // stream decoded as 4:2:0 produces wrong pixels and plane sizes).
        if sps.mono_chrome || !(sps.subsampling_x == 1 && sps.subsampling_y == 1) {
            let (cf, name) = if sps.subsampling_x == 0 && sps.subsampling_y == 0 {
                (cudaVideoChromaFormat::cudaVideoChromaFormat_444, "4:4:4")
            } else if sps.subsampling_y == 0 {
                (cudaVideoChromaFormat::cudaVideoChromaFormat_422, "4:2:2")
            } else {
                (cudaVideoChromaFormat::cudaVideoChromaFormat_420, "4:2:0")
            };
            let hw_supported = query_decoder_caps(
                cudaVideoCodec::cudaVideoCodec_AV1,
                cf,
                bit_depth.saturating_sub(8) as u32,
            )
            .map(|c| c.bIsSupported != 0)
            .unwrap_or(false);
            return Err(NvdecError::DecoderCreationFailed(format!(
                "AV1 {} {}-bit not supported by this NVDEC backend (4:2:0 only){}",
                name,
                bit_depth,
                if hw_supported {
                    " - hardware reports support but no 4:2:2/4:4:4 output path is implemented"
                } else {
                    " - device does not support this chroma format for AV1"
                }
            )));
        }

        let output_format = if bit_depth > 8 {
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_P016
        } else {
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12
        };

        let create_info = CUVIDDECODECREATEINFO {
            ulWidth: w as _,
            ulHeight: h as _,
            ulNumDecodeSurfaces: NUM_SURFACES as _,
            CodecType: cudaVideoCodec::cudaVideoCodec_AV1,
            ChromaFormat: cudaVideoChromaFormat::cudaVideoChromaFormat_420,
            ulCreationFlags: 0,
            bitDepthMinus8: (bit_depth.saturating_sub(8)) as _,
            ulIntraDecodeOnly: 0,
            ulMaxWidth: w as _,
            ulMaxHeight: h as _,
            Reserved1: 0,
            display_area: CUVIDRECT {
                left: 0,
                top: 0,
                right: w as _,
                bottom: h as _,
            },
            OutputFormat: output_format,
            DeinterlaceMode: cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Weave,
            ulTargetWidth: w as _,
            ulTargetHeight: h as _,
            ulNumOutputSurfaces: NUM_SURFACES as _,
            vidLock: std::ptr::null_mut(),
            target_rect: CUVIDRECT {
                left: 0,
                top: 0,
                right: w as _,
                bottom: h as _,
            },
            enableHistogram: 0,
            Reserved2: [0; 4],
        };

        let funcs = get_funcs()?;
        let _ = cu_ctx_set_current();

        let mut ph_decoder: CUvideodecoder = std::ptr::null_mut();
        let result = unsafe { (funcs.create_decoder)(&mut ph_decoder, &create_info) };
        if result != CUDA_SUCCESS || ph_decoder.is_null() {
            return Err(NvdecError::DecoderCreationFailed(format!(
                "cuvidCreateDecoder failed with error {}",
                result
            )));
        }
        {
            let mut decoder = self.decoder.lock().unwrap();
            *decoder = ph_decoder;
        }

        let mut info = self.info.lock().unwrap();
        *info = DecoderInfo {
            backend: "nvdec".to_string(),
            codec: VideoCodec::DecodeAv1,
            coded_size: Extent2D {
                width: w,
                height: h,
            },
            display_size: Extent2D {
                width: w,
                height: h,
            },
            chroma_subsampling: ChromaSubsampling::_420, // only supported variant
            luma_bit_depth: bit_depth_component(bit_depth),
            chroma_bit_depth: bit_depth_component(bit_depth),
            profile_idc: Some(sps.profile as u32),
            dpb_slots: NUM_SURFACES,
        };

        {
            let mut display_area = self.display_area.lock().unwrap();
            *display_area = (0, 0, w as i32, h as i32);
        }
        {
            let mut prev = self.prev_coded_size.lock().unwrap();
            *prev = (w, h);
        }
        {
            let mut initialized = self.initialized.lock().unwrap();
            *initialized = true;
        }

        Ok(())
    }

    /// Handle a coded-size change on a later frame.
    fn recreate_decoder(&mut self, fd: &Av1FrameHeader, sps: &Av1Sps) -> NvdecResult<()> {
        eprintln!(
            "[recreate] AV1 decoder reconfigured {}x{}",
            fd.frame_width, fd.frame_height
        );
        let funcs = get_funcs()?;
        let _ = cu_ctx_set_current();
        let decoder_handle = {
            let d = self.decoder.lock().unwrap();
            *d
        };

        let mut reconfigured = false;
        if !decoder_handle.is_null() {
            if let Some(reconfigure) = funcs.reconfigure_decoder {
                let w = fd.frame_width;
                let h = fd.frame_height;
                let reconfig = crate::ffi::CUVIDRECONFIGUREDECODERINFO {
                    ulWidth: w,
                    ulHeight: h,
                    ulTargetWidth: w,
                    ulTargetHeight: h,
                    ulNumDecodeSurfaces: NUM_SURFACES,
                    reserved1: [0; 12],
                    display_area: CUVIDRECT {
                        left: 0,
                        top: 0,
                        right: w as _,
                        bottom: h as _,
                    },
                    target_rect: CUVIDRECT {
                        left: 0,
                        top: 0,
                        right: w as _,
                        bottom: h as _,
                    },
                    reserved2: [0; 11],
                };
                let res = unsafe { reconfigure(decoder_handle, &reconfig) };
                if res == CUDA_SUCCESS {
                    reconfigured = true;
                } else {
                    eprintln!(
                        "[recreate] cuvidReconfigureDecoder failed ({}), falling back to destroy+recreate",
                        res
                    );
                    let _ = unsafe { (funcs.destroy_decoder)(decoder_handle) };
                }
            } else {
                let _ = unsafe { (funcs.destroy_decoder)(decoder_handle) };
            }
        }

        if !reconfigured {
            {
                let mut decoder = self.decoder.lock().unwrap();
                *decoder = std::ptr::null_mut();
            }
            self.parser.dpb_mut().reset();
            {
                let mut pending = self.pending_frames.lock().unwrap();
                pending.clear();
            }
            {
                let mut initialized = self.initialized.lock().unwrap();
                *initialized = false;
            }
            {
                let mut prev = self.prev_coded_size.lock().unwrap();
                *prev = (0, 0);
            }
            return self.create_decoder(fd, sps);
        }

        // Reconfigured in place: update info + display area + prev size.
        let w = fd.frame_width;
        let h = fd.frame_height;
        {
            let mut info = self.info.lock().unwrap();
            info.coded_size = Extent2D {
                width: w,
                height: h,
            };
            info.display_size = Extent2D {
                width: w,
                height: h,
            };
        }
        {
            let mut display_area = self.display_area.lock().unwrap();
            *display_area = (0, 0, w as i32, h as i32);
        }
        {
            let mut prev = self.prev_coded_size.lock().unwrap();
            *prev = (w, h);
        }
        Ok(())
    }

    /// Extract a decoded frame from the NVDEC decoder by surface index.
    fn extract_frame(&self, pic_index: i32) -> Option<DecodedFrame> {
        if pic_index < 0 {
            return None;
        }
        let decoder = {
            let d = self.decoder.lock().unwrap();
            if d.is_null() {
                return None;
            }
            *d
        };
        let info = {
            let i = self.info.lock().unwrap();
            if i.display_size.width == 0 || i.display_size.height == 0 {
                return None;
            }
            i.clone()
        };
        let display_width = info.display_size.width as usize;
        let display_height = info.display_size.height as usize;
        let funcs = match get_funcs() {
            Ok(f) => f,
            Err(_) => return None,
        };
        let _ = cu_ctx_set_current();

        let mut dev_ptr: CUdeviceptr = 0;
        let mut pitch: u32 = 0;
        let proc_params = CUVIDPROCPARAMS {
            progressive_frame: 1,
            second_field: 0,
            top_field_first: 0,
            unpaired_field: 0,
            reserved_flags: 0,
            reserved_zero: 0,
            raw_input_dptr: 0,
            raw_input_pitch: 0,
            raw_input_format: 0,
            raw_output_dptr: 0,
            raw_output_pitch: 0,
            Reserved1: 0,
            output_stream: std::ptr::null_mut(),
            Reserved: [0; 46],
            histogram_dptr: std::ptr::null_mut(),
            Reserved2: [std::ptr::null_mut()],
        };
        let map_result = unsafe {
            (funcs.map_video_frame64)(decoder, pic_index, &mut dev_ptr, &mut pitch, &proc_params)
        };
        if map_result != CUDA_SUCCESS {
            eprintln!("[NVDEC] cuvidMapVideoFrame64 failed: {}", map_result);
            return None;
        }
        let display_area = {
            let d = self.display_area.lock().unwrap();
            *d
        };
        let (crop_left, crop_top, _, _) = display_area;

        // P016 (10/12-bit) surfaces use 2-byte little-endian samples.
        let ss = if info.luma_bit_depth == ComponentBitDepth::Bit8 {
            1
        } else {
            2
        };

        let y_size = display_width * display_height * ss;
        let interleaved_uv_size = display_width * (display_height / 2) * ss;
        let total = y_size + interleaved_uv_size;
        let pinned_base = {
            let mut cache = self.pinned_cache.lock().unwrap();
            match &*cache {
                Some((p, sz)) if *sz >= total => *p,
                _ => {
                    if let Some((p, _)) = cache.take() {
                        let _ = unsafe { cu_mem_free_host(p) };
                    }
                    match cu_mem_host_alloc(total) {
                        Ok(p) => {
                            *cache = Some((p, total));
                            p
                        }
                        Err(_) => {
                            let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
                            return None;
                        }
                    }
                }
            }
        };
        let pinned_y = pinned_base;
        let pinned_uv = unsafe { (pinned_base as *mut u8).add(y_size) as *mut std::ffi::c_void };

        let copy_y = CUDA_MEMCPY2D {
            srcXInBytes: crop_left as u64 * ss as u64,
            srcY: crop_top as u64,
            srcMemoryType: CU_MEMORYTYPE_DEVICE,
            _reserved0: 0,
            srcHost: std::ptr::null(),
            srcDevice: dev_ptr,
            srcArray: 0,
            srcPitch: pitch as u64,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: CU_MEMORYTYPE_HOST,
            _reserved1: 0,
            dstHost: pinned_y,
            dstDevice: 0,
            dstArray: 0,
            dstPitch: display_width as u64 * ss as u64,
            WidthInBytes: display_width as u64 * ss as u64,
            Height: display_height as u64,
        };
        match unsafe { cu_memcpy_2d(&copy_y) } {
            Ok(CUDA_SUCCESS) => {}
            other => {
                eprintln!("[NVDEC] cuMemcpy2D(Y) failed: {:?}", other);
                let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
                return None;
            }
        }

        let coded_height = info.coded_size.height as u64;
        let copy_uv = CUDA_MEMCPY2D {
            srcXInBytes: crop_left as u64 * ss as u64,
            srcY: coded_height + (crop_top as u64) / 2,
            srcMemoryType: CU_MEMORYTYPE_DEVICE,
            _reserved0: 0,
            srcHost: std::ptr::null(),
            srcDevice: dev_ptr,
            srcArray: 0,
            srcPitch: pitch as u64,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: CU_MEMORYTYPE_HOST,
            _reserved1: 0,
            dstHost: pinned_uv,
            dstDevice: 0,
            dstArray: 0,
            dstPitch: display_width as u64 * ss as u64,
            WidthInBytes: display_width as u64 * ss as u64,
            Height: (display_height / 2) as u64,
        };
        match unsafe { cu_memcpy_2d(&copy_uv) } {
            Ok(CUDA_SUCCESS) => {}
            other => {
                eprintln!("[NVDEC] cuMemcpy2D(UV) failed: {:?}", other);
                let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
                return None;
            }
        }

        let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };

        let mut y_plane = vec![0u8; y_size];
        let mut interleaved_uv = vec![0u8; interleaved_uv_size];
        unsafe {
            std::ptr::copy_nonoverlapping(pinned_y as *const u8, y_plane.as_mut_ptr(), y_size);
            std::ptr::copy_nonoverlapping(
                pinned_uv as *const u8,
                interleaved_uv.as_mut_ptr(),
                interleaved_uv_size,
            );
        }

        // P016 surfaces store samples left-aligned with zeroed LSBs (10-bit:
        // << 6, 12-bit: << 4). Shift back to native bit depth.
        if ss == 2 {
            let shift = 16u32 - info.luma_bit_depth.bit_depth();
            for chunk in y_plane.chunks_exact_mut(2) {
                let s = u16::from_le_bytes([chunk[0], chunk[1]]) >> shift;
                chunk.copy_from_slice(&s.to_le_bytes());
            }
            for chunk in interleaved_uv.chunks_exact_mut(2) {
                let s = u16::from_le_bytes([chunk[0], chunk[1]]) >> shift;
                chunk.copy_from_slice(&s.to_le_bytes());
            }
        }

        // De-interleave the semi-planar UV to planar U and V (u8 or u16 LE).
        let uv_size = (display_width / 2) * (display_height / 2) * ss;
        let mut u_plane = vec![0u8; uv_size];
        let mut v_plane = vec![0u8; uv_size];
        if ss == 1 {
            for y in 0..(display_height / 2) {
                for x in 0..(display_width / 2) {
                    let src_idx = y * display_width + x * 2;
                    let dst_idx = y * (display_width / 2) + x;
                    u_plane[dst_idx] = interleaved_uv[src_idx];
                    v_plane[dst_idx] = interleaved_uv[src_idx + 1];
                }
            }
        } else {
            for y in 0..(display_height / 2) {
                for x in 0..(display_width / 2) {
                    let src_idx = y * display_width * 2 + x * 4;
                    let dst_idx = y * (display_width / 2) * 2 + x * 2;
                    u_plane[dst_idx] = interleaved_uv[src_idx];
                    u_plane[dst_idx + 1] = interleaved_uv[src_idx + 1];
                    v_plane[dst_idx] = interleaved_uv[src_idx + 2];
                    v_plane[dst_idx + 1] = interleaved_uv[src_idx + 3];
                }
            }
        }

        let mut buffer = Vec::with_capacity(y_size + uv_size * 2);
        buffer.extend_from_slice(&y_plane);
        buffer.extend_from_slice(&u_plane);
        buffer.extend_from_slice(&v_plane);

        let y_ptr = buffer.as_ptr();
        let u_ptr = unsafe { buffer.as_ptr().add(y_size) };
        let v_ptr = unsafe { buffer.as_ptr().add(y_size + uv_size) };

        // Tightly packed: pitch (bytes) == width (samples) * ss.
        let y_pitch = display_width * ss;
        let uv_pitch = (display_width / 2) * ss;
        let pixel_data = Some(PixelData {
            format: match info.luma_bit_depth {
                ComponentBitDepth::Bit8 => "I420".to_string(),
                ComponentBitDepth::Bit12 => "P012LE".to_string(),
                _ => "P010LE".to_string(),
            },
            y: PixelPlane {
                data: y_ptr,
                pitch: y_pitch,
                width: display_width,
                height: display_height,
            },
            u: PixelPlane {
                data: u_ptr,
                pitch: uv_pitch,
                width: display_width / 2,
                height: display_height / 2,
            },
            v: Some(PixelPlane {
                data: v_ptr,
                pitch: uv_pitch,
                width: display_width / 2,
                height: display_height / 2,
            }),
            buffer,
        });

        let frame_index = self.display_count;
        let poc_value = self.display_count as i32;

        Some(DecodedFrame {
            frame_index,
            timestamp: 0,
            width: info.display_size.width,
            height: info.display_size.height,
            skipped: false,
            pts_valid: false,
            poc: poc_value,
            field_flags: FieldFlags {
                progressive_frame: true,
                field_pic: false,
                bottom_field: false,
                second_field: false,
                top_field_first: true,
                unpaired_field: false,
                sync_first_ready: false,
                sync_to_first_field: false,
                repeat_first_field: 0,
                ref_pic: false,
                apply_film_grain: false,
            },
            sync_info: vacc_core::frame::FrameSyncInfo::default(),
            pixel_data,
        })
    }
}

impl Decoder for NvdecAv1Decoder {
    type Error = NvdecError;

    fn new(data: Vec<u8>) -> NvdecResult<Self>
    where
        Self: Sized,
    {
        Self::new(data)
    }

    fn new_with_format(data: Vec<u8>, codec: VideoCodec, _format: &VideoFormat) -> NvdecResult<Self>
    where
        Self: Sized,
    {
        if codec != VideoCodec::DecodeAv1 {
            return Err(NvdecError::UnsupportedCodec(codec));
        }
        Self::new(data)
    }

    fn info(&self) -> DecoderInfo {
        self.info.lock().unwrap().clone()
    }

    fn submit(&mut self, data: &[u8]) -> NvdecResult<()> {
        self.pending_data.extend_from_slice(data);
        Ok(())
    }

    fn decode(&mut self) -> NvdecResult<Option<DecodedFrame>> {
        self.parse_and_decode()?;
        Ok(self.pending_frames.lock().unwrap().pop_front())
    }

    fn flush(&mut self) -> NvdecResult<Vec<DecodedFrame>> {
        self.parse_and_decode()?;
        let mut pending = self.pending_frames.lock().unwrap();
        Ok(pending.drain(..).collect())
    }

    fn reset(&mut self) -> NvdecResult<()> {
        let _ = cu_ctx_set_current();
        let funcs = get_funcs()?;
        let decoder_handle = {
            let d = self.decoder.lock().unwrap();
            *d
        };
        if !decoder_handle.is_null() {
            let _ = unsafe { (funcs.destroy_decoder)(decoder_handle) };
            let mut d = self.decoder.lock().unwrap();
            *d = std::ptr::null_mut();
        }
        // parser.reset() also resets the common AV1 DPB.
        self.parser.reset();
        *self.sps.lock().unwrap() = None;
        {
            let mut pending = self.pending_frames.lock().unwrap();
            pending.clear();
        }
        {
            let mut count = self.frame_count.lock().unwrap();
            *count = 0;
        }
        self.display_count = 0;
        self.parsed_offset = if self.is_ivf { IVF_HEADER_SIZE } else { 0 };
        {
            let mut initialized = self.initialized.lock().unwrap();
            *initialized = false;
        }
        {
            let mut prev = self.prev_coded_size.lock().unwrap();
            *prev = (0, 0);
        }
        self.init_parser_format()?;
        self.parse_and_decode()?;
        let initialized = *self.initialized.lock().unwrap();
        if !initialized {
            return Err(NvdecError::DecoderCreationFailed(
                "Parser did not reinitialize decoder after reset".into(),
            ));
        }
        Ok(())
    }
}

impl Drop for NvdecAv1Decoder {
    fn drop(&mut self) {
        let _ = cu_ctx_set_current();
        let decoder_handle = {
            let d = self.decoder.lock().unwrap();
            *d
        };
        if !decoder_handle.is_null() {
            if let Ok(funcs) = get_funcs() {
                let _ = unsafe { (funcs.destroy_decoder)(decoder_handle) };
            }
        }
        // Free pinned host buffers to avoid leaking page-locked memory.
        if let Ok(mut cache) = self.pinned_cache.lock() {
            if let Some((ptr, _)) = cache.take() {
                let _ = unsafe { crate::device::cu_mem_free_host(ptr) };
            }
        }
        if let Ok(mut ring) = self.bitstream_ring.lock() {
            for (ptr, _) in ring.0.drain(..) {
                if !ptr.is_null() {
                    let _ = unsafe { crate::device::cu_mem_free_host(ptr) };
                }
            }
        }
    }
}

/// Map a bit depth to a [`ComponentBitDepth`].
fn bit_depth_component(bit_depth: u8) -> ComponentBitDepth {
    match bit_depth {
        8 => ComponentBitDepth::Bit8,
        10 => ComponentBitDepth::Bit10,
        12 => ComponentBitDepth::Bit12,
        _ => ComponentBitDepth::Bit8,
    }
}

/// Dump the key fields of a [`CUVIDPICPARAMS`] (AV1) for debugging.
fn dump_cuvid_av1_picparams(path: &std::path::Path, pic_num: u32, p: &CUVIDPICPARAMS) {
    use std::io::Write;
    let av1 = unsafe { &p.CodecSpecific.av1 };
    let mut f = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(f) => f,
        Err(_) => return,
    };
    let _ = writeln!(f, "=== DECODE {} ===", pic_num);
    let _ = writeln!(f, "PicWidthInMbs = {}", p.PicWidthInMbs);
    let _ = writeln!(f, "FrameHeightInMbs = {}", p.FrameHeightInMbs);
    let _ = writeln!(f, "CurrPicIdx = {}", p.CurrPicIdx);
    let _ = writeln!(f, "nBitstreamDataLen = {}", p.nBitstreamDataLen);
    let _ = writeln!(f, "nNumSlices = {}", p.nNumSlices);
    let _ = writeln!(f, "field_pic_flag = {}", p.field_pic_flag);
    let _ = writeln!(f, "bottom_field_flag = {}", p.bottom_field_flag);
    let _ = writeln!(f, "second_field = {}", p.second_field);
    let _ = writeln!(
        f,
        "Reserved = [{}]",
        p.Reserved
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    // First/last 32 bytes of the actual bitstream passed to cuvidDecodePicture
    // (it lives in the pinned host buffer, so read it directly).
    let total = p.nBitstreamDataLen as usize;
    let bs = unsafe { std::slice::from_raw_parts(p.pBitstreamData, total) };
    let first_len = total.min(32);
    let _ = writeln!(
        f,
        "BITSTREAM[0..{}) = {}",
        first_len,
        bs[..first_len]
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    );
    let last_start = total.saturating_sub(32);
    let _ = writeln!(
        f,
        "BITSTREAM[{}..{}] = {}",
        last_start,
        total,
        bs[last_start..]
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    );
    let _ = writeln!(f, "ref_pic_flag = {}", p.ref_pic_flag);
    let _ = writeln!(f, "intra_pic_flag = {}", p.intra_pic_flag);
    let _ = writeln!(f, "width = {}", av1.width);
    let _ = writeln!(f, "height = {}", av1.height);
    let _ = writeln!(f, "frame_offset = {}", av1.frame_offset);
    let _ = writeln!(f, "decodePicIdx = {}", av1.decodePicIdx);
    let _ = writeln!(
        f,
        "profile = {} use_128x128 = {} subsampling = {}/{} mono = {} bit_depth_minus8 = {}",
        av1.profile(),
        av1.use_128x128_superblock(),
        av1.subsampling_x(),
        av1.subsampling_y(),
        av1.mono_chrome(),
        av1.bit_depth_minus8()
    );
    let _ = writeln!(
        f,
        "enable_order_hint = {} order_hint_bits_minus1 = {} enable_cdef = {} enable_restoration = {} enable_superres = {} enable_fgs = {}",
        av1.enable_order_hint(), av1.order_hint_bits_minus1(), av1.enable_cdef(), av1.enable_restoration(), av1.enable_superres(), av1.enable_fgs()
    );
    let _ = writeln!(
        f,
        "frame_type = {} show_frame = {} disable_cdf_update = {} allow_sct = {} force_integer_mv = {} coded_denom = {}",
        av1.frame_type(), av1.show_frame(), av1.disable_cdf_update(), av1.allow_screen_content_tools(), av1.force_integer_mv(), av1.coded_denom()
    );
    let _ = writeln!(
        f,
        "interp_filter = {} switchable_motion_mode = {} use_ref_frame_mvs = {} tx_mode = {} reference_mode = {} reduced_tx_set = {} skip_mode = {}",
        av1.interp_filter(), av1.switchable_motion_mode(), av1.use_ref_frame_mvs(), av1.tx_mode(), av1.reference_mode(), av1.reduced_tx_set(), av1.skip_mode()
    );
    let _ = writeln!(
        f,
        "delta_q_present = {} delta_q_res = {} using_qmatrix = {} coded_lossless = {} use_superres = {}",
        av1.delta_q_present(), av1.delta_q_res(), av1.using_qmatrix(), av1.coded_lossless(), av1.use_superres()
    );
    let _ = writeln!(
        f,
        "num_tile_cols = {} num_tile_rows = {} context_update_tile_id = {}",
        av1.num_tile_cols(),
        av1.num_tile_rows(),
        av1.context_update_tile_id()
    );
    let _ = writeln!(
        f,
        "tile_widths = [{}]",
        av1.tile_widths
            .iter()
            .take(16)
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let _ = writeln!(
        f,
        "tile_heights = [{}]",
        av1.tile_heights
            .iter()
            .take(16)
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let _ = writeln!(
        f,
        "cdef_damping_minus_3 = {} cdef_bits = {}",
        av1.cdef_damping_minus_3(),
        av1.cdef_bits()
    );
    let _ = writeln!(
        f,
        "base_qindex = {} qp_y_dc = {} qp_u_dc = {} qp_v_dc = {} qp_u_ac = {} qp_v_ac = {}",
        av1.base_qindex,
        av1.qp_y_dc_delta_q,
        av1.qp_u_dc_delta_q,
        av1.qp_v_dc_delta_q,
        av1.qp_u_ac_delta_q,
        av1.qp_v_ac_delta_q
    );
    let _ = writeln!(
        f,
        "segmentation: enabled={} update_map={} update_data={} temporal_update={}",
        av1.segmentation_enabled(),
        av1.segmentation_update_map(),
        av1.segmentation_update_data(),
        av1.segmentation_temporal_update()
    );
    let _ = writeln!(
        f,
        "loop_filter: level=[{},{}] level_u={} level_v={} sharpness={} delta_enabled={} delta_update={} delta_lf_present={} delta_lf_res={} delta_lf_multi={}",
        av1.loop_filter_level[0], av1.loop_filter_level[1], av1.loop_filter_level_u, av1.loop_filter_level_v, av1.loop_filter_sharpness,
        av1.loop_filter_delta_enabled(), av1.loop_filter_delta_update(), av1.delta_lf_present(), av1.delta_lf_res(), av1.delta_lf_multi()
    );
    let _ = writeln!(
        f,
        "loop_filter_ref_deltas = [{}]",
        av1.loop_filter_ref_deltas
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let _ = writeln!(f, "primary_ref_frame = {}", av1.primary_ref_frame);
    let _ = writeln!(
        f,
        "ref_frame_map = [{}]",
        av1.ref_frame_map
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let _ = writeln!(
        f,
        "ref_frame = [{}]",
        (0..7)
            .map(|r| format!(
                "({}x{} idx={})",
                av1.ref_frame[r].width, av1.ref_frame[r].height, av1.ref_frame[r].index
            ))
            .collect::<Vec<_>>()
            .join(" ")
    );
    // Full raw byte dump of the CUVIDAV1PICPARAMS for byte-exact diffing.
    let raw = unsafe {
        std::slice::from_raw_parts(
            av1 as *const crate::ffi::CUVIDAV1PICPARAMS as *const u8,
            std::mem::size_of::<crate::ffi::CUVIDAV1PICPARAMS>(),
        )
    };
    let _ = writeln!(
        f,
        "RAW = {}",
        raw.iter().map(|b| format!("{:02x}", b)).collect::<String>()
    );
    // Full bitstream MD5 for content verification (matches the cuvid baseline tool).
    let bs_len = p.nBitstreamDataLen as usize;
    if bs_len > 0 && !p.pBitstreamData.is_null() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let full = unsafe { std::slice::from_raw_parts(p.pBitstreamData, bs_len) };
        let mut h = DefaultHasher::new();
        full.hash(&mut h);
        let _ = writeln!(f, "BITSTREAM_MD5_LEN = {} {}", bs_len, h.finish());
    }
}

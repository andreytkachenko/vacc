//! VP9 DPB state and `CUVIDPICPARAMS` construction for NVDEC.
//!
//! This module is the core of the future `NvdecVp9Decoder`. It mirrors the
//! surface-management behavior of NVIDIA's own cuvid parser, which was
//! reverse-engineered from a 300-frame parameter dump
//! (`/tmp/pixel_verify/vp9_cuvid_params.txt`) and verified to reproduce every
//! `CurrPicIdx` / reference-surface value the cuvid parser emits for the
//! original 1920x1080 bunny VP9 stream (golden fixture: frames 0-19 in
//! `tests/data/vp9_dp_golden_20f.ivf`).
//!
//! ## DPB model
//!
//! VP9 keeps up to 8 frame buffers (frame contexts). The 3 *active* reference
//! slots (LAST, GOLDEN, ALTREF) each point at a frame buffer via
//! `ref_frame_idx`. Each frame buffer maps to a decode surface. A surface is
//! *live* while any frame buffer points at it; a frame may only be decoded
//! into a non-live surface.
//!
//! The output surface for a frame is the **oldest** (by last-use frame index)
//! non-live surface. This reproduces the cuvid parser's wraparound exactly:
//! with 16 surfaces, frames 0-15 use surfaces 0-15, and frame 16 reuses
//! surface 1 (surface 0 is still live as ALTREF and must be skipped).
//!
//! ## Refresh timing
//!
//! A frame's `refresh_frame_flags` (which frame buffers the frame becomes) is
//! committed to the common DPB **after** the frame is decoded
//! (`Vp9Dpb::commit_frame`). Spec-correct; for pure inter stretches this is
//! equivalent to the cuvid parser's deferred timing, and for key frames it
//! only changes *which* free surface is picked (content identical).

use std::collections::VecDeque;
use std::os::raw::{c_int, c_uchar, c_uint};
use std::sync::Mutex;

use vacc_core::{
    codec::VideoCodec,
    decoder::{Decoder, DecoderInfo},
    format::{ChromaSubsampling, ComponentBitDepth, VideoFormat},
    frame::{DecodedFrame, FieldFlags, PixelData, PixelPlane},
    picture::Vp9FrameData,
    session::Extent2D,
};
use vacc_parser::vp9::Vp9Parser;
use vacc_parser::{DetectedVideoFormat, VideoParser};

use crate::{
    device::{
        cu_ctx_set_current, cu_ctx_synchronize, cu_mem_free_host, cu_mem_host_alloc, cu_memcpy_2d,
        get_funcs, init_nvdec, CUDA_MEMCPY2D, CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST,
    },
    error::{NvdecError, NvdecResult},
    ffi::{
        cudaVideoChromaFormat, cudaVideoCodec, cudaVideoDeinterlaceMode, cudaVideoSurfaceFormat,
        CUdeviceptr, CUvideodecoder, CUDA_SUCCESS, CUVIDDECODECREATEINFO, CUVIDPICPARAMS,
        CUVIDPROCPARAMS, CUVIDRECT, CUVIDVP9PICPARAMS,
    },
};

/// Common VP9 DPB manager shared by ALL backends (Vulkan / NVDEC / VAAPI).
/// For NVDEC, a DPB slot IS a cuvid surface index (`PicIdx`), so the common
/// slot indices can be used directly as `CurrPicIdx` / `*RefIdx` values.
pub use vacc_parser::vp9_dpb::Vp9Dpb as Vp9DpbState;

/// cuvid `PicIdx` for frame buffer `fb`: the DPB slot, or 255 if empty.
fn surface_of_frame_buffer(dpb: &Vp9DpbState, fb: usize) -> i32 {
    match dpb.slot_of_frame_buffer(fb) {
        s @ 0..=255 => s,
        _ => 255,
    }
}

/// Build a complete [`CUVIDPICPARAMS`] for one VP9 frame from parser output +
/// DPB state.
///
/// `fd` is the [`Vp9FrameData`] from [`vacc_parser::vp9::Vp9Parser`],
/// `dpb` is the running [`Vp9DpbState`], and `bitstream_ptr`/`bitstream_len`
/// point at the raw (superframe-expanded) frame bitstream. The pointer must
/// stay valid for the duration of the subsequent `cuvidDecodePicture` call.
///
/// The field mapping reproduces the cuvid parser's output exactly (verified
/// against the 300-frame dump). The DPB state (common [`Vp9DpbState`]) is read
/// for references and the output surface; the caller MUST call
/// `dpb.commit_frame(refresh_frame_flags, params.CurrPicIdx)` after decoding
/// so the frame buffers point at this frame for subsequent frames.
///
/// Callers should not invoke this for `show_existing_frame` commands (they
/// carry no bitstream data); handle those separately.
pub fn build_cuvid_vp9_picparams(
    fd: &Vp9FrameData,
    dpb: &mut Vp9DpbState,
    bitstream_ptr: *const u8,
    bitstream_len: u32,
    slice_offsets: *const c_uint,
) -> CUVIDPICPARAMS {
    let is_key = fd.frame_is_intra;

    // 1. Reference frame-buffer lookup. Inter frames use `ref_frame_idx`;
    //    key frames take no references.
    let ref_lookup = if is_key {
        [0u8, 1, 2]
    } else {
        [
            fd.ref_frame_idx[0],
            fd.ref_frame_idx[1],
            fd.ref_frame_idx[2],
        ]
    };

    let mut last_ref = 255i32;
    let mut golden_ref = 255i32;
    let mut alt_ref = 255i32;
    if !is_key {
        let slots = dpb.reference_slots(false, &ref_lookup);
        last_ref = slots[0].max(0);
        golden_ref = slots[1].max(0);
        alt_ref = slots[2].max(0);
    }

    // 2. Output surface (live set computed from the pre-decode fb state).
    let curr_pic_idx = dpb.choose_output_slot();

    let pi = &fd.picture_info;
    let cc = &fd.color_config;
    let lf = &fd.loop_filter;
    let sg = &fd.segmentation;

    // mcomp_filter_type: the parser carries the last-seen filter over to
    // key/intra-only frames (FFmpeg `h->filtermode` / cuvid convention).
    let mcomp = pi.interpolation_filter as u32;

    // 3. VP9-specific parameters.
    let mut vp9 = CUVIDVP9PICPARAMS::new();
    vp9.width = fd.frame_width;
    vp9.height = fd.frame_height;
    vp9.LastRefIdx = last_ref as c_uchar;
    vp9.GoldenRefIdx = golden_ref as c_uchar;
    vp9.AltRefIdx = alt_ref as c_uchar;
    // VP9 color space (0=unknown, 1=bt601, 2=bt709, ...). The parser persists
    // the key frame's color config across inter frames.
    vp9.colorSpace = cc.color_space as u32 as c_uchar;

    vp9.set_profile(pi.profile as u32);
    vp9.set_frame_context_idx(pi.frame_context_idx as u32);
    vp9.set_frame_type(pi.frame_type as u32);
    vp9.set_show_frame(pi.flags.show_frame as u32);
    vp9.set_error_resilient(pi.flags.error_resilient_mode as u32);
    vp9.set_frame_parallel_decoding(pi.flags.frame_parallel_decoding_mode as u32);
    vp9.set_sub_sampling_x(cc.subsampling_x as u32);
    vp9.set_sub_sampling_y(cc.subsampling_y as u32);
    vp9.set_intra_only(pi.flags.intra_only as u32);
    vp9.set_allow_high_precision_mv(pi.flags.allow_high_precision_mv as u32);
    // refresh_frame_context -> refreshEntropyProbs
    vp9.set_refresh_entropy_probs(pi.flags.refresh_frame_context as u32);
    vp9.reserved16Bits = 0;

    // refFrameSignBias[i] = (ref_frame_sign_bias_mask >> (i + 1)) & 1, [3] = 0.
    for i in 0..4 {
        vp9.refFrameSignBias[i] = if i < 3 {
            ((pi.ref_frame_sign_bias_mask >> (i + 1)) & 1) as c_uchar
        } else {
            0
        };
    }

    vp9.bitDepthMinus8Luma = cc.bit_depth.saturating_sub(8);
    vp9.bitDepthMinus8Chroma = cc.bit_depth.saturating_sub(8);
    vp9.loopFilterLevel = lf.loop_filter_level;
    vp9.loopFilterSharpness = lf.loop_filter_sharpness;
    vp9.modeRefLfEnabled = lf.flags.loop_filter_delta_enabled;
    vp9.log2_tile_columns = pi.tile_cols_log2;
    vp9.log2_tile_rows = pi.tile_rows_log2;

    vp9.set_segment_enabled(pi.flags.segmentation_enabled as u32);
    vp9.set_segment_map_update(sg.flags.segmentation_update_map as u32);
    vp9.set_segment_map_temporal_update(sg.flags.segmentation_temporal_update as u32);
    // segmentation_update_data -> segmentFeatureMode
    vp9.set_segment_feature_mode(sg.flags.segmentation_update_data as u32);

    // Segmentation data: the cuvid parser zero-initializes picparams and only
    // fills these when segmentation is enabled. vacc-parser defaults the
    // tree/pred probs to VP9_MAX_PROBABILITY (255), which does NOT match the
    // cuvid dump's 0 — so gate on `segmentation_enabled` (trust the dump).
    if pi.flags.segmentation_enabled != 0 {
        for seg in 0..8 {
            for f in 0..4 {
                vp9.segmentFeatureEnable[seg][f] = ((sg.feature_enabled[seg] >> f) & 1) as c_uchar;
                vp9.segmentFeatureData[seg][f] = sg.feature_data[seg][f] as i16;
            }
        }
        vp9.mb_segment_tree_probs = sg.segmentation_tree_probs;
        vp9.segment_pred_probs = sg.segmentation_pred_prob;
    }
    vp9.reservedSegment16Bits = [0; 2];

    vp9.qpYAc = pi.base_q_idx as c_int;
    vp9.qpYDc = pi.delta_q_y_dc as c_int;
    vp9.qpChDc = pi.delta_q_uv_dc as c_int;
    vp9.qpChAc = pi.delta_q_uv_ac as c_int;

    // activeRefIdx are FRAME BUFFER indices (not surfaces).
    vp9.activeRefIdx[0] = fd.ref_frame_idx[0] as c_uint;
    vp9.activeRefIdx[1] = fd.ref_frame_idx[1] as c_uint;
    vp9.activeRefIdx[2] = fd.ref_frame_idx[2] as c_uint;
    vp9.resetFrameContext = pi.flags.reset_frame_context as c_uint;
    vp9.mcomp_filter_type = mcomp;

    // Signed i8 deltas, sign-extended into the unsigned fields.
    for i in 0..4 {
        vp9.mbRefLfDelta[i] = lf.loop_filter_ref_deltas[i] as i32 as u32;
    }
    for i in 0..2 {
        vp9.mbModeLfDelta[i] = lf.loop_filter_mode_deltas[i] as i32 as u32;
    }

    vp9.frameTagSize = fd.compressed_header_offset;
    vp9.offsetToDctParts = fd.compressed_header_size;
    vp9.reserved128Bits = [0; 4];

    // 4. Common CUVIDPICPARAMS. (The frame's refresh is committed by the
    // caller after cuvidDecodePicture: dpb.commit_frame(pi.refresh_frame_flags, curr_pic_idx).)
    let mut params = unsafe { std::mem::zeroed::<CUVIDPICPARAMS>() };
    params.PicWidthInMbs = (fd.frame_width / 16) as c_int;
    params.FrameHeightInMbs = (fd.frame_height / 16) as c_int;
    params.CurrPicIdx = curr_pic_idx;
    params.field_pic_flag = 0;
    params.bottom_field_flag = 0;
    params.second_field = 0;
    params.nBitstreamDataLen = bitstream_len;
    params.pBitstreamData = bitstream_ptr;
    params.nNumSlices = 1;
    params.pSliceDataOffsets = slice_offsets;
    params.ref_pic_flag = 0;
    params.intra_pic_flag = is_key as c_int;
    params.Reserved = [0; 30];
    params.CodecSpecific.vp9 = vp9;

    params
}

// ============================================================================
// NvdecVp9Decoder
// ============================================================================

/// Number of decode surfaces / DPB slots (matches the cuvid parser baseline).
const NUM_SURFACES: u32 = 16;

/// IVF header size in bytes (packets start at offset 32).
const IVF_HEADER_SIZE: usize = 32;

/// A single expanded VP9 frame (from a possibly-superframed IVF payload).
struct ExpandedVp9Frame {
    /// The frame bitstream (superframe-expanded).
    data: Vec<u8>,
    /// Offset of this frame within the superframe (0 if not from a superframe).
    superframe_frame_offset: u32,
}

/// Expand a VP9 superframe payload into individual frames.
///
/// A superframe ends with a superframe index; the last byte `b` with
/// `(b & 0xE0) == 0xC0` marks it. `num_frames = (b & 7) + 1`,
/// `mag = (((b >> 3) & 3) + 1)`, `index_size = 2 + mag * num_frames`. The
/// per-frame sizes are `mag`-byte little-endian values at the end of the
/// index. Frames are sliced out of the data region preceding the index.
///
/// Copied from the `decode_nvdec_vp9_cuvid` example (examples cannot be
/// imported by the library).
fn expand_superframes(payload: &[u8]) -> Vec<ExpandedVp9Frame> {
    let mut out = Vec::new();
    let data_len = payload.len();
    if data_len < 2 {
        out.push(ExpandedVp9Frame {
            data: payload.to_vec(),
            superframe_frame_offset: 0,
        });
        return out;
    }

    let final_byte = payload[data_len - 1];
    if (final_byte & 0xE0) != 0xC0 {
        // Not a superframe.
        out.push(ExpandedVp9Frame {
            data: payload.to_vec(),
            superframe_frame_offset: 0,
        });
        return out;
    }

    let num_frames = (final_byte & 0x07) as usize + 1;
    if num_frames <= 1 {
        out.push(ExpandedVp9Frame {
            data: payload.to_vec(),
            superframe_frame_offset: 0,
        });
        return out;
    }

    let mag = (((final_byte >> 3) & 0x03) as usize) + 1;
    let index_size = 2 + mag * num_frames;
    if data_len < index_size {
        out.push(ExpandedVp9Frame {
            data: payload.to_vec(),
            superframe_frame_offset: 0,
        });
        return out;
    }

    let index_start = data_len - index_size;
    if payload[index_start] != final_byte {
        out.push(ExpandedVp9Frame {
            data: payload.to_vec(),
            superframe_frame_offset: 0,
        });
        return out;
    }

    let frame_data_size = data_len - index_size;
    let mut offset = 0usize;
    let mut x = index_start + 1;
    for _ in 0..num_frames {
        let mut this_sz = 0usize;
        for j in 0..mag {
            this_sz |= (payload[x + j] as usize) << (j * 8);
        }
        x += mag;
        if offset + this_sz <= frame_data_size {
            out.push(ExpandedVp9Frame {
                data: payload[offset..offset + this_sz].to_vec(),
                superframe_frame_offset: offset as u32,
            });
        }
        offset += this_sz;
    }
    out
}

/// Map a full bit depth (e.g. 8, 10, 12) to a [`ComponentBitDepth`].
fn bit_depth_component(bit_depth: u8) -> ComponentBitDepth {
    match bit_depth {
        8 => ComponentBitDepth::Bit8,
        10 => ComponentBitDepth::Bit10,
        12 => ComponentBitDepth::Bit12,
        _ => ComponentBitDepth::Bit8,
    }
}

/// Dump the exact [`CUVIDPICPARAMS`] submitted for one VP9 picture (DECODE
/// order) to `path`, appending (truncating on the first picture). Mirrors the
/// format of `dump_cuvid_hevc_picparams` loosely.
fn dump_cuvid_vp9_picparams(path: &std::path::Path, pic_num: u32, p: &CUVIDPICPARAMS) {
    use std::io::Write;

    let file = match std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(pic_num == 0)
        .append(pic_num != 0)
        .open(path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[NVDEC-DUMP] cannot open {}: {}", path.display(), e);
            return;
        }
    };
    let mut out = std::io::BufWriter::new(file);

    let v = unsafe { &p.CodecSpecific.vp9 };

    let mut s = String::new();
    s.push_str(&format!("=== PIC {} (decode) ===\n", pic_num));
    s.push_str(&format!(
        "PicWidthInMbs={} FrameHeightInMbs={} CurrPicIdx={} nNumSlices={} nBitstreamDataLen={}\n",
        p.PicWidthInMbs, p.FrameHeightInMbs, p.CurrPicIdx, p.nNumSlices, p.nBitstreamDataLen
    ));
    s.push_str(&format!(
        "field_pic_flag={} bottom_field_flag={} second_field={} ref_pic_flag={} intra_pic_flag={}\n",
        p.field_pic_flag, p.bottom_field_flag, p.second_field, p.ref_pic_flag, p.intra_pic_flag
    ));
    s.push_str(&format!(
        "  [vp9] width={} height={} LastRefIdx={} GoldenRefIdx={} AltRefIdx={}\n",
        v.width, v.height, v.LastRefIdx, v.GoldenRefIdx, v.AltRefIdx
    ));
    s.push_str(&format!(
        "  [vp9] profile={} frameContextIdx={} frameType={} showFrame={} errorResilient={} frameParallelDecoding={}\n",
        v.profile(), v.frame_context_idx(), v.frame_type(), v.show_frame(),
        v.error_resilient(), v.frame_parallel_decoding()
    ));
    s.push_str(&format!(
        "  [vp9] subSamplingX={} subSamplingY={} intraOnly={} allow_high_precision_mv={} refreshEntropyProbs={}\n",
        v.sub_sampling_x(), v.sub_sampling_y(), v.intra_only(),
        v.allow_high_precision_mv(), v.refresh_entropy_probs()
    ));
    s.push_str(&format!(
        "  [vp9] bitDepthLuma={} bitDepthChroma={} loopFilterLevel={} loopFilterSharpness={} modeRefLfEnabled={}\n",
        v.bitDepthMinus8Luma, v.bitDepthMinus8Chroma, v.loopFilterLevel,
        v.loopFilterSharpness, v.modeRefLfEnabled
    ));
    s.push_str(&format!(
        "  [vp9] log2_tile_columns={} log2_tile_rows={} segmentEnabled={} segmentMapUpdate={} segmentMapTemporalUpdate={} segmentFeatureMode={}\n",
        v.log2_tile_columns, v.log2_tile_rows, v.segment_enabled(),
        v.segment_map_update(), v.segment_map_temporal_update(), v.segment_feature_mode()
    ));
    s.push_str(&format!(
        "  [vp9] qpYAc={} qpYDc={} qpChDc={} qpChAc={}\n",
        v.qpYAc, v.qpYDc, v.qpChDc, v.qpChAc
    ));
    s.push_str(&format!(
        "  [vp9] activeRefIdx=[{}, {}, {}] resetFrameContext={} mcomp_filter_type={}\n",
        v.activeRefIdx[0],
        v.activeRefIdx[1],
        v.activeRefIdx[2],
        v.resetFrameContext,
        v.mcomp_filter_type
    ));
    s.push_str(&format!(
        "  [vp9] frameTagSize={} offsetToDctParts={}\n",
        v.frameTagSize, v.offsetToDctParts
    ));

    let _ = out.write_all(s.as_bytes());
    let _ = out.flush();
}

/// NVDEC VP9 decoder using vacc-parser.
///
/// Not `Send`/`Sync`; use from a single thread. The CUDA context must be set
/// current before decode methods.
///
/// Driven by the [`Vp9Parser`]: IVF packets (or a raw single frame) are
/// superframe-expanded, parsed, and each frame's [`CUVIDPICPARAMS`] is built
/// by [`build_cuvid_vp9_picparams`] (surface management via [`Vp9DpbState`])
/// and submitted to `cuvidDecodePicture`. Displayed frames are extracted in
/// display order (NV12 → planar YUV420P).
pub struct NvdecVp9Decoder {
    parser: Vp9Parser,
    decoder: Mutex<CUvideodecoder>,
    info: Mutex<DecoderInfo>,
    pending_frames: Mutex<VecDeque<DecodedFrame>>,
    /// Decode-order frame count (every picture submitted to the decoder).
    frame_count: Mutex<u32>,
    /// Display-order count; used as the `frame_index`/`poc` of output frames.
    display_count: u32,
    /// (left, top, right, bottom) crop region within the coded surface.
    display_area: Mutex<(i32, i32, i32, i32)>,
    initialized: Mutex<bool>,
    /// DPB surface management (16 surfaces).
    dpb: Mutex<Vp9DpbState>,
    /// (width, height) of the last decoder configuration.
    prev_coded_size: Mutex<(u32, u32)>,
    pending_data: Vec<u8>,
    parsed_offset: usize,
    /// True if the input is an IVF container (packets start at offset 32).
    is_ivf: bool,
    /// Cached pinned host buffer for frame extraction.
    pinned_cache: Mutex<Option<(*mut std::ffi::c_void, usize)>>,
    /// If set (via `NVDEC_DUMP_PARAMS`), dump the exact [`CUVIDPICPARAMS`]
    /// submitted for each picture (DECODE order) to this path.
    dump_params_path: Option<std::path::PathBuf>,
    dump_params_count: u32,
    /// Per-decode slice-offset storage: `[0, bitstream_len, 0, ...]`.
    ///
    /// The NVDEC front end reads `nNumSlices + 1` entries from
    /// `pSliceDataOffsets` (the last one is the terminating offset), so a
    /// single-element array would make the driver read adjacent memory and
    /// corrupt the decode. See the AV1 decoder for details.
    slice_offsets: [u32; 64],
}

impl NvdecVp9Decoder {
    /// Create a new NVDEC VP9 decoder and begin decoding the input data.
    ///
    /// `data` is either an IVF container (magic `DKIF`, packets at offset 32)
    /// or a raw single VP9 frame.
    pub fn new(data: Vec<u8>) -> NvdecResult<Self> {
        init_nvdec()?;

        let is_ivf = data.len() >= IVF_HEADER_SIZE && &data[0..4] == b"DKIF";

        let mut decoder = Self {
            parser: Vp9Parser::new(),
            decoder: Mutex::new(std::ptr::null_mut()),
            info: Mutex::new(DecoderInfo {
                backend: "nvdec".to_string(),
                codec: VideoCodec::DecodeVp9,
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
            dpb: Mutex::new(Vp9DpbState::new(NUM_SURFACES)),
            prev_coded_size: Mutex::new((0, 0)),
            pending_data: data,
            parsed_offset: if is_ivf { IVF_HEADER_SIZE } else { 0 },
            is_ivf,
            pinned_cache: Mutex::new(None),
            dump_params_path: std::env::var("NVDEC_DUMP_PARAMS")
                .ok()
                .map(std::path::PathBuf::from),
            dump_params_count: 0,
            slice_offsets: [0; 64],
        };

        decoder.init_parser_format()?;
        decoder.parse_and_decode()?;

        let initialized = *decoder.initialized.lock().unwrap();
        if !initialized {
            return Err(NvdecError::DecoderCreationFailed(
                "Parser did not initialize decoder - no VP9 frame found".into(),
            ));
        }

        Ok(decoder)
    }

    /// Initialize the parser with the VP9 format (required before parsing).
    fn init_parser_format(&mut self) -> NvdecResult<()> {
        self.parser
            .init(&DetectedVideoFormat::new(VideoCodec::DecodeVp9))
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
                let payload =
                    &self.pending_data[self.parsed_offset + 12..self.parsed_offset + 12 + size];
                for f in expand_superframes(payload) {
                    self.process_frame(&f.data, f.superframe_frame_offset)?;
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

    /// Parse and decode one (superframe-expanded) VP9 frame.
    fn process_frame(&mut self, data: &[u8], superframe_offset: u32) -> NvdecResult<()> {
        let parsed = self
            .parser
            .parse_frame_with_offset(data, superframe_offset)
            .map_err(|e| NvdecError::DecodeFailed(format!("parse_frame: {}", e)))?;

        // Create the decoder on the first frame; reconfigure/recreate on a
        // coded-size change.
        let initialized = *self.initialized.lock().unwrap();
        if !initialized {
            self.create_decoder(&parsed)?;
        } else {
            let (prev_w, prev_h) = *self.prev_coded_size.lock().unwrap();
            if parsed.frame_width != prev_w || parsed.frame_height != prev_h {
                self.recreate_decoder(&parsed)?;
            }
        }

        if parsed.show_existing_frame {
            // show_existing_frame: no decode, no DPB change — re-display an
            // already-decoded surface (always displayed).
            let surface = {
                let dpb = self.dpb.lock().unwrap();
                surface_of_frame_buffer(&dpb, parsed.frame_to_show_map_idx as usize)
            };
            if surface < 255 {
                if let Some(frame) = self.extract_frame(surface) {
                    self.pending_frames.lock().unwrap().push_back(frame);
                }
                self.display_count += 1;
            }
            return Ok(());
        }

        // Build the picparams (reads the common DPB for references and picks
        // the output surface; the frame's refresh is committed below, after
        // the decode, so subsequent frames see the updated frame buffers).
        self.slice_offsets[0] = 0;
        self.slice_offsets[1] = data.len() as u32;
        let params = {
            let mut dpb = self.dpb.lock().unwrap();
            build_cuvid_vp9_picparams(
                &parsed,
                &mut dpb,
                data.as_ptr(),
                data.len() as u32,
                self.slice_offsets.as_ptr().cast::<c_uint>(),
            )
        };

        if let Some(dump_path) = &self.dump_params_path {
            dump_cuvid_vp9_picparams(dump_path, self.dump_params_count, &params);
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

        // Decode is async; extraction must wait for completion.
        let _ = cu_ctx_synchronize();

        // Commit this frame's refresh into the common DPB so that the next
        // frame's references resolve to it. Key frames pass 0xFF (all 8
        // frame buffers refreshed).
        {
            let mut dpb = self.dpb.lock().unwrap();
            dpb.commit_frame(parsed.picture_info.refresh_frame_flags, params.CurrPicIdx);
        }

        if parsed.picture_info.flags.show_frame != 0 {
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
    ///
    /// Uses the raw coded size (no alignment) — the cuvid baseline used the
    /// raw coded size and was pixel-perfect.
    fn create_decoder(&mut self, first: &Vp9FrameData) -> NvdecResult<()> {
        let w = first.frame_width;
        let h = first.frame_height;
        let cc = &first.color_config;

        // cuvid VP9 decode only supports 4:2:0. Reject 4:4:4/4:2:2 (and RGB)
        // streams up front instead of failing mid-decode in cuvidDecodePicture.
        if cc.subsampling_x != 1 || cc.subsampling_y != 1 {
            return Err(NvdecError::DecoderCreationFailed(format!(
                "VP9 chroma subsampling {}x{} not supported by NVDEC decode (4:2:0 only)",
                cc.subsampling_x, cc.subsampling_y
            )));
        }

        // 8-bit content: NV12. 10/12-bit content: P016, which preserves the
        // full-precision samples (10-bit left-aligned in 16 bits with 6 LSBs
        // zero; 12-bit with 4 LSBs zero). The readback scales them to 8-bit
        // with round+clamp, matching the Vulkan G10X6/G12X4 path. cuvid's
        // NV12 output for 10/12-bit is a dithered down-convert (maxdiff 1 vs
        // full precision), so P016 is required for pixel-perfect verification.
        let output_format = if cc.bit_depth > 8 {
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_P016
        } else {
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12
        };

        let create_info = CUVIDDECODECREATEINFO {
            ulWidth: w as _,
            ulHeight: h as _,
            ulNumDecodeSurfaces: NUM_SURFACES as _,
            CodecType: cudaVideoCodec::cudaVideoCodec_VP9,
            ChromaFormat: cudaVideoChromaFormat::cudaVideoChromaFormat_420,
            ulCreationFlags: 0,
            bitDepthMinus8: (cc.bit_depth.saturating_sub(8)) as _,
            ulIntraDecodeOnly: 0,
            ulMaxWidth: w as _,
            ulMaxHeight: h as _,
            Reserved1: 0,
            // The decoder's display_area must match ulTargetWidth/Height (the
            // raw coded size) or cuvid scales the output.
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
                "cuvidCreateDecoder failed with error {} (VP9 {}x{} bit_depth={}; \
                 NVIDIA NVDEC supports VP9 up to 10-bit — 12-bit streams are not \
                 decodable on any NVIDIA GPU)",
                result, w, h, cc.bit_depth
            )));
        }
        {
            let mut decoder = self.decoder.lock().unwrap();
            *decoder = ph_decoder;
        }

        let mut info = self.info.lock().unwrap();
        *info = DecoderInfo {
            backend: "nvdec".to_string(),
            codec: VideoCodec::DecodeVp9,
            coded_size: Extent2D {
                width: w,
                height: h,
            },
            display_size: Extent2D {
                width: w,
                height: h,
            },
            chroma_subsampling: ChromaSubsampling::_420,
            luma_bit_depth: bit_depth_component(cc.bit_depth),
            chroma_bit_depth: bit_depth_component(cc.bit_depth),
            profile_idc: Some(first.picture_info.profile as u32),
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
    ///
    /// Prefers `cuvidReconfigureDecoder` (preserves the DPB / reference
    /// surfaces). Falls back to destroy + recreate (resets the DPB) when the
    /// symbol is unavailable or the reconfigure fails.
    fn recreate_decoder(&mut self, fd: &Vp9FrameData) -> NvdecResult<()> {
        eprintln!(
            "[recreate] VP9 decoder reconfigured {}x{}",
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
            {
                let mut dpb = self.dpb.lock().unwrap();
                dpb.reset();
            }
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
            return self.create_decoder(fd);
        }

        // Reconfigured in place: update info + display area + prev size. The
        // DPB surface mapping is unchanged (reconfigure preserves surfaces).
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
    ///
    /// Maps the surface, copies the Y plane and interleaved UV (at the crop
    /// offset), deinterleaves NV12 UV into planar U/V, and unmaps. The
    /// resulting `frame_index`/`poc` is the current [`display_count`].
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
            top_field_first: 1,
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

        // P016 output holds 16-bit samples (10/12-bit content); NV12 is 8-bit.
        let luma_bd = info.luma_bit_depth.bit_depth();
        let bps = if luma_bd > 8 { 2 } else { 1 }; // bytes per sample on device
        let row_bytes = display_width * bps;

        let y_size = row_bytes * display_height;
        let interleaved_uv_size = row_bytes * (display_height / 2);
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
            srcXInBytes: (crop_left as u64) * bps as u64,
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
            dstPitch: row_bytes as u64,
            WidthInBytes: row_bytes as u64,
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
            srcXInBytes: (crop_left as u64) * bps as u64,
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
            dstPitch: row_bytes as u64,
            WidthInBytes: row_bytes as u64,
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

        let mut raw_y = vec![0u8; y_size];
        let mut raw_uv = vec![0u8; interleaved_uv_size];
        unsafe {
            std::ptr::copy_nonoverlapping(pinned_y as *const u8, raw_y.as_mut_ptr(), y_size);
            std::ptr::copy_nonoverlapping(
                pinned_uv as *const u8,
                raw_uv.as_mut_ptr(),
                interleaved_uv_size,
            );
        }

        let ss = bps;
        let uv_size = (display_width / 2) * (display_height / 2) * ss;
        let mut y_plane = vec![0u8; display_width * display_height * ss];
        let mut u_plane = vec![0u8; uv_size];
        let mut v_plane = vec![0u8; uv_size];

        if ss == 1 {
            // NV12: samples are already 8-bit. De-interleave UV to planar.
            y_plane.copy_from_slice(&raw_y);
            for py in 0..(display_height / 2) {
                for x in 0..(display_width / 2) {
                    let src_idx = py * display_width + x * 2;
                    let dst_idx = py * (display_width / 2) + x;
                    u_plane[dst_idx] = raw_uv[src_idx];
                    v_plane[dst_idx] = raw_uv[src_idx + 1];
                }
            }
        } else {
            // P016: 16-bit LE samples, left-aligned (10-bit: 6 LSBs zero,
            // 12-bit: 4 LSBs zero). Shift back to native bit depth.
            let shift = 16u32 - luma_bd;
            for chunk in raw_y.chunks_exact_mut(2) {
                let s = u16::from_le_bytes([chunk[0], chunk[1]]) >> shift;
                chunk.copy_from_slice(&s.to_le_bytes());
            }
            for chunk in raw_uv.chunks_exact_mut(2) {
                let s = u16::from_le_bytes([chunk[0], chunk[1]]) >> shift;
                chunk.copy_from_slice(&s.to_le_bytes());
            }
            y_plane.copy_from_slice(&raw_y);
            for py in 0..(display_height / 2) {
                for x in 0..(display_width / 2) {
                    let src_idx = (py * display_width + x * 2) * 2;
                    let dst_idx = (py * (display_width / 2) + x) * 2;
                    u_plane[dst_idx] = raw_uv[src_idx];
                    u_plane[dst_idx + 1] = raw_uv[src_idx + 1];
                    v_plane[dst_idx] = raw_uv[src_idx + 2];
                    v_plane[dst_idx + 1] = raw_uv[src_idx + 3];
                }
            }
        }

        let out_y_size = display_width * display_height * ss;
        let mut buffer = Vec::with_capacity(out_y_size + uv_size * 2);
        buffer.extend_from_slice(&y_plane);
        buffer.extend_from_slice(&u_plane);
        buffer.extend_from_slice(&v_plane);

        let y_ptr = buffer.as_ptr();
        let u_ptr = unsafe { buffer.as_ptr().add(out_y_size) };
        let v_ptr = unsafe { buffer.as_ptr().add(out_y_size + uv_size) };

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

impl Decoder for NvdecVp9Decoder {
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
        if codec != VideoCodec::DecodeVp9 {
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
        self.parser.reset();
        {
            let mut dpb = self.dpb.lock().unwrap();
            dpb.reset();
        }
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

impl Drop for NvdecVp9Decoder {
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
        // Free pinned host buffer to avoid leaking page-locked memory.
        if let Ok(mut cache) = self.pinned_cache.lock() {
            if let Some((ptr, _)) = cache.take() {
                let _ = unsafe { crate::device::cu_mem_free_host(ptr) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vacc_core::VideoCodec;
    use vacc_parser::vp9::Vp9Parser;
    use vacc_parser::{DetectedVideoFormat, VideoParser};

    /// Golden fixture: frames 0-19 of the original 1920x1080 bunny VP9 stream,
    /// against which the `EXPECTED` table (cuvid parameter dump) was derived.
    const GOLDEN_IVF: &[u8] = include_bytes!("../tests/data/vp9_dp_golden_20f.ivf");

    // ── IVF container parsing (copied from the decode_nvdec_vp9_cuvid example) ──

    struct IvfPackets(Vec<Vec<u8>>);

    fn parse_ivf(data: &[u8]) -> IvfPackets {
        assert!(data.len() >= 32, "file too small for IVF header");
        assert_eq!(&data[0..4], b"DKIF", "invalid IVF magic");
        let mut packets = Vec::new();
        let mut offset = 32usize;
        while offset + 12 <= data.len() {
            let size = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 12;
            if size == 0 || offset + size > data.len() {
                break;
            }
            packets.push(data[offset..offset + size].to_vec());
            offset += size;
        }
        IvfPackets(packets)
    }

    // ── VP9 superframe expansion (copied from the decode_nvdec_vp9_cuvid example) ──

    struct ExpandedFrame {
        data: Vec<u8>,
        superframe_frame_offset: u32,
    }

    fn expand_superframes(packets: &[Vec<u8>]) -> Vec<ExpandedFrame> {
        let mut out = Vec::new();
        for frame in packets {
            let data_len = frame.len();
            if data_len < 2 {
                out.push(ExpandedFrame {
                    data: frame.clone(),
                    superframe_frame_offset: 0,
                });
                continue;
            }
            let final_byte = frame[data_len - 1];
            if (final_byte & 0xE0) != 0xC0 {
                out.push(ExpandedFrame {
                    data: frame.clone(),
                    superframe_frame_offset: 0,
                });
                continue;
            }
            let num_frames = (final_byte & 0x07) as usize + 1;
            if num_frames <= 1 {
                out.push(ExpandedFrame {
                    data: frame.clone(),
                    superframe_frame_offset: 0,
                });
                continue;
            }
            let mag = (((final_byte >> 3) & 0x03) as usize) + 1;
            let index_size = 2 + mag * num_frames;
            if data_len < index_size {
                out.push(ExpandedFrame {
                    data: frame.clone(),
                    superframe_frame_offset: 0,
                });
                continue;
            }
            let index_start = data_len - index_size;
            if frame[index_start] != final_byte {
                out.push(ExpandedFrame {
                    data: frame.clone(),
                    superframe_frame_offset: 0,
                });
                continue;
            }
            let frame_data_size = data_len - index_size;
            let mut offset = 0;
            let mut x = index_start + 1;
            for _ in 0..num_frames {
                let mut this_sz = 0usize;
                for j in 0..mag {
                    this_sz |= (frame[x + j] as usize) << (j * 8);
                }
                x += mag;
                if offset + this_sz <= frame_data_size {
                    out.push(ExpandedFrame {
                        data: frame[offset..offset + this_sz].to_vec(),
                        superframe_frame_offset: offset as u32,
                    });
                }
                offset += this_sz;
            }
        }
        out
    }

    fn load_frames(n: usize) -> Vec<ExpandedFrame> {
        let packets = parse_ivf(GOLDEN_IVF);
        let expanded = expand_superframes(&packets.0);
        assert!(expanded.len() >= n, "not enough frames in IVF");
        expanded.into_iter().take(n).collect()
    }

    /// Parse `n` frames with the Rust Vp9Parser and build picparams for each.
    fn build_frames(n: usize) -> Vec<CUVIDPICPARAMS> {
        let frames = load_frames(n);
        let mut parser = Vp9Parser::new();
        parser
            .init(&DetectedVideoFormat::new(VideoCodec::DecodeVp9))
            .expect("failed to init Vp9Parser");
        let mut dpb = Vp9DpbState::new(16);
        let mut out = Vec::new();
        for f in &frames {
            let fd = parser
                .parse_frame_with_offset(&f.data, f.superframe_frame_offset)
                .expect("failed to parse frame");
            let mut slice_offsets = [0u32; 64];
            slice_offsets[1] = f.data.len() as u32;
            let params = build_cuvid_vp9_picparams(
                &fd,
                &mut dpb,
                f.data.as_ptr(),
                f.data.len() as u32,
                slice_offsets.as_ptr().cast::<c_uint>(),
            );
            // Mirror NvdecVp9Decoder::process_frame: commit the refresh after
            // the (simulated) decode.
            dpb.commit_frame(fd.picture_info.refresh_frame_flags, params.CurrPicIdx);
            out.push(params);
        }
        out
    }

    /// Expected per-frame values from the cuvid dump
    /// (`/tmp/pixel_verify/vp9_cuvid_params.txt`), frames 0-19.
    #[derive(Debug, Clone, Copy)]
    struct Expected {
        curr_pic_idx: i32,
        last_ref: u8,
        golden_ref: u8,
        alt_ref: u8,
        active_ref_idx: [u32; 3],
        frame_type: u32,
        frame_tag_size: u32,
        offset_to_dct_parts: u32,
        mcomp_filter_type: u32,
        qp_y_ac: i32,
        loop_filter_level: u8,
        allow_high_precision_mv: u32,
    }

    const EXPECTED: [Expected; 20] = [
        //  F0 (key)
        Expected {
            curr_pic_idx: 0,
            last_ref: 255,
            golden_ref: 255,
            alt_ref: 255,
            active_ref_idx: [0, 0, 0],
            frame_type: 0,
            frame_tag_size: 18,
            offset_to_dct_parts: 487,
            mcomp_filter_type: 0,
            qp_y_ac: 46,
            loop_filter_level: 3,
            allow_high_precision_mv: 0,
        },
        //  F1..F10
        Expected {
            curr_pic_idx: 1,
            last_ref: 0,
            golden_ref: 0,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 72,
            mcomp_filter_type: 4,
            qp_y_ac: 92,
            loop_filter_level: 7,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 2,
            last_ref: 1,
            golden_ref: 0,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 20,
            mcomp_filter_type: 4,
            qp_y_ac: 165,
            loop_filter_level: 5,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 3,
            last_ref: 2,
            golden_ref: 0,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 73,
            mcomp_filter_type: 4,
            qp_y_ac: 104,
            loop_filter_level: 6,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 4,
            last_ref: 3,
            golden_ref: 0,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 93,
            mcomp_filter_type: 4,
            qp_y_ac: 98,
            loop_filter_level: 8,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 5,
            last_ref: 4,
            golden_ref: 0,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 164,
            mcomp_filter_type: 4,
            qp_y_ac: 91,
            loop_filter_level: 6,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 6,
            last_ref: 5,
            golden_ref: 0,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 30,
            mcomp_filter_type: 4,
            qp_y_ac: 102,
            loop_filter_level: 7,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 7,
            last_ref: 6,
            golden_ref: 0,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 92,
            mcomp_filter_type: 4,
            qp_y_ac: 80,
            loop_filter_level: 7,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 8,
            last_ref: 7,
            golden_ref: 0,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 19,
            mcomp_filter_type: 4,
            qp_y_ac: 106,
            loop_filter_level: 7,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 9,
            last_ref: 8,
            golden_ref: 0,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 115,
            mcomp_filter_type: 4,
            qp_y_ac: 91,
            loop_filter_level: 6,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 10,
            last_ref: 9,
            golden_ref: 0,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 333,
            mcomp_filter_type: 4,
            qp_y_ac: 49,
            loop_filter_level: 5,
            allow_high_precision_mv: 1,
        },
        //  F11..F15 (GOLDEN refreshed at F10 -> golden_ref 10)
        Expected {
            curr_pic_idx: 11,
            last_ref: 10,
            golden_ref: 10,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 49,
            mcomp_filter_type: 4,
            qp_y_ac: 123,
            loop_filter_level: 7,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 12,
            last_ref: 11,
            golden_ref: 10,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 97,
            mcomp_filter_type: 4,
            qp_y_ac: 76,
            loop_filter_level: 6,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 13,
            last_ref: 12,
            golden_ref: 10,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 23,
            mcomp_filter_type: 4,
            qp_y_ac: 96,
            loop_filter_level: 7,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 14,
            last_ref: 13,
            golden_ref: 10,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 55,
            mcomp_filter_type: 4,
            qp_y_ac: 95,
            loop_filter_level: 6,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 15,
            last_ref: 14,
            golden_ref: 10,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 19,
            mcomp_filter_type: 4,
            qp_y_ac: 111,
            loop_filter_level: 7,
            allow_high_precision_mv: 1,
        },
        //  F16..F19 (wraparound: surface 0 is live as ALTREF, skipped)
        Expected {
            curr_pic_idx: 1,
            last_ref: 15,
            golden_ref: 10,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 56,
            mcomp_filter_type: 4,
            qp_y_ac: 91,
            loop_filter_level: 7,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 2,
            last_ref: 1,
            golden_ref: 10,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 24,
            mcomp_filter_type: 4,
            qp_y_ac: 121,
            loop_filter_level: 5,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 3,
            last_ref: 2,
            golden_ref: 10,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 69,
            mcomp_filter_type: 4,
            qp_y_ac: 93,
            loop_filter_level: 7,
            allow_high_precision_mv: 1,
        },
        Expected {
            curr_pic_idx: 4,
            last_ref: 3,
            golden_ref: 10,
            alt_ref: 0,
            active_ref_idx: [0, 1, 2],
            frame_type: 1,
            frame_tag_size: 10,
            offset_to_dct_parts: 22,
            mcomp_filter_type: 4,
            qp_y_ac: 128,
            loop_filter_level: 7,
            allow_high_precision_mv: 1,
        },
    ];

    #[test]
    fn picparams_match_cuvid_dump_frames_0_19() {
        let params = build_frames(20);
        assert_eq!(params.len(), 20);

        for (i, p) in params.iter().enumerate() {
            let e = &EXPECTED[i];
            // Safe: `params` was fully initialized by `build_cuvid_vp9_picparams`,
            // which wrote the `vp9` union member last.
            let v = unsafe { &p.CodecSpecific.vp9 };

            // Common CUVIDPICPARAMS.
            assert_eq!(p.PicWidthInMbs, 120, "F{i} PicWidthInMbs");
            assert_eq!(p.FrameHeightInMbs, 67, "F{i} FrameHeightInMbs");
            assert_eq!(p.CurrPicIdx, e.curr_pic_idx, "F{i} CurrPicIdx");
            assert_eq!(p.field_pic_flag, 0, "F{i} field_pic_flag");
            assert_eq!(p.bottom_field_flag, 0, "F{i} bottom_field_flag");
            assert_eq!(p.second_field, 0, "F{i} second_field");
            assert_eq!(p.nNumSlices, 1, "F{i} nNumSlices");
            assert_eq!(
                unsafe { *p.pSliceDataOffsets },
                0,
                "F{i} pSliceDataOffsets[0]"
            );
            assert_eq!(p.ref_pic_flag, 0, "F{i} ref_pic_flag");
            let intra = if i == 0 { 1 } else { 0 };
            assert_eq!(p.intra_pic_flag, intra, "F{i} intra_pic_flag");

            // VP9 common.
            assert_eq!(v.width, 1920, "F{i} width");
            assert_eq!(v.height, 1080, "F{i} height");
            assert_eq!(v.LastRefIdx, e.last_ref, "F{i} LastRefIdx");
            assert_eq!(v.GoldenRefIdx, e.golden_ref, "F{i} GoldenRefIdx");
            assert_eq!(v.AltRefIdx, e.alt_ref, "F{i} AltRefIdx");
            assert_eq!(v.colorSpace, 0, "F{i} colorSpace");
            assert_eq!(v.profile(), 0, "F{i} profile");
            assert_eq!(v.frame_context_idx(), 0, "F{i} frameContextIdx");
            assert_eq!(v.frame_type(), e.frame_type, "F{i} frameType");
            assert_eq!(v.show_frame(), 1, "F{i} showFrame");
            assert_eq!(v.error_resilient(), 0, "F{i} errorResilient");
            assert_eq!(v.frame_parallel_decoding(), 1, "F{i} frameParallelDecoding");
            assert_eq!(v.sub_sampling_x(), 1, "F{i} subSamplingX");
            assert_eq!(v.sub_sampling_y(), 1, "F{i} subSamplingY");
            assert_eq!(v.intra_only(), 0, "F{i} intraOnly");
            assert_eq!(
                v.allow_high_precision_mv(),
                e.allow_high_precision_mv,
                "F{i} allow_high_precision_mv"
            );
            assert_eq!(v.refresh_entropy_probs(), 1, "F{i} refreshEntropyProbs");
            assert_eq!(v.refFrameSignBias, [0, 0, 0, 0], "F{i} refFrameSignBias");
            assert_eq!(v.bitDepthMinus8Luma, 0, "F{i} bitDepthMinus8Luma");
            assert_eq!(v.bitDepthMinus8Chroma, 0, "F{i} bitDepthMinus8Chroma");
            assert_eq!(
                v.loopFilterLevel, e.loop_filter_level,
                "F{i} loopFilterLevel"
            );
            assert_eq!(v.loopFilterSharpness, 0, "F{i} loopFilterSharpness");
            assert_eq!(v.modeRefLfEnabled, 1, "F{i} modeRefLfEnabled");
            assert_eq!(v.log2_tile_columns, 2, "F{i} log2_tile_columns");
            assert_eq!(v.log2_tile_rows, 0, "F{i} log2_tile_rows");

            // Segmentation (disabled for this stream).
            assert_eq!(v.segment_enabled(), 0, "F{i} segmentEnabled");
            assert_eq!(v.segment_map_update(), 0, "F{i} segmentMapUpdate");
            assert_eq!(
                v.segment_map_temporal_update(),
                0,
                "F{i} segmentMapTemporalUpdate"
            );
            assert_eq!(v.segment_feature_mode(), 0, "F{i} segmentFeatureMode");
            assert!(
                v.segmentFeatureEnable.iter().all(|r| r == &[0, 0, 0, 0]),
                "F{i} segmentFeatureEnable"
            );
            assert!(
                v.segmentFeatureData.iter().all(|r| r == &[0, 0, 0, 0]),
                "F{i} segmentFeatureData"
            );
            assert_eq!(
                v.mb_segment_tree_probs, [0; 7],
                "F{i} mb_segment_tree_probs"
            );
            assert_eq!(v.segment_pred_probs, [0, 0, 0], "F{i} segment_pred_probs");

            // Quantization.
            assert_eq!(v.qpYAc, e.qp_y_ac, "F{i} qpYAc");
            assert_eq!(v.qpYDc, 0, "F{i} qpYDc");
            assert_eq!(v.qpChDc, 0, "F{i} qpChDc");
            assert_eq!(v.qpChAc, 0, "F{i} qpChAc");

            // References.
            assert_eq!(v.activeRefIdx, e.active_ref_idx, "F{i} activeRefIdx");
            assert_eq!(v.resetFrameContext, 0, "F{i} resetFrameContext");
            assert_eq!(
                v.mcomp_filter_type, e.mcomp_filter_type,
                "F{i} mcomp_filter_type"
            );
            // [1, 0, -1, -1] sign-extended.
            assert_eq!(
                v.mbRefLfDelta,
                [1, 0, u32::MAX, u32::MAX],
                "F{i} mbRefLfDelta"
            );
            assert_eq!(v.mbModeLfDelta, [0, 0], "F{i} mbModeLfDelta");

            // Header offsets.
            assert_eq!(v.frameTagSize, e.frame_tag_size, "F{i} frameTagSize");
            assert_eq!(
                v.offsetToDctParts, e.offset_to_dct_parts,
                "F{i} offsetToDctParts"
            );
        }
    }

    #[test]
    fn wraparound_reuses_oldest_non_live_surface() {
        // After 16 distinct surfaces (frames 0-15), frame 16 must reuse the
        // oldest non-live surface. Surface 0 is live (ALTREF), so frame 16
        // gets surface 1, then 2, 3, 4.
        let params = build_frames(20);
        assert_eq!(params[16].CurrPicIdx, 1, "F16 must reuse surface 1");
        assert_eq!(params[17].CurrPicIdx, 2, "F17 must reuse surface 2");
        assert_eq!(params[18].CurrPicIdx, 3, "F18 must reuse surface 3");
        assert_eq!(params[19].CurrPicIdx, 4, "F19 must reuse surface 4");
    }

    #[test]
    fn dpb_first_frame_is_surface_0() {
        let mut dpb = Vp9DpbState::new(16);
        assert_eq!(dpb.choose_output_slot(), 0);
        // Second frame: surface 0 is now live (all fb -> 0 after key refresh),
        // so the next non-live surface is 1.
        dpb.commit_frame(0xFF, 0);
        assert_eq!(dpb.choose_output_slot(), 1);
    }

    #[test]
    fn dpb_surface_of_frame_buffer_invalid_is_255() {
        let dpb = Vp9DpbState::new(16);
        assert_eq!(surface_of_frame_buffer(&dpb, 0), 255);
        assert_eq!(surface_of_frame_buffer(&dpb, 7), 255);
        let mut dpb = dpb;
        dpb.commit_frame(0b001, 5);
        assert_eq!(surface_of_frame_buffer(&dpb, 0), 5);
        assert_eq!(surface_of_frame_buffer(&dpb, 1), 255);
    }

    #[test]
    fn dpb_reset_clears_state() {
        let mut dpb = Vp9DpbState::new(16);
        dpb.commit_frame(0xFF, 3);
        dpb.choose_output_slot();
        dpb.reset();
        assert_eq!(surface_of_frame_buffer(&dpb, 0), 255);
        assert_eq!(dpb.choose_output_slot(), 0);
    }
}

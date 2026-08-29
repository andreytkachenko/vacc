//! VAAPI video decoder implementing the Decoder trait.
//!
//! Uses cros-libva's typestate Picture pattern for safe decode operations.
//! Supports H.264, H.265, VP9 decoding with proper buffer management.

use std::collections::VecDeque;
use std::rc::Rc;

use libva::{
    BufferType, Config, Context, Display, IQMatrix, IQMatrixBufferH264,
    PictureParameter, PictureParameterBufferH264, PictureH264, H264SeqFields,
    H264PicFields, SliceParameter, SliceParameterBufferH264,
    Picture, PictureNew, PictureEnd, PictureRender, PictureSync, Surface, Image,
};
use libva::VAProfile::Type as VAProfileType;
use libva::{
    VA_INVALID_ID, VA_SLICE_DATA_FLAG_ALL,
    VA_PICTURE_H264_INVALID, VA_PICTURE_H264_SHORT_TERM_REFERENCE,
    VA_PICTURE_H264_LONG_TERM_REFERENCE, VA_PICTURE_H264_TOP_FIELD,
    VA_PICTURE_H264_BOTTOM_FIELD,
};

use vk_video_core::{
    codec::VideoCodec as CoreVideoCodec,
    decoder::{Decoder, DecoderInfo},
    frame::{DecodedFrame, PixelData, PixelPlane},
    format::{ChromaSubsampling, ComponentBitDepth, VideoFormat},
    session::Extent2D,
    picture::{H264Sps, H264Pps},
};
use vk_video_parser::{
    bitstream::BitstreamPacket, h264::H264Parser,
    h264_dpb::{H264Dpb, H264MmcoCommand, MARKING_LONG},
    h264_poc::PocCalculator,
    DetectedVideoFormat, ParseResult, SliceHeader, VideoParser,
};

use super::{Error, Result};

/// Custom surface memory descriptor that requests DRM_PRIME_2 memory type.
/// This is required for export_prime to work on NVIDIA GPUs.
#[derive(Clone, Copy, Default)]
struct DmaBufSurfaceDescriptor;

impl libva::SurfaceMemoryDescriptor for DmaBufSurfaceDescriptor {
    fn add_attrs(&mut self, attrs: &mut Vec<libva::VASurfaceAttrib>) -> Option<Box<dyn std::any::Any>> {
        // NVIDIA NVDEC requires surfaces to be allocated with DRM_PRIME_2 memory type
        // for vaExportSurfaceHandle to succeed. Without it the driver returns
        // VA_ERROR_INVALID_SURFACE ("invalid VASurfaceID") on export.
        attrs.push(libva::VASurfaceAttrib::new_memory_type(libva::MemoryType::DrmPrime2));
        None
    }
}

/// Convert H.264 zigzag-ordered 4x4 scaling list to raster order for VAAPI.
fn zigzag_to_raster_4x4(src: [u8; 16]) -> [u8; 16] {
    const ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
    let mut dst = [0u8; 16];
    for i in 0..16 {
        dst[ZIGZAG_4X4[i]] = src[i];
    }
    dst
}

/// Convert H.264 zigzag-ordered 8x8 scaling list to raster order for VAAPI.
fn zigzag_to_raster_8x8(src: [u8; 64]) -> [u8; 64] {
    const ZIGZAG_8X8: [usize; 64] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
        20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];
    let mut dst = [0u8; 64];
    for i in 0..64 {
        dst[ZIGZAG_8X8[i]] = src[i];
    }
    dst
}

/// Convert the parser's H.264 slice_type to the slice_type value the NVIDIA
/// VA-API driver expects.
///
/// The parser reports the raw H.264 spec slice_type modulo 5
/// (vk-video-parser/src/h264.rs:551), i.e. 0=P, 1=B, 2=I, 3=SP, 4=SI.
///
/// The NVIDIA driver (drv_h264.c:copyH264SliceParam) sets `intra_pic_flag = 0`
/// for any slice_type other than 2 (I) or 4 (SI). So intra slices must map to
/// 2 (I) or 4 (SI) to keep `intra_pic_flag = 1`, and inter slices (P/B/SP)
/// must map to a value outside {2,4}.
fn h264_slice_type_to_vaapi(slice_type: u32) -> u8 {
    // VASliceParameterBufferH264.slice_type uses the raw H.264 modulo-5
    // semantics (FFmpeg's ff_h264_get_slice_type returns exactly this:
    // P=0, B=1, I=2, SP=3, SI=4). The driver parses the slice NAL according
    // to this value, so a P slice sent as 1 (=B) is misparsed as a B slice
    // (extra header fields + different MB syntax) -> garbage output.
    match slice_type % 5 {
        0 => 0, // P
        1 => 1, // B
        2 => 2, // I
        3 => 3, // SP
        4 => 4, // SI
        _ => 2, // unreachable (% 5 in 0..=4)
    }
}

/// TEMP DEBUG (VACC_VA_DUMP=1): dump the exact VAPictureParameterBufferH264.
fn dump_va_pic(p: &PictureParameterBufferH264) {
    let pp = p.inner();
    let cp = &pp.CurrPic;
    println!(
        "VADUMP PIC frame={} curr=(id={},fidx={},fl={},top={},bot={}) w_mbs={} h_mbs={} bdl={} bdc={} nrf={} seq={:#010x} picf={:#010x} nsg={} sgt={} qp={} qs={} c0={} c1={}",
        pp.frame_num, cp.picture_id, cp.frame_idx, cp.flags, cp.TopFieldOrderCnt, cp.BottomFieldOrderCnt,
        pp.picture_width_in_mbs_minus1, pp.picture_height_in_mbs_minus1,
        pp.bit_depth_luma_minus8, pp.bit_depth_chroma_minus8, pp.num_ref_frames,
        unsafe { pp.seq_fields.value }, unsafe { pp.pic_fields.value },
        pp.num_slice_groups_minus1, pp.slice_group_map_type,
        pp.pic_init_qp_minus26, pp.pic_init_qs_minus26, pp.chroma_qp_index_offset, pp.second_chroma_qp_index_offset,
    );
    for (i, r) in pp.ReferenceFrames.iter().enumerate() {
        if r.flags != libva::VA_PICTURE_H264_INVALID {
            eprintln!("VADUMP  REF[{}]=(id={},fidx={},fl={},top={},bot={})", i, r.picture_id, r.frame_idx, r.flags, r.TopFieldOrderCnt, r.BottomFieldOrderCnt);
        }
    }
}

/// TEMP DEBUG (VACC_VA_DUMP=1): dump the exact VASliceParameterBufferH264.
fn dump_va_slice(s: &SliceParameterBufferH264) {
    for sp in s.inner().iter() {
        let n0 = sp.num_ref_idx_l0_active_minus1 as usize + 1;
        let n1 = sp.num_ref_idx_l1_active_minus1 as usize + 1;
        let l0: Vec<String> = sp.RefPicList0[..n0.min(32)].iter().map(|r| format!("(id={},fidx={},fl={},top={})", r.picture_id, r.frame_idx, r.flags, r.TopFieldOrderCnt)).collect();
        let l1: Vec<String> = sp.RefPicList1[..n1.min(32)].iter().map(|r| format!("(id={},fidx={},fl={},top={})", r.picture_id, r.frame_idx, r.flags, r.TopFieldOrderCnt)).collect();
        println!(
            "VADUMP SLICE size={} off={} flag={} bitoff={} firstmb={} stype={} dsp={} nrl0={} nrl1={} cabac={} qpdel={} disdeb={} alpha={} beta={} lwd={} cwd={} lw0f={} cw0f={} lw1f={} cw1f={}",
            sp.slice_data_size, sp.slice_data_offset, sp.slice_data_flag, sp.slice_data_bit_offset,
            sp.first_mb_in_slice, sp.slice_type, sp.direct_spatial_mv_pred_flag,
            sp.num_ref_idx_l0_active_minus1, sp.num_ref_idx_l1_active_minus1, sp.cabac_init_idc,
            sp.slice_qp_delta, sp.disable_deblocking_filter_idc, sp.slice_alpha_c0_offset_div2, sp.slice_beta_offset_div2,
            sp.luma_log2_weight_denom, sp.chroma_log2_weight_denom,
            sp.luma_weight_l0_flag, sp.chroma_weight_l0_flag, sp.luma_weight_l1_flag, sp.chroma_weight_l1_flag,
        );
        eprintln!("VADUMP  L0[{}]", l0.join(","));
        eprintln!("VADUMP  L1[{}]", l1.join(","));
        eprintln!("VADUMP  LW0={:?} LO0={:?} CW0={:?}", &sp.luma_weight_l0[..n0.min(32)], &sp.luma_offset_l0[..n0.min(32)], &sp.chroma_weight_l0[..n0.min(32)]);
    }
}

/// Maximum reference frames for H.264.
const MAX_H264_REFS: usize = 16;

/// Reference picture entry for DPB management.
#[derive(Debug, Clone)]
struct RefPic {
    surface_id: libva::VASurfaceID,
    frame_num: u32,
    frame_num_offset: u32,
    long_term: bool,
    long_term_pic_num: Option<u32>,
    top_field_order_cnt: i32,
    bottom_field_order_cnt: i32,
}

/// Surface state tracking.
#[derive(Debug, Clone, PartialEq)]
enum SurfaceState {
    Free,
    Pending(libva::VASurfaceID),
    Ready(libva::VASurfaceID),
}

/// Surface pool entry.
struct SurfaceEntry {
    surface: Rc<Surface<DmaBufSurfaceDescriptor>>,
    state: SurfaceState,
    ref_pic: Option<RefPic>,
}

/// Surface pool for managing decode surfaces.
struct SurfacePool {
    entries: Vec<SurfaceEntry>,
}

impl SurfacePool {
    fn new(surfaces: Vec<Surface<DmaBufSurfaceDescriptor>>) -> Self {
        let entries = surfaces.into_iter().map(|s| SurfaceEntry {
            surface: Rc::new(s),
            state: SurfaceState::Free,
            ref_pic: None,
        }).collect();
        Self { entries }
    }

    fn alloc(&mut self, refs: &[RefPic]) -> Option<(usize, Rc<Surface<DmaBufSurfaceDescriptor>>)> {
        let ref_ids: std::collections::HashSet<_> = refs.iter().map(|r| r.surface_id).collect();
        for (i, entry) in self.entries.iter_mut().enumerate() {
            // Free surfaces: available if not used by DPB refs
            // Ready surfaces: available if no longer needed as DPB refs
            let is_reusable = match entry.state {
                SurfaceState::Free => true,
                SurfaceState::Ready(_) => true,
                SurfaceState::Pending(_) => false,
            };
            if is_reusable && !ref_ids.contains(&entry.surface.id()) {
                entry.state = SurfaceState::Pending(entry.surface.id());
                entry.ref_pic = None;
                return Some((i, Rc::clone(&entry.surface)));
            }
        }
        None
    }

    /// Allocate a free surface whose pool index is NOT in `used_pool`.
    ///
    /// Used by the H.264 path to avoid handing out a surface that is still
    /// tracked by a DPB slot (even one that is logically empty but not yet
    /// reused).
    fn alloc_excluding(
        &mut self,
        used_pool: &std::collections::HashSet<usize>,
    ) -> Option<(usize, Rc<Surface<DmaBufSurfaceDescriptor>>)> {
        for (i, entry) in self.entries.iter_mut().enumerate() {
            if used_pool.contains(&i) {
                continue;
            }
            let is_reusable = match entry.state {
                SurfaceState::Free => true,
                SurfaceState::Ready(_) => true,
                SurfaceState::Pending(_) => false,
            };
            if is_reusable {
                entry.state = SurfaceState::Pending(entry.surface.id());
                entry.ref_pic = None;
                return Some((i, Rc::clone(&entry.surface)));
            }
        }
        None
    }

    fn mark_ready(&mut self, idx: usize) {
        if let Some(entry) = self.entries.get_mut(idx) {
            if let SurfaceState::Pending(id) = entry.state {
                entry.state = SurfaceState::Ready(id);
            }
        }
    }

    fn free(&mut self, idx: usize) {
        if let Some(entry) = self.entries.get_mut(idx) {
            entry.state = SurfaceState::Free;
            entry.ref_pic = None;
        }
    }

    fn sync_surface(&self, idx: usize) -> Result<()> {
        if let Some(entry) = self.entries.get(idx) {
            entry.surface.sync().map_err(|e| Error::VaApi(e.to_string()))?;
            Ok(())
        } else {
            Err(Error::InvalidState("Invalid surface index".to_string()))
        }
    }
}

// NOTE: H.264 DPB management uses the common `vk_video_parser::h264_dpb::H264Dpb`
// (ONE common DPB manager across backends). The per-slot VA surface mapping is
// kept in `H264Context::slot_surfaces`.

/// Parsed stream information.
#[derive(Debug, Clone)]
struct StreamInfo {
    codec: CoreVideoCodec,
    profile: VAProfileType,
    width: u32,
    height: u32,
    display_width: u32,
    display_height: u32,
    max_dpb: u32,
    rt_format: u32,
    /// H.264 specific
    sps: Option<H264Sps>,
    pps: Option<H264Pps>,
}

/// H.264 decode context.
///
/// Uses the common decode-state foundation from `vk-video-parser`:
/// - `dpb`: the common `H264Dpb` manager (ONE DPB manager across backends).
/// - `poc_calc`: the common `PocCalculator` (ONE POC implementation).
/// - `slot_surfaces`: maps a common-DPB slot index to a VA surface-pool index.
struct H264Context {
    dpb: H264Dpb,
    poc_calc: PocCalculator,
    /// DPB slot index -> surface pool index (None if the slot has no surface).
    slot_surfaces: Vec<Option<usize>>,
    max_frame_num: u32,
    /// POC of the current picture, computed by `poc_calc` in decode order.
    curr_poc: i32,
}

/// Holds information about a single H.264 slice for multi-slice frame assembly
struct H264SliceInfo {
    nal_data: Vec<u8>,
    slice_header: Option<SliceHeader>,
}

/// VAAPI video decoder implementing the Decoder trait.
pub struct VaapiDecoder {
    _display: Rc<Display>,
    _config: Config,
    context: Rc<Context>,
    surface_pool: SurfacePool,
    stream: StreamInfo,
    pending_data: Vec<u8>,
    /// Offset into pending_data for incremental parsing
    parse_offset: usize,
    frame_count: u32,
    /// Reorder buffer of decoded-but-not-yet-emitted frames, keyed by display order.
    pending_frames: VecDeque<(i64, DecodedFrame)>,
    /// High-water mark of decoded GOP indices (for B-frame reordering).
    reorder_watermark: i64,
    /// Current GOP index (increments on each IDR) used to build a global display-order key.
    gop_count: u64,
    /// Display-order key of the most recently decoded frame.
    pending_key: i64,
    /// Codec-specific context
    h264_ctx: Option<H264Context>,
    parser: Option<H264Parser>,
}

impl VaapiDecoder {
    /// Create a new VAAPI decoder from initial bitstream data.
    pub fn new(data: Vec<u8>) -> Result<Self> {
        let display = Display::open()
            .ok_or_else(|| Error::DecoderInit("No VA display available".to_string()))?;

        // Parse stream to get codec and dimensions
        let stream = parse_stream_info(&display, &data)?;

        // Create config with RT format attribute (like cros-codecs does)
        let config = display.create_config(
            vec![libva::VAConfigAttrib {
                type_: libva::VAConfigAttribType::VAConfigAttribRTFormat,
                value: stream.rt_format,
            }],
            stream.profile,
            libva::VAEntrypoint::VAEntrypointVLD,
        ).map_err(|e| Error::DecoderInit(e.to_string()))?;

        // Create surfaces (DPB + extra for rendering). The pool must hold every
        // DPB reference plus the picture currently being decoded, so keep +4
        // slack beyond the SPS max_num_ref_frames.
        let num_surfaces = (stream.max_dpb as usize).max(4) + 4;
        let descriptors: Vec<DmaBufSurfaceDescriptor> = (0..num_surfaces).map(|_| DmaBufSurfaceDescriptor).collect();

         let surfaces = display.create_surfaces::<DmaBufSurfaceDescriptor>(
            stream.rt_format,
            None,
            stream.width,
            stream.height,
            None,
            descriptors,
        ).map_err(|e| Error::DecoderInit(e.to_string()))?;

        // Create context WITHOUT surfaces (like FFmpeg - surfaces are assigned per-picture)
        let context = display.create_context::<DmaBufSurfaceDescriptor>(
            &config,
            stream.width,
            stream.height,
            Some(&surfaces), // register all DPB surfaces as render targets (NVIDIA NVDEC requires this)
            true, // progressive
        ).map_err(|e| Error::DecoderInit(e.to_string()))?;

        let surface_pool = SurfacePool::new(surfaces);

        let h264_ctx = if stream.codec == CoreVideoCodec::DecodeH264 {
            let sps = stream.sps.as_ref()
                .ok_or_else(|| Error::DecoderInit("H264 SPS not available".to_string()))?;
            // Common DPB manager: one slot per surface, so a surface can always
            // be mapped to a slot. num_ref_frames is clamped to the slot count.
            let num_slots = num_surfaces;
            let num_ref_frames = sps.max_num_ref_frames.min(num_slots as u32).max(1);
            let dpb = H264Dpb::new(num_slots, num_slots, num_ref_frames, sps.max_frame_num);
            Some(H264Context {
                dpb,
                poc_calc: PocCalculator::new(),
                slot_surfaces: vec![None; num_slots],
                max_frame_num: sps.max_frame_num,
                curr_poc: 0,
            })
        } else {
            None
        };

        let parser = if stream.codec == CoreVideoCodec::DecodeH264 {
            let mut p = H264Parser::new();
            p.init(&DetectedVideoFormat::new(CoreVideoCodec::DecodeH264))
                .map_err(|e| Error::Parser(e.to_string()))?;
            Some(p)
        } else {
            None
        };

        Ok(Self {
            _display: display,
            _config: config,
            context,
            surface_pool,
            stream,
            pending_data: data,
            parse_offset: 0,
            frame_count: 0,
            pending_frames: VecDeque::new(),
            reorder_watermark: i64::MIN,
            gop_count: 0,
            pending_key: 0,
            h264_ctx,
            parser,
        })
    }

    /// Decode a complete H.264 frame consisting of multiple slices.
    /// All slices share the same picture parameters but have different slice parameters.
    fn decode_h264_frame_multi_slice(
        &mut self,
        slices: &[H264SliceInfo],
        timestamp: u64,
    ) -> Result<Option<DecodedFrame>> {
        if slices.is_empty() {
            return Ok(None);
        }

        let ctx = self.h264_ctx.as_mut()
            .ok_or_else(|| Error::InvalidState("H264 context not initialized".to_string()))?;
        let sps = self.stream.sps.as_ref()
            .ok_or_else(|| Error::InvalidState("H264 SPS not available".to_string()))?;
        let pps = self.stream.pps.as_ref()
            .ok_or_else(|| Error::InvalidState("H264 PPS not available".to_string()))?;

        // Use first slice's header for frame-level parameters
        let first_slice = &slices[0];
        let slice_header = first_slice.slice_header.as_ref();

        // Extract parameters from first slice header
        let (nal_unit_type, nal_ref_idc, num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1, field_pic_flag, bottom_field, idr_pic_id, no_output_of_prior_pics_flag, frame_num, slice_type, mmco, mod_l0, mod_l1) = if let Some(SliceHeader::H264(h264_slh)) = slice_header {
            (h264_slh.nal_unit_type, h264_slh.nal_ref_idc, h264_slh.num_ref_idx_l0_active_minus1, h264_slh.num_ref_idx_l1_active_minus1, h264_slh.field_pic_flag, h264_slh.bottom_field, h264_slh.idr_pic_id, h264_slh.no_output_of_prior_pics_flag, h264_slh.frame_num, h264_slh.slice_type % 5, &h264_slh.dec_ref_pic_marking[..], &h264_slh.ref_pic_list_modification_l0[..], &h264_slh.ref_pic_list_modification_l1[..])
        } else {
            (1, 3, pps.num_ref_idx_l0_default_active_minus1, pps.num_ref_idx_l1_default_active_minus1, false, false, 0, false, 0, 2, &[] as &[vk_video_parser::h264::DecRefPicMarkingEntry], &[] as &[vk_video_parser::h264::RefPicListModificationEntry], &[] as &[vk_video_parser::h264::RefPicListModificationEntry])
        };

        let is_idr = nal_unit_type == 5 || idr_pic_id > 0;
        let is_ref = nal_ref_idc != 0;

        // POC was computed in decode order by the common PocCalculator (in
        // decode_h264_pending). Derive the per-field POCs for the VA picture.
        let poc = ctx.curr_poc;
        // FFmpeg's fill_vaapi_pic: TopFieldOrderCnt = field_poc[0],
        // BottomFieldOrderCnt = field_poc[1]. For a frame picture
        // field_poc[1] == field_poc[0] (H.264 8.2: both field order counts
        // equal for frame pictures), so BottomFieldOrderCnt must equal the
        // top POC — iHD stores it in the DDI field-order table used for
        // direct-mode and implicit-weight POC math (a zero there corrupts
        // B-frame prediction).
        let (top_field_order_cnt, bottom_field_order_cnt) = if field_pic_flag {
            if bottom_field { (0, poc) } else { (poc, 0) }
        } else {
            (poc, poc)
        };

        // Build the current picture's VA flags. FFmpeg's fill_vaapi_pic sets
        // SHORT_TERM_REFERENCE for every non-droppable picture (pic->reference
        // = picture_structure, h264_slice.c), regardless of nal_ref_idc.
        let mut flags = VA_PICTURE_H264_SHORT_TERM_REFERENCE;
        if field_pic_flag {
            if bottom_field {
                flags |= VA_PICTURE_H264_BOTTOM_FIELD;
            } else {
                flags |= VA_PICTURE_H264_TOP_FIELD;
            }
        }

        // Convert the slice header's dec_ref_pic_marking into common-DPB MMCO
        // commands (H.264 8.2.5).
        let mmco_commands: Vec<H264MmcoCommand> = mmco.iter().map(|op| {
            match op.memory_management_control_operation {
                1 => H264MmcoCommand::UnmarkShortTerm { difference_of_pic_nums_minus1: op.value },
                2 => H264MmcoCommand::UnmarkLongTerm { long_term_frame_idx: op.value },
                3 => H264MmcoCommand::AssignLongTerm { difference_of_pic_nums_minus1: 0, long_term_frame_idx: op.value },
                4 => H264MmcoCommand::SetMaxLongTermFrameIdx { max_long_term_frame_idx_plus1: op.value },
                5 => H264MmcoCommand::UnmarkAll,
                6 => H264MmcoCommand::AssignLongTermToCurrent { long_term_frame_idx: op.value },
                _ => H264MmcoCommand::UnmarkAll,
            }
        }).collect();

        // Stage the current picture in the common DPB. The reference marking
        // process (MMCO / sliding window) is applied only AFTER this picture
        // has been decoded: spec 8.2.3 (list construction) runs on the DPB
        // BEFORE 8.2.5 (marking), and FFmpeg applies the marking in
        // ff_h264_field_end, after the picture is decoded. So the reference
        // lists and ReferenceFrames built below reflect the PRE-marking DPB
        // state (verified against `ffmpeg -debug mmco` + GT ref-list dumps:
        // e.g. h264_baseline frame 3 keeps the IDR as its 3rd L0 entry; the
        // eviction happens before frame 4's lists).
        ctx.dpb.picture_start(
            frame_num,
            poc,
            is_ref,
            is_idr,
            no_output_of_prior_pics_flag,
            !mmco_commands.is_empty(),
            mmco_commands,
        );

        // Pre-extract per-slot (surface_id, frame_num, poc, marking) from the
        // PRE-marking DPB state, so the array-building closures below do not
        // need to borrow `self`.
        let slot_info: Vec<(Option<libva::VASurfaceID>, u32, i32, u8)> =
            ctx.slot_surfaces.iter().enumerate().map(|(i, s)| {
                let sid = s.map(|pool_idx| self.surface_pool.entries[pool_idx].surface.id());
                let dslot = &ctx.dpb.slots[i];
                (sid, dslot.frame_num, dslot.poc, if dslot.state == 0 { 0 } else { dslot.marking })
            }).collect();

        let make_pic = |info: &(Option<libva::VASurfaceID>, u32, i32, u8)| -> PictureH264 {
            match (info.0, info.3) {
                // Frame references: BottomFieldOrderCnt == TopFieldOrderCnt
                // (H.264 8.2; FFmpeg writes field_poc[1] == field_poc[0]).
                (Some(sid), MARKING_SHORT) => {
                    PictureH264::new(sid, info.1, VA_PICTURE_H264_SHORT_TERM_REFERENCE, info.2, info.2)
                }
                (Some(sid), MARKING_LONG) => {
                    PictureH264::new(sid, info.1, VA_PICTURE_H264_LONG_TERM_REFERENCE, info.2, info.2)
                }
                // No surface, or the DPB slot is not marked as a reference:
                // must be INVALID, else the driver may treat the surface
                // (possibly the current picture's own destination!) as a ref.
                _ => PictureH264::new(VA_INVALID_ID, 0, VA_PICTURE_H264_INVALID, 0, 0),
            }
        };
        fn invalid_pic() -> PictureH264 {
            PictureH264::new(VA_INVALID_ID, 0, VA_PICTURE_H264_INVALID, 0, 0)
        }

        // Build the reference picture lists per-slice (spec 8.2.3.1 + 8.2.3.2)
        // from the common DPB. Must be done before commit_current (uses the
        // staged current picture).
        let mut slice_ref_lists: Vec<([PictureH264; 32], [PictureH264; 32])> =
            Vec::with_capacity(slices.len());
        for slice_info in slices.iter() {
            let (st, l0m, l1m) = match &slice_info.slice_header {
                Some(SliceHeader::H264(h)) => (
                    h.slice_type % 5,
                    &h.ref_pic_list_modification_l0[..],
                    &h.ref_pic_list_modification_l1[..],
                ),
                _ => (slice_type, mod_l0, mod_l1),
            };
            let sl = ctx.dpb.build_ref_lists(
                st,
                num_ref_idx_l0_active_minus1,
                num_ref_idx_l1_active_minus1,
                l0m,
                l1m,
            );
            let l0: [PictureH264; 32] = core::array::from_fn(|i| {
                sl.l0.get(i).map(|r| make_pic(&slot_info[r.slot])).unwrap_or_else(invalid_pic)
            });
            let l1: [PictureH264; 32] = core::array::from_fn(|i| {
                sl.l1.get(i).map(|r| make_pic(&slot_info[r.slot])).unwrap_or_else(invalid_pic)
            });
            slice_ref_lists.push((l0, l1));
        }

        // ReferenceFrames: every DPB reference picture available to THIS
        // picture (pre-marking state; the current picture is not in the DPB
        // yet, matching FFmpeg's fill_vaapi_ReferenceFrames).
        let ref_slots = ctx.dpb.get_references();

        // Allocate the destination surface: any pool surface not currently
        // backing a DPB slot. Pictures that the marking process is about to
        // evict are still referenced by THIS decode, so their surfaces must
        // not be recycled until after the decode (done below).
        let used_pool: std::collections::HashSet<usize> =
            ctx.slot_surfaces.iter().filter_map(|s| *s).collect();
        let (surface_idx, surface) = self.surface_pool.alloc_excluding(&used_pool)
            .ok_or_else(|| Error::InvalidState("No free surfaces available".to_string()))?;
        let surface_id = surface.id();

        let curr_pic = PictureH264::new(
            surface_id,
            frame_num,
            flags,
            top_field_order_cnt,
            bottom_field_order_cnt,
        );

        let refs: [PictureH264; MAX_H264_REFS] = core::array::from_fn(|i| {
            ref_slots.get(i).map(|&s| make_pic(&slot_info[s])).unwrap_or_else(invalid_pic)
        });

        let seq_fields = H264SeqFields::new(
            sps.chroma_format_idc as u32,
            sps.separate_colour_plane_flag as u32,
            sps.gaps_in_frame_num_value_allowed_flag as u32,
            sps.frame_mbs_only_flag as u32,
            sps.mb_adaptive_frame_field_flag as u32,
            sps.direct_8x8_inference_flag as u32,
            // A.3.3.2: MinLumaBiPredSize8x8 is set for level >= 3.1 (FFmpeg:
            // `sps->level_idc >= 31`; level_idc is in tenths, e.g. 30 = 3.0).
            (sps.level_idc >= 31) as u32,
            sps.log2_max_frame_num_minus4 as u32,
            sps.pic_order_cnt_type as u32,
            sps.log2_max_pic_order_cnt_lsb_minus4 as u32,
            sps.delta_pic_order_always_zero_flag as u32,
        );

        let picture_height_in_mbs_minus1 = if sps.frame_mbs_only_flag {
            sps.pic_height_in_map_units_minus1
        } else {
            (((sps.pic_height_in_map_units_minus1 as u32 + 1) * 2) - 1).try_into().unwrap()
        };

        let pic_fields = H264PicFields::new(
            pps.entropy_coding_mode_flag as u32,
            pps.weighted_pred_flag as u32,
            pps.weighted_bipred_idc as u32,
            pps.transform_8x8_mode_flag as u32,
            field_pic_flag as u32,
            pps.constrained_intra_pred_flag as u32,
            // libva 1.23 renamed this bit to
            // bottom_field_pic_order_in_frame_present_flag; FFmpeg writes the
            // PPS value (not a derivation from SPS pic_order_cnt_type).
            pps.bottom_field_pic_order_in_frame_present_flag as u32,
            pps.deblocking_filter_control_present_flag as u32,
            pps.redundant_pic_cnt_present_flag as u32,
            (nal_ref_idc != 0) as u32,
        );

        // Debug: log PicParam fields before construction
        if std::env::var("DBG_H264").is_ok() {
            eprintln!("[DBG-PICPARAM] scaling_present={} 8x8={} chroma_fmt={} bitdepth_l={} bitdepth_c={} max_ref={} w_mbs={} h_mbs={} init_qp={} init_qs={} chroma_qp_off={} log2fn={} poc_type={} log2poc={} dpaz={} gaps={} fmo={} mbaff={} direct8x8={} minluma={} entropy={} wpred={} wbipred={} t8x8={} field={} cip={} bfpoc={} deblock={} redpic={} refpic={} frame_num={}",
                sps.seq_scaling_matrix_present_flag, pps.transform_8x8_mode_flag, sps.chroma_format_idc,
                sps.bit_depth_luma_minus8, sps.bit_depth_chroma_minus8, sps.max_num_ref_frames,
                sps.pic_width_in_mbs_minus1, picture_height_in_mbs_minus1,
                pps.pic_init_qp_minus26, pps.pic_init_qs_minus26, pps.chroma_qp_index_offset,
                sps.log2_max_frame_num_minus4, sps.pic_order_cnt_type, sps.log2_max_pic_order_cnt_lsb_minus4, sps.delta_pic_order_always_zero_flag,
                sps.gaps_in_frame_num_value_allowed_flag, sps.frame_mbs_only_flag, sps.mb_adaptive_frame_field_flag, sps.direct_8x8_inference_flag, (sps.level_idc >= 41) as u32,
                pps.entropy_coding_mode_flag, pps.weighted_pred_flag, pps.weighted_bipred_idc, pps.transform_8x8_mode_flag, field_pic_flag, pps.constrained_intra_pred_flag, pps.bottom_field_pic_order_in_frame_present_flag, pps.deblocking_filter_control_present_flag, pps.redundant_pic_cnt_present_flag, (nal_ref_idc != 0) as u32, frame_num);
        }

        let pic_param = PictureParameterBufferH264::new(
            curr_pic,
            refs,
            sps.pic_width_in_mbs_minus1,
            picture_height_in_mbs_minus1,
            sps.bit_depth_luma_minus8,
            sps.bit_depth_chroma_minus8,
            sps.max_num_ref_frames as u8,
            &seq_fields,
            0, 0, 0,
            pps.pic_init_qp_minus26 as i8,
            pps.pic_init_qs_minus26 as i8,
            pps.chroma_qp_index_offset as i8,
            pps.second_chroma_qp_index_offset as i8,
            &pic_fields,
            frame_num as u16,
        );

        // Build IQ matrix buffer.
        // When the SPS has no explicit scaling lists (seq_scaling_matrix_present_flag == 0),
        // the parser leaves sps.scaling_list_* as all zeros. Feeding zeros to NVDEC makes
        // inverse-quantization produce all-zero residuals, which for an IDR (no prediction)
        // decodes every block to the neutral value (128) -> uniform gray. Use the H.264
        // default scaling lists instead. Verified empirically (pixel-perfect vs FFmpeg):
        // both 4x4 and 8x8 defaults must be all 16.
        let (scaling_list_4x4, scaling_list_8x8) = if sps.seq_scaling_matrix_present_flag {
            let mut sl4 = [[0u8; 16]; 6];
            let mut sl8 = [[0u8; 64]; 2];
            for i in 0..6 {
                sl4[i] = zigzag_to_raster_4x4(sps.scaling_list_4x4[i]);
            }
            for i in 0..2 {
                sl8[i] = zigzag_to_raster_8x8(sps.scaling_list_8x8[i]);
            }
            (sl4, sl8)
        } else {
            // H.264 default scaling lists (all values identical, so scan order is irrelevant).
            ([[16u8; 16]; 6], [[16u8; 64]; 2])
        };
        let iq_matrix = IQMatrixBufferH264::new(scaling_list_4x4, scaling_list_8x8);

        if std::env::var("VACC_VA_DUMP").is_ok() {
            dump_va_pic(&pic_param);
        }

        // Create picture parameter buffer (shared across all slices)
        let pic_param_buf = self.context.create_buffer(
            BufferType::PictureParameter(PictureParameter::H264(pic_param))
        ).map_err(|e| Error::VaApi(e.to_string()))?;

        let iq_buf = self.context.create_buffer(
            BufferType::IQMatrix(IQMatrix::H264(iq_matrix))
        ).map_err(|e| Error::VaApi(e.to_string()))?;

        // Begin picture ONCE for the entire frame
        let mut picture = Picture::<PictureNew, Rc<Surface<DmaBufSurfaceDescriptor>>>::new(timestamp, Rc::clone(&self.context), Rc::clone(&surface));
        picture.add_buffer(pic_param_buf);
        picture.add_buffer(iq_buf);

        // Add all slice buffers BEFORE begin (typestate requires it)
        // VAAPI processes SliceParameter+SliceData pairs in order during render()
        for (slice_info, (ref_pic_list_0, ref_pic_list_1)) in slices.iter().zip(slice_ref_lists.into_iter()) {
            let slice_header_opt = slice_info.slice_header.as_ref();

            // Slice data buffer INCLUDES the 1-byte NAL header. Per the VA-API
            // spec, slice_data_bit_offset is relative to and includes the NAL
            // unit byte, so it must be the slice header size plus 8 (the NAL
            // header). The iHD driver uses this bit offset precisely (verified:
            // an IDR slice with a 22-bit header decodes pixel-perfect only at
            // offset 30; offsets 26-29/31-32 all corrupt the output).
            let mut slice_data: Vec<u8> = slice_info.nal_data.clone();
            // The common parser extends a NAL with the leading 0x00 of a
            // following 4-byte start code (needed by NVDEC/cuvid). VAAPI
            // expects the exact NAL bytes (FFmpeg sends slice_data_size
            // without that byte). A valid H.264 NAL never ends in 0x00
            // (rbsp_stop_one_bit), so a trailing zero is always the extra
            // byte — strip it.
            if slice_data.last() == Some(&0) {
                slice_data.pop();
            }
            // TEMP EXPERIMENT: zero out slice data to test if driver reads it
            if std::env::var("EXP_ZERO_SLICE").is_ok() {
                for b in slice_data.iter_mut() { *b = 0; }
            }
            let hbs_debug = slice_header_opt
                .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.header_bit_size), _ => None });
            let mut slice_data_bit_offset = hbs_debug.map(|h| h + 8).unwrap_or(0);
            // TEMP EXPERIMENT: override bit offset
            if let Ok(v) = std::env::var("EXP_BIT_OFF") {
                if let Ok(n) = v.parse::<u16>() { slice_data_bit_offset = n; }
            }
            // TEMP EXPERIMENT: override for non-IDR slices only (IDR keeps hbs+8)
            if let Ok(v) = std::env::var("EXP_BIT_OFF_NONIDR") {
                let is_idr = slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.nal_unit_type), _ => None })
                    == Some(5);
                if !is_idr {
                    if let Ok(n) = v.parse::<u16>() { slice_data_bit_offset = n; }
                }
            }
            if std::env::var("DBG_H264").is_ok() {
                eprintln!("[DBG-HBS] fn={} header_bit_size={:?} nal_len={}",
                    frame_num, hbs_debug, slice_info.nal_data.len());
            }

            if std::env::var("DBG_H264").is_ok() {
                let (st_, qp_d, fmb) = slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some((h.slice_type % 5, h.slice_qp_delta, h.first_mb_in_slice)), _ => None })
                    .unwrap_or((0, 0, 0));
                eprintln!("[DBG-SLICE] sid={} fn={} st={} pic_init_qp={} qp_delta={} first_mb={} data_len={} bit_off={} l0={} l1={} first8={:02x?}",
                    surface_id, frame_num, st_, pps.pic_init_qp_minus26, qp_d, fmb,
                    slice_data.len(), slice_data_bit_offset,
                    num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1,
                    &slice_data[..slice_data.len().min(8)]);
            }

            // Build slice parameter buffer for this slice

             // num_ref_idx_lX_active_minus1 are absent from I-slice headers
             // (H.264 7.4.3); FFmpeg passes 0 for them. The parser leaves them
             // at the PPS default, which the Xe driver does not ignore for I
             // slices, so force 0 here.
             let eff_slice_type = slice_header_opt
                 .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.slice_type % 5), _ => None })
                 .unwrap_or(slice_type);
             let eff_ref_l0 = if eff_slice_type == 2 { 0 } else { num_ref_idx_l0_active_minus1 };
             let eff_ref_l1 = if eff_slice_type == 2 { 0 } else { num_ref_idx_l1_active_minus1 };

             // Replicate FFmpeg's pwt state (h264_slice.c / h264_parse.c):
             // - I slices: pwt stays at its initial values (denom 0, empty
             //   tables — ref_count is 0).
             // - P/SP with weighted_pred_flag, or B with
             //   weighted_bipred_idc==1: explicit pred_weight_table from the
             //   bitstream (our parser already fills inferred defaults per
             //   FF convention; note the denoms hold the raw minus1 value,
             //   same as FF's pwt).
             // - B without an explicit table (implicit, idc==2): FF sets the
             //   denoms to 5 and the tables to 1 << 5 = 32.
             // - P/SP without a table: denoms stay 0, tables filled with
             //   1 << 0 = 1. (iHD ignores the arrays while the per-list flag
             //   is 0, but match FF anyway.)
             // slice_type after %5: 0=P, 1=B, 2=I, 3=SP, 4=SI.
             let st_va = slice_header_opt
                 .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.slice_type % 5), _ => None })
                 .unwrap_or(2);
             let has_pw_table = slice_header_opt
                 .and_then(|sh| match sh {
                     SliceHeader::H264(h) => Some(
                         ((h.slice_type % 5 == 0 || h.slice_type % 5 == 3) && pps.weighted_pred_flag)
                             || (h.slice_type % 5 == 1 && pps.weighted_bipred_idc == 1),
                     ),
                     _ => None,
                 })
                 .unwrap_or(false);
             let implicit_b = st_va == 1 && !has_pw_table;
             let n_refs0 = if st_va == 2 { 0 } else { eff_ref_l0 as usize + 1 };
             let n_refs1 = if st_va != 1 { 0 } else { eff_ref_l1 as usize + 1 };
             let (luma_log2_weight_denom, chroma_log2_weight_denom, luma_weight_l0_flag, luma_weight_l0, luma_offset_l0, chroma_weight_l0_flag, chroma_weight_l0, chroma_offset_l0, luma_weight_l1_flag, luma_weight_l1, luma_offset_l1, chroma_weight_l1_flag, chroma_weight_l1, chroma_offset_l1) = if has_pw_table {
                 let h = match slice_header_opt { Some(SliceHeader::H264(h)) => h, _ => unreachable!() };
                 (h.luma_log2_weight_denom, h.chroma_log2_weight_denom, h.luma_weight_l0_flag, h.luma_weight_l0, h.luma_offset_l0, h.chroma_weight_l0_flag, h.chroma_weight_l0, h.chroma_offset_l0, h.luma_weight_l1_flag, h.luma_weight_l1, h.luma_offset_l1, h.chroma_weight_l1_flag, h.chroma_weight_l1, h.chroma_offset_l1)
             } else if st_va == 2 {
                 (0, 0, 0, [0i16; 32], [0i16; 32], 0, [[0i16; 2]; 32], [[0i16; 2]; 32], 0, [0i16; 32], [0i16; 32], 0, [[0i16; 2]; 32], [[0i16; 2]; 32])
             } else {
                 let denom = if implicit_b { 5 } else { 0 };
                 let luma_def = 1i16 << denom;
                 let chroma_def = if sps.chroma_format_idc != 0 { 1i16 << denom } else { 0 };
                 let mut luma_weight_l0 = [0i16; 32];
                 let mut chroma_weight_l0 = [[0i16; 2]; 32];
                 let mut luma_weight_l1 = [0i16; 32];
                 let mut chroma_weight_l1 = [[0i16; 2]; 32];
                 for i in 0..n_refs0 {
                     luma_weight_l0[i] = luma_def;
                     chroma_weight_l0[i][0] = chroma_def;
                     chroma_weight_l0[i][1] = chroma_def;
                 }
                 for i in 0..n_refs1 {
                     luma_weight_l1[i] = luma_def;
                     chroma_weight_l1[i][0] = chroma_def;
                     chroma_weight_l1[i][1] = chroma_def;
                 }
                 (denom, denom, 0, luma_weight_l0, [0i16; 32], 0, chroma_weight_l0, [[0i16; 2]; 32], 0, luma_weight_l1, [0i16; 32], 0, chroma_weight_l1, [[0i16; 2]; 32])
             };

             let slice_param = SliceParameterBufferH264::new(
                  slice_data.len() as u32,
                  0,
                  VA_SLICE_DATA_FLAG_ALL,
                  slice_data_bit_offset,
                 slice_header_opt
                     .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.first_mb_in_slice as u16), _ => None })
                     .unwrap_or(0),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h264_slice_type_to_vaapi(h.slice_type)), _ => None })
                    .unwrap_or(0),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.direct_spatial_mv_pred_flag as u8), _ => None })
                    .unwrap_or(0),
                eff_ref_l0 as u8,
                eff_ref_l1 as u8,
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.cabac_init_idc), _ => None })
                    .unwrap_or(0),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.slice_qp_delta as i8), _ => None })
                    .unwrap_or(0),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.disable_deblocking_filter_idc as u8), _ => None })
                    .unwrap_or(0),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.slice_alpha_c0_offset_div2 as i8), _ => None })
                    .unwrap_or(0),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.slice_beta_offset_div2 as i8), _ => None })
                    .unwrap_or(0),
                ref_pic_list_0,
                ref_pic_list_1,
                luma_log2_weight_denom,
                chroma_log2_weight_denom,
                luma_weight_l0_flag,
                luma_weight_l0,
                luma_offset_l0,
                chroma_weight_l0_flag,
                chroma_weight_l0,
                chroma_offset_l0,
                luma_weight_l1_flag,
                luma_weight_l1,
                luma_offset_l1,
                chroma_weight_l1_flag,
                chroma_weight_l1,
                chroma_offset_l1,
            );

            if std::env::var("VACC_VA_DUMP").is_ok() {
                dump_va_slice(&slice_param);
            }

            let slice_param_buf = self.context.create_buffer(
                BufferType::SliceParameter(SliceParameter::H264(slice_param))
            ).map_err(|e| Error::VaApi(e.to_string()))?;

            let slice_data_buf = self.context.create_buffer(
                BufferType::SliceData(slice_data)
            ).map_err(|e| Error::VaApi(e.to_string()))?;

            picture.add_buffer(slice_param_buf);
            picture.add_buffer(slice_data_buf);
        }

        // Now begin and render all slices in one call
        let picture = picture
            .begin()
            .map_err(|e| Error::VaApi(e.to_string()))?;

        // Single render call processes all slice buffers
        let picture: Picture<PictureRender, Rc<Surface<DmaBufSurfaceDescriptor>>> = picture
            .render()
            .map_err(|e| Error::VaApi(e.to_string()))?;

        // End picture ONCE after all slices
        let picture: Picture<PictureEnd, Rc<Surface<DmaBufSurfaceDescriptor>>> = picture
            .end()
            .map_err(|e| Error::VaApi(e.to_string()))?;

        // Sync to ensure completion
        let _synced: Picture<PictureSync, Rc<Surface<DmaBufSurfaceDescriptor>>> = picture
            .sync()
             .map_err(|e| Error::VaApi(e.0.to_string()))?;

        // Mark surface as ready
        self.surface_pool.mark_ready(surface_idx);

        // Explicitly sync the surface before reading (some drivers need this)
        surface.sync().map_err(|e| Error::VaApi(e.to_string()))?;

        // Read pixel data from the decoded surface
        let pixel_data = read_surface_pixels(
            &surface,
            self.stream.width,
            self.stream.height,
            self.stream.display_width,
            self.stream.display_height,
        )?;

        // Commit the current picture to the common DPB. The reference marking
        // process (MMCO / sliding window) runs post-decode (spec 8.2.5; FFmpeg
        // applies it in field_end), so prepare_current applies the marking and
        // returns the slot the picture is stored into; commit_current stores it
        // and runs the display logic. Each picture must land on its own surface
        // (slot_surfaces), otherwise the driver's internal reference state
        // (keyed by surface) is corrupted by reusing one destination surface.
        let slot = ctx.dpb.prepare_current();
        for (i, s) in ctx.dpb.slots.iter().enumerate() {
            if s.state == 0 {
                ctx.slot_surfaces[i] = None;
            }
        }
        ctx.slot_surfaces[slot] = Some(surface_idx);
        ctx.dpb.commit_current(slot);

        // Compute a global display-order key for B-frame reordering.
        // POC resets to 0 at each IDR, so combine it with the GOP index to keep the
        // key monotonic across the whole stream.
        if is_idr && self.frame_count > 0 {
            self.gop_count += 1;
        }
        let display_poc = top_field_order_cnt;
        let key = self.gop_count as i64 * 1_000_000 + display_poc as i64;
        // Track the highest GOP index decoded. A frame's GOP is complete only once
        // a newer GOP has been decoded, so the watermark is a GOP index, not a
        // display-order key: within a GOP, B-frames may be decoded in non-monotonic
        // POC order (e.g. IBBBP decodes B(poc4) before B(poc2)).
        self.reorder_watermark = self.reorder_watermark.max(self.gop_count as i64);
        self.pending_key = key;

        // Create decoded frame
        let mut frame = DecodedFrame::new(
            self.frame_count,
            timestamp as i64,
            self.stream.display_width,
            self.stream.display_height,
            false,
        );
        frame.pixel_data = pixel_data;

        self.frame_count += 1;
        Ok(Some(frame))
    }

}

impl Decoder for VaapiDecoder {
    type Error = Error;

    fn new(data: Vec<u8>) -> Result<Self> {
        Self::new(data)
    }

    fn new_with_format(
        _data: Vec<u8>,
        _codec: CoreVideoCodec,
        _format: &VideoFormat,
    ) -> Result<Self> {
        Err(Error::DecoderInit("new_with_format not yet implemented".to_string()))
    }

    fn info(&self) -> DecoderInfo {
        // Derive bit depth, chroma subsampling, and profile from SPS when available
        let (chroma_subsampling, luma_bit_depth, chroma_bit_depth, profile_idc) =
            if let Some(ref sps) = self.stream.sps {
                let chroma_subsampling = match sps.chroma_format_idc {
                    0 => ChromaSubsampling::Monochrome,
                    1 => ChromaSubsampling::_420,
                    2 => ChromaSubsampling::_422,
                    3 => ChromaSubsampling::_444,
                    _ => ChromaSubsampling::_420,
                };
                let luma_bit_depth = match 8 + sps.bit_depth_luma_minus8 {
                    8 => ComponentBitDepth::Bit8,
                    10 => ComponentBitDepth::Bit10,
                    12 => ComponentBitDepth::Bit12,
                    _ => ComponentBitDepth::Bit8,
                };
                let chroma_bit_depth = match 8 + sps.bit_depth_chroma_minus8 {
                    8 => ComponentBitDepth::Bit8,
                    10 => ComponentBitDepth::Bit10,
                    12 => ComponentBitDepth::Bit12,
                    _ => ComponentBitDepth::Bit8,
                };
                let profile_idc = Some(sps.profile_idc as u32);
                (chroma_subsampling, luma_bit_depth, chroma_bit_depth, profile_idc)
            } else {
                (
                    ChromaSubsampling::_420,
                    ComponentBitDepth::Bit8,
                    ComponentBitDepth::Bit8,
                    None,
                )
            };

        DecoderInfo {
            backend: "vaapi".to_string(),
            codec: self.stream.codec,
            coded_size: Extent2D::new(self.stream.width, self.stream.height),
            display_size: Extent2D::new(self.stream.display_width, self.stream.display_height),
            chroma_subsampling,
            luma_bit_depth,
            chroma_bit_depth,
            profile_idc,
            dpb_slots: self.surface_pool.entries.len() as u32,
        }
    }

    fn submit(&mut self, data: &[u8]) -> Result<()> {
        // If we've consumed all pending data, append new data
        // Otherwise, insert new data after consumed portion
        if self.parse_offset >= self.pending_data.len() {
            self.pending_data.clear();
            self.parse_offset = 0;
        } else {
            // Keep unconsumed data
            let unconsumed = self.pending_data[self.parse_offset..].to_vec();
            self.pending_data = unconsumed;
            self.parse_offset = 0;
        }
        self.pending_data.extend_from_slice(data);
        Ok(())
    }

    fn decode(&mut self) -> Result<Option<DecodedFrame>> {
        loop {
            // 1. Emit the front of the reorder buffer if it is in display order.
            //    A frame is safe to emit once a newer GOP has been decoded (the
            //    watermark tracks the highest GOP index seen), which guarantees all
            //    frames of the front frame's GOP have been decoded, or once the
            //    stream is exhausted.
            if let Some(&(front_key, _)) = self.pending_frames.front() {
                let exhausted = self.parse_offset >= self.pending_data.len();
                let front_gop = front_key / 1_000_000;
                if exhausted || front_gop < self.reorder_watermark {
                    return Ok(Some(self.pending_frames.pop_front().unwrap().1));
                }
            }

            // 2. No frame ready to emit; decode another frame and buffer it.
            if self.parse_offset >= self.pending_data.len() {
                return Ok(None);
            }

            if self.stream.codec == CoreVideoCodec::DecodeH264 {
                let offset_before = self.parse_offset;
                match self.decode_h264_pending()? {
                    Some(frame) => {
                        // Insert into the reorder buffer in display-order (key) order.
                        let key = self.pending_key;
                        let pos = self.pending_frames
                            .iter()
                            .position(|(k, _)| *k > key)
                            .unwrap_or(self.pending_frames.len());
                        self.pending_frames.insert(pos, (key, frame));
                        continue; // Loop back to try emitting.
                    }
                    None => {
                        // No frame decoded. If we made no progress and nothing is
                        // buffered, stop to avoid spinning.
                        if self.parse_offset == offset_before && self.pending_frames.is_empty() {
                            return Ok(None);
                        }
                        continue;
                    }
                }
            }

            // Fallback: return placeholder frame for other codecs (no reordering).
            let frame = DecodedFrame::new(
                self.frame_count,
                self.frame_count as i64 * 33_333,
                self.stream.display_width,
                self.stream.display_height,
                false,
            );
            self.frame_count += 1;
            self.pending_data.clear();
            return Ok(Some(frame));
        }
    }

    fn flush(&mut self) -> Result<Vec<DecodedFrame>> {
        // Sync all Pending surfaces to make them Ready
        for i in 0..self.surface_pool.entries.len() {
            if let SurfaceState::Pending(_) = self.surface_pool.entries[i].state {
                self.surface_pool.sync_surface(i)?;
                self.surface_pool.mark_ready(i);
            }
        }

        // Free all surfaces so they can be reused
        for i in 0..self.surface_pool.entries.len() {
            self.surface_pool.free(i);
        }

        // Clear DPB state
        if let Some(ctx) = self.h264_ctx.as_mut() {
            ctx.dpb.invalidate_all();
            ctx.poc_calc.reset();
            ctx.curr_poc = 0;
            for s in ctx.slot_surfaces.iter_mut() {
                *s = None;
            }
        }

        // Clear pending data
        self.pending_data.clear();

        // Return any buffered frames in display order (key, frame) -> frame
        let frames = self.pending_frames.drain(..).map(|(_, f)| f).collect();
        Ok(frames)
    }

    fn reset(&mut self) -> Result<()> {
        self.pending_data.clear();
        self.parse_offset = 0;
        self.pending_frames.clear();
        self.frame_count = 0;
        self.reorder_watermark = i64::MIN;
        self.gop_count = 0;
        self.pending_key = 0;

        // Free all surfaces
        for i in 0..self.surface_pool.entries.len() {
            self.surface_pool.free(i);
        }

        // Reset codec context
        if let Some(ctx) = self.h264_ctx.as_mut() {
            ctx.dpb.invalidate_all();
            ctx.poc_calc.reset();
            ctx.curr_poc = 0;
            for s in ctx.slot_surfaces.iter_mut() {
                *s = None;
            }
        }

        // Reset parser state
        if let Some(ref mut parser) = self.parser {
            parser.reset();
        }

        Ok(())
    }
}

impl VaapiDecoder {
    /// Process pending H.264 data using the parser with incremental parsing.
    /// Collects all slices for a single frame before decoding.
    fn decode_h264_pending(&mut self) -> Result<Option<DecodedFrame>> {
        let parser = self.parser.as_mut()
            .ok_or_else(|| Error::InvalidState("H264 parser not initialized".to_string()))?;
        let ctx = self.h264_ctx.as_mut()
            .ok_or_else(|| Error::InvalidState("H264 context not initialized".to_string()))?;

        // If no more data to parse, return None
        if self.parse_offset >= self.pending_data.len() {
            return Ok(None);
        }

        // Loop until we find slices or run out of data
        loop {
            // Pass remaining data from parse_offset to parser
            let remaining = &self.pending_data[self.parse_offset..];
            let packet = BitstreamPacket::new(remaining.to_vec());

            match parser.parse(&packet) {
                Ok(ParseResult::ParameterSet { sps: Some(sps), pps: Some(pps), .. }) => {
                    let sps = sps.downcast_ref::<H264Sps>()
                        .ok_or_else(|| Error::DecoderInit("Invalid SPS type".to_string()))?;
                    let pps = pps.downcast_ref::<H264Pps>()
                        .ok_or_else(|| Error::DecoderInit("Invalid PPS type".to_string()))?;
                    self.stream.sps = Some(sps.clone());
                    self.stream.pps = Some(pps.clone());
                    ctx.max_frame_num = sps.max_frame_num;
                    ctx.dpb.max_frame_num = sps.max_frame_num;
                    ctx.dpb.num_ref_frames = sps.max_num_ref_frames.min(ctx.slot_surfaces.len() as u32).max(1);
                    if std::env::var("DBG_H264").is_ok() {
                        eprintln!("[DBG] SPS log2_max_fn={} poc_type={} log2_max_poc_lsb={} max_poc_lsb={} max_ref_frames={} gaps={} frame_mbs_only={}",
                            sps.log2_max_frame_num_minus4, sps.pic_order_cnt_type, sps.log2_max_pic_order_cnt_lsb_minus4,
                            sps.max_pic_order_cnt_lsb, sps.max_num_ref_frames, sps.gaps_in_frame_num_value_allowed_flag, sps.frame_mbs_only_flag);
                    }
                    ctx.poc_calc.reset();
                    ctx.curr_poc = 0;
                    // Continue loop to find slices
                    continue;
                }
                Ok(ParseResult::ParameterSet { sps: Some(sps), .. }) => {
                    let sps = sps.downcast_ref::<H264Sps>()
                        .ok_or_else(|| Error::DecoderInit("Invalid SPS type".to_string()))?;
                    self.stream.sps = Some(sps.clone());
                    ctx.max_frame_num = sps.max_frame_num;
                    ctx.dpb.max_frame_num = sps.max_frame_num;
                    ctx.dpb.num_ref_frames = sps.max_num_ref_frames.min(ctx.slot_surfaces.len() as u32).max(1);
                    ctx.poc_calc.reset();
                    ctx.curr_poc = 0;
                    // Continue loop to find slices
                    continue;
                }
                Ok(ParseResult::ParameterSet { pps: Some(pps), .. }) => {
                    let pps = pps.downcast_ref::<H264Pps>()
                        .ok_or_else(|| Error::DecoderInit("Invalid PPS type".to_string()))?;
                    self.stream.pps = Some(pps.clone());
                    // Continue loop to find slices
                    continue;
                }
                Ok(ParseResult::ParameterSet { .. }) => {
                    // Continue loop to find slices
                    continue;
                }
                Ok(ParseResult::Slice { slices: parser_slices, bytes_consumed }) => {
                    if parser_slices.is_empty() {
                        return Ok(None);
                    }

                    // Get first slice header for frame-level parameters
                    let first_slice = &parser_slices[0];
                    let first_slice_header = first_slice.slice_header.clone();

                    // Get frame_num from first slice header
                    let frame_num = if let Some(vk_video_parser::SliceHeader::H264(slh)) = &first_slice_header {
                        // Skip redundant slices: redundant_pic_cnt > 0 means this is a duplicate
                        // slice for error resilience, representing the same picture as a prior slice.
                        if slh.redundant_pic_cnt > 0 {
                            self.parse_offset += bytes_consumed;
                            return Ok(None);
                        }
                        slh.frame_num
                    } else {
                        0
                    };

                    // Calculate POC for this frame using the common PocCalculator
                    // (ONE POC implementation across backends). Called once per
                    // picture in decode order. FrameNum wraparound is handled by
                    // the common DPB (refresh_frame_num_wrap in picture_start).
                    let sps = self.stream.sps.as_ref()
                        .expect("SPS should be available for H264 slice");
                    if let Some(vk_video_parser::SliceHeader::H264(slh)) = &first_slice_header {
                        let is_ref = slh.nal_ref_idc != 0;
                        ctx.curr_poc = ctx.poc_calc.calculate(sps, slh, is_ref);
                    }

                    if std::env::var("DBG_H264").is_ok() {
                        let (st, mmco) = first_slice_header.as_ref().and_then(|sh| match sh {
                            SliceHeader::H264(h) => Some((h.slice_type % 5, h.dec_ref_pic_marking.iter().map(|e| (e.memory_management_control_operation, e.value)).collect::<Vec<_>>())),
                            _ => None
                        }).unwrap_or((2, vec![]));
                        eprintln!("[DBG] fn={} poc={} st={} mmco={:?} dpb_refs={}",
                            frame_num, ctx.curr_poc, st, mmco,
                            ctx.dpb.get_references().len());
                        for (si, se) in parser_slices.iter().enumerate() {
                            if let Some(SliceHeader::H264(h)) = &se.slice_header {
                                eprintln!("[DBG-SL] pic={si} fn={} poc_lsb={} hbs={} mmco={:?} rplm0={:?} nal0={:02x} len={}",
                                    h.frame_num, h.pic_order_cnt_lsb, h.header_bit_size,
                                    h.dec_ref_pic_marking.iter().map(|e| (e.memory_management_control_operation, e.value)).collect::<Vec<_>>(),
                                    h.ref_pic_list_modification_l0.iter().map(|e| (e.op, e.difference)).collect::<Vec<_>>(),
                                    se.nal_data.first().copied().unwrap_or(0), se.nal_data.len());
                            }
                        }
                    }

                    // Convert parser's SliceEntry to our H264SliceInfo
                    let slices: Vec<H264SliceInfo> = parser_slices.into_iter().map(|entry| {
                        H264SliceInfo {
                            nal_data: entry.nal_data,
                            slice_header: entry.slice_header,
                        }
                    }).collect();

                    // Advance offset by bytes consumed (includes start codes)
                    self.parse_offset += bytes_consumed;
                    if std::env::var("DBG_H264").is_ok() {
                        eprintln!("[DBG-OFFSET] fn={} bytes_consumed={} parse_offset={} remaining={}", frame_num, bytes_consumed, self.parse_offset, self.pending_data.len() - self.parse_offset);
                    }

                    // Decode the complete frame with all its slices
                    let timestamp = self.frame_count as u64 * 33_333;
                    return self.decode_h264_frame_multi_slice(&slices, timestamp);
                }
                Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => {
                    self.parse_offset = self.pending_data.len();
                    return Ok(None);
                }
                Err(e) => return Err(Error::Parser(e.to_string())),
            }
        }
    }
}

/// Read pixel data from a VA image (from derive_from or create_from).
///
/// The returned planes are cropped to the display size (top-left origin). The
/// underlying surface may be larger than the display size due to frame cropping
/// (H.264) or padding; only the top-left `display_width x display_height` region
/// is the visible picture.
fn read_from_image(
    image: Image,
    display_width: u32,
    display_height: u32,
) -> Result<Option<PixelData>> {
    let va_image = image.image();
    let data = image.as_ref();

    // Determine format from fourcc
    let fourcc = va_image.format.fourcc;
    let is_nv12 = fourcc == libva::VA_FOURCC_NV12;
    let format_str = if is_nv12 {
        "NV12".to_string()
    } else if fourcc == u32::from_ne_bytes(*b"YV12") {
        "YV12".to_string()
    } else if fourcc == u32::from_ne_bytes(*b"I420") {
        "I420".to_string()
    } else {
        return Err(Error::VaApi(format!("Unsupported image format: {:X}", fourcc)));
    };

    // Validate num_planes
    if va_image.num_planes < 2 {
        return Err(Error::VaApi(format!(
            "Unexpected num_planes={} for semi-planar format",
            va_image.num_planes
        )));
    }
    if !is_nv12 && va_image.num_planes < 3 {
        return Err(Error::VaApi(format!(
            "Unexpected num_planes={} for planar format {}",
            va_image.num_planes, format_str
        )));
    }

    // Copy data into owned buffer so we can drop the Image (which unmaps the surface)
    let buffer = data.to_vec();

    // Crop to the display size (top-left origin). The surface may be larger than
    // the display size due to frame cropping / padding.
    let out_width = display_width.min(va_image.width as u32) as usize;
    let out_height = display_height.min(va_image.height as u32) as usize;
    let uv_width = (out_width + 1) / 2;
    let uv_height = (out_height + 1) / 2;

    // Build plane descriptors from the copied buffer
    let y_offset = va_image.offsets[0] as usize;
    let u_offset = va_image.offsets[1] as usize;

    let y_plane = PixelPlane {
        data: unsafe { buffer.as_ptr().add(y_offset) },
        pitch: va_image.pitches[0] as usize,
        width: out_width,
        height: out_height,
    };

    let u_plane = PixelPlane {
        data: unsafe { buffer.as_ptr().add(u_offset) },
        pitch: va_image.pitches[1] as usize,
        width: uv_width,
        height: uv_height,
    };

    let v_plane = if !is_nv12 {
        let v_offset = va_image.offsets[2] as usize;
        Some(PixelPlane {
            data: unsafe { buffer.as_ptr().add(v_offset) },
            pitch: va_image.pitches[2] as usize,
            width: uv_width,
            height: uv_height,
        })
    } else {
        None
    };

    // Image is dropped here, unmapping the surface
    drop(image);

    Ok(Some(PixelData {
        format: format_str,
        y: y_plane,
        u: u_plane,
        v: v_plane,
        buffer,
    }))
}

/// Read pixel data from a VA surface after decode.
///
/// Strategy (in order):
/// 1. Image::create_from() (vaCreateImage + vaGetImage) - the NVIDIA NVDEC
///    driver's supported CPU-read path. The driver's nvGetImage syncs the
///    surface and cuMemcpy2D's the decoded frame from the backing CUDA array
///    into a host buffer. This is what FFmpeg uses (its vaDeriveImage probe
///    fails on this driver, so it falls back to exactly this path).
/// 2. Image::derive_from() - zero-copy; not supported by the NVIDIA driver
///    (returns VA_STATUS_ERROR_OPERATION_FAILED) but kept for other drivers.
///
/// Note: we deliberately do NOT use vaExportSurfaceHandle + mmap here. The
/// exported DMA-BUFs are block-linear tiled (a raw linear read is garbage) and
/// CPU-mmap of a freshly GPU-written discrete-GPU dmabuf blocks on the kernel
/// driver's implicit fence. vaGetImage avoids both problems.
fn read_surface_pixels(
    surface: &Surface<DmaBufSurfaceDescriptor>,
    width: u32,
    height: u32,
    display_width: u32,
    display_height: u32,
) -> Result<Option<PixelData>> {

    // Primary: vaCreateImage + vaGetImage (driver-supported CPU read).
    // Read the full coded size, then crop to the display size in read_from_image.
    let format = libva::VAImageFormat {
        fourcc: libva::VA_FOURCC_NV12,
        ..Default::default()
    };
    match Image::create_from(surface, format, (width, height), (width, height)) {
        Ok(image) => return read_from_image(image, display_width, display_height),
        Err(_) => {}
    }

    // Fallback: derive_from (zero-copy; unsupported on NVIDIA).
    match Image::derive_from(surface, (width, height)) {
        Ok(img) => {
            let fourcc = img.image().format.fourcc;
            if fourcc == libva::VA_FOURCC_NV12
                || fourcc == u32::from_ne_bytes(*b"YV12")
                || fourcc == u32::from_ne_bytes(*b"I420")
            {
                return read_from_image(img, display_width, display_height);
            }
        }
        Err(_) => {}
    }

    Err(Error::VaApi(
        "All surface read methods failed".to_string(),
    ))
}

/// Parse stream info from initial bitstream data.
fn parse_stream_info(display: &Display, data: &[u8]) -> Result<StreamInfo> {
    let codec = detect_codec(data);

    match codec {
        CoreVideoCodec::DecodeH264 => parse_h264_info(display, data),
        CoreVideoCodec::DecodeH265 => parse_h265_info(display, data),
        CoreVideoCodec::DecodeVp9 => parse_vp9_info(display, data),
        _ => Err(Error::CodecNotSupported(format!("Unsupported codec: {:?}", codec))),
    }
}

/// Detect codec from bitstream data.
fn detect_codec(data: &[u8]) -> CoreVideoCodec {
    // Check for IVF container (VP9)
    if data.len() >= 32 && data[0..4] == *b"DKIF" {
        return CoreVideoCodec::DecodeVp9;
    }

    // Check for VP9 frame marker
    for i in 0..data.len().min(256) {
        if data[i] == 0 {
            continue;
        }
        if (data[i] & 0xC0) == 0x80 {
            return CoreVideoCodec::DecodeVp9;
        }
        break;
    }

    // Check NAL types
    for i in 0..data.len().min(4096) {
        let start = if i + 4 <= data.len() && data[i..i+4] == [0x00, 0x00, 0x00, 0x01] {
            i + 4
        } else if i + 3 <= data.len() && data[i..i+3] == [0x00, 0x00, 0x01] {
            i + 3
        } else {
            continue;
        };
        if start >= data.len() {
            continue;
        }
        let nal_type = data[start] & 0x1F;
        // H.265 VPS/SPS/PPS
        if nal_type == 32 || nal_type == 33 || nal_type == 34 {
            return CoreVideoCodec::DecodeH265;
        }
        // H.264 SPS/PPS
        if nal_type == 7 || nal_type == 8 {
            return CoreVideoCodec::DecodeH264;
        }
    }

    // Default to H.264
    CoreVideoCodec::DecodeH264
}

/// Parse H.264 stream info.
fn parse_h264_info(display: &Display, data: &[u8]) -> Result<StreamInfo> {
    let mut parser = H264Parser::new();
    parser.init(&DetectedVideoFormat::new(CoreVideoCodec::DecodeH264))
        .map_err(|e| Error::Parser(e.to_string()))?;

    let packet = BitstreamPacket::new(data.to_vec());
    let mut width = 0u32;
    let mut height = 0u32;
    let mut display_width = 0u32;
    let mut display_height = 0u32;
    let mut max_dpb = 4u32;
    let mut profile = libva::VAProfile::VAProfileH264Main;
    let mut sps_opt: Option<H264Sps> = None;
    let mut pps_opt: Option<H264Pps> = None;

    // Parse the packet - may return ParameterSet or Slice (with SPS/PPS cached)
    match parser.parse(&packet) {
        Ok(ParseResult::ParameterSet { sps: Some(s), pps, .. }) => {
            if let Some(sps) = s.downcast_ref::<H264Sps>() {
                sps_opt = Some(sps.clone());
                if let Some(pps_box) = pps {
                    if let Some(pps_ref) = pps_box.downcast_ref::<H264Pps>() {
                        pps_opt = Some(pps_ref.clone());
                    }
                }
            }
        }
        Ok(ParseResult::Slice { .. }) => {
            // Slices found - SPS/PPS were parsed and cached
            if let Some(sps) = parser.active_sps() {
                sps_opt = Some(sps.clone());
            }
            if let Some(pps) = parser.active_pps() {
                pps_opt = Some(pps.clone());
            }
        }
        _ => {}
    }

    // Extract dimensions and other info from SPS
    if let Some(ref sps) = sps_opt {
        width = (sps.pic_width_in_mbs_minus1 as u32 + 1) * 16;
        height = if sps.frame_mbs_only_flag {
            (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
        } else {
            (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
        };
        max_dpb = sps.max_num_ref_frames as u32;

        // Calculate display dimensions with frame cropping (H.264 spec 7.4.2.1.1).
        // The frame_crop_*_offset values are expressed in units of SubWidthC /
        // SubHeightC luma samples (2 for 4:2:0, 1 for 4:4:4 / 4:0:0), NOT macroblocks.
        // e.g. crop_bottom=4 with 4:2:0 -> 4*2 = 8 luma rows cropped.
        if sps.frame_cropping_flag {
            let sub_width_c = if sps.chroma_format_idc == 1 || sps.chroma_format_idc == 2 { 2 } else { 1 };
            let sub_height_c = if sps.chroma_format_idc == 1 { 2 } else { 1 };

            let crop_left = sps.frame_crop_left_offset * sub_width_c;
            let crop_right = sps.frame_crop_right_offset * sub_width_c;
            let crop_top = sps.frame_crop_top_offset * sub_height_c;
            let crop_bottom = sps.frame_crop_bottom_offset * sub_height_c;

            display_width = if crop_left + crop_right < width {
                width - crop_left - crop_right
            } else {
                width
            };
            display_height = if crop_top + crop_bottom < height {
                height - crop_top - crop_bottom
            } else {
                height
            };
        } else {
            display_width = width;
            display_height = height;
        }

        // Determine profile from profile_idc. The iHD (Intel) driver does NOT
        // support VAProfileH264Baseline, so map baseline (66) to
        // VAProfileH264ConstrainedBaseline (a superset of baseline that the
        // driver does support). This matches the caps query in device.rs.
        profile = match sps.profile_idc {
            66 => libva::VAProfile::VAProfileH264ConstrainedBaseline,
            77 => libva::VAProfile::VAProfileH264Main,
            88 => libva::VAProfile::VAProfileH264Main,
            100 => libva::VAProfile::VAProfileH264High,
            110 => libva::VAProfile::VAProfileH264High10,
            122 | 244 => libva::VAProfile::VAProfileH264High,
            _ => libva::VAProfile::VAProfileH264Main,
        };
    }

    // Fall back to a supported profile if the preferred one is unavailable on
    // this driver (e.g. some drivers lack High; iHD lacks Baseline).
    let preferred = profile;
    let supported = |p: libva::VAProfile::Type| {
        display
            .query_config_entrypoints(p)
            .map(|e| e.contains(&libva::VAEntrypoint::VAEntrypointVLD))
            .unwrap_or(false)
    };
    if !supported(preferred) {
        for p in [
            libva::VAProfile::VAProfileH264ConstrainedBaseline,
            libva::VAProfile::VAProfileH264Main,
            libva::VAProfile::VAProfileH264High,
        ] {
            if supported(p) {
                profile = p;
                break;
            }
        }
    }

    if width == 0 || height == 0 {
        return Err(Error::DecoderInit("Failed to parse H.264 dimensions".to_string()));
    }

    let rt_format = if let Some(ref sps) = sps_opt {
        let bit_depth = 8 + sps.bit_depth_luma_minus8;
        let chroma_fmt = sps.chroma_format_idc;

        match (bit_depth, chroma_fmt) {
            (8, 0) | (8, 1) => libva::VA_RT_FORMAT_YUV420,
            (8, 2) => libva::VA_RT_FORMAT_YUV422,
            (8, 3) => libva::VA_RT_FORMAT_YUV444,
            (10, 0) | (10, 1) => libva::VA_RT_FORMAT_YUV420_10,
            (10, 2) => libva::VA_RT_FORMAT_YUV422_10,
            (10, 3) => libva::VA_RT_FORMAT_YUV444_10,
            (12, 0) | (12, 1) => libva::VA_RT_FORMAT_YUV420_12,
            (12, 2) => libva::VA_RT_FORMAT_YUV422_12,
            (12, 3) => libva::VA_RT_FORMAT_YUV444_12,
            _ => libva::VA_RT_FORMAT_YUV420,
        }
    } else {
        libva::VA_RT_FORMAT_YUV420
    };

    Ok(StreamInfo {
        codec: CoreVideoCodec::DecodeH264,
        profile,
        width,
        height,
        display_width,
        display_height,
        max_dpb: max_dpb.min(16).max(1),
        rt_format,
        sps: sps_opt,
        pps: pps_opt,
    })
}

/// Parse H.265 stream info.
fn parse_h265_info(_display: &Display, data: &[u8]) -> Result<StreamInfo> {
    use vk_video_parser::{bitstream::BitstreamPacket, h265::H265Parser, DetectedVideoFormat, ParseResult, VideoParser};

    let mut parser = H265Parser::new();
    parser.init(&DetectedVideoFormat::new(CoreVideoCodec::DecodeH265))
        .map_err(|e| Error::Parser(e.to_string()))?;

    let packet = BitstreamPacket::new(data.to_vec());
    let mut width = 0u32;
    let mut height = 0u32;
    let mut max_dpb = 4u32;

    if let Ok(ParseResult::ParameterSet { sps: Some(s), .. }) = parser.parse(&packet) {
        if let Some(sps) = s.downcast_ref::<vk_video_core::picture::H265Sps>() {
            width = ((sps.pic_width_in_luma_samples as u32) + 15) & !15;
            height = ((sps.pic_height_in_luma_samples as u32) + 15) & !15;
            max_dpb = sps.max_num_ref_frames as u32;
        }
    }

    if width == 0 || height == 0 {
        return Err(Error::DecoderInit("Failed to parse H.265 dimensions".to_string()));
    }

    Ok(StreamInfo {
        codec: CoreVideoCodec::DecodeH265,
        profile: libva::VAProfile::VAProfileHEVCMain,
        width,
        height,
        display_width: width,
        display_height: height,
        max_dpb: max_dpb.min(16).max(1),
        rt_format: libva::VA_RT_FORMAT_YUV420,
        sps: None,
        pps: None,
    })
}

/// Parse VP9 stream info.
fn parse_vp9_info(_display: &Display, data: &[u8]) -> Result<StreamInfo> {
    use vk_video_parser::{vp9::Vp9Parser, DetectedVideoFormat, VideoParser};

    let raw_frames = if data.len() >= 32 && data[0..4] == *b"DKIF" {
        if data.len() > 128 {
            vec![data[128..].to_vec()]
        } else {
            vec![data.to_vec()]
        }
    } else {
        vec![data.to_vec()]
    };

    if raw_frames.is_empty() {
        return Err(Error::DecoderInit("No VP9 frames found".to_string()));
    }

    let mut parser = Vp9Parser::new();
    parser.init(&DetectedVideoFormat::new(CoreVideoCodec::DecodeVp9))
        .map_err(|e| Error::Parser(e.to_string()))?;

    let parsed = parser.parse_frame(&raw_frames[0])
        .map_err(|e| Error::Parser(e.to_string()))?;

    let width = parsed.frame_width;
    let height = parsed.frame_height;

    if width == 0 || height == 0 {
        return Err(Error::DecoderInit("Failed to parse VP9 dimensions".to_string()));
    }

    Ok(StreamInfo {
        codec: CoreVideoCodec::DecodeVp9,
        profile: libva::VAProfile::VAProfileVP9Profile0,
        width,
        height,
        display_width: width,
        display_height: height,
        max_dpb: 8,
        rt_format: libva::VA_RT_FORMAT_YUV420,
        sps: None,
        pps: None,
    })
}

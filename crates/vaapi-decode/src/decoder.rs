//! VAAPI video decoder implementing the Decoder trait.
//!
//! Uses cros-libva's typestate Picture pattern for safe decode operations.
//! Supports H.264, H.265, VP9 decoding with proper buffer management.

use std::collections::VecDeque;
use std::rc::Rc;

use libva::{
    Buffer, BufferType, Config, Context, Display, IQMatrix, IQMatrixBufferH264,
    IQMatrixBufferHEVC, PictureParameter, PictureParameterBufferH264,
    PictureParameterBufferHEVC, PictureParameterBufferHEVCRext,
    PictureParameterBufferHEVCExtension, HevcRangeExtensionPicFields,
    PictureH264, PictureHEVC, H264SeqFields,
    H264PicFields, HevcPicFields, HevcSliceParsingFields, HevcLongSliceFlags,
    SliceParameter, SliceParameterBufferH264, SliceParameterBufferHEVC,
    SliceParameterBufferHEVCRext, SliceParameterBufferHEVCExtension, HevcSliceExtFlags,
    Picture, PictureNew, PictureEnd, PictureRender, PictureSync, Surface, Image,
    PictureParameterBufferVP9, SegmentParameterVP9, SliceParameterBufferVP9,
    VP9PicFields, VP9SegmentFlags,
};
use libva::VAProfile::Type as VAProfileType;
use libva::{
    VA_INVALID_ID, VA_SLICE_DATA_FLAG_ALL,
    VA_PICTURE_H264_INVALID, VA_PICTURE_H264_SHORT_TERM_REFERENCE,
    VA_PICTURE_H264_LONG_TERM_REFERENCE, VA_PICTURE_H264_TOP_FIELD,
    VA_PICTURE_H264_BOTTOM_FIELD,
    VA_PICTURE_HEVC_INVALID, VA_PICTURE_HEVC_LONG_TERM_REFERENCE,
    VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE, VA_PICTURE_HEVC_RPS_ST_CURR_AFTER,
    VA_PICTURE_HEVC_RPS_LT_CURR,
};

use vk_video_core::{
    codec::VideoCodec as CoreVideoCodec,
    decoder::{Decoder, DecoderInfo},
    frame::{DecodedFrame, PixelData, PixelPlane},
    format::{ChromaSubsampling, ComponentBitDepth, VideoFormat},
    session::Extent2D,
    picture::{H264Sps, H264Pps, H265Sps, H265Pps, Vp9FrameData},
};
use vk_video_parser::{
    bitstream::BitstreamPacket, h264::H264Parser, h265::H265Parser,
    h264_dpb::{H264Dpb, H264MmcoCommand, MARKING_LONG},
    h265_dpb::H265Dpb,
    h264_poc::PocCalculator,
    vp9::Vp9Parser, vp9_dpb::Vp9Dpb,
    DetectedVideoFormat, ParseResult, SliceHeader, VideoParser,
};

use super::{Error, Result};

use crate::vp9_qlookup::{VP9_AC_QLOOKUP, VP9_DC_QLOOKUP};

/// Custom surface memory descriptor that requests DRM_PRIME_2 memory type.
/// This is required for export_prime to work on NVIDIA GPUs.
#[derive(Clone, Copy, Default)]
struct DmaBufSurfaceDescriptor {
    /// 8-bit HEVC Rext 4:4:4 (Main444) on iHD: the driver only decodes into
    /// packed XYUV (VUYX) surfaces. FFmpeg creates them with
    /// VASurfaceAttribPixelFormat='XYUV' + MemoryType=VA; any other layout
    /// (444P, DRM prime) makes vaEndPicture fail with INVALID_PARAMETER.
    xyuv444: bool,
}

impl libva::SurfaceMemoryDescriptor for DmaBufSurfaceDescriptor {
    fn add_attrs(&mut self, attrs: &mut Vec<libva::VASurfaceAttrib>) -> Option<Box<dyn std::any::Any>> {
        if self.xyuv444 {
            attrs.push(libva::VASurfaceAttrib::new_pixel_format(u32::from_ne_bytes(*b"XYUV")));
            attrs.push(libva::VASurfaceAttrib::new_memory_type(libva::MemoryType::Va));
            return None;
        }
        // NVIDIA NVDEC requires surfaces to be allocated with DRM_PRIME_2 memory type
        // for vaExportSurfaceHandle to succeed. Without it the driver returns
        // VA_ERROR_INVALID_SURFACE ("invalid VASurfaceID") on export.
        attrs.push(libva::VASurfaceAttrib::new_memory_type(libva::MemoryType::DrmPrime2));
        None
    }
}

/// Map a VA runtime format to the DRM fourcc of the surface pixel format.
///
/// Only 8-bit 4:4:4 (profile 1) needs an explicit pixel-format attribute on
/// iHD: without it the driver allocates a surface its VP9 profile-1 decoder
/// rejects with VA_STATUS_INVALID_PARAM, and XYUV is the format FFmpeg's
/// VAAPI wrapper requests. For 4:2:0 (NV12/P010/P012) leaving the attribute
/// unset keeps the driver's default allocation, which decodes byte-exact;
/// forcing NV12 measurably changes the decoded output.
fn rt_format_to_fourcc(rt_format: u32) -> Option<u32> {
    match rt_format {
        libva::VA_RT_FORMAT_YUV444 => Some(0x56555958), // XYUV
        _ => None,
    }
}

const FOURCC_YV12: u32 = u32::from_ne_bytes(*b"YV12");
const FOURCC_I420: u32 = u32::from_ne_bytes(*b"I420");
const FOURCC_XYUV: u32 = u32::from_ne_bytes(*b"XYUV");
// Y410: 32-bit packed 10-bit 4:4:4. iHD's image format for 10-bit 4:4:4
// surfaces (HEVC Main444_10) — this is what FFmpeg's vaapi hwaccel requests
// for the same content. iHD's field order is U | Y<<10 | V<<20 (verified
// byte-for-byte against FF output).
const FOURCC_Y410: u32 = u32::from_ne_bytes(*b"Y410");

/// Candidate image fourccs to probe via `vaGetImage` for a render format:
/// the driver may expose a semi-planar or planar variant of the same format
/// class, so try them in preference order.
fn rt_format_candidates(rt_format: u32) -> &'static [u32] {
    match rt_format {
        libva::VA_RT_FORMAT_YUV420 => &[libva::VA_FOURCC_NV12, FOURCC_YV12, FOURCC_I420],
        libva::VA_RT_FORMAT_YUV420_10 => &[libva::VA_FOURCC_P016],
        // XYUV first: iHD stores Main444 surfaces in packed XYUV and derives
        // images in that layout; 444P is kept as a fallback for other drivers.
        libva::VA_RT_FORMAT_YUV444 => &[FOURCC_XYUV, libva::VA_FOURCC_444P],
        // 10-bit 4:4:4: iHD derives Y410 images (no 16-bit 444P/XYUV variant).
        libva::VA_RT_FORMAT_YUV444_10 => &[FOURCC_Y410],
        _ => &[libva::VA_FOURCC_NV12],
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
    /// VP9 bitstream profile (0/1/2); 0 for other codecs.
    vp9_profile: u8,
    /// Luma/chroma bit depth from the first VP9 frame; 8 for other codecs.
    vp9_bit_depth: u8,
    /// H.264 specific
    sps: Option<H264Sps>,
    pps: Option<H264Pps>,
    /// H.265 specific
    h265_sps: Option<H265Sps>,
    h265_pps: Option<H265Pps>,
}

/// H.265 decode context.
///
/// Uses the common decode-state foundation from `vk-video-parser`:
/// - `dpb`: the common `H265Dpb` manager (ONE DPB manager across backends).
/// - `slot_surfaces`: maps a common-DPB slot index to a VA surface-pool index.
struct H265Context {
    dpb: H265Dpb,
    /// DPB slot index -> surface pool index (None if the slot has no surface).
    slot_surfaces: Vec<Option<usize>>,
    /// POC of the current picture (from the slice header, decode order).
    curr_poc: i32,
}

/// Holds information about a single H.265 slice for multi-slice frame assembly
struct H265SliceInfo {
    nal_data: Vec<u8>,
    slice_header: Option<SliceHeader>,
}

/// VP9 decode context.
///
/// Uses the common decode-state foundation from `vk-video-parser`:
/// - `dpb`: the common `Vp9Dpb` manager (ONE DPB manager across backends).
/// - `slot_surfaces`: maps a common-DPB slot index to a VA surface-pool index.
struct Vp9Context {
    dpb: Vp9Dpb,
    /// DPB slot index -> surface pool index (None if the slot has no surface).
    slot_surfaces: Vec<Option<usize>>,
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
    vp9_ctx: Option<Vp9Context>,
    h265_ctx: Option<H265Context>,
    parser: Option<H264Parser>,
    vp9_parser: Option<Vp9Parser>,
    h265_parser: Option<H265Parser>,
    /// True if the input is an IVF container (packets start at offset 32).
    input_is_ivf: bool,
}

impl VaapiDecoder {
    /// Create a new VAAPI decoder from initial bitstream data.
    pub fn new(data: Vec<u8>) -> Result<Self> {
        let display = Display::open()
            .ok_or_else(|| Error::DecoderInit("No VA display available".to_string()))?;

        // Parse stream to get codec and dimensions
        let stream = parse_stream_info(&display, &data)?;

        // IVF containers carry a 32-byte header before the first packet.
        let is_ivf = data.len() >= 32 && data[0..4] == *b"DKIF";

        // Create config with RT format attribute (like cros-codecs does).
        let cfg_attrs = vec![libva::VAConfigAttrib {
            type_: libva::VAConfigAttribType::VAConfigAttribRTFormat,
            value: stream.rt_format,
        }];
        let config = display.create_config(
            cfg_attrs,
            stream.profile,
            libva::VAEntrypoint::VAEntrypointVLD,
        ).map_err(|e| Error::DecoderInit(e.to_string()))?;

        // Create surfaces (DPB + extra for rendering). The pool must hold every
        // DPB reference plus the picture currently being decoded, so keep +4
        // slack beyond the SPS max_num_ref_frames.
        let num_surfaces = (stream.max_dpb as usize).max(4) + 4;
        // iHD HEVC Main444 (8-bit Rext 4:4:4) requires packed XYUV surfaces
        // (see DmaBufSurfaceDescriptor); all other streams keep the prime path.
        let xyuv444 = stream.h265_sps.as_ref().map_or(false, |sps| {
            sps.profile_idc >= 4 && sps.chroma_format_idc == 3 && sps.bit_depth_luma_minus8 == 0
        });
        let descriptors: Vec<DmaBufSurfaceDescriptor> =
            (0..num_surfaces).map(|_| DmaBufSurfaceDescriptor { xyuv444 }).collect();

         let surfaces = display.create_surfaces::<DmaBufSurfaceDescriptor>(
            stream.rt_format,
            rt_format_to_fourcc(stream.rt_format),
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

        // VP9 uses the common parser + DPB (ONE implementation across
        // backends). One slot per surface so a surface can always be mapped
        // to a slot.
        let (vp9_ctx, vp9_parser) = if stream.codec == CoreVideoCodec::DecodeVp9 {
            let num_slots = num_surfaces;
            let mut p = Vp9Parser::new();
            p.init(&DetectedVideoFormat::new(CoreVideoCodec::DecodeVp9))
                .map_err(|e| Error::Parser(e.to_string()))?;
            (
                Some(Vp9Context {
                    dpb: Vp9Dpb::new(num_slots as u32),
                    slot_surfaces: vec![None; num_slots],
                }),
                Some(p),
            )
        } else {
            (None, None)
        };

        let h265_ctx = if stream.codec == CoreVideoCodec::DecodeH265 {
            let sps = stream.h265_sps.as_ref()
                .ok_or_else(|| Error::DecoderInit("H265 SPS not available".to_string()))?;
            // One slot per surface so a surface can always be mapped to a slot.
            let num_slots = num_surfaces;
            let mut dpb = H265Dpb::new(num_slots);
            dpb.set_max_num_reorder_frames(sps.max_num_reorder_pics[0] as u32);
            Some(H265Context {
                dpb,
                slot_surfaces: vec![None; num_slots],
                curr_poc: 0,
            })
        } else {
            None
        };

        let h265_parser = if stream.codec == CoreVideoCodec::DecodeH265 {
            let mut p = H265Parser::new();
            p.init(&DetectedVideoFormat::new(CoreVideoCodec::DecodeH265))
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
            parse_offset: if is_ivf { 32 } else { 0 },
            frame_count: 0,
            pending_frames: VecDeque::new(),
            reorder_watermark: i64::MIN,
            gop_count: 0,
            pending_key: 0,
            h264_ctx,
            vp9_ctx,
            h265_ctx,
            parser,
            vp9_parser,
            h265_parser,
            input_is_ivf: is_ivf,
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
            rt_format_candidates(self.stream.rt_format),
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
        // Derive bit depth, chroma subsampling, and profile from the stream
        // metadata (VP9 frame header / H.264 SPS) when available.
        let (chroma_subsampling, luma_bit_depth, chroma_bit_depth, profile_idc) =
            if self.stream.codec == CoreVideoCodec::DecodeVp9 {
                let chroma = if self.stream.vp9_profile == 1 {
                    ChromaSubsampling::_444
                } else {
                    ChromaSubsampling::_420
                };
                let bd = match self.stream.vp9_bit_depth {
                    10 => ComponentBitDepth::Bit10,
                    12 => ComponentBitDepth::Bit12,
                    _ => ComponentBitDepth::Bit8,
                };
                (chroma, bd, bd, Some(self.stream.vp9_profile as u32))
            } else if let Some(ref sps) = self.stream.sps {
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

            let offset_before = self.parse_offset;

            if self.stream.codec == CoreVideoCodec::DecodeVp9 {
                // VP9 has no B-frames: display order equals decode order
                // (show-existing commands re-display in place), so frames are
                // emitted directly without a reorder buffer.
                match self.decode_vp9_pending()? {
                    Some(frame) => return Ok(Some(frame)),
                    None => {
                        if self.parse_offset == offset_before {
                            return Ok(None);
                        }
                        continue;
                    }
                }
            }

            // Dispatch to the codec-specific incremental decoder.
            let decoded = if self.stream.codec == CoreVideoCodec::DecodeH264 {
                self.decode_h264_pending()?
            } else if self.stream.codec == CoreVideoCodec::DecodeH265 {
                self.decode_h265_pending()?
            } else {
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
            };

            match decoded {
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
        if let Some(ctx) = self.vp9_ctx.as_mut() {
            ctx.dpb.reset();
            for s in ctx.slot_surfaces.iter_mut() {
                *s = None;
            }
        }
        if let Some(ctx) = self.h265_ctx.as_mut() {
            ctx.dpb.invalidate_all();
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
        if let Some(ctx) = self.vp9_ctx.as_mut() {
            ctx.dpb.reset();
            for s in ctx.slot_surfaces.iter_mut() {
                *s = None;
            }
        }
        if let Some(ctx) = self.h265_ctx.as_mut() {
            ctx.dpb.invalidate_all();
            ctx.curr_poc = 0;
            for s in ctx.slot_surfaces.iter_mut() {
                *s = None;
            }
        }

        // Reset parser state
        if let Some(ref mut parser) = self.parser {
            parser.reset();
        }
        if let Some(ref mut parser) = self.vp9_parser {
            parser.reset();
        }
        if let Some(ref mut parser) = self.h265_parser {
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
                    // NOTE: do NOT reset POC state here. Encoders (e.g. x264)
                    // re-send SPS/PPS before every keyframe; a mid-stream
                    // parameter set does not restart the POC sequence. POC
                    // state is only reset for IDR pictures (see Slice arm),
                    // per H.264 8.2.1.
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
                    // (no POC reset here; see SPS arm comment above)
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
                        // Per H.264 8.2.1 an IDR picture restarts the POC
                        // state (PicOrderCntMsb = 0). Non-IDR pictures —
                        // including CRAs, which do NOT clear the DPB — must
                        // continue the existing POC sequence even if the
                        // encoder re-sent SPS/PPS just before them.
                        if slh.nal_unit_type == 5 {
                            ctx.poc_calc.reset();
                        }
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

    /// Process pending H.265 data using the common parser with incremental
    /// parsing. Collects all slice segments for a single picture, then decodes.
    fn decode_h265_pending(&mut self) -> Result<Option<DecodedFrame>> {
        let parser = self.h265_parser.as_mut()
            .ok_or_else(|| Error::InvalidState("H265 parser not initialized".to_string()))?;
        let ctx = self.h265_ctx.as_mut()
            .ok_or_else(|| Error::InvalidState("H265 context not initialized".to_string()))?;

        if self.parse_offset >= self.pending_data.len() {
            return Ok(None);
        }

        loop {
            let remaining = &self.pending_data[self.parse_offset..];
            let packet = BitstreamPacket::new(remaining.to_vec());

            match parser.parse(&packet) {
                Ok(ParseResult::ParameterSet { sps: Some(s), pps, .. }) => {
                    if let Some(sps) = s.downcast_ref::<H265Sps>() {
                        self.stream.h265_sps = Some(sps.clone());
                    }
                    if let Some(pb) = pps {
                        if let Some(pps) = pb.downcast_ref::<H265Pps>() {
                            self.stream.h265_pps = Some(pps.clone());
                        }
                    }
                    continue;
                }
                Ok(ParseResult::ParameterSet { .. }) => continue,
                Ok(ParseResult::Slice { slices, bytes_consumed }) => {
                    if slices.is_empty() {
                        return Ok(None);
                    }
                    // POC is computed by the common parser (pocTid0 logic) and
                    // stored in the slice header.
                    let poc = slices[0].slice_header.as_ref().and_then(|sh| match sh {
                        SliceHeader::H265(i) => Some(i.curr_pic_order_cnt_val),
                        _ => None,
                    }).unwrap_or(ctx.curr_poc);
                    ctx.curr_poc = poc;

                    let h265_slices: Vec<H265SliceInfo> = slices.into_iter().map(|e| {
                        H265SliceInfo { nal_data: e.nal_data, slice_header: e.slice_header }
                    }).collect();

                    self.parse_offset += bytes_consumed;
                    let timestamp = self.frame_count as u64 * 33_333;
                    return self.decode_h265_frame(&h265_slices, timestamp);
                }
                Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => {
                    self.parse_offset = self.pending_data.len();
                    return Ok(None);
                }
                Err(e) => return Err(Error::Parser(e.to_string())),
            }
        }
    }

    /// Decode a complete H.265 picture (one or more slice segments) into a VA
    /// surface using the common DPB for reference management.
    fn decode_h265_frame(
        &mut self,
        slices: &[H265SliceInfo],
        timestamp: u64,
    ) -> Result<Option<DecodedFrame>> {
        if slices.is_empty() {
            return Ok(None);
        }

        let ctx = self.h265_ctx.as_mut()
            .ok_or_else(|| Error::InvalidState("H265 context not initialized".to_string()))?;
        let sps = self.stream.h265_sps.as_ref()
            .ok_or_else(|| Error::InvalidState("H265 SPS not available".to_string()))?;
        let pps = self.stream.h265_pps.as_ref()
            .ok_or_else(|| Error::InvalidState("H265 PPS not available".to_string()))?;

        // First slice header carries the picture-level parameters.
        let first_info = match &slices[0].slice_header {
            Some(SliceHeader::H265(i)) => i,
            _ => return Ok(None),
        };
        let is_idr = first_info.is_idr;
        let is_ref = first_info.is_reference;
        let poc = ctx.curr_poc;

        // --- Stage the current picture in the common DPB (spec 8.3.2) ---
        let slot = ctx.dpb.picture_start(sps, first_info, is_ref);

        // --- ReferenceFrames: every in-use RPS reference (used + keep-alive) ---
        let in_use = ctx.dpb.in_use_refs();
        let mut reference_frames: [PictureHEVC; 15] = core::array::from_fn(|_| {
            PictureHEVC::new(VA_INVALID_ID, 0, VA_PICTURE_HEVC_INVALID)
        });
        let mut slot_to_refidx: std::collections::HashMap<usize, u8> = std::collections::HashMap::new();
        for (ri, &(s, p)) in in_use.iter().enumerate().take(15) {
            let sid = ctx.slot_surfaces.get(s).and_then(|o| *o)
                .and_then(|pi| Some(self.surface_pool.entries[pi].surface.id()));
            let is_lt = ctx.dpb.slots().get(s).map(|sl| sl.is_long_term).unwrap_or(false);
            // RPS type flags per FFmpeg find_frame_rps_type: short-term refs are
            // BEFORE (POC < curr) or AFTER (POC > curr); long-term refs get
            // LT_CURR | LONG_TERM_REFERENCE. in_use_refs() yields exactly the
            // current pic's RPS, so POC comparison is unambiguous.
            let flags = if is_lt {
                VA_PICTURE_HEVC_RPS_LT_CURR | VA_PICTURE_HEVC_LONG_TERM_REFERENCE
            } else if p < poc {
                VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE
            } else if p > poc {
                VA_PICTURE_HEVC_RPS_ST_CURR_AFTER
            } else {
                0
            };
            reference_frames[ri] = PictureHEVC::new(sid.unwrap_or(VA_INVALID_ID), p, flags);
            slot_to_refidx.insert(s, ri as u8);
        }

        // --- Per-slice RefPicList (u8 indices into ReferenceFrames) ---
        let lists = ctx.dpb.build_ref_lists();
        let make_list = |l: &Vec<vk_video_parser::h265_dpb::H265RefPic>| -> [u8; 15] {
            core::array::from_fn(|i| {
                l.get(i)
                    .and_then(|r| (r.slot >= 0).then(|| slot_to_refidx.get(&(r.slot as usize)).copied()))
                    .flatten()
                    .unwrap_or(0xFF)
            })
        };
        let slice_ref_lists: Vec<([u8; 15], [u8; 15])> =
            (0..slices.len()).map(|_| (make_list(&lists.l0), make_list(&lists.l1))).collect();

        // --- Allocate the destination surface ---
        let used_pool: std::collections::HashSet<usize> =
            ctx.slot_surfaces.iter().filter_map(|s| *s).collect();
        let (surface_idx, surface) = self.surface_pool.alloc_excluding(&used_pool)
            .ok_or_else(|| Error::InvalidState("No free surfaces available".to_string()))?;
        let surface_id = surface.id();

        // --- CurrPic ---
        let curr_pic = PictureHEVC::new(surface_id, poc, 0);

        // --- pic_fields (SPS + PPS) ---
        let pic_fields = HevcPicFields::new(
            sps.chroma_format_idc as u32,
            sps.separate_colour_plane_flag as u32,
            sps.pcm_enabled_flag as u32,
            sps.scaling_list_enabled_flag as u32,
            pps.transform_skip_enabled_flag as u32,
            sps.amp_enabled_flag as u32,
            sps.strong_intra_smoothing_enabled_flag as u32,
            pps.sign_data_hiding_enabled_flag as u32,
            pps.constrained_intra_pred_flag as u32,
            pps.cu_qp_delta_enabled_flag as u32,
            pps.weighted_pred_flag as u32,
            pps.weighted_bipred_flag as u32,
            pps.transquant_bypass_enabled_flag as u32,
            pps.tiles_enabled_flag as u32,
            pps.entropy_coding_sync_enabled_flag as u32,
            pps.pps_loop_filter_across_slices_enabled_flag as u32,
            pps.loop_filter_across_tiles_enabled_flag as u32,
            sps.pcm_loop_filter_disabled_flag as u32,
            0, // no_pic_reordering_flag
            0, // no_bi_pred_flag
        );

        // --- slice_parsing_fields (SPS + PPS + first slice) ---
        let slice_parsing_fields = HevcSliceParsingFields::new(
            pps.lists_modification_present_flag as u32,
            sps.long_term_ref_pics_present_flag as u32,
            sps.sps_temporal_mvp_enabled_flag as u32,
            pps.cabac_init_present_flag as u32,
            pps.output_flag_present_flag as u32,
            pps.dependent_slice_segments_enabled_flag as u32,
            pps.pps_slice_chroma_qp_offsets_present_flag as u32,
            sps.sample_adaptive_offset_enabled_flag as u32,
            pps.deblocking_filter_override_enabled_flag as u32,
            pps.pps_disable_deblocking_filter_flag as u32,
            pps.slice_segment_header_extension_present_flag as u32,
            first_info.is_rap as u32,        // rap_pic_flag
            first_info.is_idr as u32,        // idr_pic_flag
            (first_info.slice_type == 0) as u32, // intra_pic_flag (0=I)
        );

        // PCM fields. FFmpeg fills these from sps->pcm.* which are 0 when PCM is
        // disabled, yielding the sentinels -1/-1/-3/0 (vaapi_hevc.c). When PCM is
        // enabled the VA fields equal the parsed SPS values directly.
        let pcm_luma_minus1: u8 = if sps.pcm_enabled_flag { sps.pcm_sample_bit_depth_luma_minus1 } else { 255 };
        let pcm_chroma_minus1: u8 = if sps.pcm_enabled_flag { sps.pcm_sample_bit_depth_chroma_minus1 } else { 255 };
        let log2_min_pcm_minus3: u8 = if sps.pcm_enabled_flag { sps.log2_min_pcm_luma_coding_block_size_minus3 } else { 253 };
        let log2_diff_pcm: u8 = if sps.pcm_enabled_flag { sps.log2_diff_max_min_pcm_luma_coding_block_size } else { 0 };

        // --- PictureParameterBufferHEVC ---
        let pic_param = PictureParameterBufferHEVC::new(
            curr_pic,
            reference_frames,
            sps.pic_width_in_luma_samples,
            sps.pic_height_in_luma_samples,
            &pic_fields,
            sps.max_dec_pic_buffering_minus1[0],
            sps.bit_depth_luma_minus8,
            sps.bit_depth_chroma_minus8,
            pcm_luma_minus1,   // pcm_sample_bit_depth_luma_minus1
            pcm_chroma_minus1, // pcm_sample_bit_depth_chroma_minus1
            sps.log2_min_luma_coding_block_size_minus3,
            sps.log2_diff_max_min_luma_coding_block_size,
            sps.log2_min_luma_transform_block_size_minus2,
            sps.log2_diff_max_min_luma_transform_block_size,
            log2_min_pcm_minus3, // log2_min_pcm_luma_coding_block_size_minus3
            log2_diff_pcm,       // log2_diff_max_min_pcm_luma_coding_block_size
            sps.max_transform_hierarchy_depth_intra,
            sps.max_transform_hierarchy_depth_inter,
            pps.pps_init_qp_minus26 as i8,
            pps.diff_cu_qp_delta_depth,
            pps.pps_cb_qp_offset,
            pps.pps_cr_qp_offset,
            0, // log2_parallel_merge_level_minus2 (SPS default)
            pps.num_tile_columns_minus1,
            pps.num_tile_rows_minus1,
            pps.column_width_minus1,
            pps.row_height_minus1,
            &slice_parsing_fields,
            sps.log2_max_pic_order_cnt_lsb_minus4,
            sps.num_short_term_ref_pic_sets,
            sps.num_long_term_ref_pics_sps,
            pps.num_ref_idx_l0_default_active_minus1,
            pps.num_ref_idx_l1_default_active_minus1,
            pps.pps_beta_offset_div2,
            pps.pps_tc_offset_div2,
            pps.num_extra_slice_header_bits,
            if first_info.short_term_ref_pic_set_sps_flag { 0 } else { first_info.num_bits_for_strps_in_slice as u32 },
        );


        // REXT/SCC (sps profile_idc >= 4): the driver expects the full
        // VAPictureParameterBufferHEVCExtension size, attached as the plain
        // picture parameter type (FFmpeg vaapi_hevc.c: pic_param_size).
        let pic_param_buf = if sps.profile_idc >= 4 {
            let rext_fields = HevcRangeExtensionPicFields::new(
                sps.transform_skip_rotation_enabled_flag as u32,
                sps.transform_skip_context_enabled_flag as u32,
                sps.implicit_rdpcm_enabled_flag as u32,
                sps.explicit_rdpcm_enabled_flag as u32,
                sps.extended_precision_processing_flag as u32,
                sps.intra_smoothing_disabled_flag as u32,
                sps.high_precision_offsets_enabled_flag as u32,
                sps.persistent_rice_adaptation_enabled_flag as u32,
                sps.cabac_bypass_alignment_enabled_flag as u32,
                pps.cross_component_prediction_enabled_flag as u32,
                pps.chroma_qp_offset_list_enabled_flag as u32,
            );
            let rext = PictureParameterBufferHEVCRext::new(
                &rext_fields,
                pps.diff_cu_chroma_qp_offset_depth,
                pps.chroma_qp_offset_list_len_minus1,
                pps.log2_sao_offset_scale_luma,
                pps.log2_sao_offset_scale_chroma,
                pps.log2_max_transform_skip_block_size_minus2,
                pps.cb_qp_offset_list,
                pps.cr_qp_offset_list,
            );
            let ext = PictureParameterBufferHEVCExtension::new(&pic_param, &rext);
            self.context.create_buffer(
                BufferType::PictureParameter(PictureParameter::HEVCExtension(ext))
            ).map_err(|e| Error::VaApi(e.to_string()))?
        } else {
            self.context.create_buffer(
                BufferType::PictureParameter(PictureParameter::HEVC(pic_param))
            ).map_err(|e| Error::VaApi(e.to_string()))?
        };

        // --- IQ matrix buffer (only when scaling lists are present, like FFmpeg) ---
        let iq_buf = if pps.pps_scaling_list_data_present_flag || sps.scaling_list_enabled_flag {
            let sl = &sps.scaling_lists;
            let buf = IQMatrixBufferHEVC::new(
                sl.scaling_list_4x4,
                sl.scaling_list_8x8,
                sl.scaling_list_16x16,
                sl.scaling_list_32x32,
                core::array::from_fn(|i| sl.scaling_list_dc_coef_16x16[0][i] as u8),
                core::array::from_fn(|i| sl.scaling_list_dc_coef_32x32[0][i] as u8),
            );
            Some(self.context.create_buffer(
                BufferType::IQMatrix(IQMatrix::HEVC(buf))
            ).map_err(|e| Error::VaApi(e.to_string()))?)
        } else {
            None
        };

        // REXT/SCC: pred-weight offsets move into the rext section of the slice
        // buffer; the base struct keeps them zeroed (FFmpeg vaapi_hevc.c).
        let is_rext = sps.profile_idc >= 4;

        // Collect all buffers in render order (pic param, optional IQ matrix,
        // then per-slice param + data).
        let mut va_buffers: Vec<(String, Buffer)> = Vec::new();
        va_buffers.push(("pic_param".to_string(), pic_param_buf));
        if let Some(b) = iq_buf {
            va_buffers.push(("iq_matrix".to_string(), b));
        }

        // Add all slice buffers BEFORE begin.
        for (si, (slice_info, (ref_l0, ref_l1))) in slices.iter().zip(slice_ref_lists.into_iter()).enumerate() {
            let is_last = si == slices.len() - 1;
            let sh = match &slice_info.slice_header {
                Some(SliceHeader::H265(i)) => i,
                _ => first_info,
            };

            // slice_data_byte_offset: byte offset from the NAL start (incl. the
            // 2-byte NAL header) to the first CABAC byte. FFmpeg: read one bit
            // after the coded header then align -> ((16 + header_bit_size)>>3)+1.
            let slice_data_byte_offset = ((16u32 + sh.header_bit_size as u32) >> 3) + 1;

            // FFmpeg trims the trailing 0x00 alignment byte of a VCL NAL: the last
            // RBSP byte always carries the EOB stop bit, so a trailing 0x00 is pure
            // byte-alignment padding. Match that for slice_data_size and the data buf.
            let nal_len = slice_info.nal_data.len();
            let data_len = if nal_len > 0 && slice_info.nal_data[nal_len - 1] == 0 { nal_len - 1 } else { nal_len };

            let long_slice_flags = HevcLongSliceFlags::new(
                is_last as u32,                       // last_slice_of_pic
                0,                                     // dependent_slice_segment_flag
                match sh.slice_type { 0 => 2, 1 => 1, 2 => 0, n => n } as u32, // slice_type: VA wants de-facto ue values (B=0,P=1,I=2), not our 0=I/1=P/2=B convention
                sh.colour_plane_id as u32,             // color_plane_id
                sh.slice_sao_luma_flag as u32,
                sh.slice_sao_chroma_flag as u32,
                sh.mvd_l1_zero_flag as u32,            // mvd_l1_zero_flag (B-only)
                sh.cabac_init_flag as u32,
                sh.slice_temporal_mvp_enabled_flag as u32,
                sh.slice_deblocking_filter_disabled_flag as u32,
                sh.collocated_from_l0_flag as u32,     // collocated_from_l0_flag
                sh.slice_loop_filter_across_slices_enabled_flag as u32,
            );

            let collocated_ref_idx = if sh.slice_temporal_mvp_enabled_flag {
                sh.collocated_ref_idx
            } else {
                0xFF
            };

            // num_ref_idx_lX_active_minus1 absent for I slices.
            let (eff_l0, eff_l1) = if sh.slice_type == 0 {
                (0u8, 0u8)
            } else {
                (sh.num_ref_idx_l0_active_minus1, sh.num_ref_idx_l1_active_minus1)
            };

            let (base_luma_off_l0, base_chroma_off_l0, base_luma_off_l1, base_chroma_off_l1) =
                if is_rext {
                    ([0i8; 15], [[0i8; 2]; 15], [0i8; 15], [[0i8; 2]; 15])
                } else {
                    (sh.luma_offset_l0.map(|v| v as i8),
                     sh.chroma_offset_l0.map(|c| [c[0] as i8, c[1] as i8]),
                     sh.luma_offset_l1.map(|v| v as i8),
                     sh.chroma_offset_l1.map(|c| [c[0] as i8, c[1] as i8]))
                };

            let slice_param = SliceParameterBufferHEVC::new(
                data_len as u32,                    // slice_data_size (trailing 0x00 trimmed, like FFmpeg)
                0,                                  // slice_data_offset
                VA_SLICE_DATA_FLAG_ALL,             // slice_data_flag
                slice_data_byte_offset,             // slice_data_byte_offset
                sh.slice_segment_address,           // slice_segment_address
                [ref_l0, ref_l1],                   // RefPicList
                &long_slice_flags,
                collocated_ref_idx,
                eff_l0,
                eff_l1,
                sh.slice_qp_delta as i8,
                sh.slice_cb_qp_offset as i8,
                sh.slice_cr_qp_offset as i8,
                sh.slice_beta_offset_div2 as i8,
                sh.slice_tc_offset_div2 as i8,
                sh.luma_log2_weight_denom,
                sh.delta_chroma_log2_weight_denom,
                sh.delta_luma_weight_l0,
                base_luma_off_l0,
                sh.delta_chroma_weight_l0,
                base_chroma_off_l0,
                sh.delta_luma_weight_l1,
                base_luma_off_l1,
                sh.delta_chroma_weight_l1,
                base_chroma_off_l1,
                sh.five_minus_max_num_merge_cand,
                0, // num_entry_point_offsets: FFmpeg never fills this in the VA buffer (designated init leaves it 0)
                0, // entry_offset_to_subset_array (subsets/tiles only)
                0, // slice_data_num_emu_prevn_bytes
            );

            // REXT/SCC: full-size VASliceParameterBufferHEVCExtension with the
            // pred-weight offsets in the rext section (FFmpeg vaapi_hevc.c).
            let slice_param_buf = if is_rext {
                let rext_flags = HevcSliceExtFlags::new(
                    sh.cu_chroma_qp_offset_enabled_flag as u32,
                    sh.use_integer_mv_flag as u32,
                );
                let rext = SliceParameterBufferHEVCRext::new(
                    sh.luma_offset_l0,
                    sh.chroma_offset_l0,
                    sh.luma_offset_l1,
                    sh.chroma_offset_l1,
                    &rext_flags,
                    sh.slice_act_y_qp_offset as i8,
                    sh.slice_act_cb_qp_offset as i8,
                    sh.slice_act_cr_qp_offset as i8,
                );
                let ext = SliceParameterBufferHEVCExtension::new(&slice_param, &rext);
                self.context.create_buffer(
                    BufferType::SliceParameter(SliceParameter::HEVCExtension(ext))
                ).map_err(|e| Error::VaApi(e.to_string()))?
            } else {
                self.context.create_buffer(
                    BufferType::SliceParameter(SliceParameter::HEVC(slice_param))
                ).map_err(|e| Error::VaApi(e.to_string()))?
            };

            let slice_data_buf = self.context.create_buffer(
                BufferType::SliceData(slice_info.nal_data[..data_len].to_vec())
            ).map_err(|e| Error::VaApi(e.to_string()))?;

            va_buffers.push((format!("slice{}:param", si), slice_param_buf));
            va_buffers.push((format!("slice{}:data", si), slice_data_buf));
        }

        // Begin picture ONCE for the entire frame.
        let mut picture = Picture::<PictureNew, Rc<Surface<DmaBufSurfaceDescriptor>>>::new(
            timestamp, Rc::clone(&self.context), Rc::clone(&surface),
        );
        for (_, b) in va_buffers {
            picture.add_buffer(b);
        }
        let picture = picture.begin().map_err(|e| Error::VaApi(e.to_string()))?;
        let picture: Picture<PictureRender, Rc<Surface<DmaBufSurfaceDescriptor>>> =
            picture.render().map_err(|e| Error::VaApi(e.to_string()))?;
        let picture: Picture<PictureEnd, Rc<Surface<DmaBufSurfaceDescriptor>>> =
            picture.end().map_err(|e| Error::VaApi(e.to_string()))?;
        let _synced: Picture<PictureSync, Rc<Surface<DmaBufSurfaceDescriptor>>> =
            picture.sync().map_err(|e| Error::VaApi(e.0.to_string()))?;

        self.surface_pool.mark_ready(surface_idx);
        surface.sync().map_err(|e| Error::VaApi(e.to_string()))?;

        let pixel_data = read_surface_pixels(
            &surface,
            self.stream.width,
            self.stream.height,
            self.stream.display_width,
            self.stream.display_height,
            rt_format_candidates(self.stream.rt_format),
        )?;

        // --- Commit the current picture to the common DPB ---
        for (i, s) in ctx.dpb.slots().iter().enumerate() {
            if !s.valid {
                ctx.slot_surfaces[i] = None;
            }
        }
        ctx.slot_surfaces[slot] = Some(surface_idx);
        ctx.dpb.commit_current(slot);

        // pic_output_flag=0: the picture is decoded and committed to the DPB (it
        // may still be a reference) but is NOT emitted, matching FFmpeg which sets
        // HEVC_FRAME_FLAG_OUTPUT only when sh.pic_output_flag is set (refs.c).
        if !first_info.pic_output_flag {
            return Ok(None);
        }

        // --- Display-order reordering key ---
        if is_idr && self.frame_count > 0 {
            self.gop_count += 1;
        }
        let key = self.gop_count as i64 * 1_000_000 + poc as i64;
        self.reorder_watermark = self.reorder_watermark.max(self.gop_count as i64);
        self.pending_key = key;

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

    // Packed Y410: 4 bytes per pixel, word = U | Y<<10 | V<<20 (10 bits each,
    // top 2 bits padding). This is iHD's actual layout for 10-bit 4:4:4
    // surfaces (HEVC Main444_10) — verified byte-for-byte against FFmpeg's
    // vaapi hwaccel output, which requests Y410 for the same content. Unpack
    // to planar u16 with bottom-justified 10-bit values (yuv444p10le).
    if va_image.format.fourcc == FOURCC_Y410 && va_image.num_planes == 1 {
        let out_width = display_width.min(va_image.width as u32) as usize;
        let out_height = display_height.min(va_image.height as u32) as usize;
        let pitch = va_image.pitches[0] as usize;
        let off = va_image.offsets[0] as usize;
        let plane_size = out_width * out_height;
        let mut buffer = vec![0u8; 3 * plane_size * 2];
        let src = data.as_ptr();
        for row in 0..out_height {
            let row_off = off + row * pitch;
            for col in 0..out_width {
                let p = unsafe { src.add(row_off + col * 4) };
                let word = u32::from_ne_bytes([
                    unsafe { *p },
                    unsafe { *p.add(1) },
                    unsafe { *p.add(2) },
                    unsafe { *p.add(3) },
                ]);
                let y = ((word >> 10) & 0x3FF) as u16;
                let u = (word & 0x3FF) as u16;
                let v = ((word >> 20) & 0x3FF) as u16;
                unsafe {
                    buffer[(row * out_width + col) * 2..(row * out_width + col) * 2 + 2]
                        .copy_from_slice(&y.to_ne_bytes());
                    buffer[(plane_size + row * out_width + col) * 2..(plane_size + row * out_width + col) * 2 + 2]
                        .copy_from_slice(&u.to_ne_bytes());
                    buffer[(2 * plane_size + row * out_width + col) * 2..(2 * plane_size + row * out_width + col) * 2 + 2]
                        .copy_from_slice(&v.to_ne_bytes());
                }
            }
        }
        drop(image);
        let y_ptr = buffer.as_ptr();
        let u_ptr = unsafe { y_ptr.add(plane_size * 2) };
        let v_ptr = unsafe { y_ptr.add(2 * plane_size * 2) };
        return Ok(Some(PixelData {
            // "16" => bps=2 in the consumer; not "P016" => no top-justification shift.
            format: "Y410P16".to_string(),
            y: PixelPlane { data: y_ptr, pitch: out_width * 2, width: out_width, height: out_height },
            u: PixelPlane { data: u_ptr, pitch: out_width * 2, width: out_width, height: out_height },
            v: Some(PixelPlane { data: v_ptr, pitch: out_width * 2, width: out_width, height: out_height }),
            buffer,
        }));
    }

    // Packed XYUV (DRM 'XYUV' == FFmpeg VUYX): 4 bytes per pixel in V,U,Y,X
    // order. iHD derives Main444 surfaces in this layout; unpack to planar.
    if va_image.format.fourcc == u32::from_ne_bytes(*b"XYUV") && va_image.num_planes == 1 {
        let out_width = display_width.min(va_image.width as u32) as usize;
        let out_height = display_height.min(va_image.height as u32) as usize;
        let pitch = va_image.pitches[0] as usize;
        let off = va_image.offsets[0] as usize;
        let plane_size = out_width * out_height;
        let mut buffer = vec![0u8; 3 * plane_size];
        let src = data.as_ptr();
        for row in 0..out_height {
            let row_off = off + row * pitch;
            for col in 0..out_width {
                let px = unsafe { src.add(row_off + col * 4) };
                buffer[row * out_width + col] = unsafe { *px.add(2) }; // Y
                buffer[plane_size + row * out_width + col] = unsafe { *px.add(1) }; // U
                buffer[2 * plane_size + row * out_width + col] = unsafe { *px }; // V
            }
        }
        drop(image);
        let y_ptr = buffer.as_ptr();
        let u_ptr = unsafe { y_ptr.add(plane_size) };
        let v_ptr = unsafe { y_ptr.add(2 * plane_size) };
        return Ok(Some(PixelData {
            format: "YUV444".to_string(),
            y: PixelPlane { data: y_ptr, pitch: out_width, width: out_width, height: out_height },
            u: PixelPlane { data: u_ptr, pitch: out_width, width: out_width, height: out_height },
            v: Some(PixelPlane { data: v_ptr, pitch: out_width, width: out_width, height: out_height }),
            buffer,
        }));
    }

    // Determine format from fourcc. P010/P012 carry full-precision samples
    // (left-aligned in u16) and are scaled to 8-bit below.
    let fourcc = va_image.format.fourcc;
    let is_nv12 = fourcc == libva::VA_FOURCC_NV12;
    // XYUV (defensive): the driver's native 4:4:4 layout (Y, X-unused, U, V);
    // its 444P image view over an XYUV surface returns broken chroma on iHD.
    let is_xyuv = fourcc == u32::from_ne_bytes(*b"XYUV");
    // AYUV: single interleaved plane, 4 bytes/pixel. iHD stores it as
    // [V, U, Y, A] in memory (see the de-interleave below).
    let is_ayuv = fourcc == libva::VA_FOURCC_AYUV;
    let is_p016 = fourcc == libva::VA_FOURCC_P016;
    let p016_shift = if fourcc == libva::VA_FOURCC_P010 {
        Some(6u32)
    } else if fourcc == libva::VA_FOURCC_P012 {
        Some(4u32)
    } else {
        None
    };
    let is_444 = fourcc == libva::VA_FOURCC_444P || is_xyuv || is_ayuv;
    // P016 keeps full 10-bit precision (2 bytes per sample); the rest are 8-bit.
    let bps = if is_p016 { 2 } else { 1 };
    let format_str = if is_nv12 || p016_shift.is_some() {
        "NV12".to_string()
    } else if is_p016 {
        "P016".to_string()
    } else if is_444 {
        "YUV444".to_string()
    } else if fourcc == u32::from_ne_bytes(*b"YV12") {
        "YV12".to_string()
    } else if fourcc == u32::from_ne_bytes(*b"I420") {
        "I420".to_string()
    } else {
        return Err(Error::VaApi(format!("Unsupported image format: {:X}", fourcc)));
    };

    // Validate num_planes
    let min_planes = if is_ayuv { 1 } else if is_xyuv { 4 } else if is_nv12 || p016_shift.is_some() || is_p016 { 2 } else { 3 };
    if va_image.num_planes < min_planes {
        return Err(Error::VaApi(format!(
            "Unexpected num_planes={} for format {}",
            va_image.num_planes, format_str
        )));
    }

    // Copy data into owned buffer so we can drop the Image (which unmaps the surface)
    let buffer = data.to_vec();

    // Crop to the display size (top-left origin). The surface may be larger than
    // the display size due to frame cropping / padding.
    let out_width = display_width.min(va_image.width as u32) as usize;
    let out_height = display_height.min(va_image.height as u32) as usize;
    // 4:4:4 keeps full-resolution chroma; 4:2:0 downsamples by 2.
    let uv_width = if is_444 { out_width } else { (out_width + 1) / 2 };
    let uv_height = if is_444 { out_height } else { (out_height + 1) / 2 };
    let _ = bps; // plane widths are in samples; pitches in the image are in bytes

    // AYUV: de-interleave the single [A,Y,U,V] plane into planar Y/U/V.
    if is_ayuv {
        let y_pitch = va_image.pitches[0] as usize; // bytes per row (width*4)
        let base = va_image.offsets[0] as usize;
        let mut out = vec![0u8; out_width * 3 * out_height];
        for y in 0..out_height {
            let row = unsafe { buffer.as_ptr().add(base + y * y_pitch) };
            for x in 0..out_width {
                let p = unsafe { row.add(x * 4) };
                // iHD stores the AYUV view as [V, U, Y, A] in memory.
                out[y * out_width + x] = unsafe { *p.add(2) }; // Y
                out[out_width * out_height + y * out_width + x] = unsafe { *p.add(1) }; // U
                out[2 * out_width * out_height + y * out_width + x] = unsafe { *p.add(0) }; // V
            }
        }
        drop(image);
        return Ok(Some(PixelData {
            format: format_str,
            y: PixelPlane {
                data: unsafe { out.as_ptr() },
                pitch: out_width,
                width: out_width,
                height: out_height,
            },
            u: PixelPlane {
                data: unsafe { out.as_ptr().add(out_width * out_height) },
                pitch: out_width,
                width: out_width,
                height: out_height,
            },
            v: Some(PixelPlane {
                data: unsafe { out.as_ptr().add(2 * out_width * out_height) },
                pitch: out_width,
                width: out_width,
                height: out_height,
            }),
            buffer: out,
        }));
    }

    // P010/P012: scale the left-aligned 16-bit samples to 8-bit with
    // round-to-nearest and saturation (matches the NVDEC P016 and Vulkan
    // G10X6/G12X4 readbacks). Output keeps the semi-planar NV12 layout.
    if let Some(shift) = p016_shift {
        let bits = 16u32 - shift;
        let half = 1u32 << (bits - 1);
        let scale = |v: u16| -> u8 {
            let x = (v >> shift) as u32;
            (((x * 256 + half) >> bits).min(255)) as u8
        };
        let read_u16 = |base: *const u8, off: usize| -> u16 {
            unsafe { u16::from_ne_bytes([*base.add(off), *base.add(off + 1)]) }
        };

        let y_offset = va_image.offsets[0] as usize;
        let u_offset = va_image.offsets[1] as usize;
        let y_pitch = va_image.pitches[0] as usize / 2; // u16 samples per row
        let uv_pitch = va_image.pitches[1] as usize / 2;

        let mut out = vec![0u8; out_width * out_height + uv_width * 2 * uv_height];
        for y in 0..out_height {
            let row = unsafe { buffer.as_ptr().add(y_offset) };
            for x in 0..out_width {
                out[y * out_width + x] = scale(read_u16(row, y * y_pitch * 2 + x * 2));
            }
        }
        let uv_row = unsafe { buffer.as_ptr().add(u_offset) };
        for y in 0..uv_height {
            for x in 0..uv_width {
                let base = y * uv_pitch * 2 + x * 4;
                out[out_width * out_height + y * (uv_width * 2) + x * 2] =
                    scale(read_u16(uv_row, base));
                out[out_width * out_height + y * (uv_width * 2) + x * 2 + 1] =
                    scale(read_u16(uv_row, base + 2));
            }
        }

        // Image is dropped here, unmapping the surface
        drop(image);

        return Ok(Some(PixelData {
            format: format_str,
            y: PixelPlane {
                data: out.as_ptr(),
                pitch: out_width,
                width: out_width,
                height: out_height,
            },
            u: PixelPlane {
                data: unsafe { out.as_ptr().add(out_width * out_height) },
                pitch: uv_width * 2,
                width: uv_width,
                height: uv_height,
            },
            v: None,
            buffer: out,
        }))
    }

    // Build plane descriptors from the copied buffer. XYUV stores an unused
    // X plane at index 1, so chroma starts at index 2.
    let y_offset = va_image.offsets[0] as usize;
    let uv_base = if is_xyuv { 2 } else { 1 };
    let u_offset = va_image.offsets[uv_base] as usize;

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

    let v_plane = if !is_nv12 && !is_p016 {
        let v_offset = va_image.offsets[uv_base + 1] as usize;
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
    fourccs: &[u32],
) -> Result<Option<PixelData>> {

    // Primary: vaCreateImage + vaGetImage (driver-supported CPU read).
    // Read the full coded size, then crop to the display size in read_from_image.
    // The driver may expose a semi-planar or planar variant of the stream's
    // render format; try candidates in preference order.
    for &fourcc in fourccs {
        let format = libva::VAImageFormat {
            fourcc,
            ..Default::default()
        };
        match Image::create_from(surface, format, (width, height), (width, height)) {
            Ok(image) => return read_from_image(image, display_width, display_height),
            Err(_) => {}
        }
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
        let b0 = data[start];
        // H.265 NAL header: forbidden_zero_bit(1) | nal_unit_type(6).
        let h265_nal_type = (b0 >> 1) & 0x3F;
        // H.264 NAL header: forbidden_zero_bit(1) | nal_ref_idc(2) |
        // nal_unit_type(5).
        let h264_nal_type = b0 & 0x1F;
        // H.265 VPS/SPS/PPS
        if h265_nal_type == 32 || h265_nal_type == 33 || h265_nal_type == 34 {
            return CoreVideoCodec::DecodeH265;
        }
        // H.264 SPS/PPS
        if h264_nal_type == 7 || h264_nal_type == 8 {
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
        vp9_profile: 0,
        vp9_bit_depth: 8,
        sps: sps_opt,
        pps: pps_opt,
        h265_sps: None,
        h265_pps: None,
    })
}

/// Parse H.265 stream info.
fn parse_h265_info(display: &Display, data: &[u8]) -> Result<StreamInfo> {
    let mut parser = H265Parser::new();
    parser.init(&DetectedVideoFormat::new(CoreVideoCodec::DecodeH265))
        .map_err(|e| Error::Parser(e.to_string()))?;

    let packet = BitstreamPacket::new(data.to_vec());
    let mut width = 0u32;
    let mut height = 0u32;
    let mut display_width = 0u32;
    let mut display_height = 0u32;
    let mut max_dpb = 4u32;
    let mut sps_opt: Option<H265Sps> = None;
    let mut pps_opt: Option<H265Pps> = None;

    // The first ParameterSet result carries the SPS (and usually the PPS too).
    if let Ok(ParseResult::ParameterSet { sps: Some(s), pps, .. }) = parser.parse(&packet) {
        if let Some(sps) = s.downcast_ref::<H265Sps>() {
            sps_opt = Some(sps.clone());
        }
        if let Some(pps_box) = pps {
            if let Some(pps) = pps_box.downcast_ref::<H265Pps>() {
                pps_opt = Some(pps.clone());
            }
        }
    }

    let sps = sps_opt.as_ref()
        .ok_or_else(|| Error::DecoderInit("Failed to parse H.265 SPS".to_string()))?;

    // Coded size = the SPS luma dimensions (HEVC coded size is NOT necessarily
    // 16-aligned, unlike H.264). The surface is created at this exact size and
    // PictureParameterBufferHEVC carries the same values, so they match.
    width = sps.pic_width_in_luma_samples as u32;
    height = sps.pic_height_in_luma_samples as u32;
    max_dpb = sps.max_num_ref_frames as u32;

    // Conformance window -> display size (H.265 7.4.3.2.1).
    let (sub_w, sub_h) = match sps.chroma_format_idc {
        0 => (1u32, 1u32),
        1 => (2, 2),   // 4:2:0
        2 => (2, 1),   // 4:2:2
        _ => (1, 1),   // 4:4:4
    };
    if sps.conformance_window_flag {
        let cw = (sps.conf_win_left_offset + sps.conf_win_right_offset) * sub_w;
        let ch = (sps.conf_win_top_offset + sps.conf_win_bottom_offset) * sub_h;
        display_width = if cw < width { width - cw } else { width };
        display_height = if ch < height { height - ch } else { height };
    } else {
        display_width = width;
        display_height = height;
    }

    // Profile from profile_idc (1=Main, 2=Main10, 3=MainStillPicture, 4=Rext,
    // 5=Main10StillPicture). Rext is a superset profile: its decodable content
    // is determined by chroma_format_idc + bit depth, so map those to the
    // matching named profile (FFmpeg resolves Rext via PTL compatibility and
    // constraint flags; for decode config selection the content parameters
    // give the same result).
    let bit_depth = 8 + sps.bit_depth_luma_minus8 as u32;
    let preferred = match sps.profile_idc {
        1 | 3 => libva::VAProfile::VAProfileHEVCMain,
        2 | 5 => {
            if sps.chroma_format_idc == 3 {
                libva::VAProfile::VAProfileHEVCMain444_10
            } else {
                libva::VAProfile::VAProfileHEVCMain10
            }
        }
        4 => match (sps.chroma_format_idc, bit_depth) {
            (3, 8) => libva::VAProfile::VAProfileHEVCMain444,
            (3, 10) => libva::VAProfile::VAProfileHEVCMain444_10,
            (3, _) => libva::VAProfile::VAProfileHEVCMain444_12,
            (2, 10) => libva::VAProfile::VAProfileHEVCMain422_10,
            (2, 12) => libva::VAProfile::VAProfileHEVCMain422_12,
            (_, 10) => libva::VAProfile::VAProfileHEVCMain10,
            (_, 12) => libva::VAProfile::VAProfileHEVCMain12,
            _ => libva::VAProfile::VAProfileHEVCMain,
        },
        _ => libva::VAProfile::VAProfileHEVCMain,
    };
    let supported = |p: VAProfileType| {
        display
            .query_config_entrypoints(p)
            .map(|e| e.contains(&libva::VAEntrypoint::VAEntrypointVLD))
            .unwrap_or(false)
    };
    let mut profile = preferred;
    if !supported(profile) {
        for p in [
            libva::VAProfile::VAProfileHEVCMain10,
            libva::VAProfile::VAProfileHEVCMain444,
            libva::VAProfile::VAProfileHEVCMain444_10,
            libva::VAProfile::VAProfileHEVCMain422_10,
            libva::VAProfile::VAProfileHEVCMain,
        ] {
            if supported(p) {
                profile = p;
                break;
            }
        }
    }

    // RT format from bit depth + chroma.
    let rt_format = match (bit_depth, sps.chroma_format_idc) {
        (8, 0) | (8, 1) => libva::VA_RT_FORMAT_YUV420,
        (8, 2) => libva::VA_RT_FORMAT_YUV422,
        (8, 3) => libva::VA_RT_FORMAT_YUV444,
        (10, 0) | (10, 1) => libva::VA_RT_FORMAT_YUV420_10,
        (10, 2) => libva::VA_RT_FORMAT_YUV422_10,
        (10, 3) => libva::VA_RT_FORMAT_YUV444_10,
        _ => libva::VA_RT_FORMAT_YUV420,
    };

    Ok(StreamInfo {
        codec: CoreVideoCodec::DecodeH265,
        profile,
        width,
        height,
        display_width,
        display_height,
        max_dpb: max_dpb.min(16).max(1),
        rt_format,
        vp9_profile: 0,
        vp9_bit_depth: 8,
        sps: None,
        pps: None,
        h265_sps: sps_opt,
        h265_pps: pps_opt,
    })
}

/// Parse VP9 stream info from the first frame header.
fn parse_vp9_info(display: &Display, data: &[u8]) -> Result<StreamInfo> {
    // Extract the first frame payload. IVF containers have a 32-byte header
    // followed by per-packet headers (4-byte payload size + 8-byte pts).
    let first_payload: Vec<u8> = if data.len() >= 32 && data[0..4] == *b"DKIF" {
        if data.len() < 32 + 12 {
            return Err(Error::DecoderInit("IVF too short for a packet".to_string()));
        }
        let size = u32::from_le_bytes(data[32..36].try_into().unwrap()) as usize;
        if data.len() < 32 + 12 + size {
            return Err(Error::DecoderInit("IVF packet truncated".to_string()));
        }
        data[32 + 12..32 + 12 + size].to_vec()
    } else {
        data.to_vec()
    };

    // A single IVF packet may hold a superframe; parse its first subframe.
    let frames = expand_superframes(&first_payload);
    if frames.is_empty() {
        return Err(Error::DecoderInit("No VP9 frames found".to_string()));
    }

    let mut parser = Vp9Parser::new();
    parser.init(&DetectedVideoFormat::new(CoreVideoCodec::DecodeVp9))
        .map_err(|e| Error::Parser(e.to_string()))?;

    let parsed = parser
        .parse_frame_with_offset(&frames[0].data, frames[0].superframe_frame_offset)
        .map_err(|e| Error::Parser(e.to_string()))?;

    let width = parsed.frame_width;
    let height = parsed.frame_height;
    if width == 0 || height == 0 {
        return Err(Error::DecoderInit("Failed to parse VP9 dimensions".to_string()));
    }
    let display_width = if parsed.render_width > 0 { parsed.render_width } else { width };
    let display_height = if parsed.render_height > 0 { parsed.render_height } else { height };

    let profile_num = parsed.picture_info.profile as u8;
    let bit_depth = parsed.color_config.bit_depth;

    let profile = match profile_num {
        0 => libva::VAProfile::VAProfileVP9Profile0,
        1 => libva::VAProfile::VAProfileVP9Profile1,
        _ => libva::VAProfile::VAProfileVP9Profile2,
    };
    if !display
        .query_config_entrypoints(profile)
        .map(|e| e.contains(&libva::VAEntrypoint::VAEntrypointVLD))
        .unwrap_or(false)
    {
        return Err(Error::DecoderInit(format!(
            "VA driver lacks VP9 profile {profile_num} decode"
        )));
    }

    // Surface (rt) format per bit depth. 10/12-bit content decodes into P010/
    // P012 surfaces so readback preserves full precision (the driver's 8-bit
    // down-convert dithers and is not reproducible sample-wise).
    let rt_format = match (bit_depth, profile_num) {
        (8, 1) => libva::VA_RT_FORMAT_YUV444,
        (8, _) => libva::VA_RT_FORMAT_YUV420,
        (10, _) => libva::VA_RT_FORMAT_YUV420_10,
        (12, _) => libva::VA_RT_FORMAT_YUV420_12,
        _ => {
            return Err(Error::DecoderInit(format!(
                "Unsupported VP9 bit depth {bit_depth}"
            )))
        }
    };

    Ok(StreamInfo {
        codec: CoreVideoCodec::DecodeVp9,
        profile,
        width,
        height,
        display_width,
        display_height,
        max_dpb: 8,
        rt_format,
        vp9_profile: profile_num,
        vp9_bit_depth: bit_depth,
        sps: None,
        pps: None,
        h265_sps: None,
        h265_pps: None,
    })
}

/// One subframe extracted from an IVF packet payload (possibly a superframe).
#[derive(Debug, Clone)]
struct ExpandedVp9Frame {
    data: Vec<u8>,
    /// Offset of this frame within the original superframe payload (0 if the
    /// payload is not a superframe).
    superframe_frame_offset: u32,
}

/// Split an IVF packet payload into its subframes. A VP9 superframe carries a
/// start code (0x543210FE LE) and a per-frame size index at the tail; each
/// subframe is decoded separately. Same logic as the NVDEC path.
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

impl VaapiDecoder {
    /// FOURCC to request from `vaGetImage` for the current VP9 surface format.
    fn vp9_image_fourcc(&self) -> u32 {
        match (self.stream.vp9_profile, self.stream.vp9_bit_depth) {
            // Profile 1 is 8-bit 4:4:4. The driver allocates XYUV surfaces,
            // but its XYUV/444P image views over them return broken chroma;
            // the AYUV view (single interleaved [A,Y,U,V] plane) works.
            (1, _) => libva::VA_FOURCC_AYUV,
            (_, 10) => libva::VA_FOURCC_P010,
            (_, 12) => libva::VA_FOURCC_P012,
            _ => libva::VA_FOURCC_NV12,
        }
    }

    /// Parse and decode pending VP9 data (IVF packets or a raw frame).
    ///
    /// VP9 has no B-frames: display order equals decode order, with
    /// `show_existing_frame` commands re-displaying an earlier picture in
    /// place. Frames are therefore emitted directly (no reorder buffer).
    fn decode_vp9_pending(&mut self) -> Result<Option<DecodedFrame>> {
        if self.parse_offset >= self.pending_data.len() {
            return Ok(None);
        }

        if !self.input_is_ivf {
            // Raw single frame: the whole remaining buffer is one frame.
            let data = self.pending_data[self.parse_offset..].to_vec();
            self.parse_offset = self.pending_data.len();
            for f in expand_superframes(&data) {
                if let Some(frame) = self.decode_vp9_frame(&f.data, f.superframe_frame_offset, 0)? {
                    return Ok(Some(frame));
                }
            }
            return Ok(None);
        }

        // IVF: 12-byte packet header (4-byte payload size + 8-byte pts).
        loop {
            if self.parse_offset + 12 > self.pending_data.len() {
                return Ok(None);
            }
            let size = u32::from_le_bytes(
                self.pending_data[self.parse_offset..self.parse_offset + 4]
                    .try_into()
                    .unwrap(),
            ) as usize;
            if size == 0 || self.parse_offset + 12 + size > self.pending_data.len() {
                return Ok(None);
            }
            let pts = u64::from_le_bytes(
                self.pending_data[self.parse_offset + 4..self.parse_offset + 12]
                    .try_into()
                    .unwrap(),
            );
            let payload = self.pending_data
                [self.parse_offset + 12..self.parse_offset + 12 + size]
                .to_vec();
            self.parse_offset += 12 + size;
            for f in expand_superframes(&payload) {
                if let Some(frame) =
                    self.decode_vp9_frame(&f.data, f.superframe_frame_offset, pts)?
                {
                    return Ok(Some(frame));
                }
            }
        }
    }

    /// Decode one (superframe-expanded) VP9 frame and, if it is displayed,
    /// return the decoded picture.
    fn decode_vp9_frame(
        &mut self,
        data: &[u8],
        superframe_offset: u32,
        timestamp: u64,
    ) -> Result<Option<DecodedFrame>> {
        let parser = self.vp9_parser.as_mut()
            .ok_or_else(|| Error::InvalidState("VP9 parser not initialized".to_string()))?;

        let parsed = parser
            .parse_frame_with_offset(data, superframe_offset)
            .map_err(|e| Error::Parser(e.to_string()))?;

        // show_existing_frame: no decode, no DPB change — re-display an
        // already-decoded surface (FFmpeg re-outputs refs[frame_to_show_map_idx]).
        if parsed.show_existing_frame {
            let slot = self.vp9_ctx.as_ref()
                .ok_or_else(|| Error::InvalidState("VP9 context not initialized".to_string()))?
                .dpb
                .slot_of_frame_buffer(parsed.frame_to_show_map_idx as usize);
            if slot < 0 {
                return Err(Error::InvalidState(format!(
                    "show-existing-frame: frame buffer {} is empty",
                    parsed.frame_to_show_map_idx
                )));
            }
            let pool_idx = self.vp9_ctx.as_ref().unwrap().slot_surfaces[slot as usize]
                .ok_or_else(|| Error::InvalidState("show-existing-frame: slot has no surface".to_string()))?;
            let surface = Rc::clone(&self.surface_pool.entries[pool_idx].surface);
            self.surface_pool.sync_surface(pool_idx)?;
            let pixel_data = read_surface_pixels(
                &surface,
                self.stream.width,
                self.stream.height,
                self.stream.display_width,
                self.stream.display_height,
                &[self.vp9_image_fourcc()],
            )?;
            let mut frame = DecodedFrame::new(
                self.frame_count,
                timestamp as i64,
                self.stream.display_width,
                self.stream.display_height,
                false,
            );
            frame.pixel_data = pixel_data;
            self.frame_count += 1;
            return Ok(Some(frame));
        }

        let pi = &parsed.picture_info;
        let is_key = parsed.frame_is_intra;

        // 1. Resolve references and choose the output slot from the common
        //    DPB (pre-decode state; the frame's refresh is committed below,
        //    after the decode, so subsequent frames see the updated frame
        //    buffers). Slots no longer referenced by any frame buffer release
        //    their surface first.
        let (ref_pool_idxs, out_slot, used_pool) = {
            let ctx = self.vp9_ctx.as_mut()
                .ok_or_else(|| Error::InvalidState("VP9 context not initialized".to_string()))?;

            let live: std::collections::HashSet<usize> = ctx.dpb.frame_buffer_slots()
                .iter()
                .filter(|s| **s >= 0)
                .map(|s| *s as usize)
                .collect();
            for (s, entry) in ctx.slot_surfaces.iter_mut().enumerate() {
                if !live.contains(&s) {
                    *entry = None;
                }
            }

            let mut ref_pool_idxs = [None; 8];
            for (i, &slot) in ctx.dpb.frame_buffer_slots().iter().enumerate() {
                if slot >= 0 {
                    ref_pool_idxs[i] = ctx.slot_surfaces[slot as usize];
                }
            }

            let out_slot = ctx.dpb.choose_output_slot() as usize;
            let used_pool: std::collections::HashSet<usize> =
                ctx.slot_surfaces.iter().flatten().copied().collect();
            (ref_pool_idxs, out_slot, used_pool)
        };

        let mut ref_surfaces = [libva::VA_INVALID_ID; 8];
        for (i, pool_idx) in ref_pool_idxs.iter().enumerate() {
            if let Some(pool_idx) = pool_idx {
                ref_surfaces[i] = self.surface_pool.entries[*pool_idx].surface.id();
            }
        }

        // 2. Allocate a surface that is not currently referenced by any frame
        //    buffer (the output slot itself was just released above).
        let (pool_idx, surface) = self
            .surface_pool
            .alloc_excluding(&used_pool)
            .ok_or_else(|| Error::DecoderInit("No free VA surface for VP9 frame".to_string()))?;

        // 3. Build the VA parameter buffers (FFmpeg vaapi_vp9.c mapping).
        let (pic_type, slice_type) = build_vp9_va_buffers(&parsed, ref_surfaces, data.len() as u32);
        let pic_buf = self.context.create_buffer(pic_type)
            .map_err(|e| Error::VaApi(e.to_string()))?;
        let slice_buf = self.context.create_buffer(slice_type)
            .map_err(|e| Error::VaApi(e.to_string()))?;

        // 4. Decode: picture parameters + slice parameters + whole frame as
        //    slice data (offset 0, like FFmpeg passes the full packet).
        let mut picture = Picture::<PictureNew, Rc<Surface<DmaBufSurfaceDescriptor>>>::new(
            timestamp,
            Rc::clone(&self.context),
            surface.clone(),
        );
        picture.add_buffer(pic_buf);
        picture.add_buffer(slice_buf);
        let slice_data_buf = self
            .context
            .create_buffer(BufferType::SliceData(data.to_vec()))
            .map_err(|e| Error::VaApi(e.to_string()))?;
        picture.add_buffer(slice_data_buf);

        let picture = picture.begin().map_err(|e| Error::VaApi(e.to_string()))?;
        let picture = picture.render().map_err(|e| Error::VaApi(e.to_string()))?;
        let picture = picture.end().map_err(|e| Error::VaApi(e.to_string()))?;
        let _synced: Picture<PictureSync, Rc<Surface<DmaBufSurfaceDescriptor>>> = picture
            .sync()
            .map_err(|e| Error::VaApi(e.0.to_string()))?;

        self.surface_pool.mark_ready(pool_idx);

        // 5. Commit the frame's refresh into the common DPB and track the
        //    surface for the slot.
        {
            let ctx = self.vp9_ctx.as_mut()
                .ok_or_else(|| Error::InvalidState("VP9 context not initialized".to_string()))?;
            ctx.dpb.commit_frame(pi.refresh_frame_flags, out_slot as i32);
            ctx.slot_surfaces[out_slot] = Some(pool_idx);
        }

        // 6. Display if requested; otherwise the frame was decoded for
        //    references only.
        if pi.flags.show_frame == 0 {
            return Ok(None);
        }

        self.surface_pool.sync_surface(pool_idx)?;
        let pixel_data = read_surface_pixels(
            &surface,
            self.stream.width,
            self.stream.height,
            self.stream.display_width,
            self.stream.display_height,
            &[self.vp9_image_fourcc()],
        )?;

        let mut frame = DecodedFrame::new(
            self.frame_count,
            timestamp as i64,
            self.stream.display_width,
            self.stream.display_height,
            is_key,
        );
        frame.pixel_data = pixel_data;
        self.frame_count += 1;
        Ok(Some(frame))
    }
}

/// Build the VA picture- and slice-parameter buffers for one VP9 frame.
///
/// Field mapping follows FFmpeg's `vaapi_vp9.c` (the old-style
/// `VADecPictureParameterBufferVP9` this driver implements). The whole
/// superframe-expanded frame payload is passed as slice data with offset 0.
fn build_vp9_va_buffers(
    fd: &Vp9FrameData,
    ref_surfaces: [libva::VASurfaceID; 8],
    slice_data_size: u32,
) -> (BufferType, BufferType) {
    let pi = &fd.picture_info;
    let cc = &fd.color_config;
    let lf = &fd.loop_filter;
    let sg = &fd.segmentation;
    let is_key = fd.frame_is_intra;

    let pic_fields = VP9PicFields::new(
        cc.subsampling_x as u32,
        cc.subsampling_y as u32,
        // frame_type: 0 = key frame, 1 = inter frame (old VA spec).
        if is_key { 0 } else { 1 },
        pi.flags.show_frame as u32,
        pi.flags.error_resilient_mode as u32,
        pi.flags.intra_only as u32,
        // Key frames never use high-precision MVs (FFmpeg forces 0).
        if is_key { 0 } else { pi.flags.allow_high_precision_mv as u32 },
        // The parser's enum values match the VA mcomp_filter_type values
        // exactly (FFmpeg computes `literal ^ (literal <= 1)` from the raw
        // bitstream literal, which reduces to this direct cast).
        pi.interpolation_filter as u32,
        pi.flags.frame_parallel_decoding_mode as u32,
        pi.flags.reset_frame_context as u32,
        pi.flags.refresh_frame_context as u32,
        pi.frame_context_idx as u32,
        pi.flags.segmentation_enabled as u32,
        sg.flags.segmentation_temporal_update as u32,
        sg.flags.segmentation_update_map as u32,
        fd.ref_frame_idx[0] as u32, // last_ref_frame
        ((pi.ref_frame_sign_bias_mask >> 1) & 1) as u32,
        fd.ref_frame_idx[1] as u32, // golden_ref_frame
        ((pi.ref_frame_sign_bias_mask >> 2) & 1) as u32,
        fd.ref_frame_idx[2] as u32, // alt_ref_frame
        ((pi.ref_frame_sign_bias_mask >> 3) & 1) as u32,
        pi.lossless as u32,
    );

    let segment_pred_probs = if sg.flags.segmentation_temporal_update != 0 {
        [sg.segmentation_pred_prob[0], sg.segmentation_pred_prob[1], sg.segmentation_pred_prob[2]]
    } else {
        [255, 255, 255]
    };

    let pic_param = PictureParameterBufferVP9::new(
        fd.frame_width as u16,
        fd.frame_height as u16,
        ref_surfaces,
        &pic_fields,
        lf.loop_filter_level,
        // sharpness_level uses the raw bitstream values (0=SHARP, 1=SHARP_5TAP,
        // 2=BICUBIC) — no remap for this buffer layout.
        lf.loop_filter_sharpness,
        pi.tile_rows_log2,
        pi.tile_cols_log2,
        // The common parser leaves `uncompressed_header_size` unset; the
        // uncompressed header length is exactly `compressed_header_offset`
        // (bytes from frame start to the first partition, like cuvid's
        // frameTagSize).
        fd.compressed_header_offset.min(255) as u8, // frame_header_length_in_bytes
        fd.compressed_header_size.min(u16::MAX as u32) as u16, // first_partition_size
        sg.segmentation_tree_probs[..7].try_into().unwrap(),
        segment_pred_probs,
        pi.profile as u8,
        cc.bit_depth,
    );

    let seg_param = vp9_segment_params(fd);
    let slice_param = SliceParameterBufferVP9::new(slice_data_size, 0, VA_SLICE_DATA_FLAG_ALL, seg_param);

    (
        BufferType::PictureParameter(PictureParameter::VP9(pic_param)),
        BufferType::SliceParameter(SliceParameter::VP9(slice_param)),
    )
}

/// Clamp to an n-bit unsigned range.
fn clip_uint(v: i32, bits: u32) -> usize {
    v.clamp(0, (1i32 << bits) - 1) as usize
}

/// Build the per-segment VA parameters (quantization scales + loop filter
/// levels) following FFmpeg's `vp9.c` qmul/lflvl computation.
///
/// Parser feature order per segment: [0]=delta Q (s8), [1]=delta LF level
/// (s6), [2]=reference frame index (u2), [3]=skip flag — matching FFmpeg's
/// `q_enabled/lf_enabled/ref_enabled/skip_enabled` read order.
fn vp9_segment_params(fd: &Vp9FrameData) -> [SegmentParameterVP9; 8] {
    let pi = &fd.picture_info;
    let sg = &fd.segmentation;
    let lf = &fd.loop_filter;

    let enabled = pi.flags.segmentation_enabled != 0;
    let abs_or_delta = sg.flags.segmentation_abs_or_delta_update != 0;
    let bpp_index = match fd.color_config.bit_depth {
        10 => 1,
        12 => 2,
        _ => 0,
    };

    let yac_qi = pi.base_q_idx as i32;
    let sh = if lf.loop_filter_level >= 32 { 1 } else { 0 };
    let lf_delta_enabled = lf.flags.loop_filter_delta_enabled != 0;
    let ref_delta = &lf.loop_filter_ref_deltas;
    let mode_delta = &lf.loop_filter_mode_deltas;

    let mut out: [SegmentParameterVP9; 8] = unsafe { std::mem::zeroed() };
    // FFmpeg fills only seg[0] when segmentation is disabled ("some hwaccels
    // don't ignore these fields if segmentation is disabled" — iHD included).
    for i in 0..(if enabled { 8 } else { 1 }) {
        let fe = sg.feature_enabled[i];
        let q_enabled = enabled && (fe & 1) != 0;
        let lf_enabled = enabled && (fe & 2) != 0;
        let ref_enabled = enabled && (fe & 4) != 0;
        let skip_enabled = (fe & 8) != 0;

        // Quantization scales (FFmpeg: qmul[Y/UV][DC/AC]).
        let mut qyac: i32 = if q_enabled {
            let qv = sg.feature_data[i][0] as i32;
            if abs_or_delta { qv } else { yac_qi + qv }
        } else {
            yac_qi
        };
        let qydc = clip_uint(qyac + pi.delta_q_y_dc as i32, 8);
        let quvdc = clip_uint(qyac + pi.delta_q_uv_dc as i32, 8);
        let quvac = clip_uint(qyac + pi.delta_q_uv_ac as i32, 8);
        qyac = clip_uint(qyac, 8) as i32;

        // Loop filter level per reference/mode (FFmpeg: feat[i].lflvl[4][2]).
        let lflvl: i32 = if lf_enabled {
            let lv = sg.feature_data[i][1] as i32;
            if abs_or_delta { lv } else { lf.loop_filter_level as i32 + lv }
        } else {
            lf.loop_filter_level as i32
        };
        let mut flvl = [[0u8; 2]; 4];
        if lf_delta_enabled {
            flvl[0][0] = clip_uint(lflvl + ref_delta[0] as i32 * (1 << sh), 6) as u8;
            flvl[0][1] = flvl[0][0];
            for j in 1..4 {
                flvl[j][0] = clip_uint(lflvl + (ref_delta[j] as i32 + mode_delta[0] as i32) * (1 << sh), 6) as u8;
                flvl[j][1] = clip_uint(lflvl + (ref_delta[j] as i32 + mode_delta[1] as i32) * (1 << sh), 6) as u8;
            }
        } else {
            for j in 0..4 {
                flvl[j][0] = clip_uint(lflvl, 6) as u8;
                flvl[j][1] = flvl[j][0];
            }
        }

        out[i] = SegmentParameterVP9::new(
            &VP9SegmentFlags::new(
                if ref_enabled { 1 } else { 0 },
                sg.feature_data[i][2] as u16,
                if skip_enabled { 1 } else { 0 },
            ),
            flvl,
            VP9_AC_QLOOKUP[bpp_index][qyac as usize] as i16, // luma_ac_quant_scale
            VP9_DC_QLOOKUP[bpp_index][qydc] as i16, // luma_dc_quant_scale
            VP9_AC_QLOOKUP[bpp_index][quvac] as i16, // chroma_ac_quant_scale
            VP9_DC_QLOOKUP[bpp_index][quvdc] as i16, // chroma_dc_quant_scale
        );
    }
    out
}



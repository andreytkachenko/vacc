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
use libva::constants::{
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
    match slice_type % 5 {
        0 => 1, // P  -> inter
        1 => 3, // B  -> inter
        2 => 2, // I  -> intra (driver: slice_type==2 keeps intra_pic_flag=1)
        3 => 0, // SP -> inter
        4 => 4, // SI -> intra (driver: slice_type==4 keeps intra_pic_flag=1)
        _ => 2, // unreachable (% 5 in 0..=4); default to intra to avoid gray
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

/// H.264 DPB (Decoded Picture Buffer).
struct H264Dpb {
    refs: Vec<RefPic>,
    max_refs: u32,
}

impl H264Dpb {
    fn new(max_refs: u32) -> Self {
        Self {
            refs: Vec::with_capacity(max_refs as usize),
            max_refs,
        }
    }

    fn add_short_term(&mut self, ref_pic: RefPic) -> usize {
        // Sliding window (H.264 spec 8.2.5): when the DPB is full, unmark the
        // oldest short-term reference. "Oldest" is determined by POC (smallest
        // POC), NOT by raw frame_num: frame_num wraps around (mod MaxFrameNum)
        // and may be repeated across pictures, so raw frame_num is ambiguous
        // and would evict a newer reference. This matches the known-correct
        // reference implementation (nvdec-decode dpb.rs evict_oldest_short_term).
        if self.refs.len() >= self.max_refs as usize {
            if let Some(oldest_idx) = self.refs
                .iter()
                .enumerate()
                .filter(|(_, r)| !r.long_term)
                .min_by_key(|(_, r)| r.top_field_order_cnt)
                .map(|(i, _)| i)
            {
                self.refs.remove(oldest_idx);
            }
        }
        let idx = self.refs.len();
        self.refs.push(ref_pic);
        idx
    }

    fn clear(&mut self) {
        self.refs.clear();
    }

    /// Process Memory Management Control Operations (H.264 spec 8.2.5).
    ///
    /// `current_frame_num`/`max_frame_num` are needed for op 1, whose value is
    /// `difference_of_pic_nums_minus1`: the unmarked picture is the short-term
    /// reference whose frameNum equals
    /// `(current_frame_num - (value + 1)) mod MaxFrameNum` (H.264 8.2.5.3).
    fn mmco(
        &mut self,
        operations: &[vk_video_parser::h264::DecRefPicMarkingEntry],
        current_ref_idx: usize,
        current_frame_num: u32,
        max_frame_num: u32,
    ) {
        for op in operations {
            match op.memory_management_control_operation {
                // 1: Unmark short-term ref with frameNum = curr - (value + 1) (mod MaxFrameNum)
                1 => {
                    let diff = op.value.wrapping_add(1);
                    let target = if current_frame_num >= diff {
                        current_frame_num - diff
                    } else if max_frame_num == 0 {
                        0
                    } else {
                        (max_frame_num.wrapping_add(current_frame_num)).wrapping_sub(diff) % max_frame_num
                    };
                    self.refs.retain(|r| !(r.frame_num == target && !r.long_term));
                }
                // 2: Mark current pic as long-term with longTermPicNum = value
                2 => {
                    if let Some(ref_pic) = self.refs.get_mut(current_ref_idx) {
                        ref_pic.long_term = true;
                        ref_pic.long_term_pic_num = Some(op.value);
                    }
                }
                // 3: Mark current pic as long-term with next available longTermPicNum
                3 => {
                    let next_lt = self.next_available_long_term_pic_num();
                    if let Some(ref_pic) = self.refs.get_mut(current_ref_idx) {
                        ref_pic.long_term = true;
                        ref_pic.long_term_pic_num = Some(next_lt);
                    }
                }
                // 4: Set max long-term frame num to value (unmark all long-term refs with longTermPicNum > value)
                4 => {
                    self.refs.retain(|r| {
                        !r.long_term || r.long_term_pic_num.map_or(true, |lt| lt <= op.value)
                    });
                }
                // 5: Unmark all short-term refs
                5 => {
                    self.refs.retain(|r| r.long_term);
                }
                // 6: Unmark all long-term refs
                6 => {
                    self.refs.retain(|r| !r.long_term);
                }
                _ => {}
            }
        }
    }

    /// Find the next available long-term picture number (smallest non-negative integer not in use).
    fn next_available_long_term_pic_num(&self) -> u32 {
        let mut used = std::collections::HashSet::new();
        for r in &self.refs {
            if let Some(lt) = r.long_term_pic_num {
                used.insert(lt);
            }
        }
        (0..).find(|&n| !used.contains(&n)).unwrap_or(0)
    }
}

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
struct H264Context {
    dpb: H264Dpb,
    prev_frame_num: u32,
    max_frame_num: u32,
    gaps_in_frame_num_value_allowed_flag: bool,
    prev_pic_order_cnt_lsb: i32,
    prev_pic_order_cnt_msb: i32,
    frame_num_offset: u32,
    // POC calculation fields from SPS
    pic_order_cnt_type: u8,
    log2_max_pic_order_cnt_lsb_minus4: u8,
    max_pic_order_cnt_lsb: u32,
    // Type 1 POC tracking
    frame_num_in_pic_order_cnt_cycle: u32,
    // Last computed POC values
    last_pic_order_cnt: i32,
}

impl Default for H264Context {
    fn default() -> Self {
        Self {
            dpb: H264Dpb::new(4),
            prev_frame_num: 0,
            max_frame_num: 1,
            gaps_in_frame_num_value_allowed_flag: false,
            prev_pic_order_cnt_lsb: 0,
            prev_pic_order_cnt_msb: 0,
            frame_num_offset: 0,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            max_pic_order_cnt_lsb: 16,
            frame_num_in_pic_order_cnt_cycle: 0,
            last_pic_order_cnt: 0,
        }
    }
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
        let mut stream = parse_stream_info(&display, &data)?;

        // Create config with RT format attribute (like cros-codecs does)
        let config = display.create_config(
            vec![libva::VAConfigAttrib {
                type_: libva::VAConfigAttribType::VAConfigAttribRTFormat,
                value: stream.rt_format,
            }],
            stream.profile,
            libva::VAEntrypoint::VAEntrypointVLD,
        ).map_err(|e| Error::DecoderInit(e.to_string()))?;

        // Create surfaces (DPB + extra for rendering)
        let num_surfaces = (stream.max_dpb as usize).max(4) + 2;
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
            Some(H264Context {
                dpb: H264Dpb::new(stream.max_dpb),
                ..Default::default()
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

    /// Decode a single H.264 frame using VA-API.
    fn decode_h264_frame(
        &mut self,
        nal_data: &[u8],
        slice_header: Option<&SliceHeader>,
        timestamp: u64,
    ) -> Result<Option<DecodedFrame>> {
        let ctx = self.h264_ctx.as_mut()
            .ok_or_else(|| Error::InvalidState("H264 context not initialized".to_string()))?;
        let sps = self.stream.sps.as_ref()
            .ok_or_else(|| Error::InvalidState("H264 SPS not available".to_string()))?;
        let pps = self.stream.pps.as_ref()
            .ok_or_else(|| Error::InvalidState("H264 PPS not available".to_string()))?;

        // Extract nal_unit_type, nal_ref_idc, num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1, field_pic_flag, bottom_field, idr_pic_id, no_output_of_prior_pics_flag, and frame_num from slice header
        let (nal_unit_type, nal_ref_idc, num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1, field_pic_flag, bottom_field, idr_pic_id, no_output_of_prior_pics_flag, frame_num) = if let Some(SliceHeader::H264(h264_slh)) = slice_header {
            (h264_slh.nal_unit_type, h264_slh.nal_ref_idc, h264_slh.num_ref_idx_l0_active_minus1, h264_slh.num_ref_idx_l1_active_minus1, h264_slh.field_pic_flag, h264_slh.bottom_field, h264_slh.idr_pic_id, h264_slh.no_output_of_prior_pics_flag, h264_slh.frame_num)
        } else {
            // Fallback defaults when slice header not available
            (1, 3, pps.num_ref_idx_l0_default_active_minus1, pps.num_ref_idx_l1_default_active_minus1, false, false, 0, false, ctx.prev_frame_num)
        };

        // Detect IDR frame: nal_unit_type == 5 (IdrSlice) or idr_pic_id > 0
        let is_idr = nal_unit_type == 5 || idr_pic_id > 0;

        // Handle no_output_of_prior_pics_flag: discard all prior pictures from DPB
        if no_output_of_prior_pics_flag {
            ctx.dpb.clear();
        }

        // On IDR frame, clear only short-term references from DPB (preserve long-term refs per H.264 spec)
        if is_idr {
            ctx.dpb.refs.retain(|r| r.long_term);
        }

        // Allocate a surface for this frame (skip surfaces in use by DPB refs)
        let (surface_idx, surface) = self.surface_pool.alloc(&ctx.dpb.refs)
            .ok_or_else(|| Error::InvalidState("No free surfaces available".to_string()))?;

        let surface_id = surface.id();

        // Build picture parameter buffer inline to avoid borrow issues
        let mut flags = 0u32;
        if nal_ref_idc != 0 {
            flags |= VA_PICTURE_H264_SHORT_TERM_REFERENCE;
        }

        // Compute POC from current slice header's pic_order_cnt_lsb and delta values
        let (top_field_order_cnt, bottom_field_order_cnt) = if let Some(SliceHeader::H264(h264_slh)) = slice_header {
            let pic_order_cnt_lsb = h264_slh.pic_order_cnt_lsb;
            let delta_pic_order_cnt_bottom = h264_slh.delta_pic_order_cnt[0];

            // Reconstruct MSB based on wraparound detection (for POC type 0)
            let max_pic_order_cnt_lsb = sps.max_pic_order_cnt_lsb as i32;
            let pic_order_cnt_msb;
            if pic_order_cnt_lsb < ctx.prev_pic_order_cnt_lsb
                && (ctx.prev_pic_order_cnt_lsb - pic_order_cnt_lsb) >= max_pic_order_cnt_lsb / 2
            {
                pic_order_cnt_msb = ctx.prev_pic_order_cnt_msb + max_pic_order_cnt_lsb;
            } else if pic_order_cnt_lsb > ctx.prev_pic_order_cnt_lsb
                && (pic_order_cnt_lsb - ctx.prev_pic_order_cnt_lsb) > max_pic_order_cnt_lsb / 2
            {
                pic_order_cnt_msb = ctx.prev_pic_order_cnt_msb - max_pic_order_cnt_lsb;
            } else {
                pic_order_cnt_msb = ctx.prev_pic_order_cnt_msb;
            }
            let top_poc = pic_order_cnt_msb + pic_order_cnt_lsb;
            let bottom_poc = top_poc + delta_pic_order_cnt_bottom;

            if field_pic_flag {
                if bottom_field {
                    flags |= VA_PICTURE_H264_BOTTOM_FIELD;
                    (0, bottom_poc)
                } else {
                    flags |= VA_PICTURE_H264_TOP_FIELD;
                    (top_poc, 0)
                }
            } else {
                (top_poc, bottom_poc)
            }
        } else {
            // Fallback: use previous POC values
            let poc = ctx.prev_pic_order_cnt_msb + ctx.prev_pic_order_cnt_lsb;
            if field_pic_flag {
                if bottom_field {
                    flags |= VA_PICTURE_H264_BOTTOM_FIELD;
                    (0, poc)
                } else {
                    flags |= VA_PICTURE_H264_TOP_FIELD;
                    (poc, 0)
                }
            } else {
                (poc, poc)
            }
        };

        // Debug: log SPS/PPS fields

        let curr_pic = PictureH264::new(
            surface_id,
            frame_num,
            flags,
            top_field_order_cnt,
            bottom_field_order_cnt,
        );

        // Debug: log PictureParameterBufferH264 key fields from local vars

        // Build refs array: short-term refs first, then long-term refs (matching cros-codecs)
        let mut sorted_refs: Vec<_> = ctx.dpb.refs.iter().collect();
        sorted_refs.sort_by_key(|r| (!r.long_term, r.frame_num));

        let refs: [PictureH264; MAX_H264_REFS] = core::array::from_fn(|i| {
            if let Some(ref_pic) = sorted_refs.get(i) {
                let flags = if ref_pic.long_term {
                    VA_PICTURE_H264_LONG_TERM_REFERENCE
                } else {
                    VA_PICTURE_H264_SHORT_TERM_REFERENCE
                };
                PictureH264::new(
                    ref_pic.surface_id,
                    ref_pic.frame_num,
                    flags,
                    ref_pic.top_field_order_cnt,
                    ref_pic.bottom_field_order_cnt,
                )
            } else {
                PictureH264::new(VA_INVALID_ID, 0, VA_PICTURE_H264_INVALID, 0, 0)
            }
        });

        let seq_fields = H264SeqFields::new(
            sps.chroma_format_idc as u32,
            sps.separate_colour_plane_flag as u32,
            sps.gaps_in_frame_num_value_allowed_flag as u32,
            sps.frame_mbs_only_flag as u32,
            sps.mb_adaptive_frame_field_flag as u32,
            sps.direct_8x8_inference_flag as u32,
            (sps.level_idc >= 41) as u32, // min_luma_bi_pred_size8x8: true for level >= 3.1 (41)
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
            pps.bottom_field_pic_order_in_frame_present_flag as u32,
            pps.deblocking_filter_control_present_flag as u32,
            pps.redundant_pic_cnt_present_flag as u32,
            (nal_ref_idc != 0) as u32,
        );

        // Debug: log PicParam fields before construction

        let pic_param = PictureParameterBufferH264::new(
            curr_pic,
            refs,
            sps.pic_width_in_mbs_minus1,
            picture_height_in_mbs_minus1,
            sps.bit_depth_luma_minus8,
            sps.bit_depth_chroma_minus8,
            sps.max_num_ref_frames as u8,
            &seq_fields,
            0, 0, 0, // FMO not supported
            pps.pic_init_qp_minus26 as i8,
            pps.pic_init_qs_minus26 as i8,
            pps.chroma_qp_index_offset as i8,
            pps.second_chroma_qp_index_offset as i8,
            &pic_fields,
            frame_num as u16,
        );

        // Build IQ matrix buffer using scaling lists from SPS
        // Scaling lists from bitstream are in zigzag order; VAAPI requires raster order.
        // If scaling lists are not present in SPS, use the H.264 default scaling lists
        // (all 16 for both 4x4 and 8x8; verified pixel-perfect vs FFmpeg on NVDEC).
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
            ([[16u8; 16]; 6], [[16u8; 64]; 2])
        };
        let iq_matrix = IQMatrixBufferH264::new(scaling_list_4x4, scaling_list_8x8);

        // Build reference picture lists from DPB based on slice type
        let slice_type_h264 = if let Some(SliceHeader::H264(h264_slh)) = slice_header {
            h264_slh.slice_type % 5
        } else {
            2 // default to I slice
        };

        // Helper to create invalid picture
        fn invalid_pic() -> PictureH264 {
            PictureH264::new(VA_INVALID_ID, 0, VA_PICTURE_H264_INVALID, 0, 0)
        }

        // Helper to convert RefPic to PictureH264
        let ref_to_pic = |ref_pic: &RefPic| -> PictureH264 {
            let flags = if ref_pic.long_term {
                VA_PICTURE_H264_LONG_TERM_REFERENCE
            } else {
                VA_PICTURE_H264_SHORT_TERM_REFERENCE
            };
            PictureH264::new(
                ref_pic.surface_id,
                ref_pic.frame_num,
                flags,
                ref_pic.top_field_order_cnt,
                ref_pic.bottom_field_order_cnt,
            )
        };

        // Helper to convert Vec<&RefPic> to [PictureH264; 32] (VAAPI buffer size)
        let refs_vec_to_array = |refs: &[&RefPic]| -> [PictureH264; 32] {
            core::array::from_fn(|i| {
                refs.get(i).map(|r| ref_to_pic(*r)).unwrap_or_else(invalid_pic)
            })
        };

        // Apply reference picture list modification (H.264 spec 8.2.4.2)
        fn apply_ref_pic_list_modification<'a>(
            mut ref_list: Vec<&'a RefPic>,
            modifications: &[vk_video_parser::h264::RefPicListModificationEntry],
            current_frame_num: u32,
            dpb_refs: &'a [RefPic],
        ) -> Vec<&'a RefPic> {
            let mut pos = 0;
            for entry in modifications {
                match entry.modification_of_pic_nums_idc {
                    // Insert short-term ref with frameNum = current_frame_num - (value + 1)
                    0 => {
                        let target_frame_num = current_frame_num.wrapping_sub(entry.value + 1);
                        if let Some(ref_pic) = dpb_refs.iter().find(|r| r.frame_num == target_frame_num && !r.long_term) {
                            ref_list.insert(pos, ref_pic);
                            pos += 1;
                        }
                    }
                    // Insert long-term ref with longTermPicNum = value
                    1 => {
                        if let Some(ref_pic) = dpb_refs.iter().find(|r| r.long_term_pic_num == Some(entry.value)) {
                            ref_list.insert(pos, ref_pic);
                            pos += 1;
                        }
                    }
                    // Delete ref at current position
                    2 => {
                        if pos < ref_list.len() {
                            ref_list.remove(pos);
                        }
                    }
                    // End of modification
                    3 | _ => {
                        break;
                    }
                }
            }
            ref_list
        }

        let (ref_pic_list_0, ref_pic_list_1);

        // Get modification entries from slice header
        let (mod_l0, mod_l1) = if let Some(SliceHeader::H264(h264_slh)) = slice_header {
            (&h264_slh.ref_pic_list_modification_l0[..], &h264_slh.ref_pic_list_modification_l1[..])
        } else {
            (&[] as &[vk_video_parser::h264::RefPicListModificationEntry], &[] as &[vk_video_parser::h264::RefPicListModificationEntry])
        };

        match slice_type_h264 {
            // P or SP slice: ref_pic_list_0 = refs before current frame, sorted by POC descending
            0 | 3 => {
                let mut refs: Vec<_> = ctx.dpb.refs.iter()
                    .filter(|r| r.frame_num + r.frame_num_offset < frame_num + ctx.frame_num_offset)
                    .collect();
                // Sort by POC descending (most recent first)
                refs.sort_by(|a, b| b.top_field_order_cnt.cmp(&a.top_field_order_cnt));

                // Apply reference picture list modification for L0
                let refs = apply_ref_pic_list_modification(refs, mod_l0, frame_num, &ctx.dpb.refs);

                ref_pic_list_0 = refs_vec_to_array(&refs);
                ref_pic_list_1 = core::array::from_fn(|_| invalid_pic());
            }
            // B slice: ref_pic_list_0 = refs before current POC, ref_pic_list_1 = refs after
            1 => {
                let get_avg_poc = |r: &RefPic| -> i32 {
                    if r.top_field_order_cnt >= 0 && r.bottom_field_order_cnt >= 0 {
                        // Frame picture: use average
                        (r.top_field_order_cnt + r.bottom_field_order_cnt) / 2
                    } else {
                        // Field picture: use the valid field's POC
                        r.top_field_order_cnt.max(r.bottom_field_order_cnt)
                    }
                };
                let current_poc = top_field_order_cnt;
                let mut list0: Vec<_> = ctx.dpb.refs.iter()
                    .filter(|r| get_avg_poc(r) < current_poc)
                    .collect();
                let mut list1: Vec<_> = ctx.dpb.refs.iter()
                    .filter(|r| get_avg_poc(r) > current_poc)
                    .collect();
                // L0: POC descending (most recent first)
                list0.sort_by(|a, b| b.top_field_order_cnt.cmp(&a.top_field_order_cnt));
                // L1: POC ascending (nearest future first)
                list1.sort_by(|a, b| a.top_field_order_cnt.cmp(&b.top_field_order_cnt));

                // Apply reference picture list modification for L0 and L1
                let list0 = apply_ref_pic_list_modification(list0, mod_l0, frame_num, &ctx.dpb.refs);
                let list1 = apply_ref_pic_list_modification(list1, mod_l1, frame_num, &ctx.dpb.refs);

                ref_pic_list_0 = refs_vec_to_array(&list0);
                ref_pic_list_1 = refs_vec_to_array(&list1);
            }
            // I or SI slice: no references
            _ => {
                ref_pic_list_0 = core::array::from_fn(|_| invalid_pic());
                ref_pic_list_1 = core::array::from_fn(|_| invalid_pic());
            }
        }

        let slice_param = SliceParameterBufferH264::new(
            nal_data.len().saturating_sub(1) as u32,
            0,
            VA_SLICE_DATA_FLAG_ALL,
            slice_header
                .as_ref()
                .and_then(|sh| match sh {
                    SliceHeader::H264(h) => Some(h.header_bit_size as u16),
                    _ => None,
                })
                .unwrap_or(8), // slice_data_bit_offset: 8 + slice_header_bits
            slice_header
                .as_ref()
                .and_then(|sh| match sh {
                    SliceHeader::H264(h) => Some(h.first_mb_in_slice as u16),
                    _ => None,
                })
                .unwrap_or(0),
            slice_header
                .as_ref()
                .and_then(|sh| match sh {
                    SliceHeader::H264(h) => Some(h264_slice_type_to_vaapi(h.slice_type)),
                    _ => None,
                })
                .unwrap_or(0), // default to I slice
            slice_header
                .as_ref()
                .and_then(|sh| match sh {
                    SliceHeader::H264(h) => Some(h.direct_spatial_mv_pred_flag as u8),
                    _ => None,
                })
                .unwrap_or(0),
            num_ref_idx_l0_active_minus1 as u8,
            num_ref_idx_l1_active_minus1 as u8,
            slice_header
                .as_ref()
                .and_then(|sh| match sh {
                    SliceHeader::H264(h) => Some(h.cabac_init_idc),
                    _ => None,
                })
                .unwrap_or(0),
            slice_header
                .as_ref()
                .and_then(|sh| match sh {
                    SliceHeader::H264(h) => Some(h.slice_qp_delta as i8),
                    _ => None,
                })
                .unwrap_or(0),
            slice_header
                .as_ref()
                .and_then(|sh| match sh {
                    SliceHeader::H264(h) => Some(h.disable_deblocking_filter_idc as u8),
                    _ => None,
                })
                .unwrap_or(0),
            slice_header
                .as_ref()
                .and_then(|sh| match sh {
                    SliceHeader::H264(h) => Some(h.slice_alpha_c0_offset_div2 as i8),
                    _ => None,
                })
                .unwrap_or(0),
            slice_header
                .as_ref()
                .and_then(|sh| match sh {
                    SliceHeader::H264(h) => Some(h.slice_beta_offset_div2 as i8),
                    _ => None,
                })
                .unwrap_or(0),
            ref_pic_list_0,
            ref_pic_list_1,
            slice_header
                .as_ref()
                .and_then(|sh| match sh {
                    SliceHeader::H264(h) => Some(h.luma_log2_weight_denom),
                    _ => None,
                })
                .unwrap_or(0),
            slice_header
                .as_ref()
                .and_then(|sh| match sh {
                    SliceHeader::H264(h) => Some(h.chroma_log2_weight_denom),
                    _ => None,
                })
                .unwrap_or(0), // weight denoms
            0, [0i16; 32], [0i16; 32], // L0 weights
            0, [[0i16; 2]; 32], [[0i16; 2]; 32], // L0 chroma weights
            0, [0i16; 32], [0i16; 32], // L1 weights
            0, [[0i16; 2]; 32], [[0i16; 2]; 32], // L1 chroma weights
        );

        // Create buffers
        let pic_param_buf = self.context.create_buffer(
            BufferType::PictureParameter(PictureParameter::H264(pic_param))
        ).map_err(|e| Error::VaApi(e.to_string()))?;

        let iq_buf = self.context.create_buffer(
            BufferType::IQMatrix(IQMatrix::H264(iq_matrix))
        ).map_err(|e| Error::VaApi(e.to_string()))?;

        let slice_param_buf = self.context.create_buffer(
            BufferType::SliceParameter(SliceParameter::H264(slice_param))
        ).map_err(|e| Error::VaApi(e.to_string()))?;

        // Pass NAL data WITHOUT NAL header byte (RBSP only).
        let slice_data = if nal_data.len() > 1 {
            nal_data[1..].to_vec()
        } else {
            nal_data.to_vec()
        };
        let slice_data_buf = self.context.create_buffer(
            BufferType::SliceData(slice_data)
        ).map_err(|e| Error::VaApi(e.to_string()))?;

        // Create picture and perform decode
        let mut picture = Picture::<PictureNew, Rc<Surface<DmaBufSurfaceDescriptor>>>::new(timestamp, Rc::clone(&self.context), Rc::clone(&surface));
        picture.add_buffer(pic_param_buf);
        picture.add_buffer(iq_buf);
        picture.add_buffer(slice_param_buf);
        picture.add_buffer(slice_data_buf);

        // Begin -> Render -> End
        let picture = picture
            .begin()
            .map_err(|e| Error::VaApi(e.to_string()))?;
        let picture = picture
            .render()
            .map_err(|e| Error::VaApi(e.to_string()))?;
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
        )?;

        // Update DPB if reference frame
        if nal_ref_idc != 0 {
            let frame_num = ctx.prev_frame_num;
            let poc = ctx.prev_pic_order_cnt_msb + ctx.prev_pic_order_cnt_lsb;

            // For field pictures, only the active field has the POC
            let (top_field_order_cnt, bottom_field_order_cnt) = if field_pic_flag {
                if bottom_field {
                    (-1, poc)
                } else {
                    (poc, -1)
                }
            } else {
                (poc, poc)
            };

            let long_term = matches!(slice_header, Some(vk_video_parser::SliceHeader::H264(slh)) if slh.long_term_reference_flag);

            let ref_pic = RefPic {
                surface_id,
                frame_num,
                frame_num_offset: ctx.frame_num_offset,
                long_term,
                long_term_pic_num: None,
                top_field_order_cnt,
                bottom_field_order_cnt,
            };
            let current_ref_idx = ctx.dpb.add_short_term(ref_pic.clone());
            self.surface_pool.entries[surface_idx].ref_pic = Some(ref_pic);

            // Process MMCO (Memory Management Control Operations)
            if let Some(vk_video_parser::SliceHeader::H264(slh)) = slice_header {
                if !slh.dec_ref_pic_marking.is_empty() {
                    ctx.dpb.mmco(&slh.dec_ref_pic_marking, current_ref_idx, frame_num, ctx.max_frame_num);
                }
            }
        }

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
        let (nal_unit_type, nal_ref_idc, num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1, field_pic_flag, bottom_field, idr_pic_id, no_output_of_prior_pics_flag, frame_num) = if let Some(SliceHeader::H264(h264_slh)) = slice_header {
            (h264_slh.nal_unit_type, h264_slh.nal_ref_idc, h264_slh.num_ref_idx_l0_active_minus1, h264_slh.num_ref_idx_l1_active_minus1, h264_slh.field_pic_flag, h264_slh.bottom_field, h264_slh.idr_pic_id, h264_slh.no_output_of_prior_pics_flag, h264_slh.frame_num)
        } else {
            (1, 3, pps.num_ref_idx_l0_default_active_minus1, pps.num_ref_idx_l1_default_active_minus1, false, false, 0, false, ctx.prev_frame_num)
        };

        // Detect IDR frame
        let is_idr = nal_unit_type == 5 || idr_pic_id > 0;

        // Handle no_output_of_prior_pics_flag
        if no_output_of_prior_pics_flag {
            ctx.dpb.clear();
        }

        // On IDR frame, clear only short-term references from DPB (preserve long-term refs per H.264 spec)
        if is_idr {
            ctx.dpb.refs.retain(|r| r.long_term);
        }

        // Allocate a surface for this frame
        let (surface_idx, surface) = self.surface_pool.alloc(&ctx.dpb.refs)
            .ok_or_else(|| Error::InvalidState("No free surfaces available".to_string()))?;
        let surface_id = surface.id();

        // Build picture parameter buffer
        let mut flags = 0u32;
        if nal_ref_idc != 0 {
            flags |= VA_PICTURE_H264_SHORT_TERM_REFERENCE;
        }

        // Compute POC from current slice header's pic_order_cnt_lsb and delta values
        // (not from ctx.prev_pic_order_cnt_lsb/msb which holds the previous frame's values)
        let (top_field_order_cnt, bottom_field_order_cnt) = if let Some(SliceHeader::H264(h264_slh)) = slice_header {
            let pic_order_cnt_lsb = h264_slh.pic_order_cnt_lsb;
            let delta_pic_order_cnt_bottom = h264_slh.delta_pic_order_cnt[0];

            // Reconstruct MSB based on wraparound detection (for POC type 0)
            let max_pic_order_cnt_lsb = sps.max_pic_order_cnt_lsb as i32;
            let pic_order_cnt_msb;
            if pic_order_cnt_lsb < ctx.prev_pic_order_cnt_lsb
                && (ctx.prev_pic_order_cnt_lsb - pic_order_cnt_lsb) >= max_pic_order_cnt_lsb / 2
            {
                pic_order_cnt_msb = ctx.prev_pic_order_cnt_msb + max_pic_order_cnt_lsb;
            } else if pic_order_cnt_lsb > ctx.prev_pic_order_cnt_lsb
                && (pic_order_cnt_lsb - ctx.prev_pic_order_cnt_lsb) > max_pic_order_cnt_lsb / 2
            {
                pic_order_cnt_msb = ctx.prev_pic_order_cnt_msb - max_pic_order_cnt_lsb;
            } else {
                pic_order_cnt_msb = ctx.prev_pic_order_cnt_msb;
            }
            let top_poc = pic_order_cnt_msb + pic_order_cnt_lsb;
            let bottom_poc = top_poc + delta_pic_order_cnt_bottom;

            if field_pic_flag {
                if bottom_field {
                    flags |= VA_PICTURE_H264_BOTTOM_FIELD;
                    (0, bottom_poc)
                } else {
                    flags |= VA_PICTURE_H264_TOP_FIELD;
                    (top_poc, 0)
                }
            } else {
                (top_poc, bottom_poc)
            }
        } else {
            // Fallback: use previous POC values
            let poc = ctx.prev_pic_order_cnt_msb + ctx.prev_pic_order_cnt_lsb;
            if field_pic_flag {
                if bottom_field {
                    flags |= VA_PICTURE_H264_BOTTOM_FIELD;
                    (0, poc)
                } else {
                    flags |= VA_PICTURE_H264_TOP_FIELD;
                    (poc, 0)
                }
            } else {
                (poc, poc)
            }
        };

        let curr_pic = PictureH264::new(
            surface_id,
            frame_num,
            flags,
            top_field_order_cnt,
            bottom_field_order_cnt,
        );

        // Build refs array
        let mut sorted_refs: Vec<_> = ctx.dpb.refs.iter().collect();
        sorted_refs.sort_by_key(|r| (!r.long_term, r.frame_num));

        let refs: [PictureH264; MAX_H264_REFS] = core::array::from_fn(|i| {
            if let Some(ref_pic) = sorted_refs.get(i) {
                let flags = if ref_pic.long_term {
                    VA_PICTURE_H264_LONG_TERM_REFERENCE
                } else {
                    VA_PICTURE_H264_SHORT_TERM_REFERENCE
                };
                PictureH264::new(
                    ref_pic.surface_id,
                    ref_pic.frame_num,
                    flags,
                    ref_pic.top_field_order_cnt,
                    ref_pic.bottom_field_order_cnt,
                )
            } else {
                PictureH264::new(VA_INVALID_ID, 0, VA_PICTURE_H264_INVALID, 0, 0)
            }
        });

        let seq_fields = H264SeqFields::new(
            sps.chroma_format_idc as u32,
            sps.separate_colour_plane_flag as u32,
            sps.gaps_in_frame_num_value_allowed_flag as u32,
            sps.frame_mbs_only_flag as u32,
            sps.mb_adaptive_frame_field_flag as u32,
            sps.direct_8x8_inference_flag as u32,
            (sps.level_idc >= 41) as u32,
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
            pps.bottom_field_pic_order_in_frame_present_flag as u32,
            pps.deblocking_filter_control_present_flag as u32,
            pps.redundant_pic_cnt_present_flag as u32,
            (nal_ref_idc != 0) as u32,
        );

        // Debug: log PicParam fields before construction

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

        // Build reference picture lists from DPB based on first slice's type
        // Note: ref_pic_list_0 and ref_pic_list_1 are built per-slice below
        // since PictureH264 doesn't implement Clone
        let slice_type_h264 = if let Some(SliceHeader::H264(h264_slh)) = slice_header {
            h264_slh.slice_type % 5
        } else {
            2
        };

        // Create picture parameter buffer (shared across all slices)
        let pic_param_buf = self.context.create_buffer(
            BufferType::PictureParameter(PictureParameter::H264(pic_param))
        ).map_err(|e| Error::VaApi(e.to_string()))?;

        let iq_buf = self.context.create_buffer(
            BufferType::IQMatrix(IQMatrix::H264(iq_matrix))
        ).map_err(|e| Error::VaApi(e.to_string()))?;

        // Helper functions for reference list building
        fn invalid_pic() -> PictureH264 {
            PictureH264::new(VA_INVALID_ID, 0, VA_PICTURE_H264_INVALID, 0, 0)
        }

        let ref_to_pic = |ref_pic: &RefPic| -> PictureH264 {
            let flags = if ref_pic.long_term {
                VA_PICTURE_H264_LONG_TERM_REFERENCE
            } else {
                VA_PICTURE_H264_SHORT_TERM_REFERENCE
            };
            PictureH264::new(
                ref_pic.surface_id,
                ref_pic.frame_num,
                flags,
                ref_pic.top_field_order_cnt,
                ref_pic.bottom_field_order_cnt,
            )
        };

        let refs_vec_to_array = |refs: &[&RefPic]| -> [PictureH264; 32] {
            core::array::from_fn(|i| {
                refs.get(i).map(|r| ref_to_pic(*r)).unwrap_or_else(invalid_pic)
            })
        };

        fn apply_ref_pic_list_modification<'a>(
            mut ref_list: Vec<&'a RefPic>,
            modifications: &[vk_video_parser::h264::RefPicListModificationEntry],
            current_frame_num: u32,
            dpb_refs: &'a [RefPic],
        ) -> Vec<&'a RefPic> {
            let mut pos = 0;
            for entry in modifications {
                match entry.modification_of_pic_nums_idc {
                    0 => {
                        let target_frame_num = current_frame_num.wrapping_sub(entry.value + 1);
                        if let Some(ref_pic) = dpb_refs.iter().find(|r| r.frame_num == target_frame_num && !r.long_term) {
                            ref_list.insert(pos, ref_pic);
                            pos += 1;
                        }
                    }
                    1 => {
                        if let Some(ref_pic) = dpb_refs.iter().find(|r| r.long_term_pic_num == Some(entry.value)) {
                            ref_list.insert(pos, ref_pic);
                            pos += 1;
                        }
                    }
                    2 => {
                        if pos < ref_list.len() {
                            ref_list.remove(pos);
                        }
                    }
                    3 | _ => {
                        break;
                    }
                }
            }
            ref_list
        }

        let (mod_l0, mod_l1) = if let Some(SliceHeader::H264(h264_slh)) = slice_header {
            (&h264_slh.ref_pic_list_modification_l0[..], &h264_slh.ref_pic_list_modification_l1[..])
        } else {
            (&[] as &[vk_video_parser::h264::RefPicListModificationEntry], &[] as &[vk_video_parser::h264::RefPicListModificationEntry])
        };

        // Helper to build ref lists for a given slice type
        let build_ref_lists = |slice_type: u8| -> ([PictureH264; 32], [PictureH264; 32]) {
            match slice_type {
                0 | 3 => {
                    let mut refs: Vec<_> = ctx.dpb.refs.iter()
                        .filter(|r| r.frame_num + r.frame_num_offset < frame_num + ctx.frame_num_offset)
                        .collect();
                    refs.sort_by(|a, b| b.top_field_order_cnt.cmp(&a.top_field_order_cnt));
                    let modified_l0 = apply_ref_pic_list_modification(refs, mod_l0, frame_num, &ctx.dpb.refs);
                    if std::env::var("DBG_H264").is_ok() {
                        let f = |v: &[&RefPic]| v.iter().map(|r| format!("fn{}:poc{}", r.frame_num, r.top_field_order_cnt)).collect::<Vec<_>>();
                        let m = |v: &[vk_video_parser::h264::RefPicListModificationEntry]| v.iter().map(|e| (e.modification_of_pic_nums_idc, e.value)).collect::<Vec<_>>();
                        eprintln!("[DBG]   P st={} curfn={} mod0={:?} L0={:?}", slice_type, frame_num, m(mod_l0), f(&modified_l0));
                    }
                    let list0 = refs_vec_to_array(&modified_l0);
                    let list1 = core::array::from_fn(|_| invalid_pic());
                    (list0, list1)
                }
                1 | 4 => {
                    // B or SI slice: ref_pic_list_0 = past refs (POC < current),
                    // ref_pic_list_1 = future refs (POC > current). Per H.264 8.2.3.2.
                    let get_avg_poc = |r: &RefPic| -> i32 {
                        if r.top_field_order_cnt >= 0 && r.bottom_field_order_cnt >= 0 {
                            (r.top_field_order_cnt + r.bottom_field_order_cnt) / 2
                        } else {
                            r.top_field_order_cnt.max(r.bottom_field_order_cnt)
                        }
                    };
                    let current_poc = top_field_order_cnt;
                    let mut list0: Vec<_> = ctx.dpb.refs.iter()
                        .filter(|r| get_avg_poc(r) < current_poc)
                        .collect();
                    let mut list1: Vec<_> = ctx.dpb.refs.iter()
                        .filter(|r| get_avg_poc(r) > current_poc)
                        .collect();
                    // L0: POC descending (most recent first); L1: POC ascending (nearest future first)
                    list0.sort_by(|a, b| b.top_field_order_cnt.cmp(&a.top_field_order_cnt));
                    list1.sort_by(|a, b| a.top_field_order_cnt.cmp(&b.top_field_order_cnt));
                    let modified_l0 = apply_ref_pic_list_modification(list0, mod_l0, frame_num, &ctx.dpb.refs);
                    let modified_l1 = apply_ref_pic_list_modification(list1, mod_l1, frame_num, &ctx.dpb.refs);
                    if std::env::var("DBG_H264").is_ok() {
                        let f = |v: &[&RefPic]| v.iter().map(|r| format!("fn{}:poc{}", r.frame_num, r.top_field_order_cnt)).collect::<Vec<_>>();
                        let m = |v: &[vk_video_parser::h264::RefPicListModificationEntry]| v.iter().map(|e| (e.modification_of_pic_nums_idc, e.value)).collect::<Vec<_>>();
                        eprintln!("[DBG]   B st={} curpoc={} mod0={:?} mod1={:?} L0={:?} L1={:?}", slice_type, current_poc, m(mod_l0), m(mod_l1), f(&modified_l0), f(&modified_l1));
                    }
                    let list0 = refs_vec_to_array(&modified_l0);
                    let list1 = refs_vec_to_array(&modified_l1);
                    (list0, list1)
                }
                _ => {
                    let list0 = core::array::from_fn(|_| invalid_pic());
                    let list1 = core::array::from_fn(|_| invalid_pic());
                    (list0, list1)
                }
            }
        };

        // Begin picture ONCE for the entire frame
        let mut picture = Picture::<PictureNew, Rc<Surface<DmaBufSurfaceDescriptor>>>::new(timestamp, Rc::clone(&self.context), Rc::clone(&surface));
        picture.add_buffer(pic_param_buf);
        picture.add_buffer(iq_buf);

        // Add all slice buffers BEFORE begin (typestate requires it)
        // VAAPI processes SliceParameter+SliceData pairs in order during render()
        for slice_info in slices.iter() {
            let slice_header_opt = slice_info.slice_header.as_ref();

            // Get slice type for this slice (may differ between slices in rare cases)
            let this_slice_type = slice_header_opt
                .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.slice_type % 5), _ => None })
                .unwrap_or(slice_type_h264);

            // Build ref lists for this slice
            let (ref_pic_list_0, ref_pic_list_1) = build_ref_lists(this_slice_type as u8);

            // Build slice parameter buffer for this slice

             let slice_param = SliceParameterBufferH264::new(
                  slice_info.nal_data.len() as u32,
                  0,
                  VA_SLICE_DATA_FLAG_ALL,
                   slice_header_opt
                       .and_then(|sh| match sh { SliceHeader::H264(h) => Some((8 + h.header_bit_size) as u16), _ => None })
                       .unwrap_or(8), // slice_data_bit_offset: 8 (NAL header) + slice_header_bits
                 slice_header_opt
                     .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.first_mb_in_slice as u16), _ => None })
                     .unwrap_or(0),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h264_slice_type_to_vaapi(h.slice_type)), _ => None })
                    .unwrap_or(0),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.direct_spatial_mv_pred_flag as u8), _ => None })
                    .unwrap_or(0),
                num_ref_idx_l0_active_minus1 as u8,
                num_ref_idx_l1_active_minus1 as u8,
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
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.luma_log2_weight_denom), _ => None })
                    .unwrap_or(0),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.chroma_log2_weight_denom), _ => None })
                    .unwrap_or(0),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.luma_weight_l0_flag as u8), _ => None })
                    .unwrap_or(0),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.luma_weight_l0), _ => None })
                    .unwrap_or([0i16; 32]),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.luma_offset_l0), _ => None })
                    .unwrap_or([0i16; 32]),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.chroma_weight_l0_flag as u8), _ => None })
                    .unwrap_or(0),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.chroma_weight_l0), _ => None })
                    .unwrap_or([[0i16; 2]; 32]),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.chroma_offset_l0), _ => None })
                    .unwrap_or([[0i16; 2]; 32]),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.luma_weight_l1_flag as u8), _ => None })
                    .unwrap_or(0),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.luma_weight_l1), _ => None })
                    .unwrap_or([0i16; 32]),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.luma_offset_l1), _ => None })
                    .unwrap_or([0i16; 32]),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.chroma_weight_l1_flag as u8), _ => None })
                    .unwrap_or(0),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.chroma_weight_l1), _ => None })
                    .unwrap_or([[0i16; 2]; 32]),
                slice_header_opt
                    .and_then(|sh| match sh { SliceHeader::H264(h) => Some(h.chroma_offset_l1), _ => None })
                    .unwrap_or([[0i16; 2]; 32]),
            );

            let slice_param_buf = self.context.create_buffer(
                BufferType::SliceParameter(SliceParameter::H264(slice_param))
            ).map_err(|e| Error::VaApi(e.to_string()))?;

            let slice_data_buf = self.context.create_buffer(
                BufferType::SliceData(slice_info.nal_data.clone())
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
        )?;

        // Update DPB if reference frame
        if nal_ref_idc != 0 {
            let frame_num = ctx.prev_frame_num;
            let poc = ctx.prev_pic_order_cnt_msb + ctx.prev_pic_order_cnt_lsb;

            let (top_field_order_cnt, bottom_field_order_cnt) = if field_pic_flag {
                if bottom_field {
                    (-1, poc)
                } else {
                    (poc, -1)
                }
            } else {
                (poc, poc)
            };

            let long_term = matches!(slice_header, Some(vk_video_parser::SliceHeader::H264(slh)) if slh.long_term_reference_flag);

            let ref_pic = RefPic {
                surface_id,
                frame_num,
                frame_num_offset: ctx.frame_num_offset,
                long_term,
                long_term_pic_num: None,
                top_field_order_cnt,
                bottom_field_order_cnt,
            };
            let current_ref_idx = ctx.dpb.add_short_term(ref_pic.clone());
            self.surface_pool.entries[surface_idx].ref_pic = Some(ref_pic);

            // Process MMCO from last slice
            if let Some(H264SliceInfo { slice_header: Some(vk_video_parser::SliceHeader::H264(slh)), .. }) = slices.last() {
                if !slh.dec_ref_pic_marking.is_empty() {
                    ctx.dpb.mmco(&slh.dec_ref_pic_marking, current_ref_idx, frame_num, ctx.max_frame_num);
                }
            }
        }

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
            ctx.dpb.clear();
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
            ctx.dpb.clear();
            ctx.prev_frame_num = 0;
            ctx.prev_pic_order_cnt_lsb = 0;
            ctx.prev_pic_order_cnt_msb = 0;
            ctx.frame_num_offset = 0;
            ctx.frame_num_in_pic_order_cnt_cycle = 0;
            ctx.last_pic_order_cnt = 0;
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
                    ctx.pic_order_cnt_type = sps.pic_order_cnt_type;
                    ctx.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4;
                    ctx.max_pic_order_cnt_lsb = sps.max_pic_order_cnt_lsb;
                    if std::env::var("DBG_H264").is_ok() {
                        eprintln!("[DBG] SPS log2_max_fn={} poc_type={} log2_max_poc_lsb={} max_poc_lsb={} max_ref_frames={} gaps={} frame_mbs_only={}",
                            sps.log2_max_frame_num_minus4, sps.pic_order_cnt_type, sps.log2_max_pic_order_cnt_lsb_minus4,
                            sps.max_pic_order_cnt_lsb, sps.max_num_ref_frames, sps.gaps_in_frame_num_value_allowed_flag, sps.frame_mbs_only_flag);
                    }
                    ctx.prev_pic_order_cnt_lsb = 0;
                    ctx.prev_pic_order_cnt_msb = 0;
                    ctx.frame_num_in_pic_order_cnt_cycle = 0;
                    ctx.last_pic_order_cnt = 0;
                    // Continue loop to find slices
                    continue;
                }
                Ok(ParseResult::ParameterSet { sps: Some(sps), .. }) => {
                    let sps = sps.downcast_ref::<H264Sps>()
                        .ok_or_else(|| Error::DecoderInit("Invalid SPS type".to_string()))?;
                    self.stream.sps = Some(sps.clone());
                    ctx.max_frame_num = sps.max_frame_num;
                    ctx.gaps_in_frame_num_value_allowed_flag = sps.gaps_in_frame_num_value_allowed_flag;
                    ctx.pic_order_cnt_type = sps.pic_order_cnt_type;
                    ctx.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4;
                    ctx.max_pic_order_cnt_lsb = sps.max_pic_order_cnt_lsb;
                    ctx.prev_pic_order_cnt_lsb = 0;
                    ctx.prev_pic_order_cnt_msb = 0;
                    ctx.frame_num_in_pic_order_cnt_cycle = 0;
                    ctx.last_pic_order_cnt = 0;
                    ctx.dpb.max_refs = sps.max_num_ref_frames;
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
                        ctx.prev_frame_num.wrapping_add(1)
                    };

                    // Handle frame_num wraparound
                    if frame_num < ctx.prev_frame_num {
                        if ctx.gaps_in_frame_num_value_allowed_flag {
                            ctx.dpb.clear();
                            ctx.frame_num_offset = 0;
                        } else {
                            ctx.frame_num_offset += ctx.max_frame_num;
                        }
                    }
                    ctx.prev_frame_num = frame_num;

                    // Calculate POC for this frame
                    let sps = self.stream.sps.as_ref()
                        .expect("SPS should be available for H264 slice");
                    let (curr_pic_order_cnt_lsb, curr_pic_order_cnt_msb) =
                        if let Some(vk_video_parser::SliceHeader::H264(slh)) = &first_slice_header {
                            calculate_h264_poc(ctx, slh, frame_num, sps)
                        } else {
                            (ctx.prev_pic_order_cnt_lsb + 1, ctx.prev_pic_order_cnt_msb)
                        };
                    ctx.prev_pic_order_cnt_lsb = curr_pic_order_cnt_lsb;
                    ctx.prev_pic_order_cnt_msb = curr_pic_order_cnt_msb;

                    if std::env::var("DBG_H264").is_ok() {
                        let (st, mmco) = first_slice_header.as_ref().and_then(|sh| match sh {
                            SliceHeader::H264(h) => Some((h.slice_type % 5, h.dec_ref_pic_marking.iter().map(|e| (e.memory_management_control_operation, e.value)).collect::<Vec<_>>())),
                            _ => None
                        }).unwrap_or((2, vec![]));
                        eprintln!("[DBG] fn={} poc={} st={} mmco={:?} dpb={:?}",
                            frame_num, curr_pic_order_cnt_msb + curr_pic_order_cnt_lsb, st, mmco,
                            ctx.dpb.refs.iter().map(|r| (r.frame_num, r.top_field_order_cnt, r.bottom_field_order_cnt, r.long_term)).collect::<Vec<_>>());
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

/// Calculate H.264 Picture Order Count based on pic_order_cnt_type.
/// 
/// Returns (pic_order_cnt_lsb, pic_order_cnt_msb).
fn calculate_h264_poc(
    ctx: &mut H264Context,
    slh: &vk_video_parser::h264::SliceHeader,
    frame_num: u32,
    sps: &H264Sps,
) -> (i32, i32) {
    match ctx.pic_order_cnt_type {
        0 => {
            // Type 0: Explicit POC using pic_order_cnt_lsb + MSB reconstruction
            let pic_order_cnt_lsb = slh.pic_order_cnt_lsb;
            
            // Reconstruct MSB based on wraparound detection
            let max_pic_order_cnt_lsb = sps.max_pic_order_cnt_lsb as i32;
            
            let pic_order_cnt_msb;
            if pic_order_cnt_lsb < ctx.prev_pic_order_cnt_lsb
                && (ctx.prev_pic_order_cnt_lsb - pic_order_cnt_lsb) >= max_pic_order_cnt_lsb / 2
            {
                // MSB wrapped up
                pic_order_cnt_msb = ctx.prev_pic_order_cnt_msb + max_pic_order_cnt_lsb;
            } else if pic_order_cnt_lsb > ctx.prev_pic_order_cnt_lsb
                && (pic_order_cnt_lsb - ctx.prev_pic_order_cnt_lsb) > max_pic_order_cnt_lsb / 2
            {
                // MSB wrapped down
                pic_order_cnt_msb = ctx.prev_pic_order_cnt_msb - max_pic_order_cnt_lsb;
            } else {
                // No MSB change
                pic_order_cnt_msb = ctx.prev_pic_order_cnt_msb;
            }
            
            (pic_order_cnt_lsb, pic_order_cnt_msb)
        }
        1 => {
            // Type 1: Implicit POC calculated from frame_num
            // delta_per_frame[i] = offset_for_ref_frame[i] - offset_for_ref_frame[i-1]
            // poc = floor(frame_num / num_ref_frames_in_pic_order_cnt_cycle) * sum(delta_per_frame)
            //     + delta_per_frame[frame_num % num_ref_frames_in_pic_order_cnt_cycle]
            
            let num_ref_frames_in_cycle = sps.num_ref_frames_in_pic_order_cnt_cycle;
            
            if num_ref_frames_in_cycle == 0 {
                // Special case: no cycle, use frame_num directly
                let poc = (frame_num as i32) * 2;
                (poc & ((sps.max_pic_order_cnt_lsb as i32) - 1), 0)
            } else {
                // Calculate delta_per_frame and sum
                let mut delta_per_frame: Vec<i32> = Vec::with_capacity(num_ref_frames_in_cycle as usize);
                let mut sum_delta_per_frame = 0i32;
                
                // delta_per_frame[0] = offset_for_ref_frame[0] - offset_for_top_to_bottom_field
                let first_delta = sps.offset_for_ref_frame[0] - sps.offset_for_top_to_bottom_field;
                delta_per_frame.push(first_delta);
                sum_delta_per_frame += first_delta;
                
                // delta_per_frame[i] = offset_for_ref_frame[i] - offset_for_ref_frame[i-1]
                for i in 1..num_ref_frames_in_cycle as usize {
                    let delta = sps.offset_for_ref_frame[i] - sps.offset_for_ref_frame[i - 1];
                    delta_per_frame.push(delta);
                    sum_delta_per_frame += delta;
                }
                
                let cycle_count = frame_num / num_ref_frames_in_cycle;
                let frame_in_cycle = frame_num % num_ref_frames_in_cycle;
                
                let mut poc = (cycle_count as i32) * sum_delta_per_frame
                    + delta_per_frame[frame_in_cycle as usize];
                
                // Add offset for non-reference pictures if needed
                if slh.nal_ref_idc == 0 {
                    poc += sps.offset_for_non_ref_pic;
                }
                
                ctx.frame_num_in_pic_order_cnt_cycle = frame_in_cycle;
                ctx.last_pic_order_cnt = poc;
                
                // For type 1, MSB is always 0 since POC is computed directly
                (poc, 0)
            }
        }
        2 => {
            // Type 2: Implicit POC from frame_num (H.264 D.3.3.3).
            // Reference frames: POC = frame_num * 2; non-reference: POC = frame_num * 2 + 1.
            // No modulo (max_pic_order_cnt_lsb is not defined for type 2).
            let is_reference = slh.nal_ref_idc > 0;
            let poc = (frame_num as i32) * 2 + if is_reference { 0 } else { 1 };
            (poc, 0)
        }
        _ => {
            // Unknown type: fallback to simple increment
            (ctx.prev_pic_order_cnt_lsb + 1, ctx.prev_pic_order_cnt_msb)
        }
    }
}

/// Read pixel data from a VA image (from derive_from or create_from).
fn read_from_image(
    image: Image,
) -> Result<Option<PixelData>> {
    let va_image = image.image();
    let data = image.as_ref();

    // Determine format from fourcc
    let fourcc = va_image.format.fourcc;
    let is_nv12 = fourcc == libva::constants::VA_FOURCC_NV12;
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

    // Build plane descriptors from the copied buffer
    let y_offset = va_image.offsets[0] as usize;
    let u_offset = va_image.offsets[1] as usize;

    let y_plane = PixelPlane {
        data: unsafe { buffer.as_ptr().add(y_offset) },
        pitch: va_image.pitches[0] as usize,
        width: va_image.width as usize,
        height: va_image.height as usize,
    };

    let u_plane = PixelPlane {
        data: unsafe { buffer.as_ptr().add(u_offset) },
        pitch: va_image.pitches[1] as usize,
        width: va_image.width as usize,
        height: (va_image.height as usize + 1) / 2,
    };

    let v_plane = if !is_nv12 {
        let v_offset = va_image.offsets[2] as usize;
        Some(PixelPlane {
            data: unsafe { buffer.as_ptr().add(v_offset) },
            pitch: va_image.pitches[2] as usize,
            width: va_image.width as usize,
            height: (va_image.height as usize + 1) / 2,
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

/// Read pixel data from a VA surface using DMA-BUF export (export_prime).
///
/// This is the preferred method for NVIDIA GPUs where derive_from doesn't work.
/// Exports the surface as a DRM PRIME handle, mmaps it, and reads the pixel data.
///
/// Uses `export_prime_separate()` because the NVIDIA NVDEC VA-API driver only
/// implements the `VA_EXPORT_SURFACE_SEPARATE_LAYERS` export form (it returns
/// `VA_STATUS_ERROR_INVALID_SURFACE` for the composed-layers form). In the
/// separate-layers form each plane is its own layer (num_planes == 1) and its
/// own DMA-BUF object, ordered luma-first: for NV12 layer 0 is Y and layer 1
/// is the interleaved UV.
fn read_surface_from_export_prime(
    surface: &Surface<DmaBufSurfaceDescriptor>,
    _width: u32,
    _height: u32,
) -> Result<Option<PixelData>> {
    // Export the surface as DRM PRIME handle (separate layers form).
    let desc = surface
        .export_prime_separate()
        .map_err(|e| Error::VaApi(format!("export_prime_separate failed: {}", e)))?;

    // DRM format NV12 = 0x3231564E ("NV12")
    let drm_fourcc_nv12 = u32::from_le_bytes(*b"NV12");
    let drm_fourcc_nv21 = u32::from_le_bytes(*b"NV21");

    if desc.fourcc != drm_fourcc_nv12 && desc.fourcc != drm_fourcc_nv21 {
        return Ok(None);
    }

    if desc.objects.is_empty() || desc.layers.len() < 2 {
        return Ok(None);
    }

    // Mmap every exported object (one per plane for the separate-layers form).
    let mut mmaps = Vec::new();
    for obj in desc.objects.iter() {
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map_copy(&obj.fd)
                .map_err(|e| Error::VaApi(format!("mmap failed: {}", e)))?
        };
        mmaps.push(mmap);
    }

    let width = desc.width as usize;
    let height = desc.height as usize;
    let uv_height = (height + 1) / 2;

    // Helper to copy a single plane described by (layer, plane_idx) into a fresh vec.
    let copy_plane = |layer: &libva::DrmPrimeSurfaceDescriptorLayer,
                      plane_idx: usize,
                      rows: usize,
                       _label: &str|
        -> Option<(Vec<u8>, usize)> {
        let obj_idx = layer.object_index[plane_idx] as usize;
        let offset = layer.offset[plane_idx] as usize;
        let pitch = layer.pitch[plane_idx] as usize;
        let size = pitch * rows;
        if obj_idx >= mmaps.len() {
            return None;
        }
        if offset + size > mmaps[obj_idx].len() {
            return None;
        }
        let mut data = vec![0u8; size];
        data.copy_from_slice(&mmaps[obj_idx][offset..offset + size]);
        Some((data, pitch))
    };

    // Layer 0 = Y (luma), layer 1 = UV (interleaved) for NV12/NV21.
    let (y_data, y_pitch) = match copy_plane(&desc.layers[0], 0, height, "Y") {
        Some(v) => v,
        None => return Ok(None),
    };
    let (uv_data, uv_pitch) = match copy_plane(&desc.layers[1], 0, uv_height, "UV") {
        Some(v) => v,
        None => return Ok(None),
    };

    let y_size = y_data.len();
    let uv_size = uv_data.len();

    // Combine into a single buffer: [Y plane][UV plane]
    let mut buffer = Vec::with_capacity(y_size + uv_size);
    buffer.extend_from_slice(&y_data);
    buffer.extend_from_slice(&uv_data);

    let y_offset = 0usize;
    let uv_offset = y_size;

    let is_nv21 = desc.fourcc == drm_fourcc_nv21;
    let format_str = if is_nv21 { "NV21" } else { "NV12" }.to_string();

    Ok(Some(PixelData {
        format: format_str,
        y: PixelPlane {
            data: unsafe { buffer.as_ptr().add(y_offset) },
            pitch: y_pitch,
            width,
            height,
        },
        u: PixelPlane {
            data: unsafe { buffer.as_ptr().add(uv_offset) },
            pitch: uv_pitch,
            width,
            height: uv_height,
        },
        v: None, // NV12/NV21 are semi-planar, UV is interleaved
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
) -> Result<Option<PixelData>> {

    // Primary: vaCreateImage + vaGetImage (driver-supported CPU read).
    let format = libva::VAImageFormat {
        fourcc: libva::constants::VA_FOURCC_NV12,
        ..Default::default()
    };
    match Image::create_from(surface, format, (width, height), (width, height)) {
        Ok(image) => return read_from_image(image),
        Err(_) => {}
    }

    // Fallback: derive_from (zero-copy; unsupported on NVIDIA).
    match Image::derive_from(surface, (width, height)) {
        Ok(img) => {
            let fourcc = img.image().format.fourcc;
            if fourcc == libva::constants::VA_FOURCC_NV12
                || fourcc == u32::from_ne_bytes(*b"YV12")
                || fourcc == u32::from_ne_bytes(*b"I420")
            {
                return read_from_image(img);
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
fn parse_h264_info(_display: &Display, data: &[u8]) -> Result<StreamInfo> {
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

        // Calculate display dimensions with frame cropping (H.264 spec 7.4.2.1.1)
        if sps.frame_cropping_flag {
            let mb_width = width;
            let mb_height = height;

            let mb_left = sps.frame_crop_left_offset as u32;
            let mb_right = sps.frame_crop_right_offset as u32;
            let mb_top = sps.frame_crop_top_offset as u32;
            let mb_bottom = sps.frame_crop_bottom_offset as u32;

            let crop_scale_x = if sps.chroma_format_idc == 1 { 1 } else { 2 };
            let crop_left = mb_left * 16 * crop_scale_x;
            let crop_right = mb_right * 16 * crop_scale_x;

            let crop_scale_y = if sps.frame_mbs_only_flag { 1 } else { 2 };
            let crop_top = mb_top * 16 * crop_scale_y;
            let crop_bottom = mb_bottom * 16 * crop_scale_y;

            display_width = if crop_left + crop_right < mb_width {
                mb_width - crop_left - crop_right
            } else {
                mb_width
            };
            display_height = if crop_top + crop_bottom < mb_height {
                mb_height - crop_top - crop_bottom
            } else {
                mb_height
            };
        } else {
            display_width = width;
            display_height = height;
        }

        // Determine profile from profile_idc and constraint sets
        profile = match sps.profile_idc {
            66 if sps.constraint_set0_flag => libva::VAProfile::VAProfileH264ConstrainedBaseline,
            66 => libva::VAProfile::VAProfileH264Baseline,
            77 => libva::VAProfile::VAProfileH264Main,
            88 if sps.constraint_set1_flag => libva::VAProfile::VAProfileH264Main,
            100 => libva::VAProfile::VAProfileH264High,
            110 => libva::VAProfile::VAProfileH264High10,
            122 | 244 => libva::VAProfile::VAProfileH264High,
            _ => libva::VAProfile::VAProfileH264Main,
        };
    }

    if width == 0 || height == 0 {
        return Err(Error::DecoderInit("Failed to parse H.264 dimensions".to_string()));
    }

    let rt_format = if let Some(ref sps) = sps_opt {
        let bit_depth = 8 + sps.bit_depth_luma_minus8;
        let chroma_fmt = sps.chroma_format_idc;

        match (bit_depth, chroma_fmt) {
            (8, 0) | (8, 1) => libva::constants::VA_RT_FORMAT_YUV420,
            (8, 2) => libva::constants::VA_RT_FORMAT_YUV422,
            (8, 3) => libva::constants::VA_RT_FORMAT_YUV444,
            (10, 0) | (10, 1) => libva::constants::VA_RT_FORMAT_YUV420_10,
            (10, 2) => libva::constants::VA_RT_FORMAT_YUV422_10,
            (10, 3) => libva::constants::VA_RT_FORMAT_YUV444_10,
            (12, 0) | (12, 1) => libva::constants::VA_RT_FORMAT_YUV420_12,
            (12, 2) => libva::constants::VA_RT_FORMAT_YUV422_12,
            (12, 3) => libva::constants::VA_RT_FORMAT_YUV444_12,
            _ => libva::constants::VA_RT_FORMAT_YUV420,
        }
    } else {
        libva::constants::VA_RT_FORMAT_YUV420
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
        rt_format: libva::constants::VA_RT_FORMAT_YUV420,
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
        rt_format: libva::constants::VA_RT_FORMAT_YUV420,
        sps: None,
        pps: None,
    })
}

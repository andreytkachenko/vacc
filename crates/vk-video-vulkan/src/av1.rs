//! AV1 Vulkan video decoder.
//!
//! Implements AV1 decode command recording using Vulkan Video extension.
//! Aligned with NVIDIA's Vulkan-Video-Samples AV1 decoder (VulkanAV1Decoder.cpp).
//!
//! All StdVideo* structs match vulkan_video_codec_av1std.h and
//! vulkan_video_codec_av1std_decode.h exactly.

use ash::vk;
use ash::vk::Handle;

use super::{VideoError, VideoResult};

// ============================================================================
// AV1 Vulkan constants not in ash 0.38
// ============================================================================

pub mod av1_vk_constants {
    /// VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR
    pub const VIDEO_DECODE_AV1_CAPABILITIES_KHR: i32 = 1000512000;
    /// VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_SESSION_PARAMETERS_CREATE_INFO_KHR
    pub const VIDEO_DECODE_AV1_SESSION_PARAMETERS_CREATE_INFO_KHR: i32 = 1000512004;
    /// VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PICTURE_INFO_KHR
    pub const VIDEO_DECODE_AV1_PICTURE_INFO_KHR: i32 = 1000512001;
    /// VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR
    pub const VIDEO_DECODE_AV1_PROFILE_INFO_KHR: i32 = 1000512003;
    /// VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR
    pub const DECODE_AV1: u32 = 16;
    /// VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR
    ///
    /// NOTE: this is 7 in the Vulkan headers (VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR = 7U),
    /// matching STD_VIDEO_AV1_REFS_PER_FRAME (LAST, LAST2, LAST3, GOLDEN, BWDREF, ALTREF2, ALTREF).
    /// The struct layout of VkVideoDecodeAV1PictureInfoKHR depends on this value — using 8 here
    /// shifts frameHeaderOffset/tileCount/pTileOffsets/pTileSizes by 4 bytes and the driver
    /// reads garbage (this was the all-zero-frames bug).
    pub const MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR: i32 = 7;
}

// ============================================================================
// AV1 StdVideo types (from vulkan_video_codec_av1std.h and
// vulkan_video_codec_av1std_decode.h)
// ============================================================================

/// AV1 Frame type (from vulkan_video_codec_av1std.h).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StdVideoAV1FrameType {
    #[default]
    Key = 0,
    Inter = 1,
    IntraOnly = 2,
    Switch = 3,
}

/// AV1 Interpolation filter (from vulkan_video_codec_av1std.h).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StdVideoAV1InterpolationFilter {
    #[default]
    Eighttap = 0,
    Switchable = 1,
    Proximity = 2,
    Bilinear = 3,
    EighttapSmooth = 4,
    EighttapSharp = 5,
}

/// AV1 Tx mode (from vulkan_video_codec_av1std.h).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StdVideoAV1TxMode {
    #[default]
    Only4x4 = 0,
    Only8x8 = 1,
    Only16x16 = 2,
    Only32x32 = 3,
    Selected = 4,
}

/// AV1 Reference frame names (from vulkan_video_codec_av1std.h).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdVideoAV1RefFrame {
    IntraFrame = 0,
    LastFrame = 1,
    Last2Frame = 2,
    Last3Frame = 3,
    GbilearnFrame = 4,
    GoldenFrame = 5,
    BwdrefFrame = 6,
    Altref2Frame = 7,
    AltrefFrame = 8,
}

/// AV1 Tile info (from vulkan_video_codec_av1std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoAV1TileInfo {
    pub tile_rows_log2_minus1: u8,
    pub tile_cols_log2_minus1: u8,
    pub context_update_tile_id: u8,
    pub _marker: std::marker::PhantomData<()>,
}

/// AV1 Quantization (from vulkan_video_codec_av1std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoAV1Quantization {
    pub delta_q_y_ac: i8,
    pub delta_q_uv_ac: i8,
    pub delta_q_uv_dc: i8,
    pub base_q_idx: u8,
    pub using_qmatrix: u8,
    pub qm_y: u8,
    pub qm_uv: u8,
}

/// AV1 Segmentation flags (from vulkan_video_codec_av1std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoAV1SegmentationFlags {
    pub segmentation_update_map: u32,
    pub segmentation_temporal_update: u32,
    pub segmentation_update_data: u32,
    pub segmentation_abs_or_delta_update: u32,
}

/// AV1 Segmentation (from vulkan_video_codec_av1std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoAV1Segmentation {
    pub flags: StdVideoAV1SegmentationFlags,
    pub tree_probs: [u16; 7],
    pub pred_probs: [u16; 3],
    pub feature_enabled: [u8; 8],
    pub feature_data: [[i16; 4]; 8],
}

/// AV1 Loop filter flags (from vulkan_video_codec_av1std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoAV1LoopFilterFlags {
    pub mode_ref_delta_enabled: u32,
    pub mode_ref_delta_update: u32,
}

/// AV1 Loop filter (from vulkan_video_codec_av1std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoAV1LoopFilter {
    pub flags: StdVideoAV1LoopFilterFlags,
    pub loop_filter_level: [u8; 2],
    pub loop_filter_sharpness: u8,
    pub log2_tile_size: u8,
    pub delta_lf_from_base: [i8; 4],
    pub delta_lf_ref_deltas: [i8; 8],
    pub delta_lf_mode_deltas: [i8; 2],
}

/// AV1 CDEF (from vulkan_video_codec_av1std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoAV1CDEF {
    pub cdef_damping: u8,
    pub cdef_bits: u8,
    pub y_pri_strength: [u8; 8],
    pub y_sec_strength: [u8; 8],
    pub uv_pri_strength: [u8; 8],
    pub uv_sec_strength: [u8; 8],
}

/// AV1 Loop restoration flags (from vulkan_video_codec_av1std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoAV1LoopRestorationFlags {
    pub y_frame_restoration_type: u32,
    pub uv_frame_restoration_type: u32,
}

/// AV1 Loop restoration unit info (from vulkan_video_codec_av1std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoAV1LoopRestorationUnitInfo {
    pub restoration_left_edge: i16,
    pub restoration_right_edge: i16,
    pub y_ac_fn_type: u8,
    pub uv_ac_fn_type: u8,
    pub y_ac_level: u8,
    pub uv_ac_level: u8,
}

/// AV1 Loop restoration (from vulkan_video_codec_av1std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoAV1LoopRestoration {
    pub flags: StdVideoAV1LoopRestorationFlags,
    pub loop_restoration_unit_info: [StdVideoAV1LoopRestorationUnitInfo; 1],
}

/// AV1 Global motion (from vulkan_video_codec_av1std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoAV1GlobalMotion {
    pub gmodel_type: u8,
    pub _marker: std::marker::PhantomData<()>,
}

/// AV1 Global motion params (from vulkan_video_codec_av1std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoAV1GlobalMotionParams {
    pub global_motion_param: [i32; 6],
}

/// AV1 Film grain (from vulkan_video_codec_av1std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoAV1FilmGrain {
    pub key_frame_film_grain: u8,
    pub apply_grain: u8,
    pub chroma_scaling_from_luma: u8,
    pub ar_coeff_lag: u8,
    pub ar_coeff_shift_minus8: u8,
    pub grain_scale_shift: u8,
    pub clip_to_restricted_range: u8,
    pub overlap_flag: u8,
    pub grain_scaling_minus8: u8,
    pub ar_coeff_shift: u8,
    pub grain_scale: u16,
    pub random_seed_value: u16,
    pub num_y_points: u8,
    pub num_cb_points: u8,
    pub num_cr_points: u8,
    pub scaling_shift_minus8: u8,
}

/// AV1 Decode picture info flags (from vulkan_video_codec_av1std_decode.h).
///
/// Bitfield packed into a single u32.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StdVideoDecodeAV1PictureInfoFlags {
    pub bits: u32,
}

impl StdVideoDecodeAV1PictureInfoFlags {
    pub fn new() -> Self {
        Self { bits: 0 }
    }

    // Bit positions per vulkan_video_codec_av1std_decode.h
    pub fn set_error_resilient_mode(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 0)) | ((val & 1) << 0);
    }
    pub fn set_disable_cdf_update(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 1)) | ((val & 1) << 1);
    }
    pub fn set_use_superres(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 2)) | ((val & 1) << 2);
    }
    pub fn set_render_and_frame_size_different(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 3)) | ((val & 1) << 3);
    }
    pub fn set_allow_screen_content_tools(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 4)) | ((val & 1) << 4);
    }
    pub fn set_is_filter_switchable(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 5)) | ((val & 1) << 5);
    }
    pub fn set_force_integer_mv(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 6)) | ((val & 1) << 6);
    }
    pub fn set_frame_size_override_flag(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 7)) | ((val & 1) << 7);
    }
    pub fn set_buffer_removal_time_present_flag(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 8)) | ((val & 1) << 8);
    }
    pub fn set_allow_intrabc(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 9)) | ((val & 1) << 9);
    }
    pub fn set_frame_refs_short_signaling(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 10)) | ((val & 1) << 10);
    }
    pub fn set_allow_high_precision_mv(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 11)) | ((val & 1) << 11);
    }
    pub fn set_is_motion_mode_switchable(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 12)) | ((val & 1) << 12);
    }
    pub fn set_use_ref_frame_mvs(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 13)) | ((val & 1) << 13);
    }
    pub fn set_disable_frame_end_update_cdf(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 14)) | ((val & 1) << 14);
    }
    pub fn set_allow_warped_motion(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 15)) | ((val & 1) << 15);
    }
    pub fn set_reduced_tx_set(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 16)) | ((val & 1) << 16);
    }
    pub fn set_reference_select(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 17)) | ((val & 1) << 17);
    }
    pub fn set_skip_mode_present(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 18)) | ((val & 1) << 18);
    }
    pub fn set_delta_q_present(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 19)) | ((val & 1) << 19);
    }
    pub fn set_delta_lf_present(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 20)) | ((val & 1) << 20);
    }
    pub fn set_delta_lf_multi(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 21)) | ((val & 1) << 21);
    }
    pub fn set_segmentation_enabled(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 22)) | ((val & 1) << 22);
    }
    pub fn set_segmentation_update_map(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 23)) | ((val & 1) << 23);
    }
    pub fn set_segmentation_temporal_update(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 24)) | ((val & 1) << 24);
    }
    pub fn set_segmentation_update_data(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 25)) | ((val & 1) << 25);
    }
    pub fn set_uses_lr(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 26)) | ((val & 1) << 26);
    }
    pub fn set_uses_chroma_lr(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 27)) | ((val & 1) << 27);
    }
    pub fn set_apply_grain(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 28)) | ((val & 1) << 28);
    }
}

impl Default for StdVideoDecodeAV1PictureInfoFlags {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// AV1 Vulkan types (not in ash 0.38, defined manually)
// ============================================================================

/// AV1 Decode picture info for Vulkan Video.
///
/// Matches `VkVideoDecodeAV1PictureInfoKHR` from Vulkan spec exactly.
/// SType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PICTURE_INFO_KHR = 1000512001
#[repr(C)]
#[derive(Debug)]
pub struct VideoDecodeAV1PictureInfoKHR {
    pub s_type: vk::StructureType,
    pub p_next: *const std::os::raw::c_void,
    pub p_std_picture_info: *const ash::vk::native::StdVideoDecodeAV1PictureInfo,
    /// Maps AV1 reference names to DPB slot indices.
    /// Indexed by reference name per the Vulkan spec:
    /// [0]=LAST, [1]=LAST2, [2]=LAST3, [3]=GOLDEN, [4]=BWDREF, [5]=ALTREF2, [6]=ALTREF.
    /// Each value must equal the slotIndex of one of the reference slots passed to
    /// vkCmdDecodeVideoKHR (VUID-vkCmdDecodeVideoKHR-referenceNameSlotIndices-09262),
    /// or be negative if the reference name is not used.
    pub reference_name_slot_indices: [i32; av1_vk_constants::MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR as usize],
    pub frame_header_offset: u32,
    pub tile_count: u32,
    pub p_tile_offsets: *const u32,
    pub p_tile_sizes: *const u32,
    _marker: [u8; 0],
}

impl VideoDecodeAV1PictureInfoKHR {
    /// Create a new AV1 picture info struct for Vulkan Video.
    pub fn new(
        p_std_picture_info: *const ash::vk::native::StdVideoDecodeAV1PictureInfo,
        reference_name_slot_indices: [i32; 7],
        frame_header_offset: u32,
        tile_count: u32,
        p_tile_offsets: *const u32,
        p_tile_sizes: *const u32,
    ) -> Self {
        Self {
            s_type: vk::StructureType::from_raw(av1_vk_constants::VIDEO_DECODE_AV1_PICTURE_INFO_KHR),
            p_next: std::ptr::null(),
            p_std_picture_info,
            reference_name_slot_indices,
            frame_header_offset,
            tile_count,
            p_tile_offsets,
            p_tile_sizes,
            _marker: [],
        }
    }
}

// ============================================================================
// AV1 Picture Info Container
// ============================================================================

/// Container for AV1 picture info and its referenced sub-structures.
///
/// Holds the picture info and all sub-structures (tile info, quantization,
/// segmentation, loop filter, CDEF, loop restoration, global motion, film grain)
/// as a single stack-allocated value. This avoids memory leaks and ensures
/// all pointers remain valid during command buffer execution.
#[repr(C)]
pub struct Av1PictureInfoContainer {
    /// Must come first so &container as pointer to StdVideoDecodeAV1PictureInfo works
    pub std_picture_info: ash::vk::native::StdVideoDecodeAV1PictureInfo,
    pub tile_info: ash::vk::native::StdVideoAV1TileInfo,
    pub quantization: ash::vk::native::StdVideoAV1Quantization,
    pub segmentation: ash::vk::native::StdVideoAV1Segmentation,
    pub loop_filter: ash::vk::native::StdVideoAV1LoopFilter,
    pub cdef: ash::vk::native::StdVideoAV1CDEF,
    pub loop_restoration: ash::vk::native::StdVideoAV1LoopRestoration,
    /// Single struct; internally holds GmType[8] + gm_params[8][6] (index 0 = identity).
    pub global_motion: ash::vk::native::StdVideoAV1GlobalMotion,
    /// Film grain parameters. The Vulkan spec requires pFilmGrain to be a valid
    /// pointer; the C++ reference always sets it (zeroed when grain is absent).
    pub film_grain: ash::vk::native::StdVideoAV1FilmGrain,
    /// Tile data offsets relative to the start of the bitstream buffer.
    /// The C++ reference always sets tileCount=1 with tileOffsets[0]/tileSizes[0],
    /// even for single-tile frames; without them the NVIDIA driver decodes nothing.
    pub tile_offsets: [u32; 1],
    /// Tile data sizes in bytes.
    pub tile_sizes: [u32; 1],
    /// Tile sub-pointer arrays (StdVideoAV1TileInfo). The C++ reference always
    /// sets these non-null; null pointers made the NVIDIA driver skip the decode.
    pub tile_width_in_sbs_minus_1: Vec<u16>,
    pub tile_height_in_sbs_minus_1: Vec<u16>,
    pub tile_mi_col_starts: Vec<u16>,
    pub tile_mi_row_starts: Vec<u16>,
}

impl Av1PictureInfoContainer {
    /// Initialize the pointer fields to point to the container's own sub-structures.
    pub fn init_pointers(&mut self) {
        // Use mutable reference to set pointers
        self.std_picture_info.pTileInfo = &self.tile_info as *const _;
        self.std_picture_info.pQuantization = &self.quantization as *const _;
        self.std_picture_info.pSegmentation = &self.segmentation as *const _;
        self.std_picture_info.pLoopFilter = &self.loop_filter as *const _;
        self.std_picture_info.pCDEF = &self.cdef as *const _;
        self.std_picture_info.pLoopRestoration = &self.loop_restoration as *const _;
        self.std_picture_info.pGlobalMotion = &self.global_motion as *const _;
        self.std_picture_info.pFilmGrain = &self.film_grain as *const _;
        // Tile sub-pointers: the C++ reference always sets these non-null
        // (VulkanVideoParser.cpp:2549-2552). Null pointers made the NVIDIA
        // driver skip the decode.
        self.tile_info.pWidthInSbsMinus1 = if self.tile_width_in_sbs_minus_1.is_empty() {
            std::ptr::null()
        } else {
            self.tile_width_in_sbs_minus_1.as_ptr()
        };
        self.tile_info.pHeightInSbsMinus1 = if self.tile_height_in_sbs_minus_1.is_empty() {
            std::ptr::null()
        } else {
            self.tile_height_in_sbs_minus_1.as_ptr()
        };
        self.tile_info.pMiColStarts = if self.tile_mi_col_starts.is_empty() {
            std::ptr::null()
        } else {
            self.tile_mi_col_starts.as_ptr()
        };
        self.tile_info.pMiRowStarts = if self.tile_mi_row_starts.is_empty() {
            std::ptr::null()
        } else {
            self.tile_mi_row_starts.as_ptr()
        };
    }

    /// Get a pointer to the StdVideoDecodeAV1PictureInfo within this container.
    pub fn std_picture_info(&self) -> *const ash::vk::native::StdVideoDecodeAV1PictureInfo {
        &self.std_picture_info
    }
}

impl Default for Av1PictureInfoContainer {
    fn default() -> Self {
        Self {
            std_picture_info: unsafe { std::mem::zeroed() },
            tile_info: unsafe { std::mem::zeroed() },
            quantization: unsafe { std::mem::zeroed() },
            segmentation: unsafe { std::mem::zeroed() },
            loop_filter: unsafe { std::mem::zeroed() },
            cdef: unsafe { std::mem::zeroed() },
            loop_restoration: unsafe { std::mem::zeroed() },
            global_motion: unsafe { std::mem::zeroed() },
            film_grain: unsafe { std::mem::zeroed() },
            tile_offsets: [0; 1],
            tile_sizes: [0; 1],
            tile_width_in_sbs_minus_1: Vec::new(),
            tile_height_in_sbs_minus_1: Vec::new(),
            tile_mi_col_starts: Vec::new(),
            tile_mi_row_starts: Vec::new(),
        }
    }
}

// ============================================================================
// AV1 SPS -> StdVideo conversion (for video session parameters)
// ============================================================================

/// Convert our Av1Sps to StdVideoAV1ColorConfig.
pub fn convert_av1_color_config(
    sps: &vk_video_core::picture::Av1Sps,
) -> ash::vk::native::StdVideoAV1ColorConfig {
    let mut flags =
        unsafe { std::mem::zeroed::<ash::vk::native::StdVideoAV1ColorConfigFlags>() };
    flags.set_mono_chrome(if sps.mono_chrome { 1 } else { 0 });
    flags.set_color_range(if sps.color_range { 1 } else { 0 });
    flags.set_separate_uv_delta_q(if sps.separate_uv_delta_q { 1 } else { 0 });
    flags.set_color_description_present_flag(if sps.color_description_present { 1 } else { 0 });

    ash::vk::native::StdVideoAV1ColorConfig {
        flags,
        BitDepth: if sps.high_bitdepth {
            if sps.twelve_bit { 12 } else { 10 }
        } else {
            8
        },
        subsampling_x: sps.subsampling_x,
        subsampling_y: sps.subsampling_y,
        reserved1: 0,
        color_primaries: sps.color_primaries as u32,
        transfer_characteristics: sps.transfer_characteristics as u32,
        matrix_coefficients: sps.matrix_coefficients as u32,
        chroma_sample_position: sps.chroma_sample_position as u32,
    }
}

/// Convert our Av1Sps to StdVideoAV1TimingInfo.
pub fn convert_av1_timing_info(
    sps: &vk_video_core::picture::Av1Sps,
) -> ash::vk::native::StdVideoAV1TimingInfo {
    let mut flags = unsafe { std::mem::zeroed::<ash::vk::native::StdVideoAV1TimingInfoFlags>() };
    flags.set_equal_picture_interval(if sps.equal_picture_interval { 1 } else { 0 });

    ash::vk::native::StdVideoAV1TimingInfo {
        flags,
        num_units_in_display_tick: sps.num_units_in_display_tick,
        time_scale: sps.time_scale,
        num_ticks_per_picture_minus_1: 0,
    }
}

/// Convert our Av1Sps to StdVideoAV1SequenceHeader.
///
/// Note: `pColorConfig` and `pTimingInfo` are left null here. The caller
/// (`VideoSessionParameters::create`) must point them at stack-allocated
/// `StdVideoAV1ColorConfig` / `StdVideoAV1TimingInfo` values that outlive the
/// Vulkan calls (see the H264/H265 pattern in that function).
pub fn convert_av1_sps(
    sps: &vk_video_core::picture::Av1Sps,
) -> ash::vk::native::StdVideoAV1SequenceHeader {
    let mut flags =
        unsafe { std::mem::zeroed::<ash::vk::native::StdVideoAV1SequenceHeaderFlags>() };
    flags.set_still_picture(if sps.still_picture { 1 } else { 0 });
    flags.set_reduced_still_picture_header(if sps.reduced_still_picture_header { 1 } else { 0 });
    flags.set_use_128x128_superblock(if sps.use_128x128_superblock { 1 } else { 0 });
    flags.set_enable_filter_intra(if sps.enable_filter_intra { 1 } else { 0 });
    flags.set_enable_intra_edge_filter(if sps.enable_intra_edge_filter { 1 } else { 0 });
    flags.set_enable_interintra_compound(if sps.enable_interintra_compound { 1 } else { 0 });
    flags.set_enable_masked_compound(if sps.enable_masked_compound { 1 } else { 0 });
    flags.set_enable_warped_motion(if sps.enable_warped_motion { 1 } else { 0 });
    flags.set_enable_dual_filter(if sps.enable_dual_filter { 1 } else { 0 });
    flags.set_enable_order_hint(if sps.enable_order_hint { 1 } else { 0 });
    flags.set_enable_jnt_comp(if sps.enable_jnt_motion { 1 } else { 0 });
    flags.set_enable_ref_frame_mvs(if sps.enable_ref_frame_mvs { 1 } else { 0 });
    flags.set_frame_id_numbers_present_flag(if sps.frame_id_numbers_present_flag { 1 } else { 0 });
    flags.set_enable_superres(if sps.enable_superres { 1 } else { 0 });
    flags.set_enable_cdef(if sps.enable_cdef { 1 } else { 0 });
    flags.set_enable_restoration(if sps.enable_restoration { 1 } else { 0 });
    flags.set_film_grain_params_present(if sps.film_grain_params_present { 1 } else { 0 });
    flags.set_timing_info_present_flag(if sps.timing_info_present_flag { 1 } else { 0 });
    flags.set_initial_display_delay_present_flag(if sps.initial_display_delay_present_flag {
        1
    } else {
        0
    });

    // Map profile (0/1/2) to StdVideoAV1Profile.
    let seq_profile = match sps.profile {
        0 => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN,
        1 => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_HIGH,
        2 => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_PROFESSIONAL,
        _ => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN,
    };

    ash::vk::native::StdVideoAV1SequenceHeader {
        flags,
        seq_profile,
        // Av1Sps.frame_width_bits/height_bits store the actual bit count (the
        // parser adds 1 to the raw "minus 1" value), so subtract 1 to recover
        // the "minus 1" values expected by StdVideoAV1SequenceHeader.
        frame_width_bits_minus_1: sps.frame_width_bits.saturating_sub(1),
        frame_height_bits_minus_1: sps.frame_height_bits.saturating_sub(1),
        max_frame_width_minus_1: sps.max_frame_width_minus_1,
        max_frame_height_minus_1: sps.max_frame_height_minus_1,
        delta_frame_id_length_minus_2: sps.delta_frame_id_length_minus2,
        additional_frame_id_length_minus_1: sps.additional_frame_id_length_minus1,
        order_hint_bits_minus_1: sps.order_hint_bits_minus1,
        seq_force_integer_mv: sps.seq_force_integer_mv,
        seq_force_screen_content_tools: sps.seq_force_screen_content_tools,
        reserved1: [0; 5],
        pColorConfig: std::ptr::null(),
        pTimingInfo: std::ptr::null(),
    }
}

// ============================================================================
// AV1 Decoder
// ============================================================================

/// AV1 decoder state.
pub struct Av1Decoder {
    device: ash::Device,
    instance: ash::Instance,
    /// Session handle.
    session: vk::VideoSessionKHR,
    /// Frame buffer to DPB slot mapping.
    /// Maps AV1 frame buffer indices (0-7) to DPB slot indices.
    /// -1 means the frame buffer is not currently assigned to any DPB slot.
    frame_buffer_to_dpb_slot: [i32; 8],
    /// Order hints for each frame buffer (for reference frame management).
    frame_buffer_order_hint: [u32; 8],
    /// Coded dimensions (width, height) of the frame currently stored in each
    /// frame buffer. (0, 0) means the frame buffer has not been refreshed yet.
    /// Needed for show_existing_frame, whose frame header does not carry size.
    frame_buffer_dims: [(u32, u32); 8],
    /// Per-frame-buffer reference info for the VkVideoDecodeAV1DpbSlotInfoKHR
    /// pNext chain (C++ VulkanAV1Decoder.cpp:323-334 populates these; leaving
    /// them zero made INTER frames decode wrong). SavedOrderHints[ref_name] is
    /// the refreshing frame's OrderHints[ref_name]. ref_dist[ref_name] stores
    /// the RAW signed distance GetRelativeDist(cur_OH, OrderHints[ref_name])
    /// (C++ m_pBuffers[i].RefFrameSignBias[ref_name], initialized to 0, index 0
    /// never set); the RefFrameSignBias bitmask bit ref_name is set when
    /// ref_dist[ref_name] <= 0 (so an unrefreshed buffer, dist=0, yields all
    /// bits set — matching C++ exactly).
    frame_buffer_saved_order_hints: [[u8; 8]; 8],
    frame_buffer_ref_dist: [[i8; 8]; 8],
    frame_buffer_frame_type: [u8; 8],
    frame_buffer_disable_cdf: [u8; 8],
    frame_buffer_seg_enabled: [u8; 8],
    /// Frame counter.
    frame_count: u32,
}

impl Av1Decoder {
    pub fn new(device: ash::Device, instance: ash::Instance) -> Self {
        Self {
            device,
            instance,
            session: vk::VideoSessionKHR::null(),
            frame_count: 0,
            frame_buffer_to_dpb_slot: [-1; 8],
            frame_buffer_order_hint: [0; 8],
            frame_buffer_dims: [(0, 0); 8],
            frame_buffer_saved_order_hints: [[0; 8]; 8],
            frame_buffer_ref_dist: [[0; 8]; 8],
            frame_buffer_frame_type: [0; 8],
            frame_buffer_disable_cdf: [0; 8],
            frame_buffer_seg_enabled: [0; 8],
        }
    }

    /// Set the session handle.
    pub fn set_session(&mut self, session: &super::session::VideoSession) {
        self.session = session.handle();
    }

    /// Get the DPB slot index for an AV1 frame buffer index.
    pub fn get_pic_idx_for_frame_buffer(&self, frame_buffer_idx: usize) -> i32 {
        if frame_buffer_idx < 8 {
            self.frame_buffer_to_dpb_slot[frame_buffer_idx]
        } else {
            -1
        }
    }

    /// Set the DPB slot for an AV1 frame buffer.
    pub fn set_frame_buffer_dpb_slot(&mut self, frame_buffer_idx: usize, dpb_slot: i32) {
        if frame_buffer_idx < 8 {
            self.frame_buffer_to_dpb_slot[frame_buffer_idx] = dpb_slot;
        }
    }

    /// Set the order hint for a frame buffer.
    pub fn set_frame_buffer_order_hint(&mut self, frame_buffer_idx: usize, order_hint: u32) {
        if frame_buffer_idx < 8 {
            self.frame_buffer_order_hint[frame_buffer_idx] = order_hint;
        }
    }

    /// Get the order hint for a frame buffer.
    pub fn get_frame_buffer_order_hint(&self, frame_buffer_idx: usize) -> u32 {
        if frame_buffer_idx < 8 {
            self.frame_buffer_order_hint[frame_buffer_idx]
        } else {
            0
        }
    }

    /// Record the coded dimensions of the frame stored in a frame buffer.
    pub fn set_frame_buffer_dims(&mut self, frame_buffer_idx: usize, width: u32, height: u32) {
        if frame_buffer_idx < 8 {
            self.frame_buffer_dims[frame_buffer_idx] = (width, height);
        }
    }

    /// Get the coded dimensions of the frame stored in a frame buffer.
    /// Returns (0, 0) if the frame buffer has not been refreshed.
    pub fn get_frame_buffer_dims(&self, frame_buffer_idx: usize) -> (u32, u32) {
        if frame_buffer_idx < 8 {
            self.frame_buffer_dims[frame_buffer_idx]
        } else {
            (0, 0)
        }
    }

    /// Record the reference info for a frame buffer when it is refreshed by the
    /// current frame (C++ VulkanAV1Decoder.cpp:390-394). `order_hints` is the
    /// CURRENT (refreshing) frame's OrderHints array, `current_order_hint` its
    /// OrderHint, `ohb` = order_hint_bits_minus_1.
    pub fn set_frame_buffer_ref_info(
        &mut self,
        frame_buffer_idx: usize,
        order_hints: &[u8; 8],
        current_order_hint: u32,
        ohb: u32,
        frame_type: u8,
        disable_cdf: u8,
        seg_enabled: u8,
    ) {
          if frame_buffer_idx < 8 {
              self.frame_buffer_saved_order_hints[frame_buffer_idx] = *order_hints;
              // C++ VulkanAV1Decoder.cpp UpdateFramePointers loops refName =
              // LAST_FRAME(1) .. NUM_REF_FRAMES-1(7) and stores the RAW signed
              // distance (m_pBuffers[i].RefFrameSignBias[refName] =
              // GetRelativeDist(pStd->OrderHint, pStd->OrderHints[refName])).
              // Index 0 (INTRA) is never set (stays 0). The RefFrameSignBias
              // bitmask is computed at read time as (dist <= 0).
              for ref_name in 1..8usize {
                let rel = Self::get_relative_dist(
                    current_order_hint as i32,
                    order_hints[ref_name] as i32,
                    ohb,
                );
                self.frame_buffer_ref_dist[frame_buffer_idx][ref_name] = rel as i8;
            }
            self.frame_buffer_frame_type[frame_buffer_idx] = frame_type;
            self.frame_buffer_disable_cdf[frame_buffer_idx] = disable_cdf;
            self.frame_buffer_seg_enabled[frame_buffer_idx] = seg_enabled;
        }
    }

    /// Get the stored reference info for a frame buffer.
    pub fn get_frame_buffer_ref_info(
        &self,
        frame_buffer_idx: usize,
    ) -> Option<(&[u8; 8], u8, u8, u8, u8)> {
        if frame_buffer_idx < 8 {
            // Compute the RefFrameSignBias bitmask from the raw distances
            // (C++ VulkanAV1Decoder.cpp:331-333): bit ref_name (1..7) set when
            // ref_dist[ref_name] <= 0. Bit 0 (INTRA) is never set for a ref slot.
            let mut bias = 0u8;
            for ref_name in 1..8usize {
                if self.frame_buffer_ref_dist[frame_buffer_idx][ref_name] <= 0 {
                    bias |= 1 << ref_name;
                }
            }
            Some((
                &self.frame_buffer_saved_order_hints[frame_buffer_idx],
                bias,
                self.frame_buffer_frame_type[frame_buffer_idx],
                self.frame_buffer_disable_cdf[frame_buffer_idx],
                self.frame_buffer_seg_enabled[frame_buffer_idx],
            ))
        } else {
            None
        }
    }

    /// Find the first frame buffer currently mapped to a given DPB slot.
    pub fn get_frame_buffer_for_dpb_slot(&self, dpb_slot: i32) -> Option<usize> {
        (0..8).find(|&i| self.frame_buffer_to_dpb_slot[i] == dpb_slot)
    }

    /// AV1 GetRelativeDist (C++ VulkanAV1Decoder.cpp:352-369): signed distance
    /// from `b` to `a` in order-hint space, wrapped to [-2^(ohb), 2^(ohb)).
    fn get_relative_dist(a: i32, b: i32, ohb: u32) -> i32 {
        let bits = ohb + 1;
        let diff = a - b;
        let m = 1 << (bits - 1);
        (diff & (m - 1)) - (diff & m)
    }

    /// Reset DPB state (e.g., on key frame or discontinuity).
    pub fn reset_dpb(&mut self) {
        self.frame_buffer_to_dpb_slot.fill(-1);
        self.frame_buffer_order_hint.fill(0);
        self.frame_buffer_dims.fill((0, 0));
        self.frame_buffer_saved_order_hints.fill([0; 8]);
        self.frame_buffer_ref_dist.fill([0; 8]);
        self.frame_buffer_frame_type.fill(0);
        self.frame_buffer_disable_cdf.fill(0);
        self.frame_buffer_seg_enabled.fill(0);
    }

    /// Compute reference DPB slot indices for building the Vulkan decode command.
    ///
    /// Returns an array indexed by AV1 reference name, as required by
    /// VkVideoDecodeAV1PictureInfoKHR::referenceNameSlotIndices:
    ///   [0] = LAST_FRAME, [1] = LAST2_FRAME, [2] = LAST3_FRAME,
    ///   [3] = GOLDEN_FRAME, [4] = BWDREF_FRAME, [5] = ALTREF2_FRAME,
    ///   [6] = ALTREF_FRAME
    ///
    /// Each value is the DPB slot index of the picture referenced by that
    /// reference name (it must equal the slotIndex of one of the reference
    /// slots passed to vkCmdDecodeVideoKHR — same convention as the C++
    /// reference, which uses the DPB slot number directly), or -1 if the
    /// reference name is not used (key frame, or frame buffer not mapped).
    pub fn compute_reference_name_slot_indices(
        &self,
        is_key_frame: bool,
        ref_frame_idx: &[u8; 7],
        _primary_ref_frame: u8,
    ) -> [i32; 7] {
        if is_key_frame {
            return [-1; 7];
        }

        // ref_frame_idx (AV1 spec) is indexed by reference name:
        // [0]=LAST, [1]=LAST2, [2]=LAST3, [3]=GOLDEN, [4]=BWDREF,
        // [5]=ALTREF2, [6]=ALTREF. Each entry holds the AV1 frame buffer
        // index (1..7) of the picture that reference name references.
        let mut result = [-1i32; 7];
        for i in 0..7usize {
            let fb = ref_frame_idx[i] as usize;
            if fb < 8 {
                result[i] = self.frame_buffer_to_dpb_slot[fb];
            }
        }
        result
    }

    /// Record an AV1 decode command.
    pub fn record_decode_command(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        session: vk::VideoSessionKHR,
        session_params: vk::VideoSessionParametersKHR,
        bitstream_buffer: vk::Buffer,
        bitstream_offset: u64,
        bitstream_range: u64,
        output_image_view: vk::ImageView,
        output_image: vk::Image,
        coded_extent: vk::Extent2D,
        dpb_setup_picture: Option<vk::VideoPictureResourceInfoKHR<'static>>,
        dpb_ref_pictures: &[vk::VideoPictureResourceInfoKHR<'static>],
        dpb_ref_slot_indices: &[i32],
        dpb_ref_order_hints: &[u32],
        dpb_ref_images: &[vk::Image],
        dpb_ref_slot_layouts: &[vk::ImageLayout],
        picture_info_container: &Av1PictureInfoContainer,
        av1_decode_info: &VideoDecodeAV1PictureInfoKHR,
        is_first_frame: bool,
        output_slot_index: i32,
        output_slot_old_layout: vk::ImageLayout,
        dpb_use_image_array: bool,
    ) -> VideoResult<()> {
        let _picture_info_ptr = picture_info_container.std_picture_info();

        // TEMP DIAGNOSTIC (iteration 18): confirm session params + key decode fields
        // for EVERY frame (was frame-0-only). Prints the ACTUAL src_buffer_range.
        if self.frame_count < 16 {
            eprintln!(
                "[AV1-DIAG] frame{}: bitstream=[{}..{}) range={} frame_header_offset={}, ref_name_slots={:?}, ref_slot_indices={:?}, coded_extent={}x{}, output_slot={}",
                self.frame_count,
                bitstream_offset,
                bitstream_offset + bitstream_range,
                bitstream_range,
                av1_decode_info.frame_header_offset,
                av1_decode_info.reference_name_slot_indices,
                dpb_ref_slot_indices,
                coded_extent.width,
                coded_extent.height,
                output_slot_index,
            );
        }

        // Build reference slots for BeginVideoCoding. For AV1, the pNext chain of
        // each reference slot must include a VkVideoDecodeAV1DpbSlotInfoKHR with
        // the reference picture's StdVideoDecodeAV1ReferenceInfo (Vulkan spec
        // VUID for AV1 decode; the C++ reference does the same).
        let ref_std_infos: Vec<ash::vk::native::StdVideoDecodeAV1ReferenceInfo> = (0..dpb_ref_pictures.len())
            .map(|i| {
                let mut info =
                    unsafe { std::mem::zeroed::<ash::vk::native::StdVideoDecodeAV1ReferenceInfo>() };
                info.OrderHint = dpb_ref_order_hints.get(i).copied().unwrap_or(0) as u8;
                // Populate SavedOrderHints / RefFrameSignBias / frame_type / flags
                // from the stored per-frame-buffer reference info (C++
                // VulkanAV1Decoder.cpp:323-334). Leaving these zero made INTER
                // frames decode with motion-compensation errors.
                if let Some(&ref_slot) = dpb_ref_slot_indices.get(i) {
                    if let Some(fb) = self.get_frame_buffer_for_dpb_slot(ref_slot) {
                        if let Some((saved_oh, bias, ftype, dcdf, seg)) =
                            self.get_frame_buffer_ref_info(fb)
                        {
                            info.SavedOrderHints = *saved_oh;
                            info.RefFrameSignBias = bias;
                            info.frame_type = ftype;
                            info.flags.set_disable_frame_end_update_cdf(dcdf as u32);
                            info.flags.set_segmentation_enabled(seg as u32);
                        }
                    }
                }
                info
            })
            .collect();
        // DEBUG (iteration 28): dump the FULL StdVideoDecodeAV1ReferenceInfo for
        // every reference (mirrors C++ [CPP-REFINFO], VulkanAV1Decoder.cpp:337-366).
        if self.frame_count < 16 {
            eprintln!(
                "[RUST-REFINFO] frame{}: refNameIdx={:?} ref_slots={:?}",
                self.frame_count,
                av1_decode_info.reference_name_slot_indices,
                dpb_ref_slot_indices,
            );
            for (i, info) in ref_std_infos.iter().enumerate() {
                eprintln!(
                    "[RUST-REFINFO]   ref[{}] slot={}: OrderHint={} RefFrameSignBias={:02x} frame_type={} dcdf={} seg={} SavedOH=[{},{},{},{},{},{},{},{}]",
                    i,
                    dpb_ref_slot_indices.get(i).copied().unwrap_or(-1),
                    info.OrderHint,
                    info.RefFrameSignBias,
                    info.frame_type,
                    info.flags.disable_frame_end_update_cdf(),
                    info.flags.segmentation_enabled(),
                    info.SavedOrderHints[0], info.SavedOrderHints[1],
                    info.SavedOrderHints[2], info.SavedOrderHints[3],
                    info.SavedOrderHints[4], info.SavedOrderHints[5],
                    info.SavedOrderHints[6], info.SavedOrderHints[7],
                );
            }
        }
        let ref_std_infos_ptr = if ref_std_infos.is_empty() {
            std::ptr::null()
        } else {
            Box::leak(ref_std_infos.into_boxed_slice()).as_ptr()
        };

        let ref_slot_infos: Vec<vk::VideoDecodeAV1DpbSlotInfoKHR<'static>> = (0..dpb_ref_pictures.len())
            .map(|i| vk::VideoDecodeAV1DpbSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_AV1_DPB_SLOT_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_reference_info: unsafe { ref_std_infos_ptr.add(i) },
                _marker: Default::default(),
            })
            .collect();
        let ref_slot_infos_ptr = if ref_slot_infos.is_empty() {
            std::ptr::null()
        } else {
            Box::leak(ref_slot_infos.into_boxed_slice()).as_ptr()
        };

        let ref_slots: Vec<vk::VideoReferenceSlotInfoKHR<'static>> = dpb_ref_pictures
            .iter()
            .zip(dpb_ref_slot_indices.iter())
            .enumerate()
            .map(|(i, (res, &slot_idx))| {
                let p_next = if ref_slot_infos_ptr.is_null() {
                    std::ptr::null()
                } else {
                    (unsafe { ref_slot_infos_ptr.add(i) }) as *const _
                };
                vk::VideoReferenceSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                    p_next,
                    slot_index: slot_idx,
                    p_picture_resource: res as *const _,
                    _marker: Default::default(),
                }
            })
            .collect();
        let ref_slots_ptr = if ref_slots.is_empty() {
            std::ptr::null()
        } else {
            Box::leak(ref_slots.into_boxed_slice()).as_ptr()
        };

        // Setup slot (current frame output). Per VUID-vkCmdDecodeVideoKHR-pDecodeInfo-09254,
        // for AV1 the pNext chain of pSetupReferenceSlot MUST include a
        // VkVideoDecodeAV1DpbSlotInfoKHR. The C++ reference does the same, populating
        // the setup slot's StdVideoDecodeAV1ReferenceInfo with the current frame's
        // OrderHint / SavedOrderHints / flags. Omitting it made the driver reject the
        // decode (all-zero output).
        let setup_std_info_ptr: *mut ash::vk::native::StdVideoDecodeAV1ReferenceInfo =
            if dpb_setup_picture.is_some() {
                let mut info =
                    unsafe { std::mem::zeroed::<ash::vk::native::StdVideoDecodeAV1ReferenceInfo>() };
                info.OrderHint = picture_info_container.std_picture_info.OrderHint;
                info.SavedOrderHints = picture_info_container.std_picture_info.OrderHints;
                // C++ VulkanAV1Decoder.cpp:317-319 sets the setup slot's
                // RefFrameSignBias from m_pBuffers[0].RefFrameSignBias[av1name] <= 0
                // for av1name in 0..8 (INCLUDING bit 0/INTRA). ref_dist[0] is
                // never set (stays 0) so (0 <= 0) sets bit 0; an unrefreshed
                // buffer (all dists 0) yields all bits set, matching C++.
                let mut setup_bias = 0u8;
                for av1name in 0..8usize {
                    if self.frame_buffer_ref_dist[0][av1name] <= 0 {
                        setup_bias |= 1 << av1name;
                    }
                }
                info.RefFrameSignBias = setup_bias;
                info.flags.set_disable_frame_end_update_cdf(
                    picture_info_container
                        .std_picture_info
                        .flags
                        .disable_frame_end_update_cdf(),
                );
                info.flags.set_segmentation_enabled(
                    picture_info_container
                        .std_picture_info
                        .flags
                        .segmentation_enabled(),
                );
                Box::leak(Box::new(info))
            } else {
                std::ptr::null_mut()
            };
        // DEBUG (iteration 28): dump the setup slot's StdVideoDecodeAV1ReferenceInfo
        // (mirrors C++ [CPP-REFINFO] SETUP line, VulkanAV1Decoder.cpp:358-363).
        if self.frame_count < 16 && !setup_std_info_ptr.is_null() {
            let info = unsafe { &*setup_std_info_ptr };
            eprintln!(
                "[RUST-REFINFO]   SETUP: OrderHint={} RefFrameSignBias={:02x} SavedOH=[{},{},{},{},{},{},{},{}]",
                info.OrderHint,
                info.RefFrameSignBias,
                info.SavedOrderHints[0], info.SavedOrderHints[1],
                info.SavedOrderHints[2], info.SavedOrderHints[3],
                info.SavedOrderHints[4], info.SavedOrderHints[5],
                info.SavedOrderHints[6], info.SavedOrderHints[7],
            );
        }

        let setup_slot_info_khr_ptr: *mut vk::VideoDecodeAV1DpbSlotInfoKHR<'static> =
            if dpb_setup_picture.is_some() {
                let info = vk::VideoDecodeAV1DpbSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_DECODE_AV1_DPB_SLOT_INFO_KHR,
                    p_next: std::ptr::null(),
                    p_std_reference_info: setup_std_info_ptr as *const _,
                    _marker: Default::default(),
                };
                Box::leak(Box::new(info))
            } else {
                std::ptr::null_mut()
            };

        let setup_slot_info = dpb_setup_picture
            .as_ref()
            .map(|res| vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: if setup_slot_info_khr_ptr.is_null() {
                    std::ptr::null()
                } else {
                    setup_slot_info_khr_ptr as *const _
                },
                slot_index: output_slot_index,
                p_picture_resource: res as *const _,
                _marker: Default::default(),
            });
        let has_setup_slot = setup_slot_info.is_some();
        let setup_slot_ptr = if has_setup_slot {
            Box::leak(Box::new(setup_slot_info.unwrap())) as *const _
        } else {
            std::ptr::null()
        };

        // Build all_slots for BeginVideoCoding. The C++ reference (VulkanVideoParser.cpp
        // lines 2496-2511) fills referenceSlots[0..numRefs-1] with the reference slots
        // and then appends the setup slot to the END: referenceSlots[numRefs] =
        // setupReferenceSlot. BeginVideoCoding then passes all (numRefs+1) elements.
        // So the array order is [ref0, ..., refN, setup] — setup LAST. (We previously
        // put setup FIRST, which differs from the pixel-perfect C++ reference.)
        let mut all_slots: Vec<vk::VideoReferenceSlotInfoKHR<'static>> = Vec::new();
        if !ref_slots_ptr.is_null() {
            let ref_slice =
                unsafe { std::slice::from_raw_parts(ref_slots_ptr, dpb_ref_pictures.len()) };
            all_slots.extend_from_slice(ref_slice);
        }
        if has_setup_slot {
            all_slots.push(unsafe { *setup_slot_ptr });
        }
        let all_slots_ptr = if all_slots.is_empty() {
            std::ptr::null()
        } else {
            Box::leak(all_slots.into_boxed_slice()).as_ptr()
        };
        let all_slots_count = if all_slots_ptr.is_null() {
            0
        } else {
            has_setup_slot as u32 + dpb_ref_pictures.len() as u32
        };

        // DEBUG (iteration 10): dump decode command params for frame 1
        if self.frame_count == 1 {
            eprintln!(
                "[RUST-DEC] frame1: output_slot={} output_old_layout={:?} ref_slots={:?} ref_layouts={:?} bitstream=[{}..{}) coded_extent={}x{} all_slots_count={} dpb_use_image_array={}",
                output_slot_index, output_slot_old_layout, dpb_ref_slot_indices, dpb_ref_slot_layouts,
                bitstream_offset, bitstream_offset + bitstream_range,
                coded_extent.width, coded_extent.height, all_slots_count, dpb_use_image_array
            );
        }

        // DEBUG (iteration 20): compact per-frame reference dump for ALL frames.
        // fc = 0-based decode frame index (fc0=ext0 KEY, fc1=ext1, fc2=ext2, fc3=ext3, ...).
        let (dbg_t_off, dbg_t_size) = if av1_decode_info.tile_count > 0
            && !av1_decode_info.p_tile_offsets.is_null()
            && !av1_decode_info.p_tile_sizes.is_null()
        {
            (
                unsafe { *av1_decode_info.p_tile_offsets },
                unsafe { *av1_decode_info.p_tile_sizes },
            )
        } else {
            (0, 0)
        };
        eprintln!(
            "[RUST-REF] fc={} out_slot={} ref_slots={:?} refNameIdx={:?} tile=[{}..{}) fh_off={}",
            self.frame_count, output_slot_index, dpb_ref_slot_indices,
            av1_decode_info.reference_name_slot_indices,
            dbg_t_off, dbg_t_off + dbg_t_size, av1_decode_info.frame_header_offset
        );

        // DEBUG (iteration 16): full decode-command input dump for the first
        // multi-reference frame (Rust frame 3 = decode command 2). Prints the
        // setup (output) picture resource, every reference picture resource
        // (imageView + base_array_layer + slot_index), referenceNameSlotIndices,
        // ref_slot_indices, and the tile/frame-header offsets, so we can diff
        // against the C++ reference's [CPP-DEC]/[CPP-PI] output.
        if self.frame_count < 16 {
            let setup = dpb_setup_picture.as_ref();
            eprintln!(
                "[RUST-DEC-F3] frame3: output_slot={} output_old_layout={:?} bitstream=[{}..{}) coded_extent={}x{} all_slots_count={} dpb_use_image_array={}",
                output_slot_index, output_slot_old_layout,
                bitstream_offset, bitstream_offset + bitstream_range,
                coded_extent.width, coded_extent.height, all_slots_count, dpb_use_image_array
            );
            if let Some(sp) = setup {
                eprintln!(
                    "[RUST-DEC-F3]   SETUP picture: view={:#x} base_array_layer={} coded_offset=({},{}) coded_extent={}x{}",
                    sp.image_view_binding.as_raw(), sp.base_array_layer,
                    sp.coded_offset.x, sp.coded_offset.y, sp.coded_extent.width, sp.coded_extent.height
                );
            } else {
                eprintln!("[RUST-DEC-F3]   SETUP picture: NONE");
            }
            for (i, rp) in dpb_ref_pictures.iter().enumerate() {
                let slot = dpb_ref_slot_indices.get(i).copied().unwrap_or(-1);
                eprintln!(
                    "[RUST-DEC-F3]   REF[{}] picture: view={:#x} base_array_layer={} slot_index={} coded_extent={}x{}",
                    i, rp.image_view_binding.as_raw(), rp.base_array_layer, slot,
                    rp.coded_extent.width, rp.coded_extent.height
                );
            }
            for i in 0..dpb_ref_pictures.len() {
                let ri = unsafe { &*ref_std_infos_ptr.add(i) };
                eprintln!(
                    "[RUST-DEC-F3]   REF[{}] std_info: OrderHint={} SavedOrderHints={:?} RefFrameSignBias={:#04x} frame_type={} disable_cdf={} seg={}",
                    i, ri.OrderHint, ri.SavedOrderHints, ri.RefFrameSignBias, ri.frame_type,
                    ri.flags.disable_frame_end_update_cdf(), ri.flags.segmentation_enabled()
                );
            }
            eprintln!(
                "[RUST-DEC-F3]   SETUP std_info: OrderHint={} SavedOrderHints={:?} RefFrameSignBias={:#04x}",
                unsafe { setup_std_info_ptr.as_ref().map(|s| s.OrderHint).unwrap_or(0) },
                unsafe { setup_std_info_ptr.as_ref().map(|s| s.SavedOrderHints).unwrap_or([0; 8]) },
                unsafe { setup_std_info_ptr.as_ref().map(|s| s.RefFrameSignBias).unwrap_or(0) }
            );
            eprintln!(
                "[RUST-DEC-F3]   all_slots: {:?}",
                (0..all_slots_count)
                    .map(|i| unsafe { (*all_slots_ptr.add(i as usize)).slot_index })
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "[RUST-DEC-F3]   decode p_reference_slots: {:?} count={}",
                dpb_ref_slot_indices, dpb_ref_pictures.len()
            );
            eprintln!(
                "[RUST-DEC-F3]   referenceNameSlotIndices={:?} ref_slot_indices={:?}",
                av1_decode_info.reference_name_slot_indices, dpb_ref_slot_indices
            );
            let (t_off, t_size) = if av1_decode_info.tile_count > 0
                && !av1_decode_info.p_tile_offsets.is_null()
                && !av1_decode_info.p_tile_sizes.is_null()
            {
                (
                    unsafe { *av1_decode_info.p_tile_offsets },
                    unsafe { *av1_decode_info.p_tile_sizes },
                )
            } else {
                (0, 0)
            };
            eprintln!(
                "[RUST-DEC-F3]   frame_header_offset={} tile_count={} tile_offsets[0]={} tile_sizes[0]={}",
                av1_decode_info.frame_header_offset, av1_decode_info.tile_count, t_off, t_size
            );
        }

        // DEBUG (iteration 22): full StdVideoDecodeAV1PictureInfo dump for ext3
        // (the broken multi-ref frame) to diff field-by-field vs C++ [CPP-PI] frame3.
        if self.frame_count < 8 {
            let pi = &picture_info_container.std_picture_info;
            let fl = &pi.flags;
            eprintln!(
                "[RUST-PI-F3] fc={} type={} oh={} primref={} refresh={:08x} interp={} txmode={} dqres={} dlres={} coded_denom={}",
                self.frame_count, pi.frame_type, pi.OrderHint, pi.primary_ref_frame, pi.refresh_frame_flags,
                pi.interpolation_filter as u32, pi.TxMode as u32, pi.delta_q_res, pi.delta_lf_res, pi.coded_denom
            );
            eprintln!(
                "[RUST-PI-F3] flags: superres={} renderdiff={} screencontent={} filterswitch={} intmv={} intrabc={} frss={} highprec={} mmodesw={} refrf_mvs={} warp={} reductx={} refsel={} skipmode={} deltaq={} delf={} delfmulti={} segen={} segmap={} segtemp={} segdata={} grain={}",
                fl.use_superres(), fl.render_and_frame_size_different(), fl.allow_screen_content_tools(),
                fl.is_filter_switchable(), fl.force_integer_mv(), fl.allow_intrabc(), fl.frame_refs_short_signaling(),
                fl.allow_high_precision_mv(), fl.is_motion_mode_switchable(), fl.use_ref_frame_mvs(),
                fl.allow_warped_motion(), fl.reduced_tx_set(), fl.reference_select(), fl.skip_mode_present(),
                fl.delta_q_present(), fl.delta_lf_present(), fl.delta_lf_multi(), fl.segmentation_enabled(),
                fl.segmentation_update_map(), fl.segmentation_temporal_update(), fl.segmentation_update_data(),
                fl.apply_grain()
            );
            eprintln!(
                "[RUST-PI-F3] skipModeFrame=[{},{}] orderHints={:?} expectedFrameId={:?}",
                pi.SkipModeFrame[0], pi.SkipModeFrame[1], pi.OrderHints, pi.expectedFrameId
            );
            let q = &picture_info_container.quantization;
            eprintln!(
                "[RUST-PI-F3] quant: using_qmatrix={} diff_uv_delta={} base_q={} dQYdc={} dQUdc={} dQUac={} dQVdc={} dQVac={} qm_y={} qm_u={} qm_v={}",
                q.flags.using_qmatrix(), q.flags.diff_uv_delta(), q.base_q_idx, q.DeltaQYDc,
                q.DeltaQUDc, q.DeltaQUAc, q.DeltaQVDc, q.DeltaQVAc, q.qm_y, q.qm_u, q.qm_v
            );
            let lf = &picture_info_container.loop_filter;
            eprintln!(
                "[RUST-PI-F3] lf: delta_en={} delta_upd={} level=[{},{},{},{}] sharp={} updrefd={} updmodes={} moded=[{},{}]",
                lf.flags.loop_filter_delta_enabled(), lf.flags.loop_filter_delta_update(),
                lf.loop_filter_level[0], lf.loop_filter_level[1], lf.loop_filter_level[2], lf.loop_filter_level[3],
                lf.loop_filter_sharpness, lf.update_ref_delta, lf.update_mode_delta,
                lf.loop_filter_mode_deltas[0], lf.loop_filter_mode_deltas[1]
            );
            eprintln!(
                "[RUST-PI-F3] lf refd=[{},{},{},{},{},{},{},{}]",
                lf.loop_filter_ref_deltas[0], lf.loop_filter_ref_deltas[1], lf.loop_filter_ref_deltas[2], lf.loop_filter_ref_deltas[3],
                lf.loop_filter_ref_deltas[4], lf.loop_filter_ref_deltas[5], lf.loop_filter_ref_deltas[6], lf.loop_filter_ref_deltas[7]
            );
            let c = &picture_info_container.cdef;
            eprintln!(
                "[RUST-PI-F3] cdef: damping={} bits={} ypri=[{},{},{},{}] ysec=[{},{},{},{}] uvprim=[{},{},{},{}] uvsec=[{},{},{},{}]",
                c.cdef_damping_minus_3, c.cdef_bits,
                c.cdef_y_pri_strength[0], c.cdef_y_pri_strength[1], c.cdef_y_pri_strength[2], c.cdef_y_pri_strength[3],
                c.cdef_y_sec_strength[0], c.cdef_y_sec_strength[1], c.cdef_y_sec_strength[2], c.cdef_y_sec_strength[3],
                c.cdef_uv_pri_strength[0], c.cdef_uv_pri_strength[1], c.cdef_uv_pri_strength[2], c.cdef_uv_pri_strength[3],
                c.cdef_uv_sec_strength[0], c.cdef_uv_sec_strength[1], c.cdef_uv_sec_strength[2], c.cdef_uv_sec_strength[3]
            );
            let lr = &picture_info_container.loop_restoration;
            eprintln!(
                "[RUST-PI-F3] lr: type=[{},{},{}] size=[{},{},{}]",
                lr.FrameRestorationType[0] as u32, lr.FrameRestorationType[1] as u32, lr.FrameRestorationType[2] as u32,
                lr.LoopRestorationSize[0], lr.LoopRestorationSize[1], lr.LoopRestorationSize[2]
            );
            let gm = &picture_info_container.global_motion;
            eprintln!("[RUST-PI-F3] gm: type={:?}", gm.GmType);
            for i in 0..8 {
                eprintln!(
                    "[RUST-PI-F3] gm_params[{}]=[{},{},{},{},{},{}]",
                    i, gm.gm_params[i][0], gm.gm_params[i][1], gm.gm_params[i][2],
                    gm.gm_params[i][3], gm.gm_params[i][4], gm.gm_params[i][5]
                );
            }
            let sg = &picture_info_container.segmentation;
            let mut seg_en = String::new();
            for i in 0..8 {
                seg_en.push_str(&format!("{}", sg.FeatureEnabled[i]));
            }
            eprintln!("[RUST-PI-F3] seg: enabled=[{}]", seg_en);
        }

        unsafe {
            // Begin command buffer
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .map_err(|e| VideoError::CommandBufferRecording(format!("Begin failed: {:?}", e)))?;

            // Begin video coding with reference slots
            let begin_coding_info = vk::VideoBeginCodingInfoKHR {
                s_type: vk::StructureType::VIDEO_BEGIN_CODING_INFO_KHR,
                p_next: std::ptr::null(),
                flags: vk::VideoBeginCodingFlagsKHR::empty(),
                video_session: session,
                video_session_parameters: session_params,
                reference_slot_count: all_slots_count,
                p_reference_slots: all_slots_ptr,
                _marker: Default::default(),
            };

            self.cmd_begin_video_coding(cmd_buffer, &begin_coding_info)?;

            // RESET decoder before first frame
            if is_first_frame {
                self.cmd_control_video_coding(cmd_buffer)?;
            }

            // Bitstream buffer barrier
            let buffer_barrier = vk::BufferMemoryBarrier2 {
                s_type: vk::StructureType::BUFFER_MEMORY_BARRIER_2,
                p_next: std::ptr::null(),
                src_stage_mask: vk::PipelineStageFlags2::HOST,
                src_access_mask: vk::AccessFlags2::HOST_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                buffer: bitstream_buffer,
                offset: bitstream_offset,
                size: bitstream_range,
                _marker: Default::default(),
            };

            // Output image barrier. When the DPB is a single image with array
            // layers (device does NOT support SEPARATE_REFERENCE_IMAGES), slot
            // `output_slot_index` lives in array layer `output_slot_index`; the
            // barrier must target that layer (C++ VkVideoDecoder.cpp:840 uses
            // baseArrayLayer = currPicIdx, layerCount = 1). Otherwise each slot is
            // its own single-layer image (layer 0).
            let output_base_layer = if dpb_use_image_array {
                output_slot_index as u32
            } else {
                0
            };

            // Output image barrier
            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: output_base_layer,
                layer_count: 1,
            };

            let new_layout = if dpb_setup_picture.is_some() {
                vk::ImageLayout::VIDEO_DECODE_DPB_KHR
            } else {
                vk::ImageLayout::VIDEO_DECODE_DST_KHR
            };

            let old_layout = output_slot_old_layout;

            let image_barrier = vk::ImageMemoryBarrier2 {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                p_next: std::ptr::null(),
                src_stage_mask: vk::PipelineStageFlags2::NONE,
                src_access_mask: vk::AccessFlags2::NONE,
                dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: output_image,
                old_layout,
                new_layout,
                subresource_range,
                _marker: Default::default(),
            };

            // Build barriers for reference images
            let mut image_barriers: Vec<vk::ImageMemoryBarrier2> =
                Vec::with_capacity(1 + dpb_ref_images.len());
            image_barriers.push(image_barrier);

            for ((&ref_image, &ref_layout), &ref_slot_idx) in dpb_ref_images
                .iter()
                .zip(dpb_ref_slot_layouts.iter())
                .zip(dpb_ref_slot_indices.iter())
            {
                if ref_image == vk::Image::null()
                    || ref_layout == vk::ImageLayout::VIDEO_DECODE_DPB_KHR
                {
                    continue;
                }

                // In image-array mode the reference slot lives in array layer
                // `ref_slot_idx` of the shared DPB image (C++ VkVideoDecoder.cpp:840).
                let ref_base_layer = if dpb_use_image_array {
                    ref_slot_idx as u32
                } else {
                    0
                };

                let ref_barrier = vk::ImageMemoryBarrier2 {
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                    p_next: std::ptr::null(),
                    src_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                    src_access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR
                        | vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
                    dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                    dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: ref_image,
                    old_layout: ref_layout,
                    new_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: ref_base_layer,
                        layer_count: 1,
                    },
                    _marker: Default::default(),
                };
                image_barriers.push(ref_barrier);
            }

            let dep_info = vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                p_next: std::ptr::null(),
                dependency_flags: vk::DependencyFlags::BY_REGION,
                memory_barrier_count: 0,
                p_memory_barriers: std::ptr::null(),
                buffer_memory_barrier_count: 1,
                p_buffer_memory_barriers: &buffer_barrier,
                image_memory_barrier_count: image_barriers.len() as u32,
                p_image_memory_barriers: image_barriers.as_ptr(),
                _marker: Default::default(),
            };
            self.cmd_pipeline_barrier_2(cmd_buffer, &dep_info)?;

            // Build AV1 decode info
            let dst_picture_resource = vk::VideoPictureResourceInfoKHR {
                s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: std::ptr::null(),
                coded_offset: vk::Offset2D::default(),
                coded_extent,
                base_array_layer: 0,
                image_view_binding: output_image_view,
                _marker: Default::default(),
            };

            let decode_info = vk::VideoDecodeInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_INFO_KHR,
                p_next: av1_decode_info as *const _ as *const _,
                flags: vk::VideoDecodeFlagsKHR::empty(),
                src_buffer: bitstream_buffer,
                src_buffer_offset: bitstream_offset,
                src_buffer_range: bitstream_range,
                dst_picture_resource,
                p_setup_reference_slot: setup_slot_ptr,
                reference_slot_count: dpb_ref_pictures.len() as u32,
                p_reference_slots: ref_slots_ptr,
                _marker: Default::default(),
            };

            self.cmd_decode_video(cmd_buffer, &decode_info)?;

            // End video coding
            self.cmd_end_video_coding(cmd_buffer)?;

            // End command buffer
            self.device
                .end_command_buffer(cmd_buffer)
                .map_err(|e| VideoError::CommandBufferRecording(format!("End failed: {:?}", e)))?;
        }

        self.frame_count += 1;

        Ok(())
    }

    fn cmd_pipeline_barrier_2(
        &self,
        cmd_buffer: vk::CommandBuffer,
        dep_info: &vk::DependencyInfo<'_>,
    ) -> VideoResult<()> {
        let fn_ptr = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                c"vkCmdPipelineBarrier2KHR".as_ptr(),
            )
};
        let Some(fn_ptr) = fn_ptr else {
            return Err(VideoError::CommandBufferRecording(
                "vkCmdPipelineBarrier2KHR not found".to_string(),
            ));
};

        unsafe {
            // Note: vkCmdPipelineBarrier2KHR returns void; no result to check.
            type FnType =
                unsafe extern "system" fn(vk::CommandBuffer, *const vk::DependencyInfo<'_>);
            let f: FnType = std::mem::transmute(fn_ptr);
            f(cmd_buffer, dep_info);
        }
        Ok(())
    }

    fn cmd_begin_video_coding(
        &self,
        cmd_buffer: vk::CommandBuffer,
        info: &vk::VideoBeginCodingInfoKHR<'_>,
    ) -> VideoResult<()> {
        let fn_ptr = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                c"vkCmdBeginVideoCodingKHR".as_ptr(),
            )
};
        let Some(fn_ptr) = fn_ptr else {
            return Err(VideoError::CommandBufferRecording(
                "vkCmdBeginVideoCodingKHR not found".to_string(),
            ));
};

        unsafe {
            // Note: vkCmdBeginVideoCodingKHR returns void; no result to check.
            type FnType = unsafe extern "system" fn(
                vk::CommandBuffer,
                *const vk::VideoBeginCodingInfoKHR<'_>,
            );
            let f: FnType = std::mem::transmute(fn_ptr);
            f(cmd_buffer, info);
        }
        Ok(())
    }

    fn cmd_decode_video(&self, cmd_buffer: vk::CommandBuffer, info: &vk::VideoDecodeInfoKHR<'_>) -> VideoResult<()> {
        let fn_ptr = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                c"vkCmdDecodeVideoKHR".as_ptr(),
            )
};
        let Some(fn_ptr) = fn_ptr else {
            return Err(VideoError::CommandBufferRecording(
                "vkCmdDecodeVideoKHR not found".to_string(),
            ));
};

        unsafe {
            // Note: vkCmdDecodeVideoKHR returns void; no result to check.
            type FnType = unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoDecodeInfoKHR<'_>);
            let f: FnType = std::mem::transmute(fn_ptr);
            f(cmd_buffer, info);
        }
        Ok(())
    }

    fn cmd_control_video_coding(&self, cmd_buffer: vk::CommandBuffer) -> VideoResult<()> {
        let coding_control_info = vk::VideoCodingControlInfoKHR {
            s_type: vk::StructureType::VIDEO_CODING_CONTROL_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoCodingControlFlagsKHR::RESET,
            _marker: Default::default(),
        };
        let fn_ptr = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                c"vkCmdControlVideoCodingKHR".as_ptr(),
            )
};
        let Some(fn_ptr) = fn_ptr else {
            return Err(VideoError::CommandBufferRecording(
                "vkCmdControlVideoCodingKHR not found".to_string(),
            ));
};

        unsafe {
            // Note: vkCmdControlVideoCodingKHR returns void; no result to check.
            type FnType = unsafe extern "system" fn(
                vk::CommandBuffer,
                *const vk::VideoCodingControlInfoKHR,
            );
            let f: FnType = std::mem::transmute(fn_ptr);
            f(cmd_buffer, &coding_control_info);
        }
        Ok(())
    }

    fn cmd_end_video_coding(&self, cmd_buffer: vk::CommandBuffer) -> VideoResult<()> {
        let end_coding_info = vk::VideoEndCodingInfoKHR {
            s_type: vk::StructureType::VIDEO_END_CODING_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoEndCodingFlagsKHR::empty(),
            _marker: Default::default(),
        };

        let fn_ptr = unsafe {
            self.instance.get_device_proc_addr(
                self.device.handle(),
                c"vkCmdEndVideoCodingKHR".as_ptr(),
            )
};
        let Some(fn_ptr) = fn_ptr else {
            return Err(VideoError::CommandBufferRecording(
                "vkCmdEndVideoCodingKHR not found".to_string(),
            ));
};

        unsafe {
            // Note: vkCmdEndVideoCodingKHR returns void; no result to check.
            type FnType = unsafe extern "system" fn(
                vk::CommandBuffer,
                *const vk::VideoEndCodingInfoKHR,
            );
            let f: FnType = std::mem::transmute(fn_ptr);
            f(cmd_buffer, &end_coding_info);
        }
        Ok(())
    }
}
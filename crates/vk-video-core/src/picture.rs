//! Picture parameter sets (SPS, PPS, VPS) for each codec.
//!
//! These correspond to the standardized video structures from
//! `vk_video/vulkan_video_codecs_standard_codec_info.h`.

use std::sync::Arc;

/// The type of picture parameter set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdType {
    H264Sps,
    H264Pps,
    H265Vps,
    H265Sps,
    H265Pps,
    Av1Sps,
    Vp9ColorConfig,
}

/// The parameter type within a picture parameter set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterType {
    Pps,
    Sps,
    Vps,
    Av1Sps,
    NumOfTypes,
    Invalid,
}

/// Base trait for picture parameter sets.
///
/// All codec-specific parameter sets implement this trait.
pub trait PictureParametersSet: std::fmt::Debug {
    /// Get the standard type.
    fn std_type(&self) -> StdType;

    /// Get the parameter type.
    fn parameter_type(&self) -> ParameterType;

    /// Get the update sequence count.
    fn update_sequence_count(&self) -> u32;

    /// Set the update sequence count.
    fn set_update_sequence_count(&mut self, count: u32);
}

/// H.264 Sequence Parameter Set.
///
/// Wraps `StdVideoH264SequenceParameterSet` from the Vulkan headers.
#[derive(Debug, Clone)]
pub struct H264Sps {
    pub profile_idc: u8,
    pub constraint_set0_flag: bool,
    pub constraint_set1_flag: bool,
    pub constraint_set2_flag: bool,
    pub constraint_set3_flag: bool,
    pub constraint_set4_flag: bool,
    pub constraint_set5_flag: bool,
    pub level_idc: u8,
    pub seq_parameter_set_id: u32,
    pub chroma_format_idc: u8,
    pub separate_colour_plane_flag: bool,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub qpprime_y_zero_transform_bypass_flag: bool,
    pub seq_scaling_matrix_present_flag: bool,
    /// Scaling lists (6x4x4 + 2x8x8 matrices, each row is zigzag order)
    pub scaling_list_4x4: [[u8; 16]; 6],
    pub scaling_list_8x8: [[u8; 64]; 2],
    pub log2_max_frame_num_minus4: u8,
    pub max_frame_num: u32,
    pub pic_order_cnt_type: u8,
    /// pic_order_cnt_type==1 fields
    pub delta_pic_order_always_zero_flag: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub num_ref_frames_in_pic_order_cnt_cycle: u32,
    pub offset_for_ref_frame: Vec<i32>,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub max_pic_order_cnt_lsb: u32,
    pub max_num_ref_frames: u32,
    pub gaps_in_frame_num_value_allowed_flag: bool,
    pub pic_width_in_mbs_minus1: u16,
    pub pic_height_in_map_units_minus1: u16,
    pub frame_mbs_only_flag: bool,
    pub mb_adaptive_frame_field_flag: bool,
    pub direct_8x8_inference_flag: bool,
    pub frame_cropping_flag: bool,
    pub frame_crop_left_offset: u32,
    pub frame_crop_right_offset: u32,
    pub frame_crop_top_offset: u32,
    pub frame_crop_bottom_offset: u32,
    pub vui_parameters_present_flag: bool,
    /// VUI parameters (present when vui_parameters_present_flag is true)
    pub vui: Option<H264SpsVui>,
}

/// H.264 Sequence Parameter Set VUI parameters.
#[derive(Debug, Clone, Default)]
pub struct H264SpsVui {
    // Flags
    pub aspect_ratio_info_present_flag: bool,
    pub overscan_info_present_flag: bool,
    pub overscan_appropriate_flag: bool,
    pub video_signal_type_present_flag: bool,
    pub video_full_range_flag: bool,
    pub color_description_present_flag: bool,
    pub chroma_loc_info_present_flag: bool,
    pub timing_info_present_flag: bool,
    pub fixed_frame_rate_flag: bool,
    pub bitstream_restriction_flag: bool,
    pub nal_hrd_parameters_present_flag: bool,
    pub vcl_hrd_parameters_present_flag: bool,
    // Aspect ratio
    pub aspect_ratio_idc: u8,
    pub sar_width: u16,
    pub sar_height: u16,
    // Video signal type
    pub video_format: u8,
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    // Timing
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    // Chroma location
    pub chroma_sample_loc_type_top_field: u8,
    pub chroma_sample_loc_type_bottom_field: u8,
    // Bitstream restrictions
    pub motion_vectors_over_pic_boundaries_flag: bool,
    pub max_bytes_per_pic_denom: u8,
    pub max_bits_per_mb_denom: u8,
    pub log2_max_mv_length_horizontal: u8,
    pub log2_max_mv_length_vertical: u8,
    pub max_num_reorder_frames: u8,
    pub max_dec_frame_buffering: u8,
    pub pic_struct_present_flag: bool,
}

impl H264Sps {
    pub fn new() -> Self {
        Self {
            profile_idc: 0,
            constraint_set0_flag: false,
            constraint_set1_flag: false,
            constraint_set2_flag: false,
            constraint_set3_flag: false,
            constraint_set4_flag: false,
            constraint_set5_flag: false,
            level_idc: 0,
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: false,
            scaling_list_4x4: [[0u8; 16]; 6],
            scaling_list_8x8: [[0u8; 64]; 2],
            log2_max_frame_num_minus4: 0,
            max_frame_num: 1,
            pic_order_cnt_type: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            log2_max_pic_order_cnt_lsb_minus4: 0,
            max_pic_order_cnt_lsb: 16,
            max_num_ref_frames: 1,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs_minus1: 0,
            pic_height_in_map_units_minus1: 0,
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: false,
            frame_cropping_flag: false,
            frame_crop_left_offset: 0,
            frame_crop_right_offset: 0,
            frame_crop_top_offset: 0,
            frame_crop_bottom_offset: 0,
            vui_parameters_present_flag: false,
            vui: None,
        }
    }
}

impl Default for H264Sps {
    fn default() -> Self {
        Self::new()
    }
}

impl PictureParametersSet for H264Sps {
    fn std_type(&self) -> StdType {
        StdType::H264Sps
    }

    fn parameter_type(&self) -> ParameterType {
        ParameterType::Sps
    }

    fn update_sequence_count(&self) -> u32 {
        0
    }

    fn set_update_sequence_count(&mut self, _count: u32) {}
}

/// H.264 Picture Parameter Set.
///
/// Wraps `StdVideoH264PictureParameterSet` from the Vulkan headers.
#[derive(Debug, Clone)]
pub struct H264Pps {
    pub pic_parameter_set_id: u32,
    pub seq_parameter_set_id: u32,
    pub entropy_coding_mode_flag: bool,
    pub bottom_field_pic_order_in_frame_present_flag: bool,
    pub num_slice_groups_minus1: u32,
    pub num_ref_idx_l0_default_active_minus1: u32,
    pub num_ref_idx_l1_default_active_minus1: u32,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_idc: u8,
    pub pic_init_qp_minus26: i32,
    pub pic_init_qs_minus26: i32,
    pub chroma_qp_index_offset: i32,
    pub deblocking_filter_control_present_flag: bool,
    pub redundant_pic_cnt_present_flag: bool,
    pub transform_8x8_mode_flag: bool,
    pub constrained_intra_pred_flag: bool,
    pub second_chroma_qp_index_offset: i32,
}

impl H264Pps {
    pub fn new() -> Self {
        Self {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            entropy_coding_mode_flag: false,
            bottom_field_pic_order_in_frame_present_flag: false,
            num_slice_groups_minus1: 0,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            pic_init_qp_minus26: 0,
            pic_init_qs_minus26: 0,
            chroma_qp_index_offset: 0,
            deblocking_filter_control_present_flag: true,
            redundant_pic_cnt_present_flag: false,
            transform_8x8_mode_flag: false,
            constrained_intra_pred_flag: false,
            second_chroma_qp_index_offset: 0,
        }
    }
}

impl Default for H264Pps {
    fn default() -> Self {
        Self::new()
    }
}

impl PictureParametersSet for H264Pps {
    fn std_type(&self) -> StdType {
        StdType::H264Pps
    }

    fn parameter_type(&self) -> ParameterType {
        ParameterType::Pps
    }

    fn update_sequence_count(&self) -> u32 {
        0
    }

    fn set_update_sequence_count(&mut self, _count: u32) {}
}

/// H.265 Video Parameter Set.
///
/// Wraps `StdVideoH265VideoParameterSet` from the Vulkan headers.
#[derive(Debug, Clone)]
pub struct H265Vps {
    pub vps_video_parameter_set_id: u8,
    pub vps_base_layer_internal_flag: bool,
    pub vps_base_layer_available_flag: bool,
    pub vps_max_layers_minus1: u16,
    pub vps_max_sub_layers_minus1: u8,
    pub vps_temporal_id_nesting_flag: bool,
    pub vps_sub_layer_ordering_info_present_flag: bool,
    // Profile/level from profile_tier_level
    pub profile_idc: u8,
    pub tier_flag: bool,
    pub level_idc: u8,
    // VPS layer info
    pub vps_max_layer_id: u16,
    pub vps_num_layer_sets: u32,
    /// layer_id_included_flag[layer_set_idx][layer_id] - flattened as [layer_set][layer_id]
    /// Max 1024 layer sets x 64 layer IDs
    pub layer_id_included_flag: Vec<Vec<bool>>,
    // VPS timing info
    pub vps_timing_info_present_flag: bool,
    pub vps_num_units_in_tick: u32,
    pub vps_time_scale: u32,
    pub vps_poc_proportional_to_timing_flag: bool,
    pub vps_num_ticks_poc_diff_one_minus1: u32,
    pub vps_num_hrd_parameters: u32,
    // DPB management (from StdVideoH265DecPicBufMgr)
    pub max_dec_pic_buffering_minus1: [u8; 7], // MAX_SUB_LAYERS
    pub max_num_reorder_pics: [u8; 7],
    pub max_latency_increase_plus1: [u8; 7],
    // VPS extension
    pub vps_extension_flag: bool,
}

impl H265Vps {
    pub fn new() -> Self {
        Self {
            vps_video_parameter_set_id: 0,
            vps_base_layer_internal_flag: false,
            vps_base_layer_available_flag: false,
            vps_max_layers_minus1: 0,
            vps_max_sub_layers_minus1: 0,
            vps_temporal_id_nesting_flag: true,
            vps_sub_layer_ordering_info_present_flag: false,
            profile_idc: 1,   // Main profile default
            tier_flag: false, // Main tier default
            level_idc: 123,   // Level 4.1 default (123 = 4.1 * 30)
            vps_max_layer_id: 0,
            vps_num_layer_sets: 1,
            layer_id_included_flag: Vec::new(),
            vps_timing_info_present_flag: false,
            vps_num_units_in_tick: 0,
            vps_time_scale: 0,
            vps_poc_proportional_to_timing_flag: false,
            vps_num_ticks_poc_diff_one_minus1: 0,
            vps_num_hrd_parameters: 0,
            max_dec_pic_buffering_minus1: [0; 7],
            max_num_reorder_pics: [0; 7],
            max_latency_increase_plus1: [0; 7],
            vps_extension_flag: false,
        }
    }
}

impl Default for H265Vps {
    fn default() -> Self {
        Self::new()
    }
}

impl PictureParametersSet for H265Vps {
    fn std_type(&self) -> StdType {
        StdType::H265Vps
    }

    fn parameter_type(&self) -> ParameterType {
        ParameterType::Vps
    }

    fn update_sequence_count(&self) -> u32 {
        0
    }

    fn set_update_sequence_count(&mut self, _count: u32) {}
}

/// H.265 Sequence Parameter Set VUI parameters.
/// Matches StdVideoH265SequenceParameterSetVui layout.
#[derive(Debug, Clone, Default)]
pub struct H265SpsVui {
    // Aspect ratio
    pub aspect_ratio_info_present_flag: bool,
    pub aspect_ratio_idc: u8,
    pub sar_width: u16,
    pub sar_height: u16,
    // Overscan
    pub overscan_info_present_flag: bool,
    pub overscan_appropriate_flag: bool,
    // Video signal type
    pub video_signal_type_present_flag: bool,
    pub video_format: u8,
    pub video_full_range_flag: bool,
    pub colour_description_present_flag: bool,
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coeffs: u8,
    // Chroma location
    pub chroma_loc_info_present_flag: bool,
    pub chroma_sample_loc_type_top_field: u32,
    pub chroma_sample_loc_type_bottom_field: u32,
    // Neutral chroma
    pub neutral_chroma_indication_flag: bool,
    // Field/frame
    pub field_seq_flag: bool,
    pub frame_field_info_present_flag: bool,
    // Display window
    pub default_display_window_flag: bool,
    pub def_disp_win_left_offset: u32,
    pub def_disp_win_right_offset: u32,
    pub def_disp_win_top_offset: u32,
    pub def_disp_win_bottom_offset: u32,
    // Timing
    pub vui_timing_info_present_flag: bool,
    pub vui_num_units_in_tick: u32,
    pub vui_time_scale: u32,
    pub vui_poc_proportional_to_timing_flag: bool,
    pub vui_num_ticks_poc_diff_one_minus1: u32,
    // HRD
    pub vui_hrd_parameters_present_flag: bool,
    // Bitstream restriction
    pub bitstream_restriction_flag: bool,
    pub tiles_fixed_structure_flag: bool,
    pub motion_vectors_over_pic_boundaries_flag: bool,
    pub restricted_ref_pic_lists_flag: bool,
    pub min_spatial_segmentation_idc: u32,
    pub max_bytes_per_pic_denom: u32,
    pub max_bits_per_min_cu_denom: u32,
    pub log2_max_mv_length_horizontal: u32,
    pub log2_max_mv_length_vertical: u32,
}

/// H.265 Scaling List Entry.
#[derive(Debug, Clone)]
pub struct H265ScalingListEntry {
    pub scaling_list_pred_mode_flag: bool,
    pub scaling_list_pred_matrix_id_delta: i32,
    pub scaling_list_dc_coef_minus8: i32,
    pub scaling_list_delta_coef: [i8; 64],
}

impl Default for H265ScalingListEntry {
    fn default() -> Self {
        Self {
            scaling_list_pred_mode_flag: false,
            scaling_list_pred_matrix_id_delta: 0,
            scaling_list_dc_coef_minus8: 0,
            scaling_list_delta_coef: [0; 64],
        }
    }
}

/// H.265 Scaling Lists.
/// Matches StdVideoH265ScalingLists layout.
#[derive(Debug, Clone)]
pub struct H265ScalingLists {
    /// ScalingList4x4[6][16] - 6 matrices (3 luma + 3 chroma), 16 coefficients each
    pub scaling_list_4x4: [[u8; 16]; 6],
    /// ScalingList8x8[6][64] - 6 matrices, 64 coefficients each
    pub scaling_list_8x8: [[u8; 64]; 6],
    /// ScalingList16x16[6][64]
    pub scaling_list_16x16: [[u8; 64]; 6],
    /// ScalingList32x32[2][64]
    pub scaling_list_32x32: [[u8; 64]; 2],
    /// ScalingListDCCoef16x16[6][16]
    pub scaling_list_dc_coef_16x16: [[i8; 16]; 6],
    /// ScalingListDCCoef32x32[2][16]
    pub scaling_list_dc_coef_32x32: [[i8; 16]; 2],
}

impl Default for H265ScalingLists {
    fn default() -> Self {
        Self {
            scaling_list_4x4: [[0; 16]; 6],
            scaling_list_8x8: [[0; 64]; 6],
            scaling_list_16x16: [[0; 64]; 6],
            scaling_list_32x32: [[0; 64]; 2],
            scaling_list_dc_coef_16x16: [[0; 16]; 6],
            scaling_list_dc_coef_32x32: [[0; 16]; 2],
        }
    }
}

/// H.265 Sequence Parameter Set.
/// Matches StdVideoH265SequenceParameterSet layout.
#[derive(Debug, Clone)]
pub struct H265Sps {
    pub sps_video_parameter_set_id: u8,
    pub sps_max_sub_layers_minus1: u8,
    pub sps_temporal_id_nesting_flag: bool,
    pub sps_seq_parameter_set_id: u32,
    pub profile_idc: u8,
    pub tier_flag: bool,
    pub level_idc: u8,
    pub chroma_format_idc: u8,
    pub separate_colour_plane_flag: bool,
    pub pic_width_in_luma_samples: u16,
    pub pic_height_in_luma_samples: u16,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub max_num_ref_frames: u16,
    pub scaling_list_enabled_flag: bool,
    pub sps_sub_layer_ordering_info_present_flag: bool,
    pub conformance_window_flag: bool,
    pub sps_scaling_list_data_present_flag: bool,
    pub amp_enabled_flag: bool,
    pub sample_adaptive_offset_enabled_flag: bool,
    pub sps_temporal_mvp_enabled_flag: bool,
    pub strong_intra_smoothing_enabled_flag: bool,
    pub vui_parameters_present_flag: bool,
    pub long_term_ref_pics_present_flag: bool,

    // Block/transform sizes
    pub log2_min_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_luma_coding_block_size: u8,
    pub log2_min_luma_transform_block_size_minus2: u8,
    pub log2_diff_max_min_luma_transform_block_size: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub max_transform_hierarchy_depth_intra: u8,

    // DPB management (from StdVideoH265DecPicBufMgr) - matches Vulkan API [7]
    pub max_dec_pic_buffering_minus1: [u8; 7],
    pub max_num_reorder_pics: [u8; 7],
    pub max_latency_increase_plus1: [u8; 7],

    // Short-term reference picture sets
    pub num_short_term_ref_pic_sets: u8,
    pub short_term_ref_pic_sets: Vec<H265ShortTermRefPicSet>,

    // Long-term reference pictures
    pub num_long_term_ref_pics_sps: u8,
    pub lt_ref_pic_poc_lsb_sps: [u32; 32], // matches StdVideoH265LongTermRefPicsSps
    pub used_by_curr_pic_lt_sps_flag: u32,

    // PCM fields
    pub pcm_enabled_flag: bool,
    pub pcm_sample_bit_depth_luma_minus1: u8,
    pub pcm_sample_bit_depth_chroma_minus1: u8,
    pub log2_min_pcm_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_pcm_luma_coding_block_size: u8,

    // Palette fields
    pub palette_predictor_initialization_present_flag: bool,
    pub sps_num_palette_predictor_initializers_minus1: u8,
    pub palette_max_size: u8,
    pub delta_palette_max_predictor_size: u8,
    pub motion_vector_resolution_control_idc: u8,

    // Conformance window
    pub conf_win_left_offset: u32,
    pub conf_win_right_offset: u32,
    pub conf_win_top_offset: u32,
    pub conf_win_bottom_offset: u32,

    // VUI parameters
    pub vui: H265SpsVui,

    // Scaling lists
    pub scaling_lists: H265ScalingLists,

    // Extension flags (from sps_extension)
    pub pcm_loop_filter_disabled_flag: bool,
    pub sps_extension_present_flag: bool,
    pub sps_range_extension_flag: bool,
    pub intra_smoothing_disabled_flag: bool,
    pub palette_mode_enabled_flag: bool,
}

/// H.265 Short-Term Reference Picture Set.
/// Matches StdVideoH265ShortTermRefPicSet layout.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct H265ShortTermRefPicSet {
    pub inter_ref_pic_set_prediction_flag: bool,
    pub delta_idx_minus1: u32,
    pub use_delta_flag: u16,
    pub abs_delta_rps_minus1: u16,
    pub used_by_curr_pic_flag: u16,
    pub used_by_curr_pic_s0_flag: u16,
    pub used_by_curr_pic_s1_flag: u16,
    pub num_negative_pics: u8,
    pub num_positive_pics: u8,
    pub delta_poc_s0_minus1: [u16; 16],
    pub delta_poc_s1_minus1: [u16; 16],
}


impl H265Sps {
    pub fn new() -> Self {
        Self {
            sps_video_parameter_set_id: 0,
            sps_max_sub_layers_minus1: 0,
            sps_temporal_id_nesting_flag: true,
            sps_seq_parameter_set_id: 0,
            profile_idc: 1,
            tier_flag: false, // Main tier default
            level_idc: 123,   // Level 4.1 default (123 = 4.1 * 30)
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            pic_width_in_luma_samples: 0,
            pic_height_in_luma_samples: 0,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            max_num_ref_frames: 1,
            scaling_list_enabled_flag: false,
            sps_sub_layer_ordering_info_present_flag: false,
            conformance_window_flag: false,
            sps_scaling_list_data_present_flag: false,
            amp_enabled_flag: false,
            sample_adaptive_offset_enabled_flag: false,
            sps_temporal_mvp_enabled_flag: false,
            strong_intra_smoothing_enabled_flag: false,
            vui_parameters_present_flag: false,
            long_term_ref_pics_present_flag: false,

            log2_min_luma_coding_block_size_minus3: 0,
            log2_diff_max_min_luma_coding_block_size: 0,
            log2_min_luma_transform_block_size_minus2: 0,
            log2_diff_max_min_luma_transform_block_size: 0,
            max_transform_hierarchy_depth_inter: 0,
            max_transform_hierarchy_depth_intra: 0,

            max_dec_pic_buffering_minus1: [0; 7],
            max_num_reorder_pics: [0; 7],
            max_latency_increase_plus1: [0; 7],

            num_short_term_ref_pic_sets: 0,
            short_term_ref_pic_sets: Vec::new(),

            num_long_term_ref_pics_sps: 0,
            lt_ref_pic_poc_lsb_sps: [0; 32],
            used_by_curr_pic_lt_sps_flag: 0,

            pcm_enabled_flag: false,
            pcm_sample_bit_depth_luma_minus1: 0,
            pcm_sample_bit_depth_chroma_minus1: 0,
            log2_min_pcm_luma_coding_block_size_minus3: 0,
            log2_diff_max_min_pcm_luma_coding_block_size: 0,

            palette_predictor_initialization_present_flag: false,
            sps_num_palette_predictor_initializers_minus1: 0,
            palette_max_size: 0,
            delta_palette_max_predictor_size: 0,
            motion_vector_resolution_control_idc: 0,

            conf_win_left_offset: 0,
            conf_win_right_offset: 0,
            conf_win_top_offset: 0,
            conf_win_bottom_offset: 0,

            vui: H265SpsVui::default(),
            scaling_lists: H265ScalingLists::default(),

            pcm_loop_filter_disabled_flag: false,
            sps_extension_present_flag: false,
            sps_range_extension_flag: false,
            intra_smoothing_disabled_flag: false,
            palette_mode_enabled_flag: false,
        }
    }
}

impl Default for H265Sps {
    fn default() -> Self {
        Self::new()
    }
}

impl PictureParametersSet for H265Sps {
    fn std_type(&self) -> StdType {
        StdType::H265Sps
    }

    fn parameter_type(&self) -> ParameterType {
        ParameterType::Sps
    }

    fn update_sequence_count(&self) -> u32 {
        0
    }

    fn set_update_sequence_count(&mut self, _count: u32) {}
}

/// H.265 Picture Parameter Set.
/// Matches StdVideoH265PictureParameterSet layout.
#[derive(Debug, Clone)]
pub struct H265Pps {
    pub pps_pic_parameter_set_id: u32,
    pub pps_seq_parameter_set_id: u32,
    pub sps_video_parameter_set_id: u8,
    pub num_extra_slice_header_bits: u8,
    pub dependent_slice_segments_enabled_flag: bool,
    pub output_flag_present_flag: bool,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub sign_data_hiding_enabled_flag: bool,
    pub cabac_init_present_flag: bool,
    pub pps_init_qp_minus26: i32,
    pub constrained_intra_pred_flag: bool,
    pub transform_skip_enabled_flag: bool,
    pub cu_qp_delta_enabled_flag: bool,
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_flag: bool,
    pub transquant_bypass_enabled_flag: bool,
    pub tiles_enabled_flag: bool,
    pub entropy_coding_sync_enabled_flag: bool,
    pub uniform_spacing_flag: bool,
    pub loop_filter_across_tiles_enabled_flag: bool,
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    pub deblocking_filter_control_present_flag: bool,
    pub deblocking_filter_override_enabled_flag: bool,
    pub pps_deblocking_filter_disabled_flag: bool,
    pub pps_scaling_list_data_present_flag: bool,
    pub lists_modification_present_flag: bool,
    pub slice_segment_header_extension_present_flag: bool,
    pub pps_extension_present_flag: bool,

    // QP and deblocking fields
    pub diff_cu_qp_delta_depth: u8,
    pub pps_cb_qp_offset: i8,
    pub pps_cr_qp_offset: i8,
    pub pps_beta_offset_div2: i8,
    pub pps_tc_offset_div2: i8,

    // Merge and transform skip fields
    pub log2_parallel_merge_level_minus2: u8,
    pub log2_max_transform_skip_block_size_minus2: u8,

    // Chroma QP offset fields (from pps_range_extension)
    pub pps_range_extension_flag: bool,
    pub cross_component_prediction_enabled_flag: bool,
    pub chroma_qp_offset_list_enabled_flag: bool,
    pub diff_cu_chroma_qp_offset_depth: u8,
    pub chroma_qp_offset_list_len_minus1: u8,
    pub cb_qp_offset_list: [i8; 6],
    pub cr_qp_offset_list: [i8; 6],
    pub log2_sao_offset_scale_luma: u8,
    pub log2_sao_offset_scale_chroma: u8,

    // SAO fields
    pub log2_sao_offset_scale_luma_vui: u8,
    pub log2_sao_offset_scale_chroma_vui: u8,

    // ACT fields
    pub pps_act_y_qp_offset_plus5: i8,
    pub pps_act_cb_qp_offset_plus5: i8,
    pub pps_act_cr_qp_offset_plus3: i8,

    // Palette fields
    pub pps_num_palette_predictor_initializers: u8,
    pub luma_bit_depth_entry_minus8: u8,
    pub chroma_bit_depth_entry_minus8: u8,

    // Tile fields
    pub num_tile_columns_minus1: u8,
    pub num_tile_rows_minus1: u8,
    pub column_width_minus1: [u16; 19],
    pub row_height_minus1: [u16; 21],

    // Deblocking filter control
    pub pps_disable_deblocking_filter_flag: u8,

    // Scaling lists
    pub scaling_lists: H265ScalingLists,
}

impl H265Pps {
    pub fn new() -> Self {
        Self {
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            sps_video_parameter_set_id: 0,
            num_extra_slice_header_bits: 0,
            dependent_slice_segments_enabled_flag: false,
            output_flag_present_flag: false,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            sign_data_hiding_enabled_flag: false,
            cabac_init_present_flag: false,
            pps_init_qp_minus26: 0,
            constrained_intra_pred_flag: false,
            transform_skip_enabled_flag: false,
            cu_qp_delta_enabled_flag: false,
            pps_slice_chroma_qp_offsets_present_flag: false,
            weighted_pred_flag: false,
            weighted_bipred_flag: false,
            transquant_bypass_enabled_flag: false,
            tiles_enabled_flag: false,
            entropy_coding_sync_enabled_flag: false,
            uniform_spacing_flag: true,
            loop_filter_across_tiles_enabled_flag: false,
            pps_loop_filter_across_slices_enabled_flag: false,
            deblocking_filter_control_present_flag: false,
            deblocking_filter_override_enabled_flag: false,
            pps_deblocking_filter_disabled_flag: false,
            pps_scaling_list_data_present_flag: false,
            lists_modification_present_flag: false,
            slice_segment_header_extension_present_flag: false,
            pps_extension_present_flag: false,

            diff_cu_qp_delta_depth: 0,
            pps_cb_qp_offset: 0,
            pps_cr_qp_offset: 0,
            pps_beta_offset_div2: 0,
            pps_tc_offset_div2: 0,

            log2_parallel_merge_level_minus2: 0,
            log2_max_transform_skip_block_size_minus2: 0,

            pps_range_extension_flag: false,
            cross_component_prediction_enabled_flag: false,
            chroma_qp_offset_list_enabled_flag: false,
            diff_cu_chroma_qp_offset_depth: 0,
            chroma_qp_offset_list_len_minus1: 0,
            cb_qp_offset_list: [0; 6],
            cr_qp_offset_list: [0; 6],
            log2_sao_offset_scale_luma: 0,
            log2_sao_offset_scale_chroma: 0,

            log2_sao_offset_scale_luma_vui: 0,
            log2_sao_offset_scale_chroma_vui: 0,

            pps_act_y_qp_offset_plus5: 0,
            pps_act_cb_qp_offset_plus5: 0,
            pps_act_cr_qp_offset_plus3: 0,

            pps_num_palette_predictor_initializers: 0,
            luma_bit_depth_entry_minus8: 0,
            chroma_bit_depth_entry_minus8: 0,

            num_tile_columns_minus1: 0,
            num_tile_rows_minus1: 0,
            column_width_minus1: [0; 19],
            row_height_minus1: [0; 21],

            pps_disable_deblocking_filter_flag: 0,

            scaling_lists: H265ScalingLists::default(),
        }
    }
}

impl Default for H265Pps {
    fn default() -> Self {
        Self::new()
    }
}

impl PictureParametersSet for H265Pps {
    fn std_type(&self) -> StdType {
        StdType::H265Pps
    }

    fn parameter_type(&self) -> ParameterType {
        ParameterType::Pps
    }

    fn update_sequence_count(&self) -> u32 {
        0
    }

    fn set_update_sequence_count(&mut self, _count: u32) {}
}

/// AV1 Sequence Header (SPS equivalent).
#[derive(Debug, Clone)]
pub struct Av1Sps {
    pub profile: u8,
    pub level: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_motion: bool,
    pub enable_second_ref_frame: bool,
    pub enable_ref_frame_mvs: bool,
    pub seq_force_screen_content_tools: u8,
    pub seq_force_integer_mv: u8,
    pub separate_uv_delta_q: bool,
    pub enable_offset_unit: bool,
    pub enable_txfm_32x32: bool,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub film_grain_params_present: bool,
    pub initial_display_delay_present_flag: bool,
    pub frame_width_bits: u8,
    pub frame_height_bits: u8,
    pub max_frame_width_minus_1: u16,
    pub max_frame_height_minus_1: u16,
    pub frame_id_numbers_present_flag: bool,
    pub delta_frame_id_length_minus2: u8,
    pub additional_frame_id_length_minus1: u8,
    pub order_hint_bits_minus1: u8,
    // Timing info (from timing_info_present_flag)
    pub timing_info_present_flag: bool,
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    pub equal_picture_interval: bool,
    // Decoder model info
    pub decoder_model_info_present_flag: bool,
    pub buffer_delay_length_minus_1: u8,
    // Color config
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub mono_chrome: bool,
    pub color_description_present: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub chroma_sample_position: u8,
}

impl Av1Sps {
    pub fn new() -> Self {
        Self {
            profile: 0,
            level: 2,
            still_picture: false,
            reduced_still_picture_header: false,
            use_128x128_superblock: false,
            enable_filter_intra: false,
            enable_intra_edge_filter: true,
            enable_interintra_compound: false,
            enable_masked_compound: false,
            enable_warped_motion: false,
            enable_dual_filter: false,
            enable_order_hint: false,
            enable_jnt_motion: false,
            enable_second_ref_frame: false,
            enable_ref_frame_mvs: false,
            seq_force_screen_content_tools: 0,
            seq_force_integer_mv: 0,
            separate_uv_delta_q: false,
            enable_offset_unit: false,
            enable_txfm_32x32: false,
            enable_superres: false,
            enable_cdef: false,
            enable_restoration: false,
            film_grain_params_present: false,
            initial_display_delay_present_flag: false,
            frame_width_bits: 8,
            frame_height_bits: 9,
            max_frame_width_minus_1: 0,
            max_frame_height_minus_1: 0,
            frame_id_numbers_present_flag: false,
            delta_frame_id_length_minus2: 0,
            additional_frame_id_length_minus1: 0,
            order_hint_bits_minus1: 0,
            // Timing info
            timing_info_present_flag: false,
            num_units_in_display_tick: 0,
            time_scale: 0,
            equal_picture_interval: false,
            // Decoder model info
            decoder_model_info_present_flag: false,
            buffer_delay_length_minus_1: 0,
            // Color config
            high_bitdepth: false,
            twelve_bit: false,
            mono_chrome: false,
            color_description_present: false,
            color_primaries: 2,          // BT.709 default
            transfer_characteristics: 2, // BT.709 default
            matrix_coefficients: 2,      // BT.709 default
            color_range: false,
            subsampling_x: 1,
            subsampling_y: 1,
            chroma_sample_position: 0,
        }
    }
}

impl Default for Av1Sps {
    fn default() -> Self {
        Self::new()
    }
}

impl PictureParametersSet for Av1Sps {
    fn std_type(&self) -> StdType {
        StdType::Av1Sps
    }

    fn parameter_type(&self) -> ParameterType {
        ParameterType::Av1Sps
    }

    fn update_sequence_count(&self) -> u32 {
        0
    }

    fn set_update_sequence_count(&mut self, _count: u32) {}
}

/// A boxed, dynamically-typed picture parameter set.
///
/// Uses `Arc` for shared ownership, similar to the Vulkan samples'
/// `VkSharedBaseObj`.
#[derive(Debug, Clone)]
pub struct BoxedPictureParametersSet {
    inner: Arc<dyn AnyPictureParametersSet>,
}

impl BoxedPictureParametersSet {
    /// Create from any picture parameter set.
    pub fn new<T: PictureParametersSet + 'static>(set: T) -> Self {
        Self {
            inner: Arc::new(set),
        }
    }

    /// Get the standard type.
    pub fn std_type(&self) -> StdType {
        self.inner.std_type()
    }

    /// Get the parameter type.
    pub fn parameter_type(&self) -> ParameterType {
        self.inner.parameter_type()
    }

    /// Get the update sequence count.
    pub fn update_sequence_count(&self) -> u32 {
        self.inner.update_sequence_count()
    }

    /// Downcast to a concrete type.
    pub fn downcast_ref<T: PictureParametersSet + 'static>(&self) -> Option<&T> {
        self.inner.as_any().downcast_ref::<T>()
    }

    /// Check if this is a specific type.
    pub fn is_type<T: PictureParametersSet + 'static>(&self) -> bool {
        self.inner.as_any().is::<T>()
    }
}

impl std::ops::Deref for BoxedPictureParametersSet {
    type Target = dyn PictureParametersSet;

    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

/// Trait for any picture parameter set (for downcasting).
pub trait AnyPictureParametersSet: PictureParametersSet + 'static {
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: PictureParametersSet + 'static> AnyPictureParametersSet for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ============================================================================
// VP9 Constants (from Vulkan spec: vulkan_video_codecs_standard.h)
// ============================================================================

/// VP9 Frame marker value (2 bits)
pub const VP9_FRAME_MARKER: u8 = 0b10;

/// VP9 Frame sync code (24 bits)
pub const VP9_FRAME_SYNC_CODE: u32 = 0x498342;

/// Maximum VP9 probability value
pub const VP9_MAX_PROBABILITY: u8 = 255;

/// Minimum tile width in 64x64 superblocks
pub const VP9_MIN_TILE_WIDTH_B64: u8 = 4;

/// Maximum tile width in 64x64 superblocks
pub const VP9_MAX_TILE_WIDTH_B64: u8 = 64;

/// Number of reference frames
pub const VP9_NUM_REF_FRAMES: u32 = 8;

/// Number of references per frame (excludes current)
pub const VP9_REFS_PER_FRAME: u32 = 7;

/// Maximum number of reference frames for loop filter
pub const VP9_MAX_REF_FRAMES: u32 = 4;

/// Number of loop filter adjustments
pub const VP9_LOOP_FILTER_ADJUSTMENTS: u32 = 2;

/// Maximum number of segments
pub const VP9_MAX_SEGMENTS: u32 = 8;

/// Maximum number of segment levels
pub const VP9_SEG_LVL_MAX: u32 = 4;

/// Maximum number of segmentation tree probabilities
pub const VP9_MAX_SEGMENTATION_TREE_PROBS: u32 = 7;

/// Maximum number of segmentation prediction probabilities
pub const VP9_MAX_SEGMENTATION_PRED_PROB: u32 = 3;

/// Reference frame name: LAST
pub const VP9_REFERENCE_NAME_LAST_FRAME: u32 = 0;

/// Reference frame name: GOLDEN
pub const VP9_REFERENCE_NAME_GOLDEN_FRAME: u32 = 1;

/// Reference frame name: ALTREF
pub const VP9_REFERENCE_NAME_ALTREF_FRAME: u32 = 2;

/// Reference frame name: LAST2
pub const VP9_REFERENCE_NAME_LAST2_FRAME: u32 = 3;

/// Reference frame name: LAST3
pub const VP9_REFERENCE_NAME_LAST3_FRAME: u32 = 4;

/// Reference frame name: GOLDEN2
pub const VP9_REFERENCE_NAME_BACKWARD_FRAME: u32 = 5;

/// Reference frame name: KEY
pub const VP9_REFERENCE_NAME_KEY_FRAME: u32 = 6;

/// Segment level features
#[repr(u32)]
pub enum Vp9SegmentLevel {
    AltQ = 0,
    AltLf = 1,
    RefFrame = 2,
    Skip = 3,
}

// ============================================================================
// VP9 Data Structures
// ============================================================================

/// VP9 Profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Vp9Profile {
    #[default]
    Profile0 = 0,
    Profile1 = 1,
    Profile2 = 2,
    Profile3 = 3,
}

/// VP9 Frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Vp9FrameType {
    #[default]
    Key = 0,
    Inter = 1,
}

/// VP9 Color space (from vulkan_video_codec_vp9std.h).
/// Values: UNKNOWN=0, BT_601=1, BT_709=2, SMPTE_170=3, SMPTE_240=4,
/// BT_2020=5, RESERVED=6, RGB=7
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Vp9ColorSpace {
    #[default]
    Unknown = 0,
    Bt601 = 1,
    Bt709 = 2,
    Smpte170 = 3,
    Smpte240 = 4,
    Bt2020 = 5,
    Reserved = 6,
    Rgb = 7,
}

/// VP9 Interpolation filter (from vulkan_video_codec_vp9std.h).
/// Values: EIGHTTAP=0, EIGHTTAP_SMOOTH=1, EIGHTTAP_SHARP=2, BILINEAR=3, SWITCHABLE=4
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Vp9InterpolationFilter {
    #[default]
    EightTap = 0,
    EightTapSmooth = 1,
    EightTapSharp = 2,
    Bilinear = 3,
    Switchable = 4,
}

/// VP9 Picture info flags.
///
/// Matches the bitfield layout of StdVideoDecodeVP9PictureInfoFlags:
///   error_resilient_mode:1, intra_only:1, allow_high_precision_mv:1,
///   refresh_frame_context:1, frame_parallel_decoding_mode:1,
///   segmentation_enabled:1, show_frame:1, UsePrevFrameMvs:1
#[derive(Debug, Clone, Copy, Default)]
pub struct Vp9PictureInfoFlags {
    pub error_resilient_mode: u8,
    pub intra_only: u8,
    pub allow_high_precision_mv: u8,
    pub refresh_frame_context: u8,
    pub frame_parallel_decoding_mode: u8,
    pub segmentation_enabled: u8,
    pub show_frame: u8,
    pub use_prev_frame_mvs: u8,
    pub reset_frame_context: u8,
}

/// VP9 Picture info (decode picture info).
///
/// Internal representation for parser output. Field order matches
/// StdVideoDecodeVP9PictureInfo from Vulkan spec for easy conversion.
#[derive(Debug, Clone)]
pub struct Vp9PictureInfo {
    pub profile: Vp9Profile,
    pub frame_type: Vp9FrameType,
    pub frame_context_idx: u8,
    pub refresh_frame_flags: u8,
    pub ref_frame_sign_bias_mask: u8,
    pub interpolation_filter: Vp9InterpolationFilter,
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_uv_dc: i8,
    pub delta_q_uv_ac: i8,
    pub tile_cols_log2: u8,
    pub tile_rows_log2: u8,
    pub flags: Vp9PictureInfoFlags,
    /// Whether this frame is lossless (derived from quantization parameters).
    pub lossless: bool,
}

impl Default for Vp9PictureInfo {
    fn default() -> Self {
        Self {
            profile: Vp9Profile::Profile0,
            frame_type: Vp9FrameType::Key,
            frame_context_idx: 0,
            refresh_frame_flags: 0,
            ref_frame_sign_bias_mask: 0,
            interpolation_filter: Vp9InterpolationFilter::EightTap,
            base_q_idx: 0,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            flags: Vp9PictureInfoFlags::default(),
            lossless: false,
        }
    }
}

impl Vp9PictureInfoFlags {
    pub fn new() -> Self {
        Self::default()
    }
}

/// VP9 Color config flags.
///
/// Bitfield: color_range:1, reserved:31 = 4 bytes total.
#[derive(Debug, Clone, Copy, Default)]
pub struct Vp9ColorConfigFlags {
    pub color_range: u8,
}

/// VP9 Color configuration.
///
/// Matches `StdVideoVP9ColorConfig` from Vulkan spec:
///   flags (4B) + BitDepth (1B) + subsampling_x (1B) + subsampling_y (1B)
///   + reserved1 (1B) + color_space (4B) = 12 bytes
#[derive(Debug, Clone, Copy)]
pub struct Vp9ColorConfig {
    pub flags: Vp9ColorConfigFlags,
    pub bit_depth: u8,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub color_space: Vp9ColorSpace,
}

impl Default for Vp9ColorConfig {
    fn default() -> Self {
        Self {
            flags: Vp9ColorConfigFlags::default(),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            color_space: Vp9ColorSpace::Bt601,
        }
    }
}

/// VP9 Loop filter flags.
///
/// Bitfield: loop_filter_delta_enabled:1, loop_filter_delta_update:1, reserved:30 = 4 bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct Vp9LoopFilterFlags {
    pub loop_filter_delta_enabled: u8,
    pub loop_filter_delta_update: u8,
    pub update_ref_delta: u8,
    pub update_mode_delta: u8,
}

/// VP9 Loop filter parameters.
///
/// Matches `StdVideoVP9LoopFilter` from Vulkan spec:
///   flags (4B) + loop_filter_level (1B) + loop_filter_sharpness (1B)
///   + update_ref_delta (1B) + loop_filter_ref_deltas[4] (4B)
///   + update_mode_delta (1B) + loop_filter_mode_deltas[2] (2B) = 13 bytes
#[derive(Debug, Clone, Copy)]
pub struct Vp9LoopFilter {
    pub flags: Vp9LoopFilterFlags,
    pub loop_filter_level: u8,
    pub loop_filter_sharpness: u8,
    pub update_ref_delta: u8,
    /// Loop filter reference frame deltas [VP9_MAX_REF_FRAMES=4]
    pub loop_filter_ref_deltas: [i8; VP9_MAX_REF_FRAMES as usize],
    pub update_mode_delta: u8,
    /// Loop filter mode adjustment deltas [VP9_LOOP_FILTER_ADJUSTMENTS=2]
    pub loop_filter_mode_deltas: [i8; VP9_LOOP_FILTER_ADJUSTMENTS as usize],
}

impl Default for Vp9LoopFilter {
    fn default() -> Self {
        Self {
            flags: Vp9LoopFilterFlags::default(),
            loop_filter_level: 0,
            loop_filter_sharpness: 0,
            update_ref_delta: 0,
            loop_filter_ref_deltas: [0; VP9_MAX_REF_FRAMES as usize],
            update_mode_delta: 0,
            loop_filter_mode_deltas: [0; VP9_LOOP_FILTER_ADJUSTMENTS as usize],
        }
    }
}

/// VP9 Segmentation flags.
///
/// Bitfield: segmentation_update_map:1, segmentation_temporal_update:1,
/// segmentation_update_data:1, segmentation_abs_or_delta_update:1, reserved:28 = 4 bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct Vp9SegmentationFlags {
    pub segmentation_update_map: u8,
    pub segmentation_temporal_update: u8,
    pub segmentation_update_data: u8,
    pub segmentation_abs_or_delta_update: u8,
}

/// VP9 Segmentation parameters.
///
/// Matches `StdVideoVP9Segmentation` from Vulkan spec.
#[derive(Debug, Clone)]
pub struct Vp9Segmentation {
    pub flags: Vp9SegmentationFlags,
    /// Segmentation tree probabilities [VP9_MAX_SEGMENTATION_TREE_PROBS]
    pub segmentation_tree_probs: [u8; VP9_MAX_SEGMENTATION_TREE_PROBS as usize],
    /// Segmentation prediction probabilities [VP9_MAX_SEGMENTATION_PRED_PROB]
    pub segmentation_pred_prob: [u8; VP9_MAX_SEGMENTATION_PRED_PROB as usize],
    /// Feature enabled flags [VP9_MAX_SEGMENTS]
    pub feature_enabled: [u8; VP9_MAX_SEGMENTS as usize],
    /// Feature data [VP9_MAX_SEGMENTS][VP9_SEG_LVL_MAX]
    pub feature_data: [[i8; VP9_SEG_LVL_MAX as usize]; VP9_MAX_SEGMENTS as usize],
}

impl Default for Vp9Segmentation {
    fn default() -> Self {
        Self {
            flags: Vp9SegmentationFlags::default(),
            segmentation_tree_probs: [VP9_MAX_PROBABILITY;
                VP9_MAX_SEGMENTATION_TREE_PROBS as usize],
            segmentation_pred_prob: [VP9_MAX_PROBABILITY; VP9_MAX_SEGMENTATION_PRED_PROB as usize],
            feature_enabled: [0; VP9_MAX_SEGMENTS as usize],
            feature_data: [[0; VP9_SEG_LVL_MAX as usize]; VP9_MAX_SEGMENTS as usize],
        }
    }
}

/// VP9 parsed frame data (complete parser output).
#[derive(Debug, Clone)]
pub struct Vp9FrameData {
    /// Whether this is a "show existing frame" command.
    pub show_existing_frame: bool,
    /// Frame to show map index (when show_existing_frame is true).
    pub frame_to_show_map_idx: u8,
    /// Whether this is an intra (key) frame.
    pub frame_is_intra: bool,
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Render width in pixels.
    pub render_width: u32,
    /// Render height in pixels.
    pub render_height: u32,
    /// Macroblock columns.
    pub mi_cols: u32,
    /// Macroblock rows.
    pub mi_rows: u32,
    /// 64x64 superblock columns.
    pub sb64_cols: u32,
    /// 64x64 superblock rows.
    pub sb64_rows: u32,
    /// Number of tiles.
    pub num_tiles: u32,
    /// Picture info.
    pub picture_info: Vp9PictureInfo,
    /// Color configuration.
    pub color_config: Vp9ColorConfig,
    /// Loop filter parameters.
    pub loop_filter: Vp9LoopFilter,
    /// Segmentation parameters.
    pub segmentation: Vp9Segmentation,
    /// Compressed header size (from bitstream).
    pub compressed_header_size: u32,
    /// Uncompressed header size in bytes (from frame marker to compressed_header_size).
    pub uncompressed_header_size: u32,
    /// Offset to uncompressed header in bitstream buffer.
    pub uncompressed_header_offset: u32,
    /// Offset to compressed header in bitstream buffer.
    pub compressed_header_offset: u32,
    /// Offset to tiles data in bitstream buffer.
    pub tiles_offset: u32,
    /// Reference frame indices [VP9_REFS_PER_FRAME].
    pub ref_frame_idx: [u8; VP9_REFS_PER_FRAME as usize],
    /// Picture indices for each reference frame [VP9_NUM_REF_FRAMES].
    pub pic_idx: [i32; VP9_NUM_REF_FRAMES as usize],
    /// Offset of this frame within a superframe (0 if not in superframe).
    /// Used to adjust Vulkan decode offsets for superframe frames.
    pub superframe_frame_offset: u32,
}

impl Default for Vp9FrameData {
    fn default() -> Self {
        Self {
            show_existing_frame: false,
            frame_to_show_map_idx: 0,
            frame_is_intra: false,
            frame_width: 0,
            frame_height: 0,
            render_width: 0,
            render_height: 0,
            mi_cols: 0,
            mi_rows: 0,
            sb64_cols: 0,
            sb64_rows: 0,
            num_tiles: 0,
            picture_info: Vp9PictureInfo::default(),
            color_config: Vp9ColorConfig::default(),
            loop_filter: Vp9LoopFilter::default(),
            segmentation: Vp9Segmentation::default(),
            compressed_header_size: 0,
            uncompressed_header_size: 0,
            uncompressed_header_offset: 0,
            compressed_header_offset: 0,
            tiles_offset: 0,
            ref_frame_idx: [0; VP9_REFS_PER_FRAME as usize],
            pic_idx: [-1; VP9_NUM_REF_FRAMES as usize],
            superframe_frame_offset: 0,
        }
    }
}

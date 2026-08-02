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
    pub log2_max_frame_num_minus4: u8,
    pub max_frame_num: u32,
    pub pic_order_cnt_type: u8,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub max_pic_order_cnt_lsb: u32,
    pub max_num_ref_frames: u32,
    pub gaps_in_frame_num_value_allowed_flag: bool,
    pub pic_width_in_mbs_minus1: u16,
    pub pic_height_in_map_units_minus1: u16,
    pub frame_mbs_only_flag: bool,
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
    pub max_num_reorder_frames: u8,
    pub max_dec_frame_buffering: u8,
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
            log2_max_frame_num_minus4: 0,
            max_frame_num: 1,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            max_pic_order_cnt_lsb: 16,
            max_num_ref_frames: 1,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs_minus1: 0,
            pic_height_in_map_units_minus1: 0,
            frame_mbs_only_flag: true,
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
    pub vps_max_layers_minus1: u16,
    pub vps_max_sub_layers_minus1: u8,
    pub vps_temporal_id_nesting_flag: bool,
    pub vps_sub_layer_ordering_info_present_flag: bool,
    // VPS timing info
    pub vps_num_units_in_tick: u32,
    pub vps_time_scale: u32,
    pub vps_num_ticks_poc_diff_one_minus1: u32,
    // DPB management (from StdVideoH265DecPicBufMgr)
    pub max_dec_pic_buffering_minus1: [u8; 7], // MAX_SUB_LAYERS
    pub max_num_reorder_pics: [u8; 7],
    pub max_latency_increase_plus1: [u8; 7],
}

impl H265Vps {
    pub fn new() -> Self {
        Self {
            vps_video_parameter_set_id: 0,
            vps_max_layers_minus1: 0,
            vps_max_sub_layers_minus1: 0,
            vps_temporal_id_nesting_flag: true,
            vps_sub_layer_ordering_info_present_flag: false,
            vps_num_units_in_tick: 0,
            vps_time_scale: 0,
            vps_num_ticks_poc_diff_one_minus1: 0,
            max_dec_pic_buffering_minus1: [0; 7],
            max_num_reorder_pics: [0; 7],
            max_latency_increase_plus1: [0; 7],
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

/// H.265 Sequence Parameter Set.
/// Matches StdVideoH265SequenceParameterSet layout.
#[derive(Debug, Clone)]
pub struct H265Sps {
    pub sps_video_parameter_set_id: u8,
    pub sps_max_sub_layers_minus1: u8,
    pub sps_temporal_id_nesting_flag: bool,
    pub sps_seq_parameter_set_id: u32,
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
}

/// H.265 Short-Term Reference Picture Set.
/// Matches StdVideoH265ShortTermRefPicSet layout.
#[derive(Debug, Clone)]
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

impl Default for H265ShortTermRefPicSet {
    fn default() -> Self {
        Self {
            inter_ref_pic_set_prediction_flag: false,
            delta_idx_minus1: 0,
            use_delta_flag: 0,
            abs_delta_rps_minus1: 0,
            used_by_curr_pic_flag: 0,
            used_by_curr_pic_s0_flag: 0,
            used_by_curr_pic_s1_flag: 0,
            num_negative_pics: 0,
            num_positive_pics: 0,
            delta_poc_s0_minus1: [0; 16],
            delta_poc_s1_minus1: [0; 16],
        }
    }
}

impl H265Sps {
    pub fn new() -> Self {
        Self {
            sps_video_parameter_set_id: 0,
            sps_max_sub_layers_minus1: 0,
            sps_temporal_id_nesting_flag: true,
            sps_seq_parameter_set_id: 0,
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

    // QP and deblocking fields
    pub diff_cu_qp_delta_depth: u8,
    pub pps_cb_qp_offset: i8,
    pub pps_cr_qp_offset: i8,
    pub pps_beta_offset_div2: i8,
    pub pps_tc_offset_div2: i8,

    // Merge and transform skip fields
    pub log2_parallel_merge_level_minus2: u8,
    pub log2_max_transform_skip_block_size_minus2: u8,

    // Chroma QP offset fields
    pub diff_cu_chroma_qp_offset_depth: u8,
    pub chroma_qp_offset_list_len_minus1: u8,
    pub cb_qp_offset_list: [i8; 6],
    pub cr_qp_offset_list: [i8; 6],

    // SAO fields
    pub log2_sao_offset_scale_luma: u8,
    pub log2_sao_offset_scale_chroma: u8,

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

            diff_cu_qp_delta_depth: 0,
            pps_cb_qp_offset: 0,
            pps_cr_qp_offset: 0,
            pps_beta_offset_div2: 0,
            pps_tc_offset_div2: 0,

            log2_parallel_merge_level_minus2: 0,
            log2_max_transform_skip_block_size_minus2: 0,

            diff_cu_chroma_qp_offset_depth: 0,
            chroma_qp_offset_list_len_minus1: 0,
            cb_qp_offset_list: [0; 6],
            cr_qp_offset_list: [0; 6],

            log2_sao_offset_scale_luma: 0,
            log2_sao_offset_scale_chroma: 0,

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
    pub delta_frame_id_length_minus2: u8,
    pub additional_frame_id_length_minus1: u8,
    pub order_hint_bits_minus1: u8,
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
            delta_frame_id_length_minus2: 0,
            additional_frame_id_length_minus1: 0,
            order_hint_bits_minus1: 0,
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

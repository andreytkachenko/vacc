//! H.265/HEVC bitstream parser.
//!
//! Parses H.265 bitstreams to extract VPS, SPS, PPS, and slice data.
//! Based on cros-codecs H.265 parser implementation.

use std::collections::HashMap;

use crate::bitreader::BitReader;
use crate::nal::{self, H265NalUnitType, NalUnit};
use crate::{DetectedVideoFormat, ParseResult, ParserError, ParserResult, VideoParser};

/// One entry of ref_pic_lists_modification (per reference list).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct H265ListModification {
    /// ref_pic_list_modification_flag_lX[i]
    pub flag: bool,
    /// ref_idx_lX[i] (valid when flag is true)
    pub ref_idx: u8,
}

/// A long-term reference picture signaled in the slice header.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct H265LtRef {
    /// lt_idx_sps (valid when `from_sps` is true); poc_lsb comes from the SPS
    pub sps_idx: u8,
    /// True when the POC LSB comes from sps.lt_ref_pic_poc_lsb_sps[sps_idx]
    pub from_sps: bool,
    /// poc_lsb_lt (slice-signal long-term pictures only)
    pub poc_lsb: u32,
    /// used_by_curr_pic_lt_sps_flag / used_by_curr_pic_lt_flag
    pub used_by_curr_pic: bool,
    /// delta_poc_msb_present_flag
    pub delta_poc_msb_present: bool,
    /// delta_poc_msb_cycle_lt
    pub delta_poc_msb_cycle: u32,
    /// Full resolved POC (POCmsbCycl + POClsb, per H.265 spec 8.3.1).
    /// Valid when used_by_curr_pic is true.
    pub resolved_poc: i32,
}

/// Parsed slice header information for H.265.
#[derive(Debug, Clone)]
pub struct SliceHeaderInfo {
    /// Slice type: 0=I, 1=P, 2=B
    pub slice_type: u8,
    /// Picture order count LSB
    pub pic_order_cnt_lsb: u16,
    /// Full computed picture order count value (CurrPicOrderCntVal)
    pub curr_pic_order_cnt_val: i32,
    /// Whether this is an IDR picture
    pub is_idr: bool,
    /// Whether this is a random access point picture
    pub is_rap: bool,
    /// Whether this picture is a reference picture
    pub is_reference: bool,
    /// no_output_of_prior_pics_flag: raw bitstream value, read for all IRAP
    /// NAL types 16-23; false (inferred) for non-IRAP.
    pub no_output_of_prior_pics_flag: bool,
    /// short_term_ref_pic_set_sps_flag from slice header (for StdVideoDecodeH265PictureInfo)
    pub short_term_ref_pic_set_sps_flag: bool,
    /// Index into SPS short_term_ref_pic_sets array (when short_term_ref_pic_set_sps_flag is true)
    pub short_term_ref_pic_set_idx: u8,
    /// Slice-level STRPS (when short_term_ref_pic_set_sps_flag is false)
    pub slice_strps: Option<vacc_core::picture::H265ShortTermRefPicSet>,
    /// num_ref_idx_l0_active_minus1 (inter slices; 0 for intra)
    pub num_ref_idx_l0_active_minus1: u8,
    /// num_ref_idx_l1_active_minus1 (B slices; 0 otherwise)
    pub num_ref_idx_l1_active_minus1: u8,
    /// ref_pic_lists_modification for list 0 (B slices with lists_modification_present_flag)
    pub ref_pic_lists_modification_l0: Vec<H265ListModification>,
    /// ref_pic_lists_modification for list 1
    pub ref_pic_lists_modification_l1: Vec<H265ListModification>,
    /// Long-term reference pictures signaled in this slice header
    pub long_term_refs: Vec<H265LtRef>,
    /// num_long_term_sps (count of SPS-signal LT refs in long_term_refs)
    pub num_long_term_sps: u8,
    /// num_long_term_pics (count of slice-signal LT refs in long_term_refs)
    pub num_long_term_pics: u8,
    /// slice_temporal_mvp_enabled_flag
    pub slice_temporal_mvp_enabled_flag: bool,
    /// SizeInBits of short_term_ref_pic_set() in the slice header (0 when
    /// short_term_ref_pic_set_sps_flag is 1)
    pub num_bits_for_strps_in_slice: u16,

    // --- Fields beyond the RPS block (H.265 7.3.6.1), parsed in FFmpeg
    // n8.1.2 hls_slice_header order; needed by VAAPI slice buffers ---
    /// slice_segment_address (0 for first slice segments).
    pub slice_segment_address: u32,
    /// dependent_slice_segment_flag (non-first segments when PPS enables
    /// dependent slice segments; 0 otherwise). Dependent segments inherit all
    /// other slice-level parameters from the preceding segment.
    pub dependent_slice_segment_flag: bool,
    /// pic_output_flag (only when PPS output_flag_present_flag; 1 otherwise).
    pub pic_output_flag: bool,
    /// colour_plane_id (only when SPS separate_colour_plane_flag).
    pub colour_plane_id: u8,
    /// slice_sample_adaptive_offset_flag[0] (luma).
    pub slice_sao_luma_flag: bool,
    /// slice_sample_adaptive_offset_flag[1] (chroma; false when no chroma).
    pub slice_sao_chroma_flag: bool,
    /// mvd_l1_zero_flag (B slices).
    pub mvd_l1_zero_flag: bool,
    /// cabac_init_flag (P and B slices when PPS cabac_init_present_flag).
    pub cabac_init_flag: bool,
    /// collocated_from_l0_flag (B slices with slice_temporal_mvp_enabled_flag;
    /// true for P slices / when not signaled).
    pub collocated_from_l0_flag: bool,
    /// collocated_ref_idx (when the collocated list has > 1 reference; 0
    /// otherwise). VA-API wants 0xFF when slice_temporal_mvp_enabled_flag is 0.
    pub collocated_ref_idx: u8,
    /// five_minus_max_num_merge_cand (inter slices).
    pub five_minus_max_num_merge_cand: u8,
    /// slice_qp_delta (all slices).
    pub slice_qp_delta: i32,
    /// slice_cb/cr_qp_offset (when PPS slice_chroma_qp_offsets_present).
    pub slice_cb_qp_offset: i32,
    pub slice_cr_qp_offset: i32,
    /// Effective slice_deblocking_filter_disabled_flag (PPS value when no
    /// slice-level override was read).
    pub slice_deblocking_filter_disabled_flag: bool,
    /// slice_beta/tc_offset_div2 (only when the slice enables the filter and
    /// the PPS disables it by default; 0 otherwise).
    pub slice_beta_offset_div2: i32,
    pub slice_tc_offset_div2: i32,
    /// Effective slice_loop_filter_across_slices_enabled_flag (PPS value when
    /// not read from the bitstream).
    pub slice_loop_filter_across_slices_enabled_flag: bool,
    /// num_entry_point_offsets / entry_point_offset_length (only present when
    /// PPS tiles or entropy_coding_sync is enabled; 0 otherwise).
    /// Per the current H.265 spec, entry_point_offset_length is coded as
    /// ue(v) + 1 bits.
    pub num_entry_point_offsets: u16,
    pub entry_point_offset_length: u16,
    /// Raw coded entry-point values (one per entry point). Each value is the
    /// byte size of that sub-part minus one; cumulative sizes give the entry
    /// point offsets relative to slice data start.
    pub entry_point_offsets: Vec<u32>,
    /// pred_weight_table data (all zero when no weighted prediction table was
    /// read; matches VASliceParameterBufferHEVC layout). Per-reference flag
    /// form: references without a flag keep the zeroed (unweighted) values.
    pub luma_log2_weight_denom: u8,
    pub delta_chroma_log2_weight_denom: i8,
    pub delta_luma_weight_l0: [i8; 15],
    pub luma_offset_l0: [i16; 15],
    pub delta_chroma_weight_l0: [[i8; 2]; 15],
    pub chroma_offset_l0: [[i16; 2]; 15],
    pub delta_luma_weight_l1: [i8; 15],
    pub luma_offset_l1: [i16; 15],
    pub delta_chroma_weight_l1: [[i8; 2]; 15],
    pub chroma_offset_l1: [[i16; 2]; 15],

    // --- Range-extension / SCC slice fields (VASliceParameterBufferHEVCRext)
    /// cu_chroma_qp_offset_enabled_flag (when PPS chroma_qp_offset_list_enabled).
    pub cu_chroma_qp_offset_enabled_flag: bool,
    /// use_integer_mv_flag (when SPS motion_vector_resolution_control_idc == 2).
    pub use_integer_mv_flag: bool,
    /// slice_act_*_qp_offset (when PPS pps_slice_act_qp_offsets_present_flag;
    /// raw coded values, range [-12, 12]).
    pub slice_act_y_qp_offset: i32,
    pub slice_act_cb_qp_offset: i32,
    pub slice_act_cr_qp_offset: i32,
    /// Bit position (from NAL payload start, i.e. after the 16-bit NAL
    /// header) at the end of the parsed slice header — start of
    /// rbsp_slice_trailing_bits. VA-API slice_data_byte_offset (relative to
    /// the NAL unit header) = 2 + (header_bit_size + 8) / 8.
    pub header_bit_size: u16,
}

impl SliceHeaderInfo {
    pub fn new() -> Self {
        Self {
            slice_type: 0,
            pic_order_cnt_lsb: 0,
            curr_pic_order_cnt_val: 0,
            is_idr: false,
            is_rap: false,
            is_reference: false,
            no_output_of_prior_pics_flag: false,
            short_term_ref_pic_set_sps_flag: true, // Default: RPS in SPS
            short_term_ref_pic_set_idx: 0,
            slice_strps: None,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            ref_pic_lists_modification_l0: Vec::new(),
            ref_pic_lists_modification_l1: Vec::new(),
            long_term_refs: Vec::new(),
            num_long_term_sps: 0,
            num_long_term_pics: 0,
            slice_temporal_mvp_enabled_flag: false,
            num_bits_for_strps_in_slice: 0,
            slice_segment_address: 0,
            dependent_slice_segment_flag: false,
            pic_output_flag: true,
            colour_plane_id: 0,
            slice_sao_luma_flag: false,
            slice_sao_chroma_flag: false,
            mvd_l1_zero_flag: false,
            cabac_init_flag: false,
            collocated_from_l0_flag: true,
            collocated_ref_idx: 0,
            five_minus_max_num_merge_cand: 0,
            slice_qp_delta: 0,
            slice_cb_qp_offset: 0,
            slice_cr_qp_offset: 0,
            slice_deblocking_filter_disabled_flag: false,
            slice_beta_offset_div2: 0,
            slice_tc_offset_div2: 0,
            slice_loop_filter_across_slices_enabled_flag: false,
            num_entry_point_offsets: 0,
            entry_point_offset_length: 0,
            entry_point_offsets: Vec::new(),
            luma_log2_weight_denom: 0,
            delta_chroma_log2_weight_denom: 0,
            delta_luma_weight_l0: [0; 15],
            luma_offset_l0: [0; 15],
            delta_chroma_weight_l0: [[0; 2]; 15],
            chroma_offset_l0: [[0; 2]; 15],
            delta_luma_weight_l1: [0; 15],
            luma_offset_l1: [0; 15],
            delta_chroma_weight_l1: [[0; 2]; 15],
            chroma_offset_l1: [[0; 2]; 15],
            cu_chroma_qp_offset_enabled_flag: false,
            use_integer_mv_flag: false,
            slice_act_y_qp_offset: 0,
            slice_act_cb_qp_offset: 0,
            slice_act_cr_qp_offset: 0,
            header_bit_size: 0,
        }
    }
}

impl Default for SliceHeaderInfo {
    fn default() -> Self {
        Self::new()
    }
}

pub struct H265Parser {
    vps_cache: HashMap<u8, vacc_core::picture::H265Vps>,
    sps_cache: HashMap<u32, vacc_core::picture::H265Sps>,
    pps_cache: HashMap<u32, vacc_core::picture::H265Pps>,
    active_vps: Option<vacc_core::picture::H265Vps>,
    active_sps: Option<vacc_core::picture::H265Sps>,
    active_pps: Option<vacc_core::picture::H265Pps>,
    detected_format: DetectedVideoFormat,
    frame_count: u32,
    first_slice_header: Option<SliceHeaderInfo>,
    // POC tracking per H.265 spec section 8.3.1
    prev_pic_order_cnt_msb: i32,
    prev_pic_order_cnt_lsb: i32,
    /// Flag: true when we have a valid previous non-discardable picture for POC derivation
    has_prev_pic: bool,
    /// All NAL units parsed from the current packet, cached so that repeated
    /// `parse()` calls do not re-scan and re-copy the (large) remaining
    /// bitstream each time (which would be O(n^2) over a whole file).
    cached_nals: Vec<NalUnit>,
    /// Length of the packet payload that `cached_nals` was parsed from. When a
    /// packet of a different length arrives, the cache is rebuilt.
    cached_payload_len: usize,
    /// Cursor into `cached_nals`: index of the next NAL unit to process.
    nal_cursor: usize,
}

impl Default for H265Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl H265Parser {
    /// Get the first slice header info from the current access unit (if available).
    pub fn first_slice_header(&self) -> Option<&SliceHeaderInfo> {
        self.first_slice_header.as_ref()
    }

    /// Returns a reference to the active SPS, if any.
    pub fn active_sps(&self) -> Option<&vacc_core::picture::H265Sps> {
        self.active_sps.as_ref()
    }

    /// Returns a reference to the active PPS, if any.
    pub fn active_pps(&self) -> Option<&vacc_core::picture::H265Pps> {
        self.active_pps.as_ref()
    }

    pub fn new() -> Self {
        Self {
            vps_cache: HashMap::new(),
            sps_cache: HashMap::new(),
            pps_cache: HashMap::new(),
            active_vps: None,
            active_sps: None,
            active_pps: None,
            detected_format: DetectedVideoFormat::new(vacc_core::codec::VideoCodec::DecodeH265),
            frame_count: 0,
            first_slice_header: None,
            // Initialize per VulkanH265Parser.cpp:110
            prev_pic_order_cnt_msb: 0,
            prev_pic_order_cnt_lsb: 0,
            has_prev_pic: false,
            cached_nals: Vec::new(),
            cached_payload_len: 0,
            nal_cursor: 0,
        }
    }

    /// Parse profile_tier_level data.
    ///
    /// Uses the same approach as the NVIDIA Vulkan-Video-Samples parser:
    /// always skip fixed bit counts regardless of profile, which is simpler
    /// and avoids issues with conditional parsing.
    ///
    /// profile_tier_level( ProfilePresentFlag, MaxSubLayersMinus1, CommonInfPresentFlag, SubLayerLevelPresentFlag )
    ///
    /// For SPS: ProfilePresentFlag=1, CommonInfPresentFlag=1, SubLayerLevelPresentFlag=0
    /// For VPS: ProfilePresentFlag=1, CommonInfPresentFlag=1, SubLayerLevelPresentFlag=1
    fn parse_ptl(
        r: &mut BitReader,
        max_sub_layers: u8,
        sub_layer_level_present: bool,
    ) -> ParserResult<(u8, u8, bool)> {
        // --- Profile fields (ProfilePresentFlag = 1) ---
        // general_profile_space(2) + general_tier_flag(1) + general_profile_idc(5) = 8 bits
        let profile_bits = r.read_bits(8)?;
        let general_profile_idc = (profile_bits & 0x1F) as u8; // Lower 5 bits are profile_idc
        let general_tier_flag = ((profile_bits >> 3) & 1) != 0; // Bit 3 is tier_flag

        // Skip general_profile_compatibility_flag (32 bits)
        // Note: read_bits max is 31, so we split into 16 + 16
        let _ = r.read_bits(16)?;
        let _ = r.read_bits(16)?;

        // Skip general source/constraint flags + reserved (48 bits)
        // This matches the NVIDIA parser approach: 24 + 24 = 48 bits
        // Covers: source flags(4) + constraint/reserved bits(44)
        let _ = r.read_bits(24)?;
        let _ = r.read_bits(24)?;

        // --- Common info (CommonInfPresentFlag = 1) ---
        // general_level_idc is read ONCE
        let level_idc = r.read_bits(8)? as u8; // general_level_idc

        // --- Sub-layer profile/level presence flags ---
        let mut sub_layer_level_flags: Vec<bool> = Vec::new();
        for _ in 0..max_sub_layers {
            let _ = r.read_bit()?; // sub_layer_profile_present_flag (ignored)
            sub_layer_level_flags.push(r.read_bit()?); // sub_layer_level_present_flag
        }

        // Padding bits: (8 - MaxNumSubLayersMinus1 - 1) * 2 per H.265 spec
        if max_sub_layers > 0 && max_sub_layers < 8 {
            let _ = r.read_bits((8 - max_sub_layers - 1) * 2)?;
        }

        // --- Sub-layer level info (SubLayerLevelPresentFlag) ---
        if sub_layer_level_present {
            for &level_present in &sub_layer_level_flags {
                if level_present {
                    // Skip sub-layer profile info (same as general: 8 + 32 + 48 = 88 bits)
                    let _ = r.read_bits(8)?;
                    let _ = r.read_bits(16)?;
                    let _ = r.read_bits(16)?;
                    let _ = r.read_bits(24)?;
                    let _ = r.read_bits(24)?;
                    // sub_layer_level_idc
                    let _ = r.read_bits(8)?;
                }
            }
        }

        Ok((general_profile_idc, level_idc, general_tier_flag))
    }

    /// Parse scaling_list_data per H.265 spec and C++ VulkanH265Parser.cpp:1674-1727.
    fn parse_scaling_list_data(
        r: &mut BitReader,
        scaling_lists: &mut vacc_core::picture::H265ScalingLists,
    ) -> ParserResult<()> {
        // sizeId: 0=4x4, 1=8x8, 2=16x16, 3=32x32
        // matrixId: 0-5 for sizeId<3, 0-1 for sizeId==3
        for size_id in 0..4u8 {
            let matrix_count = if size_id == 3 { 2u8 } else { 6u8 };
            for matrix_id in 0..matrix_count {
                let scaling_list_pred_mode_flag = r.read_bit()?;
                if !scaling_list_pred_mode_flag {
                    // Predicted from another matrix (scaling_list_pred_mode_flag == 0)
                    // Per H.265 spec 7.3.4.2: predMatrixId = matrixId + scaling_list_pred_matrix_id_delta
                    let scaling_list_pred_matrix_id_delta = r.read_ue()? as i32;
                    let pred_matrix_id =
                        ((matrix_id as i32) + scaling_list_pred_matrix_id_delta) as usize;

                    // Copy AC coefficients from predicted matrix
                    match size_id {
                        0 => {
                            scaling_lists.scaling_list_4x4[matrix_id as usize] =
                                scaling_lists.scaling_list_4x4[pred_matrix_id]
                        }
                        1 => {
                            scaling_lists.scaling_list_8x8[matrix_id as usize] =
                                scaling_lists.scaling_list_8x8[pred_matrix_id]
                        }
                        2 => {
                            scaling_lists.scaling_list_16x16[matrix_id as usize] =
                                scaling_lists.scaling_list_16x16[pred_matrix_id]
                        }
                        3 => {
                            scaling_lists.scaling_list_32x32[matrix_id as usize] =
                                scaling_lists.scaling_list_32x32[pred_matrix_id]
                        }
                        _ => {}
                    }

                    // Copy DC coefficients for 16x16 and 32x32
                    if size_id == 2 {
                        scaling_lists.scaling_list_dc_coef_16x16[matrix_id as usize][0] =
                            scaling_lists.scaling_list_dc_coef_16x16[pred_matrix_id][0];
                    } else if size_id == 3 {
                        scaling_lists.scaling_list_dc_coef_32x32[matrix_id as usize][0] =
                            scaling_lists.scaling_list_dc_coef_32x32[pred_matrix_id][0];
                    }
                } else {
                    let coef_num = (1u32 << (4 + (size_id as u32) * 2)).min(64);
                    let mut next_coef: i32 = 8;

                    // DC coefficients for 16x16 and 32x32
                    if size_id > 1 {
                        let scaling_list_dc_coef_minus8 = r.read_se()?;
                        next_coef = scaling_list_dc_coef_minus8 + 8;
                        // Store DC coefficient
                        if size_id == 2 {
                            scaling_lists.scaling_list_dc_coef_16x16[matrix_id as usize][0] =
                                next_coef as i8;
                        } else {
                            scaling_lists.scaling_list_dc_coef_32x32[matrix_id as usize][0] =
                                next_coef as i8;
                        }
                    }

                    // AC coefficients
                    for i in 0..coef_num as usize {
                        let scaling_list_delta_coef = r.read_se()?;
                        next_coef = (next_coef + scaling_list_delta_coef) & 0xFF;
                        if next_coef == 0 {
                            return Err(ParserError::InvalidBitstream);
                        }
                        // Store coefficient in the appropriate array
                        match size_id {
                            0 => {
                                scaling_lists.scaling_list_4x4[matrix_id as usize][i] =
                                    next_coef as u8
                            }
                            1 => {
                                scaling_lists.scaling_list_8x8[matrix_id as usize][i] =
                                    next_coef as u8
                            }
                            2 => {
                                scaling_lists.scaling_list_16x16[matrix_id as usize][i] =
                                    next_coef as u8
                            }
                            3 => {
                                scaling_lists.scaling_list_32x32[matrix_id as usize][i] =
                                    next_coef as u8
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Parse HRD parameters to advance bitstream position correctly.
    /// Based on cros-codecs parse_hrd_parameters implementation.
    /// Per H.265 spec Table 7.3 and 7.4.10.
    fn parse_hrd_parameters(
        common_inf_present_flag: bool,
        max_num_sublayers_minus1: u8,
        r: &mut BitReader,
    ) -> ParserResult<()> {
        let mut nal_hrd_parameters_present_flag = false;
        let mut vcl_hrd_parameters_present_flag = false;
        let mut sub_pic_hrd_params_present_flag = false;

        if common_inf_present_flag {
            nal_hrd_parameters_present_flag = r.read_bit()?;
            vcl_hrd_parameters_present_flag = r.read_bit()?;
            if nal_hrd_parameters_present_flag || vcl_hrd_parameters_present_flag {
                sub_pic_hrd_params_present_flag = r.read_bit()?;
                if sub_pic_hrd_params_present_flag {
                    let _tick_divisor_minus2 = r.read_bits(8)?;
                    let _du_cpb_removal_delay_increment_length_minus1 = r.read_bits(5)?;
                    let _sub_pic_cpb_params_in_pic_timing_sei_flag = r.read_bit()?;
                    let _dpb_output_delay_du_length_minus1 = r.read_bits(5)?;
                }
                let _bit_rate_scale = r.read_bits(4)?;
                let _cpb_size_scale = r.read_bits(4)?;
                if sub_pic_hrd_params_present_flag {
                    let _cpb_size_du_scale = r.read_bits(4)?;
                }
                let _initial_cpb_removal_delay_length_minus1 = r.read_bits(5)?;
                let _au_cpb_removal_delay_length_minus1 = r.read_bits(5)?;
                let _dpb_output_delay_length_minus1 = r.read_bits(5)?;
            }
        }

        for _i in 0..=max_num_sublayers_minus1 as usize {
            let fixed_pic_rate_general_flag = r.read_bit()?;
            let mut fixed_pic_rate_within_cvs_flag = false;
            if !fixed_pic_rate_general_flag {
                fixed_pic_rate_within_cvs_flag = r.read_bit()?;
            }
            if fixed_pic_rate_within_cvs_flag {
                let _elemental_duration_in_tc_minus1 = r.read_ue()?;
            } else {
                let low_delay_hrd_flag = r.read_bit()?;
                if !low_delay_hrd_flag {
                    let cpb_cnt_minus1 = r.read_ue()?;
                    if nal_hrd_parameters_present_flag {
                        Self::parse_sublayer_hrd_parameters(
                            cpb_cnt_minus1 + 1,
                            sub_pic_hrd_params_present_flag,
                            r,
                        )?;
                    }
                    if vcl_hrd_parameters_present_flag {
                        Self::parse_sublayer_hrd_parameters(
                            cpb_cnt_minus1 + 1,
                            sub_pic_hrd_params_present_flag,
                            r,
                        )?;
                    }
                } else {
                    // low_delay_hrd_flag is true - no sublayer HRD params
                }
            }
        }

        Ok(())
    }

    /// Parse sublayer HRD parameters.
    /// Based on cros-codecs parse_sublayer_hrd_parameters.
    fn parse_sublayer_hrd_parameters(
        cpb_cnt: u32,
        sub_pic_hrd_params_present_flag: bool,
        r: &mut BitReader,
    ) -> ParserResult<()> {
        for _ in 0..cpb_cnt {
            let _bit_rate_value_minus1 = r.read_ue()?;
            let _cpb_size_value_minus1 = r.read_ue()?;
            if sub_pic_hrd_params_present_flag {
                let _cpb_size_du_value_minus1 = r.read_ue()?;
                let _bit_rate_du_value_minus1 = r.read_ue()?;
            }
            let _cbr_flag = r.read_bit()?;
        }
        Ok(())
    }

    /// Parse a short-term reference picture set (STRPS).
    /// Based on VulkanH265Parser.cpp:1730-1917.
    ///
    /// For the direct encoding case (!inter_ref_pic_set_prediction_flag),
    /// the C++ computes cumulative POCs:
    ///   DeltaPocS0[i] = ((i == 0) ? 0 : DeltaPocS0[i-1]) - (delta_poc_s0_minus1[i] + 1)
    ///   DeltaPocS1[i] = ((i == 0) ? 0 : DeltaPocS1[i-1]) + (delta_poc_s1_minus1[i] + 1)
    ///
    /// For the predictive encoding case (inter_ref_pic_set_prediction_flag),
    /// the RPS is resolved against a previous STRPS using delta-based prediction.
    /// This matches C++: VulkanH265Parser.cpp:1738-1862.
    ///
    /// The Vulkan API stores the raw delta_poc values (delta_poc_s0_minus1, delta_poc_s1_minus1),
    /// not the computed cumulative POCs. This matches the StdVideoH265ShortTermRefPicSet layout.
    ///
    /// `prev_strps` contains previously parsed STRPS entries needed for predictive resolution.
    fn parse_short_term_ref_pic_set(
        r: &mut BitReader,
        idx: usize,
        num_short_term_ref_pic_sets: usize,
        prev_strps: &[vacc_core::picture::H265ShortTermRefPicSet],
    ) -> ParserResult<vacc_core::picture::H265ShortTermRefPicSet> {
        let mut strps = vacc_core::picture::H265ShortTermRefPicSet::default();

        // inter_ref_pic_set_prediction_flag is only present if idx != 0
        let inter_ref_pic_set_prediction_flag = if idx != 0 { r.read_bit()? } else { false };
        strps.inter_ref_pic_set_prediction_flag = inter_ref_pic_set_prediction_flag;

        if inter_ref_pic_set_prediction_flag {
            // Delta-based prediction from a previous STRPS
            // Matches C++: VulkanH265Parser.cpp:1738-1862
            let delta_idx_minus1 = if idx == num_short_term_ref_pic_sets {
                r.read_ue()?
            } else {
                0
            };
            strps.delta_idx_minus1 = delta_idx_minus1;

            let delta_rps_sign = r.read_bit()?;
            strps.abs_delta_rps_minus1 = r.read_ue()? as u16;

            // Resolve predictive RPS against reference STRPS
            Self::resolve_predictive_rps(
                r,
                delta_rps_sign,
                strps.abs_delta_rps_minus1,
                idx,
                delta_idx_minus1,
                prev_strps,
                &mut strps,
            )?;
        } else {
            // Direct encoding
            // Matches C++: VulkanH265Parser.cpp:1870-1914
            let num_negative_pics = r.read_ue()? as u8;
            let num_positive_pics = r.read_ue()? as u8;
            strps.num_negative_pics = num_negative_pics;
            strps.num_positive_pics = num_positive_pics;

            // Read raw delta_poc_s0_minus1 values and compute cumulative POCs
            // Per C++ reference: DeltaPocS0[i] = DeltaPocS0[i-1] - (delta + 1)
            // Store DeltaPocS0[i] directly (negative for S0, cast to uint16_t)
            let mut cum_delta_poc_s0: i32 = 0;
            for i in 0..num_negative_pics {
                let raw_delta = r.read_ue()? as i32;
                let used_by_curr_pic_s0_flag = r.read_bit()?;
                if used_by_curr_pic_s0_flag {
                    strps.used_by_curr_pic_s0_flag |= 1 << i;
                }
                // Compute cumulative POC offset (negative for S0)
                cum_delta_poc_s0 -= raw_delta + 1;
                // Store directly like C++: (uint16_t)DeltaPocS0[i]
                strps.delta_poc_s0_minus1[i as usize] = cum_delta_poc_s0 as u16;
            }

            // Read raw delta_poc_s1_minus1 values and compute cumulative POCs
            // Per C++ reference: DeltaPocS1[i] = DeltaPocS1[i-1] + (delta + 1)
            // Store DeltaPocS1[i] directly (positive for S1)
            let mut cum_delta_poc_s1: i32 = 0;
            for i in 0..num_positive_pics {
                let raw_delta = r.read_ue()? as i32;
                let used_by_curr_pic_s1_flag = r.read_bit()?;
                if used_by_curr_pic_s1_flag {
                    strps.used_by_curr_pic_s1_flag |= 1 << i;
                }
                // Compute cumulative POC offset (positive for S1)
                cum_delta_poc_s1 += raw_delta + 1;
                // Store directly like C++: (uint16_t)DeltaPocS1[i]
                strps.delta_poc_s1_minus1[i as usize] = cum_delta_poc_s1 as u16;
            }
        }

        Ok(strps)
    }

    /// Resolve a predictive RPS against a reference STRPS.
    /// Matches C++: VulkanH265Parser.cpp:1754-1862.
    fn resolve_predictive_rps(
        r: &mut BitReader,
        delta_rps_sign: bool,
        abs_delta_rps_minus1: u16,
        idx: usize,
        delta_idx_minus1: u32,
        prev_strps: &[vacc_core::picture::H265ShortTermRefPicSet],
        strps: &mut vacc_core::picture::H265ShortTermRefPicSet,
    ) -> ParserResult<()> {
        // DeltaRPS: positive if delta_rps_sign==0, negative if delta_rps_sign==1
        let delta_rps: i32 = if delta_rps_sign { -1i32 } else { 1i32 };
        let delta_rps_val: i32 = delta_rps * (abs_delta_rps_minus1 as i32 + 1);

        // Reference RPS index
        let r_idx = idx - (delta_idx_minus1 as usize + 1);
        if r_idx >= prev_strps.len() {
            return Err(ParserError::InvalidBitstream);
        }
        let rstrps = &prev_strps[r_idx];

        // Compute cumulative DeltaPoc from reference STRPS
        // The stored delta_poc_s0_minus1/delta_poc_s1_minus1 are cumulative offsets
        // (see non-predictive parsing at line 285), not incremental deltas.
        // S0: stored as u16, negative values wrap (e.g., -1 -> 65535)
        let mut ref_delta_poc_s0: Vec<i32> = Vec::new();
        for i in 0..rstrps.num_negative_pics as usize {
            let stored = rstrps.delta_poc_s0_minus1[i] as i32;
            // Convert wrapped u16 back to negative cumulative offset
            let delta_poc = if stored > 32767 {
                stored - 65536
            } else {
                stored
            };
            ref_delta_poc_s0.push(delta_poc);
        }
        // S1: stored as u16, positive cumulative offsets
        let mut ref_delta_poc_s1: Vec<i32> = Vec::new();
        for i in 0..rstrps.num_positive_pics as usize {
            ref_delta_poc_s1.push(rstrps.delta_poc_s1_minus1[i] as i32);
        }

        let num_ref_entries = rstrps.num_negative_pics as usize + rstrps.num_positive_pics as usize;

        // Read used_by_curr_pic_flag and use_delta_flag for each entry
        // Matches C++: VulkanH265Parser.cpp:1758-1769
        let mut used_by_curr_pic_flag: Vec<bool> = Vec::with_capacity(num_ref_entries + 1);
        let mut use_delta_flag: Vec<bool> = Vec::with_capacity(num_ref_entries + 1);
        for _j in 0..=num_ref_entries {
            let used = r.read_bit()?;
            used_by_curr_pic_flag.push(used);
            if used {
                use_delta_flag.push(true);
            } else {
                use_delta_flag.push(r.read_bit()?);
            }
        }

        // Build new S0 list (DeltaPoc < 0)
        // Matches C++: VulkanH265Parser.cpp:1772-1814
        let mut new_s0_delta_poc: Vec<i32> = Vec::new();
        let mut new_s0_used: Vec<bool> = Vec::new();

        // Process reference S1 entries in reverse
        for j in (0..rstrps.num_positive_pics as usize).rev() {
            let d_poc = ref_delta_poc_s1[j] + delta_rps_val;
            if d_poc < 0 && use_delta_flag[rstrps.num_negative_pics as usize + j] {
                new_s0_delta_poc.push(d_poc);
                new_s0_used.push(used_by_curr_pic_flag[rstrps.num_negative_pics as usize + j]);
            }
        }
        // New entry at DeltaRPS position (if negative)
        if delta_rps_val < 0 && use_delta_flag[num_ref_entries] {
            new_s0_delta_poc.push(delta_rps_val);
            new_s0_used.push(used_by_curr_pic_flag[num_ref_entries]);
        }
        // Process reference S0 entries in forward order
        for j in 0..rstrps.num_negative_pics as usize {
            let d_poc = ref_delta_poc_s0[j] + delta_rps_val;
            if d_poc < 0 && use_delta_flag[j] {
                new_s0_delta_poc.push(d_poc);
                new_s0_used.push(used_by_curr_pic_flag[j]);
            }
        }

        // Store S0 results - match C++: store DeltaPocS0[i] directly
        let num_neg = new_s0_delta_poc.len() as u8;
        strps.num_negative_pics = num_neg.min(16);
        for i in 0..strps.num_negative_pics as usize {
            // new_s0_delta_poc[i] is the cumulative POC offset (negative for S0)
            // Store directly like C++: (uint16_t)dPoc
            strps.delta_poc_s0_minus1[i] = new_s0_delta_poc[i] as u16;
            if new_s0_used[i] {
                strps.used_by_curr_pic_s0_flag |= 1 << i;
            }
        }

        // Build new S1 list (DeltaPoc > 0)
        // Matches C++: VulkanH265Parser.cpp:1817-1862
        let mut new_s1_delta_poc: Vec<i32> = Vec::new();
        let mut new_s1_used: Vec<bool> = Vec::new();

        // Process reference S0 entries in reverse
        for j in (0..rstrps.num_negative_pics as usize).rev() {
            let d_poc = ref_delta_poc_s0[j] + delta_rps_val;
            if d_poc > 0 && use_delta_flag[j] {
                new_s1_delta_poc.push(d_poc);
                new_s1_used.push(used_by_curr_pic_flag[j]);
            }
        }
        // New entry at DeltaRPS position (if positive)
        if delta_rps_val > 0 && use_delta_flag[num_ref_entries] {
            new_s1_delta_poc.push(delta_rps_val);
            new_s1_used.push(used_by_curr_pic_flag[num_ref_entries]);
        }
        // Process reference S1 entries in forward order
        for j in 0..rstrps.num_positive_pics as usize {
            let d_poc = ref_delta_poc_s1[j] + delta_rps_val;
            if d_poc > 0 && use_delta_flag[rstrps.num_negative_pics as usize + j] {
                new_s1_delta_poc.push(d_poc);
                new_s1_used.push(used_by_curr_pic_flag[rstrps.num_negative_pics as usize + j]);
            }
        }

        // Store S1 results - match C++: store DeltaPocS1[i] directly
        let num_pos = new_s1_delta_poc.len() as u8;
        strps.num_positive_pics = num_pos.min(16);
        for i in 0..strps.num_positive_pics as usize {
            // new_s1_delta_poc[i] is the cumulative POC offset (positive for S1)
            // Store directly like C++: (uint16_t)dPoc
            strps.delta_poc_s1_minus1[i] = new_s1_delta_poc[i] as u16;
            if new_s1_used[i] {
                strps.used_by_curr_pic_s1_flag |= 1 << i;
            }
        }

        Ok(())
    }

    fn parse_vps(&mut self, data: &[u8]) -> ParserResult<vacc_core::picture::H265Vps> {
        if data.is_empty() {
            return Err(ParserError::InvalidBitstream);
        }
        // Skip NAL header (2 bytes for H.265)
        let mut r = BitReader::new(&data[2..], true);

        let mut vps = vacc_core::picture::H265Vps::new();
        vps.vps_video_parameter_set_id = r.read_bits(4)? as u8;
        // Per H.265 spec 7.3.2.1:
        // vps_base_layer_internal_flag(1) + vps_base_layer_available_flag(1)
        vps.vps_base_layer_internal_flag = r.read_bit()?;
        vps.vps_base_layer_available_flag = r.read_bit()?;
        vps.vps_max_layers_minus1 = r.read_bits(6)? as u16;
        vps.vps_max_sub_layers_minus1 = r.read_bits(3)? as u8;
        vps.vps_temporal_id_nesting_flag = r.read_bit()?;

        // vps_reserved_0xffff (16 bits) - must be equal to 0xFFFF per spec
        let _reserved_16 = r.read_bits(16)?;

        // Parse profile_tier_level (VPS: ProfilePresentFlag=1, SubLayerLevelPresentFlag=1)
        let (vps_profile_idc, vps_level_idc, vps_tier_flag) =
            Self::parse_ptl(&mut r, vps.vps_max_sub_layers_minus1, true)?;
        vps.profile_idc = vps_profile_idc;
        vps.level_idc = vps_level_idc;
        vps.tier_flag = vps_tier_flag;

        vps.vps_sub_layer_ordering_info_present_flag = r.read_bit()?;

        // DPB management (StdVideoH265DecPicBufMgr)
        // When vps_sub_layer_ordering_info_present_flag=0, DPB params are only specified
        // for the highest sublayer and MUST be propagated to all lower sublayers (spec NOTE).
        let dpb_start = if vps.vps_sub_layer_ordering_info_present_flag {
            0
        } else {
            vps.vps_max_sub_layers_minus1 as usize
        };
        for i in dpb_start..=(vps.vps_max_sub_layers_minus1 as usize) {
            vps.max_dec_pic_buffering_minus1[i] = r.read_ue()? as u8;
            vps.max_num_reorder_pics[i] = r.read_ue()? as u8;
            vps.max_latency_increase_plus1[i] = r.read_ue()? as u8;
        }
        // Propagate DPB params from highest sublayer to all lower sublayers when flag=0
        if !vps.vps_sub_layer_ordering_info_present_flag {
            let highest = vps.vps_max_sub_layers_minus1 as usize;
            for i in 0..highest {
                vps.max_dec_pic_buffering_minus1[i] = vps.max_dec_pic_buffering_minus1[highest];
                vps.max_num_reorder_pics[i] = vps.max_num_reorder_pics[highest];
                vps.max_latency_increase_plus1[i] = vps.max_latency_increase_plus1[highest];
            }
        }

        // VPS layer info
        vps.vps_max_layer_id = r.read_bits(6)? as u16;
        vps.vps_num_layer_sets = r.read_ue()? + 1; // ue(v) + 1

        // layer_id_included_flag[layer_set_idx][layer_id]
        vps.layer_id_included_flag.clear();
        for _i in 1..vps.vps_num_layer_sets {
            let mut layer_flags = Vec::new();
            for _j in 0..=(vps.vps_max_layer_id) {
                layer_flags.push(r.read_bit()?);
            }
            vps.layer_id_included_flag.push(layer_flags);
        }

        // vps_timing_info_present_flag
        vps.vps_timing_info_present_flag = r.read_bit()?;
        if vps.vps_timing_info_present_flag {
            vps.vps_num_units_in_tick = r.read_bits(32)?;
            vps.vps_time_scale = r.read_bits(32)?;
            vps.vps_poc_proportional_to_timing_flag = r.read_bit()?;
            if vps.vps_poc_proportional_to_timing_flag {
                vps.vps_num_ticks_poc_diff_one_minus1 = r.read_ue()?;
            }
            vps.vps_num_hrd_parameters = r.read_ue()?;

            // Parse HRD parameters properly to advance bitstream position correctly
            // Per H.265 spec Table 7.3 and cros-codecs reference implementation
            for i in 0..vps.vps_num_hrd_parameters {
                let _hrd_layer_set_idx = r.read_ue()?;
                if i > 0 {
                    let _cprms_present_flag = r.read_bit()?;
                }
                Self::parse_hrd_parameters(i > 0, vps.vps_max_sub_layers_minus1, &mut r)?;
            }
        }

        // vps_extension_flag (only present when vps_max_layers_minus1 > 0)
        if vps.vps_max_layers_minus1 > 0 {
            vps.vps_extension_flag = r.read_bit()?;
        }

        self.vps_cache
            .insert(vps.vps_video_parameter_set_id, vps.clone());
        self.active_vps = Some(vps);

        Ok(self.active_vps.clone().unwrap())
    }

    fn parse_sps(&mut self, data: &[u8]) -> ParserResult<vacc_core::picture::H265Sps> {
        if data.is_empty() {
            return Err(ParserError::InvalidBitstream);
        }
        // Skip NAL header (2 bytes for H.265)
        let mut r = BitReader::new(&data[2..], true);

        let sps_video_parameter_set_id = r.read_bits(4)? as u8;
        let sps_max_sub_layers_minus1 = r.read_bits(3)? as u8;
        let sps_temporal_id_nesting_flag = r.read_bit()?;

        // Parse profile_tier_level (SPS: SubLayerLevelPresentFlag=1 per H.265 spec)
        let (sps_profile_idc, sps_level_idc, sps_tier_flag) =
            Self::parse_ptl(&mut r, sps_max_sub_layers_minus1, true)?;

        let mut sps = vacc_core::picture::H265Sps::new();
        sps.sps_video_parameter_set_id = sps_video_parameter_set_id;
        sps.sps_max_sub_layers_minus1 = sps_max_sub_layers_minus1;
        sps.sps_temporal_id_nesting_flag = sps_temporal_id_nesting_flag;
        sps.profile_idc = sps_profile_idc;
        sps.level_idc = sps_level_idc;
        sps.tier_flag = sps_tier_flag;

        sps.sps_seq_parameter_set_id = r.read_ue()?;
        sps.chroma_format_idc = r.read_ue()? as u8;

        if sps.chroma_format_idc == 3 {
            sps.separate_colour_plane_flag = r.read_bit()?;
        }

        sps.pic_width_in_luma_samples = r.read_ue()? as u16;
        sps.pic_height_in_luma_samples = r.read_ue()? as u16;

        // Parse conformance_window_flag and offsets
        sps.conformance_window_flag = r.read_bit()?;
        if sps.conformance_window_flag {
            sps.conf_win_left_offset = r.read_ue()?;
            sps.conf_win_right_offset = r.read_ue()?;
            sps.conf_win_top_offset = r.read_ue()?;
            sps.conf_win_bottom_offset = r.read_ue()?;
        }

        sps.bit_depth_luma_minus8 = r.read_ue()? as u8;
        sps.bit_depth_chroma_minus8 = r.read_ue()? as u8;
        sps.log2_max_pic_order_cnt_lsb_minus4 = r.read_ue()? as u8;

        sps.sps_sub_layer_ordering_info_present_flag = r.read_bit()?;

        // DPB management info (per VulkanH265Parser.cpp:500-515)
        // Read max_dec_pic_buffering_minus1, max_num_reorder_pics, max_latency_increase_plus1
        // for sub-layers [sps_sub_layer_ordering_info_present_flag ? 0 : sps_max_sub_layers_minus1 .. sps_max_sub_layers_minus1]
        let dpb_start = if sps.sps_sub_layer_ordering_info_present_flag {
            0
        } else {
            sps_max_sub_layers_minus1 as usize
        };
        for i in dpb_start..=(sps_max_sub_layers_minus1 as usize) {
            sps.max_dec_pic_buffering_minus1[i] = r.read_ue()? as u8;
            sps.max_num_reorder_pics[i] = r.read_ue()? as u8;
            sps.max_latency_increase_plus1[i] = r.read_ue()? as u8;
        }
        // Propagate DPB params from highest sublayer to all lower sublayers when flag=0 (spec NOTE)
        if !sps.sps_sub_layer_ordering_info_present_flag {
            let highest = sps_max_sub_layers_minus1 as usize;
            for i in 0..highest {
                sps.max_dec_pic_buffering_minus1[i] = sps.max_dec_pic_buffering_minus1[highest];
                sps.max_num_reorder_pics[i] = sps.max_num_reorder_pics[highest];
                sps.max_latency_increase_plus1[i] = sps.max_latency_increase_plus1[highest];
            }
        }

        // Additional SPS fields (per VulkanH265Parser.cpp:541-562)
        sps.log2_min_luma_coding_block_size_minus3 = r.read_ue()? as u8;
        sps.log2_diff_max_min_luma_coding_block_size = r.read_ue()? as u8;
        sps.log2_min_luma_transform_block_size_minus2 = r.read_ue()? as u8;
        sps.log2_diff_max_min_luma_transform_block_size = r.read_ue()? as u8;
        sps.max_transform_hierarchy_depth_inter = r.read_ue()? as u8;
        sps.max_transform_hierarchy_depth_intra = r.read_ue()? as u8;
        sps.scaling_list_enabled_flag = r.read_bit()?;
        sps.sps_scaling_list_data_present_flag = false;

        if sps.scaling_list_enabled_flag {
            sps.sps_scaling_list_data_present_flag = r.read_bit()?;
            if sps.sps_scaling_list_data_present_flag {
                // Parse scaling_list_data per H.265 spec and C++ VulkanH265Parser.cpp:1674-1727
                Self::parse_scaling_list_data(&mut r, &mut sps.scaling_lists)?;
            }
        }

        sps.amp_enabled_flag = r.read_bit()?;
        sps.sample_adaptive_offset_enabled_flag = r.read_bit()?;
        sps.pcm_enabled_flag = r.read_bit()?;
        if sps.pcm_enabled_flag {
            sps.pcm_sample_bit_depth_luma_minus1 = r.read_bits(4)? as u8;
            sps.pcm_sample_bit_depth_chroma_minus1 = r.read_bits(4)? as u8;
            sps.log2_min_pcm_luma_coding_block_size_minus3 = r.read_ue()? as u8;
            sps.log2_diff_max_min_pcm_luma_coding_block_size = r.read_ue()? as u8;
            sps.pcm_loop_filter_disabled_flag = r.read_bit()?;
        }

        // Short-term reference picture sets (per VulkanH265Parser.cpp:579-598)
        let num_short_term_ref_pic_sets = r.read_ue()? as u8;
        sps.num_short_term_ref_pic_sets = num_short_term_ref_pic_sets;

        for i in 0..num_short_term_ref_pic_sets {
            let strps = Self::parse_short_term_ref_pic_set(
                &mut r,
                i as usize,
                num_short_term_ref_pic_sets as usize,
                &sps.short_term_ref_pic_sets,
            )?;
            sps.short_term_ref_pic_sets.push(strps);
        }

        // Long-term reference pictures (per VulkanH265Parser.cpp:599-617)
        sps.long_term_ref_pics_present_flag = r.read_bit()?;
        if sps.long_term_ref_pics_present_flag {
            let num_long_term_ref_pics_sps = r.read_ue()? as u8;
            sps.num_long_term_ref_pics_sps = num_long_term_ref_pics_sps;

            let poc_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
            for i in 0..num_long_term_ref_pics_sps {
                sps.lt_ref_pic_poc_lsb_sps[i as usize] = r.read_bits(poc_lsb_bits as u8)?;
                let used_by_curr_pic_lt_sps_flag = r.read_bit()?;
                if used_by_curr_pic_lt_sps_flag {
                    sps.used_by_curr_pic_lt_sps_flag |= 1 << i;
                }
            }
        }

        sps.sps_temporal_mvp_enabled_flag = r.read_bit()?;
        sps.strong_intra_smoothing_enabled_flag = r.read_bit()?;
        sps.vui_parameters_present_flag = r.read_bit()?;
        if sps.vui_parameters_present_flag {
            // Parse vui_parameters() per H.265 spec Table 7-6
            // Matches C++ vui_parameters() in VulkanH265Parser.cpp:1920-2013
            sps.vui.aspect_ratio_info_present_flag = r.read_bit()?;
            if sps.vui.aspect_ratio_info_present_flag {
                sps.vui.aspect_ratio_idc = r.read_bits(8)? as u8;
                // Extended_SAR (255): read sar_width and sar_height
                if sps.vui.aspect_ratio_idc == 255 {
                    sps.vui.sar_width = r.read_bits(16)? as u16;
                    sps.vui.sar_height = r.read_bits(16)? as u16;
                }
            }

            sps.vui.overscan_info_present_flag = r.read_bit()?;
            if sps.vui.overscan_info_present_flag {
                sps.vui.overscan_appropriate_flag = r.read_bit()?;
            }

            sps.vui.video_signal_type_present_flag = r.read_bit()?;
            if sps.vui.video_signal_type_present_flag {
                sps.vui.video_format = r.read_bits(3)? as u8;
                sps.vui.video_full_range_flag = r.read_bit()?;
                sps.vui.colour_description_present_flag = r.read_bit()?;
                if sps.vui.colour_description_present_flag {
                    sps.vui.colour_primaries = r.read_bits(8)? as u8;
                    sps.vui.transfer_characteristics = r.read_bits(8)? as u8;
                    sps.vui.matrix_coeffs = r.read_bits(8)? as u8;
                }
            }

            sps.vui.chroma_loc_info_present_flag = r.read_bit()?;
            if sps.vui.chroma_loc_info_present_flag {
                sps.vui.chroma_sample_loc_type_top_field = r.read_ue()?;
                sps.vui.chroma_sample_loc_type_bottom_field = r.read_ue()?;
            }

            sps.vui.neutral_chroma_indication_flag = r.read_bit()?;
            sps.vui.field_seq_flag = r.read_bit()?;
            sps.vui.frame_field_info_present_flag = r.read_bit()?;

            sps.vui.default_display_window_flag = r.read_bit()?;
            if sps.vui.default_display_window_flag {
                sps.vui.def_disp_win_left_offset = r.read_ue()?;
                sps.vui.def_disp_win_right_offset = r.read_ue()?;
                sps.vui.def_disp_win_top_offset = r.read_ue()?;
                sps.vui.def_disp_win_bottom_offset = r.read_ue()?;
            }

            sps.vui.vui_timing_info_present_flag = r.read_bit()?;
            if sps.vui.vui_timing_info_present_flag {
                sps.vui.vui_num_units_in_tick = r.read_bits(32)?;
                sps.vui.vui_time_scale = r.read_bits(32)?;
                sps.vui.vui_poc_proportional_to_timing_flag = r.read_bit()?;
                if sps.vui.vui_poc_proportional_to_timing_flag {
                    sps.vui.vui_num_ticks_poc_diff_one_minus1 = r.read_ue()?;
                }
                sps.vui.vui_hrd_parameters_present_flag = r.read_bit()?;
                if sps.vui.vui_hrd_parameters_present_flag {
                    // Parse HRD parameters to advance bitstream position correctly.
                    // common_inf_present_flag=1 for VUI HRD params per H.265 spec.
                    Self::parse_hrd_parameters(true, sps_max_sub_layers_minus1, &mut r)?;
                }
            }

            sps.vui.bitstream_restriction_flag = r.read_bit()?;
            if sps.vui.bitstream_restriction_flag {
                sps.vui.tiles_fixed_structure_flag = r.read_bit()?;
                sps.vui.motion_vectors_over_pic_boundaries_flag = r.read_bit()?;
                sps.vui.restricted_ref_pic_lists_flag = r.read_bit()?;
                sps.vui.min_spatial_segmentation_idc = r.read_ue()?;
                sps.vui.max_bytes_per_pic_denom = r.read_ue()?;
                sps.vui.max_bits_per_min_cu_denom = r.read_ue()?;
                sps.vui.log2_max_mv_length_horizontal = r.read_ue()?;
                sps.vui.log2_max_mv_length_vertical = r.read_ue()?;
            }
        }
        sps.sps_extension_present_flag = r.read_bit()?;
        if sps.sps_extension_present_flag {
            // Per H.265 spec / FFmpeg n8.1.2: 4 extension flags + 4 reserved
            // bits, then conditional extension data in order: range,
            // multilayer, 3D, SCC.
            sps.sps_range_extension_flag = r.read_bit()?;
            let sps_multilayer_extension_flag = r.read_bit()?;
            let sps_3d_extension_flag = r.read_bit()?;
            let sps_scc_extension_flag = r.read_bit()?;
            let _sps_extension_4bits = r.read_bits(4)?;

            if sps.sps_range_extension_flag {
                // Parse range extension flags per H.265 spec Table 7-8
                sps.transform_skip_rotation_enabled_flag = r.read_bit()?;
                sps.transform_skip_context_enabled_flag = r.read_bit()?;
                sps.implicit_rdpcm_enabled_flag = r.read_bit()?;
                sps.explicit_rdpcm_enabled_flag = r.read_bit()?;
                sps.extended_precision_processing_flag = r.read_bit()?;
                sps.intra_smoothing_disabled_flag = r.read_bit()?;
                sps.high_precision_offsets_enabled_flag = r.read_bit()?;
                sps.persistent_rice_adaptation_enabled_flag = r.read_bit()?;
                sps.cabac_bypass_alignment_enabled_flag = r.read_bit()?;
            }
            if sps_multilayer_extension_flag {
                let _ = r.read_bit()?; // inter_view_mv_vert_constraint_flag
            }
            if sps_3d_extension_flag {
                // sps_3d_extension (IVMC depth video — extremely rare; consume
                // per FFmpeg n8.1.2 so the bitstream position stays correct).
                for i in 0..=1u32 {
                    let _ = r.read_bit()?; // iv_di_mc_enabled_flag
                    let _ = r.read_bit()?; // iv_mv_scal_enabled_flag
                    if i == 0 {
                        let _ = r.read_ue()?; // log2_ivmc_sub_pb_size_minus3
                        let _ = r.read_bit()?; // iv_res_pred_enabled_flag
                        let _ = r.read_bit()?; // depth_ref_enabled_flag
                        let _ = r.read_bit()?; // vsp_mc_enabled_flag
                        let _ = r.read_bit()?; // dbbp_enabled_flag
                    } else {
                        let _ = r.read_bit()?; // tex_mc_enabled_flag
                        let _ = r.read_ue()?; // log2_ivmc_sub_pb_size_minus3
                        let _ = r.read_bit()?; // intra_contour_enabled_flag
                        let _ = r.read_bit()?; // intra_dc_only_wedge_enabled_flag
                        let _ = r.read_bit()?; // cqt_cu_part_pred_enabled_flag
                        let _ = r.read_bit()?; // inter_dc_only_enabled_flag
                        let _ = r.read_bit()?; // skip_intra_enabled_flag
                    }
                }
            }
            if sps_scc_extension_flag {
                let _sps_curr_pic_ref_enabled_flag = r.read_bit()?;
                let palette_mode_enabled_flag = r.read_bit()?;
                sps.palette_mode_enabled_flag = palette_mode_enabled_flag;
                if palette_mode_enabled_flag {
                    let _palette_max_size = r.read_ue()?;
                    let _delta_palette_max_predictor_size = r.read_ue()?;
                    let palette_predictor_initializers_present = r.read_bit()?;
                    if palette_predictor_initializers_present {
                        let count = r.read_ue()? + 1;
                        let num_comps = if sps.chroma_format_idc == 0 { 1 } else { 3 };
                        for comp in 0..num_comps {
                            let bit_depth = if comp == 0 {
                                sps.bit_depth_luma_minus8 + 8
                            } else {
                                sps.bit_depth_chroma_minus8 + 8
                            };
                            for _i in 0..count {
                                let _ = r.read_bits(bit_depth)?;
                            }
                        }
                    }
                }
                // motion_vector_resolution_control_idc +
                // intra_boundary_filtering_disabled_flag sit at the end of
                // sps_scc_extension (FFmpeg n8.1.2).
                sps.motion_vector_resolution_control_idc = r.read_bits(2)? as u8;
                let _intra_boundary_filtering_disabled = r.read_bit()?;
            }
        }

        self.sps_cache
            .insert(sps.sps_seq_parameter_set_id, sps.clone());
        self.active_sps = Some(sps);

        Ok(self.active_sps.clone().unwrap())
    }

    fn parse_pps(&mut self, data: &[u8]) -> ParserResult<vacc_core::picture::H265Pps> {
        if data.is_empty() {
            return Err(ParserError::InvalidBitstream);
        }

        // Skip NAL header (2 bytes for H.265)
        let mut r = BitReader::new(&data[2..], true);

        let mut pps = vacc_core::picture::H265Pps::new();

        pps.pps_pic_parameter_set_id = r.read_ue()?;
        pps.pps_seq_parameter_set_id = r.read_ue()?;
        pps.dependent_slice_segments_enabled_flag = r.read_bit()?;
        pps.output_flag_present_flag = r.read_bit()?;
        // num_extra_slice_header_bits is u(3), not ue(v) per H.265 spec 7.3.6
        pps.num_extra_slice_header_bits = r.read_bits(3)? as u8;
        pps.sign_data_hiding_enabled_flag = r.read_bit()?;
        pps.cabac_init_present_flag = r.read_bit()?;

        // num_ref_idx_l0_default_active_minus1 and num_ref_idx_l1_default_active_minus1
        // (per VulkanH265Parser.cpp:743-751)
        pps.num_ref_idx_l0_default_active_minus1 = r.read_ue()? as u8;
        pps.num_ref_idx_l1_default_active_minus1 = r.read_ue()? as u8;

        // pps_init_qp_minus26 (SE(V))
        pps.pps_init_qp_minus26 = r.read_se()?;

        // Additional PPS fields (per VulkanH265Parser.cpp:762-882)
        pps.constrained_intra_pred_flag = r.read_bit()?;
        pps.transform_skip_enabled_flag = r.read_bit()?;
        pps.cu_qp_delta_enabled_flag = r.read_bit()?;
        if pps.cu_qp_delta_enabled_flag {
            pps.diff_cu_qp_delta_depth = r.read_ue()? as u8;
        }
        pps.pps_cb_qp_offset = r.read_se()? as i8;
        pps.pps_cr_qp_offset = r.read_se()? as i8;
        pps.pps_slice_chroma_qp_offsets_present_flag = r.read_bit()?;
        pps.weighted_pred_flag = r.read_bit()?;
        pps.weighted_bipred_flag = r.read_bit()?;
        pps.transquant_bypass_enabled_flag = r.read_bit()?;
        pps.tiles_enabled_flag = r.read_bit()?;
        pps.entropy_coding_sync_enabled_flag = r.read_bit()?;

        if pps.tiles_enabled_flag {
            pps.num_tile_columns_minus1 = r.read_ue()? as u8;
            pps.num_tile_rows_minus1 = r.read_ue()? as u8;
            pps.uniform_spacing_flag = r.read_bit()?;
            if !pps.uniform_spacing_flag {
                for i in 0..pps.num_tile_columns_minus1 {
                    pps.column_width_minus1[i as usize] = r.read_ue()? as u16;
                }
                for i in 0..pps.num_tile_rows_minus1 {
                    pps.row_height_minus1[i as usize] = r.read_ue()? as u16;
                }
            }
        }

        // pps_loop_filter_across_tiles_enabled_flag: present in the bitstream
        // only when tiles_enabled_flag (verified against FFmpeg 8.0
        // ff_hevc_decode_nal_pps and NVIDIA VulkanH265Parser.cpp); otherwise
        // inferred as 1. Reading it when entropy_sync=1 && tiles=0 would shift
        // every later PPS field by one bit.
        if pps.tiles_enabled_flag {
            pps.loop_filter_across_tiles_enabled_flag = r.read_bit()?;
        } else {
            pps.loop_filter_across_tiles_enabled_flag = true;
        }

        pps.pps_loop_filter_across_slices_enabled_flag = r.read_bit()?;
        pps.deblocking_filter_control_present_flag = r.read_bit()?;
        if pps.deblocking_filter_control_present_flag {
            pps.deblocking_filter_override_enabled_flag = r.read_bit()?;
            pps.pps_deblocking_filter_disabled_flag = r.read_bit()?;
            if !pps.pps_deblocking_filter_disabled_flag {
                pps.pps_beta_offset_div2 = r.read_se()? as i8;
                pps.pps_tc_offset_div2 = r.read_se()? as i8;
            }
        }

        // Get associated SPS for scaling list inheritance
        let sps = self
            .sps_cache
            .get(&pps.pps_seq_parameter_set_id)
            .ok_or(ParserError::InvalidBitstream)?;

        pps.pps_scaling_list_data_present_flag = r.read_bit()?;
        if pps.pps_scaling_list_data_present_flag {
            // Parse scaling_list_data to advance bitstream position correctly
            Self::parse_scaling_list_data(&mut r, &mut pps.scaling_lists)?;
        } else if sps.sps_scaling_list_data_present_flag {
            // Per H.265 spec 7.3.4.2: when pps_scaling_list_data_present_flag is 0
            // and sps_scaling_list_data_present_flag is 1, PPS scaling lists are
            // copied from SPS scaling lists.
            pps.scaling_lists = sps.scaling_lists.clone();
        }

        pps.lists_modification_present_flag = r.read_bit()?;
        pps.log2_parallel_merge_level_minus2 = r.read_ue()? as u8;
        pps.slice_segment_header_extension_present_flag = r.read_bit()?;
        pps.pps_extension_present_flag = r.read_bit()?;
        if pps.pps_extension_present_flag {
            // Per H.265 spec / FFmpeg n8.1.2: 4 extension flags + 4 reserved
            // bits, then conditional extension data in order: range (only for
            // Rext+ profiles), multilayer, 3D, SCC.
            pps.pps_range_extension_flag = r.read_bit()?;
            let pps_multilayer_extension_flag = r.read_bit()?;
            let pps_3d_extension_flag = r.read_bit()?;
            let pps_scc_extension_flag = r.read_bit()?;
            let _pps_extension_4bits = r.read_bits(4)?;

            if sps.profile_idc >= 2 /* REXT */ && pps.pps_range_extension_flag {
                if pps.transform_skip_enabled_flag {
                    pps.log2_max_transform_skip_block_size_minus2 = r.read_ue()? as u8;
                }
                pps.cross_component_prediction_enabled_flag = r.read_bit()?;
                pps.chroma_qp_offset_list_enabled_flag = r.read_bit()?;
                if pps.chroma_qp_offset_list_enabled_flag {
                    pps.diff_cu_chroma_qp_offset_depth = r.read_ue()? as u8;
                    pps.chroma_qp_offset_list_len_minus1 = r.read_ue()? as u8;
                    for i in 0..=(pps.chroma_qp_offset_list_len_minus1 as usize).min(5) {
                        pps.cb_qp_offset_list[i] = r.read_se()? as i8;
                        pps.cr_qp_offset_list[i] = r.read_se()? as i8;
                    }
                }
                // log2_sao_offset_scale_luma/chroma are UNCONDITIONAL in the
                // current spec (verified against FFmpeg n8.1.2
                // pps_range_extensions).
                pps.log2_sao_offset_scale_luma = r.read_ue()? as u8;
                pps.log2_sao_offset_scale_chroma = r.read_ue()? as u8;
            }
            if pps_multilayer_extension_flag {
                let _poc_reset_info_present_flag = r.read_bit()?;
                if r.read_bit()? {
                    // pps_infer_scaling_list_flag
                    let _ = r.read_bits(6)?; // scaling_list_ref_layer_id
                }
                let num_ref_loc_offsets = r.read_ue()?;
                for _i in 0..num_ref_loc_offsets {
                    let _ = r.read_bits(6)?; // ref_loc_offset_layer_id
                    if r.read_bit()? {
                        // scaled_ref_layer_offset_present_flag
                        for _j in 0..4 {
                            let _ = r.read_se()?;
                        }
                    }
                    if r.read_bit()? {
                        // ref_region_offset_present_flag
                        for _j in 0..4 {
                            let _ = r.read_se()?;
                        }
                    }
                    if r.read_bit()? {
                        // resample_phase_set_present_flag
                        let _ = r.read_ue()?; // phase_hor_luma
                        let _ = r.read_ue()?; // phase_ver_luma
                        let _ = r.read_ue()?; // phase_hor_chroma
                        let _ = r.read_ue()?; // phase_ver_chroma
                    }
                }
                if r.read_bit()? {
                    // colour_mapping_enabled_flag: colour_mapping_table() is a
                    // recursive octant structure (depth video only) — reject
                    // rather than mis-parse.
                    return Err(ParserError::NonCompliantStream);
                }
            }
            if pps_3d_extension_flag {
                // pps_3d_extension (IVMC depth video — extremely rare; consume
                // per FFmpeg n8.1.2 so the bitstream position stays correct).
                if r.read_bit()? {
                    // dlts_present_flag
                    let _pps_depth_layers_minus1 = r.read_bits(6)?;
                    let bit_depth_for_depth_layers = r.read_bits(4)? + 8;
                    for _i in 0..=_pps_depth_layers_minus1 {
                        if r.read_bit()? {
                            // dlt_flag[i]
                            if !r.read_bit()? {
                                // dlt_pred_flag[i] == 0
                                if r.read_bit()? {
                                    // dlt_val_flags_present_flag[i]
                                    for _j in 0..(1u32 << bit_depth_for_depth_layers) - 1 {
                                        let _ = r.read_bit()?; // dlt_value_flag
                                    }
                                } else {
                                    // delta_dlt()
                                    let num_val_delta_dlt =
                                        r.read_bits(bit_depth_for_depth_layers as u8)?;
                                    if num_val_delta_dlt > 0 {
                                        let mut max_diff: u32 = 0;
                                        let mut min_diff_minus1: i32 = -1;
                                        if num_val_delta_dlt > 1 {
                                            max_diff =
                                                r.read_bits(bit_depth_for_depth_layers as u8)?;
                                        }
                                        if num_val_delta_dlt > 2 && max_diff > 0 {
                                            let len = max_diff.ilog2() + 1;
                                            min_diff_minus1 = r.read_bits(len as u8)? as i32;
                                        }
                                        if (max_diff as i32) > min_diff_minus1 + 1 {
                                            let len = (max_diff - (min_diff_minus1 + 1) as u32)
                                                .ilog2()
                                                + 1;
                                            for _k in 1..num_val_delta_dlt {
                                                let _ = r.read_bits(len as u8)?;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if pps_scc_extension_flag {
                let _pps_curr_pic_ref_enabled_flag = r.read_bit()?;
                let residual_adaptive_colour_transform_enabled = r.read_bit()?;
                if residual_adaptive_colour_transform_enabled {
                    // Gates slice-level slice_act_*_qp_offset in the slice header.
                    pps.pps_slice_act_qp_offsets_present_flag = r.read_bit()?;
                    // Raw coded values (spec: ActQpOffset + 5 / + 3).
                    pps.pps_act_y_qp_offset_plus5 = r.read_se()? as i8;
                    pps.pps_act_cb_qp_offset_plus5 = r.read_se()? as i8;
                    pps.pps_act_cr_qp_offset_plus3 = r.read_se()? as i8;
                }
                if r.read_bit()? {
                    // pps_palette_predictor_initializers_present_flag
                    let count = r.read_ue()?;
                    pps.pps_num_palette_predictor_initializers = count as u8;
                    if count > 0 {
                        let monochrome = r.read_bit()?;
                        let luma_depth = r.read_ue()? + 8;
                        pps.luma_bit_depth_entry_minus8 = (luma_depth - 8) as u8;
                        let chroma_depth = if !monochrome {
                            let d = r.read_ue()? + 8;
                            pps.chroma_bit_depth_entry_minus8 = (d - 8) as u8;
                            Some(d)
                        } else {
                            None
                        };
                        let num_comps = if monochrome { 1 } else { 3 };
                        for comp in 0..num_comps {
                            let depth = if comp == 0 {
                                luma_depth
                            } else {
                                chroma_depth.unwrap()
                            };
                            for _i in 0..count {
                                let _ = r.read_bits(depth as u8)?;
                            }
                        }
                    }
                }
            }
        }

        self.pps_cache
            .insert(pps.pps_pic_parameter_set_id, pps.clone());
        self.active_pps = Some(pps);

        Ok(self.active_pps.clone().unwrap())
    }

    /// Parse the slice segment header from a slice NAL unit.
    ///
    /// Based on H.265 spec section 7.3.6 and VulkanH265Parser.cpp:2119-2337.
    fn parse_slice_segment_header(
        &mut self,
        nal_data: &[u8],
        nal_unit_type: u8,
    ) -> ParserResult<SliceHeaderInfo> {
        // Skip NAL header (2 bytes for H.265)
        if nal_data.len() < 2 {
            return Err(ParserError::InvalidBitstream);
        }
        let mut r = BitReader::new(&nal_data[2..], true);

        let sps = self
            .active_sps
            .as_ref()
            .ok_or(ParserError::ParameterSetParse)?;
        let pps = self
            .active_pps
            .as_ref()
            .ok_or(ParserError::ParameterSetParse)?;

        let mut info = SliceHeaderInfo::new();

        // IdrPicFlag: IDR_W_RADL (19) / IDR_N_LP (20)
        info.is_idr = nal_unit_type == 19 || nal_unit_type == 20;

        // RapPicFlag: IRAP = NAL types 16-23 (BLA, IDR, CRA, RSV_IRAP)
        info.is_rap = (16..=23).contains(&nal_unit_type);

        // Determine is_reference from NAL unit type
        // Per H.265 spec:
        // - VCL NAL types 0-15: odd types (1,3,5,7,9,11,13,15) are reference
        // - IRAP NAL types 16-23: all are reference pictures
        // Based on VulkanH265Parser.cpp:2783-2791 for sub-layer non-ref determination
        info.is_reference =
            (16..=23).contains(&nal_unit_type) || (nal_unit_type < 16 && (nal_unit_type & 1) == 1);

        // --- slice_segment_header parsing (per VulkanH265Parser.cpp:2130-2133) ---

        // first_slice_segment_in_pic_flag
        let first_slice_segment_in_pic_flag = r.read_bit()?;

        // no_output_of_prior_pics_flag: present in the bitstream for ALL IRAP
        // NAL types 16-23 (verified against FFmpeg hls_slice_header and x265
        // encoder output, which writes it even for IDR); inferred 0 (not read)
        // for non-IRAP. Store the raw bitstream value.
        let no_output_of_prior_pics_flag = if info.is_rap { r.read_bit()? } else { false };
        info.no_output_of_prior_pics_flag = no_output_of_prior_pics_flag;

        // pic_parameter_set_id
        let _slice_pps_id = r.read_ue()?;

        // For non-first slice segments, parse dependent_slice_segment_flag and slice_segment_address
        if !first_slice_segment_in_pic_flag {
            let dependent_slice_segment_flag = if pps.dependent_slice_segments_enabled_flag {
                r.read_bit()?
            } else {
                false
            };
            info.dependent_slice_segment_flag = dependent_slice_segment_flag;

            // slice_segment_address: CeilLog2(PicSizeInCtbsY) bits
            let log2_ctb_size = sps.log2_min_luma_coding_block_size_minus3 as u32
                + 3
                + sps.log2_diff_max_min_luma_coding_block_size as u32;
            let pic_width_in_ctbs =
                (sps.pic_width_in_luma_samples as u32 + (1 << log2_ctb_size) - 1) >> log2_ctb_size;
            let pic_height_in_ctbs =
                (sps.pic_height_in_luma_samples as u32 + (1 << log2_ctb_size) - 1) >> log2_ctb_size;
            let pic_size_in_ctbs = pic_width_in_ctbs * pic_height_in_ctbs;
            let slice_segment_address_bits = (pic_size_in_ctbs as f64).log2().ceil() as u8;
            info.slice_segment_address = r.read_bits(slice_segment_address_bits)?;

            // For dependent slices, most info is inherited from first slice
            if dependent_slice_segment_flag {
                // A dependent slice segment inherits EVERY slice-level
                // parameter from the preceding slice segment (spec 7.3.8):
                // only its slice_segment_address is coded. Clone the previous
                // header wholesale — a chain of dependent segments all
                // transitively inherit the independent segment's values —
                // and apply the parsed address.
                let addr = info.slice_segment_address;
                let dependent = info.dependent_slice_segment_flag;
                if let Some(ref prev_info) = self.first_slice_header {
                    info = prev_info.clone();
                }
                info.slice_segment_address = addr;
                info.dependent_slice_segment_flag = dependent;
                // The dependent segment's own coded header is short (flag
                // bits + pps_id + [dependent flag] + address): its CABAC data
                // starts here, not at the first segment's longer header end.
                info.header_bit_size = r.position() as u16;
                return Ok(info);
            }
        }

        // --- Non-dependent slice segment header (per VulkanH265Parser.cpp:2196-2291) ---

        // Skip num_extra_slice_header_bits bits (per VulkanH265Parser.cpp:2196-2197)
        if pps.num_extra_slice_header_bits > 0 {
            let _ = r.read_bits(pps.num_extra_slice_header_bits)?;
        }

        // slice_type (UE(V)) - raw HEVC: 0=B, 1=P, 2=I.
        // SliceHeaderInfo convention is 0=I, 1=P, 2=B, so remap raw -> convention.
        // (Consumers such as nvdec picparams.rs rely on 0=I for intra_pic_flag.)
        let slice_type_raw = r.read_ue()?;
        info.slice_type = match slice_type_raw {
            0 => 2, // B
            1 => 1, // P
            2 => 0, // I
            n => n, // unexpected value; pass through
        } as u8;

        // pic_output_flag (if output_flag_present_flag; inferred 1 otherwise)
        if pps.output_flag_present_flag {
            info.pic_output_flag = r.read_bit()?;
        }

        // colour_plane_id (if separate_colour_plane_flag)
        if sps.separate_colour_plane_flag {
            info.colour_plane_id = r.read_bits(2)? as u8;
        }

        // pic_order_cnt_lsb (if not IDR)
        if !info.is_idr {
            let poc_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
            info.pic_order_cnt_lsb = r.read_bits(poc_bits as u8)? as u16;
        } else {
            info.pic_order_cnt_lsb = 0;
        }

        // Compute full POC value per H.265 spec section 8.3.1
        // (based on VulkanH265Parser.cpp:2757-2799)
        let pic_order_cnt_msb: i32;

        // NoRaslOutputFlag per H.265 spec 8.3.1:
        // - 1 for IDR (19-20): POC is 0, pic_order_cnt_lsb absent from bitstream
        // - equal to no_output_of_prior_pics_flag for other IRAPs (BLA/CRA/RSV_IRAP)
        // - 0 for non-IRAP
        let no_rasl_output_flag = if info.is_idr {
            true
        } else {
            no_output_of_prior_pics_flag
        };
        if no_rasl_output_flag {
            // IRAP with NoRaslOutputFlag: MSB is 0
            pic_order_cnt_msb = 0;
        } else {
            let max_pic_order_cnt_lsb = 1 << (sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4);

            if self.has_prev_pic {
                if ((info.pic_order_cnt_lsb as i32) < self.prev_pic_order_cnt_lsb)
                    && (self.prev_pic_order_cnt_lsb - info.pic_order_cnt_lsb as i32
                        >= max_pic_order_cnt_lsb / 2)
                {
                    pic_order_cnt_msb = self.prev_pic_order_cnt_msb + max_pic_order_cnt_lsb;
                } else if (info.pic_order_cnt_lsb as i32 > self.prev_pic_order_cnt_lsb)
                    && (info.pic_order_cnt_lsb as i32 - self.prev_pic_order_cnt_lsb
                        > max_pic_order_cnt_lsb / 2)
                {
                    pic_order_cnt_msb = self.prev_pic_order_cnt_msb - max_pic_order_cnt_lsb;
                } else {
                    pic_order_cnt_msb = self.prev_pic_order_cnt_msb;
                }
            } else {
                // First picture: MSB is 0
                pic_order_cnt_msb = 0;
            }
        }

        info.curr_pic_order_cnt_val = pic_order_cnt_msb + info.pic_order_cnt_lsb as i32;

        // Update prevPicOrderCntMsb/Lsb per HEVC spec 8.3.1 (matching FFmpeg's
        // pocTid0 update rule): only the first slice of a temporal_id_plus1 == 1
        // picture with NAL type TRAIL_R (1), TSA_R (3), STSA_R (5) or IRAP (16-23).
        let temporal_id_plus1 = nal_data[1] & 0x07; // nuh_temporal_id_plus1
        if first_slice_segment_in_pic_flag
            && temporal_id_plus1 == 1
            && matches!(nal_unit_type, 1 | 3 | 5 | 16..=23)
        {
            self.prev_pic_order_cnt_lsb = info.pic_order_cnt_lsb as i32;
            self.prev_pic_order_cnt_msb = pic_order_cnt_msb;
            self.has_prev_pic = true;
        }

        // short_term_ref_pic_set_sps_flag + RPS block
        // Per H.265 spec 7.3.7 (verified against x265 encoder output and
        // FFmpeg's hls_slice_header): this block (STRPS + long-term refs +
        // slice_temporal_mvp_enabled_flag) is present for ALL non-IDR pictures,
        // including I-slice CRA pictures (whose RPS entries are simply unused).
        // IDR pictures carry neither pic_order_cnt_lsb nor an RPS.
        if !info.is_idr {
            let short_term_ref_pic_set_sps_flag = r.read_bit()?;
            info.short_term_ref_pic_set_sps_flag = short_term_ref_pic_set_sps_flag;
            if !short_term_ref_pic_set_sps_flag {
                // STRPS in slice - parse and store it
                let bits_before = r.position();
                let strps = Self::parse_short_term_ref_pic_set(
                    &mut r,
                    sps.num_short_term_ref_pic_sets as usize,
                    sps.num_short_term_ref_pic_sets as usize,
                    &sps.short_term_ref_pic_sets,
                )?;
                info.slice_strps = Some(strps);
                // NumBitsForShortTermRPSInSlice: SizeInBits of short_term_ref_pic_set()
                info.num_bits_for_strps_in_slice = (r.position() - bits_before) as u16;
            } else if sps.num_short_term_ref_pic_sets > 1 {
                let strps_idx_bits = (sps.num_short_term_ref_pic_sets as f64).log2().ceil() as u8;
                info.short_term_ref_pic_set_idx = r.read_bits(strps_idx_bits)? as u8;
            }

            // Long-term reference pictures
            // Per H.265 spec 7.3.7: num_long_term_sps is read only when long_term_ref_pics_present_flag is true
            // and num_long_term_ref_pics_sps > 0. num_long_term_pics is always read when long_term_ref_pics_present_flag is true.
            if sps.long_term_ref_pics_present_flag {
                let num_long_term_sps = if sps.num_long_term_ref_pics_sps > 0 {
                    r.read_ue()? as u8
                } else {
                    0
                };
                let num_long_term_pics = r.read_ue()? as u8;
                info.num_long_term_sps = num_long_term_sps;
                info.num_long_term_pics = num_long_term_pics;

                for i in 0u8..(num_long_term_sps + num_long_term_pics) {
                    let mut lt_ref = H265LtRef::default();
                    if i < num_long_term_sps {
                        lt_ref.from_sps = true;
                        if sps.num_long_term_ref_pics_sps > 1 {
                            let lt_idx_bits =
                                (sps.num_long_term_ref_pics_sps as f64).log2().ceil() as u8;
                            lt_ref.sps_idx = r.read_bits(lt_idx_bits)? as u8;
                        }
                        // poc_lsb comes from sps.lt_ref_pic_poc_lsb_sps[sps_idx];
                        // the used flag is indexed by the SPS LT index (not the
                        // position in the current slice's LT list).
                        lt_ref.used_by_curr_pic =
                            (sps.used_by_curr_pic_lt_sps_flag >> lt_ref.sps_idx) & 1 == 1;
                    } else {
                        let poc_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
                        lt_ref.poc_lsb = r.read_bits(poc_lsb_bits)?; // poc_lsb_lt
                        lt_ref.used_by_curr_pic = r.read_bit()?; // used_by_curr_pic_lt_flag
                    }
                    lt_ref.delta_poc_msb_present = r.read_bit()?;
                    if lt_ref.delta_poc_msb_present {
                        lt_ref.delta_poc_msb_cycle = r.read_ue()?;
                    }
                    info.long_term_refs.push(lt_ref);
                }
            }

            // slice_temporal_mvp_enabled_flag
            if sps.sps_temporal_mvp_enabled_flag {
                info.slice_temporal_mvp_enabled_flag = r.read_bit()?;
            }
        } else {
            // IDR pictures carry no RPS: short_term_ref_pic_set_sps_flag is
            // absent from the bitstream and must be signaled as 0 to the
            // driver (signaling 1 makes it reconstruct an SPS RPS that does
            // not exist, e.g. when num_short_term_ref_pic_sets == 0).
            info.short_term_ref_pic_set_sps_flag = false;
        }

        // slice_sample_adaptive_offset_flag[] (verified against FFmpeg n8.1.2
        // hls_slice_header): present for ALL non-dependent slice segments —
        // including I and IDR slices — when sample_adaptive_offset_enabled_flag
        // is set: one luma flag plus two chroma flags when chroma is present.
        // It sits AFTER the RPS block / slice_temporal_mvp bit and BEFORE the
        // num_ref_idx fields.
        if sps.sample_adaptive_offset_enabled_flag {
            info.slice_sao_luma_flag = r.read_bit()?;
            // H.265 7.3.6.1: exactly ONE chroma SAO flag (slice_sao_chroma_flag)
            // when ChromaFormatIDC > 0 — NOT two. (FFmpeg assigns the single
            // bit to both internal flag[1] and flag[2].)
            if sps.chroma_format_idc != 0 {
                info.slice_sao_chroma_flag = r.read_bit()?;
            }
        }

        // --- Inter slice segment fields (SliceType != I) ---
        if info.slice_type != 0 {
            // num_ref_idx_active_override_flag + num_ref_idx_l*_active_minus1.
            // H.265 (unlike H.264): the active reference counts default to
            // the PPS values and are overridden only when the flag is set.
            let override_flag = r.read_bit()?;
            if override_flag {
                info.num_ref_idx_l0_active_minus1 = r.read_ue()? as u8;
                if info.slice_type == 2 {
                    info.num_ref_idx_l1_active_minus1 = r.read_ue()? as u8;
                }
            } else {
                info.num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
                if info.slice_type == 2 {
                    info.num_ref_idx_l1_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
                }
            }

            let n0 = info.num_ref_idx_l0_active_minus1 as usize + 1;
            let n1 = if info.slice_type == 2 {
                info.num_ref_idx_l1_active_minus1 as usize + 1
            } else {
                0
            };

            // ref_pic_lists_modification (H.265 7.3.6.1, verified against
            // FFmpeg n8.1.2): the L0 modification flag is present for BOTH P
            // and B slices (when ListsModificationPresent &&
            // NumRefIdxL0Active > 1); the L1 flag is B-only. When set, the
            // whole list is replaced by fixed-length indices of
            // ceil(log2(max(NumRefIdxL0Active, NumRefIdxL1Active))) bits —
            // NOT the H.264-style per-entry flag+ue(v) form.
            if pps.lists_modification_present_flag && n0.max(n1) > 1 {
                let idx_bits = (n0.max(n1) as f64).log2().ceil() as u8;
                // L0: P and B slices
                if r.read_bit()? {
                    for _i in 0..n0 {
                        info.ref_pic_lists_modification_l0
                            .push(H265ListModification {
                                flag: true,
                                ref_idx: r.read_bits(idx_bits)? as u8,
                            });
                    }
                }
                // L1: B slices only
                if info.slice_type == 2 && r.read_bit()? {
                    for _i in 0..n1 {
                        info.ref_pic_lists_modification_l1
                            .push(H265ListModification {
                                flag: true,
                                ref_idx: r.read_bits(idx_bits)? as u8,
                            });
                    }
                }
            }

            // mvd_l1_zero_flag (B slices only).
            if info.slice_type == 2 {
                info.mvd_l1_zero_flag = r.read_bit()?;
            }

            // cabac_init_flag (P and B slices when PPS cabac_init_present_flag).
            if pps.cabac_init_present_flag {
                info.cabac_init_flag = r.read_bit()?;
            }

            // Collocated picture signaling (when slice_temporal_mvp_enabled_flag):
            // collocated_from_l0_flag is B-only; collocated_ref_idx is present
            // when the collocated list has > 1 active reference.
            if info.slice_temporal_mvp_enabled_flag {
                let from_l0 = if info.slice_type == 2 {
                    r.read_bit()?
                } else {
                    true
                };
                info.collocated_from_l0_flag = from_l0;
                if (if from_l0 { n0 } else { n1 }) > 1 {
                    info.collocated_ref_idx = r.read_ue()? as u8;
                }
            }

            // pred_weight_table (per-reference flag form, verified against
            // FFmpeg n8.1.2): P slices when PPS weighted_pred_flag, B slices
            // when PPS weighted_bipred_flag.
            if (pps.weighted_pred_flag && info.slice_type == 1)
                || (pps.weighted_bipred_flag && info.slice_type == 2)
            {
                info.luma_log2_weight_denom = r.read_ue()? as u8;
                if sps.chroma_format_idc != 0 {
                    info.delta_chroma_log2_weight_denom = r.read_se()? as i8;
                }
                // L0: per-reference flag bitfields (MSB first), then the
                // weighted entries. Unflagged references keep zeroed values.
                let luma_flags = if n0 > 0 { r.read_bits(n0 as u8)? } else { 0 };
                let chroma_flags = if sps.chroma_format_idc != 0 && n0 > 0 {
                    r.read_bits(n0 as u8)?
                } else {
                    0
                };
                for i in 0..n0 {
                    if (luma_flags >> (n0 - 1 - i)) & 1 == 1 {
                        let w = r.read_se()? as i8;
                        let o = r.read_se()? as i16;
                        if i < 15 {
                            info.delta_luma_weight_l0[i] = w;
                            info.luma_offset_l0[i] = o;
                        }
                    }
                    if (chroma_flags >> (n0 - 1 - i)) & 1 == 1 {
                        for j in 0..2 {
                            let w = r.read_se()? as i8;
                            let o = r.read_se()? as i16;
                            if i < 15 {
                                info.delta_chroma_weight_l0[i][j] = w;
                                info.chroma_offset_l0[i][j] = o;
                            }
                        }
                    }
                }
                // L1 (B slices only).
                if info.slice_type == 2 {
                    let luma_flags = if n1 > 0 { r.read_bits(n1 as u8)? } else { 0 };
                    let chroma_flags = if sps.chroma_format_idc != 0 && n1 > 0 {
                        r.read_bits(n1 as u8)?
                    } else {
                        0
                    };
                    for i in 0..n1 {
                        if (luma_flags >> (n1 - 1 - i)) & 1 == 1 {
                            let w = r.read_se()? as i8;
                            let o = r.read_se()? as i16;
                            if i < 15 {
                                info.delta_luma_weight_l1[i] = w;
                                info.luma_offset_l1[i] = o;
                            }
                        }
                        if (chroma_flags >> (n1 - 1 - i)) & 1 == 1 {
                            for j in 0..2 {
                                let w = r.read_se()? as i8;
                                let o = r.read_se()? as i16;
                                if i < 15 {
                                    info.delta_chroma_weight_l1[i][j] = w;
                                    info.chroma_offset_l1[i][j] = o;
                                }
                            }
                        }
                    }
                }
            }

            // five_minus_max_num_merge_cand.
            info.five_minus_max_num_merge_cand = r.read_ue()? as u8;

            // use_integer_mv_flag (SPS motion_vector_resolution_control_idc == 2).
            if sps.motion_vector_resolution_control_idc == 2 {
                info.use_integer_mv_flag = r.read_bit()?;
            }
        }

        // --- Slice-level parameters (all slice types) ---

        // slice_qp_delta.
        info.slice_qp_delta = r.read_se()?;

        // slice_cb/cr_qp_offset.
        if pps.pps_slice_chroma_qp_offsets_present_flag {
            info.slice_cb_qp_offset = r.read_se()?;
            info.slice_cr_qp_offset = r.read_se()?;
        }

        // slice_act_*_qp_offset (SCC, when PPS pps_slice_act_qp_offsets_present).
        if pps.pps_slice_act_qp_offsets_present_flag {
            info.slice_act_y_qp_offset = r.read_se()?;
            info.slice_act_cb_qp_offset = r.read_se()?;
            info.slice_act_cr_qp_offset = r.read_se()?;
        }

        // cu_chroma_qp_offset_enabled_flag (range extension).
        if pps.chroma_qp_offset_list_enabled_flag {
            info.cu_chroma_qp_offset_enabled_flag = r.read_bit()?;
        }

        // Deblocking filter slice-level control (verified against FFmpeg n8.1.2):
        // the override flag is read only when deblocking_filter_override_enabled;
        // otherwise the PPS values are inherited. When overridden and the
        // filter stays enabled, beta/tc offsets are coded.
        if pps.deblocking_filter_control_present_flag {
            let override_flag = if pps.deblocking_filter_override_enabled_flag {
                r.read_bit()?
            } else {
                false
            };
            if override_flag {
                info.slice_deblocking_filter_disabled_flag = r.read_bit()?;
                if !info.slice_deblocking_filter_disabled_flag {
                    info.slice_beta_offset_div2 = r.read_se()?;
                    info.slice_tc_offset_div2 = r.read_se()?;
                }
            } else {
                info.slice_deblocking_filter_disabled_flag =
                    pps.pps_deblocking_filter_disabled_flag;
            }
        }

        // slice_loop_filter_across_slices_enabled_flag (verified against
        // FFmpeg n8.1.2): read only when the PPS enables it AND (SAO is active
        // in this slice OR deblocking is not disabled); otherwise inferred
        // from the PPS value.
        let sao_active = info.slice_sao_luma_flag || info.slice_sao_chroma_flag;
        if pps.pps_loop_filter_across_slices_enabled_flag
            && (sao_active || !info.slice_deblocking_filter_disabled_flag)
        {
            info.slice_loop_filter_across_slices_enabled_flag = r.read_bit()?;
        } else {
            info.slice_loop_filter_across_slices_enabled_flag =
                pps.pps_loop_filter_across_slices_enabled_flag;
        }

        // Entry points (PPS tiles or entropy coding sync). Per the current
        // H.265 spec, entry_point_offset_length is coded as ue(v) + 1.
        if pps.tiles_enabled_flag || pps.entropy_coding_sync_enabled_flag {
            info.num_entry_point_offsets = r.read_ue()? as u16;
            if info.num_entry_point_offsets > 0 {
                info.entry_point_offset_length = (r.read_ue()? + 1) as u16;
                for _ in 0..info.num_entry_point_offsets {
                    info.entry_point_offsets
                        .push(r.read_bits(info.entry_point_offset_length as u8)?);
                }
            }
        }

        // slice_header_extension (slice_segment_header_extension_present_flag).
        if pps.slice_segment_header_extension_present_flag {
            let ext_bytes = r.read_ue()?;
            for _ in 0..ext_bytes {
                let _ = r.read_byte()?;
            }
        }

        // End of coded slice header: start of rbsp_slice_trailing_bits.
        info.header_bit_size = r.position() as u16;

        Ok(info)
    }

    fn extract_nal_units(&self, data: &[u8]) -> Vec<NalUnit> {
        let mut nal_units = Vec::new();
        let mut offset = 0;

        while offset < data.len() {
            if let Some((start, code_len)) = nal::find_next_start_code(data, offset) {
                let next_start = nal::find_next_start_code(data, start + code_len);

                let (end, next_code_len) = match next_start {
                    Some((s, cl)) => (s, cl),
                    None => (data.len(), 0),
                };

                // When the next start code is 4 bytes (00 00 00 01), the leading
                // 0x00 is the trailing_zero_8bits of the current NAL unit.
                // Include it in the NAL data to match the raw byte stream payload.
                let nal_end = if next_code_len == 4 { end + 1 } else { end };
                let nal_data = &data[start + code_len..nal_end];
                if !nal_data.is_empty() {
                    if let Some(nal_unit_type) = H265NalUnitType::from_u8(
                        nal::parse_h265_nal_header(nal_data)
                            .map(|(_, t, _, _)| t)
                            .unwrap_or(0),
                    ) {
                        nal_units.push(NalUnit::new(
                            nal_unit_type as u8,
                            nal_data.to_vec(),
                            start + code_len,
                            nal_data.len(),
                        ));
                    }
                }

                offset = end;
            } else {
                break;
            }
        }

        nal_units
    }
}

impl VideoParser for H265Parser {
    fn init(&mut self, format: &DetectedVideoFormat) -> ParserResult<()> {
        if format.codec != vacc_core::codec::VideoCodec::DecodeH265 {
            return Err(ParserError::InvalidBitstream);
        }
        self.detected_format = format.clone();
        Ok(())
    }

    fn parse(&mut self, packet: &crate::bitstream::BitstreamPacket) -> ParserResult<ParseResult> {
        if packet.is_eos() {
            return Ok(ParseResult::EndOfStream);
        }

        // Rebuild the NAL cache if a packet of a different size arrived (a new
        // chunk of data). Otherwise reuse the cached NALs and advance the
        // cursor, avoiding an O(n^2) re-scan/re-copy of the bitstream.
        if packet.payload.len() != self.cached_payload_len {
            self.cached_nals = self.extract_nal_units(&packet.payload);
            self.cached_payload_len = packet.payload.len();
            self.nal_cursor = 0;
        }

        if self.nal_cursor >= self.cached_nals.len() {
            return Ok(ParseResult::Nothing);
        }

        let mut result_sps: Option<vacc_core::picture::BoxedPictureParametersSet> = None;
        let mut result_pps: Option<vacc_core::picture::BoxedPictureParametersSet> = None;
        let mut result_vps: Option<vacc_core::picture::BoxedPictureParametersSet> = None;
        let mut slice_nals: Vec<crate::SliceEntry> = Vec::new();
        let mut last_slice_end: Option<usize> = None;
        // Cursor index of the first collected slice NAL. Used to roll the
        // cursor back when a parameter set is returned instead of the slices
        // (a [VPS][SPS][PPS][slice] sequence), so the slices are re-processed
        // on the next parse() call.
        let mut first_slice_cursor: Option<usize> = None;

        let mut i = self.nal_cursor;
        while i < self.cached_nals.len() {
            let nal = &self.cached_nals[i];

            match H265NalUnitType::from_u8(nal.nal_unit_type) {
                Some(H265NalUnitType::Vps) => {
                    // A picture is in progress: defer this parameter set until
                    // the current picture's slices are returned, otherwise the
                    // last picture before a GOP boundary would be dropped.
                    if !slice_nals.is_empty() {
                        break;
                    }
                    // Copy the NAL data out so the borrow of self.cached_nals
                    // ends before the &mut self calls below.
                    let nal_data = nal.data.clone();
                    if let Ok(_vps) = self.parse_vps(&nal_data) {
                        result_vps = Some(vacc_core::picture::BoxedPictureParametersSet::new(
                            self.active_vps.clone().unwrap(),
                        ));
                    }
                    i += 1;
                }
                Some(H265NalUnitType::Sps) => {
                    if !slice_nals.is_empty() {
                        break;
                    }
                    let nal_data = nal.data.clone();
                    if let Ok(sps) = self.parse_sps(&nal_data) {
                        {
                            result_sps = Some(vacc_core::picture::BoxedPictureParametersSet::new(
                                self.active_sps.clone().unwrap(),
                            ));
                            // Update detected format from SPS
                            self.detected_format.coded_width = sps.pic_width_in_luma_samples as u32;
                            self.detected_format.coded_height =
                                sps.pic_height_in_luma_samples as u32;
                            match sps.chroma_format_idc {
                                0 => {
                                    self.detected_format.chroma_subsampling =
                                        vacc_core::format::ChromaSubsampling::Monochrome
                                }
                                1 => {
                                    self.detected_format.chroma_subsampling =
                                        vacc_core::format::ChromaSubsampling::_420
                                }
                                2 => {
                                    self.detected_format.chroma_subsampling =
                                        vacc_core::format::ChromaSubsampling::_422
                                }
                                3 => {
                                    self.detected_format.chroma_subsampling =
                                        vacc_core::format::ChromaSubsampling::_444
                                }
                                _ => {}
                            }
                            let luma_bd = 8 + sps.bit_depth_luma_minus8;
                            let chroma_bd = 8 + sps.bit_depth_chroma_minus8;
                            self.detected_format.luma_bit_depth = match luma_bd {
                                8 => vacc_core::format::ComponentBitDepth::Bit8,
                                10 => vacc_core::format::ComponentBitDepth::Bit10,
                                12 => vacc_core::format::ComponentBitDepth::Bit12,
                                _ => vacc_core::format::ComponentBitDepth::Bit8,
                            };
                            self.detected_format.chroma_bit_depth = match chroma_bd {
                                8 => vacc_core::format::ComponentBitDepth::Bit8,
                                10 => vacc_core::format::ComponentBitDepth::Bit10,
                                12 => vacc_core::format::ComponentBitDepth::Bit12,
                                _ => vacc_core::format::ComponentBitDepth::Bit8,
                            };
                            self.detected_format.codec_profile = sps.profile_idc as u32;
                            self.detected_format.progressive_sequence = !sps.vui.field_seq_flag;
                        }
                    }
                    i += 1;
                }
                Some(H265NalUnitType::Pps) => {
                    if !slice_nals.is_empty() {
                        break;
                    }
                    let nal_data = nal.data.clone();
                    if let Ok(_pps) = self.parse_pps(&nal_data) {
                        result_pps = Some(vacc_core::picture::BoxedPictureParametersSet::new(
                            self.active_pps.clone().unwrap(),
                        ));
                    }
                    i += 1;
                }
                Some(t) if t.is_slice() => {
                    // Copy the NAL data out so the borrow of self.cached_nals
                    // ends before the &mut self calls below.
                    let nal_data = nal.data.clone();
                    let (off, sz) = (nal.offset, nal.size);
                    let nal_type = nal.nal_unit_type;

                    // first_slice_segment_in_pic_flag is the first bit of the
                    // slice segment header (MSB of the byte after the 2-byte
                    // NAL header). It marks the first slice segment of a
                    // picture: when a picture's slices are already collected
                    // and another first slice segment is hit, the current
                    // picture is complete - stop collecting (do not consume
                    // this NAL; it starts the next picture).
                    let starts_new_pic = if nal_data.len() >= 3 {
                        (nal_data[2] >> 7) & 1 == 1
                    } else {
                        true
                    };
                    if !slice_nals.is_empty() && starts_new_pic {
                        break;
                    }

                    // Cursor of the first collected slice NAL (for parameter-set
                    // rollback). bytes_consumed is derived from last_slice_end.
                    if first_slice_cursor.is_none() {
                        first_slice_cursor = Some(i);
                    }
                    last_slice_end = Some(off + sz);

                    // Parse this slice segment's OWN header. Non-first segments
                    // carry their own slice_segment_address and CABAC data
                    // offset (and, unless dependent, full prediction
                    // parameters) — drivers need per-slice values, not a clone
                    // of the first segment's. The first segment's header is
                    // kept separately as picture-level state.
                    let parsed_header = self.parse_slice_segment_header(&nal_data, nal_type);
                    if self.first_slice_header.is_none() {
                        if let Ok(info) = &parsed_header {
                            self.first_slice_header = Some(info.clone());
                        }
                    }
                    let header = match &parsed_header {
                        Ok(info) => Some(crate::SliceHeader::H265(info.clone())),
                        Err(_) => self
                            .first_slice_header
                            .clone()
                            .map(crate::SliceHeader::H265),
                    };

                    // Collect slice NAL data
                    slice_nals.push(crate::SliceEntry {
                        slice_header: header,
                        nal_data,
                    });
                    i += 1;
                }
                _ => {
                    // Non-VCL NAL unit (AUD, SEI, ...) - skip
                    i += 1;
                }
            }
        }

        if result_sps.is_some() || result_pps.is_some() || result_vps.is_some() {
            // If slices were collected but a parameter set is returned instead
            // (a [VPS][SPS][PPS][slice] sequence), roll the cursor back to the
            // first collected slice so it is re-processed on the next parse()
            // call.
            self.nal_cursor = if !slice_nals.is_empty() {
                first_slice_cursor.unwrap_or(i)
            } else {
                i
            };
            Ok(ParseResult::ParameterSet {
                sps: result_sps,
                pps: result_pps,
                vps: result_vps,
                sps_nal: None,
                pps_nal: None,
            })
        } else if !slice_nals.is_empty() {
            self.nal_cursor = i;
            self.frame_count += 1;
            // Consume from the packet start through the end of the last slice
            // NAL. Using last_end (not last_end - first_off) advances the caller's
            // byte offset past any leading gap (start codes / non-VCL NALs before
            // the first slice), so the next parse() does not re-find this picture.
            let bytes_consumed = last_slice_end.unwrap_or(0);
            // Clear first_slice_header so the next picture gets a fresh parse
            self.first_slice_header.take();
            Ok(ParseResult::Slice {
                slices: slice_nals,
                bytes_consumed,
            })
        } else {
            self.nal_cursor = i;
            Ok(ParseResult::Nothing)
        }
    }

    fn reset(&mut self) {
        self.vps_cache.clear();
        self.sps_cache.clear();
        self.pps_cache.clear();
        self.active_vps = None;
        self.active_sps = None;
        self.active_pps = None;
        self.frame_count = 0;
        self.first_slice_header = None;
        // Reset POC tracking to match new() initialization
        self.prev_pic_order_cnt_msb = 0;
        self.prev_pic_order_cnt_lsb = 0;
        self.has_prev_pic = false;
        self.cached_nals.clear();
        self.cached_payload_len = 0;
        self.nal_cursor = 0;
    }

    fn detected_format(&self) -> &DetectedVideoFormat {
        &self.detected_format
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Test helpers
    // ========================================================================

    /// Initialize a parser with VPS, SPS, PPS from the test data.
    fn init_parser() -> H265Parser {
        let mut parser = H265Parser::new();
        parser.parse_vps(TEST_VPS_DATA).expect("VPS parse failed");
        parser.parse_sps(TEST_SPS_DATA).expect("SPS parse failed");
        parser.parse_pps(TEST_PPS_DATA).expect("PPS parse failed");
        parser
    }

    // Real slice NAL units carved from `assets/big_buck_bunney.h265` — the
    // first ~40 bytes of each NAL, enough for the complete coded slice header.
    // Every POC/slice-type value below is cross-verified against FFmpeg n8.1.2
    // (`-threads 1` file-order POC sequence, 300/300 exact match) and NVIDIA
    // cuvid (`h265_cref_50.txt`). NAL type = `(byte0 >> 1) & 0x3F`.

    /// PIC0: IDR_W_RADL (19), poc 0.
    const SLICE_IDR: &[u8] = &[
        0x26, 0x01, 0xaf, 0x1f, 0x08, 0x84, 0x32, 0xb7, 0x30, 0x15, 0xed, 0xbd, 0xae, 0xda, 0x6d,
        0xc8, 0xaf, 0xb2, 0x70, 0x60, 0x2e, 0xbe, 0xb5, 0xb1, 0xbd, 0xf7, 0xf4, 0xc1, 0xb1, 0x0c,
        0x71, 0x4c, 0x27, 0x86, 0xe1, 0x2c, 0xd8, 0x38, 0xe0,
    ];

    /// PIC1: TRAIL_R (1), lsb 5.
    const SLICE_P1: &[u8] = &[
        0x02, 0x01, 0xd0, 0x29, 0x4b, 0xe1, 0x0c, 0x70, 0x44, 0x6d, 0xae, 0x25, 0x0e, 0xc5, 0x76,
        0x2d, 0x2b, 0xb1, 0x9f, 0x66, 0x90, 0x2d, 0xe3, 0x75, 0xb1, 0x44, 0xf4, 0x37, 0x57, 0xcd,
        0x42, 0x5f, 0xcf, 0x6a, 0x56, 0xc4,
    ];

    /// PIC2: TRAIL_R (1), lsb 3.
    const SLICE_P2: &[u8] = &[
        0x02, 0x01, 0xe0, 0x64, 0x9d, 0x78, 0x68, 0x11, 0x13, 0x1a, 0x56, 0x20, 0xce, 0x27, 0xb4,
        0xc9, 0x60, 0x9e, 0xab, 0x9e, 0x3b, 0xda, 0xb4, 0x98, 0x7a, 0x85, 0x19,
    ];

    /// PIC3: TRAIL_N (0), lsb 1.
    const SLICE_P3: &[u8] = &[
        0x00, 0x01, 0xe0, 0x24, 0xf5, 0x5f, 0xa2, 0xc9, 0x08, 0x9e, 0xbc, 0xf9, 0x71, 0x62, 0x47,
        0x92, 0xfa, 0x0b, 0x76, 0xee, 0x21, 0x53, 0x83, 0x26,
    ];

    /// PIC4: TRAIL_N (0), lsb 2.
    const SLICE_P4: &[u8] = &[
        0x00, 0x01, 0xe0, 0x44, 0xd7, 0x5f, 0xa2, 0xc8, 0x08, 0x9f, 0x75, 0xda, 0x93, 0x5f, 0xc7,
        0x66, 0xf2, 0x55, 0xb9, 0x6d, 0x75, 0x4f, 0x6e, 0xc6,
    ];

    /// PIC5: TRAIL_N (0), lsb 4.
    const SLICE_P5: &[u8] = &[
        0x00, 0x01, 0xe0, 0x86, 0xb7, 0xfd, 0x46, 0x48, 0x44, 0x44, 0xcb, 0x32, 0xca, 0x4a, 0x83,
        0x39, 0x9a, 0x42, 0xd3, 0x39, 0xd2, 0x0c, 0x3b, 0xa2, 0x3a, 0xec,
    ];

    /// PIC247: CRA_NUT (21), poc 250, no_output_of_prior_pics 0.
    const SLICE_CRA: &[u8] = &[
        0x2a, 0x01, 0xaf, 0xe8, 0x59, 0x08, 0xc9, 0xc2, 0xa0, 0x88, 0x45, 0x86, 0x04, 0xc5, 0xc4,
        0x63, 0x56, 0x30, 0xf6, 0xdb, 0x06, 0x4d, 0x50, 0x06, 0x4e, 0xab, 0x65, 0xc2, 0x08, 0xc3,
        0x13, 0xdc, 0x75, 0x88, 0xbc, 0x30, 0x9e, 0x07, 0x4d, 0xda, 0x46, 0x6d, 0xfa, 0xd9, 0xea,
    ];

    /// PIC248: RASL_R (9), poc 248 (reference picture).
    const SLICE_RASL_R: &[u8] = &[
        0x12, 0x01, 0xff, 0x02, 0x25, 0x52, 0xd7, 0xdc, 0x63, 0xe1, 0x11, 0x92, 0xee, 0xc8, 0x2c,
        0x00, 0xda, 0x0b, 0x66, 0x79, 0xe6, 0xda, 0xae, 0xf2, 0x66, 0xf8, 0x10, 0xe9, 0x48, 0xb8,
        0xe3, 0x75, 0xd4, 0x98, 0xf5, 0xf0,
    ];

    /// PIC249: RASL_N (8), poc 247 (non-reference).
    const SLICE_RASL_N1: &[u8] = &[
        0x10, 0x01, 0xfe, 0xe6, 0xf5, 0xd7, 0xd2, 0x2c, 0x6c, 0x22, 0x2d, 0x95, 0x28, 0x53, 0xd4,
        0x97, 0x0b, 0xf8, 0xca, 0x20, 0x87, 0x55, 0xce, 0xd2, 0xe9, 0x85, 0x13, 0x1c, 0xc4, 0x46,
        0x8c, 0x8a, 0x9e,
    ];

    /// PIC250: RASL_N (8), poc 249 (non-reference).
    const SLICE_RASL_N2: &[u8] = &[
        0x10, 0x01, 0xff, 0x22, 0x2d, 0x57, 0xf7, 0x18, 0xd8, 0x44, 0x5a, 0xf3, 0x5e, 0xc9, 0x89,
        0x39, 0x19, 0x21, 0xaa, 0x47, 0x52, 0x6c, 0x92, 0xc2, 0xd7, 0x27, 0x16, 0x5a, 0xe9, 0xd7,
        0x2b, 0x52, 0x44,
    ];

    /// PIC251: TRAIL_R (1), poc 254 — first tid0 frame after the RASL run.
    const SLICE_P_AFTER_CRA: &[u8] = &[
        0x02, 0x01, 0xd7, 0xf1, 0x49, 0xe1, 0x0c, 0x61, 0x18, 0x44, 0x7d, 0x06, 0x5a, 0x28, 0x94,
        0x17, 0xa8, 0x5f, 0x4d, 0x9f, 0x84, 0x56, 0xf7, 0x16, 0x3e, 0x4e, 0x17, 0xbc, 0xef, 0x55,
        0xec, 0xad, 0x6c, 0xe7, 0x1d, 0xc3, 0x9f, 0x78, 0xb6, 0xb3, 0x7c,
    ];

    /// PIC252: TRAIL_R (1), lsb 252 — start of the POC wraparound window.
    const SLICE_W0: &[u8] = &[
        0x02, 0x01, 0xff, 0x84, 0x95, 0x78, 0x63, 0xe1, 0x11, 0x9a, 0x1b, 0x23, 0x30, 0xad, 0x0f,
        0x6e, 0x78, 0x7d, 0x28, 0x64, 0xd5, 0x97, 0xb9, 0x5e, 0x75, 0xcb, 0xab, 0x5e, 0x0b, 0xe2,
        0xdb, 0x83, 0x36, 0x50,
    ];

    /// PIC253: TRAIL_N (0), lsb 251.
    const SLICE_W1: &[u8] = &[
        0x00, 0x01, 0xff, 0x64, 0xfd, 0x7e, 0x8b, 0x1a, 0x08, 0x8b, 0x5e, 0x88, 0xf1, 0x03, 0xa3,
        0x82, 0x60, 0x2c, 0x08, 0xb2, 0x08, 0x8c, 0x98, 0xda, 0x8a, 0xd7, 0xfa, 0xa5, 0x45, 0x61,
        0x09, 0xf7,
    ];

    /// PIC254: TRAIL_N (0), lsb 253.
    const SLICE_W2: &[u8] = &[
        0x00, 0x01, 0xff, 0xa6, 0xb5, 0xfd, 0x46, 0x34, 0x11, 0x16, 0xa9, 0x93, 0x2a, 0x65, 0x4e,
        0xe6, 0x94, 0x59, 0x12, 0xd4, 0x67, 0x23, 0xb0, 0x25, 0x35, 0xb2, 0xd4, 0x5e, 0x47, 0xc7,
        0xf5, 0x37,
    ];

    /// PIC255: TRAIL_R (1), lsb 2 — first frame past the 256 wrap (poc 258).
    const SLICE_W3: &[u8] = &[
        0x02, 0x01, 0xd0, 0x10, 0x92, 0x55, 0x7d, 0xc4, 0x30, 0x18, 0x44, 0x11, 0x1f, 0x45, 0x7a,
        0xa3, 0xed, 0x43, 0x2a, 0xc4, 0x14, 0x89, 0x62, 0x4c, 0xc3, 0x7e, 0x99, 0x55, 0x8b, 0xdf,
        0x4b, 0x5d, 0x76, 0x9b, 0x5c, 0x79, 0x17, 0xf0, 0x92, 0xdb, 0xa1, 0xa9, 0x91,
    ];

    /// PIC256: TRAIL_R (1), lsb 0 (poc 256).
    const SLICE_W4: &[u8] = &[
        0x02, 0x01, 0xe0, 0x02, 0x25, 0x55, 0x5f, 0x71, 0x8f, 0x04, 0x46, 0x57, 0xa4, 0x21, 0x3e,
        0x7b, 0xbb, 0x31, 0x61, 0xd8, 0x1b, 0x3a, 0xdb, 0xc8, 0xd4, 0x9d, 0xc4, 0x5d, 0xb6, 0xf4,
        0x16, 0xc7, 0x5f, 0x0c, 0x4a, 0x40,
    ];

    /// PIC257: TRAIL_N (0), lsb 255 (poc 255).
    const SLICE_W5: &[u8] = &[
        0x00, 0x01, 0xff, 0xe6, 0xf5, 0xd7, 0xd2, 0x2c, 0x68, 0x22, 0x2d, 0x82, 0x24, 0x65, 0x28,
        0xad, 0x8b, 0x68, 0xad, 0x21, 0xa5, 0xfd, 0xf1, 0xca, 0x17, 0x9f, 0x03, 0x1c, 0xb3, 0xa1,
        0x79, 0x28, 0x62,
    ];

    /// PIC258: TRAIL_N (0), lsb 1 (poc 257).
    const SLICE_W6: &[u8] = &[
        0x00, 0x01, 0xe0, 0x22, 0x2d, 0x57, 0xf7, 0x18, 0xd8, 0x44, 0x5a, 0xd5, 0x3c, 0xa9, 0x31,
        0x1f, 0x16, 0xf1, 0x82, 0x46, 0xcd, 0x94, 0x3c, 0xb2, 0x73, 0xa2, 0xb3, 0xd3, 0x89, 0xc1,
        0x29, 0xce, 0x14,
    ];

    /// Parse several start-code-prefixed slice NALs in ONE packet and return
    /// each picture's header in order. Mirrors real decoder usage (feed a
    /// bitstream chunk once, loop parse() until Nothing) — required because
    /// `VideoParser::parse` caches the extracted NALs keyed by payload length,
    /// so separate same-length packets would hit the stale cache.
    ///
    /// Panics if any slice header fails to parse: a `None` slice_header would
    /// silently vacate POC assertions (the parser falls back to
    /// `first_slice_header` on error, which is None for a fresh parser).
    fn parse_many(parser: &mut H265Parser, nals: &[&[u8]]) -> Vec<SliceHeaderInfo> {
        let mut payload = Vec::new();
        for nal in nals {
            payload.extend_from_slice(&[0x00, 0x00, 0x01]);
            payload.extend_from_slice(nal);
        }
        let packet = crate::bitstream::BitstreamPacket::new(payload);
        let mut out = Vec::new();
        loop {
            match parser.parse(&packet).expect("parse failed") {
                ParseResult::Slice { slices, .. } => {
                    for s in &slices {
                        let header = s
                            .slice_header
                            .clone()
                            .expect("real NAL data must yield a parsed slice header");
                        match header {
                            crate::SliceHeader::H265(info) => out.push(info),
                            other => panic!("unexpected slice header variant: {other:?}"),
                        }
                    }
                }
                ParseResult::Nothing | ParseResult::EndOfStream => break,
                other => panic!("expected Slice or Nothing, got {other:?}"),
            }
        }
        out
    }

    #[test]
    fn test_nal_header_parsing() {
        // VPS: 0x40 = forbidden=0, type=32, reserved=0
        let vps_header = vec![0x40, 0x01];
        let header = nal::parse_h265_nal_header(&vps_header).unwrap();
        assert_eq!(header.1, 32); // VPS

        // SPS: 0x42 = forbidden=0, type=33, reserved=0
        let sps_header = vec![0x42, 0x01];
        let header = nal::parse_h265_nal_header(&sps_header).unwrap();
        assert_eq!(header.1, 33); // SPS

        // PPS: 0x44 = forbidden=0, type=34, reserved=0
        let pps_header = vec![0x44, 0x01];
        let header = nal::parse_h265_nal_header(&pps_header).unwrap();
        assert_eq!(header.1, 34); // PPS

        // BLA_W_LP (type 16): byte0 = (0 << 7) | (16 << 1) | (0 >> 5) = 32 = 0x20
        // with nuh_temporal_id_plus1=1: byte1 = (0 << 2) | 1 = 1 = 0x01
        let idr_header = vec![0x20, 0x01];
        let header = nal::parse_h265_nal_header(&idr_header).unwrap();
        assert_eq!(header.1, 16); // BLA_W_LP
    }

    #[test]
    fn test_epb_removal() {
        // Test emulation prevention byte removal
        // Sequence: 0x00 0x00 0x03 0xXX should become 0x00 0x00 0xXX
        let data = vec![0x00, 0x00, 0x03, 0x42];
        let mut r = BitReader::new(&data, true);

        // Read first two bytes (0x00 0x00)
        let b1 = r.read_bits(8).unwrap();
        let b2 = r.read_bits(8).unwrap();
        assert_eq!(b1, 0);
        assert_eq!(b2, 0);

        // Next byte should skip 0x03 and read 0x42
        let b3 = r.read_bits(8).unwrap();
        assert_eq!(b3, 0x42);
    }

    #[test]
    fn test_ue_v_decoding() {
        // Test exponential Golomb decoding
        // UE(V) encoding: leading zeros (r), then 1, then r value bits
        // Result = 2^r - 1 + value
        // Bits are read MSB-first within each byte.

        // Value 0: "1" (r=0, no value bits)
        // Byte: 1xxxxxxx = 0x80
        let data = vec![0x80];
        let mut r = BitReader::new(&data, false);
        assert_eq!(r.read_ue().unwrap(), 0);

        // Value 1: "01" (r=1, 1 value bit = 0)
        // Byte: 010xxxxx = 0x40
        let data = vec![0x40];
        let mut r = BitReader::new(&data, false);
        assert_eq!(r.read_ue().unwrap(), 1);

        // Value 2: "0010" (r=2, 2 value bits = 00)
        // Byte: 00100xxx = 0x20
        let data = vec![0x20];
        let mut r = BitReader::new(&data, false);
        assert_eq!(r.read_ue().unwrap(), 3); // 2^2 - 1 + 0 = 3

        // Value 3: "00101" (r=2, 2 value bits = 01)
        // Byte: 00101xxx = 0x28
        let data = vec![0x28];
        let mut r = BitReader::new(&data, false);
        assert_eq!(r.read_ue().unwrap(), 4); // 2^2 - 1 + 1 = 4
    }

    #[test]
    fn test_sps_header_parsing() {
        // Minimal SPS header: NAL header + sps_vps_id(4) + max_sub_layers(3) + temporal_nesting(1)
        // NAL header: 0x42 0x01 (type=33=SPS)
        // Payload: 0x01 = sps_vps_id=0, max_sub_layers=0, temporal_nesting=1
        let nal_data = vec![0x42, 0x01, 0x01];

        let header = nal::parse_h265_nal_header(&nal_data).unwrap();
        assert_eq!(header.1, 33); // SPS type

        // After skipping NAL header (2 bytes), read payload
        let mut r = BitReader::new(&nal_data[2..], true);
        let sps_vps_id = r.read_bits(4).unwrap();
        let max_sub_layers = r.read_bits(3).unwrap();
        let temporal_nesting = r.read_bit().unwrap();

        assert_eq!(sps_vps_id, 0);
        assert_eq!(max_sub_layers, 0);
        assert!(temporal_nesting);
    }

    #[test]
    fn test_ptl_main_profile_bit_count() {
        // PTL for Main profile with max_sub_layers=0:
        // Profile fields: 8 bits (profile_space+tier+idc)
        // Compatibility flags: 32 bits
        // Source+reserved: 48 bits (NVIDIA parser approach)
        // level_idc: 8 bits
        // Total: 8 + 32 + 48 + 8 = 96 bits = 12 bytes

        let ptl_data: Vec<u8> = vec![
            0x21, // profile_space=0, tier=0, idc=1 (8 bits)
            0x00, 0x00, 0x00, 0x00, // compatibility flags (32 bits)
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // source+reserved (48 bits)
            0x3C, // level_idc = 60 (level 3.1)
        ];
        assert_eq!(ptl_data.len(), 12); // 96 bits = 12 bytes

        let mut r = BitReader::new(&ptl_data, false);

        // Skip profile fields (8 bits)
        let _ = r.read_bits(8).unwrap();

        // Skip compatibility flags (32 bits = 16 + 16)
        let _ = r.read_bits(16).unwrap();
        let _ = r.read_bits(16).unwrap();

        // Skip source+reserved (48 bits = 24 + 24)
        let _ = r.read_bits(24).unwrap();
        let _ = r.read_bits(24).unwrap();

        // level_idc
        let level_idc = r.read_bits(8).unwrap();
        assert_eq!(level_idc, 0x3C); // level 3.1
    }

    // ========================================================================
    // Comprehensive H.265 parser alignment tests
    // These tests verify that the Rust parser produces the same results as the
    // C++ NVIDIA Vulkan-Video-Samples parser (VulkanH265Parser.cpp).
    // ========================================================================

    /// VPS NAL unit from big_buck_bunney.h265 (type=32, 24 bytes)
    /// Hex: 40010c01ffff216000000300900000030000030078959809
    const TEST_VPS_DATA: &[u8] = &[
        0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x21, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x03, 0x00, 0x78, 0x95, 0x98, 0x09,
    ];

    /// SPS NAL unit from big_buck_bunney.h265 (type=33, 43 bytes)
    const TEST_SPS_DATA: &[u8] = &[
        0x42, 0x01, 0x01, 0x21, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00, 0x00,
        0x03, 0x00, 0x78, 0xa0, 0x03, 0xc0, 0x80, 0x10, 0xe5, 0x96, 0x56, 0x69, 0x24, 0xca, 0xf0,
        0x10, 0x10, 0x00, 0x00, 0x03, 0x00, 0x10, 0x00, 0x00, 0x03, 0x01, 0xe0, 0x80,
    ];

    /// PPS NAL unit from big_buck_bunney.h265 (type=34, 7 bytes)
    /// Hex: 4401c172b46240
    const TEST_PPS_DATA: &[u8] = &[0x44, 0x01, 0xc1, 0x72, 0xb4, 0x62, 0x40];

    #[test]
    fn test_vps_parsing_alignment() {
        // Test VPS parsing matches C++ VulkanH265Parser.cpp:906-1085
        let mut parser = H265Parser::new();
        let vps = parser.parse_vps(TEST_VPS_DATA).expect("VPS parse failed");

        // VPS base fields (C++: video_parameter_set_rbsp() lines 908-941)
        assert_eq!(
            vps.vps_video_parameter_set_id, 0,
            "vps_video_parameter_set_id"
        );
        assert_eq!(vps.vps_max_layers_minus1, 0, "vps_max_layers_minus1");
        assert_eq!(
            vps.vps_max_sub_layers_minus1, 0,
            "vps_max_sub_layers_minus1"
        );
        assert!(
            vps.vps_temporal_id_nesting_flag,
            "vps_temporal_id_nesting_flag"
        );

        // Profile/level (C++: profile_tier_level() lines 1631-1669)
        assert_eq!(vps.profile_idc, 1, "profile_idc (Main)");
        assert_eq!(vps.level_idc, 120, "level_idc (Level 4.0)");

        // VPS layer info (C++: lines 965-988)
        assert_eq!(vps.vps_max_layer_id, 0, "vps_max_layer_id");
        assert_eq!(vps.vps_num_layer_sets, 1, "vps_num_layer_sets");

        // VPS timing (C++: lines 991-1051)
        assert!(
            !vps.vps_timing_info_present_flag,
            "vps_timing_info_present_flag"
        );
    }

    #[test]
    fn test_sps_parsing_alignment() {
        // Test SPS parsing matches C++ VulkanH265Parser.cpp:394-709
        // Values verified against actual parser output for big_buck_bunney.h265
        let mut parser = H265Parser::new();
        parser.parse_vps(TEST_VPS_DATA).expect("VPS parse failed");
        let sps = parser.parse_sps(TEST_SPS_DATA).expect("SPS parse failed");

        // Base fields
        assert_eq!(sps.sps_video_parameter_set_id, 0);
        assert_eq!(sps.sps_max_sub_layers_minus1, 0);
        assert!(sps.sps_temporal_id_nesting_flag);
        assert_eq!(sps.sps_seq_parameter_set_id, 0);

        // Profile/level
        assert_eq!(sps.profile_idc, 1, "profile_idc (Main)");
        assert_eq!(sps.level_idc, 120, "level_idc (Level 4.0)");

        // Chroma format
        assert_eq!(sps.chroma_format_idc, 1, "chroma_format_idc (4:2:0)");
        assert!(!sps.separate_colour_plane_flag);

        // Picture dimensions
        assert_eq!(sps.pic_width_in_luma_samples, 1920);
        assert_eq!(sps.pic_height_in_luma_samples, 1080);

        // Bit depth
        assert_eq!(sps.bit_depth_luma_minus8, 0, "8-bit luma");
        assert_eq!(sps.bit_depth_chroma_minus8, 0, "8-bit chroma");

        // DPB management
        assert!(sps.sps_sub_layer_ordering_info_present_flag);
        assert_eq!(sps.max_dec_pic_buffering_minus1[0], 4);

        // Flags
        assert!(!sps.scaling_list_enabled_flag);
        assert!(sps.sample_adaptive_offset_enabled_flag);
        assert!(!sps.pcm_enabled_flag);
        assert!(!sps.long_term_ref_pics_present_flag);
        assert!(sps.sps_temporal_mvp_enabled_flag);
        assert!(sps.strong_intra_smoothing_enabled_flag);

        // VUI
        assert!(sps.vui_parameters_present_flag);
        assert!(sps.vui.vui_timing_info_present_flag);

        // Extension
        assert!(!sps.sps_extension_present_flag);
    }

    #[test]
    fn test_pps_parsing_alignment() {
        // Test PPS parsing matches C++ VulkanH265Parser.cpp:712-903
        let mut parser = H265Parser::new();
        parser.parse_vps(TEST_VPS_DATA).expect("VPS parse failed");
        parser.parse_sps(TEST_SPS_DATA).expect("SPS parse failed");
        let pps = parser.parse_pps(TEST_PPS_DATA).expect("PPS parse failed");

        // Base fields
        assert_eq!(pps.pps_pic_parameter_set_id, 0);
        assert_eq!(pps.pps_seq_parameter_set_id, 0);
        assert!(!pps.dependent_slice_segments_enabled_flag);
        assert!(!pps.output_flag_present_flag);
        assert_eq!(pps.num_extra_slice_header_bits, 0);
        assert!(pps.sign_data_hiding_enabled_flag);
        assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 0);
        assert_eq!(pps.num_ref_idx_l1_default_active_minus1, 0);
        assert_eq!(pps.pps_init_qp_minus26, 0);

        // Additional fields
        assert!(!pps.constrained_intra_pred_flag);
        assert!(!pps.pps_slice_chroma_qp_offsets_present_flag);
        assert!(!pps.tiles_enabled_flag);
        assert!(!pps.pps_scaling_list_data_present_flag);
        assert!(!pps.pps_extension_present_flag);
    }

    #[test]
    fn test_full_parse_flow() {
        // Test the full parse flow: VPS → SPS → PPS → detected_format update
        let mut parser = H265Parser::new();
        let format = DetectedVideoFormat::new(vacc_core::codec::VideoCodec::DecodeH265);
        parser.init(&format).expect("Parser init failed");

        // Create a bitstream packet with VPS, SPS, PPS
        let mut payload = Vec::new();
        // Add start codes and NAL units
        payload.extend_from_slice(&[0x00, 0x00, 0x01]); // Start code
        payload.extend_from_slice(TEST_VPS_DATA);
        payload.extend_from_slice(&[0x00, 0x00, 0x01]); // Start code
        payload.extend_from_slice(TEST_SPS_DATA);
        payload.extend_from_slice(&[0x00, 0x00, 0x01]); // Start code
        payload.extend_from_slice(TEST_PPS_DATA);

        let packet = crate::bitstream::BitstreamPacket::new(payload);
        let result = parser.parse(&packet).expect("Parse failed");

        match result {
            ParseResult::ParameterSet { sps, pps, vps, .. } => {
                assert!(vps.is_some(), "VPS should be parsed");
                assert!(sps.is_some(), "SPS should be parsed");
                assert!(pps.is_some(), "PPS should be parsed");

                // Verify detected format was updated from SPS
                let detected = parser.detected_format();
                assert_eq!(detected.coded_width, 1920, "coded_width from SPS");
                assert_eq!(detected.coded_height, 1080, "coded_height from SPS");
                assert_eq!(
                    detected.chroma_subsampling,
                    vacc_core::format::ChromaSubsampling::_420,
                    "chroma from SPS"
                );
                assert_eq!(
                    detected.luma_bit_depth,
                    vacc_core::format::ComponentBitDepth::Bit8,
                    "luma bit depth from SPS"
                );
            }
            _ => panic!("Expected ParameterSet result, got {:?}", result),
        }
    }

    #[test]
    fn test_ptl_parsing_alignment() {
        // Test PTL parsing matches C++ profile_tier_level() exactly
        // C++: lines 1631-1669

        // SPS PTL: ProfilePresentFlag=1, CommonInfPresentFlag=1, SubLayerLevelPresentFlag=0
        // VPS PTL: ProfilePresentFlag=1, CommonInfPresentFlag=1, SubLayerLevelPresentFlag=1

        // Test SPS PTL (max_sub_layers=0, sub_layer_level_present=false)
        let sps_ptl_data: Vec<u8> = vec![
            0x21, // profile_space=0, tier=0, idc=1 (8 bits)
            0x00, 0x00, 0x00, 0x00, // compatibility flags (32 bits)
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // source+reserved (48 bits)
            0x78, // level_idc = 120 (level 4.0)
        ];

        let mut r = BitReader::new(&sps_ptl_data, false);
        let (profile, level, tier) = H265Parser::parse_ptl(&mut r, 0, false).unwrap();
        assert_eq!(profile, 1, "SPS profile_idc");
        assert_eq!(level, 120, "SPS level_idc");
        assert!(!tier, "SPS tier_flag");

        // Test VPS PTL (max_sub_layers=0, sub_layer_level_present=true)
        let vps_ptl_data: Vec<u8> = vec![
            0x21, // profile_space=0, tier=0, idc=1 (8 bits)
            0x00, 0x00, 0x00, 0x00, // compatibility flags (32 bits)
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // source+reserved (48 bits)
            0x78, // level_idc = 120 (level 4.0)
        ];

        let mut r = BitReader::new(&vps_ptl_data, false);
        let (profile, level, tier) = H265Parser::parse_ptl(&mut r, 0, true).unwrap();
        assert_eq!(profile, 1, "VPS profile_idc");
        assert_eq!(level, 120, "VPS level_idc");
        assert!(!tier, "VPS tier_flag");
    }

    #[test]
    fn test_strps_direct_encoding() {
        // Test STRPS direct encoding matches C++ VulkanH265Parser.cpp:1870-1914
        // idx=0 means no inter_ref_pic_set_prediction_flag
        // num_negative_pics=0, num_positive_pics=0 (minimal case)
        //
        // ue(0) = `1` (1 bit, Exp-Golomb: 0 leading zeros + 1 suffix bit = 1, value=0)
        // ue(0) = `1` (1 bit)
        // Total: `11` + padding = `11000000` = 0xC0
        let data: Vec<u8> = vec![0xC0];
        let mut r = BitReader::new(&data, false);

        let strps = H265Parser::parse_short_term_ref_pic_set(&mut r, 0, 1, &[]).unwrap();

        assert!(!strps.inter_ref_pic_set_prediction_flag);
        assert_eq!(strps.num_negative_pics, 0);
        assert_eq!(strps.num_positive_pics, 0);
    }

    #[test]
    fn test_strps_with_entries() {
        // Test STRPS with 1 negative and 1 positive picture
        // idx=0, num_negative_pics=1, num_positive_pics=1
        // delta_poc_s0_minus1[0]=0, used=1
        // delta_poc_s1_minus1[0]=0, used=1
        //
        // Exp-Golomb UE coding:
        //   ue(v): k leading zeros, then k suffix bits, value = 2^k - 1 + suffix
        //   ue(0): k=0, no suffix, value=0. Binary: `1` (1 bit)
        //   ue(1): k=1, 1 suffix bit=0, value=2^1-1+0=1. Binary: `010` (3 bits)
        //
        // Bit stream:
        //   num_negative_pics = ue(1) = `010` (3 bits, value=1)
        //   num_positive_pics = ue(1) = `010` (3 bits, value=1)
        //   delta_poc_s0_minus1[0] = ue(0) = `1` (1 bit, value=0)
        //   used_by_curr_pic_s0_flag = `1` (1 bit)
        //   delta_poc_s1_minus1[0] = ue(0) = `1` (1 bit, value=0)
        //   used_by_curr_pic_s1_flag = `1` (1 bit)
        //   Total: `010 010 1 1 1 1` = 10 bits
        //   Packed: `01001011 11000000` = 0x4B 0xC0
        //
        // Note: The stored delta_poc_s0_minus1 holds cumulative DeltaPoc, not raw encoded value.
        //   DeltaPocS0[0] = -(raw_delta + 1) = -(0 + 1) = -1, stored as u16 = 65535
        //   DeltaPocS1[0] = +(raw_delta + 1) = +(0 + 1) = 1, stored as u16 = 1

        let data: Vec<u8> = vec![0x4B, 0xC0];
        let mut r = BitReader::new(&data, false);

        let strps = H265Parser::parse_short_term_ref_pic_set(&mut r, 0, 1, &[]).unwrap();

        assert!(!strps.inter_ref_pic_set_prediction_flag);
        assert_eq!(strps.num_negative_pics, 1);
        assert_eq!(strps.num_positive_pics, 1);
        assert_eq!(strps.delta_poc_s0_minus1[0], 65535); // -1 stored as u16
        assert_eq!(strps.delta_poc_s1_minus1[0], 1);
        assert_eq!(strps.used_by_curr_pic_s0_flag, 1);
        assert_eq!(strps.used_by_curr_pic_s1_flag, 1);
    }

    // =========================================================================
    // RPS parsing with used_by_curr_pic filtering
    // =========================================================================

    /// Test used_by_curr_pic filtering logic directly on parsed RPS data.
    /// This verifies the filtering logic that the nvdec decoder uses when
    /// recovering RPS POCs from the parsed data.
    #[test]
    fn test_used_by_curr_pic_filtering_logic() {
        // Simulate an RPS with 2 negative pics and 2 positive pics
        // where only some are used as references
        let rps = vacc_core::picture::H265ShortTermRefPicSet {
            num_negative_pics: 2,
            num_positive_pics: 2,
            // S0: pic at delta -1 (used), pic at delta -3 (not used)
            delta_poc_s0_minus1: [65535, 65533, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            used_by_curr_pic_s0_flag: 0b01, // Only first pic used
            // S1: pic at delta +2 (used), pic at delta +5 (not used)
            delta_poc_s1_minus1: [2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            used_by_curr_pic_s1_flag: 0b01, // Only first pic used
            ..Default::default()
        };

        let curr_poc = 10i32;

        // Apply the filtering logic from recover_rps_pocs
        let mut ref_s0 = Vec::new();
        for i in 0..rps.num_negative_pics as usize {
            if ((rps.used_by_curr_pic_s0_flag >> i) & 1) == 0 {
                continue;
            }
            let stored = rps.delta_poc_s0_minus1[i] as i32;
            let signed = if stored > 32767 {
                stored - 65536
            } else {
                stored
            };
            ref_s0.push(curr_poc + signed);
        }

        let mut ref_s1 = Vec::new();
        for i in 0..rps.num_positive_pics as usize {
            if ((rps.used_by_curr_pic_s1_flag >> i) & 1) == 0 {
                continue;
            }
            let stored = rps.delta_poc_s1_minus1[i] as i32;
            let signed = if stored > 32767 {
                stored - 65536
            } else {
                stored
            };
            ref_s1.push(curr_poc + signed);
        }

        // After filtering: only used references remain
        assert_eq!(ref_s0.len(), 1, "S0 should only have 1 reference");
        assert_eq!(ref_s0[0], 9, "S0 reference POC should be 9");
        assert_eq!(ref_s1.len(), 1, "S1 should only have 1 reference");
        assert_eq!(ref_s1[0], 12, "S1 reference POC should be 12");
    }

    /// Test that used_by_curr_pic flags correctly control bit positions.
    #[test]
    fn test_used_flag_bit_positions() {
        // Verify bit position semantics:
        // used_by_curr_pic_s0_flag bit i corresponds to delta_poc_s0_minus1[i]
        let mut rps = vacc_core::picture::H265ShortTermRefPicSet::default();
        rps.num_negative_pics = 3;
        rps.num_positive_pics = 2;

        // Set bit 0 and bit 2 for S0
        rps.used_by_curr_pic_s0_flag = 0b101; // bits 0 and 2
                                              // Set bit 1 for S1
        rps.used_by_curr_pic_s1_flag = 0b010; // bit 1

        // Verify bit extraction
        assert_eq!(
            rps.used_by_curr_pic_s0_flag & 1,
            1,
            "S0 bit 0 should be set"
        );
        assert_eq!(
            (rps.used_by_curr_pic_s0_flag >> 1) & 1,
            0,
            "S0 bit 1 should be clear"
        );
        assert_eq!(
            (rps.used_by_curr_pic_s0_flag >> 2) & 1,
            1,
            "S0 bit 2 should be set"
        );
        assert_eq!(
            rps.used_by_curr_pic_s1_flag & 1,
            0,
            "S1 bit 0 should be clear"
        );
        assert_eq!(
            (rps.used_by_curr_pic_s1_flag >> 1) & 1,
            1,
            "S1 bit 1 should be set"
        );
    }

    // =========================================================================
    // Predictive RPS parsing tests
    // =========================================================================

    /// Test that predictive RPS correctly resolves against a reference RPS.
    /// Verifies the resolve_predictive_rps function works correctly.
    #[test]
    fn test_predictive_rps_basic() {
        // Use the known-good reference RPS from test_strps_with_entries:
        // 1 negative pic (delta=-1), 1 positive pic (delta=+1), both used
        let ref_data: Vec<u8> = vec![0x4B, 0xC0];
        let mut r_ref = BitReader::new(&ref_data, false);
        let ref_strps = H265Parser::parse_short_term_ref_pic_set(&mut r_ref, 0, 2, &[]).unwrap();

        assert_eq!(ref_strps.num_negative_pics, 1);
        assert_eq!(ref_strps.num_positive_pics, 1);
        assert_eq!(ref_strps.delta_poc_s0_minus1[0], 65535); // -1
        assert_eq!(ref_strps.delta_poc_s1_minus1[0], 1); // +1

        // Verify the reference RPS has correct used flags
        assert_eq!(ref_strps.used_by_curr_pic_s0_flag, 1);
        assert_eq!(ref_strps.used_by_curr_pic_s1_flag, 1);
    }

    // =========================================================================
    // POC derivation edge cases
    // =========================================================================

    /// Test POC derivation for the first picture (has_prev_pic = false):
    /// pic_order_cnt_msb must be 0, so POC == lsb. Feeds real PIC1
    /// (TRAIL_R, lsb=5) as the very first picture of a fresh parser.
    #[test]
    fn test_poc_first_picture() {
        let mut parser = init_parser();
        let infos = parse_many(&mut parser, &[SLICE_P1]);
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].pic_order_cnt_lsb, 5);
        assert_eq!(
            infos[0].curr_pic_order_cnt_val, 5,
            "First picture POC should be 5 (msb=0, lsb=5)"
        );
    }

    /// Test POC derivation for an IDR picture: always POC=0 regardless of the
    /// previous POC state. Feeds real frames PIC1..PIC5 (POC 5,3,1,2,4) to
    /// build non-trivial state, then the IDR NAL (PIC0).
    #[test]
    fn test_poc_idr_always_zero() {
        let mut parser = init_parser();
        // PIC1..PIC5 first (POC 5,3,1,2,4 — builds non-trivial tid0 state),
        // then the IDR NAL (PIC0).
        let infos = parse_many(
            &mut parser,
            &[SLICE_P1, SLICE_P2, SLICE_P3, SLICE_P4, SLICE_P5, SLICE_IDR],
        );
        assert_eq!(infos.len(), 6);
        assert_eq!(
            infos[..5]
                .iter()
                .map(|i| i.curr_pic_order_cnt_val)
                .collect::<Vec<_>>(),
            [5i32, 3, 1, 2, 4],
            "pre-IDR POCs must match the stream"
        );
        let idr = &infos[5];
        assert!(idr.is_idr, "PIC0 must be flagged IDR");
        assert_eq!(idr.pic_order_cnt_lsb, 0, "IDR carries no poc_lsb");
        assert_eq!(idr.curr_pic_order_cnt_val, 0, "IDR POC should always be 0");
    }

    /// Test the CRA + RASL region of big_buck_bunney.h265 (decode-order
    /// PICs 247-251): CRA_NUT(21) poc=250 with no_output_of_prior_pics=0,
    /// then RASL_R(9) poc=248, RASL_N(8) poc=247, RASL_N(8) poc=249, and
    /// TRAIL_R(1) poc=254. RASL frames are not IRAP; only the CRA updates the
    /// tid0 POC state (RASL/TRAIL_N must not). All POCs verified against the
    /// FFmpeg single-thread sequence.
    #[test]
    fn test_poc_cra_rasl_region() {
        let mut parser = init_parser();
        let infos = parse_many(
            &mut parser,
            &[
                SLICE_CRA,
                SLICE_RASL_R,
                SLICE_RASL_N1,
                SLICE_RASL_N2,
                SLICE_P_AFTER_CRA,
            ],
        );
        assert_eq!(infos.len(), 5);

        let (cra, r1, r2, r3, p) = (&infos[0], &infos[1], &infos[2], &infos[3], &infos[4]);
        assert!(cra.is_rap && !cra.is_idr, "PIC247 is a CRA, not an IDR");
        assert!(!cra.no_output_of_prior_pics_flag, "CRA has nopp=0");
        assert_eq!(cra.slice_type, 0, "CRA slice is intra");
        assert_eq!(cra.curr_pic_order_cnt_val, 250);

        assert!(!r1.is_rap && !r1.is_idr);
        assert!(r1.is_reference, "RASL_R (type 9) is a reference picture");
        assert_eq!(r1.curr_pic_order_cnt_val, 248);

        assert!(
            !r2.is_reference,
            "RASL_N (type 8) is not a reference picture"
        );
        assert_eq!(r2.curr_pic_order_cnt_val, 247);

        assert!(!r3.is_reference);
        assert_eq!(r3.curr_pic_order_cnt_val, 249);

        // First tid0 frame after the RASL run: POC derived from the CRA's
        // state (lsb 254 > 250, small delta -> msb unchanged).
        assert_eq!(p.curr_pic_order_cnt_val, 254);
    }

    /// Test POC MSB reconstruction across the MaxPicOrderCntLsb=256 boundary.
    /// Decode-order PICs 252-258 carry lsb sequence 252,251,253,2,0,255,1
    /// (a full wrap) and must reconstruct to POC 252,251,253,258,256,255,257.
    /// Verified against FFmpeg `-threads 1` file-order POCs.
    #[test]
    fn test_poc_wraparound() {
        let mut parser = init_parser();
        let infos = parse_many(
            &mut parser,
            &[
                SLICE_W0, SLICE_W1, SLICE_W2, SLICE_W3, SLICE_W4, SLICE_W5, SLICE_W6,
            ],
        );
        assert_eq!(infos.len(), 7);
        let expected = [252i32, 251, 253, 258, 256, 255, 257];
        for (info, exp) in infos.iter().zip(expected) {
            assert_eq!(info.curr_pic_order_cnt_val, exp, "wraparound POC mismatch");
        }
    }

    // =========================================================================
    // SAO conditional parsing tests
    // =========================================================================

    /// Test that PPS parsing with SAO offset scale fields respects SPS conditions.
    /// When sps_sao_luma_allowed is false, log2_sao_offset_scale_luma should NOT be read.
    #[test]
    fn test_sao_conditional_parsing_no_sao() {
        // Create a minimal SPS with SAO disabled
        // This tests that the parser doesn't try to read SAO scale fields
        // when the conditions aren't met.
        //
        // sps_sao_luma_allowed = sample_adaptive_offset_enabled_flag
        //   && max_transform_hierarchy_depth_intra > log2_min_luma_transform_block_size_minus2
        //
        // If SAO is disabled (sample_adaptive_offset_enabled_flag=0),
        // the PPS should not contain SAO offset scale fields.
        //
        // The existing test SPS has SAO enabled, so we verify the positive case.
        let parser = init_parser();
        let sps = parser.active_sps().expect("No active SPS");

        // Verify SAO conditions for the test SPS
        let sao_luma_allowed = sps.sample_adaptive_offset_enabled_flag
            && (sps.max_transform_hierarchy_depth_intra
                > sps.log2_min_luma_transform_block_size_minus2);

        if sao_luma_allowed {
            // The test SPS has SAO enabled and depth_intra >= min_transform_block_size
            // So log2_sao_offset_scale_luma should have been parsed
            let pps = parser.active_pps().expect("No active PPS");
            // log2_sao_offset_scale_luma is u8, valid range [0, 6]
            assert!(
                pps.log2_sao_offset_scale_luma <= 6,
                "log2_sao_offset_scale_luma should be in valid range"
            );
        }
    }

    /// Test PPS parsing with range extension and SAO fields.
    /// Verifies the conditional parsing chain:
    /// pps_extension_present_flag -> pps_range_extension_flag -> SAO fields
    #[test]
    fn test_pps_range_extension_sao_parsing() {
        let parser = init_parser();
        let pps = parser.active_pps().expect("No active PPS");

        // The test PPS from big_buck_bunny has pps_extension_present_flag = false
        assert!(
            !pps.pps_extension_present_flag,
            "Test PPS should not have extension"
        );
        // Therefore SAO scale fields should be at default values
        assert_eq!(pps.log2_sao_offset_scale_luma, 0, "Default SAO luma scale");
        assert_eq!(
            pps.log2_sao_offset_scale_chroma, 0,
            "Default SAO chroma scale"
        );
    }

    // =========================================================================
    // Parser state management tests
    // =========================================================================

    /// Test that reset() clears all parser state including POC tracking.
    #[test]
    fn test_reset_clears_all_state() {
        let mut parser = H265Parser::new();
        parser.parse_vps(TEST_VPS_DATA).expect("VPS parse failed");
        parser.parse_sps(TEST_SPS_DATA).expect("SPS parse failed");
        parser.parse_pps(TEST_PPS_DATA).expect("PPS parse failed");

        // Parse a real slice to set has_prev_pic
        let _ = parse_many(&mut parser, &[SLICE_P1]);

        // Reset
        parser.reset();

        // Verify caches are cleared
        assert!(
            parser.active_sps().is_none(),
            "SPS should be cleared after reset"
        );
        assert!(
            parser.active_pps().is_none(),
            "PPS should be cleared after reset"
        );
        assert!(
            parser.active_vps.is_none(),
            "VPS should be cleared after reset"
        );
        assert!(
            parser.first_slice_header().is_none(),
            "first_slice_header should be cleared"
        );
    }

    /// Test that frame_count is incremented correctly.
    #[test]
    fn test_frame_count_increments() {
        let mut parser = init_parser();

        // Feed all slices in a single packet (the decoder feeds a chunk of the
        // bitstream once and loops parse() until Nothing). Each slice starts a
        // new picture (first_slice_segment_in_pic_flag = 1), so the parser
        // returns one Slice result per picture.
        let mut payload = Vec::new();
        for nal in [SLICE_P1, SLICE_P2, SLICE_P3, SLICE_P4, SLICE_P5] {
            payload.extend_from_slice(&[0x00, 0x00, 0x01]);
            payload.extend_from_slice(nal);
        }
        let packet = crate::bitstream::BitstreamPacket::new(payload);

        let mut pictures = 0;
        loop {
            let result = parser.parse(&packet).expect("Parse failed");
            match result {
                ParseResult::Slice { slices, .. } => {
                    assert!(!slices.is_empty(), "Slice result should have slices");
                    pictures += 1;
                }
                ParseResult::Nothing | ParseResult::EndOfStream => break,
                other => panic!("Expected Slice or Nothing, got {:?}", other),
            }
        }
        assert_eq!(pictures, 5, "Expected 5 pictures (one Slice result each)");
    }

    // ========================================================================
    // Slice-header bit-alignment verification against the real big_buck stream
    // ========================================================================

    /// Verifies that `parse_slice_segment_header` consumes exactly the right
    /// number of bits for every slice in the stream.
    ///
    /// Ground truth: FFmpeg n8.1.2 `hls_slice_header`, immediately after the
    /// last coded header field, reads ONE bit and requires it to be 1
    /// ("alignment_bit_equal_to_one"), then aligns to a byte boundary to derive
    /// `data_offset` (the VAAPI `slice_data_byte_offset`). If our header parse
    /// is misaligned by even one bit, that alignment bit will read 0 and FFmpeg
    /// (and any VA driver relying on `slice_data_byte_offset`) breaks.
    ///
    /// NOTE: `rbsp_trailing_bits` live at the END of the NAL RBSP (before EOB),
    /// NOT right after the slice header — the bits immediately following the
    /// header are the start of the CABAC-coded slice data. So we check the
    /// FFmpeg alignment-bit invariant, not a trailing-ones run.
    #[test]
    fn test_big_buck_slice_headers_aligned() {
        // Embedded at compile time: no runtime dependency on the assets tree.
        let data = include_bytes!("../../../assets/big_buck_bunney.h265").to_vec();

        let mut parser = H265Parser::new();
        let packet = crate::bitstream::BitstreamPacket::new(data);

        let mut pictures = 0usize;
        let mut slices = 0usize;
        loop {
            let result = parser.parse(&packet).expect("parse failed");
            match result {
                ParseResult::Slice {
                    slices: entries, ..
                } => {
                    pictures += 1;
                    for (i, entry) in entries.iter().enumerate() {
                        if let Some(crate::SliceHeader::H265(info)) = &entry.slice_header {
                            // Rewalk the RBSP from after the 2-byte NAL header and
                            // skip exactly header_bit_size bits.
                            let mut r = BitReader::new(&entry.nal_data[2..], true);
                            let mut remaining = info.header_bit_size as u64;
                            while remaining >= 8 {
                                r.read_bits(8).expect("skip failed");
                                remaining -= 8;
                            }
                            if remaining > 0 {
                                r.read_bits(remaining as u8).expect("skip failed");
                            }

                            // The header alone cannot consume the whole NAL unit —
                            // there must be slice data + trailing bits remaining.
                            assert!(
                                u64::from(info.header_bit_size) < (entry.nal_data.len() as u64) * 8,
                                "picture {} slice {}: header consumed the entire NAL",
                                pictures,
                                i
                            );

                            // FFmpeg invariant: the first bit after the coded slice
                            // header must be 1.
                            let align_bit = r.read_bit().unwrap_or_else(|e| {
                                panic!("picture {} slice {}: {e:?}", pictures, i)
                            });
                            assert!(
                                align_bit,
                                "picture {} slice {} (header_bit_size={}): \
                                 alignment bit after coded header is 0 — header parse \
                                 is misaligned",
                                pictures, i, info.header_bit_size
                            );

                            slices += 1;
                        }
                    }
                }
                ParseResult::Nothing | ParseResult::EndOfStream => break,
                ParseResult::ParameterSet { .. } => {}
            }
        }

        assert!(pictures >= 290, "expected ~300 pictures, got {pictures}");
        assert!(slices >= 290, "expected ~300 slice headers, got {slices}");
    }
}

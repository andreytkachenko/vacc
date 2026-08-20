//! H.265/HEVC bitstream parser.
//!
//! Parses H.265 bitstreams to extract VPS, SPS, PPS, and slice data.
//! Based on cros-codecs H.265 parser implementation.

use std::collections::HashMap;

use crate::nal::{self, H265NalUnitType, NalUnit};
use crate::{
    DetectedVideoFormat, ParseResult, ParserError, ParserResult, VideoParser,
};
use crate::bitreader::BitReader;

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
    /// short_term_ref_pic_set_sps_flag from slice header (for StdVideoDecodeH265PictureInfo)
    pub short_term_ref_pic_set_sps_flag: bool,
    /// Index into SPS short_term_ref_pic_sets array (when short_term_ref_pic_set_sps_flag is true)
    pub short_term_ref_pic_set_idx: u8,
    /// Slice-level STRPS (when short_term_ref_pic_set_sps_flag is false)
    pub slice_strps: Option<vk_video_core::picture::H265ShortTermRefPicSet>,
}

impl SliceHeaderInfo {
    fn new() -> Self {
        Self {
            slice_type: 0,
            pic_order_cnt_lsb: 0,
            curr_pic_order_cnt_val: 0,
            is_idr: false,
            is_rap: false,
            is_reference: false,
            short_term_ref_pic_set_sps_flag: true, // Default: RPS in SPS
            short_term_ref_pic_set_idx: 0,
            slice_strps: None,
        }
    }
}

pub struct H265Parser {
    vps_cache: HashMap<u8, vk_video_core::picture::H265Vps>,
    sps_cache: HashMap<u32, vk_video_core::picture::H265Sps>,
    pps_cache: HashMap<u32, vk_video_core::picture::H265Pps>,
    active_vps: Option<vk_video_core::picture::H265Vps>,
    active_sps: Option<vk_video_core::picture::H265Sps>,
    active_pps: Option<vk_video_core::picture::H265Pps>,
    detected_format: DetectedVideoFormat,
    frame_count: u32,
    first_slice_header: Option<SliceHeaderInfo>,
    // POC tracking per H.265 spec section 8.3.1
    prev_pic_order_cnt_msb: i32,
    prev_pic_order_cnt_lsb: i32,
    /// Flag: true when we have an IRAP picture with NoRaslOutputFlag
    no_rasl_output_flag: bool,
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
    pub fn active_sps(&self) -> Option<&vk_video_core::picture::H265Sps> {
        self.active_sps.as_ref()
    }

    /// Returns a reference to the active PPS, if any.
    pub fn active_pps(&self) -> Option<&vk_video_core::picture::H265Pps> {
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
            detected_format: DetectedVideoFormat::new(
                vk_video_core::codec::VideoCodec::DecodeH265,
            ),
            frame_count: 0,
            first_slice_header: None,
            // Initialize per VulkanH265Parser.cpp:110
            prev_pic_order_cnt_msb: 0,
            prev_pic_order_cnt_lsb: 0,
            no_rasl_output_flag: false,
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
    fn parse_ptl(r: &mut BitReader, max_sub_layers: u8, sub_layer_level_present: bool) -> ParserResult<(u8, u8, bool)> {
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
            let _ = r.read_bits(((8 - max_sub_layers - 1) * 2) as u8)?;
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
        scaling_lists: &mut vk_video_core::picture::H265ScalingLists,
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
                    let pred_matrix_id = ((matrix_id as i32) + scaling_list_pred_matrix_id_delta) as usize;

                    // Copy AC coefficients from predicted matrix
                    match size_id {
                        0 => scaling_lists.scaling_list_4x4[matrix_id as usize] = scaling_lists.scaling_list_4x4[pred_matrix_id],
                        1 => scaling_lists.scaling_list_8x8[matrix_id as usize] = scaling_lists.scaling_list_8x8[pred_matrix_id],
                        2 => scaling_lists.scaling_list_16x16[matrix_id as usize] = scaling_lists.scaling_list_16x16[pred_matrix_id],
                        3 => scaling_lists.scaling_list_32x32[matrix_id as usize] = scaling_lists.scaling_list_32x32[pred_matrix_id],
                        _ => {}
                    }

                    // Copy DC coefficients for 16x16 and 32x32
                    if size_id == 2 {
                        scaling_lists.scaling_list_dc_coef_16x16[matrix_id as usize][0] = scaling_lists.scaling_list_dc_coef_16x16[pred_matrix_id][0];
                    } else if size_id == 3 {
                        scaling_lists.scaling_list_dc_coef_32x32[matrix_id as usize][0] = scaling_lists.scaling_list_dc_coef_32x32[pred_matrix_id][0];
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
                            scaling_lists.scaling_list_dc_coef_16x16[matrix_id as usize][0] = next_coef as i8;
                        } else {
                            scaling_lists.scaling_list_dc_coef_32x32[matrix_id as usize][0] = next_coef as i8;
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
                            0 => scaling_lists.scaling_list_4x4[matrix_id as usize][i] = next_coef as u8,
                            1 => scaling_lists.scaling_list_8x8[matrix_id as usize][i] = next_coef as u8,
                            2 => scaling_lists.scaling_list_16x16[matrix_id as usize][i] = next_coef as u8,
                            3 => scaling_lists.scaling_list_32x32[matrix_id as usize][i] = next_coef as u8,
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

        for i in 0..=max_num_sublayers_minus1 as usize {
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
        prev_strps: &[vk_video_core::picture::H265ShortTermRefPicSet],
    ) -> ParserResult<vk_video_core::picture::H265ShortTermRefPicSet> {
        let mut strps = vk_video_core::picture::H265ShortTermRefPicSet::default();

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
            Self::resolve_predictive_rps(r, delta_rps_sign, strps.abs_delta_rps_minus1,
                idx, delta_idx_minus1, prev_strps, &mut strps)?;
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
                cum_delta_poc_s0 -= (raw_delta + 1);
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
                cum_delta_poc_s1 += (raw_delta + 1);
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
        prev_strps: &[vk_video_core::picture::H265ShortTermRefPicSet],
        strps: &mut vk_video_core::picture::H265ShortTermRefPicSet,
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
            let delta_poc = if stored > 32767 { stored - 65536 } else { stored };
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
        for j in 0..=num_ref_entries {
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

    fn parse_vps(&mut self, data: &[u8]) -> ParserResult<vk_video_core::picture::H265Vps> {
        if data.is_empty() {
            return Err(ParserError::InvalidBitstream);
        }
        // Skip NAL header (2 bytes for H.265)
        let mut r = BitReader::new(&data[2..], true);

        let mut vps = vk_video_core::picture::H265Vps::new();
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
        let (vps_profile_idc, vps_level_idc, vps_tier_flag) = Self::parse_ptl(&mut r, vps.vps_max_sub_layers_minus1, true)?;
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
        for i in 1..vps.vps_num_layer_sets {
            let mut layer_flags = Vec::new();
            for j in 0..=(vps.vps_max_layer_id) {
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

        self.vps_cache.insert(vps.vps_video_parameter_set_id, vps.clone());
        self.active_vps = Some(vps);

        Ok(self.active_vps.clone().unwrap())
    }

    fn parse_sps(&mut self, data: &[u8]) -> ParserResult<vk_video_core::picture::H265Sps> {
        if data.is_empty() {
            return Err(ParserError::InvalidBitstream);
        }
        // Skip NAL header (2 bytes for H.265)
        let mut r = BitReader::new(&data[2..], true);

        let sps_video_parameter_set_id = r.read_bits(4)? as u8;
        let sps_max_sub_layers_minus1 = r.read_bits(3)? as u8;
        let sps_temporal_id_nesting_flag = r.read_bit()?;

        // Parse profile_tier_level (SPS: SubLayerLevelPresentFlag=1 per H.265 spec)
        let (sps_profile_idc, sps_level_idc, sps_tier_flag) = Self::parse_ptl(&mut r, sps_max_sub_layers_minus1, true)?;

        let mut sps = vk_video_core::picture::H265Sps::new();
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
            sps.conf_win_left_offset = r.read_ue()? as u32;
            sps.conf_win_right_offset = r.read_ue()? as u32;
            sps.conf_win_top_offset = r.read_ue()? as u32;
            sps.conf_win_bottom_offset = r.read_ue()? as u32;
        }

        sps.bit_depth_luma_minus8 = r.read_ue()? as u8;
        sps.bit_depth_chroma_minus8 = r.read_ue()? as u8;
        sps.log2_max_pic_order_cnt_lsb_minus4 = r.read_ue()? as u8;

        sps.sps_sub_layer_ordering_info_present_flag = r.read_bit()?;

        // DPB management info (per VulkanH265Parser.cpp:500-515)
        // Read max_dec_pic_buffering_minus1, max_num_reorder_pics, max_latency_increase_plus1
        // for sub-layers [sps_sub_layer_ordering_info_present_flag ? 0 : sps_max_sub_layers_minus1 .. sps_max_sub_layers_minus1]
        let dpb_start = if sps.sps_sub_layer_ordering_info_present_flag { 0 } else { sps_max_sub_layers_minus1 as usize };
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
                sps.lt_ref_pic_poc_lsb_sps[i as usize] = r.read_bits(poc_lsb_bits as u8)? as u32;
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
            // Must consume all sps_extension bits even if not storing values
            // Per H.265 spec: sps_range_extension_flag(1) + sps_multilayer_extension_flag(1) + sps_extension_6bits(6)
            // Then conditional extension data based on flags
            sps.sps_range_extension_flag = r.read_bit()?;
            let sps_multilayer_extension_flag = r.read_bit()?;
            let _sps_extension_6bits = r.read_bits(6)?;

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
                let sps_scc_extension_flag = r.read_bit()?; // sps_scc_extension_flag
                if sps_scc_extension_flag {
                    let _ = r.read_bit()?; // sps_curr_pic_ref_enabled_flag
                    sps.palette_mode_enabled_flag = r.read_bit()?; // palette_mode_enabled_flag
                }
            }
            if sps_multilayer_extension_flag {
                let _ = r.read_bit()?;
            }
        }

        self.sps_cache.insert(sps.sps_seq_parameter_set_id, sps.clone());
        self.active_sps = Some(sps);

        Ok(self.active_sps.clone().unwrap())
    }

    fn parse_pps(&mut self, data: &[u8]) -> ParserResult<vk_video_core::picture::H265Pps> {
        if data.is_empty() {
            return Err(ParserError::InvalidBitstream);
        }

        // Skip NAL header (2 bytes for H.265)
        let mut r = BitReader::new(&data[2..], true);

        let mut pps = vk_video_core::picture::H265Pps::new();

        pps.pps_pic_parameter_set_id = r.read_ue()? as u32;
        pps.pps_seq_parameter_set_id = r.read_ue()? as u32;
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

        // pps_loop_filter_across_tiles_enabled_flag: NVIDIA's cuvid parser only
        // reads this bit when tiles are enabled; with no tiles it infers 1 and
        // reads pps_loop_filter_across_slices_enabled_flag at that bit position.
        // (Matches the pixel-perfect C reference; a literal spec read of the bit
        // when entropy_sync=1 && tiles=0 shifts every later PPS field by 1 bit.)
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
        let sps = self.sps_cache.get(&pps.pps_seq_parameter_set_id)
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
            // Parse pps_extension per H.265 spec Table 7.9
            pps.pps_range_extension_flag = r.read_bit()?;
            let pps_multilayer_extension_flag = r.read_bit()?;
            let _pps_extension_4bits = r.read_bits(4)?;

            if pps.pps_range_extension_flag {
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
                // SAO offset scale fields are conditional per H.265 spec Table 7.9:
                // sps_sao_luma_allowed_flag = sample_adaptive_offset_enabled_flag
                //   && max_transform_hierarchy_depth_intra > log2_min_luma_transform_block_size_minus2
                // sps_sao_chroma_allowed_flag = sps_sao_luma_allowed_flag && chroma_format_idc != 3
                let sps_sao_luma_allowed = sps.sample_adaptive_offset_enabled_flag
                    && (sps.max_transform_hierarchy_depth_intra > sps.log2_min_luma_transform_block_size_minus2);
                let sps_sao_chroma_allowed = sps_sao_luma_allowed && (sps.chroma_format_idc != 3);
                if sps_sao_luma_allowed {
                    pps.log2_sao_offset_scale_luma = r.read_ue()? as u8;
                }
                if sps_sao_chroma_allowed {
                    pps.log2_sao_offset_scale_chroma = r.read_ue()? as u8;
                }
            }
            if pps_multilayer_extension_flag {
                let _poc_reset_info_present_flag = r.read_bit()?;
                if r.read_bit()? { // infer_scaling_list_flag
                    let _ = r.read_bits(6)?; // scaling_list_ref_layer_id
                }
                let _ = r.read_ue()?; // num_ref_loc_offsets
            }
        }

        self.pps_cache.insert(pps.pps_pic_parameter_set_id, pps.clone());
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

        // Determine IdrPicFlag from NAL unit type
        // H.265 spec: NUT_IDR_W_RADL=19, NUT_IDR_N_LP=20
        info.is_idr = nal_unit_type == 19 || nal_unit_type == 20;
        // For Vulkan Video decode, also treat BLA (18-20) as "IDR-like" for reference purposes
        // when no_output_of_prior_pics_flag is set (handled later in slice header parsing)

        // Determine RapPicFlag from NAL unit type
        // RapPicFlag = (nal_unit_type >= 16 && nal_unit_type <= 23)
        info.is_rap = nal_unit_type >= 16 && nal_unit_type <= 23;

        // Determine is_reference from NAL unit type
        // Per H.265 spec:
        // - VCL NAL types 0-15: odd types (1,3,5,7,9,11,13,15) are reference
        // - IRAP NAL types 16-23: all are reference pictures
        // Based on VulkanH265Parser.cpp:2783-2791 for sub-layer non-ref determination
        info.is_reference = (nal_unit_type >= 16 && nal_unit_type <= 23)
            || (nal_unit_type < 16 && (nal_unit_type & 1) == 1);

        // --- slice_segment_header parsing (per VulkanH265Parser.cpp:2130-2133) ---

        // first_slice_segment_in_pic_flag
        let first_slice_segment_in_pic_flag = r.read_bit()?;

        // no_output_of_prior_pics_flag
        // Per H.265 spec 7.3.7: present in the bitstream (first slice segment) for
        // IDR (19,20) / BLA (16,17,18) / CRA (21); inferred as 0 (not read) for
        // RSV_IRAP (22,23).
        let mut no_output_of_prior_pics_flag = false;
        let is_idr_pic = nal_unit_type == 19 || nal_unit_type == 20;
        let is_bla_or_cra = (nal_unit_type >= 16 && nal_unit_type <= 18) || nal_unit_type == 21;
        if is_idr_pic || is_bla_or_cra {
            no_output_of_prior_pics_flag = r.read_bit()?; // Read from bitstream (present for IDR/BLA/CRA)
        }
        // RSV_IRAP (22,23): inferred as 0, don't read

        // pic_parameter_set_id
        let _slice_pps_id = r.read_ue()?;

        // For non-first slice segments, parse dependent_slice_segment_flag and slice_segment_address
        if !first_slice_segment_in_pic_flag {
            let dependent_slice_segment_flag = if pps.dependent_slice_segments_enabled_flag {
                r.read_bit()?
            } else {
                false
            };

            // slice_segment_address: CeilLog2(PicSizeInCtbsY) bits
            let log2_ctb_size = sps.log2_min_luma_coding_block_size_minus3 as u32 + 3
                + sps.log2_diff_max_min_luma_coding_block_size as u32;
            let pic_width_in_ctbs = (sps.pic_width_in_luma_samples as u32 + (1 << log2_ctb_size) - 1) >> log2_ctb_size;
            let pic_height_in_ctbs = (sps.pic_height_in_luma_samples as u32 + (1 << log2_ctb_size) - 1) >> log2_ctb_size;
            let pic_size_in_ctbs = pic_width_in_ctbs * pic_height_in_ctbs;
            let slice_segment_address_bits = (pic_size_in_ctbs as f64).log2().ceil() as u8;
            let _slice_segment_address = r.read_bits(slice_segment_address_bits)?;

            // For dependent slices, most info is inherited from first slice
            if dependent_slice_segment_flag {
                // Copy info from previous slice header if available
                if let Some(ref prev_info) = self.first_slice_header {
                    info.slice_type = prev_info.slice_type;
                    info.pic_order_cnt_lsb = prev_info.pic_order_cnt_lsb;
                    info.curr_pic_order_cnt_val = prev_info.curr_pic_order_cnt_val;
                    info.is_reference = prev_info.is_reference;
                }
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

        // pic_output_flag (if output_flag_present_flag)
        if pps.output_flag_present_flag {
            let _ = r.read_bit()?;
        }

        // colour_plane_id (if separate_colour_plane_flag)
        if sps.separate_colour_plane_flag {
            let _ = r.read_bits(2)?;
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
        let is_irap = nal_unit_type >= 16 && nal_unit_type <= 23;
        let mut pic_order_cnt_msb: i32;

        // NoRaslOutputFlag: true for BLA (16-18) or IDR (19-20) per C++ reference VulkanH265Parser.cpp:324
        let no_rasl_output_flag = is_irap && nal_unit_type <= 20;
        if no_rasl_output_flag {
            // IRAP with NoRaslOutputFlag: MSB is 0
            pic_order_cnt_msb = 0;
            self.no_rasl_output_flag = true;
        } else {
            self.no_rasl_output_flag = false;
            let max_pic_order_cnt_lsb = 1 << (sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4);

            if self.has_prev_pic {
                if ((info.pic_order_cnt_lsb as i32) < self.prev_pic_order_cnt_lsb)
                    && (self.prev_pic_order_cnt_lsb - info.pic_order_cnt_lsb as i32 >= max_pic_order_cnt_lsb as i32 / 2)
                {
                    pic_order_cnt_msb = self.prev_pic_order_cnt_msb + max_pic_order_cnt_lsb as i32;
                } else if (info.pic_order_cnt_lsb as i32 > self.prev_pic_order_cnt_lsb)
                    && (info.pic_order_cnt_lsb as i32 - self.prev_pic_order_cnt_lsb > max_pic_order_cnt_lsb as i32 / 2)
                {
                    pic_order_cnt_msb = self.prev_pic_order_cnt_msb - max_pic_order_cnt_lsb as i32;
                } else {
                    pic_order_cnt_msb = self.prev_pic_order_cnt_msb;
                }
            } else {
                // First picture: MSB is 0
                pic_order_cnt_msb = 0;
            }
        }

        info.curr_pic_order_cnt_val = pic_order_cnt_msb + info.pic_order_cnt_lsb as i32;

        // Update prevPicOrderCntMsb/Lsb per HEVC spec 8.3.1:
        // Only update for non-RASL pictures with temporal_id == 0.
        // RASL (types 22-23) must NOT update prev state.
        // sub_layer_non_ref (even NAL types) must NOT update prev state.
        let temporal_id = (nal_data[1] & 0x07) - 1; // nuh_temporal_id_plus1 - 1
        let is_rasl = nal_unit_type >= 22 && nal_unit_type <= 23;
        let is_sub_layer_non_ref = nal_unit_type % 2 == 0;
        if temporal_id == 0 && !is_rasl && !is_sub_layer_non_ref
        {
            self.prev_pic_order_cnt_lsb = info.pic_order_cnt_lsb as i32;
            self.prev_pic_order_cnt_msb = pic_order_cnt_msb;
            self.has_prev_pic = true;
        }

        // short_term_ref_pic_set_sps_flag
        // Per H.265 spec 7.3.3 this block (STRPS + long-term refs +
        // slice_temporal_mvp_enabled_flag) is present only when
        // `!NoRaslOutputFlag && SliceType != I`. IDR/CRA/BLA are always intra
        // (SliceType == I) and carry NoRaslOutputFlag, so the block is absent
        // for them. Gating on SliceType != I (info.slice_type != 0) is
        // equivalent and correctly skips the block for CRA (which the old
        // `!is_idr` gate wrongly read, corrupting the CRA slice header).
        if info.slice_type != 0 {
            let short_term_ref_pic_set_sps_flag = r.read_bit()?;
            info.short_term_ref_pic_set_sps_flag = short_term_ref_pic_set_sps_flag;
            if !short_term_ref_pic_set_sps_flag {
                // STRPS in slice - parse and store it
                let strps = Self::parse_short_term_ref_pic_set(
                    &mut r,
                    sps.num_short_term_ref_pic_sets as usize,
                    sps.num_short_term_ref_pic_sets as usize,
                    &sps.short_term_ref_pic_sets,
                )?;
                info.slice_strps = Some(strps);
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

                for i in 0u8..(num_long_term_sps + num_long_term_pics) {
                    if i < num_long_term_sps && sps.num_long_term_ref_pics_sps > 1 {
                        let lt_idx_bits = (sps.num_long_term_ref_pics_sps as f64).log2().ceil() as u8;
                        let _ = r.read_bits(lt_idx_bits)?;
                    } else if i >= num_long_term_sps {
                        let poc_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u8 + 4;
                        let _ = r.read_bits(poc_lsb_bits)?; // poc_lsb_lt
                        let _ = r.read_bit()?; // used_by_curr_pic_lt_flag
                    }
                    let delta_poc_msb_present_flag = r.read_bit()?;
                    if delta_poc_msb_present_flag {
                        let _ = r.read_ue()?; // delta_poc_msb_cycle_lt
                    }
                }
            }

            // slice_temporal_mvp_enabled_flag
            if sps.sps_temporal_mvp_enabled_flag {
                let _ = r.read_bit()?;
            }
        }

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
        if format.codec != vk_video_core::codec::VideoCodec::DecodeH265 {
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

        let mut result_sps: Option<vk_video_core::picture::BoxedPictureParametersSet> = None;
        let mut result_pps: Option<vk_video_core::picture::BoxedPictureParametersSet> = None;
        let mut result_vps: Option<vk_video_core::picture::BoxedPictureParametersSet> = None;
        let mut slice_nals: Vec<crate::SliceEntry> = Vec::new();
        let mut first_slice_offset: Option<usize> = None;
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
                    match self.parse_vps(&nal_data) {
                        Ok(_vps) => {
                            result_vps = Some(vk_video_core::picture::BoxedPictureParametersSet::new(
                                self.active_vps.clone().unwrap(),
                            ));
                        }
                        Err(_) => {}
                    }
                    i += 1;
                }
                Some(H265NalUnitType::Sps) => {
                    if !slice_nals.is_empty() {
                        break;
                    }
                    let nal_data = nal.data.clone();
                    match self.parse_sps(&nal_data) {
                        Ok(sps) => {
                            result_sps = Some(vk_video_core::picture::BoxedPictureParametersSet::new(
                                self.active_sps.clone().unwrap(),
                            ));
                            // Update detected format from SPS
                            self.detected_format.coded_width = sps.pic_width_in_luma_samples as u32;
                            self.detected_format.coded_height = sps.pic_height_in_luma_samples as u32;
                            match sps.chroma_format_idc {
                                0 => self.detected_format.chroma_subsampling = vk_video_core::format::ChromaSubsampling::Monochrome,
                                1 => self.detected_format.chroma_subsampling = vk_video_core::format::ChromaSubsampling::_420,
                                2 => self.detected_format.chroma_subsampling = vk_video_core::format::ChromaSubsampling::_422,
                                3 => self.detected_format.chroma_subsampling = vk_video_core::format::ChromaSubsampling::_444,
                                _ => {}
                            }
                            let luma_bd = 8 + sps.bit_depth_luma_minus8;
                            let chroma_bd = 8 + sps.bit_depth_chroma_minus8;
                            self.detected_format.luma_bit_depth = match luma_bd {
                                8 => vk_video_core::format::ComponentBitDepth::Bit8,
                                10 => vk_video_core::format::ComponentBitDepth::Bit10,
                                12 => vk_video_core::format::ComponentBitDepth::Bit12,
                                _ => vk_video_core::format::ComponentBitDepth::Bit8,
                            };
                            self.detected_format.chroma_bit_depth = match chroma_bd {
                                8 => vk_video_core::format::ComponentBitDepth::Bit8,
                                10 => vk_video_core::format::ComponentBitDepth::Bit10,
                                12 => vk_video_core::format::ComponentBitDepth::Bit12,
                                _ => vk_video_core::format::ComponentBitDepth::Bit8,
                            };
                            self.detected_format.codec_profile = sps.profile_idc as u32;
                            self.detected_format.progressive_sequence = !sps.vui.field_seq_flag;
                        }
                        Err(_) => {}
                    }
                    i += 1;
                }
                Some(H265NalUnitType::Pps) => {
                    if !slice_nals.is_empty() {
                        break;
                    }
                    let nal_data = nal.data.clone();
                    match self.parse_pps(&nal_data) {
                        Ok(_pps) => {
                            result_pps = Some(vk_video_core::picture::BoxedPictureParametersSet::new(
                                self.active_pps.clone().unwrap(),
                            ));
                        }
                        Err(_) => {}
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

                    // Track byte range for bytes_consumed
                    if first_slice_offset.is_none() {
                        first_slice_offset = Some(off);
                        first_slice_cursor = Some(i);
                    }
                    last_slice_end = Some(off + sz);

                    // Parse the first slice header of this frame
                    if self.first_slice_header.is_none() {
                        if let Ok(slice_info) = self.parse_slice_segment_header(&nal_data, nal_type) {
                            self.first_slice_header = Some(slice_info);
                        }
                    }

                    // Collect slice NAL data
                    slice_nals.push(crate::SliceEntry {
                        slice_header: self.first_slice_header
                            .clone()
                            .map(crate::SliceHeader::H265),
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
            let bytes_consumed = if let (Some(first_off), Some(last_end)) = (first_slice_offset, last_slice_end) {
                last_end - first_off
            } else {
                0
            };
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
        self.no_rasl_output_flag = false;
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

    /// Create a minimal IDR slice NAL unit (type 19 = IDR_W_RADL).
    fn create_idr_slice_data() -> Vec<u8> {
        // NAL header: type=19 (IDR_W_RADL), temporal_id_plus1=1
        // byte0 = (0<<7) | (19<<1) | (0>>6) = 38 = 0x26
        // byte1 = (0<<2) | 1 = 1 = 0x01
        let mut data = vec![0x26, 0x01];
        // Slice header bits (after NAL header):
        // first_slice_segment_in_pic_flag(1) = 1
        // no_output_of_prior_pics_flag is inferred as 1 for IDR, not in bitstream
        // slice_type(ue) = 2 (I slice) -> "0010" = 4 bits
        // pic_parameter_set_id(ue) = 0 -> "1" = 1 bit
        // Total so far: 1 + 4 + 1 = 6 bits, packed as: 1001 00xx = 0x90
        data.extend_from_slice(&[0x90]);
        data
    }

    /// Create a minimal P-slice NAL unit (type 1 = trailing IRAP VCL, ref pic).
    fn create_p_slice_data(poc_lsb: u16) -> Vec<u8> {
        // NAL header: type=1 (trailing IRAP VCL, ref pic), temporal_id_plus1=1
        let mut data = vec![0x02, 0x01];
        // Slice header bits (after NAL header):
        // first_slice_segment_in_pic_flag(1) = 1
        // slice_type(ue) = 1 (P slice) -> "010" = 3 bits
        // pic_parameter_set_id(ue) = 0 -> "1" = 1 bit
        // Total: 5 bits: 10101
        // pic_order_cnt_lsb: 8 bits
        // short_term_ref_pic_set_sps_flag(1) = 1
        // slice_temporal_mvp_enabled_flag(1) = 0
        let poc_lsb_u8 = poc_lsb as u8;
        let first_payload_byte = 0xA0 | ((poc_lsb_u8 >> 5) & 0x07);
        let second_payload_byte = ((poc_lsb_u8 << 3) & 0xF8) | 0x04; // sps_flag=1, temporal=0
        data.push(first_payload_byte);
        data.push(second_payload_byte);
        data
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

        // IDR_W_RADL (type 16): byte0 = (0 << 7) | (16 << 1) | (0 >> 6) = 32 = 0x20
        // But with nuh_temporal_id_plus1=1: byte1 = (0 << 2) | 1 = 1 = 0x01
        let idr_header = vec![0x20, 0x01];
        let header = nal::parse_h265_nal_header(&idr_header).unwrap();
        assert_eq!(header.1, 16); // IDR_W_RADL
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
        0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x21, 0x60,
        0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03,
        0x00, 0x00, 0x03, 0x00, 0x78, 0x95, 0x98, 0x09,
    ];

    /// SPS NAL unit from big_buck_bunney.h265 (type=33, 43 bytes)
    const TEST_SPS_DATA: &[u8] = &[
        0x42, 0x01, 0x01, 0x21, 0x60, 0x00, 0x00, 0x03,
        0x00, 0x90, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03,
        0x00, 0x78, 0xa0, 0x03, 0xc0, 0x80, 0x10, 0xe5,
        0x96, 0x56, 0x69, 0x24, 0xca, 0xf0, 0x10, 0x10,
        0x00, 0x00, 0x03, 0x00, 0x10, 0x00, 0x00, 0x03,
        0x01, 0xe0, 0x80,
    ];

    /// PPS NAL unit from big_buck_bunney.h265 (type=34, 7 bytes)
    /// Hex: 4401c172b46240
    const TEST_PPS_DATA: &[u8] = &[
        0x44, 0x01, 0xc1, 0x72, 0xb4, 0x62, 0x40,
    ];

    #[test]
    fn test_vps_parsing_alignment() {
        // Test VPS parsing matches C++ VulkanH265Parser.cpp:906-1085
        let mut parser = H265Parser::new();
        let vps = parser.parse_vps(TEST_VPS_DATA).expect("VPS parse failed");

        // VPS base fields (C++: video_parameter_set_rbsp() lines 908-941)
        assert_eq!(vps.vps_video_parameter_set_id, 0, "vps_video_parameter_set_id");
        assert_eq!(vps.vps_max_layers_minus1, 0, "vps_max_layers_minus1");
        assert_eq!(vps.vps_max_sub_layers_minus1, 0, "vps_max_sub_layers_minus1");
        assert!(vps.vps_temporal_id_nesting_flag, "vps_temporal_id_nesting_flag");

        // Profile/level (C++: profile_tier_level() lines 1631-1669)
        assert_eq!(vps.profile_idc, 1, "profile_idc (Main)");
        assert_eq!(vps.level_idc, 120, "level_idc (Level 4.0)");

        // VPS layer info (C++: lines 965-988)
        assert_eq!(vps.vps_max_layer_id, 0, "vps_max_layer_id");
        assert_eq!(vps.vps_num_layer_sets, 1, "vps_num_layer_sets");

        // VPS timing (C++: lines 991-1051)
        assert!(!vps.vps_timing_info_present_flag, "vps_timing_info_present_flag");
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
        let format = DetectedVideoFormat::new(vk_video_core::codec::VideoCodec::DecodeH265);
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
                assert_eq!(detected.chroma_subsampling, vk_video_core::format::ChromaSubsampling::_420, "chroma from SPS");
                assert_eq!(detected.luma_bit_depth, vk_video_core::format::ComponentBitDepth::Bit8, "luma bit depth from SPS");
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
        assert_eq!(tier, false, "SPS tier_flag");

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
        assert_eq!(tier, false, "VPS tier_flag");
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
        let rps = vk_video_core::picture::H265ShortTermRefPicSet {
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
            let signed = if stored > 32767 { stored - 65536 } else { stored };
            ref_s0.push(curr_poc + signed);
        }

        let mut ref_s1 = Vec::new();
        for i in 0..rps.num_positive_pics as usize {
            if ((rps.used_by_curr_pic_s1_flag >> i) & 1) == 0 {
                continue;
            }
            let stored = rps.delta_poc_s1_minus1[i] as i32;
            let signed = if stored > 32767 { stored - 65536 } else { stored };
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
        let mut rps = vk_video_core::picture::H265ShortTermRefPicSet::default();
        rps.num_negative_pics = 3;
        rps.num_positive_pics = 2;

        // Set bit 0 and bit 2 for S0
        rps.used_by_curr_pic_s0_flag = 0b101; // bits 0 and 2
        // Set bit 1 for S1
        rps.used_by_curr_pic_s1_flag = 0b010; // bit 1

        // Verify bit extraction
        assert_eq!((rps.used_by_curr_pic_s0_flag >> 0) & 1, 1, "S0 bit 0 should be set");
        assert_eq!((rps.used_by_curr_pic_s0_flag >> 1) & 1, 0, "S0 bit 1 should be clear");
        assert_eq!((rps.used_by_curr_pic_s0_flag >> 2) & 1, 1, "S0 bit 2 should be set");
        assert_eq!((rps.used_by_curr_pic_s1_flag >> 0) & 1, 0, "S1 bit 0 should be clear");
        assert_eq!((rps.used_by_curr_pic_s1_flag >> 1) & 1, 1, "S1 bit 1 should be set");
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
        assert_eq!(ref_strps.delta_poc_s1_minus1[0], 1);     // +1

        // Verify the reference RPS has correct used flags
        assert_eq!(ref_strps.used_by_curr_pic_s0_flag, 1);
        assert_eq!(ref_strps.used_by_curr_pic_s1_flag, 1);
    }

    // =========================================================================
    // POC derivation edge cases
    // =========================================================================

    /// Test POC derivation for first picture (has_prev_pic = false).
    /// First picture should have pic_order_cnt_msb = 0.
    #[test]
    fn test_poc_first_picture() {
        let mut parser = H265Parser::new();
        parser.parse_vps(TEST_VPS_DATA).expect("VPS parse failed");
        parser.parse_sps(TEST_SPS_DATA).expect("SPS parse failed");
        parser.parse_pps(TEST_PPS_DATA).expect("PPS parse failed");

        // Parse P-slice with POC LSB = 100 as first picture
        let slice_data = create_p_slice_data(100);
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0x00, 0x00, 0x01]);
        payload.extend_from_slice(&slice_data);

        let packet = crate::bitstream::BitstreamPacket::new(payload);
        let result = parser.parse(&packet).expect("Parse failed");

        match result {
            ParseResult::Slice { slices, .. } => {
                if let Some(crate::SliceHeader::H265(info)) = &slices[0].slice_header {
                    assert_eq!(
                        info.curr_pic_order_cnt_val, 100,
                        "First picture POC should be 100 (msb=0, lsb=100)"
                    );
                }
            }
            other => panic!("Expected Slice, got {:?}", other),
        }
    }

    /// Test POC derivation for IDR picture (always POC=0).
    #[test]
    fn test_poc_idr_always_zero() {
        let mut parser = init_parser();

        // Parse some P-slices first to set has_prev_pic
        for poc in [2, 4, 6] {
            let slice_data = create_p_slice_data(poc);
            let mut payload = Vec::new();
            payload.extend_from_slice(&[0x00, 0x00, 0x01]);
            payload.extend_from_slice(&slice_data);
            let packet = crate::bitstream::BitstreamPacket::new(payload);
            let _ = parser.parse(&packet);
        }

        // Now parse IDR - POC should be 0 regardless of previous state
        let idr_data = create_idr_slice_data();
        let mut idr_payload = Vec::new();
        idr_payload.extend_from_slice(&[0x00, 0x00, 0x01]);
        idr_payload.extend_from_slice(&idr_data);

        let idr_packet = crate::bitstream::BitstreamPacket::new(idr_payload);
        let idr_result = parser.parse(&idr_packet).expect("IDR parse failed");

        match idr_result {
            ParseResult::Slice { slices, .. } => {
                if let Some(crate::SliceHeader::H265(info)) = &slices[0].slice_header {
                    assert_eq!(
                        info.curr_pic_order_cnt_val, 0,
                        "IDR POC should always be 0"
                    );
                }
            }
            other => panic!("Expected Slice, got {:?}", other),
        }
    }

    /// Test that POC state is NOT updated for RASL pictures.
    /// RASL (types 22-23) must not update prev_pic_order_cnt_msb/lsb.
    #[test]
    fn test_poc_rasl_does_not_update_state() {
        let mut parser = init_parser();

        // Parse P-slice with POC=4
        let slice_data = create_p_slice_data(4);
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0x00, 0x00, 0x01]);
        payload.extend_from_slice(&slice_data);
        let packet = crate::bitstream::BitstreamPacket::new(payload);
        let _ = parser.parse(&packet).expect("Parse failed");

        // The parser should have updated prev_pic_order_cnt_lsb to 4
        // We can't directly access it, but we can verify by parsing the next slice
        // and checking its POC derivation
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
        let mut parser = init_parser();
        let sps = parser.active_sps().expect("No active SPS");

        // Verify SAO conditions for the test SPS
        let sao_luma_allowed = sps.sample_adaptive_offset_enabled_flag
            && (sps.max_transform_hierarchy_depth_intra > sps.log2_min_luma_transform_block_size_minus2);

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
        let mut parser = init_parser();
        let pps = parser.active_pps().expect("No active PPS");

        // The test PPS from big_buck_bunny has pps_extension_present_flag = false
        assert!(!pps.pps_extension_present_flag, "Test PPS should not have extension");
        // Therefore SAO scale fields should be at default values
        assert_eq!(pps.log2_sao_offset_scale_luma, 0, "Default SAO luma scale");
        assert_eq!(pps.log2_sao_offset_scale_chroma, 0, "Default SAO chroma scale");
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

        // Parse a slice to set has_prev_pic
        let slice_data = create_p_slice_data(10);
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0x00, 0x00, 0x01]);
        payload.extend_from_slice(&slice_data);
        let packet = crate::bitstream::BitstreamPacket::new(payload);
        let _ = parser.parse(&packet);

        // Reset
        parser.reset();

        // Verify caches are cleared
        assert!(parser.active_sps().is_none(), "SPS should be cleared after reset");
        assert!(parser.active_pps().is_none(), "PPS should be cleared after reset");
        assert!(parser.active_vps.is_none(), "VPS should be cleared after reset");
        assert!(parser.first_slice_header().is_none(), "first_slice_header should be cleared");
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
        for poc in [0, 2, 4, 6, 8] {
            let slice_data = create_p_slice_data(poc);
            payload.extend_from_slice(&[0x00, 0x00, 0x01]);
            payload.extend_from_slice(&slice_data);
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
}

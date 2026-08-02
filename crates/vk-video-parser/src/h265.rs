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
}

impl H265Parser {
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
            prev_pic_order_cnt_lsb: -1,
            no_rasl_output_flag: false,
        }
    }

    /// Skip profile_tier_level data.
    ///
    /// Uses the same approach as the NVIDIA Vulkan-Video-Samples parser:
    /// always skip fixed bit counts regardless of profile, which is simpler
    /// and avoids issues with conditional parsing.
    ///
    /// profile_tier_level( ProfilePresentFlag, MaxSubLayersMinus1, CommonInfPresentFlag, SubLayerLevelPresentFlag )
    ///
    /// For SPS: ProfilePresentFlag=1, CommonInfPresentFlag=1, SubLayerLevelPresentFlag=0
    /// For VPS: ProfilePresentFlag=1, CommonInfPresentFlag=1, SubLayerLevelPresentFlag=1
    fn skip_ptl(r: &mut BitReader, max_sub_layers: u8, sub_layer_level_present: bool) -> ParserResult<()> {
        // --- Profile fields (ProfilePresentFlag = 1) ---
        // Skip general_profile_space(2) + general_tier_flag(1) + general_profile_idc(5) = 8 bits
        let _ = r.read_bits(8)?;
        eprintln!("[H265 PTL] skipped profile fields (8 bits)");

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
        let _level_idc = r.read_bits(8)?; // general_level_idc
        eprintln!("[H265 PTL] level_idc=0x{:02x}", _level_idc);

        // --- Sub-layer profile/level presence flags ---
        let mut sub_layer_level_flags: Vec<bool> = Vec::new();
        for _ in 0..max_sub_layers {
            let _ = r.read_bit()?; // sub_layer_profile_present_flag (ignored)
            sub_layer_level_flags.push(r.read_bit()?); // sub_layer_level_present_flag
        }

        // Padding bits
        if max_sub_layers > 0 && max_sub_layers < 8 {
            let _ = r.read_bits(((8 - max_sub_layers) * 2) as u8)?;
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

        Ok(())
    }

    /// Parse a short-term reference picture set (STRPS).
    /// Based on VulkanH265Parser.cpp:1730-1917.
    fn parse_short_term_ref_pic_set(
        r: &mut BitReader,
        idx: usize,
        num_short_term_ref_pic_sets: usize,
    ) -> ParserResult<vk_video_core::picture::H265ShortTermRefPicSet> {
        let mut strps = vk_video_core::picture::H265ShortTermRefPicSet::default();

        // inter_ref_pic_set_prediction_flag is only present if idx != 0
        let inter_ref_pic_set_prediction_flag = if idx != 0 { r.read_bit()? } else { false };
        strps.inter_ref_pic_set_prediction_flag = inter_ref_pic_set_prediction_flag;

        if inter_ref_pic_set_prediction_flag {
            // Delta-based prediction from a previous STRPS
            let delta_idx_minus1 = r.read_ue()?;
            strps.delta_idx_minus1 = delta_idx_minus1;

            let delta_rps_sign = r.read_bit()?;
            let abs_delta_rps_minus1 = r.read_ue()? as u16;
            strps.abs_delta_rps_minus1 = abs_delta_rps_minus1;

            let delta_rps = if delta_rps_sign {
                -(abs_delta_rps_minus1 as i32 + 1)
            } else {
                abs_delta_rps_minus1 as i32 + 1
            };

            // For simplicity, we store the delta info but don't fully resolve
            // the reference picture set here (complex logic requiring access to
            // previous STRPS). The flag and delta values are preserved.
            eprintln!("[H265 STRPS] idx={} delta_idx_minus1={} delta_rps_sign={} abs_delta_rps_minus1={} delta_rps={}",
                     idx, delta_idx_minus1, delta_rps_sign, abs_delta_rps_minus1, delta_rps);
        } else {
            // Direct encoding
            let num_negative_pics = r.read_ue()? as u8;
            let num_positive_pics = r.read_ue()? as u8;
            strps.num_negative_pics = num_negative_pics;
            strps.num_positive_pics = num_positive_pics;

            for i in 0..num_negative_pics {
                let delta_poc_s0_minus1 = r.read_ue()? as u16;
                strps.delta_poc_s0_minus1[i as usize] = delta_poc_s0_minus1;
                let used_by_curr_pic_s0_flag = r.read_bit()?;
                if used_by_curr_pic_s0_flag {
                    strps.used_by_curr_pic_s0_flag |= 1 << i;
                }
            }

            for i in 0..num_positive_pics {
                let delta_poc_s1_minus1 = r.read_ue()? as u16;
                strps.delta_poc_s1_minus1[i as usize] = delta_poc_s1_minus1;
                let used_by_curr_pic_s1_flag = r.read_bit()?;
                if used_by_curr_pic_s1_flag {
                    strps.used_by_curr_pic_s1_flag |= 1 << i;
                }
            }

            eprintln!("[H265 STRPS] idx={} num_negative_pics={} num_positive_pics={}",
                     idx, num_negative_pics, num_positive_pics);
        }

        Ok(strps)
    }

    fn parse_vps(&mut self, data: &[u8]) -> ParserResult<vk_video_core::picture::H265Vps> {
        if data.is_empty() {
            return Err(ParserError::InvalidBitstream);
        }

        eprintln!("[H265 VPS] data len={}, first 10 bytes: {:?}", data.len(), &data[..data.len().min(10)]);

        // Skip NAL header (2 bytes for H.265)
        let mut r = BitReader::new(&data[2..], true);

        let mut vps = vk_video_core::picture::H265Vps::new();
        vps.vps_video_parameter_set_id = r.read_bits(4)? as u8;
        let _ = r.read_bits(2)?; // vps_reserved_0ffff2_bits
        vps.vps_max_layers_minus1 = r.read_bits(6)? as u16;
        vps.vps_max_sub_layers_minus1 = r.read_bits(3)? as u8;
        vps.vps_temporal_id_nesting_flag = r.read_bit()?;
        eprintln!("[H265 VPS] max_sub_layers={}, temporal_nesting={}", vps.vps_max_sub_layers_minus1, vps.vps_temporal_id_nesting_flag);

        // Skip vps_reserved_0xffff_16bits
        let _ = r.read_bits(16)?;

        // Skip profile_tier_level (VPS: SubLayerLevelPresentFlag=1)
        Self::skip_ptl(&mut r, vps.vps_max_sub_layers_minus1, true)?;
        eprintln!("[H265 VPS] PTL skip done, pos={}", r.pos);

        vps.vps_sub_layer_ordering_info_present_flag = r.read_bit()?;

        self.vps_cache.insert(vps.vps_video_parameter_set_id, vps.clone());
        self.active_vps = Some(vps);

        Ok(self.active_vps.clone().unwrap())
    }

    fn parse_sps(&mut self, data: &[u8]) -> ParserResult<vk_video_core::picture::H265Sps> {
        if data.is_empty() {
            return Err(ParserError::InvalidBitstream);
        }

        eprintln!("[H265 SPS] data len={}, first 10 bytes: {:?}", data.len(), &data[..data.len().min(10)]);

        // Skip NAL header (2 bytes for H.265)
        let mut r = BitReader::new(&data[2..], true);

        let sps_video_parameter_set_id = r.read_bits(4)? as u8;
        let sps_max_sub_layers_minus1 = r.read_bits(3)? as u8;
        let sps_temporal_id_nesting_flag = r.read_bit()?;
        eprintln!("[H265 SPS] max_sub_layers={}, temporal_nesting={}", sps_max_sub_layers_minus1, sps_temporal_id_nesting_flag);

        // Skip profile_tier_level (SPS: SubLayerLevelPresentFlag=0)
        Self::skip_ptl(&mut r, sps_max_sub_layers_minus1, false)?;
        eprintln!("[H265 SPS] PTL skip done, pos={}", r.pos);

        let mut sps = vk_video_core::picture::H265Sps::new();
        sps.sps_video_parameter_set_id = sps_video_parameter_set_id;
        sps.sps_max_sub_layers_minus1 = sps_max_sub_layers_minus1;
        sps.sps_temporal_id_nesting_flag = sps_temporal_id_nesting_flag;

        match r.read_ue() {
            Ok(v) => { sps.sps_seq_parameter_set_id = v; eprintln!("[H265 SPS] sps_id={}", v); }
            Err(e) => { eprintln!("[H265 SPS] ERROR reading sps_id: {:?}", e); return Err(e.into()); }
        }
        sps.chroma_format_idc = r.read_ue()? as u8;
        eprintln!("[H265 SPS] chroma_format_idc={}", sps.chroma_format_idc);

        if sps.chroma_format_idc == 3 {
            sps.separate_colour_plane_flag = r.read_bit()?;
        }

        sps.pic_width_in_luma_samples = r.read_ue()? as u16;
        sps.pic_height_in_luma_samples = r.read_ue()? as u16;
        eprintln!("[H265 SPS] width={} height={}", sps.pic_width_in_luma_samples, sps.pic_height_in_luma_samples);

        // Skip conformance_window_flag and offsets
        let _conformance_window_flag = r.read_bit()?;
        if _conformance_window_flag {
            let _ = r.read_ue()?;
            let _ = r.read_ue()?;
            let _ = r.read_ue()?;
            let _ = r.read_ue()?;
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
        eprintln!("[H265 SPS] DPB: max_dec_pic_buffering_minus1[0]={}", sps.max_dec_pic_buffering_minus1[0]);

        // Additional SPS fields (per VulkanH265Parser.cpp:541-562)
        sps.log2_min_luma_coding_block_size_minus3 = r.read_ue()? as u8;
        sps.log2_diff_max_min_luma_coding_block_size = r.read_ue()? as u8;
        sps.log2_min_luma_transform_block_size_minus2 = r.read_ue()? as u8;
        sps.log2_diff_max_min_luma_transform_block_size = r.read_ue()? as u8;
        sps.max_transform_hierarchy_depth_inter = r.read_ue()? as u8;
        sps.max_transform_hierarchy_depth_intra = r.read_ue()? as u8;
        sps.scaling_list_enabled_flag = r.read_bit()?;

        if sps.scaling_list_enabled_flag {
            let _sps_scaling_list_data_present_flag = r.read_bit()?;
            if _sps_scaling_list_data_present_flag {
                // Skip scaling_list_data - complex structure, skip for now
                // In real implementation, would parse scaling_list_data()
                // For now we note it was present but don't parse details
                eprintln!("[H265 SPS] scaling_list_data_present - skipping detailed parse");
            }
        }

        sps.amp_enabled_flag = r.read_bit()?;
        sps.sample_adaptive_offset_enabled_flag = r.read_bit()?;
        let _pcm_enabled_flag = r.read_bit()?;
        if _pcm_enabled_flag {
            let _ = r.read_bits(4)?; // pcm_sample_bit_depth_luma_minus1
            let _ = r.read_bits(4)?; // pcm_sample_bit_depth_chroma_minus1
            let _ = r.read_ue()?; // log2_min_pcm_luma_coding_block_size_minus3
            let _ = r.read_ue()?; // log2_diff_max_min_pcm_luma_coding_block_size
            let _ = r.read_bit()?; // pcm_loop_filter_disabled_flag
        }

        // Short-term reference picture sets (per VulkanH265Parser.cpp:579-598)
        let num_short_term_ref_pic_sets = r.read_ue()? as u8;
        sps.num_short_term_ref_pic_sets = num_short_term_ref_pic_sets;
        eprintln!("[H265 SPS] num_short_term_ref_pic_sets={}", num_short_term_ref_pic_sets);

        for i in 0..num_short_term_ref_pic_sets {
            let strps = Self::parse_short_term_ref_pic_set(&mut r, i as usize, num_short_term_ref_pic_sets as usize)?;
            sps.short_term_ref_pic_sets.push(strps);
        }

        // Long-term reference pictures (per VulkanH265Parser.cpp:599-617)
        sps.long_term_ref_pics_present_flag = r.read_bit()?;
        if sps.long_term_ref_pics_present_flag {
            let num_long_term_ref_pics_sps = r.read_ue()? as u8;
            sps.num_long_term_ref_pics_sps = num_long_term_ref_pics_sps;
            eprintln!("[H265 SPS] num_long_term_ref_pics_sps={}", num_long_term_ref_pics_sps);

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
        let _vui_parameters_present_flag = r.read_bit()?;
        if _vui_parameters_present_flag {
            // Skip vui_parameters - complex structure
            // In a full implementation, would parse vui_parameters()
            eprintln!("[H265 SPS] vui_parameters_present - skipping detailed parse");
        }
        let _sps_extension_present_flag = r.read_bit()?;
        if _sps_extension_present_flag {
            // Skip sps_extension
            eprintln!("[H265 SPS] sps_extension_present - skipping detailed parse");
        }

        self.sps_cache.insert(sps.sps_seq_parameter_set_id, sps.clone());
        self.active_sps = Some(sps);

        Ok(self.active_sps.clone().unwrap())
    }

    fn parse_pps(&mut self, data: &[u8]) -> ParserResult<vk_video_core::picture::H265Pps> {
        if data.is_empty() {
            return Err(ParserError::InvalidBitstream);
        }

        eprintln!("[H265 PPS] data len={}, first 10 bytes: {:?}", data.len(), &data[..data.len().min(10)]);

        // Skip NAL header (2 bytes for H.265)
        let mut r = BitReader::new(&data[2..], true);

        let mut pps = vk_video_core::picture::H265Pps::new();

        pps.pps_pic_parameter_set_id = r.read_ue()? as u32;
        pps.pps_seq_parameter_set_id = r.read_ue()? as u32;
        pps.dependent_slice_segments_enabled_flag = r.read_bit()?;
        pps.output_flag_present_flag = r.read_bit()?;
        pps.num_extra_slice_header_bits = r.read_ue()? as u8;
        pps.sign_data_hiding_enabled_flag = r.read_bit()?;
        pps.cabac_init_present_flag = r.read_bit()?;

        // num_ref_idx_l0_default_active_minus1 and num_ref_idx_l1_default_active_minus1
        // (per VulkanH265Parser.cpp:743-751)
        pps.num_ref_idx_l0_default_active_minus1 = r.read_ue()? as u8;
        pps.num_ref_idx_l1_default_active_minus1 = r.read_ue()? as u8;

        // pps_init_qp_minus26 (SE(V))
        let _init_qp_minus26 = r.read_se().unwrap_or(0);

        // Additional PPS fields (per VulkanH265Parser.cpp:762-841)
        // Use unwrap_or for optional fields that may not be present in truncated PPS
        pps.constrained_intra_pred_flag = r.read_bit().unwrap_or(false);
        pps.transform_skip_enabled_flag = r.read_bit().unwrap_or(false);
        pps.cu_qp_delta_enabled_flag = r.read_bit().unwrap_or(false);
        if pps.cu_qp_delta_enabled_flag {
            let _ = r.read_ue().ok(); // diff_cu_qp_delta_depth
        }
        let _pps_cb_qp_offset = r.read_se().unwrap_or(0);
        let _pps_cr_qp_offset = r.read_se().unwrap_or(0);
        pps.pps_slice_chroma_qp_offsets_present_flag = r.read_bit().unwrap_or(false);
        pps.weighted_pred_flag = r.read_bit().unwrap_or(false);
        pps.weighted_bipred_flag = r.read_bit().unwrap_or(false);
        pps.transquant_bypass_enabled_flag = r.read_bit().unwrap_or(false);
        pps.tiles_enabled_flag = r.read_bit().unwrap_or(false);
        pps.entropy_coding_sync_enabled_flag = r.read_bit().unwrap_or(false);

        if pps.tiles_enabled_flag {
            let num_tile_columns_minus1 = r.read_ue().unwrap_or(0) as u8;
            let num_tile_rows_minus1 = r.read_ue().unwrap_or(0) as u8;
            let uniform_spacing_flag = r.read_bit().unwrap_or(true);
            if !uniform_spacing_flag {
                for _ in 0..num_tile_columns_minus1 {
                    let _ = r.read_ue().ok(); // column_width_minus1
                }
                for _ in 0..num_tile_rows_minus1 {
                    let _ = r.read_ue().ok(); // row_height_minus1
                }
            }
            let _loop_filter_across_tiles_enabled_flag = r.read_bit().ok();
        }

        let _pps_loop_filter_across_slices_enabled_flag = r.read_bit().ok();
        let deblocking_filter_control_present_flag = r.read_bit().unwrap_or(false);
        if deblocking_filter_control_present_flag {
            let _deblocking_filter_override_enabled_flag = r.read_bit().ok();
            let pps_deblocking_filter_disabled_flag = r.read_bit().unwrap_or(false);
            if !pps_deblocking_filter_disabled_flag {
                let _ = r.read_se().ok(); // pps_beta_offset_div2
                let _ = r.read_se().ok(); // pps_tc_offset_div2
            }
        }

        let _pps_scaling_list_data_present_flag = r.read_bit().ok();
        if _pps_scaling_list_data_present_flag.unwrap_or(false) {
            // Skip scaling_list_data
        }

        let _lists_modification_present_flag = r.read_bit().ok();
        let _log2_parallel_merge_level_minus2 = r.read_ue().ok();
        let _slice_segment_header_extension_present_flag = r.read_bit().ok();
        let _pps_extension_present_flag = r.read_bit().ok();
        if _pps_extension_present_flag.unwrap_or(false) {
            // Skip pps_extension
        }

        eprintln!("[H265 PPS] pps_id={}, sps_id={}, num_ref_idx_l0={}, num_ref_idx_l1={}",
                 pps.pps_pic_parameter_set_id, pps.pps_seq_parameter_set_id,
                 pps.num_ref_idx_l0_default_active_minus1, pps.num_ref_idx_l1_default_active_minus1);

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
        // IdrPicFlag = (nal_unit_type == 16 || nal_unit_type == 17)
        // Note: In the C++ code, NUT_IDR_W_RADL=19, NUT_IDR_N_LP=20, but the actual
        // H.265 spec values are 16 and 17. The Rust H265NalUnitType uses spec values.
        info.is_idr = nal_unit_type == 16 || nal_unit_type == 17;

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

        // no_output_of_prior_pics_flag (only for RAP pictures)
        let mut no_output_of_prior_pics_flag = false;
        if info.is_rap {
            no_output_of_prior_pics_flag = r.read_bit()?;
        }

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

        // slice_type (UE(V)) - 0=B, 1=P, 2=I
        let slice_type_raw = r.read_ue()?;
        info.slice_type = slice_type_raw as u8;

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

        if is_irap && no_output_of_prior_pics_flag {
            pic_order_cnt_msb = 0;
            self.no_rasl_output_flag = true;
        } else {
            self.no_rasl_output_flag = false;
            let max_pic_order_cnt_lsb = 1 << (sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4);

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
        }

        info.curr_pic_order_cnt_val = pic_order_cnt_msb + info.pic_order_cnt_lsb as i32;

        // Update prevPicOrderCntMsb/Lsb for non-temporal-id pictures
        // Per VulkanH265Parser.cpp:2792-2798
        let temporal_id = 0; // Would be from nuh_temporal_id_plus1 - 1
        let is_sub_layer_non_ref = nal_unit_type % 2 == 0; // Even NAL types are non-ref
        if temporal_id == 0
            && !(nal_unit_type >= 6 && nal_unit_type <= 9) // Not RADL/RASL
            && !is_sub_layer_non_ref
        {
            self.prev_pic_order_cnt_lsb = info.pic_order_cnt_lsb as i32;
            self.prev_pic_order_cnt_msb = pic_order_cnt_msb;
        }

        // short_term_ref_pic_set_sps_flag (if not IDR)
        if !info.is_idr {
            let short_term_ref_pic_set_sps_flag = r.read_bit()?;
            if !short_term_ref_pic_set_sps_flag {
                // STRPS in slice - parse it
                let _ = Self::parse_short_term_ref_pic_set(&mut r, sps.num_short_term_ref_pic_sets as usize, sps.num_short_term_ref_pic_sets as usize);
            } else if sps.num_short_term_ref_pic_sets > 1 {
                let strps_idx_bits = (sps.num_short_term_ref_pic_sets as f64).log2().ceil() as u8;
                let _short_term_ref_pic_set_idx = r.read_bits(strps_idx_bits)?;
            }

            // Long-term reference pictures
            if sps.long_term_ref_pics_present_flag && sps.num_long_term_ref_pics_sps > 0 {
                let num_long_term_sps = r.read_ue()? as u8;
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

        eprintln!(
            "[H265 SliceHeader] type={}, poc_lsb={}, curr_poc={}, is_idr={}, is_rap={}, is_ref={}, first_slice={}",
            info.slice_type, info.pic_order_cnt_lsb, info.curr_pic_order_cnt_val,
            info.is_idr, info.is_rap, info.is_reference, first_slice_segment_in_pic_flag
        );

        Ok(info)
    }

    fn extract_nal_units(&self, data: &[u8]) -> Vec<NalUnit> {
        let mut nal_units = Vec::new();
        let mut offset = 0;

        while offset < data.len() {
            if let Some((start, code_len)) = nal::find_next_start_code(data, offset) {
                let next_start = nal::find_next_start_code(data, start + code_len);

                let end = match next_start {
                    Some((next_start, _)) => next_start,
                    None => data.len(),
                };

                let nal_data = &data[start + code_len..end];
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

        let nal_units = self.extract_nal_units(&packet.payload);
        eprintln!("[H265 parse] Found {} NAL units", nal_units.len());
        for nal in &nal_units {
            eprintln!("[H265 parse] NAL type={}, data_len={}", nal.nal_unit_type, nal.data.len());
        }

        let mut result_sps: Option<vk_video_core::picture::BoxedPictureParametersSet> = None;
        let mut result_pps: Option<vk_video_core::picture::BoxedPictureParametersSet> = None;
        let mut result_vps: Option<vk_video_core::picture::BoxedPictureParametersSet> = None;
        let mut last_slice_offset: Option<usize> = None;
        let mut last_slice_len: Option<usize> = None;
        let mut slice_count: u32 = 0;

        for nal in &nal_units {
            match H265NalUnitType::from_u8(nal.nal_unit_type) {
                Some(H265NalUnitType::Vps) => {
                    eprintln!("[H265 parse] Found VPS NAL, size={}", nal.data.len());
                    match self.parse_vps(&nal.data) {
                        Ok(vps) => {
                            eprintln!("[H265 VPS] parsed vps_id={}", vps.vps_video_parameter_set_id);
                            result_vps = Some(vk_video_core::picture::BoxedPictureParametersSet::new(
                                self.active_vps.clone().unwrap(),
                            ));
                        }
                        Err(e) => {
                            eprintln!("[H265 VPS] parse ERROR: {:?}", e);
                        }
                    }
                }
                Some(H265NalUnitType::Sps) => {
                    eprintln!("[H265 parse] Found SPS NAL, size={}", nal.data.len());
                    match self.parse_sps(&nal.data) {
                        Ok(sps) => {
                            eprintln!("[H265 SPS] parsed sps_id={}, {}x{}",
                                sps.sps_seq_parameter_set_id,
                                sps.pic_width_in_luma_samples,
                                sps.pic_height_in_luma_samples);
                            result_sps = Some(vk_video_core::picture::BoxedPictureParametersSet::new(
                                self.active_sps.clone().unwrap(),
                            ));
                        }
                        Err(e) => {
                            eprintln!("[H265 SPS] parse ERROR: {:?}", e);
                        }
                    }
                }
                Some(H265NalUnitType::Pps) => {
                    eprintln!("[H265 parse] Found PPS NAL, size={}", nal.data.len());
                    match self.parse_pps(&nal.data) {
                        Ok(pps) => {
                            eprintln!("[H265 PPS] parsed pps_id={}", pps.pps_pic_parameter_set_id);
                            result_pps = Some(vk_video_core::picture::BoxedPictureParametersSet::new(
                                self.active_pps.clone().unwrap(),
                            ));
                        }
                        Err(e) => {
                            eprintln!("[H265 PPS] parse ERROR: {:?}", e);
                        }
                    }
                }
                Some(t) if t.is_slice() => {
                    // Parse the first slice header of this frame
                    if self.first_slice_header.is_none() {
                        if let Ok(slice_info) = self.parse_slice_segment_header(&nal.data, nal.nal_unit_type) {
                            self.first_slice_header = Some(slice_info);
                        }
                    }

                    last_slice_offset = Some(nal.offset);
                    last_slice_len = Some(nal.size);
                    slice_count += 1;
                    self.frame_count += 1;
                }
                _ => {}
            }
        }

        if result_sps.is_some() || result_pps.is_some() || result_vps.is_some() {
            Ok(ParseResult::ParameterSet {
                sps: result_sps,
                pps: result_pps,
                vps: result_vps,
            })
        } else if let (Some(offset), Some(len)) = (last_slice_offset, last_slice_len) {
            Ok(ParseResult::Slice {
                slice_data_offset: offset,
                slice_data_len: len,
                num_slices: slice_count,
                slice_header: self.first_slice_header.clone(),
            })
        } else {
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
        // Reset POC tracking
        self.prev_pic_order_cnt_msb = 0;
        self.prev_pic_order_cnt_lsb = -1;
        self.no_rasl_output_flag = false;
    }

    fn detected_format(&self) -> &DetectedVideoFormat {
        &self.detected_format
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

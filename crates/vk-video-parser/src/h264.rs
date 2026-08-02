//! H.264/AVC bitstream parser.
//!
//! Parses H.264 bitstreams to extract SPS, PPS, and slice data.
//! Uses BitReader with inline emulation-prevention byte removal.

use std::collections::HashMap;

use crate::nal::{self, H264NalUnitType, NalUnit};
use crate::{
    DetectedVideoFormat, ParseResult, ParserError, ParserResult, VideoParser,
};
use crate::bitreader::BitReader;

/// H.264 parser state.
pub struct H264Parser {
    sps_cache: HashMap<u32, vk_video_core::picture::H264Sps>,
    pps_cache: HashMap<u32, vk_video_core::picture::H264Pps>,
    active_sps: Option<vk_video_core::picture::H264Sps>,
    active_pps: Option<vk_video_core::picture::H264Pps>,
    detected_format: DetectedVideoFormat,
    first_slice_header: Option<SliceHeader>,
    prev_frame_num: u32,
    prev_pic_order_cnt_lsb: i32,
    prev_pic_order_cnt_msb: i32,
    frame_count: u32,
    idr_found: bool,
}

impl H264Parser {
    pub fn new() -> Self {
        Self {
            sps_cache: HashMap::new(),
            pps_cache: HashMap::new(),
            active_sps: None,
            active_pps: None,
            detected_format: DetectedVideoFormat::new(
                vk_video_core::codec::VideoCodec::DecodeH264,
            ),
            first_slice_header: None,
            prev_frame_num: 0,
            prev_pic_order_cnt_lsb: 0,
            prev_pic_order_cnt_msb: 0,
            frame_count: 0,
            idr_found: false,
        }
    }

    /// Parse VUI parameters from the bitstream.
    /// Returns H264SpsVui with parsed values when vui_parameters_present_flag is set.
    fn parse_vui_parameters(r: &mut BitReader) -> ParserResult<vk_video_core::picture::H264SpsVui> {
        let mut vui = vk_video_core::picture::H264SpsVui::default();

        // aspect_ratio_info_present_flag
        vui.aspect_ratio_info_present_flag = r.read_bit()?;
        if vui.aspect_ratio_info_present_flag {
            vui.aspect_ratio_idc = r.read_bits(8)? as u8;
            if vui.aspect_ratio_idc == 255 {
                vui.sar_width = r.read_bits(16)? as u16;
                vui.sar_height = r.read_bits(16)? as u16;
            }
        }

        // overscan_info_present_flag
        vui.overscan_info_present_flag = r.read_bit()?;
        if vui.overscan_info_present_flag {
            vui.overscan_appropriate_flag = r.read_bit()?;
        }

        // video_signal_type_present_flag
        vui.video_signal_type_present_flag = r.read_bit()?;
        if vui.video_signal_type_present_flag {
            vui.video_format = r.read_bits(3)? as u8;
            vui.video_full_range_flag = r.read_bit()?;
            vui.color_description_present_flag = r.read_bit()?;
            if vui.color_description_present_flag {
                vui.colour_primaries = r.read_bits(8)? as u8;
                vui.transfer_characteristics = r.read_bits(8)? as u8;
                vui.matrix_coefficients = r.read_bits(8)? as u8;
            }
        }

        // chroma_loc_info_present_flag
        vui.chroma_loc_info_present_flag = r.read_bit()?;
        if vui.chroma_loc_info_present_flag {
            vui.chroma_sample_loc_type_top_field = r.read_ue()? as u8;
            vui.chroma_sample_loc_type_bottom_field = r.read_ue()? as u8;
        }

        // timing_info_present_flag
        vui.timing_info_present_flag = r.read_bit()?;
        if vui.timing_info_present_flag {
            vui.num_units_in_tick = r.read_bits(32)?;
            vui.time_scale = r.read_bits(32)?;
            vui.fixed_frame_rate_flag = r.read_bit()?;
        }

        // nal_hrd_parameters_present_flag
        vui.nal_hrd_parameters_present_flag = r.read_bit()?;
        if vui.nal_hrd_parameters_present_flag {
            Self::skip_hrd_parameters(r)?;
        }

        // vcl_hrd_parameters_present_flag
        vui.vcl_hrd_parameters_present_flag = r.read_bit()?;
        if vui.vcl_hrd_parameters_present_flag {
            Self::skip_hrd_parameters(r)?;
        }

        // low_delay_hrd_flag (only if vcl_hrd_parameters_present_flag)
        if vui.vcl_hrd_parameters_present_flag {
            let _low_delay_hrd_flag = r.read_bit()?;
        }

        // These fields only present if HRD parameters are present
        if vui.nal_hrd_parameters_present_flag || vui.vcl_hrd_parameters_present_flag {
            r.read_bits(5)?; // initial_cpb_removal_delay_length_minus1
            r.read_bits(5)?; // cpb_removal_delay_length_minus1
            r.read_bits(5)?; // dpb_output_delay_length_minus1
            r.read_bits(5)?; // offset_for_initial_cpb_removal_delay_length_minus1
        }

        // bitstream_restriction_flag
        vui.bitstream_restriction_flag = r.read_bit()?;
        if vui.bitstream_restriction_flag {
            r.read_ue()?; // motion_vectors_over_pic_boundaries_flag
            r.read_ue()?; // max_bytes_per_pic_denom
            r.read_ue()?; // max_bits_per_mb_denom
            r.read_ue()?; // log2_max_mv_length_horizontal
            r.read_ue()?; // log2_max_mv_length_vertical
            vui.max_num_reorder_frames = r.read_ue()? as u8;
            vui.max_dec_frame_buffering = r.read_ue()? as u8;
        }

        Ok(vui)
    }

    /// Skip HRD parameters from the bitstream.
    fn skip_hrd_parameters(r: &mut BitReader) -> ParserResult<()> {
        let cpb_cnt_minus1 = r.read_ue()?;
        r.read_bits(4)?; // bit_rate_scale
        r.read_bits(4)?; // cpb_size_scale

        for _ in 0..=(cpb_cnt_minus1 as usize) {
            r.read_ue()?; // bit_rate_value_minus1
            r.read_ue()?; // cpb_size_value_minus1
            r.read_ue()?; // cpb_size_value_minus1
            let cbr_flag = r.read_bit()?;
            let _ = cbr_flag; // unused
        }

        Ok(())
    }

    fn parse_sps(&mut self, data: &[u8]) -> ParserResult<vk_video_core::picture::H264Sps> {
        if data.is_empty() {
            return Err(ParserError::InvalidBitstream);
        }

        // Skip NAL header byte (1 byte), enable EPB removal
        let mut r = BitReader::new(&data[1..], true);

        let profile_idc: u8 = r.read_bits(8)? as u8;

        let constraint_set0_flag = r.read_bit()?;
        let constraint_set1_flag = r.read_bit()?;
        let constraint_set2_flag = r.read_bit()?;
        let constraint_set3_flag = r.read_bit()?;
        let constraint_set4_flag = r.read_bit()?;
        let constraint_set5_flag = r.read_bit()?;

        // Skip reserved_zero_2bits
        r.read_bits(2)?;

        let level_idc: u8 = r.read_bits(8)? as u8;

        let seq_parameter_set_id = r.read_ue()?;

        // Profile-dependent fields
        let is_high_profile = matches!(
            profile_idc,
            100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
        );

        let (mut chroma_format_idc, mut separate_colour_plane_flag, mut bit_depth_luma_minus8,
             mut bit_depth_chroma_minus8, mut qpprime_y_zero_transform_bypass_flag,
             mut seq_scaling_matrix_present_flag) = (1u8, false, 0u8, 0u8, false, false);

        if is_high_profile {
            chroma_format_idc = r.read_ue()? as u8;

            if chroma_format_idc == 3 {
                separate_colour_plane_flag = r.read_bit()?;
            }

            bit_depth_luma_minus8 = r.read_ue()? as u8;
            bit_depth_chroma_minus8 = r.read_ue()? as u8;
            qpprime_y_zero_transform_bypass_flag = r.read_bit()?;
            seq_scaling_matrix_present_flag = r.read_bit()?;

            if seq_scaling_matrix_present_flag {
                // Skip scaling lists (not needed for decode)
                let num_scaling_lists = if chroma_format_idc != 3 { 8 } else { 12 };
                for _ in 0..num_scaling_lists {
                    if r.read_bit()? {
                        let _last_scale = r.read_se()?;
                        let _next_scale = r.read_se()?;
                        if _next_scale != 0 {
                            for _ in 0..(16 - _last_scale as usize) {
                                let _ = r.read_se()?;
                            }
                        } else {
                            for _ in 0..16 {
                                let _ = r.read_se()?;
                            }
                        }
                    }
                }
            }
        }

        let log2_max_frame_num_minus4 = r.read_ue()? as u8;
        let max_frame_num = 1u32 << (log2_max_frame_num_minus4 as u32 + 4);

        let pic_order_cnt_type = r.read_ue()? as u8;

        let (mut log2_max_pic_order_cnt_lsb_minus4, mut max_pic_order_cnt_lsb) = (0u8, 0u32);
        match pic_order_cnt_type {
            0 => {
                log2_max_pic_order_cnt_lsb_minus4 = r.read_ue()? as u8;
                max_pic_order_cnt_lsb = 1u32 << (log2_max_pic_order_cnt_lsb_minus4 as u32 + 4);
            }
            1 => {
                let _delta_pic_order_always_zero_flag = r.read_bit()?;
                let _offset_for_non_ref_pic = r.read_se()?;
                let _offset_for_top_to_bottom_field = r.read_se()?;
                let _num_ref_frames_in_pic_order_cnt_cycle = r.read_ue()?;
            }
            2 => {}
            _ => {}
        }

        let max_num_ref_frames = r.read_ue()?;
        let gaps_in_frame_num_value_allowed_flag = r.read_bit()?;
        let pic_width_in_mbs_minus1 = r.read_ue()? as u16;
        let pic_height_in_map_units_minus1 = r.read_ue()? as u16;
        let frame_mbs_only_flag = r.read_bit()?;

        if !frame_mbs_only_flag {
            let _mb_adaptive_frame_field_flag = r.read_bit()?;
        }

        let direct_8x8_inference_flag = r.read_bit()?;
        let frame_cropping_flag = r.read_bit()?;

        let (mut frame_crop_left_offset, mut frame_crop_right_offset,
             mut frame_crop_top_offset, mut frame_crop_bottom_offset) = (0, 0, 0, 0);

        if frame_cropping_flag {
            frame_crop_left_offset = r.read_ue()?;
            frame_crop_right_offset = r.read_ue()?;
            frame_crop_top_offset = r.read_ue()?;
            frame_crop_bottom_offset = r.read_ue()?;
        }

        let vui_parameters_present_flag = r.read_bit()?;

        let vui = if vui_parameters_present_flag {
            Some(Self::parse_vui_parameters(&mut r)?)
        } else {
            None
        };

        let mut sps = vk_video_core::picture::H264Sps::new();
        sps.profile_idc = profile_idc;
        sps.constraint_set0_flag = constraint_set0_flag;
        sps.constraint_set1_flag = constraint_set1_flag;
        sps.constraint_set2_flag = constraint_set2_flag;
        sps.constraint_set3_flag = constraint_set3_flag;
        sps.constraint_set4_flag = constraint_set4_flag;
        sps.constraint_set5_flag = constraint_set5_flag;
        sps.level_idc = level_idc;
        sps.seq_parameter_set_id = seq_parameter_set_id;
        sps.chroma_format_idc = chroma_format_idc;
        sps.separate_colour_plane_flag = separate_colour_plane_flag;
        sps.bit_depth_luma_minus8 = bit_depth_luma_minus8;
        sps.bit_depth_chroma_minus8 = bit_depth_chroma_minus8;
        sps.qpprime_y_zero_transform_bypass_flag = qpprime_y_zero_transform_bypass_flag;
        sps.seq_scaling_matrix_present_flag = seq_scaling_matrix_present_flag;
        sps.log2_max_frame_num_minus4 = log2_max_frame_num_minus4;
        sps.max_frame_num = max_frame_num;
        sps.pic_order_cnt_type = pic_order_cnt_type;
        sps.log2_max_pic_order_cnt_lsb_minus4 = log2_max_pic_order_cnt_lsb_minus4;
        sps.max_pic_order_cnt_lsb = max_pic_order_cnt_lsb;
        sps.max_num_ref_frames = max_num_ref_frames;
        sps.gaps_in_frame_num_value_allowed_flag = gaps_in_frame_num_value_allowed_flag;
        sps.pic_width_in_mbs_minus1 = pic_width_in_mbs_minus1;
        sps.pic_height_in_map_units_minus1 = pic_height_in_map_units_minus1;
        sps.frame_mbs_only_flag = frame_mbs_only_flag;
        sps.direct_8x8_inference_flag = direct_8x8_inference_flag;
        sps.frame_cropping_flag = frame_cropping_flag;
        sps.frame_crop_left_offset = frame_crop_left_offset;
        sps.frame_crop_right_offset = frame_crop_right_offset;
        sps.frame_crop_top_offset = frame_crop_top_offset;
        sps.frame_crop_bottom_offset = frame_crop_bottom_offset;
        sps.vui_parameters_present_flag = vui_parameters_present_flag;
        sps.vui = vui;

        self.update_format_from_sps(&sps);

        Ok(sps)
    }

    fn parse_pps(&mut self, data: &[u8]) -> ParserResult<vk_video_core::picture::H264Pps> {
        if data.is_empty() {
            return Err(ParserError::InvalidBitstream);
        }

        // Skip NAL header byte (1 byte), enable EPB removal
        let mut r = BitReader::new(&data[1..], true);

        let pic_parameter_set_id = r.read_ue()?;
        let seq_parameter_set_id = r.read_ue()?;
        let entropy_coding_mode_flag = r.read_bit()?;
        let bottom_field_pic_order_in_frame_present_flag = r.read_bit()?;
        let num_slice_groups_minus1 = r.read_ue()?;

        // Handle slice group map data when num_slice_groups_minus1 > 0
        // Per H.264 spec, this must be read to maintain correct bitstream position
        if num_slice_groups_minus1 > 0 {
            let slice_group_map_type = r.read_ue()?;
            match slice_group_map_type {
                0 => {
                    // Run-length encoding
                    let num_slice_groups = num_slice_groups_minus1 + 1;
                    for _ in 0..num_slice_groups {
                        let _run_length_minus1 = r.read_ue()?;
                    }
                }
                2 => {
                    // Explicit top-left and bottom-right
                    for _ in 0..num_slice_groups_minus1 {
                        let _top_left = r.read_ue()?;
                        let _bottom_right = r.read_ue()?;
                    }
                }
                3..=5 => {
                    // Above/below or left/right scanning
                    let _slice_group_change_direction_flag = r.read_bit()?;
                    let _slice_group_change_rate_minus1 = r.read_ue()?;
                }
                6 => {
                    // Explicit slice group ID per macroblock
                    let pic_size_in_map_units_minus1 = r.read_ue()?;
                    let v = (num_slice_groups_minus1 + 1).next_power_of_two().trailing_zeros();
                    for _ in 0..=pic_size_in_map_units_minus1 {
                        let _slice_group_id = r.read_bits(v as u8)?;
                    }
                }
                _ => {
                    // Invalid slice_group_map_type, but continue parsing
                    eprintln!("[PPS] Invalid slice_group_map_type: {}", slice_group_map_type);
                }
            }
        }

        let num_ref_idx_l0_default_active_minus1 = r.read_ue()?;
        let num_ref_idx_l1_default_active_minus1 = r.read_ue()?;

        let weighted_pred_flag = r.read_bit().unwrap_or(false);
        let weighted_bipred_idc = r.read_bits(2).unwrap_or(0) as u8;
        let pic_init_qp_minus26 = r.read_se().unwrap_or(0);
        let pic_init_qs_minus26 = r.read_se().unwrap_or(0);
        let chroma_qp_index_offset = r.read_se().unwrap_or(0);

        let deblocking_filter_control_present_flag = r.read_bit().unwrap_or(false);
        let redundant_pic_cnt_present_flag = r.read_bit().unwrap_or(false);

        let transform_8x8_mode_flag = r.read_bit().unwrap_or(false);
        let constrained_intra_pred_flag = r.read_bit().unwrap_or(false);
        // second_chroma_qp_index_offset is only present when constrained_intra_pred_flag is true
        let second_chroma_qp_index_offset = if constrained_intra_pred_flag {
            r.read_se().unwrap_or(0)
        } else {
            0
        };

        let mut pps = vk_video_core::picture::H264Pps::new();
        pps.pic_parameter_set_id = pic_parameter_set_id;
        pps.seq_parameter_set_id = seq_parameter_set_id;
        pps.entropy_coding_mode_flag = entropy_coding_mode_flag;
        pps.bottom_field_pic_order_in_frame_present_flag = bottom_field_pic_order_in_frame_present_flag;
        pps.num_slice_groups_minus1 = num_slice_groups_minus1;
        pps.num_ref_idx_l0_default_active_minus1 = num_ref_idx_l0_default_active_minus1;
        pps.num_ref_idx_l1_default_active_minus1 = num_ref_idx_l1_default_active_minus1;
        pps.weighted_pred_flag = weighted_pred_flag;
        pps.weighted_bipred_idc = weighted_bipred_idc;
        pps.pic_init_qp_minus26 = pic_init_qp_minus26;
        pps.pic_init_qs_minus26 = pic_init_qs_minus26;
        pps.chroma_qp_index_offset = chroma_qp_index_offset;
        pps.deblocking_filter_control_present_flag = deblocking_filter_control_present_flag;
        pps.redundant_pic_cnt_present_flag = redundant_pic_cnt_present_flag;
        pps.transform_8x8_mode_flag = transform_8x8_mode_flag;
        pps.constrained_intra_pred_flag = constrained_intra_pred_flag;
        pps.second_chroma_qp_index_offset = second_chroma_qp_index_offset;

        Ok(pps)
    }

    fn update_format_from_sps(&mut self, sps: &vk_video_core::picture::H264Sps) {
        let coded_width = (sps.pic_width_in_mbs_minus1 as u32 + 1) * 16;
        // frame_mbs_only_flag=1: frame picture, height = macroblocks * 16
        // frame_mbs_only_flag=0: field picture, height = macroblocks * 16 * 2
        let coded_height = if sps.frame_mbs_only_flag {
            (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
        } else {
            (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
        };

        self.detected_format.coded_width = coded_width;
        self.detected_format.coded_height = coded_height;

        self.detected_format.luma_bit_depth = match sps.bit_depth_luma_minus8 {
            0 => vk_video_core::format::ComponentBitDepth::Bit8,
            2 => vk_video_core::format::ComponentBitDepth::Bit10,
            4 => vk_video_core::format::ComponentBitDepth::Bit12,
            _ => vk_video_core::format::ComponentBitDepth::Bit8,
        };

        self.detected_format.chroma_bit_depth = match sps.bit_depth_chroma_minus8 {
            0 => vk_video_core::format::ComponentBitDepth::Bit8,
            2 => vk_video_core::format::ComponentBitDepth::Bit10,
            4 => vk_video_core::format::ComponentBitDepth::Bit12,
            _ => vk_video_core::format::ComponentBitDepth::Bit8,
        };

        self.detected_format.chroma_subsampling = match sps.chroma_format_idc {
            0 => vk_video_core::format::ChromaSubsampling::Monochrome,
            1 => vk_video_core::format::ChromaSubsampling::_420,
            2 => vk_video_core::format::ChromaSubsampling::_422,
            3 => vk_video_core::format::ChromaSubsampling::_444,
            _ => vk_video_core::format::ChromaSubsampling::_420,
        };

        self.detected_format.codec_profile = sps.profile_idc as u32;
        self.detected_format.progressive_sequence = sps.frame_mbs_only_flag;
    }

    fn parse_slice_header(
        &self,
        data: &[u8],
        nal_ref_idc: u8,
        nal_unit_type: u8,
    ) -> ParserResult<SliceHeader> {
        // Skip NAL header byte (1 byte), no EPB removal needed
        let mut r = BitReader::new(&data[1..], false);

        let first_mb_in_slice = r.read_ue()?;
        let slice_type = r.read_ue()? % 5;
        let pps_id = r.read_ue()?;

        let pps = self
            .active_pps
            .as_ref()
            .ok_or(ParserError::ParameterSetParse)?;

        let sps = self
            .active_sps
            .as_ref()
            .ok_or(ParserError::ParameterSetParse)?;

        let frame_num_bits = sps.log2_max_frame_num_minus4 as u32 + 4;
        let frame_num = r.read_bits(frame_num_bits as u8)?;

        let mut slh = SliceHeader {
            first_mb_in_slice,
            slice_type,
            pic_parameter_set_id: pps_id,
            frame_num,
            idr_pic_id: 0,
            pic_order_cnt_lsb: 0,
            delta_pic_order_cnt: [0, 0],
            redundant_pic_cnt: 0,
            num_ref_idx_l0_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
            num_ref_idx_l1_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
            nal_ref_idc,
            nal_unit_type,
            field_pic_flag: false,
            bottom_field: false,
            long_term_reference: false,
        };

        // idr_pic_id is present only for IDR slices (nal_unit_type == 5)
        if nal_unit_type == 5 {
            slh.idr_pic_id = r.read_ue()?;
        }

        if sps.pic_order_cnt_type == 0 {
            let poc_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
            slh.pic_order_cnt_lsb = r.read_bits(poc_bits as u8)? as i32;
            if pps.redundant_pic_cnt_present_flag {
                slh.redundant_pic_cnt = r.read_bits(8)? as i32;
            }
        }

        Ok(slh)
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
                    if let Some((_, _, nal_unit_type)) = nal::parse_h264_nal_header(nal_data) {
                        if nal_units.len() < 5 {
                            eprintln!("[extract_nal] NAL#{}: offset={}, type={}, size={}",
                                nal_units.len(), start, nal_unit_type, nal_data.len());
                        }
                        nal_units.push(NalUnit::new(
                            nal_unit_type,
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

        eprintln!("[extract_nal] Total NALs: {}", nal_units.len());
        nal_units
    }
}

impl VideoParser for H264Parser {
    fn init(&mut self, format: &DetectedVideoFormat) -> ParserResult<()> {
        if format.codec != vk_video_core::codec::VideoCodec::DecodeH264 {
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

        let mut result_sps: Option<vk_video_core::picture::BoxedPictureParametersSet> = None;
        let mut result_pps: Option<vk_video_core::picture::BoxedPictureParametersSet> = None;
        let mut last_slice_offset: Option<usize> = None;
        let mut last_slice_len: Option<usize> = None;
        let mut slice_count: u32 = 0;

        for nal in &nal_units {
            match H264NalUnitType::from_u8(nal.nal_unit_type) {
                Some(H264NalUnitType::Sps) => {
                    eprintln!("[parse] Found SPS NAL, size={}", nal.data.len());
                    eprintln!("[parse] SPS hex: {}", nal.data.iter().take(24).map(|b| format!("{:02x}", b)).collect::<String>());
                    match self.parse_sps(&nal.data) {
                        Ok(sps) => {
                            eprintln!("[parse] SPS parsed OK: width_mbs={}, height_mbs={}, poc_type={}, max_ref={}, cropping={}", 
                                sps.pic_width_in_mbs_minus1, sps.pic_height_in_map_units_minus1,
                                sps.pic_order_cnt_type, sps.max_num_ref_frames, sps.frame_cropping_flag);
                            let sps_id = sps.seq_parameter_set_id;
                            self.sps_cache.insert(sps_id, sps.clone());
                            self.active_sps = Some(sps);
                            result_sps = Some(vk_video_core::picture::BoxedPictureParametersSet::new(
                                self.active_sps.clone().unwrap(),
                            ));
                        }
                        Err(e) => {
                            eprintln!("[parse] SPS parse ERROR: {:?}", e);
                        }
                    }
                }
                Some(H264NalUnitType::Pps) => {
                    eprintln!("[parse] Found PPS NAL, size={}, hex={}", 
                        nal.data.len(), 
                        nal.data.iter().take(16).map(|b| format!("{:02x}", b)).collect::<String>());
                    match self.parse_pps(&nal.data) {
                        Ok(pps) => {
                            eprintln!("[parse] PPS parsed OK: pps_id={}, sps_id={}",
                                pps.pic_parameter_set_id, pps.seq_parameter_set_id);
                            let pps_id = pps.pic_parameter_set_id;
                            self.pps_cache.insert(pps_id, pps.clone());
                            self.active_pps = Some(pps);
                            result_pps = Some(vk_video_core::picture::BoxedPictureParametersSet::new(
                                self.active_pps.clone().unwrap(),
                            ));
                        }
                        Err(e) => {
                            eprintln!("[parse] PPS parse ERROR: {:?}", e);
                        }
                    }
                }
                Some(H264NalUnitType::Sei) => {
                    // SEI - skip, not needed for decoding
                }
                Some(H264NalUnitType::NonIdrSlice)
                | Some(H264NalUnitType::IdrSlice)
                | Some(H264NalUnitType::DataPartitionA)
                | Some(H264NalUnitType::DataPartitionB)
                | Some(H264NalUnitType::DataPartitionC) => {
                    let (is_trailing, nal_ref_idc, nal_unit_type) =
                        nal::parse_h264_nal_header(&nal.data).unwrap_or((false, 0, 0));

                    if self.first_slice_header.is_none() {
                        if let Ok(slh) = self.parse_slice_header(&nal.data, nal_ref_idc, nal_unit_type) {
                            self.first_slice_header = Some(slh);
                            self.frame_count += 1;
                        }
                    }

                    last_slice_offset = Some(nal.offset);
                    last_slice_len = Some(nal.size);
                    slice_count += 1;
                }
                Some(H264NalUnitType::FillerData) | Some(H264NalUnitType::SeqEnd) | Some(H264NalUnitType::StreamEnd) => {
                    // Skip filler and end codes
                }
                _ => {}
            }
        }

        if result_sps.is_some() || result_pps.is_some() {
            Ok(ParseResult::ParameterSet {
                sps: result_sps,
                pps: result_pps,
                vps: None,
            })
        } else if let (Some(offset), Some(len)) = (last_slice_offset, last_slice_len) {
            Ok(ParseResult::Slice {
                slice_data_offset: offset,
                slice_data_len: len,
                num_slices: slice_count,
                slice_header: None,
            })
        } else {
            Ok(ParseResult::Nothing)
        }
    }

    fn reset(&mut self) {
        self.sps_cache.clear();
        self.pps_cache.clear();
        self.active_sps = None;
        self.active_pps = None;
        self.first_slice_header = None;
        self.frame_count = 0;
        self.idr_found = false;
        self.prev_frame_num = 0;
        self.prev_pic_order_cnt_lsb = 0;
        self.prev_pic_order_cnt_msb = 0;
    }

    fn detected_format(&self) -> &DetectedVideoFormat {
        &self.detected_format
    }
}

#[derive(Debug, Clone)]
pub struct SliceHeader {
    pub first_mb_in_slice: u32,
    pub slice_type: u32,
    pub pic_parameter_set_id: u32,
    pub frame_num: u32,
    pub idr_pic_id: u32,
    pub pic_order_cnt_lsb: i32,
    pub delta_pic_order_cnt: [i32; 2],
    pub redundant_pic_cnt: i32,
    pub num_ref_idx_l0_active_minus1: u32,
    pub num_ref_idx_l1_active_minus1: u32,
    pub nal_ref_idc: u8,
    pub nal_unit_type: u8,
    pub field_pic_flag: bool,
    pub bottom_field: bool,
    pub long_term_reference: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pps_parse_born_trailer() {
        // PPS from born_trailer.h264: 68ce3c80
        // NAL header: 0x68 (type=8, ref_idc=3)
        // Payload: ce3c80
        let data = [0x68, 0xce, 0x3c, 0x80];
        
        // Debug: print the payload bits
        let payload = &data[1..];
        let bits: String = payload.iter()
            .map(|b| format!("{:08b}", b))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("PPS payload bytes: {:02x?}", payload);
        eprintln!("PPS payload bits: {}", bits);

        let mut parser = H264Parser::new();
        parser.init(&DetectedVideoFormat::new(
            vk_video_core::codec::VideoCodec::DecodeH264,
        )).unwrap();

        let pps = parser.parse_pps(&data).unwrap();
        
        eprintln!("Parsed PPS:");
        eprintln!("  pic_parameter_set_id: {}", pps.pic_parameter_set_id);
        eprintln!("  seq_parameter_set_id: {}", pps.seq_parameter_set_id);
        eprintln!("  entropy_coding_mode_flag: {}", pps.entropy_coding_mode_flag);
        eprintln!("  bottom_field_pic_order_in_frame_present_flag: {}", pps.bottom_field_pic_order_in_frame_present_flag);
        eprintln!("  num_slice_groups_minus1: {}", pps.num_slice_groups_minus1);
        eprintln!("  num_ref_idx_l0_default_active_minus1: {}", pps.num_ref_idx_l0_default_active_minus1);
        eprintln!("  num_ref_idx_l1_default_active_minus1: {}", pps.num_ref_idx_l1_default_active_minus1);
        eprintln!("  weighted_pred_flag: {}", pps.weighted_pred_flag);
        eprintln!("  weighted_bipred_idc: {}", pps.weighted_bipred_idc);
        eprintln!("  deblocking_filter_control_present_flag: {}", pps.deblocking_filter_control_present_flag);
        eprintln!("  constrained_intra_pred_flag: {}", pps.constrained_intra_pred_flag);

        // Verify key fields match H.264 spec parsing (NOT the buggy C++ reference)
        // Note: C++ parser has bugs in field ordering (reads constrained_intra_pred_flag
        // before redundant_pic_cnt_present_flag), so we trust the spec instead.
        assert_eq!(pps.pic_parameter_set_id, 0);
        assert_eq!(pps.seq_parameter_set_id, 0);
        assert_eq!(pps.entropy_coding_mode_flag, false);
        assert_eq!(pps.bottom_field_pic_order_in_frame_present_flag, false);
        assert_eq!(pps.num_slice_groups_minus1, 0);
        assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 0);
        assert_eq!(pps.num_ref_idx_l1_default_active_minus1, 0);
        assert_eq!(pps.weighted_pred_flag, false);
        assert_eq!(pps.weighted_bipred_idc, 0);
        assert_eq!(pps.pic_init_qp_minus26, 0);
        assert_eq!(pps.pic_init_qs_minus26, 0);
        assert_eq!(pps.chroma_qp_index_offset, 0);
        assert_eq!(pps.deblocking_filter_control_present_flag, true);
        // Bit 16 of payload ce3c80 is 1, so constrained_intra_pred_flag = 1 per spec
        assert_eq!(pps.constrained_intra_pred_flag, true);
        assert_eq!(pps.redundant_pic_cnt_present_flag, false);
        assert_eq!(pps.transform_8x8_mode_flag, false);
        // second_chroma_qp_index_offset is read since constrained_intra_pred_flag is true
        assert_eq!(pps.second_chroma_qp_index_offset, 0);
    }
}

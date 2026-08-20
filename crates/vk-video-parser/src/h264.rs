//! H.264/AVC bitstream parser.
//!
//! Parses H.264 bitstreams to extract SPS, PPS, and slice data.
//! Uses BitReader with inline emulation-prevention byte removal.

use std::collections::HashMap;

use crate::nal::{self, H264NalUnitType, NalUnit};
use crate::{
    DetectedVideoFormat, ParseResult, ParserError, ParserResult, VideoParser, SliceEntry, SliceHeader as ParserSliceHeader,
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
    /// Offset up to which NAL units have been processed (to avoid re-parsing SPS/PPS).
    processed_up_to: usize,
    /// Current frame number being collected (to group slices by frame).
    current_frame_num: Option<u32>,
    /// Current pic_order_cnt_lsb being collected (to group slices by picture).
    /// Two slice NALs belong to the same picture only if they share BOTH
    /// frame_num and pic_order_cnt_lsb.
    current_poc_lsb: Option<i32>,
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

impl Default for H264Parser {
    fn default() -> Self {
        Self::new()
    }
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
            processed_up_to: 0,
            current_frame_num: None,
            current_poc_lsb: None,
            cached_nals: Vec::new(),
            cached_payload_len: 0,
            nal_cursor: 0,
        }
    }

    /// Returns a reference to the active SPS, if any.
    pub fn active_sps(&self) -> Option<&vk_video_core::picture::H264Sps> {
        self.active_sps.as_ref()
    }

    /// Returns a reference to the active PPS, if any.
    pub fn active_pps(&self) -> Option<&vk_video_core::picture::H264Pps> {
        self.active_pps.as_ref()
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

        // low_delay_hrd_flag and pic_struct_present_flag (only if HRD parameters are present)
        // Per H.264 spec E.2.1, these follow the HRD parameters
        if vui.nal_hrd_parameters_present_flag || vui.vcl_hrd_parameters_present_flag {
            let _low_delay_hrd_flag = r.read_bit()?;
            vui.pic_struct_present_flag = r.read_bit()?;
        }

        // bitstream_restriction_flag
        vui.bitstream_restriction_flag = r.read_bit()?;
        if vui.bitstream_restriction_flag {
            vui.motion_vectors_over_pic_boundaries_flag = r.read_bit()?;
            vui.max_bytes_per_pic_denom = r.read_ue()? as u8;
            vui.max_bits_per_mb_denom = r.read_ue()? as u8;
            vui.log2_max_mv_length_horizontal = r.read_ue()? as u8;
            vui.log2_max_mv_length_vertical = r.read_ue()? as u8;
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
            let _cbr_flag = r.read_bit()?;
        }

        // Per H.264 spec E.2.1, these 4 fields are part of HRD parameters
        r.read_bits(5)?; // initial_cpb_removal_delay_length_minus1
        r.read_bits(5)?; // cpb_removal_delay_length_minus1
        r.read_bits(5)?; // dpb_output_delay_length_minus1
        r.read_bits(5)?; // time_offset_length

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
            100 | 110 | 122 | 244 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
        );

        let (mut chroma_format_idc, mut separate_colour_plane_flag, mut bit_depth_luma_minus8,
             mut bit_depth_chroma_minus8, mut qpprime_y_zero_transform_bypass_flag,
             mut seq_scaling_matrix_present_flag) = (1u8, false, 0u8, 0u8, false, false);

        // Scaling list storage — populated when seq_scaling_matrix_present_flag is true
        let mut scaling_list_4x4: [[u8; 16]; 6] = [[0u8; 16]; 6];
        let mut scaling_list_8x8: [[u8; 64]; 2] = [[0u8; 64]; 2];

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
                // Parse and store scaling lists per H.264 spec 7.3.2.1.1.1
                // Indices 0-5: 4x4 scaling lists (16 coefficients each)
                // Indices 6-7 (or 6-11 for chroma_format_idc==3): 8x8 scaling lists (64 coefficients each)
                let num_scaling_lists = if chroma_format_idc != 3 { 8 } else { 12 };

                // Extended 8x8 storage for 4:4:4 (up to 6 matrices)
                let mut scaling_8x8_ext: [[u8; 64]; 6] = [[0u8; 64]; 6];

                for idx in 0..num_scaling_lists {
                    let scaling_list_pred_mode_flag = r.read_bit()?;
                    if !scaling_list_pred_mode_flag {
                        // Predicted from another matrix
                        let scaling_list_pred_matrix_id_delta = r.read_ue()?;
                        // Determine the prediction source index per H.264 Table 7.1
                        let pred_idx = if idx < 6 {
                            // 4x4: 0→0, 1→0, 2→0, 3→2, 4→2, 5→2
                            (idx / 3) * 2
                        } else {
                            // 8x8: 6→2 (or 5 for 4:4:4), 7→6, 8→5, 9→5, 10→5, 11→5
                            if idx == 6 {
                                if chroma_format_idc == 3 { 5 } else { 2 }
                            } else if idx == 7 {
                                6
                            } else {
                                5
                            }
                        };
                        let final_pred_idx = (pred_idx + scaling_list_pred_matrix_id_delta) as usize;
                        if idx < 6 {
                            scaling_list_4x4[idx as usize] = scaling_list_4x4[final_pred_idx];
                        } else {
                            let dst_idx = (idx - 6) as usize;
                            let src_idx = if final_pred_idx < 6 {
                                // 4x4 matrix used as prediction for 8x8
                                // Per H.264 Annex E.2.1: ScalingList[6][j] = ScalingList[pred][j>>1]
                                for j in 0..64 {
                                    scaling_8x8_ext[dst_idx][j] = scaling_list_4x4[final_pred_idx][j >> 1];
                                }
                                continue; // Skip the copy at the end
                            } else {
                                (final_pred_idx - 6) as usize
                            };
                            scaling_8x8_ext[dst_idx] = scaling_8x8_ext[src_idx];
                        }
                    } else {
                        // Delta coding: last_scale starts at 8, next_scale updated per delta
                        let mut last_scale: i32 = 8;
                        let mut next_scale: i32 = 8;
                        let num_coeffs = if idx < 6 { 16 } else { 64 };
                        if idx < 6 {
                            let list = &mut scaling_list_4x4[idx as usize];
                            for val in list.iter_mut().take(num_coeffs as usize) {
                                if next_scale != 0 {
                                    let delta_scale = r.read_se()?;
                                    next_scale = ((last_scale + delta_scale) + 256) % 256;
                                    if next_scale != 0 {
                                        last_scale = next_scale;
                                        *val = last_scale as u8;
                                    }
                                } else {
                                    let _next_scale = r.read_se()?;
                                }
                            }
                        } else {
                            let list_idx = (idx - 6) as usize;
                            let list = &mut scaling_8x8_ext[list_idx];
                            for val in list.iter_mut().take(num_coeffs as usize) {
                                if next_scale != 0 {
                                    let delta_scale = r.read_se()?;
                                    next_scale = ((last_scale + delta_scale) + 256) % 256;
                                    if next_scale != 0 {
                                        last_scale = next_scale;
                                        *val = last_scale as u8;
                                    }
                                } else {
                                    let _next_scale = r.read_se()?;
                                }
                            }
                        }
                    }
                }

                // Copy the relevant 8x8 matrices (0-1 for non-4:4:4, 0-5 for 4:4:4)
                for i in 0..2 {
                    scaling_list_8x8[i] = scaling_8x8_ext[i];
                }
                if chroma_format_idc == 3 {
                    for i in 0..6 {
                        // Store in extended array; picparams will handle mapping
                        scaling_8x8_ext[i] = scaling_8x8_ext[i];
                    }
                }
            }
        }

        let log2_max_frame_num_minus4 = r.read_ue()? as u8;
        let max_frame_num = 1u32 << (log2_max_frame_num_minus4 as u32 + 4);

        let pic_order_cnt_type = r.read_ue()? as u8;

        let (mut log2_max_pic_order_cnt_lsb_minus4, mut max_pic_order_cnt_lsb) = (0u8, 0u32);
        let (mut delta_pic_order_always_zero_flag, mut offset_for_non_ref_pic,
             mut offset_for_top_to_bottom_field, mut num_ref_frames_in_pic_order_cnt_cycle) =
            (false, 0i32, 0i32, 0u32);
        let mut offset_for_ref_frame: Vec<i32> = Vec::new();

        match pic_order_cnt_type {
            0 => {
                log2_max_pic_order_cnt_lsb_minus4 = r.read_ue()? as u8;
                max_pic_order_cnt_lsb = 1u32 << (log2_max_pic_order_cnt_lsb_minus4 as u32 + 4);
            }
            1 => {
                delta_pic_order_always_zero_flag = r.read_bit()?;
                offset_for_non_ref_pic = r.read_se()?;
                offset_for_top_to_bottom_field = r.read_se()?;
                num_ref_frames_in_pic_order_cnt_cycle = r.read_ue()?;
                for _ in 0..num_ref_frames_in_pic_order_cnt_cycle {
                    offset_for_ref_frame.push(r.read_se()?);
                }
            }
            2 => {}
            _ => {}
        }

        let max_num_ref_frames = r.read_ue()?;
        let gaps_in_frame_num_value_allowed_flag = r.read_bit()?;
        let pic_width_in_mbs_minus1 = r.read_ue()? as u16;
        let pic_height_in_map_units_minus1 = r.read_ue()? as u16;
        let frame_mbs_only_flag = r.read_bit()?;

        let mb_adaptive_frame_field_flag = if !frame_mbs_only_flag {
            r.read_bit()?
        } else {
            false
        };

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
        sps.scaling_list_4x4 = scaling_list_4x4;
        sps.scaling_list_8x8 = scaling_list_8x8;
        sps.log2_max_frame_num_minus4 = log2_max_frame_num_minus4;
        sps.max_frame_num = max_frame_num;
        sps.pic_order_cnt_type = pic_order_cnt_type;
        sps.delta_pic_order_always_zero_flag = delta_pic_order_always_zero_flag;
        sps.offset_for_non_ref_pic = offset_for_non_ref_pic;
        sps.offset_for_top_to_bottom_field = offset_for_top_to_bottom_field;
        sps.num_ref_frames_in_pic_order_cnt_cycle = num_ref_frames_in_pic_order_cnt_cycle;
        sps.offset_for_ref_frame = offset_for_ref_frame;
        sps.log2_max_pic_order_cnt_lsb_minus4 = log2_max_pic_order_cnt_lsb_minus4;
        sps.max_pic_order_cnt_lsb = max_pic_order_cnt_lsb;
        sps.max_num_ref_frames = max_num_ref_frames;
        sps.gaps_in_frame_num_value_allowed_flag = gaps_in_frame_num_value_allowed_flag;
        sps.pic_width_in_mbs_minus1 = pic_width_in_mbs_minus1;
        sps.pic_height_in_map_units_minus1 = pic_height_in_map_units_minus1;
        sps.frame_mbs_only_flag = frame_mbs_only_flag;
        sps.mb_adaptive_frame_field_flag = mb_adaptive_frame_field_flag;
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

        let weighted_pred_flag = r.read_bit()?;
        let weighted_bipred_idc = r.read_bits(2)? as u8;
        let pic_init_qp_minus26 = r.read_se()?;
        let pic_init_qs_minus26 = r.read_se()?;
        let chroma_qp_index_offset = r.read_se()?;

        let deblocking_filter_control_present_flag = r.read_bit()?;
        let constrained_intra_pred_flag = r.read_bit()?;
        let redundant_pic_cnt_present_flag = r.read_bit()?;

        // transform_8x8_mode_flag, pic_scaling_matrix_present_flag, and
        // second_chroma_qp_index_offset are only present when there is more RBSP data
        // Per H.264 spec: second_chroma_qp_index_offset defaults to chroma_qp_index_offset
        let (transform_8x8_mode_flag, second_chroma_qp_index_offset) = if r.has_more_rsbp_data() {
            let transform_8x8_mode_flag = r.read_bit()?;
            let pic_scaling_matrix_present_flag = r.read_bit()?;
            if pic_scaling_matrix_present_flag {
                // Parse scaling lists to advance bitstream position correctly.
                // Same algorithm as SPS scaling lists.
                for _ in 0..6 {
                    let scaling_list_pred_mode_flag = r.read_bit()?;
                    if !scaling_list_pred_mode_flag {
                        let _scaling_list_pred_matrix_id_delta = r.read_ue()?;
                    } else {
                        let mut last_scale: i32 = 8;
                        let mut next_scale: i32 = 8;
                        for _ in 0..16 {
                            if next_scale != 0 {
                                let delta_scale = r.read_se()?;
                                next_scale = ((last_scale + delta_scale) + 256) % 256;
                                if next_scale != 0 {
                                    last_scale = next_scale;
                                }
                            } else {
                                let _next_scale = r.read_se()?;
                            }
                        }
                    }
                }
                // 8x8 scaling lists (if transform_8x8_mode_flag)
                if transform_8x8_mode_flag {
                    for _ in 0..2 {
                        let scaling_list_pred_mode_flag = r.read_bit()?;
                        if !scaling_list_pred_mode_flag {
                            let _scaling_list_pred_matrix_id_delta = r.read_ue()?;
                        } else {
                            let mut last_scale: i32 = 8;
                            let mut next_scale: i32 = 8;
                            for _ in 0..64 {
                                if next_scale != 0 {
                                    let delta_scale = r.read_se()?;
                                    next_scale = ((last_scale + delta_scale) + 256) % 256;
                                    if next_scale != 0 {
                                        last_scale = next_scale;
                                    }
                                } else {
                                    let _next_scale = r.read_se()?;
                                }
                            }
                        }
                    }
                }
            }
            let second_chroma_qp_index_offset = r.read_se()?;
            (transform_8x8_mode_flag, second_chroma_qp_index_offset)
        } else {
            (false, chroma_qp_index_offset)
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

    pub fn parse_slice_header(
        &self,
        data: &[u8],
        nal_ref_idc: u8,
        nal_unit_type: u8,
    ) -> ParserResult<SliceHeader> {
        // Skip NAL header byte (1 byte), enable EPB removal
        let mut r = BitReader::new(&data[1..], true);
        let header_start_pos = r.position();

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

        // colour_plane_id is present when separate_colour_plane_flag is set (H.264 spec 7.4.3)
        if sps.separate_colour_plane_flag {
            r.read_bits(2)?;
        }

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
            direct_spatial_mv_pred_flag: false,
            num_ref_idx_active_override_flag: false,
            cabac_init_idc: 0,
            slice_qp_delta: 0,
            disable_deblocking_filter_idc: 0,
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            ref_pic_list_modification_l0: Vec::new(),
            ref_pic_list_modification_l1: Vec::new(),
            dec_ref_pic_marking: Vec::new(),
            no_output_of_prior_pics_flag: false,
            long_term_reference_flag: false,
            header_bit_size: 0,
            luma_log2_weight_denom: 0,
            chroma_log2_weight_denom: 0,
            luma_weight_l0_flag: 0,
            luma_weight_l0: [0i16; 32],
            luma_offset_l0: [0i16; 32],
            chroma_weight_l0_flag: 0,
            chroma_weight_l0: [[0i16; 2]; 32],
            chroma_offset_l0: [[0i16; 2]; 32],
            luma_weight_l1_flag: 0,
            luma_weight_l1: [0i16; 32],
            luma_offset_l1: [0i16; 32],
            chroma_weight_l1_flag: 0,
            chroma_weight_l1: [[0i16; 2]; 32],
            chroma_offset_l1: [[0i16; 2]; 32],
        };

        // field_pic_flag and bottom_field_flag (when not frame-only)
        if !sps.frame_mbs_only_flag {
            slh.field_pic_flag = r.read_bit()?;
            if slh.field_pic_flag {
                slh.bottom_field = r.read_bit()?;
            }
        }

        // idr_pic_id is present only for IDR slices (nal_unit_type == 5)
        if nal_unit_type == 5 {
            slh.idr_pic_id = r.read_ue()?;
        }

        // POC type 0: pic_order_cnt_lsb
        if sps.pic_order_cnt_type == 0 {
            let poc_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
            slh.pic_order_cnt_lsb = r.read_bits(poc_bits as u8)? as i32;
            if pps.bottom_field_pic_order_in_frame_present_flag && !slh.field_pic_flag {
                slh.delta_pic_order_cnt[0] = r.read_se()?; // delta_pic_order_cnt_bottom
            }
        }

        // POC type 1: delta_pic_order_cnt
        if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero_flag {
            slh.delta_pic_order_cnt[0] = r.read_se()?;
            if pps.bottom_field_pic_order_in_frame_present_flag && !slh.field_pic_flag {
                slh.delta_pic_order_cnt[1] = r.read_se()?;
            }
        }

        // POC type 2: implicit, nothing to read

        if pps.redundant_pic_cnt_present_flag {
            slh.redundant_pic_cnt = r.read_ue()? as i32;
        }

        // Slice type classification (already mod 5: 0=P, 1=B, 2=I, 3=SP, 4=SI)
        let is_p = slice_type == 0;
        let is_b = slice_type == 1;
        let is_i = slice_type == 2;
        let is_sp = slice_type == 3;
        let is_si = slice_type == 4;

        // Decoded reference picture marking (H.264 spec 7.3.2.1.1). This comes
        // BEFORE num_ref_idx_active_override_flag / ref_pic_list_modification.
        //   IDR (nal_unit_type == 5): no_output_of_prior_pics_flag + long_term_reference_flag
        //   non-IDR reference (nal_ref_idc > 0): adaptive_ref_pic_marking_mode_flag + MMCO ops
        if nal_ref_idc > 0 {
            let (marking, no_output, lt_ref) = Self::parse_dec_ref_pic_marking(&mut r, nal_unit_type == 5)?;
            slh.dec_ref_pic_marking = marking;
            slh.no_output_of_prior_pics_flag = no_output;
            slh.long_term_reference_flag = lt_ref;
        }

        // P/SP/B slice: num_ref_idx_active_override_flag
        if is_p || is_sp || is_b {
            slh.num_ref_idx_active_override_flag = r.read_bit()?;
            if slh.num_ref_idx_active_override_flag {
                slh.num_ref_idx_l0_active_minus1 = r.read_ue()?;
                if is_b {
                    slh.num_ref_idx_l1_active_minus1 = r.read_ue()?;
                }
            }
        }

        // Reference picture list modification (for P/SP/B slices, H.264 spec 7.3.6).
        // Only index [0] of each list is ever modified (single flag per list).
        if is_p || is_sp || is_b {
            let (mod_l0, mod_l1) =
                Self::parse_ref_pic_list_modification(&mut r, is_b)?;
            slh.ref_pic_list_modification_l0 = mod_l0;
            slh.ref_pic_list_modification_l1 = mod_l1;
        }

        // CABAC init IDC (not for I/SI slices)
        if pps.entropy_coding_mode_flag && !is_i && !is_si {
            slh.cabac_init_idc = r.read_ue()? as u8;
        }

        // Slice QP delta
        slh.slice_qp_delta = r.read_se()?;

        // Deblocking filter parameters: present in the slice header only when
        // deblocking_filter_control_present_flag == 0 (H.264 spec 7.4.3). When the
        // flag is 1 the control lives in the PPS and nothing is read here.
        if !pps.deblocking_filter_control_present_flag {
            slh.disable_deblocking_filter_idc = r.read_ue()? as i8;
            if slh.disable_deblocking_filter_idc != 1 {
                slh.slice_alpha_c0_offset_div2 = r.read_se()?;
                slh.slice_beta_offset_div2 = r.read_se()?;
            }
        }

        // Pred weight table: present LAST in the slice header (H.264 spec 7.4.3),
        // when pps_weighted_pred_flag == 1 and slice_type != B (covers P, SP, I,
        // SI), OR when pps_weighted_bipred_idc == 1 and slice_type == B.
        let has_pred_weight_table =
            (pps.weighted_pred_flag && !is_b)
                || (pps.weighted_bipred_idc == 1 && is_b);
        if has_pred_weight_table {
            let (
                luma_log2_weight_denom, chroma_log2_weight_denom,
                luma_weight_l0_flag, luma_weight_l0, luma_offset_l0,
                chroma_weight_l0_flag, chroma_weight_l0, chroma_offset_l0,
                luma_weight_l1_flag, luma_weight_l1, luma_offset_l1,
                chroma_weight_l1_flag, chroma_weight_l1, chroma_offset_l1,
            ) = Self::parse_pred_weight_table(&mut r, sps, slice_type,
                slh.num_ref_idx_l0_active_minus1, slh.num_ref_idx_l1_active_minus1)?;
            slh.luma_log2_weight_denom = luma_log2_weight_denom;
            slh.chroma_log2_weight_denom = chroma_log2_weight_denom;
            slh.luma_weight_l0_flag = luma_weight_l0_flag;
            slh.luma_weight_l0 = luma_weight_l0;
            slh.luma_offset_l0 = luma_offset_l0;
            slh.chroma_weight_l0_flag = chroma_weight_l0_flag;
            slh.chroma_weight_l0 = chroma_weight_l0;
            slh.chroma_offset_l0 = chroma_offset_l0;
            slh.luma_weight_l1_flag = luma_weight_l1_flag;
            slh.luma_weight_l1 = luma_weight_l1;
            slh.luma_offset_l1 = luma_offset_l1;
            slh.chroma_weight_l1_flag = chroma_weight_l1_flag;
            slh.chroma_weight_l1 = chroma_weight_l1;
            slh.chroma_offset_l1 = chroma_offset_l1;
        }

        // Calculate header bit size (bits consumed from slice header start)
        slh.header_bit_size = (r.position() - header_start_pos) as u16;

        Ok(slh)
    }

    /// Parse reference picture list modification (H.264 spec 7.3.6).
    ///
    /// H.264 only ever modifies reference-list index [0] of each list — there is
    /// a single `ref_pic_list_modification_flag_lX[0]` per list, NOT a per-index
    /// loop over `numRefIdx` (that is the HEVC model). Syntax:
    ///   ref_pic_list_modification_flag_l0[0]  u(1)
    ///   if( ref_pic_list_modification_flag_l0[0] ) {
    ///       modification_of_pic_nums_idc  ue(v)
    ///       if( modification_of_pic_nums_idc == 0 )
    ///           abs_diff_pic_num_minus1   ue(v)
    ///       else if( modification_of_pic_nums_idc == 1 || == 2 )
    ///           long_term_pic_num         ue(v)
    ///   }
    ///   if( slice_type == B ) {
    ///       ref_pic_list_modification_flag_l1[0]  u(1)
    ///       ... same ...
    ///   }
    /// A `ue(v)` value is always present when the flag is set (op 0, 1, or 2).
    fn parse_ref_pic_list_modification(
        r: &mut BitReader,
        is_b: bool,
    ) -> ParserResult<(Vec<RefPicListModificationEntry>, Vec<RefPicListModificationEntry>)> {
        let mut mod_l0 = Vec::new();
        let mut mod_l1 = Vec::new();

        // Ref pic list 0 modification (for P/SP/B slices): index [0] only.
        if r.read_bit()? { // ref_pic_list_modification_flag_l0[0]
            let op = r.read_ue()?; // modification_of_pic_nums_idc
            // value: abs_diff_pic_num_minus1 (op 0) or long_term_pic_num (op 1/2), always ue(v).
            let value = r.read_ue()?;
            mod_l0.push(RefPicListModificationEntry {
                index: 0,
                length: 0,
                op,
                difference: value as i32,
            });
        }

        // Ref pic list 1 modification (for B slices only): index [0] only.
        if is_b {
            if r.read_bit()? { // ref_pic_list_modification_flag_l1[0]
                let op = r.read_ue()?; // modification_of_pic_nums_idc
                let value = r.read_ue()?;
                mod_l1.push(RefPicListModificationEntry {
                    index: 0,
                    length: 0,
                    op,
                    difference: value as i32,
                });
            }
        }

        Ok((mod_l0, mod_l1))
    }

    /// Parse prediction weight table (H.264 spec 7.4.4).
    fn parse_pred_weight_table(
        r: &mut BitReader,
        sps: &vk_video_core::picture::H264Sps,
        slice_type: u32,
        num_ref_idx_l0_active_minus1: u32,
        num_ref_idx_l1_active_minus1: u32,
    ) -> ParserResult<(
        u8, u8, // luma_log2_weight_denom, chroma_log2_weight_denom
        u8, [i16; 32], [i16; 32], // luma_weight_l0_flag, luma_weight_l0, luma_offset_l0
        u8, [[i16; 2]; 32], [[i16; 2]; 32], // chroma_weight_l0_flag, chroma_weight_l0, chroma_offset_l0
        u8, [i16; 32], [i16; 32], // luma_weight_l1_flag, luma_weight_l1, luma_offset_l1
        u8, [[i16; 2]; 32], [[i16; 2]; 32], // chroma_weight_l1_flag, chroma_weight_l1, chroma_offset_l1
    )> {
        let is_b = slice_type == 1; // B-slice after modulo 5
        let luma_log2_weight_denom = r.read_ue()? as u8;

        let chroma_log2_weight_denom = if sps.chroma_format_idc != 0 {
            r.read_ue()? as u8
        } else {
            0
        };

        // L0 weights: loop count based on actual num_ref_idx_l0_active_minus1
        // Per H.264 spec 7.4.4
        let mut luma_weight_l0_flag: u8 = 0;
        let mut luma_weight_l0 = [0i16; 32];
        let mut luma_offset_l0 = [0i16; 32];
        let mut chroma_weight_l0_flag: u8 = 0;
        let mut chroma_weight_l0 = [[0i16; 2]; 32];
        let mut chroma_offset_l0 = [[0i16; 2]; 32];

        for i in 0..=(num_ref_idx_l0_active_minus1 as usize) {
            let lw_flag = r.read_bit()?;
            if lw_flag {
                luma_weight_l0_flag |= 1 << i;
                luma_weight_l0[i] = r.read_se()? as i16;
                luma_offset_l0[i] = r.read_se()? as i16;
            }
            if sps.chroma_format_idc != 0 {
                let cw_flag = r.read_bit()?;
                if cw_flag {
                    chroma_weight_l0_flag |= 1 << i;
                    chroma_weight_l0[i][0] = r.read_se()? as i16;
                    chroma_offset_l0[i][0] = r.read_se()? as i16;
                    chroma_weight_l0[i][1] = r.read_se()? as i16;
                    chroma_offset_l0[i][1] = r.read_se()? as i16;
                }
            }
        }

        // L1 weights (B slices only)
        let mut luma_weight_l1_flag: u8 = 0;
        let mut luma_weight_l1 = [0i16; 32];
        let mut luma_offset_l1 = [0i16; 32];
        let mut chroma_weight_l1_flag: u8 = 0;
        let mut chroma_weight_l1 = [[0i16; 2]; 32];
        let mut chroma_offset_l1 = [[0i16; 2]; 32];

        if is_b {
            for i in 0..=(num_ref_idx_l1_active_minus1 as usize) {
                let lw_flag = r.read_bit()?;
                if lw_flag {
                    luma_weight_l1_flag |= 1 << i;
                    luma_weight_l1[i] = r.read_se()? as i16;
                    luma_offset_l1[i] = r.read_se()? as i16;
                }
                if sps.chroma_format_idc != 0 {
                    let cw_flag = r.read_bit()?;
                    if cw_flag {
                        chroma_weight_l1_flag |= 1 << i;
                        chroma_weight_l1[i][0] = r.read_se()? as i16;
                        chroma_offset_l1[i][0] = r.read_se()? as i16;
                        chroma_weight_l1[i][1] = r.read_se()? as i16;
                        chroma_offset_l1[i][1] = r.read_se()? as i16;
                    }
                }
            }
        }

        Ok((
            luma_log2_weight_denom, chroma_log2_weight_denom,
            luma_weight_l0_flag, luma_weight_l0, luma_offset_l0,
            chroma_weight_l0_flag, chroma_weight_l0, chroma_offset_l0,
            luma_weight_l1_flag, luma_weight_l1, luma_offset_l1,
            chroma_weight_l1_flag, chroma_weight_l1, chroma_offset_l1,
        ))
    }

    /// Parse decoded reference picture marking (H.264 spec 7.4.5).
    fn parse_dec_ref_pic_marking(
        r: &mut BitReader,
        is_idr: bool,
    ) -> ParserResult<(Vec<DecRefPicMarkingEntry>, bool, bool)> {
        let mut marking = Vec::new();
        let mut no_output_of_prior_pics_flag = false;
        let mut long_term_reference_flag = false;

        if is_idr {
            // IDR picture marking: both flags ALWAYS present per H.264 spec 7.4.5
            no_output_of_prior_pics_flag = r.read_bit()?;
            long_term_reference_flag = r.read_bit()?;
        } else {
            // Non-IDR picture marking: first read adaptive_ref_pic_marking_mode_flag (H.264 spec 7.4.5)
            let adaptive_ref_pic_marking_mode_flag = r.read_bit()?;
            if adaptive_ref_pic_marking_mode_flag {
                // Adaptive memory management: read operations until memory_management_control_operation=0
                loop {
                    let memory_management_control_operation = r.read_ue()?;
                    if memory_management_control_operation == 0 {
                        break;
                    }
                    let value = match memory_management_control_operation {
                        1 | 4 | 6 => r.read_ue()?, // difference_of_pic_nums_minus1
                        2 | 5 | 7 | 9 => r.read_ue()?, // long_term_pic_num
                        3 => r.read_ue()?, // max_long_term_frame_idx_plus1
                        8 => 0, // unmark_all_short_term: no value
                        _ => 0,
                    };
                    marking.push(DecRefPicMarkingEntry {
                        memory_management_control_operation,
                        value,
                    });
                }
            }
        }

        Ok((marking, no_output_of_prior_pics_flag, long_term_reference_flag))
    }

    fn extract_nal_units(&self, data: &[u8], start_offset: usize) -> Vec<NalUnit> {
        let mut nal_units = Vec::new();
        let mut offset = start_offset;

        while offset < data.len() {
            if let Some((start, code_len)) = nal::find_next_start_code(data, offset) {
                let next_start = nal::find_next_start_code(data, start + code_len);

                let (end, next_code_len) = match next_start {
                    Some((ns, ncl)) => (ns, ncl),
                    None => (data.len(), 0),
                };

                // A 4-byte start code (00 00 00 01) is equivalent to a trailing
                // 0x00 byte (part of the current NAL's RBSP) followed by a
                // 3-byte start code (00 00 01). Hardware decoders (e.g. NVIDIA
                // cuvid) include that leading 0x00 in the current NAL's data,
                // so we extend the NAL by one byte when the next start code is
                // 4-byte. Without this the NAL is one byte short and the
                // decoder produces corrupted output.
                let nal_end = if next_code_len == 4 { end + 1 } else { end };
                let nal_data = &data[start + code_len..nal_end];
                if !nal_data.is_empty() {
                    if let Some((_, _, nal_unit_type)) = nal::parse_h264_nal_header(nal_data) {
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

        // Rebuild the NAL cache if a packet of a different size arrived (a new
        // chunk of data). Otherwise reuse the cached NALs and advance the
        // cursor, avoiding an O(n^2) re-scan/re-copy of the bitstream.
        if packet.payload.len() != self.cached_payload_len {
            self.cached_nals = self.extract_nal_units(&packet.payload, 0);
            self.cached_payload_len = packet.payload.len();
            self.nal_cursor = 0;
            self.processed_up_to = 0;
        }

        if self.nal_cursor >= self.cached_nals.len() {
            return Ok(ParseResult::Nothing);
        }

        let mut result_sps: Option<vk_video_core::picture::BoxedPictureParametersSet> = None;
        let mut result_pps: Option<vk_video_core::picture::BoxedPictureParametersSet> = None;
        let mut result_sps_nal: Option<Vec<u8>> = None;
        let mut result_pps_nal: Option<Vec<u8>> = None;
        let mut slices: Vec<SliceEntry> = Vec::new();
        let mut first_slice_offset: Option<usize> = None;
        let mut last_slice_end: Option<usize> = None;
        // Cursor index of the first collected slice NAL. Used to roll the
        // cursor back when a parameter set is returned instead of the slices
        // (GOP boundary case), so the slices are re-processed next call.
        let mut first_slice_cursor: Option<usize> = None;

        let mut i = self.nal_cursor;
        while i < self.cached_nals.len() {
            let nal = &self.cached_nals[i];

            match H264NalUnitType::from_u8(nal.nal_unit_type) {
                Some(H264NalUnitType::Sps) => {
                    // A picture is in progress: defer this parameter set until
                    // the current picture's slices are returned, otherwise the
                    // last picture before a GOP boundary would be dropped.
                    if !slices.is_empty() {
                        break;
                    }
                    let nal_data = nal.data.clone();
                    let (off, sz) = (nal.offset, nal.size);
                    match self.parse_sps(&nal_data) {
                        Ok(sps) => {
                            let sps_id = sps.seq_parameter_set_id;
                            self.sps_cache.insert(sps_id, sps.clone());
                            self.active_sps = Some(sps);
                            result_sps = Some(vk_video_core::picture::BoxedPictureParametersSet::new(
                                self.active_sps.clone().unwrap(),
                            ));
                            result_sps_nal = Some(nal_data);
                            // Mark this NAL as processed
                            self.processed_up_to = off + sz;
                            i += 1;
                        }
                        Err(_) => {
                            i += 1;
                        }
                    }
                }
                Some(H264NalUnitType::Pps) => {
                    // Same as SPS: don't swallow a parameter set while a
                    // picture's slices have already been collected.
                    if !slices.is_empty() {
                        break;
                    }
                    let nal_data = nal.data.clone();
                    let (off, sz) = (nal.offset, nal.size);
                    match self.parse_pps(&nal_data) {
                        Ok(pps) => {
                            let pps_id = pps.pic_parameter_set_id;
                            self.pps_cache.insert(pps_id, pps.clone());
                            self.active_pps = Some(pps);
                            result_pps = Some(vk_video_core::picture::BoxedPictureParametersSet::new(
                                self.active_pps.clone().unwrap(),
                            ));
                            result_pps_nal = Some(nal_data);
                            // Mark this NAL as processed
                            self.processed_up_to = off + sz;
                            i += 1;
                        }
                        Err(_) => {
                            i += 1;
                        }
                    }
                }
                Some(H264NalUnitType::Sei) => {
                    // SEI - skip, not needed for decoding
                    i += 1;
                }
                Some(H264NalUnitType::NonIdrSlice)
                | Some(H264NalUnitType::IdrSlice)
                | Some(H264NalUnitType::DataPartitionA)
                | Some(H264NalUnitType::DataPartitionB)
                | Some(H264NalUnitType::DataPartitionC) => {
                    // Copy the NAL data out so the borrow of self.cached_nals
                    // ends before the &mut self calls below.
                    let nal_data = nal.data.clone();
                    let (off, sz) = (nal.offset, nal.size);
                    let (is_trailing, nal_ref_idc, nal_unit_type) =
                        nal::parse_h264_nal_header(&nal_data).unwrap_or((false, 0, 0));

                    if self.first_slice_header.is_none() {
                        if let Ok(slh) = self.parse_slice_header(&nal_data, nal_ref_idc, nal_unit_type) {
                            self.first_slice_header = Some(slh);
                            self.frame_count += 1;
                        }
                    }

                    // Parse slice header for this NAL
                    let slice_header = if let Ok(slh) = self.parse_slice_header(&nal_data, nal_ref_idc, nal_unit_type) {
                        Some(ParserSliceHeader::H264(slh))
                    } else {
                        None
                    };

                    // Track the first frame's frame_num and pic_order_cnt_lsb
                    let (current_frame_num, current_poc_lsb) =
                        if let Some(ParserSliceHeader::H264(slh)) = &slice_header {
                            (slh.frame_num, slh.pic_order_cnt_lsb)
                        } else {
                            // Unparseable slice header: skip this NAL.
                            i += 1;
                            continue;
                        };

                    // Only collect slices for the first frame
                    if slices.is_empty() {
                        // First slice - record its frame_num and poc_lsb
                        self.current_frame_num = Some(current_frame_num);
                        self.current_poc_lsb = Some(current_poc_lsb);
                        first_slice_cursor = Some(i);
                    }

                    // Stop collecting if frame_num OR pic_order_cnt_lsb changes
                    // (a new picture). Two slice NALs belong to the same picture
                    // only if they share both frame_num and pic_order_cnt_lsb.
                    if let (Some(first_frame_num), Some(first_poc_lsb)) =
                        (self.current_frame_num, self.current_poc_lsb)
                    {
                        if current_frame_num != first_frame_num || current_poc_lsb != first_poc_lsb {
                            // New picture detected, stop collecting (do not
                            // consume this NAL; it starts the next picture).
                            break;
                        }
                    }

                    // Track offsets for bytes_consumed calculation
                    if first_slice_offset.is_none() {
                        first_slice_offset = Some(off);
                    }
                    last_slice_end = Some(off + sz);

                    slices.push(SliceEntry {
                        slice_header,
                        nal_data,
                    });
                    i += 1;
                }
                Some(H264NalUnitType::FillerData) | Some(H264NalUnitType::SeqEnd) | Some(H264NalUnitType::StreamEnd) => {
                    // Skip filler and end codes
                    i += 1;
                }
                _ => {
                    i += 1;
                }
            }
        }
        if result_sps.is_some() || result_pps.is_some() {
            // If slices were collected but a parameter set is returned instead
            // (a [SPS][PPS][slice] sequence spanning a GOP boundary), roll the
            // cursor back to the first collected slice so it is re-processed on
            // the next parse() call. This mirrors the original behavior where
            // processed_up_to was not advanced past unreturned slices.
            self.nal_cursor = if !slices.is_empty() {
                first_slice_cursor.unwrap_or(i)
            } else {
                i
            };
            Ok(ParseResult::ParameterSet {
                sps: result_sps,
                pps: result_pps,
                vps: None,
                sps_nal: result_sps_nal,
                pps_nal: result_pps_nal,
            })
        } else if !slices.is_empty() {
            self.nal_cursor = i;
            // Calculate bytes consumed: from first slice start to last slice end
            let bytes_consumed = if let (Some(first), Some(last)) = (first_slice_offset, last_slice_end) {
                last - first
            } else {
                0
            };
            // Update processed_up_to to the end of the last slice in this frame
            if let Some(last) = last_slice_end {
                self.processed_up_to = last;
            }
            // Clear current_frame_num for next frame
            self.current_frame_num = None;
            self.current_poc_lsb = None;
            // Clear first_slice_header so the next picture gets a fresh parse
            self.first_slice_header.take();
            Ok(ParseResult::Slice { slices, bytes_consumed })
        } else {
            self.nal_cursor = i;
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
        self.processed_up_to = 0;
        self.current_frame_num = None;
        self.cached_nals.clear();
        self.cached_payload_len = 0;
        self.nal_cursor = 0;
    }

    fn detected_format(&self) -> &DetectedVideoFormat {
        &self.detected_format
    }
}

/// Reference picture list modification entry (H.264 spec 7.3.6 / 8.2.3.2).
///
/// H.264 only ever modifies reference-list index [0], so at most one entry is
/// stored per list (when `ref_pic_list_modification_flag_lX[0]` is set).
#[derive(Debug, Clone)]
pub struct RefPicListModificationEntry {
    /// Reference-list index this modification applies to. Always 0 in H.264.
    pub index: u32,
    /// Reserved (0). H.264 has no per-entry length; kept for API stability.
    pub length: u32,
    /// `modification_of_pic_nums_idc` (0, 1, or 2).
    pub op: u32,
    /// Associated value: `abs_diff_pic_num_minus1` (op 0) or `long_term_pic_num`
    /// (op 1/2). Always present when the entry exists.
    pub difference: i32,
}

/// Decoded reference picture marking operation (H.264 spec 7.4.5).
#[derive(Debug, Clone)]
pub struct DecRefPicMarkingEntry {
    /// memory_management_control_operation value.
    pub memory_management_control_operation: u32,
    /// Associated value depending on operation type.
    pub value: u32,
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
    // Additional slice header fields
    pub direct_spatial_mv_pred_flag: bool,
    pub num_ref_idx_active_override_flag: bool,
    pub cabac_init_idc: u8,
    pub slice_qp_delta: i32,
    pub disable_deblocking_filter_idc: i8,
    pub slice_alpha_c0_offset_div2: i32,
    pub slice_beta_offset_div2: i32,
    // Reference picture list modification (H.264 spec 7.4.3)
    pub ref_pic_list_modification_l0: Vec<RefPicListModificationEntry>,
    pub ref_pic_list_modification_l1: Vec<RefPicListModificationEntry>,
    // Decoded reference picture marking (H.264 spec 7.4.5)
    pub dec_ref_pic_marking: Vec<DecRefPicMarkingEntry>,
    pub no_output_of_prior_pics_flag: bool,
    pub long_term_reference_flag: bool,
    // Slice header size in bits (excluding NAL header)
    pub header_bit_size: u16,
    // Weighted prediction parameters (H.264 spec 7.4.4)
    pub luma_log2_weight_denom: u8,
    pub chroma_log2_weight_denom: u8,
    pub luma_weight_l0_flag: u8,
    pub luma_weight_l0: [i16; 32],
    pub luma_offset_l0: [i16; 32],
    pub chroma_weight_l0_flag: u8,
    pub chroma_weight_l0: [[i16; 2]; 32],
    pub chroma_offset_l0: [[i16; 2]; 32],
    pub luma_weight_l1_flag: u8,
    pub luma_weight_l1: [i16; 32],
    pub luma_offset_l1: [i16; 32],
    pub chroma_weight_l1_flag: u8,
    pub chroma_weight_l1: [[i16; 2]; 32],
    pub chroma_offset_l1: [[i16; 2]; 32],
}

// TODO: Add PPS/slice header tests with verified bitstream data from real streams

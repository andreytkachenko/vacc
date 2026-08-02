//! Comprehensive SPS/PPS comparison test.
//! 
//! Parses born_trailer.h264 and dumps ALL fields for comparison with C++ reference.

use std::fs;
use vk_video_core::codec::VideoCodec;
use vk_video_parser::{
    h264::H264Parser,
    DetectedVideoFormat,
    VideoParser,
    bitstream::BitstreamPacket,
    ParseResult,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = if args.len() >= 2 { &args[1] } else { "born_trailer.h264" };

    let data = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}", path, e);
        std::process::exit(1);
    });

    println!("=== Rust H.264 SPS/PPS Full Dump ===");
    println!("File: {} ({} bytes)\n", path, data.len());

    let mut parser = H264Parser::new();
    let format = DetectedVideoFormat::new(VideoCodec::DecodeH264);
    parser.init(&format).expect("Failed to init parser");

    let packet = BitstreamPacket::new(data.clone());
    match parser.parse(&packet) {
        Ok(ParseResult::ParameterSet { sps, pps, .. }) => {
            if let Some(sps_boxed) = sps {
                if let Some(sps) = sps_boxed.downcast_ref::<vk_video_core::picture::H264Sps>() {
                    dump_sps_full(sps);
                }
            }
            if let Some(pps_boxed) = pps {
                if let Some(pps) = pps_boxed.downcast_ref::<vk_video_core::picture::H264Pps>() {
                    dump_pps_full(pps);
                }
            }
        }
        Ok(r) => {
            eprintln!("Unexpected parse result: {:?}", r);
        }
        Err(e) => {
            eprintln!("Parse error: {}", e);
        }
    }

    println!("\n=== Expected from C++ reference (cpp_sps_pps.txt) ===");
    println!("(Note: C++ stores Vulkan enum for level_idc, not raw H.264 value)");
    println!("SPS level_idc: 11 (Vulkan enum for Level 4.1, raw=41)");
    println!("PPS entropy_coding_mode_flag: 0");
    println!("PPS num_slice_groups_minus1: 0");
    println!("PPS deblocking_filter_control_present_flag: 1");
}

fn dump_sps_full(sps: &vk_video_core::picture::H264Sps) {
    println!("=== SPS id={} ===", sps.seq_parameter_set_id);
    println!("profile_idc: {}", sps.profile_idc);
    println!("constraint_set_flags: 0x{:02x}", 
        ((sps.constraint_set0_flag as u8) << 0) |
        ((sps.constraint_set1_flag as u8) << 1) |
        ((sps.constraint_set2_flag as u8) << 2) |
        ((sps.constraint_set3_flag as u8) << 3) |
        ((sps.constraint_set4_flag as u8) << 4) |
        ((sps.constraint_set5_flag as u8) << 5));
    println!("  constraint_set0_flag: {}", sps.constraint_set0_flag as u8);
    println!("  constraint_set1_flag: {}", sps.constraint_set1_flag as u8);
    println!("  constraint_set2_flag: {}", sps.constraint_set2_flag as u8);
    println!("  constraint_set3_flag: {}", sps.constraint_set3_flag as u8);
    println!("  constraint_set4_flag: {}", sps.constraint_set4_flag as u8);
    println!("  constraint_set5_flag: {}", sps.constraint_set5_flag as u8);
    println!("level_idc: {} (raw H.264 value)", sps.level_idc);
    println!("chroma_format_idc: {}", sps.chroma_format_idc);
    println!("separate_colour_plane_flag: {}", sps.separate_colour_plane_flag as u8);
    println!("bit_depth_luma_minus8: {}", sps.bit_depth_luma_minus8);
    println!("bit_depth_chroma_minus8: {}", sps.bit_depth_chroma_minus8);
    println!("qpprime_y_zero_transform_bypass_flag: {}", sps.qpprime_y_zero_transform_bypass_flag as u8);
    println!("seq_scaling_matrix_present_flag: {}", sps.seq_scaling_matrix_present_flag as u8);
    println!("log2_max_frame_num_minus4: {}", sps.log2_max_frame_num_minus4);
    println!("pic_order_cnt_type: {}", sps.pic_order_cnt_type);
    println!("log2_max_pic_order_cnt_lsb_minus4: {}", sps.log2_max_pic_order_cnt_lsb_minus4);
    println!("max_num_ref_frames: {}", sps.max_num_ref_frames);
    println!("gaps_in_frame_num_value_allowed_flag: {}", sps.gaps_in_frame_num_value_allowed_flag as u8);
    println!("pic_width_in_mbs_minus1: {}", sps.pic_width_in_mbs_minus1);
    println!("pic_height_in_map_units_minus1: {}", sps.pic_height_in_map_units_minus1);
    println!("frame_mbs_only_flag: {}", sps.frame_mbs_only_flag as u8);
    println!("direct_8x8_inference_flag: {}", sps.direct_8x8_inference_flag as u8);
    println!("frame_cropping_flag: {}", sps.frame_cropping_flag as u8);
    if sps.frame_cropping_flag {
        println!("frame_crop_left_offset: {}", sps.frame_crop_left_offset);
        println!("frame_crop_right_offset: {}", sps.frame_crop_right_offset);
        println!("frame_crop_top_offset: {}", sps.frame_crop_top_offset);
        println!("frame_crop_bottom_offset: {}", sps.frame_crop_bottom_offset);
    }
    println!("vui_parameters_present_flag: {}", sps.vui_parameters_present_flag as u8);
    if let Some(vui) = &sps.vui {
        println!("vui.aspect_ratio_idc: {}", vui.aspect_ratio_idc);
        println!("vui.timing_info_present_flag: {}", vui.timing_info_present_flag as u8);
        if vui.timing_info_present_flag {
            println!("vui.num_units_in_tick: {}", vui.num_units_in_tick);
            println!("vui.time_scale: {}", vui.time_scale);
            println!("vui.fixed_frame_rate_flag: {}", vui.fixed_frame_rate_flag as u8);
        }
        println!("vui.max_num_reorder_frames: {}", vui.max_num_reorder_frames);
        println!("vui.max_dec_frame_buffering: {}", vui.max_dec_frame_buffering);
        println!("vui.nal_hrd_parameters_present_flag: {}", vui.nal_hrd_parameters_present_flag as u8);
    }
    println!();
}

fn dump_pps_full(pps: &vk_video_core::picture::H264Pps) {
    println!("=== PPS id={} (sps_id={}) ===", 
        pps.pic_parameter_set_id, pps.seq_parameter_set_id);
    println!("entropy_coding_mode_flag: {}", pps.entropy_coding_mode_flag as u8);
    println!("bottom_field_pic_order_in_frame_present_flag: {}", pps.bottom_field_pic_order_in_frame_present_flag as u8);
    println!("num_slice_groups_minus1: {}", pps.num_slice_groups_minus1);
    println!("num_ref_idx_l0_default_active_minus1: {}", pps.num_ref_idx_l0_default_active_minus1);
    println!("num_ref_idx_l1_default_active_minus1: {}", pps.num_ref_idx_l1_default_active_minus1);
    println!("weighted_pred_flag: {}", pps.weighted_pred_flag as u8);
    println!("weighted_bipred_idc: {}", pps.weighted_bipred_idc);
    println!("pic_init_qp_minus26: {}", pps.pic_init_qp_minus26);
    println!("pic_init_qs_minus26: {}", pps.pic_init_qs_minus26);
    println!("chroma_qp_index_offset: {}", pps.chroma_qp_index_offset);
    println!("second_chroma_qp_index_offset: {}", pps.second_chroma_qp_index_offset);
    println!("deblocking_filter_control_present_flag: {}", pps.deblocking_filter_control_present_flag as u8);
    println!("constrained_intra_pred_flag: {}", pps.constrained_intra_pred_flag as u8);
    println!("redundant_pic_cnt_present_flag: {}", pps.redundant_pic_cnt_present_flag as u8);
    println!("transform_8x8_mode_flag: {}", pps.transform_8x8_mode_flag as u8);
    println!();
}

//! Dump detailed SPS/PPS fields for comparison with C++ reference.

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

    println!("=== H.264 SPS/PPS Dump ===");
    println!("File: {} ({} bytes)\n", path, data.len());

    let mut parser = H264Parser::new();
    let format = DetectedVideoFormat::new(VideoCodec::DecodeH264);
    parser.init(&format).expect("Failed to init parser");

    let packet = BitstreamPacket::new(data.clone());
    match parser.parse(&packet) {
        Ok(ParseResult::ParameterSet { sps, pps, vps: _ }) => {
            if let Some(sps_boxed) = sps {
                if let Some(sps) = sps_boxed.downcast_ref::<vk_video_core::picture::H264Sps>() {
                    dump_sps(sps);
                }
            }
            if let Some(pps_boxed) = pps {
                if let Some(pps) = pps_boxed.downcast_ref::<vk_video_core::picture::H264Pps>() {
                    dump_pps(pps);
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
}

fn dump_sps(sps: &vk_video_core::picture::H264Sps) {
    println!("SPS id={}:", sps.seq_parameter_set_id);
    println!("  profile_idc: {} ({})", 
        sps.profile_idc, profile_name(sps.profile_idc));
    println!("  level_idc: {} ({})", 
        sps.level_idc, level_name(sps.level_idc));
    println!("  chroma_format_idc: {} ({})", 
        sps.chroma_format_idc, chroma_name(sps.chroma_format_idc));
    println!("  bit_depth_luma_minus8: {} ({}-bit)", 
        sps.bit_depth_luma_minus8, sps.bit_depth_luma_minus8 as u32 + 8);
    println!("  bit_depth_chroma_minus8: {} ({}-bit)", 
        sps.bit_depth_chroma_minus8, sps.bit_depth_chroma_minus8 as u32 + 8);
    println!("  log2_max_frame_num_minus4: {} (MaxFrameNum={})", 
        sps.log2_max_frame_num_minus4, 1u32 << (sps.log2_max_frame_num_minus4 as u32 + 4));
    println!("  pic_order_cnt_type: {}", sps.pic_order_cnt_type);
    println!("  log2_max_pic_order_cnt_lsb_minus4: {} (MaxPicOrderCntLsb={})", 
        sps.log2_max_pic_order_cnt_lsb_minus4, 
        1u32 << (sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4));
    println!("  max_num_ref_frames: {}", sps.max_num_ref_frames);
    println!("  pic_width_in_mbs_minus1: {} ({} pixels)", 
        sps.pic_width_in_mbs_minus1, (sps.pic_width_in_mbs_minus1 as u32 + 1) * 16);
    println!("  pic_height_in_map_units_minus1: {} ({} pixels)", 
        sps.pic_height_in_map_units_minus1, 
        if sps.frame_mbs_only_flag {
            (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
        } else {
            (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
        });
    println!("  frame_mbs_only_flag: {}", sps.frame_mbs_only_flag as u32);
    println!("  vui_parameters_present_flag: {}", sps.vui_parameters_present_flag as u32);

    if let Some(vui) = &sps.vui {
        println!("  vui:");
        println!("    timing_info_present_flag: {}", vui.timing_info_present_flag as u32);
        if vui.timing_info_present_flag {
            println!("    time_scale: {}", vui.time_scale);
            println!("    num_units_in_tick: {}", vui.num_units_in_tick);
        }
    }
    println!();
}

fn dump_pps(pps: &vk_video_core::picture::H264Pps) {
    println!("PPS id={} (sps_id={}):", 
        pps.pic_parameter_set_id, pps.seq_parameter_set_id);
    println!("  entropy_coding_mode_flag: {} ({})", 
        pps.entropy_coding_mode_flag as u32,
        if pps.entropy_coding_mode_flag { "CABAC" } else { "CAVLC" });
    println!("  bottom_field_pic_order_in_frame_present_flag: {}", 
        pps.bottom_field_pic_order_in_frame_present_flag as u32);
    println!("  num_slice_groups_minus1: {}", pps.num_slice_groups_minus1);
    println!("  num_ref_idx_l0_default_active_minus1: {}", 
        pps.num_ref_idx_l0_default_active_minus1);
    println!("  num_ref_idx_l1_default_active_minus1: {}", 
        pps.num_ref_idx_l1_default_active_minus1);
    println!("  deblocking_filter_control_present_flag: {}", 
        pps.deblocking_filter_control_present_flag as u32);
    println!("  transform_8x8_mode_flag: {}", 
        pps.transform_8x8_mode_flag as u32);
    println!("  constrained_intra_pred_flag: {}", 
        pps.constrained_intra_pred_flag as u32);
    println!();
}

fn profile_name(idc: u8) -> &'static str {
    match idc {
        66 => "Main Profile",
        77 => "High Profile",
        100 => "High Profile",
        _ => "Unknown",
    }
}

fn level_name(idc: u8) -> &'static str {
    match idc {
        10 => "Level 1.0",
        11 => "Level 1.1",
        12 => "Level 1.2",
        13 => "Level 1.3",
        20 => "Level 2.0",
        21 => "Level 2.1",
        22 => "Level 2.2",
        30 => "Level 3.0",
        31 => "Level 3.1",
        32 => "Level 3.2",
        40 => "Level 4.0",
        41 => "Level 4.1",
        42 => "Level 4.2",
        50 => "Level 5.0",
        51 => "Level 5.1",
        52 => "Level 5.2",
        _ => "Unknown",
    }
}

fn chroma_name(idc: u8) -> &'static str {
    match idc {
        0 => "Monochrome",
        1 => "4:2:0",
        2 => "4:2:2",
        3 => "4:4:4",
        _ => "Unknown",
    }
}

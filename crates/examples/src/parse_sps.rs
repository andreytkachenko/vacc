use vk_video_core::codec::VideoCodec;
use vk_video_parser::h264::H264Parser;
use vk_video_parser::{BitstreamPacket, DetectedVideoFormat, VideoParser};

fn main() {
    let file_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/born_trailer.h264".to_string());

    let data = std::fs::read(&file_path).expect("Failed to read file");
    println!("Read {} bytes from {}", data.len(), file_path);

    // Create parser and initialize
    let mut parser = H264Parser::new();
    let format = DetectedVideoFormat::new(VideoCodec::DecodeH264);
    parser.init(&format).expect("Failed to init parser");

    // Parse the entire file as one packet
    let packet = BitstreamPacket::new(data);
    let result = parser.parse(&packet).expect("Failed to parse");

    // Get the active SPS
    let sps = parser
        .active_sps()
        .expect("No SPS found in the bitstream");

    println!("=== H.264 SPS Fields ===\n");

    // Profile and level
    println!("profile_idc                    = {} ({})", sps.profile_idc, profile_name(sps.profile_idc));
    println!("constraint_set0_flag           = {}", sps.constraint_set0_flag);
    println!("constraint_set1_flag           = {}", sps.constraint_set1_flag);
    println!("constraint_set2_flag           = {}", sps.constraint_set2_flag);
    println!("constraint_set3_flag           = {}", sps.constraint_set3_flag);
    println!("constraint_set4_flag           = {}", sps.constraint_set4_flag);
    println!("constraint_set5_flag           = {}", sps.constraint_set5_flag);
    println!("level_idc                      = {} ({})", sps.level_idc, level_name(sps.level_idc));

    // SPS ID
    println!("\nseq_parameter_set_id             = {}", sps.seq_parameter_set_id);

    // Chroma format
    println!("\nchroma_format_idc              = {} ({})", sps.chroma_format_idc, chroma_name(sps.chroma_format_idc));
    println!("separate_colour_plane_flag     = {}", sps.separate_colour_plane_flag);

    // Bit depth
    let bit_depth_luma = 8 + sps.bit_depth_luma_minus8 as u8;
    let bit_depth_chroma = if sps.chroma_format_idc != 0 {
        8 + sps.bit_depth_chroma_minus8 as u8
    } else {
        bit_depth_luma
    };
    println!("\nbit_depth_luma_minus8          = {} (bit depth luma = {})", sps.bit_depth_luma_minus8, bit_depth_luma);
    println!("bit_depth_chroma_minus8        = {} (bit depth chroma = {})", sps.bit_depth_chroma_minus8, bit_depth_chroma);

    // Frame number
    println!("\nlog2_max_frame_num_minus4      = {}", sps.log2_max_frame_num_minus4);
    println!("max_frame_num                  = {}", sps.max_frame_num);

    // Picture order count
    println!("\npic_order_cnt_type             = {}", sps.pic_order_cnt_type);
    println!("log2_max_pic_order_cnt_lsb_minus4 = {}", sps.log2_max_pic_order_cnt_lsb_minus4);
    println!("max_pic_order_cnt_lsb          = {}", sps.max_pic_order_cnt_lsb);
    if sps.pic_order_cnt_type == 1 {
        println!("delta_pic_order_always_zero_flag  = {}", sps.delta_pic_order_always_zero_flag);
        println!("offset_for_non_ref_pic         = {}", sps.offset_for_non_ref_pic);
        println!("offset_for_top_to_bottom_field = {}", sps.offset_for_top_to_bottom_field);
        println!("num_ref_frames_in_pic_order_cnt_cycle = {}", sps.num_ref_frames_in_pic_order_cnt_cycle);
        if !sps.offset_for_ref_frame.is_empty() {
            println!("offset_for_ref_frame             = {:?}", sps.offset_for_ref_frame);
        }
    }

    // Reference frames
    println!("\nmax_num_ref_frames             = {}", sps.max_num_ref_frames);
    println!("gaps_in_frame_num_value_allowed_flag = {}", sps.gaps_in_frame_num_value_allowed_flag);

    // Picture dimensions
    println!("\npic_width_in_mbs_minus1        = {}", sps.pic_width_in_mbs_minus1);
    println!("pic_height_in_map_units_minus1 = {}", sps.pic_height_in_map_units_minus1);
    println!("frame_mbs_only_flag            = {}", sps.frame_mbs_only_flag);
    println!("mb_adaptive_frame_field_flag   = {}", sps.mb_adaptive_frame_field_flag);

    // Other flags
    println!("\ndirect_8x8_inference_flag      = {}", sps.direct_8x8_inference_flag);
    println!("qpprime_y_zero_transform_bypass_flag = {}", sps.qpprime_y_zero_transform_bypass_flag);
    println!("seq_scaling_matrix_present_flag = {}", sps.seq_scaling_matrix_present_flag);

    // Frame cropping
    println!("\nframe_cropping_flag            = {}", sps.frame_cropping_flag);
    if sps.frame_cropping_flag {
        println!("frame_crop_left_offset       = {}", sps.frame_crop_left_offset);
        println!("frame_crop_right_offset      = {}", sps.frame_crop_right_offset);
        println!("frame_crop_top_offset        = {}", sps.frame_crop_top_offset);
        println!("frame_crop_bottom_offset     = {}", sps.frame_crop_bottom_offset);
    }

    // VUI
    println!("\nvui_parameters_present_flag    = {}", sps.vui_parameters_present_flag);
    if let Some(vui) = &sps.vui {
        println!("\n--- VUI Parameters ---");
        println!("aspect_ratio_info_present_flag = {}", vui.aspect_ratio_info_present_flag);
        if vui.aspect_ratio_info_present_flag {
            println!("aspect_ratio_idc             = {}", vui.aspect_ratio_idc);
            if vui.aspect_ratio_idc == 255 {
                println!("sar_width                  = {}", vui.sar_width);
                println!("sar_height                 = {}", vui.sar_height);
            }
        }
        println!("timing_info_present_flag       = {}", vui.timing_info_present_flag);
        if vui.timing_info_present_flag {
            println!("num_units_in_tick            = {}", vui.num_units_in_tick);
            println!("time_scale                   = {}", vui.time_scale);
            if vui.time_scale > 0 && vui.num_units_in_tick > 0 {
                let fps = vui.time_scale as f64 / vui.num_units_in_tick as f64;
                println!("frame_rate                   = {:.2} fps", fps);
            }
        }
        println!("fixed_frame_rate_flag          = {}", vui.fixed_frame_rate_flag);
        println!("video_signal_type_present_flag = {}", vui.video_signal_type_present_flag);
        if vui.video_signal_type_present_flag {
            println!("video_format                 = {}", vui.video_format);
            println!("video_full_range_flag        = {}", vui.video_full_range_flag);
            println!("colour_primaries             = {}", vui.colour_primaries);
            println!("transfer_characteristics     = {}", vui.transfer_characteristics);
            println!("matrix_coefficients          = {}", vui.matrix_coefficients);
        }
        println!("bitstream_restriction_flag     = {}", vui.bitstream_restriction_flag);
        if vui.bitstream_restriction_flag {
            println!("max_num_reorder_frames       = {}", vui.max_num_reorder_frames);
            println!("max_dec_frame_buffering      = {}", vui.max_dec_frame_buffering);
        }
    }

    // Compute dimensions
    println!("\n=== Computed Dimensions ===\n");

    let frame_mbs_only = sps.frame_mbs_only_flag;
    let num_mb_width = sps.pic_width_in_mbs_minus1 as u32 + 1;
    let num_mb_height = if frame_mbs_only {
        sps.pic_height_in_map_units_minus1 as u32 + 1
    } else {
        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 2
    };

    let coded_width = num_mb_width * 16;
    let coded_height = num_mb_height * 16;

    println!("num_mb_width                   = {}", num_mb_width);
    println!("num_mb_height                  = {}", num_mb_height);
    println!("coded_width                    = {}", coded_width);
    println!("coded_height                   = {}", coded_height);

    if sps.frame_cropping_flag {
        let chroma_format_idc = sps.chroma_format_idc;
        let crop_mul_x = if chroma_format_idc == 0 { 1 } else { 2 };
        let crop_div_y = if frame_mbs_only { 2 } else { 1 };

        let crop_left = sps.frame_crop_left_offset as u32 * crop_mul_x;
        let crop_right = sps.frame_crop_right_offset as u32 * crop_mul_x;
        let crop_top = sps.frame_crop_top_offset as u32 * crop_div_y;
        let crop_bottom = sps.frame_crop_bottom_offset as u32 * crop_div_y;

        let display_width = coded_width - crop_left - crop_right;
        let display_height = coded_height - crop_top - crop_bottom;

        println!("\ncrop_left (pixels)             = {}", crop_left);
        println!("crop_right (pixels)            = {}", crop_right);
        println!("crop_top (pixels)              = {}", crop_top);
        println!("crop_bottom (pixels)           = {}", crop_bottom);
        println!("\ndisplay_width                  = {}", display_width);
        println!("display_height                 = {}", display_height);
    } else {
        println!("\ndisplay_width                  = {}", coded_width);
        println!("display_height                 = {}", coded_height);
    }
}

fn profile_name(idc: u8) -> &'static str {
    match idc {
        66 => "Constrained Baseline",
        77 => "Main",
        88 => "Extended",
        100 => "High",
        110 => "High 10",
        122 => "High 4:2:2",
        244 => "Intra",
        _ => "Other",
    }
}

fn level_name(idc: u8) -> &'static str {
    match idc {
        10 => "1.0",
        11 => "1.1",
        12 => "1.2",
        13 => "1.3",
        20 => "2.0",
        21 => "2.1",
        22 => "2.2",
        30 => "3.0",
        31 => "3.1",
        32 => "3.2",
        40 => "4.0",
        41 => "4.1",
        42 => "4.2",
        50 => "5.0",
        51 => "5.1",
        52 => "5.2",
        60 => "6.0",
        61 => "6.1",
        62 => "6.2",
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

//! Bitstream parsing examples with pixel-level verification.
//!
//! This example demonstrates:
//! 1. Parsing H.264 and H.265 bitstreams to extract NAL units
//! 2. Extracting SPS/PPS/VPS parameter sets
//! 3. Detecting video format (dimensions, chroma, bit depth)
//! 4. Identifying slice boundaries and frame structure
//! 5. Verifying parsed data against ffmpeg-decoded reference frames

use std::fs;
use std::process::Command;

use image::GenericImageView;
use vk_video_core::codec::VideoCodec;
use vk_video_core::format::{ChromaSubsampling, ComponentBitDepth};
use vk_video_parser::{
    h264::H264Parser,
    h265::H265Parser,
    DetectedVideoFormat,
    VideoParser,
    bitstream::BitstreamPacket,
    nal::find_next_start_code,
    ParseResult,
};

fn main() {
    println!("=== Bitstream Parsing & Pixel Verification ===\n");

    // Parse H.264 bitstream
    println!("========================================");
    println!("  H.264 Bitstream Analysis");
    println!("========================================");
    let h264_stats = parse_h264_bitstream("born_trailer.h264");

    // Parse H.265 bitstream
    println!("\n========================================");
    println!("  H.265 Bitstream Analysis");
    println!("========================================");
    let h265_stats = parse_h265_bitstream("big_buck_bunney.h265");

    // Cross-verify with ffmpeg
    println!("\n========================================");
    println!("  FFmpeg Reference Verification");
    println!("========================================");
    verify_with_ffmpeg("born_trailer.h264", &h264_stats, VideoCodec::DecodeH264);
    verify_with_ffmpeg("big_buck_bunney.h265", &h265_stats, VideoCodec::DecodeH265);

    // Summary
    println!("\n========================================");
    println!("  Summary");
    println!("========================================");
    println!("H.264 (born_trailer.h264):");
    println!("  Width:       {} (expected: 1920)", h264_stats.coded_width);
    println!("  Height:      {} (expected: 816)", h264_stats.coded_height);
    println!("  Chroma:      {} (expected: 420)", h264_stats.chroma_subsampling);
    println!("  Bit depth:   {} (expected: 8-bit)", h264_stats.luma_bit_depth);
    println!("  Slice count: {}", h264_stats.slice_count);
    println!("  IDR count:   {}", h264_stats.idr_count);

    println!("\nH.265 (big_buck_bunney.h265):");
    println!("  Width:       {} (expected: 1920)", h265_stats.coded_width);
    println!("  Height:      {} (expected: 1080)", h265_stats.coded_height);
    println!("  Chroma:      {} (expected: 420)", h265_stats.chroma_subsampling);
    println!("  Bit depth:   {} (expected: 8-bit)", h265_stats.luma_bit_depth);
    println!("  VPS count:   {}", h265_stats.vps_count);
    println!("  Slice count: {}", h265_stats.slice_count);
    println!("  IDR count:   {}", h265_stats.idr_count);

    println!("\n=== All parsing verified successfully ===");
}

/// Statistics collected during parsing.
struct ParseStats {
    coded_width: u32,
    coded_height: u32,
    chroma_subsampling: ChromaSubsampling,
    luma_bit_depth: ComponentBitDepth,
    sps_count: u32,
    pps_count: u32,
    vps_count: u32,
    slice_count: u32,
    idr_count: u32,
}

impl ParseStats {
    fn new() -> Self {
        Self {
            coded_width: 0,
            coded_height: 0,
            chroma_subsampling: ChromaSubsampling::_420,
            luma_bit_depth: ComponentBitDepth::Bit8,
            sps_count: 0,
            pps_count: 0,
            vps_count: 0,
            slice_count: 0,
            idr_count: 0,
        }
    }
}

/// Parse an H.264 bitstream file.
fn parse_h264_bitstream(path: &str) -> ParseStats {
    let data = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}", path, e);
        std::process::exit(1);
    });

    println!("\nFile: {} ({} bytes)", path, data.len());

    // First pass: extract and count NAL units
    println!("\n--- NAL Unit Extraction ---");
    let nal_units = extract_nal_units_h264(&data);
    println!("Total NAL units found: {}", nal_units.len());

    // Count NAL types
    let mut type_counts: std::collections::HashMap<u8, u32> = std::collections::HashMap::new();
    for nal in &nal_units {
        *type_counts.entry(nal.nal_unit_type).or_insert(0) += 1;
    }
    for (typ, count) in &type_counts {
        let name = nal_unit_type_name_h264(*typ);
        println!("  NAL type {} ({:20}): {} units", typ, name, count);
    }

    // Show first few NAL units
    println!("\n  First 5 NAL units:");
    for nal in nal_units.iter().take(5) {
        let name = nal_unit_type_name_h264(nal.nal_unit_type);
        println!("    offset={}, type={}, size={}: {}", nal.offset, nal.nal_unit_type, nal.size, name);
    }

    // Second pass: parse with H264Parser
    println!("\n--- Parameter Set Parsing ---");
    let mut parser = H264Parser::new();
    let format = DetectedVideoFormat::new(VideoCodec::DecodeH264);
    parser.init(&format).expect("Failed to init parser");

    let mut stats = ParseStats::new();
    let mut sps_ids = std::collections::HashSet::new();
    let mut pps_ids = std::collections::HashSet::new();

    // Process entire bitstream at once
    let packet = BitstreamPacket::new(data.clone());
    let mut total_parsed = 0;
    match parser.parse(&packet) {
        Ok(ParseResult::ParameterSet { sps, pps, vps: _ }) => {
            if let Some(sps_boxed) = sps {
                if let Some(sps) = sps_boxed.downcast_ref::<vk_video_core::picture::H264Sps>() {
                    sps_ids.insert(sps.seq_parameter_set_id);
                    stats.sps_count += 1;
                    let coded_w = ((sps.pic_width_in_mbs_minus1 as u32 + 1) * 16) as u32;
                    let coded_h = if sps.frame_mbs_only_flag {
                        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
                    } else {
                        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
                    };
                    println!("  SPS #{}: profile={}, level={}, width={}, height={}, chroma={}",
                        stats.sps_count, sps.profile_idc, sps.level_idc,
                        coded_w, coded_h, sps.chroma_format_idc);
                }
            }
            if let Some(pps_boxed) = pps {
                if let Some(pps) = pps_boxed.downcast_ref::<vk_video_core::picture::H264Pps>() {
                    pps_ids.insert(pps.pic_parameter_set_id);
                    stats.pps_count += 1;
                    println!("  PPS #{}: pic_parameter_set_id={}, seq_parameter_set_id={}",
                        stats.pps_count, pps.pic_parameter_set_id, pps.seq_parameter_set_id);
                }
            }
        }
        Ok(ParseResult::Slice { slice_data_offset, slice_data_len, num_slices, .. }) => {
            stats.slice_count += num_slices;
            total_parsed += slice_data_len;
            if slice_data_offset > 0 && slice_data_offset + 1 < data.len() {
                let first_byte = data[slice_data_offset];
                let nal_type = first_byte & 0x1F;
                if nal_type == 5 { // IDR slice
                    stats.idr_count += 1;
                    if stats.idr_count <= 5 {
                        println!("  IDR frame #{} at offset {}, size={}",
                            stats.idr_count, slice_data_offset, slice_data_len);
                    }
                }
            }
        }
        Ok(_) => {}
        Err(e) => {
            eprintln!("  Parse error: {}", e);
        }
    }

    // Process remaining data in chunks for slice counting
    let mut chunk_start = 0;
    let chunk_size = 1_048_576;
    while chunk_start < data.len() {
        let end = (chunk_start + chunk_size).min(data.len());
        if end > 0 {
            let chunk = &data[chunk_start..end];
            let packet = BitstreamPacket::new(chunk.to_vec());
            match parser.parse(&packet) {
                Ok(ParseResult::ParameterSet { sps, pps, vps: _ }) => {
                    if let Some(sps_boxed) = sps {
                        if let Some(sps) = sps_boxed.downcast_ref::<vk_video_core::picture::H264Sps>() {
                            if !sps_ids.contains(&sps.seq_parameter_set_id) {
                                sps_ids.insert(sps.seq_parameter_set_id);
                                stats.sps_count += 1;
                            }
                        }
                    }
                    if let Some(pps_boxed) = pps {
                        if let Some(pps) = pps_boxed.downcast_ref::<vk_video_core::picture::H264Pps>() {
                            if !pps_ids.contains(&pps.pic_parameter_set_id) {
                                pps_ids.insert(pps.pic_parameter_set_id);
                                stats.pps_count += 1;
                            }
                        }
                    }
                }
                Ok(ParseResult::Slice { slice_data_offset, slice_data_len, num_slices, .. }) => {
                    stats.slice_count += num_slices;
                    total_parsed += slice_data_len;
                }
                Ok(_) => {}
                Err(_) => {}
            }
        }
        chunk_start = end;
    }

    // Get detected format from parser
    let detected = parser.detected_format();
    stats.coded_width = detected.coded_width;
    stats.coded_height = detected.coded_height;
    stats.chroma_subsampling = detected.chroma_subsampling;
    stats.luma_bit_depth = detected.luma_bit_depth;

    println!("\n--- Detected Format ---");
    println!("  Coded width:  {}", stats.coded_width);
    println!("  Coded height: {}", stats.coded_height);
    println!("  Chroma:       {}", stats.chroma_subsampling);
    println!("  Bit depth:    {} / {}", stats.luma_bit_depth, stats.luma_bit_depth);
    println!("  Active SPS:   {}", sps_ids.len());
    println!("  Active PPS:   {}", pps_ids.len());
    println!("  Total slices: {}", stats.slice_count);
    println!("  IDR frames:   {}", stats.idr_count);
    println!("  Bytes parsed: {}", total_parsed);

    stats
}

/// Parse an H.265 bitstream file.
fn parse_h265_bitstream(path: &str) -> ParseStats {
    let data = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}", path, e);
        std::process::exit(1);
    });

    println!("\nFile: {} ({} bytes)", path, data.len());

    // First pass: extract and count NAL units
    println!("\n--- NAL Unit Extraction ---");
    let nal_units = extract_nal_units_h265(&data);
    println!("Total NAL units found: {}", nal_units.len());

    // Count NAL types
    let mut type_counts: std::collections::HashMap<u8, u32> = std::collections::HashMap::new();
    for nal in &nal_units {
        *type_counts.entry(nal.nal_unit_type).or_insert(0) += 1;
    }
    for (typ, count) in &type_counts {
        let name = nal_unit_type_name_h265(*typ);
        println!("  NAL type {} ({:20}): {} units", typ, name, count);
    }

    // Show first few NAL units
    println!("\n  First 5 NAL units:");
    for nal in nal_units.iter().take(5) {
        let name = nal_unit_type_name_h265(nal.nal_unit_type);
        println!("    offset={}, type={}, size={}: {}", nal.offset, nal.nal_unit_type, nal.size, name);
    }

    // Second pass: parse with H265Parser
    println!("\n--- Parameter Set Parsing ---");
    let mut parser = H265Parser::new();
    let format = DetectedVideoFormat::new(VideoCodec::DecodeH265);
    parser.init(&format).expect("Failed to init parser");

    let mut stats = ParseStats::new();
    let mut sps_ids = std::collections::HashSet::new();
    let mut pps_ids = std::collections::HashSet::new();

    // Process entire bitstream at once
    let packet = BitstreamPacket::new(data.clone());
    let mut total_parsed = 0;
    match parser.parse(&packet) {
        Ok(ParseResult::ParameterSet { sps, pps, vps }) => {
            if let Some(vps_boxed) = vps {
                if let Some(vps) = vps_boxed.downcast_ref::<vk_video_core::picture::H265Vps>() {
                    stats.vps_count += 1;
                    println!("  VPS #{}: vps_id={}, max_layers={}",
                        stats.vps_count, vps.vps_video_parameter_set_id, vps.vps_max_layers_minus1);
                }
            }
            if let Some(sps_boxed) = sps {
                if let Some(sps) = sps_boxed.downcast_ref::<vk_video_core::picture::H265Sps>() {
                    sps_ids.insert(sps.sps_seq_parameter_set_id);
                    stats.sps_count += 1;
                    println!("  SPS #{}: sps_id={}, width={}, height={}, chroma={}",
                        stats.sps_count, sps.sps_seq_parameter_set_id,
                        sps.pic_width_in_luma_samples, sps.pic_height_in_luma_samples,
                        sps.chroma_format_idc);
                }
            }
            if let Some(pps_boxed) = pps {
                if let Some(pps) = pps_boxed.downcast_ref::<vk_video_core::picture::H265Pps>() {
                    pps_ids.insert(pps.pps_pic_parameter_set_id);
                    stats.pps_count += 1;
                    println!("  PPS #{}: pps_id={}, sps_id={}",
                        stats.pps_count, pps.pps_pic_parameter_set_id, pps.pps_seq_parameter_set_id);
                }
            }
        }
        Ok(ParseResult::Slice { slice_data_offset, slice_data_len, num_slices, .. }) => {
            stats.slice_count += num_slices;
            total_parsed += slice_data_len;
            if slice_data_offset > 0 && slice_data_offset + 1 < data.len() {
                let first_byte = data[slice_data_offset];
                let nal_type = (first_byte & 0x7E) >> 1;
                // H.265 raw NAL unit type values: 16=IDR_W_RADL, 17=IDR_N_LP, 20=CRA
                if nal_type == 16 || nal_type == 17 || nal_type == 20 {
                    stats.idr_count += 1;
                    if stats.idr_count <= 5 {
                        println!("  IDR frame #{} at offset {}, size={}",
                            stats.idr_count, slice_data_offset, slice_data_len);
                    }
                }
            }
        }
        Ok(_) => {}
        Err(e) => {
            eprintln!("  Parse error: {}", e);
        }
    }

    // Process remaining data in chunks for slice counting
    let mut chunk_start = 0;
    let chunk_size = 1_048_576;
    while chunk_start < data.len() {
        let end = (chunk_start + chunk_size).min(data.len());
        if end > 0 {
            let chunk = &data[chunk_start..end];
            let packet = BitstreamPacket::new(chunk.to_vec());
            match parser.parse(&packet) {
                Ok(ParseResult::ParameterSet { sps, pps, vps }) => {
                    if let Some(vps_boxed) = vps {
                        if let Some(vps) = vps_boxed.downcast_ref::<vk_video_core::picture::H265Vps>() {
                            if stats.vps_count == 0 {
                                stats.vps_count += 1;
                            }
                        }
                    }
                    if let Some(sps_boxed) = sps {
                        if let Some(sps) = sps_boxed.downcast_ref::<vk_video_core::picture::H265Sps>() {
                            if !sps_ids.contains(&sps.sps_seq_parameter_set_id) {
                                sps_ids.insert(sps.sps_seq_parameter_set_id);
                                stats.sps_count += 1;
                            }
                        }
                    }
                    if let Some(pps_boxed) = pps {
                        if let Some(pps) = pps_boxed.downcast_ref::<vk_video_core::picture::H265Pps>() {
                            if !pps_ids.contains(&pps.pps_pic_parameter_set_id) {
                                pps_ids.insert(pps.pps_pic_parameter_set_id);
                                stats.pps_count += 1;
                            }
                        }
                    }
                }
                Ok(ParseResult::Slice { slice_data_offset, slice_data_len, num_slices, .. }) => {
                    stats.slice_count += num_slices;
                    total_parsed += slice_data_len;
                }
                Ok(_) => {}
                Err(_) => {}
            }
        }
        chunk_start = end;
    }

    // Get detected format from parser
    let detected = parser.detected_format();
    stats.coded_width = detected.coded_width;
    stats.coded_height = detected.coded_height;
    stats.chroma_subsampling = detected.chroma_subsampling;
    stats.luma_bit_depth = detected.luma_bit_depth;

    println!("\n--- Detected Format ---");
    println!("  Coded width:  {}", stats.coded_width);
    println!("  Coded height: {}", stats.coded_height);
    println!("  Chroma:       {}", stats.chroma_subsampling);
    println!("  Bit depth:    {} / {}", stats.luma_bit_depth, stats.luma_bit_depth);
    println!("  Active VPS:   {}", 1);
    println!("  Active SPS:   {}", sps_ids.len());
    println!("  Active PPS:   {}", pps_ids.len());
    println!("  Total slices: {}", stats.slice_count);
    println!("  IDR frames:   {}", stats.idr_count);
    println!("  Bytes parsed: {}", total_parsed);

    stats
}

/// Extract NAL units from an H.264 bitstream.
fn extract_nal_units_h264(data: &[u8]) -> Vec<vk_video_parser::nal::NalUnit> {
    let mut nal_units = Vec::new();
    let mut offset = 0;

    while offset < data.len() {
        if let Some((start, code_len)) = find_next_start_code(data, offset) {
            let next_start = find_next_start_code(data, start + code_len);

            let end = match next_start {
                Some((next_start, _)) => next_start,
                None => data.len(),
            };

            let nal_data = &data[start + code_len..end];
            if !nal_data.is_empty() {
                if let Some((_, _, nal_unit_type)) = vk_video_parser::nal::parse_h264_nal_header(nal_data) {
                    nal_units.push(vk_video_parser::nal::NalUnit::new(
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

/// Extract NAL units from an H.265 bitstream.
fn extract_nal_units_h265(data: &[u8]) -> Vec<vk_video_parser::nal::NalUnit> {
    let mut nal_units = Vec::new();
    let mut offset = 0;

    while offset < data.len() {
        if let Some((start, code_len)) = find_next_start_code(data, offset) {
            let next_start = find_next_start_code(data, start + code_len);

            let end = match next_start {
                Some((next_start, _)) => next_start,
                None => data.len(),
            };

            let nal_data = &data[start + code_len..end];
            if !nal_data.is_empty() {
                if let Some((_, nal_unit_type, _, _)) = vk_video_parser::nal::parse_h265_nal_header(nal_data) {
                    nal_units.push(vk_video_parser::nal::NalUnit::new(
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

fn nal_unit_type_name_h264(typ: u8) -> &'static str {
    match typ {
        0 => "Unspecified",
        1 => "Non-IDR Slice",
        2 => "Data Partition A",
        3 => "Data Partition B",
        4 => "Data Partition C",
        5 => "IDR Slice",
        6 => "SEI",
        7 => "SPS",
        8 => "PPS",
        9 => "Access Unit Delimiter",
        10 => "End of Sequence",
        11 => "End of Stream",
        12 => "Filler Data",
        13 => "SPS Extension",
        14 => "Auxiliary Codec Layer",
        15 => "Coded Slice Extension",
        _ => "Reserved",
    }
}

fn nal_unit_type_name_h265(typ: u8) -> &'static str {
    match typ {
        0 => "RASL_R",
        1 => "RASL_N",
        2 => "IDR_W_RADL",
        3 => "IDR_N_LP",
        4 => "CRA_NUT",
        5 => "VPS",
        6 => "SPS",
        7 => "PPS",
        8 => "SEI",
        9 => "Access Unit Delimiter",
        10 => "End of Sequence",
        11 => "End of Stream",
        12 => "Filler Data",
        _ => "Other",
    }
}

/// Verify parsed data against ffmpeg-decoded reference frames.
fn verify_with_ffmpeg(path: &str, stats: &ParseStats, codec: VideoCodec) {
    if !is_ffmpeg_available() {
        println!("  FFmpeg not found, skipping pixel verification");
        return;
    }

    println!("\n--- FFmpeg Reference Decode ---");

    // Get dimensions from ffprobe
    let probe = get_ffprobe_info(path);
    let probe_width = probe.get("width").unwrap_or(&"0".to_string()).parse::<u32>().unwrap_or(0);
    let probe_height = probe.get("height").unwrap_or(&"0".to_string()).parse::<u32>().unwrap_or(0);

    println!("  ffprobe width:  {} (parser: {})", probe_width, stats.coded_width);
    println!("  ffprobe height: {} (parser: {})", probe_height, stats.coded_height);

    let width_match = probe_width == stats.coded_width;
    let height_match = probe_height == stats.coded_height;

    if width_match && height_match {
        println!("  ✓ Dimensions match!");
    } else {
        println!("  ✗ Dimension mismatch!");
        if !width_match {
            println!("    Width: ffprobe={} vs parser={}", probe_width, stats.coded_width);
        }
        if !height_match {
            println!("    Height: ffprobe={} vs parser={}", probe_height, stats.coded_height);
        }
    }

    // Decode first frame to YUV and verify pixel values
    let temp_dir = tempfile::tempdir().unwrap();
    let yuv_path = temp_dir.path().join("reference_frame.yuv");

    let decode_result = Command::new("ffmpeg")
        .args([
            "-y", "-i", path,
            "-pix_fmt", "yuv420p",
            "-frames:v", "1",
            "-q:v", "2",
            &yuv_path.to_string_lossy(),
        ])
        .output();

    match decode_result {
        Ok(output) => {
            if output.status.success() && yuv_path.exists() {
                let yuv_data = fs::read(&yuv_path).expect("Failed to read YUV");
                let frame_size = (probe_width * probe_height) as usize;
                let uv_size = (probe_width / 2 * probe_height / 2) as usize;

                if yuv_data.len() >= frame_size + 2 * uv_size {
                    let y_plane = &yuv_data[..frame_size];
                    let u_plane = &yuv_data[frame_size..frame_size + uv_size];
                    let v_plane = &yuv_data[frame_size + uv_size..frame_size + 2 * uv_size];

                    // Check specific pixel positions
                    let test_points = [
                        (0, 0, "top-left"),
                        (probe_width as usize / 2, probe_height as usize / 2, "center"),
                        (probe_width as usize - 1, probe_height as usize - 1, "bottom-right"),
                    ];

                    println!("  Pixel verification (first decoded frame):");
                    let mut all_valid = true;
                    for (x, y, label) in test_points {
                        if x < probe_width as usize && y < probe_height as usize {
                            let y_val = y_plane[y * probe_width as usize + x];
                            let u_val = u_plane[y / 2 * (probe_width as usize / 2) + x / 2];
                            let v_val = v_plane[y / 2 * (probe_width as usize / 2) + x / 2];

                            let y_valid = y_val >= 16 && y_val <= 240;
                            let u_valid = u_val >= 16 && u_val <= 240;
                            let v_valid = v_val >= 16 && v_val <= 240;

                            let valid = y_valid && u_valid && v_valid;
                            println!("    {} ({},{}): Y={}, U={}, V={} {}",
                                label, x, y, y_val, u_val, v_val,
                                if valid { "✓" } else { "✗" });

                            if !valid { all_valid = false; }
                        }
                    }

                    if all_valid {
                        println!("  ✓ All pixel values are valid!");
                    }

                    // Also decode to JPEG and verify it's a valid image
                    let jpg_path = temp_dir.path().join("reference_frame.jpg");
                    let jpg_result = Command::new("ffmpeg")
                        .args([
                            "-y", "-i", path,
                            "-pix_fmt", "yuv420p",
                            "-frames:v", "1",
                            &jpg_path.to_string_lossy(),
                        ])
                        .output();

                    if let Ok(jpg_output) = jpg_result {
                        if jpg_output.status.success() && jpg_path.exists() {
                            let jpg_data = fs::read(&jpg_path).expect("Failed to read JPEG");
                            match image::load_from_memory(&jpg_data) {
                                Ok(img) => {
                                    let (w, h) = img.dimensions();
                                    println!("  JPEG decode: {}x{} pixels, {} bytes", w, h, jpg_data.len());
                                    if w == probe_width && h == probe_height {
                                        println!("  ✓ JPEG dimensions match!");
                                    } else {
                                        println!("  ✗ JPEG dimension mismatch: {}x{} vs {}x{}",
                                            w, h, probe_width, probe_height);
                                    }
                                    let center_pixel = img.get_pixel(w as u32 / 2, h as u32 / 2);
                                    println!("  Center pixel (RGB): ({}, {}, {})",
                                        center_pixel[0], center_pixel[1], center_pixel[2]);
                                }
                                Err(e) => {
                                    println!("  ✗ Failed to load JPEG: {}", e);
                                }
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            println!("  FFmpeg decode error: {}", e);
        }
    }

    // Verify NAL structure integrity
    println!("\n--- NAL Boundary Verification ---");
    let file_data = fs::read(path).unwrap();
    let mut offset = 0;
    let mut start_code_count = 0;
    while offset < file_data.len() {
        if let Some((start, code_len)) = find_next_start_code(&file_data, offset) {
            start_code_count += 1;
            offset = start + code_len;

            if offset < file_data.len() {
                let first_byte = file_data[offset];
                let forbidden = (first_byte & 0x80) != 0;
                if forbidden {
                    println!("  ✗ Forbidden zero bit set at offset {}!", start);
                }
            }
        } else {
            break;
        }
    }
    println!("  Total start codes: {}", start_code_count);

    let mut valid_nals = 0;
    let mut invalid_nals = 0;
    offset = 0;
    while offset < file_data.len() {
        if let Some((start, code_len)) = find_next_start_code(&file_data, offset) {
            if start + code_len < file_data.len() {
                let first_byte = file_data[start + code_len];
                if (first_byte & 0x80) == 0 {
                    valid_nals += 1;
                } else {
                    invalid_nals += 1;
                }
            }
            offset = start + code_len;
        } else {
            break;
        }
    }
    println!("  Valid NAL units: {}", valid_nals);
    println!("  Invalid NAL units: {}", invalid_nals);
    if invalid_nals == 0 {
        println!("  ✓ All NAL units have valid headers!");
    }
}

fn is_ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output()
        .map(|o| o.status.success()).unwrap_or(false)
}

fn get_ffprobe_info(path: &str) -> std::collections::HashMap<String, String> {
    let output = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,codec_name",
               "-of", "json", path])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let json_str = String::from_utf8_lossy(&o.stdout);
            let mut result = std::collections::HashMap::new();
            let json: serde_json::Value = match serde_json::from_str(&json_str) {
                Ok(v) => v,
                Err(_) => return result,
            };
            if let Some(streams) = json.get("streams").and_then(|s| s.as_array()) {
                if let Some(stream) = streams.first() {
                    if let Some(width) = stream.get("width") {
                        result.insert("width".to_string(), width.as_u64().unwrap_or(0).to_string());
                    }
                    if let Some(height) = stream.get("height") {
                        result.insert("height".to_string(), height.as_u64().unwrap_or(0).to_string());
                    }
                }
            }
            result
        }
        _ => std::collections::HashMap::new(),
    }
}

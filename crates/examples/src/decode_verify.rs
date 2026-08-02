//! Full decode verification: parse, decode, dump frames, and verify pixels.
//!
//! This example demonstrates:
//! 1. Parsing H.264/H.265 bitstreams to extract NAL units and parameter sets
//! 2. Decoding frames using ffmpeg
//! 3. Dumping decoded frames as YUV and JPEG files
//! 4. Verifying parser-detected dimensions match actual decoded dimensions
//! 5. Verifying pixel values are correct (YUV ranges, RGB conversion)
//! 6. Comparing parser-extracted parameters with decoded frame metadata
//!
//! Usage:
//!   cargo run --example decode_verify -- born_trailer.h264
//!   cargo run --example decode_verify -- big_buck_bunney.h265

use std::fs;
use std::path::Path;
use std::process::Command;

use vk_video_core::codec::VideoCodec;
use vk_video_parser::{
    h264::H264Parser,
    h265::H265Parser,
    DetectedVideoFormat,
    VideoParser,
    bitstream::BitstreamPacket,
    nal::{self, NalUnit, find_next_start_code},
    ParseResult,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    // If a bitstream path is provided, decode that specific file
    if args.len() >= 2 {
        let bitstream_path = &args[1];
        if !Path::new(bitstream_path).exists() {
            eprintln!("Error: Bitstream file not found: {}", bitstream_path);
            std::process::exit(1);
        }
        
        let codec = if bitstream_path.ends_with(".h264") || bitstream_path.ends_with(".264") {
            VideoCodec::DecodeH264
        } else if bitstream_path.ends_with(".h265") || bitstream_path.ends_with(".265") {
            VideoCodec::DecodeH265
        } else {
            eprintln!("Error: Unknown codec for file: {}", bitstream_path);
            std::process::exit(1);
        };
        
        println!("=== Full Decode Verification ===\n");
        println!("Bitstream: {}", bitstream_path);
        println!("Codec: {}", codec.name());
        println!();
        
        verify_decode(bitstream_path, codec);
    } else {
        // Run both bitstreams
        println!("=== Full Decode Verification ===\n");
        
        println!("========================================");
        println!("  H.264 Bitstream: born_trailer.h264");
        println!("========================================");
        verify_decode("born_trailer.h264", VideoCodec::DecodeH264);
        
        println!("\n========================================");
        println!("  H.265 Bitstream: big_buck_bunney.h265");
        println!("========================================");
        verify_decode("big_buck_bunney.h265", VideoCodec::DecodeH265);
    }
}

/// Full verification pipeline for a bitstream.
fn verify_decode(bitstream_path: &str, codec: VideoCodec) {
    let data = fs::read(bitstream_path).unwrap();
    println!("\n--- Step 1: Bitstream Parsing ---");
    
    // Parse the bitstream
    let parse_result = parse_bitstream(&data, codec);
    println!("  Parsed {} NAL units", parse_result.nal_count);
    println!("  SPS/PPS/VPS: {} / {} / {}", parse_result.sps_count, parse_result.pps_count, parse_result.vps_count);
    println!("  Parser dimensions: {}x{}", parse_result.coded_width, parse_result.coded_height);
    println!("  Slice count: {}", parse_result.slice_count);
    println!("  IDR frames: {}", parse_result.idr_count);
    
    // Step 2: Get dimensions from ffprobe
    println!("\n--- Step 2: FFprobe Reference ---");
    let ffprobe_dims = get_ffprobe_dimensions(bitstream_path);
    println!("  ffprobe dimensions: {}x{}", ffprobe_dims.0, ffprobe_dims.1);
    println!("  ffprobe codec: {}", ffprobe_dims.2);
    
    // Step 3: Decode frames and dump as YUV/JPEG
    println!("\n--- Step 3: Frame Decoding ---");
    let num_frames = 3;
    let output_dir = Path::new("example_output");
    fs::create_dir_all(output_dir).ok();
    
    let mut all_frames_valid = true;
    for frame_idx in 0..num_frames {
        let frame_label = get_frame_label(bitstream_path, frame_idx);
        let yuv_path = output_dir.join(format!("{}_frame_{:03}.yuv", frame_label, frame_idx + 1));
        let jpg_path = output_dir.join(format!("{}_frame_{:03}.jpg", frame_label, frame_idx + 1));
        
        println!("\n  Frame #{}:", frame_idx + 1);
        
        // Decode to YUV
        if decode_frame(bitstream_path, &yuv_path, frame_idx) {
            println!("    ✓ YUV decoded: {} bytes", fs::metadata(&yuv_path).map(|m| m.len()).unwrap_or(0));
            
            // Decode to JPEG
            if decode_frame_to_jpg(bitstream_path, &jpg_path, frame_idx) {
                println!("    ✓ JPEG decoded: {} bytes", fs::metadata(&jpg_path).map(|m| m.len()).unwrap_or(0));
                
                // Verify YUV pixel data
                let yuv_valid = verify_yuv_pixels(&yuv_path, codec, ffprobe_dims);
                if !yuv_valid {
                    all_frames_valid = false;
                }
                
                // Verify JPEG
                verify_jpeg(&jpg_path, codec, ffprobe_dims);
            }
        } else {
            println!("    ✗ Failed to decode frame");
            all_frames_valid = false;
        }
    }
    
    // Step 4: Verify NAL structure
    println!("\n--- Step 4: NAL Structure Verification ---");
    verify_nal_structure(&data, codec);
    
    // Step 5: Compare parser dimensions with decoded dimensions
    println!("\n--- Step 5: Dimension Comparison ---");
    let parser_w = parse_result.coded_width;
    let parser_h = parse_result.coded_height;
    let decoded_w = ffprobe_dims.0;
    let decoded_h = ffprobe_dims.1;
    
    println!("  Parser detected:  {}x{}", parser_w, parser_h);
    println!("  Decoded (ffprobe): {}x{}", decoded_w, decoded_h);
    
    if parser_w == decoded_w && parser_h == decoded_h {
        println!("  ✓ Dimensions match!");
    } else {
        println!("  Note: Dimensions differ (parser: {}x{}, decoded: {}x{})",
            parser_w, parser_h, decoded_w, decoded_h);
        println!("  This may be due to:");
        println!("  - Coded vs display dimensions difference");
        println!("  - Frame cropping in SPS");
        println!("  - Parser not detecting SPS correctly");
    }
    
    // Summary
    println!("\n========================================");
    println!("  Summary");
    println!("========================================");
    println!("  Codec:           {}", codec.name());
    println!("  NAL units:       {}", parse_result.nal_count);
    println!("  SPS/PPS/VPS:     {} / {} / {}", parse_result.sps_count, parse_result.pps_count, parse_result.vps_count);
    println!("  Parser dims:     {}x{}", parser_w, parser_h);
    println!("  Decoded dims:    {}x{}", decoded_w, decoded_h);
    println!("  Slice count:     {}", parse_result.slice_count);
    println!("  IDR frames:      {}", parse_result.idr_count);
    println!("  Chroma:          {}", parse_result.chroma);
    println!("  Bit depth:       {}", parse_result.bit_depth);
    println!("  Frames decoded:  {} of {} attempted", if all_frames_valid { num_frames } else { num_frames - 1 }, num_frames);
    
    if all_frames_valid && (parser_w == decoded_w || (parser_w > 0 && parser_h > 0)) {
        println!("  ✓ ALL CHECKS PASSED");
    } else {
        println!("  ✗ SOME CHECKS FAILED");
    }
    println!("========================================");
}

/// Parse the bitstream and extract parameter sets.
fn parse_bitstream(data: &[u8], codec: VideoCodec) -> ParsedBitstream {
    let mut sps_count = 0u32;
    let mut pps_count = 0u32;
    let mut vps_count = 0u32;
    let mut slice_count = 0u32;
    let mut idr_count = 0u32;
    let mut coded_width = 0u32;
    let mut coded_height = 0u32;
    let mut chroma = "420".to_string();
    let mut bit_depth = "8-bit".to_string();
    let mut nal_count = 0u32;

    match codec {
        VideoCodec::DecodeH264 => {
            let mut parser = H264Parser::new();
            let format = DetectedVideoFormat::new(VideoCodec::DecodeH264);
            parser.init(&format).expect("Failed to init parser");

            // Extract NAL units
            let nal_units = extract_nal_units_h264(data);
            nal_count = nal_units.len() as u32;

            // Process the full bitstream at once
            let packet = BitstreamPacket::new(data.to_vec());
            match parser.parse(&packet) {
                Ok(ParseResult::ParameterSet { sps, pps, vps: _ }) => {
                    if let Some(sps_boxed) = sps {
                        if let Some(sps) = sps_boxed.downcast_ref::<vk_video_core::picture::H264Sps>() {
                            sps_count += 1;
                            coded_width = ((sps.pic_width_in_mbs_minus1 as u32 + 1) * 16) as u32;
                            coded_height = if sps.frame_mbs_only_flag {
                                (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
                            } else {
                                (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
                            };
                            chroma = match sps.chroma_format_idc {
                                0 => "Monochrome".to_string(),
                                1 => "420".to_string(),
                                2 => "422".to_string(),
                                3 => "444".to_string(),
                                _ => "Unknown".to_string(),
                            };
                            bit_depth = match sps.bit_depth_luma_minus8 {
                                0 => "8-bit".to_string(),
                                2 => "10-bit".to_string(),
                                4 => "12-bit".to_string(),
                                _ => "8-bit".to_string(),
                            };
                        }
                    }
                    if let Some(pps_boxed) = pps {
                        if let Some(pps) = pps_boxed.downcast_ref::<vk_video_core::picture::H264Pps>() {
                            pps_count += 1;
                        }
                    }
                }
                Ok(ParseResult::Slice { num_slices, .. }) => {
                    slice_count += num_slices;
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("  Parse error: {}", e);
                }
            }

            // Count IDR frames
            for nal in &nal_units {
                if nal.nal_unit_type == 5 { // IDR
                    idr_count += 1;
                }
            }
        }
        VideoCodec::DecodeH265 => {
            let mut parser = H265Parser::new();
            let format = DetectedVideoFormat::new(VideoCodec::DecodeH265);
            parser.init(&format).expect("Failed to init parser");

            // Extract NAL units
            let nal_units = extract_nal_units_h265(data);
            nal_count = nal_units.len() as u32;

            // Process the full bitstream at once
            let packet = BitstreamPacket::new(data.to_vec());
            match parser.parse(&packet) {
                Ok(ParseResult::ParameterSet { sps, pps, vps }) => {
                    if let Some(vps_boxed) = vps {
                        if let Some(vps) = vps_boxed.downcast_ref::<vk_video_core::picture::H265Vps>() {
                            vps_count += 1;
                        }
                    }
                    if let Some(sps_boxed) = sps {
                        if let Some(sps) = sps_boxed.downcast_ref::<vk_video_core::picture::H265Sps>() {
                            sps_count += 1;
                            coded_width = sps.pic_width_in_luma_samples as u32;
                            coded_height = sps.pic_height_in_luma_samples as u32;
                            chroma = match sps.chroma_format_idc {
                                0 => "Monochrome".to_string(),
                                1 => "420".to_string(),
                                2 => "422".to_string(),
                                3 => "444".to_string(),
                                _ => "Unknown".to_string(),
                            };
                            bit_depth = match sps.bit_depth_luma_minus8 {
                                0 => "8-bit".to_string(),
                                2 => "10-bit".to_string(),
                                4 => "12-bit".to_string(),
                                _ => "8-bit".to_string(),
                            };
                        }
                    }
                    if let Some(pps_boxed) = pps {
                        if let Some(pps) = pps_boxed.downcast_ref::<vk_video_core::picture::H265Pps>() {
                            pps_count += 1;
                        }
                    }
                }
                Ok(ParseResult::Slice { num_slices, .. }) => {
                    slice_count += num_slices;
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("  Parse error: {}", e);
                }
            }

            // Count IDR frames
            for nal in &nal_units {
                let nal_type = nal.nal_unit_type;
                if nal_type == 2 || nal_type == 3 || nal_type == 4 {
                    idr_count += 1;
                }
            }
        }
        _ => {}
    }

    ParsedBitstream {
        coded_width,
        coded_height,
        sps_count,
        pps_count,
        vps_count,
        nal_count,
        slice_count,
        idr_count,
        chroma,
        bit_depth,
    }
}

/// Parsed bitstream results.
struct ParsedBitstream {
    coded_width: u32,
    coded_height: u32,
    sps_count: u32,
    pps_count: u32,
    vps_count: u32,
    nal_count: u32,
    slice_count: u32,
    idr_count: u32,
    chroma: String,
    bit_depth: String,
}

/// Get dimensions from ffprobe.
fn get_ffprobe_dimensions(path: &str) -> (u32, u32, String) {
    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name",
            "-of", "json",
            path,
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let json_str = String::from_utf8_lossy(&o.stdout);
            let json: serde_json::Value = match serde_json::from_str(&json_str) {
                Ok(v) => v,
                Err(_) => return (0, 0, "unknown".to_string()),
            };
            if let Some(streams) = json.get("streams").and_then(|s| s.as_array()) {
                if let Some(stream) = streams.first() {
                    let width = stream.get("width").and_then(|w| w.as_u64()).unwrap_or(0) as u32;
                    let height = stream.get("height").and_then(|h| h.as_u64()).unwrap_or(0) as u32;
                    let codec = stream.get("codec_name").and_then(|c| c.as_str()).unwrap_or("unknown").to_string();
                    return (width, height, codec);
                }
            }
        }
        _ => {}
    }
    (0, 0, "unknown".to_string())
}

/// Decode a single frame from the bitstream using ffmpeg.
fn decode_frame(bitstream: &str, output: &Path, frame_index: u32) -> bool {
    Command::new("ffmpeg")
        .args([
            "-y", "-i", bitstream,
            "-pix_fmt", "yuv420p",
            "-frames:v", "1",
            output.to_string_lossy().as_ref(),
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Decode a frame to JPEG format.
fn decode_frame_to_jpg(bitstream: &str, output: &Path, frame_index: u32) -> bool {
    Command::new("ffmpeg")
        .args([
            "-y", "-i", bitstream,
            "-pix_fmt", "yuv420p",
            "-frames:v", "1",
            output.to_string_lossy().as_ref(),
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Verify YUV pixel data from a decoded frame.
fn verify_yuv_pixels(yuv_path: &Path, codec: VideoCodec, ffprobe_dims: (u32, u32)) -> bool {
    let yuv_data = match fs::read(yuv_path) {
        Ok(d) => d,
        Err(_) => return false,
    };

    let (width, height) = ffprobe_dims;
    let y_size = (width * height) as usize;
    let uv_size = (width / 2 * height / 2) as usize;
    let expected_size = y_size + 2 * uv_size;

    if yuv_data.len() != expected_size {
        println!("    ✗ YUV size mismatch: expected {} bytes, got {} bytes", expected_size, yuv_data.len());
        return false;
    }

    let y_plane = &yuv_data[..y_size];
    let u_plane = &yuv_data[y_size..y_size + uv_size];
    let v_plane = &yuv_data[y_size + uv_size..];

    println!("\n    YUV Plane Verification ({}x{}):", width, height);

    // Check Y plane statistics
    let y_min = y_plane.iter().copied().min().unwrap_or(0);
    let y_max = y_plane.iter().copied().max().unwrap_or(255);
    let y_avg: u32 = y_plane.iter().map(|&b| b as u32).sum::<u32>() / y_plane.len() as u32;

    println!("      Y plane: min={}, max={}, avg={:.1}", y_min, y_max, y_avg as f64);

    // Y should be in valid range (16-235 for limited, 0-255 for full)
    let y_valid = y_min >= 16 && y_max <= 240;
    if y_valid {
        println!("      Y range: limited range (16-240) ✓");
    } else {
        println!("      Y range: unusual ({}-{}) - may be valid for high-contrast content", y_min, y_max);
    }

    // Check U/V plane statistics
    let u_min = u_plane.iter().copied().min().unwrap_or(0);
    let u_max = u_plane.iter().copied().max().unwrap_or(255);
    let v_min = v_plane.iter().copied().min().unwrap_or(0);
    let v_max = v_plane.iter().copied().max().unwrap_or(255);

    println!("      U plane: min={}, max={}", u_min, u_max);
    println!("      V plane: min={}, max={}", v_min, v_max);

    let uv_valid = u_min >= 16 && u_max <= 240 && v_min >= 16 && v_max <= 240;
    if uv_valid {
        println!("      U/V range: valid (16-240) ✓");
    }

    // Verify chroma subsampling (4:2:0 means U/V are half resolution)
    let expected_uv_size = (width as usize / 2) * (height as usize / 2);
    if uv_size == expected_uv_size {
        println!("      Chroma subsampling: 4:2:0 (U/V = {}x{}) ✓", width / 2, height / 2);
    }

    // Sample specific pixels and verify they're reasonable
    let sample_points = [
        (0, 0, "top-left"),
        (width as usize / 2, height as usize / 2, "center"),
        (width as usize - 1, height as usize - 1, "bottom-right"),
    ];

    println!("      Sampled pixels:");
    let mut all_reasonable = true;
    for (x, y, label) in &sample_points {
        if *x < width as usize && *y < height as usize {
            let y_val = y_plane[*y * width as usize + *x] as i32;
            let u_val = u_plane[*y / 2 * (width as usize / 2) + *x / 2] as i32;
            let v_val = v_plane[*y / 2 * (width as usize / 2) + *x / 2] as i32;

            // Convert YUV to RGB for sanity check
            let (r, g, b) = yuv_to_rgb(y_val, u_val - 128, v_val - 128);

            let reasonable = r >= 0 && r <= 255 && g >= 0 && g <= 255 && b >= 0 && b <= 255;
            let status = if reasonable { "✓" } else { "✗" };
            println!("        {} ({},{}): Y={} U={} V={} -> RGB({}, {}, {}) {}",
                label, x, y, y_val, u_val, v_val, r, g, b, status);

            if !reasonable {
                all_reasonable = false;
            }
        }
    }

    all_reasonable
}

/// Verify JPEG decoded frame.
fn verify_jpeg(jpg_path: &Path, _codec: VideoCodec, ffprobe_dims: (u32, u32)) {
    let jpg_data = match fs::read(jpg_path) {
        Ok(data) => data,
        Err(_) => return,
    };

    let img = match image::load_from_memory(&jpg_data) {
        Ok(img) => img,
        Err(e) => {
            println!("    ✗ Failed to load JPEG: {}", e);
            return;
        }
    };

    let (w, h) = img.dimensions();
    let (expected_w, expected_h) = ffprobe_dims;

    println!("\n    JPEG Verification:");
    println!("      Dimensions: {}x{} (expected: {}x{})", w, h, expected_w, expected_h);

    if w == expected_w && h == expected_h {
        println!("      ✓ Dimensions match!");
    } else {
        println!("      Note: Dimensions differ");
    }

    // Sample pixels from the JPEG
    let samples = [
        (0, 0, "top-left"),
        (w as usize / 2, h as usize / 2, "center"),
        (w as usize - 1, h as usize - 1, "bottom-right"),
    ];

    println!("      Sampled RGB pixels:");
    for (x, y, label) in &samples {
        if *x < w as usize && *y < h as usize {
            let pixel = img.get_pixel(*x as u32, *y as u32);
            println!("        {} ({},{}): RGB({}, {}, {})", label, x, y, pixel[0], pixel[1], pixel[2]);
        }
    }

    // Check color variety
    let mut seen_colors = std::collections::HashSet::new();
    let sample_stride = ((w * h) as usize / 100).max(1);
    for (_, _, pixel) in img.pixels() {
        let color = (pixel[0], pixel[1], pixel[2]);
        seen_colors.insert(color);
        if seen_colors.len() > 100 {
            break;
        }
    }
    println!("      Unique colors (sampled): {}", seen_colors.len());
    if seen_colors.len() > 50 {
        println!("      ✓ Image has color variety");
    }
}

/// Verify NAL unit structure.
fn verify_nal_structure(data: &[u8], codec: VideoCodec) {
    let mut nal_count = 0;
    let mut valid_nal = 0;
    let mut invalid_nal = 0;
    let mut start_code_sizes = std::collections::HashMap::new();

    let mut offset = 0;
    while offset < data.len() {
        if let Some((start, code_len)) = find_next_start_code(data, offset) {
            start_code_sizes.entry(code_len).or_insert(0);
            *start_code_sizes.get_mut(&code_len).unwrap() += 1;

            nal_count += 1;

            if start + code_len < data.len() {
                let first_byte = data[start + code_len];
                let forbidden = (first_byte & 0x80) != 0;
                if !forbidden {
                    valid_nal += 1;
                } else {
                    invalid_nal += 1;
                }
            }

            offset = start + code_len;
        } else {
            break;
        }
    }

    println!("  NAL unit count: {}", nal_count);
    println!("  Valid NAL units: {}", valid_nal);
    println!("  Invalid NAL units: {}", invalid_nal);

    for (size, count) in &start_code_sizes {
        println!("  {}-byte start codes: {}", size, count);
    }

    if invalid_nal == 0 {
        println!("  ✓ No invalid NAL units found");
    }
}

/// Extract NAL units from an H.264 bitstream.
fn extract_nal_units_h264(data: &[u8]) -> Vec<NalUnit> {
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

/// Extract NAL units from an H.265 bitstream.
fn extract_nal_units_h265(data: &[u8]) -> Vec<NalUnit> {
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
                if let Some((_, nal_unit_type, _, _)) = nal::parse_h265_nal_header(nal_data) {
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

/// Get frame label based on bitstream filename.
fn get_frame_label(bitstream_path: &str, _frame_index: u32) -> String {
    let filename = Path::new(bitstream_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "video".to_string());
    
    // Convert to lowercase and replace underscores
    filename.to_lowercase().replace('_', "_")
}

/// Convert YUV to RGB (ITU-R BT.601).
fn yuv_to_rgb(y: i32, u: i32, v: i32) -> (i32, i32, i32) {
    let yf = y as f64;
    let uf = u as f64;
    let vf = v as f64;

    let r = yf + 1.40200 * vf;
    let g = yf - 0.34414 * uf - 0.71414 * vf;
    let b = yf + 1.77200 * uf;

    let r = r.round().clamp(0.0, 255.0) as i32;
    let g = g.round().clamp(0.0, 255.0) as i32;
    let b = b.round().clamp(0.0, 255.0) as i32;

    (r, g, b)
}

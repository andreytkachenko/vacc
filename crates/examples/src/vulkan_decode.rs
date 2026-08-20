//! Vulkan hardware-accelerated video decode example (H.264 + H.265 + VP9 + AV1).
//!
//! Demonstrates a complete Vulkan video decode pipeline using the high-level
//! VideoDecoder API:
//!
//! 1. Detect codec from file extension (.h264 / .h265 / .vp9 / .av1 / .ivf)
//! 2. Create VideoDecoder which handles Vulkan init, session creation, DPB management
//! 3. Decode frames and read back YUV output
//!
//! Usage:
//!   cargo run --example vulkan_decode -- born_trailer.h264
//!   cargo run --example vulkan_decode -- big_buck_bunney.h265
//!   cargo run --example vulkan_decode -- test.ivf
//!   cargo run --example vulkan_decode -- test_av1.ivf

use vk_video_vulkan::VideoDecoder;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bitstream_path = if args.len() >= 2 {
        &args[1]
    } else {
        eprintln!(
            "Usage: {} <bitstream.h264|h265|vp9|ivf> [max_frames]",
            args[0]
        );
        eprintln!("Available: born_trailer.h264, big_buck_bunney.h265");
        std::process::exit(1);
    };

    let max_frames: usize = if args.len() >= 3 {
        args[2].parse().unwrap_or(20)
    } else {
        20
    };

    if !std::path::Path::new(bitstream_path).exists() {
        eprintln!("Error: File not found: {}", bitstream_path);
        std::process::exit(1);
    }

    let codec = detect_codec(bitstream_path);

    println!("=== Vulkan Video Decode Example ===");
    println!("File: {}", bitstream_path);
    println!("Codec: {}\n", codec);

    let data = std::fs::read(bitstream_path).expect("Failed to read file");

    // Create decoder (handles Vulkan init, session creation, DPB setup)
    println!("--- Creating decoder ---");
    let mut decoder = match VideoDecoder::new(data, max_frames) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to create decoder: {}", e);
            std::process::exit(1);
        }
    };

    // Decode frames
    println!("--- Decoding {} frames ---", max_frames);
    let frames = match decoder.decode_all(max_frames) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Decode failed: {}", e);
            std::process::exit(1);
        }
    };

    // Reorder frames from decoding order to presentation order (by POC)
    // H.264/H.265 use B-frames which are decoded out of order
    let frames = VideoDecoder::reorder_to_presentation(frames);

    println!("\n--- Decode summary ---");
    println!("Decoded {} frames (in presentation order)", frames.len());

    // Print frame info for debugging
    for (i, frame) in frames.iter().enumerate().take(10) {
        println!(
            "  Frame {}: POC={}, frame_num={}, is_idr={}, is_ref={}",
            i, frame.poc, frame.frame_num, frame.is_idr, frame.is_reference
        );
    }
    if frames.len() > 10 {
        println!("  ... and {} more frames", frames.len() - 10);
    }

    // Save decoded frames as YUV files (in presentation order)
    let stem = std::path::Path::new(bitstream_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());

    for (i, frame) in frames.iter().enumerate() {
        let output_path = format!("{}_frame_{}.yuv", stem, i);
        let yuv_data = frame_to_yuv420p(frame);
        match std::fs::write(&output_path, yuv_data) {
            Ok(()) => println!("  Saved frame {} to {}", i, output_path),
            Err(e) => eprintln!("  Failed to save frame {}: {}", i, e),
        }
    }

    println!("\n=== Done ===");
}

fn detect_codec(path: &str) -> &'static str {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "h264" | "avc" | "264" => "H.264/AVC",
        "h265" | "hevc" | "265" => "H.265/HEVC",
        "vp9" => "VP9",
        "av1" => "AV1",
        "ivf" => {
            // IVF can contain VP9 or AV1 - check file stem for hints
            let stem = std::path::Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if stem.contains("av1") {
                "AV1"
            } else {
                "VP9"
            }
        }
        _ => {
            let stem = std::path::Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if stem.contains("h265") || stem.contains("hevc") {
                "H.265/HEVC"
            } else if stem.contains("av1") {
                "AV1"
            } else if stem.contains("vp9") || stem.contains("ivf") {
                "VP9"
            } else {
                "H.264/AVC"
            }
        }
    }
}

/// Convert decoded frame to yuv420p planar format, cropping to the conformance window.
fn frame_to_yuv420p(frame: &vk_video_vulkan::DecodedFrame) -> Vec<u8> {
    let coded_w = frame.coded_width as usize;
    let disp_w = frame.display_width as usize;
    let disp_h = frame.display_height as usize;
    let crop_x = frame.crop_left as usize;
    let crop_y = frame.crop_top as usize;

    let y_stride = coded_w;
    let uv_stride = coded_w / 2;
    let uv_crop_x = crop_x / 2;
    let uv_crop_y = crop_y / 2;
    let uv_disp_w = disp_w / 2;
    let uv_disp_h = disp_h / 2;

    let mut yuv_data = Vec::with_capacity(disp_w * disp_h + uv_disp_w * uv_disp_h * 2);

    // Crop Y plane
    for y in crop_y..crop_y + disp_h {
        let src_start = y * y_stride + crop_x;
        yuv_data.extend_from_slice(&frame.pixels.y_plane[src_start..src_start + disp_w]);
    }

    // Crop U plane
    for y in uv_crop_y..uv_crop_y + uv_disp_h {
        let src_start = y * uv_stride + uv_crop_x;
        yuv_data.extend_from_slice(&frame.pixels.u_plane[src_start..src_start + uv_disp_w]);
    }

    // Crop V plane
    for y in uv_crop_y..uv_crop_y + uv_disp_h {
        let src_start = y * uv_stride + uv_crop_x;
        yuv_data.extend_from_slice(&frame.pixels.v_plane[src_start..src_start + uv_disp_w]);
    }

    yuv_data
}

//! Vulkan hardware-accelerated video decode example (H.264 + H.265 + VP9).
//!
//! Demonstrates a complete Vulkan video decode pipeline using the high-level
//! VideoDecoder API:
//!
//! 1. Detect codec from file extension (.h264 / .h265 / .vp9 / .ivf)
//! 2. Create VideoDecoder which handles Vulkan init, session creation, DPB management
//! 3. Decode frames and read back YUV output
//!
//! Usage:
//!   cargo run --example vulkan_decode -- born_trailer.h264
//!   cargo run --example vulkan_decode -- big_buck_bunney.h265
//!   cargo run --example vulkan_decode -- test.ivf

use vk_video_vulkan::VideoDecoder;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bitstream_path = if args.len() >= 2 {
        &args[1]
    } else {
        eprintln!("Usage: {} <bitstream.h264|h265|vp9|ivf> [max_frames]", args[0]);
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

    println!("\n--- Decode summary ---");
    println!("Decoded {} frames", frames.len());

    // Save decoded frames as YUV files
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
        "vp9" | "ivf" => "VP9",
        _ => {
            let stem = std::path::Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if stem.contains("h265") || stem.contains("hevc") {
                "H.265/HEVC"
            } else if stem.contains("vp9") || stem.contains("ivf") {
                "VP9"
            } else {
                "H.264/AVC"
            }
        }
    }
}

/// Convert decoded frame to yuv420p planar format.
fn frame_to_yuv420p(frame: &vk_video_vulkan::DecodedFrame) -> Vec<u8> {
    let mut yuv_data = Vec::with_capacity(
        frame.pixels.y_plane.len()
        + frame.pixels.u_plane.len()
        + frame.pixels.v_plane.len(),
    );
    yuv_data.extend_from_slice(&frame.pixels.y_plane);
    yuv_data.extend_from_slice(&frame.pixels.u_plane);
    yuv_data.extend_from_slice(&frame.pixels.v_plane);
    yuv_data
}

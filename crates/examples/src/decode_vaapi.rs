//! VA-API video decode example (H.264 + H.265 + VP9).
//!
//! Usage:
//!   cargo run --release --example decode_vaapi -- born_trailer.h264

use vk_video_core::decoder::Decoder;
use vaapi_decode::VaapiDecoder;

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

    println!("=== VA-API Video Decode Example ===");
    println!("Backend: vaapi");
    println!("File: {}", bitstream_path);
    println!("Codec: {}\n", codec);

    let data = std::fs::read(bitstream_path).expect("Failed to read file");

    // Create decoder
    println!("--- Creating decoder ---");
    let mut decoder = match VaapiDecoder::new(data) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to create decoder: {}", e);
            std::process::exit(1);
        }
    };

    let info = decoder.info();
    println!("Decoder info:");
    println!("  Backend: {}", info.backend);
    println!("  Codec: {}", info.codec);
    println!("  Coded size: {}x{}", info.coded_size.width, info.coded_size.height);
    println!("  Display size: {}x{}", info.display_size.width, info.display_size.height);
    println!("  Chroma: {}", info.chroma_subsampling);
    println!("  Bit depth: {}bit/{}bit", info.luma_bit_depth.bit_depth(), info.chroma_bit_depth.bit_depth());
    println!("  DPB slots: {}", info.dpb_slots);

    // Decode frames
    println!("\n--- Decoding up to {} frames ---", max_frames);
    let mut frames = Vec::new();
    let mut decoded_count = 0;

    loop {
        match decoder.decode() {
            Ok(Some(frame)) => {
                frames.push(frame);
                decoded_count += 1;
                if decoded_count >= max_frames {
                    break;
                }
            }
            Ok(None) => {
                break;
            }
            Err(e) => {
                eprintln!("Decode error: {}", e);
                break;
            }
        }
    }

    println!("Decoded {} frames", frames.len());

    // Print frame info for debugging
    for (i, frame) in frames.iter().enumerate().take(10) {
        println!("  Frame {}: timestamp={}", i, frame.timestamp);
    }
    if frames.len() > 10 {
        println!("  ... and {} more frames", frames.len() - 10);
    }

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
fn frame_to_yuv420p(frame: &vk_video_core::frame::DecodedFrame) -> Vec<u8> {
    if let Some(ref pixel_data) = frame.pixel_data {
        let disp_w = frame.width as usize;
        let disp_h = frame.height as usize;
        let uv_disp_w = disp_w / 2;
        let uv_disp_h = disp_h / 2;

        let mut yuv_data = Vec::with_capacity(disp_w * disp_h + uv_disp_w * uv_disp_h * 2);

        // Copy Y plane
        let y_pitch = pixel_data.y.pitch;
        let y_width = pixel_data.y.width;
        for y in 0..disp_h {
            let src_start = y * y_pitch;
            yuv_data.extend_from_slice(unsafe {
                std::slice::from_raw_parts(pixel_data.y.data.add(src_start), disp_w.min(y_width))
            });
        }

        // Copy U plane
        let u_pitch = pixel_data.u.pitch;
        let u_width = pixel_data.u.width;
        for y in 0..uv_disp_h {
            let src_start = y * u_pitch;
            yuv_data.extend_from_slice(unsafe {
                std::slice::from_raw_parts(pixel_data.u.data.add(src_start), uv_disp_w.min(u_width))
            });
        }

        // Copy V plane (handle NV12 vs planar)
        if let Some(ref v_plane) = pixel_data.v {
            let v_pitch = v_plane.pitch;
            let v_width = v_plane.width;
            for y in 0..uv_disp_h {
                let src_start = y * v_pitch;
                yuv_data.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(v_plane.data.add(src_start), uv_disp_w.min(v_width))
                });
            }
        } else {
            // NV12: V is interleaved with U, copy every other byte starting at offset 1
            let u_pitch = pixel_data.u.pitch;
            for y in 0..uv_disp_h {
                let src_start = y * u_pitch;
                for x in 0..uv_disp_w {
                    let v_byte = unsafe { *pixel_data.u.data.add(src_start + x * 2 + 1) };
                    yuv_data.push(v_byte);
                }
            }
        }

        yuv_data
    } else {
        Vec::new()
    }
}

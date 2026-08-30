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
        let yuv_data = frame_to_yuv(frame);
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

/// Convert a decoded frame to planar YUV in its native chroma layout
/// (yuv420p for 4:2:0, yuv444p for 4:4:4). NV12 input is deinterleaved.
fn frame_to_yuv(frame: &vk_video_core::frame::DecodedFrame) -> Vec<u8> {
    if let Some(ref pd) = frame.pixel_data {
        let yw = pd.y.width as usize;
        let yh = pd.y.height as usize;
        let uw = pd.u.width as usize;
        let uh = pd.u.height as usize;

        let mut out = Vec::with_capacity(yw * yh + uw * uh * 2);

        for y in 0..yh {
            out.extend_from_slice(unsafe {
                std::slice::from_raw_parts(pd.y.data.add(y * pd.y.pitch), yw)
            });
        }

        if pd.v.is_none() {
            // NV12: U and V interleaved in the u plane; deinterleave into
            // planar U then planar V.
            for y in 0..uh {
                let row = unsafe { pd.u.data.add(y * pd.u.pitch) };
                for x in 0..uw {
                    out.push(unsafe { *row.add(x * 2) });
                }
            }
            for y in 0..uh {
                let row = unsafe { pd.u.data.add(y * pd.u.pitch) };
                for x in 0..uw {
                    out.push(unsafe { *row.add(x * 2 + 1) });
                }
            }
        } else {
            for y in 0..uh {
                out.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(pd.u.data.add(y * pd.u.pitch), uw)
                });
            }
            let v = pd.v.as_ref().unwrap();
            for y in 0..uh {
                out.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(v.data.add(y * v.pitch), uw)
                });
            }
        }

        out
    } else {
        Vec::new()
    }
}

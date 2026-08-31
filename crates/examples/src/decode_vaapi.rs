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

/// Append `w` samples from `src` (row start) to `out`, honoring `bps`.
/// iHD top-justifies 10-bit P016 samples (value << 6); when `top_justified`
/// is set, normalize to the bottom-justified yuv420p10le layout.
fn push_samples(out: &mut Vec<u8>, src: *const u8, w: usize, bps: usize, top_justified: bool) {
    if top_justified && bps == 2 {
        for i in 0..w {
            let v = u16::from_le_bytes(unsafe { [*src.add(i * 2), *src.add(i * 2 + 1)] }) >> 6;
            out.extend_from_slice(&v.to_le_bytes());
        }
    } else {
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(src, w * bps) });
    }
}

/// Convert a decoded frame to canonical planar Y+U+V raw bytes (rows packed,
/// `bps` bytes per sample). Matches FFmpeg's planar raw layouts:
/// yuv420p / yuv420p10le (4:2:0) and gbrp / yuv444p (4:4:4).
fn frame_to_yuv(frame: &vk_video_core::frame::DecodedFrame) -> Vec<u8> {
    if let Some(ref pixel_data) = frame.pixel_data {
        // Bytes per sample: 16-bit formats carry "16" in the name.
        let bps = if pixel_data.format.contains("16") { 2 } else { 1 };
        // iHD top-justifies 10-bit P016 samples (value << 6); normalize to the
        // bottom-justified yuv420p10le layout for FFmpeg comparison.
        let p016 = pixel_data.format == "P016";

        let y_w = pixel_data.y.width as usize;
        let y_h = pixel_data.y.height as usize;
        let c_w = pixel_data.u.width as usize;
        let c_h = pixel_data.u.height as usize;

        let mut out = Vec::with_capacity((y_w * y_h + c_w * c_h * 2) * bps);

        // Y plane: row by row, honoring pitch, `bps` bytes per sample.
        {
            let y_pitch = pixel_data.y.pitch;
            for row in 0..y_h {
                let src = unsafe { pixel_data.y.data.add(row * y_pitch) };
                push_samples(&mut out, src, y_w, bps, p016);
            }
        }

        if pixel_data.v.is_none() {
            // Semi-planar (NV12/P016): U and V interleaved in the u plane.
            // De-interleave into planar U then planar V.
            let u_pitch = pixel_data.u.pitch;
            for row in 0..c_h {
                let src = unsafe { pixel_data.u.data.add(row * u_pitch) };
                for col in 0..c_w {
                    push_samples(&mut out, unsafe { src.add(col * 2 * bps) }, 1, bps, p016);
                }
            }
            for row in 0..c_h {
                let src = unsafe { pixel_data.u.data.add(row * u_pitch) };
                for col in 0..c_w {
                    push_samples(&mut out, unsafe { src.add(col * 2 * bps + bps) }, 1, bps, p016);
                }
            }
        } else {
            // Planar: copy U plane then V plane.
            let u_pitch = pixel_data.u.pitch;
            for row in 0..c_h {
                let src = unsafe { pixel_data.u.data.add(row * u_pitch) };
                push_samples(&mut out, src, c_w, bps, p016);
            }
            let v = pixel_data.v.as_ref().unwrap();
            let v_pitch = v.pitch;
            for row in 0..c_h {
                let src = unsafe { v.data.add(row * v_pitch) };
                push_samples(&mut out, src, c_w, bps, p016);
            }
        }

        out
    } else {
        Vec::new()
    }
}

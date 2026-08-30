//! NVDEC HEVC (H.265) decode example with pixel output and params dump.
//!
//! Demonstrates hardware-accelerated HEVC decoding using NVIDIA NVDEC:
//!
//! 1. Load HEVC bitstream file
//! 2. Create NVDEC decoder (parses VPS/SPS/PPS, initializes hardware)
//! 3. Decode frames with pixel data copied from GPU (NV12, display order)
//! 4. Optionally dump the exact CUVIDPICPARAMS per picture (for diffing)
//!
//! Usage:
//!   cargo run --example decode_nvdec_h265 -- big_buck_bunney.h265 [max_frames] [dump.txt] [out_prefix]

use nvdec_decode::NvdecH265Decoder;
use vk_video_core::decoder::Decoder;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bitstream_path = if args.len() >= 2 {
        &args[1]
    } else {
        eprintln!(
            "Usage: {} <bitstream.h265> [max_frames] [dump.txt] [out_prefix]",
            args[0]
        );
        std::process::exit(1);
    };

    let max_frames: usize = if args.len() >= 3 {
        args[2].parse().unwrap_or(300)
    } else {
        300
    };
    let dump_path: Option<String> = args.get(3).cloned();
    let out_prefix: String = if args.len() >= 5 {
        args[4].clone()
    } else {
        "nvdec_h265".to_string()
    };

    if !std::path::Path::new(bitstream_path).exists() {
        eprintln!("Error: File not found: {}", bitstream_path);
        std::process::exit(1);
    }

    // Set the params-dump path BEFORE creating the decoder (read in `new`).
    if let Some(p) = &dump_path {
        std::env::set_var("NVDEC_DUMP_PARAMS", p);
    }

    println!("=== NVDEC HEVC Decode Example ===");
    println!("Backend: nvdec");
    println!("File: {}", bitstream_path);
    println!("Codec: HEVC/H.265\n");

    if !nvdec_decode::is_available() {
        eprintln!("Error: NVDEC not available on this system");
        std::process::exit(1);
    }

    let data = std::fs::read(bitstream_path).expect("Failed to read file");
    println!("Loaded {} bytes", data.len());

    println!("\n--- Creating NVDEC HEVC decoder ---");
    let mut decoder = match NvdecH265Decoder::new(data) {
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
    println!(
        "  Coded size: {}x{}",
        info.coded_size.width, info.coded_size.height
    );
    println!(
        "  Display size: {}x{}",
        info.display_size.width, info.display_size.height
    );
    println!("  Chroma: {}", info.chroma_subsampling);
    println!(
        "  Bit depth: {}bit/{}bit",
        info.luma_bit_depth.bit_depth(),
        info.chroma_bit_depth.bit_depth()
    );
    println!("  Profile: {:?}", info.profile_idc);
    println!("  DPB slots: {}", info.dpb_slots);

    println!("\n--- Decoding up to {} frames ---", max_frames);
    let mut frames: Vec<vk_video_core::frame::DecodedFrame> = Vec::new();

    while frames.len() < max_frames {
        match decoder.decode() {
            Ok(Some(frame)) => frames.push(frame),
            Ok(None) => break,
            Err(e) => {
                eprintln!("Decode error: {}", e);
                break;
            }
        }
    }

    // Drain any frames still held back in the decoder pipeline.
    if frames.len() < max_frames {
        match decoder.flush() {
            Ok(mut flushed) => frames.append(&mut flushed),
            Err(e) => eprintln!("Flush error: {}", e),
        }
    }
    frames.truncate(max_frames);

    let frames_decoded = frames.len();

    let bps = if info.luma_bit_depth.bit_depth() >= 10 { 2 } else { 1 };

    for (frame_idx, frame) in frames.iter().enumerate() {
        if let Some(ref pixel_data) = frame.pixel_data {
            let output_path = format!("{}_disp{}.yuv", out_prefix, frame_idx);
            let nv12 = frame_to_nv12(pixel_data, bps);
            match std::fs::write(&output_path, &nv12) {
                Ok(()) => {
                    if frames_decoded <= 8 || frame_idx % 50 == 0 {
                        println!(
                            "  Frame {}: POC={} {}x{} -> {} ({} bytes)",
                            frame_idx,
                            frame.poc,
                            frame.width,
                            frame.height,
                            output_path,
                            nv12.len()
                        );
                    }
                }
                Err(e) => eprintln!("    Failed to save: {}", e),
            }
        }
    }

    println!("\n--- Decode summary ---");
    println!("Decoded {} frames", frames_decoded);
    if let Some(p) = &dump_path {
        println!("Params dump: {}", p);
    }
    if frames_decoded > 0 {
        println!("Success: NVDEC HEVC decoding with pixel output is working!");
    } else {
        eprintln!("Warning: No frames were decoded. Check input file.");
    }
    println!("\n=== Done ===");
}

/// Convert planar [`PixelData`] to raw bytes (`bps` = bytes per sample):
/// - monochrome (u.width == 0): Y only
/// - 4:4:4 (u.width == y.width): planar Y + U + V, full resolution
/// - 4:2:0: Y plane + interleaved UV (NV12 layout)
fn frame_to_nv12(pixel_data: &vk_video_core::frame::PixelData, bps: usize) -> Vec<u8> {
    let y = &pixel_data.y;

    // Monochrome: Y plane only.
    if pixel_data.u.width == 0 {
        let mut out = Vec::with_capacity(y.width as usize * y.height as usize * bps);
        for row in 0..y.height as usize {
            let src = unsafe { y.data.add(row * y.pitch as usize) };
            out.extend_from_slice(unsafe {
                std::slice::from_raw_parts(src, y.width as usize * bps)
            });
        }
        return out;
    }

    let u = &pixel_data.u;
    let v = pixel_data.v.as_ref().expect("planar format must have a V plane");

    // 4:4:4: dump each plane at full resolution (matches FFmpeg gbrp raw layout).
    if u.width as usize == y.width as usize {
        let mut out = Vec::with_capacity(y.width as usize * y.height as usize * bps * 3);
        for plane in [&y, &u, v] {
            for row in 0..plane.height as usize {
                let src = unsafe { plane.data.add(row * plane.pitch as usize) };
                out.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(src, plane.width as usize * bps)
                });
            }
        }
        return out;
    }

    let uv_w = u.width as usize;
    let uv_h = u.height as usize;
    let uv_size = uv_w * uv_h;

    let y_bytes = y.width as usize * y.height as usize * bps;
    let mut out = Vec::with_capacity(y_bytes + uv_size * 2 * bps);

    // Y plane (row by row, honoring pitch).
    for row in 0..y.height as usize {
        let src = unsafe { y.data.add(row * y.pitch as usize) };
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(src, y.width as usize * bps) });
    }

    // Interleaved UV: U[0] V[0] U[1] V[1] ... (each sample `bps` bytes).
    for row in 0..uv_h {
        let u_row = unsafe { u.data.add(row * u.pitch as usize) };
        let v_row = unsafe { v.data.add(row * v.pitch as usize) };
        for col in 0..uv_w {
            out.extend_from_slice(unsafe { std::slice::from_raw_parts(u_row.add(col * bps), bps) });
            out.extend_from_slice(unsafe { std::slice::from_raw_parts(v_row.add(col * bps), bps) });
        }
    }

    out
}

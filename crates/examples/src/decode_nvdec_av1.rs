//! NVDEC AV1 decode example with pixel output and params dump.
//!
//! Demonstrates hardware-accelerated AV1 decoding using NVIDIA NVDEC, driven
//! by the Rust `vk-video-parser` (Av1Parser) via [`NvdecAv1Decoder`]:
//!
//! 1. Load an AV1 IVF file (or raw single frame)
//! 2. Create the NVDEC decoder (IVF + OBU walk, DPB management)
//! 3. Decode frames with pixel data copied from the GPU (NV12 -> YUV420P)
//! 4. Optionally dump the exact CUVIDPICPARAMS per picture (for diffing)
//!
//! Usage:
//!   cargo run --example decode_nvdec_av1 -- \
//!     [ivf_file] [max_frames] [dump.txt] [out_prefix]
//!   defaults: assets/big_buck_bunny_av1.ivf 300 None nvdec_av1

use nvdec_decode::NvdecAv1Decoder;
use vk_video_core::decoder::Decoder;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ivf_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "assets/big_buck_bunny_av1.ivf".to_string());
    let max_frames: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(300);
    let dump_path: Option<String> = args.get(3).cloned();
    let out_prefix: String = args
        .get(4)
        .cloned()
        .unwrap_or_else(|| "nvdec_av1".to_string());

    if !std::path::Path::new(&ivf_path).exists() {
        eprintln!("Error: File not found: {}", ivf_path);
        std::process::exit(1);
    }

    // Set the params-dump path BEFORE creating the decoder (read in `new`).
    if let Some(p) = &dump_path {
        std::env::set_var("NVDEC_DUMP_PARAMS", p);
    }

    println!("=== NVDEC AV1 Decode Example ===");
    println!("Backend: nvdec");
    println!("File: {}", ivf_path);
    println!("Codec: AV1\n");

    if !nvdec_decode::is_available() {
        eprintln!("Error: NVDEC not available on this system");
        std::process::exit(1);
    }

    let data = std::fs::read(&ivf_path).expect("Failed to read file");
    println!("Loaded {} bytes", data.len());

    println!("\n--- Creating NVDEC AV1 decoder ---");
    let mut decoder = match NvdecAv1Decoder::new(data) {
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

    for (frame_idx, frame) in frames.iter().enumerate() {
        match &frame.pixel_data {
            Some(pixel_data) => {
                let output_path = format!("{}_frame_{}.yuv", out_prefix, frame_idx);
                let yuv = frame_to_yuv420p(pixel_data);
                match std::fs::write(&output_path, &yuv) {
                    Ok(()) => {
                        if frames_decoded <= 8 || frame_idx % 50 == 0 {
                            println!(
                                "  Frame {}: POC={} {}x{} -> {} ({} bytes)",
                                frame_idx,
                                frame.poc,
                                frame.width,
                                frame.height,
                                output_path,
                                yuv.len()
                            );
                        }
                    }
                    Err(e) => eprintln!("    Failed to save: {}", e),
                }
            }
            None => {
                if frames_decoded <= 8 {
                    println!(
                        "  Frame {}: POC={} {}x{} (no pixel data)",
                        frame_idx, frame.poc, frame.width, frame.height
                    );
                }
            }
        }
    }

    println!("\n--- Decode summary ---");
    println!("Decoded {} frames", frames_decoded);
    if let Some(p) = &dump_path {
        println!("Params dump: {}", p);
    }
    if frames_decoded > 0 {
        println!("Success: NVDEC AV1 decoding with pixel output is working!");
    } else {
        eprintln!("Warning: No frames were decoded. Check input file.");
    }
    println!("\n=== Done ===");
}

/// Convert I420 [`PixelData`] to planar YUV420P bytes (Y plane, U plane, V
/// plane), honoring each plane's pitch.
fn frame_to_yuv420p(pixel_data: &vk_video_core::frame::PixelData) -> Vec<u8> {
    let mut yuv_data = Vec::with_capacity(pixel_data.buffer.len());

    // Copy Y plane
    for y in 0..pixel_data.y.height {
        let src_ptr = unsafe { pixel_data.y.data.add(y * pixel_data.y.pitch) };
        yuv_data
            .extend_from_slice(unsafe { std::slice::from_raw_parts(src_ptr, pixel_data.y.width) });
    }

    // Copy U plane
    for y in 0..pixel_data.u.height {
        let src_ptr = unsafe { pixel_data.u.data.add(y * pixel_data.u.pitch) };
        yuv_data
            .extend_from_slice(unsafe { std::slice::from_raw_parts(src_ptr, pixel_data.u.width) });
    }

    // Copy V plane
    if let Some(ref v) = pixel_data.v {
        for y in 0..v.height {
            let src_ptr = unsafe { v.data.add(y * v.pitch) };
            yuv_data
                .extend_from_slice(unsafe { std::slice::from_raw_parts(src_ptr, v.width) });
        }
    }

    yuv_data
}

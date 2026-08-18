//! NVDEC H.264 decode example with pixel-perfect output.
//!
//! Demonstrates hardware-accelerated H.264 decoding using NVIDIA NVDEC:
//!
//! 1. Load H.264 bitstream file
//! 2. Create NVDEC decoder (parses SPS/PPS, initializes hardware)
//! 3. Decode frames with pixel data copied from GPU
//! 4. Save decoded YUV420P frames for comparison with ffmpeg
//!
//! Usage:
//!   cargo run --example decode_nvdec -- born_trailer.h264 [max_frames]

use nvdec_decode::NvdecDecoder;
use vk_video_core::decoder::Decoder;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bitstream_path = if args.len() >= 2 {
        &args[1]
    } else {
        eprintln!("Usage: {} <bitstream.h264> [max_frames] [out_prefix]", args[0]);
        eprintln!("Available: born_trailer.h264");
        std::process::exit(1);
    };

    let max_frames: usize = if args.len() >= 3 {
        args[2].parse().unwrap_or(3)
    } else {
        3
    };

    let out_prefix: String = if args.len() >= 4 {
        args[3].clone()
    } else {
        "nvdec".to_string()
    };

    if !std::path::Path::new(bitstream_path).exists() {
        eprintln!("Error: File not found: {}", bitstream_path);
        std::process::exit(1);
    }

    println!("=== NVDEC H.264 Decode Example ===");
    println!("Backend: nvdec");
    println!("File: {}", bitstream_path);
    println!("Codec: H.264/AVC\n");

    // Check NVDEC availability
    if !nvdec_decode::is_available() {
        eprintln!("Error: NVDEC not available on this system");
        std::process::exit(1);
    }

    let data = std::fs::read(bitstream_path).expect("Failed to read file");
    println!("Loaded {} bytes", data.len());

    // Create decoder (handles device init, session creation, DPB setup)
    println!("\n--- Creating NVDEC decoder ---");
    let mut decoder = match NvdecDecoder::new(data) {
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
    println!("  Profile: {:?}", info.profile_idc);
    println!("  DPB slots: {}", info.dpb_slots);

    // Decode frames (in display order; B-frame reordering is handled inside
    // the decoder, with the tail drained via flush() at end of stream)
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

    // Drain any frames still held back in the decoder pipeline (B-frame
    // reorder depth) so ALL frames are emitted.
    if frames.len() < max_frames {
        match decoder.flush() {
            Ok(mut flushed) => frames.append(&mut flushed),
            Err(e) => eprintln!("Flush error: {}", e),
        }
    }
    frames.truncate(max_frames);

    let frames_decoded = frames.len();

    for (frame_idx, frame) in frames.iter().enumerate() {
        let has_pixel_data = frame.pixel_data.is_some();
        println!(
            "  Frame {}: index={}, POC={}, width={}x{}, ref={}, pixel_data={}",
            frame_idx,
            frame.frame_index,
            frame.poc,
            frame.width,
            frame.height,
            frame.is_reference(),
            has_pixel_data
        );

        // Save frame as YUV420P
        if let Some(ref pixel_data) = frame.pixel_data {
            let output_path = format!("{}_frame_{}.yuv", out_prefix, frame_idx);
            let yuv_data = frame_to_yuv420p(pixel_data);
            match std::fs::write(&output_path, &yuv_data) {
                Ok(()) => println!("    Saved to {} ({} bytes)", output_path, yuv_data.len()),
                Err(e) => eprintln!("    Failed to save: {}", e),
            }
        }
    }

    println!("\n--- Decode summary ---");
    println!("Decoded {} frames", frames_decoded);

    if frames_decoded > 0 {
        println!("Success: NVDEC H.264 decoding with pixel output is working!");
        println!("\nTo verify pixel-perfect output against ffmpeg:");
        println!("  ffmpeg -y -i {} -vframes {} -f rawvideo -pix_fmt yuv420p ffmpeg_ref.yuv", bitstream_path, frames_decoded);
        println!("  diff {}_frame_0.yuv <(dd if=ffmpeg_ref.yuv bs={} count=1 status=none)", out_prefix, frame_size(&info));
    } else {
        eprintln!("Warning: No frames were decoded. Check input file.");
    }

    println!("\n=== Done ===");
}

/// Convert PixelData to YUV420P planar format bytes.
fn frame_to_yuv420p(pixel_data: &vk_video_core::frame::PixelData) -> Vec<u8> {
    let mut yuv_data = Vec::with_capacity(pixel_data.buffer.len());

    // Copy Y plane
    for y in 0..pixel_data.y.height {
        let src_ptr = unsafe { pixel_data.y.data.add(y * pixel_data.y.pitch) };
        yuv_data.extend_from_slice(unsafe { std::slice::from_raw_parts(src_ptr, pixel_data.y.width) });
    }

    // Copy U plane
    for y in 0..pixel_data.u.height {
        let src_ptr = unsafe { pixel_data.u.data.add(y * pixel_data.u.pitch) };
        yuv_data.extend_from_slice(unsafe { std::slice::from_raw_parts(src_ptr, pixel_data.u.width) });
    }

    // Copy V plane
    if let Some(ref v) = pixel_data.v {
        for y in 0..v.height {
            let src_ptr = unsafe { v.data.add(y * v.pitch) };
            yuv_data.extend_from_slice(unsafe { std::slice::from_raw_parts(src_ptr, v.width) });
        }
    }

    yuv_data
}

/// Calculate YUV420P frame size.
fn frame_size(info: &vk_video_core::decoder::DecoderInfo) -> usize {
    let w = info.display_size.width as usize;
    let h = info.display_size.height as usize;
    w * h + (w / 2) * (h / 2) * 2
}

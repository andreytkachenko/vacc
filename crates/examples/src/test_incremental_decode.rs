//! Test incremental submit/decode pattern (streaming use case).
//!
//! Tests the decoder's ability to handle data submitted in chunks:
//! - NAL units split across chunks
//! - Multiple NAL units in one chunk
//! - Start codes at chunk boundaries
//!
//! Usage:
//!   cargo run --example test_incremental_decode -- assets/born_trailer.h264

use nvdec_decode::NvdecDecoder;
use vk_video_core::decoder::Decoder;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bitstream_path = if args.len() >= 2 {
        &args[1]
    } else {
        "assets/born_trailer.h264"
    };

    if !std::path::Path::new(bitstream_path).exists() {
        eprintln!("Error: File not found: {}", bitstream_path);
        std::process::exit(1);
    }

    println!("=== Incremental Submit/Decode Test ===");
    println!("File: {}", bitstream_path);

    // Check NVDEC availability
    if !nvdec_decode::is_available() {
        eprintln!("Error: NVDEC not available on this system");
        std::process::exit(1);
    }

    let data = std::fs::read(bitstream_path).expect("Failed to read file");
    println!("Loaded {} bytes\n", data.len());

    // ---- Baseline: non-incremental decode ----
    println!("--- Test 1: Non-incremental decode (baseline) ---");
    let baseline_result = decode_non_incremental(&data);
    println!(
        "Baseline: {} total frames ({} from flush)",
        baseline_result.total, baseline_result.flushed
    );

    // ---- Test 64KB chunks ----
    println!("\n--- Test 2: Incremental decode (64KB chunks) ---");
    let chunk_64k = decode_incremental(&data, 64 * 1024);
    println!(
        "64KB chunks: {} total frames ({} from flush)",
        chunk_64k.total, chunk_64k.flushed
    );

    // ---- Test 1KB chunks ----
    println!("\n--- Test 3: Incremental decode (1KB chunks) ---");
    let chunk_1k = decode_incremental(&data, 1024);
    println!(
        "1KB chunks: {} total frames ({} from flush)",
        chunk_1k.total, chunk_1k.flushed
    );

    // ---- Test 1MB chunks ----
    println!("\n--- Test 4: Incremental decode (1MB chunks) ---");
    let chunk_1m = decode_incremental(&data, 1024 * 1024);
    println!(
        "1MB chunks: {} total frames ({} from flush)",
        chunk_1m.total, chunk_1m.flushed
    );

    // ---- Results ----
    println!("\n=== Results ===");
    println!(
        "Baseline:     {} frames (flushed: {})",
        baseline_result.total, baseline_result.flushed
    );
    println!(
        "64KB chunks:  {} frames (flushed: {})",
        chunk_64k.total, chunk_64k.flushed
    );
    println!(
        "1KB chunks:   {} frames (flushed: {})",
        chunk_1k.total, chunk_1k.flushed
    );
    println!(
        "1MB chunks:   {} frames (flushed: {})",
        chunk_1m.total, chunk_1m.flushed
    );

    let all_match = baseline_result.total == chunk_64k.total
        && baseline_result.total == chunk_1k.total
        && baseline_result.total == chunk_1m.total;

    if all_match {
        println!("\nPASS: All chunk sizes produced identical frame counts!");
    } else {
        println!("\nFAIL: Frame count mismatch between chunk sizes!");
        std::process::exit(1);
    }
}

struct DecodeResult {
    total: usize,
    flushed: usize,
}

/// Decode the entire file at once (non-incremental baseline).
fn decode_non_incremental(data: &[u8]) -> DecodeResult {
    let mut decoder = NvdecDecoder::new(data.to_vec()).unwrap();
    let info = decoder.info();
    println!(
        "  Decoder: {}x{} @ {}bps",
        info.coded_size.width, info.coded_size.height, info.codec
    );

    let mut total_frames = 0;
    loop {
        match decoder.decode() {
            Ok(Some(_frame)) => total_frames += 1,
            Ok(None) => break,
            Err(e) => {
                eprintln!("  Decode error: {}", e);
                break;
            }
        }
    }

    let flushed = decoder.flush().unwrap().len();
    total_frames += flushed;

    DecodeResult {
        total: total_frames,
        flushed,
    }
}

/// Decode using the submit/decode pattern with the given chunk size.
fn decode_incremental(data: &[u8], chunk_size: usize) -> DecodeResult {
    // Find the minimum data needed for SPS/PPS + first frame
    let min_init = find_min_init_size(data);

    // Initialize decoder with enough data to contain SPS/PPS + first frame
    // Use max(chunk_size, min_init) to ensure we have enough data
    let init_size = std::cmp::max(chunk_size, min_init);
    let init_size = std::cmp::min(init_size, data.len());

    let (mut decoder, actual_init_size) = match NvdecDecoder::new(data[..init_size].to_vec()) {
        Ok(d) => (d, init_size),
        Err(e) => {
            eprintln!(
                "  Failed with init_size={} (min={}), err: {}",
                init_size, min_init, e
            );
            // Try with a larger buffer
            let larger = std::cmp::min(init_size + 64 * 1024, data.len());
            match NvdecDecoder::new(data[..larger].to_vec()) {
                Ok(d) => {
                    println!("  Retry succeeded with {} bytes", larger);
                    (d, larger)
                }
                Err(e2) => {
                    eprintln!("  Retry also failed with {} bytes: {}", larger, e2);
                    panic!("Cannot create decoder");
                }
            }
        }
    };

    let info = decoder.info();
    println!(
        "  Decoder: {}x{} @ {}bps (init with {} bytes, min_au={})",
        info.coded_size.width, info.coded_size.height, info.codec, actual_init_size, min_init
    );

    let mut total_frames = 0;
    let mut chunks_submitted = 0;

    // Submit remaining data in chunks (start from actual_init_size, not min_init!)
    for chunk in data[actual_init_size..].chunks(chunk_size) {
        decoder.submit(chunk).unwrap();
        chunks_submitted += 1;

        // Try to decode frames after each submit
        loop {
            match decoder.decode() {
                Ok(Some(_frame)) => total_frames += 1,
                Ok(None) => break,
                Err(e) => {
                    eprintln!("  Decode error on chunk {}: {}", chunks_submitted, e);
                    break;
                }
            }
        }
    }

    println!("  Submitted {} chunks", chunks_submitted);

    // Flush remaining frames
    let flushed = decoder.flush().unwrap().len();
    total_frames += flushed;

    DecodeResult {
        total: total_frames,
        flushed,
    }
}

/// Find the minimum number of bytes needed to contain SPS/PPS + first frame.
/// The NVDEC parser needs a complete access unit (SPS+PPS+first frame) to initialize.
fn find_min_init_size(data: &[u8]) -> usize {
    // H.264 NAL types: 7=SPS, 8=PPS, 1-5=slice
    // We need to find the first complete access unit (ends after first slice)
    let mut pos = 0;
    let mut found_sps = false;
    let mut found_pps = false;

    while pos + 4 < data.len() {
        // Find start code
        if data[pos] == 0 && data[pos + 1] == 0 {
            let sc_len = if pos + 3 < data.len() && data[pos + 2] == 0 && data[pos + 3] == 1 {
                4
            } else if data[pos + 2] == 1 {
                3
            } else {
                pos += 1;
                continue;
            };
            let nal_type = data[pos + sc_len] & 0x1f;

            if nal_type == 7 {
                found_sps = true;
            } else if nal_type == 8 {
                found_pps = true;
            }

            // Found a slice NAL (types 1-5) and we have SPS+PPS
            if (1..=5).contains(&nal_type) && found_sps && found_pps {
                // Find the end of this NAL unit
                let next = find_next_start_code(data, pos + sc_len + 1);
                return next;
            }

            // Find next start code
            let next = find_next_start_code(data, pos + sc_len + 1);
            pos = next;
        } else {
            pos += 1;
        }
    }
    // Fallback: return entire data
    data.len()
}

/// Find the offset of the next start code after the given position.
fn find_next_start_code(data: &[u8], start: usize) -> usize {
    let mut pos = start;
    while pos + 3 < data.len() {
        if data[pos] == 0 && data[pos + 1] == 0 {
            if data[pos + 2] == 1
                || (pos + 3 < data.len() && data[pos + 2] == 0 && data[pos + 3] == 1)
            {
                return pos;
            }
        }
        pos += 1;
    }
    data.len()
}

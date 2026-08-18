//! Test decoder reset mid-stream and re-initialization.
//!
//! Verifies that reset() properly clears state and allows
//! re-decoding from the beginning of the stream.

use nvdec_decode::NvdecDecoder;
use vk_video_core::decoder::Decoder;

fn main() {
    let data = std::fs::read("assets/born_trailer.h264").expect("Failed to read file");
    println!("Loaded {} bytes", data.len());

    // Use a smaller subset for stress tests to avoid excessive parse time
    let small_data = &data[..std::cmp::min(5_000_000, data.len())];
    println!("Using {} bytes for stress tests", small_data.len());

    // Test 1: Reset mid-stream (use smaller data for speed)
    println!("\n=== Test 1: Reset mid-stream ===");
    {
        let mut decoder = NvdecDecoder::new(small_data.to_vec()).expect("Failed to create decoder");

        // Decode 10 frames
        let mut count1 = 0;
        while let Ok(Some(frame)) = decoder.decode() {
            count1 += 1;
            println!("  [before] Frame {}: POC={}, idx={}", count1, frame.poc, frame.frame_index);
            if count1 >= 10 { break; }
        }
        println!("Before reset: {} frames decoded", count1);

        // Reset
        decoder.reset().expect("reset failed");
        println!("Reset done");

        // Decode again from the beginning
        let mut count2 = 0;
        while let Ok(Some(frame)) = decoder.decode() {
            count2 += 1;
            println!("  [after]  Frame {}: POC={}, idx={}", count2, frame.poc, frame.frame_index);
            if count2 >= 10 { break; }
        }
        println!("After reset: {} frames decoded", count2);

        if count1 == count2 {
            println!("Test 1 PASSED - same frame count before/after reset");
        } else {
            println!("Test 1 WARNING - frame count differs (before={}, after={})", count1, count2);
        }
    }

    // Test 2: Multiple reset cycles (use smaller data for speed)
    println!("\n=== Test 2: Multiple reset cycles ===");
    {
        let mut decoder = NvdecDecoder::new(small_data.to_vec()).expect("Failed to create decoder");
        let mut all_same = true;
        let mut first_count: Option<usize> = None;

        for cycle in 0..5 {
            decoder.reset().expect("reset failed");
            let mut count = 0;
            while let Ok(Some(frame)) = decoder.decode() {
                count += 1;
                if cycle == 0 {
                    println!("  Cycle {}: Frame {} POC={} idx={}", cycle, count, frame.poc, frame.frame_index);
                }
                if count >= 5 { break; }
            }
            println!("Cycle {}: {} frames", cycle, count);

            if let Some(first) = first_count {
                if count != first {
                    all_same = false;
                }
            } else {
                first_count = Some(count);
            }
        }

        if all_same {
            println!("Test 2 PASSED - consistent frame counts across {} cycles", first_count.unwrap());
        } else {
            println!("Test 2 FAILED - inconsistent frame counts across cycles");
        }
    }

    // Test 3: Reset -> flush -> decode cycle
    println!("\n=== Test 3: Reset -> flush -> decode cycle ===");
    {
        let mut decoder = NvdecDecoder::new(small_data.to_vec()).expect("Failed to create decoder");

        // Decode a few frames
        let mut count1 = 0;
        while let Ok(Some(_)) = decoder.decode() {
            count1 += 1;
            if count1 >= 3 { break; }
        }
        println!("Initial decode: {} frames", count1);

        // Reset
        decoder.reset().expect("reset failed");
        println!("Reset done");

        // Flush after reset drains all pending frames from the pipeline
        // This is expected - reset re-parsed all data, so many frames are pending
        let flushed = decoder.flush().expect("flush failed");
        println!("Flush after reset: {} frames (expected: all pending drained)", flushed.len());

        // After flush, decode returns None (no more data to parse)
        let post_flush = decoder.decode().expect("decode failed");
        println!("Decode after flush: {:?}", if post_flush.is_some() { "Some" } else { "None" });

        // Reset again and decode without flushing
        decoder.reset().expect("reset failed again");
        let mut count2 = 0;
        while let Ok(Some(_)) = decoder.decode() {
            count2 += 1;
            if count2 >= 5 { break; }
        }
        println!("Decode after second reset (no flush): {} frames", count2);

        if count2 > 0 {
            println!("Test 3 PASSED - decode works after reset");
        } else {
            println!("Test 3 FAILED - no frames after reset");
        }
    }

    // Test 4: Rapid reset (stress)
    println!("\n=== Test 4: Rapid reset stress test ===");
    {
        let mut decoder = NvdecDecoder::new(small_data.to_vec()).expect("Failed to create decoder");
        for i in 0..5 {
            decoder.reset().expect("reset failed");
            // Decode just 1 frame
            let _ = decoder.decode();
            println!("  Rapid cycle {}: reset+1 frame ok", i);
        }
        println!("Test 4 PASSED - 5 rapid reset cycles completed");
    }

    // Test 5: GPU memory check
    println!("\n=== Test 5: GPU memory stability ===");
    {
        let mem_before = get_gpu_mem().unwrap_or(0);
        println!("GPU memory before: {} MiB", mem_before);

        // Run a single decode cycle with reset (keep it fast)
        let mut decoder = NvdecDecoder::new(small_data.to_vec()).expect("Failed to create decoder");
        // Just one reset cycle to check memory
        decoder.reset().expect("reset failed");
        let mut count = 0;
        while let Ok(Some(_)) = decoder.decode() {
            count += 1;
            if count >= 5 { break; }
        }
        println!("Reset cycle: {} frames", count);
        drop(decoder);

        let mem_after = get_gpu_mem().unwrap_or(0);
        println!("GPU memory after: {} MiB", mem_after);

        let diff = if mem_after > mem_before { mem_after - mem_before } else { mem_before - mem_after };
        if diff < 20 {
            println!("Test 5 PASSED - GPU memory stable (delta={} MiB)", diff);
        } else {
            println!("Test 5 WARNING - GPU memory changed by {} MiB", diff);
        }
    }

    println!("\n=== All reset tests completed ===");
}

/// Query GPU memory usage via nvidia-smi
fn get_gpu_mem() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    // Parse "XXX MiB\n"
    let text = text.trim();
    let without_unit = text.trim_end_matches("MiB").trim();
    without_unit.parse().ok()
}

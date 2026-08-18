//! Concurrent NVDEC decoder thread safety test.
//!
//! Tests multiple decoder instances running concurrently in separate threads
//! and sequential create/destroy cycles.
//!
//! Usage:
//!   cargo run --example test_nvdec_concurrent -- [bitstream.h264]

use nvdec_decode::NvdecDecoder;
use std::thread;
use vk_video_core::decoder::Decoder;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bitstream_path = if args.len() >= 2 {
        &args[1]
    } else {
        "assets/born_trailer.h264"
    };

    // Smaller file for sequential/interleaved tests
    let small_file = "assets/test_baseline.h264";

    if !std::path::Path::new(bitstream_path).exists() {
        eprintln!("Error: File not found: {}", bitstream_path);
        std::process::exit(1);
    }

    // Check NVDEC availability
    if !nvdec_decode::is_available() {
        eprintln!("Error: NVDEC not available on this system");
        std::process::exit(1);
    }

    println!("=== NVDEC Concurrent Decoder Thread Safety Test ===");
    println!("File: {}", bitstream_path);
    println!();

    let mut all_ok = true;
    let mut seq_ok = true;
    let mut interleaved_ok = true;

    // -------------------------------------------------------
    // Test 1: Concurrent decoder instances (4 threads)
    // -------------------------------------------------------
    println!("--- Test 1: Concurrent decoders (4 threads, 10 frames each) ---");
    let data = std::fs::read(bitstream_path).expect("Failed to read file");
    println!("Loaded {} bytes", data.len());

    let num_threads = 4;
    let frames_per_thread = 10;
    let mut handles = vec![];

    for i in 0..num_threads {
        let data_clone = data.clone();
        handles.push(thread::spawn(move || {
            let start = std::time::Instant::now();
            let mut decoder = match NvdecDecoder::new(data_clone) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("[Thread {}] Failed to create decoder: {}", i, e);
                    return Err(e);
                }
            };

            let info = decoder.info();
            println!(
                "[Thread {}] Decoder created: {}x{} profile={:?}",
                i,
                info.display_size.width,
                info.display_size.height,
                info.profile_idc
            );

            let mut count = 0;
            while let Ok(Some(_frame)) = decoder.decode() {
                count += 1;
                if count >= frames_per_thread {
                    break;
                }
            }

            let elapsed = start.elapsed();
            println!(
                "[Thread {}] Decoded {} frames in {:?}",
                i, count, elapsed
            );
            Ok::<_, nvdec_decode::NvdecError>((count, elapsed))
        }));
    }

    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok((count, elapsed))) => {
                println!(
                    "  Thread {}: OK - {} frames in {:?}",
                    i, count, elapsed
                );
                if count == 0 {
                    eprintln!("  WARNING: Thread {} decoded 0 frames!", i);
                    all_ok = false;
                }
            }
            Ok(Err(e)) => {
                eprintln!("  Thread {}: ERROR - {}", i, e);
                all_ok = false;
            }
            Err(e) => {
                eprintln!("  Thread {}: PANIC - {:?}", i, e);
                all_ok = false;
            }
        }
    }

    if all_ok {
        println!("  PASSED: All threads decoded frames successfully");
    } else {
        println!("  FAILED: Some threads had errors");
    }
    println!();

    // -------------------------------------------------------
    // Test 2: Sequential create/destroy (10 iterations, 5 frames each)
    // Uses smaller file for speed
    // -------------------------------------------------------
    if !std::path::Path::new(small_file).exists() {
        println!("--- Test 2: SKIPPED ({} not found) ---\n", small_file);
        seq_ok = true; // Treat skipped as passed
    } else {
        println!("--- Test 2: Sequential create/destroy (10 iterations, 5 frames each) ---");
        for i in 0..10 {
            let data = match std::fs::read(small_file) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("  Iteration {}: Read error: {}", i, e);
                    seq_ok = false;
                    continue;
                }
            };
            let mut decoder = match NvdecDecoder::new(data) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("  Iteration {}: Failed to create decoder: {}", i, e);
                    seq_ok = false;
                    continue;
                }
            };

            let mut count = 0;
            while let Ok(Some(_frame)) = decoder.decode() {
                count += 1;
                if count >= 5 {
                    break;
                }
            }

            drop(decoder);
            println!("  Iteration {}: {} frames", i, count);

            if count == 0 {
                seq_ok = false;
            }
        }

        if seq_ok {
            println!("  PASSED: All sequential iterations succeeded");
        } else {
            println!("  FAILED: Some sequential iterations had errors");
        }
        println!();
    }

    // -------------------------------------------------------
    // Test 3: Interleaved create/destroy in threads
    // Uses smaller file for speed
    // -------------------------------------------------------
    if !std::path::Path::new(small_file).exists() {
        println!("--- Test 3: SKIPPED ({} not found) ---\n", small_file);
        interleaved_ok = true; // Treat skipped as passed
    } else {
        println!("--- Test 3: Interleaved create/destroy (4 threads x 3 cycles) ---");
        let mut handles = vec![];

        for i in 0..4 {
            let path = small_file.to_string();
            handles.push(thread::spawn(move || {
                let mut cycles_ok = 0;
                for cycle in 0..3 {
                    let data = match std::fs::read(&path) {
                        Ok(d) => d,
                        Err(e) => {
                            eprintln!("[T{} C{}] Read error: {}", i, cycle, e);
                            continue;
                        }
                    };

                    let mut decoder = match NvdecDecoder::new(data) {
                        Ok(d) => d,
                        Err(e) => {
                            eprintln!("[T{} C{}] Create error: {}", i, cycle, e);
                            continue;
                        }
                    };

                    let mut count = 0;
                    while let Ok(Some(_frame)) = decoder.decode() {
                        count += 1;
                        if count >= 3 {
                            break;
                        }
                    }

                    drop(decoder);

                    if count > 0 {
                        cycles_ok += 1;
                    }
                    println!("[T{} C{}] {} frames", i, cycle, count);
                }
                cycles_ok
            }));
        }

        for (i, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(cycles_ok) => {
                    println!("  Thread {}: {} successful cycles", i, cycles_ok);
                    if cycles_ok < 3 {
                        interleaved_ok = false;
                    }
                }
                Err(e) => {
                    eprintln!("  Thread {}: PANIC - {:?}", i, e);
                    interleaved_ok = false;
                }
            }
        }

        if interleaved_ok {
            println!("  PASSED: All interleaved cycles succeeded");
        } else {
            println!("  FAILED: Some interleaved cycles had errors");
        }
        println!();
    }

    // -------------------------------------------------------
    // Summary
    // -------------------------------------------------------
    println!("=== Summary ===");
    if all_ok && seq_ok && interleaved_ok {
        println!("ALL TESTS PASSED");
        println!("Thread safety: OK - concurrent decoders work correctly");
    } else {
        println!("SOME TESTS FAILED");
        if !all_ok {
            println!("  - Concurrent test: FAILED");
        }
        if !seq_ok {
            println!("  - Sequential test: FAILED");
        }
        if !interleaved_ok {
            println!("  - Interleaved test: FAILED");
        }
        std::process::exit(1);
    }
}

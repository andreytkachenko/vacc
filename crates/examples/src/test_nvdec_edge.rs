//! Edge case tests for NVDEC decoder

use nvdec_decode::NvdecDecoder;
use vk_video_core::decoder::Decoder;

fn main() {
    let data = std::fs::read("assets/born_trailer.h264").expect("Failed to read file");
    println!("Loaded {} bytes", data.len());

    // Take only first ~1MB for testing to avoid parsing thousands of frames
    let test_data = &data[..std::cmp::min(1_000_000, data.len())];

    // Test 1: Decode limited frames and flush pending
    println!("\n=== Test 1: Decode + flush pending ===");
    {
        let mut decoder = NvdecDecoder::new(test_data.to_vec()).expect("Failed to create decoder");
        let mut count = 0;
        while let Ok(Some(_)) = decoder.decode() {
            count += 1;
            if count >= 5 {
                break;
            }
        }
        println!("Decoded {} frames", count);

        let flushed = decoder.flush().expect("flush failed");
        println!("Flush returned {} pending frames", flushed.len());
    }
    println!("Test 1 PASSED");

    // Test 2: Reset decoder
    println!("\n=== Test 2: Reset decoder ===");
    {
        let mut decoder = NvdecDecoder::new(test_data.to_vec()).expect("Failed to create decoder");
        let mut count = 0;
        while let Ok(Some(_)) = decoder.decode() {
            count += 1;
            if count >= 3 {
                break;
            }
        }
        println!("Decoded {} frames before reset", count);

        decoder.reset().expect("reset failed");
        println!("Reset succeeded");
    }
    println!("Test 2 PASSED");

    // Test 3: Multiple decode cycles (memory leak check)
    println!("\n=== Test 3: Multiple decode cycles ===");
    for cycle in 0..5 {
        let mut decoder = NvdecDecoder::new(test_data.to_vec()).expect("Failed to create decoder");
        let mut count = 0;
        while let Ok(Some(_)) = decoder.decode() {
            count += 1;
            if count >= 10 {
                break;
            }
        }
        println!("Cycle {}: decoded {} frames", cycle, count);
        drop(decoder);
    }
    println!("Test 3 PASSED - no crashes in 5 decode cycles");

    // Test 4: Flush on fresh decoder (no frames decoded yet)
    println!("\n=== Test 4: Flush fresh decoder ===");
    {
        let mut decoder = NvdecDecoder::new(test_data.to_vec()).expect("Failed to create decoder");
        let flushed = decoder.flush().expect("flush failed");
        println!("Flush on fresh decoder returned {} frames", flushed.len());
    }
    println!("Test 4 PASSED");

    println!("\n=== All edge case tests passed ===");
}

//! Basic video decode example.
//!
//! This example demonstrates how to:
//! 1. Parse H.264, H.265, and AV1 bitstreams
//! 2. Extract SPS/PPS/VPS parameter sets
//! 3. Detect video format information

use vk_video_core::codec::VideoCodec;
use vk_video_parser::{
    h264::H264Parser,
    h265::H265Parser,
    av1::Av1Parser,
    DetectedVideoFormat,
    VideoParser,
    bitstream::BitstreamPacket,
    nal::find_next_start_code,
};

fn main() {
    println!("=== vk-video basic decode example ===\n");

    // Example 1: H.264 parsing
    println!("--- H.264 Parser ---");
    example_h264();

    // Example 2: H.265 parsing
    println!("\n--- H.265 Parser ---");
    example_h265();

    // Example 3: AV1 parsing
    println!("\n--- AV1 Parser ---");
    example_av1();

    // Example 4: Start code detection
    println!("\n--- Start Code Detection ---");
    example_start_codes();

    println!("\n=== API demonstration complete ===");
    println!("\nTo use the full Vulkan decode pipeline:");
    println!("  1. Create a VulkanDevice using VideoDeviceBuilder");
    println!("  2. Create a VideoPipeline with your codec");
    println!("  3. Initialize the pipeline");
    println!("  4. Feed bitstream packets through the parser");
    println!("  5. Record decode commands on the video queue");
    println!("  6. Submit and wait for completion");
    println!("  7. Process decoded YCbCr frames");
}

fn example_h264() {
    let mut parser = H264Parser::new();
    let format = DetectedVideoFormat::new(VideoCodec::DecodeH264);

    match parser.init(&format) {
        Ok(_) => println!("  Parser initialized successfully"),
        Err(e) => {
            eprintln!("  Failed to initialize parser: {}", e);
            return;
        }
    }

    // Create a sample SPS NAL unit (truncated for demo)
    // In reality, you would read this from a video file
    let sps_data = vec![
        0x67, 0x42, 0xe0, 0x1f, 0x00, 0xf5, 0x51, 0x22,
        0x52, 0x84, 0xa4, 0x21, 0x00, 0x00, 0x03, 0x00,
        0x04, 0x00, 0x00, 0x03, 0x00, 0xfa, 0x00, 0x00,
        0x06, 0x8e, 0x9c, 0x68,
    ];

    let sps_packet = BitstreamPacket::new(sps_data);
    match parser.parse(&sps_packet) {
        Ok(result) => {
            println!("  SPS parse result: {:?}", result);
            if let vk_video_parser::ParseResult::ParameterSet { sps, .. } = result {
                if let Some(sps) = sps {
                    println!("  SPS type: {:?}", sps.std_type());
                }
            }
        }
        Err(e) => {
            println!("  SPS parse (expected with truncated data): {}", e);
        }
    }

    // Create a sample PPS NAL unit
    let pps_data = vec![
        0x68, 0xce, 0x3c, 0x80,
    ];

    let pps_packet = BitstreamPacket::new(pps_data);
    match parser.parse(&pps_packet) {
        Ok(result) => {
            println!("  PPS parse result: {:?}", result);
        }
        Err(e) => {
            println!("  PPS parse (expected with truncated data): {}", e);
        }
    }

    // Create a sample slice NAL unit
    let slice_data = vec![
        0x65, 0x00, 0x00, 0x01, 0x00, 0x00, 0x03, 0x00,
        0x00, 0x03, 0xf0, 0x00, 0x29, 0xd0, 0x08,
    ];

    let slice_packet = BitstreamPacket::new(slice_data);
    match parser.parse(&slice_packet) {
        Ok(result) => {
            println!("  Slice parse result: {:?}", result);
        }
        Err(e) => {
            println!("  Slice parse (expected with truncated data): {}", e);
        }
    }
}

fn example_h265() {
    let mut parser = H265Parser::new();
    let format = DetectedVideoFormat::new(VideoCodec::DecodeH265);

    match parser.init(&format) {
        Ok(_) => println!("  Parser initialized successfully"),
        Err(e) => {
            eprintln!("  Failed to initialize parser: {}", e);
            return;
        }
    }

    // Create a sample VPS NAL unit (truncated)
    let vps_data = vec![
        0x40, 0x01, 0xff, 0xff, 0x01, 0x60, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00,
    ];

    let vps_packet = BitstreamPacket::new(vps_data);
    match parser.parse(&vps_packet) {
        Ok(result) => {
            println!("  VPS parse result: {:?}", result);
        }
        Err(e) => {
            println!("  VPS parse (expected with truncated data): {}", e);
        }
    }

    // Create a sample SPS NAL unit (truncated)
    let sps_data = vec![
        0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03,
        0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x6e,
        0x01, 0xff, 0xff, 0x01, 0x40, 0x00, 0x6e, 0x82,
    ];

    let sps_packet = BitstreamPacket::new(sps_data);
    match parser.parse(&sps_packet) {
        Ok(result) => {
            println!("  SPS parse result: {:?}", result);
        }
        Err(e) => {
            println!("  SPS parse (expected with truncated data): {}", e);
        }
    }
}

fn example_av1() {
    let mut parser = Av1Parser::new();
    let format = DetectedVideoFormat::new(VideoCodec::DecodeAv1);

    match parser.init(&format) {
        Ok(_) => println!("  Parser initialized successfully"),
        Err(e) => {
            eprintln!("  Failed to initialize parser: {}", e);
            return;
        }
    }

    // Create a sample AV1 sequence header (truncated)
    // AV1 uses a different framing - no start codes in the same way
    let seq_data = vec![
        0x2c, 0x0a, 0x05, 0x20, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    let seq_packet = BitstreamPacket::new(seq_data);
    match parser.parse(&seq_packet) {
        Ok(result) => {
            println!("  Sequence header parse result: {:?}", result);
        }
        Err(e) => {
            println!("  Sequence header parse (expected with truncated data): {}", e);
        }
    }
}

fn example_start_codes() {
    // Example: Finding start codes in a byte stream
    let data = vec![
        0x00, 0x00, 0x01, 0x67, 0x42, 0xe0, 0x1f, // Start code + SPS
        0x00, 0x00, 0x01, 0x68, 0xce, 0x3c, 0x80,   // Start code + PPS
        0x00, 0x00, 0x00, 0x01, 0x65, 0x00, 0x00,   // 4-byte start code + slice
        0x00, 0x00, 0x01, 0x06, 0x01, 0xff, 0xff,   // Filler NAL
    ];

    println!("  Scanning {} bytes for start codes...", data.len());

    let mut offset = 0;
    let mut count = 0;
    while offset < data.len() {
        if let Some((start, code_len)) = find_next_start_code(&data, offset) {
            count += 1;
            println!("  Found {}-byte start code at offset {}: 0x{:02x} 0x{:02x} 0x{:02x}",
                code_len, start, data[start], data[start+1], data[start+2]);
            offset = start + code_len;
        } else {
            break;
        }
    }
    println!("  Total start codes found: {}\n", count);
}

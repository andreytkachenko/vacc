//! Diagnostic tool: analyze bitstream construction for frame 1.
//!
//! This tool:
//! 1. Extracts NAL units from the original bitstream
//! 2. Identifies frame boundaries (IDR frames and picture boundaries)
//! 3. Shows the exact bytes for frame 1
//! 4. Parses slice headers to check parameters

use std::fs;
use vk_video_parser::nal::{self, H264NalUnitType, find_next_start_code};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <bitstream.h264>", args[0]);
        std::process::exit(1);
    }
    let path = &args[1];
    let data = fs::read(path).unwrap();
    println!("Bitstream: {} ({} bytes)", path, data.len());

    // Extract all NAL units
    let nal_units = extract_nal_units(&data);
    println!("\n=== All NAL Units (first 20) ===");
    for (i, nal) in nal_units.iter().take(20).enumerate() {
        let name = nal_unit_type_name(nal.nal_unit_type);
        println!("  NAL[{}]: offset=0x{:06x} type={} ({}) size={} bytes",
            i, nal.offset, nal.nal_unit_type, name, nal.size);
        if nal.size <= 32 {
            println!("         data: {}", nal.data.iter().map(|b| format!("{:02x}", b)).collect::<String>());
        }
    }

    // Identify frame boundaries
    println!("\n=== Frame Boundaries ===");
    let mut frame_boundaries = Vec::new();
    let mut current_frame_start = 0usize;
    let mut in_frame = false;

    for (i, nal) in nal_units.iter().enumerate() {
        match H264NalUnitType::from_u8(nal.nal_unit_type) {
            Some(H264NalUnitType::IdrSlice) => {
                if in_frame {
                    frame_boundaries.push((current_frame_start, i));
                }
                current_frame_start = i;
                in_frame = true;
                println!("  Frame boundary at NAL[{}]: IDR Slice (type=5)", i);
            }
            Some(H264NalUnitType::NonIdrSlice) => {
                if !in_frame {
                    current_frame_start = i;
                    in_frame = true;
                }
            }
            Some(H264NalUnitType::AccessUnitDelimiter) => {
                if in_frame {
                    frame_boundaries.push((current_frame_start, i));
                    in_frame = false;
                    println!("  Frame boundary at NAL[{}]: Access Unit Delimiter", i);
                }
            }
            _ => {}
        }
    }
    if in_frame {
        frame_boundaries.push((current_frame_start, nal_units.len()));
    }

    println!("\n=== Frame Summary ===");
    for (idx, (start, end)) in frame_boundaries.iter().take(5).enumerate() {
        let first_nal = &nal_units[*start];
        let nal_type_name = nal_unit_type_name(first_nal.nal_unit_type);
        println!("  Frame {}: NAL[{}..{}], first NAL type={}, offset=0x{:06x}",
            idx, start, end, nal_type_name, first_nal.offset);
    }

    // Analyze frame 1 specifically
    if frame_boundaries.len() >= 2 {
        let (frame1_start, frame1_end) = frame_boundaries[1];
        println!("\n=== Frame 1 Analysis ===");
        println!("  NAL range: [{}..{})", frame1_start, frame1_end);

        let mut total_size = 0usize;
        let mut slice_nals = Vec::new();
        let mut sei_nals = Vec::new();
        let mut other_nals = Vec::new();

        for i in frame1_start..frame1_end {
            let nal = &nal_units[i];
            total_size += nal.size;
            match H264NalUnitType::from_u8(nal.nal_unit_type) {
                Some(H264NalUnitType::NonIdrSlice) | Some(H264NalUnitType::IdrSlice) => {
                    slice_nals.push(i);
                }
                Some(H264NalUnitType::Sei) => {
                    sei_nals.push(i);
                }
                _ => {
                    other_nals.push(i);
                }
            }
        }

        println!("  Total frame size: {} bytes", total_size);
        println!("  Slice NALs: {} (indices: {:?})", slice_nals.len(), slice_nals);
        println!("  SEI NALs: {} (indices: {:?})", sei_nals.len(), sei_nals);
        println!("  Other NALs: {} (indices: {:?})", other_nals.len(), other_nals);

        // Print first 100 bytes of frame 1
        let first_nal_offset = nal_units[frame1_start].offset;
        println!("\n  First 100 bytes of frame 1 (from 0x{:06x}):", first_nal_offset);
        let frame_data = &data[first_nal_offset..first_nal_offset + total_size.min(100)];
        print!("    ");
        for (j, b) in frame_data.iter().enumerate() {
            if j % 16 == 0 && j > 0 {
                println!();
                print!("    ");
            }
            print!("{:02x} ", b);
        }
        println!();

        // Parse slice header from first slice NAL
        if let Some(slice_idx) = slice_nals.first() {
            let slice_nal = &nal_units[*slice_idx];
            println!("\n  === First Slice Header ===");
            parse_slice_header_info(&slice_nal.data);

            // Save frame 1 data to file
            let frame1_data = &data[first_nal_offset..first_nal_offset + total_size];
            fs::write("frame1_original.h264", frame1_data).unwrap();
            println!("\n  Saved frame 1 data to frame1_original.h264 ({} bytes)", total_size);
        }
    }

    // Compare with rust_frame2_bitstream.bin if it exists
    if let Ok(rust_data) = fs::read("rust_frame2_bitstream.bin") {
        println!("\n=== Comparison with rust_frame2_bitstream.bin ===");
        println!("  Size: {} bytes", rust_data.len());
        println!("  First 100 bytes:");
        print!("    ");
        for (j, b) in rust_data.iter().take(100).enumerate() {
            if j % 16 == 0 && j > 0 {
                println!();
                print!("    ");
            }
            print!("{:02x} ", b);
        }
        println!();

        // Parse first NAL unit
        if rust_data.len() >= 4 {
            let first_byte = rust_data[3];
            let nal_type = first_byte & 0x1F;
            let nal_ref_idc = (first_byte >> 5) & 0x03;
            println!("  First NAL: type={}, nal_ref_idc={}", nal_type, nal_ref_idc);
            parse_slice_header_info(&rust_data[3..]);
        }
    }
}

fn extract_nal_units(data: &[u8]) -> Vec<NalUnit> {
    let mut nal_units = Vec::new();
    let mut offset = 0;

    while offset < data.len() {
        if let Some((start, code_len)) = find_next_start_code(data, offset) {
            let next_start = find_next_start_code(data, start + code_len);
            let end = match next_start {
                Some((next_start, _)) => next_start,
                None => data.len(),
            };

            let nal_data = &data[start + code_len..end];
            if !nal_data.is_empty() {
                if let Some((_, _, nal_unit_type)) = nal::parse_h264_nal_header(nal_data) {
                    nal_units.push(NalUnit {
                        nal_unit_type,
                        data: nal_data.to_vec(),
                        offset: start + code_len,
                        size: nal_data.len(),
                    });
                }
            }
            offset = end;
        } else {
            break;
        }
    }
    nal_units
}

fn nal_unit_type_name(typ: u8) -> &'static str {
    match H264NalUnitType::from_u8(typ) {
        Some(H264NalUnitType::NonIdrSlice) => "Non-IDR Slice",
        Some(H264NalUnitType::IdrSlice) => "IDR Slice",
        Some(H264NalUnitType::Sei) => "SEI",
        Some(H264NalUnitType::Sps) => "SPS",
        Some(H264NalUnitType::Pps) => "PPS",
        Some(H264NalUnitType::AccessUnitDelimiter) => "AUD",
        Some(H264NalUnitType::SeqEnd) => "SeqEnd",
        Some(H264NalUnitType::StreamEnd) => "StreamEnd",
        Some(H264NalUnitType::FillerData) => "Filler",
        _ => "Unknown",
    }
}

fn parse_slice_header_info(data: &[u8]) {
    if data.is_empty() {
        println!("    (empty slice data)");
        return;
    }

    let first_byte = data[0];
    let forbidden = (first_byte & 0x80) != 0;
    let nal_ref_idc = (first_byte >> 5) & 0x03;
    let nal_unit_type = first_byte & 0x1F;

    println!("    NAL header: forbidden={}, nal_ref_idc={}, nal_unit_type={}",
        forbidden, nal_ref_idc, nal_unit_type);

    // Parse slice header (simplified)
    if nal_unit_type == 1 || nal_unit_type == 5 {
        // Skip NAL header byte
        let payload = &data[1..];
        if payload.len() < 2 {
            println!("    (slice header too short)");
            return;
        }

        // Simple parsing without EPB removal for first few fields
        let mut pos = 0;
        let mut bits_left = 0;
        let mut bit_buffer: u32 = 0;

        let mut read_bits = |n: u8| -> u32 {
            while bits_left < n {
                if pos >= payload.len() {
                    return 0xFFFFFFFF;
                }
                bit_buffer = (bit_buffer << 8) | (payload[pos] as u32);
                bits_left += 8;
                pos += 1;
            }
            bits_left -= n;
            let val = (bit_buffer >> bits_left) & ((1u32 << n) - 1);
            bit_buffer &= (1u32 << bits_left) - 1;
            val
        };

        let mut read_ue = || -> u32 {
            let mut leading_zeros = 0;
            while pos < payload.len() && (payload[pos] & 0x80) == 0 {
                leading_zeros += 1;
                pos += 1;
            }
            if pos >= payload.len() {
                return 0xFFFFFFFF;
            }
            let mut val = (payload[pos] & 0x7F) as u32;
            pos += 1;
            if leading_zeros > 0 && pos < payload.len() {
                val = (val << 8) | (payload[pos] as u32);
                pos += 1;
            }
            ((1u32 << leading_zeros) - 1) + val
        };

        let first_mb_in_slice = read_ue();
        let slice_type_raw = read_ue();
        let slice_type = slice_type_raw % 5;
        let pps_id = read_ue();

        println!("    first_mb_in_slice: {}", first_mb_in_slice);
        println!("    slice_type: {} ({})", slice_type, match slice_type {
            0 => "P", 1 => "B", 2 => "SP", 3 => "SI", _ => "I"
        });
        println!("    pps_id: {}", pps_id);
    }
}

#[derive(Debug)]
struct NalUnit {
    nal_unit_type: u8,
    data: Vec<u8>,
    offset: usize,
    size: usize,
}

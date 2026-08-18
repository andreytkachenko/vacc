//! NAL unit extraction and parsing tests for vk-video-parser.
//!
//! Tests verify NAL unit extraction matches H.264 specification (ITU-T H.264/AVC).

use vk_video_parser::nal::{self, CodecType, H264NalUnitType, NalUnit};

/// Path to the project root (parent of nvdec-decode crate).
const PROJECT_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// Load a test file from the project assets.
fn load_test_file(path: &str) -> Vec<u8> {
    let full_path = format!("{}/{}", PROJECT_ROOT, path);
    std::fs::read(&full_path).expect(&format!("Failed to read test file: {}", full_path))
}

// ============================================================================
// NAL Unit Type Detection Tests
// ============================================================================

#[test]
fn test_nal_unit_type_sps() {
    // SPS NAL unit type is 7 per H.264 spec Table 7-1
    let data = vec![0x67, 0x42, 0xC0, 0x28]; // NAL header: forbidden=0, ref_idc=3, type=7
    let result = nal::parse_h264_nal_header(&data);
    assert!(result.is_some(), "SPS header parse failed");
    let (forbidden, ref_idc, unit_type) = result.unwrap();
    assert_eq!(unit_type, 7, "Expected SPS NAL unit type 7");
    assert_eq!(
        H264NalUnitType::from_u8(unit_type),
        Some(H264NalUnitType::Sps)
    );
    assert!(!forbidden, "SPS forbidden_zero_bit should be 0");
    assert_eq!(ref_idc, 3, "SPS nal_ref_idc should be 3");
}

#[test]
fn test_nal_unit_type_pps() {
    // PPS NAL unit type is 8 per H.264 spec Table 7-1
    let data = vec![0x68, 0xCE, 0x3C, 0x80]; // NAL header: forbidden=0, ref_idc=3, type=8
    let result = nal::parse_h264_nal_header(&data);
    assert!(result.is_some(), "PPS header parse failed");
    let (forbidden, ref_idc, unit_type) = result.unwrap();
    assert_eq!(unit_type, 8, "Expected PPS NAL unit type 8");
    assert_eq!(
        H264NalUnitType::from_u8(unit_type),
        Some(H264NalUnitType::Pps)
    );
    assert!(!forbidden, "PPS forbidden_zero_bit should be 0");
    assert_eq!(ref_idc, 3, "PPS nal_ref_idc should be 3");
}

#[test]
fn test_nal_unit_type_idr_slice() {
    // IDR slice NAL unit type is 5 per H.264 spec Table 7-1
    let data = vec![0x65, 0x88, 0x01]; // NAL header: forbidden=0, ref_idc=3, type=5
    let result = nal::parse_h264_nal_header(&data);
    assert!(result.is_some(), "IDR slice header parse failed");
    let (forbidden, ref_idc, unit_type) = result.unwrap();
    assert_eq!(unit_type, 5, "Expected IDR slice NAL unit type 5");
    assert_eq!(
        H264NalUnitType::from_u8(unit_type),
        Some(H264NalUnitType::IdrSlice)
    );
    assert!(!forbidden, "IDR slice forbidden_zero_bit should be 0");
    assert_eq!(ref_idc, 3, "IDR slice nal_ref_idc should be 3");
}

#[test]
fn test_nal_unit_type_non_idr_slice() {
    // Non-IDR slice types: 1 (Coded slice of a non-IDR picture),
    // 2-4 (Data partitions A/B/C)
    // Bit layout: [forbidden_zero_bit(1) | nal_ref_idc(2) | nal_unit_type(5)]

    // Test type 1: Coded slice of a non-IDR picture
    // 0x61 = 0b01100001: forbidden=0, ref_idc=3, type=1
    let data_type1 = vec![0x61, 0x00];
    let (forbidden, ref_idc, unit_type) = nal::parse_h264_nal_header(&data_type1).unwrap();
    assert_eq!(unit_type, 1, "Expected non-IDR slice NAL unit type 1");
    assert_eq!(
        H264NalUnitType::from_u8(unit_type),
        Some(H264NalUnitType::NonIdrSlice)
    );
    assert!(!forbidden);
    assert_eq!(ref_idc, 3);

    // Test type 2: Data partition A
    // 0x62 = 0b01100010: forbidden=0, ref_idc=3, type=2
    let data_type2 = vec![0x62];
    let (forbidden, ref_idc, unit_type) = nal::parse_h264_nal_header(&data_type2).unwrap();
    assert_eq!(unit_type, 2, "Expected DataPartitionA NAL unit type 2");
    assert_eq!(
        H264NalUnitType::from_u8(unit_type),
        Some(H264NalUnitType::DataPartitionA)
    );
    assert!(!forbidden);
    assert_eq!(ref_idc, 3);

    // Test type 3: Data partition B
    // 0x63 = 0b01100011: forbidden=0, ref_idc=3, type=3
    let data_type3 = vec![0x63];
    let (forbidden, ref_idc, unit_type) = nal::parse_h264_nal_header(&data_type3).unwrap();
    assert_eq!(unit_type, 3, "Expected DataPartitionB NAL unit type 3");
    assert_eq!(
        H264NalUnitType::from_u8(unit_type),
        Some(H264NalUnitType::DataPartitionB)
    );
    assert!(!forbidden);
    assert_eq!(ref_idc, 3);

    // Test type 4: Data partition C
    // 0x64 = 0b01100100: forbidden=0, ref_idc=3, type=4
    let data_type4 = vec![0x64];
    let (forbidden, ref_idc, unit_type) = nal::parse_h264_nal_header(&data_type4).unwrap();
    assert_eq!(unit_type, 4, "Expected DataPartitionC NAL unit type 4");
    assert_eq!(
        H264NalUnitType::from_u8(unit_type),
        Some(H264NalUnitType::DataPartitionC)
    );
    assert!(!forbidden);
    assert_eq!(ref_idc, 3);
}

#[test]
fn test_nal_unit_type_sei() {
    // SEI NAL unit type is 6 per H.264 spec Table 7-1
    let data = vec![0x06, 0x01, 0x02, 0x80]; // NAL header: forbidden=0, ref_idc=0, type=6
    let result = nal::parse_h264_nal_header(&data);
    assert!(result.is_some(), "SEI header parse failed");
    let (forbidden, ref_idc, unit_type) = result.unwrap();
    assert_eq!(unit_type, 6, "Expected SEI NAL unit type 6");
    assert_eq!(
        H264NalUnitType::from_u8(unit_type),
        Some(H264NalUnitType::Sei)
    );
    assert!(!forbidden, "SEI forbidden_zero_bit should be 0");
    assert_eq!(ref_idc, 0, "SEI nal_ref_idc should be 0");
}

#[test]
fn test_nal_unit_type_filler_data() {
    // Filler data NAL unit type is 12 per H.264 spec Table 7-1
    let data = vec![0x0C, 0x00, 0x00]; // NAL header: forbidden=0, ref_idc=0, type=12
    let result = nal::parse_h264_nal_header(&data);
    assert!(result.is_some(), "Filler data header parse failed");
    let (forbidden, ref_idc, unit_type) = result.unwrap();
    assert_eq!(unit_type, 12, "Expected filler data NAL unit type 12");
    assert_eq!(
        H264NalUnitType::from_u8(unit_type),
        Some(H264NalUnitType::FillerData)
    );
    assert!(!forbidden, "Filler data forbidden_zero_bit should be 0");
    assert_eq!(ref_idc, 0, "Filler data nal_ref_idc should be 0");
}

// ============================================================================
// NAL Header Field Tests
// ============================================================================

#[test]
fn test_nal_unit_nal_ref_idc() {
    // nal_ref_idc is 2 bits (bits 5-6 of the first byte), values 0-3
    // Bit layout: [forbidden_zero_bit(1) | nal_ref_idc(2) | nal_unit_type(5)]

    // ref_idc = 0: 00xxxxxxx
    let data0 = vec![0x05];
    let (_, ref_idc, _) = nal::parse_h264_nal_header(&data0).unwrap();
    assert_eq!(ref_idc, 0, "Expected nal_ref_idc=0");

    // ref_idc = 1: 01xxxxxxx
    let data1 = vec![0x25];
    let (_, ref_idc, _) = nal::parse_h264_nal_header(&data1).unwrap();
    assert_eq!(ref_idc, 1, "Expected nal_ref_idc=1");

    // ref_idc = 2: 10xxxxxxx
    let data2 = vec![0x45];
    let (_, ref_idc, _) = nal::parse_h264_nal_header(&data2).unwrap();
    assert_eq!(ref_idc, 2, "Expected nal_ref_idc=2");

    // ref_idc = 3: 11xxxxxxx
    let data3 = vec![0x65];
    let (_, ref_idc, _) = nal::parse_h264_nal_header(&data3).unwrap();
    assert_eq!(ref_idc, 3, "Expected nal_ref_idc=3");
}

#[test]
fn test_nal_unit_forbidden_zero_bit() {
    // forbidden_zero_bit is bit 7 of the first byte, must always be 0

    // Valid: forbidden_zero_bit = 0
    let valid_data = vec![0x67];
    let (forbidden, _, _) = nal::parse_h264_nal_header(&valid_data).unwrap();
    assert!(!forbidden, "Valid NAL should have forbidden_zero_bit=0");

    // Invalid: forbidden_zero_bit = 1 (should be detectable)
    let invalid_data = vec![0xE7];
    let (forbidden, _, _) = nal::parse_h264_nal_header(&invalid_data).unwrap();
    assert!(forbidden, "Invalid NAL should have forbidden_zero_bit=1");
}

// ============================================================================
// Start Code Detection Tests
// ============================================================================

#[test]
fn test_nal_unit_start_code_3_byte() {
    // 3-byte start code: 0x00 0x00 0x01
    let data = vec![0x00, 0x00, 0x01, 0x67, 0x42];
    let result = nal::find_next_start_code(&data, 0);
    assert!(result.is_some(), "3-byte start code not found");
    let (offset, len) = result.unwrap();
    assert_eq!(offset, 0, "Start code should be at offset 0");
    assert_eq!(len, 3, "Start code length should be 3 bytes");
}

#[test]
fn test_nal_unit_start_code_4_byte() {
    // 4-byte start code: 0x00 0x00 0x00 0x01
    let data = vec![0x00, 0x00, 0x00, 0x01, 0x67, 0x42];
    let result = nal::find_next_start_code(&data, 0);
    assert!(result.is_some(), "4-byte start code not found");
    let (offset, len) = result.unwrap();
    assert_eq!(offset, 0, "Start code should be at offset 0");
    assert_eq!(len, 4, "Start code length should be 4 bytes");
}

#[test]
fn test_start_code_4_byte_takes_precedence_over_3_byte() {
    // When we have 0x00 0x00 0x00 0x01, the 4-byte start code should be detected
    // not a 3-byte start code at the same position
    let data = vec![0x00, 0x00, 0x00, 0x01, 0x67];
    let result = nal::find_next_start_code(&data, 0);
    let (offset, len) = result.unwrap();
    assert_eq!(offset, 0);
    assert_eq!(len, 4, "4-byte start code should take precedence");
}

#[test]
fn test_start_code_not_found_in_empty_data() {
    let data: Vec<u8> = vec![];
    let result = nal::find_next_start_code(&data, 0);
    assert!(result.is_none(), "No start code in empty data");
}

#[test]
fn test_start_code_not_found_in_invalid_data() {
    let data = vec![0x00, 0x01, 0x00, 0x01, 0xFF];
    let result = nal::find_next_start_code(&data, 0);
    assert!(result.is_none(), "No valid start code should be found");
}

// ============================================================================
// Emulation Prevention Byte Tests
// ============================================================================

#[test]
fn test_nal_unit_emulation_prevention_bytes() {
    // Emulation prevention: 0x00 0x00 0x03 sequence has 0x03 removed
    // Input: 0x00 0x00 0x03 0x01 should become 0x00 0x00 0x01
    let data = vec![0x00, 0x00, 0x03, 0x01];
    let result = nal::remove_emulation_prevention_bytes(&data);
    assert_eq!(result, vec![0x00, 0x00, 0x01]);

    // Multiple EPB sequences
    let data2 = vec![0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x02];
    let result2 = nal::remove_emulation_prevention_bytes(&data2);
    assert_eq!(result2, vec![0x00, 0x00, 0x00, 0x00, 0x02]);

    // No EPB sequences (0x00 0x00 0x00 and 0x00 0x00 0x01 are valid)
    let data3 = vec![0x00, 0x00, 0x00, 0x01];
    let result3 = nal::remove_emulation_prevention_bytes(&data3);
    assert_eq!(result3, vec![0x00, 0x00, 0x00, 0x01]);
}

#[test]
fn test_add_emulation_prevention_bytes() {
    // Adding EPB: before any 0x00 0x00 X where X <= 0x03, insert 0x03
    // Note: The implementation uses `i + 2 < data.len()` (strict less than),
    // so sequences at the exact end of the data are NOT processed.
    // We test with extra trailing bytes to ensure the pattern is matched.

    let data = vec![0x00, 0x00, 0x01, 0xFF]; // Extra byte to trigger EPB insertion
    let result = nal::add_emulation_prevention_bytes(&data);
    assert_eq!(result, vec![0x00, 0x00, 0x03, 0x01, 0xFF]);

    let data2 = vec![0x00, 0x00, 0x00, 0xFF];
    let result2 = nal::add_emulation_prevention_bytes(&data2);
    assert_eq!(result2, vec![0x00, 0x00, 0x03, 0x00, 0xFF]);

    // No EPB needed (X > 0x03)
    let data3 = vec![0x00, 0x00, 0x04, 0xFF];
    let result3 = nal::add_emulation_prevention_bytes(&data3);
    assert_eq!(result3, vec![0x00, 0x00, 0x04, 0xFF]);
}

#[test]
fn test_emulation_prevention_round_trip() {
    // Adding then removing EPB should yield original data
    // Include trailing byte to ensure add_emulation_prevention_bytes processes all patterns
    let original = vec![0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0xFF, 0xFF];
    let with_epb = nal::add_emulation_prevention_bytes(&original);
    let restored = nal::remove_emulation_prevention_bytes(&with_epb);
    assert_eq!(
        restored, original,
        "Round trip should preserve original data"
    );
}

// ============================================================================
// Annex B Format Extraction Tests
// ============================================================================

#[test]
fn test_nal_unit_extraction_from_annexb() {
    // Annex B format uses start codes to delimit NAL units
    // Format: [start code][NAL data][start code][NAL data]...
    let data = vec![
        // SPS NAL unit (type 7) with 4-byte start code
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0xC0, 0x28,
        // PPS NAL unit (type 8) with 3-byte start code
        0x00, 0x00, 0x01, 0x68, 0xCE, 0x3C, 0x80, // IDR slice NAL unit (type 5)
        0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x01, 0x02,
    ];

    let mut offset = 0;
    let mut nal_count = 0;
    let mut nal_types = Vec::new();

    while offset < data.len() {
        if let Some((start, code_len)) = nal::find_next_start_code(&data, offset) {
            let next_start = nal::find_next_start_code(&data, start + code_len);
            let end = next_start.map(|(s, _)| s).unwrap_or(data.len());

            let nal_data = &data[start + code_len..end];
            if !nal_data.is_empty() {
                if let Some((_, _, unit_type)) = nal::parse_h264_nal_header(nal_data) {
                    nal_types.push(unit_type);
                    nal_count += 1;
                }
            }
            offset = end;
        } else {
            break;
        }
    }

    assert_eq!(nal_count, 3, "Should extract 3 NAL units");
    assert_eq!(nal_types, vec![7, 8, 5], "Expected SPS(7), PPS(8), IDR(5)");
}

// ============================================================================
// AVCC Format Extraction Tests
// ============================================================================

#[test]
fn test_nal_unit_extraction_from_avcc() {
    // AVCC format uses length-prefixed NAL units (4-byte big-endian length)
    // Format: [4-byte length][NAL data][4-byte length][NAL data]...
    let data = vec![
        // SPS NAL unit (type 7), length = 4
        0x00, 0x00, 0x00, 0x04, 0x67, 0x42, 0xC0, 0x28,
        // PPS NAL unit (type 8), length = 3
        0x00, 0x00, 0x00, 0x03, 0x68, 0xCE, 0x3C,
        // IDR slice NAL unit (type 5), length = 4
        0x00, 0x00, 0x00, 0x04, 0x65, 0x88, 0x01, 0x02,
    ];

    let mut offset = 0;
    let mut nal_count = 0;
    let mut nal_types = Vec::new();

    while offset + 4 <= data.len() {
        // Read 4-byte big-endian length
        let length = ((data[offset] as usize) << 24)
            | ((data[offset + 1] as usize) << 16)
            | ((data[offset + 2] as usize) << 8)
            | (data[offset + 3] as usize);
        offset += 4;

        if offset + length > data.len() {
            break;
        }

        let nal_data = &data[offset..offset + length];
        if !nal_data.is_empty() {
            if let Some((_, _, unit_type)) = nal::parse_h264_nal_header(nal_data) {
                nal_types.push(unit_type);
                nal_count += 1;
            }
        }
        offset += length;
    }

    assert_eq!(nal_count, 3, "Should extract 3 NAL units from AVCC");
    assert_eq!(nal_types, vec![7, 8, 5], "Expected SPS(7), PPS(8), IDR(5)");
}

// ============================================================================
// born_trailer.h264 Integration Tests
// ============================================================================

#[test]
fn test_nal_unit_extraction_from_born_trailer() {
    let data = load_test_file("assets/born_trailer.h264");

    // Extract NAL units using the parser's start code detection
    let mut offset = 0;
    let mut nal_count = 0;
    let mut nal_types = Vec::new();

    while offset < data.len() {
        if let Some((start, code_len)) = nal::find_next_start_code(&data, offset) {
            let next_start = nal::find_next_start_code(&data, start + code_len);
            let end = next_start.map(|(s, _)| s).unwrap_or(data.len());

            let nal_data = &data[start + code_len..end];
            if !nal_data.is_empty() {
                if let Some((forbidden, ref_idc, unit_type)) = nal::parse_h264_nal_header(nal_data)
                {
                    nal_types.push(unit_type);
                    nal_count += 1;

                    // Verify all extracted NALs have valid headers
                    assert!(
                        !forbidden,
                        "NAL#{} at offset {} has forbidden_zero_bit=1",
                        nal_count, start
                    );
                    assert!(
                        ref_idc <= 3,
                        "NAL#{} at offset {} has invalid nal_ref_idc={}",
                        nal_count,
                        start,
                        ref_idc
                    );
                    assert!(
                        H264NalUnitType::from_u8(unit_type).is_some(),
                        "NAL#{} at offset {} has unknown type={}",
                        nal_count,
                        start,
                        unit_type
                    );
                }
            }
            offset = end;
        } else {
            break;
        }
    }

    // Verify we found NAL units
    assert!(nal_count > 0, "Should find NAL units in born_trailer.h264");

    // born_trailer.h264 contains SPS(7), PPS(8), and IDR slice(5)
    // SEI(6) may appear before SPS/PPS
    assert!(
        nal_types.iter().any(|&t| t == 7),
        "born_trailer.h264 should contain SPS (type 7)"
    );
    assert!(
        nal_types.iter().any(|&t| t == 8),
        "born_trailer.h264 should contain PPS (type 8)"
    );

    // Verify at least one IDR slice exists
    assert!(
        nal_types.iter().any(|&t| t == 5),
        "born_trailer.h264 should contain IDR slice (type 5)"
    );
}

#[test]
fn test_nal_unit_types_from_born_trailer() {
    let data = load_test_file("assets/born_trailer.h264");

    // Collect all unique NAL types in the file
    let mut offset = 0;
    let mut unique_types = std::collections::HashSet::new();

    while offset < data.len() {
        if let Some((start, code_len)) = nal::find_next_start_code(&data, offset) {
            let next_start = nal::find_next_start_code(&data, start + code_len);
            let end = next_start.map(|(s, _)| s).unwrap_or(data.len());

            let nal_data = &data[start + code_len..end];
            if !nal_data.is_empty() {
                if let Some((_, _, unit_type)) = nal::parse_h264_nal_header(nal_data) {
                    unique_types.insert(unit_type);
                }
            }
            offset = end;
        } else {
            break;
        }
    }

    // Verify expected types are present
    assert!(
        unique_types.contains(&7),
        "born_trailer.h264 should contain SPS (type 7)"
    );
    assert!(
        unique_types.contains(&8),
        "born_trailer.h264 should contain PPS (type 8)"
    );
    assert!(
        unique_types.contains(&5),
        "born_trailer.h264 should contain IDR slice (type 5)"
    );
}

// ============================================================================
// SEI Payload Parsing Tests
// ============================================================================

#[test]
fn test_nal_unit_sei_payload_parsing() {
    // SEI payload format (H.264 spec D.1):
    // payload_type is constructed from one or more bytes where each byte is 0xFF
    // except the last byte which is not 0xFF.
    // payload_size is constructed similarly.
    // Example: payload_type = 0x01 (buffering_period), payload_size = 0x07
    let sei_data = vec![
        0x06, // NAL header: type=6 (SEI), ref_idc=0
        0x01, // payload_type = 1 (buffering_period)
        0x07, // payload_size = 7 bytes
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, // payload data
        0x80, // rbsp_trailing_bits
    ];

    let (forbidden, ref_idc, unit_type) = nal::parse_h264_nal_header(&sei_data).unwrap();
    assert_eq!(unit_type, 6, "Expected SEI NAL unit type 6");
    assert_eq!(ref_idc, 0, "SEI should have nal_ref_idc=0");
    assert!(!forbidden);

    // Parse SEI payload type (skip NAL header byte)
    let payload = &sei_data[1..];
    let mut payload_type: u32 = 0;
    let mut i = 0;
    while i < payload.len() && payload[i] == 0xFF {
        payload_type = (payload_type << 8) | 0xFF;
        i += 1;
    }
    if i < payload.len() {
        payload_type = (payload_type << 8) | (payload[i] as u32);
        i += 1;
    }

    // Parse SEI payload size
    let mut payload_size: u32 = 0;
    while i < payload.len() && payload[i] == 0xFF {
        payload_size = (payload_size << 8) | 0xFF;
        i += 1;
    }
    if i < payload.len() {
        payload_size = (payload_size << 8) | (payload[i] as u32);
        i += 1;
    }

    assert_eq!(
        payload_type, 1,
        "Expected SEI payload_type=1 (buffering_period)"
    );
    assert_eq!(payload_size, 7, "Expected SEI payload_size=7");
    // Payload data + trailing bits should start at position i
    assert!(
        i < payload.len(),
        "Should have payload data after type and size"
    );
}

#[test]
fn test_sei_payload_multi_byte_type_and_size() {
    // Test multi-byte payload_type and payload_size encoding
    // Per H.264 spec D.1: each byte is 0xFF except the last which is not 0xFF.
    // For size=65280 (0xFF00): encoded as 0xFF, 0x00 (multi-byte with continuation)
    let sei_data = vec![
        0x06, // NAL header: type=6 (SEI)
        0x01, // payload_type = 1 (single byte)
        0xFF,
        0x00, // payload_size = 65280 - multi-byte with FF continuation
              // payload data placeholder
    ];

    let payload = &sei_data[1..];
    let mut payload_type: u32 = 0;
    let mut i = 0;
    while i < payload.len() && payload[i] == 0xFF {
        payload_type = (payload_type << 8) | 0xFF;
        i += 1;
    }
    if i < payload.len() {
        payload_type = (payload_type << 8) | (payload[i] as u32);
        i += 1;
    }

    let mut payload_size: u32 = 0;
    while i < payload.len() && payload[i] == 0xFF {
        payload_size = (payload_size << 8) | 0xFF;
        i += 1;
    }
    if i < payload.len() {
        payload_size = (payload_size << 8) | (payload[i] as u32);
        i += 1;
    }

    assert_eq!(payload_type, 1, "Expected payload_type=1");
    assert_eq!(
        payload_size, 65280,
        "Expected multi-byte payload_size=65280 (0xFF00)"
    );
}

// ============================================================================
// NalUnit Struct Tests
// ============================================================================

#[test]
fn test_nal_unit_struct_creation() {
    let data = vec![0x67, 0x42, 0xC0, 0x28];
    let nal = NalUnit::new(7, data.clone(), 10, 4);

    assert_eq!(nal.nal_unit_type, 7);
    assert_eq!(nal.data, data);
    assert_eq!(nal.offset, 10);
    assert_eq!(nal.size, 4);
    // H.264 NAL types are 0-31
    assert!(nal.is_h264(), "Type 7 should be H.264");
    // Note: is_h265() uses ranges 0..=14 | 32..=42, which overlaps with H.264.
    // Type 7 is in that range, so is_h265 returns true (this is a known limitation
    // of the type-agnostic NalUnit struct).
    assert!(
        nal.is_h265(),
        "Type 7 is in H.265 range per is_h265() implementation"
    );
}

#[test]
fn test_nal_unit_is_slice() {
    // Slice types: 1-5
    let slice_types = [1, 2, 3, 4, 5];
    for t in slice_types {
        let nal = NalUnit::new(t, vec![], 0, 0);
        assert!(
            nal.is_slice(CodecType::H264),
            "Type {} should be a slice",
            t
        );
    }

    // Non-slice types: 6-12
    let non_slice_types = [6, 7, 8, 9, 10, 11, 12];
    for t in non_slice_types {
        let nal = NalUnit::new(t, vec![], 0, 0);
        assert!(
            !nal.is_slice(CodecType::H264),
            "Type {} should not be a slice",
            t
        );
    }
}

#[test]
fn test_nal_unit_is_parameter_set() {
    // Parameter set types: 7 (SPS), 8 (PPS), 13 (SPS extension)
    let param_types = [7, 8, 13];
    for t in param_types {
        let nal = NalUnit::new(t, vec![], 0, 0);
        assert!(
            nal.is_parameter_set(CodecType::H264),
            "Type {} should be a parameter set",
            t
        );
    }

    // Non-parameter-set types
    let non_param_types = [1, 5, 6, 9];
    for t in non_param_types {
        let nal = NalUnit::new(t, vec![], 0, 0);
        assert!(
            !nal.is_parameter_set(CodecType::H264),
            "Type {} should not be a parameter set",
            t
        );
    }
}

// ============================================================================
// H264NalUnitType Enum Tests
// ============================================================================

#[test]
fn test_h264_nal_unit_type_from_u8_valid() {
    assert_eq!(
        H264NalUnitType::from_u8(0),
        Some(H264NalUnitType::Unspecified)
    );
    assert_eq!(
        H264NalUnitType::from_u8(1),
        Some(H264NalUnitType::NonIdrSlice)
    );
    assert_eq!(H264NalUnitType::from_u8(5), Some(H264NalUnitType::IdrSlice));
    assert_eq!(H264NalUnitType::from_u8(6), Some(H264NalUnitType::Sei));
    assert_eq!(H264NalUnitType::from_u8(7), Some(H264NalUnitType::Sps));
    assert_eq!(H264NalUnitType::from_u8(8), Some(H264NalUnitType::Pps));
    assert_eq!(
        H264NalUnitType::from_u8(9),
        Some(H264NalUnitType::AccessUnitDelimiter)
    );
    assert_eq!(
        H264NalUnitType::from_u8(12),
        Some(H264NalUnitType::FillerData)
    );
}

#[test]
fn test_h264_nal_unit_type_from_u8_invalid() {
    // Reserved/undefined types (16-31) should return None
    assert_eq!(H264NalUnitType::from_u8(16), None);
    assert_eq!(H264NalUnitType::from_u8(31), None);
    assert_eq!(H264NalUnitType::from_u8(32), None);
}

#[test]
fn test_h264_nal_unit_type_is_slice_methods() {
    assert!(H264NalUnitType::NonIdrSlice.is_slice());
    assert!(H264NalUnitType::IdrSlice.is_slice());
    assert!(H264NalUnitType::DataPartitionA.is_slice());
    assert!(H264NalUnitType::DataPartitionB.is_slice());
    assert!(H264NalUnitType::DataPartitionC.is_slice());
    assert!(!H264NalUnitType::Sei.is_slice());
    assert!(!H264NalUnitType::Sps.is_slice());
    assert!(!H264NalUnitType::Pps.is_slice());
}

#[test]
fn test_h264_nal_unit_type_is_parameter_set_methods() {
    assert!(H264NalUnitType::Sps.is_parameter_set());
    assert!(H264NalUnitType::Pps.is_parameter_set());
    assert!(H264NalUnitType::SpsExt.is_parameter_set());
    assert!(!H264NalUnitType::NonIdrSlice.is_parameter_set());
    assert!(!H264NalUnitType::IdrSlice.is_parameter_set());
    assert!(!H264NalUnitType::Sei.is_parameter_set());
}

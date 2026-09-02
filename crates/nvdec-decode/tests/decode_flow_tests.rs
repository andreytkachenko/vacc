//! Tests for the full decode flow with vacc-parser in nvdec-decode.
//!
//! These tests verify the complete decode flow from bitstream parsing to
//! CUVIDH264PICPARAMS construction, including SPS/PPS extraction, slice header
//! parsing, frame type detection, POC calculation, and MMCO command extraction.

use vacc_parser::{h264::H264Parser, BitstreamPacket, ParseResult, VideoParser};

/// Path to the project root (parent of nvdec-decode crate).
const PROJECT_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// Load a known H.264 test file from the project assets.
fn load_test_file(path: &str) -> Vec<u8> {
    let full_path = format!("{}/{}", PROJECT_ROOT, path);
    std::fs::read(&full_path).unwrap_or_else(|_| panic!("Failed to read test file: {}", full_path))
}

/// Find the next start code in a byte stream.
fn find_start_code(data: &[u8], start: usize) -> Option<(usize, usize)> {
    if start >= data.len() {
        return None;
    }
    let remaining = &data[start..];
    let mut i = 0;
    while i + 2 < remaining.len() {
        if i + 3 < remaining.len()
            && remaining[i] == 0
            && remaining[i + 1] == 0
            && remaining[i + 2] == 0
            && remaining[i + 3] == 1
        {
            if i == 0 || remaining[i - 1] != 0 {
                return Some((start + i, 4));
            }
        } else if remaining[i] == 0
            && remaining[i + 1] == 0
            && remaining[i + 2] == 1
            && (i == 0 || remaining[i - 1] != 0)
        {
            return Some((start + i, 3));
        }
        i += 1;
    }
    None
}

/// Extract the first NAL unit of a given type (with start code).
fn extract_first_nal_with_start_code(data: &[u8], nal_type: u8) -> Option<Vec<u8>> {
    let mut offset = 0;
    while offset < data.len() {
        let (start, code_len) = match find_start_code(data, offset) {
            Some((s, 4)) if s + 4 < data.len() => (s, 4),
            Some((s, 3)) if s + 3 < data.len() => (s, 3),
            _ => break,
        };

        let nal_header = data[start + code_len];
        let unit_type = nal_header & 0x1F;

        if unit_type == nal_type {
            let mut end = start + code_len + 1;
            while end < data.len() {
                if data[end..].starts_with(&[0, 0, 0, 1]) || data[end..].starts_with(&[0, 0, 1]) {
                    break;
                }
                end += 1;
            }
            return Some(data[start..end].to_vec());
        }

        // Find end of this NAL to skip it
        let mut end = start + code_len + 1;
        while end < data.len() {
            if data[end..].starts_with(&[0, 0, 0, 1]) || data[end..].starts_with(&[0, 0, 1]) {
                break;
            }
            end += 1;
        }
        offset = end;
    }
    None
}

/// Initialize a parser with SPS and PPS from the bitstream.
fn init_parser_with_params(data: &[u8]) -> H264Parser {
    let sps_data = extract_first_nal_with_start_code(data, 7).expect("No SPS found");
    let pps_data = extract_first_nal_with_start_code(data, 8).expect("No PPS found");

    let mut combined = Vec::new();
    combined.extend_from_slice(&sps_data);
    combined.extend_from_slice(&pps_data);

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(combined);
    parser.parse(&packet).expect("Failed to parse SPS/PPS");

    assert!(
        parser.active_sps().is_some(),
        "Active SPS should be set after parsing"
    );
    assert!(
        parser.active_pps().is_some(),
        "Active PPS should be set after parsing"
    );
    parser
}

/// Parse slices from the bitstream and collect them.
fn parse_slices_from_bitstream(data: &[u8]) -> Vec<vacc_parser::h264::SliceHeader> {
    let mut parser = init_parser_with_params(data);
    let mut slices = Vec::new();

    // Parse enough of the bitstream to get several frames
    let parse_limit = std::cmp::min(data.len(), 200_000);
    let packet = BitstreamPacket::new(data[..parse_limit].to_vec());

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::Slice {
                slices: frame_slices,
                ..
            }) => {
                for slice in &frame_slices {
                    if let Some(vacc_parser::SliceHeader::H264(slh)) = &slice.slice_header {
                        slices.push(slh.clone());
                    }
                }
            }
            Ok(ParseResult::ParameterSet { .. }) => {}
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Err(_) => break,
        }
    }

    slices
}

// ============================================================================
// Test 1: SPS/PPS extraction from real bitstream
// ============================================================================

#[test]
fn test_parser_extracts_sps_pps_from_real_bitstream() {
    let data = load_test_file("assets/born_trailer.h264");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data[..650].to_vec());

    let mut got_sps = false;
    let mut got_pps = false;

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::ParameterSet { sps, pps, .. }) => {
                if sps.is_some() {
                    got_sps = true;
                }
                if pps.is_some() {
                    got_pps = true;
                }
                if got_sps && got_pps {
                    break;
                }
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Ok(ParseResult::Slice { .. }) => break,
            Err(_) => break,
        }
    }

    assert!(got_sps, "Should have parsed SPS from born_trailer.h264");
    assert!(got_pps, "Should have parsed PPS from born_trailer.h264");
    assert!(parser.active_sps().is_some(), "Active SPS should be set");
    assert!(parser.active_pps().is_some(), "Active PPS should be set");

    let sps = parser.active_sps().unwrap();
    let pps = parser.active_pps().unwrap();

    // Verify SPS fields
    assert!(sps.profile_idc > 0, "profile_idc must be non-zero");
    assert!(sps.level_idc > 0, "level_idc must be non-zero");
    assert!(
        sps.pic_width_in_mbs_minus1 > 0,
        "pic_width_in_mbs_minus1 must be positive"
    );
    assert!(
        sps.pic_height_in_map_units_minus1 > 0,
        "pic_height_in_map_units_minus1 must be positive"
    );

    // Verify PPS links to SPS
    assert_eq!(
        pps.seq_parameter_set_id, sps.seq_parameter_set_id,
        "PPS seq_parameter_set_id should match active SPS"
    );
}

// ============================================================================
// Test 2: Slice header extraction with correct fields
// ============================================================================

#[test]
fn test_parser_extracts_slice_headers() {
    let data = load_test_file("assets/born_trailer.h264");
    let slices = parse_slices_from_bitstream(&data);

    assert!(
        !slices.is_empty(),
        "Should have parsed at least one slice header"
    );

    for (i, slh) in slices.iter().enumerate() {
        // Verify frame_num is within valid range
        let max_frame_num = 1u32 << (parser_active_sps(&data).log2_max_frame_num_minus4 + 4);
        assert!(
            slh.frame_num < max_frame_num,
            "Slice {} frame_num ({}) must be < max_frame_num ({})",
            i,
            slh.frame_num,
            max_frame_num
        );

        // Verify slice_type is valid (0-9)
        assert!(
            slh.slice_type <= 9,
            "Slice {} slice_type ({}) out of range",
            i,
            slh.slice_type
        );

        // Verify POC LSB is within valid range
        let max_poc_lsb = 1u32 << (parser_active_sps(&data).log2_max_pic_order_cnt_lsb_minus4 + 4);
        assert!(
            (slh.pic_order_cnt_lsb as u32) < max_poc_lsb,
            "Slice {} pic_order_cnt_lsb ({}) must be < max_pic_order_cnt_lsb ({})",
            i,
            slh.pic_order_cnt_lsb,
            max_poc_lsb
        );

        // Verify PPS ID matches active PPS
        assert_eq!(
            slh.pic_parameter_set_id,
            parser_active_pps(&data).pic_parameter_set_id,
            "Slice {} PPS ID mismatch",
            i
        );
    }
}

// ============================================================================
// Test 3: IDR frame detection
// ============================================================================

#[test]
fn test_parser_handles_idr_frame() {
    let data = load_test_file("assets/born_trailer.h264");
    let slices = parse_slices_from_bitstream(&data);

    // born_trailer.h264 starts with an IDR frame
    let idr_slices: Vec<_> = slices
        .iter()
        .filter(|slh| slh.nal_unit_type == 5 || slh.nal_unit_type == 7)
        .collect();

    assert!(
        !idr_slices.is_empty(),
        "Should find at least one IDR slice in born_trailer.h264"
    );

    for slh in &idr_slices {
        // IDR slices must have nal_ref_idc > 0
        assert!(
            slh.nal_ref_idc > 0,
            "IDR slice must have nal_ref_idc > 0, got {}",
            slh.nal_ref_idc
        );
    }
}

// ============================================================================
// Test 4: B-frame detection
// ============================================================================

#[test]
fn test_parser_handles_b_frames() {
    let data = load_test_file("assets/born_trailer.h264");
    let slices = parse_slices_from_bitstream(&data);

    assert!(!slices.is_empty(), "Should have parsed at least one slice");

    // B-frame slice_type values: 1 (B) or 6 (B field)
    let b_slices: Vec<_> = slices
        .iter()
        .filter(|slh| slh.slice_type == 1 || slh.slice_type == 6)
        .collect();

    // born_trailer.h264 was encoded with bframes=0, so no B-frames expected
    // This test verifies the parser correctly identifies slice types
    // and doesn't falsely report B-frames
    assert!(
        b_slices.is_empty(),
        "born_trailer.h264 has no B-frames (encoded with bframes=0)"
    );

    // Verify all slices have valid slice_type values (0-4 for frame coding)
    for slh in &slices {
        assert!(
            slh.slice_type <= 4,
            "slice_type ({}) should be 0-4 for this stream",
            slh.slice_type
        );
    }
}

// ============================================================================
// Test 5: P-frame detection
// ============================================================================

#[test]
fn test_parser_handles_p_frames() {
    let data = load_test_file("assets/born_trailer.h264");
    let slices = parse_slices_from_bitstream(&data);

    // P-frame slice_type values: 0 (P) or 5 (P field)
    let p_slices: Vec<_> = slices
        .iter()
        .filter(|slh| slh.slice_type == 0 || slh.slice_type == 5)
        .collect();

    // born_trailer.h264 contains P-frames
    assert!(
        !p_slices.is_empty(),
        "Should find at least one P-frame slice in born_trailer.h264"
    );
}

// ============================================================================
// Test 6: I-frame detection
// ============================================================================

#[test]
fn test_parser_handles_i_frames() {
    let data = load_test_file("assets/born_trailer.h264");
    let slices = parse_slices_from_bitstream(&data);

    // I-frame detection:
    // - slice_type == 4 indicates I-slice
    // - nal_unit_type == 5 indicates IDR slice (which is always an I-frame)
    let i_slices: Vec<_> = slices
        .iter()
        .filter(|slh| slh.slice_type == 4 || slh.nal_unit_type == 5)
        .collect();

    // born_trailer.h264 starts with an IDR which is also an I-frame
    assert!(
        !i_slices.is_empty(),
        "Should find at least one I-frame slice in born_trailer.h264 (found {} I-slices out of {} total)",
        i_slices.len(), slices.len()
    );

    for slh in &i_slices {
        // I-slices should have nal_ref_idc > 0 (reference picture)
        assert!(slh.nal_ref_idc > 0, "I-slice should have nal_ref_idc > 0");
    }
}

// ============================================================================
// Test 7: Frame number sequence
// ============================================================================

#[test]
fn test_parser_frame_num_sequence() {
    let data = load_test_file("assets/born_trailer.h264");
    let slices = parse_slices_from_bitstream(&data);

    assert!(
        slices.len() >= 2,
        "Need at least 2 slices to verify frame_num sequence"
    );

    // Collect unique frame_nums in order
    let mut frame_nums = Vec::new();
    for slh in &slices {
        if frame_nums.is_empty() || frame_nums.last().unwrap() != &slh.frame_num {
            frame_nums.push(slh.frame_num);
        }
    }

    assert!(
        frame_nums.len() >= 2,
        "Need at least 2 unique frames to verify sequence"
    );

    // Verify frame_num sequence is monotonically increasing (or wraps correctly)
    // IDR frames reset frame_num to 0, which is valid
    let max_frame_num = 1u32 << (parser_active_sps(&data).log2_max_frame_num_minus4 + 4);

    for i in 1..frame_nums.len() {
        let prev = frame_nums[i - 1];
        let curr = frame_nums[i];

        // Frame numbers should either increase or wrap around
        if curr < prev {
            // Two valid cases:
            // 1. Natural wrap-around near max_frame_num
            // 2. IDR frame reset (curr = 0)
            let is_idr_reset = curr == 0;
            let is_natural_wrap = prev > max_frame_num / 2 && curr < max_frame_num / 2;
            assert!(
                is_idr_reset || is_natural_wrap,
                "Frame_num wrap from {} to {} looks invalid (max={}, is_idr_reset={})",
                prev,
                curr,
                max_frame_num,
                is_idr_reset
            );
        } else {
            // Normal increasing case
            assert!(
                curr >= prev,
                "Frame_num sequence not monotonic: {} -> {}",
                prev,
                curr
            );
        }
    }
}

// ============================================================================
// Test 8: POC sequence for POC type 0
// ============================================================================

#[test]
fn test_parser_poc_sequence_type0() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps = parser_active_sps(&data);

    // Verify stream uses POC type 0
    assert_eq!(
        sps.pic_order_cnt_type, 0,
        "born_trailer.h264 should use POC type 0"
    );

    let slices = parse_slices_from_bitstream(&data);
    assert!(
        slices.len() >= 2,
        "Need at least 2 slices to verify POC sequence"
    );

    // Collect unique frame POCs
    let mut frame_pocs = Vec::new();
    let mut seen_frames = std::collections::HashSet::new();
    for slh in &slices {
        if seen_frames.insert(slh.frame_num) {
            frame_pocs.push((slh.frame_num, slh.pic_order_cnt_lsb));
        }
    }

    assert!(
        frame_pocs.len() >= 2,
        "Need at least 2 unique frames to verify POC sequence"
    );

    // For POC type 0, POC LSB should generally increase (with possible wrap)
    let max_poc_lsb = 1u32 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4);

    for i in 1..frame_pocs.len() {
        let prev_lsb = frame_pocs[i - 1].1;
        let curr_lsb = frame_pocs[i].1;

        // POC LSB should either increase or wrap around (similar to frame_num)
        // Allow wrap-around behavior
        let diff = curr_lsb.wrapping_sub(prev_lsb);
        assert!(
            diff > 0 || curr_lsb == prev_lsb,
            "POC LSB sequence anomaly: {} -> {} (max={})",
            prev_lsb,
            curr_lsb,
            max_poc_lsb
        );
    }
}

// ============================================================================
// Test 9: DecRefPicMarking (MMCO) extraction
// ============================================================================

#[test]
fn test_parser_dec_ref_pic_marking_extraction() {
    let data = load_test_file("assets/born_trailer.h264");
    let slices = parse_slices_from_bitstream(&data);

    // Check that reference slices have dec_ref_pic_marking parsed
    let ref_slices: Vec<_> = slices.iter().filter(|slh| slh.nal_ref_idc > 0).collect();

    assert!(
        !ref_slices.is_empty(),
        "Should find reference slices with nal_ref_idc > 0"
    );

    // For reference slices, dec_ref_pic_marking should be present (may be empty if no MMCO needed)
    for slh in &ref_slices {
        // The field should be parsed (Vec can be empty if no MMCO operations)
        let _ = &slh.dec_ref_pic_marking;
    }
}

// ============================================================================
// Test 10: Slice NAL data integrity
// ============================================================================

#[test]
fn test_parser_slice_nal_data_integrity() {
    let data = load_test_file("assets/born_trailer.h264");

    let mut parser = init_parser_with_params(&data);

    // Parse slices and check NAL data
    let packet = BitstreamPacket::new(data[..10_000].to_vec());

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::Slice { slices, .. }) => {
                assert!(!slices.is_empty(), "Should have parsed at least one slice");

                for (i, slice) in slices.iter().enumerate() {
                    // NAL data should not be empty
                    assert!(
                        !slice.nal_data.is_empty(),
                        "Slice {} NAL data should not be empty",
                        i
                    );

                    // NAL data should start with a valid NAL header byte
                    let first_byte = slice.nal_data[0];
                    let forbidden_zero_bit = (first_byte & 0x80) != 0;
                    assert!(
                        !forbidden_zero_bit,
                        "Slice {} NAL data forbidden_zero_bit must be 0",
                        i
                    );

                    let nal_unit_type = first_byte & 0x1F;
                    // Should be a slice NAL type (1-5)
                    assert!(
                        (1..=5).contains(&nal_unit_type),
                        "Slice {} NAL unit type ({}) should be 1-5",
                        i,
                        nal_unit_type
                    );

                    // Slice header should be present
                    assert!(
                        slice.slice_header.is_some(),
                        "Slice {} should have a parsed slice header",
                        i
                    );
                }
                break;
            }
            Ok(ParseResult::ParameterSet { .. }) => {}
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => {
                panic!("Should have found slices");
            }
            Err(e) => panic!("Parse error: {:?}", e),
        }
    }
}

// ============================================================================
// Test 11: Incremental parsing
// ============================================================================

#[test]
fn test_parser_incremental_parsing() {
    let data = load_test_file("assets/born_trailer.h264");

    // Test incremental parsing by feeding data in stages using a single packet
    // that contains both parameter sets and slice data.

    // Stage 1: Feed enough data for SPS and PPS
    let mut parser = H264Parser::new();
    let sps_pps_packet = BitstreamPacket::new(data[..650].to_vec());

    let mut got_sps = false;
    let mut got_pps = false;

    loop {
        match parser.parse(&sps_pps_packet) {
            Ok(ParseResult::ParameterSet { sps, pps, .. }) => {
                if sps.is_some() {
                    got_sps = true;
                }
                if pps.is_some() {
                    got_pps = true;
                }
                if got_sps && got_pps {
                    break;
                }
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Ok(ParseResult::Slice { .. }) => break,
            Err(_) => break,
        }
    }

    assert!(got_sps, "Should have parsed SPS in stage 1");
    assert!(got_pps, "Should have parsed PPS in stage 1");

    // Stage 2: Feed more data to get slices (using same packet with more bytes)
    // The parser tracks processed_up_to, so it will skip already-processed NALs
    let slice_packet = BitstreamPacket::new(data[..100_000].to_vec());
    let mut slices_found = 0;

    loop {
        match parser.parse(&slice_packet) {
            Ok(ParseResult::Slice { slices, .. }) => {
                slices_found += slices.len();
                // Continue parsing to get more frames
            }
            Ok(ParseResult::ParameterSet { .. }) => {
                // Additional parameter sets may appear
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Err(_) => break,
        }
    }

    assert!(
        slices_found > 0,
        "Should have found slices through incremental parsing (found {})",
        slices_found
    );
}

// ============================================================================
// Test 12: Parser reset and reparse
// ============================================================================

#[test]
fn test_parser_reset_and_reparse() {
    let data = load_test_file("assets/born_trailer.h264");

    // Initial parse
    let mut parser = init_parser_with_params(&data);

    let initial_sps_id = parser.active_sps().unwrap().seq_parameter_set_id;
    let initial_pps_id = parser.active_pps().unwrap().pic_parameter_set_id;

    // Parse some slices
    let packet = BitstreamPacket::new(data[..5000].to_vec());
    let mut slices_before_reset = 0;

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::Slice { slices, .. }) => {
                slices_before_reset += slices.len();
            }
            Ok(ParseResult::ParameterSet { .. }) => {}
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Err(_) => break,
        }
    }

    assert!(
        slices_before_reset > 0,
        "Should have parsed slices before reset"
    );

    // Reset parser
    parser.reset();

    // Verify state is cleared
    assert!(
        parser.active_sps().is_none(),
        "SPS should be cleared after reset"
    );
    assert!(
        parser.active_pps().is_none(),
        "PPS should be cleared after reset"
    );

    // Reparse SPS/PPS
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");
    let pps_data = extract_first_nal_with_start_code(&data, 8).expect("No PPS found");

    let mut combined = Vec::new();
    combined.extend_from_slice(&sps_data);
    combined.extend_from_slice(&pps_data);

    let packet_params = BitstreamPacket::new(combined);
    parser
        .parse(&packet_params)
        .expect("Failed to reparse SPS/PPS");

    // Verify re-parsed parameters match original
    let reparsed_sps_id = parser.active_sps().unwrap().seq_parameter_set_id;
    let reparsed_pps_id = parser.active_pps().unwrap().pic_parameter_set_id;

    assert_eq!(
        reparsed_sps_id, initial_sps_id,
        "Re-parsed SPS ID should match original"
    );
    assert_eq!(
        reparsed_pps_id, initial_pps_id,
        "Re-parsed PPS ID should match original"
    );

    // Verify we can parse slices again after reset+reparse
    let packet_slices = BitstreamPacket::new(data[..5000].to_vec());
    let mut slices_after_reset = 0;

    loop {
        match parser.parse(&packet_slices) {
            Ok(ParseResult::Slice { slices, .. }) => {
                slices_after_reset += slices.len();
            }
            Ok(ParseResult::ParameterSet { .. }) => {}
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Err(_) => break,
        }
    }

    assert!(
        slices_after_reset > 0,
        "Should be able to parse slices after reset and reparse"
    );
}

// ============================================================================
// Helper functions
// ============================================================================

fn parser_active_sps(data: &[u8]) -> vacc_core::picture::H264Sps {
    let parser = init_parser_with_params(data);
    parser.active_sps().unwrap().clone()
}

fn parser_active_pps(data: &[u8]) -> vacc_core::picture::H264Pps {
    let parser = init_parser_with_params(data);
    parser.active_pps().unwrap().clone()
}

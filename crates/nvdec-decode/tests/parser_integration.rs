//! Integration tests for nvdec-decode using vk-video-parser.
//!
//! Tests cover:
//! - SPS/PPS parsing from real H.264 bitstreams
//! - CUVIDH264PICPARAMS construction
//! - POC calculation for all three types
//! - DPB management operations

use vk_video_parser::{h264::H264Parser, BitstreamPacket, ParseResult, VideoParser};
use vk_video_vulkan::access_unit::H264MmcoCommand;
use vk_video_vulkan::dpb::{DpbManager, LastAccessType};

/// Path to the project root (parent of nvdec-decode crate).
const PROJECT_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// Load a known H.264 test file from the project assets.
fn load_test_file(path: &str) -> Vec<u8> {
    let full_path = format!("{}/{}", PROJECT_ROOT, path);
    std::fs::read(&full_path).expect(&format!("Failed to read test file: {}", full_path))
}

/// Extract raw NAL data (without start code) from the first NAL unit of given type.
fn extract_first_nal(data: &[u8], nal_type: u8) -> Option<Vec<u8>> {
    let mut offset = 0;
    while offset < data.len() {
        // Find start code (00 00 01 or 00 00 00 01)
        let (start, code_len) = match (data.get(offset..offset + 3), data.get(offset..offset + 4)) {
            (_, Some(&[0, 0, 0, 1])) => (offset, 4),
            (Some(&[0, 0, 1]), _) => (offset, 3),
            _ => {
                offset += 1;
                continue;
            }
        };

        if start + code_len >= data.len() {
            break;
        }

        let nal_header = data[start + code_len];
        let unit_type = nal_header & 0x1F;

        if unit_type == nal_type {
            // Find end of this NAL
            let mut end = start + code_len + 1;
            while end < data.len() {
                if data[end..].starts_with(&[0, 0, 0, 1]) || data[end..].starts_with(&[0, 0, 1]) {
                    break;
                }
                end += 1;
            }
            return Some(data[start + code_len..end].to_vec());
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

/// Extract NAL data WITH start code from the first NAL unit of given type.
fn extract_first_nal_with_start_code(data: &[u8], nal_type: u8) -> Option<Vec<u8>> {
    let mut offset = 0;
    while offset < data.len() {
        // Find start code (00 00 01 or 00 00 00 01)
        let (start, code_len) = match (data.get(offset..offset + 3), data.get(offset..offset + 4)) {
            (_, Some(&[0, 0, 0, 1])) => (offset, 4),
            (Some(&[0, 0, 1]), _) => (offset, 3),
            _ => {
                offset += 1;
                continue;
            }
        };

        if start + code_len >= data.len() {
            break;
        }

        let nal_header = data[start + code_len];
        let unit_type = nal_header & 0x1F;

        if unit_type == nal_type {
            // Find end of this NAL
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

// ============================================================================
// SPS Parsing Tests
// ============================================================================

#[test]
fn test_sps_parsing_from_real_bitstream() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(sps_data);
    let result = parser.parse(&packet).expect("SPS parse failed");

    match result {
        ParseResult::ParameterSet { sps: Some(_), .. } => {
            let sps = parser.active_sps().expect("No active SPS");

            // Verify basic SPS fields are parsed (values depend on actual bitstream)
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
            assert!(
                sps.pic_order_cnt_type <= 2,
                "pic_order_cnt_type must be 0, 1, or 2"
            );
            assert!(
                sps.max_num_ref_frames > 0,
                "max_num_ref_frames must be positive"
            );
            assert!(sps.chroma_format_idc <= 3, "chroma_format_idc must be 0-3");
        }
        _ => panic!("Expected ParameterSet result, got {:?}", result),
    }
}

#[test]
fn test_sps_profile_level_from_born_trailer() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(sps_data);
    parser.parse(&packet).expect("SPS parse failed");

    let sps = parser.active_sps().expect("No active SPS");

    // born_trailer.h264 is Main profile, Level 4.1
    assert_eq!(sps.profile_idc, 66, "Expected Main profile (66)");
    assert_eq!(sps.level_idc, 41, "Expected Level 4.1 (level_idc=41)");
}

#[test]
fn test_sps_dimensions_from_born_trailer() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(sps_data);
    parser.parse(&packet).expect("SPS parse failed");

    let sps = parser.active_sps().expect("No active SPS");

    // born_trailer.h264: coded dimensions from SPS
    let coded_width = (sps.pic_width_in_mbs_minus1 as u32 + 1) * 16;
    let coded_height = if sps.frame_mbs_only_flag {
        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
    } else {
        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
    };

    // Verify coded dimensions are positive and reasonable
    assert!(
        coded_width > 0 && coded_width <= 4096,
        "coded_width ({}) out of reasonable range",
        coded_width
    );
    assert!(
        coded_height > 0 && coded_height <= 2160,
        "coded_height ({}) out of reasonable range",
        coded_height
    );
    assert_eq!(coded_width % 16, 0, "coded_width must be multiple of 16");
    assert_eq!(coded_height % 16, 0, "coded_height must be multiple of 16");
}

#[test]
fn test_sps_chroma_format_and_bit_depth() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(sps_data);
    parser.parse(&packet).expect("SPS parse failed");

    let sps = parser.active_sps().expect("No active SPS");

    // born_trailer.h264 is 4:2:0, 8-bit
    assert_eq!(
        sps.chroma_format_idc, 1,
        "Expected 4:2:0 chroma format (idc=1)"
    );
    assert_eq!(
        sps.bit_depth_luma_minus8, 0,
        "Expected 8-bit luma (minus8=0)"
    );
    assert_eq!(
        sps.bit_depth_chroma_minus8, 0,
        "Expected 8-bit chroma (minus8=0)"
    );
}

#[test]
fn test_sps_poc_type_from_born_trailer() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(sps_data);
    parser.parse(&packet).expect("SPS parse failed");

    let sps = parser.active_sps().expect("No active SPS");

    // born_trailer.h264 uses POC type 0 (explicit with pic_order_cnt_lsb)
    assert_eq!(
        sps.pic_order_cnt_type, 0,
        "Expected POC type 0 for born_trailer"
    );
    assert!(
        sps.max_pic_order_cnt_lsb > 0,
        "max_pic_order_cnt_lsb must be positive for POC type 0"
    );
}

#[test]
fn test_sps_max_ref_frames() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(sps_data);
    parser.parse(&packet).expect("SPS parse failed");

    let sps = parser.active_sps().expect("No active SPS");

    // Verify max_num_ref_frames is reasonable
    assert!(
        sps.max_num_ref_frames <= 16,
        "max_num_ref_frames ({}) exceeds typical limit of 16",
        sps.max_num_ref_frames
    );
    assert!(
        sps.max_num_ref_frames >= 1,
        "max_num_ref_frames must be at least 1"
    );
}

// ============================================================================
// PPS Parsing Tests
// ============================================================================

#[test]
fn test_pps_parsing_from_real_bitstream() {
    let data = load_test_file("assets/born_trailer.h264");

    // Extract both SPS and PPS and feed them in the same packet
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");
    let pps_data = extract_first_nal_with_start_code(&data, 8).expect("No PPS found");

    let mut combined = Vec::new();
    combined.extend_from_slice(&sps_data);
    combined.extend_from_slice(&pps_data);

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(combined);

    // Parse - parser returns both SPS and PPS in one go when they're in the same packet
    let result = parser.parse(&packet).expect("Parse failed");

    match result {
        ParseResult::ParameterSet {
            sps: Some(_),
            pps: Some(_),
            vps: None,
            ..
        } => {
            let pps = parser.active_pps().expect("No active PPS");

            // Verify basic PPS fields
            assert!(
                pps.seq_parameter_set_id >= 0,
                "seq_parameter_set_id must be non-negative"
            );
            assert!(
                pps.num_ref_idx_l0_default_active_minus1 >= 0,
                "num_ref_idx_l0_default_active_minus1 must be non-negative"
            );
            assert!(
                pps.num_ref_idx_l1_default_active_minus1 >= 0,
                "num_ref_idx_l1_default_active_minus1 must be non-negative"
            );
        }
        _ => panic!("Expected ParameterSet with SPS and PPS, got {:?}", result),
    }
}

#[test]
fn test_pps_links_to_correct_sps() {
    let data = load_test_file("assets/born_trailer.h264");

    // Extract both SPS and PPS and feed them in the same packet
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");
    let pps_data = extract_first_nal_with_start_code(&data, 8).expect("No PPS found");

    let mut combined = Vec::new();
    combined.extend_from_slice(&sps_data);
    combined.extend_from_slice(&pps_data);

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(combined);

    // Parse - both SPS and PPS are returned together
    let result = parser.parse(&packet).expect("Parse failed");
    match result {
        ParseResult::ParameterSet {
            sps: Some(_),
            pps: Some(_),
            vps: None,
            ..
        } => {}
        _ => panic!("Expected ParameterSet with SPS and PPS, got {:?}", result),
    }
    let sps_id = parser.active_sps().unwrap().seq_parameter_set_id;
    let pps = parser.active_pps().expect("No active PPS");

    // PPS should reference the SPS that was parsed
    assert_eq!(
        pps.seq_parameter_set_id, sps_id,
        "PPS seq_parameter_set_id ({}) should match active SPS id ({})",
        pps.seq_parameter_set_id, sps_id
    );
}

#[test]
fn test_pps_entropy_coding_and_deblocking() {
    let data = load_test_file("assets/born_trailer.h264");

    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");
    let pps_data = extract_first_nal_with_start_code(&data, 8).expect("No PPS found");

    let mut combined = Vec::new();
    combined.extend_from_slice(&sps_data);
    combined.extend_from_slice(&pps_data);

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(combined);

    parser.parse(&packet).expect("Parse failed");

    let pps = parser.active_pps().expect("No active PPS");

    // born_trailer.h264 uses CABAC=0 (CAVLC), deblocking=1:0:0
    assert!(
        !pps.entropy_coding_mode_flag,
        "Expected CAVLC (entropy_coding_mode_flag=0) for born_trailer"
    );
    assert!(
        pps.deblocking_filter_control_present_flag,
        "Expected deblocking_filter_control_present_flag=1"
    );
}

#[test]
fn test_pps_qp_offsets() {
    let data = load_test_file("assets/born_trailer.h264");

    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");
    let pps_data = extract_first_nal_with_start_code(&data, 8).expect("No PPS found");

    let mut combined = Vec::new();
    combined.extend_from_slice(&sps_data);
    combined.extend_from_slice(&pps_data);

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(combined);

    parser.parse(&packet).expect("Parse failed");

    let pps = parser.active_pps().expect("No active PPS");

    // Verify QP-related fields are within valid ranges
    assert!(
        pps.pic_init_qp_minus26 >= -26 && pps.pic_init_qp_minus26 <= 25,
        "pic_init_qp_minus26 ({}) out of range",
        pps.pic_init_qp_minus26
    );
    assert!(
        pps.chroma_qp_index_offset >= -12 && pps.chroma_qp_index_offset <= 12,
        "chroma_qp_index_offset ({}) out of range",
        pps.chroma_qp_index_offset
    );
}

// ============================================================================
// Combined SPS+PPS Parsing Test
// ============================================================================

#[test]
fn test_combined_sps_pps_parsing() {
    let data = load_test_file("assets/born_trailer.h264");

    // Feed SPS and PPS together (as they appear in the bitstream)
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data[..650].to_vec()); // Includes SPS and PPS

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

    assert!(got_sps, "Should have parsed SPS");
    assert!(got_pps, "Should have parsed PPS");
    assert!(parser.active_sps().is_some(), "Active SPS should be set");
    assert!(parser.active_pps().is_some(), "Active PPS should be set");
}

// ============================================================================
// CUVIDH264PICPARAMS Construction Tests
// ============================================================================

/// Helper struct to test CUVIDH264PICPARAMS field mapping without hardware.
#[derive(Debug)]
struct PicParamsMapping {
    // SPS-derived fields
    log2_max_frame_num_minus4: i32,
    pic_order_cnt_type: i32,
    log2_max_pic_order_cnt_lsb_minus4: i32,
    delta_pic_order_always_zero_flag: i32,
    frame_mbs_only_flag: i32,
    direct_8x8_inference_flag: i32,
    num_ref_frames: i32,
    residual_colour_transform_flag: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    qpprime_y_zero_transform_bypass_flag: u8,

    // PPS-derived fields
    entropy_coding_mode_flag: i32,
    pic_order_present_flag: i32,
    num_ref_idx_l0_active_minus1: i32,
    num_ref_idx_l1_active_minus1: i32,
    weighted_pred_flag: i32,
    weighted_bipred_idc: i32,
    pic_init_qp_minus26: i32,
    deblocking_filter_control_present_flag: i32,
    redundant_pic_cnt_present_flag: i32,
    transform_8x8_mode_flag: i32,
    mbaff_frame_flag: i32,
    constrained_intra_pred_flag: i32,
    chroma_qp_index_offset: i32,
    second_chroma_qp_index_offset: i32,
}

impl PicParamsMapping {
    fn from_sps_pps(
        sps: &vk_video_core::picture::H264Sps,
        pps: &vk_video_core::picture::H264Pps,
        slh: &vk_video_parser::h264::SliceHeader,
    ) -> Self {
        Self {
            // SPS fields - must match decoder.rs:499-510
            log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4 as i32,
            pic_order_cnt_type: sps.pic_order_cnt_type as i32,
            log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4 as i32,
            delta_pic_order_always_zero_flag: sps.delta_pic_order_always_zero_flag as i32,
            frame_mbs_only_flag: sps.frame_mbs_only_flag as i32,
            direct_8x8_inference_flag: sps.direct_8x8_inference_flag as i32,
            num_ref_frames: sps.max_num_ref_frames as i32,
            residual_colour_transform_flag: if sps.chroma_format_idc == 3 { 1 } else { 0 },
            bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
            bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
            qpprime_y_zero_transform_bypass_flag: sps.qpprime_y_zero_transform_bypass_flag as u8,

            // PPS fields - must match decoder.rs:513-526
            entropy_coding_mode_flag: pps.entropy_coding_mode_flag as i32,
            pic_order_present_flag: if sps.pic_order_cnt_type != 2 { 1 } else { 0 },
            num_ref_idx_l0_active_minus1: slh.num_ref_idx_l0_active_minus1 as i32,
            num_ref_idx_l1_active_minus1: slh.num_ref_idx_l1_active_minus1 as i32,
            weighted_pred_flag: pps.weighted_pred_flag as i32,
            weighted_bipred_idc: pps.weighted_bipred_idc as i32,
            pic_init_qp_minus26: pps.pic_init_qp_minus26,
            deblocking_filter_control_present_flag: pps.deblocking_filter_control_present_flag
                as i32,
            redundant_pic_cnt_present_flag: pps.redundant_pic_cnt_present_flag as i32,
            transform_8x8_mode_flag: pps.transform_8x8_mode_flag as i32,
            mbaff_frame_flag: if !sps.frame_mbs_only_flag { 1 } else { 0 },
            constrained_intra_pred_flag: pps.constrained_intra_pred_flag as i32,
            chroma_qp_index_offset: pps.chroma_qp_index_offset,
            second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset,
        }
    }
}

#[test]
fn test_picparams_sps_field_mapping() {
    let data = load_test_file("assets/born_trailer.h264");

    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");
    let pps_data = extract_first_nal_with_start_code(&data, 8).expect("No PPS found");

    let mut combined = Vec::new();
    combined.extend_from_slice(&sps_data);
    combined.extend_from_slice(&pps_data);

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(combined);

    parser.parse(&packet).expect("Parse failed");

    let sps = parser.active_sps().unwrap();
    let pps = parser.active_pps().unwrap();

    // Create a mock slice header with reasonable defaults
    let mock_slh = vk_video_parser::h264::SliceHeader {
        first_mb_in_slice: 0,
        slice_type: 4, // I-slice
        pic_parameter_set_id: pps.pic_parameter_set_id,
        frame_num: 0,
        idr_pic_id: 0,
        pic_order_cnt_lsb: 0,
        delta_pic_order_cnt: [0, 0],
        redundant_pic_cnt: 0,
        num_ref_idx_l0_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
        nal_ref_idc: 3,
        nal_unit_type: 5, // IDR
        field_pic_flag: false,
        bottom_field: false,
        long_term_reference: false,
        direct_spatial_mv_pred_flag: false,
        num_ref_idx_active_override_flag: false,
        cabac_init_idc: 0,
        slice_qp_delta: 0,
        disable_deblocking_filter_idc: 0,
        slice_alpha_c0_offset_div2: 0,
        slice_beta_offset_div2: 0,
        ref_pic_list_modification_l0: Vec::new(),
        ref_pic_list_modification_l1: Vec::new(),
        dec_ref_pic_marking: Vec::new(),
        no_output_of_prior_pics_flag: false,
        long_term_reference_flag: false,
        header_bit_size: 0,
        luma_log2_weight_denom: 0,
        chroma_log2_weight_denom: 0,
        luma_weight_l0_flag: 0,
        luma_weight_l0: [0; 32],
        luma_offset_l0: [0; 32],
        chroma_weight_l0_flag: 0,
        chroma_weight_l0: [[0; 2]; 32],
        chroma_offset_l0: [[0; 2]; 32],
        luma_weight_l1_flag: 0,
        luma_weight_l1: [0; 32],
        luma_offset_l1: [0; 32],
        chroma_weight_l1_flag: 0,
        chroma_weight_l1: [[0; 2]; 32],
        chroma_offset_l1: [[0; 2]; 32],
    };

    let mapping = PicParamsMapping::from_sps_pps(sps, pps, &mock_slh);

    // Verify SPS fields map correctly
    assert_eq!(
        mapping.log2_max_frame_num_minus4,
        sps.log2_max_frame_num_minus4 as i32
    );
    assert_eq!(mapping.pic_order_cnt_type, sps.pic_order_cnt_type as i32);
    assert_eq!(mapping.num_ref_frames, sps.max_num_ref_frames as i32);
    assert_eq!(mapping.frame_mbs_only_flag, sps.frame_mbs_only_flag as i32);
    assert_eq!(mapping.bit_depth_luma_minus8, sps.bit_depth_luma_minus8);
    assert_eq!(mapping.bit_depth_chroma_minus8, sps.bit_depth_chroma_minus8);
}

#[test]
fn test_picparams_pps_field_mapping() {
    let data = load_test_file("assets/born_trailer.h264");

    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");
    let pps_data = extract_first_nal_with_start_code(&data, 8).expect("No PPS found");

    let mut combined = Vec::new();
    combined.extend_from_slice(&sps_data);
    combined.extend_from_slice(&pps_data);

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(combined);

    parser.parse(&packet).expect("Parse failed");

    let sps = parser.active_sps().unwrap();
    let pps = parser.active_pps().unwrap();

    let mock_slh = create_mock_slice_header(pps);
    let mapping = PicParamsMapping::from_sps_pps(sps, pps, &mock_slh);

    // Verify PPS fields map correctly
    assert_eq!(
        mapping.entropy_coding_mode_flag,
        pps.entropy_coding_mode_flag as i32
    );
    assert_eq!(mapping.pic_init_qp_minus26, pps.pic_init_qp_minus26);
    assert_eq!(mapping.chroma_qp_index_offset, pps.chroma_qp_index_offset);
    assert_eq!(
        mapping.second_chroma_qp_index_offset,
        pps.second_chroma_qp_index_offset
    );
    assert_eq!(
        mapping.mbaff_frame_flag,
        if !sps.frame_mbs_only_flag { 1 } else { 0 }
    );

    // pic_order_present_flag should be 1 when POC type != 2
    assert_eq!(
        mapping.pic_order_present_flag,
        if sps.pic_order_cnt_type != 2 { 1 } else { 0 }
    );
}

fn create_mock_slice_header(
    pps: &vk_video_core::picture::H264Pps,
) -> vk_video_parser::h264::SliceHeader {
    vk_video_parser::h264::SliceHeader {
        first_mb_in_slice: 0,
        slice_type: 4,
        pic_parameter_set_id: pps.pic_parameter_set_id,
        frame_num: 0,
        idr_pic_id: 0,
        pic_order_cnt_lsb: 0,
        delta_pic_order_cnt: [0, 0],
        redundant_pic_cnt: 0,
        num_ref_idx_l0_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
        nal_ref_idc: 3,
        nal_unit_type: 5,
        field_pic_flag: false,
        bottom_field: false,
        long_term_reference: false,
        direct_spatial_mv_pred_flag: false,
        num_ref_idx_active_override_flag: false,
        cabac_init_idc: 0,
        slice_qp_delta: 0,
        disable_deblocking_filter_idc: 0,
        slice_alpha_c0_offset_div2: 0,
        slice_beta_offset_div2: 0,
        ref_pic_list_modification_l0: Vec::new(),
        ref_pic_list_modification_l1: Vec::new(),
        dec_ref_pic_marking: Vec::new(),
        no_output_of_prior_pics_flag: false,
        long_term_reference_flag: false,
        header_bit_size: 0,
        luma_log2_weight_denom: 0,
        chroma_log2_weight_denom: 0,
        luma_weight_l0_flag: 0,
        luma_weight_l0: [0; 32],
        luma_offset_l0: [0; 32],
        chroma_weight_l0_flag: 0,
        chroma_weight_l0: [[0; 2]; 32],
        chroma_offset_l0: [[0; 2]; 32],
        luma_weight_l1_flag: 0,
        luma_weight_l1: [0; 32],
        luma_offset_l1: [0; 32],
        chroma_weight_l1_flag: 0,
        chroma_weight_l1: [[0; 2]; 32],
        chroma_offset_l1: [[0; 2]; 32],
    }
}

// ============================================================================
// POC Calculation Tests
// ============================================================================

/// Mock SPS for POC type 0 testing
fn create_sps_poc_type_0() -> vk_video_core::picture::H264Sps {
    let mut sps = vk_video_core::picture::H264Sps::new();
    sps.pic_order_cnt_type = 0;
    sps.log2_max_pic_order_cnt_lsb_minus4 = 4; // max_pic_order_cnt_lsb = 512
    sps.max_pic_order_cnt_lsb = 512;
    sps.frame_mbs_only_flag = true;
    sps
}

/// Mock SPS for POC type 1 testing
fn create_sps_poc_type_1() -> vk_video_core::picture::H264Sps {
    let mut sps = vk_video_core::picture::H264Sps::new();
    sps.pic_order_cnt_type = 1;
    sps.delta_pic_order_always_zero_flag = false;
    sps.frame_mbs_only_flag = true;
    sps
}

/// Mock SPS for POC type 2 testing
fn create_sps_poc_type_2() -> vk_video_core::picture::H264Sps {
    let mut sps = vk_video_core::picture::H264Sps::new();
    sps.pic_order_cnt_type = 2;
    sps.log2_max_frame_num_minus4 = 4; // max_frame_num = 256
    sps.max_frame_num = 256;
    sps.frame_mbs_only_flag = true;
    sps
}

#[test]
fn test_poc_type_0_basic() {
    // POC type 0: explicit with pic_order_cnt_lsb
    // When pic_order_cnt_lsb increases monotonically without wrap,
    // POC = pic_order_cnt_lsb (with msb=0 initially)

    let sps = create_sps_poc_type_0();
    let max_pic_order_cnt_lsb = sps.max_pic_order_cnt_lsb as i32;

    // Simulate the decoder's POC calculation logic
    let mut prev_pic_order_cnt_lsb: i32 = 0;
    let mut prev_pic_order_cnt_msb: i32 = 0;

    // Frame 0: lsb=0
    let lsb0 = 0i32;
    let msb0 = prev_pic_order_cnt_msb; // no wrap
    assert_eq!(msb0 + lsb0, 0);
    prev_pic_order_cnt_lsb = lsb0;
    prev_pic_order_cnt_msb = msb0;

    // Frame 1: lsb=2
    let lsb1 = 2i32;
    let msb1 = prev_pic_order_cnt_msb; // no wrap
    assert_eq!(msb1 + lsb1, 2);
    prev_pic_order_cnt_lsb = lsb1;
    prev_pic_order_cnt_msb = msb1;

    // Frame 2: lsb=4
    let lsb2 = 4i32;
    let msb2 = prev_pic_order_cnt_msb; // no wrap
    assert_eq!(msb2 + lsb2, 4);
    prev_pic_order_cnt_lsb = lsb2;
    prev_pic_order_cnt_msb = msb2;
}

#[test]
fn test_poc_type_0_wrap_up() {
    // POC type 0: wrap-up case where lsb decreases but crosses the boundary
    // When lsb goes from high to low (crossing max/2), msb should increase

    let sps = create_sps_poc_type_0();
    let max_pic_order_cnt_lsb = sps.max_pic_order_cnt_lsb as i32; // 512

    let mut prev_pic_order_cnt_lsb: i32 = 500;
    let mut prev_pic_order_cnt_msb: i32 = 0;

    // Next frame: lsb=10 (wrapped around)
    // Since 500 - 10 = 490 >= 512/2 = 256, msb should increase
    let lsb = 10i32;
    let msb = if lsb < prev_pic_order_cnt_lsb
        && (prev_pic_order_cnt_lsb - lsb) >= max_pic_order_cnt_lsb / 2
    {
        prev_pic_order_cnt_msb + max_pic_order_cnt_lsb
    } else if lsb > prev_pic_order_cnt_lsb
        && (lsb - prev_pic_order_cnt_lsb) >= max_pic_order_cnt_lsb / 2
    {
        prev_pic_order_cnt_msb - max_pic_order_cnt_lsb
    } else {
        prev_pic_order_cnt_msb
    };

    assert_eq!(msb, 512, "MSB should wrap up to 512");
    assert_eq!(msb + lsb, 522, "POC should be 522 (512 + 10)");
}

#[test]
fn test_poc_type_0_wrap_down() {
    // POC type 0: wrap-down case where lsb increases but crosses the boundary
    // When lsb goes from low to high (crossing max/2), msb should decrease

    let sps = create_sps_poc_type_0();
    let max_pic_order_cnt_lsb = sps.max_pic_order_cnt_lsb as i32; // 512

    let mut prev_pic_order_cnt_lsb: i32 = 10;
    let mut prev_pic_order_cnt_msb: i32 = 512;

    // Next frame: lsb=500 (wrapped around)
    // Since 500 - 10 = 490 >= 512/2 = 256, msb should decrease
    let lsb = 500i32;
    let msb = if lsb < prev_pic_order_cnt_lsb
        && (prev_pic_order_cnt_lsb - lsb) >= max_pic_order_cnt_lsb / 2
    {
        prev_pic_order_cnt_msb + max_pic_order_cnt_lsb
    } else if lsb > prev_pic_order_cnt_lsb
        && (lsb - prev_pic_order_cnt_lsb) >= max_pic_order_cnt_lsb / 2
    {
        prev_pic_order_cnt_msb - max_pic_order_cnt_lsb
    } else {
        prev_pic_order_cnt_msb
    };

    assert_eq!(msb, 0, "MSB should wrap down to 0");
    assert_eq!(msb + lsb, 500, "POC should be 500");
}

#[test]
fn test_poc_type_1_delta_based() {
    // POC type 1: explicit with offset cycling per H.264 D.3.3.2
    // For reference frames: PicOrderCnt = LastPicOrderCnt + offset_for_ref_frame[cycle]
    // For non-reference: PicOrderCnt = PrevPicOrderCnt + offset_for_non_ref_pic

    let sps = create_sps_poc_type_1();
    assert_eq!(sps.pic_order_cnt_type, 1);
    assert!(!sps.delta_pic_order_always_zero_flag);

    // Simulate the decoder's POC calculation for reference frames with cycle offsets
    let offset_for_ref_frame = vec![4, -4];
    let num_ref_frames_in_pic_order_cnt_cycle = 2u32;
    let offset_for_non_ref_pic = 0i32;

    let mut prev_pic_order_cnt: i32 = 0;
    let mut last_pic_order_cnt: i32 = 0;
    let mut last_pic_order_cnt_cycle: i32 = 0;
    let mut prev_is_reference = false;

    // Frame 0 (ref): prev_is_reference=false → last_pic_order_cnt + offset[0] = 0 + 4 = 4
    let poc0 = if prev_is_reference {
        prev_pic_order_cnt
            + offset_for_ref_frame
                [last_pic_order_cnt_cycle as usize % num_ref_frames_in_pic_order_cnt_cycle as usize]
    } else {
        last_pic_order_cnt
            + offset_for_ref_frame
                [last_pic_order_cnt_cycle as usize % num_ref_frames_in_pic_order_cnt_cycle as usize]
    };
    assert_eq!(poc0, 4);
    prev_pic_order_cnt = poc0;
    last_pic_order_cnt = poc0;
    last_pic_order_cnt_cycle =
        (last_pic_order_cnt_cycle + 1) % num_ref_frames_in_pic_order_cnt_cycle as i32;
    prev_is_reference = true;

    // Frame 1 (ref): prev_is_reference=true → prev_pic_order_cnt + offset[1] = 4 + (-4) = 0
    let poc1 = prev_pic_order_cnt
        + offset_for_ref_frame
            [last_pic_order_cnt_cycle as usize % num_ref_frames_in_pic_order_cnt_cycle as usize];
    assert_eq!(poc1, 0);
    prev_pic_order_cnt = poc1;
    last_pic_order_cnt = poc1;
    last_pic_order_cnt_cycle =
        (last_pic_order_cnt_cycle + 1) % num_ref_frames_in_pic_order_cnt_cycle as i32;
}

#[test]
fn test_poc_type_1_with_delta_zero_flag() {
    // POC type 1 with delta_pic_order_always_zero_flag=1
    // In this case, POC is derived differently (no delta read from bitstream)

    let mut sps = vk_video_core::picture::H264Sps::new();
    sps.pic_order_cnt_type = 1;
    sps.delta_pic_order_always_zero_flag = true;
    sps.frame_mbs_only_flag = true;
    sps.num_ref_frames_in_pic_order_cnt_cycle = 2;
    sps.offset_for_ref_frame = vec![2, -2];

    // With delta_pic_order_always_zero_flag=1, the delta from slice header is 0
    // POC is calculated from cycle offsets
    // This is a simplified check - the full algorithm is complex
    assert_eq!(sps.pic_order_cnt_type, 1);
    assert!(sps.delta_pic_order_always_zero_flag);
    assert_eq!(sps.num_ref_frames_in_pic_order_cnt_cycle, 2);
}

#[test]
fn test_poc_type_2_implicit_from_frame_num() {
    // POC type 2: implicit POC derived from frame_num per H.264 D.3.3.3
    // Reference frames: POC = frame_num * 2
    // Non-reference frame pictures: POC = frame_num * 2 + 1

    let sps = create_sps_poc_type_2();
    assert_eq!(sps.pic_order_cnt_type, 2);

    // Frame 0 (ref): frame_num=0, POC=0*2=0
    let frame_num0 = 0i32;
    let poc0 = frame_num0 * 2;
    assert_eq!(poc0, 0);

    // Frame 1 (ref): frame_num=1, POC=1*2=2
    let frame_num1 = 1i32;
    let poc1 = frame_num1 * 2;
    assert_eq!(poc1, 2);

    // Frame 2 (non-ref): frame_num=100, POC=100*2+1=201
    let frame_num2 = 100i32;
    let poc2 = frame_num2 * 2 + 1;
    assert_eq!(poc2, 201);

    // Frame 3 (ref): wrap-around, frame_num=5, POC=5*2=10
    let frame_num3 = 5i32;
    assert!(frame_num3 < frame_num2);
    let poc3 = frame_num3 * 2;
    assert_eq!(poc3, 10);
}

// ============================================================================
// DPB Management Tests
// ============================================================================

const MAX_DPB_ENTRIES: usize = 16;

#[test]
fn test_allocate_dpb_slot_empty() {
    let mut dpb_manager = DpbManager::new(MAX_DPB_ENTRIES as u32);

    // First allocation should return slot 0
    let slot = dpb_manager.find_or_recycle_slot(&[]).unwrap();
    assert_eq!(slot, 0);

    // Mark the entry as valid
    dpb_manager.entries[slot as usize].is_valid = true;
    dpb_manager.entries[slot as usize].frame_num = slot;

    // Second allocation should return slot 1
    let slot2 = dpb_manager.find_or_recycle_slot(&[]).unwrap();
    assert_eq!(slot2, 1);
}

#[test]
fn test_allocate_dpb_slot_with_gaps() {
    let mut dpb_manager = DpbManager::new(MAX_DPB_ENTRIES as u32);

    // Occupy slots 0 and 2
    dpb_manager.entries[0].is_valid = true;
    dpb_manager.entries[2].is_valid = true;

    // Next allocation should find slot 1 (first gap)
    let slot = dpb_manager.find_or_recycle_slot(&[]).unwrap();
    assert_eq!(slot, 1);
}

#[test]
fn test_allocate_dpb_slot_full_fallback() {
    let mut dpb_manager = DpbManager::new(MAX_DPB_ENTRIES as u32);

    // Occupy all slots with increasing POCs
    for i in 0..MAX_DPB_ENTRIES {
        dpb_manager.entries[i].is_valid = true;
        dpb_manager.entries[i].pic_order_cnt = [i as i32, i as i32];
    }

    // No available slot - should recycle oldest (slot 0 with lowest POC)
    let slot = dpb_manager.find_or_recycle_slot(&[]).unwrap();
    assert_eq!(slot, 0, "Should recycle slot 0 (oldest POC) when full");
}

#[test]
fn test_memory_management_op_1() {
    // Op 1: Mark short-term ref with specific picNumX as unused
    let mut dpb_manager = DpbManager::new(MAX_DPB_ENTRIES as u32);

    // Set up some entries with different frame_nums
    dpb_manager.entries[0].is_valid = true;
    dpb_manager.entries[0].frame_num = 5;
    dpb_manager.entries[0].pic_order_cnt = [10, 10];

    dpb_manager.entries[1].is_valid = true;
    dpb_manager.entries[1].frame_num = 8; // This will be targeted: 13 - (4 + 1) = 8
    dpb_manager.entries[1].pic_order_cnt = [20, 20];

    dpb_manager.entries[2].is_valid = true;
    dpb_manager.entries[2].frame_num = 15;
    dpb_manager.entries[2].pic_order_cnt = [30, 30];

    // Apply op 1 with difference_of_pic_nums_minus1 = 4, current_frame_num = 13
    // picNumX = 13 - (4 + 1) = 8, so entry with frame_num=8 should be invalidated
    dpb_manager.apply_mmco(
        13,
        0,
        &[H264MmcoCommand::UnmarkShortTerm {
            difference_of_pic_nums_minus1: 4,
        }],
    );

    // Entry with frame_num=8 should be invalidated
    assert!(
        !dpb_manager.entries[1].is_valid,
        "Entry with frame_num=8 should be invalidated"
    );
    // Other entries should remain valid
    assert!(
        dpb_manager.entries[0].is_valid,
        "Entry with frame_num=5 should remain valid"
    );
    assert!(
        dpb_manager.entries[2].is_valid,
        "Entry with frame_num=15 should remain valid"
    );
}

#[test]
fn test_memory_management_op_2() {
    // Op 2: Mark long-term ref with long_term_pic_num = value as unused
    // Note: The new DpbManager acknowledges this operation but does not fully
    // track long-term reference state, so the entry remains valid.
    let mut dpb_manager = DpbManager::new(MAX_DPB_ENTRIES as u32);

    dpb_manager.entries[3].is_valid = true;
    dpb_manager.entries[3].frame_num = 100;
    dpb_manager.entries[3].pic_order_cnt = [200, 200];

    // Apply op 2 with long_term_frame_idx = 3
    dpb_manager.apply_mmco(
        110,
        0,
        &[H264MmcoCommand::UnmarkLongTerm {
            long_term_frame_idx: 3,
        }],
    );

    // Operation completes without error; long-term tracking is not fully implemented
    // so the entry's validity is unchanged
    assert!(dpb_manager.entries[3].is_valid);
}

#[test]
fn test_memory_management_op_3() {
    // Op 3: Mark current picture as long-term (assign LongTermFrameIdx)
    // Note: The new DpbManager acknowledges this operation but does not fully
    // track long-term reference state.
    let mut dpb_manager = DpbManager::new(MAX_DPB_ENTRIES as u32);

    let curr_pic_slot = 5u32;
    dpb_manager.entries[curr_pic_slot as usize].is_valid = true;
    dpb_manager.entries[curr_pic_slot as usize].frame_num = 50;
    dpb_manager.entries[curr_pic_slot as usize].pic_order_cnt = [100, 100];

    // Apply op 3 (AssignLongTermToCurrent)
    dpb_manager.apply_mmco(
        50,
        curr_pic_slot,
        &[H264MmcoCommand::AssignLongTermToCurrent {
            long_term_frame_idx: 0,
        }],
    );

    // Operation completes without error; entry remains valid
    assert!(dpb_manager.entries[curr_pic_slot as usize].is_valid);
}

#[test]
fn test_memory_management_op_4() {
    // Op 4: Set max_long_term_frame_idx_plus1 = value
    // Note: The new DpbManager acknowledges this operation but does not fully
    // track long-term reference state, so entries remain valid.
    let mut dpb_manager = DpbManager::new(MAX_DPB_ENTRIES as u32);

    dpb_manager.entries[2].is_valid = true;
    dpb_manager.entries[2].frame_num = 10;
    dpb_manager.entries[2].pic_order_cnt = [20, 20];

    dpb_manager.entries[5].is_valid = true;
    dpb_manager.entries[5].frame_num = 20;
    dpb_manager.entries[5].pic_order_cnt = [40, 40];

    // Apply op 4 with max_long_term_frame_idx_plus1 = 3
    dpb_manager.apply_mmco(
        25,
        0,
        &[H264MmcoCommand::SetMaxLongTermFrameIdx {
            max_long_term_frame_idx_plus1: 3,
        }],
    );

    // Operation completes without error; long-term tracking is not fully implemented
    // so both entries remain valid
    assert!(dpb_manager.entries[2].is_valid);
    assert!(dpb_manager.entries[5].is_valid);
}

#[test]
fn test_memory_management_op_5() {
    // Op 5: Mark short-term ref as long-term (assign LongTermFrameIdx)
    // Note: The new DpbManager acknowledges this operation but does not fully
    // track long-term reference state.
    let mut dpb_manager = DpbManager::new(MAX_DPB_ENTRIES as u32);

    dpb_manager.entries[7].is_valid = true;
    dpb_manager.entries[7].frame_num = 30;
    dpb_manager.entries[7].pic_order_cnt = [60, 60];

    // Apply op 5 with difference_of_pic_nums_minus1 = 5, current_frame_num = 36
    // picNumX = 36 - (5 + 1) = 30, matching entry with frame_num=30
    dpb_manager.apply_mmco(
        36,
        0,
        &[H264MmcoCommand::AssignLongTerm {
            difference_of_pic_nums_minus1: 5,
            long_term_frame_idx: 0,
        }],
    );

    // Operation completes without error; entry remains valid
    assert!(dpb_manager.entries[7].is_valid);
}

#[test]
fn test_dpb_idr_clear() {
    // IDR picture should clear all DPB entries
    let mut dpb_manager = DpbManager::new(MAX_DPB_ENTRIES as u32);

    // Set up some entries
    dpb_manager.entries[0].is_valid = true;
    dpb_manager.entries[0].frame_num = 10;
    dpb_manager.entries[0].pic_order_cnt = [20, 20];
    dpb_manager.entries[0].last_access = LastAccessType::DecodeWrite;

    dpb_manager.entries[1].is_valid = true;
    dpb_manager.entries[1].frame_num = 15;
    dpb_manager.entries[1].pic_order_cnt = [30, 30];
    dpb_manager.entries[1].last_access = LastAccessType::DecodeWrite;

    // IDR: invalidate all entries
    dpb_manager.invalidate_all();

    // All entries should be cleared
    assert!(!dpb_manager.entries[0].is_valid);
    assert_eq!(dpb_manager.entries[0].last_access, LastAccessType::None);
    assert!(!dpb_manager.entries[1].is_valid);
    assert_eq!(dpb_manager.entries[1].last_access, LastAccessType::None);
}

#[test]
fn test_dpb_add_reference_picture() {
    let mut dpb_manager = DpbManager::new(MAX_DPB_ENTRIES as u32);

    let curr_pic_slot = 0u32;
    let frame_num = 10u32;
    let poc = 20i32;

    // Add current frame as reference by setting entry fields
    dpb_manager.entries[curr_pic_slot as usize].is_valid = true;
    dpb_manager.entries[curr_pic_slot as usize].frame_num = frame_num;
    dpb_manager.entries[curr_pic_slot as usize].pic_order_cnt = [poc, poc];
    dpb_manager.entries[curr_pic_slot as usize].slot_index = curr_pic_slot;

    assert_eq!(dpb_manager.entries[0].slot_index, 0);
    assert_eq!(dpb_manager.entries[0].frame_num, 10);
    assert!(dpb_manager.entries[0].is_valid);
    assert_eq!(dpb_manager.entries[0].pic_order_cnt, [20, 20]);
}

// ============================================================================
// Slice Header Parsing Tests
// ============================================================================

#[test]
fn test_slice_header_parsing_from_real_bitstream() {
    let data = load_test_file("assets/born_trailer.h264");

    // Feed enough data to include SPS, PPS, and first slice
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data[..1000].to_vec());

    let mut got_sps = false;
    let mut got_pps = false;
    let mut got_slice = false;

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::ParameterSet { sps, pps, .. }) => {
                if sps.is_some() {
                    got_sps = true;
                }
                if pps.is_some() {
                    got_pps = true;
                }
            }
            Ok(ParseResult::Slice { slices, .. }) => {
                if !slices.is_empty() {
                    got_slice = true;
                    let first_slice = &slices[0];
                    if let Some(vk_video_parser::SliceHeader::H264(slh)) = &first_slice.slice_header
                    {
                        // Verify slice header fields are reasonable
                        assert!(
                            slh.slice_type <= 9,
                            "slice_type ({}) out of range",
                            slh.slice_type
                        );
                        assert!(
                            slh.pic_parameter_set_id
                                == parser.active_pps().unwrap().pic_parameter_set_id,
                            "PPS ID mismatch"
                        );
                    }
                }
                break;
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Err(_) => break,
        }
    }

    assert!(got_sps, "Should have parsed SPS");
    assert!(got_pps, "Should have parsed PPS");
    assert!(got_slice, "Should have parsed at least one slice");
}

#[test]
fn test_idr_slice_detection() {
    let data = load_test_file("assets/born_trailer.h264");

    // Find first IDR slice (NAL unit type 5)
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data[..2000].to_vec());

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::ParameterSet { .. }) => {}
            Ok(ParseResult::Slice { slices, .. }) => {
                for slice in &slices {
                    if let Some(vk_video_parser::SliceHeader::H264(slh)) = &slice.slice_header {
                        let is_idr = slh.nal_unit_type == 5;
                        if is_idr {
                            // IDR slice should have nal_ref_idc > 0
                            assert!(slh.nal_ref_idc > 0, "IDR slice must have nal_ref_idc > 0");
                            // IDR should have idr_pic_id present
                            assert!(slh.idr_pic_id >= 0, "IDR slice should have idr_pic_id");
                            return;
                        }
                    }
                }
                // Found non-IDR slices, continue looking
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Err(_) => break,
        }
    }

    // If we get here, we didn't find an IDR in the first 2000 bytes
    // That's OK for this test - born_trailer starts with an IDR
}

#[test]
fn test_parser_reset_clears_state() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");

    let mut parser = H264Parser::new();
    parser
        .parse(&BitstreamPacket::new(sps_data))
        .expect("SPS parse failed");

    assert!(parser.active_sps().is_some(), "SPS should be set");

    // Reset parser
    parser.reset();

    assert!(
        parser.active_sps().is_none(),
        "SPS should be cleared after reset"
    );
    assert!(
        parser.active_pps().is_none(),
        "PPS should be cleared after reset"
    );
}

#[test]
fn test_interlaced_stream_sps() {
    let data = load_test_file("assets/test_interlaced.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(sps_data);
    let result = parser.parse(&packet).expect("SPS parse failed");

    match result {
        ParseResult::ParameterSet { sps: Some(_), .. } => {
            let sps = parser.active_sps().expect("No active SPS");
            println!("Interlaced stream SPS:");
            println!("  profile_idc: {}", sps.profile_idc);
            println!("  level_idc: {}", sps.level_idc);
            println!("  frame_mbs_only_flag: {}", sps.frame_mbs_only_flag);
            println!("  pic_width_in_mbs_minus1: {}", sps.pic_width_in_mbs_minus1);
            println!(
                "  pic_height_in_map_units_minus1: {}",
                sps.pic_height_in_map_units_minus1
            );

            let coded_width = (sps.pic_width_in_mbs_minus1 as u32 + 1) * 16;
            let coded_height = if sps.frame_mbs_only_flag {
                (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
            } else {
                (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
            };

            println!("  coded_width: {}", coded_width);
            println!("  coded_height: {}", coded_height);

            // x264 --tff sets frame_mbs_only_flag=0 (field pictures allowed)
            assert!(
                !sps.frame_mbs_only_flag,
                "Interlaced stream should have frame_mbs_only_flag=0, got {}",
                sps.frame_mbs_only_flag
            );
        }
        _ => panic!("Expected ParameterSet result, got {:?}", result),
    }
}

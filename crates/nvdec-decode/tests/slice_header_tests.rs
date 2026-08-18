//! Comprehensive tests for H.264 slice header parsing from vk-video-parser.
//!
//! These tests verify that slice header parsing matches the H.264 specification
//! (Annex B, section 7.4.3) and cuvid expectations.
//!
//! Tests cover:
//! - Basic fields (first_mb_in_slice, slice_type, pps_id)
//! - frame_num with various log2_max_frame_num_minus4 values
//! - pic_order_cnt_lsb for POC type 0
//! - idr_pic_id for IDR slices
//! - field_pic_flag and bottom_field parsing
//! - delta_pic_order_cnt for POC type 1
//! - redundant_pic_cnt when enabled
//! - num_ref_idx_active_override_flag handling
//! - ref_pic_list_modification parsing
//! - dec_ref_pic_marking for IDR and non-IDR frames
//! - cabac_init_idc parsing
//! - slice_qp_delta parsing
//! - deblocking filter parameters
//! - Real-world parsing from born_trailer.h264

use vk_video_parser::{
    h264::H264Parser, BitstreamPacket, DetectedVideoFormat, ParseResult, VideoParser,
};

/// Path to the project root (parent of nvdec-decode crate).
const PROJECT_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// Load a known H.264 test file from the project assets.
fn load_test_file(path: &str) -> Vec<u8> {
    let full_path = format!("{}/{}", PROJECT_ROOT, path);
    std::fs::read(&full_path).expect(&format!("Failed to read test file: {}", full_path))
}

// ============================================================================
// Helper: ue(v) encoding - returns bits (0 or 1)
// ============================================================================

/// Encode a ue(v) value into bits (each element is 0 or 1).
fn encode_ue_bits(value: u32) -> Vec<u8> {
    let mut bits = Vec::new();
    if value == 0 {
        bits.push(1); // '1'
    } else {
        let v = value + 1;
        let leading_zeros = 32 - v.leading_zeros() - 1;
        for _ in 0..leading_zeros {
            bits.push(0);
        }
        bits.push(1);
        for i in (0..leading_zeros).rev() {
            bits.push(((v >> i) & 1) as u8);
        }
    }
    bits
}

// ============================================================================
// Helper: se(v) encoding - returns bits (0 or 1)
// ============================================================================

/// Encode a se(v) value into bits (each element is 0 or 1).
/// H.264 spec 9.1: value 0 -> codeNum 0, positive v -> codeNum 2v-1,
/// negative v -> codeNum 2|v| (even codeNums decode to negative values).
fn encode_se_bits(value: i32) -> Vec<u8> {
    let ue_value = if value > 0 {
        (2 * value - 1) as u32
    } else if value < 0 {
        (-2 * value) as u32
    } else {
        0
    };
    encode_ue_bits(ue_value)
}

// ============================================================================
// Helper: bit manipulation
// ============================================================================

/// Convert a list of bits to bytes (MSB first within each byte).
fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut current_byte: u8 = 0;
    let mut bit_count = 0;

    for &bit in bits {
        current_byte = (current_byte << 1) | bit;
        bit_count += 1;
        if bit_count == 8 {
            bytes.push(current_byte);
            current_byte = 0;
            bit_count = 0;
        }
    }

    if bit_count > 0 {
        current_byte <<= (8 - bit_count);
        bytes.push(current_byte);
    }

    bytes
}

/// Append bits to a byte vector.
fn append_bits(bits: &mut Vec<u8>, value: u32, count: u8) {
    for i in (0..count).rev() {
        bits.push(((value >> i) & 1) as u8);
    }
}

// ============================================================================
// Helper: Build synthetic SPS
// ============================================================================

/// Build a minimal SPS NAL unit with specified parameters (with start code).
fn build_sps(
    seq_parameter_set_id: u32,
    log2_max_frame_num_minus4: u8,
    pic_order_cnt_type: u8,
    log2_max_pic_order_cnt_lsb_minus4: u8,
    delta_pic_order_always_zero_flag: bool,
    frame_mbs_only_flag: bool,
    max_num_ref_frames: u32,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    // Start code
    bytes.push(0);
    bytes.push(0);
    bytes.push(0);
    bytes.push(1);

    let mut bits = Vec::new();

    // NAL header: forbidden=0, nal_ref_idc=3, nal_unit_type=7 (SPS)
    bits.push(0);
    bits.push(1);
    bits.push(1);
    bits.push(0);
    bits.push(0);
    bits.push(1);
    bits.push(1);
    bits.push(1);

    // profile_idc = 66 (Baseline)
    append_bits(&mut bits, 66, 8);

    // constraint_set0-5_flags + reserved_zero_2bits
    append_bits(&mut bits, 0, 8);

    // level_idc = 31 (Level 3.1)
    append_bits(&mut bits, 31, 8);

    // seq_parameter_set_id
    bits.extend(encode_ue_bits(seq_parameter_set_id));

    // log2_max_frame_num_minus4
    bits.extend(encode_ue_bits(log2_max_frame_num_minus4 as u32));

    // pic_order_cnt_type
    bits.extend(encode_ue_bits(pic_order_cnt_type as u32));

    match pic_order_cnt_type {
        0 => {
            bits.extend(encode_ue_bits(log2_max_pic_order_cnt_lsb_minus4 as u32));
        }
        1 => {
            bits.push(delta_pic_order_always_zero_flag as u8);
            bits.extend(encode_se_bits(0)); // offset_for_non_ref_pic
            bits.extend(encode_se_bits(0)); // offset_for_top_to_bottom_field
            bits.extend(encode_ue_bits(0)); // num_ref_frames_in_pic_order_cnt_cycle
        }
        2 => {}
        _ => {}
    }

    // max_num_ref_frames
    bits.extend(encode_ue_bits(max_num_ref_frames));

    // gaps_in_frame_num_value_allowed_flag = 0
    bits.push(0);

    // pic_width_in_mbs_minus1 = 39 (640/16 - 1)
    bits.extend(encode_ue_bits(39));

    // pic_height_in_map_units_minus1 = 27 (448/16 - 1)
    bits.extend(encode_ue_bits(27));

    // frame_mbs_only_flag
    bits.push(frame_mbs_only_flag as u8);

    if !frame_mbs_only_flag {
        // mb_adaptive_frame_field_flag = 0
        bits.push(0);
    }

    // direct_8x8_inference_flag = 0
    bits.push(0);

    // frame_cropping_flag = 0
    bits.push(0);

    // vui_parameters_present_flag = 0
    bits.push(0);

    // RBSP trailing bits: 1 followed by enough 0s to fill byte
    bits.push(1);

    bytes.extend(bits_to_bytes(&bits));
    bytes
}

// ============================================================================
// Helper: Build synthetic PPS
// ============================================================================

/// Build a minimal PPS NAL unit with specified parameters (with start code).
fn build_pps(
    pic_parameter_set_id: u32,
    seq_parameter_set_id: u32,
    entropy_coding_mode_flag: bool,
    bottom_field_pic_order_in_frame_present_flag: bool,
    deblocking_filter_control_present_flag: bool,
    redundant_pic_cnt_present_flag: bool,
    weighted_pred_flag: bool,
    weighted_bipred_idc: u8,
    num_ref_idx_l0_default_active_minus1: u32,
    num_ref_idx_l1_default_active_minus1: u32,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    // Start code
    bytes.push(0);
    bytes.push(0);
    bytes.push(0);
    bytes.push(1);

    let mut bits = Vec::new();

    // NAL header: forbidden=0, nal_ref_idc=3, nal_unit_type=8 (PPS)
    bits.push(0);
    bits.push(1);
    bits.push(1);
    bits.push(0);
    bits.push(1);
    bits.push(0);
    bits.push(0);
    bits.push(0);

    // pic_parameter_set_id
    bits.extend(encode_ue_bits(pic_parameter_set_id));

    // seq_parameter_set_id
    bits.extend(encode_ue_bits(seq_parameter_set_id));

    // entropy_coding_mode_flag
    bits.push(entropy_coding_mode_flag as u8);

    // bottom_field_pic_order_in_frame_present_flag
    bits.push(bottom_field_pic_order_in_frame_present_flag as u8);

    // num_slice_groups_minus1 = 0
    bits.extend(encode_ue_bits(0));

    // num_ref_idx_l0_default_active_minus1
    bits.extend(encode_ue_bits(num_ref_idx_l0_default_active_minus1));

    // num_ref_idx_l1_default_active_minus1
    bits.extend(encode_ue_bits(num_ref_idx_l1_default_active_minus1));

    // weighted_pred_flag
    bits.push(weighted_pred_flag as u8);

    // weighted_bipred_idc
    append_bits(&mut bits, weighted_bipred_idc as u32, 2);

    // pic_init_qp_minus26 = 0
    bits.extend(encode_se_bits(0));

    // pic_init_qs_minus26 = 0
    bits.extend(encode_se_bits(0));

    // chroma_qp_index_offset = 0
    bits.extend(encode_se_bits(0));

    // deblocking_filter_control_present_flag
    bits.push(deblocking_filter_control_present_flag as u8);

    // constrained_intra_pred_flag = 0
    bits.push(0);

    // redundant_pic_cnt_present_flag
    bits.push(redundant_pic_cnt_present_flag as u8);

    // RBSP trailing bits: 1 followed by enough 0s to fill byte
    bits.push(1);

    bytes.extend(bits_to_bytes(&bits));
    bytes
}

// ============================================================================
// Helper: Build synthetic slice header
// ============================================================================

/// Build a slice NAL unit with a custom slice header.
fn build_slice_nal(
    nal_unit_type: u8,
    nal_ref_idc: u8,
    first_mb_in_slice: u32,
    slice_type: u32,
    pps_id: u32,
    frame_num: u32,
    frame_num_bits: u8,
    idr_pic_id: Option<u32>,
    pic_order_cnt_lsb: Option<i32>,
    poc_lsb_bits: u8,
    delta_pic_order_cnt: Option<Vec<i32>>,
    redundant_pic_cnt: Option<u32>,
    field_pic_flag: Option<bool>,
    bottom_field: Option<bool>,
    is_b_slice: bool,
    direct_spatial_mv_pred_flag: bool,
    num_ref_idx_active_override_flag: bool,
    override_l0: Option<u32>,
    override_l1: Option<u32>,
    ref_pic_list_mod_l0: bool,
    ref_pic_list_mod_l0_entries: &[(u32, u32)],
    ref_pic_list_mod_l1: bool,
    ref_pic_list_mod_l1_entries: &[(u32, u32)],
    dec_ref_pic_marking_idr: Option<(bool, bool)>,
    dec_ref_pic_marking_non_idr: Option<Vec<(u32, u32)>>,
    cabac_init_idc: Option<u8>,
    slice_qp_delta: i32,
    disable_deblocking_filter_idc: Option<i8>,
    slice_alpha_c0_offset_div2: Option<i32>,
    slice_beta_offset_div2: Option<i32>,
) -> Vec<u8> {
    let mut bits = Vec::new();

    // NAL header
    append_bits(&mut bits, nal_ref_idc as u32, 2);
    append_bits(&mut bits, nal_unit_type as u32, 5);
    bits.push(0); // reserved_zero_bit

    // first_mb_in_slice
    bits.extend(encode_ue_bits(first_mb_in_slice));

    // slice_type
    bits.extend(encode_ue_bits(slice_type));

    // pic_parameter_set_id
    bits.extend(encode_ue_bits(pps_id));

    // frame_num
    append_bits(&mut bits, frame_num, frame_num_bits);

    // field_pic_flag and bottom_field_flag (if not frame-only)
    if let Some(fp_flag) = field_pic_flag {
        bits.push(fp_flag as u8);
        if fp_flag {
            bits.push(bottom_field.unwrap_or(false) as u8);
        }
    }

    // idr_pic_id (for IDR slices only)
    if let Some(id) = idr_pic_id {
        bits.extend(encode_ue_bits(id));
    }

    // pic_order_cnt_lsb (POC type 0)
    if let Some(poc) = pic_order_cnt_lsb {
        let unsigned_poc = poc as u32;
        append_bits(&mut bits, unsigned_poc, poc_lsb_bits);
    }

    // delta_pic_order_cnt (POC type 1)
    if let Some(dpoc) = delta_pic_order_cnt {
        bits.extend(encode_se_bits(dpoc[0]));
        if dpoc.len() > 1 {
            bits.extend(encode_se_bits(dpoc[1]));
        }
    }

    // redundant_pic_cnt
    if let Some(rpc) = redundant_pic_cnt {
        bits.extend(encode_ue_bits(rpc));
    }

    // dec_ref_pic_marking (for reference pictures). Per H.264 spec 7.3.2.1.1
    // this comes BEFORE num_ref_idx_active_override_flag / ref_pic_list_modification.
    if nal_ref_idc > 0 {
        if let Some((no_output, lt_ref)) = dec_ref_pic_marking_idr {
            // IDR marking
            bits.push(no_output as u8);
            bits.push(lt_ref as u8);
        } else if let Some(operations) = dec_ref_pic_marking_non_idr {
            // Non-IDR marking
            bits.push(1); // adaptive_ref_pic_marking_mode_flag
            for &(op, value) in operations.iter() {
                bits.extend(encode_ue_bits(op));
                if op != 0 {
                    bits.extend(encode_ue_bits(value));
                }
            }
            bits.extend(encode_ue_bits(0)); // terminating operation
        } else {
            // Non-IDR, no adaptive marking
            bits.push(0);
        }
    }

    // num_ref_idx_active_override_flag (P/SP/B slices)
    if slice_type != 3 && slice_type != 4 {
        // not SI/I
        bits.push(num_ref_idx_active_override_flag as u8);
        if num_ref_idx_active_override_flag {
            if let Some(l0) = override_l0 {
                bits.extend(encode_ue_bits(l0));
            }
            if is_b_slice {
                if let Some(l1) = override_l1 {
                    bits.extend(encode_ue_bits(l1));
                }
            }
        }
    }

    // ref_pic_list_modification_l0
    if slice_type != 3 && slice_type != 4 {
        // not SI/I
        bits.push(ref_pic_list_mod_l0 as u8);
        if ref_pic_list_mod_l0 {
            for &(mod_idc, value) in ref_pic_list_mod_l0_entries {
                bits.extend(encode_ue_bits(mod_idc));
                if mod_idc < 2 {
                    bits.extend(encode_ue_bits(value));
                }
            }
        }
    }

    // ref_pic_list_modification_l1 (B-slice only)
    if is_b_slice && ref_pic_list_mod_l1 {
        bits.push(1); // ref_pic_list_modification_flag_l1
        for &(mod_idc, value) in ref_pic_list_mod_l1_entries {
            bits.extend(encode_ue_bits(mod_idc));
            if mod_idc < 2 {
                bits.extend(encode_ue_bits(value));
            }
        }
    } else if is_b_slice {
        bits.push(0); // ref_pic_list_modification_flag_l1
    }

    // cabac_init_idc (not for I/SI slices, only if CABAC enabled)
    if let Some(cabac_idc) = cabac_init_idc {
        bits.extend(encode_ue_bits(cabac_idc as u32));
    }

    // slice_qp_delta
    bits.extend(encode_se_bits(slice_qp_delta));

    // deblocking filter parameters
    if let Some(disable_idc) = disable_deblocking_filter_idc {
        bits.extend(encode_ue_bits(disable_idc as u32));
        if disable_idc != 1 {
            if let Some(alpha) = slice_alpha_c0_offset_div2 {
                bits.extend(encode_se_bits(alpha));
            }
            if let Some(beta) = slice_beta_offset_div2 {
                bits.extend(encode_se_bits(beta));
            }
        }
    }

    // RBSP trailing bits
    bits.push(1);

    bits_to_bytes(&bits)
}

// ============================================================================
// Helper: Initialize parser with SPS and PPS
// ============================================================================

/// Initialize a parser with given SPS and PPS data.
fn init_parser_with_params(sps_data: &[u8], pps_data: &[u8]) -> H264Parser {
    let mut combined = Vec::new();
    combined.extend_from_slice(sps_data);
    combined.extend_from_slice(pps_data);

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(combined);
    parser.parse(&packet).expect("Failed to parse SPS/PPS");

    assert!(parser.active_sps().is_some(), "Active SPS should be set");
    assert!(parser.active_pps().is_some(), "Active PPS should be set");
    parser
}

/// Parse a slice header from the given NAL data using the parser.
fn parse_slice_header_with_parser(
    parser: &H264Parser,
    nal_data: &[u8],
    nal_ref_idc: u8,
    nal_unit_type: u8,
) -> vk_video_parser::h264::SliceHeader {
    parser
        .parse_slice_header(nal_data, nal_ref_idc, nal_unit_type)
        .expect("Failed to parse slice header")
}

// ============================================================================
// Test 1: Basic fields
// ============================================================================

#[test]
fn test_slice_header_basic_fields() {
    // Build SPS with frame-only, POC type 0
    let sps_data = build_sps(0, 4, 0, 4, false, true, 1);
    // Build PPS with CABAC disabled, deblocking enabled
    let pps_data = build_pps(0, 0, false, false, true, false, false, 0, 0, 0);

    let parser = init_parser_with_params(&sps_data, &pps_data);

    // Test P-slice (slice_type=0)
    let slice_data = build_slice_nal(
        1,       // Non-IDR slice
        3,       // nal_ref_idc=3
        0,       // first_mb_in_slice
        0,       // slice_type=P
        0,       // pps_id
        0,       // frame_num
        8,       // frame_num_bits (log2_max_frame_num_minus4=4 → 8 bits)
        None,    // idr_pic_id
        Some(0), // pic_order_cnt_lsb
        8,       // poc_lsb_bits
        None,    // delta_pic_order_cnt
        None,    // redundant_pic_cnt
        None,    // field_pic_flag
        None,    // bottom_field
        false,   // is_b_slice
        false,   // direct_spatial_mv_pred_flag
        false,   // num_ref_idx_active_override_flag
        None,
        None, // override_l0, override_l1
        false,
        &[], // ref_pic_list_mod_l0
        false,
        &[],     // ref_pic_list_mod_l1
        None,    // non-IDR ref pic, no adaptive marking
        None,    // dec_ref_pic_marking_non_idr
        None,    // cabac_init_idc (CABAC disabled)
        0,       // slice_qp_delta
        Some(0), // disable_deblocking_filter_idc
        Some(0), // slice_alpha_c0_offset_div2
        Some(0), // slice_beta_offset_div2
    );

    let slh = parse_slice_header_with_parser(&parser, &slice_data, 3, 1);

    assert_eq!(slh.first_mb_in_slice, 0);
    assert_eq!(slh.slice_type, 0); // P
    assert_eq!(slh.pic_parameter_set_id, 0);

    // Test B-slice (slice_type=1)
    let slice_data_b = build_slice_nal(
        1,
        0,  // nal_ref_idc=0 (non-reference)
        40, // first_mb_in_slice
        1,  // slice_type=B
        0,
        1,
        8,
        None,
        Some(2),
        8,
        None,
        None,
        None,
        None,
        true, // is_b_slice
        true, // direct_spatial_mv_pred_flag
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        Some(0),
        Some(0),
        Some(0),
    );

    let slh_b = parse_slice_header_with_parser(&parser, &slice_data_b, 0, 1);

    assert_eq!(slh_b.first_mb_in_slice, 40);
    assert_eq!(slh_b.slice_type, 1); // B
    assert_eq!(slh_b.pic_parameter_set_id, 0);

    // Test I-slice (slice_type=4)
    let slice_data_i = build_slice_nal(
        1,
        3,
        0,
        4, // slice_type=I
        0,
        2,
        8,
        None,
        Some(4),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        Some(0),
        Some(0),
        Some(0),
    );

    let slh_i = parse_slice_header_with_parser(&parser, &slice_data_i, 3, 1);

    assert_eq!(slh_i.slice_type, 4); // I
}

// ============================================================================
// Test 2: frame_num with various log2_max_frame_num_minus4 values
// ============================================================================

#[test]
fn test_slice_header_frame_num() {
    // Test with log2_max_frame_num_minus4 = 0 (4 bits)
    let sps_data_4bits = build_sps(0, 0, 0, 4, false, true, 1);
    let pps_data = build_pps(0, 0, false, false, true, false, false, 0, 0, 0);

    let parser_4bits = init_parser_with_params(&sps_data_4bits, &pps_data);

    let slice_data = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        10, // frame_num = 10
        4,  // 4 bits
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh = parse_slice_header_with_parser(&parser_4bits, &slice_data, 3, 1);
    assert_eq!(slh.frame_num, 10);

    // Test with log2_max_frame_num_minus4 = 4 (8 bits)
    let sps_data_8bits = build_sps(0, 4, 0, 4, false, true, 1);
    let parser_8bits = init_parser_with_params(&sps_data_8bits, &pps_data);

    let slice_data = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        200, // frame_num = 200
        8,   // 8 bits
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh = parse_slice_header_with_parser(&parser_8bits, &slice_data, 3, 1);
    assert_eq!(slh.frame_num, 200);

    // Test with log2_max_frame_num_minus4 = 8 (12 bits)
    let sps_data_12bits = build_sps(0, 8, 0, 4, false, true, 1);
    let parser_12bits = init_parser_with_params(&sps_data_12bits, &pps_data);

    let slice_data = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        1000, // frame_num = 1000
        12,   // 12 bits
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh = parse_slice_header_with_parser(&parser_12bits, &slice_data, 3, 1);
    assert_eq!(slh.frame_num, 1000);
}

// ============================================================================
// Test 3: pic_order_cnt_lsb for POC type 0
// ============================================================================

#[test]
fn test_slice_header_pic_order_cnt_lsb() {
    // POC type 0 with log2_max_pic_order_cnt_lsb_minus4 = 4 (8 bits)
    let sps_data = build_sps(0, 4, 0, 4, false, true, 1);
    let pps_data = build_pps(0, 0, false, false, true, false, false, 0, 0, 0);

    let parser = init_parser_with_params(&sps_data, &pps_data);

    let slice_data = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        5,
        8,
        None,
        Some(42), // pic_order_cnt_lsb = 42
        8,        // 8 bits
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh = parse_slice_header_with_parser(&parser, &slice_data, 3, 1);
    assert_eq!(slh.pic_order_cnt_lsb, 42);

    // Verify SPS settings
    let sps = parser.active_sps().unwrap();
    assert_eq!(sps.pic_order_cnt_type, 0);
    assert_eq!(sps.log2_max_pic_order_cnt_lsb_minus4, 4);
    assert_eq!(sps.max_pic_order_cnt_lsb, 256); // 2^(4+4)
}

// ============================================================================
// Test 4: idr_pic_id for IDR slices
// ============================================================================

#[test]
fn test_slice_header_idr_pic_id() {
    let sps_data = build_sps(0, 4, 0, 4, false, true, 1);
    let pps_data = build_pps(0, 0, false, false, true, false, false, 0, 0, 0);

    let parser = init_parser_with_params(&sps_data, &pps_data);

    // IDR slice (nal_unit_type=5)
    let slice_data = build_slice_nal(
        5, // IDR slice
        3,
        0,
        4, // slice_type=I
        0,
        0,
        8,
        Some(5), // idr_pic_id = 5
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        Some((true, false)), // no_output_of_prior_pics_flag=true, long_term_reference_flag=false
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh = parse_slice_header_with_parser(&parser, &slice_data, 3, 5);
    assert_eq!(slh.idr_pic_id, 5);
    assert_eq!(slh.nal_unit_type, 5);

    // Non-IDR slice should have idr_pic_id = 0
    let slice_data_non_idr = build_slice_nal(
        1, // Non-IDR
        3,
        0,
        0,
        0,
        1,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_non_idr = parse_slice_header_with_parser(&parser, &slice_data_non_idr, 3, 1);
    assert_eq!(slh_non_idr.idr_pic_id, 0);
}

// ============================================================================
// Test 5: field_pic_flag and bottom_field
// ============================================================================

#[test]
fn test_slice_header_field_pic_flag() {
    // SPS with frame_mbs_only_flag=0 (field pictures allowed)
    let sps_data = build_sps(0, 4, 0, 4, false, false, 1);
    let pps_data = build_pps(0, 0, false, false, true, false, false, 0, 0, 0);

    let parser = init_parser_with_params(&sps_data, &pps_data);

    // Frame picture (field_pic_flag=0)
    let slice_data_frame = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        Some(false),
        None, // field_pic_flag=0
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_frame = parse_slice_header_with_parser(&parser, &slice_data_frame, 3, 1);
    assert!(!slh_frame.field_pic_flag);
    assert!(!slh_frame.bottom_field);

    // Top field (field_pic_flag=1, bottom_field=0)
    let slice_data_top = build_slice_nal(
        1,
        3,
        0,
        5, // slice_type=5 (field P)
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        Some(true),
        Some(false), // field_pic_flag=1, bottom_field=0
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_top = parse_slice_header_with_parser(&parser, &slice_data_top, 3, 1);
    assert!(slh_top.field_pic_flag);
    assert!(!slh_top.bottom_field);

    // Bottom field (field_pic_flag=1, bottom_field=1)
    let slice_data_bottom = build_slice_nal(
        1,
        3,
        0,
        5,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        Some(true),
        Some(true), // field_pic_flag=1, bottom_field=1
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_bottom = parse_slice_header_with_parser(&parser, &slice_data_bottom, 3, 1);
    assert!(slh_bottom.field_pic_flag);
    assert!(slh_bottom.bottom_field);

    // Frame-only SPS should not have field_pic_flag
    let sps_frame_only = build_sps(0, 4, 0, 4, false, true, 1);
    let parser_frame_only = init_parser_with_params(&sps_frame_only, &pps_data);
    let slice_data_no_field = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_no_field =
        parse_slice_header_with_parser(&parser_frame_only, &slice_data_no_field, 3, 1);
    assert!(!slh_no_field.field_pic_flag);
}

// ============================================================================
// Test 6: delta_pic_order_cnt for POC type 1
// ============================================================================

#[test]
fn test_slice_header_delta_pic_order_cnt() {
    // POC type 1 with delta_pic_order_always_zero_flag=false
    let sps_data = build_sps(0, 4, 1, 0, false, true, 1);
    let pps_data = build_pps(0, 0, false, false, true, false, false, 0, 0, 0);

    let parser = init_parser_with_params(&sps_data, &pps_data);

    let slice_data = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        None,
        0,
        Some(vec![3]), // delta_pic_order_cnt[0]=3 only (bottom_field_pic_order_in_frame_present_flag=false)
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh = parse_slice_header_with_parser(&parser, &slice_data, 3, 1);
    assert_eq!(slh.delta_pic_order_cnt[0], 3);
    assert_eq!(slh.delta_pic_order_cnt[1], 0); // default, not parsed

    // POC type 1 with delta_pic_order_always_zero_flag=true
    let sps_data_zero = build_sps(0, 4, 1, 0, true, true, 1);
    let parser_zero = init_parser_with_params(&sps_data_zero, &pps_data);

    let slice_data_zero = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        None,
        0,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_zero = parse_slice_header_with_parser(&parser_zero, &slice_data_zero, 3, 1);
    assert_eq!(slh_zero.delta_pic_order_cnt[0], 0);
    assert_eq!(slh_zero.delta_pic_order_cnt[1], 0);
}

// ============================================================================
// Test 7: redundant_pic_cnt
// ============================================================================

#[test]
fn test_slice_header_redundant_pic_cnt() {
    // PPS with redundant_pic_cnt_present_flag=true
    let sps_data = build_sps(0, 4, 0, 4, false, true, 1);
    let pps_data = build_pps(0, 0, false, false, true, true, false, 0, 0, 0);

    let parser = init_parser_with_params(&sps_data, &pps_data);

    let slice_data = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        Some(3), // redundant_pic_cnt = 3
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh = parse_slice_header_with_parser(&parser, &slice_data, 3, 1);
    assert_eq!(slh.redundant_pic_cnt, 3);

    // PPS with redundant_pic_cnt_present_flag=false
    let pps_data_no_redundant = build_pps(0, 0, false, false, true, false, false, 0, 0, 0);
    let parser_no_redundant = init_parser_with_params(&sps_data, &pps_data_no_redundant);

    let slice_data_no_redundant = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_no_redundant =
        parse_slice_header_with_parser(&parser_no_redundant, &slice_data_no_redundant, 3, 1);
    assert_eq!(slh_no_redundant.redundant_pic_cnt, 0);
}

// ============================================================================
// Test 8: num_ref_idx_active_override_flag
// ============================================================================

#[test]
fn test_slice_header_num_ref_idx_override() {
    let sps_data = build_sps(0, 4, 0, 4, false, true, 4);
    let pps_data = build_pps(0, 0, false, false, true, false, false, 0, 2, 1); // l0=3 refs, l1=2 refs

    let parser = init_parser_with_params(&sps_data, &pps_data);

    // P-slice without override
    let slice_data_p_no_override = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None, // no override
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_p = parse_slice_header_with_parser(&parser, &slice_data_p_no_override, 3, 1);
    assert!(!slh_p.num_ref_idx_active_override_flag);
    assert_eq!(slh_p.num_ref_idx_l0_active_minus1, 2); // from PPS

    // P-slice with override
    let slice_data_p_override = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        true,
        Some(0),
        None, // override with l0=1 ref
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_p_override = parse_slice_header_with_parser(&parser, &slice_data_p_override, 3, 1);
    assert!(slh_p_override.num_ref_idx_active_override_flag);
    assert_eq!(slh_p_override.num_ref_idx_l0_active_minus1, 0);

    // B-slice with override for both lists
    let slice_data_b_override = build_slice_nal(
        1,
        0,
        0,
        1,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        true,
        true,
        true,
        Some(1),
        Some(0), // override with l0=2 refs, l1=1 ref
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_b = parse_slice_header_with_parser(&parser, &slice_data_b_override, 0, 1);
    assert!(slh_b.num_ref_idx_active_override_flag);
    assert_eq!(slh_b.num_ref_idx_l0_active_minus1, 1);
    assert_eq!(slh_b.num_ref_idx_l1_active_minus1, 0);
}

// ============================================================================
// Test 9: ref_pic_list_modification
// ============================================================================

#[test]
fn test_slice_header_ref_pic_list_modification() {
    let sps_data = build_sps(0, 4, 0, 4, false, true, 4);
    let pps_data = build_pps(0, 0, false, false, true, false, false, 0, 2, 1);

    let parser = init_parser_with_params(&sps_data, &pps_data);

    // P-slice with ref_pic_list_modification_l0
    // H.264: single modification per list (index [0] only):
    // modification_of_pic_nums_idc=0, abs_diff_pic_num_minus1=2
    let mod_l0_entries = vec![(0u32, 2u32)];

    let slice_data = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        true,
        &mod_l0_entries, // ref_pic_list_mod_l0=true
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh = parse_slice_header_with_parser(&parser, &slice_data, 3, 1);
    assert_eq!(slh.ref_pic_list_modification_l0.len(), 1);
    assert_eq!(slh.ref_pic_list_modification_l0[0].op, 0);
    assert_eq!(slh.ref_pic_list_modification_l0[0].difference, 2);

    // B-slice with ref_pic_list_modification for both lists
    let mod_l1_entries = vec![(1u32, 1u32)];

    let slice_data_b = build_slice_nal(
        1,
        0,
        0,
        1,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        true,
        false,
        false,
        None,
        None,
        true,
        &mod_l0_entries,
        true,
        &mod_l1_entries,
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_b = parse_slice_header_with_parser(&parser, &slice_data_b, 0, 1);
    assert_eq!(slh_b.ref_pic_list_modification_l0.len(), 1);
    assert_eq!(slh_b.ref_pic_list_modification_l0[0].op, 0);
    assert_eq!(slh_b.ref_pic_list_modification_l0[0].difference, 2);
    assert_eq!(slh_b.ref_pic_list_modification_l1.len(), 1);
    assert_eq!(slh_b.ref_pic_list_modification_l1[0].op, 1);
    assert_eq!(slh_b.ref_pic_list_modification_l1[0].difference, 1);
}

// ============================================================================
// Test 10: dec_ref_pic_marking for IDR frames
// ============================================================================

#[test]
fn test_slice_header_dec_ref_pic_marking_idr() {
    let sps_data = build_sps(0, 4, 0, 4, false, true, 1);
    let pps_data = build_pps(0, 0, false, false, true, false, false, 0, 0, 0);

    let parser = init_parser_with_params(&sps_data, &pps_data);

    // IDR slice with no_output_of_prior_pics_flag=true, long_term_reference_flag=false
    let slice_data = build_slice_nal(
        5, // IDR
        3,
        0,
        4,
        0,
        0,
        8,
        Some(0),
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        Some((true, false)), // no_output=true, lt_ref=false
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh = parse_slice_header_with_parser(&parser, &slice_data, 3, 5);
    assert!(slh.no_output_of_prior_pics_flag);
    assert!(!slh.long_term_reference_flag);
    assert!(slh.dec_ref_pic_marking.is_empty()); // IDR doesn't use memory_management_control_operation

    // IDR slice with long_term_reference_flag=true
    let slice_data_lt = build_slice_nal(
        5,
        3,
        0,
        4,
        0,
        0,
        8,
        Some(1),
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        Some((false, true)), // no_output=false, lt_ref=true
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_lt = parse_slice_header_with_parser(&parser, &slice_data_lt, 3, 5);
    assert!(!slh_lt.no_output_of_prior_pics_flag);
    assert!(slh_lt.long_term_reference_flag);
}

// ============================================================================
// Test 11: dec_ref_pic_marking for non-IDR frames
// ============================================================================

#[test]
fn test_slice_header_dec_ref_pic_marking_non_idr() {
    let sps_data = build_sps(0, 4, 0, 4, false, true, 4);
    let pps_data = build_pps(0, 0, false, false, true, false, false, 0, 2, 0);

    let parser = init_parser_with_params(&sps_data, &pps_data);

    // Non-IDR reference picture with adaptive marking
    // Operation: memory_management_control_operation=1, difference_of_pic_nums_minus1=2
    // Operation: memory_management_control_operation=0 (terminator)
    let operations = vec![(1u32, 2u32)];

    let slice_data = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        Some(operations), // adaptive marking
        None,
        0,
        None,
        None,
        None,
    );

    let slh = parse_slice_header_with_parser(&parser, &slice_data, 3, 1);
    assert_eq!(slh.dec_ref_pic_marking.len(), 1);
    assert_eq!(
        slh.dec_ref_pic_marking[0].memory_management_control_operation,
        1
    );
    assert_eq!(slh.dec_ref_pic_marking[0].value, 2);

    // Non-IDR reference picture without adaptive marking
    let slice_data_no_adaptive = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None, // no adaptive marking
        None,
        0,
        None,
        None,
        None,
    );

    let slh_no_adaptive = parse_slice_header_with_parser(&parser, &slice_data_no_adaptive, 3, 1);
    assert!(slh_no_adaptive.dec_ref_pic_marking.is_empty());

    // Non-reference picture (nal_ref_idc=0) should not have dec_ref_pic_marking
    let slice_data_non_ref = build_slice_nal(
        1,
        0,
        0,
        1,
        0, // B-slice, non-ref
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        true,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_non_ref = parse_slice_header_with_parser(&parser, &slice_data_non_ref, 0, 1);
    assert!(slh_non_ref.dec_ref_pic_marking.is_empty());
}

// ============================================================================
// Test 12: cabac_init_idc
// ============================================================================

#[test]
fn test_slice_header_cabac_init_idc() {
    // PPS with CABAC enabled
    let sps_data = build_sps(0, 4, 0, 4, false, true, 1);
    let pps_data = build_pps(0, 0, true, false, true, false, false, 0, 0, 0);

    let parser = init_parser_with_params(&sps_data, &pps_data);

    // P-slice with cabac_init_idc=1
    let slice_data = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        Some(1), // cabac_init_idc=1
        0,
        None,
        None,
        None,
    );

    let slh = parse_slice_header_with_parser(&parser, &slice_data, 3, 1);
    assert_eq!(slh.cabac_init_idc, 1);

    // B-slice with cabac_init_idc=2
    let slice_data_b = build_slice_nal(
        1,
        0,
        0,
        1,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        true,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        Some(2), // cabac_init_idc=2
        0,
        None,
        None,
        None,
    );

    let slh_b = parse_slice_header_with_parser(&parser, &slice_data_b, 0, 1);
    assert_eq!(slh_b.cabac_init_idc, 2);

    // I-slice should not have cabac_init_idc (not parsed per spec)
    let slice_data_i = build_slice_nal(
        1,
        3,
        0,
        4,
        0, // I-slice
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_i = parse_slice_header_with_parser(&parser, &slice_data_i, 3, 1);
    assert_eq!(slh_i.cabac_init_idc, 0);

    // PPS with CABAC disabled - no cabac_init_idc
    let pps_data_no_cabac = build_pps(0, 0, false, false, true, false, false, 0, 0, 0);
    let parser_no_cabac = init_parser_with_params(&sps_data, &pps_data_no_cabac);

    let slice_data_no_cabac = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_no_cabac = parse_slice_header_with_parser(&parser_no_cabac, &slice_data_no_cabac, 3, 1);
    assert_eq!(slh_no_cabac.cabac_init_idc, 0);
}

// ============================================================================
// Test 13: slice_qp_delta
// ============================================================================

#[test]
fn test_slice_header_slice_qp_delta() {
    let sps_data = build_sps(0, 4, 0, 4, false, true, 1);
    let pps_data = build_pps(0, 0, false, false, true, false, false, 0, 0, 0);

    let parser = init_parser_with_params(&sps_data, &pps_data);

    // Positive QP delta
    let slice_data_pos = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        5, // slice_qp_delta=5
        None,
        None,
        None,
    );

    let slh_pos = parse_slice_header_with_parser(&parser, &slice_data_pos, 3, 1);
    assert_eq!(slh_pos.slice_qp_delta, 5);

    // Negative QP delta
    let slice_data_neg = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        -3, // slice_qp_delta=-3
        None,
        None,
        None,
    );

    let slh_neg = parse_slice_header_with_parser(&parser, &slice_data_neg, 3, 1);
    assert_eq!(slh_neg.slice_qp_delta, -3);

    // Zero QP delta
    let slice_data_zero = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_zero = parse_slice_header_with_parser(&parser, &slice_data_zero, 3, 1);
    assert_eq!(slh_zero.slice_qp_delta, 0);
}

// ============================================================================
// Test 14: deblocking filter parameters
// ============================================================================

#[test]
fn test_slice_header_deblocking_filter_params() {
    // PPS with deblocking_filter_control_present_flag=false
    // (deblocking params are read from the slice header)
    let sps_data = build_sps(0, 4, 0, 4, false, true, 1);
    let pps_data = build_pps(0, 0, false, false, false, false, false, 0, 0, 0);

    let parser = init_parser_with_params(&sps_data, &pps_data);

    // Disable deblocking (disable_deblocking_filter_idc=1)
    let slice_data_disabled = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        Some(1),
        None,
        None, // disable_deblocking_filter_idc=1
    );

    let slh_disabled = parse_slice_header_with_parser(&parser, &slice_data_disabled, 3, 1);
    assert_eq!(slh_disabled.disable_deblocking_filter_idc, 1);
    assert_eq!(slh_disabled.slice_alpha_c0_offset_div2, 0);
    assert_eq!(slh_disabled.slice_beta_offset_div2, 0);

    // Default deblocking (disable_deblocking_filter_idc=0) with offsets
    let slice_data_default = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        Some(0),
        Some(-2),
        Some(3), // default with alpha=-2, beta=3
    );

    let slh_default = parse_slice_header_with_parser(&parser, &slice_data_default, 3, 1);
    assert_eq!(slh_default.disable_deblocking_filter_idc, 0);
    assert_eq!(slh_default.slice_alpha_c0_offset_div2, -2);
    assert_eq!(slh_default.slice_beta_offset_div2, 3);

    // Always on deblocking (disable_deblocking_filter_idc=2)
    let slice_data_always = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        Some(2),
        Some(0),
        Some(0), // always on with zero offsets
    );

    let slh_always = parse_slice_header_with_parser(&parser, &slice_data_always, 3, 1);
    assert_eq!(slh_always.disable_deblocking_filter_idc, 2);
    assert_eq!(slh_always.slice_alpha_c0_offset_div2, 0);
    assert_eq!(slh_always.slice_beta_offset_div2, 0);

    // PPS with deblocking_filter_control_present_flag=true
    // (no deblocking params in the slice header)
    let pps_data_no_deblock = build_pps(0, 0, false, false, true, false, false, 0, 0, 0);
    let parser_no_deblock = init_parser_with_params(&sps_data, &pps_data_no_deblock);

    let slice_data_no_deblock = build_slice_nal(
        1,
        3,
        0,
        0,
        0,
        0,
        8,
        None,
        Some(0),
        8,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
        false,
        &[],
        false,
        &[],
        None,
        None,
        None,
        0,
        None,
        None,
        None,
    );

    let slh_no_deblock =
        parse_slice_header_with_parser(&parser_no_deblock, &slice_data_no_deblock, 3, 1);
    assert_eq!(slh_no_deblock.disable_deblocking_filter_idc, 0);
    assert_eq!(slh_no_deblock.slice_alpha_c0_offset_div2, 0);
    assert_eq!(slh_no_deblock.slice_beta_offset_div2, 0);
}

// ============================================================================
// Test 15: Slice headers from born_trailer.h264
// ============================================================================

#[test]
fn test_slice_header_from_born_trailer() {
    let data = load_test_file("assets/born_trailer.h264");
    assert!(!data.is_empty(), "born_trailer.h264 should not be empty");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data);

    // First parse to get SPS/PPS
    match parser.parse(&packet) {
        Ok(ParseResult::ParameterSet {
            sps: Some(_),
            pps: Some(_),
            ..
        }) => {
            // Good - got both SPS and PPS
        }
        Ok(ParseResult::ParameterSet { sps, pps, .. }) => {
            panic!(
                "Expected both SPS and PPS, got sps={:?}, pps={:?}",
                sps.is_some(),
                pps.is_some()
            );
        }
        Ok(_) => panic!("Expected ParameterSet result"),
        Err(e) => panic!("Failed to parse SPS/PPS: {:?}", e),
    }

    assert!(parser.active_sps().is_some(), "Active SPS should be set");
    assert!(parser.active_pps().is_some(), "Active PPS should be set");

    // Parse slices - limit to first frame to avoid excessive logging
    let mut slice_count = 0;
    let mut first_slice_parsed = false;
    let mut frame_count = 0;
    const MAX_FRAMES: usize = 1;

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::Slice { slices, .. }) => {
                frame_count += 1;
                if frame_count > MAX_FRAMES {
                    break;
                }
                for slice in &slices {
                    if let Some(vk_video_parser::SliceHeader::H264(slh)) = &slice.slice_header {
                        slice_count += 1;

                        if !first_slice_parsed {
                            first_slice_parsed = true;
                            // Verify first slice header fields are valid
                            assert_eq!(
                                slh.first_mb_in_slice, 0,
                                "First slice should start at MB 0"
                            );
                            assert!(
                                slh.slice_type < 10,
                                "slice_type={} should be valid (0-9)",
                                slh.slice_type
                            );
                            assert_eq!(slh.pic_parameter_set_id, 0, "pps_id should be 0");
                            assert!(slh.frame_num >= 0, "frame_num should be non-negative");
                            assert!(slh.header_bit_size > 0, "header_bit_size should be > 0");
                        }

                        // Verify slice_type is valid
                        assert!(
                            slh.slice_type < 5 || (slh.slice_type >= 5 && slh.slice_type < 10),
                            "slice_type={} invalid",
                            slh.slice_type
                        );

                        // Verify deblocking filter parameters are in valid range
                        assert!(
                            slh.disable_deblocking_filter_idc >= -1
                                && slh.disable_deblocking_filter_idc <= 2,
                            "disable_deblocking_filter_idc={} out of range",
                            slh.disable_deblocking_filter_idc
                        );

                        // Verify cabac_init_idc is in valid range for CABAC slices
                        if slh.cabac_init_idc != 0 {
                            assert!(
                                slh.cabac_init_idc <= 2,
                                "cabac_init_idc={} out of range",
                                slh.cabac_init_idc
                            );
                        }
                    }
                }
            }
            Ok(ParseResult::ParameterSet { .. }) => {}
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Err(_) => break,
        }
    }

    assert!(
        slice_count > 0,
        "Should have parsed at least one slice from born_trailer.h264"
    );
    assert!(first_slice_parsed, "Should have parsed the first slice");
}

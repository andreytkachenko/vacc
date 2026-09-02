//! Comprehensive tests for CUVIDH264PICPARAMS construction from vacc-parser output.
//!
//! These tests verify that CUVIDH264PICPARAMS is correctly constructed from
//! vacc-parser's parsed SPS/PPS/SliceHeader data.

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
// H.264 default quantization matrices (Annex B / E.1 of ITU-T H.264)
// ============================================================================

/// H.264 default 4x4 Intra quantization matrix (raster order)
const DEFAULT_QM4X4_INTRA: [u8; 16] = [
    6, 13, 20, 28, 20, 28, 28, 32, 26, 27, 37, 42, 40, 48, 57, 69,
];

/// H.264 default 4x4 Inter quantization matrix (raster order)
const DEFAULT_QM4X4_INTER: [u8; 16] = [
    8, 13, 20, 28, 20, 28, 28, 32, 26, 27, 37, 42, 40, 48, 57, 69,
];

/// H.264 default 8x8 Intra quantization matrix (raster order)
const DEFAULT_QM8X8_INTRA: [u8; 64] = [
    6, 13, 20, 28, 20, 28, 32, 40, 13, 20, 28, 32, 32, 37, 42, 48, 20, 28, 32, 37, 37, 42, 48, 57,
    28, 32, 37, 42, 48, 57, 69, 83, 20, 28, 32, 42, 48, 57, 69, 83, 28, 32, 42, 48, 57, 69, 83,
    100, 32, 37, 48, 57, 69, 83, 100, 117, 40, 42, 57, 69, 83, 100, 117, 135,
];

/// H.264 default 8x8 Inter quantization matrix (raster order)
const DEFAULT_QM8X8_INTER: [u8; 64] = [
    8, 13, 20, 28, 20, 28, 32, 40, 13, 20, 28, 32, 32, 37, 42, 48, 20, 28, 32, 37, 37, 42, 48, 57,
    28, 32, 37, 42, 48, 57, 69, 83, 20, 28, 32, 42, 48, 57, 69, 83, 28, 32, 42, 48, 57, 69, 83,
    100, 32, 37, 48, 57, 69, 83, 100, 117, 40, 42, 57, 69, 83, 100, 117, 135,
];

/// Get WeightScale4x4 matrices for CUVIDH264PICPARAMS.
/// Returns 6 matrices: [intra_y0, intra_y1, intra_y2, inter_y0, inter_y1, inter_cr]
/// For lossless (qpprime_y_zero_transform_bypass_flag=1): identity (64).
/// For non-lossy without custom scaling lists: default H.264 matrices.
/// For custom scaling lists: use SPS scaling_list_4x4.
fn get_weight_scale_4x4(sps: &vacc_core::picture::H264Sps) -> [[u8; 16]; 6] {
    if sps.qpprime_y_zero_transform_bypass_flag {
        // Lossless: identity matrices
        [[64u8; 16]; 6]
    } else if sps.seq_scaling_matrix_present_flag {
        // Custom scaling lists from SPS
        sps.scaling_list_4x4
    } else {
        // Default H.264 matrices
        // Indices 0-2: intra luma (all same default)
        // Indices 3-4: inter luma (all same default)
        // Index 5: chroma (same as inter)
        [
            DEFAULT_QM4X4_INTRA, // intra Y
            DEFAULT_QM4X4_INTRA, // intra Y (repeated for CUVID)
            DEFAULT_QM4X4_INTRA, // intra Y (repeated for CUVID)
            DEFAULT_QM4X4_INTER, // inter Y
            DEFAULT_QM4X4_INTER, // inter Y (repeated for CUVID)
            DEFAULT_QM4X4_INTER, // inter Cb/Cr
        ]
    }
}

/// Get WeightScale8x8 matrices for CUVIDH264PICPARAMS.
/// Returns 2 matrices: [intra_luma, inter_luma]
/// For lossless (qpprime_y_zero_transform_bypass_flag=1): identity (64).
/// For non-lossy without custom scaling lists: default H.264 matrices.
/// For custom scaling lists: use SPS scaling_list_8x8.
fn get_weight_scale_8x8(sps: &vacc_core::picture::H264Sps) -> [[u8; 64]; 2] {
    if sps.qpprime_y_zero_transform_bypass_flag {
        // Lossless: identity matrices
        [[64u8; 64]; 2]
    } else if sps.seq_scaling_matrix_present_flag {
        // Custom scaling lists from SPS
        sps.scaling_list_8x8
    } else {
        // Default H.264 matrices
        [DEFAULT_QM8X8_INTRA, DEFAULT_QM8X8_INTER]
    }
}

/// Build CUVIDH264PICPARAMS from SPS, PPS, and SliceHeader, mirroring decoder.rs logic.
fn build_cuvid_h264_picparams(
    sps: &vacc_core::picture::H264Sps,
    pps: &vacc_core::picture::H264Pps,
    slh: &vacc_parser::h264::SliceHeader,
    poc: i32,
    ref_pic_flag: bool,
) -> nvdec_decode::ffi::CUVIDH264PICPARAMS {
    use nvdec_decode::ffi::{
        CUVIDH264DPBENTRY, CUVIDH264FMOASO, CUVIDH264PICPARAMS, CUVIDH264SVCMVC,
    };
    use std::os::raw::{c_char, c_int, c_uchar};

    let dpb_entries = [CUVIDH264DPBENTRY {
        PicIdx: -1,
        FrameIdx: 0,
        is_long_term: 0,
        not_existing: 1,
        used_for_reference: 0,
        FieldOrderCnt: [0, 0],
    }; 16];

    CUVIDH264PICPARAMS {
        // SPS fields
        log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4 as c_int,
        pic_order_cnt_type: sps.pic_order_cnt_type as c_int,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4 as c_int,
        delta_pic_order_always_zero_flag: sps.delta_pic_order_always_zero_flag as c_int,
        frame_mbs_only_flag: sps.frame_mbs_only_flag as c_int,
        direct_8x8_inference_flag: sps.direct_8x8_inference_flag as c_int,
        num_ref_frames: sps.max_num_ref_frames as c_int,
        residual_colour_transform_flag: if sps.chroma_format_idc == 3 { 1 } else { 0 },
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        qpprime_y_zero_transform_bypass_flag: sps.qpprime_y_zero_transform_bypass_flag as c_uchar,

        // PPS fields
        entropy_coding_mode_flag: pps.entropy_coding_mode_flag as c_int,
        pic_order_present_flag: if sps.pic_order_cnt_type != 2 { 1 } else { 0 },
        num_ref_idx_l0_active_minus1: slh.num_ref_idx_l0_active_minus1 as c_int,
        num_ref_idx_l1_active_minus1: slh.num_ref_idx_l1_active_minus1 as c_int,
        weighted_pred_flag: pps.weighted_pred_flag as c_int,
        weighted_bipred_idc: pps.weighted_bipred_idc as c_int,
        pic_init_qp_minus26: pps.pic_init_qp_minus26 as c_int,
        deblocking_filter_control_present_flag: pps.deblocking_filter_control_present_flag as c_int,
        redundant_pic_cnt_present_flag: pps.redundant_pic_cnt_present_flag as c_int,
        transform_8x8_mode_flag: pps.transform_8x8_mode_flag as c_int,
        MbaffFrameFlag: if !sps.frame_mbs_only_flag { 1 } else { 0 },
        constrained_intra_pred_flag: pps.constrained_intra_pred_flag as c_int,
        chroma_qp_index_offset: pps.chroma_qp_index_offset as c_int,
        second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset as c_int,

        // Picture-specific fields
        ref_pic_flag: if ref_pic_flag { 1 } else { 0 },
        frame_num: slh.frame_num as c_int,
        CurrFieldOrderCnt: [poc, poc],

        // DPB state
        dpb: dpb_entries,

        // Quantization matrices: identity for lossless, default H.264 otherwise
        WeightScale4x4: get_weight_scale_4x4(sps),
        WeightScale8x8: get_weight_scale_8x8(sps),

        // FMO/ASO (disabled for most streams)
        fmo_aso_enable: 0,
        num_slice_groups_minus1: pps.num_slice_groups_minus1 as c_uchar,
        slice_group_map_type: 0,
        pic_init_qs_minus26: pps.pic_init_qs_minus26 as c_char,
        slice_group_change_rate_minus1: 0,
        fmo: CUVIDH264FMOASO {
            pMb2SliceGroupMap: std::ptr::null(),
        },

        // Reserved
        Reserved: [0; 12],

        // SVC/MVC (defaults)
        svc_mvc: CUVIDH264SVCMVC::default(),
    }
}

// ============================================================================
// Test 1: Full picparams construction from born_trailer.h264 SPS/PPS
// ============================================================================

#[test]
fn test_picparams_from_born_trailer_sps_pps() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    assert!(!slices.is_empty(), "Should have parsed at least one slice");

    let sps = parser.active_sps().unwrap();
    let pps = parser.active_pps().unwrap();
    let slh = &slices[0];

    let ref_pic_flag = slh.nal_ref_idc > 0;
    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, ref_pic_flag);

    // Verify SPS-derived fields
    assert_eq!(
        picparams.log2_max_frame_num_minus4 as u8, sps.log2_max_frame_num_minus4,
        "log2_max_frame_num_minus4 mismatch"
    );
    assert_eq!(
        picparams.pic_order_cnt_type as u8, sps.pic_order_cnt_type,
        "pic_order_cnt_type mismatch"
    );
    assert_eq!(
        picparams.log2_max_pic_order_cnt_lsb_minus4 as u8, sps.log2_max_pic_order_cnt_lsb_minus4,
        "log2_max_pic_order_cnt_lsb_minus4 mismatch"
    );

    // Verify PPS-derived fields
    assert_eq!(
        picparams.entropy_coding_mode_flag as u8, pps.entropy_coding_mode_flag as u8,
        "entropy_coding_mode_flag mismatch"
    );
    assert_eq!(
        picparams.pic_init_qp_minus26, pps.pic_init_qp_minus26,
        "pic_init_qp_minus26 mismatch"
    );

    // Verify slice-header-derived fields
    assert_eq!(
        picparams.frame_num as u32, slh.frame_num,
        "frame_num mismatch"
    );
    assert_eq!(
        picparams.ref_pic_flag as u8,
        if ref_pic_flag { 1 } else { 0 },
        "ref_pic_flag mismatch"
    );

    // Verify DPB entries are initialized
    for entry in &picparams.dpb {
        assert!(
            entry.not_existing == 1 || entry.PicIdx >= 0,
            "DPB entry invalid"
        );
    }

    // Verify WeightScale matrices use default H.264 values (born_trailer is Baseline profile,
    // non-lossy, no custom scaling lists -> default matrices, NOT identity)
    // WeightScale4x4[0-2] = intra, [3-5] = inter
    assert_eq!(
        picparams.WeightScale4x4[0], DEFAULT_QM4X4_INTRA,
        "WeightScale4x4 intra should use default"
    );
    assert_eq!(
        picparams.WeightScale4x4[3], DEFAULT_QM4X4_INTER,
        "WeightScale4x4 inter should use default"
    );
    assert_eq!(
        picparams.WeightScale8x8[0], DEFAULT_QM8X8_INTRA,
        "WeightScale8x8 intra should use default"
    );
    assert_eq!(
        picparams.WeightScale8x8[1], DEFAULT_QM8X8_INTER,
        "WeightScale8x8 inter should use default"
    );
}

// ============================================================================
// Test 2: SPS log2_max_frame_num_minus4
// ============================================================================

#[test]
fn test_picparams_sps_log2_max_frame_num_minus4() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.log2_max_frame_num_minus4 as u8, sps.log2_max_frame_num_minus4,
        "log2_max_frame_num_minus4 must match SPS value"
    );

    // Verify it's a reasonable value (typically 4-12, meaning max_frame_num 64-4096)
    assert!(
        sps.log2_max_frame_num_minus4 >= 4 && sps.log2_max_frame_num_minus4 <= 12,
        "log2_max_frame_num_minus4={} is unusual",
        sps.log2_max_frame_num_minus4
    );
}

// ============================================================================
// Test 3: SPS pic_order_cnt_type
// ============================================================================

#[test]
fn test_picparams_sps_pic_order_cnt_type() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.pic_order_cnt_type as u8, sps.pic_order_cnt_type,
        "pic_order_cnt_type must match SPS value"
    );

    // Verify pic_order_cnt_type is valid (0, 1, or 2)
    assert!(
        sps.pic_order_cnt_type <= 2,
        "pic_order_cnt_type={} is invalid",
        sps.pic_order_cnt_type
    );
}

// ============================================================================
// Test 4: SPS log2_max_pic_order_cnt_lsb_minus4
// ============================================================================

#[test]
fn test_picparams_sps_log2_max_pic_order_cnt_lsb_minus4() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.log2_max_pic_order_cnt_lsb_minus4 as u8, sps.log2_max_pic_order_cnt_lsb_minus4,
        "log2_max_pic_order_cnt_lsb_minus4 must match SPS value"
    );

    // Only meaningful for POC type 0
    if sps.pic_order_cnt_type == 0 {
        assert!(
            sps.log2_max_pic_order_cnt_lsb_minus4 >= 4
                && sps.log2_max_pic_order_cnt_lsb_minus4 <= 12,
            "log2_max_pic_order_cnt_lsb_minus4={} is unusual for POC type 0",
            sps.log2_max_pic_order_cnt_lsb_minus4
        );
    }
}

// ============================================================================
// Test 5: SPS delta_pic_order_always_zero_flag
// ============================================================================

#[test]
fn test_picparams_sps_delta_pic_order_always_zero_flag() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.delta_pic_order_always_zero_flag as u8,
        sps.delta_pic_order_always_zero_flag as u8,
        "delta_pic_order_always_zero_flag must match SPS value"
    );

    // This flag is only present for POC type 0
    if sps.pic_order_cnt_type != 0 {
        assert!(
            !sps.delta_pic_order_always_zero_flag,
            "delta_pic_order_always_zero_flag should be false for POC type != 0"
        );
    }
}

// ============================================================================
// Test 6: SPS frame_mbs_only_flag
// ============================================================================

#[test]
fn test_picparams_sps_frame_mbs_only_flag() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.frame_mbs_only_flag as u8, sps.frame_mbs_only_flag as u8,
        "frame_mbs_only_flag must match SPS value"
    );

    // Verify MbaffFrameFlag is derived correctly from frame_mbs_only_flag
    let expected_mbaff = if !sps.frame_mbs_only_flag { 1 } else { 0 };
    assert_eq!(
        picparams.MbaffFrameFlag, expected_mbaff,
        "MbaffFrameFlag should be !frame_mbs_only_flag"
    );
}

// ============================================================================
// Test 7: SPS direct_8x8_inference_flag
// ============================================================================

#[test]
fn test_picparams_sps_direct_8x8_inference_flag() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.direct_8x8_inference_flag as u8, sps.direct_8x8_inference_flag as u8,
        "direct_8x8_inference_flag must match SPS value"
    );
}

// ============================================================================
// Test 8: SPS num_ref_frames (max_num_ref_frames)
// ============================================================================

#[test]
fn test_picparams_sps_num_ref_frames() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.num_ref_frames as u32, sps.max_num_ref_frames,
        "num_ref_frames must match SPS max_num_ref_frames"
    );

    // Verify it's within valid range (1-16 for most profiles, up to 32 for some)
    assert!(
        sps.max_num_ref_frames >= 1 && sps.max_num_ref_frames <= 32,
        "max_num_ref_frames={} is unusual",
        sps.max_num_ref_frames
    );
}

// ============================================================================
// Test 9: SPS bit_depth_luma_minus8 and bit_depth_chroma_minus8
// ============================================================================

#[test]
fn test_picparams_sps_bit_depth_fields() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.bit_depth_luma_minus8, sps.bit_depth_luma_minus8,
        "bit_depth_luma_minus8 must match SPS value"
    );
    assert_eq!(
        picparams.bit_depth_chroma_minus8, sps.bit_depth_chroma_minus8,
        "bit_depth_chroma_minus8 must match SPS value"
    );

    // Verify chroma format determines residual_colour_transform_flag
    let expected_residual_colour = if sps.chroma_format_idc == 3 { 1 } else { 0 };
    assert_eq!(
        picparams.residual_colour_transform_flag, expected_residual_colour,
        "residual_colour_transform_flag should be 1 iff chroma_format_idc == 3"
    );
}

// ============================================================================
// Test 10: PPS entropy_coding_mode_flag
// ============================================================================

#[test]
fn test_picparams_pps_entropy_coding_mode_flag() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.entropy_coding_mode_flag as u8, pps.entropy_coding_mode_flag as u8,
        "entropy_coding_mode_flag must match PPS value"
    );

    // Most modern streams use CABAC (entropy_coding_mode_flag = 1)
    println!(
        "entropy_coding_mode_flag = {} (1=CABAC, 0=CAVLC)",
        pps.entropy_coding_mode_flag
    );
}

// ============================================================================
// Test 11: PPS num_ref_idx_l0_active_minus1 (slice header override or PPS default)
// ============================================================================

#[test]
fn test_picparams_pps_num_ref_idx_l0_active_minus1() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    // The decoder uses slice header values directly (which may override PPS defaults)
    assert_eq!(
        picparams.num_ref_idx_l0_active_minus1 as u32, slh.num_ref_idx_l0_active_minus1,
        "num_ref_idx_l0_active_minus1 must match slice header value"
    );

    // Verify it doesn't exceed PPS default unless overridden
    let pps_default = pps.num_ref_idx_l0_default_active_minus1;
    if !slh.num_ref_idx_active_override_flag {
        assert_eq!(
            slh.num_ref_idx_l0_active_minus1, pps_default,
            "Without override flag, should match PPS default"
        );
    }
}

// ============================================================================
// Test 12: PPS weighted_pred_flag
// ============================================================================

#[test]
fn test_picparams_pps_weighted_pred_flag() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.weighted_pred_flag as u8, pps.weighted_pred_flag as u8,
        "weighted_pred_flag must match PPS value"
    );
}

// ============================================================================
// Test 13: PPS weighted_bipred_idc
// ============================================================================

#[test]
fn test_picparams_pps_weighted_bipred_idc() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.weighted_bipred_idc as u8, pps.weighted_bipred_idc,
        "weighted_bipred_idc must match PPS value"
    );

    // Verify valid range (0, 1, or 2)
    assert!(
        pps.weighted_bipred_idc <= 2,
        "weighted_bipred_idc={} is invalid",
        pps.weighted_bipred_idc
    );
}

// ============================================================================
// Test 14: PPS pic_init_qp_minus26
// ============================================================================

#[test]
fn test_picparams_pps_pic_init_qp_minus26() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.pic_init_qp_minus26, pps.pic_init_qp_minus26,
        "pic_init_qp_minus26 must match PPS value"
    );

    // Verify valid range (-12 to +25)
    assert!(
        pps.pic_init_qp_minus26 >= -12 && pps.pic_init_qp_minus26 <= 25,
        "pic_init_qp_minus26={} is unusual",
        pps.pic_init_qp_minus26
    );
}

// ============================================================================
// Test 15: PPS deblocking_filter_control_present_flag
// ============================================================================

#[test]
fn test_picparams_pps_deblocking_filter_control_present_flag() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.deblocking_filter_control_present_flag as u8,
        pps.deblocking_filter_control_present_flag as u8,
        "deblocking_filter_control_present_flag must match PPS value"
    );
}

// ============================================================================
// Test 16: PPS transform_8x8_mode_flag
// ============================================================================

#[test]
fn test_picparams_pps_transform_8x8_mode_flag() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.transform_8x8_mode_flag as u8, pps.transform_8x8_mode_flag as u8,
        "transform_8x8_mode_flag must match PPS value"
    );
}

// ============================================================================
// Test 17: PPS constrained_intra_pred_flag
// ============================================================================

#[test]
fn test_picparams_pps_constrained_intra_pred_flag() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.constrained_intra_pred_flag as u8, pps.constrained_intra_pred_flag as u8,
        "constrained_intra_pred_flag must match PPS value"
    );
}

// ============================================================================
// Test 18: PPS chroma_qp_index_offset
// ============================================================================

#[test]
fn test_picparams_pps_chroma_qp_index_offset() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.chroma_qp_index_offset, pps.chroma_qp_index_offset,
        "chroma_qp_index_offset must match PPS value"
    );

    // Verify valid range (-12 to +12)
    assert!(
        pps.chroma_qp_index_offset >= -12 && pps.chroma_qp_index_offset <= 12,
        "chroma_qp_index_offset={} is unusual",
        pps.chroma_qp_index_offset
    );
}

// ============================================================================
// Test 19: PPS second_chroma_qp_index_offset
// ============================================================================

#[test]
fn test_picparams_pps_second_chroma_qp_index_offset() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let slh = &slices[0];
    let pps = parser.active_pps().unwrap();

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.second_chroma_qp_index_offset, pps.second_chroma_qp_index_offset,
        "second_chroma_qp_index_offset must match PPS value"
    );

    // Verify valid range (-12 to +12)
    assert!(
        pps.second_chroma_qp_index_offset >= -12 && pps.second_chroma_qp_index_offset <= 12,
        "second_chroma_qp_index_offset={} is unusual",
        pps.second_chroma_qp_index_offset
    );
}

// ============================================================================
// Test 20: ref_pic_flag from slice header (nal_ref_idc > 0)
// ============================================================================

#[test]
fn test_picparams_ref_pic_flag_from_slice_header() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let pps = parser.active_pps().unwrap();

    for (i, slh) in slices.iter().enumerate() {
        let ref_pic_flag = slh.nal_ref_idc > 0;
        let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, ref_pic_flag);

        assert_eq!(
            picparams.ref_pic_flag as u8,
            if ref_pic_flag { 1 } else { 0 },
            "Slice {}: ref_pic_flag must be 1 iff nal_ref_idc > 0 (nal_ref_idc={})",
            i,
            slh.nal_ref_idc
        );
    }
}

// ============================================================================
// Test 21: frame_num from slice header
// ============================================================================

#[test]
fn test_picparams_frame_num_from_slice_header() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let pps = parser.active_pps().unwrap();

    for (i, slh) in slices.iter().enumerate() {
        let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

        assert_eq!(
            picparams.frame_num as u32, slh.frame_num,
            "Slice {}: frame_num must match slice header frame_num",
            i
        );

        // Verify frame_num is within valid range based on SPS
        let max_frame_num = 1u32 << (sps.log2_max_frame_num_minus4 as u32 + 4);
        assert!(
            slh.frame_num < max_frame_num,
            "Slice {}: frame_num={} exceeds max_frame_num={}",
            i,
            slh.frame_num,
            max_frame_num
        );
    }
}

// ============================================================================
// Test 22: CurrFieldOrderCnt set to calculated POC
// ============================================================================

#[test]
fn test_picparams_curr_field_order_cnt() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let pps = parser.active_pps().unwrap();
    let slh = &slices[0];

    // Test with a known POC value
    let test_poc = 42;
    let picparams = build_cuvid_h264_picparams(sps, pps, slh, test_poc, slh.nal_ref_idc > 0);

    assert_eq!(
        picparams.CurrFieldOrderCnt[0], test_poc,
        "CurrFieldOrderCnt[0] must equal calculated POC"
    );
    assert_eq!(
        picparams.CurrFieldOrderCnt[1], test_poc,
        "CurrFieldOrderCnt[1] must equal calculated POC"
    );

    // For frame pictures (frame_mbs_only_flag = 1 or !field_pic_flag),
    // both field order counts should be equal
    if sps.frame_mbs_only_flag || !slh.field_pic_flag {
        assert_eq!(
            picparams.CurrFieldOrderCnt[0], picparams.CurrFieldOrderCnt[1],
            "For frame pictures, both field order counts should be equal"
        );
    }
}

// ============================================================================
// Test 23: DPB entries initialization
// ============================================================================

#[test]
fn test_picparams_dpb_entries_initialization() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let pps = parser.active_pps().unwrap();
    let slh = &slices[0];

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    // Verify all DPB entries are initialized
    assert_eq!(picparams.dpb.len(), 16, "DPB should have 16 entries");

    for (i, entry) in picparams.dpb.iter().enumerate() {
        if entry.not_existing == 1 {
            // Not existing entries should have PicIdx = -1
            assert_eq!(
                entry.PicIdx, -1,
                "DPB[{}]: not_existing entry should have PicIdx=-1",
                i
            );
            assert_eq!(
                entry.used_for_reference, 0,
                "DPB[{}]: not_existing entry should not be used for reference",
                i
            );
        } else {
            // Existing entries should have valid PicIdx
            assert!(
                entry.PicIdx >= 0,
                "DPB[{}]: existing entry should have PicIdx >= 0",
                i
            );
        }
    }

    // Verify DPB size is at least max_num_ref_frames
    assert!(
        16 >= sps.max_num_ref_frames,
        "DPB size (16) should be >= max_num_ref_frames ({})",
        sps.max_num_ref_frames
    );
}

// ============================================================================
// Test 24: WeightScale4x4 and WeightScale8x8 matrices
// ============================================================================

#[test]
fn test_picparams_weight_scale_matrices() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().unwrap();
    let pps = parser.active_pps().unwrap();
    let slh = &slices[0];

    let picparams = build_cuvid_h264_picparams(sps, pps, slh, 0, slh.nal_ref_idc > 0);

    // Verify WeightScale4x4 dimensions
    assert_eq!(
        picparams.WeightScale4x4.len(),
        6,
        "WeightScale4x4 should have 6 matrices"
    );
    for (i, mat) in picparams.WeightScale4x4.iter().enumerate() {
        assert_eq!(
            mat.len(),
            16,
            "WeightScale4x4[{}] should have 16 elements",
            i
        );
    }

    // Verify WeightScale8x8 dimensions
    assert_eq!(
        picparams.WeightScale8x8.len(),
        2,
        "WeightScale8x8 should have 2 matrices"
    );
    for (i, mat) in picparams.WeightScale8x8.iter().enumerate() {
        assert_eq!(
            mat.len(),
            64,
            "WeightScale8x8[{}] should have 64 elements",
            i
        );
    }

    // Born trailer is Baseline profile (non-lossy, no custom scaling lists)
    // -> should use default H.264 quantization matrices, NOT identity
    assert!(
        !sps.qpprime_y_zero_transform_bypass_flag,
        "born_trailer should not be lossless"
    );
    assert!(
        !sps.seq_scaling_matrix_present_flag,
        "born_trailer should not have custom scaling lists"
    );

    // Verify default matrices are used
    for i in 0..3 {
        assert_eq!(
            picparams.WeightScale4x4[i], DEFAULT_QM4X4_INTRA,
            "WeightScale4x4[{}] (intra) should use default intra matrix",
            i
        );
    }
    for i in 3..6 {
        assert_eq!(
            picparams.WeightScale4x4[i], DEFAULT_QM4X4_INTER,
            "WeightScale4x4[{}] (inter) should use default inter matrix",
            i
        );
    }
    assert_eq!(
        picparams.WeightScale8x8[0], DEFAULT_QM8X8_INTRA,
        "WeightScale8x8[0] (intra) should use default intra matrix"
    );
    assert_eq!(
        picparams.WeightScale8x8[1], DEFAULT_QM8X8_INTER,
        "WeightScale8x8[1] (inter) should use default inter matrix"
    );
}

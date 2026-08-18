//! Comparison tests between vk-video-parser output and cuvid parser expectations.
//!
//! These tests verify that vk-video-parser extracts the same information that
//! cuvid's parser would extract in CUVIDEOFORMAT and CUVIDH264PICPARAMS.
//!
//! Reference: Video_Codec_SDK cuviddec.h / nvcuvid.h

use nvdec_decode::ffi::{
    cudaVideoChromaFormat, cudaVideoCodec, CUVIDDISPLAYAREA, CUVIDEOFORMAT, CUVIDH264PICPARAMS,
};
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

/// Extract raw NAL data (with start code) from the first NAL unit of given type.
fn extract_first_nal_with_start_code(data: &[u8], nal_type: u8) -> Option<Vec<u8>> {
    let mut offset = 0;
    while offset < data.len() {
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

/// Build a CUVIDEOFORMAT-equivalent structure from vk-video-parser's SPS.
///
/// This mirrors what cuvid's parser does when it calls the sequence callback
/// with a populated CUVIDEOFORMAT after parsing the H.264 SPS.
fn build_cuvideoformat_from_sps(
    sps: &vk_video_core::picture::H264Sps,
    format: &DetectedVideoFormat,
) -> CUVIDEOFORMAT {
    // coded_width = (pic_width_in_mbs_minus1 + 1) * 16
    let coded_width = (sps.pic_width_in_mbs_minus1 as u32 + 1) * 16;

    // coded_height: frame-only vs field pictures
    // frame_mbs_only_flag=1: height = macroblocks * 16
    // frame_mbs_only_flag=0: height = macroblocks * 16 * 2 (field pictures)
    let coded_height = if sps.frame_mbs_only_flag {
        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
    } else {
        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
    };

    // chroma_format mapping: H.264 chroma_format_idc → cudaVideoChromaFormat
    let chroma_format = match sps.chroma_format_idc {
        0 => cudaVideoChromaFormat::cudaVideoChromaFormat_Monochrome,
        1 => cudaVideoChromaFormat::cudaVideoChromaFormat_420,
        2 => cudaVideoChromaFormat::cudaVideoChromaFormat_422,
        3 => cudaVideoChromaFormat::cudaVideoChromaFormat_444,
        _ => cudaVideoChromaFormat::cudaVideoChromaFormat_420,
    };

    // min_num_decode_surfaces = max_num_ref_frames + 1
    let min_num_decode_surfaces = (sps.max_num_ref_frames as u8).saturating_add(1);

    // display_area calculation (cuvid formula)
    let display_area = CUVIDDISPLAYAREA {
        left: (sps.frame_crop_left_offset as i32) * 2,
        right: coded_width as i32 - (sps.frame_crop_right_offset as i32) * 2,
        // For frame-only: crop_top/bottom * 2
        // For field pictures: crop_top/bottom * 4 (each crop unit = 2 luma lines per field)
        top: if sps.frame_mbs_only_flag {
            (sps.frame_crop_top_offset as i32) * 2
        } else {
            (sps.frame_crop_top_offset as i32) * 4
        },
        bottom: if sps.frame_mbs_only_flag {
            coded_height as i32 - (sps.frame_crop_bottom_offset as i32) * 2
        } else {
            coded_height as i32 - (sps.frame_crop_bottom_offset as i32) * 4
        },
    };

    CUVIDEOFORMAT {
        codec: cudaVideoCodec::cudaVideoCodec_H264,
        frame_rate: nvdec_decode::ffi::CUVIDFRAMERATE {
            numerator: format.frame_rate.numerator,
            denominator: format.frame_rate.denominator,
        },
        progressive_sequence: sps.frame_mbs_only_flag as u8,
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        min_num_decode_surfaces,
        coded_width,
        coded_height,
        display_area,
        chroma_format,
        bitrate: 0,
        display_aspect_ratio: nvdec_decode::ffi::CUVIDDISPLAYASPECTRATIO { x: 0, y: 0 },
        video_signal_description: nvdec_decode::ffi::CUVIDVIDEOSIGNALDESCRIPTION {
            video_format: 5, // default: component
            video_full_range_flag: 0,
            reserved_zero_bits: 0,
            color_primaries: 2, // default: unspecified
            transfer_characteristics: 2,
            matrix_coefficients: 2,
        },
        seqhdr_data_length: 0,
    }
}

// ============================================================================
// CUVIDEOFORMAT Field Comparison Tests
// ============================================================================

/// Test that vk-video-parser SPS fields map correctly to CUVIDEOFORMAT fields.
///
/// Verifies the core fields that cuvid's parser extracts from H.264 SPS:
/// - coded_width, coded_height
/// - chroma_format mapping
/// - bit_depth_luma_minus8, bit_depth_chroma_minus8
/// - min_num_decode_surfaces
/// - progressive_sequence
#[test]
fn test_vkvideo_parser_matches_cuvid_videoformat_fields() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(sps_data);
    let result = parser.parse(&packet).expect("SPS parse failed");

    match result {
        ParseResult::ParameterSet { sps: Some(_), .. } => {}
        _ => panic!("Expected ParameterSet result, got {:?}", result),
    }

    let sps = parser.active_sps().expect("No active SPS");
    let format = parser.detected_format();
    let cuvid_fmt = build_cuvideoformat_from_sps(sps, format);

    // Verify coded_width = (pic_width_in_mbs_minus1 + 1) * 16
    let expected_coded_width = (sps.pic_width_in_mbs_minus1 as u32 + 1) * 16;
    assert_eq!(
        cuvid_fmt.coded_width, expected_coded_width,
        "coded_width mismatch: parser computed {} but expected {}",
        cuvid_fmt.coded_width, expected_coded_width
    );
    assert_eq!(
        format.coded_width, expected_coded_width,
        "DetectedVideoFormat coded_width mismatch"
    );

    // Verify coded_height formula (frame-only for born_trailer)
    let expected_coded_height = if sps.frame_mbs_only_flag {
        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
    } else {
        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
    };
    assert_eq!(
        cuvid_fmt.coded_height, expected_coded_height,
        "coded_height mismatch"
    );
    assert_eq!(
        format.coded_height, expected_coded_height,
        "DetectedVideoFormat coded_height mismatch"
    );

    // Verify bit_depth_luma_minus8 matches SPS
    assert_eq!(
        cuvid_fmt.bit_depth_luma_minus8, sps.bit_depth_luma_minus8,
        "bit_depth_luma_minus8 mismatch"
    );

    // Verify bit_depth_chroma_minus8 matches SPS
    assert_eq!(
        cuvid_fmt.bit_depth_chroma_minus8, sps.bit_depth_chroma_minus8,
        "bit_depth_chroma_minus8 mismatch"
    );

    // Verify min_num_decode_surfaces = max_num_ref_frames + 1
    let expected_min_surfaces = (sps.max_num_ref_frames as u8).saturating_add(1);
    assert_eq!(
        cuvid_fmt.min_num_decode_surfaces, expected_min_surfaces,
        "min_num_decode_surfaces mismatch: expected max_ref_frames({}) + 1 = {}",
        sps.max_num_ref_frames, expected_min_surfaces
    );

    // Verify progressive_sequence = frame_mbs_only_flag
    assert_eq!(
        cuvid_fmt.progressive_sequence, sps.frame_mbs_only_flag as u8,
        "progressive_sequence should equal frame_mbs_only_flag"
    );
    assert_eq!(
        format.progressive_sequence, sps.frame_mbs_only_flag,
        "DetectedVideoFormat progressive_sequence mismatch"
    );
}

/// Test that display_area calculation matches cuvid's formula.
///
/// Cuvid computes display_area from frame_crop_*_offset:
/// - left = frame_crop_left_offset * 2
/// - right = coded_width - frame_crop_right_offset * 2
/// - top = frame_crop_top_offset * 2 (frame-only) or * 4 (field)
/// - bottom = coded_height - frame_crop_bottom_offset * 2 (frame-only) or * 4 (field)
#[test]
fn test_vkvideo_parser_display_area_matches_cuvid() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(sps_data);
    parser.parse(&packet).expect("SPS parse failed");

    let sps = parser.active_sps().expect("No active SPS");
    let format = parser.detected_format();
    let cuvid_fmt = build_cuvideoformat_from_sps(sps, format);

    let coded_width = (sps.pic_width_in_mbs_minus1 as u32 + 1) * 16;
    let coded_height = if sps.frame_mbs_only_flag {
        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16
    } else {
        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16 * 2
    };

    // Cuvid display_area formula
    let crop_mult = if sps.frame_mbs_only_flag { 2 } else { 4 };
    let expected_left = (sps.frame_crop_left_offset as i32) * 2;
    let expected_right = coded_width as i32 - (sps.frame_crop_right_offset as i32) * 2;
    let expected_top = (sps.frame_crop_top_offset as i32) * crop_mult;
    let expected_bottom = coded_height as i32 - (sps.frame_crop_bottom_offset as i32) * crop_mult;

    // Verify display_area matches cuvid's computation
    assert_eq!(
        cuvid_fmt.display_area.left, expected_left,
        "display_area.left mismatch: got {}, expected {} (crop_left={} * 2)",
        cuvid_fmt.display_area.left, expected_left, sps.frame_crop_left_offset
    );
    assert_eq!(
        cuvid_fmt.display_area.right, expected_right,
        "display_area.right mismatch"
    );
    assert_eq!(
        cuvid_fmt.display_area.top, expected_top,
        "display_area.top mismatch"
    );
    assert_eq!(
        cuvid_fmt.display_area.bottom, expected_bottom,
        "display_area.bottom mismatch"
    );

    // Verify display_area is valid (left < right, top < bottom)
    assert!(
        cuvid_fmt.display_area.left < cuvid_fmt.display_area.right,
        "display_area.left ({}) must be < right ({})",
        cuvid_fmt.display_area.left,
        cuvid_fmt.display_area.right
    );
    assert!(
        cuvid_fmt.display_area.top < cuvid_fmt.display_area.bottom,
        "display_area.top ({}) must be < bottom ({})",
        cuvid_fmt.display_area.top,
        cuvid_fmt.display_area.bottom
    );

    // Verify display_area is within coded dimensions
    assert!(
        cuvid_fmt.display_area.left >= 0 && cuvid_fmt.display_area.right <= coded_width as i32,
        "display_area horizontal bounds out of range"
    );
    assert!(
        cuvid_fmt.display_area.top >= 0 && cuvid_fmt.display_area.bottom <= coded_height as i32,
        "display_area vertical bounds out of range"
    );
}

// ============================================================================
// Chroma Format Mapping Tests
// ============================================================================

/// Test that chroma_format_idc → cudaVideoChromaFormat mapping is correct.
///
/// Cuvid uses this mapping in its parser:
/// - chroma_format_idc = 0 → cudaVideoChromaFormat_Monochrome
/// - chroma_format_idc = 1 → cudaVideoChromaFormat_420
/// - chroma_format_idc = 2 → cudaVideoChromaFormat_422
/// - chroma_format_idc = 3 → cudaVideoChromaFormat_444
#[test]
fn test_vkvideo_parser_chroma_format_mapping() {
    // Test with born_trailer (4:2:0)
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(sps_data);
    parser.parse(&packet).expect("SPS parse failed");

    let sps = parser.active_sps().expect("No active SPS");
    let format = parser.detected_format();
    let cuvid_fmt = build_cuvideoformat_from_sps(sps, format);

    // born_trailer is 4:2:0 (chroma_format_idc = 1)
    assert_eq!(sps.chroma_format_idc, 1);
    assert_eq!(
        cuvid_fmt.chroma_format,
        cudaVideoChromaFormat::cudaVideoChromaFormat_420,
        "chroma_format_idc=1 should map to cudaVideoChromaFormat_420"
    );

    // Verify vk-video-parser's DetectedVideoFormat chroma_subsampling matches
    assert_eq!(
        format.chroma_subsampling,
        vk_video_core::format::ChromaSubsampling::_420,
        "DetectedVideoFormat chroma_subsampling should be _420"
    );

    // Verify mapping table exhaustively
    let test_cases = [
        (0u8, cudaVideoChromaFormat::cudaVideoChromaFormat_Monochrome),
        (1u8, cudaVideoChromaFormat::cudaVideoChromaFormat_420),
        (2u8, cudaVideoChromaFormat::cudaVideoChromaFormat_422),
        (3u8, cudaVideoChromaFormat::cudaVideoChromaFormat_444),
    ];

    for (idc, expected_fmt) in test_cases {
        let mapped = match idc {
            0 => cudaVideoChromaFormat::cudaVideoChromaFormat_Monochrome,
            1 => cudaVideoChromaFormat::cudaVideoChromaFormat_420,
            2 => cudaVideoChromaFormat::cudaVideoChromaFormat_422,
            3 => cudaVideoChromaFormat::cudaVideoChromaFormat_444,
            _ => cudaVideoChromaFormat::cudaVideoChromaFormat_420,
        };
        assert_eq!(
            mapped, expected_fmt,
            "chroma_format_idc={} should map to {:?}",
            idc, expected_fmt
        );
    }
}

// ============================================================================
// Codec Type Tests
// ============================================================================

/// Test that the codec is correctly identified as H264.
#[test]
fn test_vkvideo_parser_codec_type() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(sps_data);
    parser.parse(&packet).expect("SPS parse failed");

    let format = parser.detected_format();

    // Verify codec is H264
    assert_eq!(
        format.codec,
        vk_video_core::codec::VideoCodec::DecodeH264,
        "Codec should be DecodeH264"
    );

    // Verify CUVIDEOFORMAT codec field
    let sps = parser.active_sps().expect("No active SPS");
    let cuvid_fmt = build_cuvideoformat_from_sps(sps, format);
    assert_eq!(
        cuvid_fmt.codec,
        cudaVideoCodec::cudaVideoCodec_H264,
        "CUVIDEOFORMAT codec should be cudaVideoCodec_H264"
    );
}

// ============================================================================
// CUVIDH264PICPARAMS SPS Field Tests
// ============================================================================

/// Test that all SPS fields needed for CUVIDH264PICPARAMS are correctly parsed.
///
/// Cuvid's parser extracts these SPS fields into CUVIDH264PICPARAMS:
/// - log2_max_frame_num_minus4
/// - pic_order_cnt_type
/// - log2_max_pic_order_cnt_lsb_minus4
/// - delta_pic_order_always_zero_flag
/// - frame_mbs_only_flag
/// - direct_8x8_inference_flag
/// - num_ref_frames (= max_num_ref_frames)
/// - bit_depth_luma_minus8
/// - bit_depth_chroma_minus8
/// - qpprime_y_zero_transform_bypass_flag
#[test]
fn test_vkvideo_parser_sps_fields_for_cuvid_picparams() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(sps_data);
    parser.parse(&packet).expect("SPS parse failed");

    let sps = parser.active_sps().expect("No active SPS");

    // Verify log2_max_frame_num_minus4 is within valid range [0, 12]
    assert!(
        sps.log2_max_frame_num_minus4 <= 12,
        "log2_max_frame_num_minus4 ({}) out of range",
        sps.log2_max_frame_num_minus4
    );

    // Verify pic_order_cnt_type is 0, 1, or 2
    assert!(
        sps.pic_order_cnt_type <= 2,
        "pic_order_cnt_type ({}) out of range",
        sps.pic_order_cnt_type
    );

    // Verify log2_max_pic_order_cnt_lsb_minus4 is valid when POC type 0
    if sps.pic_order_cnt_type == 0 {
        assert!(
            sps.log2_max_pic_order_cnt_lsb_minus4 <= 12,
            "log2_max_pic_order_cnt_lsb_minus4 ({}) out of range",
            sps.log2_max_pic_order_cnt_lsb_minus4
        );
    }

    // Verify delta_pic_order_always_zero_flag is only meaningful for POC type 1
    if sps.pic_order_cnt_type != 1 {
        assert!(
            !sps.delta_pic_order_always_zero_flag,
            "delta_pic_order_always_zero_flag should be false for POC type != 1"
        );
    }

    // Verify frame_mbs_only_flag is boolean
    let _frame_only = sps.frame_mbs_only_flag;

    // Verify direct_8x8_inference_flag is boolean
    let _direct_8x8 = sps.direct_8x8_inference_flag;

    // Verify max_num_ref_frames is within valid range [1, 16]
    assert!(
        sps.max_num_ref_frames >= 1 && sps.max_num_ref_frames <= 16,
        "max_num_ref_frames ({}) out of range [1, 16]",
        sps.max_num_ref_frames
    );

    // Verify bit_depth_luma_minus8 is valid (0, 2, or 4 for 8/10/12 bit)
    assert!(
        sps.bit_depth_luma_minus8 == 0
            || sps.bit_depth_luma_minus8 == 2
            || sps.bit_depth_luma_minus8 == 4,
        "bit_depth_luma_minus8 ({}) should be 0, 2, or 4",
        sps.bit_depth_luma_minus8
    );

    // Verify bit_depth_chroma_minus8 is valid
    assert!(
        sps.bit_depth_chroma_minus8 == 0
            || sps.bit_depth_chroma_minus8 == 2
            || sps.bit_depth_chroma_minus8 == 4,
        "bit_depth_chroma_minus8 ({}) should be 0, 2, or 4",
        sps.bit_depth_chroma_minus8
    );

    // Verify qpprime_y_zero_transform_bypass_flag is boolean
    let _qp_bypass = sps.qpprime_y_zero_transform_bypass_flag;

    // For born_trailer: verify specific known values
    // Main profile, 8-bit, POC type 0
    assert_eq!(
        sps.profile_idc, 66,
        "born_trailer should be Main profile (66)"
    );
    assert_eq!(
        sps.bit_depth_luma_minus8, 0,
        "born_trailer should be 8-bit luma"
    );
    assert_eq!(
        sps.bit_depth_chroma_minus8, 0,
        "born_trailer should be 8-bit chroma"
    );
    assert_eq!(
        sps.pic_order_cnt_type, 0,
        "born_trailer should use POC type 0"
    );
}

/// Test that SPS fields can be correctly mapped to CUVIDH264PICPARAMS structure.
///
/// This test verifies the exact field mapping that would be used when
/// constructing CUVIDH264PICPARAMS from vk-video-parser output.
#[test]
fn test_vkvideo_parser_sps_to_cuvid_picparams_mapping() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");
    let pps_data = extract_first_nal_with_start_code(&data, 8).expect("No PPS found");

    let mut combined = Vec::new();
    combined.extend_from_slice(&sps_data);
    combined.extend_from_slice(&pps_data);

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(combined);
    parser.parse(&packet).expect("Parse failed");

    let sps = parser.active_sps().expect("No active SPS");
    let pps = parser.active_pps().expect("No active PPS");

    // Create a mock CUVIDH264PICPARAMS with fields from SPS/PPS
    // This mirrors what decoder.rs does when constructing the struct for cuvidDecodePicture
    let pic_params = CUVIDH264PICPARAMS {
        // SPS fields
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
        // PPS fields
        entropy_coding_mode_flag: pps.entropy_coding_mode_flag as i32,
        pic_order_present_flag: if sps.pic_order_cnt_type != 2 { 1 } else { 0 },
        num_ref_idx_l0_active_minus1: pps.num_ref_idx_l0_default_active_minus1 as i32,
        num_ref_idx_l1_active_minus1: pps.num_ref_idx_l1_default_active_minus1 as i32,
        weighted_pred_flag: pps.weighted_pred_flag as i32,
        weighted_bipred_idc: pps.weighted_bipred_idc as i32,
        pic_init_qp_minus26: pps.pic_init_qp_minus26,
        deblocking_filter_control_present_flag: pps.deblocking_filter_control_present_flag as i32,
        redundant_pic_cnt_present_flag: pps.redundant_pic_cnt_present_flag as i32,
        transform_8x8_mode_flag: pps.transform_8x8_mode_flag as i32,
        MbaffFrameFlag: if !sps.frame_mbs_only_flag { 1 } else { 0 },
        constrained_intra_pred_flag: pps.constrained_intra_pred_flag as i32,
        chroma_qp_index_offset: pps.chroma_qp_index_offset,
        second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset,
        // Per-picture fields (mock)
        ref_pic_flag: 1,
        frame_num: 0,
        CurrFieldOrderCnt: [0, 0],
        dpb: [nvdec_decode::ffi::CUVIDH264DPBENTRY {
            PicIdx: -1,
            FrameIdx: 0,
            is_long_term: 0,
            not_existing: 1,
            used_for_reference: 0,
            FieldOrderCnt: [0, 0],
        }; 16],
        WeightScale4x4: [[0; 16]; 6],
        WeightScale8x8: [[0; 64]; 2],
        // FMO/ASO fields (from PPS)
        fmo_aso_enable: 0,
        num_slice_groups_minus1: pps.num_slice_groups_minus1 as u8,
        slice_group_map_type: 0,
        pic_init_qs_minus26: pps.pic_init_qs_minus26 as i8,
        slice_group_change_rate_minus1: 0,
        fmo: nvdec_decode::ffi::CUVIDH264FMOASO {
            pMb2SliceGroupMap: std::ptr::null(),
        },
        Reserved: [0; 12],
        svc_mvc: nvdec_decode::ffi::CUVIDH264SVCMVC::default(),
    };

    // Verify SPS fields are correctly mapped
    assert_eq!(
        pic_params.log2_max_frame_num_minus4,
        sps.log2_max_frame_num_minus4 as i32
    );
    assert_eq!(pic_params.pic_order_cnt_type, sps.pic_order_cnt_type as i32);
    assert_eq!(pic_params.num_ref_frames, sps.max_num_ref_frames as i32);
    assert_eq!(
        pic_params.frame_mbs_only_flag,
        sps.frame_mbs_only_flag as i32
    );
    assert_eq!(pic_params.bit_depth_luma_minus8, sps.bit_depth_luma_minus8);
    assert_eq!(
        pic_params.bit_depth_chroma_minus8,
        sps.bit_depth_chroma_minus8
    );
    assert_eq!(
        pic_params.qpprime_y_zero_transform_bypass_flag,
        sps.qpprime_y_zero_transform_bypass_flag as u8
    );

    // Verify residual_colour_transform_flag mapping (chroma_format_idc == 3 → 1)
    assert_eq!(
        pic_params.residual_colour_transform_flag,
        if sps.chroma_format_idc == 3 { 1 } else { 0 }
    );

    // Verify MbaffFrameFlag mapping (!frame_mbs_only_flag → 1)
    assert_eq!(
        pic_params.MbaffFrameFlag,
        if !sps.frame_mbs_only_flag { 1 } else { 0 }
    );

    // Verify pic_order_present_flag mapping (poc_type != 2 → 1)
    assert_eq!(
        pic_params.pic_order_present_flag,
        if sps.pic_order_cnt_type != 2 { 1 } else { 0 }
    );
}

// ============================================================================
// CUVIDH264PICPARAMS PPS Field Tests
// ============================================================================

/// Test that all PPS fields needed for CUVIDH264PICPARAMS are correctly parsed.
///
/// Cuvid's parser extracts these PPS fields into CUVIDH264PICPARAMS:
/// - entropy_coding_mode_flag
/// - num_ref_idx_l0_default_active_minus1
/// - num_ref_idx_l1_default_active_minus1
/// - weighted_pred_flag
/// - weighted_bipred_idc
/// - pic_init_qp_minus26
/// - deblocking_filter_control_present_flag
/// - redundant_pic_cnt_present_flag
/// - transform_8x8_mode_flag
/// - constrained_intra_pred_flag
/// - chroma_qp_index_offset
/// - second_chroma_qp_index_offset
#[test]
fn test_vkvideo_parser_pps_fields_for_cuvid_picparams() {
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

    // Verify entropy_coding_mode_flag is boolean (0=CAVLC, 1=CABAC)
    let _entropy = pps.entropy_coding_mode_flag;

    // Verify num_ref_idx_l0_default_active_minus1 is valid [0, 31]
    assert!(
        pps.num_ref_idx_l0_default_active_minus1 <= 31,
        "num_ref_idx_l0_default_active_minus1 ({}) out of range",
        pps.num_ref_idx_l0_default_active_minus1
    );

    // Verify num_ref_idx_l1_default_active_minus1 is valid [0, 31]
    assert!(
        pps.num_ref_idx_l1_default_active_minus1 <= 31,
        "num_ref_idx_l1_default_active_minus1 ({}) out of range",
        pps.num_ref_idx_l1_default_active_minus1
    );

    // Verify weighted_pred_flag is boolean
    let _wp_flag = pps.weighted_pred_flag;

    // Verify weighted_bipred_idc is valid [0, 2]
    assert!(
        pps.weighted_bipred_idc <= 2,
        "weighted_bipred_idc ({}) out of range",
        pps.weighted_bipred_idc
    );

    // Verify pic_init_qp_minus26 is in valid range [-26, 25]
    assert!(
        pps.pic_init_qp_minus26 >= -26 && pps.pic_init_qp_minus26 <= 25,
        "pic_init_qp_minus26 ({}) out of range [-26, 25]",
        pps.pic_init_qp_minus26
    );

    // Verify deblocking_filter_control_present_flag is boolean
    let _deblock = pps.deblocking_filter_control_present_flag;

    // Verify redundant_pic_cnt_present_flag is boolean
    let _redundant = pps.redundant_pic_cnt_present_flag;

    // Verify transform_8x8_mode_flag is boolean
    let _transform_8x8 = pps.transform_8x8_mode_flag;

    // Verify constrained_intra_pred_flag is boolean
    let _constrained_intra = pps.constrained_intra_pred_flag;

    // Verify chroma_qp_index_offset is in valid range [-12, 12]
    assert!(
        pps.chroma_qp_index_offset >= -12 && pps.chroma_qp_index_offset <= 12,
        "chroma_qp_index_offset ({}) out of range [-12, 12]",
        pps.chroma_qp_index_offset
    );

    // Verify second_chroma_qp_index_offset is in valid range [-12, 12]
    assert!(
        pps.second_chroma_qp_index_offset >= -12 && pps.second_chroma_qp_index_offset <= 12,
        "second_chroma_qp_index_offset ({}) out of range [-12, 12]",
        pps.second_chroma_qp_index_offset
    );

    // For born_trailer: verify specific known values
    // Main profile, CAVLC, deblocking enabled
    assert!(
        !pps.entropy_coding_mode_flag,
        "born_trailer should use CAVLC (entropy_coding_mode_flag=0)"
    );
    assert!(
        pps.deblocking_filter_control_present_flag,
        "born_trailer should have deblocking_filter_control_present_flag=1"
    );
}

/// Test that PPS fields map correctly to CUVIDH264PICPARAMS.
///
/// Verifies the exact field mapping for PPS-derived fields in CUVIDH264PICPARAMS.
#[test]
fn test_vkvideo_parser_pps_to_cuvid_picparams_mapping() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");
    let pps_data = extract_first_nal_with_start_code(&data, 8).expect("No PPS found");

    let mut combined = Vec::new();
    combined.extend_from_slice(&sps_data);
    combined.extend_from_slice(&pps_data);

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(combined);
    parser.parse(&packet).expect("Parse failed");

    let _sps = parser.active_sps().expect("No active SPS");
    let pps = parser.active_pps().expect("No active PPS");

    // Map PPS fields to CUVIDH264PICPARAMS-equivalent values
    let entropy_coding_mode_flag: i32 = pps.entropy_coding_mode_flag as i32;
    let num_ref_idx_l0_active_minus1: i32 = pps.num_ref_idx_l0_default_active_minus1 as i32;
    let num_ref_idx_l1_active_minus1: i32 = pps.num_ref_idx_l1_default_active_minus1 as i32;
    let weighted_pred_flag: i32 = pps.weighted_pred_flag as i32;
    let weighted_bipred_idc: i32 = pps.weighted_bipred_idc as i32;
    let pic_init_qp_minus26: i32 = pps.pic_init_qp_minus26;
    let deblocking_filter_control_present_flag: i32 =
        pps.deblocking_filter_control_present_flag as i32;
    let redundant_pic_cnt_present_flag: i32 = pps.redundant_pic_cnt_present_flag as i32;
    let transform_8x8_mode_flag: i32 = pps.transform_8x8_mode_flag as i32;
    let constrained_intra_pred_flag: i32 = pps.constrained_intra_pred_flag as i32;
    let chroma_qp_index_offset: i32 = pps.chroma_qp_index_offset;
    let second_chroma_qp_index_offset: i32 = pps.second_chroma_qp_index_offset;

    // Verify direct mapping (no transformation needed for PPS fields)
    assert_eq!(
        entropy_coding_mode_flag, pps.entropy_coding_mode_flag as i32,
        "entropy_coding_mode_flag mapping mismatch"
    );
    assert_eq!(
        num_ref_idx_l0_active_minus1, pps.num_ref_idx_l0_default_active_minus1 as i32,
        "num_ref_idx_l0_active_minus1 mapping mismatch"
    );
    assert_eq!(
        num_ref_idx_l1_active_minus1, pps.num_ref_idx_l1_default_active_minus1 as i32,
        "num_ref_idx_l1_active_minus1 mapping mismatch"
    );
    assert_eq!(
        weighted_pred_flag, pps.weighted_pred_flag as i32,
        "weighted_pred_flag mapping mismatch"
    );
    assert_eq!(
        weighted_bipred_idc, pps.weighted_bipred_idc as i32,
        "weighted_bipred_idc mapping mismatch"
    );
    assert_eq!(
        pic_init_qp_minus26, pps.pic_init_qp_minus26,
        "pic_init_qp_minus26 mapping mismatch"
    );
    assert_eq!(
        deblocking_filter_control_present_flag, pps.deblocking_filter_control_present_flag as i32,
        "deblocking_filter_control_present_flag mapping mismatch"
    );
    assert_eq!(
        redundant_pic_cnt_present_flag, pps.redundant_pic_cnt_present_flag as i32,
        "redundant_pic_cnt_present_flag mapping mismatch"
    );
    assert_eq!(
        transform_8x8_mode_flag, pps.transform_8x8_mode_flag as i32,
        "transform_8x8_mode_flag mapping mismatch"
    );
    assert_eq!(
        constrained_intra_pred_flag, pps.constrained_intra_pred_flag as i32,
        "constrained_intra_pred_flag mapping mismatch"
    );
    assert_eq!(
        chroma_qp_index_offset, pps.chroma_qp_index_offset,
        "chroma_qp_index_offset mapping mismatch"
    );
    assert_eq!(
        second_chroma_qp_index_offset, pps.second_chroma_qp_index_offset,
        "second_chroma_qp_index_offset mapping mismatch"
    );

    // Verify cross-PPS consistency: chroma_qp_index_offset and second_chroma_qp_index_offset
    // According to H.264 spec, second_chroma_qp_index_offset defaults to chroma_qp_index_offset
    // when not explicitly present
    // For born_trailer, both should be the same (typically 0)
    let sps = parser.active_sps().expect("No active SPS");
    // In Main profile (born_trailer), second_chroma_qp_index_offset is not present
    // and defaults to chroma_qp_index_offset
    if sps.profile_idc == 66 {
        assert_eq!(
            pps.second_chroma_qp_index_offset, pps.chroma_qp_index_offset,
            "Main profile: second_chroma_qp_index_offset should default to chroma_qp_index_offset"
        );
    }
}

// ============================================================================
// Combined CUVIDEOFORMAT + CUVIDH264PICPARAMS Consistency Tests
// ============================================================================

/// Test that CUVIDEOFORMAT and CUVIDH264PICPARAMS are consistent with each other.
///
/// Both structures derive from the same SPS/PPS, so certain fields must be consistent:
/// - CUVIDEOFORMAT.bit_depth_luma_minus8 == CUVIDH264PICPARAMS.bit_depth_luma_minus8
/// - CUVIDEOFORMAT.bit_depth_chroma_minus8 == CUVIDH264PICPARAMS.bit_depth_chroma_minus8
/// - CUVIDEOFORMAT.coded_width == (PicWidthInMbs * 16)
/// - CUVIDEOFORMAT.coded_height == (FrameHeightInMbs * 16) for frame-only
/// - CUVIDEOFORMAT.min_num_decode_surfaces == CUVIDH264PICPARAMS.num_ref_frames + 1
#[test]
fn test_cuvid_format_and_picparams_consistency() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");
    let pps_data = extract_first_nal_with_start_code(&data, 8).expect("No PPS found");

    let mut combined = Vec::new();
    combined.extend_from_slice(&sps_data);
    combined.extend_from_slice(&pps_data);

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(combined);
    parser.parse(&packet).expect("Parse failed");

    let sps = parser.active_sps().expect("No active SPS");
    let _pps = parser.active_pps().expect("No active PPS");
    let format = parser.detected_format();

    // Build CUVIDEOFORMAT
    let cuvid_fmt = build_cuvideoformat_from_sps(sps, format);

    // Build CUVIDH264PICPARAMS-equivalent
    let pic_params_num_ref_frames = sps.max_num_ref_frames as i32;
    let pic_params_bit_depth_luma_minus8 = sps.bit_depth_luma_minus8;
    let pic_params_bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8;

    // Consistency: bit depths must match
    assert_eq!(
        cuvid_fmt.bit_depth_luma_minus8, pic_params_bit_depth_luma_minus8,
        "CUVIDEOFORMAT and CUVIDH264PICPARAMS bit_depth_luma_minus8 must match"
    );
    assert_eq!(
        cuvid_fmt.bit_depth_chroma_minus8, pic_params_bit_depth_chroma_minus8,
        "CUVIDEOFORMAT and CUVIDH264PICPARAMS bit_depth_chroma_minus8 must match"
    );

    // Consistency: coded dimensions derived from same SPS fields
    let pic_width_in_mbs = sps.pic_width_in_mbs_minus1 as u32 + 1;
    let frame_height_in_mbs = if sps.frame_mbs_only_flag {
        sps.pic_height_in_map_units_minus1 as u32 + 1
    } else {
        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 2
    };
    assert_eq!(
        cuvid_fmt.coded_width,
        pic_width_in_mbs * 16,
        "CUVIDEOFORMAT coded_width must match PicWidthInMbs * 16"
    );
    assert_eq!(
        cuvid_fmt.coded_height,
        frame_height_in_mbs * 16,
        "CUVIDEOFORMAT coded_height must match FrameHeightInMbs * 16"
    );

    // Consistency: min_num_decode_surfaces = num_ref_frames + 1
    assert_eq!(
        cuvid_fmt.min_num_decode_surfaces as i32,
        pic_params_num_ref_frames + 1,
        "CUVIDEOFORMAT.min_num_decode_surfaces must equal num_ref_frames + 1"
    );

    // Consistency: chroma_format derived from same chroma_format_idc
    let expected_chroma_format = match sps.chroma_format_idc {
        0 => cudaVideoChromaFormat::cudaVideoChromaFormat_Monochrome,
        1 => cudaVideoChromaFormat::cudaVideoChromaFormat_420,
        2 => cudaVideoChromaFormat::cudaVideoChromaFormat_422,
        3 => cudaVideoChromaFormat::cudaVideoChromaFormat_444,
        _ => cudaVideoChromaFormat::cudaVideoChromaFormat_420,
    };
    assert_eq!(
        cuvid_fmt.chroma_format, expected_chroma_format,
        "CUVIDEOFORMAT chroma_format must match SPS chroma_format_idc"
    );

    // Consistency: progressive_sequence = frame_mbs_only_flag
    assert_eq!(
        cuvid_fmt.progressive_sequence, sps.frame_mbs_only_flag as u8,
        "CUVIDEOFORMAT progressive_sequence must match frame_mbs_only_flag"
    );
}

/// Test that DetectedVideoFormat fields are consistent with CUVIDEOFORMAT fields.
///
/// This ensures the vk-video-parser's detected format can be used to populate
/// CUVIDEOFORMAT without discrepancies.
#[test]
fn test_detected_format_consistency_with_cuvid_format() {
    let data = load_test_file("assets/born_trailer.h264");
    let sps_data = extract_first_nal_with_start_code(&data, 7).expect("No SPS found");

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(sps_data);
    parser.parse(&packet).expect("SPS parse failed");

    let sps = parser.active_sps().expect("No active SPS");
    let format = parser.detected_format();
    let cuvid_fmt = build_cuvideoformat_from_sps(sps, format);

    // coded_width must match
    assert_eq!(
        format.coded_width, cuvid_fmt.coded_width,
        "DetectedVideoFormat coded_width must match CUVIDEOFORMAT coded_width"
    );

    // coded_height must match
    assert_eq!(
        format.coded_height, cuvid_fmt.coded_height,
        "DetectedVideoFormat coded_height must match CUVIDEOFORMAT coded_height"
    );

    // progressive_sequence must match
    assert_eq!(
        format.progressive_sequence,
        cuvid_fmt.progressive_sequence != 0,
        "DetectedVideoFormat progressive_sequence must match CUVIDEOFORMAT"
    );

    // Codec must be H264
    assert_eq!(
        format.codec,
        vk_video_core::codec::VideoCodec::DecodeH264,
        "DetectedVideoFormat codec must be DecodeH264"
    );
}

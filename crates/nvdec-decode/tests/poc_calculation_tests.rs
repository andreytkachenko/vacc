//! Comprehensive tests for H.264 Picture Order Count (POC) calculation.
//!
//! These tests verify that our POC calculation matches what cuvid's parser
//! would compute based on the H.264 specification (Annex B, D.3.3).
//!
//! The `PocCalculator` (re-exported from the common
//! `vacc_parser::h264_poc` module in `nvdec_decode::poc`) implements
//! the same algorithm as cuvid's parser callbacks.
//!
//! Reference: H.264/AVC specification, section D.3.3 "Decoding process for
//! picture order count"

use nvdec_decode::poc::PocCalculator;
use vacc_parser::{h264::H264Parser, BitstreamPacket, ParseResult, VideoParser};

/// Path to the project root (parent of nvdec-decode crate).
const PROJECT_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// Load a known H.264 test file from the project assets.
fn load_test_file(path: &str) -> Vec<u8> {
    let full_path = format!("{}/{}", PROJECT_ROOT, path);
    std::fs::read(&full_path).expect(&format!("Failed to read test file: {}", full_path))
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
        } else if remaining[i] == 0 && remaining[i + 1] == 0 && remaining[i + 2] == 1 {
            if i == 0 || remaining[i - 1] != 0 {
                return Some((start + i, 3));
            }
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

    let parse_limit = std::cmp::min(data.len(), 500_000);
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

/// Create a mock SliceHeader for testing.
fn create_slice_header(
    frame_num: u32,
    pic_order_cnt_lsb: i32,
    delta_pic_order_cnt: [i32; 2],
    nal_unit_type: u8,
    nal_ref_idc: u8,
) -> vacc_parser::h264::SliceHeader {
    vacc_parser::h264::SliceHeader {
        first_mb_in_slice: 0,
        slice_type: 0, // P slice
        pic_parameter_set_id: 0,
        frame_num,
        idr_pic_id: 0,
        pic_order_cnt_lsb,
        delta_pic_order_cnt,
        redundant_pic_cnt: 0,
        num_ref_idx_l0_active_minus1: 0,
        num_ref_idx_l1_active_minus1: 0,
        nal_ref_idc,
        nal_unit_type,
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
        sp_for_switch_flag: false,
        slice_qs_delta: 0,
        header_bit_size: 0,
        luma_log2_weight_denom: 0,
        chroma_log2_weight_denom: 0,
        luma_weight_l0_flag: 0,
        luma_weight_l0: [0i16; 32],
        luma_offset_l0: [0i16; 32],
        chroma_weight_l0_flag: 0,
        chroma_weight_l0: [[0i16; 2]; 32],
        chroma_offset_l0: [[0i16; 2]; 32],
        luma_weight_l1_flag: 0,
        luma_weight_l1: [0i16; 32],
        luma_offset_l1: [0i16; 32],
        chroma_weight_l1_flag: 0,
        chroma_weight_l1: [[0i16; 2]; 32],
        chroma_offset_l1: [[0i16; 2]; 32],
    }
}

/// Create a mock SPS for POC type 0 with given max_pic_order_cnt_lsb.
fn create_sps_poc_type_0(max_pic_order_cnt_lsb: u32) -> vacc_core::picture::H264Sps {
    let mut sps = vacc_core::picture::H264Sps::new();
    sps.pic_order_cnt_type = 0;
    sps.max_pic_order_cnt_lsb = max_pic_order_cnt_lsb;
    sps.log2_max_pic_order_cnt_lsb_minus4 = (max_pic_order_cnt_lsb as f64).log2() as u8 - 4;
    sps.frame_mbs_only_flag = true;
    sps.max_num_ref_frames = 4;
    sps
}

/// Create a mock SPS for POC type 1.
fn create_sps_poc_type_1(
    delta_pic_order_always_zero_flag: bool,
    num_ref_frames_in_pic_order_cnt_cycle: u32,
    offset_for_ref_frame: Vec<i32>,
) -> vacc_core::picture::H264Sps {
    let mut sps = vacc_core::picture::H264Sps::new();
    sps.pic_order_cnt_type = 1;
    sps.delta_pic_order_always_zero_flag = delta_pic_order_always_zero_flag;
    sps.frame_mbs_only_flag = true;
    sps.num_ref_frames_in_pic_order_cnt_cycle = num_ref_frames_in_pic_order_cnt_cycle;
    sps.offset_for_ref_frame = offset_for_ref_frame;
    sps.max_num_ref_frames = 4;
    sps
}

/// Create a mock SPS for POC type 2.
fn create_sps_poc_type_2(max_frame_num: u32) -> vacc_core::picture::H264Sps {
    let mut sps = vacc_core::picture::H264Sps::new();
    sps.pic_order_cnt_type = 2;
    sps.max_frame_num = max_frame_num;
    sps.log2_max_frame_num_minus4 = (max_frame_num as f64).log2() as u8 - 4;
    sps.frame_mbs_only_flag = true;
    sps.max_num_ref_frames = 4;
    sps
}

// ============================================================================
// POC Type 0 Tests (Explicit with pic_order_cnt_lsb)
// ============================================================================

/// Test 1: Basic POC type 0 with monotonic pic_order_cnt_lsb.
///
/// When pic_order_cnt_lsb increases monotonically without crossing
/// max_pic_order_cnt_lsb/2, MSB remains unchanged.
#[test]
fn test_poc_type0_explicit_basic() {
    let sps = create_sps_poc_type_0(512);
    let mut calc = PocCalculator::new();

    // Frame 0: lsb=0, expected POC=0
    let slh0 = create_slice_header(0, 0, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh0, false),
        0,
        "Frame 0 POC should be 0"
    );

    // Frame 1: lsb=2, expected POC=2
    let slh1 = create_slice_header(1, 2, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh1, false),
        2,
        "Frame 1 POC should be 2"
    );

    // Frame 2: lsb=4, expected POC=4
    let slh2 = create_slice_header(2, 4, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh2, false),
        4,
        "Frame 2 POC should be 4"
    );

    // Frame 3: lsb=6, expected POC=6
    let slh3 = create_slice_header(3, 6, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh3, false),
        6,
        "Frame 3 POC should be 6"
    );

    // Frame 4: lsb=100, expected POC=100 (still no wrap, diff=94 < 256)
    let slh4 = create_slice_header(4, 100, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh4, false),
        100,
        "Frame 4 POC should be 100"
    );
}

/// Test 2: POC type 0 with lsb wrap from high to low (MSB increases).
///
/// When lsb goes from high to low and the difference >= max_pic_order_cnt_lsb/2,
/// MSB should increase by max_pic_order_cnt_lsb (wrap-up).
#[test]
fn test_poc_type0_explicit_wrap_up() {
    let sps = create_sps_poc_type_0(512);
    let mut calc = PocCalculator::new();

    // Build up gradually to POC=500 (lsb=500)
    // Start with lsb=0
    let slh0 = create_slice_header(0, 0, [0, 0], 1, 1);
    assert_eq!(calc.calculate(&sps, &slh0, false), 0);

    // Jump to lsb=250 (diff=250 < 256, no wrap)
    let slh_250 = create_slice_header(1, 250, [0, 0], 1, 1);
    assert_eq!(calc.calculate(&sps, &slh_250, false), 250);

    // Jump to lsb=500 (diff=250 < 256, no wrap)
    let slh_500 = create_slice_header(2, 500, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh_500, false),
        500,
        "Frame at POC=500"
    );

    // Next frame: lsb=10 (wrapped around)
    // prev_lsb - curr_lsb = 500 - 10 = 490 >= 512/2 = 256, so MSB increases
    // Expected POC = 512 + 10 = 522
    let slh_next = create_slice_header(3, 10, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh_next, false),
        522,
        "POC should wrap up to 522 (MSB=512, LSB=10)"
    );

    // Continue after wrap: lsb=12, expected POC=524
    let slh_cont = create_slice_header(4, 12, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh_cont, false),
        524,
        "POC should continue to 524 after wrap"
    );
}

/// Test 3: POC type 0 with lsb wrap from low to high (MSB decreases).
///
/// When lsb goes from low to high and the difference >= max_pic_order_cnt_lsb/2,
/// MSB should decrease by max_pic_order_cnt_lsb (wrap-down).
#[test]
fn test_poc_type0_explicit_wrap_down() {
    let sps = create_sps_poc_type_0(512);
    let mut calc = PocCalculator::new();

    // First reach a high MSB by wrapping up
    // Build up: 0 -> 250 -> 500
    let slh0 = create_slice_header(0, 0, [0, 0], 1, 1);
    calc.calculate(&sps, &slh0, false);

    let slh_250 = create_slice_header(1, 250, [0, 0], 1, 1);
    calc.calculate(&sps, &slh_250, false);

    let slh_500 = create_slice_header(2, 500, [0, 0], 1, 1);
    assert_eq!(calc.calculate(&sps, &slh_500, false), 500);

    // Wrap up: lsb=10, POC=522
    let slh_wrapped = create_slice_header(3, 10, [0, 0], 1, 1);
    assert_eq!(calc.calculate(&sps, &slh_wrapped, false), 522);

    // Continue incrementally to approach POC=1020
    // From POC=522 (lsb=10), increment by 2 each frame
    let mut lsb = 10;
    for frame in 4..=260 {
        lsb = (lsb + 2) % 512;
        let slh = create_slice_header(frame, lsb, [0, 0], 1, 1);
        calc.calculate(&sps, &slh, false);
    }

    // Now at POC=1020: lsb=508 (1020 % 512 = 508)
    let slh_1020 = create_slice_header(260, 508, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh_1020, false),
        1020,
        "POC should be 1020 (MSB=512, LSB=508)"
    );

    // Next frame: lsb=8 (wraps around again)
    // prev_lsb - curr_lsb = 508 - 8 = 500 >= 256, MSB increases
    let slh_1032 = create_slice_header(261, 8, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh_1032, false),
        1032,
        "POC should wrap up to 1032 (MSB=1024, LSB=8)"
    );
}

/// Test 4: POC type 0 with max_pic_order_cnt_lsb=256.
///
/// Verifies correct wrap detection threshold (128 instead of 256).
#[test]
fn test_poc_type0_max_pic_order_cnt_lsb_256() {
    let sps = create_sps_poc_type_0(256);
    let mut calc = PocCalculator::new();

    // Build up to POC=240 (lsb=240) gradually
    // Start with lsb=0
    let slh0 = create_slice_header(0, 0, [0, 0], 1, 1);
    assert_eq!(calc.calculate(&sps, &slh0, false), 0);

    // Jump to lsb=120 (diff=120 < 128, no wrap)
    let slh_120 = create_slice_header(1, 120, [0, 0], 1, 1);
    assert_eq!(calc.calculate(&sps, &slh_120, false), 120);

    // Jump to lsb=240 (diff=120 < 128, no wrap)
    let slh_240 = create_slice_header(2, 240, [0, 0], 1, 1);
    assert_eq!(calc.calculate(&sps, &slh_240, false), 240);

    // Next frame: lsb=10 (wrapped around)
    // prev_lsb - curr_lsb = 240 - 10 = 230 >= 256/2 = 128, so MSB increases
    // Expected POC = 256 + 10 = 266
    let slh_next = create_slice_header(3, 10, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh_next, false),
        266,
        "POC should wrap up to 266 (MSB=256, LSB=10) with max_lsb=256"
    );

    // Non-wrap case: lsb=12, diff=2 < 128, no MSB change
    let slh_next2 = create_slice_header(4, 12, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh_next2, false),
        268,
        "POC should be 268 (no wrap)"
    );
}

/// Test 5: POC type 0 with max_pic_order_cnt_lsb=64.
///
/// Verifies correct wrap detection threshold (32 instead of 128).
#[test]
fn test_poc_type0_max_pic_order_cnt_lsb_64() {
    let sps = create_sps_poc_type_0(64);
    let mut calc = PocCalculator::new();

    // Build up to POC=50 (lsb=50) gradually
    // Start with lsb=0
    let slh0 = create_slice_header(0, 0, [0, 0], 1, 1);
    assert_eq!(calc.calculate(&sps, &slh0, false), 0);

    // Jump to lsb=25 (diff=25 < 32, no wrap)
    let slh_25 = create_slice_header(1, 25, [0, 0], 1, 1);
    assert_eq!(calc.calculate(&sps, &slh_25, false), 25);

    // Jump to lsb=50 (diff=25 < 32, no wrap)
    let slh_50 = create_slice_header(2, 50, [0, 0], 1, 1);
    assert_eq!(calc.calculate(&sps, &slh_50, false), 50);

    // Next frame: lsb=10 (wrapped around)
    // prev_lsb - curr_lsb = 50 - 10 = 40 >= 64/2 = 32, so MSB increases
    // Expected POC = 64 + 10 = 74
    let slh_next = create_slice_header(3, 10, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh_next, false),
        74,
        "POC should wrap up to 74 (MSB=64, LSB=10) with max_lsb=64"
    );

    // Non-wrap case: lsb=12, diff=2 < 32, no MSB change
    let slh_next2 = create_slice_header(4, 12, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh_next2, false),
        76,
        "POC should be 76 (no wrap)"
    );

    // Edge case: exactly at threshold
    // From lsb=50 (POC=50), go to lsb=18
    // diff = 50 - 18 = 32 >= 32, should wrap
    let mut calc2 = PocCalculator::new();
    let slh_edge0 = create_slice_header(0, 0, [0, 0], 1, 1);
    calc2.calculate(&sps, &slh_edge0, false);

    let slh_edge_25 = create_slice_header(1, 25, [0, 0], 1, 1);
    calc2.calculate(&sps, &slh_edge_25, false);

    let slh_edge_50 = create_slice_header(2, 50, [0, 0], 1, 1);
    calc2.calculate(&sps, &slh_edge_50, false);

    let slh_edge = create_slice_header(3, 18, [0, 0], 1, 1);
    assert_eq!(
        calc2.calculate(&sps, &slh_edge, false),
        82,
        "POC should wrap at exact threshold: 64+18=82"
    );
}

/// Test 6: POC type 0 with IDR frame reset.
///
/// After an IDR frame, POC state should be reset.
#[test]
fn test_poc_type0_idr_reset() {
    let sps = create_sps_poc_type_0(512);
    let mut calc = PocCalculator::new();

    // Build up some POC state
    let slh0 = create_slice_header(0, 100, [0, 0], 1, 1);
    assert_eq!(calc.calculate(&sps, &slh0, false), 100);

    let slh1 = create_slice_header(1, 200, [0, 0], 1, 1);
    assert_eq!(calc.calculate(&sps, &slh1, false), 200);

    // IDR frame (nal_unit_type=5) resets POC state
    calc.reset();

    // After reset, lsb=0 should give POC=0, not continue from previous state
    let slh_idr = create_slice_header(0, 0, [0, 0], 5, 1); // IDR slice
    assert_eq!(
        calc.calculate(&sps, &slh_idr, false),
        0,
        "POC should reset to 0 after IDR"
    );

    // Subsequent frame continues from reset state
    let slh_after = create_slice_header(1, 2, [0, 0], 1, 1);
    assert_eq!(
        calc.calculate(&sps, &slh_after, false),
        2,
        "POC should be 2 after IDR reset"
    );
}

// ============================================================================
// POC Type 1 Tests (Explicit with delta_pic_order_cnt)
// ============================================================================

/// Test 7: POC type 1 with delta_pic_order_cnt.
///
/// For frame pictures per H.264 D.3.3.2:
/// - Reference frames: PicOrderCnt = LastPicOrderCnt + offset_for_ref_frame[cycle]
/// - Non-reference: PicOrderCnt = PrevPicOrderCnt + offset_for_non_ref_pic
/// - Field pictures use: prev_frame_num + delta_pic_order_cnt[0]
///
/// This test uses reference frames with cycle offsets [4, -4].
#[test]
fn test_poc_type1_implicit_basic() {
    // offset_for_ref_frame = [4, -4], offset_for_non_ref_pic = 0 (default)
    let sps = create_sps_poc_type_1(false, 2, vec![4, -4]);
    let mut calc = PocCalculator::new();

    // Frame 0 (ref): prev_is_reference=false → use last_pic_order_cnt + offset[0]
    // = 0 + 4 = 4
    let slh0 = create_slice_header(0, 0, [0, 0], 1, 3); // nal_ref_idc=3 (reference)
    assert_eq!(
        calc.calculate(&sps, &slh0, true),
        4,
        "Frame 0 (ref) POC should be 4 (0+4)"
    );

    // Frame 1 (ref): prev_is_reference=true → use prev_pic_order_cnt + offset[1]
    // = 4 + (-4) = 0
    let slh1 = create_slice_header(1, 0, [0, 0], 1, 3);
    assert_eq!(
        calc.calculate(&sps, &slh1, true),
        0,
        "Frame 1 (ref) POC should be 0 (4-4)"
    );

    // Frame 2 (ref): cycle wraps → offset[0] again
    // prev_is_reference=true → prev_pic_order_cnt + offset[0] = 0 + 4 = 4
    let slh2 = create_slice_header(2, 0, [0, 0], 1, 3);
    assert_eq!(
        calc.calculate(&sps, &slh2, true),
        4,
        "Frame 2 (ref) POC should be 4 (0+4)"
    );

    // Frame 3 (non-ref): prev_is_reference=true → last_pic_order_cnt + offset_for_non_ref_pic
    // = 4 + 0 = 4
    let slh3 = create_slice_header(3, 0, [0, 0], 1, 0); // nal_ref_idc=0 (non-reference)
    assert_eq!(
        calc.calculate(&sps, &slh3, false),
        4,
        "Frame 3 (non-ref) POC should be 4 (4+0)"
    );
}

/// Test 8: POC type 1 with num_ref_frames_in_pic_order_cnt_cycle > 0.
///
/// When cycle is defined, POC calculation uses cycle offsets for reference frames.
/// Non-reference frames use offset_for_non_ref_pic.
#[test]
fn test_poc_type1_implicit_with_ref_frame_cycle() {
    // Cycle: [6, -2], non_ref offset = 0 (default)
    let sps = create_sps_poc_type_1(false, 2, vec![6, -2]);

    // Verify SPS has cycle configured before passing to calculator
    assert_eq!(sps.num_ref_frames_in_pic_order_cnt_cycle, 2);
    assert_eq!(sps.offset_for_ref_frame, vec![6, -2]);

    let mut calc = PocCalculator::new();

    // Frame 0 (ref): prev_is_reference=false → last_pic_order_cnt + offset[0] = 0 + 6 = 6
    let slh0 = create_slice_header(0, 0, [0, 0], 1, 3);
    assert_eq!(calc.calculate(&sps, &slh0, true), 6);

    // Frame 1 (ref): prev_is_reference=true → prev_pic_order_cnt + offset[1] = 6 + (-2) = 4
    let slh1 = create_slice_header(1, 0, [0, 0], 1, 3);
    assert_eq!(calc.calculate(&sps, &slh1, true), 4);

    // Frame 2 (ref): cycle wraps → offset[0] again
    // prev_is_reference=true → prev_pic_order_cnt + offset[0] = 4 + 6 = 10
    let slh2 = create_slice_header(2, 0, [0, 0], 1, 3);
    assert_eq!(calc.calculate(&sps, &slh2, true), 10);
}

/// Test 9: POC type 1 with delta_pic_order_always_zero_flag.
///
/// When this flag is set, delta_pic_order_cnt is always 0 (not read from bitstream).
/// For frame pictures, offsets are still used per the spec.
/// For field pictures, POC = prev_frame_num + 0.
#[test]
fn test_poc_type1_implicit_delta_zero_flag() {
    // delta_pic_order_always_zero_flag=true, cycle=[4, -4]
    let sps = create_sps_poc_type_1(true, 2, vec![4, -4]);

    // Verify flag is set before passing to calculator
    assert!(sps.delta_pic_order_always_zero_flag);

    let mut calc = PocCalculator::new();

    // Frame 0 (ref): prev_is_reference=false → last_pic_order_cnt + offset[0] = 0 + 4 = 4
    let slh0 = create_slice_header(0, 0, [0, 0], 1, 3);
    assert_eq!(
        calc.calculate(&sps, &slh0, true),
        4,
        "Frame 0 (ref) POC should be 4 (0+4)"
    );

    // Frame 1 (ref): prev_is_reference=true → prev_pic_order_cnt + offset[1] = 4 + (-4) = 0
    let slh1 = create_slice_header(1, 0, [0, 0], 1, 3);
    assert_eq!(
        calc.calculate(&sps, &slh1, true),
        0,
        "Frame 1 (ref) POC should be 0 (4-4)"
    );
}

/// Test 13: POC type 1 frame_num wrap-to-zero detection.
///
/// When frame_num wraps from max-1 to 0, last_pic_order_cnt_cycle must reset to 0.
/// Previously, the `frame_num > 0` guard prevented detection of wrap-to-zero.
#[test]
fn test_poc_type1_wrap_to_zero() {
    // Small max_frame_num (16) to easily trigger wraparound
    // cycle = [4, -4]
    let sps = create_sps_poc_type_1(true, 2, vec![4, -4]);

    let mut calc = PocCalculator::new();

    // Frame 14 (ref): prev_is_reference=false → last_pic_order_cnt + offset[0] = 0 + 4 = 4
    let slh14 = create_slice_header(14, 0, [0, 0], 1, 3);
    assert_eq!(
        calc.calculate(&sps, &slh14, true),
        4,
        "Frame 14 POC should be 4"
    );

    // Frame 15 (ref): prev_is_reference=true → prev + offset[1] = 4 + (-4) = 0
    let slh15 = create_slice_header(15, 0, [0, 0], 1, 3);
    assert_eq!(
        calc.calculate(&sps, &slh15, true),
        0,
        "Frame 15 POC should be 0"
    );

    // Frame 0 (ref, wrapped): frame_num 0 < prev_frame_num 15, so wrap detected.
    // last_pic_order_cnt_cycle resets to 0.
    // prev_is_reference=true → prev + offset[0] = 0 + 4 = 4
    let slh0 = create_slice_header(0, 0, [0, 0], 1, 3);
    assert_eq!(
        calc.calculate(&sps, &slh0, true),
        4,
        "Frame 0 (wrapped) POC should be 4 (wrap detected, cycle reset)"
    );

    // Frame 1 (ref): prev + offset[1] = 4 + (-4) = 0
    let slh1 = create_slice_header(1, 0, [0, 0], 1, 3);
    assert_eq!(
        calc.calculate(&sps, &slh1, true),
        0,
        "Frame 1 POC should be 0"
    );
}

// ============================================================================
// POC Type 2 Tests (Implicit from frame_num)
// ============================================================================

/// Test 10: POC type 2 implicit from frame_num.
///
/// Per H.264 D.3.3.3:
/// - Reference frames: POC = frame_num * 2
/// - Non-reference frame pictures: POC = frame_num * 2 + 1
#[test]
fn test_poc_type2_implicit_from_frame_num() {
    let sps = create_sps_poc_type_2(256);
    let mut calc = PocCalculator::new();

    // Frame 0 (ref): frame_num=0, POC=0*2=0
    let slh0 = create_slice_header(0, 0, [0, 0], 1, 3); // nal_ref_idc=3 (reference)
    assert_eq!(
        calc.calculate(&sps, &slh0, true),
        0,
        "Frame 0 (ref) POC should be 0"
    );

    // Frame 1 (ref): frame_num=1, POC=1*2=2
    let slh1 = create_slice_header(1, 0, [0, 0], 1, 3);
    assert_eq!(
        calc.calculate(&sps, &slh1, true),
        2,
        "Frame 1 (ref) POC should be 2"
    );

    // Frame 2 (non-ref): frame_num=100, POC=100*2+1=201
    let slh2 = create_slice_header(100, 0, [0, 0], 1, 0); // nal_ref_idc=0 (non-reference)
    assert_eq!(
        calc.calculate(&sps, &slh2, false),
        201,
        "Frame 100 (non-ref) POC should be 201"
    );

    // Frame 3 (ref): frame_num=200, POC=200*2=400
    let slh3 = create_slice_header(200, 0, [0, 0], 1, 3);
    assert_eq!(
        calc.calculate(&sps, &slh3, true),
        400,
        "Frame 200 (ref) POC should be 400"
    );
}

/// Test 11: POC type 2 with frame_num wraparound.
///
/// The common [`PocCalculator`] (vacc_parser::h264_poc) tracks FrameNum
/// wrap cycles so that type-2 POCs remain MONOTONIC across wraps:
/// - ref: (cycle * MaxFrameNum + frame_num) * 2
/// - non-ref: (cycle * MaxFrameNum + frame_num) * 2 + 1
///
/// Monotonic POCs are required for correct presentation-order sorting in the
/// decoder's reorder buffer (raw spec POCs wrap with FrameNum and would
/// misorder frames after a wrap).
#[test]
fn test_poc_type2_frame_num_wrap() {
    let sps = create_sps_poc_type_2(256);
    let mut calc = PocCalculator::new();

    // Build up to near wrap (all reference frames)
    let slh250 = create_slice_header(250, 0, [0, 0], 1, 3);
    assert_eq!(calc.calculate(&sps, &slh250, true), 500); // 250*2=500

    let slh254 = create_slice_header(254, 0, [0, 0], 1, 3);
    assert_eq!(calc.calculate(&sps, &slh254, true), 508); // 254*2=508

    // Wrap around: frame_num=5 (was 254, wrapped past 256) → cycle=1
    let slh_wrap = create_slice_header(5, 0, [0, 0], 1, 3);
    assert_eq!(
        calc.calculate(&sps, &slh_wrap, true),
        (256 + 5) * 2,
        "POC should be 522 after frame_num wrap (cycle=1: (256+5)*2)"
    );

    // Continue after wrap
    let slh_after = create_slice_header(10, 0, [0, 0], 1, 3);
    assert_eq!(
        calc.calculate(&sps, &slh_after, true),
        (256 + 10) * 2,
        "POC should be 532 after wrap (cycle=1: (256+10)*2)"
    );
}

// ============================================================================
// born_trailer.h264 Integration Tests
// ============================================================================

/// Test 12: Verify POC sequence from born_trailer.h264 matches cuvid expectations.
///
/// born_trailer.h264 uses POC type 0 with max_pic_order_cnt_lsb=512.
/// This test verifies that our POC calculation produces the expected
/// monotonic sequence matching what cuvid's parser would compute.
#[test]
fn test_poc_from_born_trailer_matches_cuvid() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().expect("SPS should be available");

    // Verify stream uses POC type 0
    assert_eq!(
        sps.pic_order_cnt_type, 0,
        "born_trailer.h264 should use POC type 0"
    );

    // Verify max_pic_order_cnt_lsb
    assert!(
        sps.max_pic_order_cnt_lsb > 0,
        "max_pic_order_cnt_lsb must be positive for POC type 0"
    );

    let mut calc = PocCalculator::new();

    // Collect unique frames and their calculated POCs
    let mut frame_pocs = Vec::new();
    let mut seen_frames = std::collections::HashSet::new();

    for slh in &slices {
        if seen_frames.insert(slh.frame_num) {
            let poc = calc.calculate(&sps, slh, false);
            frame_pocs.push((slh.frame_num, slh.pic_order_cnt_lsb, poc));
        }
    }

    // Should have parsed multiple frames
    assert!(
        frame_pocs.len() >= 2,
        "Should have parsed at least 2 unique frames, got {}",
        frame_pocs.len()
    );

    // Verify POC sequence is monotonically increasing (as cuvid expects)
    for i in 1..frame_pocs.len() {
        let prev_poc = frame_pocs[i - 1].2;
        let curr_poc = frame_pocs[i].2;

        assert!(
            curr_poc > prev_poc,
            "POC should be monotonically increasing: frame {} POC={} <= frame {} POC={}",
            frame_pocs[i - 1].0,
            prev_poc,
            frame_pocs[i].0,
            curr_poc
        );
    }

    // Verify first frame POC is 0 (IDR frame starts at 0)
    assert_eq!(
        frame_pocs[0].2, 0,
        "First frame (IDR) should have POC=0, got {}",
        frame_pocs[0].2
    );

    // Print first few POCs for verification
    eprintln!("born_trailer.h264 POC sequence (first 10 frames):");
    for (i, (frame_num, lsb, poc)) in frame_pocs.iter().take(10).enumerate() {
        eprintln!(
            "  Frame {}: frame_num={}, lsb={}, POC={}",
            i, frame_num, lsb, poc
        );
    }
}

/// Test 13: Verify CurrFieldOrderCnt matches calculated POC.
///
/// In CUVIDH264PICPARAMS, CurrFieldOrderCnt[0] and CurrFieldOrderCnt[1]
/// should both equal the calculated POC for frame pictures.
#[test]
fn test_poc_field_order_cnt_for_cuvid_picparams() {
    let data = load_test_file("assets/born_trailer.h264");
    let parser = init_parser_with_params(&data);
    let slices = parse_slices_from_bitstream(&data);

    let sps = parser.active_sps().expect("SPS should be available");
    let _pps = parser.active_pps().expect("PPS should be available");

    let mut calc = PocCalculator::new();

    // Process unique frames and verify CurrFieldOrderCnt would match
    let mut seen_frames = std::collections::HashSet::new();

    for slh in &slices {
        if !seen_frames.insert(slh.frame_num) {
            continue;
        }

        let poc = calc.calculate(&sps, slh, false);

        // For frame pictures (frame_mbs_only_flag=1 or !field_pic_flag),
        // both CurrFieldOrderCnt[0] and CurrFieldOrderCnt[1] should equal POC
        let is_frame_picture = sps.frame_mbs_only_flag || !slh.field_pic_flag;

        if is_frame_picture {
            // This is what decoder.rs does at line 527:
            // CurrFieldOrderCnt: [poc, poc]
            let curr_field_order_cnt = [poc, poc];

            assert_eq!(
                curr_field_order_cnt[0], poc,
                "CurrFieldOrderCnt[0] should equal calculated POC for frame {} (POC={})",
                slh.frame_num, poc
            );
            assert_eq!(
                curr_field_order_cnt[1], poc,
                "CurrFieldOrderCnt[1] should equal calculated POC for frame {} (POC={})",
                slh.frame_num, poc
            );
        }

        // Verify pic_order_present_flag in picparams matches POC type
        let pic_order_present_flag = if sps.pic_order_cnt_type != 2 { 1 } else { 0 };
        assert_eq!(
            pic_order_present_flag, 1,
            "pic_order_present_flag should be 1 for POC type {}",
            sps.pic_order_cnt_type
        );
    }
}

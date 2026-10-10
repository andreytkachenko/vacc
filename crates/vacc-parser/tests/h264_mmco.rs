//! H.264 MMCO parser tests with synthetic slice headers (spec 7.4.5).
//!
//! The in-tree samples only use MMCO op 1, so the long-term reference ops
//! (2-6) are exercised here with hand-built bitstreams. They reuse the real
//! SPS/PPS from `assets/samples/h264_main.h264` (same constants as
//! h264_slice_inline.rs): poc_type=0 (lsb u6), log2_max_frame_num_minus4=0
//! (frame_num u4), frame-only, CABAC, adaptive_ref_pic_marking_mode_flag=1.
//!
//! Field order follows the parser's slice-header model (verified against
//! x264 real data in h264_slice_inline.rs): first_mb ue, slice_type ue,
//! pps_id ue, frame_num u4, poc_lsb u6, ref-override flag, rplm-L0 flag,
//! pred_weight_table (this PPS has weighted_pred_flag=1: two log2 denoms +
//! per-ref luma/chroma flag bits, all zero here), adaptive flag, MMCO ops +
//! ue(0) terminator, cabac_init_idc ue (P+CABAC), slice_qp_delta se,
//! deblock idc ue.

use vacc_core::codec::VideoCodec;
use vacc_parser::h264::H264Parser;
use vacc_parser::{BitstreamPacket, DetectedVideoFormat, ParseResult, SliceHeader, VideoParser};

const SPS_NAL: &[u8] = &[
    0x67, 0x4d, 0x40, 0x1e, 0xec, 0xa0, 0x50, 0x17, 0xfc, 0xb8, 0x08, 0x80, 0x00, 0x00, 0x03, 0x00,
    0x80, 0x00, 0x00, 0x1e, 0x07, 0x8b, 0x16, 0xcb, 0x00,
];
const PPS_NAL: &[u8] = &[0x68, 0xeb, 0xe3, 0xcb, 0x20];

/// Exp-Golomb ue(v) bit pattern (MSB first): z = floor(log2(v+1)) zeros, then
/// (v+1) in z+1 bits. ue(0)=`1`, ue(1)=`010`, ue(2)=`011`, ue(3)=`00100`.
fn ue_bits(v: u32) -> Vec<bool> {
    if v == 0 {
        return vec![true];
    }
    let k = 31 - (v + 1).leading_zeros(); // floor(log2(v+1))
    let mut out = vec![false; k as usize];
    out.push(true);
    let suffix = v - ((1u32 << k) - 1); // low k bits of (v+1)
    for i in (0..k).rev() {
        out.push((suffix >> i) & 1 == 1);
    }
    out
}

fn se_bits(v: i32) -> Vec<bool> {
    let ue_val = if v <= 0 { (-2 * v) as u32 } else { (2 * v - 1) as u32 };
    ue_bits(ue_val)
}

/// Build a P-slice NAL (nal_ref_idc=2) with the given MMCO ops. Each op is
/// `(op, value, value2)` where value2 is only used by op 3.
fn build_p_slice_nal(frame_num: u32, poc_lsb: u32, mmco: &[(u32, u32, u32)]) -> Vec<u8> {
    let mut bits: Vec<bool> = Vec::new();
    bits.extend(ue_bits(0)); // first_mb_in_slice = 0
    bits.extend(ue_bits(0)); // slice_type = P (0)
    bits.extend(ue_bits(0)); // pps_id = 0
    for i in (0..4).rev() {
        bits.push(((frame_num >> i) & 1) == 1);
    } // frame_num u(4)
    for i in (0..6).rev() {
        bits.push(((poc_lsb >> i) & 1) == 1);
    } // pic_order_cnt_lsb u(6)
    bits.push(false); // num_ref_idx_active_override_flag = 0
    bits.push(false); // ref_pic_list_modification_flag_l0 = 0 (no ops)
    // pred_weight_table (PPS weighted_pred_flag=1): all-zero weights.
    bits.extend(ue_bits(4)); // luma_log2_weight_denom
    bits.extend(ue_bits(4)); // chroma_log2_weight_denom
    for _ in 0..3 { // num_ref_idx_l0_active_minus1 = 2 (PPS default)
        bits.push(false); // luma_weight_l0_flag[i]
        bits.push(false); // chroma_weight_l0_flag[i]
    }
    bits.push(true); // adaptive_ref_pic_marking_mode_flag = 1
    for &(op, value, value2) in mmco {
        bits.extend(ue_bits(op));
        match op {
            1 | 2 | 4 | 5 => bits.extend(ue_bits(value)),
            3 => {
                bits.extend(ue_bits(value));
                bits.extend(ue_bits(value2));
            }
            _ => {} // op 6: no parameters
        }
    }
    bits.extend(ue_bits(0)); // MMCO terminator
    bits.extend(ue_bits(0)); // cabac_init_idc (parser reads it for P + CABAC)
    bits.extend(se_bits(0)); // slice_qp_delta = 0
    bits.extend(ue_bits(1)); // disable_deblocking_filter_idc = 1 (no offsets)
    bits.push(true); // rbsp_stop_one_bit

    // MSB-first packing: pad to a full byte so the last partial chunk lands in
    // the high bits (H.264 rbsp trailing bits are zero).
    while bits.len() % 8 != 0 {
        bits.push(false);
    }
    let mut nal = vec![0x41u8]; // nal_ref_idc=2, nal_unit_type=1
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for b in chunk {
            byte = (byte << 1) | *b as u8;
        }
        nal.push(byte);
    }
    nal
}

/// Parse SPS+PPS plus one synthetic slice NAL; return its H.264 slice header.
fn parse_synthetic(nal: &[u8]) -> vacc_parser::h264::SliceHeader {
    let mut payload = Vec::new();
    for n in [SPS_NAL, PPS_NAL, nal] {
        payload.extend_from_slice(&[0x00, 0x00, 0x01]);
        payload.extend_from_slice(n);
    }
    let pkt = BitstreamPacket::new(payload);
    let mut parser = H264Parser::new();
    parser.init(&DetectedVideoFormat::new(VideoCodec::DecodeH264)).expect("init");
    loop {
        match parser.parse(&pkt).expect("parse failed") {
            ParseResult::ParameterSet { .. } => {}
            ParseResult::Slice { slices, .. } => {
                let header = slices[0]
                    .slice_header
                    .clone()
                    .expect("slice header must parse");
                match header {
                    SliceHeader::H264(slh) => return slh,
                    other => panic!("unexpected header variant: {other:?}"),
                }
            }
            ParseResult::Nothing | ParseResult::EndOfStream => break,
        }
    }
    panic!("slice NAL was never parsed")
}

/// Op 5 reads max_long_term_frame_idx_plus1 and op 6 has no parameters: the
/// classic off-by-one table desynced the bit reader here. The trailing fields
/// (frame_num / poc_lsb on a second parse of the same layout) must stay
/// aligned for the whole header to decode.
#[test]
fn mmco_ops_5_and_6_keep_bit_sync() {
    let nal = build_p_slice_nal(7, 20, &[(5, 2, 0), (6, 0, 0)]);
    let slh = parse_synthetic(&nal);

    assert_eq!(slh.frame_num, 7, "frame_num misaligned -> MMCO param count wrong");
    assert_eq!(slh.pic_order_cnt_lsb, 20);
    let ops: Vec<(u32, u32, u32)> = slh
        .dec_ref_pic_marking
        .iter()
        .map(|e| (e.memory_management_control_operation, e.value, e.value2))
        .collect();
    assert_eq!(ops, vec![(5, 2, 0), (6, 0, 0)]);
}

/// Op 3 carries two values: difference_of_pic_nums_minus1 AND long_term_pic_num.
#[test]
fn mmco_op3_stores_both_values() {
    let nal = build_p_slice_nal(3, 8, &[(3, 1, 4)]);
    let slh = parse_synthetic(&nal);

    assert_eq!(slh.frame_num, 3);
    let ops: Vec<(u32, u32, u32)> = slh
        .dec_ref_pic_marking
        .iter()
        .map(|e| (e.memory_management_control_operation, e.value, e.value2))
        .collect();
    assert_eq!(ops, vec![(3, 1, 4)], "op 3 must keep diff in value and lt idx in value2");
}

/// Mixed op sequence: mark current as LT (4), unmark LT (2), unmark short (1).
#[test]
fn mmco_mixed_ops_4_2_1() {
    let nal = build_p_slice_nal(11, 44, &[(4, 2, 0), (2, 2, 0), (1, 3, 0)]);
    let slh = parse_synthetic(&nal);

    assert_eq!(slh.frame_num, 11);
    assert_eq!(slh.pic_order_cnt_lsb, 44);
    let ops: Vec<(u32, u32, u32)> = slh
        .dec_ref_pic_marking
        .iter()
        .map(|e| (e.memory_management_control_operation, e.value, e.value2))
        .collect();
    assert_eq!(ops, vec![(4, 2, 0), (2, 2, 0), (1, 3, 0)]);
}

//! Corrupted parameter-set NALs must surface as `Err`, not be silently
//! skipped: skipping leaves stale (or absent) parameter sets active, so
//! either subsequent slices misdecode against the wrong sets or fail with
//! an opaque error one frame later. A complete NAL (followed by a start
//! code or payload end without TRUNCATED_TAIL) that fails to parse is
//! corrupt by definition — incomplete tail NALs are held back separately.

use vacc_core::codec::VideoCodec;
use vacc_parser::h264::H264Parser;
use vacc_parser::h265::H265Parser;
use vacc_parser::{BitstreamPacket, DetectedVideoFormat, VideoParser};

fn one_nal(header: u8, rbsp_len: usize) -> BitstreamPacket {
    let mut payload = vec![0x00, 0x00, 0x01, header];
    payload.extend(std::iter::repeat(0u8).take(rbsp_len));
    BitstreamPacket::new(payload)
}

/// An all-zero RBSP cannot carry a valid SPS/PPS/VPS: the first ue(v) field
/// runs past the 32-leading-zero limit (or the trailing-bit check fails), so
/// parsing must error.
#[test]
fn h264_corrupt_sps_is_error() {
    let pkt = one_nal(0x67, 24); // SPS
    let mut parser = H264Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeH264))
        .expect("init");
    match parser.parse(&pkt) {
        Err(e) => eprintln!("expected error: {e}"),
        Ok(r) => panic!("corrupt SPS was silently skipped: {r:?}"),
    }
}

#[test]
fn h264_corrupt_pps_is_error() {
    let pkt = one_nal(0x68, 5); // PPS
    let mut parser = H264Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeH264))
        .expect("init");
    match parser.parse(&pkt) {
        Err(e) => eprintln!("expected error: {e}"),
        Ok(r) => panic!("corrupt PPS was silently skipped: {r:?}"),
    }
}

#[test]
fn h265_corrupt_sps_is_error() {
    let pkt = one_nal(0x42, 32); // SPS (nal_unit_type 33)
    let mut parser = H265Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeH265))
        .expect("init");
    match parser.parse(&pkt) {
        Err(e) => eprintln!("expected error: {e}"),
        Ok(r) => panic!("corrupt SPS was silently skipped: {r:?}"),
    }
}

#[test]
fn h265_corrupt_pps_is_error() {
    let pkt = one_nal(0x44, 8); // PPS (nal_unit_type 34)
    let mut parser = H265Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeH265))
        .expect("init");
    match parser.parse(&pkt) {
        Err(e) => eprintln!("expected error: {e}"),
        Ok(r) => panic!("corrupt PPS was silently skipped: {r:?}"),
    }
}

/// Valid SPS+PPS must still parse (guard against over-strict rejection).
#[test]
fn h264_valid_sps_pps_still_parse() {
    let sps: &[u8] = &[
        0x67, 0x4d, 0x40, 0x1e, 0xec, 0xa0, 0x50, 0x17, 0xfc, 0xb8, 0x08, 0x80, 0x00, 0x00, 0x03,
        0x00, 0x80, 0x00, 0x00, 0x1e, 0x07, 0x8b, 0x16, 0xcb, 0x00,
    ];
    let pps: &[u8] = &[0x68, 0xeb, 0xe3, 0xcb, 0x20];
    let mut payload = Vec::new();
    for n in [sps, pps] {
        payload.extend_from_slice(&[0x00, 0x00, 0x01]);
        payload.extend_from_slice(n);
    }
    let pkt = BitstreamPacket::new(payload);
    let mut parser = H264Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeH264))
        .expect("init");
    match parser.parse(&pkt).expect("valid parameter sets must parse") {
        vacc_parser::ParseResult::ParameterSet { sps, pps, .. } => {
            assert!(sps.is_some() && pps.is_some());
        }
        other => panic!("unexpected result: {other:?}"),
    }
}

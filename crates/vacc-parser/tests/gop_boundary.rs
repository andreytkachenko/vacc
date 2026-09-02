//! Regression test: the parser must not drop the last picture of a GOP
//! when a single parse() call spans a GOP boundary
//! ([last slice][SPS][PPS][IDR of next GOP]).
//!
//! bframe_test.h264 contains 10 GOPs of 14 pictures each (10 IDRs +
//! 140 non-IDR single-slice pictures = 150 pictures total), with SPS/PPS
//! repeated before every IDR.
use vacc_core::codec::VideoCodec;
use vacc_parser::h264::H264Parser;
use vacc_parser::{BitstreamPacket, DetectedVideoFormat, ParseResult, VideoParser};

#[test]
fn gop_boundary_keeps_last_picture() {
    // Embedded at compile time: no runtime dependency on the assets tree.
    let data = include_bytes!("../../../assets/bframe_test.h264").to_vec();
    let mut parser = H264Parser::new();
    let format = DetectedVideoFormat::new(VideoCodec::DecodeH264);
    parser.init(&format).expect("init");

    // Same pattern as NvdecDecoder::parse_and_decode: one packet, loop parse().
    let packet = BitstreamPacket::new(data);
    let mut pictures = 0usize;
    let mut slices = 0usize;
    let mut param_sets = 0usize;
    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::Slice { slices: sl, .. }) => {
                pictures += 1;
                slices += sl.len();
            }
            Ok(ParseResult::ParameterSet { .. }) => param_sets += 1,
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Err(e) => panic!("parse error: {e}"),
        }
    }
    assert_eq!(slices, 150, "parser dropped pictures at GOP boundaries");
    assert_eq!(pictures, 150);
    assert_eq!(param_sets, 10);
}

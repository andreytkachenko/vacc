//! H.264 parser tests with real bitstream data from `assets/samples/h264_main.h264`
//! (640x368, Main profile, poc_type=0, MaxPicOrderCntLsb=64, 300 frames).
//!
//! Ground truth: the dumper output for this sample, cross-verified against an
//! independent Python oracle implementing H.264 spec D.3.3 (POC reconstruction)
//! and 8.2.3.1/8.2.3.2 (reference list initialization/modification) - 0
//! mismatches on all three H.264 samples. Slice NAL constants below are the
//! first 1 + ceil(header_bit_size/8) bytes of each real NAL (1-byte NAL header
//! + the complete coded slice header), so they parse deterministically.
//!
//! The stream contains B-frame reordering, ref_pic_lists_modification, MMCO,
//! and a POC wraparound at PIC 32 (lsb 60 -> 6, POC 60 -> 70).

use vacc_core::codec::VideoCodec;
use vacc_core::picture::H264Sps;
use vacc_parser::h264::H264Parser;
use vacc_parser::h264::SliceHeader as H264Slh;
use vacc_parser::h264_poc::PocCalculator;
use vacc_parser::{BitstreamPacket, DetectedVideoFormat, ParseResult, SliceHeader, VideoParser};

// SPS/PPS NAL units from the sample (first occurrence).
const SPS_NAL: &[u8] = &[
    0x67, 0x4d, 0x40, 0x1e, 0xec, 0xa0, 0x50, 0x17, 0xfc, 0xb8, 0x08, 0x80, 0x00, 0x00, 0x03, 0x00,
    0x80, 0x00, 0x00, 0x1e, 0x07, 0x8b, 0x16, 0xcb, 0x00,
];
const PPS_NAL: &[u8] = &[0x68, 0xeb, 0xe3, 0xcb, 0x20];

// Real slice NAL units (complete coded slice header), decode order.
/// PIC0: fn=0 poc=0 slt=2 lsb=0 nal_ref=3
const SLICE_IDR: &[u8] = &[0x65, 0x88, 0x84, 0x03, 0xff];
/// PIC1: fn=1 poc=4 slt=0 lsb=4 nal_ref=2
const SLICE_P1: &[u8] = &[0x41, 0x9a, 0x22, 0x6c, 0x7f];
/// PIC2: fn=2 poc=2 slt=1 lsb=2 nal_ref=0
const SLICE_B2: &[u8] = &[0x01, 0x9e, 0x41, 0x79, 0x19, 0xff];
/// PIC3: fn=2 poc=6 slt=0 lsb=6 nal_ref=2 rplm=[0:0:0,0:0:15,0:0:0]
const SLICE_P3_RPLM: &[u8] = &[0x41, 0x9a, 0x43, 0x3c, 0x21, 0x93, 0x29, 0x87, 0xff];
/// PIC4: fn=3 poc=14 slt=0 lsb=14 nal_ref=2 rplm=[0:0:0,0:0:15,0:0:0,0:0:0]
const SLICE_P4: &[u8] = &[0x41, 0x9a, 0x67, 0x49, 0xe1, 0x0f, 0x26, 0x53, 0x03, 0xff];
/// PIC5: fn=4 poc=10 slt=1 lsb=10 nal_ref=2 mmco=[1:4,1:1]
const SLICE_B5_MMCO: &[u8] = &[0x41, 0x9e, 0x85, 0x45, 0x11, 0x3c, 0x47];
/// PIC29: fn=0 poc=58 slt=1 lsb=58 nal_ref=2 mmco=[1:4,1:1]
const SLICE_W29: &[u8] = &[0x41, 0x9e, 0x1d, 0x45, 0x15, 0x2c, 0xdf];
/// PIC30: fn=1 poc=56 slt=1 lsb=56 nal_ref=0
const SLICE_W30: &[u8] = &[0x01, 0x9e, 0x3c, 0x74, 0x45, 0x7f];
/// PIC31: fn=1 poc=60 slt=1 lsb=60 nal_ref=0
const SLICE_W31: &[u8] = &[0x01, 0x9e, 0x3e, 0x6a, 0x45, 0x7f];
/// PIC32: fn=1 poc=70 slt=0 lsb=6 nal_ref=2 rplm=[0:0:1,0:0:15,1:0:0,0:0:2] (POC wrap)
const SLICE_W32_WRAP: &[u8] = &[
    0x41, 0x9a, 0x23, 0x49, 0xa8, 0x41, 0x6c, 0x99, 0x4c, 0x0b, 0xff,
];
/// PIC33: fn=2 poc=66 slt=1 lsb=2 nal_ref=2 mmco=[1:4,1:1]
const SLICE_W33: &[u8] = &[0x41, 0x9e, 0x41, 0x45, 0x15, 0x2c, 0xdf];
/// PIC34: fn=3 poc=64 slt=1 lsb=0 nal_ref=0
const SLICE_W34: &[u8] = &[0x01, 0x9e, 0x60, 0x74, 0x45, 0x7f];
/// PIC35: fn=3 poc=68 slt=1 lsb=4 nal_ref=0
const SLICE_W35: &[u8] = &[0x01, 0x9e, 0x62, 0x6a, 0x45, 0x7f];

/// Frame the given NALs with 3-byte start codes into one packet.
fn packet(nals: &[&[u8]]) -> BitstreamPacket {
    let mut payload = Vec::new();
    for nal in nals {
        payload.extend_from_slice(&[0x00, 0x00, 0x01]);
        payload.extend_from_slice(nal);
    }
    BitstreamPacket::new(payload)
}

/// Parse one packet: expect the initial ParameterSet (SPS+PPS), then loop
/// parse() until Nothing, returning each picture's H.264 slice header in
/// order. Panics if a slice header fails to parse (a None header would
/// silently vacate the assertions).
fn parse_headers(parser: &mut H264Parser, nals: &[&[u8]]) -> Vec<H264Slh> {
    let pkt = packet(nals);
    let mut sps_seen = false;
    let mut out = Vec::new();
    loop {
        match parser.parse(&pkt).expect("parse failed") {
            ParseResult::ParameterSet { sps, pps, .. } => {
                assert!(sps.is_some() && pps.is_some(), "SPS and PPS expected");
                sps_seen = true;
            }
            ParseResult::Slice { slices, .. } => {
                for s in &slices {
                    let header = s
                        .slice_header
                        .clone()
                        .expect("real NAL data must yield a parsed slice header");
                    match header {
                        SliceHeader::H264(slh) => out.push(slh),
                        other => panic!("unexpected slice header variant: {other:?}"),
                    }
                }
            }
            ParseResult::Nothing | ParseResult::EndOfStream => break,
        }
    }
    assert!(sps_seen, "parameter set was never returned");
    out
}

/// Run the POC calculator over parsed headers exactly like the dumper does:
/// reset on IDR (nal_unit_type 5), is_reference = nal_ref_idc != 0.
fn pocs(sps: &H264Sps, headers: &[H264Slh]) -> Vec<i32> {
    let mut calc = PocCalculator::new();
    headers
        .iter()
        .map(|slh| {
            let is_ref = slh.nal_ref_idc != 0;
            if slh.nal_unit_type == 5 {
                calc.reset();
            }
            calc.calculate(sps, slh, is_ref)
        })
        .collect()
}

/// Extract the SPS from an initial ParameterSet parse.
fn sps_from(parser: &mut H264Parser, nals: &[&[u8]]) -> H264Sps {
    let pkt = packet(nals);
    match parser.parse(&pkt).expect("parse failed") {
        ParseResult::ParameterSet { sps, .. } => {
            let sps = sps.expect("SPS expected");
            sps.downcast_ref::<H264Sps>().cloned().unwrap()
        }
        other => panic!("expected ParameterSet, got {other:?}"),
    }
}

/// SPS/PPS fields must match the sample's known values (dump-verified).
#[test]
fn test_sps_pps_fields() {
    let mut parser = H264Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeH264))
        .expect("init");
    let sps = sps_from(&mut parser, &[SPS_NAL, PPS_NAL]);
    assert_eq!(sps.profile_idc, 77, "Main profile");
    assert_eq!(sps.level_idc, 30);
    assert_eq!(sps.chroma_format_idc, 1);
    assert_eq!(sps.bit_depth_luma_minus8, 0);
    assert_eq!(sps.log2_max_frame_num_minus4, 0);
    assert_eq!(sps.pic_order_cnt_type, 0);
    assert_eq!(sps.max_pic_order_cnt_lsb, 64);
    assert_eq!(sps.max_num_ref_frames, 4);
    assert_eq!((sps.pic_width_in_mbs_minus1 as u32 + 1) * 16, 640);
    assert_eq!((sps.pic_height_in_map_units_minus1 as u32 + 1) * 16, 368);

    let pps = parser.active_pps().expect("PPS expected");
    assert_eq!(pps.pic_parameter_set_id, 0);
    assert_eq!(pps.seq_parameter_set_id, 0);
    assert!(pps.entropy_coding_mode_flag, "CABAC");
    assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 2);
    assert_eq!(pps.num_ref_idx_l1_default_active_minus1, 0);
}

/// Per-picture slice header fields for the first six decode-order pictures
/// (IDR, P, B, P with ref_pic_lists_modification, P, B with MMCO).
#[test]
fn test_slice_headers_real_data() {
    let mut parser = H264Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeH264))
        .expect("init");
    let headers = parse_headers(
        &mut parser,
        &[
            SPS_NAL,
            PPS_NAL,
            SLICE_IDR,
            SLICE_P1,
            SLICE_B2,
            SLICE_P3_RPLM,
            SLICE_P4,
            SLICE_B5_MMCO,
        ],
    );
    assert_eq!(headers.len(), 6);

    let (idr, p1, b2, p3, p4, b5) = (
        &headers[0],
        &headers[1],
        &headers[2],
        &headers[3],
        &headers[4],
        &headers[5],
    );

    // PIC0: IDR frame.
    assert_eq!(idr.nal_unit_type, 5, "IDR slice NAL");
    assert_eq!(idr.frame_num, 0);
    assert_eq!(idr.slice_type, 2, "I slice");
    assert_eq!(idr.pic_parameter_set_id, 0);
    assert_eq!(idr.pic_order_cnt_lsb, 0);
    assert_eq!(idr.nal_ref_idc, 3);
    assert_eq!(idr.header_bit_size, 26);

    // PIC1: P frame.
    assert_eq!(p1.nal_unit_type, 1);
    assert_eq!(p1.frame_num, 1);
    assert_eq!(p1.slice_type, 0);
    assert_eq!(p1.pic_order_cnt_lsb, 4);
    assert_eq!(p1.nal_ref_idc, 2);
    assert_eq!(p1.header_bit_size, 30);

    // PIC2: B frame (non-reference NAL: nal_ref_idc=0, still NonIdrSlice type 1).
    assert_eq!(b2.nal_unit_type, 1);
    assert_eq!(b2.frame_num, 2);
    assert_eq!(b2.slice_type, 1);
    assert_eq!(b2.pic_order_cnt_lsb, 2);
    assert_eq!(b2.nal_ref_idc, 0);

    // PIC3: P frame with ref_pic_lists_modification (3 L0 entries).
    assert_eq!(p3.frame_num, 2);
    assert_eq!(p3.slice_type, 0);
    assert_eq!(p3.pic_order_cnt_lsb, 6);
    assert_eq!(p3.num_ref_idx_l0_active_minus1, 2);
    assert_eq!(
        p3.ref_pic_list_modification_l0
            .iter()
            .map(|e| (e.op, e.difference))
            .collect::<Vec<_>>(),
        vec![(0u32, 0i32), (0, 15), (0, 0)],
    );

    // PIC4: P frame overriding nr0 to 3, with 4 L0 modifications.
    assert_eq!(p4.num_ref_idx_l0_active_minus1, 3);
    assert!(p4.num_ref_idx_active_override_flag);
    assert_eq!(
        p4.ref_pic_list_modification_l0
            .iter()
            .map(|e| (e.op, e.difference))
            .collect::<Vec<_>>(),
        vec![(0u32, 0i32), (0, 15), (0, 0), (0, 0)],
    );

    // PIC5: B frame with two MMCO ops (1:3, 1:2).
    assert_eq!(b5.slice_type, 1);
    assert_eq!(b5.pic_order_cnt_lsb, 10);
    let mmco: Vec<(u32, u32)> = b5
        .dec_ref_pic_marking
        .iter()
        .map(|e| (e.memory_management_control_operation, e.value))
        .collect();
    assert_eq!(mmco, vec![(1, 3), (1, 2)]);
}

/// POC reconstruction across the MaxPicOrderCntLsb=64 boundary: decode-order
/// PICs 29-35 carry lsb 58,56,60,6,2,0,4 and must reconstruct to POC
/// 58,56,60,70,66,64,68 (the wrap at PIC32: lsb 6 -> POC 70).
#[test]
fn test_poc_wraparound() {
    let mut parser = H264Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeH264))
        .expect("init");
    let sps = sps_from(&mut parser, &[SPS_NAL, PPS_NAL]);

    // Fresh parser for the slice packet (only SPS/PPS cached there).
    let mut parser2 = H264Parser::new();
    parser2
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeH264))
        .expect("init");
    let headers = parse_headers(
        &mut parser2,
        &[
            SPS_NAL,
            PPS_NAL,
            SLICE_W29,
            SLICE_W30,
            SLICE_W31,
            SLICE_W32_WRAP,
            SLICE_W33,
            SLICE_W34,
            SLICE_W35,
        ],
    );
    assert_eq!(headers.len(), 7);
    let got = pocs(&sps, &headers);
    assert_eq!(
        got,
        [58i32, 56, 60, 70, 66, 64, 68],
        "wraparound POC mismatch"
    );
}

/// Full-stream POC sequence: all 300 pictures of h264_main.h264 (embedded at
/// compile time) must reconstruct to the oracle-verified POC values.
#[test]
fn test_full_stream_pocs() {
    let data = include_bytes!("../../../assets/samples/h264_main.h264").to_vec();
    let mut parser = H264Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeH264))
        .expect("init");
    let pkt = BitstreamPacket::new(data);

    let mut sps: Option<H264Sps> = None;
    let mut headers: Vec<H264Slh> = Vec::new();
    loop {
        match parser.parse(&pkt).expect("parse failed") {
            ParseResult::ParameterSet { sps: s, .. } => {
                if let Some(s) = s {
                    sps = Some(s.downcast_ref::<H264Sps>().cloned().unwrap());
                }
            }
            ParseResult::Slice { slices, .. } => {
                for s in &slices {
                    let header = s.slice_header.clone().expect("slice header must parse");
                    if let SliceHeader::H264(slh) = header
                        && slh.redundant_pic_cnt <= 0
                    {
                        headers.push(slh);
                    }
                }
            }
            ParseResult::Nothing | ParseResult::EndOfStream => break,
        }
    }

    let sps = sps.expect("SPS expected");
    assert_eq!(headers.len(), 300, "expected 300 pictures");
    let got = pocs(&sps, &headers);
    let expected: [i32; 300] = [
        0, 4, 2, 6, 14, 10, 8, 12, 22, 18, 16, 20, 30, 26, 24, 28, 38, 34, 32, 36, 46, 42, 40, 44,
        54, 50, 48, 52, 62, 58, 56, 60, 70, 66, 64, 68, 78, 74, 72, 76, 86, 82, 80, 84, 94, 90, 88,
        92, 102, 98, 96, 100, 110, 106, 104, 108, 118, 114, 112, 116, 126, 122, 120, 124, 134, 130,
        128, 132, 142, 138, 136, 140, 150, 146, 144, 148, 158, 154, 152, 156, 166, 162, 160, 164,
        174, 170, 168, 172, 182, 178, 176, 180, 190, 186, 184, 188, 198, 194, 192, 196, 206, 202,
        200, 204, 214, 210, 208, 212, 222, 218, 216, 220, 230, 226, 224, 228, 238, 234, 232, 236,
        246, 242, 240, 244, 254, 250, 248, 252, 262, 258, 256, 260, 270, 266, 264, 268, 278, 274,
        272, 276, 286, 282, 280, 284, 294, 290, 288, 292, 302, 298, 296, 300, 310, 306, 304, 308,
        318, 314, 312, 316, 326, 322, 320, 324, 334, 330, 328, 332, 342, 338, 336, 340, 350, 346,
        344, 348, 358, 354, 352, 356, 366, 362, 360, 364, 374, 370, 368, 372, 382, 378, 376, 380,
        390, 386, 384, 388, 398, 394, 392, 396, 406, 402, 400, 404, 414, 410, 408, 412, 422, 418,
        416, 420, 430, 426, 424, 428, 438, 434, 432, 436, 446, 442, 440, 444, 454, 450, 448, 452,
        462, 458, 456, 460, 470, 466, 464, 468, 478, 474, 472, 476, 486, 482, 480, 484, 494, 490,
        488, 492, 498, 496, 0, 8, 4, 2, 6, 16, 12, 10, 14, 24, 20, 18, 22, 32, 28, 26, 30, 40, 36,
        34, 38, 48, 44, 42, 46, 56, 52, 50, 54, 64, 60, 58, 62, 72, 68, 66, 70, 80, 76, 74, 78, 88,
        84, 82, 86, 96, 92, 90, 94, 98,
    ];
    assert_eq!(got, expected, "full-stream POC mismatch");
}

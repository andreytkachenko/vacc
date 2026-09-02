//! AV1 parser tests with real bitstream data from `assets/samples/av1_main.ivf`
//! (profile 0, level 1, 640x360, order_hint_bits_minus1=6, 448 frames in 300
//! IVF packets).
//!
//! Ground truth: the dumper output for this sample, cross-validated by an
//! independent Python oracle implementing FFmpeg's cbs_av1_syntax_template.c
//! bitstream syntax directly - 0 mismatches on all 300 dumped frames across
//! frame_type, show_existing_frame, frame_to_show_map_idx, frame size,
//! order_hint, refresh_frame_flags, show_frame and ref_frame_idx. The stream
//! is one true KEY frame (PIC 0), 95 show-existing-frame pictures, and the
//! rest inter frames with a cyclic last/golden/altref reference pattern.

use vacc_core::codec::VideoCodec;
use vacc_parser::av1::{Av1FrameHeader, Av1Parser};
use vacc_parser::{DetectedVideoFormat, VideoParser};

fn ivf_packets(data: &[u8]) -> Vec<&[u8]> {
    assert_eq!(&data[0..4], b"DKIF", "expected IVF container");
    let hsz = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut out = Vec::new();
    let mut off = hsz;
    while off + 12 <= data.len() {
        let size = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        if size == 0 || off + 12 + size > data.len() {
            break;
        }
        out.push(&data[off + 12..off + 12 + size]);
        off += 12 + size;
    }
    out
}

fn leb128(data: &[u8], mut i: usize) -> (usize, usize) {
    let mut value = 0usize;
    let mut shift = 0;
    loop {
        let b = data[i];
        i += 1;
        value |= ((b & 0x7f) as usize) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (value, i)
}

/// Walk the OBUs of one low-overhead (IVF) payload.
///
/// OBU header is a single byte: [forbidden(1), obu_type(4),
/// extension_flag(1), has_size_field(1), reserved(1)], plus one extension
/// byte (temporal_id/spatial_id) when the extension flag is set, plus a
/// leb128 payload size when the size field is present. Temporal delimiters
/// may omit the size field; all other OBUs must have it in this format.
fn walk_obus(payload: &[u8]) -> Vec<(u8, &[u8])> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < payload.len() {
        let b0 = payload[i];
        let obu_type = (b0 >> 3) & 0xf;
        let extension = b0 >> 2 & 1 == 1;
        let has_size = b0 >> 1 & 1 == 1;
        i += 1 + usize::from(extension);
        if has_size {
            let (sz, ni) = leb128(payload, i);
            out.push((obu_type, &payload[ni..ni + sz]));
            i = ni + sz;
        } else if obu_type == 2 {
            // Temporal delimiter without a size field.
            out.push((obu_type, &[]));
        } else {
            panic!("low-overhead stream: OBU type {obu_type} without size field");
        }
    }
    out
}

/// Parse the whole IVF: SPS from the first sequence header OBU, then every
/// frame / show-existing frame header in decode order.
fn parse_all(data: &[u8]) -> (vacc_core::picture::Av1Sps, Vec<Av1FrameHeader>) {
    let mut parser = Av1Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeAv1))
        .expect("init");
    let mut sps = None;
    let mut frames = Vec::new();
    for pkt in ivf_packets(data) {
        for (obu_type, payload) in walk_obus(&pkt) {
            match obu_type {
                1 => {
                    if sps.is_none() {
                        sps = Some(
                            parser
                                .parse_sequence_header_obu(payload)
                                .expect("sequence header parse"),
                        );
                    }
                }
                3 | 6 => {
                    let sps = sps.as_ref().expect("SPS before frames");
                    frames.push(
                        parser
                            .parse_frame_header(payload, sps)
                            .expect("frame header parse"),
                    );
                }
                _ => {}
            }
        }
    }
    (sps.expect("sequence header expected"), frames)
}

/// Sequence header fields must match the sample's known values
/// (oracle-verified).
#[test]
fn test_sps_fields() {
    let data = include_bytes!("../../../assets/samples/av1_main.ivf");
    let (sps, frames) = parse_all(data);
    assert_eq!(sps.profile, 0, "Main profile");
    assert_eq!(sps.level, 1);
    assert!(!sps.still_picture);
    assert!(!sps.reduced_still_picture_header);
    assert_eq!(sps.max_frame_width_minus_1 as u32 + 1, 640);
    assert_eq!(sps.max_frame_height_minus_1 as u32 + 1, 360);
    assert!(!sps.frame_id_numbers_present_flag);
    assert!(sps.enable_order_hint);
    assert_eq!(sps.order_hint_bits_minus1, 6, "7-bit order hints");
    assert_eq!(sps.seq_force_screen_content_tools, 2, "SCT select");
    assert_eq!(sps.seq_force_integer_mv, 2, "integer MV select");
    assert!(!sps.enable_superres);
    assert!(sps.enable_cdef);
    assert!(!sps.enable_restoration);
    assert_eq!(frames.len(), 448, "frame count");
}

/// Per-picture fields for the first thirteen decode-order pictures: one true
/// KEY frame, inter frames with the cyclic reference pattern, and three
/// show-existing-frame pictures (which carry parser defaults).
#[test]
fn test_first_frames() {
    let data = include_bytes!("../../../assets/samples/av1_main.ivf");
    let (_, frames) = parse_all(data);

    // PIC 0: the only true KEY frame.
    let f = &frames[0];
    assert_eq!(f.frame_type, 0, "KEY");
    assert!(!f.show_existing_frame);
    assert_eq!(f.order_hint, 0);
    assert_eq!(f.refresh_frame_flags, 0xff, "key refreshes all buffers");
    assert!(f.show_frame);
    assert_eq!((f.frame_width, f.frame_height), (640, 360));

    // PIC 1-6: inter frames cycling through the reference buffers.
    let expected: &[(u8, u32, u8, bool, [u8; 7])] = &[
        (1, 32, 2, false, [0, 0, 0, 0, 0, 0, 0]), // PIC1
        (1, 16, 4, false, [0, 0, 0, 0, 0, 0, 1]), // PIC2
        (1, 8, 8, false, [0, 0, 0, 0, 2, 0, 1]), // PIC3
        (1, 4, 16, false, [0, 0, 0, 0, 3, 2, 1]), // PIC4
        (1, 2, 32, false, [0, 0, 2, 0, 4, 3, 1]), // PIC5
        (1, 1, 64, true, [0, 3, 2, 0, 5, 4, 1]), // PIC6
    ];
    for (i, (ftype, oh, refresh, show, refs)) in expected.iter().enumerate() {
        let f = &frames[1 + i];
        assert_eq!(f.frame_type, *ftype, "PIC{} frame type", 1 + i);
        assert!(!f.show_existing_frame, "PIC{}", 1 + i);
        assert_eq!(f.order_hint, *oh, "PIC{} order hint", 1 + i);
        assert_eq!(f.refresh_frame_flags, *refresh, "PIC{} refresh", 1 + i);
        assert_eq!(f.show_frame, *show, "PIC{} show", 1 + i);
        assert_eq!(&f.ref_frame_idx[..], &refs[..], "PIC{} refs", 1 + i);
    }

    // PIC 7, 9, 12: show-existing-frame pictures (re-display of an existing
    // buffer; no new content, parser reports defaults).
    for (idx, map) in [(7usize, 5u8), (9, 4), (12, 6)] {
        let f = &frames[idx];
        assert!(f.show_existing_frame, "PIC{idx} show existing");
        assert_eq!(f.frame_to_show_map_idx, map, "PIC{idx} map idx");
        assert_eq!(f.refresh_frame_flags, 0, "PIC{idx} refresh");
    }

    // PIC 8: inter frame using all seven reference buffers.
    let f = &frames[8];
    assert_eq!(f.frame_type, 1);
    assert!(!f.show_existing_frame);
    assert_eq!(f.order_hint, 3);
    assert_eq!(f.refresh_frame_flags, 128);
    assert!(f.show_frame);
    assert_eq!(&f.ref_frame_idx[..], &[5, 6, 2, 0, 4, 3, 1]);
}

/// (frame_type, order_hint, refresh_frame_flags) for the first 300 pictures,
/// oracle-verified. The remaining 148 frames are counted but not field-checked
/// (the dump was capped at 300).
const EXPECTED_FIRST_300: &[u8] = &include!("data/av1_expected_first300.rs");

#[test]
fn test_full_stream_order_hints() {
    let data = include_bytes!("../../../assets/samples/av1_main.ivf");
    let (_, frames) = parse_all(data);
    assert_eq!(frames.len(), 448, "expected 448 frames");
    assert_eq!(EXPECTED_FIRST_300.len(), 900);

    for (i, f) in frames.iter().take(300).enumerate() {
        let e = &EXPECTED_FIRST_300[i * 3..i * 3 + 3];
        assert_eq!(f.frame_type, e[0] as u8, "PIC{i} frame type");
        assert_eq!(f.order_hint, e[1] as u32, "PIC{i} order hint");
        assert_eq!(
            f.refresh_frame_flags,
            e[2],
            "PIC{i} refresh flags (type={} oh={})",
            f.frame_type,
            f.order_hint
        );
    }

    // The only true KEY frame in the first 300 pictures is PIC 0; every other
    // type-0 picture is a show-existing-frame default.
    let true_keys: Vec<usize> = frames
        .iter()
        .take(300)
        .enumerate()
        .filter(|(_, f)| f.frame_type == 0 && !f.show_existing_frame)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(true_keys, vec![0]);
}

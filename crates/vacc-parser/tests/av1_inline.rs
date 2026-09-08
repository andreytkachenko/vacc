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
/// Returns (obu_type, temporal_id, spatial_id, payload).
fn walk_obus(payload: &[u8]) -> Vec<(u8, u32, u32, &[u8])> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < payload.len() {
        let b0 = payload[i];
        let obu_type = (b0 >> 3) & 0xf;
        let extension = b0 >> 2 & 1 == 1;
        let has_size = b0 >> 1 & 1 == 1;
        let ext_byte = if extension { Some(payload[i + 1]) } else { None };
        i += 1 + usize::from(extension);
        let (temporal_id, spatial_id) = match ext_byte {
            Some(e) => (((e >> 5) & 0x7) as u32, ((e >> 0) & 0x1f) as u32),
            None => (0, 0),
        };
        if has_size {
            let (sz, ni) = leb128(payload, i);
            out.push((obu_type, temporal_id, spatial_id, &payload[ni..ni + sz]));
            i = ni + sz;
        } else if obu_type == 2 {
            // Temporal delimiter without a size field.
            out.push((obu_type, temporal_id, spatial_id, &[]));
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
        for (obu_type, temporal_id, spatial_id, payload) in walk_obus(pkt) {
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
                            .parse_frame_header(payload, sps, temporal_id, spatial_id)
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
        (1, 8, 8, false, [0, 0, 0, 0, 2, 0, 1]),  // PIC3
        (1, 4, 16, false, [0, 0, 0, 0, 3, 2, 1]), // PIC4
        (1, 2, 32, false, [0, 0, 2, 0, 4, 3, 1]), // PIC5
        (1, 1, 64, true, [0, 3, 2, 0, 5, 4, 1]),  // PIC6
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
        assert_eq!(f.frame_type, e[0], "PIC{i} frame type");
        assert_eq!(f.order_hint, e[1] as u32, "PIC{i} order hint");
        assert_eq!(
            f.refresh_frame_flags, e[2],
            "PIC{i} refresh flags (type={} oh={})",
            f.frame_type, f.order_hint
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

// =====================================================================
// Synthetic decoder-model streams (audit issues 1+2): an SPS with
// timing_info + decoder_model_info and a KEY frame header carrying
// frame_presentation_time (temporal_point_info) and the
// buffer_removal_time block. The bitstreams are built bit-by-bit below;
// the parser must consume exactly those bits — `frame_header_size` pins
// the total, so a missing/misplaced timing read desyncs the size.
// =====================================================================

struct BitWriter {
    bytes: Vec<u8>,
    bitpos: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bitpos: 0,
        }
    }

    /// Write the low `n` bits of `value`, MSB-first (AV1 spec f(n)).
    fn write(&mut self, value: u64, n: u32) {
        for i in (0..n).rev() {
            let bit = (value >> i) & 1;
            let byte_idx = (self.bitpos / 8) as usize;
            if byte_idx >= self.bytes.len() {
                self.bytes.push(0);
            }
            if bit == 1 {
                self.bytes[byte_idx] |= 1 << (7 - (self.bitpos % 8));
            }
            self.bitpos += 1;
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Minimal Main-profile SPS: timing_info present, equal_picture_interval=0,
/// decoder_model_info present (FPT/BRT lengths minus 1 = 5 → n=6), one or
/// two operating points with decoder_model_present_for_this_op set.
fn synthetic_timing_sps(op_count: usize, op1_idc: u32, grain_present: bool) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write(0, 3); // seq_profile Main
    w.write(0, 1); // still_picture
    w.write(0, 1); // reduced_still_picture_header
    w.write(1, 1); // timing_info_present_flag
    w.write(30, 32); // num_units_in_display_tick
    w.write(1000, 32); // time_scale
    w.write(0, 1); // equal_picture_interval = 0 → FPT read per frame
    w.write(1, 1); // decoder_model_info_present_flag
    w.write(5, 5); // buffer_delay_length_minus_1 (n=6)
    w.write(30, 32); // num_units_in_decoding_tick
    w.write(5, 5); // buffer_removal_time_length_minus_1 (n=6)
    w.write(5, 5); // frame_presentation_time_length_minus_1 (n=6)
    w.write(0, 1); // initial_display_delay_present_flag
    w.write((op_count - 1) as u64, 5); // operating_points_cnt_minus_1
    for op in 0..op_count {
        w.write(if op == 0 { 0 } else { op1_idc as u64 }, 12); // operating_point_idc
        w.write(1, 5); // seq_level_idx (level 1, no tier bit)
        w.write(1, 1); // decoder_model_present_for_this_op
        w.write(0, 6); // decoder_buffer_delay
        w.write(0, 6); // encoder_buffer_delay
        w.write(0, 1); // low_delay_mode_flag
    }
    w.write(5, 4); // frame_width_bits_minus_1 (6-bit widths)
    w.write(5, 4); // frame_height_bits_minus_1
    w.write(63, 6); // max_frame_width_minus_1 → 64
    w.write(35, 6); // max_frame_height_minus_1 → 36
    w.write(0, 1); // frame_id_numbers_present_flag
    w.write(0, 1); // use_128x128_superblock
    w.write(0, 1); // enable_filter_intra
    w.write(0, 1); // enable_intra_edge_filter
    w.write(0, 1); // enable_interintra_compound
    w.write(0, 1); // enable_masked_compound
    w.write(0, 1); // enable_warped_motion
    w.write(0, 1); // enable_dual_filter
    w.write(0, 1); // enable_order_hint = 0 → no order-hint bits in frames
    w.write(0, 1); // seq_choose_screen_content_tools
    w.write(0, 1); // seq_force_screen_content_tools = 0
    w.write(0, 1); // enable_superres
    w.write(0, 1); // enable_cdef
    w.write(0, 1); // enable_restoration
    // color_config (profile 0: no twelve_bit; 4:2:0 subsampling implicit)
    w.write(0, 1); // high_bitdepth
    w.write(0, 1); // mono_chrome
    w.write(0, 1); // color_description_present
    w.write(0, 1); // color_range
    w.write(0, 2); // chroma_sample_position (4:2:0)
    w.write(0, 1); // separate_uv_delta_q
    w.write(grain_present as u64, 1); // film_grain_params_present
    // trailing bits: 1 bit then zero pad to the byte boundary
    w.write(1, 1);
    if w.bitpos % 8 != 0 {
        w.write(0, 8 - (w.bitpos % 8));
    }
    w.into_bytes()
}

/// Minimal KEY frame header (64x36, single tile, base_q=0) with
/// frame_presentation_time f(6) after show_frame and the
/// buffer_removal_time block (present flag + one 6-bit value for op idc 0).
/// 48 bits total → frame_header_size must be 6.
fn synthetic_key_frame_with_timing() -> (Vec<u8>, u32) {
    let mut w = BitWriter::new();
    w.write(0, 1); // show_existing_frame
    w.write(0, 2); // frame_type KEY
    w.write(1, 1); // show_frame
    w.write(0x2a, 6); // frame_presentation_time (issue 1)
    w.write(0, 1); // disable_cdf_update
    w.write(1, 1); // frame_size_override_flag
    w.write(1, 1); // buffer_removal_time_present_flag (issue 2)
    w.write(0x15, 6); // buffer_removal_time[op0] (op idc 0 → read) (issue 2)
    w.write(63, 6); // frame_width_minus_1 → 64
    w.write(35, 6); // frame_height_minus_1 → 36
    w.write(0, 1); // render_and_frame_size_different
    w.write(0, 1); // disable_frame_end_update_cdf
    w.write(1, 1); // uniform_tile_spacing_flag (1x1 grid → no further bits)
    w.write(0, 8); // base_q_index = 0
    w.write(0, 1); // delta_q_y_dc present
    w.write(0, 1); // delta_q_u_dc present
    w.write(0, 1); // delta_q_u_ac present
    w.write(0, 1); // using_qmatrix
    w.write(0, 1); // segmentation_enabled
    w.write(0, 1); // reduced_tx_set
    let nbits = w.bitpos;
    (w.into_bytes(), nbits)
}

#[test]
fn test_decoder_model_timing_reads() {
    let sps_bytes = synthetic_timing_sps(1, 0, false);
    let mut parser = Av1Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeAv1))
        .expect("init");
    let sps = parser
        .parse_sequence_header_obu(&sps_bytes)
        .expect("SPS parse");

    // SPS fields stored for the frame-header timing reads.
    assert!(sps.timing_info_present_flag);
    assert!(!sps.equal_picture_interval);
    assert!(sps.decoder_model_info_present_flag);
    assert_eq!(sps.frame_presentation_time_length_minus_1, 5);
    assert_eq!(sps.buffer_removal_time_length_minus_1, 5);
    assert_eq!(sps.operating_points_cnt_minus_1, 0);
    assert_eq!(sps.operating_point_idc[0], 0);
    assert!(sps.decoder_model_present_for_this_op[0]);

    let (frame_bytes, nbits) = synthetic_key_frame_with_timing();
    assert_eq!(nbits, 48, "synthetic header is 48 bits");

    let fh = parser
        .parse_frame_header(&frame_bytes, &sps, 0, 0)
        .expect("frame header parse");

    // Exactly the written bits were consumed — this pins both timing reads
    // (13 bits: 6-bit FPT + 1-bit BRT flag + 6-bit BRT) at their spec
    // positions; a missing or misplaced read shifts frame_header_size.
    assert_eq!(fh.frame_header_size, 6, "48 bits -> 6 header bytes");
    assert_eq!(fh.frame_type, 0);
    assert!(fh.show_frame);
    assert_eq!(fh.refresh_frame_flags, 0xff);
    assert_eq!((fh.frame_width, fh.frame_height), (64, 36));
    assert!(fh.coded_lossless);
}

#[test]
fn test_brt_operating_point_gating() {
    // op1 idc = 4: with temporal_id=0/spatial_id=0, inTemporal=(4>>0)&1=0
    // and inSpatial=(4>>8)&1=0 → its BRT value is NOT read (only op0's is).
    let sps_bytes = synthetic_timing_sps(2, 4, false);
    let mut parser = Av1Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeAv1))
        .expect("init");
    let sps = parser
        .parse_sequence_header_obu(&sps_bytes)
        .expect("SPS parse");
    assert_eq!(sps.operating_point_idc[1], 4);

    let (frame_bytes, nbits) = synthetic_key_frame_with_timing();
    assert_eq!(nbits, 48);
    let fh = parser
        .parse_frame_header(&frame_bytes, &sps, 0, 0)
        .expect("frame header parse");
    // Still exactly one BRT value consumed (op0 only).
    assert_eq!(fh.frame_header_size, 6, "gated op1 must not add a BRT read");

    // op idc = 0x101: inTemporal=(0x101>>0)&1=1 and inSpatial=(0x101>>8)&1=1
    // → for a second such op the BRT value IS read (header grows by 6 bits).
    let mut w = BitWriter::new();
    w.write(0, 1); // show_existing_frame
    w.write(0, 2); // frame_type KEY
    w.write(1, 1); // show_frame
    w.write(0x2a, 6); // frame_presentation_time
    w.write(0, 1); // disable_cdf_update
    w.write(1, 1); // frame_size_override_flag
    w.write(1, 1); // buffer_removal_time_present_flag
    w.write(0x15, 6); // BRT op0 (idc 0)
    w.write(0x3c, 6); // BRT op1 (idc 0x101: in both layers)
    w.write(63, 4); // frame_width_minus_1
    w.write(35, 4); // frame_height_minus_1
    w.write(0, 1); // render_and_frame_size_different
    w.write(0, 1); // disable_frame_end_update_cdf
    w.write(1, 1); // uniform_tile_spacing_flag
    w.write(0, 8); // base_q_index
    w.write(0, 1); // delta_q_y_dc present
    w.write(0, 1); // delta_q_u_dc present
    w.write(0, 1); // delta_q_u_ac present
    w.write(0, 1); // using_qmatrix
    w.write(0, 1); // segmentation_enabled
    w.write(0, 1); // reduced_tx_set
    let nbits = w.bitpos;
    let frame_bytes = w.into_bytes();
    assert_eq!(nbits, 50);

    let sps_bytes = synthetic_timing_sps(2, 0x101, false);
    let mut parser = Av1Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeAv1))
        .expect("init");
    let sps = parser
        .parse_sequence_header_obu(&sps_bytes)
        .expect("SPS parse");
    let fh = parser
        .parse_frame_header(&frame_bytes, &sps, 0, 0)
        .expect("frame header parse");
    assert_eq!(fh.frame_header_size, 7, "54 bits -> 7 header bytes");
}

/// Minimal Main-profile SPS with film_grain_params_present=1 and NO
/// timing/decoder-model info (isolates the grain block).
fn synthetic_grain_sps() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write(0, 3); // seq_profile Main
    w.write(0, 1); // still_picture
    w.write(0, 1); // reduced_still_picture_header
    w.write(0, 1); // timing_info_present_flag = 0 -> no decoder model
    w.write(0, 1); // initial_display_delay_present_flag
    w.write(0, 5); // operating_points_cnt_minus_1 = 0
    w.write(0, 12); // op0 operating_point_idc
    w.write(1, 5); // op0 seq_level_idx
    w.write(5, 4); // frame_width_bits_minus_1 (6-bit widths)
    w.write(5, 4); // frame_height_bits_minus_1
    w.write(63, 6); // max_frame_width_minus_1 -> 64
    w.write(35, 6); // max_frame_height_minus_1 -> 36
    w.write(0, 1); // frame_id_numbers_present_flag
    w.write(0, 1); // use_128x128_superblock
    w.write(0, 1); // enable_filter_intra
    w.write(0, 1); // enable_intra_edge_filter
    w.write(0, 1); // enable_interintra_compound
    w.write(0, 1); // enable_masked_compound
    w.write(0, 1); // enable_warped_motion
    w.write(0, 1); // enable_dual_filter
    w.write(0, 1); // enable_order_hint = 0
    w.write(0, 1); // seq_choose_screen_content_tools
    w.write(0, 1); // seq_force_screen_content_tools = 0
    w.write(0, 1); // enable_superres
    w.write(0, 1); // enable_cdef
    w.write(0, 1); // enable_restoration
    // color_config (profile 0: no twelve_bit; 4:2:0 subsampling implicit)
    w.write(0, 1); // high_bitdepth
    w.write(0, 1); // mono_chrome
    w.write(0, 1); // color_description_present
    w.write(0, 1); // color_range
    w.write(0, 2); // chroma_sample_position (4:2:0)
    w.write(0, 1); // separate_uv_delta_q
    w.write(1, 1); // film_grain_params_present = 1
    // trailing bits: 1 bit then zero pad to byte boundary
    w.write(1, 1);
    if w.bitpos % 8 != 0 {
        w.write(0, 8 - (w.bitpos % 8));
    }
    w.into_bytes()
}

/// Minimal KEY frame header (64x36) whose last element is a full
/// film_grain_params block: apply_grain=1, 1 y point, chroma_scaling_from_
/// luma=1, ar_coeff_lag=1 (numPosLuma=4, numPosChroma=5). Base header is 35
/// bits, grain block 160 bits -> 195 bits total -> frame_header_size 25.
fn synthetic_key_frame_with_grain() -> (Vec<u8>, u32) {
    let mut w = BitWriter::new();
    w.write(0, 1); // show_existing_frame
    w.write(0, 2); // frame_type KEY
    w.write(1, 1); // show_frame
    w.write(0, 1); // disable_cdf_update
    w.write(1, 1); // frame_size_override_flag
    w.write(63, 6); // frame_width_minus_1 -> 64
    w.write(35, 6); // frame_height_minus_1 -> 36
    w.write(0, 1); // render_and_frame_size_different
    w.write(0, 1); // disable_frame_end_update_cdf
    w.write(1, 1); // uniform_tile_spacing_flag (single tile)
    w.write(0, 8); // base_q_index
    w.write(0, 1); // delta_q_y_dc present
    w.write(0, 1); // delta_q_u_dc present
    w.write(0, 1); // delta_q_u_ac present
    w.write(0, 1); // using_qmatrix
    w.write(0, 1); // segmentation_enabled
    w.write(0, 1); // reduced_tx_set
    // ---- film_grain_params (last element) ----
    w.write(1, 1); // apply_grain
    w.write(0x1234, 16); // grain_seed
    // update_grain inferred 1 for KEY (no bit)
    w.write(1, 4); // num_y_points = 1
    w.write(0x80, 8); // point_y_value[0]
    w.write(0x7f, 8); // point_y_scaling[0]
    w.write(1, 1); // chroma_scaling_from_luma = 1 (skips cb/cr points)
    w.write(2, 2); // grain_scaling_minus_8
    w.write(1, 2); // ar_coeff_lag = 1 -> numPosLuma=4, numPosChroma=5
    for v in [0x7f, 0x80, 0x81, 0x82] {
        w.write(v as u64, 8); // ar_coeffs_y_plus_128[0..4]
    }
    for v in [0x00, 0x01, 0x02, 0x03, 0x04] {
        w.write(v as u64, 8); // ar_coeffs_cb_plus_128[0..5]
    }
    for v in [0x05, 0x06, 0x07, 0x08, 0x09] {
        w.write(v as u64, 8); // ar_coeffs_cr_plus_128[0..5]
    }
    w.write(1, 2); // ar_coeff_shift_minus_6
    w.write(3, 2); // grain_scale_shift
    w.write(1, 1); // overlap_flag
    w.write(0, 1); // clip_to_restricted_range
    let nbits = w.bitpos;
    (w.into_bytes(), nbits)
}

#[test]
fn test_film_grain_params_reads() {
    let sps_bytes = synthetic_grain_sps();
    let mut parser = Av1Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeAv1))
        .expect("init");
    let sps = parser
        .parse_sequence_header_obu(&sps_bytes)
        .expect("SPS parse");
    assert!(sps.film_grain_params_present);

    let (frame_bytes, nbits) = synthetic_key_frame_with_grain();
    assert_eq!(nbits, 195, "35 base + 160 grain bits");

    let fh = parser
        .parse_frame_header(&frame_bytes, &sps, 0, 0)
        .expect("frame header parse");

    // The full grain block must have been consumed (pins its position as the
    // last element of uncompressed_header); a short/long read shifts the size.
    assert_eq!(fh.frame_header_size, 25, "195 bits -> 25 header bytes");
    assert!(fh.apply_grain);
    let g = &fh.film_grain;
    assert!(g.apply_grain);
    assert_eq!(g.grain_seed, 0x1234);
    assert!(g.update_grain);
    assert_eq!(g.num_y_points, 1);
    assert_eq!(g.point_y_value[0], 0x80);
    assert_eq!(g.point_y_scaling[0], 0x7f);
    assert!(g.chroma_scaling_from_luma);
    assert_eq!(g.grain_scaling_minus_8, 2);
    assert_eq!(g.ar_coeff_lag, 1);
    assert_eq!(&g.ar_coeffs_y_plus_128[..4], [0x7f_i8, 0x80_u8 as i8, 0x81_u8 as i8, 0x82_u8 as i8]);
    assert_eq!(&g.ar_coeffs_cb_plus_128[..5], [0, 1, 2, 3, 4]);
    assert_eq!(&g.ar_coeffs_cr_plus_128[..5], [5, 6, 7, 8, 9]);
    assert_eq!(g.ar_coeff_shift_minus_6, 1);
    assert_eq!(g.grain_scale_shift, 3);
    assert!(g.overlap_flag);
    assert!(!g.clip_to_restricted_range);
}

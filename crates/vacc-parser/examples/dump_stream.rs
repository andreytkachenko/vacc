//! dump_stream — collect parser / DPB / POC ground-truth data from a sample
//! bitstream for inlining into unit tests.
//!
//! Usage:
//!   cargo run -p vacc-parser --example dump_stream -- <h264|h265|vp9|av1> <file> [max_frames]
//!
//! Output format (text, consumed by the test-suite data files):
//!   SPS/PPS/VPS  : `SET <kind> <hex>` — raw parameter-set NAL/OBU bytes
//!   PIC n        : one line per decoded picture with key=value fields,
//!                  `L0`/`L1` reference-list POCs (as computed by the common
//!                  DPB) and `RAW` = slice-NAL prefix bytes (header only).

use std::fs;
use std::process::exit;

use vacc_core::codec::VideoCodec;
use vacc_parser::h264_dpb::{H264Dpb, H264MmcoCommand};
use vacc_parser::h264_poc::PocCalculator;
use vacc_parser::h265_dpb::H265Dpb;
use vacc_parser::vp9_dpb::Vp9Dpb;
use vacc_parser::{BitstreamPacket, DetectedVideoFormat, ParseResult, SliceHeader, VideoParser};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dump_stream <h264|h265|vp9|av1> <file> [max_frames]");
        exit(1);
    }
    let codec = args[1].as_str();
    let data = fs::read(&args[2]).unwrap_or_else(|e| panic!("read {}: {e}", args[2]));
    let max_frames = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(300);

    match codec {
        "h264" => dump_h264(&data, max_frames),
        "h265" => dump_h265(&data, max_frames),
        "vp9" => dump_vp9(&data, max_frames),
        "av1" => dump_av1(&data, max_frames),
        _ => {
            eprintln!("unknown codec {codec}");
            exit(1);
        }
    }
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}

fn mmco_vec(mmco: &[(u32, u32)]) -> Vec<H264MmcoCommand> {
    mmco.iter()
        .map(|(op, value)| match *op {
            1 => H264MmcoCommand::UnmarkShortTerm {
                difference_of_pic_nums_minus1: *value,
            },
            2 => H264MmcoCommand::UnmarkLongTerm {
                long_term_frame_idx: *value,
            },
            3 => H264MmcoCommand::AssignLongTerm {
                difference_of_pic_nums_minus1: 0,
                long_term_frame_idx: *value,
            },
            4 => H264MmcoCommand::SetMaxLongTermFrameIdx {
                max_long_term_frame_idx_plus1: *value,
            },
            5 => H264MmcoCommand::UnmarkAll,
            6 => H264MmcoCommand::AssignLongTermToCurrent {
                long_term_frame_idx: *value,
            },
            _ => H264MmcoCommand::UnmarkAll,
        })
        .collect()
}

fn dump_h264(data: &[u8], max_frames: usize) {
    let mut parser = vacc_parser::h264::H264Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeH264))
        .expect("init");
    let packet = BitstreamPacket::new(data.to_vec());

    let mut sps = None;
    let mut poc_calc = PocCalculator::new();
    let mut dpb: Option<H264Dpb> = None;
    let mut pic_idx = 0usize;

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::ParameterSet {
                sps: s,
                pps,
                sps_nal,
                pps_nal,
                ..
            }) => {
                if let Some(s) = s {
                    sps = Some(
                        s.downcast_ref::<vacc_core::picture::H264Sps>()
                            .cloned()
                            .unwrap(),
                    );
                    let sps = sps.as_ref().unwrap();
                    println!(
                        "SPS profile={} level={} chroma={} luma_bd={} log2_mfn_minus4={} poc_type={} \
                         max_poc_lsb={} max_ref_frames={} w={} h={}",
                        sps.profile_idc,
                        sps.level_idc,
                        sps.chroma_format_idc,
                        8 + sps.bit_depth_luma_minus8 as u32,
                        sps.log2_max_frame_num_minus4,
                        sps.pic_order_cnt_type,
                        sps.max_pic_order_cnt_lsb,
                        sps.max_num_ref_frames,
                        (sps.pic_width_in_mbs_minus1 as u32 + 1) * 16,
                        (sps.pic_height_in_map_units_minus1 as u32 + 1) * 16,
                    );
                    if let Some(nal) = &sps_nal {
                        println!("SET sps {}", hex(nal));
                    }
                }
                if let (Some(p), Some(nal)) = (pps, pps_nal) {
                    let pps = p
                        .downcast_ref::<vacc_core::picture::H264Pps>()
                        .cloned()
                        .unwrap();
                    println!(
                        "PPS id={} sps_id={} entropy={} nr0={} nr1={} weighted={} redundant={}",
                        pps.pic_parameter_set_id,
                        pps.seq_parameter_set_id,
                        u32::from(pps.entropy_coding_mode_flag),
                        pps.num_ref_idx_l0_default_active_minus1,
                        pps.num_ref_idx_l1_default_active_minus1,
                        u32::from(pps.weighted_pred_flag),
                        u32::from(pps.redundant_pic_cnt_present_flag),
                    );
                    println!("SET pps {}", hex(&nal));
                }
            }
            Ok(ParseResult::Slice { slices, .. }) => {
                let first = &slices[0];
                let SliceHeader::H264(slh) = first.slice_header.as_ref().expect("h264 slh") else {
                    panic!();
                };
                if slh.redundant_pic_cnt > 0 {
                    continue;
                }
                let sps = sps.clone().expect("sps");
                let is_ref = slh.nal_ref_idc != 0;
                if slh.nal_unit_type == 5 {
                    poc_calc.reset();
                    let num_ref_frames = sps.max_num_ref_frames.clamp(1, 16);
                    dpb = Some(H264Dpb::new(16, 16, num_ref_frames, sps.max_frame_num));
                }
                let poc = poc_calc.calculate(&sps, slh, is_ref);
                let mmco: Vec<(u32, u32)> = slh
                    .dec_ref_pic_marking
                    .iter()
                    .map(|e| (e.memory_management_control_operation, e.value))
                    .collect();
                let cmds = mmco_vec(&mmco);
                let no_output = slh.no_output_of_prior_pics_flag;
                let is_idr = slh.nal_unit_type == 5;
                let dpb = dpb.as_mut().expect("dpb");
                dpb.picture_start(
                    slh.frame_num,
                    poc,
                    is_ref,
                    is_idr,
                    no_output,
                    !cmds.is_empty(),
                    cmds,
                );
                let lists = dpb.build_ref_lists(
                    slh.slice_type % 5,
                    slh.num_ref_idx_l0_active_minus1,
                    slh.num_ref_idx_l1_active_minus1,
                    &slh.ref_pic_list_modification_l0,
                    &slh.ref_pic_list_modification_l1,
                );
                let slot = dpb.prepare_current();
                dpb.commit_current(slot);

                let pic = pic_idx;
                pic_idx += 1;
                if pic >= max_frames {
                    break;
                }
                let rplm = |v: &[vacc_parser::h264::RefPicListModificationEntry]| -> String {
                    v.iter()
                        .map(|e| format!("{}:{}:{}", e.op, e.index, e.difference))
                        .collect::<Vec<_>>()
                        .join(",")
                };
                let l0: Vec<i32> = lists.l0.iter().map(|r| r.poc).collect();
                let l1: Vec<i32> = lists.l1.iter().map(|r| r.poc).collect();
                let mmco_s: Vec<String> = mmco.iter().map(|(o, v)| format!("{o}:{v}")).collect();
                println!(
                    "PIC {pic}\n  fn={} poc={} slt={} pps={} idr={} poc_lsb={} nal_ref={} \
                     nr0={} nr1={} nopp={} mmco=[{}] rplm0=[{}] rplm1=[{}] hbs={}\n  L0 [{}]\n  L1 [{}]",
                    slh.frame_num,
                    poc,
                    slh.slice_type,
                    slh.pic_parameter_set_id,
                    u32::from(is_idr),
                    slh.pic_order_cnt_lsb,
                    slh.nal_ref_idc,
                    slh.num_ref_idx_l0_active_minus1,
                    slh.num_ref_idx_l1_active_minus1,
                    u32::from(no_output),
                    mmco_s.join(","),
                    rplm(&slh.ref_pic_list_modification_l0),
                    rplm(&slh.ref_pic_list_modification_l1),
                    slh.header_bit_size,
                    l0.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "),
                    l1.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "),
                );
                let header_bytes = (slh.header_bit_size as usize).div_ceil(8);
                let raw = &first.nal_data[..header_bytes.min(first.nal_data.len())];
                println!("  RAW {}", hex(raw));
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Err(e) => panic!("parse error: {e}"),
        }
    }
    eprintln!("h264: {pic_idx} pictures");
}

fn dump_h265(data: &[u8], max_frames: usize) {
    let mut parser = vacc_parser::h265::H265Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeH265))
        .expect("init");
    let packet = BitstreamPacket::new(data.to_vec());

    let mut sps = None;
    let mut dpb = H265Dpb::new(16);
    let mut pic_idx = 0usize;

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::ParameterSet {
                sps: s, pps, vps, ..
            }) => {
                if let Some(s) = s {
                    sps = Some(
                        s.downcast_ref::<vacc_core::picture::H265Sps>()
                            .cloned()
                            .unwrap(),
                    );
                    let sps = sps.as_ref().unwrap();
                    println!(
                        "SPS profile={} level={} chroma={} luma_bd={} log2_mpc_lsb={} max_ref={} w={} h={} reorder={}",
                        sps.profile_idc,
                        sps.level_idc,
                        sps.chroma_format_idc,
                        8 + sps.bit_depth_luma_minus8 as u32,
                        sps.log2_max_pic_order_cnt_lsb_minus4,
                        sps.max_dec_pic_buffering_minus1[0] + 1,
                        sps.pic_width_in_luma_samples,
                        sps.pic_height_in_luma_samples,
                        sps.max_num_reorder_pics[0],
                    );
                    dpb.set_max_num_reorder_frames(sps.max_num_reorder_pics[0] as u32);
                }
                if let Some(p) = pps {
                    let pps = p
                        .downcast_ref::<vacc_core::picture::H265Pps>()
                        .cloned()
                        .unwrap();
                    println!(
                        "PPS id={} sps_id={} nr0={} nr1={} output_flag_present={}",
                        pps.pps_pic_parameter_set_id,
                        pps.pps_seq_parameter_set_id,
                        pps.num_ref_idx_l0_default_active_minus1,
                        pps.num_ref_idx_l1_default_active_minus1,
                        u32::from(pps.output_flag_present_flag),
                    );
                }
                if let Some(v) = vps {
                    let _ = v.downcast_ref::<vacc_core::picture::H265Vps>().unwrap();
                    println!("SET vps (see raw NAL walk below)");
                }
            }
            Ok(ParseResult::Slice { slices, .. }) => {
                let first = &slices[0];
                let SliceHeader::H265(info) = first.slice_header.as_ref().expect("h265 info")
                else {
                    panic!();
                };
                let sps = sps.clone().expect("sps");
                let pic = pic_idx;
                pic_idx += 1;
                if pic >= max_frames {
                    break;
                }

                let slot = dpb.picture_start(&sps, info, info.is_reference);
                let lists = dpb.build_ref_lists();
                let l0: Vec<i32> = lists.l0.iter().map(|r| r.poc).collect();
                let l1: Vec<i32> = lists.l1.iter().map(|r| r.poc).collect();
                dpb.commit_current(slot);

                let rplm = |v: &[vacc_parser::h265::H265ListModification]| -> String {
                    v.iter()
                        .map(|e| {
                            if e.flag {
                                format!("1:{}", e.ref_idx)
                            } else {
                                "0".to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(",")
                };
                println!(
                    "PIC {pic}\n  poc={} lsb={} slt={} idr={} rap={} ref={} nopp={} strps_sps={} strps_idx={} \
                     nr0={} nr1={} rplm0=[{}] rplm1=[{}] lt_sps={} lt_pics={} addr={} dep={}\n  L0 [{}]\n  L1 [{}]",
                    info.curr_pic_order_cnt_val,
                    info.pic_order_cnt_lsb,
                    info.slice_type,
                    u32::from(info.is_idr),
                    u32::from(info.is_rap),
                    u32::from(info.is_reference),
                    u32::from(info.no_output_of_prior_pics_flag),
                    u32::from(info.short_term_ref_pic_set_sps_flag),
                    info.short_term_ref_pic_set_idx,
                    info.num_ref_idx_l0_active_minus1,
                    info.num_ref_idx_l1_active_minus1,
                    rplm(&info.ref_pic_lists_modification_l0),
                    rplm(&info.ref_pic_lists_modification_l1),
                    info.num_long_term_sps,
                    info.num_long_term_pics,
                    info.slice_segment_address,
                    u32::from(info.dependent_slice_segment_flag),
                    l0.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "),
                    l1.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "),
                );
                let header_bytes = 2 + (info.header_bit_size as usize).div_ceil(8);
                let raw = &first.nal_data[..header_bytes.min(first.nal_data.len())];
                println!("  RAW {}", hex(raw));
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Err(e) => panic!("parse error: {e}"),
        }
    }
    eprintln!("h265: {pic_idx} pictures");

    // Raw VPS/SPS/PPS NAL bytes (first occurrence of each type).
    let mut seen: [bool; 3] = [false; 3];
    let mut offset = 0usize;
    while let Some((start, code_len)) = vacc_parser::nal::find_next_start_code(data, offset) {
        let next = vacc_parser::nal::find_next_start_code(data, start + code_len);
        let end = match next {
            Some((s, 4)) => s + 1,
            Some((s, _)) => s,
            None => data.len(),
        };
        let nal = &data[start + code_len..end];
        if let Some((_, t, _, _)) = vacc_parser::nal::parse_h265_nal_header(nal) {
            match t {
                32 if !seen[0] => {
                    seen[0] = true;
                    println!("SET vps {}", hex(nal));
                }
                33 if !seen[1] => {
                    seen[1] = true;
                    println!("SET sps {}", hex(nal));
                }
                34 if !seen[2] => {
                    seen[2] = true;
                    println!("SET pps {}", hex(nal));
                }
                _ => {}
            }
        }
        offset = match next {
            Some((s, _)) => s,
            None => break,
        };
    }
}

/// Minimal IVF demuxer (header + fixed-size packets).
fn ivf_packets(data: &[u8]) -> Vec<Vec<u8>> {
    assert!(
        data.len() >= 32 && &data[0..4] == b"DKIF",
        "not an IVF file"
    );
    let mut out = Vec::new();
    let mut off = 32usize;
    while off + 12 <= data.len() {
        let size = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        off += 12;
        if size == 0 || off + size > data.len() {
            break;
        }
        out.push(data[off..off + size].to_vec());
        off += size;
    }
    out
}

fn dump_vp9(data: &[u8], max_frames: usize) {
    let mut parser = vacc_parser::vp9::Vp9Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeVp9))
        .expect("init");
    let packets = ivf_packets(data);
    let mut dpb = Vp9Dpb::new(8);
    for (i, pkt) in packets.into_iter().enumerate() {
        if i >= max_frames {
            break;
        }
        let payload = vacc_parser::vp9::Vp9Parser::skip_superframe_index(&pkt);
        let fd = parser
            .parse_frame(payload)
            .unwrap_or_else(|e| panic!("frame {i}: {e}"));
        let r3 = [
            fd.ref_frame_idx[0],
            fd.ref_frame_idx[1],
            fd.ref_frame_idx[2],
        ];
        let refs = dpb.reference_slots(fd.frame_is_intra, &r3);
        let slot = if fd.show_existing_frame {
            -1
        } else {
            dpb.choose_output_slot()
        };
        println!(
            "PIC {i}\n  intra={} show_existing={} map={} w={} h={} ctx={} refresh={} bias={} \
             refs=[{}:{}] slot={}",
            u32::from(fd.frame_is_intra),
            u32::from(fd.show_existing_frame),
            fd.frame_to_show_map_idx,
            fd.frame_width,
            fd.frame_height,
            fd.picture_info.frame_context_idx,
            fd.picture_info.refresh_frame_flags,
            fd.picture_info.ref_frame_sign_bias_mask,
            fd.ref_frame_idx
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(" "),
            refs.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(" "),
            slot,
        );
        if !fd.show_existing_frame {
            dpb.commit_frame(fd.picture_info.refresh_frame_flags, slot);
        }
        let raw = &payload[..64.min(payload.len())];
        println!("  RAW {}", hex(raw));
    }
    eprintln!("vp9: {} frames", max_frames);
}

/// Read a leb128 value at `off`; returns (value, new_off).
fn leb128(data: &[u8], mut off: usize) -> (usize, usize) {
    let mut value = 0usize;
    let mut shift = 0;
    loop {
        let b = *data.get(off).expect("leb128 truncated");
        off += 1;
        value |= ((b & 0x7F) as usize) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (value, off)
}

/// Walk one IVF packet's OBUs; invoke `f(type, payload)` for each OBU.
/// Header layout mirrors `Av1Parser::parse_obu_header`:
/// forbidden(1) type(4) extension(1) has_size_field(1) reserved(1),
/// + extension byte when set, + leb128 payload size when has_size_field.
fn walk_obus(pkt: &[u8], mut f: impl FnMut(u8, Option<&[u8]>)) {
    let mut off = 0usize;
    while off < pkt.len() {
        let b0 = pkt[off];
        let obu_type = (b0 >> 3) & 0x0F;
        let ext = b0 & 0x04 != 0;
        let has_size = b0 & 0x02 != 0;
        off += 1 + ext as usize;
        if !has_size {
            f(obu_type, None);
            continue;
        }
        let (size, payload_off) = leb128(pkt, off);
        off = payload_off + size;
        f(obu_type, Some(&pkt[payload_off..payload_off + size]));
    }
}

fn dump_av1(data: &[u8], max_frames: usize) {
    let mut parser = vacc_parser::av1::Av1Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeAv1))
        .expect("init");
    let packets = ivf_packets(data);
    let mut sps = None;
    let mut pic_idx = 0usize;
    const RAW_PREFIX: usize = 96;

    for pkt in packets {
        walk_obus(&pkt, |obu_type, payload| {
            let Some(payload) = payload else { return }; // temporal delimiter etc.
            if obu_type == 1 && sps.is_none() {
                // Sequence header OBU.
                if let Ok(sh) = parser.parse_sequence_header_obu(payload) {
                    sps = Some(sh);
                    let sh = sps.as_ref().unwrap();
                    println!("SET seqheader {}", hex(payload));
                    println!(
                        "SPS profile={} level={} still={} w={} h={} order_hint_bits={}",
                        sh.profile,
                        sh.level,
                        u32::from(sh.reduced_still_picture_header),
                        sh.max_frame_width_minus_1 as u32 + 1,
                        sh.max_frame_height_minus_1 as u32 + 1,
                        sh.order_hint_bits_minus1,
                    );
                }
            } else if (obu_type == 3 || obu_type == 6) && sps.is_some() {
                // FrameHeader OBU (type 3, show-existing frames) or full
                // Frame OBU (type 6).
                let sps = sps.clone().unwrap();
                match parser.parse_frame_header(payload, &sps) {
                    Ok(fh) => {
                        let pic = pic_idx;
                        pic_idx += 1;
                        if pic >= max_frames {
                            return;
                        }
                        // Self-check: the RAW prefix (first RAW_PREFIX payload
                        // bytes) must re-parse to identical key fields.
                        let prefix = &payload[..RAW_PREFIX.min(payload.len())];
                        let ok = parser
                            .parse_frame_header(prefix, &sps)
                            .map(|p| {
                                p.frame_type == fh.frame_type
                                    && p.show_existing_frame == fh.show_existing_frame
                                    && p.frame_to_show_map_idx == fh.frame_to_show_map_idx
                                    && p.ref_frame_idx == fh.ref_frame_idx
                                    && p.order_hint == fh.order_hint
                                    && p.refresh_frame_flags == fh.refresh_frame_flags
                                    && p.show_frame == fh.show_frame
                            })
                            .unwrap_or(false);
                        println!(
                            "PIC {pic}\n  type={} show_existing={} map={} w={} h={} oh={} ohb={} \
                             refresh={} show={} refs=[{}] raw_ok={}",
                            fh.frame_type,
                            u32::from(fh.show_existing_frame),
                            fh.frame_to_show_map_idx,
                            fh.frame_width,
                            fh.frame_height,
                            fh.order_hint,
                            sps.order_hint_bits_minus1,
                            fh.refresh_frame_flags,
                            u32::from(fh.show_frame),
                            fh.ref_frame_idx
                                .iter()
                                .map(|v| v.to_string())
                                .collect::<Vec<_>>()
                                .join(" "),
                            u32::from(ok),
                        );
                        if ok {
                            println!("  RAW {}", hex(prefix));
                        }
                    }
                    Err(e) => eprintln!("frame header parse error: {e}"),
                }
            }
        });
    }
    eprintln!("av1: {pic_idx} frames");
}

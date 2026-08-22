//! Diagnostic: extract AV1 Frame OBUs (Vulkan EB128 logic) and parse each
//! header with Av1Parser. Prints (idx, payload_start, payload_size,
//! frame_header_size, ...) for correlation with the cuvid param dump.
//!
//! Usage: cargo run --release --example diag_av1_obu -- [ivf_file] [max]

use vk_video_core::picture::Av1Sps;
use vk_video_core::VideoCodec;
use vk_video_parser::av1::Av1Parser;
use vk_video_parser::{DetectedVideoFormat, VideoParser};

/// Replicate the Vulkan decoder's extract_frame_obus_from_packet (EB128, no
/// temporal-id skip).
fn extract_frame_obus(packet: &[u8]) -> Vec<(u8, usize, usize)> {
    let mut obus = Vec::new();
    let mut pos = 0;
    let n = packet.len();
    while pos < n.saturating_sub(1) {
        let first = packet[pos];
        let obu_type = (first >> 3) & 0x0F;
        let ext = (first >> 2) & 1;
        let has_size = (first >> 1) & 1 != 0;
        let header_size = 1 + ext as usize;
        if has_size && pos + header_size < n {
            let mut size: usize = 0;
            let mut shift = 0;
            let mut size_pos = pos + header_size;
            loop {
                if size_pos >= n {
                    break;
                }
                let b = packet[size_pos];
                size |= ((b & 0x7F) as usize) << shift;
                shift += 7;
                size_pos += 1;
                if b & 0x80 == 0 {
                    break;
                }
            }
            let is_se = obu_type == 3
                && size > 0
                && size_pos < n
                && (packet[size_pos] & 0x80) != 0;
            let keep = obu_type == 1 || obu_type == 6 || is_se;
            if keep {
                let ps = size_pos;
                let pe = (ps + size).min(n);
                obus.push((obu_type, ps, pe - ps));
            }
            let next = size_pos + size;
            pos = if next > pos { next } else { size_pos + 1 };
        } else {
            pos += header_size.max(1);
        }
    }
    obus
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "assets/big_buck_bunny_av1.ivf".to_string());
    let max: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(40);

    let data = std::fs::read(&path).expect("read file");
    assert_eq!(&data[0..4], b"DKIF");
    let mut parser = Av1Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeAv1))
        .expect("init");

    let mut sps: Option<Av1Sps> = None;
    let mut off = 32usize;
    let mut idx = 0u32;
    while off + 12 <= data.len() && (idx as usize) < max {
        let sz = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        let payload = &data[off + 12..off + 12 + sz];
        for (typ, ps, psize) in extract_frame_obus(payload) {
            let obu = &payload[ps..ps + psize];
            if typ == 1 {
                // SequenceHeader
                if sps.is_none() {
                    match parser.parse_sequence_header_obu(obu) {
                        Ok(s) => {
                            println!(
                                "SPS profile={} maxw-1={} maxh-1={} ohb-1={} cdef={} superres={} restoration={} warped={} refmvs={} sct={} intmv={} subs={}/{} mono={} highbit={} 12bit={}",
                                s.profile,
                                s.max_frame_width_minus_1,
                                s.max_frame_height_minus_1,
                                s.order_hint_bits_minus1,
                                s.enable_cdef,
                                s.enable_superres,
                                s.enable_restoration,
                                s.enable_warped_motion,
                                s.enable_ref_frame_mvs,
                                s.seq_force_screen_content_tools,
                                s.seq_force_integer_mv,
                                s.subsampling_x,
                                s.subsampling_y,
                                s.mono_chrome,
                                s.high_bitdepth,
                                s.twelve_bit
                            );
                            sps = Some(s);
                        }
                        Err(e) => println!("SPS PARSE_ERR={}", e),
                    }
                }
                continue;
            }
            match &sps {
                Some(s) => match parser.parse_frame_header(obu, s) {
                    Ok(fh) => {
                        println!(
                            "F{} type={} ps={} psize={} hdr={} tiles={}({}x{}) oh={} refresh={:08b} refidx={:?} prim={} show={} ftype={} w={}x{} rw={}x{}",
                            idx,
                            typ,
                            ps,
                            psize,
                            fh.frame_header_size,
                            fh.tile_count,
                            fh.tile_cols,
                            fh.tile_rows,
                            fh.order_hint,
                            fh.refresh_frame_flags,
                            fh.ref_frame_idx,
                            fh.primary_ref_frame,
                            fh.show_frame,
                            fh.frame_type,
                            fh.frame_width,
                            fh.frame_height,
                            fh.render_width,
                            fh.render_height
                        );
                        idx += 1;
                    }
                    Err(e) => {
                        println!("F{} type={} ps={} psize={} PARSE_ERR={}", idx, typ, ps, psize, e);
                        idx += 1;
                    }
                },
                None => println!("F{} type={} ps={} psize={} NO_SPS", idx, typ, ps, psize),
            }
        }
        off += 12 + sz;
    }
}

//! Diagnostic: parse an HEVC stream with H265Parser and print per-picture
//! POC / RPS info to derive cuvid NumBitsForShortTermRPSInSlice and DPB rules.
use vk_video_core::picture::H265Sps;
use vk_video_parser::{h265::H265Parser, BitstreamPacket, ParseResult, VideoParser};

fn ue_bits(v: u32) -> u32 {
    if v == 0 {
        return 1;
    }
    let n = v + 1;
    let k = 32 - n.leading_zeros(); // bits in n
    2 * k - 1
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let maxp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12);
    let data = std::fs::read(path).expect("read");

    let mut parser = H265Parser::new();
    parser
        .init(&vk_video_parser::DetectedVideoFormat::new(
            vk_video_core::codec::VideoCodec::DecodeH265,
        ))
        .unwrap();

    let pkt = BitstreamPacket::new(data);
    let mut pic = 0usize;
    loop {
        match parser.parse(&pkt) {
            Ok(ParseResult::ParameterSet { sps, .. }) => {
                if let Some(b) = sps {
                    if let Some(s) = b.downcast_ref::<H265Sps>() {
                        println!(
                            "[SPS] w={} h={} log2maxpoc={} maxref={} numstrps={} lt={} tmvp={}",
                            s.pic_width_in_luma_samples,
                            s.pic_height_in_luma_samples,
                            s.log2_max_pic_order_cnt_lsb_minus4,
                            s.max_num_ref_frames,
                            s.num_short_term_ref_pic_sets,
                            s.long_term_ref_pics_present_flag,
                            s.sps_temporal_mvp_enabled_flag
                        );
                    }
                }
            }
            Ok(ParseResult::Slice { slices, .. }) => {
                if let Some(vk_video_parser::SliceHeader::H265(info)) =
                    slices[0].slice_header.as_ref()
                {
                    let rps = info.slice_strps.as_ref();
                    // recover raw deltas from cumulative offsets
                    let mut raw0: Vec<i32> = vec![
                        0;
                        info.slice_strps
                            .as_ref()
                            .map(|r| r.num_negative_pics as usize)
                            .unwrap_or(0)
                    ];
                    let mut raw1: Vec<i32> = vec![
                        0;
                        info.slice_strps
                            .as_ref()
                            .map(|r| r.num_positive_pics as usize)
                            .unwrap_or(0)
                    ];
                    let mut refpoc0: Vec<i32> = vec![];
                    let mut refpoc1: Vec<i32> = vec![];
                    if let Some(r) = rps {
                        let mut cum_prev: i32 = 0;
                        for i in 0..r.num_negative_pics as usize {
                            let stored = r.delta_poc_s0_minus1[i] as i32;
                            let signed = if stored > 32767 {
                                stored - 65536
                            } else {
                                stored
                            };
                            let raw = cum_prev - signed; // = (raw_delta+1)
                            raw0[i] = raw - 1;
                            refpoc0.push(info.curr_pic_order_cnt_val + signed);
                            cum_prev = signed;
                        }
                        let mut cum_prev: i32 = 0;
                        for i in 0..r.num_positive_pics as usize {
                            let stored = r.delta_poc_s1_minus1[i] as i32;
                            let signed = stored;
                            let raw = signed - cum_prev;
                            raw1[i] = raw - 1;
                            refpoc1.push(info.curr_pic_order_cnt_val + signed);
                            cum_prev = signed;
                        }
                    }
                    // bit counts under different models
                    let (nn, np) = (
                        rps.map(|r| r.num_negative_pics as u32).unwrap_or(0),
                        rps.map(|r| r.num_positive_pics as u32).unwrap_or(0),
                    );
                    let sum_raw0: u32 = raw0.iter().map(|&d| ue_bits((d + 1) as u32)).sum();
                    let sum_raw1: u32 = raw1.iter().map(|&d| ue_bits((d + 1) as u32)).sum();
                    let model_a = ue_bits(nn) + sum_raw0 + nn + ue_bits(np) + sum_raw1 + np; // with used flags
                    let model_b = ue_bits(nn) + sum_raw0 + ue_bits(np) + sum_raw1; // no flags
                    println!(
                        "PIC {pic}: type={} poc={} idr={} rap={} ref={} nn={} np={} raw0={:?} raw1={:?} ref0={:?} ref1={:?} bitsA={} bitsB={} hbs={} nal={}",
                        info.slice_type, info.curr_pic_order_cnt_val, info.is_idr,
                        info.is_rap, info.is_reference, nn, np, raw0, raw1,
                        refpoc0, refpoc1, model_a, model_b, info.header_bit_size,
                        slices[0].nal_data.len()
                    );
                    println!(
                        "     q={} cb={} cr={} saoL={} saoC={} deb={:?} beta={} tc={} lfas={} tmvp={} stflag={} stidx={} nref0={} nref1={} cabinit={} mvd1z={} coll0={} ext={}",
                        info.slice_qp_delta, info.slice_cb_qp_offset, info.slice_cr_qp_offset,
                        info.slice_sao_luma_flag, info.slice_sao_chroma_flag,
                        info.slice_deblocking_filter_disabled_flag, info.slice_beta_offset_div2,
                        info.slice_tc_offset_div2, info.slice_loop_filter_across_slices_enabled_flag,
                        info.slice_temporal_mvp_enabled_flag, info.short_term_ref_pic_set_sps_flag,
                        info.short_term_ref_pic_set_idx, info.num_ref_idx_l0_active_minus1,
                        info.num_ref_idx_l1_active_minus1, info.cabac_init_flag,
                        info.mvd_l1_zero_flag, info.collocated_from_l0_flag,
                        info.num_entry_point_offsets
                    );
                }
                pic += 1;
                if pic >= maxp {
                    break;
                }
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Err(e) => {
                eprintln!("parse err: {e}");
                break;
            }
        }
    }
}

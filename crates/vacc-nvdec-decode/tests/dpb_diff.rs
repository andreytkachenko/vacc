//! Differential DPB test: common parser DPB vs private NVDEC DPB.
//!
//! Drives both DPBs with the same stream, mirroring each backend's exact
//! call sequence:
//!   common (SW/VAAPI): picture_start -> build_ref_lists
//!                      -> prepare_current_avoiding -> commit_current
//!   private (NVDEC):   apply_idr_reset (pre) -> to_cuvid_dpb_entries (pre-decode)
//!                      -> add_frame (post-decode) -> apply_mmco_ops (post-add)
//!
//! Invariants per picture:
//!   1. What the private DPB PRESENTS to cuvid (valid+reference entries,
//!      i.e. the state the driver uses to rebuild RefPicList0/1) must equal
//!      the common DPB's reference set after the PREVIOUS picture's commit.
//!   2. The private DPB's post-update reference set must equal the common
//!      DPB's post-commit reference set.
//!
//! Run: `cargo test -p vacc-nvdec-decode --test dpb_diff` (optionally with
//! DPB_DIFF_STREAM=/path/to/stream.h264).

use std::fs;

use vacc_core::picture::{H264Pps, H264Sps};
use vacc_parser::bitstream::BitstreamPacket;
use vacc_parser::h264::SliceHeader as H264SliceHeader;
use vacc_parser::h264_dpb::{H264Dpb, H264MmcoCommand, MARKING_UNUSED};
use vacc_parser::h264_poc::PocCalculator;
use vacc_parser::{ParseResult, SliceHeader, VideoParser};
use vacc_nvdec_decode::dpb::NvdecDpbManager;

type RefSet = Vec<(u32, i32)>; // (frame_num, poc)

fn sorted(set: &RefSet) -> Vec<(u32, i32)> {
    let mut v = set.clone();
    v.sort();
    v
}

fn mmco_commands(slh: &H264SliceHeader) -> Vec<H264MmcoCommand> {
    slh.dec_ref_pic_marking
        .iter()
        .map(|op| match op.memory_management_control_operation {
            1 => H264MmcoCommand::UnmarkShortTerm {
                difference_of_pic_nums_minus1: op.value,
            },
            2 => H264MmcoCommand::UnmarkLongTerm {
                long_term_frame_idx: op.value,
            },
            3 => H264MmcoCommand::AssignLongTerm {
                difference_of_pic_nums_minus1: op.value,
                long_term_frame_idx: op.value2,
            },
            4 => H264MmcoCommand::AssignLongTermToCurrent {
                long_term_frame_idx: op.value,
            },
            5 => H264MmcoCommand::SetMaxLongTermFrameIdx {
                max_long_term_frame_idx_plus1: op.value,
            },
            6 => H264MmcoCommand::UnmarkAll,
            _ => H264MmcoCommand::UnmarkAll,
        })
        .collect()
}

fn fmt_set(set: &RefSet) -> String {
    let mut v = set.clone();
    v.sort();
    v.iter()
        .map(|(fn_, poc)| format!("fn{fn_}/poc{poc}"))
        .collect::<Vec<_>>()
        .join(",")
}

#[test]
fn common_dpb_matches_nvdec_private_dpb() {
    let path = std::env::var("DPB_DIFF_STREAM").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/samples/h264_main.h264").to_string()
    });
    let data = fs::read(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    println!("stream: {path} ({} bytes)", data.len());

    let mut parser = vacc_parser::h264::H264Parser::new();
    let mut poc_calc = PocCalculator::new();
    let mut sps: Option<H264Sps> = None;
    let mut _pps: Option<H264Pps> = None;
    let mut common: Option<H264Dpb> = None;
    let mut priv_dpb: Option<NvdecDpbManager> = None;

    let mut offset = 0usize;
    let mut pic = 0u32;
    let mut prev_common_refs: Option<RefSet> = None;
    let mut first_mismatch: Option<String> = None;
    // Surface-lifecycle simulation state (mirrors the decoder's reorder map
    // and POC-wrap tracker).
    let mut reorder: std::collections::BTreeMap<i32, (i32, i32)> =
        std::collections::BTreeMap::new();
    let mut uw_prev: Option<i32> = None;
    let mut uw_cycle = 0i32;
    let mut poc_period = 0i32;

    while offset < data.len() {
        let remaining = &data[offset..];
        let packet = BitstreamPacket::new(remaining.to_vec());
        match parser.parse(&packet) {
            Ok(ParseResult::ParameterSet { sps: b_sps, pps: b_pps, .. }) => {
                if let Some(b) = b_sps {
                    let new_sps = b.downcast_ref::<H264Sps>().expect("SPS type").clone();
                    sps = Some(new_sps.clone());
                    let max_dpb = new_sps.max_num_ref_frames.clamp(1, 16) as usize;
                    let n_slots = max_dpb.max(4) + 4;
                    common = Some(H264Dpb::new(
                        n_slots,
                        n_slots,
                        (new_sps.max_num_ref_frames.min(n_slots as u32)).max(1),
                        new_sps.max_frame_num,
                    ));
                    if let Some(vui) = &new_sps.vui {
                        common.as_mut().unwrap().set_max_num_reorder_frames(
                            vui.max_num_reorder_frames as u32,
                        );
                    }
                    // Mirror NvdecH264Decoder (decoder.rs): decode surface count
                    // is max_num_ref_frames + 4 clamped to [5, 32], and the DPB
                    // manager wraps CurrPicIdx at exactly that value.
                    let num_surfaces = new_sps.max_num_ref_frames.saturating_add(4).clamp(5, 32);
                    let mut p = NvdecDpbManager::new(16); // default; updated below
                    p.set_max_decode_surfaces(num_surfaces as i32);
                    p.set_max_frame_num(new_sps.max_frame_num);
                    p.set_max_dpb_size(new_sps.max_num_ref_frames as usize);
                    priv_dpb = Some(p);
                    poc_calc.reset();
                    prev_common_refs = None;
                    poc_period = new_sps.max_pic_order_cnt_lsb as i32;
                    reorder.clear();
                    uw_prev = None;
                    uw_cycle = 0;
                    eprintln!(
                        "SPS: num_ref_frames={} max_frame_num={} poc_type={} max_poc_lsb={} log2_max_fn_minus4={}",
                        new_sps.max_num_ref_frames, new_sps.max_frame_num,
                        new_sps.pic_order_cnt_type, new_sps.max_pic_order_cnt_lsb,
                        new_sps.log2_max_frame_num_minus4
                    );
                }
                if let Some(b) = b_pps {
                    _pps = Some(b.downcast_ref::<H264Pps>().expect("PPS type").clone());
                }
                continue;
            }
            Ok(ParseResult::Slice { slices, bytes_consumed }) => {
                offset += bytes_consumed.max(1);
                if slices.is_empty() {
                    continue;
                }
                let slh = match &slices[0].slice_header {
                    Some(SliceHeader::H264(h)) => h,
                    _ => panic!("missing H.264 slice header"),
                };
                if slh.redundant_pic_cnt > 0 {
                    continue; // redundant slice: no DPB update (mirrors SW)
                }
                let sps = sps.as_ref().expect("SPS before first slice");
                let is_idr = slh.nal_unit_type == 5;
                let is_ref = slh.nal_ref_idc > 0;
                if is_idr {
                    poc_calc.reset();
                }
                let poc = poc_calc.calculate(sps, slh, is_ref);
                if pic <= 16 {
                    eprintln!(
                        "pic{pic:3} fn={:3} poc={:3} st={} idr={} ref={} l0m1={} l1m1={} override={} mmco={:?}",
                        slh.frame_num, poc, slh.slice_type % 5, is_idr, is_ref,
                        slh.num_ref_idx_l0_active_minus1, slh.num_ref_idx_l1_active_minus1,
                        slh.num_ref_idx_active_override_flag,
                        slh.dec_ref_pic_marking.iter().map(|o| (o.memory_management_control_operation, o.value)).collect::<Vec<_>>()
                    );
                }
                pic += 1;

                // ---- common DPB (SW/VAAPI sequence) ----
                let mmco = mmco_commands(slh);
                {
                    let c = common.as_mut().expect("common DPB");
                    c.picture_start(
                        slh.frame_num,
                        poc,
                        is_ref,
                        is_idr,
                        slh.long_term_reference,
                        slh.no_output_of_prior_pics_flag,
                        !mmco.is_empty(),
                        mmco,
                    );
                    let lists = c.build_ref_lists(
                        slh.slice_type % 5,
                        slh.num_ref_idx_l0_active_minus1,
                        slh.num_ref_idx_l1_active_minus1,
                        &slh.ref_pic_list_modification_l0,
                        &slh.ref_pic_list_modification_l1,
                    );
                    if pic <= 16 {
                        eprintln!(
                            "      L0=[{}] L1=[{}]",
                            lists.l0.iter().map(|r| r.frame_num.to_string()).collect::<Vec<_>>().join(","),
                            lists.l1.iter().map(|r| r.frame_num.to_string()).collect::<Vec<_>>().join(","),
                        );
                    }
                    let mut used: Vec<usize> = Vec::new();
                    for r in lists.l0.iter().chain(lists.l1.iter()) {
                        if !used.contains(&r.slot) {
                            used.push(r.slot);
                        }
                    }
                    let slot = c.prepare_current_avoiding(&used);
                    c.commit_current(slot);
                }
                let common_refs: RefSet = common
                    .as_ref()
                    .unwrap()
                    .slots
                    .iter()
                    .filter(|s| s.marking != MARKING_UNUSED)
                    .map(|s| (s.frame_num, s.poc))
                    .collect();

                // ---- private NVDEC DPB (decoder.rs sequence) + surface model ----
                let p = priv_dpb.as_mut().expect("private DPB");
                if is_idr {
                    // drain_reorder at IDR: extract every pending frame first.
                    for &(_, s) in reorder.values() {
                        p.mark_unmapped(s);
                        p.mark_extracted(s);
                    }
                    reorder.clear();
                    uw_prev = None;
                    uw_cycle = 0;
                    p.apply_idr_reset(slh.long_term_reference_flag);
                }
                let presented: RefSet = p
                    .to_cuvid_dpb_entries()
                    .iter()
                    .filter(|e| e.PicIdx >= 0)
                    .map(|e| (e.FrameIdx as u32, e.FieldOrderCnt[0]))
                    .collect();

                // Invariant 1: presented state == common state after previous commit.
                if let Some(prev) = &prev_common_refs {
                    if sorted(&presented) != sorted(prev) && first_mismatch.is_none() {
                        first_mismatch = Some(format!(
                            "pic {} (fn={} poc={} is_idr={} is_ref={} slice_type={}): \
                             PRESENTED state diverged\n  presented: [{}]\n  expected : [{}]",
                            pic, slh.frame_num, poc, is_idr, is_ref, slh.slice_type,
                            fmt_set(&presented), fmt_set(prev)
                        ));
                    }
                }

                // Invariant 3 (surface safety): the chosen decode surface must
                // not currently hold another live picture (unextracted frame
                // or valid reference). Overwriting it corrupts that picture's
                // pixels for every future consumer.
                let pre_idx = p.get_next_pic_index();
                if first_mismatch.is_none() {
                    let mut occupants: std::collections::HashMap<i32, Vec<i32>> =
                        std::collections::HashMap::new();
                    for (&_, &(pi, s)) in &reorder {
                        occupants.entry(pi).or_default().push(s);
                    }
                    if std::env::var("DPB_DIFF_DEBUG").is_ok() && pic < 16 {
                        eprintln!(
                            "[dbg] pic={} fn={} poc={} type={:?} pre_idx={} reorder={:?} valid_refs={:?}",
                            pic, slh.frame_num, poc, slh.slice_type, pre_idx,
                            reorder.values().collect::<Vec<_>>(),
                            p.valid_entries()
                                .filter(|e| e.is_reference)
                                .map(|e| (e.pic_index, e.seq, e.frame_num, e.extracted, e.unmapped))
                                .collect::<Vec<_>>()
                        );
                    }
                    for e in p.valid_entries().filter(|e| e.is_reference) {
                        let v = occupants.entry(e.pic_index).or_default();
                        if !v.contains(&e.seq) {
                            v.push(e.seq);
                        }
                    }
                    if let Some(seq) = occupants.get(&pre_idx).and_then(|v| v.first()) {
                        first_mismatch = Some(format!(
                            "pic {} (fn={} poc={}): decode surface {} already holds live \
                             picture seq={} — overwrite corrupts reference",
                            pic, slh.frame_num, poc, pre_idx, seq
                        ));
                    }
                }

                let pic_index = p.add_frame(slh.frame_num, poc, is_ref);
                let seq = p.last_seq();
                assert_eq!(
                    pre_idx, pic_index,
                    "choose_surface differs between pre-decode and add_frame"
                );
                if !is_idr {
                    p.apply_mmco_ops(slh.frame_num, slh);
                }

                // Unwrapped POC (mirrors decoder.rs::unwrapped_poc) + reorder
                // bookkeeping + extraction (mirrors extract_ready_frames; host
                // staging unmaps synchronously during extraction).
                let uw = {
                    if poc_period <= 1 {
                        uw_prev = Some(poc);
                        poc
                    } else {
                        if let Some(prev) = uw_prev {
                            if poc < prev - poc_period / 2 {
                                uw_cycle += 1;
                            } else if poc > prev + poc_period / 2 {
                                uw_cycle -= 1;
                            }
                        }
                        uw_prev = Some(poc);
                        poc + uw_cycle * poc_period
                    }
                };
                reorder.insert(uw, (pic_index, seq));
                while let Some((&k, &(_, s))) = reorder.iter().next() {
                    if k >= uw {
                        break;
                    }
                    p.mark_unmapped(s);
                    p.mark_extracted(s);
                    reorder.remove(&k);
                }

                // Invariant 4: no two live pictures share a decode surface.
                if first_mismatch.is_none() {
                    let mut occupants: std::collections::HashMap<i32, Vec<i32>> =
                        std::collections::HashMap::new();
                    for (&_, &(pi, s)) in &reorder {
                        occupants.entry(pi).or_default().push(s);
                    }
                    for e in p.valid_entries().filter(|e| e.is_reference) {
                        let v = occupants.entry(e.pic_index).or_default();
                        if !v.contains(&e.seq) {
                            v.push(e.seq);
                        }
                    }
                    if let Some((pi, seqs)) = occupants
                        .iter()
                        .find(|(_, v)| v.len() > 1)
                        .map(|(k, v)| (*k, v.clone()))
                    {
                        first_mismatch = Some(format!(
                            "pic {} (fn={} poc={}): surface {} holds multiple live \
                             pictures seqs={:?}",
                            pic, slh.frame_num, poc, pi, seqs
                        ));
                    }
                }

                let priv_refs: RefSet = p
                    .valid_entries()
                    .filter(|e| e.is_reference)
                    .map(|e| (e.frame_num, e.pic_order_cnt))
                    .collect();

                // Invariant 2: post-update private state == common post-commit state.
                if sorted(&priv_refs) != sorted(&common_refs) && first_mismatch.is_none() {
                    first_mismatch = Some(format!(
                        "pic {} (fn={} poc={} is_idr={} is_ref={} slice_type={} mmco={:?}): \
                         POST-UPDATE state diverged\n  private : [{}]\n  common  : [{}]",
                        pic, slh.frame_num, poc, is_idr, is_ref, slh.slice_type,
                        slh.dec_ref_pic_marking.iter().map(|o| (o.memory_management_control_operation, o.value, o.value2)).collect::<Vec<_>>(),
                        fmt_set(&priv_refs), fmt_set(&common_refs)
                    ));
                }

                prev_common_refs = Some(common_refs);
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => {
                // All complete pictures consumed (a held-back tail NAL, if any,
                // is incomplete and irrelevant to DPB state).
                break;
            }
            Err(e) => panic!("parse error at offset {offset}: {e}"),
        }
    }

    println!("parsed {pic} pictures");
    assert!(
        first_mismatch.is_none(),
        "DPB state divergence:\n{}",
        first_mismatch.unwrap()
    );
}

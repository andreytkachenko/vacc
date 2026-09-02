//! Integration test: `H265Parser` + common `H265Dpb` over
//! `assets/big_buck_bunney.h265`.
//!
//! The stream contains B-frame reordering (decode order != display order), a
//! mid-stream CRA with NoRaslOutputFlag=0 at frame 250, and POC wraparound
//! (MaxPicOrderCntLsb=256 < 300 frames). This exercises the full DPB
//! lifecycle: marking, future-use RPS keep-alive, eviction, CRA handling.
//!
//! Checks:
//! - for the first 50 pictures, POC and used short-term ref counts match the
//!   cuvid ground truth (`h265_cref_50.txt`);
//! - for EVERY picture in the stream, every resolved L0/L1 reference matches
//!   a live DPB slot (no missing references) — i.e. the DPB kept-alive
//!   management never frees a picture a future slice needs.

static GT: &str = include_str!("h265_cref_50.txt");

use vacc_core::codec::VideoCodec;
use vacc_core::picture::H265Sps;
use vacc_parser::h265::H265Parser;
use vacc_parser::h265_dpb::{resolve_refs, H265Dpb};
use vacc_parser::{BitstreamPacket, DetectedVideoFormat, ParseResult, SliceHeader, VideoParser};

struct GtPic {
    poc: i32,
    nstb: usize,
    nsta: usize,
}

fn parse_gt() -> Vec<GtPic> {
    let sections: Vec<&str> = GT
        .split("=== PIC ")
        .filter(|s| s.starts_with(|c: char| c.is_ascii_digit()))
        .collect();
    assert_eq!(sections.len(), 50, "expected 50 ground-truth pictures");
    sections
        .iter()
        .map(|sec| {
            let get = |key: &str| -> i64 {
                for l in sec.lines() {
                    for t in l.split_whitespace() {
                        if let Some((k, v)) = t.split_once('=') {
                            if k == key {
                                return v.parse().unwrap();
                            }
                        }
                    }
                }
                panic!("GT: missing field {key}")
            };
            GtPic {
                poc: get("CurrPicOrderCntVal") as i32,
                nstb: get("NumPocStCurrBefore") as usize,
                nsta: get("NumPocStCurrAfter") as usize,
            }
        })
        .collect()
}

#[test]
fn h265_dpb_full_stream_no_missing_refs() {
    let gt = parse_gt();
    // Embedded at compile time: no runtime dependency on the assets tree.
    let data = include_bytes!("../../../assets/big_buck_bunney.h265").to_vec();

    let mut parser = H265Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeH265))
        .expect("init");
    let packet = BitstreamPacket::new(data);

    // 16 slots: the stream's MinDecPicBuffering is 8 (GT SEQUENCE line).
    let mut dpb = H265Dpb::new(16);
    let mut sps: Option<H265Sps> = None;
    let mut pic_idx = 0usize;

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::ParameterSet { sps: s, .. }) => {
                if let Some(s) = s {
                    sps = s.downcast_ref::<H265Sps>().cloned();
                }
            }
            Ok(ParseResult::Slice { slices, .. }) => {
                let SliceHeader::H265(info) = slices[0]
                    .slice_header
                    .as_ref()
                    .expect("slice header parsed")
                else {
                    panic!("expected H265 slice header");
                };
                let sps = sps.as_ref().expect("SPS seen before first picture");
                let pic = pic_idx;

                // --- GT comparison for the first 50 pictures ---
                if pic < gt.len() {
                    let g = &gt[pic];
                    assert_eq!(
                        info.curr_pic_order_cnt_val, g.poc,
                        "pic {pic}: POC mismatch (parser={} gt={})",
                        info.curr_pic_order_cnt_val, g.poc
                    );
                    let resolved = resolve_refs(sps, info);
                    assert_eq!(
                        resolved.st_curr_before.iter().filter(|r| r.used).count(),
                        g.nstb,
                        "pic {pic}: used S0 count mismatch"
                    );
                    assert_eq!(
                        resolved.st_curr_after.iter().filter(|r| r.used).count(),
                        g.nsta,
                        "pic {pic}: used S1 count mismatch"
                    );
                }

                // --- DPB lifecycle: no missing references, ever ---
                let slot = dpb.picture_start(sps, info, info.is_reference);
                let lists = dpb.build_ref_lists();
                for list in [&lists.l0, &lists.l1] {
                    for r in list {
                        assert_ne!(
                            r.slot, -1,
                            "pic {pic}: missing reference POC {} (DPB lost a live ref)",
                            r.poc
                        );
                    }
                }
                dpb.commit_current(slot);

                pic_idx += 1;
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Err(e) => panic!("parse error: {e}"),
        }
    }

    assert!(
        pic_idx >= 300,
        "expected >= 300 pictures in the stream, got {pic_idx}"
    );
    eprintln!("OK: {pic_idx} pictures decoded through the common H265 DPB with no missing references (first 50 GT-checked)");
}

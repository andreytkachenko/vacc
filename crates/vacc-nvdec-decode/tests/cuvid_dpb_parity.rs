//! Parity test: Rust parser + DPB manager + POC calculator vs NVIDIA cuvidParser.
//!
//! This reads `assets/bframe_test.h264`, parses it with `vacc-parser`, drives
//! the `NvdecDpbManager` exactly the way the production decoder does (IDR reset
//! before `add_frame`, DPB entries captured before `add_frame`, non-IDR MMCO
//! applied after `add_frame`), and compares per picture:
//!
//!   * the computed POC (`CurrFieldOrderCnt`), and
//!   * the DPB reference set — the multiset of
//!     `(FrameIdx, FieldOrderCnt, used_for_reference, not_existing, is_long_term)`
//!     over the non-empty DPB slots —
//!
//! against ground-truth data captured from NVIDIA's own `cuvidParser`
//! (`CUVIDH264PICPARAMS.dpb[]`), hardcoded below.
//!
//! The reference set is compared slot-independently (sorted) because the Rust
//! decoder and cuvidParser may assign decode-surface indices (`PicIdx`)
//! differently while still describing the same set of reference pictures. What
//! must match exactly is *which* pictures are references (and their flags) and
//! the POC of every picture.
//!
//! The sample has 4 GOPs of 15 pictures (IDRs at pic 0/15/30/45) with B-frame
//! reordering and up to 4 live references, so it exercises IDR resets, POC
//! wraparound, reference-list evolution and DPB eviction.

use vacc_nvdec_decode::dpb::NvdecDpbManager;
use vacc_nvdec_decode::poc::PocCalculator;
use vacc_parser::{h264::H264Parser, BitstreamPacket, ParseResult, VideoParser};

/// Project root (parent of the vacc-nvdec-decode crate).
const PROJECT_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
const SAMPLE: &str = "assets/bframe_test.h264";
/// Number of pictures to validate against the hardcoded ground truth.
const NUM_PICS: usize = 50;
/// min_num_decode_surfaces reported by cuvidParser for the sample.
const MAX_DECODE_SURFACES: i32 = 5;

/// A single DPB reference: (frame_idx, foc, used_for_reference, not_existing, is_long_term).
type Ref = (i32, i32, i32, i32, i32);

// Auto-generated from an NVIDIA cuvidParser dump of assets/bframe_test.h264
// (regenerated 2026-09-01 — the previous capture predated the current asset
// revision and disagreed with the bitstream's actual decode order).
// Each entry: (poc, &[ (frame_idx, foc, used_for_reference, not_existing, is_long_term), ... ])
// Refs sorted for slot-independent comparison.
const CUIDATA: &[(i32, &[Ref])] = &[
    (0, &[]),
    (2, &[(0, 0, 3, 0, 0)]),
    (4, &[(0, 0, 3, 0, 0), (1, 2, 3, 0, 0)]),
    (6, &[(0, 0, 3, 0, 0), (1, 2, 3, 0, 0), (2, 4, 3, 0, 0)]),
    (
        8,
        &[
            (0, 0, 3, 0, 0),
            (1, 2, 3, 0, 0),
            (2, 4, 3, 0, 0),
            (3, 6, 3, 0, 0),
        ],
    ),
    (
        10,
        &[
            (1, 2, 3, 0, 0),
            (2, 4, 3, 0, 0),
            (3, 6, 3, 0, 0),
            (4, 8, 3, 0, 0),
        ],
    ),
    (
        12,
        &[
            (2, 4, 3, 0, 0),
            (3, 6, 3, 0, 0),
            (4, 8, 3, 0, 0),
            (5, 10, 3, 0, 0),
        ],
    ),
    (
        14,
        &[
            (3, 6, 3, 0, 0),
            (4, 8, 3, 0, 0),
            (5, 10, 3, 0, 0),
            (6, 12, 3, 0, 0),
        ],
    ),
    (
        16,
        &[
            (4, 8, 3, 0, 0),
            (5, 10, 3, 0, 0),
            (6, 12, 3, 0, 0),
            (7, 14, 3, 0, 0),
        ],
    ),
    (
        18,
        &[
            (5, 10, 3, 0, 0),
            (6, 12, 3, 0, 0),
            (7, 14, 3, 0, 0),
            (8, 16, 3, 0, 0),
        ],
    ),
    (
        20,
        &[
            (6, 12, 3, 0, 0),
            (7, 14, 3, 0, 0),
            (8, 16, 3, 0, 0),
            (9, 18, 3, 0, 0),
        ],
    ),
    (
        22,
        &[
            (7, 14, 3, 0, 0),
            (8, 16, 3, 0, 0),
            (9, 18, 3, 0, 0),
            (10, 20, 3, 0, 0),
        ],
    ),
    (
        28,
        &[
            (8, 16, 3, 0, 0),
            (9, 18, 3, 0, 0),
            (10, 20, 3, 0, 0),
            (11, 22, 3, 0, 0),
        ],
    ),
    (
        24,
        &[
            (9, 18, 3, 0, 0),
            (10, 20, 3, 0, 0),
            (11, 22, 3, 0, 0),
            (12, 28, 3, 0, 0),
        ],
    ),
    (
        26,
        &[(11, 22, 3, 0, 0), (12, 28, 3, 0, 0), (13, 24, 3, 0, 0)],
    ),
    (0, &[]),
    (6, &[(0, 0, 3, 0, 0)]),
    (2, &[(0, 0, 3, 0, 0), (1, 6, 3, 0, 0)]),
    (4, &[(0, 0, 3, 0, 0), (1, 6, 3, 0, 0), (2, 2, 3, 0, 0)]),
    (12, &[(0, 0, 3, 0, 0), (1, 6, 3, 0, 0), (2, 2, 3, 0, 0)]),
    (
        8,
        &[
            (0, 0, 3, 0, 0),
            (1, 6, 3, 0, 0),
            (2, 2, 3, 0, 0),
            (3, 12, 3, 0, 0),
        ],
    ),
    (10, &[(1, 6, 3, 0, 0), (3, 12, 3, 0, 0), (4, 8, 3, 0, 0)]),
    (18, &[(1, 6, 3, 0, 0), (3, 12, 3, 0, 0), (4, 8, 3, 0, 0)]),
    (
        14,
        &[
            (1, 6, 3, 0, 0),
            (3, 12, 3, 0, 0),
            (4, 8, 3, 0, 0),
            (5, 18, 3, 0, 0),
        ],
    ),
    (16, &[(3, 12, 3, 0, 0), (5, 18, 3, 0, 0), (6, 14, 3, 0, 0)]),
    (24, &[(3, 12, 3, 0, 0), (5, 18, 3, 0, 0), (6, 14, 3, 0, 0)]),
    (
        20,
        &[
            (3, 12, 3, 0, 0),
            (5, 18, 3, 0, 0),
            (6, 14, 3, 0, 0),
            (7, 24, 3, 0, 0),
        ],
    ),
    (22, &[(5, 18, 3, 0, 0), (7, 24, 3, 0, 0), (8, 20, 3, 0, 0)]),
    (28, &[(5, 18, 3, 0, 0), (7, 24, 3, 0, 0), (8, 20, 3, 0, 0)]),
    (26, &[(7, 24, 3, 0, 0), (8, 20, 3, 0, 0), (9, 28, 3, 0, 0)]),
    (0, &[]),
    (6, &[(0, 0, 3, 0, 0)]),
    (2, &[(0, 0, 3, 0, 0), (1, 6, 3, 0, 0)]),
    (4, &[(0, 0, 3, 0, 0), (1, 6, 3, 0, 0), (2, 2, 3, 0, 0)]),
    (12, &[(0, 0, 3, 0, 0), (1, 6, 3, 0, 0), (2, 2, 3, 0, 0)]),
    (
        8,
        &[
            (0, 0, 3, 0, 0),
            (1, 6, 3, 0, 0),
            (2, 2, 3, 0, 0),
            (3, 12, 3, 0, 0),
        ],
    ),
    (10, &[(1, 6, 3, 0, 0), (3, 12, 3, 0, 0), (4, 8, 3, 0, 0)]),
    (18, &[(1, 6, 3, 0, 0), (3, 12, 3, 0, 0), (4, 8, 3, 0, 0)]),
    (
        14,
        &[
            (1, 6, 3, 0, 0),
            (3, 12, 3, 0, 0),
            (4, 8, 3, 0, 0),
            (5, 18, 3, 0, 0),
        ],
    ),
    (16, &[(3, 12, 3, 0, 0), (5, 18, 3, 0, 0), (6, 14, 3, 0, 0)]),
    (24, &[(3, 12, 3, 0, 0), (5, 18, 3, 0, 0), (6, 14, 3, 0, 0)]),
    (
        20,
        &[
            (3, 12, 3, 0, 0),
            (5, 18, 3, 0, 0),
            (6, 14, 3, 0, 0),
            (7, 24, 3, 0, 0),
        ],
    ),
    (22, &[(5, 18, 3, 0, 0), (7, 24, 3, 0, 0), (8, 20, 3, 0, 0)]),
    (28, &[(5, 18, 3, 0, 0), (7, 24, 3, 0, 0), (8, 20, 3, 0, 0)]),
    (26, &[(7, 24, 3, 0, 0), (8, 20, 3, 0, 0), (9, 28, 3, 0, 0)]),
    (0, &[]),
    (6, &[(0, 0, 3, 0, 0)]),
    (2, &[(0, 0, 3, 0, 0), (1, 6, 3, 0, 0)]),
    (4, &[(0, 0, 3, 0, 0), (1, 6, 3, 0, 0), (2, 2, 3, 0, 0)]),
    (12, &[(0, 0, 3, 0, 0), (1, 6, 3, 0, 0), (2, 2, 3, 0, 0)]),
];

/// Collect the non-empty DPB reference set from cuvid entries, sorted so the
/// comparison is independent of slot (`PicIdx`) assignment.
fn ref_set(entries: &[vacc_nvdec_decode::ffi::CUVIDH264DPBENTRY; 16]) -> Vec<Ref> {
    let mut refs: Vec<Ref> = entries
        .iter()
        .filter(|e| e.PicIdx != -1)
        .map(|e| {
            (
                e.FrameIdx,
                e.FieldOrderCnt[0],
                e.used_for_reference,
                e.not_existing,
                e.is_long_term,
            )
        })
        .collect();
    refs.sort();
    refs
}

#[test]
fn test_parser_dpb_poc_matches_cuvid_parser() {
    let data = std::fs::read(format!("{}/{}", PROJECT_ROOT, SAMPLE))
        .unwrap_or_else(|e| panic!("failed to read {}: {}", SAMPLE, e));

    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data);

    let mut poc_calc = PocCalculator::new();
    let mut dpb: Option<NvdecDpbManager> = None;
    let mut pic_idx = 0usize;

    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::ParameterSet { sps: sps_box, .. }) => {
                // Initialize the DPB manager from the first SPS, mirroring the
                // production decoder.
                if dpb.is_none() {
                    if let Some(sb) = sps_box {
                        if let Some(sps) = sb.downcast_ref::<vacc_core::picture::H264Sps>() {
                            let mut m = NvdecDpbManager::new(sps.max_num_ref_frames as usize);
                            m.set_max_frame_num(sps.max_frame_num);
                            m.set_max_dpb_size(sps.max_num_ref_frames as usize);
                            m.set_max_decode_surfaces(MAX_DECODE_SURFACES);
                            dpb = Some(m);
                        }
                    }
                }
            }
            Ok(ParseResult::Slice { slices, .. }) => {
                if slices.is_empty() {
                    break;
                }
                let slh = match &slices[0].slice_header {
                    Some(vacc_parser::SliceHeader::H264(h)) => h.clone(),
                    _ => break,
                };
                let dpb = dpb
                    .as_mut()
                    .expect("DPB not initialized before first slice");

                let is_idr = slh.nal_unit_type == 5;
                let is_reference = slh.nal_ref_idc > 0;

                // POC (reset the calculator on IDR, exactly like the decoder).
                let poc = {
                    let sps = parser.active_sps().expect("no active SPS");
                    if is_idr {
                        poc_calc.reset();
                    }
                    poc_calc.calculate(sps, &slh, is_reference)
                };

                // IDR reset BEFORE the picture is added.
                if is_idr {
                    dpb.apply_idr_reset(slh.long_term_reference_flag);
                }

                // DPB entries as seen by the decoder for THIS picture: captured
                // before add_frame (references only, current pic not yet added).
                let entries = dpb.to_cuvid_dpb_entries();

                if pic_idx < NUM_PICS {
                    let (exp_poc, exp_refs) = CUIDATA[pic_idx];
                    assert_eq!(
                        poc, exp_poc,
                        "pic {}: POC mismatch — got {}, cuvidParser {}",
                        pic_idx, poc, exp_poc
                    );
                    let got = ref_set(&entries);
                    let mut exp: Vec<Ref> = exp_refs.to_vec();
                    exp.sort();
                    assert_eq!(
                        got, exp,
                        "pic {}: DPB reference set mismatch\n  got {:?}\n  cuvidParser {:?}",
                        pic_idx, got, exp
                    );
                }

                // Advance the DPB exactly like the decoder: add the current
                // frame, then apply non-IDR MMCO (affects subsequent pictures).
                dpb.add_frame(slh.frame_num, poc, is_reference);
                if !is_idr {
                    dpb.apply_mmco_ops(slh.frame_num, &slh);
                }

                pic_idx += 1;
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            Err(e) => panic!("parse error: {}", e),
        }
    }

    assert!(
        pic_idx >= NUM_PICS,
        "expected to parse at least {} pictures, got {}",
        NUM_PICS,
        pic_idx
    );
}

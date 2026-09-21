//! Tier E golden tests — real-stream slice-segment replay.
//!
//! Each picture of a sample stream is decoded in Rust from parser-derived
//! syntax (SPS/PPS/slice headers via `syntax_map`, the same mapping the
//! production driver uses) with synthetic deterministic reference pictures.
//! The Rust path runs twice — WPP (rayon) and forced serial — which must
//! agree exactly, and the serial output's every category (samples, CU grid,
//! intra modes, motion, filter grids, SAO params, per-segment bit positions)
//! is pinned by SHA-256 goldens generated from the build that verified
//! byte-exact agreement with the C++ oracle.

use std::env;
use std::fs;
use std::path::PathBuf;

use vacc_core::picture::{H265Pps, H265Sps};
use vacc_parser::h265::{H265Parser, SliceHeaderInfo};
use vacc_parser::{BitstreamPacket, ParseResult, SliceHeader, VideoParser};

use crate::hevc::bitreader::{BitstreamReader, extract_rbsp_with_epb};
use crate::hevc::goldens;
use crate::hevc::cabac::{CabacContext, CabacEngine};
use crate::hevc::cabac_tables::NUM_CABAC_CONTEXTS;
use crate::hevc::coding_tree::{CuInfo, DecodingContext, SaoParams, decode_slice_segment_data, hevc_trace};
use crate::hevc::inter_prediction::{DpbView, PlaneView, RefPic};
use crate::hevc::picture::{Picture, PuMotionInfo};
use crate::hevc::syntax_map;
use crate::hevc::transform::ScalingListData;
use crate::hevc::types::{ChromaFormat, Mv, PartMode, PredMode, Pps, SliceHeader as HevcSh, Sps};

/// Synthetic reference list length (streams use ≤3/≤2 active refs).
const N_LIST: i32 = 6;

// ============================================================
// Access-unit grouping (Rust parser)
// ============================================================

/// One access unit from the Rust parser: current SPS/PPS, per-NAL slice
/// entries, and the raw AU bytes (for EPB-pattern picture selection).
struct Au {
    sps_h: H265Sps,
    pps_h: H265Pps,
    slices: Vec<vacc_parser::SliceEntry>,
    bytes: Vec<u8>,
}

// ============================================================
// Synthetic reference pictures
// ============================================================

/// Deterministic synthetic reference pool — identical content on both the
/// C++ and Rust sides.
struct RefData {
    poc: Vec<i32>,
    st_ref: Vec<i32>,
    lt_ref: Vec<i32>,
    y: Vec<u16>,
    cb: Vec<u16>,
    cr: Vec<u16>,
    /// Structured motion for the Rust decoder: `[n_refs * grid]`.
    motion_rt: Vec<PuMotionInfo>,
    /// Ref POC lists for MV scaling: `[n_refs][2][16]`.
    refpoc: Vec<i32>,
}

fn synth_sample(x: i32, y: i32, comp: i32, r: i32, bd: i32) -> u16 {
    let v = (x as u32)
        .wrapping_mul(3)
        ^ (y as u32).wrapping_mul(5)
        ^ ((comp * 17 + r * 29 + 11) as u32);
    ((v.wrapping_mul(7)) & ((1u32 << bd) - 1)) as u16
}

#[allow(clippy::too_many_arguments)] // test harness: flat oracle parameters
fn make_refs(
    pic_w: i32,
    pic_h: i32,
    comp_w: i32,
    comp_h: i32,
    bd: i32,
    n_refs: i32,
    grid_w: i32,
    grid_h: i32,
) -> RefData {
    let mut d = RefData {
        poc: Vec::new(),
        st_ref: Vec::new(),
        lt_ref: Vec::new(),
        y: Vec::new(),
        cb: Vec::new(),
        cr: Vec::new(),
        motion_rt: Vec::new(),
        refpoc: Vec::new(),
    };
    for r in 0..n_refs {
        d.poc.push(100 + 7 * r);
        d.st_ref.push(1);
        d.lt_ref.push(0);
        for y in 0..pic_h {
            for x in 0..pic_w {
                d.y.push(synth_sample(x, y, 0, r, bd));
            }
        }
        for y in 0..comp_h {
            for x in 0..comp_w {
                d.cb.push(synth_sample(x, y, 1, r, bd));
                d.cr.push(synth_sample(x, y, 2, r, bd));
            }
        }
        for by in 0..grid_h {
            for bx in 0..grid_w {
                let seed = bx * 7 + by * 13 + r * 31;
                let pf0 = seed.rem_euclid(3) != 0;
                let ri0: i8 = if pf0 {
                    seed.rem_euclid(N_LIST) as i8
                } else {
                    -1
                };
                let pf1 = (seed + 1).rem_euclid(4) != 0;
                let ri1: i8 = if pf1 {
                    (seed + 5).rem_euclid(N_LIST) as i8
                } else {
                    -1
                };
                let mvx0 = seed.wrapping_mul(3).wrapping_add(bx).rem_euclid(129) - 64;
                let mvy0 = seed.wrapping_mul(5).wrapping_add(by).rem_euclid(129) - 64;
                let mvx1 = seed.wrapping_mul(7).wrapping_add(by).rem_euclid(129) - 64;
                let mvy1 = seed.wrapping_mul(11).wrapping_add(bx).rem_euclid(129) - 64;
                d.motion_rt.push(PuMotionInfo {
                    mv: [
                        Mv {
                            x: mvx0 as i16,
                            y: mvy0 as i16,
                        },
                        Mv {
                            x: mvx1 as i16,
                            y: mvy1 as i16,
                        },
                    ],
                    ref_idx: [ri0, ri1],
                    pred_flag: [pf0, pf1],
                });
            }
        }
        for l in 0..2i32 {
            let dir = if l == 0 { -4 } else { 4 };
            for k in 0..16i32 {
                d.refpoc.push(100 + 7 * r + dir * (k + 1));
            }
        }
    }
    d
}

// ============================================================
// Rust decode driver
// ============================================================

/// One slice segment driven by Rust-parsed syntax.
struct Seg<'a> {
    nal: &'a [u8],
    sh: &'a HevcSh,
    /// Coded-bit size of the slice header (parser domain).
    header_bit_size: usize,
}

fn cu_default() -> CuInfo {
    CuInfo {
        pred_mode: PredMode::Intra,
        part_mode: PartMode::Part2Nx2N,
        log2_cb_size: 0,
        intra_mode_luma: 1,
        qp_y: 26,
        is_pcm: false,
        cu_transquant_bypass: false,
        merge_flag: false,
    }
}

struct RustOut {
    y: Vec<u16>,
    cb: Vec<u16>,
    cr: Vec<u16>,
    cu: Vec<CuInfo>,
    intra_luma: Vec<i32>,
    intra_chroma: Vec<i32>,
    motion: Vec<PuMotionInfo>,
    cbf: Vec<u8>,
    log2_tu: Vec<u8>,
    edge_v: Vec<u8>,
    edge_h: Vec<u8>,
    sao: Vec<SaoParams>,
    slice_idx: Vec<u8>,
    bit_pos: Vec<usize>,
}

fn decode_rust(
    sps: &Sps,
    pps: &Pps,
    segs: &[Seg],
    refs: &RefData,
    cur_poc: i32,
    wpp_enabled: bool,
) -> RustOut {
    let pic_w = sps.pic_width_in_luma_samples;
    let pic_h = sps.pic_height_in_luma_samples;
    let comp_w = pic_w / sps.sub_width_c;
    let comp_h = pic_h / sps.sub_height_c;
    let fmt = if sps.chroma_array_type == 0 {
        ChromaFormat::Monochrome
    } else {
        ChromaFormat::Yuv420
    };

    let mut pic = Picture::default();
    pic.allocate(pic_w, pic_h, fmt, sps.bit_depth_y, sps.bit_depth_c);
    pic.poc = cur_poc;

    let grid_w = pic_w / sps.min_tb_size_y;
    let grid_h = pic_h / sps.min_tb_size_y;
    let grid = (grid_w * grid_h) as usize;
    let min_cb_w = pic_w >> sps.min_cb_log2_size_y;
    let min_cb_h = pic_h >> sps.min_cb_log2_size_y;
    let min_cbs = (min_cb_w * min_cb_h) as usize;
    let ctb_count = sps.pic_size_in_ctbs_y as usize;

    let mut cu = vec![cu_default(); min_cbs];
    let mut intra_luma = vec![1i32; grid];
    let mut intra_chroma = vec![0i32; grid];
    let mut motion = vec![PuMotionInfo::default(); grid];
    let mut cbf = vec![0u8; grid];
    let mut log2_tu = vec![sps.ctb_log2_size_y as u8; grid];
    let mut edge_v = vec![0u8; grid];
    let mut edge_h = vec![0u8; grid];
    let mut sao = vec![SaoParams::default(); ctb_count];
    let mut slice_idx = vec![0u8; ctb_count];

    // Reference pool → DpbView
    let n_refs = refs.poc.len();
    let gy = (pic_w * pic_h) as usize;
    let gc = (comp_w * comp_h) as usize;
    let pics: Vec<RefPic> = (0..n_refs)
        .map(|r| {
            RefPic {
                poc: refs.poc[r],
                used_for_short_term_ref: refs.st_ref[r] != 0,
                used_for_long_term_ref: refs.lt_ref[r] != 0,
                planes: [
                    Some(PlaneView {
                        data: &refs.y[r * gy..(r + 1) * gy],
                        width: pic_w,
                        height: pic_h,
                        stride: pic_w,
                    }),
                    Some(PlaneView {
                        data: &refs.cb[r * gc..(r + 1) * gc],
                        width: comp_w,
                        height: comp_h,
                        stride: comp_w,
                    }),
                    Some(PlaneView {
                        data: &refs.cr[r * gc..(r + 1) * gc],
                        width: comp_w,
                        height: comp_h,
                        stride: comp_w,
                    }),
                ],
                motion_info: &refs.motion_rt[r * grid..(r + 1) * grid],
                motion_stride: grid_w,
                ref_poc: [
                    &refs.refpoc[r * 32..r * 32 + 16],
                    &refs.refpoc[r * 32 + 16..r * 32 + 32],
                ],
            }
        })
        .collect();
    // Identity mapping, mirroring decode_cpp: list entry i -> pool slot i.
    let list0: Vec<i32> = (0..N_LIST).collect();
    let list1: Vec<i32> = (0..N_LIST).collect();
    let dpb = DpbView {
        pics: &pics,
        list0: &list0,
        list1: &list1,
        col_pic_idx: 0,
        no_backward_pred_flag: false,
    };

    let sl = ScalingListData::default();
    // WPP context carryover across segments (the C++ ctx stays alive for the
    // whole picture; each independent segment still re-inits at its start).
    let mut wpp_saved = [CabacContext::default(); NUM_CABAC_CONTEXTS];
    let mut wpp_avail = false;
    let mut bit_pos = vec![0usize; segs.len()];

    for (s, seg) in segs.iter().enumerate() {
        let sh = seg.sh;
        // The parser's `header_bit_size` is the coded-bit position where the
        // header ends. C++ `byte_alignment()` ALWAYS consumes >=1 bit then
        // pads to the byte boundary, so the start byte is (header_bits + 8) / 8.
        let (rbsp, epb) = extract_rbsp_with_epb(&seg.nal[2..]);
        let sh_coded = (seg.header_bit_size + 8) / 8;
        // RBSP byte where slice data starts: the coded size minus the EP bytes
        // it absorbed (counted with the C++ condition, a stable fixpoint).
        let epb_in_hdr = epb.iter().filter(|&&e| e < sh_coded + 2).count();
        let mut reader = BitstreamReader::new(&rbsp);
        let seek_target = sh_coded - epb_in_hdr;
        if hevc_trace() {
            eprintln!(
                "RUST seg {} seek: sh_coded={} epb_in_hdr={} target={} epb={:?}",
                s, sh_coded, epb_in_hdr, seek_target, &epb[..epb.len().min(8)]
            );
        }
        reader.seek_to_byte(seek_target);
        let mut cabac = CabacEngine::new(&mut reader);

        let (ok, wpp_out, bp) = {
            let mut ctx = DecodingContext {
                sps,
                pps,
                sh,
                pic: &mut pic,
                dpb: &dpb,
                cabac: &mut cabac,
                sps_scaling_list_enabled: false,
                sps_scaling_list: &sl,
                pps_scaling_list_present: false,
                pps_scaling_list: &sl,
                cu_info: &mut cu,
                cu_info_stride: min_cb_w,
                intra_pred_mode_y: &mut intra_luma,
                intra_pred_mode_c: &mut intra_chroma,
                intra_pred_mode_stride: grid_w,
                motion_info: &mut motion,
                motion_info_stride: grid_w,
                cbf_luma_grid: &mut cbf,
                log2_tu_size_grid: &mut log2_tu,
                edge_flags_v: &mut edge_v,
                edge_flags_h: &mut edge_h,
                filter_grid_stride: grid_w,
                sao_params: &mut sao,
                sao_params_stride: sps.pic_width_in_ctbs_y,
                slice_idx: Some(&mut slice_idx),
                current_slice_idx: s as i32,
                qp_y_prev: sh.slice_qp_y,
                qp_y_prev_qg: sh.slice_qp_y,
                is_cu_qp_delta_coded: false,
                cu_qp_delta_val: 0,
                cu_x0: 0,
                cu_y0: 0,
                wpp_saved_contexts: wpp_saved,
                wpp_contexts_available: wpp_avail,
                wpp_enabled,
                mc_l0: Vec::new(),
                mc_l1: Vec::new(),
                mc_out: Vec::new(),
                fir_tmp: Vec::new(),
            };
            let (ok, bp) = decode_slice_segment_data(&mut ctx, &epb, sh_coded);
            // `bp` is the segment's true end position (serial: this reader;
            // WPP: max over the per-row private readers).
            (ok, (ctx.wpp_saved_contexts, ctx.wpp_contexts_available), bp)
        };
        assert!(ok, "segment {s} decode failed");
        wpp_saved = wpp_out.0;
        wpp_avail = wpp_out.1;
        bit_pos[s] = bp;
    }

    RustOut {
        y: pic.planes[0].clone(),
        cb: pic.planes[1].clone(),
        cr: pic.planes[2].clone(),
        cu,
        intra_luma,
        intra_chroma,
        motion,
        cbf,
        log2_tu,
        edge_v,
        edge_h,
        sao,
        slice_idx,
        bit_pos,
    }
}

// ============================================================
// Golden serialization
// ============================================================

/// Flatten one RustOut into per-category byte buffers in the same layout the
/// C++ oracle comparison used (little-endian integers).
fn serialize_rust_out(rust: &RustOut) -> Vec<(&'static str, Vec<u8>)> {
    let mut out = Vec::new();
    let plane = |p: &[u16]| {
        let mut b = Vec::with_capacity(p.len() * 2);
        for v in p {
            goldens::push_u16(&mut b, *v);
        }
        b
    };
    out.push(("y", plane(&rust.y)));
    out.push(("cb", plane(&rust.cb)));
    out.push(("cr", plane(&rust.cr)));

    let mut b = Vec::new();
    for c in &rust.cu {
        goldens::push_i32(&mut b, c.pred_mode as i32);
        goldens::push_i32(&mut b, c.part_mode as i32);
        goldens::push_i32(&mut b, c.log2_cb_size);
        goldens::push_i32(&mut b, c.intra_mode_luma);
        goldens::push_i32(&mut b, c.qp_y);
        goldens::push_i32(&mut b, c.is_pcm as i32);
        goldens::push_i32(&mut b, c.cu_transquant_bypass as i32);
        goldens::push_i32(&mut b, c.merge_flag as i32);
    }
    out.push(("cu", b));

    let ints = |p: &[i32]| {
        let mut b = Vec::with_capacity(p.len() * 4);
        for v in p {
            goldens::push_i32(&mut b, *v);
        }
        b
    };
    out.push(("intra_luma", ints(&rust.intra_luma)));
    out.push(("intra_chroma", ints(&rust.intra_chroma)));

    let mut b = Vec::new();
    for m in &rust.motion {
        goldens::push_i32(&mut b, m.mv[0].x as i32);
        goldens::push_i32(&mut b, m.mv[0].y as i32);
        goldens::push_i32(&mut b, m.ref_idx[0] as i32);
        goldens::push_i32(&mut b, m.pred_flag[0] as i32);
        goldens::push_i32(&mut b, m.mv[1].x as i32);
        goldens::push_i32(&mut b, m.mv[1].y as i32);
        goldens::push_i32(&mut b, m.ref_idx[1] as i32);
        goldens::push_i32(&mut b, m.pred_flag[1] as i32);
    }
    out.push(("motion", b));

    out.push(("cbf", rust.cbf.clone()));
    out.push(("log2_tu", rust.log2_tu.clone()));
    out.push(("edge_v", rust.edge_v.clone()));
    out.push(("edge_h", rust.edge_h.clone()));

    let mut b = Vec::new();
    for s in &rust.sao {
        for v in s.sao_type_idx {
            goldens::push_i32(&mut b, v);
        }
        for v in s.sao_eo_class {
            goldens::push_i32(&mut b, v);
        }
        for v in s.sao_band_position {
            goldens::push_i32(&mut b, v);
        }
        for k in 0..3 {
            for j in 0..5 {
                goldens::push_i32(&mut b, s.sao_offset_val[k][j]);
            }
        }
    }
    out.push(("sao", b));

    out.push(("slice_idx", rust.slice_idx.clone()));

    let mut b = Vec::new();
    for v in &rust.bit_pos {
        b.extend_from_slice(&(*v as u64).to_le_bytes());
    }
    out.push(("bit_pos", b));

    out
}

// ============================================================
// Comparison
// ============================================================

fn first_diff<T: PartialEq + std::fmt::Debug>(label: &str, a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut n_diff = 0usize;
    let mut diffs: Vec<(usize, &T, &T)> = Vec::new();
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x != y {
            n_diff += 1;
            if diffs.len() < 24 {
                diffs.push((i, x, y));
            }
        }
    }
    if !diffs.is_empty() {
        let mut msg = format!(
            "{label}: {n_diff} mismatches (of {}), first 24: {:?}",
            a.len(),
            diffs
        );
        // annotate with (x,y) for luma-sized planes
        if a.len() == 640 * 360 {
            let annotated: Vec<String> = diffs
                .iter()
                .map(|(i, x, y)| format!("({},{}) cpp={:?} rust={:?}", i % 640, i / 640, x, y))
                .collect();
            msg.push_str(&format!("\n  coords: {}", annotated.join(", ")));
        }
        panic!("{msg}");
    }
}

/// WPP and serial must produce identical output (pure-Rust invariant that
/// used to be implied by both matching the C++ oracle).
fn compare_wpp_vs_serial(name: &str, pi: usize, wpp: &RustOut, ser: &RustOut) {
    let a = serialize_rust_out(wpp);
    let b = serialize_rust_out(ser);
    for ((cat, va), (_, vb)) in a.iter().zip(b.iter()) {
        first_diff(&format!("{name} pic {pi} wpp-vs-serial: {cat}"), va, vb);
    }
}

// ============================================================
// Stream driver
// ============================================================

/// Decode one picture (WPP + serial), check the WPP-vs-serial invariant, and
/// return the per-category golden hashes. `None` in trace mode.
fn test_picture(name: &str, pi: usize, au: &Au) -> Option<Vec<(String, String)>> {
    // SPS/PPS via the Rust parser (the production mapping).
    let sps = syntax_map::map_sps(&au.sps_h);
    let pps = syntax_map::map_pps(&au.pps_h, &sps);
    let pic_w = sps.pic_width_in_luma_samples;
    let pic_h = sps.pic_height_in_luma_samples;
    let comp_w = pic_w / sps.sub_width_c;
    let comp_h = pic_h / sps.sub_height_c;
    let grid_w = pic_w / sps.min_tb_size_y;
    let grid_h = pic_h / sps.min_tb_size_y;

    let refs = make_refs(
        pic_w,
        pic_h,
        comp_w,
        comp_h,
        sps.bit_depth_y,
        N_LIST,
        grid_w,
        grid_h,
    );
    // Deterministic, distinct from the synthetic ref POCs (100+7r).
    let cur_poc = pi as i32 * 4 - 500;

    // Slice segments from the Rust parser, with dependent-slice inheritance
    // from the last independent segment (mirrors the production driver).
    let has_chroma = sps.chroma_array_type != 0;
    let mut last_independent: Option<&SliceHeaderInfo> = None;
    let mut entries: Vec<(&vacc_parser::SliceEntry, HevcSh, usize)> = Vec::new();
    for e in &au.slices {
        let Some(SliceHeader::H265(info)) = &e.slice_header else {
            continue;
        };
        let sh = syntax_map::map_sh(info, last_independent, &au.pps_h, has_chroma);
        entries.push((e, sh, info.header_bit_size as usize));
        if !info.dependent_slice_segment_flag {
            last_independent = Some(info);
        }
    }
    assert!(!entries.is_empty(), "{name} pic {pi}: no slice segments parsed");
    let segs: Vec<Seg> = entries
        .iter()
        .map(|(e, sh, hbs)| Seg {
            nal: e.nal_data.as_slice(),
            sh,
            header_bit_size: *hbs,
        })
        .collect();

    // Trace mode: serial only — Rust logs per-CTU bit positions; diff against
    // an independent implementation's log to find the first diverging CTU.
    if std::env::var_os("TIER_E_TRACE").is_some() {
        let _ = decode_rust(&sps, &pps, &segs, &refs, cur_poc, false);
        return None;
    }

    let rust_wpp = decode_rust(&sps, &pps, &segs, &refs, cur_poc, true);
    let rust_ser = decode_rust(&sps, &pps, &segs, &refs, cur_poc, false);
    compare_wpp_vs_serial(name, pi, &rust_wpp, &rust_ser);

    Some(
        serialize_rust_out(&rust_ser)
            .into_iter()
            .map(|(cat, data)| {
                (format!("tier_e::{name}::pic{pi}::{cat}"), goldens::sha256_hex(&data))
            })
            .collect(),
    )
}

fn sample_dir() -> Option<PathBuf> {
    if let Ok(d) = env::var("VACC_SAMPLES_DIR") {
        return Some(PathBuf::from(d));
    }
    let p = PathBuf::from("/home/atkachenko/apps/vacc/assets/samples");
    p.is_dir().then_some(p)
}

fn run_stream(name: &str) -> Option<Vec<(String, String)>> {
    let dir = match sample_dir() {
        Some(d) => d,
        None => {
            eprintln!("skip {name}: no samples dir (set VACC_SAMPLES_DIR)");
            return None;
        }
    };
    let path = dir.join(format!("h265_{name}.h265"));
    let data = match fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip {name}: {e}");
            return None;
        }
    };
    // Parse the stream with the Rust parser (same driving pattern as tier_f).
    let mut parser = H265Parser::new();
    let mut sps_h: Option<H265Sps> = None;
    let mut pps_h: Option<H265Pps> = None;
    let mut aus: Vec<Au> = Vec::new();
    let mut parse_offset = 0usize;

    while parse_offset < data.len() {
        let mut got_au = false;
        'au: while parse_offset < data.len() {
            let remaining = &data[parse_offset..];
            let packet = BitstreamPacket::new(remaining.to_vec());
            match parser.parse(&packet) {
                Ok(ParseResult::ParameterSet { sps, pps, .. }) => {
                    if let Some(b) = sps
                        && let Some(s) = b.downcast_ref::<H265Sps>()
                    {
                        sps_h = Some(s.clone());
                    }
                    if let Some(b) = pps
                        && let Some(p) = b.downcast_ref::<H265Pps>()
                    {
                        pps_h = Some(p.clone());
                    }
                    continue 'au;
                }
                Ok(ParseResult::Slice { slices, bytes_consumed }) => {
                    if slices.is_empty() {
                        break 'au;
                    }
                    let au_end = parse_offset + bytes_consumed;
                    assert!(
                        au_end <= data.len(),
                        "{name}: bytes_consumed exceeds data"
                    );
                    let (s, p) = match (sps_h.clone(), pps_h.clone()) {
                        (Some(s), Some(p)) => (s, p),
                        _ => panic!("{name}: SPS/PPS not ready for a picture"),
                    };
                    let slices_owned: Vec<vacc_parser::SliceEntry> = slices;
                    aus.push(Au {
                        sps_h: s,
                        pps_h: p,
                        slices: slices_owned,
                        bytes: data[parse_offset..au_end].to_vec(),
                    });
                    parse_offset = au_end;
                    got_au = true;
                    break 'au;
                }
                Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break 'au,
                Err(e) => panic!("{name}: parse error: {e}"),
            }
        }
        if !got_au {
            break;
        }
    }
    assert!(!aus.is_empty(), "{name}: no pictures found");

    let full = env::var("TIER_E_FULL").is_ok_and(|v| v == "1");
    let mut entries: Vec<(String, String)> = Vec::new();
    let mut tested = 0usize;
    for (pi, au) in aus.iter().enumerate() {
        // Default: first 24 pictures plus any NAL carrying an EP-byte pattern.
        let has_epb = au.bytes.windows(3).any(|w| w == [0u8, 0, 3]);
        if !full && pi >= 24 && !has_epb {
            continue;
        }
        if let Some(e) = test_picture(name, pi, au) {
            entries.extend(e);
        }
        tested += 1;
    }
    println!("{name}: {tested}/{} pictures verified", aus.len());
    Some(entries)
}

#[test]
fn tier_e_main() {
    check_stream_goldens("main");
}

#[test]
fn tier_e_main10() {
    check_stream_goldens("main10");
}

#[test]
fn tier_e_cra() {
    check_stream_goldens("cra");
}

#[test]
fn tier_e_msp() {
    check_stream_goldens("msp");
}

fn check_stream_goldens(name: &str) {
    if let Some(entries) = run_stream(name) {
        for (key, hash) in entries {
            goldens::assert_hash(&key, &hash);
        }
    }
}

pub(crate) fn golden_entries() -> Vec<(String, String)> {
    let mut v = Vec::new();
    for name in ["main", "main10", "cra", "msp"] {
        if let Some(entries) = run_stream(name) {
            v.extend(entries);
        }
    }
    v
}

//! Tier E differential tests — real-stream slice-segment replay.
//!
//! Each picture of a sample stream is decoded three ways and compared
//! byte-for-byte:
//!   1. C++ oracle (`hevcdec_test_frame_*`): the original
//!      `decode_slice_segment_data` (always serial),
//!   2. Rust `decode_slice_segment_data` with the WPP (rayon) path,
//!   3. Rust `decode_slice_segment_data` forced serial.
//!
//! Reference pictures are fully synthetic (deterministic, identical on both
//! sides), so no prior-frame decoding is needed and every output — samples,
//! CU grid, intra modes, motion, filter grids, SAO params, per-segment bit
//! positions — is comparable.

use std::env;
use std::fs;
use std::path::PathBuf;

use crate::ffi_test as ffi;
use crate::hevc::bitreader::{BitstreamReader, extract_rbsp_with_epb};
use crate::hevc::cabac::{CabacContext, CabacEngine};
use crate::hevc::cabac_tables::NUM_CABAC_CONTEXTS;
use crate::hevc::coding_tree::{CuInfo, DecodingContext, SaoParams, decode_slice_segment_data, hevc_trace};
use crate::hevc::inter_prediction::{DpbView, PlaneView, RefPic};
use crate::hevc::interpolation::PredWeightTable;
use crate::hevc::picture::{Picture, PuMotionInfo};
use crate::hevc::transform::ScalingListData;
use crate::hevc::types::{ChromaFormat, Mv, PartMode, PredMode, Pps, SliceHeader, SliceType, Sps};

const SPS_FLAT: usize = 30;
const PPS_FLAT: usize = 48;
const SH_FLAT: usize = 343;
const MAX_SEGS: usize = 16;
/// Synthetic reference list length (streams use ≤3/≤2 active refs).
const N_LIST: i32 = 6;

// ============================================================
// NAL splitting / picture grouping
// ============================================================

struct RawNal {
    ty: u8,
    /// Full NAL bytes including the 2-byte header.
    bytes: Vec<u8>,
}

fn push_nal(nals: &mut Vec<RawNal>, bytes: &[u8]) {
    if bytes.len() < 3 {
        return;
    }
    let ty = (bytes[0] >> 1) & 0x3f;
    nals.push(RawNal {
        ty,
        bytes: bytes.to_vec(),
    });
}

/// Annex B start-code split (3- or 4-byte).
fn split_nals(data: &[u8]) -> Vec<RawNal> {
    let mut nals = Vec::new();
    let mut i = 0usize;
    let mut start: Option<usize> = None;
    while i + 3 <= data.len() {
        if data[i] == 0
            && data[i + 1] == 0
            && (data[i + 2] == 1 || (data[i + 2] == 0 && data[i + 3] == 1))
        {
            let sc_len = if data[i + 2] == 1 { 3 } else { 4 };
            if let Some(s) = start {
                push_nal(&mut nals, &data[s..i]);
            }
            start = Some(i + sc_len);
            i += sc_len;
        } else {
            i += 1;
        }
    }
    if let Some(s) = start {
        push_nal(&mut nals, &data[s..]);
    }
    nals
}

/// Group NALs into pictures (mirrors `Decoder::decode`'s VCL grouping: a new
/// picture starts at every VCL with `first_slice_segment_in_pic_flag == 1`).
/// Returns (first SPS idx, first PPS idx, per-picture VCL nal indices).
fn group_pictures(nals: &[RawNal]) -> (usize, usize, Vec<Vec<usize>>) {
    let mut sps_i = None;
    let mut pps_i = None;
    let mut pics: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for (i, nal) in nals.iter().enumerate() {
        match nal.ty {
            33 if sps_i.is_none() => sps_i = Some(i),
            34 if pps_i.is_none() => pps_i = Some(i),
            t if t <= 31 => {
                // first bit of the RBSP; the first payload byte can never be
                // an emulation prevention byte.
                let first_seg = nal.bytes[2] & 0x80 != 0;
                if !cur.is_empty() && first_seg {
                    pics.push(std::mem::take(&mut cur));
                }
                cur.push(i);
            }
            _ => {}
        }
    }
    if !cur.is_empty() {
        pics.push(cur);
    }
    (
        sps_i.expect("no SPS NAL"),
        pps_i.expect("no PPS NAL"),
        pics,
    )
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
    /// Flat motion for FFI: `[n_refs][grid * 8]`.
    motion_flat: Vec<i32>,
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
        motion_flat: Vec::new(),
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
                d.motion_flat.extend_from_slice(&[
                    mvx0,
                    mvy0,
                    ri0 as i32,
                    pf0 as i32,
                    mvx1,
                    mvy1,
                    ri1 as i32,
                    pf1 as i32,
                ]);
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
// Flat → Rust struct conversion
// ============================================================

fn sps_from_flat(f: &[i32]) -> Sps {
    Sps {
        pic_width_in_luma_samples: f[0],
        pic_height_in_luma_samples: f[1],
        bit_depth_y: f[2],
        bit_depth_c: f[3],
        chroma_array_type: f[4],
        ctb_size_y: f[5],
        min_tb_size_y: f[6],
        pic_width_in_ctbs_y: f[7],
        sub_width_c: f[8],
        sub_height_c: f[9],
        intra_smoothing_disabled_flag: f[10] != 0,
        strong_intra_smoothing_enabled_flag: f[11] != 0,
        min_cb_log2_size_y: f[12],
        ctb_log2_size_y: f[13],
        min_cb_size_y: f[14],
        pic_height_in_ctbs_y: f[15],
        pic_size_in_ctbs_y: f[16],
        min_tb_log2_size_y: f[17],
        max_tb_log2_size_y: f[18],
        qp_bd_offset_y: f[19],
        qp_bd_offset_c: f[20],
        amp_enabled_flag: f[21] != 0,
        pcm_enabled_flag: f[22] != 0,
        pcm_sample_bit_depth_luma_minus1: f[23],
        pcm_sample_bit_depth_chroma_minus1: f[24],
        log2_min_ipcm_cb_size_y: f[25],
        log2_max_ipcm_cb_size_y: f[26],
        max_transform_hierarchy_depth_inter: f[27],
        max_transform_hierarchy_depth_intra: f[28],
        cabac_bypass_alignment_enabled_flag: f[29] != 0,
        // Not exported by the oracle flat array; the tier_e decode path does
        // not apply in-loop filters, so these gates are irrelevant there.
        sample_adaptive_offset_enabled_flag: true,
        pcm_loop_filter_disabled_flag: false,
    }
}

fn pps_from_flat(f: &[i32], sps: &Sps) -> Pps {
    let ntc = (f[13] + 1) as usize;
    let ntr = (f[14] + 1) as usize;
    let mut p = Pps {
        sign_data_hiding_enabled_flag: f[0] != 0,
        transform_skip_enabled_flag: f[1] != 0,
        cu_qp_delta_enabled_flag: f[2] != 0,
        // f[3] is diff_cu_qp_delta_depth; derive per spec (§7.4.3.2.1), matching pps.cpp
        log2_min_cu_qp_delta_size: sps.ctb_log2_size_y - f[3],
        pps_cb_qp_offset: f[4],
        pps_cr_qp_offset: f[5],
        weighted_pred_flag: f[6] != 0,
        weighted_bipred_flag: f[7] != 0,
        transquant_bypass_enabled_flag: f[8] != 0,
        tiles_enabled_flag: f[9] != 0,
        entropy_coding_sync_enabled_flag: f[10] != 0,
        pps_loop_filter_across_slices_enabled_flag: f[11] != 0,
        log2_parallel_merge_level_minus2: f[12],
        num_tile_columns_minus1: f[13],
        num_tile_rows_minus1: f[14],
        uniform_spacing_flag: f[15] != 0,
        column_width_minus1: f[16..16 + ntc].iter().map(|&v| v as u32).collect(),
        row_height_minus1: f[32..32 + ntr].iter().map(|&v| v as u32).collect(),
        ..Default::default()
    };
    p.derive_tile_scan(sps);
    p
}

#[allow(clippy::field_reassign_with_default)] // weight tables filled per-entry in a loop
fn sh_from_flat(f: &[i32]) -> SliceHeader {
    let mut pwt = PredWeightTable::default();
    pwt.luma_log2_weight_denom = f[21] as u32;
    pwt.delta_chroma_log2_weight_denom = f[22];
    for list in 0..2 {
        for r in 0..16 {
            let base = 23 + list * 128 + r * 8;
            let dst = if list == 0 { &mut pwt.l0 } else { &mut pwt.l1 };
            dst[r].luma_weight = f[base + 2] as i16;
            dst[r].luma_offset = f[base + 3] as i16;
            dst[r].chroma_weight = [f[base + 4] as i16, f[base + 6] as i16];
            dst[r].chroma_offset = [f[base + 5] as i16, f[base + 7] as i16];
        }
    }
    let n_eps = f[19] as usize;
    SliceHeader {
        slice_segment_address: f[0],
        dependent_slice_segment_flag: f[1] != 0,
        slice_type: match f[2] {
            0 => SliceType::B,
            1 => SliceType::P,
            _ => SliceType::I,
        },
        pic_output_flag: f[3] != 0,
        slice_temporal_mvp_enabled_flag: f[4] != 0,
        slice_sao_luma_flag: f[5] != 0,
        slice_sao_chroma_flag: f[6] != 0,
        num_ref_idx_l0_active_minus1: f[7],
        num_ref_idx_l1_active_minus1: f[8],
        mvd_l1_zero_flag: f[9] != 0,
        cabac_init_flag: f[10] != 0,
        collocated_from_l0_flag: f[11] != 0,
        collocated_ref_idx: f[12],
        five_minus_max_num_merge_cand: f[13],
        slice_qp_delta: f[14],
        slice_cb_qp_offset: f[15],
        slice_cr_qp_offset: f[16],
        slice_qp_y: f[17],
        max_num_merge_cand: f[18],
        pred_weight_table: pwt,
        num_entry_point_offsets: n_eps as i32,
        entry_point_offset_minus1: (0..n_eps).map(|i| f[279 + i] as u32).collect(),
    }
}

// ============================================================
// Rust decode driver
// ============================================================

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

#[allow(clippy::too_many_arguments)] // test harness: mirrors the oracle call shape
fn decode_rust(
    sps: &Sps,
    pps: &Pps,
    vcl_nals: &[&[u8]],
    sh_flat: &[i32],
    n_segs: usize,
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
    let mut bit_pos = vec![0usize; n_segs];

    for s in 0..n_segs {
        let f = &sh_flat[s * SH_FLAT..(s + 1) * SH_FLAT];
        let sh = sh_from_flat(f);
        // C++ NalParser extracts RBSP from the NAL payload (after the 2-byte
        // header); match it so all positions are payload-relative.
        let (rbsp, epb) = extract_rbsp_with_epb(&vcl_nals[s][2..]);
        let sh_coded = f[20] as usize;
        // RBSP byte where slice data starts: the C++ coded size minus the EP
        // bytes it absorbed (counted with the C++ condition, a stable fixpoint).
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
                sh: &sh,
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
// C++ oracle driver
// ============================================================

struct CppOut {
    n_segs: usize,
    sh_flat: Vec<i32>,
    bit_pos: Vec<u32>,
    y: Vec<u16>,
    cb: Vec<u16>,
    cr: Vec<u16>,
    /// `[min_cbs * 8]`
    cu: Vec<i32>,
    intra_luma: Vec<i32>,
    intra_chroma: Vec<i32>,
    /// `[grid * 8]`
    motion: Vec<i32>,
    cbf: Vec<u8>,
    log2_tu: Vec<u8>,
    edge_v: Vec<u8>,
    edge_h: Vec<u8>,
    /// `[ctb_count * 24]`
    sao: Vec<i32>,
    slice_idx: Vec<u8>,
}

#[allow(clippy::too_many_arguments)] // test harness: flat FFI parameters
fn decode_cpp(
    sps_nal: &[u8],
    pps_nal: &[u8],
    vcl_bytes: &[u8],
    refs: &RefData,
    cur_poc: i32,
    pic_w: i32,
    pic_h: i32,
    comp_w: i32,
    comp_h: i32,
    grid: usize,
    min_cbs: usize,
    ctb_count: usize,
) -> CppOut {
    let n_refs = refs.poc.len() as i32;
    let mut out_sps = vec![0i32; SPS_FLAT];
    let mut out_pps = vec![0i32; PPS_FLAT];
    let mut state = std::ptr::null_mut();
    let list0: Vec<i32> = (0..N_LIST).collect();
    let list1: Vec<i32> = (0..N_LIST).collect();
    let rc = unsafe {
        ffi::hevcdec_test_frame_new(
            sps_nal.as_ptr(),
            sps_nal.len() as i32,
            pps_nal.as_ptr(),
            pps_nal.len() as i32,
            cur_poc,
            n_refs,
            refs.poc.as_ptr(),
            refs.st_ref.as_ptr(),
            refs.lt_ref.as_ptr(),
            refs.y.as_ptr(),
            refs.cb.as_ptr(),
            refs.cr.as_ptr(),
            refs.motion_flat.as_ptr(),
            refs.refpoc.as_ptr(),
            list0.as_ptr(),
            N_LIST,
            list1.as_ptr(),
            N_LIST,
            0,
            0,
            out_sps.as_mut_ptr(),
            out_pps.as_mut_ptr(),
            &mut state,
        )
    };
    assert_eq!(rc, 0, "frame_new failed");

    let mut n_segs = 0i32;
    let mut sh_flat = vec![0i32; MAX_SEGS * SH_FLAT];
    let mut bit_pos = vec![0u32; MAX_SEGS];
    let mut y = vec![0u16; (pic_w * pic_h) as usize];
    let mut cb = vec![0u16; (comp_w * comp_h) as usize];
    let mut cr = vec![0u16; (comp_w * comp_h) as usize];
    let mut cu = vec![0i32; min_cbs * 8];
    let mut intra_luma = vec![0i32; grid];
    let mut intra_chroma = vec![0i32; grid];
    let mut motion = vec![0i32; grid * 8];
    let mut cbf = vec![0u8; grid];
    let mut log2_tu = vec![0u8; grid];
    let mut edge_v = vec![0u8; grid];
    let mut edge_h = vec![0u8; grid];
    let mut sao = vec![0i32; ctb_count * 24];
    let mut slice_idx = vec![0u8; ctb_count];

    let rc = unsafe {
        ffi::hevcdec_test_frame_decode(
            state,
            vcl_bytes.as_ptr(),
            vcl_bytes.len() as i32,
            &mut n_segs,
            sh_flat.as_mut_ptr(),
            bit_pos.as_mut_ptr(),
            y.as_mut_ptr(),
            cb.as_mut_ptr(),
            cr.as_mut_ptr(),
            cu.as_mut_ptr(),
            intra_luma.as_mut_ptr(),
            intra_chroma.as_mut_ptr(),
            motion.as_mut_ptr(),
            cbf.as_mut_ptr(),
            log2_tu.as_mut_ptr(),
            edge_v.as_mut_ptr(),
            edge_h.as_mut_ptr(),
            sao.as_mut_ptr(),
            slice_idx.as_mut_ptr(),
        )
    };
    unsafe {
        ffi::hevcdec_test_frame_free(state);
    }
    assert_eq!(rc, 0, "frame_decode failed");

    CppOut {
        n_segs: n_segs as usize,
        sh_flat: sh_flat[..(n_segs as usize) * SH_FLAT].to_vec(),
        bit_pos: bit_pos[..n_segs as usize].to_vec(),
        y,
        cb,
        cr,
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
    }
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

fn compare_picture(name: &str, pi: usize, cpp: &CppOut, rust: &RustOut) {
    let prefix = format!("{name} pic {pi}");
    // TEMP(debug): motion before pixels to diagnose flat-frame inter bug.
    let rust_motion_dbg: Vec<i32> = rust
        .motion
        .iter()
        .flat_map(|m| {
            [
                m.mv[0].x as i32,
                m.mv[0].y as i32,
                m.ref_idx[0] as i32,
                m.pred_flag[0] as i32,
                m.mv[1].x as i32,
                m.mv[1].y as i32,
                m.ref_idx[1] as i32,
                m.pred_flag[1] as i32,
            ]
        })
        .collect();
    first_diff(&format!("{prefix}: motion"), &cpp.motion, &rust_motion_dbg);
    first_diff(&format!("{prefix}: luma"), &cpp.y, &rust.y);
    first_diff(&format!("{prefix}: cb"), &cpp.cb, &rust.cb);
    first_diff(&format!("{prefix}: cr"), &cpp.cr, &rust.cr);

    let rust_cu: Vec<i32> = rust
        .cu
        .iter()
        .flat_map(|c| {
            [
                c.pred_mode as i32,
                c.part_mode as i32,
                c.log2_cb_size,
                c.intra_mode_luma,
                c.qp_y,
                c.is_pcm as i32,
                c.cu_transquant_bypass as i32,
                c.merge_flag as i32,
            ]
        })
        .collect();
    first_diff(&format!("{prefix}: cu_info"), &cpp.cu, &rust_cu);

    first_diff(&format!("{prefix}: intra_luma"), &cpp.intra_luma, &rust.intra_luma);
    first_diff(
        &format!("{prefix}: intra_chroma"),
        &cpp.intra_chroma,
        &rust.intra_chroma,
    );

    let rust_motion: Vec<i32> = rust
        .motion
        .iter()
        .flat_map(|m| {
            [
                m.mv[0].x as i32,
                m.mv[0].y as i32,
                m.ref_idx[0] as i32,
                m.pred_flag[0] as i32,
                m.mv[1].x as i32,
                m.mv[1].y as i32,
                m.ref_idx[1] as i32,
                m.pred_flag[1] as i32,
            ]
        })
        .collect();
    first_diff(&format!("{prefix}: motion"), &cpp.motion, &rust_motion);

    first_diff(&format!("{prefix}: cbf_luma"), &cpp.cbf, &rust.cbf);
    first_diff(&format!("{prefix}: log2_tu"), &cpp.log2_tu, &rust.log2_tu);
    first_diff(&format!("{prefix}: edge_v"), &cpp.edge_v, &rust.edge_v);
    first_diff(&format!("{prefix}: edge_h"), &cpp.edge_h, &rust.edge_h);

    let rust_sao: Vec<i32> = rust
        .sao
        .iter()
        .flat_map(|s| {
            let mut v = [0i32; 24];
            v[..3].copy_from_slice(&s.sao_type_idx);
            v[3..6].copy_from_slice(&s.sao_eo_class);
            v[6..9].copy_from_slice(&s.sao_band_position);
            for k in 0..3 {
                for j in 0..5 {
                    v[9 + k * 5 + j] = s.sao_offset_val[k][j];
                }
            }
            v
        })
        .collect();
    first_diff(&format!("{prefix}: sao_params"), &cpp.sao, &rust_sao);

    first_diff(&format!("{prefix}: slice_idx"), &cpp.slice_idx, &rust.slice_idx);

    assert_eq!(cpp.bit_pos.len(), rust.bit_pos.len(), "{prefix}: seg count");
    for (s, (a, b)) in cpp.bit_pos.iter().zip(rust.bit_pos.iter()).enumerate() {
        assert_eq!(*a, *b as u32, "{prefix}: segment {s} bit position (cpp={a} rust={b})");
    }
}

// ============================================================
// Stream driver
// ============================================================

fn test_picture(
    name: &str,
    pi: usize,
    sps_nal: &[u8],
    pps_nal: &[u8],
    vcl_idx: &[usize],
    nals: &[RawNal],
) {
    // The C++ NalParser expects Annex B start codes; prepend one per NAL.
    // Safe: the emulation-prevention rule forbids 0x00 0x00 0x01 inside a NAL
    // payload, and these NALs end with an rbsp stop bit (last byte >= 0x80).
    let mut vcl_bytes = Vec::new();
    for &i in vcl_idx {
        vcl_bytes.extend_from_slice(&[0u8, 0, 1]);
        vcl_bytes.extend_from_slice(&nals[i].bytes);
    }
    let vcl_slices: Vec<&[u8]> = vcl_idx.iter().map(|&i| nals[i].bytes.as_slice()).collect();

    // SPS/PPS via the C++ parser (flattened for the Rust structs).
    let mut sps_buf = vec![0u8, 0, 1];
    sps_buf.extend_from_slice(sps_nal);
    let mut pps_buf = vec![0u8, 0, 1];
    pps_buf.extend_from_slice(pps_nal);
    let mut sps_f = vec![0i32; SPS_FLAT];
    let mut pps_f = vec![0i32; PPS_FLAT];
    let rc = unsafe {
        ffi::hevcdec_test_parse_sps_pps(
            sps_buf.as_ptr(),
            sps_buf.len() as i32,
            pps_buf.as_ptr(),
            pps_buf.len() as i32,
            sps_f.as_mut_ptr(),
            pps_f.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "{name} pic {pi}: SPS/PPS parse failed");

    let sps = sps_from_flat(&sps_f);
    let pps = pps_from_flat(&pps_f, &sps);
    let pic_w = sps.pic_width_in_luma_samples;
    let pic_h = sps.pic_height_in_luma_samples;
    let comp_w = pic_w / sps.sub_width_c;
    let comp_h = pic_h / sps.sub_height_c;
    let grid_w = pic_w / sps.min_tb_size_y;
    let grid_h = pic_h / sps.min_tb_size_y;
    let grid = (grid_w * grid_h) as usize;
    let min_cbs = ((pic_w >> sps.min_cb_log2_size_y) * (pic_h >> sps.min_cb_log2_size_y)) as usize;
    let ctb_count = sps.pic_size_in_ctbs_y as usize;

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

    let cpp = decode_cpp(
        &sps_buf,
        &pps_buf,
        &vcl_bytes,
        &refs,
        cur_poc,
        pic_w,
        pic_h,
        comp_w,
        comp_h,
        grid,
        min_cbs,
        ctb_count,
    );
    // Trace mode: serial only — the C++ oracle logs per-CTU bit positions
    // (HEVC_DEBUG_FILTER=TREE) and Rust logs them too; diff to find the first
    // diverging CTU. WPP is skipped so its panic can't preempt the serial run.
    if std::env::var_os("TIER_E_TRACE").is_some() {
        let _ = decode_rust(
            &sps,
            &pps,
            &vcl_slices,
            &cpp.sh_flat,
            cpp.n_segs,
            &refs,
            cur_poc,
            false,
        );
        return;
    }

    let rust_wpp = decode_rust(
        &sps,
        &pps,
        &vcl_slices,
        &cpp.sh_flat,
        cpp.n_segs,
        &refs,
        cur_poc,
        true,
    );
    let rust_ser = decode_rust(
        &sps,
        &pps,
        &vcl_slices,
        &cpp.sh_flat,
        cpp.n_segs,
        &refs,
        cur_poc,
        false,
    );

    compare_picture(name, pi, &cpp, &rust_ser);
    compare_picture(name, pi, &cpp, &rust_wpp);
}

fn sample_dir() -> Option<PathBuf> {
    if let Ok(d) = env::var("VACC_SAMPLES_DIR") {
        return Some(PathBuf::from(d));
    }
    let p = PathBuf::from("/home/atkachenko/apps/vacc/assets/samples");
    p.is_dir().then_some(p)
}

fn run_stream(name: &str) {
    let dir = match sample_dir() {
        Some(d) => d,
        None => {
            eprintln!("skip {name}: no samples dir (set VACC_SAMPLES_DIR)");
            return;
        }
    };
    let path = dir.join(format!("h265_{name}.h265"));
    let data = match fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip {name}: {e}");
            return;
        }
    };
    let nals = split_nals(&data);
    let (sps_i, pps_i, pics) = group_pictures(&nals);
    assert!(!pics.is_empty(), "{name}: no pictures found");

    let full = env::var("TIER_E_FULL").is_ok_and(|v| v == "1");
    let mut tested = 0usize;
    for (pi, vcl_idx) in pics.iter().enumerate() {
        // Default: first 24 pictures plus any NAL carrying an EP-byte pattern.
        let has_epb = vcl_idx
            .iter()
            .any(|&i| nals[i].bytes.windows(3).any(|w| w == [0u8, 0, 3]));
        if !full && pi >= 24 && !has_epb {
            continue;
        }
        test_picture(name, pi, &nals[sps_i].bytes, &nals[pps_i].bytes, vcl_idx, &nals);
        tested += 1;
    }
    println!("{name}: {tested}/{} pictures verified", pics.len());
}

#[test]
fn tier_e_main() {
    run_stream("main");
}

#[test]
fn tier_e_main10() {
    run_stream("main10");
}

#[test]
fn tier_e_cra() {
    run_stream("cra");
}

#[test]
fn tier_e_msp() {
    run_stream("msp");
}

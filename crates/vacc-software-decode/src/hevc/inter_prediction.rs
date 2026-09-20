//! Port of `hevc/decoding/inter_prediction.{h,cpp}` and
//! `perform_inter_prediction` from `interpolation.cpp` — §8.5.3:
//! merge mode (§8.5.3.2.2), AMVP (§8.5.3.2.6), MV scaling (§8.5.3.2.12),
//! temporal MV prediction (§8.5.3.2.8) and motion-compensated sample
//! derivation (§8.5.3.3).

use crate::hevc::coding_tree::DecodingContext;
use crate::hevc::interpolation::{
    PredWeightTable, interpolate_chroma, interpolate_luma, weighted_pred_default,
    weighted_pred_explicit,
};
use crate::hevc::picture::PuMotionInfo;
use crate::hevc::syntax_elements::*;
use crate::hevc::types::{Mv, Pps, SliceHeader, SliceType, Sps, clip3};

// ============================================================
// DPB view for inter prediction
// ============================================================

/// View over one reference plane (data + dims), mirroring
/// `Picture::planes[c]` / `width[c]` / `height[c]` / `stride[c]`.
#[derive(Clone, Copy)]
pub struct PlaneView<'a> {
    pub data: &'a [u16],
    pub width: i32,
    pub height: i32,
    pub stride: i32,
}

/// Immutable view of a reference picture for inter prediction (TMVP + MC).
#[derive(Clone, Copy)]
pub struct RefPic<'a> {
    pub poc: i32,
    pub used_for_short_term_ref: bool,
    pub used_for_long_term_ref: bool,
    /// `None` for chroma planes of a monochrome picture.
    pub planes: [Option<PlaneView<'a>>; 3],
    /// Per-PU motion info (4x4 granularity) — TMVP source.
    pub motion_info: &'a [PuMotionInfo],
    pub motion_stride: i32,
    /// Ref POC list snapshots (§8.5.3.2.9 MV scaling).
    pub ref_poc: [&'a [i32]; 2],
}

/// Reference picture lists + collocated picture, mirroring the subset of the
/// C++ `DPB` consumed by inter prediction.
pub struct DpbView<'a> {
    /// Reference picture pool.
    pub pics: &'a [RefPic<'a>],
    /// Pool indices into `pics`; -1 = no reference at that list entry.
    pub list0: &'a [i32],
    pub list1: &'a [i32],
    /// Pool index of the collocated picture; -1 = none.
    pub col_pic_idx: i32,
    pub no_backward_pred_flag: bool,
}

impl<'a> DpbView<'a> {
    pub fn ref_pic_list0(&self, idx: i32) -> Option<&RefPic<'a>> {
        if idx >= 0 && (idx as usize) < self.list0.len() {
            let pool = self.list0[idx as usize];
            return if pool >= 0 { Some(&self.pics[pool as usize]) } else { None };
        }
        None
    }

    pub fn ref_pic_list1(&self, idx: i32) -> Option<&RefPic<'a>> {
        if idx >= 0 && (idx as usize) < self.list1.len() {
            let pool = self.list1[idx as usize];
            return if pool >= 0 { Some(&self.pics[pool as usize]) } else { None };
        }
        None
    }

    pub fn col_pic(&self) -> Option<&RefPic<'a>> {
        if self.col_pic_idx >= 0 {
            Some(&self.pics[self.col_pic_idx as usize])
        } else {
            None
        }
    }
}

// ============================================================
// PU motion grid access
// ============================================================

pub fn store_pu_motion(
    ctx: &mut DecodingContext,
    x_pb: i32,
    y_pb: i32,
    n_pb_w: i32,
    n_pb_h: i32,
    mi: PuMotionInfo,
) {
    let min_pu_size = ctx.sps.min_tb_size_y; // 4x4 granularity
    let stride = ctx.motion_info_stride;
    for y in (0..n_pb_h).step_by(min_pu_size as usize) {
        for x in (0..n_pb_w).step_by(min_pu_size as usize) {
            let idx = (((y_pb + y) / min_pu_size) * stride + ((x_pb + x) / min_pu_size)) as usize;
            ctx.motion_info[idx] = mi;
        }
    }
}

pub fn get_pu_motion(ctx: &DecodingContext, x: i32, y: i32) -> PuMotionInfo {
    let min_pu_size = ctx.sps.min_tb_size_y;
    let stride = ctx.motion_info_stride;
    let idx = ((y / min_pu_size) * stride + (x / min_pu_size)) as usize;
    ctx.motion_info[idx]
}

// ============================================================
// §6.4.2 — Prediction block availability (simplified for inter)
// ============================================================

/// Z-scan (Morton code): interleave bits of bx and by.
fn zscan(bx: i32, by: i32) -> u32 {
    let spread = |mut v: u32| -> u32 {
        v &= 0xFF;
        v = (v | (v << 8)) & 0x00FF00FF;
        v = (v | (v << 4)) & 0x0F0F0F0F;
        v = (v | (v << 2)) & 0x33333333;
        v = (v | (v << 1)) & 0x55555555;
        v
    };
    spread(bx as u32) | (spread(by as u32) << 1)
}

/// §6.4.2 availability check shared by merge and AMVP neighbour positions.
#[allow(clippy::too_many_arguments)]
fn is_nb_available(
    ctx: &DecodingContext,
    x_cb: i32,
    y_cb: i32,
    n_cb_s: i32,
    x_pb: i32,
    y_pb: i32,
    n_pb_w: i32,
    n_pb_h: i32,
    x_nb: i32,
    y_nb: i32,
    part_idx: i32,
) -> bool {
    let pic_w = ctx.sps.pic_width_in_luma_samples;
    let pic_h = ctx.sps.pic_height_in_luma_samples;
    if x_nb < 0 || y_nb < 0 || x_nb >= pic_w || y_nb >= pic_h {
        return false;
    }

    // §6.4.2: check if neighbor is in the same coding block
    let same_cb = x_nb >= x_cb && x_nb < x_cb + n_cb_s && y_nb >= y_cb && y_nb < y_cb + n_cb_s;

    let available_n = if !same_cb {
        // §6.4.2: invoke §6.4.1 z-scan availability
        let ctb_size = ctx.sps.ctb_size_y;
        let cur_ctb_x = x_pb / ctb_size;
        let cur_ctb_y = y_pb / ctb_size;
        let nb_ctb_x = x_nb / ctb_size;
        let nb_ctb_y = y_nb / ctb_size;
        if nb_ctb_x != cur_ctb_x || nb_ctb_y != cur_ctb_y {
            let nb_addr = nb_ctb_y * ctx.sps.pic_width_in_ctbs_y + nb_ctb_x;
            let cur_addr = cur_ctb_y * ctx.sps.pic_width_in_ctbs_y + cur_ctb_x;
            if nb_addr > cur_addr {
                return false;
            }
            // §6.4.1: SliceAddrRs must match
            if let Some(slice_idx) = &ctx.slice_idx
                && slice_idx[nb_addr as usize] != slice_idx[cur_addr as usize]
            {
                return false;
            }
            // §6.4.1: TileId must match
            let pps = ctx.pps;
            if !pps.tile_id.is_empty()
                && pps.tile_id[pps.ctb_addr_rs_to_ts[nb_addr as usize] as usize]
                    != pps.tile_id[pps.ctb_addr_rs_to_ts[cur_addr as usize] as usize]
            {
                return false;
            }
            true
        } else {
            // Same CTU — z-scan check
            let min_tb = ctx.sps.min_tb_size_y;
            let ctb_org_x = cur_ctb_x * ctb_size;
            let ctb_org_y = cur_ctb_y * ctb_size;
            let cur_z = zscan((x_pb - ctb_org_x) / min_tb, (y_pb - ctb_org_y) / min_tb);
            let nb_z = zscan((x_nb - ctb_org_x) / min_tb, (y_nb - ctb_org_y) / min_tb);
            if nb_z > cur_z {
                return false;
            }
            true
        }
    } else {
        // §6.4.2: same coding block — available unless NxN partition special
        // case (partIdx==1 cannot reference the third PU below-left)
        let is_nxn = n_pb_w * 2 == n_cb_s && n_pb_h * 2 == n_cb_s;
        !(is_nxn && part_idx == 1 && y_nb >= y_cb + n_pb_h && x_nb < x_cb + n_pb_w)
    };

    if !available_n {
        return false;
    }

    // §6.4.2: must not be intra
    ctx.cu_at(x_nb, y_nb).pred_mode != crate::hevc::types::PredMode::Intra
}

// ============================================================
// §8.5.3.2.12 — MV scaling (by POC distance)
// ============================================================

fn scale_mv(mv: Mv, curr_poc: i32, curr_ref_poc: i32, col_poc: i32, col_ref_poc: i32) -> Mv {
    // Spec §8.5.3.2.12 eq 8-218 to 8-221
    let td = clip3(-128, 127, col_poc - col_ref_poc);
    let tb = clip3(-128, 127, curr_poc - curr_ref_poc);

    if td == 0 || td == tb {
        return mv;
    }

    // §8.5.3.2.12 eq 8-219: tx = (16384 + (Abs(td) >> 1)) / td
    let tx = (16384 + (td.abs() >> 1)) / td;
    // §8.5.3.2.12 eq 8-220: distScaleFactor = Clip3(-4096, 4095, (tb * tx + 32) >> 6)
    let dist_scale_factor = clip3(-4096, 4095, (tb * tx + 32) >> 6);

    // §8.5.3.2.9 eq 8-207: Sign(p) * ((Abs(p) + 127) >> 8)
    let scale_comp = |dist: i32, comp: i16| -> i16 {
        let product = dist * comp as i32;
        let sign = if product >= 0 { 1 } else { -1 };
        clip3(-32768, 32767, sign * ((product.abs() + 127) >> 8)) as i16
    };
    Mv {
        x: scale_comp(dist_scale_factor, mv.x),
        y: scale_comp(dist_scale_factor, mv.y),
    }
}

// ============================================================
// §8.5.3.2.8 — Temporal MV prediction (TMVP)
// ============================================================

/// §8.5.3.2.9: derive collocated MV from a specific position in colPic.
fn derive_collocated_mv_at(
    ctx: &DecodingContext,
    col_pic: &RefPic,
    x_col: i32,
    y_col: i32,
    ref_idx_lx: i32,
    list_x: i32,
) -> Option<Mv> {
    let min_pu_size = ctx.sps.min_tb_size_y;
    let x_col_q = (x_col >> 4) << 4;
    let y_col_q = (y_col >> 4) << 4;
    let mi_idx = ((y_col_q / min_pu_size) * col_pic.motion_stride + (x_col_q / min_pu_size)) as usize;
    if mi_idx >= col_pic.motion_info.len() {
        return None;
    }
    let col_mi = &col_pic.motion_info[mi_idx];

    // §8.5.3.2.9: if colPb is intra -> not available
    if !col_mi.pred_flag[0] && !col_mi.pred_flag[1] {
        return None;
    }

    // Determine which list to use from collocated PU
    let col_list = if col_mi.pred_flag[0] && !col_mi.pred_flag[1] {
        0
    } else if !col_mi.pred_flag[0] && col_mi.pred_flag[1] {
        1
    } else {
        // Both available: §8.5.3.2.9 — Lx = collocated_from_l0_flag (matches C++)
        if ctx.dpb.no_backward_pred_flag {
            list_x
        } else {
            ctx.sh.collocated_from_l0_flag as i32
        }
    };

    if col_list < 0
        || !col_mi.pred_flag[col_list as usize]
        || col_mi.ref_idx[col_list as usize] < 0
        || (col_mi.ref_idx[col_list as usize] as usize) >= col_pic.ref_poc[col_list as usize].len()
    {
        return None;
    }

    let col_mv = Mv {
        x: col_mi.mv[col_list as usize].x,
        y: col_mi.mv[col_list as usize].y,
    };
    let col_ref_poc = col_pic.ref_poc[col_list as usize][col_mi.ref_idx[col_list as usize] as usize];

    // Scale MV by POC distance
    let curr_poc = ctx.pic.poc;
    let curr_ref = if list_x == 0 {
        ctx.dpb.ref_pic_list0(ref_idx_lx)
    } else {
        ctx.dpb.ref_pic_list1(ref_idx_lx)
    };
    let curr_ref = curr_ref?;
    let curr_ref_poc = curr_ref.poc;

    Some(scale_mv(col_mv, curr_poc, curr_ref_poc, col_pic.poc, col_ref_poc))
}

fn derive_temporal_mv(
    ctx: &DecodingContext,
    x_pb: i32,
    y_pb: i32,
    n_pb_w: i32,
    n_pb_h: i32,
    ref_idx_lx: i32,
    list_x: i32,
) -> Option<Mv> {
    // §8.5.3.2.8: temporal luma motion vector prediction
    let col_pic = ctx.dpb.col_pic()?;
    let pic_w = ctx.sps.pic_width_in_luma_samples;
    let pic_h = ctx.sps.pic_height_in_luma_samples;
    if col_pic.motion_info.is_empty() || col_pic.motion_stride == 0 {
        return None;
    }

    // Step 1: try bottom-right position
    let x_col_br = x_pb + n_pb_w; // eq 8-198
    let y_col_br = y_pb + n_pb_h; // eq 8-199

    let br_valid = x_col_br < pic_w
        && y_col_br < pic_h
        && (y_col_br >> ctx.sps.ctb_log2_size_y) == (y_pb >> ctx.sps.ctb_log2_size_y);

    if br_valid
        && let Some(mv) = derive_collocated_mv_at(ctx, col_pic, x_col_br, y_col_br, ref_idx_lx, list_x)
    {
        return Some(mv);
    }

    // Step 2: if bottom-right unavailable or intra, try center (eq 8-200, 8-201)
    let x_col_ctr = x_pb + (n_pb_w >> 1);
    let y_col_ctr = y_pb + (n_pb_h >> 1);
    derive_collocated_mv_at(ctx, col_pic, x_col_ctr, y_col_ctr, ref_idx_lx, list_x)
}

// ============================================================
// §8.5.3.2.3 — Spatial merge candidates
// ============================================================

#[allow(clippy::too_many_arguments)]
fn derive_spatial_merge_candidates(
    ctx: &DecodingContext,
    x_cb: i32,
    y_cb: i32,
    n_cb_s: i32,
    x_pb: i32,
    y_pb: i32,
    n_pb_w: i32,
    n_pb_h: i32,
    part_idx: i32,
) -> ([PuMotionInfo; 5], [bool; 5]) {
    use crate::hevc::types::PartMode;

    // §8.5.3.2.3: candidate positions
    // A1: (xPb - 1, yPb + nPbH - 1)
    // B1: (xPb + nPbW - 1, yPb - 1)
    // B0: (xPb + nPbW, yPb - 1)
    // A0: (xPb - 1, yPb + nPbH)
    // B2: (xPb - 1, yPb - 1)
    let nb_pos = [
        (x_pb - 1, y_pb + n_pb_h - 1), // A1
        (x_pb + n_pb_w - 1, y_pb - 1), // B1
        (x_pb + n_pb_w, y_pb - 1), // B0
        (x_pb - 1, y_pb + n_pb_h), // A0
        (x_pb - 1, y_pb - 1), // B2
    ];

    let log2_par_mrg_level = ctx.pps.log2_parallel_merge_level_minus2 + 2;

    let mut cands = [PuMotionInfo::default(); 5];
    let mut avail = [false; 5];

    // §8.5.3.2.3: for each candidate, we need BOTH the raw block availability
    // (availableX) and the motion info at that position, because pruning
    // conditions use raw availability, not the filtered availableFlagX.
    let mut raw_avail = [false; 5];
    let mut raw_motion = [PuMotionInfo::default(); 5];

    // First pass: compute raw availability and motion for all 5 positions
    let pm = ctx.cu_at(x_pb, y_pb).part_mode;
    for i in 0..5 {
        let (x_nb, y_nb) = nb_pos[i];

        if !is_nb_available(ctx, x_cb, y_cb, n_cb_s, x_pb, y_pb, n_pb_w, n_pb_h, x_nb, y_nb, part_idx) {
            continue;
        }

        // §8.5.3.2.3: parallel merge level constraint
        if (x_pb >> log2_par_mrg_level) == (x_nb >> log2_par_mrg_level)
            && (y_pb >> log2_par_mrg_level) == (y_nb >> log2_par_mrg_level)
        {
            continue;
        }

        // §8.5.3.2.3: partition-specific exclusions
        if i == 0 && part_idx == 1 {
            // A1
            if pm == PartMode::PartNx2N || pm == PartMode::PartNlx2N || pm == PartMode::PartNRx2N {
                continue;
            }
        }
        if i == 1 && part_idx == 1 {
            // B1
            if pm == PartMode::Part2NxN || pm == PartMode::Part2NxnU || pm == PartMode::Part2NxnD {
                continue;
            }
        }

        raw_avail[i] = true;
        raw_motion[i] = get_pu_motion(ctx, x_nb, y_nb);
    }

    // Second pass: apply pruning per spec, using rawAvail for conditions
    let same_motion_raw = |a: &PuMotionInfo, b: &PuMotionInfo| -> bool {
        a.mv[0].x == b.mv[0].x
            && a.mv[0].y == b.mv[0].y
            && a.mv[1].x == b.mv[1].x
            && a.mv[1].y == b.mv[1].y
            && a.ref_idx[0] == b.ref_idx[0]
            && a.ref_idx[1] == b.ref_idx[1]
            && a.pred_flag[0] == b.pred_flag[0]
            && a.pred_flag[1] == b.pred_flag[1]
    };

    for i in 0..5 {
        if !raw_avail[i] {
            continue;
        }

        let mut prune = false;
        // §8.5.3.2.3: B1 pruned if availableA1 AND same motion as A1
        if i == 1 && raw_avail[0] {
            prune = same_motion_raw(&raw_motion[1], &raw_motion[0]);
        }
        // §8.5.3.2.3: B0 pruned if availableB1 AND same motion as B1
        if i == 2 && raw_avail[1] {
            prune = same_motion_raw(&raw_motion[2], &raw_motion[1]);
        }
        // §8.5.3.2.3: A0 pruned if availableA1 AND same motion as A1
        if i == 3 && raw_avail[0] {
            prune = same_motion_raw(&raw_motion[3], &raw_motion[0]);
        }
        // §8.5.3.2.3: B2 pruned if availableA1 AND same motion as A1,
        // OR availableB1 AND same motion as B1
        if i == 4 {
            // B2 only checked if < 4 candidates so far
            let cnt = avail[0..4].iter().filter(|&&a| a).count();
            if cnt >= 4 {
                continue;
            }
            if raw_avail[0] {
                prune = same_motion_raw(&raw_motion[4], &raw_motion[0]);
            }
            if !prune && raw_avail[1] {
                prune = same_motion_raw(&raw_motion[4], &raw_motion[1]);
            }
        }

        if prune {
            continue;
        }

        avail[i] = true;
        cands[i] = raw_motion[i];
    }

    (cands, avail)
}

// ============================================================
// §8.5.3.2.2 — Merge mode: build candidate list + select
// ============================================================

#[allow(clippy::too_many_arguments)]
fn derive_merge_mode(
    ctx: &DecodingContext,
    x_cb: i32,
    y_cb: i32,
    n_cb_s: i32,
    x_pb: i32,
    y_pb: i32,
    n_pb_w: i32,
    n_pb_h: i32,
    part_idx: i32,
    merge_idx: i32,
) -> PuMotionInfo {
    let sh = ctx.sh;
    // §8.5.3.2.2 eq 8-110..8-113: Log2ParMrgLevel override for 8x8 CU
    let log2_par_mrg_level = ctx.pps.log2_parallel_merge_level_minus2 + 2;
    let mut x_pb = x_pb;
    let mut y_pb = y_pb;
    let mut n_pb_w = n_pb_w;
    let mut n_pb_h = n_pb_h;
    let mut part_idx = part_idx;
    if log2_par_mrg_level > 2 && n_cb_s == 8 {
        x_pb = x_cb;
        y_pb = y_cb;
        n_pb_w = n_cb_s;
        n_pb_h = n_cb_s;
        part_idx = 0;
    }
    let max_num_merge_cand = sh.max_num_merge_cand;

    // Build merge candidate list — fixed-size array (max 5 candidates per spec)
    let mut merge_cand_list = [PuMotionInfo::default(); 5];
    let mut num_cands = 0i32;

    // Step 1: spatial candidates (§8.5.3.2.3)
    let (spatial_cands, spatial_avail) =
        derive_spatial_merge_candidates(ctx, x_cb, y_cb, n_cb_s, x_pb, y_pb, n_pb_w, n_pb_h, part_idx);

    // §8.5.3.2.2 eq 8-119: add in order A1, B1, B0, A0, B2
    for i in 0..5 {
        if num_cands >= max_num_merge_cand {
            break;
        }
        if spatial_avail[i] {
            merge_cand_list[num_cands as usize] = spatial_cands[i];
            num_cands += 1;
        }
    }

    // Step 2-4: temporal candidate (§8.5.3.2.8)
    if num_cands < max_num_merge_cand && sh.slice_temporal_mvp_enabled_flag {
        let mut col_cand = PuMotionInfo::default();
        let mv_l0_col = derive_temporal_mv(ctx, x_pb, y_pb, n_pb_w, n_pb_h, 0, 0);
        if let Some(mv) = mv_l0_col {
            col_cand.mv[0] = mv;
            col_cand.ref_idx[0] = 0;
            col_cand.pred_flag[0] = true;
        }
        if sh.slice_type == SliceType::B {
            let mv_l1_col = derive_temporal_mv(ctx, x_pb, y_pb, n_pb_w, n_pb_h, 0, 1);
            if let Some(mv) = mv_l1_col {
                col_cand.mv[1] = mv;
                col_cand.ref_idx[1] = 0;
                col_cand.pred_flag[1] = true;
            }
        }
        if col_cand.pred_flag[0] || col_cand.pred_flag[1] {
            merge_cand_list[num_cands as usize] = col_cand;
            num_cands += 1;
        }
    }

    // Step 7: combined bi-pred candidates (§8.5.3.2.4) — B slices only
    if sh.slice_type == SliceType::B {
        let num_orig_merge_cand = num_cands;
        if num_orig_merge_cand > 1 && num_orig_merge_cand < max_num_merge_cand {
            // Table 8-7
            const L0_IDX: [i32; 12] = [0, 1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 3];
            const L1_IDX: [i32; 12] = [1, 0, 2, 0, 2, 1, 3, 0, 3, 1, 3, 2];
            let mut comb_idx = 0i32;
            while comb_idx < num_orig_merge_cand * (num_orig_merge_cand - 1)
                && num_cands < max_num_merge_cand
            {
                let l0 = L0_IDX[comb_idx as usize];
                let l1 = L1_IDX[comb_idx as usize];
                comb_idx += 1;
                if l0 >= num_orig_merge_cand || l1 >= num_orig_merge_cand {
                    continue;
                }
                let c0 = &merge_cand_list[l0 as usize];
                let c1 = &merge_cand_list[l1 as usize];
                if c0.pred_flag[0] && c1.pred_flag[1] {
                    // §8.5.3.2.4: check different ref or different MV
                    let ref0 = ctx.dpb.ref_pic_list0(c0.ref_idx[0] as i32);
                    let ref1 = ctx.dpb.ref_pic_list1(c1.ref_idx[1] as i32);
                    if ref0.is_none() != ref1.is_none()
                        || (ref0.is_some() && ref1.is_some() && !std::ptr::eq(ref0.unwrap(), ref1.unwrap()))
                        || c0.mv[0].x != c1.mv[1].x
                        || c0.mv[0].y != c1.mv[1].y
                    {
                        let mut comb = PuMotionInfo::default();
                        comb.mv[0] = c0.mv[0];
                        comb.ref_idx[0] = c0.ref_idx[0];
                        comb.pred_flag[0] = true;
                        comb.mv[1] = c1.mv[1];
                        comb.ref_idx[1] = c1.ref_idx[1];
                        comb.pred_flag[1] = true;
                        merge_cand_list[num_cands as usize] = comb;
                        num_cands += 1;
                    }
                }
            }
        }
    }

    // Step 8: zero motion vector candidates (§8.5.3.2.5)
    {
        let num_ref_idx = if sh.slice_type == SliceType::P {
            sh.num_ref_idx_l0_active_minus1 + 1
        } else {
            (sh.num_ref_idx_l0_active_minus1 + 1).min(sh.num_ref_idx_l1_active_minus1 + 1)
        };
        let mut zero_idx = 0i32;
        while num_cands < max_num_merge_cand {
            let mut zero = PuMotionInfo::default();
            zero.ref_idx[0] = if zero_idx < num_ref_idx { zero_idx } else { 0 } as i8;
            zero.pred_flag[0] = true;
            zero.mv[0] = Mv { x: 0, y: 0 };
            if sh.slice_type == SliceType::B {
                zero.ref_idx[1] = if zero_idx < num_ref_idx { zero_idx } else { 0 } as i8;
                zero.pred_flag[1] = true;
                zero.mv[1] = Mv { x: 0, y: 0 };
            }
            merge_cand_list[num_cands as usize] = zero;
            num_cands += 1;
            zero_idx += 1;
        }
    }

    // Step 9: select the merge candidate (§8.5.3.2.2 eq 8-120 to 8-125)
    let sel = merge_cand_list[merge_idx as usize];
    let mut result = PuMotionInfo {
        mv: [sel.mv[0], sel.mv[1]],
        ref_idx: [sel.ref_idx[0], sel.ref_idx[1]],
        pred_flag: [sel.pred_flag[0], sel.pred_flag[1]],
    };

    // §8.5.3.2.2 step 10: bi-pred restriction for small PUs
    if result.pred_flag[0] && result.pred_flag[1] && (n_pb_w + n_pb_h) == 12 {
        result.ref_idx[1] = -1;
        result.pred_flag[1] = false;
    }

    result
}

// ============================================================
// §8.5.3.2.6 — AMVP mode (MV prediction + MVD)
// ============================================================

/// §8.5.3.2.7 — derivation process for motion vector predictor candidates.
#[allow(clippy::too_many_arguments)]
fn derive_amvp_predictor(
    ctx: &DecodingContext,
    x_cb: i32,
    y_cb: i32,
    n_cb_s: i32,
    x_pb: i32,
    y_pb: i32,
    n_pb_w: i32,
    n_pb_h: i32,
    ref_idx_lx: i32,
    list_x: i32,
    part_idx: i32,
    mvp_flag: i32,
) -> Mv {
    let x = list_x;
    let y = 1 - x;
    let curr_ref = if x == 0 {
        ctx.dpb.ref_pic_list0(ref_idx_lx)
    } else {
        ctx.dpb.ref_pic_list1(ref_idx_lx)
    };
    let curr_ref_poc = curr_ref.map_or(0, |p| p.poc);
    let curr_poc = ctx.pic.poc;

    // §8.5.3.2.7 step 1: positions
    let nb_a0 = (x_pb - 1, y_pb + n_pb_h);
    let nb_a1 = (x_pb - 1, y_pb + n_pb_h - 1);
    let nb_b0 = (x_pb + n_pb_w, y_pb - 1);
    let nb_b1 = (x_pb + n_pb_w - 1, y_pb - 1);
    let nb_b2 = (x_pb - 1, y_pb - 1);

    // §8.5.3.2.7 step 2: init
    let mut mv_lxa = Mv::default();
    let mut mv_lxb = Mv::default();
    let mut avail_flag_a = false;
    let mut avail_flag_b = false;

    // §8.5.3.2.7 steps 3-4: check A0, A1 availability
    let available_a0 = is_nb_available(ctx, x_cb, y_cb, n_cb_s, x_pb, y_pb, n_pb_w, n_pb_h, nb_a0.0, nb_a0.1, part_idx);
    let available_a1 = is_nb_available(ctx, x_cb, y_cb, n_cb_s, x_pb, y_pb, n_pb_w, n_pb_h, nb_a1.0, nb_a1.1, part_idx);

    // §8.5.3.2.7 step 5: isScaledFlagLX
    let is_scaled_flag_lx = available_a0 || available_a1;

    // §8.5.3.2.7 step 6: A group — same POC match (no scaling)
    let a_pos = [nb_a0, nb_a1];
    let a_avail = [available_a0, available_a1];
    for k in 0..2 {
        if avail_flag_a || !a_avail[k] {
            continue;
        }
        let mi = get_pu_motion(ctx, a_pos[k].0, a_pos[k].1);
        // Try same list X first, then other list Y
        if mi.pred_flag[x as usize] {
            let nb_ref = if x == 0 {
                ctx.dpb.ref_pic_list0(mi.ref_idx[x as usize] as i32)
            } else {
                ctx.dpb.ref_pic_list1(mi.ref_idx[x as usize] as i32)
            };
            if nb_ref.is_some() && nb_ref.unwrap().poc == curr_ref_poc {
                mv_lxa = mi.mv[x as usize]; // eq 8-171
                avail_flag_a = true;
                continue;
            }
        }
        if mi.pred_flag[y as usize] {
            let nb_ref = if y == 0 {
                ctx.dpb.ref_pic_list0(mi.ref_idx[y as usize] as i32)
            } else {
                ctx.dpb.ref_pic_list1(mi.ref_idx[y as usize] as i32)
            };
            if nb_ref.is_some() && nb_ref.unwrap().poc == curr_ref_poc {
                mv_lxa = mi.mv[y as usize]; // eq 8-172
                avail_flag_a = true;
            }
        }
    }

    // §8.5.3.2.7 step 7: A group — scaling pass (only if step 6 failed)
    if !avail_flag_a {
        for k in 0..2 {
            if avail_flag_a || !a_avail[k] {
                continue;
            }
            let mi = get_pu_motion(ctx, a_pos[k].0, a_pos[k].1);
            'lists: for l in 0..2 {
                let try_l = if l == 0 { x } else { y };
                if !mi.pred_flag[try_l as usize] {
                    continue;
                }
                let nb_ref = if try_l == 0 {
                    ctx.dpb.ref_pic_list0(mi.ref_idx[try_l as usize] as i32)
                } else {
                    ctx.dpb.ref_pic_list1(mi.ref_idx[try_l as usize] as i32)
                };
                let nb_ref = match nb_ref {
                    Some(r) => r,
                    None => continue,
                };
                if nb_ref.used_for_long_term_ref != curr_ref.is_some_and(|r| r.used_for_long_term_ref) {
                    continue;
                }
                avail_flag_a = true;
                if nb_ref.poc != curr_ref_poc
                    && nb_ref.used_for_short_term_ref
                    && curr_ref.is_some_and(|r| r.used_for_short_term_ref)
                {
                    mv_lxa = scale_mv(mi.mv[try_l as usize], curr_poc, curr_ref_poc, curr_poc, nb_ref.poc);
                } else {
                    mv_lxa = mi.mv[try_l as usize];
                }
                break 'lists;
            }
        }
    }

    // §8.5.3.2.7 B group step 3: B0, B1, B2 — same POC match (no scaling)
    let b_pos = [nb_b0, nb_b1, nb_b2];
    let mut b_avail_arr = [false; 3];
    for k in 0..3 {
        b_avail_arr[k] = is_nb_available(ctx, x_cb, y_cb, n_cb_s, x_pb, y_pb, n_pb_w, n_pb_h, b_pos[k].0, b_pos[k].1, part_idx);
    }

    for k in 0..3 {
        if avail_flag_b || !b_avail_arr[k] {
            continue;
        }
        let mi = get_pu_motion(ctx, b_pos[k].0, b_pos[k].1);
        if mi.pred_flag[x as usize] {
            let nb_ref = if x == 0 {
                ctx.dpb.ref_pic_list0(mi.ref_idx[x as usize] as i32)
            } else {
                ctx.dpb.ref_pic_list1(mi.ref_idx[x as usize] as i32)
            };
            if nb_ref.is_some() && nb_ref.unwrap().poc == curr_ref_poc {
                mv_lxb = mi.mv[x as usize]; // eq 8-184
                avail_flag_b = true;
                continue;
            }
        }
        if mi.pred_flag[y as usize] {
            let nb_ref = if y == 0 {
                ctx.dpb.ref_pic_list0(mi.ref_idx[y as usize] as i32)
            } else {
                ctx.dpb.ref_pic_list1(mi.ref_idx[y as usize] as i32)
            };
            if nb_ref.is_some() && nb_ref.unwrap().poc == curr_ref_poc {
                mv_lxb = mi.mv[y as usize]; // eq 8-185
                avail_flag_b = true;
            }
        }
    }

    // §8.5.3.2.7 step 4: when isScaledFlagLX == 0 and B found, copy B->A
    if !is_scaled_flag_lx && avail_flag_b {
        avail_flag_a = true;
        mv_lxa = mv_lxb; // eq 8-186
    }

    // §8.5.3.2.7 step 5: B group scaling — ONLY when isScaledFlagLX == 0
    if !is_scaled_flag_lx {
        avail_flag_b = false;
        for k in 0..3 {
            if avail_flag_b || !b_avail_arr[k] {
                continue;
            }
            let mi = get_pu_motion(ctx, b_pos[k].0, b_pos[k].1);
            'lists: for l in 0..2 {
                let try_l = if l == 0 { x } else { y };
                if !mi.pred_flag[try_l as usize] {
                    continue;
                }
                let nb_ref = if try_l == 0 {
                    ctx.dpb.ref_pic_list0(mi.ref_idx[try_l as usize] as i32)
                } else {
                    ctx.dpb.ref_pic_list1(mi.ref_idx[try_l as usize] as i32)
                };
                let nb_ref = match nb_ref {
                    Some(r) => r,
                    None => continue,
                };
                if nb_ref.used_for_long_term_ref != curr_ref.is_some_and(|r| r.used_for_long_term_ref) {
                    continue;
                }
                avail_flag_b = true;
                if nb_ref.poc != curr_ref_poc
                    && nb_ref.used_for_short_term_ref
                    && curr_ref.is_some_and(|r| r.used_for_short_term_ref)
                {
                    mv_lxb = scale_mv(mi.mv[try_l as usize], curr_poc, curr_ref_poc, curr_poc, nb_ref.poc);
                } else {
                    mv_lxb = mi.mv[try_l as usize];
                }
                break 'lists;
            }
        }
    }

    // §8.5.3.2.6 step 2: temporal candidate. Skip temporal ONLY when both A
    // and B are available AND they differ. When A==B or either is
    // unavailable, temporal IS derived.
    let mut avail_flag_col = false;
    let mut mv_col = Mv::default();
    if avail_flag_a && avail_flag_b && (mv_lxa.x != mv_lxb.x || mv_lxa.y != mv_lxb.y) {
        avail_flag_col = false;
    } else {
        if ctx.sh.slice_temporal_mvp_enabled_flag {
            mv_col = derive_temporal_mv(ctx, x_pb, y_pb, n_pb_w, n_pb_h, ref_idx_lx, list_x)
                .unwrap_or_default();
            avail_flag_col = true;
        }
    }

    // §8.5.3.2.6 step 3: build AMVP list (eq 8-170)
    let mut mvp_list = [Mv::default(); 2];
    let mut num_mvp = 0i32;

    if avail_flag_a {
        mvp_list[num_mvp as usize] = mv_lxa;
        num_mvp += 1;
        if avail_flag_b && (mv_lxa.x != mv_lxb.x || mv_lxa.y != mv_lxb.y) {
            mvp_list[num_mvp as usize] = mv_lxb;
            num_mvp += 1;
        }
    } else if avail_flag_b {
        mvp_list[num_mvp as usize] = mv_lxb;
        num_mvp += 1;
    }
    if num_mvp < 2 && avail_flag_col {
        mvp_list[num_mvp as usize] = mv_col;
        num_mvp += 1;
    }
    while num_mvp < 2 {
        mvp_list[num_mvp as usize] = Mv { x: 0, y: 0 };
        num_mvp += 1;
    }

    mvp_list[mvp_flag as usize]
}

// ============================================================
// §7.3.8.6 — prediction_unit (inter) parsing
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn decode_prediction_unit_inter(
    ctx: &mut DecodingContext,
    x_cb: i32,
    y_cb: i32,
    n_cb_s: i32,
    x_pb: i32,
    y_pb: i32,
    n_pb_w: i32,
    n_pb_h: i32,
    part_idx: i32,
) {
    let sh = ctx.sh;
    let cu_pred_mode = ctx.cu_at(x_pb, y_pb).pred_mode;
    let mut result = PuMotionInfo::default();

    if cu_pred_mode == crate::hevc::types::PredMode::Skip {
        // §7.3.8.6: cu_skip -> merge mode
        let merge_idx = if sh.max_num_merge_cand > 1 {
            decode_merge_idx(ctx.cabac, sh.max_num_merge_cand)
        } else {
            0
        };

        // Store merge_flag for rqt_root_cbf condition
        ctx.cu_at_mut(x_pb, y_pb).merge_flag = true;

        result = derive_merge_mode(ctx, x_cb, y_cb, n_cb_s, x_pb, y_pb, n_pb_w, n_pb_h, part_idx, merge_idx);
    } else {
        // §7.3.8.6: MODE_INTER — check merge_flag
        let merge_flag = decode_merge_flag(ctx.cabac) != 0;
        ctx.cu_at_mut(x_pb, y_pb).merge_flag = merge_flag;

        if merge_flag {
            let merge_idx = if sh.max_num_merge_cand > 1 {
                decode_merge_idx(ctx.cabac, sh.max_num_merge_cand)
            } else {
                0
            };

            result = derive_merge_mode(ctx, x_cb, y_cb, n_cb_s, x_pb, y_pb, n_pb_w, n_pb_h, part_idx, merge_idx);
        } else {
            // AMVP mode
            let mut inter_pred_idc = 0i32; // 0=PRED_L0, 1=PRED_L1, 2=PRED_BI
            if sh.slice_type == SliceType::B {
                // §9.3.4.2.3 Table 9-48: ctxInc for inter_pred_idc binIdx=0
                // is CtDepth[x0][y0] = CtbLog2SizeY - log2CbSize
                let mut s = n_cb_s;
                let mut log2_cb_size = 0i32;
                while s > 1 {
                    s >>= 1;
                    log2_cb_size += 1;
                }
                let ct_depth = ctx.sps.ctb_log2_size_y - log2_cb_size;
                inter_pred_idc = decode_inter_pred_idc(ctx.cabac, n_pb_w, n_pb_h, ct_depth);
            }

            // L0
            if inter_pred_idc != 1 {
                // PRED_L0 or PRED_BI
                let ref_idx_l0 = if sh.num_ref_idx_l0_active_minus1 > 0 {
                    decode_ref_idx(ctx.cabac, sh.num_ref_idx_l0_active_minus1)
                } else {
                    0
                };

                let mvd_l0 = decode_mvd(ctx.cabac);
                let mvp_l0_flag = decode_mvp_flag(ctx.cabac);

                let mvp_l0 = derive_amvp_predictor(
                    ctx,
                    x_cb,
                    y_cb,
                    n_cb_s,
                    x_pb,
                    y_pb,
                    n_pb_w,
                    n_pb_h,
                    ref_idx_l0,
                    0,
                    part_idx,
                    mvp_l0_flag,
                );

                // §8.5.3.2.1 eq 8-94..8-97: modular 16-bit MV addition
                result.mv[0].x = ((mvp_l0.x as i32 + mvd_l0.x as i32 + 0x10000) % 0x10000) as i16;
                result.mv[0].y = ((mvp_l0.y as i32 + mvd_l0.y as i32 + 0x10000) % 0x10000) as i16;
                result.ref_idx[0] = ref_idx_l0 as i8;
                result.pred_flag[0] = true;
            }

            // L1
            if inter_pred_idc != 0 {
                // PRED_L1 or PRED_BI
                let ref_idx_l1 = if sh.num_ref_idx_l1_active_minus1 > 0 {
                    decode_ref_idx(ctx.cabac, sh.num_ref_idx_l1_active_minus1)
                } else {
                    0
                };

                let mvd_l1 = if sh.mvd_l1_zero_flag && inter_pred_idc == 2 {
                    // §7.3.8.6: MvdL1 = 0 when mvd_l1_zero_flag and PRED_BI
                    Mv { x: 0, y: 0 }
                } else {
                    decode_mvd(ctx.cabac)
                };

                let mvp_l1_flag = decode_mvp_flag(ctx.cabac);

                let mvp_l1 = derive_amvp_predictor(
                    ctx,
                    x_cb,
                    y_cb,
                    n_cb_s,
                    x_pb,
                    y_pb,
                    n_pb_w,
                    n_pb_h,
                    ref_idx_l1,
                    1,
                    part_idx,
                    mvp_l1_flag,
                );

                result.mv[1].x = ((mvp_l1.x as i32 + mvd_l1.x as i32 + 0x10000) % 0x10000) as i16;
                result.mv[1].y = ((mvp_l1.y as i32 + mvd_l1.y as i32 + 0x10000) % 0x10000) as i16;
                result.ref_idx[1] = ref_idx_l1 as i8;
                result.pred_flag[1] = true;
            }
        }
    }

    // Store MV info in the motion grid
    store_pu_motion(ctx, x_pb, y_pb, n_pb_w, n_pb_h, result);
}

// ============================================================
// §8.5.3.3 — Motion compensation (perform_inter_prediction)
// ============================================================

/// Derive motion-compensated prediction samples for one PU/component,
/// mirroring C++ `perform_inter_prediction`. Writes `n_samples` = compW*compH
/// values into `pred_samples`; `pred_l0`/`pred_l1` are caller-provided
/// scratch of at least `n_samples` (all samples are written before use).
#[allow(clippy::too_many_arguments)]
pub fn perform_inter_prediction(
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    dpb: &DpbView,
    x_pb: i32,
    y_pb: i32,
    n_pb_w: i32,
    n_pb_h: i32,
    c_idx: i32,
    mv_l0: Mv,
    mv_l1: Mv,
    ref_idx_l0: i32,
    ref_idx_l1: i32,
    pred_flag_l0: bool,
    pred_flag_l1: bool,
    pred_l0: &mut [i16],
    pred_l1: &mut [i16],
    pred_samples: &mut [i16],
) {
    let bit_depth = if c_idx == 0 { sps.bit_depth_y } else { sps.bit_depth_c };

    // Component dimensions and MV conversion
    let sub_w = if c_idx > 0 { sps.sub_width_c } else { 1 };
    let sub_h = if c_idx > 0 { sps.sub_height_c } else { 1 };
    let comp_w = n_pb_w / sub_w;
    let comp_h = n_pb_h / sub_h;
    let n_samples = (comp_w * comp_h) as usize;

    // L0 prediction
    let mut pred_flag_l0 = pred_flag_l0;
    if pred_flag_l0 {
        let ref_pic = if ref_idx_l0 >= 0 { dpb.ref_pic_list0(ref_idx_l0) } else { None };
        match ref_pic {
            Some(ref_pic) => {
                if c_idx == 0 {
                    // Luma: MV in 1/4 pel
                    let x_int = x_pb + (mv_l0.x as i32 >> 2);
                    let y_int = y_pb + (mv_l0.y as i32 >> 2);
                    let x_frac = mv_l0.x as i32 & 3;
                    let y_frac = mv_l0.y as i32 & 3;
                    let plane = ref_pic.planes[0].expect("luma plane");
                    interpolate_luma(
                        plane.data,
                        plane.width,
                        plane.height,
                        plane.stride,
                        x_int,
                        y_int,
                        x_frac,
                        y_frac,
                        comp_w as usize,
                        comp_h as usize,
                        bit_depth,
                        pred_l0,
                    );
                } else {
                    // §8.5.3.3.2: chroma MV derivation from luma MV. For 4:2:0
                    // the luma 1/4-pel value maps to chroma 1/8 pel.
                    let x_pb_c = x_pb / sub_w;
                    let y_pb_c = y_pb / sub_h;
                    let x_int = x_pb_c + (mv_l0.x as i32 >> 3);
                    let y_int = y_pb_c + (mv_l0.y as i32 >> 3);
                    let x_frac = mv_l0.x as i32 & 7;
                    let y_frac = mv_l0.y as i32 & 7;
                    let plane = ref_pic.planes[c_idx as usize].expect("chroma plane");
                    interpolate_chroma(
                        plane.data,
                        c_idx,
                        plane.width,
                        plane.height,
                        plane.stride,
                        x_int,
                        y_int,
                        x_frac,
                        y_frac,
                        comp_w as usize,
                        comp_h as usize,
                        bit_depth,
                        pred_l0,
                    );
                }
            }
            // Nothing wrote predL0: uni/bi selection follows what the list
            // actually holds, not what the PU asked for.
            None => pred_flag_l0 = false,
        }
    }

    // L1 prediction
    let mut pred_flag_l1 = pred_flag_l1;
    if pred_flag_l1 {
        let ref_pic = if ref_idx_l1 >= 0 { dpb.ref_pic_list1(ref_idx_l1) } else { None };
        match ref_pic {
            Some(ref_pic) => {
                if c_idx == 0 {
                    let x_int = x_pb + (mv_l1.x as i32 >> 2);
                    let y_int = y_pb + (mv_l1.y as i32 >> 2);
                    let x_frac = mv_l1.x as i32 & 3;
                    let y_frac = mv_l1.y as i32 & 3;
                    let plane = ref_pic.planes[0].expect("luma plane");
                    interpolate_luma(
                        plane.data,
                        plane.width,
                        plane.height,
                        plane.stride,
                        x_int,
                        y_int,
                        x_frac,
                        y_frac,
                        comp_w as usize,
                        comp_h as usize,
                        bit_depth,
                        pred_l1,
                    );
                } else {
                    let x_pb_c = x_pb / sub_w;
                    let y_pb_c = y_pb / sub_h;
                    let x_int = x_pb_c + (mv_l1.x as i32 >> 3);
                    let y_int = y_pb_c + (mv_l1.y as i32 >> 3);
                    let x_frac = mv_l1.x as i32 & 7;
                    let y_frac = mv_l1.y as i32 & 7;
                    let plane = ref_pic.planes[c_idx as usize].expect("chroma plane");
                    interpolate_chroma(
                        plane.data,
                        c_idx,
                        plane.width,
                        plane.height,
                        plane.stride,
                        x_int,
                        y_int,
                        x_frac,
                        y_frac,
                        comp_w as usize,
                        comp_h as usize,
                        bit_depth,
                        pred_l1,
                    );
                }
            }
            None => pred_flag_l1 = false,
        }
    }

    // Both lists came up empty — conceal with neutral grey instead of reading
    // the buffers.
    if !pred_flag_l0 && !pred_flag_l1 {
        // The SPS allows bitDepth 16, where 1 << 15 overflows the i16 buffer.
        let grey = (1i32 << (bit_depth - 1)).min(32767);
        pred_samples.iter_mut().take(n_samples).for_each(|p| *p = grey as i16);
        return;
    }

    // §8.5.3.3.4.1: determine weightedPredFlag
    let weighted_pred_flag = if sh.slice_type == SliceType::P {
        pps.weighted_pred_flag
    } else if sh.slice_type == SliceType::B {
        pps.weighted_bipred_flag
    } else {
        false
    };

    if weighted_pred_flag {
        // §8.5.3.3.4.3: explicit weighted sample prediction
        weighted_pred_explicit(
            &pred_l0,
            &pred_l1,
            pred_flag_l0,
            pred_flag_l1,
            ref_idx_l0,
            ref_idx_l1,
            c_idx,
            n_samples as i32,
            bit_depth,
            &sh.pred_weight_table,
            pred_samples,
        );
    } else {
        // §8.5.3.3.4.2: default weighted sample prediction
        weighted_pred_default(
            &pred_l0,
            &pred_l1,
            pred_flag_l0,
            pred_flag_l1,
            n_samples as i32,
            bit_depth,
            pred_samples,
        );
    }
}

/// Weight table type re-export for the slice header.
pub type PredWeightTableAlias = PredWeightTable;

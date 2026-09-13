//! Port of `hevc/filters/deblocking.cpp` — deblocking filter, spec §8.7.2.
//!
//! Mirrors `hevc::apply_deblocking` control flow and arithmetic exactly;
//! verified byte-for-byte against the C++ oracle via `hevcdec_test_deblock_run`.
//!
//! NOTE: like the C++, this port assumes the plane buffers cover all samples
//! the 8-pixel-aligned edge loop can touch. For picture dims with W/H mod 8
//! in {1,2,3} the C++ reads/writes up to 3 samples past the last row/column
//! (latent quirk — real streams have safe dims); Rust indexing would panic
//! there instead of corrupting adjacent memory.

use crate::hevc::types::{clip3, Mv, Plane, Tiles};

/// Table 8-12: beta' from Q (spec §8.7.2.5.3).
const BETA_TABLE: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
    64,
];

/// Table 8-12: tC' from Q (spec §8.7.2.5.3).
const TC_TABLE: [i32; 54] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];

/// Table 8-10: QpC from qPi (chroma QP mapping, spec §8.6.1).
const QPC_TABLE: [i32; 58] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51,
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum EdgeType {
    Ver = 0,
    Hor = 1,
}

/// Per-slice parameters used by the deblocking filter.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeblockSliceParams {
    /// slice_deblocking_filter_disabled_flag.
    pub deblocking_disabled: bool,
    /// slice_loop_filter_across_slices_enabled_flag.
    pub across_slices_enabled: bool,
    pub beta_offset_div2: i32,
    pub tc_offset_div2: i32,
}

/// Per-min-CB (4x4) CU state used by the deblocking filter.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeblockCu {
    /// PredMode: 0 = inter, 1 = intra.
    pub pred_mode: u8,
    pub qp_y: i32,
    pub is_pcm: bool,
    pub transquant_bypass: bool,
}

/// Per-4x4 motion info (min-TB granularity).
#[derive(Clone, Copy, Debug, Default)]
pub struct MotionInfo {
    /// mvL0 / mvL1 (1/4 pel).
    pub mv: [Mv; 2],
    pub ref_idx: [i8; 2],
    pub pred_flag: [bool; 2],
}

/// Context for `apply_deblocking` — mirrors the DecodingContext fields the
/// C++ kernel reads.
pub struct DeblockCtx<'a> {
    // SPS-derived
    pub pic_w: i32,
    pub pic_h: i32,
    pub bit_depth_y: i32,
    pub bit_depth_c: i32,
    /// pcm_loop_filter_disabled_flag.
    pub pcm_filter_disabled: bool,
    pub sub_w: i32,
    pub sub_h: i32,
    /// CtbLog2SizeY.
    pub ctb_log2: i32,
    /// PicWidthInCtbsY.
    pub ctbs_w: i32,
    /// ChromaArrayType (0 = monochrome).
    pub chroma_array_type: i32,
    // PPS-derived
    /// loop_filter_across_tiles_enabled_flag.
    pub loop_filter_across_tiles: bool,
    /// None = no tiles (empty pps.TileId).
    pub tiles: Option<&'a Tiles<'a>>,
    pub pps_cb_qp_offset: i32,
    pub pps_cr_qp_offset: i32,
    // Slice state. `sh[0]` is the fallback when slice_idx is None
    // (single-slice pictures), matching `sh_at_ctb`.
    /// Per-CTB slice index; None = single slice.
    pub slice_idx: Option<&'a [u8]>,
    pub sh: &'a [DeblockSliceParams],
    // CU grid at min-CB granularity, raster order (C++ `cu_info`).
    pub cu: &'a [DeblockCu],
    /// CU grid stride = PicWidthInMinCbsY (C++ `cu_info_stride`). Distinct from
    /// `grid_stride` when MinCbSize != MinTbSize.
    pub cu_stride: i32,
    /// MinCbLog2SizeY — the shift `cu_at` applies to luma coords.
    pub min_cb_log2: i32,
    /// = picW / MinTbSizeY. The C++ motion_info_stride and filter_grid_stride.
    pub grid_stride: i32,
    // Motion + filter grids at 4x4 granularity (stride = grid_stride).
    pub motion: &'a [MotionInfo],
    /// 1 if TU has nonzero luma coefficients.
    pub cbf_luma: &'a [u8],
    /// log2 of TU size covering this 4x4 block.
    pub log2_tu_size: &'a [u8],
    /// 1 if there's a vertical / horizontal edge at this 4x4 position.
    pub edge_v: &'a [u8],
    pub edge_h: &'a [u8],
    // DPB reference picture POCs (L0 / L1).
    pub poc_l0: &'a [i32],
    pub poc_l1: &'a [i32],
}

impl<'a> DeblockCtx<'a> {
    /// CU info at a luma position (min-CB grid) — `DecodingContext::cu_at`.
    fn cu_at(&self, x: i32, y: i32) -> &DeblockCu {
        assert!(x >= 0 && y >= 0);
        let idx = ((y >> self.min_cb_log2) * self.cu_stride + (x >> self.min_cb_log2)) as usize;
        &self.cu[idx]
    }

    /// Slice header for a CTU — `DecodingContext::sh_at_ctb`.
    fn sh_at_ctb(&self, ctb_addr: i32) -> &DeblockSliceParams {
        if let Some(slice_idx) = self.slice_idx {
            let si = slice_idx[ctb_addr as usize] as usize;
            if si < self.sh.len() {
                return &self.sh[si];
            }
        }
        &self.sh[0]
    }
}

/// Reference picture POC for list/index, or -999999 when unavailable —
/// mirrors the `get_ref_poc` lambda in `derive_bs`.
fn get_ref_poc(ctx: &DeblockCtx, mi: &MotionInfo, list: usize) -> i32 {
    if !mi.pred_flag[list] || mi.ref_idx[list] < 0 {
        return -999_999;
    }
    let pocs = if list == 0 { ctx.poc_l0 } else { ctx.poc_l1 };
    if (mi.ref_idx[list] as usize) < pocs.len() {
        return pocs[mi.ref_idx[list] as usize];
    }
    -999_999
}

/// Is position (x,y) at a picture/tile/slice boundary? Returns true if
/// filtering should NOT cross this boundary — §8.7.2.1.
fn is_boundary_excluded(ctx: &DeblockCtx, x: i32, y: i32, edge_type: EdgeType) -> bool {
    let pic_w = ctx.pic_w;
    let pic_h = ctx.pic_h;

    // Picture boundary
    if edge_type == EdgeType::Ver && x == 0 {
        return true;
    }
    if edge_type == EdgeType::Hor && y == 0 {
        return true;
    }
    if edge_type == EdgeType::Ver && x >= pic_w {
        return true;
    }
    if edge_type == EdgeType::Hor && y >= pic_h {
        return true;
    }

    // §8.7.2.1: slice_deblocking_filter_disabled_flag for the slice containing
    // Q-side. Grid indices: coordinates are non-negative here (picture-boundary
    // edges returned above), so shifting matches the original division.
    let ctb_log2 = ctx.ctb_log2;
    let addr_q = (y >> ctb_log2) * ctx.ctbs_w + (x >> ctb_log2);
    let sh_q = ctx.sh_at_ctb(addr_q);
    if sh_q.deblocking_disabled {
        return true;
    }

    // Tile boundary
    if let Some(tiles) = ctx.tiles
        && !ctx.loop_filter_across_tiles
    {
        let (addr_q2, addr_p) = if edge_type == EdgeType::Ver {
            let rx_q = x >> ctb_log2;
            let rx_p = (x - 1) >> ctb_log2;
            let ry = y >> ctb_log2;
            (ry * ctx.ctbs_w + rx_q, ry * ctx.ctbs_w + rx_p)
        } else {
            let ry_q = y >> ctb_log2;
            let ry_p = (y - 1) >> ctb_log2;
            let rx = x >> ctb_log2;
            (ry_q * ctx.ctbs_w + rx, ry_p * ctx.ctbs_w + rx)
        };
        if (addr_q2 as usize) < tiles.tile_id.len() && (addr_p as usize) < tiles.tile_id.len() {
            let ts_q = tiles.ctb_addr_rs_to_ts[addr_q2 as usize];
            let ts_p = tiles.ctb_addr_rs_to_ts[addr_p as usize];
            if tiles.tile_id[ts_q as usize] != tiles.tile_id[ts_p as usize] {
                return true;
            }
        }
    }

    // Slice boundary — §8.7.2.1: exclude edges at slice boundaries when
    // slice_loop_filter_across_slices_enabled_flag == 0 (Q-side slice flag).
    if let Some(slice_idx) = ctx.slice_idx {
        let addr_p = if edge_type == EdgeType::Ver {
            (y >> ctb_log2) * ctx.ctbs_w + ((x - 1) >> ctb_log2)
        } else {
            ((y - 1) >> ctb_log2) * ctx.ctbs_w + (x >> ctb_log2)
        };
        if slice_idx[addr_q as usize] != slice_idx[addr_p as usize]
            && !sh_q.across_slices_enabled
        {
            return true;
        }
    }

    false
}

/// Boundary strength derivation — §8.7.2.4, for the edge between P=(xP,yP)
/// and Q=(xQ,yQ).
fn derive_bs(ctx: &DeblockCtx, x_p: i32, y_p: i32, x_q: i32, y_q: i32) -> i32 {
    let pic_w = ctx.pic_w;
    let pic_h = ctx.pic_h;

    // Out of bounds check
    if x_p < 0 || y_p < 0 || x_p >= pic_w || y_p >= pic_h {
        return 0;
    }
    if x_q < 0 || y_q < 0 || x_q >= pic_w || y_q >= pic_h {
        return 0;
    }

    let cu_p = ctx.cu_at(x_p, y_p);
    let cu_q = ctx.cu_at(x_q, y_q);

    // Bs=2 if either side is intra (or PCM treated as intra)
    if cu_p.pred_mode == 1 || cu_q.pred_mode == 1 {
        return 2;
    }

    // Check if edge is also a TU boundary with nonzero coefficients
    let stride = ctx.grid_stride;
    let gx_p = x_p / 4;
    let gy_p = y_p / 4;
    let gx_q = x_q / 4;
    let gy_q = y_q / 4;
    let cbf_p = ctx.cbf_luma[(gy_p * stride + gx_p) as usize] != 0;
    let cbf_q = ctx.cbf_luma[(gy_q * stride + gx_q) as usize] != 0;

    let tu_log_p = ctx.log2_tu_size[(gy_p * stride + gx_p) as usize] as i32;
    let tu_log_q = ctx.log2_tu_size[(gy_q * stride + gx_q) as usize] as i32;
    let tu_size_p = 1 << tu_log_p;
    let tu_size_q = 1 << tu_log_q;

    // Is this edge a TU boundary?
    let is_tu_edge = if x_p != x_q {
        // Vertical edge
        (x_q % tu_size_q == 0) || ((x_p + 1) % tu_size_p == 0 && (x_p + 1) == x_q)
    } else {
        // Horizontal edge
        (y_q % tu_size_q == 0) || ((y_p + 1) % tu_size_p == 0 && (y_p + 1) == y_q)
    };

    if is_tu_edge && (cbf_p || cbf_q) {
        return 1;
    }

    // Inter prediction comparison
    let min_tb = 4; // sps.MinTbSizeY (spec-fixed)
    let mi_p = &ctx.motion[(y_p / min_tb * stride + x_p / min_tb) as usize];
    let mi_q = &ctx.motion[(y_q / min_tb * stride + x_q / min_tb) as usize];

    let n_ref_p = (mi_p.pred_flag[0] as i32) + (mi_p.pred_flag[1] as i32);
    let n_ref_q = (mi_q.pred_flag[0] as i32) + (mi_q.pred_flag[1] as i32);

    if n_ref_p != n_ref_q {
        return 1;
    }

    if n_ref_p == 1 {
        // Uni-prediction
        let list_p = if mi_p.pred_flag[0] { 0 } else { 1 };
        let list_q = if mi_q.pred_flag[0] { 0 } else { 1 };
        let poc_p = get_ref_poc(ctx, mi_p, list_p);
        let poc_q = get_ref_poc(ctx, mi_q, list_q);
        if poc_p != poc_q {
            return 1;
        }
        let mv_p = mi_p.mv[list_p];
        let mv_q = mi_q.mv[list_q];
        // i32 arithmetic: C++ promotes int16 operands to int.
        if (mv_p.x as i32 - mv_q.x as i32).abs() >= 4 || (mv_p.y as i32 - mv_q.y as i32).abs() >= 4 {
            return 1;
        }
    } else if n_ref_p == 2 {
        // Bi-prediction — §8.7.2.4.5: check both orderings
        let poc_pl0 = get_ref_poc(ctx, mi_p, 0);
        let poc_pl1 = get_ref_poc(ctx, mi_p, 1);
        let poc_ql0 = get_ref_poc(ctx, mi_q, 0);
        let poc_ql1 = get_ref_poc(ctx, mi_q, 1);

        let same_order = poc_pl0 == poc_ql0 && poc_pl1 == poc_ql1;
        let swapped = poc_pl0 == poc_ql1 && poc_pl1 == poc_ql0;

        if !same_order && !swapped {
            return 1;
        }

        if same_order && !swapped {
            if (mi_p.mv[0].x as i32 - mi_q.mv[0].x as i32).abs() >= 4
                || (mi_p.mv[0].y as i32 - mi_q.mv[0].y as i32).abs() >= 4
                || (mi_p.mv[1].x as i32 - mi_q.mv[1].x as i32).abs() >= 4
                || (mi_p.mv[1].y as i32 - mi_q.mv[1].y as i32).abs() >= 4
            {
                return 1;
            }
        } else if !same_order && swapped {
            if (mi_p.mv[0].x as i32 - mi_q.mv[1].x as i32).abs() >= 4
                || (mi_p.mv[0].y as i32 - mi_q.mv[1].y as i32).abs() >= 4
                || (mi_p.mv[1].x as i32 - mi_q.mv[0].x as i32).abs() >= 4
                || (mi_p.mv[1].y as i32 - mi_q.mv[0].y as i32).abs() >= 4
            {
                return 1;
            }
        } else {
            // Both orderings match — Bs=0 only if at least one ordering gives
            // small diffs
            let order1_ok = (mi_p.mv[0].x as i32 - mi_q.mv[0].x as i32).abs() < 4
                && (mi_p.mv[0].y as i32 - mi_q.mv[0].y as i32).abs() < 4
                && (mi_p.mv[1].x as i32 - mi_q.mv[1].x as i32).abs() < 4
                && (mi_p.mv[1].y as i32 - mi_q.mv[1].y as i32).abs() < 4;
            let order2_ok = (mi_p.mv[0].x as i32 - mi_q.mv[1].x as i32).abs() < 4
                && (mi_p.mv[0].y as i32 - mi_q.mv[1].y as i32).abs() < 4
                && (mi_p.mv[1].x as i32 - mi_q.mv[0].x as i32).abs() < 4
                && (mi_p.mv[1].y as i32 - mi_q.mv[0].y as i32).abs() < 4;
            if !order1_ok && !order2_ok {
                return 1;
            }
        }
    }

    0
}

/// Edge detection: stored edge flags set during decoding — §8.7.2.2/§8.7.2.3.
fn has_edge(ctx: &DeblockCtx, x: i32, y: i32, edge_type: EdgeType) -> bool {
    let idx = (y / 4 * ctx.grid_stride + x / 4) as usize;
    if edge_type == EdgeType::Ver {
        ctx.edge_v[idx] != 0
    } else {
        ctx.edge_h[idx] != 0
    }
}

/// Decision process for a luma sample — §8.7.2.5.6.
fn decision_luma_sample(p0: i32, p3: i32, q0: i32, q3: i32, dpq: i32, beta: i32, t_c: i32) -> i32 {
    // §8.7.2.5.6: dSam = 1 if all three conditions met
    if dpq < (beta >> 2)
        && (p3 - p0).abs() + (q0 - q3).abs() < (beta >> 3)
        && (p0 - q0).abs() < ((5 * t_c + 1) >> 1)
    {
        return 1;
    }
    0
}

/// Luma filtering — §8.7.2.5.7. Returns (nDp, nDq, pOut[3], qOut[3]).
#[allow(clippy::too_many_arguments)] // faithful port of the C++ signature
fn filter_luma_sample(
    p: [i32; 4],
    q: [i32; 4],
    d_e: i32,
    d_ep: i32,
    d_eq: i32,
    t_c: i32,
    bit_depth: i32,
    pcm_p: bool,
    pcm_q: bool,
    bypass_p: bool,
    bypass_q: bool,
    pcm_filter_disabled: bool,
) -> (i32, i32, [i32; 3], [i32; 3]) {
    let max_val = (1 << bit_depth) - 1;
    let mut n_dp = 0;
    let mut n_dq = 0;
    let mut p_out = [p[0], p[1], p[2]];
    let mut q_out = [q[0], q[1], q[2]];

    if d_e == 2 {
        // Strong filter — eq 8-389 to 8-394
        n_dp = 3;
        n_dq = 3;
        p_out[0] = clip3(
            p[0] - 2 * t_c,
            p[0] + 2 * t_c,
            (p[2] + 2 * p[1] + 2 * p[0] + 2 * q[0] + q[1] + 4) >> 3,
        );
        p_out[1] = clip3(
            p[1] - 2 * t_c,
            p[1] + 2 * t_c,
            (p[2] + p[1] + p[0] + q[0] + 2) >> 2,
        );
        p_out[2] = clip3(
            p[2] - 2 * t_c,
            p[2] + 2 * t_c,
            (2 * p[3] + 3 * p[2] + p[1] + p[0] + q[0] + 4) >> 3,
        );
        q_out[0] = clip3(
            q[0] - 2 * t_c,
            q[0] + 2 * t_c,
            (p[1] + 2 * p[0] + 2 * q[0] + 2 * q[1] + q[2] + 4) >> 3,
        );
        q_out[1] = clip3(
            q[1] - 2 * t_c,
            q[1] + 2 * t_c,
            (p[0] + q[0] + q[1] + q[2] + 2) >> 2,
        );
        q_out[2] = clip3(
            q[2] - 2 * t_c,
            q[2] + 2 * t_c,
            (p[0] + q[0] + q[1] + 3 * q[2] + 2 * q[3] + 4) >> 3,
        );
    } else {
        // Weak filter — eq 8-395 to 8-402
        let mut delta = (9 * (q[0] - p[0]) - 3 * (q[1] - p[1]) + 8) >> 4;
        if delta.abs() < t_c * 10 {
            delta = clip3(-t_c, t_c, delta);
            p_out[0] = clip3(0, max_val, p[0] + delta);
            q_out[0] = clip3(0, max_val, q[0] - delta);

            if d_ep == 1 {
                let delta_p = clip3(
                    -(t_c >> 1),
                    t_c >> 1,
                    (((p[2] + p[0] + 1) >> 1) - p[1] + delta) >> 1,
                );
                p_out[1] = clip3(0, max_val, p[1] + delta_p);
            }
            if d_eq == 1 {
                let delta_q = clip3(
                    -(t_c >> 1),
                    t_c >> 1,
                    (((q[2] + q[0] + 1) >> 1) - q[1] - delta) >> 1,
                );
                q_out[1] = clip3(0, max_val, q[1] + delta_q);
            }
            n_dp = d_ep + 1;
            n_dq = d_eq + 1;
        }
    }

    // §8.7.2.5.7: PCM or transquant_bypass suppresses filtering on that side
    if n_dp > 0 && ((pcm_filter_disabled && pcm_p) || bypass_p) {
        n_dp = 0;
    }
    if n_dq > 0 && ((pcm_filter_disabled && pcm_q) || bypass_q) {
        n_dq = 0;
    }

    (n_dp, n_dq, p_out, q_out)
}

/// Chroma filtering — §8.7.2.5.8. Returns (p0Out, q0Out).
#[allow(clippy::too_many_arguments)] // faithful port of the C++ signature
fn filter_chroma_sample(
    p: [i32; 2],
    q: [i32; 2],
    t_c: i32,
    bit_depth: i32,
    pcm_p: bool,
    pcm_q: bool,
    bypass_p: bool,
    bypass_q: bool,
    pcm_filter_disabled: bool,
) -> (i32, i32) {
    let max_val = (1 << bit_depth) - 1;
    // eq 8-403
    let delta = clip3(-t_c, t_c, (((q[0] - p[0]) << 2) + p[1] - q[1] + 4) >> 3);
    let mut p0_out = clip3(0, max_val, p[0] + delta);
    let mut q0_out = clip3(0, max_val, q[0] - delta);

    // §8.7.2.5.8: suppress for PCM or transquant_bypass
    if (pcm_filter_disabled && pcm_p) || bypass_p {
        p0_out = p[0];
    }
    if (pcm_filter_disabled && pcm_q) || bypass_q {
        q0_out = q[0];
    }

    (p0_out, q0_out)
}

/// Main deblocking entry point — §8.7.2.1.
pub fn apply_deblocking(ctx: &DeblockCtx, planes: &mut [Plane]) {
    let pic_w = ctx.pic_w;
    let pic_h = ctx.pic_h;
    let bit_depth_y = ctx.bit_depth_y;
    let bit_depth_c = ctx.bit_depth_c;
    let pcm_filter_disabled = ctx.pcm_filter_disabled;
    let sub_w = ctx.sub_w;
    let sub_h = ctx.sub_h;

    // §8.7.2.1: Process vertical edges first, then horizontal
    for pass in 0..2 {
        let edge_type = if pass == 0 { EdgeType::Ver } else { EdgeType::Hor };

        // Iterate over all 8-pixel-aligned edge positions
        for y_e in (0..pic_h).step_by(8) {
            for x_e in (0..pic_w).step_by(8) {
                // For each 8-pixel edge position, process 4-sample segments
                for seg in 0..2 {
                    let (x, y) = if edge_type == EdgeType::Ver {
                        (x_e, y_e + seg * 4)
                    } else {
                        (x_e + seg * 4, y_e)
                    };

                    if x >= pic_w || y >= pic_h {
                        continue;
                    }

                    // Check if there's an edge here
                    if !has_edge(ctx, x, y, edge_type) {
                        continue;
                    }

                    // Check boundary exclusions
                    if is_boundary_excluded(ctx, x, y, edge_type) {
                        continue;
                    }

                    // Derive Bs
                    let (x_p, y_p, x_q, y_q) = if edge_type == EdgeType::Ver {
                        (x - 1, y, x, y)
                    } else {
                        (x, y - 1, x, y)
                    };
                    let b_s = derive_bs(ctx, x_p, y_p, x_q, y_q);
                    if b_s == 0 {
                        continue;
                    }

                    // Get QP for both sides
                    let cu_p = ctx.cu_at(x_p, y_p);
                    let cu_q = ctx.cu_at(x_q, y_q);
                    let pcm_p = cu_p.is_pcm;
                    let pcm_q = cu_q.is_pcm;
                    let bypass_p = cu_p.transquant_bypass;
                    let bypass_q = cu_q.transquant_bypass;

                    // §8.7.2.5.3: slice parameters for the slice containing q0,0
                    let addr_q = (y_q >> ctx.ctb_log2) * ctx.ctbs_w + (x_q >> ctx.ctb_log2);
                    let sh_filt = ctx.sh_at_ctb(addr_q);

                    // ---- LUMA ----
                    {
                        let qp_p = cu_p.qp_y;
                        let qp_q = cu_q.qp_y;
                        let q_pl = (qp_q + qp_p + 1) >> 1; // eq 8-347

                        let q_beta = clip3(0, 51, q_pl + (sh_filt.beta_offset_div2 << 1));
                        let beta_prime = BETA_TABLE[q_beta as usize];
                        let beta = beta_prime * (1 << (bit_depth_y - 8)); // eq 8-349

                        let q_tc = clip3(
                            0,
                            53,
                            q_pl + 2 * (b_s - 1) + (sh_filt.tc_offset_div2 << 1),
                        );
                        let tc_prime = TC_TABLE[q_tc as usize];
                        let t_c = tc_prime * (1 << (bit_depth_y - 8)); // eq 8-351

                        // Decision process — §8.7.2.5.3
                        // Read p0..p3 and q0..q3 for lines k=0 and k=3
                        let mut p_samp = [[0i32; 2]; 4];
                        let mut q_samp = [[0i32; 2]; 4];
                        {
                            let luma = &planes[0].data;
                            let stride = planes[0].stride;
                            for kk in 0..2 {
                                let k = kk * 3; // k = 0 and 3
                                if edge_type == EdgeType::Ver {
                                    for i in 0..4 {
                                        q_samp[i as usize][kk as usize] =
                                            luma[((y_q + k) * stride + (x_q + i)) as usize] as i32;
                                        p_samp[i as usize][kk as usize] =
                                            luma[((y_q + k) * stride + (x_q - i - 1)) as usize] as i32;
                                    }
                                } else {
                                    for i in 0..4 {
                                        q_samp[i as usize][kk as usize] =
                                            luma[((y_q + i) * stride + x_q + k) as usize] as i32;
                                        p_samp[i as usize][kk as usize] =
                                            luma[((y_q - i - 1) * stride + x_q + k) as usize] as i32;
                                    }
                                }
                            }
                        }

                        // dp, dq, d (eq 8-352 to 8-360 / 8-361 to 8-369)
                        let dp0 = (p_samp[2][0] - 2 * p_samp[1][0] + p_samp[0][0]).abs();
                        let dp3 = (p_samp[2][1] - 2 * p_samp[1][1] + p_samp[0][1]).abs();
                        let dq0 = (q_samp[2][0] - 2 * q_samp[1][0] + q_samp[0][0]).abs();
                        let dq3 = (q_samp[2][1] - 2 * q_samp[1][1] + q_samp[0][1]).abs();
                        let dpq0 = dp0 + dq0;
                        let dpq3 = dp3 + dq3;
                        let dp = dp0 + dp3;
                        let dq = dq0 + dq3;
                        let d = dpq0 + dpq3;

                        let (mut d_e, mut d_ep, mut d_eq) = (0, 0, 0);
                        if d < beta {
                            // §8.7.2.5.6 for line 0
                            let d_sam0 = decision_luma_sample(
                                p_samp[0][0],
                                p_samp[3][0],
                                q_samp[0][0],
                                q_samp[3][0],
                                2 * dpq0,
                                beta,
                                t_c,
                            );
                            // §8.7.2.5.6 for line 3
                            let d_sam3 = decision_luma_sample(
                                p_samp[0][1],
                                p_samp[3][1],
                                q_samp[0][1],
                                q_samp[3][1],
                                2 * dpq3,
                                beta,
                                t_c,
                            );

                            d_e = 1;
                            if d_sam0 == 1 && d_sam3 == 1 {
                                d_e = 2;
                            }
                            if dp < ((beta + (beta >> 1)) >> 3) {
                                d_ep = 1;
                            }
                            if dq < ((beta + (beta >> 1)) >> 3) {
                                d_eq = 1;
                            }
                        }

                        // §8.7.2.5.4: Filter all 4 luma lines (only if dE > 0)
                        if d_e > 0 {
                            for k in 0..4 {
                                let mut p_line = [0i32; 4];
                                let mut q_line = [0i32; 4];
                                {
                                    let luma = &planes[0].data;
                                    let stride = planes[0].stride;
                                    if edge_type == EdgeType::Ver {
                                        for i in 0..4 {
                                            q_line[i as usize] =
                                                luma[((y_q + k) * stride + (x_q + i)) as usize] as i32;
                                            p_line[i as usize] =
                                                luma[((y_q + k) * stride + (x_q - i - 1)) as usize] as i32;
                                        }
                                    } else {
                                        for i in 0..4 {
                                            q_line[i as usize] =
                                                luma[((y_q + i) * stride + x_q + k) as usize] as i32;
                                            p_line[i as usize] =
                                                luma[((y_q - i - 1) * stride + x_q + k) as usize] as i32;
                                        }
                                    }
                                }

                                let (n_dp, n_dq, p_out, q_out) = filter_luma_sample(
                                    p_line,
                                    q_line,
                                    d_e,
                                    d_ep,
                                    d_eq,
                                    t_c,
                                    bit_depth_y,
                                    pcm_p,
                                    pcm_q,
                                    bypass_p,
                                    bypass_q,
                                    pcm_filter_disabled,
                                );

                                // Write back filtered samples
                                let luma = &mut planes[0].data;
                                let stride = planes[0].stride;
                                if edge_type == EdgeType::Ver {
                                    for i in 0..n_dp {
                                        luma[((y_q + k) * stride + (x_q - i - 1)) as usize] =
                                            p_out[i as usize] as u16;
                                    }
                                    for j in 0..n_dq {
                                        luma[((y_q + k) * stride + (x_q + j)) as usize] =
                                            q_out[j as usize] as u16;
                                    }
                                } else {
                                    for i in 0..n_dp {
                                        luma[((y_q - i - 1) * stride + x_q + k) as usize] =
                                            p_out[i as usize] as u16;
                                    }
                                    for j in 0..n_dq {
                                        luma[((y_q + j) * stride + x_q + k) as usize] =
                                            q_out[j as usize] as u16;
                                    }
                                }
                            }
                        }
                    }

                    // ---- CHROMA ----
                    // §8.7.2.5.1/5.2: chroma filtered only when Bs == 2 and edge
                    // is on the 8-sample chroma grid
                    if b_s == 2 && ctx.chroma_array_type != 0 {
                        // Check 8-sample grid alignment in chroma space
                        let chroma_x = x / sub_w;
                        let chroma_y = y / sub_h;
                        let chroma_aligned = if edge_type == EdgeType::Ver {
                            ((chroma_x >> 3) << 3) == chroma_x
                        } else {
                            ((chroma_y >> 3) << 3) == chroma_y
                        };

                        if chroma_aligned {
                            for c_idx in 1..=2 {
                                let c_qp_pic_offset = if c_idx == 1 {
                                    ctx.pps_cb_qp_offset
                                } else {
                                    ctx.pps_cr_qp_offset
                                };
                                let q_pi = ((cu_q.qp_y + cu_p.qp_y + 1) >> 1) + c_qp_pic_offset;

                                // §8.7.2.5.5: QpC from Table 8-10
                                let q_p_c = if ctx.chroma_array_type == 1 {
                                    QPC_TABLE[clip3(0, 57, q_pi) as usize]
                                } else {
                                    q_pi.min(51)
                                };

                                let q_tc =
                                    clip3(0, 53, q_p_c + 2 + (sh_filt.tc_offset_div2 << 1));
                                let tc_prime = TC_TABLE[q_tc as usize];
                                let t_c = tc_prime * (1 << (bit_depth_c - 8));

                                // Filter chroma lines along the edge
                                let chroma_segs = if edge_type == EdgeType::Ver {
                                    4 / sub_h
                                } else {
                                    4 / sub_w
                                };
                                let c_stride = planes[c_idx as usize].stride;
                                for k in 0..chroma_segs {
                                    let (cx, cy) = if edge_type == EdgeType::Ver {
                                        (x / sub_w, y / sub_h + k)
                                    } else {
                                        (x / sub_w + k, y / sub_h)
                                    };
                                    let (p_c, q_c) = {
                                        let plane = &planes[c_idx as usize].data;
                                        if edge_type == EdgeType::Ver {
                                            let row = cy as usize * c_stride as usize;
                                            (
                                                [
                                                    plane[row + cx as usize - 1] as i32,
                                                    plane[row + cx as usize - 2] as i32,
                                                ],
                                                [
                                                    plane[row + cx as usize] as i32,
                                                    plane[row + cx as usize + 1] as i32,
                                                ],
                                            )
                                        } else {
                                            let col = cx as usize;
                                            (
                                                [
                                                    plane[(cy - 1) as usize * c_stride as usize + col] as i32,
                                                    plane[(cy - 2) as usize * c_stride as usize + col] as i32,
                                                ],
                                                [
                                                    plane[cy as usize * c_stride as usize + col] as i32,
                                                    plane[(cy + 1) as usize * c_stride as usize + col] as i32,
                                                ],
                                            )
                                        }
                                    };

                                    let (p0_out, q0_out) = filter_chroma_sample(
                                        p_c,
                                        q_c,
                                        t_c,
                                        bit_depth_c,
                                        cu_p.is_pcm,
                                        cu_q.is_pcm,
                                        bypass_p,
                                        bypass_q,
                                        pcm_filter_disabled,
                                    );

                                    let plane = &mut planes[c_idx as usize].data;
                                    if edge_type == EdgeType::Ver {
                                        let row = cy as usize * c_stride as usize;
                                        plane[row + cx as usize - 1] = p0_out as u16;
                                        plane[row + cx as usize] = q0_out as u16;
                                    } else {
                                        plane[(cy - 1) as usize * c_stride as usize + cx as usize] =
                                            p0_out as u16;
                                        plane[cy as usize * c_stride as usize + cx as usize] =
                                            q0_out as u16;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi_test;

    /// Deterministic xorshift64* RNG (same scheme as other kernel tests).
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
        /// i32 in [0, n).
        fn i32(&mut self, n: u64) -> i32 {
            self.below(n) as i32
        }
    }

    /// One randomized deblocking scenario run through both implementations.
    /// Picture dims stay multiples of 8 (safe region for the C++ edge loop).
    fn run_deblock_case(
        rng: &mut Rng,
        multi_slice: bool,
        use_tiles: bool,
        dense_pcm: bool,
    ) {
        let bit_depth = if rng.below(2) == 0 { 8 } else { 10 };
        let chroma_array_type = if rng.below(3) == 0 { 0 } else { 1 };
        let (sub_w, sub_h) = if chroma_array_type == 0 { (1, 1) } else { (2, 2) };
        let ctb_log2 = 5 + rng.i32(3);
        let ctb_size = 1 << ctb_log2;
        // Multiples of 8 keep both implementations well-defined.
        let max_ctbs = ((ctb_size * 3) / 8 + 1).min(32);
        let pic_w = 8 * (8 + rng.i32(max_ctbs as u64));
        let pic_h = 8 * (8 + rng.i32(max_ctbs as u64));
        let ctbs_w = (pic_w + ctb_size - 1) / ctb_size;
        let num_ctbs = ctbs_w * ((pic_h + ctb_size - 1) / ctb_size);
        let comp_w = pic_w / sub_w;
        let comp_h = pic_h / sub_h;

        // Planes: random noise or a two-region gradient (the latter triggers
        // strong/weak luma filtering more often).
        let max_val = (1 << bit_depth) - 1;
        let mut fill_plane = |w: i32, h: i32| -> Vec<u16> {
            let mut out = vec![0u16; (w * h) as usize];
            if rng.below(3) == 0 {
                // Vertical step edge at a random x with small noise.
                let x_edge = 8 + rng.i32(((w - 16) / 8).max(1) as u64) * 8;
                let base_a = rng.i32(max_val as u64 / 2);
                let delta = (rng.i32(128) + 16) * if rng.below(2) == 0 { 1 } else { -1 };
                let base_b = base_a + delta;
                for i in 0..w * h {
                    let x = i / w;
                    let base = if x < x_edge { base_a } else { base_b };
                    out[i as usize] = (base + rng.i32(9) - 4).clamp(0, max_val) as u16;
                }
            } else {
                for i in 0..w * h {
                    out[i as usize] = rng.below(max_val as u64 + 1) as u16;
                }
            }
            out
        };
        let plane_y = fill_plane(pic_w, pic_h);
        let plane_cb = fill_plane(comp_w, comp_h);
        let plane_cr = fill_plane(comp_w, comp_h);

        // Slice params: 4 ints per slice.
        let n_slices: i32 = if multi_slice { 2 } else { 1 };
        let mut sh_params: Vec<i32> = Vec::with_capacity((n_slices * 4) as usize);
        for _ in 0..n_slices {
            sh_params.push(if rng.below(12) == 0 { 1 } else { 0 }); // deblocking_disabled (rare)
            sh_params.push(if rng.below(2) == 0 { 1 } else { 0 });  // across_slices_enabled
            sh_params.push(rng.i32(13) - 6);                        // beta_offset_div2 in [-6, 6]
            sh_params.push(rng.i32(13) - 6);                        // tc_offset_div2 in [-6, 6]
        }

        let mut slice_idx = vec![0u8; num_ctbs as usize];
        if multi_slice {
            // Straight vertical cut through the CTB grid.
            let cut = 1 + rng.i32((ctbs_w - 1).max(1) as u64);
            for i in 0..num_ctbs {
                slice_idx[i as usize] = if (i % ctbs_w) >= cut { 1 } else { 0 };
            }
        }

        // Tiles: left/right halves of the CTB grid, identity RsToTs.
        let mut tile_id: Vec<i32> = Vec::new();
        let mut rs2ts: Vec<i32> = Vec::new();
        if use_tiles {
            for i in 0..num_ctbs {
                let rx = i % ctbs_w;
                tile_id.push(if rx < (ctbs_w / 2).max(1) { 0 } else { 1 });
                rs2ts.push(i);
            }
        }
        // Byte-packed form for the C++ oracle (one byte per CTB).
        let tile_id_bytes: Vec<u8> = tile_id.iter().map(|&t| t as u8).collect();

        // CU grid + motion + filter grids, per 4x4 block.
        let n_blocks = (pic_w / 4) * (pic_h / 4);
        let n_ref_l0 = rng.i32(3);
        let n_ref_l1 = rng.i32(3);
        // Small POC pool with duplicates: exercises sameOrder/swapped/both.
        let poc_pool = [-10i32, -4, 0, 2, 8];
        let poc_l0: Vec<i32> = (0..n_ref_l0).map(|_| poc_pool[rng.i32(5) as usize]).collect();
        let poc_l1: Vec<i32> = (0..n_ref_l1).map(|_| poc_pool[rng.i32(5) as usize]).collect();

        let pcm_density = if dense_pcm { 4 } else { 12 };
        let bypass_density = if dense_pcm { 4 } else { 16 };
        let mut cu_fields: Vec<i32> = Vec::with_capacity((n_blocks * 4) as usize);
        let mut motion: Vec<i32> = Vec::with_capacity((n_blocks * 8) as usize);
        let mut cbf_luma = vec![0u8; n_blocks as usize];
        let mut log2_tu = vec![0u8; n_blocks as usize];
        let mut edge_v = vec![0u8; n_blocks as usize];
        let mut edge_h = vec![0u8; n_blocks as usize];

        for _ in 0..n_blocks {
            // CU fields: pred_mode, qp_y, is_pcm, bypass
            cu_fields.push(if rng.below(4) == 0 { 1 } else { 0 });
            cu_fields.push(-6 + rng.i32(64)); // qp_y in [-6, 57]
            cu_fields.push((rng.below(pcm_density as u64) == 0) as i32);
            cu_fields.push((rng.below(bypass_density as u64) == 0) as i32);

            // Motion: random pred pattern / refs / MVs.
            let pattern = rng.i32(10);
            let (pf0, pf1) = match pattern {
                0..=3 => (false, false), // no prediction (intra CUs)
                4..=5 => (true, false),  // uni L0
                6..=7 => (false, true),  // uni L1
                _ => (true, true),       // bi
            };
            let ri0 = if pf0 && n_ref_l0 > 0 { rng.i32(n_ref_l0 as u64) } else { -1 };
            let ri1 = if pf1 && n_ref_l1 > 0 { rng.i32(n_ref_l1 as u64) } else { -1 };
            let mv_range = if rng.below(5) == 0 { 320 } else { 8 };
            motion.push(rng.i32(mv_range as u64 * 2 + 1) - mv_range);
            motion.push(rng.i32(mv_range as u64 * 2 + 1) - mv_range);
            motion.push(rng.i32(mv_range as u64 * 2 + 1) - mv_range);
            motion.push(rng.i32(mv_range as u64 * 2 + 1) - mv_range);
            motion.push(ri0);
            motion.push(ri1);
            motion.push(pf0 as i32);
            motion.push(pf1 as i32);
        }
        // Filter grids.
        for i in 0..n_blocks {
            if rng.below(10) < 3 {
                cbf_luma[i as usize] = 1;
            }
            log2_tu[i as usize] = (2 + rng.i32(4)) as u8; // TU size 4..32
            if rng.below(10) < 3 {
                edge_v[i as usize] = 1;
            }
            if rng.below(10) < 3 {
                edge_h[i as usize] = 1;
            }
        }

        // --- Rust side ---
        let mut cu_rs = Vec::with_capacity(n_blocks as usize);
        for i in 0..n_blocks {
            let f = &cu_fields[(i * 4) as usize..(i * 4 + 4) as usize];
            cu_rs.push(DeblockCu {
                pred_mode: f[0] as u8,
                qp_y: f[1],
                is_pcm: f[2] != 0,
                transquant_bypass: f[3] != 0,
            });
        }
        let mut motion_rs = Vec::with_capacity(n_blocks as usize);
        for i in 0..n_blocks {
            let m = &motion[(i * 8) as usize..(i * 8 + 8) as usize];
            motion_rs.push(MotionInfo {
                mv: [Mv { x: m[0] as i16, y: m[1] as i16 }, Mv { x: m[2] as i16, y: m[3] as i16 }],
                ref_idx: [m[4] as i8, m[5] as i8],
                pred_flag: [m[6] != 0, m[7] != 0],
            });
        }
        let mut sh_rs: Vec<DeblockSliceParams> = Vec::with_capacity(n_slices as usize);
        for s in 0..n_slices {
            sh_rs.push(DeblockSliceParams {
                deblocking_disabled: sh_params[(s * 4) as usize] != 0,
                across_slices_enabled: sh_params[(s * 4 + 1) as usize] != 0,
                beta_offset_div2: sh_params[(s * 4 + 2) as usize],
                tc_offset_div2: sh_params[(s * 4 + 3) as usize],
            });
        }

        let tiles_opt = if use_tiles {
            Some(Tiles { tile_id: &tile_id, ctb_addr_rs_to_ts: &rs2ts })
        } else {
            None
        };
        let ctx = DeblockCtx {
            pic_w,
            pic_h,
            bit_depth_y: bit_depth,
            bit_depth_c: bit_depth,
            pcm_filter_disabled: dense_pcm && rng.below(2) == 0,
            sub_w,
            sub_h,
            ctb_log2,
            ctbs_w,
            chroma_array_type,
            loop_filter_across_tiles: !(use_tiles && rng.below(2) == 0),
            tiles: tiles_opt.as_ref(),
            pps_cb_qp_offset: rng.i32(25) - 12,
            pps_cr_qp_offset: rng.i32(25) - 12,
            slice_idx: if multi_slice { Some(&slice_idx) } else { None },
            sh: &sh_rs,
            cu: &cu_rs,
            cu_stride: pic_w / 4,
            min_cb_log2: 2,
            grid_stride: pic_w / 4,
            motion: &motion_rs,
            cbf_luma: &cbf_luma,
            log2_tu_size: &log2_tu,
            edge_v: &edge_v,
            edge_h: &edge_h,
            poc_l0: &poc_l0,
            poc_l1: &poc_l1,
        };

        let mut out_y = plane_y.clone();
        let mut out_cb = plane_cb.clone();
        let mut out_cr = plane_cr.clone();
        apply_deblocking(
            &ctx,
            &mut [
                Plane { data: &mut out_y, width: pic_w, height: pic_h, stride: pic_w },
                Plane { data: &mut out_cb, width: comp_w, height: comp_h, stride: comp_w },
                Plane { data: &mut out_cr, width: comp_w, height: comp_h, stride: comp_w },
            ],
        );

        // --- C++ oracle ---
        let mut c_y = vec![0u16; (pic_w * pic_h) as usize];
        let mut c_cb = vec![0u16; (comp_w * comp_h) as usize];
        let mut c_cr = vec![0u16; (comp_w * comp_h) as usize];
        unsafe {
            let rc = ffi_test::hevcdec_test_deblock_run(
                pic_w,
                pic_h,
                ctb_log2,
                sub_w,
                sub_h,
                chroma_array_type,
                bit_depth,
                bit_depth,
                ctx.pcm_filter_disabled as i32,
                ctx.loop_filter_across_tiles as i32,
                if use_tiles { tile_id_bytes.as_ptr() } else { std::ptr::null() },
                if use_tiles { rs2ts.as_ptr() } else { std::ptr::null() },
                ctx.pps_cb_qp_offset,
                ctx.pps_cr_qp_offset,
                plane_y.as_ptr(),
                plane_cb.as_ptr(),
                plane_cr.as_ptr(),
                pic_w,
                comp_w,
                comp_w,
                if multi_slice { slice_idx.as_ptr() } else { std::ptr::null() },
                n_slices,
                sh_params.as_ptr(),
                cu_fields.as_ptr(),
                motion.as_ptr(),
                cbf_luma.as_ptr(),
                log2_tu.as_ptr(),
                edge_v.as_ptr(),
                edge_h.as_ptr(),
                if n_ref_l0 > 0 { poc_l0.as_ptr() } else { std::ptr::null() },
                n_ref_l0,
                if n_ref_l1 > 0 { poc_l1.as_ptr() } else { std::ptr::null() },
                n_ref_l1,
                c_y.as_mut_ptr(),
                c_cb.as_mut_ptr(),
                c_cr.as_mut_ptr(),
            );
            assert_eq!(rc, 0, "oracle rejected args");
        }

        first_diff(&out_y, &c_y, pic_w, "luma");
        if chroma_array_type != 0 {
            first_diff(&out_cb, &c_cb, comp_w, "Cb");
            first_diff(&out_cr, &c_cr, comp_w, "Cr");
        }
    }

    /// Byte-exactness check that reports the first differing sample instead of
    /// dumping both full planes.
    fn first_diff(a: &[u16], b: &[u16], w: i32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length mismatch");
        for i in 0..a.len() {
            if a[i] != b[i] {
                panic!(
                    "{what}: first diff at x={} y={} (idx {}): rust={} cpp={}",
                    (i as i32) % w,
                    (i as i32) / w,
                    i,
                    a[i],
                    b[i]
                );
            }
        }
    }

    #[test]
    fn deblock_single_slice_matches_cpp() {
        let mut rng = Rng::new(0xdb01);
        for _ in 0..80 {
            run_deblock_case(&mut rng, false, false, false);
        }
    }

    #[test]
    fn deblock_multi_slice_matches_cpp() {
        let mut rng = Rng::new(0xdb02);
        for _ in 0..60 {
            run_deblock_case(&mut rng, true, false, false);
        }
    }

    #[test]
    fn deblock_tiles_matches_cpp() {
        let mut rng = Rng::new(0xdb03);
        for _ in 0..60 {
            let ms = rng.below(2) == 0;
            run_deblock_case(&mut rng, ms, true, false);
        }
    }

    #[test]
    fn deblock_pcm_bypass_matches_cpp() {
        let mut rng = Rng::new(0xdb04);
        for _ in 0..50 {
            let ms = rng.below(2) == 0;
            run_deblock_case(&mut rng, ms, false, true);
        }
    }

    #[test]
    fn spec_tables_spot_check() {
        // Table 8-12 (beta'/tC') and Table 8-10 (QpC) anchor values.
        assert_eq!(BETA_TABLE[16], 6);
        assert_eq!(BETA_TABLE[17], 7);
        assert_eq!(BETA_TABLE[29], 20);
        assert_eq!(BETA_TABLE[51], 64);
        assert_eq!(TC_TABLE[18], 1);
        assert_eq!(TC_TABLE[27], 2);
        assert_eq!(TC_TABLE[31], 3);
        assert_eq!(TC_TABLE[42], 7);
        assert_eq!(TC_TABLE[53], 24);
        assert_eq!(QPC_TABLE[0], 0);
        assert_eq!(QPC_TABLE[28], 28);
        assert_eq!(QPC_TABLE[29], 29);
        assert_eq!(QPC_TABLE[30], 29); // Table 8-10 plateau at Qp' = 29/30
        assert_eq!(QPC_TABLE[57], 51);
    }

    #[test]
    fn no_edges_is_noop() {
        let mut plane = vec![7u16; 64 * 64];
        let n_blocks = 16 * 16;
        let cu = vec![DeblockCu::default(); n_blocks];
        let motion = vec![MotionInfo::default(); n_blocks];
        let zeros = vec![0u8; n_blocks];
        let sh = [DeblockSliceParams::default()];
        let ctx = DeblockCtx {
            pic_w: 64,
            pic_h: 64,
            bit_depth_y: 8,
            bit_depth_c: 8,
            pcm_filter_disabled: false,
            sub_w: 2,
            sub_h: 2,
            ctb_log2: 6,
            ctbs_w: 1,
            chroma_array_type: 1,
            loop_filter_across_tiles: true,
            tiles: None,
            pps_cb_qp_offset: 0,
            pps_cr_qp_offset: 0,
            slice_idx: None,
            sh: &sh,
            cu: &cu,
            cu_stride: 16,
            min_cb_log2: 2,
            grid_stride: 16,
            motion: &motion,
            cbf_luma: &zeros,
            log2_tu_size: &zeros,
            edge_v: &zeros,
            edge_h: &zeros,
            poc_l0: &[],
            poc_l1: &[],
        };
        apply_deblocking(
            &ctx,
            &mut [Plane { data: &mut plane, width: 64, height: 64, stride: 64 }],
        );
        assert!(plane.iter().all(|&s| s == 7));
    }
}

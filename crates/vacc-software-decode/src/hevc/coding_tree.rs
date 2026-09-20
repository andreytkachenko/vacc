//! Port of `hevc/decoding/coding_tree.{h,cpp}` — slice_segment_data,
//! coding_quadtree and coding_unit (§7.3.8), QP derivation (§8.6.1), SAO
//! parameter parsing, and WPP parallel decode (§9.2.2) using rayon as the
//! thread pool (replaces the C++ `ThreadPool`).

use std::cell::RefCell;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Condvar, Mutex};

use crate::hevc::cabac::{CabacContext, CabacEngine};
use crate::hevc::cabac_tables::NUM_CABAC_CONTEXTS;
use crate::hevc::bitreader::{BitstreamReader, coded_to_rbsp_offset};
use crate::hevc::interpolation::FIR_SCRATCH_MAX;
use crate::hevc::inter_prediction::{
    decode_prediction_unit_inter, get_pu_motion, perform_inter_prediction, DpbView,
};
use crate::hevc::intra_prediction::perform_intra_prediction;
use crate::hevc::picture::{Picture, PuMotionInfo};
use crate::hevc::residual_coding::decode_residual_coding;
use crate::hevc::syntax_elements::*;
use crate::hevc::transform::{
    DequantParams, ScalingListData, perform_dequant, perform_transform_inverse,
};
use crate::hevc::types::{
    PartMode, PredMode, SliceType, Sps, Pps, SliceHeader, clip3,
};

// Test-only: per-CTU bit-position trace for differential debugging (compare
// against the C++ `HEVC_DEBUG_FILTER=TREE` output).
pub(crate) fn hevc_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("TIER_E_TRACE").is_some() || std::env::var_os("RUST_TREE_TRACE").is_some()
    })
}

// ============================================================
// Per-CU info stored in the picture grid (for neighbour access)
// ============================================================

#[derive(Clone, Copy, Debug)]
pub struct CuInfo {
    pub pred_mode: PredMode,
    pub part_mode: PartMode,
    pub log2_cb_size: i32,
    /// DC default.
    pub intra_mode_luma: i32,
    pub qp_y: i32,
    pub is_pcm: bool,
    pub cu_transquant_bypass: bool,
    /// For the rqt_root_cbf condition (§7.3.8.5).
    pub merge_flag: bool,
}

impl Default for CuInfo {
    fn default() -> Self {
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
}

// ============================================================
// SAO params per CTU (parsed in coding_tree, applied by sao.rs)
// ============================================================

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SaoParams {
    /// 0=off, 1=band, 2=edge.
    pub sao_type_idx: [i32; 3],
    /// Derived offset values (spec §7.4.9.3); index 2 (flat) is always 0.
    pub sao_offset_val: [[i32; 5]; 3],
    pub sao_band_position: [i32; 3],
    pub sao_eo_class: [i32; 3],
}

// ============================================================
// Decoding context passed through the coding tree
// ============================================================

pub struct DecodingContext<'bs, 'a> {
    pub sps: &'a Sps,
    pub pps: &'a Pps,
    pub sh: &'a SliceHeader,
    pub pic: &'a mut Picture,
    pub dpb: &'a DpbView<'a>,
    /// CABAC engine (owns the bitstream reader; raw-bit access via
    /// `cabac.bitstream()`).
    pub cabac: &'bs mut CabacEngine<'bs>,

    // Dequant scaling lists (§8.6.3)
    pub sps_scaling_list_enabled: bool,
    pub sps_scaling_list: &'a ScalingListData,
    pub pps_scaling_list_present: bool,
    pub pps_scaling_list: &'a ScalingListData,

    // CU info grid: indexed by min-CB position (x >> MinCbLog2SizeY, ...)
    pub cu_info: &'a mut [CuInfo],
    /// = PicWidthInMinCbsY.
    pub cu_info_stride: i32,

    // Intra mode storage per PU at min-TB granularity (for neighbour MPM)
    pub intra_pred_mode_y: &'a mut [i32],
    pub intra_pred_mode_c: &'a mut [i32],
    pub intra_pred_mode_stride: i32,

    // Inter: per-PU motion info at min-PU (4x4) granularity
    pub motion_info: &'a mut [PuMotionInfo],
    /// = pic_width / MinTbSizeY.
    pub motion_info_stride: i32,

    // Deblocking data at min-TB (4x4) granularity
    pub cbf_luma_grid: &'a mut [u8],
    pub log2_tu_size_grid: &'a mut [u8],
    pub edge_flags_v: &'a mut [u8],
    pub edge_flags_h: &'a mut [u8],
    /// = pic_width / 4.
    pub filter_grid_stride: i32,

    // SAO params per CTU
    pub sao_params: &'a mut [SaoParams],
    /// = PicWidthInCtbsY.
    pub sao_params_stride: i32,

    // Slice index per CTU (cross-slice boundary detection)
    pub slice_idx: Option<&'a mut [u8]>,
    /// Slice index being decoded.
    pub current_slice_idx: i32,

    // QP tracking (§8.6.1)
    /// QP of last CU in previous QG.
    pub qp_y_prev: i32,
    /// Saved QpY_prev at QG boundary start.
    pub qp_y_prev_qg: i32,
    pub is_cu_qp_delta_coded: bool,
    pub cu_qp_delta_val: i32,
    /// Current CU position for §8.6.1 QP derivation in TU.
    pub cu_x0: i32,
    pub cu_y0: i32,

    // WPP context storage (serial path carryover; the parallel path keeps
    // per-row copies in `decode_wpp_parallel`).
    pub wpp_saved_contexts: [CabacContext; NUM_CABAC_CONTEXTS],
    pub wpp_contexts_available: bool,

    /// Test knob: force the serial CTU scan even when WPP is enabled. Mirrors
    /// the C++ gate `thread_pool && num_workers > 1` (the oracle has no pool).
    pub wpp_enabled: bool,

    // Reusable MC scratch buffers (grow-only, never re-zeroed): every MC
    // consumer writes all samples of the block it fills (interpolation paths
    // cover the full PU; weighted prediction clips and covers the full block),
    // so stale contents from the previous user are never read.
    pub mc_l0: Vec<i16>,
    pub mc_l1: Vec<i16>,
    /// Final MC output for the current component (written by weighted
    /// prediction, consumed by `write_mc_block`).
    pub mc_out: Vec<i16>,
    /// 2D FIR intermediate (max 64*71 luma / 32*35 chroma). Grow-only: pass 1
    /// writes every element pass 2 reads, so no zero-init is needed.
    pub fir_tmp: Vec<i16>,
}

impl<'bs, 'a> DecodingContext<'bs, 'a> {
    /// Get CU info at luma sample position.
    #[inline]
    pub fn cu_at(&self, x: i32, y: i32) -> &CuInfo {
        let shift = self.sps.min_cb_log2_size_y;
        &self.cu_info[((y >> shift) * self.cu_info_stride + (x >> shift)) as usize]
    }

    #[inline]
    pub fn cu_at_mut(&mut self, x: i32, y: i32) -> &mut CuInfo {
        let shift = self.sps.min_cb_log2_size_y;
        &mut self.cu_info[((y >> shift) * self.cu_info_stride + (x >> shift)) as usize]
    }

    /// Get intra mode at position (luma) — min-TB granularity.
    #[inline]
    pub fn intra_mode_at(&self, x: i32, y: i32) -> i32 {
        let shift = self.sps.min_tb_log2_size_y;
        self.intra_pred_mode_y[((y >> shift) * self.intra_pred_mode_stride + (x >> shift)) as usize]
    }

    #[inline]
    pub fn chroma_mode_at(&self, x: i32, y: i32) -> i32 {
        let shift = self.sps.min_tb_log2_size_y;
        self.intra_pred_mode_c[((y >> shift) * self.intra_pred_mode_stride + (x >> shift)) as usize]
    }

    pub fn set_intra_mode(&mut self, x: i32, y: i32, size: i32, mode: i32) {
        let min_tb_size = self.sps.min_tb_size_y;
        let x0 = x / min_tb_size;
        let y0 = y / min_tb_size;
        let n = (size / min_tb_size).max(1);
        for j in 0..n {
            for i in 0..n {
                self.intra_pred_mode_y[((y0 + j) * self.intra_pred_mode_stride + (x0 + i)) as usize] = mode;
            }
        }
    }

    pub fn set_chroma_mode(&mut self, x: i32, y: i32, size: i32, mode: i32) {
        let min_tb_size = self.sps.min_tb_size_y;
        let x0 = x / min_tb_size;
        let y0 = y / min_tb_size;
        let n = (size / min_tb_size).max(1);
        for j in 0..n {
            for i in 0..n {
                self.intra_pred_mode_c[((y0 + j) * self.intra_pred_mode_stride + (x0 + i)) as usize] = mode;
            }
        }
    }

    /// Build the per-row context for WPP from shared state (see the SAFETY
    /// invariant on [`WppShared`]).
    fn from_wpp(g: &WppShared<'a>, cabac: &'bs mut CabacEngine<'bs>) -> Self {
        DecodingContext {
            sps: g.sps,
            pps: g.pps,
            sh: g.sh,
            pic: g.pic_mut(),
            dpb: g.dpb,
            cabac,
            sps_scaling_list_enabled: g.sps_scaling_list_enabled,
            sps_scaling_list: g.sps_scaling_list,
            pps_scaling_list_present: g.pps_scaling_list_present,
            pps_scaling_list: g.pps_scaling_list,
            cu_info: g.cu_info_mut(),
            cu_info_stride: g.cu_info_stride,
            intra_pred_mode_y: g.intra_pred_mode_y_mut(),
            intra_pred_mode_c: g.intra_pred_mode_c_mut(),
            intra_pred_mode_stride: g.intra_pred_mode_stride,
            motion_info: g.motion_info_mut(),
            motion_info_stride: g.motion_info_stride,
            cbf_luma_grid: g.cbf_luma_grid_mut(),
            log2_tu_size_grid: g.log2_tu_size_grid_mut(),
            edge_flags_v: g.edge_flags_v_mut(),
            edge_flags_h: g.edge_flags_h_mut(),
            filter_grid_stride: g.filter_grid_stride,
            sao_params: g.sao_params_mut(),
            sao_params_stride: g.sao_params_stride,
            slice_idx: g.slice_idx_opt(),
            current_slice_idx: g.current_slice_idx,
            qp_y_prev: 0,
            qp_y_prev_qg: 0,
            is_cu_qp_delta_coded: false,
            cu_qp_delta_val: 0,
            cu_x0: 0,
            cu_y0: 0,
            wpp_saved_contexts: [CabacContext::default(); NUM_CABAC_CONTEXTS],
            wpp_contexts_available: false,
            wpp_enabled: false,
            mc_l0: Vec::new(),
            mc_l1: Vec::new(),
            mc_out: Vec::new(),
            fir_tmp: Vec::new(),
        }
    }
}

// ============================================================
// SAO parsing (§7.3.8.3)
// ============================================================

fn decode_sao(ctx: &mut DecodingContext, rx: i32, ry: i32) {
    let sps = ctx.sps;
    let pps = ctx.pps;
    let sh = ctx.sh;
    let ctb_addr_rs = ry * sps.pic_width_in_ctbs_y + rx;
    let mut sao = SaoParams::default();

    let mut sao_merge_left_flag = false;
    let mut sao_merge_up_flag = false;

    if rx > 0 {
        let left_in_slice = ctb_addr_rs > sh.slice_segment_address;
        let ts = pps.ctb_addr_rs_to_ts[ctb_addr_rs as usize];
        let left_ts = pps.ctb_addr_rs_to_ts[(ctb_addr_rs - 1) as usize];
        let left_in_tile = pps.tile_id.is_empty() || pps.tile_id[ts as usize] == pps.tile_id[left_ts as usize];
        if left_in_slice && left_in_tile {
            sao_merge_left_flag = decode_sao_merge_flag(ctx.cabac) != 0;
        }
    }

    if ry > 0 && !sao_merge_left_flag {
        let up_addr = ctb_addr_rs - sps.pic_width_in_ctbs_y;
        let up_in_slice = up_addr >= sh.slice_segment_address;
        let ts = pps.ctb_addr_rs_to_ts[ctb_addr_rs as usize];
        let up_ts = pps.ctb_addr_rs_to_ts[up_addr as usize];
        let up_in_tile = pps.tile_id.is_empty() || pps.tile_id[ts as usize] == pps.tile_id[up_ts as usize];
        if up_in_slice && up_in_tile {
            sao_merge_up_flag = decode_sao_merge_flag(ctx.cabac) != 0;
        }
    }

    if sao_merge_left_flag {
        // Copy all params from left CTU
        let left = ctx.sao_params[(ry * ctx.sao_params_stride + (rx - 1)) as usize];
        return store_sao(ctx, rx, ry, left);
    }
    if sao_merge_up_flag {
        // Copy all params from above CTU
        let up = ctx.sao_params[((ry - 1) * ctx.sao_params_stride + rx) as usize];
        return store_sao(ctx, rx, ry, up);
    }

    let num_comp = if sps.chroma_array_type != 0 { 3 } else { 1 };
    // §7.4.9.3: SaoTypeIdx[2] = SaoTypeIdx[1]
    let mut sao_type_idx_chroma = 0i32;
    for c_idx in 0..num_comp {
        if (sh.slice_sao_luma_flag && c_idx == 0) || (sh.slice_sao_chroma_flag && c_idx > 0) {
            let sao_type_idx = if c_idx == 0 {
                decode_sao_type_idx(ctx.cabac)
            } else if c_idx == 1 {
                let t = decode_sao_type_idx(ctx.cabac);
                sao_type_idx_chroma = t;
                t
            } else {
                // §7.4.9.3: SaoTypeIdx[2][rx][ry] = SaoTypeIdx[1][rx][ry]
                sao_type_idx_chroma
            };
            sao.sao_type_idx[c_idx as usize] = sao_type_idx;

            if sao_type_idx != 0 {
                let bit_depth = if c_idx == 0 { sps.bit_depth_y } else { sps.bit_depth_c };
                let max_off = (1 << (bit_depth.min(10) - 5)) - 1;
                let mut sao_offset_abs_val = [0i32; 4];
                for abs_val in sao_offset_abs_val.iter_mut() {
                    // sao_offset_abs: TR cMax=maxOff, bypass
                    let mut val = 0i32;
                    for _k in 0..max_off {
                        if ctx.cabac.decode_bypass() == 0 {
                            break;
                        }
                        val += 1;
                    }
                    *abs_val = val;
                }
                let mut sao_offset_sign = [0i32; 4];
                if sao_type_idx == 1 {
                    // Band offset: sign parsed, band_position parsed
                    for i in 0..4 {
                        if sao_offset_abs_val[i] != 0 {
                            sao_offset_sign[i] = ctx.cabac.decode_bypass();
                        }
                    }
                    sao.sao_band_position[c_idx as usize] = ctx.cabac.decode_bypass_bins(5);
                } else {
                    // Edge offset: sign is fixed per category, eo_class parsed
                    // §7.4.9.3: Cr shares Cb eo_class — coded once, at cIdx==1
                    if c_idx == 0 {
                        sao.sao_eo_class[c_idx as usize] = ctx.cabac.decode_bypass_bins(2);
                    } else if c_idx == 1 {
                        let eo_class = ctx.cabac.decode_bypass_bins(2);
                        sao.sao_eo_class[1] = eo_class;
                        sao.sao_eo_class[2] = eo_class;
                    }
                }

                // §7.4.9.3: derive SaoOffsetVal
                sao.sao_offset_val[c_idx as usize][2] = 0; // flat
                if sao_type_idx == 2 {
                    // Edge offset: categories 0,1 positive; 3,4 negative
                    sao.sao_offset_val[c_idx as usize][0] = sao_offset_abs_val[0]; // valley
                    sao.sao_offset_val[c_idx as usize][1] = sao_offset_abs_val[1]; // concave
                    sao.sao_offset_val[c_idx as usize][3] = -sao_offset_abs_val[2]; // convex
                    sao.sao_offset_val[c_idx as usize][4] = -sao_offset_abs_val[3]; // peak
                } else {
                    // Band offset: sign from bitstream
                    for i in 0..4 {
                        let mut val = sao_offset_abs_val[i];
                        if sao_offset_sign[i] != 0 {
                            val = -val;
                        }
                        sao.sao_offset_val[c_idx as usize][i] = val;
                    }
                    sao.sao_offset_val[c_idx as usize][4] = 0; // unused for band
                }
            }
        }
    }
    store_sao(ctx, rx, ry, sao);
}

#[inline]
fn store_sao(ctx: &mut DecodingContext, rx: i32, ry: i32, sao: SaoParams) {
    ctx.sao_params[(ry * ctx.sao_params_stride + rx) as usize] = sao;
}

// ============================================================
// split_cu_flag / cu_skip_flag context derivation (§9.3.4.2.2)
// ============================================================

/// §6.4.1: check if neighbour CTU at (nx,ny) is available (same slice, same
/// tile, already decoded).
fn is_ctb_available(ctx: &DecodingContext, cur_x: i32, cur_y: i32, nb_x: i32, nb_y: i32) -> bool {
    let sps = ctx.sps;
    let pps = ctx.pps;
    if nb_x < 0
        || nb_y < 0
        || nb_x >= sps.pic_width_in_luma_samples
        || nb_y >= sps.pic_height_in_luma_samples
    {
        return false;
    }
    let ctb_size = 1 << sps.ctb_log2_size_y;
    let cur_addr = (cur_y / ctb_size) * sps.pic_width_in_ctbs_y + (cur_x / ctb_size);
    let nb_addr = (nb_y / ctb_size) * sps.pic_width_in_ctbs_y + (nb_x / ctb_size);
    // Must be already decoded (in tile scan order)
    if pps.ctb_addr_rs_to_ts[nb_addr as usize] > pps.ctb_addr_rs_to_ts[cur_addr as usize] {
        return false;
    }
    // §6.4.1: SliceAddrRs must match
    if let Some(slice_idx) = &ctx.slice_idx
        && slice_idx[nb_addr as usize] != slice_idx[cur_addr as usize]
    {
        return false;
    }
    // Same tile check
    if !pps.tile_id.is_empty()
        && pps.tile_id[pps.ctb_addr_rs_to_ts[nb_addr as usize] as usize]
            != pps.tile_id[pps.ctb_addr_rs_to_ts[cur_addr as usize] as usize]
    {
        return false;
    }
    true
}

fn derive_split_cu_flag_ctx(ctx: &DecodingContext, x0: i32, y0: i32, log2_cb_size: i32) -> i32 {
    let mut ctx_inc = 0i32;

    // Left neighbour — §6.4.1 availability
    let x_l = x0 - 1;
    if x_l >= 0 && is_ctb_available(ctx, x0, y0, x_l, y0) {
        let cu_l = ctx.cu_at(x_l, y0);
        if cu_l.log2_cb_size > 0 && cu_l.log2_cb_size < log2_cb_size {
            ctx_inc += 1;
        }
    }

    // Above neighbour — §6.4.1 availability
    let y_a = y0 - 1;
    if y_a >= 0 && is_ctb_available(ctx, x0, y0, x0, y_a) {
        let cu_a = ctx.cu_at(x0, y_a);
        if cu_a.log2_cb_size > 0 && cu_a.log2_cb_size < log2_cb_size {
            ctx_inc += 1;
        }
    }

    ctx_inc
}

fn derive_cu_skip_flag_ctx(ctx: &DecodingContext, x0: i32, y0: i32) -> i32 {
    let mut ctx_inc = 0i32;

    let x_l = x0 - 1;
    if x_l >= 0
        && is_ctb_available(ctx, x0, y0, x_l, y0)
        && ctx.cu_at(x_l, y0).pred_mode == PredMode::Skip
    {
        ctx_inc += 1;
    }

    let y_a = y0 - 1;
    if y_a >= 0
        && is_ctb_available(ctx, x0, y0, x0, y_a)
        && ctx.cu_at(x0, y_a).pred_mode == PredMode::Skip
    {
        ctx_inc += 1;
    }

    ctx_inc
}

// ============================================================
// MPM derivation (§8.4.2)
// ============================================================

fn derive_mpm(ctx: &DecodingContext, x0: i32, y0: i32, cand_mode_list: &mut [i32; 3]) {
    // Left neighbour — §8.4.2 step 2 with §6.4.1 availability
    let x_l = x0 - 1;
    let mut cand_a = 1i32; // DC default if not available
    if x_l >= 0 && is_ctb_available(ctx, x0, y0, x_l, y0) {
        let cu_l = ctx.cu_at(x_l, y0);
        if cu_l.pred_mode == PredMode::Intra {
            cand_a = ctx.intra_mode_at(x_l, y0);
        }
    }

    // Above neighbour — §8.4.2 step 2: cross-CTU-row -> DC, plus §6.4.1
    let y_a = y0 - 1;
    let mut cand_b = 1i32; // DC default if not available
    if y_a >= 0 {
        // §8.4.2: if yPb-1 < ((yPb >> CtbLog2SizeY) << CtbLog2SizeY), the
        // above neighbour is in a different CTB row -> use DC
        let ctb_row_start = (y0 >> ctx.sps.ctb_log2_size_y) << ctx.sps.ctb_log2_size_y;
        if y_a >= ctb_row_start && is_ctb_available(ctx, x0, y0, x0, y_a) {
            let cu_a = ctx.cu_at(x0, y_a);
            if cu_a.pred_mode == PredMode::Intra {
                cand_b = ctx.intra_mode_at(x0, y_a);
            }
        }
    }

    // Derive 3 MPM candidates
    if cand_a == cand_b {
        if cand_a < 2 {
            cand_mode_list[0] = 0; // Planar
            cand_mode_list[1] = 1; // DC
            cand_mode_list[2] = 26; // Vertical
        } else {
            cand_mode_list[0] = cand_a;
            cand_mode_list[1] = 2 + ((cand_a + 29) % 32); // eq 8-25
            cand_mode_list[2] = 2 + ((cand_a - 2 + 1) % 32); // eq 8-26
        }
    } else {
        cand_mode_list[0] = cand_a;
        cand_mode_list[1] = cand_b;
        if cand_a != 0 && cand_b != 0 {
            cand_mode_list[2] = 0; // Planar
        } else if cand_a != 1 && cand_b != 1 {
            cand_mode_list[2] = 1; // DC
        } else {
            cand_mode_list[2] = 26; // Vertical
        }
    }
}

// ============================================================
// Chroma intra mode derivation (§8.4.3)
// ============================================================

fn derive_chroma_intra_mode(coded_mode: i32, luma_mode: i32) -> i32 {
    // coded_mode: 0=Planar, 1=V(26), 2=H(10), 3=DC(1), 4=DM(=luma_mode)
    if coded_mode == 4 {
        return luma_mode;
    }

    let chroma_cand = [0, 26, 10, 1];
    let mut mode = chroma_cand[coded_mode as usize];

    // If chroma mode equals luma mode, replace with mode 34
    if mode == luma_mode {
        mode = 34;
    }

    mode
}

// ============================================================
// QP derivation (§8.6.1)
// ============================================================

fn derive_qp_y(ctx: &DecodingContext, x_cb: i32, y_cb: i32) -> i32 {
    let sps = ctx.sps;
    let pps = ctx.pps;
    let sh = ctx.sh;

    if !pps.cu_qp_delta_enabled_flag {
        return sh.slice_qp_y;
    }

    // §8.6.1: derive QG top-left coordinates
    let qg_mask = (1 << pps.log2_min_cu_qp_delta_size) - 1;
    let x_qg = x_cb - (x_cb & qg_mask);
    let y_qg = y_cb - (y_cb & qg_mask);

    // §8.6.1 step 1: qPY_PREV — saved at QG boundary, not updated within QG
    let qp_prev = ctx.qp_y_prev_qg;

    // §8.6.1 step 2: qPY_A from CU at (xQg - 1, yQg)
    let mut qp_pred_a = qp_prev;
    if x_qg >= 1 {
        // Check that (xQg-1, yQg) is in the same CTB as current
        let ctb_addr_cur = (y_cb >> sps.ctb_log2_size_y) * sps.pic_width_in_ctbs_y
            + (x_cb >> sps.ctb_log2_size_y);
        let ctb_addr_a = (y_qg >> sps.ctb_log2_size_y) * sps.pic_width_in_ctbs_y
            + ((x_qg - 1) >> sps.ctb_log2_size_y);
        if pps.ctb_addr_rs_to_ts[ctb_addr_a as usize] == pps.ctb_addr_rs_to_ts[ctb_addr_cur as usize] {
            qp_pred_a = ctx.cu_at(x_qg - 1, y_qg).qp_y;
        }
    }

    // §8.6.1 step 3: qPY_B from CU at (xQg, yQg - 1)
    let mut qp_pred_b = qp_prev;
    if y_qg >= 1 {
        let ctb_addr_cur = (y_cb >> sps.ctb_log2_size_y) * sps.pic_width_in_ctbs_y
            + (x_cb >> sps.ctb_log2_size_y);
        let ctb_addr_b = ((y_qg - 1) >> sps.ctb_log2_size_y) * sps.pic_width_in_ctbs_y
            + (x_qg >> sps.ctb_log2_size_y);
        if pps.ctb_addr_rs_to_ts[ctb_addr_b as usize] == pps.ctb_addr_rs_to_ts[ctb_addr_cur as usize] {
            qp_pred_b = ctx.cu_at(x_qg, y_qg - 1).qp_y;
        }
    }

    // §8.6.1 step 4: predicted QP
    let qp_pred = (qp_pred_a + qp_pred_b + 1) >> 1;

    // §8.6.1 eq 8-283
    let q = (qp_pred + ctx.cu_qp_delta_val + 52 + 2 * sps.qp_bd_offset_y) % (52 + sps.qp_bd_offset_y)
        - sps.qp_bd_offset_y;
    if !((0..=63).contains(&q)) {
        eprintln!(
            "DBG derive_qp_y OUT: cu=({}, {}) qg=({}, {}) a={} b={} pred={} delta={} bdoff={:?} -> {}",
            x_cb, y_cb, x_qg, y_qg, qp_pred_a, qp_pred_b, qp_pred, ctx.cu_qp_delta_val, sps.qp_bd_offset_y, q
        );
    }
    q
}

// ============================================================
// Reconstruction: pred + residual, clipping (§8.6.5)
// ============================================================

fn reconstruct_block(
    ctx: &mut DecodingContext,
    x0: i32,
    y0: i32,
    log2_size: i32,
    c_idx: i32,
    pred: &[i16],
    residual: &[i16],
) {
    let size = 1 << log2_size;
    let bit_depth = if c_idx == 0 { ctx.sps.bit_depth_y } else { ctx.sps.bit_depth_c };
    let max_val = (1 << bit_depth) - 1;

    // For chroma, convert luma coordinates to chroma
    let x_c = if c_idx > 0 { x0 / ctx.sps.sub_width_c } else { x0 };
    let y_c = if c_idx > 0 { y0 / ctx.sps.sub_height_c } else { y0 };

    let pic_w = if c_idx == 0 {
        ctx.sps.pic_width_in_luma_samples
    } else {
        ctx.sps.pic_width_in_luma_samples / ctx.sps.sub_width_c
    };
    let pic_h = if c_idx == 0 {
        ctx.sps.pic_height_in_luma_samples
    } else {
        ctx.sps.pic_height_in_luma_samples / ctx.sps.sub_height_c
    };

    for j in 0..size {
        for i in 0..size {
            if x_c + i >= pic_w || y_c + j >= pic_h {
                continue;
            }
            let val = pred[(j * size + i) as usize] as i32 + residual[(j * size + i) as usize] as i32;
            let val = clip3(0, max_val, val);
            *ctx.pic.sample_mut(c_idx as usize, x_c + i, y_c + j) = val as u16;
        }
    }
}

// ============================================================
// WPP parallel decode — §7.3.8.1 with wavefront parallelism
// ============================================================

/// Per-row synchronization state for the WPP pipeline.
struct RowSync {
    /// Last completed column (-1 = not started).
    completed_col: AtomicI32,
    lock: Mutex<RowState>,
    cv: Condvar,
}

/// State guarded by [`RowSync::lock`].
struct RowState {
    /// §9.3.2.4: this row's CABAC contexts, saved after its 2nd CTU and
    /// consumed by the next row via [`RowSync::wait_col_take_contexts`].
    wpp_contexts: [CabacContext; NUM_CABAC_CONTEXTS],
}

impl RowSync {
    fn new() -> Self {
        RowSync {
            completed_col: AtomicI32::new(-1),
            lock: Mutex::new(RowState {
                wpp_contexts: [CabacContext::default(); NUM_CABAC_CONTEXTS],
            }),
            cv: Condvar::new(),
        }
    }

    fn wait_col(&self, col: i32) {
        let guard = self.lock.lock().unwrap();
        let _guard = self
            .cv
            .wait_while(guard, |_guard| self.completed_col.load(Ordering::Acquire) < col)
            .unwrap();
    }

    /// Publish this row's saved contexts. Called after CTU col 1, before
    /// [`RowSync::complete_col`] for the same column, so any waiter that
    /// observes `completed_col >= 1` (same mutex) also sees these contexts.
    fn save_contexts(&self, ctx: &[CabacContext]) {
        let mut state = self.lock.lock().unwrap();
        state.wpp_contexts.copy_from_slice(ctx);
    }

    /// Wait for `col` to complete, then take this row's saved contexts.
    /// The mutex makes the save happen-before the take.
    fn wait_col_take_contexts(&self, col: i32) -> [CabacContext; NUM_CABAC_CONTEXTS] {
        let guard = self.lock.lock().unwrap();
        let state = self
            .cv
            .wait_while(guard, |_guard| self.completed_col.load(Ordering::Acquire) < col)
            .unwrap();
        state.wpp_contexts
    }

    fn complete_col(&self, col: i32) {
        self.completed_col.store(col, Ordering::Release);
        self.cv.notify_all();
    }
}

/// Shared decode state for WPP row tasks.
///
/// The grid/picture fields are raw pointers into caller-owned buffers, never
/// exposed directly: all access goes through the accessors below. They alias
/// each other in a controlled way, justified by one invariant:
///
/// SAFETY (basis of the `Send`/`Sync` impls and the `&mut`-returning
/// accessors): the WPP diagonal dependency (row r+1 column c only proceeds
/// after row r column min(c+1, last) completed, synchronized by `RowSync`
/// condvars) guarantees that any grid cell is read by a later row only after
/// the writing row has finished it — the same invariant the C++ ThreadPool
/// condvars enforce. Rows write disjoint CTU regions of every grid and of the
/// picture; the only cross-row reads are of already-completed neighbour rows.
pub struct WppShared<'a> {
    pub sps: &'a Sps,
    pub pps: &'a Pps,
    pub sh: &'a SliceHeader,
    pub dpb: &'a DpbView<'a>,
    /// Full slice-segment RBSP (all rows share the same read-only buffer;
    /// each row seeks to its substream start).
    pub rbsp: &'a [u8],

    pub sps_scaling_list_enabled: bool,
    pub sps_scaling_list: &'a ScalingListData,
    pub pps_scaling_list_present: bool,
    pub pps_scaling_list: &'a ScalingListData,

    pic: *mut Picture,
    cu_info: (*mut CuInfo, usize),
    pub cu_info_stride: i32,
    intra_pred_mode_y: (*mut i32, usize),
    intra_pred_mode_c: (*mut i32, usize),
    pub intra_pred_mode_stride: i32,
    motion_info: (*mut PuMotionInfo, usize),
    pub motion_info_stride: i32,
    cbf_luma_grid: (*mut u8, usize),
    log2_tu_size_grid: (*mut u8, usize),
    edge_flags_v: (*mut u8, usize),
    edge_flags_h: (*mut u8, usize),
    pub filter_grid_stride: i32,
    sao_params: (*mut SaoParams, usize),
    pub sao_params_stride: i32,
    /// Null = no slice index tracking.
    slice_idx: *mut u8,
    slice_idx_len: usize,
    pub current_slice_idx: i32,
}

// See the SAFETY invariant on `WppShared`.
unsafe impl<'a> Send for WppShared<'a> {}
unsafe impl<'a> Sync for WppShared<'a> {}

impl<'a> WppShared<'a> {
    // Each accessor materializes a reference from a stored raw pointer.
    // SAFETY: the `WppShared` invariant — concurrent row tasks only touch
    // disjoint CTU regions, synchronized by the WPP diagonal dependency.

    #[inline]
    fn pic_mut(&self) -> &'a mut Picture {
        unsafe { &mut *self.pic }
    }

    #[inline]
    fn cu_info_mut(&self) -> &'a mut [CuInfo] {
        let (p, len) = self.cu_info;
        unsafe { std::slice::from_raw_parts_mut(p, len) }
    }

    #[inline]
    fn intra_pred_mode_y_mut(&self) -> &'a mut [i32] {
        let (p, len) = self.intra_pred_mode_y;
        unsafe { std::slice::from_raw_parts_mut(p, len) }
    }

    #[inline]
    fn intra_pred_mode_c_mut(&self) -> &'a mut [i32] {
        let (p, len) = self.intra_pred_mode_c;
        unsafe { std::slice::from_raw_parts_mut(p, len) }
    }

    #[inline]
    fn motion_info_mut(&self) -> &'a mut [PuMotionInfo] {
        let (p, len) = self.motion_info;
        unsafe { std::slice::from_raw_parts_mut(p, len) }
    }

    #[inline]
    fn cbf_luma_grid_mut(&self) -> &'a mut [u8] {
        let (p, len) = self.cbf_luma_grid;
        unsafe { std::slice::from_raw_parts_mut(p, len) }
    }

    #[inline]
    fn log2_tu_size_grid_mut(&self) -> &'a mut [u8] {
        let (p, len) = self.log2_tu_size_grid;
        unsafe { std::slice::from_raw_parts_mut(p, len) }
    }

    #[inline]
    fn edge_flags_v_mut(&self) -> &'a mut [u8] {
        let (p, len) = self.edge_flags_v;
        unsafe { std::slice::from_raw_parts_mut(p, len) }
    }

    #[inline]
    fn edge_flags_h_mut(&self) -> &'a mut [u8] {
        let (p, len) = self.edge_flags_h;
        unsafe { std::slice::from_raw_parts_mut(p, len) }
    }

    #[inline]
    fn sao_params_mut(&self) -> &'a mut [SaoParams] {
        let (p, len) = self.sao_params;
        unsafe { std::slice::from_raw_parts_mut(p, len) }
    }

    /// Slice-ownership grid, or `None` when tracking is off.
    fn slice_idx_opt(&self) -> Option<&'a mut [u8]> {
        if self.slice_idx.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts_mut(self.slice_idx, self.slice_idx_len) })
        }
    }

    /// Record which slice owns CTU `idx` (no-op when tracking is off).
    #[inline]
    fn set_slice_idx(&self, idx: usize, v: u8) {
        if !self.slice_idx.is_null() {
            unsafe { *self.slice_idx.add(idx) = v };
        }
    }
}

/// Decode one CTU row of the WPP pipeline.
///
/// The per-row CABAC context handoff (§9.3.2.4) goes through `RowSync`
/// state: this row saves its contexts at CTU col 1, and the next row takes
/// them after waiting on this row's `RowSync` — no shared mutable storage.
#[allow(clippy::too_many_arguments)]
fn decode_wpp_row(
    g: &WppShared,
    row: i32,
    start_row: i32,
    substream_byte_pos: &[usize],
    sync: &[RowSync],
) -> usize {
    let sps = g.sps;
    let sh = g.sh;
    let num_cols = sps.pic_width_in_ctbs_y;

    // Per-row BitstreamReader and CabacEngine (shared read-only RBSP buffer)
    let mut row_bs = BitstreamReader::new(g.rbsp);

    // Seek to this row's substream (before the engine borrows the reader)
    let substream_idx = row - start_row;
    if substream_idx >= 0 && (substream_idx as usize) < substream_byte_pos.len() {
        row_bs.seek_to_byte(substream_byte_pos[substream_idx as usize]);
    }

    let mut row_cabac = CabacEngine::new(&mut row_bs);

    // Init CABAC contexts
    if row == start_row {
        if !sh.dependent_slice_segment_flag {
            row_cabac.init_contexts(sh.slice_type as i32, sh.slice_qp_y, sh.cabac_init_flag);
        }
    } else {
        // Wait for row-1 col 1 (2nd CTU) before restoring contexts
        let saved = sync[(row - 1) as usize].wait_col_take_contexts(1);
        row_cabac.load_contexts(&saved);
    }
    row_cabac.init_decoder();

    // Per-row DecodingContext (shallow share — shared grids, pic, etc.)
    let mut row_ctx = DecodingContext::from_wpp(g, &mut row_cabac);
    row_ctx.qp_y_prev = sh.slice_qp_y;
    row_ctx.qp_y_prev_qg = sh.slice_qp_y;

    for col in 0..num_cols {
        let ctb_addr_rs = row * num_cols + col;

        // Diagonal dependency: wait for row-1 to complete col+1
        if row > start_row && col > 0 {
            let needed = (col + 1).min(num_cols - 1);
            sync[(row - 1) as usize].wait_col(needed);
        }

        let x_ctb = col << sps.ctb_log2_size_y;
        let y_ctb = row << sps.ctb_log2_size_y;

        // Record slice ownership
        g.set_slice_idx(ctb_addr_rs as usize, g.current_slice_idx as u8);

        decode_coding_tree_unit(&mut row_ctx, x_ctb, y_ctb);

        // §9.3.2.4: save contexts after 2nd CTU (col 1) for next row
        if col == 1 {
            let mut saved = [CabacContext::default(); NUM_CABAC_CONTEXTS];
            row_ctx.cabac.save_contexts(&mut saved);
            sync[row as usize].save_contexts(&saved);
        }

        // Signal completion and notify waiting rows
        sync[row as usize].complete_col(col);

        let end_of_slice = decode_end_of_slice_segment_flag(row_ctx.cabac) != 0;
        if end_of_slice || col == num_cols - 1 {
            break;
        }
    }

    // Final notify for rows that might wait on the last column
    sync[row as usize].complete_col(num_cols - 1);

    // This row's reader is private (rows share only the read-only buffer);
    // its final position is where this substream's decode stopped. The max
    // over all rows is the true end of the slice segment. Read via the
    // engine (still borrowed) so no extra borrow of `row_bs` is needed.
    row_ctx.cabac.bitstream().bits_read()
}

/// WPP parallel decode via rayon. Each CTU row is a unit of work submitted
/// to the pool (mirroring the C++ ThreadPool design): rows form a deep
/// pipeline — row r+1 tracks row r one column behind, row r+2 tracks row r+1,
/// and so on — so with enough workers several rows decode concurrently and
/// each new row starts as soon as its predecessor's 2nd CTU completes instead
/// of after the whole previous row. The output is byte-identical to the
/// serial path — only the scheduling differs.
///
/// Deadlock-freedom: a row only ever waits on the row directly above it, and
/// only after that row has made progress (col >= 1). For the lowest-indexed
/// blocked row, its producer is therefore either running or finished — never
/// queued behind blocked rows — so some worker is always free to make progress.
///
/// Returns the final bit position of the slice segment (max over all rows'
/// private readers — substreams are laid out sequentially, so only the last
/// row can reach the true end).
pub fn decode_wpp_parallel(g: &WppShared, substream_byte_pos: &[usize]) -> usize {
    let sps = g.sps;
    let sh = g.sh;

    let num_rows = sps.pic_height_in_ctbs_y;
    let start_row = sh.slice_segment_address / sps.pic_width_in_ctbs_y;
    // With WPP (no tiles) each CTU row of the slice segment is one substream,
    // so the slice spans exactly (num_entry_point_offsets + 1) rows.
    let end_row = start_row + substream_byte_pos.len() as i32;

    let sync: Vec<RowSync> = (0..num_rows).map(|_| RowSync::new()).collect();
    let final_pos = std::sync::atomic::AtomicUsize::new(0);

    rayon::scope(|s| {
        let sync = &sync;
        let final_pos = &final_pos;
        // Submit worker rows; run the first row on the current thread (same
        // shape as the C++ original: pool jobs + inline front row).
        for r in (start_row + 1)..end_row {
            s.spawn(move |_| {
                let p = decode_wpp_row(g, r, start_row, substream_byte_pos, sync);
                final_pos.fetch_max(p, std::sync::atomic::Ordering::Relaxed);
            });
        }
        let p = decode_wpp_row(g, start_row, start_row, substream_byte_pos, sync);
        final_pos.fetch_max(p, std::sync::atomic::Ordering::Relaxed);
    });

    final_pos.load(std::sync::atomic::Ordering::Relaxed)
}

// ============================================================
// slice_segment_data (§7.3.8.1)
// ============================================================

/// Decode a complete slice segment. `ctx.cabac`'s bitstream reader must be
/// positioned at the start of the slice data (after the slice header).
/// `epb_positions`: byte positions of removed emulation prevention bytes in
/// the NAL; `slice_header_coded_size`: size of the slice header in the coded
/// NAL (before EP removal).
/// Returns `(ok, final_bit_pos)`: the bit position where the segment's
/// decode stopped (true end of slice data for both serial and WPP paths).
pub fn decode_slice_segment_data(
    ctx: &mut DecodingContext,
    epb_positions: &[usize],
    slice_header_coded_size: usize,
) -> (bool, usize) {
    let sps = ctx.sps;
    let pps = ctx.pps;
    let sh = ctx.sh;

    // §7.3.8.1: compute RBSP byte positions of WPP/tile substreams.
    // entry_point_offsets are in bytes of the coded slice data (including EP
    // bytes). Convert to RBSP positions (EP bytes removed).
    let slice_data_rbsp_start = ctx.cabac.bitstream().byte_position();
    let mut substream_byte_pos: Vec<usize> = Vec::new();
    if sh.num_entry_point_offsets > 0 {
        substream_byte_pos.resize(sh.num_entry_point_offsets as usize + 1, 0);
        substream_byte_pos[0] = slice_data_rbsp_start;

        // Cumulative coded offsets from slice data start
        let mut coded_offset: usize = 0;
        for i in 0..sh.num_entry_point_offsets {
            coded_offset += sh.entry_point_offset_minus1[i as usize] as usize + 1;
            if !epb_positions.is_empty() {
                // Convert coded offset to RBSP offset accounting for EP bytes
                let rbsp_off =
                    coded_to_rbsp_offset(coded_offset, slice_header_coded_size, epb_positions);
                substream_byte_pos[(i + 1) as usize] = slice_data_rbsp_start + rbsp_off;
            } else {
                // No EP bytes tracked — use coded offset directly
                substream_byte_pos[(i + 1) as usize] = slice_data_rbsp_start + coded_offset;
            }
        }
    }

    // WPP parallel path: when entropy_coding_sync is enabled, tiles disabled,
    // and we have entry point offsets, decode rows in parallel via rayon.
    if ctx.wpp_enabled
        && pps.entropy_coding_sync_enabled_flag
        && !pps.tiles_enabled_flag
        && !substream_byte_pos.is_empty()
    {
        let start_row = sh.slice_segment_address / sps.pic_width_in_ctbs_y;
        let end_row = start_row + substream_byte_pos.len() as i32;
        if end_row - start_row > 1 {
            // Exclusive access to all buffers for the duration of the call
            // is guaranteed by the `&mut ctx` borrow held through `g`
            // (see the SAFETY invariant on WppShared).
            let g = wpp_shared_from_ctx(ctx);
            return (true, decode_wpp_parallel(&g, &substream_byte_pos));
        }
    }

    let mut wpp_substream_idx = 0usize;

    // Initialize CABAC
    if !sh.dependent_slice_segment_flag {
        ctx.cabac
            .init_contexts(sh.slice_type as i32, sh.slice_qp_y, sh.cabac_init_flag);
    }
    ctx.cabac.init_decoder();
    if hevc_trace() {
        eprintln!(
            "RUST init_decoder: range={} offset={} bits={}",
            ctx.cabac.dbg_range(),
            ctx.cabac.dbg_offset(),
            ctx.cabac.bitstream().bits_read()
        );
    }

    // Init QP
    ctx.qp_y_prev = sh.slice_qp_y;
    ctx.qp_y_prev_qg = sh.slice_qp_y;

    // CTU scan
    let mut ctb_addr_in_ts = pps.ctb_addr_rs_to_ts[sh.slice_segment_address as usize];
    let mut ctb_addr_in_rs = sh.slice_segment_address;

    let mut end_of_slice = false;
    while !end_of_slice {
        let x_ctb = (ctb_addr_in_rs % sps.pic_width_in_ctbs_y) << sps.ctb_log2_size_y;
        let y_ctb = (ctb_addr_in_rs / sps.pic_width_in_ctbs_y) << sps.ctb_log2_size_y;
        let ctu_col = ctb_addr_in_rs % sps.pic_width_in_ctbs_y;

        // Record slice ownership for this CTU
        if let Some(slice_idx) = &mut ctx.slice_idx {
            slice_idx[ctb_addr_in_rs as usize] = ctx.current_slice_idx as u8;
        }

        if hevc_trace() {
            eprintln!(
                "RUST CTU addr_rs={} pos=({}, {}) bits={}",
                ctb_addr_in_rs, x_ctb, y_ctb, ctx.cabac.bitstream().bits_read()
            );
        }

        decode_coding_tree_unit(ctx, x_ctb, y_ctb);

        // §9.2.2: WPP — save contexts after the 2nd CTU of each row (col 1)
        if pps.entropy_coding_sync_enabled_flag && ctu_col == 1 {
            ctx.cabac.save_contexts(&mut ctx.wpp_saved_contexts);
            ctx.wpp_contexts_available = true;
        }

        end_of_slice = decode_end_of_slice_segment_flag(ctx.cabac) != 0;

        if !end_of_slice {
            ctb_addr_in_ts += 1;
            if ctb_addr_in_ts >= sps.pic_size_in_ctbs_y {
                break;
            }
            ctb_addr_in_rs = pps.ctb_addr_ts_to_rs[ctb_addr_in_ts as usize];

            // Tile boundary: subset end + byte alignment + CABAC reinit
            if pps.tiles_enabled_flag
                && pps.tile_id[ctb_addr_in_ts as usize] != pps.tile_id[(ctb_addr_in_ts - 1) as usize]
            {
                ctx.cabac.decode_terminate(); // end_of_subset_one_bit
                ctx.cabac.bitstream().byte_alignment().ok();
                ctx.cabac.init_decoder();
                // §8.6.1: reset QpY_prev at first QG in a tile
                ctx.qp_y_prev = sh.slice_qp_y;
                ctx.qp_y_prev_qg = sh.slice_qp_y;
            }

            // WPP boundary (§7.3.8.1 / §9.2.2): at end of each CTU row, seek
            // to the next substream (via entry_point_offset), reinit CABAC,
            // and restore contexts from the 2nd CTU of the previous row.
            if pps.entropy_coding_sync_enabled_flag {
                let prev_rs = pps.ctb_addr_ts_to_rs[(ctb_addr_in_ts - 1) as usize];
                let prev_row = prev_rs / sps.pic_width_in_ctbs_y;
                let cur_row = ctb_addr_in_rs / sps.pic_width_in_ctbs_y;
                if cur_row != prev_row {
                    wpp_substream_idx += 1;
                    // Seek to the exact byte position of this substream
                    if wpp_substream_idx < substream_byte_pos.len() {
                        ctx.cabac
                            .bitstream()
                            .seek_to_byte(substream_byte_pos[wpp_substream_idx]);
                    }
                    ctx.cabac.init_decoder();
                    // §8.6.1: reset QpY_prev at first QG of each CTB row (WPP)
                    ctx.qp_y_prev = sh.slice_qp_y;
                    ctx.qp_y_prev_qg = sh.slice_qp_y;
                    // §9.2.2: restore contexts from 2nd CTU of previous row
                    if ctx.wpp_contexts_available {
                        ctx.cabac.load_contexts(&ctx.wpp_saved_contexts);
                    } else {
                        ctx.cabac.init_contexts(
                            sh.slice_type as i32,
                            sh.slice_qp_y,
                            sh.cabac_init_flag,
                        );
                    }
                }
            }
        }
    }

    (true, ctx.cabac.bitstream().bits_read())
}

/// Build a `WppShared` from the current context's references. Exclusive
/// access to all referenced buffers for the duration of the parallel call is
/// guaranteed by the `&mut ctx` borrow that `WppShared<'w>` keeps alive.
fn wpp_shared_from_ctx<'bs, 'a, 'w>(ctx: &'w mut DecodingContext<'bs, 'a>) -> WppShared<'w> {
    let rbsp = ctx.cabac.bitstream().data();
    WppShared {
        sps: ctx.sps,
        pps: ctx.pps,
        sh: ctx.sh,
        dpb: ctx.dpb,
        rbsp,
        sps_scaling_list_enabled: ctx.sps_scaling_list_enabled,
        sps_scaling_list: ctx.sps_scaling_list,
        pps_scaling_list_present: ctx.pps_scaling_list_present,
        pps_scaling_list: ctx.pps_scaling_list,
        pic: ctx.pic as *const Picture as *mut Picture,
        cu_info: (ctx.cu_info.as_mut_ptr(), ctx.cu_info.len()),
        cu_info_stride: ctx.cu_info_stride,
        intra_pred_mode_y: (ctx.intra_pred_mode_y.as_mut_ptr(), ctx.intra_pred_mode_y.len()),
        intra_pred_mode_c: (ctx.intra_pred_mode_c.as_mut_ptr(), ctx.intra_pred_mode_c.len()),
        intra_pred_mode_stride: ctx.intra_pred_mode_stride,
        motion_info: (ctx.motion_info.as_mut_ptr(), ctx.motion_info.len()),
        motion_info_stride: ctx.motion_info_stride,
        cbf_luma_grid: (ctx.cbf_luma_grid.as_mut_ptr(), ctx.cbf_luma_grid.len()),
        log2_tu_size_grid: (ctx.log2_tu_size_grid.as_mut_ptr(), ctx.log2_tu_size_grid.len()),
        edge_flags_v: (ctx.edge_flags_v.as_mut_ptr(), ctx.edge_flags_v.len()),
        edge_flags_h: (ctx.edge_flags_h.as_mut_ptr(), ctx.edge_flags_h.len()),
        filter_grid_stride: ctx.filter_grid_stride,
        sao_params: (ctx.sao_params.as_mut_ptr(), ctx.sao_params.len()),
        sao_params_stride: ctx.sao_params_stride,
        slice_idx: match &mut ctx.slice_idx {
            Some(s) => s.as_mut_ptr(),
            None => std::ptr::null_mut(),
        },
        slice_idx_len: ctx.slice_idx.as_ref().map_or(0, |s| s.len()),
        current_slice_idx: ctx.current_slice_idx,
    }
}

// ============================================================
// coding_tree_unit (§7.3.8.2)
// ============================================================

pub fn decode_coding_tree_unit(ctx: &mut DecodingContext, x_ctb: i32, y_ctb: i32) {
    // SAO parsing (store params for later filtering)
    if ctx.sh.slice_sao_luma_flag || ctx.sh.slice_sao_chroma_flag {
        let rx = x_ctb >> ctx.sps.ctb_log2_size_y;
        let ry = y_ctb >> ctx.sps.ctb_log2_size_y;
        decode_sao(ctx, rx, ry);
    }

    // QP group reset (CTU level)
    if ctx.pps.cu_qp_delta_enabled_flag {
        ctx.is_cu_qp_delta_coded = false;
        ctx.cu_qp_delta_val = 0;
        ctx.qp_y_prev_qg = ctx.qp_y_prev;
    }

    decode_coding_quadtree(ctx, x_ctb, y_ctb, ctx.sps.ctb_log2_size_y);
}

// ============================================================
// coding_quadtree (§7.3.8.4)
// ============================================================

fn decode_coding_quadtree(ctx: &mut DecodingContext, x0: i32, y0: i32, log2_cb_size: i32) {
    let sps = ctx.sps;
    let pps = ctx.pps;
    let cb_size = 1 << log2_cb_size;

    let mut split = false;

    let at_boundary = (x0 + cb_size > sps.pic_width_in_luma_samples)
        || (y0 + cb_size > sps.pic_height_in_luma_samples);

    if at_boundary {
        // Force split if we exceed picture boundaries and can still split
        if log2_cb_size > sps.min_cb_log2_size_y {
            split = true;
        }
    } else if log2_cb_size > sps.min_cb_log2_size_y {
        let ctx_inc = derive_split_cu_flag_ctx(ctx, x0, y0, log2_cb_size);
        split = decode_split_cu_flag(ctx.cabac, ctx_inc) != 0;
    }

    if hevc_trace() {
        eprintln!("RUST QT ({},{}) {}x{} split={} bits={}", x0, y0, cb_size, cb_size,
            split as i32, ctx.cabac.bitstream().bits_read());
    }

    // QP group boundary — §8.6.1
    if pps.cu_qp_delta_enabled_flag && log2_cb_size >= pps.log2_min_cu_qp_delta_size {
        ctx.is_cu_qp_delta_coded = false;
        ctx.cu_qp_delta_val = 0;
        ctx.qp_y_prev_qg = ctx.qp_y_prev;
    }

    if split {
        let x1 = x0 + (1 << (log2_cb_size - 1));
        let y1 = y0 + (1 << (log2_cb_size - 1));
        let pic_w = sps.pic_width_in_luma_samples;
        let pic_h = sps.pic_height_in_luma_samples;

        decode_coding_quadtree(ctx, x0, y0, log2_cb_size - 1);
        if x1 < pic_w {
            decode_coding_quadtree(ctx, x1, y0, log2_cb_size - 1);
        }
        if y1 < pic_h {
            decode_coding_quadtree(ctx, x0, y1, log2_cb_size - 1);
        }
        if x1 < pic_w && y1 < pic_h {
            decode_coding_quadtree(ctx, x1, y1, log2_cb_size - 1);
        }
    } else {
        decode_coding_unit(ctx, x0, y0, log2_cb_size);
    }
}

// ============================================================
// coding_unit (§7.3.8.5)
// ============================================================

fn decode_coding_unit(ctx: &mut DecodingContext, x0: i32, y0: i32, log2_cb_size: i32) {
    let sps = ctx.sps;
    let pps = ctx.pps;
    let sh = ctx.sh;
    let cb_size = 1 << log2_cb_size;

    let mut cu_transquant_bypass = false;
    if pps.transquant_bypass_enabled_flag {
        cu_transquant_bypass = decode_cu_transquant_bypass_flag(ctx.cabac) != 0;
    }

    let mut cu_skip = false;
    if sh.slice_type != SliceType::I {
        let ctx_inc = derive_cu_skip_flag_ctx(ctx, x0, y0);
        cu_skip = decode_cu_skip_flag(ctx.cabac, ctx_inc) != 0;
    }

    let mut pred_mode = PredMode::Intra;
    let mut part_mode = PartMode::Part2Nx2N;

    if cu_skip {
        // §7.3.8.5: Skip mode = merge without residual
        pred_mode = PredMode::Skip;

        if hevc_trace() {
            eprintln!("RUST CU ({},{}) {}x{} SKIP bits={}", x0, y0, cb_size, cb_size,
                ctx.cabac.bitstream().bits_read());
        }

        // Store pred_mode in grid BEFORE decode_prediction_unit_inter so
        // that it correctly selects the merge path (not AMVP)
        let n = cb_size / sps.min_cb_size_y;
        for j in 0..n {
            for i in 0..n {
                ctx.cu_at_mut(x0 + i * sps.min_cb_size_y, y0 + j * sps.min_cb_size_y)
                    .pred_mode = PredMode::Skip;
            }
        }

        // prediction_unit for skip (always 2Nx2N) + motion compensation
        decode_prediction_unit_inter(ctx, x0, y0, cb_size, x0, y0, cb_size, cb_size, 0);
        let mi = get_pu_motion(ctx, x0, y0);
        mc_write_block(ctx, x0, y0, cb_size, cb_size, &mi);
    } else {
        if sh.slice_type != SliceType::I {
            pred_mode = if decode_pred_mode_flag(ctx.cabac) != 0 {
                PredMode::Intra
            } else {
                PredMode::Inter
            };
        }

        if pred_mode != PredMode::Intra || log2_cb_size == sps.min_cb_log2_size_y {
            part_mode = PartMode::from_u32(decode_part_mode(
                ctx.cabac,
                pred_mode,
                log2_cb_size,
                sps.min_cb_log2_size_y,
                sps.amp_enabled_flag,
            ) as u32);
        }

        if hevc_trace() {
            eprintln!("RUST CU ({},{}) {}x{} pred={} part={} bits={}", x0, y0, cb_size, cb_size,
                pred_mode as u8 as i32, part_mode as u8 as i32, ctx.cabac.bitstream().bits_read());
        }

        // §6.4.2: store pred_mode and part_mode early so PU-level AMVP/merge
        // can see them for intra-CU neighbor availability (sameCb path)
        let n_early = cb_size / sps.min_cb_size_y;
        for j in 0..n_early {
            for i in 0..n_early {
                let cu_early = ctx.cu_at_mut(x0 + i * sps.min_cb_size_y, y0 + j * sps.min_cb_size_y);
                cu_early.pred_mode = pred_mode;
                cu_early.part_mode = part_mode;
            }
        }

        if pred_mode == PredMode::Intra {
            // Check PCM — §7.3.8.5
            let mut is_pcm = false;
            if part_mode == PartMode::Part2Nx2N
                && sps.pcm_enabled_flag
                && log2_cb_size >= sps.log2_min_ipcm_cb_size_y
                && log2_cb_size <= sps.log2_max_ipcm_cb_size_y
            {
                is_pcm = ctx.cabac.decode_terminate() != 0;
            }

            if is_pcm {
                decode_pcm_samples(ctx, x0, y0, log2_cb_size);
                // Store CU info
                let n = cb_size / sps.min_cb_size_y;
                let qp = ctx.qp_y_prev;
                for j in 0..n {
                    for i in 0..n {
                        let cu = ctx.cu_at_mut(x0 + i * sps.min_cb_size_y, y0 + j * sps.min_cb_size_y);
                        cu.pred_mode = PredMode::Intra;
                        cu.log2_cb_size = log2_cb_size;
                        cu.is_pcm = true;
                        cu.qp_y = qp;
                    }
                }
                return;
            }

            // Intra prediction
            decode_prediction_unit_intra(ctx, x0, y0, log2_cb_size, part_mode);
        }
        if pred_mode == PredMode::Inter {
            // §7.3.8.5: parse prediction_unit(s) and perform motion compensation
            let mc_pu = |ctx: &mut DecodingContext, x_pb: i32, y_pb: i32, n_pb_w: i32, n_pb_h: i32, part_idx: i32| {
                decode_prediction_unit_inter(ctx, x0, y0, cb_size, x_pb, y_pb, n_pb_w, n_pb_h, part_idx);
                let mi = get_pu_motion(ctx, x_pb, y_pb);
                mc_write_block(ctx, x_pb, y_pb, n_pb_w, n_pb_h, &mi);
            };
            match part_mode {
                PartMode::Part2Nx2N => mc_pu(ctx, x0, y0, cb_size, cb_size, 0),
                PartMode::Part2NxN => {
                    mc_pu(ctx, x0, y0, cb_size, cb_size / 2, 0);
                    mc_pu(ctx, x0, y0 + cb_size / 2, cb_size, cb_size / 2, 1);
                }
                PartMode::PartNx2N => {
                    mc_pu(ctx, x0, y0, cb_size / 2, cb_size, 0);
                    mc_pu(ctx, x0 + cb_size / 2, y0, cb_size / 2, cb_size, 1);
                }
                PartMode::Part2NxnU => {
                    mc_pu(ctx, x0, y0, cb_size, cb_size / 4, 0);
                    mc_pu(ctx, x0, y0 + cb_size / 4, cb_size, cb_size * 3 / 4, 1);
                }
                PartMode::Part2NxnD => {
                    mc_pu(ctx, x0, y0, cb_size, cb_size * 3 / 4, 0);
                    mc_pu(ctx, x0, y0 + cb_size * 3 / 4, cb_size, cb_size / 4, 1);
                }
                PartMode::PartNlx2N => {
                    mc_pu(ctx, x0, y0, cb_size / 4, cb_size, 0);
                    mc_pu(ctx, x0 + cb_size / 4, y0, cb_size * 3 / 4, cb_size, 1);
                }
                PartMode::PartNRx2N => {
                    mc_pu(ctx, x0, y0, cb_size * 3 / 4, cb_size, 0);
                    mc_pu(ctx, x0 + cb_size * 3 / 4, y0, cb_size / 4, cb_size, 1);
                }
                PartMode::PartNxN => {
                    // §7.3.8.5: NxN inter (only at min CB size)
                    let half = cb_size / 2;
                    mc_pu(ctx, x0, y0, half, half, 0);
                    mc_pu(ctx, x0 + half, y0, half, half, 1);
                    mc_pu(ctx, x0, y0 + half, half, half, 2);
                    mc_pu(ctx, x0 + half, y0 + half, half, half, 3);
                }
            }
        }
    }

    // Store CU info in grid
    let n = cb_size / sps.min_cb_size_y;
    for j in 0..n {
        for i in 0..n {
            let cu = ctx.cu_at_mut(x0 + i * sps.min_cb_size_y, y0 + j * sps.min_cb_size_y);
            cu.pred_mode = pred_mode;
            cu.part_mode = part_mode;
            cu.log2_cb_size = log2_cb_size;
            cu.cu_transquant_bypass = cu_transquant_bypass;
        }
    }

    // Initialize grids for this CU
    {
        let stride = ctx.filter_grid_stride;
        for dy in (0..cb_size).step_by(4) {
            for dx in (0..cb_size).step_by(4) {
                let gx = (x0 + dx) / 4;
                let gy = (y0 + dy) / 4;
                ctx.log2_tu_size_grid[(gy * stride + gx) as usize] = log2_cb_size as u8;
                ctx.cbf_luma_grid[(gy * stride + gx) as usize] = 0;
            }
        }

        // §8.7.2.2: CU left edge (vertical)
        for dy in (0..cb_size).step_by(4) {
            ctx.edge_flags_v[(((y0 + dy) / 4) * stride + x0 / 4) as usize] = 1;
        }
        // §8.7.2.2: CU top edge (horizontal)
        for dx in (0..cb_size).step_by(4) {
            ctx.edge_flags_h[((y0 / 4) * stride + (x0 + dx) / 4) as usize] = 1;
        }

        // §8.7.2.3: PU boundary edges within CU
        if part_mode == PartMode::PartNx2N || part_mode == PartMode::PartNxN {
            let px = x0 + cb_size / 2;
            for dy in (0..cb_size).step_by(4) {
                ctx.edge_flags_v[(((y0 + dy) / 4) * stride + px / 4) as usize] = 1;
            }
        }
        if part_mode == PartMode::PartNlx2N {
            let px = x0 + cb_size / 4;
            for dy in (0..cb_size).step_by(4) {
                ctx.edge_flags_v[(((y0 + dy) / 4) * stride + px / 4) as usize] = 1;
            }
        }
        if part_mode == PartMode::PartNRx2N {
            let px = x0 + 3 * cb_size / 4;
            for dy in (0..cb_size).step_by(4) {
                ctx.edge_flags_v[(((y0 + dy) / 4) * stride + px / 4) as usize] = 1;
            }
        }
        if part_mode == PartMode::Part2NxN || part_mode == PartMode::PartNxN {
            let py = y0 + cb_size / 2;
            for dx in (0..cb_size).step_by(4) {
                ctx.edge_flags_h[((py / 4) * stride + (x0 + dx) / 4) as usize] = 1;
            }
        }
        if part_mode == PartMode::Part2NxnU {
            let py = y0 + cb_size / 4;
            for dx in (0..cb_size).step_by(4) {
                ctx.edge_flags_h[((py / 4) * stride + (x0 + dx) / 4) as usize] = 1;
            }
        }
        if part_mode == PartMode::Part2NxnD {
            let py = y0 + 3 * cb_size / 4;
            for dx in (0..cb_size).step_by(4) {
                ctx.edge_flags_h[((py / 4) * stride + (x0 + dx) / 4) as usize] = 1;
            }
        }
    }

    // §8.6.1: store CU position for QP derivation in transform units
    ctx.cu_x0 = x0;
    ctx.cu_y0 = y0;

    // Transform tree (only for non-skip, non-PCM)
    if !cu_skip {
        let mut rqt_root_cbf = true;
        if pred_mode != PredMode::Intra {
            // §7.3.8.5: rqt_root_cbf parsed when NOT (PART_2Nx2N && merge_flag)
            let is_merge_2nx2n = part_mode == PartMode::Part2Nx2N && ctx.cu_at(x0, y0).merge_flag;
            if !is_merge_2nx2n {
                rqt_root_cbf = decode_rqt_root_cbf(ctx.cabac) != 0;
            }
        }

        if rqt_root_cbf {
            decode_transform_tree(ctx, x0, y0, x0, y0, log2_cb_size, 0, 0, true, true);
        }
    }

    // §8.6.1: derive QpY for this CU. Uses CuQpDeltaVal (0 if not coded).
    let qp_y = derive_qp_y(ctx, x0, y0);
    ctx.qp_y_prev = qp_y;
    if hevc_trace() {
        eprintln!("RUST QPY ({},{}) {}x{} qp={} prev={}", x0, y0, cb_size, cb_size, qp_y, ctx.qp_y_prev);
    }

    // Store QP in grid
    for j in 0..n {
        for i in 0..n {
            ctx.cu_at_mut(x0 + i * sps.min_cb_size_y, y0 + j * sps.min_cb_size_y).qp_y = qp_y;
        }
    }
}

/// Motion-compensate one PU and write prediction to the picture (luma +
/// chroma), mirroring the C++ `mc_pu` lambda in `decode_coding_unit`.
/// Write MC prediction into the picture plane. Values are already clipped to
/// `[0, 2^bd-1]` by weighted prediction, so i16 → u16 is a plain widening and
/// each row is a bulk copy (no per-sample clamp/index arithmetic).
fn write_mc_block(pic: &mut Picture, c: usize, x0: i32, y0: i32, w: i32, h: i32, pred: &[i16]) {
    let stride = pic.stride[c];
    let plane = &mut pic.planes[c];
    for y in 0..h {
        let row = ((y0 + y) * stride + x0) as usize;
        let off = (y * w) as usize;
        let dst = &mut plane[row..row + w as usize];
        let src = &pred[off..off + w as usize];
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = *s as u16;
        }
    }
}

fn mc_write_block(ctx: &mut DecodingContext, x_pb: i32, y_pb: i32, n_pb_w: i32, n_pb_h: i32, mi: &PuMotionInfo) {
    let sps = ctx.sps;
    if ctx.fir_tmp.len() < FIR_SCRATCH_MAX {
        ctx.fir_tmp.resize(FIR_SCRATCH_MAX, 0);
    }
    // Luma
    {
        let n = (n_pb_w * n_pb_h) as usize;
        if ctx.mc_l0.len() < n {
            ctx.mc_l0.resize(n, 0);
        }
        if ctx.mc_l1.len() < n {
            ctx.mc_l1.resize(n, 0);
        }
        if ctx.mc_out.len() < n {
            ctx.mc_out.resize(n, 0);
        }
        let l0 = &mut ctx.mc_l0[..n];
        let l1 = &mut ctx.mc_l1[..n];
        let out = &mut ctx.mc_out[..n];
        let ft = &mut ctx.fir_tmp[..FIR_SCRATCH_MAX];
        perform_inter_prediction(
            sps,
            ctx.pps,
            ctx.sh,
            ctx.dpb,
            x_pb,
            y_pb,
            n_pb_w,
            n_pb_h,
            0,
            mi.mv[0],
            mi.mv[1],
            mi.ref_idx[0] as i32,
            mi.ref_idx[1] as i32,
            mi.pred_flag[0],
            mi.pred_flag[1],
            l0,
            l1,
            out,
            ft,
        );
        write_mc_block(ctx.pic, 0, x_pb, y_pb, n_pb_w, n_pb_h, out);
    }
    // Chroma (4:2:0)
    if sps.chroma_array_type != 0 {
        let c_w = n_pb_w / sps.sub_width_c;
        let c_h = n_pb_h / sps.sub_height_c;
        let x_c = x_pb / sps.sub_width_c;
        let y_c = y_pb / sps.sub_height_c;
        let c_n = (c_w * c_h) as usize;
        if ctx.mc_l0.len() < c_n {
            ctx.mc_l0.resize(c_n, 0);
        }
        if ctx.mc_l1.len() < c_n {
            ctx.mc_l1.resize(c_n, 0);
        }
        if ctx.mc_out.len() < c_n {
            ctx.mc_out.resize(c_n, 0);
        }
        let ft = &mut ctx.fir_tmp[..FIR_SCRATCH_MAX];
        for c in 1..=2 {
            let l0 = &mut ctx.mc_l0[..c_n];
            let l1 = &mut ctx.mc_l1[..c_n];
            let out = &mut ctx.mc_out[..c_n];
            perform_inter_prediction(
                sps,
                ctx.pps,
                ctx.sh,
                ctx.dpb,
                x_pb,
                y_pb,
                n_pb_w,
                n_pb_h,
                c,
                mi.mv[0],
                mi.mv[1],
                mi.ref_idx[0] as i32,
                mi.ref_idx[1] as i32,
                mi.pred_flag[0],
                mi.pred_flag[1],
                l0,
                l1,
                out,
                ft,
            );
            write_mc_block(ctx.pic, c as usize, x_c, y_c, c_w, c_h, out);
        }
    }
}

// ============================================================
// prediction_unit intra (§7.3.8.8 — intra part)
// ============================================================

fn decode_prediction_unit_intra(ctx: &mut DecodingContext, x0: i32, y0: i32, log2_cb_size: i32, part_mode: PartMode) {
    let sps = ctx.sps;
    let cb_size = 1 << log2_cb_size;
    let pb_offset = if part_mode == PartMode::PartNxN { cb_size / 2 } else { cb_size };

    // First pass: decode prev_intra_luma_pred_flag for all PUs
    let mut prev_flag = [false; 4];
    let mut mpm_idx_val = [0i32; 4];
    let mut rem_mode = [0i32; 4];

    let mut pu = 0usize;
    for _j in (0..cb_size).step_by(pb_offset as usize) {
        for _i in (0..cb_size).step_by(pb_offset as usize) {
            prev_flag[pu] = decode_prev_intra_luma_pred_flag(ctx.cabac) != 0;
            pu += 1;
        }
    }

    // Second pass: decode mpm_idx or rem_intra_luma_pred_mode
    pu = 0;
    for _j in (0..cb_size).step_by(pb_offset as usize) {
        for _i in (0..cb_size).step_by(pb_offset as usize) {
            if prev_flag[pu] {
                mpm_idx_val[pu] = decode_mpm_idx(ctx.cabac);
            } else {
                rem_mode[pu] = decode_rem_intra_luma_pred_mode(ctx.cabac);
            }
            pu += 1;
        }
    }

    // Derive luma intra modes
    pu = 0;
    for j in (0..cb_size).step_by(pb_offset as usize) {
        for i in (0..cb_size).step_by(pb_offset as usize) {
            let px = x0 + i;
            let py = y0 + j;

            let mut cand_mode_list = [0i32; 3];
            derive_mpm(ctx, px, py, &mut cand_mode_list);

            let intra_mode = if prev_flag[pu] {
                cand_mode_list[mpm_idx_val[pu] as usize]
            } else {
                // Sort candidates
                cand_mode_list.sort_unstable();

                let mut intra_mode = rem_mode[pu];
                for cand in cand_mode_list.iter() {
                    if intra_mode >= *cand {
                        intra_mode += 1;
                    }
                }
                intra_mode
            };

            ctx.set_intra_mode(px, py, pb_offset, intra_mode);
            if hevc_trace() {
                eprintln!("RUST PU ({},{}) luma_mode={} (prev={} mpm_idx={} rem={})",
                    px, py, intra_mode, prev_flag[pu] as i32, mpm_idx_val[pu], rem_mode[pu]);
            }
            pu += 1;
        }
    }

    // Chroma mode
    if sps.chroma_array_type != 0 {
        let coded_chroma = decode_intra_chroma_pred_mode(ctx.cabac);
        // For 4:2:0/4:2:2, one chroma mode per CU
        let luma_mode_for_chroma = ctx.intra_mode_at(x0, y0);
        let chroma_mode = derive_chroma_intra_mode(coded_chroma, luma_mode_for_chroma);
        ctx.set_chroma_mode(x0, y0, cb_size, chroma_mode);
        if hevc_trace() {
            eprintln!("RUST CU ({}, {}) chroma_mode={} (coded={} luma={})",
                x0, y0, chroma_mode, coded_chroma, luma_mode_for_chroma);
        }
    }
}

// ============================================================
// transform_tree (§7.3.8.8)
// ============================================================

#[allow(clippy::too_many_arguments)]
fn decode_transform_tree(
    ctx: &mut DecodingContext,
    x0: i32,
    y0: i32,
    x_base: i32,
    y_base: i32,
    log2_trafo_size: i32,
    trafo_depth: i32,
    blk_idx: i32,
    cbf_cb_parent: bool,
    cbf_cr_parent: bool,
) {
    let sps = ctx.sps;

    // IntraSplitFlag — Table 7-10: only set when PartMode == NxN for intra
    let cu_pred_mode = ctx.cu_at(x0, y0).pred_mode;
    let cu_part_mode = ctx.cu_at(x0, y0).part_mode;
    let intra_split_flag = cu_pred_mode == PredMode::Intra && cu_part_mode == PartMode::PartNxN;
    let max_trafo_depth = if cu_pred_mode == PredMode::Intra {
        sps.max_transform_hierarchy_depth_intra + (if intra_split_flag { 1 } else { 0 })
    } else {
        sps.max_transform_hierarchy_depth_inter
    };

    // §7.4.9.4: interSplitFlag — force split for non-2Nx2N inter CUs when
    // max_transform_hierarchy_depth_inter == 0
    let inter_split_flag = sps.max_transform_hierarchy_depth_inter == 0
        && cu_pred_mode == PredMode::Inter
        && cu_part_mode != PartMode::Part2Nx2N
        && trafo_depth == 0;

    // Determine split_transform_flag
    let split = if log2_trafo_size <= sps.max_tb_log2_size_y
        && log2_trafo_size > sps.min_tb_log2_size_y
        && trafo_depth < max_trafo_depth
        && !(intra_split_flag && trafo_depth == 0)
        && !inter_split_flag
    {
        decode_split_transform_flag(ctx.cabac, log2_trafo_size) != 0
    } else {
        // Implicit split
        log2_trafo_size > sps.max_tb_log2_size_y
            || (intra_split_flag && trafo_depth == 0)
            || inter_split_flag
    };

    // §7.3.8.8: Chroma CBF parsed when log2TrafoSize > 2 (4:2:0) or
    // ChromaArrayType == 3. When not parsed, inherit parent values for
    // deferred chroma (§7.3.8.10 cbfDepthC)
    let (mut cbf_cb, mut cbf_cr) = (cbf_cb_parent, cbf_cr_parent);
    if (log2_trafo_size > 2 && sps.chroma_array_type != 0) || sps.chroma_array_type == 3 {
        if trafo_depth == 0 || cbf_cb_parent {
            cbf_cb = decode_cbf_chroma(ctx.cabac, trafo_depth) != 0;
        } else {
            cbf_cb = false;
        }
        if trafo_depth == 0 || cbf_cr_parent {
            cbf_cr = decode_cbf_chroma(ctx.cabac, trafo_depth) != 0;
        } else {
            cbf_cr = false;
        }
    }

    if split {
        let x1 = x0 + (1 << (log2_trafo_size - 1));
        let y1 = y0 + (1 << (log2_trafo_size - 1));

        if hevc_trace() {
            eprintln!("RUST TT ({},{}) log2={} depth={} split=1 cbf_cb={} cbf_cr={}",
                x0, y0, log2_trafo_size, trafo_depth, cbf_cb as i32, cbf_cr as i32);
        }

        decode_transform_tree(ctx, x0, y0, x0, y0, log2_trafo_size - 1, trafo_depth + 1, 0, cbf_cb, cbf_cr);
        decode_transform_tree(ctx, x1, y0, x0, y0, log2_trafo_size - 1, trafo_depth + 1, 1, cbf_cb, cbf_cr);
        decode_transform_tree(ctx, x0, y1, x0, y0, log2_trafo_size - 1, trafo_depth + 1, 2, cbf_cb, cbf_cr);
        decode_transform_tree(ctx, x1, y1, x0, y0, log2_trafo_size - 1, trafo_depth + 1, 3, cbf_cb, cbf_cr);
    } else {
        // Leaf: read cbf_luma and decode transform unit
        let cbf_luma = if cu_pred_mode == PredMode::Intra
            || trafo_depth != 0
            || cbf_cb
            || cbf_cr
        {
            decode_cbf_luma(ctx.cabac, trafo_depth) != 0
        } else {
            true
        };

        if hevc_trace() {
            eprintln!("RUST TU ({},{}) log2={} depth={} cbf_luma={} cbf_cb={} cbf_cr={}",
                x0, y0, log2_trafo_size, trafo_depth, cbf_luma as i32, cbf_cb as i32, cbf_cr as i32);
        }

        decode_transform_unit(
            ctx,
            x0,
            y0,
            x_base,
            y_base,
            log2_trafo_size,
            trafo_depth,
            blk_idx,
            cbf_luma,
            cbf_cb,
            cbf_cr,
        );

        // Store TU info for deblocking
        let tr_size = 1 << log2_trafo_size;
        let stride = ctx.filter_grid_stride;
        for dy in (0..tr_size).step_by(4) {
            for dx in (0..tr_size).step_by(4) {
                let gx = (x0 + dx) / 4;
                let gy = (y0 + dy) / 4;
                ctx.cbf_luma_grid[(gy * stride + gx) as usize] = if cbf_luma { 1 } else { 0 };
                ctx.log2_tu_size_grid[(gy * stride + gx) as usize] = log2_trafo_size as u8;
            }
        }
        // §8.7.2.2: TU left edge (vertical)
        for dy in (0..tr_size).step_by(4) {
            ctx.edge_flags_v[(((y0 + dy) / 4) * stride + x0 / 4) as usize] = 1;
        }
        // §8.7.2.2: TU top edge (horizontal)
        for dx in (0..tr_size).step_by(4) {
            ctx.edge_flags_h[((y0 / 4) * stride + (x0 + dx) / 4) as usize] = 1;
        }
    }
}

// ============================================================
// Per-thread TU scratch (eliminates per-TU heap allocations)
// ============================================================

/// Max-TU (64×64) sample count; HEVC caps log2TrafoSize at 6.
const MAX_TU_SAMPLES: usize = 64 * 64;

/// Fixed offset of the zero slot: beyond the largest possible writable
/// region (`4 * MAX_TU_SAMPLES`), so no visit layout can overlap it and its
/// allocation-time zeros persist for all visit sizes.
const ZERO_SLOT_OFFSET: usize = 4 * MAX_TU_SAMPLES;

thread_local! {
    /// Per-thread TU scratch. Layout for a visit with `n` samples:
    /// `[0, n)` coefficients, `[n, 2n)` scaled, `[2n, 3n)` residual,
    /// `[3n, 4n)` pred_samples, plus the zero slot at
    /// `[ZERO_SLOT_OFFSET, ZERO_SLOT_OFFSET + n)`. Rayon reuses worker
    /// threads, so capacity persists across frames and steady-state
    /// allocation is zero.
    static TU_SCRATCH: RefCell<Vec<i16>> = const { RefCell::new(Vec::new()) };
}

/// Five disjoint slices over the TU scratch. The `zero` slot is never
/// written to.
struct TuSlots<'a> {
    coefficients: &'a mut [i16],
    scaled: &'a mut [i16],
    residual: &'a mut [i16],
    pred_samples: &'a mut [i16],
    zero: &'a [i16],
}

/// Split the per-thread scratch into the five slots for a TU visit of `n`
/// samples. The caller must keep the scratch's `RefMut` alive for as long as
/// the returned slices are used (re-entrant access then panics on the
/// double borrow instead of reallocating under live slots).
fn tu_slots(scratch: &mut [i16], n: usize) -> TuSlots<'_> {
    // Layout (absolute offsets): [0, n) coefficients, [n, 2n) scaled,
    // [2n, 3n) residual, [3n, 4n) pred_samples, and the zero slot at the
    // fixed absolute offset ZERO_SLOT_OFFSET (beyond every writable region,
    // so its allocation-time zeros persist).
    let (writable, zero_region) = scratch.split_at_mut(ZERO_SLOT_OFFSET);
    let (coefficients, rest) = writable.split_at_mut(n);
    let (scaled, rest) = rest.split_at_mut(n);
    let (residual, rest) = rest.split_at_mut(n);
    let (pred_samples, _) = rest.split_at_mut(n);
    let zero = &zero_region[..n];
    TuSlots { coefficients, scaled, residual, pred_samples, zero }
}

// ============================================================
// transform_unit (§7.3.8.10)
// ============================================================

#[allow(clippy::too_many_arguments)]
fn decode_transform_unit(
    ctx: &mut DecodingContext,
    x0: i32,
    y0: i32,
    x_base: i32,
    y_base: i32,
    log2_trafo_size: i32,
    _trafo_depth: i32,
    blk_idx: i32,
    cbf_luma: bool,
    cbf_cb: bool,
    cbf_cr: bool,
) {
    let sps = ctx.sps;
    let pps = ctx.pps;
    let sh = ctx.sh;
    // Copy the CU fields out: the context is mutably borrowed many times
    // below (CABAC, grids) while these must stay readable.
    let cu_pred_mode = ctx.cu_at(x0, y0).pred_mode;
    let cu_bypass = ctx.cu_at(x0, y0).cu_transquant_bypass;

    let tr_size = 1i32 << log2_trafo_size;
    let n = (tr_size * tr_size) as usize;
    debug_assert!(n <= MAX_TU_SAMPLES, "TU larger than 64x64");
    // Per-thread scratch for this TU visit (5 slots of tr_size^2 samples).
    // `coefficients` is zeroed by decode_residual_coding itself; the other
    // writable slots are fully overwritten by their producers, and `zero`
    // is never written to. No per-visit zeroing: the zero slot lives beyond
    // every writable region, so it stays at its allocation-time zeros;
    // pred_samples is zeroed at its intra-prediction call sites (matching
    // C++'s per-visit `int16_t pred_samples[64*64] = {}`), and the inter
    // path overwrites it fully.
    // The RefMut must be held for the whole visit; thread_local::with does
    // not let borrows escape the closure, so the entire TU decode runs
    // inside it. Anything re-entering the scratch cell mid-visit then
    // panics (double borrow) instead of a reallocation dangling live slots.
    TU_SCRATCH.with(|c| {
        let mut scratch = c.borrow_mut();
        if scratch.len() < ZERO_SLOT_OFFSET + n {
            scratch.resize(ZERO_SLOT_OFFSET + n, 0);
        }
        let TuSlots { coefficients, scaled, residual, pred_samples, zero } = tu_slots(&mut scratch[..], n);

        // QP delta
        if (cbf_luma || cbf_cb || cbf_cr) && pps.cu_qp_delta_enabled_flag && !ctx.is_cu_qp_delta_coded {
            ctx.cu_qp_delta_val = decode_cu_qp_delta(ctx.cabac);
            ctx.is_cu_qp_delta_coded = true;
        }

        // §8.6.1: QP derivation uses CU position (xCb, yCb), not TU position
        let qp_y = derive_qp_y(ctx, ctx.cu_x0, ctx.cu_y0);

        // Luma residual
        if cbf_luma {
            let mut transform_skip = false;
            if pps.transform_skip_enabled_flag && !cu_bypass && log2_trafo_size <= 2 {
                transform_skip = decode_transform_skip_flag(ctx.cabac, 0) != 0;
            }

            decode_residual_coding(ctx, x0, y0, log2_trafo_size, 0, coefficients);

            if !cu_bypass {
                let qp_prime = qp_y + sps.qp_bd_offset_y;
                let dq = dequant_params(ctx, cu_pred_mode);
                perform_dequant(&dq, log2_trafo_size as u32, 0, qp_prime, coefficients, scaled);

                perform_transform_inverse(
                    log2_trafo_size as u32,
                    0,
                    cu_pred_mode == PredMode::Intra,
                    transform_skip,
                    sps.bit_depth_y as u32,
                    scaled,
                    residual,
                );
            } else {
                residual.copy_from_slice(coefficients);
            }

            // Prediction for luma
            if cu_pred_mode == PredMode::Intra {
                let intra_mode = ctx.intra_mode_at(x0, y0);
                // Intra predictors (planar/DC/angular) write every sample of
                // the block, so no zero-init is needed.
                perform_intra_prediction(
                    ctx.pic,
                    ctx.sps,
                    ctx.pps,
                    x0,
                    y0,
                    log2_trafo_size,
                    0,
                    intra_mode,
                    ctx.slice_idx.as_deref(),
                    pred_samples,
                );
            } else {
                // Inter: pred already in picture from PU-level MC, read it back
                for y in 0..tr_size {
                    for x in 0..tr_size {
                        pred_samples[(y * tr_size + x) as usize] = ctx.pic.sample(0, x0 + x, y0 + y) as i16;
                    }
                }
            }

            // Reconstruct
            reconstruct_block(ctx, x0, y0, log2_trafo_size, 0, pred_samples, residual);
        } else if cu_pred_mode == PredMode::Intra {
            // No residual but still need intra prediction
            let intra_mode = ctx.intra_mode_at(x0, y0);
            perform_intra_prediction(
                ctx.pic,
                ctx.sps,
                ctx.pps,
                x0,
                y0,
                log2_trafo_size,
                0,
                intra_mode,
                ctx.slice_idx.as_deref(),
                pred_samples,
            );

            // Reconstruct with zero residual
            reconstruct_block(ctx, x0, y0, log2_trafo_size, 0, pred_samples, zero);
        }

        // Chroma residual (4:2:0: chroma TU is log2TrafoSize-1, min 2)
        if sps.chroma_array_type != 0 {
            let log2_trafo_size_c = (log2_trafo_size - 1).max(2);
            let tr_size_c = 1i32 << log2_trafo_size_c;

            // For 4:2:0 with log2TrafoSize==2, chroma is deferred to blkIdx==3
            let process_chroma = log2_trafo_size > 2 || blk_idx == 3;

            // §7.3.8.10: chroma position uses xBase/yBase when log2TrafoSize==2
            let x_c = if sps.chroma_array_type != 3 && log2_trafo_size == 2 { x_base } else { x0 };
            let y_c = if sps.chroma_array_type != 3 && log2_trafo_size == 2 { y_base } else { y0 };

            if process_chroma {
                for c_idx in 1..=2 {
                    let cbf_c = if c_idx == 1 { cbf_cb } else { cbf_cr };
                    if cbf_c {
                        let n_c = (tr_size_c * tr_size_c) as usize;
                        let coefficients = &mut coefficients[..n_c];
                        let scaled = &mut scaled[..n_c];
                        let residual = &mut residual[..n_c];

                        let mut transform_skip = false;
                        if pps.transform_skip_enabled_flag
                            && !cu_bypass
                            && log2_trafo_size_c <= 2
                        {
                            transform_skip = decode_transform_skip_flag(ctx.cabac, c_idx) != 0;
                        }

                        decode_residual_coding(ctx, x_c, y_c, log2_trafo_size_c, c_idx, coefficients);

                        if !cu_bypass {
                            // Chroma QP derivation
                            let qp_offset = if c_idx == 1 {
                                pps.pps_cb_qp_offset + sh.slice_cb_qp_offset
                            } else {
                                pps.pps_cr_qp_offset + sh.slice_cr_qp_offset
                            };
                            let q_p_i = clip3(-sps.qp_bd_offset_c, 57, qp_y + qp_offset);
                            let q_p_c = if q_p_i < 0 {
                                q_p_i
                            } else if q_p_i < 58 {
                                crate::hevc::cabac_tables::QP_CHROMA_TABLE[q_p_i as usize] as i32
                            } else {
                                q_p_i - 6
                            };
                            let qp_prime_c = q_p_c + sps.qp_bd_offset_c;

                            let dq = dequant_params(ctx, cu_pred_mode);
                            perform_dequant(
                                &dq,
                                log2_trafo_size_c as u32,
                                c_idx as u32,
                                qp_prime_c,
                                coefficients,
                                scaled,
                            );
                            perform_transform_inverse(
                                log2_trafo_size_c as u32,
                                c_idx as u32,
                                cu_pred_mode == PredMode::Intra,
                                transform_skip,
                                sps.bit_depth_c as u32,
                                scaled,
                                residual,
                            );
                        } else {
                            residual.copy_from_slice(coefficients);
                        }

                        // Chroma prediction
                        let pred_samples = &mut pred_samples[..n_c];
                        if cu_pred_mode == PredMode::Intra {
                            let chroma_mode = ctx.chroma_mode_at(x_c, y_c);
                            perform_intra_prediction(
                                ctx.pic,
                                ctx.sps,
                                ctx.pps,
                                x_c,
                                y_c,
                                log2_trafo_size_c,
                                c_idx,
                                chroma_mode,
                                ctx.slice_idx.as_deref(),
                                pred_samples,
                            );
                        } else {
                            // Inter: pred already written by PU-level MC
                            let x_cc = x_c / sps.sub_width_c;
                            let y_cc = y_c / sps.sub_height_c;
                            for y in 0..tr_size_c {
                                for x in 0..tr_size_c {
                                    pred_samples[(y * tr_size_c + x) as usize] =
                                        ctx.pic.sample(c_idx as usize, x_cc + x, y_cc + y) as i16;
                                }
                            }
                        }

                        reconstruct_block(ctx, x_c, y_c, log2_trafo_size_c, c_idx, pred_samples, residual);
                    } else if cu_pred_mode == PredMode::Intra {
                        let chroma_mode = ctx.chroma_mode_at(x_c, y_c);
                        let n_c = (tr_size_c * tr_size_c) as usize;
                        let pred_samples = &mut pred_samples[..n_c];
                        perform_intra_prediction(
                            ctx.pic,
                            ctx.sps,
                            ctx.pps,
                            x_c,
                            y_c,
                            log2_trafo_size_c,
                            c_idx,
                            chroma_mode,
                            ctx.slice_idx.as_deref(),
                            pred_samples,
                        );
                        let zero = &zero[..n_c];
                        reconstruct_block(ctx, x_c, y_c, log2_trafo_size_c, c_idx, pred_samples, zero);
                    }
                }
            }
        }
    });
}

/// Build dequant parameters from the context's scaling list state.
fn dequant_params<'a>(ctx: &DecodingContext<'_, 'a>, cu_pred_mode: PredMode) -> DequantParams<'a> {
    DequantParams {
        bit_depth_luma: ctx.sps.bit_depth_y as u32,
        bit_depth_chroma: ctx.sps.bit_depth_c as u32,
        scaling_list_enabled: ctx.sps_scaling_list_enabled,
        sps_scaling_list: ctx.sps_scaling_list,
        pps_scaling_list_present: ctx.pps_scaling_list_present,
        pps_scaling_list: ctx.pps_scaling_list,
        cu_pred_mode,
    }
}

// ============================================================
// PCM mode (§7.3.10.2)
// ============================================================

fn decode_pcm_samples(ctx: &mut DecodingContext, x0: i32, y0: i32, log2_cb_size: i32) {
    let sps = ctx.sps;
    let bs = ctx.cabac.bitstream();

    // Byte alignment before PCM data
    bs.byte_alignment().ok();

    let cb_size = 1 << log2_cb_size;
    let num_luma_samples = cb_size * cb_size;
    let luma_bits = sps.pcm_sample_bit_depth_luma_minus1 + 1;

    // Read luma samples
    for i in 0..num_luma_samples {
        let y = i / cb_size;
        let x = i % cb_size;
        let val = bs.read_bits(luma_bits as usize).unwrap_or(0) as u16;
        *ctx.pic.sample_mut(0, x0 + x, y0 + y) = val;
    }

    // Read chroma samples
    if sps.chroma_array_type != 0 {
        let chroma_bits = sps.pcm_sample_bit_depth_chroma_minus1 + 1;
        let chroma_w = cb_size / sps.sub_width_c;
        let chroma_h = cb_size / sps.sub_height_c;
        let num_chroma_samples = chroma_w * chroma_h;

        for c_idx in 1..=2 {
            let x_c = x0 / sps.sub_width_c;
            let y_c = y0 / sps.sub_height_c;
            for i in 0..num_chroma_samples {
                let y = i / chroma_w;
                let x = i % chroma_w;
                let val = bs.read_bits(chroma_bits as usize).unwrap_or(0) as u16;
                *ctx.pic.sample_mut(c_idx as usize, x_c + x, y_c + y) = val;
            }
        }
    }

    // Reset CABAC contexts after PCM (§9.3.1.2)
    ctx.cabac.init_contexts(ctx.sh.slice_type as i32, ctx.sh.slice_qp_y, ctx.sh.cabac_init_flag);
    ctx.cabac.init_decoder();
}

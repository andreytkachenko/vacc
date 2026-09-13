//! Port of `hevc/filters/sao.cpp` — Sample Adaptive Offset, spec §8.7.3.
//!
//! Mirrors `hevc::apply_sao` control flow and arithmetic exactly; verified
//! byte-for-byte against the C++ hevc.js oracle (since removed); outputs are
//! now pinned by the SHA-256 goldens in `hevc::goldens`.

#[cfg(test)]
pub(crate) use self::tests::golden_entries;

use crate::hevc::types::{clip3, Plane, Tiles};

/// SAO parameters for one CTU (spec §7.4.9). Canonical definition lives in
/// [`crate::hevc::coding_tree`] (the coding-tree decode path writes these);
/// re-exported here so the SAO filter consumes the same type.
pub use crate::hevc::coding_tree::SaoParams;

/// §8.7.3.2: EO class direction offsets.
/// Class 0 (H): (-1,0)/(1,0); 1 (V): (0,-1)/(0,1); 2 (D135); 3 (D45).
const EO_DX: [[i32; 2]; 4] = [[-1, 1], [0, 0], [-1, 1], [1, -1]];
const EO_DY: [[i32; 2]; 4] = [[0, 0], [-1, 1], [-1, 1], [-1, 1]];

/// Per-min-CB PCM/transquant-bypass flags (C++ `cu_info`, min-CB granularity).
pub struct CuGrid<'a> {
    pub is_pcm: &'a [u8],
    pub bypass: &'a [u8],
    /// PicWidthInMinCbsY (C++ `cu_info_stride`).
    pub stride: i32,
    /// MinCbSizeY — the divisor applied to luma coords when indexing.
    pub min_cb: i32,
}

/// Context for `apply_sao` — mirrors the DecodingContext fields the C++
/// kernel reads.
pub struct SaoCtx<'a> {
    // SPS-derived
    /// sample_adaptive_offset_enabled_flag.
    pub sao_enabled: bool,
    /// 1 << CtbLog2SizeY.
    pub ctb_size: i32,
    pub sub_w: i32,
    pub sub_h: i32,
    /// 1 (monochrome) or 3.
    pub num_comp: i32,
    pub bit_depth_y: i32,
    pub bit_depth_c: i32,
    /// pcm_loop_filter_disabled_flag.
    pub pcm_filter_disabled: bool,
    pub pic_w: i32,
    pub pic_h: i32,
    /// PicWidthInCtbsY / PicHeightInCtbsY.
    pub ctbs_w: i32,
    pub ctbs_h: i32,
    // PPS-derived
    /// transquant_bypass_enabled_flag.
    pub transquant_bypass_enabled: bool,
    /// loop_filter_across_tiles_enabled_flag.
    pub loop_filter_across_tiles: bool,
    /// None = no tiles (empty pps.TileId).
    pub tiles: Option<&'a Tiles<'a>>,
    // Slice state
    /// Per-CTB slice index; None = single slice (fast path only).
    pub slice_idx: Option<&'a [u8]>,
    /// Per-slice slice_loop_filter_across_slices_enabled_flag.
    pub slice_across_slices: &'a [bool],
    // SAO state
    /// Per-CTU parameters, raster order.
    pub sao_params: &'a [SaoParams],
    /// = ctbs_w.
    pub sao_stride: i32,
    /// PCM/bypass CU grid at min-CB granularity.
    pub cu: &'a CuGrid<'a>,
}

/// True when the 3x3 CTB neighbourhood around (rx, ry) is uniform — same
/// slice and same tile everywhere. SAO neighbours never reach past that
/// neighbourhood, so no per-sample cross-boundary test can fire inside this
/// CTB.
fn ctb_neighbourhood_uniform(ctx: &SaoCtx, rx: i32, ry: i32) -> bool {
    let Some(slice_idx) = ctx.slice_idx else {
        return true;
    };

    let w = ctx.ctbs_w;
    let h = ctx.ctbs_h;
    let cur_addr = (ry * w + rx) as usize;
    let si_cur = slice_idx[cur_addr];

    let check_tiles = !ctx.loop_filter_across_tiles && ctx.tiles.is_some();
    let tile_cur = if check_tiles {
        let tiles = ctx.tiles.unwrap();
        tiles.tile_id[tiles.ctb_addr_rs_to_ts[cur_addr] as usize]
    } else {
        0
    };

    for dy in -1..=1 {
        let ny = ry + dy;
        if ny < 0 || ny >= h {
            continue;
        }
        for dx in -1..=1 {
            let nx = rx + dx;
            if nx < 0 || nx >= w {
                continue;
            }
            let nbr_addr = (ny * w + nx) as usize;
            if slice_idx[nbr_addr] != si_cur {
                return false;
            }
            if check_tiles {
                let tiles = ctx.tiles.unwrap();
                if tiles.tile_id[tiles.ctb_addr_rs_to_ts[nbr_addr] as usize] != tile_cur {
                    return false;
                }
            }
        }
    }
    true
}

/// Apply SAO to all components of the picture (spec §8.7.3).
/// Mirrors `hevc::apply_sao` — same control flow and arithmetic.
pub fn apply_sao(ctx: &SaoCtx, planes: &mut [Plane]) {
    if !ctx.sao_enabled {
        return;
    }

    let ctb_size = ctx.ctb_size;
    let sub_w = ctx.sub_w;
    let sub_h = ctx.sub_h;
    let num_comp = ctx.num_comp as usize;

    // Quick check: skip entirely if no CTU has SAO enabled
    let mut any_sao = false;
    for i in 0..(ctx.ctbs_w * ctx.ctbs_h) {
        if any_sao {
            break;
        }
        for c in 0..num_comp {
            if ctx.sao_params[i as usize].sao_type_idx[c] != 0 {
                any_sao = true;
                break;
            }
        }
    }
    if !any_sao {
        return;
    }

    // §8.7.3.1: SAO operates on a copy of the deblocked picture
    let orig: Vec<Vec<u16>> = planes[..num_comp].iter().map(|p| p.data.to_vec()).collect();

    // Process each CTU
    for ry in 0..ctx.ctbs_h {
        for rx in 0..ctx.ctbs_w {
            let sao = &ctx.sao_params[(ry * ctx.sao_stride + rx) as usize];

            // Once per CTB, not once per sample: decides whether the edge-offset
            // loop can skip the cross-slice/tile tests entirely.
            let ctb_uniform = ctb_neighbourhood_uniform(ctx, rx, ry);

            // Does this CTU hold any PCM or transquant_bypass CU? Scanned on luma
            // coordinates, so it is the same answer for all three components.
            let mut ctb_has_pcm_or_bypass = false;
            if ctx.pcm_filter_disabled || ctx.transquant_bypass_enabled {
                // Only scan when PCM or transquant_bypass are possible in this stream
                let x_yctb = rx * ctb_size;
                let y_yctb = ry * ctb_size;
                let min_cb = ctx.cu.min_cb; // sps.MinCbSizeY
                let cb_end_x = (x_yctb + ctb_size).min(ctx.pic_w);
                let cb_end_y = (y_yctb + ctb_size).min(ctx.pic_h);
                let mut cy = y_yctb;
                'scan: while cy < cb_end_y {
                    let mut cx = x_yctb;
                    while cx < cb_end_x {
                        let idx = ((cy / min_cb) * ctx.cu.stride + cx / min_cb) as usize;
                        if (ctx.pcm_filter_disabled && ctx.cu.is_pcm[idx] != 0)
                            || ctx.cu.bypass[idx] != 0
                        {
                            ctb_has_pcm_or_bypass = true;
                            break 'scan;
                        }
                        cx += min_cb;
                    }
                    cy += min_cb;
                }
            }

            for c_idx in 0..num_comp as i32 {
                if sao.sao_type_idx[c_idx as usize] == 0 {
                    continue;
                }

                let bit_depth = if c_idx == 0 { ctx.bit_depth_y } else { ctx.bit_depth_c };
                let max_val = (1 << bit_depth) - 1;
                let pcm_filter_disabled = ctx.pcm_filter_disabled;

                // CTB dimensions in this component
                let (n_ctb_sw, n_ctb_sh) = if c_idx == 0 {
                    (ctb_size, ctb_size)
                } else {
                    (ctb_size / sub_w, ctb_size / sub_h)
                };

                let x_ctb = rx * n_ctb_sw;
                let y_ctb = ry * n_ctb_sh;
                let comp_w = planes[c_idx as usize].width;
                let comp_h = planes[c_idx as usize].height;
                let stride = planes[c_idx as usize].stride;

                // Cross-boundary tests can only fire on a non-uniform neighbourhood
                let need_boundary_check = !ctb_uniform;

                // Edge offset reads two neighbours, so it needs a uniform
                // neighbourhood on top of having no per-sample CU lookups to do.
                let fast_path_edge = ctb_uniform && !ctb_has_pcm_or_bypass;
                // Band offset reads no neighbour at all (§8.7.3.3): no cross-slice
                // or cross-tile test can ever apply to it.
                let fast_path_band = !ctb_has_pcm_or_bypass;

                let src = &orig[c_idx as usize];
                let dst = &mut planes[c_idx as usize].data;

                if sao.sao_type_idx[c_idx as usize] == 2 {
                    // Edge offset — §8.7.3.2
                    let eo_class = sao.sao_eo_class[c_idx as usize] as usize;
                    let dx0 = EO_DX[eo_class][0];
                    let dy0 = EO_DY[eo_class][0];
                    let dx1 = EO_DX[eo_class][1];
                    let dy1 = EO_DY[eo_class][1];

                    if fast_path_edge {
                        // Hoist the picture-boundary test out of the loop.
                        let min_dx = dx0.min(dx1);
                        let max_dx = dx0.max(dx1);
                        let min_dy = dy0.min(dy1);
                        let max_dy = dy0.max(dy1);
                        let x_lo = x_ctb.max(-min_dx);
                        let x_hi = (x_ctb + n_ctb_sw).min(comp_w - max_dx);
                        let y_lo = y_ctb.max(-min_dy);
                        let y_hi = (y_ctb + n_ctb_sh).min(comp_h - max_dy);

                        // The C++ loop simply does not run when a range is empty.
                        if x_lo < x_hi && y_lo < y_hi {
                            let offs = &sao.sao_offset_val[c_idx as usize];
                            let n_off1 = dy0 * stride + dx0;
                            let n_off2 = dy1 * stride + dx1;

                            for y in y_lo..y_hi {
                                // All address math stays in i32: neighbor offsets can be
                                // negative (vertical EO), and the final address is only
                                // non-negative once the row base is added.
                                let base = y * stride;
                                for x in x_lo..x_hi {
                                    let addr = base + x;
                                    let c_val = src[addr as usize] as i32;
                                    let a = src[(addr + n_off1) as usize] as i32;
                                    let b = src[(addr + n_off2) as usize] as i32;
                                    // edgeIdx = 2 + sign(c-a) + sign(c-b); offs[2] is always 0
                                    // (§7.4.9.3) so the flat category writes back c_val unchanged.
                                    let edge_idx = 2
                                        + ((c_val > a) as i32 - (c_val < a) as i32)
                                        + ((c_val > b) as i32 - (c_val < b) as i32);
                                    dst[addr as usize] =
                                        clip3(0, max_val, c_val + offs[edge_idx as usize]) as u16;
                                }
                            }
                        }
                        continue;
                    }

                    for j in 0..n_ctb_sh {
                        let y_sj = y_ctb + j;
                        if y_sj >= comp_h {
                            break;
                        }
                        for i in 0..n_ctb_sw {
                            let x_si = x_ctb + i;
                            if x_si >= comp_w {
                                break;
                            }

                            // §8.7.3.2: skip PCM and transquant_bypass
                            let x_y = if c_idx == 0 { x_si } else { x_si * sub_w };
                            let y_y = if c_idx == 0 { y_sj } else { y_sj * sub_h };
                            if ctb_has_pcm_or_bypass {
                                let idx = ((y_y / ctx.cu.min_cb) * ctx.cu.stride + x_y / ctx.cu.min_cb) as usize;
                                if (pcm_filter_disabled && ctx.cu.is_pcm[idx] != 0)
                                    || ctx.cu.bypass[idx] != 0
                                {
                                    continue;
                                }
                            }

                            // Neighbor positions
                            let x_n1 = x_si + dx0;
                            let y_n1 = y_sj + dy0;
                            let x_n2 = x_si + dx1;
                            let y_n2 = y_sj + dy1;

                            // §8.7.3.2: out-of-picture neighbors → no modification
                            if x_n1 < 0 || x_n1 >= comp_w || y_n1 < 0 || y_n1 >= comp_h {
                                continue;
                            }
                            if x_n2 < 0 || x_n2 >= comp_w || y_n2 < 0 || y_n2 >= comp_h {
                                continue;
                            }

                            // §8.7.3.2: cross-slice / cross-tile boundary checks
                            let mut skip_edge = false;
                            if need_boundary_check {
                                let slice_idx = ctx.slice_idx.unwrap();
                                let cur_addr = (y_y / ctb_size) * ctx.ctbs_w + (x_y / ctb_size);
                                let si_cur = slice_idx[cur_addr as usize];
                                for nk in 0..2u32 {
                                    if skip_edge {
                                        break;
                                    }
                                    let (x_nk, y_nk) = if nk == 0 {
                                        (x_n1, y_n1)
                                    } else {
                                        (x_n2, y_n2)
                                    };
                                    let x_yn = if c_idx == 0 { x_nk } else { x_nk * sub_w };
                                    let y_yn = if c_idx == 0 { y_nk } else { y_nk * sub_h };
                                    let nbr_addr =
                                        (y_yn / ctb_size) * ctx.ctbs_w + (x_yn / ctb_size);
                                    let si_nbr = slice_idx[nbr_addr as usize];
                                    if si_cur != si_nbr {
                                        // §8.7.3.2: check the flag of the slice whose entry
                                        // boundary is being crossed
                                        if si_nbr < si_cur {
                                            if !ctx.slice_across_slices[si_cur as usize] {
                                                skip_edge = true;
                                            }
                                        } else {
                                            if !ctx.slice_across_slices[si_nbr as usize] {
                                                skip_edge = true;
                                            }
                                        }
                                    }
                                }
                                // §8.7.3.2: cross-tile boundary check
                                if let Some(tiles) = ctx.tiles
                                    && !skip_edge
                                    && !ctx.loop_filter_across_tiles
                                {
                                    let ts_cur = tiles.ctb_addr_rs_to_ts[cur_addr as usize];
                                    for nk in 0..2u32 {
                                        if skip_edge {
                                            break;
                                        }
                                        let (x_nk, y_nk) = if nk == 0 {
                                            (x_n1, y_n1)
                                        } else {
                                            (x_n2, y_n2)
                                        };
                                        let x_yn = if c_idx == 0 { x_nk } else { x_nk * sub_w };
                                        let y_yn = if c_idx == 0 { y_nk } else { y_nk * sub_h };
                                        let nbr_addr =
                                            (y_yn / ctb_size) * ctx.ctbs_w + (x_yn / ctb_size);
                                        let ts_nbr = tiles.ctb_addr_rs_to_ts[nbr_addr as usize];
                                        if tiles.tile_id[ts_cur as usize]
                                            != tiles.tile_id[ts_nbr as usize]
                                        {
                                            skip_edge = true;
                                        }
                                    }
                                }
                            }
                            if skip_edge {
                                continue;
                            }

                            let c_val = src[(y_sj * stride + x_si) as usize] as i32;
                            let a = src[(y_n1 * stride + x_n1) as usize] as i32;
                            let b = src[(y_n2 * stride + x_n2) as usize] as i32;

                            // §8.7.3.2: edge index categorization
                            let edge_idx = 2
                                + ((c_val > a) as i32 - (c_val < a) as i32)
                                + ((c_val > b) as i32 - (c_val < b) as i32);

                            let offset = sao.sao_offset_val[c_idx as usize][edge_idx as usize];
                            if offset != 0 {
                                dst[(y_sj * stride + x_si) as usize] =
                                    clip3(0, max_val, c_val + offset) as u16;
                            }
                        }
                    }
                } else {
                    // Band offset — §8.7.3.3
                    let band_shift = bit_depth - 5;
                    let band_pos = sao.sao_band_position[c_idx as usize];

                    if fast_path_band {
                        // Fold the "is this sample in the offset window" test into a
                        // 32-entry table. No wrap-around: bands past 31 don't exist.
                        let mut band_offs = [0i32; 32];
                        for k in 0..4 {
                            let b = band_pos + k;
                            if b < 32 {
                                band_offs[b as usize] = sao.sao_offset_val[c_idx as usize][k as usize];
                            }
                        }

                        let x_hi = (x_ctb + n_ctb_sw).min(comp_w);
                        let y_hi = (y_ctb + n_ctb_sh).min(comp_h);

                        for y in y_ctb..y_hi {
                            let base = (y * stride) as usize;
                            for x in x_ctb..x_hi {
                                let sample = src[base + x as usize] as i32;
                                // & 31 is a no-op while sample <= maxVal; it keeps a
                                // corrupt plane from indexing past the table.
                                dst[base + x as usize] = clip3(
                                    0,
                                    max_val,
                                    sample + band_offs[((sample >> band_shift) & 31) as usize],
                                ) as u16;
                            }
                        }
                        continue;
                    }

                    for j in 0..n_ctb_sh {
                        let y_sj = y_ctb + j;
                        if y_sj >= comp_h {
                            break;
                        }
                        for i in 0..n_ctb_sw {
                            let x_si = x_ctb + i;
                            if x_si >= comp_w {
                                break;
                            }

                            if ctb_has_pcm_or_bypass {
                                let x_y = if c_idx == 0 { x_si } else { x_si * sub_w };
                                let y_y = if c_idx == 0 { y_sj } else { y_sj * sub_h };
                                let idx = ((y_y / ctx.cu.min_cb) * ctx.cu.stride + x_y / ctx.cu.min_cb) as usize;
                                if (pcm_filter_disabled && ctx.cu.is_pcm[idx] != 0)
                                    || ctx.cu.bypass[idx] != 0
                                {
                                    continue;
                                }
                            }

                            let sample = src[(y_sj * stride + x_si) as usize] as i32;
                            let band = sample >> band_shift;
                            let band_idx = band - band_pos;
                            if (0..4).contains(&band_idx) {
                                let offset = sao.sao_offset_val[c_idx as usize][band_idx as usize];
                                if offset != 0 {
                                    dst[(y_sj * stride + x_si) as usize] =
                                        clip3(0, max_val, sample + offset) as u16;
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
    use crate::hevc::goldens;

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

    /// One randomized SAO scenario through the Rust implementation; the
    /// filtered planes are appended to `out` (golden serialization).
    /// `multi_slice`/`use_tiles` exercise the slow path; `with_pcm` adds
    /// PCM/transquant-bypass CUs.
    fn run_sao_case(
        rng: &mut Rng,
        multi_slice: bool,
        use_tiles: bool,
        with_pcm: bool,
        out: &mut Vec<u8>,
    ) {
        let bit_depth = if rng.below(2) == 0 { 8 } else { 10 };
        let chroma_array_type = if rng.below(4) == 0 { 0 } else { 1 };
        let (sub_w, sub_h) = if chroma_array_type == 0 { (1, 1) } else { (2, 2) };
        let ctb_log2 = 5 + rng.i32(3); // 32..128 CTUs
        let ctb_size = 1 << ctb_log2;
        // Dims stay multiples of MinCbSize (4) like real HEVC streams, so the
        // min-CB grids on both sides are exactly sized. Up to ~3 CTBs per axis.
        let max_dim = 3 * ctb_size;
        let pic_w = 16 + 4 * rng.i32(((max_dim - 16) / 4) as u64);
        let pic_h = 16 + 4 * rng.i32(((max_dim - 16) / 4) as u64);
        // Derive the CTB grid from the actual dims (the C++ oracle does the
        // same with ceil(pic/ctb)).
        let ctbs_w = (pic_w + ctb_size - 1) / ctb_size;
        let ctbs_h = (pic_h + ctb_size - 1) / ctb_size;
        let comp_w = pic_w / sub_w;
        let comp_h = pic_h / sub_h;

        // Planes: random samples in [0, maxVal].
        let max_val = (1 << bit_depth) - 1;
        let plane_y: Vec<u16> =
            (0..pic_w * pic_h).map(|_| rng.below(max_val as u64 + 1) as u16).collect();
        let plane_cb: Vec<u16> =
            (0..comp_w * comp_h).map(|_| rng.below(max_val as u64 + 1) as u16).collect();
        let plane_cr: Vec<u16> =
            (0..comp_w * comp_h).map(|_| rng.below(max_val as u64 + 1) as u16).collect();

        // SAO params per CTU: flat 24-int layout shared with the C++ oracle.
        let num_ctbs = ctbs_w * ctbs_h;
        let off_mag = if bit_depth == 8 { 15 } else { 63 };
        let mut sao_params = vec![0i32; (num_ctbs * 24) as usize];
        for ctb in 0..num_ctbs {
            for c in 0..3 {
                if chroma_array_type == 0 && c > 0 {
                    continue;
                }
                let t = match rng.i32(10) {
                    0..=2 => 0,
                    3..=5 => 1,
                    _ => 2,
                };
                sao_params[(ctb * 24 + c) as usize] = t;
                if t == 2 {
                    sao_params[(ctb * 24 + 3 + c) as usize] = rng.i32(4); // eo_class
                    for k in 0..5 {
                        let mag = rng.i32(off_mag as u64 + 1);
                        sao_params[(ctb * 24 + 9 + c * 5 + k) as usize] =
                            if rng.below(2) == 0 { mag } else { -mag };
                    }
                } else if t == 1 {
                    sao_params[(ctb * 24 + 6 + c) as usize] = rng.i32(32); // band_position
                    for k in 0..4 {
                        let mag = rng.i32(off_mag as u64 + 1);
                        sao_params[(ctb * 24 + 9 + c * 5 + k) as usize] =
                            if rng.below(2) == 0 { mag } else { -mag };
                    }
                }
            }
        }

        // Per-min-CB PCM/bypass flags.
        let n_min_cbs = (pic_w / 4) * (pic_h / 4);
        let mut cu_pcm = vec![0u8; n_min_cbs as usize];
        let mut cu_bypass = vec![0u8; n_min_cbs as usize];
        if with_pcm {
            for i in 0..n_min_cbs {
                cu_pcm[i as usize] = (rng.below(10) == 0) as u8;
                cu_bypass[i as usize] = (rng.below(8) == 0) as u8;
            }
        }

        // Slices: a single straight cut when multi_slice.
        let n_slices = if multi_slice { 2 } else { 0 };
        let mut slice_idx = vec![0u8; num_ctbs as usize];
        let mut slice_across = vec![false; n_slices.max(1) as usize];
        for s in 0..n_slices {
            slice_across[s as usize] = rng.below(2) == 0;
        }
        if multi_slice {
            if rng.below(2) == 0 {
                let cut = 1 + rng.i32((ctbs_h - 1).max(1) as u64);
                for ry in 0..ctbs_h {
                    for rx in 0..ctbs_w {
                        slice_idx[(ry * ctbs_w + rx) as usize] = if ry >= cut { 1 } else { 0 };
                    }
                }
            } else {
                let cut = 1 + rng.i32((ctbs_w - 1).max(1) as u64);
                for ry in 0..ctbs_h {
                    for rx in 0..ctbs_w {
                        slice_idx[(ry * ctbs_w + rx) as usize] = if rx >= cut { 1 } else { 0 };
                    }
                }
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
        // --- Rust side ---
        let mut sao_params_rs = Vec::with_capacity(num_ctbs as usize);
        for ctb in 0..num_ctbs {
            let p = &sao_params[(ctb * 24) as usize..(ctb * 24 + 24) as usize];
            let mut sp = SaoParams::default();
            for c in 0..3usize {
                sp.sao_type_idx[c] = p[c];
                sp.sao_eo_class[c] = p[3 + c];
                sp.sao_band_position[c] = p[6 + c];
                for k in 0..5usize {
                    sp.sao_offset_val[c][k] = p[9 + c * 5 + k];
                }
            }
            sao_params_rs.push(sp);
        }

        let tiles_opt = if use_tiles {
            Some(Tiles { tile_id: &tile_id, ctb_addr_rs_to_ts: &rs2ts })
        } else {
            None
        };
        let cu = CuGrid { is_pcm: &cu_pcm, bypass: &cu_bypass, stride: pic_w / 4, min_cb: 4 };
        let ctx = SaoCtx {
            sao_enabled: true,
            ctb_size,
            sub_w,
            sub_h,
            num_comp: if chroma_array_type == 0 { 1 } else { 3 },
            bit_depth_y: bit_depth,
            bit_depth_c: bit_depth,
            pcm_filter_disabled: with_pcm && rng.below(2) == 0,
            pic_w,
            pic_h,
            ctbs_w,
            ctbs_h,
            transquant_bypass_enabled: with_pcm,
            loop_filter_across_tiles: !(use_tiles && rng.below(2) == 0),
            tiles: tiles_opt.as_ref(),
            slice_idx: if multi_slice { Some(&slice_idx) } else { None },
            slice_across_slices: &slice_across,
            sao_params: &sao_params_rs,
            sao_stride: ctbs_w,
            cu: &cu,
        };

        let mut out_y = plane_y.clone();
        let mut out_cb = plane_cb.clone();
        let mut out_cr = plane_cr.clone();
        apply_sao(
            &ctx,
            &mut [
                Plane { data: &mut out_y, width: pic_w, height: pic_h, stride: pic_w },
                Plane { data: &mut out_cb, width: comp_w, height: comp_h, stride: comp_w },
                Plane { data: &mut out_cr, width: comp_w, height: comp_h, stride: comp_w },
            ],
        );

        // --- Golden output ---
        for v in &out_y {
            goldens::push_u16(out, *v);
        }
        if chroma_array_type != 0 {
            for v in &out_cb {
                goldens::push_u16(out, *v);
            }
            for v in &out_cr {
                goldens::push_u16(out, *v);
            }
        }
    }

    fn compute_sao_fast_path() -> Vec<(String, Vec<u8>)> {
        let mut rng = Rng::new(0x5a01);
        let mut buf = Vec::new();
        for _ in 0..80 {
            run_sao_case(&mut rng, false, false, false, &mut buf);
        }
        vec![("sao::fast_path".to_string(), buf)]
    }

    #[test]
    fn sao_fast_path_matches_golden() {
        for (key, data) in compute_sao_fast_path() {
            goldens::assert_golden(&key, &data);
        }
    }

    fn compute_sao_multi_slice_tiles() -> Vec<(String, Vec<u8>)> {
        let mut rng = Rng::new(0x5a02);
        let mut buf = Vec::new();
        for _ in 0..80 {
            run_sao_case(&mut rng, true, true, false, &mut buf);
        }
        vec![("sao::multi_slice_tiles".to_string(), buf)]
    }

    #[test]
    fn sao_multi_slice_tiles_matches_golden() {
        for (key, data) in compute_sao_multi_slice_tiles() {
            goldens::assert_golden(&key, &data);
        }
    }

    fn compute_sao_pcm_bypass() -> Vec<(String, Vec<u8>)> {
        let mut rng = Rng::new(0x5a03);
        let mut buf = Vec::new();
        for _ in 0..60 {
            let ms = rng.below(2) == 0;
            let tl = rng.below(2) == 0;
            run_sao_case(&mut rng, ms, tl, true, &mut buf);
        }
        vec![("sao::pcm_bypass".to_string(), buf)]
    }

    #[test]
    fn sao_pcm_bypass_matches_golden() {
        for (key, data) in compute_sao_pcm_bypass() {
            goldens::assert_golden(&key, &data);
        }
    }

    #[test]
    fn sao_disabled_or_empty_params_is_noop() {
        // sao_enabled = false: planes untouched.
        let mut plane = vec![3u16; 64 * 64];
        let sao_params = [SaoParams::default(); 1];
        let cu = CuGrid { is_pcm: &[0; 256], bypass: &[0; 256], stride: 16, min_cb: 4 };
        let ctx = SaoCtx {
            sao_enabled: false,
            ctb_size: 64,
            sub_w: 2,
            sub_h: 2,
            num_comp: 3,
            bit_depth_y: 8,
            bit_depth_c: 8,
            pcm_filter_disabled: false,
            pic_w: 64,
            pic_h: 64,
            ctbs_w: 1,
            ctbs_h: 1,
            transquant_bypass_enabled: false,
            loop_filter_across_tiles: true,
            tiles: None,
            slice_idx: None,
            slice_across_slices: &[true],
            sao_params: &sao_params,
            sao_stride: 1,
            cu: &cu,
        };
        apply_sao(&ctx, &mut [Plane { data: &mut plane, width: 64, height: 64, stride: 64 }]);
        assert!(plane.iter().all(|&s| s == 3));
    }

    pub(crate) fn golden_entries() -> Vec<(String, String)> {
        let mut v = Vec::new();
        for (k, b) in compute_sao_fast_path() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        for (k, b) in compute_sao_multi_slice_tiles() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        for (k, b) in compute_sao_pcm_bypass() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        v
    }
}

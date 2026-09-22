//! Port of `hevc/decoding/intra_prediction.{h,cpp}` — intra prediction,
//! spec §8.4.4.2 (35 modes: Planar, DC, Angular 2-34).

#[cfg(test)]
pub(crate) use self::tests::golden_entries;

use crate::hevc::picture::Picture;
use crate::hevc::types::{clip3, Pps, Sps};

/// Intra prediction angle tables (Table 8-4, 8-5).
pub const INTRA_PRED_ANGLE: [i32; 35] = [
    0, 0, 32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17,
    -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
];

pub const INV_ANGLE: [i32; 35] = [
    0, 0, 256, 315, 390, 482, 630, 910, 1638, 4096, 0, 4096, 1638, 910, 630, 482, 390, 315, 256,
    315, 390, 482, 630, 910, 1638, 4096, 0, 4096, 1638, 910, 630, 482, 390, 315, 256,
];

/// Z-scan address from min-CB coordinates within a CTU: interleave bits,
/// x in even positions, y in odd positions.
#[inline]
fn zscan_addr(bx: i32, by: i32) -> u32 {
    let mut z = 0u32;
    for i in 0..8 {
        z |= (((bx >> i) & 1) as u32) << (2 * i);
        z |= (((by >> i) & 1) as u32) << (2 * i + 1);
    }
    z
}

/// Reference sample construction — spec §8.4.4.2.2.
/// Returns `(ref_top, ref_left)`, each of `2*n_tbs+1` samples:
/// - `ref_top[0]` = top-left corner, `ref_top[1..]` = top row left→right;
/// - `ref_left[0]` = top-left corner, `ref_left[1..]` = left column top→bottom.
///
/// Availability: a sample is available if it belongs to a previously decoded
/// CTU (raster order) or precedes the current TU in Z-scan order at min-TB
/// granularity within the same CTU. Cross-CTU samples additionally require
/// the same slice and tile as the current CTU (§6.4.1); `slice_idx` is the
/// per-CTU slice-index array (None = single slice), `pps` provides the tile
/// scan tables.
#[allow(clippy::too_many_arguments)] // faithful port of the C++ signature
pub fn build_reference_samples(
    pic: &Picture,
    sps: &Sps,
    pps: &Pps,
    x0: i32,
    y0: i32,
    n_tbs: i32,
    c_idx: i32,
    slice_idx: Option<&[u8]>,
) -> (Vec<i16>, Vec<i16>) {
    let (pic_w, pic_h) = if c_idx == 0 {
        (sps.pic_width_in_luma_samples, sps.pic_height_in_luma_samples)
    } else {
        (
            sps.pic_width_in_luma_samples / sps.sub_width_c,
            sps.pic_height_in_luma_samples / sps.sub_height_c,
        )
    };

    // Convert luma coords to component coords
    let x_c = if c_idx > 0 { x0 / sps.sub_width_c } else { x0 };
    let y_c = if c_idx > 0 { y0 / sps.sub_height_c } else { y0 };

    let bit_depth = if c_idx == 0 { sps.bit_depth_y } else { sps.bit_depth_c };
    let default_val = 1 << (bit_depth - 1);

    // Total reference samples needed: 4*nTbS + 1
    let total_samples = 4 * n_tbs + 1;
    let mut available = [false; 4 * 64 + 1];
    let mut samples = [0i16; 4 * 64 + 1];

    let ctb_size = if c_idx > 0 { sps.ctb_size_y / sps.sub_width_c } else { sps.ctb_size_y };
    let cur_ctb_x = x_c / ctb_size;
    let cur_ctb_y = y_c / ctb_size;

    // Use min-TB granularity (4x4) for Z-scan to handle NxN sub-PU correctly
    let min_blk_size = sps.min_tb_size_y;
    let ctb_origin_x = cur_ctb_x * sps.ctb_size_y;
    let ctb_origin_y = cur_ctb_y * sps.ctb_size_y;
    // Current TU's Z-scan address (in min-TB units within CTU)
    let cur_zscan = zscan_addr(
        (x0 - ctb_origin_x) / min_blk_size,
        (y0 - ctb_origin_y) / min_blk_size,
    );

    let is_reconstructed = |rx: i32, ry: i32| -> bool {
        if rx < 0 || ry < 0 || rx >= pic_w || ry >= pic_h {
            return false;
        }

        let ref_ctb_x = rx / ctb_size;
        let ref_ctb_y = ry / ctb_size;

        // Different CTU: available if the CTU precedes in raster order
        // AND is in the same slice/tile (§6.4.1).
        if ref_ctb_x != cur_ctb_x || ref_ctb_y != cur_ctb_y {
            let ref_ctb_addr = ref_ctb_y * sps.pic_width_in_ctbs_y + ref_ctb_x;
            let cur_ctb_addr = cur_ctb_y * sps.pic_width_in_ctbs_y + cur_ctb_x;
            if ref_ctb_addr >= cur_ctb_addr {
                return false;
            }
            // §6.4.1: SliceAddrRs must match
            if let Some(si) = slice_idx
                && si[ref_ctb_addr as usize] != si[cur_ctb_addr as usize]
            {
                return false;
            }
            // §6.4.1: TileId must match
            if !pps.tile_id.is_empty() {
                let ts_ref = pps.ctb_addr_rs_to_ts[ref_ctb_addr as usize];
                let ts_cur = pps.ctb_addr_rs_to_ts[cur_ctb_addr as usize];
                if pps.tile_id[ts_ref as usize] != pps.tile_id[ts_cur as usize] {
                    return false;
                }
            }
            return true;
        }

        // Same CTU: compare Z-scan addresses at min-TB granularity
        let luma_rx = if c_idx > 0 { rx * sps.sub_width_c } else { rx };
        let luma_ry = if c_idx > 0 { ry * sps.sub_height_c } else { ry };

        let ref_zscan = zscan_addr(
            (luma_rx - ctb_origin_x) / min_blk_size,
            (luma_ry - ctb_origin_y) / min_blk_size,
        );
        ref_zscan < cur_zscan
    };

    // Sample ordering in the reference array:
    // Index 0: bottom-left extension (p[-1][2*nTbS-1]) ... Index 2*nTbS-1: p[-1][0]
    // Index 2*nTbS: p[-1][-1] (top-left corner)
    // Index 2*nTbS+1: p[0][-1] ... Index 4*nTbS: p[2*nTbS-1][-1]

    // Bottom-left to top along left edge
    for k in 0..2 * n_tbs {
        let ref_y = y_c + 2 * n_tbs - 1 - k;
        let ref_x = x_c - 1;
        if is_reconstructed(ref_x, ref_y) {
            samples[k as usize] = pic.sample(c_idx as usize, ref_x, ref_y) as i16;
            available[k as usize] = true;
        }
    }

    // Top-left corner
    {
        let idx = 2 * n_tbs;
        if is_reconstructed(x_c - 1, y_c - 1) {
            samples[idx as usize] = pic.sample(c_idx as usize, x_c - 1, y_c - 1) as i16;
            available[idx as usize] = true;
        }
    }

    // Top edge, left to right, then top-right extension
    for k in 0..2 * n_tbs {
        let idx = 2 * n_tbs + 1 + k;
        if is_reconstructed(x_c + k, y_c - 1) {
            samples[idx as usize] = pic.sample(c_idx as usize, x_c + k, y_c - 1) as i16;
            available[idx as usize] = true;
        }
    }

    // Substitution: replace unavailable samples
    let first_avail = (0..total_samples).find(|&i| available[i as usize]);

    match first_avail {
        None => {
            // No neighbours available at all — fill with default
            for i in 0..total_samples {
                samples[i as usize] = default_val as i16;
            }
        }
        Some(first) => {
            // Propagate: fill unavailable before first with first available
            for i in 0..first {
                samples[i as usize] = samples[first as usize];
            }
            // Propagate forward
            for i in (first + 1)..total_samples {
                if !available[i as usize] {
                    samples[i as usize] = samples[(i - 1) as usize];
                }
            }
        }
    }

    // Convert to refTop/refLeft format
    let len = (2 * n_tbs + 1) as usize;
    let mut ref_top = vec![0i16; len];
    let mut ref_left = vec![0i16; len];

    ref_top[0] = samples[2 * n_tbs as usize];
    for i in 0..2 * n_tbs {
        ref_top[1 + i as usize] = samples[(2 * n_tbs + 1 + i) as usize];
    }

    ref_left[0] = samples[2 * n_tbs as usize];
    for i in 0..2 * n_tbs {
        ref_left[1 + i as usize] = samples[(2 * n_tbs - 1 - i) as usize];
    }

    (ref_top, ref_left)
}

/// §8.4.4.2.3: is reference filtering required for this mode/size?
pub fn needs_filtering(intra_mode: i32, log2_blk_size: i32) -> bool {
    // §8.4.4.2.3: filterFlag = 0 when DC mode or nTbS == 4
    if intra_mode == 1 {
        return false; // INTRA_DC
    }
    if log2_blk_size == 2 {
        return false; // nTbS == 4
    }

    // minDistVerHor = Min(|mode - 26|, |mode - 10|)
    let min_dist_ver_hor = (intra_mode - 26).abs().min((intra_mode - 10).abs());
    // Table 8-4: intraHorVerDistThres — nTbS = 8, 16, 32
    let thresholds = [7, 1, 0];
    if (3..=5).contains(&log2_blk_size) {
        min_dist_ver_hor > thresholds[log2_blk_size as usize - 3]
    } else {
        false
    }
}

/// §8.4.4.2.3: in-place [1,2,1]/4 (or bilinear) filtering of `2*n_tbs+1`
/// reference samples. The corner `ref[0]` is not filtered here; the caller
/// applies the cross-filtered corner (eq 8-41).
pub fn filter_reference_samples(ref_arr: &mut [i16], n_tbs: i32, bi_int_flag: bool) {
    let mut filtered = [0i16; 2 * 64 + 1];

    if bi_int_flag {
        // §8.4.4.2.3: bilinear interpolation between endpoints
        let top_left = ref_arr[0] as i32;
        let end_val = ref_arr[2 * n_tbs as usize] as i32;
        filtered[0] = ref_arr[0];
        for i in 1..2 * n_tbs {
            filtered[i as usize] =
                (((2 * n_tbs - i) * top_left + i * end_val + n_tbs) / (2 * n_tbs)) as i16;
        }
        filtered[2 * n_tbs as usize] = ref_arr[2 * n_tbs as usize];
    } else {
        // Standard [1,2,1]/4 filter (eq 8-41 to 8-45)
        filtered[0] = ref_arr[0]; // Corner not filtered
        for i in 1..2 * n_tbs {
            filtered[i as usize] = ((ref_arr[(i - 1) as usize] as i32
                + 2 * ref_arr[i as usize] as i32
                + ref_arr[(i + 1) as usize] as i32
                + 2)
                >> 2) as i16;
        }
        filtered[2 * n_tbs as usize] = ref_arr[2 * n_tbs as usize];
    }

    let len = (2 * n_tbs + 1) as usize;
    ref_arr[..len].copy_from_slice(&filtered[..len]);
}

/// Planar prediction (mode 0) — spec §8.4.4.2.4.
pub fn predict_planar(ref_top: &[i16], ref_left: &[i16], n_tbs: i32, pred: &mut [i16]) {
    let mut log2_n = 0;
    while (1 << log2_n) < n_tbs {
        log2_n += 1;
    }

    // refTop[nTbS+1] = p[nTbS][-1], refLeft[nTbS+1] = p[-1][nTbS]
    let top_right = ref_top[(n_tbs + 1) as usize] as i32;
    let bottom_left = ref_left[(n_tbs + 1) as usize] as i32;

    for y in 0..n_tbs {
        for x in 0..n_tbs {
            pred[(y * n_tbs + x) as usize] = (((n_tbs - 1 - x) * ref_left[(y + 1) as usize] as i32
                + (x + 1) * top_right
                + (n_tbs - 1 - y) * ref_top[(x + 1) as usize] as i32
                + (y + 1) * bottom_left
                + n_tbs)
                >> (log2_n + 1)) as i16;
        }
    }
}

/// DC prediction (mode 1) — spec §8.4.4.2.5.
pub fn predict_dc(
    ref_top: &[i16],
    ref_left: &[i16],
    n_tbs: i32,
    log2_blk_size: i32,
    c_idx: i32,
    pred: &mut [i16],
) {
    let mut sum = 0i32;
    for i in 1..=n_tbs {
        sum += ref_top[i as usize] as i32 + ref_left[i as usize] as i32;
    }
    let dc_val = (sum + n_tbs) >> (log2_blk_size + 1);

    for y in 0..n_tbs {
        for x in 0..n_tbs {
            pred[(y * n_tbs + x) as usize] = dc_val as i16;
        }
    }

    // §8.4.4.2.5 eq 8-48..8-51: DC boundary filter (cIdx == 0 only, nTbS < 32)
    if c_idx == 0 && n_tbs < 32 {
        pred[0] = ((ref_top[1] as i32 + ref_left[1] as i32 + 2 * dc_val + 2) >> 2) as i16;

        for x in 1..n_tbs {
            pred[x as usize] = ((ref_top[(x + 1) as usize] as i32 + 3 * dc_val + 2) >> 2) as i16;
        }

        for y in 1..n_tbs {
            pred[(y * n_tbs) as usize] =
                ((ref_left[(y + 1) as usize] as i32 + 3 * dc_val + 2) >> 2) as i16;
        }
    }
}

/// Angular prediction (modes 2-34) — spec §8.4.4.2.6.
pub fn predict_angular(
    ref_top: &[i16],
    ref_left: &[i16],
    n_tbs: i32,
    intra_mode: i32,
    c_idx: i32,
    bit_depth: i32,
    pred: &mut [i16],
) {
    let angle = INTRA_PRED_ANGLE[intra_mode as usize];
    let is_vertical = intra_mode >= 18;

    // Select reference arrays: vertical modes use top as main, horizontal use left.
    // For negative angles the main reference is extended with side-reference
    // projections at negative positions (eq 8-54/8-62). `ref_base` shifts slice
    // indices into the extended buffer (C++: `refMain = refMainExt + offset`).
    let mut ref_main_ext = [0i16; 2 * 64 + 1 + 64];
    let (ref_base, ref_main): (i32, &[i16]) = if angle < 0 {
        let inv_a = INV_ANGLE[intra_mode as usize];
        // Range is x = -1 .. floor((nTbS * angle) / 32); the spec requires floor
        // division, so use an explicit ceil of the magnitude for n < 0.
        let n_times_angle = n_tbs * angle; // negative
        let num_neg = (-n_times_angle + 31) >> 5; // ceil(|nTimesAngle| / 32)

        let offset = num_neg;
        let main = if is_vertical { ref_top } else { ref_left };
        for i in 0..=2 * n_tbs {
            ref_main_ext[(offset + i) as usize] = main[i as usize];
        }

        // Project side reference into negative positions (stored positive invAngle,
        // so use (-i) to compensate).
        let side = if is_vertical { ref_left } else { ref_top };
        for i in (-num_neg..=-1).rev() {
            let mut side_idx = ((-i * inv_a + 128) >> 8) as usize;
            if side_idx > 2 * n_tbs as usize {
                side_idx = 2 * n_tbs as usize;
            }
            ref_main_ext[(offset + i) as usize] = side[side_idx];
        }

        (num_neg, &ref_main_ext[..=offset as usize + 2 * n_tbs as usize])
    } else if is_vertical {
        (0, ref_top)
    } else {
        (0, ref_left)
    };

    // Generate prediction samples
    for y in 0..n_tbs {
        for x in 0..n_tbs {
            let (i_idx, i_fact) = if is_vertical {
                let t = (y + 1) * angle;
                (t >> 5, t & 31)
            } else {
                let t = (x + 1) * angle;
                (t >> 5, t & 31)
            };

            let ref_idx = if is_vertical { x + 1 + i_idx } else { y + 1 + i_idx };

            debug_assert!(
                ref_idx + ref_base >= 0 && ref_idx + ref_base < ref_main.len() as i32,
                "mode={intra_mode} n={n_tbs} vert={is_vertical} x={x} y={y} iIdx={i_idx} iFact={i_fact} refIdx={ref_idx} base={ref_base} len={}",
                ref_main.len()
            );

            let val = if i_fact != 0 {
                ((32 - i_fact) * ref_main[(ref_idx + ref_base) as usize] as i32
                    + i_fact * ref_main[(ref_idx + 1 + ref_base) as usize] as i32
                    + 16)
                    >> 5
            } else {
                ref_main[(ref_idx + ref_base) as usize] as i32
            };

            pred[(y * n_tbs + x) as usize] = val as i16;
        }
    }

    // §8.4.4.2.6 eq 8-60/8-68: post-filtering for exact H/V (cIdx == 0 only)
    if c_idx == 0 && intra_mode == 26 && n_tbs < 32 {
        // Vertical: filter first column with left reference
        for y in 0..n_tbs {
            let i = y * n_tbs;
            pred[i as usize] = clip3(
                0,
                (1 << bit_depth) - 1,
                pred[i as usize] as i32 + ((ref_left[(y + 1) as usize] as i32 - ref_left[0] as i32) >> 1),
            ) as i16;
        }
    } else if c_idx == 0 && intra_mode == 10 && n_tbs < 32 {
        // Horizontal: filter first row with top reference
        for x in 0..n_tbs {
            pred[x as usize] = clip3(
                0,
                (1 << bit_depth) - 1,
                pred[x as usize] as i32 + ((ref_top[(x + 1) as usize] as i32 - ref_top[0] as i32) >> 1),
            ) as i16;
        }
    }
}

/// Full intra prediction for a block — spec §8.4.4.2: reference sample
/// construction, optional smoothing (§8.4.4.2.3), and mode dispatch.
/// `pred`: output of `n_tbs * n_tbs` int16 samples, where
/// `n_tbs = 1 << log2_pred_size`.
#[allow(clippy::too_many_arguments)] // faithful port of the C++ signature
pub fn perform_intra_prediction(
    pic: &Picture,
    sps: &Sps,
    pps: &Pps,
    x0: i32,
    y0: i32,
    log2_pred_size: i32,
    c_idx: i32,
    intra_mode: i32,
    slice_idx: Option<&[u8]>,
    pred: &mut [i16],
) {
    let n_tbs = 1 << log2_pred_size;
    let bit_depth = if c_idx == 0 { sps.bit_depth_y } else { sps.bit_depth_c };

    // Build reference samples
    let (mut ref_top, mut ref_left) =
        build_reference_samples(pic, sps, pps, x0, y0, n_tbs, c_idx, slice_idx);

    // §8.4.4.2.3: Reference sample filtering
    // Spec: filtering applies when intra_smoothing_disabled_flag == 0 AND
    // (cIdx == 0 OR ChromaArrayType == 3)
    let do_filter = !sps.intra_smoothing_disabled_flag
        && (c_idx == 0 || sps.chroma_array_type == 3)
        && needs_filtering(intra_mode, log2_pred_size);
    if do_filter {
        // §8.4.4.2.3: biIntFlag requires ALL conditions:
        // strong_intra_smoothing, cIdx==0, nTbS==32,
        // top smoothness check AND left smoothness check
        let mut bi_int_flag = false;
        if sps.strong_intra_smoothing_enabled_flag && c_idx == 0 && n_tbs == 32 {
            let threshold = 1 << (bit_depth - 5);
            let top_left = ref_top[0] as i32;
            let top_smooth = (top_left + ref_top[2 * n_tbs as usize] as i32
                - 2 * ref_top[n_tbs as usize] as i32)
                .abs()
                < threshold;
            let left_smooth = (top_left + ref_left[2 * n_tbs as usize] as i32
                - 2 * ref_left[n_tbs as usize] as i32)
                .abs()
                < threshold;
            bi_int_flag = top_smooth && left_smooth;
        }
        // §8.4.4.2.3 eq 8-41: corner sample filtered using both neighbours
        let filtered_corner = (ref_left[1] as i32 + 2 * ref_top[0] as i32 + ref_top[1] as i32 + 2)
            >> 2;
        filter_reference_samples(&mut ref_top, n_tbs, bi_int_flag);
        filter_reference_samples(&mut ref_left, n_tbs, bi_int_flag);
        // Apply cross-filtered corner to both arrays
        if !bi_int_flag {
            ref_top[0] = filtered_corner as i16;
            ref_left[0] = filtered_corner as i16;
        }
    }

    // Dispatch to prediction mode
    match intra_mode {
        0 => predict_planar(&ref_top, &ref_left, n_tbs, pred),
        1 => predict_dc(&ref_top, &ref_left, n_tbs, log2_pred_size, c_idx, pred),
        _ => predict_angular(&ref_top, &ref_left, n_tbs, intra_mode, c_idx, bit_depth, pred),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::goldens;
    use crate::hevc::types::ChromaFormat;

    use vacc_common::rng::SplitMix64;

    #[test]
    fn angle_tables_match_spec() {
        // Mode 0/1: no angle. Modes 2-9: positive (horizontal). Mode 10: exact
        // horizontal (angle 0). Modes 11-17: negative. Mode 26: exact vertical.
        assert_eq!(INTRA_PRED_ANGLE[0], 0);
        assert_eq!(INTRA_PRED_ANGLE[1], 0);
        assert_eq!(INTRA_PRED_ANGLE[2], 32);
        assert_eq!(INTRA_PRED_ANGLE[9], 2);
        assert_eq!(INTRA_PRED_ANGLE[10], 0);
        assert_eq!(INTRA_PRED_ANGLE[11], -2);
        assert_eq!(INTRA_PRED_ANGLE[17], -26);
        assert_eq!(INTRA_PRED_ANGLE[18], -32);
        assert_eq!(INTRA_PRED_ANGLE[26], 0);
        assert_eq!(INTRA_PRED_ANGLE[34], 32);
        assert_eq!(INV_ANGLE[26], 0);
        assert_eq!(INV_ANGLE[2], 256);
        assert_eq!(INV_ANGLE[9], 4096);
        assert_eq!(INV_ANGLE[18], 256);
    }

    #[test]
    fn needs_filtering_table() {
        // DC and 4x4 never filter.
        assert!(!needs_filtering(1, 2));
        assert!(!needs_filtering(26, 2));
        // nTbS=8 (log2=3): threshold 7 → only |mode-26| and |mode-10| both > 7.
        assert!(needs_filtering(2, 3)); // min(|2-26|,|2-10|) = 8 > 7
        assert!(needs_filtering(18, 3)); // min(8, 8) = 8 > 7
        assert!(!needs_filtering(19, 3)); // min(7, 9) = 7, not > 7
        assert!(!needs_filtering(25, 3)); // min(1, 15) = 1
        assert!(!needs_filtering(10, 3)); // dist 0
        assert!(!needs_filtering(26, 3)); // dist 0
        // nTbS=16 (log2=4): threshold 1.
        assert!(needs_filtering(24, 4)); // min(2, 14) = 2 > 1
        assert!(!needs_filtering(25, 4)); // min(1, 15) = 1, not > 1
        // nTbS=32 (log2=5): threshold 0 → everything except exact H/V/DC.
        assert!(needs_filtering(25, 5)); // dist 1 > 0
        assert!(!needs_filtering(26, 5)); // dist 0
    }

    fn compute_intra() -> Vec<(String, Vec<u8>)> {
        let mut rng = SplitMix64::seeded(0x5EED_0010);
        let pic_w = 64i32;
        let pic_h = 64i32;
        let mut buf = Vec::new();

        for bit_depth in [8i32, 10] {
            let max_val = (1u32 << bit_depth) as u16;
            for ctb_size in [32i32, 64] {
                let sps = Sps {
                    pic_width_in_luma_samples: pic_w,
                    pic_height_in_luma_samples: pic_h,
                    bit_depth_y: bit_depth,
                    bit_depth_c: bit_depth,
                    chroma_array_type: 1, // 4:2:0
                    ctb_size_y: ctb_size,
                    min_tb_size_y: 4,
                    pic_width_in_ctbs_y: (pic_w + ctb_size - 1) / ctb_size,
                    sub_width_c: 2,
                    sub_height_c: 2,
                    intra_smoothing_disabled_flag: false,
                    strong_intra_smoothing_enabled_flag: true,
                    ..Default::default()
                };

                // Positions exercise: picture corner (no neighbours), top edge,
                // cross-CTU boundary (left neighbour in a previous CTU), CTU
                // corners, and Z-scan ordering within a CTU.
                let positions = [(0i32, 0), (4, 4), (8, 0), (0, 8), (32, 0), (32, 32), (60, 0), (56, 60)];

                for &(x0, y0) in &positions {
                    for log2_size in 2..=5i32 {
                        let n_tbs = 1 << log2_size;
                        for c_idx in 0..2i32 {
                            let (comp_w, comp_h) = if c_idx == 0 {
                                (pic_w, pic_h)
                            } else {
                                (pic_w / 2, pic_h / 2)
                            };
                            // Keep the block inside the picture.
                            if x0 + n_tbs > comp_w || y0 + n_tbs > comp_h {
                                continue;
                            }
                            let stride = comp_w;
                            let plane: Vec<u16> = (0..comp_w * comp_h)
                                .map(|_| rng.below(max_val as u64) as u16)
                                .collect();

                            let mut pic = Picture::default();
                            pic.planes[c_idx as usize] = plane.clone();
                            pic.width[c_idx as usize] = comp_w;
                            pic.height[c_idx as usize] = comp_h;
                            pic.stride[c_idx as usize] = stride;

                            for mode in 0..35i32 {
                                let n_samples = (n_tbs * n_tbs) as usize;
                                let mut pred_rs = vec![0i16; n_samples];

                                perform_intra_prediction(
                                    &pic, &sps, &Pps::default(), x0, y0, log2_size, c_idx, mode,
                                    None, &mut pred_rs,
                                );
                                for v in &pred_rs {
                                    goldens::push_i16(&mut buf, *v);
                                }
                            }
                        }
                    }
                }
            }
        }
        vec![("intra::prediction".to_string(), buf)]
    }

    #[test]
    fn intra_prediction_matches_golden() {
        for (key, data) in compute_intra() {
            goldens::assert_golden(&key, &data);
        }
    }

    fn compute_intra_smoothing_disabled() -> Vec<(String, Vec<u8>)> {
        // intra_smoothing_disabled_flag = 1 (skips §8.4.4.2.3).
        let mut rng = SplitMix64::seeded(0x5EED_0011);
        let pic_w = 64i32;
        let pic_h = 64i32;
        let bit_depth = 8;
        let mut buf = Vec::new();
        let sps = Sps {
            pic_width_in_luma_samples: pic_w,
            pic_height_in_luma_samples: pic_h,
            bit_depth_y: bit_depth,
            bit_depth_c: bit_depth,
            chroma_array_type: 1,
            ctb_size_y: 32,
            min_tb_size_y: 4,
            pic_width_in_ctbs_y: 2,
            sub_width_c: 2,
            sub_height_c: 2,
            intra_smoothing_disabled_flag: true,
            strong_intra_smoothing_enabled_flag: false,
            ..Default::default()
        };

        let (x0, y0) = (16i32, 4);
        let log2_size = 4; // 16x16 — filtering would apply without the flag
        let n_tbs = 1 << log2_size;
        let plane: Vec<u16> = (0..pic_w * pic_h).map(|_| rng.below(256) as u16).collect();

        let mut pic = Picture::default();
        pic.planes[0] = plane.clone();
        pic.width[0] = pic_w;
        pic.height[0] = pic_h;
        pic.stride[0] = pic_w;

        for mode in [0i32, 1, 5, 10, 18, 26, 34] {
            let n_samples = (n_tbs * n_tbs) as usize;
            let mut pred_rs = vec![0i16; n_samples];

            perform_intra_prediction(
                &pic, &sps, &Pps::default(), x0, y0, log2_size, 0, mode, None, &mut pred_rs,
            );
            for v in &pred_rs {
                goldens::push_i16(&mut buf, *v);
            }
        }
        vec![("intra::smoothing_disabled".to_string(), buf)]
    }

    #[test]
    fn intra_smoothing_disabled_matches_golden() {
        for (key, data) in compute_intra_smoothing_disabled() {
            goldens::assert_golden(&key, &data);
        }
    }

    fn compute_intra_biint_path() -> Vec<(String, Vec<u8>)> {
        // Constant plane at the picture corner: all neighbours substitute to the
        // default value, top/left smoothness checks pass → biIntFlag = true for
        // 32x32 luma with strong smoothing (bilinear reference filtering path).
        let pic_w = 64i32;
        let pic_h = 64i32;
        let bit_depth = 8;
        let sps = Sps {
            pic_width_in_luma_samples: pic_w,
            pic_height_in_luma_samples: pic_h,
            bit_depth_y: bit_depth,
            bit_depth_c: bit_depth,
            chroma_array_type: 1,
            ctb_size_y: 64,
            min_tb_size_y: 4,
            pic_width_in_ctbs_y: 1,
            sub_width_c: 2,
            sub_height_c: 2,
            intra_smoothing_disabled_flag: false,
            strong_intra_smoothing_enabled_flag: true,
            ..Default::default()
        };

        let n_tbs = 32;
        let log2_size = 5;
        let plane = vec![100u16; (pic_w * pic_h) as usize];

        let mut pic = Picture::default();
        pic.planes[0] = plane.clone();
        pic.width[0] = pic_w;
        pic.height[0] = pic_h;
        pic.stride[0] = pic_w;

        let mut buf = Vec::new();
        for mode in [2i32, 10, 26, 34] {
            let n_samples = (n_tbs * n_tbs) as usize;
            let mut pred_rs = vec![0i16; n_samples];

            perform_intra_prediction(
                &pic, &sps, &Pps::default(), 0, 0, log2_size, 0, mode, None, &mut pred_rs,
            );
            for v in &pred_rs {
                goldens::push_i16(&mut buf, *v);
            }
        }
        vec![("intra::constant_plane_biint".to_string(), buf)]
    }

    #[test]
    fn intra_constant_plane_biint_path_matches_golden() {
        for (key, data) in compute_intra_biint_path() {
            goldens::assert_golden(&key, &data);
        }
    }

    #[test]
    fn picture_chroma_format_used_by_allocate() {
        // Sanity: the allocate() path used by tests produces the dims the
        // intra oracle expects for 4:2:0.
        let mut pic = Picture::default();
        pic.allocate(64, 64, ChromaFormat::Yuv420, 8, 8);
        assert_eq!(pic.width[1], 32);
        assert_eq!(pic.height[1], 32);
    }

    pub(crate) fn golden_entries() -> Vec<(String, String)> {
        let mut v = Vec::new();
        for (k, b) in compute_intra() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        for (k, b) in compute_intra_smoothing_disabled() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        for (k, b) in compute_intra_biint_path() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        v
    }
}

//! Port of `hevc/decoding/interpolation.{h,cpp}` — motion compensation kernels,
//! spec §8.5.3.3.
//!
//! Luma 8-tap (§8.5.3.3.3), chroma 4-tap, and weighted prediction
//! (§8.5.3.3.4). The top-level `perform_inter_prediction` dispatcher (which
//! resolves reference pictures from the DPB) is ported with the inter
//! prediction layer; these kernels are its building blocks.

use crate::hevc::types::{clip3, Mv};

/// Luma 8-tap interpolation filter coefficients — Table 8-1.
/// Index 0 = integer (not used in filtering, just copy).
pub const LUMA_FILTER: [[i16; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0], // frac=0 (integer)
    [-1, 4, -10, 58, 17, -5, 1, 0], // frac=1 (1/4)
    [-1, 4, -11, 40, 40, -11, 4, -1], // frac=2 (1/2)
    [0, 1, -5, 17, 58, -10, 4, -1], // frac=3 (3/4)
];

/// Chroma 4-tap interpolation filter coefficients — Table 8-2.
pub const CHROMA_FILTER: [[i16; 4]; 8] = [
    [0, 64, 0, 0], // frac=0 (integer)
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// Per-reference-entry weights — spec §7.3.6.3.
/// (The `*_weight_flag` syntax fields are omitted: the kernels only consume
/// the numeric weight/offset values.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefWeight {
    pub luma_weight: i16,
    pub luma_offset: i16,
    /// Cb, Cr.
    pub chroma_weight: [i16; 2],
    /// Cb, Cr.
    pub chroma_offset: [i16; 2],
}

/// Prediction weight table — spec §7.3.6.3 (max 16 refs per list).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PredWeightTable {
    pub luma_log2_weight_denom: u32,
    pub delta_chroma_log2_weight_denom: i32,
    pub l0: [RefWeight; 16],
    pub l1: [RefWeight; 16],
}

/// Luma 8-tap interpolation — spec §8.5.3.3.3.
/// Output in extended precision (not clipped to `[0, 2^bitDepth-1]`).
/// `plane`: reference luma samples, row-major with `stride` samples/row.
/// `pred`: output of `n_pb_w * n_pb_h` int16 samples.
#[allow(clippy::too_many_arguments)] // faithful port of the C++ signature
pub fn interpolate_luma(
    plane: &[u16],
    pic_w: i32,
    pic_h: i32,
    stride: i32,
    x_int: i32,
    y_int: i32,
    x_frac: i32,
    y_frac: i32,
    n_pb_w: usize,
    n_pb_h: usize,
    bit_depth: i32,
    pred: &mut [i16],
) {
    // §8.5.3.3.3: shift1 = Min(4, BitDepthY - 8), shift2 = 6, shift3 = Max(2, 14 - BitDepthY)
    let shift1 = 4.min(bit_depth - 8);
    let shift2 = 6;
    let shift3 = 2.max(14 - bit_depth);

    // Check if all reference accesses are within bounds (including filter margin)
    let interior = x_int - 3 >= 0
        && y_int - 3 >= 0
        && x_int + n_pb_w as i32 + 4 <= pic_w
        && y_int + n_pb_h as i32 + 4 <= pic_h;

    if interior {
        // Fast direct access (no bounds check) — used for interior PUs
        let ref_fn = |x: i32, y: i32| plane[(y * stride + x) as usize] as i32;
        luma_interp_core(
            ref_fn, x_int, y_int, x_frac, y_frac, n_pb_w, n_pb_h, shift1, shift2, shift3, pred,
        );
    } else {
        // Safe clamped access — used for edge PUs
        let ref_fn = |x: i32, y: i32| {
            let x = x.clamp(0, pic_w - 1);
            let y = y.clamp(0, pic_h - 1);
            plane[(y * stride + x) as usize] as i32
        };
        luma_interp_core(
            ref_fn, x_int, y_int, x_frac, y_frac, n_pb_w, n_pb_h, shift1, shift2, shift3, pred,
        );
    }
}

/// Shared filter body for interior (direct) and edge (clamped) access.
#[allow(clippy::too_many_arguments)]
fn luma_interp_core<R: Fn(i32, i32) -> i32>(
    ref_fn: R,
    x_int: i32,
    y_int: i32,
    x_frac: i32,
    y_frac: i32,
    n_pb_w: usize,
    n_pb_h: usize,
    shift1: i32,
    shift2: i32,
    shift3: i32,
    pred: &mut [i16],
) {
    if x_frac == 0 && y_frac == 0 {
        for y in 0..n_pb_h {
            for x in 0..n_pb_w {
                pred[y * n_pb_w + x] = (ref_fn(x_int + x as i32, y_int + y as i32) << shift3) as i16;
            }
        }
    } else if y_frac == 0 {
        let f = LUMA_FILTER[x_frac as usize];
        for y in 0..n_pb_h {
            for x in 0..n_pb_w {
                let mut sum = 0i32;
                for (k, &tap) in f.iter().enumerate() {
                    sum += tap as i32 * ref_fn(x_int + x as i32 + k as i32 - 3, y_int + y as i32);
                }
                pred[y * n_pb_w + x] = (sum >> shift1) as i16;
            }
        }
    } else if x_frac == 0 {
        let f = LUMA_FILTER[y_frac as usize];
        for y in 0..n_pb_h {
            for x in 0..n_pb_w {
                let mut sum = 0i32;
                for (k, &tap) in f.iter().enumerate() {
                    sum += tap as i32 * ref_fn(x_int + x as i32, y_int + y as i32 + k as i32 - 3);
                }
                pred[y * n_pb_w + x] = (sum >> shift1) as i16;
            }
        }
    } else {
        let tmp_h = n_pb_h + 7;
        debug_assert!(n_pb_w <= 64 && tmp_h <= 71);
        let mut tmp = [0i16; 64 * 71];
        let f_h = LUMA_FILTER[x_frac as usize];
        for y in 0..tmp_h {
            for x in 0..n_pb_w {
                let mut sum = 0i32;
                for (k, &tap) in f_h.iter().enumerate() {
                    sum += tap as i32 * ref_fn(x_int + x as i32 + k as i32 - 3, y_int + y as i32 - 3);
                }
                tmp[y * n_pb_w + x] = (sum >> shift1) as i16;
            }
        }
        let f_v = LUMA_FILTER[y_frac as usize];
        for y in 0..n_pb_h {
            for x in 0..n_pb_w {
                let mut sum = 0i32;
                for (k, &tap) in f_v.iter().enumerate() {
                    sum += tap as i32 * tmp[(y + k) * n_pb_w + x] as i32;
                }
                pred[y * n_pb_w + x] = (sum >> shift2) as i16;
            }
        }
    }
}

/// Chroma 4-tap interpolation — spec §8.5.3.3.3 (chroma part).
/// Output in extended precision. `plane`: reference chroma samples for
/// component `c_idx`, row-major with `stride` samples/row.
#[allow(clippy::too_many_arguments)] // faithful port of the C++ signature
pub fn interpolate_chroma(
    plane: &[u16],
    _c_idx: i32,
    pic_w: i32,
    pic_h: i32,
    stride: i32,
    x_int: i32,
    y_int: i32,
    x_frac: i32,
    y_frac: i32,
    n_pb_wc: usize,
    n_pb_hc: usize,
    bit_depth: i32,
    pred: &mut [i16],
) {
    let shift1 = 4.min(bit_depth - 8);
    let shift2 = 6;
    let shift3 = 2.max(14 - bit_depth);

    // Chroma filter margin is 1 (4-tap: positions -1..+2)
    let interior = x_int > 0
        && y_int > 0
        && x_int + n_pb_wc as i32 + 2 <= pic_w
        && y_int + n_pb_hc as i32 + 2 <= pic_h;

    if interior {
        let ref_fn = |x: i32, y: i32| plane[(y * stride + x) as usize] as i32;
        chroma_interp_core(
            ref_fn,
            x_int,
            y_int,
            x_frac,
            y_frac,
            n_pb_wc,
            n_pb_hc,
            shift1,
            shift2,
            shift3,
            pred,
        );
    } else {
        let ref_fn = |x: i32, y: i32| {
            let x = x.clamp(0, pic_w - 1);
            let y = y.clamp(0, pic_h - 1);
            plane[(y * stride + x) as usize] as i32
        };
        chroma_interp_core(
            ref_fn,
            x_int,
            y_int,
            x_frac,
            y_frac,
            n_pb_wc,
            n_pb_hc,
            shift1,
            shift2,
            shift3,
            pred,
        );
    }
}

/// Shared 4-tap filter body for interior (direct) and edge (clamped) access.
#[allow(clippy::too_many_arguments)]
fn chroma_interp_core<R: Fn(i32, i32) -> i32>(
    ref_fn: R,
    x_int: i32,
    y_int: i32,
    x_frac: i32,
    y_frac: i32,
    n_pb_wc: usize,
    n_pb_hc: usize,
    shift1: i32,
    shift2: i32,
    shift3: i32,
    pred: &mut [i16],
) {
    if x_frac == 0 && y_frac == 0 {
        for y in 0..n_pb_hc {
            for x in 0..n_pb_wc {
                pred[y * n_pb_wc + x] = (ref_fn(x_int + x as i32, y_int + y as i32) << shift3) as i16;
            }
        }
    } else if y_frac == 0 {
        let f = CHROMA_FILTER[x_frac as usize];
        for y in 0..n_pb_hc {
            for x in 0..n_pb_wc {
                let mut sum = 0i32;
                for (k, &tap) in f.iter().enumerate() {
                    sum += tap as i32 * ref_fn(x_int + x as i32 + k as i32 - 1, y_int + y as i32);
                }
                pred[y * n_pb_wc + x] = (sum >> shift1) as i16;
            }
        }
    } else if x_frac == 0 {
        let f = CHROMA_FILTER[y_frac as usize];
        for y in 0..n_pb_hc {
            for x in 0..n_pb_wc {
                let mut sum = 0i32;
                for (k, &tap) in f.iter().enumerate() {
                    sum += tap as i32 * ref_fn(x_int + x as i32, y_int + y as i32 + k as i32 - 1);
                }
                pred[y * n_pb_wc + x] = (sum >> shift1) as i16;
            }
        }
    } else {
        let tmp_h = n_pb_hc + 3;
        debug_assert!(n_pb_wc <= 32 && tmp_h <= 35);
        let mut tmp = [0i16; 32 * 35];
        let f_h = CHROMA_FILTER[x_frac as usize];
        for y in 0..tmp_h {
            for x in 0..n_pb_wc {
                let mut sum = 0i32;
                for (k, &tap) in f_h.iter().enumerate() {
                    sum += tap as i32 * ref_fn(x_int + x as i32 + k as i32 - 1, y_int + y as i32 - 1);
                }
                tmp[y * n_pb_wc + x] = (sum >> shift1) as i16;
            }
        }
        let f_v = CHROMA_FILTER[y_frac as usize];
        for y in 0..n_pb_hc {
            for x in 0..n_pb_wc {
                let mut sum = 0i32;
                for (k, &tap) in f_v.iter().enumerate() {
                    sum += tap as i32 * tmp[(y + k) * n_pb_wc + x] as i32;
                }
                pred[y * n_pb_wc + x] = (sum >> shift2) as i16;
            }
        }
    }
}

/// Luma MV decomposition — 1/4 pel precision.
/// Returns `(x_int, y_int, x_frac, y_frac)` relative to the PU origin.
#[inline]
pub fn luma_mv_position(x_pb: i32, y_pb: i32, mv: Mv) -> (i32, i32, i32, i32) {
    let mx = mv.x as i32;
    let my = mv.y as i32;
    (x_pb + (mx >> 2), y_pb + (my >> 2), mx & 3, my & 3)
}

/// Chroma MV decomposition for 4:2:0 — the luma 1/4 pel maps to chroma
/// 1/8 pel (spec §8.5.3.3.2). `(x_pb_c, y_pb_c)` are PU coordinates in
/// chroma samples. Returns `(x_int, y_int, x_frac, y_frac)`.
#[inline]
pub fn chroma_mv_position(x_pb_c: i32, y_pb_c: i32, mv: Mv) -> (i32, i32, i32, i32) {
    let mx = mv.x as i32;
    let my = mv.y as i32;
    (x_pb_c + (mx >> 3), y_pb_c + (my >> 3), mx & 7, my & 7)
}

/// Default weighted sample prediction — spec §8.5.3.3.4.2.
/// `n_samples` = number of samples in the PU for this component.
pub fn weighted_pred_default(
    pred_l0: &[i16],
    pred_l1: &[i16],
    flag_l0: bool,
    flag_l1: bool,
    n_samples: i32,
    bit_depth: i32,
    output: &mut [i16],
) {
    // §8.5.3.3.4.2
    let shift1 = 2.max(14 - bit_depth);
    let offset1 = 1 << (shift1 - 1);
    let shift2 = 3.max(15 - bit_depth);
    let offset2 = 1 << (shift2 - 1);
    let max_val = (1 << bit_depth) - 1;

    if flag_l0 && !flag_l1 {
        // eq 8-262: uni-pred L0
        for i in 0..n_samples {
            output[i as usize] =
                clip3(0, max_val, (pred_l0[i as usize] as i32 + offset1) >> shift1) as i16;
        }
    } else if !flag_l0 && flag_l1 {
        // eq 8-263: uni-pred L1
        for i in 0..n_samples {
            output[i as usize] =
                clip3(0, max_val, (pred_l1[i as usize] as i32 + offset1) >> shift1) as i16;
        }
    } else {
        // eq 8-264: bi-pred
        for i in 0..n_samples {
            output[i as usize] = clip3(
                0,
                max_val,
                (pred_l0[i as usize] as i32 + pred_l1[i as usize] as i32 + offset2) >> shift2,
            ) as i16;
        }
    }
}

/// Explicit weighted sample prediction — spec §8.5.3.3.4.3.
#[allow(clippy::too_many_arguments)] // faithful port of the C++ signature
pub fn weighted_pred_explicit(
    pred_l0: &[i16],
    pred_l1: &[i16],
    flag_l0: bool,
    flag_l1: bool,
    ref_idx_l0: i32,
    ref_idx_l1: i32,
    c_idx: i32,
    n_samples: i32,
    bit_depth: i32,
    pwt: &PredWeightTable,
    output: &mut [i16],
) {
    // §8.5.3.3.4.3
    let shift1 = 2.max(14 - bit_depth);
    let max_val = (1 << bit_depth) - 1;

    let (log2_wd, w0, w1, o0, o1) = if c_idx == 0 {
        // Luma — eq 8-265..8-269
        let l0 = &pwt.l0[ref_idx_l0.max(0) as usize];
        let l1 = &pwt.l1[ref_idx_l1.max(0) as usize];
        // WpOffsetBdShiftY = BitDepthY - 8
        let wp_shift_y = bit_depth - 8;
        (
            pwt.luma_log2_weight_denom as i32 + shift1,
            l0.luma_weight as i32,
            l1.luma_weight as i32,
            (l0.luma_offset as i32) << wp_shift_y,
            (l1.luma_offset as i32) << wp_shift_y,
        )
    } else {
        // Chroma — eq 8-270..8-274
        let chroma_log2_weight_denom =
            pwt.luma_log2_weight_denom as i32 + pwt.delta_chroma_log2_weight_denom;
        let ci = (c_idx - 1) as usize; // 0=Cb, 1=Cr
        let l0 = &pwt.l0[ref_idx_l0.max(0) as usize];
        let l1 = &pwt.l1[ref_idx_l1.max(0) as usize];
        // WpOffsetBdShiftC = BitDepthC - 8
        let wp_shift_c = bit_depth - 8;
        (
            chroma_log2_weight_denom + shift1,
            l0.chroma_weight[ci] as i32,
            l1.chroma_weight[ci] as i32,
            (l0.chroma_offset[ci] as i32) << wp_shift_c,
            (l1.chroma_offset[ci] as i32) << wp_shift_c,
        )
    };

    if flag_l0 && !flag_l1 {
        // eq 8-275: uni-pred L0
        let round = 1 << (log2_wd - 1);
        for i in 0..n_samples {
            output[i as usize] = clip3(
                0,
                max_val,
                ((pred_l0[i as usize] as i32 * w0 + round) >> log2_wd) + o0,
            ) as i16;
        }
    } else if !flag_l0 && flag_l1 {
        // eq 8-276: uni-pred L1
        let round = 1 << (log2_wd - 1);
        for i in 0..n_samples {
            output[i as usize] = clip3(
                0,
                max_val,
                ((pred_l1[i as usize] as i32 * w1 + round) >> log2_wd) + o1,
            ) as i16;
        }
    } else {
        // eq 8-277: bi-pred
        for i in 0..n_samples {
            output[i as usize] = clip3(
                0,
                max_val,
                (pred_l0[i as usize] as i32 * w0
                    + pred_l1[i as usize] as i32 * w1
                    + ((o0 + o1 + 1) << log2_wd))
                    >> (log2_wd + 1),
            ) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi_test;

    /// Deterministic xorshift64* RNG (same scheme as bitreader tests).
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
    }

    #[test]
    fn filter_tables_match_spec() {
        // Table 8-1 / Table 8-2 spot checks (sum = 64 per row).
        for row in LUMA_FILTER {
            assert_eq!(row.iter().map(|&v| v as i32).sum::<i32>(), 64);
        }
        for row in CHROMA_FILTER {
            assert_eq!(row.iter().map(|&v| v as i32).sum::<i32>(), 64);
        }
        assert_eq!(LUMA_FILTER[0], [0, 0, 0, 64, 0, 0, 0, 0]);
        assert_eq!(CHROMA_FILTER[4], [-4, 36, 36, -4]);
    }

    #[test]
    fn luma_mv_position_decomposition() {
        // Positive and negative MVs (arithmetic shift / two's-complement &).
        assert_eq!(luma_mv_position(0, 0, Mv { x: 0, y: 0 }), (0, 0, 0, 0));
        assert_eq!(luma_mv_position(8, 4, Mv { x: 5, y: -1 }), (9, 3, 1, 3));
        assert_eq!(luma_mv_position(0, 0, Mv { x: -4, y: -4 }), (-1, -1, 0, 0));
        assert_eq!(luma_mv_position(2, 2, Mv { x: -1, y: 7 }), (1, 3, 3, 3));

        // Chroma 4:2:0: 1/4 luma pel = 1/8 chroma pel.
        assert_eq!(chroma_mv_position(4, 2, Mv { x: 5, y: -1 }), (4, 1, 5, 7));
        assert_eq!(chroma_mv_position(0, 0, Mv { x: -8, y: 8 }), (-1, 1, 0, 0));
    }

    #[test]
    fn luma_interpolation_matches_cpp() {
        let mut rng = Rng::new(0x5EED_0001);
        let pic_w = 64i32;
        let pic_h = 64i32;
        let stride = pic_w;
        let plane: Vec<u16> = (0..pic_w * pic_h).map(|_| rng.below(1024) as u16).collect();

        for &(n_pb_w, n_pb_h) in &[(4usize, 4), (8, 16), (16, 8), (32, 32), (64, 64)] {
            for bit_depth in [8i32, 10] {
                for iter in 0..40 {
                    // xInt/yInt span negative (clamped), interior, and edge positions.
                    let x_int = rng.below(pic_w as u64 + 16) as i32 - 8;
                    let y_int = rng.below(pic_h as u64 + 16) as i32 - 8;
                    let x_frac = rng.below(4) as i32;
                    let y_frac = rng.below(4) as i32;

                    let n_samples = n_pb_w * n_pb_h;
                    let mut pred_rs = vec![0i16; n_samples];
                    let mut pred_cpp = vec![0i16; n_samples];

                    interpolate_luma(
                        &plane, pic_w, pic_h, stride, x_int, y_int, x_frac, y_frac,
                        n_pb_w, n_pb_h, bit_depth, &mut pred_rs,
                    );
                    let rc = unsafe {
                        ffi_test::hevcdec_test_interpolate_luma(
                            plane.as_ptr(), pic_w, pic_h, stride, x_int, y_int, x_frac, y_frac,
                            n_pb_w as i32, n_pb_h as i32, bit_depth, pred_cpp.as_mut_ptr(),
                        )
                    };
                    assert_eq!(rc, 0);

                    if pred_rs != pred_cpp {
                        let diff: Vec<usize> = (0..n_samples)
                            .filter(|&i| pred_rs[i] != pred_cpp[i])
                            .take(8)
                            .collect();
                        panic!(
                            "luma mismatch bd={bit_depth} size={n_pb_w}x{n_pb_h} pos=({x_int},{y_int}) frac=({x_frac},{y_frac}) iter={iter} first_diffs={diff:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn chroma_interpolation_matches_cpp() {
        let mut rng = Rng::new(0x5EED_0002);
        let pic_w = 32i32;
        let pic_h = 16i32;
        let stride = pic_w;
        let plane: Vec<u16> = (0..pic_w * pic_h).map(|_| rng.below(1024) as u16).collect();

        for &(n_pb_wc, n_pb_hc) in &[(4usize, 4), (8, 8), (16, 16), (32, 16)] {
            for bit_depth in [8i32, 10] {
                for iter in 0..40 {
                    let x_int = rng.below(pic_w as u64 + 8) as i32 - 4;
                    let y_int = rng.below(pic_h as u64 + 8) as i32 - 4;
                    let x_frac = rng.below(8) as i32;
                    let y_frac = rng.below(8) as i32;

                    let n_samples = n_pb_wc * n_pb_hc;
                    let mut pred_rs = vec![0i16; n_samples];
                    let mut pred_cpp = vec![0i16; n_samples];

                    interpolate_chroma(
                        &plane, 1, pic_w, pic_h, stride, x_int, y_int, x_frac, y_frac,
                        n_pb_wc, n_pb_hc, bit_depth, &mut pred_rs,
                    );
                    let rc = unsafe {
                        ffi_test::hevcdec_test_interpolate_chroma(
                            plane.as_ptr(), 1, pic_w, pic_h, stride, x_int, y_int, x_frac, y_frac,
                            n_pb_wc as i32, n_pb_hc as i32, bit_depth, pred_cpp.as_mut_ptr(),
                        )
                    };
                    assert_eq!(rc, 0);

                    if pred_rs != pred_cpp {
                        let diff: Vec<usize> = (0..n_samples)
                            .filter(|&i| pred_rs[i] != pred_cpp[i])
                            .take(8)
                            .collect();
                        panic!(
                            "chroma mismatch bd={bit_depth} size={n_pb_wc}x{n_pb_hc} pos=({x_int},{y_int}) frac=({x_frac},{y_frac}) iter={iter} first_diffs={diff:?}"
                        );
                    }
                }
            }
        }
    }

    fn random_pwt(
        rng: &mut Rng,
        log2_denom: u32,
    ) -> (PredWeightTable, [i16; 32], [i16; 32], [i16; 64], [i16; 64]) {
        let mut pwt = PredWeightTable {
            luma_log2_weight_denom: log2_denom,
            ..Default::default()
        };
        let mut w_luma = [0i16; 32];
        let mut o_luma = [0i16; 32];
        let mut w_chroma = [0i16; 64];
        let mut o_chroma = [0i16; 64];
        for list in 0..2 {
            for r in 0..16 {
                let lw = (rng.below(65) as i32 - 32) as i16;
                let lo = (rng.below(41) as i32 - 20) as i16;
                w_luma[list * 16 + r] = lw;
                o_luma[list * 16 + r] = lo;
                for c in 0..2 {
                    let cw = (rng.below(65) as i32 - 32) as i16;
                    let co = (rng.below(41) as i32 - 20) as i16;
                    w_chroma[(list * 2 + c) * 16 + r] = cw;
                    o_chroma[(list * 2 + c) * 16 + r] = co;
                }
            }
        }
        // Fill the Rust PWT from the same flat arrays the oracle receives.
        for list in 0..2 {
            let dst = if list == 0 { &mut pwt.l0 } else { &mut pwt.l1 };
            for r in 0..16 {
                dst[r].luma_weight = w_luma[list * 16 + r];
                dst[r].luma_offset = o_luma[list * 16 + r];
                for c in 0..2 {
                    dst[r].chroma_weight[c] = w_chroma[(list * 2 + c) * 16 + r];
                    dst[r].chroma_offset[c] = o_chroma[(list * 2 + c) * 16 + r];
                }
            }
        }
        (pwt, w_luma, o_luma, w_chroma, o_chroma)
    }

    #[test]
    fn weighted_pred_default_matches_cpp() {
        let mut rng = Rng::new(0x5EED_0003);
        for bit_depth in [8i32, 10] {
            for &(flag_l0, flag_l1) in &[(true, false), (false, true), (true, true)] {
                for _iter in 0..20 {
                    let n_samples = 16;
                    let pred_l0: Vec<i16> = (0..n_samples).map(|_| (rng.below(4097) as i32 - 2048) as i16).collect();
                    let pred_l1: Vec<i16> = (0..n_samples).map(|_| (rng.below(4097) as i32 - 2048) as i16).collect();

                    let mut out_rs = vec![0i16; n_samples];
                    let mut out_cpp = vec![0i16; n_samples];

                    weighted_pred_default(
                        &pred_l0, &pred_l1, flag_l0, flag_l1, n_samples as i32, bit_depth,
                        &mut out_rs,
                    );
                    let rc = unsafe {
                        ffi_test::hevcdec_test_weighted_pred_default(
                            pred_l0.as_ptr(), pred_l1.as_ptr(), flag_l0 as i32, flag_l1 as i32,
                            n_samples as i32, bit_depth, out_cpp.as_mut_ptr(),
                        )
                    };
                    assert_eq!(rc, 0);
                    assert_eq!(out_rs, out_cpp, "bd={bit_depth} flags=({flag_l0},{flag_l1})");
                }
            }
        }
    }

    #[test]
    fn weighted_pred_explicit_matches_cpp() {
        let mut rng = Rng::new(0x5EED_0004);
        for bit_depth in [8i32, 10] {
            for log2_denom in 0..3u32 {
                for c_idx in 0..3i32 {
                    for &(flag_l0, flag_l1) in &[(true, false), (false, true), (true, true)] {
                        let (pwt, w_luma, o_luma, w_chroma, o_chroma) = random_pwt(&mut rng, log2_denom);
                        let ref_idx_l0 = if rng.below(4) == 0 { -1 } else { rng.below(2) as i32 };
                        let ref_idx_l1 = if rng.below(4) == 0 { -1 } else { rng.below(2) as i32 };

                        let n_samples = 16;
                        let pred_l0: Vec<i16> = (0..n_samples).map(|_| (rng.below(4097) as i32 - 2048) as i16).collect();
                        let pred_l1: Vec<i16> = (0..n_samples).map(|_| (rng.below(4097) as i32 - 2048) as i16).collect();

                        let mut out_rs = vec![0i16; n_samples];
                        let mut out_cpp = vec![0i16; n_samples];

                        weighted_pred_explicit(
                            &pred_l0, &pred_l1, flag_l0, flag_l1, ref_idx_l0, ref_idx_l1,
                            c_idx, n_samples as i32, bit_depth, &pwt, &mut out_rs,
                        );
                        let rc = unsafe {
                            ffi_test::hevcdec_test_weighted_pred_explicit(
                                pred_l0.as_ptr(), pred_l1.as_ptr(), flag_l0 as i32, flag_l1 as i32,
                                ref_idx_l0, ref_idx_l1, c_idx, n_samples as i32, bit_depth,
                                log2_denom, 0,
                                w_luma.as_ptr(), o_luma.as_ptr(), w_chroma.as_ptr(), o_chroma.as_ptr(),
                                out_cpp.as_mut_ptr(),
                            )
                        };
                        assert_eq!(rc, 0);
                        assert_eq!(
                            out_rs, out_cpp,
                            "bd={bit_depth} cIdx={c_idx} log2Denom={log2_denom} flags=({flag_l0},{flag_l1}) idx=({ref_idx_l0},{ref_idx_l1})"
                        );
                    }
                }
            }
        }
    }
}

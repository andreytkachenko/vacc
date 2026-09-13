//! Port of `hevc/decoding/interpolation.{h,cpp}` — motion compensation kernels,
//! spec §8.5.3.3.
//!
//! Luma 8-tap (§8.5.3.3.3), chroma 4-tap, and weighted prediction
//! (§8.5.3.3.4). The top-level `perform_inter_prediction` dispatcher (which
//! resolves reference pictures from the DPB) is ported with the inter
//! prediction layer; these kernels are its building blocks.

#[cfg(test)]
pub(crate) use self::tests::golden_entries;

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
        // Fast direct access (no bounds check) — SIMD FIR on x86_64 when the
        // block width vectorizes, scalar core otherwise.
        #[cfg(target_arch = "x86_64")]
        if n_pb_w.is_multiple_of(4) {
            // Interior bounds were checked above; SSE2 is x86_64 baseline.
            sse2::luma_interior(
                plane, stride, x_int, y_int, x_frac, y_frac, n_pb_w, n_pb_h, shift1, shift2,
                shift3, pred,
            );
        } else {
            luma_interior_scalar(
                plane, stride, x_int, y_int, x_frac, y_frac, n_pb_w, n_pb_h, shift1, shift2,
                shift3, pred,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        luma_interior_scalar(
            plane, stride, x_int, y_int, x_frac, y_frac, n_pb_w, n_pb_h, shift1, shift2, shift3,
            pred,
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

/// Interior scalar path (direct access, no clamping) — used on non-x86_64
/// targets and when the block width does not vectorize.
#[inline]
#[allow(clippy::too_many_arguments)]
fn luma_interior_scalar(
    plane: &[u16],
    stride: i32,
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
    let ref_fn = |x: i32, y: i32| plane[(y * stride + x) as usize] as i32;
    luma_interp_core(
        ref_fn, x_int, y_int, x_frac, y_frac, n_pb_w, n_pb_h, shift1, shift2, shift3, pred,
    );
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
        // Fast direct access (no bounds check) — SIMD FIR on x86_64 when the
        // chroma block width vectorizes, scalar core otherwise.
        #[cfg(target_arch = "x86_64")]
        if n_pb_wc.is_multiple_of(4) {
            // Interior bounds were checked above; SSE2 is x86_64 baseline.
            sse2::chroma_interior(
                plane, stride, x_int, y_int, x_frac, y_frac, n_pb_wc, n_pb_hc, shift1, shift2,
                shift3, pred,
            );
        } else {
            chroma_interior_scalar(
                plane, stride, x_int, y_int, x_frac, y_frac, n_pb_wc, n_pb_hc, shift1, shift2,
                shift3, pred,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        chroma_interior_scalar(
            plane, stride, x_int, y_int, x_frac, y_frac, n_pb_wc, n_pb_hc, shift1, shift2,
            shift3, pred,
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

/// Interior scalar path (direct access, no clamping) — used on non-x86_64
/// targets and when the chroma block width does not vectorize.
#[inline]
#[allow(clippy::too_many_arguments)]
fn chroma_interior_scalar(
    plane: &[u16],
    stride: i32,
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

// ---------------------------------------------------------------------------
// SSE2 interior kernels (x86_64).
//
// Byte-exact with the scalar cores above: horizontal-pass results are wrapped
// to i16 before the vertical pass (`as i16` low-16-bit truncation == C++
// `static_cast<int16_t>` on x86_64), and all arithmetic stays in i32 where
// overflow is impossible:
//   horizontal FIR lane: |2 · 4095 · 58| ≈ 4.7e5, hsum ≤ ~1.9e6
//   vertical madd lane:  |32767 · 58| ≈ 1.9e6, 8-tap sum ≤ ~1.5e7
//
// The `[c, 0, c, 0]` madd pairing turns one `_mm_madd_epi16` into two
// same-coefficient products (even/odd sample pairs), so the vertical pass
// needs no 16→32 sign extension and stays on baseline SSE2. Callers route
// here only when the block width is a multiple of 4; anything else uses the
// scalar core.

#[cfg(target_arch = "x86_64")]
mod sse2 {
    use core::arch::x86_64::{
        __m128i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_loadu_si128,
        _mm_madd_epi16, _mm_setr_epi16, _mm_setr_epi32, _mm_setzero_si128, _mm_slli_epi32,
        _mm_srli_si128, _mm_shuffle_epi32, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
    };

    use super::{CHROMA_FILTER, LUMA_FILTER};

    /// Horizontal sum of 4 i32 lanes (result in lane 0).
    #[inline(always)]
    fn hsum4(v: __m128i) -> __m128i {
        unsafe {
            let a = _mm_add_epi32(v, _mm_shuffle_epi32::<0x4E>(v));
            _mm_add_epi32(a, _mm_shuffle_epi32::<0x39>(a))
        }
    }

    /// Zero-extend the low 4 i16 lanes to i32 (reference samples are < 2^15).
    #[inline(always)]
    fn zext4(v: __m128i) -> __m128i {
        unsafe { _mm_unpacklo_epi16(v, _mm_setzero_si128()) }
    }

    /// Left-shift 4 i32 lanes by `n` (always 0..=6 in this module).
    #[inline(always)]
    fn slli32(v: __m128i, n: i32) -> __m128i {
        match n {
            0 => v,
            1 => unsafe { _mm_slli_epi32::<1>(v) },
            2 => unsafe { _mm_slli_epi32::<2>(v) },
            3 => unsafe { _mm_slli_epi32::<3>(v) },
            4 => unsafe { _mm_slli_epi32::<4>(v) },
            5 => unsafe { _mm_slli_epi32::<5>(v) },
            6 => unsafe { _mm_slli_epi32::<6>(v) },
            _ => {
                // Unreachable for our shift values; exact scalar fallback.
                unsafe {
                    let l0 = _mm_cvtsi128_si32(v);
                    let l1 = _mm_cvtsi128_si32(_mm_srli_si128::<4>(v));
                    let l2 = _mm_cvtsi128_si32(_mm_srli_si128::<8>(v));
                    let l3 = _mm_cvtsi128_si32(_mm_srli_si128::<12>(v));
                    _mm_setr_epi32(l0 << n, l1 << n, l2 << n, l3 << n)
                }
            }
        }
    }

    /// Extract the 4 i32 lanes of `r` and write `(v >> shift) as i16` each —
    /// same wrap semantics as the scalar cores.
    #[inline(always)]
    fn store4(r: __m128i, shift: i32, out: &mut [i16]) {
        unsafe {
            out[0] = (_mm_cvtsi128_si32(r) >> shift) as i16;
            out[1] = (_mm_cvtsi128_si32(_mm_srli_si128::<4>(r)) >> shift) as i16;
            out[2] = (_mm_cvtsi128_si32(_mm_srli_si128::<8>(r)) >> shift) as i16;
            out[3] = (_mm_cvtsi128_si32(_mm_srli_si128::<12>(r)) >> shift) as i16;
        }
    }

    /// Luma 8-tap horizontal FIR for one row. `base` must point at a row with
    /// at least `out.len() + 7` valid u16 samples (interior guarantee).
    #[inline]
    fn luma_hfir_row(base: *const u16, f: [i16; 8], shift: i32, out: &mut [i16]) {
        let c = unsafe { _mm_setr_epi16(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]) };
        let mut x = 0usize;
        while x + 4 <= out.len() {
            for j in 0..4usize {
                let v = unsafe {
                    let s = _mm_loadu_si128(base.add(x + j) as *const __m128i);
                    _mm_cvtsi128_si32(hsum4(_mm_madd_epi16(s, c)))
                };
                out[x + j] = (v >> shift) as i16;
            }
            x += 4;
        }
        while x < out.len() {
            let mut sum = 0i32;
            for (k, &tap) in f.iter().enumerate() {
                sum += tap as i32 * unsafe { *base.add(x + k) } as i32;
            }
            out[x] = (sum >> shift) as i16;
            x += 1;
        }
    }

    /// Luma 8-tap vertical FIR over `tmp` (width `n`, multiple of 4, at least
    /// `rows_out + 7` rows). Signed i16 inputs via the madd pairing.
    #[inline]
    fn luma_vfir(
        tmp: &[i16],
        n: usize,
        rows_out: usize,
        f: [i16; 8],
        shift: i32,
        out: &mut [i16],
    ) {
        let zero = unsafe { _mm_setzero_si128() };
        let mut bevens = [zero; 8];
        for k in 0..8usize {
            bevens[k] = unsafe { _mm_setr_epi16(f[k], 0, f[k], 0, 0, 0, 0, 0) };
        }
        let mut y = 0usize;
        while y < rows_out {
            let mut x0 = 0usize;
            while x0 + 4 <= n {
                let (mut even_acc, mut odd_acc) = (zero, zero);
                for (k, &bev) in bevens.iter().enumerate() {
                    unsafe {
                        let base = tmp.as_ptr().add((y + k) * n + x0);
                        let v = _mm_loadl_epi64(base as *const __m128i); // [T0 T1 T2 T3 | 0 ..]
                        even_acc = _mm_add_epi32(even_acc, _mm_madd_epi16(v, bev));
                        odd_acc = _mm_add_epi32(
                            odd_acc,
                            _mm_madd_epi16(_mm_srli_si128::<2>(v), bev),
                        );
                    }
                }
                store4(
                    unsafe { _mm_unpacklo_epi32(even_acc, odd_acc) },
                    shift,
                    &mut out[y * n + x0..y * n + x0 + 4],
                );
                x0 += 4;
            }
            y += 1;
        }
    }

    /// Integer-MV copy with `<< shift3`, 4 samples at a time. `base` must
    /// point at a row with at least `n` valid u16 samples.
    #[inline]
    fn copy_shifted(base: *const u16, n: usize, shift: i32, out: &mut [i16]) {
        let mut x = 0usize;
        while x + 4 <= n {
            unsafe {
                let v = _mm_loadl_epi64(base.add(x) as *const __m128i);
                store4(slli32(zext4(v), shift), 0, &mut out[x..x + 4]);
            }
            x += 4;
        }
        while x < n {
            let s = unsafe { *base.add(x) };
            out[x] = ((s as i32) << shift) as i16;
            x += 1;
        }
    }

    /// Luma interior kernel — all four frac combinations, direct access.
    /// Precondition: the interior bounds checked by `interpolate_luma` hold
    /// (all reference accesses, including filter margins, are in range) and
    /// `n_pb_w % 4 == 0`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn luma_interior(
        plane: &[u16],
        stride: i32,
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
        debug_assert_eq!(n_pb_w % 4, 0);
        if x_frac == 0 && y_frac == 0 {
            for y in 0..n_pb_h {
                let base = unsafe {
                    plane.as_ptr().add(((y_int + y as i32) * stride + x_int) as usize)
                };
                copy_shifted(base, n_pb_w, shift3, &mut pred[y * n_pb_w..(y + 1) * n_pb_w]);
            }
        } else if y_frac == 0 {
            let f = LUMA_FILTER[x_frac as usize];
            for y in 0..n_pb_h {
                let base = unsafe {
                    plane.as_ptr().add(((y_int + y as i32) * stride + (x_int - 3)) as usize)
                };
                luma_hfir_row(base, f, shift1, &mut pred[y * n_pb_w..(y + 1) * n_pb_w]);
            }
        } else if x_frac == 0 {
            let f = LUMA_FILTER[y_frac as usize];
            let zero = unsafe { _mm_setzero_si128() };
            let mut bevens = [zero; 8];
            for k in 0..8usize {
                bevens[k] = unsafe { _mm_setr_epi16(f[k], 0, f[k], 0, 0, 0, 0, 0) };
            }
            for y in 0..n_pb_h {
                let row_out = &mut pred[y * n_pb_w..(y + 1) * n_pb_w];
                let mut x0 = 0usize;
                while x0 + 4 <= n_pb_w {
                    let (mut even_acc, mut odd_acc) = (zero, zero);
                    for (k, &bev) in bevens.iter().enumerate() {
                        unsafe {
                            let base = plane.as_ptr().add(
                                ((y_int + y as i32 + k as i32 - 3) * stride + x_int + x0 as i32)
                                    as usize,
                            );
                            let v = _mm_loadl_epi64(base as *const __m128i);
                            even_acc = _mm_add_epi32(even_acc, _mm_madd_epi16(v, bev));
                            odd_acc = _mm_add_epi32(
                                odd_acc,
                                _mm_madd_epi16(_mm_srli_si128::<2>(v), bev),
                            );
                        }
                    }
                    store4(
                        unsafe { _mm_unpacklo_epi32(even_acc, odd_acc) },
                        shift1,
                        &mut row_out[x0..x0 + 4],
                    );
                    x0 += 4;
                }
            }
        } else {
            let tmp_h = n_pb_h + 7;
            debug_assert!(n_pb_w <= 64 && tmp_h <= 71);
            let mut tmp = [0i16; 64 * 71];
            let f_h = LUMA_FILTER[x_frac as usize];
            for y in 0..tmp_h {
                let base = unsafe {
                    plane.as_ptr().add(((y_int + y as i32 - 3) * stride + (x_int - 3)) as usize)
                };
                luma_hfir_row(base, f_h, shift1, &mut tmp[y * n_pb_w..(y + 1) * n_pb_w]);
            }
            let f_v = LUMA_FILTER[y_frac as usize];
            luma_vfir(&tmp, n_pb_w, n_pb_h, f_v, shift2, pred);
        }
    }

    /// Chroma 4-tap horizontal FIR for one row. `base` must point at a row
    /// with at least `out.len() + 3` valid u16 samples (interior guarantee).
    #[inline]
    fn chroma_hfir_row(base: *const u16, f: [i16; 4], shift: i32, out: &mut [i16]) {
        let c = unsafe { _mm_setr_epi16(f[0], f[1], f[2], f[3], 0, 0, 0, 0) };
        let mut x = 0usize;
        while x + 4 <= out.len() {
            for j in 0..4usize {
                let v = unsafe {
                    let s = _mm_loadl_epi64(base.add(x + j) as *const __m128i);
                    _mm_cvtsi128_si32(hsum4(_mm_madd_epi16(s, c)))
                };
                out[x + j] = (v >> shift) as i16;
            }
            x += 4;
        }
        while x < out.len() {
            let mut sum = 0i32;
            for (k, &tap) in f.iter().enumerate() {
                sum += tap as i32 * unsafe { *base.add(x + k) } as i32;
            }
            out[x] = (sum >> shift) as i16;
            x += 1;
        }
    }

    /// Chroma 4-tap vertical FIR over `tmp` (width `n`, multiple of 4, at
    /// least `rows_out + 3` rows).
    #[inline]
    fn chroma_vfir(
        tmp: &[i16],
        n: usize,
        rows_out: usize,
        f: [i16; 4],
        shift: i32,
        out: &mut [i16],
    ) {
        let zero = unsafe { _mm_setzero_si128() };
        let mut bevens = [zero; 4];
        for k in 0..4usize {
            bevens[k] = unsafe { _mm_setr_epi16(f[k], 0, f[k], 0, 0, 0, 0, 0) };
        }
        let mut y = 0usize;
        while y < rows_out {
            let mut x0 = 0usize;
            while x0 + 4 <= n {
                let (mut even_acc, mut odd_acc) = (zero, zero);
                for (k, &bev) in bevens.iter().enumerate() {
                    unsafe {
                        let base = tmp.as_ptr().add((y + k) * n + x0);
                        let v = _mm_loadl_epi64(base as *const __m128i); // [T0 T1 T2 T3 | 0 ..]
                        even_acc = _mm_add_epi32(even_acc, _mm_madd_epi16(v, bev));
                        odd_acc = _mm_add_epi32(
                            odd_acc,
                            _mm_madd_epi16(_mm_srli_si128::<2>(v), bev),
                        );
                    }
                }
                store4(
                    unsafe { _mm_unpacklo_epi32(even_acc, odd_acc) },
                    shift,
                    &mut out[y * n + x0..y * n + x0 + 4],
                );
                x0 += 4;
            }
            y += 1;
        }
    }

    /// Chroma interior kernel — all four frac combinations, direct access.
    /// Precondition: the interior bounds checked by `interpolate_chroma` hold
    /// and `n_pb_wc % 4 == 0`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn chroma_interior(
        plane: &[u16],
        stride: i32,
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
        debug_assert_eq!(n_pb_wc % 4, 0);
        if x_frac == 0 && y_frac == 0 {
            for y in 0..n_pb_hc {
                let base = unsafe {
                    plane.as_ptr().add(((y_int + y as i32) * stride + x_int) as usize)
                };
                copy_shifted(base, n_pb_wc, shift3, &mut pred[y * n_pb_wc..(y + 1) * n_pb_wc]);
            }
        } else if y_frac == 0 {
            let f = CHROMA_FILTER[x_frac as usize];
            for y in 0..n_pb_hc {
                let base = unsafe {
                    plane.as_ptr().add(((y_int + y as i32) * stride + (x_int - 1)) as usize)
                };
                chroma_hfir_row(base, f, shift1, &mut pred[y * n_pb_wc..(y + 1) * n_pb_wc]);
            }
        } else if x_frac == 0 {
            let f = CHROMA_FILTER[y_frac as usize];
            let zero = unsafe { _mm_setzero_si128() };
            let mut bevens = [zero; 4];
            for k in 0..4usize {
                bevens[k] = unsafe { _mm_setr_epi16(f[k], 0, f[k], 0, 0, 0, 0, 0) };
            }
            for y in 0..n_pb_hc {
                let row_out = &mut pred[y * n_pb_wc..(y + 1) * n_pb_wc];
                let mut x0 = 0usize;
                while x0 + 4 <= n_pb_wc {
                    let (mut even_acc, mut odd_acc) = (zero, zero);
                    for (k, &bev) in bevens.iter().enumerate() {
                        unsafe {
                            let base = plane.as_ptr().add(
                                ((y_int + y as i32 + k as i32 - 1) * stride + x_int + x0 as i32)
                                    as usize,
                            );
                            let v = _mm_loadl_epi64(base as *const __m128i);
                            even_acc = _mm_add_epi32(even_acc, _mm_madd_epi16(v, bev));
                            odd_acc = _mm_add_epi32(
                                odd_acc,
                                _mm_madd_epi16(_mm_srli_si128::<2>(v), bev),
                            );
                        }
                    }
                    store4(
                        unsafe { _mm_unpacklo_epi32(even_acc, odd_acc) },
                        shift1,
                        &mut row_out[x0..x0 + 4],
                    );
                    x0 += 4;
                }
            }
        } else {
            let tmp_h = n_pb_hc + 3;
            debug_assert!(n_pb_wc <= 32 && tmp_h <= 35);
            let mut tmp = [0i16; 32 * 35];
            let f_h = CHROMA_FILTER[x_frac as usize];
            for y in 0..tmp_h {
                let base = unsafe {
                    plane.as_ptr().add(((y_int + y as i32 - 1) * stride + (x_int - 1)) as usize)
                };
                chroma_hfir_row(base, f_h, shift1, &mut tmp[y * n_pb_wc..(y + 1) * n_pb_wc]);
            }
            let f_v = CHROMA_FILTER[y_frac as usize];
            chroma_vfir(&tmp, n_pb_wc, n_pb_hc, f_v, shift2, pred);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::goldens;

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

    fn compute_luma_interp() -> Vec<(String, Vec<u8>)> {
        let mut rng = Rng::new(0x5EED_0001);
        let pic_w = 64i32;
        let pic_h = 64i32;
        let stride = pic_w;
        let plane: Vec<u16> = (0..pic_w * pic_h).map(|_| rng.below(1024) as u16).collect();
        let mut buf = Vec::new();

        for &(n_pb_w, n_pb_h) in &[(4usize, 4), (8, 16), (16, 8), (32, 32), (64, 64)] {
            for bit_depth in [8i32, 10] {
                for _iter in 0..40 {
                    // xInt/yInt span negative (clamped), interior, and edge positions.
                    let x_int = rng.below(pic_w as u64 + 16) as i32 - 8;
                    let y_int = rng.below(pic_h as u64 + 16) as i32 - 8;
                    let x_frac = rng.below(4) as i32;
                    let y_frac = rng.below(4) as i32;

                    let n_samples = n_pb_w * n_pb_h;
                    let mut pred_rs = vec![0i16; n_samples];

                    interpolate_luma(
                        &plane, pic_w, pic_h, stride, x_int, y_int, x_frac, y_frac,
                        n_pb_w, n_pb_h, bit_depth, &mut pred_rs,
                    );
                    for v in &pred_rs {
                        goldens::push_i16(&mut buf, *v);
                    }
                }
            }
        }
        vec![("interp::luma".to_string(), buf)]
    }

    #[test]
    fn luma_interpolation_matches_golden() {
        for (key, data) in compute_luma_interp() {
            goldens::assert_golden(&key, &data);
        }
    }

    fn compute_chroma_interp() -> Vec<(String, Vec<u8>)> {
        let mut rng = Rng::new(0x5EED_0002);
        let pic_w = 32i32;
        let pic_h = 16i32;
        let stride = pic_w;
        let plane: Vec<u16> = (0..pic_w * pic_h).map(|_| rng.below(1024) as u16).collect();
        let mut buf = Vec::new();

        for &(n_pb_wc, n_pb_hc) in &[(4usize, 4), (8, 8), (16, 16), (32, 16)] {
            for bit_depth in [8i32, 10] {
                for _iter in 0..40 {
                    let x_int = rng.below(pic_w as u64 + 8) as i32 - 4;
                    let y_int = rng.below(pic_h as u64 + 8) as i32 - 4;
                    let x_frac = rng.below(8) as i32;
                    let y_frac = rng.below(8) as i32;

                    let n_samples = n_pb_wc * n_pb_hc;
                    let mut pred_rs = vec![0i16; n_samples];

                    interpolate_chroma(
                        &plane, 1, pic_w, pic_h, stride, x_int, y_int, x_frac, y_frac,
                        n_pb_wc, n_pb_hc, bit_depth, &mut pred_rs,
                    );
                    for v in &pred_rs {
                        goldens::push_i16(&mut buf, *v);
                    }
                }
            }
        }
        vec![("interp::chroma".to_string(), buf)]
    }

    #[test]
    fn chroma_interpolation_matches_golden() {
        for (key, data) in compute_chroma_interp() {
            goldens::assert_golden(&key, &data);
        }
    }

    /// Random weight table (same RNG draw order as the original oracle test).
    fn random_pwt(rng: &mut Rng, log2_denom: u32) -> PredWeightTable {
        let mut pwt = PredWeightTable {
            luma_log2_weight_denom: log2_denom,
            ..Default::default()
        };
        for list in 0..2 {
            let dst = if list == 0 { &mut pwt.l0 } else { &mut pwt.l1 };
            for entry in dst.iter_mut() {
                entry.luma_weight = (rng.below(65) as i32 - 32) as i16;
                entry.luma_offset = (rng.below(41) as i32 - 20) as i16;
                for c in 0..2 {
                    entry.chroma_weight[c] = (rng.below(65) as i32 - 32) as i16;
                    entry.chroma_offset[c] = (rng.below(41) as i32 - 20) as i16;
                }
            }
        }
        pwt
    }

    fn compute_weighted_pred_default() -> Vec<(String, Vec<u8>)> {
        let mut rng = Rng::new(0x5EED_0003);
        let mut buf = Vec::new();
        for bit_depth in [8i32, 10] {
            for &(flag_l0, flag_l1) in &[(true, false), (false, true), (true, true)] {
                for _iter in 0..20 {
                    let n_samples = 16;
                    let pred_l0: Vec<i16> = (0..n_samples).map(|_| (rng.below(4097) as i32 - 2048) as i16).collect();
                    let pred_l1: Vec<i16> = (0..n_samples).map(|_| (rng.below(4097) as i32 - 2048) as i16).collect();

                    let mut out_rs = vec![0i16; n_samples];

                    weighted_pred_default(
                        &pred_l0, &pred_l1, flag_l0, flag_l1, n_samples as i32, bit_depth,
                        &mut out_rs,
                    );
                    for v in &out_rs {
                        goldens::push_i16(&mut buf, *v);
                    }
                }
            }
        }
        vec![("interp::weighted_pred_default".to_string(), buf)]
    }

    #[test]
    fn weighted_pred_default_matches_golden() {
        for (key, data) in compute_weighted_pred_default() {
            goldens::assert_golden(&key, &data);
        }
    }

    fn compute_weighted_pred_explicit() -> Vec<(String, Vec<u8>)> {
        let mut rng = Rng::new(0x5EED_0004);
        let mut buf = Vec::new();
        for bit_depth in [8i32, 10] {
            for log2_denom in 0..3u32 {
                for c_idx in 0..3i32 {
                    for &(flag_l0, flag_l1) in &[(true, false), (false, true), (true, true)] {
                        let pwt = random_pwt(&mut rng, log2_denom);
                        let ref_idx_l0 = if rng.below(4) == 0 { -1 } else { rng.below(2) as i32 };
                        let ref_idx_l1 = if rng.below(4) == 0 { -1 } else { rng.below(2) as i32 };

                        let n_samples = 16;
                        let pred_l0: Vec<i16> = (0..n_samples).map(|_| (rng.below(4097) as i32 - 2048) as i16).collect();
                        let pred_l1: Vec<i16> = (0..n_samples).map(|_| (rng.below(4097) as i32 - 2048) as i16).collect();

                        let mut out_rs = vec![0i16; n_samples];

                        weighted_pred_explicit(
                            &pred_l0, &pred_l1, flag_l0, flag_l1, ref_idx_l0, ref_idx_l1,
                            c_idx, n_samples as i32, bit_depth, &pwt, &mut out_rs,
                        );
                        for v in &out_rs {
                            goldens::push_i16(&mut buf, *v);
                        }
                    }
                }
            }
        }
        vec![("interp::weighted_pred_explicit".to_string(), buf)]
    }

    #[test]
    fn weighted_pred_explicit_matches_golden() {
        for (key, data) in compute_weighted_pred_explicit() {
            goldens::assert_golden(&key, &data);
        }
    }

    pub(crate) fn golden_entries() -> Vec<(String, String)> {
        let mut v = Vec::new();
        for (k, b) in compute_luma_interp() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        for (k, b) in compute_chroma_interp() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        for (k, b) in compute_weighted_pred_default() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        for (k, b) in compute_weighted_pred_explicit() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        v
    }
}

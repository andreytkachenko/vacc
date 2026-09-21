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
/// AVX2 availability (CPUID leaf 7, EBX bit 5), cached by std after first use.
#[cfg(target_arch = "x86_64")]
fn detect_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_avx2() -> bool {
    false
}

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
    // MC scratch; must hold at least FIR_SCRATCH_MAX samples.
    fir_tmp: &mut [i16],
) {
    // §8.5.3.3.3: shift1 = Min(4, BitDepthY - 8), shift2 = 6, shift3 = Max(2, 14 - BitDepthY)
    let shift1 = 4.min(bit_depth - 8);
    let shift2 = 6;
    let shift3 = 2.max(14 - bit_depth);

    // Check if all reference accesses are within bounds (including filter
    // margin). The right margin is 5, not 4: the AVX2 horizontal FIR reads
    // one vector past its last needed sample, and PUs short of that take the
    // clamped-window edge path instead.
    let interior = x_int - 3 >= 0
        && y_int - 3 >= 0
        && x_int + n_pb_w as i32 + 5 <= pic_w
        && y_int + n_pb_h as i32 + 4 <= pic_h;

    if interior {
        // Fast direct access (no bounds check) — SIMD FIR on x86_64 when the
        // block width vectorizes, scalar core otherwise.
        #[cfg(target_arch = "x86_64")]
        if n_pb_w.is_multiple_of(4) {
            // Interior bounds were checked above. AVX2 when available, SSE2
            // (x86_64 baseline) otherwise. Detection is cached in std.
            if detect_avx2() {
                unsafe {
                    avx2::luma_interior(
                        plane, stride, x_int, y_int, x_frac, y_frac, n_pb_w, n_pb_h, shift1,
                        shift2, shift3, pred, fir_tmp,
                    );
                }
            } else {
                sse2::luma_interior(
                    plane, stride, x_int, y_int, x_frac, y_frac, n_pb_w, n_pb_h, shift1, shift2,
                    shift3, pred, fir_tmp,
                );
            }
        } else {
            luma_interior_scalar(
                plane, stride, x_int, y_int, x_frac, y_frac, n_pb_w, n_pb_h, shift1, shift2,
                shift3, pred, fir_tmp,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        luma_interior_scalar(
            plane, stride, x_int, y_int, x_frac, y_frac, n_pb_w, n_pb_h, shift1, shift2, shift3,
            pred, fir_tmp,
        );
    } else {
        // Edge PU: the filter window crosses a picture border.
        #[cfg(target_arch = "x86_64")]
        {
            luma_edge(
                plane, pic_w, pic_h, stride, x_int, y_int, x_frac, y_frac, n_pb_w, n_pb_h, shift1,
                shift2, shift3, pred, fir_tmp,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Safe clamped access — used for edge PUs
            let ref_fn = |x: i32, y: i32| {
                let x = x.clamp(0, pic_w - 1);
                let y = y.clamp(0, pic_h - 1);
                plane[(y * stride + x) as usize] as i32
            };
            luma_interp_core(
                ref_fn, x_int, y_int, x_frac, y_frac, n_pb_w, n_pb_h, shift1, shift2, shift3,
                pred, fir_tmp,
            );
        }
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
    tmp: &mut [i16],
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
        // Pass 1 below writes every row/col pass 2 reads, so the buffer needs
        // no initialization.
        let tmp = &mut tmp[..n_pb_w * tmp_h];
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
    fir_tmp: &mut [i16],
) {
    let ref_fn = |x: i32, y: i32| plane[(y * stride + x) as usize] as i32;
    luma_interp_core(
        ref_fn, x_int, y_int, x_frac, y_frac, n_pb_w, n_pb_h, shift1, shift2, shift3, pred,
        fir_tmp,
    );
}

/// Max MC scratch: luma 2D FIR intermediate (64x71) plus one clamped
/// edge-window row (71).
pub const FIR_SCRATCH_MAX: usize = 64 * 71 + 71;

/// Bit-cast a buffer of non-negative i16 reference samples to u16. Samples
/// are < 2^15 for bit depth <= 12, so the bit pattern is preserved.
#[cfg(target_arch = "x86_64")]
#[inline]
fn as_u16(buf: &[i16]) -> &[u16] {
    unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u16, buf.len()) }
}

/// Horizontal FIR row with AVX2/SSE2 dispatch — the edge paths run on any
/// x86_64 (AVX2 detection is cached in std).
#[cfg(target_arch = "x86_64")]
#[inline]
fn hfir_row_luma(row: &[u16], f: [i16; 8], shift: i32, out: &mut [i16]) {
    if detect_avx2() {
        unsafe { avx2::luma_hfir_row(row, f, shift, out) }
    } else {
        sse2::luma_hfir_row(row, f, shift, out)
    }
}

/// Chroma variant of [`hfir_row_luma`].
#[cfg(target_arch = "x86_64")]
#[inline]
fn hfir_row_chroma(row: &[u16], f: [i16; 4], shift: i32, out: &mut [i16]) {
    if detect_avx2() {
        unsafe { avx2::chroma_hfir_row(row, f, shift, out) }
    } else {
        sse2::chroma_hfir_row(row, f, shift, out)
    }
}

/// Fill `dst` with the reference row starting at column `x_start`,
/// replicating edge samples for out-of-range columns (spec §8.5.3.3.3).
/// The in-range middle is a straight copy; only the ends are per-sample.
#[cfg(target_arch = "x86_64")]
#[inline]
fn clamped_row(dst: &mut [i16], src: &[u16], x_start: i32, pic_w: i32) {
    let w = dst.len() as i32;
    let lo = (-x_start).max(0).min(w);
    let hi = (pic_w - x_start).max(0).min(w);
    if lo > 0 {
        let v = src[0] as i16;
        for d in &mut dst[..lo as usize] {
            *d = v;
        }
    }
    if lo < hi {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr().add((x_start + lo) as usize) as *const i16,
                dst.as_mut_ptr().add(lo as usize),
                (hi - lo) as usize,
            );
        }
    }
    if hi < w {
        let v = src[(pic_w - 1) as usize] as i16;
        for d in &mut dst[hi as usize..] {
            *d = v;
        }
    }
}

/// Edge PU: the filter window crosses a picture border, so direct access is
/// impossible. Materialize the clamped reference window into `fir_tmp` (one
/// edge-replicating copy per sample) and run the interior SIMD kernels on
/// it — far cheaper than the scalar core's per-tap clamp + bounds-checked
/// index.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn luma_edge(
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
    shift1: i32,
    shift2: i32,
    shift3: i32,
    pred: &mut [i16],
    fir_tmp: &mut [i16],
) {
    if x_frac != 0 && y_frac != 0 {
        // Two-pass: materialize one window row (n_pb_w + 7 samples) at a
        // time, horizontal FIR into `tmp`, then vertical FIR. Fits in
        // FIR_SCRATCH_MAX = 71 + 64*71.
        let win_w = n_pb_w + 7;
        let tmp_h = n_pb_h + 7;
        debug_assert!(n_pb_w <= 64 && tmp_h <= 71);
        let (row_buf, rest) = fir_tmp.split_at_mut(win_w);
        let tmp = &mut rest[..n_pb_w * tmp_h];
        let f_h = LUMA_FILTER[x_frac as usize];
        for wy in 0..tmp_h {
            let src_y = (y_int - 3 + wy as i32).clamp(0, pic_h - 1);
            clamped_row(row_buf, &plane[(src_y * stride) as usize..], x_int - 3, pic_w);
            hfir_row_luma(
                as_u16(row_buf),
                f_h,
                shift1,
                &mut tmp[wy * n_pb_w..(wy + 1) * n_pb_w],
            );
        }
        sse2::luma_vfir(tmp, n_pb_w, n_pb_h, LUMA_FILTER[y_frac as usize], shift2, pred);
    } else if y_frac == 0 {
        if x_frac == 0 {
            // Integer MV: clamped copy with << shift3.
            for wy in 0..n_pb_h {
                let src_y = (y_int + wy as i32).clamp(0, pic_h - 1);
                let row = &mut pred[wy * n_pb_w..(wy + 1) * n_pb_w];
                clamped_row(row, &plane[(src_y * stride) as usize..], x_int, pic_w);
                for s in row.iter_mut() {
                    *s = (*s << shift3) as i16;
                }
            }
        } else {
            // Horizontal only: per-row window + 8-tap FIR.
            let win_w = n_pb_w + 7;
            let f = LUMA_FILTER[x_frac as usize];
            for wy in 0..n_pb_h {
                let src_y = (y_int + wy as i32).clamp(0, pic_h - 1);
                let row = &mut fir_tmp[wy * win_w..(wy + 1) * win_w];
                clamped_row(row, &plane[(src_y * stride) as usize..], x_int - 3, pic_w);
                hfir_row_luma(
                    as_u16(row),
                    f,
                    shift1,
                    &mut pred[wy * n_pb_w..(wy + 1) * n_pb_w],
                );
            }
        }
    } else {
        // x_frac == 0, y_frac != 0: vertical only. Materialize the whole
        // w x (h+7) window, then vertical FIR.
        let win_h = n_pb_h + 7;
        for wy in 0..win_h {
            let src_y = (y_int - 3 + wy as i32).clamp(0, pic_h - 1);
            clamped_row(
                &mut fir_tmp[wy * n_pb_w..(wy + 1) * n_pb_w],
                &plane[(src_y * stride) as usize..],
                x_int,
                pic_w,
            );
        }
        sse2::luma_vfir(
            &fir_tmp[..n_pb_w * win_h],
            n_pb_w,
            n_pb_h,
            LUMA_FILTER[y_frac as usize],
            shift1,
            pred,
        );
    }
}

/// Edge PU for chroma (4-tap filter, margins 1/2) — see `luma_edge`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn chroma_edge(
    plane: &[u16],
    pic_w: i32,
    pic_h: i32,
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
    fir_tmp: &mut [i16],
) {
    if x_frac != 0 && y_frac != 0 {
        let win_w = n_pb_wc + 3;
        let tmp_h = n_pb_hc + 3;
        debug_assert!(n_pb_wc <= 32 && tmp_h <= 35);
        let (row_buf, rest) = fir_tmp.split_at_mut(win_w);
        let tmp = &mut rest[..n_pb_wc * tmp_h];
        let f_h = CHROMA_FILTER[x_frac as usize];
        for wy in 0..tmp_h {
            let src_y = (y_int - 1 + wy as i32).clamp(0, pic_h - 1);
            clamped_row(row_buf, &plane[(src_y * stride) as usize..], x_int - 1, pic_w);
            hfir_row_chroma(
                as_u16(row_buf),
                f_h,
                shift1,
                &mut tmp[wy * n_pb_wc..(wy + 1) * n_pb_wc],
            );
        }
        sse2::chroma_vfir(tmp, n_pb_wc, n_pb_hc, CHROMA_FILTER[y_frac as usize], shift2, pred);
    } else if y_frac == 0 {
        if x_frac == 0 {
            for wy in 0..n_pb_hc {
                let src_y = (y_int + wy as i32).clamp(0, pic_h - 1);
                let row = &mut pred[wy * n_pb_wc..(wy + 1) * n_pb_wc];
                clamped_row(row, &plane[(src_y * stride) as usize..], x_int, pic_w);
                for s in row.iter_mut() {
                    *s = (*s << shift3) as i16;
                }
            }
        } else {
            let win_w = n_pb_wc + 3;
            let f = CHROMA_FILTER[x_frac as usize];
            for wy in 0..n_pb_hc {
                let src_y = (y_int + wy as i32).clamp(0, pic_h - 1);
                let row = &mut fir_tmp[wy * win_w..(wy + 1) * win_w];
                clamped_row(row, &plane[(src_y * stride) as usize..], x_int - 1, pic_w);
                hfir_row_chroma(
                    as_u16(row),
                    f,
                    shift1,
                    &mut pred[wy * n_pb_wc..(wy + 1) * n_pb_wc],
                );
            }
        }
    } else {
        let win_h = n_pb_hc + 3;
        for wy in 0..win_h {
            let src_y = (y_int - 1 + wy as i32).clamp(0, pic_h - 1);
            clamped_row(
                &mut fir_tmp[wy * n_pb_wc..(wy + 1) * n_pb_wc],
                &plane[(src_y * stride) as usize..],
                x_int,
                pic_w,
            );
        }
        sse2::chroma_vfir(
            &fir_tmp[..n_pb_wc * win_h],
            n_pb_wc,
            n_pb_hc,
            CHROMA_FILTER[y_frac as usize],
            shift1,
            pred,
        );
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
    // MC scratch; must hold at least FIR_SCRATCH_MAX samples.
    fir_tmp: &mut [i16],
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
            // Interior bounds were checked above. AVX2 when available, SSE2
            // (x86_64 baseline) otherwise. Detection is cached in std.
            if detect_avx2() {
                unsafe {
                    avx2::chroma_interior(
                        plane, stride, x_int, y_int, x_frac, y_frac, n_pb_wc, n_pb_hc, shift1,
                        shift2, shift3, pred, fir_tmp,
                    );
                }
            } else {
                sse2::chroma_interior(
                    plane, stride, x_int, y_int, x_frac, y_frac, n_pb_wc, n_pb_hc, shift1,
                    shift2, shift3, pred, fir_tmp,
                );
            }
        } else {
            chroma_interior_scalar(
                plane, stride, x_int, y_int, x_frac, y_frac, n_pb_wc, n_pb_hc, shift1, shift2,
                shift3, pred, fir_tmp,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        chroma_interior_scalar(
            plane, stride, x_int, y_int, x_frac, y_frac, n_pb_wc, n_pb_hc, shift1, shift2,
            shift3, pred, fir_tmp,
        );
    } else {
        // Edge PU: the filter window crosses a picture border.
        #[cfg(target_arch = "x86_64")]
        {
            chroma_edge(
                plane, pic_w, pic_h, stride, x_int, y_int, x_frac, y_frac, n_pb_wc, n_pb_hc,
                shift1, shift2, shift3, pred, fir_tmp,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
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
                fir_tmp,
            );
        }
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
    tmp: &mut [i16],
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
        // Pass 1 below writes every row/col pass 2 reads — no init needed.
        let tmp = &mut tmp[..n_pb_wc * tmp_h];
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
    fir_tmp: &mut [i16],
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
        fir_tmp,
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

    /// Luma 8-tap horizontal FIR for one row. `row` must hold at least
    /// `out.len() + 7` valid u16 samples (interior guarantee).
    #[inline]
    pub(super) fn luma_hfir_row(row: &[u16], f: [i16; 8], shift: i32, out: &mut [i16]) {
        let c = unsafe { _mm_setr_epi16(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]) };
        let base = row.as_ptr();
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
                sum += tap as i32 * row[x + k] as i32;
            }
            out[x] = (sum >> shift) as i16;
            x += 1;
        }
    }

    /// Luma 8-tap vertical FIR over `tmp` (width `n`, at least `rows_out + 7`
    /// rows). Signed i16 inputs via the madd pairing; scalar tail for widths
    /// that are not a multiple of 4.
    #[inline]
    pub(super) fn luma_vfir(
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
            while x0 < n {
                let mut sum = 0i32;
                for (k, &tap) in f.iter().enumerate() {
                    sum += tap as i32 * tmp[(y + k) * n + x0] as i32;
                }
                out[y * n + x0] = (sum >> shift) as i16;
                x0 += 1;
            }
            y += 1;
        }
    }

    /// Integer-MV copy with `<< shift3`, 4 samples at a time. `row` must
    /// hold at least `out.len()` valid u16 samples.
    #[inline]
    fn copy_shifted(row: &[u16], shift: i32, out: &mut [i16]) {
        let base = row.as_ptr();
        let n = out.len();
        let mut x = 0usize;
        while x + 4 <= n {
            unsafe {
                let v = _mm_loadl_epi64(base.add(x) as *const __m128i);
                store4(slli32(zext4(v), shift), 0, &mut out[x..x + 4]);
            }
            x += 4;
        }
        while x < n {
            let s = row[x];
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
        fir_tmp: &mut [i16],
    ) {
        debug_assert_eq!(n_pb_w % 4, 0);
        if x_frac == 0 && y_frac == 0 {
            for y in 0..n_pb_h {
                let off = ((y_int + y as i32) * stride + x_int) as usize;
                copy_shifted(&plane[off..off + n_pb_w], shift3, &mut pred[y * n_pb_w..(y + 1) * n_pb_w]);
            }
        } else if y_frac == 0 {
            let f = LUMA_FILTER[x_frac as usize];
            for y in 0..n_pb_h {
                let off = ((y_int + y as i32) * stride + (x_int - 3)) as usize;
                luma_hfir_row(
                    &plane[off..off + n_pb_w + 8],
                    f,
                    shift1,
                    &mut pred[y * n_pb_w..(y + 1) * n_pb_w],
                );
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
                        let off =
                            ((y_int + y as i32 + k as i32 - 3) * stride + x_int + x0 as i32)
                                as usize;
                        let row = &plane[off..off + 4];
                        unsafe {
                            let v = _mm_loadl_epi64(row.as_ptr() as *const __m128i);
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
            // Pass 1 writes every row/col pass 2 reads — no init needed.
            let tmp = &mut fir_tmp[..n_pb_w * tmp_h];
            let f_h = LUMA_FILTER[x_frac as usize];
            for y in 0..tmp_h {
                let off = ((y_int + y as i32 - 3) * stride + (x_int - 3)) as usize;
                luma_hfir_row(
                    &plane[off..off + n_pb_w + 8],
                    f_h,
                    shift1,
                    &mut tmp[y * n_pb_w..(y + 1) * n_pb_w],
                );
            }
            let f_v = LUMA_FILTER[y_frac as usize];
            luma_vfir(&tmp, n_pb_w, n_pb_h, f_v, shift2, pred);
        }
    }

    /// Chroma 4-tap horizontal FIR for one row. `row` must hold at least
    /// `out.len() + 3` valid u16 samples (interior guarantee).
    #[inline]
    pub(super) fn chroma_hfir_row(row: &[u16], f: [i16; 4], shift: i32, out: &mut [i16]) {
        let c = unsafe { _mm_setr_epi16(f[0], f[1], f[2], f[3], 0, 0, 0, 0) };
        let base = row.as_ptr();
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
                sum += tap as i32 * row[x + k] as i32;
            }
            out[x] = (sum >> shift) as i16;
            x += 1;
        }
    }

    /// Chroma 4-tap vertical FIR over `tmp` (width `n`, at least
    /// `rows_out + 3` rows); scalar tail for widths not a multiple of 4.
    #[inline]
    pub(super) fn chroma_vfir(
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
            while x0 < n {
                let mut sum = 0i32;
                for (k, &tap) in f.iter().enumerate() {
                    sum += tap as i32 * tmp[(y + k) * n + x0] as i32;
                }
                out[y * n + x0] = (sum >> shift) as i16;
                x0 += 1;
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
        fir_tmp: &mut [i16],
    ) {
        debug_assert_eq!(n_pb_wc % 4, 0);
        if x_frac == 0 && y_frac == 0 {
            for y in 0..n_pb_hc {
                let off = ((y_int + y as i32) * stride + x_int) as usize;
                copy_shifted(
                    &plane[off..off + n_pb_wc],
                    shift3,
                    &mut pred[y * n_pb_wc..(y + 1) * n_pb_wc],
                );
            }
        } else if y_frac == 0 {
            let f = CHROMA_FILTER[x_frac as usize];
            for y in 0..n_pb_hc {
                let off = ((y_int + y as i32) * stride + (x_int - 1)) as usize;
                chroma_hfir_row(
                    &plane[off..off + n_pb_wc + 3],
                    f,
                    shift1,
                    &mut pred[y * n_pb_wc..(y + 1) * n_pb_wc],
                );
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
                        let off =
                            ((y_int + y as i32 + k as i32 - 1) * stride + x_int + x0 as i32)
                                as usize;
                        let row = &plane[off..off + 4];
                        unsafe {
                            let v = _mm_loadl_epi64(row.as_ptr() as *const __m128i);
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
            // Pass 1 writes every row/col pass 2 reads — no init needed.
            let tmp = &mut fir_tmp[..n_pb_wc * tmp_h];
            let f_h = CHROMA_FILTER[x_frac as usize];
            for y in 0..tmp_h {
                let off = ((y_int + y as i32 - 1) * stride + (x_int - 1)) as usize;
                chroma_hfir_row(
                    &plane[off..off + n_pb_wc + 3],
                    f_h,
                    shift1,
                    &mut tmp[y * n_pb_wc..(y + 1) * n_pb_wc],
                );
            }
            let f_v = CHROMA_FILTER[y_frac as usize];
            chroma_vfir(&tmp, n_pb_wc, n_pb_hc, f_v, shift2, pred);
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 interior kernels (x86_64 with AVX2).
//
// Byte-exact with the scalar cores: identical i32 arithmetic, and the final
// `_mm256_packs_epi32(r, r)` truncates each i32 lane to its low 16 bits —
// the same wrap semantics as `(sum >> shift) as i16`.
//
// Vertical FIR: one 16-sample load per tap row plus two madds with the
// `[f,0]` / `[0,f]` patterns yields the even- and odd-column products in the
// same register — no sign extension, no element shifts. A 16-wide chunk is
// four unpacks + one pack/store.
//
// Integer-pel copy: zero-extend 8 u16 samples to i32, shift, pack — exact
// for any input (no saturation).
//
// Horizontal FIR: each output splits its taps into even/odd pairs (2j,
// 2j+1); a `madd` of the sample window starting at offset 2j with the
// broadcast `[f2j, f2j+1]` yields both products for one output in a single
// i32 lane. Even outputs use windows aligned to the chunk base, odd outputs
// the same windows shifted by one sample (`alignr`) — every lane is a real
// product, so no zero-fill correction. Windows are built from 128-bit
// `alignr` (the 256-bit form shifts each half independently); an 8-output
// pass covers `[R0..R15]`, and a 16-wide chunk is two such passes. Chunks
// that would read past `row` fall through to smaller chunks, then to the
// SSE2 row kernel for the tail.
//
// Callers route here only when the block width is a multiple of 4; anything
// else uses the scalar core.

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi32, _mm256_alignr_epi8, _mm256_and_si256,
        _mm256_castsi128_si256, _mm256_castsi256_si128, _mm256_extracti128_si256,
        _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_si256, _mm256_srai_epi32,
        _mm_add_epi32, _mm_and_si128, _mm_loadu_si128, _mm_madd_epi16, _mm_or_si128,
        _mm_packus_epi32, _mm_set1_epi32, _mm_setr_epi16, _mm_setr_epi32, _mm_setzero_si128,
        _mm_slli_epi16, _mm_slli_si128, _mm_srai_epi32, _mm_srli_si128, _mm_storeu_si128,
        _mm_unpackhi_epi32, _mm_unpacklo_epi32,
    };

    use super::sse2;
    use super::{CHROMA_FILTER, LUMA_FILTER};

    /// Logical left-shift 8 i16 lanes by `n` (always 2..=6 in this module).
    #[inline(always)]
    fn slli16(v: __m128i, n: i32) -> __m128i {
        match n {
            0 => v,
            1 => unsafe { _mm_slli_epi16::<1>(v) },
            2 => unsafe { _mm_slli_epi16::<2>(v) },
            3 => unsafe { _mm_slli_epi16::<3>(v) },
            4 => unsafe { _mm_slli_epi16::<4>(v) },
            5 => unsafe { _mm_slli_epi16::<5>(v) },
            6 => unsafe { _mm_slli_epi16::<6>(v) },
            _ => {
                // Unreachable for our shift values; exact scalar fallback.
                let mut a = [0i16; 8];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &v as *const __m128i as *const i16,
                        a.as_mut_ptr(),
                        8,
                    );
                }
                for x in a.iter_mut() {
                    *x = *x << n;
                }
                unsafe {
                    _mm_setr_epi16(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
                }
            }
        }
    }

    /// Arithmetic right-shift 8 i32 lanes by `n` (always 0..=6 in this module).
    #[inline]
    #[target_feature(enable = "avx2")]
    fn srai32(v: __m256i, n: i32) -> __m256i {
        match n {
            0 => v,
            1 => unsafe { _mm256_srai_epi32::<1>(v) },
            2 => unsafe { _mm256_srai_epi32::<2>(v) },
            3 => unsafe { _mm256_srai_epi32::<3>(v) },
            4 => unsafe { _mm256_srai_epi32::<4>(v) },
            5 => unsafe { _mm256_srai_epi32::<5>(v) },
            6 => unsafe { _mm256_srai_epi32::<6>(v) },
            _ => {
                // Unreachable for our shift values; exact scalar fallback.
                let mut a = [0i32; 8];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &v as *const __m256i as *const i32,
                        a.as_mut_ptr(),
                        8,
                    );
                }
                for x in a.iter_mut() {
                    *x = *x >> n;
                }
                unsafe {
                    _mm256_setr_epi32(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
                }
            }
        }
    }

    /// Merge even/odd column accumulators into in-order output, shift, and
    /// store 16 i16 samples. `even` lane j holds column `2j`, `odd` lane j
    /// column `2j + 1` (i32, unshifted).
    ///
    /// Packs lay out sequentially per half (`[a0..a3, b0..b3, ...]`), so the
    /// even/odd lanes are interleaved first with unpacklo/hi. Masking to 16
    /// bits replicates the truncating `(sum >> shift) as i16` of the scalar
    /// core — FIR outputs can exceed the i16 range, so saturating packs
    /// would not be exact.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn merge_shift_store16(even: __m256i, odd: __m256i, shift: i32, out: &mut [i16]) {
        unsafe {
            let mask = _mm256_set1_epi32(0xFFFF);
            let e = _mm256_and_si256(srai32(even, shift), mask);
            let o = _mm256_and_si256(srai32(odd, shift), mask);
            let e_lo = _mm256_castsi256_si128(e);
            let o_lo = _mm256_castsi256_si128(o);
            let e_hi = _mm256_extracti128_si256::<1>(e);
            let o_hi = _mm256_extracti128_si256::<1>(o);
            // Columns 0..7: [E0,O0,E1,O1] then [E2,O2,E3,O3]; 8..15 likewise.
            let p0 = _mm_packus_epi32(
                _mm_unpacklo_epi32(e_lo, o_lo),
                _mm_unpackhi_epi32(e_lo, o_lo),
            );
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, p0);
            let p1 = _mm_packus_epi32(
                _mm_unpacklo_epi32(e_hi, o_hi),
                _mm_unpackhi_epi32(e_hi, o_hi),
            );
            _mm_storeu_si128(out.as_mut_ptr().add(8) as *mut __m128i, p1);
        }
    }

    /// Broadcast tap pair `(a, b)` into every i16 lane (low 16 = `a`, high 16
    /// = `b`, per i32 lane) — 256-bit and 128-bit forms.
    #[inline]
    fn pair256(a: i16, b: i16) -> __m256i {
        unsafe { _mm256_set1_epi32((a as i32 & 0xFFFF) | ((b as i32) << 16)) }
    }

    #[inline]
    fn pair128(a: i16, b: i16) -> __m128i {
        unsafe { _mm_set1_epi32((a as i32 & 0xFFFF) | ((b as i32) << 16)) }
    }

    /// Arithmetic right-shift 4 i32 lanes by `n` (always 0..=6 in this module).
    #[inline]
    fn srai32_128(v: __m128i, n: i32) -> __m128i {
        match n {
            0 => v,
            1 => unsafe { _mm_srai_epi32::<1>(v) },
            2 => unsafe { _mm_srai_epi32::<2>(v) },
            3 => unsafe { _mm_srai_epi32::<3>(v) },
            4 => unsafe { _mm_srai_epi32::<4>(v) },
            5 => unsafe { _mm_srai_epi32::<5>(v) },
            6 => unsafe { _mm_srai_epi32::<6>(v) },
            _ => {
                // Unreachable for our shift values; exact scalar fallback.
                let mut a = [0i32; 4];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &v as *const __m128i as *const i32,
                        a.as_mut_ptr(),
                        4,
                    );
                }
                for x in a.iter_mut() {
                    *x = *x >> n;
                }
                unsafe { _mm_setr_epi32(a[0], a[1], a[2], a[3]) }
            }
        }
    }

    /// Merge even/odd column accumulators into in-order output, shift, and
    /// store 8 i16 samples — 128-bit variant of `merge_shift_store16`.
    #[inline]
    fn merge_shift_store8(even: __m128i, odd: __m128i, shift: i32, out: &mut [i16]) {
        unsafe {
            let mask = _mm_set1_epi32(0xFFFF);
            let e = _mm_and_si128(srai32_128(even, shift), mask);
            let o = _mm_and_si128(srai32_128(odd, shift), mask);
            // Columns 0..7: [E0,O0,E1,O1] then [E2,O2,E3,O3].
            let p = _mm_packus_epi32(_mm_unpacklo_epi32(e, o), _mm_unpackhi_epi32(e, o));
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, p);
        }
    }

    /// Vertical FIR over rows at `base + y * row_stride` — width `n`
    /// (multiple of 4, at least `rows_out + TAPS` rows). The i16
    /// reinterpretation of u16 plane samples is exact (samples < 2^15).
    /// 16-wide chunks, scalar tail.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn vfir_core<const TAPS: usize>(
        base: *const i16,
        row_stride: usize,
        n: usize,
        rows_out: usize,
        f: [i16; TAPS],
        shift: i32,
        out: &mut [i16],
    ) {
        let zero = unsafe { _mm256_setzero_si256() };
        let mask16 = unsafe { _mm256_set1_epi32(0xFFFF) };
        let mut y = 0usize;
        while y < rows_out {
            let mut x0 = 0usize;
            while x0 + 16 <= n {
                let (mut even_acc, mut odd_acc) = (zero, zero);
                for k in 0..TAPS {
                    unsafe {
                        let v = _mm256_loadu_si256(
                            base.add((y + k) * row_stride + x0) as *const __m256i,
                        );
                        // The coefficient depends on tap k, not on the column:
                        // broadcast f[k] to every even (resp. odd) i16 slot so
                        // madd pairs each sample with f[k].
                        let fk = f[k] as i32;
                        let b_even = _mm256_and_si256(_mm256_set1_epi32(fk), mask16);
                        let b_odd = _mm256_set1_epi32(fk << 16);
                        even_acc = _mm256_add_epi32(even_acc, _mm256_madd_epi16(v, b_even));
                        odd_acc = _mm256_add_epi32(odd_acc, _mm256_madd_epi16(v, b_odd));
                    }
                }
                merge_shift_store16(even_acc, odd_acc, shift, &mut out[y * n + x0..y * n + x0 + 16]);
                x0 += 16;
            }
            // Tail: n % 16 in {0, 4, 8, 12} — exact scalar FIR.
            while x0 < n {
                let mut sum = 0i32;
                for k in 0..TAPS {
                    sum += f[k] as i32 * unsafe { *base.add((y + k) * row_stride + x0) } as i32;
                }
                out[y * n + x0] = (sum >> shift) as i16;
                x0 += 1;
            }
            y += 1;
        }
    }

    /// Integer-MV copy with `<< shift3`, 8 samples at a time. `row` must hold
    /// at least `out.len()` valid u16 samples.
    #[inline]
    fn copy_shifted(row: &[u16], shift: i32, out: &mut [i16]) {
        let base = row.as_ptr();
        let n = out.len();
        let mut x = 0usize;
        while x + 8 <= n {
            unsafe {
                // `sample << shift3` always fits in i16 (max 16380 at
                // bit depth 12), so the 16-bit lanes can be shifted directly.
                let v = _mm_loadu_si128(base.add(x) as *const __m128i);
                _mm_storeu_si128(
                    out.as_mut_ptr().add(x) as *mut __m128i,
                    slli16(v, shift),
                );
            }
            x += 8;
        }
        while x < n {
            let s = row[x];
            out[x] = ((s as i32) << shift) as i16;
            x += 1;
        }
    }

    /// 8 luma outputs from the 16-sample window held in `[a || b]` (i16
    /// lanes: `a` = R[0..7], `b` = R[8..15]).
    ///
    /// Even output 2i, tap pair j needs R[2i+2j], R[2i+2j+1]; odd output
    /// 2i+1 needs R[2i+2j+1], R[2i+2j+2]. So even pair j uses the window
    /// starting at sample 2j (byte 4j of the 128-bit concat) and odd pair j
    /// the window starting at sample 2j+1 (byte 4j+2). A `madd` of that
    /// window with the broadcast tap pair yields both products for one
    /// output in a single i32 lane.
    #[inline]
    fn hfir8_luma(a: __m128i, b: __m128i, d: [__m128i; 4], shift: i32, out: &mut [i16]) {
        unsafe {
            // Window k = bytes 2k..2k+15 of the 32-byte stream [a || b]:
            // shift `a` right by 2k bytes and `b` left by 16-2k, then or.
            let e1 = _mm_or_si128(_mm_srli_si128::<4>(a), _mm_slli_si128::<12>(b));
            let e2 = _mm_or_si128(_mm_srli_si128::<8>(a), _mm_slli_si128::<8>(b));
            let e3 = _mm_or_si128(_mm_srli_si128::<12>(a), _mm_slli_si128::<4>(b));
            let o0 = _mm_or_si128(_mm_srli_si128::<2>(a), _mm_slli_si128::<14>(b));
            let o1 = _mm_or_si128(_mm_srli_si128::<6>(a), _mm_slli_si128::<10>(b));
            let o2 = _mm_or_si128(_mm_srli_si128::<10>(a), _mm_slli_si128::<6>(b));
            let o3 = _mm_or_si128(_mm_srli_si128::<14>(a), _mm_slli_si128::<2>(b));
            let even_acc = _mm_add_epi32(
                _mm_add_epi32(_mm_madd_epi16(a, d[0]), _mm_madd_epi16(e1, d[1])),
                _mm_add_epi32(_mm_madd_epi16(e2, d[2]), _mm_madd_epi16(e3, d[3])),
            );
            let odd_acc = _mm_add_epi32(
                _mm_add_epi32(_mm_madd_epi16(o0, d[0]), _mm_madd_epi16(o1, d[1])),
                _mm_add_epi32(_mm_madd_epi16(o2, d[2]), _mm_madd_epi16(o3, d[3])),
            );
            merge_shift_store8(even_acc, odd_acc, shift, out);
        }
    }

    /// Luma 8-tap horizontal FIR for one row. `row` must hold at least
    /// `out.len() + 7` valid u16 samples (interior guarantee).
    ///
    /// See the module docs for the tap-pair window design. A 16-wide chunk
    /// is one 256-bit computation over `[R0..R23]` (three 128-bit loads);
    /// an 8-wide chunk is one 128-bit computation. Chunks that would read
    /// past `row` (tight edge windows) fall through to smaller chunks and
    /// finally to the SSE2 row kernel for the tail.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn luma_hfir_row(row: &[u16], f: [i16; 8], shift: i32, out: &mut [i16]) {
        let d = [
            pair128(f[0], f[1]),
            pair128(f[2], f[3]),
            pair128(f[4], f[5]),
            pair128(f[6], f[7]),
        ];
        let base = row.as_ptr();
        let n = out.len();
        let mut x = 0usize;
        while x + 16 <= n && x + 24 <= row.len() {
            unsafe {
                let a = _mm_loadu_si128(base.add(x) as *const __m128i); // R[0..7]
                let b = _mm_loadu_si128(base.add(x + 8) as *const __m128i); // R[8..15]
                let c = _mm_loadu_si128(base.add(x + 16) as *const __m128i); // R[16..23]
                // v0 = [R0..R7 | R8..R15], v1 = [R8..R15 | R16..R23].
                // `_mm256_alignr_epi8` shifts each 128-bit half over its own
                // 32-byte stream `[v0_half || v1_half]`, so for even k the
                // low half holds the window of even output k/2 in [R0..R15]
                // and the high half the window of even output 8 + k/2 in
                // [R8..R23]; one madd accumulates both at once.
                let v0 = _mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1);
                let v1 = _mm256_inserti128_si256(_mm256_castsi128_si256(b), c, 1);
                // Tap-pair j (byte shift 4j / 4j+2) pairs every lane with
                // (f[2j], f[2j+1]); accumulate across the four shifts.
                let d = [
                    pair256(f[0], f[1]),
                    pair256(f[2], f[3]),
                    pair256(f[4], f[5]),
                    pair256(f[6], f[7]),
                ];
                let (mut even_acc, mut odd_acc) = (_mm256_setzero_si256(), _mm256_setzero_si256());
                even_acc = _mm256_add_epi32(
                    even_acc,
                    _mm256_madd_epi16(_mm256_alignr_epi8::<0>(v1, v0), d[0]),
                );
                even_acc = _mm256_add_epi32(
                    even_acc,
                    _mm256_madd_epi16(_mm256_alignr_epi8::<4>(v1, v0), d[1]),
                );
                even_acc = _mm256_add_epi32(
                    even_acc,
                    _mm256_madd_epi16(_mm256_alignr_epi8::<8>(v1, v0), d[2]),
                );
                even_acc = _mm256_add_epi32(
                    even_acc,
                    _mm256_madd_epi16(_mm256_alignr_epi8::<12>(v1, v0), d[3]),
                );
                odd_acc = _mm256_add_epi32(
                    odd_acc,
                    _mm256_madd_epi16(_mm256_alignr_epi8::<2>(v1, v0), d[0]),
                );
                odd_acc = _mm256_add_epi32(
                    odd_acc,
                    _mm256_madd_epi16(_mm256_alignr_epi8::<6>(v1, v0), d[1]),
                );
                odd_acc = _mm256_add_epi32(
                    odd_acc,
                    _mm256_madd_epi16(_mm256_alignr_epi8::<10>(v1, v0), d[2]),
                );
                odd_acc = _mm256_add_epi32(
                    odd_acc,
                    _mm256_madd_epi16(_mm256_alignr_epi8::<14>(v1, v0), d[3]),
                );
                merge_shift_store16(even_acc, odd_acc, shift, &mut out[x..x + 16]);
            }
            x += 16;
        }
        while x + 8 <= n && x + 16 <= row.len() {
            unsafe {
                let a = _mm_loadu_si128(base.add(x) as *const __m128i); // R[0..7]
                let b = _mm_loadu_si128(base.add(x + 8) as *const __m128i); // R[8..15]
                hfir8_luma(a, b, d, shift, &mut out[x..x + 8]);
            }
            x += 8;
        }
        if x < n {
            sse2::luma_hfir_row(&row[x..], f, shift, &mut out[x..]);
        }
    }

    /// Chroma 4-tap horizontal FIR for one row. `row` must hold at least
    /// `out.len() + 3` valid u16 samples (interior guarantee).
    ///
    /// Same tap-pair window design as `luma_hfir_row` with two tap pairs. A
    /// 16-wide chunk reads exactly `R[0..18]` (two 128-bit loads plus three
    /// scalar tail samples), so it fits the interior `+3` margin; an 8-wide
    /// chunk reads `R[0..10]`. Tighter windows fall through to the SSE2 row
    /// kernel for the tail.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn chroma_hfir_row(row: &[u16], f: [i16; 4], shift: i32, out: &mut [i16]) {
        let c = [pair256(f[0], f[1]), pair256(f[2], f[3])];
        let d = [pair128(f[0], f[1]), pair128(f[2], f[3])];
        let base = row.as_ptr();
        let n = out.len();
        let mut x = 0usize;
        while x + 16 <= n && x + 19 <= row.len() {
            unsafe {
                let a = _mm_loadu_si128(base.add(x) as *const __m128i); // R[0..7]
                let b = _mm_loadu_si128(base.add(x + 8) as *const __m128i); // R[8..15]
                let xt = _mm_setr_epi16(
                    *base.add(x + 16) as i16,
                    *base.add(x + 17) as i16,
                    *base.add(x + 18) as i16,
                    0,
                    0,
                    0,
                    0,
                    0,
                );
                let zero = _mm256_setzero_si256();
                let (mut even_acc, mut odd_acc) = (zero, zero);
                // j=0: windows R[0..15] / R[1..16].
                let w0e = _mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1);
                even_acc = _mm256_add_epi32(even_acc, _mm256_madd_epi16(w0e, c[0]));
                let w0o = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(
                        _mm_or_si128(_mm_srli_si128::<2>(a), _mm_slli_si128::<14>(b)),
                    ),
                    _mm_or_si128(_mm_srli_si128::<2>(b), _mm_slli_si128::<14>(xt)),
                    1,
                );
                odd_acc = _mm256_add_epi32(odd_acc, _mm256_madd_epi16(w0o, c[0]));
                // j=1: windows R[2..17] / R[3..18].
                let w1e = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(
                        _mm_or_si128(_mm_srli_si128::<4>(a), _mm_slli_si128::<12>(b)),
                    ),
                    _mm_or_si128(_mm_srli_si128::<4>(b), _mm_slli_si128::<12>(xt)),
                    1,
                );
                even_acc = _mm256_add_epi32(even_acc, _mm256_madd_epi16(w1e, c[1]));
                let w1o = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(
                        _mm_or_si128(_mm_srli_si128::<6>(a), _mm_slli_si128::<10>(b)),
                    ),
                    _mm_or_si128(_mm_srli_si128::<6>(b), _mm_slli_si128::<10>(xt)),
                    1,
                );
                odd_acc = _mm256_add_epi32(odd_acc, _mm256_madd_epi16(w1o, c[1]));
                merge_shift_store16(even_acc, odd_acc, shift, &mut out[x..x + 16]);
            }
            x += 16;
        }
        while x + 8 <= n && x + 11 <= row.len() {
            unsafe {
                let a = _mm_loadu_si128(base.add(x) as *const __m128i); // R[0..7]
                let xt = _mm_setr_epi16(
                    *base.add(x + 8) as i16,
                    *base.add(x + 9) as i16,
                    *base.add(x + 10) as i16,
                    0,
                    0,
                    0,
                    0,
                    0,
                );
                let zero = _mm_setzero_si128();
                let (mut even_acc, mut odd_acc) = (zero, zero);
                // j=0: windows R[0..7] / R[1..8].
                even_acc = _mm_add_epi32(even_acc, _mm_madd_epi16(a, d[0]));
                odd_acc = _mm_add_epi32(
                    odd_acc,
                    _mm_madd_epi16(
                        _mm_or_si128(_mm_srli_si128::<2>(a), _mm_slli_si128::<14>(xt)),
                        d[0],
                    ),
                );
                // j=1: windows R[2..9] / R[3..10].
                even_acc = _mm_add_epi32(
                    even_acc,
                    _mm_madd_epi16(
                        _mm_or_si128(_mm_srli_si128::<4>(a), _mm_slli_si128::<12>(xt)),
                        d[1],
                    ),
                );
                odd_acc = _mm_add_epi32(
                    odd_acc,
                    _mm_madd_epi16(
                        _mm_or_si128(_mm_srli_si128::<6>(a), _mm_slli_si128::<10>(xt)),
                        d[1],
                    ),
                );
                merge_shift_store8(even_acc, odd_acc, shift, &mut out[x..x + 8]);
            }
            x += 8;
        }
        if x < n {
            sse2::chroma_hfir_row(&row[x..], f, shift, &mut out[x..]);
        }
    }

    /// Luma interior kernel — all four frac combinations, direct access.
    /// Precondition: the interior bounds checked by `interpolate_luma` hold
    /// and `n_pb_w % 4 == 0`.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    pub fn luma_interior(
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
        fir_tmp: &mut [i16],
    ) {
        debug_assert_eq!(n_pb_w % 4, 0);
        if x_frac == 0 && y_frac == 0 {
            for y in 0..n_pb_h {
                let off = ((y_int + y as i32) * stride + x_int) as usize;
                copy_shifted(
                    &plane[off..off + n_pb_w],
                    shift3,
                    &mut pred[y * n_pb_w..(y + 1) * n_pb_w],
                );
            }
        } else if y_frac == 0 {
            let f = LUMA_FILTER[x_frac as usize];
            for y in 0..n_pb_h {
                let off = ((y_int + y as i32) * stride + (x_int - 3)) as usize;
                luma_hfir_row(
                    &plane[off..off + n_pb_w + 8],
                    f,
                    shift1,
                    &mut pred[y * n_pb_w..(y + 1) * n_pb_w],
                );
            }
        } else if x_frac == 0 {
            let f = LUMA_FILTER[y_frac as usize];
            let base =
                unsafe { plane.as_ptr().cast::<i16>().add(((y_int - 3) * stride + x_int) as usize) };
            vfir_core(base, stride as usize, n_pb_w, n_pb_h, f, shift1, pred);
        } else {
            let tmp_h = n_pb_h + 7;
            debug_assert!(n_pb_w <= 64 && tmp_h <= 71);
            // Pass 1 writes every row/col pass 2 reads — no init needed.
            let tmp = &mut fir_tmp[..n_pb_w * tmp_h];
            let f_h = LUMA_FILTER[x_frac as usize];
            for y in 0..tmp_h {
                let off = ((y_int + y as i32 - 3) * stride + (x_int - 3)) as usize;
                luma_hfir_row(
                    &plane[off..off + n_pb_w + 8],
                    f_h,
                    shift1,
                    &mut tmp[y * n_pb_w..(y + 1) * n_pb_w],
                );
            }
            let f_v = LUMA_FILTER[y_frac as usize];
            vfir_core(tmp.as_ptr(), n_pb_w, n_pb_w, n_pb_h, f_v, shift2, pred);
        }
    }

    /// Chroma interior kernel — all four frac combinations, direct access.
    /// Precondition: the interior bounds checked by `interpolate_chroma` hold
    /// and `n_pb_wc % 4 == 0`.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    pub fn chroma_interior(
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
        fir_tmp: &mut [i16],
    ) {
        debug_assert_eq!(n_pb_wc % 4, 0);
        if x_frac == 0 && y_frac == 0 {
            for y in 0..n_pb_hc {
                let off = ((y_int + y as i32) * stride + x_int) as usize;
                copy_shifted(
                    &plane[off..off + n_pb_wc],
                    shift3,
                    &mut pred[y * n_pb_wc..(y + 1) * n_pb_wc],
                );
            }
        } else if y_frac == 0 {
            let f = CHROMA_FILTER[x_frac as usize];
            for y in 0..n_pb_hc {
                let off = ((y_int + y as i32) * stride + (x_int - 1)) as usize;
                chroma_hfir_row(
                    &plane[off..off + n_pb_wc + 3],
                    f,
                    shift1,
                    &mut pred[y * n_pb_wc..(y + 1) * n_pb_wc],
                );
            }
        } else if x_frac == 0 {
            let f = CHROMA_FILTER[y_frac as usize];
            let base =
                unsafe { plane.as_ptr().cast::<i16>().add(((y_int - 1) * stride + x_int) as usize) };
            vfir_core(base, stride as usize, n_pb_wc, n_pb_hc, f, shift1, pred);
        } else {
            let tmp_h = n_pb_hc + 3;
            debug_assert!(n_pb_wc <= 32 && tmp_h <= 35);
            // Pass 1 writes every row/col pass 2 reads — no init needed.
            let tmp = &mut fir_tmp[..n_pb_wc * tmp_h];
            let f_h = CHROMA_FILTER[x_frac as usize];
            for y in 0..tmp_h {
                let off = ((y_int + y as i32 - 1) * stride + (x_int - 1)) as usize;
                chroma_hfir_row(
                    &plane[off..off + n_pb_wc + 3],
                    f_h,
                    shift1,
                    &mut tmp[y * n_pb_wc..(y + 1) * n_pb_wc],
                );
            }
            let f_v = CHROMA_FILTER[y_frac as usize];
            vfir_core(tmp.as_ptr(), n_pb_wc, n_pb_wc, n_pb_hc, f_v, shift2, pred);
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

                    let mut fir = [0i16; FIR_SCRATCH_MAX];
                    interpolate_luma(
                        &plane, pic_w, pic_h, stride, x_int, y_int, x_frac, y_frac,
                        n_pb_w, n_pb_h, bit_depth, &mut pred_rs, &mut fir[..],
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

                    let mut fir = [0i16; FIR_SCRATCH_MAX];
                    interpolate_chroma(
                        &plane, 1, pic_w, pic_h, stride, x_int, y_int, x_frac, y_frac,
                        n_pb_wc, n_pb_hc, bit_depth, &mut pred_rs, &mut fir[..],
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

    /// AVX2 interior kernels must be byte-exact with the scalar cores: all
    /// frac combinations, widths that exercise 16-wide chunks plus tails.
    #[test]
    fn avx2_interior_kernels_match_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = Rng::new(0xA11CE_0005);
        for &(w, h) in &[(4usize, 4), (8, 8), (12, 6), (16, 16), (20, 8), (24, 12), (32, 8), (64, 16)] {
            for bit_depth in [8i32, 10] {
                let shift1 = 4.min(bit_depth - 8);
                let shift2 = 6;
                let shift3 = 2.max(14 - bit_depth);
                let pic_w = (w + 16) as i32;
                let pic_h = (h + 16) as i32;
                let stride = pic_w;
                for _ in 0..8 {
                    let plane: Vec<u16> = (0..pic_w * pic_h)
                        .map(|_| rng.below(1u64 << (bit_depth as u32)) as u16)
                        .collect();
                    let (x_int, y_int) = (8i32, 8i32); // interior for all sizes here
                    for x_frac in 0..4 {
                        for y_frac in 0..4 {
                            let mut out_avx = vec![0i16; w * h];
                            let mut out_scalar = vec![0i16; w * h];
                            let mut fir_a = [0i16; FIR_SCRATCH_MAX];
                            let mut fir_s = [0i16; FIR_SCRATCH_MAX];
                            interpolate_luma(
                                &plane, pic_w, pic_h, stride, x_int, y_int, x_frac, y_frac, w, h,
                                bit_depth, &mut out_avx, &mut fir_a[..],
                            );
                            luma_interior_scalar(
                                &plane, stride, x_int, y_int, x_frac, y_frac, w, h, shift1,
                                shift2, shift3, &mut out_scalar, &mut fir_s[..],
                            );
                            assert_eq!(
                                out_avx, out_scalar,
                                "luma {w}x{h} bd={bit_depth} frac=({x_frac},{y_frac})"
                            );
                        }
                    }
                }
            }
        }
        for &(wc, hc) in &[(4usize, 4), (8, 8), (16, 16), (24, 8), (32, 16)] {
            for bit_depth in [8i32, 10] {
                let shift1 = 4.min(bit_depth - 8);
                let shift2 = 6;
                let shift3 = 2.max(14 - bit_depth);
                let pic_w = (wc + 8) as i32;
                let pic_h = (hc + 8) as i32;
                let stride = pic_w;
                for _ in 0..8 {
                    let plane: Vec<u16> = (0..pic_w * pic_h)
                        .map(|_| rng.below(1u64 << (bit_depth as u32)) as u16)
                        .collect();
                    let (x_int, y_int) = (4i32, 4i32);
                    for x_frac in 0..8 {
                        for y_frac in 0..8 {
                            let mut out_avx = vec![0i16; wc * hc];
                            let mut out_scalar = vec![0i16; wc * hc];
                            let mut fir_a = [0i16; 32 * 35];
                            let mut fir_s = [0i16; 32 * 35];
                            interpolate_chroma(
                                &plane, 1, pic_w, pic_h, stride, x_int, y_int, x_frac, y_frac,
                                wc, hc, bit_depth, &mut out_avx, &mut fir_a[..],
                            );
                            chroma_interior_scalar(
                                &plane, stride, x_int, y_int, x_frac, y_frac, wc, hc, shift1,
                                shift2, shift3, &mut out_scalar, &mut fir_s[..],
                            );
                            assert_eq!(
                                out_avx, out_scalar,
                                "chroma {wc}x{hc} bd={bit_depth} frac=({x_frac},{y_frac})"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Direct coverage of the AVX2 horizontal FIR rows: every chunk/tail
    /// boundary (16-wide, 8-wide, SSE2 tail) and both margin variants for
    /// luma (tight edge window `+7` vs interior `+8`). Byte-exact against a
    /// scalar reference.
    #[test]
    fn avx2_hfir_rows_match_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = Rng::new(0xA11CE_0006);
        for &w in &[4usize, 8, 12, 16, 20, 24, 28, 32, 44, 48, 52, 64] {
            for x_frac in 1..4 {
                let shift = 4i32;
                for margin in [7usize, 8] {
                    let row: Vec<u16> = (0..w + margin).map(|_| rng.below(1025) as u16).collect();
                    let mut out_avx = vec![0i16; w];
                    let mut out_ref = vec![0i16; w];
                    unsafe { avx2::luma_hfir_row(&row, LUMA_FILTER[x_frac], shift, &mut out_avx) };
                    for (c, o) in out_ref.iter_mut().enumerate() {
                        let sum: i32 = LUMA_FILTER[x_frac]
                            .iter()
                            .enumerate()
                            .map(|(t, &tap)| tap as i32 * row[c + t] as i32)
                            .sum();
                        *o = (sum >> shift) as i16;
                    }
                    assert_eq!(
                        out_avx, out_ref,
                        "luma hfir w={w} frac={x_frac} margin={margin}"
                    );
                }
            }
        }
        for &w in &[4usize, 8, 12, 16, 20, 24, 32] {
            for x_frac in 1..8 {
                let shift = 4i32;
                let row: Vec<u16> = (0..w + 3).map(|_| rng.below(1025) as u16).collect();
                let mut out_avx = vec![0i16; w];
                let mut out_ref = vec![0i16; w];
                unsafe { avx2::chroma_hfir_row(&row, CHROMA_FILTER[x_frac], shift, &mut out_avx) };
                for (c, o) in out_ref.iter_mut().enumerate() {
                    let sum: i32 = CHROMA_FILTER[x_frac]
                        .iter()
                        .enumerate()
                        .map(|(t, &tap)| tap as i32 * row[c + t] as i32)
                        .sum();
                    *o = (sum >> shift) as i16;
                }
                assert_eq!(out_avx, out_ref, "chroma hfir w={w} frac={x_frac}");
            }
        }
    }

    #[test]
    fn dbg_hfir_dump() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = Rng::new(0xA11CE_0006);
        for &w in &[4usize, 8, 12] {
            for _x_frac in 1..4 {
                for &margin in &[7usize, 8] {
                    let _row: Vec<u16> = (0..w + margin).map(|_| rng.below(1025) as u16).collect();
                }
            }
        }
        let w = 16usize;
        let x_frac = 1;
        for &margin in &[7usize, 8] {
            let row: Vec<u16> = (0..w + margin).map(|_| rng.below(1025) as u16).collect();
            eprintln!("margin={margin} row: {row:?}");
            let f = LUMA_FILTER[x_frac];
            let mut out_ref = vec![0i16; w];
            for (c, o) in out_ref.iter_mut().enumerate() {
                let sum: i32 =
                    f.iter().enumerate().map(|(t, &tap)| tap as i32 * row[c + t] as i32).sum();
                *o = (sum >> 4) as i16;
            }
            eprintln!("  ref: {out_ref:?}");
            let mut out_avx = vec![0i16; w];
            unsafe { avx2::luma_hfir_row(&row, f, 4, &mut out_avx) };
            eprintln!("  avx: {out_avx:?}");
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

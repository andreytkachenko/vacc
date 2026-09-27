//! SSE4.1 128-bit vertical-pass kernels.
//!
//! Used for both the `Sse` and the `Avx2` kernel selections (the `Avx2`
//! selection additionally gets a gather-based horizontal pass in `avx2.rs`).
//!
//! In the vertical pass the tap-row weights are uniform across output
//! pixels, so each chunk of output pixels is: load the tap-row chunks, widen
//! to `f32` lanes, `acc += w * px` per tap (same order as the scalar
//! reference), round/clamp/store. Tails run the same scalar arithmetic, so
//! every kernel produces byte-identical output.

use core::arch::x86_64::*;

use crate::resize::{TapAxis, round_clamp};

/// One vertical-pass output row (`dst_row` holds `width * ch` bytes).
#[target_feature(enable = "sse4.1")]
pub fn v_pass_row(
    base: *const u8,
    pitch: usize,
    width: usize,
    ch: usize,
    taps: &TapAxis,
    oy: usize,
    dst_row: &mut [u8],
) {
    let (prow, wrow) = taps.row(oy);
    let n = taps.n_taps;
    match ch {
        1 => v_pass_row_c1(base, pitch, width, prow, wrow, n, dst_row),
        3 => v_pass_row_c3(base, pitch, width, prow, wrow, n, dst_row),
        _ => v_pass_row_c4(base, pitch, width, prow, wrow, n, dst_row),
    }
}

/// Scalar tail pixel (identical arithmetic to the scalar vertical pass).
#[inline]
fn tail_px(
    base: *const u8,
    pitch: usize,
    prow: &[i32],
    wrow: &[f32],
    n: usize,
    ox: usize,
    ch: usize,
    dst_row: &mut [u8],
) {
    let mut acc: [f32; 4] = [0.0; 4];
    for k in 0..n {
        let row = unsafe { base.add(prow[k] as usize * pitch) };
        let w = wrow[k];
        for c in 0..ch {
            acc[c] += w * unsafe { *row.add(ox * ch + c) } as f32;
        }
    }
    for c in 0..ch {
        dst_row[ox * ch + c] = round_clamp(acc[c]);
    }
}

/// Round half up (floor(x + 0.5)) + clamp to [0, 255] -> i32x4.
#[inline]
#[target_feature(enable = "sse4.1")]
fn round_clamp_i32(v: __m128) -> __m128i {
    let r = _mm_floor_ps(_mm_add_ps(v, _mm_set1_ps(0.5)));
    let c = _mm_min_ps(_mm_max_ps(r, _mm_setzero_ps()), _mm_set1_ps(255.0));
    _mm_cvtps_epi32(c) // exact: values are rounded integers
}

/// Store 4 consecutive rounded bytes (ch = 1) at `dst + off`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn store_rounded(v: __m128, dst_row: &mut [u8], off: usize) {
    let i = round_clamp_i32(v);
    dst_row[off] = _mm_extract_epi32::<0>(i) as u8;
    dst_row[off + 1] = _mm_extract_epi32::<1>(i) as u8;
    dst_row[off + 2] = _mm_extract_epi32::<2>(i) as u8;
    dst_row[off + 3] = _mm_extract_epi32::<3>(i) as u8;
}

/// Zero-extended i32x4 -> f32x4 (callers do the `cvtepu8` step).
#[inline]
#[target_feature(enable = "sse2")]
fn widen4(v: __m128i) -> __m128 {
    _mm_cvtepi32_ps(v)
}

/// ch = 1 (YUV planes): 16 pixels per iteration.
#[target_feature(enable = "sse4.1")]
fn v_pass_row_c1(
    base: *const u8,
    pitch: usize,
    width: usize,
    prow: &[i32],
    wrow: &[f32],
    n: usize,
    dst_row: &mut [u8],
) {
    let mut ox = 0usize;
    while ox + 16 <= width {
        let mut a = _mm_setzero_ps();
        let mut b = _mm_setzero_ps();
        let mut c = _mm_setzero_ps();
        let mut d = _mm_setzero_ps();
        for k in 0..n {
            let w = _mm_set1_ps(wrow[k]);
            let row = unsafe { base.add(prow[k] as usize * pitch) };
            let v = unsafe { _mm_loadu_si128(row.add(ox) as *const __m128i) };
            a = _mm_add_ps(a, _mm_mul_ps(w, widen4(_mm_cvtepu8_epi32(v))));
            b = _mm_add_ps(b, _mm_mul_ps(w, widen4(_mm_cvtepu8_epi32(_mm_srli_si128::<4>(v)))));
            c = _mm_add_ps(c, _mm_mul_ps(w, widen4(_mm_cvtepu8_epi32(_mm_srli_si128::<8>(v)))));
            d = _mm_add_ps(d, _mm_mul_ps(w, widen4(_mm_cvtepu8_epi32(_mm_srli_si128::<12>(v)))));
        }
        store_rounded(a, dst_row, ox);
        store_rounded(b, dst_row, ox + 4);
        store_rounded(c, dst_row, ox + 8);
        store_rounded(d, dst_row, ox + 12);
        ox += 16;
    }
    while ox < width {
        tail_px(base, pitch, prow, wrow, n, ox, 1, dst_row);
        ox += 1;
    }
}

/// ch = 3 (RGB24): 4 pixels per iteration.
#[target_feature(enable = "sse4.1")]
fn v_pass_row_c3(
    base: *const u8,
    pitch: usize,
    width: usize,
    prow: &[i32],
    wrow: &[f32],
    n: usize,
    dst_row: &mut [u8],
) {
    // pshufb masks de-interleaving 4 RGB pixels (12 bytes) into per-channel
    // vectors; the upper mask lanes re-select byte 0 but are never read.
    let mr = _mm_setr_epi8(0, 3, 6, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    let mg = _mm_setr_epi8(1, 4, 7, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    let mb = _mm_setr_epi8(2, 5, 8, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    let mut ox = 0usize;
    while ox + 4 <= width {
        let mut ar = _mm_setzero_ps();
        let mut ag = _mm_setzero_ps();
        let mut ab = _mm_setzero_ps();
        for k in 0..n {
            let w = _mm_set1_ps(wrow[k]);
            let row = unsafe { base.add(prow[k] as usize * pitch) };
            let v = unsafe { _mm_loadu_si128(row.add(ox * 3) as *const __m128i) };
            ar = _mm_add_ps(ar, _mm_mul_ps(w, widen4(_mm_cvtepu8_epi32(_mm_shuffle_epi8(v, mr)))));
            ag = _mm_add_ps(ag, _mm_mul_ps(w, widen4(_mm_cvtepu8_epi32(_mm_shuffle_epi8(v, mg)))));
            ab = _mm_add_ps(ab, _mm_mul_ps(w, widen4(_mm_cvtepu8_epi32(_mm_shuffle_epi8(v, mb)))));
        }
        // Pixels are interleaved (stride 3): scatter each channel's lanes.
        let mut r = [0i32; 4];
        let mut g = [0i32; 4];
        let mut b = [0i32; 4];
        unsafe {
            _mm_storeu_si128(r.as_mut_ptr() as *mut __m128i, round_clamp_i32(ar));
            _mm_storeu_si128(g.as_mut_ptr() as *mut __m128i, round_clamp_i32(ag));
            _mm_storeu_si128(b.as_mut_ptr() as *mut __m128i, round_clamp_i32(ab));
        }
        for lane in 0..4 {
            let base = (ox + lane) * 3;
            dst_row[base] = r[lane] as u8;
            dst_row[base + 1] = g[lane] as u8;
            dst_row[base + 2] = b[lane] as u8;
        }
        ox += 4;
    }
    while ox < width {
        tail_px(base, pitch, prow, wrow, n, ox, 3, dst_row);
        ox += 1;
    }
}

/// ch = 4 (RGBA32): 4 pixels per iteration.
#[target_feature(enable = "sse4.1")]
fn v_pass_row_c4(
    base: *const u8,
    pitch: usize,
    width: usize,
    prow: &[i32],
    wrow: &[f32],
    n: usize,
    dst_row: &mut [u8],
) {
    let mr = _mm_setr_epi8(0, 4, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    let mg = _mm_setr_epi8(1, 5, 9, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    let mb = _mm_setr_epi8(2, 6, 10, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    let ma = _mm_setr_epi8(3, 7, 11, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    let mut ox = 0usize;
    while ox + 4 <= width {
        let mut ar = _mm_setzero_ps();
        let mut ag = _mm_setzero_ps();
        let mut ab = _mm_setzero_ps();
        let mut aa = _mm_setzero_ps();
        for k in 0..n {
            let w = _mm_set1_ps(wrow[k]);
            let row = unsafe { base.add(prow[k] as usize * pitch) };
            let v = unsafe { _mm_loadu_si128(row.add(ox * 4) as *const __m128i) };
            ar = _mm_add_ps(ar, _mm_mul_ps(w, widen4(_mm_cvtepu8_epi32(_mm_shuffle_epi8(v, mr)))));
            ag = _mm_add_ps(ag, _mm_mul_ps(w, widen4(_mm_cvtepu8_epi32(_mm_shuffle_epi8(v, mg)))));
            ab = _mm_add_ps(ab, _mm_mul_ps(w, widen4(_mm_cvtepu8_epi32(_mm_shuffle_epi8(v, mb)))));
            aa = _mm_add_ps(aa, _mm_mul_ps(w, widen4(_mm_cvtepu8_epi32(_mm_shuffle_epi8(v, ma)))));
        }
        // Pixels are interleaved (stride 4): scatter each channel's lanes.
        let mut r = [0i32; 4];
        let mut g = [0i32; 4];
        let mut b = [0i32; 4];
        let mut a = [0i32; 4];
        unsafe {
            _mm_storeu_si128(r.as_mut_ptr() as *mut __m128i, round_clamp_i32(ar));
            _mm_storeu_si128(g.as_mut_ptr() as *mut __m128i, round_clamp_i32(ag));
            _mm_storeu_si128(b.as_mut_ptr() as *mut __m128i, round_clamp_i32(ab));
            _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, round_clamp_i32(aa));
        }
        for lane in 0..4 {
            let base = (ox + lane) * 4;
            dst_row[base] = r[lane] as u8;
            dst_row[base + 1] = g[lane] as u8;
            dst_row[base + 2] = b[lane] as u8;
            dst_row[base + 3] = a[lane] as u8;
        }
        ox += 4;
    }
    while ox < width {
        tail_px(base, pitch, prow, wrow, n, ox, 4, dst_row);
        ox += 1;
    }
}

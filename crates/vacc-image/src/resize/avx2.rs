//! AVX2 horizontal-pass kernel for RGBA32 (gather-based).
//!
//! Per 8 output pixels and per tap: gather the 8 source dwords (one RGBA
//! pixel each), widen each channel to `f32x8`, multiply by the per-lane tap
//! weights and accumulate — the same left-to-right order as the scalar
//! reference, so results are byte-identical. Tap positions are clamped into
//! `[0, src_width - 1]` by the tap table, so every gathered read stays
//! within the tight source row.

use core::arch::x86_64::*;
use core::ptr;

use crate::resize::{TapAxis, h_pass_px};

/// One horizontal-pass output row (`dst_row` holds `dw * 4` bytes).
#[target_feature(enable = "avx2")]
pub fn h_pass_rgba_row(
    base: *const u8,
    pitch: usize,
    src_width: usize,
    taps: &TapAxis,
    oy: usize,
    dst_row: &mut [u8],
) {
    unsafe {
        let dw = dst_row.len() / 4;
        let n = taps.n_taps;
        let src_row = base.add(oy * pitch);
        let mut ox0 = 0usize;
        while ox0 + 8 <= dw {
        let mut ar = _mm256_setzero_ps();
        let mut ag = _mm256_setzero_ps();
        let mut ab = _mm256_setzero_ps();
        let mut aa = _mm256_setzero_ps();
        for k in 0..n {
            // Transposed tables: contiguous across output pixels for tap `k`.
            let xs =
                _mm256_loadu_si256(taps.pos_t.as_ptr().add(k * taps.len() + ox0) as *const __m256i);
            let px = _mm256_i32gather_epi32::<4>(src_row as *const i32, xs);
            let w = _mm256_loadu_ps(taps.weights_t.as_ptr().add(k * taps.len() + ox0));
            ar = _mm256_add_ps(ar, _mm256_mul_ps(w, widen(px, mask_r())));
            ag = _mm256_add_ps(ag, _mm256_mul_ps(w, widen(px, mask_g())));
            ab = _mm256_add_ps(ab, _mm256_mul_ps(w, widen(px, mask_b())));
            aa = _mm256_add_ps(aa, _mm256_mul_ps(w, widen(px, mask_a())));
        }
        let ir = round_clamp_vec(ar);
        let ig = round_clamp_vec(ag);
        let ib = round_clamp_vec(ab);
        let ia = round_clamp_vec(aa);
        // Spill the clamped i32x8 vectors to the stack, then pack RGBA dwords.
        let mut r = [0i32; 8];
        let mut g = [0i32; 8];
        let mut b = [0i32; 8];
        let mut a = [0i32; 8];
        {
            _mm_storeu_si128(r.as_mut_ptr() as *mut __m128i, _mm256_castsi256_si128(ir));
            _mm_storeu_si128(
                r.as_mut_ptr().add(4) as *mut __m128i,
                _mm256_extracti128_si256::<1>(ir),
            );
            _mm_storeu_si128(g.as_mut_ptr() as *mut __m128i, _mm256_castsi256_si128(ig));
            _mm_storeu_si128(
                g.as_mut_ptr().add(4) as *mut __m128i,
                _mm256_extracti128_si256::<1>(ig),
            );
            _mm_storeu_si128(b.as_mut_ptr() as *mut __m128i, _mm256_castsi256_si128(ib));
            _mm_storeu_si128(
                b.as_mut_ptr().add(4) as *mut __m128i,
                _mm256_extracti128_si256::<1>(ib),
            );
            _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, _mm256_castsi256_si128(ia));
            _mm_storeu_si128(
                a.as_mut_ptr().add(4) as *mut __m128i,
                _mm256_extracti128_si256::<1>(ia),
            );
        }
        for lane in 0..8 {
            let px = (r[lane] as u32)
                | ((g[lane] as u32) << 8)
                | ((b[lane] as u32) << 16)
                | ((a[lane] as u32) << 24);
            ptr::write_unaligned((dst_row.as_mut_ptr() as *mut u32).add(ox0 + lane), px);
        }
        ox0 += 8;
        }
        // Scalar tail: same arithmetic as the scalar horizontal pass.
        let srow = std::slice::from_raw_parts(src_row, src_width * 4);
        while ox0 < dw {
            h_pass_px(srow, taps, ox0, 4, dst_row);
            ox0 += 1;
        }
    }
}

/// pshufb masks selecting one RGBA channel of the four low pixels.
#[inline]
#[target_feature(enable = "sse2")]
fn mask_r() -> __m128i {
    _mm_setr_epi8(0, 4, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}
#[inline]
#[target_feature(enable = "sse2")]
fn mask_g() -> __m128i {
    _mm_setr_epi8(1, 5, 9, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}
#[inline]
#[target_feature(enable = "sse2")]
fn mask_b() -> __m128i {
    _mm_setr_epi8(2, 6, 10, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}
#[inline]
#[target_feature(enable = "sse2")]
fn mask_a() -> __m128i {
    _mm_setr_epi8(3, 7, 11, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}

/// Widen one channel of 8 gathered RGBA pixels to `f32x8`.
#[inline]
#[target_feature(enable = "avx2")]
fn widen(px: __m256i, m: __m128i) -> __m256 {
    let lo = _mm_cvtepu8_epi32(_mm_shuffle_epi8(
        _mm256_castsi256_si128(px),
        m,
    ));
    let hi = _mm_cvtepu8_epi32(_mm_shuffle_epi8(
        _mm256_extracti128_si256::<1>(px),
        m,
    ));
    let v = _mm256_insertf128_si256::<1>(_mm256_castsi128_si256(lo), hi);
    _mm256_cvtepi32_ps(v)
}

/// Round-to-nearest-even + clamp to [0, 255] -> i32x8.
#[inline]
#[target_feature(enable = "avx")]
fn round_clamp_vec(v: __m256) -> __m256i {
    // Round half up (floor(x + 0.5)) to match the scalar `round_clamp`.
    let r = _mm256_floor_ps(_mm256_add_ps(v, _mm256_set1_ps(0.5)));
    let c = _mm256_min_ps(_mm256_max_ps(r, _mm256_setzero_ps()), _mm256_set1_ps(255.0));
    _mm256_cvtps_epi32(c)
}

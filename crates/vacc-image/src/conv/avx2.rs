//! AVX2 row kernels: 8-bit Y'CbCr(4:2:0) -> RGB(x) and P010/P012 -> 8-bit
//! down-cast.
//!
//! All entry points are `unsafe`: they require the documented source
//! lengths. Kernel selection lives in `conv.rs`, behind runtime CPU
//! feature detection.
//!
//! Y'CbCr uses 256-bit 8-lane i32 arithmetic, advancing 8 luma pixels per
//! step. Every load is sized so it stays in bounds given
//! `x + 8 <= width`:
//! - Y: an unaligned 8-byte read at `y + x`.
//! - planar Cb/Cr: an unaligned 4-byte read at `c + x / 2` covering the 4
//!   samples that feed 8 luma; `x / 2 + 4 <= (w - 8) / 2 + 4 = w / 2 <=
//!   ceil(w / 2) = chroma width`.
//! - semi-planar UV: an unaligned 16-byte read at `uv + x` (8
//!   interleaved pairs); `x + 16 <= 2w = uv row length`.
//!
//! Each Cb/Cr sample is duplicated onto its two luma pairs with one
//! `pshufb`, then widened `u8 -> u16 -> i32`. The Q14 multiply-add runs in
//! i32 (|sum| < 2^31), after `>> 14` the values are i16-safe, and packing
//! to 8 bits is done by `packs`/`packus` — the same saturation
//! `conv_px` produces with its `clamp(0, 255)`.

use core::arch::x86_64::*;
use core::ptr;

use crate::coeff::{Conv8, RND};

/// One tight `u16` * N luma row -> `u8` * N luma (right-shift by `S`).
#[target_feature(enable = "avx2")]
unsafe fn u16_shift_row_c<const S: i32>(row: *const u8, out: &mut [u8]) {
    unsafe {
        let mut x = 0usize;
        while x + 16 <= out.len() {
            // 16 top-justified `u16` samples in 32 bytes -> `u16 >> S` in 16-bit
            // lanes (values fit in 8 bits after the shift) -> dense bytes via a
            // 128-bit `packus` on the two halves.
            let src = _mm256_loadu_si256(row.add(x * 2) as *const __m256i);
            let sh = _mm256_srli_epi16(src, S);
            let packed = _mm_packus_epi16(
                _mm256_castsi256_si128(sh),
                _mm256_extracti128_si256(sh, 1),
            );
            _mm_storeu_si128(out.as_mut_ptr().add(x) as *mut __m128i, packed);
            x += 16;
        }
        while x < out.len() {
            // The SIMD main loop saturates to 255 (packus); mirror it here.
            let lo = *row.add(x * 2) as u32;
            let hi = *row.add(x * 2 + 1) as u32;
            out[x] = ((((hi << 8) | lo) >> S).min(255)) as u8;
            x += 1;
        }
    }
}

/// One tight `u16` row -> `u8` row (right-shift by `shift`).
///
/// The AVX2 path needs a compile-time shift, so only the shifts that occur
/// in practice (P012: 4, P010: 6) get SIMD; anything else falls back to
/// the scalar loop in `conv.rs`.
#[target_feature(enable = "avx2")]
pub unsafe fn u16_shift_row(row: *const u8, out: &mut [u8], shift: u32) {
    match shift {
        4 => unsafe { u16_shift_row_c::<4>(row, out) },
        6 => unsafe { u16_shift_row_c::<6>(row, out) },
        _ => crate::conv::sc_shift_row(row, out, shift),
    }
}

/// One tight interleaved `u16` CbCr row (2 * w bytes) -> two tight `u8` rows
/// of `w` bytes each (right-shift by `S`).
#[target_feature(enable = "avx2")]
unsafe fn u16_semi_shift_row_c<const S: i32>(
    uv: *const u8,
    out_u: &mut [u8],
    out_v: &mut [u8],
    width: usize,
) {
    unsafe {
        let mut i = 0usize;
        while i + 8 <= width {
            let p = _mm256_loadu_si256(uv.add(i * 4) as *const __m256i);
            macro_rules! pair {
                ($k:literal) => {{
                    let word = _mm256_extract_epi32(p, $k);
                    // Mirror the saturating 16 -> 8 semantics of the scalar
                    // fallback and the SIMD main loop for the tails.
                    out_u[i + $k] = (((word as u16 as u32) >> S).min(255)) as u8;
                    out_v[i + $k] = ((((word >> 16) as u16 as u32) >> S).min(255)) as u8;
                }};
            }
            pair!(0);
            pair!(1);
            pair!(2);
            pair!(3);
            pair!(4);
            pair!(5);
            pair!(6);
            pair!(7);
            i += 8;
        }
        while i < width {
            let w = ptr::read_unaligned(uv.add(i * 4) as *const u32);
            out_u[i] = (((w as u16 as u32) >> S).min(255)) as u8;
            out_v[i] = ((((w >> 16) as u16 as u32) >> S).min(255)) as u8;
            i += 1;
        }
    }
}

/// One tight interleaved `u16` CbCr row -> two `u8` rows (right-shift).
#[target_feature(enable = "avx2")]
pub unsafe fn u16_semi_shift_row(
    uv: *const u8,
    out_u: &mut [u8],
    out_v: &mut [u8],
    shift: u32,
    width: usize,
) {
    match shift {
        4 => unsafe { u16_semi_shift_row_c::<4>(uv, out_u, out_v, width) },
        6 => unsafe { u16_semi_shift_row_c::<6>(uv, out_u, out_v, width) },
        _ => crate::conv::sc_semi_u16_row(uv, out_u, out_v, shift, width),
    }
}

// ─────────────── 8-bit conversion kernels (8 luma pixels / iteration) ───────────────

/// Rotate 4 Cb/Cr samples up onto their two luma pairs each:
/// `[c0 c1 c2 c3] -> [c0 c0 c1 c1 c2 c2 c3 c3]`.
#[target_feature(enable = "avx2")]
fn dup4_mask() -> __m256i {
    _mm256_setr_epi8(
        0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    )
}

/// One luma row (planar I420 layout) via AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn planar_row(
    y_row: *const u8,
    cb_row: *const u8,
    cr_row: *const u8,
    width: usize,
    c: &Conv8,
    dst_row: &mut [u8],
) {
    unsafe {
        let rgba = dst_row.len() == width * 4;
        let ky = _mm256_set1_epi32(c.ky);
        let rcr = _mm256_set1_epi32(c.r_cr);
        let gcb = _mm256_set1_epi32(c.g_cb);
        let gcr = _mm256_set1_epi32(c.g_cr);
        let bcb = _mm256_set1_epi32(c.b_cb);
        let rcp = _mm256_set1_epi32(c.r_off + RND);
        let gcp = _mm256_set1_epi32(c.g_off + RND);
        let bcp = _mm256_set1_epi32(c.b_off + RND);
        let dup = dup4_mask();
        let d8 = dst_row.as_mut_ptr();

        let mut x = 0usize;
        while x + 8 <= width {
            // Y: widen 8 luma bytes -> 8 i32 (low half only; upper lanes 0).
            let y = _mm256_set_epi64x(
                0,
                0,
                0,
                ptr::read_unaligned(y_row.add(x) as *const u64) as i64,
            );
            let y32 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(y));

            // Cb/Cr: 4 samples each; duplicate onto their two luma pairs, widen
            // -> 8 i32.
            let cb = _mm256_set_epi32(
                0, 0, 0, 0, 0, 0, 0,
                ptr::read_unaligned(cb_row.add(x / 2) as *const u32) as i32,
            );
            let cr = _mm256_set_epi32(
                0, 0, 0, 0, 0, 0, 0,
                ptr::read_unaligned(cr_row.add(x / 2) as *const u32) as i32,
            );
            let cb32 =
                _mm256_cvtepu8_epi32(_mm256_castsi256_si128(_mm256_shuffle_epi8(cb, dup)));
            let cr32 =
                _mm256_cvtepu8_epi32(_mm256_castsi256_si128(_mm256_shuffle_epi8(cr, dup)));

            q14_to_rgb(
                &ky, &rcr, &gcb, &gcr, &bcb, &rcp, &gcp, &bcp,
                y32, cb32, cr32, d8, x, if rgba { 4 } else { 3 },
            );
            x += 8;
        }

        // Scalar tail.
        while x < width {
            let yy = *y_row.add(x) as i32;
            let c0 = *cb_row.add(x >> 1) as i32;
            let cr = *cr_row.add(x >> 1) as i32;
            let (r, g, b) = crate::conv::conv_px(c, yy, c0, cr);
            let o = x * if rgba { 4 } else { 3 };
            *d8.add(o) = r;
            *d8.add(o + 1) = g;
            *d8.add(o + 2) = b;
            if rgba {
                *d8.add(o + 3) = 255;
            }
            x += 1;
        }
    }
}

/// One luma row (semi-planar NV12 layout) via AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn semi_row(
    y_row: *const u8,
    uv_row: *const u8,
    width: usize,
    c: &Conv8,
    dst_row: &mut [u8],
) {
    unsafe {
        let rgba = dst_row.len() == width * 4;
        let ky = _mm256_set1_epi32(c.ky);
        let rcr = _mm256_set1_epi32(c.r_cr);
        let gcb = _mm256_set1_epi32(c.g_cb);
        let gcr = _mm256_set1_epi32(c.g_cr);
        let bcb = _mm256_set1_epi32(c.b_cb);
        let rcp = _mm256_set1_epi32(c.r_off + RND);
        let gcp = _mm256_set1_epi32(c.g_off + RND);
        let bcp = _mm256_set1_epi32(c.b_off + RND);
        let dup = dup4_mask();
        let d8 = dst_row.as_mut_ptr();
        // maddubs pair selector: [1, 0] repeated -> even bytes * 1 + odd * 0.
        let ev = _mm256_setr_epi32(0x00010001, 0x00010001, 0x00010001, 0x00010001,
                                   0x00010001, 0x00010001, 0x00010001, 0x00010001);
        let od = _mm256_setr_epi32(0x01000100, 0x01000100, 0x01000100, 0x01000100,
                                   0x01000100, 0x01000100, 0x01000100, 0x01000100);

        let mut x = 0usize;
        while x + 8 <= width {
            // Y: widen 8 luma bytes -> 8 i32.
            let y = _mm256_set_epi64x(
                0,
                0,
                0,
                ptr::read_unaligned(y_row.add(x) as *const u64) as i64,
            );
            let y32 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(y));

            // CbCr: 8 bytes = 4 (u, v) pairs for the 8 luma pixels; each
            // chroma sample covers 2 luma pixels, so load only what is needed
            // (keeps reads in bounds on tight rows at the last chunk) and
            // deinterleave u/v.
            let uv = _mm256_set_epi64x(
                0,
                0,
                0,
                ptr::read_unaligned(uv_row.add(x) as *const u64) as i64,
            );
            // u = even bytes, v = odd bytes (lanes 0..3 hold u0..u3, the rest
            // are zero from the 8-byte load).
            let u8 = _mm256_maddubs_epi16(uv, ev);
            let v8 = _mm256_maddubs_epi16(uv, od);
            // Pack the deinterleaved i16 back to bytes, duplicate each sample
            // onto its two luma pairs, widen -> 8 i32.
            let cb32 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(_mm256_shuffle_epi8(
                _mm256_castsi128_si256(_mm_packus_epi16(
                    _mm256_castsi256_si128(u8),
                    _mm256_extracti128_si256(u8, 1),
                )),
                dup,
            )));
            let cr32 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(_mm256_shuffle_epi8(
                _mm256_castsi128_si256(_mm_packus_epi16(
                    _mm256_castsi256_si128(v8),
                    _mm256_extracti128_si256(v8, 1),
                )),
                dup,
            )));

            q14_to_rgb(
                &ky, &rcr, &gcb, &gcr, &bcb, &rcp, &gcp, &bcp,
                y32, cb32, cr32, d8, x, if rgba { 4 } else { 3 },
            );
            x += 8;
        }

        while x < width {
            let yy = *y_row.add(x) as i32;
            // Luma x is covered by CbCr pair x / 2 => bytes (2 * (x / 2), +1).
            let uo = x & !1;
            let cbv = *uv_row.add(uo) as i32;
            let crv = *uv_row.add(uo + 1) as i32;
            let (r, g, b) = crate::conv::conv_px(c, yy, cbv, crv);
            let o = x * if rgba { 4 } else { 3 };
            *d8.add(o) = r;
            *d8.add(o + 1) = g;
            *d8.add(o + 2) = b;
            if rgba {
                *d8.add(o + 3) = 255;
            }
            x += 1;
        }
    }
}

// ─────────────── shared Q14 core + store ───────────────

#[target_feature(enable = "avx2")]
unsafe fn q14_to_rgb(
    ky: &__m256i,
    rcr: &__m256i,
    gcb: &__m256i,
    gcr: &__m256i,
    bcb: &__m256i,
    rcp: &__m256i,
    gcp: &__m256i,
    bcp: &__m256i,
    y32: __m256i,
    cb32: __m256i,
    cr32: __m256i,
    d8: *mut u8,
    x: usize,
    ch: usize,
) {
    let ky_y = _mm256_mullo_epi32(*ky, y32);
    let r = _mm256_add_epi32(
        _mm256_add_epi32(ky_y, _mm256_mullo_epi32(*rcr, cr32)),
        *rcp,
    );
    let g = _mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(*gcb, cb32), _mm256_mullo_epi32(*gcr, cr32)),
            ky_y,
        ),
        *gcp,
    );
    let b = _mm256_add_epi32(_mm256_add_epi32(ky_y, _mm256_mullo_epi32(*bcb, cb32)), *bcp);
    unsafe {
        store_rgb(&r, &g, &b, d8, x, ch);
    }
}

/// Store 8 saturated Q14 sums (post-`>> 14`) as `ch` bytes per pixel.
/// Saturation through i16 keeps clamp behavior identical to
/// `conv_px`'s `clamp(0, 255)`.
#[target_feature(enable = "avx2")]
unsafe fn store_rgb(r: &__m256i, g: &__m256i, b: &__m256i, d8: *mut u8, x: usize, ch: usize) {
    unsafe {
        macro_rules! px {
            ($k:literal) => {{
                let sel = |v: &__m256i| -> u8 {
                    // Arithmetic >>14 on two's-complement: matches conv_px's i32 shift.
                    (((_mm256_extract_epi32(*v, $k) as u32) as i32) >> 14)
                        .clamp(0, 255)
                        as u8
                };
                let o = (x + $k) * ch;
                *d8.add(o) = sel(r);
                *d8.add(o + 1) = sel(g);
                *d8.add(o + 2) = sel(b);
                if ch == 4 {
                    *d8.add(o + 3) = 255;
                }
            }};
        }
        px!(0);
        px!(1);
        px!(2);
        px!(3);
        px!(4);
        px!(5);
        px!(6);
        px!(7);
    }
}

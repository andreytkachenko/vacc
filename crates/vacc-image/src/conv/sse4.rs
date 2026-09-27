//! SSE4.1 row kernels: 8-bit Y'CbCr(4:2:0) -> RGB(x).
//!
//! All entry points are `unsafe`: they require the documented source
//! lengths; no alignment is required (unaligned 16-byte loads and 1-byte
//! reads). Kernel selection lives in `conv.rs`, behind runtime CPU feature
//! detection.
//!
//! 16 luma pixels are consumed per iteration (16-byte Y/UV load, in bounds
//! by construction), processed as 4-pixel quads in 4-lane i32 registers.
//! Planar chroma is read as 1-byte loads (always in bounds); semi-planar
//! UV in 16-byte blocks. Stores saturate per pixel and are extract-based
//! on purpose, so byte results are exactly the scalar reference
//! (`conv_px`).

use crate::coeff::{Conv8, RND};

use core::ptr;

use core::arch::x86_64::*;

/// Standard `_MM_SHUFFLE(1, 1, 0, 0)` semantics as stdarch's packed const
/// (selection for lane i at bits `2i`): `[a b . .] -> [a a b b . .]`.
const DUP2: i32 = 0x50;

/// Saturate 4 Q14-sum lanes and write one 4-pixel RGBA32 group
/// (`[R G B 0xFF] x 4`, little-endian) at `d8.add(o)`; `o` is a multiple
/// of 4 so the words are aligned.
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn store_rgba4(r: __m128i, g: __m128i, b: __m128i, d8: *mut u8, o: usize) {
    unsafe {
        let o = d8.add(o);
        macro_rules! px {
            ($k:literal) => {{
                let sv = |v: i32| ((v >> 14).clamp(0, 255)) as u32;
                let p = 0xFF00_0000u32
                    | (sv(_mm_extract_epi32(b, $k)) << 16)
                    | (sv(_mm_extract_epi32(g, $k)) << 8)
                    | sv(_mm_extract_epi32(r, $k));
                (o.add($k * 4) as *mut u32).write_unaligned(p);
            }};
        }
        px!(0);
        px!(1);
        px!(2);
        px!(3);
    }
}

/// Saturate 4 Q14-sum lanes and write one 4-pixel RGB24 group: the three
/// 4-byte destination words are `[R0 G0 B0 | R1]`, `[G1 B1 R2 G2]`,
/// `[B2 R3 G3 B3]` (little-endian). `o` is not generally 4-byte aligned,
/// hence `write_unaligned`.
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn store_rgb24_4(r: __m128i, g: __m128i, b: __m128i, d8: *mut u8, o: usize) {
    unsafe {
        let sat = |v: i32| ((v >> 14).clamp(0, 255)) as u32;
        let r0 = sat(_mm_extract_epi32(r, 0));
        let r1 = sat(_mm_extract_epi32(r, 1));
        let r2 = sat(_mm_extract_epi32(r, 2));
        let r3 = sat(_mm_extract_epi32(r, 3));
        let g0 = sat(_mm_extract_epi32(g, 0));
        let g1 = sat(_mm_extract_epi32(g, 1));
        let g2 = sat(_mm_extract_epi32(g, 2));
        let g3 = sat(_mm_extract_epi32(g, 3));
        let b0 = sat(_mm_extract_epi32(b, 0));
        let b1 = sat(_mm_extract_epi32(b, 1));
        let b2 = sat(_mm_extract_epi32(b, 2));
        let b3 = sat(_mm_extract_epi32(b, 3));
        let t0 = (r1 << 24) | (b0 << 16) | (g0 << 8) | r0;
        let t1 = (g2 << 24) | (r2 << 16) | (b1 << 8) | g1;
        let t2 = (b3 << 24) | (g3 << 16) | (r3 << 8) | b2;
        let o = d8.add(o);
        (o as *mut u32).write_unaligned(t0);
        (o.add(4) as *mut u32).write_unaligned(t1);
        (o.add(8) as *mut u32).write_unaligned(t2);
    }
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn chroma_quad(
    lo: u32,
    hi: u32,
    k: u32,
) -> __m128i {
    // k is 0..3; quad k (luma 4k..4k+3) uses chroma samples 2k and 2k+1.
    // Two samples per u32 word: lo covers samples 0..3, hi covers 4..7.
    // Sample `i` sits in the `(i % 2)`-th u32 word, byte-pair `(i / 2)`.
    let base = if k < 2 { lo } else { hi };
    let s = (k & 1) * 16;
    let c0_ = ((base >> s) & 0xff) as i32;
    let c1_ = ((base >> (s + 8)) & 0xff) as i32;
    _mm_setr_epi32(c0_, c0_, c1_, c1_)
}

/// One luma row (planar I420 layout) via SSE4.1.
#[target_feature(enable = "sse4.1")]
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
        let ky = _mm_set1_epi32(c.ky);
        let rcr = _mm_set1_epi32(c.r_cr);
        let gcb = _mm_set1_epi32(c.g_cb);
        let gcr = _mm_set1_epi32(c.g_cr);
        let bcb = _mm_set1_epi32(c.b_cb);
        let rcp = _mm_set1_epi32(c.r_off + RND);
        let gcp = _mm_set1_epi32(c.g_off + RND);
        let bcp = _mm_set1_epi32(c.b_off + RND);
        let d8 = dst_row.as_mut_ptr();

        let mut x = 0usize;
        while x + 16 <= width {
            let y16 = _mm_loadu_si128(y_row.add(x) as *const __m128i);
            // 8 chroma samples for the 16 luma pixels. `x / 2` is a multiple
            // of 8 and `x / 2 + 8 <= (w - 16) / 2 + 8 = w / 2 <= ceil(w / 2)`,
            // so one 8-byte read per plane stays in bounds on tight planes.
            let cb64 = ptr::read_unaligned(cb_row.add(x / 2) as *const u64);
            let cr64 = ptr::read_unaligned(cr_row.add(x / 2) as *const u64);
            let cb_lo = cb64 as u32;
            let cb_hi = (cb64 >> 32) as u32;
            let cr_lo = cr64 as u32;
            let cr_hi = (cr64 >> 32) as u32;

            macro_rules! quad {
                ($k:literal) => {{
                    let xq = x as u32 + 4 * $k;
                    let yq = _mm_srli_si128(y16, (4 * $k) as i32);
                    let y4 = _mm_cvtepi16_epi32(_mm_cvtepu8_epi16(yq));

                    let c4 = chroma_quad(cb_lo, cb_hi, $k);
                    let r4 = chroma_quad(cr_lo, cr_hi, $k);

                    // Q14 multiply-adds (i32, |sum| < 2^31).
                    let ky_y = _mm_mullo_epi32(ky, y4);
                    let r4v = _mm_add_epi32(
                        _mm_add_epi32(ky_y, _mm_mullo_epi32(rcr, r4)),
                        rcp,
                    );
                    let g4 = _mm_add_epi32(
                        _mm_add_epi32(
                            _mm_add_epi32(_mm_mullo_epi32(gcb, c4), _mm_mullo_epi32(gcr, r4)),
                            ky_y,
                        ),
                        gcp,
                    );
                    let b4 = _mm_add_epi32(
                        _mm_add_epi32(ky_y, _mm_mullo_epi32(bcb, c4)),
                        bcp,
                    );

                    if rgba {
                        store_rgba4(r4v, g4, b4, d8, (xq as usize) * 4);
                    } else {
                        store_rgb24_4(r4v, g4, b4, d8, (xq as usize) * 3);
                    }
                }};
            }
            quad!(0);
            quad!(1);
            quad!(2);
            quad!(3);
            x += 16;
        }

        // Scalar tail (width % 16 pixels).
        while x < width {
            let yy = *y_row.add(x) as i32;
            let cbv = *cb_row.add(x >> 1) as i32;
            let crv = *cr_row.add(x >> 1) as i32;
            let (r, g, b) = crate::conv::conv_px(c, yy, cbv, crv);
            if rgba {
                *d8.add(x * 4) = r;
                *d8.add(x * 4 + 1) = g;
                *d8.add(x * 4 + 2) = b;
                *d8.add(x * 4 + 3) = 255;
            } else {
                *d8.add(x * 3) = r;
                *d8.add(x * 3 + 1) = g;
                *d8.add(x * 3 + 2) = b;
            }
            x += 1;
        }
    }
}

/// One luma row (semi-planar NV12 layout) via SSE4.1.
#[target_feature(enable = "sse4.1")]
pub unsafe fn semi_row(
    y_row: *const u8,
    uv_row: *const u8,
    width: usize,
    c: &Conv8,
    dst_row: &mut [u8],
) {
    unsafe {
        let rgba = dst_row.len() == width * 4;
        let ky = _mm_set1_epi32(c.ky);
        let rcr = _mm_set1_epi32(c.r_cr);
        let gcb = _mm_set1_epi32(c.g_cb);
        let gcr = _mm_set1_epi32(c.g_cr);
        let bcb = _mm_set1_epi32(c.b_cb);
        let rcp = _mm_set1_epi32(c.r_off + RND);
        let gcp = _mm_set1_epi32(c.g_off + RND);
        let bcp = _mm_set1_epi32(c.b_off + RND);
        let d8 = dst_row.as_mut_ptr();
        let p10 = _mm_set1_epi16(0x0001);
        let p01 = _mm_set1_epi16(0x0100);

        let mut x = 0usize;
        while x + 16 <= width {
            let y16 = _mm_loadu_si128(y_row.add(x) as *const __m128i);
            // 16 bytes = 8 interleaved (u, v) pairs for the 16 luma pixels.
            let uv16 = _mm_loadu_si128(uv_row.add(x) as *const __m128i);

            macro_rules! quad {
                ($k:literal) => {{
                    let xq = x as u32 + 4 * $k;
                    let yq = _mm_srli_si128(y16, (4 * $k) as i32);
                    let y4 = _mm_cvtepi16_epi32(_mm_cvtepu8_epi16(yq));

                    // Two (u, v) pairs for the quad: luma 4k..4k+3 uses chroma
                    // pairs 2k..2k+1, i.e. raw bytes 4k..4k+5 of `uv16`.
                    // Sliding by 4k bytes zero-fills the front, so `maddubs`
                    // puts sample 2k in i16 lane 0 and 2k+1 in lane 1; widen
                    // the low lanes and duplicate each sample onto its two
                    // luma pairs.
                    let uvp = _mm_srli_si128(uv16, (4 * $k) as i32);
                    let u_i16 = _mm_maddubs_epi16(uvp, p10);
                    let v_i16 = _mm_maddubs_epi16(uvp, p01);
                    let c4 = _mm_shuffle_epi32(_mm_cvtepi16_epi32(u_i16), DUP2);
                    let r4 = _mm_shuffle_epi32(_mm_cvtepi16_epi32(v_i16), DUP2);

                    // Q14 multiply-adds (i32, |sum| < 2^31).
                    let ky_y = _mm_mullo_epi32(ky, y4);
                    let r4v = _mm_add_epi32(
                        _mm_add_epi32(ky_y, _mm_mullo_epi32(rcr, r4)),
                        rcp,
                    );
                    let g4 = _mm_add_epi32(
                        _mm_add_epi32(
                            _mm_add_epi32(
                                _mm_mullo_epi32(gcb, c4),
                                _mm_mullo_epi32(gcr, r4),
                            ),
                            ky_y,
                        ),
                        gcp,
                    );
                    let b4 = _mm_add_epi32(
                        _mm_add_epi32(ky_y, _mm_mullo_epi32(bcb, c4)),
                        bcp,
                    );

                    if rgba {
                        store_rgba4(r4v, g4, b4, d8, (xq as usize) * 4);
                    } else {
                        store_rgb24_4(r4v, g4, b4, d8, (xq as usize) * 3);
                    }
                }};
            }
            quad!(0);
            quad!(1);
            quad!(2);
            quad!(3);
            x += 16;
        }

        while x < width {
            let yy = *y_row.add(x) as i32;
            // Luma x is covered by CbCr pair x / 2 => bytes (2 * (x / 2), +1).
            let uo = x & !1;
            let cbv = *uv_row.add(uo) as i32;
            let crv = *uv_row.add(uo + 1) as i32;
            let (r, g, b) = crate::conv::conv_px(c, yy, cbv, crv);
            if rgba {
                *d8.add(x * 4) = r;
                *d8.add(x * 4 + 1) = g;
                *d8.add(x * 4 + 2) = b;
                *d8.add(x * 4 + 3) = 255;
            } else {
                *d8.add(x * 3) = r;
                *d8.add(x * 3 + 1) = g;
                *d8.add(x * 3 + 2) = b;
            }
            x += 1;
        }
    }
}

#![allow(non_snake_case)]

//! Port of the edge264 residual kernels — dequant + IDCT (Tier B). Bit-exact
//! with c/src/edge264_residual.c (8-bit path only; >8-bit is a no-op there).
//!
//! The C code operates on 128-bit vectors; the port re-expresses the same
//! arithmetic. Both IDCT passes apply the 1D butterfly PER LANE across the
//! coefficient vectors (columns of the matrix), verified against C dumps:
//!   pass1[i][j] = butterfly([d[0][j], d[1][j], ...])[i]
//! then a transpose (plain matrix transpose for both 4x4 and 8x8) with +32
//! on the first row, then pass2 in the same shape (the C 8x8 loop breaks
//! BEFORE a second transpose). Wrap/saturation points mirror C exactly: dequant byte products are UNSIGNED (C `cvtlo8u16`); transform
//! math wraps in i32 (4x4) / i16 (8x8); 4x4 final residual saturates
//! i32->i16 (C `packs32`) after >> 6; pixel add wraps in i16 then saturates
//! to u8 (C `packus16`).

/// Residual state for one macroblock: unscaled coefficients in zigzag order,
/// scaling lists, per-plane QP, and the mb inter flag (selects the list set).
#[derive(Clone, Copy)]
pub struct Residual {
    pub c: [i32; 64],
    pub qp: [u8; 3],
    /// weightScale4x4: slot = iYCbCr + inter*3 (C `weightScale4x4_v`).
    pub ws4: [[i8; 16]; 6],
    /// weightScale8x8: slot = iYCbCr*2 + inter, 64 bytes each.
    pub ws8: [[i8; 64]; 6],
    pub inter: bool,
}

/// C `normAdjust4x4` (edge264_residual.c:77).
const NORM_ADJUST_4X4: [[i8; 16]; 6] = [
    [
        10, 13, 10, 13, 13, 16, 13, 16, 10, 13, 10, 13, 13, 16, 13, 16,
    ],
    [
        11, 14, 11, 14, 14, 18, 14, 18, 11, 14, 11, 14, 14, 18, 14, 18,
    ],
    [
        13, 16, 13, 16, 16, 20, 16, 20, 13, 16, 13, 16, 16, 20, 16, 20,
    ],
    [
        14, 18, 14, 18, 18, 23, 18, 23, 14, 18, 14, 18, 18, 23, 18, 23,
    ],
    [
        16, 20, 16, 20, 20, 25, 20, 25, 16, 20, 16, 20, 20, 25, 20, 25,
    ],
    [
        18, 23, 18, 23, 23, 29, 23, 29, 18, 23, 18, 23, 23, 29, 23, 29,
    ],
];

/// C `normAdjust8x8` (edge264_residual.c:85); row m*2(+1) for qP%6 = m.
const NORM_ADJUST_8X8: [[i8; 16]; 12] = [
    [
        20, 19, 25, 19, 20, 19, 25, 19, 19, 18, 24, 18, 19, 18, 24, 18,
    ],
    [
        25, 24, 32, 24, 25, 24, 32, 24, 19, 18, 24, 18, 19, 18, 24, 18,
    ],
    [
        22, 21, 28, 21, 22, 21, 28, 21, 21, 19, 26, 19, 21, 19, 26, 19,
    ],
    [
        28, 26, 35, 26, 28, 26, 35, 26, 21, 19, 26, 19, 21, 19, 26, 19,
    ],
    [
        26, 24, 33, 24, 26, 24, 33, 24, 24, 23, 31, 23, 24, 23, 31, 23,
    ],
    [
        33, 31, 42, 31, 33, 31, 42, 31, 24, 23, 31, 23, 24, 23, 31, 23,
    ],
    [
        28, 26, 35, 26, 28, 26, 35, 26, 26, 25, 33, 25, 26, 25, 33, 25,
    ],
    [
        35, 33, 45, 33, 35, 33, 45, 33, 26, 25, 33, 25, 26, 25, 33, 25,
    ],
    [
        32, 30, 40, 30, 32, 30, 40, 30, 30, 28, 38, 28, 30, 28, 38, 28,
    ],
    [
        40, 38, 51, 38, 40, 38, 51, 38, 30, 28, 38, 28, 30, 28, 38, 28,
    ],
    [
        36, 34, 46, 34, 36, 34, 46, 34, 34, 32, 43, 32, 34, 32, 43, 32,
    ],
    [
        46, 43, 58, 43, 46, 43, 58, 43, 34, 32, 43, 32, 34, 32, 43, 32,
    ],
];

/// C `packs32` lane: saturating i32 -> i16.
#[inline]
fn sat16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// C `packus16` lane: saturating i16 -> u8.
#[inline]
fn sat8(v: i16) -> u8 {
    v.clamp(0, i16::from(u8::MAX)) as u8
}

/// 4-point IDCT butterfly (C e0..e3/f0..f3): output components 0..3.
#[inline]
fn butterfly4(v: [i32; 4]) -> [i32; 4] {
    let a = v[0].wrapping_add(v[2]);
    let b = v[0].wrapping_sub(v[2]);
    let c = (v[1] >> 1).wrapping_sub(v[3]);
    let e = (v[3] >> 1).wrapping_add(v[1]);
    [
        a.wrapping_add(e),
        b.wrapping_add(c),
        b.wrapping_sub(c),
        a.wrapping_sub(e),
    ]
}

/// 8-point IDCT butterfly (C e0..e7/f0..f7): output components 0..7.
#[inline]
fn butterfly8(v: [i16; 8]) -> [i16; 8] {
    let (v0, v1, v2, v3, v4, v5, v6, v7) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    let e0 = v0.wrapping_add(v4);
    let e1 = v5.wrapping_sub(v3).wrapping_sub((v7 >> 1).wrapping_add(v7));
    let e2 = v0.wrapping_sub(v4);
    let e3 = v1.wrapping_add(v7).wrapping_sub((v3 >> 1).wrapping_add(v3));
    let e4 = (v2 >> 1).wrapping_sub(v6);
    let e5 = v7.wrapping_sub(v1).wrapping_add((v5 >> 1).wrapping_add(v5));
    let e6 = (v6 >> 1).wrapping_add(v2);
    let e7 = v3.wrapping_add(v5).wrapping_add((v1 >> 1).wrapping_add(v1));
    let f0 = e0.wrapping_add(e6);
    let f1 = (e7 >> 2).wrapping_add(e1);
    let f2 = e2.wrapping_add(e4);
    let f3 = (e5 >> 2).wrapping_add(e3);
    let f4 = e2.wrapping_sub(e4);
    let f5 = (e3 >> 2).wrapping_sub(e5);
    let f6 = e0.wrapping_sub(e6);
    let f7 = e7.wrapping_sub(e1 >> 2);
    [
        f0.wrapping_add(f7),
        f2.wrapping_add(f5),
        f4.wrapping_add(f3),
        f6.wrapping_add(f1),
        f6.wrapping_sub(f1),
        f4.wrapping_sub(f3),
        f2.wrapping_sub(f5),
        f0.wrapping_sub(f7),
    ]
}

/// Pass shape: apply the butterfly per lane across the 4 vectors.
#[inline]
fn pass4(d: &[[i32; 4]; 4]) -> [[i32; 4]; 4] {
    let mut f = [[0i32; 4]; 4];
    for j in 0..4 {
        let o = butterfly4([d[0][j], d[1][j], d[2][j], d[3][j]]);
        for i in 0..4 {
            f[i][j] = o[i];
        }
    }
    f
}

/// Pass shape for 8x8: apply the butterfly per lane across the 8 vectors.
#[inline]
fn pass8(d: &[[i16; 8]; 8]) -> [[i16; 8]; 8] {
    let mut f = [[0i16; 8]; 8];
    for j in 0..8 {
        let o = butterfly8([
            d[0][j], d[1][j], d[2][j], d[3][j], d[4][j], d[5][j], d[6][j], d[7][j],
        ]);
        for i in 0..8 {
            f[i][j] = o[i];
        }
    }
    f
}

/// The 8x8 zip cascade (C x0..xF) is a plain matrix transpose:
/// t[i][k] = f[k][i]. Verified against C dumps (dbg_res8).
#[inline]
fn perm8(f: &[[i16; 8]; 8]) -> [[i16; 8]; 8] {
    let mut t = [[0i16; 8]; 8];
    for i in 0..8 {
        for k in 0..8 {
            t[i][k] = f[k][i];
        }
    }
    t
}

// --- DC-transform lane helpers (Tier E0) — C SSE semantics, i32x4 ---

#[inline]
fn add4(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
    [
        a[0].wrapping_add(b[0]),
        a[1].wrapping_add(b[1]),
        a[2].wrapping_add(b[2]),
        a[3].wrapping_add(b[3]),
    ]
}

#[inline]
fn sub4(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
    [
        a[0].wrapping_sub(b[0]),
        a[1].wrapping_sub(b[1]),
        a[2].wrapping_sub(b[2]),
        a[3].wrapping_sub(b[3]),
    ]
}

/// C `ziplo32` (i32x4): [a0, b0, a1, b1].
#[inline]
fn ziplo4(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
    [a[0], b[0], a[1], b[1]]
}

/// C `ziphi32` (i32x4): [a2, b2, a3, b3].
#[inline]
fn ziphi4(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
    [a[2], b[2], a[3], b[3]]
}

/// C `ziplo64` (i32x4): [a0, a1, b0, b1].
#[inline]
fn ziplo64(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
    [a[0], a[1], b[0], b[1]]
}

/// C `ziphi64` (i32x4): [a2, a3, b2, b3].
#[inline]
fn ziphi64(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
    [a[2], a[3], b[2], b[3]]
}

/// C `unziplo32` = shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0)): [a0, a2, b0, b2].
#[inline]
fn unziplo4(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
    [a[0], a[2], b[0], b[2]]
}

/// C `unziphi32` = shuffle_ps(a, b, _MM_SHUFFLE(3,1,3,1)): [a1, a3, b1, b3].
#[inline]
fn unziphi4(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
    [a[1], a[3], b[1], b[3]]
}

#[inline]
fn map4(f: &[i32; 4], mut g: impl FnMut(i32) -> i32) -> [i32; 4] {
    [g(f[0]), g(f[1]), g(f[2]), g(f[3])]
}

/// C `add_idct4x4` (edge264_residual.c:108): dequant+IDCT c[0..15] and add to
/// the 4x4 block at the start of `pix` (rows at r*stride). If `dcidx >= 0`,
/// the dequantized DC is replaced by the raw c[16+dcidx]. Zeroes c[0..15] on
/// exit. `ws` is the active scaling list (C indexes
/// `weightScale4x4_v[iYCbCr + mbIsInterFlag * 3]`; caller resolves it).
///
/// # Safety
/// `pix` must point at a block origin with `4*stride` bytes addressable
/// (rows 0..3, 4 samples each), as with the C kernel's raw pointer use.
pub unsafe fn add_idct4x4(
    c: &mut [i32; 64],
    qp: u8,
    ws: &[i8; 16],
    dcidx: i32,
    pix: *mut u8,
    stride: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            unsafe {
                return sse::add_idct4x4_sse(c, qp, ws, dcidx, pix, stride);
            }
        }
    }
    unsafe {
        add_idct4x4_scalar(c, qp, ws, dcidx, pix, stride)
    }
}

/// Scalar fallback for [`add_idct4x4`].
#[inline]
pub(crate) unsafe fn add_idct4x4_scalar(
    c: &mut [i32; 64],
    qp: u8,
    ws: &[i8; 16],
    dcidx: i32,
    pix: *mut u8,
    stride: usize,
) {
    let qpu = qp as u32;
    let sh = qpu / 6;
    let na = NORM_ADJUST_4X4[(qpu % 6) as usize];
    // Dequant: ((c * LS) << sh + 8) >> 4 in i32 (C `shlrrs32`, wrapping).
    let mut d = [[0i32; 4]; 4];
    for (k, &coeff) in c[..16].iter().enumerate() {
        // C multiplies the bytes unsignedly (cvtlo8u16); product fits i16.
        let ls = (ws[k] as u8 as u16 * na[k] as u16) as i32;
        d[k / 4][k % 4] = (coeff.wrapping_mul(ls).wrapping_shl(sh).wrapping_add(8)) >> 4;
    }
    c[..16].fill(0);
    if dcidx >= 0 {
        d[0][0] = c[16 + dcidx as usize];
    }
    // pass1 -> transpose (+32 on first row) -> pass2.
    let f = pass4(&d);
    let mut t = [[0i32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            t[i][j] = f[j][i];
        }
    }
    for v in t[0].iter_mut() {
        *v = v.wrapping_add(32);
    }
    let h = pass4(&t);
    // Pixel row i, col j += saturating(h[i][j] >> 6), clamped to u8.
    for (i, hrow) in h.iter().enumerate() {
        let row = unsafe { std::slice::from_raw_parts_mut(pix.add(i * stride), 4) };
        for (px, &hv) in row.iter_mut().zip(hrow.iter()) {
            *px = sat8((u16::from(*px) as i16).wrapping_add(sat16(hv >> 6)));
        }
    }
}

/// C `add_dc4x4` (edge264_residual.c:174): broadcast the DC residual over a
/// 4x4 block at the start of `pix` (rows at r*stride). Does NOT touch c
/// (unlike add_idct4x4). `dcidx` must be in 0..=7 (indexes `c[16 + dcidx]`).
///
/// # Safety
/// `pix` must point at a block origin with rows 0..3 addressable, as with the
/// C kernel's raw pointer use.
pub unsafe fn add_dc4x4(c: &[i32; 64], dcidx: i32, pix: *mut u8, stride: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            unsafe {
                return sse::add_dc4x4_sse(c, dcidx, pix, stride);
            }
        }
    }
    unsafe {
        add_dc4x4_scalar(c, dcidx, pix, stride)
    }
}

/// Scalar fallback for [`add_dc4x4`].
#[inline]
pub(crate) unsafe fn add_dc4x4_scalar(c: &[i32; 64], dcidx: i32, pix: *mut u8, stride: usize) {
    // C `set16((c + 32) >> 6)`: i32 shift, then truncated (not saturated) to i16.
    let r = ((c[16 + dcidx as usize]).wrapping_add(32) >> 6) as i16;
    for i in 0..4 {
        let row = unsafe { std::slice::from_raw_parts_mut(pix.add(i * stride), 4) };
        for px in row.iter_mut() {
            *px = sat8((u16::from(*px) as i16).wrapping_add(r));
        }
    }
}

/// C `add_idct8x8` (edge264_residual.c:194): dequant+IDCT c[0..63] and add to
/// the 8x8 block at the start of `pix` (rows at r*stride). Zeroes all of c on
/// exit. `ws` is the active scaling list (C indexes
/// `weightScale8x8_v[iYCbCr * 2 + mbIsInterFlag]`; caller resolves it).
///
/// # Safety
/// `pix` must point at a block origin with `8*stride` bytes addressable, as
/// with the C kernel's raw pointer use.
pub unsafe fn add_idct8x8(c: &mut [i32; 64], qp: u8, ws: &[i8; 64], pix: *mut u8, stride: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            unsafe {
                return sse::add_idct8x8_sse(c, qp, ws, pix, stride);
            }
        }
    }
    unsafe {
        add_idct8x8_scalar(c, qp, ws, pix, stride)
    }
}

/// Scalar fallback for [`add_idct8x8`].
#[inline]
pub(crate) unsafe fn add_idct8x8_scalar(
    c: &mut [i32; 64],
    qp: u8,
    ws: &[i8; 64],
    pix: *mut u8,
    stride: usize,
) {
    let qpu = qp as u32;
    let div = qpu / 6;
    let m = (qpu % 6) as usize;
    let na0 = NORM_ADJUST_8X8[m * 2];
    let na1 = NORM_ADJUST_8X8[m * 2 + 1];
    // Dequant (C `scale32`), saturated to i16. Coeff k belongs to vector
    // block=k/8, lane k%8; norm row alternates na0/na1 per pair of blocks.
    let mut d = [[0i16; 8]; 8];
    if div < 6 {
        let off = 1i32 << (5 - div);
        let sh = 6 - div;
        for k in 0..64 {
            let block = k / 8;
            let na = [na0, na1, na0, na1][block / 2];
            let ls = (ws[k] as u8 as u16 * na[k - (block / 2) * 16] as u16) as i32;
            d[block][k % 8] = sat16((ls.wrapping_mul(c[k]).wrapping_add(off)) >> sh);
        }
    } else {
        let sh = div - 6;
        for k in 0..64 {
            let block = k / 8;
            let na = [na0, na1, na0, na1][block / 2];
            let ls = (ws[k] as u8 as u16 * na[k - (block / 2) * 16] as u16) as i16;
            // C: packs32(c) * (LS << sh) — all i16 wraparound.
            d[block][k % 8] = sat16(c[k]).wrapping_mul(ls.wrapping_shl(sh));
        }
    }
    c.fill(0);
    // pass1 -> transpose (+32 on first row) -> pass2. The C loop breaks
    // BEFORE the transpose on the second iteration, so no final perm8.
    let f = pass8(&d);
    let mut t = perm8(&f);
    for v in t[0].iter_mut() {
        *v = v.wrapping_add(32);
    }
    let d = pass8(&t);
    // Final residual: row i, col j += d[i][j] >> 6 (arithmetic), clamped.
    for (i, drow) in d.iter().enumerate() {
        let row = unsafe { std::slice::from_raw_parts_mut(pix.add(i * stride), 8) };
        for (px, &dv) in row.iter_mut().zip(drow.iter()) {
            *px = sat8((u16::from(*px) as i16).wrapping_add(dv >> 6));
        }
    }
}

/// C `transform_dc4x4` (edge264_residual.c:352) — 16x16 luma DC transform.
/// Reads c[0..16] (parsed DC block, column-major), zeroes it, then either
/// stores the 16 first-stage DC values into c[16..32] (`store_later`; C
/// guards on `mb->bits[0] & 1<<5` — the second (c+32)>>6 stage is then
/// done by add_idct4x4/add_dc4x4) or adds them directly to the predicted
/// 16x16 block at the start of `pix` (rows at r*stride). C uses QP[0] for
/// every plane ("FIXME 4:4:4") and the INTRA list (`weightScale4x4[iYCbCr]`).
///
/// # Safety
/// When `!store_later`, `pix` must point at a block origin with `16*stride`
/// bytes addressable, as with the C kernel's raw pointer use.
pub unsafe fn transform_dc4x4(
    c: &mut [i32; 64],
    ws: &[i8; 16],
    qp: u8,
    store_later: bool,
    pix: *mut u8,
    stride: usize,
) {
    // C c_v[k] = c[4k..4k+4]
    let cv0 = [c[0], c[1], c[2], c[3]];
    let cv1 = [c[4], c[5], c[6], c[7]];
    let cv2 = [c[8], c[9], c[10], c[11]];
    let cv3 = [c[12], c[13], c[14], c[15]];
    // multiply right
    let x0 = add4(&cv0, &cv1);
    let x1 = add4(&cv2, &cv3);
    let x2 = sub4(&cv0, &cv1);
    let x3 = sub4(&cv2, &cv3);
    c[..16].fill(0);
    let x4 = add4(&x0, &x1);
    let x5 = sub4(&x0, &x1);
    let x6 = sub4(&x2, &x3);
    let x7 = add4(&x2, &x3);
    // transpose (C zip cascade)
    let x8 = ziplo4(&x4, &x5);
    let x9 = ziplo4(&x6, &x7);
    let xa = ziphi4(&x4, &x5);
    let xb = ziphi4(&x6, &x7);
    let xc = ziplo64(&x8, &x9);
    let xd = ziphi64(&x8, &x9);
    let xe = ziplo64(&xa, &xb);
    let xf = ziphi64(&xa, &xb);
    // multiply left
    let xg = add4(&xc, &xd);
    let xh = add4(&xe, &xf);
    let xi = sub4(&xc, &xd);
    let xj = sub4(&xe, &xf);
    let f0 = add4(&xg, &xh);
    let f1 = sub4(&xg, &xh);
    let f2 = sub4(&xi, &xj);
    let f3 = add4(&xi, &xj);
    // scale: LS = ws[0] * normAdjust4x4[qp%6][0] << (qp/6). C uses the
    // UNSIGNED byte value (promotion to int), so no sign extension.
    let qpu = qp as u32;
    let ls = ((ws[0] as u8 as u32) * (NORM_ADJUST_4X4[(qpu % 6) as usize][0] as u32)) << (qpu / 6);
    // first stage: dc_k = (f_k * LS + 32) >> 6 (C shrrs32, wrapping)
    let ls = ls as i32;
    let dc = [
        map4(&f0, |v| (v.wrapping_mul(ls).wrapping_add(32)) >> 6),
        map4(&f1, |v| (v.wrapping_mul(ls).wrapping_add(32)) >> 6),
        map4(&f2, |v| (v.wrapping_mul(ls).wrapping_add(32)) >> 6),
        map4(&f3, |v| (v.wrapping_mul(ls).wrapping_add(32)) >> 6),
    ];
    if store_later {
        // C: c_v[4..7] = ziplo/hi64 cascade of dc0..dc3
        c[16..20].copy_from_slice(&ziplo64(&dc[0], &dc[1]));
        c[20..24].copy_from_slice(&ziphi64(&dc[0], &dc[1]));
        c[24..28].copy_from_slice(&ziplo64(&dc[2], &dc[3]));
        c[28..32].copy_from_slice(&ziphi64(&dc[2], &dc[3]));
    } else {
        // second stage r = (dc + 32) >> 6, truncated to i16 by the C
        // broadcast. Row group g (subblock row g): byte j of each 16-byte
        // row += r[g][j/4] (broadcastlo32(r) = [r[0]x4, r[1]x4] on the low
        // 8 bytes, broadcasthi32(r) = [r[2]x4, r[3]x4] on the high 8).
        for (g, dc_g) in dc.iter().enumerate() {
            let r = map4(dc_g, |v| v.wrapping_add(32) >> 6);
            for row in 0..4 {
                let base = (4 * g + row) * stride;
                let row_slice = unsafe { std::slice::from_raw_parts_mut(pix.add(base), 16) };
                for (j, px) in row_slice.iter_mut().enumerate() {
                    *px = sat8((u16::from(*px) as i16).wrapping_add(r[j / 4] as i16));
                }
            }
        }
    }
}

/// C `transform_dc2x2` (edge264_residual.c:456) — chroma DC transform.
/// c[0..8] holds the Cb/Cr 2x2 DC blocks lane-interleaved
/// ([cb_a, cr_a, cb_b, cr_b | cb_c, cr_c, cb_d, cr_d]); zeroes it, then
/// either stores the 4 scaled Cb values into c[16..20] and Cr into
/// c[20..24] (`store_later`; C guards on `CodedBlockPatternChromaAC`) or
/// adds them to the predicted chroma samples at the start of `pix`: 16
/// rows of 8 bytes at row stride `stride >> 1`, Cb rows and Cr rows
/// alternating (see chroma_plane_layout). C uses QP[1]/QP[2] and
/// `weightScale4x4[1 + mbIsInterFlag * 3]` / `[2 + ...]`.
///
/// # Safety
/// When `!store_later`, `pix` must point at the MB's Cb top-left with 8 rows
/// of 8 bytes addressable (row stride `stride >> 1`), as with the C kernel's
/// raw pointer use.
pub unsafe fn transform_dc2x2(
    c: &mut [i32; 64],
    ws4: &[[i8; 16]; 6],
    inter: bool,
    // C `QP` array (3 elements); callers may pass longer arrays — only
    // `qp[1]`/`qp[2]` are read.
    qp: &[u8],
    store_later: bool,
    pix: *mut u8,
    stride: usize,
) {
    // C indexes `weightScale4x4[1 + mbIsInterFlag * 3]` / `[2 + ...]` and
    // QP[1]/QP[2] internally.
    let i3 = inter as usize * 3;
    let (ws_b, ws_r) = (&ws4[1 + i3], &ws4[2 + i3]);
    let (qp_b, qp_r) = (qp[1], qp[2]);
    // C c_v[k] = c[4k..4k+4]
    let cv0 = [c[0], c[1], c[2], c[3]];
    let cv1 = [c[4], c[5], c[6], c[7]];
    // multiply right (Cb/Cr stay lane-interleaved)
    let d0 = add4(&cv0, &cv1);
    let d1 = sub4(&cv0, &cv1);
    c[..8].fill(0);
    // transpose and multiply left
    let e0 = ziplo64(&d0, &d1);
    let e1 = ziphi64(&d0, &d1);
    let f0 = add4(&e0, &e1);
    let f1 = sub4(&e0, &e1);
    // deinterlace and scale. weightScale4x4 is uint8_t (unsigned byte values).
    let qpbu = qp_b as u32;
    let qpru = qp_r as u32;
    let lsb_ =
        ((ws_b[0] as u8 as u32) * (NORM_ADJUST_4X4[(qpbu % 6) as usize][0] as u32)) << (qpbu / 6);
    let lsr_ =
        ((ws_r[0] as u8 as u32) * (NORM_ADJUST_4X4[(qpru % 6) as usize][0] as u32)) << (qpru / 6);
    // plain wrapping >> 5 (no rounding), unlike the 4x4 path.
    let (lsb_, lsr_) = (lsb_ as i32, lsr_ as i32);
    let dc_cb = map4(&unziplo4(&f0, &f1), |v| v.wrapping_mul(lsb_) >> 5);
    let dc_cr = map4(&unziphi4(&f0, &f1), |v| v.wrapping_mul(lsr_) >> 5);
    if store_later {
        c[16..20].copy_from_slice(&dc_cb);
        c[20..24].copy_from_slice(&dc_cr);
    } else {
        // Second stage rb/rr = (dc + 32) >> 6, then the store: group q
        // covers rows q*4..q*4+3; Cb rows {0,2} get [v_lo x4, v_hi x4],
        // Cr rows {1,3} likewise. Groups 0-1 use rb[0]/rb[1] (lor),
        // groups 2-3 use rb[2]/rb[3] (hir). C's packus16 concatenates
        // (low 8 bytes from the first operand), so each row is written
        // contiguously.
        let rb = map4(&dc_cb, |v| v.wrapping_add(32) >> 6);
        let rr = map4(&dc_cr, |v| v.wrapping_add(32) >> 6);
        let s = stride >> 1;
        for q in 0..4 {
            let (b_lo, b_hi) = if q < 2 {
                (rb[0], rb[1])
            } else {
                (rb[2], rb[3])
            };
            let (r_lo, r_hi) = if q < 2 {
                (rr[0], rr[1])
            } else {
                (rr[2], rr[3])
            };
            for k in 0..4 {
                let (v_lo, v_hi) = if k % 2 == 0 {
                    (b_lo, b_hi)
                } else {
                    (r_lo, r_hi)
                };
                let base = (q * 4 + k) * s;
                let row_slice = unsafe { std::slice::from_raw_parts_mut(pix.add(base), 8) };
                for (j, px) in row_slice.iter_mut().enumerate() {
                    let v = if j < 4 { v_lo } else { v_hi };
                    *px = sat8((u16::from(*px) as i16).wrapping_add(v as i16));
                }
            }
        }
    }
}

impl Residual {
    pub fn add_idct4x4(&mut self, plane: usize, dcidx: i32, pix: &mut [u8], stride: usize) {
        unsafe {
            add_idct4x4(
                &mut self.c,
                self.qp[plane],
                &self.ws4[plane + self.inter as usize * 3],
                dcidx,
                pix.as_mut_ptr(),
                stride,
            )
        }
    }

    pub fn add_dc4x4(&mut self, _plane: usize, dcidx: i32, pix: &mut [u8], stride: usize) {
        unsafe { add_dc4x4(&self.c, dcidx, pix.as_mut_ptr(), stride) }
    }

    pub fn add_idct8x8(&mut self, plane: usize, pix: &mut [u8], stride: usize) {
        unsafe {
            add_idct8x8(
                &mut self.c,
                self.qp[plane],
                &self.ws8[plane * 2 + self.inter as usize],
                pix.as_mut_ptr(),
                stride,
            )
        }
    }

    pub fn transform_dc4x4(
        &mut self,
        iYCbCr: usize,
        store_later: bool,
        pix: &mut [u8],
        stride: usize,
    ) {
        // C `transform_dc4x4` uses QP[0] ("FIXME 4:4:4") and the INTRA list.
        unsafe {
            transform_dc4x4(
                &mut self.c,
                &self.ws4[iYCbCr],
                self.qp[0],
                store_later,
                pix.as_mut_ptr(),
                stride,
            )
        }
    }

    pub fn transform_dc2x2(&mut self, store_later: bool, pix: &mut [u8], stride: usize) {
        unsafe {
            transform_dc2x2(
                &mut self.c,
                &self.ws4,
                self.inter,
                &self.qp,
                store_later,
                pix.as_mut_ptr(),
                stride,
            )
        }
    }
}

// --- SSE4.1 ports (x86_64) — bit-exact with the scalar kernels above; C
// reference edge264_residual.c. Unlike C's `scale32` madd shortcut, the
// dequant keeps the full i32 coefficient (goldens pin wrapping_mul; fuzz
// drives i32::MAX). ---

#[cfg(target_arch = "x86_64")]
mod sse {
    use super::*;
    use core::arch::x86_64::*;

    /// Keep i16 lanes 0..3 (bytes 0..7), zero the rest.
    #[inline]
    fn lo4_mask() -> __m128i {
        unsafe { _mm_setr_epi32(-1, -1, 0, 0) }
    }

    /// Zero-extend the low 4 bytes -> i16x8 [b0,b1,b2,b3,0,0,0,0].
    #[inline]
    fn ext4(v: __m128i) -> __m128i {
        unsafe { _mm_unpacklo_epi8(v, _mm_setzero_si128()) }
    }

    /// C `mullou8`: unsigned byte products of the low halves -> i16x8 lanes
    /// 0..7 (high 4 lanes zero). Raw `_mm_mullo_epi16` on the byte vectors
    /// would multiply 16-bit pairs instead — NOT equivalent.
    #[inline]
    fn byte_mul_lo(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let z = _mm_setzero_si128();
            _mm_mullo_epi16(_mm_unpacklo_epi8(a, z), _mm_unpacklo_epi8(b, z))
        }
    }

    /// C `mulhiu8`: unsigned byte products of the high halves -> i16x8 lanes
    /// 0..7 (high 4 lanes zero).
    #[inline]
    fn byte_mul_hi(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let z = _mm_setzero_si128();
            _mm_mullo_epi16(_mm_unpackhi_epi8(a, z), _mm_unpackhi_epi8(b, z))
        }
    }

    /// Zero-extend the low 8 bytes -> i16x8.
    #[inline]
    fn ext8(v: __m128i) -> __m128i {
        unsafe {
            let z = _mm_setzero_si128();
            _mm_or_si128(_mm_unpacklo_epi8(v, z), _mm_slli_si128(_mm_unpackhi_epi8(v, z), 8))
        }
    }

    /// Exact low 32 bits of `a * b` per i32 lane (a: i32x4; b: i16x8 — lanes
    /// 0..3 hold the four weights). Full-width multiply, exact for all a and
    /// any i16 b: with a = u + hi*2^16 (u unsigned low 16), PMULLW/PMULHW
    /// read u as signed (u - 65536k); the k*b lost in bits 16..31 is added
    /// back via `kb`. The weights are spread to even i16 lanes because each
    /// i32 lane of a contributes only its low i16 half to the products.
    #[inline]
    fn mul32_16(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let zero = _mm_setzero_si128();
            let b = _mm_unpacklo_epi16(b, zero);
            let u = _mm_and_si128(a, _mm_set1_epi32(0x0000FFFF));
            let hi = _mm_srai_epi32(a, 16);
            let x0 = _mm_mullo_epi16(u, b);
            let x1 = _mm_mulhi_epi16(u, b);
            let y0 = _mm_mullo_epi16(hi, b);
            let kb = _mm_mullo_epi16(_mm_cmpgt_epi16(zero, u), _mm_sub_epi16(zero, b));
            _mm_or_si128(x0, _mm_slli_epi32(_mm_add_epi16(_mm_add_epi16(x1, y0), kb), 16))
        }
    }

    /// Saturating i32x4 -> i16x4 in the low 4 i16 lanes (high 8 bytes zero).
    #[inline]
    fn narrow_sat16_lo(x: __m128i) -> __m128i {
        unsafe {
            let clamped = _mm_max_epi32(
                _mm_min_epi32(x, _mm_set1_epi32(32767)),
                _mm_set1_epi32(-32768),
            );
            let gather = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
            _mm_shuffle_epi8(_mm_and_si128(clamped, _mm_set1_epi32(0x0000FFFF)), gather)
        }
    }

    /// `v << n`, n masked to i32 width (matches scalar wrapping_shl).
    #[inline]
    fn sll32(v: __m128i, n: u32) -> __m128i {
        unsafe {
            let n = n & 31;
            let v = match n & 15 {
                0 => v,
                1 => _mm_slli_epi32(v, 1),
                2 => _mm_slli_epi32(v, 2),
                3 => _mm_slli_epi32(v, 3),
                4 => _mm_slli_epi32(v, 4),
                5 => _mm_slli_epi32(v, 5),
                6 => _mm_slli_epi32(v, 6),
                7 => _mm_slli_epi32(v, 7),
                8 => _mm_slli_epi32(v, 8),
                9 => _mm_slli_epi32(v, 9),
                10 => _mm_slli_epi32(v, 10),
                11 => _mm_slli_epi32(v, 11),
                12 => _mm_slli_epi32(v, 12),
                13 => _mm_slli_epi32(v, 13),
                14 => _mm_slli_epi32(v, 14),
                _ => _mm_slli_epi32(v, 15),
            };
            if n >= 16 {
                _mm_slli_epi32(v, 16)
            } else {
                v
            }
        }
    }

    /// `v >> n` (arithmetic), n masked to i32 width.
    #[inline]
    fn sra32(v: __m128i, n: u32) -> __m128i {
        unsafe {
            let n = n & 31;
            let v = match n & 15 {
                0 => v,
                1 => _mm_srai_epi32(v, 1),
                2 => _mm_srai_epi32(v, 2),
                3 => _mm_srai_epi32(v, 3),
                4 => _mm_srai_epi32(v, 4),
                5 => _mm_srai_epi32(v, 5),
                6 => _mm_srai_epi32(v, 6),
                7 => _mm_srai_epi32(v, 7),
                8 => _mm_srai_epi32(v, 8),
                9 => _mm_srai_epi32(v, 9),
                10 => _mm_srai_epi32(v, 10),
                11 => _mm_srai_epi32(v, 11),
                12 => _mm_srai_epi32(v, 12),
                13 => _mm_srai_epi32(v, 13),
                14 => _mm_srai_epi32(v, 14),
                _ => _mm_srai_epi32(v, 15),
            };
            if n >= 16 {
                _mm_srai_epi32(v, 16)
            } else {
                v
            }
        }
    }

    /// `v << n` (i16 lanes), n masked to i16 width.
    #[inline]
    fn sll16(v: __m128i, n: u32) -> __m128i {
        unsafe {
            match n & 15 {
                0 => v,
                1 => _mm_slli_epi16(v, 1),
                2 => _mm_slli_epi16(v, 2),
                3 => _mm_slli_epi16(v, 3),
                4 => _mm_slli_epi16(v, 4),
                5 => _mm_slli_epi16(v, 5),
                6 => _mm_slli_epi16(v, 6),
                7 => _mm_slli_epi16(v, 7),
                8 => _mm_slli_epi16(v, 8),
                9 => _mm_slli_epi16(v, 9),
                10 => _mm_slli_epi16(v, 10),
                11 => _mm_slli_epi16(v, 11),
                12 => _mm_slli_epi16(v, 12),
                13 => _mm_slli_epi16(v, 13),
                14 => _mm_slli_epi16(v, 14),
                _ => _mm_slli_epi16(v, 15),
            }
        }
    }

    /// 4x4 IDCT pass (C e0..e3/f0..f3), wrapping i32.
    #[inline]
    fn pass4x(
        d0: __m128i,
        d1: __m128i,
        d2: __m128i,
        d3: __m128i,
    ) -> (__m128i, __m128i, __m128i, __m128i) {
        unsafe {
            let e0 = _mm_add_epi32(d0, d2);
        let e1 = _mm_sub_epi32(d0, d2);
        let e2 = _mm_sub_epi32(_mm_srai_epi32(d1, 1), d3);
        let e3 = _mm_add_epi32(_mm_srai_epi32(d3, 1), d1);
        (
            _mm_add_epi32(e0, e3),
            _mm_add_epi32(e1, e2),
            _mm_sub_epi32(e1, e2),
            _mm_sub_epi32(e0, e3),
        )
        }
    }

    /// 8x8 IDCT pass (C e0..e7/f0..f7/d0'..d7'), wrapping i16.
    #[inline]
    fn pass8x(d: [__m128i; 8]) -> [__m128i; 8] {
        unsafe {
            let (d0, d1, d2, d3, d4, d5, d6, d7) = (d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
        let e0 = _mm_add_epi16(d0, d4);
        let e1 = _mm_sub_epi16(_mm_sub_epi16(d5, d3), _mm_add_epi16(_mm_srai_epi16(d7, 1), d7));
        let e2 = _mm_sub_epi16(d0, d4);
        let e3 = _mm_sub_epi16(_mm_add_epi16(d1, d7), _mm_add_epi16(_mm_srai_epi16(d3, 1), d3));
        let e4 = _mm_sub_epi16(_mm_srai_epi16(d2, 1), d6);
        let e5 = _mm_add_epi16(_mm_sub_epi16(d7, d1), _mm_add_epi16(_mm_srai_epi16(d5, 1), d5));
        let e6 = _mm_add_epi16(_mm_srai_epi16(d6, 1), d2);
        let e7 = _mm_add_epi16(_mm_add_epi16(d3, d5), _mm_add_epi16(_mm_srai_epi16(d1, 1), d1));
        let f0 = _mm_add_epi16(e0, e6);
        let f1 = _mm_add_epi16(_mm_srai_epi16(e7, 2), e1);
        let f2 = _mm_add_epi16(e2, e4);
        let f3 = _mm_add_epi16(_mm_srai_epi16(e5, 2), e3);
        let f4 = _mm_sub_epi16(e2, e4);
        let f5 = _mm_sub_epi16(_mm_srai_epi16(e3, 2), e5);
        let f6 = _mm_sub_epi16(e0, e6);
        let f7 = _mm_sub_epi16(e7, _mm_srai_epi16(e1, 2));
        [
            _mm_add_epi16(f0, f7),
            _mm_add_epi16(f2, f5),
            _mm_add_epi16(f4, f3),
            _mm_add_epi16(f6, f1),
            _mm_sub_epi16(f6, f1),
            _mm_sub_epi16(f4, f3),
            _mm_sub_epi16(f2, f5),
            _mm_sub_epi16(f0, f7),
        ]
        }
    }

    /// 8x8 i16 transpose: t[i] = column i of `f` (C x0..xF zip cascade).
    #[inline]
    fn transpose8(f: [__m128i; 8]) -> [__m128i; 8] {
        unsafe {
            let a0 = _mm_unpacklo_epi16(f[0], f[1]);
        let a1 = _mm_unpackhi_epi16(f[0], f[1]);
        let b0 = _mm_unpacklo_epi16(f[2], f[3]);
        let b1 = _mm_unpackhi_epi16(f[2], f[3]);
        let c0 = _mm_unpacklo_epi16(f[4], f[5]);
        let c1 = _mm_unpackhi_epi16(f[4], f[5]);
        let e0 = _mm_unpacklo_epi16(f[6], f[7]);
        let e1 = _mm_unpackhi_epi16(f[6], f[7]);
        let p0 = _mm_unpacklo_epi32(a0, b0);
        let p1 = _mm_unpackhi_epi32(a0, b0);
        let p2 = _mm_unpacklo_epi32(a1, b1);
        let p3 = _mm_unpackhi_epi32(a1, b1);
        let p4 = _mm_unpacklo_epi32(c0, e0);
        let p5 = _mm_unpackhi_epi32(c0, e0);
        let p6 = _mm_unpacklo_epi32(c1, e1);
        let p7 = _mm_unpackhi_epi32(c1, e1);
        [
            _mm_unpacklo_epi64(p0, p4),
            _mm_unpackhi_epi64(p0, p4),
            _mm_unpacklo_epi64(p1, p5),
            _mm_unpackhi_epi64(p1, p5),
            _mm_unpacklo_epi64(p2, p6),
            _mm_unpackhi_epi64(p2, p6),
            _mm_unpacklo_epi64(p3, p7),
            _mm_unpackhi_epi64(p3, p7),
        ]
        }
    }

    /// Dequantize one 4x4 row: ((c * ls) << sh + 8) >> 4, wrapping i32.
    #[inline]
    fn deq4x(c: __m128i, w: __m128i, sh: u32) -> __m128i {
        unsafe { sra32(_mm_add_epi32(sll32(mul32_16(c, w), sh), _mm_set1_epi32(8)), 4) }
    }

    /// Add sat16(h >> 6) to four 4-byte pixel rows (C final store).
    #[inline]
    fn add4_pix(pix: *mut u8, stride: usize, h0: __m128i, h1: __m128i, h2: __m128i, h3: __m128i) {
        let hs = unsafe {
            [
                narrow_sat16_lo(_mm_srai_epi32(h0, 6)),
                narrow_sat16_lo(_mm_srai_epi32(h1, 6)),
                narrow_sat16_lo(_mm_srai_epi32(h2, 6)),
                narrow_sat16_lo(_mm_srai_epi32(h3, 6)),
            ]
        };
        for (i, r) in hs.iter().enumerate() {
            unsafe {
                let pv = std::ptr::read_unaligned(pix.add(i * stride) as *const u32);
                let sum = _mm_add_epi16(ext4(_mm_cvtsi32_si128(pv as i32)), *r);
                let out = _mm_packus_epi16(sum, sum);
                std::ptr::write_unaligned(pix.add(i * stride) as *mut u32, _mm_cvtsi128_si32(out) as u32);
            }
        }
    }

    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn add_idct4x4_sse(
        c: &mut [i32; 64],
        qp: u8,
        ws: &[i8; 16],
        dcidx: i32,
        pix: *mut u8,
        stride: usize,
    ) {
        let qpu = qp as u32;
        let sh = qpu / 6;
        let na = unsafe {
            _mm_loadu_si128(NORM_ADJUST_4X4[(qpu % 6) as usize].as_ptr() as *const __m128i)
        };
        let wsv = unsafe { _mm_loadu_si128(ws.as_ptr() as *const __m128i) };
        // LS = unsigned byte products (C mullou8/mulhiu8).
        let ls_lo = byte_mul_lo(wsv, na);
        let ls_hi = byte_mul_hi(wsv, na);
        let lo4 = lo4_mask();
        let w0 = _mm_and_si128(ls_lo, lo4);
        let w1 = _mm_srli_si128(ls_lo, 8);
        let w2 = _mm_and_si128(ls_hi, lo4);
        let w3 = _mm_srli_si128(ls_hi, 8);
        let base = c.as_ptr();
        let mut d0 = deq4x(unsafe { _mm_loadu_si128(base as *const __m128i) }, w0, sh);
        let d1 = deq4x(unsafe { _mm_loadu_si128(base.add(4) as *const __m128i) }, w1, sh);
        let d2 = deq4x(unsafe { _mm_loadu_si128(base.add(8) as *const __m128i) }, w2, sh);
        let d3 = deq4x(unsafe { _mm_loadu_si128(base.add(12) as *const __m128i) }, w3, sh);
        if dcidx >= 0 {
            // C: d0[0] = c[16 + DCidx] (raw coefficient, no dequant).
            d0 = _mm_insert_epi32(d0, c[16 + dcidx as usize], 0);
        }
        c[..16].fill(0);
        // pass1 -> transpose (+32 on the first row) -> pass2.
        let (f0, f1, f2, f3) = pass4x(d0, d1, d2, d3);
        let x0 = _mm_unpacklo_epi32(f0, f1);
        let x1 = _mm_unpackhi_epi32(f0, f1);
        let x2 = _mm_unpacklo_epi32(f2, f3);
        let x3 = _mm_unpackhi_epi32(f2, f3);
        let t0 = _mm_add_epi32(_mm_unpacklo_epi64(x0, x2), _mm_set1_epi32(32));
        let t1 = _mm_unpackhi_epi64(x0, x2);
        let t2 = _mm_unpacklo_epi64(x1, x3);
        let t3 = _mm_unpackhi_epi64(x1, x3);
        let (h0, h1, h2, h3) = pass4x(t0, t1, t2, t3);
        add4_pix(pix, stride, h0, h1, h2, h3);
    }

    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn add_dc4x4_sse(c: &[i32; 64], dcidx: i32, pix: *mut u8, stride: usize) {
        // C `set16((c + 32) >> 6)`: i32 shift, truncated (not saturated) to i16.
        let r = _mm_set1_epi16(((c[16 + dcidx as usize]).wrapping_add(32) >> 6) as i16);
        for i in 0..4 {
            unsafe {
                let pv = std::ptr::read_unaligned(pix.add(i * stride) as *const u32);
                let sum = _mm_add_epi16(ext4(_mm_cvtsi32_si128(pv as i32)), r);
                let out = _mm_packus_epi16(sum, sum);
                std::ptr::write_unaligned(pix.add(i * stride) as *mut u32, _mm_cvtsi128_si32(out) as u32);
            }
        }
    }

    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn add_idct8x8_sse(
        c: &mut [i32; 64],
        qp: u8,
        ws: &[i8; 64],
        pix: *mut u8,
        stride: usize,
    ) {
        let qpu = qp as u32;
        let div = qpu / 6;
        let m = (qpu % 6) as usize;
        let na0 = unsafe { _mm_loadu_si128(NORM_ADJUST_8X8[m * 2].as_ptr() as *const __m128i) };
        let na1 =
            unsafe { _mm_loadu_si128(NORM_ADJUST_8X8[m * 2 + 1].as_ptr() as *const __m128i) };
        let wbase = ws.as_ptr() as *const __m128i;
        let (w0, w1, w2, w3) = (
            unsafe { _mm_loadu_si128(wbase) },
            unsafe { _mm_loadu_si128(wbase.add(1)) },
            unsafe { _mm_loadu_si128(wbase.add(2)) },
            unsafe { _mm_loadu_si128(wbase.add(3)) },
        );
        // LSb = unsigned byte products (C mullou8/mulhiu8), i16 lanes 0..7.
        let ls = [
            byte_mul_lo(w0, na0),
            byte_mul_hi(w0, na0),
            byte_mul_lo(w1, na1),
            byte_mul_hi(w1, na1),
            byte_mul_lo(w2, na0),
            byte_mul_hi(w2, na0),
            byte_mul_lo(w3, na1),
            byte_mul_hi(w3, na1),
        ];
        let cbase = c.as_ptr();
        let cv = |k: usize| unsafe { _mm_loadu_si128(cbase.add(4 * k) as *const __m128i) };
        let lo4 = lo4_mask();
        let d = if div < 6 {
            let off = _mm_set1_epi32(1i32 << (5 - div));
            let sh = 6 - div;
            std::array::from_fn(|b| {
                let wlo = _mm_and_si128(ls[b], lo4);
                let whi = _mm_srli_si128(ls[b], 8);
                let lo = narrow_sat16_lo(sra32(_mm_add_epi32(mul32_16(cv(2 * b), wlo), off), sh));
                let hi = narrow_sat16_lo(sra32(
                    _mm_add_epi32(mul32_16(cv(2 * b + 1), whi), off),
                    sh,
                ));
                _mm_or_si128(lo, _mm_slli_si128(hi, 8))
            })
        } else {
            let sh = div - 6;
            std::array::from_fn(|b| {
                let sat_lo = narrow_sat16_lo(cv(2 * b));
                let sat_hi = narrow_sat16_lo(cv(2 * b + 1));
                _mm_mullo_epi16(_mm_or_si128(sat_lo, _mm_slli_si128(sat_hi, 8)), sll16(ls[b], sh))
            })
        };
        c.fill(0);
        // pass1 -> transpose (+32 on the first row) -> pass2.
        let mut t = transpose8(pass8x(d));
        t[0] = _mm_add_epi16(t[0], _mm_set1_epi16(32));
        let d = pass8x(t);
        // row i += d[i] >> 6 (arithmetic), clamped.
        for (i, row) in d.iter().enumerate() {
            unsafe {
                let pv = std::ptr::read_unaligned(pix.add(i * stride) as *const u64);
                let sum = _mm_add_epi16(ext8(_mm_cvtsi64_si128(pv as i64)), _mm_srai_epi16(*row, 6));
                let out = _mm_packus_epi16(sum, sum);
                std::ptr::write_unaligned(pix.add(i * stride) as *mut u64, _mm_cvtsi128_si64(out) as u64);
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64-style LCG (same family as the deblock params test).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    /// SSE (via the dispatchers) vs scalar fallback for the three IDCT/DC
    /// kernels: random + extreme coefficients (i32::MAX/MIN/full range), full
    /// QP range (exercises wrapping-shift masking beyond qp 51), dcidx -1..7.
    #[test]
    fn sse_vs_scalar_idct() {
        let mut rng = Lcg(0x01dc_755e);
        for _ in 0..20_000 {
            let mk_c = |rng: &mut Lcg| -> [i32; 64] {
                core::array::from_fn(|_| match rng.below(16) {
                    0 => i32::MAX,
                    1 => i32::MIN,
                    2 => rng.next() as i32, // full range
                    _ => (rng.below(4096) as i32) - 2048,
                })
            };
            let ws4: [i8; 16] = core::array::from_fn(|_| rng.next() as i8);
            let ws8: [i8; 64] = core::array::from_fn(|_| rng.next() as i8);
            // Mostly spec range (qp <= 51), occasionally beyond.
            let qp = if rng.below(32) == 0 {
                rng.next() as u8
            } else {
                rng.below(52) as u8
            };
            let dcidx = if rng.below(4) == 0 { -1 } else { rng.below(8) as i32 };

            // add_idct4x4 (zeroes c[0..16]).
            let stride4 = 4 + (rng.below(4) as usize) * 4;
            let mut pix = [0u8; 4 * 20];
            for v in pix.iter_mut() {
                *v = rng.next() as u8;
            }
            let c0 = mk_c(&mut rng);
            let (mut c_a, mut c_b) = (c0, c0);
            let (mut pa, mut pb) = (pix, pix);
            unsafe {
                add_idct4x4(&mut c_a, qp, &ws4, dcidx, pa.as_mut_ptr(), stride4);
                add_idct4x4_scalar(&mut c_b, qp, &ws4, dcidx, pb.as_mut_ptr(), stride4);
            }
            assert_eq!(c_a, c_b, "idct4x4: c post-state (qp={qp})");
            for i in 0..4 {
                assert_eq!(
                    &pa[i * stride4..i * stride4 + 4],
                    &pb[i * stride4..i * stride4 + 4],
                    "idct4x4: row {i} (qp={qp}, dcidx={dcidx})"
                );
            }

            // add_dc4x4 (read-only on c). dcidx must be in range (a DC-only
            // 4x4 block always carries its DC coeff); -1 is invalid here.
            let dc = rng.below(8) as i32;
            let (mut pa, mut pb) = (pix, pix);
            unsafe {
                add_dc4x4(&c_a, dc, pa.as_mut_ptr(), stride4);
                add_dc4x4_scalar(&c_a, dc, pb.as_mut_ptr(), stride4);
            }
            for i in 0..4 {
                assert_eq!(
                    &pa[i * stride4..i * stride4 + 4],
                    &pb[i * stride4..i * stride4 + 4],
                    "dc4x4: row {i} (qp={qp}, dcidx={dc})"
                );
            }

            // add_idct8x8 (zeroes all of c).
            let stride8 = 8 + (rng.below(4) as usize) * 8;
            let mut pix8 = [0u8; 8 * 40];
            for v in pix8.iter_mut() {
                *v = rng.next() as u8;
            }
            let c0 = mk_c(&mut rng);
            let (mut c_a, mut c_b) = (c0, c0);
            let (mut pa, mut pb) = (pix8, pix8);
            unsafe {
                add_idct8x8(&mut c_a, qp, &ws8, pa.as_mut_ptr(), stride8);
                add_idct8x8_scalar(&mut c_b, qp, &ws8, pb.as_mut_ptr(), stride8);
            }
            assert_eq!(c_a, c_b, "idct8x8: c post-state (qp={qp})");
            for i in 0..8 {
                assert_eq!(
                    &pa[i * stride8..i * stride8 + 8],
                    &pb[i * stride8..i * stride8 + 8],
                    "idct8x8: row {i} (qp={qp})"
                );
            }
        }
    }
}

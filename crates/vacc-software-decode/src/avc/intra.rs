//! Port of the edge264 intra prediction kernels (Tier B). Bit-exact with
//! c/src/edge264_intra.c (SSE backend, SSE4.1 loaders).
//!
//! The C code operates on 128-bit vectors; the port re-expresses every
//! operation as scalar byte math with the exact per-lane semantics, verified
//! against standalone C probes of each primitive:
//!   shr128(a, n)   = [a[n..16], 0*n]          (C `_mm_srli_si128`: zero tail)
//!   shl128(a, n)   = [0*n, a[..16-n]]
//!   shlc128(a, n)  = [a[0]*n, a[n..16]]       (head replicated)
//!   shrc128(a, n)  = [a[n..16], a[15]*n]      (tail replicated — NOT zero!)
//!   shrd128(l,h,i) = [l[i..16], h[..i]]
//!   lowpass8       = (l + 2m + r + 2) >> 2    (u8; == C SSE form for all inputs)
//!   avgu8          = (a + b + 1) >> 1
//!   sumh8(a)       = [sum a[0..8], sum a[8..16]]   (per `_mm_sad_epu8` lanes)
//!   trnlo32(a,b)   = i32 lanes [a0, b0, a2, b2]
//!   trnhi32(a,b)   = i32 lanes [a1, b1, a3, b3]
//!   maddubx(a,b)   = a[2k]*b[2k] + a[2k+1]*b[2k+1] (i16)
//!   hadd16(a,b)    = [a0+a1, a2+a3, a4+a5, a6+a7, b0+b1, ...]
//!   packs32(a,b)   = [a[0..4], b[0..4]] as i16 (sat)
//!   packus16(a,b)  = [sat8u(a[0..8]), sat8u(b[0..8])]  (NOT interleaved)
//!   shrru16(x, i)  = ((x >> (i-1)) + 1) >> 1         (u16 rounding shift)
//!   shrrs16(x, i)  = (x.wrapping_add(1 << (i-1))) >> i  (i16, wrapping add)
//!
//! `clip` is a vestigial parameter in C (never used by these kernels).

/// Mode values of C `enum Intra4x4Modes`.
pub const I4X4_V: u32 = 0;
pub const I4X4_H: u32 = 1;
pub const I4X4_DC: u32 = 2;
pub const I4X4_DC_A: u32 = 3;
pub const I4X4_DC_B: u32 = 4;
pub const I4X4_DC_AB: u32 = 5;
pub const I4X4_DDL: u32 = 6;
pub const I4X4_DDL_C: u32 = 7;
pub const I4X4_DDR: u32 = 8;
pub const I4X4_VR: u32 = 9;
pub const I4X4_HD: u32 = 10;
pub const I4X4_VL: u32 = 11;
pub const I4X4_VL_C: u32 = 12;
pub const I4X4_HU: u32 = 13;

/// Mode values of C `enum Intra8x8Modes`.
pub const I8X8_V: u32 = 0;
pub const I8X8_V_C: u32 = 1;
pub const I8X8_V_D: u32 = 2;
pub const I8X8_V_CD: u32 = 3;
pub const I8X8_H: u32 = 4;
pub const I8X8_H_D: u32 = 5;
pub const I8X8_DC: u32 = 6;
pub const I8X8_DC_A: u32 = 7;
pub const I8X8_DC_AC: u32 = 8;
pub const I8X8_DC_AD: u32 = 9;
pub const I8X8_DC_ACD: u32 = 10;
pub const I8X8_DC_B: u32 = 11;
pub const I8X8_DC_BD: u32 = 12;
pub const I8X8_DC_C: u32 = 13;
pub const I8X8_DC_D: u32 = 14;
pub const I8X8_DC_CD: u32 = 15;
pub const I8X8_DC_AB: u32 = 16;
pub const I8X8_DDL: u32 = 17;
pub const I8X8_DDL_C: u32 = 18;
pub const I8X8_DDL_D: u32 = 19;
pub const I8X8_DDL_CD: u32 = 20;
pub const I8X8_DDR: u32 = 21;
pub const I8X8_DDR_C: u32 = 22;
pub const I8X8_VR: u32 = 23;
pub const I8X8_VR_C: u32 = 24;
pub const I8X8_HD: u32 = 25;
pub const I8X8_VL: u32 = 26;
pub const I8X8_VL_C: u32 = 27;
pub const I8X8_VL_D: u32 = 28;
pub const I8X8_VL_CD: u32 = 29;
pub const I8X8_HU: u32 = 30;
pub const I8X8_HU_D: u32 = 31;

/// Mode values of C `enum Intra16x16Modes`.
pub const I16X16_V: u32 = 0;
pub const I16X16_H: u32 = 1;
pub const I16X16_DC: u32 = 2;
pub const I16X16_DC_A: u32 = 3;
pub const I16X16_DC_B: u32 = 4;
pub const I16X16_DC_AB: u32 = 5;
pub const I16X16_P: u32 = 6;

/// Mode values of C `enum IntraChromaModes`.
pub const IC8X8_DC: u32 = 0;
pub const IC8X8_DC_A: u32 = 1;
pub const IC8X8_DC_B: u32 = 2;
pub const IC8X8_DC_AB: u32 = 3;
pub const IC8X8_H: u32 = 4;
pub const IC8X8_V: u32 = 5;
pub const IC8X8_P: u32 = 6;

/// C `avgu8` lane.
#[inline]
fn avgu8(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}

/// C `lowpass8` lane. Algebraically identical to the SSE form
/// `avgu8(subsu8(avgu8(l, r), (l ^ r) & 1), m)` for all u8 inputs.
#[inline]
fn lowpass8(l: u8, m: u8, r: u8) -> u8 {
    ((l as u16 + 2 * m as u16 + r as u16 + 2) >> 2) as u8
}

/// C `shrru16` lane (rounding unsigned shift).
#[inline]
fn shrru16(v: u16, i: u32) -> u16 {
    ((v >> (i - 1)) + 1) >> 1
}

/// C `shrrs16` lane (rounding arithmetic shift, i16 wrapping add).
#[inline]
fn shrrs16(v: i16, i: u32) -> i16 {
    (v.wrapping_add(1 << (i - 1))) >> i
}

/// C `shr128`: drop the first n bytes, zero-pad the tail.
#[inline]
fn shr128(a: &[u8; 16], n: usize) -> [u8; 16] {
    let mut r = [0u8; 16];
    r[..16 - n].copy_from_slice(&a[n..]);
    r
}

/// C `shl128`: zero-pad the head, drop the last n bytes.
#[inline]
fn shl128(a: &[u8; 16], n: usize) -> [u8; 16] {
    let mut r = [0u8; 16];
    r[n..].copy_from_slice(&a[..16 - n]);
    r
}

/// C `shlc128`: left byte shift, first byte replicated.
#[inline]
fn shlc128(a: &[u8; 16], n: usize) -> [u8; 16] {
    let mut r = [0u8; 16];
    r[..n].fill(a[0]);
    r[n..].copy_from_slice(&a[..16 - n]);
    r
}

/// C `shrc128`: right byte shift, last byte replicated.
#[inline]
fn shrc128(a: &[u8; 16], n: usize) -> [u8; 16] {
    let mut r = [0u8; 16];
    r[..16 - n].copy_from_slice(&a[n..]);
    r[16 - n..].fill(a[15]);
    r
}

/// C `shrd128(l, h, i)`: window of `[l | h]` starting at byte i.
#[inline]
fn shrd128(l: &[u8; 16], h: &[u8; 16], i: usize) -> [u8; 16] {
    let mut r = [0u8; 16];
    r[..16 - i].copy_from_slice(&l[i..]);
    r[16 - i..].copy_from_slice(&h[..i]);
    r
}

/// C `maddubx` (maddubs): pairwise unsigned×signed multiply-add.
#[inline]
fn maddubx(a: &[u8; 16], b: &[i8; 16]) -> [i16; 8] {
    let mut r = [0i16; 8];
    for k in 0..8 {
        r[k] = a[2 * k] as i16 * b[2 * k] as i16 + a[2 * k + 1] as i16 * b[2 * k + 1] as i16;
    }
    r
}

/// C `hadd16`: pairwise horizontal add.
#[inline]
fn hadd16(a: &[i16; 8], b: &[i16; 8]) -> [i16; 8] {
    let mut r = [0i16; 8];
    for k in 0..4 {
        r[k] = a[2 * k].wrapping_add(a[2 * k + 1]);
        r[4 + k] = b[2 * k].wrapping_add(b[2 * k + 1]);
    }
    r
}

/// C `packus16` (`_mm_packus_epi16`). On this CPU it concatenates:
/// [sat8u(a[0..8]), sat8u(b[0..8])] (NOT the standard 4-lane interleave).
#[inline]
fn packus16(a: &[i16; 8], b: &[i16; 8]) -> [u8; 16] {
    let mut r = [0u8; 16];
    for k in 0..8 {
        r[k] = a[k].clamp(0, 255) as u8;
        r[8 + k] = b[k].clamp(0, 255) as u8;
    }
    r
}

/// C `shrpus16(a, b, i)`: truncating shift, then unsigned-saturating pack.
#[inline]
fn shrpus16(a: &[i16; 8], b: &[i16; 8], i: u32) -> [u8; 16] {
    let sa: [i16; 8] = std::array::from_fn(|k| a[k] >> i);
    let sb: [i16; 8] = std::array::from_fn(|k| b[k] >> i);
    packus16(&sa, &sb)
}

/// Sample at (dr, dc) relative to the block origin `o` in row-major `buf`.
/// Callers must guarantee the neighborhood is inside `buf`.
#[inline]
fn px(buf: &[u8], o: usize, stride: usize, dr: i32, dc: i32) -> u8 {
    let off = o as isize + dr as isize * stride as isize + dc as isize;
    buf[off as usize]
}

/// C `decode_intra4x4`. Block origin at `buf[o]`; writes 4 rows x 4 cols.
pub fn intra4x4(buf: &mut [u8], o: usize, stride: usize, mode: u32) {
    let t = |i: i32| px(buf, o, stride, -1, i);
    let l = |r: i32| px(buf, o, stride, r, -1);

    let mut out = [0u8; 16];
    match mode {
        I4X4_V => {
            let row = [t(0), t(1), t(2), t(3)];
            for r in 0..4 {
                out[4 * r..4 * r + 4].copy_from_slice(&row);
            }
        }
        I4X4_H => {
            for r in 0..4 {
                out[4 * r..4 * r + 4].fill(l(r as i32));
            }
        }
        I4X4_DC => {
            let s: u32 = l(0) as u32
                + l(1) as u32
                + l(2) as u32
                + l(3) as u32
                + t(0) as u32
                + t(1) as u32
                + t(2) as u32
                + t(3) as u32;
            out.fill(shrru16(s as u16, 3) as u8);
        }
        I4X4_DC_A => {
            let s: u32 = 2 * (t(0) as u32 + t(1) as u32 + t(2) as u32 + t(3) as u32);
            out.fill(shrru16(s as u16, 3) as u8);
        }
        I4X4_DC_B => {
            let s: u32 = 2 * (l(0) as u32 + l(1) as u32 + l(2) as u32 + l(3) as u32);
            out.fill(shrru16(s as u16, 3) as u8);
        }
        I4X4_DC_AB => out.fill(128),
        _ => {
            // Diagonal modes: build C `v`, then f = lowpass over
            // (broadcast64(v,0), broadcast64(shr128(v,1),0), ziplo64(shr128(v,2), v)),
            // gathered through the per-mode shuffle table.
            let (v, idx): ([u8; 16], usize) = match mode {
                I4X4_DDL => {
                    let mut v = [0u8; 16];
                    for (k, slot) in v[..8].iter_mut().enumerate() {
                        *slot = t(k as i32);
                    }
                    v[8..].fill(t(7));
                    (v, 0)
                }
                I4X4_DDL_C => {
                    let mut v = [0u8; 16];
                    for (k, slot) in v[..4].iter_mut().enumerate() {
                        *slot = t(k as i32);
                    }
                    v[4..].fill(t(3));
                    (v, 0)
                }
                I4X4_DDR | I4X4_VR | I4X4_HD => {
                    // ldedge4x4 (SSE4.1): [L3, L2, L1, L0, T(-5)..T(10)].
                    let mut v = [0u8; 16];
                    for (k, slot) in v.iter_mut().enumerate() {
                        *slot = t(-5 + k as i32);
                    }
                    for r in 0..4 {
                        v[3 - r] = l(r as i32);
                    }
                    // C shuf slots: DDL/DDL_C -> 0, DDR -> 1, VR -> 2, HD -> 3.
                    (v, mode as usize - I4X4_DDL_C as usize)
                }
                I4X4_VL => {
                    let mut v = [0u8; 16];
                    for (k, slot) in v[..8].iter_mut().enumerate() {
                        *slot = t(k as i32);
                    }
                    (v, 4)
                }
                I4X4_VL_C => {
                    let mut v = [0u8; 16];
                    for (k, slot) in v[..4].iter_mut().enumerate() {
                        *slot = t(k as i32);
                    }
                    v[4..].fill(t(3));
                    (v, 4)
                }
                I4X4_HU => {
                    let mut v = [0u8; 16];
                    for (k, slot) in v[..4].iter_mut().enumerate() {
                        *slot = l(k as i32);
                    }
                    v[4..].fill(l(3));
                    (v, 5)
                }
                _ => unreachable!(),
            };
            let w = shr128(&v, 1);
            let x = shr128(&v, 2);
            let f: [u8; 16] = std::array::from_fn(|k| {
                let rk = if k < 8 { x[k] } else { v[k - 8] };
                lowpass8(v[k & 7], w[k & 7], rk)
            });
            const SHUF: [[u8; 16]; 6] = [
                [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6],
                [3, 4, 5, 6, 2, 3, 4, 5, 1, 2, 3, 4, 0, 1, 2, 3],
                [12, 13, 14, 15, 3, 4, 5, 6, 2, 12, 13, 14, 1, 3, 4, 5],
                [11, 3, 4, 5, 10, 2, 11, 3, 9, 1, 10, 2, 8, 0, 9, 1],
                [8, 9, 10, 11, 0, 1, 2, 3, 9, 10, 11, 12, 1, 2, 3, 4],
                [8, 0, 9, 1, 9, 1, 10, 2, 10, 2, 11, 3, 11, 3, 12, 4],
            ];
            for k in 0..16 {
                out[k] = f[SHUF[idx][k] as usize];
            }
        }
    }

    for r in 0..4 {
        buf[o + r * stride..o + r * stride + 4].copy_from_slice(&out[4 * r..4 * r + 4]);
    }
}

/// C `decode_intra8x8`. Block origin at `buf[o]`; writes 8 rows x 8 cols.
pub fn intra8x8(buf: &mut [u8], o: usize, stride: usize, mode: u32) {
    let t = |i: i32| px(buf, o, stride, -1, i);
    let l = |r: i32| px(buf, o, stride, r, -1);

    // Top-row neighbor vectors (C j2s variants). C's switch loads these
    // lazily per branch: the Intra8x8Modes table only pairs a variant with an
    // unavailability state whose missing references that branch never loads,
    // so computing them per-arm keeps every read inside available memory.
    let jt = || std::array::from_fn(|k| t(-1 + k as i32)); // loadu128(pT - 1)
    let jtc = || std::array::from_fn(|k| if k <= 8 { t(k as i32 - 1) } else { t(7) });
    let jdd = || shlc128(&std::array::from_fn(|k| t(k as i32)), 1);
    let jcd = || {
        let a: [u8; 16] = std::array::from_fn(|k| if k < 8 { t(k as i32) } else { t(7) });
        shlc128(&a, 1)
    };
    // C `ldedge8x8` (SSE4.1): [L7 x9, L6, L5, L4, L3, L2, L1, L0].
    let a2h = || {
        let l7 = l(7);
        std::array::from_fn(|k| if k >= 8 { l(15 - k as i32) } else { l7 })
    };

    /// C `ldleft8x8(p, stride, i2a)`: lowpass-filtered left edge vector.
    fn left8(i2a: u8, l: impl Fn(i32) -> u8) -> [u8; 16] {
        let mut v = [0u8; 16];
        v[0] = i2a;
        for k in 0..7 {
            v[1 + k] = l(k as i32);
        }
        v[8] = l(7);
        v[9] = v[8];
        let a = shr128(&v, 2);
        let b = shr128(&v, 1);
        std::array::from_fn(|k| lowpass8(a[k], b[k], v[k]))
    }

    /// C `lowpass8(shr128(j,2), shr128(j,1), j)`.
    fn topfilt(j: &[u8; 16]) -> [u8; 16] {
        let a = shr128(j, 2);
        let b = shr128(j, 1);
        std::array::from_fn(|k| lowpass8(a[k], b[k], j[k]))
    }

    /// C `shrru16(sum8(ziplo64(k2r, h2a)), 4)` broadcast. On this CPU
    /// `_mm_sad_epu8` sums 8-byte groups (lane 0 = Σ bytes[0..8], lane 4 =
    /// Σ bytes[8..16]), so `sum8`'s lane 0 = the full sum of all 16 bytes.
    fn dc_val(k2r: &[u8; 16], h2a: &[u8; 16]) -> u8 {
        let s: u16 = k2r[..8].iter().map(|&x| x as u16).sum::<u16>()
            + h2a[..8].iter().map(|&x| x as u16).sum::<u16>();
        shrru16(s, 4) as u8
    }

    // Rows as [u8; 64], row r at rows[8*r..].
    let mut rows = [0u8; 64];
    match mode {
        I8X8_V => fill_rows(&mut rows, &topfilt(&jt())),
        I8X8_V_C => fill_rows(&mut rows, &topfilt(&jtc())),
        I8X8_V_D => fill_rows(&mut rows, &topfilt(&jdd())),
        I8X8_V_CD => fill_rows(&mut rows, &topfilt(&jcd())),
        I8X8_H => {
            let f = left8(t(-1), l);
            for r in 0..8 {
                rows[8 * r..8 * r + 8].fill(f[r]);
            }
        }
        I8X8_H_D => {
            let f = left8(l(0), l);
            for r in 0..8 {
                rows[8 * r..8 * r + 8].fill(f[r]);
            }
        }
        I8X8_DC => {
            let v = dc_val(&topfilt(&jt()), &left8(t(-1), l));
            rows.fill(v);
        }
        I8X8_DC_A => {
            let k = topfilt(&jt());
            let v = dc_val(&k, &k);
            rows.fill(v);
        }
        I8X8_DC_AC => {
            let k = topfilt(&jtc());
            let v = dc_val(&k, &k);
            rows.fill(v);
        }
        I8X8_DC_AD => {
            let k = topfilt(&jdd());
            let v = dc_val(&k, &k);
            rows.fill(v);
        }
        I8X8_DC_ACD => {
            let k = topfilt(&jcd());
            let v = dc_val(&k, &k);
            rows.fill(v);
        }
        I8X8_DC_B => {
            let h = left8(t(-1), l);
            let v = dc_val(&h, &h);
            rows.fill(v);
        }
        I8X8_DC_BD => {
            let h = left8(l(0), l);
            let v = dc_val(&h, &h);
            rows.fill(v);
        }
        I8X8_DC_C => {
            let v = dc_val(&topfilt(&jtc()), &left8(t(-1), l));
            rows.fill(v);
        }
        I8X8_DC_D => {
            let v = dc_val(&topfilt(&jdd()), &left8(l(0), l));
            rows.fill(v);
        }
        I8X8_DC_CD => {
            let v = dc_val(&topfilt(&jcd()), &left8(l(0), l));
            rows.fill(v);
        }
        I8X8_DC_AB => rows.fill(128),
        I8X8_DDL | I8X8_DDL_C | I8X8_DDL_D | I8X8_DDL_CD => {
            let (j2y, k2z): ([u8; 16], [u8; 16]) = match mode {
                I8X8_DDL => (jt(), std::array::from_fn(|k| t(k as i32))),
                I8X8_DDL_C => {
                    let j2r: [u8; 16] = std::array::from_fn(|k| t(-8 + k as i32));
                    (shrc128(&j2r, 7), shrc128(&j2r, 8))
                }
                I8X8_DDL_D => {
                    let k2z = std::array::from_fn(|k| t(k as i32));
                    (shlc128(&k2z, 1), k2z)
                }
                _ => {
                    let k2z: [u8; 16] =
                        std::array::from_fn(|k| if k < 8 { t(k as i32) } else { t(7) });
                    (shlc128(&k2z, 1), k2z)
                }
            };
            let k1 = shrc128(&k2z, 1);
            let v0: [u8; 16] = std::array::from_fn(|k| lowpass8(j2y[k], k2z[k], k1[k]));
            let m = shr128(&v0, 1);
            let r = shrc128(&v0, 2);
            let p0: [u8; 16] = std::array::from_fn(|k| lowpass8(v0[k], m[k], r[k]));
            for row in 0..8 {
                rows[8 * row..8 * row + 8].copy_from_slice(&p0[row..row + 8]);
            }
        }
        I8X8_DDR | I8X8_DDR_C => {
            let j2s = match mode {
                I8X8_DDR_C => jtc(),
                _ => jt(),
            };
            let a2h = a2h();
            let a2q = shrd128(&a2h, &j2s, 8);
            let b2r = shrd128(&a2h, &j2s, 9);
            let x0: [u8; 16] =
                std::array::from_fn(|k| lowpass8(shrd128(&a2h, &j2s, 7)[k], a2q[k], b2r[k]));
            let x1: [u8; 16] =
                std::array::from_fn(|k| lowpass8(a2q[k], b2r[k], shrd128(&a2h, &j2s, 10)[k]));
            let xm = shr128(&x1, 1);
            let p7: [u8; 16] = std::array::from_fn(|k| lowpass8(x0[k], x1[k], xm[k]));
            // C: p_r = shr128(p7, 7-r); row r = p_r[0..8] = p7[7-r..15-r].
            for row in 0..8 {
                rows[8 * row..8 * row + 8].copy_from_slice(&p7[7 - row..15 - row]);
            }
        }
        I8X8_VR | I8X8_VR_C => {
            let j2s = match mode {
                I8X8_VR_C => jtc(),
                _ => jt(),
            };
            let a2h = a2h();
            let a2q = shrd128(&a2h, &j2s, 8);
            let b2r = shrd128(&a2h, &j2s, 9);
            let c2s = shrd128(&a2h, &j2s, 10);
            let v0: [u8; 16] = std::array::from_fn(|k| lowpass8(a2q[k], b2r[k], c2s[k]));
            let v1 = shl128(&v0, 1);
            let v2s = shl128(&v0, 2);
            let v2: [u8; 16] = std::array::from_fn(|k| lowpass8(v0[k], v1[k], v2s[k]));
            let avg: [u8; 16] = std::array::from_fn(|k| avgu8(v0[k], v1[k]));
            let p0 = shr128(&avg, 8);
            let p1 = shr128(&v2, 8);
            let p2 = shrd128(&shl128(&v2, 8), &p0, 15);
            let p3 = shrd128(&shl128(&v2, 9), &p1, 15);
            let p4 = shrd128(&shl128(&v2, 10), &p2, 15);
            let p5 = shrd128(&shl128(&v2, 11), &p3, 15);
            let p6 = shrd128(&shl128(&v2, 12), &p4, 15);
            let p7 = shrd128(&shl128(&v2, 13), &p5, 15);
            for (row, pv) in [&p0 as &[u8; 16], &p1, &p2, &p3, &p4, &p5, &p6, &p7]
                .iter()
                .enumerate()
            {
                rows[8 * row..8 * row + 8].copy_from_slice(&pv[..8]);
            }
        }
        I8X8_HD => {
            let (a2h, jt) = (a2h(), jt());
            let a2p = shrd128(&a2h, &jt, 7);
            let a2q = shrd128(&a2h, &jt, 8);
            let b2r = shrd128(&a2h, &jt, 9);
            let v0: [u8; 16] = std::array::from_fn(|k| lowpass8(a2p[k], a2q[k], b2r[k]));
            let v1 = shr128(&v0, 1);
            let v2s = shr128(&v0, 2);
            let v2: [u8; 16] = std::array::from_fn(|k| lowpass8(v0[k], v1[k], v2s[k]));
            let avg: [u8; 16] = std::array::from_fn(|k| avgu8(v0[k], v1[k]));
            // p7 = ziplo8(avg, v2) (interleaved), p3 = ziphi64(p7, v2).
            let mut p7 = [0u8; 16];
            for k in 0..8 {
                p7[2 * k] = avg[k];
                p7[2 * k + 1] = v2[k];
            }
            let mut p3 = [0u8; 16];
            p3[..8].copy_from_slice(&p7[8..]);
            p3[8..].copy_from_slice(&v2[8..]);
            for row in 4..8 {
                let off = 2 * (7 - row);
                rows[8 * row..8 * row + 8].copy_from_slice(&p7[off..off + 8]);
            }
            for row in 0..4 {
                let off = 2 * (3 - row);
                rows[8 * row..8 * row + 8].copy_from_slice(&p3[off..off + 8]);
            }
        }
        I8X8_VL | I8X8_VL_C | I8X8_VL_D | I8X8_VL_CD => {
            let (j2y, k2z): ([u8; 16], [u8; 16]) = match mode {
                I8X8_VL => (jt(), std::array::from_fn(|k| t(k as i32))),
                I8X8_VL_C => {
                    let j2r: [u8; 16] = std::array::from_fn(|k| t(-8 + k as i32));
                    (shrc128(&j2r, 7), shrc128(&j2r, 8))
                }
                I8X8_VL_D => {
                    let k2z = std::array::from_fn(|k| t(k as i32));
                    (shlc128(&k2z, 1), k2z)
                }
                _ => {
                    let k2z: [u8; 16] =
                        std::array::from_fn(|k| if k < 8 { t(k as i32) } else { t(7) });
                    (shlc128(&k2z, 1), k2z)
                }
            };
            let k1 = shr128(&k2z, 1); // C VL uses shr128 (zero tail), unlike DDL's shrc128
            let v0: [u8; 16] = std::array::from_fn(|k| lowpass8(j2y[k], k2z[k], k1[k]));
            let v1 = shr128(&v0, 1);
            let p0: [u8; 16] = std::array::from_fn(|k| avgu8(v0[k], v1[k]));
            let v2s = shr128(&v0, 2);
            let p1: [u8; 16] = std::array::from_fn(|k| lowpass8(v0[k], v1[k], v2s[k]));
            for row in 0..8 {
                let src = if row % 2 == 0 { &p0 } else { &p1 };
                rows[8 * row..8 * row + 8].copy_from_slice(&src[row / 2..row / 2 + 8]);
            }
        }
        I8X8_HU | I8X8_HU_D => {
            let f = left8(if mode == I8X8_HU { t(-1) } else { l(0) }, l);
            // C `spreadh8`: [f[0..8], f[7]*8].
            let mut v0 = [0u8; 16];
            v0[..8].copy_from_slice(&f[..8]);
            v0[8..].fill(f[7]);
            let v1 = shr128(&v0, 1);
            let v2s = shr128(&v0, 2);
            // p0 = ziplo8(avgu8(v0, v1), lowpass8(...)) — interleaved.
            let mut p0 = [0u8; 16];
            for k in 0..8 {
                p0[2 * k] = avgu8(v0[k], v1[k]);
                p0[2 * k + 1] = lowpass8(v0[k], v1[k], v2s[k]);
            }
            let mut p4 = [0u8; 16];
            p4[..8].copy_from_slice(&p0[8..]);
            p4[8..].copy_from_slice(&v0[8..]);
            for row in 0..4 {
                rows[8 * row..8 * row + 8].copy_from_slice(&p0[2 * row..2 * row + 8]);
            }
            for row in 4..8 {
                let off = 2 * (row - 4);
                rows[8 * row..8 * row + 8].copy_from_slice(&p4[off..off + 8]);
            }
        }
        _ => unreachable!(),
    }

    for r in 0..8 {
        buf[o + r * stride..o + r * stride + 8].copy_from_slice(&rows[8 * r..8 * r + 8]);
    }
}

/// C `decode_intra16x16`. Block origin at `buf[o]`; writes 16 rows x 16 cols.
pub fn intra16x16(buf: &mut [u8], o: usize, stride: usize, mode: u32) {
    let t = |i: i32| px(buf, o, stride, -1, i);
    let l = |r: i32| px(buf, o, stride, r, -1);

    let mut rows = [0u8; 256];
    match mode {
        I16X16_V => {
            for (k, slot) in rows[..16].iter_mut().enumerate() {
                *slot = t(k as i32);
            }
            let row0: [u8; 16] = rows[..16].try_into().unwrap();
            for r in 1..16 {
                rows[16 * r..16 * r + 16].copy_from_slice(&row0);
            }
        }
        I16X16_H => {
            for r in 0..16 {
                rows[16 * r..16 * r + 16].fill(l(r as i32));
            }
        }
        I16X16_DC => {
            let s: u16 = (0..16i32).map(|k| t(k) as u16).sum::<u16>()
                + (0..16i32).map(|r| l(r) as u16).sum::<u16>();
            rows.fill(shrru16(s, 5) as u8);
        }
        I16X16_DC_A => {
            let s: u16 = (0..16i32).map(|k| t(k) as u16).sum();
            rows.fill(shrru16(s, 4) as u8);
        }
        I16X16_DC_B => {
            let s: i32 = (0..16i32).map(|r| l(r) as i32).sum();
            rows.fill(((s + 8) >> 4) as u8);
        }
        I16X16_DC_AB => rows.fill(128),
        I16X16_P => {
            // C `loadu64x2(pT - 1, pT + 8)` and `ldleftP16` (L7 skipped).
            let mut tv = [0u8; 16];
            for k in 0..8 {
                tv[k] = t(-1 + k as i32);
                tv[8 + k] = t(8 + k as i32);
            }
            let mut lv = [0u8; 16];
            lv[0] = t(-1);
            lv[1..8].copy_from_slice(&[l(0), l(1), l(2), l(3), l(4), l(5), l(6)]);
            lv[8..].copy_from_slice(&[l(8), l(9), l(10), l(11), l(12), l(13), l(14), l(15)]);
            let m: [i8; 16] = [8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8];
            let tm = maddubx(&tv, &m);
            let lm = maddubx(&lv, &m);
            let v0 = hadd16(&tm, &lm);
            // v1 = ((u32x4)v0 >> 16) + v0
            let v1: [i16; 8] = std::array::from_fn(|k| {
                if k % 2 == 0 {
                    v0[k].wrapping_add(v0[k + 1])
                } else {
                    v0[k]
                }
            });
            // HV = ((u64x2)v1 >> 32) - v1; shifted = [v1[2], v1[3], 0, 0, v1[6], v1[7], 0, 0]
            let sh: [i16; 8] = [v1[2], v1[3], 0, 0, v1[6], v1[7], 0, 0];
            let hv: [i16; 8] = std::array::from_fn(|k| sh[k].wrapping_sub(v1[k]));
            let v2: [i16; 8] = std::array::from_fn(|k| shrrs16(hv[k].wrapping_add(hv[k] >> 2), 4));
            let a = ((tv[15] as i32 + lv[15] as i32 + 1) << 4) as i16;
            let b = v2[0];
            let c = v2[4];
            let mul: [i16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
            let mut w0: [i16; 8] = std::array::from_fn(|k| {
                a.wrapping_add(c)
                    .wrapping_sub(c.wrapping_mul(8))
                    .wrapping_add(b.wrapping_mul(mul[k]))
            });
            let mut w1: [i16; 8] = std::array::from_fn(|k| w0[k].wrapping_sub(b.wrapping_mul(8)));
            for r in 0..16 {
                let packed = shrpus16(&w1, &w0, 5);
                rows[16 * r..16 * r + 16].copy_from_slice(&packed);
                for k in 0..8 {
                    w0[k] = w0[k].wrapping_add(c);
                    w1[k] = w1[k].wrapping_add(c);
                }
            }
        }
        _ => unreachable!(),
    }

    for r in 0..16 {
        buf[o + r * stride..o + r * stride + 16].copy_from_slice(&rows[16 * r..16 * r + 16]);
    }
}

/// C `decode_intraChroma`. Block origin at `buf[o]`; writes 16 rows x 8 cols.
pub fn intra_chroma(buf: &mut [u8], o: usize, stride: usize, mode: u32) {
    // Row -2 / row -1 samples and the left column (rows -2..15).
    let tu = |i: i32| px(buf, o, stride, -2, i);
    let tt = |i: i32| px(buf, o, stride, -1, i);
    let l = |r: i32| px(buf, o, stride, r, -1);

    let mut rows = [0u8; 128]; // 16 rows x 8 cols
    match mode {
        IC8X8_DC | IC8X8_DC_A | IC8X8_DC_B => {
            // t = loada64x2(pU, pT), l = ldleftC(p, stride, 0) (or duplicated).
            let mut tv = [0u8; 16];
            for k in 0..8 {
                tv[k] = tu(k as i32);
                tv[8 + k] = tt(k as i32);
            }
            // ldleftC(p, stride, 0): [L0, L2, L4, L6, L8, L10, L12, L14, L1, ... L15].
            let lvec: [u8; 16] = std::array::from_fn(|k| match k {
                0..=7 => l(2 * k as i32),
                _ => l(2 * (k as i32 - 8) + 1),
            });
            let (ta, la): ([u8; 16], [u8; 16]) = match mode {
                IC8X8_DC => (tv, lvec),
                IC8X8_DC_A => (tv, tv),
                _ => (lvec, lvec),
            };
            // trnlo32/trnhi32 as i32 lanes; sums of 8-byte halves.
            let t32 = [0, 1, 2, 3].map(|i| lane32(&ta, i));
            let l32 = [0, 1, 2, 3].map(|i| lane32(&la, i));
            let br: [[u8; 16]; 4] = [
                pack32(&[t32[0], l32[0], t32[2], l32[2]]),
                pack32(&[t32[1], t32[1], t32[3], t32[3]]),
                pack32(&[l32[1], l32[1], l32[3], l32[3]]),
                pack32(&[l32[1], t32[1], l32[3], t32[3]]),
            ];
            let s: [u16; 8] = std::array::from_fn(|i| {
                br[i / 2][(i % 2) * 8..(i % 2) * 8 + 8]
                    .iter()
                    .map(|&x| x as u16)
                    .sum()
            });
            // v0 = shrru16(s, 3); bpred/rpred pick its low bytes per shuf table.
            let q: [u8; 8] = std::array::from_fn(|i| shrru16(s[i], 3) as u8);
            let pick = |sel: &[usize; 4]| -> [u8; 16] {
                let mut r = [0u8; 16];
                for (k, &s) in sel.iter().enumerate() {
                    r[4 * k..4 * k + 4].fill(q[s]);
                }
                r
            };
            let (bpred, rpred): ([u8; 16], [u8; 16]) = match mode {
                IC8X8_DC => (pick(&[0, 2, 4, 6]), pick(&[1, 3, 5, 7])),
                IC8X8_DC_A => (pick(&[0, 6, 0, 6]), pick(&[1, 7, 1, 7])),
                _ => (pick(&[0, 0, 6, 6]), pick(&[1, 1, 7, 7])),
            };
            // C store: row 2i = bpred[0], row 2i+1 = rpred[0], row 2i+8 = bpred[1],
            // row 2i+9 = rpred[1] (p + stride*8 is 8 rows down, not 8 cols right).
            for i in 0..4 {
                rows[8 * (2 * i)..8 * (2 * i) + 8].copy_from_slice(&bpred[..8]);
                rows[8 * (2 * i + 1)..8 * (2 * i + 1) + 8].copy_from_slice(&rpred[..8]);
                rows[8 * (2 * i + 8)..8 * (2 * i + 8) + 8].copy_from_slice(&bpred[8..]);
                rows[8 * (2 * i + 9)..8 * (2 * i + 9) + 8].copy_from_slice(&rpred[8..]);
            }
        }
        IC8X8_DC_AB => rows.fill(128),
        IC8X8_H => {
            for r in 0..16 {
                rows[8 * r..8 * r + 8].fill(l(r as i32));
            }
        }
        IC8X8_V => {
            // C: bpred = set64(pU row), rpred = set64(pT row) — each 16 bytes.
            let mut v = [0u8; 16];
            for k in 0..8 {
                v[k] = tu(k as i32);
                v[8 + k] = tt(k as i32);
            }
            for r in 0..16 {
                // C writes 16 bytes/row (both halves equal); low 8 compared.
                let src = if r % 2 == 0 { &v[..8] } else { &v[8..] };
                rows[8 * r..8 * r + 8].copy_from_slice(src);
            }
        }
        IC8X8_P => {
            // t = ziplo64(shuffle(loadu128(pU-8), s), shuffle(loadu128(pT-8), s))
            // with s = {7,8,9,10,12,13,14,15}: cols -1..2 and 4..7 (col 3 skipped).
            const C: [i32; 8] = [-1, 0, 1, 2, 4, 5, 6, 7];
            let mut tv = [0u8; 16];
            for k in 0..8 {
                tv[k] = tu(C[k]);
                tv[8 + k] = tt(C[k]);
            }
            // l = ldleftC(pU, stride, 2*stride): p0=L-2, p4=L2, p8=L8, pC=L12.
            // [L-2, L0, L2, L4, L8, L10, L12, L14, L-1, L1, L3, L5, L9, L11, L13, L15].
            const R: [i32; 16] = [-2, 0, 2, 4, 8, 10, 12, 14, -1, 1, 3, 5, 9, 11, 13, 15];
            let lv: [u8; 16] = std::array::from_fn(|k| l(R[k]));
            let m: [i8; 16] = [4, 3, 2, 1, 1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4];
            let v0 = hadd16(&maddubx(&tv, &m), &maddubx(&lv, &m));
            // HV = ((u32x4)v0 >> 16) - v0: [Hb, _, Hr, _, Vb, _, Vr, _]
            let hv: [i16; 8] = std::array::from_fn(|k| {
                if k % 2 == 0 {
                    v0[k + 1].wrapping_sub(v0[k])
                } else {
                    v0[k].wrapping_neg()
                }
            });
            let v1: [i16; 8] = std::array::from_fn(|k| shrrs16(hv[k].wrapping_add(hv[k] >> 4), 1));
            // v2[k] = (t[2k+1] + l[2k+1] + 1) << 4
            let v2: [i16; 8] = std::array::from_fn(|k| {
                ((tv[2 * k + 1] as i32 + lv[2 * k + 1] as i32 + 1) << 4) as i16
            });
            let ba = v2[3];
            let ra = v2[7];
            let bb = v1[0];
            let rb = v1[2];
            let bc = v1[4];
            let rc = v1[6];
            let mut bp: [i16; 8] = std::array::from_fn(|k| {
                ba.wrapping_sub(bc.wrapping_mul(3))
                    .wrapping_add(bb.wrapping_mul((k as i32 - 3) as i16))
            });
            let mut rp: [i16; 8] = std::array::from_fn(|k| {
                ra.wrapping_sub(rc.wrapping_mul(3))
                    .wrapping_add(rb.wrapping_mul((k as i32 - 3) as i16))
            });
            for i in 0..8 {
                let packed = shrpus16(&bp, &rp, 5);
                rows[8 * (2 * i)..8 * (2 * i) + 8].copy_from_slice(&packed[..8]);
                rows[8 * (2 * i + 1)..8 * (2 * i + 1) + 8].copy_from_slice(&packed[8..]);
                for k in 0..8 {
                    bp[k] = bp[k].wrapping_add(bc);
                    rp[k] = rp[k].wrapping_add(rc);
                }
            }
        }
        _ => unreachable!(),
    }

    for r in 0..16 {
        buf[o + r * stride..o + r * stride + 8].copy_from_slice(&rows[8 * r..8 * r + 8]);
    }
}

/// C `store1_8x8`: replicate one 8-byte row to all 8 rows.
fn fill_rows(rows: &mut [u8; 64], row: &[u8; 16]) {
    for r in 0..8 {
        rows[8 * r..8 * r + 8].copy_from_slice(&row[..8]);
    }
}

/// i32 lane `i` of a 16-byte vector, as 4 bytes (little-endian order).
fn lane32(v: &[u8; 16], i: usize) -> [u8; 4] {
    v[4 * i..4 * i + 4].try_into().unwrap()
}

/// Pack four 4-byte groups into 16 bytes (i32 lane order).
fn pack32(lanes: &[[u8; 4]; 4]) -> [u8; 16] {
    let mut r = [0u8; 16];
    for i in 0..4 {
        r[4 * i..4 * i + 4].copy_from_slice(&lanes[i]);
    }
    r
}

//! Pure-Rust reimplementation of edge264's deblocking filter (Tier B).
//!
//! Faithful scalar port of `c/src/edge264_deblock.c`: the C kernel is SSE SIMD
//! that processes 16 samples per vector with heavy shuffling. Here we mirror the
//! exact operations in scalar form (no re-derivation of the H.264 algorithm)
//! and pin the results to golden hashes in `tests.rs`.
//!
//! The SIMD primitives below replicate the precise semantics of the intrinsics
//! used by the C code (saturating ops, packed unpacks, byte shuffles, 128-bit
//! shifts). Types are reinterpreted the same way the C code type-puns them.
//!
//! Identifiers intentionally mirror the C source (`edge264_deblock.c`) for
//! line-by-line traceability of the port, so `non_snake_case` is allowed here.
#![allow(non_snake_case)]

// --- vector types (mirror edge264's i8x16 / u8x16 / i16x8 / i32x4 / i64x2) ---
type V8 = [i8; 16];
type U8 = [u8; 16];
type V16 = [i16; 8];

/// Serialized macroblock state for the deblock tests: 202 bytes per MB,
/// 3 MBs in order {top, left, current}. Layout per MB: QP[3]@0,
/// mbIsInterFlag@3, filter_edges@4, inter_eqs_s@5 (LE u32),
/// transform_size_8x8_flag@9, nC[48]@10, refIdx[8]@58, refPic[8]@66,
/// mvs[64]@74 (i16 LE).
pub const DEBLOCK_MB_STATE_LEN: usize = 202;

// Canonical plane geometry of the deblock test inputs.
pub const DEBLOCK_LY_STRIDE: usize = 64;
pub const DEBLOCK_LY_ROWS: usize = 96;
pub const DEBLOCK_LY_SIZE: usize = DEBLOCK_LY_ROWS * DEBLOCK_LY_STRIDE;
pub const DEBLOCK_LC_STRIDE: usize = 32;
pub const DEBLOCK_LC_ROWS: usize = 96;
pub const DEBLOCK_LC_SIZE: usize = DEBLOCK_LC_ROWS * DEBLOCK_LC_STRIDE;
/// Current luma MB (16x16) top-left within the canonical luma plane.
pub const DEBLOCK_Y_ROW: usize = 48;
pub const DEBLOCK_Y_COL: usize = 32;
/// Cb row 0 of the current chroma MB (8x8). Cr row k is at buffer-row
/// DEBLOCK_C_ROW + 2k + 1; each run is 8 bytes at col DEBLOCK_C_COL.
pub const DEBLOCK_C_ROW: usize = 48;
pub const DEBLOCK_C_COL: usize = 8;
type V32 = [i32; 4];
type V64 = [i64; 2];

// --- reinterpretation (little-endian, like the C type puns) ---
fn v8_as_u8(v: V8) -> U8 {
    v.map(|x| x as u8)
}
fn u8_as_v8(v: U8) -> V8 {
    v.map(|x| x as i8)
}

/// Reinterpret 16 bytes as 8 little-endian i16.
fn v8_as_v16(v: &V8) -> V16 {
    let b = v8_as_u8(*v);
    let mut o = [0i16; 8];
    for i in 0..8 {
        o[i] = b[2 * i] as i16 | ((b[2 * i + 1] as i16) << 8);
    }
    o
}
fn v32_as_v8(v: &V32) -> V8 {
    let mut o = [0u8; 16];
    for i in 0..4 {
        o[4 * i] = (v[i] & 0xFF) as u8;
        o[4 * i + 1] = ((v[i] >> 8) & 0xFF) as u8;
        o[4 * i + 2] = ((v[i] >> 16) & 0xFF) as u8;
        o[4 * i + 3] = ((v[i] >> 24) & 0xFF) as u8;
    }
    u8_as_v8(o)
}
fn v8_as_v32(v: &V8) -> V32 {
    let b = v8_as_u8(*v);
    let mut o = [0i32; 4];
    for i in 0..4 {
        let w = (b[4 * i] as u32)
            | ((b[4 * i + 1] as u32) << 8)
            | ((b[4 * i + 2] as u32) << 16)
            | ((b[4 * i + 3] as u32) << 24);
        o[i] = w as i32;
    }
    o
}
fn v64_as_v8(v: &V64) -> V8 {
    let mut o = [0u8; 16];
    for i in 0..2 {
        for k in 0..8 {
            o[8 * i + k] = ((v[i] >> (8 * k)) & 0xFF) as u8;
        }
    }
    u8_as_v8(o)
}

// --- scalar intrinsic equivalents (SSE semantics) ---

/// _mm_set1_epi8
#[inline]
fn set8(i: i32) -> V8 {
    [i as i8; 16]
}

// The C code flows every filter operand as i8x16 and applies the saturating
// ops with unsigned semantics (via the macro casts). We mirror that: all of the
// following take/return V8 (=[i8;16]) and reinterpret bytes as unsigned where
// the intrinsic demands it. Bit patterns match the SSE intrinsics exactly.

/// _mm_subs_epu8: saturated unsigned subtract, max(a-b, 0), byte-wise.
#[inline]
fn subsu8(a: V8, b: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = ((a[i] as u8) as i16 - (b[i] as u8) as i16).max(0) as i8;
    }
    o
}
/// _mm_adds_epu8: saturated unsigned add, min(a+b, 255), byte-wise.
#[inline]
fn addsu8(a: V8, b: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = (((a[i] as u8) as u16 + (b[i] as u8) as u16).min(255)) as i8;
    }
    o
}
/// _mm_max_epu8: unsigned max, byte-wise.
#[inline]
fn maxu8(a: V8, b: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = (a[i] as u8).max(b[i] as u8) as i8;
    }
    o
}
/// _mm_min_epu8: unsigned min, byte-wise.
#[inline]
fn minu8(a: V8, b: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = (a[i] as u8).min(b[i] as u8) as i8;
    }
    o
}
/// _mm_avg_epu8: (a + b + 1) >> 1, unsigned, byte-wise.
#[inline]
fn avgu8(a: V8, b: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = (((a[i] as u8) as u16 + (b[i] as u8) as u16 + 1) >> 1) as i8;
    }
    o
}

// --- raw byte-wise ops. In the C build these are GCC vector int8_t operators,
// i.e. element-wise wrapping arithmetic (mod 256) and plain bitwise logic. ---
#[inline]
fn bor(a: V8, b: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = a[i] | b[i];
    }
    o
}
#[inline]
fn band(a: V8, b: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = a[i] & b[i];
    }
    o
}
#[inline]
fn bxor(a: V8, b: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = a[i] ^ b[i];
    }
    o
}
#[inline]
fn bnot(a: V8) -> V8 {
    a.map(|x| !x)
}
/// Wrapping byte add (GCC vector `+`).
#[inline]
fn bwadd(a: V8, b: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = a[i].wrapping_add(b[i]);
    }
    o
}
/// Wrapping byte subtract (GCC vector `-`).
#[inline]
fn bwsub(a: V8, b: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = a[i].wrapping_sub(b[i]);
    }
    o
}

// --- per-lane comparisons producing a mask (0xFF byte where true, 0x00 else) ---
#[inline]
fn cmplt0(a: V8) -> V8 {
    a.map(|x| if x < 0 { -1 } else { 0 })
}
#[inline]
fn cmpeq0(a: V8) -> V8 {
    a.map(|x| if x == 0 { -1 } else { 0 })
}
#[inline]
fn cmpgt0(a: V8) -> V8 {
    a.map(|x| if x > 0 { -1 } else { 0 })
}

/// sat16 -> sat8 (signed), the narrowing used by _mm_packs_epi16.
#[inline]
fn pack_sat8(x: i16) -> i8 {
    x.clamp(i8::MIN as i16, i8::MAX as i16) as i8
}

/// _mm_packs_epi16(a, b): result = [sat8(a_l0..7), sat8(b_l0..7)] — all 16
/// bytes are defined (hardware-verified; NOT low4(a)+high4(b) with a zeroed
/// upper half).
fn packs16(a: &V16, b: &V16) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..8 {
        o[i] = pack_sat8(a[i]);
        o[8 + i] = pack_sat8(b[i]);
    }
    o
}

/// _mm_unpacklo/hi_epi8: interleave bytes.
#[inline]
fn ziplo8(a: V8, b: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..8 {
        o[2 * i] = a[i];
        o[2 * i + 1] = b[i];
    }
    o
}
#[inline]
fn ziphi8(a: V8, b: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..8 {
        o[2 * i] = a[8 + i];
        o[2 * i + 1] = b[8 + i];
    }
    o
}
#[inline]
fn ziplo32(a: V32, b: V32) -> V32 {
    [a[0], b[0], a[1], b[1]]
}
#[inline]
fn ziphi32(a: V32, b: V32) -> V32 {
    [a[2], b[2], a[3], b[3]]
}
/// _mm_shuffle_epi32(_mm_shuffle_ps(a,b,_MM_SHUFFLE(2,0,2,0)),_MM_SHUFFLE(3,1,2,0))
/// = trnlo32: [a0, b0, a2, b2].
#[inline]
fn trnlo32(a: V32, b: V32) -> V32 {
    [a[0], b[0], a[2], b[2]]
}

/// _mm_shuffle_epi8: out[i] = a[shuf[i]&0xF] if shuf[i] high bit clear, else 0.
#[inline]
fn shuffle(a: V8, shuf: &[i8; 16]) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        let idx = shuf[i];
        o[i] = if idx < 0 { 0 } else { a[(idx & 0x0F) as usize] };
    }
    o
}

/// shuffle3(p, m): gather from a flat table of 48/144 bytes (3 or 9 i8x16).
/// out[i] = table[m[i]] where m[i] in 0..len. Mirrors the C `shuffle3` which
/// selects across three consecutive 128-bit vectors by byte index.
#[inline]
fn shuffle3(table: &[u8], m: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = table[(m[i] & 0x7F) as usize] as i8;
    }
    o
}

/// C `unziplo32` = shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0)): [a0, a2, b0, b2].
#[inline]
fn unziplo32(a: V32, b: V32) -> V32 {
    [a[0], a[2], b[0], b[2]]
}
/// C `unziphi32` = shuffle_ps(a, b, _MM_SHUFFLE(3,1,3,1)): [a1, a3, b1, b3].
#[inline]
fn unziphi32(a: V32, b: V32) -> V32 {
    [a[1], a[3], b[1], b[3]]
}

/// _mm_srli_si128: shift the 16 bytes right by `n` bytes, zero-fill.
#[inline]
fn shr128(a: V8, n: i32) -> V8 {
    let mut o = [0i8; 16];
    if !(0..16).contains(&n) {
        return o;
    }
    let n = n as usize;
    // C `_mm_srli_si128`: data shifts toward index 0, zero-fill the tail.
    o[..16 - n].copy_from_slice(&a[n..]);
    o
}

/// _mm_alignr_epi8(h, l, n): bytes [n..n+16) of the 32-byte concat [l (0-15), h (16-31)].
#[inline]
fn shrd128(l: V8, h: V8, n: i32) -> V8 {
    let mut cat = [0i8; 32];
    cat[..16].copy_from_slice(&l);
    cat[16..].copy_from_slice(&h);
    let mut o = [0i8; 16];
    let n = if n < 0 { 0 } else { n as usize };
    for (i, slot) in o.iter_mut().enumerate() {
        let idx = n + i;
        if idx < 32 {
            *slot = cat[idx];
        }
    }
    o
}

/// _mm_blendv_epi8(f, t, v): per byte, v<0 ? t : f. (mask selects t where set)
#[inline]
fn ifelse_mask(mask: V8, t: V8, f: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = if mask[i] < 0 { t[i] } else { f[i] };
    }
    o
}

/// broadcast8(a, i) = shuffle(a, set8(i)) = all lanes = a[i].
#[inline]
fn broadcast8(a: V8, i: i32) -> V8 {
    [a[i as usize]; 16]
}

/// expand4(int32): each of the 4 bytes of `a` repeated 4 times.
fn expand4(a: i32) -> V8 {
    let bs = [
        (a & 0xFF) as i8,
        ((a >> 8) & 0xFF) as i8,
        ((a >> 16) & 0xFF) as i8,
        ((a >> 24) & 0xFF) as i8,
    ];
    let mut o = [0i8; 16];
    for (i, bb) in bs.iter().enumerate() {
        for k in 0..4 {
            o[4 * i + k] = *bb;
        }
    }
    o
}

/// expand2(int64): the 8 bytes of `a` each duplicated.
fn expand2(a: i64) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..8 {
        let b = ((a >> (8 * i)) & 0xFF) as i8;
        o[2 * i] = b;
        o[2 * i + 1] = b;
    }
    o
}

/// _mm_subs_epi16: plain (wrapping) signed 16-bit subtract.
#[inline]
fn subs16(a: V16, b: V16) -> V16 {
    let mut o = [0i16; 8];
    for i in 0..8 {
        o[i] = a[i].wrapping_sub(b[i]);
    }
    o
}

/// abs8: per-byte absolute value (|x|, with -128 -> 128 as unsigned).
#[inline]
fn abs8(v: V8) -> U8 {
    let mut o = [0u8; 16];
    for i in 0..16 {
        o[i] = (if v[i] < 0 {
            -(v[i] as i16)
        } else {
            v[i] as i16
        }) as u8;
    }
    o
}

/// packabd16(a,b,c,d) = abs8(packs16(sub16(a,b), sub16(c,d))):
/// |a-b| and |c-d| narrowed to 8 bits (clamped to 0..127 magnitude).
/// Returns V8 (the C u8x16, reinterpreted as i8x16 for downstream ops).
fn packabd16(a: V16, b: V16, c: V16, d: V16) -> V8 {
    u8_as_v8(abs8(packs16(&subs16(a, b), &subs16(c, d))))
}

/// Per-lane equality mask (0xFF where a[i]==b[i]).
#[inline]
fn cmpeq(a: V8, b: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = if a[i] == b[i] { -1 } else { 0 };
    }
    o
}
/// _mm_max_epi8: signed max, byte-wise.
#[inline]
fn max8s(a: V8, b: V8) -> V8 {
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = a[i].max(b[i]);
    }
    o
}

/// Reinterpret 8 i16 as 4 little-endian i32.
#[inline]
fn v16_as_v32(v: &V16) -> V32 {
    let mut o = [0i32; 4];
    for i in 0..4 {
        // mask halves to 16 bits: a signed i16->u32 cast sign-extends into the
        // upper half and would corrupt the OR otherwise.
        o[i] = (((v[2 * i] as u32) & 0xFFFF) | (((v[2 * i + 1] as u32) & 0xFFFF) << 16)) as i32;
    }
    o
}
/// Reinterpret 4 i32 as 8 little-endian i16.
#[inline]
fn v32_as_v16(v: &V32) -> V16 {
    let mut o = [0i16; 8];
    for i in 0..4 {
        o[2 * i] = (v[i] & 0xFFFF) as i16;
        o[2 * i + 1] = ((v[i] >> 16) & 0xFFFF) as i16;
    }
    o
}
/// Reinterpret 8 i16 as 2 little-endian i64.
#[inline]
fn v16_as_v64(v: &V16) -> V64 {
    let mut o = [0i64; 2];
    for h in 0..2 {
        let mut w: u64 = 0;
        for k in 0..4 {
            w |= ((v[4 * h + k] as u16) as u64) << (16 * k);
        }
        o[h] = w as i64;
    }
    o
}
/// Reinterpret 2 i64 as 8 little-endian i16.
#[inline]
fn v64_as_v16(v: &V64) -> V16 {
    let mut o = [0i16; 8];
    for h in 0..2 {
        for k in 0..4 {
            o[4 * h + k] = ((v[h] >> (16 * k)) & 0xFFFF) as i16;
        }
    }
    o
}

// ============================================================================
// Deblocking filter kernels (SSE branch of edge264_deblock.c, ported to V8).
// Each operates on 16 lanes where each lane is one position along an edge.
// The caller gathers the edge's samples into these vectors and scatters back.
// ============================================================================

const SHUFAB: [i8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2];

/// DEBLOCK_LUMA_SOFT (edge264_deblock.c:95). bS in [0..3].
/// Returns the updated (p1, p0, q0, q1); p2/q2 unchanged.
#[inline]
#[allow(clippy::too_many_arguments)]
fn luma_soft(
    p2: V8,
    mut p1: V8,
    mut p0: V8,
    mut q0: V8,
    mut q1: V8,
    q2: V8,
    ialpha: i32,
    ibeta: i32,
    itC0: i32,
) -> (V8, V8, V8, V8) {
    let pq0 = subsu8(p0, q0);
    let qp0 = subsu8(q0, p0);
    let sub0 = subsu8(addsu8(qp0, set8(-128)), pq0); // 128+q0-p0
    let abs0 = bor(pq0, qp0);
    let abs1 = bor(subsu8(p1, p0), subsu8(p0, p1));
    let abs2 = bor(subsu8(q1, q0), subsu8(q0, q1));
    let beta = set8(ibeta);
    let and = minu8(subsu8(set8(ialpha), abs0), subsu8(beta, maxu8(abs1, abs2)));
    let tC0 = expand4(itC0);
    let ignore = bor(cmpeq0(and), cmplt0(tC0)); // (and==0)|(tC0<0)
    let ftC0 = band(tC0, bnot(ignore));
    let c1 = set8(1);
    let x0 = avgu8(p0, q0); // (p0+q0+1)>>1
    let x1 = bwsub(avgu8(p2, x0), band(bxor(p2, x0), c1));
    let x2 = bwsub(avgu8(q2, x0), band(bxor(q2, x0), c1));
    let pp1 = minu8(maxu8(x1, subsu8(p1, ftC0)), addsu8(p1, ftC0));
    let qp1 = minu8(maxu8(x2, subsu8(q1, ftC0)), addsu8(q1, ftC0));
    let cm1 = set8(-1);
    let bm1 = bwadd(beta, cm1);
    let sub1 = avgu8(p1, bxor(q1, cm1)); // 128+((p1-q1)>>1)
    let apltb = cmpeq(subsu8(subsu8(p2, p0), bm1), subsu8(subsu8(p0, p2), bm1));
    let aqltb = cmpeq(subsu8(subsu8(q2, q0), bm1), subsu8(subsu8(q0, q2), bm1));
    p1 = ifelse_mask(apltb, pp1, p1);
    q1 = ifelse_mask(aqltb, qp1, q1);
    let ftC = band(bwsub(bwsub(ftC0, apltb), aqltb), bnot(ignore));
    let x3 = avgu8(sub0, avgu8(sub1, set8(127)));
    let c128 = set8(-128);
    let delta = minu8(subsu8(x3, c128), ftC);
    let ndelta = minu8(subsu8(c128, x3), ftC);
    p0 = subsu8(addsu8(p0, delta), ndelta);
    q0 = subsu8(addsu8(q0, ndelta), delta);
    (p1, p0, q0, q1)
}

/// DEBLOCK_CHROMA_SOFT (edge264_deblock.c:130). alpha/beta packed as
/// int32 {left_Y,left_Cb,left_Cr,0}; shufab picks Cb (low 8) / Cr (high 8).
/// Returns updated (p0, q0); p1/q1 unchanged.
#[inline]
fn chroma_soft(
    p1: V8,
    mut p0: V8,
    mut q0: V8,
    q1: V8,
    ialpha: i32,
    ibeta: i32,
    itC0: i64,
) -> (V8, V8) {
    let c128 = set8(-128);
    let pq0 = subsu8(p0, q0);
    let qp0 = subsu8(q0, p0);
    let sub0 = subsu8(addsu8(qp0, c128), pq0); // 128+q0-p0
    let abs0 = bor(pq0, qp0);
    let abs1 = bor(subsu8(p1, p0), subsu8(p0, p1));
    let abs2 = bor(subsu8(q1, q0), subsu8(q0, q1));
    let alpha = shuffle(v32_as_v8(&[ialpha; 4]), &SHUFAB);
    let beta = shuffle(v32_as_v8(&[ibeta; 4]), &SHUFAB);
    let and = minu8(subsu8(alpha, abs0), subsu8(beta, maxu8(abs1, abs2)));
    let ignore = cmpeq0(and);
    let cm1 = set8(-1);
    let ftC = band(bwsub(expand2(itC0), cm1), bnot(ignore));
    let sub1 = avgu8(p1, bxor(q1, cm1)); // 128+((p1-q1)>>1)
    let x3 = avgu8(sub0, avgu8(sub1, set8(127)));
    let delta = minu8(subsu8(x3, c128), ftC);
    let ndelta = minu8(subsu8(c128, x3), ftC);
    p0 = subsu8(addsu8(p0, delta), ndelta);
    q0 = subsu8(addsu8(q0, ndelta), delta);
    (p0, q0)
}

/// DEBLOCK_LUMA_HARD (edge264_deblock.c:213). bS=4. Uses beta-1 (not for beta=0).
/// Returns updated (p0, p1, p2, q0, q1, q2); p3/q3 unchanged.
#[inline]
#[allow(clippy::too_many_arguments)]
fn luma_hard(
    p3: V8,
    mut p2: V8,
    mut p1: V8,
    mut p0: V8,
    mut q0: V8,
    mut q1: V8,
    mut q2: V8,
    q3: V8,
    ialpha: i32,
    ibeta: i32,
) -> (V8, V8, V8, V8, V8, V8) {
    let alpha = set8(ialpha);
    let beta = set8(ibeta);
    let abs0 = bor(subsu8(p0, q0), subsu8(q0, p0));
    let abs1 = bor(subsu8(p1, p0), subsu8(p0, p1));
    let abs2 = bor(subsu8(q1, q0), subsu8(q0, q1));
    let ignore = cmpeq0(minu8(subsu8(alpha, abs0), subsu8(beta, maxu8(abs1, abs2))));
    let c1 = set8(1);
    let zero = [0i8; 16];
    // C uses subsu8 (unsigned saturating) here: max(|diff|-(beta-1), 0) == 0
    // is the "<= beta-1" test; a wrapping sub would go negative and fail it.
    let condpq = subsu8(abs0, avgu8(avgu8(alpha, c1), zero)); // abs0-((alpha>>2)+1), sat to 0
    let bm1 = bwsub(beta, c1);
    let condp = cmpeq0(bor(
        subsu8(bor(subsu8(p2, p0), subsu8(p0, p2)), bm1),
        condpq,
    ));
    let condq = cmpeq0(bor(
        subsu8(bor(subsu8(q2, q0), subsu8(q0, q2)), bm1),
        condpq,
    ));
    let fix0 = band(bxor(p0, q0), c1);
    let pq0 = bwsub(avgu8(p0, q0), fix0);
    let and0 = bxor(fix0, c1);
    let p2q1 = bwsub(avgu8(p2, q1), band(bxor(p2, q1), c1));
    let q2p1 = bwsub(avgu8(q2, p1), band(bxor(q2, p1), c1));
    let p21q1 = bwsub(avgu8(p2q1, p1), band(bxor(p2q1, p1), and0));
    let q21p1 = bwsub(avgu8(q2p1, q1), band(bxor(q2p1, q1), and0));
    let pp0a = avgu8(p21q1, pq0);
    let qp0a = avgu8(q21p1, pq0);
    let pp0b = avgu8(p1, bwsub(avgu8(p0, q1), band(bxor(p0, q1), c1)));
    let qp0b = avgu8(q1, bwsub(avgu8(q0, p1), band(bxor(q0, p1), c1)));
    p0 = ifelse_mask(ignore, p0, ifelse_mask(condp, pp0a, pp0b));
    q0 = ifelse_mask(ignore, q0, ifelse_mask(condq, qp0a, qp0b));
    let fcondp = band(condp, bnot(ignore));
    let fcondq = band(condq, bnot(ignore));
    let p21 = bwsub(avgu8(p2, p1), band(bxor(p2, p1), and0));
    let q21 = bwsub(avgu8(q2, q1), band(bxor(q2, q1), and0));
    let pp1 = avgu8(p21, pq0);
    let qp1 = avgu8(q21, pq0);
    p1 = ifelse_mask(fcondp, pp1, p1);
    q1 = ifelse_mask(fcondq, qp1, q1);
    let fix1 = band(bxor(p21, pq0), c1);
    let fix2 = band(bxor(q21, pq0), c1);
    let p210q0 = bwsub(pp1, fix1);
    let q210p0 = bwsub(qp1, fix2);
    let p3p2 = bwsub(avgu8(p3, p2), band(bxor(p3, p2), bxor(fix1, c1)));
    let q3q2 = bwsub(avgu8(q3, q2), band(bxor(q3, q2), bxor(fix2, c1)));
    let pp2 = avgu8(p3p2, p210q0);
    let qp2 = avgu8(q3q2, q210p0);
    p2 = ifelse_mask(fcondp, pp2, p2);
    q2 = ifelse_mask(fcondq, qp2, q2);
    (p0, p1, p2, q0, q1, q2)
}

/// DEBLOCK_CHROMA_HARD (edge264_deblock.c:261). bS=4.
/// Returns updated (p0, q0); p1/q1 unchanged.
#[inline]
fn chroma_hard(p1: V8, mut p0: V8, mut q0: V8, q1: V8, ialpha: i32, ibeta: i32) -> (V8, V8) {
    let abs0 = bor(subsu8(p0, q0), subsu8(q0, p0));
    let abs1 = bor(subsu8(p1, p0), subsu8(p0, p1));
    let abs2 = bor(subsu8(q1, q0), subsu8(q0, q1));
    let alpha = shuffle(v32_as_v8(&[ialpha; 4]), &SHUFAB);
    let beta = shuffle(v32_as_v8(&[ibeta; 4]), &SHUFAB);
    let and = minu8(subsu8(alpha, abs0), subsu8(beta, maxu8(abs1, abs2)));
    let ignore = cmpeq0(and);
    let c1 = set8(1);
    let pp0b = avgu8(p1, bwsub(avgu8(p0, q1), band(bxor(p0, q1), c1)));
    let qp0b = avgu8(q1, bwsub(avgu8(q0, p1), band(bxor(q0, p1), c1)));
    p0 = ifelse_mask(ignore, p0, pp0b);
    q0 = ifelse_mask(ignore, q0, qp0b);
    (p0, q0)
}

// ============================================================================
// MB state + deblock parameter computation (deblock_mb, edge264_deblock.c:927).
// ============================================================================

/// idx2alpha (edge264_deblock.c:929): 3 x i8x16 = 48 bytes.
const IDX2ALPHA: [u8; 48] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20, 22, 25, 28,
    32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226, 255, 255,
];
/// idx2beta (edge264_deblock.c:931).
const IDX2BETA: [u8; 48] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8, 9, 9, 10,
    10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];
/// idx2tC0[0] (bS=1) (edge264_deblock.c:934).
const IDX2TC0_0: [u8; 48] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13,
];
/// idx2tC0[1] (bS=2) (edge264_deblock.c:935).
const IDX2TC0_1: [u8; 48] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 10, 11, 12, 13, 15, 17,
];
/// idx2tC0[2] (bS=3 / intra) (edge264_deblock.c:936).
const IDX2TC0_2: [u8; 48] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
    2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 23, 25,
];

const SHUFVHAB: [i8; 16] = [0, 2, 1, 3, 0, 1, 2, 3, 9, 11, 0, 2, 14, 15, 0, 1];
const SHUFV: [i8; 16] = [0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15];
const SHUFH: [i8; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
const SHUF_INTRA: [i8; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 2, 2, 2, 2];
const SHUF0: [i8; 16] = [8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const SHUF1: [i8; 16] = [12, 12, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const SHUF2: [i8; 16] = [9, 9, 9, 9, 10, 10, 10, 10, 1, 1, 1, 1, 2, 2, 2, 2];
const SHUF3: [i8; 16] = [13, 13, 13, 13, 14, 14, 14, 14, 1, 1, 1, 1, 2, 2, 2, 2];

/// A macroblock's deblock-relevant state (mirrors Edge264Macroblock).
#[derive(Clone, Copy)]
pub(crate) struct Mb {
    qy: i32,
    qcb: i32,
    qcr: i32,
    inter: i32,
    fe: i32,
    inter_eqs_s: u32,
    ts8x8: i32,
    nC: [i8; 48],
    refIdx: [i8; 8],
    refPic: [i8; 8],
    mvs: [i16; 64],
}

impl Mb {
    /// nC_v[0]: the first i8x16 of the coded-block pattern.
    #[inline]
    fn nC_v0(&self) -> V8 {
        let mut o = [0i8; 16];
        o.copy_from_slice(&self.nC[0..16]);
        o
    }

    /// Re-serialize in the 202-byte layout (test round-trip helper).
    #[cfg(test)]
    pub(crate) fn to_bytes(self) -> [u8; 202] {
        let mut s = [0u8; 202];
        s[0] = self.qy as u8;
        s[1] = self.qcb as u8;
        s[2] = self.qcr as u8;
        s[3] = self.inter as u8;
        s[4] = self.fe as u8;
        s[5..9].copy_from_slice(&self.inter_eqs_s.to_le_bytes());
        s[9] = self.ts8x8 as u8;
        for (i, v) in s[10..58].iter_mut().enumerate() {
            *v = self.nC[i] as u8;
        }
        for (i, v) in s[58..66].iter_mut().enumerate() {
            *v = self.refIdx[i] as u8;
        }
        for (i, v) in s[66..74].iter_mut().enumerate() {
            *v = self.refPic[i] as u8;
        }
        for i in 0..64 {
            s[74 + 2 * i..76 + 2 * i].copy_from_slice(&self.mvs[i].to_le_bytes());
        }
        s
    }
}

/// Parse one 202-byte serialized MB (see sw264_test_deblock_run).
pub(crate) fn parse_mb(s: &[u8]) -> Mb {
    let mut nC = [0i8; 48];
    for i in 0..48 {
        nC[i] = s[10 + i] as i8;
    }
    let mut refIdx = [0i8; 8];
    for i in 0..8 {
        refIdx[i] = s[58 + i] as i8;
    }
    let mut refPic = [0i8; 8];
    for i in 0..8 {
        refPic[i] = s[66 + i] as i8;
    }
    let mut mvs = [0i16; 64];
    for i in 0..64 {
        mvs[i] = i16::from_le_bytes([s[74 + 2 * i], s[75 + 2 * i]]);
    }
    Mb {
        qy: s[0] as i32,
        qcb: s[1] as i32,
        qcr: s[2] as i32,
        inter: s[3] as i32,
        fe: s[4] as i32,
        inter_eqs_s: u32::from_le_bytes([s[5], s[6], s[7], s[8]]),
        ts8x8: s[9] as i32,
        nC,
        refIdx,
        refPic,
        mvs,
    }
}

/// mvs_v[i]: the i-th i16x8 of mb->mvs (L0 blocks 0-15 in [0..3], L1 in [4..7]).
#[inline]
fn mvsv(mb: &Mb, i: usize) -> V16 {
    let mut o = [0i16; 8];
    for (k, slot) in o.iter_mut().enumerate() {
        *slot = mb.mvs[8 * i + k];
    }
    o
}
/// refIdx_s[which] as a signed int32 (little-endian of refIdx[which*4..+4]).
#[inline]
fn ri32(mb: &Mb, which: usize) -> i32 {
    i32::from_le_bytes([
        mb.refIdx[which * 4] as u8,
        mb.refIdx[which * 4 + 1] as u8,
        mb.refIdx[which * 4 + 2] as u8,
        mb.refIdx[which * 4 + 3] as u8,
    ])
}
/// refPic_s[which] as a signed int32.
#[inline]
fn rp32(mb: &Mb, which: usize) -> i32 {
    i32::from_le_bytes([
        mb.refPic[which * 4] as u8,
        mb.refPic[which * 4 + 1] as u8,
        mb.refPic[which * 4 + 2] as u8,
        mb.refPic[which * 4 + 3] as u8,
    ])
}
/// refPic_l: full 8 refPic bytes as a little-endian int64.
#[inline]
fn rpl(mb: &Mb) -> i64 {
    let mut b = [0u8; 8];
    for (i, slot) in b.iter_mut().enumerate() {
        *slot = mb.refPic[i] as u8;
    }
    i64::from_le_bytes(b)
}

// i16x8 <-> i32x4 / i64x2 shuffles (reinterpret then permute).
#[inline]
fn uzp1(a: V16, b: V16) -> V16 {
    v32_as_v16(&unziplo32(v16_as_v32(&a), v16_as_v32(&b)))
}
#[inline]
fn uzp2(a: V16, b: V16) -> V16 {
    v32_as_v16(&unziphi32(v16_as_v32(&a), v16_as_v32(&b)))
}
#[inline]
fn z64lo(a: V16, b: V16) -> V16 {
    let av = v16_as_v64(&a);
    let bv = v16_as_v64(&b);
    v64_as_v16(&[av[0], bv[0]])
}
#[inline]
fn z64hi(a: V16, b: V16) -> V16 {
    let av = v16_as_v64(&a);
    let bv = v16_as_v64(&b);
    v64_as_v16(&[av[1], bv[1]])
}
/// unziplo for i64x2 (SSE `_mm_shuffle_ps(.., _MM_SHUFFLE(2,0,2,0))`):
/// low 32 bits of each 64-bit lane — [a_lo0, a_lo1, b_lo0, b_lo1].
#[inline]
fn uzp1_v64(a: V64, b: V64) -> V64 {
    [
        (a[0] & 0xFFFF_FFFF) | ((a[1] & 0xFFFF_FFFF) << 32),
        (b[0] & 0xFFFF_FFFF) | ((b[1] & 0xFFFF_FFFF) << 32),
    ]
}
/// unziphi for i64x2 (SSE `_mm_shuffle_ps(.., _MM_SHUFFLE(3,1,3,1))`):
/// high 32 bits of each 64-bit lane — [a_hi0, a_hi1, b_hi0, b_hi1].
/// Mask after the arithmetic shift: a negative top byte must not
/// sign-extend into the neighboring lane's slot.
#[inline]
fn uzp2_v64(a: V64, b: V64) -> V64 {
    [
        ((a[0] >> 32) & 0xFFFF_FFFF) | (((a[1] >> 32) & 0xFFFF_FFFF) << 32),
        ((b[0] >> 32) & 0xFFFF_FFFF) | (((b[1] >> 32) & 0xFFFF_FFFF) << 32),
    ]
}

const ZERO_V16: V16 = [0i16; 8];
const ZERO_V32: V32 = [0i32; 4];

#[inline]
fn zero_v16() -> V16 {
    ZERO_V16
}

/// Compute alpha[16], beta[16] and tC0_s[16] (deblock_mb, edge264_deblock.c:927).
pub(crate) fn deblock_params(
    mbs: &[Mb; 3],
    entropy: i32,
    offA: i32,
    offB: i32,
) -> ([u8; 16], [u8; 16], [i32; 16]) {
    let mb = &mbs[2];
    let mbA = if mb.fe & 1 != 0 { &mbs[1] } else { mb };
    let mbB = if mb.fe & 2 != 0 { &mbs[0] } else { mb };

    // --- alpha / beta (QP averaging: mid/mid/A/B per plane) ---
    let av = |a: i32, b: i32| ((a + b + 1) >> 1) as i8;
    let qPav: V8 = [
        mb.qy as i8,
        mb.qcb as i8,
        mb.qcr as i8,
        0,
        mb.qy as i8,
        mb.qcb as i8,
        mb.qcr as i8,
        0,
        av(mb.qy, mbA.qy),
        av(mb.qcb, mbA.qcb),
        av(mb.qcr, mbA.qcr),
        0,
        av(mb.qy, mbB.qy),
        av(mb.qcb, mbB.qcb),
        av(mb.qcr, mbB.qcr),
        0,
    ];
    let zero: V8 = [0; 16];
    let indexA = minu8(max8s(bwadd(qPav, set8(offA)), zero), set8(51));
    let indexB = minu8(max8s(bwadd(qPav, set8(offB)), zero), set8(51));
    let Am4 = subsu8(indexA, set8(4));
    let alpha_v = shuffle3(&IDX2ALPHA, Am4);
    let beta_v = shuffle3(&IDX2BETA, subsu8(indexB, set8(4)));

    // --- tC0 (bS-dependent) ---
    let mut tC0_v: [V8; 4] = [[0i8; 16]; 4];
    if mb.inter == 0 {
        // intra: tC0 from idx2tC0[2]
        let tC03 = shuffle3(&IDX2TC0_2, Am4);
        let b0 = broadcast8(tC03, 0);
        tC0_v[0] = b0;
        tC0_v[1] = b0;
        tC0_v[2] = shuffle(tC03, &SHUF_INTRA);
        tC0_v[3] = tC0_v[2];
    } else {
        let tC01 = shuffle3(&IDX2TC0_0, Am4);
        let tC02 = shuffle3(&IDX2TC0_1, Am4);
        let c3 = set8(3);
        let (bS0aceg, bS0bdfh) = if (ri32(mb, 1) & ri32(mbA, 1) & ri32(mbB, 1)) == -1 {
            // P macroblocks
            let mvsv0 = uzp2(mvsv(mbA, 1), mvsv(mbA, 3));
            let mvsv1 = uzp1(mvsv(mb, 0), mvsv(mb, 2));
            let mvsv2 = uzp2(mvsv(mb, 0), mvsv(mb, 2));
            let mvsv3 = uzp1(mvsv(mb, 1), mvsv(mb, 3));
            let mvsv4 = uzp2(mvsv(mb, 1), mvsv(mb, 3));
            let mvsh0 = z64hi(mvsv(mbB, 2), mvsv(mbB, 3));
            let mvsh1 = z64lo(mvsv(mb, 0), mvsv(mb, 1));
            let mvsh2 = z64hi(mvsv(mb, 0), mvsv(mb, 1));
            let mvsh3 = z64lo(mvsv(mb, 2), mvsv(mb, 3));
            let mvsh4 = z64hi(mvsv(mb, 2), mvsv(mb, 3));
            let mvsac = packabd16(mvsv0, mvsv1, mvsv2, mvsv3);
            let mvsbd = packabd16(mvsv1, mvsv2, mvsv3, mvsv4);
            let mvseg = packabd16(mvsh0, mvsh1, mvsh2, mvsh3);
            let mvsfh = packabd16(mvsh1, mvsh2, mvsh3, mvsh4);
            let mvsaceg = packs16(
                &v8_as_v16(&subsu8(mvsac, c3)),
                &v8_as_v16(&subsu8(mvseg, c3)),
            );
            let mvsbdfh = packs16(
                &v8_as_v16(&subsu8(mvsbd, c3)),
                &v8_as_v16(&subsu8(mvsfh, c3)),
            );
            let refs = shuffle(
                v32_as_v8(&[rp32(mb, 0), 0, rp32(mbA, 0), rp32(mbB, 0)]),
                &SHUFVHAB,
            );
            let neq = bxor(refs, shr128(refs, 8));
            let refsaceg = ziplo8(neq, neq);
            let bS0aceg = cmpeq0(bor(refsaceg, mvsaceg));
            let bS0bdfh = cmpeq0(mvsbdfh);
            (bS0aceg, bS0bdfh)
        } else if mb.inter_eqs_s == 0x1b5fbbff {
            // 16x16 B macroblock
            let mvsv0l0 = uzp2(mvsv(mbA, 1), mvsv(mbA, 3));
            let mvsv1l0 = uzp1(mvsv(mb, 0), mvsv(mb, 2));
            let mvsv0l1 = uzp2(mvsv(mbA, 5), mvsv(mbA, 7));
            let mvsv1l1 = uzp1(mvsv(mb, 4), mvsv(mb, 6));
            let mvsh0l0 = z64hi(mvsv(mbB, 2), mvsv(mbB, 3));
            let mvsh1l0 = z64lo(mvsv(mb, 0), mvsv(mb, 1));
            let mvsh0l1 = z64hi(mvsv(mbB, 6), mvsv(mbB, 7));
            let mvsh1l1 = z64lo(mvsv(mb, 4), mvsv(mb, 5));
            let mvsael00 = packabd16(mvsv0l0, mvsv1l0, mvsh0l0, mvsh1l0);
            let mvsael01 = packabd16(mvsv0l0, mvsv1l1, mvsh0l0, mvsh1l1);
            let mvsael10 = packabd16(mvsv0l1, mvsv1l0, mvsh0l1, mvsh1l0);
            let mvsael11 = packabd16(mvsv0l1, mvsv1l1, mvsh0l1, mvsh1l1);
            let mvsaep = subsu8(maxu8(mvsael00, mvsael11), c3);
            let mvsaec = subsu8(maxu8(mvsael01, mvsael10), c3);
            let pp = packs16(&v8_as_v16(&mvsaep), &zero_v16());
            let mvsacegp = v32_as_v8(&ziplo32(v8_as_v32(&pp), ZERO_V32));
            let pc = packs16(&v8_as_v16(&mvsaec), &zero_v16());
            let mvsacegc = v32_as_v8(&ziplo32(v8_as_v32(&pc), ZERO_V32));
            let refPic: V64 = [rpl(mb), 0];
            let refPicAB: V64 = [rpl(mbA), rpl(mbB)];
            let refs0 = shuffle(v64_as_v8(&uzp1_v64(refPic, refPicAB)), &SHUFVHAB);
            let refs1 = shuffle(v64_as_v8(&uzp2_v64(refPic, refPicAB)), &SHUFVHAB);
            let neq0 = bxor(refs0, shrd128(refs1, refs0, 8));
            let neq1 = bxor(refs1, shrd128(refs0, refs1, 8));
            let refsaceg = bor(neq0, neq1);
            let refsacegc = ziplo8(refsaceg, refsaceg);
            let refsacegp = ziphi8(refsaceg, refsaceg);
            let neq3 = bor(minu8(refsacegp, refsacegc), minu8(mvsacegp, mvsacegc));
            let neq4 = bor(minu8(refsacegp, mvsacegc), minu8(mvsacegp, refsacegc));
            let bS0aceg = cmpeq0(bor(neq3, neq4));
            (bS0aceg, set8(-1))
        } else {
            // other B macroblocks
            let mvsv0l0 = uzp2(mvsv(mbA, 1), mvsv(mbA, 3));
            let mvsv1l0 = uzp1(mvsv(mb, 0), mvsv(mb, 2));
            let mvsv2l0 = uzp2(mvsv(mb, 0), mvsv(mb, 2));
            let mvsv3l0 = uzp1(mvsv(mb, 1), mvsv(mb, 3));
            let mvsv4l0 = uzp2(mvsv(mb, 1), mvsv(mb, 3));
            let mvsv0l1 = uzp2(mvsv(mbA, 5), mvsv(mbA, 7));
            let mvsv1l1 = uzp1(mvsv(mb, 4), mvsv(mb, 6));
            let mvsv2l1 = uzp2(mvsv(mb, 4), mvsv(mb, 6));
            let mvsv3l1 = uzp1(mvsv(mb, 5), mvsv(mb, 7));
            let mvsv4l1 = uzp2(mvsv(mb, 5), mvsv(mb, 7));
            let mvsacl00 = packabd16(mvsv0l0, mvsv1l0, mvsv2l0, mvsv3l0);
            let mvsbdl00 = packabd16(mvsv1l0, mvsv2l0, mvsv3l0, mvsv4l0);
            let mvsacl01 = packabd16(mvsv0l0, mvsv1l1, mvsv2l0, mvsv3l1);
            let mvsbdl01 = packabd16(mvsv1l0, mvsv2l1, mvsv3l0, mvsv4l1);
            let mvsacl10 = packabd16(mvsv0l1, mvsv1l0, mvsv2l1, mvsv3l0);
            let mvsbdl10 = packabd16(mvsv1l1, mvsv2l0, mvsv3l1, mvsv4l0);
            let mvsacl11 = packabd16(mvsv0l1, mvsv1l1, mvsv2l1, mvsv3l1);
            let mvsbdl11 = packabd16(mvsv1l1, mvsv2l1, mvsv3l1, mvsv4l1);
            let mvsacp = subsu8(maxu8(mvsacl00, mvsacl11), c3);
            let mvsbdp = subsu8(maxu8(mvsbdl00, mvsbdl11), c3);
            let mvsacc = subsu8(maxu8(mvsacl01, mvsacl10), c3);
            let mvsbdc = subsu8(maxu8(mvsbdl01, mvsbdl10), c3);
            let mvsh0l0 = z64hi(mvsv(mbB, 2), mvsv(mbB, 3));
            let mvsh1l0 = z64lo(mvsv(mb, 0), mvsv(mb, 1));
            let mvsh2l0 = z64hi(mvsv(mb, 0), mvsv(mb, 1));
            let mvsh3l0 = z64lo(mvsv(mb, 2), mvsv(mb, 3));
            let mvsh4l0 = z64hi(mvsv(mb, 2), mvsv(mb, 3));
            let mvsh0l1 = z64hi(mvsv(mbB, 6), mvsv(mbB, 7));
            let mvsh1l1 = z64lo(mvsv(mb, 4), mvsv(mb, 5));
            let mvsh2l1 = z64hi(mvsv(mb, 4), mvsv(mb, 5));
            let mvsh3l1 = z64lo(mvsv(mb, 6), mvsv(mb, 7));
            let mvsh4l1 = z64hi(mvsv(mb, 6), mvsv(mb, 7));
            let mvsegl00 = packabd16(mvsh0l0, mvsh1l0, mvsh2l0, mvsh3l0);
            let mvsfhl00 = packabd16(mvsh1l0, mvsh2l0, mvsh3l0, mvsh4l0);
            let mvsegl01 = packabd16(mvsh0l0, mvsh1l1, mvsh2l0, mvsh3l1);
            let mvsfhl01 = packabd16(mvsh1l0, mvsh2l1, mvsh3l0, mvsh4l1);
            let mvsegl10 = packabd16(mvsh0l1, mvsh1l0, mvsh2l1, mvsh3l0);
            let mvsfhl10 = packabd16(mvsh1l1, mvsh2l0, mvsh3l1, mvsh4l0);
            let mvsegl11 = packabd16(mvsh0l1, mvsh1l1, mvsh2l1, mvsh3l1);
            let mvsfhl11 = packabd16(mvsh1l1, mvsh2l1, mvsh3l1, mvsh4l1);
            let mvsegp = subsu8(maxu8(mvsegl00, mvsegl11), c3);
            let mvsfhp = subsu8(maxu8(mvsfhl00, mvsfhl11), c3);
            let mvsegc = subsu8(maxu8(mvsegl01, mvsegl10), c3);
            let mvsfhc = subsu8(maxu8(mvsfhl01, mvsfhl10), c3);
            let mvsacegp = packs16(&v8_as_v16(&mvsacp), &v8_as_v16(&mvsegp));
            let mvsbdfhp = packs16(&v8_as_v16(&mvsbdp), &v8_as_v16(&mvsfhp));
            let mvsacegc = packs16(&v8_as_v16(&mvsacc), &v8_as_v16(&mvsegc));
            let mvsbdfhc = packs16(&v8_as_v16(&mvsbdc), &v8_as_v16(&mvsfhc));
            let refPic: V64 = [rpl(mb), 0];
            let refPicAB: V64 = [rpl(mbA), rpl(mbB)];
            let refs0 = shuffle(v64_as_v8(&uzp1_v64(refPic, refPicAB)), &SHUFVHAB);
            let refs1 = shuffle(v64_as_v8(&uzp2_v64(refPic, refPicAB)), &SHUFVHAB);
            let neq0 = bxor(refs0, shrd128(refs1, refs0, 8));
            let neq1 = bxor(refs1, shrd128(refs0, refs1, 8));
            let neq2 = bxor(refs0, refs1);
            let refsaceg = bor(neq0, neq1);
            let refsacegc = ziplo8(refsaceg, refsaceg);
            let refsacegp = ziphi8(refsaceg, refsaceg);
            let refsbdfhc = ziplo8(neq2, neq2);
            let neq3 = bor(minu8(refsacegp, refsacegc), minu8(mvsacegp, mvsacegc));
            let neq4 = bor(minu8(refsacegp, mvsacegc), minu8(mvsacegp, refsacegc));
            (
                cmpeq0(bor(neq3, neq4)),
                cmpeq0(minu8(mvsbdfhp, bor(refsbdfhc, mvsbdfhc))),
            )
        };

        // 8x8 blocks with CAVLC: broadcast transform tokens beforehand.
        // C: `nC = (i8x16)((i32x4)nC == 0) - -1` — a per-i32-group nonzero
        // mask: each 4-byte group becomes all-1s if that i32 is nonzero,
        // all-0s otherwise (NOT an all-or-nothing test).
        let mut nC = mb.nC_v0();
        if entropy == 0 && mb.ts8x8 != 0 {
            for j in 0..4 {
                let grp = i32::from_le_bytes([
                    nC[4 * j] as u8,
                    nC[4 * j + 1] as u8,
                    nC[4 * j + 2] as u8,
                    nC[4 * j + 3] as u8,
                ]);
                let v = if grp != 0 { 1i8 } else { 0i8 };
                for k in 0..4 {
                    nC[4 * j + k] = v;
                }
            }
        }

        // bS=2 masks from coded-block flags
        let nnzv = shuffle(nC, &SHUFV);
        let nnzl = shuffle(mbA.nC_v0(), &SHUFV);
        let nnzh = shuffle(nC, &SHUFH);
        let nnzt = shuffle(mbB.nC_v0(), &SHUFH);
        let bS2abcd = cmpgt0(bor(nnzv, shrd128(nnzl, nnzv, 12)));
        let bS2efgh = cmpgt0(bor(nnzh, shrd128(nnzt, nnzh, 12)));
        let bS2aacc = v32_as_v8(&trnlo32(v8_as_v32(&bS2abcd), v8_as_v32(&bS2abcd)));
        let bS2eegg = v32_as_v8(&trnlo32(v8_as_v32(&bS2efgh), v8_as_v32(&bS2efgh)));

        // shuffle, blend and store tC0
        let bS0abcd = ziplo32(v8_as_v32(&bS0aceg), v8_as_v32(&bS0bdfh));
        let bS0efgh = ziphi32(v8_as_v32(&bS0aceg), v8_as_v32(&bS0bdfh));
        let bS0aacc = ziplo32(v8_as_v32(&bS0aceg), v8_as_v32(&bS0aceg));
        let bS0eegg = ziphi32(v8_as_v32(&bS0aceg), v8_as_v32(&bS0aceg));
        tC0_v[0] = ifelse_mask(
            bS2abcd,
            shuffle(tC02, &SHUF0),
            bor(v32_as_v8(&bS0abcd), shuffle(tC01, &SHUF0)),
        );
        tC0_v[1] = ifelse_mask(
            bS2efgh,
            shuffle(tC02, &SHUF1),
            bor(v32_as_v8(&bS0efgh), shuffle(tC01, &SHUF1)),
        );
        tC0_v[2] = ifelse_mask(
            bS2aacc,
            shuffle(tC02, &SHUF2),
            bor(v32_as_v8(&bS0aacc), shuffle(tC01, &SHUF2)),
        );
        tC0_v[3] = ifelse_mask(
            bS2eegg,
            shuffle(tC02, &SHUF3),
            bor(v32_as_v8(&bS0eegg), shuffle(tC01, &SHUF3)),
        );
    }

    let mut alpha = [0u8; 16];
    for i in 0..16 {
        alpha[i] = alpha_v[i] as u8;
    }
    let mut beta = [0u8; 16];
    for i in 0..16 {
        beta[i] = beta_v[i] as u8;
    }
    let mut tC0_s = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let v = &tC0_v[i];
            tC0_s[4 * i + j] = i32::from_le_bytes([
                v[4 * j] as u8,
                v[4 * j + 1] as u8,
                v[4 * j + 2] as u8,
                v[4 * j + 3] as u8,
            ]);
        }
    }
    (alpha, beta, tC0_s)
}

// ============================================================================
// Per-edge application: gather window -> filter -> scatter back.
//
// Mirrors deblock_Y_8bit (edge264_deblock.c:530) and deblock_CbCr_8bit (:284).
// The C keeps intermediate edges in registers; here each edge round-trips
// through the plane, which is equivalent because later edges gather the
// already-filtered pixels. Edges run in deblocking order a,b,c,d,e,f,g,h.
// ============================================================================

const LY_STRIDE: usize = 64;
const LY_ROW: usize = 48; // current luma MB top-left row (canonical plane)
const LY_COL: usize = 32; // current luma MB top-left col
const LC_STRIDE: usize = 32; // chroma filter row stride (stride[1]>>1)
const LC_ROW: usize = 48; // Cb row 0 buffer-row
const LC_COL: usize = 8; // chroma run start col

/// Luma sample at MB-relative (r, c) as i8.
#[inline]
fn ly_get(y: &[u8], r: isize, c: isize) -> i8 {
    y[((LY_ROW as isize + r) * LY_STRIDE as isize + (LY_COL as isize + c)) as usize] as i8
}
#[inline]
fn ly_set(y: &mut [u8], r: isize, c: isize, v: i8) {
    y[((LY_ROW as isize + r) * LY_STRIDE as isize + (LY_COL as isize + c)) as usize] = v as u8;
}
/// Gather a vertical luma edge window at p0-col `x0`: returns [p3..q3] columns
/// (each V8 lane = one of the 16 rows).
fn ly_gather_v(y: &[u8], x0: isize) -> [V8; 8] {
    let mut cols = [[0i8; 16]; 8];
    for (k, col) in cols.iter_mut().enumerate() {
        let c = x0 + k as isize - 3; // p3..q3
        for (r, slot) in col.iter_mut().enumerate() {
            *slot = ly_get(y, r as isize, c);
        }
    }
    cols
}
/// Gather a horizontal luma edge window at p0-row `y0`: returns [p3..q3] rows.
fn ly_gather_h(y: &[u8], y0: isize) -> [V8; 8] {
    let mut rows = [[0i8; 16]; 8];
    for (k, row) in rows.iter_mut().enumerate() {
        let r = y0 + k as isize - 3;
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = ly_get(y, r, c as isize);
        }
    }
    rows
}
#[inline]
fn ly_set_col(y: &mut [u8], x: isize, v: V8) {
    for (r, &val) in v.iter().enumerate() {
        ly_set(y, r as isize, x, val);
    }
}
#[inline]
fn ly_set_row(y: &mut [u8], rr: isize, v: V8) {
    for (c, &val) in v.iter().enumerate() {
        ly_set(y, rr, c as isize, val);
    }
}

/// Chroma sample at (buffer-row `brow`, col `col`).
#[inline]
fn lc_get(c: &[u8], brow: isize, col: isize) -> i8 {
    c[((brow * LC_STRIDE as isize) + col) as usize] as i8
}
#[inline]
fn lc_set(c: &mut [u8], brow: isize, col: isize, v: i8) {
    c[((brow * LC_STRIDE as isize) + col) as usize] = v as u8;
}
/// buffer-row for a chroma lane (0..7 = Cb rows, 8..15 = Cr rows).
#[inline]
fn lc_vrow(lane: usize) -> isize {
    LC_ROW as isize + 2 * (lane & 7) as isize + (lane >> 3) as isize
}
/// Gather a vertical chroma edge window at p0-col `cc`: returns [p1,p0,q0,q1].
fn lc_gather_v(c: &[u8], cc: isize) -> [V8; 4] {
    let mut out = [[0i8; 16]; 4];
    for (k, row) in out.iter_mut().enumerate() {
        let col = cc + k as isize - 1; // p1..q1
        for (lane, slot) in row.iter_mut().enumerate() {
            *slot = lc_get(c, lc_vrow(lane), LC_COL as isize + col);
        }
    }
    out
}
/// Gather a horizontal chroma edge window at p0-row `rr`: returns [p1,p0,q0,q1].
fn lc_gather_h(c: &[u8], rr: isize) -> [V8; 4] {
    let mut out = [[0i8; 16]; 4];
    for (k, vec) in out.iter_mut().enumerate() {
        let row = rr + k as isize - 1; // p1..q1 (chroma rows)
        for (lane, slot) in vec.iter_mut().enumerate() {
            let brow = LC_ROW as isize + 2 * row + (lane >> 3) as isize;
            *slot = lc_get(c, brow, LC_COL as isize + (lane & 7) as isize);
        }
    }
    out
}
/// Scatter a vertical chroma edge result: p0 at col `cc`, q0 at col `cc+1`.
fn lc_scatter_v(c: &mut [u8], cc: isize, p0: V8, q0: V8) {
    for lane in 0..16 {
        let brow = lc_vrow(lane);
        lc_set(c, brow, LC_COL as isize + cc, p0[lane]);
        lc_set(c, brow, LC_COL as isize + cc + 1, q0[lane]);
    }
}
/// Scatter a horizontal chroma edge result: p0 at row `rr`, q0 at row `rr+1`.
fn lc_scatter_h(c: &mut [u8], rr: isize, p0: V8, q0: V8) {
    for lane in 0..16 {
        let brow = LC_ROW as isize + 2 * rr + (lane >> 3) as isize;
        let col = LC_COL as isize + (lane & 7) as isize;
        lc_set(c, brow, col, p0[lane]);
        lc_set(c, brow + 2, col, q0[lane]); // q0 is one chroma-row down = +2 buffer rows
    }
}

/// alpha_s[N]: the N-th int32 lane of the 16-byte alpha union (bytes base..base+4).
#[inline]
fn pack_s32(a: &[u8; 16], base: usize) -> i32 {
    ((a[base] as u32)
        | ((a[base + 1] as u32) << 8)
        | ((a[base + 2] as u32) << 16)
        | ((a[base + 3] as u32) << 24)) as i32
}
/// tC0_l[N]: two consecutive int32 lanes packed as a little-endian i64.
/// Each lane is zero-extended through u32 so a negative low lane (e.g. -1)
/// does not sign-extend into the high 32 bits.
#[inline]
fn pack_l64(t: &[i32; 16], base: usize) -> i64 {
    (t[base] as u32 as i64) | ((t[base + 1] as u32 as i64) << 32)
}

/// Luma deblock (deblock_Y_8bit). Edges a..h in order; boundary edges gated by
/// filter_edges, internal 4x4 edges gated by !transform_size_8x8_flag.
fn deblock_y(
    y: &mut [u8],
    mbs: &[Mb; 3],
    alpha: &[u8; 16],
    beta: &[u8; 16],
    tC0_s: &[i32; 16],
    fe: i32,
) {
    let mb = &mbs[2];
    let mbA = &mbs[1];
    let mbB = &mbs[0];
    let ts = mb.ts8x8 != 0;
    let aInt = alpha[0] as i32;
    let bInt = beta[0] as i32;

    // --- vertical edges (p0-col: a=-1, b=3, c=7, d=11) ---
    if fe & 1 != 0 {
        let x0: isize = -1;
        let (a, b) = (alpha[8] as i32, beta[8] as i32);
        if mbA.inter & mb.inter != 0 {
            if tC0_s[0] != -1 {
                let [_p3, p2, p1, p0, q0, q1, q2, _q3] = ly_gather_v(y, x0);
                let (p1o, p0o, q0o, q1o) = luma_soft(p2, p1, p0, q0, q1, q2, a, b, tC0_s[0]);
                ly_set_col(y, x0 - 1, p1o);
                ly_set_col(y, x0, p0o);
                ly_set_col(y, x0 + 1, q0o);
                ly_set_col(y, x0 + 2, q1o);
            }
        } else if a != 0 {
            let [p3, p2, p1, p0, q0, q1, q2, q3] = ly_gather_v(y, x0);
            let (p0o, p1o, p2o, q0o, q1o, q2o) = luma_hard(p3, p2, p1, p0, q0, q1, q2, q3, a, b);
            ly_set_col(y, x0, p0o);
            ly_set_col(y, x0 - 1, p1o);
            ly_set_col(y, x0 - 2, p2o);
            ly_set_col(y, x0 + 1, q0o);
            ly_set_col(y, x0 + 2, q1o);
            ly_set_col(y, x0 + 3, q2o);
        }
    }
    if !ts && tC0_s[1] != -1 {
        let x0: isize = 3;
        let [_p3, p2, p1, p0, q0, q1, q2, _q3] = ly_gather_v(y, x0);
        let (p1o, p0o, q0o, q1o) = luma_soft(p2, p1, p0, q0, q1, q2, aInt, bInt, tC0_s[1]);
        ly_set_col(y, x0 - 1, p1o);
        ly_set_col(y, x0, p0o);
        ly_set_col(y, x0 + 1, q0o);
        ly_set_col(y, x0 + 2, q1o);
    }
    if tC0_s[2] != -1 {
        let x0: isize = 7;
        let [_p3, p2, p1, p0, q0, q1, q2, _q3] = ly_gather_v(y, x0);
        let (p1o, p0o, q0o, q1o) = luma_soft(p2, p1, p0, q0, q1, q2, aInt, bInt, tC0_s[2]);
        ly_set_col(y, x0 - 1, p1o);
        ly_set_col(y, x0, p0o);
        ly_set_col(y, x0 + 1, q0o);
        ly_set_col(y, x0 + 2, q1o);
    }
    if !ts && tC0_s[3] != -1 {
        let x0: isize = 11;
        let [_p3, p2, p1, p0, q0, q1, q2, _q3] = ly_gather_v(y, x0);
        let (p1o, p0o, q0o, q1o) = luma_soft(p2, p1, p0, q0, q1, q2, aInt, bInt, tC0_s[3]);
        ly_set_col(y, x0 - 1, p1o);
        ly_set_col(y, x0, p0o);
        ly_set_col(y, x0 + 1, q0o);
        ly_set_col(y, x0 + 2, q1o);
    }

    // --- horizontal edges (p0-row: e=-1, f=3, g=7, h=11) ---
    if fe & 2 != 0 {
        let y0: isize = -1;
        let (a, b) = (alpha[12] as i32, beta[12] as i32);
        if mbB.inter & mb.inter != 0 {
            if tC0_s[4] != -1 {
                let [_p3, p2, p1, p0, q0, q1, q2, _q3] = ly_gather_h(y, y0);
                let (p1o, p0o, q0o, q1o) = luma_soft(p2, p1, p0, q0, q1, q2, a, b, tC0_s[4]);
                ly_set_row(y, y0 - 1, p1o);
                ly_set_row(y, y0, p0o);
                ly_set_row(y, y0 + 1, q0o);
                ly_set_row(y, y0 + 2, q1o);
            }
        } else if a != 0 {
            let [p3, p2, p1, p0, q0, q1, q2, q3] = ly_gather_h(y, y0);
            let (p0o, p1o, p2o, q0o, q1o, q2o) = luma_hard(p3, p2, p1, p0, q0, q1, q2, q3, a, b);
            ly_set_row(y, y0, p0o);
            ly_set_row(y, y0 - 1, p1o);
            ly_set_row(y, y0 - 2, p2o);
            ly_set_row(y, y0 + 1, q0o);
            ly_set_row(y, y0 + 2, q1o);
            ly_set_row(y, y0 + 3, q2o);
        }
    }
    if !ts && tC0_s[5] != -1 {
        let y0: isize = 3;
        let [_p3, p2, p1, p0, q0, q1, q2, _q3] = ly_gather_h(y, y0);
        let (p1o, p0o, q0o, q1o) = luma_soft(p2, p1, p0, q0, q1, q2, aInt, bInt, tC0_s[5]);
        ly_set_row(y, y0 - 1, p1o);
        ly_set_row(y, y0, p0o);
        ly_set_row(y, y0 + 1, q0o);
        ly_set_row(y, y0 + 2, q1o);
    }
    if tC0_s[6] != -1 {
        let y0: isize = 7;
        let [_p3, p2, p1, p0, q0, q1, q2, _q3] = ly_gather_h(y, y0);
        let (p1o, p0o, q0o, q1o) = luma_soft(p2, p1, p0, q0, q1, q2, aInt, bInt, tC0_s[6]);
        ly_set_row(y, y0 - 1, p1o);
        ly_set_row(y, y0, p0o);
        ly_set_row(y, y0 + 1, q0o);
        ly_set_row(y, y0 + 2, q1o);
    }
    if !ts && tC0_s[7] != -1 {
        let y0: isize = 11;
        let [_p3, p2, p1, p0, q0, q1, q2, _q3] = ly_gather_h(y, y0);
        let (p1o, p0o, q0o, q1o) = luma_soft(p2, p1, p0, q0, q1, q2, aInt, bInt, tC0_s[7]);
        ly_set_row(y, y0 - 1, p1o);
        ly_set_row(y, y0, p0o);
        ly_set_row(y, y0 + 1, q0o);
        ly_set_row(y, y0 + 2, q1o);
    }
}

/// Chroma deblock (deblock_CbCr_8bit). Edges a,c vertical; e,g horizontal.
fn deblock_cbcr(
    c: &mut [u8],
    mbs: &[Mb; 3],
    alpha: &[u8; 16],
    beta: &[u8; 16],
    tC0_s: &[i32; 16],
    fe: i32,
) {
    let mb = &mbs[2];
    let mbA = &mbs[1];
    let mbB = &mbs[0];
    let a_s0 = pack_s32(alpha, 0);
    let b_s0 = pack_s32(beta, 0);
    let a_s2 = pack_s32(alpha, 8);
    let b_s2 = pack_s32(beta, 8);
    let a_s3 = pack_s32(alpha, 12);
    let b_s3 = pack_s32(beta, 12);
    let t4 = pack_l64(tC0_s, 8);
    let t5 = pack_l64(tC0_s, 10);
    let t6 = pack_l64(tC0_s, 12);
    let t7 = pack_l64(tC0_s, 14);

    // edge a (left boundary vertical, p0-col -1), gated fe&1
    if fe & 1 != 0 {
        let cc: isize = -1;
        if mbA.inter & mb.inter != 0 {
            if t4 != -1 {
                let [p1, p0, q0, q1] = lc_gather_v(c, cc);
                let (p0o, q0o) = chroma_soft(p1, p0, q0, q1, a_s2, b_s2, t4);
                lc_scatter_v(c, cc, p0o, q0o);
            }
        } else {
            let [p1, p0, q0, q1] = lc_gather_v(c, cc);
            let (p0o, q0o) = chroma_hard(p1, p0, q0, q1, a_s2, b_s2);
            lc_scatter_v(c, cc, p0o, q0o);
        }
    }
    // edge c (internal vertical, p0-col 3)
    if t5 != -1 {
        let cc: isize = 3;
        let [p1, p0, q0, q1] = lc_gather_v(c, cc);
        let (p0o, q0o) = chroma_soft(p1, p0, q0, q1, a_s0, b_s0, t5);
        lc_scatter_v(c, cc, p0o, q0o);
    }
    // edge e (top boundary horizontal, p0-row -1), gated fe&2
    if fe & 2 != 0 {
        let rr: isize = -1;
        if mbB.inter & mb.inter != 0 {
            if t6 != -1 {
                let [p1, p0, q0, q1] = lc_gather_h(c, rr);
                let (p0o, q0o) = chroma_soft(p1, p0, q0, q1, a_s3, b_s3, t6);
                lc_scatter_h(c, rr, p0o, q0o);
            }
        } else {
            let [p1, p0, q0, q1] = lc_gather_h(c, rr);
            let (p0o, q0o) = chroma_hard(p1, p0, q0, q1, a_s3, b_s3);
            lc_scatter_h(c, rr, p0o, q0o);
        }
    }
    // edge g (internal horizontal, p0-row 3)
    if t7 != -1 {
        let rr: isize = 3;
        let [p1, p0, q0, q1] = lc_gather_h(c, rr);
        let (p0o, q0o) = chroma_soft(p1, p0, q0, q1, a_s0, b_s0, t7);
        lc_scatter_h(c, rr, p0o, q0o);
    }
}

/// Rust port of one macroblock's deblocking filter. Mirrors the C oracle
/// `sw264_test_deblock_run`: takes serialized MB state {top,left,current} plus
/// the full canonical luma/chroma planes and returns the filtered planes.
#[allow(clippy::too_many_arguments)]
pub fn run_rust_deblock(
    mb_state: &[u8],
    filter_edges: i32,
    entropy_coding_mode_flag: i32,
    filter_offset_a: i32,
    filter_offset_b: i32,
    y_in: &[u8],
    c_in: &[u8],
) -> (Vec<u8>, Vec<u8>) {
    let mut y = y_in.to_vec();
    let mut c = c_in.to_vec();
    run_rust_deblock_inplace(
        mb_state,
        filter_edges,
        entropy_coding_mode_flag,
        filter_offset_a,
        filter_offset_b,
        &mut y,
        &mut c,
    );
    (y, c)
}

/// In-place core: filters the canonical window planes `y`/`c` in place.
pub fn run_rust_deblock_inplace(
    mb_state: &[u8],
    filter_edges: i32,
    entropy_coding_mode_flag: i32,
    filter_offset_a: i32,
    filter_offset_b: i32,
    y: &mut [u8],
    c: &mut [u8],
) {
    if filter_edges == 0 {
        return; // deblock_mb returns early
    }
    let mut mbs = [
        parse_mb(&mb_state[0..202]),
        parse_mb(&mb_state[202..404]),
        parse_mb(&mb_state[404..606]),
    ];
    mbs[2].fe = filter_edges; // C sets mbs[2].filter_edges explicitly
    let (alpha, beta, tC0_s) = deblock_params(
        &mbs,
        entropy_coding_mode_flag,
        filter_offset_a,
        filter_offset_b,
    );
    deblock_y(y, &mbs, &alpha, &beta, &tC0_s, filter_edges);
    deblock_cbcr(c, &mbs, &alpha, &beta, &tC0_s, filter_edges);
}

/// Location of an MB's pixel window in a real frame plane. `x`/`y`: the MB's
/// top-left luma sample (luma) or Cb row-0 sample (chroma), in units of
/// `stride` rows. For chroma, `stride` must be the C filter row stride
/// (`stride[1] >> 1`): Cb row r and Cr row r then sit at rows 2r and 2r+1, so
/// MB row mby starts at y = 16*mby. `w`/`h`: plane extent in samples / rows.
#[derive(Clone, Copy)]
pub struct DeblockRect {
    pub stride: usize,
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// In-slice deblock of one macroblock (row-lag / slice-tail call sites).
/// Copies the pixel window around the MB from the real frame planes into the
/// reusable canonical window buffers (`dblk_y`/`dblk_c`, MB origin at
/// (LY_ROW, LY_COL) / (LC_ROW, LC_COL)), runs the fuzz-verified
/// `run_rust_deblock_inplace`, and copies the filtered window back.
///
/// Interior MBs (full 20x20 luma / 20x10 chroma footprint inside the frame)
/// use bounds-check-free row copies. Edge MBs zero the in-window region first
/// (emulating C's zero padding — the kernel never reads outside the filled
/// window, proven by golden + poison runs) and use clamped copies.
/// `lc == None`: no chroma plane (4:0:0); the chroma pass is skipped (C reads
/// a non-existent plane there).
#[allow(clippy::too_many_arguments)]
pub fn deblock_mb_inplace(
    mb_state: &[u8; 606],
    filter_edges: i32,
    entropy_coding_mode_flag: i32,
    filter_offset_a: i32,
    filter_offset_b: i32,
    dblk_y: &mut [u8],
    dblk_c: &mut [u8],
    y_plane: &mut [u8],
    ly: &DeblockRect,
    c_plane: &mut [u8],
    lc: Option<&DeblockRect>,
) {
    if filter_edges == 0 {
        return; // deblock_mb returns early
    }
    // The filter's read/write footprint (MB-relative): luma rows/cols -4..15,
    // chroma buffer rows -4..15 / cols -2..7; nothing else is ever touched.
    let interior = ly.x >= 4 && ly.y >= 4 && ly.x + 16 <= ly.w && ly.y + 16 <= ly.h;
    if !interior {
        // Zero the in-window region: out-of-frame corners must be zero (C's
        // padding emulation) and stale bytes from the previous MB must not
        // leak into the filter.
        for r in -4isize..16 {
            let row = ((LY_ROW as isize + r) * LY_STRIDE as isize) as usize;
            dblk_y[row + (LY_COL - 4)..row + LY_COL + 16].fill(0);
            if lc.is_some() {
                let crow = ((LC_ROW as isize + r) * LC_STRIDE as isize) as usize;
                dblk_c[crow + (LC_COL - 2)..crow + LC_COL + 8].fill(0);
            }
        }
    }
    // Luma window: 20x20 at MB-relative (-4..15) — the filter's read
    // footprint (vertical edges read cols -4..15 x rows 0..15, horizontal
    // edges read rows -4..15 x cols 0..15). Writes are a subset of this.
    if interior {
        for r in 0..20usize {
            let srow = (ly.y as isize + r as isize - 4) * ly.stride as isize + (ly.x as isize - 4);
            let drow =
                (LY_ROW as isize + r as isize - 4) * LY_STRIDE as isize + (LY_COL as isize - 4);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    y_plane.as_ptr().add(srow as usize),
                    dblk_y.as_mut_ptr().add(drow as usize),
                    20,
                );
            }
        }
    } else {
        for r in -4isize..16 {
            for col in -4isize..16 {
                let sr = ly.y as isize + r;
                let sc = ly.x as isize + col;
                if (0..ly.h as isize).contains(&sr) && (0..ly.w as isize).contains(&sc) {
                    dblk_y[((LY_ROW as isize + r) * LY_STRIDE as isize + (LY_COL as isize + col))
                        as usize] = y_plane[(sr * ly.stride as isize + sc) as usize];
                }
            }
        }
    }
    if let Some(lc) = lc {
        // Chroma window: buffer rows -4..15 (vertical edges read all 16
        // buffer rows), cols -2..7 (horizontal edges read all 8 chroma cols).
        if interior {
            for r in 0..20usize {
                let srow =
                    (lc.y as isize + r as isize - 4) * lc.stride as isize + (lc.x as isize - 2);
                let drow =
                    (LC_ROW as isize + r as isize - 4) * LC_STRIDE as isize + (LC_COL as isize - 2);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        c_plane.as_ptr().add(srow as usize),
                        dblk_c.as_mut_ptr().add(drow as usize),
                        10,
                    );
                }
            }
        } else {
            for r in -4isize..16 {
                for col in -2isize..8 {
                    let sr = lc.y as isize + r;
                    let sc = lc.x as isize + col;
                    if (0..lc.h as isize).contains(&sr) && (0..lc.w as isize).contains(&sc) {
                        dblk_c[((LC_ROW as isize + r) * LC_STRIDE as isize
                            + (LC_COL as isize + col)) as usize] =
                            c_plane[(sr * lc.stride as isize + sc) as usize];
                    }
                }
            }
        }
    }
    run_rust_deblock_inplace(
        mb_state,
        filter_edges,
        entropy_coding_mode_flag,
        filter_offset_a,
        filter_offset_b,
        dblk_y,
        dblk_c,
    );
    // Copy the same window back (the filter only writes inside it).
    if interior {
        for r in 0..20usize {
            let srow =
                (LY_ROW as isize + r as isize - 4) * LY_STRIDE as isize + (LY_COL as isize - 4);
            let drow = (ly.y as isize + r as isize - 4) * ly.stride as isize + (ly.x as isize - 4);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    dblk_y.as_ptr().add(srow as usize),
                    y_plane.as_mut_ptr().add(drow as usize),
                    20,
                );
            }
        }
    } else {
        for r in -4isize..16 {
            for col in -4isize..16 {
                let sr = ly.y as isize + r;
                let sc = ly.x as isize + col;
                if (0..ly.h as isize).contains(&sr) && (0..ly.w as isize).contains(&sc) {
                    y_plane[(sr * ly.stride as isize + sc) as usize] =
                        dblk_y[((LY_ROW as isize + r) * LY_STRIDE as isize
                            + (LY_COL as isize + col)) as usize];
                }
            }
        }
    }
    if let Some(lc) = lc {
        if interior {
            for r in 0..20usize {
                let srow =
                    (LC_ROW as isize + r as isize - 4) * LC_STRIDE as isize + (LC_COL as isize - 2);
                let drow =
                    (lc.y as isize + r as isize - 4) * lc.stride as isize + (lc.x as isize - 2);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        dblk_c.as_ptr().add(srow as usize),
                        c_plane.as_mut_ptr().add(drow as usize),
                        10,
                    );
                }
            }
        } else {
            for r in -4isize..16 {
                for col in -2isize..8 {
                    let sr = lc.y as isize + r;
                    let sc = lc.x as isize + col;
                    if (0..lc.h as isize).contains(&sr) && (0..lc.w as isize).contains(&sc) {
                        c_plane[(sr * lc.stride as isize + sc) as usize] =
                            dblk_c[((LC_ROW as isize + r) * LC_STRIDE as isize
                                + (LC_COL as isize + col))
                                as usize];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) use self::prim_tests::golden_entries;

#[cfg(test)]
mod prim_tests {
    use super::*;
    use crate::rust::goldens;

    #[test]
    fn reinterpret_i16_i32_negatives() {
        // Regression: i16->u32 sign-extension used to corrupt the OR in
        // v16_as_v32 for negative halves (broke uzp1/uzp2 on negative MVs).
        // Expected outputs cross-checked against C header macros (probe_mvsac.c).
        let a: V16 = [1, 2, 3, 4, 5, 6, 7, 8];
        let b: V16 = [9, 10, 11, 12, 13, 14, 15, 16];
        assert_eq!(uzp1(a, b), [1, 2, 5, 6, 9, 10, 13, 14]);
        assert_eq!(uzp2(a, b), [3, 4, 7, 8, 11, 12, 15, 16]);
        // negative halves must survive the i16->i32 reinterpret
        let a: V16 = [-2, -1, -2, -1, -2, -1, -2, -1];
        let b: V16 = [-2, -5, -2, -5, -2, -5, -2, -5];
        assert_eq!(v16_as_v32(&b)[0] as u32, 0xFFFB_FFFE);
        assert_eq!(uzp2(a, b), [-2, -1, -2, -1, -2, -5, -2, -5]);
    }

    #[test]
    fn reinterpret_roundtrip() {
        let v: V8 = [1, -2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let w = v8_as_v16(&v);
        // little-endian i16 of (1,-2) = 1 + (-2<<8) = 1 - 512 = -511
        assert_eq!(w[0], 1 + (-2i16 << 8));
        assert_eq!(w[1], 3 + (-4i16 << 8));
        assert_eq!(w[7], 15 + (16i16 << 8));
    }

    #[test]
    fn sat_ops() {
        // V8 is [i8;16]; high bytes (>=128) are negative as i8 but the ops
        // reinterpret them unsigned. Expected values asserted as u8.
        let a: V8 = [
            0,
            100,
            200i32 as i8,
            255i32 as i8,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
        ];
        let b: V8 = [
            50,
            100,
            255i32 as i8,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
        ];
        let s = subsu8(a, b);
        assert_eq!(s[0], 0); // 0-50 -> 0 (saturate)
        assert_eq!(s[1], 0); // 100-100
        assert_eq!(s[2], 0); // 200-255 -> 0 (saturate)
        let ad = addsu8(a, b);
        assert_eq!(ad[3] as u8, 255); // 255+0
        let big: V8 = [250i32 as i8; 16];
        let one: V8 = [10; 16];
        assert_eq!(addsu8(big, one)[0] as u8, 255); // saturate
        let av = avgu8(a, b);
        assert_eq!(av[0], (50 + 1) >> 1);
    }

    #[test]
    fn shifts_and_blend() {
        let a: V8 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert_eq!(
            shr128(a, 4),
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0]
        );
        let l: V8 = [0; 16];
        let h: V8 = [100; 16];
        // shrd128(l,h,8) = bytes 8..24 of [l(0-15),h(16-31)] = [0x8..0xF of l? no]
        // cat = [0*16, 100*16]; bytes 8..23 = [0,0,0,0,0,0,0,0, 100,100,100,100,100,100,100,100]
        let r = shrd128(l, h, 8);
        assert_eq!(r[..8], [0i8; 8]);
        assert_eq!(r[8..], [100i8; 8]);
        let m: V8 = [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0];
        let t: V8 = [9; 16];
        let f: V8 = [7; 16];
        let r2 = ifelse_mask(m, t, f);
        assert_eq!(r2[0], 9);
        assert_eq!(r2[1], 7);
    }

    #[test]
    fn expand_and_packabd() {
        // expand4 of a small positive int: byte0 repeated, rest zero.
        let e = expand4(5);
        assert_eq!(e[..4], [5i8; 4]);
        assert_eq!(e[4..], [0i8; 12]);
        let en = expand4(-1);
        assert_eq!(en[..4], [-1i8; 4]);
        assert_eq!(en[4..], [-1i8; 12]);
        // packabd16: |a-b|,|c-d|
        let a: V16 = [100; 8];
        let b: V16 = [30; 8];
        let c: V16 = [-50; 8];
        let d: V16 = [20; 8];
        let r = packabd16(a, b, c, d);
        assert_eq!(r[0], 70); // |100-30|
        assert_eq!(r[4], 70); // |-50-20|
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn byte(&mut self) -> u8 {
            self.next() as u8
        }
    }

    /// One in-slice deblock must equal the canonical run on the window and
    /// leave every pixel outside the window untouched.
    #[test]
    fn inplace_matches_canonical() {
        let mut rng = Lcg(0x1b5e_269c_4d00);
        for iter in 0..200usize {
            let prof = (rng.next() % 5) as u32;
            let mut mb_state = [0u8; 606];
            for slot in 0..3usize {
                let b = slot * 202;
                mb_state[b] = (rng.next() % 52) as u8;
                mb_state[b + 1] = (rng.next() % 52) as u8;
                mb_state[b + 2] = (rng.next() % 52) as u8;
                mb_state[b + 3] = match prof {
                    0..=2 => 1,
                    3 => 0,
                    _ => (rng.next() & 1) as u8,
                };
                mb_state[b + 4] = (rng.next() % 4) as u8;
                let eq = if prof == 1 && slot == 2 {
                    0x1b5f_bbff
                } else {
                    rng.next() as u32
                };
                mb_state[b + 5..b + 9].copy_from_slice(&eq.to_le_bytes());
                mb_state[b + 9] = (rng.next() % 2) as u8;
                for k in 0..48 {
                    mb_state[b + 10 + k] = if rng.next().is_multiple_of(5) { 0 } else { 1 };
                }
                for k in 0..8 {
                    mb_state[b + 58 + k] = if prof == 0 {
                        (-1i8) as u8
                    } else {
                        [-1i8, -1, 0, 1][(rng.next() % 4) as usize] as u8
                    };
                }
                for k in 0..8 {
                    mb_state[b + 66 + k] = (rng.next() % 3) as u8;
                }
                for k in 0..64 {
                    let mv = ((rng.next() % 21) as i16) - 10;
                    mb_state[b + 74 + 2 * k..b + 74 + 2 * k + 2].copy_from_slice(&mv.to_le_bytes());
                }
            }
            let fe = 1 + (rng.next() % 3) as i32;
            let entropy = (rng.next() & 1) as i32;
            let off_a = ((rng.next() % 3) as i32) - 2;
            let off_b = ((rng.next() % 3) as i32) - 2;

            // Real-frame geometry: a few luma MBs wide/tall with a stride pad.
            let w_mbs = 1 + (rng.next() % 4) as usize; // 1..4
            let h_mbs = 1 + (rng.next() % 3) as usize; // 1..3
            let pad = (rng.next() % 8) as usize;
            let w = w_mbs * 16;
            let h = h_mbs * 16;
            let ly_stride = w + pad;
            let lc_stride = w / 2 + pad; // buffer-row stride
            let y_in: Vec<u8> = (0..ly_stride * h).map(|_| rng.byte()).collect();
            let c_in: Vec<u8> = (0..lc_stride * h).map(|_| rng.byte()).collect();

            // MB under test: any position; corner positions exercise the
            // clamped (zero-filled) window parts.
            let mbx = (rng.next() % w_mbs as u64) as usize;
            let mby = (rng.next() % h_mbs as u64) as usize;
            let ly = DeblockRect {
                stride: ly_stride,
                x: mbx * 16,
                y: mby * 16,
                w,
                h,
            };
            let lc = DeblockRect {
                stride: lc_stride,
                x: mbx * 8,
                y: mby * 16,
                w: w / 2,
                h,
            };

            let mut y_a = y_in.clone();
            let mut c_a = c_in.clone();
            let mut dblk_y = [0u8; DEBLOCK_LY_SIZE];
            let mut dblk_c = [0u8; DEBLOCK_LC_SIZE];
            deblock_mb_inplace(
                &mb_state,
                fe,
                entropy,
                off_a,
                off_b,
                &mut dblk_y,
                &mut dblk_c,
                &mut y_a,
                &ly,
                &mut c_a,
                Some(&lc),
            );

            // Reference: canonical run over a zero-filled window copy.
            let mut y_win = [0u8; DEBLOCK_LY_SIZE];
            let mut c_win = [0u8; DEBLOCK_LC_SIZE];
            for r in -4isize..16 {
                for col in -4isize..16 {
                    let sr = ly.y as isize + r;
                    let sc = ly.x as isize + col;
                    if (0..h as isize).contains(&sr) && (0..w as isize).contains(&sc) {
                        y_win[((LY_ROW as isize + r) * LY_STRIDE as isize + (LY_COL as isize + col))
                            as usize] = y_in[(sr * ly_stride as isize + sc) as usize];
                    }
                }
            }
            for r in -4isize..16 {
                for col in -2isize..8 {
                    let sr = lc.y as isize + r;
                    let sc = lc.x as isize + col;
                    if (0..h as isize).contains(&sr) && (0..(w / 2) as isize).contains(&sc) {
                        c_win[((LC_ROW as isize + r) * LC_STRIDE as isize + (LC_COL as isize + col))
                            as usize] = c_in[(sr * lc_stride as isize + sc) as usize];
                    }
                }
            }
            let (y_ref, c_ref) =
                run_rust_deblock(&mb_state, fe, entropy, off_a, off_b, &y_win, &c_win);

            // Window regions must match the canonical result byte-for-byte.
            for r in -4isize..16 {
                for col in -4isize..16 {
                    let sr = ly.y as isize + r;
                    let sc = ly.x as isize + col;
                    if (0..h as isize).contains(&sr) && (0..w as isize).contains(&sc) {
                        assert_eq!(
                            y_a[(sr * ly_stride as isize + sc) as usize],
                            y_ref[((LY_ROW as isize + r) * LY_STRIDE as isize
                                + (LY_COL as isize + col))
                                as usize],
                            "iter {iter} mbx={mbx} mby={mby} fe={fe} luma ({sr},{sc})"
                        );
                    }
                }
            }
            for r in -4isize..16 {
                for col in -2isize..8 {
                    let sr = lc.y as isize + r;
                    let sc = lc.x as isize + col;
                    if (0..h as isize).contains(&sr) && (0..(w / 2) as isize).contains(&sc) {
                        assert_eq!(
                            c_a[(sr * lc_stride as isize + sc) as usize],
                            c_ref[((LC_ROW as isize + r) * LC_STRIDE as isize
                                + (LC_COL as isize + col))
                                as usize],
                            "iter {iter} mbx={mbx} mby={mby} fe={fe} chroma ({sr},{sc})"
                        );
                    }
                }
            }
            // Everything outside the window is untouched.
            for (i, &p) in y_a.iter().enumerate() {
                let (r, c) = (i / ly_stride, i % ly_stride);
                let in_win = (ly.y as isize - 4..ly.y as isize + 16).contains(&(r as isize))
                    && (ly.x as isize - 4..ly.x as isize + 16).contains(&(c as isize));
                if !in_win {
                    assert_eq!(p, y_in[i], "iter {iter} luma outside window ({r},{c})");
                }
            }
            for (i, &p) in c_a.iter().enumerate() {
                let (r, c) = (i / lc_stride, i % lc_stride);
                let in_win = (lc.y as isize - 4..lc.y as isize + 16).contains(&(r as isize))
                    && (lc.x as isize - 2..lc.x as isize + 8).contains(&(c as isize));
                if !in_win {
                    assert_eq!(p, c_in[i], "iter {iter} chroma outside window ({r},{c})");
                }
            }
        }
    }

    /// Golden pinning for the inter deblock path: the all-intra stream goldens
    /// only exercise the intra (hard) branch, so this fuzzes the inter bS
    /// (MV/ref-equality) path in `deblock_params` directly with random inter MB
    /// states (P, 16x16-B, other-B) and pins the full filtered planes.
    #[test]
    fn inter_deblock_golden() {
        inter_deblock_fuzz_body();
    }

    fn inter_deblock_fuzz_body() {
        use super::{DEBLOCK_LC_SIZE, DEBLOCK_LY_SIZE};
        let mut rng = Lcg(0x1b5e_269c_4d00);
        for iter in 0..500usize {
            let kind = (rng.next() % 3) as u32; // 0=P, 1=B-16x16, 2=B-other
            let mut mb_state = [0u8; 606];
            for slot in 0..3usize {
                let b = slot * 202;
                mb_state[b] = (rng.next() % 52) as u8; // QP_y
                mb_state[b + 1] = (rng.next() % 52) as u8; // QP_cb
                mb_state[b + 2] = (rng.next() % 52) as u8; // QP_cr
                mb_state[b + 3] = 1; // inter
                mb_state[b + 4] = 7; // filter_edges (all bits)
                let eq = if kind == 1 {
                    0x1b5f_bbff
                } else {
                    rng.next() as u32
                };
                mb_state[b + 5..b + 9].copy_from_slice(&eq.to_le_bytes());
                mb_state[b + 9] = (rng.next() % 2) as u8; // ts8x8
                for k in 0..48 {
                    mb_state[b + 10 + k] = if rng.next().is_multiple_of(4) { 0 } else { 1 };
                }
                for k in 0..8 {
                    mb_state[b + 58 + k] = if k < 4 {
                        (rng.next() % 4) as u8 // L0 refIdx valid
                    } else if kind == 0 {
                        (-1i8) as u8 // P: L1 unused
                    } else {
                        [-1i8, -1, 0, 1][(rng.next() % 4) as usize] as u8
                    };
                }
                for k in 0..8 {
                    mb_state[b + 66 + k] = if k < 4 {
                        (rng.next() % 3) as u8
                    } else {
                        (-1i8) as u8
                    };
                }
                for k in 0..64 {
                    // wide MV range to cross the bS |dMV|<=3 threshold both ways
                    let mv = ((rng.next() % 401) as i16) - 200;
                    mb_state[b + 74 + 2 * k..b + 74 + 2 * k + 2].copy_from_slice(&mv.to_le_bytes());
                }
            }
            let fe = 1 + (rng.next() % 3) as i32;
            let entropy = (rng.next() & 1) as i32;
            let off_a = ((rng.next() % 5) as i32) - 2;
            let off_b = ((rng.next() % 5) as i32) - 2;
            let y_in: Vec<u8> = (0..DEBLOCK_LY_SIZE).map(|_| rng.byte()).collect();
            let c_in: Vec<u8> = (0..DEBLOCK_LC_SIZE).map(|_| rng.byte()).collect();

            let (ry, rc) = run_rust_deblock(&mb_state, fe, entropy, off_a, off_b, &y_in, &c_in);
            let mut data = Vec::with_capacity(ry.len() + rc.len());
            data.extend_from_slice(&ry);
            data.extend_from_slice(&rc);
            let key = format!("deblock-inter #{iter} kind={kind}");
            goldens::assert_golden(&key, &data);
            goldens::record(&key, &data);
        }
    }

    pub(crate) fn golden_entries() -> Vec<(String, String)> {
        crate::rust::goldens::collect(inter_deblock_fuzz_body)
    }

    /// fe=0 must be a guaranteed no-op (deblock_mb returns early).
    #[test]
    fn inplace_fe0_noop() {
        let mut y = [7u8; 64 * 48];
        let mut c = [9u8; 32 * 48];
        let ly = DeblockRect {
            stride: 64,
            x: 32,
            y: 32,
            w: 64,
            h: 48,
        };
        let lc = DeblockRect {
            stride: 32,
            x: 8,
            y: 32,
            w: 32,
            h: 48,
        };
        let mut dblk_y = [0u8; DEBLOCK_LY_SIZE];
        let mut dblk_c = [0u8; DEBLOCK_LC_SIZE];
        deblock_mb_inplace(
            &[0; 606],
            0,
            0,
            0,
            0,
            &mut dblk_y,
            &mut dblk_c,
            &mut y,
            &ly,
            &mut c,
            Some(&lc),
        );
        assert_eq!(y, [7u8; 64 * 48]);
        assert_eq!(c, [9u8; 32 * 48]);
    }
}

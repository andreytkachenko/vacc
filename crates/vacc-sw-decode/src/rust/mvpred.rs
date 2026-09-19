//! Tier C: motion-vector prediction — faithful port of `c/src/edge264_mvpred.c`.
//!
//! Ports `decode_P_skip`, the five explicit-MVP parsers (`decode_inter_16x16`,
//! `decode_inter_8x16_left/right`, `decode_inter_16x8_top/bottom`) and both
//! direct-prediction modes. The trailing `decode_inter` calls are pixel-side
//! effects already verified in Tier B (and read-only w.r.t. the macroblock
//! state); here we compute exactly the macroblock state they consume
//! (refIdx / mvs / refPic / absMvd / inter_eqs) and pin the results to
//! golden hashes in `tests.rs`.
//!
//! Machine-verified SIMD semantics on this build (SSE, -march=native):
//! - vector comparisons yield all-bits-set / all-bits-clear per element;
//! - `_mm_shuffle_epi8` returns 0 for mask bytes with the MSB set (not
//!   `a[m & 15]`), so `shufflen` = `m < 0 ? 0xFF : a[m & 15]` and
//!   `shuffle2z` = `m < 0 ? 0 : m < 16 ? a[m] : b[m & 15]`;
//! - `temporal_scale(mv, dfs)` = sat16((mv*dfs + 128 + (dfs<0)) >> 8) with a
//!   floor (arithmetic) shift;
//! - `mvs_near_zero` per i32 lane: `(abs16(x) >> 1) == 0 && (abs16(y) >> 1) == 0`
//!   where abs16(-32768) wraps to -32768;
//! - `colZeroFlags = movemask(packs16(packs32(cm0,cm1), packs32(cm2,cm3)))`:
//!   bit 4N+j set iff colZeroMaskN lane j is FF (packs16(a,b) =
//!   [sat8(a_l0..7), sat8(b_l0..7)], hardware-verified).
//! - `packs16(a,b)` fills ALL 16 bytes: [sat8(a_l0..7), sat8(b_l0..7)]
//!   (NOT low4(a)+high4(b) with zeroed upper half).

#![allow(non_snake_case)]

/// Neighbour macroblock state as read by mvpred: refIdx bytes ([LX][i8x8])
/// and mvs as packed i32 pairs (pair = x | y << 16).
#[derive(Clone, Copy)]
pub struct Nb {
    pub ref_idx: [i8; 8],
    pub mvs_s: [i32; 32],
}

impl Nb {
    fn ref_idx_l(&self) -> i64 {
        let b = self.ref_idx.map(|x| x as u8);
        i64::from_le_bytes(b)
    }
}

/// Collocated macroblock state as read by the direct modes.
#[derive(Clone, Copy)]
pub struct Col {
    pub ref_idx_s: [i32; 2],
    pub ref_pic_s: [i32; 2],
    pub mvs: [i16; 64],
    pub inter_eqs: u32,
}

/// Macroblock state written by mvpred (the part the oracle reports).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Mb {
    pub ref_idx: [i8; 8],
    pub ref_pic: [i8; 8],
    pub mvs: [i16; 64],
    pub abs_mvd: [u8; 64],
    pub inter_eqs: u32, // f.inter_eqs_s
}

impl Default for Mb {
    fn default() -> Self {
        Mb {
            ref_idx: [0; 8],
            ref_pic: [0; 8],
            mvs: [0; 64],
            abs_mvd: [0; 64],
            inter_eqs: 0,
        }
    }
}

/// Serialized mvpred test input: 6 macroblocks of 148 bytes in order
/// {current, A, B, C, D, Col}; per MB: refIdx[8]@0, mvs[64 i16]@8,
/// refPic_s[2 i32]@136, inter_eqs_s@144. Then ctx fields: unavail4x4[48]@888,
/// RefPicList[2][32]@936, MapPicToList0[32]@1000, DistScaleFactor[32 i16]@1032,
/// col_short_term@1096, direct_8x8_inference_flag@1097,
/// direct_spatial_mv_pred_flag@1098, mvd{dx,dy}@1099 (i16), direct_flags u32@1103.
pub const MVPRED_IN_LEN: usize = 1107;
/// Per-macroblock slice of the input above.
pub const MVPRED_MB_LEN: usize = 148;
/// Serialized output state: refIdx[8]@0, mvs[64 i16]@8, refPic_s[2 i32]@136,
/// inter_eqs_s@144, absMvd[64]@148.
pub const MVPRED_OUT_LEN: usize = 212;

/// Full input state for one mvpred op (parsed from the test byte layout).
pub struct State {
    pub mb: Mb,
    pub mb_a: Nb,
    pub mb_b: Nb,
    pub mb_c: Nb,
    pub mb_d: Nb,
    pub mb_col: Col,
    pub unavail4x4: [i8; 48],
    pub ref_pic_list0: [i8; 32],
    pub ref_pic_list1: [i8; 32],
    pub map_pic_to_list0: [i8; 32],
    pub dist_scale_factor: [i16; 32],
    pub col_short_term: bool,
    pub direct_8x8_inference_flag: bool,
}

impl State {
    /// Parse the 1107-byte test input layout (see `MVPRED_IN_LEN` above).
    pub fn parse(in_bytes: &[u8]) -> Self {
        assert_eq!(in_bytes.len(), 1107);
        let parse_mb = |off: usize| -> Mb {
            let s = &in_bytes[off..off + 148];
            let mut mvs = [0i16; 64];
            for i in 0..64 {
                mvs[i] = i16::from_le_bytes([s[8 + 2 * i], s[9 + 2 * i]]);
            }
            Mb {
                ref_idx: std::array::from_fn(|i| s[i] as i8),
                ref_pic: std::array::from_fn(|i| s[136 + i] as i8),
                mvs,
                abs_mvd: [0; 64],
                inter_eqs: u32::from_le_bytes(s[144..148].try_into().unwrap()),
            }
        };
        let parse_nb = |off: usize| -> Nb {
            let s = &in_bytes[off..off + 148];
            let mut mvs_s = [0i32; 32];
            for i in 0..32 {
                mvs_s[i] = i32::from_le_bytes(s[8 + 4 * i..12 + 4 * i].try_into().unwrap());
            }
            let mut ref_idx = [0i8; 8];
            for i in 0..8 {
                ref_idx[i] = s[i] as i8;
            }
            Nb { ref_idx, mvs_s }
        };
        let s = &in_bytes[5 * 148..5 * 148 + 148];
        let mut col = Col {
            ref_idx_s: [0; 2],
            ref_pic_s: [0; 2],
            mvs: [0; 64],
            inter_eqs: 0,
        };
        col.ref_idx_s[0] = i32::from_le_bytes(s[0..4].try_into().unwrap());
        col.ref_idx_s[1] = i32::from_le_bytes(s[4..8].try_into().unwrap());
        col.ref_pic_s[0] = i32::from_le_bytes(s[136..140].try_into().unwrap());
        col.ref_pic_s[1] = i32::from_le_bytes(s[140..144].try_into().unwrap());
        for i in 0..64 {
            col.mvs[i] = i16::from_le_bytes([s[8 + 2 * i], s[9 + 2 * i]]);
        }
        col.inter_eqs = u32::from_le_bytes(s[144..148].try_into().unwrap());
        let o = 6 * 148;
        let read_i8 = |dst: &mut [i8], off: usize| {
            for i in 0..dst.len() {
                dst[i] = in_bytes[off + i] as i8;
            }
        };
        let mut unavail4x4 = [0i8; 48];
        read_i8(&mut unavail4x4, o);
        let mut ref_pic_list0 = [0i8; 32];
        read_i8(&mut ref_pic_list0, o + 48);
        let mut ref_pic_list1 = [0i8; 32];
        read_i8(&mut ref_pic_list1, o + 80);
        let mut map_pic_to_list0 = [0i8; 32];
        read_i8(&mut map_pic_to_list0, o + 112);
        let mut dist_scale_factor = [0i16; 32];
        for i in 0..32 {
            dist_scale_factor[i] =
                i16::from_le_bytes([in_bytes[o + 144 + 2 * i], in_bytes[o + 145 + 2 * i]]);
        }
        State {
            mb: parse_mb(0),
            mb_a: parse_nb(148),
            mb_b: parse_nb(296),
            mb_c: parse_nb(444),
            mb_d: parse_nb(592),
            mb_col: col,
            unavail4x4,
            ref_pic_list0,
            ref_pic_list1,
            map_pic_to_list0,
            dist_scale_factor,
            col_short_term: in_bytes[o + 208] != 0,
            direct_8x8_inference_flag: in_bytes[o + 209] != 0,
        }
    }
}

// ---- packed i32 (x | y << 16) helpers -------------------------------------

fn pair_x(p: i32) -> i16 {
    p as i16
}
fn pair_y(p: i32) -> i16 {
    (p >> 16) as i16
}
fn pack(x: i16, y: i16) -> i32 {
    (x as u16 as i32) | ((y as u16 as i32) << 16)
}
/// `mvp + mvd` with i16 wrapping, per component.
fn add_pair(a: i32, b: i32) -> i32 {
    pack(
        (pair_x(a) as i32 + pair_x(b) as i32) as i16,
        (pair_y(a) as i32 + pair_y(b) as i32) as i16,
    )
}
/// `median16` on a pair: component-wise median of three pairs.
fn median_pair(a: i32, b: i32, c: i32) -> i32 {
    let med = |x: i16, y: i16, z: i16| x.max(y).min(z).max(x.min(y));
    pack(
        med(pair_x(a), pair_x(b), pair_x(c)),
        med(pair_y(a), pair_y(b), pair_y(c)),
    )
}
/// Current-MB packed mv at `mvs_s` index (i16 offset 2*idx).
fn mb_mvs_s(mb: &Mb, idx: usize) -> i32 {
    pack(mb.mvs[2 * idx], mb.mvs[2 * idx + 1])
}
/// packs16 saturation to i8 then abs8 (per component).
fn sat8abs(v: i16) -> u8 {
    v.clamp(-128, 127).unsigned_abs() as u8
}
/// `pack_absMvd(mvd)` as i8x16: packs16(a,b) = [sat8(a_l0..7), sat8(b_l0..7)]
/// (verified on hardware: all 16 bytes), so with x = broadcast32(mvd,0):
/// [ax, ay] x 8.
fn pack_absmvd_bytes(mvd: i32) -> [u8; 16] {
    let ax = sat8abs(pair_x(mvd));
    let ay = sat8abs(pair_y(mvd));
    let mut v = [0u8; 16];
    for j in 0..8 {
        v[2 * j] = ax;
        v[2 * j + 1] = ay;
    }
    v
}
/// `((i64x2)pack_absMvd(mvd))[0]`: the i8x16 result is byte-addressed, so the
/// low 8 bytes are [ax, ay, ax, ay, ax, ay, ax, ay].
fn pack_absmvd_lo64(mvd: i32) -> [u8; 8] {
    let ax = sat8abs(pair_x(mvd));
    let ay = sat8abs(pair_y(mvd));
    [ax, ay, ax, ay, ax, ay, ax, ay]
}

// ---- machine-verified vector helpers ---------------------------------------

/// `shufflen` on one byte: m < 0 -> 0xFF, else a[m & 15].
fn shufflen_byte(a: &[i8; 16], m: i8) -> i8 {
    if m < 0 { -1 } else { a[m as usize & 15] }
}
/// `shuffle2z` on one byte: m < 0 -> 0, m < 16 -> a[m], else b[m & 15].
fn shuffle2z_byte(a: &[i8; 16], b: &[i8; 16], m: i8) -> i8 {
    if m < 0 {
        0
    } else if m < 16 {
        a[m as usize]
    } else {
        b[m as usize & 15]
    }
}
/// `temporal_scale` per component (verified: sat16((mv*dfs + 128 + (dfs<0)) >> 8), floor shift).
fn temporal_scale(mv: i16, dfs: i16) -> i16 {
    let t = (mv as i32) * (dfs as i32) + 128 + (dfs < 0) as i32;
    (t >> 8).clamp(i16::MIN as i32, i16::MAX as i32) as i16
}
/// `mvs_near_zero` per i32 lane: both components satisfy abs16(v) >> 1 == 0.
fn near_zero_pair(x: i16, y: i16) -> bool {
    let ax = (x.wrapping_neg()).max(x) as u16;
    let ay = (y.wrapping_neg()).max(y) as u16;
    (ax >> 1) == 0 && (ay >> 1) == 0
}

/// Spatial-direct refIdx vector: `shuffle((u64x2){l} >> shift, shuf)` with
/// shuf = {0,0,0,0, 4,4,4,4, -1 x8}; on this CPU the -1 lanes yield 0.
fn refidx_vec(l: i64, shift: u32) -> [i8; 16] {
    let bytes = ((l as u64) >> shift).to_le_bytes();
    let mut v = [0i8; 16];
    for i in 0..4 {
        v[i] = bytes[0] as i8;
        v[4 + i] = bytes[4] as i8;
    }
    v
}

/// `(i32x4){mvs_s[i], mvs_s[j]}` as an i16x8: pairs at lanes 0-1 / 2-3, rest 0.
fn mv_vec(mvs_s: &[i32; 32], i: usize, j: usize) -> [i16; 8] {
    [
        pair_x(mvs_s[i]),
        pair_y(mvs_s[i]),
        pair_x(mvs_s[j]),
        pair_y(mvs_s[j]),
        0,
        0,
        0,
        0,
    ]
}

/// `mvs &= ~mask` where mask is FF/00 per i32 lane (pairs of equal i16 lanes).
fn zero_by(v: &mut [i16; 8], cm: &[bool; 4]) {
    for j in 0..4 {
        if cm[j] {
            v[2 * j] = 0;
            v[2 * j + 1] = 0;
        }
    }
}
/// Write one i16x8 chunk as `broadcast32(mv, 0)`: [X, Y, X, Y, X, Y, X, Y].
fn set_mv_chunk(v: &mut [i16; 64], base: usize, mv: i32) {
    let x = pair_x(mv);
    let y = pair_y(mv);
    for j in 0..8 {
        v[base + j] = if j & 1 == 0 { x } else { y };
    }
}

// ---- the ported functions ---------------------------------------------------

/// `decode_P_skip` (edge264_mvpred.c L44).
pub fn p_skip(s: &State) -> Mb {
    let mut out = s.mb;
    out.inter_eqs = 0x1b5fbbff;
    let r = s.ref_pic_list0[0];
    for i in 0..4 {
        out.ref_pic[i] = r;
    }
    for i in 4..8 {
        out.ref_pic[i] = -1;
    }
    let ref_idx_a = s.mb_a.ref_idx[1] as i32;
    let ref_idx_b = s.mb_b.ref_idx[2] as i32;
    let mv_a = s.mb_a.mvs_s[5];
    let mv_b = s.mb_b.mvs_s[10];
    let mut mv: i32 = 0;
    if (ref_idx_a | mv_a) != 0 && (ref_idx_b | mv_b) != 0 && (s.unavail4x4[0] & 3) == 0 {
        let (ref_idx_c, mv_c) = if (s.unavail4x4[5] & 4) != 0 {
            (s.mb_d.ref_idx[3] as i32, s.mb_d.mvs_s[15])
        } else {
            (s.mb_c.ref_idx[2] as i32, s.mb_c.mvs_s[10])
        };
        // C sets `mv` to the C/D neighbour first, then only overwrites it in the
        // median / eq!=4 branches. When eq==4 neither branch runs, so the C/D
        // value must be retained — not the zero initializer.
        mv = mv_c;
        let eq =
            (ref_idx_a == 0) as i32 + (ref_idx_b == 0) as i32 * 2 + (ref_idx_c == 0) as i32 * 4;
        if (0xe9 >> eq & 1) != 0 {
            mv = median_pair(mv_a, mv_b, mv_c);
        } else if eq != 4 {
            mv = if eq == 1 { mv_a } else { mv_b };
        }
    }
    // mvs_v[0..4] = broadcast32(mv, 0); mvs_v[4..8] = {}.
    for k in 0..4 {
        set_mv_chunk(&mut out.mvs, 8 * k, mv);
    }
    for i in 32..64 {
        out.mvs[i] = 0;
    }
    out
}

/// `decode_inter_16x16` (edge264_mvpred.c L83).
pub fn inter_16x16(s: &State, mvd: i32, lx: usize) -> Mb {
    let mut out = s.mb;
    let ref_idx = out.ref_idx[lx * 4] as i32;
    let ref_idx_a = s.mb_a.ref_idx[lx * 4 + 1] as i32;
    let ref_idx_b = s.mb_b.ref_idx[lx * 4 + 2] as i32;
    let mut eq_a = (ref_idx == ref_idx_a) as i32;
    let (ref_idx_c, mvp_c) = if (s.unavail4x4[5] & 4) != 0 {
        eq_a |= (s.unavail4x4[0] == 14) as i32;
        (
            s.mb_d.ref_idx[lx * 4 + 3] as i32,
            s.mb_d.mvs_s[lx * 16 + 15],
        )
    } else {
        (
            s.mb_c.ref_idx[lx * 4 + 2] as i32,
            s.mb_c.mvs_s[lx * 16 + 10],
        )
    };
    let eq = eq_a + (ref_idx == ref_idx_b) as i32 * 2 + (ref_idx == ref_idx_c) as i32 * 4;
    let mvp = if (0xe9 >> eq & 1) != 0 {
        median_pair(s.mb_a.mvs_s[lx * 16 + 5], s.mb_b.mvs_s[lx * 16 + 10], mvp_c)
    } else if eq == 1 {
        s.mb_a.mvs_s[lx * 16 + 5]
    } else if eq == 2 {
        s.mb_b.mvs_s[lx * 16 + 10]
    } else {
        mvp_c
    };
    let mv = add_pair(mvp, mvd);
    let base = lx * 32;
    for k in 0..4 {
        set_mv_chunk(&mut out.mvs, base + 8 * k, mv);
    }
    // absMvd_v[lx*2] = absMvd_v[lx*2+1] = pack_absMvd(mvd).
    let p = pack_absmvd_bytes(mvd);
    out.abs_mvd[base..base + 16].copy_from_slice(&p);
    out.abs_mvd[base + 16..base + 32].copy_from_slice(&p);
    out
}

/// `decode_inter_8x16_left` (edge264_mvpred.c L119).
pub fn inter_8x16_left(s: &State, mvd: i32, lx: usize) -> Mb {
    let mut out = s.mb;
    let ref_idx = out.ref_idx[lx * 4] as i32;
    let ref_idx_a = s.mb_a.ref_idx[lx * 4 + 1] as i32;
    let mvp = if ref_idx == ref_idx_a || s.unavail4x4[0] == 14 {
        s.mb_a.mvs_s[lx * 16 + 5]
    } else {
        let ref_idx_b = s.mb_b.ref_idx[lx * 4 + 2] as i32;
        let (ref_idx_c, mv_c) = if (s.unavail4x4[0] & 2) != 0 {
            (
                s.mb_d.ref_idx[lx * 4 + 3] as i32,
                s.mb_d.mvs_s[lx * 16 + 15],
            )
        } else {
            (
                s.mb_b.ref_idx[lx * 4 + 3] as i32,
                s.mb_b.mvs_s[lx * 16 + 14],
            )
        };
        if ref_idx == ref_idx_b {
            let mut m = s.mb_b.mvs_s[lx * 16 + 10];
            if ref_idx == ref_idx_c {
                m = median_pair(s.mb_a.mvs_s[lx * 16 + 5], m, mv_c);
            }
            m
        } else {
            let mut m = mv_c;
            if ref_idx != ref_idx_c {
                m = median_pair(s.mb_a.mvs_s[lx * 16 + 5], s.mb_b.mvs_s[lx * 16 + 10], m);
            }
            m
        }
    };
    let mv = add_pair(mvp, mvd);
    let base = lx * 32;
    for k in [0usize, 2] {
        set_mv_chunk(&mut out.mvs, base + 8 * k, mv);
    }
    // absMvd_l[lx*4] = absMvd_l[lx*4+2] = ((i64x2)pack_absMvd(mvd))[0].
    let p = pack_absmvd_lo64(mvd);
    out.abs_mvd[base..base + 8].copy_from_slice(&p);
    out.abs_mvd[base + 16..base + 24].copy_from_slice(&p);
    out
}

/// `decode_inter_8x16_right` (edge264_mvpred.c L161).
pub fn inter_8x16_right(s: &State, mvd: i32, lx: usize) -> Mb {
    let mut out = s.mb;
    let ref_idx = out.ref_idx[lx * 4 + 1] as i32;
    let (ref_idx_c, mv_c) = if (s.unavail4x4[5] & 4) != 0 {
        (
            s.mb_b.ref_idx[lx * 4 + 2] as i32,
            s.mb_b.mvs_s[lx * 16 + 11],
        )
    } else {
        (
            s.mb_c.ref_idx[lx * 4 + 2] as i32,
            s.mb_c.mvs_s[lx * 16 + 10],
        )
    };
    let mvp = if ref_idx == ref_idx_c {
        mv_c
    } else {
        let ref_idx_a = out.ref_idx[lx * 4] as i32;
        let ref_idx_b = s.mb_b.ref_idx[lx * 4 + 3] as i32;
        if ref_idx == ref_idx_b {
            let mut m = s.mb_b.mvs_s[lx * 16 + 14];
            if ref_idx == ref_idx_a {
                m = median_pair(mb_mvs_s(&out, lx * 16), m, mv_c);
            }
            m
        } else {
            let mut m = mb_mvs_s(&out, lx * 16);
            if ref_idx != ref_idx_a && s.unavail4x4[5] != 14 {
                m = median_pair(m, s.mb_b.mvs_s[lx * 16 + 14], mv_c);
            }
            m
        }
    };
    let mv = add_pair(mvp, mvd);
    let base = lx * 32;
    for k in [1usize, 3] {
        set_mv_chunk(&mut out.mvs, base + 8 * k, mv);
    }
    // absMvd_l[lx*4+1] = absMvd_l[lx*4+3] = ((i64x2)pack_absMvd(mvd))[0].
    let p = pack_absmvd_lo64(mvd);
    out.abs_mvd[base + 8..base + 16].copy_from_slice(&p);
    out.abs_mvd[base + 24..base + 32].copy_from_slice(&p);
    out
}

/// `decode_inter_16x8_top` (edge264_mvpred.c L202).
pub fn inter_16x8_top(s: &State, mvd: i32, lx: usize) -> Mb {
    let mut out = s.mb;
    let ref_idx = out.ref_idx[lx * 4] as i32;
    let ref_idx_b = s.mb_b.ref_idx[lx * 4 + 2] as i32;
    let mvp = if ref_idx == ref_idx_b {
        s.mb_b.mvs_s[lx * 16 + 10]
    } else {
        let ref_idx_a = s.mb_a.ref_idx[lx * 4 + 1] as i32;
        let mut eq_a = (ref_idx == ref_idx_a) as i32;
        let (ref_idx_c, mv_c) = if (s.unavail4x4[5] & 4) != 0 {
            eq_a |= (s.unavail4x4[0] == 14) as i32;
            (
                s.mb_d.ref_idx[lx * 4 + 3] as i32,
                s.mb_d.mvs_s[lx * 16 + 15],
            )
        } else {
            (
                s.mb_c.ref_idx[lx * 4 + 2] as i32,
                s.mb_c.mvs_s[lx * 16 + 10],
            )
        };
        if ref_idx == ref_idx_c {
            let mut m = mv_c;
            if eq_a != 0 {
                m = median_pair(s.mb_a.mvs_s[lx * 16 + 5], s.mb_b.mvs_s[lx * 16 + 10], m);
            }
            m
        } else {
            let mut m = s.mb_a.mvs_s[lx * 16 + 5];
            if eq_a == 0 {
                m = median_pair(m, s.mb_b.mvs_s[lx * 16 + 10], mv_c);
            }
            m
        }
    };
    let mv = add_pair(mvp, mvd);
    let base = lx * 32;
    for k in [0usize, 1] {
        set_mv_chunk(&mut out.mvs, base + 8 * k, mv);
    }
    // absMvd_v[lx*2] = pack_absMvd(mvd).
    let p = pack_absmvd_bytes(mvd);
    out.abs_mvd[base..base + 16].copy_from_slice(&p);
    out
}

/// `decode_inter_16x8_bottom` (edge264_mvpred.c L246).
pub fn inter_16x8_bottom(s: &State, mvd: i32, lx: usize) -> Mb {
    let mut out = s.mb;
    let ref_idx = out.ref_idx[lx * 4 + 2] as i32;
    let ref_idx_a = s.mb_a.ref_idx[lx * 4 + 3] as i32;
    let mvp = if ref_idx == ref_idx_a {
        s.mb_a.mvs_s[lx * 16 + 13]
    } else {
        let ref_idx_b = out.ref_idx[lx * 4] as i32;
        let ref_idx_c = s.mb_a.ref_idx[lx * 4 + 1] as i32;
        if ref_idx == ref_idx_b {
            let mut m = mb_mvs_s(&out, lx * 16);
            if ref_idx == ref_idx_c {
                m = median_pair(s.mb_a.mvs_s[lx * 16 + 13], m, s.mb_a.mvs_s[lx * 16 + 7]);
            }
            m
        } else {
            let mut m = s.mb_a.mvs_s[lx * 16 + 7];
            if ref_idx != ref_idx_c {
                m = median_pair(s.mb_a.mvs_s[lx * 16 + 13], mb_mvs_s(&out, lx * 16), m);
            }
            m
        }
    };
    let mv = add_pair(mvp, mvd);
    let base = lx * 32;
    for k in [2usize, 3] {
        set_mv_chunk(&mut out.mvs, base + 8 * k, mv);
    }
    // absMvd_v[lx*2+1] = pack_absMvd(mvd).
    let p = pack_absmvd_bytes(mvd);
    out.abs_mvd[base + 16..base + 32].copy_from_slice(&p);
    out
}

/// C `decode_inter` pixel ops collected by the direct-mvpred ports. The
/// record computation runs to completion first; the caller executes these in
/// order after applying the record (bi-pred L0-at-`i` before L1-at-`i+16` is
/// preserved by C's own call order).
#[derive(Debug, Default)]
pub struct McOps {
    ops: [(usize, u32, u32); 32],
    len: usize,
}

impl McOps {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn push(&mut self, i: usize, w: u32, h: u32) {
        debug_assert!(self.len < self.ops.len(), "McOps overflow");
        self.ops[self.len] = (i, w, h);
        self.len += 1;
    }

    pub fn iter(&self) -> std::slice::Iter<'_, (usize, u32, u32)> {
        self.ops[..self.len].iter()
    }
}

/// `decode_direct_spatial_mv_pred` (edge264_mvpred.c L288).
pub fn direct_spatial(s: &State, direct_flags: u32, mc: &mut McOps) -> Mb {
    let mut out = s.mb;
    // load all refIdxN and mvN in vector registers
    let ref_idx_a = refidx_vec(s.mb_a.ref_idx_l(), 8);
    let ref_idx_b = refidx_vec(s.mb_b.ref_idx_l(), 16);
    let mv_a = mv_vec(&s.mb_a.mvs_s, 5, 21);
    let mv_b = mv_vec(&s.mb_b.mvs_s, 10, 26);
    let (mv_c, ref_idx_c) = if (s.unavail4x4[5] & 4) != 0 {
        (
            mv_vec(&s.mb_d.mvs_s, 15, 31),
            refidx_vec(s.mb_d.ref_idx_l(), 24),
        )
    } else {
        (
            mv_vec(&s.mb_c.mvs_s, 10, 26),
            refidx_vec(s.mb_c.ref_idx_l(), 16),
        )
    };

    // initialize mv along refIdx since it will equal one of refIdxA/B/C
    // (unsigned per-byte min/median selection). The shuffled refIdx vectors
    // are constant within each 4-byte group, so the byte-wise C blends reduce
    // to per-i16-lane selection; lanes 4-7 and bytes 8-15 stay zero.
    let mut ref_idx = [0i8; 16];
    let mut mv01 = [0i16; 8];
    let mut c_ab = [false; 4];
    let mut c_mc = [false; 4];
    let mut c_med = [false; 4];
    for i in 0..4 {
        let k = 2 * i;
        let cmp_ab = (ref_idx_a[k] as u8) < (ref_idx_b[k] as u8); // unsigned comparisons
        c_ab[i] = cmp_ab;
        let ref_m = if cmp_ab { ref_idx_a[k] } else { ref_idx_b[k] }; // umin(refIdxA, refIdxB)
        let ref_M = if cmp_ab { ref_idx_b[k] } else { ref_idx_a[k] }; // umax(refIdxA, refIdxB)
        let mv_m = if cmp_ab { mv_a[i] } else { mv_b[i] };
        let cmp_mc = (ref_m as u8) < (ref_idx_c[k] as u8);
        c_mc[i] = cmp_mc;
        let rv = if cmp_mc { ref_m } else { ref_idx_c[k] }; // umin(refIdxm, refIdxC)
        ref_idx[k] = rv;
        ref_idx[k + 1] = rv;
        let mv_mm = if cmp_mc { mv_m } else { mv_c[i] };
        // select median if refIdx equals another of refIdxA/B/C (3 cases: A=B<C, A=C<B, B=C<A)
        let cmp_med = ref_m == ref_idx_c[k] || rv == ref_M;
        c_med[i] = cmp_med;
        let med = |x: i16, y: i16, z: i16| x.max(y).min(z).max(x.min(y));
        mv01[i] = if cmp_med {
            med(mv_a[i], mv_b[i], mv_c[i])
        } else {
            mv_mm
        };
    }
    // broadcast32(mv01, N): replicate i32 lane N (an x/y pair) to all 4 lanes
    let mvs0 = [
        mv01[0], mv01[1], mv01[0], mv01[1], mv01[0], mv01[1], mv01[0], mv01[1],
    ]; // broadcast32(mv01, 0)
    let mvs4 = [
        mv01[2], mv01[3], mv01[2], mv01[3], mv01[2], mv01[3], mv01[2], mv01[3],
    ]; // broadcast32(mv01, 1)

    // direct zero prediction applies only to refIdx (mvLX are zero already):
    // a whole 8-byte half is -1 iff all neighbours were unavailable for both lists
    let to_u64 = |src: &[i8]| -> u64 {
        let mut b = [0u8; 8];
        for i in 0..8 {
            b[i] = src[i] as u8;
        }
        u64::from_le_bytes(b)
    };
    let lo = to_u64(&ref_idx[0..8]);
    let hi = to_u64(&ref_idx[8..16]);
    if lo == u64::MAX {
        for b in &mut ref_idx[0..8] {
            *b = 0;
        }
    }
    if hi == u64::MAX {
        for b in &mut ref_idx[8..16] {
            *b = 0;
        }
    }

    // mb->refPic_s[0/1] = shufflen(RefPicList_v[0/2], refIdx)[0/1]
    let rpl0: [i8; 16] = s.ref_pic_list0[..16].try_into().unwrap();
    let rpl1: [i8; 16] = s.ref_pic_list1[..16].try_into().unwrap();
    for (op, &ri) in out.ref_pic[..4].iter_mut().zip(&ref_idx[..4]) {
        *op = shufflen_byte(&rpl0, ri);
    }
    for (op, &ri) in out.ref_pic[4..].iter_mut().zip(&ref_idx[4..]) {
        *op = shufflen_byte(&rpl1, ri);
    }

    // trick from ffmpeg: skip computations on refCol/mvCol if both mvs are zero
    if mv01[0] != 0 || mv01[1] != 0 || mv01[2] != 0 || mv01[3] != 0 || direct_flags != 0xffffffff {
        let mut col_zero_mask = [[false; 4]; 4];
        let mut col_zero_flags: u32 = 0;
        if s.col_short_term {
            let col = &s.mb_col;
            // offsets = refColL0 & 32: byte K selects the L1 mv half when the
            // collocated block's L0 refIdx is negative
            let mut mvcol = [[0i16; 8]; 4];
            for (k, slot) in mvcol.iter_mut().enumerate() {
                let b = (col.ref_idx_s[0] >> (8 * k)) as i8;
                let off = ((b as u8) & 32) as usize;
                *slot = col.mvs[8 * k + off..8 * k + off + 8].try_into().unwrap();
            }
            // refCol = ifelse_msb(refColL0, refIdx_s[1], refColL0), bytes 0-3.
            // C: refColZero = ((i32x4)(refCol == 0))[0] — byte k is FF/00, so
            // the zero flag for byte k occupies bits 8k..8k+7 (not bit k).
            let mut ref_col_zero: u32 = 0;
            for k in 0..4 {
                let b0 = (col.ref_idx_s[0] >> (8 * k)) as i8;
                let b1 = (col.ref_idx_s[1] >> (8 * k)) as i8;
                if (if b0 < 0 { b1 } else { b0 }) == 0 {
                    ref_col_zero |= 0xFF << (8 * k);
                }
            }
            if s.direct_8x8_inference_flag {
                for n in 0..4 {
                    // broadcast32(mvColN, N): replicate pair N to all 4 pairs
                    let p = [mvcol[n][2 * n], mvcol[n][2 * n + 1]];
                    mvcol[n] = [p[0], p[1], p[0], p[1], p[0], p[1], p[0], p[1]];
                }
            }
            for (k, bit) in [(0usize, 1u32), (1, 1 << 8), (2, 1 << 16), (3, 1 << 24)] {
                if ref_col_zero & bit != 0 {
                    for j in 0..4 {
                        col_zero_mask[k][j] = near_zero_pair(mvcol[k][2 * j], mvcol[k][2 * j + 1]);
                    }
                }
            }
            // colZeroFlags = movemask(packs16(packs32(cm0,cm1), packs32(cm2,cm3))):
            // packs16(a,b) = [sat8(a_l0..7), sat8(b_l0..7)] (all 16 bytes,
            // hardware-verified) and packs32(a,b) = [a_l0..3, b_l0..3], so
            // bit 4N+j is set iff colZeroMaskN lane j is FF.
            for (n, row) in col_zero_mask.iter().enumerate() {
                for (j, &flag) in row.iter().enumerate() {
                    if flag {
                        col_zero_flags |= 1 << (4 * n + j);
                    }
                }
            }
        }

        // skip computations on colZeroFlags if none are set
        if col_zero_flags != 0 || direct_flags != 0xffffffff {
            let mut mvd_flags = direct_flags;
            let mut v = [mvs0; 4]; // mvs0..mvs3 (L0 vectors)
            let mut w = [mvs4; 4]; // mvs4..mvs7 (L1 vectors)
            if ref_idx[0] == 0 {
                col_zero_flags = col_zero_flags.wrapping_add(col_zero_flags << 16);
                for k in 0..4 {
                    zero_by(&mut v[k], &col_zero_mask[k]);
                }
            } else {
                col_zero_flags <<= 16;
                if ref_idx[0] < 0 {
                    mvd_flags &= 0xffff_0000;
                }
            }
            if ref_idx[4] == 0 {
                for k in 0..4 {
                    zero_by(&mut w[k], &col_zero_mask[k]);
                }
            } else {
                col_zero_flags &= 0x0000_ffff;
                if ref_idx[4] < 0 {
                    mvd_flags &= 0x0000_ffff;
                }
            }

            // conditional memory storage
            for (k, bit) in [(0usize, 1u32), (1, 1 << 4), (2, 1 << 8), (3, 1 << 12)] {
                if direct_flags & bit != 0 {
                    out.ref_idx[k] = ref_idx[0];
                    out.ref_idx[4 + k] = ref_idx[4];
                    out.mvs[8 * k..8 * k + 8].copy_from_slice(&v[k]);
                    out.mvs[32 + 8 * k..32 + 8 * k + 8].copy_from_slice(&w[k]);
                }
            }

            // iteratively cut the area into blocks with uniform colZeroFlags values
            const SCOPES: [u16; 16] = [
                0xffff, 0x505, 0x33, 0x1, 0xf0f, 0x505, 0x3, 0x1, 0xff, 0x5, 0x33, 0x1, 0xf, 0x5,
                0x3, 0x1,
            ];
            const MASKS: [u16; 8] = [0xffff, 0xff, 0xf0f, 0xf, 0x33, 0x3, 0x5, 0x1];
            const EQS: [u32; 8] = [0x1b5fbbff, 0x1b5f, 0x1b00bb, 0x1b, 0x0105, 0x1, 0x2, 0];
            let mut inter_eqs: u64 = 0;
            while mvd_flags != 0 {
                let i = mvd_flags.trailing_zeros();
                let t = ((mvd_flags >> i) & (SCOPES[i as usize & 15] as u32)) as u16;
                let c = (col_zero_flags >> i) as u16;
                // type = ctz(movemask(((mt&mc)==masks) | ((mt&~mc)==masks))) >> 1:
                // the lowest matched lane (movemask sets bit pairs 2k,2k+1)
                let ty = (0..8)
                    .find(|&k| {
                        let m = MASKS[k] as u32;
                        let mt = (t as u32) & m;
                        let mc = (c as u32) & 0xffff;
                        (mt & mc) == m || (mt & !mc) == m
                    })
                    .expect("mvpred spatial: no uniform scope type");
                mvd_flags ^= (MASKS[ty] as u32) << i;
                inter_eqs |= (EQS[ty] as u64) << (i * 2);
                // C decode_inter(ctx, i, widths[type], heights[type])
                const SPATIAL_WIDTHS: [u32; 8] = [16, 16, 8, 8, 16, 8, 4, 4];
                const SPATIAL_HEIGHTS: [u32; 8] = [16, 8, 16, 8, 4, 4, 8, 4];
                mc.push(i as usize, SPATIAL_WIDTHS[ty], SPATIAL_HEIGHTS[ty]);
            }
            out.inter_eqs |= (inter_eqs & (inter_eqs >> 32)) as u32;
            return out;
        }
    }

    // fallback if we did not need colZeroFlags
    // (C: mb->refIdx_l = ((i64x2)refIdx)[0] — only the first 8 bytes)
    out.ref_idx.copy_from_slice(&ref_idx[..8]);
    for k in 0..4 {
        out.mvs[8 * k..8 * k + 8].copy_from_slice(&mvs0);
        out.mvs[32 + 8 * k..32 + 8 * k + 8].copy_from_slice(&mvs4);
    }
    out.inter_eqs = 0x1b5fbbff;
    // C: if (refIdx[0] >= 0) decode_inter(ctx, 0, 16, 16);
    //     if (refIdx[4] >= 0) decode_inter(ctx, 16, 16, 16);
    if out.ref_idx[0] >= 0 {
        mc.push(0, 16, 16);
    }
    if out.ref_idx[4] >= 0 {
        mc.push(16, 16, 16);
    }
    out
}

/// C `extract_neighbours` (edge264_internal.h L1189): `_pext_u32(f, 0x27)` —
/// keep bits 0-2 and bit 5 of `f`, packed to the low nibble.
#[inline]
fn extract_neighbours(f: u32) -> u32 {
    (f & 7) | (f >> 2 & 8)
}

/// `decode_direct_temporal_mv_pred` (edge264_mvpred.c L445).
pub fn direct_temporal(s: &State, direct_flags: u32, mc: &mut McOps) -> Mb {
    let mut out = s.mb;
    let col = &s.mb_col;
    // offsets = refPicColL0 & 32: byte K selects the L1 mv half when the
    // collocated block's L0 refPic is negative
    let mut mvcol = [[0i16; 8]; 4];
    for (k, slot) in mvcol.iter_mut().enumerate() {
        let b = (col.ref_pic_s[0] >> (8 * k)) as i8;
        let off = ((b as u8) & 32) as usize;
        *slot = col.mvs[8 * k + off..8 * k + off + 8].try_into().unwrap();
    }
    let mut inter_eqs = col.inter_eqs;
    if s.direct_8x8_inference_flag {
        for n in 0..4 {
            // broadcast32(mvColN, N): replicate pair N to all 4 pairs
            let p = [mvcol[n][2 * n], mvcol[n][2 * n + 1]];
            mvcol[n] = [p[0], p[1], p[0], p[1], p[0], p[1], p[0], p[1]];
        }
        inter_eqs |= 0x1b1b1b1b;
    }

    // refIdx = shuffle2z(MapPicToList0_v[0], MapPicToList0_v[1], refPicCol)
    let mptl_lo: [i8; 16] = s.map_pic_to_list0[..16].try_into().unwrap();
    let mptl_hi: [i8; 16] = s.map_pic_to_list0[16..].try_into().unwrap();
    let mut ri = [0i8; 4];
    for (k, slot) in ri.iter_mut().enumerate() {
        // refPicCol byte K = ifelse_msb(refPicColL0, refPic_s[1], refPicColL0)
        let b0 = (col.ref_pic_s[0] >> (8 * k)) as i8;
        let b1 = (col.ref_pic_s[1] >> (8 * k)) as i8;
        *slot = shuffle2z_byte(&mptl_lo, &mptl_hi, if b0 < 0 { b1 } else { b0 });
    }

    // mb->refPic_s[0] = shufflen(RefPicList_v[0], refIdx)[0];
    // mb->refPic_s[1] = broadcast8(RefPicList_v[2], 0)[0] (refIdxL1 is 0)
    let rpl0: [i8; 16] = s.ref_pic_list0[..16].try_into().unwrap();
    for (op, &ri) in out.ref_pic[..4].iter_mut().zip(&ri) {
        *op = shufflen_byte(&rpl0, ri);
    }
    for op in &mut out.ref_pic[4..] {
        *op = s.ref_pic_list1[0];
    }

    let consts: [u32; 3] = [0x0000_00ff, 0x0000_bb44, 0x005f_00a0];
    for (k, bit) in [(0usize, 1u32), (1, 1 << 4), (2, 1 << 8), (3, 1 << 12)] {
        if direct_flags & bit != 0 {
            out.ref_idx[k] = ri[k];
            out.ref_idx[4 + k] = 0; // refIdxL1 is 0
            let dfs = s.dist_scale_factor[ri[k] as usize];
            // C: mb->mvs_v[k] = temporal_scale(mvColK, ...); mb->mvs_v[4+k] = mb->mvs_v[k] - mvColK;
            let mut l0 = [0i16; 8];
            for (op, &mv) in l0.iter_mut().zip(&mvcol[k]) {
                *op = temporal_scale(mv, dfs);
            }
            out.mvs[8 * k..8 * k + 8].copy_from_slice(&l0);
            for (op, (&a, &b)) in out.mvs[32 + 8 * k..32 + 8 * k + 8]
                .iter_mut()
                .zip(l0.iter().zip(&mvcol[k]))
            {
                *op = a.wrapping_sub(b);
            }
        } else if k < 3 {
            inter_eqs &= !consts[k];
        } else {
            // edge case: 16x16 with a direct8x8 block on the bottom-right corner
            inter_eqs = if inter_eqs == 0x1b5fbbff {
                0x001b1b5f
            } else {
                inter_eqs & !0x1b44a000
            };
        }
    }

    // mb->f.inter_eqs_s |= little_endian32(inter_eqs)
    out.inter_eqs |= inter_eqs;
    // execute decode_inter for the positions given in the mask (C L496-507):
    // each L0 block i also MCs its L1 twin at i+16.
    const T_MASKS: [u32; 16] = [
        0x1, 0x3, 0x5, 0xf, 0x1, 0x33, 0x5, 0xff, 0x1, 0x3, 0x0505, 0x0f0f, 0x1, 0x33, 0x0505,
        0xffff,
    ];
    const T_WIDTHS: [u32; 16] = [4, 8, 4, 8, 4, 16, 4, 16, 4, 8, 4, 8, 4, 16, 4, 16];
    const T_HEIGHTS: [u32; 16] = [4, 4, 8, 8, 4, 4, 8, 8, 4, 4, 16, 16, 4, 4, 16, 16];
    let mut flags = direct_flags & 0xffff;
    while flags != 0 {
        let i = flags.trailing_zeros();
        // C: type = extract_neighbours(inter_eqs >> i * 2) & ~i
        let ty = (extract_neighbours(inter_eqs >> (i * 2)) & !i) as usize;
        flags ^= T_MASKS[ty] << i;
        mc.push(i as usize, T_WIDTHS[ty], T_HEIGHTS[ty]);
        mc.push((i + 16) as usize, T_WIDTHS[ty], T_HEIGHTS[ty]);
    }
    out
}

/// `decode_direct_mv_pred` dispatcher (edge264_mvpred.c L517).
pub fn direct(s: &State, spatial_flag: bool, direct_flags: u32, mc: &mut McOps) -> Mb {
    if spatial_flag {
        direct_spatial(s, direct_flags, mc)
    } else {
        direct_temporal(s, direct_flags, mc)
    }
}

/// Run one op (0..12, matching the C oracle op codes) and serialize the
/// resulting macroblock state in the oracle output layout (212 bytes).
pub fn run_rust_mvpred(in_bytes: &[u8], op: i32) -> Vec<u8> {
    let s = State::parse(in_bytes);
    let o = 6 * 148 + 208;
    let mvd = pack(
        i16::from_le_bytes([in_bytes[o + 3], in_bytes[o + 4]]),
        i16::from_le_bytes([in_bytes[o + 5], in_bytes[o + 6]]),
    );
    let direct_flags = u32::from_le_bytes(in_bytes[o + 7..o + 11].try_into().unwrap());
    let mb = match op {
        0 => p_skip(&s),
        1 => inter_16x16(&s, mvd, 0),
        2 => inter_16x16(&s, mvd, 1),
        3 => inter_8x16_left(&s, mvd, 0),
        4 => inter_8x16_left(&s, mvd, 1),
        5 => inter_8x16_right(&s, mvd, 0),
        6 => inter_8x16_right(&s, mvd, 1),
        7 => inter_16x8_top(&s, mvd, 0),
        8 => inter_16x8_top(&s, mvd, 1),
        9 => inter_16x8_bottom(&s, mvd, 0),
        10 => inter_16x8_bottom(&s, mvd, 1),
        // slice.c:1193 skips decode_direct_mv_pred when direct_flags == 0
        // (the MB keeps its parsed state); mirror that so the do-while is
        // never entered on ctz(0).
        11 if direct_flags != 0 => direct_spatial(&s, direct_flags, &mut McOps::new()),
        12 if direct_flags != 0 => direct_temporal(&s, direct_flags, &mut McOps::new()),
        11 | 12 => s.mb,
        _ => panic!("bad op {op}"),
    };
    let mut out = vec![0u8; 212];
    for (op, &ri) in out[..8].iter_mut().zip(&mb.ref_idx) {
        *op = ri as u8;
    }
    for i in 0..64 {
        out[8 + 2 * i..10 + 2 * i].copy_from_slice(&mb.mvs[i].to_le_bytes());
    }
    for i in 0..8 {
        out[136 + i] = mb.ref_pic[i] as u8;
    }
    out[144..148].copy_from_slice(&mb.inter_eqs.to_le_bytes());
    out[148..212].copy_from_slice(&mb.abs_mvd);
    out
}

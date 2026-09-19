//! Inter motion compensation (Tier B). Ports C `decode_inter_luma` /
//! `decode_inter_chroma` from `c/src/edge264_inter.c` bit-exactly.
//!
//! Mode encoding (matches C): `mode = base + yFrac*4 + xFrac`, with bases
//! 4xH=0, 8xH=16, 16xH=32 and xFrac/yFrac in 0..3 (luma quarter-pel). The C
//! enum names are QPEL_XY with X=xFrac leading, so `QPEL_00` (integer pel) is
//! mode 0 / 16 / 32.
//!
//! The scalar model below was validated sample-by-sample against the C kernel
//! for all 48 luma modes (see /tmp/ziptest/model.py + interp_out.txt).

/// `no_weight` wod pattern.
pub const WOD_NO_WEIGHT: [i16; 8] = [256, 0, 0, 0, 256, 256, 0, 0];

const F: [i32; 6] = [1, -5, 20, 20, -5, 1];

#[inline]
fn sat8(x: i32) -> u8 {
    x.clamp(0, 255) as u8
}

#[inline]
fn sat16(x: i32) -> i32 {
    x.clamp(i16::MIN as i32, i16::MAX as i32)
}

/// C `sixtapHV` — exact factored form with i16 arithmetic shifts.
#[inline]
fn sixtap_hv(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    let af = a + f;
    let be = b + e;
    let cd = c + d;
    ((((af - be) >> 2) + (cd - be)) >> 2) + cd
}

/// Luma inter MC. `src` is the neighborhood around the block: rows Y-2..Y+h+2,
/// cols X-2..X+w+2 at `sstride`, with the integer-pel block top-left at
/// `src[2*sstride + 2]`. `dst` holds the initial w x h pixels at `dstride`
/// (q values for weighted prediction) and receives the MC result in place.
#[allow(clippy::too_many_arguments)]
pub fn inter_luma(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    mode: u32,
    sstride: usize,
    dstride: usize,
    wod: &[i16; 8],
) {
    let x = (mode & 3) as isize;
    let y = ((mode >> 2) & 3) as isize;
    // Block-relative sample; r in -2..h+2, c in -2..w+2.
    let g =
        |r: isize, c: isize| -> i32 { src[(r + 2) as usize * sstride + (c + 2) as usize] as i32 };
    // Raw 6-tap sums (unrounded).
    let h6 = |row: isize, c: isize| -> i32 {
        let mut s = 0i32;
        for (k, &f) in F.iter().enumerate() {
            s += f * g(row, c + k as isize);
        }
        s
    };
    let v6 = |col: isize, row: isize| -> i32 {
        let mut s = 0i32;
        for (k, &f) in F.iter().enumerate() {
            s += f * g(row - 2 + k as isize, col);
        }
        s
    };
    // Half-pel u8 samples.
    let hp_h = |row: isize, c: isize| -> u8 { sat8((h6(row, c - 2) + 16) >> 5) };
    let hp_v = |row: isize, col: isize| -> u8 { sat8((v6(col, row) + 16) >> 5) };
    // 2D half-half u8 sample at (r+1/2, c+1/2).
    let hh = |row: isize, c: isize| -> u8 {
        let s = sixtap_hv(
            h6(row - 2, c - 2),
            h6(row - 1, c - 2),
            h6(row, c - 2),
            h6(row + 1, c - 2),
            h6(row + 2, c - 2),
            h6(row + 3, c - 2),
        );
        sat8((s + 32) >> 6)
    };
    let avg = |a: u8, b: u8| -> u8 { ((a as u16 + b as u16 + 1) >> 1) as u8 };

    let wq = (wod[0] & 0xFF) as i8 as i32;
    let wp = ((wod[0] >> 8) & 0xFF) as i8 as i32;
    let oy = wod[1] as i32;

    for r in 0..h as isize {
        for c in 0..w as isize {
            let p: u8 = match (x, y) {
                (0, 0) => g(r, c) as u8,
                (1, 0) => avg(g(r, c) as u8, hp_h(r, c)),
                (2, 0) => hp_h(r, c),
                (3, 0) => avg(g(r, c + 1) as u8, hp_h(r, c)),
                (0, 1) => avg(g(r, c) as u8, hp_v(r, c)),
                (0, 2) => hp_v(r, c),
                (0, 3) => avg(g(r + 1, c) as u8, hp_v(r, c)),
                (1, 1) => avg(hp_v(r, c), hp_h(r, c)),
                (3, 1) => avg(hp_v(r, c + 1), hp_h(r, c)),
                (1, 3) => avg(hp_v(r, c), hp_h(r + 1, c)),
                (3, 3) => avg(hp_v(r, c + 1), hp_h(r + 1, c)),
                (2, 2) => hh(r, c),
                (2, 1) => avg(hh(r, c), hp_h(r, c)),
                (2, 3) => avg(hh(r, c), hp_h(r + 1, c)),
                (1, 2) => avg(hh(r, c), hp_v(r, c)),
                (3, 2) => avg(hh(r, c), hp_v(r, c + 1)),
                _ => unreachable!("bad inter mode {mode}"),
            };
            // C maddshrL: this toolchain lowers _mm_sra_epi16 to a uniform
            // shift by wd[0] = wod[2] on every lane (verified empirically).
            let q = dst[(r as usize) * dstride + c as usize] as i32;
            let v = sat16(sat16(q * wq + p as i32 * wp) + oy) >> wod[2] as u32;
            dst[(r as usize) * dstride + c as usize] = sat8(v);
        }
    }
}

/// C `_mm_sra_epi16` on this machine: every lane shifts by the low i64 count
/// with floor-division semantics (verified empirically, /tmp/ziptest/mpsra*).
/// For i16-range values that is a plain shift for k < 15 and sign-saturation
/// beyond; `v` is always in the i16 range here.
#[inline]
fn sra_machine(v: i32, k: u32) -> i32 {
    if k < 15 {
        v >> k
    } else if v < 0 {
        -1
    } else {
        0
    }
}

/// Combined Cb+Cr inter chroma MC. `src` holds (h+2) reference rows x (cw+1)
/// cols at `sstride`: row 2k = Cb ref row k, row 2k+1 = Cr ref row k (the
/// kernel reads up to row h+1). `dst` holds the initial h rows x cw cols at
/// `dstride` (row 2k = Cb output row k, row 2k+1 = Cr output row k) and
/// receives the result in place. `xFrac`/`yFrac` are the sub-chroma-pel
/// positions (0..7; chroma quarter-pel = luma eighth-pel).
#[allow(clippy::too_many_arguments)]
pub fn inter_chroma(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    x_frac: u32,
    y_frac: u32,
    sstride: usize,
    dstride: usize,
    wod: &[i16; 8],
) {
    let cw = w / 2;
    let x_f = x_frac & 7;
    let y_f = y_frac & 7;
    // Bilinear coeffs (all in 0..=64, sum to 64).
    let a = ((8 - x_f) * (8 - y_f)) as i32;
    let b = (x_f * (8 - y_f)) as i32;
    let c = ((8 - x_f) * y_f) as i32;
    let d = (x_f * y_f) as i32;
    // C: wd = (u64x2)wod >> 48; this machine shifts every lane by the low i64
    // count (= wod[3], zero-extended) with floor-division semantics.
    let sh = wod[3] as u16 as u32;
    // Weight bytes are the signed second operand of pmaddubsw.
    let wt =
        |i: usize| -> (i32, i32) { (((wod[i] & 0xFF) as i8) as i32, ((wod[i] >> 8) as i8) as i32) };
    // plane 0 = Cb (out row 2k), plane 1 = Cr (out row 2k+1). Bilinear between
    // ref row k and ref row k+1, which sit in buffer rows 2k+plane and
    // 2(k+1)+plane.
    for k in 0..(h / 2) {
        for (plane, wi, oi) in [(0usize, 4usize, 6usize), (1, 5, 7)] {
            let (r0, r1) = ((2 * k + plane) * sstride, (2 * (k + 1) + plane) * sstride);
            let (wq, wp) = wt(wi);
            let o = wod[oi] as i32;
            let row = (2 * k + plane) * dstride;
            for col in 0..cw {
                // Bilinear: unsigned samples, non-negative coeffs; x <= 64*255
                // so no i16 saturation is reachable.
                let x = a * src[r0 + col] as i32
                    + b * src[r0 + col + 1] as i32
                    + c * src[r1 + col] as i32
                    + d * src[r1 + col + 1] as i32;
                let p = sat8((x + 32) >> 6);
                let q = dst[row + col] as i32;
                let v = sra_machine(sat16(sat16(q * wq + p as i32 * wp) + o), sh);
                dst[row + col] = sat8(v);
            }
        }
    }
}

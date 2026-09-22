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
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            unsafe {
                // `src` is the neighborhood (block top-left minus 2 rows/cols);
                // the C kernel's `src2` is the block top-left itself.
                return sse::inter_luma_sse(
                    src.as_ptr().add(2 * sstride + 2),
                    dst.as_mut_ptr(),
                    h,
                    mode,
                    sstride,
                    dstride,
                    wod,
                );
            }
        }
    }
    inter_luma_scalar(src, dst, w, h, mode, sstride, dstride, wod);
}

#[allow(clippy::too_many_arguments)]
fn inter_luma_scalar(
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
pub(super) fn sra_machine(v: i32, k: u32) -> i32 {
    if k < 15 {
        v >> k
    } else if v < 0 {
        -1
    } else {
        0
    }
}

#[cfg(test)]
mod oracle_tests {
    use super::*;

    fn pack_w(w0: i32, w1: i32) -> i16 {
        ((w1 << 8) | (w0 & 255)) as i16
    }

    /// Bit-exact check of the luma MC against the real C `decode_inter_luma`
    /// (edge264_inter.c) with a deterministic LCG neighborhood. The C source
    /// was removed in the E4 cutover but lives on in git history (9cdae1f^).
    /// `oracle_luma.txt` pins this machine's GCC/SSE output for all 48 luma
    /// modes x 7 (w,h) combos x 3 weight patterns; it encodes machine-specific
    /// SIMD behavior (e.g. the _mm_sra_epi16 floor-division quirk), so
    /// regenerate it ON THE MACHINE where the tests run, via `oracle/harness.c`
    /// (extract the C tree first: `git archive 9cdae1f^ crates/vacc-sw-decode/c | tar -x`
    /// and point the compile line's -I at the extracted c/src). See the
    /// harness header for the exact steps.
    #[test]
    fn oracle_luma() {
        let text = include_str!("oracle_luma.txt");
        let mut lines = text.lines();
        let mut rng: u32 = 0x12345678;
        let rnd = |st: &mut u32| -> u8 {
            *st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            (*st >> 24) as u8
        };

        let combos = [(4, 4), (4, 8), (8, 4), (8, 8), (8, 16), (16, 8), (16, 16)];
        let wods: [[i16; 8]; 3] = [
            [256, 0, 0, 0, 256, 256, 0, 0],
            [257, 1, 1, 1, 257, 257, 1, 1],
            [pack_w(64, 120), 33, 6, 6, pack_w(-30, 90), pack_w(20, -50), 45, 99],
        ];

        for (zi, wod) in wods.iter().enumerate() {
            for &(w, h) in &combos {
                let base = if w == 4 { 0 } else if w == 8 { 16 } else { 32 };
                for xy in 0..16 {
                    let mut nb = [0u8; 64 * 32];
                    let mut dstb = [0u8; 64 * 32];
                    for b in nb.iter_mut() {
                        *b = rnd(&mut rng);
                    }
                    for b in dstb.iter_mut() {
                        *b = rnd(&mut rng);
                    }

                    let header = lines.next().unwrap();
                    assert_eq!(header, format!("M{} W{} H{} Z{}", base + xy, w, h, zi));
                    let mut expected = Vec::with_capacity(w * h);
                    for _ in 0..h {
                        let line = lines.next().unwrap();
                        for i in (0..line.len()).step_by(2) {
                            expected.push(u8::from_str_radix(&line[i..i + 2], 16).unwrap());
                        }
                    }

                    // Rust view starts at (yInt-2, xInt-2) = nb[4*32+4]; C's
                    // src2 is the block top-left at nb[6*32+6].
                    let sstride = 32usize;
                    let src = &nb[4 * sstride + 4..(4 * sstride + 4) + (h + 4) * sstride + w + 5];
                    let dst = &mut dstb[6 * sstride + 6..(6 * sstride + 6) + (h - 1) * sstride + w];
                    inter_luma(src, dst, w, h, (base + xy) as u32, sstride, sstride, wod);
                    for r in 0..h {
                        assert_eq!(
                            &dst[r * sstride..r * sstride + w],
                            &expected[r * w..(r + 1) * w],
                            "mismatch C mode {} w={} h={} z={} row {}",
                            base + xy,
                            w,
                            h,
                            zi,
                            r
                        );
                    }
                }
            }
        }
    }
}

/// Combined Cb+Cr inter chroma MC. `src` holds (h+2) reference rows at
/// `sstride`: row 2k = Cb ref row k, row 2k+1 = Cr ref row k (the kernel
/// reads up to row h+1; the widest load is 16 bytes per row, of which only
/// the first cw+1 cols are used). `dst` holds the initial h rows x cw cols
/// at `dstride` (row 2k = Cb output row k, row 2k+1 = Cr output row k) and
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
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            unsafe {
                return sse::inter_chroma_sse(
                    src.as_ptr(),
                    dst.as_mut_ptr(),
                    w,
                    h,
                    x_frac,
                    y_frac,
                    sstride,
                    dstride,
                    wod,
                );
            }
        }
    }
    inter_chroma_scalar(src, dst, w, h, x_frac, y_frac, sstride, dstride, wod);
}

#[allow(clippy::too_many_arguments)]
fn inter_chroma_scalar(
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

/// SSE/SSSE3/SSE4.1 port of C `decode_inter_luma` (edge264_inter.c), bit-exact.
/// Requires SSE4.1 (implies SSE2/SSE3/SSSE3). Dispatched from [`inter_luma`]
/// when the CPU supports it; the scalar core is the fallback.
#[cfg(target_arch = "x86_64")]
mod sse {
    use core::arch::x86_64::*;

    // ---- loads (all unaligned; dst reads are exactly the valid block) ----
    #[inline(always)] fn loadu128(p: *const u8) -> __m128i {
        unsafe { _mm_loadu_si128(p as *const _) }
    }
    #[inline(always)] fn loadu64(p: *const u8) -> __m128i {
        unsafe { _mm_loadl_epi64(p as *const _) }
    }
    #[inline(always)] fn loadu32(p: *const u8) -> __m128i {
        unsafe {
            // C: (i32x4){*(int32_t *)(p)} — value in lane 0, rest zero.
            _mm_setr_epi32((p as *const u32).read_unaligned() as i32, 0, 0, 0)
        }
    }
    #[inline(always)] fn loadu32x4(
        p0: *const u8,
        p1: *const u8,
        p2: *const u8,
        p3: *const u8,
    ) -> __m128i {
        unsafe {
            _mm_setr_epi32(
                (p0 as *const u32).read_unaligned() as i32,
                (p1 as *const u32).read_unaligned() as i32,
                (p2 as *const u32).read_unaligned() as i32,
                (p3 as *const u32).read_unaligned() as i32,
            )
        }
    }
    #[inline(always)] fn loadu64x2(p0: *const u8, p1: *const u8) -> __m128i {
        unsafe {
            _mm_set_epi64x(
                (p1 as *const i64).read_unaligned(),
                (p0 as *const i64).read_unaligned(),
            )
        }
    }

    // ---- byte moves / shuffles (semantics verified vs C on this machine) ----
    // C shrd128(l, h, i) = _mm_alignr_epi8(h, l, i): 16 bytes from offset i of
    // [l | h] — first arg is the LOW half. Runtime shift via pshufb (robust;
    // matches alignr for i in 0..16).
    #[inline(always)] fn shrd128(l: __m128i, h: __m128i, i: i32) -> __m128i {
        unsafe {
            let base = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            let ml = _mm_add_epi8(base, _mm_set1_epi8(i as i8)); // i + j
            let mh = _mm_sub_epi8(ml, _mm_set1_epi8(16)); // i + j - 16
            let sel_l = _mm_cmplt_epi8(ml, _mm_set1_epi8(16)); // 0xFF where i+j < 16
            let from_l = _mm_shuffle_epi8(l, ml);
            let from_h = _mm_shuffle_epi8(h, mh);
            _mm_blendv_epi8(from_h, from_l, sel_l)
        }
    }
    #[inline(always)] fn shr128<const N: i32>(a: __m128i) -> __m128i {
        unsafe { _mm_srli_si128::<N>(a) }
    }
    #[inline(always)] fn ziplo8(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_unpacklo_epi8(a, b) }
    }
    #[inline(always)] fn ziphi8(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_unpackhi_epi8(a, b) }
    }
    #[inline(always)] fn ziplo16(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_unpacklo_epi16(a, b) }
    }
    #[inline(always)] fn ziphi16(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_unpackhi_epi16(a, b) }
    }
    #[inline(always)] fn ziplo32(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_unpacklo_epi32(a, b) }
    }
    #[inline(always)] fn ziphi32(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_unpackhi_epi32(a, b) }
    }
    #[inline(always)] fn ziplo64(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_unpacklo_epi64(a, b) }
    }
    #[inline(always)] fn ziphi64(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_unpackhi_epi64(a, b) }
    }
    // == C shufps(a,b,0x91): [a[4..8], a[8..12], b[4..8], b[8..12]] (probe-verified).
    #[inline(always)] fn zipmd64(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            _mm_setr_epi32(
                _mm_cvtsi128_si32(_mm_srli_si128::<4>(a)),
                _mm_cvtsi128_si32(_mm_srli_si128::<8>(a)),
                _mm_cvtsi128_si32(_mm_srli_si128::<4>(b)),
                _mm_cvtsi128_si32(_mm_srli_si128::<8>(b)),
            )
        }
    }
    // == C shufps(a,b,0x88): [a[0..4], a[8..12], b[0..4], b[8..12]] (probe-verified).
    #[inline(always)] fn unziplo32(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            _mm_setr_epi32(
                _mm_cvtsi128_si32(a),
                _mm_cvtsi128_si32(_mm_srli_si128::<8>(a)),
                _mm_cvtsi128_si32(b),
                _mm_cvtsi128_si32(_mm_srli_si128::<8>(b)),
            )
        }
    }

    // ---- 6-tap filter constants (i8x16) ----
    const MUL15: __m128i = unsafe {
        core::mem::transmute([1i8, -5, 1, -5, 1, -5, 1, -5, 1, -5, 1, -5, 1, -5, 1, -5])
    };
    const MUL20: __m128i = unsafe { core::mem::transmute([20i8; 16]) };
    const MUL51: __m128i = unsafe {
        core::mem::transmute([-5i8, 1, -5, 1, -5, 1, -5, 1, -5, 1, -5, 1, -5, 1, -5, 1])
    };

    #[inline(always)] fn maddubs(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_maddubs_epi16(a, b) }
    }
    #[inline(always)] fn add16(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_add_epi16(a, b) }
    }

    #[inline(always)] fn sixtap_vlo(
        a: __m128i,
        b: __m128i,
        c: __m128i,
        d: __m128i,
        e: __m128i,
        f: __m128i,
    ) -> __m128i {
        let r = maddubs(ziplo8(a, b), MUL15);
        let r = add16(r, maddubs(ziplo8(c, d), MUL20));
        add16(r, maddubs(ziplo8(e, f), MUL51))
    }
    #[inline(always)] fn sixtap_vhi(
        a: __m128i,
        b: __m128i,
        c: __m128i,
        d: __m128i,
        e: __m128i,
        f: __m128i,
    ) -> __m128i {
        let r = maddubs(ziphi8(a, b), MUL15);
        let r = add16(r, maddubs(ziphi8(c, d), MUL20));
        add16(r, maddubs(ziphi8(e, f), MUL51))
    }
    #[inline(always)] fn sixtap_h4(l0: __m128i, l1: __m128i) -> __m128i {
        let a = ziplo8(l0, shr128::<1>(l0));
        let b = ziplo8(l1, shr128::<1>(l1));
        let r = maddubs(ziplo64(a, b), MUL15);
        let r = add16(r, maddubs(zipmd64(a, b), MUL20));
        add16(r, maddubs(ziphi64(a, b), MUL51))
    }
    #[inline(always)] fn sixtap_h8(a: __m128i) -> __m128i {
        let a1 = shr128::<1>(a);
        let ab = ziplo8(a, a1);
        let ij = ziphi8(a, a1);
        let r = maddubs(ab, MUL15);
        let r = add16(r, maddubs(shrd128(ab, ij, 4), MUL20));
        add16(r, maddubs(shrd128(ab, ij, 8), MUL51))
    }
    // C sixtapHV: ((((a+f)-(b+e))>>2 + ((c+d)-(b+e)))>>2) + (c+d), i16 wrapping.
    #[inline(always)] fn sixtap_hv(
        a: __m128i,
        b: __m128i,
        c: __m128i,
        d: __m128i,
        e: __m128i,
        f: __m128i,
    ) -> __m128i {
        unsafe {
            let af = _mm_add_epi16(a, f);
            let be = _mm_add_epi16(b, e);
            let cd = _mm_add_epi16(c, d);
            let t1 = _mm_srai_epi16::<2>(_mm_sub_epi16(af, be));
            let t2 = _mm_srai_epi16::<2>(_mm_add_epi16(t1, _mm_sub_epi16(cd, be)));
            _mm_add_epi16(t2, cd)
        }
    }
    // C SIXTAPH16 macro: two 16-wide horizontal taps from l0 + next-row lG.
    #[inline(always)] fn sixtap_h16(l0: __m128i, lg: __m128i) -> (__m128i, __m128i) {
        let r0 = maddubs(l0, MUL15);
        let r0 = add16(r0, maddubs(shrd128(l0, lg, 2), MUL20));
        let r0 = add16(r0, maddubs(shrd128(l0, lg, 4), MUL51));
        let r1 = maddubs(shrd128(l0, lg, 1), MUL15);
        let r1 = add16(r1, maddubs(shrd128(l0, lg, 3), MUL20));
        let r1 = add16(r1, maddubs(shrd128(l0, lg, 5), MUL51));
        (ziplo16(r0, r1), ziphi16(r0, r1))
    }

    // packus16((a+(1<<(i-1)))>>i, (b+(1<<(i-1)))>>i); i is 5 or 6.
    // MUST use the immediate form (_mm_srai_epi16): the vector-count form
    // (_mm_sra_epi16) is broken on this CPU (nonzero counts yield 0), and
    // GCC lowers C's constant `>> i` to the immediate form as well.
    #[inline(always)] fn shrrpus16(a: __m128i, b: __m128i, i: i32) -> __m128i {
        unsafe {
            let (sa, sb) = if i == 5 {
                (
                    _mm_srai_epi16::<5>(_mm_add_epi16(a, _mm_set1_epi16(16))),
                    _mm_srai_epi16::<5>(_mm_add_epi16(b, _mm_set1_epi16(16))),
                )
            } else {
                (
                    _mm_srai_epi16::<6>(_mm_add_epi16(a, _mm_set1_epi16(32))),
                    _mm_srai_epi16::<6>(_mm_add_epi16(b, _mm_set1_epi16(32))),
                )
            };
            _mm_packus_epi16(sa, sb)
        }
    }

    #[inline(always)] fn ifelse_mask(v: __m128i, t: __m128i, f: __m128i) -> __m128i {
        unsafe { _mm_blendv_epi8(f, t, v) }
    }
    #[inline(always)] fn shuffle(a: __m128i, m: __m128i) -> __m128i {
        unsafe { _mm_shuffle_epi8(a, m) }
    }
    // 32-byte gather over [a | b]: where m<16 take a[m], else b[m-16] — matches
    // C shuffle2 = ifelse_mask(16 > m, pshufb(a,m), pshufb(b,m)) (on this machine
    // pshufb wraps, so for m<16 pshufb(b, m-16) == b[m] and for m>=16 pshufb(a,m)
    // == a[m-16]; the select picks the non-wrapping side).
    #[inline(always)] fn shuffle2(a: __m128i, b: __m128i, m: __m128i) -> __m128i {
        unsafe {
            let sa = _mm_shuffle_epi8(a, m);
            let sb = _mm_shuffle_epi8(b, _mm_sub_epi8(m, _mm_set1_epi8(16)));
            _mm_blendv_epi8(sb, sa, _mm_cmplt_epi8(m, _mm_set1_epi8(16)))
        }
    }
    #[inline(always)] fn avgu8(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_avg_epu8(a, b) }
    }

    // C maddshrL(q,p,w,o,wd): weighted-prediction weight/offset/shift. `wd`'s
    // low byte (wod[2]) is the uniform shift count for every lane.
    #[inline(always)] fn maddshr_l(
        q: __m128i,
        p: __m128i,
        w: __m128i,
        o: __m128i,
        wd: __m128i,
    ) -> __m128i {
        unsafe {
            let x0 = _mm_sra_epi16(_mm_adds_epi16(maddubs(ziplo8(q, p), w), o), wd);
            let x1 = _mm_sra_epi16(_mm_adds_epi16(maddubs(ziphi8(q, p), w), o), wd);
            _mm_packus_epi16(x0, x1)
        }
    }

    // ---- stride helpers (n may be negative) ----
    #[inline(always)] fn sadd(p: *const u8, n: i32, sstride: usize) -> *const u8 {
        unsafe { p.offset((n as isize) * (sstride as isize)) }
    }
    #[inline(always)] fn daddr(p: *mut u8, n: i32, dstride: usize) -> *mut u8 {
        unsafe { p.offset((n as isize) * (dstride as isize)) }
    }

    // ---- output stores ----
    // 4 rows x 4 bytes (one __m128i of maddshr_l output).
    #[inline(always)] fn store4(d: *mut u8, dstride: usize, r: __m128i) {
        unsafe {
            (d as *mut i32).write_unaligned(_mm_cvtsi128_si32(r));
            (d.add(dstride) as *mut i32).write_unaligned(_mm_cvtsi128_si32(_mm_srli_si128::<4>(r)));
            (d.add(dstride * 2) as *mut i32).write_unaligned(_mm_cvtsi128_si32(_mm_srli_si128::<8>(r)));
            (d.add(dstride * 3) as *mut i32).write_unaligned(_mm_cvtsi128_si32(_mm_srli_si128::<12>(r)));
        }
    }
    // 2 rows x 8 bytes.
    #[inline(always)] fn store2x64(d: *mut u8, dstride: usize, r: __m128i) {
        unsafe {
            (d as *mut i64).write_unaligned(_mm_cvtsi128_si64(r));
            (d.add(dstride) as *mut i64).write_unaligned(_mm_cvtsi128_si64(_mm_srli_si128::<8>(r)));
        }
    }
    // 1 row x 16 bytes.
    #[inline(always)] fn store16(d: *mut u8, r: __m128i) {
        unsafe { _mm_storeu_si128(d as *mut __m128i, r); }
    }

    // ---- C decode_inter_chroma (edge264_inter.c L977) helpers ----
    // C shrrpu16(a, b, 6) = packus(avg_epu16(a>>5, 0), avg_epu16(b>>5, 0)):
    // round(x/64) per non-negative lane.
    #[inline(always)] fn shrrp16(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let zero = _mm_setzero_si128();
            _mm_packus_epi16(
                _mm_avg_epu16(_mm_srli_epi16(a, 5), zero),
                _mm_avg_epu16(_mm_srli_epi16(b, 5), zero),
            )
        }
    }
    // C maddABCD(ab, cd, shuf, AB, CD) = pmaddubsw(pshufb(ab, shuf), AB)
    // + pmaddubsw(pshufb(cd, shuf), CD): x[i] = sab[2i]*A + sab[2i+1]*B
    // + scd[2i]*C + scd[2i+1]*D (2D bilinear weights, sum 64).
    #[inline(always)] fn madd_abcd(
        ab: __m128i,
        cd: __m128i,
        shuf: __m128i,
        ab_w: __m128i,
        cd_w: __m128i,
    ) -> __m128i {
        add16(maddubs(shuffle(ab, shuf), ab_w), maddubs(shuffle(cd, shuf), cd_w))
    }
    // C loada64x2(p0, p1): two 8-byte dst reads (w==16 chroma q).
    #[inline(always)] fn loadq64x2(p0: *const u8, p1: *const u8) -> __m128i {
        unsafe {
            _mm_set_epi64x(
                (p1 as *const i64).read_unaligned(),
                (p0 as *const i64).read_unaligned(),
            )
        }
    }
    // C w==4 dst read: i16x8 q = {*(int16_t *)DADDR(dst, 0..3)} — the four rows
    // are packed DENSELY into the low 8 bytes (i16 lanes 0..3); maddshrC4 only
    // consumes ziplo8 (low 8 bytes), so the high lanes may be zero.
    #[inline(always)] fn load4x16(d: *const u8, dstride: usize) -> __m128i {
        unsafe {
            _mm_setr_epi16(
                (d as *const i16).read_unaligned(),
                (d.add(dstride) as *const i16).read_unaligned(),
                (d.add(dstride * 2) as *const i16).read_unaligned(),
                (d.add(dstride * 3) as *const i16).read_unaligned(),
                0,
                0,
                0,
                0,
            )
        }
    }
    // C w==4 dst write: *(int16_t *)DADDR(dst, i) = v[i] — two pixels per
    // row (v's i16 lanes hold pixel pairs from packus(a, a)).
    #[inline(always)] fn store4x16(d: *mut u8, dstride: usize, r: __m128i) {
        unsafe {
            (d as *mut i16).write_unaligned(_mm_extract_epi16(r, 0) as i16);
            (d.add(dstride) as *mut i16).write_unaligned(_mm_extract_epi16(r, 1) as i16);
            (d.add(dstride * 2) as *mut i16).write_unaligned(_mm_extract_epi16(r, 2) as i16);
            (d.add(dstride * 3) as *mut i16).write_unaligned(_mm_extract_epi16(r, 3) as i16);
        }
    }

    /// SSE port of C `decode_inter_chroma` (edge264_inter.c L977), bit-exact.
    /// `src`/`dst` rows are interleaved Cb/Cr (row 2k = Cb k, row 2k+1 = Cr
    /// k). The w/h are luma block dimensions; chroma is w/2 x h/2 per plane.
    #[allow(clippy::too_many_arguments)] // mirrors C `decode_inter_chroma` arity
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn inter_chroma_sse(
        src: *const u8,
        dst: *mut u8,
        w: usize,
        h: usize,
        x_frac: u32,
        y_frac: u32,
        sstride: usize,
        dstride: usize,
        wod: &[i16; 8],
    ) {
        let xf = (x_frac & 7) as i32;
        let yf = (y_frac & 7) as i32;
        // C packs the 2D bilinear weights into one dword (ABCD): A=(8-xF)
        // (8-yF), B=xF(8-yF), C=(8-xF)yF, D=xF*yF — all <= 64, sum 64.
        let a = ((8 - xf) * (8 - yf)) as i8;
        let b = (xf * (8 - yf)) as i8;
        let c = ((8 - xf) * yf) as i8;
        let d = (xf * yf) as i8;
        let ab_w = _mm_setr_epi8(a, b, a, b, a, b, a, b, a, b, a, b, a, b, a, b);
        let cd_w = _mm_setr_epi8(c, d, c, d, c, d, c, d, c, d, c, d, c, d, c, d);
        // C wo8 = ziphi16(wod, wod): {wCb, wCr, oCb, oCr} i16 lanes doubled.
        let wo8 = _mm_setr_epi8(
            (wod[4] & 0xFF) as i8,
            (wod[4] >> 8) as i8,
            (wod[4] & 0xFF) as i8,
            (wod[4] >> 8) as i8,
            (wod[5] & 0xFF) as i8,
            (wod[5] >> 8) as i8,
            (wod[5] & 0xFF) as i8,
            (wod[5] >> 8) as i8,
            (wod[6] & 0xFF) as i8,
            (wod[6] >> 8) as i8,
            (wod[6] & 0xFF) as i8,
            (wod[6] >> 8) as i8,
            (wod[7] & 0xFF) as i8,
            (wod[7] >> 8) as i8,
            (wod[7] & 0xFF) as i8,
            (wod[7] >> 8) as i8,
        );
        // C wd = (u64x2)wod >> 48: per-lane u64 shift -> {wod[3], wod[7]}.
        // This machine's PSRAW shifts every lane by the low i64 count
        // (= wod[3], the chroma log2 weight denom), matching C exactly.
        let wd = _mm_set_epi64x(wod[7] as i64, wod[3] as i64);

        if w == 16 {
            // chroma 8 wide; h even.
            let shuf = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
            let wcb = _mm_shuffle_epi32::<0x00>(wo8);
            let wcr = _mm_shuffle_epi32::<0x55>(wo8);
            let ocb = _mm_shuffle_epi32::<0xAA>(wo8);
            let ocr = _mm_shuffle_epi32::<0xFF>(wo8);
            let mut l0 = loadu128(src);
            let mut l1 = loadu128(sadd(src, 1, sstride));
            let mut d = dst;
            let mut s2 = sadd(src, 2, sstride);
            let mut rem = h;
            while rem > 0 {
                let l2 = loadu128(s2);
                let l3 = loadu128(sadd(s2, 1, sstride));
                let x0 = madd_abcd(l0, l2, shuf, ab_w, cd_w);
                let x1 = madd_abcd(l1, l3, shuf, ab_w, cd_w);
                let p = shrrp16(x0, x1);
                // q = [8 Cb dst bytes | 8 Cr dst bytes]
                let q = loadq64x2(d, daddr(d, 1, dstride));
                // C maddshrC16: per-half pmaddubsw + offset + PSRAW(wd).
                let xc = _mm_sra_epi16(_mm_adds_epi16(maddubs(ziplo8(q, p), wcb), ocb), wd);
                let xr = _mm_sra_epi16(_mm_adds_epi16(maddubs(ziphi8(q, p), wcr), ocr), wd);
                store2x64(d, dstride, _mm_packus_epi16(xc, xr));
                l0 = l2;
                l1 = l3;
                s2 = sadd(s2, 2, sstride);
                d = daddr(d, 2, dstride);
                rem -= 2;
            }
        } else if w == 8 {
            // chroma 4 wide; h a multiple of 4.
            let shuf = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 8, 9, 9, 10, 10, 11, 11, 12);
            let w0 = ziplo32(wo8, wo8);
            let o = ziphi32(wo8, wo8);
            let mut l0 = loadu64x2(src, sadd(src, 1, sstride));
            let mut d = dst;
            let mut s2 = sadd(src, 2, sstride);
            let mut rem = h;
            while rem > 0 {
                let l1 = loadu64x2(s2, sadd(s2, 1, sstride));
                let l2 = loadu64x2(sadd(s2, 2, sstride), sadd(s2, 3, sstride));
                let x0 = madd_abcd(l0, l1, shuf, ab_w, cd_w);
                let x1 = madd_abcd(l1, l2, shuf, ab_w, cd_w);
                let p = shrrp16(x0, x1);
                // q = 4 rows x 4 dst bytes (Cb k, Cr k, Cb k+1, Cr k+1)
                let q = loadu32x4(d, daddr(d, 1, dstride), daddr(d, 2, dstride), daddr(d, 3, dstride));
                // C maddshrC8 (= maddshrL): shared w0/o for both halves.
                let xl = _mm_sra_epi16(_mm_adds_epi16(maddubs(ziplo8(q, p), w0), o), wd);
                let xh = _mm_sra_epi16(_mm_adds_epi16(maddubs(ziphi8(q, p), w0), o), wd);
                store4(d, dstride, _mm_packus_epi16(xl, xh));
                l0 = l2;
                s2 = sadd(s2, 4, sstride);
                d = daddr(d, 4, dstride);
                rem -= 4;
            }
        } else {
            // w == 4: chroma 2 wide; h a multiple of 4.
            let shuf = _mm_setr_epi8(0, 1, 1, 2, 4, 5, 5, 6, 8, 9, 9, 10, 12, 13, 13, 14);
            let w0 = _mm_shuffle_epi32::<0x44>(wo8); // C broadcast64(wo8, 0)
            let o = ziphi64(wo8, wo8);
            let mut l0 = ziplo32(loadu32(src), loadu32(sadd(src, 1, sstride)));
            let mut d = dst;
            let mut s2 = sadd(src, 2, sstride);
            let mut rem = h;
            while rem > 0 {
                let l1 = loadu32x4(
                    s2,
                    sadd(s2, 1, sstride),
                    sadd(s2, 2, sstride),
                    sadd(s2, 3, sstride),
                );
                let x0 = madd_abcd(ziplo64(l0, l1), l1, shuf, ab_w, cd_w);
                let p = shrrp16(x0, _mm_setzero_si128());
                let q = load4x16(d, dstride);
                // C maddshrC4: packus(a, a) — the i16 lanes hold pixel pairs.
                let a = _mm_sra_epi16(_mm_adds_epi16(maddubs(ziplo8(q, p), w0), o), wd);
                store4x16(d, dstride, _mm_packus_epi16(a, a));
                l0 = shr128::<8>(l1);
                s2 = sadd(s2, 4, sstride);
                d = daddr(d, 4, dstride);
                rem -= 4;
            }
        }
    }

    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn inter_luma_sse(
        src2: *const u8,
        dst: *mut u8,
        h: usize,
        mode: u32,
        sstride: usize,
        dstride: usize,
        wod: &[i16; 8],
    ) {
        let wod_v = unsafe { _mm_loadu_si128(wod.as_ptr() as *const __m128i) };
        // broadcast16(wod,0): all i16 lanes = wod[0]; broadcast16(wod,1) = wod[1].
        // Register-only intrinsics are safe here: this fn carries
        // #[target_feature] and is dispatched behind an sse4.1 CPU check.
        let w0 = _mm_shuffle_epi32::<0>(_mm_shufflelo_epi16::<0>(wod_v));
        let wd = _mm_set_epi64x(wod[6] as i64, wod[2] as i64);
        let o = _mm_shuffle_epi32::<0>(_mm_shufflelo_epi16::<5>(wod_v));
        let m0 = _mm_set1_epi8((-(0xd888 >> (mode & 15) & 1)) as i8);
        let m1 = _mm_set1_epi8((-(0xa504 >> (mode & 15) & 1)) as i8);
        let shufx_base = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);
        let shufx = _mm_sub_epi8(shufx_base, m0);
        let src0 = src2.wrapping_sub(2);

        match mode {
            // ================= 4xH (mode 0..15) =================
            0 => {
                let mut s = src2;
                let mut d = dst;
                let mut rem = h;
                loop {
                    let p = loadu32x4(
                        sadd(s, 0, sstride),
                        sadd(s, 1, sstride),
                        sadd(s, 2, sstride),
                        sadd(s, 3, sstride),
                    );
                    let q = loadu32x4(
                        daddr(d, 0, dstride) as *const u8,
                        daddr(d, 1, dstride) as *const u8,
                        daddr(d, 2, dstride) as *const u8,
                        daddr(d, 3, dstride) as *const u8,
                    );
                    store4(d, dstride, maddshr_l(q, p, w0, o, wd));
                    d = daddr(d, 4, dstride);
                    s = sadd(s, 4, sstride);
                    rem -= 4;
                    if rem == 0 {
                        break;
                    }
                }
            }
            1..=3 => {
                let mut s = src0;
                let mut d = dst;
                let mut rem = h;
                loop {
                    let l2 = loadu128(sadd(s, 0, sstride));
                    let l3 = loadu128(sadd(s, 1, sstride));
                    let l4 = loadu128(sadd(s, 2, sstride));
                    let l5 = loadu128(sadd(s, 3, sstride));
                    let h01 = shrrpus16(sixtap_h4(l2, l3), sixtap_h4(l4, l5), 5);
                    let s0 = shuffle(ziplo64(l2, l3), shufx);
                    let s1 = shuffle(ziplo64(l4, l5), shufx);
                    let sv = unziplo32(s0, s1);
                    let q = loadu32x4(
                        daddr(d, 0, dstride) as *const u8,
                        daddr(d, 1, dstride) as *const u8,
                        daddr(d, 2, dstride) as *const u8,
                        daddr(d, 3, dstride) as *const u8,
                    );
                    store4(
                        d,
                        dstride,
                        maddshr_l(q, avgu8(ifelse_mask(m1, h01, sv), h01), w0, o, wd),
                    );
                    d = daddr(d, 4, dstride);
                    s = sadd(s, 4, sstride);
                    rem -= 4;
                    if rem == 0 {
                        break;
                    }
                }
            }
            4 | 8 | 12 => {
                let mut s = src2;
                let mut m02 = loadu32x4(
                    sadd(s, -2, sstride),
                    sadd(s, -1, sstride),
                    sadd(s, 0, sstride),
                    sadd(s, 1, sstride),
                );
                let mut m12 = shrd128(m02, loadu32(sadd(s, 2, sstride)), 4);
                let mut d = dst;
                let mut rem = h;
                loop {
                    s = sadd(s, 4, sstride);
                    let m52 = loadu32x4(
                        sadd(s, -1, sstride),
                        sadd(s, 0, sstride),
                        sadd(s, 1, sstride),
                        sadd(s, 2, sstride),
                    );
                    let m22 = shrd128(m12, m52, 4);
                    let m32 = shrd128(m12, m52, 8);
                    let m42 = shrd128(m12, m52, 12);
                    let v0 = sixtap_vlo(m02, m12, m22, m32, m42, m52);
                    let v1 = sixtap_vhi(m02, m12, m22, m32, m42, m52);
                    let v01 = shrrpus16(v0, v1, 5);
                    let sv = ifelse_mask(m1, v01, ifelse_mask(m0, m32, m22));
                    let q = loadu32x4(
                        daddr(d, 0, dstride) as *const u8,
                        daddr(d, 1, dstride) as *const u8,
                        daddr(d, 2, dstride) as *const u8,
                        daddr(d, 3, dstride) as *const u8,
                    );
                    store4(d, dstride, maddshr_l(q, avgu8(sv, v01), w0, o, wd));
                    m02 = m42;
                    m12 = m52;
                    d = daddr(d, 4, dstride);
                    rem -= 4;
                    if rem == 0 {
                        break;
                    }
                }
            }
            5 | 7 | 13 | 15 => {
                let mut s = src0;
                let mut l0 = loadu128(sadd(s, -2, sstride));
                let mut l1 = loadu128(sadd(s, -1, sstride));
                let mut l2 = loadu128(sadd(s, 0, sstride));
                let mut l3 = loadu128(sadd(s, 1, sstride));
                let mut l4 = loadu128(sadd(s, 2, sstride));
                let mut d = dst;
                let mut rem = h;
                loop {
                    s = sadd(s, 4, sstride);
                    let l5 = loadu128(sadd(s, -1, sstride));
                    let l6 = loadu128(sadd(s, 0, sstride));
                    let l7 = loadu128(sadd(s, 1, sstride));
                    let l8 = loadu128(sadd(s, 2, sstride));
                    let l02 = shuffle(l0, shufx);
                    let l12 = shuffle(l1, shufx);
                    let l22 = shuffle(l2, shufx);
                    let l32 = shuffle(l3, shufx);
                    let l42 = shuffle(l4, shufx);
                    let l52 = shuffle(l5, shufx);
                    let l62 = shuffle(l6, shufx);
                    let l72 = shuffle(l7, shufx);
                    let l82 = shuffle(l8, shufx);
                    let m02 = ziplo32(l02, l12);
                    let m12 = ziplo32(l12, l22);
                    let m22 = ziplo32(l22, l32);
                    let m32 = ziplo32(l32, l42);
                    let m42 = ziplo32(l42, l52);
                    let m52 = ziplo32(l52, l62);
                    let m62 = ziplo32(l62, l72);
                    let m72 = ziplo32(l72, l82);
                    let v0 = sixtap_vlo(m02, m12, m22, m32, m42, m52);
                    let v1 = sixtap_vlo(m22, m32, m42, m52, m62, m72);
                    let v01 = shrrpus16(v0, v1, 5);
                    let h0 = sixtap_h4(ifelse_mask(m1, l3, l2), ifelse_mask(m1, l4, l3));
                    let h1 = sixtap_h4(ifelse_mask(m1, l5, l4), ifelse_mask(m1, l6, l5));
                    let sv = avgu8(v01, shrrpus16(h0, h1, 5));
                    let q = loadu32x4(
                        daddr(d, 0, dstride) as *const u8,
                        daddr(d, 1, dstride) as *const u8,
                        daddr(d, 2, dstride) as *const u8,
                        daddr(d, 3, dstride) as *const u8,
                    );
                    store4(d, dstride, maddshr_l(q, sv, w0, o, wd));
                    l0 = l4;
                    l1 = l5;
                    l2 = l6;
                    l3 = l7;
                    l4 = l8;
                    d = daddr(d, 4, dstride);
                    rem -= 4;
                    if rem == 0 {
                        break;
                    }
                }
            }
            9 | 11 => {
                let mut s = src0;
                let mut l0 = loadu128(sadd(s, -2, sstride));
                let mut l1 = loadu128(sadd(s, -1, sstride));
                let mut l2 = loadu128(sadd(s, 0, sstride));
                let mut l3 = loadu128(sadd(s, 1, sstride));
                let mut l4 = loadu128(sadd(s, 2, sstride));
                let mut d = dst;
                let mut rem = h;
                loop {
                    s = sadd(s, 4, sstride);
                    let l5 = loadu128(sadd(s, -1, sstride));
                    let l6 = loadu128(sadd(s, 0, sstride));
                    let l7 = loadu128(sadd(s, 1, sstride));
                    let l8 = loadu128(sadd(s, 2, sstride));
                    let r0 = ziplo16(ziphi8(l0, l1), ziphi8(l2, l3));
                    let r1 = ziplo16(ziphi8(l4, l5), ziphi8(l6, l7));
                    let r2 = _mm_set_epi64x(
                        _mm_cvtsi128_si64(_mm_srli_si128::<8>(l8)),
                        _mm_cvtsi128_si64(ziplo32(r0, r1)),
                    );
                    let v08 = sixtap_h8(r2);
                    let v00 = sixtap_vlo(l0, l1, l2, l3, l4, l5);
                    let v10 = sixtap_vlo(l1, l2, l3, l4, l5, l6);
                    let v20 = sixtap_vlo(l2, l3, l4, l5, l6, l7);
                    let v30 = sixtap_vlo(l3, l4, l5, l6, l7, l8);
                    let v01 = shrd128(v00, v08, 2);
                    let v11 = shrd128(v10, shr128::<2>(v08), 2);
                    let v21 = shrd128(v20, shr128::<4>(v08), 2);
                    let v31 = shrd128(v30, shr128::<6>(v08), 2);
                    let m00 = ziplo64(v00, v10);
                    let m01 = ziplo64(v01, v11);
                    let m02 = zipmd64(v00, v10);
                    let m03 = zipmd64(v01, v11);
                    let m04 = ziphi64(v00, v10);
                    let m05 = ziphi64(v01, v11);
                    let m20 = ziplo64(v20, v30);
                    let m21 = ziplo64(v21, v31);
                    let m22 = zipmd64(v20, v30);
                    let m23 = zipmd64(v21, v31);
                    let m24 = ziphi64(v20, v30);
                    let m25 = ziphi64(v21, v31);
                    let vh0 = sixtap_hv(m00, m01, m02, m03, m04, m05);
                    let vh1 = sixtap_hv(m20, m21, m22, m23, m24, m25);
                    let vh = shrrpus16(vh0, vh1, 6);
                    let sv = shrrpus16(
                        ifelse_mask(m0, m03, m02),
                        ifelse_mask(m0, m23, m22),
                        5,
                    );
                    let q = loadu32x4(
                        daddr(d, 0, dstride) as *const u8,
                        daddr(d, 1, dstride) as *const u8,
                        daddr(d, 2, dstride) as *const u8,
                        daddr(d, 3, dstride) as *const u8,
                    );
                    store4(d, dstride, maddshr_l(q, avgu8(sv, vh), w0, o, wd));
                    l0 = l4;
                    l1 = l5;
                    l2 = l6;
                    l3 = l7;
                    l4 = l8;
                    d = daddr(d, 4, dstride);
                    rem -= 4;
                    if rem == 0 {
                        break;
                    }
                }
            }
            6 | 10 | 14 => {
                let mut s = src0;
                let l0 = loadu128(sadd(s, -2, sstride));
                let l1 = loadu128(sadd(s, -1, sstride));
                let l2 = loadu128(sadd(s, 0, sstride));
                let l3 = loadu128(sadd(s, 1, sstride));
                let l4 = loadu128(sadd(s, 2, sstride));
                let mut h0 = sixtap_h4(l0, l1);
                let mut h2 = sixtap_h4(l2, l3);
                let mut h3 = sixtap_h4(l3, l4);
                let mut d = dst;
                let mut rem = h;
                loop {
                    s = sadd(s, 4, sstride);
                    let l5 = loadu128(sadd(s, -1, sstride));
                    let l6 = loadu128(sadd(s, 0, sstride));
                    let l7 = loadu128(sadd(s, 1, sstride));
                    let l8 = loadu128(sadd(s, 2, sstride));
                    let h5 = sixtap_h4(l5, l6);
                    let h7 = sixtap_h4(l7, l8);
                    let h1 = shrd128(h0, h2, 8);
                    let h4 = shrd128(h3, h5, 8);
                    let h6 = shrd128(h5, h7, 8);
                    let hv0 = sixtap_hv(h0, h1, h2, h3, h4, h5);
                    let hv1 = sixtap_hv(h2, h3, h4, h5, h6, h7);
                    let hv = shrrpus16(hv0, hv1, 6);
                    let sv = shrrpus16(
                        ifelse_mask(m0, h3, h2),
                        ifelse_mask(m0, h5, h4),
                        5,
                    );
                    let q = loadu32x4(
                        daddr(d, 0, dstride) as *const u8,
                        daddr(d, 1, dstride) as *const u8,
                        daddr(d, 2, dstride) as *const u8,
                        daddr(d, 3, dstride) as *const u8,
                    );
                    store4(
                        d,
                        dstride,
                        maddshr_l(q, avgu8(ifelse_mask(m1, hv, sv), hv), w0, o, wd),
                    );
                    h0 = h4;
                    h2 = h6;
                    h3 = h7;
                    d = daddr(d, 4, dstride);
                    rem -= 4;
                    if rem == 0 {
                        break;
                    }
                }
            }

            // ================= 8xH (mode 16..31) =================
            16 => {
                let mut s = src2;
                let mut d = dst;
                let mut rem = h;
                loop {
                    let p0 = loadu64x2(sadd(s, 0, sstride), sadd(s, 1, sstride));
                    let p1 = loadu64x2(sadd(s, 2, sstride), sadd(s, 3, sstride));
                    let q0 = loadu64x2(
                        daddr(d, 0, dstride) as *const u8,
                        daddr(d, 1, dstride) as *const u8,
                    );
                    let q1 = loadu64x2(
                        daddr(d, 2, dstride) as *const u8,
                        daddr(d, 3, dstride) as *const u8,
                    );
                    let r0 = maddshr_l(q0, p0, w0, o, wd);
                    let r1 = maddshr_l(q1, p1, w0, o, wd);
                    store2x64(d, dstride, r0);
                    store2x64(daddr(d, 2, dstride), dstride, r1);
                    s = sadd(s, 4, sstride);
                    d = daddr(d, 4, dstride);
                    rem -= 4;
                    if rem == 0 {
                        break;
                    }
                }
            }
            17..=19 => {
                let mut s = src0;
                let mut d = dst;
                let mut rem = h;
                loop {
                    let l0 = loadu128(sadd(s, 0, sstride));
                    let l1 = loadu128(sadd(s, 1, sstride));
                    let h01 = shrrpus16(sixtap_h8(l0), sixtap_h8(l1), 5);
                    let sv = ziplo64(shuffle(l0, shufx), shuffle(l1, shufx));
                    let q = loadu64x2(
                        daddr(d, 0, dstride) as *const u8,
                        daddr(d, 1, dstride) as *const u8,
                    );
                    store2x64(
                        d,
                        dstride,
                        maddshr_l(q, avgu8(ifelse_mask(m1, h01, sv), h01), w0, o, wd),
                    );
                    s = sadd(s, 2, sstride);
                    d = daddr(d, 2, dstride);
                    rem -= 2;
                    if rem == 0 {
                        break;
                    }
                }
            }
            20 | 24 | 28 => {
                let mut s = src2;
                let mut l0 = loadu64(sadd(s, -2, sstride));
                let mut l1 = loadu64(sadd(s, -1, sstride));
                let mut l2 = loadu64(sadd(s, 0, sstride));
                let mut l3 = loadu64(sadd(s, 1, sstride));
                let mut l4 = loadu64(sadd(s, 2, sstride));
                let mut d = dst;
                let mut rem = h;
                loop {
                    s = sadd(s, 2, sstride);
                    let l5 = loadu64(sadd(s, 1, sstride));
                    let l6 = loadu64(sadd(s, 2, sstride));
                    let v0 = sixtap_vlo(l0, l1, l2, l3, l4, l5);
                    let v1 = sixtap_vlo(l1, l2, l3, l4, l5, l6);
                    let v01 = shrrpus16(v0, v1, 5);
                    let sv = ifelse_mask(m0, ziplo64(l3, l4), ziplo64(l2, l3));
                    let q = loadu64x2(
                        daddr(d, 0, dstride) as *const u8,
                        daddr(d, 1, dstride) as *const u8,
                    );
                    store2x64(
                        d,
                        dstride,
                        maddshr_l(q, avgu8(ifelse_mask(m1, v01, sv), v01), w0, o, wd),
                    );
                    l0 = l2;
                    l1 = l3;
                    l2 = l4;
                    l3 = l5;
                    l4 = l6;
                    d = daddr(d, 2, dstride);
                    rem -= 2;
                    if rem == 0 {
                        break;
                    }
                }
            }
            21 | 23 | 29 | 31 => {
                let mut s = src0;
                let mut l02 = shuffle(loadu128(sadd(s, -2, sstride)), shufx);
                let mut l12 = shuffle(loadu128(sadd(s, -1, sstride)), shufx);
                let mut l2 = loadu128(sadd(s, 0, sstride));
                let mut l3 = loadu128(sadd(s, 1, sstride));
                let mut l4 = loadu128(sadd(s, 2, sstride));
                let mut d = dst;
                let mut rem = h;
                loop {
                    s = sadd(s, 2, sstride);
                    let l5 = loadu128(sadd(s, 1, sstride));
                    let l6 = loadu128(sadd(s, 2, sstride));
                    let l22 = shuffle(l2, shufx);
                    let l32 = shuffle(l3, shufx);
                    let l42 = shuffle(l4, shufx);
                    let l52 = shuffle(l5, shufx);
                    let l62 = shuffle(l6, shufx);
                    let v0 = sixtap_vlo(l02, l12, l22, l32, l42, l52);
                    let v1 = sixtap_vlo(l12, l22, l32, l42, l52, l62);
                    let v01 = shrrpus16(v0, v1, 5);
                    let h0 = sixtap_h8(ifelse_mask(m1, l3, l2));
                    let h1 = sixtap_h8(ifelse_mask(m1, l4, l3));
                    let sv = avgu8(v01, shrrpus16(h0, h1, 5));
                    let q = loadu64x2(
                        daddr(d, 0, dstride) as *const u8,
                        daddr(d, 1, dstride) as *const u8,
                    );
                    store2x64(d, dstride, maddshr_l(q, sv, w0, o, wd));
                    l02 = l22;
                    l12 = l32;
                    l2 = l4;
                    l3 = l5;
                    l4 = l6;
                    d = daddr(d, 2, dstride);
                    rem -= 2;
                    if rem == 0 {
                        break;
                    }
                }
            }
            25 | 27 => {
                let mut s = src0;
                let mut l0 = loadu128(sadd(s, -2, sstride));
                let mut l1 = loadu128(sadd(s, -1, sstride));
                let mut l2 = loadu128(sadd(s, 0, sstride));
                let mut l3 = loadu128(sadd(s, 1, sstride));
                let mut l4 = loadu128(sadd(s, 2, sstride));
                let mut d = dst;
                let mut rem = h;
                loop {
                    s = sadd(s, 2, sstride);
                    let l5 = loadu128(sadd(s, 1, sstride));
                    let l6 = loadu128(sadd(s, 2, sstride));
                    let x00 = sixtap_vlo(l0, l1, l2, l3, l4, l5);
                    let x10 = sixtap_vlo(l1, l2, l3, l4, l5, l6);
                    let x08 = sixtap_vhi(l0, l1, l2, l3, l4, l5);
                    let x18 = sixtap_vhi(l1, l2, l3, l4, l5, l6);
                    let x01 = shrd128(x00, x08, 2);
                    let x11 = shrd128(x10, x18, 2);
                    let x02 = shrd128(x00, x08, 4);
                    let x12 = shrd128(x10, x18, 4);
                    let x03 = shrd128(x00, x08, 6);
                    let x13 = shrd128(x10, x18, 6);
                    let x04 = shrd128(x00, x08, 8);
                    let x14 = shrd128(x10, x18, 8);
                    let x05 = shrd128(x00, x08, 10);
                    let x15 = shrd128(x10, x18, 10);
                    let vh0 = sixtap_hv(x00, x01, x02, x03, x04, x05);
                    let vh1 = sixtap_hv(x10, x11, x12, x13, x14, x15);
                    let vh = shrrpus16(vh0, vh1, 6);
                    let sv = shrrpus16(
                        ifelse_mask(m0, x03, x02),
                        ifelse_mask(m0, x13, x12),
                        5,
                    );
                    let q = loadu64x2(
                        daddr(d, 0, dstride) as *const u8,
                        daddr(d, 1, dstride) as *const u8,
                    );
                    store2x64(d, dstride, maddshr_l(q, avgu8(vh, sv), w0, o, wd));
                    l0 = l2;
                    l1 = l3;
                    l2 = l4;
                    l3 = l5;
                    l4 = l6;
                    d = daddr(d, 2, dstride);
                    rem -= 2;
                    if rem == 0 {
                        break;
                    }
                }
            }
            22 | 26 | 30 => {
                let mut s = src0;
                let mut v0 = sixtap_h8(loadu128(sadd(s, -2, sstride)));
                let mut v1 = sixtap_h8(loadu128(sadd(s, -1, sstride)));
                let mut v2 = sixtap_h8(loadu128(sadd(s, 0, sstride)));
                let mut v3 = sixtap_h8(loadu128(sadd(s, 1, sstride)));
                let mut v4 = sixtap_h8(loadu128(sadd(s, 2, sstride)));
                let mut d = dst;
                let mut rem = h;
                loop {
                    s = sadd(s, 2, sstride);
                    let v5 = sixtap_h8(loadu128(sadd(s, 1, sstride)));
                    let v6 = sixtap_h8(loadu128(sadd(s, 2, sstride)));
                    let hv0 = sixtap_hv(v0, v1, v2, v3, v4, v5);
                    let hv1 = sixtap_hv(v1, v2, v3, v4, v5, v6);
                    let hv = shrrpus16(hv0, hv1, 6);
                    let sv = shrrpus16(
                        ifelse_mask(m0, v3, v2),
                        ifelse_mask(m0, v4, v3),
                        5,
                    );
                    let q = loadu64x2(
                        daddr(d, 0, dstride) as *const u8,
                        daddr(d, 1, dstride) as *const u8,
                    );
                    store2x64(
                        d,
                        dstride,
                        maddshr_l(q, avgu8(ifelse_mask(m1, hv, sv), hv), w0, o, wd),
                    );
                    v0 = v2;
                    v1 = v3;
                    v2 = v4;
                    v3 = v5;
                    v4 = v6;
                    d = daddr(d, 2, dstride);
                    rem -= 2;
                    if rem == 0 {
                        break;
                    }
                }
            }

            // ================= 16xH (mode 32..47) =================
            32 => {
                let mut s = src2;
                let mut d = dst;
                let mut rem = h;
                loop {
                    for k in 0..4 {
                        let q = loadu128(daddr(d, k, dstride));
                        let p = loadu128(sadd(s, k, sstride));
                        store16(daddr(d, k, dstride), maddshr_l(q, p, w0, o, wd));
                    }
                    s = sadd(s, 4, sstride);
                    d = daddr(d, 4, dstride);
                    rem -= 4;
                    if rem == 0 {
                        break;
                    }
                }
            }
            33..=35 => {
                let mut s = src0;
                let mut d = dst;
                let mut rem = h;
                loop {
                    let l0 = loadu128(s);
                    let lg = loadu64(unsafe { s.add(16) });
                    let (h0, h8) = sixtap_h16(l0, lg);
                    let h01 = shrrpus16(h0, h8, 5);
                    let sv = ifelse_mask(m1, h01, shuffle2(l0, lg, shufx));
                    let q = loadu128(d);
                    store16(d, maddshr_l(q, avgu8(sv, h01), w0, o, wd));
                    s = sadd(s, 1, sstride);
                    d = daddr(d, 1, dstride);
                    rem -= 1;
                    if rem == 0 {
                        break;
                    }
                }
            }
            36 | 40 | 44 => {
                let mut s = src2;
                let mut l0 = loadu128(sadd(s, -2, sstride));
                let mut l1 = loadu128(sadd(s, -1, sstride));
                let mut l2 = loadu128(sadd(s, 0, sstride));
                let mut l3 = loadu128(sadd(s, 1, sstride));
                let mut l4 = loadu128(sadd(s, 2, sstride));
                let mut d = dst;
                let mut rem = h;
                loop {
                    s = sadd(s, 1, sstride);
                    let l5 = loadu128(sadd(s, 2, sstride));
                    let v0 = sixtap_vlo(l0, l1, l2, l3, l4, l5);
                    let v8 = sixtap_vhi(l0, l1, l2, l3, l4, l5);
                    let v01 = shrrpus16(v0, v8, 5);
                    let sv = ifelse_mask(m1, v01, ifelse_mask(m0, l3, l2));
                    let q = loadu128(d);
                    store16(d, maddshr_l(q, avgu8(sv, v01), w0, o, wd));
                    l0 = l1;
                    l1 = l2;
                    l2 = l3;
                    l3 = l4;
                    l4 = l5;
                    d = daddr(d, 1, dstride);
                    rem -= 1;
                    if rem == 0 {
                        break;
                    }
                }
            }
            37 | 39 | 45 | 47 => {
                let mut s = src0;
                let mut sg = unsafe { src0.add(16) };
                let mut l02 = shuffle2(
                    loadu128(sadd(s, -2, sstride)),
                    loadu64(sadd(sg, -2, sstride)),
                    shufx,
                );
                let mut l12 = shuffle2(
                    loadu128(sadd(s, -1, sstride)),
                    loadu64(sadd(sg, -1, sstride)),
                    shufx,
                );
                let mut l20 = loadu128(sadd(s, 0, sstride));
                let mut l2g = loadu64(sadd(sg, 0, sstride));
                let mut l30 = loadu128(sadd(s, 1, sstride));
                let mut l3g = loadu64(sadd(sg, 1, sstride));
                let mut l40 = loadu128(sadd(s, 2, sstride));
                let mut l4g = loadu64(sadd(sg, 2, sstride));
                let mut d = dst;
                let mut rem = h;
                loop {
                    s = sadd(s, 1, sstride);
                    sg = sadd(sg, 1, sstride);
                    let l50 = loadu128(sadd(s, 2, sstride));
                    let l5g = loadu64(sadd(sg, 2, sstride));
                    let l22 = shuffle2(l20, l2g, shufx);
                    let l32 = shuffle2(l30, l3g, shufx);
                    let l42 = shuffle2(l40, l4g, shufx);
                    let l52 = shuffle2(l50, l5g, shufx);
                    let v0 = sixtap_vlo(l02, l12, l22, l32, l42, l52);
                    let v1 = sixtap_vhi(l02, l12, l22, l32, l42, l52);
                    let v01 = shrrpus16(v0, v1, 5);
                    let s0 = ifelse_mask(m1, l30, l20);
                    let s1 = ifelse_mask(m1, l3g, l2g);
                    let (h0, h1) = sixtap_h16(s0, s1);
                    let h01 = shrrpus16(h0, h1, 5);
                    let q = loadu128(d);
                    store16(d, maddshr_l(q, avgu8(v01, h01), w0, o, wd));
                    l02 = l12;
                    l12 = l22;
                    let n_l20 = l30;
                    let n_l2g = l3g;
                    let n_l30 = l40;
                    let n_l3g = l4g;
                    let n_l40 = l50;
                    let n_l4g = l5g;
                    l20 = n_l20;
                    l2g = n_l2g;
                    l30 = n_l30;
                    l3g = n_l3g;
                    l40 = n_l40;
                    l4g = n_l4g;
                    d = daddr(d, 1, dstride);
                    rem -= 1;
                    if rem == 0 {
                        break;
                    }
                }
            }
            41 | 43 => {
                let mut s = src0;
                let mut sg = unsafe { src0.add(16) };
                let mut l00 = loadu128(sadd(s, -2, sstride));
                let mut l0g = loadu64(sadd(sg, -2, sstride));
                let mut l10 = loadu128(sadd(s, -1, sstride));
                let mut l1g = loadu64(sadd(sg, -1, sstride));
                let mut l20 = loadu128(sadd(s, 0, sstride));
                let mut l2g = loadu64(sadd(sg, 0, sstride));
                let mut l30 = loadu128(sadd(s, 1, sstride));
                let mut l3g = loadu64(sadd(sg, 1, sstride));
                let mut l40 = loadu128(sadd(s, 2, sstride));
                let mut l4g = loadu64(sadd(sg, 2, sstride));
                let mut d = dst;
                let mut rem = h;
                loop {
                    s = sadd(s, 1, sstride);
                    sg = sadd(sg, 1, sstride);
                    let l50 = loadu128(sadd(s, 2, sstride));
                    let l5g = loadu64(sadd(sg, 2, sstride));
                    let v0 = sixtap_vlo(l00, l10, l20, l30, l40, l50);
                    let v8 = sixtap_vhi(l00, l10, l20, l30, l40, l50);
                    let vg = sixtap_vlo(l0g, l1g, l2g, l3g, l4g, l5g);
                    let v1 = shrd128(v0, v8, 2);
                    let v2 = shrd128(v0, v8, 4);
                    let v3 = shrd128(v0, v8, 6);
                    let v4 = shrd128(v0, v8, 8);
                    let v5 = shrd128(v0, v8, 10);
                    let vh0 = sixtap_hv(v0, v1, v2, v3, v4, v5);
                    let v9 = shrd128(v8, vg, 2);
                    let va = shrd128(v8, vg, 4);
                    let vb = shrd128(v8, vg, 6);
                    let vc = shrd128(v8, vg, 8);
                    let vd = shrd128(v8, vg, 10);
                    let vh1 = sixtap_hv(v8, v9, va, vb, vc, vd);
                    let vh = shrrpus16(vh0, vh1, 6);
                    let sv = shrrpus16(
                        ifelse_mask(m0, v3, v2),
                        ifelse_mask(m0, vb, va),
                        5,
                    );
                    let q = loadu128(d);
                    store16(d, maddshr_l(q, avgu8(sv, vh), w0, o, wd));
                    l00 = l10;
                    l10 = l20;
                    l20 = l30;
                    l30 = l40;
                    l40 = l50;
                    let n_l0g = l1g;
                    let n_l1g = l2g;
                    let n_l2g = l3g;
                    let n_l3g = l4g;
                    let n_l4g = l5g;
                    l0g = n_l0g;
                    l1g = n_l1g;
                    l2g = n_l2g;
                    l3g = n_l3g;
                    l4g = n_l4g;
                    d = daddr(d, 1, dstride);
                    rem -= 1;
                    if rem == 0 {
                        break;
                    }
                }
            }
            38 | 42 | 46 => {
                let mut s = src0;
                let mut sg = unsafe { src0.add(16) };
                let l00 = loadu128(sadd(s, -2, sstride));
                let l0g = loadu64(sadd(sg, -2, sstride));
                let (mut h00, mut h01) = sixtap_h16(l00, l0g);
                let l10 = loadu128(sadd(s, -1, sstride));
                let l1g = loadu64(sadd(sg, -1, sstride));
                let (mut h10, mut h11) = sixtap_h16(l10, l1g);
                let l20 = loadu128(sadd(s, 0, sstride));
                let l2g = loadu64(sadd(sg, 0, sstride));
                let (mut h20, mut h21) = sixtap_h16(l20, l2g);
                let l30 = loadu128(sadd(s, 1, sstride));
                let l3g = loadu64(sadd(sg, 1, sstride));
                let (mut h30, mut h31) = sixtap_h16(l30, l3g);
                let l40 = loadu128(sadd(s, 2, sstride));
                let l4g = loadu64(sadd(sg, 2, sstride));
                let (mut h40, mut h41) = sixtap_h16(l40, l4g);
                let mut d = dst;
                let mut rem = h;
                loop {
                    s = sadd(s, 1, sstride);
                    sg = sadd(sg, 1, sstride);
                    let l50 = loadu128(sadd(s, 2, sstride));
                    let l5g = loadu64(sadd(sg, 2, sstride));
                    let (h50, h51) = sixtap_h16(l50, l5g);
                    let hv0 = sixtap_hv(h00, h10, h20, h30, h40, h50);
                    let hv1 = sixtap_hv(h01, h11, h21, h31, h41, h51);
                    let hv = shrrpus16(hv0, hv1, 6);
                    let sv = shrrpus16(
                        ifelse_mask(m0, h30, h20),
                        ifelse_mask(m0, h31, h21),
                        5,
                    );
                    let q = loadu128(d);
                    store16(
                        d,
                        maddshr_l(q, avgu8(ifelse_mask(m1, hv, sv), hv), w0, o, wd),
                    );
                    let n_h00 = h10;
                    let n_h01 = h11;
                    let n_h10 = h20;
                    let n_h11 = h21;
                    let n_h20 = h30;
                    let n_h21 = h31;
                    let n_h30 = h40;
                    let n_h31 = h41;
                    let n_h40 = h50;
                    let n_h41 = h51;
                    h00 = n_h00;
                    h01 = n_h01;
                    h10 = n_h10;
                    h11 = n_h11;
                    h20 = n_h20;
                    h21 = n_h21;
                    h30 = n_h30;
                    h31 = n_h31;
                    h40 = n_h40;
                    h41 = n_h41;
                    d = daddr(d, 1, dstride);
                    rem -= 1;
                    if rem == 0 {
                        break;
                    }
                }
            }
            _ => unreachable!("bad inter mode {mode}"),
        }
    }

}
#[cfg(test)]
mod chroma_sse_diff_tests {
    use super::*;

    fn pack_w(w0: i32, w1: i32) -> i16 {
        ((w1 << 8) | (w0 & 255)) as i16
    }

    #[test]
    fn sse_chroma_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("sse4.1") {
            return;
        }
        let wods: [[i16; 8]; 4] = [
            WOD_NO_WEIGHT,
            [257, 1, 1, 1, 257, 257, 1, 1],
            [pack_w(0, 300), 0, 0, 2, pack_w(1, 300), pack_w(2, 260), 128, -200],
            [pack_w(100, 120), 77, 3, 4, pack_w(-5, 90), pack_w(20, -50), 200, -250],
        ];
        for (zi, wod) in wods.iter().enumerate() {
            for &(w, h) in &[
                (16usize, 16usize),
                (16, 8),
                (8, 16),
                (8, 8),
                (8, 4),
                (4, 8),
                (4, 4),
            ] {
                for xf in 0..8u32 {
                    for yf in 0..8u32 {
                        let cw = w / 2;
                        for &sstride in &[4usize, 5, 6, 8, 9, 12, 16] {
                            // dst stride must be >= chroma width (real H.264 always
                            // satisfies this); overlapping rows would diverge between
                            // the chunked SSE read-before-write and sequential scalar.
                            let dstride = sstride.max(cw);
                            let mut seed = ((w as u64) << 40)
                                | ((h as u64) << 32)
                                | ((xf as u64) << 24)
                                | ((yf as u64) << 16)
                                | ((sstride as u64) << 8)
                                | (zi as u64);
                            let mut next = || {
                                seed = seed
                                    .wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                                (seed >> 33) as u8
                            };
                            let mut src = vec![0u8; (h + 2) * sstride + 16];
                            for b in src.iter_mut() {
                                *b = next();
                            }
                            let mut dst0 = vec![0u8; (h - 1) * dstride + cw];
                            for b in dst0.iter_mut() {
                                *b = next();
                            }
                            let mut dst_ref = dst0.clone();
                            let mut dst_sse = dst0.clone();
                            inter_chroma_scalar(
                                &src,
                                &mut dst_ref,
                                w,
                                h,
                                xf,
                                yf,
                                sstride,
                                dstride,
                                wod,
                            );
                            inter_chroma(&src, &mut dst_sse, w, h, xf, yf, sstride, dstride, wod);
                            assert_eq!(
                                dst_sse, dst_ref,
                                "w={w} h={h} xf={xf} yf={yf} ss={sstride} z={zi}"
                            );
                        }
                    }
                }
            }
        }
    }
}

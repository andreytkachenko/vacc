//! Port of `hevc/decoding/transform.cpp` — inverse transform (§8.6.4) and
//! dequantization (§8.6.3).
//!
//! Bit-for-bit port of the hevc.js implementation: same butterfly structures,
//! same shift/rounding, same i32/i64 split in dequantization. Verified
//! differentially against the C++ hevc.js core (since removed); outputs are
//! now pinned by the SHA-256 goldens in `hevc::goldens`.

#[cfg(test)]
pub(crate) use self::tests::golden_entries;

use crate::hevc::cabac_tables::LEVEL_SCALE;
use crate::hevc::types::{clip3, PredMode};

/// AVX2 availability (CPUID leaf 7, EBX bit 5), cached by std after first use.
#[cfg(target_arch = "x86_64")]
fn detect_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_avx2() -> bool {
    false
}

// ============================================================
// Partial butterfly inverse transforms
// ============================================================

// The `k * line + j` column indexing intentionally mirrors the hevc.js
// butterfly layout so this port stays diffable against the C++ reference.
#[allow(clippy::erasing_op, clippy::identity_op)]
/// DST-VII 4x4 inverse (Table 8-12), one column of `line` samples.
fn idst4(src: &[i16], dst: &mut [i16], shift: i32, line: usize) {
    let add = 1 << (shift - 1);
    for j in 0..line {
        let c0 = src[0 * line + j] as i32;
        let c1 = src[1 * line + j] as i32;
        let c2 = src[2 * line + j] as i32;
        let c3 = src[3 * line + j] as i32;

        // DST-VII inverse uses M^T (columns of forward matrix as rows)
        let s0 = 29 * c0 + 74 * c1 + 84 * c2 + 55 * c3;
        let s1 = 55 * c0 + 74 * c1 - 29 * c2 - 84 * c3;
        let s2 = 74 * c0 + 0 * c1 - 74 * c2 + 74 * c3;
        let s3 = 84 * c0 - 74 * c1 + 55 * c2 - 29 * c3;

        dst[0 * line + j] = clip3(-32768, 32767, (s0 + add) >> shift) as i16;
        dst[1 * line + j] = clip3(-32768, 32767, (s1 + add) >> shift) as i16;
        dst[2 * line + j] = clip3(-32768, 32767, (s2 + add) >> shift) as i16;
        dst[3 * line + j] = clip3(-32768, 32767, (s3 + add) >> shift) as i16;
    }
}

#[allow(clippy::erasing_op, clippy::identity_op)]
fn idct4(src: &[i16], dst: &mut [i16], shift: i32, line: usize) {
    let add = 1 << (shift - 1);
    for j in 0..line {
        let c0 = src[0 * line + j] as i32;
        let c1 = src[1 * line + j] as i32;
        let c2 = src[2 * line + j] as i32;
        let c3 = src[3 * line + j] as i32;

        let e0 = 64 * c0 + 64 * c2;
        let e1 = 64 * c0 - 64 * c2;
        let o0 = 83 * c1 + 36 * c3;
        let o1 = 36 * c1 - 83 * c3;

        dst[0 * line + j] = clip3(-32768, 32767, (e0 + o0 + add) >> shift) as i16;
        dst[1 * line + j] = clip3(-32768, 32767, (e1 + o1 + add) >> shift) as i16;
        dst[2 * line + j] = clip3(-32768, 32767, (e1 - o1 + add) >> shift) as i16;
        dst[3 * line + j] = clip3(-32768, 32767, (e0 - o0 + add) >> shift) as i16;
    }
}

#[allow(clippy::erasing_op, clippy::identity_op)]
fn idct8(src: &[i16], dst: &mut [i16], shift: i32, line: usize) {
    let add = 1 << (shift - 1);
    for j in 0..line {
        let c = |k: usize| src[k * line + j] as i32;

        let ee0 = 64 * c(0) + 64 * c(4);
        let ee1 = 64 * c(0) - 64 * c(4);
        let eo0 = 83 * c(2) + 36 * c(6);
        let eo1 = 36 * c(2) - 83 * c(6);

        let e0 = ee0 + eo0;
        let e3 = ee0 - eo0;
        let e1 = ee1 + eo1;
        let e2 = ee1 - eo1;

        let o0 = 89 * c(1) + 75 * c(3) + 50 * c(5) + 18 * c(7);
        let o1 = 75 * c(1) - 18 * c(3) - 89 * c(5) - 50 * c(7);
        let o2 = 50 * c(1) - 89 * c(3) + 18 * c(5) + 75 * c(7);
        let o3 = 18 * c(1) - 50 * c(3) + 75 * c(5) - 89 * c(7);

        dst[0 * line + j] = clip3(-32768, 32767, (e0 + o0 + add) >> shift) as i16;
        dst[1 * line + j] = clip3(-32768, 32767, (e1 + o1 + add) >> shift) as i16;
        dst[2 * line + j] = clip3(-32768, 32767, (e2 + o2 + add) >> shift) as i16;
        dst[3 * line + j] = clip3(-32768, 32767, (e3 + o3 + add) >> shift) as i16;
        dst[4 * line + j] = clip3(-32768, 32767, (e3 - o3 + add) >> shift) as i16;
        dst[5 * line + j] = clip3(-32768, 32767, (e2 - o2 + add) >> shift) as i16;
        dst[6 * line + j] = clip3(-32768, 32767, (e1 - o1 + add) >> shift) as i16;
        dst[7 * line + j] = clip3(-32768, 32767, (e0 - o0 + add) >> shift) as i16;
    }
}

fn idct16(src: &[i16], dst: &mut [i16], shift: i32, line: usize) {
    let add = 1 << (shift - 1);
    for j in 0..line {
        let c = |k: usize| src[k * line + j] as i32;

        // Even-even
        let eee0 = 64 * c(0) + 64 * c(8);
        let eee1 = 64 * c(0) - 64 * c(8);
        let eeo0 = 83 * c(4) + 36 * c(12);
        let eeo1 = 36 * c(4) - 83 * c(12);

        let ee0 = eee0 + eeo0;
        let ee3 = eee0 - eeo0;
        let ee1 = eee1 + eeo1;
        let ee2 = eee1 - eeo1;

        // Even-odd
        let eo0 = 89 * c(2) + 75 * c(6) + 50 * c(10) + 18 * c(14);
        let eo1 = 75 * c(2) - 18 * c(6) - 89 * c(10) - 50 * c(14);
        let eo2 = 50 * c(2) - 89 * c(6) + 18 * c(10) + 75 * c(14);
        let eo3 = 18 * c(2) - 50 * c(6) + 75 * c(10) - 89 * c(14);

        let e = [
            ee0 + eo0,
            ee1 + eo1,
            ee2 + eo2,
            ee3 + eo3,
            ee3 - eo3,
            ee2 - eo2,
            ee1 - eo1,
            ee0 - eo0,
        ];

        // Odd
        let o = [
            90 * c(1) + 87 * c(3) + 80 * c(5) + 70 * c(7) + 57 * c(9) + 43 * c(11) + 25 * c(13)
                + 9 * c(15),
            87 * c(1) + 57 * c(3) + 9 * c(5) - 43 * c(7) - 80 * c(9) - 90 * c(11) - 70 * c(13)
                - 25 * c(15),
            80 * c(1) + 9 * c(3) - 70 * c(5) - 87 * c(7) - 25 * c(9) + 57 * c(11) + 90 * c(13)
                + 43 * c(15),
            70 * c(1) - 43 * c(3) - 87 * c(5) + 9 * c(7) + 90 * c(9) + 25 * c(11) - 80 * c(13)
                - 57 * c(15),
            57 * c(1) - 80 * c(3) - 25 * c(5) + 90 * c(7) - 9 * c(9) - 87 * c(11) + 43 * c(13)
                + 70 * c(15),
            43 * c(1) - 90 * c(3) + 57 * c(5) + 25 * c(7) - 87 * c(9) + 70 * c(11) + 9 * c(13)
                - 80 * c(15),
            25 * c(1) - 70 * c(3) + 90 * c(5) - 80 * c(7) + 43 * c(9) + 9 * c(11) - 57 * c(13)
                + 87 * c(15),
            9 * c(1) - 25 * c(3) + 43 * c(5) - 57 * c(7) + 70 * c(9) - 80 * c(11) + 87 * c(13)
                - 90 * c(15),
        ];

        for k in 0..8 {
            dst[k * line + j] = clip3(-32768, 32767, (e[k] + o[k] + add) >> shift) as i16;
            dst[(15 - k) * line + j] = clip3(-32768, 32767, (e[k] - o[k] + add) >> shift) as i16;
        }
    }
}

/// DCT coefficients from the spec tables (Table 8-5, 32-point).
const TM_32: [[i32; 32]; 32] = [
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    [90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4, -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90],
    [90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -25, -9, 9, 25, 43, 57, 70, 80, 87, 90],
    [90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13, 13, 38, 61, 78, 88, 90, 85, 73, 54, 31, 4, -22, -46, -67, -82, -90],
    [89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89],
    [88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22, -22, -61, -85, -90, -73, -38, 4, 46, 78, 90, 82, 54, 13, -31, -67, -88],
    [87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80, 90, 70, 25, -25, -70, -90, -80, -43, 9, 57, 87],
    [85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31, 31, 78, 90, 61, 4, -54, -88, -82, -38, 22, 73, 90, 67, 13, -46, -85],
    [83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83],
    [82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38, -38, -88, -73, -4, 67, 90, 46, -31, -85, -78, -13, 61, 90, 54, -22, -82],
    [80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80, -80, -9, 70, 87, 25, -57, -90, -43, 43, 90, 57, -25, -87, -70, 9, 80],
    [78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46, 46, 90, 38, -54, -90, -31, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78],
    [75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75],
    [73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54, -54, -85, 4, 88, 46, -61, -82, 13, 90, 38, -67, -78, 22, 90, 31, -73],
    [70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90, -25, 80, 57, -57, -80, 25, 90, 9, -87, -43, 70],
    [67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61, 61, 73, -46, -82, 31, 88, -13, -90, -4, 90, 22, -85, -38, 78, 54, -67],
    [64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64],
    [61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67, -67, -54, 78, 38, -85, -22, 90, 4, -90, 13, 88, -31, -82, 46, 73, -61],
    [57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57, -57, 80, 25, -90, 9, 87, -43, -70, 70, 43, -87, -9, 90, -25, -80, 57],
    [54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73, 73, 31, -90, 22, 78, -67, -38, 90, -13, -82, 61, 46, -88, 4, 85, -54],
    [50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50, 50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50],
    [46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78, -78, -4, 82, -73, -13, 85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46],
    [43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43, -43, 90, -57, -25, 87, -70, -9, 80, -80, 9, 70, -87, 25, 57, -90, 43],
    [38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82, 82, -22, -54, 90, -61, -13, 78, -85, 31, 46, -90, 67, 4, -73, 88, -38],
    [36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36],
    [31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85, -85, 46, 13, -67, 90, -73, 22, 38, -82, 88, -54, -4, 61, -90, 78, -31],
    [25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25, -25, 70, -90, 80, -43, -9, 57, -87, 87, -57, 9, 43, -80, 90, -70, 25],
    [22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88, 88, -67, 31, 13, -54, 82, -90, 78, -46, 4, 38, -73, 90, -85, 61, -22],
    [18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18, 18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18],
    [13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90, -90, 82, -67, 46, -22, -4, 31, -54, 73, -85, 90, -88, 78, -61, 38, -13],
    [9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9, -9, 25, -43, 57, -70, 80, -87, 90, -90, 87, -80, 70, -57, 43, -25, 9],
    [4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90, 90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4],
];

fn idct32(src: &[i16], dst: &mut [i16], shift: i32, line: usize) {
    let add = 1 << (shift - 1);

    for j in 0..line {
        let c = |k: usize| src[k * line + j] as i32;

        // Odd
        let mut o = [0i32; 16];
        for k in 0..16 {
            let mut sum = 0;
            for n in 0..16 {
                sum += TM_32[2 * n + 1][k] * c(2 * n + 1);
            }
            o[k] = sum;
        }
        // Even-Odd
        let mut eo = [0i32; 8];
        for k in 0..8 {
            let mut sum = 0;
            for n in 0..8 {
                sum += TM_32[2 * (2 * n + 1)][k] * c(2 * (2 * n + 1));
            }
            eo[k] = sum;
        }
        // Even-Even-Odd
        let mut eeo = [0i32; 4];
        for k in 0..4 {
            let mut sum = 0;
            for n in 0..4 {
                sum += TM_32[4 * (2 * n + 1)][k] * c(4 * (2 * n + 1));
            }
            eeo[k] = sum;
        }
        // Even-Even-Even-Even and Even-Even-Even-Odd (2-point + 2-point)
        let eeee0 = 64 * c(0) + 64 * c(16);
        let eeee1 = 64 * c(0) - 64 * c(16);
        let eeoo0 = 83 * c(8) + 36 * c(24);
        let eeoo1 = 36 * c(8) - 83 * c(24);
        let eee = [eeee0 + eeoo0, eeee1 + eeoo1, eeee1 - eeoo1, eeee0 - eeoo0];

        // Build up
        let ee = [
            eee[0] + eeo[0],
            eee[1] + eeo[1],
            eee[2] + eeo[2],
            eee[3] + eeo[3],
            eee[3] - eeo[3],
            eee[2] - eeo[2],
            eee[1] - eeo[1],
            eee[0] - eeo[0],
        ];

        let mut e = [0i32; 16];
        for k in 0..8 {
            e[k] = ee[k] + eo[k];
            e[15 - k] = ee[k] - eo[k];
        }

        for k in 0..16 {
            dst[k * line + j] = clip3(-32768, 32767, (e[k] + o[k] + add) >> shift) as i16;
            dst[(31 - k) * line + j] = clip3(-32768, 32767, (e[k] - o[k] + add) >> shift) as i16;
        }
    }
}

// ============================================================
// 2D inverse transform (§8.6.4.2)
// Vertical pass -> transpose -> horizontal pass
// ============================================================

fn inverse_transform_2d(
    log2_trafo_size: u32,
    use_dst: bool,
    bit_depth: u32,
    coeff: &[i16],
    residual: &mut [i16],
) {
    let tr_size = 1usize << log2_trafo_size;
    let mut tmp = vec![0i16; tr_size * tr_size];
    let mut tmp2 = vec![0i16; tr_size * tr_size];

    // Vertical pass: shift1 = 7
    let shift1: i32 = 7;
    // Horizontal pass: shift2 = 20 - BitDepth
    let shift2 = 20 - bit_depth as i32;

    let pass = |src: &[i16], dst: &mut [i16], shift: i32| {
        #[cfg(target_arch = "x86_64")]
        if detect_avx2() {
            unsafe {
                match log2_trafo_size {
                    2 => {
                        if use_dst {
                            avx2::idst4(src, dst, shift, tr_size);
                        } else {
                            avx2::idct4(src, dst, shift, tr_size);
                        }
                    }
                    3 => avx2::idct8(src, dst, shift, tr_size),
                    4 => avx2::idct16(src, dst, shift, tr_size),
                    5 => avx2::idct32(src, dst, shift, tr_size),
                    _ => unreachable!("invalid log2TrafoSize"),
                }
            }
            return;
        }
        if use_dst && log2_trafo_size == 2 {
            idst4(src, dst, shift, tr_size);
        } else {
            match log2_trafo_size {
                2 => idct4(src, dst, shift, tr_size),
                3 => idct8(src, dst, shift, tr_size),
                4 => idct16(src, dst, shift, tr_size),
                5 => idct32(src, dst, shift, tr_size),
                _ => unreachable!("invalid log2TrafoSize"),
            }
        }
    };

    // Pass 1: vertical (columns)
    pass(coeff, &mut tmp, shift1);

    // Transpose between passes so pass 2 transforms rows.
    for y in 0..tr_size {
        for x in 0..tr_size {
            tmp2[y * tr_size + x] = tmp[x * tr_size + y];
        }
    }

    // Pass 2: horizontal (rows)
    pass(&tmp2, &mut tmp, shift2);

    // Transpose back to get the final residual in row-major order.
    for y in 0..tr_size {
        for x in 0..tr_size {
            residual[y * tr_size + x] = tmp[x * tr_size + y];
        }
    }
}

// ============================================================
// Dequantization (§8.6.3)
// ============================================================

/// Scaling list data — spec §7.3.4 (mirrors `hevc::ScalingListData`).
#[derive(Clone, Copy, Debug)]
pub struct ScalingListData {
    /// `scaling_list[sizeId][matrixId][coefIdx]`, flattened as
    /// `[sizeId * 6 + matrixId][64]`.
    pub scaling_list: [[u8; 64]; 24],
    /// DC coefficients for 16x16 and 32x32: `[(sizeId - 2) * 6 + matrixId]`.
    pub scaling_list_dc: [u8; 12],
}

// Arrays of 64+ elements do not implement Default; provide it manually.
impl Default for ScalingListData {
    fn default() -> Self {
        Self {
            scaling_list: [[0u8; 64]; 24],
            scaling_list_dc: [0u8; 12],
        }
    }
}

/// Default 8x8 intra scaling list (spec Table 7-4).
const DEFAULT_8X8_INTRA: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 21, 24, 16, 16, 16, 16, 17, 19, 22, 25, 16, 16, 17, 18, 20, 22, 25,
    29, 16, 16, 18, 21, 24, 27, 31, 36, 17, 17, 20, 24, 30, 35, 41, 47, 18, 19, 22, 27, 35, 44,
    54, 65, 21, 22, 25, 31, 41, 54, 70, 88, 24, 25, 29, 36, 47, 65, 88, 115,
];

/// Default 8x8 inter scaling list (spec Table 7-5).
const DEFAULT_8X8_INTER: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 20, 24, 16, 16, 16, 17, 18, 20, 24, 25, 16, 16, 17, 18, 20, 24, 25,
    28, 16, 17, 18, 20, 24, 25, 28, 33, 17, 18, 20, 24, 25, 28, 33, 41, 18, 20, 24, 25, 28, 33,
    41, 54, 20, 24, 25, 28, 33, 41, 54, 71, 24, 25, 28, 33, 41, 54, 71, 91,
];

impl ScalingListData {
    /// Initialize with default values (spec Tables 7-3 to 7-5).
    pub fn set_defaults(&mut self) {
        // sizeId 0 (4x4): all flat 16
        for matrix_id in 0..6 {
            self.scaling_list[matrix_id].fill(16);
        }
        // sizeId 1 (8x8): intra for matrixId 0-2, inter for 3-5
        for matrix_id in 0..6 {
            let src = if matrix_id < 3 { &DEFAULT_8X8_INTRA } else { &DEFAULT_8X8_INTER };
            self.scaling_list[6 + matrix_id].copy_from_slice(src);
        }
        // sizeId 2 (16x16): same as sizeId 1 (uses 8x8 coefficients)
        for matrix_id in 0..6 {
            let src = if matrix_id < 3 { &DEFAULT_8X8_INTRA } else { &DEFAULT_8X8_INTER };
            self.scaling_list[12 + matrix_id].copy_from_slice(src);
        }
        // sizeId 3 (32x32): matrixId 0 = intra, matrixId 3 = inter
        self.scaling_list[18].copy_from_slice(&DEFAULT_8X8_INTRA);
        self.scaling_list[21].copy_from_slice(&DEFAULT_8X8_INTER);

        // Default DC coefficients = 16
        self.scaling_list_dc.fill(16);
    }
}

/// Parameters for dequantization that the C++ implementation takes from the
/// `DecodingContext` (SPS/PPS + the current CU's prediction mode).
pub struct DequantParams<'a> {
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub scaling_list_enabled: bool,
    pub sps_scaling_list: &'a ScalingListData,
    pub pps_scaling_list_present: bool,
    pub pps_scaling_list: &'a ScalingListData,
    /// Prediction mode of the CU covering the TU (§8.6.3 CuPredMode).
    pub cu_pred_mode: PredMode,
}

/// Dequantization (§8.6.3). `coefficients`/`scaled` hold trSize^2 samples.
pub fn perform_dequant(
    p: &DequantParams<'_>,
    log2_trafo_size: u32,
    c_idx: u32,
    qp: i32,
    coefficients: &[i16],
    scaled: &mut [i16],
) {
    let tr_size = 1usize << log2_trafo_size;

    // §8.6.3 — bdShift = BitDepth + Log2(nTbS) + 10 - log2TransformRange
    // log2TransformRange = 15 for Main profile
    let bit_depth = if c_idx == 0 {
        p.bit_depth_luma
    } else {
        p.bit_depth_chroma
    };
    let bd_shift = bit_depth as i32 + log2_trafo_size as i32 + 10 - 15;
    let add = if bd_shift > 0 { 1i64 << (bd_shift - 1) } else { 0 };

    debug_assert!(
        (0..=63).contains(&qp),
        "QP out of range: {qp} (c_idx={c_idx}, log2={log2_trafo_size})"
    );
    let qp_per = qp / 6;
    let scale = LEVEL_SCALE[(qp % 6) as usize];

    // §8.6.3 collapses to 32-bit arithmetic whenever qpPer <= bdShift.
    // coeff * m * scale always fits in i32 (|32768 * 255 * 72| < 2^31); only
    // the << qpPer overflows, and dividing numerator and denominator by 2^qpPer
    // turns ((X << qpPer) + (1 << (bdShift-1))) >> bdShift into
    // (X + (1 << (s-1))) >> s. At s == 0 the rounding term vanishes and the
    // result is X.
    let shift_down = bd_shift - qp_per;
    let narrow = bd_shift > 0 && shift_down >= 0;
    let narrow_add = if shift_down >= 1 { 1 << (shift_down - 1) } else { 0 };

    let use_scaling_list = p.scaling_list_enabled;

    for y in 0..tr_size {
        for x in 0..tr_size {
            let coeff = coefficients[y * tr_size + x] as i32;
            if coeff == 0 {
                scaled[y * tr_size + x] = 0;
                continue;
            }

            let m: i32 = if use_scaling_list {
                let size_id = match log2_trafo_size {
                    2 => 0,
                    3 => 1,
                    4 => 2,
                    _ => 3,
                };

                let matrix_id = if size_id < 3 {
                    let base = match c_idx {
                        0 => 0,
                        1 => 1,
                        _ => 2,
                    };
                    // §8.6.3: CuPredMode[xTbY][yTbY] — use current CU, not (0,0)
                    if p.cu_pred_mode != PredMode::Intra {
                        base + 3
                    } else {
                        base
                    }
                } else {
                    if p.cu_pred_mode == PredMode::Intra { 0 } else { 3 }
                };

                let sl = if p.pps_scaling_list_present {
                    p.pps_scaling_list
                } else {
                    p.sps_scaling_list
                };

                if size_id == 0 {
                    sl.scaling_list[size_id * 6 + matrix_id][y * tr_size + x] as i32
                } else {
                    // Upscale from 8x8 matrix
                    let ratio = (tr_size / 8).max(1);
                    let idx = ((y / ratio) * 8 + (x / ratio)).min(63);
                    let mut m = sl.scaling_list[size_id * 6 + matrix_id % 6][idx] as i32;

                    // DC coeff override for 16x16 and 32x32
                    if (size_id == 2 || size_id == 3) && x == 0 && y == 0 {
                        m = sl.scaling_list_dc[(size_id - 2) * 6 + matrix_id % 6] as i32;
                        if m == 0 {
                            m = 16;
                        }
                    }
                    m
                }
            } else {
                // Flat scaling (no scaling list)
                16
            };

            // §8.6.3: d[x][y] = Clip3(coeffMin, coeffMax,
            //   ((coeff * m * levelScale[qP%6] << (qP/6)) + (1<<(bdShift-1))) >> bdShift)
            let val32: i32;
            if narrow {
                let prod = coeff * m * scale as i32;
                val32 = if shift_down >= 1 {
                    (prod + narrow_add) >> shift_down
                } else {
                    prod
                };
            } else {
                let mut val = (coeff as i64) * (m as i64) * (scale as i64);
                val = (val << qp_per) + add;
                // C++ does `val >>= bdShift` — an arithmetic shift; bdShift can
                // be negative (e.g. 8-bit 4x4: -1), which GCC compiles to a
                // left shift.
                val = if bd_shift >= 0 { val >> bd_shift } else { val << (-bd_shift) };
                val32 = val as i32;
            }
            scaled[y * tr_size + x] = clip3(-32768, 32767, val32) as i16;
        }
    }
}

// ============================================================
// Transform inverse entry point
// ============================================================

/// Inverse transform (§8.6.4). `scaled`/`residual` hold trSize^2 samples.
pub fn perform_transform_inverse(
    log2_trafo_size: u32,
    c_idx: u32,
    is_intra: bool,
    transform_skip: bool,
    bit_depth: u32,
    scaled: &[i16],
    residual: &mut [i16],
) {
    let tr_size = 1usize << log2_trafo_size;

    if transform_skip {
        // Transform skip: shift = 15 - BitDepth
        let shift = (15 - bit_depth as i32).max(0);
        let add = if shift > 0 { 1 << (shift - 1) } else { 0 };
        for i in 0..tr_size * tr_size {
            residual[i] = ((scaled[i] as i32 + add) >> shift) as i16;
        }
        return;
    }

    // Use DST for 4x4 luma intra
    let use_dst = is_intra && c_idx == 0 && log2_trafo_size == 2;

    inverse_transform_2d(log2_trafo_size, use_dst, bit_depth, scaled, residual);
}

// ============================================================
// AVX2 inverse transform — vectorized across columns
// ============================================================
//
// Each pass transforms `line` independent 1-D transforms (the columns of
// the current pass). Lanes hold one column each, so every butterfly term —
// all constant×coefficient sums — becomes a lane-wise i32 op. Batches are
// 8 columns (AVX2 width); 4x4 uses the low half of a 256-bit vector. The
// i32 lanes keep the exact scalar arithmetic, and the final shift plus
// saturating pack to i16 reproduces `clip3(-32768, 32767, x)` exactly.

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_loadl_epi64, _mm_loadu_si128, _mm_packs_epi32,
        _mm_storeu_si128, _mm256_add_epi32, _mm256_castsi256_si128,
        _mm256_cvtepi16_epi32, _mm256_extracti128_si256, _mm256_mullo_epi32,
        _mm256_set1_epi32, _mm256_setzero_si256, _mm256_srai_epi32,
        _mm256_sub_epi32,
    };

    use super::TM_32;

    /// Load 8 contiguous i16s, sign-extended to 8 i32 lanes.
    #[inline]
    fn ld8(p: *const i16) -> __m256i {
        unsafe { _mm256_cvtepi16_epi32(_mm_loadu_si128(p as *const __m128i)) }
    }

    /// Load 4 contiguous i16s into the low half (upper lanes zero).
    #[inline]
    fn ld4(p: *const i16) -> __m256i {
        unsafe { _mm256_cvtepi16_epi32(_mm_loadl_epi64(p as *const __m128i)) }
    }

    #[inline]
    fn mulc(v: __m256i, c: i32) -> __m256i {
        unsafe { _mm256_mullo_epi32(v, _mm256_set1_epi32(c)) }
    }

    #[inline]
    fn vadd(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_add_epi32(a, b) }
    }

    #[inline]
    fn vsub(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_sub_epi32(a, b) }
    }

    /// `(v + add) >> shift` — exact i32 arithmetic right shift.
    ///
    /// `shift` is 7 (vertical pass) or `20 - bit_depth` with bit depth 8..=12,
    /// so only these values are reachable.
    #[inline]
    fn fin(v: __m256i, add: i32, shift: i32) -> __m256i {
        let v = unsafe { vadd(v, _mm256_set1_epi32(add)) };
        match shift {
            7 => unsafe { _mm256_srai_epi32::<7>(v) },
            8 => unsafe { _mm256_srai_epi32::<8>(v) },
            9 => unsafe { _mm256_srai_epi32::<9>(v) },
            10 => unsafe { _mm256_srai_epi32::<10>(v) },
            11 => unsafe { _mm256_srai_epi32::<11>(v) },
            12 => unsafe { _mm256_srai_epi32::<12>(v) },
            s => unreachable!("unsupported transform shift {s}"),
        }
    }

    /// Store 8 i32 lanes as 8 contiguous saturated i16s.
    #[inline]
    fn store8(p: *mut i16, v: __m256i) {
        unsafe {
            let lo = _mm256_castsi256_si128(v);
            let hi = _mm256_extracti128_si256(v, 1);
            _mm_storeu_si128(p as *mut __m128i, _mm_packs_epi32(lo, hi));
        }
    }

    /// Store the low halves of two i32 vectors as 8 contiguous saturated i16s.
    #[inline]
    fn store4(p: *mut i16, a: __m256i, b: __m256i) {
        unsafe {
            _mm_storeu_si128(
                p as *mut __m128i,
                _mm_packs_epi32(_mm256_castsi256_si128(a), _mm256_castsi256_si128(b)),
            );
        }
    }

    /// DST-VII 4x4 — 4 columns per batch (low half of a 256-bit vector).
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn idst4(src: &[i16], dst: &mut [i16], shift: i32, line: usize) {
        debug_assert!(line % 4 == 0);
        let add = 1i32 << (shift - 1);
        let src = src.as_ptr();
        let dst = dst.as_mut_ptr();
        let mut j = 0usize;
        while j + 4 <= line {
            unsafe {
                let v0 = ld4(src.add(j));
                let v1 = ld4(src.add(line + j));
                let v2 = ld4(src.add(2 * line + j));
                let v3 = ld4(src.add(3 * line + j));
                let r0 = fin(
                    vadd(vadd(vadd(mulc(v0, 29), mulc(v1, 74)), mulc(v2, 84)), mulc(v3, 55)),
                    add,
                    shift,
                );
                let r1 = fin(
                    vsub(vadd(mulc(v0, 55), mulc(v1, 74)), vadd(mulc(v2, 29), mulc(v3, 84))),
                    add,
                    shift,
                );
                let r2 = fin(vadd(vadd(mulc(v0, 74), mulc(v3, 74)), mulc(v2, -74)), add, shift);
                let r3 = fin(
                    vsub(vadd(mulc(v0, 84), mulc(v2, 55)), vadd(mulc(v1, 74), mulc(v3, 29))),
                    add,
                    shift,
                );
                store4(dst.add(j), r0, r1);
                store4(dst.add(2 * line + j), r2, r3);
            }
            j += 4;
        }
    }

    /// IDCT-4 — 4 columns per batch (low half of a 256-bit vector).
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn idct4(src: &[i16], dst: &mut [i16], shift: i32, line: usize) {
        debug_assert!(line % 4 == 0);
        let add = 1i32 << (shift - 1);
        let src = src.as_ptr();
        let dst = dst.as_mut_ptr();
        let mut j = 0usize;
        while j + 4 <= line {
            unsafe {
                let v0 = ld4(src.add(j));
                let v1 = ld4(src.add(line + j));
                let v2 = ld4(src.add(2 * line + j));
                let v3 = ld4(src.add(3 * line + j));
                let m0 = mulc(v0, 64);
                let m2 = mulc(v2, 64);
                let e0 = vadd(m0, m2);
                let e1 = vsub(m0, m2);
                let a = mulc(v1, 83);
                let b = mulc(v3, 36);
                let o0 = vadd(a, b);
                let o1 = vsub(mulc(v1, 36), mulc(v3, 83));
                let r0 = fin(vadd(e0, o0), add, shift);
                let r1 = fin(vadd(e1, o1), add, shift);
                let r2 = fin(vsub(e1, o1), add, shift);
                let r3 = fin(vsub(e0, o0), add, shift);
                store4(dst.add(j), r0, r1);
                store4(dst.add(2 * line + j), r2, r3);
            }
            j += 4;
        }
    }

    /// IDCT-8 — 8 columns per batch.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn idct8(src: &[i16], dst: &mut [i16], shift: i32, line: usize) {
        debug_assert!(line % 8 == 0);
        let add = 1i32 << (shift - 1);
        let src = src.as_ptr();
        let dst = dst.as_mut_ptr();
        let mut j = 0usize;
        while j + 8 <= line {
            unsafe {
                let v0 = ld8(src.add(j));
                let v1 = ld8(src.add(line + j));
                let v2 = ld8(src.add(2 * line + j));
                let v3 = ld8(src.add(3 * line + j));
                let v4 = ld8(src.add(4 * line + j));
                let v5 = ld8(src.add(5 * line + j));
                let v6 = ld8(src.add(6 * line + j));
                let v7 = ld8(src.add(7 * line + j));

                let m0 = mulc(v0, 64);
                let m4 = mulc(v4, 64);
                let ee0 = vadd(m0, m4);
                let ee1 = vsub(m0, m4);
                let m2 = mulc(v2, 83);
                let m6 = mulc(v6, 36);
                let eo0 = vadd(m2, m6);
                let eo1 = vsub(mulc(v2, 36), mulc(v6, 83));
                let e0 = vadd(ee0, eo0);
                let e3 = vsub(ee0, eo0);
                let e1 = vadd(ee1, eo1);
                let e2 = vsub(ee1, eo1);

                let a1 = mulc(v1, 89);
                let b1 = mulc(v1, 75);
                let c1 = mulc(v1, 50);
                let d1 = mulc(v1, 18);
                let a3 = mulc(v3, 75);
                let b3 = mulc(v3, -18);
                let c3 = mulc(v3, -89);
                let d3 = mulc(v3, -50);
                let a5 = mulc(v5, 50);
                let b5 = mulc(v5, -89);
                let c5 = mulc(v5, 18);
                let d5 = mulc(v5, 75);
                let a7 = mulc(v7, 18);
                let b7 = mulc(v7, -50);
                let c7 = mulc(v7, 75);
                let d7 = mulc(v7, -89);
                let o0 = vadd(vadd(a1, a3), vadd(a5, a7));
                let o1 = vadd(vadd(b1, b3), vadd(b5, b7));
                let o2 = vadd(vadd(c1, c3), vadd(c5, c7));
                let o3 = vadd(vadd(d1, d3), vadd(d5, d7));

                let r0 = fin(vadd(e0, o0), add, shift);
                let r1 = fin(vadd(e1, o1), add, shift);
                let r2 = fin(vadd(e2, o2), add, shift);
                let r3 = fin(vadd(e3, o3), add, shift);
                let r4 = fin(vsub(e3, o3), add, shift);
                let r5 = fin(vsub(e2, o2), add, shift);
                let r6 = fin(vsub(e1, o1), add, shift);
                let r7 = fin(vsub(e0, o0), add, shift);
                store8(dst.add(j), r0);
                store8(dst.add(line + j), r1);
                store8(dst.add(2 * line + j), r2);
                store8(dst.add(3 * line + j), r3);
                store8(dst.add(4 * line + j), r4);
                store8(dst.add(5 * line + j), r5);
                store8(dst.add(6 * line + j), r6);
                store8(dst.add(7 * line + j), r7);
            }
            j += 8;
        }
    }

    /// IDCT-16 — 8 columns per batch.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn idct16(src: &[i16], dst: &mut [i16], shift: i32, line: usize) {
        debug_assert!(line % 8 == 0);
        let add = 1i32 << (shift - 1);
        let src = src.as_ptr();
        let dst = dst.as_mut_ptr();
        // Odd submatrix over c(1), c(3), ..., c(15) — same constants as the
        // scalar butterfly below.
        const M16: [[i32; 8]; 8] = [
            [90, 87, 80, 70, 57, 43, 25, 9],
            [87, 57, 9, -43, -80, -90, -70, -25],
            [80, 9, -70, -87, -25, 57, 90, 43],
            [70, -43, -87, 9, 90, 25, -80, -57],
            [57, -80, -25, 90, -9, -87, 43, 70],
            [43, -90, 57, 25, -87, 70, 9, -80],
            [25, -70, 90, -80, 43, 9, -57, 87],
            [9, -25, 43, -57, 70, -80, 87, -90],
        ];
        let mut j = 0usize;
        while j + 8 <= line {
            unsafe {
                // Odd part: o[k] = sum_n M16[k][n] * c(2n+1).
                let zero = _mm256_setzero_si256();
                let mut o = [zero; 8];
                for n in 0..8usize {
                    let v = ld8(src.add((2 * n + 1) * line + j));
                    for k in 0..8usize {
                        o[k] = vadd(o[k], mulc(v, M16[k][n]));
                    }
                }
                // Even part.
                let v0 = ld8(src.add(j));
                let v4 = ld8(src.add(4 * line + j));
                let v8 = ld8(src.add(8 * line + j));
                let v12 = ld8(src.add(12 * line + j));
                let m0 = mulc(v0, 64);
                let m8 = mulc(v8, 64);
                let eee0 = vadd(m0, m8);
                let eee1 = vsub(m0, m8);
                let m4 = mulc(v4, 83);
                let m12 = mulc(v12, 36);
                let eeo0 = vadd(m4, m12);
                let eeo1 = vsub(mulc(v4, 36), mulc(v12, 83));
                let ee0 = vadd(eee0, eeo0);
                let ee3 = vsub(eee0, eeo0);
                let ee1 = vadd(eee1, eeo1);
                let ee2 = vsub(eee1, eeo1);
                let v2 = ld8(src.add(2 * line + j));
                let v6 = ld8(src.add(6 * line + j));
                let v10 = ld8(src.add(10 * line + j));
                let v14 = ld8(src.add(14 * line + j));
                let a2 = mulc(v2, 89);
                let b2 = mulc(v2, 75);
                let c2 = mulc(v2, 50);
                let d2 = mulc(v2, 18);
                let a6 = mulc(v6, 75);
                let b6 = mulc(v6, -18);
                let c6 = mulc(v6, -89);
                let d6 = mulc(v6, -50);
                let a10 = mulc(v10, 50);
                let b10 = mulc(v10, -89);
                let c10 = mulc(v10, 18);
                let d10 = mulc(v10, 75);
                let a14 = mulc(v14, 18);
                let b14 = mulc(v14, -50);
                let c14 = mulc(v14, 75);
                let d14 = mulc(v14, -89);
                let eo0 = vadd(vadd(a2, a6), vadd(a10, a14));
                let eo1 = vadd(vadd(b2, b6), vadd(b10, b14));
                let eo2 = vadd(vadd(c2, c6), vadd(c10, c14));
                let eo3 = vadd(vadd(d2, d6), vadd(d10, d14));
                let e = [
                    vadd(ee0, eo0),
                    vadd(ee1, eo1),
                    vadd(ee2, eo2),
                    vadd(ee3, eo3),
                    vsub(ee3, eo3),
                    vsub(ee2, eo2),
                    vsub(ee1, eo1),
                    vsub(ee0, eo0),
                ];
                for k in 0..8usize {
                    store8(dst.add(k * line + j), fin(vadd(e[k], o[k]), add, shift));
                    store8(
                        dst.add((15 - k) * line + j),
                        fin(vsub(e[k], o[k]), add, shift),
                    );
                }
            }
            j += 8;
        }
    }

    /// IDCT-32 — 8 columns per batch. The odd/eo/eeo parts are the dense
    /// TM_32 submatrices of the scalar butterfly; lanes are columns.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn idct32(src: &[i16], dst: &mut [i16], shift: i32, line: usize) {
        debug_assert!(line % 8 == 0);
        let add = 1i32 << (shift - 1);
        let src = src.as_ptr();
        let dst = dst.as_mut_ptr();
        let mut j = 0usize;
        while j + 8 <= line {
            unsafe {
                let zero = _mm256_setzero_si256();
                // Odd part: o[k] = sum_n TM_32[2n+1][k] * c(2n+1).
                let mut o_lo = [zero; 8];
                for n in 0..16usize {
                    let v = ld8(src.add((2 * n + 1) * line + j));
                    for k in 0..8usize {
                        o_lo[k] = vadd(o_lo[k], mulc(v, TM_32[2 * n + 1][k]));
                    }
                }
                let mut o_hi = [zero; 8];
                for n in 0..16usize {
                    let v = ld8(src.add((2 * n + 1) * line + j));
                    for k in 0..8usize {
                        o_hi[k] = vadd(o_hi[k], mulc(v, TM_32[2 * n + 1][k + 8]));
                    }
                }
                // Even-odd: eo[k] = sum_n TM_32[2(2n+1)][k] * c(2(2n+1)).
                let mut eo = [zero; 8];
                for n in 0..8usize {
                    let v = ld8(src.add((2 * (2 * n + 1)) * line + j));
                    for k in 0..8usize {
                        eo[k] = vadd(eo[k], mulc(v, TM_32[2 * (2 * n + 1)][k]));
                    }
                }
                // Even-even-odd: eeo[k] = sum_n TM_32[4(2n+1)][k] * c(4(2n+1)).
                let mut eeo = [zero; 4];
                for n in 0..4usize {
                    let v = ld8(src.add((4 * (2 * n + 1)) * line + j));
                    for k in 0..4usize {
                        eeo[k] = vadd(eeo[k], mulc(v, TM_32[4 * (2 * n + 1)][k]));
                    }
                }
                // Even-even-even: 2-point butterflies.
                let v0 = ld8(src.add(j));
                let v16 = ld8(src.add(16 * line + j));
                let m0 = mulc(v0, 64);
                let m16 = mulc(v16, 64);
                let eeee0 = vadd(m0, m16);
                let eeee1 = vsub(m0, m16);
                let v8 = ld8(src.add(8 * line + j));
                let v24 = ld8(src.add(24 * line + j));
                let m8 = mulc(v8, 83);
                let m24 = mulc(v24, 36);
                let eeoo0 = vadd(m8, m24);
                let eeoo1 = vsub(mulc(v8, 36), mulc(v24, 83));
                let eee = [
                    vadd(eeee0, eeoo0),
                    vadd(eeee1, eeoo1),
                    vsub(eeee1, eeoo1),
                    vsub(eeee0, eeoo0),
                ];
                let ee = [
                    vadd(eee[0], eeo[0]),
                    vadd(eee[1], eeo[1]),
                    vadd(eee[2], eeo[2]),
                    vadd(eee[3], eeo[3]),
                    vsub(eee[3], eeo[3]),
                    vsub(eee[2], eeo[2]),
                    vsub(eee[1], eeo[1]),
                    vsub(eee[0], eeo[0]),
                ];
                let mut e = [zero; 16];
                for k in 0..8usize {
                    e[k] = vadd(ee[k], eo[k]);
                    e[15 - k] = vsub(ee[k], eo[k]);
                }
                for k in 0..8usize {
                    store8(dst.add(k * line + j), fin(vadd(e[k], o_lo[k]), add, shift));
                    store8(
                        dst.add((31 - k) * line + j),
                        fin(vsub(e[k], o_lo[k]), add, shift),
                    );
                }
                for k in 0..8usize {
                    store8(
                        dst.add((8 + k) * line + j),
                        fin(vadd(e[8 + k], o_hi[k]), add, shift),
                    );
                    store8(
                        dst.add((23 - k) * line + j),
                        fin(vsub(e[8 + k], o_hi[k]), add, shift),
                    );
                }
            }
            j += 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::goldens;

    /// Deterministic PRNG (splitmix64).
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    /// Coefficients biased toward small values with occasional extremes, so
    /// both the rounding and the clip paths get exercised.
    fn random_coeff(rng: &mut Rng) -> i16 {
        match rng.below(10) {
            0 => -32768,
            1 => 32767,
            2 => -(1 << 14),
            3 => 1 << 14,
            _ => rng.below(2001) as i16 - 1000,
        }
    }

    fn fill_scaling_list(flat: [u8; 1536], dc: [u8; 12]) -> ScalingListData {
        let mut s = ScalingListData::default();
        for (dst, src) in s.scaling_list.iter_mut().zip(flat.chunks_exact(64)) {
            dst.copy_from_slice(src);
        }
        s.scaling_list_dc.copy_from_slice(&dc);
        s
    }

    // ---- Golden tests -----------------------------------------------------
    // Outputs are pinned by SHA-256 goldens generated from the build that
    // verified byte-exact agreement with the C++ oracle.

    fn compute_transform_inverse() -> Vec<(String, Vec<u8>)> {
        let mut rng = Rng::new(42);
        let mut buf = Vec::new();
        for log2 in 2..=5u32 {
            let n = (1 << log2) * (1 << log2);
            for c_idx in 0..3u32 {
                for is_intra in [false, true] {
                    for skip in [false, true] {
                        for bit_depth in [8u32, 10] {
                            for _iter in 0..50 {
                                let coeffs: Vec<i16> = (0..n).map(|_| random_coeff(&mut rng)).collect();
                                let mut rust_out = vec![0i16; n];
                                perform_transform_inverse(
                                    log2, c_idx, is_intra, skip, bit_depth, &coeffs, &mut rust_out,
                                );
                                for v in &rust_out {
                                    goldens::push_i16(&mut buf, *v);
                                }
                            }
                        }
                    }
                }
            }
        }
        vec![("transform::inverse".to_string(), buf)]
    }

    #[test]
    fn transform_inverse_matches_golden() {
        for (key, data) in compute_transform_inverse() {
            goldens::assert_golden(&key, &data);
        }
    }

    fn compute_dequant_flat() -> Vec<(String, Vec<u8>)> {
        let mut buf = Vec::new();
        let mut rng = Rng::new(7);
        for log2 in 2..=5u32 {
            let n = (1 << log2) * (1 << log2);
            for c_idx in 0..3u32 {
                for qp in 0..=63i32 {
                    // Extreme QPs cover the narrow/wide path boundary.
                    if !(0..=5).contains(&qp) && !(60..=63).contains(&qp) && qp % 7 != 0 {
                        continue;
                    }
                    for cu_is_intra in [false, true] {
                        for _iter in 0..10 {
                            let coeffs: Vec<i16> = (0..n).map(|_| random_coeff(&mut rng)).collect();
                            let mut rust_out = vec![0i16; n];
                            let sps_sl = ScalingListData::default();
                            let params = DequantParams {
                                bit_depth_luma: 8,
                                bit_depth_chroma: 8,
                                scaling_list_enabled: false,
                                sps_scaling_list: &sps_sl,
                                pps_scaling_list_present: false,
                                pps_scaling_list: &sps_sl,
                                cu_pred_mode: if cu_is_intra { PredMode::Intra } else { PredMode::Inter },
                            };
                            perform_dequant(&params, log2, c_idx, qp, &coeffs, &mut rust_out);
                            for v in &rust_out {
                                goldens::push_i16(&mut buf, *v);
                            }
                        }
                    }
                }
            }
        }
        vec![("transform::dequant_flat".to_string(), buf)]
    }

    #[test]
    fn dequant_matches_golden_flat() {
        for (key, data) in compute_dequant_flat() {
            goldens::assert_golden(&key, &data);
        }
    }

    fn compute_dequant_scaling_lists() -> Vec<(String, Vec<u8>)> {
        let mut buf = Vec::new();
        let mut rng = Rng::new(11);
        for log2 in 2..=5u32 {
            let n = (1 << log2) * (1 << log2);
            for c_idx in 0..3u32 {
                for bit_depth in [8u32, 10] {
                    for cu_is_intra in [false, true] {
                        for pps_present in [false, true] {
                            for _iter in 0..5 {
                                // Random scaling lists (full u8 range, including
                                // zeros to exercise the DC override m==0 -> 16).
                                let sl: ([u8; 1536], [u8; 12]) = (
                                    std::array::from_fn(|_| rng.below(256) as u8),
                                    std::array::from_fn(|i| {
                                        if i % 4 == 0 { 0 } else { rng.below(256) as u8 }
                                    }),
                                );
                                let pps_sl: ([u8; 1536], [u8; 12]) = (
                                    std::array::from_fn(|_| rng.below(256) as u8),
                                    std::array::from_fn(|_| rng.below(256) as u8),
                                );
                                let coeffs: Vec<i16> = (0..n).map(|_| random_coeff(&mut rng)).collect();

                                let sps_sl = fill_scaling_list(sl.0, sl.1);
                                let pps_sl_struct = fill_scaling_list(pps_sl.0, pps_sl.1);
                                let mut rust_out = vec![0i16; n];
                                let params = DequantParams {
                                    bit_depth_luma: bit_depth,
                                    bit_depth_chroma: bit_depth,
                                    scaling_list_enabled: true,
                                    sps_scaling_list: &sps_sl,
                                    pps_scaling_list_present: pps_present,
                                    pps_scaling_list: &pps_sl_struct,
                                    cu_pred_mode: if cu_is_intra { PredMode::Intra } else { PredMode::Inter },
                                };
                                perform_dequant(&params, log2, c_idx, 26, &coeffs, &mut rust_out);

                                for v in &rust_out {
                                    goldens::push_i16(&mut buf, *v);
                                }
                            }
                        }
                    }
                }
            }
        }
        vec![("transform::dequant_scaling_lists".to_string(), buf)]
    }

    #[test]
    fn dequant_matches_golden_scaling_lists() {
        for (key, data) in compute_dequant_scaling_lists() {
            goldens::assert_golden(&key, &data);
        }
    }

    fn compute_dequant_default_lists() -> Vec<(String, Vec<u8>)> {
        // Spec-default scaling lists (ScalingListData::set_defaults()).
        let mut buf = Vec::new();
        let mut rng = Rng::new(13);
        for log2 in 2..=5u32 {
            let n = (1 << log2) * (1 << log2);
            for c_idx in 0..3u32 {
                for cu_is_intra in [false, true] {
                    let coeffs: Vec<i16> = (0..n).map(|_| random_coeff(&mut rng)).collect();
                    let mut rust_out = vec![0i16; n];
                    let sl = {
                        let mut s = ScalingListData::default();
                        s.set_defaults();
                        s
                    };
                    let params = DequantParams {
                        bit_depth_luma: 8,
                        bit_depth_chroma: 8,
                        scaling_list_enabled: true,
                        sps_scaling_list: &sl,
                        pps_scaling_list_present: false,
                        pps_scaling_list: &sl,
                        cu_pred_mode: if cu_is_intra { PredMode::Intra } else { PredMode::Inter },
                    };
                    perform_dequant(&params, log2, c_idx, 34, &coeffs, &mut rust_out);
                    for v in &rust_out {
                        goldens::push_i16(&mut buf, *v);
                    }
                }
            }
        }
        vec![("transform::dequant_default_lists".to_string(), buf)]
    }

    #[test]
    fn dequant_matches_golden_default_lists() {
        for (key, data) in compute_dequant_default_lists() {
            goldens::assert_golden(&key, &data);
        }
    }

    #[test]
    fn transform_skip_identity_at_15bit() {
        // At 15-bit depth the transform-skip shift is 0: identity.
        let coeffs = [3i16, -4, 5, 6, 7, -8, 9, 10, 11, -12, 13, 14, 15, -16, 17, 18];
        let mut out = vec![0i16; 16];
        perform_transform_inverse(2, 1, false, true, 15, &coeffs, &mut out);
        assert_eq!(out, coeffs);
    }

    #[test]
    fn scaling_list_defaults_layout() {
        let mut sl = ScalingListData::default();
        sl.set_defaults();
        // sizeId 0: flat 16
        assert!(sl.scaling_list[0].iter().all(|&v| v == 16));
        // sizeId 1/2: intra for matrixId < 3, inter otherwise
        assert_eq!(sl.scaling_list[6], DEFAULT_8X8_INTRA);
        assert_eq!(sl.scaling_list[9], DEFAULT_8X8_INTER);
        assert_eq!(sl.scaling_list[12], DEFAULT_8X8_INTRA);
        assert_eq!(sl.scaling_list[15], DEFAULT_8X8_INTER);
        // sizeId 3: only matrixId 0 and 3 set
        assert_eq!(sl.scaling_list[18], DEFAULT_8X8_INTRA);
        assert_eq!(sl.scaling_list[21], DEFAULT_8X8_INTER);
        assert_eq!(sl.scaling_list[19], [0u8; 64]);
        assert!(sl.scaling_list_dc.iter().all(|&v| v == 16));
    }

    /// The AVX2 kernels must be bit-identical to the scalar butterflies.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_idct_matches_scalar() {
        if !detect_avx2() {
            return;
        }
        let mut rng = Rng::new(0xA7A2);
        for log2 in 2..=5u32 {
            let tr = 1usize << log2;
            for use_dst in [false, true] {
                if use_dst && log2 != 2 {
                    continue;
                }
                for shift in [7i32, 8, 10, 12] {
                    for iter in 0..25 {
                        let n = tr * tr;
                        let mut src = vec![0i16; n];
                        for s in src.iter_mut() {
                            *s = random_coeff(&mut rng);
                        }
                        let mut out_scalar = vec![0i16; n];
                        let mut out_avx2 = vec![0i16; n];
                        if use_dst {
                            idst4(&src, &mut out_scalar, shift, tr);
                        } else {
                            match log2 {
                                2 => idct4(&src, &mut out_scalar, shift, tr),
                                3 => idct8(&src, &mut out_scalar, shift, tr),
                                4 => idct16(&src, &mut out_scalar, shift, tr),
                                5 => idct32(&src, &mut out_scalar, shift, tr),
                                _ => unreachable!(),
                            }
                        }
                        unsafe {
                            if use_dst {
                                avx2::idst4(&src, &mut out_avx2, shift, tr);
                            } else {
                                match log2 {
                                    2 => avx2::idct4(&src, &mut out_avx2, shift, tr),
                                    3 => avx2::idct8(&src, &mut out_avx2, shift, tr),
                                    4 => avx2::idct16(&src, &mut out_avx2, shift, tr),
                                    5 => avx2::idct32(&src, &mut out_avx2, shift, tr),
                                    _ => unreachable!(),
                                }
                            }
                        }
                        assert_eq!(
                            out_scalar, out_avx2,
                            "avx2 mismatch: log2={} use_dst={} shift={} iter={}",
                            log2, use_dst, shift, iter
                        );
                    }
                }
            }
        }
    }

    pub(crate) fn golden_entries() -> Vec<(String, String)> {
        let mut v = Vec::new();
        for (k, b) in compute_transform_inverse() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        for (k, b) in compute_dequant_flat() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        for (k, b) in compute_dequant_scaling_lists() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        for (k, b) in compute_dequant_default_lists() {
            v.push((k, goldens::sha256_hex(&b)));
        }
        v
    }
}

//! Port of `hevc/decoding/transform.cpp` — inverse transform (§8.6.4) and
//! dequantization (§8.6.3).
//!
//! Bit-for-bit port of the hevc.js implementation: same butterfly structures,
//! same shift/rounding, same i32/i64 split in dequantization. Verified
//! differentially against the C++ core via the `hevcdec_test_*` FFI exports.

use crate::hevc::cabac_tables::LEVEL_SCALE;
use crate::hevc::types::{clip3, PredMode};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi_test;

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

    // ---- C++ oracle wrappers --------------------------------------------

    fn oracle_transform_inverse(
        log2: u32,
        c_idx: u32,
        is_intra: bool,
        skip: bool,
        bit_depth: u32,
        scaled: &[i16],
    ) -> Vec<i16> {
        let mut out = vec![0i16; scaled.len()];
        let rc = unsafe {
            ffi_test::hevcdec_test_transform_inverse(
                log2 as i32,
                c_idx as i32,
                is_intra as i32,
                skip as i32,
                bit_depth as i32,
                scaled.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0);
        out
    }

    #[allow(clippy::too_many_arguments)] // mirrors the C++ oracle signature
    fn oracle_dequant(
        bit_depth_luma: u32,
        bit_depth_chroma: u32,
        c_idx: u32,
        log2: u32,
        qp: i32,
        cu_is_intra: bool,
        use_sl: bool,
        pps_present: bool,
        sl: Option<(&[u8; 1536], &[u8; 12])>,
        pps_sl: Option<(&[u8; 1536], &[u8; 12])>,
        coefficients: &[i16],
    ) -> Vec<i16> {
        let mut out = vec![0i16; coefficients.len()];
        let rc = unsafe {
            ffi_test::hevcdec_test_dequant(
                bit_depth_luma as i32,
                bit_depth_chroma as i32,
                c_idx as i32,
                log2 as i32,
                qp,
                cu_is_intra as i32,
                use_sl as i32,
                pps_present as i32,
                sl.map_or(std::ptr::null(), |s| s.0.as_ptr()),
                sl.map_or(std::ptr::null(), |s| s.1.as_ptr()),
                pps_sl.map_or(std::ptr::null(), |s| s.0.as_ptr()),
                pps_sl.map_or(std::ptr::null(), |s| s.1.as_ptr()),
                coefficients.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0);
        out
    }

    // ---- Differential tests ----------------------------------------------

    #[test]
    fn transform_inverse_matches_cpp() {
        let mut rng = Rng::new(42);
        for log2 in 2..=5u32 {
            let n = (1 << log2) * (1 << log2);
            for c_idx in 0..3u32 {
                for is_intra in [false, true] {
                    for skip in [false, true] {
                        for bit_depth in [8u32, 10] {
                            for iter in 0..50 {
                                let coeffs: Vec<i16> = (0..n).map(|_| random_coeff(&mut rng)).collect();
                                let mut rust_out = vec![0i16; n];
                                perform_transform_inverse(
                                    log2, c_idx, is_intra, skip, bit_depth, &coeffs, &mut rust_out,
                                );
                                let cpp_out = oracle_transform_inverse(
                                    log2, c_idx, is_intra, skip, bit_depth, &coeffs,
                                );
                                assert_eq!(
                                    rust_out, cpp_out,
                                    "log2={log2} cIdx={c_idx} intra={is_intra} \
                                     skip={skip} bd={bit_depth} iter={iter}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn dequant_matches_cpp_flat() {
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
                        for iter in 0..10 {
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
                            let cpp_out = oracle_dequant(
                                8, 8, c_idx, log2, qp, cu_is_intra, false, false,
                                None, None, &coeffs,
                            );
                            assert_eq!(
                                rust_out, cpp_out,
                                "log2={log2} cIdx={c_idx} qp={qp} intra={cu_is_intra} iter={iter}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn dequant_matches_cpp_scaling_lists() {
        let mut rng = Rng::new(11);
        for log2 in 2..=5u32 {
            let n = (1 << log2) * (1 << log2);
            for c_idx in 0..3u32 {
                for bit_depth in [8u32, 10] {
                    for cu_is_intra in [false, true] {
                        for pps_present in [false, true] {
                            for iter in 0..5 {
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

                                let cpp_out = oracle_dequant(
                                    bit_depth, bit_depth, c_idx, log2, 26, cu_is_intra, true,
                                    pps_present,
                                    Some((&sl.0, &sl.1)),
                                    Some((&pps_sl.0, &pps_sl.1)),
                                    &coeffs,
                                );
                                assert_eq!(
                                    rust_out, cpp_out,
                                    "log2={log2} cIdx={c_idx} bd={bit_depth} \
                                     intra={cu_is_intra} pps={pps_present} iter={iter}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn dequant_matches_cpp_default_scaling_lists() {
        // NULL scaling list pointers -> C++ uses spec defaults; Rust uses
        // ScalingListData::set_defaults().
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
                    let cpp_out = oracle_dequant(
                        8, 8, c_idx, log2, 34, cu_is_intra, true, false, None, None, &coeffs,
                    );
                    assert_eq!(
                        rust_out, cpp_out,
                        "log2={log2} cIdx={c_idx} intra={cu_is_intra}"
                    );
                }
            }
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
}

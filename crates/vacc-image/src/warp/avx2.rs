//! AVX2 affine warp kernel (RGBA32, 8 pixels per iteration).
//!
//! Uses inverse mapping with bilinear interpolation. Coordinates are computed
//! in scalar code; the 8-pixel batches are laid out SIMD-ready for future
//! vectorization (interpolation is currently scalar).

use core::ptr;

use crate::warp::{Affine, round_clamp};

/// One output row (RGBA32) via AVX2-accelerated bilinear warp.
#[target_feature(enable = "avx2")]
pub unsafe fn warp_rgba_row(
    src_pixels: *const u8,
    src_pitch: usize,
    src_width: usize,
    src_height: usize,
    inv: Affine,
    oy: usize,
    dst_row: &mut [u8],
) {
    unsafe {
        let dst_width = dst_row.len() / 4;

        let mut x = 0usize;
        while x + 8 <= dst_width {
            // Compute inverse-mapped coordinates for 8 consecutive pixels.
            let mut coords = [[0.0f32; 2]; 8];
            for k in 0..8 {
                let ox = x + k;
                coords[k][0] = inv.m00 * ox as f32 + inv.m01 * oy as f32 + inv.m02;
                coords[k][1] = inv.m10 * ox as f32 + inv.m11 * oy as f32 + inv.m12;
            }

            // For each pixel, find the four source pixels and interpolation weights.
            let mut src_rows: [isize; 8] = [0; 8];
            let mut src_cols: [isize; 8] = [0; 8];
            let mut fx_vals: [f32; 8] = [0.0; 8];
            let mut fy_vals: [f32; 8] = [0.0; 8];

            for k in 0..8 {
                let sx = coords[k][0];
                let sy = coords[k][1];
                src_rows[k] = sy.floor() as isize;
                src_cols[k] = sx.floor() as isize;
                fx_vals[k] = sx - src_cols[k] as f32;
                fy_vals[k] = sy - src_rows[k] as f32;
            }

            // Load the four source pixels for each output pixel.
            // Each pixel is 4 bytes (RGBA), so we load 8x4=32 bytes per row.
            let mut p00 = [0u8; 32];
            let mut p10 = [0u8; 32];
            let mut p01 = [0u8; 32];
            let mut p11 = [0u8; 32];

            for k in 0..8 {
                let x0 = src_cols[k];
                let y0 = src_rows[k];
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                let load_pixel = |x: isize, y: isize, _offset: usize| -> [u8; 4] {
                    if x < 0 || y < 0 || x >= src_width as isize || y >= src_height as isize {
                        return [0, 0, 0, 255];
                    }
                    let row = src_pixels.add(y as usize * src_pitch);
                    let px = row.add(x as usize * 4);
                    let val = ptr::read_unaligned(px as *const u32);
                    val.to_le_bytes()
                };

                let l00 = load_pixel(x0, y0, k * 4);
                let l10 = load_pixel(x1, y0, k * 4);
                let l01 = load_pixel(x0, y1, k * 4);
                let l11 = load_pixel(x1, y1, k * 4);

                p00[k * 4..k * 4 + 4].copy_from_slice(&l00);
                p10[k * 4..k * 4 + 4].copy_from_slice(&l10);
                p01[k * 4..k * 4 + 4].copy_from_slice(&l01);
                p11[k * 4..k * 4 + 4].copy_from_slice(&l11);
            }

            // Bilinear interpolation for each pixel (scalar, but data is SIMD-ready).
            for k in 0..8 {
                let fx = fx_vals[k];
                let fy = fy_vals[k];
                let off = k * 4;

                let mut result = [0u8; 4];
                for c in 0..4 {
                    let top = p00[off + c] as f32 * (1.0 - fx) + p10[off + c] as f32 * fx;
                    let bot = p01[off + c] as f32 * (1.0 - fx) + p11[off + c] as f32 * fx;
                    result[c] = round_clamp(top * (1.0 - fy) + bot * fy);
                }
                dst_row[x * 4 + k * 4..x * 4 + k * 4 + 4].copy_from_slice(&result);
            }

            x += 8;
        }

        // Scalar tail for remaining pixels.
        while x < dst_width {
            let ox = x;
            let sx = inv.m00 * ox as f32 + inv.m01 * oy as f32 + inv.m02;
            let sy = inv.m10 * ox as f32 + inv.m11 * oy as f32 + inv.m12;

            let x0 = sx.floor() as isize;
            let y0 = sy.floor() as isize;
            let x1 = x0 + 1;
            let y1 = y0 + 1;
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            let load_pixel = |x: isize, y: isize| -> [u8; 4] {
                if x < 0 || y < 0 || x >= src_width as isize || y >= src_height as isize {
                    return [0, 0, 0, 255];
                }
                let row = src_pixels.add(y as usize * src_pitch);
                let px = row.add(x as usize * 4);
                let val = ptr::read_unaligned(px as *const u32);
                val.to_le_bytes()
            };

            let p00 = load_pixel(x0, y0);
            let p10 = load_pixel(x1, y0);
            let p01 = load_pixel(x0, y1);
            let p11 = load_pixel(x1, y1);

            for c in 0..4 {
                let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
                let bot = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
                dst_row[ox * 4 + c] = round_clamp(top * (1.0 - fy) + bot * fy);
            }
            x += 1;
        }
    }
}

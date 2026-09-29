//! Affine warping: rotation, translation, scaling, shearing.
//!
//! Applies a 2D affine transformation to an image using inverse mapping with
//! bilinear interpolation. The transformation is specified as a 2x3 matrix:
//!
//! ```text
//! [x']   [m00 m01 m02] [x]
//! [y'] = [m10 m11 m12] [y]
//! [ 1]   [ 0   0   1 ] [1]
//! ```
//!
//! The inverse is computed internally so that every output pixel maps to a
//! source position (no holes).

use crate::conv::{Kernel, parallel_rows, simd_features};
use crate::error::{ImageError, ImageResult};
use crate::pixel::RgbImage;
use crate::spec::Interpolation;

pub(crate) mod avx2;

/// A 2D affine transformation matrix (row-major).
///
/// Forward transform: `dst = M * src`, where `M` is:
/// ```text
/// [m00 m01 m02]
/// [m10 m11 m12]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Affine {
    /// Row 0: [scale_x, shear_y, tx]
    pub m00: f32,
    pub m01: f32,
    pub m02: f32,
    /// Row 1: [shear_x, scale_y, ty]
    pub m10: f32,
    pub m11: f32,
    pub m12: f32,
}

impl PartialEq for Affine {
    fn eq(&self, other: &Self) -> bool {
        self.m00.to_bits() == other.m00.to_bits()
            && self.m01.to_bits() == other.m01.to_bits()
            && self.m02.to_bits() == other.m02.to_bits()
            && self.m10.to_bits() == other.m10.to_bits()
            && self.m11.to_bits() == other.m11.to_bits()
            && self.m12.to_bits() == other.m12.to_bits()
    }
}

impl Eq for Affine {}

impl Affine {
    /// Identity transformation.
    pub const fn identity() -> Self {
        Self {
            m00: 1.0, m01: 0.0, m02: 0.0,
            m10: 0.0, m11: 1.0, m12: 0.0,
        }
    }

    /// Translation by (tx, ty).
    pub const fn translate(tx: f32, ty: f32) -> Self {
        Self {
            m00: 1.0, m01: 0.0, m02: tx,
            m10: 0.0, m11: 1.0, m12: ty,
        }
    }

    /// Uniform scaling by factor `s`.
    pub const fn scale(s: f32) -> Self {
        Self {
            m00: s, m01: 0.0, m02: 0.0,
            m10: 0.0, m11: s, m12: 0.0,
        }
    }

    /// Non-uniform scaling by (sx, sy).
    pub const fn scale_xy(sx: f32, sy: f32) -> Self {
        Self {
            m00: sx, m01: 0.0, m02: 0.0,
            m10: 0.0, m11: sy, m12: 0.0,
        }
    }

    /// Rotation by `theta` radians (counter-clockwise).
    pub fn rotate(theta: f32) -> Self {
        let c = theta.cos();
        let s = theta.sin();
        Self {
            m00: c, m01: -s, m02: 0.0,
            m10: s, m11: c, m12: 0.0,
        }
    }

    /// Rotation by `theta` radians around center (cx, cy).
    pub fn rotate_around(theta: f32, cx: f32, cy: f32) -> Self {
        let t = Self::translate(cx, cy);
        let r = Self::rotate(theta);
        let t_inv = Self::translate(-cx, -cy);
        t * r * t_inv
    }

    /// Matrix multiplication: `self * other`.
    pub fn mul(&self, other: &Affine) -> Affine {
        Self {
            m00: self.m00 * other.m00 + self.m01 * other.m10,
            m01: self.m00 * other.m01 + self.m01 * other.m11,
            m02: self.m00 * other.m02 + self.m01 * other.m12 + self.m02,
            m10: self.m10 * other.m00 + self.m11 * other.m10,
            m11: self.m10 * other.m01 + self.m11 * other.m11,
            m12: self.m10 * other.m02 + self.m11 * other.m12 + self.m12,
        }
    }

    /// Compute the inverse of this affine transformation.
    pub fn invert(&self) -> Option<Affine> {
        let det = self.m00 * self.m11 - self.m01 * self.m10;
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Affine {
            m00: self.m11 * inv_det,
            m01: -self.m01 * inv_det,
            m02: (self.m01 * self.m12 - self.m11 * self.m02) * inv_det,
            m10: -self.m10 * inv_det,
            m11: self.m00 * inv_det,
            m12: (self.m10 * self.m02 - self.m00 * self.m12) * inv_det,
        })
    }
}

impl std::ops::Mul for Affine {
    type Output = Affine;
    fn mul(self, rhs: Affine) -> Affine {
        // Matrix multiplication for 2x3 affine transforms
        // [a b c]   [e f g]   [ae+bf ag+bh ai+bj+bk]
        // [d e f] * [h i j] = [de+ei dg+ej di+ej+fj]
        Affine {
            m00: self.m00 * rhs.m00 + self.m01 * rhs.m10,
            m01: self.m00 * rhs.m01 + self.m01 * rhs.m11,
            m02: self.m00 * rhs.m02 + self.m01 * rhs.m12 + self.m02,
            m10: self.m10 * rhs.m00 + self.m11 * rhs.m10,
            m11: self.m10 * rhs.m01 + self.m11 * rhs.m11,
            m12: self.m10 * rhs.m02 + self.m11 * rhs.m12 + self.m12,
        }
    }
}

/// Round half up + clamp to [0, 255] (matches the resize kernels).
#[inline]
fn round_clamp(v: f32) -> u8 {
    (v + 0.5).floor().clamp(0.0, 255.0) as u8
}

/// Warp an RGB image using the given affine transformation.
///
/// `dst` must be large enough to hold `width * height * channels` bytes.
/// Pixels that map outside the source are filled with black; 4-channel
/// output always carries alpha 0xFF. [`Interpolation::Box`] is rejected:
/// area coverage is undefined under a general affine map.
pub fn warp_rgb(
    src: &RgbImage,
    transform: Affine,
    interpolation: Interpolation,
    width: usize,
    height: usize,
    dst: &mut [u8],
) -> ImageResult<()> {
    if !interpolation.supports_warp() {
        return Err(ImageError::Unsupported(
            "box interpolation is not supported for affine warp".into(),
        ));
    }
    if width == 0 || height == 0 {
        return Err(ImageError::InvalidDimensions("output size must be non-zero".into()));
    }
    let ch = src.channels as usize;
    if ch != 3 && ch != 4 {
        return Err(ImageError::Unsupported(format!("unsupported channel count {ch}")));
    }
    let need = width * height * ch;
    if dst.len() < need {
        return Err(ImageError::OutputTooSmall { need, have: dst.len() });
    }

    // Compute inverse for inverse mapping.
    let inv = transform
        .invert()
        .ok_or_else(|| ImageError::Unsupported("singular affine matrix".into()))?;

    // One band per output row.
    let mut bands: Vec<&mut [u8]> = Vec::with_capacity(height);
    let mut rest = dst;
    for _ in 0..height {
        let (a, b) = rest.split_at_mut(width * ch);
        bands.push(a);
        rest = b;
    }

    // `Kernel::Avx2` is only ever reported on x86_64 (see `simd_features`),
    // so the guard below falls through to the scalar path elsewhere.
    let simd = pick_kernel(Kernel::Auto);
    let cl = |a: usize, rows: &mut [&mut [u8]]| {
        for (i, row) in rows.iter_mut().enumerate() {
            let oy = a + i;
            match interpolation {
                Interpolation::Bilinear if ch == 4 && matches!(simd, Some(Kernel::Avx2)) => {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        avx2::warp_rgba_row(
                            src.pixels_ptr(),
                            src.pitch,
                            src.width,
                            src.height,
                            inv,
                            oy,
                            row,
                        );
                    }
                }
                Interpolation::Bilinear => scalar_warp_row(
                    src.pixels_ptr(),
                    src.pitch,
                    src.width,
                    src.height,
                    ch,
                    inv,
                    oy,
                    row,
                ),
                Interpolation::Nearest => scalar_warp_row_nearest(
                    src.pixels_ptr(),
                    src.pitch,
                    src.width,
                    src.height,
                    ch,
                    inv,
                    oy,
                    row,
                ),
                Interpolation::Bicubic => scalar_warp_row_bicubic(
                    src.pixels_ptr(),
                    src.pitch,
                    src.width,
                    src.height,
                    ch,
                    inv,
                    oy,
                    row,
                ),
                Interpolation::Box => unreachable!("rejected above"),
            }
        }
    };
    parallel_rows(&mut bands, &cl);
    Ok(())
}

/// Pick the SIMD kernel (mirrors conv::convert_rows semantics).
fn pick_kernel(kernel: Kernel) -> Option<Kernel> {
    let simd = simd_features();
    match kernel {
        Kernel::Auto => {
            if simd.avx2 {
                Some(Kernel::Avx2)
            } else if simd.sse41 {
                Some(Kernel::Sse)
            } else {
                None
            }
        }
        Kernel::Sse => simd.sse41.then_some(Kernel::Sse),
        Kernel::Avx2 => simd.avx2.then_some(Kernel::Avx2),
        Kernel::Scalar => None,
    }
}

/// Load one source pixel as RGBA floats; out-of-bounds reads are zero
/// (alpha 0). For 3-channel sources alpha is synthesized as 255.
#[inline]
fn sample_pixel(
    src_pixels: *const u8,
    src_pitch: usize,
    src_width: usize,
    src_height: usize,
    ch: usize,
    x: i32,
    y: i32,
) -> [f32; 4] {
    if x < 0 || y < 0 || x >= src_width as i32 || y >= src_height as i32 {
        return [0.0; 4];
    }
    let row = unsafe { src_pixels.add(y as usize * src_pitch) };
    let px = unsafe { row.add(x as usize * ch) };
    let mut vals = [0.0f32; 4];
    for c in 0..ch {
        vals[c] = unsafe { *px.add(c) } as f32;
    }
    if ch == 3 {
        vals[3] = 255.0;
    }
    vals
}

/// Scalar implementation: one output row via inverse mapping + bilinear interpolation.
fn scalar_warp_row(
    src_pixels: *const u8,
    src_pitch: usize,
    src_width: usize,
    src_height: usize,
    ch: usize,
    inv: Affine,
    oy: usize,
    dst_row: &mut [u8],
) {
    let dst_width = dst_row.len() / ch;
    for ox in 0..dst_width {
        // Inverse map: find source position for this output pixel.
        let sx = inv.m00 * ox as f32 + inv.m01 * oy as f32 + inv.m02;
        let sy = inv.m10 * ox as f32 + inv.m11 * oy as f32 + inv.m12;

        // Bilinear interpolation.
        let x0 = sx.floor() as i32;
        let y0 = sy.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let fx = sx - x0 as f32;
        let fy = sy - y0 as f32;

        let p00 = sample_pixel(src_pixels, src_pitch, src_width, src_height, ch, x0, y0);
        let p10 = sample_pixel(src_pixels, src_pitch, src_width, src_height, ch, x1, y0);
        let p01 = sample_pixel(src_pixels, src_pitch, src_width, src_height, ch, x0, y1);
        let p11 = sample_pixel(src_pixels, src_pitch, src_width, src_height, ch, x1, y1);

        let mut result = [0.0; 4];
        for c in 0..ch {
            let top = p00[c] * (1.0 - fx) + p10[c] * fx;
            let bot = p01[c] * (1.0 - fx) + p11[c] * fx;
            result[c] = top * (1.0 - fy) + bot * fy;
        }
        if ch == 3 {
            result[3] = 255.0;
        }

        let dst_px = &mut dst_row[ox * ch..ox * ch + ch];
        for c in 0..ch {
            dst_px[c] = round_clamp(result[c]);
        }
        if ch == 4 {
            dst_row[ox * 4 + 3] = 255;
        }
    }
}

/// Scalar implementation: one output row via inverse mapping + nearest
/// neighbor (the floor of the source position).
fn scalar_warp_row_nearest(
    src_pixels: *const u8,
    src_pitch: usize,
    src_width: usize,
    src_height: usize,
    ch: usize,
    inv: Affine,
    oy: usize,
    dst_row: &mut [u8],
) {
    let dst_width = dst_row.len() / ch;
    for ox in 0..dst_width {
        let sx = inv.m00 * ox as f32 + inv.m01 * oy as f32 + inv.m02;
        let sy = inv.m10 * ox as f32 + inv.m11 * oy as f32 + inv.m12;
        let p = sample_pixel(
            src_pixels, src_pitch, src_width, src_height, ch,
            sx.floor() as i32,
            sy.floor() as i32,
        );
        let dst_px = &mut dst_row[ox * ch..ox * ch + ch];
        for c in 0..ch {
            dst_px[c] = round_clamp(p[c]);
        }
        if ch == 4 {
            dst_row[ox * 4 + 3] = 255;
        }
    }
}

/// Mitchell-Netravali cubic (B = C = 0.5) at distance `d >= 0`, f32 variant
/// (mirrors `resize::mitchell`).
#[inline]
fn mitchell(d: f32) -> f32 {
    if d < 1.0 {
        (4.5 * d * d * d - 9.0 * d * d + 5.0) / 6.0
    } else if d < 2.0 {
        (-3.5 * d * d * d + 18.0 * d * d - 30.0 * d + 16.0) / 6.0
    } else {
        0.0
    }
}

/// Scalar implementation: one output row via inverse mapping + separable
/// 4x4 cubic convolution (Mitchell, B = C = 0.5). Out-of-bounds taps
/// contribute zero, so edges fade to black like the bilinear path.
fn scalar_warp_row_bicubic(
    src_pixels: *const u8,
    src_pitch: usize,
    src_width: usize,
    src_height: usize,
    ch: usize,
    inv: Affine,
    oy: usize,
    dst_row: &mut [u8],
) {
    let dst_width = dst_row.len() / ch;
    for ox in 0..dst_width {
        let sx = inv.m00 * ox as f32 + inv.m01 * oy as f32 + inv.m02;
        let sy = inv.m10 * ox as f32 + inv.m11 * oy as f32 + inv.m12;
        let x0 = sx.floor() as i32;
        let y0 = sy.floor() as i32;

        let mut result = [0.0f32; 4];
        for kx in -1i32..=2 {
            let wx = mitchell((sx - (x0 + kx) as f32).abs());
            if wx == 0.0 {
                continue;
            }
            for ky in -1i32..=2 {
                let wy = mitchell((sy - (y0 + ky) as f32).abs());
                if wy == 0.0 {
                    continue;
                }
                let p = sample_pixel(src_pixels, src_pitch, src_width, src_height, ch, x0 + kx, y0 + ky);
                let w = wx * wy;
                for c in 0..ch {
                    result[c] += p[c] * w;
                }
            }
        }

        let dst_px = &mut dst_row[ox * ch..ox * ch + ch];
        for c in 0..ch {
            dst_px[c] = round_clamp(result[c]);
        }
        if ch == 4 {
            dst_row[ox * 4 + 3] = 255;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::RgbImage;

    fn make_test_image(w: usize, h: usize) -> (Vec<u8>, RgbImage<'static>) {
        let mut buf = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 3;
                buf[p] = ((x * 255) / w.saturating_sub(1).max(1)) as u8;
                buf[p + 1] = ((y * 255) / h.saturating_sub(1).max(1)) as u8;
                buf[p + 2] = 128;
            }
        }
        let data: &'static [u8] = Box::leak(buf.clone().into_boxed_slice());
        (buf, RgbImage::new(data, w * 3, w, h, 3))
    }

    #[test]
    fn identity_warp_is_exact() {
        let (_, img) = make_test_image(100, 60);
        let mut out = vec![0u8; 100 * 60 * 3];
        // Identity should reproduce the source (within interpolation tolerance).
        warp_rgb(&img, Affine::identity(), Interpolation::Bilinear, 100, 60, &mut out).unwrap();
        assert_eq!(out.len(), img.pixels().len());
    }

    #[test]
    fn nearest_identity_is_exact() {
        let (buf, img) = make_test_image(100, 60);
        let mut out = vec![0u8; 100 * 60 * 3];
        warp_rgb(&img, Affine::identity(), Interpolation::Nearest, 100, 60, &mut out).unwrap();
        assert_eq!(out, buf);
    }

    #[test]
    fn nearest_integer_translate_is_exact() {
        let (buf, img) = make_test_image(100, 60);
        let mut out = vec![0u8; 100 * 60 * 3];
        warp_rgb(&img, Affine::translate(10.0, 0.0), Interpolation::Nearest, 100, 60, &mut out)
            .unwrap();
        // Nearest + integer shift: each output pixel is the exact source
        // pixel shifted by 10 (left strip black).
        for y in 0..60 {
            assert!(out[y * 300..y * 300 + 30].iter().all(|&v| v == 0));
            for x in 10..100 {
                let s = &buf[(y * 100 + (x - 10)) * 3..(y * 100 + (x - 10)) * 3 + 3];
                assert_eq!(&out[(y * 100 + x) * 3..(y * 100 + x) * 3 + 3], s);
            }
        }
    }

    #[test]
    fn bicubic_identity_stays_close_to_source() {
        let (buf, img) = make_test_image(100, 60);
        let mut out = vec![0u8; 100 * 60 * 3];
        warp_rgb(&img, Affine::identity(), Interpolation::Bicubic, 100, 60, &mut out).unwrap();
        // Interior pixels of a smooth gradient stay within +-2 of the source;
        // edge rows fade slightly toward black (OOB taps are zero).
        for y in 1..59 {
            for x in 1..99 {
                let p = (y * 100 + x) * 3;
                for c in 0..3 {
                    assert!(
                        (out[p + c] as i32 - buf[p + c] as i32).abs() <= 2,
                        "drift at ({x}, {y}) ch {c}: {} vs {}",
                        out[p + c],
                        buf[p + c]
                    );
                }
            }
        }
    }

    #[test]
    fn box_interpolation_rejected() {
        let (_, img) = make_test_image(16, 8);
        let mut out = vec![0u8; 16 * 8 * 3];
        let res = warp_rgb(&img, Affine::identity(), Interpolation::Box, 16, 8, &mut out);
        assert!(matches!(res, Err(ImageError::Unsupported(_))));
    }

    #[test]
    fn translate_warp() {
        let (_, img) = make_test_image(100, 60);
        let mut out = vec![0u8; 100 * 60 * 3];
        // Translate right by 10 pixels.
        warp_rgb(&img, Affine::translate(10.0, 0.0), Interpolation::Bilinear, 100, 60, &mut out)
            .unwrap();
        // The left edge should now be black (mapped outside source).
        assert_eq!(out[0], 0);
        // The right edge should have the original left-edge color.
        let right_edge_x = 89;
        let src_color = ((right_edge_x * 255) / 99) as u8;
        assert!((out[(100 - 1) * 3] as i32 - src_color as i32).abs() <= 2);
    }

    #[test]
    fn rotate_warp_smoke() {
        let (_, img) = make_test_image(100, 100);
        let center = Affine::rotate_around(std::f32::consts::PI / 4.0, 50.0, 50.0);
        let mut out = vec![0u8; 100 * 100 * 3];
        warp_rgb(&img, center, Interpolation::Bilinear, 100, 100, &mut out).unwrap();
        // Just check it doesn't crash and produces non-uniform output.
        assert!(out.iter().any(|&v| v > 0));
    }

    #[test]
    fn scale_up_warp() {
        let (_, img) = make_test_image(50, 50);
        let scale = Affine::scale(2.0);
        let mut out = vec![0u8; 100 * 100 * 3];
        warp_rgb(&img, scale, Interpolation::Bilinear, 100, 100, &mut out).unwrap();
        // Upscaled image should be non-zero in the top-left quadrant.
        let center_val = out[50 * 100 * 3 + 50 * 3];
        assert!(center_val > 0);
    }

    #[test]
    fn scale_down_warp() {
        let (_, img) = make_test_image(100, 100);
        let scale = Affine::scale(0.5);
        let mut out = vec![0u8; 50 * 50 * 3];
        warp_rgb(&img, scale, Interpolation::Bilinear, 50, 50, &mut out).unwrap();
        // Downscaled image should preserve the gradient direction.
        let top_left = out[0];
        let bottom_right = out[(49 * 50 + 49) * 3];
        assert!(bottom_right > top_left);
    }

    #[test]
    fn shear_x_warp() {
        let (_, img) = make_test_image(100, 50);
        // Shear in x direction: x' = x + 0.5*y
        let shear = Affine {
            m00: 1.0, m01: 0.5, m02: 0.0,
            m10: 0.0, m11: 1.0, m12: 0.0,
        };
        let mut out = vec![0u8; 100 * 50 * 3];
        warp_rgb(&img, shear, Interpolation::Bilinear, 100, 50, &mut out).unwrap();
        // Sheared image should be non-uniform.
        assert!(out.iter().any(|&v| v > 0));
    }

    #[test]
    fn combined_translate_rotate() {
        let (_, img) = make_test_image(100, 100);
        let translate = Affine::translate(10.0, 5.0);
        let rotate = Affine::rotate_around(std::f32::consts::PI / 6.0, 50.0, 50.0);
        let combined = translate * rotate;
        let mut out = vec![0u8; 100 * 100 * 3];
        warp_rgb(&img, combined, Interpolation::Bilinear, 100, 100, &mut out).unwrap();
        assert!(out.iter().any(|&v| v > 0));
    }

    fn approx_eq(a: Affine, b: Affine, tol: f32) -> bool {
        (a.m00 - b.m00).abs() < tol && (a.m01 - b.m01).abs() < tol && (a.m02 - b.m02).abs() < tol
            && (a.m10 - b.m10).abs() < tol && (a.m11 - b.m11).abs() < tol && (a.m12 - b.m12).abs() < tol
    }

    #[test]
    fn invert_identity() {
        let inv = Affine::identity().invert();
        assert!(inv.is_some());
        assert!(approx_eq(inv.unwrap(), Affine::identity(), 1e-6));
    }

    #[test]
    fn invert_translate() {
        let t = Affine::translate(10.0, 20.0);
        let inv = t.invert();
        assert!(inv.is_some());
        // Inverse of translate(10, 20) is translate(-10, -20).
        let expected = Affine::translate(-10.0, -20.0);
        assert!(approx_eq(inv.unwrap(), expected, 1e-6));
    }

    #[test]
    fn invert_scale() {
        let s = Affine::scale(2.0);
        let inv = s.invert();
        assert!(inv.is_some());
        // Inverse of scale(2) is scale(0.5).
        let expected = Affine::scale(0.5);
        assert!(approx_eq(inv.unwrap(), expected, 1e-6));
    }

    #[test]
    fn invert_rotation() {
        let theta = std::f32::consts::PI / 4.0;
        let r = Affine::rotate(theta);
        let inv = r.invert();
        assert!(inv.is_some());
        // Inverse of rotate(theta) is rotate(-theta).
        let expected = Affine::rotate(-theta);
        // Compare with tolerance due to floating point.
        let inv_r = inv.unwrap();
        assert!((inv_r.m00 - expected.m00).abs() < 1e-6);
        assert!((inv_r.m01 - expected.m01).abs() < 1e-6);
        assert!((inv_r.m10 - expected.m10).abs() < 1e-6);
        assert!((inv_r.m11 - expected.m11).abs() < 1e-6);
    }

    #[test]
    fn invert_composition() {
        let t = Affine::translate(5.0, 10.0);
        let s = Affine::scale(2.0);
        let combined = t * s;
        let inv = combined.invert();
        assert!(inv.is_some());
        // (T*S)^-1 = S^-1 * T^-1
        let expected = s.invert().unwrap() * t.invert().unwrap();
        let inv_c = inv.unwrap();
        assert!((inv_c.m00 - expected.m00).abs() < 1e-6);
        assert!((inv_c.m01 - expected.m01).abs() < 1e-6);
        assert!((inv_c.m02 - expected.m02).abs() < 1e-6);
        assert!((inv_c.m10 - expected.m10).abs() < 1e-6);
        assert!((inv_c.m11 - expected.m11).abs() < 1e-6);
        assert!((inv_c.m12 - expected.m12).abs() < 1e-6);
    }

    #[test]
    fn small_image_warp() {
        let (_, img) = make_test_image(4, 4);
        let scale = Affine::scale(2.0);
        let mut out = vec![0u8; 8 * 8 * 3];
        warp_rgb(&img, scale, Interpolation::Bilinear, 8, 8, &mut out).unwrap();
        assert!(out.iter().any(|&v| v > 0));
    }

    #[test]
    fn output_size_validation() {
        let (_, img) = make_test_image(100, 60);
        let mut too_small = vec![0u8; 1000];
        let result = warp_rgb(&img, Affine::identity(), Interpolation::Bilinear, 100, 60, &mut too_small);
        assert!(result.is_err());
    }

    #[test]
    fn singular_matrix_invert() {
        // Zero determinant matrix.
        let singular = Affine {
            m00: 0.0, m01: 0.0, m02: 0.0,
            m10: 0.0, m11: 0.0, m12: 0.0,
        };
        let inv = singular.invert();
        assert!(inv.is_none());
    }
}

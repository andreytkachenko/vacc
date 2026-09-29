//! Image scaling: scalar reference + SIMD dispatch.
//!
//! The public entry points are [`resize_rgb`] (packed RGB24 / RGBA32) and
//! [`resize_yuv`] (8-bit 4:2:0, planar or semi-planar). Both implement the
//! interpolations from [`Interpolation`]:
//!
//! - `Nearest`: single tap at the rounded source position.
//! - `Bilinear`: two-tap linear interpolation.
//! - `Box`: area (coverage) averaging — proper anti-aliasing on downscale,
//!   partial-coverage weights on upscale.
//! - `Bicubic`: Mitchell (B = 0.5, C = 0.5); the kernel is widened by
//!   `1 / scale` on downscale for anti-aliasing.
//!
//! ## Implementation notes
//!
//! Scaling is separable: a horizontal pass runs over every source row into a
//! scratch buffer, then a vertical pass blends scratch rows into the output.
//! Per-axis tap tables (positions + `f32` weights, uniform tap count with
//! zero-weight padding) are computed **once** in `f64` and consumed by every
//! kernel, so scalar / SSE / AVX2 results are byte-identical.
//!
//! The vertical pass vectorizes over output pixels (weights are uniform per
//! tap row); the horizontal pass is scalar except for an AVX2 gather-based
//! fast path for RGBA32. Edge handling replicates the border pixel.

use crate::conv::{Kernel, parallel_rows, simd_features};
use crate::error::{ImageError, ImageResult};
use crate::pixel::{YuvImage};
use crate::spec::{Interpolation, Scale};

pub(crate) mod avx2;
pub(crate) mod sse4;

/// Maximum taps kept per output pixel (pathological downscale ratios are
/// clamped here and renormalized).
const MAX_TAPS: usize = 64;

/// Per-axis tap table shared by all kernels.
///
/// For output pixel `ox`, tap `k` reads source position `pos[ox * n_taps + k]`
/// with weight `weights[ox * n_taps + k]`. The tap count is uniform across
/// pixels (short rows are padded with zero-weight taps at valid positions),
/// which keeps the SIMD loops branch-free.
pub(crate) struct TapAxis {
    pub(crate) pos: Vec<i32>,
    pub(crate) weights: Vec<f32>,
    pub(crate) n_taps: usize,
    /// Transposed tap tables for the AVX2 gather H-pass, indexed
    /// `[k * len + ox]` so a fixed tap's positions/weights are contiguous
    /// across output pixels (the gather loads 8 in a row).
    pub(crate) pos_t: Vec<i32>,
    pub(crate) weights_t: Vec<f32>,
}

impl TapAxis {
    /// Build the tap table for one axis of `src` -> `dst` samples.
    fn build(filter: Interpolation, src: usize, dst: usize) -> Self {
        assert!(src >= 1 && dst >= 1);
        let scale = src as f64 / dst as f64;
        // Exact even-integer-ratio downscale (4x, 6x, ...) degenerates the
        // cubic discrete sum to zero: the kernel's zero crossings land
        // exactly on every candidate tap (d = |c-t|*s is a multiple of
        // s/2 >= 2), so no tap survives and the output turns black. The
        // correct area-resampling limit at integer ratio is a box of width
        // `scale`; fall back to its coverage weights there. (At 2x this
        // matches what the renormalized cubic already produced.)
        let filter = match filter {
            Interpolation::Bicubic
                if scale >= 2.0
                    && (scale - scale.round()).abs() < 1e-6
                    && scale.round() % 2.0 == 0.0 =>
            {
                Interpolation::Box
            }
            f => f,
        };
        let mut rows: Vec<Vec<(i32, f64)>> = (0..dst)
            .map(|ox| {
                let center = (ox as f64 + 0.5) * scale - 0.5;
                match filter {
                    Interpolation::Nearest => {
                        vec![(
                            center.round().clamp(0.0, src as f64 - 1.0) as i32,
                            1.0,
                        )]
                    }
                    Interpolation::Bilinear => {
                        let i0 = center.floor() as i32;
                        let f = center - center.floor();
                        vec![
                            (i0.clamp(0, src as i32 - 1), 1.0 - f),
                            ((i0 + 1).clamp(0, src as i32 - 1), f),
                        ]
                    }
                    Interpolation::Box => {
                        let half = scale / 2.0;
                        let lo = center - half;
                        let hi = center + half;
                        let t_min = lo.floor() as i32;
                        let t_max = hi.ceil() as i32;
                        (t_min..=t_max)
                            .filter(|&t| {
                                let a = (t as f64 - 0.5).max(lo);
                                let b = (t as f64 + 0.5).min(hi);
                                b > a
                            })
                            .map(|t| {
                                let a = (t as f64 - 0.5).max(lo);
                                let b = (t as f64 + 0.5).min(hi);
                                (t.clamp(0, src as i32 - 1), b - a)
                            })
                            .collect()
                    }
                    Interpolation::Bicubic => {
                        // The kernel is evaluated in destination units
                        // (d = |c-t|*scale, support 2), so the tap range in
                        // source space is always 2/scale — narrower than 2
                        // when downscaling.
                        let support = 2.0 / scale;
                        let t_min = (center - support).floor() as i32;
                        let t_max = (center + support).ceil() as i32;
                        (t_min..=t_max)
                            .filter_map(|t| {
                                let d = ((center - t as f64) * scale).abs();
                                (d < 2.0).then(|| {
                                    (t.clamp(0, src as i32 - 1), mitchell(d) / scale)
                                })
                            })
                            .collect()
                    }
                }
            })
            .collect();

        // Renormalize the filters whose raw weights do not sum to exactly 1
        // (box coverage, widened cubic). Bilinear is exact as-is.
        if matches!(filter, Interpolation::Box | Interpolation::Bicubic) {
            for row in rows.iter_mut() {
                let sum: f64 = row.iter().map(|&(_, w)| w).sum();
                if sum > 0.0 {
                    for (_, w) in row.iter_mut() {
                        *w /= sum;
                    }
                }
            }
        }

        let n_taps = rows
            .iter()
            .map(|r| r.len().min(MAX_TAPS))
            .max()
            .unwrap_or(1)
            .max(1);
        let mut pos = Vec::with_capacity(dst * n_taps);
        let mut weights = Vec::with_capacity(dst * n_taps);
        for row in &rows {
            let last = row.last().map(|&(p, _)| p).unwrap_or(0);
            for k in 0..n_taps {
                match row.get(k) {
                    Some(&(p, w)) => {
                        pos.push(p);
                        weights.push(w as f32);
                    }
                    None => {
                        // Zero-weight padding must stay at a valid position:
                        // the AVX2 gather path still reads it.
                        pos.push(last);
                        weights.push(0.0);
                    }
                }
            }
        }
        let len = dst;
        let mut pos_t = vec![0i32; n_taps * len];
        let mut weights_t = vec![0f32; n_taps * len];
        for ox in 0..len {
            for k in 0..n_taps {
                pos_t[k * len + ox] = pos[ox * n_taps + k];
                weights_t[k * len + ox] = weights[ox * n_taps + k];
            }
        }
        Self {
            pos,
            weights,
            n_taps,
            pos_t,
            weights_t,
        }
    }
}

/// Mitchell-Netravali cubic (B = C = 0.5) at distance `d >= 0`.
#[inline]
fn mitchell(d: f64) -> f64 {
    if d < 1.0 {
        // ((12 - 9B - 6C) d^3 + (-18 + 12B + 6C) d^2 + (6 - 2B)) / 6, B=C=0.5
        (4.5 * d * d * d - 9.0 * d * d + 5.0) / 6.0
    } else if d < 2.0 {
        // ((-B - 6C) d^3 + (6B + 30C) d^2 - (12B - 48C) d + (8B + 24C)) / 6
        (-3.5 * d * d * d + 18.0 * d * d - 30.0 * d + 16.0) / 6.0
    } else {
        0.0
    }
}

// ─────────────────────────── row views ───────────────────────────

/// A row-major plane with an explicit pitch (tight when `pitch == width*ch`).
struct PlaneRows {
    base: *const u8,
    pitch: usize,
    width: usize,
    height: usize,
    ch: usize, // bytes per sample/pixel (1 for YUV planes, 3 or 4 for RGB)
}

// SAFETY: `base` points to immutable caller-owned data that outlives every
// borrow of `PlaneRows`; sharing `&PlaneRows` across threads only reads it.
unsafe impl Sync for PlaneRows {}

impl PlaneRows {
    #[inline]
    fn row(&self, y: usize) -> &[u8] {
        let start = unsafe { self.base.add(y * self.pitch) };
        // SAFETY: the caller guarantees `height` rows of `pitch` bytes.
        unsafe { std::slice::from_raw_parts(start, self.pitch) }
    }

    /// One tight row of `width * ch` bytes.
    #[inline]
    fn tight_row(&self, y: usize) -> &[u8] {
        &self.row(y)[..self.width * self.ch]
    }
}

// ─────────────────────────── scalar passes ───────────────────────────

/// Round half up (`floor(x + 0.5)`) + clamp to [0, 255]; mirrors the SIMD
/// kernels' `floor_ps(add_ps(v, 0.5))` followed by min/max.
#[inline]
pub(crate) fn round_clamp(v: f32) -> u8 {
    (v + 0.5).floor().clamp(0.0, 255.0) as u8
}

/// One horizontal-pass output pixel (scalar reference; also used as the tail
/// of the SIMD rows, so results stay byte-identical).
#[inline]
pub(crate) fn h_pass_px(
    srow: &[u8],
    taps: &TapAxis,
    ox: usize,
    ch: usize,
    drow: &mut [u8],
) {
    let (prow, wrow) = taps.row(ox);
    let mut acc: [f32; 4] = [0.0; 4];
    for k in 0..taps.n_taps {
        let p = prow[k] as usize * ch;
        let w = wrow[k];
        for c in 0..ch {
            acc[c] += w * srow[p + c] as f32;
        }
    }
    for c in 0..ch {
        drow[ox * ch + c] = round_clamp(acc[c]);
    }
}

/// Horizontal pass: `src.height` rows of `src.width` samples -> tight scratch
/// of `taps.len()` columns x `src.height` rows. `taps` maps output column ->
/// source taps.
fn h_pass_scalar(src: &PlaneRows, taps: &TapAxis, tmp: &mut [u8]) {
    let dw = taps.pos.len() / taps.n_taps;
    for oy in 0..src.height {
        let srow = src.tight_row(oy);
        let drow = &mut tmp[oy * dw * src.ch..oy * dw * src.ch + dw * src.ch];
        for ox in 0..dw {
            h_pass_px(srow, taps, ox, src.ch, drow);
        }
    }
}



impl TapAxis {
    /// (tap positions, weights) for output pixel `ox`.
    #[inline]
    fn row(&self, ox: usize) -> (&[i32], &[f32]) {
        let o = ox * self.n_taps;
        (&self.pos[o..o + self.n_taps], &self.weights[o..o + self.n_taps])
    }

    /// Number of output pixels in this table.
    #[inline]
    fn len(&self) -> usize {
        self.pos.len() / self.n_taps
    }
}

// ─────────────────────────── SIMD dispatch ───────────────────────────

/// Pick the SIMD kernel to run (mirrors `conv::convert_rows` semantics:
/// `None` means scalar).
fn pick(kernel: Kernel) -> Option<Kernel> {
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

/// Vertical pass over `n_out` output rows: SIMD when available for this
/// channel count, scalar otherwise. `src` rows are tight (`width * ch`).
fn v_pass(
    src: &PlaneRows,
    taps: &TapAxis,
    width: usize,
    ch: usize,
    kernel: Kernel,
    dst: &mut [u8],
) {
    let n = taps.len();
    // One band per output row.
    let mut bands: Vec<&mut [u8]> = Vec::with_capacity(n);
    let mut rest = dst;
    for _ in 0..n {
        let (a, b) = rest.split_at_mut(width * ch);
        bands.push(a);
        rest = b;
    }
    let simd = pick(kernel);
    let cl = |a: usize, rows: &mut [&mut [u8]]| {
        for (i, row) in rows.iter_mut().enumerate() {
            let oy = a + i;
            #[cfg(target_arch = "x86_64")]
            match simd {
                Some(Kernel::Sse) | Some(Kernel::Avx2) => {
                    unsafe {
                        sse4::v_pass_row(src.base, src.pitch, width, ch, taps, oy, row);
                    }
                    continue;
                }
                _ => {}
            }
            let (prow, wrow) = taps.row(oy);
            for ox in 0..width {
                let mut acc: [f32; 4] = [0.0; 4];
                for k in 0..taps.n_taps {
                    let srow = src.tight_row(prow[k] as usize);
                    let w = wrow[k];
                    for c in 0..ch {
                        acc[c] += w * srow[ox * ch + c] as f32;
                    }
                }
                for c in 0..ch {
                    row[ox * ch + c] = round_clamp(acc[c]);
                }
            }
        }
    };
    parallel_rows(&mut bands, &cl);
}

// ─────────────────────────── public API ───────────────────────────

/// Resize a packed RGB24/RGBA32 image to `scale.width x scale.height`.
///
/// `dst` must hold at least `width * height * src.channels` bytes (tight
/// rows). The scratch buffer for the horizontal pass is allocated here.
pub fn resize_rgb(
    src: &crate::pixel::RgbImage,
    scale: Scale,
    kernel: Kernel,
    dst: &mut [u8],
) -> ImageResult<()> {
    let (dw, dh) = (scale.width as usize, scale.height as usize);
    if dw == 0 || dh == 0 {
        return Err(ImageError::InvalidDimensions(
            "scale target must be non-zero".into(),
        ));
    }
    let ch = src.channels as usize;
    if ch != 3 && ch != 4 {
        return Err(ImageError::Unsupported(format!(
            "unsupported channel count {ch}"
        )));
    }
    if src.width == 0 || src.height == 0 {
        return Err(ImageError::InvalidDimensions("empty source".into()));
    }
    let need = dw * dh * ch;
    if dst.len() < need {
        return Err(ImageError::OutputTooSmall { need, have: dst.len() });
    }

    let sx = dw == src.width;
    let sy = dh == src.height;
    if sx && sy {
        // Identity for every filter (centers land exactly on source pixels).
        for oy in 0..src.height {
            dst[oy * dw * ch..oy * dw * ch + dw * ch]
                .copy_from_slice(&src.row(oy)[..dw * ch]);
        }
        return Ok(());
    }

    let x_taps = TapAxis::build(scale.filter, src.width, dw);
    let y_taps = TapAxis::build(scale.filter, src.height, dh);
    let src_rows = PlaneRows {
        base: src.pixels_ptr(),
        pitch: src.pitch,
        width: src.width,
        height: src.height,
        ch,
    };

    if sy {
        // Vertical is identity: the H pass writes the output directly.
        h_pass_scalar(&src_rows, &x_taps, dst);
        return Ok(());
    }
    if sx {
        // Horizontal is identity: the V pass reads source rows directly.
        v_pass(&src_rows, &y_taps, src.width, ch, kernel, dst);
        return Ok(());
    }

    let mut tmp = vec![0u8; dw * src.height * ch];
    h_pass(&src_rows, &x_taps, kernel, &mut tmp);
    let tmp_rows = PlaneRows {
        base: tmp.as_ptr(),
        pitch: dw * ch,
        width: dw,
        height: src.height,
        ch,
    };
    v_pass(&tmp_rows, &y_taps, dw, ch, kernel, dst);
    Ok(())
}

/// Horizontal pass with SIMD dispatch (RGBA32 gets the AVX2 gather path).
fn h_pass(src: &PlaneRows, taps: &TapAxis, kernel: Kernel, tmp: &mut [u8]) {
    let dw = taps.len();
    let n = src.height;
    let mut bands: Vec<&mut [u8]> = Vec::with_capacity(n);
    let mut rest = tmp;
    for _ in 0..n {
        let (a, b) = rest.split_at_mut(dw * src.ch);
        bands.push(a);
        rest = b;
    }
    let simd = pick(kernel);
    let cl = |a: usize, rows: &mut [&mut [u8]]| {
        for (i, row) in rows.iter_mut().enumerate() {
            let oy = a + i;
            #[cfg(target_arch = "x86_64")]
            if matches!(simd, Some(Kernel::Avx2)) && src.ch == 4 {
                unsafe {
                    avx2::h_pass_rgba_row(src.base, src.pitch, src.width, taps, oy, row);
                }
                continue;
            }
            let srow = src.tight_row(oy);
            for ox in 0..dw {
                h_pass_px(srow, taps, ox, src.ch, row);
            }
        }
    };
    parallel_rows(&mut bands, &cl);
}

/// Resize an 8-bit 4:2:0 Y'CbCr image to `scale.width x scale.height`.
///
/// The output layout matches the input (planar -> tight I420, semi-planar ->
/// tight NV12). Each plane is scaled independently: luma to `dw x dh`,
/// chroma to `(dw+1)/2 x (dh+1)/2`.
pub fn resize_yuv(
    src: &YuvImage,
    scale: Scale,
    kernel: Kernel,
    dst: &mut [u8],
) -> ImageResult<()> {
    if src.bits_per_sample != 8 {
        return Err(ImageError::Unsupported(format!(
            "resize supports 8-bit sources, got {}-bit (down-cast first)",
            src.bits_per_sample
        )));
    }
    let (dw, dh) = (scale.width as usize, scale.height as usize);
    if dw == 0 || dh == 0 {
        return Err(ImageError::InvalidDimensions(
            "scale target must be non-zero".into(),
        ));
    }
    if src.width == 0 || src.height == 0 {
        return Err(ImageError::InvalidDimensions("empty source".into()));
    }

    let semi = src.layout() == crate::conv::Layout::Semi;
    let (sw, sh) = (src.width, src.height);
    let (swc, shc) = (src.chroma_width(), src.chroma_height());
    let (dwc, dhc) = ((dw + 1) / 2, (dh + 1) / 2);

    if semi {
        let need = dw * dh + dwc * 2 * dhc;
        if dst.len() < need {
            return Err(ImageError::OutputTooSmall {
                need,
                have: dst.len(),
            });
        }
    } else {
        let need = crate::conv::i420_size(dw, dh);
        if dst.len() < need {
            return Err(ImageError::OutputTooSmall {
                need,
                have: dst.len(),
            });
        }
    }

    let x_luma = TapAxis::build(scale.filter, sw, dw);
    let y_luma = TapAxis::build(scale.filter, sh, dh);
    let x_chroma = TapAxis::build(scale.filter, swc, dwc);
    let y_chroma = TapAxis::build(scale.filter, shc, dhc);

    // Luma: full pipeline (H scratch -> V).
    {
        let rows = PlaneRows {
            base: src.y.as_ptr(),
            pitch: src.y_pitch,
            width: sw,
            height: sh,
            ch: 1,
        };
        let y_out = &mut dst[..dw * dh];
        if dw == sw && dh == sh {
            for oy in 0..sh {
                y_out[oy * dw..oy * dw + dw].copy_from_slice(rows.tight_row(oy));
            }
        } else if dh == sh {
            h_pass_scalar(&rows, &x_luma, y_out);
        } else if dw == sw {
            v_pass(&rows, &y_luma, sw, 1, kernel, y_out);
        } else {
            let mut tmp = vec![0u8; dw * sh];
            h_pass(&rows, &x_luma, kernel, &mut tmp);
            let trows = PlaneRows {
                base: tmp.as_ptr(),
                pitch: dw,
                width: dw,
                height: sh,
                ch: 1,
            };
            v_pass(&trows, &y_luma, dw, 1, kernel, y_out);
        }
    }

    // Chroma: resize Cb and Cr into the target layout.
    if semi {
        // Cb/Cr live at even/odd byte positions of each interleaved row:
        // de-interleave into tight rows, resize, re-interleave.
        let uv = &mut dst[dw * dh..];
        let mut u_src = vec![0u8; swc * shc];
        let mut v_src = vec![0u8; swc * shc];
        for oy in 0..shc {
            let srow = unsafe {
                std::slice::from_raw_parts(src.cb.as_ptr().add(oy * src.cb_pitch), swc * 2)
            };
            for ox in 0..swc {
                u_src[oy * swc + ox] = srow[ox * 2];
                v_src[oy * swc + ox] = srow[ox * 2 + 1];
            }
        }
        let mut u_tmp = vec![0u8; dwc * dhc];
        let mut v_tmp = vec![0u8; dwc * dhc];
        resize_plane(
            u_src.as_ptr(),
            swc,
            swc,
            shc,
            &x_chroma,
            &y_chroma,
            kernel,
            &mut u_tmp,
        );
        resize_plane(
            v_src.as_ptr(),
            swc,
            swc,
            shc,
            &x_chroma,
            &y_chroma,
            kernel,
            &mut v_tmp,
        );
        for oy in 0..dhc {
            for ox in 0..dwc {
                uv[oy * dwc * 2 + ox * 2] = u_tmp[oy * dwc + ox];
                uv[oy * dwc * 2 + ox * 2 + 1] = v_tmp[oy * dwc + ox];
            }
        }
    } else {
        let cr_off = dwc * dhc;
        resize_plane(
            src.cb.as_ptr(),
            src.cb_pitch,
            swc,
            shc,
            &x_chroma,
            &y_chroma,
            kernel,
            &mut dst[dw * dh..dw * dh + cr_off],
        );
        let cr = src
            .cr
            .expect("planar layout requires a Cr plane");
        resize_plane(
            cr.as_ptr(),
            src.cr_pitch,
            swc,
            shc,
            &x_chroma,
            &y_chroma,
            kernel,
            &mut dst[dw * dh + cr_off..],
        );
    }
    Ok(())
}

/// Resize one 8-bit plane (`base`, pitch `pitch`, `sw x sh`) into tight
/// `dwc x dhc` output using the shared chroma tap tables.
fn resize_plane(
    base: *const u8,
    pitch: usize,
    sw: usize,
    sh: usize,
    x_taps: &TapAxis,
    y_taps: &TapAxis,
    kernel: Kernel,
    out: &mut [u8],
) {
    let rows = PlaneRows {
        base,
        pitch,
        width: sw,
        height: sh,
        ch: 1,
    };
    let (dw, dh) = (x_taps.len(), y_taps.len());
    if dw == sw && dh == sh {
        for oy in 0..sh {
            out[oy * dw..oy * dw + dw].copy_from_slice(rows.tight_row(oy));
        }
    } else if dh == sh {
        h_pass_scalar(&rows, x_taps, out);
    } else if dw == sw {
        v_pass(&rows, y_taps, sw, 1, kernel, out);
    } else {
        let mut tmp = vec![0u8; dw * sh];
        h_pass(&rows, x_taps, kernel, &mut tmp);
        let trows = PlaneRows {
            base: tmp.as_ptr(),
            pitch: dw,
            width: dw,
            height: sh,
            ch: 1,
        };
        v_pass(&trows, y_taps, dw, 1, kernel, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::{RgbImage};
    use crate::spec::{ColorRange, ColorSpec, MatrixCoefficients, RgbChannels};

    /// Deterministic gradient RGB image (owned buffer + view).
    fn grad_rgb(w: usize, h: usize, ch: u8) -> (Vec<u8>, RgbImage<'static>) {
        let mut buf = vec![0u8; w * h * ch as usize];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * ch as usize;
                buf[p] = ((x * 255) / w.saturating_sub(1).max(1)) as u8;
                buf[p + 1] = ((y * 255) / h.saturating_sub(1).max(1)) as u8;
                buf[p + 2] = (((x + y) * 255) / (w + h - 2).max(1)) as u8;
                if ch == 4 {
                    buf[p + 3] = 255;
                }
            }
        }
        let data: &'static [u8] = Box::leak(buf.clone().into_boxed_slice());
        (buf, RgbImage::new(data, w * ch as usize, w, h, ch))
    }

    fn resize_all_kernels(src: &RgbImage, scale: Scale) {
        let ch = src.channels as usize;
        let need = scale.width as usize * scale.height as usize * ch;
        let mut out = vec![0u8; need];
        resize_rgb(src, scale, Kernel::Scalar, &mut out).unwrap();
        let mut out2 = out.clone();
        resize_rgb(src, scale, Kernel::Auto, &mut out2).unwrap();
        assert_eq!(out, out2, "auto vs scalar ({scale:?})");
        if simd_features().sse41 {
            resize_rgb(src, scale, Kernel::Sse, &mut out2).unwrap();
            assert_eq!(out, out2, "sse vs scalar ({scale:?})");
        }
        if simd_features().avx2 {
            resize_rgb(src, scale, Kernel::Avx2, &mut out2).unwrap();
            assert_eq!(out, out2, "avx2 vs scalar ({scale:?})");
        }
    }

    #[test]
    fn kernels_agree_all_filters() {
        for (w, h) in [(320usize, 240usize), (97usize, 61usize)] {
            for ch in [3u8, 4] {
                let (_, img) = grad_rgb(w, h, ch);
                for filter in [Interpolation::Bilinear, Interpolation::Box, Interpolation::Bicubic] {
                    for (tw, th) in [(200usize, 150usize), (640, 480), (80, 60), (13, 7)] {
                        resize_all_kernels(&img, Scale::new(tw as u32, th as u32, filter));
                    }
                }
            }
        }
    }

    #[test]
    fn identity_scale_is_exact() {
        for ch in [3u8, 4] {
            let (buf, img) = grad_rgb(100, 60, ch);
            for filter in [Interpolation::Bilinear, Interpolation::Box, Interpolation::Bicubic] {
                let mut out = vec![0u8; buf.len()];
                resize_rgb(&img, Scale::new(100, 60, filter), Kernel::Auto, &mut out).unwrap();
                assert_eq!(buf, out, "identity {filter:?} ch={ch}");
            }
        }
    }

    #[test]
    fn monochrome_stays_monochrome() {
        let (w, h) = (128usize, 96usize);
        for ch in [3u8, 4] {
            let mut buf = vec![0u8; w * h * ch as usize];
            for y in 0..h {
                for x in 0..w {
                    let v = ((x + y) % 256) as u8;
                    let p = (y * w + x) * ch as usize;
                    buf[p] = v;
                    buf[p + 1] = v;
                    buf[p + 2] = v;
                    if ch == 4 {
                        buf[p + 3] = 255;
                    }
                }
            }
            let data: &'static [u8] = Box::leak(buf.clone().into_boxed_slice());
            let img = RgbImage::new(data, w * ch as usize, w, h, ch);
            for filter in [Interpolation::Bilinear, Interpolation::Box, Interpolation::Bicubic] {
                let (tw, th) = (300usize, 200usize);
                let mut out = vec![0u8; tw * th * ch as usize];
                resize_rgb(&img, Scale::new(tw as u32, th as u32, filter), Kernel::Auto, &mut out).unwrap();
                for y in 0..th {
                    for x in 0..tw {
                        let p = (y * tw + x) * ch as usize;
                        // Source value at the ideal position must come back.
                        let sx = ((x as f64 + 0.5) * w as f64 / tw as f64 - 0.5).clamp(0.0, (w - 1) as f64);
                        let sy = ((y as f64 + 0.5) * h as f64 / th as f64 - 0.5).clamp(0.0, (h - 1) as f64);
                        // Interpolation of a smooth diagonal gradient stays within +-2 of the center value.
                        let expect = ((sx.round() as usize + sy.round() as usize) % 256) as i32;
                        let got = out[p] as i32;
                        assert!(
                            (expect - got).abs() <= 2,
                            "monochrome drift at ({x},{y}): expect {expect} got {got} {filter:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn box_downscale_2x_equals_block_average() {
        // 64x48 -> 32x24 with box must equal the exact 2x2 average.
        let (w, h) = (64usize, 48usize);
        let mut buf = vec![0u8; w * h * 3];
        for i in 0..w * h {
            let p = i * 3;
            buf[p] = (i % 251) as u8;
            buf[p + 1] = ((i * 7) % 256) as u8;
            buf[p + 2] = ((i * 13) % 256) as u8;
        }
        let data: &'static [u8] = Box::leak(buf.clone().into_boxed_slice());
        let img = RgbImage::new(data, w * 3, w, h, 3);
        let (tw, th) = (32usize, 24usize);
        let mut out = vec![0u8; tw * th * 3];
        resize_rgb(&img, Scale::new(tw as u32, th as u32, Interpolation::Box), Kernel::Scalar, &mut out).unwrap();
        for y in 0..th {
            for x in 0..tw {
                let s = |ox: usize, oy: usize, c: usize| buf[(oy * w + ox) * 3 + c] as f32;
                for c in 0..3 {
                    let avg = (s(2 * x, 2 * y, c) + s(2 * x + 1, 2 * y, c) + s(2 * x, 2 * y + 1, c) + s(2 * x + 1, 2 * y + 1, c)) / 4.0;
                    let expect = avg.round().clamp(0.0, 255.0) as i32;
                    let got = out[(y * tw + x) * 3 + c] as i32;
                    // The two-pass path rounds the horizontal result to a byte
                    // before the vertical pass, so it can differ from the direct
                    // block average by 1 at rounding boundaries.
                    assert!((expect - got).abs() <= 1, "({x},{y},ch{c}): expect {expect} got {got}");
                }
            }
        }
    }

    #[test]
    fn bicubic_integer_downscale_matches_box() {
        // Regression: exact even-integer-ratio downscale (4x) used to
        // degenerate the cubic discrete sum to zero — the kernel's zero
        // crossings land on every candidate tap, so no tap survived and the
        // output was all black. It must now equal area (box) resampling.
        let (w, h) = (32usize, 24usize);
        let mut buf = vec![0u8; w * h * 3];
        for i in 0..w * h {
            let p = i * 3;
            buf[p] = (i % 251) as u8;
            buf[p + 1] = ((i * 7) % 256) as u8;
            buf[p + 2] = ((i * 13) % 256) as u8;
        }
        let data: &'static [u8] = Box::leak(buf.clone().into_boxed_slice());
        let img = RgbImage::new(data, w * 3, w, h, 3);
        for (tw, th) in [(8usize, 6usize), (16, 12)] {
            let mut out = vec![0u8; tw * th * 3];
            resize_rgb(&img, Scale::new(tw as u32, th as u32, Interpolation::Bicubic), Kernel::Scalar, &mut out).unwrap();
            assert!(!out.iter().all(|&v| v == 0), "all-black output at {tw}x{th}");
            let mut box_out = vec![0u8; tw * th * 3];
            resize_rgb(&img, Scale::new(tw as u32, th as u32, Interpolation::Box), Kernel::Scalar, &mut box_out).unwrap();
            assert_eq!(out, box_out, "bicubic integer downscale != box at {tw}x{th}");
        }
    }

    // ─────────────────────────── YUV resize tests ───────────────────────────

    fn grad_i420(w: usize, h: usize) -> (Vec<u8>, YuvImage<'static>) {
        let cw = (w + 1) / 2;
        let chh = (h + 1) / 2;
        let mut buf = vec![0u8; crate::conv::i420_size(w, h)];
        for y in 0..h {
            for x in 0..w {
                buf[y * w + x] = ((x * 255) / w.saturating_sub(1).max(1)) as u8;
            }
        }
        let (cb, cr) = buf.split_at_mut(w * h + cw * chh);
        let cb = &mut cb[w * h..];
        for i in 0..cw * chh {
            cb[i] = ((i as u32 * 400) % 256) as u8;
            cr[i] = ((i as u32 * 900) % 256) as u8;
        }
        let data: &'static [u8] = Box::leak(buf.clone().into_boxed_slice());
        let img = YuvImage::planar(
            &data[..w * h],
            w,
            &data[w * h..w * h + cw * chh],
            cw,
            &data[w * h + cw * chh..],
            cw,
            w,
            h,
            8,
        );
        (buf, img)
    }

    fn resize_yuv_all_kernels(src: &YuvImage, scale: Scale) {
        let semi = src.layout() == crate::conv::Layout::Semi;
        let need = if semi {
            scale.width as usize * scale.height as usize
                + ((scale.width as usize + 1) / 2) * 2 * ((scale.height as usize + 1) / 2)
        } else {
            crate::conv::i420_size(scale.width as usize, scale.height as usize)
        };
        let mut out = vec![0u8; need];
        resize_yuv(src, scale, Kernel::Scalar, &mut out).unwrap();
        let mut out2 = out.clone();
        resize_yuv(src, scale, Kernel::Auto, &mut out2).unwrap();
        assert_eq!(out, out2, "auto vs scalar ({scale:?})");
        if simd_features().sse41 {
            resize_yuv(src, scale, Kernel::Sse, &mut out2).unwrap();
            assert_eq!(out, out2, "sse vs scalar ({scale:?})");
        }
        if simd_features().avx2 {
            resize_yuv(src, scale, Kernel::Avx2, &mut out2).unwrap();
            assert_eq!(out, out2, "avx2 vs scalar ({scale:?})");
        }
    }

    #[test]
    fn yuv_kernels_agree() {
        for (w, h) in [(320usize, 240usize), (97usize, 61usize)] {
            let (_, img) = grad_i420(w, h);
            for filter in [Interpolation::Bilinear, Interpolation::Box, Interpolation::Bicubic] {
                for (tw, th) in [(200usize, 150usize), (640, 480), (80, 60)] {
                    resize_yuv_all_kernels(&img, Scale::new(tw as u32, th as u32, filter));
                }
            }
        }
    }

    #[test]
    fn yuv_identity_is_exact() {
        let (buf, img) = grad_i420(100, 60);
        let mut out = buf.clone();
        resize_yuv(&img, Scale::new(100, 60, Interpolation::Bilinear), Kernel::Auto, &mut out).unwrap();
        assert_eq!(buf, out);
    }

    #[test]
    fn yuv_neutral_stays_neutral_after_convert() {
        // Cb = Cr = 128 everywhere: resized + converted pixels must be grey.
        let (w, h) = (160usize, 120usize);
        let cw = w / 2;
        let chh = h / 2;
        let mut buf = vec![0u8; crate::conv::i420_size(w, h)];
        for y in 0..h {
            for x in 0..w {
                buf[y * w + x] = ((x * 5) % 256) as u8;
            }
        }
        for i in buf[w * h..].iter_mut() {
            *i = 128;
        }
        let data: &'static [u8] = Box::leak(buf.clone().into_boxed_slice());
        let img = YuvImage::planar(
            &data[..w * h],
            w,
            &data[w * h..w * h + cw * chh],
            cw,
            &data[w * h + cw * chh..],
            cw,
            w,
            h,
            8,
        );
        let (tw, th) = (400usize, 300usize);
        let mut out = vec![0u8; crate::conv::i420_size(tw, th)];
        resize_yuv(&img, Scale::new(tw as u32, th as u32, Interpolation::Bicubic), Kernel::Auto, &mut out).unwrap();
        let view = crate::conv::scratch_view(&out, tw, th);
        let mut rgb = vec![0u8; tw * th * 3];
        crate::conv::yuv_to_rgb(&view, ColorSpec::default(), RgbChannels::Rgb24, &mut rgb).unwrap();
        for i in 0..tw * th {
            let (r, g, b) = (rgb[i * 3] as i32, rgb[i * 3 + 1] as i32, rgb[i * 3 + 2] as i32);
            assert!(
                r.max(g).max(b) - r.min(g).min(b) <= 1,
                "pixel {i} not grey: {r},{g},{b}"
            );
        }
    }

    #[test]
    fn yuv_semi_resize_matches_planar_luma() {
        // The luma plane of a semi-planar resize must equal the planar one.
        let (w, h) = (128usize, 96usize);
        let cw = w / 2;
        let chh = h / 2;
        let mut y = vec![0u8; w * h];
        for i in 0..w * h {
            y[i] = ((i as u32 * 31) % 256) as u8;
        }
        let mut uv = vec![0u8; cw * 2 * chh];
        for i in 0..cw * chh {
            // Semi-planar: Cb/Cr are interleaved 2-byte pairs.
            uv[i * 2] = ((i as u32 * 7) % 256) as u8;
            uv[i * 2 + 1] = ((i as u32 * 11) % 256) as u8;
        }
        let mut pbuf = vec![0u8; crate::conv::i420_size(w, h)];
        pbuf[..w * h].copy_from_slice(&y);
        for r in 0..chh {
            for x in 0..cw {
                pbuf[w * h + r * cw + x] = uv[r * cw * 2 + x * 2];
                pbuf[w * h + cw * chh + r * cw + x] = uv[r * cw * 2 + x * 2 + 1];
            }
        }
        let mut sbuf = vec![0u8; w * h + cw * 2 * chh];
        sbuf[..w * h].copy_from_slice(&y);
        sbuf[w * h..].copy_from_slice(&uv);
        let pdata: &'static [u8] = Box::leak(pbuf.clone().into_boxed_slice());
        let sdata: &'static [u8] = Box::leak(sbuf.clone().into_boxed_slice());
        let pimg = YuvImage::planar(
            &pdata[..w * h],
            w,
            &pdata[w * h..w * h + cw * chh],
            cw,
            &pdata[w * h + cw * chh..],
            cw,
            w,
            h,
            8,
        );
        let simg = YuvImage::semi(&sdata[..w * h], w, &sdata[w * h..], w, w, h, 8);

        let scale = Scale::new(200, 150, Interpolation::Bilinear);
        let mut pout = vec![0u8; crate::conv::i420_size(200, 150)];
        resize_yuv(&pimg, scale, Kernel::Auto, &mut pout).unwrap();
        let mut sout = vec![0u8; 200 * 150 + 100 * 2 * 75];
        resize_yuv(&simg, scale, Kernel::Auto, &mut sout).unwrap();
        assert_eq!(&pout[..200 * 150], &sout[..200 * 150], "luma differs");
    }

    #[test]
    fn tiny_images_do_not_crash() {
        for ch in [3u8, 4] {
            let (_, img) = grad_rgb(1, 1, ch);
            for filter in [Interpolation::Bilinear, Interpolation::Box, Interpolation::Bicubic] {
                let mut out = vec![0u8; 64 * 32 * ch as usize];
                resize_rgb(&img, Scale::new(64, 32, filter), Kernel::Auto, &mut out).unwrap();
            }
        }
        let (_, img) = grad_i420(1, 1);
        let mut out = vec![0u8; crate::conv::i420_size(8, 8)];
        resize_yuv(&img, Scale::new(8, 8, Interpolation::Bicubic), Kernel::Auto, &mut out).unwrap();
    }

    #[test]
    fn mitchell_values() {
        // B = C = 0.5: k(0) = 5/6, k(1) = 1/12, support ends at d = 2.
        assert!((mitchell(0.0) - 5.0 / 6.0).abs() < 1e-12);
        assert!((mitchell(1.0) - 0.5 / 6.0).abs() < 1e-12);
        assert_eq!(mitchell(2.0), 0.0);
        assert_eq!(mitchell(3.0), 0.0);
        // Continuous across the d = 1 knot.
        assert!((mitchell(1.0 - 1e-9) - mitchell(1.0 + 1e-9)).abs() < 1e-6);
    }

    #[allow(dead_code)]
    fn _unused(spec: ColorSpec) {
        let _ = (ColorRange::Limited, MatrixCoefficients::Bt709, spec);
    }
}

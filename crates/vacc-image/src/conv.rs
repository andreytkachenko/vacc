//! Y'CbCr -> RGB8 conversion: scalar reference + SIMD dispatch.
//!
//! The public entry point is [`yuv_to_rgb`]; 10/12-bit packed (P010/P012)
//! sources are down-cast to 8-bit first (`yuv_high_to_i420`), which keeps
//! the hot color-conversion kernels independent of bit depth.
//!
//! Support matrix (x86_64, detected at runtime):
//!
//! | kernel               | SIMD                   |
//! |----------------------|------------------------|
//! | planar 8-bit rows    | AVX2 / SSE4.1 / scalar |
//! | semi-planar 8-bit    | AVX2 / SSE4.1 / scalar |
//! | u16 -> u8 downcast   | AVX2 / scalar          |
//!
//! Rows are independent, so for large frames the row band is split up and
//! processed by helper threads.

use crate::coeff::{Conv8, RND, table};
use crate::error::{ImageError, ImageResult};
use crate::pixel::{YuvImage};
use crate::spec::{ColorSpec, RgbChannels};

pub(crate) mod avx2;
pub(crate) mod sse4;

/// SIMD features detected once per process.
#[derive(Clone, Copy, Debug, Default)]
pub struct SimdAvail {
    pub sse41: bool,
    pub avx2: bool,
}

static SIMD: std::sync::OnceLock<SimdAvail> = std::sync::OnceLock::new();

/// Probe the CPU once (cached). On non-x86 targets everything is off.
pub fn simd_features() -> &'static SimdAvail {
    &SIMD.get_or_init(|| SimdAvail {
        sse41: is_x86_sse41(),
        avx2: is_x86_avx2(),
    })
}

#[inline]
fn is_x86_sse41() -> bool {
    cfg!(target_arch = "x86_64")
        && std::arch::is_x86_feature_detected!("sse4.1")
}

#[inline]
fn is_x86_avx2() -> bool {
    cfg!(target_arch = "x86_64")
        && std::arch::is_x86_feature_detected!("avx2")
}

/// Which code path to run (`Auto` dispatches to the fastest available).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Kernel {
    #[default]
    Auto,
    Scalar,
    Sse,
    Avx2,
}

/// Minimum row count before spawning helper threads.
const PAR_ROW_THRESHOLD: usize = 256;
/// Target rows per helper thread.
const ROWS_PER_THREAD: usize = 64;
/// Hard cap on helper threads.
const MAX_THREADS: usize = 16;

/// Split a per-row band view into strips and process each strip on a
/// helper thread.
///
/// `bands` holds one entry per luma row; strip `start` owns exactly its
/// `(start .. min(start + len, n))` slice, so concurrent invocations of
/// `f` never overlap.
pub fn parallel_rows<T: Send>(
    bands: &mut [T],
    f: &(impl Fn(usize, &mut [T]) + Send + Sync),
) {
    let n = bands.len();
    let strips = ((n + ROWS_PER_THREAD - 1) / ROWS_PER_THREAD).min(MAX_THREADS);
    if n < PAR_ROW_THRESHOLD || strips <= 1 {
        f(0, bands);
        return;
    }
    let strip = (n + strips - 1) / strips;
    std::thread::scope(|s| {
        let mut rest = bands;
        for start in (0usize..).step_by(strip) {
            if start >= n {
                break;
            }
            let len = (strip).min(n - start);
            let (band, tail) = rest.split_at_mut(len);
            s.spawn(move || f(start, band));
            rest = tail;
        }
    });
}

// ─────────────────────────── public API ───────────────────────────

/// Convert a 4:2:0 Y'CbCr image (8-bit, P010 or P012 packed in `u16`
/// *top-justified*) to packed RGB (24) or RGBA (32).
///
/// `dst` must have length at least `width * height * channels`.
/// For 10/12-bit sources an exact 8-bit down-cast happens first.
pub fn yuv_to_rgb(
    src: &YuvImage,
    spec: ColorSpec,
    channels: RgbChannels,
    dst: &mut [u8],
) -> ImageResult<()> {
    let bpp = match src.bits_per_sample {
        8 | 10 | 12 => src.bps_bytes(),
        _ => {
            return Err(ImageError::Unsupported(format!(
                "unsupported bit depth {}",
                src.bits_per_sample
            )))
        }
    };
    validate_planes(src, bpp)?;

    let chb = channels.bytes() as usize;
    let need = src.width * src.height * chb;
    if dst.len() < need {
        return Err(ImageError::OutputTooSmall { need, have: dst.len() });
    }

    let coeffs = table(spec);

    if src.bits_per_sample == 8 {
        convert_rows(src, dst, src.width * chb, chb, &coeffs, Kernel::Auto);
    } else {
        // P010/P012 -> exact 8-bit I420 scratch -> 8-bit conversion.
        let mut scratch = vec![0u8; i420_size(src.width, src.height)];
        downcast_rows(src, &mut scratch);
        let view = scratch_view(&scratch, src.width, src.height);
        convert_rows(&view, dst, src.width * chb, chb, &coeffs, Kernel::Auto);
    }
    Ok(())
}

/// Down-cast a P010/P012 (`u16` top-justified) image to a tight 8-bit I420
/// (planar) buffer. `dst` must have length `i420_size(w, h)`.
///
/// The metadata handling (top-justified) makes this an exact shift:
/// P010 stores `v10 << 6` so `>> 6` recovers `v10`; P012 stores `v12 << 4`
/// so `>> 4` recovers `v12`.
pub fn yuv_high_to_i420(src: &YuvImage, dst: &mut [u8]) -> ImageResult<()> {
    let bps = src.bits_per_sample;
    if bps != 10 && bps != 12 {
        return Err(ImageError::Unsupported(format!(
            "P010/P012 expected, got {bps}-bit"
        )));
    }
    let need = i420_size(src.width, src.height);
    if dst.len() < need {
        return Err(ImageError::OutputTooSmall {
            need,
            have: dst.len(),
        });
    }
    validate_planes(src, 2)?;
    downcast_rows(src, dst);
    Ok(())
}

/// Length of a tight 8-bit I420 buffer for `w * h`: luma plane plus two
/// chroma (Cb/Cr) planes, each `ceil(w/2) * ceil(h/2)` bytes.
pub const fn i420_size(width: usize, height: usize) -> usize {
    let y = width * height;
    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;
    y + 2 * cw * ch
}

/// A view over a tight 8-bit I420 buffer produced by the down-casters.
pub fn scratch_view<'a>(buf: &'a [u8], w: usize, h: usize) -> YuvImage<'a> {
    let y = &buf[..w * h];
    let cw = (w + 1) / 2;
    let chh = (h + 1) / 2;
    let cb = &buf[w * h..w * h + cw * chh];
    let cr = &buf[w * h + cw * chh..w * h + 2 * cw * chh];
    YuvImage::planar(y, w, cb, cw, cr, cw, w, h, 8)
}

// ─────────────────────────── validation ───────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Layout {
    Planar,
    Semi,
}

impl YuvImage<'_> {
    /// Whether the Cb/Cr data is two separate planes or interleaved.
    pub(crate) fn layout(&self) -> Layout {
        if self.cr.is_some() {
            Layout::Planar
        } else {
            Layout::Semi
        }
    }
}

fn validate_planes(src: &YuvImage, bps: usize) -> ImageResult<()> {
    let line = |rows: &[u8], pitch: usize, w: usize, rows_count: usize, name: &str| -> ImageResult<()> {
        let row_bytes = w * bps;
        if pitch < row_bytes {
            return Err(ImageError::InvalidDimensions(format!(
                "{name} pitch {pitch} < width {w} * bps {bps}"
            )));
        }
        let need_rows = if rows_count == 0 { 0 } else { pitch * (rows_count - 1) + row_bytes };
        if rows.len() < need_rows {
            return Err(ImageError::InvalidDimensions(format!(
                "{name} plane too short: have {} bytes, need at least {need_rows}",
                rows.len()
            )));
        }
        Ok(())
    };
    line(&src.y, src.y_pitch, src.width, src.height, "Y")?;
    let cw = src.chroma_width();
    match src.layout() {
        Layout::Planar => {
            let cr = src
                .cr
                .ok_or_else(|| ImageError::InvalidDimensions("Cr plane missing".into()))?;
            line(&src.cb, src.cb_pitch, cw, src.chroma_height(), "Cb")?;
            line(cr, src.cr_pitch, cw, src.chroma_height(), "Cr")?;
        }
        Layout::Semi => {
            line(&src.cb, src.cb_pitch, cw * 2, src.chroma_height(), "CbCr")?;
        }
    }
    Ok(())
}

// ─────────────────────────── 8-bit conversion ───────────────────────────

/// Convert all luma rows of an 8-bit image into `dst` (packed RGB(x)), where
/// `dst` is split into rows of `pitch` bytes.
pub fn convert_rows(
    src: &YuvImage,
    dst: &mut [u8],
    pitch: usize,
    channels: usize,
    c: &Conv8,
    kernel: Kernel,
) {
    // `Some(k)` only ever names a SIMD kernel that is actually available on
    // this CPU; `None` means "use the scalar rows" (explicit `Scalar`, or a
    // requested/auto SIMD this CPU lacks).
    let simd = simd_features();
    let pick = |want_sse: bool, want_avx: bool| -> Option<Kernel> {
        match kernel {
            Kernel::Auto => {
                if want_avx && simd.avx2 {
                    Some(Kernel::Avx2)
                } else if want_sse && simd.sse41 {
                    Some(Kernel::Sse)
                } else {
                    None
                }
            }
            Kernel::Sse => simd.sse41.then_some(Kernel::Sse),
            Kernel::Avx2 => simd.avx2.then_some(Kernel::Avx2),
            Kernel::Scalar => None,
        }
    };

    // One band per luma row, each `pitch` bytes; strips never overlap.
    let n = src.height;
    let mut bands: Vec<&mut [u8]> = Vec::with_capacity(n);
    let mut rest = dst;
    for _ in 0..n {
        let (a, b) = rest.split_at_mut(pitch);
        bands.push(a);
        rest = b;
    }
    let row_bytes = src.width * channels;

    match src.layout() {
        Layout::Planar => {
            let cr = src.cr.expect("planar layout requires a Cr plane");
            let cl = |a: usize, rows: &mut [&mut [u8]]| {
                for (i, row) in rows.iter_mut().enumerate() {
                    let y = a + i;
                    let (y_ptr, cb_ptr, cr_ptr) = unsafe {
                        (
                            src.y.as_ptr().add(y * src.y_pitch),
                            src.cb.as_ptr().add((y >> 1) * src.cb_pitch),
                            cr.as_ptr().add((y >> 1) * src.cr_pitch),
                        )
                    };
                    let dst_row = &mut (**row)[..row_bytes];
                    #[cfg(target_arch = "x86_64")]
                    match pick(true, true) {
                        Some(Kernel::Avx2) => unsafe {
                            avx2::planar_row(y_ptr, cb_ptr, cr_ptr, src.width, c, dst_row);
                            continue;
                        },
                        Some(Kernel::Sse) => unsafe {
                            sse4::planar_row(y_ptr, cb_ptr, cr_ptr, src.width, c, dst_row);
                            continue;
                        },
                        // Stays correct even if `pick` ever reports a
                        // non-SIMD kernel again: fall through to scalar.
                        _ => {}
                    }
                    scalar_planar_row(y_ptr, cb_ptr, cr_ptr, src.width, c, dst_row);
                }
            };
            parallel_rows(&mut bands, &cl);
        }
        Layout::Semi => {
            let cl = |a: usize, rows: &mut [&mut [u8]]| {
                for (i, row) in rows.iter_mut().enumerate() {
                    let y = a + i;
                    let (y_ptr, uv_ptr) = unsafe {
                        (
                            src.y.as_ptr().add(y * src.y_pitch),
                            src.cb.as_ptr().add((y >> 1) * src.cb_pitch),
                        )
                    };
                    let dst_row = &mut (**row)[..row_bytes];
                    #[cfg(target_arch = "x86_64")]
                    match pick(true, true) {
                        Some(Kernel::Avx2) => unsafe {
                            avx2::semi_row(y_ptr, uv_ptr, src.width, c, dst_row);
                            continue;
                        },
                        Some(Kernel::Sse) => unsafe {
                            sse4::semi_row(y_ptr, uv_ptr, src.width, c, dst_row);
                            continue;
                        },
                        _ => {}
                    }
                    scalar_semi_row(y_ptr, uv_ptr, src.width, c, dst_row);
                }
            };
            parallel_rows(&mut bands, &cl);
        }
    }
}

/// One luma row (planar layout) via the scalar kernel.
fn scalar_planar_row(
    y_row: *const u8,
    cb_row: *const u8,
    cr_row: *const u8,
    width: usize,
    c: &Conv8,
    dst_row: &mut [u8],
) {
    let ch = if dst_row.len() == width * 4 { 4 } else { 3 };
    for x in 0..width {
        let yy = unsafe { *y_row.add(x) } as i32;
        let cc = unsafe { *cb_row.add(x >> 1) } as i32;
        let cr = unsafe { *cr_row.add(x >> 1) } as i32;
        let (r, g, b) = conv_px(c, yy, cc, cr);
        let d = &mut dst_row[x * ch..x * ch + 3];
        d[0] = r;
        d[1] = g;
        d[2] = b;
        if ch == 4 {
            dst_row[x * 4 + 3] = 255;
        }
    }
}

/// One luma row (semi-planar NV12) via the scalar kernel.
fn scalar_semi_row(
    y_row: *const u8,
    uv_row: *const u8,
    width: usize,
    c: &Conv8,
    dst_row: &mut [u8],
) {
    let ch = if dst_row.len() == width * 4 { 4 } else { 3 };
    for x in 0..width {
        let yy = unsafe { *y_row.add(x) } as i32;
        // Luma x is covered by CbCr pair x / 2 => bytes (2 * (x / 2), +1).
        let uo = x & !1;
        let cc = unsafe { *uv_row.add(uo) } as i32;
        let cr = unsafe { *uv_row.add(uo + 1) } as i32;
        let (r, g, b) = conv_px(c, yy, cc, cr);
        let d = &mut dst_row[x * ch..x * ch + 3];
        d[0] = r;
        d[1] = g;
        d[2] = b;
        if ch == 4 {
            dst_row[x * 4 + 3] = 255;
        }
    }
}

/// One-pixel conversion via the Q14 table (scalar reference).
#[inline]
pub fn conv_px(c: &Conv8, y: i32, cb: i32, cr: i32) -> (u8, u8, u8) {
    let r = ((c.ky * y + c.r_cr * cr + c.r_off + RND) >> 14).clamp(0, 255);
    let g = ((c.ky * y + c.g_cb * cb + c.g_cr * cr + c.g_off + RND) >> 14).clamp(0, 255);
    let b = ((c.ky * y + c.b_cb * cb + c.b_off + RND) >> 14).clamp(0, 255);
    (r as u8, g as u8, b as u8)
}

// ─────────────────────────── 16 -> 8 downcast ───────────────────────────

fn downcast_rows(src: &YuvImage, dst: &mut [u8]) {
    let w = src.width;
    let h = src.height;
    let shift = 16 - src.bits_per_sample as u32;
    let cw = (w + 1) / 2;
    let chh = (h + 1) / 2;
    let (y_part, rest) = dst.split_at_mut(w * h);
    let (cb_part, cr_part) = rest.split_at_mut(cw * chh);

    // One band per luma row, each `w` bytes.
    let mut bands: Vec<&mut [u8]> = Vec::with_capacity(h);
    let mut rest = y_part;
    for _ in 0..h {
        let (a, b) = rest.split_at_mut(w);
        bands.push(a);
        rest = b;
    }
    let cl = |a: usize, rows: &mut [&mut [u8]]| {
        for (i, row) in rows.iter_mut().enumerate() {
            let y = a + i;
            let y_ptr = unsafe { src.y.as_ptr().add(y * src.y_pitch) };
            #[cfg(target_arch = "x86_64")]
            if is_x86_avx2() {
                unsafe {
                    avx2::u16_shift_row(y_ptr, &mut (**row)[..], shift);
                }
                continue;
            }
            sc_shift_row(y_ptr, &mut (**row)[..], shift);
        }
    };
    parallel_rows(&mut bands, &cl);

    let semi = src.layout() == Layout::Semi;
    // One band per chroma row; each band is (Cb, Cr) halves of `cw` bytes.
    let mut bands: Vec<(&mut [u8], &mut [u8])> = Vec::with_capacity(chh);
    let mut ru = cb_part;
    let mut rv = cr_part;
    for _ in 0..chh {
        let (a, rest_u) = ru.split_at_mut(cw);
        let (b, rest_v) = rv.split_at_mut(cw);
        bands.push((a, b));
        ru = rest_u;
        rv = rest_v;
    }
    let cl = |a: usize, rows: &mut [(&mut [u8], &mut [u8])]| {
        for (i, (ou, ov)) in rows.iter_mut().enumerate() {
            let y = a + i;
            let out_u = &mut (**ou)[..];
            let out_v = &mut (**ov)[..];
            let semi_ptrs = unsafe {
                if semi {
                    Some(src.cb.as_ptr().add(y * src.cb_pitch))
                } else {
                    None
                }
            };
            if semi {
                let uv = semi_ptrs.expect("semi set");
                #[cfg(target_arch = "x86_64")]
                if is_x86_avx2() {
                    unsafe {
                        avx2::u16_semi_shift_row(uv, out_u, out_v, shift, cw);
                    }
                    continue;
                }
                sc_semi_u16_row(uv, out_u, out_v, shift, cw);
            } else {
                let (su, sv) = unsafe {
                    let cr = src
                        .cr
                        .expect("planar layout requires a Cr plane");
                    (
                        src.cb.as_ptr().add(y * src.cb_pitch),
                        cr.as_ptr().add(y * src.cr_pitch),
                    )
                };
                #[cfg(target_arch = "x86_64")]
                if is_x86_avx2() {
                    unsafe {
                        avx2::u16_shift_row(su, out_u, shift);
                        avx2::u16_shift_row(sv, out_v, shift);
                    }
                    continue;
                }
                sc_shift_row(su, out_u, shift);
                sc_shift_row(sv, out_v, shift);
            }
        }
    };
    parallel_rows(&mut bands, &cl);
}

/// One tight `u16` row -> `u8` row (right-shift by `shift`, saturating to
/// 255 so the scalar tail agrees with `u16_shift_row`'s SIMD `packus`).
#[inline]
fn sc_shift_row(row: *const u8, out: &mut [u8], shift: u32) {
    for i in 0..out.len() {
        let lo = unsafe { *row.add(i * 2) } as u32;
        let hi = unsafe { *row.add(i * 2 + 1) } as u32;
        let v = ((hi << 8) | lo) >> shift;
        out[i] = v.min(255) as u8;
    }
}

/// One interleaved `u16` CbCr row -> two `u8` rows (right-shift, saturating).
pub(crate) fn sc_semi_u16_row(
    uv: *const u8,
    out_u: &mut [u8],
    out_v: &mut [u8],
    shift: u32,
    width: usize,
) {
    for i in 0..width {
        let base = i * 4;
        let au =
            u16::from_le_bytes(unsafe { [*uv.add(base), *uv.add(base + 1)] });
        let av = u16::from_le_bytes(unsafe { [*uv.add(base + 2), *uv.add(base + 3)] });
        out_u[i] = (au >> shift).min(255) as u8;
        out_v[i] = (av >> shift).min(255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{ColorRange, MatrixCoefficients};

    #[allow(dead_code)]
    fn grad_image(w: usize, h: usize, planar: bool) -> (Vec<u8>, YuvImage<'static>) {
        let y: Vec<u8> = (0..w * h).map(|i| ((i % w) as u32 * 255 / w.max(1) as u32) as u8).collect();
        let cw = (w + 1) / 2;
        let chh = (h + 1) / 2;
        let cb: Vec<u8> = (0..cw * chh).map(|i| ((i as u32) * 400 % 256) as u8).collect();
        let cr: Vec<u8> = (0..cw * chh).map(|i| ((i as u32) * 900 % 256) as u8).collect();
        if planar {
            let mut buf = Vec::new();
            buf.extend_from_slice(&y);
            buf.extend_from_slice(&cb);
            buf.extend_from_slice(&cr);
            // Leak a static copy for the view; return the owned buffer.
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
        } else {
            let mut uv = vec![0u8; cw * 2 * chh];
            for r in 0..chh {
                for x in 0..cw {
                    uv[r * cw * 2 + x * 2] = cb[r * cw + x];
                    uv[r * cw * 2 + x * 2 + 1] = cr[r * cw + x];
                }
            }
            let mut buf = Vec::new();
            buf.extend_from_slice(&y);
            buf.extend_from_slice(&uv);
            let data: &'static [u8] = Box::leak(buf.clone().into_boxed_slice());
            // 8-bit semi: each chroma row holds w/2 (u, v) pairs = w bytes.
            let img = YuvImage::semi(&data[..w * h], w, &data[w * h..], w, w, h, 8);
            (buf, img)
        }
    }

    /// Independent f64 reference (different rounding from Q14).
    fn f64_ref(spec: ColorSpec, y: u8, cb: u8, cr: u8) -> (i32, i32, i32) {
        let m = spec.matrix;
        let limited = spec.range == ColorRange::Limited;
        let (ky, kr, gcb, gcr, bcb): (f64, f64, f64, f64, f64) = if limited {
            match m {
                MatrixCoefficients::Bt601 => (1.164, 1.596, 0.3917, 0.8135, 2.0172),
                MatrixCoefficients::Bt709 => (
                    1.164384, 1.574764, 0.391749, 0.812908, 2.017221
                ),
            }
        } else {
            match m {
                MatrixCoefficients::Bt601 => (1.0, 1.402, 0.344136, 0.714136, 1.772),
                MatrixCoefficients::Bt709 => (1.0, 1.5748, 0.344136, 0.714136, 1.8814),
            }
        };
        let clip = |v: f64| v.clamp(0.0, 255.0).round() as i32;
        let yv = f64::from(y) - if limited { 16.0 } else { 0.0 };
        let cbv = f64::from(cb) - 128.0;
        let crv = f64::from(cr) - 128.0;
        (
            clip(ky * yv + kr * crv),
            clip(ky * yv - gcb * cbv - gcr * crv),
            clip(ky * yv + bcb * cbv),
        )
    }

    #[test]
    fn yuv_to_rgb_all_kernels_agree() {
        let spec = ColorSpec::default();
        let c = table(spec);
        for planar in [false, true] {
            let (_, img) = grad_image(640, 480, planar);
            for chb in [3usize, 4] {
                let mut out = vec![0u8; 640 * 480 * chb];
                convert_rows(&img, &mut out, 640 * chb, chb, &c, Kernel::Scalar);
                let mut out2 = out.clone();
                convert_rows(&img, &mut out2, 640 * chb, chb, &c, Kernel::Auto);
                assert_eq!(out, out2, "auto vs scalar (planar={planar}, ch={chb})");
                if simd_features().sse41 {
                    convert_rows(
                        &img,
                        &mut out2,
                        640 * chb,
                        chb,
                        &c,
                        Kernel::Sse,
                    );
                    assert_eq!(out, out2, "sse4.1 kernel differs (planar={planar}, ch={chb})");
                }
                if simd_features().avx2 {
                    convert_rows(
                        &img,
                        &mut out2,
                        640 * chb,
                        chb,
                        &c,
                        Kernel::Avx2,
                    );
                    assert_eq!(out, out2, "avx2 kernel differs (planar={planar}, ch={chb})");
                }
            }
        }
    }

    #[test]
    fn yuv_to_rgb_within_f64_reference() {
        let spec = ColorSpec::default();
        let (w, h) = (96usize, 92usize); // odd dims on purpose
        let cw = (w + 1) / 2;
        for planar in [false, true] {
            let (_, img) = grad_image(w, h, planar);
            let c = table(spec);
            let mut out = vec![0u8; w * h * 3];
            convert_rows(&img, &mut out, w * 3, 3, &c, Kernel::Auto);
            let mut bad = 0usize;
            for r in 0..h {
                for x in 0..w {
                    let yv = ((x as u32 * 255) / w as u32) as u8;
                    let cb_idx = (r >> 1) * cw + (x >> 1);
                    let cb = ((cb_idx as u32 * 400) % 256) as u8;
                    let cr = ((cb_idx as u32 * 900) % 256) as u8;
                    let (fr, fg, fb) = f64_ref(spec, yv, cb, cr);
                    let (gr, gg, gb) = (
                        out[(r * w + x) * 3] as i32,
                        out[(r * w + x) * 3 + 1] as i32,
                        out[(r * w + x) * 3 + 2] as i32,
                    );
                    if [fr, fg, fb].iter().zip([gr, gg, gb].iter()).any(|(a, b)| a.abs_diff(*b) > 1) {
                        bad += 1;
                    }
                }
            }
            assert_eq!(bad, 0, "{bad} pixels differ by >1 from the f64 reference (planar={planar})");
        }
    }

    #[test]
    fn rgba_alpha_is_0xff() {
        let (_, img) = grad_image(128, 96, true);
        let mut out = vec![0u8; 128 * 96 * 4];
        yuv_to_rgb(&img, ColorSpec::default(), RgbChannels::Rgba32, &mut out).unwrap();
        for i in (3..out.len()).step_by(4) {
            assert_eq!(out[i], 255);
        }
    }

    #[test]
    fn downcast_p010_is_exact() {
        let (w, h) = (32usize, 16usize);
        let cw = w / 2;
        let chh = h / 2;
        let yv = |i: usize| ((i as u32 * 7) % 1024) as u16;
        let u10 = |r: usize, x: usize| (900u32 + r as u32 + x as u32) as u16;
        let v10 = |r: usize, x: usize| (100u32 + (r * 3) as u32 + x as u32) as u16;
        let mut buf: Vec<u8> = Vec::new();
        for i in 0..w * h {
            buf.extend_from_slice(&(yv(i) << 6).to_le_bytes());
        }
        for r in 0..chh {
            for x in 0..cw {
                buf.extend_from_slice(&(u10(r, x) << 6).to_le_bytes());
                buf.extend_from_slice(&(v10(r, x) << 6).to_le_bytes());
            }
        }
        let y_slice = &buf[..w * h * 2];
        let uv_slice = &buf[w * h * 2..];
        let img = YuvImage::semi(y_slice, w * 2, uv_slice, w * 2, w, h, 10);
        let mut out = vec![0u8; i420_size(w, h)];
        yuv_high_to_i420(&img, &mut out).unwrap();
        for r in 0..h {
            for x in 0..w {
                let expect = |v: u16| v.min(255) as u8;
                assert_eq!(out[r * w + x], expect(yv(r * w + x)), "Y pixel {r},{x}");
            }
        }
        let cb_base = w * h;
        let cr_base = w * h + cw * chh;
        for r in 0..chh {
            for x in 0..cw {
                let expect = |v: u16| v.min(255) as u8;
                assert_eq!(out[cb_base + r * cw + x], expect(u10(r, x)), "Cb pixel {r},{x}");
                assert_eq!(out[cr_base + r * cw + x], expect(v10(r, x)), "Cr pixel {r},{x}");
            }
        }
    }

    /// The AVX2 down-cast tail (row widths not divisible by 16) must produce
    /// the exact shifted values. Regression: the tail once lacked its
    /// `x += 1` and spun forever on e.g. 8x8 planar P010.
    #[test]
    fn downcast_unaligned_widths_are_exact() {
        let val = |i: usize| ((i as u32 * 37) % 1024) as u16;
        for planar in [false, true] {
            for (w, h) in [(8usize, 8usize), (24usize, 16usize), (33usize, 17usize)] {
                let cw = (w + 1) / 2;
                let chh = (h + 1) / 2;
                // Luma values, then Cb/Cr values (interleaved for semi).
                let yv = |i: usize| val(i);
                let uv = |plane: usize, i: usize| val(w * h * (plane + 1) + i);
                let mut buf: Vec<u8> = Vec::new();
                for i in 0..w * h {
                    buf.extend_from_slice(&(yv(i) << 6).to_le_bytes());
                }
                if planar {
                    for i in 0..cw * chh {
                        buf.extend_from_slice(&(uv(0, i) << 6).to_le_bytes());
                    }
                    for i in 0..cw * chh {
                        buf.extend_from_slice(&(uv(1, i) << 6).to_le_bytes());
                    }
                } else {
                    for i in 0..cw * chh {
                        buf.extend_from_slice(&(uv(0, i) << 6).to_le_bytes());
                        buf.extend_from_slice(&(uv(1, i) << 6).to_le_bytes());
                    }
                }
                let y_len = w * h * 2;
                let (y, rest) = buf.split_at(y_len);
                let img = if planar {
                    let u_len = cw * chh * 2;
                    let (u, v) = rest.split_at(u_len);
                    YuvImage::planar(y, w * 2, u, cw * 2, v, cw * 2, w, h, 10)
                } else {
                    YuvImage::semi(y, w * 2, rest, cw * 4, w, h, 10)
                };
                let mut out = vec![0u8; i420_size(w, h)];
                yuv_high_to_i420(&img, &mut out).unwrap();
                for r in 0..h {
                    for x in 0..w {
                        assert_eq!(
                            out[r * w + x],
                            yv(r * w + x).min(255) as u8,
                            "planar={planar} {w}x{h} Y {r},{x}"
                        );
                    }
                }
                let cb_base = w * h;
                let cr_base = cb_base + cw * chh; // down-cast output is always planar I420
                for r in 0..chh {
                    for x in 0..cw {
                        let i = r * cw + x;
                        assert_eq!(out[cb_base + i], uv(0, i).min(255) as u8, "planar={planar} {w}x{h} Cb {r},{x}");
                        assert_eq!(out[cr_base + i], uv(1, i).min(255) as u8, "planar={planar} {w}x{h} Cr {r},{x}");
                    }
                }
            }
        }
    }
}

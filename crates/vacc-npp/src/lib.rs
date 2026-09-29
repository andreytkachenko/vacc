//! NVIDIA NPP backend for the vacc image pipeline (Y'CbCr -> RGB, resize).
//!
//! This crate implements the same [`ImageConfig`](vacc_image::ImageConfig)
//! pipeline as the SIMD software reference in `vacc-image`
//! (downcast -> scale -> rgb), using classic NPP (`nppi..._Ctx`) entry points
//! resolved at runtime through `libloading`:
//!
//! - resize: `nppiResize_8u_C1R_Ctx` per 4:2:0 plane (luma + Cb + Cr);
//! - color conversion: `nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx` with a
//!   twist matrix derived from the exact Q14 coefficient tables of
//!   [`ColorSpec`](vacc_image::ColorSpec), so NPP applies the same
//!   coefficients as the software kernels (float vs fixed-point rounding may
//!   differ by +-1 LSB).
//!
//! There is no build-time dependency on NPP (it is resolved at runtime). The
//! CUDA *driver* is linked at build time when available (see `build.rs`):
//! the driver's primary context only works when reached through the library
//! PLT. If the driver was not linked, or a library/symbol/device is
//! unavailable, [`Npp::load`] fails and callers fall back to
//! `vacc_image::process`.
//!
//! NPP 13 `_Ctx` entry points operate on **device memory**: this crate
//! allocates device buffers and enqueues async H2D/D2H copies on its own
//! CUDA stream around each call, then synchronizes the stream before
//! returning results to the host.
//!
//! Differences from the software pipeline (expected, tested with tolerance):
//! - `Interpolation::Box` maps to NPP linear interpolation (NPP has no
//!   area-average filter), so downscaled Box results are not identical.
//! - NPP cubic is its own 4-tap kernel, not the Mitchell (B=0.5, C=0.5) used
//!   by the software pipeline.
//! - Affine warp is rejected up front: `nppiWarpAffine_8u_C1R_Ctx`
//!   segfaults on NPP 13.1 even for an identity transform with valid buffers
//!   (reproduced in a standalone C program), so [`process`] returns
//!   [`ImageError::Unsupported`] for any warp request and the caller falls
//!   back to Vulkan compute / software. [`Npp::warp_rgb`] remains available
//!   for hosts whose NPP build handles it.

mod ffi;

use std::os::raw::{c_int, c_void};
use std::sync::OnceLock;

use thiserror::Error;
use vacc_image::{
    i420_size, scratch_view, table, yuv_high_to_i420, ColorSpec, ImageConfig, ImageError,
    ImageResult, Interpolation, ProcessedFrame, RgbChannels, RgbOutput, Scale, YuvImage, YuvLayout,
    YuvOutput,
};

use crate::ffi::{
    Ffi, NppiRect, NppiSize, NPPI_INTER_CUBIC, NPPI_INTER_LINEAR, NPPI_INTER_NEAREST, NPP_SUCCESS,
};
use vacc_image::warp::Affine;

/// Errors from loading or calling NPP.
#[derive(Debug, Error)]
pub enum NppError {
    /// None of the candidate library names could be opened.
    #[error("NPP library not found (tried {names}): {detail}")]
    LibraryNotFound { names: String, detail: String },

    /// A required symbol was not exported.
    #[error("NPP symbol `{symbol}` missing: {detail}")]
    SymbolMissing { symbol: String, detail: String },

    /// The CUDA driver was not linked into this build (non-NVIDIA host).
    #[error("CUDA driver not linked at build time; NPP backend unavailable")]
    CudaNotLinked,

    /// `cuInit` failed.
    #[error("CUDA driver initialization failed")]
    CudaInitFailed,

    /// No CUDA device is present.
    #[error("no CUDA device available")]
    NoCudaDevice,

    /// An NPP call returned a non-success status.
    #[error("NPP call `{fn_name}` failed with status {status}")]
    Status { fn_name: &'static str, status: i32 },
}

/// A loaded NPP backend (resolved entry points + stream context).
pub struct Npp {
    ffi: Ffi,
}

impl std::fmt::Debug for Npp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Npp").finish_non_exhaustive()
    }
}

static GLOBAL: OnceLock<Option<Npp>> = OnceLock::new();

impl Npp {
    /// Load the NPP libraries, initialize CUDA device 0, and create the
    /// non-blocking stream all NPP calls are enqueued on.
    pub fn load() -> Result<Self, NppError> {
        Ok(Self { ffi: Ffi::load()? })
    }

    /// Make the primary context current on the calling thread. Driver API
    /// calls (alloc, copies, sync) require it; a context does not carry over
    /// across threads.
    fn ensure_ctx(&self) -> ImageResult<()> {
        let status = self.ffi.set_current();
        if status != 0 {
            return Err(cuda_err("cuCtxSetCurrent", status));
        }
        Ok(())
    }

    /// Diagnostics probe (examples only).
    #[doc(hidden)]
    pub fn debug_probe(&self) -> String {
        let (st_get, cur, flags) = self.ffi.get_current();
        let same = !cur.is_null() && cur == self.ffi.primary_ctx;
        let st_alloc = match self.ffi.mem_alloc(64) {
            Ok(p) => {
                self.ffi.mem_free_async(p);
                0
            }
            Err(st) => st,
        };
        format!(
            "get_current: status={st_get} cur={cur:?} flags={flags} primary={:?} same={same}; alloc status={st_alloc}",
            self.ffi.primary_ctx
        )
    }

    /// Synchronize the NPP stream. NPP calls with a user stream context are
    /// asynchronous; outputs must not be read before this returns.
    fn sync(&self) -> ImageResult<()> {
        let status = self.ffi.stream_sync();
        if status != 0 {
            return Err(ImageError::Unsupported(format!(
                "cuStreamSynchronize failed with status {status}"
            )));
        }
        Ok(())
    }

    /// The process-wide backend, loaded lazily on first use.
    pub fn global() -> Option<&'static Npp> {
        match GLOBAL.get_or_init(|| Npp::load().ok()) {
            Some(npp) => Some(npp),
            None => None,
        }
    }

    /// Whether NPP (libraries + a CUDA device) is available on this host.
    pub fn is_available() -> bool {
        Self::global().is_some()
    }

    /// Resize an 8-bit 4:2:0 frame into `dst` with the same tight layout
    /// convention as [`vacc_image::resize_yuv`] (planar I420 in -> I420 out,
    /// semi-planar NV12 in -> NV12 out).
    pub fn resize_yuv(&self, src: &YuvImage, scale: Scale, dst: &mut [u8]) -> ImageResult<()> {
        if src.bits_per_sample != 8 {
            return Err(ImageError::Unsupported(format!(
                "NPP resize supports 8-bit sources, got {}-bit (down-cast first)",
                src.bits_per_sample
            )));
        }
        self.ensure_ctx()?;
        let (dw, dh) = (scale.width as usize, scale.height as usize);
        if dw == 0 || dh == 0 {
            return Err(ImageError::InvalidDimensions("scale target must be non-zero".into()));
        }
        let semi = src.cr.is_none();
        let (sw, sh) = (src.width, src.height);
        let (swc, shc) = (src.chroma_width(), src.chroma_height());
        let (dwc, dhc) = ((dw + 1) / 2, (dh + 1) / 2);

        if semi {
            let need = dw * dh + dwc * 2 * dhc;
            if dst.len() < need {
                return Err(ImageError::OutputTooSmall { need, have: dst.len() });
            }
        } else {
            let need = i420_size(dw, dh);
            if dst.len() < need {
                return Err(ImageError::OutputTooSmall { need, have: dst.len() });
            }
        }

        let interp = interp(scale.filter);

        // Luma.
        self.resize_plane(src.y, src.y_pitch, sw, sh, &mut dst[..dw * dh], dw, dh, interp)?;

        // Chroma: Cb and Cr at half resolution.
        if semi {
            // De-interleave the interleaved UV plane into tight U/V rows,
            // resize each, re-interleave into the destination.
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
            let uv = &mut dst[dw * dh..];
            let mut u_tmp = vec![0u8; dwc * dhc];
            self.resize_plane(&u_src, swc, swc, shc, &mut u_tmp, dwc, dhc, interp)?;
            let mut v_tmp = vec![0u8; dwc * dhc];
            self.resize_plane(&v_src, swc, swc, shc, &mut v_tmp, dwc, dhc, interp)?;
            for oy in 0..dhc {
                for ox in 0..dwc {
                    uv[oy * dwc * 2 + ox * 2] = u_tmp[oy * dwc + ox];
                    uv[oy * dwc * 2 + ox * 2 + 1] = v_tmp[oy * dwc + ox];
                }
            }
        } else {
            let cr_off = dwc * dhc;
            self.resize_plane(
                src.cb,
                src.cb_pitch,
                swc,
                shc,
                &mut dst[dw * dh..dw * dh + cr_off],
                dwc,
                dhc,
                interp,
            )?;
            self.resize_plane(
                src.cr.expect("planar layout requires a Cr plane"),
                src.cr_pitch,
                swc,
                shc,
                &mut dst[dw * dh + cr_off..],
                dwc,
                dhc,
                interp,
            )?;
        }
        Ok(())
    }

    /// Convert an 8-bit 4:2:0 frame (planar I420 or semi-planar NV12) to
    /// packed RGB(R)(A) using the exact coefficients of `spec`.
    pub fn yuv_to_rgb(&self, src: &YuvImage, spec: ColorSpec, channels: RgbChannels, dst: &mut [u8]) -> ImageResult<()> {
        if src.bits_per_sample != 8 {
            return Err(ImageError::Unsupported(format!(
                "NPP conversion supports 8-bit sources, got {}-bit (down-cast first)",
                src.bits_per_sample
            )));
        }
        self.ensure_ctx()?;
        let (w, h) = (src.width, src.height);
        let chb = channels.bytes() as usize;
        let need = w * h * chb;
        if dst.len() < need {
            return Err(ImageError::OutputTooSmall { need, have: dst.len() });
        }

        let c = table(spec);
        let q = 1.0f32 / 16384.0;
        // dst[ch] = ky*Y + cb_k*Cb + cr_k*Cr + off, in Q14 (see `conv_px`).
        let twist: [f32; 12] = [
            c.ky as f32 * q,
            0.0,
            c.r_cr as f32 * q,
            (c.r_off + vacc_image::RND) as f32 * q,
            c.ky as f32 * q,
            c.g_cb as f32 * q,
            c.g_cr as f32 * q,
            (c.g_off + vacc_image::RND) as f32 * q,
            c.ky as f32 * q,
            c.b_cb as f32 * q,
            0.0,
            (c.b_off + vacc_image::RND) as f32 * q,
        ];

        // NPP's NV12 conversion takes device pointers [Y, interleaved UV];
        // compact each plane to a tight row layout (step = width) first.
        let cw = src.chroma_width();
        let chh = src.chroma_height();
        let mut tight_y = Vec::new();
        let y_view: &[u8] = if src.y_pitch == w {
            src.y
        } else {
            tight_y.resize(w * h, 0);
            for r in 0..h {
                tight_y[r * w..(r + 1) * w].copy_from_slice(&src.y[r * src.y_pitch..r * src.y_pitch + w]);
            }
            &tight_y
        };

        let mut uv = vec![0u8; cw * 2 * chh];
        if src.cr.is_some() {
            // Planar: interleave the Cb and Cr rows.
            for r in 0..chh {
                let crow = &src.cb[r * src.cb_pitch..r * src.cb_pitch + cw];
                let rrow = &src.cr.expect("planar layout requires a Cr plane")[r * src.cr_pitch..r * src.cr_pitch + cw];
                for x in 0..cw {
                    uv[r * cw * 2 + x * 2] = crow[x];
                    uv[r * cw * 2 + x * 2 + 1] = rrow[x];
                }
            }
        } else {
            for r in 0..chh {
                uv[r * cw * 2..(r + 1) * cw * 2]
                    .copy_from_slice(&src.cb[r * src.cb_pitch..r * src.cb_pitch + cw * 2]);
            }
        }

        let d_y = DevBuf::upload(&self.ffi, y_view)?;
        let d_uv = DevBuf::upload(&self.ffi, &uv)?;
        let rgb_size = w * h * 3;
        let d_rgb = DevBuf::alloc(&self.ffi, rgb_size)?;

        let status = unsafe {
            (self.ffi.nv12_to_rgb_twist)(
                [d_y.ptr as *const u8, d_uv.ptr as *const u8].as_ptr(),
                [w as c_int, (cw * 2) as c_int].as_ptr(),
                d_rgb.ptr as *mut u8,
                (w * 3) as c_int,
                NppiSize { n_width: w as c_int, n_height: h as c_int },
                twist.as_ptr(),
                self.ffi.ctx,
            )
        };
        if status != NPP_SUCCESS {
            return Err(npp_status_error("nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx", status));
        }

        // The twist call produces 3 channels; RGBA is expanded afterwards.
        if chb == 3 {
            d_rgb.download(&mut dst[..rgb_size])?;
        } else {
            let mut scratch = vec![0u8; rgb_size];
            d_rgb.download(&mut scratch)?;
            for y in 0..h {
                let srow = &scratch[y * w * 3..(y + 1) * w * 3];
                let drow = &mut dst[y * w * chb..(y + 1) * w * chb];
                for x in 0..w {
                    let s = &srow[x * 3..x * 3 + 3];
                    let d = &mut drow[x * chb..x * chb + 4];
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                    d[3] = 255;
                }
            }
        }
        // Drop order enqueues the stream-ordered frees after the download.
        drop(d_rgb);
        drop(d_uv);
        drop(d_y);
        self.sync()
    }
}

fn interp(interpolation: Interpolation) -> c_int {
    match interpolation {
        Interpolation::Nearest => NPPI_INTER_NEAREST,
        Interpolation::Bilinear => NPPI_INTER_LINEAR,
        Interpolation::Bicubic => NPPI_INTER_CUBIC,
        // NPP has no area-averaging filter; linear is the closest available.
        Interpolation::Box => NPPI_INTER_LINEAR,
    }
}

fn npp_status_error(fn_name: &'static str, status: c_int) -> ImageError {
    ImageError::Unsupported(NppError::Status { fn_name, status }.to_string())
}

/// Apply `cfg` to the decoded frame `src` using NPP (downcast -> scale -> rgb).
///
/// Returns `Err` whenever NPP is unavailable or a call fails; callers should
/// then fall back to [`vacc_image::process`].
pub fn process(src: &YuvImage, cfg: &ImageConfig) -> ImageResult<ProcessedFrame> {
    if cfg.is_noop() {
        return Err(ImageError::Unsupported(
            "no-op ImageConfig: neither rgb nor scale is set".into(),
        ));
    }
    let npp = Npp::global()
        .ok_or_else(|| ImageError::Unsupported("NPP unavailable on this host".into()))?;

    // Warp is disabled for the pipeline: nppiWarpAffine_8u_C1R_Ctx segfaults
    // on NPP 13.1 (see module docs). Rejecting here routes the warp to the
    // Vulkan compute or software backend instead of crashing the process.
    if cfg.affine.is_some() {
        return Err(ImageError::Unsupported(
            "NPP warp is disabled (nppiWarpAffine crashes on this NPP build)".into(),
        ));
    }

    match (cfg.scale, cfg.affine, cfg.rgb) {
        (None, None, Some(ch)) => {
            let (w, h) = (src.width, src.height);
            let mut data = vec![0u8; w * h * ch.bytes() as usize];
            if src.bits_per_sample == 8 {
                npp.yuv_to_rgb(src, cfg.spec, ch, &mut data)?;
            } else {
                let mid = downcast(src)?;
                let view = mid.view();
                npp.yuv_to_rgb(&view, cfg.spec, ch, &mut data)?;
            }
            Ok(ProcessedFrame::Rgb(RgbOutput {
                width: w as u32,
                height: h as u32,
                channels: ch.bytes(),
                data,
            }))
        }
        (Some(scale), None, None) => {
            let (buf, layout, w, h) = scaled(npp, src, scale)?;
            Ok(ProcessedFrame::Yuv(YuvOutput {
                width: w as u32,
                height: h as u32,
                layout,
                data: buf,
            }))
        }
        (Some(scale), None, Some(ch)) => {
            let (buf, layout, w, h) = scaled(npp, src, scale)?;
            let view = tight_view(&buf, layout, w, h);
            let mut data = vec![0u8; w * h * ch.bytes() as usize];
            npp.yuv_to_rgb(&view, cfg.spec, ch, &mut data)?;
            Ok(ProcessedFrame::Rgb(RgbOutput {
                width: w as u32,
                height: h as u32,
                channels: ch.bytes(),
                data,
            }))
        }
        // Warp paths: convert to RGB first, then warp.
        (None, Some(transform), Some(ch)) => {
            let (w, h) = (src.width, src.height);
            let mut rgb_data = vec![0u8; w * h * 4];
            if src.bits_per_sample == 8 {
                npp.yuv_to_rgb(src, cfg.spec, RgbChannels::Rgba32, &mut rgb_data)?;
            } else {
                let mid = downcast(src)?;
                let view = mid.view();
                npp.yuv_to_rgb(&view, cfg.spec, RgbChannels::Rgba32, &mut rgb_data)?;
            }
            let src_img = vacc_image::RgbImage::new(&rgb_data, w * 4, w, h, 4);
            let mut warped = vec![0u8; w * h * 4];
            npp.warp_rgb(&src_img, transform.transform, transform.interpolation, &mut warped)?;
            if ch == RgbChannels::Rgb24 {
                let mut rgb24 = vec![0u8; w * h * 3];
                for y in 0..h {
                    for x in 0..w {
                        let src_off = (y * w + x) * 4;
                        let dst_off = (y * w + x) * 3;
                        rgb24[dst_off] = warped[src_off];
                        rgb24[dst_off + 1] = warped[src_off + 1];
                        rgb24[dst_off + 2] = warped[src_off + 2];
                    }
                }
                Ok(ProcessedFrame::Rgb(RgbOutput {
                    width: w as u32,
                    height: h as u32,
                    channels: 3,
                    data: rgb24,
                }))
            } else {
                Ok(ProcessedFrame::Rgb(RgbOutput {
                    width: w as u32,
                    height: h as u32,
                    channels: 4,
                    data: warped,
                }))
            }
        }
        (Some(scale), Some(transform), Some(ch)) => {
            let (buf, layout, w, h) = scaled(npp, src, scale)?;
            let view = tight_view(&buf, layout, w, h);
            let mut rgb_data = vec![0u8; w * h * 4];
            npp.yuv_to_rgb(&view, cfg.spec, RgbChannels::Rgba32, &mut rgb_data)?;
            let src_img = vacc_image::RgbImage::new(&rgb_data, w * 4, w, h, 4);
            let mut warped = vec![0u8; w * h * 4];
            npp.warp_rgb(&src_img, transform.transform, transform.interpolation, &mut warped)?;
            if ch == RgbChannels::Rgb24 {
                let mut rgb24 = vec![0u8; w * h * 3];
                for y in 0..h {
                    for x in 0..w {
                        let src_off = (y * w + x) * 4;
                        let dst_off = (y * w + x) * 3;
                        rgb24[dst_off] = warped[src_off];
                        rgb24[dst_off + 1] = warped[src_off + 1];
                        rgb24[dst_off + 2] = warped[src_off + 2];
                    }
                }
                Ok(ProcessedFrame::Rgb(RgbOutput {
                    width: w as u32,
                    height: h as u32,
                    channels: 3,
                    data: rgb24,
                }))
            } else {
                Ok(ProcessedFrame::Rgb(RgbOutput {
                    width: w as u32,
                    height: h as u32,
                    channels: 4,
                    data: warped,
                }))
            }
        }
        (_, Some(_), None) => {
            Err(ImageError::Unsupported(
                "affine warp on YUV requires RGB output".into(),
            ))
        }
        (None, None, None) => unreachable!("no-op configs are rejected above"),
    }
}

/// Stage 1+2 via NPP: optional down-cast followed by the resize, into an
/// owned tight buffer (layout preserved; the down-cast path yields I420).
fn scaled(npp: &Npp, src: &YuvImage, scale: Scale) -> ImageResult<(Vec<u8>, YuvLayout, usize, usize)> {
    let (dw, dh) = (scale.width as usize, scale.height as usize);
    if dw == 0 || dh == 0 {
        return Err(ImageError::InvalidDimensions("scale target must be non-zero".into()));
    }
    if src.bits_per_sample == 8 {
        let layout = if src.cr.is_some() { YuvLayout::Planar } else { YuvLayout::Semi };
        let mut buf = vec![0u8; out_size(layout, dw, dh)];
        npp.resize_yuv(src, scale, &mut buf)?;
        Ok((buf, layout, dw, dh))
    } else {
        let mid = downcast(src)?;
        let view = mid.view();
        let mut buf = vec![0u8; i420_size(dw, dh)];
        npp.resize_yuv(&view, scale, &mut buf)?;
        Ok((buf, YuvLayout::Planar, dw, dh))
    }
}

/// P010/P012 -> exact 8-bit I420 (shared with the software pipeline).
struct Downcasted {
    buf: Vec<u8>,
    width: usize,
    height: usize,
}

impl Downcasted {
    fn view(&self) -> YuvImage<'_> {
        scratch_view(&self.buf, self.width, self.height)
    }
}

fn downcast(src: &YuvImage) -> ImageResult<Downcasted> {
    let mut buf = vec![0u8; i420_size(src.width, src.height)];
    yuv_high_to_i420(src, &mut buf)?;
    Ok(Downcasted {
        buf,
        width: src.width,
        height: src.height,
    })
}

/// Byte length of a tight 4:2:0 buffer.
fn out_size(layout: YuvLayout, w: usize, h: usize) -> usize {
    let cw = (w + 1) / 2;
    let ch = (h + 1) / 2;
    match layout {
        YuvLayout::Planar => i420_size(w, h),
        YuvLayout::Semi => w * h + 2 * cw * ch,
    }
}

/// A view over a tight 8-bit 4:2:0 buffer.
fn tight_view(buf: &[u8], layout: YuvLayout, w: usize, h: usize) -> YuvImage<'_> {
    match layout {
        YuvLayout::Planar => scratch_view(buf, w, h),
        YuvLayout::Semi => {
            let cw = (w + 1) / 2;
            YuvImage::semi(&buf[..w * h], w, &buf[w * h..], cw * 2, w, h, 8)
        }
    }
}



/// One 4:2:0 plane resize call (device-memory NPP).
impl Npp {

/// Warp an RGB image using the given affine transformation.
///
/// Warning: `nppiWarpAffine_8u_C1R_Ctx` segfaults on some NPP builds
/// (NPP 13.1 on GA106, even for identity transforms); the pipeline
/// therefore never calls this and routes warp to Vulkan compute /
/// software instead.
pub fn warp_rgb(
    &self,
    src: &vacc_image::RgbImage,
    transform: Affine,
    interpolation: Interpolation,
    dst: &mut [u8],
) -> ImageResult<()> {
    self.ensure_ctx()?;
    let (w, h) = (src.width, src.height);
    let ch = src.channels as usize;
    if ch != 3 && ch != 4 {
        return Err(ImageError::Unsupported(format!("unsupported channel count {ch}")));
    }
    let need = w * h * ch;
    if dst.len() < need {
        return Err(ImageError::OutputTooSmall { need, have: dst.len() });
    }

    // NPP uses inverse mapping: provide the inverse of the forward transform.
    let inv = transform.invert().ok_or_else(|| {
        ImageError::Unsupported("singular affine matrix".into())
    })?;

    // Upload source.
    let d_src = DevBuf::upload(&self.ffi, src.pixels())?;
    let d_dst = DevBuf::alloc(&self.ffi, need)?;

    // Warp each channel separately (NPP's warp_affine is single-channel).
    // For RGB, we process 3 channels; for RGBA, 4.
    for c in 0..ch {
        // Extract channel plane from source (interleaved -> planar on host).
        let mut channel_plane = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                channel_plane[y * w + x] = src.pixels()[y * src.pitch + x * ch + c];
            }
        }
        let d_ch_src = DevBuf::upload(&self.ffi, &channel_plane)?;

        // NPP warp coefficients: [a, b, c, d, e, f] for inverse mapping
        // x_src = a*x_dst + b*y_dst + c
        // y_src = d*x_dst + e*y_dst + f
        let coeffs: [f32; 6] = [
            inv.m00, inv.m01, inv.m02,
            inv.m10, inv.m11, inv.m12,
        ];
        // Upload coefficients as raw bytes (6 * 4 = 24 bytes).
        let coeffs_bytes: Vec<u8> = coeffs.iter().flat_map(|f| f.to_ne_bytes()).collect();
        let d_coeffs = DevBuf::upload(&self.ffi, &coeffs_bytes)?;

        let status = unsafe {
            (self.ffi.warp_affine_c1r)(
                d_ch_src.ptr as *const u8,
                w as c_int,
                NppiSize { n_width: w as c_int, n_height: h as c_int },
                NppiRect { n_x: 0, n_y: 0, n_width: w as c_int, n_height: h as c_int },
                d_dst.ptr as *mut u8,
                w as c_int,
                NppiSize { n_width: w as c_int, n_height: h as c_int },
                NppiRect { n_x: 0, n_y: 0, n_width: w as c_int, n_height: h as c_int },
                d_coeffs.ptr as *const f32,
                interp(interpolation),
                self.ffi.ctx,
            )
        };
        if status != NPP_SUCCESS {
            return Err(npp_status_error("nppiWarpAffine_8u_C1R_Ctx", status));
        }

        drop(d_ch_src);
        drop(d_coeffs);
    }

    // Download result.
    d_dst.download(dst)?;
    drop(d_dst);
    drop(d_src);
    self.sync()
}
    fn resize_plane(
        &self,
        src: &[u8],
        src_pitch: usize,
        sw: usize,
        sh: usize,
        dst: &mut [u8],
        dw: usize,
        dh: usize,
        interp: c_int,
    ) -> ImageResult<()> {
        assert_eq!(dst.len(), dw * dh, "tight destination expected");
        // NPP 13 operates on device memory: compact the source rows to step
        // `sw`, upload, resize, and download into the tight destination.
        let mut tight_src = Vec::new();
        let src_view: &[u8] = if src_pitch == sw {
            src
        } else {
            tight_src.resize(sw * sh, 0);
            for r in 0..sh {
                tight_src[r * sw..(r + 1) * sw].copy_from_slice(&src[r * src_pitch..r * src_pitch + sw]);
            }
            &tight_src
        };

        let d_src = DevBuf::upload(&self.ffi, src_view)?;
        let d_dst = DevBuf::alloc(&self.ffi, dw * dh)?;

        let status = unsafe {
            (self.ffi.resize_c1r)(
                d_src.ptr as *const u8,
                sw as c_int,
                NppiSize { n_width: sw as c_int, n_height: sh as c_int },
                NppiRect { n_x: 0, n_y: 0, n_width: sw as c_int, n_height: sh as c_int },
                d_dst.ptr as *mut u8,
                dw as c_int,
                NppiSize { n_width: dw as c_int, n_height: dh as c_int },
                NppiRect { n_x: 0, n_y: 0, n_width: dw as c_int, n_height: dh as c_int },
                interp,
                self.ffi.ctx,
            )
        };
        if status != NPP_SUCCESS {
            return Err(npp_status_error("nppiResize_8u_C1R_Ctx", status));
        }
        d_dst.download(dst)?;
        // Drop order enqueues the stream-ordered frees after the download.
        drop(d_dst);
        drop(d_src);
        self.sync()
    }
}

/// A device buffer on the NPP stream. The H2D upload (if any) is enqueued at
/// construction, [`DevBuf::download`] enqueues the D2H copy, and dropping
/// enqueues a stream-ordered free. Everything is ordered on the same stream,
/// so a final [`Npp::sync`] after all downloads makes results visible to the
/// host.
struct DevBuf<'a> {
    ffi: &'a Ffi,
    ptr: usize,
}

impl<'a> DevBuf<'a> {
    fn alloc(ffi: &'a Ffi, size: usize) -> ImageResult<Self> {
        let ptr = ffi.mem_alloc(size).map_err(|st| cuda_err("cuMemAlloc_v2", st))?;
        Ok(Self { ffi, ptr })
    }

    /// Allocate and enqueue an async H2D copy of `host`.
    fn upload(ffi: &'a Ffi, host: &[u8]) -> ImageResult<Self> {
        let buf = Self::alloc(ffi, host.len())?;
        let st = ffi.h2d_async(buf.ptr, host.as_ptr() as *const c_void, host.len());
        if st != 0 {
            return Err(cuda_err("cuMemcpyHtoDAsync_v2", st));
        }
        Ok(buf)
    }

    /// Enqueue an async D2H copy of the buffer into `dst`.
    fn download(&self, dst: &mut [u8]) -> ImageResult<()> {
        let st = self.ffi.d2h_async(dst.as_mut_ptr() as *mut c_void, self.ptr, dst.len());
        if st != 0 {
            return Err(cuda_err("cuMemcpyDtoHAsync_v2", st));
        }
        Ok(())
    }
}

impl Drop for DevBuf<'_> {
    fn drop(&mut self) {
        // Stream-ordered: safe even before the final sync.
        let _ = self.ffi.mem_free_async(self.ptr);
    }
}

fn cuda_err(fn_name: &'static str, status: c_int) -> ImageError {
    ImageError::Unsupported(format!("CUDA call `{fn_name}` failed with status {status}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serialize the NPP-using tests: they share one CUDA context and the
    /// default stream, and concurrent kernel launches can interleave.
    static GPU_LOCK: Mutex<()> = Mutex::new(());

    fn require_npp() -> Option<std::sync::MutexGuard<'static, ()>> {
        if !Npp::is_available() {
            eprintln!("NPP unavailable; skipping");
            return None;
        }
        // Recover from poisoning so one failing test doesn't mask the real
        // failures of the others (they serialize on this lock).
        Some(GPU_LOCK.lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// Deterministic gradient 8-bit 4:2:0 (owned tight buffer + static view).
    fn grad_yuv(w: usize, h: usize, planar: bool) -> (Vec<u8>, YuvImage<'static>) {
        let y: Vec<u8> = (0..w * h)
            .map(|i| ((i % w) as u32 * 255 / w.max(1) as u32) as u8)
            .collect();
        let cw = (w + 1) / 2;
        let chh = (h + 1) / 2;
        // Smooth diagonal ramps that never wrap: NPP's resize phase
        // convention differs from the software's (corner- vs center-based),
        // so the comparison signal must not contain high-frequency edges.
        // A modulo-wrapped ramp would create full-range step discontinuities
        // where midpoint interpolation and exact-pixel picks legitimately
        // diverge by up to 255.
        let cb: Vec<u8> = (0..chh)
            .flat_map(|r| (
                (0..cw).map(move |x| ((x + r) * 255 / (cw + chh - 1)) as u8)
            ))
            .collect();
        let cr: Vec<u8> = (0..chh)
            .flat_map(|r| (
                (0..cw).map(move |x| ((x * 2 + r) * 255 / (2 * cw + chh - 1)) as u8)
            ))
            .collect();
        if planar {
            let mut buf = Vec::new();
            buf.extend_from_slice(&y);
            buf.extend_from_slice(&cb);
            buf.extend_from_slice(&cr);
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
            let img = YuvImage::semi(&data[..w * h], w, &data[w * h..], cw * 2, w, h, 8);
            (buf, img)
        }
    }

    /// Max/mean absolute difference between two equal-length buffers.
    fn diff(a: &[u8], b: &[u8]) -> (i32, f64) {
        assert_eq!(a.len(), b.len());
        let mut max = 0i32;
        let mut sum = 0i64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (*x as i32 - *y as i32).abs();
            max = max.max(d);
            sum += d as i64;
        }
        (max, sum as f64 / a.len() as f64)
    }

    #[test]
    fn load_on_nvidia_host() {
        if let Some(_guard) = require_npp() {
            assert!(Npp::is_available());
        }
    }

    #[test]
    fn rgb_matches_software_within_tolerance() {
        let Some(_guard) = require_npp() else { return };
        let npp = Npp::global().unwrap();
        for planar in [false, true] {
            let (_, img) = grad_yuv(96, 92, planar);
            for spec in [
                ColorSpec::default(),
                ColorSpec {
                    matrix: vacc_image::MatrixCoefficients::Bt601,
                    range: vacc_image::ColorRange::Full,
                },
            ] {
                for ch in [RgbChannels::Rgb24, RgbChannels::Rgba32] {
                    let mut npp_out = vec![0u8; img.width * img.height * ch.bytes() as usize];
                    npp.yuv_to_rgb(&img, spec, ch, &mut npp_out).unwrap();
                    let mut sw_out = vec![0u8; npp_out.len()];
                    vacc_image::yuv_to_rgb(&img, spec, ch, &mut sw_out).unwrap();
                    let (max, mean) = diff(&npp_out, &sw_out);
                    assert!(
                        max <= 1 && mean < 0.5,
                        "rgb drift planar={planar} spec={spec:?} ch={ch:?}: max={max} mean={mean}"
                    );
                }
            }
        }
    }

    #[test]
    fn resize_matches_software_within_tolerance() {
        let Some(_guard) = require_npp() else { return };
        let npp = Npp::global().unwrap();
        for planar in [false, true] {
            let (_, img) = grad_yuv(320, 240, planar);
            let layout = if planar { YuvLayout::Planar } else { YuvLayout::Semi };
            for (tw, th) in [(200usize, 150), (640, 480), (80, 60)] {
                for filter in [Interpolation::Bilinear, Interpolation::Bicubic] {
                    let scale = Scale::new(tw as u32, th as u32, filter);
                    let need = out_size(layout, tw, th);
                    let mut npp_out = vec![0u8; need];
                    npp.resize_yuv(&img, scale, &mut npp_out).unwrap();
                    let mut sw_out = vec![0u8; need];
                    vacc_image::resize_yuv(&img, scale, vacc_image::Kernel::Auto, &mut sw_out).unwrap();
                    let (max, mean) = diff(&npp_out, &sw_out);
                    // Different kernels and phase conventions (NPP corner-based
                    // linear/cubic vs center-based bilinear/Mitchell): allow a
                    // small per-pixel drift on the smooth ramps. Measured worst
                    // case is the 4x downscale (uniform -1..-4 across planes,
                    // mean ~2.2); structural errors show means in the tens.
                    assert!(
                        max <= 8 && mean < 3.0,
                        "resize drift planar={planar} {tw}x{th} {filter:?}: max={max} mean={mean}"
                    );
                }
            }
        }
    }

    #[test]
    fn pipeline_smoke() {
        let Some(_guard) = require_npp() else { return };
        for planar in [false, true] {
            let (_, img) = grad_yuv(320, 240, planar);
            let cfg = ImageConfig {
                rgb: Some(RgbChannels::Rgb24),
                scale: Some(Scale::new(160, 120, Interpolation::Bilinear)),
                affine: None,
                spec: ColorSpec::auto(1080),
            };
            let npp_res = process(&img, &cfg).unwrap();
            let sw_res = vacc_image::process(&img, &cfg, vacc_image::Kernel::Auto).unwrap();
            match (npp_res, sw_res) {
                (ProcessedFrame::Rgb(a), ProcessedFrame::Rgb(b)) => {
                    assert_eq!((a.width, a.height), (160, 120));
                    let (max, mean) = diff(&a.data, &b.data);
                    assert!(max <= 8 && mean < 2.0, "pipeline drift: max={max} mean={mean}");
                }
                _ => panic!("expected rgb outputs"),
            }
        }
    }
}

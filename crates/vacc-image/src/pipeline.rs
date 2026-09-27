//! `ImageConfig` pipeline: downcast -> scale -> rgb.
//!
//! Applies an [`ImageConfig`] to a decoded 4:2:0 Y'CbCr frame in the
//! documented order (see [`ImageConfig`]):
//!
//! 1. If scaling is requested and the source is P010/P012, the frame is first
//!    down-cast to 8-bit I420.
//! 2. If scaling is requested, the Y'CbCr frame is resized with
//!    `scale.filter`.
//! 3. If RGB is requested, the (possibly resized) frame is converted to packed
//!    RGB(R)(A) using `spec`.
//!
//! This is the reference software implementation of the pipeline. Backends
//! with a fast GPU primitive (NVIDIA: NPP) may execute the same steps on the
//! GPU; this crate remains the fallback and the ground truth used by the
//! verification tooling.

use crate::conv::{i420_size, scratch_view, yuv_high_to_i420, yuv_to_rgb, Kernel, Layout};
use crate::error::{ImageError, ImageResult};
use crate::pixel::{YuvImage, YuvLayout};
use crate::resize::resize_yuv;
use crate::spec::{ImageConfig, Scale};
use crate::warp::warp_rgb;

/// 8-bit 4:2:0 Y'CbCr result of the pipeline.
#[derive(Debug, Clone)]
pub struct YuvOutput {
    /// Width in luma pixels.
    pub width: u32,
    /// Height in luma pixels.
    pub height: u32,
    /// Layout of `data`: tight planar I420 or tight semi-planar NV12.
    pub layout: YuvLayout,
    /// Tightly packed rows (no padding).
    pub data: Vec<u8>,
}

/// Packed RGB(R)(A) result of the pipeline.
#[derive(Debug, Clone)]
pub struct RgbOutput {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Channels per pixel: 3 (RGB24) or 4 (RGBA32).
    pub channels: u8,
    /// Tightly packed rows: `width * height * channels` bytes.
    pub data: Vec<u8>,
}

/// The frame after the pipeline has been applied.
#[derive(Debug, Clone)]
pub enum ProcessedFrame {
    /// The frame stays in 8-bit Y'CbCr (possibly resized).
    Yuv(YuvOutput),
    /// The frame was converted to packed RGB(R)(A) (possibly resized).
    Rgb(RgbOutput),
}

/// Apply `cfg` to the decoded frame `src`, dispatching the resize with
/// `kernel`. See the module docs for the stage order.
pub fn process(src: &YuvImage, cfg: &ImageConfig, kernel: Kernel) -> ImageResult<ProcessedFrame> {
    if cfg.is_noop() {
        return Err(ImageError::Unsupported(
            "no-op ImageConfig: neither rgb nor scale is set".into(),
        ));
    }
    match (cfg.scale, cfg.affine, cfg.rgb) {
        (None, None, Some(ch)) => {
            let (w, h) = (src.width, src.height);
            let mut data = vec![0u8; w * h * ch.bytes() as usize];
            yuv_to_rgb(src, cfg.spec, ch, &mut data)?;
            Ok(ProcessedFrame::Rgb(RgbOutput {
                width: w as u32,
                height: h as u32,
                channels: ch.bytes(),
                data,
            }))
        }
        (Some(scale), None, None) => {
            let s = scaled(src, scale, kernel)?;
            Ok(ProcessedFrame::Yuv(YuvOutput {
                width: s.width as u32,
                height: s.height as u32,
                layout: s.layout,
                data: s.buf,
            }))
        }
        (Some(scale), None, Some(ch)) => {
            let s = scaled(src, scale, kernel)?;
            let view = s.view();
            let (w, h) = (s.width, s.height);
            let mut data = vec![0u8; w * h * ch.bytes() as usize];
            yuv_to_rgb(&view, cfg.spec, ch, &mut data)?;
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
            yuv_to_rgb(src, cfg.spec, crate::spec::RgbChannels::Rgba32, &mut rgb_data)?;
            let src_img = crate::pixel::RgbImage::new(&rgb_data, w * 4, w, h, 4);
            let mut warped = vec![0u8; w * h * 4];
            warp_rgb(&src_img, transform, w, h, &mut warped)?;
            // If RGB24 was requested, drop the alpha channel.
            if ch == crate::spec::RgbChannels::Rgb24 {
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
            let s = scaled(src, scale, kernel)?;
            let view = s.view();
            let (w, h) = (s.width, s.height);
            let mut rgb_data = vec![0u8; w * h * 4];
            yuv_to_rgb(&view, cfg.spec, crate::spec::RgbChannels::Rgba32, &mut rgb_data)?;
            let src_img = crate::pixel::RgbImage::new(&rgb_data, w * 4, w, h, 4);
            let mut warped = vec![0u8; w * h * 4];
            warp_rgb(&src_img, transform, w, h, &mut warped)?;
            if ch == crate::spec::RgbChannels::Rgb24 {
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
        // YUV warp not yet supported; require RGB output.
        (_, Some(_), None) => {
            Err(ImageError::Unsupported(
                "affine warp on YUV requires RGB output".into(),
            ))
        }
        (None, None, None) => unreachable!("no-op configs are rejected above"),
    }
}

/// Stages 1+2: an optional high-bit-depth down-cast followed by the resize.
struct Scaled8 {
    /// Tight 8-bit buffer (I420 or NV12, see `layout`).
    buf: Vec<u8>,
    layout: YuvLayout,
    width: usize,
    height: usize,
}

impl Scaled8 {
    fn view(&self) -> YuvImage<'_> {
        tight_view(&self.buf, self.layout, self.width, self.height)
    }
}

/// Run stages 1+2. The output layout matches the source layout (I420 or
/// NV12); the down-cast path always produces I420.
fn scaled(src: &YuvImage, scale: Scale, kernel: Kernel) -> ImageResult<Scaled8> {
    let (dw, dh) = (scale.width as usize, scale.height as usize);
    if dw == 0 || dh == 0 {
        return Err(ImageError::InvalidDimensions("scale target must be non-zero".into()));
    }
    if src.bits_per_sample == 8 {
        let layout = to_layout(src.layout());
        let mut buf = vec![0u8; out_size(layout, dw, dh)];
        resize_yuv(src, scale, kernel, &mut buf)?;
        Ok(Scaled8 { buf, layout, width: dw, height: dh })
    } else {
        // Stage 1: P010/P012 -> exact 8-bit I420 down-cast.
        let mut mid = vec![0u8; i420_size(src.width, src.height)];
        yuv_high_to_i420(src, &mut mid)?;
        let mid_view = scratch_view(&mid, src.width, src.height);
        // Stage 2: resize the 8-bit frame.
        let mut buf = vec![0u8; i420_size(dw, dh)];
        resize_yuv(&mid_view, scale, kernel, &mut buf)?;
        Ok(Scaled8 {
            buf,
            layout: YuvLayout::Planar,
            width: dw,
            height: dh,
        })
    }
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

fn to_layout(layout: Layout) -> YuvLayout {
    match layout {
        Layout::Planar => YuvLayout::Planar,
        Layout::Semi => YuvLayout::Semi,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{ColorSpec, Filter, MatrixCoefficients, RgbChannels};

    /// Deterministic gradient 8-bit 4:2:0 (owned tight buffer + static view).
    fn grad_yuv(w: usize, h: usize, planar: bool) -> (Vec<u8>, YuvImage<'static>) {
        let y: Vec<u8> = (0..w * h)
            .map(|i| ((i % w) as u32 * 255 / w.max(1) as u32) as u8)
            .collect();
        let cw = (w + 1) / 2;
        let chh = (h + 1) / 2;
        let cb: Vec<u8> = (0..cw * chh).map(|i| ((i as u32) * 400 % 256) as u8).collect();
        let cr: Vec<u8> = (0..cw * chh).map(|i| ((i as u32) * 900 % 256) as u8).collect();
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

    /// P010 semi-planar gradient (u16 LE, top-justified values).
    fn grad_p010(w: usize, h: usize) -> (Vec<u8>, YuvImage<'static>) {
        let cw = (w + 1) / 2;
        let chh = (h + 1) / 2;
        let mut buf = Vec::new();
        for i in 0..w * h {
            let v = (((i % w) as u32 * 1023) / w.max(1) as u32) << 6;
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for i in 0..cw * chh {
            let cb = ((i as u32 * 400) % 1024) << 6;
            let cr = ((i as u32 * 900) % 1024) << 6;
            buf.extend_from_slice(&cb.to_le_bytes());
            buf.extend_from_slice(&cr.to_le_bytes());
        }
        let data: &'static [u8] = Box::leak(buf.clone().into_boxed_slice());
        let img = YuvImage::semi(&data[..w * h * 2], w * 2, &data[w * h * 2..], cw * 4, w, h, 10);
        (buf, img)
    }

    #[test]
    fn noop_rejected() {
        let (_, img) = grad_yuv(64, 48, true);
        assert!(process(&img, &ImageConfig::none(), Kernel::Auto).is_err());
    }

    #[test]
    fn rgb_only_matches_direct_conversion() {
        let spec = crate::spec::ColorSpec::default();
        for planar in [false, true] {
            let (_, img) = grad_yuv(96, 92, planar);
            for ch in [RgbChannels::Rgb24, RgbChannels::Rgba32] {
                let mut expected = vec![0u8; img.width * img.height * ch.bytes() as usize];
                yuv_to_rgb(&img, spec, ch, &mut expected).unwrap();
                let cfg = ImageConfig { rgb: Some(ch), scale: None, affine: None, spec };
                match process(&img, &cfg, Kernel::Auto).unwrap() {
                    ProcessedFrame::Rgb(out) => {
                        assert_eq!((out.width, out.height), (img.width as u32, img.height as u32));
                        assert_eq!(out.channels, ch.bytes());
                        assert_eq!(out.data, expected);
                    }
                    _ => panic!("expected rgb output"),
                }
            }
        }
    }

    #[test]
    fn scale_only_matches_resize_yuv() {
        for planar in [false, true] {
            let (_, img) = grad_yuv(128, 96, planar);
            let layout = if planar { YuvLayout::Planar } else { YuvLayout::Semi };
            for filter in [Filter::Bilinear, Filter::Box, Filter::Bicubic] {
                for (tw, th) in [(300usize, 200usize), (40, 30), (128, 96)] {
                    let scale = Scale::new(tw as u32, th as u32, filter);
                    let mut expected = vec![0u8; out_size(layout, tw, th)];
                    resize_yuv(&img, scale, Kernel::Auto, &mut expected).unwrap();
                    let cfg = ImageConfig {
                        rgb: None,
                        scale: Some(scale),
                        affine: None,
                        spec: ColorSpec::default(),
                    };
                    match process(&img, &cfg, Kernel::Auto).unwrap() {
                        ProcessedFrame::Yuv(out) => {
                            assert_eq!((out.width, out.height), (tw as u32, th as u32));
                            assert_eq!(out.layout, layout);
                            assert_eq!(out.data, expected);
                        }
                        _ => panic!("expected yuv output"),
                    }
                }
            }
        }
    }

    /// Scaling to the source size is an identity for every filter, so
    /// `scale + rgb` must equal a direct conversion.
    #[test]
    fn identity_scale_then_rgb_equals_direct() {
        let spec = crate::spec::ColorSpec {
            matrix: MatrixCoefficients::Bt709,
            range: crate::spec::ColorRange::Limited,
        };
        for planar in [false, true] {
            let (_, img) = grad_yuv(96, 92, planar);
            for filter in [Filter::Bilinear, Filter::Box, Filter::Bicubic] {
                let mut expected = vec![0u8; img.width * img.height * 3];
                yuv_to_rgb(&img, spec, RgbChannels::Rgb24, &mut expected).unwrap();
                let scale = Scale::new(img.width as u32, img.height as u32, filter);
                let cfg = ImageConfig {
                    rgb: Some(RgbChannels::Rgb24),
                    scale: Some(scale),
                    affine: None,
                    spec,
                };
                match process(&img, &cfg, Kernel::Auto).unwrap() {
                    ProcessedFrame::Rgb(out) => assert_eq!(out.data, expected),
                    _ => panic!("expected rgb output"),
                }
            }
        }
    }

    #[test]
    fn scale_then_rgb_matches_two_stage() {
        let spec = crate::spec::ColorSpec::default();
        for planar in [false, true] {
            let (_, img) = grad_yuv(192, 108, planar);
            let layout = if planar { YuvLayout::Planar } else { YuvLayout::Semi };
            for (tw, th) in [(64usize, 36), (384, 216)] {
                for filter in [Filter::Bilinear, Filter::Box, Filter::Bicubic] {
                    let scale = Scale::new(tw as u32, th as u32, filter);
                    // Manual two-stage: resize YUV, then convert.
                    let mut mid = vec![0u8; out_size(layout, tw, th)];
                    resize_yuv(&img, scale, Kernel::Auto, &mut mid).unwrap();
                    let view = tight_view(&mid, layout, tw, th);
                    let mut expected = vec![0u8; tw * th * 3];
                    yuv_to_rgb(&view, spec, RgbChannels::Rgb24, &mut expected).unwrap();

                    let cfg = ImageConfig {
                        rgb: Some(RgbChannels::Rgb24),
                        scale: Some(scale),
                        affine: None,
                        spec,
                    };
                    match process(&img, &cfg, Kernel::Auto).unwrap() {
                        ProcessedFrame::Rgb(out) => assert_eq!(out.data, expected),
                        _ => panic!("expected rgb output"),
                    }
                }
            }
        }
    }

    #[test]
    fn p010_scale_downcasts_first() {
        let (w, h) = (96usize, 92usize);
        let (_, img) = grad_p010(w, h);
        for (tw, th) in [(300usize, 200usize), (48, 46)] {
            let scale = Scale::new(tw as u32, th as u32, Filter::Bilinear);
            // Manual: down-cast -> resize.
            let mut i420 = vec![0u8; i420_size(w, h)];
            yuv_high_to_i420(&img, &mut i420).unwrap();
            let mid = scratch_view(&i420, w, h);
            let mut expected = vec![0u8; i420_size(tw, th)];
            resize_yuv(&mid, scale, Kernel::Auto, &mut expected).unwrap();

            let cfg = ImageConfig {
                rgb: None,
                scale: Some(scale),
                affine: None,
                spec: crate::spec::ColorSpec::default(),
            };
            match process(&img, &cfg, Kernel::Auto).unwrap() {
                ProcessedFrame::Yuv(out) => {
                    assert_eq!((out.width, out.height), (tw as u32, th as u32));
                    assert_eq!(out.layout, YuvLayout::Planar);
                    assert_eq!(out.data, expected);
                }
                _ => panic!("expected yuv output"),
            }
        }
    }

    #[test]
    fn p010_scale_then_rgb() {
        let (w, h) = (96usize, 92usize);
        let (_, img) = grad_p010(w, h);
        let spec = crate::spec::ColorSpec::default();
        for (tw, th) in [(48usize, 46), (192, 184)] {
            let scale = Scale::new(tw as u32, th as u32, Filter::Bicubic);
            // Manual: down-cast -> resize -> convert.
            let mut i420 = vec![0u8; i420_size(w, h)];
            yuv_high_to_i420(&img, &mut i420).unwrap();
            let mid = scratch_view(&i420, w, h);
            let mut resized = vec![0u8; i420_size(tw, th)];
            resize_yuv(&mid, scale, Kernel::Auto, &mut resized).unwrap();
            let view = scratch_view(&resized, tw, th);
            let mut expected = vec![0u8; tw * th * 4];
            yuv_to_rgb(&view, spec, RgbChannels::Rgba32, &mut expected).unwrap();

            let cfg = ImageConfig {
                rgb: Some(RgbChannels::Rgba32),
                scale: Some(scale),
                affine: None,
                spec,
            };
            match process(&img, &cfg, Kernel::Auto).unwrap() {
                ProcessedFrame::Rgb(out) => {
                    assert_eq!((out.width, out.height), (tw as u32, th as u32));
                    assert_eq!(out.channels, 4);
                    assert_eq!(out.data, expected);
                }
                _ => panic!("expected rgb output"),
            }
        }
    }
}

//! Post-decode image pipeline (downcast -> scale -> rgb) applied to frames
//! from any backend.
//!
//! [`apply`] maps the frame's [`PixelData`] to a [`YuvImage`], runs the
//! configured [`ImageConfig`] through NPP on NVIDIA hosts or Vulkan compute
//! on any other GPU (falling back to the SIMD software pipeline otherwise),
//! and writes the
//! result back into the frame:
//!
//! - scale only -> `pixel_data` is replaced with the scaled 8-bit 4:2:0
//!   buffer (tight I420 or NV12) and `width`/`height` are updated.
//! - rgb (with or without scale) -> `rgb_pixels` holds the packed RGB(R)(A)
//!   output, `pixel_data` is dropped, and `width`/`height` describe the
//!   (possibly scaled) RGB output.
//!
//! Only 4:2:0 Y'CbCr sources are transformable; any other format is passed
//! through untouched ([`ApplyOutcome::Skipped`]).

use vacc_core::frame::{DecodedFrame, PixelData, PixelPlane, RgbFrame};
use vacc_core::gpu::{GpuDevice, GpuFrame};
use vacc_image::{ImageConfig, Kernel, ProcessedFrame, YuvImage, YuvLayout};

use crate::error::{UnifiedError, UnifiedResult};

/// Whether the frame was transformed or passed through.
#[derive(Debug)]
pub(crate) enum ApplyOutcome {
    /// The frame was transformed in place.
    Transformed,
    /// The frame was passed through untouched; the reason is for logging.
    Skipped(String),
}

/// Apply `cfg` to `frame`. A no-op config or an untransformable format yields
/// [`ApplyOutcome::Skipped`] rather than an error; genuine pipeline failures
/// (bad scale target, allocation, NPP+software both failing) yield `Err`.
pub(crate) fn apply(frame: &mut DecodedFrame, cfg: &ImageConfig) -> UnifiedResult<ApplyOutcome> {
    if cfg.is_noop() {
        return Ok(ApplyOutcome::Skipped("no-op image config".into()));
    }
    // GPU decode track: run the zero-copy NPP pipeline on the device frame.
    // An unsupported combination (e.g. 10-bit sources) or a missing NPP falls
    // back to a readback + the host pipeline below.
    if let Some(src) = frame.gpu.as_ref() {
        match gpu_pipeline(src, cfg) {
            Some(out) => {
                write_back_gpu(frame, out);
                return Ok(ApplyOutcome::Transformed);
            }
            None => readback_into_pixel_data(frame)?,
        }
    }
    let Some(pd) = frame.pixel_data.as_ref() else {
        return Ok(ApplyOutcome::Skipped("frame carries no pixel data".into()));
    };

    // The source mapping borrows `pd`; the pipeline output is owned, so the
    // borrow ends before the write-back below. Unsupported formats are a
    // pass-through (the stream may legitimately not be 4:2:0 Y'CbCr).
    let processed = match map_source(pd) {
        Ok(src) => run_pipeline(&src.image(), cfg)
            .map_err(|e| UnifiedError::ImageProcessing { message: e.to_string() })?,
        Err(vacc_image::ImageError::Unsupported(reason)) => return Ok(ApplyOutcome::Skipped(reason)),
        Err(e) => return Err(UnifiedError::ImageProcessing { message: e.to_string() }),
    };

    write_back(frame, processed, cfg);
    Ok(ApplyOutcome::Transformed)
}

/// Zero-copy GPU pipeline over a device-resident frame, dispatched on the
/// owning device (NPP for CUDA frames, Vulkan compute otherwise). Returns the
/// transformed device frame, or `None` when the engine is unavailable or the
/// source/config combination is unsupported (the caller then reads back and
/// uses the host pipeline).
fn gpu_pipeline(src: &GpuFrame, cfg: &ImageConfig) -> Option<GpuFrame> {
    let result = match src.device {
        GpuDevice::Cuda { .. } => {
            if !vacc_npp::Npp::is_available() {
                log::warn!("cuda gpu frame but NPP unavailable; falling back to readback + host pipeline");
                return None;
            }
            vacc_npp::process_gpu(src, cfg)
        }
        GpuDevice::Vulkan { .. } => vacc_vkimage::process_device(src, cfg),
    };
    match result {
        Ok(out) => Some(out),
        Err(e) => {
            log::warn!("gpu image pipeline failed ({e}); falling back to readback + host pipeline");
            None
        }
    }
}

/// Move a GPU frame's contents to the host as tight NV12 `pixel_data` and
/// drop the device handle, so the host pipeline can process it. Only used for
/// the 8-bit NV12 frames the NVDEC gpu mode produces.
fn readback_into_pixel_data(frame: &mut DecodedFrame) -> UnifiedResult<()> {
    let src = frame.gpu.take().expect("gpu checked by caller");
    let buf = match src.device {
        GpuDevice::Cuda { .. } => vacc_npp::readback(&src),
        GpuDevice::Vulkan { .. } => vacc_vkimage::readback(&src),
    }
    .map_err(|e| UnifiedError::ImageProcessing { message: e.to_string() })?;
    let (w, h) = (src.width as usize, src.height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    frame.pixel_data = Some(PixelData {
        format: "NV12".to_string(),
        y: PixelPlane { data: buf.as_ptr(), pitch: w, width: w, height: h },
        u: PixelPlane {
            data: unsafe { buf.as_ptr().add(w * h) },
            pitch: cw * 2,
            width: cw,
            height: ch,
        },
        v: None,
        buffer: buf,
    });
    Ok(())
}

/// Install the GPU pipeline output into the frame; the result stays
/// device-resident and any host copies are dropped.
fn write_back_gpu(frame: &mut DecodedFrame, out: GpuFrame) {
    let (w, h) = (out.width, out.height);
    frame.gpu = Some(out);
    frame.pixel_data = None;
    frame.rgb_pixels = None;
    frame.width = w;
    frame.height = h;
}

/// A 4:2:0 source for the pipeline: either a direct view over the decoded
/// planes or an owned top-justified copy of a bottom-justified u16 source.
pub(crate) enum MappedSource<'a> {
    Direct(YuvImage<'a>),
    /// Top-justified (v << 6) tight copy; all backends store u16 LE.
    Normalized { buf: Vec<u8>, width: usize, height: usize },
}

impl MappedSource<'_> {
    pub(crate) fn image(&self) -> YuvImage<'_> {
        match self {
            Self::Direct(img) => *img,
            Self::Normalized { buf, width, height } => planar_u16_view(buf, *width, *height),
        }
    }
}

/// Map a decoded frame's planes to a pipeline source.
pub(crate) fn map_source(pd: &PixelData) -> Result<MappedSource<'_>, vacc_image::ImageError> {
    use vacc_image::ImageError;

    let (w, h) = (pd.y.width, pd.y.height);
    if w == 0 || h == 0 {
        return Err(ImageError::InvalidDimensions("zero-sized frame".into()));
    }
    match pd.format.as_str() {
        // Planar 8-bit (Vulkan, NVDEC, software, VAAPI).
        "I420" => Ok(MappedSource::Direct(planar_view(pd, 8)?)),
        // Semi-planar 8-bit: `u` holds interleaved CbCr, `v` is absent.
        "NV12" => {
            if pd.v.is_some() {
                return Err(ImageError::Unsupported(
                    "NV12 frame must not carry a separate V plane".into(),
                ));
            }
            let y = plane_slice(&pd.y);
            let uv = plane_slice(&pd.u);
            Ok(MappedSource::Direct(YuvImage::semi(y, pd.y.pitch, uv, pd.u.pitch, w, h, 8)))
        }
        // Planar u16, top-justified: Vulkan P010, NVDEC I420_16BIT (P016).
        "P010" | "I420_16BIT" => Ok(MappedSource::Direct(planar_view(pd, 10)?)),
        // Planar u16, bottom-justified (raw 10-bit in low bits): software
        // I420-10, VAAPI Y410P16. The pipeline expects top-justified values,
        // so copy and shift.
        "I420-10" | "Y410P16" => {
            let buf = normalize_u16_to_top(pd);
            Ok(MappedSource::Normalized { buf, width: w, height: h })
        }
        other => Err(ImageError::Unsupported(format!(
            "format '{other}' is not a supported 4:2:0 Y'CbCr source"
        ))),
    }
}

fn planar_view<'a>(pd: &'a PixelData, bps: u8) -> Result<YuvImage<'a>, vacc_image::ImageError> {
    use vacc_image::ImageError;

    let v = pd
        .v
        .as_ref()
        .ok_or_else(|| ImageError::Unsupported(format!("format '{}' must carry a V plane", pd.format)))?;
    Ok(YuvImage::planar(
        plane_slice(&pd.y),
        pd.y.pitch,
        plane_slice(&pd.u),
        pd.u.pitch,
        plane_slice(v),
        v.pitch,
        pd.y.width,
        pd.y.height,
        bps,
    ))
}

/// Copy each u16 plane (respecting pitch) and shift values to top-justified
/// (`v << 6`); output rows are tight.
fn normalize_u16_to_top(pd: &PixelData) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        pd.y.height * pd.y.pitch + pd.u.height * pd.u.pitch
            + pd.v.as_ref().map_or(0, |v| v.height * v.pitch),
    );
    for p in [Some(&pd.y), Some(&pd.u), pd.v.as_ref()] {
        if let Some(p) = p {
            for row in 0..p.height {
                let r = unsafe { std::slice::from_raw_parts(p.data.add(row * p.pitch), p.width * 2) };
                for i in (0..r.len()).step_by(2) {
                    let v = u16::from_le_bytes([r[i], r[i + 1]]);
                    out.extend_from_slice(&(v << 6).to_le_bytes());
                }
            }
        }
    }
    out
}

/// Tight planar u16 view over a normalized buffer (y, then u, then v).
fn planar_u16_view(buf: &[u8], w: usize, h: usize) -> YuvImage<'_> {
    let (cw, ch) = ((w + 1) / 2, (h + 1) / 2);
    let y_len = w * h * 2;
    let u_len = cw * ch * 2;
    let (y, rest) = buf.split_at(y_len);
    let (u, v) = rest.split_at(u_len);
    YuvImage::planar(y, w * 2, u, cw * 2, v, cw * 2, w, h, 10)
}

/// Run the pipeline: NPP on NVIDIA hosts, Vulkan compute otherwise (any
/// GPU), software as the last resort.
fn run_pipeline(img: &YuvImage, cfg: &ImageConfig) -> vacc_image::ImageResult<ProcessedFrame> {
    if vacc_npp::Npp::is_available() {
        match vacc_npp::process(img, cfg) {
            Ok(p) => return Ok(p),
            Err(e) => log::warn!("npp pipeline failed ({e}); using fallback"),
        }
    } else if vacc_vkimage::is_available() {
        match vacc_vkimage::process(img, cfg) {
            Ok(p) => return Ok(p),
            Err(e) => log::warn!("vulkan image pipeline failed ({e}); using software fallback"),
        }
    }
    vacc_image::process(img, cfg, Kernel::Auto)
}

/// Replace the frame's pixel data with the pipeline output. When only RGB
/// conversion is requested (no scaling), keep the original YUV alongside it.
fn write_back(frame: &mut DecodedFrame, out: ProcessedFrame, img_cfg: &ImageConfig) {
    match out {
        ProcessedFrame::Yuv(y) => {
            let (w, h) = (y.width as usize, y.height as usize);
            let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
            let buffer = y.data;
            frame.pixel_data = match y.layout {
                YuvLayout::Planar => {
                    let u_off = w * h;
                    let v_off = u_off + cw * ch;
                    Some(PixelData {
                        format: "I420".to_string(),
                        y: PixelPlane { data: buffer.as_ptr(), pitch: w, width: w, height: h },
                        u: PixelPlane {
                            data: unsafe { buffer.as_ptr().add(u_off) },
                            pitch: cw,
                            width: cw,
                            height: ch,
                        },
                        v: Some(PixelPlane {
                            data: unsafe { buffer.as_ptr().add(v_off) },
                            pitch: cw,
                            width: cw,
                            height: ch,
                        }),
                        buffer,
                    })
                }
                YuvLayout::Semi => {
                    let uv_off = w * h;
                    Some(PixelData {
                        format: "NV12".to_string(),
                        y: PixelPlane { data: buffer.as_ptr(), pitch: w, width: w, height: h },
                        u: PixelPlane {
                            data: unsafe { buffer.as_ptr().add(uv_off) },
                            pitch: cw * 2,
                            width: cw,
                            height: ch,
                        },
                        v: None,
                        buffer,
                    })
                }
            };
            frame.rgb_pixels = None;
            frame.width = y.width;
            frame.height = y.height;
        }
        ProcessedFrame::Rgb(r) => {
            // Only drop the original YUV if scaling or warping was involved.
            // For a plain rgb conversion the caller may want both representations.
            if img_cfg.scale.is_some() || img_cfg.affine.is_some() {
                frame.pixel_data = None;
            }
            frame.rgb_pixels = Some(RgbFrame {
                data: r.data,
                width: r.width,
                height: r.height,
                channels: r.channels,
            });
            frame.width = r.width;
            frame.height = r.height;
        }
    }
}

/// View over one plane's storage: `height * pitch` bytes starting at `data`.
/// All backend producers point the planes into `PixelData::buffer`.
fn plane_slice(p: &PixelPlane) -> &[u8] {
    unsafe { std::slice::from_raw_parts(p.data, p.height * p.pitch) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DecoderConfig;
    use vacc_image::{Affine, Interpolation, RgbChannels, Scale, Warp};

    /// Tight planar I420 frame with a horizontal Y gradient and neutral
    /// chroma, in one backing buffer (the backend convention).
    fn i420_frame(w: usize, h: usize) -> DecodedFrame {
        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let mut buf = vec![0u8; w * h + 2 * cw * ch];
        for r in 0..h {
            for c in 0..w {
                buf[r * w + c] = (c * 255 / w.max(1)) as u8;
            }
        }
        // Neutral chroma for limited range is 128.
        let uv_off = w * h;
        for i in uv_off..uv_off + 2 * cw * ch {
            buf[i] = 128;
        }
        let mut frame = DecodedFrame::new(0, 0, w as u32, h as u32, false);
        frame.pixel_data = Some(PixelData {
            format: "I420".to_string(),
            y: PixelPlane { data: buf.as_ptr(), pitch: w, width: w, height: h },
            u: PixelPlane { data: unsafe { buf.as_ptr().add(uv_off) }, pitch: cw, width: cw, height: ch },
            v: Some(PixelPlane {
                data: unsafe { buf.as_ptr().add(uv_off + cw * ch) },
                pitch: cw,
                width: cw,
                height: ch,
            }),
            buffer: buf,
        });
        frame
    }

    fn config(scale: Option<Scale>, rgb: Option<RgbChannels>) -> DecoderConfig {
        let image = ImageConfig { scale, rgb, ..Default::default() };
        DecoderConfig::new([crate::backend::Backend::Software]).with_image(image)
    }

    #[test]
    fn scale_only_replaces_pixel_data() {
        let mut frame = i420_frame(64, 32);
        let cfg = config(Some(Scale::new(32, 16, Interpolation::Bilinear)), None);
        match apply(&mut frame, &cfg.image()) {
            Ok(ApplyOutcome::Transformed) => {}
            other => panic!("expected transform, got {other:?}"),
        }
        let pd = frame.pixel_data.as_ref().unwrap();
        assert_eq!(pd.format, "I420");
        assert_eq!((frame.width, frame.height), (32, 16));
        assert_eq!(pd.buffer.len(), 32 * 16 + 2 * 16 * 8);
        // Gradient survives the resize: left edge dark, right edge bright.
        assert!(pd.buffer[0] < 40);
        assert!(pd.buffer[31] > 200);
    }

    #[test]
    fn rgb_only_keeps_yuv_and_adds_rgb() {
        let mut frame = i420_frame(64, 32);
        let cfg = config(None, Some(RgbChannels::Rgb24));
        match apply(&mut frame, &cfg.image()) {
            Ok(ApplyOutcome::Transformed) => {}
            other => panic!("expected transform, got {other:?}"),
        }
        let rgb = frame.rgb_pixels.as_ref().unwrap();
        assert_eq!((rgb.width, rgb.height, rgb.channels), (64, 32, 3));
        assert_eq!(rgb.data.len(), 64 * 32 * 3);
        assert!(frame.pixel_data.is_some());
        // Red gradient: first pixel's R channel is low, last row's is high.
        assert!(rgb.data[0] < 80);
        assert!(rgb.data[(64 * 32 - 1) as usize * 3] > 150);
    }

    #[test]
    fn scale_and_rgb_yields_only_rgb() {
        let mut frame = i420_frame(64, 32);
        let cfg = config(Some(Scale::new(32, 16, Interpolation::Box)), Some(RgbChannels::Rgba32));
        match apply(&mut frame, &cfg.image()) {
            Ok(ApplyOutcome::Transformed) => {}
            other => panic!("expected transform, got {other:?}"),
        }
        assert!(frame.pixel_data.is_none());
        let rgb = frame.rgb_pixels.as_ref().unwrap();
        assert_eq!((rgb.width, rgb.height, rgb.channels), (32, 16, 4));
        assert_eq!((frame.width, frame.height), (32, 16));
    }

    #[test]
    fn unsupported_format_is_skipped() {
        let mut frame = i420_frame(8, 8);
        frame.pixel_data.as_mut().unwrap().format = "GRAY".to_string();
        let cfg = config(None, Some(RgbChannels::Rgb24));
        match apply(&mut frame, &cfg.image()) {
            Ok(ApplyOutcome::Skipped(reason)) => assert!(reason.contains("GRAY")),
            other => panic!("expected skip, got {other:?}"),
        }
        // Untouched.
        assert!(frame.rgb_pixels.is_none());
    }

    #[test]
    fn bottom_justified_u16_is_normalized() {
        // 8x8 "I420-10": Y = 100 (raw 10-bit in the low bits), Cb/Cr = 512.
        let (w, h) = (8usize, 8usize);
        let (cw, ch) = (4usize, 4usize);
        let mut buf = vec![0u16 as u8; (w * h + 2 * cw * ch) * 2];
        for i in 0..w * h {
            buf[i * 2..i * 2 + 2].copy_from_slice(&100u16.to_le_bytes());
        }
        for i in w * h..w * h + 2 * cw * ch {
            buf[i * 2..i * 2 + 2].copy_from_slice(&512u16.to_le_bytes());
        }
        let mut frame = DecodedFrame::new(0, 0, w as u32, h as u32, false);
        frame.pixel_data = Some(PixelData {
            format: "I420-10".to_string(),
            y: PixelPlane { data: buf.as_ptr(), pitch: w * 2, width: w, height: h },
            u: PixelPlane { data: unsafe { buf.as_ptr().add(w * h * 2) }, pitch: cw * 2, width: cw, height: ch },
            v: Some(PixelPlane {
                data: unsafe { buf.as_ptr().add((w * h + cw * ch) * 2) },
                pitch: cw * 2,
                width: cw,
                height: ch,
            }),
            buffer: buf,
        });

        let cfg = config(Some(Scale::new(8, 8, Interpolation::Bilinear)), None);
        match apply(&mut frame, &cfg.image()) {
            Ok(ApplyOutcome::Transformed) => {}
            other => panic!("expected transform, got {other:?}"),
        }
        let pd = frame.pixel_data.as_ref().unwrap();
        assert_eq!(pd.format, "I420");
        // 10-bit Y=100 down-casts exactly to 8-bit 100; identity-scale
        // bilinear must not drift it.
        assert!((98..=102).contains(&pd.buffer[0]), "got {}", pd.buffer[0]);
    }

    #[test]
    fn nv12_scale_preserves_semi_planar() {
        // 16x8 NV12: Y gradient, neutral interleaved chroma.
        let (w, h) = (16usize, 8usize);
        let (cw, ch) = (8usize, 4usize);
        let mut buf = vec![0u8; w * h + 2 * cw * ch];
        for r in 0..h {
            for c in 0..w {
                buf[r * w + c] = (c * 255 / (w - 1)) as u8;
            }
        }
        let uv_off = w * h;
        for i in uv_off..uv_off + 2 * cw * ch {
            buf[i] = 128;
        }
        let mut frame = DecodedFrame::new(0, 0, w as u32, h as u32, false);
        frame.pixel_data = Some(PixelData {
            format: "NV12".to_string(),
            y: PixelPlane { data: buf.as_ptr(), pitch: w, width: w, height: h },
            u: PixelPlane { data: unsafe { buf.as_ptr().add(uv_off) }, pitch: cw * 2, width: cw, height: ch },
            v: None,
            buffer: buf,
        });

        let cfg = config(Some(Scale::new(8, 4, Interpolation::Bilinear)), None);
        match apply(&mut frame, &cfg.image()) {
            Ok(ApplyOutcome::Transformed) => {}
            other => panic!("expected transform, got {other:?}"),
        }
        let pd = frame.pixel_data.as_ref().unwrap();
        assert_eq!(pd.format, "NV12");
        assert!(pd.v.is_none());
        assert_eq!((frame.width, frame.height), (8, 4));
        assert_eq!(pd.buffer.len(), 8 * 4 + 2 * 4 * 2);
    }

    #[test]
    fn warp_yields_rgb_and_drops_yuv() {
        let mut frame = i420_frame(64, 32);
        let image = ImageConfig {
            rgb: Some(RgbChannels::Rgba32),
            affine: Some(Warp::bilinear(Affine::translate(4.0, 2.0))),
            ..Default::default()
        };
        let cfg = DecoderConfig::new([crate::backend::Backend::Software]).with_image(image);
        match apply(&mut frame, &cfg.image()) {
            Ok(ApplyOutcome::Transformed) => {}
            other => panic!("expected transform, got {other:?}"),
        }
        // Warping drops the original YUV; only the warped RGB remains.
        assert!(frame.pixel_data.is_none());
        let rgb = frame.rgb_pixels.as_ref().unwrap();
        assert_eq!((rgb.width, rgb.height, rgb.channels), (64, 32, 4));
        // Forward translate(4, 2): the top-left corner maps outside the
        // source and must come out black.
        assert!(rgb.data[0] == 0 && rgb.data[1] == 0 && rgb.data[2] == 0);
    }
}

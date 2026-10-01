//! The unified decoder: one streaming API over all backends, with fallback.
//!
//! The public surface is streaming-only:
//!
//! 1. Construct from an initial bitstream chunk (header + first access
//!    units) via [`VaccDecoder::new`] / [`VaccDecoder::new_auto`]. That chunk
//!    is consumed as the first input; it must be large enough for the
//!    selected backend to find its parameter sets (a few tens of KiB always
//!    suffices for Annex-B / IVF streams).
//! 2. Feed the remainder of the stream with [`Decoder::submit`] and pull
//!    frames with [`Decoder::decode`], interleaving as data arrives.
//! 3. At end of stream, drain the reordering buffer with [`Decoder::flush`].
//!
//! No API takes or returns the whole stream: input is owned by the caller
//! and output is pulled one frame at a time.

use std::collections::VecDeque;

use vacc_core::codec::VideoCodec;
use vacc_core::decoder::{Decoder, DecoderInfo};
use vacc_core::format::{ChromaSubsampling, ComponentBitDepth, VideoFormat};
use vacc_core::frame::{DecodedFrame, PixelData, PixelPlane};
use vacc_core::session::Extent2D;

use vacc_image::ImageConfig;

use crate::backend::Backend;
use crate::codec::detect_codec;
use crate::config::DecoderConfig;
use crate::error::{BackendFailure, UnifiedError, UnifiedResult};
use crate::transform::ApplyOutcome;

fn be(e: impl std::error::Error + Send + Sync + 'static) -> UnifiedError {
    UnifiedError::Backend { source: Box::new(e) }
}

/// Type-erased backend decoder.
pub trait AnyDecode {
    fn info(&self) -> DecoderInfo;
    fn submit(&mut self, data: &[u8]) -> UnifiedResult<()>;
    fn decode(&mut self) -> UnifiedResult<Option<DecodedFrame>>;
    fn flush(&mut self) -> UnifiedResult<Vec<DecodedFrame>>;
    fn reset(&mut self) -> UnifiedResult<()>;
}

/// Blanket delegation for types that implement [`vacc_core::Decoder`].
macro_rules! impl_any_decode {
    ($ty:ty) => {
        impl AnyDecode for $ty {
            fn info(&self) -> DecoderInfo {
                <Self as Decoder>::info(self)
            }
            fn submit(&mut self, data: &[u8]) -> UnifiedResult<()> {
                <Self as Decoder>::submit(self, data).map_err(|e| UnifiedError::Backend { source: Box::new(e) })
            }
            fn decode(&mut self) -> UnifiedResult<Option<DecodedFrame>> {
                <Self as Decoder>::decode(self).map_err(|e| UnifiedError::Backend { source: Box::new(e) })
            }
            fn flush(&mut self) -> UnifiedResult<Vec<DecodedFrame>> {
                <Self as Decoder>::flush(self).map_err(|e| UnifiedError::Backend { source: Box::new(e) })
            }
            fn reset(&mut self) -> UnifiedResult<()> {
                <Self as Decoder>::reset(self).map_err(|e| UnifiedError::Backend { source: Box::new(e) })
            }
        }
    };
}

#[cfg(feature = "nvdec")]
impl_any_decode!(vacc_nvdec_decode::NvdecH264Decoder);
#[cfg(feature = "nvdec")]
impl_any_decode!(vacc_nvdec_decode::NvdecH265Decoder);
#[cfg(feature = "nvdec")]
impl_any_decode!(vacc_nvdec_decode::NvdecVp9Decoder);
#[cfg(feature = "nvdec")]
impl_any_decode!(vacc_nvdec_decode::NvdecAv1Decoder);
#[cfg(feature = "vaapi")]
impl_any_decode!(vacc_vaapi_decode::VaapiDecoder);
#[cfg(feature = "sw")]
impl_any_decode!(vacc_software_decode::SwH264Decoder);
#[cfg(feature = "sw")]
impl_any_decode!(vacc_software_decode::SoftwareH265Decoder);

/// Adapter around the Vulkan backend, which decodes the whole stream at once
/// (its inner decoder has no incremental submit/decode API). Submitted data is
/// buffered; on the first `decode()` call the accumulated stream is decoded in
/// one pass and frames are served one by one.
///
/// Memory note: unlike the streaming backends, this backend holds the entire
/// input stream (buffered here, plus a GPU bitstream copy) and every decoded
/// frame until it is pulled. For long-running streams prefer `nvdec`,
/// `vaapi` or `software` in the backend order.
#[cfg(feature = "vulkan")]
struct VulkanAdapter {
    data: Vec<u8>,
    queue: VecDeque<vacc_vulkan_decode::DecodedFrame>,
    next_index: u32,
    started: bool,
    gpu: bool,
    /// GPU mode only: kept alive because queued frames reference the
    /// decoder's instance/device memory (their `GpuFrame` handles are only
    /// valid while the owning `VulkanDecoder` lives).
    decoder: Option<vacc_vulkan_decode::VulkanDecoder>,
}

#[cfg(feature = "vulkan")]
impl VulkanAdapter {
    fn new(data: Vec<u8>, gpu: bool) -> UnifiedResult<Self> {
        // Validate eagerly so a bad stream falls through to the next backend.
        if gpu {
            vacc_vulkan_decode::VulkanDecoder::new_gpu(data.clone()).map_err(be)?;
        } else {
            vacc_vulkan_decode::VulkanDecoder::new(data.clone()).map_err(be)?;
        }
        Ok(Self {
            data,
            queue: VecDeque::new(),
            next_index: 0,
            started: false,
            gpu,
            decoder: None,
        })
    }

    fn start(&mut self) -> UnifiedResult<()> {
        if self.started {
            return Ok(());
        }
        self.started = true;
        let mut decoder = if self.gpu {
            vacc_vulkan_decode::VulkanDecoder::new_gpu(std::mem::take(&mut self.data))
        } else {
            vacc_vulkan_decode::VulkanDecoder::new(std::mem::take(&mut self.data))
        }
        .map_err(be)?;
        let frames = decoder.decode_all(usize::MAX).map_err(be)?;
        if self.gpu {
            // The frames' device buffers live on this decoder's device; keep
            // it alive until the frames (and its clones) are dropped.
            self.decoder = Some(decoder);
        }
        self.queue.extend(frames);
        Ok(())
    }

    fn pop(&mut self) -> Option<DecodedFrame> {
        let frame = self.queue.pop_front()?;
        let index = self.next_index;
        self.next_index += 1;
        Some(to_core_frame(frame, index))
    }
}

#[cfg(feature = "vulkan")]
impl Drop for VulkanAdapter {
    fn drop(&mut self) {
        // Forget vkimage compute state before the decoder's device is
        // destroyed: the registry keys on raw pointers, which a later
        // device may reuse.
        if let Some(dec) = &self.decoder {
            vacc_vkimage::forget_device(dec.device());
        }
    }
}

/// Convert a Vulkan backend frame (coded-size planes) into the core frame
/// type with cropped, display-size planes — the same convention the other
/// backends use. All three planes are packed into a single allocation: one
/// `Vec`, one copy per pixel, no per-plane temporaries.
#[cfg(feature = "vulkan")]
fn to_core_frame(vk_frame: vacc_vulkan_decode::DecodedFrame, index: u32) -> DecodedFrame {
    // GPU track: device-resident frame, no host pixels (mirrors the NVDEC
    // gpu-mode frames).
    if let Some(gpu) = vk_frame.gpu {
        let mut frame =
            DecodedFrame::new(index, 0, vk_frame.display_width, vk_frame.display_height, false);
        frame.poc = vk_frame.poc;
        frame.gpu = Some(gpu);
        return frame;
    }
    let pixels = vk_frame.pixels;
    let ss = pixels.sample_size.max(1) as usize;
    let (dw, dh) = (vk_frame.display_width as usize, vk_frame.display_height as usize);

    // Chroma crop is in chroma-sample space (half-pel for 4:2:0).
    let half = pixels.chroma_width * 2 == vk_frame.coded_width;
    let (cl_c, ct_c) = if half {
        (vk_frame.crop_left / 2, vk_frame.crop_top / 2)
    } else {
        (vk_frame.crop_left, vk_frame.crop_top)
    };

    // Exact output size: luma rows are cropped to the display width; chroma
    // rows exist while the source row exists and run from the crop offset to
    // the end of the source row.
    let y_row_bytes = (vk_frame.coded_width as usize)
        .saturating_sub(vk_frame.crop_left as usize)
        .min(dw)
        * ss;
    let ch_rows = (pixels.chroma_height as usize).saturating_sub(ct_c as usize).min(dh);
    let ch_row_bytes =
        (pixels.chroma_width as usize).saturating_sub(cl_c as usize).min(dw) * ss;

    let mut buffer = Vec::with_capacity(y_row_bytes * dh + ch_row_bytes * ch_rows * 2);

    // Copy the cropped rows of `plane` into `buffer`; rows beyond the plane
    // contribute nothing (same per-row clamp as before).
    let crop_into = |buffer: &mut Vec<u8>, plane: &[u8], row_stride: usize, left: u32, top: u32| {
        let out_row = dw * ss;
        for r in 0..dh {
            let src = plane.chunks_exact(row_stride).nth(top as usize + r).unwrap_or(&[]);
            let start = (left as usize) * ss;
            buffer.extend_from_slice(&src[start..start.saturating_add(out_row).min(src.len())]);
        }
    };

    crop_into(
        &mut buffer,
        &pixels.y_plane,
        vk_frame.coded_width as usize * ss,
        vk_frame.crop_left,
        vk_frame.crop_top,
    );
    let u_off = buffer.len();
    let chroma_stride = pixels.chroma_width as usize * ss;
    crop_into(&mut buffer, &pixels.u_plane, chroma_stride, cl_c, ct_c);
    let v_off = buffer.len();
    crop_into(&mut buffer, &pixels.v_plane, chroma_stride, cl_c, ct_c);

    let (cw, ch) = (pixels.chroma_width as usize, pixels.chroma_height as usize);
    let frame = DecodedFrame::new(index, 0, dw as u32, dh as u32, false);
    let mut frame = frame;
    frame.poc = vk_frame.poc;
    frame.pixel_data = Some(PixelData {
        format: if ss == 2 { "P010".to_string() } else { "I420".to_string() },
        y: PixelPlane { data: buffer.as_ptr(), pitch: dw * ss, width: dw, height: dh },
        u: PixelPlane {
            data: unsafe { buffer.as_ptr().add(u_off) },
            pitch: cw * ss,
            width: cw,
            height: ch,
        },
        v: Some(PixelPlane {
            data: unsafe { buffer.as_ptr().add(v_off) },
            pitch: cw * ss,
            width: cw,
            height: ch,
        }),
        buffer,
    });
    frame
}

#[cfg(feature = "vulkan")]
impl AnyDecode for VulkanAdapter {
    fn info(&self) -> DecoderInfo {
        DecoderInfo {
            backend: "vulkan".to_string(),
            codec: VideoCodec::None,
            coded_size: Extent2D::new(0, 0),
            display_size: Extent2D::new(0, 0),
            chroma_subsampling: ChromaSubsampling::_420,
            luma_bit_depth: ComponentBitDepth::Bit8,
            chroma_bit_depth: ComponentBitDepth::Bit8,
            profile_idc: None,
            dpb_slots: 0,
        }
    }

    fn submit(&mut self, data: &[u8]) -> UnifiedResult<()> {
        if self.started {
            return Err(UnifiedError::Unsupported {
                message: "vulkan backend decodes the whole stream on the first decode() call; submit all data before decoding".to_string(),
            });
        }
        self.data.extend_from_slice(data);
        Ok(())
    }

    fn decode(&mut self) -> UnifiedResult<Option<DecodedFrame>> {
        self.start()?;
        Ok(self.pop())
    }

    fn flush(&mut self) -> UnifiedResult<Vec<DecodedFrame>> {
        self.start()?;
        let mut frames = Vec::new();
        while let Some(frame) = self.pop() {
            frames.push(frame);
        }
        Ok(frames)
    }

    fn reset(&mut self) -> UnifiedResult<()> {
        Ok(())
    }
}

/// A video decoder that tries a configured list of backends in order and uses
/// the first one that can initialize the stream.
///
/// The API is streaming-only: construct from an initial chunk of the
/// bitstream, feed the rest with [`Decoder::submit`], and pull frames one at
/// a time with [`Decoder::decode`].
///
/// ```no_run
/// use std::io::Read;
/// use vacc_core::decoder::Decoder;
/// use vacc::{Backend, DecoderConfig, VaccDecoder};
///
/// // Seed the decoder with the stream head (parameter sets + first access
/// // units). The chunk is consumed as the first input; submit the rest.
/// let mut file = std::fs::File::open("video.h264").unwrap();
/// let mut probe = [0u8; 64 * 1024];
/// let n = file.read(&mut probe).unwrap();
///
/// // One-liner: default chain vulkan -> nvdec -> vaapi -> software.
/// let mut decoder = VaccDecoder::new_auto(&probe[..n]).unwrap();
/// println!("using backend: {}", decoder.backend());
///
/// // Or pick the preferred order yourself:
/// // let config = DecoderConfig::new([Backend::Nvdec, Backend::Software]);
/// // let mut decoder = VaccDecoder::new(&probe[..n], &config).unwrap();
///
/// let mut buf = vec![0u8; 1024 * 1024];
/// loop {
///     let n = file.read(&mut buf).unwrap();
///     if n == 0 {
///         break;
///     }
///     decoder.submit(&buf[..n]).unwrap();
///     while let Some(frame) = decoder.decode().unwrap() {
///         println!("frame {}: {}x{}", frame.frame_index, frame.width, frame.height);
///     }
/// }
/// for frame in decoder.flush().unwrap() {
///     println!("frame {}: {}x{}", frame.frame_index, frame.width, frame.height);
/// }
/// ```
pub struct VaccDecoder {
    backend: Backend,
    inner: Box<dyn AnyDecode>,
    /// Next synthetic PTS (microseconds) for frames without a usable
    /// timestamp.
    pts_next: i64,
    /// PTS of the previously emitted frame (monotonicity check).
    last_pts: Option<i64>,
    /// Post-decode image pipeline (no-op unless configured).
    image: ImageConfig,
    /// One-shot warning guard for skipped transforms.
    warned_transform: bool,
}

/// Synthetic PTS step for streams that carry no timestamps: 1/30 s in
/// microseconds — the same convention the sw and vaapi backends use.
const SYNTHETIC_PTS_STEP: i64 = 33_333;

impl VaccDecoder {
    /// Create a decoder, trying each backend in `config.order()` until one
    /// succeeds. If all fail, the error lists every per-backend failure.
    ///
    /// `probe` is the initial chunk of the bitstream (header + first access
    /// units). It is consumed as the first input: keep feeding the remainder
    /// of the stream with [`Decoder::submit`], starting right after the bytes
    /// passed here. The chunk must contain enough of the stream head for the
    /// selected backend to find its parameter sets; a few tens of KiB always
    /// suffices for Annex-B and IVF streams.
    pub fn new(probe: &[u8], config: &DecoderConfig) -> UnifiedResult<Self> {
        // GPU decode track: only Vulkan and NVDEC emit device-resident
        // frames, so restrict the configured order to them while preserving
        // the caller's explicit choice (e.g. `only(Nvdec)`).
        let order: Vec<Backend> = if config.gpu() {
            config
                .order()
                .iter()
                .copied()
                .filter(|b| matches!(b, Backend::Vulkan | Backend::Nvdec))
                .collect()
        } else {
            config.order().to_vec()
        };
        if order.is_empty() {
            return Err(UnifiedError::EmptyBackendOrder);
        }
        let mut failures = Vec::new();
        for &backend in &order {
            // A panicking backend (e.g. an out-of-bounds bug) must not take
            // down the caller: treat it like any other init failure and fall
            // through to the next backend.
            let created = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create(backend, probe, config.gpu())
            }));
            match created {
                Ok(Ok(inner)) => {
                    log::info!(
                        "unified decoder: using {} backend (config: {})",
                        backend,
                        config
                    );
                    return Ok(Self {
                        backend,
                        inner,
                        pts_next: 0,
                        last_pts: None,
                        image: config.image(),
                        warned_transform: false,
                    });
                }
                Ok(Err(e)) => {
                    log::warn!("unified decoder: {} backend failed: {}", backend, e);
                    failures.push(BackendFailure { backend, message: e.to_string() });
                }
                Err(_) => {
                    log::warn!("unified decoder: {} backend panicked during init; falling back", backend);
                    failures.push(BackendFailure {
                        backend,
                        message: "backend panicked during initialization".to_string(),
                    });
                }
            }
        }
        Err(UnifiedError::AllBackendsFailed { failures })
    }

    /// Create a decoder with the default fallback chain
    /// (`vulkan -> nvdec -> vaapi -> software`). See [`Self::new`] for the
    /// meaning of `probe`.
    pub fn new_auto(probe: &[u8]) -> UnifiedResult<Self> {
        Self::new(probe, &DecoderConfig::default())
    }

    /// The backend actually in use.
    pub fn backend(&self) -> Backend {
        self.backend
    }
}

fn create(backend: Backend, data: &[u8], gpu: bool) -> UnifiedResult<Box<dyn AnyDecode>> {
    match backend {
        #[cfg(feature = "vulkan")]
        Backend::Vulkan => Ok(Box::new(VulkanAdapter::new(data.to_vec(), gpu)?)),
        #[cfg(not(feature = "vulkan"))]
        Backend::Vulkan => Err(disabled("vulkan")),

        #[cfg(feature = "nvdec")]
        Backend::Nvdec => {
            let codec = detect_codec(data).ok_or(UnifiedError::CodecNotDetected)?;
            match codec {
                VideoCodec::DecodeH264 => {
                    let d = if gpu {
                        vacc_nvdec_decode::NvdecH264Decoder::new_gpu(data.to_vec())
                    } else {
                        vacc_nvdec_decode::NvdecH264Decoder::new(data.to_vec())
                    };
                    Ok(Box::new(d.map_err(be)?))
                }
                VideoCodec::DecodeH265 => {
                    let d = if gpu {
                        vacc_nvdec_decode::NvdecH265Decoder::new_gpu(data.to_vec())
                    } else {
                        vacc_nvdec_decode::NvdecH265Decoder::new(data.to_vec())
                    };
                    Ok(Box::new(d.map_err(be)?))
                }
                VideoCodec::DecodeVp9 => {
                    let d = if gpu {
                        vacc_nvdec_decode::NvdecVp9Decoder::new_gpu(data.to_vec())
                    } else {
                        vacc_nvdec_decode::NvdecVp9Decoder::new(data.to_vec())
                    };
                    Ok(Box::new(d.map_err(be)?))
                }
                VideoCodec::DecodeAv1 => {
                    let d = if gpu {
                        vacc_nvdec_decode::NvdecAv1Decoder::new_gpu(data.to_vec())
                    } else {
                        vacc_nvdec_decode::NvdecAv1Decoder::new(data.to_vec())
                    };
                    Ok(Box::new(d.map_err(be)?))
                }
                other => Err(UnifiedError::Unsupported {
                    message: format!("nvdec backend does not support {}", other.name()),
                }),
            }
        }
        #[cfg(not(feature = "nvdec"))]
        Backend::Nvdec => Err(disabled("nvdec")),

        #[cfg(feature = "vaapi")]
        Backend::Vaapi => Ok(Box::new(vacc_vaapi_decode::VaapiDecoder::new(data.to_vec()).map_err(be)?)),
        #[cfg(not(feature = "vaapi"))]
        Backend::Vaapi => Err(disabled("vaapi")),

        #[cfg(feature = "sw")]
        Backend::Software => {
            let codec = detect_codec(data).ok_or(UnifiedError::CodecNotDetected)?;
            match codec {
                VideoCodec::DecodeH264 => Ok(Box::new(
                    vacc_software_decode::SwH264Decoder::new(data.to_vec()).map_err(be)?,
                )),
                VideoCodec::DecodeH265 => Ok(Box::new(
                    vacc_software_decode::SoftwareH265Decoder::new(data.to_vec()).map_err(be)?,
                )),
                other => Err(UnifiedError::Unsupported {
                    message: format!("software backend does not support {}", other.name()),
                }),
            }
        }
        #[cfg(not(feature = "sw"))]
        Backend::Software => Err(disabled("software")),
    }
}

#[cfg(any(
    not(feature = "vulkan"),
    not(feature = "nvdec"),
    not(feature = "vaapi"),
    not(feature = "sw")
))]
fn disabled(name: &str) -> UnifiedError {
    UnifiedError::Unsupported {
        message: format!("{name} backend is not enabled in this build (enable the feature)"),
    }
}

impl std::fmt::Debug for VaccDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VaccDecoder").field("backend", &self.backend).finish()
    }
}

impl Decoder for VaccDecoder {
    type Error = UnifiedError;

    fn new(data: Vec<u8>) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        Self::new_auto(&data)
    }

    fn new_with_format(
        _data: Vec<u8>,
        _codec: VideoCodec,
        _format: &VideoFormat,
    ) -> Result<Self, Self::Error> {
        Err(UnifiedError::Unsupported {
            message: "use VaccDecoder::new(data, &config) with bitstream data".to_string(),
        })
    }

    fn info(&self) -> DecoderInfo {
        self.inner.info()
    }

    fn submit(&mut self, data: &[u8]) -> Result<(), Self::Error> {
        self.inner.submit(data)
    }

    fn decode(&mut self) -> Result<Option<DecodedFrame>, Self::Error> {
        let mut frame = self.inner.decode()?;
        if let Some(f) = &mut frame {
            self.normalize_pts(f);
            self.apply_image(f)?;
        }
        Ok(frame)
    }

    fn flush(&mut self) -> Result<Vec<DecodedFrame>, Self::Error> {
        let mut frames = self.inner.flush()?;
        for f in &mut frames {
            self.normalize_pts(f);
            self.apply_image(f)?;
        }
        Ok(frames)
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.pts_next = 0;
        self.last_pts = None;
        self.inner.reset()
    }
}

impl VaccDecoder {
    /// Run the configured image pipeline over a decoded frame. A no-op
    /// config is a free pass-through; an untransformable frame (e.g. a
    /// non-4:2:0 format) is passed through with a one-shot warning.
    fn apply_image(&mut self, frame: &mut DecodedFrame) -> UnifiedResult<()> {
        if self.image.is_noop() {
            return Ok(());
        }
        match crate::transform::apply(frame, &self.image) {
            Ok(ApplyOutcome::Transformed) => Ok(()),
            Ok(ApplyOutcome::Skipped(reason)) => {
                if !self.warned_transform {
                    log::warn!("image transform skipped: {reason}");
                    self.warned_transform = true;
                }
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Guarantee a presentation timestamp on every output frame.
    ///
    /// Backends that decode raw bitstreams (no container PTS) either leave
    /// `pts_valid` false or stamp timestamps in *decode* order, which is not
    /// monotonic once B-frames reorder the output. In both cases a synthetic
    /// monotonic PTS is assigned ([`SYNTHETIC_PTS_STEP`] per frame). A
    /// backend-provided PTS is kept only when it is valid and strictly
    /// greater than the previously emitted one.
    fn normalize_pts(&mut self, frame: &mut DecodedFrame) {
        let monotonic = self.last_pts.is_none_or(|last| frame.timestamp > last);
        if !frame.pts_valid || !monotonic {
            frame.timestamp = self.pts_next;
            frame.pts_valid = true;
        }
        self.last_pts = Some(frame.timestamp);
        // Keep the synthetic clock ahead of any real PTS provided.
        self.pts_next = self.pts_next.max(frame.timestamp + SYNTHETIC_PTS_STEP);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vacc_core::gpu::GpuPixelFormat;
    use vacc_image::{Interpolation, RgbChannels, Scale};

    fn sample(name: &str) -> Vec<u8> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/samples/");
        std::fs::read(format!("{path}{name}")).unwrap()
    }

    /// Drain everything: pull ready frames, then flush at end of stream.
    fn drain<D: Decoder>(d: &mut D) -> Vec<DecodedFrame> {
        let mut frames = drain_ready(d);
        frames.extend(d.flush().unwrap());
        frames
    }

    /// Pull frames that are ready *without* flushing. Flushing mid-stream is
    /// an end-of-stream operation: it releases reorder-held frames before
    /// their display-order successors have been decoded.
    fn drain_ready<D: Decoder>(d: &mut D) -> Vec<DecodedFrame> {
        let mut frames = Vec::new();
        while let Some(f) = Decoder::decode(d).unwrap() {
            frames.push(f);
        }
        frames
    }

    #[test]
    fn empty_config_rejected() {
        let data = sample("h264_main.h264");
        let err = VaccDecoder::new(&data, &DecoderConfig::new([])).unwrap_err();
        assert!(matches!(err, UnifiedError::EmptyBackendOrder));
    }

    #[test]
    fn software_only_fails_for_unsupported_codec() {
        // AV1 has no software backend; the error must list the failure.
        let data = sample("av1_main.ivf");
        let err = VaccDecoder::new(&data, &DecoderConfig::only(Backend::Software)).unwrap_err();
        match err {
            UnifiedError::AllBackendsFailed { failures } => {
                assert_eq!(failures.len(), 1);
                assert_eq!(failures[0].backend, Backend::Software);
                assert!(failures[0].message.contains("software backend"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn unified_matches_direct_sw_h264() {
        let data = sample("h264_main.h264");
        let mut unified =
            VaccDecoder::new(&data, &DecoderConfig::only(Backend::Software)).unwrap();
        assert_eq!(unified.backend(), Backend::Software);
        let a = drain(&mut unified);
        assert_pts(&a);

        let mut direct = vacc_software_decode::SwH264Decoder::new(data).unwrap();
        let b = drain(&mut direct);

        assert!(!a.is_empty());
        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.iter().zip(&b) {
            assert_eq!((fa.width, fa.height), (fb.width, fb.height));
            let pa = fa.pixel_data.as_ref().unwrap();
            let pb = fb.pixel_data.as_ref().unwrap();
            assert_eq!(pa.buffer, pb.buffer);
        }
    }

    /// Every frame from VaccDecoder must carry a valid, strictly increasing PTS.
    fn assert_pts(frames: &[DecodedFrame]) {
        let mut prev = -1i64;
        for f in frames {
            assert!(f.pts_valid, "frame {} has no valid pts", f.frame_index);
            assert!(f.timestamp > prev, "pts not increasing at frame {}", f.frame_index);
            prev = f.timestamp;
        }
    }

    /// The streaming contract: feeding the stream in arbitrary chunks (split
    /// mid-NAL) must produce exactly the same frames as feeding it whole.
    #[test]
    fn streaming_chunked_input_matches_whole_stream() {
        let data = sample("h264_main.h264");
        let config = DecoderConfig::only(Backend::Software);

        // Prime with a head chunk, then feed the rest in 70 KiB pieces,
        // pulling ready frames after every submit. Flush only at end of
        // stream (it releases reorder-held frames).
        let (head, rest) = data.split_at(70_001);
        let mut unified = VaccDecoder::new(head, &config).unwrap();
        let mut a: Vec<DecodedFrame> = drain_ready(&mut unified);
        for chunk in rest.chunks(70_001) {
            Decoder::submit(&mut unified, chunk).unwrap();
            a.extend(drain_ready(&mut unified));
        }
        a.extend(unified.flush().unwrap());
        assert_pts(&a);

        let mut direct = vacc_software_decode::SwH264Decoder::new(data).unwrap();
        let b = drain(&mut direct);

        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.iter().zip(&b) {
            assert_eq!((fa.width, fa.height), (fb.width, fb.height));
            let pa = fa.pixel_data.as_ref().unwrap();
            let pb = fb.pixel_data.as_ref().unwrap();
            assert_eq!(pa.buffer, pb.buffer);
        }
    }

    #[test]
    fn unified_matches_direct_sw_h265() {
        let data = sample("h265_main.h265");
        let mut unified =
            VaccDecoder::new(&data, &DecoderConfig::only(Backend::Software)).unwrap();
        assert_eq!(unified.backend(), Backend::Software);
        let a = drain(&mut unified);
        assert_pts(&a);

        let mut direct = vacc_software_decode::SoftwareH265Decoder::new(data).unwrap();
        let b = drain(&mut direct);

        assert!(!a.is_empty());
        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.iter().zip(&b) {
            assert_eq!((fa.width, fa.height), (fb.width, fb.height));
            let pa = fa.pixel_data.as_ref().unwrap();
            let pb = fb.pixel_data.as_ref().unwrap();
            assert_eq!(pa.buffer, pb.buffer);
        }
    }

    #[test]
    fn streaming_chunked_h265_input_matches_whole_stream() {
        // Same contract as the H.264 chunked test: a NAL split across two
        // submits must not be decoded truncated (70_001 lands mid-NAL in this
        // sample).
        let data = sample("h265_main.h265");
        let config = DecoderConfig::only(Backend::Software);

        let (head, rest) = data.split_at(70_001);
        let mut unified = VaccDecoder::new(head, &config).unwrap();
        let mut a: Vec<DecodedFrame> = drain_ready(&mut unified);
        for chunk in rest.chunks(70_001) {
            Decoder::submit(&mut unified, chunk).unwrap();
            a.extend(drain_ready(&mut unified));
        }
        a.extend(unified.flush().unwrap());
        assert_pts(&a);

        let mut direct = vacc_software_decode::SoftwareH265Decoder::new(data).unwrap();
        let b = drain(&mut direct);

        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.iter().zip(&b) {
            assert_eq!((fa.width, fa.height), (fb.width, fb.height));
            let pa = fa.pixel_data.as_ref().unwrap();
            let pb = fb.pixel_data.as_ref().unwrap();
            assert_eq!(pa.buffer, pb.buffer);
        }
    }

    /// GPU variant of the chunked-streaming test: feeding the stream in small
    /// pieces (NAL units split across submits) must produce exactly the frames
    /// a whole-stream feed produces. Skips when the backend's hardware is not
    /// available on this machine.
    fn chunked_matches_whole(backend: Backend, sample_name: &str) {
        let data = sample(sample_name);
        let config = DecoderConfig::only(backend);

        let (head, rest) = data.split_at(70_001);
        let mut unified = match VaccDecoder::new(head, &config) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {backend:?}: backend unavailable: {e}");
                return;
            }
        };
        let mut a: Vec<DecodedFrame> = drain_ready(&mut unified);
        for chunk in rest.chunks(70_001) {
            Decoder::submit(&mut unified, chunk).unwrap();
            a.extend(drain_ready(&mut unified));
        }
        a.extend(unified.flush().unwrap());
        assert_pts(&a);

        let mut whole = VaccDecoder::new(&data, &config).unwrap();
        let b = drain(&mut whole);

        assert!(!a.is_empty());
        assert_eq!(a.len(), b.len(), "frame count mismatch");
        for (fa, fb) in a.iter().zip(&b) {
            assert_eq!((fa.width, fa.height), (fb.width, fb.height));
            let pa = fa.pixel_data.as_ref().unwrap();
            let pb = fb.pixel_data.as_ref().unwrap();
            assert_eq!(pa.buffer, pb.buffer);
        }
    }

    #[test]
    fn streaming_chunked_input_matches_whole_stream_nvdec() {
        chunked_matches_whole(Backend::Nvdec, "h264_main.h264");
    }

    #[test]
    fn streaming_chunked_input_matches_whole_stream_vaapi() {
        chunked_matches_whole(Backend::Vaapi, "h264_main.h264");
    }

    /// GPU decode track: frames must come out device-resident and the GPU
    /// image pipeline must match the host pipeline (same NPP kernels on the
    /// same decoded surface).
    #[cfg(feature = "nvdec")]
    #[test]
    fn gpu_track_matches_cpu_path() {
        if !vacc_npp::Npp::is_available() {
            eprintln!("NPP unavailable; skipping gpu track test");
            return;
        }
        let data = sample("h264_main.h264");
        for scale in [None, Some(Scale::new(320, 180, Interpolation::Bilinear))] {
            let img_cfg = ImageConfig {
                rgb: Some(RgbChannels::Rgb24),
                scale,
                affine: None,
                ..Default::default()
            };
            let mut cpu =
                VaccDecoder::new(&data, &DecoderConfig::only(Backend::Nvdec).with_image(img_cfg)).unwrap();
            let cpu_frames = drain(&mut cpu);
            let mut gpu = VaccDecoder::new(
                &data,
                &DecoderConfig::only(Backend::Nvdec).with_image(img_cfg).with_gpu(),
            )
            .unwrap();
            assert_eq!(gpu.backend(), Backend::Nvdec);
            let gpu_frames = drain(&mut gpu);
            assert_eq!(cpu_frames.len(), gpu_frames.len(), "frame count mismatch (scale={scale:?})");
            for (c, g) in cpu_frames.iter().zip(&gpu_frames) {
                let dev = g
                    .gpu
                    .as_ref()
                    .unwrap_or_else(|| panic!(
                        "frame {} carries no gpu buffer (pixel_data={}, rgb={})",
                        c.frame_index,
                        g.pixel_data.is_some(),
                        g.rgb_pixels.is_some(),
                    ));
                assert_eq!(dev.format, GpuPixelFormat::Rgb24);
                let host = vacc_npp::readback(dev).unwrap();
                let ref_rgb = c
                    .rgb_pixels
                    .as_ref()
                    .unwrap_or_else(|| panic!("cpu frame {} has no rgb", c.frame_index));
                assert_eq!(host.len(), ref_rgb.data.len());
                let off = host
                    .iter()
                    .zip(ref_rgb.data.iter())
                    .filter(|&(a, b)| (i32::from(*a) - i32::from(*b)).abs() > 2)
                    .count();
                assert!(
                    off * 100 < host.len(),
                    "gpu/cpu rgb drift: {off}/{} bytes differ by more than 2",
                    host.len()
                );
            }
        }
    }

    /// Vulkan decode track: frames stay in the decoder's device memory and
    /// the image pipeline runs on-GPU (vkimage). Compared against the software
    /// reference pipeline run on the same backend's readback YUV. The host
    /// pipeline is deliberately not the reference: on NPP hosts it routes
    /// scaling through nppiResize, whose phase convention deviates from the
    /// reference bilinear (an exact 2x downscale degenerates to
    /// nearest-neighbor decimation), while vkimage implements the reference
    /// tap mapping.
    #[test]
    fn vulkan_gpu_track_matches_sw_reference() {
        let data = sample("h264_main.h264");
        for scale in [None, Some(Scale::new(320, 180, Interpolation::Bilinear))] {
            let img_cfg = ImageConfig {
                rgb: Some(RgbChannels::Rgb24),
                scale,
                affine: None,
                ..Default::default()
            };
            // Reference: plain decode + software pipeline on the readback YUV.
            let mut ref_dec =
                VaccDecoder::new(&data, &DecoderConfig::only(Backend::Vulkan)).unwrap();
            let ref_frames = drain(&mut ref_dec);
            let mut gpu = match VaccDecoder::new(
                &data,
                &DecoderConfig::only(Backend::Vulkan).with_image(img_cfg).with_gpu(),
            ) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("vulkan gpu track unavailable ({e}); skipping");
                    return;
                }
            };
            assert_eq!(gpu.backend(), Backend::Vulkan);
            let gpu_frames = drain(&mut gpu);
            assert_eq!(ref_frames.len(), gpu_frames.len(), "frame count mismatch (scale={scale:?})");
            for (r, g) in ref_frames.iter().zip(&gpu_frames) {
                let pd = r
                    .pixel_data
                    .as_ref()
                    .unwrap_or_else(|| panic!("reference frame {} has no yuv", r.frame_index));
                let src = crate::transform::map_source(pd)
                    .unwrap_or_else(|e| panic!("mapping reference frame {}: {e}", r.frame_index));
                let sw_rgb = match vacc_image::process(&src.image(), &img_cfg, vacc_image::Kernel::Auto) {
                    Ok(vacc_image::ProcessedFrame::Rgb(r)) => r.data,
                    other => panic!("sw reference on frame {} did not yield rgb: {other:?})", r.frame_index),
                };
                let dev = g
                    .gpu
                    .as_ref()
                    .unwrap_or_else(|| panic!(
                        "frame {} carries no gpu buffer (pixel_data={}, rgb={})",
                        g.frame_index,
                        g.pixel_data.is_some(),
                        g.rgb_pixels.is_some(),
                    ));
                assert_eq!(dev.format, GpuPixelFormat::Rgb24);
                let host = vacc_vkimage::readback(dev).unwrap();
                assert_eq!(host.len(), sw_rgb.len());
                let off = host
                    .iter()
                    .zip(sw_rgb.iter())
                    .filter(|&(a, b)| (i32::from(*a) - i32::from(*b)).abs() > 2)
                    .count();
                assert!(
                    off * 100 < host.len(),
                    "vulkan gpu/sw rgb drift: {off}/{} bytes differ by more than 2 (frame {})",
                    host.len(),
                    g.frame_index
                );
            }
        }
    }
}

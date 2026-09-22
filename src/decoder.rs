//! The unified decoder: one API over all backends, with fallback.

use std::collections::VecDeque;

use vacc_core::codec::VideoCodec;
use vacc_core::decoder::{Decoder, DecoderInfo};
use vacc_core::format::{ChromaSubsampling, ComponentBitDepth, VideoFormat};
use vacc_core::frame::{DecodedFrame, PixelData, PixelPlane};
use vacc_core::session::Extent2D;

use crate::backend::Backend;
use crate::codec::detect_codec;
use crate::config::DecoderConfig;
use crate::error::{BackendFailure, UnifiedError, UnifiedResult};

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
/// buffered; on the first `decode()` call the full stream is decoded and
/// frames are served one by one.
#[cfg(feature = "vulkan")]
struct VulkanAdapter {
    data: Vec<u8>,
    queue: VecDeque<vacc_vulkan_decode::DecodedFrame>,
    next_index: u32,
    started: bool,
}

#[cfg(feature = "vulkan")]
impl VulkanAdapter {
    fn new(data: Vec<u8>) -> UnifiedResult<Self> {
        // Validate eagerly so a bad stream falls through to the next backend.
        vacc_vulkan_decode::VulkanDecoder::new(data.clone()).map_err(be)?;
        Ok(Self { data, queue: VecDeque::new(), next_index: 0, started: false })
    }

    fn start(&mut self) -> UnifiedResult<()> {
        if self.started {
            return Ok(());
        }
        self.started = true;
        let mut decoder = vacc_vulkan_decode::VulkanDecoder::new(std::mem::take(&mut self.data)).map_err(be)?;
        let frames = decoder.decode_all(usize::MAX).map_err(be)?;
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

/// Convert a Vulkan backend frame (coded-size planes) into the core frame
/// type with cropped, display-size planes — the same convention the other
/// backends use.
#[cfg(feature = "vulkan")]
fn to_core_frame(vk_frame: vacc_vulkan_decode::DecodedFrame, index: u32) -> DecodedFrame {
    let pixels = vk_frame.pixels;
    let ss = pixels.sample_size.max(1) as usize;

    let crop_y = |plane: &[u8], w: u32, left: u32, top: u32| -> Vec<u8> {
        let (dw, dh) = (vk_frame.display_width, vk_frame.display_height);
        let row = w as usize * ss;
        let out_row = dw as usize * ss;
        let mut out = Vec::with_capacity(out_row * dh as usize);
        for r in 0..dh {
            let src = plane
                .chunks_exact(row)
                .nth((top + r) as usize)
                .unwrap_or(&[]);
            let start = (left as usize) * ss;
            out.extend_from_slice(&src[start..start.saturating_add(out_row).min(src.len())]);
        }
        out
    };

    // Chroma crop is in chroma-sample space (half-pel for 4:2:0).
    let half = pixels.chroma_width * 2 == vk_frame.coded_width;
    let (cl_c, ct_c) = if half {
        (vk_frame.crop_left / 2, vk_frame.crop_top / 2)
    } else {
        (vk_frame.crop_left, vk_frame.crop_top)
    };

    let y = crop_y(&pixels.y_plane, vk_frame.coded_width, vk_frame.crop_left, vk_frame.crop_top);
    let u = crop_y(&pixels.u_plane, pixels.chroma_width, cl_c, ct_c);
    let v = crop_y(&pixels.v_plane, pixels.chroma_width, cl_c, ct_c);

    let mut buffer = Vec::new();
    buffer.extend_from_slice(&y);
    let u_off = buffer.len();
    buffer.extend_from_slice(&u);
    let v_off = buffer.len();
    buffer.extend_from_slice(&v);

    let (dw, dh) = (vk_frame.display_width as usize, vk_frame.display_height as usize);
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
/// ```no_run
/// use vacc_core::decoder::Decoder;
/// use vacc::{Backend, DecoderConfig, VaccDecoder};
///
/// let data = std::fs::read("video.h264").unwrap();
///
/// // One-liner: default chain vulkan -> nvdec -> vaapi -> software.
/// let mut decoder = VaccDecoder::new_auto(data).unwrap();
/// println!("using backend: {}", decoder.backend());
/// for frame in decoder.decode_all(usize::MAX).unwrap() {
///     println!("frame {}: {}x{}", frame.frame_index, frame.width, frame.height);
/// }
///
/// // Or pick the preferred order yourself:
/// let data = std::fs::read("video.h264").unwrap();
/// let config = DecoderConfig::new([Backend::Nvdec, Backend::Software]);
/// let decoder = VaccDecoder::new(data, &config).unwrap();
/// ```
pub struct VaccDecoder {
    backend: Backend,
    inner: Box<dyn AnyDecode>,
    /// Next synthetic PTS (microseconds) for frames without a usable
    /// timestamp.
    pts_next: i64,
    /// PTS of the previously emitted frame (monotonicity check).
    last_pts: Option<i64>,
}

/// Synthetic PTS step for streams that carry no timestamps: 1/30 s in
/// microseconds — the same convention the sw and vaapi backends use.
const SYNTHETIC_PTS_STEP: i64 = 33_333;

impl VaccDecoder {
    /// Create a decoder, trying each backend in `config.order()` until one
    /// succeeds. If all fail, the error lists every per-backend failure.
    pub fn new(data: Vec<u8>, config: &DecoderConfig) -> UnifiedResult<Self> {
        if config.is_empty() {
            return Err(UnifiedError::EmptyBackendOrder);
        }
        let mut failures = Vec::new();
        for &backend in config.order() {
            // A panicking backend (e.g. an out-of-bounds bug) must not take
            // down the caller: treat it like any other init failure and fall
            // through to the next backend.
            let created = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create(backend, &data)
            }));
            match created {
                Ok(Ok(inner)) => {
                    log::info!(
                        "unified decoder: using {} backend (config: {})",
                        backend,
                        config
                    );
                    return Ok(Self { backend, inner, pts_next: 0, last_pts: None });
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
    /// (`vulkan -> nvdec -> vaapi -> software`).
    pub fn new_auto(data: Vec<u8>) -> UnifiedResult<Self> {
        Self::new(data, &DecoderConfig::default())
    }

    /// The backend actually in use.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Decode up to `max_frames` frames: drain `decode()` until the stream is
    /// exhausted, then flush any frames still held back by reordering.
    pub fn decode_all(&mut self, max_frames: usize) -> UnifiedResult<Vec<DecodedFrame>> {
        // Go through the trait methods so PTS normalization applies.
        let mut frames = Vec::new();
        while frames.len() < max_frames {
            match self.decode()? {
                Some(frame) => frames.push(frame),
                None => break,
            }
        }
        if frames.len() < max_frames {
            frames.extend(self.flush()?);
            frames.truncate(max_frames);
        }
        Ok(frames)
    }
}

/// One-shot convenience: decode `data` with the default fallback chain and
/// return every frame.
pub fn decode_all(data: Vec<u8>) -> UnifiedResult<Vec<DecodedFrame>> {
    let mut decoder = VaccDecoder::new_auto(data)?;
    decoder.decode_all(usize::MAX)
}

fn create(backend: Backend, data: &[u8]) -> UnifiedResult<Box<dyn AnyDecode>> {
    match backend {
        #[cfg(feature = "vulkan")]
        Backend::Vulkan => Ok(Box::new(VulkanAdapter::new(data.to_vec())?)),
        #[cfg(not(feature = "vulkan"))]
        Backend::Vulkan => Err(disabled("vulkan")),

        #[cfg(feature = "nvdec")]
        Backend::Nvdec => {
            let codec = detect_codec(data).ok_or(UnifiedError::CodecNotDetected)?;
            match codec {
                VideoCodec::DecodeH264 => Ok(Box::new(
                    vacc_nvdec_decode::NvdecH264Decoder::new(data.to_vec()).map_err(be)?,
                )),
                VideoCodec::DecodeH265 => Ok(Box::new(
                    vacc_nvdec_decode::NvdecH265Decoder::new(data.to_vec()).map_err(be)?,
                )),
                VideoCodec::DecodeVp9 => Ok(Box::new(
                    vacc_nvdec_decode::NvdecVp9Decoder::new(data.to_vec()).map_err(be)?,
                )),
                VideoCodec::DecodeAv1 => Ok(Box::new(
                    vacc_nvdec_decode::NvdecAv1Decoder::new(data.to_vec()).map_err(be)?,
                )),
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
        Self::new_auto(data)
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
        }
        Ok(frame)
    }

    fn flush(&mut self) -> Result<Vec<DecodedFrame>, Self::Error> {
        let mut frames = self.inner.flush()?;
        for f in &mut frames {
            self.normalize_pts(f);
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
    /// Guarantee a presentation timestamp on every output frame.
    ///
    /// Backends that decode raw bitstreams (no container PTS) either leave
    /// `pts_valid` false or stamp timestamps in *decode* order, which is not
    /// monotonic once B-frames reorder the output. In both cases a synthetic
    /// monotonic PTS is assigned ([`SYNTHETIC_PTS_STEP`] per frame). A
    /// backend-provided PTS is kept only when it is valid and strictly
    /// greater than the previously emitted one.
    fn normalize_pts(&mut self, frame: &mut DecodedFrame) {
        let monotonic = self.last_pts.map_or(true, |last| frame.timestamp > last);
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

    fn sample(name: &str) -> Vec<u8> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/samples/");
        std::fs::read(format!("{path}{name}")).unwrap()
    }

    fn drain<D: Decoder>(d: &mut D) -> Vec<DecodedFrame> {
        let mut frames = Vec::new();
        while let Some(f) = Decoder::decode(d).unwrap() {
            frames.push(f);
        }
        frames.extend(d.flush().unwrap());
        frames
    }

    #[test]
    fn empty_config_rejected() {
        let data = sample("h264_main.h264");
        let err = VaccDecoder::new(data, &DecoderConfig::new([])).unwrap_err();
        assert!(matches!(err, UnifiedError::EmptyBackendOrder));
    }

    #[test]
    fn software_only_fails_for_unsupported_codec() {
        // AV1 has no software backend; the error must list the failure.
        let data = sample("av1_main.ivf");
        let err = VaccDecoder::new(data, &DecoderConfig::only(Backend::Software)).unwrap_err();
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
            VaccDecoder::new(data.clone(), &DecoderConfig::only(Backend::Software)).unwrap();
        assert_eq!(unified.backend(), Backend::Software);
        let a = unified.decode_all(usize::MAX).unwrap();
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

    #[test]
    fn unified_matches_direct_sw_h265() {
        let data = sample("h265_main.h265");
        let mut unified =
            VaccDecoder::new(data.clone(), &DecoderConfig::only(Backend::Software)).unwrap();
        assert_eq!(unified.backend(), Backend::Software);
        let a = unified.decode_all(usize::MAX).unwrap();
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
}

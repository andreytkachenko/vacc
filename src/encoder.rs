//! The unified encoder: one API over all encode backends, with fallback.
//!
//! [`VaccEncoder`] accepts an [`EncoderConfig`](crate::EncoderConfig) with a
//! preferred backend order (Vulkan, NVENC, VAAPI, software) and falls back to
//! the next backend in the list whenever the previous one is unavailable or
//! cannot encode the configured codec/size.
//!
//! The API is frame-based: construct from an [`EncoderConfig`], feed NV12
//! frames with [`vacc_core::encoder::Encoder::encode`], and drain any
//! in-flight output at end of stream with [`Encoder::flush`]. Each call to
//! `encode` returns the access units (Annex B) produced for that picture —
//! zero to several when a backend reorders frames (B-frames).

use vacc_core::encoder::{EncodedFrame, Encoder, EncoderInfo, EncoderConfig as CoreEncoderConfig, InputFrame};

use crate::backend::Backend;
use crate::config::EncoderConfig;
use crate::error::{BackendFailure, UnifiedError, UnifiedResult};

fn be(e: impl std::error::Error + Send + Sync + 'static) -> UnifiedError {
    UnifiedError::Backend { source: Box::new(e) }
}

/// Type-erased backend encoder.
pub trait AnyEncode {
    fn info(&self) -> EncoderInfo;
    fn encode(&mut self, frame: &InputFrame, timestamp: i64) -> UnifiedResult<Vec<EncodedFrame>>;
    fn flush(&mut self) -> UnifiedResult<Vec<EncodedFrame>>;
}

/// Blanket delegation for types that implement [`vacc_core::Encoder`].
macro_rules! impl_any_encode {
    ($ty:ty) => {
        impl AnyEncode for $ty {
            fn info(&self) -> EncoderInfo {
                <Self as Encoder>::info(self)
            }
            fn encode(&mut self, frame: &InputFrame, timestamp: i64) -> UnifiedResult<Vec<EncodedFrame>> {
                <Self as Encoder>::encode(self, frame, timestamp).map_err(be)
            }
            fn flush(&mut self) -> UnifiedResult<Vec<EncodedFrame>> {
                <Self as Encoder>::flush(self).map_err(be)
            }
        }
    };
}

#[cfg(feature = "vulkan")]
impl_any_encode!(vacc_vulkan_encode::VulkanEncoder);
#[cfg(feature = "nvdec")]
impl_any_encode!(vacc_nvenc_encode::NvencEncoder);
#[cfg(feature = "vaapi")]
impl_any_encode!(vacc_vaapi_encode::VaapiEncoder);
#[cfg(feature = "sw")]
impl_any_encode!(vacc_software_encode::SoftwareEncoder);

/// The unified encoder: one API over all encode backends, with fallback.
pub struct VaccEncoder {
    backend: Backend,
    inner: Box<dyn AnyEncode>,
}

impl VaccEncoder {
    /// Create an encoder, trying each backend in `config.order()` until one
    /// succeeds. If all fail, the error lists every per-backend failure.
    pub fn new(config: &EncoderConfig) -> UnifiedResult<Self> {
        if config.is_empty() {
            return Err(UnifiedError::EmptyBackendOrder);
        }
        let params = config.params();
        let mut failures = Vec::new();
        for &backend in config.order() {
            // A panicking backend (e.g. an out-of-bounds bug) must not take
            // down the caller: treat it like any other init failure and fall
            // through to the next backend.
            let created = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create(backend, params)
            }));
            match created {
                Ok(Ok(inner)) => {
                    log::info!("unified encoder: using {} backend (config: {})", backend, config);
                    return Ok(Self { backend, inner });
                }
                Ok(Err(e)) => {
                    log::warn!("unified encoder: {} backend failed: {}", backend, e);
                    failures.push(BackendFailure { backend, message: e.to_string() });
                }
                Err(_) => {
                    log::warn!("unified encoder: {} backend panicked during init; falling back", backend);
                    failures.push(BackendFailure {
                        backend,
                        message: "backend panicked during initialization".to_string(),
                    });
                }
            }
        }
        Err(UnifiedError::AllBackendsFailed { failures })
    }

    /// Create an encoder with the default fallback chain
    /// (`vulkan -> nvdec -> vaapi -> software`) from core parameters.
    pub fn new_auto(params: &CoreEncoderConfig) -> UnifiedResult<Self> {
        Self::new(&EncoderConfig::new(params.clone()))
    }

    /// The backend actually in use.
    pub fn backend(&self) -> Backend {
        self.backend
    }
}

fn create(backend: Backend, params: &CoreEncoderConfig) -> UnifiedResult<Box<dyn AnyEncode>> {
    match backend {
        #[cfg(feature = "vulkan")]
        Backend::Vulkan => Ok(Box::new(vacc_vulkan_encode::VulkanEncoder::new(params).map_err(be)?)),
        #[cfg(not(feature = "vulkan"))]
        Backend::Vulkan => Err(disabled("vulkan")),

        #[cfg(feature = "nvdec")]
        Backend::Nvdec => Ok(Box::new(vacc_nvenc_encode::NvencEncoder::new(params).map_err(be)?)),
        #[cfg(not(feature = "nvdec"))]
        Backend::Nvdec => Err(disabled("nvdec")),

        #[cfg(feature = "vaapi")]
        Backend::Vaapi => Ok(Box::new(vacc_vaapi_encode::VaapiEncoder::new(params).map_err(be)?)),
        #[cfg(not(feature = "vaapi"))]
        Backend::Vaapi => Err(disabled("vaapi")),

        #[cfg(feature = "sw")]
        Backend::Software => Ok(Box::new(vacc_software_encode::SoftwareEncoder::new(params).map_err(be)?)),
        #[cfg(not(feature = "sw"))]
        Backend::Software => Err(disabled("sw")),
    }
}

#[cfg(any(not(feature = "vulkan"), not(feature = "nvdec"), not(feature = "vaapi"), not(feature = "sw")))]
fn disabled(name: &str) -> UnifiedError {
    UnifiedError::Unsupported {
        message: format!("{name} backend is not enabled in this build (enable the feature)"),
    }
}

impl std::fmt::Debug for VaccEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VaccEncoder").field("backend", &self.backend).finish()
    }
}

impl Encoder for VaccEncoder {
    type Error = UnifiedError;

    fn new(config: &CoreEncoderConfig) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        Self::new_auto(config)
    }

    fn info(&self) -> EncoderInfo {
        self.inner.info()
    }

    fn encode(
        &mut self,
        frame: &InputFrame,
        timestamp: i64,
    ) -> Result<Vec<EncodedFrame>, Self::Error> {
        self.inner.encode(frame, timestamp)
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>, Self::Error> {
        self.inner.flush()
    }
}

//! VAAPI encoder device — display management and encode capability probing.

use libva::{Display, VAEntrypoint};
use std::rc::Rc;
use vacc_core::codec::VideoCodec as CoreVideoCodec;

use super::{Error, Result};

/// VAAPI encoder device that owns the VA display.
pub struct VaapiEncoderDevice {
    display: Rc<Display>,
}

impl VaapiEncoderDevice {
    /// Create a new VAAPI encoder device.
    ///
    /// If the `NVD_GPU` environment variable is set (0-based index), opens
    /// `/dev/dri/renderD{128+idx}`; otherwise opens the first available display.
    pub fn new() -> Result<Self> {
        let display = match std::env::var("NVD_GPU")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
        {
            Some(idx) => {
                let path = format!("/dev/dri/renderD{}", 128 + idx);
                Display::open_drm_display(&path).map_err(|e| {
                    Error::VaApi(format!("Failed to open VA display on {}: {}", path, e))
                })?
            }
            None => Display::open()
                .ok_or_else(|| Error::VaApi("No VA display available".to_string()))?,
        };
        Ok(Self { display })
    }

    /// Get a reference to the inner Display.
    pub fn display(&self) -> &Rc<Display> {
        &self.display
    }

    /// Check whether H.264 Main profile encode is supported on this display.
    pub fn supports_h264_encode(&self) -> bool {
        self.supports_profile(libva::VAProfile::VAProfileH264Main)
    }

    /// Check if a VA profile exposes the slice-encode entrypoint.
    pub fn supports_profile(&self, profile: libva::VAProfile::Type) -> bool {
        match self.display.query_config_entrypoints(profile) {
            Ok(entrypoints) => entrypoints.contains(&VAEntrypoint::VAEntrypointEncSlice),
            Err(_) => false,
        }
    }

    /// List encode codecs supported by this device.
    pub fn supported_codecs(&self) -> Vec<CoreVideoCodec> {
        let mut codecs = Vec::new();
        if self.supports_h264_encode() {
            codecs.push(CoreVideoCodec::EncodeH264);
        }
        codecs
    }

    /// Backend name.
    pub fn backend_name(&self) -> &str {
        "vaapi"
    }
}

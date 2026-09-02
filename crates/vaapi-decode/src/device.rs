//! VAAPI decoder device - implements DecoderDevice trait.

use libva::VAProfile::Type as VAProfileType;
use libva::{Config, Display};
use std::rc::Rc;
use vacc_core::{
    codec::VideoCodec as CoreVideoCodec,
    device::{DecodeCapabilities, DecoderDevice},
    format::{ChromaSubsampling, ComponentBitDepth, VideoFormat},
    session::Extent2D,
};

use super::{Error, Result};

/// VAAPI decoder device that implements the DecoderDevice trait.
pub struct VaapiDecoderDevice {
    display: Rc<Display>,
}

impl VaapiDecoderDevice {
    /// Create a new VAAPI decoder device.
    ///
    /// If the `NVD_GPU` environment variable is set (0-based index), opens
    /// `/dev/dri/renderD{128+idx}`; otherwise opens the first available DRM device.
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
    pub fn display(&self) -> &Display {
        &self.display
    }

    /// Check if a VA profile is supported for video decode.
    fn supports_profile(&self, profile: VAProfileType) -> bool {
        match self.display.query_config_entrypoints(profile) {
            Ok(entrypoints) => entrypoints.contains(&libva::VAEntrypoint::VAEntrypointVLD),
            Err(_) => false,
        }
    }

    /// Map a core VideoCodec to VA profiles.
    fn codec_profiles(codec: CoreVideoCodec) -> Vec<VAProfileType> {
        match codec {
            CoreVideoCodec::DecodeH264 => vec![
                libva::VAProfile::VAProfileH264ConstrainedBaseline,
                libva::VAProfile::VAProfileH264Main,
                libva::VAProfile::VAProfileH264High,
            ],
            CoreVideoCodec::DecodeH265 => vec![
                libva::VAProfile::VAProfileHEVCMain,
                libva::VAProfile::VAProfileHEVCMain10,
            ],
            CoreVideoCodec::DecodeVp9 => vec![
                libva::VAProfile::VAProfileVP9Profile0,
                libva::VAProfile::VAProfileVP9Profile2,
            ],
            CoreVideoCodec::DecodeAv1 => vec![libva::VAProfile::VAProfileAV1Profile0],
            _ => vec![],
        }
    }

    /// Try to create a config for the given codec to verify support.
    fn try_create_config(&self, codec: CoreVideoCodec) -> Result<Config> {
        let profiles = Self::codec_profiles(codec);
        for profile in profiles {
            if self.supports_profile(profile) {
                let config = self.display.create_config(
                    Vec::new(),
                    profile,
                    libva::VAEntrypoint::VAEntrypointVLD,
                );
                if let Ok(config) = config {
                    return Ok(config);
                }
            }
        }
        Err(Error::CodecNotSupported(format!(
            "No supported VA profile for {:?}",
            codec
        )))
    }
}

impl DecoderDevice for VaapiDecoderDevice {
    type Error = Error;

    fn backend_name(&self) -> &str {
        "vaapi"
    }

    fn supports_codec(&self, codec: CoreVideoCodec) -> bool {
        self.try_create_config(codec).is_ok()
    }

    fn supported_codecs(&self) -> Vec<CoreVideoCodec> {
        let mut codecs = Vec::new();
        for codec in [
            CoreVideoCodec::DecodeH264,
            CoreVideoCodec::DecodeH265,
            CoreVideoCodec::DecodeVp9,
            CoreVideoCodec::DecodeAv1,
        ] {
            if self.supports_codec(codec) {
                codecs.push(codec);
            }
        }
        codecs
    }

    fn query_capabilities(
        &self,
        codec: CoreVideoCodec,
        _chroma_subsampling: ChromaSubsampling,
        _luma_bit_depth: ComponentBitDepth,
        _chroma_bit_depth: ComponentBitDepth,
        _profile_idc: Option<u32>,
    ) -> Result<DecodeCapabilities> {
        // Verify the codec is supported
        let _config = self.try_create_config(codec)?;

        Ok(DecodeCapabilities {
            codec_operations: codec,
            min_bitstream_buffer_offset_alignment: 1,
            min_bitstream_buffer_size_alignment: 16,
            picture_access_granularity: Extent2D::new(1, 1),
            min_coded_extent: Extent2D::new(16, 16),
            max_coded_extent: Extent2D::new(4096, 2160),
            max_dpb_slots: 16,
            max_active_reference_pictures: 16,
            supported_formats: vec![
                VideoFormat::new(
                    codec,
                    ChromaSubsampling::_420,
                    ComponentBitDepth::Bit8,
                    ComponentBitDepth::Bit8,
                ),
                VideoFormat::new(
                    codec,
                    ChromaSubsampling::_420,
                    ComponentBitDepth::Bit10,
                    ComponentBitDepth::Bit10,
                ),
            ],
        })
    }

    fn query_supported_formats(&self, codec: CoreVideoCodec) -> Result<Vec<VideoFormat>> {
        // Verify codec is supported
        let _config = self.try_create_config(codec)?;

        Ok(vec![
            VideoFormat::new(
                codec,
                ChromaSubsampling::_420,
                ComponentBitDepth::Bit8,
                ComponentBitDepth::Bit8,
            ),
            VideoFormat::new(
                codec,
                ChromaSubsampling::_420,
                ComponentBitDepth::Bit10,
                ComponentBitDepth::Bit10,
            ),
        ])
    }
}

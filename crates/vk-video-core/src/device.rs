//! Decoder device trait - abstracts backend device for capability queries.
//!
//! This trait allows querying a hardware decoder device (Vulkan, VAAPI, etc.)
//! for supported codecs, profiles, and capabilities before creating a decoder.

use crate::codec::VideoCodec;
use crate::format::{ChromaSubsampling, ComponentBitDepth, VideoFormat};
use crate::session::Extent2D;

/// Decode capabilities for a specific codec/profile combination.
#[derive(Debug, Clone)]
pub struct DecodeCapabilities {
    /// Codec operations supported.
    pub codec_operations: VideoCodec,
    /// Minimum bitstream buffer offset alignment.
    pub min_bitstream_buffer_offset_alignment: u32,
    /// Minimum bitstream buffer size alignment.
    pub min_bitstream_buffer_size_alignment: u32,
    /// Picture access granularity (width, height).
    pub picture_access_granularity: Extent2D,
    /// Minimum coded extent.
    pub min_coded_extent: Extent2D,
    /// Maximum coded extent.
    pub max_coded_extent: Extent2D,
    /// Maximum DPB slots supported.
    pub max_dpb_slots: u32,
    /// Maximum active reference pictures.
    pub max_active_reference_pictures: u32,
    /// Supported output formats.
    pub supported_formats: Vec<VideoFormat>,
}

impl DecodeCapabilities {
    /// Check if the given coded extent is within supported limits.
    pub fn supports_extent(&self, width: u32, height: u32) -> bool {
        let extent = Extent2D::new(width, height);
        extent.width >= self.min_coded_extent.width
            && extent.height >= self.min_coded_extent.height
            && extent.width <= self.max_coded_extent.width
            && extent.height <= self.max_coded_extent.height
    }
}

/// A decoder device that can be queried for capabilities and used to create decoders.
///
/// This trait is implemented by backend-specific device types (e.g., VulkanDevice,
/// VaapiDevice) to provide a uniform interface for capability queries.
pub trait DecoderDevice {
    /// Error type for device operations.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Get the backend name (e.g., "vulkan", "vaapi").
    fn backend_name(&self) -> &str;

    /// Check if the device supports a specific codec.
    fn supports_codec(&self, codec: VideoCodec) -> bool;

    /// Query the list of codecs supported by this device.
    fn supported_codecs(&self) -> Vec<VideoCodec>;

    /// Query decode capabilities for a specific codec and format.
    fn query_capabilities(
        &self,
        codec: VideoCodec,
        chroma_subsampling: ChromaSubsampling,
        luma_bit_depth: ComponentBitDepth,
        chroma_bit_depth: ComponentBitDepth,
        profile_idc: Option<u32>,
    ) -> Result<DecodeCapabilities, Self::Error>;

    /// Query supported output formats for a codec.
    fn query_supported_formats(&self, codec: VideoCodec)
        -> Result<Vec<VideoFormat>, Self::Error>;
}

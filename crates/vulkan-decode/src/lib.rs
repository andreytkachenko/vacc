//! # vulkan-decode
//!
//! Vulkan Video Decode implementation.
//! Implements `vacc_core::Decoder` and `vacc_core::DecoderDevice` traits
//! by wrapping the vacc-vulkan crate.

pub mod decoder;
pub mod device;

pub use decoder::VulkanDecoder;
pub use device::VulkanDecoderDevice;

/// Re-export common types from vacc-vulkan.
pub use vacc_vulkan::{
    readback_decoded_image, AccessUnit, BitstreamBuffer, BitstreamBufferPool, CodecProfileInfo,
    DecodedPixels, DpbEntry, DpbManager, H264OrH265Pps, H264OrH265Sps, LastAccessType, VideoCodec,
    VideoCodec as AccessUnitCodec, VideoSession, VideoSessionParameters, VideoSessionParams,
    Vp9Frame,
};

/// Result type for Vulkan video operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Vulkan video error types.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vacc_vulkan::VideoError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

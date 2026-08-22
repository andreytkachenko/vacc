//! # vulkan-decode
//!
//! Vulkan Video Decode implementation.
//! Implements `vk_video_core::Decoder` and `vk_video_core::DecoderDevice` traits
//! by wrapping the vk-video-vulkan crate.

pub mod device;
pub mod decoder;

pub use device::VulkanDecoderDevice;
pub use decoder::VulkanDecoder;

/// Re-export common types from vk-video-vulkan.
pub use vk_video_vulkan::{
    VideoCodec,
    VideoSession,
    VideoSessionParams,
    VideoSessionParameters,
    CodecProfileInfo,
    BitstreamBuffer,
    BitstreamBufferPool,
    DpbManager,
    DpbEntry,
    LastAccessType,
    AccessUnit,
    H264OrH265Sps,
    H264OrH265Pps,
    VideoCodec as AccessUnitCodec,
    Vp9Frame,
    DecodedPixels,
    readback_decoded_image,
};

/// Result type for Vulkan video operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Vulkan video error types.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk_video_vulkan::VideoError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

//! # vk-video-vulkan
//!
//! Vulkan Video Decoder implementation using `ash` for Vulkan bindings.

pub mod device;
pub mod session;
pub mod buffer;
pub mod frame;
pub mod h264;
pub mod h265;
pub mod vp9;
pub mod av1;
pub mod image;
pub mod codec_types;
pub mod dpb;
pub mod access_unit;
pub mod readback;
pub mod profile_chain;
pub mod decoder;

pub use device::{VideoCodec, VulkanDevice, VideoDeviceBuilder, QueueFamilies};
pub use session::{VideoSession, VideoSessionParams, VideoSessionParameters, CodecProfileInfo};
pub use buffer::{BitstreamBuffer, BitstreamBufferPool};
pub use frame::{DecodedFrame as FrameDecodedFrame, YCbCrPlane};
pub use image::{create_output_image, create_output_image_with_pnext, StagingImage};
pub use codec_types::*;
pub use dpb::{DpbManager, DpbEntry, LastAccessType};
pub use access_unit::{AccessUnit, H264OrH265Sps, H264OrH265Pps, VideoCodec as AccessUnitCodec, Vp9Frame};
pub use readback::{DecodedPixels, readback_decoded_image};
pub use profile_chain::{create_output_image_with_profile, create_bitstream_buffer_with_profile};
pub use decoder::{VideoDecoder, DecodedFrame};

/// Result type for Vulkan video operations.
pub type VideoResult<T> = std::result::Result<T, VideoError>;

/// Vulkan video error types.
#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("Vulkan initialization failed: {0}")]
    VulkanInit(String),

    #[error("Video decode not supported: {0}")]
    VideoNotSupported(String),

    #[error("Device creation failed: {0}")]
    DeviceCreation(String),

    #[error("Session creation failed: {0}")]
    SessionCreation(String),

    #[error("Buffer allocation failed: {0}")]
    BufferAllocation(String),

    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),

    #[error("Image creation failed: {0}")]
    ImageCreation(String),

    #[error("Command buffer recording failed: {0}")]
    CommandBufferRecording(String),

    #[error("Queue submission failed: {0}")]
    QueueSubmission(String),

    #[error("Fence wait failed: {0}")]
    FenceWait(String),

    #[error("Decoder initialization failed: {0}")]
    DecoderInit(String),

    #[error("Format not supported: {0}")]
    FormatNotSupported(String),

    #[error("Capability not available: {0}")]
    CapabilityNotAvailable(String),

    #[error("Codec not supported: {0}")]
    CodecNotSupported(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Application information.
#[derive(Debug, Clone)]
pub struct AppInfo {
    pub name: String,
    pub engine_name: String,
    pub api_version: u32,
}

impl Default for AppInfo {
    fn default() -> Self {
        Self {
            name: "vk-video".to_string(),
            engine_name: "vk-video-vulkan".to_string(),
            api_version: ash::vk::API_VERSION_1_2,
        }
    }
}

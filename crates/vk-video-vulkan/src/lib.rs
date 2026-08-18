//! # vk-video-vulkan
//!
//! Vulkan Video Decoder implementation using `ash` for Vulkan bindings.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::type_complexity)]
#![allow(clippy::field_reassign_with_default)]

pub mod access_unit;
pub mod av1;
pub mod buffer;
pub mod codec_types;
pub mod decoder;
pub mod device;
pub mod dpb;
pub mod frame;
pub mod h264;
pub mod h265;
pub mod image;
pub mod profile_chain;
pub mod readback;
pub mod session;
pub mod vp9;

pub use access_unit::{
    AccessUnit, H264MmcoCommand, H264OrH265Pps, H264OrH265Sps, VideoCodec as AccessUnitCodec,
    Vp9Frame,
};
pub use buffer::{BitstreamBuffer, BitstreamBufferPool};
pub use codec_types::*;
pub use decoder::{DecodedFrame, VideoDecoder};
pub use device::{QueueFamilies, VideoCodec, VideoDeviceBuilder, VulkanDevice};
pub use dpb::{DpbEntry, DpbManager, LastAccessType};
pub use frame::{DecodedFrame as FrameDecodedFrame, YCbCrPlane};
pub use image::{create_output_image, create_output_image_with_pnext, StagingImage};
pub use profile_chain::{create_bitstream_buffer_with_profile, create_output_image_with_profile};
pub use readback::{readback_decoded_image, DecodedPixels};
pub use session::{CodecProfileInfo, VideoSession, VideoSessionParameters, VideoSessionParams};

/// Returns true when `VACC_DEBUG=1` is set. Gates the verbose per-frame
/// debug dumps (picture-info dumps, DPB state, fence tracing, ...).
/// Off by default so normal decodes stay quiet.
pub fn vacc_debug() -> bool {
    std::env::var("VACC_DEBUG").ok().unwrap_or_default() == "1"
}

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

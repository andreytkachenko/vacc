//! # vk-video-core
//!
//! Core types, traits, and abstractions for Vulkan Video decoding.
//! This crate defines the codec-agnostic interface that all decoder
//! implementations must conform to.
//!
//! ## Architecture
//!
//! ```text
//! +-------------------+     +-------------------+     +-------------------+
//! |  VideoDecoder     |     |  VideoCodec       |     |  VideoFormat      |
//! |  (trait)          |<--->|  (enum)           |<--->|  (struct)         |
//! +-------------------+     +-------------------+     +-------------------+
//!        |                          |
//!        v                          v
//! +-------------------+     +-------------------+
//! |  DecodedFrame     |     | PictureParameters |
//! |  (struct)         |     |  (codec-specific) |
//! +-------------------+     +-------------------+
//! ```

pub mod codec;
pub mod format;
pub mod picture;
pub mod frame;
pub mod error;
pub mod session;
pub mod decoder;
pub mod device;

pub use codec::{VideoCodec, VideoCodecFlagBits, VideoCodecOperation};
pub use format::{VideoFormat, ChromaSubsampling, ComponentBitDepth, VideoProfile};
pub use picture::{PictureParametersSet, StdType, ParameterType};
pub use frame::{DecodedFrame, FieldFlags, FrameSyncInfo, PixelData, PixelPlane};
pub use error::{VideoResult, VideoError};
pub use session::{VideoSessionParams, VideoDecodeInfo, PictureResourceInfo};
pub use decoder::{Decoder, DecoderInfo};
pub use device::{DecoderDevice, DecodeCapabilities};

/// Maximum number of DPB (Decoded Picture Buffer) reference slots.
pub const MAX_DPB_REF_SLOTS: usize = 16;

/// Maximum number of slices per picture.
pub const MAX_SLICES: usize = 8192;

/// Maximum frame delay between decode and display.
pub const MAX_FRAME_DELAY: usize = 32;

/// Maximum PTS queue size.
pub const MAX_QUEUED_PTS: usize = 16;

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
pub mod decoder;
pub mod error;
pub mod format;
pub mod frame;
pub mod picture;
pub mod session;

pub use codec::{VideoCodec, VideoCodecFlagBits, VideoCodecOperation};
pub use decoder::{Decoder, DecoderInfo};
pub use error::{VideoError, VideoResult};
pub use format::{ChromaSubsampling, ComponentBitDepth, VideoFormat, VideoProfile};
pub use frame::{DecodedFrame, FieldFlags, FrameSyncInfo};
pub use picture::{ParameterType, PictureParametersSet, StdType};
pub use session::{PictureResourceInfo, VideoDecodeInfo, VideoSessionParams};

/// Maximum number of DPB (Decoded Picture Buffer) reference slots.
pub const MAX_DPB_REF_SLOTS: usize = 16;

/// Maximum number of slices per picture.
pub const MAX_SLICES: usize = 8192;

/// Maximum frame delay between decode and display.
pub const MAX_FRAME_DELAY: usize = 32;

/// Maximum PTS queue size.
pub const MAX_QUEUED_PTS: usize = 16;

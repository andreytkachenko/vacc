//! Stateful decoder trait.
//!
//! This trait defines the interface for a stateful video decoder that manages
//! its own session, DPB, and codec state. Backend implementations (Vulkan, VAAPI)
//! implement this trait to provide a uniform decode API.
//!
//! ## Usage pattern
//!
//! ```text
//! decoder.submit(&bitstream_data);  // push frame for decode
//! if let Some(frame) = decoder.decode()? {
//!     // use decoded frame
//! }
//! ```

use crate::codec::VideoCodec;
use crate::frame::DecodedFrame;
use crate::format::{ChromaSubsampling, ComponentBitDepth, VideoFormat};
use crate::session::Extent2D;

/// Information about a decoder instance.
#[derive(Debug, Clone)]
pub struct DecoderInfo {
    /// Backend name (e.g., "vulkan", "vaapi").
    pub backend: String,
    /// Codec being decoded.
    pub codec: VideoCodec,
    /// Coded (aligned) resolution.
    pub coded_size: Extent2D,
    /// Display resolution (after crop).
    pub display_size: Extent2D,
    /// Chroma subsampling.
    pub chroma_subsampling: ChromaSubsampling,
    /// Luma bit depth.
    pub luma_bit_depth: ComponentBitDepth,
    /// Chroma bit depth.
    pub chroma_bit_depth: ComponentBitDepth,
    /// Profile ID.
    pub profile_idc: Option<u32>,
    /// Number of DPB slots in use.
    pub dpb_slots: u32,
}

/// A stateful video decoder.
///
/// Implementations manage their own internal state including:
/// - Hardware session/context
/// - Decoded Picture Buffer (DPB)
/// - Reference frame management
/// - Bitstream buffers
pub trait Decoder {
    /// Error type for decode operations.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Create a new decoder from initial bitstream data.
    ///
    /// Parses the initial data to detect codec, profile, and format,
    /// then initializes the hardware decoder session.
    fn new(data: Vec<u8>) -> Result<Self, Self::Error>
    where
        Self: Sized;

    /// Create a new decoder with explicit format parameters.
    fn new_with_format(
        data: Vec<u8>,
        codec: VideoCodec,
        format: &VideoFormat,
    ) -> Result<Self, Self::Error>
    where
        Self: Sized;

    /// Get decoder information.
    fn info(&self) -> DecoderInfo;

    /// Submit bitstream data for decoding.
    ///
    /// Pushes an access unit (or frame) into the decoder pipeline.
    /// Call `decode()` afterwards to retrieve the decoded frame.
    fn submit(&mut self, data: &[u8]) -> Result<(), Self::Error>;

    /// Pull the next decoded frame.
    ///
    /// Returns `Ok(Some(frame))` if a decoded frame is available,
    /// `Ok(None)` if no frame is ready yet (call `submit()` first or try later).
    fn decode(&mut self) -> Result<Option<DecodedFrame>, Self::Error>;

    /// Flush the decoder, draining any pending frames.
    ///
    /// Returns frames that were in flight.
    fn flush(&mut self) -> Result<Vec<DecodedFrame>, Self::Error>;

    /// Reset the decoder state.
    ///
    /// Invalidates the DPB and prepares for a new stream or seek.
    fn reset(&mut self) -> Result<(), Self::Error>;
}



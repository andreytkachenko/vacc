//! Stateful encoder trait.
//!
//! This trait defines the interface for a stateful video encoder that manages
//! its own session, reference pool, and codec state. Backend implementations
//! (VAAPI, NVENC, Vulkan) implement this trait to provide a uniform encode API.
//!
//! ## Usage pattern
//!
//! ```text
//! let mut enc = encoder::new(&config)?;
//! for frame in input_frames {
//!     let out = enc.encode(frame, pts)?;  // encoded access unit (Annex B)
//! }
//! ```

use crate::codec::{VideoCodec, H264ProfileIdc};
use crate::session::Extent2D;

/// Configuration for creating an encoder.
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    /// Codec to encode (must be an encode variant).
    pub codec: VideoCodec,
    /// Input frame width in pixels (must be a multiple of 16 for H.264).
    pub width: u32,
    /// Input frame height in pixels (must be a multiple of 16 for H.264).
    pub height: u32,
    /// Frame rate numerator (e.g. 30).
    pub fps_num: u32,
    /// Frame rate denominator (e.g. 1).
    pub fps_den: u32,
    /// Constant quantization parameter for CQP rate control (0-51).
    pub qp: u8,
    /// Emit an IDR frame every `intra_period` frames.
    pub intra_period: u32,
    /// Maximum number of reference frames (H.264 SPS max_num_ref_frames).
    pub max_ref_frames: u32,
    /// H.264 profile to encode with.
    pub h264_profile: H264ProfileIdc,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            codec: VideoCodec::EncodeH264,
            width: 320,
            height: 240,
            fps_num: 30,
            fps_den: 1,
            qp: 26,
            intra_period: 30,
            max_ref_frames: 1,
            h264_profile: H264ProfileIdc::Main,
        }
    }
}

impl EncoderConfig {
    /// Validate the configuration, returning an error message if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if !self.codec.is_encode() {
            return Err(format!("codec {:?} is not an encode codec", self.codec));
        }
        if !self.width.is_multiple_of(16) || !self.height.is_multiple_of(16) {
            return Err(format!(
                "H.264 requires width/height to be multiples of 16 (got {}x{})",
                self.width, self.height
            ));
        }
        if self.width == 0 || self.height == 0 {
            return Err("width and height must be non-zero".into());
        }
        if self.fps_num == 0 || self.fps_den == 0 {
            return Err("fps_num and fps_den must be non-zero".into());
        }
        if self.intra_period == 0 {
            return Err("intra_period must be non-zero".into());
        }
        Ok(())
    }
}

/// Information about an encoder instance.
#[derive(Debug, Clone)]
pub struct EncoderInfo {
    /// Backend name (e.g., "vaapi", "nvenc", "vulkan").
    pub backend: String,
    /// Codec being encoded.
    pub codec: VideoCodec,
    /// Coded resolution.
    pub coded_size: Extent2D,
    /// Quantization parameter (CQP mode).
    pub qp: u8,
    /// Frame rate numerator.
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
}

/// NV12 input frame for encoding.
///
/// The luma plane is `width x height`; the interleaved chroma plane is
/// `width x height/2` bytes (one byte per 2x2 pixel block). Rows may be
/// padded — use the pitch fields.
#[derive(Debug)]
pub struct InputFrame<'a> {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Luma (Y) plane data.
    pub y: &'a [u8],
    /// Luma row pitch in bytes (>= width).
    pub y_pitch: usize,
    /// Interleaved chroma (UV) plane data.
    pub uv: &'a [u8],
    /// Chroma row pitch in bytes (>= width).
    pub uv_pitch: usize,
}

impl<'a> InputFrame<'a> {
    /// Create an input frame from tightly-packed NV12 data (pitch == width).
    pub fn packed(width: u32, height: u32, y: &'a [u8], uv: &'a [u8]) -> Self {
        Self {
            width,
            height,
            y,
            y_pitch: width as usize,
            uv,
            uv_pitch: width as usize,
        }
    }

    /// Create an input frame from planar `PixelData` in NV12 layout.
    pub fn from_nv12(width: u32, height: u32, y: &'a [u8], y_pitch: usize, uv: &'a [u8], uv_pitch: usize) -> Self {
        Self {
            width,
            height,
            y,
            y_pitch,
            uv,
            uv_pitch,
        }
    }
}

/// Coded picture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// Intra-coded picture.
    I,
    /// Forward-interpredicted picture.
    P,
    /// Bi-interpredicted picture.
    B,
}

/// An encoded access unit.
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// Frame index in the input sequence (0-based).
    pub frame_index: u32,
    /// Presentation timestamp in nanoseconds.
    pub timestamp: i64,
    /// Decoding timestamp in nanoseconds (`<=` `timestamp`; equal for backends
    /// without B-frame reordering).
    pub dts: i64,
    /// Coded data in Annex B format (NAL units with 00 00 01 start codes).
    pub data: Vec<u8>,
    /// Whether this frame is an IDR (intra) frame.
    pub is_idr: bool,
    /// Coded picture type.
    pub kind: FrameKind,
    /// Picture Order Count.
    pub poc: i32,
}

impl EncodedFrame {
    /// Size of the coded data in bytes.
    pub const fn size(&self) -> usize {
        self.data.len()
    }
}

/// A stateful video encoder.
///
/// Implementations manage their own internal state including:
/// - Hardware session/context
/// - Reference picture pool (surfaces/DPB slots)
/// - Codec parameter sets (SPS/PPS generation and emission)
pub trait Encoder {
    /// Error type for encode operations.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Create a new encoder for the given configuration.
    fn new(config: &EncoderConfig) -> Result<Self, Self::Error>
    where
        Self: Sized;

    /// Get encoder information.
    fn info(&self) -> EncoderInfo;

    /// Encode one input frame.
    ///
    /// The frame is uploaded to the backend, encoded, and the resulting
    /// access units (Annex B NAL data) are returned in decoding order. A
    /// frame-based GOP without reordering produces exactly one access unit;
    /// backends with B-frame reordering may return zero (the picture is held
    /// for reordering) or several per call. The first frame encoded is
    /// always an IDR frame.
    fn encode(
        &mut self,
        frame: &InputFrame,
        timestamp: i64,
    ) -> Result<Vec<EncodedFrame>, Self::Error>;

    /// Flush the encoder, draining any in-flight frames.
    ///
    /// Returns any pending encoded frames (empty for synchronous backends).
    fn flush(&mut self) -> Result<Vec<EncodedFrame>, Self::Error>;
}

//! # vk-video-parser
//!
//! Bitstream parser for H.264, H.265, VP9, and AV1 video codecs.
//!
//! This crate provides:
//! - Start-code detection and NAL unit extraction
//! - RBSP (Raw Byte Sequence Payload) parsing
//! - SPS/PPS/VPS extraction for H.264 and H.265
//! - VP9 frame header parsing with superframe support
//! - AV1 sequence header parsing
//! - Bitstream buffer management
//!
//! The parser is designed to feed decoded picture parameter sets
//! to the Vulkan video decoder layer.

#![allow(clippy::field_reassign_with_default)]

pub mod av1;
pub mod bitreader;
pub mod bitstream;
pub mod h264;
pub mod h265;
pub mod nal;
pub mod vp9;

pub use bitreader::BitReader;

pub use bitstream::{BitstreamBuffer, BitstreamPacket, PacketFlags};
pub use nal::{NalUnit, NalUnitType};

/// Parser result type.
pub type ParserResult<T> = std::result::Result<T, ParserError>;

/// Parser error types.
#[derive(Debug, thiserror::Error)]
pub enum ParserError {
    #[error("Invalid bitstream data")]
    InvalidBitstream,

    #[error("Start code not found")]
    StartCodeNotFound,

    #[error("NAL unit type not supported: {0}")]
    UnsupportedNalType(u8),

    #[error("SPS/PPS/VPS parse error")]
    ParameterSetParse,

    #[error("Buffer too small: needed {needed}, got {have}")]
    BufferTooSmall { needed: usize, have: usize },

    #[error("Emulation prevention byte error")]
    EmulationPreventionError,

    #[error("Trailing bits error")]
    TrailingBitsError,

    #[error("Non-compliant stream")]
    NonCompliantStream,

    #[error("EOS reached")]
    EndOfStream,

    #[error("Bit reader error")]
    BitReader,
}

impl From<crate::bitreader::ParserError> for ParserError {
    fn from(_: crate::bitreader::ParserError) -> Self {
        ParserError::BitReader
    }
}

/// Detected video format from parsing.
#[derive(Debug, Clone)]
pub struct DetectedVideoFormat {
    /// Codec type.
    pub codec: vk_video_core::codec::VideoCodec,
    /// Chroma subsampling.
    pub chroma_subsampling: vk_video_core::format::ChromaSubsampling,
    /// Luma bit depth.
    pub luma_bit_depth: vk_video_core::format::ComponentBitDepth,
    /// Chroma bit depth.
    pub chroma_bit_depth: vk_video_core::format::ComponentBitDepth,
    /// Coded width.
    pub coded_width: u32,
    /// Coded height.
    pub coded_height: u32,
    /// Display area.
    pub display_area: DisplayArea,
    /// Frame rate (numerator/denominator).
    pub frame_rate: FrameRate,
    /// Video signal description.
    pub video_signal: VideoSignalDescription,
    /// Codec profile.
    pub codec_profile: u32,
    /// Film grain support (AV1).
    pub film_grain_used: bool,
    /// Sequence update flag.
    pub sequence_update: bool,
    /// Progressive sequence.
    pub progressive_sequence: bool,
}

impl DetectedVideoFormat {
    /// Create a new detected format.
    pub fn new(codec: vk_video_core::codec::VideoCodec) -> Self {
        Self {
            codec,
            chroma_subsampling: vk_video_core::format::ChromaSubsampling::_420,
            luma_bit_depth: vk_video_core::format::ComponentBitDepth::Bit8,
            chroma_bit_depth: vk_video_core::format::ComponentBitDepth::Bit8,
            coded_width: 0,
            coded_height: 0,
            display_area: DisplayArea::default(),
            frame_rate: FrameRate::default(),
            video_signal: VideoSignalDescription::default(),
            codec_profile: 0,
            film_grain_used: false,
            sequence_update: false,
            progressive_sequence: true,
        }
    }
}

/// Display area within the coded frame.
#[derive(Debug, Clone, Default)]
pub struct DisplayArea {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

/// Frame rate specification.
#[derive(Debug, Clone, Default)]
pub struct FrameRate {
    pub numerator: u32,
    pub denominator: u32,
}

/// Video signal description (VUI parameters).
#[derive(Debug, Clone, Default)]
pub struct VideoSignalDescription {
    pub video_format: u8,
    pub video_full_range_flag: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
}

/// Parser state for a codec.
pub trait VideoParser {
    /// Initialize the parser.
    fn init(&mut self, format: &DetectedVideoFormat) -> ParserResult<()>;

    /// Parse a bitstream packet.
    fn parse(&mut self, packet: &BitstreamPacket) -> ParserResult<ParseResult>;

    /// Reset the parser state.
    fn reset(&mut self);

    /// Get the detected format (after parsing SPS/PPS).
    fn detected_format(&self) -> &DetectedVideoFormat;
}

/// Result of parsing a bitstream packet.
#[derive(Debug, Clone)]
pub enum ParseResult {
    /// Nothing parsed yet.
    Nothing,
    /// SPS/PPS/VPS parameter set received.
    ParameterSet {
        sps: Option<vk_video_core::picture::BoxedPictureParametersSet>,
        pps: Option<vk_video_core::picture::BoxedPictureParametersSet>,
        vps: Option<vk_video_core::picture::BoxedPictureParametersSet>,
    },
    /// Slice data received (ready for decode).
    Slice {
        /// Slice data in the bitstream buffer.
        slice_data_offset: usize,
        /// Slice data length.
        slice_data_len: usize,
        /// Number of slices.
        num_slices: u32,
        /// Parsed slice header information (from the first slice of the frame).
        slice_header: Option<crate::h265::SliceHeaderInfo>,
    },
    /// End of stream.
    EndOfStream,
}

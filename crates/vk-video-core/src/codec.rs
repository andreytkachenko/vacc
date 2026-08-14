//! Video codec identification and operations.
//!
//! Maps directly to Vulkan's `VkVideoCodecOperationFlagBitsKHR`.

use bitflags::bitflags;
use std::fmt;

/// Video codec types (maps to `VkVideoCodecOperationFlagBitsKHR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum VideoCodec {
    /// No codec
    None = 0,
    /// H.264/AVC decode
    DecodeH264 = 1,
    /// H.265/HEVC decode
    DecodeH265 = 2,
    /// AV1 decode
    DecodeAv1 = 4,
    /// VP9 decode
    DecodeVp9 = 8,
    /// H.264/AVC encode
    EncodeH264 = 0x10,
    /// H.265/HEVC encode
    EncodeH265 = 0x20,
    /// AV1 encode
    EncodeAv1 = 0x40,
}

impl VideoCodec {
    /// Returns the codec name as a string.
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::DecodeH264 => "h264",
            Self::DecodeH265 => "h265",
            Self::DecodeAv1 => "av1",
            Self::DecodeVp9 => "vp9",
            Self::EncodeH264 => "encode_h264",
            Self::EncodeH265 => "encode_h265",
            Self::EncodeAv1 => "encode_av1",
        }
    }

    /// Returns true if this is a decode operation.
    pub const fn is_decode(self) -> bool {
        matches!(
            self,
            Self::DecodeH264 | Self::DecodeH265 | Self::DecodeAv1 | Self::DecodeVp9
        )
    }

    /// Returns true if this is an encode operation.
    pub const fn is_encode(self) -> bool {
        matches!(self, Self::EncodeH264 | Self::EncodeH265 | Self::EncodeAv1)
    }

    /// Returns the Vulkan structure type for codec-specific capabilities.
    pub const fn vk_decode_capabilities_stype(self) -> ash::vk::StructureType {
        match self {
            Self::None => ash::vk::StructureType::from_raw(0),
            Self::DecodeH264 => ash::vk::StructureType::VIDEO_DECODE_H264_CAPABILITIES_KHR,
            Self::DecodeH265 => ash::vk::StructureType::VIDEO_DECODE_H265_CAPABILITIES_KHR,
            Self::DecodeAv1 => ash::vk::StructureType::VIDEO_DECODE_AV1_CAPABILITIES_KHR,
            // ash 0.38 doesn't expose VIDEO_DECODE_VP9_CAPABILITIES_KHR
            Self::DecodeVp9 => ash::vk::StructureType::from_raw(1_000_028_003),
            Self::EncodeH264 | Self::EncodeH265 | Self::EncodeAv1 => {
                ash::vk::StructureType::from_raw(0)
            }
        }
    }
}

impl fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Video codec operations as bitflags.
#[allow(unused_doc_comments)]
bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VideoCodecFlagBits: u32 {
        const NONE = 0;
        const DECODE_H264 = 1;
        const DECODE_H265 = 2;
        const DECODE_AV1 = 4;
        const DECODE_VP9 = 8;
        const ENCODE_H264 = 0x10;
        const ENCODE_H265 = 0x20;
        const ENCODE_AV1 = 0x40;
        const ALL_DECODE = Self::DECODE_H264.bits()
            | Self::DECODE_H265.bits()
            | Self::DECODE_AV1.bits()
            | Self::DECODE_VP9.bits();
        const ALL_ENCODE = Self::ENCODE_H264.bits()
            | Self::ENCODE_H265.bits()
            | Self::ENCODE_AV1.bits();
        const ALL = Self::ALL_DECODE.bits() | Self::ALL_ENCODE.bits();
    }
}

impl VideoCodecFlagBits {
    /// Check if a specific codec is supported.
    pub fn has_codec(self, codec: VideoCodec) -> bool {
        let flag = match codec {
            VideoCodec::None => Self::NONE,
            VideoCodec::DecodeH264 => Self::DECODE_H264,
            VideoCodec::DecodeH265 => Self::DECODE_H265,
            VideoCodec::DecodeAv1 => Self::DECODE_AV1,
            VideoCodec::DecodeVp9 => Self::DECODE_VP9,
            VideoCodec::EncodeH264 => Self::ENCODE_H264,
            VideoCodec::EncodeH265 => Self::ENCODE_H265,
            VideoCodec::EncodeAv1 => Self::ENCODE_AV1,
        };
        self.contains(flag)
    }
}

impl Default for VideoCodecFlagBits {
    fn default() -> Self {
        Self::NONE
    }
}

/// Video operation type (decode or encode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodecOperation {
    Decode(VideoCodec),
    Encode(VideoCodec),
}

impl VideoCodecOperation {
    /// Returns the underlying codec.
    pub const fn codec(self) -> VideoCodec {
        match self {
            Self::Decode(c) | Self::Encode(c) => c,
        }
    }

    /// Returns true if this is a decode operation.
    pub const fn is_decode(self) -> bool {
        matches!(self, Self::Decode(_))
    }

    /// Returns true if this is an encode operation.
    pub const fn is_encode(self) -> bool {
        matches!(self, Self::Encode(_))
    }
}

impl fmt::Display for VideoCodecOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(c) => write!(f, "decode {c}"),
            Self::Encode(c) => write!(f, "encode {c}"),
        }
    }
}

/// Standard profile IDs for each codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdProfileIdc {
    /// H.264 profiles
    H264(H264ProfileIdc),
    /// H.265 profiles
    H265(H265ProfileIdc),
    /// AV1 profiles
    Av1(Av1Profile),
}

/// H.264 profile IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum H264ProfileIdc {
    Baseline = 66,
    Main = 77,
    High = 100,
    High444Predictive = 110,
}

/// H.265 profile IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum H265ProfileIdc {
    Main = 1,
    Main10 = 2,
    MainStillPicture = 3,
    FormatRangeExtensions = 4,
    SccExtensions = 9,
}

/// AV1 profile IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Av1Profile {
    Main = 0,
    High = 1,
    Professional = 2,
}

impl StdProfileIdc {
    /// Returns the profile ID as a u32.
    pub const fn as_u32(self) -> u32 {
        match self {
            Self::H264(p) => p as u32,
            Self::H265(p) => p as u32,
            Self::Av1(p) => p as u32,
        }
    }
}

impl fmt::Display for StdProfileIdc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::H264(p) => write!(
                f,
                "H264_{}",
                match p {
                    H264ProfileIdc::Baseline => "BASELINE",
                    H264ProfileIdc::Main => "MAIN",
                    H264ProfileIdc::High => "HIGH",
                    H264ProfileIdc::High444Predictive => "HIGH_444_PREDICTIVE",
                }
            ),
            Self::H265(p) => write!(
                f,
                "H265_{}",
                match p {
                    H265ProfileIdc::Main => "MAIN",
                    H265ProfileIdc::Main10 => "MAIN_10",
                    H265ProfileIdc::MainStillPicture => "MAIN_STILL_PICTURE",
                    H265ProfileIdc::FormatRangeExtensions => "FORMAT_RANGE_EXTENSIONS",
                    H265ProfileIdc::SccExtensions => "SCC_EXTENSIONS",
                }
            ),
            Self::Av1(p) => write!(
                f,
                "AV1_{}",
                match p {
                    Av1Profile::Main => "MAIN",
                    Av1Profile::High => "HIGH",
                    Av1Profile::Professional => "PROFESSIONAL",
                }
            ),
        }
    }
}

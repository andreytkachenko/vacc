//! Video format information.
//!
//! Maps to Vulkan's `VkVideoChromaSubsamplingFlagBitsKHR`,
//! `VkVideoComponentBitDepthFlagBitsKHR`, and `VkVideoProfileInfoKHR`.

use std::fmt;

/// Chroma subsampling mode (maps to `VkVideoChromaSubsamplingFlagBitsKHR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ChromaSubsampling {
    Monochrome = 1,
    _420 = 2,
    _422 = 4,
    _444 = 8,
}

impl ChromaSubsampling {
    /// Returns the Vulkan flag value.
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Returns the Vulkan flag name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Monochrome => "MONOCHROME",
            Self::_420 => "420",
            Self::_422 => "422",
            Self::_444 => "444",
        }
    }

    /// Get the number of planes for this chroma subsampling.
    pub const fn num_planes(self) -> usize {
        match self {
            Self::Monochrome => 1,
            Self::_420 | Self::_422 | Self::_444 => 2, // Semi-planar (Y + UV)
        }
    }

    /// Get the chroma width/height divisor.
    pub const fn chroma_divisor(self) -> u32 {
        match self {
            Self::Monochrome => 0,
            Self::_420 => 2,
            Self::_422 => 2,
            Self::_444 => 1,
        }
    }
}

impl fmt::Display for ChromaSubsampling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Component bit depth (maps to `VkVideoComponentBitDepthFlagBitsKHR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ComponentBitDepth {
    Bit8 = 1,
    Bit10 = 2,
    Bit12 = 4,
}

impl ComponentBitDepth {
    /// Returns the bit depth as u32.
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Returns the actual bit depth.
    pub const fn bit_depth(self) -> u32 {
        match self {
            Self::Bit8 => 8,
            Self::Bit10 => 10,
            Self::Bit12 => 12,
        }
    }

    /// Returns the Vulkan flag name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bit8 => "8-bit",
            Self::Bit10 => "10-bit",
            Self::Bit12 => "12-bit",
        }
    }
}

impl fmt::Display for ComponentBitDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Video format specification.
///
/// This corresponds to `VkVideoProfileInfoKHR` in Vulkan,
/// describing the complete video profile including codec,
/// chroma subsampling, and bit depths.
#[derive(Debug, Clone)]
pub struct VideoFormat {
    /// The codec type.
    pub codec: super::codec::VideoCodec,
    /// Chroma subsampling mode.
    pub chroma_subsampling: ChromaSubsampling,
    /// Luma bit depth.
    pub luma_bit_depth: ComponentBitDepth,
    /// Chroma bit depth.
    pub chroma_bit_depth: ComponentBitDepth,
    /// Codec-specific profile ID.
    pub profile_idc: Option<super::codec::StdProfileIdc>,
    /// Whether film grain is supported (AV1).
    pub film_grain_support: bool,
    /// Picture layout (H.264 specific).
    pub h264_picture_layout: H264PictureLayout,
}

impl VideoFormat {
    /// Create a new video format.
    pub const fn new(
        codec: super::codec::VideoCodec,
        chroma_subsampling: ChromaSubsampling,
        luma_bit_depth: ComponentBitDepth,
        chroma_bit_depth: ComponentBitDepth,
    ) -> Self {
        Self {
            codec,
            chroma_subsampling,
            luma_bit_depth,
            chroma_bit_depth,
            profile_idc: None,
            film_grain_support: false,
            h264_picture_layout: H264PictureLayout::Progressive,
        }
    }

    /// Set the profile ID.
    pub fn with_profile(mut self, profile_idc: super::codec::StdProfileIdc) -> Self {
        self.profile_idc = Some(profile_idc);
        self
    }

    /// Set film grain support.
    pub fn with_film_grain(mut self, support: bool) -> Self {
        self.film_grain_support = support;
        self
    }

    /// Check if this format is 16-bit (10 or 12 bit depth).
    pub fn is_16bit(&self) -> bool {
        self.luma_bit_depth != ComponentBitDepth::Bit8
            || self.chroma_bit_depth != ComponentBitDepth::Bit8
    }

    /// Get the Vulkan format for this video format.
    ///
    /// Returns the appropriate `VK_FORMAT_*` value based on
    /// chroma subsampling and bit depth.
    pub fn vk_format(&self, is_semi_planar: bool) -> Option<VkVideoFormat> {
        let semi_planar = is_semi_planar;
        match (self.chroma_subsampling, self.luma_bit_depth) {
            (ChromaSubsampling::Monochrome, ComponentBitDepth::Bit8) => {
                Some(VkVideoFormat::R8_UNORM)
            }
            (ChromaSubsampling::Monochrome, ComponentBitDepth::Bit10) => {
                Some(VkVideoFormat::R10X6_UNORM_PACK16)
            }
            (ChromaSubsampling::Monochrome, ComponentBitDepth::Bit12) => {
                Some(VkVideoFormat::R12X4_UNORM_PACK16)
            }
            (ChromaSubsampling::_420, ComponentBitDepth::Bit8) => {
                if semi_planar {
                    Some(VkVideoFormat::G8_B8R8_2PLANE_420_UNORM)
                } else {
                    Some(VkVideoFormat::G8_B8_R8_3PLANE_420_UNORM)
                }
            }
            (ChromaSubsampling::_420, ComponentBitDepth::Bit10) => {
                if semi_planar {
                    Some(VkVideoFormat::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16)
                } else {
                    Some(VkVideoFormat::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16)
                }
            }
            (ChromaSubsampling::_420, ComponentBitDepth::Bit12) => {
                if semi_planar {
                    Some(VkVideoFormat::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16)
                } else {
                    Some(VkVideoFormat::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16)
                }
            }
            (ChromaSubsampling::_422, ComponentBitDepth::Bit8) => {
                if semi_planar {
                    Some(VkVideoFormat::G8_B8R8_2PLANE_422_UNORM)
                } else {
                    Some(VkVideoFormat::G8_B8_R8_3PLANE_422_UNORM)
                }
            }
            (ChromaSubsampling::_422, ComponentBitDepth::Bit10) => {
                if semi_planar {
                    Some(VkVideoFormat::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16)
                } else {
                    Some(VkVideoFormat::G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16)
                }
            }
            (ChromaSubsampling::_422, ComponentBitDepth::Bit12) => {
                if semi_planar {
                    Some(VkVideoFormat::G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16)
                } else {
                    Some(VkVideoFormat::G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16)
                }
            }
            (ChromaSubsampling::_444, ComponentBitDepth::Bit8) => {
                if semi_planar {
                    Some(VkVideoFormat::G8_B8R8_2PLANE_444_UNORM_EXT)
                } else {
                    Some(VkVideoFormat::G8_B8_R8_3PLANE_444_UNORM)
                }
            }
            (ChromaSubsampling::_444, ComponentBitDepth::Bit10) => {
                if semi_planar {
                    Some(VkVideoFormat::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16_EXT)
                } else {
                    Some(VkVideoFormat::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16)
                }
            }
            (ChromaSubsampling::_444, ComponentBitDepth::Bit12) => {
                if semi_planar {
                    Some(VkVideoFormat::G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16_EXT)
                } else {
                    Some(VkVideoFormat::G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16)
                }
            }
        }
    }
}

impl fmt::Display for VideoFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {}bit/{:0>2}bit",
            self.codec.name(),
            self.chroma_subsampling,
            self.luma_bit_depth.bit_depth(),
            self.chroma_bit_depth.bit_depth()
        )
    }
}

/// Vulkan format representation for video.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum VkVideoFormat {
    R8_UNORM = 41,
    R10X6_UNORM_PACK16 = 97,
    R12X4_UNORM_PACK16 = 98,
    G8_B8R8_2PLANE_420_UNORM = 112,
    G8_B8_R8_3PLANE_420_UNORM = 113,
    G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 = 101,
    G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 = 102,
    G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 = 105,
    G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16 = 106,
    G8_B8R8_2PLANE_422_UNORM = 114,
    G8_B8_R8_3PLANE_422_UNORM = 115,
    G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16 = 103,
    G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16 = 104,
    G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16 = 107,
    G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16 = 108,
    G8_B8R8_2PLANE_444_UNORM_EXT = 1004,
    G8_B8_R8_3PLANE_444_UNORM = 116,
    G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16_EXT = 1005,
    G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16 = 1006,
    G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16_EXT = 1007,
    G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16 = 1008,
}

/// H.264 picture layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum H264PictureLayout {
    Progressive = 1,
    Colocated = 2,
    TopField = 4,
    BottomField = 8,
}

impl Default for H264PictureLayout {
    fn default() -> Self {
        Self::Progressive
    }
}

/// Video profile - a complete specification of video capabilities.
///
/// This corresponds to `VkVideoProfileInfoKHR` + codec-specific
/// extension structures.
#[derive(Debug, Clone)]
pub struct VideoProfile {
    /// The codec operation.
    pub operation: super::codec::VideoCodecOperation,
    /// Video format.
    pub format: VideoFormat,
}

impl VideoProfile {
    /// Create a new video profile.
    pub fn new(
        operation: super::codec::VideoCodecOperation,
        format: VideoFormat,
    ) -> Self {
        Self { operation, format }
    }

    /// Check if this profile is valid.
    pub fn is_valid(&self) -> bool {
        !matches!(self.format.codec, super::codec::VideoCodec::None)
    }
}

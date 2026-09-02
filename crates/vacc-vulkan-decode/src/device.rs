//! Vulkan decoder device - implements DecoderDevice trait.

use ash::vk;
use vacc_core::{
    codec::VideoCodec as CoreVideoCodec,
    device::{DecodeCapabilities, DecoderDevice},
    format::{ChromaSubsampling, ComponentBitDepth, VideoFormat},
    session::Extent2D,
};

use super::{Error, Result};

/// Vulkan decoder device that implements the DecoderDevice trait.
pub struct VulkanDecoderDevice {
    inner: vacc_vulkan::VulkanDevice,
}

impl VulkanDecoderDevice {
    /// Create a new Vulkan decoder device using the builder.
    pub fn build(builder: vacc_vulkan::VideoDeviceBuilder) -> Result<Self> {
        let inner = builder.build().map_err(Error::Vulkan)?;
        Ok(Self { inner })
    }

    /// Get a reference to the inner VulkanDevice.
    pub fn as_inner(&self) -> &vacc_vulkan::VulkanDevice {
        &self.inner
    }

    /// Get the Vulkan instance.
    pub fn instance(&self) -> &ash::Instance {
        &self.inner.instance
    }

    /// Get the Vulkan device.
    pub fn device(&self) -> &ash::Device {
        &self.inner.device
    }

    /// Get the physical device.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.inner.physical_device
    }

    /// Get memory properties.
    pub fn memory_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        self.inner.memory_properties
    }

    /// Query video decode capabilities for a given codec profile.
    pub fn query_video_capabilities(
        &self,
        codec: vacc_vulkan::VideoCodec,
        profile_idc: u32,
        chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
        luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
        chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    ) -> Result<vk::VideoCapabilitiesKHR<'_>> {
        self.inner
            .query_video_capabilities(
                codec,
                profile_idc,
                chroma_subsampling,
                luma_bit_depth,
                chroma_bit_depth,
            )
            .map_err(Error::Vulkan)
    }
}

impl DecoderDevice for VulkanDecoderDevice {
    type Error = Error;

    fn backend_name(&self) -> &str {
        "vulkan"
    }

    fn supports_codec(&self, codec: CoreVideoCodec) -> bool {
        let _vk_codec = match codec {
            CoreVideoCodec::DecodeH264 => vacc_vulkan::VideoCodec::DecodeH264,
            CoreVideoCodec::DecodeH265 => vacc_vulkan::VideoCodec::DecodeH265,
            CoreVideoCodec::DecodeAv1 => vacc_vulkan::VideoCodec::DecodeAv1,
            CoreVideoCodec::DecodeVp9 => vacc_vulkan::VideoCodec::DecodeVp9,
            _ => return false,
        };

        // Check if the queue family supports this codec
        let queue_family = match self.inner.queue_families.video_decode {
            Some(qf) => qf,
            None => return false,
        };

        let queue_props = unsafe {
            self.inner
                .instance
                .get_physical_device_queue_family_properties(self.inner.physical_device)
        };

        if queue_family as usize >= queue_props.len() {
            return false;
        }

        let qf = &queue_props[queue_family as usize];
        qf.queue_flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR)
    }

    fn supported_codecs(&self) -> Vec<CoreVideoCodec> {
        let mut codecs = Vec::new();
        for codec in [
            CoreVideoCodec::DecodeH264,
            CoreVideoCodec::DecodeH265,
            CoreVideoCodec::DecodeAv1,
            CoreVideoCodec::DecodeVp9,
        ] {
            if self.supports_codec(codec) {
                codecs.push(codec);
            }
        }
        codecs
    }

    fn query_capabilities(
        &self,
        codec: CoreVideoCodec,
        chroma_subsampling: ChromaSubsampling,
        luma_bit_depth: ComponentBitDepth,
        chroma_bit_depth: ComponentBitDepth,
        profile_idc: Option<u32>,
    ) -> Result<DecodeCapabilities> {
        let vk_codec = match codec {
            CoreVideoCodec::DecodeH264 => vacc_vulkan::VideoCodec::DecodeH264,
            CoreVideoCodec::DecodeH265 => vacc_vulkan::VideoCodec::DecodeH265,
            CoreVideoCodec::DecodeAv1 => vacc_vulkan::VideoCodec::DecodeAv1,
            CoreVideoCodec::DecodeVp9 => vacc_vulkan::VideoCodec::DecodeVp9,
            _ => {
                return Err(Error::Vulkan(vacc_vulkan::VideoError::CodecNotSupported(
                    format!("{:?}", codec),
                )));
            }
        };

        let chroma = match chroma_subsampling {
            ChromaSubsampling::Monochrome => vk::VideoChromaSubsamplingFlagsKHR::MONOCHROME,
            ChromaSubsampling::_420 => vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            ChromaSubsampling::_422 => vk::VideoChromaSubsamplingFlagsKHR::TYPE_422,
            ChromaSubsampling::_444 => vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
        };

        let luma = match luma_bit_depth {
            ComponentBitDepth::Bit8 => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            ComponentBitDepth::Bit10 => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
            ComponentBitDepth::Bit12 => vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
        };

        let chroma_bd = match chroma_bit_depth {
            ComponentBitDepth::Bit8 => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            ComponentBitDepth::Bit10 => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
            ComponentBitDepth::Bit12 => vk::VideoComponentBitDepthFlagsKHR::TYPE_12,
        };

        let profile = profile_idc.unwrap_or(1);

        let caps = self.query_video_capabilities(vk_codec, profile, chroma, luma, chroma_bd)?;

        Ok(DecodeCapabilities {
            codec_operations: codec,
            min_bitstream_buffer_offset_alignment: caps.min_bitstream_buffer_offset_alignment
                as u32,
            min_bitstream_buffer_size_alignment: caps.min_bitstream_buffer_size_alignment as u32,
            picture_access_granularity: Extent2D {
                width: caps.picture_access_granularity.width,
                height: caps.picture_access_granularity.height,
            },
            min_coded_extent: Extent2D {
                width: caps.min_coded_extent.width,
                height: caps.min_coded_extent.height,
            },
            max_coded_extent: Extent2D {
                width: caps.max_coded_extent.width,
                height: caps.max_coded_extent.height,
            },
            max_dpb_slots: caps.max_dpb_slots,
            max_active_reference_pictures: caps.max_active_reference_pictures,
            supported_formats: Vec::new(),
        })
    }

    fn query_supported_formats(&self, codec: CoreVideoCodec) -> Result<Vec<VideoFormat>> {
        let vk_codec = match codec {
            CoreVideoCodec::DecodeH264 => vacc_vulkan::VideoCodec::DecodeH264,
            CoreVideoCodec::DecodeH265 => vacc_vulkan::VideoCodec::DecodeH265,
            CoreVideoCodec::DecodeAv1 => vacc_vulkan::VideoCodec::DecodeAv1,
            CoreVideoCodec::DecodeVp9 => vacc_vulkan::VideoCodec::DecodeVp9,
            _ => {
                return Err(Error::Vulkan(vacc_vulkan::VideoError::CodecNotSupported(
                    format!("{:?}", codec),
                )));
            }
        };

        let formats = self.inner.query_supported_formats(vk_codec);

        let mut result = Vec::new();
        for fmt in formats {
            if fmt.format == vk::Format::G8_B8R8_2PLANE_420_UNORM {
                result.push(VideoFormat::new(
                    codec,
                    ChromaSubsampling::_420,
                    ComponentBitDepth::Bit8,
                    ComponentBitDepth::Bit8,
                ));
            } else if fmt.format == vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 {
                result.push(VideoFormat::new(
                    codec,
                    ChromaSubsampling::_420,
                    ComponentBitDepth::Bit10,
                    ComponentBitDepth::Bit10,
                ));
            } else if fmt.format == vk::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 {
                result.push(VideoFormat::new(
                    codec,
                    ChromaSubsampling::_420,
                    ComponentBitDepth::Bit12,
                    ComponentBitDepth::Bit12,
                ));
            }
        }

        Ok(result)
    }
}

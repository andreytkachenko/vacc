//! Vulkan video session parameters.
//!
//! These correspond to `VkVideoSessionCreateInfoKHR`,
//! `VkVideoBeginCodingInfoKHR`, `VkVideoDecodeInfoKHR`,
//! and `VkVideoPictureResourceInfoKHR`.

/// Video session creation parameters.
///
/// Corresponds to `VkVideoSessionCreateInfoKHR` with codec-specific
/// extension structures.
#[derive(Debug)]
pub struct VideoSessionParams {
    /// Session create flags.
    pub flags: u32,
    /// Video queue family index.
    pub queue_family_index: u32,
    /// Picture format.
    pub picture_format: u32, // VkFormat
    /// Reference pictures format.
    pub reference_picture_format: u32,
    /// Maximum coded extent.
    pub max_coded_extent: Extent2D,
    /// Maximum DPB slots.
    pub max_dpb_slots: u32,
    /// Maximum active reference pictures.
    pub max_active_reference_pictures: u32,
    /// Codec-specific create info (opaque).
    pub codec_create_info: Option<Box<dyn CodecCreateInfo>>,
}

impl VideoSessionParams {
    pub fn new(
        queue_family_index: u32,
        picture_format: u32,
        reference_picture_format: u32,
        max_coded_extent: Extent2D,
        max_dpb_slots: u32,
        max_active_reference_pictures: u32,
    ) -> Self {
        Self {
            flags: 0,
            queue_family_index,
            picture_format,
            reference_picture_format,
            max_coded_extent,
            max_dpb_slots,
            max_active_reference_pictures,
            codec_create_info: None,
        }
    }
}

/// Codec-specific session create info trait.
pub trait CodecCreateInfo: std::fmt::Debug {
    fn structure_type(&self) -> ash::vk::StructureType;
}

/// Video coded extent.
#[derive(Debug, Clone, Copy, Default)]
pub struct Extent2D {
    pub width: u32,
    pub height: u32,
}

impl Extent2D {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

/// Video begin coding info.
///
/// Corresponds to `VkVideoBeginCodingInfoKHR`.
#[derive(Debug, Clone)]
pub struct VideoBeginCodingInfo {
    /// Queue handle (opaque).
    pub queue: u64,
    /// Coding mode (forward or backward).
    pub coding_mode: VideoCodingModeKHR,
}

/// Video coding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum VideoCodingModeKHR {
    Forward = 0,
    Backward = 1,
}

/// Video decode info.
///
/// Corresponds to `VkVideoDecodeInfoKHR`.
#[derive(Debug, Clone)]
pub struct VideoDecodeInfo {
    /// Flags.
    pub flags: u32,
    /// Reference picture layout.
    pub reference_picture_layout: ash::vk::ImageLayout,
    /// POC (Picture Order Count) slot.
    pub poc: i32,
    /// Bitstream buffer.
    pub bitstream_buffer: BitstreamBufferView,
    /// DPB (Decoded Picture Buffer) setup picture resource.
    pub dpb_setup_picture: Option<PictureResourceInfo>,
    /// DPB reference picture resources.
    pub dpb_ref_picture: Vec<PictureResourceInfo>,
}

impl VideoDecodeInfo {
    pub fn new(
        flags: u32,
        reference_picture_layout: ash::vk::ImageLayout,
        poc: i32,
        bitstream_view: BitstreamBufferView,
    ) -> Self {
        Self {
            flags,
            reference_picture_layout,
            poc,
            bitstream_buffer: bitstream_view,
            dpb_setup_picture: None,
            dpb_ref_picture: Vec::new(),
        }
    }
}

/// Bitstream buffer view.
///
/// Corresponds to `VkBufferView`.
#[derive(Debug, Clone, Copy, Default)]
pub struct BitstreamBufferView {
    /// Buffer handle (opaque).
    pub buffer: u64,
    /// Offset in the buffer.
    pub offset: u64,
    /// Range/size.
    pub range: u64,
}

impl BitstreamBufferView {
    pub const fn new(buffer: u64, offset: u64, range: u64) -> Self {
        Self {
            buffer,
            offset,
            range,
        }
    }

    pub const fn empty() -> Self {
        Self {
            buffer: 0,
            offset: 0,
            range: 0,
        }
    }
}

/// Picture resource info.
///
/// Corresponds to `VkVideoPictureResourceInfoKHR`.
#[derive(Debug, Clone, Copy)]
pub struct PictureResourceInfo {
    /// Combined layout and aspect.
    pub combined_layout: ash::vk::ImageLayout,
    /// Image view handle (opaque).
    pub image_view: u64,
    /// Subresource range.
    pub subresource_range: ImageSubresourceRange,
}

impl PictureResourceInfo {
    pub const fn new(
        combined_layout: ash::vk::ImageLayout,
        image_view: u64,
        subresource_range: ImageSubresourceRange,
    ) -> Self {
        Self {
            combined_layout,
            image_view,
            subresource_range,
        }
    }
}

/// Image subresource range.
///
/// Corresponds to `VkImageSubresourceRange`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageSubresourceRange {
    /// Aspect mask.
    pub aspect_mask: u32,
    /// Mip level.
    pub base_mip_level: u32,
    /// Level count.
    pub level_count: u32,
    /// Array layer.
    pub base_array_layer: u32,
    /// Layer count.
    pub layer_count: u32,
}

impl ImageSubresourceRange {
    pub const fn new(aspect_mask: u32) -> Self {
        Self {
            aspect_mask,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
}

/// Bitstream buffer descriptor.
///
/// Used to allocate and manage bitstream buffers.
#[derive(Debug, Clone)]
pub struct BitstreamBufferDescriptor {
    /// Minimum buffer offset alignment.
    pub min_bitstream_buffer_offset_alignment: u32,
    /// Minimum buffer size alignment.
    pub min_bitstream_buffer_size_alignment: u32,
    /// Desired buffer size.
    pub size: u64,
}

impl BitstreamBufferDescriptor {
    pub fn new(min_offset_alignment: u32, min_size_alignment: u32, size: u64) -> Self {
        Self {
            min_bitstream_buffer_offset_alignment: min_offset_alignment,
            min_bitstream_buffer_size_alignment: min_size_alignment,
            size,
        }
    }

    /// Calculate aligned size.
    pub fn aligned_size(&self) -> u64 {
        let align = self.min_bitstream_buffer_size_alignment as u64;
        (self.size + align - 1) & !(align - 1)
    }

    /// Calculate aligned offset.
    pub fn aligned_offset(&self, offset: u64) -> u64 {
        let align = self.min_bitstream_buffer_offset_alignment as u64;
        (offset + align - 1) & !(align - 1)
    }
}

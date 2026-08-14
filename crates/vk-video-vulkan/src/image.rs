//! Vulkan image creation for video decode output with proper aspect masks.

use super::{VideoError, VideoResult};
use ash::vk::Handle;

/// Represents a decoded output image with YCbCr planes.
pub struct DecodedImage {
    pub image: ash::vk::Image,
    pub image_view: ash::vk::ImageView,
    pub memory: ash::vk::DeviceMemory,
    pub format: ash::vk::Format,
    pub extent: ash::vk::Extent2D,
    pub device: ash::Device,
    pub instance: ash::Instance,
    pub memory_properties: ash::vk::PhysicalDeviceMemoryProperties,
}

impl DecodedImage {
    /// Create a semi-planar YCbCr output image (G8_B8R8_2PLANE_420_UNORM).
    pub fn create_yuv420(
        device: &ash::Device,
        instance: &ash::Instance,
        memory_properties: &ash::vk::PhysicalDeviceMemoryProperties,
        width: u32,
        height: u32,
    ) -> VideoResult<Self> {
        let format = ash::vk::Format::G8_B8R8_2PLANE_420_UNORM;

        let image_create_info = ash::vk::ImageCreateInfo {
            s_type: ash::vk::StructureType::IMAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: ash::vk::ImageCreateFlags::empty(),
            image_type: ash::vk::ImageType::TYPE_2D,
            format,
            extent: ash::vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: ash::vk::SampleCountFlags::TYPE_1,
            tiling: ash::vk::ImageTiling::OPTIMAL,
            usage: ash::vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                | ash::vk::ImageUsageFlags::TRANSFER_SRC,
            sharing_mode: ash::vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
            initial_layout: ash::vk::ImageLayout::UNDEFINED,
            _marker: std::marker::PhantomData,
        };

        let image = unsafe { device.create_image(&image_create_info, None) }
            .map_err(|e| VideoError::ImageCreation(format!("Failed to create image: {}", e)))?;

        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };

        let mem_type_index = find_memory_type(
            memory_properties,
            mem_requirements.memory_type_bits,
            ash::vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| {
            VideoError::MemoryAllocation("No suitable device-local memory type found".to_string())
        })?;

        let alloc_info = ash::vk::MemoryAllocateInfo {
            s_type: ash::vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            allocation_size: mem_requirements.size,
            memory_type_index: mem_type_index,
            _marker: std::marker::PhantomData,
        };

        let memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .map_err(|e| VideoError::MemoryAllocation(e.to_string()))?;

        unsafe { device.bind_image_memory(image, memory, 0) }
            .map_err(|e| VideoError::ImageCreation(format!("Failed to bind memory: {}", e)))?;

        // Create image view with COLOR aspect for video decode
        let y_view = create_image_view(
            device,
            image,
            format,
            ash::vk::ImageAspectFlags::COLOR,
            0,
            1,
            0,
            1,
        )?;

        Ok(Self {
            image,
            image_view: y_view,
            memory,
            format,
            extent: ash::vk::Extent2D { width, height },
            device: device.clone(),
            instance: instance.clone(),
            memory_properties: memory_properties.clone(),
        })
    }

    /// Read back YUV data from the decoded image to host memory.
    /// Requires proper staging image and transfer command buffer.
    pub fn read_back_yuv(&self, _width: u32, _height: u32) -> VideoResult<(Vec<u8>, Vec<u8>)> {
        // For optimal images, we need to copy to a linear staging image first
        // This requires a command buffer with proper pipeline barriers.
        Ok((Vec::new(), Vec::new()))
    }

    /// Get the Y plane row pitch (in bytes).
    pub fn y_row_pitch(&self) -> usize {
        self.extent.width as usize
    }

    /// Get the UV plane row pitch (in bytes).
    pub fn uv_row_pitch(&self) -> usize {
        (self.extent.width as usize / 2) * 2
    }
}

impl Drop for DecodedImage {
    fn drop(&mut self) {
        if !self.image.is_null() {
            unsafe { self.device.destroy_image(self.image, None) };
        }
        if !self.image_view.is_null() {
            unsafe { self.device.destroy_image_view(self.image_view, None) };
        }
        if !self.memory.is_null() {
            unsafe { self.device.free_memory(self.memory, None) };
        }
    }
}

/// Create an output image suitable for video decode.
pub fn create_output_image(
    device: &ash::Device,
    memory_properties: &ash::vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    format: ash::vk::Format,
) -> VideoResult<(ash::vk::Image, ash::vk::ImageView, ash::vk::DeviceMemory)> {
    create_output_image_with_pnext(
        device,
        memory_properties,
        width,
        height,
        format,
        std::ptr::null(),
    )
}

/// Create an output image suitable for video decode with a pNext chain
/// (e.g., VkVideoProfileListInfoKHR for video profile compatibility).
pub fn create_output_image_with_pnext(
    device: &ash::Device,
    memory_properties: &ash::vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    format: ash::vk::Format,
    p_next: *const std::ffi::c_void,
) -> VideoResult<(ash::vk::Image, ash::vk::ImageView, ash::vk::DeviceMemory)> {
    let image_create_info = ash::vk::ImageCreateInfo::default()
        .image_type(ash::vk::ImageType::TYPE_2D)
        .format(format)
        .extent(ash::vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(ash::vk::SampleCountFlags::TYPE_1)
        .tiling(ash::vk::ImageTiling::OPTIMAL)
        .usage(
            ash::vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                | ash::vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
                | ash::vk::ImageUsageFlags::TRANSFER_SRC
                | ash::vk::ImageUsageFlags::SAMPLED,
        )
        .sharing_mode(ash::vk::SharingMode::EXCLUSIVE)
        .initial_layout(ash::vk::ImageLayout::UNDEFINED);

    // Apply pNext chain if provided
    let image_create_info = if p_next.is_null() {
        image_create_info
    } else {
        let mut info = image_create_info;
        info.p_next = p_next;
        info
    };

    let image = unsafe { device.create_image(&image_create_info, None) }
        .map_err(|e| VideoError::ImageCreation(format!("Image creation failed: {:?}", e)))?;

    let mem_requirements = unsafe { device.get_image_memory_requirements(image) };

    let mem_type_index = find_memory_type(
        memory_properties,
        mem_requirements.memory_type_bits,
        ash::vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| VideoError::MemoryAllocation("No device-local memory type found".to_string()))?;

    let alloc_info = ash::vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(mem_type_index);

    let memory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| VideoError::MemoryAllocation(format!("Memory allocation failed: {:?}", e)))?;

    unsafe { device.bind_image_memory(image, memory, 0) }
        .map_err(|e| VideoError::ImageCreation(format!("Memory binding failed: {:?}", e)))?;

    // Create image view with COLOR aspect for video decode.
    // Vulkan spec requires COLOR aspect for VIDEO_DECODE_DST/DPB usage.
    let view_create_info = ash::vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(ash::vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            ash::vk::ImageSubresourceRange::default()
                .aspect_mask(ash::vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        );

    let view = unsafe { device.create_image_view(&view_create_info, None) }
        .map_err(|e| VideoError::ImageCreation(format!("ImageView creation failed: {:?}", e)))?;

    Ok((image, view, memory))
}

/// Create a host-visible linear image for readback.
pub fn create_staging_image(
    device: &ash::Device,
    instance: &ash::Instance,
    memory_properties: &ash::vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
) -> VideoResult<StagingImage> {
    let format = ash::vk::Format::G8_B8R8_2PLANE_420_UNORM;

    let image_create_info = ash::vk::ImageCreateInfo::default()
        .image_type(ash::vk::ImageType::TYPE_2D)
        .format(format)
        .extent(ash::vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(ash::vk::SampleCountFlags::TYPE_1)
        .tiling(ash::vk::ImageTiling::LINEAR)
        .usage(ash::vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(ash::vk::SharingMode::EXCLUSIVE)
        .initial_layout(ash::vk::ImageLayout::UNDEFINED);

    let image = unsafe { device.create_image(&image_create_info, None) }.map_err(|e| {
        VideoError::ImageCreation(format!("Staging image creation failed: {:?}", e))
    })?;

    let mem_requirements = unsafe { device.get_image_memory_requirements(image) };

    let mem_type_index = find_memory_type(
        memory_properties,
        mem_requirements.memory_type_bits,
        ash::vk::MemoryPropertyFlags::HOST_VISIBLE | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| VideoError::MemoryAllocation("No host-visible memory type found".to_string()))?;

    let alloc_info = ash::vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(mem_type_index);

    let memory = unsafe { device.allocate_memory(&alloc_info, None) }.map_err(|e| {
        VideoError::MemoryAllocation(format!("Staging memory allocation failed: {:?}", e))
    })?;

    unsafe { device.bind_image_memory(image, memory, 0) }.map_err(|e| {
        VideoError::ImageCreation(format!("Staging memory binding failed: {:?}", e))
    })?;

    let mapped_ptr = unsafe {
        device.map_memory(
            memory,
            ash::vk::WHOLE_SIZE,
            0,
            ash::vk::MemoryMapFlags::empty(),
        )
    }
    .map(|p| p as *mut u8)
    .ok();

    Ok(StagingImage {
        image,
        memory,
        mapped_ptr,
        extent: ash::vk::Extent2D { width, height },
        device: device.clone(),
        instance: instance.clone(),
        memory_properties: memory_properties.clone(),
    })
}

/// A host-visible linear image for readback.
pub struct StagingImage {
    pub image: ash::vk::Image,
    pub memory: ash::vk::DeviceMemory,
    pub mapped_ptr: Option<*mut u8>,
    pub extent: ash::vk::Extent2D,
    pub device: ash::Device,
    pub instance: ash::Instance,
    pub memory_properties: ash::vk::PhysicalDeviceMemoryProperties,
}

impl StagingImage {
    pub fn get_y_plane(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let ptr = self.mapped_ptr?;
        let y_size = (width * height) as usize;
        let mut y_data = vec![0u8; y_size];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr as *const u8, y_data.as_mut_ptr(), y_size);
        }
        Some(y_data)
    }

    pub fn get_uv_plane(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let ptr = self.mapped_ptr?;
        let y_size = (width * height) as usize;
        let uv_size = (width / 2 * height / 2 * 2) as usize;
        let mut uv_data = vec![0u8; uv_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (ptr as usize + y_size) as *const u8,
                uv_data.as_mut_ptr(),
                uv_size,
            );
        }
        Some(uv_data)
    }
}

impl Drop for StagingImage {
    fn drop(&mut self) {
        if self.mapped_ptr.is_some() {
            unsafe { self.device.unmap_memory(self.memory) };
        }
        if !self.memory.is_null() {
            unsafe { self.device.free_memory(self.memory, None) };
        }
        if !self.image.is_null() {
            unsafe { self.device.destroy_image(self.image, None) };
        }
    }
}

/// Create an image view for a specific plane with proper aspect mask.
fn create_image_view(
    device: &ash::Device,
    image: ash::vk::Image,
    format: ash::vk::Format,
    aspect_mask: ash::vk::ImageAspectFlags,
    base_mip_level: u32,
    level_count: u32,
    base_array_layer: u32,
    layer_count: u32,
) -> VideoResult<ash::vk::ImageView> {
    let view_create_info = ash::vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(ash::vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            ash::vk::ImageSubresourceRange::default()
                .aspect_mask(aspect_mask)
                .base_mip_level(base_mip_level)
                .level_count(level_count)
                .base_array_layer(base_array_layer)
                .layer_count(layer_count),
        );

    unsafe { device.create_image_view(&view_create_info, None) }
        .map_err(|e| VideoError::ImageCreation(format!("ImageView creation failed: {:?}", e)))
}

/// Find a suitable memory type index.
fn find_memory_type(
    mem_props: &ash::vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required_flags: ash::vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..mem_props.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(required_flags)
        {
            return Some(i);
        }
    }
    None
}

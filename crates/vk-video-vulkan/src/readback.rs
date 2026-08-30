//! Pixel readback from decoded video images.

use super::VideoError;
use ash::vk::{self};

/// Decoded pixel data for YUV 420 planar format.
#[derive(Debug, Clone)]
pub struct DecodedPixels {
    pub y_plane: Vec<u8>,
    pub u_plane: Vec<u8>,
    pub v_plane: Vec<u8>,
}

/// Readback decoded image pixels from GPU to CPU (8-bit NV12 source).
pub fn readback_decoded_image(
    instance: &ash::Instance,
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    queue_family: u32,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    image: vk::Image,
    base_array_layer: u32,
    width: u32,
    height: u32,
    old_layout: vk::ImageLayout,
) -> Result<DecodedPixels, VideoError> {
    readback_decoded_image_format(
        instance,
        device,
        memory_properties,
        queue_family,
        command_pool,
        fence,
        image,
        base_array_layer,
        width,
        height,
        old_layout,
        vk::Format::G8_B8R8_2PLANE_420_UNORM,
    )
}

/// Source format descriptor for readback.
#[derive(Clone, Copy)]
enum HdrSource {
    /// 8-bit NV12: plane0 = 1 byte/px, plane1 = interleaved 1 byte U/V.
    B8,
    /// G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 (or the 12-bit equivalent):
    /// plane0 = 2 bytes/px, plane1 = 4 bytes per (U,V) pair. Each sample is a
    /// u16 with the `bits`-bit value in the HIGH bits (`value << (16 - bits)`,
    /// i.e. G10X6 = 10 bits + 6 pad, G12X4 = 12 bits + 4 pad).
    B16 { bits: u32 },
}

fn hdr_source_for_format(format: vk::Format) -> Option<HdrSource> {
    match format {
        vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR => Some(HdrSource::B16 { bits: 10 }),
        vk::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR => Some(HdrSource::B16 { bits: 12 }),
        _ => None,
    }
}

/// Readback decoded image pixels from GPU to CPU.
///
/// `source_format` must match the format of `image`. 8-bit NV12 is returned
/// as-is; 10/12-bit sources are down-converted to 8-bit (rounded).
pub fn readback_decoded_image_format(
    instance: &ash::Instance,
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    queue_family: u32,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    image: vk::Image,
    base_array_layer: u32,
    width: u32,
    height: u32,
    old_layout: vk::ImageLayout,
    source_format: vk::Format,
) -> Result<DecodedPixels, VideoError> {
    let uv_width = width.div_ceil(2);
    let uv_height = height.div_ceil(2);

    // Plane sizes in bytes depend on the source format.
    let (y_size, uv_size) = match hdr_source_for_format(source_format) {
        Some(HdrSource::B16 { .. }) => {
            let y = (width * height * 2) as usize;
            let uv = (uv_width * uv_height * 4) as usize;
            (y, uv)
        }
        _ => {
            let y = (width * height) as usize;
            let uv = (uv_width * uv_height * 2) as usize;
            (y, uv)
        }
    };
    let total_size = (y_size + uv_size) as u64;

    let buffer_create_info = vk::BufferCreateInfo::default()
        .size(total_size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST);

    let buffer = unsafe {
        device
            .create_buffer(&buffer_create_info, None)
            .map_err(|e| VideoError::BufferAllocation(e.to_string()))?
    };

    let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_type_index = find_memory_type(
        memory_properties,
        mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .or_else(|| {
        find_memory_type(
            memory_properties,
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
    })
    .ok_or_else(|| {
        VideoError::MemoryAllocation("No suitable memory type for staging buffer".to_string())
    })?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(mem_type_index);

    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| VideoError::MemoryAllocation(e.to_string()))?
    };

    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| VideoError::BufferAllocation(e.to_string()))?;
    }

    let mapped_ptr = unsafe {
        device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .map_err(|e| VideoError::Io(std::io::Error::other(e.to_string())))?
    };

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd_buffers = unsafe {
        device
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| VideoError::CommandBufferRecording(e.to_string()))?
    };
    let cmd_buffer = cmd_buffers[0];

    unsafe {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(cmd_buffer, &begin_info)
            .map_err(|e| VideoError::CommandBufferRecording(e.to_string()))?;

        let plane0_barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            src_access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
            dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            dst_access_mask: vk::AccessFlags2::TRANSFER_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            old_layout,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::PLANE_0,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer,
                layer_count: 1,
            },
            _marker: Default::default(),
        };

        let plane1_barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            src_access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
            dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            dst_access_mask: vk::AccessFlags2::TRANSFER_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            old_layout,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::PLANE_1,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer,
                layer_count: 1,
            },
            _marker: Default::default(),
        };

        let image_barriers = [plane0_barrier, plane1_barrier];
        let dep_info = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::BY_REGION,
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: std::ptr::null(),
            image_memory_barrier_count: image_barriers.len() as u32,
            p_image_memory_barriers: image_barriers.as_ptr(),
            _marker: Default::default(),
        };
        cmd_pipeline_barrier_2(instance, device.handle(), cmd_buffer, &dep_info);

        device.cmd_copy_image_to_buffer(
            cmd_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            &[vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                        .mip_level(0)
                        .base_array_layer(base_array_layer)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D::default())
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })],
        );

        device.cmd_copy_image_to_buffer(
            cmd_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            &[vk::BufferImageCopy::default()
                .buffer_offset(y_size as u64)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                        .mip_level(0)
                        .base_array_layer(base_array_layer)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D::default())
                .image_extent(vk::Extent3D {
                    width: uv_width,
                    height: uv_height,
                    depth: 1,
                })],
        );

        let buffer_barrier = vk::BufferMemoryBarrier2 {
            s_type: vk::StructureType::BUFFER_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::HOST,
            dst_access_mask: vk::AccessFlags2::HOST_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer,
            offset: 0,
            size: vk::WHOLE_SIZE,
            _marker: Default::default(),
        };

        let buffer_barriers = [buffer_barrier];
        let dep_info = vk::DependencyInfo::default().buffer_memory_barriers(&buffer_barriers);
        cmd_pipeline_barrier_2(instance, device.handle(), cmd_buffer, &dep_info);

        let plane0_restore = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            src_access_mask: vk::AccessFlags2::TRANSFER_READ,
            dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::PLANE_0,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer,
                layer_count: 1,
            },
            _marker: Default::default(),
        };

        let plane1_restore = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            src_access_mask: vk::AccessFlags2::TRANSFER_READ,
            dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
            dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::PLANE_1,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer,
                layer_count: 1,
            },
            _marker: Default::default(),
        };

        let restore_barriers = [plane0_restore, plane1_restore];
        let restore_dep_info = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::BY_REGION,
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: std::ptr::null(),
            image_memory_barrier_count: restore_barriers.len() as u32,
            p_image_memory_barriers: restore_barriers.as_ptr(),
            _marker: Default::default(),
        };
        cmd_pipeline_barrier_2(instance, device.handle(), cmd_buffer, &restore_dep_info);

        device
            .end_command_buffer(cmd_buffer)
            .map_err(|e| VideoError::CommandBufferRecording(e.to_string()))?;

        device
            .reset_fences(&[fence])
            .map_err(|e| VideoError::FenceWait(e.to_string()))?;

        device
            .queue_submit(
                device.get_device_queue(queue_family, 0),
                &[vk::SubmitInfo::default().command_buffers(&[cmd_buffer])],
                fence,
            )
            .map_err(|e| VideoError::QueueSubmission(e.to_string()))?;

        let result = device.wait_for_fences(&[fence], true, 10_000_000_000);
        if let Err(e) = result {
            return Err(VideoError::FenceWait(e.to_string()));
        }

        let mut y_plane = vec![0u8; (width * height) as usize];
        let uv_plane_size = (uv_width * uv_height) as usize;
        let mut u_plane = vec![0u8; uv_plane_size];
        let mut v_plane = vec![0u8; uv_plane_size];

        match hdr_source_for_format(source_format) {
            Some(HdrSource::B16 { bits }) => {
                // Plane 0: one u16 per luma sample (value in low `bits` bits).
                let src = mapped_ptr as *const u16;
                for i in 0..(width * height) as usize {
                    let v = unsafe { *src.add(i) };
                    y_plane[i] = scale_to_8(v, bits);
                }
                // Plane 1: interleaved u16 U, u16 V pairs.
                let src = (mapped_ptr.add(y_size as usize)) as *const u16;
                for i in 0..uv_plane_size {
                    u_plane[i] = scale_to_8(unsafe { *src.add(i * 2) }, bits);
                    v_plane[i] = scale_to_8(unsafe { *src.add(i * 2 + 1) }, bits);
                }
            }
            _ => {
                // 8-bit NV12: copy, then deinterleave UV.
                std::ptr::copy_nonoverlapping(
                    mapped_ptr as *const u8,
                    y_plane.as_mut_ptr(),
                    y_size,
                );
                let mut uv_plane = vec![0u8; uv_size];
                std::ptr::copy_nonoverlapping(
                    mapped_ptr.add(y_size) as *const u8,
                    uv_plane.as_mut_ptr(),
                    uv_size,
                );
                for i in 0..uv_plane_size {
                    u_plane[i] = uv_plane[i * 2];
                    v_plane[i] = uv_plane[i * 2 + 1];
                }
            }
        }

        device.unmap_memory(memory);
        device.free_memory(memory, None);
        device.destroy_buffer(buffer, None);

        Ok(DecodedPixels {
            y_plane,
            u_plane,
            v_plane,
        })
    }
}

/// Scale a `bits`-bit sample stored in the HIGH bits of a u16 (G10X6/G12X4
/// packing: `value << (16 - bits)`) to 8-bit, rounded.
#[inline]
fn scale_to_8(v: u16, bits: u32) -> u8 {
    let x = (v as u32) >> (16 - bits);
    // For the max sample value the rounded result is 256; saturate instead of
    // wrapping to 0 via the u8 cast.
    (((x << 8) + (1u32 << (bits - 1))) >> bits).min(255) as u8
}

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..mem_props.memory_type_count).find(|&i| {
        (type_bits & (1 << i)) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(required_flags)
    })
}

fn cmd_pipeline_barrier_2(
    instance: &ash::Instance,
    device: vk::Device,
    cmd_buffer: vk::CommandBuffer,
    dep_info: &vk::DependencyInfo<'_>,
) {
    let fn_ptr =
        unsafe { instance.get_device_proc_addr(device, c"vkCmdPipelineBarrier2KHR".as_ptr()) };
    if let Some(ptr) = fn_ptr {
        unsafe {
            type FnType =
                unsafe extern "system" fn(vk::CommandBuffer, *const vk::DependencyInfo<'_>);
            let f: FnType = std::mem::transmute(ptr);
            f(cmd_buffer, dep_info);
        }
    }
}

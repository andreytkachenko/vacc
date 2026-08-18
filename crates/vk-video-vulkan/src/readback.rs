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

/// Readback decoded image pixels from GPU to CPU.
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
) -> Result<DecodedPixels, VideoError> {
    let y_size = (width * height) as usize;
    let uv_width = width.div_ceil(2);
    let uv_height = height.div_ceil(2);
    let uv_size = (uv_width * uv_height * 2) as usize;
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
            .map_err(|e| {
                VideoError::Io(std::io::Error::other(
                    e.to_string(),
                ))
            })?
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
            old_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
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
            old_layout: vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
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

        let mut y_plane = vec![0u8; y_size];
        let mut uv_plane = vec![0u8; uv_size];

        std::ptr::copy_nonoverlapping(mapped_ptr as *const u8, y_plane.as_mut_ptr(), y_size);
        std::ptr::copy_nonoverlapping(
            mapped_ptr.add(y_size) as *const u8,
            uv_plane.as_mut_ptr(),
            uv_size,
        );

        let uv_plane_size = (uv_width * uv_height) as usize;
        let mut u_plane = vec![0u8; uv_plane_size];
        let mut v_plane = vec![0u8; uv_plane_size];
        for i in 0..uv_plane_size {
            u_plane[i] = uv_plane[i * 2];
            v_plane[i] = uv_plane[i * 2 + 1];
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

/// Readback variant that copies the DPB slot to a SEPARATE staging image first,
/// then reads the staging image to a buffer. The DPB slot itself is NEVER
/// transitioned out of VIDEO_DECODE_DPB_KHR, so it cannot be corrupted by a
/// layout transition. Used to test whether the direct readback's
/// VIDEO_DECODE_DPB_KHR -> TRANSFER_SRC_OPTIMAL -> back transition corrupts the
/// reference data left in the DPB for future frames.
pub fn readback_decoded_image_via_staging(
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
) -> Result<DecodedPixels, VideoError> {
    let y_size = (width * height) as usize;
    let uv_width = width.div_ceil(2);
    let uv_height = height.div_ceil(2);
    let uv_size = (uv_width * uv_height * 2) as usize;
    let total_size = (y_size + uv_size) as u64;

    // --- Staging image (same format as DPB, single layer) ---
    let staging_format = vk::Format::G8_B8R8_2PLANE_420_UNORM;
    let staging_image = unsafe {
        device
            .create_image(
                &vk::ImageCreateInfo {
                    image_type: vk::ImageType::TYPE_2D,
                    format: staging_format,
                    extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                    initial_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    queue_family_index_count: 1,
                    p_queue_family_indices: &queue_family as *const u32,
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| VideoError::ImageCreation(e.to_string()))?
    };
    let smem_reqs = unsafe { device.get_image_memory_requirements(staging_image) };
    let smem_type = find_memory_type(
        memory_properties,
        smem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| {
        VideoError::MemoryAllocation("No suitable memory type for staging image".to_string())
    })?;
    let smem = unsafe {
        device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(smem_reqs.size)
                    .memory_type_index(smem_type),
                None,
            )
            .map_err(|e| VideoError::MemoryAllocation(e.to_string()))?
    };
    unsafe {
        device
            .bind_image_memory(staging_image, smem, 0)
            .map_err(|e| VideoError::MemoryAllocation(e.to_string()))?;
    }

    // --- Staging buffer ---
    let buffer = unsafe {
        device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(total_size)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
            .map_err(|e| VideoError::BufferAllocation(e.to_string()))?
    };
    let bmem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let bmem_type = find_memory_type(
        memory_properties,
        bmem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .or_else(|| {
        find_memory_type(
            memory_properties,
            bmem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
    })
    .ok_or_else(|| {
        VideoError::MemoryAllocation("No suitable memory type for staging buffer".to_string())
    })?;
    let bmem = unsafe {
        device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(bmem_reqs.size)
                    .memory_type_index(bmem_type),
                None,
            )
            .map_err(|e| VideoError::MemoryAllocation(e.to_string()))?
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, bmem, 0)
            .map_err(|e| VideoError::BufferAllocation(e.to_string()))?;
    }
    let mapped_ptr = unsafe {
        device
            .map_memory(bmem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .map_err(|e| VideoError::Io(std::io::Error::other(e.to_string())))?
    };

    // --- Command buffer ---
    let cmd_buffers = unsafe {
        device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| VideoError::CommandBufferRecording(e.to_string()))?
    };
    let cmd_buffer = cmd_buffers[0];

    unsafe {
        device
            .begin_command_buffer(
                cmd_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| VideoError::CommandBufferRecording(e.to_string()))?;

        // 1) Image-to-image copy: DPB slot (VIDEO_DECODE_DPB_KHR) -> staging
        //    (TRANSFER_DST_OPTIMAL). The DPB slot is NOT transitioned.
        let copy_regions = [
            vk::ImageCopy::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                        .mip_level(0)
                        .base_array_layer(base_array_layer)
                        .layer_count(1),
                )
                .src_offset(vk::Offset3D::default())
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_offset(vk::Offset3D::default())
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                }),
            vk::ImageCopy::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                        .mip_level(0)
                        .base_array_layer(base_array_layer)
                        .layer_count(1),
                )
                .src_offset(vk::Offset3D::default())
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_offset(vk::Offset3D::default())
                .extent(vk::Extent3D {
                    width: uv_width,
                    height: uv_height,
                    depth: 1,
                }),
        ];
        device.cmd_copy_image(
            cmd_buffer,
            image,
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            staging_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &copy_regions,
        );

        // 2) Transition staging -> TRANSFER_SRC_OPTIMAL.
        let staging_barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            dst_access_mask: vk::AccessFlags2::TRANSFER_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: staging_image,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            _marker: Default::default(),
        };
        let dep_info = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::BY_REGION,
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: std::ptr::null(),
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &staging_barrier,
            _marker: Default::default(),
        };
        cmd_pipeline_barrier_2(instance, device.handle(), cmd_buffer, &dep_info);

        // 3) Image-to-buffer copy: staging -> buffer.
        device.cmd_copy_image_to_buffer(
            cmd_buffer,
            staging_image,
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
                        .base_array_layer(0)
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
            staging_image,
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
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D::default())
                .image_extent(vk::Extent3D {
                    width: uv_width,
                    height: uv_height,
                    depth: 1,
                })],
        );

        // 4) Buffer barrier: TRANSFER -> HOST.
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

        let mut y_plane = vec![0u8; y_size];
        let mut uv_plane = vec![0u8; uv_size];
        std::ptr::copy_nonoverlapping(mapped_ptr as *const u8, y_plane.as_mut_ptr(), y_size);
        std::ptr::copy_nonoverlapping(
            mapped_ptr.add(y_size) as *const u8,
            uv_plane.as_mut_ptr(),
            uv_size,
        );
        let uv_plane_size = (uv_width * uv_height) as usize;
        let mut u_plane = vec![0u8; uv_plane_size];
        let mut v_plane = vec![0u8; uv_plane_size];
        for i in 0..uv_plane_size {
            u_plane[i] = uv_plane[i * 2];
            v_plane[i] = uv_plane[i * 2 + 1];
        }

        device.unmap_memory(bmem);
        device.free_memory(bmem, None);
        device.destroy_buffer(buffer, None);
        device.destroy_image(staging_image, None);
        device.free_memory(smem, None);

        Ok(DecodedPixels {
            y_plane,
            u_plane,
            v_plane,
        })
    }
}

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..mem_props.memory_type_count).find(|&i| (type_bits & (1 << i)) != 0
        && mem_props.memory_types[i as usize]
            .property_flags
            .contains(required_flags))
}

fn cmd_pipeline_barrier_2(
    instance: &ash::Instance,
    device: vk::Device,
    cmd_buffer: vk::CommandBuffer,
    dep_info: &vk::DependencyInfo<'_>,
) {
    let fn_ptr = unsafe {
        instance.get_device_proc_addr(device, c"vkCmdPipelineBarrier2KHR".as_ptr())
    };
    if let Some(ptr) = fn_ptr {
        unsafe {
            type FnType =
                unsafe extern "system" fn(vk::CommandBuffer, *const vk::DependencyInfo<'_>);
            let f: FnType = std::mem::transmute(ptr);
            f(cmd_buffer, dep_info);
        }
    }
}

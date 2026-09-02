//! Pixel readback from decoded video images.

use super::VideoError;
use ash::vk::{self};

/// Decoded pixel data for a YUV image.
///
/// Planes are stored as raw bytes, `sample_size` bytes per sample, row-major.
/// `y_plane` is coded_width x coded_height samples; `u_plane`/`v_plane` are
/// chroma_width x chroma_height samples (full size for 4:4:4 and mono).
#[derive(Debug, Clone)]
pub struct DecodedPixels {
    pub y_plane: Vec<u8>,
    pub u_plane: Vec<u8>,
    pub v_plane: Vec<u8>,
    /// Bytes per sample (1 or 2).
    pub sample_size: u32,
    /// Chroma plane width in samples.
    pub chroma_width: u32,
    /// Chroma plane height in samples.
    pub chroma_height: u32,
}

enum Layout {
    /// Plane 0 = luma, plane 1 = interleaved Cb/Cr (NV12/P010/P016/422).
    TwoPlane { uv_width: u32, uv_height: u32 },
    /// Three full-resolution planes: 0=G, 1=B(Cb), 2=R(Cr) (3PLANE 444).
    ThreePlane,
    /// Single interleaved RGBA-style plane, one sample per channel.
    Packed4 { channels: u32 },
    /// Single luma-only plane.
    Mono,
}

/// Classify the decode output format into a readback layout.
fn classify_format(format: vk::Format, width: u32, height: u32) -> Option<(u32, u32, u32, Layout)> {
    let (sample_size, chroma_w, chroma_h, layout) = match format {
        vk::Format::G8_B8R8_2PLANE_420_UNORM => (
            1,
            width.div_ceil(2),
            height.div_ceil(2),
            Layout::TwoPlane {
                uv_width: width.div_ceil(2),
                uv_height: height.div_ceil(2),
            },
        ),
        vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
        | vk::Format::G16_B16R16_2PLANE_420_UNORM => (
            2,
            width.div_ceil(2),
            height.div_ceil(2),
            Layout::TwoPlane {
                uv_width: width.div_ceil(2),
                uv_height: height.div_ceil(2),
            },
        ),
        vk::Format::G8_B8R8_2PLANE_422_UNORM => (
            1,
            width,
            height.div_ceil(2),
            Layout::TwoPlane {
                uv_width: width,
                uv_height: height.div_ceil(2),
            },
        ),
        // Two-plane 4:4:4 (Rext): plane 0 = G at full res, plane 1 = BR
        // interleaved at full res.
        vk::Format::G8_B8R8_2PLANE_444_UNORM => (
            1,
            width,
            height,
            Layout::TwoPlane {
                uv_width: width,
                uv_height: height,
            },
        ),
        vk::Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16
        | vk::Format::G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16
        | vk::Format::G16_B16R16_2PLANE_444_UNORM => (
            2,
            width,
            height,
            Layout::TwoPlane {
                uv_width: width,
                uv_height: height,
            },
        ),
        vk::Format::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16
        | vk::Format::G16_B16R16_2PLANE_422_UNORM => (
            2,
            width,
            height.div_ceil(2),
            Layout::TwoPlane {
                uv_width: width,
                uv_height: height.div_ceil(2),
            },
        ),
        // Three-plane 4:4:4: plane 0 = G, plane 1 = B, plane 2 = R, all full res.
        vk::Format::G8_B8_R8_3PLANE_444_UNORM => (1, width, height, Layout::ThreePlane),
        vk::Format::G16_B16_R16_3PLANE_444_UNORM => (2, width, height, Layout::ThreePlane),
        vk::Format::R8G8B8A8_UNORM => (1, width, height, Layout::Packed4 { channels: 4 }),
        vk::Format::R16G16B16A16_UNORM => (2, width, height, Layout::Packed4 { channels: 4 }),
        vk::Format::R8_UNORM => (1, width, height, Layout::Mono),
        vk::Format::R16_UNORM => (2, width, height, Layout::Mono),
        _ => return None,
    };
    Some((sample_size, chroma_w, chroma_h, layout))
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
    format: vk::Format,
    base_array_layer: u32,
    width: u32,
    height: u32,
    old_layout: vk::ImageLayout,
) -> Result<DecodedPixels, VideoError> {
    let (sample_size, chroma_width, chroma_height, layout) = classify_format(format, width, height)
        .ok_or_else(|| {
            eprintln!(
                "[READBACK] WARNING: unknown decode format {:?}, falling back to NV12 layout",
                format
            );
            VideoError::DecoderInit(format!("unsupported decode output format {:?}", format))
        })?;

    let ss = sample_size as u64;
    // Per-plane copy regions (aspect, buffer offset, row length in bytes, extent).
    #[derive(Clone, Copy)]
    struct Region {
        aspect: vk::ImageAspectFlags,
        offset: u64,
        /// Row pitch in bytes as reported by vkGetImageSubresourceLayout;
        /// 0 means tightly packed.
        row_len: u32,
        /// Tightly-packed row size in bytes (extent width * texel block size).
        packed_row: u32,
        width: u32,
        height: u32,
        plane_size: u64,
    }
    let mut regions: Vec<Region> = Vec::new();
    match &layout {
        Layout::TwoPlane {
            uv_width,
            uv_height,
        } => {
            let y_bytes = width as u64 * height as u64 * ss;
            regions.push(Region {
                aspect: vk::ImageAspectFlags::PLANE_0,
                offset: 0,
                row_len: 0,
                packed_row: (width as u64 * ss) as u32,
                width,
                height,
                plane_size: 0,
            });
            regions.push(Region {
                aspect: vk::ImageAspectFlags::PLANE_1,
                offset: y_bytes,
                row_len: 0,
                packed_row: (*uv_width as u64 * 2 * ss) as u32,
                width: *uv_width,
                height: *uv_height,
                plane_size: 0,
            });
        }
        Layout::ThreePlane => {
            for aspect in [
                vk::ImageAspectFlags::PLANE_0,
                vk::ImageAspectFlags::PLANE_1,
                vk::ImageAspectFlags::PLANE_2,
            ] {
                regions.push(Region {
                    aspect,
                    offset: 0,
                    row_len: 0,
                    packed_row: (width as u64 * ss) as u32,
                    width,
                    height,
                    plane_size: 0,
                });
            }
        }
        Layout::Packed4 { channels } => {
            regions.push(Region {
                aspect: vk::ImageAspectFlags::COLOR,
                offset: 0,
                row_len: 0,
                packed_row: (width as u64 * ss * *channels as u64) as u32,
                width,
                height,
                plane_size: 0,
            });
        }
        Layout::Mono => {
            regions.push(Region {
                aspect: vk::ImageAspectFlags::COLOR,
                offset: 0,
                row_len: 0,
                packed_row: (width as u64 * ss) as u32,
                width,
                height,
                plane_size: 0,
            });
        }
    }

    // DPB images are created with VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, so the
    // driver may pad rows. Query the actual per-plane layout and use it for
    // both the row pitch and the staging buffer size (VUID-00183: the buffer
    // must cover all accessed buffer locations; assuming tightly-packed rows
    // writes out of bounds and loses the device). Offsets are cumulative over
    // the actual (possibly padded) plane sizes.
    let mut off = 0u64;
    for r in regions.iter_mut() {
        let sub = vk::ImageSubresource::default()
            .aspect_mask(r.aspect)
            .mip_level(0)
            .array_layer(base_array_layer);
        let l = unsafe { device.get_image_subresource_layout(image, sub) };
        r.offset = off;
        off += l.size;
        r.row_len = l.row_pitch as u32;
        r.plane_size = l.size;
    }

    // Effective row pitch: the queried pitch, or the packed row size when the
    // layout reports 0 (tightly packed).
    let eff_pitch = |r: &Region| -> usize {
        if r.row_len == 0 {
            r.packed_row as usize
        } else {
            r.row_len as usize
        }
    };

    let total_size: u64 = regions.iter().map(|r| r.plane_size).sum();

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

        // Transition every plane being copied to TRANSFER_SRC_OPTIMAL.
        let barriers: Vec<vk::ImageMemoryBarrier2> = regions
            .iter()
            .map(|r| vk::ImageMemoryBarrier2 {
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
                    aspect_mask: r.aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer,
                    layer_count: 1,
                },
                _marker: Default::default(),
            })
            .collect();
        let dep_info = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::BY_REGION,
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: std::ptr::null(),
            image_memory_barrier_count: barriers.len() as u32,
            p_image_memory_barriers: barriers.as_ptr(),
            _marker: Default::default(),
        };
        cmd_pipeline_barrier_2(instance, device.handle(), cmd_buffer, &dep_info);

        for r in &regions {
            device.cmd_copy_image_to_buffer(
                cmd_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                &[vk::BufferImageCopy::default()
                    .buffer_offset(r.offset)
                    .buffer_row_length(r.row_len)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(r.aspect)
                            .mip_level(0)
                            .base_array_layer(base_array_layer)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D::default())
                    .image_extent(vk::Extent3D {
                        width: r.width,
                        height: r.height,
                        depth: 1,
                    })],
            );
        }

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

        // Restore planes to the decode layout.
        let restore: Vec<vk::ImageMemoryBarrier2> = regions
            .iter()
            .map(|r| vk::ImageMemoryBarrier2 {
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
                    aspect_mask: r.aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer,
                    layer_count: 1,
                },
                _marker: Default::default(),
            })
            .collect();
        let dep_info = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::BY_REGION,
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: std::ptr::null(),
            image_memory_barrier_count: restore.len() as u32,
            p_image_memory_barriers: restore.as_ptr(),
            _marker: Default::default(),
        };
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

        // Wait indefinitely: the staging copy must be complete before the CPU
        // reads it. (A finite tiny timeout here raced the GPU and produced
        // corrupted readbacks of tail frames.)
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| VideoError::FenceWait(format!("readback fence wait: {e:?}")))?;

        let ss = sample_size as usize;
        let mut pixels: Vec<u8> = vec![0u8; total_size as usize];
        std::ptr::copy_nonoverlapping(
            mapped_ptr as *const u8,
            pixels.as_mut_ptr(),
            total_size as usize,
        );

        device.unmap_memory(memory);
        device.free_memory(memory, None);
        device.destroy_buffer(buffer, None);

        // Extract planes row-by-row, honoring the (possibly padded) row pitch
        // of the mutable image.
        let mut y_plane: Vec<u8> = Vec::new();
        let mut u_plane: Vec<u8> = Vec::new();
        let mut v_plane: Vec<u8> = Vec::new();

        match layout {
            Layout::TwoPlane {
                uv_width,
                uv_height,
            } => {
                let (r0, r1) = (&regions[0], &regions[1]);
                let p0 = eff_pitch(r0);
                let p1 = eff_pitch(r1);
                let y_row = width as usize * ss;
                y_plane.resize((width as usize) * (height as usize) * ss, 0);
                for y in 0..height as usize {
                    let src = r0.offset as usize + y * p0;
                    y_plane[y * y_row..y * y_row + y_row]
                        .copy_from_slice(&pixels[src..src + y_row]);
                }
                let n_uvw = uv_width as usize;
                let n_uvrows = uv_height as usize;
                u_plane.resize(n_uvw * n_uvrows * ss, 0);
                v_plane.resize(n_uvw * n_uvrows * ss, 0);
                for uy in 0..n_uvrows {
                    let src = r1.offset as usize + uy * p1;
                    for ux in 0..n_uvw {
                        let dst = (uy * n_uvw + ux) * ss;
                        u_plane[dst..dst + ss]
                            .copy_from_slice(&pixels[src + ux * 2 * ss..src + (ux * 2 + 1) * ss]);
                        v_plane[dst..dst + ss].copy_from_slice(
                            &pixels[src + (ux * 2 + 1) * ss..src + (ux * 2 + 2) * ss],
                        );
                    }
                }
            }
            Layout::ThreePlane => {
                let row_bytes = width as usize * ss;
                for (r, plane) in regions
                    .iter()
                    .zip([&mut y_plane, &mut u_plane, &mut v_plane])
                {
                    let p = eff_pitch(r);
                    plane.resize((width as usize) * (height as usize) * ss, 0);
                    for y in 0..height as usize {
                        let src = r.offset as usize + y * p;
                        plane[y * row_bytes..y * row_bytes + row_bytes]
                            .copy_from_slice(&pixels[src..src + row_bytes]);
                    }
                }
            }
            Layout::Packed4 { channels } => {
                let r = &regions[0];
                let p = eff_pitch(r);
                let npix = (width as usize) * (height as usize);
                y_plane.resize(npix * ss, 0);
                u_plane.resize(npix * ss, 0);
                v_plane.resize(npix * ss, 0);
                for y in 0..height as usize {
                    let src = r.offset as usize + y * p;
                    for x in 0..width as usize {
                        let base = src + x * ss * channels as usize;
                        let dst = (y * width as usize + x) * ss;
                        y_plane[dst..dst + ss].copy_from_slice(&pixels[base..base + ss]);
                        u_plane[dst..dst + ss].copy_from_slice(&pixels[base + ss..base + 2 * ss]);
                        v_plane[dst..dst + ss]
                            .copy_from_slice(&pixels[base + 2 * ss..base + 3 * ss]);
                    }
                }
            }
            Layout::Mono => {
                let r = &regions[0];
                let p = eff_pitch(r);
                let row_bytes = width as usize * ss;
                y_plane.resize((width as usize) * (height as usize) * ss, 0);
                for y in 0..height as usize {
                    let src = r.offset as usize + y * p;
                    y_plane[y * row_bytes..y * row_bytes + row_bytes]
                        .copy_from_slice(&pixels[src..src + row_bytes]);
                }
            }
        }

        // Packed 16-bit formats store the sample in the TOP bits of each word
        // (e.g. P010: 10-bit sample in the top 10 bits, bottom 6 unused).
        // Normalize to plain little-endian sample values.
        let shift = match format {
            vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
            | vk::Format::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16
            | vk::Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16
            | vk::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16
            | vk::Format::G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16
            | vk::Format::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16 => 6,
            vk::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16
            | vk::Format::G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16
            | vk::Format::G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16
            | vk::Format::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16
            | vk::Format::G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16 => 4,
            _ => 0,
        };
        if shift > 0 {
            for plane in [&mut y_plane, &mut u_plane, &mut v_plane] {
                let mut i = 0;
                while i + 1 < plane.len() {
                    let v = u16::from_le_bytes([plane[i], plane[i + 1]]);
                    let n = v >> shift;
                    plane[i..i + 2].copy_from_slice(&n.to_le_bytes());
                    i += 2;
                }
            }
        }

        Ok(DecodedPixels {
            y_plane,
            u_plane,
            v_plane,
            sample_size,
            chroma_width,
            chroma_height,
        })
    }
}

/// Source format descriptor for readback.
#[derive(Clone, Copy)]
enum HdrSource {
    /// G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 (or the 12-bit equivalent):
    /// plane0 = 2 bytes/px, plane1 = 4 bytes per (U,V) pair. Each sample is a
    /// u16 with the `bits`-bit value in the HIGH bits (`value << (16 - bits)`,
    /// i.e. G10X6 = 10 bits + 6 pad, G12X4 = 12 bits + 4 pad).
    B16 { bits: u32 },
}

fn hdr_source_for_format(format: vk::Format) -> Option<HdrSource> {
    match format {
        vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR => {
            Some(HdrSource::B16 { bits: 10 })
        }
        vk::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR => {
            Some(HdrSource::B16 { bits: 12 })
        }
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

        // Wait indefinitely: the staging copy must be complete before the CPU
        // reads it. (A finite tiny timeout here raced the GPU and produced
        // corrupted readbacks of tail frames.)
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| VideoError::FenceWait(format!("readback fence wait: {e:?}")))?;

        let uv_plane_size = (uv_width * uv_height) as usize;
        let mut bit_depth = 8u32;

        let (y_plane, u_plane, v_plane) = match hdr_source_for_format(source_format) {
            Some(HdrSource::B16 { bits }) => {
                bit_depth = bits;
                // Planes store little-endian 16-bit samples (value in the
                // low `bits` bits), matching ffmpeg rawvideo
                // yuv420p{10,12}le layout. The GPU stores G10X6/G12X4 with
                // the value in the HIGH bits of each u16.
                let ss = 2usize;
                let mut y_plane = vec![0u8; (width * height) as usize * ss];
                let mut u_plane = vec![0u8; uv_plane_size * ss];
                let mut v_plane = vec![0u8; uv_plane_size * ss];
                // Plane 0: one u16 per luma sample.
                let src = mapped_ptr as *const u16;
                for i in 0..(width * height) as usize {
                    let x = (*src.add(i)) as u32 >> (16 - bits);
                    y_plane[i * ss] = x as u8;
                    y_plane[i * ss + 1] = (x >> 8) as u8;
                }
                // Plane 1: interleaved u16 U, u16 V pairs.
                let src = (mapped_ptr.add(y_size)) as *const u16;
                for i in 0..uv_plane_size {
                    let u = (*src.add(i * 2)) as u32 >> (16 - bits);
                    let v = (*src.add(i * 2 + 1)) as u32 >> (16 - bits);
                    u_plane[i * ss] = u as u8;
                    u_plane[i * ss + 1] = (u >> 8) as u8;
                    v_plane[i * ss] = v as u8;
                    v_plane[i * ss + 1] = (v >> 8) as u8;
                }
                (y_plane, u_plane, v_plane)
            }
            _ => {
                // 8-bit NV12: copy, then deinterleave UV.
                let mut y_plane = vec![0u8; (width * height) as usize];
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
                let mut u_plane = vec![0u8; uv_plane_size];
                let mut v_plane = vec![0u8; uv_plane_size];
                for i in 0..uv_plane_size {
                    u_plane[i] = uv_plane[i * 2];
                    v_plane[i] = uv_plane[i * 2 + 1];
                }
                (y_plane, u_plane, v_plane)
            }
        };

        device.unmap_memory(memory);
        device.free_memory(memory, None);
        device.destroy_buffer(buffer, None);

        Ok(DecodedPixels {
            y_plane,
            u_plane,
            v_plane,
            sample_size: if bit_depth > 8 { 2 } else { 1 },
            chroma_width: uv_width,
            chroma_height: uv_height,
        })
    }
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

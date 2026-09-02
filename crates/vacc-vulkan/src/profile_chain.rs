//! Helpers for creating Vulkan resources with VkVideoProfileListInfoKHR pNext chains.

use super::vp9::vp9_vk_constants;
use super::{VideoError, VideoResult, buffer::BitstreamBuffer, device::VideoCodec};
use ash::vk;
use ash::vk::Handle;

/// Create an output image with VkVideoProfileListInfoKHR in the pNext chain.
pub fn create_output_image_with_profile(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    format: vk::Format,
    codec: VideoCodec,
    profile_idc: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    queue_family_index: u32,
) -> VideoResult<(vk::Image, vk::ImageView, vk::DeviceMemory)> {
    let image = create_image_with_profile_chain(
        device,
        width,
        height,
        format,
        codec,
        profile_idc,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
        queue_family_index,
        1,
    )?;

    let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_type_index = find_memory_type(
        memory_properties,
        mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| {
        VideoError::MemoryAllocation("No suitable memory type for output image".to_string())
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
            .bind_image_memory(image, memory, 0)
            .map_err(|e| VideoError::MemoryAllocation(e.to_string()))?;
    }

    let view_create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect_mask_for_format(format),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let view = unsafe {
        device
            .create_image_view(&view_create_info, None)
            .map_err(|e| VideoError::ImageCreation(e.to_string()))?
    };

    Ok((image, view, memory))
}

/// Create a SINGLE DPB image with `num_slots` array layers plus one image view
/// per slot, each view selecting exactly one layer (`base_array_layer = slot`,
/// `layer_count = 1`).
///
/// Used when the device does NOT support
/// `VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR` (dpbAndOutput
/// coincide), in which case the Vulkan spec (VUID-VkVideoBeginCodingInfoKHR-
/// flags-07244) requires ALL reference imageViews to come from the SAME image.
/// This matches the C++ reference exactly:
///   - `if(!(flags & SEPARATE_REFERENCE_IMAGES)) m_useImageArray = VK_TRUE`
///     (VkVideoDecoder.cpp:349-353)
///   - `imageSpecDpb.createInfo.arrayLayers = m_useImageArray ? numDecodeSurfaces : 1`
///     (VkVideoDecoder.cpp:544)
///   - per-slot view: `VkImageSubresourceRange{COLOR, 0, 1, baseArrayLayer=slot, 1}`
///     (VulkanVideoFrameBuffer.cpp:845-846)
///
/// Returns (shared_image, per_slot_views, memory). Every element of
/// `per_slot_views` references the same `shared_image` at a distinct layer.
pub fn create_dpb_image_array_with_profile(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    format: vk::Format,
    codec: VideoCodec,
    profile_idc: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    queue_family_index: u32,
    num_slots: u32,
) -> VideoResult<(vk::Image, Vec<vk::ImageView>, vk::DeviceMemory)> {
    let image = create_image_with_profile_chain(
        device,
        width,
        height,
        format,
        codec,
        profile_idc,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
        queue_family_index,
        num_slots,
    )?;

    let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_type_index = find_memory_type(
        memory_properties,
        mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| {
        VideoError::MemoryAllocation("No suitable memory type for DPB image array".to_string())
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
            .bind_image_memory(image, memory, 0)
            .map_err(|e| VideoError::MemoryAllocation(e.to_string()))?;
    }

    let mut views = Vec::with_capacity(num_slots as usize);
    for slot in 0..num_slots {
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: aspect_mask_for_format(format),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: slot,
            layer_count: 1,
        };
        let view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(subresource_range);

        // DEBUG (iteration 10): print exact VkImageViewCreateInfo for DPB views
        if slot == 0 {
            eprintln!(
                "[DPB-IV-CREATE] slot=0: image={:#x} viewType={:?} format={:?} subresourceRange={{ aspectMask={:?} baseMipLevel={} levelCount={} baseArrayLayer={} layerCount={} }}",
                image.as_raw(),
                view_create_info.view_type,
                view_create_info.format,
                view_create_info.subresource_range.aspect_mask,
                view_create_info.subresource_range.base_mip_level,
                view_create_info.subresource_range.level_count,
                view_create_info.subresource_range.base_array_layer,
                view_create_info.subresource_range.layer_count,
            );
            eprintln!(
                "[DPB-IV-CREATE]   image extent: {}x{}x{} arrayLayers={} flags={:?}",
                width,
                height,
                1,
                num_slots,
                vk::ImageCreateFlags::MUTABLE_FORMAT
            );
        }

        let view = unsafe {
            device
                .create_image_view(&view_create_info, None)
                .map_err(|e| VideoError::ImageCreation(e.to_string()))?
        };
        views.push(view);
    }

    Ok((image, views, memory))
}

fn create_image_with_profile_chain(
    device: &ash::Device,
    width: u32,
    height: u32,
    format: vk::Format,
    codec: VideoCodec,
    profile_idc: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    queue_family_index: u32,
    array_layers: u32,
) -> VideoResult<vk::Image> {
    let codec_op = codec.to_vk_flag();

    let mut h264_profile = vk::VideoDecodeH264ProfileInfoKHR::default();
    let mut h265_profile = vk::VideoDecodeH265ProfileInfoKHR::default();
    let mut av1_profile = vk::VideoDecodeAV1ProfileInfoKHR::default();
    let mut vp9_profile = super::vp9::VideoDecodeVP9ProfileInfoKHR::default();

    let profile_next: *const std::ffi::c_void = match codec {
        VideoCodec::DecodeH264 => {
            h264_profile.std_profile_idc = profile_idc;
            h264_profile.picture_layout = vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE;
            &h264_profile as *const _ as *const std::ffi::c_void
        }
        VideoCodec::DecodeH265 => {
            h265_profile.std_profile_idc = profile_idc;
            &h265_profile as *const _ as *const std::ffi::c_void
        }
        VideoCodec::DecodeAv1 => {
            // Required by the Vulkan spec for AV1 decode: the profile chain must
            // include VkVideoDecodeAV1ProfileInfoKHR with the AV1 profile idc.
            av1_profile.s_type = vk::StructureType::VIDEO_DECODE_AV1_PROFILE_INFO_KHR;
            av1_profile.p_next = std::ptr::null();
            av1_profile.std_profile = profile_idc;
            av1_profile.film_grain_support = 0;
            &av1_profile as *const _ as *const std::ffi::c_void
        }
        VideoCodec::DecodeVp9 => {
            vp9_profile.s_type =
                vk::StructureType::from_raw(vp9_vk_constants::VIDEO_DECODE_VP9_PROFILE_INFO_KHR);
            vp9_profile.p_next = std::ptr::null();
            vp9_profile.std_profile = profile_idc;
            &vp9_profile as *const _ as *const std::ffi::c_void
        }
    };

    let profile_info = vk::VideoProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
        p_next: profile_next as *const _,
        video_codec_operation: codec_op,
        chroma_subsampling,
        luma_bit_depth,
        chroma_bit_depth,
        _marker: Default::default(),
    };

    let profile_list = vk::VideoProfileListInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_LIST_INFO_KHR,
        p_next: std::ptr::null(),
        profile_count: 1,
        p_profiles: &profile_info as *const _,
        _marker: Default::default(),
    };

    let image_create_info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        p_next: &profile_list as *const _ as *const _,
        // The C++ reference (VulkanVideoImagePool.cpp) creates the DPB images with
        // VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT; keep it to match.
        flags: vk::ImageCreateFlags::MUTABLE_FORMAT,
        image_type: vk::ImageType::TYPE_2D,
        format,
        extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
        mip_levels: 1,
        array_layers,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
            | vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
            | vk::ImageUsageFlags::TRANSFER_SRC,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        // The C++ reference explicitly owns the DPB images by the video decode
        // queue family (queueFamilyIndexCount = 1). With count = 0 the owning
        // queue family is left to the driver, which on NVIDIA can make the
        // decode queue unable to write the image -> decode silently skipped.
        queue_family_index_count: 1,
        p_queue_family_indices: &queue_family_index as *const u32,
        initial_layout: vk::ImageLayout::UNDEFINED,
        _marker: Default::default(),
    };

    // DEBUG (iteration 10): print image creation parameters
    if array_layers > 1 {
        eprintln!(
            "[DPB-IMG-CREATE] format={:?} extent={{ {}x{}x1 }} arrayLayers={} flags={:?} usage={:?} queueFamilyIdx={}",
            format,
            width,
            height,
            array_layers,
            image_create_info.flags,
            image_create_info.usage,
            queue_family_index
        );
        eprintln!(
            "[DPB-IMG-CREATE]   profile={{ codecOp={:?} chromaSubsampling={:?} lumaBitDepth={:?} chromaBitDepth={:?} }}",
            profile_info.video_codec_operation,
            profile_info.chroma_subsampling,
            profile_info.luma_bit_depth,
            profile_info.chroma_bit_depth,
        );
    }

    let image = unsafe {
        device
            .create_image(&image_create_info, None)
            .map_err(|e| VideoError::ImageCreation(e.to_string()))?
    };

    Ok(image)
}

/// Create a bitstream buffer for video decode.
///
/// The C++ reference (VulkanBistreamBufferImpl.cpp) creates the bitstream buffer
/// with VK_BUFFER_CREATE_VIDEO_PROFILE_INDEPENDENT_BIT_KHR (no profile list in
/// pNext). A profile-restricted buffer (VkVideoProfileListInfoKHR) made the
/// NVIDIA driver silently skip the decode, so we match the reference exactly.
pub fn create_bitstream_buffer_with_profile(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    _codec: VideoCodec,
    _profile_idc: u32,
    _chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    _luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    _chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    queue_family_index: u32,
) -> VideoResult<BitstreamBuffer> {
    BitstreamBuffer::create_with_pnext(
        device,
        memory_properties,
        size,
        1,
        256,
        std::ptr::null(),
        vk::BufferCreateFlags::VIDEO_PROFILE_INDEPENDENT_KHR,
        queue_family_index,
    )
}

/// Aspect mask for an image view over a decode output format.
///
/// Multi-planar DPB views/barriers MUST use the COLOR aspect: combining
/// multiple multi-planar bits (PLANE_0|PLANE_1) in one view is forbidden by
/// VUID-VkImageViewCreateInfo-subresourceRange-07818, and the NVIDIA driver
/// traps at submit time on such views. FFmpeg and the C++ reference both use
/// COLOR-aspect views for 2-plane/3-plane formats (the earlier "COLOR breaks
/// P010" note was a misattribution — the real bugs were the missing
/// query-pool reset and the Begin-slot state).
pub fn aspect_mask_for_format(_format: vk::Format) -> vk::ImageAspectFlags {
    vk::ImageAspectFlags::COLOR
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

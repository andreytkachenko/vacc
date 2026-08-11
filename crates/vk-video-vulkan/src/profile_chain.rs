//! Helpers for creating Vulkan resources with VkVideoProfileListInfoKHR pNext chains.

use ash::vk::{self, Handle};
use super::vp9::vp9_vk_constants;
use super::{device::VideoCodec, buffer::BitstreamBuffer, VideoError, VideoResult};

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
            aspect_mask: vk::ImageAspectFlags::COLOR,
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
) -> VideoResult<vk::Image> {
    let codec_op = codec.to_vk_flag();

    let mut h264_profile = vk::VideoDecodeH264ProfileInfoKHR::default();
    let mut h265_profile = vk::VideoDecodeH265ProfileInfoKHR::default();
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
        VideoCodec::DecodeVp9 => {
            vp9_profile.s_type = vk::StructureType::from_raw(vp9_vk_constants::VIDEO_DECODE_VP9_PROFILE_INFO_KHR);
            vp9_profile.p_next = std::ptr::null();
            vp9_profile.std_profile = profile_idc;
            &vp9_profile as *const _ as *const std::ffi::c_void
        }
        _ => std::ptr::null(),
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
        flags: vk::ImageCreateFlags::MUTABLE_FORMAT,
        image_type: vk::ImageType::TYPE_2D,
        format,
        extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
            | vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
            | vk::ImageUsageFlags::TRANSFER_SRC,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: std::ptr::null(),
        initial_layout: vk::ImageLayout::UNDEFINED,
        _marker: Default::default(),
    };

    let image = unsafe {
        device
            .create_image(&image_create_info, None)
            .map_err(|e| VideoError::ImageCreation(e.to_string()))?
    };

    Ok(image)
}

/// Create a bitstream buffer with VkVideoProfileListInfoKHR in the pNext chain.
pub fn create_bitstream_buffer_with_profile(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    codec: VideoCodec,
    profile_idc: u32,
    chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR,
    luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
    chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR,
) -> VideoResult<BitstreamBuffer> {
    let codec_op = codec.to_vk_flag();

    let mut h264_profile = vk::VideoDecodeH264ProfileInfoKHR::default();
    let mut h265_profile = vk::VideoDecodeH265ProfileInfoKHR::default();
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
        VideoCodec::DecodeVp9 => {
            vp9_profile.s_type = vk::StructureType::from_raw(vp9_vk_constants::VIDEO_DECODE_VP9_PROFILE_INFO_KHR);
            vp9_profile.p_next = std::ptr::null();
            vp9_profile.std_profile = profile_idc;
            &vp9_profile as *const _ as *const std::ffi::c_void
        }
        _ => std::ptr::null(),
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

    BitstreamBuffer::create_with_pnext(
        device,
        memory_properties,
        size,
        1,
        256,
        &profile_list as *const _ as *const std::ffi::c_void,
    )
}

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required_flags: vk::MemoryPropertyFlags,
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

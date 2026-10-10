//! Vulkan Video H.264 encode session: all GPU resources and the per-frame
//! encode/readback pipeline.
//!
//! Design notes (verified against the Khronos Vulkan-Video-Samples reference
//! implementation and the Vulkan 1.3 spec):
//! - Motion estimation is done by the driver; the app only supplies SPS/PPS,
//!   reference lists (by DPB slot index) and slice parameters.
//! - CQP = `VkVideoEncodeRateControlInfoKHR::rate_control_mode == DISABLED`
//!   plus per-slice `constantQp`.
//! - Bitstream size is determined by zeroing the host-coherent bitstream
//!   buffer before each encode and scanning for the last non-zero byte after
//!   the encode — the driver's feedback query does not report sizes
//!   reliably on NVIDIA (returns 0/COMPLETE with a full buffer).
//! - Reference picture marking is implicit (`adaptive_ref_pic_marking_mode_flag = 0`);
//!   the driver applies the standard rule from `max_num_ref_frames`.
//! - ash 0.38 exposes no VK Video entry points, so every video function is
//!   loaded dynamically via vkGetDeviceProcAddr (same pattern as vacc-vulkan).
//!
//! Per-frame command flow (mirrors VkVideoEncoder::RecordVideoCodingCmd):
//!   [outside coding region]
//!     barrier HOST_WRITE -> TRANSFER_READ        (staging buffer)
//!     vkCmdCopyBufferToImage  staging -> input image (VIDEO_ENCODE_SRC_KHR)
//!     barrier TRANSFER -> VIDEO_ENCODE_READ      (input image)
//!   [coding region]
//!     vkCmdBeginVideoCodingKHR (pNext = rate control; slots as in encode info)
//!     vkCmdControlVideoCodingKHR(RESET)          first frame only
//!     vkCmdEncodeVideoKHR
//!     image barrier VIDEO_ENCODE_WRITE -> READ   (setup DPB slot)
//!     vkCmdEndVideoCodingKHR

use std::ffi::CString;

use ash::vk;
use ash::vk::Handle;
use ash::vk::native;
use vacc_vulkan::{VideoDeviceBuilder, VulkanDevice};

use super::Error;
use super::h264_params::H264Params;

/// NV12 (Y plane + interleaved UV, 4:2:0) as exposed to Vulkan video.
pub const NV12: vk::Format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

const STD_HEADER_NAME: &str = "VK_STD_vulkan_video_codec_h264_encode";

fn build_std_header_version(extension_name: &str) -> vk::ExtensionProperties {
    let mut props = vk::ExtensionProperties::default();
    let bytes = format!("{}\0", extension_name).into_bytes();
    props
        .extension_name
        .iter_mut()
        .zip(bytes.iter())
        .for_each(|(c, &b)| *c = b as std::os::raw::c_char);
    props.spec_version = 1u32 << 22;
    props
}

/// All VK Video entry points (ash 0.38 does not wrap them).
struct VideoFns {
    create_session: unsafe extern "system" fn(
        vk::Device,
        *const vk::VideoSessionCreateInfoKHR<'_>,
        *const vk::AllocationCallbacks<'_>,
        *mut vk::VideoSessionKHR,
    ) -> vk::Result,
    destroy_session:
        unsafe extern "system" fn(vk::Device, vk::VideoSessionKHR, *const vk::AllocationCallbacks<'_>),
    get_session_mem_reqs: unsafe extern "system" fn(
        vk::Device,
        vk::VideoSessionKHR,
        *mut u32,
        *mut vk::VideoSessionMemoryRequirementsKHR<'_>,
    ) -> vk::Result,
    bind_session_memory: unsafe extern "system" fn(
        vk::Device,
        vk::VideoSessionKHR,
        u32,
        *const vk::BindVideoSessionMemoryInfoKHR<'_>,
    ) -> vk::Result,
    create_params: unsafe extern "system" fn(
        vk::Device,
        *const vk::VideoSessionParametersCreateInfoKHR<'_>,
        *const vk::AllocationCallbacks<'_>,
        *mut vk::VideoSessionParametersKHR,
    ) -> vk::Result,
    destroy_params: unsafe extern "system" fn(
        vk::Device,
        vk::VideoSessionParametersKHR,
        *const vk::AllocationCallbacks<'_>,
    ),
    cmd_begin_coding:
        unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoBeginCodingInfoKHR<'_>),
    cmd_encode_video: unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoEncodeInfoKHR<'_>),
    cmd_end_coding: unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoEndCodingInfoKHR<'_>),
    cmd_control_coding:
        unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoCodingControlInfoKHR<'_>),
}

fn load_fn<T>(instance: &ash::Instance, device: vk::Device, name: &str) -> Result<T, Error> {
    let cname = CString::new(name).unwrap();
    let ptr = unsafe { instance.get_device_proc_addr(device, cname.as_ptr()) }
        .ok_or_else(|| Error::EncoderInit(format!("{name} not found")))?;
    let raw: usize = ptr as *const () as usize;
    if raw == 0 {
        return Err(Error::EncoderInit(format!("{name} is null")));
    }
    // SAFETY: every T used here is a 64-bit extern "system" fn pointer,
    // the same size and alignment as usize on this platform.
    Ok(unsafe { std::ptr::read_unaligned(&raw as *const _ as *const T) })
}

impl VideoFns {
    fn load(instance: &ash::Instance, device: vk::Device) -> Result<Self, Error> {
        Ok(Self {
            create_session: load_fn(instance, device, "vkCreateVideoSessionKHR")?,
            destroy_session: load_fn(instance, device, "vkDestroyVideoSessionKHR")?,
            get_session_mem_reqs: load_fn(instance, device, "vkGetVideoSessionMemoryRequirementsKHR")?,
            bind_session_memory: load_fn(instance, device, "vkBindVideoSessionMemoryKHR")?,
            create_params: load_fn(instance, device, "vkCreateVideoSessionParametersKHR")?,
            destroy_params: load_fn(instance, device, "vkDestroyVideoSessionParametersKHR")?,
            cmd_begin_coding: load_fn(instance, device, "vkCmdBeginVideoCodingKHR")?,
            cmd_encode_video: load_fn(instance, device, "vkCmdEncodeVideoKHR")?,
            cmd_end_coding: load_fn(instance, device, "vkCmdEndVideoCodingKHR")?,
            cmd_control_coding: load_fn(instance, device, "vkCmdControlVideoCodingKHR")?,
        })
    }
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    props
        .memory_types
        .iter()
        .enumerate()
        .find(|(i, mt)| (type_bits >> i) & 1 == 1 && mt.property_flags.contains(flags))
        .map(|(i, _)| i as u32)
}

fn alloc_memory(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory, Error> {
    let index = find_memory_type(mem_props, type_bits, flags)
        .ok_or_else(|| Error::EncoderInit("no suitable memory type".into()))?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(size)
        .memory_type_index(index);
    unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| Error::Vulkan(format!("allocate_memory: {e}")))
}

/// Query + allocate + bind the memory of a video session or session parameters.
/// (vkBindVideoSessionMemoryKHR accepts either object type per the spec.)
fn bind_video_memories<T: Handle + Copy>(
    fns: &VideoFns,
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    get_reqs: unsafe extern "system" fn(
        vk::Device,
        T,
        *mut u32,
        *mut vk::VideoSessionMemoryRequirementsKHR<'_>,
    ) -> vk::Result,
    bind_target: T,
) -> Result<Vec<vk::DeviceMemory>, Error> {
    let mut req_count: u32 = 0;
    match unsafe { get_reqs(device.handle(), bind_target, &mut req_count, std::ptr::null_mut()) } {
        vk::Result::SUCCESS => {}
        r => return Err(Error::EncoderInit(format!("get memory requirements (count): {r:?}"))),
    }
    let mut reqs = vec![vk::VideoSessionMemoryRequirementsKHR::default(); req_count as usize];
    for (i, req) in reqs.iter_mut().enumerate() {
        req.memory_bind_index = i as u32;
    }
    match unsafe { get_reqs(device.handle(), bind_target, &mut req_count, reqs.as_mut_ptr()) } {
        vk::Result::SUCCESS => {}
        r => return Err(Error::EncoderInit(format!("get memory requirements: {r:?}"))),
    }

    let mut memories = Vec::new();
    let mut bind_infos = Vec::new();
    for (i, req) in reqs.iter().enumerate() {
        let mr = &req.memory_requirements;
        if mr.memory_type_bits == 0 {
            return Err(Error::EncoderInit("video memory requirement has no valid memory types".into()));
        }
        let mem = alloc_memory(
            device,
            mem_props,
            mr.size.max(1),
            mr.memory_type_bits,
            vk::MemoryPropertyFlags::empty(),
        )?;
        bind_infos.push(vk::BindVideoSessionMemoryInfoKHR {
            s_type: vk::StructureType::BIND_VIDEO_SESSION_MEMORY_INFO_KHR,
            p_next: std::ptr::null(),
            memory_bind_index: i as u32,
            memory: mem,
            memory_offset: 0,
            memory_size: mr.size,
            _marker: Default::default(),
        });
        memories.push(mem);
    }
    // vkBindVideoSessionMemoryKHR accepts either VkVideoSession or
    // VkVideoSessionParameters (both opaque 64-bit handles).
    // SAFETY: T is a 64-bit Vulkan handle, same layout as VideoSessionKHR.
    let bind_target_session =
        unsafe { std::ptr::read_unaligned(&bind_target as *const _ as *const vk::VideoSessionKHR) };
    match unsafe {
        (fns.bind_session_memory)(
            device.handle(),
            bind_target_session,
            bind_infos.len() as u32,
            bind_infos.as_ptr(),
        )
    } {
        vk::Result::SUCCESS => {}
        r => return Err(Error::EncoderInit(format!("bind_video_session_memory: {r:?}"))),
    }
    Ok(memories)
}

fn create_video_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    extent: vk::Extent3D,
    usage: vk::ImageUsageFlags,
    initial_layout: vk::ImageLayout,
    profile_list: &vk::VideoProfileListInfoKHR,
) -> Result<(vk::Image, vk::DeviceMemory), Error> {
    let mut info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(NV12)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .initial_layout(initial_layout);
    info.p_next = profile_list as *const _ as *const _;
    let image = unsafe { device.create_image(&info, None) }
        .map_err(|e| Error::Vulkan(format!("create_image: {e}")))?;
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem = alloc_memory(device, mem_props, reqs.size, reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    unsafe { device.bind_image_memory(image, mem, 0) }
        .map_err(|e| Error::Vulkan(format!("bind_image_memory: {e}")))?;
    Ok((image, mem))
}

fn create_full_image_view(device: &ash::Device, image: vk::Image) -> Result<vk::ImageView, Error> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(NV12)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        );
    unsafe { device.create_image_view(&info, None) }
        .map_err(|e| Error::Vulkan(format!("create_image_view: {e}")))
}

fn create_host_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
    mem_flags: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), Error> {
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .map_err(|e| Error::Vulkan(format!("create_buffer: {e}")))?;
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem = alloc_memory(device, mem_props, reqs.size, reqs.memory_type_bits, mem_flags)?;
    unsafe { device.bind_buffer_memory(buffer, mem, 0) }
        .map_err(|e| Error::Vulkan(format!("bind_buffer_memory: {e}")))?;
    Ok((buffer, mem))
}

/// All GPU resources for one H.264 encode session.
pub struct EncodeSession {
    pub device: VulkanDevice,
    fns: VideoFns,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,

    session: vk::VideoSessionKHR,
    session_memories: Vec<vk::DeviceMemory>,
    parameters: vk::VideoSessionParametersKHR,
    param_memories: Vec<vk::DeviceMemory>,

    dpb_images: Vec<vk::Image>,
    dpb_views: Vec<vk::ImageView>,
    dpb_memories: Vec<vk::DeviceMemory>,

    input_image: vk::Image,
    input_view: vk::ImageView,
    input_memory: vk::DeviceMemory,

    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,

    bitstream_buffer: vk::Buffer,
    bitstream_memory: vk::DeviceMemory,
    /// Capacity of the bitstream buffer in bytes.
    pub bitstream_capacity: u32,

    /// Number of DPB slots (session maxDpbSlots).
    pub slots: u32,
    /// Clamped max active reference pictures.
    pub max_refs: u32,
    pub width: u32,
    pub height: u32,
    /// Driver can emit SPS/PPS prefix NALUs itself (generatePrefixNalu).
    pub generate_prefix_nalu: bool,
}

impl EncodeSession {
    pub fn new(
        params: &H264Params,
        width: u32,
        height: u32,
        cfg_max_refs: u32,
    ) -> Result<Self, Error> {
        let device = VideoDeviceBuilder::new()
            .with_video_codecs(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .build()
            .map_err(|e| Error::CodecNotSupported(format!("no Vulkan H.264 encode device: {e}")))?;

        let fns = VideoFns::load(&device.instance, device.device.handle())?;

        let encode_qf = device
            .queue_families
            .video_encode
            .ok_or_else(|| Error::CodecNotSupported("no video encode queue".into()))?;

        let (caps, _enc_caps, h264_caps) = device
            .query_video_encode_capabilities_h264(
                params.profile_idc,
                vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
                vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
                vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            )
            .map_err(|e| Error::CodecNotSupported(format!("H.264 encode caps: {e}")))?;

        if h264_caps.max_slice_count < 1 {
            return Err(Error::CodecNotSupported("maxSliceCount == 0".into()));
        }
        let slots = caps.max_dpb_slots.max(2);
        let max_refs = cfg_max_refs
            .min(caps.max_active_reference_pictures)
            .min(slots - 1)
            .max(1);

        // Profile chain (shared by session + image creation).
        let h264_profile = vk::VideoEncodeH264ProfileInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H264_PROFILE_INFO_KHR,
            p_next: std::ptr::null(),
            std_profile_idc: params.profile_idc,
            _marker: Default::default(),
        };
        let profile_info = vk::VideoProfileInfoKHR {
            s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
            p_next: &h264_profile as *const _ as *const _,
            video_codec_operation: vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
            chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            _marker: Default::default(),
        };
        let profile_list = vk::VideoProfileListInfoKHR {
            s_type: vk::StructureType::VIDEO_PROFILE_LIST_INFO_KHR,
            p_next: std::ptr::null(),
            profile_count: 1,
            p_profiles: &profile_info as *const _,
            _marker: Default::default(),
        };

        // ---- Session ----
        let std_header_version = build_std_header_version(STD_HEADER_NAME);
        let session_create_info = vk::VideoSessionCreateInfoKHR {
            s_type: vk::StructureType::VIDEO_SESSION_CREATE_INFO_KHR,
            p_next: std::ptr::null(),
            queue_family_index: encode_qf,
            flags: vk::VideoSessionCreateFlagsKHR::empty(),
            p_video_profile: &profile_info as *const _,
            picture_format: NV12,
            max_coded_extent: vk::Extent2D { width, height },
            reference_picture_format: NV12,
            max_dpb_slots: slots,
            max_active_reference_pictures: max_refs,
            p_std_header_version: &std_header_version,
            _marker: Default::default(),
        };
        let mut session = vk::VideoSessionKHR::null();
        match unsafe {
            (fns.create_session)(device.device.handle(), &session_create_info, std::ptr::null(), &mut session)
        } {
            vk::Result::SUCCESS => {}
            r => return Err(Error::EncoderInit(format!("create_video_session: {r:?}"))),
        }

        let session_memories = bind_video_memories(
            &fns,
            &device.device,
            &device.memory_properties,
            fns.get_session_mem_reqs,
            session,
        )?;

        // ---- Session parameters (SPS/PPS) ----
        let params_add_info = vk::VideoEncodeH264SessionParametersAddInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
            p_next: std::ptr::null(),
            std_sps_count: 1,
            p_std_sp_ss: &params.sps,
            std_pps_count: 1,
            p_std_pp_ss: &params.pps,
            _marker: Default::default(),
        };
        // The driver locates SPS/PPS through an intermediate H264 params-create
        // struct in the pNext chain (NOT the AddInfo directly) — see Khronos
        // reference VideoSessionParametersInfo.
        let h264_params_create = vk::VideoEncodeH264SessionParametersCreateInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
            p_next: std::ptr::null(),
            max_std_sps_count: 1,
            max_std_pps_count: 1,
            p_parameters_add_info: &params_add_info as *const _,
            _marker: Default::default(),
        };
        let params_create_info = vk::VideoSessionParametersCreateInfoKHR {
            s_type: vk::StructureType::VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
            p_next: &h264_params_create as *const _ as *const _,
            flags: vk::VideoSessionParametersCreateFlagsKHR::empty(),
            video_session: session,
            video_session_parameters_template: vk::VideoSessionParametersKHR::null(),
            _marker: Default::default(),
        };
        let mut parameters = vk::VideoSessionParametersKHR::null();
        match unsafe {
            (fns.create_params)(device.device.handle(), &params_create_info, std::ptr::null(), &mut parameters)
        } {
            vk::Result::SUCCESS => {}
            r => return Err(Error::EncoderInit(format!("create_video_session_parameters: {r:?}"))),
        }

        // Video session parameters use driver-managed memory (created above with
        // pAllocator=NULL); NVIDIA does not export vkGetVideoSessionParametersMemory-
        // RequirementsKHR, and the Khronos reference performs no separate alloc/bind.
        let param_memories = Vec::new();

        // ---- DPB images ----
        let extent = vk::Extent3D { width, height, depth: 1 };
        let mut dpb_images = Vec::with_capacity(slots as usize);
        let mut dpb_views = Vec::with_capacity(slots as usize);
        let mut dpb_memories = Vec::with_capacity(slots as usize);
        for _ in 0..slots {
            let (img, mem) = create_video_image(
                &device.device,
                &device.memory_properties,
                extent,
                vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
                vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
                &profile_list,
            )?;
            let view = create_full_image_view(&device.device, img)?;
            dpb_images.push(img);
            dpb_views.push(view);
            dpb_memories.push(mem);
        }

        // ---- Input image + staging buffer ----
        let (input_image, input_memory) = create_video_image(
            &device.device,
            &device.memory_properties,
            extent,
            vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
            &profile_list,
        )?;
        let input_view = create_full_image_view(&device.device, input_image)?;

        let staging_size = (width as u64) * (height as u64) * 2;
        let (staging_buffer, staging_memory) = create_host_buffer(
            &device.device,
            &device.memory_properties,
            staging_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // ---- Bitstream buffer ----
        let bitstream_capacity = (width as u64)
            .max(1024)
            .saturating_mul(height as u64)
            .max(caps.min_bitstream_buffer_size_alignment) as u32;
        let (bitstream_buffer, bitstream_memory) = create_host_buffer(
            &device.device,
            &device.memory_properties,
            bitstream_capacity as u64,
            vk::BufferUsageFlags::VIDEO_ENCODE_DST_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // ---- Command pool / buffer / fence ----
        let command_pool = unsafe {
            device
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(encode_qf),
                    None,
                )
        }
        .map_err(|e| Error::EncoderInit(format!("create_command_pool: {e}")))?;
        let command_buffer = unsafe {
            device.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| Error::EncoderInit(format!("allocate_command_buffers: {e}")))?[0];
        let fence = unsafe { device.device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| Error::EncoderInit(format!("create_fence: {e}")))?;
        let queue = unsafe { device.device.get_device_queue(encode_qf, 0) };
        let generate_prefix_nalu = h264_caps
            .flags
            .contains(vk::VideoEncodeH264CapabilityFlagsKHR::GENERATE_PREFIX_NALU);

        Ok(Self {
            device,
            fns,
            queue,
            command_pool,
            command_buffer,
            fence,
            session,
            session_memories,
            parameters,
            param_memories,
            dpb_images,
            dpb_views,
            dpb_memories,
            input_image,
            input_view,
            input_memory,
            staging_buffer,
            staging_memory,
            bitstream_buffer,
            bitstream_memory,
            bitstream_capacity,
            slots,
            max_refs,
            width,
            height,
            generate_prefix_nalu,
        })
    }

    /// Encode one frame synchronously; returns the raw Annex B VCL bytes
    /// (plus SPS/PPS prefix NALUs when `emit_prefix` is set and the driver
    /// supports generatePrefixNalu). `reset_session` must be true for the
    /// first frame (clears stale session state, per the reference impl).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_one_frame(
        &self,
        frame: &vacc_core::encoder::InputFrame,
        index: u32,
        is_idr: bool,
        qp: u8,
        poc: i32,
        frame_num: u32,
        emit_prefix: bool,
        reset_session: bool,
    ) -> Result<Vec<u8>, Error> {
        let width = self.width;
        let height = self.height;

        // ---- Upload NV12 to the input image via a host-visible staging buffer.
        let plane_bytes = (width as u64) * (height as u64) * 2;
        let mapped = unsafe { self.device.device.map_memory(self.staging_memory, 0, plane_bytes, vk::MemoryMapFlags::empty()) }
            .map_err(|e| Error::Vulkan(format!("map staging: {e}")))?;
        {
            let dst = unsafe { std::slice::from_raw_parts_mut(mapped as *mut u8, (width as usize) * (height as usize) * 2) };
            let y_len = (width as usize) * (height as usize);
            // Luma rows.
            for row in 0..height as usize {
                dst[row * width as usize..(row + 1) * width as usize]
                    .copy_from_slice(&frame.y[row * frame.y_pitch..row * frame.y_pitch + width as usize]);
            }
            // Interleaved chroma rows (width bytes each, height/2 rows).
            for row in 0..(height / 2) as usize {
                let s = row * frame.uv_pitch;
                let d = y_len + row * width as usize;
                dst[d..d + width as usize].copy_from_slice(&frame.uv[s..s + width as usize]);
            }
        }
        unsafe { self.device.device.unmap_memory(self.staging_memory) };

        // ---- Per-frame std structures (must live through vkCmdEncodeVideoKHR).
        let nref = if is_idr { 0 } else { self.max_refs.min(index).max(1) };
        let setup_slot = (index % self.slots) as usize;

        // Reference list L0: most recent `nref` frames (DPB slot indices).
        let mut refpic_list0 = [0xFFu8; 32];
        for (k, slot) in refpic_list0.iter_mut().take(nref as usize).enumerate() {
            *slot = ((index - 1 - k as u32) % self.slots) as u8;
        }

        let ref_lists: Option<Box<native::StdVideoEncodeH264ReferenceListsInfo>> = if is_idr {
            None
        } else {
            Some(Box::new(native::StdVideoEncodeH264ReferenceListsInfo {
                flags: unsafe { std::mem::zeroed() },
                num_ref_idx_l0_active_minus1: (nref - 1) as u8,
                num_ref_idx_l1_active_minus1: 0,
                RefPicList0: refpic_list0,
                RefPicList1: [0xFFu8; 32],
                refList0ModOpCount: 0,
                refList1ModOpCount: 0,
                refPicMarkingOpCount: 0, // implicit marking (see module docs)
                reserved1: [0u8; 7],
                pRefList0ModOperations: std::ptr::null(),
                pRefList1ModOperations: std::ptr::null(),
                pRefPicMarkingOperations: std::ptr::null(),
            }))
        };

        let mut pic_flags = unsafe { std::mem::zeroed::<native::StdVideoEncodeH264PictureInfoFlags>() };
        pic_flags.set_IdrPicFlag(is_idr as u32);
        pic_flags.set_is_reference(1);
        let std_picture_info = native::StdVideoEncodeH264PictureInfo {
            flags: pic_flags,
            seq_parameter_set_id: 0,
            pic_parameter_set_id: 0,
            idr_pic_id: if is_idr { 1 } else { 0 },
            primary_pic_type: if is_idr {
                native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
            } else {
                native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P
            },
            frame_num,
            PicOrderCnt: poc,
            temporal_id: 0,
            reserved1: [0u8; 3],
            pRefLists: ref_lists.as_ref().map_or(std::ptr::null(), |rl| &**rl as *const _),
        };

        let slice_header = native::StdVideoEncodeH264SliceHeader {
            flags: unsafe { std::mem::zeroed() },
            first_mb_in_slice: 0,
            slice_type: if is_idr {
                native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_I
            } else {
                native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_P
            },
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            slice_qp_delta: 0, // CQP via constantQp
            reserved1: 0,
            cabac_init_idc: 0, // CAVLC stream — ignored
            disable_deblocking_filter_idc: 0,
            pWeightTable: std::ptr::null(),
        };

        let nalu_slice_info = vk::VideoEncodeH264NaluSliceInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H264_NALU_SLICE_INFO_KHR,
            p_next: std::ptr::null(),
            constant_qp: qp as i32,
            p_std_slice_header: &slice_header,
            _marker: Default::default(),
        };

        let picture_info = vk::VideoEncodeH264PictureInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H264_PICTURE_INFO_KHR,
            p_next: std::ptr::null(),
            nalu_slice_entry_count: 1,
            p_nalu_slice_entries: &nalu_slice_info,
            p_std_picture_info: &std_picture_info,
            generate_prefix_nalu: emit_prefix as u32,
            _marker: Default::default(),
        };

        // ---- Picture resources + reference slots.
        // Layout: slot 0 = setup (current picture in its DPB image), 1..=nref = active refs.
        let mut res_infos: Vec<vk::VideoPictureResourceInfoKHR> = Vec::with_capacity(nref as usize + 1);
        res_infos.push(vk::VideoPictureResourceInfoKHR {
            s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
            p_next: std::ptr::null(),
            coded_offset: vk::Offset2D { x: 0, y: 0 },
            coded_extent: vk::Extent2D { width, height },
            base_array_layer: 0,
            image_view_binding: self.dpb_views[setup_slot],
            _marker: Default::default(),
        });
        for k in 0..nref {
            res_infos.push(vk::VideoPictureResourceInfoKHR {
                s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: std::ptr::null(),
                coded_offset: vk::Offset2D { x: 0, y: 0 },
                coded_extent: vk::Extent2D { width, height },
                base_array_layer: 0,
                image_view_binding: self.dpb_views[((index - 1 - k) % self.slots) as usize],
                _marker: Default::default(),
            });
        }

        // Standard reference info for each active reference (frame_num/poc).
        // The StdVideo structs MUST outlive vkCmdEncodeVideoKHR — keep them in a
        // stable Vec (pointing at loop-locals would dangle).
        let mut ref_infos_vec: Vec<native::StdVideoEncodeH264ReferenceInfo> = Vec::with_capacity(nref as usize);
        for k in 0..nref {
            ref_infos_vec.push(native::StdVideoEncodeH264ReferenceInfo {
                flags: unsafe { std::mem::zeroed() },
                primary_pic_type: native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P,
                FrameNum: index - 1 - k,
                // POC type 2 derivation for the reference frame (2 * its frame_num).
                PicOrderCnt: ((index - 1 - k) * 2) as i32,
                long_term_pic_num: 0,
                long_term_frame_idx: 0,
                temporal_id: 0,
            });
        }
        let mut dpb_slot_infos: Vec<vk::VideoEncodeH264DpbSlotInfoKHR> = Vec::with_capacity(nref as usize);
        for k in 0..nref {
            dpb_slot_infos.push(vk::VideoEncodeH264DpbSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_reference_info: &ref_infos_vec[k as usize],
                _marker: Default::default(),
            });
        }

        let mut slot_infos: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::with_capacity(nref as usize + 1);
        // Setup slot (current picture's write target), listed in BeginCoding with its
        // valid DPB index. (Empirically the NVIDIA driver requires a valid index here —
        // setting it to -1 makes even the IDR produce 0 bytes.)
        slot_infos.push(vk::VideoReferenceSlotInfoKHR {
            s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
            p_next: std::ptr::null(),
            slot_index: setup_slot as i32,
            p_picture_resource: &res_infos[0],
            _marker: Default::default(),
        });
        for k in 0..nref {
            slot_infos.push(vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: &dpb_slot_infos[k as usize] as *const _ as *const _,
                slot_index: ((index - 1 - k) % self.slots) as i32,
                p_picture_resource: &res_infos[k as usize + 1],
                _marker: Default::default(),
            });
        }

        let encode_info = vk::VideoEncodeInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_INFO_KHR,
            p_next: &picture_info as *const _ as *const _,
            flags: vk::VideoEncodeFlagsKHR::empty(),
            dst_buffer: self.bitstream_buffer,
            dst_buffer_offset: 0,
            dst_buffer_range: self.bitstream_capacity as u64,
            src_picture_resource: vk::VideoPictureResourceInfoKHR {
                s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: std::ptr::null(),
                coded_offset: vk::Offset2D { x: 0, y: 0 },
                coded_extent: vk::Extent2D { width, height },
                base_array_layer: 0,
                image_view_binding: self.input_view,
                _marker: Default::default(),
            },
            // pReferenceSlots holds ONLY the references (setup is separate via
            // pSetupReferenceSlot) — including it here makes the driver treat the
            // current picture's DPB slot as a self-reference (crash on P-frames).
            p_setup_reference_slot: &slot_infos[0],
            reference_slot_count: nref,
            p_reference_slots: if nref > 0 { unsafe { slot_infos.as_ptr().add(1) } } else { std::ptr::null() },
            preceding_externally_encoded_bytes: 0,
            _marker: Default::default(),
        };

        // CQP: rate control disabled + per-slice constant_qp.
        let rate_control_info = vk::VideoEncodeRateControlInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_RATE_CONTROL_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoEncodeRateControlFlagsKHR::empty(),
            rate_control_mode: vk::VideoEncodeRateControlModeFlagsKHR::DISABLED,
            layer_count: 0,
            p_layers: std::ptr::null(),
            virtual_buffer_size_in_ms: 0,
            initial_virtual_buffer_size_in_ms: 0,
            _marker: Default::default(),
        };
        let begin_info = vk::VideoBeginCodingInfoKHR {
            s_type: vk::StructureType::VIDEO_BEGIN_CODING_INFO_KHR,
            p_next: &rate_control_info as *const _ as *const _,
            flags: vk::VideoBeginCodingFlagsKHR::empty(),
            video_session: self.session,
            video_session_parameters: self.parameters,
            reference_slot_count: slot_infos.len() as u32,
            p_reference_slots: slot_infos.as_ptr(),
            _marker: Default::default(),
        };
        let end_info = vk::VideoEndCodingInfoKHR {
            s_type: vk::StructureType::VIDEO_END_CODING_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoEndCodingFlagsKHR::empty(),
            _marker: Default::default(),
        };
        // Dedicated session-init rate control (matches reference InitRateControl):
        // a RESET control carrying per-layer minQp/maxQp so the driver knows the QP
        // operating point before any picture is encoded. The layer info is chained
        // directly onto the control command (as the Khronos reference does).
        let h264_qp = vk::VideoEncodeH264QpKHR {
            qp_i: qp as i32,
            qp_p: qp as i32,
            qp_b: qp as i32,
        };
        let h264_frame_size = vk::VideoEncodeH264FrameSizeKHR {
            frame_i_size: 0,
            frame_p_size: 0,
            frame_b_size: 0,
        };
        let h264_rc_layer = vk::VideoEncodeH264RateControlLayerInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H264_RATE_CONTROL_LAYER_INFO_KHR,
            p_next: std::ptr::null(),
            use_min_qp: 1,
            min_qp: h264_qp,
            use_max_qp: 1,
            max_qp: h264_qp,
            use_max_frame_size: 1,
            max_frame_size: h264_frame_size,
            _marker: Default::default(),
        };
        let rc_layer = vk::VideoEncodeRateControlLayerInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_RATE_CONTROL_LAYER_INFO_KHR,
            p_next: &h264_rc_layer as *const _ as *const _,
            average_bitrate: 0,
            max_bitrate: 0,
            frame_rate_numerator: 0,
            frame_rate_denominator: 0,
            _marker: Default::default(),
        };
        let control_info = vk::VideoCodingControlInfoKHR {
            s_type: vk::StructureType::VIDEO_CODING_CONTROL_INFO_KHR,
            p_next: &rc_layer as *const _ as *const _,
            flags: vk::VideoCodingControlFlagsKHR::RESET,
            _marker: Default::default(),
        };

        // ---- Dedicated session init (RESET + QP) before the first encode.
        // Recorded and submitted as its own coding region, exactly like the
        // reference's InitRateControl — NOT folded into the first frame's encode.
        if reset_session {
            let dev = &self.device.device;
            let cmd = self.command_buffer;
            unsafe { dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()) }
                .map_err(|e| Error::Vulkan(format!("init reset_command_buffer: {e}")))?;
            unsafe { dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()) }
                .map_err(|e| Error::Vulkan(format!("init begin_command_buffer: {e}")))?;
            let init_begin = vk::VideoBeginCodingInfoKHR {
                s_type: vk::StructureType::VIDEO_BEGIN_CODING_INFO_KHR,
                p_next: std::ptr::null(),
                flags: vk::VideoBeginCodingFlagsKHR::empty(),
                video_session: self.session,
                video_session_parameters: self.parameters,
                reference_slot_count: 0,
                p_reference_slots: std::ptr::null(),
                _marker: Default::default(),
            };
            unsafe { (self.fns.cmd_begin_coding)(cmd, &init_begin) };
            unsafe { (self.fns.cmd_control_coding)(cmd, &control_info) };
            unsafe { (self.fns.cmd_end_coding)(cmd, &end_info) };
            unsafe { dev.end_command_buffer(cmd) }
                .map_err(|e| Error::Vulkan(format!("init end_command_buffer: {e}")))?;
            unsafe { dev.reset_fences(&[self.fence]) }
                .map_err(|e| Error::Vulkan(format!("init reset_fences: {e}")))?;
            unsafe {
                dev.queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                    self.fence,
                )
            }
            .map_err(|e| Error::Vulkan(format!("init queue_submit: {e}")))?;
            unsafe { dev.wait_for_fences(&[self.fence], true, 30_000_000_000) }
                .map_err(|e| Error::Vulkan(format!("init wait_for_fences: {e}")))?;
            eprintln!("[ENC] frame {index}: session init (reset+qp) done");
        }

        // Zero the bitstream buffer before encoding. The NVIDIA driver does not
        // report a reliable feedback-query size for this encode, so we determine
        // the coded size by scanning for the last non-zero byte after the encode
        // (CAVLC always ends on a non-zero stop-bit byte). Host-coherent memory:
        // CPU writes are immediately visible to the GPU.
        let dev = &self.device.device;
        {
            let m = unsafe { dev.map_memory(self.bitstream_memory, 0, self.bitstream_capacity as u64, vk::MemoryMapFlags::empty()) }
                .map_err(|e| Error::Vulkan(format!("map bitstream (zero): {e}")))?;
            unsafe { std::ptr::write_bytes(m as *mut u8, 0, self.bitstream_capacity as usize) };
            unsafe { dev.unmap_memory(self.bitstream_memory) };
        }

        // ---- Record + submit.
        let cmd = self.command_buffer;
        unsafe { dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()) }
            .map_err(|e| Error::Vulkan(format!("reset_command_buffer: {e}")))?;
        unsafe { dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()) }
            .map_err(|e| Error::Vulkan(format!("begin_command_buffer: {e}")))?;

        // [outside coding region] staging buffer was host-written.
        let buf_barrier = vk::BufferMemoryBarrier2 {
            s_type: vk::StructureType::BUFFER_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::NONE,
            src_access_mask: vk::AccessFlags2::HOST_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            dst_access_mask: vk::AccessFlags2::TRANSFER_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer: self.staging_buffer,
            offset: 0,
            size: plane_bytes,
            _marker: Default::default(),
        };
        let dep = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::empty(),
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 1,
            p_buffer_memory_barriers: &buf_barrier,
            image_memory_barrier_count: 0,
            p_image_memory_barriers: std::ptr::null(),
            _marker: Default::default(),
        };
        unsafe { dev.cmd_pipeline_barrier2(cmd, &dep) };

        let copy_region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0, // tightly packed
            buffer_image_height: height,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D { width, height, depth: 1 },
        };
        unsafe {
            dev.cmd_copy_buffer_to_image(
                cmd,
                self.staging_buffer,
                self.input_image,
                vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
                &[copy_region],
            )
        };
        eprintln!("[ENC] frame {index}: copied");

        // [outside coding region] input image now readable by video encode.
        let img_barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            dst_access_mask: vk::AccessFlags2::VIDEO_ENCODE_READ_KHR,
            old_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
            new_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: self.input_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            _marker: Default::default(),
        };
        let dep2 = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::empty(),
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: std::ptr::null(),
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &img_barrier,
            _marker: Default::default(),
        };
        unsafe { dev.cmd_pipeline_barrier2(cmd, &dep2) };

        // [coding region] (session already initialized via the dedicated reset+qp
        // submit above when reset_session is set).
        unsafe { (self.fns.cmd_begin_coding)(cmd, &begin_info) };
        unsafe { (self.fns.cmd_encode_video)(cmd, &encode_info) };

        // The setup slot now holds the reconstructed current picture: make it
        // readable for future reference (per the reference impl).
        let dpb_barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            src_access_mask: vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR,
            dst_stage_mask: vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            dst_access_mask: vk::AccessFlags2::VIDEO_ENCODE_READ_KHR,
            old_layout: vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
            new_layout: vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: self.dpb_images[setup_slot],
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            _marker: Default::default(),
        };
        let dep3 = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::empty(),
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: std::ptr::null(),
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &dpb_barrier,
            _marker: Default::default(),
        };
        unsafe { dev.cmd_pipeline_barrier2(cmd, &dep3) };

        unsafe { (self.fns.cmd_end_coding)(cmd, &end_info) };
        eprintln!("[ENC] frame {index}: end_coding");
        unsafe { dev.end_command_buffer(cmd) }
            .map_err(|e| Error::Vulkan(format!("end_command_buffer: {e}")))?;
        eprintln!("[ENC] frame {index}: recorded, submitting");

        unsafe { dev.reset_fences(&[self.fence]) }
            .map_err(|e| Error::Vulkan(format!("reset_fences: {e}")))?;
        unsafe {
            dev.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                self.fence,
            )
        }
        .map_err(|e| Error::Vulkan(format!("queue_submit: {e}")))?;
        unsafe { dev.wait_for_fences(&[self.fence], true, 30_000_000_000) }
            .map_err(|e| Error::Vulkan(format!("wait_for_fences: {e}")))?;

        // ---- Determine the coded size by scanning for the last non-zero byte.
        // The buffer was zeroed before the encode and the bitstream is written
        // contiguously from offset 0, so the trailing run of zeros marks the end
        // of the frame. (CAVLC always ends on a non-zero stop-bit byte.) This
        // sidesteps the driver's unreliable feedback-query size reporting.
        let mapped = unsafe { dev.map_memory(self.bitstream_memory, 0, self.bitstream_capacity as u64, vk::MemoryMapFlags::empty()) }
            .map_err(|e| Error::Vulkan(format!("map bitstream: {e}")))?;
        let buf = unsafe { std::slice::from_raw_parts(mapped as *const u8, self.bitstream_capacity as usize) };
        let size = buf.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
        let data = buf[..size].to_vec();
        unsafe { dev.unmap_memory(self.bitstream_memory) };

        Ok(data)
    }
}

impl Drop for EncodeSession {
    fn drop(&mut self) {
        let d = &self.device.device;
        unsafe {
            let _ = d.device_wait_idle();
            d.destroy_buffer(self.bitstream_buffer, None);
            d.free_memory(self.bitstream_memory, None);
            d.destroy_buffer(self.staging_buffer, None);
            d.free_memory(self.staging_memory, None);
            d.destroy_image_view(self.input_view, None);
            d.destroy_image(self.input_image, None);
            d.free_memory(self.input_memory, None);
            for i in 0..self.dpb_images.len() {
                d.destroy_image_view(self.dpb_views[i], None);
                d.destroy_image(self.dpb_images[i], None);
                d.free_memory(self.dpb_memories[i], None);
            }
            (self.fns.destroy_params)(d.handle(), self.parameters, std::ptr::null());
            for mem in &self.param_memories {
                d.free_memory(*mem, None);
            }
            (self.fns.destroy_session)(d.handle(), self.session, std::ptr::null());
            for mem in &self.session_memories {
                d.free_memory(*mem, None);
            }
            d.destroy_fence(self.fence, None);
            d.free_command_buffers(self.command_pool, &[self.command_buffer]);
            d.destroy_command_pool(self.command_pool, None);
        };
    }
}

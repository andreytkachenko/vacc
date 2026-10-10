//! Diagnostic: isolate why vkCreateSessionParametersKHR fails on the A10.
use std::ffi::CString;
use ash::vk;
use ash::vk::native::{
    StdVideoH264PictureParameterSet, StdVideoH264SequenceParameterSet, StdVideoH264SequenceParameterSetVui,
};
use vacc_vulkan::VideoDeviceBuilder;

fn load_fn<T>(instance: &ash::Instance, device: vk::Device, name: &str) -> Option<T> {
    let cname = CString::new(name).unwrap();
    let ptr = unsafe { instance.get_device_proc_addr(device, cname.as_ptr()) }?;
    let raw: usize = ptr as *const () as usize;
    if raw == 0 { return None; }
    Some(unsafe { std::ptr::read_unaligned(&raw as *const _ as *const T) })
}

fn main() {
    let dev = VideoDeviceBuilder::new()
        .with_video_codecs(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
        .build().expect("encode device");
    let create_session: unsafe extern "system" fn(
        vk::Device, *const vk::VideoSessionCreateInfoKHR<'_>, *const vk::AllocationCallbacks<'_>, *mut vk::VideoSessionKHR,
    ) -> vk::Result = load_fn(&dev.instance, dev.device.handle(), "vkCreateVideoSessionKHR").unwrap();
    let get_session_mem_reqs: unsafe extern "system" fn(
        vk::Device, vk::VideoSessionKHR, *mut u32, *mut vk::VideoSessionMemoryRequirementsKHR<'_>,
    ) -> vk::Result = load_fn(&dev.instance, dev.device.handle(), "vkGetVideoSessionMemoryRequirementsKHR").unwrap();
    let bind_session_memory: unsafe extern "system" fn(
        vk::Device, vk::VideoSessionKHR, u32, vk::DeviceMemory, vk::DeviceSize,
    ) -> vk::Result = load_fn(&dev.instance, dev.device.handle(), "vkBindVideoSessionMemoryKHR").unwrap();
    let create_params: unsafe extern "system" fn(
        vk::Device, *const vk::VideoSessionParametersCreateInfoKHR<'_>, *const vk::AllocationCallbacks<'_>, *mut vk::VideoSessionParametersKHR,
    ) -> vk::Result = load_fn(&dev.instance, dev.device.handle(), "vkCreateVideoSessionParametersKHR").unwrap();

    let encode_qf = dev.queue_families.video_encode.unwrap();
    let (caps, _, _) = dev.query_video_encode_capabilities_h264(77,
        vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
        vk::VideoComponentBitDepthFlagsKHR::TYPE_8, vk::VideoComponentBitDepthFlagsKHR::TYPE_8).expect("caps");
    let slots = caps.max_dpb_slots;
    let width: u32 = 320;
    let height: u32 = 240;

    let h264_profile = vk::VideoEncodeH264ProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_ENCODE_H264_PROFILE_INFO_KHR, p_next: std::ptr::null(),
        std_profile_idc: 77, _marker: Default::default(),
    };
    let profile_info = vk::VideoProfileInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
        p_next: &h264_profile as *const _ as *const _,
        video_codec_operation: vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
        chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
        luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR::TYPE_8, _marker: Default::default(),
    };
    let profile_list = vk::VideoProfileListInfoKHR {
        s_type: vk::StructureType::VIDEO_PROFILE_LIST_INFO_KHR, p_next: std::ptr::null(),
        profile_count: 1, p_profiles: &profile_info as *const _, _marker: Default::default(),
    };
    let mut hdr = unsafe { std::mem::zeroed::<vk::ExtensionProperties>() };
    { let b = b"VK_STD_vulkan_video_codec_h264_encode\0";
      for i in 0..256usize { hdr.extension_name[i] = if i < b.len() { b[i] as i8 } else { 0 }; } }
    hdr.spec_version = 1u32 << 22;

    let sci = vk::VideoSessionCreateInfoKHR {
        s_type: vk::StructureType::VIDEO_SESSION_CREATE_INFO_KHR, p_next: std::ptr::null(),
        queue_family_index: encode_qf, flags: vk::VideoSessionCreateFlagsKHR::empty(),
        p_video_profile: &profile_info as *const _, picture_format: vk::Format::G8_B8R8_2PLANE_420_UNORM,
        max_coded_extent: vk::Extent2D { width, height }, reference_picture_format: vk::Format::G8_B8R8_2PLANE_420_UNORM,
        max_dpb_slots: slots, max_active_reference_pictures: 1,
        p_std_header_version: &hdr, _marker: Default::default(),
    };
    let mut session = vk::VideoSessionKHR::null();
    assert_eq!(unsafe { create_session(dev.device.handle(), &sci, std::ptr::null(), &mut session) }, vk::Result::SUCCESS);
    let mut count = 1u32;
    let mut reqs = unsafe { std::mem::zeroed::<vk::VideoSessionMemoryRequirementsKHR>() };
    unsafe { let _ = get_session_mem_reqs(dev.device.handle(), session, &mut count, &mut reqs); };
    if count > 0 {
        let r = &reqs.memory_requirements;
        let mp = dev.memory_properties;
        let mt = (0..mp.memory_types.len()).find(|&i| (r.memory_type_bits >> i) & 1 == 1
            && mp.memory_types[i].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)).unwrap();
        let mi = vk::MemoryAllocateInfo { s_type: vk::StructureType::MEMORY_ALLOCATE_INFO, p_next: std::ptr::null(),
            allocation_size: r.size, memory_type_index: mt as u32, _marker: Default::default() };
        let mem = unsafe { dev.device.allocate_memory(&mi, None) }.unwrap();
        unsafe { let _ = bind_session_memory(dev.device.handle(), session, 0, mem, 0); };
    }

    let vui = unsafe { std::mem::zeroed::<StdVideoH264SequenceParameterSetVui>() };

    for (label, cs1, with_vui) in [
        ("cs=0", 0u8, false),
        ("cs1=1", 1u8, false),
        ("cs1=1+vui", 1u8, true),
    ] {
        let mut sps = unsafe { std::mem::zeroed::<StdVideoH264SequenceParameterSet>() };
        sps.flags.set_frame_mbs_only_flag(1);
        sps.flags.set_direct_8x8_inference_flag(1);
        if cs1 == 1 { sps.flags.set_constraint_set1_flag(1); }
        sps.profile_idc = 77;
        sps.level_idc = 31;
        sps.chroma_format_idc = 1;
        sps.log2_max_frame_num_minus4 = 4;
        sps.pic_order_cnt_type = 2;
        sps.max_num_ref_frames = 1;
        sps.pic_width_in_mbs_minus1 = width / 16 - 1;
        sps.pic_height_in_map_units_minus1 = height / 16 - 1;
        if with_vui {
            sps.flags.set_vui_parameters_present_flag(1);
            sps.pSequenceParameterSetVui = &vui;
        }
        let mut pps = unsafe { std::mem::zeroed::<StdVideoH264PictureParameterSet>() };
        pps.flags.set_deblocking_filter_control_present_flag(1);
        pps.num_ref_idx_l0_default_active_minus1 = 0;

        let add_info = vk::VideoEncodeH264SessionParametersAddInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
            p_next: std::ptr::null(), std_sps_count: 1, p_std_sp_ss: &sps,
            std_pps_count: 1, p_std_pp_ss: &pps, _marker: Default::default(),
        };
        let h264_params_create = vk::VideoEncodeH264SessionParametersCreateInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
            p_next: std::ptr::null(), max_std_sps_count: 1, max_std_pps_count: 1,
            p_parameters_add_info: &add_info as *const _, _marker: Default::default(),
        };
        let pci = vk::VideoSessionParametersCreateInfoKHR {
            s_type: vk::StructureType::VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
            p_next: &h264_params_create as *const _ as *const _,
            flags: vk::VideoSessionParametersCreateFlagsKHR::empty(),
            video_session: session, video_session_parameters_template: vk::VideoSessionParametersKHR::null(),
            _marker: Default::default(),
        };
        let mut params = vk::VideoSessionParametersKHR::null();
        let pr = unsafe { create_params(dev.device.handle(), &pci, std::ptr::null(), &mut params) };
        eprintln!("[{label}] create_params -> {pr:?}");
    }

    // Query pool creation tests: vary the p_next profile (bare vs list) + flags.
    let mk_pool = |flags, qcount, pnext: *const ()| {
        let fci = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR {
            s_type: vk::StructureType::QUERY_POOL_VIDEO_ENCODE_FEEDBACK_CREATE_INFO_KHR,
            p_next: pnext as *const _,
            encode_feedback_flags: vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
                | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN,
            _marker: Default::default(),
        };
        let qpi = vk::QueryPoolCreateInfo {
            s_type: vk::StructureType::QUERY_POOL_CREATE_INFO,
            p_next: &fci as *const _ as *const _,
            flags, query_type: vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR,
            query_count: qcount, pipeline_statistics: vk::QueryPipelineStatisticFlags::empty(),
            _marker: Default::default(),
        };
        unsafe { dev.device.create_query_pool(&qpi, None) }
    };
    eprintln!("[bare_profile]  -> {:?}", mk_pool(vk::QueryPoolCreateFlags::empty(), 1, &profile_info as *const _ as *const ()));
    eprintln!("[list_profile]  -> {:?}", mk_pool(vk::QueryPoolCreateFlags::empty(), 1, &profile_list as *const _ as *const ()));
    eprintln!("[reset_bare_n4] -> {:?}", mk_pool(vk::QueryPoolCreateFlags::from_raw(1), 4, &profile_info as *const _ as *const ()));
}

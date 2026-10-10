//! Capability probe: does the selected GPU expose VK Video H.264 *encode*?
//! Run: cargo test -p vacc-vulkan-encode --test probe_capabilities -- --nocapture

use ash::vk;
use vacc_vulkan::{VideoCodec, VideoDeviceBuilder};

fn device_name(dev: &vacc_vulkan::VulkanDevice) -> String {
    let props = unsafe { dev.instance.get_physical_device_properties(dev.physical_device) };
    props
        .device_name
        .iter()
        .take_while(|&&b| b != 0)
        .map(|b| *b as u8 as char)
        .collect()
}

#[test]
fn probe_h264_encode_capabilities() {
    let dev = VideoDeviceBuilder::new()
        .with_video_codecs(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
        .build()
        .expect("vulkan device with encode queue");

    let name = device_name(&dev);
    eprintln!("[PROBE] selected device: {name}");
    eprintln!(
        "[PROBE] queue families: decode={:?} encode={:?} graphics={:?} compute={:?}",
        dev.queue_families.video_decode,
        dev.queue_families.video_encode,
        dev.queue_families.graphics,
        dev.queue_families.compute
    );
    eprintln!(
        "[PROBE] enabled extensions: {:?}",
        dev.enabled_extensions.iter().filter(|e| e.contains("video")).collect::<Vec<_>>()
    );

    let (caps, encode_caps, h264_caps) = dev
        .query_video_encode_capabilities_h264(
            77, // STD_VIDEO_H264_PROFILE_IDC_MAIN
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        )
        .expect("H.264 encode capabilities query");

    eprintln!("[PROBE] ===== VkVideoCapabilitiesKHR =====");
    eprintln!("[PROBE]   minCodedExtent       = {}x{}", caps.min_coded_extent.width, caps.min_coded_extent.height);
    eprintln!("[PROBE]   maxCodedExtent       = {}x{}", caps.max_coded_extent.width, caps.max_coded_extent.height);
    eprintln!("[PROBE]   picAccessGranularity = {}x{}", caps.picture_access_granularity.width, caps.picture_access_granularity.height);
    eprintln!("[PROBE]   maxDpbSlots          = {}", caps.max_dpb_slots);
    eprintln!("[PROBE]   maxActiveRefPictures = {}", caps.max_active_reference_pictures);
    eprintln!("[PROBE]   minBsOffsetAlign     = {}", caps.min_bitstream_buffer_offset_alignment);
    eprintln!("[PROBE]   minBsSizeAlign       = {}", caps.min_bitstream_buffer_size_alignment);
    eprintln!("[PROBE] ===== VkVideoEncodeCapabilitiesKHR =====");
    eprintln!("[PROBE]   flags                = {:?}", encode_caps.flags);
    eprintln!("[PROBE]   rateControlModes     = {:?}", encode_caps.rate_control_modes);
    eprintln!("[PROBE]   maxBitrate           = {}", encode_caps.max_bitrate);
    eprintln!("[PROBE]   maxQualityLevels     = {}", encode_caps.max_quality_levels);
    eprintln!("[PROBE]   inputPicGranularity  = {}x{}", encode_caps.encode_input_picture_granularity.width, encode_caps.encode_input_picture_granularity.height);
    eprintln!("[PROBE]   feedbackFlags        = {:?}", encode_caps.supported_encode_feedback_flags);
    eprintln!("[PROBE] ===== VkVideoEncodeH264CapabilitiesKHR =====");
    eprintln!("[PROBE]   flags                = {:?}", h264_caps.flags);
    eprintln!("[PROBE]   maxLevelIdc          = {}", h264_caps.max_level_idc);
    eprintln!("[PROBE]   maxSliceCount        = {}", h264_caps.max_slice_count);
    eprintln!("[PROBE]   maxPRefL0            = {}", h264_caps.max_p_picture_l0_reference_count);
    eprintln!("[PROBE]   minQp/maxQp          = {}/{}", h264_caps.min_qp, h264_caps.max_qp);
    eprintln!("[PROBE]   gopRemainingFrames   = prefers={} requires={}", h264_caps.prefers_gop_remaining_frames, h264_caps.requires_gop_remaining_frames);
    eprintln!("[PROBE]   stdSyntaxFlags       = {:?}", h264_caps.std_syntax_flags);

    // Sanity: CQP works via rateControlMode == DISABLED + per-slice constantQp.
    assert!(
        encode_caps.rate_control_modes.contains(vk::VideoEncodeRateControlModeFlagsKHR::DISABLED),
        "DISABLED rate control (CQP) not supported"
    );
    eprintln!("[PROBE] OK — H.264 encode supported on {name}");
}

#[test]
fn probe_h264_encode_main10() {
    let dev = VideoDeviceBuilder::new()
        .with_video_codecs(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
        .build()
        .expect("vulkan device with encode queue");

    let res = dev.query_video_encode_capabilities_h264(
        88, // STD_VIDEO_H264_PROFILE_IDC_MAIN_10
        vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
        vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
    );
    match res {
        Ok((caps, _, h264_caps)) => eprintln!(
            "[PROBE-10] Main10 supported: maxDpb={} maxLevel={}",
            caps.max_dpb_slots, h264_caps.max_level_idc
        ),
        Err(e) => eprintln!("[PROBE-10] Main10 NOT supported: {e}"),
    }
}

// Keep the VideoCodec import referenced so the encode variant is exercised.
#[test]
fn video_codec_encode_flag() {
    assert_eq!(
        VideoCodec::EncodeH264.to_vk_flag(),
        vk::VideoCodecOperationFlagsKHR::ENCODE_H264
    );
}

#[test]
fn diag_all_devices() {
    let entry = unsafe { ash::Entry::load() }.unwrap();
    let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 2, 0));
    let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) }
        .expect("instance");
    for pd in unsafe { instance.enumerate_physical_devices() }.unwrap() {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let name: String = props.device_name.iter().take_while(|&&b| b != 0).map(|b| *b as u8 as char).collect();
        let qfs = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        eprintln!("[DIAG] device: {name} ({} queue families)", qfs.len());
        for (i, qf) in qfs.iter().enumerate() {
            eprintln!("[DIAG]   qf[{}]: flags={:?}", i, qf.queue_flags);
        }
        let exts = unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap();
        let names: Vec<String> = exts
            .iter()
            .map(|e| {
                e.extension_name
                    .iter()
                    .take_while(|&&b| b != 0)
                    .map(|b| *b as u8 as char)
                    .collect()
            })
            .collect();
        let vexts: Vec<&String> = names.iter().filter(|n| n.contains("video")).collect();
        eprintln!("[DIAG]   video exts: {vexts:?}");
    }
}

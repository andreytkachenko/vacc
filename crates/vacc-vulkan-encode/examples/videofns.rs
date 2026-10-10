//! Diagnostic: which VK Video functions does the encode device export via
//! vkGetDeviceProcAddr? Run on a GPU with VK_KHR_video_encode_h264.

use std::ffi::CString;
use ash::vk;
use vacc_vulkan::VideoDeviceBuilder;

const FNS: &[&str] = &[
    "vkCreateVideoSessionKHR",
    "vkDestroyVideoSessionKHR",
    "vkGetVideoSessionMemoryRequirementsKHR",
    "vkBindVideoSessionMemoryKHR",
    "vkCreateVideoSessionParametersKHR",
    "vkDestroyVideoSessionParametersKHR",
    "vkGetVideoSessionParametersMemoryRequirementsKHR",
    "vkCmdBeginVideoCodingKHR",
    "vkCmdEncodeVideoKHR",
    "vkCmdEndVideoCodingKHR",
    "vkCmdControlVideoCodingKHR",
];

fn main() {
    let dev = match VideoDeviceBuilder::new()
        .with_video_codecs(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
        .build()
    {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no encode device: {e}");
            return;
        }
    };
    let props = unsafe { dev.instance.get_physical_device_properties(dev.physical_device) };
    let name: String = props
        .device_name
        .iter()
        .take_while(|&&b| b != 0)
        .map(|b| *b as u8 as char)
        .collect();
    eprintln!("device: {name}");
    let entry = unsafe { ash::Entry::load() }.unwrap();
    for f in FNS {
        let cname = CString::new(*f).unwrap();
        let via_device = unsafe { dev.instance.get_device_proc_addr(dev.device.handle(), cname.as_ptr()) };
        let via_instance = unsafe { entry.get_instance_proc_addr(dev.instance.handle(), cname.as_ptr()) };
        eprintln!(
            "{:<52} device={:<5} instance={:<5}",
            f,
            if via_device.is_some() { "OK" } else { "NULL" },
            if via_instance.is_some() { "OK" } else { "NULL" }
        );
    }
}

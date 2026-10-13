//! Regression: concurrent Vulkan entry-level initialization must not crash.
//!
//! The system Vulkan loader (reproduced on 1.4.341 + NVIDIA driver 610.43,
//! RTX 3060) races in its lazy ICD-manifest init: one thread entering through
//! `vkCreateInstance` while another calls an entry-level query such as
//! `vkEnumerateInstanceExtensionProperties` lands in a NULL function pointer
//! inside libvulkan (SIGSEGV). Reproduced in pure C without ash/vacc.
//! Holding [`vacc_core::VULKAN_INIT_LOCK`] on every init path serializes the
//! race away; this test pins that contract under the exact mixed ordering
//! that crashed.

use ash::vk;
use std::thread;

#[test]
fn concurrent_mixed_ordering_init() {
    // Each iteration: a raw-ash thread (instance-first, like a diagnostics
    // path) racing a VideoDeviceBuilder thread (enumeration-first). Without
    // the shared init lock this SIGSEGV'd within a few iterations.
    for _ in 0..10 {
        let raw = thread::spawn(|| {
            let _init = vacc_core::vulkan_init::vulkan_init_guard();
            let entry = unsafe { ash::Entry::load() }.unwrap();
            let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 2, 0));
            let instance = unsafe {
                entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
            }
            .unwrap();
            assert!(!unsafe { instance.enumerate_physical_devices() }.unwrap().is_empty());
        });
        let builder = thread::spawn(|| {
            // build() takes the init lock internally.
            vacc_vulkan::VideoDeviceBuilder::new()
                .with_video_codecs(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
                .build()
                .expect("vulkan device with encode queue");
        });
        raw.join().unwrap();
        builder.join().unwrap();
    }
}

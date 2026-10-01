//! Compiled SPIR-V (built by `build.rs` from the WGSL sources).
pub static YUV2RGB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/yuv2rgb.spv"));
pub static RESIZE_YUV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/resize_yuv.spv"));
pub static WARP_RGB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/warp_rgb.spv"));
pub static NV12_TO_RGBA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nv12_to_rgba.spv"));
pub static NV12_RESIZE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nv12_resize.spv"));
pub static RGBA_TO_RGB24: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rgba_to_rgb24.spv"));

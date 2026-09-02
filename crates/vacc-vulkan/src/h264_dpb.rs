//! H.264 DPB manager.
//!
//! The implementation now lives in the common parser crate
//! (`vacc_parser::h264_dpb`) so all backends (Vulkan, NVDEC, VAAPI) can
//! share it. This module re-exports it to keep existing
//! `crate::h264_dpb::*` paths compiling.
pub use vacc_parser::h264_dpb::*;

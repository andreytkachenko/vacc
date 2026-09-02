//! H.264 Picture Order Count calculation.
//!
//! The POC calculation logic lives in [`vacc_parser::h264_poc`] so that
//! all backends (Vulkan, NVDEC, VAAPI) share ONE common implementation.
//! This module is a thin re-export for backward compatibility with code that
//! imports `vacc_nvdec_decode::poc::PocCalculator`.

pub use vacc_parser::h264_poc::PocCalculator;

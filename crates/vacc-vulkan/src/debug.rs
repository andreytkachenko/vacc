//! Gated debug tracing for the Vulkan backend.
//!
//! Set `VACC_VULKAN_DEBUG=1` to enable verbose stderr traces (session
//! creation, DPB/MMCO operations, capability dumps, etc.). Off by default:
//! a library must not write to stderr on the decode hot path.

/// Returns true when `VACC_VULKAN_DEBUG` is set.
#[inline]
pub fn enabled() -> bool {
    std::env::var_os("VACC_VULKAN_DEBUG").is_some()
}

/// Debug-trace `eprintln!`: prints only when `VACC_VULKAN_DEBUG` is set.
#[macro_export]
macro_rules! vkdbg {
    ($($arg:tt)*) => {{
        if $crate::debug::enabled() {
            eprintln!($($arg)*);
        }
    }};
}

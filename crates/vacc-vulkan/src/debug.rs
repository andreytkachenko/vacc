//! Gated debug tracing for the Vulkan backend.
//!
//! Set `VACC_DEBUG=1` to enable verbose stderr traces (session creation,
//! capability dumps, DPB/MMCO operations, per-frame picture-info dumps).
//! Off by default: a library must not write to stderr on the decode hot path.
//!
//! The gate is the crate-wide [`crate::vacc_debug`] switch, so all Vulkan
//! debug output shares a single env var.

/// Debug-trace `eprintln!`: prints only when `VACC_DEBUG=1` is set.
#[macro_export]
macro_rules! vkdbg {
    ($($arg:tt)*) => {{
        if $crate::vacc_debug() {
            eprintln!($($arg)*);
        }
    }};
}

//! Process-wide serialization of Vulkan entry-level initialization.
//!
//! # Why this exists
//!
//! The system Vulkan loader (reproduced on 1.4.341 with the NVIDIA 610.43
//! driver, RTX 3060) has a race in its lazy ICD-manifest initialization: if
//! one thread enters Vulkan through `vkCreateInstance` while another thread
//! concurrently calls an entry-level query such as
//! `vkEnumerateInstanceExtensionProperties`, the second thread can jump to a
//! NULL function pointer inside `libvulkan.so.1` and SIGSEGV. The crash was
//! reproduced in pure C (no ash, no vacc code) with two threads using
//! *mixed* call ordering — one creating an instance first, the other
//! enumerating instance extensions first. Same-ordering concurrency does not
//! trigger it, which is why identical parallel initializations can appear
//! fine while a raw-diagnostic thread alongside a builder crashes.
//!
//! The Vulkan spec places no threading requirement on the application for
//! entry-level calls, so this cannot be fixed in our code's *usage* — only
//! by serializing it.
//!
//! # Contract
//!
//! Every code path in the process that performs Vulkan entry-level
//! initialization — `Entry::load()` + `vkCreateInstance` + the first
//! entry-level enumerations — must hold [`vulkan_init_guard`] for the whole
//! window. This applies to all vacc crates (they do it internally) and to any
//! user code that creates its own `ash::Instance` alongside vacc.
//!
//! **The lock is NOT reentrant.** A path that already holds it must not call
//! into library code that acquires it again (e.g. a test that takes the
//! guard and then calls `VideoDeviceBuilder::build()` deadlocks — `build()`
//! locks internally). Take the guard on exactly one level of the stack.
//!
//! Device-level calls after initialization need not take the lock.

use std::sync::{Mutex, MutexGuard};

/// The process-wide Vulkan init lock. Prefer [`vulkan_init_guard`].
pub static VULKAN_INIT_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the Vulkan init lock, tolerating poisoning (a panic while
/// initializing must not wedge every subsequent decoder/encoder startup).
pub fn vulkan_init_guard() -> MutexGuard<'static, ()> {
    VULKAN_INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

//! Vulkan Video entry points that ash's safe API does not cover.
//!
//! ash 0.38 exposes the core Vulkan commands as `unsafe fn`s and generates
//! no safe wrappers for the Vulkan Video (KHR_video) commands, so the codec
//! modules resolve those entry points through `vkGetDeviceProcAddr` /
//! `vkGetPhysicalDeviceProcAddr` and call them through typed function
//! pointers.
//!
//! # SAFETY (covers every `video_fn!` use site in this crate)
//!
//! 1. **Pointer provenance.** Every address passed to [`video_fn!`] comes
//!    from `vkGetDeviceProcAddr`/`vkGetPhysicalDeviceProcAddr` (or the
//!    instance equivalent), i.e. from the Vulkan loader. The loader returns
//!    a valid code pointer for every entry point of an extension the device
//!    reports, and each lookup null-checks the `Option` before use.
//! 2. **ABI.** The function-pointer type at each use site copies the
//!    signature from the Vulkan specification (identical to ash's
//!    declarations for the same entry point where ash has one), so calling
//!    through it is well defined.
//! 3. **transmute soundness.** `std::mem::transmute::<usize, F>` is used
//!    with a concrete `F` at every use site (the type is inferred from the
//!    binding annotation). On every platform this crate targets (x86_64) a
//!    function pointer and `usize` have the same size and representation,
//!    so the bit pattern is preserved exactly. The transmute must stay
//!    monomorphic: a generic helper cannot prove the size match (E0512).
//! 4. **Lifetime of borrowed struct arguments** (`*const vk::XxxInfo<'_>`)
//!    is guaranteed at each call site: the pointed-to struct outlives the
//!    call (it is a local or a field borrow held across the call).
//!
//! The `unsafe { ... }` blocks around the actual calls remain at the call
//! sites because calling an `unsafe extern "system" fn` is unsafe by
//! language rule — that is the floor for this pattern.

/// Reinterpret a loader-provided code pointer as a function pointer of the
/// type inferred from the binding (e.g. `let f: FnType = video_fn!(addr);`).
///
/// SAFETY: see the module-level documentation — the address must come from
/// `vkGetDeviceProcAddr`/`vkGetPhysicalDeviceProcAddr`, and the inferred
/// type must match the Vulkan-spec signature of that entry point.
#[macro_export]
macro_rules! video_fn {
    ($addr:expr) => {{
        // No turbofish possible: the type is inferred from the binding
        // annotation, and a generic transmute cannot prove the size match
        // (E0512). See the module-level SAFETY docs.
        #[allow(unused_unsafe, clippy::missing_transmute_annotations)]
        let __vacc_video_fn = unsafe { std::mem::transmute($addr) };
        __vacc_video_fn
    }};
}

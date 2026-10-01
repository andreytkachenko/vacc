//! Runtime-loaded FFI to classic NPP (`nppi..._Ctx`) and the CUDA driver API.
//!
//! NPP is resolved with `libloading` at runtime; there is no build-time link
//! dependency on NPP. The CUDA *driver* (`libcuda`) is different: `build.rs`
//! links it when available (see below) because the driver's primary context
//! only works when its entry points are reached through the library PLT. If
//! the driver was not linked, or any NPP library or symbol is missing,
//! loading fails and the caller falls back to the software pipeline.

use std::ffi::CString;
use std::os::raw::{c_int, c_void};
use std::path::Path;

use libloading::Library;

use crate::NppError;

/// The CUDA driver entry points, called through the normal PLT.
///
/// `build.rs` links `libcuda` when it is available and sets the
/// `vacc_npp_cuda_linked` cfg.
///
/// NOTE: the symbol names are deliberately the `_v2` variants. The CUDA C
/// headers silently redirect `cuMemAlloc` -> `cuMemAlloc_v2` (and the async
/// memcpy variants) on 64-bit, so C code never sees the legacy entry points.
/// The legacy v1 paths in this driver skip fetching the thread's current
/// context unless an internal legacy flag is set, which it never is for
/// modern primary-context usage — they return 201 (INVALID_CONTEXT) with no
/// kernel interaction. Rust has no such header redirection, so the `_v2`
/// names must be declared explicitly.
#[cfg(vacc_npp_cuda_linked)]
mod cuda_driver {
    use std::os::raw::{c_int, c_void};

    unsafe extern "C" {
        pub fn cuInit(flags: u32) -> c_int;
        pub fn cuDeviceGetCount(count: *mut c_int) -> c_int;
        pub fn cuDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        pub fn cuDevicePrimaryCtxRetain(pctx: *mut *mut c_void, device: c_int) -> c_int;
        pub fn cuCtxSetCurrent(ctx: *mut c_void) -> c_int;
        pub fn cuCtxGetCurrent(pctx: *mut *mut c_void) -> c_int;
        pub fn cuStreamCreate(stream: *mut *mut c_void, flags: u32) -> c_int;
        pub fn cuStreamSynchronize(stream: *mut c_void) -> c_int;
        pub fn cuMemAlloc_v2(ptr: *mut usize, size: usize) -> c_int;
        pub fn cuMemFree_v2(ptr: usize) -> c_int;
        pub fn cuMemFreeAsync(ptr: usize, stream: *mut c_void) -> c_int;
        pub fn cuMemcpyHtoDAsync_v2(dst: usize, src: *const c_void, size: usize, stream: *mut c_void) -> c_int;
        pub fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: usize, size: usize, stream: *mut c_void) -> c_int;
    }
}

/// `NppStatus` (success is 0).
pub const NPP_SUCCESS: c_int = 0;

/// `NppInterpolation` values (nppdefs.h).
pub const NPPI_INTER_NEAREST: c_int = 1;
pub const NPPI_INTER_LINEAR: c_int = 2;
pub const NPPI_INTER_CUBIC: c_int = 4;

/// CUDA driver device attributes (cuda.h).
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 3;
const CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK: c_int = 8;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: c_int = 39;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

/// `NppiSize`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NppiSize {
    pub n_width: c_int,
    pub n_height: c_int,
}

/// `NppiRect`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NppiRect {
    pub n_x: c_int,
    pub n_y: c_int,
    pub n_width: c_int,
    pub n_height: c_int,
}

/// `NppStreamContext` (nppdefs.h): application-managed, filled from CUDA
/// driver queries. Passed by value to every `_Ctx` call.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NppStreamContext {
    pub h_stream: *mut c_void,
    pub n_cuda_device_id: c_int,
    pub n_multi_processor_count: c_int,
    pub n_max_threads_per_multi_processor: c_int,
    pub n_max_threads_per_block: c_int,
    // C type is `size_t` (8 bytes on x86_64): a narrower Rust type here would
    // shift every following field and corrupt the context NPP reads.
    pub n_shared_mem_per_block: usize,
    pub n_compute_capability_major: c_int,
    pub n_compute_capability_minor: c_int,
    pub n_stream_flags: u32,
    pub n_reserved0: c_int,
}

// The stream handle is never dereferenced in Rust; it is passed by value to
// NPP (null means the default stream). The struct is only ever copied.
unsafe impl Send for NppStreamContext {}
unsafe impl Sync for NppStreamContext {}

impl NppStreamContext {
    pub fn zeroed() -> Self {
        Self {
            h_stream: std::ptr::null_mut(),
            n_cuda_device_id: 0,
            n_multi_processor_count: 0,
            n_max_threads_per_multi_processor: 0,
            n_max_threads_per_block: 0,
            n_shared_mem_per_block: 0,
            n_compute_capability_major: 0,
            n_compute_capability_minor: 0,
            n_stream_flags: 0,
            n_reserved0: 0,
        }
    }
}

/// `nppiResize_8u_C1R_Ctx` (libnppig).
pub type ResizeC1R = unsafe extern "C" fn(
    p_src: *const u8,
    n_src_step: c_int,
    o_src_size: NppiSize,
    o_src_rect_roi: NppiRect,
    p_dst: *mut u8,
    n_dst_step: c_int,
    o_dst_size: NppiSize,
    o_dst_rect_roi: NppiRect,
    e_interpolation: c_int,
    ctx: NppStreamContext,
) -> c_int;

/// `nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx` (libnppicc).
///
/// `a_twist` is a 3x4 matrix applied as
/// `dst[c] = a_twist[c][0]*Y + a_twist[c][1]*Cb + a_twist[c][2]*Cr + a_twist[c][3]`.
pub type Nv12ToRgbTwist = unsafe extern "C" fn(
    p_src: *const *const u8,
    a_src_step: *const c_int,
    p_dst: *mut u8,
    n_dst_step: c_int,
    o_size_roi: NppiSize,
    a_twist: *const f32,
    ctx: NppStreamContext,
) -> c_int;

/// `nppiResize_8u_P3R_Ctx` (libnppig): resize a planar 3-channel image
/// (steps `[Y, Cb, Cr]`) in a single call.
/// `nppiNV12ToYUV420_8u_P2P3R_Ctx` (libnppicc): semi-planar NV12 to planar
/// Y'CbCr 4:2:0 at the same size. `a_src_step` is `[Y, interleaved UV]`,
/// `a_dst_step` is `[Y, Cb, Cr]`.
pub type Nv12ToPlanar420 = unsafe extern "C" fn(
    p_src: *const *const u8,
    // A single step used for both source planes (see header docs).
    n_src_step: c_int,
    p_dst: *mut *mut u8,
    a_dst_step: *const c_int,
    o_size_roi: NppiSize,
    ctx: NppStreamContext,
) -> c_int;

/// `nppiYCbCr420_8u_P3P2R_Ctx` (libnppicc): planar Y'CbCr 4:2:0 to semi-planar
/// NV12 at the same size. The destination is two separate (pointer, step)
/// pairs — not an array.
pub type Planar420ToNv12 = unsafe extern "C" fn(
    p_src: *const *const u8,
    r_src_step: *const c_int,
    p_dst_y: *mut u8,
    n_dst_y_step: c_int,
    p_dst_cbcr: *mut u8,
    n_dst_cbcr_step: c_int,
    o_size_roi: NppiSize,
    ctx: NppStreamContext,
) -> c_int;

/// `nppiWarpAffine_8u_C1R_Ctx` (libnppig) - single-channel affine warp.
pub type WarpAffineC1R = unsafe extern "C" fn(
    p_src: *const u8,
    n_src_step: c_int,
    o_src_size: NppiSize,
    o_src_rect_roi: NppiRect,
    p_dst: *mut u8,
    n_dst_step: c_int,
    o_dst_size: NppiSize,
    o_dst_rect_roi: NppiRect,
    a_coeffs: *const f32,
    e_interpolation: c_int,
    ctx: NppStreamContext,
) -> c_int;

/// `CU_STREAM_NON_BLOCKING`.
const CU_STREAM_NON_BLOCKING: u32 = 0x01;

/// Resolved NPP entry points and the CUDA stream/context to run on.
///
/// NPP 13 `_Ctx` calls operate on **device** memory: the caller allocates
/// with [`Ffi::mem_alloc`], uploads/downloads with the async copies (all
/// ordered on `stream`), then synchronizes `stream` before reading results.
pub struct Ffi {
    /// Kept alive for the lifetime of the owning `Npp`.
    #[allow(dead_code)]
    libs: Vec<Library>,
    pub resize_c1r: ResizeC1R,
    pub nv12_to_rgb_twist: Nv12ToRgbTwist,
    pub nv12_to_planar420: Nv12ToPlanar420,
    pub planar420_to_nv12: Planar420ToNv12,
    pub warp_affine_c1r: WarpAffineC1R,
    /// The non-blocking stream all NPP calls are enqueued on.
    #[allow(dead_code)]
    stream: *mut c_void,
    /// Primary context kept current (process lifetime).
    pub primary_ctx: *mut c_void,
    pub ctx: NppStreamContext,
}

// The raw pointers are CUDA driver handles never dereferenced from Rust; the
// context and stream live for the process lifetime and are only used through
// the PLT-backed methods below. `Npp` is shared via a `OnceLock` static.
unsafe impl Send for Ffi {}
unsafe impl Sync for Ffi {}

impl Ffi {
    /// Load the NPP libraries, initialize CUDA device 0, make its primary
    /// context current, create a non-blocking stream, and fill in the
    /// `NppStreamContext` from device properties.
    ///
    /// Only succeeds when `build.rs` linked the CUDA driver
    /// (`vacc_npp_cuda_linked`); on other hosts the backend is unavailable
    /// and callers fall back to the software pipeline.
    #[cfg(vacc_npp_cuda_linked)]
    pub fn load() -> Result<Self, NppError> {
        let nppig = open_library(&["libnppig.so.13", "libnppig.so"])?;
        let nppicc = open_library(&["libnppicc.so.13", "libnppicc.so"])?;

        let resize_c1r: ResizeC1R = unsafe { get(&nppig, b"nppiResize_8u_C1R_Ctx")? };
        let warp_affine_c1r: WarpAffineC1R = unsafe { get(&nppig, b"nppiWarpAffine_8u_C1R_Ctx")? };
        let nv12_to_rgb_twist: Nv12ToRgbTwist =
            unsafe { get(&nppicc, b"nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx")? };
        let nv12_to_planar420: Nv12ToPlanar420 =
            unsafe { get(&nppicc, b"nppiNV12ToYUV420_8u_P2P3R_Ctx")? };
        let planar420_to_nv12: Planar420ToNv12 =
            unsafe { get(&nppicc, b"nppiYCbCr420_8u_P3P2R_Ctx")? };

        let (primary_ctx, stream, ctx) = build_cuda()?;

        let ffi = Self {
            libs: vec![nppig, nppicc],
            resize_c1r,
            nv12_to_rgb_twist,
            nv12_to_planar420,
            planar420_to_nv12,
            warp_affine_c1r,
            stream,
            primary_ctx,
            ctx,
        };

        // Probe: confirm the primary context can actually allocate device
        // memory before declaring the backend available. If it cannot (no
        // working GPU/driver), fail so callers fall back to software cleanly
        // instead of silently producing garbage downstream.
        let probe = ffi.mem_alloc(64).map_err(|st| NppError::Status {
            fn_name: "cuMemAlloc_v2 (context probe)",
            status: st,
        })?;
        // Stream-ordered free of the probe buffer.
        ffi.mem_free_async(probe);

        Ok(ffi)
    }

    #[cfg(not(vacc_npp_cuda_linked))]
    pub fn load() -> Result<Self, NppError> {
        Err(NppError::CudaNotLinked)
    }

    // Stub fields for non-CUDA builds (never constructed).
    #[cfg(not(vacc_npp_cuda_linked))]
    pub fn warp_affine_c1r(&self) -> WarpAffineC1R {
        unreachable!()
    }
}

/// Driver-backed memory and stream operations. Every call goes through the
/// PLT (see `cuda_driver`); never route these through function pointers.
#[cfg(vacc_npp_cuda_linked)]
impl Ffi {
    /// Allocate `size` bytes of device memory in the primary context.
    pub fn mem_alloc(&self, size: usize) -> Result<usize, c_int> {
        let mut ptr: usize = 0;
        let st = unsafe { cuda_driver::cuMemAlloc_v2(&mut ptr, size) };
        if st != 0 || ptr == 0 {
            return Err(st);
        }
        Ok(ptr)
    }

    /// Stream-ordered free: enqueued after any prior work on the stream.
    pub fn mem_free_async(&self, ptr: usize) -> c_int {
        unsafe { cuda_driver::cuMemFreeAsync(ptr, self.stream) }
    }

    /// Synchronous free: blocks until all in-flight GPU work completes. Used
    /// when releasing buffers handed to external consumers (inference).
    pub fn mem_free(&self, ptr: usize) -> c_int {
        unsafe { cuda_driver::cuMemFree_v2(ptr) }
    }

    /// Enqueue an async H2D copy on the NPP stream.
    pub fn h2d_async(&self, dst: usize, src: *const c_void, size: usize) -> c_int {
        unsafe { cuda_driver::cuMemcpyHtoDAsync_v2(dst, src, size, self.stream) }
    }

    /// Enqueue an async D2H copy on the NPP stream.
    pub fn d2h_async(&self, dst: *mut c_void, src: usize, size: usize) -> c_int {
        unsafe { cuda_driver::cuMemcpyDtoHAsync_v2(dst, src, size, self.stream) }
    }

    /// Synchronize the NPP stream.
    pub fn stream_sync(&self) -> c_int {
        unsafe { cuda_driver::cuStreamSynchronize(self.stream) }
    }


    /// Make the primary context current on the calling thread.
    pub fn set_current(&self) -> c_int {
        unsafe { cuda_driver::cuCtxSetCurrent(self.primary_ctx) }
    }

    /// `cuCtxGetCurrent` (diagnostics). Returns `(status, ctx, flags)`.
    pub fn get_current(&self) -> (c_int, *mut c_void, c_int) {
        let mut cur: *mut c_void = std::ptr::null_mut();
        let st = unsafe { cuda_driver::cuCtxGetCurrent(&mut cur) };
        (st, cur, 0)
    }
}

/// Stubs for hosts built without a linked CUDA driver. `Ffi` can never be
/// constructed there (`load` fails first), so these are unreachable; they
/// exist only so the call sites compile uniformly.
#[cfg(not(vacc_npp_cuda_linked))]
impl Ffi {
    pub fn mem_alloc(&self, _size: usize) -> Result<usize, c_int> {
        Err(-1)
    }

    pub fn mem_free_async(&self, _ptr: usize) -> c_int {
        -1
    }

    pub fn h2d_async(&self, _dst: usize, _src: *const c_void, _size: usize) -> c_int {
        -1
    }

    pub fn d2h_async(&self, _dst: *mut c_void, _src: usize, _size: usize) -> c_int {
        -1
    }

    pub fn stream_sync(&self) -> c_int {
        -1
    }

    pub fn set_current(&self) -> c_int {
        -1
    }

    pub fn get_current(&self) -> (c_int, *mut c_void, c_int) {
        (-1, std::ptr::null_mut(), 0)
    }
}

fn open_library(names: &[&str]) -> Result<Library, NppError> {
    let mut last_err = String::new();
    for name in names {
        match unsafe { Library::new(name) } {
            Ok(lib) => return Ok(lib),
            Err(e) => last_err = e.to_string(),
        }
    }
    // Also try the CUDA toolkit's lib directory (common install location).
    if let Some(cuda_home) = std::env::var_os("CUDA_HOME")
        .or_else(|| std::env::var_os("CUDA_PATH"))
    {
        let dir = Path::new(&cuda_home).join("lib64");
        for name in names {
            let full = dir.join(name);
            if full.exists() {
                match unsafe { Library::new(&full) } {
                    Ok(lib) => return Ok(lib),
                    Err(e) => last_err = e.to_string(),
                }
            }
        }
    }
    Err(NppError::LibraryNotFound {
        names: names.join(", "),
        detail: last_err,
    })
}

unsafe fn get<T: Copy>(lib: &Library, name: &[u8]) -> Result<T, NppError> {
    let c_name = CString::new(name).unwrap();
    let sym: libloading::Symbol<'_, T> = unsafe { lib.get(c_name.as_bytes_with_nul()) }.map_err(|e| {
        NppError::SymbolMissing {
            symbol: String::from_utf8_lossy(name).into_owned(),
            detail: e.to_string(),
        }
    })?;
    Ok(*sym)
}

/// Initialize CUDA device 0, make its primary context current, create a
/// non-blocking stream, and fill in the `NppStreamContext` from device
/// properties.
///
/// NPP calls with a user stream are asynchronous; the caller must
/// synchronize the returned stream after every call.
#[cfg(vacc_npp_cuda_linked)]
fn build_cuda() -> Result<(*mut c_void, *mut c_void, NppStreamContext), NppError> {
    unsafe {
        if cuda_driver::cuInit(0) != 0 {
            return Err(NppError::CudaInitFailed);
        }
        let mut count: c_int = 0;
        if cuda_driver::cuDeviceGetCount(&mut count) != 0 || count == 0 {
            return Err(NppError::NoCudaDevice);
        }
        let mut primary_ctx: *mut c_void = std::ptr::null_mut();
        if cuda_driver::cuDevicePrimaryCtxRetain(&mut primary_ctx, 0) != 0 {
            return Err(NppError::CudaInitFailed);
        }
        let st_set = cuda_driver::cuCtxSetCurrent(primary_ctx);
        if st_set != 0 {
            return Err(NppError::CudaInitFailed);
        }
        let mut stream: *mut c_void = std::ptr::null_mut();
        if cuda_driver::cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING) != 0 {
            return Err(NppError::CudaInitFailed);
        }

        let attr = |a: c_int| -> Result<c_int, NppError> {
            let mut v: c_int = 0;
            if cuda_driver::cuDeviceGetAttribute(&mut v, a, 0) != 0 {
                return Err(NppError::CudaInitFailed);
            }
            Ok(v)
        };

        let mut ctx = NppStreamContext::zeroed();
        ctx.h_stream = stream;
        ctx.n_cuda_device_id = 0;
        ctx.n_multi_processor_count = attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        ctx.n_max_threads_per_multi_processor = attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?;
        ctx.n_max_threads_per_block = attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?;
        ctx.n_shared_mem_per_block = attr(CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK)? as usize;
        ctx.n_compute_capability_major = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        ctx.n_compute_capability_minor = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        // CU_STREAM_NON_BLOCKING.
        ctx.n_stream_flags = CU_STREAM_NON_BLOCKING;
        Ok((primary_ctx, stream, ctx))
    }
}

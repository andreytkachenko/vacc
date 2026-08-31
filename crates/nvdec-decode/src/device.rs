//! NVDEC device management and CUDA initialization.
//!
//! This module handles:
//!
//! - Loading `libcuda.so` (CUDA Driver API) and `libnvcuvid.so` (Video Codec SDK)
//! - Resolving function symbols at runtime via `libloading`
//! - Creating and managing the CUDA context
//! - Querying decoder capabilities
//!
//! ## Initialization Flow
//!
//! 1. Call [`init_nvdec()`] (or use [`is_available()`] for a check)
//! 2. The CUDA driver API is initialized and a context created
//! 3. NVDEC library functions are resolved
//! 4. Decoder operations can proceed
//!
//! Initialization is idempotent — subsequent calls are no-ops.
//!
//! ## Thread Safety
//!
//! Library handles and function pointers are stored in `OnceLock` singletons,
//! making them safe to access from any thread. The CUDA context is shared
//! via [`cu_ctx_set_current()`], which must be called before each CUDA API
//! invocation on a given thread.

use std::sync::OnceLock;

use libloading::Library;

use crate::error::{NvdecError, NvdecResult};
use crate::ffi::{cudaVideoChromaFormat, cudaVideoCodec, CUDA_SUCCESS, CUVIDDECODECAPS};

/// CUDA memory type (matches `CUmemorytype` in cuda.h).
pub const CU_MEMORYTYPE_HOST: u32 = 1;
pub const CU_MEMORYTYPE_DEVICE: u32 = 2;
pub const CU_MEMORYTYPE_ARRAY: u32 = 3;

/// CUDA 2D memory copy structure (matches CUDA_MEMCPY2D_v2 from cuda.h).
///
/// Describes a 2D rectangular memory copy operation. Total size must be
/// exactly 128 bytes. Field names match the CUDA API naming convention
/// (camelCase) for FFI compatibility.
///
/// # Example
///
/// ```ignore
/// let copy = CUDA_MEMCPY2D {
///     srcMemoryType: CU_MEMORYTYPE_DEVICE,
///     srcDevice: src_ptr,
///     dstMemoryType: CU_MEMORYTYPE_HOST,
///     dstHost: dst_ptr as _,
///     WidthInBytes: width,
///     Height: height,
///     // ... other fields
/// };
/// unsafe { cu_memcpy_2d(&copy) }?;
/// ```
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_snake_case)]
pub struct CUDA_MEMCPY2D {
    /// Source X offset in bytes.
    pub srcXInBytes: u64,
    /// Source Y offset.
    pub srcY: u64,
    /// Source memory type (e.g., [`CU_MEMORYTYPE_DEVICE`]).
    pub srcMemoryType: u32,
    /// Reserved.
    pub _reserved0: u32,
    /// Source host pointer (if `srcMemoryType` is [`CU_MEMORYTYPE_HOST`]).
    pub srcHost: *const std::ffi::c_void,
    /// Source device pointer (if `srcMemoryType` is [`CU_MEMORYTYPE_DEVICE`]).
    pub srcDevice: u64,
    /// Source CUDA array (if `srcMemoryType` is `CU_MEMORYTYPE_ARRAY`).
    pub srcArray: u64,
    /// Source pitch (row stride in bytes).
    pub srcPitch: u64,
    /// Destination X offset in bytes.
    pub dstXInBytes: u64,
    /// Destination Y offset.
    pub dstY: u64,
    /// Destination memory type (e.g., [`CU_MEMORYTYPE_HOST`]).
    pub dstMemoryType: u32,
    /// Reserved.
    pub _reserved1: u32,
    /// Destination host pointer (if `dstMemoryType` is [`CU_MEMORYTYPE_HOST`]).
    pub dstHost: *mut std::ffi::c_void,
    /// Destination device pointer (if `dstMemoryType` is [`CU_MEMORYTYPE_DEVICE`]).
    pub dstDevice: u64,
    /// Destination CUDA array (if `dstMemoryType` is `CU_MEMORYTYPE_ARRAY`).
    pub dstArray: u64,
    /// Destination pitch (row stride in bytes).
    pub dstPitch: u64,
    /// Width of the copy region in bytes.
    pub WidthInBytes: u64,
    /// Height of the copy region in rows.
    pub Height: u64,
}

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::size_of::<CUDA_MEMCPY2D>() == 128);
};

/// CUDA driver API function pointers.
struct CudaFuncs {
    cu_init: unsafe extern "C" fn(u32) -> u32,
    cu_device_get: unsafe extern "C" fn(*mut i32, i32) -> u32,
    cu_ctx_create_v2: unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32, i32) -> u32,
    cu_device_primary_ctx_retain: unsafe extern "C" fn(*mut *mut std::ffi::c_void, i32) -> u32,
    cu_device_primary_ctx_release: unsafe extern "C" fn(i32) -> u32,
    cu_ctx_set_current: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
    cu_ctx_synchronize: unsafe extern "C" fn() -> u32,
    cu_memcpy_2d: unsafe extern "C" fn(*const CUDA_MEMCPY2D) -> u32,
    cu_memcpy_2d_async: unsafe extern "C" fn(*const CUDA_MEMCPY2D, *mut std::ffi::c_void) -> u32,
    cu_memcpy_dtoh:
        unsafe extern "C" fn(*mut std::ffi::c_void, crate::ffi::CUdeviceptr, usize) -> u32,
    cu_stream_create: unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32) -> u32,
    cu_stream_synchronize: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
    cu_stream_destroy: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
    cu_mem_host_alloc: unsafe extern "C" fn(*mut *mut std::ffi::c_void, usize, u32) -> u32,
    cu_mem_free_host: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
    cu_mem_alloc_v2: unsafe extern "C" fn(*mut crate::ffi::CUdeviceptr, usize) -> u32,
    cu_mem_free_v2: unsafe extern "C" fn(crate::ffi::CUdeviceptr) -> u32,
}

/// Loaded NVDEC library functions as raw function pointers.
///
/// Contains resolved function pointers for all NVDEC API calls.
/// Obtained via [`get_funcs()`].
///
/// # Thread Safety
///
/// This struct is accessed through a `OnceLock` singleton, making it
/// safe to read from any thread after initialization.
pub struct NvdecFuncs {
    /// Query decoder capabilities (`cuvidGetDecoderCaps`).
    pub get_decoder_caps: unsafe extern "C" fn(*mut CUVIDDECODECAPS) -> u32,
    /// Create a decoder (`cuvidCreateDecoder`).
    pub create_decoder: unsafe extern "C" fn(
        *mut *mut std::ffi::c_void,
        *const crate::ffi::CUVIDDECODECREATEINFO,
    ) -> u32,
    /// Destroy a decoder (`cuvidDestroyDecoder`).
    pub destroy_decoder: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
    /// Decode a picture (`cuvidDecodePicture`).
    pub decode_picture: unsafe extern "C" fn(
        *mut std::ffi::c_void,
        *const crate::ffi::CUVIDPICPARAMS,
        *const crate::ffi::CUVIDPROCPARAMS,
    ) -> u32,
    /// Map a video frame for reading (`cuvidMapVideoFrame64`).
    pub map_video_frame64: unsafe extern "C" fn(
        *mut std::ffi::c_void,
        i32,
        *mut u64,
        *mut u32,
        *const crate::ffi::CUVIDPROCPARAMS,
    ) -> u32,
    /// Unmap a previously mapped video frame (`cuvidUnmapVideoFrame64`).
    pub unmap_video_frame64: unsafe extern "C" fn(*mut std::ffi::c_void, u64) -> u32,
    /// Reconfigure decoder (optional — may not exist on older drivers).
    pub reconfigure_decoder: Option<
        unsafe extern "C" fn(
            *mut std::ffi::c_void,
            *const crate::ffi::CUVIDRECONFIGUREDECODERINFO,
        ) -> u32,
    >,
    /// Get decode status (`cuvidGetDecodeStatus`).
    pub get_decode_status: unsafe extern "C" fn(
        *mut std::ffi::c_void,
        i32,
        *mut crate::ffi::CUVIDGETDECODESTATUS,
    ) -> u32,
    /// Create a video parser (`cuvidCreateVideoParser`).
    pub create_video_parser: unsafe extern "C" fn(
        *mut *mut std::ffi::c_void,
        *const crate::ffi::CUVIDPARSERPARAMS,
    ) -> u32,
    /// Parse video data (`cuvidParseVideoData`).
    pub parse_video_data: unsafe extern "C" fn(
        *mut std::ffi::c_void,
        *const crate::ffi::CUVIDSOURCEDATAPACKET,
    ) -> u32,
    /// Destroy a video parser (`cuvidDestroyVideoParser`).
    pub destroy_video_parser: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
}

static CUDA_LIB: OnceLock<(Library, CudaFuncs)> = OnceLock::new();
static NVDEC_LIB: OnceLock<(Library, NvdecFuncs)> = OnceLock::new();

/// CUDA context wrapper for thread safety.
struct CtxPtr(*mut std::ffi::c_void);
unsafe impl Send for CtxPtr {}
unsafe impl Sync for CtxPtr {}

static CUDA_CTX: OnceLock<CtxPtr> = OnceLock::new();

/// Initialize CUDA driver API.
fn init_cuda() -> NvdecResult<()> {
    CUDA_LIB
        .set(load_cuda_lib()?)
        .map_err(|_| NvdecError::LibLoadError("CUDA already initialized".to_string()))?;

    let (_, funcs) = CUDA_LIB.get().unwrap();

    // Initialize CUDA
    let result = unsafe { (funcs.cu_init)(0) };
    if result != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!(
            "cuInit failed with error {}",
            result
        )));
    }

    // Get first GPU device
    let mut device: i32 = 0;
    let result = unsafe { (funcs.cu_device_get)(&mut device, 0) };
    if result != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!(
            "cuDeviceGet failed with error {}",
            result
        )));
    }

    // Create CUDA context
    let mut ctx: *mut std::ffi::c_void = std::ptr::null_mut();
    let result = unsafe { (funcs.cu_ctx_create_v2)(&mut ctx, 0, device) };
    if result != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!(
            "cuCtxCreate_v2 failed with error {}",
            result
        )));
    }

    // Set as current context
    let result = unsafe { (funcs.cu_ctx_set_current)(ctx) };
    if result != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!(
            "cuCtxSetCurrent failed with error {}",
            result
        )));
    }

    CUDA_CTX.set(CtxPtr(ctx)).ok();

    Ok(())
}

/// Load CUDA driver library.
fn load_cuda_lib() -> NvdecResult<(Library, CudaFuncs)> {
    let lib_paths = [
        "libcuda.so",
        "libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    ];

    let lib = lib_paths
        .iter()
        .find_map(|path| unsafe { Library::new(path).ok() })
        .ok_or_else(|| {
            NvdecError::LibLoadError("Failed to load libcuda.so from any known path".into())
        })?;

    unsafe {
        let cu_init = *lib
            .get::<unsafe extern "C" fn(u32) -> u32>(b"cuInit_v2\0")
            .or_else(|_| lib.get::<unsafe extern "C" fn(u32) -> u32>(b"cuInit\0"))
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuInit: {}", e)))?;

        let cu_device_get = *lib
            .get::<unsafe extern "C" fn(*mut i32, i32) -> u32>(b"cuDeviceGet_v2\0")
            .or_else(|_| lib.get::<unsafe extern "C" fn(*mut i32, i32) -> u32>(b"cuDeviceGet\0"))
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuDeviceGet: {}", e))
            })?;

        let cu_ctx_create_v2 = *lib
            .get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32, i32) -> u32>(
                b"cuCtxCreate_v2\0",
            )
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuCtxCreate_v2: {}", e))
            })?;
        let cu_device_primary_ctx_retain = *lib
            .get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, i32) -> u32>(
                b"cuDevicePrimaryCtxRetain\0",
            )
            .map_err(|e| {
                NvdecError::LibLoadError(format!(
                    "Failed to resolve cuDevicePrimaryCtxRetain: {}",
                    e
                ))
            })?;
        let cu_device_primary_ctx_release = *lib
            .get::<unsafe extern "C" fn(i32) -> u32>(b"cuDevicePrimaryCtxRelease\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!(
                    "Failed to resolve cuDevicePrimaryCtxRelease: {}",
                    e
                ))
            })?;
        let cu_ctx_set_current = *lib
            .get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>(b"cuCtxSetCurrent\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuCtxSetCurrent: {}", e))
            })?;

        let cu_ctx_synchronize = *lib
            .get::<unsafe extern "C" fn() -> u32>(b"cuCtxSynchronize\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuCtxSynchronize: {}", e))
            })?;

        let cu_memcpy_2d = *lib
            .get::<unsafe extern "C" fn(*const CUDA_MEMCPY2D) -> u32>(b"cuMemcpy2D_v2\0")
            .or_else(|_| {
                lib.get::<unsafe extern "C" fn(*const CUDA_MEMCPY2D) -> u32>(b"cuMemcpy2D\0")
            })
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuMemcpy2D: {}", e))
            })?;

        let cu_memcpy_2d_async = *lib
            .get::<unsafe extern "C" fn(*const CUDA_MEMCPY2D, *mut std::ffi::c_void) -> u32>(
                b"cuMemcpy2DAsync_v2\0",
            )
            .or_else(|_| {
                lib.get::<unsafe extern "C" fn(*const CUDA_MEMCPY2D, *mut std::ffi::c_void) -> u32>(
                    b"cuMemcpy2DAsync\0",
                )
            })
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuMemcpy2DAsync: {}", e))
            })?;

        let cu_stream_create = *lib
            .get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32) -> u32>(
                b"cuStreamCreate\0",
            )
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuStreamCreate: {}", e))
            })?;

        let cu_stream_synchronize = *lib
            .get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>(b"cuStreamSynchronize\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuStreamSynchronize: {}", e))
            })?;

        let cu_stream_destroy = *lib
            .get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>(b"cuStreamDestroy\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuStreamDestroy: {}", e))
            })?;

        let cu_mem_host_alloc = *lib
            .get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, usize, u32) -> u32>(
                b"cuMemHostAlloc\0",
            )
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuMemHostAlloc: {}", e))
            })?;

        let cu_mem_free_host = *lib
            .get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>(b"cuMemFreeHost\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuMemFreeHost: {}", e))
            })?;

        let cu_mem_alloc_v2 = *lib
            .get::<unsafe extern "C" fn(*mut crate::ffi::CUdeviceptr, usize) -> u32>(b"cuMemAlloc_v2\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuMemAlloc_v2: {}", e))
            })?;

        let cu_mem_free_v2 = *lib
            .get::<unsafe extern "C" fn(crate::ffi::CUdeviceptr) -> u32>(b"cuMemFree_v2\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuMemFree_v2: {}", e))
            })?;

        let cu_memcpy_dtoh = *lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void, crate::ffi::CUdeviceptr, usize) -> u32>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void, crate::ffi::CUdeviceptr, usize) -> u32>(b"cuMemcpyDtoH\0"))
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuMemcpyDtoH: {}", e)))?;

        let funcs = CudaFuncs {
            cu_init,
            cu_device_get,
            cu_ctx_create_v2,
            cu_device_primary_ctx_retain,
            cu_device_primary_ctx_release,
            cu_ctx_set_current,
            cu_ctx_synchronize,
            cu_memcpy_2d,
            cu_memcpy_2d_async,
            cu_memcpy_dtoh,
            cu_stream_create,
            cu_stream_synchronize,
            cu_stream_destroy,
            cu_mem_host_alloc,
            cu_mem_free_host,
            cu_mem_alloc_v2,
            cu_mem_free_v2,
        };

        Ok((lib, funcs))
    }
}

/// Initialize the NVDEC library.
///
/// Loads `libcuda.so`, creates a CUDA context on the first GPU device,
/// then loads `libnvcuvid.so` and resolves all function symbols.
///
/// This function is idempotent — calling it multiple times is safe.
/// Subsequent calls return `Ok(())` without reinitializing.
///
/// # Errors
///
/// * [`NvdecError::LibLoadError`] — Cannot find `libcuda.so` or `libnvcuvid.so`
/// * [`NvdecError::CudaError`] — CUDA initialization failed (no GPU, driver issue)
///
/// # Example
///
/// ```no_run
/// use nvdec_decode::init_nvdec;
///
/// init_nvdec().expect("Failed to initialize NVDEC");
/// ```
pub fn init_nvdec() -> NvdecResult<()> {
    // Initialize CUDA first
    if CUDA_LIB.get().is_none() {
        init_cuda()?;
    }

    // Load NVDEC
    if NVDEC_LIB.get().is_none() {
        NVDEC_LIB
            .set(load_nvdec_lib()?)
            .map_err(|_| NvdecError::LibLoadError("NVDEC already initialized".to_string()))?;
    }

    Ok(())
}

/// Load and resolve NVDEC library.
fn load_nvdec_lib() -> NvdecResult<(Library, NvdecFuncs)> {
    // Try common paths for libnvcuvid.so
    let lib_paths = [
        "libnvcuvid.so",
        "/usr/lib/libnvcuvid.so",
        "/usr/lib/x86_64-linux-gnu/libnvcuvid.so",
        "/usr/local/cuda/lib64/stubs/libnvcuvid.so",
    ];

    let lib = lib_paths
        .iter()
        .find_map(|path| unsafe { Library::new(path).ok() })
        .ok_or_else(|| {
            NvdecError::LibLoadError("Failed to load libnvcuvid.so from any known path".into())
        })?;

    unsafe {
        let get_decoder_caps = *lib
            .get::<unsafe extern "C" fn(*mut CUVIDDECODECAPS) -> u32>(b"cuvidGetDecoderCaps\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuvidGetDecoderCaps: {}", e))
            })?;

        let create_decoder = *lib
            .get::<unsafe extern "C" fn(
                *mut *mut std::ffi::c_void,
                *const crate::ffi::CUVIDDECODECREATEINFO,
            ) -> u32>(b"cuvidCreateDecoder\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuvidCreateDecoder: {}", e))
            })?;

        let destroy_decoder = *lib
            .get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>(b"cuvidDestroyDecoder\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuvidDestroyDecoder: {}", e))
            })?;

        let decode_picture =
            *lib.get::<unsafe extern "C" fn(
                *mut std::ffi::c_void,
                *const crate::ffi::CUVIDPICPARAMS,
                *const crate::ffi::CUVIDPROCPARAMS,
            ) -> u32>(b"cuvidDecodePicture\0")
                .map_err(|e| {
                    NvdecError::LibLoadError(format!("Failed to resolve cuvidDecodePicture: {}", e))
                })?;

        let map_video_frame64 = *lib
            .get::<unsafe extern "C" fn(
                *mut std::ffi::c_void,
                i32,
                *mut u64,
                *mut u32,
                *const crate::ffi::CUVIDPROCPARAMS,
            ) -> u32>(b"cuvidMapVideoFrame64\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuvidMapVideoFrame64: {}", e))
            })?;

        let unmap_video_frame64 = *lib
            .get::<unsafe extern "C" fn(*mut std::ffi::c_void, u64) -> u32>(
                b"cuvidUnmapVideoFrame64\0",
            )
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuvidUnmapVideoFrame64: {}", e))
            })?;

        // Optional: cuvidReconfigureDecoder (available since Video Codec SDK 11.3)
        let reconfigure_decoder = lib
            .get::<unsafe extern "C" fn(
                *mut std::ffi::c_void,
                *const crate::ffi::CUVIDRECONFIGUREDECODERINFO,
            ) -> u32>(b"cuvidReconfigureDecoder\0")
            .ok()
            .map(|sym| *sym);

        let get_decode_status = *lib
            .get::<unsafe extern "C" fn(
                *mut std::ffi::c_void,
                i32,
                *mut crate::ffi::CUVIDGETDECODESTATUS,
            ) -> u32>(b"cuvidGetDecodeStatus\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuvidGetDecodeStatus: {}", e))
            })?;

        let create_video_parser = *lib
            .get::<unsafe extern "C" fn(
                *mut *mut std::ffi::c_void,
                *const crate::ffi::CUVIDPARSERPARAMS,
            ) -> u32>(b"cuvidCreateVideoParser\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!(
                    "Failed to resolve cuvidCreateVideoParser: {}",
                    e
                ))
            })?;

        let parse_video_data = *lib
            .get::<unsafe extern "C" fn(
                *mut std::ffi::c_void,
                *const crate::ffi::CUVIDSOURCEDATAPACKET,
            ) -> u32>(b"cuvidParseVideoData\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuvidParseVideoData: {}", e))
            })?;

        let destroy_video_parser = *lib
            .get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>(
                b"cuvidDestroyVideoParser\0",
            )
            .map_err(|e| {
                NvdecError::LibLoadError(format!(
                    "Failed to resolve cuvidDestroyVideoParser: {}",
                    e
                ))
            })?;

        let funcs = NvdecFuncs {
            get_decoder_caps,
            create_decoder,
            destroy_decoder,
            decode_picture,
            map_video_frame64,
            unmap_video_frame64,
            reconfigure_decoder,
            get_decode_status,
            create_video_parser,
            parse_video_data,
            destroy_video_parser,
        };

        Ok((lib, funcs))
    }
}

/// Get the loaded NVDEC function pointers.
///
/// Returns a reference to the [`NvdecFuncs`] struct containing resolved
/// function pointers for all NVDEC API calls.
///
/// # Errors
///
/// Returns [`NvdecError::LibLoadError`] if [`init_nvdec()`] has not been
/// called yet.
pub fn get_funcs() -> NvdecResult<&'static NvdecFuncs> {
    NVDEC_LIB.get().map(|(_, funcs)| funcs).ok_or_else(|| {
        NvdecError::LibLoadError("NVDEC not initialized. Call init_nvdec() first.".into())
    })
}

/// Perform a synchronous 2D memory copy.
///
/// Wraps `cuMemcpy2D` from the CUDA Driver API. Copies a 2D rectangular
/// region of memory as described by the [`CUDA_MEMCPY2D`] structure.
///
/// # Safety
///
/// The `ptr` must point to a valid [`CUDA_MEMCPY2D`] structure with valid
/// memory pointers and sizes. Source and destination memory regions must
/// not overlap.
///
/// # Errors
///
/// Returns [`NvdecError::LibLoadError`] if CUDA is not initialized.
pub unsafe fn cu_memcpy_2d(ptr: *const CUDA_MEMCPY2D) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    Ok(unsafe { (funcs.cu_memcpy_2d)(ptr) })
}

/// Synchronize the CUDA context.
///
/// Blocks until all preceding CUDA operations in the current context
/// have completed.
///
/// # Errors
///
/// Returns [`NvdecError::LibLoadError`] if CUDA is not initialized.
pub fn cu_ctx_synchronize() -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    Ok(unsafe { (funcs.cu_ctx_synchronize)() })
}

/// Set the CUDA context as current for the calling thread.
///
/// CUDA contexts are thread-local. This function sets the global context
/// (created during [`init_nvdec()`]) as the current context for the
/// calling thread. Must be called before any CUDA API invocation.
///
/// # Errors
///
/// Returns [`NvdecError::LibLoadError`] if the CUDA context has not been
/// initialized.
pub fn cu_ctx_set_current() -> NvdecResult<u32> {
    let ctx = CUDA_CTX
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA context not initialized".into()))?;
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    Ok(unsafe { (funcs.cu_ctx_set_current)(ctx.0) })
}

/// Check if NVDEC is available on the system.
///
/// Attempts to initialize NVDEC and returns `true` if successful.
/// This is a non-destructive check — calling [`init_nvdec()`] afterward
/// will succeed if this returns `true`.
///
/// Returns `false` if:
/// - NVIDIA GPU not found
/// - `libcuda.so` or `libnvcuvid.so` not installed
/// - CUDA driver API initialization fails
///
/// # Example
///
/// ```no_run
/// use nvdec_decode::is_available;
///
/// if is_available() {
///     println!("NVDEC is available");
/// } else {
///     println!("Falling back to software decoder");
/// }
/// ```
pub fn is_available() -> bool {
    init_nvdec().is_ok()
}

/// Query decoder capabilities for a specific codec and format.
///
/// Calls `cuvidGetDecoderCaps` to retrieve hardware decoder capabilities
/// for the given codec, chroma format, and bit depth.
///
/// # Arguments
///
/// * `codec` — Video codec to query (e.g., `cudaVideoCodec_H264`)
/// * `chroma_format` — Chroma subsampling format
/// * `bit_depth_minus8` — Bit depth minus 8 (0 = 8-bit, 2 = 10-bit, 4 = 12-bit)
///
/// # Returns
///
/// A [`CUVIDDECODECAPS`] struct with capability information, including:
/// - `bIsSupported` — Whether the codec/format is supported
/// - `nMaxWidth` / `nMaxHeight` — Maximum supported resolution
/// - `nMaxMBCount` — Maximum macroblock count
/// - `nOutputFormatMask` — Supported output format bitmask
///
/// # Errors
///
/// Returns [`NvdecError::CudaError`] if the query fails.
///
/// # Example
///
/// ```no_run
/// use nvdec_decode::{query_decoder_caps, init_nvdec};
/// use nvdec_decode::ffi::{cudaVideoCodec, cudaVideoChromaFormat};
///
/// init_nvdec().unwrap();
/// let caps = query_decoder_caps(
///     cudaVideoCodec::cudaVideoCodec_H264,
///     cudaVideoChromaFormat::cudaVideoChromaFormat_420,
///     0, // 8-bit
/// ).unwrap();
///
/// if caps.bIsSupported != 0 {
///     println!("H.264 decode supported, max {}x{}", caps.nMaxWidth, caps.nMaxHeight);
/// }
/// ```
pub fn query_decoder_caps(
    codec: cudaVideoCodec,
    chroma_format: cudaVideoChromaFormat,
    bit_depth_minus8: u32,
) -> NvdecResult<CUVIDDECODECAPS> {
    let funcs = get_funcs()?;

    let mut caps = CUVIDDECODECAPS {
        eCodecType: codec,
        eChromaFormat: chroma_format,
        nBitDepthMinus8: bit_depth_minus8,
        reserved1: [0; 3],
        bIsSupported: 0,
        nNumNVDECs: 0,
        nOutputFormatMask: 0,
        nMaxWidth: 0,
        nMaxHeight: 0,
        nMaxMBCount: 0,
        nMinWidth: 0,
        nMinHeight: 0,
        bIsHistogramSupported: 0,
        nCounterBitDepth: 0,
        nMaxHistogramBins: 0,
        reserved3: [0; 10],
    };

    let result = unsafe { (funcs.get_decoder_caps)(&mut caps) };

    if result != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!(
            "cuvidGetDecoderCaps failed with error {}",
            result
        )));
    }

    Ok(caps)
}

/// Check if a codec is supported by the hardware decoder.
///
/// Queries decoder capabilities for the given codec with 4:2:0 chroma
/// subsampling and 8-bit depth. Returns `true` if the codec is supported.
///
/// # Example
///
/// ```no_run
/// use nvdec_decode::{is_codec_supported, init_nvdec};
/// use nvdec_decode::ffi::cudaVideoCodec;
///
/// init_nvdec().unwrap();
/// if is_codec_supported(cudaVideoCodec::cudaVideoCodec_H264) {
///     println!("H.264 is supported");
/// }
/// ```
pub fn is_codec_supported(codec: cudaVideoCodec) -> bool {
    let caps = query_decoder_caps(codec, cudaVideoChromaFormat::cudaVideoChromaFormat_420, 0);
    caps.map(|c| c.bIsSupported != 0).unwrap_or(false)
}

/// Create a CUDA stream for asynchronous operations.
///
/// Creates a stream with default flags (`CU_STREAM_DEFAULT`).
///
/// # Returns
///
/// A raw pointer to the stream handle. Pass to [`cu_stream_synchronize`]
/// and [`cu_stream_destroy`] for lifecycle management.
///
/// # Errors
///
/// Returns [`NvdecError::CudaError`] if stream creation fails.
pub fn cu_stream_create() -> NvdecResult<*mut std::ffi::c_void> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    let mut stream: *mut std::ffi::c_void = std::ptr::null_mut();
    // CU_STREAM_DEFAULT = 0
    let result = unsafe { (funcs.cu_stream_create)(&mut stream, 0) };
    if result != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!(
            "cuStreamCreate failed with error {}",
            result
        )));
    }
    Ok(stream)
}

/// Synchronize a CUDA stream.
///
/// # Safety
/// The `stream` must be a valid stream handle returned by [`cu_stream_create`].
pub unsafe fn cu_stream_synchronize(stream: *mut std::ffi::c_void) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    Ok(unsafe { (funcs.cu_stream_synchronize)(stream) })
}

/// Destroy a CUDA stream.
///
/// # Safety
/// The `stream` must be a valid stream handle returned by [`cu_stream_create`]
/// and must not be used after this call.
pub unsafe fn cu_stream_destroy(stream: *mut std::ffi::c_void) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    Ok(unsafe { (funcs.cu_stream_destroy)(stream) })
}

/// Allocate pinned (page-locked) host memory.
///
/// Allocates host memory that is directly accessible by the GPU.
/// Uses `CU_MEMHOSTALLOC_PORTABLE` flag so the memory is accessible
/// from any CUDA context.
///
/// # Arguments
///
/// * `size` — Number of bytes to allocate
///
/// # Returns
///
/// A pointer to the allocated memory. Free with [`cu_mem_free_host`].
///
/// # Errors
///
/// Returns [`NvdecError::CudaError`] if allocation fails.
pub fn cu_mem_host_alloc(size: usize) -> NvdecResult<*mut std::ffi::c_void> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    // CU_MEMHOSTALLOC_PORTABLE = 0x01
    let result = unsafe { (funcs.cu_mem_host_alloc)(&mut ptr, size, 0x01) };
    if result != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!(
            "cuMemHostAlloc failed with error {}",
            result
        )));
    }
    Ok(ptr)
}

/// Free pinned (page-locked) host memory.
///
/// # Safety
///
/// The `ptr` must be a valid pointer returned by [`cu_mem_host_alloc`]
/// and must not be used after this call. Calling with a null or invalid
/// pointer causes undefined behavior.
pub unsafe fn cu_mem_free_host(ptr: *mut std::ffi::c_void) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    Ok(unsafe { (funcs.cu_mem_free_host)(ptr) })
}

/// Allocate device memory.
///
/// # Errors
///
/// Returns [`NvdecError::CudaError`] if allocation fails.
pub fn cu_mem_alloc_device(size: usize) -> NvdecResult<crate::ffi::CUdeviceptr> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    let mut ptr: crate::ffi::CUdeviceptr = 0;
    let result = unsafe { (funcs.cu_mem_alloc_v2)(&mut ptr, size) };
    if result != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!(
            "cuMemAlloc_v2 failed with error {}",
            result
        )));
    }
    Ok(ptr)
}

/// Free device memory allocated with [`cu_mem_alloc_device`].
pub unsafe fn cu_mem_free_device(ptr: crate::ffi::CUdeviceptr) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    Ok(unsafe { (funcs.cu_mem_free_v2)(ptr) })
}

/// Perform an asynchronous 2D memory copy on a CUDA stream.
///
/// Wraps `cuMemcpy2DAsync` from the CUDA Driver API. The copy is
/// enqueued on the given stream and completes asynchronously.
///
/// # Safety
///
/// The `copy_params` must point to a valid [`CUDA_MEMCPY2D`] structure.
/// The `stream` must be a valid stream handle returned by [`cu_stream_create`].
///
/// # Errors
///
/// Returns [`NvdecError::LibLoadError`] if CUDA is not initialized.
pub unsafe fn cu_memcpy_2d_async(
    copy_params: *const CUDA_MEMCPY2D,
    stream: *mut std::ffi::c_void,
) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    Ok(unsafe { (funcs.cu_memcpy_2d_async)(copy_params, stream) })
}

/// Perform a simple device-to-host memory copy.
///
/// Wraps `cuMemcpyDtoH` from the CUDA Driver API. Copies `size` bytes
/// from GPU device memory to host memory.
///
/// # Safety
///
/// The `dst` must point to a valid host memory region of at least `size`
/// bytes. The `src` must be a valid CUDA device pointer with at least
/// `size` bytes of accessible memory.
///
/// # Errors
///
/// Returns [`NvdecError::LibLoadError`] if CUDA is not initialized.
pub unsafe fn cu_memcpy_dtoh(
    dst: *mut std::ffi::c_void,
    src: crate::ffi::CUdeviceptr,
    size: usize,
) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    Ok(unsafe { (funcs.cu_memcpy_dtoh)(dst, src, size) })
}

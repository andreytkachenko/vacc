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

use std::sync::{Arc, OnceLock};

use libloading::Library;

use crate::error::{NvdecError, NvdecResult};
use crate::ffi::{CUDA_SUCCESS, CUVIDDECODECAPS, cudaVideoChromaFormat, cudaVideoCodec};

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
    _cu_ctx_create_v2: unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32, i32) -> u32,
    _cu_device_primary_ctx_retain: unsafe extern "C" fn(*mut *mut std::ffi::c_void, i32) -> u32,
    _cu_device_primary_ctx_release: unsafe extern "C" fn(i32) -> u32,
    cu_ctx_set_current: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
    cu_ctx_synchronize: unsafe extern "C" fn() -> u32,
    cu_memcpy_2d: unsafe extern "C" fn(*const CUDA_MEMCPY2D) -> u32,
    cu_memcpy_2d_async: unsafe extern "C" fn(*const CUDA_MEMCPY2D, *mut std::ffi::c_void) -> u32,
    cu_memcpy_dtoh:
        unsafe extern "C" fn(*mut std::ffi::c_void, crate::ffi::CUdeviceptr, usize) -> u32,
    cu_stream_create: unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32) -> u32,
    cu_stream_synchronize: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
    cu_stream_destroy: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
    cu_event_create: unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32) -> u32,
    cu_event_record: unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> u32,
    cu_event_query: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
    cu_event_synchronize: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
    cu_stream_wait_event:
        unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void, u32) -> u32,
    cu_event_destroy: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
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

/// Device index of the shared primary context (needed to re-retain it after
/// an in-process reset, see [`heal_primary_ctx`]).
static CUDA_DEVICE: OnceLock<i32> = OnceLock::new();

/// Serializes one-time library/context initialization so concurrent
/// initializers don't race the `OnceLock::set` calls.
static INIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Bind the shared CUDA context to the calling thread (contexts are
/// thread-local). Errors if initialization has not completed.
fn bind_current_context() -> NvdecResult<()> {
    bind_primary_ctx()
}

/// Initialize CUDA driver API (idempotent; thread-safe).
///
/// Binds the shared context to the calling thread. A thread that finds the
/// context already initialized (by another thread) simply re-binds it.
fn init_cuda() -> NvdecResult<()> {
    if CUDA_CTX.get().is_some() {
        return bind_current_context();
    }

    let _init_lock = INIT_LOCK.lock().unwrap();
    if CUDA_CTX.get().is_some() {
        // Lost the race: another thread completed initialization while we
        // waited on the lock. Bind its context to this thread.
        return bind_current_context();
    }

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
    CUDA_DEVICE.set(device).ok();

    // Use the device's PRIMARY context, shared with every other CUDA user in
    // this process (cudarc, ORT). A private context makes cross-context
    // synchronization unreliable on current drivers: stream waits on foreign
    // streams and enqueues against foreign streams can silently fail to
    // order, which corrupts pipeline handoffs. Sharing the primary context
    // makes all events/streams same-context and stream-ordered waits reliable.
    let mut ctx: *mut std::ffi::c_void = std::ptr::null_mut();
    let result = unsafe { (funcs._cu_device_primary_ctx_retain)(&mut ctx, device) };
    if result != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!(
            "cuDevicePrimaryCtxRetain failed with error {}",
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

        let cu_event_create = *lib
            .get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32) -> u32>(b"cuEventCreate\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuEventCreate: {}", e))
            })?;

        let cu_event_record = *lib
            .get::<unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> u32>(b"cuEventRecord\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuEventRecord: {}", e))
            })?;

        let cu_event_query = *lib
            .get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>(b"cuEventQuery\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuEventQuery: {}", e))
            })?;

        let cu_event_synchronize = *lib
            .get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>(b"cuEventSynchronize\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuEventSynchronize: {}", e))
            })?;

        let cu_stream_wait_event = *lib
            .get::<unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void, u32) -> u32>(b"cuStreamWaitEvent\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuStreamWaitEvent: {}", e))
            })?;

        let cu_event_destroy = *lib
            .get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>(b"cuEventDestroy\0")
            .map_err(|e| {
                NvdecError::LibLoadError(format!("Failed to resolve cuEventDestroy: {}", e))
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
            .get::<unsafe extern "C" fn(*mut crate::ffi::CUdeviceptr, usize) -> u32>(
                b"cuMemAlloc_v2\0",
            )
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
            _cu_ctx_create_v2: cu_ctx_create_v2,
            _cu_device_primary_ctx_retain: cu_device_primary_ctx_retain,
            _cu_device_primary_ctx_release: cu_device_primary_ctx_release,
            cu_ctx_set_current,
            cu_ctx_synchronize,
            cu_memcpy_2d,
            cu_memcpy_2d_async,
            cu_memcpy_dtoh,
            cu_stream_create,
            cu_stream_synchronize,
            cu_stream_destroy,
            cu_event_create,
            cu_event_record,
            cu_event_query,
            cu_event_synchronize,
            cu_stream_wait_event,
            cu_event_destroy,
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
/// use vacc_nvdec_decode::init_nvdec;
///
/// init_nvdec().expect("Failed to initialize NVDEC");
/// ```
pub fn init_nvdec() -> NvdecResult<()> {
    // Initialize CUDA first (idempotent; binds the context to this thread)
    init_cuda()?;

    // Load NVDEC
    if NVDEC_LIB.get().is_none() {
        let _init_lock = INIT_LOCK.lock().unwrap();
        if NVDEC_LIB.get().is_none() {
            NVDEC_LIB
                .set(load_nvdec_lib()?)
                .map_err(|_| NvdecError::LibLoadError("NVDEC already initialized".to_string()))?;
        }
    }

    if std::env::var("VACC_PROBE_CUVID").is_ok() {
        probe_cuvid_caps();
    }

    Ok(())
}

/// Diagnostic (set `VACC_PROBE_CUVID=1`): dump cuvid decoder caps for the
/// codec/chroma/bit-depth combinations relevant to the verification matrix.
fn probe_cuvid_caps() {
    use crate::ffi::{cudaVideoChromaFormat as CF, cudaVideoCodec as CC};
    let combos: &[(&str, cudaVideoCodec, cudaVideoChromaFormat, u32)] = &[
        (
            "H264  8b 420",
            CC::cudaVideoCodec_H264,
            CF::cudaVideoChromaFormat_420,
            0,
        ),
        (
            "H264 10b 420",
            CC::cudaVideoCodec_H264,
            CF::cudaVideoChromaFormat_420,
            2,
        ),
        (
            "H264 8b 422",
            CC::cudaVideoCodec_H264,
            CF::cudaVideoChromaFormat_422,
            0,
        ),
        (
            "H264 8b 444",
            CC::cudaVideoCodec_H264,
            CF::cudaVideoChromaFormat_444,
            0,
        ),
        (
            "HEVC 8b 420",
            CC::cudaVideoCodec_HEVC,
            CF::cudaVideoChromaFormat_420,
            0,
        ),
        (
            "HEVC 10b 420",
            CC::cudaVideoCodec_HEVC,
            CF::cudaVideoChromaFormat_420,
            2,
        ),
        (
            "HEVC 10b 422",
            CC::cudaVideoCodec_HEVC,
            CF::cudaVideoChromaFormat_422,
            2,
        ),
        (
            "HEVC 10b 444",
            CC::cudaVideoCodec_HEVC,
            CF::cudaVideoChromaFormat_444,
            2,
        ),
        (
            "VP9 P0 8b 420",
            CC::cudaVideoCodec_VP9,
            CF::cudaVideoChromaFormat_420,
            0,
        ),
        (
            "VP9 P1 8b 444",
            CC::cudaVideoCodec_VP9,
            CF::cudaVideoChromaFormat_444,
            0,
        ),
        (
            "VP9 P3 10b 444",
            CC::cudaVideoCodec_VP9,
            CF::cudaVideoChromaFormat_444,
            2,
        ),
        (
            "AV1 main 8b",
            CC::cudaVideoCodec_AV1,
            CF::cudaVideoChromaFormat_420,
            0,
        ),
        (
            "AV1 high 10b",
            CC::cudaVideoCodec_AV1,
            CF::cudaVideoChromaFormat_420,
            2,
        ),
    ];
    for (name, codec, chroma, bdm8) in combos {
        match query_decoder_caps(*codec, *chroma, *bdm8) {
            Ok(c) => eprintln!(
                "[PROBE-CUVID] {:<14} supported={} outmask={:#x} max={}x{}",
                name, c.bIsSupported, c.nOutputFormatMask, c.nMaxWidth, c.nMaxHeight
            ),
            Err(e) => eprintln!("[PROBE-CUVID] {:<14} query error: {}", name, e),
        }
    }
}

/// Load and resolve NVDEC library.
fn load_nvdec_lib() -> NvdecResult<(Library, NvdecFuncs)> {
    // Try common paths for libnvcuvid.so
    let lib_paths = [
        "libnvcuvid.so",
        "libnvcuvid.so.1",
        "/usr/lib/libnvcuvid.so",
        "/usr/lib/x86_64-linux-gnu/libnvcuvid.so",
        "/usr/lib/x86_64-linux-gnu/libnvcuvid.so.1",
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

        let decode_picture = *lib
            .get::<unsafe extern "C" fn(
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
                NvdecError::LibLoadError(format!("Failed to resolve cuvidCreateVideoParser: {}", e))
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
            .get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>(b"cuvidDestroyVideoParser\0")
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
    let code = unsafe { (funcs.cu_ctx_set_current)(ctx.0) };
    if code == CUDA_ERROR_DEINITIALIZED {
        return heal_primary_ctx(funcs);
    }
    Ok(code)
}

/// Re-activate the shared primary context after another in-process CUDA user
/// reset it, then re-bind it on the calling thread.
///
/// Observed failure: with NVDEC decoders running and an ORT TensorRT session
/// being created concurrently (device-resident inference pool), the shared
/// primary context can end up inactive — `cuCtxSetCurrent` then fails with
/// [`CUDA_ERROR_DEINITIALIZED`] and every subsequent driver call on vacc
/// resources fails (`cuvidMapVideoFrame64: 208`, event syncs, ...). The
/// primary context handle is stable across retain/release cycles, so a plain
/// re-retain + re-bind restores the binding. NOTE: if the context was truly
/// RESET (not merely deactivated), resources created before the reset are
/// gone; they will then fail with invalid-handle errors, which is loud and
/// diagnosable rather than a silent hang.
fn heal_primary_ctx(funcs: &CudaFuncs) -> NvdecResult<u32> {
    let ctx = CUDA_CTX
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA context not initialized".into()))?;
    let device = CUDA_DEVICE.get().copied().unwrap_or_else(|| {
        let mut d: i32 = 0;
        if unsafe { (funcs.cu_device_get)(&mut d, 0) } != CUDA_SUCCESS {
            0
        } else {
            d
        }
    });
    eprintln!(
        "[NVDEC] shared primary CUDA context inactive (reset by another in-process CUDA user); re-retaining device {device}"
    );
    let mut healed: *mut std::ffi::c_void = std::ptr::null_mut();
    let code = unsafe { (funcs._cu_device_primary_ctx_retain)(&mut healed, device) };
    if code != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!(
            "primary ctx re-retain failed with error {code}"
        )));
    }
    Ok(unsafe { (funcs.cu_ctx_set_current)(ctx.0) })
}

/// Bind the shared primary context to the calling thread, self-healing an
/// inactive (reset) context on the way. Shared by [`bind_current_context`]
/// and [`ensure_ctx`].
fn bind_primary_ctx() -> NvdecResult<()> {
    match cu_ctx_set_current() {
        Ok(code) if code == CUDA_SUCCESS => Ok(()),
        Ok(code) => Err(NvdecError::CudaError(format!(
            "cuCtxSetCurrent failed with error {code}"
        ))),
        Err(e) => Err(e),
    }
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
/// use vacc_nvdec_decode::is_available;
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
/// use vacc_nvdec_decode::{query_decoder_caps, init_nvdec};
/// use vacc_nvdec_decode::ffi::{cudaVideoCodec, cudaVideoChromaFormat};
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
/// use vacc_nvdec_decode::{is_codec_supported, init_nvdec};
/// use vacc_nvdec_decode::ffi::cudaVideoCodec;
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

/// Event creation flag: no timing functionality. Required for an event to be
/// waited on from a stream in a different CUDA context (e.g. a consumer
/// pipeline that runs in its own context).
pub const CU_EVENT_DISABLE_TIMING: u32 = 2;

/// Result of [`cu_event_query`]: the event has not completed yet. Not an
/// error condition — just "still pending".
pub const CUDA_ERROR_NOT_READY: u32 = 600;

/// `cuCtxSetCurrent` on a context that was deactivated/reset by another
/// in-process CUDA user (e.g. an ORT TensorRT session init/teardown).
pub const CUDA_ERROR_DEINITIALIZED: u32 = 4;

/// Create a CUDA event with [`CU_EVENT_DISABLE_TIMING`] (cross-context use,
/// no timing overhead).
///
/// # Errors
///
/// Returns [`NvdecError::CudaError`] if creation fails.
pub fn cu_event_create() -> NvdecResult<*mut std::ffi::c_void> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    let mut event: *mut std::ffi::c_void = std::ptr::null_mut();
    let result = unsafe { (funcs.cu_event_create)(&mut event, CU_EVENT_DISABLE_TIMING) };
    if result != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!(
            "cuEventCreate failed with error {}",
            result
        )));
    }
    Ok(event)
}

/// Record an event on a stream: it is signaled once all work enqueued before
/// this call has completed. Re-recording an event resets it to the pending
/// state immediately (at this host call), so a stream waiting on the event
/// blocks until the NEW record completes.
///
/// # Safety
///
/// `event` and `stream` must be valid handles; the CUDA context must be
/// current for this thread.
pub unsafe fn cu_event_record(event: *mut std::ffi::c_void, stream: *mut std::ffi::c_void) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    Ok(unsafe { (funcs.cu_event_record)(event, stream) })
}

/// Query whether an event has completed without blocking.
///
/// Returns [`CUDA_SUCCESS`] when signaled, [`CUDA_ERROR_NOT_READY`] while
/// still pending.
///
/// # Safety
///
/// `event` must be a valid event handle; the CUDA context must be current
/// for this thread.
pub unsafe fn cu_event_query(event: *mut std::ffi::c_void) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    Ok(unsafe { (funcs.cu_event_query)(event) })
}

/// Block until the event is signaled.
///
/// # Safety
///
/// `event` must be a valid event handle; the CUDA context must be current
/// for this thread.
pub unsafe fn cu_event_synchronize(event: *mut std::ffi::c_void) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    Ok(unsafe { (funcs.cu_event_synchronize)(event) })
}

/// Make a stream wait for an event recorded on (possibly another) stream.
/// If the event is already signaled the call adds no dependency. The event
/// must have been created with [`CU_EVENT_DISABLE_TIMING`] to be used across
/// contexts.
///
/// # Safety
///
/// `stream` and `event` must be valid handles; the CUDA context must be
/// current for this thread.
pub unsafe fn cu_stream_wait_event(
    stream: *mut std::ffi::c_void,
    event: *mut std::ffi::c_void,
) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    Ok(unsafe { (funcs.cu_stream_wait_event)(stream, event, 0) })
}

/// Destroy a CUDA event.
///
/// # Safety
///
/// The `event` must be a valid handle returned by [`cu_event_create`] and
/// must not be in use (no pending records/waits) when destroyed.
pub unsafe fn cu_event_destroy(event: *mut std::ffi::c_void) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB
        .get()
        .ok_or_else(|| NvdecError::LibLoadError("CUDA not initialized".into()))?;
    Ok(unsafe { (funcs.cu_event_destroy)(event) })
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
///
/// # Safety
///
/// `ptr` must be a valid device pointer previously returned by
/// [`cu_mem_alloc_device`] and not yet freed.
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

// ---------------------------------------------------------------------------
// Safe RAII handles
//
// The raw `cu_*` functions above take/return raw driver handles and several
// of them are `unsafe`. These wrappers own their handles, confine every
// unsafe driver call to this crate, and are `Send + Sync` so they can be
// shared across threads (decode thread -> inference pool -> frame drop on
// any worker) without `unsafe impl`s at the call sites.
// ---------------------------------------------------------------------------

/// Bind the process-wide primary CUDA context to the calling thread.
///
/// Driver calls require a current context; frames and their handles may be
/// dropped on any thread, so every public method in this section binds it
/// first. Errors are propagated (a missing library means CUDA never
/// initialized) — `Drop` impls below treat them as best-effort.
fn ensure_ctx() -> NvdecResult<()> {
    bind_primary_ctx()
}

/// Owning CUDA stream handle (RAII).
///
/// The stream is destroyed on drop, after a best-effort synchronize:
/// destroying a stream with work in flight is undefined behavior.
pub struct CuStream {
    ptr: *mut std::ffi::c_void,
}

// Raw driver handle; the stream is only ever touched through this type's
// methods (which bind the context) or by name via [`CuStream::raw`].
unsafe impl Send for CuStream {}
unsafe impl Sync for CuStream {}

impl CuStream {
    /// Create a new default stream.
    pub fn new() -> NvdecResult<Self> {
        cu_stream_create().map(|ptr| Self { ptr })
    }

    /// Raw `CUstream` handle (interop with other CUDA toolkits). Valid while
    /// this stream is alive.
    pub const fn raw(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    /// Block until all work enqueued on this stream has completed.
    pub fn synchronize(&self) -> NvdecResult<()> {
        ensure_ctx()?;
        unsafe { cu_stream_synchronize(self.ptr) }.map(|_| ())
    }

    /// Enqueue a stream-ordered wait on an event recorded elsewhere.
    pub fn wait_event(&self, event: &CuEvent) -> NvdecResult<()> {
        ensure_ctx()?;
        unsafe { cu_stream_wait_event(self.ptr, event.ptr) }.map(|_| ())
    }

    /// Record an event on this stream: it is signaled once all work enqueued
    /// before this call has completed.
    pub fn record_event(&self, event: &CuEvent) -> NvdecResult<()> {
        ensure_ctx()?;
        unsafe { cu_event_record(event.ptr, self.ptr) }.map(|_| ())
    }
}

impl Drop for CuStream {
    fn drop(&mut self) {
        let _ = ensure_ctx();
        let _ = unsafe { cu_stream_synchronize(self.ptr) };
        let _ = unsafe { cu_stream_destroy(self.ptr) };
    }
}

/// Owning CUDA event handle (RAII). Created with [`CU_EVENT_DISABLE_TIMING`]
/// so it can be waited on across contexts.
pub struct CuEvent {
    ptr: *mut std::ffi::c_void,
}

unsafe impl Send for CuEvent {}
unsafe impl Sync for CuEvent {}

impl CuEvent {
    /// Create a new event.
    pub fn new() -> NvdecResult<Self> {
        cu_event_create().map(|ptr| Self { ptr })
    }

    /// Raw `CUevent` handle (interop with other CUDA toolkits). Valid while
    /// this event is alive.
    pub const fn raw(&self) -> *const std::ffi::c_void {
        self.ptr as *const std::ffi::c_void
    }

    /// Record on a stream: signaled once all work enqueued before this call
    /// has completed. Re-recording resets the event to pending immediately.
    pub fn record(&self, stream: &CuStream) -> NvdecResult<()> {
        ensure_ctx()?;
        unsafe { cu_event_record(self.ptr, stream.ptr) }.map(|_| ())
    }

    /// Block the host until the event is signaled.
    pub fn synchronize(&self) -> NvdecResult<()> {
        ensure_ctx()?;
        unsafe { cu_event_synchronize(self.ptr) }.map(|_| ())
    }

    /// Non-blocking check: `true` once the event is signaled.
    pub fn is_complete(&self) -> bool {
        matches!(
            ensure_ctx().and_then(|()| unsafe { cu_event_query(self.ptr) }),
            Ok(CUDA_SUCCESS)
        )
    }

    /// A non-owning shared view of this event for
    /// [`vacc_core::gpu::GpuFrame`]s. The frame clones do not extend the
    /// lifetime of the underlying handle: it stays owned by this `CuEvent`.
    pub fn view(&self) -> vacc_core::gpu::GpuEvent {
        vacc_core::gpu::GpuEvent::wrap(Arc::new(CuEventView(self.ptr)))
    }
}

impl Drop for CuEvent {
    fn drop(&mut self) {
        // Callers must have ordered any stream that waited on this event
        // (a pending wait on a destroyed event is undefined behavior).
        let _ = ensure_ctx();
        let _ = unsafe { cu_event_destroy(self.ptr) };
    }
}

/// Non-owning view over a live `CUevent` implementing the backend-agnostic
/// [`vacc_core::gpu::EventHandle`] interface. The event itself is owned by a
/// [`CuEvent`] (e.g. a ring slot barrier or a decoder's copy-event pool).
pub struct CuEventView(*const std::ffi::c_void);

impl CuEventView {
    /// A view over a live `CUevent` owned elsewhere (e.g. an event from a
    /// decoder's copy-event pool). The handle must outlive every use of the
    /// resulting view.
    pub const fn new(event: *const std::ffi::c_void) -> Self {
        Self(event)
    }
}

unsafe impl Send for CuEventView {}
unsafe impl Sync for CuEventView {}

impl vacc_core::gpu::EventHandle for CuEventView {
    fn synchronize(&self) -> Result<(), String> {
        ensure_ctx().map_err(|e| e.to_string())?;
        unsafe { cu_event_synchronize(self.0 as *mut std::ffi::c_void) }
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn is_complete(&self) -> bool {
        matches!(
            ensure_ctx().and_then(|()| unsafe { cu_event_query(self.0 as *mut std::ffi::c_void) }),
            Ok(CUDA_SUCCESS)
        )
    }

    fn raw(&self) -> *const std::ffi::c_void {
        self.0
    }

    fn wait_on_stream(&self, stream: *const std::ffi::c_void) -> Result<(), String> {
        ensure_ctx().map_err(|e| e.to_string())?;
        unsafe { cu_stream_wait_event(stream as *mut std::ffi::c_void, self.0 as *mut std::ffi::c_void) }
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn record_on_stream(&self, stream: *const std::ffi::c_void) -> Result<(), String> {
        ensure_ctx().map_err(|e| e.to_string())?;
        unsafe { cu_event_record(self.0 as *mut std::ffi::c_void, stream as *mut std::ffi::c_void) }
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

/// Owning device memory allocation (RAII). Freed on drop after binding the
/// primary context; the free is best-effort (CUDA may already be torn down
/// by then).
pub struct DeviceBuffer {
    ptr: crate::ffi::CUdeviceptr,
    len: usize,
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    /// Allocate `len` bytes of device memory.
    pub fn alloc(len: usize) -> NvdecResult<Self> {
        cu_mem_alloc_device(len).map(|ptr| Self { ptr, len })
    }

    /// Device pointer of the allocation (`CUdeviceptr`). Valid while this
    /// buffer is alive.
    pub const fn ptr(&self) -> u64 {
        self.ptr as u64
    }

    /// Allocation size in bytes.
    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy the whole allocation to host memory (synchronous).
    pub fn read_to_host(&self, dst: &mut [u8]) -> NvdecResult<()> {
        assert!(dst.len() >= self.len, "host buffer too small");
        copy_device_to_host(&mut dst[..self.len], self.ptr())
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        let _ = ensure_ctx();
        let _ = unsafe { cu_mem_free_device(self.ptr) };
    }
}

/// Owning pinned (page-locked) host memory allocation (RAII).
pub struct PinnedHostBuffer {
    ptr: *mut std::ffi::c_void,
    len: usize,
}

unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

impl PinnedHostBuffer {
    /// Allocate `len` bytes of pinned host memory (portable across contexts).
    pub fn alloc(len: usize) -> NvdecResult<Self> {
        cu_mem_host_alloc(len).map(|ptr| Self { ptr, len })
    }

    /// Allocation size in bytes.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// The allocation as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    /// The allocation as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        let _ = ensure_ctx();
        let _ = unsafe { cu_mem_free_host(self.ptr) };
    }
}

/// Synchronous device-to-host linear copy (binds the primary context first,
/// so it may run on any thread).
pub fn copy_device_to_host(dst: &mut [u8], src: u64) -> NvdecResult<()> {
    ensure_ctx()?;
    let code = unsafe { cu_memcpy_dtoh(dst.as_mut_ptr().cast(), src as crate::ffi::CUdeviceptr, dst.len()) }?;
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(NvdecError::CudaError(format!(
            "cuMemcpyDtoH failed with error {code}"
        )))
    }
}

/// Synchronous device-to-host copy of pitched rows: `rows` rows of
/// `row_bytes` bytes each, taken from device memory at `src` with row stride
/// `src_pitch` (binds the primary context first). `dst` must hold at least
/// `rows * row_bytes` bytes and receives the rows tightly packed.
pub fn copy_device_to_host_pitched(
    dst: &mut [u8],
    src: u64,
    src_pitch: usize,
    row_bytes: usize,
    rows: usize,
) -> NvdecResult<()> {
    assert!(dst.len() >= rows * row_bytes, "host buffer too small");
    ensure_ctx()?;
    let params = CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: CU_MEMORYTYPE_DEVICE,
        _reserved0: 0,
        srcHost: std::ptr::null(),
        srcDevice: src,
        srcArray: 0,
        srcPitch: src_pitch as u64,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: CU_MEMORYTYPE_HOST,
        _reserved1: 0,
        dstHost: dst.as_mut_ptr().cast(),
        dstDevice: 0,
        dstArray: 0,
        dstPitch: row_bytes as u64,
        WidthInBytes: row_bytes as u64,
        Height: rows as u64,
    };
    let code = unsafe { cu_memcpy_2d(&params) }?;
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(NvdecError::CudaError(format!(
            "cuMemcpy2D (device->host) failed with error {code}"
        )))
    }
}

/// A fixed-size ring of device buffers with per-slot reader barriers.
///
/// For asynchronous producers (decoders): the producer copies into slots
/// round-robin; before overwriting slot `i` it host-syncs
/// [`barrier(i)`](Self::barrier), and consumers record that event after
/// their last read of the slot. A never-recorded event blocks forever, so
/// every barrier is armed (recorded + synchronized once) at construction.
pub struct DeviceRing {
    buffers: Vec<DeviceBuffer>,
    barriers: Vec<CuEvent>,
}

impl DeviceRing {
    /// Allocate `slots` buffers of `frame_bytes` each and arm every slot's
    /// reader barrier.
    pub fn new(slots: usize, frame_bytes: usize) -> NvdecResult<Self> {
        let mut buffers = Vec::with_capacity(slots);
        for _ in 0..slots {
            buffers.push(DeviceBuffer::alloc(frame_bytes)?);
        }

        let mut barriers = Vec::with_capacity(slots);
        for _ in 0..slots {
            let ev = CuEvent::new()?;
            // Arm: record once on a temporary stream and wait for it, so the
            // first reuse of the slot does not block on a stale record.
            let st = CuStream::new()?;
            ev.record(&st)?;
            st.synchronize()?;
            barriers.push(ev);
        }

        Ok(Self { buffers, barriers })
    }

    /// Number of slots.
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    /// Device pointer of every slot (round-robin order).
    pub fn ptrs(&self) -> Vec<u64> {
        self.buffers.iter().map(|b| b.ptr()).collect()
    }

    /// Slot index for a ring device pointer.
    pub fn slot_of(&self, ptr: u64) -> Option<usize> {
        self.buffers.iter().position(|b| b.ptr() == ptr)
    }

    /// Reader barrier of a slot (shared view; record it after the last read
    /// of the slot so the producer may reuse it).
    pub fn barrier(&self, slot: usize) -> vacc_core::gpu::GpuEvent {
        self.barriers[slot].view()
    }

    /// Raw `CUevent` per slot (for the decoder's slot-wait protocol).
    pub fn slot_waits_raw(&self) -> Vec<Vec<*const std::ffi::c_void>> {
        self.barriers.iter().map(|e| vec![e.raw()]).collect()
    }
}

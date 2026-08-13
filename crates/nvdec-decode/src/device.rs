//! NVDEC device management.
//!
//! Handles loading of libnvcuvid, CUDA driver API, and basic device operations.

use std::sync::OnceLock;

use libloading::Library;

use crate::error::{NvdecError, NvdecResult};
use crate::ffi::{cudaVideoCodec, cudaVideoChromaFormat, CUVIDDECODECAPS, CUDA_SUCCESS};

/// CUDA memcpy kind
pub const CU_MEMORYTYPE_DEVICE: u32 = 1;
pub const CU_MEMORYTYPE_HOST: u32 = 2;

/// CUDA 2D memory copy structure (matches CUDA_MEMCPY2D_v2 from cuda.h)
/// Total size must be exactly 128 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUDA_MEMCPY2D {
    pub srcXInBytes: u64,
    pub srcY: u64,
    pub srcMemoryType: u32,
    pub _reserved0: u32,
    pub srcHost: *const std::ffi::c_void,
    pub srcDevice: u64,
    pub srcArray: u64,
    pub srcPitch: u64,
    pub dstXInBytes: u64,
    pub dstY: u64,
    pub dstMemoryType: u32,
    pub _reserved1: u32,
    pub dstHost: *mut std::ffi::c_void,
    pub dstDevice: u64,
    pub dstArray: u64,
    pub dstPitch: u64,
    pub WidthInBytes: u64,
    pub Height: u64,
}

#[cfg(target_pointer_width = "64")]
const _: () = { assert!(std::mem::size_of::<CUDA_MEMCPY2D>() == 128); };

/// CUDA driver API function pointers.
struct CudaFuncs {
    cu_init: unsafe extern "C" fn(u32) -> u32,
    cu_device_get: unsafe extern "C" fn(*mut i32, i32) -> u32,
    cu_ctx_create_v2: unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32, i32) -> u32,
    cu_ctx_set_current: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
    cu_ctx_synchronize: unsafe extern "C" fn() -> u32,
    cu_memcpy_2d: unsafe extern "C" fn(*const CUDA_MEMCPY2D) -> u32,
    cu_memcpy_2d_async: unsafe extern "C" fn(*const CUDA_MEMCPY2D, *mut std::ffi::c_void) -> u32,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut std::ffi::c_void, crate::ffi::CUdeviceptr, usize) -> u32,
    cu_stream_create: unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32) -> u32,
    cu_stream_synchronize: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
    cu_stream_destroy: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
    cu_mem_host_alloc: unsafe extern "C" fn(*mut *mut std::ffi::c_void, usize, u32) -> u32,
    cu_mem_free_host: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
}

/// Loaded NVDEC library functions as raw function pointers.
pub struct NvdecFuncs {
    pub get_decoder_caps: unsafe extern "C" fn(*mut CUVIDDECODECAPS) -> u32,
    pub create_decoder: unsafe extern "C" fn(*mut *mut std::ffi::c_void, *const crate::ffi::CUVIDDECODECREATEINFO) -> u32,
    pub destroy_decoder: unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
    pub decode_picture: unsafe extern "C" fn(*mut std::ffi::c_void, *const crate::ffi::CUVIDPICPARAMS) -> u32,
    pub map_video_frame64: unsafe extern "C" fn(*mut std::ffi::c_void, i32, *mut u64, *mut u32, *const crate::ffi::CUVIDPROCPARAMS) -> u32,
    pub unmap_video_frame64: unsafe extern "C" fn(*mut std::ffi::c_void, u64) -> u32,
    // Parser functions
    pub create_video_parser: unsafe extern "C" fn(*mut *mut std::ffi::c_void, *const crate::ffi::CUVIDPARSERPARAMS) -> u32,
    pub parse_video_data: unsafe extern "C" fn(*mut std::ffi::c_void, *const crate::ffi::CUVIDSOURCEDATAPACKET) -> u32,
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
    CUDA_LIB.set(load_cuda_lib()?).map_err(|_| NvdecError::LibLoadError("CUDA already initialized".to_string()))?;
    
    let (_, funcs) = CUDA_LIB.get().unwrap();
    
    // Initialize CUDA
    let result = unsafe { (funcs.cu_init)(0) };
    if result != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!("cuInit failed with error {}", result)));
    }
    
    // Get first GPU device
    let mut device: i32 = 0;
    let result = unsafe { (funcs.cu_device_get)(&mut device, 0) };
    if result != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!("cuDeviceGet failed with error {}", result)));
    }
    
    // Create CUDA context
    let mut ctx: *mut std::ffi::c_void = std::ptr::null_mut();
    let result = unsafe { (funcs.cu_ctx_create_v2)(&mut ctx, 0, device) };
    if result != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!("cuCtxCreate_v2 failed with error {}", result)));
    }
    
    // Set as current context
    let result = unsafe { (funcs.cu_ctx_set_current)(ctx) };
    if result != CUDA_SUCCESS {
        return Err(NvdecError::CudaError(format!("cuCtxSetCurrent failed with error {}", result)));
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
        let cu_init = *lib.get::<unsafe extern "C" fn(u32) -> u32>(b"cuInit_v2\0")
            .or_else(|_| lib.get::<unsafe extern "C" fn(u32) -> u32>(b"cuInit\0"))
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuInit: {}", e)))?;

        let cu_device_get = *lib.get::<unsafe extern "C" fn(*mut i32, i32) -> u32>(b"cuDeviceGet_v2\0")
            .or_else(|_| lib.get::<unsafe extern "C" fn(*mut i32, i32) -> u32>(b"cuDeviceGet\0"))
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuDeviceGet: {}", e)))?;

        let cu_ctx_create_v2 = *lib.get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32, i32) -> u32>(b"cuCtxCreate_v2\0")
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuCtxCreate_v2: {}", e)))?;

        let cu_ctx_set_current = *lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>(b"cuCtxSetCurrent\0")
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuCtxSetCurrent: {}", e)))?;

        let cu_ctx_synchronize = *lib.get::<unsafe extern "C" fn() -> u32>(b"cuCtxSynchronize\0")
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuCtxSynchronize: {}", e)))?;

        let cu_memcpy_2d = *lib.get::<unsafe extern "C" fn(*const CUDA_MEMCPY2D) -> u32>(b"cuMemcpy2D_v2\0")
            .or_else(|_| lib.get::<unsafe extern "C" fn(*const CUDA_MEMCPY2D) -> u32>(b"cuMemcpy2D\0"))
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuMemcpy2D: {}", e)))?;

        let cu_memcpy_2d_async = *lib.get::<unsafe extern "C" fn(*const CUDA_MEMCPY2D, *mut std::ffi::c_void) -> u32>(b"cuMemcpy2DAsync_v2\0")
            .or_else(|_| lib.get::<unsafe extern "C" fn(*const CUDA_MEMCPY2D, *mut std::ffi::c_void) -> u32>(b"cuMemcpy2DAsync\0"))
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuMemcpy2DAsync: {}", e)))?;

        let cu_stream_create = *lib.get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32) -> u32>(b"cuStreamCreate\0")
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuStreamCreate: {}", e)))?;

        let cu_stream_synchronize = *lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>(b"cuStreamSynchronize\0")
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuStreamSynchronize: {}", e)))?;

        let cu_stream_destroy = *lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>(b"cuStreamDestroy\0")
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuStreamDestroy: {}", e)))?;

        let cu_mem_host_alloc = *lib.get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, usize, u32) -> u32>(b"cuMemHostAlloc\0")
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuMemHostAlloc: {}", e)))?;

        let cu_mem_free_host = *lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>(b"cuMemFreeHost\0")
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuMemFreeHost: {}", e)))?;

        let cu_memcpy_dtoh = *lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void, crate::ffi::CUdeviceptr, usize) -> u32>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void, crate::ffi::CUdeviceptr, usize) -> u32>(b"cuMemcpyDtoH\0"))
            .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuMemcpyDtoH: {}", e)))?;

        let funcs = CudaFuncs {
            cu_init,
            cu_device_get,
            cu_ctx_create_v2,
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
        };

        Ok((lib, funcs))
    }
}

/// Initialize the NVDEC library.
///
/// Loads libcuda.so, creates a CUDA context, then loads libnvcuvid.so and resolves function symbols.
pub fn init_nvdec() -> NvdecResult<()> {
    // Initialize CUDA first
    if CUDA_LIB.get().is_none() {
        init_cuda()?;
    }
    
    // Load NVDEC
    if NVDEC_LIB.get().is_none() {
        NVDEC_LIB.set(load_nvdec_lib()?).map_err(|_| NvdecError::LibLoadError("NVDEC already initialized".to_string()))?;
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
        let get_decoder_caps = *lib.get::<
            unsafe extern "C" fn(*mut CUVIDDECODECAPS) -> u32,
        >(b"cuvidGetDecoderCaps\0")
        .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuvidGetDecoderCaps: {}", e)))?;

        let create_decoder = *lib.get::<
            unsafe extern "C" fn(*mut *mut std::ffi::c_void, *const crate::ffi::CUVIDDECODECREATEINFO) -> u32,
        >(b"cuvidCreateDecoder\0")
        .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuvidCreateDecoder: {}", e)))?;

        let destroy_decoder = *lib.get::<
            unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
        >(b"cuvidDestroyDecoder\0")
        .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuvidDestroyDecoder: {}", e)))?;

        let decode_picture = *lib.get::<
            unsafe extern "C" fn(*mut std::ffi::c_void, *const crate::ffi::CUVIDPICPARAMS) -> u32,
        >(b"cuvidDecodePicture\0")
        .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuvidDecodePicture: {}", e)))?;

        let map_video_frame64 = *lib.get::<
            unsafe extern "C" fn(*mut std::ffi::c_void, i32, *mut u64, *mut u32, *const crate::ffi::CUVIDPROCPARAMS) -> u32,
        >(b"cuvidMapVideoFrame64\0")
        .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuvidMapVideoFrame64: {}", e)))?;

        let unmap_video_frame64 = *lib.get::<
            unsafe extern "C" fn(*mut std::ffi::c_void, u64) -> u32,
        >(b"cuvidUnmapVideoFrame64\0")
        .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuvidUnmapVideoFrame64: {}", e)))?;

        // Parser functions
        let create_video_parser = *lib.get::<
            unsafe extern "C" fn(*mut *mut std::ffi::c_void, *const crate::ffi::CUVIDPARSERPARAMS) -> u32,
        >(b"cuvidCreateVideoParser\0")
        .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuvidCreateVideoParser: {}", e)))?;

        let parse_video_data = *lib.get::<
            unsafe extern "C" fn(*mut std::ffi::c_void, *const crate::ffi::CUVIDSOURCEDATAPACKET) -> u32,
        >(b"cuvidParseVideoData\0")
        .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuvidParseVideoData: {}", e)))?;

        let destroy_video_parser = *lib.get::<
            unsafe extern "C" fn(*mut std::ffi::c_void) -> u32,
        >(b"cuvidDestroyVideoParser\0")
        .map_err(|e| NvdecError::LibLoadError(format!("Failed to resolve cuvidDestroyVideoParser: {}", e)))?;

        let funcs = NvdecFuncs {
            get_decoder_caps,
            create_decoder,
            destroy_decoder,
            decode_picture,
            map_video_frame64,
            unmap_video_frame64,
            create_video_parser,
            parse_video_data,
            destroy_video_parser,
        };

        Ok((lib, funcs))
    }
}

/// Get the loaded NVDEC functions.
pub fn get_funcs() -> NvdecResult<&'static NvdecFuncs> {
    NVDEC_LIB
        .get()
        .map(|(_, funcs)| funcs)
        .ok_or_else(|| {
            NvdecError::LibLoadError("NVDEC not initialized. Call init_nvdec() first.".into())
        })
}

/// Get CUDA memcpy2d function.
pub fn cu_memcpy_2d(ptr: *const CUDA_MEMCPY2D) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB.get().ok_or_else(|| {
        NvdecError::LibLoadError("CUDA not initialized".into())
    })?;
    Ok(unsafe { (funcs.cu_memcpy_2d)(ptr) })
}

/// Synchronize CUDA context.
pub fn cu_ctx_synchronize() -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB.get().ok_or_else(|| {
        NvdecError::LibLoadError("CUDA not initialized".into())
    })?;
    Ok(unsafe { (funcs.cu_ctx_synchronize)() })
}

/// Set CUDA context current.
pub fn cu_ctx_set_current() -> NvdecResult<u32> {
    let ctx = CUDA_CTX.get().ok_or_else(|| {
        NvdecError::LibLoadError("CUDA context not initialized".into())
    })?;
    let (_, funcs) = CUDA_LIB.get().ok_or_else(|| {
        NvdecError::LibLoadError("CUDA not initialized".into())
    })?;
    Ok(unsafe { (funcs.cu_ctx_set_current)(ctx.0) })
}

/// Check if NVDEC is available on the system.
pub fn is_available() -> bool {
    init_nvdec().is_ok()
}

/// Query decoder capabilities for a specific codec.
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

/// Check if a codec is supported.
pub fn is_codec_supported(codec: cudaVideoCodec) -> bool {
    let caps = query_decoder_caps(codec, cudaVideoChromaFormat::cudaVideoChromaFormat_420, 0);
    caps.map(|c| c.bIsSupported != 0).unwrap_or(false)
}

/// Create a CUDA stream.
pub fn cu_stream_create() -> NvdecResult<*mut std::ffi::c_void> {
    let (_, funcs) = CUDA_LIB.get().ok_or_else(|| {
        NvdecError::LibLoadError("CUDA not initialized".into())
    })?;
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
pub fn cu_stream_synchronize(stream: *mut std::ffi::c_void) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB.get().ok_or_else(|| {
        NvdecError::LibLoadError("CUDA not initialized".into())
    })?;
    Ok(unsafe { (funcs.cu_stream_synchronize)(stream) })
}

/// Destroy a CUDA stream.
pub fn cu_stream_destroy(stream: *mut std::ffi::c_void) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB.get().ok_or_else(|| {
        NvdecError::LibLoadError("CUDA not initialized".into())
    })?;
    Ok(unsafe { (funcs.cu_stream_destroy)(stream) })
}

/// Allocate pinned (page-locked) host memory.
pub fn cu_mem_host_alloc(size: usize) -> NvdecResult<*mut std::ffi::c_void> {
    let (_, funcs) = CUDA_LIB.get().ok_or_else(|| {
        NvdecError::LibLoadError("CUDA not initialized".into())
    })?;
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

/// Free pinned host memory.
pub fn cu_mem_free_host(ptr: *mut std::ffi::c_void) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB.get().ok_or_else(|| {
        NvdecError::LibLoadError("CUDA not initialized".into())
    })?;
    Ok(unsafe { (funcs.cu_mem_free_host)(ptr) })
}

/// Async 2D memory copy using a stream.
pub fn cu_memcpy_2d_async(copy_params: *const CUDA_MEMCPY2D, stream: *mut std::ffi::c_void) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB.get().ok_or_else(|| {
        NvdecError::LibLoadError("CUDA not initialized".into())
    })?;
    Ok(unsafe { (funcs.cu_memcpy_2d_async)(copy_params, stream) })
}

/// Simple device-to-host memory copy.
pub fn cu_memcpy_dtoh(dst: *mut std::ffi::c_void, src: crate::ffi::CUdeviceptr, size: usize) -> NvdecResult<u32> {
    let (_, funcs) = CUDA_LIB.get().ok_or_else(|| {
        NvdecError::LibLoadError("CUDA not initialized".into())
    })?;
    Ok(unsafe { (funcs.cu_memcpy_dtoh)(dst, src, size) })
}

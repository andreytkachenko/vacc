//! Minimal CUDA driver API wrapper for NVENC (dynamic loading via libloading).
//!
//! Mirrors the loading pattern of `vacc-nvdec-decode`: resolve `_v2` symbol
//! variants first, fall back to unversioned names.

use std::ffi::c_void;

use libloading::Library;

use crate::Error;
use crate::Result;

/// Opaque CUDA device pointer (64-bit GPU virtual address).
pub type CUdeviceptr = u64;

type CUresult = u32;

const CUDA_SUCCESS: CUresult = 0;

fn check(res: CUresult, what: &str) -> Result<()> {
    if res == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda(format!("{what} failed with CUDA error {res}")))
    }
}

/// A CUDA context backed by the primary context of one device, plus the
/// driver library handle keeping the symbols alive.
pub struct Cuda {
    #[allow(dead_code)]
    lib: Library,
    ctx: *mut c_void,
    cu_mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, u64) -> CUresult,
    cu_mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    cu_memcpy_htod: unsafe extern "C" fn(CUdeviceptr, *const c_void, u64) -> CUresult,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CUdeviceptr, u64) -> CUresult,
    cu_stream_create: unsafe extern "C" fn(*mut *mut c_void, u32) -> CUresult,
    cu_stream_synchronize: unsafe extern "C" fn(*mut c_void) -> CUresult,
    cu_stream_destroy: unsafe extern "C" fn(*mut c_void) -> CUresult,
}

impl Cuda {
    /// Initialize the driver and retain the primary context of `device`.
    pub fn open(device_index: i32) -> Result<Self> {
        let lib_paths = [
            "libcuda.so",
            "libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        ];
        let lib = lib_paths
            .iter()
            .find_map(|path| unsafe { Library::new(path).ok() })
            .ok_or_else(|| Error::LibraryNotFound("libcuda.so not found".into()))?;

        unsafe {
            let cu_init = *lib
                .get::<unsafe extern "C" fn(u32) -> CUresult>(b"cuInit_v2\0")
                .or_else(|_| lib.get::<unsafe extern "C" fn(u32) -> CUresult>(b"cuInit\0"))
                .map_err(|e| Error::LibraryNotFound(format!("cuInit: {e}")))?;
            check(cu_init(0), "cuInit")?;

            let cu_device_get = *lib
                .get::<unsafe extern "C" fn(*mut i32, i32) -> CUresult>(b"cuDeviceGet_v2\0")
                .or_else(|_| lib.get::<unsafe extern "C" fn(*mut i32, i32) -> CUresult>(b"cuDeviceGet\0"))
                .map_err(|e| Error::LibraryNotFound(format!("cuDeviceGet: {e}")))?;
            let mut dev = 0i32;
            check(cu_device_get(&mut dev, device_index), "cuDeviceGet")?;

            let cu_retain = *lib
                .get::<unsafe extern "C" fn(*mut *mut c_void, i32) -> CUresult>(b"cuDevicePrimaryCtxRetain\0")
                .map_err(|e| Error::LibraryNotFound(format!("cuDevicePrimaryCtxRetain: {e}")))?;
            let mut ctx: *mut c_void = std::ptr::null_mut();
            check(cu_retain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
            if ctx.is_null() {
                return Err(Error::Cuda("primary context is null".into()));
            }

            let cu_set_current = *lib
                .get::<unsafe extern "C" fn(*mut c_void) -> CUresult>(b"cuCtxSetCurrent\0")
                .map_err(|e| Error::LibraryNotFound(format!("cuCtxSetCurrent: {e}")))?;
            check(cu_set_current(ctx), "cuCtxSetCurrent")?;

            let cu_mem_alloc = *lib
                .get::<unsafe extern "C" fn(*mut CUdeviceptr, u64) -> CUresult>(b"cuMemAlloc_v2\0")
                .or_else(|_| lib.get::<unsafe extern "C" fn(*mut CUdeviceptr, u64) -> CUresult>(b"cuMemAlloc\0"))
                .map_err(|e| Error::LibraryNotFound(format!("cuMemAlloc: {e}")))?;
            let cu_mem_free = *lib
                .get::<unsafe extern "C" fn(CUdeviceptr) -> CUresult>(b"cuMemFree_v2\0")
                .or_else(|_| lib.get::<unsafe extern "C" fn(CUdeviceptr) -> CUresult>(b"cuMemFree\0"))
                .map_err(|e| Error::LibraryNotFound(format!("cuMemFree: {e}")))?;
            let cu_memcpy_htod = *lib
                .get::<unsafe extern "C" fn(CUdeviceptr, *const c_void, u64) -> CUresult>(b"cuMemcpyHtoD_v2\0")
                .or_else(|_| {
                    lib.get::<unsafe extern "C" fn(CUdeviceptr, *const c_void, u64) -> CUresult>(b"cuMemcpyHtoD\0")
                })
                .map_err(|e| Error::LibraryNotFound(format!("cuMemcpyHtoD: {e}")))?;
            let cu_memcpy_dtoh = *lib
                .get::<unsafe extern "C" fn(*mut c_void, CUdeviceptr, u64) -> CUresult>(b"cuMemcpyDtoH_v2\0")
                .or_else(|_| lib.get::<unsafe extern "C" fn(*mut c_void, CUdeviceptr, u64) -> CUresult>(b"cuMemcpyDtoH\0"))
                .map_err(|e| Error::LibraryNotFound(format!("cuMemcpyDtoH: {e}")))?;
            let cu_stream_create = *lib
                .get::<unsafe extern "C" fn(*mut *mut c_void, u32) -> CUresult>(b"cuStreamCreate\0")
                .map_err(|e| Error::LibraryNotFound(format!("cuStreamCreate: {e}")))?;
            let cu_stream_synchronize = *lib
                .get::<unsafe extern "C" fn(*mut c_void) -> CUresult>(b"cuStreamSynchronize\0")
                .map_err(|e| Error::LibraryNotFound(format!("cuStreamSynchronize: {e}")))?;
            let cu_stream_destroy = *lib
                .get::<unsafe extern "C" fn(*mut c_void) -> CUresult>(b"cuStreamDestroy_v2\0")
                .or_else(|_| lib.get::<unsafe extern "C" fn(*mut c_void) -> CUresult>(b"cuStreamDestroy\0"))
                .map_err(|e| Error::LibraryNotFound(format!("cuStreamDestroy: {e}")))?;

            Ok(Self {
                lib,
                ctx,
                cu_mem_alloc,
                cu_mem_free,
                cu_memcpy_htod,
                cu_memcpy_dtoh,
                cu_stream_create,
                cu_stream_synchronize,
                cu_stream_destroy,
            })
        }
    }

    /// The retained context handle (passed to NVENC as the CUDA device).
    pub fn context(&self) -> *mut c_void {
        self.ctx
    }

    /// Allocate `size` bytes of device memory.
    pub fn mem_alloc(&self, size: u64) -> Result<CUdeviceptr> {
        let mut ptr = 0;
        unsafe {
            check((self.cu_mem_alloc)(&mut ptr, size), "cuMemAlloc")?;
        }
        Ok(ptr)
    }

    /// Free device memory.
    pub fn mem_free(&self, ptr: CUdeviceptr) -> Result<()> {
        unsafe { check((self.cu_mem_free)(ptr), "cuMemFree") }
    }

    /// Copy host memory to device memory.
    pub fn memcpy_htod(&self, dst: CUdeviceptr, src: &[u8]) -> Result<()> {
        unsafe { check((self.cu_memcpy_htod)(dst, src.as_ptr().cast(), src.len() as u64), "cuMemcpyHtoD") }
    }

    /// Copy device memory to host memory.
    pub fn memcpy_dtoh(&self, dst: &mut [u8], src: CUdeviceptr) -> Result<()> {
        unsafe { check((self.cu_memcpy_dtoh)(dst.as_mut_ptr().cast(), src, dst.len() as u64), "cuMemcpyDtoH") }
    }

    /// Create a default stream.
    pub fn stream_create(&self) -> Result<*mut c_void> {
        let mut s: *mut c_void = std::ptr::null_mut();
        unsafe { check((self.cu_stream_create)(&mut s, 0), "cuStreamCreate")?; }
        Ok(s)
    }

    /// Synchronize a stream.
    /// # Safety
    /// `s` must be a valid live stream handle.
    pub unsafe fn stream_synchronize(&self, s: *mut c_void) -> Result<()> {
        unsafe { check((self.cu_stream_synchronize)(s), "cuStreamSynchronize") }
    }

    /// Destroy a stream.
    /// # Safety
    /// `s` must be a valid live stream handle.
    pub unsafe fn stream_destroy(&self, s: *mut c_void) -> Result<()> {
        unsafe { check((self.cu_stream_destroy)(s), "cuStreamDestroy") }
    }
}

impl Drop for Cuda {
    fn drop(&mut self) {
        // Release the primary context (the library handle is dropped after).
        unsafe {
            if let Ok(release) =
                self.lib.get::<unsafe extern "C" fn(i32) -> u32>(b"cuDevicePrimaryCtxRelease_v2\0")
            {
                let _ = (*release)(0);
            }
        }
    }
}

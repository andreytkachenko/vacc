//! NVENC API session: loads `libnvidia-encode.so`, populates the function
//! vtable via `NvEncodeAPICreateInstance`, and wraps the calls used by the
//! encoder.
//!
//! The library exports only two entry points (`NvEncodeAPICreateInstance` and
//! `NvEncodeAPIGetMaxSupportedVersion`); everything else is reached through
//! the vtable the library fills in for us.

use std::ffi::c_void;
use std::mem::zeroed;

use libloading::Library;

use crate::cuda::Cuda;
use crate::nvenc_api as api;
use crate::Error;
use crate::Result;

fn check(st: api::_NVENCSTATUS, what: &str) -> Result<()> {
    if st == api::_NVENCSTATUS_NV_ENC_SUCCESS {
        Ok(())
    } else {
        Err(Error::Api(format!("{what} failed with NVENC status {st:?}")))
    }
}

/// The NVENC API function vtable (filled in by the driver).
pub struct NvencApi {
    #[allow(dead_code)]
    lib: Library,
    vtable: api::_NV_ENCODE_API_FUNCTION_LIST,
}

impl NvencApi {
    /// Load `libnvidia-encode.so` and create the API instance (vtable).
    pub fn open() -> Result<Self> {
        let lib_paths = [
            "libnvidia-encode.so.1",
            "libnvidia-encode.so",
            "/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1",
            "/usr/lib/x86_64-linux-gnu/libnvidia-encode.so",
        ];
        let lib = lib_paths
            .iter()
            .find_map(|path| unsafe { Library::new(path).ok() })
            .ok_or_else(|| Error::LibraryNotFound("libnvidia-encode.so not found".into()))?;

        let create: unsafe extern "C" fn(*mut api::_NV_ENCODE_API_FUNCTION_LIST) -> api::_NVENCSTATUS =
            unsafe {
                *lib
                    .get(b"NvEncodeAPICreateInstance\0")
                    .map_err(|e| Error::LibraryNotFound(format!("NvEncodeAPICreateInstance: {e}")))?
            };

        let mut vtable: api::_NV_ENCODE_API_FUNCTION_LIST = unsafe { zeroed() };
        vtable.version = api::NV_ENCODE_API_FUNCTION_LIST_VER;
        vtable.reserved = 0;
        check(unsafe { create(&mut vtable) }, "NvEncodeAPICreateInstance")?;

        // Sanity: the calls we rely on must be populated.
        if vtable.nvEncOpenEncodeSessionEx.is_none()
            || vtable.nvEncInitializeEncoder.is_none()
            || vtable.nvEncEncodePicture.is_none()
            || vtable.nvEncLockBitstream.is_none()
        {
            return Err(Error::Api("driver did not populate the NVENC vtable".into()));
        }
        Ok(Self { lib, vtable })
    }

    /// Open an encode session on a CUDA context. The returned session holds a
    /// raw pointer into this API's vtable and must not outlive it.
    pub fn open_session(&self, cuda: &Cuda) -> Result<Session> {
        let mut params: api::_NV_ENC_OPEN_ENCODE_SESSIONEX_PARAMS = unsafe { zeroed() };
        params.version = api::NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        params.deviceType = api::_NV_ENC_DEVICE_TYPE_NV_ENC_DEVICE_TYPE_CUDA;
        params.device = cuda.context();
        params.reserved = std::ptr::null_mut();
        params.apiVersion = api::NVENCAPI_VERSION;

        let mut handle: *mut c_void = std::ptr::null_mut();
        let open = self.vtable.nvEncOpenEncodeSessionEx.ok_or_else(|| {
            Error::Api("vtable missing nvEncOpenEncodeSessionEx".into())
        })?;
        check(unsafe { open(&mut params, &mut handle) }, "NvEncOpenEncodeSessionEx")?;
        if handle.is_null() {
            return Err(Error::Api("encode session handle is null".into()));
        }
        Ok(Session { vtable: &self.vtable, handle })
    }

    /// Retrieve the preset configuration for a codec/preset pair.
    /// # Safety
    /// `handle` must be a valid live encoder session handle.
    /// Fetch the preset config for a codec/preset pair.
    ///
    /// Driver 590+ removed `nvEncGetEncodePresetConfig` (it returns
    /// `NV_ENC_ERR_UNSUPPORTED_PARAM`); prefer `nvEncGetEncodePresetConfigEx`
    /// when the driver populates it, falling back to the legacy entry on
    /// older drivers. Both the outer `NV_ENC_PRESET_CONFIG::version` and the
    /// inner `presetCfg.version` must be set.
    pub unsafe fn get_preset_config(
        &self,
        handle: *mut c_void,
        encode_guid: api::GUID,
        preset_guid: api::GUID,
        tuning_info: api::NV_ENC_TUNING_INFO,
    ) -> Result<api::NV_ENC_CONFIG> {
        let mut out: api::_NV_ENC_PRESET_CONFIG = unsafe { zeroed() };
        out.version = api::NV_ENC_PRESET_CONFIG_VER;
        out.presetCfg.version = api::NV_ENC_CONFIG_VER;

        if let Some(f) = self.vtable.nvEncGetEncodePresetConfigEx {
            check(
                unsafe { f(handle, encode_guid, preset_guid, tuning_info, &mut out) },
                "NvEncGetEncodePresetConfigEx",
            )?;
        } else {
            let f = self
                .vtable
                .nvEncGetEncodePresetConfig
                .ok_or_else(|| Error::Api("vtable missing nvEncGetEncodePresetConfig".into()))?;
            check(
                unsafe { f(handle, encode_guid, preset_guid, &mut out) },
                "NvEncGetEncodePresetConfig",
            )?;
        }
        Ok(out.presetCfg)
    }

    /// Initialize the encoder with the given parameters.
    /// # Safety
    /// `handle` must be a valid live encoder session handle.
    pub unsafe fn initialize(&self, handle: *mut c_void, params: &mut api::NV_ENC_INITIALIZE_PARAMS) -> Result<()> {
        let f = self
            .vtable
            .nvEncInitializeEncoder
            .ok_or_else(|| Error::Api("vtable missing nvEncInitializeEncoder".into()))?;
        check(unsafe { f(handle, params) }, "NvEncInitializeEncoder")
    }

    /// Create an input buffer (GPU memory) for NV12 frames of the given size.
    /// Returns the buffer pointer and its byte size.
    /// # Safety
    /// `handle` must be a valid live encoder session handle.
    pub unsafe fn create_input_buffer(
        &self,
        handle: *mut c_void,
        width: u32,
        height: u32,
    ) -> Result<(api::NV_ENC_INPUT_PTR, usize)> {
        let mut params: api::_NV_ENC_CREATE_INPUT_BUFFER = unsafe { zeroed() };
        params.version = api::NV_ENC_CREATE_INPUT_BUFFER_VER;
        params.width = width;
        params.height = height;
        params.memoryHeap = api::_NV_ENC_MEMORY_HEAP_NV_ENC_MEMORY_HEAP_AUTOSELECT;
        params.bufferFmt = api::_NV_ENC_BUFFER_FORMAT_NV_ENC_BUFFER_FORMAT_NV12;
        let f = self
            .vtable
            .nvEncCreateInputBuffer
            .ok_or_else(|| Error::Api("vtable missing nvEncCreateInputBuffer".into()))?;
        check(unsafe { f(handle, &mut params) }, "NvEncCreateInputBuffer")?;
        if params.inputBuffer.is_null() {
            return Err(Error::Api("input buffer is null".into()));
        }
        // NV12: luma w*h + interleaved chroma w*h/2, no padding.
        Ok((params.inputBuffer, (width * height * 3 / 2) as usize))
    }

    /// Create a bitstream (output) buffer of `size` bytes.
    /// # Safety
    /// `handle` must be a valid live encoder session handle.
    pub unsafe fn create_bitstream_buffer(&self, handle: *mut c_void, size: usize) -> Result<api::NV_ENC_OUTPUT_PTR> {
        let mut params: api::_NV_ENC_CREATE_BITSTREAM_BUFFER = unsafe { zeroed() };
        params.version = api::NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        params.size = size as u32;
        let f = self
            .vtable
            .nvEncCreateBitstreamBuffer
            .ok_or_else(|| Error::Api("vtable missing nvEncCreateBitstreamBuffer".into()))?;
        check(unsafe { f(handle, &mut params) }, "NvEncCreateBitstreamBuffer")?;
        if params.bitstreamBuffer.is_null() {
            return Err(Error::Api("bitstream buffer is null".into()));
        }
        Ok(params.bitstreamBuffer)
    }

    /// Submit one frame for encoding (synchronous mode).
    /// # Safety
    /// `handle` must be a valid live encoder session handle.
    pub unsafe fn encode_picture(&self, handle: *mut c_void, params: &mut api::NV_ENC_PIC_PARAMS) -> Result<()> {
        let f = self
            .vtable
            .nvEncEncodePicture
            .ok_or_else(|| Error::Api("vtable missing nvEncEncodePicture".into()))?;
        check(unsafe { f(handle, params) }, "NvEncEncodePicture")
    }

    /// Lock the output bitstream buffer; returns (host pointer, size, hw status).
    /// # Safety
    /// `handle` must be a valid live encoder session handle and `buf` a buffer
    /// created by this session.
    pub unsafe fn lock_bitstream(
        &self,
        handle: *mut c_void,
        buf: api::NV_ENC_OUTPUT_PTR,
    ) -> Result<(*const u8, usize, u32)> {
        let mut params: api::_NV_ENC_LOCK_BITSTREAM = unsafe { zeroed() };
        params.version = api::NV_ENC_LOCK_BITSTREAM_VER;
        params.outputBitstream = buf;
        let f = self
            .vtable
            .nvEncLockBitstream
            .ok_or_else(|| Error::Api("vtable missing nvEncLockBitstream".into()))?;
        check(unsafe { f(handle, &mut params) }, "NvEncLockBitstream")?;
        if params.bitstreamBufferPtr.is_null() {
            return Err(Error::Api("locked bitstream pointer is null".into()));
        }
        Ok((
            params.bitstreamBufferPtr as *const u8,
            params.bitstreamSizeInBytes as usize,
            params.hwEncodeStatus,
        ))
    }

    /// Unlock a previously locked bitstream buffer.
    /// # Safety
    /// `handle` must be a valid live encoder session handle and `buf` a buffer
    /// created by this session.
    pub unsafe fn unlock_bitstream(&self, handle: *mut c_void, buf: api::NV_ENC_OUTPUT_PTR) -> Result<()> {
        let f = self
            .vtable
            .nvEncUnlockBitstream
            .ok_or_else(|| Error::Api("vtable missing nvEncUnlockBitstream".into()))?;
        check(unsafe { f(handle, buf) }, "NvEncUnlockBitstream")
    }

    /// Fetch the SPS/PPS header bytes for an initialized encoder (global
    /// header mode). Returns the number of valid bytes in `buf`.
    /// # Safety
    /// `handle` must be a valid live, initialized encoder session handle.
    pub unsafe fn get_sequence_params(&self, handle: *mut c_void, buf: &mut [u8]) -> Result<usize> {
        let mut params: api::_NV_ENC_SEQUENCE_PARAM_PAYLOAD = unsafe { zeroed() };
        params.version = api::NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
        params.inBufferSize = buf.len() as u32;
        params.spsppsBuffer = buf.as_mut_ptr().cast();
        let mut size: u32 = 0;
        params.outSPSPPSPayloadSize = &mut size;
        let f = self
            .vtable
            .nvEncGetSequenceParams
            .ok_or_else(|| Error::Api("vtable missing nvEncGetSequenceParams".into()))?;
        check(unsafe { f(handle, &mut params) }, "NvEncGetSequenceParams")?;
        if size as usize > buf.len() {
            return Err(Error::Api(format!(
                "SPS/PPS payload too large: {size} > {}",
                buf.len()
            )));
        }
        Ok(size as usize)
    }

    /// Lock an input buffer for CPU writes. Returns (host pointer, pitch).
    /// The host view is valid until [`Self::unlock_input_buffer`].
    /// # Safety
    /// `handle` must be a valid live encoder session handle and `buf` a buffer
    /// created by this session.
    pub unsafe fn lock_input_buffer(&self, handle: *mut c_void, buf: api::NV_ENC_INPUT_PTR) -> Result<(*mut u8, u32)> {
        let mut params: api::_NV_ENC_LOCK_INPUT_BUFFER = unsafe { zeroed() };
        params.version = api::NV_ENC_LOCK_INPUT_BUFFER_VER;
        params.inputBuffer = buf;
        let f = self
            .vtable
            .nvEncLockInputBuffer
            .ok_or_else(|| Error::Api("vtable missing nvEncLockInputBuffer".into()))?;
        check(unsafe { f(handle, &mut params) }, "NvEncLockInputBuffer")?;
        if params.bufferDataPtr.is_null() {
            return Err(Error::Api("locked input buffer pointer is null".into()));
        }
        Ok((params.bufferDataPtr as *mut u8, params.pitch))
    }

    /// Unlock a previously locked input buffer.
    /// # Safety
    /// `handle` must be a valid live encoder session handle and `buf` a buffer
    /// created by this session.
    pub unsafe fn unlock_input_buffer(&self, handle: *mut c_void, buf: api::NV_ENC_INPUT_PTR) -> Result<()> {
        let f = self
            .vtable
            .nvEncUnlockInputBuffer
            .ok_or_else(|| Error::Api("vtable missing nvEncUnlockInputBuffer".into()))?;
        check(unsafe { f(handle, buf) }, "NvEncUnlockInputBuffer")
    }

    /// Destroy an input buffer.
    /// # Safety
    /// `handle` must be a valid live encoder session handle and `buf` a buffer
    /// created by this session.
    pub unsafe fn destroy_input_buffer(&self, handle: *mut c_void, buf: api::NV_ENC_INPUT_PTR) -> Result<()> {
        let f = self
            .vtable
            .nvEncDestroyInputBuffer
            .ok_or_else(|| Error::Api("vtable missing nvEncDestroyInputBuffer".into()))?;
        check(unsafe { f(handle, buf) }, "NvEncDestroyInputBuffer")
    }

    /// Destroy a bitstream buffer.
    /// # Safety
    /// `handle` must be a valid live encoder session handle and `buf` a buffer
    /// created by this session.
    pub unsafe fn destroy_bitstream_buffer(&self, handle: *mut c_void, buf: api::NV_ENC_OUTPUT_PTR) -> Result<()> {
        let f = self
            .vtable
            .nvEncDestroyBitstreamBuffer
            .ok_or_else(|| Error::Api("vtable missing nvEncDestroyBitstreamBuffer".into()))?;
        check(unsafe { f(handle, buf) }, "NvEncDestroyBitstreamBuffer")
    }

    /// Human-readable description of the last error on `handle` (empty if none).
    pub unsafe fn last_error(&self, handle: *mut c_void) -> String {
        let Some(f) = self.vtable.nvEncGetLastErrorString else {
            return String::new();
        };
        let ptr = unsafe { f(handle) };
        if ptr.is_null() {
            return String::new();
        }
        unsafe { std::ffi::CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
    }
}

/// An NVENC encode session. Holds a raw pointer into the owning
/// [`NvencApi`] vtable; callers must drop the session before the API (the
/// encoder struct declares its fields in that order).
pub struct Session {
    vtable: *const api::_NV_ENCODE_API_FUNCTION_LIST,
    handle: *mut c_void,
}

// The session is only used from the thread that created it in practice, but
// the raw vtable pointer refers to immutable data.
unsafe impl Send for Session {}

impl Session {
    /// The raw session handle.
    pub fn handle(&self) -> *mut c_void {
        self.handle
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(f) = unsafe { (*self.vtable).nvEncDestroyEncoder } {
            unsafe {
                let _ = f(self.handle);
            }
        }
    }
}

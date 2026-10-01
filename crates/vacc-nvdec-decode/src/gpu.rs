//! Zero-copy frame extraction: device-to-device copy of mapped NVDEC
//! surfaces into owned device buffers.
//!
//! In GPU mode decoded frames never touch host memory: the mapped cuvid
//! surface is copied on-device into a tight buffer (Y, then interleaved CbCr
//! at display size) and handed out as a [`GpuFrame`]. The copy exists because
//! cuvid owns and recycles its DPB surfaces; an owned buffer keeps the frame
//! valid for the whole lifetime of the handle.

use std::ffi::c_void;

use crate::device::{
    CUDA_MEMCPY2D, CU_MEMORYTYPE_DEVICE, cu_ctx_set_current, cu_ctx_synchronize,
    cu_mem_alloc_device, cu_mem_free_device, cu_memcpy_2d,
};
use crate::ffi::{CUDA_SUCCESS, CUdeviceptr};
use crate::device::NvdecFuncs;
use vacc_core::gpu::{GpuDevice, GpuFrame, GpuPixelFormat};

/// Copy the mapped surface at `dev_ptr` into an owned device buffer and
/// return it as a [`GpuFrame`]. The surface is unmapped on every path.
///
/// Output layout is tight: `display_width * bps` bytes per row, Y first
/// (`display_height` rows), then the interleaved CbCr plane (half height)
/// when `has_chroma`. Cropping follows the display area (`crop_left` /
/// `crop_top` in samples; chroma rows start at `coded_height`).
#[allow(clippy::too_many_arguments)] // mirrors the per-frame surface state passed in by each codec path
pub(crate) fn extract_gpu_surface(
    decoder: *mut c_void,
    funcs: &NvdecFuncs,
    dev_ptr: CUdeviceptr,
    pitch: u32,
    display_width: usize,
    display_height: usize,
    bps: usize,
    crop_left: i32,
    crop_top: i32,
    coded_height: i32,
    has_chroma: bool,
) -> Option<GpuFrame> {
    let row_bytes = display_width * bps;
    let y_size = row_bytes * display_height;
    let uv_size = if has_chroma { row_bytes * (display_height / 2) } else { 0 };

    let base = match cu_mem_alloc_device(y_size + uv_size) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[NVDEC] GPU frame buffer alloc failed: {e}");
            let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
            return None;
        }
    };

    let copy_2d = |dst: CUdeviceptr, src_y: u64, height: u64| -> bool {
        let params = CUDA_MEMCPY2D {
            srcXInBytes: (crop_left as usize * bps) as u64,
            srcY: src_y,
            srcMemoryType: CU_MEMORYTYPE_DEVICE,
            _reserved0: 0,
            srcHost: std::ptr::null(),
            srcDevice: dev_ptr,
            srcArray: 0,
            srcPitch: pitch as u64,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: CU_MEMORYTYPE_DEVICE,
            _reserved1: 0,
            dstHost: std::ptr::null_mut(),
            dstDevice: dst,
            dstArray: 0,
            dstPitch: row_bytes as u64,
            WidthInBytes: row_bytes as u64,
            Height: height,
        };
        match unsafe { cu_memcpy_2d(&params) } {
            Ok(CUDA_SUCCESS) => true,
            other => {
                eprintln!("[NVDEC] GPU D2D copy failed: {other:?}");
                false
            }
        }
    };

    let uv_ok = !has_chroma
        || copy_2d(
            base + y_size as u64,
            coded_height as u64 + (crop_top as u64) / 2,
            (display_height / 2) as u64,
        );
    let ok = copy_2d(base, crop_top as u64, display_height as u64) && uv_ok;

    if !ok {
        let _ = unsafe { cu_mem_free_device(base) };
        let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
        return None;
    }

    let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };

    // The D2D copy is enqueued on the legacy stream; make it complete before
    // handing out the frame so consumers (NPP on its own stream, inference)
    // never read half-written data.
    let _ = cu_ctx_synchronize();

    Some(GpuFrame::new_owned(
        GpuDevice::Cuda { index: 0 },
        if bps == 2 { GpuPixelFormat::P016 } else { GpuPixelFormat::Nv12 },
        display_width as u32,
        display_height as u32,
        row_bytes,
        y_size,
        base as usize,
        None,
        Box::new(move |ptr: usize| {
            // Synchronous free: waits for in-flight GPU work (e.g. an
            // inference engine still reading the buffer) before releasing.
            let _ = cu_ctx_set_current();
            if let Err(e) = unsafe { cu_mem_free_device(ptr as CUdeviceptr) } {
                eprintln!("[NVDEC] failed to free GPU frame buffer: {e}");
            }
        }),
    ))
}

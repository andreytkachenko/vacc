//! GPU-resident frame handles for the zero-copy decode path.
//!
//! A [`GpuFrame`] describes a video frame whose pixel data lives in device
//! memory (e.g. an NVDEC output surface copied into an owned CUDA buffer, or
//! a Vulkan video DPB image copied into an owned `VkBuffer`). Backends that
//! support it fill [`DecodedFrame::gpu`](crate::frame::DecodedFrame) instead
//! of copying pixels to the host; image pipelines (NPP on NVIDIA hosts,
//! Vulkan compute elsewhere) then run directly on the device buffer, and
//! inference engines can consume the same memory afterwards via
//! [`GpuFrame::device_ptr`].

use std::sync::Arc;

/// Raw Vulkan handle values identifying the device that owns a frame's
/// memory. Handles are stored as plain `usize` so this crate stays free of
/// Vulkan bindings; consumers reconstruct them with e.g.
/// `ash::vk::Device::from_raw(h.device as u64)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanHandles {
    /// `vk::Instance` raw handle.
    pub instance: usize,
    /// `vk::PhysicalDevice` raw handle.
    pub physical: usize,
    /// `vk::Device` raw handle that owns the frame's allocations. Views and
    /// memory imports for inference must be created on this device.
    pub device: usize,
    /// Queue family index used by the producing decoder; it must support
    /// compute for on-GPU image processing of this frame.
    pub queue_family: u32,
}

/// The kind of device that owns a [`GpuFrame`]'s pixel memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDevice {
    /// CUDA device memory (NVDEC / NPP path). `index` is the CUDA device
    /// ordinal the allocation lives on.
    Cuda { index: i32 },
    /// Vulkan device memory (Vulkan video decode path). `index` is an opaque
    /// device ordinal; `handles` identifies the owning `VkDevice`.
    Vulkan { index: i32, handles: VulkanHandles },
}

/// Pixel format of a [`GpuFrame`] in device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPixelFormat {
    /// Semi-planar 8-bit Y'CbCr 4:2:0: a Y plane followed by one interleaved
    /// CbCr plane at half resolution (rows of `width` bytes).
    Nv12,
    /// Semi-planar 10-bit Y'CbCr 4:2:0 in u16 samples, top-justified codes
    /// (`code << 6`), same layout as [`GpuPixelFormat::Nv12`].
    P016,
    /// Packed 8-bit RGB, tight or pitched rows of `width * 3` bytes.
    Rgb24,
    /// Packed 8-bit RGBA, rows of `width * 4` bytes.
    Rgba32,
}

impl GpuPixelFormat {
    /// True for packed RGB formats (no separate chroma plane).
    pub const fn is_rgb(self) -> bool {
        matches!(self, Self::Rgb24 | Self::Rgba32)
    }

    /// Bytes per pixel (packed RGB) or per luma sample (YUV).
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Nv12 => 1,
            Self::P016 => 2,
            Self::Rgb24 => 3,
            Self::Rgba32 => 4,
        }
    }

    /// True when the format carries an interleaved CbCr chroma plane.
    pub const fn has_chroma(self) -> bool {
        matches!(self, Self::Nv12 | Self::P016)
    }
}

/// Owning handle for device memory; frees it when the last clone drops.
///
/// The free closure is backend-provided (e.g. a synchronous `cuMemFree`), so
/// dropping a [`GpuFrame`] on any thread safely waits for in-flight GPU work
/// before releasing the allocation.
struct Owner {
    ptr: usize,
    // Send + Sync so Arc<Owner> (and thus GpuFrame) can move across threads
    // to inference engines.
    free: Option<Box<dyn FnOnce(usize) + Send + Sync>>,
}

impl Drop for Owner {
    fn drop(&mut self) {
        if let Some(free) = self.free.take() {
            free(self.ptr);
        }
    }
}

/// A video frame whose pixel data resides in device memory.
///
/// Cloning is cheap (shared ownership): the backing memory is released when
/// the last clone is dropped.
#[derive(Clone)]
pub struct GpuFrame {
    /// The device that owns the memory.
    pub device: GpuDevice,
    /// Pixel format of the stored planes.
    pub format: GpuPixelFormat,
    /// Width in luma pixels (or RGB pixels).
    pub width: u32,
    /// Height in luma pixels (or RGB rows).
    pub height: u32,
    /// Row stride in bytes. For YUV formats this is the luma row stride and
    /// chroma rows use the same stride; for RGB formats it is
    /// `width * channels`.
    pub pitch: usize,
    /// Device pointer of the first plane (luma, or packed RGB rows). For
    /// CUDA frames this is a `CUdeviceptr`; for Vulkan frames it is the
    /// `VkBuffer` raw handle.
    pub ptr: usize,
    /// Vulkan frames only: the `vk::DeviceMemory` raw handle backing
    /// [`GpuFrame::ptr`]. CUDA frames carry `None` (the pointer itself is
    /// the full address).
    pub memory: Option<usize>,
    /// Byte offset from [`GpuFrame::ptr`] to the chroma plane (YUV formats);
    /// 0 for RGB formats. For Vulkan frames this is a byte offset within the
    /// buffer (planes are not pointer-addressable like CUDA memory).
    pub chroma_offset: usize,
    owner: Option<Arc<Owner>>,
}

impl GpuFrame {
    /// Create a handle that owns its device memory. `free` is invoked with
    /// `ptr` when the last clone of this frame is dropped (synchronously,
    /// from the dropping thread).
    #[allow(clippy::too_many_arguments)] // flat descriptor of a device plane layout
    pub fn new_owned(
        device: GpuDevice,
        format: GpuPixelFormat,
        width: u32,
        height: u32,
        pitch: usize,
        chroma_offset: usize,
        ptr: usize,
        memory: Option<usize>,
        free: Box<dyn FnOnce(usize) + Send + Sync>,
    ) -> Self {
        Self {
            device,
            format,
            width,
            height,
            pitch,
            ptr,
            memory,
            chroma_offset,
            owner: Some(Arc::new(Owner { ptr, free: Some(free) })),
        }
    }

    /// The raw device pointer (e.g. `CUdeviceptr` for CUDA/TensorRT
    /// inference engines; the `VkBuffer` handle for Vulkan frames). Valid
    /// while at least one clone of this frame (or a handle derived from it)
    /// is alive.
    pub const fn device_ptr(&self) -> usize {
        self.ptr
    }

    /// Device pointer of the chroma plane (YUV formats only; panics for RGB
    /// formats).
    pub fn chroma_ptr(&self) -> usize {
        assert!(self.format.has_chroma(), "RGB frame has no chroma plane");
        self.ptr + self.chroma_offset
    }

    /// True when dropping the last clone releases the device memory.
    pub const fn is_owned(&self) -> bool {
        self.owner.is_some()
    }

    /// Size of the first plane in bytes (`pitch * height`).
    pub fn luma_bytes(&self) -> usize {
        self.pitch * self.height as usize
    }

    /// Total allocation size for this frame's layout: the luma plane plus,
    /// for YUV formats, the interleaved chroma plane at half height.
    pub fn total_bytes(&self) -> usize {
        let luma = self.luma_bytes();
        if self.format.has_chroma() {
            luma + self.pitch * (self.height as usize / 2)
        } else {
            luma
        }
    }
}

impl std::fmt::Debug for GpuFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuFrame")
            .field("device", &self.device)
            .field("format", &self.format)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("pitch", &self.pitch)
            .field("ptr", &format_args!("{:#x}", self.ptr))
            .field("chroma_offset", &self.chroma_offset)
            .field("owned", &self.is_owned())
            .finish()
    }
}

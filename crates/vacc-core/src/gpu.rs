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

/// A backend completion signal (fence/event) shared across threads.
///
/// This is a non-owning view: the underlying backend handle (a CUDA event, a
/// Vulkan semaphore, ...) is created and destroyed by its owner (e.g. a
/// decoder's event pool or a ring of per-slot reader barriers); dropping the
/// last [`GpuEvent`] clone does not destroy it.
#[derive(Clone)]
pub struct GpuEvent(Arc<dyn EventHandle>);

/// Backend-specific implementation of a GPU completion signal.
///
/// Implementations wrap a live backend handle and must be `Send + Sync` so a
/// [`GpuEvent`] can cross threads (decode thread -> inference pool -> frame
/// drop on any worker). All methods are safe: the unsafe driver calls stay
/// inside the implementation.
pub trait EventHandle: Send + Sync {
    /// Block the host until this event is signaled.
    fn synchronize(&self) -> Result<(), String>;

    /// Non-blocking check: `true` once the event is signaled.
    fn is_complete(&self) -> bool;

    /// Raw backend handle (e.g. `CUevent`) for interop with other toolkits.
    /// Valid while the owning backend object is alive.
    fn raw(&self) -> *const std::ffi::c_void;

    /// Enqueue a stream-ordered wait on this event. `stream` is a raw
    /// backend stream handle (e.g. `CUstream`) from any toolkit sharing the
    /// same device.
    fn wait_on_stream(&self, stream: *const std::ffi::c_void) -> Result<(), String>;

    /// Record this event on a stream: it is signaled once all work enqueued
    /// before this call has completed. `stream` is a raw backend stream
    /// handle (e.g. `CUstream`).
    fn record_on_stream(&self, stream: *const std::ffi::c_void) -> Result<(), String>;
}

impl GpuEvent {
    /// Wrap an existing backend handle (non-owning).
    pub fn wrap(handle: Arc<dyn EventHandle>) -> Self {
        Self(handle)
    }

    /// Block the host until this event is signaled.
    pub fn synchronize(&self) -> Result<(), String> {
        self.0.synchronize()
    }

    /// Non-blocking check: `true` once the event is signaled.
    pub fn is_complete(&self) -> bool {
        self.0.is_complete()
    }

    /// Raw backend handle (e.g. `CUevent`) for interop with other toolkits.
    pub fn raw(&self) -> *const std::ffi::c_void {
        self.0.raw()
    }

    /// Enqueue a stream-ordered wait on this event (raw backend stream).
    pub fn wait_on_stream(&self, stream: *const std::ffi::c_void) -> Result<(), String> {
        self.0.wait_on_stream(stream)
    }

    /// Record this event on a stream (raw backend stream).
    pub fn record_on_stream(&self, stream: *const std::ffi::c_void) -> Result<(), String> {
        self.0.record_on_stream(stream)
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
    /// Signaled once the producer has fully written the pixel data at
    /// [`ptr`](Self::ptr). Set for asynchronously produced frames (e.g. a
    /// decoder's device-to-device copy); consumers must wait on it
    /// (host-side or stream-ordered) before reading. `None` when the data is
    /// ready synchronously.
    pub copy_done: Option<GpuEvent>,
    /// Reader barrier for reusable ring slots: consumers record it after
    /// their last read of [`ptr`](Self::ptr); the producer waits on it before
    /// overwriting the slot. `None` unless this frame occupies a ring slot.
    pub consume_barrier: Option<GpuEvent>,
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
            chroma_offset,
            ptr,
            memory,
            copy_done: None,
            consume_barrier: None,
            owner: Some(Arc::new(Owner { ptr, free: Some(free) })),
        }
    }

    /// Create a handle that does NOT own its device memory (e.g. a frame
    /// occupying a slot in a caller-managed ring; the ring's owner releases
    /// the memory). No release action runs when the last clone drops unless
    /// one is attached with [`with_release`](Self::with_release).
    #[allow(clippy::too_many_arguments)] // flat descriptor of a device plane layout
    pub fn new_borrowed(
        device: GpuDevice,
        format: GpuPixelFormat,
        width: u32,
        height: u32,
        pitch: usize,
        chroma_offset: usize,
        ptr: usize,
        memory: Option<usize>,
    ) -> Self {
        Self {
            device,
            format,
            width,
            height,
            pitch,
            chroma_offset,
            ptr,
            memory,
            copy_done: None,
            consume_barrier: None,
            owner: None,
        }
    }

    /// The raw device pointer (e.g. `CUdeviceptr` for CUDA/TensorRT
    /// inference engines; the `VkBuffer` handle for Vulkan frames). Valid
    /// while at least one clone of this frame (or a handle derived from it)
    /// is alive.
    pub const fn device_ptr(&self) -> usize {
        self.ptr
    }

    /// Attach the producer-ready fence: consumers must wait on it before
    /// reading [`ptr`](Self::ptr).
    pub fn with_copy_done(mut self, event: GpuEvent) -> Self {
        self.copy_done = Some(event);
        self
    }

    /// Attach the reader barrier of the ring slot this frame occupies.
    pub fn with_consume_barrier(mut self, event: GpuEvent) -> Self {
        self.consume_barrier = Some(event);
        self
    }

    /// Attach a one-shot release action run exactly once, when the last clone
    /// of this frame drops (e.g. returning a ring slot to its owner so it may
    /// be reused). Replaces any previous ownership.
    pub fn with_release(mut self, release: Box<dyn FnOnce(usize) + Send + Sync>) -> Self {
        self.owner = Some(Arc::new(Owner { ptr: self.ptr, free: Some(release) }));
        self
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

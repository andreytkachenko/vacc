//! Decoded frame representation and metadata.

/// Presentation information for a decoded frame.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// Frame index in the output sequence.
    pub frame_index: u32,
    /// Presentation timestamp (in nanoseconds, or codec-specific units).
    pub timestamp: i64,
    /// Frame dimensions.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Whether the frame was skipped (not decoded).
    pub skipped: bool,
    /// Whether PTS is valid.
    pub pts_valid: bool,
    /// Picture Order Count.
    pub poc: i32,
    /// Field flags for interlaced content.
    pub field_flags: FieldFlags,
    /// Frame sync information.
    pub sync_info: FrameSyncInfo,
    /// Pixel data for the decoded frame (if available).
    pub pixel_data: Option<PixelData>,
    /// Raw device address of NV12 pixel data when the backend was configured
    /// for device output (e.g. the NVDEC decoder's `set_device_output_ring`).
    /// When set, `pixel_data` is `None` and the pixels live at this device
    /// address in tightly-packed NV12 layout (Y plane followed by the
    /// interleaved UV plane) at `width x height`. The backing buffer belongs
    /// to the caller's ring of device buffers; it stays valid until the
    /// backend reuses it for a later frame.
    pub device_ptr: Option<usize>,
    /// CUDA event (`CUevent` handle value) that is signaled when the
    /// device-to-device copy into [`device_ptr`](Self::device_ptr) has
    /// completed. Device-output extraction is asynchronous: the backend
    /// enqueues the copies and returns before they finish, so consumers must
    /// complete the copy before reading `device_ptr`. The event belongs to
    /// the backend's CUDA context; host-side `cuEventSynchronize` works from
    /// any context. NOTE: a stream-ordered `cuStreamWaitEvent` on this event
    /// from a *different* context is not reliable (observed to not order on
    /// current NVIDIA drivers), so do not rely on it — host-sync instead, and
    /// call the backend's copy-done notification (see [`device_seq`]).
    pub device_copy_event: Option<usize>,
    /// DPB sequence id of the source surface for device-output frames. Pass
    /// it back to the backend's copy-done notification once
    /// [`device_copy_event`](Self::device_copy_event) has been host-synced,
    /// so the backend can release (unmap) the mapped surface. `None` unless
    /// `device_ptr` is set.
    pub device_seq: Option<i32>,
    /// Packed RGB/RGBA pixel data. Populated when a backend (or the
    /// facade) converted the decoded frame to RGB, optionally after
    /// scaling. `None` unless image conversion was requested.
    pub rgb_pixels: Option<RgbFrame>,
    /// GPU-resident pixel data (zero-copy decode path). When set, the frame's
    /// pixels live in device memory and [`pixel_data`](Self::pixel_data) is
    /// `None`; the image pipeline and inference engines operate on the device
    /// buffer directly.
    pub gpu: Option<crate::gpu::GpuFrame>,
}

// `DecodedFrame` is auto `Send + Sync`: every field is a plain value, an
// owned buffer, or an opaque integer handle (`device_ptr`,
// `device_copy_event`) carried as metadata — valid only while its producer
// (decoder ring / event pool) is alive, with access ordered by the fences it
// carries. No raw pointers escape the type.

impl DecodedFrame {
    /// Create a new decoded frame.
    pub fn new(frame_index: u32, timestamp: i64, width: u32, height: u32, skipped: bool) -> Self {
        Self {
            frame_index,
            timestamp,
            width,
            height,
            skipped,
            pts_valid: false,
            poc: 0,
            field_flags: FieldFlags::default(),
            sync_info: FrameSyncInfo::default(),
            pixel_data: None,
            device_ptr: None,
            device_copy_event: None,
            device_seq: None,
            rgb_pixels: None,
            gpu: None,
        }
    }

    /// Returns true if this is a reference frame.
    pub const fn is_reference(&self) -> bool {
        self.field_flags.ref_pic
    }

    /// Returns true if this is a progressive frame.
    pub const fn is_progressive(&self) -> bool {
        self.field_flags.progressive_frame
    }
}

impl Default for DecodedFrame {
    fn default() -> Self {
        Self {
            frame_index: 0,
            timestamp: 0,
            width: 0,
            height: 0,
            skipped: true,
            pts_valid: false,
            poc: 0,
            field_flags: FieldFlags::default(),
            sync_info: FrameSyncInfo::default(),
            pixel_data: None,
            device_ptr: None,
            device_copy_event: None,
            device_seq: None,
            rgb_pixels: None,
            gpu: None,
        }
    }
}

/// Field flags for interlaced/field-based video.
#[derive(Debug, Clone, Copy, Default)]
pub struct FieldFlags {
    /// Frame is progressive.
    pub progressive_frame: bool,
    /// 0 = frame picture, 1 = field picture.
    pub field_pic: bool,
    /// 0 = top field, 1 = bottom field.
    pub bottom_field: bool,
    /// Second field of a complementary field pair.
    pub second_field: bool,
    /// Frame pictures only - top field first.
    pub top_field_first: bool,
    /// Incomplete (half) frame.
    pub unpaired_field: bool,
    /// Synchronize the second field to the first one.
    pub sync_first_ready: bool,
    /// Synchronize to first field.
    pub sync_to_first_field: bool,
    /// For 3:2 pulldown (number of additional fields).
    pub repeat_first_field: u8,
    /// Frame is a reference frame.
    pub ref_pic: bool,
    /// Valid for AV1 only - apply film grain.
    pub apply_film_grain: bool,
}

impl FieldFlags {
    pub fn as_u32(&self) -> u32 {
        let mut flags = 0u32;
        if self.progressive_frame {
            flags |= 1 << 0;
        }
        if self.field_pic {
            flags |= 1 << 1;
        }
        if self.bottom_field {
            flags |= 1 << 2;
        }
        if self.second_field {
            flags |= 1 << 3;
        }
        if self.top_field_first {
            flags |= 1 << 4;
        }
        if self.unpaired_field {
            flags |= 1 << 5;
        }
        if self.sync_first_ready {
            flags |= 1 << 6;
        }
        if self.sync_to_first_field {
            flags |= 1 << 7;
        }
        flags |= (self.repeat_first_field as u32) << 8;
        if self.ref_pic {
            flags |= 1 << 11;
        }
        if self.apply_film_grain {
            flags |= 1 << 12;
        }
        flags
    }

    pub fn from_u32(value: u32) -> Self {
        Self {
            progressive_frame: value & (1 << 0) != 0,
            field_pic: value & (1 << 1) != 0,
            bottom_field: value & (1 << 2) != 0,
            second_field: value & (1 << 3) != 0,
            top_field_first: value & (1 << 4) != 0,
            unpaired_field: value & (1 << 5) != 0,
            sync_first_ready: value & (1 << 6) != 0,
            sync_to_first_field: value & (1 << 7) != 0,
            repeat_first_field: ((value >> 8) & 0x7) as u8,
            ref_pic: value & (1 << 11) != 0,
            apply_film_grain: value & (1 << 12) != 0,
        }
    }
}

/// Frame synchronization information.
#[derive(Debug, Clone, Default)]
pub struct FrameSyncInfo {
    /// Whether to generate a semaphore reference.
    pub unpaired_field: bool,
    /// Whether to use semaphore from unpaired field.
    pub sync_to_first_field: bool,
    /// Debug interface handle (reserved for future use).
    pub debug_interface: Option<usize>,
}

/// A single plane of pixel data (Y, U, or V).
///
/// A plane is a view into the owning [`PixelData::buffer`]: `data_offset`
/// is the byte offset of the plane's first sample within that buffer. Every
/// producer stores its planes inside the buffer, so no raw pointers are
/// needed and the type is trivially `Send + Sync`.
#[derive(Debug, Clone, Copy)]
pub struct PixelPlane {
    /// Byte offset of the first sample within [`PixelData::buffer`].
    pub data_offset: usize,
    /// Row pitch in bytes.
    pub pitch: usize,
    /// Plane width in pixels.
    pub width: usize,
    /// Plane height in pixels.
    pub height: usize,
}

impl PixelPlane {
    /// Byte offset, within the owning buffer, of the start of row `row`.
    #[inline]
    pub const fn row_offset(&self, row: usize) -> usize {
        self.data_offset + row * self.pitch
    }
}

/// Packed (interleaved) RGB(X) pixel data.
///
/// Produced by the image-conversion routines (YUV -> RGB) and resize.
/// Rows are tightly packed: buffer length is `width * height * channels`.
#[derive(Debug, Clone)]
pub struct RgbFrame {
    /// Pixel buffer (row-major, tight rows).
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of channels: 3 = RGB24, 4 = RGBA32.
    pub channels: u8,
}

impl RgbFrame {
    /// Number of bytes per row (tight packing).
    pub const fn row_bytes(&self) -> usize {
        (self.width as usize) * self.channels as usize
    }

    /// Number of bytes per pixel.
    pub const fn bytes_per_pixel(&self) -> usize {
        self.channels as usize
    }
}

/// Planar pixel data for a decoded frame.
#[derive(Debug, Clone)]
pub struct PixelData {
    /// Format string (e.g., "NV12", "YV12", "I420").
    pub format: String,
    /// Luma (Y) plane.
    pub y: PixelPlane,
    /// Chroma U plane.
    pub u: PixelPlane,
    /// Chroma V plane (None for semi-planar formats like NV12).
    pub v: Option<PixelPlane>,
    /// Owned buffer backing the planes.
    pub buffer: Vec<u8>,
}

impl PixelData {
    /// View of one plane's storage: `height * pitch` bytes from its offset.
    #[inline]
    pub fn plane_view(&self, p: &PixelPlane) -> &[u8] {
        let end = p.data_offset + p.height * p.pitch;
        &self.buffer[p.data_offset..end]
    }

    /// One row of a plane: the first `row_bytes` bytes of the row at `row`
    /// (typically `width * bytes_per_sample`).
    #[inline]
    pub fn plane_row(&self, p: &PixelPlane, row: usize, row_bytes: usize) -> &[u8] {
        let start = p.row_offset(row);
        &self.buffer[start..start + row_bytes]
    }
}

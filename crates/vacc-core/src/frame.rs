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
    /// Packed RGB/RGBA pixel data. Populated when a backend (or the
    /// facade) converted the decoded frame to RGB, optionally after
    /// scaling. `None` unless image conversion was requested.
    pub rgb_pixels: Option<RgbFrame>,
}

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
            rgb_pixels: None,
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
            rgb_pixels: None,
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
    /// Debug interface pointer (reserved for future use).
    pub debug_interface: Option<*mut std::ffi::c_void>,
}

/// A single plane of pixel data (Y, U, or V).
#[derive(Debug, Clone, Copy)]
pub struct PixelPlane {
    /// Pointer to plane data.
    pub data: *const u8,
    /// Row pitch in bytes.
    pub pitch: usize,
    /// Plane width in pixels.
    pub width: usize,
    /// Plane height in pixels.
    pub height: usize,
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
#[derive(Debug)]
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

impl Clone for PixelData {
    fn clone(&self) -> Self {
        // Plane pointers are offsets into `buffer`; rebase them onto the
        // cloned buffer so the clone is self-contained. A derived clone would
        // leave them pointing at the original buffer (dangling once the
        // original drops).
        let buffer = self.buffer.clone();
        let base = self.buffer.as_ptr() as usize;
        let len = self.buffer.len();
        let rebased = |p: &PixelPlane| -> PixelPlane {
            let addr = p.data as usize;
            if addr >= base && addr < base + len {
                let mut q = *p;
                q.data = unsafe { buffer.as_ptr().add(addr - base) };
                q
            } else {
                *p
            }
        };
        PixelData {
            format: self.format.clone(),
            y: rebased(&self.y),
            u: rebased(&self.u),
            v: self.v.as_ref().map(rebased),
            buffer,
        }
    }
}

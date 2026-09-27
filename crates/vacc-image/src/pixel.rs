//! Borrowed views over Y'CbCr and RGB(X) pixel data.
//!
//! All views carry explicit row pitches so tight (`pitch == width * bps`)
//! and padded planes are both supported. Chroma is always 4:2:0 (YUV420 /
//! NV12 / P010 / P012 variants).

/// Interleaved-UV layout (semi-planar) vs three separate planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YuvLayout {
    /// Three separate planes (I420 / YUV420P at any bit depth).
    Planar,
    /// Y plane + interleaved CbCr plane (NV12 / P010 / P012).
    Semi,
}

impl Default for YuvLayout {
    fn default() -> Self {
        YuvLayout::Planar
    }
}

/// Borrowed view over a 4:2:0 Y'CbCr image.
///
/// Slices are in *bytes*. For `bits_per_sample == 8` each sample is 1 byte;
/// for P010/P012 each sample is a little-endian `u16` with the value
/// *top-justified* (i.e. 10-bit value `v` is stored as `v << 6`).
#[derive(Debug, Clone, Copy)]
pub struct YuvImage<'a> {
    /// Luma plane. Row length in bytes is `y_pitch`.
    pub y: &'a [u8],
    /// Luma row pitch (bytes).
    pub y_pitch: usize,
    /// Cb plane (planar layout).
    pub cb: &'a [u8],
    /// Cb row pitch (bytes).
    pub cb_pitch: usize,
    /// Cr plane (planar layout); for the semi-planar layout this is
    /// `None` and the CbCr interleaved plane is referenced through `cb` /
    /// `cb_pitch` with `cr_offset_bytes` pointing at the first Cr sample.
    pub cr: Option<&'a [u8]>,
    /// Cr row pitch (bytes); for the semi-planar layout equals `cb_pitch`.
    pub cr_pitch: usize,
    /// For the semi-planar layout: byte offset from the Cb plane start to
    /// the first Cr sample (1 byte for 8-bit, 2 bytes for P010/P012).
    pub cr_offset_bytes: usize,
    /// Width in luma pixels.
    pub width: usize,
    /// Height in luma pixels.
    pub height: usize,
    /// Bits per sample: 8, 10 or 12.
    pub bits_per_sample: u8,
    /// Whether the sample is full-range ("Jpeg"); see `ColorRange`.
    ///
    /// Kept out of the hot path; conversions take `ColorSpec` explicitly.
    #[allow(dead_code)]
    pub layout_hint: YuvLayout,
}

impl<'a> YuvImage<'a> {
    /// Planar 4:2:0 (I420 for 8-bit, equivalent for 10/12-bit packed u16).
    #[allow(clippy::too_many_arguments)]
    pub fn planar(
        y: &'a [u8],
        y_pitch: usize,
        cb: &'a [u8],
        cb_pitch: usize,
        cr: &'a [u8],
        cr_pitch: usize,
        width: usize,
        height: usize,
        bits_per_sample: u8,
    ) -> Self {
        Self {
            y,
            y_pitch,
            cb,
            cb_pitch,
            cr: Some(cr),
            cr_pitch,
            cr_offset_bytes: 0,
            width,
            height,
            bits_per_sample,
            layout_hint: YuvLayout::Planar,
        }
    }

    /// Semi-planar 4:2:0 (NV12 for 8-bit, P010/P012 for 10/12-bit).
    pub fn semi(
        y: &'a [u8],
        y_pitch: usize,
        cbcr: &'a [u8],
        cbcr_pitch: usize,
        width: usize,
        height: usize,
        bits_per_sample: u8,
    ) -> Self {
        Self {
            y,
            y_pitch,
            cb: cbcr,
            cb_pitch: cbcr_pitch,
            cr: None,
            cr_pitch: cbcr_pitch,
            cr_offset_bytes: bps_bytes(bits_per_sample),
            width,
            height,
            bits_per_sample,
            layout_hint: YuvLayout::Semi,
        }
    }

    /// Bytes per sample (1 for 8-bit, 2 for 10/12-bit packed in u16).
    pub const fn bps_bytes(&self) -> usize {
        bps_bytes(self.bits_per_sample)
    }

    /// Number of samples that fit on one Cb/Cr row.
    pub const fn chroma_width(&self) -> usize {
        (self.width + 1) / 2
    }

    /// Number of Cb/Cr rows.
    pub const fn chroma_height(&self) -> usize {
        (self.height + 1) / 2
    }

    /// One luma row (`y_pitch` bytes).
    pub fn luma_row(&self, row: usize) -> &'a [u8] {
        let start = row * self.y_pitch;
        &self.y[start..start + self.y_pitch]
    }

    /// One Cb row (samples), for the planar layout.
    pub fn cb_row(&self, row: usize) -> &'a [u8] {
        let start = row * self.cb_pitch;
        &self.cb[start..start + self.cb_pitch]
    }

    /// One Cr row (samples), for the planar layout.
    pub fn cr_row(&self, row: usize) -> &'a [u8] {
        let cr = self
            .cr
            .expect("cr_plane missing on planar view");
        let start = row * self.cr_pitch;
        &cr[start..start + self.cr_pitch]
    }

    /// One interleaved CbCr row (samples), for the semi-planar layout.
    pub fn cbcr_row(&self, row: usize) -> &'a [u8] {
        let start = row * self.cb_pitch;
        &self.cb[start..start + self.cb_pitch]
    }
}

/// Bytes per sample for a Y'CbCr bit depth (1 for 8-bit, 2 for 10/12-bit).
pub const fn bps_bytes(bits_per_sample: u8) -> usize {
    if bits_per_sample > 8 {
        2
    } else {
        1
    }
}

/// Borrowed view over a packed RGB/RGBA image.
///
/// Row length in bytes is `pitch`. `channels` is 3 (RGB24) or 4 (RGBA32).
#[derive(Debug, Clone, Copy)]
pub struct RgbImage<'a> {
    pixels: &'a [u8],
    pub pitch: usize,
    pub width: usize,
    pub height: usize,
    pub channels: u8,
}

impl<'a> RgbImage<'a> {
    pub fn new(pixels: &'a [u8], pitch: usize, width: usize, height: usize, channels: u8) -> Self {
        assert!(pitch >= width * channels as usize, "pitch smaller than row size");
        Self {
            pixels,
            pitch,
            width,
            height,
            channels,
        }
    }

    /// One row in pixels.
    pub fn row(&self, row: usize) -> &'a [u8] {
        let start = row * self.pitch;
        &self.pixels[start..start + self.pitch]
    }

    /// Pointer to the first pixel byte (for SIMD row kernels).
    pub(crate) fn pixels_ptr(&self) -> *const u8 {
        self.pixels.as_ptr()
    }

    /// Raw pixel data.
    pub fn pixels(&self) -> &'a [u8] {
        self.pixels
    }
}

/// Mutable counterpart of [`RgbImage`].
pub struct RgbImageMut<'a> {
    pixels: &'a mut [u8],
    pub pitch: usize,
    pub width: usize,
    pub height: usize,
    pub channels: u8,
}

impl<'a> RgbImageMut<'a> {
    pub fn new(pixels: &'a mut [u8], pitch: usize, width: usize, height: usize, channels: u8) -> Self {
        Self {
            pixels,
            pitch,
            width,
            height,
            channels,
        }
    }

    /// One row in pixels.
    pub fn row(&mut self, row: usize) -> &mut [u8] {
        let start = row * self.pitch;
        &mut self.pixels[start..start + self.pitch]
    }
}

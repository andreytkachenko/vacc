//! Pixel formats, color specifications and image configuration.

/// Packed-output channel count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RgbChannels {
    /// RGB, 3 bytes per pixel.
    Rgb24,
    /// RGBA, 4 bytes per pixel (alpha = 0xFF).
    Rgba32,
}

impl RgbChannels {
    /// Number of bytes per pixel.
    pub const fn bytes(self) -> u8 {
        match self {
            RgbChannels::Rgb24 => 3,
            RgbChannels::Rgba32 => 4,
        }
    }
}

impl Default for RgbChannels {
    fn default() -> Self {
        RgbChannels::Rgb24
    }
}

/// Color-range convention of the source `Y'CbCr` samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorRange {
    /// Limited ("studio") range: Y in `16..=235`, Cb/Cr centered at 128.
    #[default]
    Limited,
    /// Full ("Jpeg") range: Y in `0..=255`, Cb/Cr centered at 128.
    Full,
}

/// Colorimetry matrix coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixCoefficients {
    /// BT.601 (SD).
    Bt601,
    /// BT.709 (HD default).
    #[default]
    Bt709,
}

/// Y'CbCr -> RGB conversion parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ColorSpec {
    /// Matrix coefficients.
    pub matrix: MatrixCoefficients,
    /// Sample range.
    pub range: ColorRange,
}

impl ColorSpec {
    /// Sensible default for a stream of `height` lines (mirrors common
    /// player behavior): full-range content is rare in movie formats, so this
    /// is limited-range; BT.709 for HD and above, BT.601 otherwise.
    pub fn auto(height: u32) -> Self {
        Self {
            matrix: if height >= 720 {
                MatrixCoefficients::Bt709
            } else {
                MatrixCoefficients::Bt601
            },
            range: ColorRange::Limited,
        }
    }

    /// Q14 fixed-point conversion coefficients.
    ///
    /// `dst = clamp( (ky*(y - yo) + kv*(cb - 128)*cb_sign ...) >> 14 )` —
    /// the actual per-row tables (see [`conv_coeffs`]) carry the full x/y
    /// offsets so the hot loop stays multiply/add/shift only.
    #[allow(dead_code)]
    const fn q14(v: f64) -> i32 {
        (v * 16384.0).round() as i32
    }
}

/// `Filter` selection for the resize kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Filter {
    /// Box filter (downscale averages, or nearest neighbor for single taps).
    Box,
    /// Bilinear.
    #[default]
    Bilinear,
    /// Cubic convolution (Mitchell-style, B=0.5 C=0.5) with
    /// anti-aliasing on downscale.
    Bicubic,
}

/// A resize request (target size + filter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scale {
    /// Target width in pixels.
    pub width: u32,
    /// Target height in pixels.
    pub height: u32,
    /// Resampling filter.
    pub filter: Filter,
}

impl Scale {
    pub const fn new(width: u32, height: u32, filter: Filter) -> Self {
        Self {
            width,
            height,
            filter,
        }
    }
}

/// Image-processing options shared by the unified API and the backends.
///
/// Semantics (applied in this order):
/// 1. If `scale` is set and the input is high-bit-depth (P010/P012), the
///    frame is first converted to 8-bit limits (10/12 -> 8 downcast).
/// 2. If `scale` is set, the frame (Y'CbCr or RGB) is resized to
///    `scale.width x scale.height` with `scale.filter`.
/// 3. If `affine` is set, the frame is warped using the given transformation.
/// 4. If `rgb` is set, the frame is converted to packed RGB(R)(A) using
///    `spec` (Y'CbCr only; an already-converted frame is passed through).
///
/// Backends with a fast GPU primitive (NVIDIA: NPP) may implement the same
/// pipeline inside the convert-and-scale crate; the *rest* of the workspace
/// uses the SIMD host routines in this crate as the fallback.
#[derive(Debug, Clone, Copy)]
pub struct ImageConfig {
    /// Packed RGB output request (e.g. `Some(RgbChannels::Rgb24)`).
    /// `None` keeps the frame in Y'CbCr.
    pub rgb: Option<RgbChannels>,
    /// Resize request. `None` keeps the decoded size.
    pub scale: Option<Scale>,
    /// Affine warp request. `None` keeps the original geometry.
    pub affine: Option<crate::warp::Affine>,
    /// Color conversion parameters (used when converting to RGB).
    pub spec: ColorSpec,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            rgb: None,
            scale: None,
            affine: None,
            spec: ColorSpec::default(),
        }
    }
}

impl PartialEq for ImageConfig {
    fn eq(&self, other: &Self) -> bool {
        self.rgb == other.rgb
            && self.scale == other.scale
            && self.affine == other.affine
            && self.spec == other.spec
    }
}

impl Eq for ImageConfig {}

impl ImageConfig {
    /// No conversion, no scaling, no warp (legacy behavior).
    pub fn none() -> Self {
        Self::default()
    }

    /// Whether any post-processing is requested.
    pub const fn is_noop(&self) -> bool {
        self.rgb.is_none() && self.scale.is_none() && self.affine.is_none()
    }

    /// Whether a scaled output is requested.
    pub const fn wants_scaled(&self) -> bool {
        self.scale.is_some()
    }

    /// Whether an affine warp is requested.
    pub const fn wants_warp(&self) -> bool {
        self.affine.is_some()
    }

    /// Whether a packed RGB output is requested.
    pub const fn wants_rgb(&self) -> bool {
        self.rgb.is_some()
    }

    /// Number of output channels when RGB is requested (3 or 4).
    pub fn rgb_bytes(&self) -> u8 {
        self.rgb.map_or(0, |r| r.bytes())
    }
}

//! Decoder backend configuration: preferred order with fallback.

use std::fmt;

use vacc_image::ImageConfig;

use crate::backend::Backend;

/// Ordered list of backends to try when creating a
/// [`VaccDecoder`](crate::VaccDecoder), plus the optional post-decode image
/// pipeline ([`ImageConfig`]).
///
/// The first backend that successfully initializes the stream is used; the
/// rest are fallbacks. Duplicates are dropped, order is preserved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderConfig {
    order: Vec<Backend>,
    /// Post-decode image pipeline (scale and/or Y'CbCr -> RGB). No-op by
    /// default: frames are emitted exactly as the backend produces them.
    image: ImageConfig,
}

impl DecoderConfig {
    /// Build a config from an iterable of backends (e.g.
    /// `[Backend::Nvdec, Backend::Software]`). Duplicates are dropped,
    /// preserving first-seen order.
    pub fn new(order: impl IntoIterator<Item = Backend>) -> Self {
        let mut seen = [false; 4];
        let mut out = Vec::new();
        for b in order {
            let idx = match b {
                Backend::Vulkan => 0,
                Backend::Nvdec => 1,
                Backend::Vaapi => 2,
                Backend::Software => 3,
            };
            if !seen[idx] {
                seen[idx] = true;
                out.push(b);
            }
        }
        Self { order: out, image: ImageConfig::default() }
    }

    /// Set the post-decode image pipeline (scale and/or RGB conversion).
    pub fn with_image(mut self, image: ImageConfig) -> Self {
        self.image = image;
        self
    }

    /// The default fallback chain: `vulkan -> nvdec -> vaapi -> software`.
    pub fn default_order() -> Self {
        Self::new(Backend::ALL)
    }

    /// A config with a single backend and no fallback.
    pub fn only(backend: Backend) -> Self {
        Self::new([backend])
    }

    /// The backend order (preferred first).
    pub fn order(&self) -> &[Backend] {
        &self.order
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The configured post-decode image pipeline (copied out; it is `Copy`).
    pub const fn image(&self) -> ImageConfig {
        self.image
    }
}

impl Default for DecoderConfig {
    /// `vulkan -> nvdec -> vaapi -> software`.
    fn default() -> Self {
        Self::default_order()
    }
}

impl FromIterator<Backend> for DecoderConfig {
    fn from_iter<I: IntoIterator<Item = Backend>>(iter: I) -> Self {
        Self::new(iter)
    }
}

impl fmt::Display for DecoderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.order.iter().map(|b| b.name()).collect();
        f.write_str(&names.join(" -> "))?;
        if !self.image.is_noop() {
            write!(f, " [image")?;
            if let Some(s) = self.image.scale {
                write!(f, " scale={}x{}:{}", s.width, s.height, filter_name(s.filter))?;
            }
            if let Some(r) = self.image.rgb {
                write!(f, " rgb={}", if r == vacc_image::RgbChannels::Rgb24 { "rgb24" } else { "rgba32" })?;
            }
            f.write_str("]")?;
        }
        Ok(())
    }
}

fn filter_name(filter: vacc_image::Interpolation) -> &'static str {
    match filter {
        vacc_image::Interpolation::Nearest => "nearest",
        vacc_image::Interpolation::Box => "box",
        vacc_image::Interpolation::Bilinear => "bilinear",
        vacc_image::Interpolation::Bicubic => "bicubic",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_order_is_vulkan_nvdec_vaapi_software() {
        let cfg = DecoderConfig::default();
        assert_eq!(
            cfg.order(),
            &[Backend::Vulkan, Backend::Nvdec, Backend::Vaapi, Backend::Software]
        );
        assert_eq!(cfg.to_string(), "vulkan -> nvdec -> vaapi -> software");
    }

    #[test]
    fn new_dedups_preserving_order() {
        let cfg = DecoderConfig::new([
            Backend::Software,
            Backend::Nvdec,
            Backend::Software,
            Backend::Vulkan,
        ]);
        assert_eq!(cfg.order(), &[Backend::Software, Backend::Nvdec, Backend::Vulkan]);
    }

    #[test]
    fn only_single_backend() {
        let cfg = DecoderConfig::only(Backend::Vaapi);
        assert_eq!(cfg.order(), &[Backend::Vaapi]);
    }
}

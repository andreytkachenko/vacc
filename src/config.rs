//! Decoder backend configuration: preferred order with fallback.

use std::fmt;

use crate::backend::Backend;

/// Ordered list of backends to try when creating a
/// [`VaccDecoder`](crate::VaccDecoder).
///
/// The first backend that successfully initializes the stream is used; the
/// rest are fallbacks. Duplicates are dropped, order is preserved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderConfig {
    order: Vec<Backend>,
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
        Self { order: out }
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
        f.write_str(&names.join(" -> "))
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

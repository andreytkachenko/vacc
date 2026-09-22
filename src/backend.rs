//! Decode backends selectable by the [`DecoderConfig`](crate::DecoderConfig).

use std::fmt;
use std::str::FromStr;

/// A decode backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Vulkan Video (GPU).
    Vulkan,
    /// NVIDIA NVDEC (GPU, CUDA).
    Nvdec,
    /// VAAPI (GPU, libva).
    Vaapi,
    /// Pure-Rust software decode (CPU): H.264 + H.265.
    Software,
}

impl Backend {
    /// All backends in the default fallback order.
    pub const ALL: [Backend; 4] =
        [Backend::Vulkan, Backend::Nvdec, Backend::Vaapi, Backend::Software];

    /// Backend name (as used in logs and config strings).
    pub fn name(self) -> &'static str {
        match self {
            Backend::Vulkan => "vulkan",
            Backend::Nvdec => "nvdec",
            Backend::Vaapi => "vaapi",
            Backend::Software => "software",
        }
    }

    /// Build a backend from a name: `vulkan`, `nvdec`, `vaapi`, `software`
    /// (or `sw`).
    pub fn parse(s: &str) -> Option<Backend> {
        match s.trim().to_ascii_lowercase().as_str() {
            "vulkan" => Some(Backend::Vulkan),
            "nvdec" | "cuda" => Some(Backend::Nvdec),
            "vaapi" => Some(Backend::Vaapi),
            "software" | "sw" | "cpu" => Some(Backend::Software),
            _ => None,
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s).ok_or_else(|| format!("unknown backend '{}' (expected vulkan|nvdec|vaapi|software)", s))
    }
}

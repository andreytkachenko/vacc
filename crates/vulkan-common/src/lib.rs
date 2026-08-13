//! # vulkan-common
//!
//! Common Vulkan device initialization, queue management, and utility types.
//! Used by vulkan-decode and other Vulkan-based crates.

pub mod device;
pub mod queue;
pub mod debug;

pub use device::{VulkanDevice, DeviceBuilder, VideoCodec, QueueFamilies};
pub use queue::QueueHandle;
pub use debug::DebugMessenger;

/// Result type for Vulkan operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Vulkan error types.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Vulkan initialization failed: {0}")]
    Init(String),

    #[error("Device creation failed: {0}")]
    Device(String),

    #[error("Extension not available: {0}")]
    ExtensionMissing(String),

    #[error("Layer not available: {0}")]
    LayerMissing(String),

    #[error("No suitable device found")]
    NoSuitableDevice,

    #[error("Queue not found: {0}")]
    QueueNotFound(String),

    #[error("Memory allocation failed: {0}")]
    Memory(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

//! Vulkan debug messenger utilities.

use ash::vk::{self, Handle};

/// Debug messenger handle.
#[derive(Debug, Clone, Copy)]
pub struct DebugMessenger {
    pub messenger: vk::DebugUtilsMessengerEXT,
}

impl DebugMessenger {
    pub fn is_valid(&self) -> bool {
        !self.messenger.is_null()
    }

    pub fn null() -> Self {
        Self {
            messenger: vk::DebugUtilsMessengerEXT::null(),
        }
    }
}

impl Default for DebugMessenger {
    fn default() -> Self {
        Self::null()
    }
}

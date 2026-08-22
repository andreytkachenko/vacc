//! Queue handle abstraction.

use ash::vk::{self, Handle};

/// A handle to a Vulkan queue with its family index.
#[derive(Debug, Clone, Copy)]
pub struct QueueHandle {
    pub queue: vk::Queue,
    pub family_index: u32,
}

impl QueueHandle {
    pub fn new(queue: vk::Queue, family_index: u32) -> Self {
        Self { queue, family_index }
    }

    pub fn is_null(&self) -> bool {
        self.queue.is_null()
    }
}

//! Bitstream buffer management.
//!
//! This module provides the bitstream buffer abstraction that maps to
//! Vulkan's `VkBuffer` for video decode. The bitstream buffer holds
//! the compressed video data that the hardware decoder reads.

use std::sync::Arc;

/// Packet flags for bitstream data.
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PacketFlags: u32 {
        const END_OF_STREAM = 0x01;
        const TIMESTAMP = 0x02;
        const DISCONTINUITY = 0x04;
        const END_OF_PICTURE = 0x08;
    }
}

/// A packet of bitstream data.
///
/// Corresponds to `VkParserSourceDataPacket` in the Vulkan samples.
#[derive(Debug, Clone)]
pub struct BitstreamPacket {
    /// Packet flags.
    pub flags: PacketFlags,
    /// Payload data.
    pub payload: Vec<u8>,
    /// Presentation timestamp (10MHz clock units).
    pub timestamp: i64,
}

impl BitstreamPacket {
    /// Create a new bitstream packet.
    pub fn new(payload: Vec<u8>) -> Self {
        Self {
            flags: PacketFlags::empty(),
            payload,
            timestamp: 0,
        }
    }

    /// Create a new bitstream packet with timestamp.
    pub fn with_timestamp(payload: Vec<u8>, timestamp: i64) -> Self {
        Self {
            flags: PacketFlags::TIMESTAMP,
            payload,
            timestamp,
        }
    }

    /// Mark as end of stream.
    pub fn end_of_stream() -> Self {
        Self {
            flags: PacketFlags::END_OF_STREAM,
            payload: Vec::new(),
            timestamp: 0,
        }
    }

    /// Check if this is an end-of-stream packet.
    pub fn is_eos(&self) -> bool {
        self.flags.contains(PacketFlags::END_OF_STREAM)
    }

    /// Check if this packet has a valid timestamp.
    pub fn has_timestamp(&self) -> bool {
        self.flags.contains(PacketFlags::TIMESTAMP)
    }
}

/// Bitstream buffer abstraction.
///
/// In the Vulkan implementation, this wraps a `VkBuffer` that holds
/// compressed video data. The hardware decoder reads directly from this buffer.
///
/// In a pure Rust implementation (without Vulkan), this can hold data in
/// host memory.
#[derive(Debug, Clone)]
pub struct BitstreamBuffer {
    /// The underlying data.
    data: Arc<BitstreamData>,
    /// Buffer size alignment requirement.
    pub min_size_alignment: u32,
    /// Buffer offset alignment requirement.
    pub min_offset_alignment: u32,
}

impl BitstreamBuffer {
    /// Create a new bitstream buffer.
    pub fn new(data: Vec<u8>, min_offset_alignment: u32, min_size_alignment: u32) -> Self {
        Self {
            data: Arc::new(BitstreamData { data }),
            min_size_alignment,
            min_offset_alignment,
        }
    }

    /// Create an empty bitstream buffer with the given capacity.
    pub fn with_capacity(
        capacity: usize,
        min_offset_alignment: u32,
        min_size_alignment: u32,
    ) -> Self {
        Self {
            data: Arc::new(BitstreamData {
                data: Vec::with_capacity(capacity),
            }),
            min_size_alignment,
            min_offset_alignment,
        }
    }

    /// Get the buffer data slice.
    pub fn data(&self) -> &[u8] {
        &self.data.data
    }

    /// Get the current size.
    pub fn len(&self) -> usize {
        self.data.data.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.data.is_empty()
    }


    /// Clone the buffer with new size.
    pub fn clone_with_size(&self, new_size: usize) -> Self {
        let mut new_data = self.data.data.clone();
        new_data.resize(new_size, 0);
        Self {
            data: Arc::new(BitstreamData { data: new_data }),
            min_size_alignment: self.min_size_alignment,
            min_offset_alignment: self.min_offset_alignment,
        }
    }

    /// Get a shared reference (for sharing between parser and decoder).
    pub fn shared(&self) -> Arc<BitstreamData> {
        Arc::clone(&self.data)
    }

    /// Add a stream marker at the given offset.
    ///
    /// Stream markers indicate slice boundaries in the bitstream.
    pub fn add_stream_marker(&self, _offset: u32) -> u32 {
        // In the Vulkan implementation, this tracks slice offsets
        // for random access.
        0
    }

    /// Reset stream markers.
    pub fn reset_stream_markers(&self) {
        // In the Vulkan implementation, this resets the marker list.
    }

    /// Flush the buffer (make data visible to the GPU).
    ///
    /// In the Vulkan implementation, this calls `vkFlushMappedMemoryRanges`.
    pub fn flush(&self) {
        // No-op for host-memory buffer.
    }

    /// Invalidate the buffer (make CPU-visible data fresh).
    ///
    /// In the Vulkan implementation, this calls `vkInvalidateMappedMemoryRanges`.
    pub fn invalidate(&self) {
        // No-op for host-memory buffer.
    }
}

/// Shared bitstream data container.
#[derive(Debug)]
pub struct BitstreamData {
    pub data: Vec<u8>,
}

impl std::ops::Deref for BitstreamBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data.data
    }
}

/// Bitstream stream abstraction for incremental parsing.
///
/// Similar to `VulkanBitstreamBufferStream` in the Vulkan samples.
pub struct BitstreamStream<'a> {
    buffer: &'a mut BitstreamBuffer,
    max_access: usize,
}

impl<'a> BitstreamStream<'a> {
    /// Create a new bitstream stream.
    pub fn new(buffer: &'a mut BitstreamBuffer) -> Self {
        Self {
            buffer,
            max_access: 0,
        }
    }

    /// Get the buffer size.
    pub fn max_size(&self) -> usize {
        self.buffer.len()
    }

    /// Get a reference to the byte at the given index.
    pub fn get(&self, index: usize) -> Option<u8> {
        if index < self.buffer.len() {
            Some(self.buffer.data()[index])
        } else {
            None
        }
    }

    /// Commit the buffer (flush changes).
    pub fn commit(&mut self) -> usize {
        let commit_size = self.max_access;
        self.max_access = 0;
        self.buffer.flush();
        commit_size
    }

    /// Check if a start code exists at the given offset.
    pub fn has_slice_start_code_at_offset(&self, index: usize) -> bool {
        if index + 2 >= self.buffer.len() {
            return false;
        }
        self.buffer.data()[index] == 0
            && self.buffer.data()[index + 1] == 0
            && self.buffer.data()[index + 2] == 1
    }

    /// Get a pointer to the bitstream data.
    pub fn data_ptr(&self) -> &[u8] {
        self.buffer.data()
    }
}

impl std::ops::Index<usize> for BitstreamStream<'_> {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        &self.buffer.data()[index]
    }
}


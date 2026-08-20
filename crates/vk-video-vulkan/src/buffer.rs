//! Bitstream buffer management.

use super::{VideoError, VideoResult};
use ash::vk::Handle;

/// Bitstream buffer for video decode.
pub struct BitstreamBuffer {
    buffer: ash::vk::Buffer,
    memory: ash::vk::DeviceMemory,
    size: u64,
    offset_alignment: u32,
    size_alignment: u32,
    device: ash::Device,
    #[allow(dead_code)]
    memory_properties: ash::vk::PhysicalDeviceMemoryProperties,
    mapped_ptr: Option<*mut u8>,
}

impl BitstreamBuffer {
    /// Create a null BitstreamBuffer (for cleanup purposes).
    pub fn null(device: &ash::Device) -> Self {
        Self {
            buffer: ash::vk::Buffer::null(),
            memory: ash::vk::DeviceMemory::null(),
            size: 0,
            offset_alignment: 0,
            size_alignment: 0,
            device: device.clone(),
            memory_properties: ash::vk::PhysicalDeviceMemoryProperties::default(),
            mapped_ptr: None,
        }
    }

    pub fn create(
        device: &ash::Device,
        memory_properties: &ash::vk::PhysicalDeviceMemoryProperties,
        size: u64,
        offset_alignment: u32,
        size_alignment: u32,
        queue_family_index: u32,
    ) -> VideoResult<Self> {
        Self::create_with_pnext(
            device,
            memory_properties,
            size,
            offset_alignment,
            size_alignment,
            std::ptr::null(),
            ash::vk::BufferCreateFlags::empty(),
            queue_family_index,
        )
    }

    /// Create a bitstream buffer with a pNext chain (e.g., VkVideoProfileListInfoKHR)
    /// and optional create flags (e.g., VIDEO_PROFILE_INDEPENDENT_KHR).
    ///
    /// The buffer is owned by `queue_family_index` (EXCLUSIVE sharing). The C++
    /// reference (VulkanBistreamBufferImpl.cpp) creates the bitstream buffer owned
    /// by the video decode queue family (queueFamilyIndexCount = 1); with count = 0
    /// the owning family is left to the driver, which on NVIDIA can make the decode
    /// queue unable to read the buffer -> decode silently skipped (all-zero DPB).
    pub fn create_with_pnext(
        device: &ash::Device,
        memory_properties: &ash::vk::PhysicalDeviceMemoryProperties,
        size: u64,
        offset_alignment: u32,
        size_alignment: u32,
        p_next: *const std::ffi::c_void,
        flags: ash::vk::BufferCreateFlags,
        queue_family_index: u32,
    ) -> VideoResult<Self> {
        let aligned_size = Self::aligned_size(size, size_alignment);

        let buffer_create_info = ash::vk::BufferCreateInfo {
            s_type: ash::vk::StructureType::BUFFER_CREATE_INFO,
            p_next,
            flags,
            size: aligned_size,
            usage: ash::vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR
                | ash::vk::BufferUsageFlags::TRANSFER_DST,
            sharing_mode: ash::vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: &queue_family_index as *const u32,
            _marker: std::marker::PhantomData,
        };

        let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }
            .map_err(|e| VideoError::BufferAllocation(e.to_string()))?;

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let mem_type_index = Self::find_memory_type(
            memory_properties,
            mem_requirements.memory_type_bits,
            ash::vk::MemoryPropertyFlags::HOST_VISIBLE
                | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| VideoError::MemoryAllocation("No suitable memory type found".to_string()))?;

        let alloc_info = ash::vk::MemoryAllocateInfo {
            s_type: ash::vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            allocation_size: mem_requirements.size,
            memory_type_index: mem_type_index,
            _marker: std::marker::PhantomData,
        };

        let memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .map_err(|e| VideoError::MemoryAllocation(e.to_string()))?;

        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| VideoError::BufferAllocation(e.to_string()))?;

        let mapped_ptr = unsafe {
            device.map_memory(
                memory,
                0,
                ash::vk::WHOLE_SIZE,
                ash::vk::MemoryMapFlags::empty(),
            )
        }
        .map(|p| p as *mut u8)
        .ok();

        Ok(Self {
            buffer,
            memory,
            size: aligned_size,
            offset_alignment,
            size_alignment,
            device: device.clone(),
            memory_properties: *memory_properties,
            mapped_ptr,
        })
    }

    pub fn create_pool(
        device: &ash::Device,
        memory_properties: &ash::vk::PhysicalDeviceMemoryProperties,
        count: usize,
        buffer_size: u64,
        offset_alignment: u32,
        size_alignment: u32,
        queue_family_index: u32,
    ) -> VideoResult<BitstreamBufferPool> {
        let mut buffers = Vec::with_capacity(count);
        for _ in 0..count {
            buffers.push(Self::create(
                device,
                memory_properties,
                buffer_size,
                offset_alignment,
                size_alignment,
                queue_family_index,
            )?);
        }

        Ok(BitstreamBufferPool { buffers })
    }

    fn find_memory_type(
        mem_props: &ash::vk::PhysicalDeviceMemoryProperties,
        type_bits: u32,
        required_flags: ash::vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        (0..mem_props.memory_type_count).find(|&i| {
            (type_bits & (1 << i)) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(required_flags)
        })
    }

    fn aligned_size(size: u64, alignment: u32) -> u64 {
        let align = alignment as u64;
        (size + align - 1) & !(align - 1)
    }

    pub fn buffer(&self) -> ash::vk::Buffer {
        self.buffer
    }

    pub fn memory(&self) -> ash::vk::DeviceMemory {
        self.memory
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn offset_alignment(&self) -> u32 {
        self.offset_alignment
    }

    pub fn size_alignment(&self) -> u32 {
        self.size_alignment
    }

    pub fn write(&mut self, data: &[u8]) -> VideoResult<()> {
        if let Some(ptr) = self.mapped_ptr {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
        } else {
            return Err(VideoError::BufferAllocation(
                "Buffer is not host-mapped".to_string(),
            ));
        }
        Ok(())
    }

    /// Write data at a specific offset in the mapped buffer.
    pub fn write_at(&mut self, data: &[u8], offset: u64) -> VideoResult<()> {
        if let Some(ptr) = self.mapped_ptr {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset as usize), data.len());
            }
        } else {
            return Err(VideoError::BufferAllocation(
                "Buffer is not host-mapped".to_string(),
            ));
        }
        Ok(())
    }

    pub fn data_ptr(&self) -> Option<*mut u8> {
        self.mapped_ptr
    }

    /// Zero out a range in the mapped buffer.
    pub fn zero_range(&self, offset: u64, size: u64) {
        if let Some(ptr) = self.mapped_ptr {
            unsafe {
                std::ptr::write_bytes(ptr.add(offset as usize), 0, size as usize);
            }
        }
    }

    pub fn flush_range(&self, offset: u64, size: u64) -> VideoResult<()> {
        unsafe {
            self.device
                .flush_mapped_memory_ranges(&[ash::vk::MappedMemoryRange {
                    s_type: ash::vk::StructureType::MAPPED_MEMORY_RANGE,
                    p_next: std::ptr::null(),
                    memory: self.memory,
                    offset,
                    size: if size == 0 { ash::vk::WHOLE_SIZE } else { size },
                    _marker: std::marker::PhantomData,
                }])
        }
        .map_err(|e| VideoError::BufferAllocation(e.to_string()))
    }

    pub fn invalidate_range(&self, offset: u64, size: u64) -> VideoResult<()> {
        unsafe {
            self.device
                .invalidate_mapped_memory_ranges(&[ash::vk::MappedMemoryRange {
                    s_type: ash::vk::StructureType::MAPPED_MEMORY_RANGE,
                    p_next: std::ptr::null(),
                    memory: self.memory,
                    offset,
                    size: if size == 0 { ash::vk::WHOLE_SIZE } else { size },
                    _marker: std::marker::PhantomData,
                }])
        }
        .map_err(|e| VideoError::BufferAllocation(e.to_string()))
    }
}

impl Drop for BitstreamBuffer {
    fn drop(&mut self) {
        if self.mapped_ptr.is_some() {
            unsafe {
                self.device.unmap_memory(self.memory);
            }
        }

        if !self.memory.is_null() {
            unsafe {
                self.device.free_memory(self.memory, None);
            }
        }

        if !self.buffer.is_null() {
            unsafe {
                self.device.destroy_buffer(self.buffer, None);
            }
        }
    }
}

/// Pool of bitstream buffers for efficient allocation.
pub struct BitstreamBufferPool {
    buffers: Vec<BitstreamBuffer>,
}

impl BitstreamBufferPool {
    pub fn get(&self, index: usize) -> Option<&BitstreamBuffer> {
        self.buffers.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut BitstreamBuffer> {
        self.buffers.get_mut(index)
    }

    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &BitstreamBuffer> {
        self.buffers.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut BitstreamBuffer> {
        self.buffers.iter_mut()
    }
}

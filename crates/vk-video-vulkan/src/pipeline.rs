//! Video decode pipeline orchestration with proper command pool,
//! session management, and synchronization.

use super::{VideoError, VideoResult, device::VideoCodec};

#[derive(Debug, Clone)]
pub struct VideoPipelineConfig {
    pub codec: VideoCodec,
    pub picture_format: ash::vk::Format,
    pub reference_picture_format: ash::vk::Format,
    pub max_coded_extent: (u32, u32),
    pub max_dpb_slots: u32,
    pub max_active_reference_pictures: u32,
    pub bitstream_buffer_size: u64,
    pub bitstream_buffer_pool_size: usize,
}

impl Default for VideoPipelineConfig {
    fn default() -> Self {
        Self {
            codec: VideoCodec::DecodeH264,
            picture_format: ash::vk::Format::G8_B8R8_2PLANE_420_UNORM,
            reference_picture_format: ash::vk::Format::G8_B8R8_2PLANE_420_UNORM,
            max_coded_extent: (1920, 1088),
            max_dpb_slots: 16,
            max_active_reference_pictures: 16,
            bitstream_buffer_size: 1024 * 1024,
            bitstream_buffer_pool_size: 4,
        }
    }
}

/// A complete Vulkan video decode pipeline.
pub struct VideoPipeline {
    device: ash::Device,
    instance: ash::Instance,
    memory_properties: ash::vk::PhysicalDeviceMemoryProperties,
    decode_queue_family: u32,
    decode_queue: ash::vk::Queue,
    command_pool: ash::vk::CommandPool,
    command_buffer: ash::vk::CommandBuffer,
    fence: ash::vk::Fence,
    session: Option<super::session::VideoSession>,
    session_parameters: Option<super::session::VideoSessionParameters>,
    bitstream_buffers: Option<super::buffer::BitstreamBufferPool>,
    decoder: Option<DecoderWrapper>,
    output_images: Vec<OutputImageInfo>,
    config: VideoPipelineConfig,
    frame_count: u32,
}

#[derive(Debug, Clone)]
struct OutputImageInfo {
    image: ash::vk::Image,
    image_view: ash::vk::ImageView,
    memory: ash::vk::DeviceMemory,
}

pub enum DecoderWrapper {
    H264(super::h264::H264Decoder),
    H265(super::h265::H265Decoder),
    Av1(super::av1::Av1Decoder),
    Vp9(super::vp9::Vp9Decoder),
}

impl VideoPipeline {
    pub fn new(device: &super::device::VulkanDevice, codec: VideoCodec) -> VideoResult<Self> {
        let config = VideoPipelineConfig {
            codec,
            ..VideoPipelineConfig::default()
        };

        let decode_queue_family = device
            .queue_families
            .video_decode
            .ok_or_else(|| VideoError::VideoNotSupported("No decode queue".to_string()))?;

        let decode_queue = unsafe {
            device.device.get_device_queue(decode_queue_family, 0)
        };

        Ok(Self {
            device: device.device.clone(),
            instance: device.instance.clone(),
            memory_properties: device.memory_properties.clone(),
            decode_queue_family,
            decode_queue,
            command_pool: ash::vk::CommandPool::null(),
            command_buffer: ash::vk::CommandBuffer::null(),
            fence: ash::vk::Fence::null(),
            session: None,
            session_parameters: None,
            bitstream_buffers: None,
            decoder: None,
            output_images: Vec::new(),
            config,
            frame_count: 0,
        })
    }

    /// Initialize the pipeline: create session, parameters, command resources.
    pub fn init(&mut self) -> VideoResult<()> {
        // Create video session with proper profile chain
        let session_params = super::session::VideoSessionParams {
            flags: 0,
            queue_family_index: self.decode_queue_family,
            picture_format: self.config.picture_format,
            reference_picture_format: self.config.reference_picture_format,
            max_coded_extent: self.config.max_coded_extent,
            max_dpb_slots: self.config.max_dpb_slots,
            max_active_reference_pictures: self.config.max_active_reference_pictures,
            codec_profile_info: None, // Will be set by codec-specific init
        };

        let session = super::session::VideoSession::create(
            &self.device, &self.instance, &session_params,
        )?;
        self.session = Some(session);

        // Create codec-specific decoder
        self.decoder = Some(match self.config.codec {
            VideoCodec::DecodeH264 => {
                DecoderWrapper::H264(super::h264::H264Decoder::new(
                    self.device.clone(), self.instance.clone(),
                ))
            }
            VideoCodec::DecodeH265 => {
                DecoderWrapper::H265(super::h265::H265Decoder::new(
                    self.device.clone(), self.instance.clone(),
                ))
            }
            VideoCodec::DecodeAv1 => {
                DecoderWrapper::Av1(super::av1::Av1Decoder::new(
                    self.device.clone(), self.instance.clone(),
                ))
            }
            VideoCodec::DecodeVp9 => {
                DecoderWrapper::Vp9(super::vp9::Vp9Decoder::new(
                    self.device.clone(), self.instance.clone(),
                ))
            }
        });

        // Link session to decoder
        if let (Some(ref mut decoder), Some(ref session)) = (&mut self.decoder, &self.session) {
            match decoder {
                DecoderWrapper::H264(d) => d.set_session(session),
                DecoderWrapper::H265(d) => d.set_session(session),
                DecoderWrapper::Av1(d) => d.set_session(session),
                DecoderWrapper::Vp9(d) => d.set_session(session),
            }
        }

        // Create session parameters (codec-specific)
        if let Some(ref mut decoder) = self.decoder {
            let session_params = match decoder {
                DecoderWrapper::H264(d) => d.create_session_parameters()?,
                DecoderWrapper::H265(d) => d.create_session_parameters()?,
                DecoderWrapper::Av1(d) => d.create_session_parameters()?,
                DecoderWrapper::Vp9(d) => d.create_session_parameters()?,
            };
            self.session_parameters = Some(session_params);

            // Link session parameters to decoder
            if let Some(ref params) = self.session_parameters {
                match decoder {
                    DecoderWrapper::H264(d) => d.set_session_parameters(params.clone()),
                    DecoderWrapper::H265(d) => d.set_session_parameters(params.clone()),
                    DecoderWrapper::Av1(d) => d.set_session_parameters(params.clone()),
                    DecoderWrapper::Vp9(d) => d.set_session_parameters(params.clone()),
                }
            }
        }

        // Create command pool
        self.command_pool = unsafe {
            self.device.create_command_pool(
                &ash::vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.decode_queue_family)
                    .flags(ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            ).map_err(|e| VideoError::CommandBufferRecording(format!("Command pool creation failed: {:?}", e)))?
        };

        // Allocate command buffer
        let buffers = unsafe {
            self.device.allocate_command_buffers(
                &ash::vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(ash::vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            ).map_err(|e| VideoError::CommandBufferRecording(format!("Command buffer allocation failed: {:?}", e)))?
        };
        self.command_buffer = buffers[0];

        // Create fence (initially signaled for first use)
        self.fence = unsafe {
            self.device.create_fence(
                &ash::vk::FenceCreateInfo::default()
                    .flags(ash::vk::FenceCreateFlags::SIGNALED),
                None,
            ).map_err(|e| VideoError::FenceWait(format!("Fence creation failed: {:?}", e)))?
        };

        // Create bitstream buffer pool
        let bitstream_buffers = super::buffer::BitstreamBuffer::create_pool(
            &self.device,
            &self.memory_properties,
            self.config.bitstream_buffer_pool_size,
            self.config.bitstream_buffer_size,
            256,
            256,
        )?;
        self.bitstream_buffers = Some(bitstream_buffers);

        Ok(())
    }

    /// Update session parameters with codec-specific SPS/PPS/VPS data.
    pub fn update_session_parameters(
        &mut self,
        sps_h264: Option<&vk_video_core::picture::H264Sps>,
        pps_h264: Option<&vk_video_core::picture::H264Pps>,
        vps_h265: Option<&vk_video_core::picture::H265Vps>,
        sps_h265: Option<&vk_video_core::picture::H265Sps>,
        pps_h265: Option<&vk_video_core::picture::H265Pps>,
    ) -> VideoResult<()> {
        let session_params = self.session_parameters
            .as_ref()
            .ok_or_else(|| VideoError::InvalidState("No session parameters".to_string()))?
            .parameters();

        match self.config.codec {
            VideoCodec::DecodeH264 => {
                if let Some(ref mut decoder) = self.decoder {
                    if let DecoderWrapper::H264(d) = decoder {
                        d.update_session_parameters(session_params, sps_h264, pps_h264)?;
                    }
                }
            }
            VideoCodec::DecodeH265 => {
                if let Some(ref mut decoder) = self.decoder {
                    if let DecoderWrapper::H265(d) = decoder {
                        d.update_session_parameters(session_params, vps_h265, sps_h265, pps_h265)?;
                    }
                }
            }
            VideoCodec::DecodeAv1 => {}
            VideoCodec::DecodeVp9 => {}
        }

        Ok(())
    }

    /// Create output images for decoded frames.
    pub fn create_output_images(&mut self, count: u32, width: u32, height: u32) -> VideoResult<()> {
        for _ in 0..count {
            let (image, image_view, memory) = super::image::create_output_image(
                &self.device,
                &self.memory_properties,
                width,
                height,
                ash::vk::Format::G8_B8R8_2PLANE_420_UNORM,
            )?;
            self.output_images.push(OutputImageInfo {
                image,
                image_view,
                memory,
            });
        }
        Ok(())
    }

    /// Decode a single frame using the bitstream data.
    pub fn decode_frame(
        &mut self,
        bitstream_data: &[u8],
        frame_index: u32,
    ) -> VideoResult<Option<ash::vk::ImageView>> {
        let session = self.session
            .as_ref()
            .ok_or_else(|| VideoError::InvalidState("No video session".to_string()))?
            .session();

        let session_params = self.session_parameters
            .as_ref()
            .ok_or_else(|| VideoError::InvalidState("No session parameters".to_string()))?
            .parameters();

        // Get an output image
        let output_idx = frame_index as usize % self.output_images.len();
        let output = &self.output_images[output_idx];

        // Upload bitstream data to a bitstream buffer
        let bitstream_buffer = self.bitstream_buffers
            .as_ref()
            .ok_or_else(|| VideoError::InvalidState("No bitstream buffers".to_string()))?
            .get(0)
            .ok_or_else(|| VideoError::InvalidState("No bitstream buffer available".to_string()))?;

        let mut buffers = self.bitstream_buffers.as_mut().unwrap();
        let mut bs_buf = buffers.get_mut(0).unwrap();
        bs_buf.write(bitstream_data)?;
        bs_buf.flush_range(0, bitstream_data.len() as u64)?;

        let bs_buffer = bs_buf.buffer();
        let bs_size = bitstream_data.len() as u64;

        // Reset fence
        unsafe {
            self.device.reset_fence(self.fence)
                .map_err(|e| VideoError::FenceWait(format!("Fence reset failed: {:?}", e)))?;
        }

        // Record decode command
        if let Some(ref decoder) = self.decoder {
            match decoder {
                DecoderWrapper::H264(d) => {
                    d.record_decode_command(
                        self.command_buffer,
                        session,
                        session_params,
                        bs_buffer,
                        0,
                        bs_size,
                        output.image_view,
                        output.image,
                        ash::vk::Extent2D {
                            width: self.config.max_coded_extent.0,
                            height: self.config.max_coded_extent.1,
                        },
                        None,
                        &[],
                        &[],
                        None,
                        None,
                        None,
                        None,
                    )?;
                }
                DecoderWrapper::H265(d) => {
                    d.record_decode_command(
                        self.command_buffer,
                        session,
                        session_params,
                        bs_buffer,
                        0,
                        bs_size,
                        output.image_view,
                        output.image,
                        ash::vk::Extent2D {
                            width: self.config.max_coded_extent.0,
                            height: self.config.max_coded_extent.1,
                        },
                        None,
                        &[],
                        &[],
                        None,
                        None,
                        None,
                    )?;
                }
                DecoderWrapper::Av1(d) => {
                    d.record_decode_command(
                        self.command_buffer,
                        session,
                        session_params,
                        bs_buffer,
                        0,
                        bs_size,
                        output.image_view,
                        output.image,
                        ash::vk::Extent2D {
                            width: self.config.max_coded_extent.0,
                            height: self.config.max_coded_extent.1,
                        },
                    )?;
                }
                DecoderWrapper::Vp9(d) => {
                    // VP9 decode requires parsed frame data; use direct API for now
                    // Pipeline integration for VP9 needs frame parser integration
                    let _ = (d, bs_buffer, bs_size, output, session, session_params);
                }
            }
        }

        // Submit command buffer
        unsafe {
            self.device.queue_submit(
                self.decode_queue,
                &[ash::vk::SubmitInfo::default()
                    .command_buffers(&[self.command_buffer])],
                self.fence,
            ).map_err(|e| VideoError::QueueSubmission(format!("Queue submit failed: {:?}", e)))?;

            // Wait for completion
            self.device.wait_for_fences(&[self.fence], true, 5_000_000_000)
                .map_err(|e| VideoError::FenceWait(format!("Fence wait failed: {:?}", e)))?;
        }

        self.frame_count += 1;
        Ok(Some(output.image_view))
    }

    /// Decode a single VP9 frame with parsed frame data.
    ///
    /// Unlike the generic [`decode_frame`](Self::decode_frame), this method accepts
    /// VP9-specific parsed frame information (DPB references, picture info, offsets,
    /// etc.) so the decoder can record a proper VP9 decode command.
    pub fn decode_vp9_frame(
        &mut self,
        bitstream_data: &[u8],
        frame_index: u32,
        picture_info_container: &super::vp9::Vp9PictureInfoContainer,
        vp9_decode_info: &super::vp9::VideoDecodeVP9PictureInfoKHR,
        dpb_setup_picture: Option<ash::vk::VideoPictureResourceInfoKHR<'static>>,
        dpb_ref_pictures: &[ash::vk::VideoPictureResourceInfoKHR<'static>],
        dpb_ref_slot_indices: &[i32],
        output_slot_index: i32,
        is_first_frame: bool,
    ) -> VideoResult<Option<ash::vk::ImageView>> {
        let session = self.session
            .as_ref()
            .ok_or_else(|| VideoError::InvalidState("No video session".to_string()))?
            .session();

        let session_params = self.session_parameters
            .as_ref()
            .ok_or_else(|| VideoError::InvalidState("No session parameters".to_string()))?
            .parameters();

        // Get an output image
        let output_idx = frame_index as usize % self.output_images.len();
        let output = &self.output_images[output_idx];

        // Upload bitstream data to a bitstream buffer
        let mut buffers = self.bitstream_buffers.as_mut().unwrap();
        let mut bs_buf = buffers.get_mut(0).unwrap();
        bs_buf.write(bitstream_data)?;
        bs_buf.flush_range(0, bitstream_data.len() as u64)?;

        let bs_buffer = bs_buf.buffer();
        let bs_size = bitstream_data.len() as u64;

        // Reset fence
        unsafe {
            self.device.reset_fence(self.fence)
                .map_err(|e| VideoError::FenceWait(format!("Fence reset failed: {:?}", e)))?;
        }

        // Record VP9 decode command
        if let Some(ref mut decoder) = self.decoder {
            if let DecoderWrapper::Vp9(d) = decoder {
                d.record_decode_command(
                    self.command_buffer,
                    session,
                    session_params,
                    bs_buffer,
                    0,
                    bs_size,
                    output.image_view,
                    output.image,
                    ash::vk::Extent2D {
                        width: self.config.max_coded_extent.0,
                        height: self.config.max_coded_extent.1,
                    },
                    dpb_setup_picture,
                    dpb_ref_pictures,
                    dpb_ref_slot_indices,
                    picture_info_container,
                    vp9_decode_info,
                    is_first_frame,
                    output_slot_index,
                )?;
            }
        }

        // Submit command buffer
        unsafe {
            self.device.queue_submit(
                self.decode_queue,
                &[ash::vk::SubmitInfo::default()
                    .command_buffers(&[self.command_buffer])],
                self.fence,
            ).map_err(|e| VideoError::QueueSubmission(format!("Queue submit failed: {:?}", e)))?;

            // Wait for completion
            self.device.wait_for_fences(&[self.fence], true, 5_000_000_000)
                .map_err(|e| VideoError::FenceWait(format!("Fence wait failed: {:?}", e)))?;
        }

        self.frame_count += 1;
        Ok(Some(output.image_view))
    }

    /// Read back decoded YUV data from an output image to a staging buffer.
    pub fn readback_frame(&self, frame_index: u32, width: u32, height: u32) -> VideoResult<(Vec<u8>, Vec<u8>)> {
        let output_idx = frame_index as usize % self.output_images.len();
        let output = &self.output_images[output_idx];

        // For now, return empty data since proper readback requires
        // a staging image and transfer command buffer
        // This is a placeholder for the full implementation
        let y_size = (width * height) as usize;
        let uv_size = ((width / 2) * (height / 2) * 2) as usize;
        Ok((vec![0u8; y_size], vec![0u8; uv_size]))
    }

    pub fn session(&self) -> Option<&super::session::VideoSession> {
        self.session.as_ref()
    }

    pub fn decoder(&self) -> Option<&DecoderWrapper> {
        self.decoder.as_ref()
    }

    pub fn decoder_mut(&mut self) -> Option<&mut DecoderWrapper> {
        self.decoder.as_mut()
    }

    pub fn get_bitstream_buffer(&self, index: usize) -> Option<&super::buffer::BitstreamBuffer> {
        self.bitstream_buffers.as_ref().and_then(|b| b.get(index))
    }

    pub fn bitstream_buffers(&self) -> Option<&super::buffer::BitstreamBufferPool> {
        self.bitstream_buffers.as_ref()
    }

    pub fn next_frame(&mut self) {
        self.frame_count += 1;
    }

    pub fn frame_count(&self) -> u32 {
        self.frame_count
    }

    pub fn config(&self) -> &VideoPipelineConfig {
        &self.config
    }

    pub fn decode_queue_family(&self) -> u32 {
        self.decode_queue_family
    }

    pub fn command_pool(&self) -> ash::vk::CommandPool {
        self.command_pool
    }

    pub fn command_buffer(&self) -> ash::vk::CommandBuffer {
        self.command_buffer
    }

    pub fn fence(&self) -> ash::vk::Fence {
        self.fence
    }
}

impl Drop for VideoPipeline {
    fn drop(&mut self) {
        unsafe {
            // Wait for idle before cleanup
            if !self.fence.is_null() {
                let _ = self.device.wait_for_fences(&[self.fence], true, 1_000_000_000);
            }
            if !self.command_pool.is_null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
        }
    }
}

/// Decode info for recording commands.
#[derive(Debug, Clone)]
pub struct VideoDecodeInfo {
    pub bitstream_buffer: ash::vk::Buffer,
    pub bitstream_offset: u64,
    pub bitstream_range: u64,
    pub output_image: ash::vk::ImageView,
    pub output_layout: ash::vk::ImageLayout,
    pub dpb_references: Vec<ash::vk::VideoPictureResourceInfoKHR<'static>>,
}

impl VideoDecodeInfo {
    pub fn new(
        bitstream_buffer: ash::vk::Buffer,
        bitstream_offset: u64,
        bitstream_range: u64,
        output_image: ash::vk::ImageView,
        output_layout: ash::vk::ImageLayout,
    ) -> Self {
        Self {
            bitstream_buffer,
            bitstream_offset,
            bitstream_range,
            output_image,
            output_layout,
            dpb_references: Vec::new(),
        }
    }
}

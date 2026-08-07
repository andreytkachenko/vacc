//! VP9 Vulkan video decoder.
//!
//! Implements VP9 decode command recording using Vulkan Video extension.
//! Aligned with NVIDIA's Vulkan-Video-Samples VP9 decoder (VulkanVP9Decoder.cpp).
//!
//! All StdVideo* structs match vulkan_video_codec_vp9std.h and
//! vulkan_video_codec_vp9std_decode.h exactly.

use ash::vk;
use ash::vk::Handle;

use super::device::VideoCodec;
use super::{VideoError, VideoResult};

// ============================================================================
// VP9 Decoder
// ============================================================================

/// VP9 decoder state.
pub struct Vp9Decoder {
    device: ash::Device,
    instance: ash::Instance,
    /// Session handle (set via set_session()).
    session: vk::VideoSessionKHR,
    /// Session parameters (set via set_session_parameters()).
    session_params: Option<super::session::VideoSessionParameters>,
    /// Frame counter.
    frame_count: u32,
    /// DPB reference frame name to picture index mapping.
    /// Maps VP9 reference frame names (LAST, GOLDEN, ALTREF, LAST2, LAST3, BACKWARD, KEY)
    /// to picture indices. Updated after each decode via update_frame_pointers().
    /// -1 means the reference frame name is not currently assigned.
    pic_idx: [i32; vk_video_core::picture::VP9_NUM_REF_FRAMES as usize],
    /// Maximum DPB slots available.
    max_dpb_slots: u32,
}

impl Vp9Decoder {
    pub fn new(device: ash::Device, instance: ash::Instance) -> Self {
        Self {
            device,
            instance,
            session: vk::VideoSessionKHR::null(),
            session_params: None,
            frame_count: 0,
            pic_idx: [-1; vk_video_core::picture::VP9_NUM_REF_FRAMES as usize],
            max_dpb_slots: 8, // Default VP9 DPB size
        }
    }

    /// Set the maximum DPB slots.
    pub fn set_max_dpb_slots(&mut self, max_dpb_slots: u32) {
        self.max_dpb_slots = max_dpb_slots;
    }

    /// Update reference frame pointers after decode.
    ///
    /// Per VP9 spec section 7.10: When a frame is decoded, the reference frame
    /// slots specified by refresh_frame_flags are updated to point to the
    /// current decoded frame.
    ///
    /// # Arguments
    /// * `refresh_frame_flags` - Bitmask of which ref frame names to update
    /// * `current_pic_idx` - Picture index of the current decoded frame
    pub fn update_frame_pointers(&mut self, refresh_frame_flags: u8, current_pic_idx: i32) {
        let mut mask = refresh_frame_flags;
        let mut ref_index: u32 = 0;
        while mask != 0 {
            if mask & 1 != 0 {
                if ref_index < vk_video_core::picture::VP9_NUM_REF_FRAMES as u32 {
                    self.pic_idx[ref_index as usize] = current_pic_idx;
                }
            }
            mask >>= 1;
            ref_index += 1;
        }
    }

    /// Compute reference_name_slot_indices for Vulkan decode command.
    ///
    /// Maps the 3 VP9 primary reference frame names (LAST, GOLDEN, ALTREF)
    /// to their DPB slot indices via pic_idx.
    ///
    /// Per the Vulkan spec, referenceNameSlotIndices[i] is the DPB slot index
    /// for the i-th reference frame name (0=LAST, 1=GOLDEN, 2=ALTREF), or -1
    /// if that reference frame name is not currently assigned.
    pub fn compute_reference_name_slot_indices(&self, is_key_frame: bool) -> [i32; 3] {
        if is_key_frame {
            return [-1, -1, -1];
        }
        // pic_idx[i] holds the DPB slot index for reference frame name i,
        // or -1 if that name is not currently assigned to any frame.
        // Directly map the 3 primary reference names (LAST=0, GOLDEN=1, ALTREF=2).
        [
            self.pic_idx[0], // LAST_FRAME
            self.pic_idx[1], // GOLDEN_FRAME
            self.pic_idx[2], // ALTREF_FRAME
        ]
    }

    /// Reset DPB state (e.g., on key frame or discontinuity).
    pub fn reset_dpb(&mut self) {
        self.pic_idx.fill(-1);
    }

    /// Set the session handle.
    pub fn set_session(&mut self, session: &super::session::VideoSession) {
        self.session = session.handle();
    }

    /// Set the session parameters handle.
    pub fn set_session_parameters(&mut self, params: super::session::VideoSessionParameters) {
        self.session_params = Some(params);
    }

    /// Create session parameters for VP9.
    pub fn create_session_parameters(&self) -> VideoResult<super::session::VideoSessionParameters> {
        if self.session.is_null() {
            return Err(VideoError::SessionCreation(
                "VP9: session must be set before creating session parameters".to_string(),
            ));
        }
        super::session::VideoSessionParameters::create(
            &self.instance,
            &self.device,
            self.session,
            VideoCodec::DecodeVp9,
        )
    }

    /// Record a VP9 decode command.
    ///
    /// Matches the command recording flow of H.264/H.265 decoders:
    /// 1. Begin command buffer
    /// 2. Begin video coding (activate DPB slots)
    /// 3. Reset decoder on first frame (inside coding block)
    /// 4. Memory barriers (after Begin, before Decode)
    /// 5. Decode command
    /// 6. End video coding
    /// 7. End command buffer
    ///
    /// # Slot index semantics
    ///
    /// `output_slot_index` and `dpb_ref_slot_indices` must be actual DPB slot indices
    /// (typically 0..max_dpb_slots-1). These values are stored in the decoder's `pic_idx`
    /// array via `update_frame_pointers` and returned by `compute_reference_name_slot_indices`.
    /// The slot index uniquely identifies a DPB entry in the Vulkan video session.
    ///
    /// # Arguments
    /// * `output_slot_index` - DPB slot index for the current output frame
    /// * `dpb_ref_slot_indices` - DPB slot indices for each reference picture (same order as `dpb_ref_pictures`)
    pub fn record_decode_command(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        session: vk::VideoSessionKHR,
        session_params: vk::VideoSessionParametersKHR,
        bitstream_buffer: vk::Buffer,
        bitstream_offset: u64,
        bitstream_range: u64,
        output_image_view: vk::ImageView,
        output_image: vk::Image,
        coded_extent: vk::Extent2D,
        dpb_setup_picture: Option<vk::VideoPictureResourceInfoKHR<'static>>,
        dpb_ref_pictures: &[vk::VideoPictureResourceInfoKHR<'static>],
        dpb_ref_slot_indices: &[i32],
        picture_info_container: &Vp9PictureInfoContainer,
        vp9_decode_info: &VideoDecodeVP9PictureInfoKHR,
        is_first_frame: bool,
        output_slot_index: i32,
    ) -> VideoResult<()> {
        let picture_info_ptr = picture_info_container.std_picture_info();

        // DEBUG: print all values being passed to Vulkan
        let std_info = &picture_info_container.std_picture_info;
        println!(
            "  [Vulkan] frame_type={}",
            if std_info.frame_type as u32 == 0 { "KEY" } else { "INTER" }
        );
        println!(
            "  [Vulkan] ref_name_slots=[{}, {}, {}]",
            vp9_decode_info.reference_name_slot_indices[0],
            vp9_decode_info.reference_name_slot_indices[1],
            vp9_decode_info.reference_name_slot_indices[2],
        );
        println!(
            "  [Vulkan] header_offsets: uncomp={} comp={} tiles={}",
            vp9_decode_info.uncompressed_header_offset,
            vp9_decode_info.compressed_header_offset,
            vp9_decode_info.tiles_offset,
        );
        println!(
            "  [Vulkan] bitstream: offset={} range={}",
            bitstream_offset, bitstream_range
        );
        println!(
            "  [Vulkan] dpb: setup_slot={} ref_count={}",
            output_slot_index, dpb_ref_pictures.len()
        );

        // Build reference slots for BeginVideoCoding (setup + references)
        // Per Vulkan spec: DPB slots become active when used in BeginVideoCodingKHR.
        // This happens DURING the call, so the first frame can activate its own slot.
        let mut all_slots: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::new();

        // Reference picture slots - use actual DPB slot indices
        let ref_slots: Vec<vk::VideoReferenceSlotInfoKHR> = dpb_ref_pictures
            .iter()
            .zip(dpb_ref_slot_indices.iter())
            .map(|(res, &slot_idx)| vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: std::ptr::null(),
                slot_index: slot_idx,
                p_picture_resource: &*res as *const _,
                _marker: Default::default(),
            })
            .collect();
        all_slots.extend(ref_slots.iter().cloned());

        // Setup slot (current frame output) - use actual DPB slot index
        let setup_slot = dpb_setup_picture.as_ref().map(|res| {
            vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: std::ptr::null(),
                slot_index: output_slot_index,
                p_picture_resource: res as *const _,
                _marker: Default::default(),
            }
        });
        if let Some(ref slot) = setup_slot {
            all_slots.push(slot.clone());
        }

        unsafe {
            // Begin command buffer
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .map_err(|e| {
                    VideoError::CommandBufferRecording(format!("Begin failed: {:?}", e))
                })?;

            // Begin video coding with reference slots
            let begin_coding_info = vk::VideoBeginCodingInfoKHR {
                s_type: vk::StructureType::VIDEO_BEGIN_CODING_INFO_KHR,
                p_next: std::ptr::null(),
                flags: vk::VideoBeginCodingFlagsKHR::empty(),
                video_session: session,
                video_session_parameters: session_params,
                reference_slot_count: all_slots.len() as u32,
                p_reference_slots: if all_slots.is_empty() {
                    std::ptr::null()
                } else {
                    all_slots.as_ptr()
                },
                _marker: Default::default(),
            };

            self.cmd_begin_video_coding(cmd_buffer, &begin_coding_info);

            // RESET decoder before first frame (required by Vulkan spec)
            // Must be INSIDE video coding block (after Begin, before Decode)
            if is_first_frame {
                self.cmd_control_video_coding(cmd_buffer);
            }

            // Barriers AFTER BeginVideoCoding and BEFORE DecodeVideo
            // This matches C++ reference VkVideoDecoder.cpp:1216-1227

            // Bitstream buffer barrier: HOST_WRITE -> VIDEO_DECODE_READ
            let buffer_barrier = vk::BufferMemoryBarrier2 {
                s_type: vk::StructureType::BUFFER_MEMORY_BARRIER_2,
                p_next: std::ptr::null(),
                src_stage_mask: vk::PipelineStageFlags2::HOST,
                src_access_mask: vk::AccessFlags2::HOST_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_READ_KHR,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                buffer: bitstream_buffer,
                offset: bitstream_offset,
                size: bitstream_range,
                _marker: Default::default(),
            };

            // Output image barrier
            // Use COLOR aspect for the full image (matches H.265 decoder pattern)
            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            let new_layout = if dpb_setup_picture.is_some() {
                // dpb_setup_picture points to the same image, so use DPB layout
                vk::ImageLayout::VIDEO_DECODE_DPB_KHR
            } else {
                vk::ImageLayout::VIDEO_DECODE_DST_KHR
            };

            let image_barrier = vk::ImageMemoryBarrier2 {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                p_next: std::ptr::null(),
                src_stage_mask: vk::PipelineStageFlags2::NONE,
                src_access_mask: vk::AccessFlags2::NONE,
                dst_stage_mask: vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                dst_access_mask: vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: output_image,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout,
                subresource_range,
                _marker: Default::default(),
            };

            let dep_info = vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                p_next: std::ptr::null(),
                dependency_flags: vk::DependencyFlags::BY_REGION,
                memory_barrier_count: 0,
                p_memory_barriers: std::ptr::null(),
                buffer_memory_barrier_count: 1,
                p_buffer_memory_barriers: &buffer_barrier,
                image_memory_barrier_count: 1,
                p_image_memory_barriers: &image_barrier,
                _marker: Default::default(),
            };
            self.cmd_pipeline_barrier_2(cmd_buffer, &dep_info);

            // Build VP9 decode info
            let dst_picture_resource = vk::VideoPictureResourceInfoKHR {
                s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: std::ptr::null(),
                coded_offset: vk::Offset2D::default(),
                coded_extent,
                base_array_layer: 0,
                image_view_binding: output_image_view,
                _marker: Default::default(),
            };

            let decode_info = vk::VideoDecodeInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_INFO_KHR,
                p_next: vp9_decode_info as *const _ as *const _,
                flags: vk::VideoDecodeFlagsKHR::empty(),
                src_buffer: bitstream_buffer,
                src_buffer_offset: bitstream_offset,
                src_buffer_range: bitstream_range,
                dst_picture_resource: dst_picture_resource,
                p_setup_reference_slot: setup_slot
                    .as_ref()
                    .map_or(std::ptr::null(), |s| s as *const _),
                reference_slot_count: ref_slots.len() as u32,
                p_reference_slots: if ref_slots.is_empty() {
                    std::ptr::null()
                } else {
                    ref_slots.as_ptr()
                },
                _marker: Default::default(),
            };

            // DEBUG: dump VideoDecodeVP9PictureInfoKHR and VkVideoDecodeInfoKHR for frame 0
            if self.frame_count == 0 {
                eprintln!("\n=== DEBUG: VideoDecodeVP9PictureInfoKHR ===");
                eprintln!("  size_of: expected=44 actual={}", std::mem::size_of::<VideoDecodeVP9PictureInfoKHR>());
                eprintln!("  offset_of s_type: {}", std::mem::offset_of!(VideoDecodeVP9PictureInfoKHR, s_type));
                eprintln!("  offset_of p_next: {}", std::mem::offset_of!(VideoDecodeVP9PictureInfoKHR, p_next));
                eprintln!("  offset_of p_std_picture_info: {}", std::mem::offset_of!(VideoDecodeVP9PictureInfoKHR, p_std_picture_info));
                eprintln!("  offset_of reference_name_slot_indices: {}", std::mem::offset_of!(VideoDecodeVP9PictureInfoKHR, reference_name_slot_indices));
                eprintln!("  offset_of uncompressed_header_offset: {}", std::mem::offset_of!(VideoDecodeVP9PictureInfoKHR, uncompressed_header_offset));
                eprintln!("  offset_of compressed_header_offset: {}", std::mem::offset_of!(VideoDecodeVP9PictureInfoKHR, compressed_header_offset));
                eprintln!("  offset_of tiles_offset: {}", std::mem::offset_of!(VideoDecodeVP9PictureInfoKHR, tiles_offset));
                let vp9_bytes = unsafe {
                    std::slice::from_raw_parts(
                        &vp9_decode_info as *const _ as *const u8,
                        std::mem::size_of::<VideoDecodeVP9PictureInfoKHR>(),
                    )
                };
                let hex: String = vp9_bytes.iter().map(|b| format!("{:02x} ", b)).collect();
                eprintln!("  raw bytes: {}", hex.trim());

                eprintln!("\n=== DEBUG: VkVideoDecodeInfoKHR ===");
                eprintln!("  size_of: {}", std::mem::size_of::<vk::VideoDecodeInfoKHR>());
                eprintln!("  offset_of s_type: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, s_type));
                eprintln!("  offset_of p_next: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, p_next));
                eprintln!("  offset_of flags: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, flags));
                eprintln!("  offset_of src_buffer: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, src_buffer));
                eprintln!("  offset_of src_buffer_offset: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, src_buffer_offset));
                eprintln!("  offset_of src_buffer_range: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, src_buffer_range));
                eprintln!("  offset_of dst_picture_resource: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, dst_picture_resource));
                eprintln!("  offset_of p_setup_reference_slot: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, p_setup_reference_slot));
                eprintln!("  offset_of reference_slot_count: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, reference_slot_count));
                eprintln!("  offset_of p_reference_slots: {}", std::mem::offset_of!(vk::VideoDecodeInfoKHR, p_reference_slots));
                let di_bytes = unsafe {
                    std::slice::from_raw_parts(
                        &decode_info as *const _ as *const u8,
                        std::mem::size_of::<vk::VideoDecodeInfoKHR>(),
                    )
                };
                let hex: String = di_bytes.iter().map(|b| format!("{:02x} ", b)).collect();
                eprintln!("  raw bytes: {}", hex.trim());
            }

            self.cmd_decode_video(cmd_buffer, &decode_info);

            println!("  [Vulkan] cmdDecodeVideoKHR submitted");

            // End video coding
            self.cmd_end_video_coding(cmd_buffer);

            // End command buffer
            self.device
                .end_command_buffer(cmd_buffer)
                .map_err(|e| {
                    VideoError::CommandBufferRecording(format!("End failed: {:?}", e))
                })?;
        }

        self.frame_count += 1;

        Ok(())
    }

    // Helper: dispatch cmdPipelineBarrier2
    fn cmd_pipeline_barrier_2(
        &self,
        cmd_buffer: vk::CommandBuffer,
        dep_info: &vk::DependencyInfo<'_>,
    ) {
        let fn_ptr = unsafe {
            self.instance
                .get_device_proc_addr(
                    self.device.handle(),
                    b"vkCmdPipelineBarrier2KHR\0".as_ptr().cast(),
                )
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType =
                    unsafe extern "system" fn(vk::CommandBuffer, *const vk::DependencyInfo<'_>);
                let f: FnType = std::mem::transmute(ptr);
                f(cmd_buffer, dep_info);
            }
        }
    }

    // Helper: dispatch cmdBeginVideoCodingKHR
    fn cmd_begin_video_coding(
        &self,
        cmd_buffer: vk::CommandBuffer,
        info: &vk::VideoBeginCodingInfoKHR<'_>,
    ) {
        let fn_ptr = unsafe {
            self.instance
                .get_device_proc_addr(
                    self.device.handle(),
                    b"vkCmdBeginVideoCodingKHR\0".as_ptr().cast(),
                )
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType =
                    unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoBeginCodingInfoKHR<'_>);
                let f: FnType = std::mem::transmute(ptr);
                f(cmd_buffer, info);
            }
        }
    }

    // Helper: dispatch cmdDecodeVideoKHR
    fn cmd_decode_video(
        &self,
        cmd_buffer: vk::CommandBuffer,
        info: &vk::VideoDecodeInfoKHR<'_>,
    ) {
        let fn_ptr = unsafe {
            self.instance
                .get_device_proc_addr(
                    self.device.handle(),
                    b"vkCmdDecodeVideoKHR\0".as_ptr().cast(),
                )
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType =
                    unsafe extern "system" fn(vk::CommandBuffer, *const vk::VideoDecodeInfoKHR<'_>);
                let f: FnType = std::mem::transmute(ptr);
                f(cmd_buffer, info);
            }
        }
    }

    // Helper: dispatch cmdControlVideoCodingKHR (for RESET)
    fn cmd_control_video_coding(&self, cmd_buffer: vk::CommandBuffer) {
        let coding_control_info = vk::VideoCodingControlInfoKHR {
            s_type: vk::StructureType::VIDEO_CODING_CONTROL_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoCodingControlFlagsKHR::RESET,
            _marker: Default::default(),
        };
        let fn_ptr = unsafe {
            self.instance
                .get_device_proc_addr(
                    self.device.handle(),
                    b"vkCmdControlVideoCodingKHR\0".as_ptr().cast(),
                )
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType = unsafe extern "system" fn(
                    vk::CommandBuffer,
                    *const vk::VideoCodingControlInfoKHR<'_>,
                );
                let f: FnType = std::mem::transmute(ptr);
                f(cmd_buffer, &coding_control_info);
            }
        }
    }

    // Helper: dispatch cmdEndVideoCodingKHR
    fn cmd_end_video_coding(&self, cmd_buffer: vk::CommandBuffer) {
        let end_coding_info = vk::VideoEndCodingInfoKHR {
            s_type: vk::StructureType::VIDEO_END_CODING_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoEndCodingFlagsKHR::empty(),
            _marker: Default::default(),
        };

        let fn_ptr = unsafe {
            self.instance
                .get_device_proc_addr(
                    self.device.handle(),
                    b"vkCmdEndVideoCodingKHR\0".as_ptr().cast(),
                )
        };
        if let Some(ptr) = fn_ptr {
            unsafe {
                type FnType = unsafe extern "system" fn(
                    vk::CommandBuffer,
                    *const vk::VideoEndCodingInfoKHR<'_>,
                );
                let f: FnType = std::mem::transmute(ptr);
                f(cmd_buffer, &end_coding_info);
            }
        }
    }
}

// ============================================================================
// VP9 Vulkan types (not in ash 0.38, defined manually)
// ============================================================================

/// VP9 Decode capabilities for Vulkan Video capability query.
///
/// Matches `VkVideoDecodeVP9CapabilitiesKHR` from Vulkan spec.
/// SType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_CAPABILITIES_KHR = 1000514001
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct VideoDecodeVP9CapabilitiesKHR {
    pub s_type: vk::StructureType,
    pub p_next: *mut std::os::raw::c_void,
    pub max_level: u32, // StdVideoVP9Level
}

/// VP9 Profile info for Vulkan Video session creation.
///
/// Matches `VkVideoDecodeVP9ProfileInfoKHR` from Vulkan spec.
/// SType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PROFILE_INFO_KHR = 1000514003
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct VideoDecodeVP9ProfileInfoKHR {
    pub s_type: vk::StructureType,
    pub p_next: *const std::os::raw::c_void,
    pub std_profile: u32, // StdVideoVP9Profile
    pub _marker: std::marker::PhantomData<()>,
}

/// VP9 Decode picture info for Vulkan Video.
///
/// Matches `VkVideoDecodeVP9PictureInfoKHR` from Vulkan spec exactly.
/// SType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PICTURE_INFO_KHR = 1000514002
#[repr(C)]
#[derive(Debug)]
pub struct VideoDecodeVP9PictureInfoKHR {
    pub s_type: vk::StructureType,
    pub p_next: *const std::os::raw::c_void,
    pub p_std_picture_info: *const StdVideoDecodeVP9PictureInfo,
    /// Maps VP9 reference frame names to DPB slot indices.
    /// Index 0 = LAST, Index 1 = GOLDEN, Index 2 = ALTREF
    /// Value of -1 means the reference is not used.
    pub reference_name_slot_indices: [i32; 3],
    pub uncompressed_header_offset: u32,
    pub compressed_header_offset: u32,
    pub tiles_offset: u32,
    _marker: [u8; 0],
}

impl VideoDecodeVP9PictureInfoKHR {
    /// Create a new VP9 picture info struct for Vulkan Video.
    pub fn new(
        p_std_picture_info: *const StdVideoDecodeVP9PictureInfo,
        reference_name_slot_indices: [i32; 3],
        uncompressed_header_offset: u32,
        compressed_header_offset: u32,
        tiles_offset: u32,
    ) -> Self {
        Self {
            s_type: vk::StructureType::from_raw(vp9_vk_constants::VIDEO_DECODE_VP9_PICTURE_INFO_KHR),
            p_next: std::ptr::null(),
            p_std_picture_info,
            reference_name_slot_indices,
            uncompressed_header_offset,
            compressed_header_offset,
            tiles_offset,
            _marker: [],
        }
    }
}

/// VP9 Session parameters create info (not in ash 0.38).
/// SType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_SESSION_PARAMETERS_CREATE_INFO_KHR = 1000514001
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct VideoDecodeVP9SessionParametersCreateInfoKHR {
    pub s_type: vk::StructureType,
    pub p_next: *const std::os::raw::c_void,
}

/// VP9-specific Vulkan constants not in ash 0.38.
pub mod vp9_vk_constants {
    /// VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_CAPABILITIES_KHR
    pub const VIDEO_DECODE_VP9_CAPABILITIES_KHR: i32 = 1000514001;
    /// VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_SESSION_PARAMETERS_CREATE_INFO_KHR
    pub const VIDEO_DECODE_VP9_SESSION_PARAMETERS_CREATE_INFO_KHR: i32 = 1000514000;
    /// VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PICTURE_INFO_KHR
    pub const VIDEO_DECODE_VP9_PICTURE_INFO_KHR: i32 = 1000514002;
    /// VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PROFILE_INFO_KHR
    pub const VIDEO_DECODE_VP9_PROFILE_INFO_KHR: i32 = 1000514003;
    /// VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR
    pub const DECODE_VP9: u32 = 8;
}

// ============================================================================
// VP9 StdVideo types (from vulkan_video_codec_vp9std.h and
// vulkan_video_codec_vp9std_decode.h)
// ============================================================================

/// VP9 Profile (from vulkan_video_codec_vp9std.h).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StdVideoVP9Profile {
    #[default]
    Profile0 = 0,
    Profile1 = 1,
    Profile2 = 2,
    Profile3 = 3,
}

/// VP9 Frame type (from vulkan_video_codec_vp9std.h).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StdVideoVP9FrameType {
    #[default]
    Key = 0,
    NonKey = 1,
}

/// VP9 Interpolation filter (from vulkan_video_codec_vp9std.h).
/// Values: EIGHTTAP=0, EIGHTTAP_SMOOTH=1, EIGHTTAP_SHARP=2, BILINEAR=3, SWITCHABLE=4
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StdVideoVP9InterpolationFilter {
    #[default]
    EightTap = 0,
    EightTapSmooth = 1,
    EightTapSharp = 2,
    Bilinear = 3,
    Switchable = 4,
}

/// VP9 Color space (from vulkan_video_codec_vp9std.h).
/// Values: UNKNOWN=0, BT_601=1, BT_709=2, SMPTE_170=3, SMPTE_240=4,
/// BT_2020=5, RESERVED=6, RGB=7
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StdVideoVP9ColorSpace {
    #[default]
    Unknown = 0,
    Bt601 = 1,
    Bt709 = 2,
    Smpte170 = 3,
    Smpte240 = 4,
    Bt2020 = 5,
    Reserved = 6,
    Rgb = 7,
}

/// VP9 Color config flags (from vulkan_video_codec_vp9std.h).
/// Bitfield: color_range:1, reserved:31 = 4 bytes total.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoVP9ColorConfigFlags {
    pub color_range: u32,
}

/// VP9 Color config (from vulkan_video_codec_vp9std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoVP9ColorConfig {
    pub flags: StdVideoVP9ColorConfigFlags,
    pub bit_depth: u8,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub reserved1: u8,
    pub color_space: StdVideoVP9ColorSpace,
}

/// VP9 Loop filter flags (from vulkan_video_codec_vp9std.h).
/// Bitfield: loop_filter_delta_enabled:1, loop_filter_delta_update:1, reserved:30 = 4 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoVP9LoopFilterFlags {
    pub loop_filter_delta_enabled: u32,
}

/// VP9 Loop filter (from vulkan_video_codec_vp9std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StdVideoVP9LoopFilter {
    pub flags: StdVideoVP9LoopFilterFlags,
    pub loop_filter_level: u8,
    pub loop_filter_sharpness: u8,
    pub update_ref_delta: u8,
    pub loop_filter_ref_deltas: [i8; 4], // STD_VIDEO_VP9_MAX_REF_FRAMES = 4
    pub update_mode_delta: u8,
    pub loop_filter_mode_deltas: [i8; 2], // STD_VIDEO_VP9_LOOP_FILTER_ADJUSTMENTS = 2
}

impl Default for StdVideoVP9LoopFilter {
    fn default() -> Self {
        Self {
            flags: StdVideoVP9LoopFilterFlags::default(),
            loop_filter_level: 0,
            loop_filter_sharpness: 0,
            update_ref_delta: 0,
            loop_filter_ref_deltas: [0; 4],
            update_mode_delta: 0,
            loop_filter_mode_deltas: [0; 2],
        }
    }
}

/// VP9 Segmentation flags (from vulkan_video_codec_vp9std.h).
/// Bitfield: 4 flags + 28 reserved = 4 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdVideoVP9SegmentationFlags {
    pub segmentation_update_map: u32,
}

/// VP9 Segmentation (from vulkan_video_codec_vp9std.h).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StdVideoVP9Segmentation {
    pub flags: StdVideoVP9SegmentationFlags,
    pub segmentation_tree_probs: [u8; 7],  // STD_VIDEO_VP9_MAX_SEGMENTATION_TREE_PROBS
    pub segmentation_pred_prob: [u8; 3],   // STD_VIDEO_VP9_MAX_SEGMENTATION_PRED_PROB
    pub feature_enabled: [u8; 8],          // STD_VIDEO_VP9_MAX_SEGMENTS
    pub feature_data: [[i16; 4]; 8],       // [STD_VIDEO_VP9_MAX_SEGMENTS][STD_VIDEO_VP9_SEG_LVL_MAX]
}

impl Default for StdVideoVP9Segmentation {
    fn default() -> Self {
        Self {
            flags: StdVideoVP9SegmentationFlags::default(),
            segmentation_tree_probs: [255; 7],
            segmentation_pred_prob: [255; 3],
            feature_enabled: [0; 8],
            feature_data: [[0; 4]; 8],
        }
    }
}

/// VP9 Decode picture info flags (from vulkan_video_codec_vp9std_decode.h).
///
/// Bitfield packed into a single u32:
///   error_resilient_mode:1, intra_only:1, allow_high_precision_mv:1,
///   refresh_frame_context:1, frame_parallel_decoding_mode:1,
///   segmentation_enabled:1, show_frame:1, UsePrevFrameMvs:1, reserved:24
///
/// Total size: 4 bytes (one u32).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StdVideoDecodeVP9PictureInfoFlags {
    /// Bit 0: error_resilient_mode
    /// Bit 1: intra_only
    /// Bit 2: allow_high_precision_mv
    /// Bit 3: refresh_frame_context
    /// Bit 4: frame_parallel_decoding_mode
    /// Bit 5: segmentation_enabled
    /// Bit 6: show_frame
    /// Bit 7: UsePrevFrameMvs
    /// Bits 8-31: reserved
    pub bits: u32,
}

impl StdVideoDecodeVP9PictureInfoFlags {
    pub fn new() -> Self {
        Self { bits: 0 }
    }

    pub fn set_error_resilient_mode(&mut self, val: u32) {
        self.bits = (self.bits & !1) | (val & 1);
    }
    pub fn error_resilient_mode(&self) -> u32 {
        self.bits & 1
    }

    pub fn set_intra_only(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 1)) | ((val & 1) << 1);
    }
    pub fn intra_only(&self) -> u32 {
        (self.bits >> 1) & 1
    }

    pub fn set_allow_high_precision_mv(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 2)) | ((val & 1) << 2);
    }
    pub fn allow_high_precision_mv(&self) -> u32 {
        (self.bits >> 2) & 1
    }

    pub fn set_refresh_frame_context(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 3)) | ((val & 1) << 3);
    }
    pub fn refresh_frame_context(&self) -> u32 {
        (self.bits >> 3) & 1
    }

    pub fn set_frame_parallel_decoding_mode(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 4)) | ((val & 1) << 4);
    }
    pub fn frame_parallel_decoding_mode(&self) -> u32 {
        (self.bits >> 4) & 1
    }

    pub fn set_segmentation_enabled(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 5)) | ((val & 1) << 5);
    }
    pub fn segmentation_enabled(&self) -> u32 {
        (self.bits >> 5) & 1
    }

    pub fn set_show_frame(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 6)) | ((val & 1) << 6);
    }
    pub fn show_frame(&self) -> u32 {
        (self.bits >> 6) & 1
    }

    pub fn set_use_prev_frame_mvs(&mut self, val: u32) {
        self.bits = (self.bits & !(1 << 7)) | ((val & 1) << 7);
    }
    pub fn use_prev_frame_mvs(&self) -> u32 {
        (self.bits >> 7) & 1
    }
}

impl Default for StdVideoDecodeVP9PictureInfoFlags {
    fn default() -> Self {
        Self::new()
    }
}

/// VP9 Decode picture info (from vulkan_video_codec_vp9std_decode.h).
///
/// Field order matches the Vulkan spec exactly:
///   flags (4B) + profile (4B) + frame_type (4B) + frame_context_idx (1B)
///   + reset_frame_context (1B) + refresh_frame_flags (1B)
///   + ref_frame_sign_bias_mask (1B) + interpolation_filter (4B)
///   + base_q_idx (1B) + delta_q_y_dc (1B) + delta_q_uv_dc (1B)
///   + delta_q_uv_ac (1B) + tile_cols_log2 (1B) + tile_rows_log2 (1B)
///   + reserved1[3] (6B) + pColorConfig (8B) + pLoopFilter (8B)
///   + pSegmentation (8B) = 56 bytes total
#[repr(C)]
#[derive(Debug, Clone)]
pub struct StdVideoDecodeVP9PictureInfo {
    pub flags: StdVideoDecodeVP9PictureInfoFlags,
    pub profile: StdVideoVP9Profile,
    pub frame_type: StdVideoVP9FrameType,
    pub frame_context_idx: u8,
    pub reset_frame_context: u8,
    pub refresh_frame_flags: u8,
    pub ref_frame_sign_bias_mask: u8,
    pub interpolation_filter: StdVideoVP9InterpolationFilter,
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_uv_dc: i8,
    pub delta_q_uv_ac: i8,
    pub tile_cols_log2: u8,
    pub tile_rows_log2: u8,
    pub reserved1: [u16; 3],
    pub p_color_config: *const StdVideoVP9ColorConfig,
    pub p_loop_filter: *const StdVideoVP9LoopFilter,
    pub p_segmentation: *const StdVideoVP9Segmentation,
}

impl Default for StdVideoDecodeVP9PictureInfo {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Container for VP9 picture info and its referenced sub-structures.
///
/// Holds the picture info, color config, loop filter, and segmentation
/// as a single stack-allocated value. This avoids memory leaks from Box::leak
/// and ensures all pointers remain valid during command buffer execution.
///
/// Usage: Create on the stack, pass `std_picture_info()` pointer to Vulkan,
/// keep alive until command execution completes.
#[repr(C)]
pub struct Vp9PictureInfoContainer {
    /// Must come first so &container as pointer to StdVideoDecodeVP9PictureInfo works
    pub std_picture_info: StdVideoDecodeVP9PictureInfo,
    pub color_config: StdVideoVP9ColorConfig,
    pub loop_filter: StdVideoVP9LoopFilter,
    pub segmentation: StdVideoVP9Segmentation,
}

/// Convert our Vp9PictureInfo to a Vp9PictureInfoContainer (stack-allocated).
///
/// Returns a container holding the Vulkan std struct and its referenced
/// sub-structures. All data is stack-allocated, no memory leaks.
pub fn convert_vp9_picture_info(
    info: &vk_video_core::picture::Vp9PictureInfo,
    color_config: &vk_video_core::picture::Vp9ColorConfig,
    loop_filter: &vk_video_core::picture::Vp9LoopFilter,
    segmentation: &vk_video_core::picture::Vp9Segmentation,
) -> Vp9PictureInfoContainer {
    let mut flags = StdVideoDecodeVP9PictureInfoFlags::new();
    flags.set_error_resilient_mode(info.flags.error_resilient_mode as u32);
    flags.set_intra_only(info.flags.intra_only as u32);
    flags.set_allow_high_precision_mv(info.flags.allow_high_precision_mv as u32);
    flags.set_refresh_frame_context(info.flags.refresh_frame_context as u32);
    flags.set_frame_parallel_decoding_mode(info.flags.frame_parallel_decoding_mode as u32);
    flags.set_segmentation_enabled(info.flags.segmentation_enabled as u32);
    flags.set_show_frame(info.flags.show_frame as u32);
    flags.set_use_prev_frame_mvs(info.flags.use_prev_frame_mvs as u32);

    // Convert color config (stack-allocated, no leak)
    let std_color_config = StdVideoVP9ColorConfig {
        flags: StdVideoVP9ColorConfigFlags {
            color_range: color_config.flags.color_range as u32,
        },
        bit_depth: color_config.bit_depth,
        subsampling_x: color_config.subsampling_x,
        subsampling_y: color_config.subsampling_y,
        reserved1: 0,
        color_space: match color_config.color_space {
            vk_video_core::picture::Vp9ColorSpace::Unknown => StdVideoVP9ColorSpace::Unknown,
            vk_video_core::picture::Vp9ColorSpace::Bt601 => StdVideoVP9ColorSpace::Bt601,
            vk_video_core::picture::Vp9ColorSpace::Bt709 => StdVideoVP9ColorSpace::Bt709,
            vk_video_core::picture::Vp9ColorSpace::Smpte170 => StdVideoVP9ColorSpace::Smpte170,
            vk_video_core::picture::Vp9ColorSpace::Smpte240 => StdVideoVP9ColorSpace::Smpte240,
            vk_video_core::picture::Vp9ColorSpace::Bt2020 => StdVideoVP9ColorSpace::Bt2020,
            vk_video_core::picture::Vp9ColorSpace::Reserved => StdVideoVP9ColorSpace::Reserved,
            vk_video_core::picture::Vp9ColorSpace::Rgb => StdVideoVP9ColorSpace::Rgb,
        },
    };

    // Convert loop filter (stack-allocated, no leak)
    let std_loop_filter = StdVideoVP9LoopFilter {
        flags: StdVideoVP9LoopFilterFlags {
            loop_filter_delta_enabled: loop_filter.flags.loop_filter_delta_enabled as u32
                | (loop_filter.flags.loop_filter_delta_update as u32) << 1,
        },
        loop_filter_level: loop_filter.loop_filter_level as u8,
        loop_filter_sharpness: loop_filter.loop_filter_sharpness,
        update_ref_delta: loop_filter.flags.update_ref_delta,
        loop_filter_ref_deltas: loop_filter.loop_filter_ref_deltas,
        update_mode_delta: loop_filter.flags.update_mode_delta,
        loop_filter_mode_deltas: loop_filter.loop_filter_mode_deltas,
    };

    // Convert segmentation (stack-allocated, no leak)
    let std_segmentation = StdVideoVP9Segmentation {
        flags: StdVideoVP9SegmentationFlags {
            segmentation_update_map: segmentation.flags.segmentation_update_map as u32
                | (segmentation.flags.segmentation_temporal_update as u32) << 1
                | (segmentation.flags.segmentation_update_data as u32) << 2
                | (segmentation.flags.segmentation_abs_or_delta_update as u32) << 3,
        },
        segmentation_tree_probs: segmentation.segmentation_tree_probs,
        segmentation_pred_prob: segmentation.segmentation_pred_prob,
        feature_enabled: segmentation.feature_enabled,
        feature_data: segmentation
            .feature_data
            .map(|row| row.map(|v| v as i16)),
    };

    Vp9PictureInfoContainer {
        std_picture_info: StdVideoDecodeVP9PictureInfo {
            flags,
            profile: match info.profile {
                vk_video_core::picture::Vp9Profile::Profile0 => StdVideoVP9Profile::Profile0,
                vk_video_core::picture::Vp9Profile::Profile1 => StdVideoVP9Profile::Profile1,
                vk_video_core::picture::Vp9Profile::Profile2 => StdVideoVP9Profile::Profile2,
                vk_video_core::picture::Vp9Profile::Profile3 => StdVideoVP9Profile::Profile3,
            },
            frame_type: match info.frame_type {
                vk_video_core::picture::Vp9FrameType::Key => StdVideoVP9FrameType::Key,
                vk_video_core::picture::Vp9FrameType::Inter => StdVideoVP9FrameType::NonKey,
            },
            frame_context_idx: info.frame_context_idx,
            reset_frame_context: info.flags.reset_frame_context,
            refresh_frame_flags: info.refresh_frame_flags,
            ref_frame_sign_bias_mask: info.ref_frame_sign_bias_mask,
            interpolation_filter: match info.interpolation_filter {
                vk_video_core::picture::Vp9InterpolationFilter::EightTapSmooth =>
                    StdVideoVP9InterpolationFilter::EightTapSmooth,
                vk_video_core::picture::Vp9InterpolationFilter::EightTap =>
                    StdVideoVP9InterpolationFilter::EightTap,
                vk_video_core::picture::Vp9InterpolationFilter::EightTapSharp =>
                    StdVideoVP9InterpolationFilter::EightTapSharp,
                vk_video_core::picture::Vp9InterpolationFilter::Bilinear =>
                    StdVideoVP9InterpolationFilter::Bilinear,
                vk_video_core::picture::Vp9InterpolationFilter::Switchable =>
                    StdVideoVP9InterpolationFilter::Switchable,
            },
            base_q_idx: info.base_q_idx as u8,
            delta_q_y_dc: info.delta_q_y_dc,
            delta_q_uv_dc: info.delta_q_uv_dc,
            delta_q_uv_ac: info.delta_q_uv_ac,
            tile_cols_log2: info.tile_cols_log2,
            tile_rows_log2: info.tile_rows_log2,
            reserved1: [0; 3],
            p_color_config: std::ptr::null(), // Set below
            p_loop_filter: std::ptr::null(),  // Set below
            p_segmentation: std::ptr::null(), // Set below
        },
        color_config: std_color_config,
        loop_filter: std_loop_filter,
        segmentation: std_segmentation,
    }
}

impl Vp9PictureInfoContainer {
    /// Initialize the pointer fields to point to the container's own sub-structures.
    /// Call this after creating the container, before passing to Vulkan.
    ///
    /// # Safety
    /// The container must remain alive for the duration of the Vulkan command execution.
    pub fn init_pointers(&mut self) {
        self.std_picture_info.p_color_config = &self.color_config as *const _;
        self.std_picture_info.p_loop_filter = &self.loop_filter as *const _;
        self.std_picture_info.p_segmentation = &self.segmentation as *const _;
    }

    /// Get a pointer to the StdVideoDecodeVP9PictureInfo within this container.
    pub fn std_picture_info(&self) -> *const StdVideoDecodeVP9PictureInfo {
        &self.std_picture_info
    }
}

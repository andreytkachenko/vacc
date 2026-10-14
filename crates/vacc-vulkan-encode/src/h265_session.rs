//! Vulkan Video H.265 encode session: all GPU resources and the per-frame
//! encode/readback pipeline. Mirrors `session.rs` (H.264).
//!
//! Design notes (verified against the Khronos reference implementation):
//! - Motion estimation is done by the driver; the app only supplies VPS/SPS/PPS,
//!   reference lists (by DPB slot index), per-picture short-term RPS and slice
//!   segment parameters.
//! - CQP = generic rate control DISABLED plus per-slice-segment `constantQp`.
//! - Unlike H.264 there is no `generatePrefixNalu` for H.265: the VPS/SPS/PPS
//!   are emitted by the application on the first frame. The preferred source
//!   is `vkGetEncodedVideoSessionParametersKHR` (the driver serializes our own
//!   std structs, guaranteeing consistency); `H265Params::prefix_nalus()` is
//!   the fallback if that call is unavailable.
//! - POC increases by 1 per frame (explicit `PicOrderCntVal`, no derivation
//!   constraints); the reference pattern is flat (previous N frames).
//! - Bitstream size is determined by zeroing the host-coherent bitstream
//!   buffer before each encode and scanning for the last non-zero byte after
//!   the encode (same as the H.264 session).

use ash::vk;
use ash::vk::native;
use vacc_vulkan::{VideoDeviceBuilder, VulkanDevice};

use super::Error;
use super::h265_params::H265Params;
use super::session::{
    bind_video_memories, build_std_header_version, create_full_image_view, create_host_buffer,
    create_video_image, VideoFns, NV12,
};

const STD_HEADER_NAME: &str = "VK_STD_vulkan_video_codec_h265_encode";

/// All GPU resources for one H.265 encode session.
pub struct EncodeSession {
    pub device: VulkanDevice,
    fns: VideoFns,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,

    session: vk::VideoSessionKHR,
    session_memories: Vec<vk::DeviceMemory>,
    parameters: vk::VideoSessionParametersKHR,
    param_memories: Vec<vk::DeviceMemory>,

    dpb_images: Vec<vk::Image>,
    dpb_views: Vec<vk::ImageView>,
    dpb_memories: Vec<vk::DeviceMemory>,

    input_image: vk::Image,
    input_view: vk::ImageView,
    input_memory: vk::DeviceMemory,

    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,

    bitstream_buffer: vk::Buffer,
    bitstream_memory: vk::DeviceMemory,
    /// Capacity of the bitstream buffer in bytes.
    pub bitstream_capacity: u32,

    /// Number of DPB slots (session maxDpbSlots).
    pub slots: u32,
    /// Clamped max active reference pictures.
    pub max_refs: u32,
    pub width: u32,
    pub height: u32,
    /// Coded picture extent. The driver aligns coded dimensions up to the CTB
    /// size (32x32 on NVIDIA H.265) and signals the crop back to the display
    /// size via a conformance window; every video image must cover this
    /// extent or inter prediction reads out-of-bounds rows (garbage P frames).
    pub coded_width: u32,
    pub coded_height: u32,
    /// IDR period (which frames are IRAP — used for reference pic types).
    intra_period: u32,
    /// SPS sample_adaptive_offset_enabled_flag (slice SAO flags must match).
    sao: bool,
    /// SPS sps_temporal_mvp_enabled_flag.
    temporal_mvp: bool,
    /// VPS/SPS/PPS Annex B prefix for the first frame (driver-serialized when
    /// available, else our own RBSP — see module docs).
    pub prefix_nalus: Vec<u8>,
}

impl EncodeSession {
    pub fn new(
        width: u32,
        height: u32,
        level_idc: native::StdVideoH265LevelIdc,
        cfg_max_refs: u32,
        intra_period: u32,
    ) -> Result<Self, Error> {
        let device = VideoDeviceBuilder::new()
            .with_video_codecs(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
            .build()
            .map_err(|e| Error::CodecNotSupported(format!("no Vulkan H.265 encode device: {e}")))?;

        let fns = VideoFns::load(&device.instance, device.device.handle())?;

        let encode_qf = device
            .queue_families
            .video_encode
            .ok_or_else(|| Error::CodecNotSupported("no video encode queue".into()))?;

        // 8-bit 4:2:0 Main profile only (NV12 input path).
        let profile_idc: u32 = native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN;
        let (caps, _enc_caps, h265_caps) = device
            .query_video_encode_capabilities_h265(
                profile_idc,
                vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
                vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
                vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            )
            .map_err(|e| Error::CodecNotSupported(format!("H.265 encode caps: {e}")))?;

        if h265_caps.max_slice_segment_count < 1 {
            return Err(Error::CodecNotSupported("maxSliceSegmentCount == 0".into()));
        }
        let slots = caps.max_dpb_slots.max(2);
        let max_refs = cfg_max_refs
            .min(caps.max_active_reference_pictures)
            .min(slots - 1)
            .max(1);

        // Coded extent: align display size up to the largest advertised CTB
        // size (the driver picks its CTB from this set and crops via a
        // conformance window — e.g. 320x240 -> 320x256 with bottom offset 8).
        let ctb = if h265_caps.ctb_sizes.contains(vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_64) {
            64
        } else if h265_caps.ctb_sizes.contains(vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_32) {
            32
        } else if h265_caps.ctb_sizes.contains(vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_16) {
            16
        } else {
            8
        };
        let coded_width = (width + ctb - 1) / ctb * ctb;
        let coded_height = (height + ctb - 1) / ctb * ctb;

        // VPS/SPS/PPS (SAO/temporal-MVP flags derived from the caps above).
        let params = H265Params::new(width, height, level_idc, max_refs, &h265_caps);

        // Profile chain (shared by session + image creation).
        let h265_profile = vk::VideoEncodeH265ProfileInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H265_PROFILE_INFO_KHR,
            p_next: std::ptr::null(),
            std_profile_idc: native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
            _marker: Default::default(),
        };
        let profile_info = vk::VideoProfileInfoKHR {
            s_type: vk::StructureType::VIDEO_PROFILE_INFO_KHR,
            p_next: &h265_profile as *const _ as *const _,
            video_codec_operation: vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
            chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            _marker: Default::default(),
        };
        let profile_list = vk::VideoProfileListInfoKHR {
            s_type: vk::StructureType::VIDEO_PROFILE_LIST_INFO_KHR,
            p_next: std::ptr::null(),
            profile_count: 1,
            p_profiles: &profile_info as *const _,
            _marker: Default::default(),
        };

        // ---- Session ----
        let std_header_version = build_std_header_version(STD_HEADER_NAME);
        let session_create_info = vk::VideoSessionCreateInfoKHR {
            s_type: vk::StructureType::VIDEO_SESSION_CREATE_INFO_KHR,
            p_next: std::ptr::null(),
            queue_family_index: encode_qf,
            flags: vk::VideoSessionCreateFlagsKHR::empty(),
            p_video_profile: &profile_info as *const _,
            picture_format: NV12,
            max_coded_extent: vk::Extent2D {
                width: coded_width,
                height: coded_height,
            },
            reference_picture_format: NV12,
            max_dpb_slots: slots,
            max_active_reference_pictures: max_refs,
            p_std_header_version: &std_header_version,
            _marker: Default::default(),
        };
        let mut session = vk::VideoSessionKHR::null();
        match unsafe {
            (fns.create_session)(device.device.handle(), &session_create_info, std::ptr::null(), &mut session)
        } {
            vk::Result::SUCCESS => {}
            r => return Err(Error::EncoderInit(format!("create_video_session: {r:?}"))),
        }

        let session_memories = bind_video_memories(
            &fns,
            &device.device,
            &device.memory_properties,
            fns.get_session_mem_reqs,
            session,
        )?;

        // ---- Session parameters (VPS/SPS/PPS) ----
        let params_add_info = vk::VideoEncodeH265SessionParametersAddInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR,
            p_next: std::ptr::null(),
            std_vps_count: 1,
            p_std_vp_ss: &params.vps,
            std_sps_count: 1,
            p_std_sp_ss: &params.sps,
            std_pps_count: 1,
            p_std_pp_ss: &params.pps,
            _marker: Default::default(),
        };
        // The driver locates VPS/SPS/PPS through an intermediate H265 params-create
        // struct in the pNext chain (NOT the AddInfo directly) — same as H.264.
        let h265_params_create = vk::VideoEncodeH265SessionParametersCreateInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR,
            p_next: std::ptr::null(),
            max_std_vps_count: 1,
            max_std_sps_count: 1,
            max_std_pps_count: 1,
            p_parameters_add_info: &params_add_info as *const _,
            _marker: Default::default(),
        };
        let params_create_info = vk::VideoSessionParametersCreateInfoKHR {
            s_type: vk::StructureType::VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
            p_next: &h265_params_create as *const _ as *const _,
            flags: vk::VideoSessionParametersCreateFlagsKHR::empty(),
            video_session: session,
            video_session_parameters_template: vk::VideoSessionParametersKHR::null(),
            _marker: Default::default(),
        };
        let mut parameters = vk::VideoSessionParametersKHR::null();
        match unsafe {
            (fns.create_params)(device.device.handle(), &params_create_info, std::ptr::null(), &mut parameters)
        } {
            vk::Result::SUCCESS => {}
            r => return Err(Error::EncoderInit(format!("create_video_session_parameters: {r:?}"))),
        }

        // Video session parameters use driver-managed memory (created above with
        // pAllocator=NULL); no separate alloc/bind (same as the H.264 session).
        let param_memories = Vec::new();

        // ---- VPS/SPS/PPS prefix for the first frame ----
        // Preferred: the driver serializes our std structs (guaranteed to match
        // what it expects). Fallback: our own RBSP writer.
        // Prefer the driver-serialized prefix (guaranteed consistent with the
        // std structs we passed); fall back to our own RBSP writer.
        let prefix_nalus = Self::fetch_driver_prefix(&fns, &device.device, parameters)
            .unwrap_or_else(|| params.prefix_nalus());

        // ---- DPB images ----
        let extent = vk::Extent3D {
            width: coded_width,
            height: coded_height,
            depth: 1,
        };
        let mut dpb_images = Vec::with_capacity(slots as usize);
        let mut dpb_views = Vec::with_capacity(slots as usize);
        let mut dpb_memories = Vec::with_capacity(slots as usize);
        for _ in 0..slots {
            let (img, mem) = create_video_image(
                &device.device,
                &device.memory_properties,
                extent,
                vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
                vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
                &profile_list,
            )?;
            let view = create_full_image_view(&device.device, img)?;
            dpb_images.push(img);
            dpb_views.push(view);
            dpb_memories.push(mem);
        }

        // ---- Input image + staging buffer ----
        let (input_image, input_memory) = create_video_image(
            &device.device,
            &device.memory_properties,
            extent,
            vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
            &profile_list,
        )?;
        let input_view = create_full_image_view(&device.device, input_image)?;

        let staging_size = (coded_width as u64) * (coded_height as u64) * 2;
        let (staging_buffer, staging_memory) = create_host_buffer(
            &device.device,
            &device.memory_properties,
            staging_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // ---- Bitstream buffer ----
        let bitstream_capacity = (width as u64)
            .max(1024)
            .saturating_mul(height as u64)
            .max(caps.min_bitstream_buffer_size_alignment) as u32;
        let (bitstream_buffer, bitstream_memory) = create_host_buffer(
            &device.device,
            &device.memory_properties,
            bitstream_capacity as u64,
            vk::BufferUsageFlags::VIDEO_ENCODE_DST_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // ---- Command pool / buffer / fence ----
        let command_pool = unsafe {
            device
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(encode_qf),
                    None,
                )
        }
        .map_err(|e| Error::EncoderInit(format!("create_command_pool: {e}")))?;
        let command_buffer = unsafe {
            device.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| Error::EncoderInit(format!("allocate_command_buffers: {e}")))?[0];
        let fence = unsafe { device.device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| Error::EncoderInit(format!("create_fence: {e}")))?;
        let queue = unsafe { device.device.get_device_queue(encode_qf, 0) };

        Ok(Self {
            device,
            fns,
            queue,
            command_pool,
            command_buffer,
            fence,
            session,
            session_memories,
            parameters,
            param_memories,
            dpb_images,
            dpb_views,
            dpb_memories,
            input_image,
            input_view,
            input_memory,
            staging_buffer,
            staging_memory,
            bitstream_buffer,
            bitstream_memory,
            bitstream_capacity,
            slots,
            max_refs,
            width,
            height,
            coded_width,
            coded_height,
            intra_period: intra_period.max(1),
            sao: params.sao_enabled,
            temporal_mvp: params.temporal_mvp_enabled,
            prefix_nalus,
        })
    }

    /// Ask the driver to serialize our session parameters (VPS/SPS/PPS) via
    /// vkGetEncodedVideoSessionParametersKHR. Returns None if unavailable.
    fn fetch_driver_prefix(
        fns: &VideoFns,
        device: &ash::Device,
        parameters: vk::VideoSessionParametersKHR,
    ) -> Option<Vec<u8>> {
        let h265_get_info = vk::VideoEncodeH265SessionParametersGetInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H265_SESSION_PARAMETERS_GET_INFO_KHR,
            p_next: std::ptr::null(),
            write_std_vps: 1,
            write_std_sps: 1,
            write_std_pps: 1,
            std_vps_id: 0,
            std_sps_id: 0,
            std_pps_id: 0,
            _marker: Default::default(),
        };
        let get_info = vk::VideoEncodeSessionParametersGetInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_SESSION_PARAMETERS_GET_INFO_KHR,
            p_next: &h265_get_info as *const _ as *const _,
            video_session_parameters: parameters,
            _marker: Default::default(),
        };

        let mut size: usize = 0;
        match unsafe {
            (fns.get_encoded_session_params)(
                device.handle(),
                &get_info,
                std::ptr::null_mut(),
                &mut size,
                std::ptr::null_mut(),
            )
        } {
            vk::Result::SUCCESS if size > 0 => {}
            _ => return None,
        }
        let mut buf = vec![0u8; size];
        match unsafe {
            (fns.get_encoded_session_params)(
                device.handle(),
                &get_info,
                std::ptr::null_mut(),
                &mut size,
                buf.as_mut_ptr().cast(),
            )
        } {
            vk::Result::SUCCESS => Some(buf),
            _ => None,
        }
    }

    /// Encode one frame synchronously; returns the raw Annex B VCL bytes for
    /// the picture. `reset_session` must be true for the first frame (clears
    /// stale session state, per the reference impl). The caller prepends
    /// [`Self::prefix_nalus`] to the first frame's output.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_one_frame(
        &self,
        frame: &vacc_core::encoder::InputFrame,
        index: u32,
        is_idr: bool,
        qp: u8,
        poc: i32,
        reset_session: bool,
    ) -> Result<Vec<u8>, Error> {
        let width = self.width;
        let height = self.height;
        let cw = self.coded_width;
        let ch = self.coded_height;

        // ---- Upload NV12 to the input image via a host-visible staging buffer.
        // The coded picture is CTB-aligned (see new()); zero-pad the extra
        // rows/columns — the driver crops them out via the conformance window.
        let plane_bytes = (cw as u64) * (ch as u64) * 2;
        let mapped = unsafe { self.device.device.map_memory(self.staging_memory, 0, plane_bytes, vk::MemoryMapFlags::empty()) }
            .map_err(|e| Error::Vulkan(format!("map staging: {e}")))?;
        {
            let dst = unsafe { std::slice::from_raw_parts_mut(mapped as *mut u8, (cw as usize) * (ch as usize) * 2) };
            dst.fill(0);
            let y_len = (cw as usize) * (ch as usize);
            // Luma rows (content width bytes, coded-width stride).
            for row in 0..height as usize {
                dst[row * cw as usize..row * cw as usize + width as usize]
                    .copy_from_slice(&frame.y[row * frame.y_pitch..row * frame.y_pitch + width as usize]);
            }
            // Interleaved chroma rows (width bytes each — W/2 UV pairs —
            // height/2 rows, coded-width stride).
            for row in 0..(height / 2) as usize {
                let s = row * frame.uv_pitch;
                let d = y_len + row * cw as usize;
                dst[d..d + width as usize].copy_from_slice(&frame.uv[s..s + width as usize]);
            }
        }
        unsafe { self.device.device.unmap_memory(self.staging_memory) };

        // ---- Per-frame std structures (must live through vkCmdEncodeVideoKHR).
        let nref = if is_idr { 0 } else { self.max_refs.min(index).max(1) };
        let setup_slot = (index % self.slots) as usize;

        // Reference list L0: most recent `nref` frames (DPB slot indices).
        let mut refpic_list0 = [0xFFu8; 15];
        for (k, slot) in refpic_list0.iter_mut().take(nref as usize).enumerate() {
            *slot = ((index - 1 - k as u32) % self.slots) as u8;
        }

        let ref_lists: Option<Box<native::StdVideoEncodeH265ReferenceListsInfo>> = if is_idr {
            None
        } else {
            Some(Box::new(native::StdVideoEncodeH265ReferenceListsInfo {
                flags: unsafe { std::mem::zeroed() },
                num_ref_idx_l0_active_minus1: (nref - 1) as u8,
                num_ref_idx_l1_active_minus1: 0,
                RefPicList0: refpic_list0,
                RefPicList1: [0xFFu8; 15],
                list_entry_l0: [0u8; 15],
                list_entry_l1: [0u8; 15],
            }))
        };

        // Per-picture inline RPS: the `nref` previous frames (POC-1 .. POC-nref).
        // (The SPS still declares one RPS set — required by the driver's inter
        // path — but slices carry their own RPS, like the reference encoder.)
        let st_rps = H265Params::short_term_ref_pic_set(nref);

        let mut pic_flags = unsafe { std::mem::zeroed::<native::StdVideoEncodeH265PictureInfoFlags>() };
        pic_flags.set_is_reference(1);
        pic_flags.set_IrapPicFlag(is_idr as u32);
        pic_flags.set_pic_output_flag(1);
        pic_flags.set_short_term_ref_pic_set_sps_flag(0); // inline RPS via pShortTermRefPicSet
        // Must be 0 for I slices (H.265 7.4.7.1) and when the SPS disables it.
        pic_flags.set_slice_temporal_mvp_enabled_flag(
            (self.temporal_mvp && !is_idr) as u32,
        );
        let std_picture_info = native::StdVideoEncodeH265PictureInfo {
            flags: pic_flags,
            pic_type: if is_idr {
                native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_IDR
            } else {
                native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P
            },
            sps_video_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            pps_pic_parameter_set_id: 0,
            short_term_ref_pic_set_idx: 0,
            PicOrderCntVal: poc,
            TemporalId: 0,
            reserved1: [0u8; 7],
            pRefLists: ref_lists.as_ref().map_or(std::ptr::null(), |rl| &**rl as *const _),
            pShortTermRefPicSet: &st_rps,
            pLongTermRefPics: std::ptr::null(),
        };

        let mut slice_flags = unsafe { std::mem::zeroed::<native::StdVideoEncodeH265SliceSegmentHeaderFlags>() };
        slice_flags.set_first_slice_segment_in_pic_flag(1);
        slice_flags.set_dependent_slice_segment_flag(0);
        // SAO flags must match the SPS sample_adaptive_offset_enabled_flag.
        slice_flags.set_slice_sao_luma_flag(self.sao as u32);
        slice_flags.set_slice_sao_chroma_flag(self.sao as u32);
        // Present only for P slices; 1 makes the driver emit the L0 active count
        // from the reference list info.
        slice_flags.set_num_ref_idx_active_override_flag(if is_idr { 0 } else { 1 });
        slice_flags.set_mvd_l1_zero_flag(0);
        slice_flags.set_cabac_init_flag(0);
        slice_flags.set_cu_chroma_qp_offset_enabled_flag(0);
        slice_flags.set_deblocking_filter_override_flag(0);
        slice_flags.set_slice_deblocking_filter_disabled_flag(0);
        slice_flags.set_collocated_from_l0_flag(1);
        slice_flags.set_slice_loop_filter_across_slices_enabled_flag(0);
        let slice_header = native::StdVideoEncodeH265SliceSegmentHeader {
            flags: slice_flags,
            slice_type: if is_idr {
                native::StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_I
            } else {
                native::StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_P
            },
            slice_segment_address: 0,
            collocated_ref_idx: 0,
            // MaxNumMergeCand = 5 is the H.265 default; anything else is written
            // as an explicit syntax element and some drivers reject it.
            MaxNumMergeCand: 5,
            slice_cb_qp_offset: 0,
            slice_cr_qp_offset: 0,
            slice_beta_offset_div2: 0,
            slice_tc_offset_div2: 0,
            slice_act_y_qp_offset: 0,
            slice_act_cb_qp_offset: 0,
            slice_act_cr_qp_offset: 0,
            slice_qp_delta: 0, // CQP via constantQp
            reserved1: 0,
            pWeightTable: std::ptr::null(),
        };

        let nalu_slice_info = vk::VideoEncodeH265NaluSliceSegmentInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H265_NALU_SLICE_SEGMENT_INFO_KHR,
            p_next: std::ptr::null(),
            constant_qp: qp as i32,
            p_std_slice_segment_header: &slice_header,
            _marker: Default::default(),
        };

        let picture_info = vk::VideoEncodeH265PictureInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H265_PICTURE_INFO_KHR,
            p_next: std::ptr::null(),
            nalu_slice_segment_entry_count: 1,
            p_nalu_slice_segment_entries: &nalu_slice_info,
            p_std_picture_info: &std_picture_info,
            _marker: Default::default(),
        };

        // ---- Picture resources + reference slots.
        // Layout: slot 0 = setup (current picture in its DPB image), 1..=nref = active refs.
        let mut res_infos: Vec<vk::VideoPictureResourceInfoKHR> = Vec::with_capacity(nref as usize + 1);
        res_infos.push(vk::VideoPictureResourceInfoKHR {
            s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
            p_next: std::ptr::null(),
            coded_offset: vk::Offset2D { x: 0, y: 0 },
            // Coded extent is the CTB-aligned picture size (see new()); the
            // image carries zero padding beyond the display size.
            coded_extent: vk::Extent2D { width: cw, height: ch },
            base_array_layer: 0,
            image_view_binding: self.dpb_views[setup_slot],
            _marker: Default::default(),
        });
        for k in 0..nref {
            res_infos.push(vk::VideoPictureResourceInfoKHR {
                s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: std::ptr::null(),
                coded_offset: vk::Offset2D { x: 0, y: 0 },
                // Coded extent is the CTB-aligned picture size (see new()); the
            // image carries zero padding beyond the display size.
            coded_extent: vk::Extent2D { width: cw, height: ch },
                base_array_layer: 0,
                image_view_binding: self.dpb_views[((index - 1 - k) % self.slots) as usize],
                _marker: Default::default(),
            });
        }

        // Standard reference info for each active reference (pic type/poc).
        // The StdVideo structs MUST outlive vkCmdEncodeVideoKHR — keep them in a
        // stable Vec (pointing at loop-locals would dangle).
        let mut ref_infos_vec: Vec<native::StdVideoEncodeH265ReferenceInfo> = Vec::with_capacity(nref as usize);
        for k in 0..nref {
            let ref_index = index - 1 - k;
            let ref_is_idr = ref_index.is_multiple_of(self.intra_period);
            ref_infos_vec.push(native::StdVideoEncodeH265ReferenceInfo {
                flags: unsafe { std::mem::zeroed() },
                pic_type: if ref_is_idr {
                    native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_IDR
                } else {
                    native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P
                },
                PicOrderCntVal: ref_index as i32, // POC = frame index (increments by 1)
                TemporalId: 0,
            });
        }
        let mut dpb_slot_infos: Vec<vk::VideoEncodeH265DpbSlotInfoKHR> = Vec::with_capacity(nref as usize);
        for k in 0..nref {
            dpb_slot_infos.push(vk::VideoEncodeH265DpbSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_ENCODE_H265_DPB_SLOT_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_reference_info: &ref_infos_vec[k as usize],
                _marker: Default::default(),
            });
        }

        let mut slot_infos: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::with_capacity(nref as usize + 1);
        // Setup slot (current picture's write target), listed in BeginCoding with its
        // valid DPB index (the driver requires a valid index here — see H.264 notes).
        slot_infos.push(vk::VideoReferenceSlotInfoKHR {
            s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
            p_next: std::ptr::null(),
            slot_index: setup_slot as i32,
            p_picture_resource: &res_infos[0],
            _marker: Default::default(),
        });
        for k in 0..nref {
            slot_infos.push(vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: &dpb_slot_infos[k as usize] as *const _ as *const _,
                slot_index: ((index - 1 - k) % self.slots) as i32,
                p_picture_resource: &res_infos[k as usize + 1],
                _marker: Default::default(),
            });
        }

        let encode_info = vk::VideoEncodeInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_INFO_KHR,
            p_next: &picture_info as *const _ as *const _,
            flags: vk::VideoEncodeFlagsKHR::empty(),
            dst_buffer: self.bitstream_buffer,
            dst_buffer_offset: 0,
            dst_buffer_range: self.bitstream_capacity as u64,
            src_picture_resource: vk::VideoPictureResourceInfoKHR {
                s_type: vk::StructureType::VIDEO_PICTURE_RESOURCE_INFO_KHR,
                p_next: std::ptr::null(),
                coded_offset: vk::Offset2D { x: 0, y: 0 },
                // Coded extent is the CTB-aligned picture size (see new()); the
            // image carries zero padding beyond the display size.
            coded_extent: vk::Extent2D { width: cw, height: ch },
                base_array_layer: 0,
                image_view_binding: self.input_view,
                _marker: Default::default(),
            },
            // pReferenceSlots holds ONLY the references (setup is separate via
            // pSetupReferenceSlot) — same as the H.264 session.
            p_setup_reference_slot: &slot_infos[0],
            reference_slot_count: nref,
            p_reference_slots: if nref > 0 { unsafe { slot_infos.as_ptr().add(1) } } else { std::ptr::null() },
            preceding_externally_encoded_bytes: 0,
            _marker: Default::default(),
        };

        // CQP: rate control disabled + per-slice-segment constant_qp.
        let rate_control_info = vk::VideoEncodeRateControlInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_RATE_CONTROL_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoEncodeRateControlFlagsKHR::empty(),
            rate_control_mode: vk::VideoEncodeRateControlModeFlagsKHR::DISABLED,
            layer_count: 0,
            p_layers: std::ptr::null(),
            virtual_buffer_size_in_ms: 0,
            initial_virtual_buffer_size_in_ms: 0,
            _marker: Default::default(),
        };
        let begin_info = vk::VideoBeginCodingInfoKHR {
            s_type: vk::StructureType::VIDEO_BEGIN_CODING_INFO_KHR,
            p_next: &rate_control_info as *const _ as *const _,
            flags: vk::VideoBeginCodingFlagsKHR::empty(),
            video_session: self.session,
            video_session_parameters: self.parameters,
            reference_slot_count: slot_infos.len() as u32,
            p_reference_slots: slot_infos.as_ptr(),
            _marker: Default::default(),
        };
        let end_info = vk::VideoEndCodingInfoKHR {
            s_type: vk::StructureType::VIDEO_END_CODING_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::VideoEndCodingFlagsKHR::empty(),
            _marker: Default::default(),
        };
        // Dedicated session-init rate control (matches the H.264 session): a
        // RESET control carrying per-layer minQp/maxQp so the driver knows the
        // QP operating point before any picture is encoded.
        let h265_qp = vk::VideoEncodeH265QpKHR {
            qp_i: qp as i32,
            qp_p: qp as i32,
            qp_b: qp as i32,
        };
        let h265_frame_size = vk::VideoEncodeH265FrameSizeKHR {
            frame_i_size: 0,
            frame_p_size: 0,
            frame_b_size: 0,
        };
        let h265_rc_layer = vk::VideoEncodeH265RateControlLayerInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H265_RATE_CONTROL_LAYER_INFO_KHR,
            p_next: std::ptr::null(),
            use_min_qp: 1,
            min_qp: h265_qp,
            use_max_qp: 1,
            max_qp: h265_qp,
            use_max_frame_size: 1,
            max_frame_size: h265_frame_size,
            _marker: Default::default(),
        };
        let rc_layer = vk::VideoEncodeRateControlLayerInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_RATE_CONTROL_LAYER_INFO_KHR,
            p_next: &h265_rc_layer as *const _ as *const _,
            average_bitrate: 0,
            max_bitrate: 0,
            frame_rate_numerator: 0,
            frame_rate_denominator: 0,
            _marker: Default::default(),
        };
        let control_info = vk::VideoCodingControlInfoKHR {
            s_type: vk::StructureType::VIDEO_CODING_CONTROL_INFO_KHR,
            p_next: &rc_layer as *const _ as *const _,
            flags: vk::VideoCodingControlFlagsKHR::RESET,
            _marker: Default::default(),
        };

        // ---- Dedicated session init (RESET + QP) before the first encode.
        if reset_session {
            let dev = &self.device.device;
            let cmd = self.command_buffer;
            unsafe { dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()) }
                .map_err(|e| Error::Vulkan(format!("init reset_command_buffer: {e}")))?;
            unsafe { dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()) }
                .map_err(|e| Error::Vulkan(format!("init begin_command_buffer: {e}")))?;
            let init_begin = vk::VideoBeginCodingInfoKHR {
                s_type: vk::StructureType::VIDEO_BEGIN_CODING_INFO_KHR,
                p_next: std::ptr::null(),
                flags: vk::VideoBeginCodingFlagsKHR::empty(),
                video_session: self.session,
                video_session_parameters: self.parameters,
                reference_slot_count: 0,
                p_reference_slots: std::ptr::null(),
                _marker: Default::default(),
            };
            unsafe { (self.fns.cmd_begin_coding)(cmd, &init_begin) };
            unsafe { (self.fns.cmd_control_coding)(cmd, &control_info) };
            unsafe { (self.fns.cmd_end_coding)(cmd, &end_info) };
            unsafe { dev.end_command_buffer(cmd) }
                .map_err(|e| Error::Vulkan(format!("init end_command_buffer: {e}")))?;
            unsafe { dev.reset_fences(&[self.fence]) }
                .map_err(|e| Error::Vulkan(format!("init reset_fences: {e}")))?;
            unsafe {
                dev.queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                    self.fence,
                )
            }
            .map_err(|e| Error::Vulkan(format!("init queue_submit: {e}")))?;
            unsafe { dev.wait_for_fences(&[self.fence], true, 30_000_000_000) }
                .map_err(|e| Error::Vulkan(format!("init wait_for_fences: {e}")))?;
        }

        // Zero the bitstream buffer before encoding (host-coherent memory), then
        // scan for the last non-zero byte after the encode to get the size.
        let dev = &self.device.device;
        {
            let m = unsafe { dev.map_memory(self.bitstream_memory, 0, self.bitstream_capacity as u64, vk::MemoryMapFlags::empty()) }
                .map_err(|e| Error::Vulkan(format!("map bitstream (zero): {e}")))?;
            unsafe { std::ptr::write_bytes(m as *mut u8, 0, self.bitstream_capacity as usize) };
            unsafe { dev.unmap_memory(self.bitstream_memory) };
        }

        // ---- Record + submit.
        let cmd = self.command_buffer;
        unsafe { dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()) }
            .map_err(|e| Error::Vulkan(format!("reset_command_buffer: {e}")))?;
        unsafe { dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()) }
            .map_err(|e| Error::Vulkan(format!("begin_command_buffer: {e}")))?;

        // [outside coding region] staging buffer was host-written.
        let buf_barrier = vk::BufferMemoryBarrier2 {
            s_type: vk::StructureType::BUFFER_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::NONE,
            src_access_mask: vk::AccessFlags2::HOST_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            dst_access_mask: vk::AccessFlags2::TRANSFER_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer: self.staging_buffer,
            offset: 0,
            size: plane_bytes,
            _marker: Default::default(),
        };
        let dep = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::empty(),
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 1,
            p_buffer_memory_barriers: &buf_barrier,
            image_memory_barrier_count: 0,
            p_image_memory_barriers: std::ptr::null(),
            _marker: Default::default(),
        };
        unsafe { dev.cmd_pipeline_barrier2(cmd, &dep) };

        // NV12 is a multi-plane format: each plane needs its own copy region.
        // Copy the full coded (CTB-aligned) extent; padding was zeroed above.
        let y_size = (cw as u64) * (ch as u64);
        let copy_regions = [
            vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0, // tightly packed
                buffer_image_height: ch,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_0,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D { width: cw, height: ch, depth: 1 },
            },
            vk::BufferImageCopy {
                buffer_offset: y_size,
                buffer_row_length: 0, // tightly packed
                buffer_image_height: ch / 2,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_1,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D { width: cw / 2, height: ch / 2, depth: 1 },
            },
        ];
        unsafe {
            dev.cmd_copy_buffer_to_image(
                cmd,
                self.staging_buffer,
                self.input_image,
                vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
                &copy_regions,
            )
        };

        // [outside coding region] input image now readable by video encode.
        let img_barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            dst_access_mask: vk::AccessFlags2::VIDEO_ENCODE_READ_KHR,
            old_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
            new_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: self.input_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            _marker: Default::default(),
        };
        let dep2 = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::empty(),
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: std::ptr::null(),
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &img_barrier,
            _marker: Default::default(),
        };
        unsafe { dev.cmd_pipeline_barrier2(cmd, &dep2) };

        // [coding region]
        unsafe { (self.fns.cmd_begin_coding)(cmd, &begin_info) };
        unsafe { (self.fns.cmd_encode_video)(cmd, &encode_info) };

        // The setup slot now holds the reconstructed current picture: make it
        // readable for future reference (per the reference impl).
        let dpb_barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            src_access_mask: vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR,
            dst_stage_mask: vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            dst_access_mask: vk::AccessFlags2::VIDEO_ENCODE_READ_KHR,
            old_layout: vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
            new_layout: vk::ImageLayout::VIDEO_ENCODE_DPB_KHR,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: self.dpb_images[setup_slot],
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            _marker: Default::default(),
        };
        let dep3 = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::empty(),
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: std::ptr::null(),
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &dpb_barrier,
            _marker: Default::default(),
        };
        unsafe { dev.cmd_pipeline_barrier2(cmd, &dep3) };

        unsafe { (self.fns.cmd_end_coding)(cmd, &end_info) };
        unsafe { dev.end_command_buffer(cmd) }
            .map_err(|e| Error::Vulkan(format!("end_command_buffer: {e}")))?;

        unsafe { dev.reset_fences(&[self.fence]) }
            .map_err(|e| Error::Vulkan(format!("reset_fences: {e}")))?;
        unsafe {
            dev.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                self.fence,
            )
        }
        .map_err(|e| Error::Vulkan(format!("queue_submit: {e}")))?;
        unsafe { dev.wait_for_fences(&[self.fence], true, 30_000_000_000) }
            .map_err(|e| Error::Vulkan(format!("wait_for_fences: {e}")))?;

        // ---- Determine the coded size by scanning for the last non-zero byte.
        let mapped = unsafe { dev.map_memory(self.bitstream_memory, 0, self.bitstream_capacity as u64, vk::MemoryMapFlags::empty()) }
            .map_err(|e| Error::Vulkan(format!("map bitstream: {e}")))?;
        let buf = unsafe { std::slice::from_raw_parts(mapped as *const u8, self.bitstream_capacity as usize) };
        let size = buf.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
        let data = buf[..size].to_vec();
        unsafe { dev.unmap_memory(self.bitstream_memory) };

        Ok(data)
    }
}

impl Drop for EncodeSession {
    fn drop(&mut self) {
        let d = &self.device.device;
        unsafe {
            let _ = d.device_wait_idle();
            d.destroy_buffer(self.bitstream_buffer, None);
            d.free_memory(self.bitstream_memory, None);
            d.destroy_buffer(self.staging_buffer, None);
            d.free_memory(self.staging_memory, None);
            d.destroy_image_view(self.input_view, None);
            d.destroy_image(self.input_image, None);
            d.free_memory(self.input_memory, None);
            for i in 0..self.dpb_images.len() {
                d.destroy_image_view(self.dpb_views[i], None);
                d.destroy_image(self.dpb_images[i], None);
                d.free_memory(self.dpb_memories[i], None);
            }
            (self.fns.destroy_params)(d.handle(), self.parameters, std::ptr::null());
            for mem in &self.param_memories {
                d.free_memory(*mem, None);
            }
            (self.fns.destroy_session)(d.handle(), self.session, std::ptr::null());
            for mem in &self.session_memories {
                d.free_memory(*mem, None);
            }
            d.destroy_fence(self.fence, None);
            d.free_command_buffers(self.command_pool, &[self.command_buffer]);
            d.destroy_command_pool(self.command_pool, None);
        };
    }
}


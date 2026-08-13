//! NVDEC H.264 decoder implementation using NVIDIA's parser.
//!
//! Uses NVIDIA's cuvidParseVideoData parser which properly handles SPS/PPS,
//! DPB management, and CUVIDPICPARAMS population.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::os::raw::{c_int, c_void};

use vk_video_core::{
    codec::VideoCodec,
    decoder::{Decoder, DecoderInfo},
    frame::{DecodedFrame, FieldFlags, PixelData, PixelPlane},
    format::{ChromaSubsampling, ComponentBitDepth, VideoFormat},
    session::Extent2D,
};

use crate::device::{cu_ctx_set_current, cu_ctx_synchronize, cu_mem_free_host, cu_mem_host_alloc, cu_memcpy_dtoh, get_funcs, init_nvdec};
use crate::error::{NvdecError, NvdecResult};
use crate::ffi::{
    cudaVideoChromaFormat, cudaVideoCodec, cudaVideoCreateFlags, cudaVideoDeinterlaceMode,
    cudaVideoSurfaceFormat, CUVIDDECODECREATEINFO, CUVIDPARSERDISPINFO,
    CUVIDPARSERPARAMS, CUVIDPICPARAMS, CUVIDPROCPARAMS, CUVIDSOURCEDATAPACKET, CUvideodecoder,
    CUvideoparser, CUDA_SUCCESS, CUdeviceptr, CUVIDEOFORMAT,
};



/// Maximum number of decode surfaces.
const MAX_DECODE_SURFACES: usize = 32;

/// Decoder state passed to parser callbacks via pUserData.
struct DecoderState {
    /// NVDEC decoder handle (set in sequence callback).
    decoder: Mutex<CUvideodecoder>,

    /// Decoder info.
    info: Mutex<DecoderInfo>,

    /// Pending decoded frames queue.
    /// Frames are copied immediately in display_callback before DPB slots are reused.
    pending_frames: Mutex<VecDeque<DecodedFrame>>,

    /// Frame count for ordering.
    frame_count: Mutex<u32>,

    /// Display area.
    display_area: Mutex<(i32, i32, i32, i32)>,

    /// Whether decoder is initialized.
    initialized: Mutex<bool>,

    /// Last decode error.
    last_error: Mutex<Option<String>>,
}

impl DecoderState {
    fn new() -> Self {
        Self {
            decoder: Mutex::new(std::ptr::null_mut()),
            info: Mutex::new(DecoderInfo {
                backend: "nvdec".to_string(),
                codec: VideoCodec::DecodeH264,
                coded_size: Extent2D { width: 0, height: 0 },
                display_size: Extent2D { width: 0, height: 0 },
                chroma_subsampling: ChromaSubsampling::_420,
                luma_bit_depth: ComponentBitDepth::Bit8,
                chroma_bit_depth: ComponentBitDepth::Bit8,
                profile_idc: None,
                dpb_slots: 0,
            }),
            pending_frames: Mutex::new(VecDeque::new()),
            frame_count: Mutex::new(0),
            display_area: Mutex::new((0, 0, 0, 0)),
            initialized: Mutex::new(false),
            last_error: Mutex::new(None),
        }
    }
}

/// Sequence callback: called by parser when SPS/PPS is found.
/// Creates the NVDEC decoder here.
unsafe extern "C" fn sequence_callback(pUserData: *mut c_void, pVideoFormat: *mut CUVIDEOFORMAT) -> c_int {
    if pUserData.is_null() || pVideoFormat.is_null() {
        return 0;
    }

    let state = &*(pUserData as *const DecoderState);
    let video_format = *pVideoFormat;

    // Determine output format
    let output_format = if video_format.bit_depth_luma_minus8 > 0 {
        cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_P016
    } else {
        cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12
    };

    // Build decoder creation info
    let create_info = CUVIDDECODECREATEINFO {
        ulWidth: video_format.coded_width as _,
        ulHeight: video_format.coded_height as _,
        ulNumDecodeSurfaces: (video_format.min_num_decode_surfaces as usize + 2) as _,
        CodecType: video_format.codec,
        ChromaFormat: video_format.chroma_format,
        ulCreationFlags: cudaVideoCreateFlags::cudaVideoCreate_PreferCUVID as _,
        bitDepthMinus8: video_format.bit_depth_luma_minus8 as _,
        ulIntraDecodeOnly: 0,
        ulMaxWidth: video_format.coded_width as _,
        ulMaxHeight: video_format.coded_height as _,
        Reserved1: 0,
        display_area: crate::ffi::CUVIDRECT {
            left: video_format.display_area.left as _,
            top: video_format.display_area.top as _,
            right: video_format.display_area.right as _,
            bottom: video_format.display_area.bottom as _,
        },
        OutputFormat: output_format,
        DeinterlaceMode: if video_format.progressive_sequence != 0 {
            cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Weave
        } else {
            cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Adaptive
        },
        ulTargetWidth: video_format.coded_width as _,
        ulTargetHeight: video_format.coded_height as _,
        ulNumOutputSurfaces: 4,
        vidLock: std::ptr::null_mut(),
        target_rect: crate::ffi::CUVIDRECT {
            left: 0,
            top: 0,
            right: video_format.coded_width as _,
            bottom: video_format.coded_height as _,
        },
        enableHistogram: 0,
        Reserved2: [0; 4],
    };

    let funcs = match get_funcs() {
        Ok(f) => f,
        Err(_) => return 0,
    };

    let _ = cu_ctx_set_current();

    let mut ph_decoder: CUvideodecoder = std::ptr::null_mut();
    let result = unsafe { (funcs.create_decoder)(&mut ph_decoder, &create_info) };

    if result != CUDA_SUCCESS || ph_decoder.is_null() {
        return 0;
    }

    // Store decoder handle
    let mut decoder = state.decoder.lock().unwrap();
    *decoder = ph_decoder;

    // Calculate display size
    let display_width = (video_format.display_area.right - video_format.display_area.left) as u32;
    let display_height = (video_format.display_area.bottom - video_format.display_area.top) as u32;

    // Update info
    let mut info = state.info.lock().unwrap();
    *info = DecoderInfo {
        backend: "nvdec".to_string(),
        codec: VideoCodec::DecodeH264,
        coded_size: Extent2D {
            width: video_format.coded_width,
            height: video_format.coded_height,
        },
        display_size: Extent2D {
            width: display_width,
            height: display_height,
        },
        chroma_subsampling: match video_format.chroma_format {
            cudaVideoChromaFormat::cudaVideoChromaFormat_Monochrome => ChromaSubsampling::Monochrome,
            cudaVideoChromaFormat::cudaVideoChromaFormat_420 => ChromaSubsampling::_420,
            cudaVideoChromaFormat::cudaVideoChromaFormat_422 => ChromaSubsampling::_422,
            cudaVideoChromaFormat::cudaVideoChromaFormat_444 => ChromaSubsampling::_444,
        },
        luma_bit_depth: match video_format.bit_depth_luma_minus8 {
            0 => ComponentBitDepth::Bit8,
            2 => ComponentBitDepth::Bit10,
            4 => ComponentBitDepth::Bit12,
            _ => ComponentBitDepth::Bit8,
        },
        chroma_bit_depth: match video_format.bit_depth_chroma_minus8 {
            0 => ComponentBitDepth::Bit8,
            2 => ComponentBitDepth::Bit10,
            4 => ComponentBitDepth::Bit12,
            _ => ComponentBitDepth::Bit8,
        },
        profile_idc: None,
        dpb_slots: video_format.min_num_decode_surfaces as u32 + 1,
    };

    // Store display area
    let mut display_area = state.display_area.lock().unwrap();
    *display_area = (
        video_format.display_area.left,
        video_format.display_area.top,
        video_format.display_area.right,
        video_format.display_area.bottom,
    );

    // Mark as initialized
    let mut initialized = state.initialized.lock().unwrap();
    *initialized = true;

    // Return min_num_decode_surfaces to tell parser how many surfaces we need
    video_format.min_num_decode_surfaces as c_int + 1
}

/// Decode callback: called by parser when a frame is ready to decode.
/// Calls cuvidDecodePicture with the parser-provided CUVIDPICPARAMS.
unsafe extern "C" fn decode_callback(pUserData: *mut c_void, pPicParams: *mut CUVIDPICPARAMS) -> c_int {
    if pUserData.is_null() || pPicParams.is_null() {
        return 0;
    }

    let state = &*(pUserData as *const DecoderState);
    let pic_params = *pPicParams;

    let decoder = {
        let d = state.decoder.lock().unwrap();
        if d.is_null() {
            return 0;
        }
        *d
    };

    let funcs = match get_funcs() {
        Ok(f) => f,
        Err(_) => return 0,
    };

    let _ = cu_ctx_set_current();

    let result = unsafe { (funcs.decode_picture)(decoder, &pic_params) };
    if result != CUDA_SUCCESS {
        let mut err = state.last_error.lock().unwrap();
        *err = Some(format!("cuvidDecodePicture failed: {}", result));
        return 0;
    }

    let _ = cu_ctx_synchronize();

    1
}

/// Display callback: called by parser when a frame is ready for display.
/// We immediately map and copy the frame data before DPB slots are reused.
unsafe extern "C" fn display_callback(pUserData: *mut c_void, pDispInfo: *mut CUVIDPARSERDISPINFO) -> c_int {
    if pUserData.is_null() || pDispInfo.is_null() {
        return 0;
    }

    let state = &*(pUserData as *const DecoderState);
    let disp_info = *pDispInfo;

    if disp_info.picture_index < 0 {
        return 1;
    }

    // Get decoder handle
    let decoder = {
        let d = state.decoder.lock().unwrap();
        if d.is_null() {
            return 0;
        }
        *d
    };

    // Get display size from info
    let info = {
        let i = state.info.lock().unwrap();
        if i.display_size.width == 0 || i.display_size.height == 0 {
            return 1; // Not ready yet
        }
        i.clone()
    };

    let display_width = info.display_size.width as usize;
    let display_height = info.display_size.height as usize;

    // Allocate pinned host memory
    let y_size = display_width * display_height;
    let pinned_y = match cu_mem_host_alloc(y_size) {
        Ok(p) => p,
        Err(_) => return 0,
    };
    let interleaved_uv_size = display_width * (display_height / 2);
    let pinned_uv = match cu_mem_host_alloc(interleaved_uv_size) {
        Ok(p) => p,
        Err(_) => { let _ = cu_mem_free_host(pinned_y); return 0; }
    };

    // Map the decoded frame
    let mut dev_ptr: CUdeviceptr = 0;
    let mut pitch: u32 = 0;
    let funcs = match get_funcs() {
        Ok(f) => f,
        Err(_) => { let _ = cu_mem_free_host(pinned_y); let _ = cu_mem_free_host(pinned_uv); return 0; }
    };
    let _ = cu_ctx_set_current();

    let proc_params = CUVIDPROCPARAMS {
        progressive_frame: disp_info.progressive_frame,
        second_field: 0,
        top_field_first: disp_info.top_field_first,
        unpaired_field: if disp_info.repeat_first_field < 0 { 1 } else { 0 },
        reserved_flags: 0,
        reserved_zero: 0,
        raw_input_dptr: 0,
        raw_input_pitch: 0,
        raw_input_format: 0,
        raw_output_dptr: 0,
        raw_output_pitch: 0,
        Reserved1: 0,
        output_stream: std::ptr::null_mut(),
        Reserved: [0; 46],
        histogram_dptr: std::ptr::null_mut(),
        Reserved2: [std::ptr::null_mut()],
    };

    let result = unsafe {
        (funcs.map_video_frame64)(
            decoder,
            disp_info.picture_index,
            &mut dev_ptr,
            &mut pitch,
            &proc_params,
        )
    };
    if result != CUDA_SUCCESS {
        let _ = cu_mem_free_host(pinned_y);
        let _ = cu_mem_free_host(pinned_uv);
        return 0;
    }

    // Copy Y plane
    let src_y_ptr = dev_ptr as *mut u8;
    let dst_y_ptr = pinned_y as *mut u8;
    for row in 0..display_height {
        let src_row = unsafe { src_y_ptr.add(row * pitch as usize) };
        let dst_row = unsafe { dst_y_ptr.add(row * display_width) };
        let _ = cu_memcpy_dtoh(dst_row as *mut std::ffi::c_void, src_row as CUdeviceptr, display_width);
    }

    // Copy UV plane (NV12: interleaved UV after Y)
    let uv_offset = (pitch as usize) * display_height;
    let src_uv_ptr = unsafe { (dev_ptr as *mut u8).add(uv_offset) };
    let dst_uv_ptr = pinned_uv as *mut u8;
    let uv_height = display_height / 2;
    for row in 0..uv_height {
        let src_row = unsafe { src_uv_ptr.add(row * pitch as usize) };
        let dst_row = unsafe { dst_uv_ptr.add(row * display_width) };
        let _ = cu_memcpy_dtoh(dst_row as *mut std::ffi::c_void, src_row as CUdeviceptr, display_width);
    }

    // Unmap the frame
    let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };

    // Copy from pinned memory to owned buffers
    let mut y_plane = vec![0u8; y_size];
    let mut interleaved_uv = vec![0u8; interleaved_uv_size];
    unsafe {
        std::ptr::copy_nonoverlapping(pinned_y as *const u8, y_plane.as_mut_ptr(), y_size);
        std::ptr::copy_nonoverlapping(pinned_uv as *const u8, interleaved_uv.as_mut_ptr(), interleaved_uv_size);
    }
    let _ = cu_mem_free_host(pinned_y);
    let _ = cu_mem_free_host(pinned_uv);

    // De-interleave NV12 UV to planar U and V
    let uv_size = (display_width / 2) * (display_height / 2);
    let mut u_plane = vec![0u8; uv_size];
    let mut v_plane = vec![0u8; uv_size];
    for y in 0..(display_height / 2) {
        for x in 0..(display_width / 2) {
            let src_idx = y * display_width + x * 2;
            let dst_idx = y * (display_width / 2) + x;
            u_plane[dst_idx] = interleaved_uv[src_idx];
            v_plane[dst_idx] = interleaved_uv[src_idx + 1];
        }
    }

    // Build output buffer
    let mut buffer = Vec::with_capacity(y_size + uv_size * 2);
    buffer.extend_from_slice(&y_plane);
    buffer.extend_from_slice(&u_plane);
    buffer.extend_from_slice(&v_plane);

    let y_ptr = buffer.as_ptr();
    let u_ptr = unsafe { buffer.as_ptr().add(y_size) };
    let v_ptr = unsafe { buffer.as_ptr().add(y_size + uv_size) };

    let pixel_data = Some(PixelData {
        format: "I420".to_string(),
        y: PixelPlane {
            data: y_ptr,
            pitch: display_width,
            width: display_width,
            height: display_height,
        },
        u: PixelPlane {
            data: u_ptr,
            pitch: display_width / 2,
            width: display_width / 2,
            height: display_height / 2,
        },
        v: Some(PixelPlane {
            data: v_ptr,
            pitch: display_width / 2,
            width: display_width / 2,
            height: display_height / 2,
        }),
        buffer,
    });

    // Get frame index
    let frame_index = {
        let mut count = state.frame_count.lock().unwrap();
        let idx = *count;
        *count += 1;
        idx
    };

    // Create decoded frame and push to queue
    let frame = DecodedFrame {
        frame_index,
        timestamp: 0,
        width: info.display_size.width,
        height: info.display_size.height,
        skipped: false,
        pts_valid: false,
        poc: 0,
        field_flags: FieldFlags {
            progressive_frame: disp_info.progressive_frame != 0,
            field_pic: false,
            bottom_field: false,
            second_field: false,
            top_field_first: disp_info.top_field_first != 0,
            unpaired_field: disp_info.repeat_first_field < 0,
            sync_first_ready: false,
            sync_to_first_field: false,
            repeat_first_field: disp_info.repeat_first_field as u8,
            ref_pic: false,
            apply_film_grain: false,
        },
        sync_info: vk_video_core::frame::FrameSyncInfo::default(),
        pixel_data,
    };

    let mut pending = state.pending_frames.lock().unwrap();
    pending.push_back(frame);

    1
}

/// NVDEC H.264 Decoder using NVIDIA's parser.
pub struct NvdecH264Decoder {
    /// Parser handle.
    parser: CUvideoparser,

    /// Decoder state (shared with callbacks).
    state: Box<DecoderState>,

    /// Pending bitstream data.
    pending_data: Vec<u8>,

    /// Offset in pending_data that has been parsed.
    parsed_offset: usize,

    /// Whether parser is initialized.
    parser_initialized: bool,
}

impl NvdecH264Decoder {
    /// Create a new NVDEC H.264 decoder.
    pub fn new(data: Vec<u8>) -> NvdecResult<Self> {
        init_nvdec()?;

        let state = Box::new(DecoderState::new());

        // Create parser with callbacks
        let parser_params = CUVIDPARSERPARAMS {
            CodecType: cudaVideoCodec::cudaVideoCodec_H264,
            ulMaxNumDecodeSurfaces: MAX_DECODE_SURFACES as _,
            ulClockRate: 10000000, // 10MHz default
            ulErrorThreshold: 100,
            ulMaxDisplayDelay: 0, // Zero latency
            bAnnexb_and_reserved: 1, // bAnnexb=1 (bit 0), rest reserved
            uReserved1: [0; 4],
            pUserData: &*state as *const DecoderState as *mut c_void,
            pfnSequenceCallback: Some(sequence_callback),
            pfnDecodePicture: Some(decode_callback),
            pfnDisplayPicture: Some(display_callback),
            pfnGetOperatingPoint: std::ptr::null_mut(),
            pfnGetSEIMsg: std::ptr::null_mut(),
            pvReserved2: [std::ptr::null_mut(); 5],
            pExtVideoInfo: std::ptr::null_mut(),
        };

        let funcs = get_funcs()?;
        let _ = cu_ctx_set_current();

        let mut parser: CUvideoparser = std::ptr::null_mut();
        let result = unsafe { (funcs.create_video_parser)(&mut parser, &parser_params) };

        if result != CUDA_SUCCESS || parser.is_null() {
            return Err(NvdecError::DecoderCreationFailed(format!(
                "cuvidCreateVideoParser failed with error {}",
                result
            )));
        }



        let mut decoder = Self {
            parser,
            state,
            pending_data: data,
            parsed_offset: 0,
            parser_initialized: true,
        };

        // Parse all initial data - parser handles frame ordering internally
        decoder.parse_data()?;

        let initialized = *decoder.state.initialized.lock().unwrap();
        if !initialized {
            return Err(NvdecError::DecoderCreationFailed(
                "Parser did not initialize decoder - no SPS/PPS found".into(),
            ));
        }

        Ok(decoder)
    }

    /// Parse remaining pending data with the parser.
    fn parse_data(&mut self) -> NvdecResult<()> {
        if self.parsed_offset >= self.pending_data.len() {
            return Ok(());
        }

        let funcs = get_funcs()?;
        let remaining = &self.pending_data[self.parsed_offset..];

        let packet = CUVIDSOURCEDATAPACKET {
            flags: 0, // No special flags
            payload_size: remaining.len() as _,
            payload: remaining.as_ptr(),
            timestamp: 0,
        };

        let result = unsafe { (funcs.parse_video_data)(self.parser, &packet) };
        if result != CUDA_SUCCESS {
            return Err(NvdecError::DecodeFailed(format!(
                "cuvidParseVideoData failed with error {}",
                result
            )));
        }

        self.parsed_offset = self.pending_data.len();
        Ok(())
    }

    /// Get the next decoded frame if available.
    fn get_decoded_frame(&self) -> NvdecResult<Option<DecodedFrame>> {
        let frame = {
            let mut pending = self.state.pending_frames.lock().unwrap();
            pending.pop_front()
        };

        Ok(frame)
    }
}

impl Decoder for NvdecH264Decoder {
    type Error = NvdecError;

    fn new(data: Vec<u8>) -> NvdecResult<Self>
    where
        Self: Sized,
    {
        Self::new(data)
    }

    fn new_with_format(
        data: Vec<u8>,
        codec: VideoCodec,
        _format: &VideoFormat,
    ) -> NvdecResult<Self>
    where
        Self: Sized,
    {
        if codec != VideoCodec::DecodeH264 {
            return Err(NvdecError::UnsupportedCodec(codec));
        }
        Self::new(data)
    }

    fn info(&self) -> DecoderInfo {
        self.state.info.lock().unwrap().clone()
    }

    fn submit(&mut self, data: &[u8]) -> NvdecResult<()> {
        self.pending_data.extend_from_slice(data);
        Ok(())
    }

    fn decode(&mut self) -> NvdecResult<Option<DecodedFrame>> {
        if !self.parser_initialized {
            return Err(NvdecError::InvalidState("Parser not initialized".into()));
        }

        // Parse any pending data
        self.parse_data()?;

        // Get decoded frame if available
        self.get_decoded_frame()
    }

    fn flush(&mut self) -> NvdecResult<Vec<DecodedFrame>> {
        let frames = {
            let mut pending = self.state.pending_frames.lock().unwrap();
            pending.drain(..).collect()
        };
        Ok(frames)
    }

    fn reset(&mut self) -> NvdecResult<()> {
        if !self.parser.is_null() {
            let funcs = get_funcs()?;
            unsafe { (funcs.destroy_video_parser)(self.parser) };
            self.parser = std::ptr::null_mut();
        }

        self.pending_data.clear();
        self.parser_initialized = false;

        Ok(())
    }
}

impl Drop for NvdecH264Decoder {
    fn drop(&mut self) {
        // Destroy decoder first
        let decoder_handle = {
            let d = self.state.decoder.lock().unwrap();
            *d
        };
        if !decoder_handle.is_null() {
            if let Ok(funcs) = get_funcs() {
                unsafe { (funcs.destroy_decoder)(decoder_handle) };
            }
        }

        // Then destroy parser
        if !self.parser.is_null() {
            if let Ok(funcs) = get_funcs() {
                unsafe { (funcs.destroy_video_parser)(self.parser) };
            }
            self.parser = std::ptr::null_mut();
        }
    }
}

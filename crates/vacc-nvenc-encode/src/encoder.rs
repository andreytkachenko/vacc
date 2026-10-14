//! NVENC H.264 / H.265 encoder implementing `vacc_core::Encoder`.
//!
//! Synchronous CQP mode with the driver's internal motion estimation:
//! NV12 frames in, Annex B access units out (SPS/PPS emitted by the driver).

use std::mem::zeroed;

use vacc_core::codec::{H264ProfileIdc, H265ProfileIdc, VideoCodec};
use vacc_core::encoder::{EncodedFrame, Encoder, EncoderConfig, EncoderInfo, FrameKind, InputFrame};
use vacc_core::session::Extent2D;

use crate::cuda::Cuda;
use crate::nvenc::{NvencApi, Session};
use crate::nvenc_api as api;
use crate::Error;

/// Input surface pool size (frames the driver may hold references to).
const N_INPUTS: usize = 6;
/// Output bitstream buffer pool size.
const N_OUTPUTS: usize = 2;

/// NVENC H.264/H.265 encoder.
pub struct NvencEncoder {
    // Drop order matters: `session` (uses the vtable) must drop before `api`.
    session: Session,
    api: NvencApi,
    /// Keeps the primary context alive for the encoder's lifetime; released on
    /// drop (after the NVENC session, which references it).
    #[allow(dead_code)]
    cuda: Cuda,

    codec: VideoCodec,
    width: u32,
    height: u32,
    qp: u8,
    intra_period: u32,
    fps_num: u32,
    fps_den: u32,
    frame_index: u32,
    out_next: usize,

    /// SPS+PPS header bytes (global-header mode); prepended to every IDR AU.
    sps_pps: Vec<u8>,

    /// Kept alive for the session's lifetime (passed by pointer to
    /// NvEncInitializeEncoder).
    #[allow(dead_code)]
    config: Box<api::NV_ENC_CONFIG>,
    inputs: Vec<(api::NV_ENC_INPUT_PTR, usize)>,
    outputs: Vec<api::NV_ENC_OUTPUT_PTR>,
}

fn profile_guid(profile: H264ProfileIdc) -> api::GUID {
    match profile {
        H264ProfileIdc::Baseline => api::NV_ENC_H264_PROFILE_BASELINE_GUID,
        H264ProfileIdc::Main => api::NV_ENC_H264_PROFILE_MAIN_GUID,
        H264ProfileIdc::High => api::NV_ENC_H264_PROFILE_HIGH_GUID,
        H264ProfileIdc::High444Predictive => api::NV_ENC_H264_PROFILE_HIGH_444_GUID,
    }
}

/// HEVC level from the max luma sample rate (w*h*fps), Main tier.
fn hevc_level(width: u32, height: u32, fps_num: u32, fps_den: u32) -> u32 {
    let samples = width as u64 * height as u64 * fps_num as u64 / fps_den.max(1) as u64;
    if samples <= 552_960 {
        api::_NV_ENC_LEVEL_NV_ENC_LEVEL_HEVC_1
    } else if samples <= 1_105_920 {
        api::_NV_ENC_LEVEL_NV_ENC_LEVEL_HEVC_2
    } else if samples <= 2_211_840 {
        api::_NV_ENC_LEVEL_NV_ENC_LEVEL_HEVC_3
    } else if samples <= 4_423_680 {
        api::_NV_ENC_LEVEL_NV_ENC_LEVEL_HEVC_31
    } else if samples <= 7_903_488 {
        api::_NV_ENC_LEVEL_NV_ENC_LEVEL_HEVC_4
    } else if samples <= 12_636_960 {
        api::_NV_ENC_LEVEL_NV_ENC_LEVEL_HEVC_41
    } else if samples <= 25_273_920 {
        api::_NV_ENC_LEVEL_NV_ENC_LEVEL_HEVC_5
    } else {
        api::_NV_ENC_LEVEL_NV_ENC_LEVEL_HEVC_51
    }
}

impl NvencEncoder {
    /// Open a CUDA context + NVENC session for the given configuration.
    pub fn open(config: &EncoderConfig) -> Result<Self, Error> {
        config.validate().map_err(Error::InvalidState)?;
        let (codec_guid, profile_guid) = match config.codec {
            VideoCodec::EncodeH264 => (api::NV_ENC_CODEC_H264_GUID, profile_guid(config.h264_profile)),
            VideoCodec::EncodeH265 if config.h265_profile == H265ProfileIdc::Main => (
                api::NV_ENC_CODEC_HEVC_GUID,
                api::NV_ENC_HEVC_PROFILE_MAIN_GUID,
            ),
            other => {
                return Err(Error::CodecNotSupported(format!(
                    "nvenc-encode supports EncodeH264 and 8-bit HEVC Main (got {:?})",
                    other
                )));
            }
        };

        // Device selection: NVD_GPU env var (0-based index), default 0.
        let device_index: i32 = std::env::var("NVD_GPU")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        let cuda = Cuda::open(device_index).map_err(|e| Error::EncoderInit(format!("cuda: {e}")))?;
        let api = NvencApi::open()?;
        let session = api
            .open_session(&cuda)
            .map_err(|e| Error::EncoderInit(format!("open session: {e}")))?;

        // Start from the P3 preset config, then force CQP + our GOP shape.
        // Driver 590+ requires the Ex variant (tuning-aware); older drivers
        // fall back inside `get_preset_config`.
        let mut enc_config = unsafe {
            api.get_preset_config(
                session.handle(),
                codec_guid,
                api::NV_ENC_PRESET_P3_GUID,
                api::NV_ENC_TUNING_INFO_NV_ENC_TUNING_INFO_LOW_LATENCY,
            )
        }
        .map_err(|e| Error::EncoderInit(format!("preset config: {e}")))?;
        enc_config.rcParams.version = api::NV_ENC_RC_PARAMS_VER;
        enc_config.rcParams.rateControlMode = api::_NV_ENC_PARAMS_RC_MODE_NV_ENC_PARAMS_RC_CONSTQP;
        let qp = config.qp as u32;
        enc_config.rcParams.constQP.qpInterP = qp;
        enc_config.rcParams.constQP.qpInterB = qp;
        enc_config.rcParams.constQP.qpIntra = qp;
        enc_config.gopLength = config.intra_period;
        enc_config.frameIntervalP = 1;
        enc_config.frameFieldMode = api::_NV_ENC_PARAMS_FRAME_FIELD_MODE_NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
        enc_config.profileGUID = profile_guid;
        if config.codec == VideoCodec::EncodeH265 {
            // 8-bit Main: pin the bit depths (the preset may default to 10) and
            // size the DPB; level from the sample rate.
            enc_config.encodeCodecConfig.hevcConfig.inputBitDepth =
                api::_NV_ENC_BIT_DEPTH_NV_ENC_BIT_DEPTH_8;
            enc_config.encodeCodecConfig.hevcConfig.outputBitDepth =
                api::_NV_ENC_BIT_DEPTH_NV_ENC_BIT_DEPTH_8;
            enc_config.encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB = config.max_ref_frames;
            enc_config.encodeCodecConfig.hevcConfig.level =
                hevc_level(config.width, config.height, config.fps_num, config.fps_den);
        }

        let mut init: api::_NV_ENC_INITIALIZE_PARAMS = unsafe { zeroed() };
        init.version = api::NV_ENC_INITIALIZE_PARAMS_VER;
        init.encodeGUID = codec_guid;
        init.presetGUID = api::NV_ENC_PRESET_P3_GUID;
        init.encodeWidth = config.width;
        init.encodeHeight = config.height;
        init.darWidth = config.width;
        init.darHeight = config.height;
        init.frameRateNum = config.fps_num;
        init.frameRateDen = config.fps_den;
        init.enableEncodeAsync = 0;
        // The driver's HEVC path rejects the first inter frame (GENERIC) unless
        // picture-type decision is handed to the API; ffmpeg sets this too.
        init.enablePTD = 1;
        init.tuningInfo = api::NV_ENC_TUNING_INFO_NV_ENC_TUNING_INFO_LOW_LATENCY;
        init.bufferFormat = api::_NV_ENC_BUFFER_FORMAT_NV_ENC_BUFFER_FORMAT_NV12;

        let config_box: Box<api::NV_ENC_CONFIG> = Box::new(enc_config);
        init.encodeConfig = &*config_box as *const api::NV_ENC_CONFIG as *mut _;
        unsafe { api.initialize(session.handle(), &mut init) }
            .map_err(|e| Error::EncoderInit(format!("initialize encoder: {e}")))?;

        // Global-header mode: fetch SPS/PPS once; each IDR AU we emit is
        // self-contained (sps_pps prefix + slice), matching the other backends.
        let mut sps_buf = vec![0u8; 4096];
        let sps_len = unsafe { api.get_sequence_params(session.handle(), &mut sps_buf) }
            .map_err(|e| Error::EncoderInit(format!("sequence params: {e}")))?;
        sps_buf.truncate(sps_len);

        // Buffer pools.
        let mut inputs = Vec::with_capacity(N_INPUTS);
        for _ in 0..N_INPUTS {
            let buf = unsafe {
                api.create_input_buffer(session.handle(), config.width, config.height)
            }
            .map_err(|e| Error::EncoderInit(format!("input buffer: {e}")))?;
            inputs.push(buf);
        }
        let mut outputs = Vec::with_capacity(N_OUTPUTS);
        for _ in 0..N_OUTPUTS {
            // Same headroom as the VAAPI probe: w*h*2.
            let buf = unsafe {
                api.create_bitstream_buffer(session.handle(), (config.width * config.height * 2) as usize)
            }
            .map_err(|e| Error::EncoderInit(format!("bitstream buffer: {e}")))?;
            outputs.push(buf);
        }

        Ok(Self {
            session,
            api,
            cuda,
            codec: config.codec,
            width: config.width,
            height: config.height,
            qp: config.qp,
            intra_period: config.intra_period,
            fps_num: config.fps_num,
            fps_den: config.fps_den,
            frame_index: 0,
            out_next: 0,
            sps_pps: sps_buf,
            config: config_box,
            inputs,
            outputs,
        })
    }

    /// Upload one NV12 frame into an input buffer via the CPU staging path:
    /// lock the buffer (host-visible, pitched), copy rows, unlock. Returns the
    /// buffer pitch for `NV_ENC_PIC_PARAMS::inputPitch`.
    fn upload(&self, buf: api::NV_ENC_INPUT_PTR, frame: &InputFrame) -> Result<u32, Error> {
        if frame.width != self.width || frame.height != self.height {
            return Err(Error::InvalidState(format!(
                "input frame {}x{} does not match encoder {}x{}",
                frame.width, frame.height, self.width, self.height
            )));
        }
        // NVENC input buffers are not addressable via cuMemcpyHtoD on current
        // drivers (the pointer is opaque to user CUDA calls); LockInputBuffer
        // exposes the documented host staging view.
        let (host, pitch) = unsafe { self.api.lock_input_buffer(self.session.handle(), buf) }?;
        let w = self.width as usize;
        let pitch = pitch as usize;
        let h = self.height as usize;
        unsafe {
            for row in 0..h {
                std::ptr::copy_nonoverlapping(
                    frame.y[row * frame.y_pitch..row * frame.y_pitch + w].as_ptr(),
                    host.add(row * pitch),
                    w,
                );
            }
            for row in 0..h / 2 {
                std::ptr::copy_nonoverlapping(
                    frame.uv[row * frame.uv_pitch..row * frame.uv_pitch + w].as_ptr(),
                    host.add(pitch * (h + row)),
                    w,
                );
            }
        }
        unsafe { self.api.unlock_input_buffer(self.session.handle(), buf) }?;
        Ok(pitch as u32)
    }

    /// Read back the coded access unit from a bitstream buffer.
    fn read_coded(&self, out: api::NV_ENC_OUTPUT_PTR) -> Result<Vec<u8>, Error> {
        let (ptr, size, hw_status) =
            unsafe { self.api.lock_bitstream(self.session.handle(), out) }?;
        // Status 0x2 (input dropped from the internal queue) still yields a
        // valid access unit on driver 610; an empty output is the real error.
        if size == 0 {
            return Err(Error::Api(format!("hw encode status {hw_status:#x}: no output")));
        }
        let mut data = vec![0u8; size];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), size);
        }
        unsafe { self.api.unlock_bitstream(self.session.handle(), out) }?;
        Ok(data)
    }
}

impl Encoder for NvencEncoder {
    type Error = Error;

    fn new(config: &EncoderConfig) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        Self::open(config)
    }

    fn info(&self) -> EncoderInfo {
        EncoderInfo {
            backend: "nvenc".into(),
            codec: self.codec,
            coded_size: Extent2D::new(self.width, self.height),
            qp: self.qp,
            fps_num: self.fps_num,
            fps_den: self.fps_den,
        }
    }

    fn encode(&mut self, frame: &InputFrame, timestamp: i64) -> Result<Vec<EncodedFrame>, Self::Error> {
        let idx = (self.frame_index as usize) % self.inputs.len();
        let is_idr = self.frame_index == 0 || self.frame_index.is_multiple_of(self.intra_period);

        // 1. Upload input pixels (returns the buffer pitch).
        let pitch = self.upload(self.inputs[idx].0, frame)?;

        // 2. Encode (synchronous).
        let out_idx = self.out_next;
        self.out_next = (self.out_next + 1) % self.outputs.len();
        let mut pic: api::_NV_ENC_PIC_PARAMS = unsafe { zeroed() };
        pic.version = api::NV_ENC_PIC_PARAMS_VER;
        pic.inputWidth = self.width;
        pic.inputPitch = pitch;
        pic.inputHeight = self.height;
        pic.frameIdx = self.frame_index;
        pic.inputTimeStamp = timestamp as u64;
        pic.inputBuffer = self.inputs[idx].0;
        pic.outputBitstream = self.outputs[out_idx];
        pic.bufferFmt = api::_NV_ENC_BUFFER_FORMAT_NV_ENC_BUFFER_FORMAT_NV12;
        pic.pictureStruct = api::_NV_ENC_PIC_STRUCT_NV_ENC_PIC_STRUCT_FRAME;
        match unsafe { self.api.encode_picture(self.session.handle(), &mut pic) } {
            Ok(()) => {}
            Err(e) => {
                // Surface the driver's own description; status 20 (GENERIC)
                // is otherwise a black box.
                let detail = unsafe { self.api.last_error(self.session.handle()) };
                return Err(Error::Api(format!("NvEncEncodePicture failed: {e}; last_error: {detail}")));
            }
        }

        // 3. Read back the access unit; IDRs carry the SPS/PPS prefix so each
        //    IDR AU is self-contained (same contract as the other backends).
        let mut data = self.read_coded(self.outputs[out_idx])?;
        if is_idr {
            let mut prefixed = Vec::with_capacity(self.sps_pps.len() + data.len());
            prefixed.extend_from_slice(&self.sps_pps);
            prefixed.extend_from_slice(&data);
            data = prefixed;
        }
        let index = self.frame_index;
        self.frame_index += 1;

        Ok(vec![EncodedFrame {
            frame_index: index,
            timestamp,
            dts: timestamp,
            data,
            is_idr,
            kind: if is_idr { FrameKind::I } else { FrameKind::P },
            poc: index as i32,
        }])
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>, Self::Error> {
        // Synchronous backend: nothing in flight.
        Ok(Vec::new())
    }
}

impl Drop for NvencEncoder {
    fn drop(&mut self) {
        // Destroy buffer pools before the session (field order already drops
        // `session` first; buffers are NVENC-owned and destroyed with it, but
        // explicit destruction keeps the driver happy on the way out).
        for (buf, _size) in &self.inputs {
            let _ = unsafe { self.api.destroy_input_buffer(self.session.handle(), *buf) };
        }
        for buf in &self.outputs {
            let _ = unsafe { self.api.destroy_bitstream_buffer(self.session.handle(), *buf) };
        }
    }
}

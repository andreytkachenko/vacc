//! NVENC H.264 encoder implementing `vacc_core::Encoder`.
//!
//! Synchronous CQP mode with the driver's internal motion estimation:
//! NV12 frames in, Annex B access units out (SPS/PPS emitted by the driver).

use std::mem::zeroed;

use vacc_core::codec::H264ProfileIdc;
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

/// NVENC H.264 encoder.
pub struct NvencEncoder {
    // Drop order matters: `session` (uses the vtable) must drop before `api`.
    session: Session,
    api: NvencApi,
    cuda: Cuda,

    width: u32,
    height: u32,
    qp: u8,
    intra_period: u32,
    fps_num: u32,
    fps_den: u32,
    frame_index: u32,
    out_next: usize,

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

impl NvencEncoder {
    /// Open a CUDA context + NVENC session for the given configuration.
    pub fn open(config: &EncoderConfig) -> Result<Self, Error> {
        config.validate().map_err(Error::InvalidState)?;
        if config.codec != vacc_core::VideoCodec::EncodeH264 {
            return Err(Error::CodecNotSupported(format!(
                "nvenc-encode supports EncodeH264 only (got {:?})",
                config.codec
            )));
        }

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
        let mut enc_config = unsafe {
            api.get_preset_config(
                session.handle(),
                api::NV_ENC_CODEC_H264_GUID,
                api::NV_ENC_PRESET_P3_GUID,
            )
        }
        .map_err(|e| Error::EncoderInit(format!("preset config: {e}")))?;
        enc_config.rcParams.rateControlMode = api::_NV_ENC_PARAMS_RC_MODE_NV_ENC_PARAMS_RC_CONSTQP;
        let qp = config.qp as u32;
        enc_config.rcParams.constQP.qpInterP = qp;
        enc_config.rcParams.constQP.qpInterB = qp;
        enc_config.rcParams.constQP.qpIntra = qp;
        enc_config.gopLength = config.intra_period;
        enc_config.frameIntervalP = 1;
        enc_config.frameFieldMode = api::_NV_ENC_PARAMS_FRAME_FIELD_MODE_NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
        enc_config.profileGUID = profile_guid(config.h264_profile);

        let mut init: api::_NV_ENC_INITIALIZE_PARAMS = unsafe { zeroed() };
        init.version = api::NV_ENC_INITIALIZE_PARAMS_VER;
        init.encodeGUID = api::NV_ENC_CODEC_H264_GUID;
        init.presetGUID = api::NV_ENC_PRESET_P3_GUID;
        init.encodeWidth = config.width;
        init.encodeHeight = config.height;
        init.darWidth = config.width;
        init.darHeight = config.height;
        init.frameRateNum = config.fps_num;
        init.frameRateDen = config.fps_den;
        init.enableEncodeAsync = 0;
        init.enablePTD = 0;
        init.tuningInfo = api::NV_ENC_TUNING_INFO_NV_ENC_TUNING_INFO_LOW_LATENCY;
        init.bufferFormat = api::_NV_ENC_BUFFER_FORMAT_NV_ENC_BUFFER_FORMAT_NV12;

        let config_box: Box<api::NV_ENC_CONFIG> = Box::new(enc_config);
        init.encodeConfig = &*config_box as *const api::NV_ENC_CONFIG as *mut _;
        unsafe { api.initialize(session.handle(), &mut init) }
            .map_err(|e| Error::EncoderInit(format!("initialize encoder: {e}")))?;

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
            width: config.width,
            height: config.height,
            qp: config.qp,
            intra_period: config.intra_period,
            fps_num: config.fps_num,
            fps_den: config.fps_den,
            frame_index: 0,
            out_next: 0,
            config: config_box,
            inputs,
            outputs,
        })
    }

    /// Upload one NV12 frame into an input buffer (luma then chroma).
    fn upload(&self, buf: api::NV_ENC_INPUT_PTR, frame: &InputFrame) -> Result<(), Error> {
        if frame.width != self.width || frame.height != self.height {
            return Err(Error::InvalidState(format!(
                "input frame {}x{} does not match encoder {}x{}",
                frame.width, frame.height, self.width, self.height
            )));
        }
        let w = self.width as usize;
        // Pack pitched rows into contiguous planes for a single HtoD copy each.
        let mut y = Vec::with_capacity(w * self.height as usize);
        for row in 0..self.height as usize {
            y.extend_from_slice(&frame.y[row * frame.y_pitch..row * frame.y_pitch + w]);
        }
        let mut uv = Vec::with_capacity(w * (self.height / 2) as usize);
        for row in 0..(self.height / 2) as usize {
            uv.extend_from_slice(&frame.uv[row * frame.uv_pitch..row * frame.uv_pitch + w]);
        }
        let base = buf as crate::cuda::CUdeviceptr;
        self.cuda.memcpy_htod(base, &y)?;
        self.cuda.memcpy_htod(base + y.len() as u64, &uv)?;
        Ok(())
    }

    /// Read back the coded access unit from a bitstream buffer.
    fn read_coded(&self, out: api::NV_ENC_OUTPUT_PTR) -> Result<Vec<u8>, Error> {
        let (ptr, size, hw_status) =
            unsafe { self.api.lock_bitstream(self.session.handle(), out) }?;
        if hw_status != 0 {
            return Err(Error::Api(format!("hw encode status {hw_status:#x}")));
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
            codec: vacc_core::VideoCodec::EncodeH264,
            coded_size: Extent2D::new(self.width, self.height),
            qp: self.qp,
            fps_num: self.fps_num,
            fps_den: self.fps_den,
        }
    }

    fn encode(&mut self, frame: &InputFrame, timestamp: i64) -> Result<Vec<EncodedFrame>, Self::Error> {
        let idx = (self.frame_index as usize) % self.inputs.len();
        let is_idr = self.frame_index == 0 || self.frame_index.is_multiple_of(self.intra_period);

        // 1. Upload input pixels.
        self.upload(self.inputs[idx].0, frame)?;

        // 2. Encode (synchronous).
        let out_idx = self.out_next;
        self.out_next = (self.out_next + 1) % self.outputs.len();
        let mut pic: api::_NV_ENC_PIC_PARAMS = unsafe { zeroed() };
        pic.version = api::NV_ENC_PIC_PARAMS_VER;
        pic.inputWidth = self.width;
        pic.inputHeight = self.height;
        pic.frameIdx = self.frame_index;
        pic.inputTimeStamp = timestamp as u64;
        pic.inputBuffer = self.inputs[idx].0;
        pic.outputBitstream = self.outputs[out_idx];
        pic.bufferFmt = api::_NV_ENC_BUFFER_FORMAT_NV_ENC_BUFFER_FORMAT_NV12;
        pic.pictureStruct = api::_NV_ENC_PIC_STRUCT_NV_ENC_PIC_STRUCT_FRAME;
        unsafe { self.api.encode_picture(self.session.handle(), &mut pic) }?;

        // 3. Read back the access unit.
        let data = self.read_coded(self.outputs[out_idx])?;
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

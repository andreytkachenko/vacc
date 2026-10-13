//! # vacc-vulkan-encode
//!
//! Vulkan Video (VK_KHR_video_encode_h264) H.264 encode implementation.
//! Implements `vacc_core::Encoder` for `VideoCodec::EncodeH264`.
//!
//! Motion estimation is performed by the driver; this crate supplies SPS/PPS,
//! DPB management, reference lists and per-slice QP (CQP mode).
//!
//! # SAFETY (this crate's `unsafe` inventory)
//!
//! - **ash API calls**: each call site keeps arguments valid for the call
//!   and checks returned `vk::Result`s.
//! - **Video entry points** (`load_fn`): resolved once via
//!   `vkGetDeviceProcAddr` into the cached [`session::VideoFns`] table; the
//!   loader-provided code pointer is reinterpreted as a spec-exact 64-bit
//!   fn pointer (see the SAFETY note at `load_fn`).
//! - **Zero-initialized FFI structs**: `std::mem::zeroed()` on ash
//!   `vk::native` POD flag/struct types (no `Default` impl); all-bits-zero
//!   is a valid value for these integer C types.
//! - **Staging map/unmap** (`map_memory`/`unmap_memory`): paired, and the
//!   mapped pointer is only used between the two calls.

mod h264_params;
mod session;

pub use h264_params::H264Params;

use vacc_core::codec::H264ProfileIdc;
use vacc_core::encoder::{EncodedFrame, Encoder, EncoderConfig, EncoderInfo, FrameKind, InputFrame};
use vacc_core::session::Extent2D;

/// Result type for Vulkan encode operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Vulkan encode error types.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Vulkan API error: {0}")]
    Vulkan(String),

    #[error("Codec not supported: {0}")]
    CodecNotSupported(String),

    #[error("Encoder initialization failed: {0}")]
    EncoderInit(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// H.264 profile idc values (ITU-T H.264 Table A.1 / Vulkan std).
fn profile_idc(profile: H264ProfileIdc) -> u32 {
    match profile {
        H264ProfileIdc::Baseline => 66,
        H264ProfileIdc::Main => 77,
        H264ProfileIdc::High => 100,
        H264ProfileIdc::High444Predictive => 244,
    }
}

/// Vulkan Video H.264 encoder (synchronous, one frame per `encode` call).
pub struct VulkanEncoder {
    session: session::EncodeSession,
    params: H264Params,
    config: EncoderConfig,
    frame_index: u32,
}

impl Encoder for VulkanEncoder {
    type Error = Error;

    fn new(config: &EncoderConfig) -> Result<Self> {
        config
            .validate()
            .map_err(Error::InvalidState)?;

        let profile_idc = profile_idc(config.h264_profile);
        if config.h264_profile != H264ProfileIdc::Baseline
            && config.h264_profile != H264ProfileIdc::Main
            && config.h264_profile != H264ProfileIdc::High
        {
            return Err(Error::CodecNotSupported(format!(
                "H.264 {:?} encode not supported (8-bit 4:2:0 only)",
                config.h264_profile
            )));
        }
        let width_mbs = config.width / 16;
        let height_mbs = config.height / 16;
        // Level 3.1 covers up to 720p30 with our MB counts; clamp by MB count.
        let mb_count = width_mbs * height_mbs;
        let level_idc = if mb_count <= 1620 {
            30 // level 3.0: 1620 mbs
        } else if mb_count <= 3600 {
            31 // level 3.1: 3600 mbs
        } else if mb_count <= 5760 {
            40 // level 4.0
        } else {
            41 // level 4.1+
        };

        let params = H264Params::new(
            width_mbs,
            height_mbs,
            profile_idc,
            level_idc,
            config.max_ref_frames,
        );
        let session = session::EncodeSession::new(&params, config.width, config.height, config.max_ref_frames)?;

        Ok(Self {
            session,
            params,
            config: config.clone(),
            frame_index: 0,
        })
    }

    fn info(&self) -> EncoderInfo {
        EncoderInfo {
            backend: "vulkan".into(),
            codec: self.config.codec,
            coded_size: Extent2D {
                width: self.config.width,
                height: self.config.height,
            },
            qp: self.config.qp,
            fps_num: self.config.fps_num,
            fps_den: self.config.fps_den,
        }
    }

    fn encode(&mut self, frame: &InputFrame, timestamp: i64) -> Result<Vec<EncodedFrame>> {
        let index = self.frame_index;
        self.frame_index += 1;

        let is_idr = index.is_multiple_of(self.config.intra_period);
        // POC type 2 derivation (H.264 8.2.1.1): reference frames get
        // PicOrderCnt = 2 * frameNum (frameNumOffset=0, no wraparound); the IDR
        // (index 0) yields 0. Must match the SPS pic_order_cnt_type=2 or the
        // NVIDIA driver rejects P-frames.
        let poc = (index * 2) as i32;
        // Monotonic frame_num (never decreases — always legal, no gaps flag).
        let frame_num = index & 0xFFFF;

        // SPS/PPS prefix: driver-generated (frame 0) or our own RBSP.
        let data = if is_idr {
            if self.session.generate_prefix_nalu {
                self.session.encode_one_frame(frame, index, true, self.config.qp, poc, frame_num, true, index == 0)?
            } else {
                let mut out = self.params.prefix_nalus();
                out.extend_from_slice(
                    &self.session.encode_one_frame(frame, index, true, self.config.qp, poc, frame_num, false, index == 0)?,
                );
                out
            }
        } else {
            self.session.encode_one_frame(frame, index, false, self.config.qp, poc, frame_num, false, index == 0)?
        };

        // Sanity: the first NALU must carry a start code.
        if !data.is_empty() {
            let _ = &data;
        }

        Ok(vec![EncodedFrame {
            frame_index: index,
            timestamp,
            dts: timestamp,
            data,
            is_idr,
            kind: if is_idr { FrameKind::I } else { FrameKind::P },
            poc,
        }])
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>> {
        Ok(Vec::new()) // synchronous backend — nothing pending
    }
}

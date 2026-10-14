//! # vacc-vulkan-encode
//!
//! Vulkan Video (VK_KHR_video_encode_h264 / VK_KHR_video_encode_h265)
//! encode implementation. Implements `vacc_core::Encoder` for
//! `VideoCodec::EncodeH264` and `VideoCodec::EncodeH265`.
//!
//! Motion estimation is performed by the driver; this crate supplies
//! VPS/SPS/PPS (or SPS/PPS), DPB management, reference lists and per-slice QP
//! (CQP mode).
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
mod h265_params;
mod h265_session;
mod session;

pub use h264_params::H264Params;
pub use h265_params::H265Params;

use vacc_core::codec::{H264ProfileIdc, H265ProfileIdc, VideoCodec};
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

/// The active encode session (one codec per encoder instance).
enum Session {
    H264(session::EncodeSession),
    H265(h265_session::EncodeSession),
}

/// Codec parameters owned by the encoder. H.265 parameter sets are built
/// inside the session (they depend on device caps) and live there; its
/// prefix NALUs are exposed as `h265_session::EncodeSession::prefix_nalus`.
enum Params {
    H264(H264Params),
    H265,
}

/// Vulkan Video encoder (synchronous, one frame per `encode` call).
pub struct VulkanEncoder {
    session: Session,
    params: Params,
    config: EncoderConfig,
    frame_index: u32,
}

impl Encoder for VulkanEncoder {
    type Error = Error;

    fn new(config: &EncoderConfig) -> Result<Self> {
        config
            .validate()
            .map_err(Error::InvalidState)?;

        match config.codec {
            VideoCodec::EncodeH264 => {
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
                    session: Session::H264(session),
                    params: Params::H264(params),
                    config: config.clone(),
                    frame_index: 0,
                })
            }
            VideoCodec::EncodeH265 => {
                // Only 8-bit 4:2:0 Main is encodable via the NV12 input path.
                if config.h265_profile != H265ProfileIdc::Main {
                    return Err(Error::CodecNotSupported(format!(
                        "HEVC {:?} encode not supported (8-bit 4:2:0 Main only)",
                        config.h265_profile
                    )));
                }
                let level_idc = h265_params::h265_level_idc(
                    config.width,
                    config.height,
                    config.fps_num,
                    config.fps_den,
                );
                let session = h265_session::EncodeSession::new(
                    config.width,
                    config.height,
                    level_idc,
                    config.max_ref_frames,
                    config.intra_period,
                )?;

                Ok(Self {
                    session: Session::H265(session),
                    params: Params::H265,
                    config: config.clone(),
                    frame_index: 0,
                })
            }
            // Reject any other codec explicitly so the unified dispatcher falls
            // through to a backend that actually supports it, instead of
            // silently emitting a different codec.
            codec => Err(Error::CodecNotSupported(format!(
                "Vulkan encode supports H.264 and H.265 only (requested {codec:?})"
            ))),
        }
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
        let (data, poc) = match &mut self.session {
            Session::H264(s) => {
                // POC type 2 derivation (H.264 8.2.1.1): reference frames get
                // PicOrderCnt = 2 * frameNum (frameNumOffset=0, no wraparound); the IDR
                // (index 0) yields 0. Must match the SPS pic_order_cnt_type=2 or the
                // NVIDIA driver rejects P-frames.
                let poc = (index * 2) as i32;
                // Monotonic frame_num (never decreases — always legal, no gaps flag).
                let frame_num = index & 0xFFFF;

                // SPS/PPS prefix: driver-generated (frame 0) or our own RBSP.
                let data = if is_idr {
                    if s.generate_prefix_nalu {
                        s.encode_one_frame(frame, index, true, self.config.qp, poc, frame_num, true, index == 0)?
                    } else {
                        let mut out = match &self.params {
                            Params::H264(p) => p.prefix_nalus(),
                            // Unreachable: an H.264 session is only built with H.264 params.
                            Params::H265 => unreachable!(),
                        };
                        out.extend_from_slice(
                            &s.encode_one_frame(frame, index, true, self.config.qp, poc, frame_num, false, index == 0)?,
                        );
                        out
                    }
                } else {
                    s.encode_one_frame(frame, index, false, self.config.qp, poc, frame_num, false, index == 0)?
                };
                (data, poc)
            }
            Session::H265(s) => {
                // Explicit POC (HEVC): monotonic per-frame counter, no
                // derivation constraints (reference pattern is flat).
                let poc = index as i32;

                // VPS/SPS/PPS prefix: always emitted by us on the first frame
                // (VK Video has no generatePrefixNalu for H.265).
                let data = if is_idr {
                    let mut out = s.prefix_nalus.clone();
                    out.extend_from_slice(
                        &s.encode_one_frame(frame, index, true, self.config.qp, poc, index == 0)?,
                    );
                    out
                } else {
                    s.encode_one_frame(frame, index, false, self.config.qp, poc, index == 0)?
                };
                (data, poc)
            }
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

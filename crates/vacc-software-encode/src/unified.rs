//! [`SoftwareEncoder`]: the core [`vacc_core::encoder::Encoder`] implementation over the
//! native [`H264Encoder`], used by the unified `VaccEncoder` facade (NV12 in, Annex B out).

use crate::{Error, FrameType, H264Encoder, Packet, Preset, Profile, RateControl, YuvFrame};
use vacc_core::codec::H264ProfileIdc;
use vacc_core::encoder::{EncodedFrame, Encoder, EncoderConfig as CoreEncoderConfig, EncoderInfo, FrameKind, InputFrame};
use vacc_core::session::Extent2D;

/// An input picture that has not been emitted as an access unit yet.
struct Pending {
    index: u32,
    pts: i64,
}

/// Software H.264 encoder behind the shared core trait.
///
/// The unified path uses CQP (from `EncoderConfig.qp`), IDR + P for Baseline and
/// IDR + P + B (two B-frames) for Main/High. For the full feature set (CRF/ABR/VBR,
/// presets, AQ, slices, 2-pass) use [`H264Encoder`] directly.
pub struct SoftwareEncoder {
    inner: H264Encoder,
    qp: u8,
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    input_count: u32,
    pending: Vec<Pending>,
}

impl SoftwareEncoder {
    /// Split the NV12 interleaved chroma plane into separate Cb/Cr planes.
    fn deinterleave(frame: &InputFrame) -> (Vec<u8>, Vec<u8>) {
        let cw = frame.width as usize / 2;
        let ch = frame.height as usize / 2;
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        for row in 0..ch {
            let src = row * frame.uv_pitch;
            let dst = row * cw;
            for col in 0..cw {
                u[dst + col] = frame.uv[src + 2 * col];
                v[dst + col] = frame.uv[src + 2 * col + 1];
            }
        }
        (u, v)
    }

    /// Map an output access unit back to its input index by presentation timestamp.
    /// B-frame reordering emits pictures out of input order; the pending list holds
    /// only in-flight pictures (bounded by the B-frame count).
    fn frame_index_for(&mut self, pts: i64) -> u32 {
        match self.pending.iter().position(|p| p.pts == pts) {
            Some(pos) => self.pending.remove(pos).index,
            None => 0,
        }
    }

    fn to_encoded(p: Packet, index: u32) -> EncodedFrame {
        EncodedFrame {
            frame_index: index,
            timestamp: p.pts,
            dts: p.dts,
            data: p.data,
            is_idr: p.keyframe,
            kind: match p.frame_type {
                FrameType::Idr | FrameType::I => FrameKind::I,
                FrameType::P => FrameKind::P,
                FrameType::B => FrameKind::B,
            },
            poc: p.poc,
        }
    }
}

impl Encoder for SoftwareEncoder {
    type Error = Error;

    fn new(config: &CoreEncoderConfig) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        config.validate().map_err(Error::InvalidConfig)?;
        let profile = match config.h264_profile {
            H264ProfileIdc::Baseline => Profile::Baseline,
            H264ProfileIdc::Main => Profile::Main,
            H264ProfileIdc::High => Profile::High,
            other => {
                return Err(Error::InvalidConfig(format!(
                    "profile {other:?} is not supported by the software encoder"
                )))
            }
        };
        let mut cfg = crate::EncoderConfig::new(config.width, config.height, config.fps_num, config.fps_den);
        cfg.profile = profile;
        cfg.preset = Preset::Balanced;
        cfg.rate = RateControl::Qp(config.qp.min(51));
        cfg.keyint = config.intra_period.max(1);
        cfg.bframes = if profile == Profile::Baseline { 0 } else { 2 };
        let inner = H264Encoder::new(cfg)?;
        Ok(Self {
            inner,
            qp: config.qp,
            width: config.width,
            height: config.height,
            fps_num: config.fps_num,
            fps_den: config.fps_den,
            input_count: 0,
            pending: Vec::new(),
        })
    }

    fn info(&self) -> EncoderInfo {
        EncoderInfo {
            backend: "software".into(),
            codec: vacc_core::VideoCodec::EncodeH264,
            coded_size: Extent2D::new(self.width, self.height),
            qp: self.qp,
            fps_num: self.fps_num,
            fps_den: self.fps_den,
        }
    }

    fn encode(&mut self, frame: &InputFrame, timestamp: i64) -> Result<Vec<EncodedFrame>, Self::Error> {
        let (u, v) = Self::deinterleave(frame);
        let yuv = YuvFrame {
            y: frame.y,
            u: &u,
            v: &v,
            y_stride: frame.y_pitch,
            uv_stride: frame.width as usize / 2,
        };
        let index = self.input_count;
        self.input_count += 1;
        self.pending.push(Pending { index, pts: timestamp });
        let aus = self.inner.encode(&yuv, timestamp)?;
        Ok(aus
            .into_iter()
            .map(|p| {
                let pts = p.pts;
                Self::to_encoded(p, self.frame_index_for(pts))
            })
            .collect())
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>, Self::Error> {
        let aus = self.inner.flush();
        Ok(aus
            .into_iter()
            .map(|p| {
                let pts = p.pts;
                Self::to_encoded(p, self.frame_index_for(pts))
            })
            .collect())
    }
}

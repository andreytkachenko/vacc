//! Video codec detection from raw bitstream data (Annex-B or IVF).

use vacc_core::VideoCodec;

/// Detect the video codec of `data`.
///
/// Recognizes H.264, H.265, VP9 and AV1 in Annex-B form, plus VP9/AV1 in an
/// IVF container (disambiguated by the IVF fourcc). Returns `None` when no
/// known codec marker is found.
pub fn detect_codec(data: &[u8]) -> Option<VideoCodec> {
    // IVF container: the codec fourcc at [8..12] disambiguates AV01 from VP09.
    if data.len() >= 32 && &data[0..4] == b"DKIF" {
        return match &data[8..12] {
            b"AV01" => Some(VideoCodec::DecodeAv1),
            _ => Some(VideoCodec::DecodeVp9),
        };
    }

    // AV1 OBU header: marker bit 0b01 (temporal_id | type | size | ext).
    if data.first().is_some_and(|&b| b & 0xC0 == 0x40) {
        return Some(VideoCodec::DecodeAv1);
    }

    // VP9 frame marker: the first non-zero byte has the top two bits set.
    for &b in data.iter().take(256) {
        if b == 0 {
            continue;
        }
        if b & 0xC0 == 0x80 {
            return Some(VideoCodec::DecodeVp9);
        }
        break;
    }

    // NAL-based codecs. H.265 is checked first: NAL types 32-34 (VPS/SPS/PPS)
    // cannot be expressed as H.264 NAL types, and in conformant streams the
    // parameter sets precede slices.
    for i in 0..data.len().min(4096) {
        let start = if i + 4 <= data.len() && data[i..i + 4] == [0x00, 0x00, 0x00, 0x01] {
            i + 4
        } else if i + 3 <= data.len() && data[i..i + 3] == [0x00, 0x00, 0x01] {
            i + 3
        } else {
            continue;
        };
        if start >= data.len() {
            continue;
        }
        let b0 = data[start];
        let h265_nal_type = (b0 >> 1) & 0x3F;
        if h265_nal_type == 32 || h265_nal_type == 33 || h265_nal_type == 34 {
            return Some(VideoCodec::DecodeH265);
        }
        let h264_nal_type = b0 & 0x1F;
        if h264_nal_type == 7 || h264_nal_type == 8 {
            return Some(VideoCodec::DecodeH264);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(name: &str) -> Vec<u8> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/samples/");
        std::fs::read(format!("{path}{name}")).unwrap()
    }

    #[test]
    fn detects_all_codecs() {
        assert_eq!(detect_codec(&sample("h264_main.h264")), Some(VideoCodec::DecodeH264));
        assert_eq!(detect_codec(&sample("h265_main.h265")), Some(VideoCodec::DecodeH265));
        assert_eq!(detect_codec(&sample("vp9_profile1_444.ivf")), Some(VideoCodec::DecodeVp9));
        assert_eq!(detect_codec(&sample("av1_main.ivf")), Some(VideoCodec::DecodeAv1));
    }

    #[test]
    fn rejects_unknown_data() {
        assert_eq!(detect_codec(&[]), None);
        assert_eq!(detect_codec(&[0u8; 64]), None);
        assert_eq!(detect_codec(&[1u8, 2, 3, 4, 5]), None);
    }
}

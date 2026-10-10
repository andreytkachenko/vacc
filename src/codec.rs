//! Video codec detection from raw bitstream data (Annex-B, raw OBU, or IVF).

use vacc_core::VideoCodec;

/// Detect the video codec of `data`.
///
/// Recognizes H.264 and H.265 in Annex-B form, VP9 and AV1 raw, plus
/// VP9/AV1 in an IVF container (disambiguated by the IVF fourcc). Returns
/// `None` when no known codec marker is found.
pub fn detect_codec(data: &[u8]) -> Option<VideoCodec> {
    // IVF container: the codec fourcc at [8..12] disambiguates AV01 from VP09.
    if data.len() >= 32 && &data[0..4] == b"DKIF" {
        return match &data[8..12] {
            b"AV01" => Some(VideoCodec::DecodeAv1),
            _ => Some(VideoCodec::DecodeVp9),
        };
    }

    // Raw AV1 OBU stream (low-overhead or Annex B format).
    if detect_av1_obu(data) {
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

/// Detect a raw (unwrapped) AV1 OBU stream, in either of the two bitstream
/// formats handled by `vacc_parser::av1`:
/// - Annex B (length-delimited): starts with temporal/frame unit LEB128
///   sizes, followed by length-prefixed OBUs; the first OBU is a temporal
///   delimiter.
/// - Low overhead: back-to-back OBUs, each carrying its own size field; the
///   stream opens with a temporal delimiter OBU and the first frame unit
///   contains a sequence header OBU.
fn detect_av1_obu(data: &[u8]) -> bool {
    detect_av1_annexb(data) || detect_av1_low_overhead(data)
}

/// Read an unsigned LEB128 integer at `pos`; returns `(value, bytes used)`.
fn read_leb128(data: &[u8], pos: usize) -> Option<(usize, usize)> {
    let mut value = 0usize;
    let mut shift = 0;
    let mut i = pos;
    while i - pos < 8 {
        let &b = data.get(i)?;
        i += 1;
        value = value.checked_add(((b & 0x7F) as usize) << shift)?;
        if b & 0x80 == 0x00 {
            return Some((value, i - pos));
        }
        shift += 7;
    }
    None
}

/// Parse the 1- or 2-byte OBU header at `pos` in a new-format AV1 stream:
/// `forbidden(1) type(4) extension(1) has_size(1) reserved(1)`, with an
/// optional second byte when the extension flag is set. Returns
/// `(obu_type, has_size_field, position after the header)`.
fn av1_obu_header(data: &[u8], pos: usize) -> Option<(u8, bool, usize)> {
    let &b0 = data.get(pos)?;
    if b0 & 0x80 != 0x00 {
        return None; // forbidden bit must be zero
    }
    let obu_type = (b0 >> 3) & 0x0F;
    // Types 1..=8 are defined OBUs; 0 and 9..=14 are reserved.
    if !(1..=8).contains(&obu_type) {
        return None;
    }
    let has_size = b0 & 0x02 != 0x00;
    Some((obu_type, has_size, pos + 1 + usize::from(b0 & 0x04 != 0x00)))
}

/// Annex B probe: temporal unit size, frame unit size, first OBU length,
/// then a temporal delimiter header — mirroring `Av1Parser::annexb_probe`.
fn detect_av1_annexb(data: &[u8]) -> bool {
    let Some((tu, n)) = read_leb128(data, 0) else { return false };
    if tu == 0 {
        return false;
    }
    let Some((fu, m)) = read_leb128(data, n) else { return false };
    if fu == 0 || fu > tu {
        return false;
    }
    let Some((len, k)) = read_leb128(data, n + m) else { return false };
    if len == 0 || len > fu {
        return false;
    }
    let Some((obu_type, _, _)) = av1_obu_header(data, n + m + k) else { return false };
    // The first OBU of each temporal unit must be a temporal delimiter.
    obu_type == 2
}

/// Low-overhead probe: walk the sized OBU chain from offset zero and require
/// both a temporal delimiter and a sequence header OBU before the chain
/// breaks or the data ends.
fn detect_av1_low_overhead(data: &[u8]) -> bool {
    let mut i = 0;
    let mut seen_delimiter = false;
    let mut seen_sequence = false;
    while i < data.len() {
        let (obu_type, has_size, hdr_end) = match av1_obu_header(data, i) {
            Some(h) => h,
            None => break,
        };
        // In the low-overhead format every OBU carries a size field.
        if !has_size {
            break;
        }
        let (size, n) = match read_leb128(data, hdr_end) {
            Some(v) => v,
            None => break,
        };
        seen_delimiter |= obu_type == 2;
        seen_sequence |= obu_type == 1;
        let end = match hdr_end.checked_add(n).and_then(|e| e.checked_add(size)) {
            Some(e) => e,
            None => break,
        };
        if end > data.len() {
            break; // truncated tail of a longer stream
        }
        i = end;
    }
    seen_delimiter && seen_sequence
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

    /// Raw AV1, low-overhead format: temporal delimiter (empty), sequence
    /// header (3-byte payload), frame (5-byte payload).
    #[test]
    fn detects_raw_av1_low_overhead() {
        let stream: Vec<u8> = [
            0x12, 0x00, // TD OBU: type 2, has_size, size 0
            0x0A, 0x03, 0xAA, 0xBB, 0xCC, // sequence header, size 3
            0x32, 0x05, 0x01, 0x02, 0x03, 0x04, 0x05, // frame, size 5
        ]
        .to_vec();
        assert_eq!(detect_codec(&stream), Some(VideoCodec::DecodeAv1));
        // A truncated prefix (TD + partial sequence header) is still AV1.
        assert_eq!(detect_codec(&stream[..6]), Some(VideoCodec::DecodeAv1));
    }

    /// Raw AV1, Annex B (length-delimited) format: unit sizes, then a
    /// length-prefixed temporal delimiter OBU.
    #[test]
    fn detects_raw_av1_annexb() {
        let stream: Vec<u8> = [
            0x06, // temporal unit size
            0x06, // frame unit size
            0x04, // first OBU length
            0x10, // TD OBU header: type 2, no size field
            0xAA, 0xBB, 0xCC, // TD payload
        ]
        .to_vec();
        assert_eq!(detect_codec(&stream), Some(VideoCodec::DecodeAv1));
    }

    /// H.264 Annex-B data must not be confused with a raw AV1 OBU stream.
    #[test]
    fn raw_av1_probe_rejects_annexb_nal() {
        let h264: Vec<u8> = [0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0xC0, 0x1E].to_vec();
        assert!(!detect_av1_obu(&h264));
    }
}

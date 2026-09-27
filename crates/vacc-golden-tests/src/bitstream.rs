//! Annex-B access-unit splitting for the common incremental test.

/// True if the stream looks like HEVC: its first NAL unit is a 6-bit-type
/// VPS/SPS/PPS (0x40/0x42/0x44 with forbidden bit 0). H.264 streams open
/// with SPS (0x67) or IDR (0x65), which never match.
pub fn is_hevc_stream(data: &[u8]) -> bool {
    let Some(first) = first_nal_byte(data) else {
        return false;
    };
    matches!(first, 0x40 | 0x42 | 0x44)
}

/// First NAL header byte of an Annex-B stream (after the first start code).
fn first_nal_byte(data: &[u8]) -> Option<u8> {
    let mut i = 0usize;
    while i + 3 < data.len() {
        let sc = if data[i..i + 4] == [0, 0, 0, 1] {
            4
        } else if data[i..i + 3] == [0, 0, 1] {
            3
        } else {
            i += 1;
            continue;
        };
        if i + sc < data.len() {
            return Some(data[i + sc]);
        }
        return None;
    }
    None
}

/// True for NAL unit types that carry slice data. `hevc` selects the family:
/// HEVC VCL units are 0..=31 (truncated unary code; VPS/SPS/PPS/AUD/SEI are
/// 32+), while H.264 slices are 1..=5 / 19..=20 with parameter sets and SEI
/// in between.
pub fn is_slice_ty(nal_unit_type: u8, hevc: bool) -> bool {
    if hevc {
        nal_unit_type <= 31
    } else {
        nal_unit_type != 0 && !(6..=14).contains(&nal_unit_type)
    }
}

/// Split annex-B `data` into `(start_code_offset, nal_unit_type, bytes)`
/// units, where `bytes` spans from the unit's start code to the next. The
/// type is extracted per family: H.264 uses the low 5 bits of the NAL byte;
/// HEVC uses the 6-bit field `(byte >> 1) & 0x3F`.
pub fn split(data: &[u8]) -> Vec<(usize, u8, Vec<u8>)> {
    let hevc = is_hevc_stream(data);
    let mut heads: Vec<(usize, u8)> = Vec::new();
    let mut i = 0usize;
    while i + 3 < data.len() {
        let sc = if data[i..i + 4] == [0, 0, 0, 1] {
            4
        } else if data[i..i + 3] == [0, 0, 1] {
            3
        } else {
            i += 1;
            continue;
        };
        if i + sc < data.len() {
            let b = data[i + sc];
            let t = if hevc { (b >> 1) & 0x3F } else { b & 0x1F };
            heads.push((i, t));
        }
        i += sc;
    }
    heads
        .windows(2)
        .map(|w| (w[0].0, w[0].1, data[w[0].0..w[1].0].to_vec()))
        .chain(std::iter::once({
            let (o, t) = heads.last().unwrap();
            (*o, *t, data[*o..].to_vec())
        }))
        .collect()
}

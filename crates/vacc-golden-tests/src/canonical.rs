//! Canonical pixel normalization for cross-backend golden pinning.
//!
//! A backend `PixelData` is normalized to canonical planar Y+U+V bytes
//! (packed rows, cropped to the display size). This is the same canonical
//! form used by `verify-all.py` and the `decode` example binary, so goldens
//! pinned here are comparable across backends and against ffmpeg:
//!
//! - semi-planar (NV12/P016, `v == None`): de-interleave U/V;
//! - YV12: the u/v fields are swapped relative to I420;
//! - 16-bit samples: cuvid P016 and iHD P016 surfaces store 10-bit values
//!   top-justified (value << 6) — shift to bottom-justified (matching
//!   ffmpeg's yuv420p10le layout); the `*_LE` / Y410P16 formats are already
//!   bottom-justified;
//! - GRAY: luma only.

use vacc_core::frame::{PixelData, PixelPlane};

/// Normalize a backend `PixelData` to canonical planar Y+U+V bytes.
pub fn canonical_pixels(pd: &PixelData) -> Vec<u8> {
    // (bytes per sample, top-justified?)
    let (bps, top_justified) = match pd.format.as_str() {
        "P016" | "I420_16BIT" | "GRAY_16BIT" | "YUV444_16BIT" => (2, true),
        "Y410P16" | "P010LE" | "P012LE" | "I420-10" => (2, false),
        _ => (1, false),
    };

    let mut out = Vec::with_capacity(pd.buffer.len());
    copy_plane(&mut out, &pd.y, bps, top_justified);

    if pd.format.starts_with("GRAY") {
        return out;
    }

    if pd.v.is_none() {
        // Semi-planar: U and V interleaved in the u plane. De-interleave
        // into planar U then planar V.
        for row in 0..pd.u.height {
            let src = unsafe { pd.u.data.add(row * pd.u.pitch) };
            for col in 0..pd.u.width {
                push_sample(&mut out, unsafe { src.add(col * 2 * bps) }, bps, top_justified);
            }
        }
        for row in 0..pd.u.height {
            let src = unsafe { pd.u.data.add(row * pd.u.pitch) };
            for col in 0..pd.u.width {
                push_sample(&mut out, unsafe { src.add(col * 2 * bps + bps) }, bps, top_justified);
            }
        }
    } else if let Some(v) = pd.v.as_ref() {
        if pd.format == "YV12" {
            // YV12 stores V before U.
            copy_plane(&mut out, v, bps, top_justified);
            copy_plane(&mut out, &pd.u, bps, top_justified);
        } else {
            copy_plane(&mut out, &pd.u, bps, top_justified);
            copy_plane(&mut out, v, bps, top_justified);
        }
    }

    out
}

/// Append one sample from `src`, honoring bytes-per-sample and the iHD
/// P016 top-justification (value << 6, normalized with >> 6).
fn push_sample(out: &mut Vec<u8>, src: *const u8, bps: usize, top_justified: bool) {
    if top_justified && bps == 2 {
        let v = u16::from_le_bytes(unsafe { [*src, *src.add(1)] }) >> 6;
        out.extend_from_slice(&v.to_le_bytes());
    } else {
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(src, bps) });
    }
}

/// Copy one plane row by row (honoring pitch) into `out`.
fn copy_plane(out: &mut Vec<u8>, plane: &PixelPlane, bps: usize, top_justified: bool) {
    for row in 0..plane.height {
        let src = unsafe { plane.data.add(row * plane.pitch) };
        if top_justified && bps == 2 {
            for col in 0..plane.width {
                push_sample(out, unsafe { src.add(col * bps) }, bps, true);
            }
        } else {
            out.extend_from_slice(unsafe { std::slice::from_raw_parts(src, plane.width * bps) });
        }
    }
}

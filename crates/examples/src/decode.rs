//! Unified GPU video decode example (H.264 / H.265 / VP9 / AV1).
//!
//! Decodes a whole file with the selected backend and prints, for every
//! decoded display frame: presentation timestamp (ms), size, and an FNV-1a
//! 64-bit hash of the canonical planar YUV pixels (Y+U+V packed rows, bytes
//! per sample matching the bit depth). The hash lets you compare pixel
//! output across backends line by line.
//!
//! Input:
//! - IVF container (VP9 / AV1): pts comes from the container timebase and is
//!   mapped 1:1 onto display frames (hidden frames are skipped, matching the
//!   decoders).
//! - Raw bitstream (.h264 / .h265 / .vp9): there is no container; pts is the
//!   presentation-order frame index on an assumed 1/30 s timebase.
//!
//! Usage:
//!   cargo run --release -p examples --example decode -- -b <backend> -i <file> [-n max_frames]
//!
//! Backends: vaapi (H.264/H.265/VP9), vulkan (H.264/H.265/VP9/AV1),
//! nvdec (H.264/H.265/VP9/AV1, requires an NVIDIA GPU).
//!
//! Examples:
//!   cargo run --release -p examples --example decode -- -b vulkan -i assets/big_buck_bunney_vp9.ivf
//!   cargo run --release -p examples --example decode -- -b nvdec  -i samples/av1_gop1.ivf -n 30
//!   cargo run --release -p examples --example decode -- -b vaapi  -i assets/born_trailer.h264

use std::time::Instant;

use vk_video_core::decoder::Decoder;
use vk_video_core::frame::{DecodedFrame as CoreFrame, PixelData, PixelPlane};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    Vaapi,
    Vulkan,
    Nvdec,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Backend::Vaapi => "vaapi",
            Backend::Vulkan => "vulkan",
            Backend::Nvdec => "nvdec",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Codec {
    H264,
    H265,
    Vp9,
    Av1,
}

impl Codec {
    fn name(self) -> &'static str {
        match self {
            Codec::H264 => "H.264",
            Codec::H265 => "H.265",
            Codec::Vp9 => "VP9",
            Codec::Av1 => "AV1",
        }
    }
}

struct Args {
    backend: Backend,
    input: String,
    max_frames: usize,
}

/// Upper bound used when the user does not pass -n. Large enough for any
/// real stream; kept finite because some decode paths multiply it (e.g.
/// `max_frames * 2` access-unit budgets).
const MAX_FRAMES_DEFAULT: usize = 1_000_000;

fn usage() -> ! {
    eprintln!(
        "Usage: decode -b <vaapi|vulkan|nvdec> -i <file.ivf|h264|h265|vp9> [-n max_frames]\n\
         \n\
         Decodes the file and prints per display frame: pts (ms), size, and a\n\
         hash of the canonical planar YUV pixels.\n\
         \n\
         Options:\n\
           -b, --backend <vaapi|vulkan|nvdec>   decode backend (required)\n\
           -i, --input <file>                   input file (required; IVF for VP9/AV1,\n\
                                                raw Annex-B .h264/.h265 or .vp9 also accepted)\n\
           -n, --max-frames <N>                 stop after N display frames (default: all)"
    );
    std::process::exit(1);
}

fn die(msg: &str) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(1);
}

fn parse_args() -> Args {
    let mut backend = None;
    let mut input = None;
    let mut max_frames = MAX_FRAMES_DEFAULT;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-b" | "--backend" => {
                let value = match args.next() {
                    Some(v) => v,
                    None => usage(),
                };
                backend = Some(match value.as_str() {
                    "vaapi" => Backend::Vaapi,
                    "vulkan" => Backend::Vulkan,
                    "nvdec" => Backend::Nvdec,
                    other => die(&format!("unknown backend '{}' (expected vaapi, vulkan or nvdec)", other)),
                });
            }
            "-i" | "--input" => input = Some(args.next().unwrap_or_else(|| usage())),
            "-n" | "--max-frames" => {
                let value = args.next().unwrap_or_else(|| usage());
                max_frames = value.parse().unwrap_or_else(|_| die(&format!("invalid -n value: {}", value)));
            }
            "-h" | "--help" => usage(),
            other => die(&format!("unknown argument: {}", other)),
        }
    }

    Args {
        backend: backend.unwrap_or_else(|| usage()),
        input: input.unwrap_or_else(|| usage()),
        max_frames,
    }
}

/// IVF container (magic "DKIF"): 32-byte header followed by packets of
/// [u32 payload size][u64 pts][payload]. Header layout (ffmpeg ivfenc.c):
/// codec fourcc @8, u16 width @12, u16 height @14, u32 time_base.den @16,
/// u32 time_base.num @20, u32 frame_count @24.
struct Ivf {
    codec: Codec,
    width: u32,
    height: u32,
    tb_num: u32,
    tb_den: u32,
    declared_frames: u32,
    packets: Vec<(u64, Vec<u8>)>,
}

fn parse_ivf(data: &[u8]) -> Option<Ivf> {
    if data.len() < 32 || &data[0..4] != b"DKIF" {
        return None;
    }
    let codec = match &data[8..12] {
        b"VP90" => Codec::Vp9,
        b"AV01" => Codec::Av1,
        other => {
            eprintln!("warning: unsupported IVF codec fourcc {:?}; treating as opaque", String::from_utf8_lossy(other));
            return None;
        }
    };
    let header_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    if header_len < 32 || data.len() <= header_len {
        eprintln!("warning: malformed IVF header (len {})", header_len);
        return None;
    }
    let width = u16::from_le_bytes([data[12], data[13]]) as u32;
    let height = u16::from_le_bytes([data[14], data[15]]) as u32;
    let tb_den = u32::from_le_bytes(data[16..20].try_into().unwrap());
    let tb_num = u32::from_le_bytes(data[20..24].try_into().unwrap());
    let declared_frames = u32::from_le_bytes(data[24..28].try_into().unwrap());

    let mut packets = Vec::new();
    let mut off = header_len;
    while off + 12 <= data.len() {
        let size = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        if size == 0 || off + 12 + size > data.len() {
            break;
        }
        let pts = u64::from_le_bytes(data[off + 4..off + 12].try_into().unwrap());
        packets.push((pts, data[off + 12..off + 12 + size].to_vec()));
        off += 12 + size;
    }

    Some(Ivf {
        codec,
        width,
        height,
        tb_num,
        tb_den,
        declared_frames,
        packets,
    })
}

/// Split a VP9 IVF payload into its subframes. A superframe carries a start
/// code and a per-frame size index at the tail; each subframe is one frame.
fn vp9_subframes(payload: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    if payload.len() < 2 {
        out.push(payload);
        return out;
    }
    let final_byte = payload[payload.len() - 1];
    if (final_byte & 0xE0) != 0xC0 {
        // Not a superframe.
        out.push(payload);
        return out;
    }
    let num_frames = (final_byte & 0x07) as usize + 1;
    if num_frames <= 1 {
        out.push(payload);
        return out;
    }
    let mag = ((final_byte >> 3) & 0x03) as usize + 1;
    let index_size = 2 + mag * num_frames;
    if payload.len() < index_size {
        out.push(payload);
        return out;
    }
    let index_start = payload.len() - index_size;
    if payload[index_start] != final_byte {
        out.push(payload);
        return out;
    }
    let frame_data_size = index_start;
    let mut offset = 0usize;
    let mut x = index_start + 1;
    for _ in 0..num_frames {
        let mut this_sz = 0usize;
        for j in 0..mag {
            this_sz |= (payload[x + j] as usize) << (j * 8);
        }
        x += mag;
        if offset + this_sz <= frame_data_size {
            out.push(&payload[offset..offset + this_sz]);
        }
        offset += this_sz;
    }
    out
}

/// VP9: does this (sub)frame produce a display picture?
/// Frame header byte 0: bit 7 = show_existing_frame, bit 6 = show_frame.
fn vp9_shows(data: &[u8]) -> bool {
    data.first().map_or(false, |&b| b & 0xC0 != 0)
}

/// AV1: walk the OBU sequence of an IVF packet and report whether its Frame
/// OBU is displayed (bit 7 = show_existing_frame, bit 6 = show_frame of the
/// frame header).
fn av1_shows(payload: &[u8]) -> bool {
    let mut i = 0usize;
    while i < payload.len() {
        let b = payload[i];
        let obu_type = (b >> 3) & 7;
        let has_size = b & 1 == 1;
        let mut start = i + 1;
        let size: usize;
        if has_size {
            let mut v = 0usize;
            let mut shift = 0u32;
            loop {
                let Some(&c) = payload.get(start) else {
                    return false;
                };
                v |= ((c & 0x7f) as usize) << shift;
                start += 1;
                shift += 7;
                if c & 0x80 == 0 {
                    break;
                }
            }
            size = v;
        } else {
            // Size-less OBU: assume it runs to the end of the packet.
            size = payload.len() - start;
        }
        if obu_type == 1 {
            return payload.get(start).map_or(false, |&fb| fb & 0xC0 != 0);
        }
        let next = start.saturating_add(size);
        if next <= i {
            break; // malformed; avoid spinning
        }
        i = next;
    }
    false
}

/// Container pts (ticks) of every display frame, in display order. VP9 and
/// AV1 have no B-frames, so display order equals packet order; hidden
/// frames are skipped exactly like the decoders do.
fn ivf_display_pts(ivf: &Ivf) -> Vec<u64> {
    let mut out = Vec::with_capacity(ivf.packets.len());
    for (pts, payload) in &ivf.packets {
        match ivf.codec {
            Codec::Vp9 => {
                for sub in vp9_subframes(payload) {
                    if vp9_shows(sub) {
                        out.push(*pts);
                    }
                }
            }
            Codec::Av1 => {
                if av1_shows(payload) {
                    out.push(*pts);
                }
            }
            _ => out.push(*pts),
        }
    }
    out
}

/// Detect the codec of a raw (non-IVF) bitstream: file extension first, then
/// parameter-set NAL scanning.
fn detect_codec(data: &[u8], path: &str) -> Codec {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "h264" | "avc" | "264" => return Codec::H264,
        "h265" | "hevc" | "265" => return Codec::H265,
        "vp9" => return Codec::Vp9,
        "av1" => return Codec::Av1,
        _ => {}
    }

    // Scan for start codes and classify by parameter-set NAL type.
    let scan = data.len().min(4096);
    for i in 0..scan {
        let nal = if i + 4 <= data.len() && data[i..i + 4] == [0, 0, 0, 1] {
            data.get(i + 4).copied()
        } else if i + 3 <= data.len() && data[i..i + 3] == [0, 0, 1] {
            data.get(i + 3).copied()
        } else {
            continue;
        };
        match nal {
            Some(n) if n & 0x1f == 7 || n & 0x1f == 8 => return Codec::H264, // SPS / PPS
            Some(n) if (n >> 1) == 32 || (n >> 1) == 33 || (n >> 1) == 34 => return Codec::H265, // VPS/SPS/PPS
            Some(n) if n & 0xC0 == 0x80 && (n & 0x1F) != 7 && (n & 0x1F) != 8 => return Codec::Vp9,
            _ => {}
        }
    }

    die(&format!(
        "cannot detect codec of {} (use an .ivf/.h264/.h265/.vp9 file or fix the extension)",
        path
    ));
}

/// FNV-1a 64-bit.
fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcb_f2_9c_e4_84_22_23_25;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x00_00_01_00_00_00_01_b3);
    }
    hash
}

/// Format milliseconds without a trailing dot (e.g. 0, 33.333, 100).
fn fmt_ms(v: f64) -> String {
    let s = format!("{:.3}", v);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-" {
        "0".to_string()
    } else {
        s.to_string()
    }
}

/// One decoded display frame, regardless of backend.
enum Frame {
    Core(CoreFrame),
    Vk(vk_video_vulkan::DecodedFrame),
}

impl Frame {
    /// Display size (width, height).
    fn size(&self) -> (u32, u32) {
        match self {
            Frame::Core(f) => (f.width, f.height),
            Frame::Vk(f) => (f.display_width, f.display_height),
        }
    }

    /// Canonical planar Y+U+V bytes (packed rows, `bps` bytes per sample,
    /// cropped to the display size), or None for skipped/empty frames.
    fn canonical_pixels(&self) -> Option<Vec<u8>> {
        match self {
            Frame::Core(f) => f.pixel_data.as_ref().map(canonical_core),
            Frame::Vk(f) => Some(canonical_vk(f)),
        }
    }
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

/// Normalize a backend `PixelData` to canonical planar Y+U+V bytes:
/// - semi-planar (NV12/P016, `v == None`): de-interleave U/V;
/// - YV12: the u/v fields are swapped relative to I420;
/// - 16-bit samples (P016 / *_16BIT) are always 10-bit content and are
///   top-justified by both NVIDIA and iHD (value << 6); shift them to the
///   bottom-justified yuv420p10le layout (matching ffmpeg);
/// - GRAY: luma only.
fn canonical_core(pd: &PixelData) -> Vec<u8> {
    let bps = if matches!(pd.format.as_str(), "P016" | "Y410P16") || pd.format.ends_with("_16BIT") {
        2
    } else {
        1
    };
    // Every bps=2 core format carries top-justified 10-bit P016 samples.
    let top_justified = bps == 2;

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
    } else {
        let v = pd.v.as_ref().unwrap();
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

/// Copy a cropped region of a Vulkan readback plane (samples per row =
/// `stride_samples`) into `out`.
fn crop_plane(out: &mut Vec<u8>, plane: &[u8], stride_samples: usize, x0: usize, y0: usize, w: usize, h: usize, bps: usize) {
    for y in y0..y0 + h {
        let start = (y * stride_samples + x0) * bps;
        out.extend_from_slice(&plane[start..start + w * bps]);
    }
}

/// Chroma subsampling factor along one axis: 1 for full-resolution chroma
/// (4:4:4/mono), 2 when the chroma plane is half the coded size (odd coded
/// sizes round up, so compare against `coded - 1`).
fn chroma_sub(coded: u32, chroma: u32) -> usize {
    if chroma == 0 || chroma >= coded {
        1
    } else if chroma * 2 >= coded.saturating_sub(1) {
        2
    } else {
        1
    }
}

/// Normalize a Vulkan `DecodedFrame` to canonical planar Y+U+V bytes,
/// cropped to the display size. Planes are full coded size with
/// `sample_size` bytes per sample.
fn canonical_vk(frame: &vk_video_vulkan::DecodedFrame) -> Vec<u8> {
    let bps = frame.pixels.sample_size as usize;
    let disp_w = frame.display_width as usize;
    let disp_h = frame.display_height as usize;
    let h_sub = chroma_sub(frame.coded_width, frame.pixels.chroma_width);
    let v_sub = chroma_sub(frame.coded_height, frame.pixels.chroma_height);

    let mut out = Vec::with_capacity(frame.pixels.y_plane.len() + frame.pixels.u_plane.len() + frame.pixels.v_plane.len());
    crop_plane(
        &mut out,
        &frame.pixels.y_plane,
        frame.coded_width as usize,
        frame.crop_left as usize,
        frame.crop_top as usize,
        disp_w,
        disp_h,
        bps,
    );
    // Mono images carry no chroma planes.
    if !frame.pixels.u_plane.is_empty() {
        crop_plane(
            &mut out,
            &frame.pixels.u_plane,
            frame.pixels.chroma_width as usize,
            (frame.crop_left as usize) / h_sub,
            (frame.crop_top as usize) / v_sub,
            disp_w / h_sub,
            disp_h / v_sub,
            bps,
        );
    }
    if !frame.pixels.v_plane.is_empty() {
        crop_plane(
            &mut out,
            &frame.pixels.v_plane,
            frame.pixels.chroma_width as usize,
            (frame.crop_left as usize) / h_sub,
            (frame.crop_top as usize) / v_sub,
            disp_w / h_sub,
            disp_h / v_sub,
            bps,
        );
    }
    out
}

/// Drain a core-trait decoder (VAAPI / NVDEC) into display-order frames.
fn decode_all_core<D: Decoder>(decoder: &mut D, max_frames: usize) -> Vec<CoreFrame> {
    let mut frames = Vec::new();
    loop {
        match decoder.decode() {
            Ok(Some(frame)) => {
                frames.push(frame);
                if frames.len() >= max_frames {
                    break;
                }
            }
            Ok(None) => break,
            Err(e) => {
                eprintln!("decode error: {}", e);
                break;
            }
        }
    }
    // Drain frames still held back by B-frame reordering.
    if frames.len() < max_frames {
        match decoder.flush() {
            Ok(mut pending) => {
                frames.append(&mut pending);
                frames.truncate(max_frames);
            }
            Err(e) => eprintln!("flush error: {}", e),
        }
    }
    frames
}

fn main() {
    let args = parse_args();

    if !std::path::Path::new(&args.input).exists() {
        die(&format!("file not found: {}", args.input));
    }
    if args.backend == Backend::Nvdec && !nvdec_decode::is_available() {
        die("NVDEC not available on this system (NVIDIA GPU + CUDA driver required)");
    }

    let data = std::fs::read(&args.input).unwrap_or_else(|e| die(&format!("failed to read {}: {}", args.input, e)));

    let ivf = parse_ivf(&data);
    let codec = match &ivf {
        Some(ivf) => ivf.codec,
        None => detect_codec(&data, &args.input),
    };
    if args.backend == Backend::Vaapi && codec == Codec::Av1 {
        die("vaapi backend does not support AV1 (use vulkan or nvdec)");
    }

    let pts_table = ivf.as_ref().map(|i| ivf_display_pts(i)).unwrap_or_default();

    match &ivf {
        Some(ivf) => println!(
            "backend={} file={} codec={} size={}x{} timebase={}/{}s container_frames={}",
            args.backend,
            args.input,
            codec.name(),
            ivf.width,
            ivf.height,
            ivf.tb_num,
            ivf.tb_den,
            ivf.declared_frames
        ),
        None => println!(
            "backend={} file={} codec={} timebase=1/30s (assumed; raw bitstream has no container pts)",
            args.backend, args.input, codec.name()
        ),
    }

    let start = Instant::now();
    let frames: Vec<Frame> = match args.backend {
        Backend::Vulkan => {
            let mut decoder =
                vulkan_decode::VulkanDecoder::new(data).unwrap_or_else(|e| die(&format!("vulkan decoder init: {}", e)));
            let frames = decoder
                .decode_all(args.max_frames)
                .unwrap_or_else(|e| die(&format!("vulkan decode: {}", e)));
            vulkan_decode::VulkanDecoder::reorder_to_presentation(frames)
                .into_iter()
                .map(Frame::Vk)
                .collect()
        }
        Backend::Vaapi => {
            let mut decoder =
                vaapi_decode::VaapiDecoder::new(data).unwrap_or_else(|e| die(&format!("vaapi decoder init: {}", e)));
            decode_all_core(&mut decoder, args.max_frames)
                .into_iter()
                .map(Frame::Core)
                .collect()
        }
        Backend::Nvdec => {
            let frames = match codec {
                Codec::H264 => {
                    let mut d = nvdec_decode::NvdecH264Decoder::new(data).unwrap_or_else(|e| die(&format!("nvdec init: {}", e)));
                    decode_all_core(&mut d, args.max_frames)
                }
                Codec::H265 => {
                    let mut d = nvdec_decode::NvdecH265Decoder::new(data).unwrap_or_else(|e| die(&format!("nvdec init: {}", e)));
                    decode_all_core(&mut d, args.max_frames)
                }
                Codec::Vp9 => {
                    let mut d = nvdec_decode::NvdecVp9Decoder::new(data).unwrap_or_else(|e| die(&format!("nvdec init: {}", e)));
                    decode_all_core(&mut d, args.max_frames)
                }
                Codec::Av1 => {
                    let mut d = nvdec_decode::NvdecAv1Decoder::new(data).unwrap_or_else(|e| die(&format!("nvdec init: {}", e)));
                    decode_all_core(&mut d, args.max_frames)
                }
            };
            frames.into_iter().map(Frame::Core).collect()
        }
    };

    if frames.is_empty() {
        die("no frames decoded");
    }

    let mut pts_warned = false;
    for (i, frame) in frames.iter().enumerate() {
        let (w, h) = frame.size();
        let pts_ms: f64 = match &ivf {
            Some(ivf) => match pts_table.get(i) {
                Some(&ticks) => ticks as f64 * ivf.tb_num as f64 * 1000.0 / ivf.tb_den as f64,
                None => {
                    if !pts_warned {
                        eprintln!(
                            "warning: {} decoded frames but only {} container display pts; remaining pts set to -1",
                            frames.len(),
                            pts_table.len()
                        );
                        pts_warned = true;
                    }
                    -1.0
                }
            },
            None => i as f64 * 1000.0 / 30.0, // synthetic: frame index on assumed 30 fps
        };

        match frame.canonical_pixels() {
            Some(pixels) => println!(
                "frame {}: pts={}ms size={}x{} hash={:016x}",
                i,
                fmt_ms(pts_ms),
                w,
                h,
                fnv1a64(&pixels)
            ),
            None => println!("frame {}: pts={}ms size={}x{} hash=- (no pixel data)", i, fmt_ms(pts_ms), w, h),
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "decoded {} frames in {:.2}s ({:.1} fps)",
        frames.len(),
        elapsed,
        frames.len() as f64 / elapsed.max(1e-9)
    );
}

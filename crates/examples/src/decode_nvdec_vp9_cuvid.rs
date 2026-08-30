//! NVDEC VP9 decode via NVIDIA's own cuvid parser — baseline parameter dump.
//!
//! Iteration 1 baseline: decodes a VP9 IVF file using NVIDIA's cuvid parser
//! (libnvcuvid.so) and dumps EVERY parameter struct the parser produces:
//!
//! - `CUVIDEOFORMAT` (sequence callback)
//! - `CUVIDPICPARAMS` + `CUVIDVP9PICPARAMS` (decode callback)
//! - `CUVIDPARSERDISPINFO` (display callback)
//!
//! Decoded NV12 frames are extracted (cuvidMapVideoFrame64 + cuMemcpyDtoH),
//! deinterleaved to planar YUV420P, and written as `<out_prefix>_frame_<j>.yuv`
//! for byte-exact comparison against ffmpeg.
//!
//! In parallel, the same IVF is parsed with the Rust `vk-video-parser`
//! (Vp9Parser) and its `Vp9FrameData` output is dumped to a second file for
//! side-by-side comparison.
//!
//! Dumps:
//!   /tmp/pixel_verify/vp9_cuvid_params.txt   (cuvid side)
//!   /tmp/pixel_verify/vp9_vkparser_params.txt (vk-video-parser side)
//!
//! Usage:
//!   cargo run --release --example decode_nvdec_vp9_cuvid -- \
//!     [ivf_file] [max_frames] [out_prefix]
//!   defaults: assets/big_buck_bunney_vp9.ivf 300 nvdec_vp9_cuvid

use std::ffi::c_void;
use std::fs::File;
use std::io::Write;
use std::os::raw::c_int;
use std::path::Path;

use nvdec_decode::device::{cu_ctx_synchronize, cu_ctx_set_current, cu_memcpy_dtoh, get_funcs};
use nvdec_decode::ffi::{
    cudaVideoCodec, cudaVideoDeinterlaceMode, cudaVideoSurfaceFormat, CUVIDPARSERDISPINFO,
    CUVIDPARSERPARAMS, CUVIDPICPARAMS, CUVIDPROCPARAMS, CUVIDRECT, CUVIDSOURCEDATAPACKET,
    CUVIDEOFORMAT, CUvideodecoder, CUvideopacketflags, CUvideoparser, CUDA_SUCCESS,
};
use vk_video_core::picture::Vp9FrameData;
use vk_video_core::VideoCodec;
use vk_video_parser::vp9::Vp9Parser;
use vk_video_parser::{DetectedVideoFormat, VideoParser};

/// Number of decode surfaces (MUST match ulMaxNumDecodeSurfaces of the parser).
const NUM_SURFACES: u32 = 16;
const CUVID_DUMP: &str = "/tmp/pixel_verify/vp9_cuvid_params.txt";
const VK_DUMP: &str = "/tmp/pixel_verify/vp9_vkparser_params.txt";

// ============================================================================
// IVF container parsing (copied from vulkan_decode_vp9.rs, extended with pts)
// ============================================================================

/// IVF file: header info + packets as (payload, pts).
struct IvfFile {
    width: u16,
    height: u16,
    timebase_rate: u32,
    timebase_scale: u32,
    packets: Vec<(Vec<u8>, u64)>,
}

/// Parse an IVF container into raw VP9 packets, keeping the u64 pts per packet.
///
/// Standard IVF header (32 bytes):
///   magic(4) version(4) header_size(4) codec(4) width(2) height(2)
///   timebase_rate(4) timebase_scale(4) length(4)
/// Some writers omit header_size (codec tag at offset 8, 4 reserved bytes at
/// the end). Both layouts are 32 bytes; packets start at offset 32 either way.
/// The codec tag is NOT used for filtering — payloads are fed as-is.
fn parse_ivf_container(data: &[u8]) -> Result<IvfFile, String> {
    if data.len() < 32 {
        return Err("File too small for IVF header".to_string());
    }
    if data[0..4] != *b"DKIF" {
        return Err("Invalid IVF magic".to_string());
    }

    let (wh_off, tb_off) = if data[12..16] == *b"VP90" || data[12..16] == *b"VP9 " {
        (16usize, 20usize) // standard layout
    } else if data[8..12] == *b"VP90" || data[8..12] == *b"VP9 " {
        (12usize, 16usize) // header_size omitted
    } else {
        (16usize, 20usize) // default to standard
    };
    let width = u16::from_le_bytes([data[wh_off], data[wh_off + 1]]);
    let height = u16::from_le_bytes([data[wh_off + 2], data[wh_off + 3]]);
    let timebase_rate = u32::from_le_bytes(data[tb_off..tb_off + 4].try_into().unwrap());
    let timebase_scale = u32::from_le_bytes(data[tb_off + 4..tb_off + 8].try_into().unwrap());

    let mut packets = Vec::new();
    let mut offset = 32usize;
    while offset < data.len() {
        // Need at least 12 bytes for packet header (4 size + 8 pts)
        if offset + 12 > data.len() {
            break;
        }
        let packet_size = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        let pts = u64::from_le_bytes(data[offset + 4..offset + 12].try_into().unwrap());
        offset += 12;

        if packet_size == 0 || offset + packet_size > data.len() {
            eprintln!(
                "Warning: invalid packet size {} at offset {}",
                packet_size,
                offset - 12
            );
            break;
        }
        packets.push((data[offset..offset + packet_size].to_vec(), pts));
        offset += packet_size;
    }

    if packets.is_empty() {
        return Err("No frames found in IVF container".to_string());
    }
    Ok(IvfFile {
        width,
        height,
        timebase_rate,
        timebase_scale,
        packets,
    })
}

// ============================================================================
// VP9 superframe expansion (copied from vulkan_decode_vp9.rs)
// ============================================================================

/// Frame data with superframe information.
#[derive(Clone)]
struct FrameInfo {
    /// The frame data (extracted from superframe if applicable)
    data: Vec<u8>,
    /// IVF presentation timestamp of the containing packet
    pts: u64,
    /// Offset of this frame within a superframe (0 if not from superframe)
    superframe_frame_offset: usize,
}

/// Expand superframes into individual frames while tracking packet pts.
///
/// A superframe contains multiple VP9 frames concatenated together, with a
/// superframe index at the end specifying the size of each constituent frame.
fn expand_superframes(packets: &[(Vec<u8>, u64)]) -> Vec<FrameInfo> {
    let mut expanded = Vec::new();

    for (frame, pts) in packets.iter() {
        let data_len = frame.len();
        if data_len < 2 {
            expanded.push(FrameInfo {
                data: frame.clone(),
                pts: *pts,
                superframe_frame_offset: 0,
            });
            continue;
        }

        // Check for superframe index at the end of the data
        let final_byte = frame[data_len - 1];
        if (final_byte & 0xE0) != 0xC0 {
            // Not a superframe
            expanded.push(FrameInfo {
                data: frame.clone(),
                pts: *pts,
                superframe_frame_offset: 0,
            });
            continue;
        }

        let num_frames = (final_byte & 0x07) as usize + 1;
        if num_frames <= 1 {
            expanded.push(FrameInfo {
                data: frame.clone(),
                pts: *pts,
                superframe_frame_offset: 0,
            });
            continue;
        }

        let mag = (((final_byte >> 3) & 0x03) as usize) + 1;
        let index_size = 2 + mag * num_frames;

        if data_len < index_size {
            expanded.push(FrameInfo {
                data: frame.clone(),
                pts: *pts,
                superframe_frame_offset: 0,
            });
            continue;
        }

        let index_start = data_len - index_size;
        if frame[index_start] != final_byte {
            expanded.push(FrameInfo {
                data: frame.clone(),
                pts: *pts,
                superframe_frame_offset: 0,
            });
            continue;
        }

        // Parse frame sizes from the superframe index
        let frame_data_size = data_len - index_size;
        let mut offset = 0;
        let mut x = index_start + 1;
        for _i in 0..num_frames {
            let mut this_sz: usize = 0;
            for j in 0..mag {
                this_sz |= (frame[x + j] as usize) << (j * 8);
            }
            x += mag;

            if offset + this_sz <= frame_data_size {
                expanded.push(FrameInfo {
                    data: frame[offset..offset + this_sz].to_vec(),
                    pts: *pts,
                    superframe_frame_offset: offset,
                });
            }
            offset += this_sz;
        }
    }

    expanded
}

// ============================================================================
// cuvid parser state + callbacks
// ============================================================================

/// State passed to the cuvid parser callbacks via pUserData.
struct State {
    decoder: Option<CUvideodecoder>,
    dump: Option<std::io::BufWriter<File>>,
    sequence_count: u32,
    decode_count: u32,
    decode_with_data: u32,
    decode_picture_ok: u32,
    decode_picture_fail: u32,
    display_count: u32,
    frames_written: u32,
    out_prefix: String,
    coded_width: u32,
    coded_height: u32,
    /// CurrPicIdx of every frame actually submitted to cuvidDecodePicture.
    /// Used to detect show_existing_frame re-displays in the display callback.
    decoded_surfaces: Vec<u32>,
}

/// Sequence callback: dump CUVIDEOFORMAT, create the decoder.
unsafe extern "C" fn sequence_callback(
    pUserData: *mut c_void,
    pVideoFormat: *mut CUVIDEOFORMAT,
) -> c_int {
    let state = &mut *(pUserData as *mut State);
    state.sequence_count += 1;
    let seq = state.sequence_count - 1;
    let f = &*pVideoFormat;

    let dump_block = |dump: &mut std::io::BufWriter<File>| -> std::io::Result<()> {
        writeln!(dump, "=== SEQUENCE {} ===", seq)?;
        writeln!(dump, "codec = {:?}", f.codec)?;
        writeln!(
            dump,
            "frame_rate = {}/{}",
            f.frame_rate.numerator, f.frame_rate.denominator
        )?;
        writeln!(dump, "progressive_sequence = {}", f.progressive_sequence)?;
        writeln!(dump, "bit_depth_luma_minus8 = {}", f.bit_depth_luma_minus8)?;
        writeln!(dump, "bit_depth_chroma_minus8 = {}", f.bit_depth_chroma_minus8)?;
        writeln!(dump, "min_num_decode_surfaces = {}", f.min_num_decode_surfaces)?;
        writeln!(dump, "coded_width = {}", f.coded_width)?;
        writeln!(dump, "coded_height = {}", f.coded_height)?;
        writeln!(
            dump,
            "display_area = [{}, {}, {}, {}]",
            f.display_area.left, f.display_area.top, f.display_area.right, f.display_area.bottom
        )?;
        writeln!(dump, "chroma_format = {:?}", f.chroma_format)?;
        writeln!(dump, "bitrate = {}", f.bitrate)?;
        writeln!(
            dump,
            "display_aspect_ratio = {}/{}",
            f.display_aspect_ratio.x, f.display_aspect_ratio.y
        )?;
        let vsd = &f.video_signal_description;
        writeln!(dump, "video_signal_description.video_format = {}", vsd.video_format)?;
        writeln!(
            dump,
            "video_signal_description.video_full_range_flag = {}",
            vsd.video_full_range_flag
        )?;
        writeln!(
            dump,
            "video_signal_description.reserved_zero_bits = {}",
            vsd.reserved_zero_bits
        )?;
        writeln!(
            dump,
            "video_signal_description.color_primaries = {}",
            vsd.color_primaries
        )?;
        writeln!(
            dump,
            "video_signal_description.transfer_characteristics = {}",
            vsd.transfer_characteristics
        )?;
        writeln!(
            dump,
            "video_signal_description.matrix_coefficients = {}",
            vsd.matrix_coefficients
        )?;
        writeln!(dump, "seqhdr_data_length = {}", f.seqhdr_data_length)?;
        Ok(())
    };

    println!(
        "[sequence {}] codec={:?} {}x{} chroma={:?} bitdepth={}+{} min_surfaces={} progressive={}",
        seq,
        f.codec,
        f.coded_width,
        f.coded_height,
        f.chroma_format,
        f.bit_depth_luma_minus8,
        f.bit_depth_chroma_minus8,
        f.min_num_decode_surfaces,
        f.progressive_sequence
    );
    if let Some(dump) = &mut state.dump {
        if let Err(e) = dump_block(dump) {
            eprintln!("[sequence] dump error: {}", e);
        }
    }

    if state.decoder.is_some() {
        println!("[sequence] decoder already exists, skipping re-create");
        return 1;
    }

    let funcs = match get_funcs() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[sequence] get_funcs failed: {}", e);
            return 0;
        }
    };

    let cw = f.coded_width;
    let ch = f.coded_height;
    let create_info = nvdec_decode::ffi::CUVIDDECODECREATEINFO {
        ulWidth: cw as u64,
        ulHeight: ch as u64,
        ulNumDecodeSurfaces: NUM_SURFACES as u64, // MUST equal parser ulMaxNumDecodeSurfaces
        CodecType: cudaVideoCodec::cudaVideoCodec_VP9,
        ChromaFormat: f.chroma_format,
        ulCreationFlags: 0,
        bitDepthMinus8: f.bit_depth_luma_minus8 as u64,
        ulIntraDecodeOnly: 0,
        ulMaxWidth: cw as u64,
        ulMaxHeight: ch as u64,
        Reserved1: 0,
        display_area: CUVIDRECT {
            left: 0,
            top: 0,
            right: cw as i16,
            bottom: ch as i16,
        },
        OutputFormat: cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12,
        DeinterlaceMode: cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Weave,
        ulTargetWidth: cw as u64,
        ulTargetHeight: ch as u64,
        ulNumOutputSurfaces: NUM_SURFACES as u64,
        vidLock: std::ptr::null_mut(),
        target_rect: CUVIDRECT {
            left: 0,
            top: 0,
            right: cw as i16,
            bottom: ch as i16,
        },
        enableHistogram: 0,
        Reserved2: [0; 4],
    };

    let mut decoder: CUvideodecoder = std::ptr::null_mut();
    let res = unsafe { (funcs.create_decoder)(&mut decoder, &create_info) };
    println!(
        "[sequence] cuvidCreateDecoder result = {} {}",
        res,
        if res == CUDA_SUCCESS {
            "(CUDA_SUCCESS)"
        } else {
            "(FAILED)"
        }
    );
    if res == CUDA_SUCCESS {
        state.decoder = Some(decoder);
        state.coded_width = cw;
        state.coded_height = ch;
        // Return value semantics: 0=fail, 1=succeeded, >1=override parser DPB size.
        return 1;
    }
    eprintln!("[sequence] ERROR: cuvidCreateDecoder failed with {}", res);
    0
}

/// Decode callback: dump CUVIDPICPARAMS + CUVIDVP9PICPARAMS, submit decode.
unsafe extern "C" fn decode_callback(
    pUserData: *mut c_void,
    pPicParams: *mut CUVIDPICPARAMS,
) -> c_int {
    let state = &mut *(pUserData as *mut State);
    state.decode_count += 1;
    let i = state.decode_count - 1;
    let pp = &*pPicParams;
    // The codec-specific union part (CUVIDVP9PICPARAMS, 220 bytes — the ffi
    // struct layout matches the SDK header exactly, bitfields included).
    let vp9 = &pp.CodecSpecific.vp9;

    if pp.nBitstreamDataLen == 0 {
        println!(
            "[decode {}] nBitstreamDataLen=0 (show_existing_frame?), CurrPicIdx={}",
            i, pp.CurrPicIdx
        );
        if let Some(dump) = &mut state.dump {
            let _ = writeln!(
                dump,
                "=== DECODE {} === (nBitstreamDataLen=0, skipped — likely show_existing_frame; CurrPicIdx={})",
                i, pp.CurrPicIdx
            );
        }
        // Nothing to decode; report success so the parser still schedules display.
        return 1;
    }

    state.decode_with_data += 1;

    if let Some(dump) = &mut state.dump {
        let _ = writeln!(dump, "=== DECODE {} ===", i);
        let _ = writeln!(dump, "-- CUVIDPICPARAMS (common) --");
        let _ = writeln!(dump, "PicWidthInMbs = {}", pp.PicWidthInMbs);
        let _ = writeln!(dump, "FrameHeightInMbs = {}", pp.FrameHeightInMbs);
        let _ = writeln!(dump, "CurrPicIdx = {}", pp.CurrPicIdx);
        let _ = writeln!(dump, "field_pic_flag = {}", pp.field_pic_flag);
        let _ = writeln!(dump, "bottom_field_flag = {}", pp.bottom_field_flag);
        let _ = writeln!(dump, "second_field = {}", pp.second_field);
        let _ = writeln!(dump, "nBitstreamDataLen = {}", pp.nBitstreamDataLen);
        let _ = writeln!(dump, "nNumSlices = {}", pp.nNumSlices);
        if pp.nNumSlices > 0 {
            let first = unsafe { *pp.pSliceDataOffsets };
            let _ = writeln!(dump, "pSliceDataOffsets[0] = {}", first);
        }
        let _ = writeln!(dump, "ref_pic_flag = {}", pp.ref_pic_flag);
        let _ = writeln!(dump, "intra_pic_flag = {}", pp.intra_pic_flag);
        let _ = writeln!(dump, "-- CUVIDVP9PICPARAMS --");
        let _ = writeln!(dump, "width = {}", vp9.width);
        let _ = writeln!(dump, "height = {}", vp9.height);
        let _ = writeln!(dump, "LastRefIdx = {}", vp9.LastRefIdx);
        let _ = writeln!(dump, "GoldenRefIdx = {}", vp9.GoldenRefIdx);
        let _ = writeln!(dump, "AltRefIdx = {}", vp9.AltRefIdx);
        let _ = writeln!(dump, "colorSpace = {}", vp9.colorSpace);
        let _ = writeln!(dump, "profile = {}", vp9.profile());
        let _ = writeln!(dump, "frameContextIdx = {}", vp9.frame_context_idx());
        let _ = writeln!(dump, "frameType = {}", vp9.frame_type());
        let _ = writeln!(dump, "showFrame = {}", vp9.show_frame());
        let _ = writeln!(dump, "errorResilient = {}", vp9.error_resilient());
        let _ = writeln!(dump, "frameParallelDecoding = {}", vp9.frame_parallel_decoding());
        let _ = writeln!(dump, "subSamplingX = {}", vp9.sub_sampling_x());
        let _ = writeln!(dump, "subSamplingY = {}", vp9.sub_sampling_y());
        let _ = writeln!(dump, "intraOnly = {}", vp9.intra_only());
        let _ = writeln!(dump, "allow_high_precision_mv = {}", vp9.allow_high_precision_mv());
        let _ = writeln!(dump, "refreshEntropyProbs = {}", vp9.refresh_entropy_probs());
        let _ = writeln!(
            dump,
            "refFrameSignBias = [{}]",
            vp9.refFrameSignBias
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(dump, "bitDepthMinus8Luma = {}", vp9.bitDepthMinus8Luma);
        let _ = writeln!(dump, "bitDepthMinus8Chroma = {}", vp9.bitDepthMinus8Chroma);
        let _ = writeln!(dump, "loopFilterLevel = {}", vp9.loopFilterLevel);
        let _ = writeln!(dump, "loopFilterSharpness = {}", vp9.loopFilterSharpness);
        let _ = writeln!(dump, "modeRefLfEnabled = {}", vp9.modeRefLfEnabled);
        let _ = writeln!(dump, "log2_tile_columns = {}", vp9.log2_tile_columns);
        let _ = writeln!(dump, "log2_tile_rows = {}", vp9.log2_tile_rows);
        let _ = writeln!(dump, "segmentEnabled = {}", vp9.segment_enabled());
        let _ = writeln!(dump, "segmentMapUpdate = {}", vp9.segment_map_update());
        let _ = writeln!(dump, "segmentMapTemporalUpdate = {}", vp9.segment_map_temporal_update());
        let _ = writeln!(dump, "segmentFeatureMode = {}", vp9.segment_feature_mode());
        let _ = writeln!(
            dump,
            "segmentFeatureEnable = [{}]",
            (0..8)
                .map(|r| {
                    format!(
                        "[{}]",
                        vp9.segmentFeatureEnable[r]
                            .iter()
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        );
        let _ = writeln!(
            dump,
            "segmentFeatureData = [{}]",
            (0..8)
                .map(|r| {
                    format!(
                        "[{}]",
                        vp9.segmentFeatureData[r]
                            .iter()
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        );
        let _ = writeln!(
            dump,
            "mb_segment_tree_probs = [{}]",
            vp9.mb_segment_tree_probs
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(
            dump,
            "segment_pred_probs = [{}]",
            vp9.segment_pred_probs
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(dump, "qpYAc = {}", vp9.qpYAc);
        let _ = writeln!(dump, "qpYDc = {}", vp9.qpYDc);
        let _ = writeln!(dump, "qpChDc = {}", vp9.qpChDc);
        let _ = writeln!(dump, "qpChAc = {}", vp9.qpChAc);
        let _ = writeln!(
            dump,
            "activeRefIdx = [{}]",
            vp9.activeRefIdx
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(dump, "resetFrameContext = {}", vp9.resetFrameContext);
        let _ = writeln!(dump, "mcomp_filter_type = {}", vp9.mcomp_filter_type);
        let _ = writeln!(
            dump,
            "mbRefLfDelta = [{}]",
            vp9.mbRefLfDelta
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(
            dump,
            "mbModeLfDelta = [{}]",
            vp9.mbModeLfDelta
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(dump, "frameTagSize = {}", vp9.frameTagSize);
        let _ = writeln!(dump, "offsetToDctParts = {}", vp9.offsetToDctParts);
    }

    let decoder = match state.decoder {
        Some(d) => d,
        None => {
            eprintln!(
                "[decode {}] WARNING: decoder not created yet, skipping cuvidDecodePicture",
                i
            );
            return 0;
        }
    };

    let funcs = match get_funcs() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[decode] get_funcs failed: {}", e);
            return 0;
        }
    };
    let procparams = nvdec_decode::ffi::default_procparams();
    let res = unsafe { (funcs.decode_picture)(decoder, pPicParams, &procparams) };
    if res == CUDA_SUCCESS {
        state.decode_picture_ok += 1;
        state.decoded_surfaces.push(pp.CurrPicIdx as u32);
        // Return value semantics: 0=fail, >=1=succeeded.
        1
    } else {
        state.decode_picture_fail += 1;
        eprintln!(
            "[decode {}] cuvidDecodePicture FAILED: {} (CurrPicIdx={})",
            i, res, pp.CurrPicIdx
        );
        0
    }
}

/// Display callback: dump CUVIDPARSERDISPINFO, extract NV12 -> YUV420P file.
unsafe extern "C" fn display_callback(
    pUserData: *mut c_void,
    pDispInfo: *mut CUVIDPARSERDISPINFO,
) -> c_int {
    let state = &mut *(pUserData as *mut State);
    state.display_count += 1;
    let j = state.display_count - 1;
    let info = &*pDispInfo;

    // A picture_index that was never submitted to cuvidDecodePicture means
    // this is a re-display of an already-decoded surface (show_existing_frame).
    let is_redisplay = !state
        .decoded_surfaces
        .contains(&(info.picture_index as u32));

    if let Some(dump) = &mut state.dump {
        let _ = writeln!(dump, "=== DISPLAY {} ===", j);
        let _ = writeln!(dump, "picture_index = {}", info.picture_index);
        let _ = writeln!(dump, "progressive_frame = {}", info.progressive_frame);
        let _ = writeln!(dump, "top_field_first = {}", info.top_field_first);
        let _ = writeln!(dump, "repeat_first_field = {}", info.repeat_first_field);
        let _ = writeln!(dump, "timestamp = {}", info.timestamp);
        if is_redisplay {
            let _ = writeln!(
                dump,
                "note: re-display of already-decoded surface (show_existing_frame)"
            );
        }
    }
    println!(
        "[display {}] picture_index={} timestamp={}{}",
        j,
        info.picture_index,
        info.timestamp,
        if is_redisplay { " (re-display)" } else { "" }
    );

    let decoder = match state.decoder {
        Some(d) => d,
        None => {
            eprintln!("[display {}] no decoder, skipping extraction", j);
            return 0;
        }
    };
    let funcs = match get_funcs() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[display] get_funcs failed: {}", e);
            return 0;
        }
    };

    // Ensure the decode is complete before mapping/reading the surface.
    let _ = cu_ctx_synchronize();

    let mut dev_ptr: u64 = 0;
    let mut pitch: u32 = 0;
    let mut proc = unsafe { std::mem::zeroed::<CUVIDPROCPARAMS>() };
    proc.progressive_frame = 1;
    let res = unsafe {
        (funcs.map_video_frame64)(
            decoder,
            info.picture_index,
            &mut dev_ptr,
            &mut pitch,
            &proc,
        )
    };
    if res != CUDA_SUCCESS {
        eprintln!(
            "[display {}] cuvidMapVideoFrame64 FAILED: {} (picture_index={})",
            j, res, info.picture_index
        );
        return 0;
    }

    let w = state.coded_width as usize;
    let h = state.coded_height as usize;
    let pitch = pitch as usize;
    // NV12 surface: Y plane (h rows) + interleaved UV plane (h/2 rows), each `pitch` wide.
    let surface_bytes = pitch * h * 3 / 2;
    let mut host: Vec<u8> = vec![0u8; surface_bytes];
    let copy_res = unsafe { cu_memcpy_dtoh(host.as_mut_ptr().cast(), dev_ptr, surface_bytes) };
    let unmap_res = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
    let copy_code = match copy_res {
        Ok(code) => code,
        Err(e) => {
            eprintln!("[display {}] cuMemcpyDtoH error: {}", j, e);
            let _ = unsafe { (funcs.unmap_video_frame64)(decoder, dev_ptr) };
            return 0;
        }
    };
    if copy_code != CUDA_SUCCESS || unmap_res != CUDA_SUCCESS {
        eprintln!(
            "[display {}] cuMemcpyDtoH/cuvidUnmapVideoFrame64 failed: {} / {}",
            j, copy_code, unmap_res
        );
        return 0;
    }

    // Deinterleave NV12 -> planar YUV420P (Y, U, V).
    let mut yuv = vec![0u8; w * h * 3 / 2];
    for row in 0..h {
        let s = row * pitch;
        yuv[row * w..row * w + w].copy_from_slice(&host[s..s + w]);
    }
    let uv_base = h * pitch;
    let half_w = w / 2;
    let half_h = h / 2;
    // Planar layout: Y [0, w*h), U [w*h, w*h + w*h/4), V [w*h + w*h/4, w*h*3/2)
    for row in 0..half_h {
        let s = uv_base + row * pitch;
        for x in 0..half_w {
            yuv[w * h + row * half_w + x] = host[s + 2 * x]; // U
            yuv[w * h + w * h / 4 + row * half_w + x] = host[s + 2 * x + 1]; // V
        }
    }

    let path = format!("{}_frame_{}.yuv", state.out_prefix, j);
    match std::fs::write(&path, &yuv) {
        Ok(()) => {
            state.frames_written += 1;
            // Return value semantics: 0=fail, >=1=succeeded.
            1
        }
        Err(e) => {
            eprintln!("[display {}] failed to write {}: {}", j, path, e);
            0
        }
    }
}

// ============================================================================
// vk-video-parser comparison dump
// ============================================================================

fn dump_vp9_frame_data(
    w: &mut impl Write,
    i: usize,
    fd: &Vp9FrameData,
    pts: u64,
) -> std::io::Result<()> {
    writeln!(w, "=== VPPARSER {} ===", i)?;
    writeln!(w, "pts = {}", pts)?;
    writeln!(w, "show_existing_frame = {}", fd.show_existing_frame)?;
    writeln!(w, "frame_to_show_map_idx = {}", fd.frame_to_show_map_idx)?;
    writeln!(w, "frame_is_intra = {}", fd.frame_is_intra)?;
    writeln!(w, "frame_width = {}", fd.frame_width)?;
    writeln!(w, "frame_height = {}", fd.frame_height)?;
    writeln!(w, "render_width = {}", fd.render_width)?;
    writeln!(w, "render_height = {}", fd.render_height)?;
    writeln!(w, "mi_cols = {}", fd.mi_cols)?;
    writeln!(w, "mi_rows = {}", fd.mi_rows)?;
    writeln!(w, "sb64_cols = {}", fd.sb64_cols)?;
    writeln!(w, "sb64_rows = {}", fd.sb64_rows)?;
    writeln!(w, "num_tiles = {}", fd.num_tiles)?;
    let pi = &fd.picture_info;
    writeln!(w, "picture_info.profile = {:?}", pi.profile)?;
    writeln!(w, "picture_info.frame_type = {:?}", pi.frame_type)?;
    writeln!(w, "picture_info.frame_context_idx = {}", pi.frame_context_idx)?;
    writeln!(w, "picture_info.refresh_frame_flags = {}", pi.refresh_frame_flags)?;
    writeln!(
        w,
        "picture_info.ref_frame_sign_bias_mask = {}",
        pi.ref_frame_sign_bias_mask
    )?;
    writeln!(
        w,
        "picture_info.interpolation_filter = {:?}",
        pi.interpolation_filter
    )?;
    writeln!(w, "picture_info.base_q_idx = {}", pi.base_q_idx)?;
    writeln!(w, "picture_info.delta_q_y_dc = {}", pi.delta_q_y_dc)?;
    writeln!(w, "picture_info.delta_q_uv_dc = {}", pi.delta_q_uv_dc)?;
    writeln!(w, "picture_info.delta_q_uv_ac = {}", pi.delta_q_uv_ac)?;
    writeln!(w, "picture_info.tile_cols_log2 = {}", pi.tile_cols_log2)?;
    writeln!(w, "picture_info.tile_rows_log2 = {}", pi.tile_rows_log2)?;
    let fl = &pi.flags;
    writeln!(
        w,
        "picture_info.flags.error_resilient_mode = {}",
        fl.error_resilient_mode
    )?;
    writeln!(w, "picture_info.flags.intra_only = {}", fl.intra_only)?;
    writeln!(
        w,
        "picture_info.flags.allow_high_precision_mv = {}",
        fl.allow_high_precision_mv
    )?;
    writeln!(
        w,
        "picture_info.flags.refresh_frame_context = {}",
        fl.refresh_frame_context
    )?;
    writeln!(
        w,
        "picture_info.flags.frame_parallel_decoding_mode = {}",
        fl.frame_parallel_decoding_mode
    )?;
    writeln!(
        w,
        "picture_info.flags.segmentation_enabled = {}",
        fl.segmentation_enabled
    )?;
    writeln!(w, "picture_info.flags.show_frame = {}", fl.show_frame)?;
    writeln!(
        w,
        "picture_info.flags.use_prev_frame_mvs = {}",
        fl.use_prev_frame_mvs
    )?;
    writeln!(
        w,
        "picture_info.flags.reset_frame_context = {}",
        fl.reset_frame_context
    )?;
    writeln!(w, "picture_info.lossless = {}", pi.lossless)?;
    let cc = &fd.color_config;
    writeln!(w, "color_config.flags.color_range = {}", cc.flags.color_range)?;
    writeln!(w, "color_config.bit_depth = {}", cc.bit_depth)?;
    writeln!(w, "color_config.subsampling_x = {}", cc.subsampling_x)?;
    writeln!(w, "color_config.subsampling_y = {}", cc.subsampling_y)?;
    writeln!(w, "color_config.color_space = {:?}", cc.color_space)?;
    let lf = &fd.loop_filter;
    writeln!(
        w,
        "loop_filter.flags.loop_filter_delta_enabled = {}",
        lf.flags.loop_filter_delta_enabled
    )?;
    writeln!(
        w,
        "loop_filter.flags.loop_filter_delta_update = {}",
        lf.flags.loop_filter_delta_update
    )?;
    writeln!(
        w,
        "loop_filter.flags.update_ref_delta = {}",
        lf.flags.update_ref_delta
    )?;
    writeln!(
        w,
        "loop_filter.flags.update_mode_delta = {}",
        lf.flags.update_mode_delta
    )?;
    writeln!(w, "loop_filter.loop_filter_level = {}", lf.loop_filter_level)?;
    writeln!(
        w,
        "loop_filter.loop_filter_sharpness = {}",
        lf.loop_filter_sharpness
    )?;
    writeln!(w, "loop_filter.update_ref_delta = {}", lf.update_ref_delta)?;
    writeln!(
        w,
        "loop_filter.loop_filter_ref_deltas = [{}]",
        lf.loop_filter_ref_deltas
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )?;
    writeln!(w, "loop_filter.update_mode_delta = {}", lf.update_mode_delta)?;
    writeln!(
        w,
        "loop_filter.loop_filter_mode_deltas = [{}]",
        lf.loop_filter_mode_deltas
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )?;
    let sg = &fd.segmentation;
    writeln!(
        w,
        "segmentation.flags.segmentation_update_map = {}",
        sg.flags.segmentation_update_map
    )?;
    writeln!(
        w,
        "segmentation.flags.segmentation_temporal_update = {}",
        sg.flags.segmentation_temporal_update
    )?;
    writeln!(
        w,
        "segmentation.flags.segmentation_update_data = {}",
        sg.flags.segmentation_update_data
    )?;
    writeln!(
        w,
        "segmentation.flags.segmentation_abs_or_delta_update = {}",
        sg.flags.segmentation_abs_or_delta_update
    )?;
    writeln!(
        w,
        "segmentation.segmentation_tree_probs = [{}]",
        sg.segmentation_tree_probs
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )?;
    writeln!(
        w,
        "segmentation.segmentation_pred_prob = [{}]",
        sg.segmentation_pred_prob
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )?;
    writeln!(
        w,
        "segmentation.feature_enabled = [{}]",
        sg.feature_enabled
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )?;
    writeln!(
        w,
        "segmentation.feature_data = [{}]",
        (0..8)
            .map(|r| {
                format!(
                    "[{}]",
                    sg.feature_data[r]
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    )?;
    writeln!(w, "compressed_header_size = {}", fd.compressed_header_size)?;
    writeln!(w, "uncompressed_header_size = {}", fd.uncompressed_header_size)?;
    writeln!(
        w,
        "uncompressed_header_offset = {}",
        fd.uncompressed_header_offset
    )?;
    writeln!(
        w,
        "compressed_header_offset = {}",
        fd.compressed_header_offset
    )?;
    writeln!(w, "tiles_offset = {}", fd.tiles_offset)?;
    writeln!(
        w,
        "ref_frame_idx = [{}]",
        fd.ref_frame_idx
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )?;
    writeln!(
        w,
        "pic_idx = [{}]",
        fd.pic_idx
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )?;
    writeln!(w, "superframe_frame_offset = {}", fd.superframe_frame_offset)?;
    Ok(())
}

/// Parse the same IVF with the Rust vk-video-parser and dump Vp9FrameData.
fn run_vkparser(ivf: &IvfFile, out_path: &Path) {
    let expanded = expand_superframes(&ivf.packets);
    println!(
        "[vkparser] {} packets -> {} expanded frames",
        ivf.packets.len(),
        expanded.len()
    );

    let mut parser = Vp9Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeVp9))
        .expect("Failed to init VP9 parser");

    let file = File::create(out_path).expect("failed to create vkparser dump file");
    let mut w = std::io::BufWriter::new(file);

    let mut ok = 0usize;
    let mut err = 0usize;
    for (i, frame) in expanded.iter().enumerate() {
        match parser.parse_frame_with_offset(&frame.data, frame.superframe_frame_offset as u32) {
            Ok(fd) => {
                ok += 1;
                if let Err(e) = dump_vp9_frame_data(&mut w, i, &fd, frame.pts) {
                    eprintln!("[vkparser] dump error at frame {}: {}", i, e);
                }
            }
            Err(e) => {
                err += 1;
                let _ = writeln!(w, "=== VPPARSER {} ===", i);
                let _ = writeln!(w, "PARSE ERROR: {}", e);
            }
        }
    }
    let _ = w.flush();
    println!(
        "[vkparser] parsed {} frames ({} errors) -> {}",
        ok,
        err,
        out_path.display()
    );
}

fn print_first_lines(path: &Path, n: usize) {
    match std::fs::read_to_string(path) {
        Ok(content) => {
            println!("--- first {} lines of {} ---", n, path.display());
            for line in content.lines().take(n) {
                println!("  {}", line);
            }
        }
        Err(e) => eprintln!("failed to read {}: {}", path.display(), e),
    }
}

// ============================================================================
// main
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ivf_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "assets/big_buck_bunney_vp9.ivf".to_string());
    let max_frames: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(300);
    let out_prefix = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "nvdec_vp9_cuvid".to_string());

    if !std::path::Path::new(&ivf_path).exists() {
        eprintln!("Error: File not found: {}", ivf_path);
        std::process::exit(1);
    }

    println!("=== NVDEC VP9 cuvid-parser baseline ===");
    println!("File: {}", ivf_path);
    println!(
        "max_frames (packets to feed): {}, out_prefix: {}",
        max_frames, out_prefix
    );

    // Step 1: init NVDEC
    if let Err(e) = nvdec_decode::init_nvdec() {
        eprintln!("Error: NVDEC init failed: {}", e);
        std::process::exit(1);
    }
    if !nvdec_decode::is_available() {
        eprintln!("Error: NVDEC not available on this system");
        std::process::exit(1);
    }
    let _ = cu_ctx_set_current();
    println!("NVDEC initialized, context set current");

    // Step 2: read + parse IVF
    let data = std::fs::read(&ivf_path).expect("Failed to read file");
    let ivf = parse_ivf_container(&data).expect("Failed to parse IVF container");
    println!(
        "IVF: {}x{} timebase={}/{} packets={}",
        ivf.width,
        ivf.height,
        ivf.timebase_rate,
        ivf.timebase_scale,
        ivf.packets.len()
    );

    // Step 3: prepare dump dir + state
    std::fs::create_dir_all("/tmp/pixel_verify").expect("failed to create /tmp/pixel_verify");
    let cuvid_dump = File::create(CUVID_DUMP).expect("failed to create cuvid dump file");

    let state = Box::new(State {
        decoder: None,
        dump: Some(std::io::BufWriter::new(cuvid_dump)),
        sequence_count: 0,
        decode_count: 0,
        decode_with_data: 0,
        decode_picture_ok: 0,
        decode_picture_fail: 0,
        display_count: 0,
        frames_written: 0,
        out_prefix: out_prefix.clone(),
        coded_width: 0,
        coded_height: 0,
        decoded_surfaces: Vec::new(),
    });
    let state_ptr = Box::into_raw(state) as *mut c_void;

    // Step 4: create the cuvid parser
    let parser_params = CUVIDPARSERPARAMS {
        CodecType: cudaVideoCodec::cudaVideoCodec_VP9,
        ulMaxNumDecodeSurfaces: NUM_SURFACES,
        ulClockRate: 90000,
        ulErrorThreshold: 0,
        ulMaxDisplayDelay: 1,
        bAnnexb_and_reserved: 0, // raw IVF payloads, NOT annexb
        uReserved1: [0; 4],
        pUserData: state_ptr,
        pfnSequenceCallback: Some(sequence_callback),
        pfnDecodePicture: Some(decode_callback),
        pfnDisplayPicture: Some(display_callback),
        pfnGetOperatingPoint: std::ptr::null_mut(),
        pfnGetSEIMsg: std::ptr::null_mut(),
        pvReserved2: [std::ptr::null_mut(); 5],
        pExtVideoInfo: std::ptr::null_mut(),
    };

    let funcs = get_funcs().expect("get_funcs failed after init");
    let mut parser: CUvideoparser = std::ptr::null_mut();
    let res = unsafe { (funcs.create_video_parser)(&mut parser, &parser_params) };
    if res != CUDA_SUCCESS {
        eprintln!("Error: cuvidCreateVideoParser failed with {}", res);
        std::process::exit(1);
    }
    println!("cuvidCreateVideoParser: OK");

    // Step 5: feed packets (raw payloads — the cuvid parser handles
    // superframes internally)
    let n_packets = ivf.packets.len().min(max_frames);
    for (idx, (payload, pts)) in ivf.packets.iter().take(n_packets).enumerate() {
        let ts = (*pts as u64)
            * 90000
            * ivf.timebase_scale as u64
            / ivf.timebase_rate as u64;
        let packet = CUVIDSOURCEDATAPACKET {
            flags: CUvideopacketflags::CUVID_PKT_TIMESTAMP as u64,
            payload_size: payload.len() as u64,
            payload: payload.as_ptr(),
            timestamp: ts as i64,
        };
        let res = unsafe { (funcs.parse_video_data)(parser, &packet) };
        if res != CUDA_SUCCESS {
            eprintln!("Error: cuvidParseVideoData failed on packet {} with {}", idx, res);
            break;
        }
    }
    println!("fed {} packets to cuvid parser", n_packets);

    // Step 6: end of stream
    let eos = CUVIDSOURCEDATAPACKET {
        flags: CUvideopacketflags::CUVID_PKT_ENDOFSTREAM as u64,
        payload_size: 0,
        payload: std::ptr::null(),
        timestamp: -1,
    };
    let res = unsafe { (funcs.parse_video_data)(parser, &eos) };
    println!("cuvidParseVideoData(EOS): {}", res);

    // Step 7: destroy parser, then decoder
    let res = unsafe { (funcs.destroy_video_parser)(parser) };
    println!("cuvidDestroyVideoParser: {}", res);

    let mut state = unsafe { Box::from_raw(state_ptr as *mut State) };
    if let Some(decoder) = state.decoder {
        let res = unsafe { (funcs.destroy_decoder)(decoder) };
        println!("cuvidDestroyDecoder: {}", res);
    } else {
        eprintln!("WARNING: decoder was never created (sequence callback?)");
    }
    // Flush + close the dump file before reading it back.
    if let Some(mut dump) = state.dump.take() {
        let _ = dump.flush();
    }
    let summary = (
        state.sequence_count,
        state.decode_count,
        state.decode_with_data,
        state.decode_picture_ok,
        state.decode_picture_fail,
        state.display_count,
        state.frames_written,
    );
    drop(state);

    // Step 8: vk-video-parser comparison
    run_vkparser(&ivf, Path::new(VK_DUMP));

    // Step 9: summary
    let (
        sequence_count,
        decode_count,
        decode_with_data,
        decode_picture_ok,
        decode_picture_fail,
        display_count,
        frames_written,
    ) = summary;
    println!("\n=== SUMMARY ===");
    println!("sequence callbacks:  {}", sequence_count);
    println!("decode callbacks:    {}", decode_count);
    println!("  with bitstream:    {}", decode_with_data);
    println!("  cuvidDecodePicture: {} ok / {} fail", decode_picture_ok, decode_picture_fail);
    println!("display callbacks:   {}", display_count);
    println!("frames written:      {} ({}_frame_*.yuv)", frames_written, out_prefix);
    println!();
    print_first_lines(Path::new(CUVID_DUMP), 10);
    println!();
    print_first_lines(Path::new(VK_DUMP), 10);
}

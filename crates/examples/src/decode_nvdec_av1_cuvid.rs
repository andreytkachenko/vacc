//! NVDEC AV1 decode via NVIDIA's own cuvid parser — baseline parameter dump.
//!
//! Decodes an AV1 IVF file using NVIDIA's cuvid parser (libnvcuvid.so) and
//! dumps EVERY parameter struct the parser produces:
//!
//! - `CUVIDEOFORMAT` (sequence callback)
//! - `CUVIDPICPARAMS` + `CUVIDAV1PICPARAMS` (decode callback)
//! - `CUVIDPARSERDISPINFO` (display callback)
//!
//! Decoded NV12 frames are extracted (cuvidMapVideoFrame64 + cuMemcpyDtoH),
//! deinterleaved to planar YUV420P, and written as `<out_prefix>_frame_<j>.yuv`
//! for byte-exact comparison against ffmpeg.
//!
//! In parallel, the same IVF is parsed with the Rust `vk-video-parser`
//! (Av1Parser) and its `Av1FrameHeader` output is dumped to a second file for
//! side-by-side comparison.
//!
//! Dumps:
//!   /tmp/pixel_verify/av1_cuvid_params.txt   (cuvid side)
//!   /tmp/pixel_verify/av1_vkparser_params.txt (vk-video-parser side)
//!
//! Usage:
//!   cargo run --release --example decode_nvdec_av1 -- \
//!     [ivf_file] [max_frames] [out_prefix]
//!   defaults: assets/big_buck_bunny_av1.ivf 300 nvdec_av1_cuvid

use std::ffi::c_void;
use std::fs::File;
use std::io::Write;
use std::os::raw::c_int;
use std::path::Path;

use nvdec_decode::device::{cu_ctx_synchronize, cu_ctx_set_current, cu_memcpy_dtoh, get_funcs};
use nvdec_decode::ffi::{
    cudaVideoCodec, cudaVideoDeinterlaceMode, cudaVideoSurfaceFormat, CUVIDAV1PICPARAMS,
    CUVIDPARSERDISPINFO, CUVIDPARSERPARAMS, CUVIDPICPARAMS, CUVIDPROCPARAMS, CUVIDRECT,
    CUVIDSOURCEDATAPACKET, CUVIDEOFORMAT, CUvideodecoder, CUvideopacketflags, CUvideoparser,
    CUDA_SUCCESS,
};
use vk_video_core::VideoCodec;
use vk_video_parser::av1::{Av1FrameHeader, Av1Parser};
use vk_video_parser::{DetectedVideoFormat, VideoParser};

/// Number of decode surfaces (MUST match ulMaxNumDecodeSurfaces of the parser).
const NUM_SURFACES: u32 = 16;
const CUVID_DUMP: &str = "/tmp/pixel_verify/av1_cuvid_params.txt";
const VK_DUMP: &str = "/tmp/pixel_verify/av1_vkparser_params.txt";

// ============================================================================
// IVF container parsing (same layout as the VP9 example; codec tag is not
// used for filtering — payloads are fed as-is)
// ============================================================================

/// IVF file: header info + packets as (payload, pts).
struct IvfFile {
    width: u16,
    height: u16,
    timebase_rate: u32,
    timebase_scale: u32,
    packets: Vec<(Vec<u8>, u64)>,
}

/// Parse an IVF container into raw packets, keeping the u64 pts per packet.
fn parse_ivf_container(data: &[u8]) -> Result<IvfFile, String> {
    if data.len() < 32 {
        return Err("File too small for IVF header".to_string());
    }
    if data[0..4] != *b"DKIF" {
        return Err("Invalid IVF magic".to_string());
    }

    let (wh_off, tb_off) = if data[12..16] == *b"VP90" || data[12..16] == *b"VP9 " {
        (16usize, 20usize) // standard layout
    } else if data[8..12] == *b"VP90"
        || data[8..12] == *b"VP9 "
        || data[8..12] == *b"AV01"
    {
        (12usize, 16usize) // legacy layout: version(2) + header_size(2), codec at 8
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
// OBU walking (for the vk-video-parser comparison pass)
//
// OBU header: bit7=forbidden, bits6-2=type, bit2=ext, bit1=has_size, bit0=res.
// OBU numbering as used by this stream / NVIDIA parser: 1=SequenceHeader,
// 2=TemporalDelimiter, 3=FrameHeader, 4=TileGroup, 5=Metadata, 6=Frame,
// 8=TileList, 15=Padding.
// ============================================================================

/// Walk all OBUs in a packet, returning (obu_type, payload) pairs.
fn walk_obus<'a>(data: &'a [u8]) -> Vec<(u8, &'a [u8])> {
    let mut out = Vec::new();
    let mut o = 0usize;
    while o < data.len() {
        let b = data[o];
        if b & 0x01 != 0 {
            // reserved bit set — not a valid OBU header; stop
            break;
        }
        let typ = (b >> 3) & 0x0F;
        let ext = (b >> 2) & 1 == 1;
        let has_size = (b >> 1) & 1 == 1;
        o += 1;
        if ext {
            o = o.saturating_add(1); // temporal_id(2) + spatial_id(3) + reserved(3)
        }
        let mut size: usize = 0;
        if has_size {
            for _ in 0..4 {
                if o >= data.len() {
                    size = 0;
                    break;
                }
                let c = data[o] as usize;
                o += 1;
                size = (size << 7) | (c & 0x7F);
                if c & 0x80 == 0 {
                    break;
                }
            }
        }
        // Payload starts AFTER the size varint.
        let payload_start = o;
        if !has_size {
            size = data.len() - payload_start;
        }
        if payload_start + size > data.len() {
            size = data.len().saturating_sub(payload_start);
        }
        out.push((typ, &data[payload_start..payload_start + size]));
        o = payload_start + size;
    }
    out
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

    if let Some(dump) = &mut state.dump {
        let _ = writeln!(dump, "=== SEQUENCE {} ===", seq);
        let _ = writeln!(dump, "codec = {:?}", f.codec);
        let _ = writeln!(
            dump,
            "frame_rate = {}/{}",
            f.frame_rate.numerator, f.frame_rate.denominator
        );
        let _ = writeln!(dump, "progressive_sequence = {}", f.progressive_sequence);
        let _ = writeln!(dump, "bit_depth_luma_minus8 = {}", f.bit_depth_luma_minus8);
        let _ = writeln!(dump, "bit_depth_chroma_minus8 = {}", f.bit_depth_chroma_minus8);
        let _ = writeln!(dump, "min_num_decode_surfaces = {}", f.min_num_decode_surfaces);
        let _ = writeln!(dump, "coded_width = {}", f.coded_width);
        let _ = writeln!(dump, "coded_height = {}", f.coded_height);
        let _ = writeln!(
            dump,
            "display_area = [{}, {}, {}, {}]",
            f.display_area.left, f.display_area.top, f.display_area.right, f.display_area.bottom
        );
        let _ = writeln!(dump, "chroma_format = {:?}", f.chroma_format);
        let _ = writeln!(dump, "bitrate = {}", f.bitrate);
        let _ = writeln!(
            dump,
            "display_aspect_ratio = {}/{}",
            f.display_aspect_ratio.x, f.display_aspect_ratio.y
        );
        let vsd = &f.video_signal_description;
        let _ = writeln!(dump, "video_signal_description.video_format = {}", vsd.video_format);
        let _ = writeln!(
            dump,
            "video_signal_description.video_full_range_flag = {}",
            vsd.video_full_range_flag
        );
        let _ = writeln!(
            dump,
            "video_signal_description.color_primaries = {}",
            vsd.color_primaries
        );
        let _ = writeln!(
            dump,
            "video_signal_description.transfer_characteristics = {}",
            vsd.transfer_characteristics
        );
        let _ = writeln!(
            dump,
            "video_signal_description.matrix_coefficients = {}",
            vsd.matrix_coefficients
        );
        let _ = writeln!(dump, "seqhdr_data_length = {}", f.seqhdr_data_length);
    }

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
        CodecType: cudaVideoCodec::cudaVideoCodec_AV1,
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

/// Decode callback: dump CUVIDPICPARAMS + CUVIDAV1PICPARAMS, submit decode.
unsafe extern "C" fn decode_callback(
    pUserData: *mut c_void,
    pPicParams: *mut CUVIDPICPARAMS,
) -> c_int {
    let state = &mut *(pUserData as *mut State);
    state.decode_count += 1;
    let i = state.decode_count - 1;
    let pp = &*pPicParams;
    // The codec-specific union part (CUVIDAV1PICPARAMS — the ffi struct layout
    // matches the SDK header exactly).
    let av1 = &pp.CodecSpecific.av1;

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
        let _ = writeln!(dump, "nBitstreamDataLen = {}", pp.nBitstreamDataLen);
        let _ = writeln!(dump, "nNumSlices = {}", pp.nNumSlices);
        let _ = writeln!(dump, "field_pic_flag = {}", pp.field_pic_flag);
        let _ = writeln!(dump, "bottom_field_flag = {}", pp.bottom_field_flag);
        let _ = writeln!(dump, "second_field = {}", pp.second_field);
        let _ = writeln!(dump, "Reserved = [{}]", pp.Reserved.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "));
        let _ = writeln!(dump, "ref_pic_flag = {}", pp.ref_pic_flag);
        let _ = writeln!(dump, "intra_pic_flag = {}", pp.intra_pic_flag);
        let _ = writeln!(dump, "-- CUVIDAV1PICPARAMS --");
        let _ = writeln!(dump, "width = {}", av1.width);
        let _ = writeln!(dump, "height = {}", av1.height);
        let _ = writeln!(dump, "frame_offset = {}", av1.frame_offset);
        let _ = writeln!(dump, "decodePicIdx = {}", av1.decodePicIdx);
        let _ = writeln!(
            dump,
            "profile = {} use_128x128 = {} subsampling = {}/{} mono = {} bit_depth_minus8 = {}",
            av1.profile(),
            av1.use_128x128_superblock(),
            av1.subsampling_x(),
            av1.subsampling_y(),
            av1.mono_chrome(),
            av1.bit_depth_minus8()
        );
        let _ = writeln!(
            dump,
            "enable_order_hint = {} order_hint_bits_minus1 = {} enable_cdef = {} enable_restoration = {} enable_superres = {} enable_fgs = {}",
            av1.enable_order_hint(), av1.order_hint_bits_minus1(), av1.enable_cdef(), av1.enable_restoration(), av1.enable_superres(), av1.enable_fgs()
        );
        let _ = writeln!(
            dump,
            "frame_type = {} show_frame = {} disable_cdf_update = {} allow_sct = {} force_integer_mv = {} coded_denom = {}",
            av1.frame_type(), av1.show_frame(), av1.disable_cdf_update(), av1.allow_screen_content_tools(), av1.force_integer_mv(), av1.coded_denom()
        );
        let _ = writeln!(
            dump,
            "interp_filter = {} switchable_motion_mode = {} use_ref_frame_mvs = {} tx_mode = {} reference_mode = {} reduced_tx_set = {} skip_mode = {}",
            av1.interp_filter(), av1.switchable_motion_mode(), av1.use_ref_frame_mvs(), av1.tx_mode(), av1.reference_mode(), av1.reduced_tx_set(), av1.skip_mode()
        );
        let _ = writeln!(
            dump,
            "delta_q_present = {} delta_q_res = {} using_qmatrix = {} coded_lossless = {} use_superres = {}",
            av1.delta_q_present(), av1.delta_q_res(), av1.using_qmatrix(), av1.coded_lossless(), av1.use_superres()
        );
        let _ = writeln!(
            dump,
            "num_tile_cols = {} num_tile_rows = {} context_update_tile_id = {}",
            av1.num_tile_cols(), av1.num_tile_rows(), av1.context_update_tile_id()
        );
        let _ = writeln!(
            dump,
            "tile_widths = [{}]",
            av1.tile_widths.iter().take(16).map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
        );
        let _ = writeln!(
            dump,
            "tile_heights = [{}]",
            av1.tile_heights.iter().take(16).map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
        );
        let _ = writeln!(
            dump,
            "cdef_damping_minus_3 = {} cdef_bits = {}",
            av1.cdef_damping_minus_3(), av1.cdef_bits()
        );
        let _ = writeln!(
            dump,
            "base_qindex = {} qp_y_dc = {} qp_u_dc = {} qp_v_dc = {} qp_u_ac = {} qp_v_ac = {}",
            av1.base_qindex,
            av1.qp_y_dc_delta_q,
            av1.qp_u_dc_delta_q,
            av1.qp_v_dc_delta_q,
            av1.qp_u_ac_delta_q,
            av1.qp_v_ac_delta_q
        );
        let _ = writeln!(
            dump,
            "segmentation: enabled={} update_map={} update_data={} temporal_update={}",
            av1.segmentation_enabled(),
            av1.segmentation_update_map(),
            av1.segmentation_update_data(),
            av1.segmentation_temporal_update()
        );
        let _ = writeln!(
            dump,
            "loop_filter: level=[{},{}] level_u={} level_v={} sharpness={} delta_enabled={} delta_update={} delta_lf_present={} delta_lf_res={} delta_lf_multi={}",
            av1.loop_filter_level[0],
            av1.loop_filter_level[1],
            av1.loop_filter_level_u,
            av1.loop_filter_level_v,
            av1.loop_filter_sharpness,
            av1.loop_filter_delta_enabled(),
            av1.loop_filter_delta_update(),
            av1.delta_lf_present(),
            av1.delta_lf_res(),
            av1.delta_lf_multi()
        );
        let _ = writeln!(
            dump,
            "loop_filter_ref_deltas = [{}]",
            av1.loop_filter_ref_deltas
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(
            dump,
            "loop_filter_mode_deltas = [{}]",
            av1.loop_filter_mode_deltas
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(
            dump,
            "lr_unit_size = [{} {} {}]",
            av1.lr_unit_size[0], av1.lr_unit_size[1], av1.lr_unit_size[2]
        );
        let _ = writeln!(
            dump,
            "lr_type = [{} {} {}]",
            av1.lr_type[0], av1.lr_type[1], av1.lr_type[2]
        );
        let _ = writeln!(dump, "primary_ref_frame = {}", av1.primary_ref_frame);
        let _ = writeln!(
            dump,
            "ref_frame_map = [{}]",
            av1.ref_frame_map
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(
            dump,
            "temporal_layer_id = {} spatial_layer_id = {}",
            av1.temporal_layer_id(), av1.spatial_layer_id()
        );
        let _ = writeln!(
            dump,
            "ref_frame = [{}]",
            (0..7)
                .map(|r| {
                    format!(
                        "({}x{} idx={})",
                        av1.ref_frame[r].width, av1.ref_frame[r].height, av1.ref_frame[r].index
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        );
        let _ = writeln!(
            dump,
            "global_motion = [{}]",
            (0..7)
                .map(|r| {
                    format!(
                        "(invalid={} wmtype={} mat=[{} {} {} {} {} {}])",
                        av1.global_motion[r].invalid(),
                        av1.global_motion[r].wmtype(),
                        av1.global_motion[r].wmmat[0],
                        av1.global_motion[r].wmmat[1],
                        av1.global_motion[r].wmmat[2],
                        av1.global_motion[r].wmmat[3],
                        av1.global_motion[r].wmmat[4],
                        av1.global_motion[r].wmmat[5]
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        );
        let _ = writeln!(dump, "apply_grain = {}", av1.apply_grain());
        // First 32 bytes of the actual bitstream passed to cuvidDecodePicture.
        let bs = std::slice::from_raw_parts(pp.pBitstreamData, (pp.nBitstreamDataLen as usize).min(32));
        let _ = writeln!(dump, "BITSTREAM[0..32] = {}", bs.iter().map(|b| format!("{:02x}", b)).collect::<String>());
        // Full raw byte dump of the CUVIDAV1PICPARAMS for byte-exact diffing.
        let raw = unsafe {
            std::slice::from_raw_parts(
                av1 as *const CUVIDAV1PICPARAMS as *const u8,
                std::mem::size_of::<CUVIDAV1PICPARAMS>(),
            )
        };
        let _ = writeln!(dump, "RAW = {}", raw.iter().map(|b| format!("{:02x}", b)).collect::<String>());
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
    // [DEBUG] Dump the exact full bitstream passed to cuvidDecodePicture so it
    // can be byte-compared against our decoder.
    if let Ok(dir) = std::env::var("NVDEC_BS_DUMP_DIR") {
        let n = pp.nBitstreamDataLen as usize;
        let bs = std::slice::from_raw_parts(pp.pBitstreamData, n);
        let _ = std::fs::create_dir_all(&dir);
        let fp = std::path::Path::new(&dir).join(format!("base_bs_{}.bin", i));
        let _ = std::fs::write(&fp, bs);
    }

    let cuda_log = std::env::var("NVDEC_CUDA_LOG").is_ok();
    let t0 = std::time::Instant::now();
    let procparams = nvdec_decode::ffi::default_procparams();
    let res = unsafe { (funcs.decode_picture)(decoder, pPicParams, &procparams) };
    if cuda_log {
        eprintln!(
            "[CUDA-LOG] decode#{} CurrPicIdx={} nBitLen={} decode_picture={}us",
            i, pp.CurrPicIdx, pp.nBitstreamDataLen, t0.elapsed().as_micros()
        );
    }
    if res == CUDA_SUCCESS {
        state.decode_picture_ok += 1;
        state.decoded_surfaces.push(pp.CurrPicIdx as u32);
        // [DEBUG] Poll per-picture decode status (gated by NVDEC_DEBUG_STATUS).
        if std::env::var("NVDEC_DEBUG_STATUS").is_ok() {
            let mut ds = nvdec_decode::ffi::CUVIDGETDECODESTATUS {
                decodeStatus: nvdec_decode::ffi::cuvidDecodeStatus::cuvidDecodeStatus_Invalid,
                reserved: [0; 31],
                pReserved: [std::ptr::null_mut(); 8],
            };
            let mut api: u32 = 0;
            for _ in 0..50 {
                api = unsafe {
                    (funcs.get_decode_status)(decoder, pp.CurrPicIdx, &mut ds)
                };
                if ds.decodeStatus
                    != nvdec_decode::ffi::cuvidDecodeStatus::cuvidDecodeStatus_InProgress
                {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
                let _ = cu_ctx_synchronize();
            }
            eprintln!(
                "[STATUS] decode#{} CurrPicIdx={} status={} api={}",
                i, pp.CurrPicIdx, ds.decodeStatus as u32, api
            );
        }
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
    // [DEBUG] Dump the raw NV12 surface for byte-comparison against our decoder.
    if let Ok(dir) = std::env::var("NVDEC_RAW_DUMP") {
        let _ = std::fs::create_dir_all(&dir);
        let fp = std::path::Path::new(&dir).join(format!(
            "base_raw_{}.bin",
            info.picture_index
        ));
        let _ = std::fs::write(&fp, &host);
        eprintln!(
            "[RAW] pic={} pitch={} h={} surface_bytes={}",
            info.picture_index,
            pitch,
            state.coded_height,
            host.len()
        );
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

fn dump_av1_frame_header(w: &mut impl Write, i: usize, fh: &Av1FrameHeader, pts: u64) -> std::io::Result<()> {
    writeln!(w, "=== AV1PARSER {} ===", i)?;
    writeln!(w, "pts = {}", pts)?;
    writeln!(w, "show_existing_frame = {}", fh.show_existing_frame)?;
    writeln!(w, "frame_to_show_map_idx = {}", fh.frame_to_show_map_idx)?;
    writeln!(w, "frame_type = {}", fh.frame_type)?;
    writeln!(w, "primary_ref_frame = {}", fh.primary_ref_frame)?;
    writeln!(
        w,
        "ref_frame_idx = [{}]",
        fh.ref_frame_idx
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )?;
    writeln!(w, "frame_width = {}", fh.frame_width)?;
    writeln!(w, "frame_height = {}", fh.frame_height)?;
    writeln!(w, "render_width = {}", fh.render_width)?;
    writeln!(w, "render_height = {}", fh.render_height)?;
    writeln!(w, "tile_cols_log2 = {}", fh.tile_cols_log2)?;
    writeln!(w, "tile_rows_log2 = {}", fh.tile_rows_log2)?;
    writeln!(w, "tile_cols = {}", fh.tile_cols)?;
    writeln!(w, "tile_rows = {}", fh.tile_rows)?;
    writeln!(w, "uniform_tile_spacing_flag = {}", fh.uniform_tile_spacing_flag)?;
    writeln!(w, "context_update_tile_id = {}", fh.context_update_tile_id)?;
    writeln!(w, "order_hint = {}", fh.order_hint)?;
    writeln!(w, "error_resilient_mode = {}", fh.error_resilient_mode)?;
    writeln!(w, "refresh_frame_flags = {}", fh.refresh_frame_flags)?;
    writeln!(w, "show_frame = {}", fh.show_frame)?;
    writeln!(w, "use_superres = {}", fh.use_superres)?;
    writeln!(
        w,
        "allow_screen_content_tools = {} force_integer_mv = {} frame_refs_short_signaling = {}",
        fh.allow_screen_content_tools, fh.force_integer_mv, fh.frame_refs_short_signaling
    )?;
    writeln!(
        w,
        "is_filter_switchable = {} interpolation_filter = {} use_ref_frame_mvs = {}",
        fh.is_filter_switchable, fh.interpolation_filter, fh.use_ref_frame_mvs
    )?;
    writeln!(
        w,
        "disable_cdf_update = {} disable_frame_end_update_cdf = {} reduced_tx_set = {} reference_select = {} tx_mode = {}",
        fh.disable_cdf_update, fh.disable_frame_end_update_cdf, fh.reduced_tx_set, fh.reference_select, fh.tx_mode
    )?;
    writeln!(
        w,
        "delta_q_present = {} delta_q_res = {} using_qmatrix = {} base_q_index = {}",
        fh.delta_q_present, fh.delta_q_res, fh.using_qmatrix, fh.base_q_index
    )?;
    writeln!(
        w,
        "delta_q_y_dc = {} delta_q_u_dc = {} delta_q_u_ac = {} delta_q_v_dc = {} delta_q_v_ac = {}",
        fh.delta_q_y_dc, fh.delta_q_u_dc, fh.delta_q_u_ac, fh.delta_q_v_dc, fh.delta_q_v_ac
    )?;
    writeln!(
        w,
        "segmentation: enabled={} update_map={} temporal_update={} update_data={} abs_or_delta={}",
        fh.segmentation_enabled,
        fh.segmentation_update_map,
        fh.segmentation_temporal_update,
        fh.segmentation_update_data,
        fh.segmentation_abs_or_delta_update
    )?;
    writeln!(
        w,
        "loop_filter: level=[{},{}] level_uv=[{},{}] sharpness={} delta_enabled={} delta_update={}",
        fh.loop_filter_level[0],
        fh.loop_filter_level[1],
        fh.loop_filter_level_uv[0],
        fh.loop_filter_level_uv[1],
        fh.loop_filter_sharpness,
        fh.loop_filter_delta_enabled,
        fh.loop_filter_delta_update
    )?;
    writeln!(
        w,
        "loop_filter_ref_deltas = [{}]",
        fh.loop_filter_ref_deltas
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )?;
    writeln!(
        w,
        "cdef_damping = {} cdef_bits = {}",
        fh.cdef_damping, fh.cdef_bits
    )?;
    writeln!(
        w,
        "coded_lossless = {} all_lossless = {} apply_grain = {} showable_frame = {} coded_denom = {}",
        fh.coded_lossless, fh.all_lossless, fh.apply_grain, fh.showable_frame, fh.coded_denom
    )?;
    writeln!(
        w,
        "order_hints = [{}]",
        fh.order_hints
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )?;
    writeln!(
        w,
        "loop_restoration_type = [{} {} {}] uses_lr = {}",
        fh.loop_restoration_type[0],
        fh.loop_restoration_type[1],
        fh.loop_restoration_type[2],
        fh.uses_lr
    )?;
    writeln!(
        w,
        "global_motion_type = [{}]",
        fh.global_motion_type
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )?;
    writeln!(w, "frame_header_size = {}", fh.frame_header_size)?;
    Ok(())
}

/// Parse the same IVF with the Rust vk-video-parser (Av1Parser) and dump
/// Av1FrameHeader for every frame.
fn run_vkparser(ivf: &IvfFile, out_path: &Path) {
    let mut parser = Av1Parser::new();
    parser
        .init(&DetectedVideoFormat::new(VideoCodec::DecodeAv1))
        .expect("Failed to init AV1 parser");

    let file = File::create(out_path).expect("failed to create vkparser dump file");
    let mut w = std::io::BufWriter::new(file);

    let mut sps = None;
    let mut frames = 0usize;
    let mut show_existing = 0usize;
    let mut err = 0usize;

    for (i, (payload, pts)) in ivf.packets.iter().enumerate() {
        let obus = walk_obus(payload);
        let mut handled = false;
        for (typ, obu_data) in obus {
            match typ {
                1 => {
                    // SequenceHeader
                    if sps.is_none() {
                        match parser.parse_sequence_header_obu(obu_data) {
                            Ok(s) => {
                                println!(
                                    "[vkparser] SPS: profile={} maxw-1={} maxh-1={} ohb-1={} cdef={} restoration={} superres={}",
                                    s.profile,
                                    s.max_frame_width_minus_1,
                                    s.max_frame_height_minus_1,
                                    s.order_hint_bits_minus1,
                                    s.enable_cdef,
                                    s.enable_restoration,
                                    s.enable_superres
                                );
                                let _ = writeln!(w, "=== SPS ===");
                                let _ = writeln!(w, "profile = {}", s.profile);
                                let _ = writeln!(w, "max_frame_width_minus_1 = {}", s.max_frame_width_minus_1);
                                let _ = writeln!(w, "max_frame_height_minus_1 = {}", s.max_frame_height_minus_1);
                                let _ = writeln!(w, "order_hint_bits_minus1 = {}", s.order_hint_bits_minus1);
                                let _ = writeln!(w, "enable_order_hint = {}", s.enable_order_hint);
                                let _ = writeln!(w, "enable_cdef = {}", s.enable_cdef);
                                let _ = writeln!(w, "enable_restoration = {}", s.enable_restoration);
                                let _ = writeln!(w, "enable_superres = {}", s.enable_superres);
                                let _ = writeln!(w, "enable_warped_motion = {}", s.enable_warped_motion);
                                let _ = writeln!(w, "enable_ref_frame_mvs = {}", s.enable_ref_frame_mvs);
                                let _ = writeln!(w, "seq_force_screen_content_tools = {}", s.seq_force_screen_content_tools);
                                let _ = writeln!(w, "seq_force_integer_mv = {}", s.seq_force_integer_mv);
                                let _ = writeln!(w, "separate_uv_delta_q = {}", s.separate_uv_delta_q);
                                let _ = writeln!(w, "high_bitdepth = {}", s.high_bitdepth);
                                let _ = writeln!(w, "twelve_bit = {}", s.twelve_bit);
                                let _ = writeln!(w, "subsampling_x = {}", s.subsampling_x);
                                let _ = writeln!(w, "subsampling_y = {}", s.subsampling_y);
                                let _ = writeln!(w, "color_range = {}", s.color_range);
                                sps = Some(s);
                            }
                            Err(e) => {
                                err += 1;
                                let _ = writeln!(w, "SPS PARSE ERROR: {}", e);
                            }
                        }
                    }
                }
                3 | 6 => {
                    // FrameHeader (show_existing) or Frame
                    if let Some(s) = &sps {
                        match parser.parse_frame_header(obu_data, s) {
                            Ok(fh) => {
                                if fh.show_existing_frame {
                                    show_existing += 1;
                                } else {
                                    frames += 1;
                                }
                                if let Err(e) = dump_av1_frame_header(&mut w, i, &fh, *pts) {
                                    eprintln!("[vkparser] dump error at packet {}: {}", i, e);
                                }
                            }
                            Err(e) => {
                                err += 1;
                                let _ = writeln!(w, "=== AV1PARSER {} ===", i);
                                let _ = writeln!(w, "FRAME HEADER PARSE ERROR: {}", e);
                            }
                        }
                        handled = true;
                    }
                }
                _ => {}
            }
        }
        if !handled && sps.is_some() {
            // Packet with no recognized frame OBUs (e.g. pure temporal delimiter).
            let _ = writeln!(w, "=== AV1PARSER {} === (no frame OBU)", i);
        }
    }
    let _ = w.flush();
    println!(
        "[vkparser] {} packets -> {} frames + {} show_existing ({} errors) -> {}",
        ivf.packets.len(),
        frames,
        show_existing,
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
        .unwrap_or_else(|| "assets/big_buck_bunny_av1.ivf".to_string());
    let max_frames: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(300);
    let out_prefix = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "nvdec_av1_cuvid".to_string());

    if !std::path::Path::new(&ivf_path).exists() {
        eprintln!("Error: File not found: {}", ivf_path);
        std::process::exit(1);
    }

    println!("=== NVDEC AV1 cuvid-parser baseline ===");
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
        CodecType: cudaVideoCodec::cudaVideoCodec_AV1,
        ulMaxNumDecodeSurfaces: NUM_SURFACES,
        ulClockRate: 90000,
        ulErrorThreshold: 0,
        ulMaxDisplayDelay: 1,
        bAnnexb_and_reserved: 0, // raw OBU payloads (IVF low-overhead), NOT annexb
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

    // Step 5: feed packets (raw OBU payloads)
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

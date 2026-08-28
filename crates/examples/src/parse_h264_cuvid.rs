//! Reference-data dumper for the NVIDIA cuvid H.264 parser.
//!
//! Feeds an Annex-B H.264 stream to `cuvidCreateVideoParser` and dumps, for
//! every picture, the full parsed state that cuvid exposes:
//!   - `CUVIDEOFORMAT` (sequence callback)
//!   - `CUVIDPICPARAMS` + `CUVIDH264PICPARAMS` incl. DPB entries (decode cb)
//!   - `CUVIDPARSERDISPINFO` (display callback, display order)
//!
//! Usage: parse_h264_cuvid <stream.h264> [max_frames] [out.txt]
//!
//! The dump file format is `key = value` lines under `=== SECTION ===`
//! headers so it can be diffed / parsed mechanically.

use std::ffi::{c_int, c_void};
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write;

use nvdec_decode::device::{cu_ctx_set_current, get_funcs};
use nvdec_decode::ffi::{
    CUVIDH264PICPARAMS, CUVIDPARSERDISPINFO, CUVIDPARSERPARAMS, CUVIDPICPARAMS, CUVIDEOFORMAT,
    CUVIDSOURCEDATAPACKET, CUvideoparser, CUDA_SUCCESS, cudaVideoCodec,
};
use nvdec_decode::{init_nvdec, is_available};

const NUM_SURFACES: u32 = 16;

struct State {
    dump: Option<std::io::BufWriter<File>>,
    sequence_count: u32,
    decode_count: u32,
    display_count: u32,
}

/// NAL type of a span starting with a start code (`00 00 [0] 01`).
fn nal_type(span: &[u8]) -> u8 {
    // The '1' of the start code is always at offset 2 from the pushed start.
    span[2] & 0x1F
}

/// Split an Annex-B stream into access units. Each unit starts at the first
/// slice NAL (type 1-5 / 19-20) after parameter sets / SEIs; multi-slice
/// frames stay in one unit.
fn split_access_units(data: &[u8]) -> Vec<Vec<u8>> {
    // Locate every start code (first zero of `00 00 [0] 01`).
    let mut starts: Vec<usize> = Vec::new();
    let mut i = 0;
    while i + 2 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    // NAL k spans [starts[k], starts[k+1]) (last one to EOF).
    let mut nals: Vec<(usize, usize)> = Vec::new();
    for k in 0..starts.len() {
        let end = starts.get(k + 1).copied().unwrap_or(data.len());
        nals.push((starts[k], end));
    }

    let is_slice_type = |t: u8| (1..=5).contains(&t) || (19..=20).contains(&t);

    let mut aus: Vec<Vec<u8>> = Vec::new();
    let mut pending: Vec<(usize, usize)> = Vec::new();
    let mut last_was_slice = false;
    for (s, e) in nals {
        let t = nal_type(&data[s..e]);
        if is_slice_type(t) {
            // Multi-slice: previous NAL was also a slice -> same access unit.
            if pending.is_empty() && last_was_slice {
                aus.last_mut().unwrap().extend_from_slice(&data[s..e]);
            } else {
                let mut au: Vec<u8> = Vec::new();
                for (ps, pe) in pending.drain(..) {
                    au.extend_from_slice(&data[ps..pe]);
                }
                au.extend_from_slice(&data[s..e]);
                aus.push(au);
            }
        } else {
            pending.push((s, e));
        }
        last_was_slice = is_slice_type(t);
    }
    // Trailing non-slice NALs (rare) attach to the last AU.
    if !pending.is_empty() {
        if let Some(last) = aus.last_mut() {
            for (ps, pe) in pending {
                last.extend_from_slice(&data[ps..pe]);
            }
        }
    }
    aus
}

unsafe extern "C" fn sequence_callback(
    pUserData: *mut c_void,
    pVideoFormat: *mut CUVIDEOFORMAT,
) -> c_int {
    let state = &mut *(pUserData as *mut State);
    state.sequence_count += 1;
    let seq = state.sequence_count - 1;
    let f = &*pVideoFormat;
    if let Some(dump) = state.dump.as_mut() {
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
        let _ = writeln!(
            dump,
            "min_num_decode_surfaces = {}",
            f.min_num_decode_surfaces
        );
        let _ = writeln!(dump, "coded_width = {}", f.coded_width);
        let _ = writeln!(dump, "coded_height = {}", f.coded_height);
        let _ = writeln!(
            dump,
            "display_area = [left={}, top={}, right={}, bottom={}]",
            f.display_area.left,
            f.display_area.top,
            f.display_area.right,
            f.display_area.bottom
        );
        let _ = writeln!(dump, "chroma_format = {:?}", f.chroma_format);
        let _ = writeln!(
            dump,
            "display_aspect_ratio = {}/{}",
            f.display_aspect_ratio.x, f.display_aspect_ratio.y
        );
    }
    1
}

unsafe extern "C" fn decode_callback(
    pUserData: *mut c_void,
    pPicParams: *mut CUVIDPICPARAMS,
) -> c_int {
    let state = &mut *(pUserData as *mut State);
    state.decode_count += 1;
    let pic = state.decode_count - 1;
    let common = &*pPicParams;
    let h264: &CUVIDH264PICPARAMS = &common.CodecSpecific.h264;

    if let Some(dump) = state.dump.as_mut() {
        let _ = writeln!(dump, "=== PIC {} (decode order) ===", pic);
        // Common CUVIDPICPARAMS
        let _ = writeln!(dump, "PicWidthInMbs = {}", common.PicWidthInMbs);
        let _ = writeln!(dump, "FrameHeightInMbs = {}", common.FrameHeightInMbs);
        let _ = writeln!(dump, "CurrPicIdx = {}", common.CurrPicIdx);
        let _ = writeln!(dump, "field_pic_flag = {}", common.field_pic_flag);
        let _ = writeln!(dump, "bottom_field_flag = {}", common.bottom_field_flag);
        let _ = writeln!(dump, "second_field = {}", common.second_field);
        let _ = writeln!(dump, "nBitstreamDataLen = {}", common.nBitstreamDataLen);
        let _ = writeln!(dump, "nNumSlices = {}", common.nNumSlices);
        let mut offs = String::new();
        for i in 0..common.nNumSlices {
            if !common.pSliceDataOffsets.is_null() {
                let off = *common.pSliceDataOffsets.add(i as usize);
                let _ = write!(offs, "{} ", off);
            }
        }
        let _ = writeln!(dump, "slice_data_offsets = [{}]", offs.trim_end());
        let _ = writeln!(dump, "ref_pic_flag = {}", common.ref_pic_flag);
        let _ = writeln!(dump, "intra_pic_flag = {}", common.intra_pic_flag);

        // SPS part
        let _ = writeln!(
            dump,
            "log2_max_frame_num_minus4 = {}",
            h264.log2_max_frame_num_minus4
        );
        let _ = writeln!(dump, "pic_order_cnt_type = {}", h264.pic_order_cnt_type);
        let _ = writeln!(
            dump,
            "log2_max_pic_order_cnt_lsb_minus4 = {}",
            h264.log2_max_pic_order_cnt_lsb_minus4
        );
        let _ = writeln!(
            dump,
            "delta_pic_order_always_zero_flag = {}",
            h264.delta_pic_order_always_zero_flag
        );
        let _ = writeln!(dump, "frame_mbs_only_flag = {}", h264.frame_mbs_only_flag);
        let _ = writeln!(
            dump,
            "direct_8x8_inference_flag = {}",
            h264.direct_8x8_inference_flag
        );
        let _ = writeln!(dump, "num_ref_frames = {}", h264.num_ref_frames);
        let _ = writeln!(
            dump,
            "bit_depth_luma_minus8 = {}",
            h264.bit_depth_luma_minus8
        );
        let _ = writeln!(
            dump,
            "bit_depth_chroma_minus8 = {}",
            h264.bit_depth_chroma_minus8
        );
        // PPS part
        let _ = writeln!(
            dump,
            "entropy_coding_mode_flag = {}",
            h264.entropy_coding_mode_flag
        );
        let _ = writeln!(dump, "pic_order_present_flag = {}", h264.pic_order_present_flag);
        let _ = writeln!(
            dump,
            "num_ref_idx_l0_active_minus1 = {}",
            h264.num_ref_idx_l0_active_minus1
        );
        let _ = writeln!(
            dump,
            "num_ref_idx_l1_active_minus1 = {}",
            h264.num_ref_idx_l1_active_minus1
        );
        let _ = writeln!(dump, "weighted_pred_flag = {}", h264.weighted_pred_flag);
        let _ = writeln!(dump, "weighted_bipred_idc = {}", h264.weighted_bipred_idc);
        let _ = writeln!(
            dump,
            "pic_init_qp_minus26 = {}",
            h264.pic_init_qp_minus26
        );
        let _ = writeln!(
            dump,
            "deblocking_filter_control_present_flag = {}",
            h264.deblocking_filter_control_present_flag
        );
        let _ = writeln!(
            dump,
            "redundant_pic_cnt_present_flag = {}",
            h264.redundant_pic_cnt_present_flag
        );
        let _ = writeln!(
            dump,
            "transform_8x8_mode_flag = {}",
            h264.transform_8x8_mode_flag
        );
        let _ = writeln!(dump, "MbaffFrameFlag = {}", h264.MbaffFrameFlag);
        let _ = writeln!(
            dump,
            "constrained_intra_pred_flag = {}",
            h264.constrained_intra_pred_flag
        );
        let _ = writeln!(
            dump,
            "chroma_qp_index_offset = {}",
            h264.chroma_qp_index_offset
        );
        let _ = writeln!(
            dump,
            "second_chroma_qp_index_offset = {}",
            h264.second_chroma_qp_index_offset
        );
        // Slice / picture part
        let _ = writeln!(dump, "ref_pic_flag_h264 = {}", h264.ref_pic_flag);
        let _ = writeln!(dump, "frame_num = {}", h264.frame_num);
        let _ = writeln!(
            dump,
            "CurrFieldOrderCnt = [{}, {}]",
            h264.CurrFieldOrderCnt[0], h264.CurrFieldOrderCnt[1]
        );

        // DPB entries
        let mut any = false;
        for (i, e) in h264.dpb.iter().enumerate() {
            if e.not_existing != 0 || e.used_for_reference != 0 || e.PicIdx >= 0 {
                any = true;
                let _ = writeln!(
                    dump,
                    "dpb[{}] = {{ PicIdx={}, FrameIdx={}, is_long_term={}, not_existing={}, used_for_reference={}, FieldOrderCnt=[{}, {}] }}",
                    i,
                    e.PicIdx,
                    e.FrameIdx,
                    e.is_long_term,
                    e.not_existing,
                    e.used_for_reference,
                    e.FieldOrderCnt[0],
                    e.FieldOrderCnt[1]
                );
            }
        }
        if !any {
            let _ = writeln!(dump, "dpb = [empty]");
        }

        // Scaling lists (raster order) — dump only when non-default.
        let mut s4: Vec<String> = Vec::new();
        for (i, m) in h264.WeightScale4x4.iter().enumerate() {
            let all16 = m.iter().all(|&v| v == 16);
            if !all16 {
                s4.push(format!(
                    "{}=[{}]",
                    i,
                    m.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")
                ));
            }
        }
        if !s4.is_empty() {
            let _ = writeln!(dump, "WeightScale4x4_nondefault = {{{}}}", s4.join(" "));
        }
        let mut s8: Vec<String> = Vec::new();
        for (i, m) in h264.WeightScale8x8.iter().enumerate() {
            let all16 = m.iter().all(|&v| v == 16);
            if !all16 {
                s8.push(format!(
                    "{}=[{}]",
                    i,
                    m.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")
                ));
            }
        }
        if !s8.is_empty() {
            let _ = writeln!(dump, "WeightScale8x8_nondefault = {{{}}}", s8.join(" "));
        }
        let _ = writeln!(dump, "num_slice_groups_minus1 = {}", h264.num_slice_groups_minus1);
        let _ = writeln!(dump, "slice_group_map_type = {}", h264.slice_group_map_type);
        let _ = writeln!(dump, "pic_init_qs_minus26 = {}", h264.pic_init_qs_minus26);
    }
    1
}

unsafe extern "C" fn display_callback(
    pUserData: *mut c_void,
    pPicture: *mut CUVIDPARSERDISPINFO,
) -> c_int {
    let state = &mut *(pUserData as *mut State);
    state.display_count += 1;
    let disp = state.display_count - 1;
    let d = &*pPicture;
    if let Some(dump) = state.dump.as_mut() {
        let _ = writeln!(dump, "=== DISPLAY {} (display order) ===", disp);
        let _ = writeln!(dump, "picture_index = {}", d.picture_index);
        let _ = writeln!(dump, "progressive_frame = {}", d.progressive_frame);
        let _ = writeln!(dump, "top_field_first = {}", d.top_field_first);
        let _ = writeln!(dump, "repeat_first_field = {}", d.repeat_first_field);
        let _ = writeln!(dump, "timestamp = {}", d.timestamp);
    }
    1
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <stream.h264> [max_frames] [out.txt]", args[0]);
        std::process::exit(1);
    }
    let stream_path = &args[1];
    let max_frames: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(300);
    let out_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "cuvid_h264_ref.txt".to_string());

    if !std::path::Path::new(stream_path).exists() {
        eprintln!("Error: File not found: {}", stream_path);
        std::process::exit(1);
    }
    let data = std::fs::read(stream_path).expect("Failed to read file");

    if init_nvdec().is_err() || !is_available() {
        eprintln!("Error: NVDEC not available");
        std::process::exit(1);
    }
    if let Err(e) = cu_ctx_set_current() {
        eprintln!("Error: cuCtxSetCurrent failed: {}", e);
        std::process::exit(1);
    }

    let aus = split_access_units(&data);
    println!(
        "{}: {} bytes, {} access units (max {})",
        stream_path,
        data.len(),
        aus.len(),
        max_frames
    );

    let file = File::create(&out_path).expect("failed to create dump file");
    let state = Box::new(State {
        dump: Some(std::io::BufWriter::new(file)),
        sequence_count: 0,
        decode_count: 0,
        display_count: 0,
    });
    let state_ptr = Box::into_raw(state) as *mut c_void;

    let parser_params = CUVIDPARSERPARAMS {
        CodecType: cudaVideoCodec::cudaVideoCodec_H264,
        ulMaxNumDecodeSurfaces: NUM_SURFACES,
        ulClockRate: 90000,
        ulErrorThreshold: 0,
        ulMaxDisplayDelay: 1,
        bAnnexb_and_reserved: 1, // Annex-B input
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

    for (idx, au) in aus.iter().take(max_frames).enumerate() {
        let packet = CUVIDSOURCEDATAPACKET {
            flags: 0x02, // CUVID_PKT_TIMESTAMP
            payload_size: au.len() as u64,
            payload: au.as_ptr(),
            timestamp: (idx as i64) * 3600, // 25 fps in 90 kHz clock
        };
        let res = unsafe { (funcs.parse_video_data)(parser, &packet) };
        if res != CUDA_SUCCESS {
            eprintln!("Error: cuvidParseVideoData failed on AU {} with {}", idx, res);
            break;
        }
    }

    let eos = CUVIDSOURCEDATAPACKET {
        flags: 0x01, // CUVID_PKT_ENDOFSTREAM
        payload_size: 0,
        payload: std::ptr::null(),
        timestamp: -1,
    };
    let _ = unsafe { (funcs.parse_video_data)(parser, &eos) };
    let _ = unsafe { (funcs.destroy_video_parser)(parser) };

    let mut state = unsafe { Box::from_raw(state_ptr as *mut State) };
    if let Some(mut dump) = state.dump.take() {
        let _ = dump.flush();
    }
    println!(
        "sequences={} pictures={} displays={}\nDump written to {}",
        state.sequence_count, state.decode_count, state.display_count, out_path
    );
}

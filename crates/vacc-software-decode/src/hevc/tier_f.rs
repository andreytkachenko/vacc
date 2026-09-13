//! Tier F end-to-end differential: Rust pipeline (vacc-parser + hevc driver)
//! vs the C++ `hevcdec_decode_picture` core, access unit by access unit on
//! real streams.
//!
//! This validates the full Rust decode path — parser SPS/PPS/slice-header
//! mapping (`syntax_map`), DPB reference resolution (`H265Dpb`), slice-data
//! reconstruction (seek from `header_bit_size`, CABAC, prediction, residual),
//! and the in-loop filters (deblocking + SAO) — i.e. everything needed to
//! replace the C++ control plane byte-for-byte.
//!
//! Both sides decode the *same* AU bytes: the C++ core keeps its own DPB and
//! resolves references internally; the Rust side drives the ported kernels
//! from parser output. Output planes are compared sample-by-sample per AU.

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    use vacc_core::picture::{H265Pps, H265Sps};
    use vacc_parser::h265::{H265Parser, SliceHeaderInfo};
    use vacc_parser::h265_dpb::H265Dpb;
    use vacc_parser::{BitstreamPacket, ParseResult, SliceHeader, VideoParser};

    use crate::ffi;
    use crate::hevc::driver::{decode_picture, PictureStore, RefListEntry, SliceInput};
    use crate::hevc::syntax_map;
    use crate::hevc::transform::ScalingListData;

    const MAX_DPB_SLOTS: usize = 64;

    fn sample_dir() -> Option<PathBuf> {
        if let Ok(d) = env::var("VACC_SAMPLES_DIR") {
            return Some(PathBuf::from(d));
        }
        let p = PathBuf::from("/home/atkachenko/apps/vacc/assets/samples");
        p.is_dir().then_some(p)
    }

    /// Per-plane sample comparison over the coded region, reporting the first
    /// mismatches with (x,y) coordinates.
    #[allow(clippy::too_many_arguments)] // flat per-plane compare; mirrors plane layout
    fn first_diff(name: &str, cpp: &[u8], bps: usize, stride_c: usize, rust: &[u16], stride_r: usize, w: i32, h: i32) {
        let mut n = 0usize;
        let mut shown = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let ci = (y as usize) * stride_c + x as usize;
                let cpp_v = if bps == 1 {
                    cpp[ci] as u16
                } else {
                    u16::from_le_bytes([cpp[2 * ci], cpp[2 * ci + 1]])
                };
                let ri = (y as usize) * stride_r + x as usize;
                if cpp_v != rust[ri] {
                    n += 1;
                    if shown.len() < 16 {
                        shown.push(format!("({x},{y}) cpp={cpp_v} rust={}", rust[ri]));
                    }
                }
            }
        }
        assert_eq!(n, 0, "{name}: {n} sample mismatches, first: {}", shown.join(", "));
    }

    /// Decode one AU with both backends and compare the filtered output.
    #[allow(clippy::too_many_arguments)] // test harness: mirrors the pipeline shape
    fn decode_and_compare(
        name: &str,
        au_idx: usize,
        cpp_ctx: *mut ffi::hevcdec_context,
        au: &[u8],
        slices: &[vacc_parser::SliceEntry],
        sps_h: &H265Sps,
        pps_h: &H265Pps,
        first_info: &SliceHeaderInfo,
        dpb: &mut H265Dpb,
        store: &mut PictureStore,
        coded_w: u32,
        coded_h: u32,
        chroma_w: u32,
        chroma_h: u32,
        bps: usize,
        ystride: usize,
        cstride: usize,
    ) {
        let prefix = format!("{name} au {au_idx}");

        // ---- C++ core (ground truth) ----
        let mut out_y = vec![0u8; coded_h as usize * ystride * bps + 16];
        let mut out_u = vec![0u8; chroma_h as usize * cstride * bps + 16];
        let mut out_v = vec![0u8; chroma_h as usize * cstride * bps + 16];
        let rc = unsafe {
            ffi::hevcdec_decode_picture(
                cpp_ctx,
                au.as_ptr(),
                au.len(),
                out_y.as_mut_ptr(),
                ystride as i32,
                out_u.as_mut_ptr(),
                cstride as i32,
                out_v.as_mut_ptr(),
                cstride as i32,
            )
        };
        assert_eq!(rc, ffi::HEVCDEC_OK, "{prefix}: C++ decode failed ({rc})");

        // ---- Rust pipeline ----
        let sps = syntax_map::map_sps(sps_h);
        let pps = syntax_map::map_pps(pps_h, &sps);
        let has_chroma = sps.chroma_array_type != 0;

        let sps_scaling_enabled = sps_h.sps_scaling_list_data_present_flag;
        let sps_scaling = if sps_scaling_enabled {
            syntax_map::map_scaling_list(&sps_h.scaling_lists)
        } else {
            ScalingListData::default()
        };
        let pps_scaling_present = pps_h.pps_scaling_list_data_present_flag;
        let pps_scaling = if pps_scaling_present {
            syntax_map::map_scaling_list(&pps_h.scaling_lists)
        } else {
            ScalingListData::default()
        };

        // Build the per-segment slice inputs (dependent-slice inheritance from
        // the last independent segment, mirroring C++ decoder.cpp).
        let mut last_independent: Option<&SliceHeaderInfo> = None;
        let mut slice_inputs: Vec<SliceInput> = Vec::new();
        for e in slices {
            let Some(SliceHeader::H265(info)) = &e.slice_header else {
                continue;
            };
            let sh = syntax_map::map_sh(info, last_independent, pps_h, has_chroma);
            let deblock = syntax_map::map_deblock_params(info, pps_h);
            slice_inputs.push(SliceInput {
                nal: &e.nal_data,
                sh,
                deblock,
                header_bit_size: info.header_bit_size,
            });
            if !info.dependent_slice_segment_flag {
                last_independent = Some(info);
            }
        }
        assert!(!slice_inputs.is_empty(), "{prefix}: no slice segments parsed");

        let slot = dpb.picture_start(sps_h, first_info, first_info.is_reference);
        let lists = dpb.build_ref_lists();
        let to_entries = |v: &[vacc_parser::h265_dpb::H265RefPic]| -> Vec<RefListEntry> {
            v.iter()
                .map(|r| RefListEntry {
                    slot: r.slot,
                    poc: r.poc,
                    long_term: if r.slot >= 0 { dpb.slot_is_long_term(r.slot as usize) } else { false },
                })
                .collect()
        };
        let refs_l0 = to_entries(&lists.l0);
        let refs_l1 = to_entries(&lists.l1);

        let cur_poc = first_info.curr_pic_order_cnt_val;
        let pic = decode_picture(
            &sps,
            &pps,
            &slice_inputs,
            &refs_l0,
            &refs_l1,
            store,
            cur_poc,
            sps_scaling_enabled,
            &sps_scaling,
            pps_scaling_present,
            &pps_scaling,
        )
        .unwrap_or_else(|e| panic!("{prefix}: Rust decode failed: {e}"));

        // ---- Compare filtered planes over the coded region ----
        first_diff(
            &format!("{prefix}: Y"),
            &out_y,
            bps,
            ystride,
            &pic.planes[0],
            pic.stride[0] as usize,
            coded_w as i32,
            coded_h as i32,
        );
        if has_chroma {
            first_diff(
                &format!("{prefix}: Cb"),
                &out_u,
                bps,
                cstride,
                &pic.planes[1],
                pic.stride[1] as usize,
                chroma_w as i32,
                chroma_h as i32,
            );
            first_diff(
                &format!("{prefix}: Cr"),
                &out_v,
                bps,
                cstride,
                &pic.planes[2],
                pic.stride[2] as usize,
                chroma_w as i32,
                chroma_h as i32,
            );
        }

        // ---- Commit to the Rust DPB + picture store ----
        store.store(slot, pic);
        dpb.commit_current(slot);
    }

    fn run_stream(name: &str) {
        let dir = match sample_dir() {
            Some(d) => d,
            None => {
                eprintln!("skip {name}: no samples dir (set VACC_SAMPLES_DIR)");
                return;
            }
        };
        let path = dir.join(format!("h265_{name}.h265"));
        let data = match fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {name}: {e}");
                return;
            }
        };

        let cpp_ctx = unsafe { ffi::hevcdec_create(0) };
        assert!(!cpp_ctx.is_null(), "{name}: hevcdec_create failed");

        let mut parser = H265Parser::new();
        let mut dpb: Option<H265Dpb> = None;
        let mut sps_h: Option<H265Sps> = None;
        let mut pps_h: Option<H265Pps> = None;
        let mut store = PictureStore::new(MAX_DPB_SLOTS);

        // Picture layout (set once the SPS is known).
        let mut coded_w = 0u32;
        let mut coded_h = 0u32;
        let mut chroma_w = 0u32;
        let mut chroma_h = 0u32;
        let mut bps = 1usize;
        let mut ystride = 0usize;
        let mut cstride = 0usize;

        let mut parse_offset = 0usize;
        let mut au_idx = 0usize;

        while parse_offset < data.len() {
            // Inner loop: consume parameter sets until a slice group (or end).
            let mut got_au = false;
            'au: while parse_offset < data.len() {
                let remaining = &data[parse_offset..];
                let packet = BitstreamPacket::new(remaining.to_vec());
                match parser.parse(&packet) {
                    Ok(ParseResult::ParameterSet { sps, pps, .. }) => {
                        if let Some(b) = sps
                            && let Some(s) = b.downcast_ref::<H265Sps>()
                        {
                            on_sps(
                                s,
                                &mut dpb,
                                &mut sps_h,
                                &mut coded_w,
                                &mut coded_h,
                                &mut chroma_w,
                                &mut chroma_h,
                                &mut bps,
                                &mut ystride,
                                &mut cstride,
                            );
                        }
                        if let Some(b) = pps
                            && let Some(p) = b.downcast_ref::<H265Pps>()
                        {
                            pps_h = Some(p.clone());
                        }
                        // PS NALs stay in the pending region; re-parse for slices.
                        continue 'au;
                    }
                    Ok(ParseResult::Slice { slices, bytes_consumed }) => {
                        if slices.is_empty() {
                            break 'au;
                        }
                        let au_end = parse_offset + bytes_consumed;
                        assert!(
                            au_end <= data.len(),
                            "{name}: bytes_consumed exceeds data"
                        );
                        let au = &data[parse_offset..au_end];
                        // Clone what we need out (parser borrow ends here).
                        let slices_owned: Vec<vacc_parser::SliceEntry> = slices;
                        let first_info = match &slices_owned[0].slice_header {
                            Some(SliceHeader::H265(i)) => i.clone(),
                            _ => panic!("{name}: no H265 slice header"),
                        };
                        let (sps_h, pps_h) = match (sps_h.clone(), pps_h.clone()) {
                            (Some(s), Some(p)) => (s, p),
                            _ => panic!("{name}: SPS/PPS not ready for a picture"),
                        };
                        let dpb = dpb.get_or_insert_with(|| {
                            let mut d = H265Dpb::new(MAX_DPB_SLOTS);
                            d.set_max_num_reorder_frames(sps_h.max_num_reorder_pics[0] as u32);
                            d
                        });
                        store.ensure(dpb.slots().len());

                        decode_and_compare(
                            name,
                            au_idx,
                            cpp_ctx,
                            au,
                            &slices_owned,
                            &sps_h,
                            &pps_h,
                            &first_info,
                            dpb,
                            &mut store,
                            coded_w,
                            coded_h,
                            chroma_w,
                            chroma_h,
                            bps,
                            ystride,
                            cstride,
                        );
                        parse_offset = au_end;
                        au_idx += 1;
                        got_au = true;
                        break 'au;
                    }
                    Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break 'au,
                    Err(e) => panic!("{name}: parse error: {e}"),
                }
            }
            if !got_au {
                // No slice produced from the remaining data; stop.
                break;
            }
        }

        unsafe { ffi::hevcdec_destroy(cpp_ctx) };
        println!("{name}: {au_idx} access units verified (Rust == C++)");
    }

    /// Apply a newly parsed SPS: set up the picture layout and (re)create the
    /// Rust DPB. Mirrors `SoftwareH265Decoder::on_sps`.
    #[allow(clippy::too_many_arguments)] // test harness
    fn on_sps(
        sps: &H265Sps,
        dpb: &mut Option<H265Dpb>,
        sps_h: &mut Option<H265Sps>,
        coded_w: &mut u32,
        coded_h: &mut u32,
        chroma_w: &mut u32,
        chroma_h: &mut u32,
        bps: &mut usize,
        ystride: &mut usize,
        cstride: &mut usize,
    ) {
        let (sw, sh) = match sps.chroma_format_idc {
            0 => (1, 1),
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        let w = sps.pic_width_in_luma_samples as u32;
        let h = sps.pic_height_in_luma_samples as u32;
        let bd = 8 + sps.bit_depth_luma_minus8 as u32;
        *coded_w = w;
        *coded_h = h;
        *chroma_w = w.div_ceil(sw);
        *chroma_h = h.div_ceil(sh);
        *bps = if bd > 8 { 2 } else { 1 };
        *ystride = (w as usize).div_ceil(16) * 16;
        *cstride = (*chroma_w as usize).div_ceil(16) * 16;

        let num_slots = (1 + sps.max_dec_pic_buffering_minus1[0] as usize).clamp(4, MAX_DPB_SLOTS);
        if dpb.is_none() {
            let mut d = H265Dpb::new(num_slots);
            d.set_max_num_reorder_frames(sps.max_num_reorder_pics[0] as u32);
            *dpb = Some(d);
        }
        *sps_h = Some(sps.clone());
    }

    #[test]
    fn tier_f_main() {
        run_stream("main");
    }

    #[test]
    fn tier_f_main10() {
        run_stream("main10");
    }

    #[test]
    fn tier_f_cra() {
        run_stream("cra");
    }

    #[test]
    fn tier_f_msp() {
        run_stream("msp");
    }
}

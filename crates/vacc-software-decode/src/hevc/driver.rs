//! Picture-level decode driver.
//!
//! Drives the ported hevc kernels from vacc-parser output (mapped SPS/PPS/
//! slice headers + DPB-resolved reference lists), mirroring the C++
//! `Decoder::decode_picture` pipeline: per-slice-segment CABAC decode, then
//! deblocking and SAO once after all slices.

use crate::hevc::bitreader::{BitstreamReader, extract_rbsp_with_epb};
use crate::hevc::cabac::{CabacContext, CabacEngine};
use crate::hevc::cabac_tables::NUM_CABAC_CONTEXTS;
use crate::hevc::coding_tree::{CuInfo, DecodingContext, SaoParams, decode_slice_segment_data};
use crate::hevc::deblocking::{DeblockCtx, DeblockCu, DeblockSliceParams, MotionInfo, apply_deblocking};
use crate::hevc::inter_prediction::{DpbView, PlaneView, RefPic};
use crate::hevc::picture::{Picture, PuMotionInfo};
use crate::hevc::sao::{CuGrid, SaoCtx, apply_sao};
use crate::hevc::transform::ScalingListData;
use crate::hevc::types::{ChromaFormat, Pps, Plane, Sps, SliceHeader, SliceType, Tiles};

/// One slice segment input to [`decode_picture`].
pub struct SliceInput<'a> {
    /// Full NAL bytes including the 2-byte header.
    pub nal: &'a [u8],
    /// Mapped slice header (dependent-slice inheritance already applied).
    pub sh: SliceHeader,
    /// Effective per-slice deblocking parameters.
    pub deblock: DeblockSliceParams,
    /// Parsed slice-header coded bit size (payload-relative, EPB-inclusive
    /// domain — the parser's `header_bit_size`).
    pub header_bit_size: u16,
}

/// One entry of a resolved reference list (from `H265Dpb::build_ref_lists`).
#[derive(Clone, Copy, Debug)]
pub struct RefListEntry {
    /// Picture store slot; -1 = reference not in the DPB.
    pub slot: i32,
    pub poc: i32,
    /// Long-term reference classification for the current access unit.
    pub long_term: bool,
}

/// Owned pool of fully decoded pictures, indexed by DPB slot.
#[derive(Default)]
pub struct PictureStore {
    pics: Vec<Option<Picture>>,
}

impl PictureStore {
    pub fn new(cap: usize) -> Self {
        Self { pics: vec![None; cap] }
    }

    pub fn cap(&self) -> usize {
        self.pics.len()
    }

    pub fn ensure(&mut self, cap: usize) {
        if self.pics.len() < cap {
            self.pics.resize_with(cap, || None);
        }
    }

    pub fn store(&mut self, slot: usize, pic: Picture) {
        // The DPB can assign slots beyond the initial capacity (it is bounded
        // only by stream length, not a fixed size), so grow on demand rather
        // than silently dropping the picture — a dropped reference would be
        // read back as "unavailable" and corrupt later predictions.
        if slot >= self.pics.len() {
            self.pics.resize_with(slot + 1, || None);
        }
        self.pics[slot] = Some(pic);
    }

    /// Remove and return the picture in a slot (eviction cleanup).
    pub fn take(&mut self, slot: usize) -> Option<Picture> {
        self.pics.get_mut(slot)?.take()
    }

    pub fn get(&self, slot: usize) -> Option<&Picture> {
        self.pics.get(slot)?.as_ref()
    }
}

/// Derive the collocated picture and NoBackwardPredFlag (C++
/// `DPB::derive_colpic`, §8.3.5). Returns `(store slot of ColPic, -1 = none,
/// no_backward_pred_flag)`.
fn derive_colpic(
    sh: &SliceHeader,
    cur_poc: i32,
    l0: &[RefListEntry],
    l1: &[RefListEntry],
) -> (i32, bool) {
    if !sh.slice_temporal_mvp_enabled_flag || sh.slice_type == SliceType::I {
        return (-1, false);
    }

    let col = if sh.slice_type == SliceType::B && !sh.collocated_from_l0_flag {
        l1.get(sh.collocated_ref_idx as usize)
    } else {
        l0.get(sh.collocated_ref_idx as usize)
    };
    let col_slot = col.map(|e| e.slot).unwrap_or(-1);

    let mut no_backward = true;
    for e in l0.iter().chain(l1) {
        if e.slot >= 0 && e.poc > cur_poc {
            no_backward = false;
            break;
        }
    }
    (col_slot, no_backward)
}

/// Decode one picture from parsed access-unit data.
///
/// `refs_l0`/`refs_l1` are the resolved reference lists (`H265Dpb::
/// build_ref_lists`), `store` holds the previously decoded pictures indexed
/// by DPB slot. Returns the fully filtered picture (planes + motion info +
/// ref POC snapshots for TMVP).
#[allow(clippy::too_many_arguments)] // mirrors the C++ pipeline shape (sets + scaling + refs + store)
pub fn decode_picture(
    sps: &Sps,
    pps: &Pps,
    slices: &[SliceInput<'_>],
    refs_l0: &[RefListEntry],
    refs_l1: &[RefListEntry],
    store: &PictureStore,
    cur_poc: i32,
    sps_scaling_enabled: bool,
    sps_scaling: &'_ ScalingListData,
    pps_scaling_present: bool,
    pps_scaling: &'_ ScalingListData,
) -> Result<Picture, String> {
    let pic_w = sps.pic_width_in_luma_samples;
    let pic_h = sps.pic_height_in_luma_samples;
    let fmt = if sps.chroma_array_type == 0 {
        ChromaFormat::Monochrome
    } else {
        ChromaFormat::Yuv420
    };

    let mut pic = Picture::default();
    pic.allocate(pic_w, pic_h, fmt, sps.bit_depth_y, sps.bit_depth_c);
    pic.poc = cur_poc;

    // Grids (mirrors C++ decoder.cpp allocation/initialization).
    let grid_w = pic_w / sps.min_tb_size_y;
    let grid_h = pic_h / sps.min_tb_size_y;
    let grid = (grid_w * grid_h) as usize;
    let min_cb_w = pic_w >> sps.min_cb_log2_size_y;
    let min_cb_h = pic_h >> sps.min_cb_log2_size_y;
    let min_cbs = (min_cb_w * min_cb_h) as usize;
    let ctb_count = sps.pic_size_in_ctbs_y as usize;

    let mut cu = vec![CuInfo::default(); min_cbs];
    let mut intra_luma = vec![1i32; grid];
    let mut intra_chroma = vec![0i32; grid];
    let mut motion = vec![PuMotionInfo::default(); grid];
    let mut cbf = vec![0u8; grid];
    let mut log2_tu = vec![sps.ctb_log2_size_y as u8; grid];
    let mut edge_v = vec![0u8; grid];
    let mut edge_h = vec![0u8; grid];
    let mut sao = vec![SaoParams::default(); ctb_count];
    let mut slice_idx = vec![0u8; ctb_count];

    // Reference pool: compact list of RefPic views into the store.
    let mut pool: Vec<RefPic> = Vec::new();
    let mut slot_to_pool: Vec<i32> = vec![-1; store.cap()];
    for list in [refs_l0, refs_l1] {
        for e in list {
            if e.slot < 0 || slot_to_pool[e.slot as usize] >= 0 {
                continue;
            }
            let Some(refpic) = store.get(e.slot as usize).map(|p| RefPic {
                poc: p.poc,
                used_for_short_term_ref: !e.long_term,
                used_for_long_term_ref: e.long_term,
                planes: [
                    Some(PlaneView {
                        data: &p.planes[0],
                        width: p.width[0],
                        height: p.height[0],
                        stride: p.stride[0],
                    }),
                    Some(PlaneView {
                        data: &p.planes[1],
                        width: p.width[1],
                        height: p.height[1],
                        stride: p.stride[1],
                    }),
                    Some(PlaneView {
                        data: &p.planes[2],
                        width: p.width[2],
                        height: p.height[2],
                        stride: p.stride[2],
                    }),
                ],
                motion_info: &p.motion_info_buf,
                motion_stride: p.motion_info_stride,
                ref_poc: [&p.ref_poc[0], &p.ref_poc[1]],
            }) else {
                continue; // reference missing from the store — treat as unavailable
            };
            slot_to_pool[e.slot as usize] = pool.len() as i32;
            pool.push(refpic);
        }
    }
    let list0: Vec<i32> = refs_l0.iter().map(|e| slot_to_pool.get(e.slot as usize).copied().unwrap_or(-1)).collect();
    let list1: Vec<i32> = refs_l1.iter().map(|e| slot_to_pool.get(e.slot as usize).copied().unwrap_or(-1)).collect();

    // §8.3.5: collocated picture (C++ derives once per picture from the
    // first slice header, non-I slices only).
    let first_sh = &slices[0].sh;
    let (col_slot, no_backward) = derive_colpic(first_sh, cur_poc, refs_l0, refs_l1);
    let col_pic_idx = if col_slot >= 0 {
        slot_to_pool.get(col_slot as usize).copied().unwrap_or(-1)
    } else {
        -1
    };

    let dpb = DpbView {
        pics: &pool,
        list0: &list0,
        list1: &list1,
        col_pic_idx,
        no_backward_pred_flag: no_backward,
    };

    // WPP context carryover across segments (the C++ ctx stays alive for the
    // whole picture; each independent segment still re-inits at its start).
    let mut wpp_saved = [CabacContext::default(); NUM_CABAC_CONTEXTS];
    let mut wpp_avail = false;

    for (s, seg) in slices.iter().enumerate() {
        let sh = &seg.sh;
        // C++ NalParser extracts RBSP from the NAL payload (after the 2-byte
        // header); all positions below are payload-relative.
        let (rbsp, epb) = extract_rbsp_with_epb(&seg.nal[2..]);
        // RBSP byte where slice data starts, in the coded (EPB-present) domain.
        // The parser's `header_bit_size` is the coded-bit position where the
        // header ends. C++ `byte_alignment()` ALWAYS consumes >=1 bit
        // (alignment_bit_equal_to_one) then pads to the byte boundary, so the
        // start byte is floor(header_bits/8) + 1 = (header_bits + 8) / 8 — NOT
        // ceil(header_bits/8). The two coincide unless the header length is an
        // exact multiple of 8, in which case ceil under-counts by one byte.
        // `seek_target` then subtracts the EPB bytes the header absorbed
        // (C++ fixpoint condition `ep < sh_coded + 2` — see tier_e /
        // hevc_test_api.cpp) to land in the RBSP (EPB-removed) domain.
        let sh_coded = (seg.header_bit_size as usize + 8) / 8;
        let epb_in_hdr = epb.iter().filter(|&&e| e < sh_coded + 2).count();
        let seek_target = sh_coded - epb_in_hdr;
        let mut reader = BitstreamReader::new(&rbsp);
        reader.seek_to_byte(seek_target);
        let mut cabac = CabacEngine::new(&mut reader);

        let (ok, wpp_out) = {
            let mut ctx = DecodingContext {
                sps,
                pps,
                sh,
                pic: &mut pic,
                dpb: &dpb,
                cabac: &mut cabac,
                sps_scaling_list_enabled: sps_scaling_enabled,
                sps_scaling_list: sps_scaling,
                pps_scaling_list_present: pps_scaling_present,
                pps_scaling_list: pps_scaling,
                cu_info: &mut cu,
                cu_info_stride: min_cb_w,
                intra_pred_mode_y: &mut intra_luma,
                intra_pred_mode_c: &mut intra_chroma,
                intra_pred_mode_stride: grid_w,
                motion_info: &mut motion,
                motion_info_stride: grid_w,
                cbf_luma_grid: &mut cbf,
                log2_tu_size_grid: &mut log2_tu,
                edge_flags_v: &mut edge_v,
                edge_flags_h: &mut edge_h,
                filter_grid_stride: grid_w,
                sao_params: &mut sao,
                sao_params_stride: sps.pic_width_in_ctbs_y,
                slice_idx: Some(&mut slice_idx),
                current_slice_idx: s as i32,
                qp_y_prev: sh.slice_qp_y,
                qp_y_prev_qg: sh.slice_qp_y,
                is_cu_qp_delta_coded: false,
                cu_qp_delta_val: 0,
                cu_x0: 0,
                cu_y0: 0,
                wpp_saved_contexts: wpp_saved,
                wpp_contexts_available: wpp_avail,
                wpp_enabled: true,
                mc_l0: Vec::new(),
                mc_l1: Vec::new(),
                mc_out: Vec::new(),
                fir_tmp: Vec::new(),
            };
            let (ok, _) = decode_slice_segment_data(&mut ctx, &epb, sh_coded);
            (ok, (ctx.wpp_saved_contexts, ctx.wpp_contexts_available))
        };
        if !ok {
            return Err(format!("slice segment {s} decode failed"));
        }
        wpp_saved = wpp_out.0;
        wpp_avail = wpp_out.1;
    }

    // Deblocking/SAO motion grid (built before `motion` is moved into the
    // picture for TMVP storage).
    let motion_db: Vec<MotionInfo> = motion
        .iter()
        .map(|m| MotionInfo {
            mv: m.mv,
            ref_idx: m.ref_idx,
            pred_flag: m.pred_flag,
        })
        .collect();

    // Motion info + ref POC snapshots for TMVP (C++ decoder.cpp after the
    // slice loop). Missing references store POC 0, mirroring the C++ DPB.
    pic.motion_info_buf = std::mem::take(&mut motion);
    pic.motion_info_stride = grid_w;
    pic.ref_poc[0] = refs_l0.iter().map(|e| if e.slot >= 0 { e.poc } else { 0 }).collect();
    pic.ref_poc[1] = refs_l1.iter().map(|e| if e.slot >= 0 { e.poc } else { 0 }).collect();

    // §8.7: in-loop filters — deblocking then SAO, once after all slices.
    let cu_db: Vec<DeblockCu> = cu
        .iter()
        .map(|c| DeblockCu {
            pred_mode: c.pred_mode as u8,
            qp_y: c.qp_y,
            is_pcm: c.is_pcm,
            transquant_bypass: c.cu_transquant_bypass,
        })
        .collect();
    let is_pcm_db: Vec<u8> = cu.iter().map(|c| c.is_pcm as u8).collect();
    let bypass_db: Vec<u8> = cu.iter().map(|c| c.cu_transquant_bypass as u8).collect();
    let across_slices: Vec<bool> = slices.iter().map(|s| s.deblock.across_slices_enabled).collect();

    let tiles = Some(Tiles {
        tile_id: &pps.tile_id,
        ctb_addr_rs_to_ts: &pps.ctb_addr_rs_to_ts,
    });
    let sh_params: Vec<DeblockSliceParams> = slices.iter().map(|s| s.deblock).collect();
    let poc_l0: Vec<i32> = refs_l0.iter().map(|e| if e.slot >= 0 { e.poc } else { -999_999 }).collect();
    let poc_l1: Vec<i32> = refs_l1.iter().map(|e| if e.slot >= 0 { e.poc } else { -999_999 }).collect();

    let [p0, p1, p2] = &mut pic.planes;
    let mut planes = [
        Plane {
            data: p0,
            width: pic.width[0],
            height: pic.height[0],
            stride: pic.stride[0],
        },
        Plane {
            data: p1,
            width: pic.width[1],
            height: pic.height[1],
            stride: pic.stride[1],
        },
        Plane {
            data: p2,
            width: pic.width[2],
            height: pic.height[2],
            stride: pic.stride[2],
        },
    ];

    let db_ctx = DeblockCtx {
        pic_w,
        pic_h,
        bit_depth_y: sps.bit_depth_y,
        bit_depth_c: sps.bit_depth_c,
        pcm_filter_disabled: sps.pcm_loop_filter_disabled_flag,
        sub_w: sps.sub_width_c,
        sub_h: sps.sub_height_c,
        ctb_log2: sps.ctb_log2_size_y,
        ctbs_w: sps.pic_width_in_ctbs_y,
        chroma_array_type: sps.chroma_array_type,
        loop_filter_across_tiles: pps.loop_filter_across_tiles_enabled_flag,
        tiles: tiles.as_ref(),
        pps_cb_qp_offset: pps.pps_cb_qp_offset,
        pps_cr_qp_offset: pps.pps_cr_qp_offset,
        slice_idx: Some(&slice_idx),
        sh: &sh_params,
        cu: &cu_db,
        cu_stride: min_cb_w,
        min_cb_log2: sps.min_cb_log2_size_y,
        grid_stride: grid_w,
        motion: &motion_db,
        cbf_luma: &cbf,
        log2_tu_size: &log2_tu,
        edge_v: &edge_v,
        edge_h: &edge_h,
        poc_l0: &poc_l0,
        poc_l1: &poc_l1,
    };
    apply_deblocking(&db_ctx, &mut planes);

    let sao_ctx = SaoCtx {
        sao_enabled: sps.sample_adaptive_offset_enabled_flag,
        ctb_size: 1 << sps.ctb_log2_size_y,
        sub_w: sps.sub_width_c,
        sub_h: sps.sub_height_c,
        num_comp: if sps.chroma_array_type == 0 { 1 } else { 3 },
        bit_depth_y: sps.bit_depth_y,
        bit_depth_c: sps.bit_depth_c,
        pcm_filter_disabled: sps.pcm_loop_filter_disabled_flag,
        pic_w,
        pic_h,
        ctbs_w: sps.pic_width_in_ctbs_y,
        ctbs_h: sps.pic_height_in_ctbs_y,
        transquant_bypass_enabled: pps.transquant_bypass_enabled_flag,
        loop_filter_across_tiles: pps.loop_filter_across_tiles_enabled_flag,
        tiles: tiles.as_ref(),
        slice_idx: Some(&slice_idx),
        slice_across_slices: &across_slices,
        sao_params: &sao,
        sao_stride: sps.pic_width_in_ctbs_y,
        cu: &CuGrid {
            is_pcm: &is_pcm_db,
            bypass: &bypass_db,
            stride: min_cb_w,
            min_cb: 1 << sps.min_cb_log2_size_y,
        },
    };
    apply_sao(&sao_ctx, &mut planes);

    Ok(pic)
}

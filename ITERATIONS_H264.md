# H.264 Pixel-Perfect Iteration Log

Goal: H.264 Baseline / Constrained-Baseline / Main / High pixel-perfect vs ffmpeg (yuv420p, 30 frames)
on Vulkan, NVDEC, VAAPI. ONE common parser, DPB manager, POC calculator, ref-list builder.

Workdir: /home/atkachenko/apps/vacc
Verify: `python3 /tmp/matrix.py h264` (cells VK/NV/VA x h264_{baseline,constrained_baseline,main,high})
Full matrix (regression check): `python3 /tmp/matrix.py`
Build: `cargo build --release --examples`

Samples: assets/samples/h264_{baseline,constrained_baseline,main,high}.h264 (640x360, yuv420p, crop_bottom=4)
Ground truth: /tmp/cpp_run/cpp_full.txt (C++ 30-frame dump, h264_main), /tmp/parse_h264_v2.py (python slice-header parser)
C++ oracle: /home/atkachenko/apps/Vulkan-Video-Samples (VulkanH264Parser.cpp, VulkanVideoParser.cpp)

## Baseline 2026-08-25 (WIP uncommitted state)
| Cell | Status |
|---|---|
| VK/h264_baseline | 1/30 md=150 (REGRESSION: was 30/30 before WIP) |
| VK/h264_constrained_baseline | 1/30 md=150 (REGRESSION: was 30/30) |
| VK/h264_main | 5/30 md=56 (was 4/30) |
| VK/h264_high | 5/30 md=55 (was 4/30) |
| NV/h264 all 4 | 0/30 md~180 (systematic error even on IDR, ~23dB) |
| VA/h264_baseline | FAIL init "requested VAProfile is not supported" |
| VA/h264_constrained_baseline | 0/30 md=179 |
| VA/h264_main, high | NO_OUTPUT (display 640x304 crop bug, "No free surfaces available") |

## Architecture (current)
- Common parser: vacc-parser/src/h264.rs (H264Parser: SPS/PPS/slice header incl. ref_pic_list_modification, dec_ref_pic_marking, pred_weight_table). Used by VK, NVDEC, VAAPI.
- POC: Vulkan has H264PocState in vacc-vulkan/src/access_unit.rs (type 0 only?); NVDEC has PocCalculator in nvdec-decode/src/poc.rs (types 0/1/2); VAAPI has inline POC in vaapi-decode/src/decoder.rs. NOT common.
- DPB: Vulkan has WIP H264Dpb in vacc-vulkan/src/h264_dpb.rs (C++ state-machine port); NVDEC has nvdec-decode/src/dpb.rs; VAAPI has H264Dpb in vaapi-decode/src/decoder.rs:188. NOT common.
- Ref-list order: H264Dpb::get_references() returns DPB SLOT order, NOT spec 8.2.3.1+8.2.3.2 order. Known root cause of VK main/high failures (see memory h264_vk_reflist_bug.md).
- H264MmcoCommand: enum in vacc-vulkan/src/access_unit.rs:45.
- Crate deps: core <- parser <- {vulkan, nvdec(->vulkan), vaapi}. Shared H264 decode-state code should live in vacc-parser (all 3 backends depend on it) or vacc-core.

## Plan
1. Iter 1: shared foundation (refactor, no behavior change): H264Dpb + PocCalculator + ref-list builder (8.2.3.1+8.2.3.2) into common crate; re-exports.
2. Iter 2: VK H264 30/30 all 4 samples.
3. Iter 3: NVDEC H264 30/30 all 4 samples.
4. Iter 4: VAAPI H264 30/30 all 4 samples.
5. Iter 5: full-matrix regression check + cleanup.

## Iteration log
### Iter 1 (2026-08-25): shared H.264 decode-state foundation (refactor, no behavior change)
- **In**: baseline matrix as in "Baseline 2026-08-25" table.
- **Done** (all in `vacc-parser`, re-export shims keep old paths compiling):
  - `src/h264_dpb.rs` (NEW): H264Dpb + H264DpbSlot + CurrentPic + marking consts (now `pub`) moved from vacc-vulkan; `H264MmcoCommand` enum moved here from access_unit.rs. VACC_DBG_H264 prints kept. Added `H264Dpb::build_ref_lists()` convenience (maps slots -> DpbRefState -> builder).
  - `src/h264_poc.rs` (NEW): PocCalculator moved from nvdec-decode (body byte-identical).
  - `src/h264_reflist.rs` (NEW): spec 8.2.3.1+8.2.3.2 ref-list builder (`build_ref_pic_lists`, `RefPic`, `RefPicLists`, `DpbRefState`) + 7 unit tests.
  - Shims: vacc-vulkan `h264_dpb.rs` + `access_unit.rs` (enum -> re-export), nvdec-decode `poc.rs`.
- **Deviation**: mission described 8.2.3.2 as "op=0 move-to-end / op=1 swap" (that is HEVC semantics). Implemented actual H.264 8.2.3.2 per spec + C++ oracle (encoder DPB `VkEncoderDpbH264.cpp`, the only in-repo implementation; decoder-side `reference_picture_list_initialization_*` are declared in VulkanH264Decoder.h but NOT implemented in the repo's .cpp): idc0 = entry at pos (abs_diff_pic_num_minus1+1) moves to PicNumIdx; idc1/2 = assign long-term ref (LongTermFrameIdx == long_term_pic_num / 0) at PicNumIdx (pure assignment); idc3 = stop.
- **Verify**: build green; parser lib tests 53/53 (incl. 7 reflist); h264 matrix UNCHANGED (VK 1/1/5/5, NV 0x4, VA FAIL/0/NO_OUTPUT/NO_OUTPUT — exact match incl. md values).
- **Pre-existing failures (NOT caused by this refactor)**: `vacc-parser` integration test `h265_cref` (pic 0 lists_mod mismatch vs cuvid GT; H.265-only, files unmodified vs HEAD) and nvdec `cuvid_dpb_parity` (pic 1 POC got 2 vs cuvid 6; POC code byte-identical to HEAD).
- **Next (iter 2)**: wire `build_ref_lists` into VK H264 decode path (replace `get_references()` slot-order list with spec-order list for refIdxL0/L1) -> target VK 30/30 on all 4 samples.


### Iter 2 prep — investigation (coordinator, 2026-08-25)
SPS facts (verified with parse_sps example + python):
- h264_baseline / constrained_baseline: profile 66, **poc_type=2** (NO POC bits in slice header; POC derived from FrameNum per 8.2.1), log2_max_frame_num_minus4=0, max_num_ref_frames=3, no B frames.
- h264_main: profile 77, poc_type=0, log2_max_poc_lsb_minus4=2, max_num_ref_frames=4, crop_bottom=4 (8px).
- h264_high: profile 100, poc_type=0, log2_max_poc_lsb_minus4=2, max_num_ref_frames=4, crop_bottom=4.

ROOT CAUSE CANDIDATES (VK):
1. **Baseline/CB 1/30 regression**: vacc-vulkan/src/access_unit.rs `parse_h264_slice_header` (~line 150-266) is a SIMPLIFIED slice parser that UNCONDITIONALLY reads pic_order_cnt_lsb bits (lines 188-189) — no poc_type branch. For poc_type=2 streams this reads 4 non-existent bits → desync → wrong ref-fields/MMCO parse → wrong DPB state → wrong slots. Also: reads only ONE ref_pic_list_modification entry per list (no idc=3 loop), no pred_weight_table, no redundant_pic_cnt, `idc > 2 → return None` (idc=3 is a valid terminator!).
   FIX DIRECTION: use the COMMON H264Parser::parse_slice_header (vacc-parser/src/h264.rs:661, handles poc_type 0/1/2, full rplm, pred_weight_table, redundant_pic_cnt) in the VK AU-extraction path instead of the simplified one (user requirement: ONE common parser). AU extraction loop is access_unit.rs ~850-1000 (in-band SPS/PPS via parse_h264_inband_params → ExtractedItem::ParameterSet; slice NALs → parse_h264_slice_header at line 918). AccessUnit must then carry num_ref_idx_l0/l1_active_minus1 + ref_pic_list_modification_l0/l1 (parser SliceHeader already has all of it).
2. **Main/high 5/30**: ref-list ORDER (slot order vs spec 8.2.3.1+8.2.3.2 order). Wire H264Dpb::build_ref_lists() (added iter 1) into decoder.rs: h264_current_refs (set at decoder.rs:598, used in record_decode_command ~2549 for pReferenceSlots) must become spec order.
3. POC: Vulkan path uses H264PocState (access_unit.rs:168, type-0-only inline msb calc). Should switch to shared PocCalculator (vacc-parser/src/h264_poc.rs, types 0/1/2) — "one common POC calculator".

### Iter 2 (2026-08-26): VK H.264 IDR fix + common foundation wiring
- **In**: VK H.264 baseline/CB byte-perfect on IDR; main/high IDR decoded as a plausible but MUCH BRIGHTER image (our Y mean ~217 vs ffmpeg ~97). P/B frames wrong on all 4 (1/30).
- **ROOT CAUSE (IDR brightness, fixed)**: `convert_h264_pps` (vacc-vulkan/src/h264.rs) did NOT set `entropy_coding_mode_flag`. The driver picks CABAC vs CAVLC slice parsing from the **session PPS** (this flag), not the slice header. main/high are CABAC (flag=1); with the flag unset the driver misparsed CABAC bitstreams as CAVLC → brighter image. baseline/CB are CAVLC (flag=0) so were unaffected.
  - **Non-standard ABI (critical)**: system + C++-oracle Vulkan headers AND the `ash` crate put `entropy_coding_mode_flag` at bit 0 of `StdVideoH264PpsFlags` (before `weighted_pred_flag`) and make `StdVideoH264LevelIdc` **0-based** (level 3.0 = enum 7, not raw 30). C++ oracle sets `pps->flags.entropy_coding_mode_flag` (VulkanH264Parser.cpp).
- **Done**:
  - IDR fix (vacc-vulkan/src/h264.rs): set `entropy_coding_mode_flag` + `bottom_field_pic_order_in_frame_present_flag` in `convert_h264_pps`; added `h264_level_idc_to_vulkan()` (raw level_idc → 0-based `StdVideoH264LevelIdc`, matching C++ `levelIdcToVulkanLevelIdcEnum`) used in `convert_h264_sps`; VACC_DEBUG full-field SPS/PPS dump.
  - Foundation wiring (ONE common parser/POC/ref-list across backends):
    - vacc-parser/src/lib.rs: declared `pub mod h264_dpb; h264_poc; h264_reflist;`.
    - vacc-parser/src/h264.rs: fixed MMCO value bit-counts (op 3 reads 2 ue(v), op 5 reads 0) + added `H264Parser::set_sps/set_pps`.
    - vacc-vulkan/src/access_unit.rs: H.264 slice branch now uses common `H264Parser::parse_slice_header` + `PocCalculator::calculate` (replaces the simplified local `parse_h264_slice_header` + inline POC); AccessUnit carries `h264_slice: Option<SliceHeader>`; in-band SPS/PPS updates feed `h264_parser.set_sps/set_pps`.
    - vacc-vulkan/src/decoder.rs `record_h264_decode`: reference slots now ordered via common `build_ref_pic_lists` (L0 then L1, PicNum/POC order) instead of generic DpbManager slot order.
- **Verify**: `cargo build --release --examples` green. `python3 /tmp/idr_check.py VK` → **all 4 IDR_OK** (345600 B identical). Regression clean: `matrix.py VK/av1_main` 30/30 PERFECT, `VK/vp9_profile0` 30/30 PERFECT. Parser lib tests 53/53.
- **REMAINING (P/B, NOT fixed by this iter)**: `matrix.py VK/h264_*` still **1/30** (only IDR correct; P-frame meandiff ~104, maxdiff ~233). Debug (VACC_DBG_SPS) shows references are CORRECT: right count (baseline 3, main/high 4), right FrameNum/PicOrderCnt, right L0-then-L1 order, `is_intra=false` for P, `ref_slots` matches. So the bug is NOT the ref-list order, POC, or SPS/PPS — it is in the **reference-image / decode path** (reference image content or how the driver consumes pReferenceSlots). Next: compare a P-frame reference image readback vs the stored IDR image; inspect `VkVideoDecodeH264PictureInfoKHR` pNext / `StdVideoDecodeH264ReferenceInfo.ReferenceIdx` vs C++ oracle; check image layout barriers for ref slots.
- **Pre-existing (NOT mine)**: parser integration test `h265_parser_matches_cuvid_ground_truth` (log2_par_merge_minus2; H.265-only, untouched).

### Iter 3 (2026-08-26): NVDEC H.264 IDR fix (display_area) + POC unification
- **In**: NV/h264 all 4 cells IDR_BAD (maxdiff ~188, systematic error even on the intra-only IDR frame); 0/30 on the 30-frame matrix.
- **ROOT CAUSE (IDR, fixed)**: `CUVIDDECODECREATEINFO.display_area` was set to the DISPLAY size (640x360) instead of the CODED size (640x368). When `display_area != ulTargetWidth/Height`, cuvid SCALES the output: measured our NVDEC row y == ffmpeg row y*(360/368) — content stretched 368/360 into the 368-row surface, and we read the top 360 rows. This is a systematic geometric error present even on the intra-only IDR. The working H.265 NVDEC path already sets `display_area` = coded size with an explicit comment (nvdec-decode/src/h265.rs:1045-1053); the H.264 path did not.
- **Done**:
  - `create_decoder` (nvdec-decode/src/decoder.rs:685): `display_area = {0,0,coded_width,coded_height}` (coded 640x368), NOT the display size. The picture crop is still applied at readback via `self.display_area` (extract_frame).
  - `pic_order_present_flag` (nvdec-decode/src/picparams.rs:92): was hardcoded `0` ("EXPERIMENT: match GT"); now `poc_type != 2 ? 1 : 0` (correct for main/high poc_type=0; baseline/CB poc_type=2 unchanged).
  - POC unification (ONE common POC across backends): nvdec-decode/src/poc.rs is now a re-export shim of `vacc_parser::h264_poc::PocCalculator` (the common impl, which also adds FrameNum-wrap monotonicity for type 2). decoder.rs imports unchanged (`crate::poc::PocCalculator`).
  - `num_ref_idx_l0/l1_active_minus1` (picparams.rs): now taken from the slice header (parser already applies `num_ref_idx_active_override_flag`, falling back to PPS defaults) instead of always the PPS defaults. (No measurable effect on the 4 samples; more correct.)
  - Test update: `test_poc_type2_frame_num_wrap` now expects the common calculator's MONOTONIC type-2 POC across a FrameNum wrap ((256+5)*2=522, (256+10)*2=532) instead of the raw spec POC (10, 20). Monotonic POCs are required for correct presentation-order sorting in the reorder buffer (raw type-2 POCs wrap with FrameNum and would misorder frames after a wrap; the decoder's `unwrapped_poc` only unwraps type 0).
- **DPB deviation (documented)**: NVDEC keeps its local `NvdecDpbManager` (nvdec-decode/src/dpb.rs) rather than the common `H264Dpb`. Reason: NVDEC must map DPB slots to CUVID `PicIdx` decode-surface indices (ring-buffer wrap at `ulNumDecodeSurfaces`, surface recycling, `used_for_reference`/`not_existing` cuvid conventions) — a non-trivial mapping the common `H264Dpb` (slot-based, no surface indices) does not provide. NvdecDpbManager already consumes the common parser's slice headers (MMCO ops, frame_num) and the POC now comes from the common `PocCalculator`. Unifying the DPB is deferred.
- **Verify**:
  - `cargo build --release --examples` green.
  - `python3 /tmp/idr_check.py NV` → **all 4 IDR_OK** (345600 B identical).
  - 30-frame matrix: **NV/h264_baseline 30/30 PERFECT, NV/h264_constrained_baseline 30/30 PERFECT** (were 0/30). NV/h264_main 1/30 md=240, NV/h264_high 1/30 md=220 (still failing — see REMAINING).
  - No regressions: NV/vp9_profile0 30/30 PERFECT (unchanged); NV/h265 cells unchanged (0/30 md~163-167, pre-existing).
  - `cargo test -p nvdec-decode`: 54 unit + 10 cuvid_comparison + 14 poc_calculation all PASS. `cuvid_dpb_parity` still FAILS for the KNOWN stale-GT reason (pic 1 POC got 2 vs hardcoded cuvid 6; our POC pic1=2 is spec-correct for poc_type=2 frame_num=1; the GT is the defect — NOT "fixed" to match the stale GT).
- **REMAINING (main/high P/B, NOT fixed by this iter)**: NV/h264_main & high still 1/30 (only IDR correct). These two samples differ from baseline/CB in exactly three ways: **poc_type=0** (vs 2), **B-frames** (vs none), and **weighted prediction** (weighted_pred_flag=1, weighted_bipred_idc=2). Debug (NVDEC_DUMP_PARAMS) shows our picparams look correct: POC values (0,4,2,6,14,10,8,12,22,...) match the spec type-0 calc; DPB reference sets are correct (right pictures, right FOC, non-ref B-frames excluded); bitstream data correct. So the bug is in how cuvid consumes the picparams for B-frames / poc_type=0 / weighted-pred — most likely the ref-list construction (cuvid builds L0/L1 from the DPB + slice-header ref_pic_list_modification) or weighted-pred weight application. Next: dump cuvidParser's own CUVIDH264PICPARAMS for h264_main (adapt decode_nvdec_vp9_cuvid.rs to H.264) and diff field-by-field against ours to isolate the mismatch.

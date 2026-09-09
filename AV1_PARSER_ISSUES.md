# AV1 parser issues — ALL FIXED (2026-09-08)

Audit of `crates/vacc-parser/src/av1.rs` against the AV1 bitstream spec, cros-codecs
(`src/codec/av1/parser.rs`), and the NVIDIA `VulkanAV1Decoder.cpp` original that this file ports.
Original audit date: 2026-09-07. **All 10 issues fixed and committed on 2026-09-08**, each
verified with a regression test (synthetic bitstreams in `crates/vacc-parser/tests/av1_inline.rs`
where real samples don't exercise the path) and the full pixel-perfect verification matrix
(`verify-all.py --samples av1 --max-frames 300`: 12/12 PASS on vulkan/vaapi/nvdec).

Fix commits:
| Issue | Commit |
|---|---|
| 1 + 2 (timing reads) | `80bebe9` |
| 3 (film grain) | `3726ffc` |
| 4 (signed seg features + keyframe DPB wipe) | `0cd6a0d` |
| 5 (CodedLossless from ALT_Q) | `a9e7411` (plus GM-identity prerequisite `625e6bb`) |
| 6 (Annex B unit accounting + probe) | `57311fb` |
| 7 (per-ref delta_frame_id_minus_1) | `c890ddb` |
| 8 (spec LR unit sizes + backend conversions) | `d3cdbb0` |
| 9 (show-existing temporal_point_info) | `e0ea303` |
| 10 (segmentation feature clipping) | `eb1f985` |

Overall: `av1.rs` is a faithful port of the NVIDIA C++ reference on the core structural path —
OBU headers, sequence header (incl. color config/mono_chrome), frame tag, short-signaling
reference derivation, spec 7.8 `set_frame_refs` (verified correct in both repos), quantization,
loop filter, CDEF, global motion all match. The problems are **missing bitstream reads** that
desync any stream carrying decoder-model timing, grain metadata, or frame-ID numbers — extremely
common in real content — plus two segmentation logic bugs and an incomplete Annex B state machine.

## HIGH severity

### 1. `temporal_point_info` (frame presentation time) never read — FIXED (`80bebe9`)
- Spec 5.9.2: after `show_frame`, when `show_frame && decoder_model_info_present_flag &&
  !equal_picture_interval`, `temporal_point_info()` = `frame_presentation_time f(n)`
  (n = `decoder_model_info` bits) MUST be read.
- vacc: `av1.rs:1029-1051` jumps straight from `show_frame` to `error_resilient_mode`.
- C++ original reads it (`VulkanAV1Decoder.cpp:1808, 1864`); cros-codecs reads it too.
- Impact: every VFR stream with timing info desyncs from this point on.
- Note: cros-codecs has its own bug here — `parser.rs:3384` uses the **inverted** condition
  (`equal_picture_interval` instead of `!equal_picture_interval`), so do NOT copy it blindly;
  its show-existing-frame path (`parser.rs:3316-3319`) has the correct condition.

### 2. `buffer_removal_time` block never read — FIXED (`80bebe9`)
- Spec: after `primary_ref_frame`, when `decoder_model_info_present_flag`:
  `buffer_removal_time_present_flag f(1)` + per-operation BRT values
  (with the `opPtIdc == 0 || (inTemporal && inSpatial)` gating).
- vacc: `av1.rs:1143-1170` jumps from `primary_ref_frame` directly to `refresh_frame_flags`.
- C++ reads it (`VulkanAV1Decoder.cpp:1968-1976`); cros-codecs is spec-exact (`parser.rs:3510-3528`).
- Impact: streams with BRT (most CBR/ABR AV1) desync.

### 3. `film_grain_params` never read — FIXED (`3726ffc`)
- Spec: `film_grain_params()` is the **last** element of `uncompressed_header`; when SPS
  `film_grain_params_present = 1` it starts with `apply_grain f(1)` + grain data.
- vacc: `av1.rs:1448-1449` hardcodes `fh.apply_grain = false` ("not present in our SPS") and
  never reads the block. No code path rejects a grain SPS (`film_grain_used` is set at
  `av1.rs:2285` but consumed nowhere).
- Impact is severe because `frame_header_size` (`av1.rs:1455`) feeds the **tile-data offset** for
  both HW consumers (`vk-video-vulkan/src/decoder.rs:2079-2081`, `nvdec-decode/src/av1.rs:1019-1021`)
  — grain streams get a wrong tile offset and decode failure.
- C++ parses it (`VulkanAV1Decoder.cpp:995-1010`); cros-codecs fully parses it (`parser.rs:3108-3248`).

### 4. Segmentation signed-feature bit width off-by-one — FIXED (`0cd6a0d`)
- Spec 5.9.14: signed features are `su(1 + bitsToRead)` — ALT_Q (bitsToRead=8) is **9 bits**.
- vacc: `av1.rs:1938-1942` calls `r.read_signed_bits(bits)`, which reads exactly `bits`
  two's-complement bits (`bitreader.rs:86-93`) — one bit short per enabled signed feature.
- Port regression: the C++ original's `ReadSignedBits(n)` reads **n+1** bits
  (`VulkanAV1Decoder.cpp:1294-1299`); cros-codecs is correct (`read_su(1 + bits_to_read)`,
  `parser.rs:2530-2533`).
- Other signed call sites in vacc were adjusted correctly during the port — `read_delta_q` uses
  `read_signed_bits(7)` = spec `su(1+6)` (`av1.rs:1842-1848`) and loop-filter deltas use
  `read_signed_bits(7)` (`av1.rs:2026-2035`) — so only the segmentation site was missed.
- Impact: desync whenever any signed segmentation feature is enabled.

### 5. CodedLossless computed from the wrong segment feature — FIXED (`a9e7411`)
- Spec (uncompressed header, per-segment quantization): `qindex += FeatureData[seg][SEG_LVL_ALT_Q]`
  — feature **0**.
- vacc: `av1.rs:1367-1386` tests `segment_feature_enabled[i] & (1 << 2)` and adds
  `segment_feature_data[i][2]` — index 2 is **SEG_LVL_ALT_LF_U** in the feature order vacc uses
  everywhere (parse loop `av1.rs:1929-1947`, bits `{8,6,6,6,6,3,0,0}`).
- Impact: wrong `coded_lossless`/`all_lossless` flips the early-returns in `parse_loop_filter`
  (`av1.rs:1997-2001`), `parse_cdef` (`av1.rs:2063-2075`), `parse_loop_restoration`
  (`av1.rs:2107-2110`) → desync whenever ALT_LF_U state diverges from ALT_Q.
- vacc matches **neither** the spec nor the C++ (C++ has its own quirk `(FeatureEnabled[i] & 0)` —
  always ignoring ALT_Q, `VulkanAV1Decoder.cpp:2172-2174`); cros-codecs is spec-correct via
  `get_qindex`/`SEG_LVL_ALT_Q=0` (`parser.rs:3707-3728`).

### 6. Annex B (start-code) unit accounting is an empty stub — FIXED (`57311fb`)
- vacc: `av1.rs:536-539` — the `if let StreamFormat::AnnexB { .. }` block in `read_obu` is **empty**;
  `temporal_unit_consumed`/`frame_unit_consumed` are never incremented with OBU bytes
  (`current_annexb_obu_length` only counts size-field header bytes).
- Impact: the unit-boundary checks (`av1.rs:556, 569`) never fire, and from the **second temporal
  unit onward** the new `temporal_unit_size`/`frame_unit_size` leb128s are never read — the next
  temporal-unit size is misread as an OBU length, desynchronizing **all** Annex B (start-code)
  streams. IVF samples are unaffected (no unit sizes).
- cros-codecs updates both counters with `obu_size` after each OBU (`parser.rs:1842-1848`).

## MEDIUM severity

### 7. Per-ref `delta_frame_id_minus_1` not read — FIXED (`c890ddb`)
- Spec: inside the 7-iteration ref loop, when `frame_id_numbers_present_flag`, each ref carries
  `delta_frame_id_minus_1 f(delta + 2)`.
- vacc: `av1.rs:1261-1278` reads only `ref_frame_idx[0..7]` in both signaling modes.
- Note: the one-shot `current_frame_id f(idLen)` right after `force_integer_mv` **is** read
  correctly (`av1.rs:1108-1114`) — only the per-ref part is missing.
- cros-codecs and C++ read per-ref deltas (`parser.rs:3581-3608`, `VulkanAV1Decoder.cpp:2056-2077`).
- Triggers only on SPSes with frame-ID numbers enabled (low-overhead / multi-layer encodes).

### 8. Loop-restoration unit size stored non-spec — FIXED (`d3cdbb0`)
Note: the original "NVDEC always falls back to 32x32" description was stale — by fix time the
NVDEC consumer was a direct copy that happened to be correct for luma codes; the real bug was the
non-spec stored representation plus the C++-inherited double chroma shift. The parser now stores
spec pixel sizes ({64,128,256} luma; single-shift chroma) and each backend converts: NVDEC and
Vulkan use code log2(px)-5 (0:32, 1:64, 2:128, 3:256 — confirmed against cuviddec.h and the AV1
encoder sample), VAAPI derives raw spec shifts.
- Bit reads are spec-correct in both repos (spec 5.9.20). But vacc stores a non-spec internal
  representation: `loop_restoration_size[0] = 1 + lr_unit_shift` ∈ {1,2,3}, with the
  C++-inherited double shift at `av1.rs:2158`.
- The NVDEC consumer matches on pixel sizes `{32,64,128,256}` and falls through to `_ => 0`
  (`nvdec-decode/src/av1.rs:643-649`) — **always** 32x32 for LR frames.
- cros-codecs stores spec sizes (`RESTORATION_TILESIZE_MAX=256 >> (2-shift)` → {64,128,256}).
- The Vulkan consumer passes the raw value into `StdVideoAV1LoopRestoration`
  (`vk-video-vulkan/src/decoder.rs:2051-2056`), behavior-identical to C++
  (`VulkanAV1Decoder.cpp:1538, 1555-1556`) — so this is a representation mismatch in the new NVDEC
  integration, not a porting error.

## LOW severity

### 9. show-existing-frame path truncation — FIXED (`e0ea303`)
- On `show_existing_frame`, vacc returns after `frame_to_show_map_idx f(3)` (`av1.rs:1018-1021`),
  skipping `temporal_point_info` and `display_frame_id f(idLen)`.
- Harmless for HW decode (frame not decoded, `frame_header_size=0`), but the display-ID
  conformance check cros-codecs performs (`parser.rs:3325-3332`) is absent.

### 10. No clipping of segmentation feature values — FIXED (`eb1f985`)
- Spec applies `Clip3(-limit, limit, ...)` / `Clip3(0, limit, ...)`; vacc stores raw values
  (`av1.rs:1938-1943`). C++ clamps (`VulkanAV1Decoder.cpp:1396, 1399`); cros-codecs clips
  (`parser.rs:2532, 2536-2542`). Only observable on non-conforming streams.

## INFO / design differences (no action required)

- **Spec version**: both repos track the *current* AV1 spec, not AV1 1.0.0: `REFS_PER_FRAME=7`
  (B3 removed), `TOTAL_REFS_PER_FRAME=8`, `NUM_REF_FRAMES=8`. Superres constants in the current
  spec are `SUPERRES_NUM=8`, `SUPERRES_DENOM_MIN=9`; vacc's raw `coded_denom` passthrough
  (`av1.rs:1193`) is bit-compatible either way.
- **Architecture**: vacc is a HW-decode metadata extractor — top-level `parse()` emits
  `ParameterSet`/`Slice` only; the frame header is parsed on demand by consumers and
  `frame_header_size` (byte count, `av1.rs:290-293, 1455`) locates tile data. cros-codecs is a
  monolithic state machine with full conformance validation (`display_frame_id` checks,
  `ref_valid` marking).
- **OBU level**: vacc hard-errors on low-overhead OBUs without a size field; cros-codecs
  `assert!`s (release no-op) and implements spec `drop_obu()` operating-point filtering. vacc
  resyncs byte-by-byte on OBU errors (`av1.rs:2368-2371`).
- **DPB state**: vacc keeps per-frame-buffer arrays (`ref_segmentation`, `ref_global_models`,
  `last_cdef`, `ref_loop_filter`, `ref_frame_sizes`, `ref_frame_order_hints`) mirroring the C++
  buffer pool; cros-codecs uses `ReferenceFrameInfo[8]`.

## Verified CORRECT in both repos (do not "fix")

- Spec 7.8 `set_frame_refs` reference-list update (the critical area): vacc `av1.rs:1461-1581` +
  `update_ref_frames` `av1.rs:2237-2266`; cros-codecs `parser.rs:1554-1675` + `ref_frame_update`
  `parser.rs:3907-3956` — both match the spec.
- Short-signaling reference derivation; `mono_chrome` gating by profile 1
  (`av1.rs:881-884`; `parser.rs:1895-1899`).
- `current_frame_id` placement (`av1.rs:1108-1114`; `parser.rs:3435-3438`).
- Superres bit reads (`av1.rs:1188-1202`; `parser.rs:1528-1551`).

## Verification actually performed (2026-09-08)

- Every fix landed with a regression test in `crates/vacc-parser/tests/av1_inline.rs` (99 tests
total): real-sample end-to-end parses for the timing/grain/frame-ID paths, plus synthetic
  bitstreams (hand-built SPS + frame headers via a BitWriter) for CodedLossless/ALT_Q, Annex B
  two-temporal-unit streams, loop-restoration unit sizes, and segmentation clipping. Each synthetic
  test was verified to FAIL on the pre-fix code.
- Pixel-perfect verification on RTX 3060 (GA106): `verify-all.py --samples av1 --max-frames 300`
  → 12/12 PASS (vulkan/vaapi/nvdec × av1_main, av1_high, av1_grain, av1_seg; professional is
  hw-unsupported on this GPU). Samples: aomenc main/high/grain + rav1e `av1_seg.ivf`
  (multi-OBU packets, GM, segmentation).
- Workspace-wide test suite: 358 passed, 0 failed.

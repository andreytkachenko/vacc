//! Common H.265/HEVC DPB management and reference-list computation.
//!
//! Single source of truth shared by all decode backends (Vulkan Video, VAAPI,
//! NVDEC), mirroring the H.264 split (`h264_dpb.rs` / `h264_reflist.rs`):
//!
//! - [`resolve_refs`] — pure spec computation (H.265 7.3.7 / 8.3.2 / 8.3.3):
//!   resolves the picture's short-term reference set (STRPS), long-term
//!   references (with cumulative DeltaPocMsbCycleLt), and constructs the
//!   final RefPicList0/RefPicList1 POC lists (truncation, padding, and
//!   ref_pic_lists_modification). Verified against FFmpeg
//!   (`ff_hevc_slice_rpl` / `decode_lt_rps`) and bit-level encoder output.
//! - [`H265Dpb`] — DPB slot state machine: NoRaslOutput reset, per-access-unit
//!   reference marking (used + future-use RPS entries), eviction of
//!   unreferenced pictures, slot allocation, and display-order bumping.
//!
//! List construction order (verified against FFmpeg and the NVIDIA cuvid
//! ground-truth dump for B slices):
//! - RefPicList0 initial = [S0 used..., S1 used..., LT used...]
//! - RefPicList1 initial = [S1 used..., S0 used..., LT used...] (B slices);
//!   for P slices RefPicList1 = RefPicList0.
//!
//! Reference matching: short-term refs match by full POC; long-term refs
//! match by full POC when delta_poc_msb_present_flag is set, otherwise by
//! pic_order_cnt_lsb only (FFmpeg `find_ref_idx` semantics).

use crate::h265::{H265ListModification, SliceHeaderInfo};
use vacc_core::picture::{H265ShortTermRefPicSet, H265Sps};

/// One short-term reference picture resolved from the RPS (used or not).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H265StRef {
    /// Derived full POC of the reference picture.
    pub poc: i32,
    /// UsedByCurrPicSxFlag — used as a reference by the current picture.
    pub used: bool,
}

/// One long-term reference with spec-resolved DeltaPocMsbCycleLt and POC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H265LtRefResolved {
    /// poc_lsb_lt (from SPS or slice).
    pub poc_lsb: u32,
    /// used_by_curr_pic_lt_flag[i].
    pub used: bool,
    /// delta_poc_msb_present_flag[i].
    pub msb_present: bool,
    /// Spec DeltaPocMsbCycleLt[i] (cumulative within the slice-level group).
    pub delta_poc_msb_cycle_lt: i32,
    /// Derived full POC; valid only when `msb_present` (otherwise equals the
    /// raw poc_lsb value, matching FFmpeg's `rps->poc`).
    pub poc: i32,
}

/// One entry of a final L0/L1 reference list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H265RefEntry {
    /// Derived full POC (raw poc_lsb for LT refs without msb_present).
    pub poc: i32,
    /// Match against the DPB by pic_order_cnt_lsb only (long-term reference
    /// without delta_poc_msb_present_flag).
    pub lsb_match: bool,
}

/// Fully resolved reference data for one picture (first slice header).
#[derive(Debug, Clone, Default)]
pub struct H265ResolvedRefs {
    /// S0 (negative DeltaPoc) references in RPS index order.
    pub st_curr_before: Vec<H265StRef>,
    /// S1 (positive DeltaPoc) references in RPS index order.
    pub st_curr_after: Vec<H265StRef>,
    /// Long-term references in bitstream order.
    pub long_term: Vec<H265LtRefResolved>,
    /// Final RefPicList0 (after truncation/padding/modification). Empty for
    /// I slices and IDR pictures.
    pub l0: Vec<H265RefEntry>,
    /// Final RefPicList1. For P slices, equal to `l0`.
    pub l1: Vec<H265RefEntry>,
}

/// One entry of a resolved L0/L1 list matched against DPB slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H265RefPic {
    /// DPB slot index, or -1 if the reference picture is not in the DPB.
    pub slot: i32,
    /// POC of the reference picture.
    pub poc: i32,
}

#[derive(Debug, Clone, Default)]
pub struct H265RefLists {
    pub l0: Vec<H265RefPic>,
    pub l1: Vec<H265RefPic>,
}

/// Resolve the picture's reference sets and final L0/L1 lists.
///
/// Pure computation (H.265 7.3.7 / 8.3.3) — does not touch DPB state. The
/// caller supplies the active SPS and the parser's slice header info.
pub fn resolve_refs(sps: &H265Sps, info: &SliceHeaderInfo) -> H265ResolvedRefs {
    let mut out = H265ResolvedRefs::default();

    // IDR pictures carry neither pic_order_cnt_lsb nor an RPS.
    if info.is_idr {
        return out;
    }

    // --- Short-term references (STRPS from SPS or in-slice) ---
    let strps: &H265ShortTermRefPicSet = if info.short_term_ref_pic_set_sps_flag {
        &sps
            .short_term_ref_pic_sets
            .get(info.short_term_ref_pic_set_idx as usize)
            .expect("SPS STRPS index out of range")
    } else {
        info.slice_strps
            .as_ref()
            .expect("in-slice STRPS missing for non-SPS RPS")
    };

    // delta_poc_s0/s1_minus1 hold the *cumulative* DeltaPoc (signed, stored as
    // u16 two's complement): ref POC = curr POC + DeltaPoc.
    for i in 0..strps.num_negative_pics as usize {
        let stored = strps.delta_poc_s0_minus1[i] as i32;
        let delta = if stored > 32767 { stored - 65536 } else { stored };
        out.st_curr_before.push(H265StRef {
            poc: info.curr_pic_order_cnt_val + delta,
            used: (strps.used_by_curr_pic_s0_flag >> i) & 1 == 1,
        });
    }
    for i in 0..strps.num_positive_pics as usize {
        let delta = strps.delta_poc_s1_minus1[i] as i32;
        out.st_curr_after.push(H265StRef {
            poc: info.curr_pic_order_cnt_val + delta,
            used: (strps.used_by_curr_pic_s1_flag >> i) & 1 == 1,
        });
    }

    // --- Long-term references (FFmpeg decode_lt_rps) ---
    let max_poc_lsb = 1i64 << (sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4);
    let nb_sps = info.num_long_term_sps as i64;
    let mut prev_delta = 0i64;
    for (k, lt) in info.long_term_refs.iter().enumerate() {
        let i = k as i64;
        let (poc_lsb, used) = if lt.from_sps {
            (
                sps.lt_ref_pic_poc_lsb_sps[lt.sps_idx as usize],
                ((sps.used_by_curr_pic_lt_sps_flag >> lt.sps_idx) & 1) == 1,
            )
        } else {
            (lt.poc_lsb, lt.used_by_curr_pic)
        };

        let mut cycle = 0i64;
        let mut poc = poc_lsb as i64; // raw poc_lsb when !msb_present (FFmpeg)
        if lt.delta_poc_msb_present {
            cycle = lt.delta_poc_msb_cycle as i64;
            // Cumulative within the slice-level (non-SPS) group only, and not
            // for the first entry of that group (spec 7.3.7.2).
            if i != 0 && i != nb_sps {
                cycle += prev_delta;
            }
            poc = poc_lsb as i64
                + info.curr_pic_order_cnt_val as i64
                - cycle * max_poc_lsb
                - info.pic_order_cnt_lsb as i64;
        }
        prev_delta = if lt.delta_poc_msb_present { cycle } else { 0 };

        out.long_term.push(H265LtRefResolved {
            poc_lsb,
            used,
            msb_present: lt.delta_poc_msb_present,
            delta_poc_msb_cycle_lt: cycle as i32,
            poc: poc as i32,
        });
    }

    // --- Initial L0/L1 lists (spec 8.3.3; FFmpeg ff_hevc_slice_rpl) ---
    if info.slice_type != 0 {
        // Inter slice: L0 = [S0 used, S1 used, LT used];
        // L1 (B) = [S1 used, S0 used, LT used].
        let mut l0: Vec<H265RefEntry> = Vec::new();
        let mut l1: Vec<H265RefEntry> = Vec::new();
        for r in &out.st_curr_before {
            if r.used {
                l0.push(H265RefEntry { poc: r.poc, lsb_match: false });
            }
        }
        for r in &out.st_curr_after {
            if r.used {
                l0.push(H265RefEntry { poc: r.poc, lsb_match: false });
            }
        }
        for r in &out.long_term {
            if r.used {
                l0.push(H265RefEntry { poc: r.poc, lsb_match: !r.msb_present });
            }
        }
        if info.slice_type == 2 {
            // B slice: L1 starts with S1 (future) refs.
            for r in &out.st_curr_after {
                if r.used {
                    l1.push(H265RefEntry { poc: r.poc, lsb_match: false });
                }
            }
            for r in &out.st_curr_before {
                if r.used {
                    l1.push(H265RefEntry { poc: r.poc, lsb_match: false });
                }
            }
            for r in &out.long_term {
                if r.used {
                    l1.push(H265RefEntry { poc: r.poc, lsb_match: !r.msb_present });
                }
            }
        } else {
            // P slice: RefPicList1 = RefPicList0.
            l1 = l0.clone();
        }

        // Truncate to NumRefIdxLxActive, padding cyclically with the candidate
        // entries if the initial list is shorter (FFmpeg behavior).
        // num_ref_idx_l1_active_minus1 is absent from the bitstream for P
        // slices: NumRefIdxL1Active = NumRefIdxL0Active there (spec 7.3.7).
        let n_l0 = info.num_ref_idx_l0_active_minus1 as usize + 1;
        let n_l1 = if info.slice_type == 2 {
            info.num_ref_idx_l1_active_minus1 as usize + 1
        } else {
            n_l0
        };
        out.l0 = pad_list(&l0, n_l0);
        out.l1 = pad_list(&l1, n_l1);

        // ref_pic_lists_modification (B slices only, spec 7.3.7.1) — gather
        // form: RefPicListX[i] = initial[ flag ? ref_idx : i ].
        if info.slice_type == 2 {
            out.l0 = apply_mod(&out.l0, &info.ref_pic_lists_modification_l0);
            out.l1 = apply_mod(&out.l1, &info.ref_pic_lists_modification_l1);
        }
    }

    out
}

/// Pad `init` to exactly `n` entries by cyclically re-adding its elements
/// (FFmpeg `ff_hevc_slice_rpl` behavior when used refs < NumRefIdxLx).
fn pad_list(init: &[H265RefEntry], n: usize) -> Vec<H265RefEntry> {
    if init.is_empty() || n == 0 {
        return Vec::new();
    }
    let mut out: Vec<H265RefEntry> = init[..n.min(init.len())].to_vec();
    while out.len() < n {
        for p in init {
            if out.len() >= n {
                break;
            }
            out.push(*p);
        }
    }
    out
}

/// Apply ref_pic_lists_modification (gather form). `mods[i]` corresponds to
/// list position i.
fn apply_mod(list: &[H265RefEntry], mods: &[H265ListModification]) -> Vec<H265RefEntry> {
    let mut out = list.to_vec();
    for (i, m) in mods.iter().enumerate() {
        if i >= out.len() {
            break;
        }
        if m.flag {
            let idx = (m.ref_idx as usize).min(out.len().saturating_sub(1));
            out[i] = list[idx];
        }
    }
    out
}

/// Match one reference entry against a list of slot states
/// `(slot_index, poc, poc_lsb)`.
///
/// Short-term entries (and LT entries with msb_present) match by full POC;
/// LT entries without msb_present match by pic_order_cnt_lsb only. Returns
/// the slot index, or `None` if the reference picture is not in the DPB.
pub fn match_entry(
    slots: &[(usize, i32, u16)],
    e: &H265RefEntry,
    log2_max_poc_lsb: u32,
    exclude_poc: Option<i32>,
) -> Option<usize> {
    let mask = (1i32 << log2_max_poc_lsb) - 1;
    for &(slot_i, poc, _) in slots.iter() {
        if Some(poc) == exclude_poc {
            continue;
        }
        let ok = if e.lsb_match {
            poc & mask == e.poc & mask
        } else {
            poc == e.poc
        };
        if ok {
            return Some(slot_i);
        }
    }
    None
}

/// One DPB slot.
#[derive(Debug, Clone)]
pub struct H265DpbSlot {
    /// Slot holds a decoded picture.
    pub valid: bool,
    /// Full POC of the picture in this slot.
    pub poc: i32,
    /// pic_order_cnt_lsb (for long-term lsb matching).
    pub poc_lsb: u16,
    /// RefPicFlag — the picture is a reference picture.
    pub is_ref: bool,
    /// Marked as a long-term reference by the current access unit.
    pub is_long_term: bool,
    /// Referenced (used or future-use RPS entry) by the current access unit.
    pub referenced_by_curr: bool,
    /// Still pending output in display order.
    pub needed_for_output: bool,
}

impl H265DpbSlot {
    fn empty() -> Self {
        Self {
            valid: false,
            poc: 0,
            poc_lsb: 0,
            is_ref: false,
            is_long_term: false,
            referenced_by_curr: false,
            needed_for_output: false,
        }
    }
}

/// Staged current picture (between `picture_start` and `commit_current`).
struct H265CurPic {
    poc: i32,
    poc_lsb: u16,
    is_ref: bool,
    log2_max_poc_lsb: u32,
    resolved: H265ResolvedRefs,
}

/// POC-based HEVC decoded picture buffer (common to all backends).
///
/// Mirrors [`crate::h264_dpb::H264Dpb`] usage:
/// 1. `picture_start(sps, info, is_ref)` — stage the picture, apply the
///    NoRaslOutput reset, mark referenced slots (used + future-use RPS
///    entries), evict unreferenced non-output pictures, and reserve a slot.
/// 2. `build_ref_lists()` — match the resolved L0/L1 lists against slots.
/// 3. decode into the reserved slot.
/// 4. `commit_current(slot)` — store the picture and run display logic.
pub struct H265Dpb {
    slots: Vec<H265DpbSlot>,
    max_num_reorder_frames: u32,
    cur: Option<H265CurPic>,
}

impl H265Dpb {
    pub fn new(num_slots: usize) -> Self {
        Self {
            slots: (0..num_slots).map(|_| H265DpbSlot::empty()).collect(),
            max_num_reorder_frames: 0,
            cur: None,
        }
    }

    pub fn set_max_num_reorder_frames(&mut self, v: u32) {
        self.max_num_reorder_frames = v;
    }

    /// NoRaslOutputFlag per H.265 8.3.1: 1 for IDR (always), otherwise equal
    /// to no_output_of_prior_pics_flag (0 for non-IRAP).
    fn no_rasl_output(info: &SliceHeaderInfo) -> bool {
        info.is_idr || info.no_output_of_prior_pics_flag
    }

    /// Stage the current picture and return the slot it will be stored into.
    ///
    /// Applies, in order (spec 8.3.2):
    /// - NoRaslOutput reset: an IRAP with NoRaslOutputFlag removes all
    ///   reference pictures from the DPB;
    /// - marking: every RPS entry of the current picture (used OR future-use)
    ///   marks its DPB slot as referenced by the current access unit;
    /// - eviction: unmarked slots that are not pending output are freed;
    /// - allocation: the first empty slot is reserved for the current picture.
    pub fn picture_start(
        &mut self,
        sps: &H265Sps,
        info: &SliceHeaderInfo,
        is_ref: bool,
    ) -> usize {
        let resolved = resolve_refs(sps, info);
        let cur_poc = info.curr_pic_order_cnt_val;

        // 1. Clear reference marks from the previous access unit (FFmpeg
        //    mark_ref(0) on all frames). For a NoRaslOutput IRAP the RPS is
        //    empty, so every picture ends up unmarked below and is evicted —
        //    implementing the spec 8.3.2 "remove all reference pictures" rule.
        for s in &mut self.slots {
            s.referenced_by_curr = false;
            s.is_long_term = false;
        }

        // 2. Mark referenced slots: used + future-use RPS entries keep their
        //    pictures alive (FFmpeg marks ST_FOLL/LT_FOLL lists the same way).
        let states = self.slot_states();
        let log2 = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
        for r in &resolved.st_curr_before {
            if let Some(i) = match_entry(&states, &H265RefEntry { poc: r.poc, lsb_match: false }, log2, None) {
                self.slots[i].referenced_by_curr = true;
                self.slots[i].is_long_term = false;
            }
        }
        for r in &resolved.st_curr_after {
            if let Some(i) = match_entry(&states, &H265RefEntry { poc: r.poc, lsb_match: false }, log2, None) {
                self.slots[i].referenced_by_curr = true;
                self.slots[i].is_long_term = false;
            }
        }
        for r in &resolved.long_term {
            if let Some(i) = match_entry(&states, &H265RefEntry { poc: r.poc, lsb_match: !r.msb_present }, log2, None) {
                self.slots[i].referenced_by_curr = true;
                self.slots[i].is_long_term = true;
            }
        }

        // 3. Evict: unreferenced and not pending output.
        for s in &mut self.slots {
            if s.valid && !s.referenced_by_curr && !s.needed_for_output {
                *s = H265DpbSlot::empty();
            }
        }

        // 4. Reserve the first empty slot for the current picture.
        let slot = self
            .slots
            .iter()
            .position(|s| !s.valid)
            .unwrap_or_else(|| {
                // DPB full (non-conforming stream or undersized backend):
                // recycle the oldest referenced slot rather than stall.
                eprintln!(
                    "[H265DPB] WARNING: no free slot for poc={cur_poc}, recycling oldest"
                );
                self.slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.valid)
                    .min_by_key(|(_, s)| s.poc)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            });

        self.cur = Some(H265CurPic {
            poc: cur_poc,
            poc_lsb: info.pic_order_cnt_lsb,
            is_ref,
            log2_max_poc_lsb: sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4,
            resolved,
        });
        slot
    }

    /// Match the staged picture's resolved L0/L1 lists against the current
    /// DPB slots. Must be called after `picture_start`, before
    /// `commit_current`.
    pub fn build_ref_lists(&self) -> H265RefLists {
        let cur = self.cur.as_ref().expect("picture_start not called");
        let states = self.slot_states();
        let mut lists = H265RefLists::default();
        for e in &cur.resolved.l0 {
            lists.l0.push(H265RefPic {
                slot: match_entry(&states, e, cur.log2_max_poc_lsb, Some(cur.poc))
                    .map(|i| i as i32)
                    .unwrap_or(-1),
                poc: e.poc,
            });
        }
        for e in &cur.resolved.l1 {
            lists.l1.push(H265RefPic {
                slot: match_entry(&states, e, cur.log2_max_poc_lsb, Some(cur.poc))
                    .map(|i| i as i32)
                    .unwrap_or(-1),
                poc: e.poc,
            });
        }
        lists
    }

    /// Match the staged picture's RPS entries against DPB slots, returning
    /// slot indices for `RefPicSetStCurrBefore` / `RefPicSetStCurrAfter` /
    /// `RefPicSetLtCurr` (or -1 when the entry's picture is not in the DPB).
    ///
    /// Must be called after `picture_start`, before `commit_current`.
    pub fn match_rps_slots(&self) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let cur = self.cur.as_ref().expect("picture_start not called");
        let states = self.slot_states();
        let mut before = Vec::new();
        for r in &cur.resolved.st_curr_before {
            before.push(
                match_entry(
                    &states,
                    &H265RefEntry { poc: r.poc, lsb_match: false },
                    cur.log2_max_poc_lsb,
                    Some(cur.poc),
                )
                .map(|i| i as i32)
                .unwrap_or(-1),
            );
        }
        let mut after = Vec::new();
        for r in &cur.resolved.st_curr_after {
            after.push(
                match_entry(
                    &states,
                    &H265RefEntry { poc: r.poc, lsb_match: false },
                    cur.log2_max_poc_lsb,
                    Some(cur.poc),
                )
                .map(|i| i as i32)
                .unwrap_or(-1),
            );
        }
        let mut lt = Vec::new();
        for r in &cur.resolved.long_term {
            lt.push(
                match_entry(
                    &states,
                    &H265RefEntry { poc: r.poc, lsb_match: !r.msb_present },
                    cur.log2_max_poc_lsb,
                    Some(cur.poc),
                )
                .map(|i| i as i32)
                .unwrap_or(-1),
            );
        }
        (before, after, lt)
    }

    /// All reference pictures of the staged current picture — every RPS
    /// entry (used AND future-use keep-alive) matched against live DPB
    /// slots — as `(slot, poc)` pairs deduplicated by slot.
    ///
    /// The Vulkan Video driver requires every `slot_index` value appearing
    /// in `StdVideoDecodeH265PictureInfo::RefPicSet*` to be resolvable in
    /// the `pReferenceSlots` array (video.xml: "slotIndex as used in
    /// VkVideoReferenceSlotInfoKHR structures representing pReferenceSlots
    /// in VkVideoDecodeInfoKHR"). Backends must therefore pass these
    /// pictures as reference slots, not merely the final L0/L1 union
    /// (which omits unused keep-alive RPS entries). Matches the C++
    /// reference `FillDpbH265State`, which passes every in-use reference.
    pub fn in_use_refs(&self) -> Vec<(usize, i32)> {
        let cur = self.cur.as_ref().expect("picture_start not called");
        let states = self.slot_states();
        let mut out: Vec<(usize, i32)> = Vec::new();
        let mut add = |e: &H265RefEntry| {
            if let Some(i) = match_entry(&states, e, cur.log2_max_poc_lsb, Some(cur.poc)) {
                if !out.iter().any(|(s, _)| *s == i) {
                    let poc = states
                        .iter()
                        .find(|(si, _, _)| *si == i)
                        .expect("matched slot in states")
                        .1;
                    out.push((i, poc));
                }
            }
        };
        for r in &cur.resolved.st_curr_before {
            add(&H265RefEntry { poc: r.poc, lsb_match: false });
        }
        for r in &cur.resolved.st_curr_after {
            add(&H265RefEntry { poc: r.poc, lsb_match: false });
        }
        for r in &cur.resolved.long_term {
            add(&H265RefEntry { poc: r.poc, lsb_match: !r.msb_present });
        }
        out
    }

    /// Store the current picture into `slot` and run the display logic.
    pub fn commit_current(&mut self, slot: usize) {
        let cur = match self.cur.take() {
            Some(c) => c,
            None => return,
        };
        if slot < self.slots.len() {
            self.slots[slot] = H265DpbSlot {
                valid: true,
                poc: cur.poc,
                poc_lsb: cur.poc_lsb,
                is_ref: cur.is_ref,
                is_long_term: false,
                referenced_by_curr: false,
                needed_for_output: true,
            };
        }

        // Display: output the smallest-POC pending picture once the reordering
        // delay exceeds MaxNumReorderPics (mirrors H264Dpb display_bump).
        if self.reordering_delay() > self.max_num_reorder_frames {
            self.display_bump();
        }
    }

    /// Number of pictures pending output.
    fn reordering_delay(&self) -> u32 {
        self.slots
            .iter()
            .filter(|s| s.valid && s.needed_for_output)
            .count() as u32
    }

    /// Output the smallest-POC pending picture (clears needed_for_output; the
    /// slot is freed by the next picture's eviction step if it is not a
    /// reference).
    fn display_bump(&mut self) {
        let mut i_min = None;
        let mut poc_min = i32::MAX;
        for (i, s) in self.slots.iter().enumerate() {
            if s.valid && s.needed_for_output && s.poc <= poc_min {
                if s.poc == poc_min && i_min.is_some() {
                    return; // duplicate POC -> bail
                }
                poc_min = s.poc;
                i_min = Some(i);
            }
        }
        if let Some(i) = i_min {
            self.slots[i].needed_for_output = false;
        }
    }

    /// Slots currently holding reference pictures (for backends protecting
    /// surfaces).
    pub fn get_references(&self) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.valid && s.is_ref)
            .map(|(i, _)| i)
            .collect()
    }

    /// POC of the picture in `slot`, if any.
    pub fn slot_poc(&self, i: usize) -> Option<i32> {
        self.slots.get(i).and_then(|s| s.valid.then_some(s.poc))
    }

    /// Full slot state (for backends keeping per-slot surface maps).
    pub fn slots(&self) -> &[H265DpbSlot] {
        &self.slots
    }

    /// Invalidate all slots (sequence start / reset).
    pub fn invalidate_all(&mut self) {
        for s in &mut self.slots {
            *s = H265DpbSlot::empty();
        }
        self.cur = None;
    }

    /// `(slot_index, poc, poc_lsb)` for each valid slot (aligned with the
    /// real slot indices so match results can index `self.slots` directly).
    fn slot_states(&self) -> Vec<(usize, i32, u16)> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.valid)
            .map(|(i, s)| (i, s.poc, s.poc_lsb))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h265::H265LtRef;
    use vacc_core::picture::H265ShortTermRefPicSet;

    fn sps_with(max_poc_lsb_minus4: u8) -> H265Sps {
        let mut sps = H265Sps::new();
        sps.log2_max_pic_order_cnt_lsb_minus4 = max_poc_lsb_minus4;
        sps
    }

    /// STRPS with one used S0 ref at -1 and one unused S1 ref at +2.
    fn strps_s0_1_used_s1_2_unused() -> H265ShortTermRefPicSet {
        let mut s = H265ShortTermRefPicSet::default();
        s.num_negative_pics = 1;
        s.num_positive_pics = 1;
        s.delta_poc_s0_minus1[0] = (-1i32) as u16; // cumulative DeltaPocS0[0] = -1
        s.used_by_curr_pic_s0_flag = 0b01;
        s.delta_poc_s1_minus1[0] = 2; // cumulative DeltaPocS1[0] = +2
        s.used_by_curr_pic_s1_flag = 0b00; // unused (future-use keep-alive)
        s
    }

    fn info_b(curr_poc: i32, poc_lsb: u16, strps: H265ShortTermRefPicSet) -> SliceHeaderInfo {
        let mut info = SliceHeaderInfo::new();
        info.slice_type = 2; // B
        info.curr_pic_order_cnt_val = curr_poc;
        info.pic_order_cnt_lsb = poc_lsb;
        info.short_term_ref_pic_set_sps_flag = false;
        info.slice_strps = Some(strps);
        info.num_ref_idx_l0_active_minus1 = 1; // 2 refs in L0
        info.num_ref_idx_l1_active_minus1 = 0; // 1 ref in L1
        info
    }

    #[test]
    fn resolve_refs_b_slice_lists() {
        let sps = sps_with(4);
        let info = info_b(10, 10, strps_s0_1_used_s1_2_unused());
        let r = resolve_refs(&sps, &info);

        // ST refs (used or not): S0[0] = POC 9 (used), S1[0] = POC 12 (unused).
        assert_eq!(r.st_curr_before.len(), 1);
        assert_eq!(r.st_curr_before[0].poc, 9);
        assert!(r.st_curr_before[0].used);
        assert_eq!(r.st_curr_after.len(), 1);
        assert_eq!(r.st_curr_after[0].poc, 12);
        assert!(!r.st_curr_after[0].used);

        // L0 initial = [S0 used, S1 used] = [9]; padded to NumRefIdxL0=2 -> [9, 9].
        assert_eq!(r.l0.len(), 2);
        assert_eq!(r.l0[0].poc, 9);
        assert_eq!(r.l0[1].poc, 9);
        // L1 initial = [S1 used, S0 used] = [9]; truncated to NumRefIdxL1=1 -> [9].
        assert_eq!(r.l1.len(), 1);
        assert_eq!(r.l1[0].poc, 9);
    }

    #[test]
    fn resolve_refs_b_slice_list_modification() {
        let sps = sps_with(4);
        let mut info = info_b(10, 10, strps_s0_1_used_s1_2_unused());
        // Two used refs: add a second used S0 at -3.
        let rps = info.slice_strps.as_mut().unwrap();
        rps.num_negative_pics = 2;
        rps.delta_poc_s0_minus1[1] = (-3i32) as u16;
        rps.used_by_curr_pic_s0_flag = 0b11;
        info.num_ref_idx_l0_active_minus1 = 2; // 3 refs in L0
        info.num_ref_idx_l1_active_minus1 = 2; // 3 refs in L1
        // Swap L0 positions 0 and 2.
        info.ref_pic_lists_modification_l0.push(H265ListModification { flag: true, ref_idx: 0 });
        info.ref_pic_lists_modification_l0.push(H265ListModification { flag: false, ref_idx: 0 });
        info.ref_pic_lists_modification_l0.push(H265ListModification { flag: true, ref_idx: 0 });

        let r = resolve_refs(&sps, &info);
        // L0 initial = [9, 7] (S0 used, S1 unused) -> padded to 3: [9, 7, 9].
        // After mod (swap 0<->2): [9, 7, 9] (positions 0 and 2 both hold 9).
        assert_eq!(r.l0.len(), 3);
        assert_eq!(r.l0[0].poc, 9);
        assert_eq!(r.l0[1].poc, 7);
        assert_eq!(r.l0[2].poc, 9);
        // L1 initial = [S1 used (none), S0 used] = [9, 7] -> padded to 3: [9, 7, 9].
        assert_eq!(r.l1.len(), 3);
        assert_eq!(r.l1[0].poc, 9);
        assert_eq!(r.l1[1].poc, 7);
    }

    /// Mark `info` as using an in-slice STRPS (new() defaults to SPS RPS).
    fn use_in_slice_rps(info: &mut SliceHeaderInfo) {
        info.short_term_ref_pic_set_sps_flag = false;
    }

    #[test]
    fn resolve_refs_lt_cumulative_delta() {
        let mut sps = sps_with(4); // max_poc_lsb = 256
        sps.long_term_ref_pics_present_flag = true;
        sps.num_long_term_ref_pics_sps = 1;
        sps.lt_ref_pic_poc_lsb_sps[0] = 8;
        sps.used_by_curr_pic_lt_sps_flag = 0b001; // SPS LT ref used

        let mut info = SliceHeaderInfo::new();
        info.slice_type = 1; // P
        info.curr_pic_order_cnt_val = 300;
        info.pic_order_cnt_lsb = 44; // 300 - 256
        use_in_slice_rps(&mut info);
        info.slice_strps = Some(H265ShortTermRefPicSet::default()); // no ST refs
        info.num_ref_idx_l0_active_minus1 = 2; // 3 refs in L0
        // SPS LT ref (poc_lsb=8, used, msb cycle 1) + slice LT ref (poc_lsb=10,
        // unused, msb cycle 1 -> cumulative 2).
        info.num_long_term_sps = 1;
        info.num_long_term_pics = 2;
        info.long_term_refs.push(H265LtRef {
            from_sps: true,
            sps_idx: 0,
            used_by_curr_pic: true,
            delta_poc_msb_present: true,
            delta_poc_msb_cycle: 1,
            ..Default::default()
        });
        info.long_term_refs.push(H265LtRef {
            from_sps: false,
            poc_lsb: 10,
            used_by_curr_pic: false,
            delta_poc_msb_present: true,
            delta_poc_msb_cycle: 1, // first slice-level entry: not cumulative
            ..Default::default()
        });
        info.long_term_refs.push(H265LtRef {
            from_sps: false,
            poc_lsb: 12,
            used_by_curr_pic: false,
            delta_poc_msb_present: true,
            delta_poc_msb_cycle: 3, // cumulative = prev(1) + 3 = 4
            ..Default::default()
        });

        let r = resolve_refs(&sps, &info);
        assert_eq!(r.long_term.len(), 3);

        // LT[0] (SPS): poc = 8 + 300 - 1*256 - 44 = 8.
        assert_eq!(r.long_term[0].poc_lsb, 8);
        assert!(r.long_term[0].used);
        assert!(r.long_term[0].msb_present);
        assert_eq!(r.long_term[0].delta_poc_msb_cycle_lt, 1);
        assert_eq!(r.long_term[0].poc, 8);

        // LT[1] (first slice-level entry, i == nb_sps): NOT cumulative —
        // cycle stays the raw read value 1.
        assert_eq!(r.long_term[1].poc_lsb, 10);
        assert!(!r.long_term[1].used);
        assert_eq!(r.long_term[1].delta_poc_msb_cycle_lt, 1);
        assert_eq!(r.long_term[1].poc, 10 + 300 - 1 * 256 - 44);

        // LT[2] (second slice-level entry): cumulative = 1 + 3 = 4.
        assert_eq!(r.long_term[2].delta_poc_msb_cycle_lt, 4);
        assert_eq!(r.long_term[2].poc, 12 + 300 - 4 * 256 - 44);

        // L0 (P): [ST used (none), LT used] = [8]; padded to 3: [8, 8, 8].
        assert_eq!(r.l0.len(), 3);
        assert!(r.l0.iter().all(|e| e.poc == 8 && !e.lsb_match));
        // L1 = L0 for P slices.
        assert_eq!(r.l1, r.l0);
    }

    #[test]
    fn resolve_refs_lt_lsb_match_flag() {
        let mut sps = sps_with(4);
        sps.long_term_ref_pics_present_flag = true;
        let mut info = SliceHeaderInfo::new();
        info.slice_type = 1; // P
        info.curr_pic_order_cnt_val = 300;
        info.pic_order_cnt_lsb = 44;
        use_in_slice_rps(&mut info);
        info.slice_strps = Some(H265ShortTermRefPicSet::default()); // no ST refs
        info.num_ref_idx_l0_active_minus1 = 0; // 1 ref in L0
        info.num_long_term_pics = 1;
        // No msb_present: used for lsb-only matching, poc stays raw poc_lsb.
        info.long_term_refs.push(H265LtRef {
            from_sps: false,
            poc_lsb: 40,
            used_by_curr_pic: true,
            delta_poc_msb_present: false,
            ..Default::default()
        });

        let r = resolve_refs(&sps, &info);
        assert_eq!(r.long_term[0].poc, 40);
        assert_eq!(r.l0.len(), 1);
        assert_eq!(r.l0[0].poc, 40);
        assert!(r.l0[0].lsb_match);
    }

    #[test]
    fn resolve_refs_idr_empty() {
        let sps = sps_with(4);
        let mut info = SliceHeaderInfo::new();
        info.is_idr = true;
        info.curr_pic_order_cnt_val = 0;
        let r = resolve_refs(&sps, &info);
        assert!(r.st_curr_before.is_empty());
        assert!(r.st_curr_after.is_empty());
        assert!(r.long_term.is_empty());
        assert!(r.l0.is_empty());
        assert!(r.l1.is_empty());
    }

    /// DPB lifecycle: IDR -> P -> B pyramid; verify marking, keep-alive of
    /// unused RPS entries, eviction, and NoRaslOutput reset.
    #[test]
    fn dpb_lifecycle() {
        let sps = sps_with(4);
        let mut dpb = H265Dpb::new(8);

        // PIC 0: IDR (POC 0, ref).
        let mut info = SliceHeaderInfo::new();
        info.is_idr = true;
        info.curr_pic_order_cnt_val = 0;
        let slot = dpb.picture_start(&sps, &info, true);
        assert_eq!(slot, 0);
        let lists = dpb.build_ref_lists();
        assert!(lists.l0.is_empty());
        dpb.commit_current(slot);
        assert_eq!(dpb.slot_poc(0), Some(0));

        // PIC 1: P (POC 1) referencing POC 0.
        let mut info = SliceHeaderInfo::new();
        info.slice_type = 1;
        info.curr_pic_order_cnt_val = 1;
        info.pic_order_cnt_lsb = 1;
        info.short_term_ref_pic_set_sps_flag = false;
        info.slice_strps = Some({
            let mut s = H265ShortTermRefPicSet::default();
            s.num_negative_pics = 1;
            s.delta_poc_s0_minus1[0] = (-1i32) as u16;
            s.used_by_curr_pic_s0_flag = 0b01;
            s
        });
        info.num_ref_idx_l0_active_minus1 = 0;
        let slot = dpb.picture_start(&sps, &info, true);
        let lists = dpb.build_ref_lists();
        assert_eq!(lists.l0.len(), 1);
        assert_eq!(lists.l0[0].slot, 0);
        assert_eq!(lists.l0[0].poc, 0);
        dpb.commit_current(slot);

        // PIC 2: B (POC 3) referencing POC 1 (used) and POC 5 (future-use,
        // not yet decoded -> missing), keeping POC 0 alive via unused entry.
        let mut info = SliceHeaderInfo::new();
        info.slice_type = 2;
        info.curr_pic_order_cnt_val = 3;
        info.pic_order_cnt_lsb = 3;
        info.short_term_ref_pic_set_sps_flag = false;
        info.slice_strps = Some({
            let mut s = H265ShortTermRefPicSet::default();
            s.num_negative_pics = 2;
            s.delta_poc_s0_minus1[0] = (-2i32) as u16; // POC 1, used
            s.delta_poc_s0_minus1[1] = (-3i32) as u16; // POC 0, unused (keep-alive)
            s.used_by_curr_pic_s0_flag = 0b01;
            s.num_positive_pics = 1;
            s.delta_poc_s1_minus1[0] = 2; // POC 5, used (not in DPB yet)
            s.used_by_curr_pic_s1_flag = 0b01;
            s
        });
        info.num_ref_idx_l0_active_minus1 = 0; // 1 ref in L0
        info.num_ref_idx_l1_active_minus1 = 0; // 1 ref in L1
        let slot = dpb.picture_start(&sps, &info, true);
        let lists = dpb.build_ref_lists();
        // L0 initial = [S0 used, S1 used] = [POC 1, POC 5 (missing)] ->
        // truncated to NumRefIdxL0=1: [POC 1].
        // L1 initial = [S1 used, S0 used] = [POC 5, POC 1] -> [POC 5].
        // (The unused S0 entry for POC 0 keeps it alive but is not in L0/L1.)
        assert_eq!(lists.l0.len(), 1);
        assert_eq!(lists.l0[0].poc, 1);
        assert_eq!(lists.l0[0].slot, 1);
        assert_eq!(lists.l1.len(), 1);
        assert_eq!(lists.l1[0].poc, 5);
        assert_eq!(lists.l1[0].slot, -1, "POC 5 not decoded yet -> missing ref");
        dpb.commit_current(slot);

        // POC 0 must still be alive (unused RPS entry keep-alive).
        let refs = dpb.get_references();
        assert!(refs.iter().any(|&i| dpb.slot_poc(i) == Some(0)));

        // PIC 3: CRA with no_output_of_prior_pics=1 -> all refs removed.
        let mut info = SliceHeaderInfo::new();
        info.slice_type = 0; // I
        info.is_rap = true;
        info.no_output_of_prior_pics_flag = true;
        info.curr_pic_order_cnt_val = 8;
        info.pic_order_cnt_lsb = 8;
        info.short_term_ref_pic_set_sps_flag = false;
        info.slice_strps = Some(H265ShortTermRefPicSet::default()); // empty RPS
        let slot = dpb.picture_start(&sps, &info, true);
        // After reset+eviction, only output-pending pics may remain.
        let lists = dpb.build_ref_lists();
        assert!(lists.l0.is_empty());
        dpb.commit_current(slot);
        let live_refs: Vec<i32> = dpb
            .get_references()
            .into_iter()
            .filter_map(|i| dpb.slot_poc(i))
            .collect();
        // The CRA itself is the only reference now (old refs unmarked + evicted
        // unless still pending output; with max_num_reorder_frames=0 they were
        // bumped at each commit).
        assert_eq!(live_refs, vec![8]);
    }

    #[test]
    fn dpb_eviction_of_unreferenced() {
        let sps = sps_with(4);
        let mut dpb = H265Dpb::new(8);

        // Two P pictures: POC 0, then POC 1 referencing POC 0.
        let mut info = SliceHeaderInfo::new();
        info.is_idr = true;
        info.curr_pic_order_cnt_val = 0;
        let slot = dpb.picture_start(&sps, &info, true);
        dpb.commit_current(slot);

        let mut mk_p = |poc: i32| {
            let mut info = SliceHeaderInfo::new();
            info.slice_type = 1;
            info.curr_pic_order_cnt_val = poc;
            info.pic_order_cnt_lsb = poc as u16;
            info.short_term_ref_pic_set_sps_flag = false;
            info.slice_strps = Some({
                let mut s = H265ShortTermRefPicSet::default();
                s.num_negative_pics = 1;
                s.delta_poc_s0_minus1[0] = (-1i32) as u16;
                s.used_by_curr_pic_s0_flag = 0b01;
                s
            });
            info.num_ref_idx_l0_active_minus1 = 0;
            info
        };

        let slot = dpb.picture_start(&sps, &mk_p(1), true);
        dpb.commit_current(slot);

        // POC 2 references only POC 1 -> POC 0 is unreferenced and not pending
        // output (bumped already) -> evicted.
        let slot = dpb.picture_start(&sps, &mk_p(2), true);
        let lists = dpb.build_ref_lists();
        assert_eq!(lists.l0.len(), 1);
        assert_eq!(lists.l0[0].poc, 1);
        dpb.commit_current(slot);

        let pocs: Vec<i32> = (0..8)
            .filter_map(|i| dpb.slot_poc(i))
            .collect();
        assert!(!pocs.contains(&0), "POC 0 should be evicted: {pocs:?}");
        assert!(pocs.contains(&1));
        assert!(pocs.contains(&2));
    }
}

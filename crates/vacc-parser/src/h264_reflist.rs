//! H.264 reference picture list builder (spec 8.2.3.1 + 8.2.3.2).
//!
//! Common decode-state foundation shared by the Vulkan, NVDEC, and VAAPI
//! backends. Given the DPB slot state and the slice header's
//! `ref_pic_list_modification` data, this produces the ordered reference
//! lists exactly as the H.264 spec defines them:
//!
//! - **8.2.3.1 (initialization)**:
//!   - P slices (8.2.4.2.1): L0 = short-term references in DESCENDING PicNum
//!     order (PicNum = FrameNum with wraparound, 8.2.4.1), followed by
//!     long-term references in ascending LongTermFrameIdx order.
//!   - B slices (8.2.4.2.3): L0 = short-term refs with POC <= currPOC in
//!     descending POC order, then short-term refs with POC > currPOC in
//!     ascending POC order, then long-term refs in ascending LongTermFrameIdx
//!     order. L1 is the reverse (POC > currPOC ascending, then POC <= currPOC
//!     descending, then long-term). If L0 == L1 and L1 has > 1 entry, swap
//!     L1's first two.
//!     Lists are truncated to `num_ref_idx_lN_active_minus1 + 1`.
//! - **8.2.3.2 (reordering)**: each `ref_pic_list_modification` entry is
//!   applied in order:
//!   - idc 0/1: short-term reference with `FrameNum == picNumLX`, where
//!     picNumLX walks from the current FrameNum by -(v+1) / +(v+1) modulo
//!     maxPicNum (FFmpeg matches on absolute frame_num, h264_refs.c:331).
//!   - idc 2: long-term reference with `LongTermFrameIdx == long_term_pic_num`.
//!   - idc 3: end of reordering (stop).
//!
//!   Placement mirrors FFmpeg `ff_h264_build_ref_list` exactly: the list is a
//!   fixed-size array of exactly `num_ref_idx_lN_active` entries (initial list
//!   truncated if longer, zero-padded with empty refs if shorter). Each op
//!   scans `[index, active)` for an existing occurrence of the target and
//!   shifts right; if absent from the window the tail entry is dropped; if
//!   the target picture is missing entirely the position is cleared. After all
//!   ops, any still-empty slot is filled with the default reference (first
//!   initial-list entry, h264_refs.c:212 / 391-404).
//!
//! Note: this is the H.264 bitstream syntax/semantics (7.3.6 / 8.2.3.2), which
//! differs from the HEVC-style "move-to-end / swap" operations.

use crate::h264::RefPicListModificationEntry;
use crate::h264_dpb::{MARKING_LONG, MARKING_SHORT};

/// A single reference picture in an ordered reference list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefPic {
    /// DPB slot index of the reference picture.
    pub slot: usize,
    /// Picture Order Count of the reference picture.
    pub poc: i32,
    /// FrameNum (unwrapped) of the reference picture.
    pub frame_num: u32,
    /// True if the reference is a long-term reference picture.
    pub is_long_term: bool,
}

/// Ordered reference lists for L0 and L1 after the 8.2.3.1 initialization and
/// 8.2.3.2 reordering processes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RefPicLists {
    pub l0: Vec<RefPic>,
    pub l1: Vec<RefPic>,
}

/// Per-slot DPB state needed to build reference lists.
///
/// Backend-agnostic: each backend (Vulkan `H264Dpb`, NVDEC DPB, VAAPI DPB)
/// maps its own slot state into this struct.
#[derive(Debug, Clone, Copy)]
pub struct DpbRefState {
    /// FrameNum of the picture in this slot.
    pub frame_num: u32,
    /// PicNum = FrameNum with wraparound relative to the current picture (8.2.4.1).
    pub frame_num_wrap: i32,
    /// Picture Order Count.
    pub poc: i32,
    /// `MARKING_SHORT` / `MARKING_LONG` (see `crate::h264_dpb`); any other
    /// value means the slot is not a reference picture.
    pub marking: u8,
    /// LongTermFrameIdx (meaningful only when `marking == MARKING_LONG`).
    pub long_term_frame_idx: u32,
}

/// Build the H.264 reference picture lists per spec 8.2.3.1 + 8.2.3.2.
///
/// - `slots`: DPB state in slot order (index = DPB slot index).
/// - `slice_type`: modulo-5 slice type (0=P, 1=B, 2=I, 3=SP, 4=SI).
/// - `num_ref_idx_lN_active_minus1`: active reference count per list (after
///   applying `num_ref_idx_active_override_flag`).
/// - `mod_l0` / `mod_l1`: `ref_pic_list_modification` entries per list.
///
/// I/SI slices produce empty lists.
#[allow(clippy::too_many_arguments)] // one parameter per spec input (8.2.4)
pub fn build_ref_pic_lists(
    slots: &[DpbRefState],
    slice_type: u32,
    num_ref_idx_l0_active_minus1: u32,
    num_ref_idx_l1_active_minus1: u32,
    mod_l0: &[RefPicListModificationEntry],
    mod_l1: &[RefPicListModificationEntry],
    curr_frame_num: u32,
    curr_poc: i32,
    max_frame_num: u32,
) -> RefPicLists {
    let is_b = slice_type == 1;
    if slice_type == 2 || slice_type == 4 {
        // I / SI slices have no reference lists.
        return RefPicLists::default();
    }

    // 8.2.3.1: reference picture list initialization.
    //
    // P slices (8.2.4.2.1): L0 = short-term refs in DESCENDING PicNum order,
    //   followed by long-term refs in ascending LongTermFrameIdx order.
    //
    // B slices (8.2.4.2.3):
    //   L0 = short-term refs with POC <= currPOC in DESCENDING POC order, then
    //        short-term refs with POC > currPOC in ASCENDING POC order, then
    //        long-term refs in ascending LongTermFrameIdx order.
    //   L1 = short-term refs with POC > currPOC in ASCENDING POC order, then
    //        short-term refs with POC <= currPOC in DESCENDING POC order, then
    //        long-term refs in ascending LongTermFrameIdx order.
    //   If L0 == L1 (identical slots) and L1 has > 1 entry, swap L1's first two.
    let long_sorted: Vec<usize> = {
        let mut v: Vec<usize> = slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.marking == MARKING_LONG)
            .map(|(i, _)| i)
            .collect();
        v.sort_by_key(|&i| slots[i].long_term_frame_idx); // ascending LongTermFrameIdx
        v
    };

    let short_slots: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.marking == MARKING_SHORT)
        .map(|(i, _)| i)
        .collect();

    let to_refpic = |slot: usize, is_long_term: bool| RefPic {
        slot,
        poc: slots[slot].poc,
        frame_num: slots[slot].frame_num,
        is_long_term,
    };

    let mut l0: Vec<RefPic>;
    let mut l1: Vec<RefPic> = Vec::new();

    if is_b {
        // B slices: split short-term refs at currPOC.
        let mut below: Vec<usize> = short_slots
            .iter()
            .copied()
            .filter(|&i| slots[i].poc <= curr_poc)
            .collect();
        below.sort_by(|&a, &b| slots[b].poc.cmp(&slots[a].poc)); // descending POC
        let mut above: Vec<usize> = short_slots
            .iter()
            .copied()
            .filter(|&i| slots[i].poc > curr_poc)
            .collect();
        above.sort_by(|&a, &b| slots[a].poc.cmp(&slots[b].poc)); // ascending POC

        // L0 = below (desc) + above (asc) + long (asc LTFrameIdx)
        l0 = below
            .iter()
            .map(|&s| to_refpic(s, false))
            .chain(above.iter().map(|&s| to_refpic(s, false)))
            .chain(long_sorted.iter().map(|&s| to_refpic(s, true)))
            .collect();
        // L1 = above (asc) + below (desc) + long (asc LTFrameIdx)
        l1 = above
            .iter()
            .map(|&s| to_refpic(s, false))
            .chain(below.iter().map(|&s| to_refpic(s, false)))
            .chain(long_sorted.iter().map(|&s| to_refpic(s, true)))
            .collect();
        // If L0 == L1 and L1 has > 1 entry, swap L1's first two (8.2.4.2.3).
        if l1.len() > 1
            && l0.len() == l1.len()
            && l0.iter().zip(l1.iter()).all(|(a, b)| a.slot == b.slot)
        {
            l1.swap(0, 1);
        }
    } else {
        // P slices: short-term in descending PicNum, then long-term.
        let mut short_desc: Vec<usize> = short_slots.clone();
        short_desc.sort_by(|&a, &b| slots[b].frame_num_wrap.cmp(&slots[a].frame_num_wrap)); // descending PicNum
        l0 = short_desc
            .iter()
            .map(|&s| to_refpic(s, false))
            .chain(long_sorted.iter().map(|&s| to_refpic(s, true)))
            .collect();
    }

    // 8.2.3.2: reordering on a fixed-size array of exactly `active` entries
    // (see module docs / ff_h264_build_ref_list): truncate excess, pad the
    // shortfall with empty refs, reorder, then backfill empties with the
    // default ref (first initial-list entry).
    let active0 = (num_ref_idx_l0_active_minus1 as usize).saturating_add(1);
    let active1 = (num_ref_idx_l1_active_minus1 as usize).saturating_add(1);
    let default0 = l0.first().copied();
    let default1 = l1.first().copied();
    for (list, active) in [(&mut l0, active0), (&mut l1, active1)] {
        list.truncate(active);
        list.resize_with(active, empty_ref);
    }
    apply_reordering(
        &mut l0,
        slots,
        mod_l0,
        curr_frame_num,
        max_frame_num,
        active0,
    );
    apply_reordering(
        &mut l1,
        slots,
        mod_l1,
        curr_frame_num,
        max_frame_num,
        active1,
    );
    backfill_default(&mut l0, default0);
    backfill_default(&mut l1, default1);

    RefPicLists { l0, l1 }
}

/// Apply the 8.2.3.2 reference picture list reordering process to one list.
///
/// Mirrors FFmpeg `ff_h264_build_ref_list` exactly:
/// - `picNumLXPred` starts at the current picture's FrameNum and is updated
///   cumulatively with wraparound over `max_pic_num` (modulo).
/// - idc 0: short-term, picNumLX = (picNumLXPred - (v+1)) mod maxPicNum.
/// - idc 1: short-term, picNumLX = (picNumLXPred + (v+1)) mod maxPicNum.
/// - idc 2: long-term with LongTermFrameIdx == v.
/// - idc 3: end of reordering.
/// - The target is the marked reference with absolute `frame_num == picNumLX`
///   (FFmpeg h264_refs.c:331-338; NOT a wrapped/PicNum comparison).
/// - Placement: scan [index, len) for an existing occurrence of the target;
///   if found at j, shift list[index..j] right and write at index; otherwise
///   shift the tail right (dropping its last entry) and write at index. If the
///   target picture is missing entirely, position `index` is cleared (empty
///   ref) without shifting; `backfill_default` restores it afterwards.
///
/// `list` must be exactly `window`-sized (see `build_ref_pic_lists`).
fn apply_reordering(
    list: &mut [RefPic],
    slots: &[DpbRefState],
    mods: &[RefPicListModificationEntry],
    curr_frame_num: u32,
    max_frame_num: u32,
    window: usize,
) {
    let max_pic_num = max_frame_num.max(1) as i32;
    let mut pic_num_pred = curr_frame_num as i32;
    for (ref_idx, m) in mods.iter().enumerate() {
        if ref_idx >= window {
            break; // beyond the active window (FFmpeg flags this as an error)
        }
        let target: Option<usize> = match m.op {
            // idc 0: short-term subtract.
            0 => {
                pic_num_pred = (pic_num_pred - (m.difference.max(0) + 1)).rem_euclid(max_pic_num);
                slots
                    .iter()
                    .position(|s| s.marking == MARKING_SHORT && s.frame_num == pic_num_pred as u32)
            }
            // idc 1: short-term add (forward reference).
            1 => {
                pic_num_pred = (pic_num_pred + (m.difference.max(0) + 1)).rem_euclid(max_pic_num);
                slots
                    .iter()
                    .position(|s| s.marking == MARKING_SHORT && s.frame_num == pic_num_pred as u32)
            }
            // idc 2: long-term with LongTermFrameIdx == long_term_pic_num.
            2 => {
                let lt_idx = m.difference.max(0) as u32;
                slots
                    .iter()
                    .position(|s| s.marking == MARKING_LONG && s.long_term_frame_idx == lt_idx)
            }
            // idc 3: end of reordering; 4/5 invalid per spec.
            _ => return,
        };
        match target {
            Some(slot) => window_insert(list, slot, ref_idx, slots),
            // FFmpeg: "reference picture missing during reorder" -> clear the
            // position (no shift); backfill_default restores default_ref.
            None => list[ref_idx] = empty_ref(),
        }
    }
}

/// Sentinel slot index for an empty reference-list entry (FFmpeg's
/// zero-filled `H264Ref`); replaced by the default ref after reordering.
const EMPTY_SLOT: usize = usize::MAX;

fn empty_ref() -> RefPic {
    RefPic {
        slot: EMPTY_SLOT,
        poc: 0,
        frame_num: 0,
        is_long_term: false,
    }
}

/// FFmpeg final pass (h264_refs.c:391-404): any still-empty list slot is
/// filled with the default reference (first initial-list entry). Without a
/// default (no marked refs at all — invalid stream) empty slots are dropped.
fn backfill_default(list: &mut Vec<RefPic>, default_ref: Option<RefPic>) {
    match default_ref {
        Some(d) => {
            for r in list.iter_mut() {
                if r.slot == EMPTY_SLOT {
                    *r = d;
                }
            }
        }
        None => list.retain(|r| r.slot != EMPTY_SLOT),
    }
}

/// FFmpeg-style in-window placement (see `apply_reordering`). The list is
/// exactly window-sized; when the target is absent from [index, len) the tail
/// entry is dropped (shift right) before writing at index.
fn window_insert(list: &mut [RefPic], slot: usize, index: usize, slots: &[DpbRefState]) {
    if index >= list.len() {
        return;
    }
    let new_ref = RefPic {
        slot,
        poc: slots[slot].poc,
        frame_num: slots[slot].frame_num,
        is_long_term: slots[slot].marking == MARKING_LONG,
    };
    let mut j = index;
    while j < list.len() && list[j].slot != slot {
        j += 1;
    }
    let shift_end = if j < list.len() { j } else { list.len() - 1 };
    for i in (index + 1..=shift_end).rev() {
        list[i] = list[i - 1];
    }
    list[index] = new_ref;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(frame_num: u32, wrap: i32, poc: i32, marking: u8, lt_idx: u32) -> DpbRefState {
        DpbRefState {
            frame_num,
            frame_num_wrap: wrap,
            poc,
            marking,
            long_term_frame_idx: lt_idx,
        }
    }

    fn entry(index: u32, op: u32, difference: i32) -> RefPicListModificationEntry {
        RefPicListModificationEntry {
            index,
            length: 0,
            op,
            difference,
        }
    }

    /// (a) P-slice init ordering: short-term refs must come out in DESCENDING
    /// PicNum (frame_num_wrap) order, NOT slot or POC order (8.2.4.2.1).
    #[test]
    fn init_ordering_unsorted_pocs() {
        let slots = [
            st(1, 1, 0, MARKING_SHORT, 0), // slot 0: PicNum 1
            st(3, 3, 4, MARKING_SHORT, 0), // slot 1: PicNum 3
            st(2, 2, 2, MARKING_SHORT, 0), // slot 2: PicNum 2
        ];
        // P slice, 3 active refs in L0.
        let lists = build_ref_pic_lists(&slots, 0, 2, 0, &[], &[], 4, 0, 16);
        assert_eq!(
            lists.l0.iter().map(|r| r.frame_num).collect::<Vec<_>>(),
            vec![3, 2, 1]
        );
        assert_eq!(
            lists.l0.iter().map(|r| r.slot).collect::<Vec<_>>(),
            vec![1, 2, 0]
        );
        assert!(lists.l1.is_empty());
    }

    /// (b) op=0 (short-term subtract): PicNum starts at CurrFn and is decremented
    /// by (abs_diff+1); the matching short-term ref is inserted at PicNumIdx
    /// (insert-with-dedup).
    #[test]
    fn reorder_op0_subtract() {
        let slots = [
            st(1, 1, 0, MARKING_SHORT, 0),
            st(2, 2, 2, MARKING_SHORT, 0),
            st(3, 3, 4, MARKING_SHORT, 0),
        ];
        // Initial L0 = [fn3, fn2, fn1] (descending PicNum); curr=4;
        // op=0, abs_diff=1 -> PicNum=2 (fn2). Insert fn2 at position 0 (dedup):
        // [fn2, fn3, fn1].
        let mods = vec![entry(0, 0, 1)];
        let lists = build_ref_pic_lists(&slots, 0, 2, 0, &mods, &[], 4, 0, 16);
        assert_eq!(
            lists.l0.iter().map(|r| r.frame_num).collect::<Vec<_>>(),
            vec![2, 3, 1]
        );
    }

    /// (c) op=1: SHORT-TERM forward — picNumLX = (picNumLXPred + (v+1)) mod
    /// maxPicNum (spec 8.2.3.2 / FFmpeg ff_h264_build_ref_list case 1).
    #[test]
    fn reorder_op1_short_term_add() {
        let slots = [
            st(1, 1, 0, MARKING_SHORT, 0),
            st(5, 5, 4, MARKING_SHORT, 0),
            st(6, 6, 6, MARKING_SHORT, 0),
        ];
        // Initial L0 = [fn6, fn5, fn1] (descending PicNum); curr=4;
        // op=1, v=0 -> picNumLX = 4+1 = 5 (fn5 at pos 1) -> move to pos 0:
        // [fn5, fn6, fn1].
        let mods = vec![entry(0, 1, 0)];
        let lists = build_ref_pic_lists(&slots, 0, 2, 0, &mods, &[], 4, 0, 16);
        assert_eq!(
            lists.l0.iter().map(|r| r.frame_num).collect::<Vec<_>>(),
            vec![5, 6, 1]
        );
    }

    /// (c2) op=2: long-term with LongTermFrameIdx == v (the value IS the
    /// LongTermFrameIdx; FFmpeg case 2 uses `val` directly).
    #[test]
    fn reorder_op2_long_term_by_idx() {
        let slots = [
            st(1, 1, 0, MARKING_SHORT, 0),
            st(7, 7, 8, MARKING_LONG, 0),  // LT FrameIdx 0
            st(8, 8, 10, MARKING_LONG, 1), // LT FrameIdx 1
        ];
        // Initial L0 = [fn1, LT0(fn7), LT1(fn8)] (short desc, then LT asc by idx);
        // op=2, v=1 -> target LT1 (fn8) at pos 2 -> move to pos 0:
        // [LT1(fn8), fn1, LT0(fn7)].
        let mods = vec![entry(0, 2, 1)];
        let lists = build_ref_pic_lists(&slots, 0, 2, 0, &mods, &[], 4, 0, 16);
        assert_eq!(lists.l0.len(), 3);
        assert_eq!(lists.l0[0].frame_num, 8);
        assert!(lists.l0[0].is_long_term);
        assert_eq!(lists.l0[1].frame_num, 1);
        assert_eq!(lists.l0[2].frame_num, 7);
    }

    /// (c3) reordering operates on the full initial list and truncates AFTER:
    /// a target beyond the active window (initially truncated away) is still
    /// found and placed; the window tail is dropped (FFmpeg semantics).
    #[test]
    fn reorder_truncates_after() {
        let slots = [
            st(1, 1, 0, MARKING_SHORT, 0),
            st(2, 2, 2, MARKING_SHORT, 0),
            st(3, 3, 4, MARKING_SHORT, 0),
            st(4, 4, 6, MARKING_SHORT, 0),
        ];
        // P slice, 3 active: initial full L0 = [fn4, fn3, fn2, fn1].
        // op=0, v=3 -> picNumLX = 5-4 = 1 (fn1, beyond the 3-entry window)
        // -> place at pos 0, drop window tail: [fn1, fn4, fn3].
        let mods = vec![entry(0, 0, 3)];
        let lists = build_ref_pic_lists(&slots, 0, 2, 0, &mods, &[], 5, 0, 16);
        assert_eq!(
            lists.l0.iter().map(|r| r.frame_num).collect::<Vec<_>>(),
            vec![1, 4, 3]
        );
    }

    /// (d) B slice: L0/L1 are initialized by POC relative to currPOC (8.2.4.2.3):
    /// L0 = POC<=currPOC (desc) then POC>currPOC (asc); L1 = the reverse.
    #[test]
    fn b_slice_poc_init() {
        let slots = [
            st(1, 1, 0, MARKING_SHORT, 0), // poc 0 (below)
            st(2, 2, 2, MARKING_SHORT, 0), // poc 2 (below)
            st(3, 3, 4, MARKING_SHORT, 0), // poc 4 (above)
        ];
        // B slice, curr_poc=3, 3 active refs per list, no rplm:
        //   below (POC<=3, desc): [fn2(poc2), fn1(poc0)]
        //   above (POC>3, asc):  [fn3(poc4)]
        //   L0 = below + above = [fn2, fn1, fn3]
        //   L1 = above + below = [fn3, fn2, fn1]
        let lists = build_ref_pic_lists(&slots, 1, 2, 2, &[], &[], 3, 3, 16);
        assert_eq!(
            lists.l0.iter().map(|r| r.frame_num).collect::<Vec<_>>(),
            vec![2, 1, 3]
        );
        assert_eq!(
            lists.l1.iter().map(|r| r.frame_num).collect::<Vec<_>>(),
            vec![3, 2, 1]
        );
    }

    /// I slice: no reference lists.
    #[test]
    fn i_slice_empty() {
        let slots = [st(1, 1, 0, MARKING_SHORT, 0)];
        let lists = build_ref_pic_lists(&slots, 2, 0, 0, &[], &[], 4, 0, 16);
        assert!(lists.l0.is_empty());
        assert!(lists.l1.is_empty());
    }

    /// Long-term refs follow short-term refs, in ascending LongTermFrameIdx order.
    #[test]
    fn long_term_after_short_term() {
        let slots = [
            st(9, 9, 8, MARKING_LONG, 1),  // slot 0: LT1
            st(1, 1, 0, MARKING_SHORT, 0), // slot 1: PicNum 1
            st(7, 7, 6, MARKING_LONG, 0),  // slot 2: LT0
            st(2, 2, 2, MARKING_SHORT, 0), // slot 3: PicNum 2
        ];
        let lists = build_ref_pic_lists(&slots, 0, 3, 0, &[], &[], 10, 0, 16);
        assert_eq!(
            lists
                .l0
                .iter()
                .map(|r| (r.slot, r.is_long_term))
                .collect::<Vec<_>>(),
            vec![(3, false), (1, false), (2, true), (0, true)]
        );
    }

    /// idc=3 terminates reordering; later entries are ignored.
    #[test]
    fn op3_terminates() {
        let slots = [
            st(1, 1, 0, MARKING_SHORT, 0),
            st(2, 2, 2, MARKING_SHORT, 0),
            st(3, 3, 4, MARKING_SHORT, 0),
        ];
        let mods = vec![
            entry(0, 0, 0), // subtract: PicNum=3 (fn3) -> already at 0: [fn3, fn2, fn1]
            entry(0, 3, 0), // end
            entry(0, 0, 1), // ignored
        ];
        let lists = build_ref_pic_lists(&slots, 0, 2, 0, &mods, &[], 4, 0, 16);
        assert_eq!(
            lists.l0.iter().map(|r| r.frame_num).collect::<Vec<_>>(),
            vec![3, 2, 1]
        );
    }

    /// The list is padded to the active count with empty refs and the tail is
    /// backfilled with the default ref (first initial entry) — FFmpeg
    /// h264_refs.c:181-182 / 391-404. Two marked refs, active=3, no RPLM:
    /// initial [fn2, fn1] + empty -> [fn2, fn1, fn2].
    #[test]
    fn pads_to_active_with_default_backfill() {
        let slots = [st(1, 1, 0, MARKING_SHORT, 0), st(2, 2, 2, MARKING_SHORT, 0)];
        let lists = build_ref_pic_lists(&slots, 0, 2, 0, &[], &[], 3, 4, 16);
        assert_eq!(
            lists.l0.iter().map(|r| r.frame_num).collect::<Vec<_>>(),
            vec![2, 1, 2]
        );
    }

    /// FrameNum-wrap scenario (regression for the 8 genuine GT mismatches at
    /// every 32-picture wrap): curr_fn=1, marked refs {fn13, fn15, fn0} with
    /// signed wraps {-3, -1, 0}. Targets must match on ABSOLUTE frame_num
    /// (FFmpeg h264_refs.c:331), the list grows via padding, and tail-drop
    /// inserts fill the padded slot. Expected final L0 (verified against
    /// instrumented FFmpeg GT-REFLST): [fn15, fn15, fn0, fn13].
    #[test]
    fn reorder_wrap_absolute_frame_num_target() {
        let slots = [
            st(13, -3, 118, MARKING_SHORT, 0),
            st(15, -1, 126, MARKING_SHORT, 0),
            st(0, 0, 122, MARKING_SHORT, 0),
        ];
        // RPLM as in the main-stream P-slices at fn=1:
        // i=0: idc0 d=1 -> pred=(1-2)%16=15 -> fn15 @0
        // i=1: idc0 d=15 -> pred=(15-16)%16=15 -> fn15 @1 (dup, tail drop)
        // i=2: idc1 d=0 -> pred=(15+1)%16=0  -> fn0 @2
        // i=3: idc0 d=2 -> pred=(0-3)%16=13  -> fn13 @3
        let mods = vec![
            entry(0, 0, 1),
            entry(1, 0, 15),
            entry(2, 1, 0),
            entry(3, 0, 2),
        ];
        let lists = build_ref_pic_lists(&slots, 0, 3, 0, &mods, &[], 1, 134, 16);
        assert_eq!(
            lists.l0.iter().map(|r| r.frame_num).collect::<Vec<_>>(),
            vec![15, 15, 0, 13]
        );
    }
}

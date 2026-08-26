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
//!   Lists are truncated to `num_ref_idx_lN_active_minus1 + 1`.
//! - **8.2.3.2 (reordering)**: each `ref_pic_list_modification` entry is
//!   applied in order:
//!   - idc 0: `RefPicListX[PicNumIdx] = RefPicListX[abs_diff_pic_num_minus1 + 1]`
//!     (the entry at position `abs_diff_pic_num_minus1 + 1` moves to
//!     `PicNumIdx`; entries in between shift down by one).
//!   - idc 1: `RefPicListX[PicNumIdx]` = the long-term reference picture with
//!     `LongTermFrameIdx == long_term_pic_num`.
//!   - idc 2: `RefPicListX[PicNumIdx]` = the long-term reference picture with
//!     `LongTermFrameIdx == 0`.
//!   - idc 3: end of reordering (stop).
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

    let mut l0: Vec<RefPic> = Vec::new();
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
            let tmp = l1[0];
            l1[0] = l1[1];
            l1[1] = tmp;
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

    if std::env::var("VACC_DBG_H264").is_ok() {
        let inp: Vec<String> = slots
            .iter()
            .enumerate()
            .map(|(i, s)| format!("s{}/f{}/w{}/m{}/p{}", i, s.frame_num, s.frame_num_wrap, s.marking, s.poc))
            .collect();
        let i0: Vec<String> = l0.iter().map(|r| format!("s{}/w{}/f{}/p{}", r.slot, slots[r.slot].frame_num_wrap, r.frame_num, r.poc)).collect();
        let i1: Vec<String> = l1.iter().map(|r| format!("s{}/w{}/f{}/p{}", r.slot, slots[r.slot].frame_num_wrap, r.frame_num, r.poc)).collect();
        eprintln!("REFLIST-DBG slice={} l0={} l1={} currpoc={} in=[{}] init_l0=[{}] init_l1=[{}]", slice_type, num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1, curr_poc, inp.join(","), i0.join(","), i1.join(","));
    }

    l0.truncate((num_ref_idx_l0_active_minus1 as usize).saturating_add(1));
    l1.truncate((num_ref_idx_l1_active_minus1 as usize).saturating_add(1));

    // 8.2.3.2: reordering (spec arithmetic + insert-with-dedup, matching the
    // C++ oracle VkEncoderDpbH264::RefPicListReorderingLX).
    apply_reordering(&mut l0, slots, mod_l0, curr_frame_num, max_frame_num);
    apply_reordering(&mut l1, slots, mod_l1, curr_frame_num, max_frame_num);

    if std::env::var("VACC_DBG_H264").is_ok() {
        let f0: Vec<String> = l0.iter().map(|r| format!("s{}/f{}/p{}", r.slot, r.frame_num, r.poc)).collect();
        let f1: Vec<String> = l1.iter().map(|r| format!("s{}/f{}/p{}", r.slot, r.frame_num, r.poc)).collect();
        let m0: Vec<String> = mod_l0.iter().map(|m| format!("op{}/idx{}/d{}", m.op, m.index, m.difference)).collect();
        let m1: Vec<String> = mod_l1.iter().map(|m| format!("op{}/idx{}/d{}", m.op, m.index, m.difference)).collect();
        eprintln!("REFLIST-FINAL slice={} currfn={} currpoc={} l0=[{}] l1=[{}] mod0=[{}] mod1=[{}]", slice_type, curr_frame_num, curr_poc, f0.join(","), f1.join(","), m0.join(","), m1.join(","));
    }

    RefPicLists { l0, l1 }
}

/// Apply the 8.2.3.2 reference picture list reordering process to one list.
///
/// Mirrors the C++ oracle `VkEncDpbH264::RefPicListReorderingLX`:
/// - `picNumLXPred` starts at the current picture's FrameNum and is updated
///   cumulatively (spec 8-35/8-36) with wraparound over `max_frame_num`.
/// - Each short-term modification places the reference whose PicNum equals the
///   running value at the next list position, using insert-with-dedup (spec 8-38):
///   the reference is removed from its current position (if present) and inserted
///   at the target position.
fn apply_reordering(
    list: &mut Vec<RefPic>,
    slots: &[DpbRefState],
    mods: &[RefPicListModificationEntry],
    curr_frame_num: u32,
    max_frame_num: u32,
) {
    let max_pic_num = max_frame_num.max(1) as i32;
    let curr_pic_num = curr_frame_num as i32;
    let mut pic_num_pred = curr_pic_num;
    let mut ref_idx = 0usize;
    for m in mods {
        match m.op {
            // idc 0: short-term subtract (spec 8-35).
            0 => {
                let diff = m.difference.max(0) as i32 + 1;
                let mut pic_num_no_wrap = pic_num_pred - diff;
                if pic_num_no_wrap < 0 {
                    pic_num_no_wrap += max_pic_num;
                }
                pic_num_pred = pic_num_no_wrap;
                let pic_num_lx = if pic_num_no_wrap > curr_pic_num {
                    pic_num_no_wrap - max_pic_num
                } else {
                    pic_num_no_wrap
                };
                if let Some(slot) = slots
                    .iter()
                    .position(|s| s.marking == MARKING_SHORT && s.frame_num_wrap == pic_num_lx)
                {
                    insert_with_dedup(list, slot, ref_idx, slots);
                }
                ref_idx += 1;
            }
            // idc 1: long-term with LongTermFrameIdx == long_term_pic_num.
            1 => {
                let lt_idx = m.difference.max(0) as u32;
                if let Some(slot) = slots
                    .iter()
                    .position(|s| s.marking == MARKING_LONG && s.long_term_frame_idx == lt_idx)
                {
                    insert_with_dedup(list, slot, ref_idx, slots);
                }
                ref_idx += 1;
            }
            // idc 2: long-term with LongTermFrameIdx == 0.
            2 => {
                if let Some(slot) = slots
                    .iter()
                    .position(|s| s.marking == MARKING_LONG && s.long_term_frame_idx == 0)
                {
                    insert_with_dedup(list, slot, ref_idx, slots);
                }
                ref_idx += 1;
            }
            // idc 3: end of reordering.
            3 => break,
            // idc 4/5: invalid per spec; stop processing.
            _ => break,
        }
    }
}

/// Insert the reference in `slot` at position `ref_idx`, removing any existing
/// occurrence of the same slot first (spec 8-38 insert-with-dedup).
fn insert_with_dedup(list: &mut Vec<RefPic>, slot: usize, ref_idx: usize, slots: &[DpbRefState]) {
    let new_ref = RefPic {
        slot,
        poc: slots[slot].poc,
        frame_num: slots[slot].frame_num,
        is_long_term: slots[slot].marking == MARKING_LONG,
    };
    if let Some(pos) = list.iter().position(|r| r.slot == slot) {
        list.remove(pos);
    }
    let idx = ref_idx.min(list.len());
    list.insert(idx, new_ref);
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

    /// (c) op=1: the entry at PicNumIdx is replaced by the long-term reference
    /// picture with LongTermFrameIdx == long_term_pic_num (spec: pure
    /// assignment — other entries are untouched, list length unchanged).
    #[test]
    fn reorder_op1_long_term() {
        let slots = [
            st(1, 1, 0, MARKING_SHORT, 0),
            st(5, 5, 10, MARKING_LONG, 2), // long-term, LongTermFrameIdx = 2
            st(2, 2, 2, MARKING_SHORT, 0),
        ];
        // Initial L0 = [fn2, fn1, LT2] (descending PicNum, then long-term);
        // op=1, long_term_pic_num=2 -> insert LT2 at 0. Insert-with-dedup:
        // [LT2, fn2, fn1].
        let mods = vec![entry(0, 1, 2)];
        let lists = build_ref_pic_lists(&slots, 0, 2, 0, &mods, &[], 6, 0, 16);
        assert_eq!(lists.l0.len(), 3);
        assert!(lists.l0[0].is_long_term);
        assert_eq!(lists.l0[0].slot, 1);
        assert_eq!(lists.l0[1].frame_num, 2);
        assert_eq!(lists.l0[2].frame_num, 1);
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
            st(9, 9, 8, MARKING_LONG, 1), // slot 0: LT1
            st(1, 1, 0, MARKING_SHORT, 0), // slot 1: PicNum 1
            st(7, 7, 6, MARKING_LONG, 0), // slot 2: LT0
            st(2, 2, 2, MARKING_SHORT, 0), // slot 3: PicNum 2
        ];
        let lists = build_ref_pic_lists(&slots, 0, 3, 0, &[], &[], 10, 0, 16);
        assert_eq!(
            lists.l0
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
}

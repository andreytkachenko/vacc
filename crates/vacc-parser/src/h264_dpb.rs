//! H.264 DPB manager that mirrors the C++ `VulkanH264Parser` DPB state machine.
//!
//! The Vulkan HW decoder indexes reference slots by POSITION (refIdxL0[i] -> slot[i]),
//! so the reference list order MUST match the C++ oracle's DPB slot order for a
//! pixel-perfect decode. This manager replicates that state machine:
//!
//! - Reference list = the short-term refs in DPB slot order (no sorting).
//! - New pictures are stored in the first empty slot.
//! - Ref-set management: MMCO (adaptive) for slices with `adaptive_ref_pic_marking_mode_flag`,
//!   sliding window otherwise.
//! - FrameNum-conflict unmarking (non-conforming repeated frame_nums).
//! - Display logic: output the smallest-POC pending picture, freeing its slot if non-ref.
//!
//! See `memory: h264_cpp_dpb_state_machine.md` for the verified ground-truth facts.
//!
//! This module is the common H.264 decode-state foundation shared by the Vulkan,
//! NVDEC, and VAAPI backends (moved from `vacc-vulkan`).

use crate::h264::RefPicListModificationEntry;
use crate::h264_reflist::{DpbRefState, RefPicLists, build_ref_pic_lists};

/// H.264 Memory Management Control Operation (MMCO) command.
/// See H.264 spec 8.2.5.4 for details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Adaptive reference picture marking operations (H.264 spec 8.2.5.1).
/// Op numbers are the bitstream's memory_management_control_operation values.
pub enum H264MmcoCommand {
    /// MMCO 1: unmark short-term reference with PicNum == picNumX
    UnmarkShortTerm { difference_of_pic_nums_minus1: u32 },
    /// MMCO 2: unmark long-term reference with LongTermFrameIdx
    UnmarkLongTerm { long_term_frame_idx: u32 },
    /// MMCO 3: assign LongTermFrameIdx to the short-term reference with
    /// PicNum == picNumX (it becomes a long-term reference)
    AssignLongTerm {
        difference_of_pic_nums_minus1: u32,
        long_term_frame_idx: u32,
    },
    /// MMCO 4: mark the current picture as a long-term reference
    AssignLongTermToCurrent { long_term_frame_idx: u32 },
    /// MMCO 5: set MaxLongTermFrameIdxPlus1 (constrains future ops only; no DPB state change)
    SetMaxLongTermFrameIdx { max_long_term_frame_idx_plus1: u32 },
    /// MMCO 6: unmark all short-term reference pictures
    UnmarkAll,
}

/// Slot is not a reference picture.
pub const MARKING_UNUSED: u8 = 0;
/// Slot holds a short-term reference picture.
pub const MARKING_SHORT: u8 = 1;
/// Slot holds a long-term reference picture.
pub const MARKING_LONG: u8 = 2;

/// A single DPB slot (logical state; physical images are managed by the decoder).
#[derive(Debug, Clone)]
pub struct H264DpbSlot {
    /// 0 = empty, 3 = full frame stored.
    pub state: u8,
    pub frame_num: u32,
    /// FrameNumWrap relative to the last current frame (see `refresh_frame_num_wrap`).
    pub frame_num_wrap: i32,
    pub poc: i32,
    /// MARKING_UNUSED / MARKING_SHORT / MARKING_LONG.
    pub marking: u8,
    /// LongTermFrameIdx (meaningful only when `marking == MARKING_LONG`).
    pub long_term_frame_idx: u32,
    pub needed_for_output: bool,
}

impl H264DpbSlot {
    fn empty() -> Self {
        Self {
            state: 0,
            frame_num: 0,
            frame_num_wrap: 0,
            poc: 0,
            marking: MARKING_UNUSED,
            long_term_frame_idx: 0,
            needed_for_output: false,
        }
    }
    #[inline]
    pub fn is_ref(&self) -> bool {
        self.state != 0 && (self.marking == MARKING_SHORT || self.marking == MARKING_LONG)
    }
}

/// H.264 DPB manager mirroring the C++ state machine.
///
/// `num_slots` active slots (0..num_slots-1). The current picture is tracked separately
/// (not in a slot) until `commit_current` stores it into a real slot.
pub struct H264Dpb {
    pub slots: Vec<H264DpbSlot>,
    /// Effective DPB size for the `dpb_full` check (C++ `m_MaxDpbSize`).
    pub max_dpb_size: usize,
    pub num_ref_frames: u32,
    /// 1 << (log2_max_frame_num_minus4 + 4).
    pub max_frame_num: u32,
    /// SPS VUI max_num_reorder_frames (display latency limit).
    pub max_num_reorder_frames: u32,
    /// The current picture being decoded (staged, not yet in a slot).
    cur: Option<CurrentPic>,
}

#[derive(Debug, Clone)]
struct CurrentPic {
    frame_num: u32,
    poc: i32,
    is_ref: bool,
    is_idr: bool,
    /// IDR slice header's long_term_reference_flag (8.2.5.1): the picture is
    /// committed as a long-term reference with LongTermFrameIdx 0.
    long_term_reference: bool,
    no_output_of_prior_pics: bool,
    mmco: bool,
    mmco_commands: Vec<H264MmcoCommand>,
}

impl H264Dpb {
    pub fn new(
        num_slots: usize,
        max_dpb_size: usize,
        num_ref_frames: u32,
        max_frame_num: u32,
    ) -> Self {
        Self {
            slots: (0..num_slots).map(|_| H264DpbSlot::empty()).collect(),
            max_dpb_size: max_dpb_size.min(num_slots),
            num_ref_frames,
            max_frame_num,
            max_num_reorder_frames: 0,
            cur: None,
        }
    }

    pub fn set_max_num_reorder_frames(&mut self, v: u32) {
        self.max_num_reorder_frames = v;
    }

    fn fullness(&self) -> usize {
        self.slots.iter().filter(|s| s.state != 0).count()
    }

    #[inline]
    fn dpb_full(&self) -> bool {
        let f = self.fullness();
        f > 0 && f >= self.max_dpb_size
    }

    /// Recompute FrameNumWrap for all slots relative to `cur_fn` (C++ picture_numbers).
    fn refresh_frame_num_wrap(&mut self, cur_fn: u32) {
        for s in &mut self.slots {
            if s.state == 0 {
                continue;
            }
            s.frame_num_wrap = if s.frame_num > cur_fn {
                s.frame_num as i32 - self.max_frame_num as i32
            } else {
                s.frame_num as i32
            };
        }
    }

    /// Stage the current picture (C++ dpb_picture_start). Must be called before
    /// `get_references` / `prepare_current`.
    #[allow(clippy::too_many_arguments)] // mirrors the slice-header fields it consumes
    pub fn picture_start(
        &mut self,
        frame_num: u32,
        poc: i32,
        is_ref: bool,
        is_idr: bool,
        long_term_reference: bool,
        no_output_of_prior_pics: bool,
        mmco: bool,
        mmco_commands: Vec<H264MmcoCommand>,
    ) {
        self.refresh_frame_num_wrap(frame_num);
        self.cur = Some(CurrentPic {
            frame_num,
            poc,
            is_ref,
            is_idr,
            long_term_reference,
            no_output_of_prior_pics,
            mmco,
            mmco_commands,
        });
    }

    /// The reference list for the current picture: short-term refs in slot order
    /// (C++ BeginPicture). Must be called after `picture_start`, before `commit_current`.
    pub fn get_references(&self) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_ref())
            .map(|(i, _)| i)
            .collect()
    }

    /// Apply the current picture's reference marking (MMCO or sliding window) and the
    /// free-slots step, then return the slot the current picture will be stored into
    /// (first empty slot, C++ C.4.5.1/4.5.2). Does NOT store the current yet.
    pub fn prepare_current(&mut self) -> usize {
        self.prepare_current_avoiding(&[])
    }

    /// [`prepare_current`] that never returns (and never frees) a slot in `forbidden`
    /// — the DPB slots this picture's slices still reference. H.264 guarantees at
    /// least one unused buffer exists, so a valid slot is always found. Without this
    /// guard the sliding window can unmark and free a reference that the current
    /// slice still reads, and "first empty slot" hands back that same slot: the
    /// picture then decodes INTO its own reference (aliasing), corrupting inter
    /// prediction from the first P-frame onward.
    pub fn prepare_current_avoiding(&mut self, forbidden: &[usize]) -> usize {
        let cur = self.cur.as_ref().expect("picture_start not called").clone();
        if cur.is_idr {
            // H.264 8.2.5: when the current picture is an IDR, all short-term
            // and long-term reference pictures are unmarked. Without this the
            // previous GOP's refs stay marked; frame_num wraps modulo
            // MaxFrameNum, so an old ref can then sort ahead of the new GOP's
            // frames in the P-slice list (descending PicNum) and be selected
            // as L0[0] for the first P-frame after the IDR (corrupting it and
            // everything that references it). FFmpeg unmarks all at an IDR;
            // other backends reset their DPB explicitly on IDRs.
            for s in &mut self.slots {
                s.marking = MARKING_UNUSED;
            }
        } else if cur.is_ref {
            if cur.mmco {
                self.apply_mmco(&cur);
            } else {
                self.apply_sliding_window(&cur);
            }
        }

        // C.4.4: IDR with no_output_of_prior_pics clears all slots.
        if cur.is_idr && cur.no_output_of_prior_pics {
            for s in &mut self.slots {
                s.state = 0;
                s.marking = MARKING_UNUSED;
            }
        }

        // Free slots that are not refs, not needed for output, and not still
        // referenced by the current picture (those must stay alive until this
        // picture finishes decoding).
        for (i, s) in self.slots.iter_mut().enumerate() {
            if s.state != 0 && !s.is_ref() && !s.needed_for_output && !forbidden.contains(&i) {
                *s = H264DpbSlot::empty();
            }
        }

        // Bump (output) while the DPB is full, to make room.
        let mut guard = 0;
        while self.dpb_full() && guard < 32 {
            self.bump();
            guard += 1;
        }

        // First empty slot not referenced by the current picture.
        for i in 0..self.slots.len() {
            if self.slots[i].state == 0 && !forbidden.contains(&i) {
                return i;
            }
        }
        0
    }

    /// Store the current picture into `slot` and run the display logic
    /// (C++ dpb_picture_end C.4.5 + display_bumping).
    pub fn commit_current(&mut self, slot: usize) {
        let cur = match self.cur.take() {
            Some(c) => c,
            None => return,
        };
        // C++ picture_numbers (eq. 8-28): FrameNumWrap = FrameNum or FrameNum - MaxFrameNum.
        // For the current picture, PicNum = FrameNum (it's always the newest).
        // The stored frame_num_wrap is a placeholder; refresh_frame_num_wrap recalculates
        // all wraps on the next picture_start call. We store frame_num directly to match
        // the C++ oracle which doesn't pre-compute a persistent FrameNumWrap.
        let frame_num_wrap = cur.frame_num as i32;
        // 8.2.5.1: the current picture is committed as a long-term reference
        // when (a) it is an IDR with long_term_reference_flag (LongTermFrameIdx
        // is always 0 for IDRs), or (b) MMCO 4 assigned it a LongTermFrameIdx.
        let mut lt_idx = 0u32;
        let lt_assigned = cur
            .mmco_commands
            .iter()
            .find_map(|c| match c {
                H264MmcoCommand::AssignLongTermToCurrent { long_term_frame_idx } => {
                    lt_idx = *long_term_frame_idx;
                    Some(())
                }
                _ => None,
            })
            .is_some();
        let marking = if !cur.is_ref {
            MARKING_UNUSED
        } else if cur.long_term_reference || lt_assigned {
            MARKING_LONG
        } else {
            MARKING_SHORT
        };

        if slot < self.slots.len() {
            let s = &mut self.slots[slot];
            *s = H264DpbSlot {
                state: 3,
                frame_num: cur.frame_num,
                frame_num_wrap,
                poc: cur.poc,
                marking,
                long_term_frame_idx: lt_idx,
                needed_for_output: true,
            };
        }

        // Display: output the smallest-POC pending picture if reordering delay exceeds the limit.
        if self.reordering_delay() > self.max_num_reorder_frames {
            self.display_bump();
        }
    }

    /// Number of full-frame pictures pending output (C++ dpb_reordering_delay).
    fn reordering_delay(&self) -> u32 {
        self.slots
            .iter()
            .filter(|s| s.state == 3 && s.needed_for_output)
            .count() as u32
    }

    /// Output the smallest-POC pending picture (C++ display_bumping). Clears
    /// needed_for_output; frees the slot if the picture is not a reference.
    fn display_bump(&mut self) {
        let mut i_min = -1i64;
        let mut poc_min = i32::MAX;
        for (i, s) in self.slots.iter().enumerate() {
            if s.state != 0 && s.needed_for_output && s.poc <= poc_min {
                if s.poc == poc_min && i_min != -1 {
                    return; // duplicate poc -> bail (C++ behavior)
                }
                poc_min = s.poc;
                i_min = i as i64;
            }
        }
        if i_min >= 0 {
            let i = i_min as usize;
            // C++ display_bumping only clears needed_for_output; the slot is freed
            // by the next frame's free step (not here).
            self.slots[i].needed_for_output = false;
        }
    }

    /// Output the smallest-POC pending picture, freeing its slot if non-ref
    /// (C++ dpb_bumping). Used to make room when the DPB is full.
    fn bump(&mut self) {
        self.display_bump();
    }

    /// Sliding-window decoded reference picture marking (H.264 8.2.5.3).
    fn apply_sliding_window(&mut self, cur: &CurrentPic) {
        // FrameNum-conflict unmarking (C++ sliding_window:4138-4149): a short-term ref
        // with the same FrameNum as the current is unmarked (non-conforming stream).
        for s in self.slots.iter_mut() {
            if s.is_ref() && s.frame_num == cur.frame_num {
                s.marking = MARKING_UNUSED;
            }
        }

        // Count existing DPB refs (NOT including the current picture, which has
        // no slot yet). Spec 8.2.5.3 evicts only when the DPB already holds
        // `max_num_ref_frames` references; FFmpeg matches this exactly
        // (generate_sliding_window_mmcos: `short_ref_count >= ref_frame_count`
        // before the current picture is added to the list). Verified against
        // `ffmpeg -debug mmco` on h264_baseline: refs grow to max_num_ref_frames
        // (3) before the oldest is evicted. Eviction targets SHORT-TERM refs
        // only (8.2.5.3); long-term refs are never slid out.
        let num_refs = self.slots.iter().filter(|s| s.is_ref()).count();
        if (num_refs as u32) >= self.num_ref_frames {
            let mut imin = None;
            let mut min_wrap = i32::MAX;
            for (i, s) in self.slots.iter().enumerate() {
                if s.marking == MARKING_SHORT && s.frame_num_wrap < min_wrap {
                    min_wrap = s.frame_num_wrap;
                    imin = Some(i);
                }
            }
            if let Some(i) = imin {
                self.slots[i].marking = MARKING_UNUSED;
            }
        }
    }

    /// Adaptive (MMCO) decoded reference picture marking (H.264 8.2.5.1).
    ///
    /// MMCO 4 (assign long-term index to the CURRENT picture) cannot be applied
    /// here — the current picture has no slot yet; `commit_current` applies it.
    fn apply_mmco(&mut self, cur: &CurrentPic) {
        for cmd in &cur.mmco_commands {
            match cmd {
                H264MmcoCommand::UnmarkShortTerm {
                    difference_of_pic_nums_minus1,
                } => {
                    let pic_num_x = self.pic_num_x(cur.frame_num, *difference_of_pic_nums_minus1);
                    for s in &mut self.slots {
                        if s.is_ref() && s.marking == MARKING_SHORT && s.frame_num == pic_num_x {
                            s.marking = MARKING_UNUSED;
                        }
                    }
                }
                H264MmcoCommand::UnmarkLongTerm { long_term_frame_idx } => {
                    for s in &mut self.slots {
                        if s.state != 0
                            && s.marking == MARKING_LONG
                            && s.long_term_frame_idx == *long_term_frame_idx
                        {
                            s.marking = MARKING_UNUSED;
                        }
                    }
                }
                H264MmcoCommand::AssignLongTerm {
                    difference_of_pic_nums_minus1,
                    long_term_frame_idx,
                } => {
                    let pic_num_x = self.pic_num_x(cur.frame_num, *difference_of_pic_nums_minus1);
                    for s in &mut self.slots {
                        if s.is_ref()
                            && s.marking == MARKING_SHORT
                            && s.frame_num == pic_num_x
                        {
                            s.marking = MARKING_LONG;
                            s.long_term_frame_idx = *long_term_frame_idx;
                        }
                    }
                }
                H264MmcoCommand::SetMaxLongTermFrameIdx { .. } => {
                    // 8.2.5.1 op 5: sets MaxLongTermFrameIdxPlus1, which only
                    // constrains future long_term_pic_num values — no DPB state change.
                }
                H264MmcoCommand::AssignLongTermToCurrent { .. } => {
                    // Applied in commit_current (the current picture is not in a slot yet).
                }
                H264MmcoCommand::UnmarkAll => {
                    // 8.2.5.1 op 6: unmarks all SHORT-TERM references; long-term
                    // references are untouched.
                    for s in &mut self.slots {
                        if s.marking == MARKING_SHORT {
                            s.marking = MARKING_UNUSED;
                        }
                    }
                }
            }
        }
    }

    /// picNumX = CurrPicNum - (difference_of_pic_nums_minus1 + 1), with wraparound
    /// (C++: `picNumX = FrameNum - diff; if (picNumX < 0) picNumX += MaxFrameNum;`).
    fn pic_num_x(&self, cur_fn: u32, difference_of_pic_nums_minus1: u32) -> u32 {
        let diff = difference_of_pic_nums_minus1 + 1;
        ((cur_fn as i64 - diff as i64).rem_euclid(self.max_frame_num as i64)) as u32
    }

    /// Invalidate all slots (sequence start / reset).
    pub fn invalidate_all(&mut self) {
        for s in &mut self.slots {
            *s = H264DpbSlot::empty();
        }
        self.cur = None;
    }

    /// Build the spec 8.2.3.1 + 8.2.3.2 reference lists from the current DPB
    /// slot state (see `crate::h264_reflist::build_ref_pic_lists`). Must be
    /// called after `picture_start` so `frame_num_wrap` is up to date.
    pub fn build_ref_lists(
        &self,
        slice_type: u32,
        num_ref_idx_l0_active_minus1: u32,
        num_ref_idx_l1_active_minus1: u32,
        mod_l0: &[RefPicListModificationEntry],
        mod_l1: &[RefPicListModificationEntry],
    ) -> RefPicLists {
        let states: Vec<DpbRefState> = self
            .slots
            .iter()
            .map(|s| DpbRefState {
                frame_num: s.frame_num,
                frame_num_wrap: s.frame_num_wrap,
                poc: s.poc,
                marking: if s.state == 0 {
                    MARKING_UNUSED
                } else {
                    s.marking
                },
                long_term_frame_idx: s.long_term_frame_idx,
            })
            .collect();
        let curr_fn = self.cur.as_ref().map(|c| c.frame_num).unwrap_or(0);
        let curr_poc = self.cur.as_ref().map(|c| c.poc).unwrap_or(0);
        build_ref_pic_lists(
            &states,
            slice_type,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            mod_l0,
            mod_l1,
            curr_fn,
            curr_poc,
            self.max_frame_num,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DPB with 8 slots, max 3 short-term refs, MaxFrameNum 64.
    fn dpb() -> H264Dpb {
        H264Dpb::new(8, 8, 3, 64)
    }

    fn slot(frame_num: u32, poc: i32, marking: u8, lt_idx: u32) -> H264DpbSlot {
        H264DpbSlot {
            state: 3,
            frame_num,
            frame_num_wrap: frame_num as i32,
            poc,
            marking,
            long_term_frame_idx: lt_idx,
            needed_for_output: false,
        }
    }

    /// Commit a plain short-term reference picture into `slot`.
    fn commit_short_ref(dpb: &mut H264Dpb, slot: usize, frame_num: u32, poc: i32) {
        dpb.picture_start(frame_num, poc, true, false, false, false, false, Vec::new());
        dpb.commit_current(slot);
    }

    #[test]
    fn mmco_op3_assigns_long_term_idx_to_picnumx() {
        let mut dpb = dpb();
        commit_short_ref(&mut dpb, 0, 4, 8);

        // Current frame_num=10, diff_minus1=5 -> picNumX = 10 - 6 = 4 (slot 0).
        dpb.picture_start(
            10,
            20,
            true,
            false,
            false,
            false,
            true,
            vec![H264MmcoCommand::AssignLongTerm {
                difference_of_pic_nums_minus1: 5,
                long_term_frame_idx: 2,
            }],
        );
        dpb.prepare_current_avoiding(&[]);

        assert_eq!(dpb.slots[0].marking, MARKING_LONG);
        assert_eq!(dpb.slots[0].long_term_frame_idx, 2);
    }

    #[test]
    fn mmco_op2_unmarks_long_term() {
        let mut dpb = dpb();
        dpb.slots[0] = slot(1, 2, MARKING_LONG, 3);

        dpb.picture_start(
            10,
            20,
            true,
            false,
            false,
            false,
            true,
            vec![H264MmcoCommand::UnmarkLongTerm { long_term_frame_idx: 3 }],
        );
        dpb.prepare_current_avoiding(&[]);

        assert_eq!(dpb.slots[0].marking, MARKING_UNUSED);
    }

    #[test]
    fn mmco_op6_unmarks_short_only() {
        let mut dpb = dpb();
        dpb.slots[0] = slot(1, 2, MARKING_SHORT, 0);
        dpb.slots[1] = slot(2, 4, MARKING_LONG, 0);

        dpb.picture_start(
            10,
            20,
            true,
            false,
            false,
            false,
            true,
            vec![H264MmcoCommand::UnmarkAll],
        );
        dpb.prepare_current_avoiding(&[]);

        assert_eq!(dpb.slots[0].marking, MARKING_UNUSED);
        assert_eq!(dpb.slots[1].marking, MARKING_LONG, "op 6 must keep long-term refs");
    }

    #[test]
    fn mmco_op4_marks_current_as_long_term() {
        let mut dpb = dpb();
        dpb.picture_start(
            1,
            2,
            true,
            false,
            false,
            false,
            true,
            vec![H264MmcoCommand::AssignLongTermToCurrent { long_term_frame_idx: 3 }],
        );
        let s = dpb.prepare_current_avoiding(&[]);
        dpb.commit_current(s);

        assert_eq!(dpb.slots[s].marking, MARKING_LONG);
        assert_eq!(dpb.slots[s].long_term_frame_idx, 3);
    }

    #[test]
    fn idr_long_term_reference_flag_commits_lt_idx_zero() {
        let mut dpb = dpb();
        // Keep the previous ref pending output so it is not freed before the IDR
        // (with the default max_num_reorder_frames=0 every frame displays at once).
        dpb.max_num_reorder_frames = 4;
        commit_short_ref(&mut dpb, 0, 4, 8);

        dpb.picture_start(8, 16, true, /* is_idr */ true, /* lt_ref */ true, false, false, Vec::new());
        let s = dpb.prepare_current_avoiding(&[]);
        dpb.commit_current(s);

        assert_eq!(dpb.slots[s].marking, MARKING_LONG);
        assert_eq!(dpb.slots[s].long_term_frame_idx, 0);
        // IDR unmarked the previous short-term ref.
        assert_eq!(dpb.slots[0].marking, MARKING_UNUSED);
    }

    #[test]
    fn ref_lists_append_long_term_in_ascending_idx() {
        let mut dpb = dpb();
        dpb.slots[0] = slot(1, 2, MARKING_LONG, 2);
        dpb.slots[1] = slot(2, 4, MARKING_LONG, 1);
        dpb.slots[2] = slot(5, 10, MARKING_SHORT, 0);

        // P slice: L0 = short-term (descending PicNum) then long-term
        // (ascending LongTermFrameIdx). active=3 so the list is exact (no
        // backfill padding).
        let lists = dpb.build_ref_lists(0, 2, 0, &[], &[]);
        assert_eq!(
            lists.l0.iter().map(|r| r.slot).collect::<Vec<_>>(),
            vec![2, 1, 0],
            "ST first, then LT idx 1 before idx 2"
        );
        assert!(!lists.l0[0].is_long_term);
        assert!(lists.l0[1].is_long_term && lists.l0[2].is_long_term);
    }

    #[test]
    fn sliding_window_never_evicts_long_term() {
        let mut dpb = dpb();
        // 3 refs total (>= max_num_ref_frames=3) triggers eviction.
        dpb.slots[0] = slot(1, 2, MARKING_SHORT, 0);
        dpb.slots[1] = slot(2, 4, MARKING_SHORT, 0);
        dpb.slots[2] = slot(3, 6, MARKING_LONG, 0);

        dpb.picture_start(10, 20, true, false, false, false, false, Vec::new());
        dpb.prepare_current_avoiding(&[]);

        assert_eq!(dpb.slots[0].marking, MARKING_UNUSED, "oldest short-term evicted");
        assert_eq!(dpb.slots[1].marking, MARKING_SHORT);
        assert_eq!(dpb.slots[2].marking, MARKING_LONG, "long-term ref must survive");
    }

    #[test]
    fn mmco_op1_does_not_touch_long_term_with_same_framenum() {
        let mut dpb = dpb();
        dpb.slots[0] = slot(4, 8, MARKING_LONG, 0);

        // Current frame_num=10, diff_minus1=5 -> picNumX = 4. The only slot with
        // that FrameNum is a LONG-TERM ref: op 1 unmarks short-term refs only.
        dpb.picture_start(
            10,
            20,
            true,
            false,
            false,
            false,
            true,
            vec![H264MmcoCommand::UnmarkShortTerm { difference_of_pic_nums_minus1: 5 }],
        );
        dpb.prepare_current_avoiding(&[]);

        assert_eq!(dpb.slots[0].marking, MARKING_LONG);
    }
}

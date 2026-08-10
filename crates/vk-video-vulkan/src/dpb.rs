//! Decoded Picture Buffer (DPB) management for reference frame tracking.

use ash::vk;
use crate::access_unit::H264MmcoCommand;

/// Type of the last access to a DPB slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LastAccessType {
    DecodeWrite,
    TransferRead,
    None,
}

/// A DPB entry tracking a reference frame.
#[derive(Debug, Clone)]
pub struct DpbEntry {
    pub frame_num: u32,
    pub pic_order_cnt: [i32; 2],
    pub slot_index: u32,
    pub is_valid: bool,
    pub image_view: vk::ImageView,
    pub image: vk::Image,
    pub current_layout: vk::ImageLayout,
    pub last_access: LastAccessType,
}

/// DPB manager for tracking reference frames during decode.
pub struct DpbManager {
    pub entries: Vec<DpbEntry>,
    max_dpb_slots: u32,
    pub next_slot: u32,
    max_num_ref_frames: u32,
    /// max_frame_num from SPS (1 << (log2_max_frame_num_minus4 + 4))
    /// Used for wraparound-aware frame_num comparison.
    max_frame_num: u32,
}

impl DpbManager {
    pub fn new(max_dpb_slots: u32) -> Self {
        Self {
            entries: (0..max_dpb_slots as usize)
                .map(|i| DpbEntry {
                    frame_num: 0,
                    pic_order_cnt: [0, 0],
                    slot_index: i as u32,
                    is_valid: false,
                    image_view: vk::ImageView::null(),
                    image: vk::Image::null(),
                    current_layout: vk::ImageLayout::UNDEFINED,
                    last_access: LastAccessType::None,
                })
                .collect(),
            max_dpb_slots,
            next_slot: 0,
            max_num_ref_frames: 16,
            max_frame_num: 64,
        }
    }

    pub fn set_max_num_ref_frames(&mut self, max_num_ref_frames: u32) {
        self.max_num_ref_frames = max_num_ref_frames;
    }

    /// Set max_frame_num from SPS for wraparound-aware comparisons.
    /// max_frame_num = 1 << (log2_max_frame_num_minus4 + 4)
    pub fn set_max_frame_num(&mut self, max_frame_num: u32) {
        self.max_frame_num = max_frame_num;
    }

    /// Compute a wraparound-aware "wrapped" frame number relative to current.
    /// Returns negative values for frames from the previous wrap cycle.
    /// Based on VulkanH264Parser.cpp FrameNumWrap computation.
    fn frame_num_wrap(&self, entry_frame_num: u32, current_frame_num: u32) -> i32 {
        if entry_frame_num > current_frame_num {
            // Entry is from previous wrap cycle
            (entry_frame_num as i32) - (self.max_frame_num as i32)
        } else {
            // Entry is from current wrap cycle
            entry_frame_num as i32
        }
    }

    /// Apply sliding window decoded reference picture marking.
    /// Only used when adaptive_ref_pic_marking_mode_flag is false.
    pub fn apply_sliding_window(&mut self, current_frame_num: u32) {
        let max_refs = self.max_num_ref_frames.max(1) as usize;

        let num_short_term = self
            .entries
            .iter()
            .filter(|e| e.is_valid && e.frame_num != current_frame_num)
            .count();

        if num_short_term >= max_refs {
            let mut oldest_idx: Option<usize> = None;
            let mut oldest_wrap = i32::MAX;

            for (i, entry) in self.entries.iter().enumerate() {
                if entry.is_valid && entry.frame_num != current_frame_num {
                    let wrap = self.frame_num_wrap(entry.frame_num, current_frame_num);
                    if wrap < oldest_wrap {
                        oldest_wrap = wrap;
                        oldest_idx = Some(i);
                    }
                }
            }

            if let Some(idx) = oldest_idx {
                self.entries[idx].is_valid = false;
            }
        }
    }

    /// Apply H.264 MMCO (Memory Management Control Operations) commands.
    /// See H.264 spec 8.2.5.4 for details.
    ///
    /// `current_frame_num` is the frame_num of the current picture being decoded.
    /// `current_slot_index` is the slot index of the current picture.
    ///
    /// Only used when adaptive_ref_pic_marking_mode_flag is true.
    pub fn apply_mmco(
        &mut self,
        current_frame_num: u32,
        current_slot_index: u32,
        mmco_commands: &[H264MmcoCommand],
    ) {
        eprintln!("[DEBUG] DPB::apply_mmco: current_frame_num={}, current_slot={}, commands={}",
                  current_frame_num, current_slot_index, mmco_commands.len());

        for cmd in mmco_commands {
            eprintln!("[DEBUG]   MMCO command: {:?}", cmd);
        }

        for cmd in mmco_commands {
            match cmd {
                // MMCO 1: Mark short-term reference as unused
                // picNumX = CurrPicNum - (difference_of_pic_nums_minus1 + 1)
                H264MmcoCommand::UnmarkShortTerm { difference_of_pic_nums_minus1 } => {
                    let pic_num_x = self.compute_pic_num(current_frame_num, *difference_of_pic_nums_minus1);
                    eprintln!("[DEBUG]   MMCO 1: unmark short-term picNumX={}", pic_num_x);
                    for entry in &mut self.entries {
                        if entry.is_valid && entry.frame_num == pic_num_x {
                            entry.is_valid = false;
                            eprintln!("[DEBUG]     invalidated slot {} (frame_num={})", entry.slot_index, entry.frame_num);
                        }
                    }
                }

                // MMCO 2: Mark long-term reference as unused
                // (We don't fully track long-term refs, but mark by frame_num if known)
                H264MmcoCommand::UnmarkLongTerm { long_term_frame_idx } => {
                    eprintln!("[DEBUG]   MMCO 2: unmark long-term long_term_frame_idx={} (not fully tracked)", long_term_frame_idx);
                    // For now, skip - long-term reference tracking would require additional state
                }

                // MMCO 3: Assign LongTermFrameIdx to short-term reference
                H264MmcoCommand::AssignLongTerm { difference_of_pic_nums_minus1, long_term_frame_idx } => {
                    let pic_num_x = self.compute_pic_num(current_frame_num, *difference_of_pic_nums_minus1);
                    eprintln!("[DEBUG]   MMCO 3: assign LongTermFrameIdx={} to picNumX={} (not fully tracked)",
                              long_term_frame_idx, pic_num_x);
                    // For now, skip - long-term reference tracking would require additional state
                }

                // MMCO 4: Set MaxLongTermFrameIdx
                H264MmcoCommand::SetMaxLongTermFrameIdx { max_long_term_frame_idx_plus1 } => {
                    eprintln!("[DEBUG]   MMCO 4: set MaxLongTermFrameIdx={} (not fully tracked)",
                              max_long_term_frame_idx_plus1);
                    // For now, skip - long-term reference tracking would require additional state
                }

                // MMCO 5: Unmark all references
                H264MmcoCommand::UnmarkAll => {
                    eprintln!("[DEBUG]   MMCO 5: unmark ALL references");
                    for entry in &mut self.entries {
                        if entry.is_valid {
                            entry.is_valid = false;
                        }
                    }
                }

                // MMCO 6: Assign LongTermFrameIdx to current picture
                H264MmcoCommand::AssignLongTermToCurrent { long_term_frame_idx } => {
                    eprintln!("[DEBUG]   MMCO 6: assign LongTermFrameIdx={} to current (slot {}) (not fully tracked)",
                              long_term_frame_idx, current_slot_index);
                    // For now, skip - long-term reference tracking would require additional state
                }
            }
        }
    }

    /// Compute picNumX from difference_of_pic_nums_minus1.
    /// picNumX = CurrPicNum - (difference_of_pic_nums_minus1 + 1)
    /// Handles wraparound using max_frame_num.
    fn compute_pic_num(&self, current_frame_num: u32, difference_of_pic_nums_minus1: u32) -> u32 {
        let diff = difference_of_pic_nums_minus1 + 1;
        if current_frame_num >= diff {
            current_frame_num - diff
        } else {
            // Wraparound case
            (self.max_frame_num + current_frame_num) - diff
        }
    }

    /// Find an empty slot or recycle the oldest reference.
    ///
    /// `protected_pocs` is a list of reference POCs that the current frame needs.
    /// Slots containing frames with these POCs will NOT be recycled, preventing
    /// destruction of reference pictures needed for the current decode.
    pub fn find_or_recycle_slot(&mut self, protected_pocs: &[i32]) -> Option<u32> {
        for i in 0..self.max_dpb_slots as usize {
            if !self.entries[i].is_valid {
                return Some(i as u32);
            }
        }

        let mut oldest_idx = None;
        let mut oldest_poc = i32::MAX;
        for i in 0..self.max_dpb_slots as usize {
            if self.entries[i].is_valid {
                let poc = self.entries[i].pic_order_cnt[0];
                if protected_pocs.contains(&poc) {
                    continue;
                }
                if poc < oldest_poc {
                    oldest_poc = poc;
                    oldest_idx = Some(i as u32);
                }
            }
        }

        oldest_idx
    }

    /// Find an empty slot or recycle the oldest reference, excluding specific slots.
    ///
    /// For VP9: `exclude_slots` is a list of DPB slot indices that must not be used
    /// as output (they contain reference frames needed for the current decode).
    pub fn find_or_recycle_slot_excluding(&mut self, exclude_slots: &[i32]) -> Option<u32> {
        // First try to find an empty slot that is not excluded
        for i in 0..self.max_dpb_slots as usize {
            if !self.entries[i].is_valid && !exclude_slots.contains(&(i as i32)) {
                return Some(i as u32);
            }
        }

        // Recycle the oldest valid slot that is not excluded
        let mut oldest_idx = None;
        let mut oldest_frame_num = u32::MAX;
        for i in 0..self.max_dpb_slots as usize {
            if self.entries[i].is_valid
                && !exclude_slots.contains(&(i as i32))
                && self.entries[i].frame_num < oldest_frame_num
            {
                oldest_frame_num = self.entries[i].frame_num;
                oldest_idx = Some(i as u32);
            }
        }

        oldest_idx
    }

    /// Mark all entries as invalid (for IDR frames).
    pub fn invalidate_all(&mut self) {
        for entry in &mut self.entries {
            entry.is_valid = false;
            entry.current_layout = vk::ImageLayout::UNDEFINED;
            entry.last_access = LastAccessType::None;
        }
    }

    pub fn get_slot_layout(&self, slot_index: u32) -> vk::ImageLayout {
        self.entries[slot_index as usize].current_layout
    }

    pub fn set_slot_layout(&mut self, slot_index: u32, layout: vk::ImageLayout) {
        self.entries[slot_index as usize].current_layout = layout;
    }

    pub fn get_slot_last_access(&self, slot_index: u32) -> LastAccessType {
        self.entries[slot_index as usize].last_access
    }

    pub fn set_slot_last_access(&mut self, slot_index: u32, access: LastAccessType) {
        self.entries[slot_index as usize].last_access = access;
    }

    pub fn find_by_frame_num(&self, frame_num: u32) -> Option<(usize, &DpbEntry)> {
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.is_valid && entry.frame_num == frame_num {
                return Some((i, entry));
            }
        }
        None
    }

    pub fn get_references(&self) -> Vec<&DpbEntry> {
        self.entries.iter().filter(|e| e.is_valid).collect()
    }

    pub fn get_references_mut(&mut self) -> Vec<&mut DpbEntry> {
        self.entries.iter_mut().filter(|e| e.is_valid).collect()
    }

    /// Register a frame in a DPB slot (VP9-style, without POC tracking).
    pub fn register_frame(&mut self, slot: u32, frame_count: u32) {
        if slot < self.entries.len() as u32 {
            self.entries[slot as usize].is_valid = true;
            self.entries[slot as usize].frame_num = frame_count;
            self.entries[slot as usize].slot_index = slot;
        }
    }
}

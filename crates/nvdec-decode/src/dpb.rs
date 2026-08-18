//! Decoded Picture Buffer (DPB) management for NVDEC.
//!
//! Tracks reference frames and applies H.264 Memory Management Control
//! Operations (MMCO) to maintain correct DPB state for hardware decode.

use crate::ffi::CUVIDH264DPBENTRY;
use std::os::raw::c_int;
#[cfg(not(test))]
use vk_video_parser::h264::SliceHeader;
#[cfg(test)]
use vk_video_parser::h264::{DecRefPicMarkingEntry, SliceHeader};

/// Single entry in the NVDEC DPB.
#[derive(Debug, Clone)]
pub struct NvdecDpbEntry {
    pub pic_index: i32,
    pub frame_num: u32,
    pub pic_order_cnt: i32,
    pub is_reference: bool,
    pub is_long_term: bool,
    pub is_valid: bool,
    /// Long-term frame index (long_term_pic_num) assigned by MMCO Op 3 or Op 6.
    /// Only meaningful when is_long_term is true.
    pub long_term_frame_idx: Option<u32>,
    /// Unique, non-wrapping picture id assigned at add_frame time.
    /// Unlike pic_index (which wraps at max_decode_surfaces), seq is
    /// monotonically increasing and identifies a specific decoded frame.
    pub seq: i32,
    /// Whether this frame has been extracted (presented) by the decoder.
    /// A decode surface may only be reused once its occupant is extracted
    /// and no longer a valid reference.
    pub extracted: bool,
}

/// DPB manager for NVDEC H.264 decoding.
///
/// Maintains the Decoded Picture Buffer state and applies MMCO commands
/// from slice headers. Converts DPB state to CUVID format for
/// [`CUVIDH264DPBENTRY`](crate::ffi::CUVIDH264DPBENTRY).
///
/// Uses a Vec-based storage so reference frames are never overwritten
/// by ring-buffer wraparound. The DPB is limited to `max_dpb_size`
/// reference frames via sliding-window eviction.
pub struct NvdecDpbManager {
    entries: Vec<NvdecDpbEntry>,
    /// Maximum number of reference frames (from SPS max_num_ref_frames).
    max_dpb_size: usize,
    next_pic_index: i32,
    /// Monotonically increasing counter for unique (non-wrapping) picture ids.
    next_seq: i32,
    max_frame_num: u32,
    current_is_long_term: bool,
    /// Long-term frame index to assign to the next picture added (from MMCO Op 9).
    current_long_term_frame_idx: Option<u32>,
    /// Maximum decode surfaces for CurrPicIdx wrapping.
    max_decode_surfaces: i32,
}

impl NvdecDpbManager {
    /// Create a new DPB manager with the given maximum number of reference frames.
    pub fn new(max_dpb_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_dpb_size,
            next_pic_index: 0,
            next_seq: 0,
            max_frame_num: 65536,
            current_is_long_term: false,
            current_long_term_frame_idx: None,
            max_decode_surfaces: 32, // Default to MAX_DECODE_SURFACES
        }
    }

    /// Set the maximum number of decode surfaces for CurrPicIdx wrapping.
    pub fn set_max_decode_surfaces(&mut self, max_decode_surfaces: i32) {
        self.max_decode_surfaces = max_decode_surfaces;
    }

    /// Set max_frame_num from SPS for wraparound-aware frame_num comparison.
    /// max_frame_num = 1 << (log2_max_frame_num_minus4 + 4)
    pub fn set_max_frame_num(&mut self, max_frame_num: u32) {
        self.max_frame_num = max_frame_num;
    }

    /// Set max_dpb_size from SPS max_num_ref_frames.
    pub fn set_max_dpb_size(&mut self, max_dpb_size: usize) {
        self.max_dpb_size = max_dpb_size;
    }

    /// Add a frame to the DPB and return its NVDEC picture index.
    ///
    /// For reference frames, applies sliding-window eviction when the DPB
    /// is full (evicts the oldest short-term reference).
    pub fn add_frame(&mut self, frame_num: u32, poc: i32, is_reference: bool) -> i32 {
        let pic_index = self.choose_surface();
        self.next_pic_index += 1;
        let seq = self.next_seq;
        self.next_seq += 1;

        if is_reference {
            // Apply sliding window: evict oldest short-term references
            // until we have room for the new one.
            while self.count_references() >= self.max_dpb_size {
                self.evict_oldest_short_term();
            }
        }

        let is_long_term = self.current_is_long_term;
        let long_term_frame_idx = self.current_long_term_frame_idx.take();
        self.current_is_long_term = false;

        self.entries.push(NvdecDpbEntry {
            pic_index,
            frame_num,
            pic_order_cnt: poc,
            is_reference,
            is_long_term,
            is_valid: true,
            long_term_frame_idx,
            seq,
            extracted: false,
        });

        pic_index
    }

    /// Choose the decode surface for the next picture.
    ///
    /// A surface is only reused when its current occupant is (a) already
    /// extracted and (b) no longer a valid reference. Scan from surface 0
    /// upward for the first such surface — this matches the NVIDIA cuvid
    /// parser's surface assignment (verified against ground truth), which
    /// reuses the lowest-numbered recyclable surface rather than a
    /// round-robin. If no surface is recyclable, fall back to the next
    /// round-robin slot (overwriting its occupant, as wraparound does).
    ///
    /// This is a pure function of DPB state, so it returns the same value
    /// when called from [`get_next_pic_index`](Self::get_next_pic_index)
    /// (pre-decode) and from [`add_frame`](Self::add_frame) (post-decode).
    fn choose_surface(&self) -> i32 {
        for idx in 0..self.max_decode_surfaces {
            if let Some(e) = self.occupant_of(idx) {
                // Reusable only if already extracted AND not a valid reference.
                if !e.extracted || (e.is_valid && e.is_reference) {
                    continue;
                }
            }
            return idx;
        }
        self.next_pic_index % self.max_decode_surfaces
    }

    /// The most recent entry occupying the given decode surface, if any.
    fn occupant_of(&self, pic_index: i32) -> Option<&NvdecDpbEntry> {
        self.entries.iter().rev().find(|e| e.pic_index == pic_index)
    }


    /// Sequence id of the most recently added frame (`next_seq - 1`).
    pub fn last_seq(&self) -> i32 {
        self.next_seq - 1
    }

    /// Mark the entry with the given unique sequence id as extracted,
    /// allowing its decode surface to be recycled.
    pub fn mark_extracted(&mut self, seq: i32) {
        for e in &mut self.entries {
            if e.seq == seq {
                e.extracted = true;
            }
        }
    }

    /// Compute picNumX from difference_of_pic_nums_minus1 with wraparound support.
    ///
    /// Per H.264 8.2.1 (MMCO), picNumX =
    /// (frameNumCurrPic - differenceOfPicNums) modulo MaxFrameNum.
    /// Uses wrapping arithmetic so a garbage (huge) difference_of_pic_nums_minus1
    /// from an upstream mis-parse can never panic.
    fn compute_pic_num(&self, current_frame_num: u32, difference_of_pic_nums_minus1: u32) -> u32 {
        let diff = difference_of_pic_nums_minus1.wrapping_add(1);
        if current_frame_num >= diff {
            current_frame_num - diff
        } else if self.max_frame_num == 0 {
            0
        } else {
            (self.max_frame_num.wrapping_add(current_frame_num)).wrapping_sub(diff)
                % self.max_frame_num
        }
    }

    /// Apply Memory Management Control Operations from the slice header.
    ///
    /// If `is_idr` is true, all entries are invalidated.
    /// If `is_idr` is true and `idr_long_term_ref_flag` is true, the current
    /// picture will be marked as a long-term reference when added to the DPB.
    /// Otherwise, each entry in `slh.dec_ref_pic_marking` is applied in order.
    /// Apply the IDR reset: invalidate all entries and set up long-term
    /// reference state for the IDR picture. Must be called BEFORE adding the
    /// IDR to the DPB (so old references are cleared first).
    pub fn apply_idr_reset(&mut self, idr_long_term_ref_flag: bool) {
        self.reset();
        // Per H.264 spec 7.4.3.2: if long_term_reference_flag is 1,
        // the current picture is a long-term reference with auto-assigned
        // long_term_pic_num (smallest non-negative value not used).
        if idr_long_term_ref_flag {
            self.current_is_long_term = true;
            let used_indices: std::collections::HashSet<u32> = self.entries
                .iter()
                .filter(|e| e.is_valid && e.is_long_term)
                .filter_map(|e| e.long_term_frame_idx)
                .collect();
            let mut idx = 0u32;
            while used_indices.contains(&idx) {
                idx += 1;
            }
            self.current_long_term_frame_idx = Some(idx);
        }
    }

    /// Apply the non-IDR Memory Management Control Operations from the slice
    /// header. Must be called AFTER the current picture is added to the DPB,
    /// so the operations affect the DPB state seen by subsequent pictures.
    ///
    /// Implements the H.264 spec 8.2.1 `memory_management_control_operation`
    /// values:
    ///   1 = unmark_short_term                 (value: difference_of_pic_nums_minus1)
    ///   2 = unmark_long_term                  (value: long_term_pic_num)
    ///   3 = set_num_long_term                 (value: max_long_term_frame_idx_plus1)
    ///   4 = mark_short_term_frame_num         (value: difference_of_pic_nums_minus1)
    ///   5 = mark_long_term_frame_num          (value: long_term_pic_num)
    ///   6 = mark_future_short_term            (value: difference_of_pic_nums_minus1)
    ///   7 = mark_future_long_term             (value: long_term_pic_num)
    ///   8 = unmark_all_short_term             (no value)
    ///   9 = unmark_all_long_term              (value: long_term_pic_num)
    /// For ops 1, 4, 6: PicCntX = (frame_num - difference_of_pic_nums_minus1 - 1) mod MaxFrameNum.
    pub fn apply_mmco_ops(&mut self, frame_num: u32, slh: &SliceHeader) {
        for entry in &slh.dec_ref_pic_marking {
            // A `value` this large is never legitimate (difference_of_pic_nums_minus1
            // is bounded by MaxFrameNum in practice, long_term_pic_num by 16); it
            // indicates an upstream mis-parse. Treat the op as a no-op rather than
            // risk mis-evicting DPB state.
            if entry.value > 1024 {
                log::debug!(
                    "MMCO op {}: implausible value={} (frame_num={}); ignoring",
                    entry.memory_management_control_operation,
                    entry.value,
                    frame_num
                );
                continue;
            }
            match entry.memory_management_control_operation {
                // Op 1: unmark_short_term — invalidate the short-term picture
                // with picNumX.
                1 => {
                    let pic_num_x = self.compute_pic_num(frame_num, entry.value);
                    for e in &mut self.entries {
                        if e.is_valid && !e.is_long_term && e.frame_num == pic_num_x {
                            e.is_valid = false;
                            e.is_reference = false;
                        }
                    }
                }

                // Op 2: unmark_long_term — invalidate the long-term picture with
                // the given long_term_pic_num.
                2 => {
                    for e in &mut self.entries {
                        if e.is_valid
                            && e.is_long_term
                            && e.long_term_frame_idx == Some(entry.value)
                        {
                            e.is_valid = false;
                            e.is_reference = false;
                        }
                    }
                }

                // Op 3: set_num_long_term — invalidate all long-term pictures
                // with long_term_pic_num >= max_long_term_frame_idx_plus1.
                3 => {
                    for e in &mut self.entries {
                        if e.is_valid && e.is_long_term {
                            if let Some(idx) = e.long_term_frame_idx {
                                if idx >= entry.value {
                                    e.is_valid = false;
                                    e.is_reference = false;
                                }
                            }
                        }
                    }
                }

                // Op 4: mark_short_term_frame_num — mark the short-term picture
                // with picNumX as a reference.
                4 => {
                    let pic_num_x = self.compute_pic_num(frame_num, entry.value);
                    for e in &mut self.entries {
                        if e.is_valid && !e.is_long_term && e.frame_num == pic_num_x {
                            e.is_reference = true;
                        }
                    }
                }

                // Op 5: mark_long_term_frame_num — mark the long-term picture
                // with the given long_term_pic_num as a reference.
                5 => {
                    for e in &mut self.entries {
                        if e.is_valid
                            && e.is_long_term
                            && e.long_term_frame_idx == Some(entry.value)
                        {
                            e.is_reference = true;
                        }
                    }
                }

                // Op 6: mark_future_short_term — mark the short-term picture
                // with picNumX as a reference.
                6 => {
                    let pic_num_x = self.compute_pic_num(frame_num, entry.value);
                    for e in &mut self.entries {
                        if e.is_valid && !e.is_long_term && e.frame_num == pic_num_x {
                            e.is_reference = true;
                        }
                    }
                }

                // Op 7: mark_future_long_term — mark the long-term picture with
                // the given long_term_pic_num as a reference.
                7 => {
                    for e in &mut self.entries {
                        if e.is_valid
                            && e.is_long_term
                            && e.long_term_frame_idx == Some(entry.value)
                        {
                            e.is_reference = true;
                        }
                    }
                }

                // Op 8: unmark_all_short_term — invalidate all short-term pictures.
                8 => {
                    for e in &mut self.entries {
                        if e.is_valid && !e.is_long_term {
                            e.is_valid = false;
                            e.is_reference = false;
                        }
                    }
                }

                // Op 9: unmark_all_long_term — invalidate all long-term pictures
                // with long_term_frame_idx >= long_term_pic_num.
                9 => {
                    for e in &mut self.entries {
                        if e.is_valid && e.is_long_term {
                            if let Some(idx) = e.long_term_frame_idx {
                                if idx >= entry.value {
                                    e.is_valid = false;
                                    e.is_reference = false;
                                }
                            }
                        }
                    }
                }

                // Op 0: End of MMCO list (should not appear in parsed entries)
                0 => {}

                _ => {}
            }
        }
    }

    /// Apply Memory Management Control Operations from the slice header.
    ///
    /// If `is_idr` is true, all entries are invalidated (call this BEFORE
    /// adding the picture). Otherwise, each entry in `slh.dec_ref_pic_marking`
    /// is applied (call this AFTER adding the picture). Kept for backward
    /// compatibility; prefer [`apply_idr_reset`](Self::apply_idr_reset) /
    /// [`apply_mmco_ops`](Self::apply_mmco_ops) with explicit timing.
    pub fn apply_mmco(
        &mut self,
        frame_num: u32,
        slh: &SliceHeader,
        is_idr: bool,
        idr_long_term_ref_flag: bool,
    ) {
        if is_idr {
            self.apply_idr_reset(idr_long_term_ref_flag);
        } else {
            self.apply_mmco_ops(frame_num, slh);
        }
    }

    /// Convert DPB state to CUVID format for CUVIDPICPARAMS.
    ///
    /// Returns an array of 16 [`CUVIDH264DPBENTRY`] values matching the
    /// convention used by the NVIDIA cuvid parser:
    /// - Valid reference frames: `PicIdx` = surface index, `FrameIdx` = frame_num,
    ///   `not_existing = 0`, `used_for_reference = 3` (both fields).
    /// - Empty slots: `PicIdx = -1`, `not_existing = 0`, `used_for_reference = 0`.
    ///   (Emptiness is signaled by `PicIdx = -1`, not by `not_existing`.)
    pub fn to_cuvid_dpb_entries(&self) -> [CUVIDH264DPBENTRY; 16] {
        let mut cuvid_entries = [CUVIDH264DPBENTRY {
            PicIdx: -1,
            FrameIdx: 0,
            is_long_term: 0,
            not_existing: 0,
            used_for_reference: 0,
            FieldOrderCnt: [0, 0],
        }; 16];

        let mut cuvid_idx = 0;
        for entry in self.entries.iter() {
            if cuvid_idx >= 16 {
                break;
            }
            // Only valid reference frames appear in the CUVID DPB
            if entry.is_valid && entry.is_reference {
                cuvid_entries[cuvid_idx] = CUVIDH264DPBENTRY {
                    PicIdx: entry.pic_index as c_int,
                    FrameIdx: entry.frame_num as c_int,
                    is_long_term: if entry.is_long_term { 1 } else { 0 },
                    not_existing: 0,
                    used_for_reference: 3,
                    FieldOrderCnt: [entry.pic_order_cnt as c_int, entry.pic_order_cnt as c_int],
                };
                cuvid_idx += 1;
            }
        }

        cuvid_entries
    }

    /// Reset all DPB entries to invalid state.
    ///
    /// Only the validity/reference flags are cleared; frame_num, POC and
    /// other metadata are preserved so entries can still be looked up by
    /// seq for presentation metadata after an IDR reset.
    pub fn reset(&mut self) {
        for entry in &mut self.entries {
            entry.is_valid = false;
            entry.is_reference = false;
            entry.is_long_term = false;
            entry.long_term_frame_idx = None;
        }
        self.current_is_long_term = false;
        self.current_long_term_frame_idx = None;
    }

    /// Return the next available NVDEC picture index (wrapped to max_decode_surfaces).
    ///
    /// The CurrPicIdx must be in range [0, ulNumDecodeSurfaces) for cuvidDecodePicture.
    /// The driver internally manages surface recycling.
    pub fn get_next_pic_index(&self) -> i32 {
        self.choose_surface()
    }

    /// Get the number of DPB entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get a reference to the entry at the given index.
    pub fn get_entry(&self, idx: usize) -> Option<&NvdecDpbEntry> {
        self.entries.get(idx)
    }

    /// Look up a DPB entry by its NVDEC picture index.
    ///
    /// Note: pic_index wraps at max_decode_surfaces, so this returns the
    /// OLDEST entry with that index. Prefer [`get_entry_by_seq`](Self::get_entry_by_seq)
    /// when identifying a specific decoded frame.
    pub fn get_entry_by_pic_index(&self, pic_index: i32) -> Option<&NvdecDpbEntry> {
        self.entries.iter().find(|e| e.pic_index == pic_index)
    }

    /// Look up a DPB entry by its unique (non-wrapping) sequence id.
    pub fn get_entry_by_seq(&self, seq: i32) -> Option<&NvdecDpbEntry> {
        self.entries.iter().find(|e| e.seq == seq)
    }

    /// Iterate over all valid DPB entries.
    pub fn valid_entries(&self) -> impl Iterator<Item = &NvdecDpbEntry> {
        self.entries.iter().filter(|e| e.is_valid)
    }

    /// Count the number of valid reference frames in the DPB.
    fn count_references(&self) -> usize {
        self.entries.iter()
            .filter(|e| e.is_valid && e.is_reference)
            .count()
    }

    /// Evict the oldest short-term reference frame (by POC).
    ///
    /// Long-term references are never evicted by sliding window.
    fn evict_oldest_short_term(&mut self) {
        let mut oldest_idx = None;
        let mut oldest_poc = i32::MAX;

        for (i, entry) in self.entries.iter().enumerate() {
            if entry.is_valid && entry.is_reference && !entry.is_long_term {
                if entry.pic_order_cnt < oldest_poc {
                    oldest_poc = entry.pic_order_cnt;
                    oldest_idx = Some(i);
                }
            }
        }

        if let Some(idx) = oldest_idx {
            self.entries[idx].is_valid = false;
            self.entries[idx].is_reference = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_slice_header(mmcos: Vec<DecRefPicMarkingEntry>) -> SliceHeader {
        SliceHeader {
            first_mb_in_slice: 0,
            slice_type: 0,
            pic_parameter_set_id: 0,
            frame_num: 0,
            idr_pic_id: 0,
            pic_order_cnt_lsb: 0,
            delta_pic_order_cnt: [0, 0],
            redundant_pic_cnt: 0,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            nal_ref_idc: 1,
            nal_unit_type: 1,
            field_pic_flag: false,
            bottom_field: false,
            long_term_reference: false,
            direct_spatial_mv_pred_flag: false,
            num_ref_idx_active_override_flag: false,
            cabac_init_idc: 0,
            slice_qp_delta: 0,
            disable_deblocking_filter_idc: 0,
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            ref_pic_list_modification_l0: Vec::new(),
            ref_pic_list_modification_l1: Vec::new(),
            dec_ref_pic_marking: mmcos,
            no_output_of_prior_pics_flag: false,
            long_term_reference_flag: false,
            header_bit_size: 0,
            luma_log2_weight_denom: 0,
            chroma_log2_weight_denom: 0,
            luma_weight_l0_flag: 0,
            luma_weight_l0: [0; 32],
            luma_offset_l0: [0; 32],
            chroma_weight_l0_flag: 0,
            chroma_weight_l0: [[0; 2]; 32],
            chroma_offset_l0: [[0; 2]; 32],
            luma_weight_l1_flag: 0,
            luma_weight_l1: [0; 32],
            luma_offset_l1: [0; 32],
            chroma_weight_l1_flag: 0,
            chroma_weight_l1: [[0; 2]; 32],
            chroma_offset_l1: [[0; 2]; 32],
        }
    }

    #[test]
    fn test_new_initialization() {
        let dpb = NvdecDpbManager::new(16);
        assert_eq!(dpb.entries.len(), 0);
        assert_eq!(dpb.next_pic_index, 0);
        assert_eq!(dpb.max_dpb_size, 16);
    }

    #[test]
    fn test_add_frame() {
        let mut dpb = NvdecDpbManager::new(16);
        let idx0 = dpb.add_frame(0, 0, true);
        assert_eq!(idx0, 0);
        let idx1 = dpb.add_frame(1, 2, true);
        assert_eq!(idx1, 1);

        assert!(dpb.entries[0].is_valid);
        assert_eq!(dpb.entries[0].frame_num, 0);
        assert_eq!(dpb.entries[0].pic_order_cnt, 0);
        assert!(dpb.entries[0].is_reference);

        assert!(dpb.entries[1].is_valid);
        assert_eq!(dpb.entries[1].frame_num, 1);
        assert_eq!(dpb.entries[1].pic_order_cnt, 2);
    }

    #[test]
    fn test_reset() {
        let mut dpb = NvdecDpbManager::new(16);
        dpb.add_frame(0, 0, true);
        dpb.add_frame(1, 2, true);
        dpb.reset();
        for e in &dpb.entries {
            assert!(!e.is_valid);
        }
    }

    #[test]
    fn test_apply_mmco_idr() {
        let mut dpb = NvdecDpbManager::new(16);
        dpb.add_frame(0, 0, true);
        dpb.add_frame(1, 2, true);

        let slh = make_slice_header(Vec::new());
        dpb.apply_mmco(2, &slh, true, false);

        for e in &dpb.entries {
            assert!(!e.is_valid);
        }
    }

    #[test]
    fn test_apply_mmco_unmark_short_term() {
        let mut dpb = NvdecDpbManager::new(16);
        dpb.add_frame(5, 10, true);
        dpb.add_frame(10, 20, true);
        dpb.add_frame(15, 30, true);

        // Current frame_num=20, unmark frame with picNumX = 20 - (4+1) = 15 -> matches frame_num=15
        let mmco = vec![DecRefPicMarkingEntry {
            memory_management_control_operation: 1, // unmark_short_term
            value: 4,
        }];
        let slh = make_slice_header(mmco);
        dpb.apply_mmco(20, &slh, false, false);

        assert!(dpb.entries[0].is_valid); // frame_num=5
        assert!(dpb.entries[1].is_valid); // frame_num=10
        assert!(!dpb.entries[2].is_valid); // frame_num=15 unmarked
    }

    #[test]
    fn test_apply_mmco_unmark_all_short_term() {
        let mut dpb = NvdecDpbManager::new(16);
        dpb.add_frame(5, 10, true);
        dpb.add_frame(10, 20, true);

        let mmco = vec![DecRefPicMarkingEntry {
            memory_management_control_operation: 8, // unmark_all_short_term
            value: 0,
        }];
        let slh = make_slice_header(mmco);
        dpb.apply_mmco(15, &slh, false, false);

        assert!(!dpb.entries[0].is_valid);
        assert!(!dpb.entries[1].is_valid);
    }

    #[test]
    fn test_apply_mmco_unmark_long_term() {
        let mut dpb = NvdecDpbManager::new(16);

        // Create a long-term reference with auto-assigned long_term_pic_num=0.
        dpb.apply_idr_reset(true);
        dpb.add_frame(10, 20, true);
        assert!(dpb.entries[0].is_long_term);
        assert_eq!(dpb.entries[0].long_term_frame_idx, Some(0));

        // Op 2: unmark the long-term picture with long_term_pic_num=0.
        let mmco = vec![DecRefPicMarkingEntry {
            memory_management_control_operation: 2,
            value: 0,
        }];
        let slh = make_slice_header(mmco);
        dpb.apply_mmco(15, &slh, false, false);

        assert!(!dpb.entries[0].is_valid);
        assert!(!dpb.entries[0].is_reference); // long-term unmarked
    }

    #[test]
    fn test_apply_mmco_unmark_all_long_term() {
        let mut dpb = NvdecDpbManager::new(16);
        dpb.entries.push(NvdecDpbEntry {
            pic_index: 0,
            frame_num: 0,
            pic_order_cnt: 0,
            is_reference: true,
            is_long_term: true,
            is_valid: true,
            long_term_frame_idx: Some(0),
            seq: 0,
            extracted: false,
        });
        dpb.entries.push(NvdecDpbEntry {
            pic_index: 1,
            frame_num: 1,
            pic_order_cnt: 2,
            is_reference: true,
            is_long_term: true,
            is_valid: true,
            long_term_frame_idx: Some(3),
            seq: 1,
            extracted: false,
        });

        // Op 9: unmark all long-term pictures with long_term_pic_num >= 3.
        let mmco = vec![DecRefPicMarkingEntry {
            memory_management_control_operation: 9,
            value: 3,
        }];
        let slh = make_slice_header(mmco);
        dpb.apply_mmco(5, &slh, false, false);

        assert!(dpb.entries[0].is_valid);   // lt_idx=0 < 3, kept
        assert!(!dpb.entries[1].is_valid); // lt_idx=3 >= 3, unmarked
    }

    #[test]
    fn test_to_cuvid_dpb_entries() {
        let mut dpb = NvdecDpbManager::new(16);
        dpb.add_frame(0, 0, true);
        dpb.add_frame(1, 2, true);

        let cuvid = dpb.to_cuvid_dpb_entries();

        assert_eq!(cuvid[0].not_existing, 0);
        assert_eq!(cuvid[0].PicIdx, 0);
        assert_eq!(cuvid[0].FrameIdx, 0);
        assert_eq!(cuvid[0].used_for_reference, 3);
        assert_eq!(cuvid[0].FieldOrderCnt, [0, 0]);

        assert_eq!(cuvid[1].not_existing, 0);
        assert_eq!(cuvid[1].PicIdx, 1);
        assert_eq!(cuvid[1].FrameIdx, 1);
        assert_eq!(cuvid[1].FieldOrderCnt, [2, 2]);

        // Entry 2 is empty (emptiness signaled by PicIdx=-1, not_existing=0)
        assert_eq!(cuvid[2].not_existing, 0);
        assert_eq!(cuvid[2].PicIdx, -1);
        assert_eq!(cuvid[2].used_for_reference, 0);
    }

    #[test]
    fn test_to_cuvid_dpb_entries_non_reference_excluded() {
        let mut dpb = NvdecDpbManager::new(16);
        dpb.add_frame(0, 0, true);   // reference
        dpb.add_frame(1, 2, false);  // non-reference
        dpb.add_frame(2, 4, true);   // reference

        let cuvid = dpb.to_cuvid_dpb_entries();

        // Only reference frames should appear as valid entries
        assert_eq!(cuvid[0].not_existing, 0);
        assert_eq!(cuvid[0].PicIdx, 0);
        assert_eq!(cuvid[0].used_for_reference, 3);

        assert_eq!(cuvid[1].not_existing, 0);
        assert_eq!(cuvid[1].PicIdx, 2);
        assert_eq!(cuvid[1].used_for_reference, 3);

        // Non-reference frame and remaining slots are empty (PicIdx=-1)
        assert_eq!(cuvid[2].not_existing, 0);
        assert_eq!(cuvid[2].PicIdx, -1);
        assert_eq!(cuvid[2].used_for_reference, 0);
    }

    #[test]
    fn test_get_next_pic_index() {
        let mut dpb = NvdecDpbManager::new(16);
        assert_eq!(dpb.get_next_pic_index(), 0);
        dpb.add_frame(0, 0, true);
        assert_eq!(dpb.get_next_pic_index(), 1);
    }

    #[test]
    fn test_compute_pic_num_wraparound() {
        let mut dpb = NvdecDpbManager::new(16);
        dpb.set_max_frame_num(64);

        // current=5, diff=8: picNumX = 64 + 5 - 9 = 60
        let pic_num = dpb.compute_pic_num(5, 8);
        assert_eq!(pic_num, 60);

        // current=20, diff=5: picNumX = 20 - 6 = 14 (no wrap)
        let pic_num = dpb.compute_pic_num(20, 5);
        assert_eq!(pic_num, 14);
    }

    #[test]
    fn test_sliding_window_eviction() {
        // DPB size limited to 3 reference frames
        let mut dpb = NvdecDpbManager::new(3);

        dpb.add_frame(0, 0, true);   // pic_index=0, poc=0
        dpb.add_frame(1, 2, true);   // pic_index=1, poc=2
        dpb.add_frame(2, 4, true);   // pic_index=2, poc=4

        assert_eq!(dpb.count_references(), 3);

        // Adding 4th reference should evict oldest (poc=0)
        dpb.add_frame(3, 6, true);   // pic_index=3, poc=6

        assert_eq!(dpb.count_references(), 3);

        // Verify oldest was evicted
        assert!(!dpb.entries[0].is_valid);  // poc=0 evicted
        assert!(dpb.entries[1].is_valid);   // poc=2
        assert!(dpb.entries[2].is_valid);   // poc=4
        assert!(dpb.entries[3].is_valid);   // poc=6

        // Verify CUVID output only has 3 valid entries
        let cuvid = dpb.to_cuvid_dpb_entries();
        assert_eq!(cuvid[0].not_existing, 0);
        assert_eq!(cuvid[1].not_existing, 0);
        assert_eq!(cuvid[2].not_existing, 0);
        assert_eq!(cuvid[3].not_existing, 0);
        assert_eq!(cuvid[3].PicIdx, -1);
    }

    #[test]
    fn test_sliding_window_preserves_long_term() {
        // DPB size limited to 2 reference frames
        let mut dpb = NvdecDpbManager::new(2);

        // Add frame 0 as a long-term reference (auto long_term_pic_num=0)
        dpb.apply_idr_reset(true);
        dpb.add_frame(0, 0, true); // pic_index=0, poc=0, long-term

        dpb.add_frame(1, 2, true); // pic_index=1, poc=2, short-term

        assert!(dpb.entries[0].is_long_term);

        // Adding 3rd reference should evict oldest SHORT-TERM (poc=2), not long-term
        dpb.add_frame(2, 4, true);   // pic_index=2, poc=4

        assert!(dpb.entries[0].is_valid);   // long-term preserved
        assert!(!dpb.entries[1].is_valid);  // short-term evicted
        assert!(dpb.entries[2].is_valid);   // new frame
    }


    #[test]
    fn test_get_entry_by_pic_index() {
        let mut dpb = NvdecDpbManager::new(16);
        dpb.add_frame(5, 10, true);
        dpb.add_frame(10, 20, true);

        let entry = dpb.get_entry_by_pic_index(0);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().frame_num, 5);

        let entry = dpb.get_entry_by_pic_index(1);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().frame_num, 10);

        let entry = dpb.get_entry_by_pic_index(99);
        assert!(entry.is_none());
    }

    #[test]
    fn test_dpb_no_ring_buffer_wraparound() {
        // With pic_index wrapping at max_decode_surfaces (32),
        // pic_indices cycle through 0..31.
        // The DPB still stores all entries, but pic_indices wrap.
        let mut dpb = NvdecDpbManager::new(16);
        dpb.set_max_decode_surfaces(8); // Use small value for testing

        // Add 12 frames with max_decode_surfaces=8
        // pic_indices: 0,1,2,3,4,5,6,7,0,1,2,3
        for i in 0u32..12 {
            dpb.add_frame(i, i as i32 * 2, false); // non-reference to avoid eviction
        }

        // get_next_pic_index should wrap
        assert_eq!(dpb.get_next_pic_index(), 4); // 12 % 8 = 4

        // All entries are stored (non-reference frames aren't evicted)
        assert_eq!(dpb.entries.len(), 12);

        // pic_indices wrap correctly
        assert_eq!(dpb.entries[0].pic_index, 0);
        assert_eq!(dpb.entries[7].pic_index, 7);
        assert_eq!(dpb.entries[8].pic_index, 0); // wrapped
        assert_eq!(dpb.entries[11].pic_index, 3); // wrapped
    }

    #[test]
    fn test_set_max_dpb_size() {
        let mut dpb = NvdecDpbManager::new(4);
        dpb.set_max_dpb_size(8);
        assert_eq!(dpb.max_dpb_size, 8);
    }

    #[test]
    fn test_idr_long_term_reference_flag() {
        // IDR with long_term_reference_flag=1 should mark current picture as long-term
        let mut dpb = NvdecDpbManager::new(16);
        dpb.add_frame(0, 0, true);
        dpb.add_frame(1, 2, true);

        // IDR with long_term_reference_flag=true
        let slh = make_slice_header(Vec::new());
        dpb.apply_mmco(2, &slh, true, true);

        // DPB should be reset (all entries invalid)
        assert!(!dpb.entries[0].is_valid);
        assert!(!dpb.entries[1].is_valid);

        // Current frame should be marked as long-term when added
        dpb.add_frame(2, 4, true);
        assert!(dpb.entries[2].is_valid);
        assert!(dpb.entries[2].is_long_term);
    }

    #[test]
    fn test_idr_without_long_term_reference_flag() {
        // IDR with long_term_reference_flag=0 should mark current picture as short-term
        let mut dpb = NvdecDpbManager::new(16);
        dpb.add_frame(0, 0, true);

        // IDR with long_term_reference_flag=false
        let slh = make_slice_header(Vec::new());
        dpb.apply_mmco(1, &slh, true, false);

        // DPB should be reset
        assert!(!dpb.entries[0].is_valid);

        // Current frame should be short-term
        dpb.add_frame(1, 2, true);
        assert!(dpb.entries[1].is_valid);
        assert!(!dpb.entries[1].is_long_term);
    }

    #[test]
    fn test_dpb_entries_after_idr_all_empty() {
        // After IDR reset, all CUVID DPB entries should be empty
        // (PicIdx=-1, not_existing=0, used_for_reference=0)
        let mut dpb = NvdecDpbManager::new(16);
        dpb.add_frame(0, 0, true);
        dpb.add_frame(1, 2, true);

        let slh = make_slice_header(Vec::new());
        dpb.apply_mmco(2, &slh, true, false);

        let cuvid = dpb.to_cuvid_dpb_entries();
        for i in 0..16 {
            assert_eq!(cuvid[i].PicIdx, -1, "Entry {} should be empty after IDR reset", i);
            assert_eq!(cuvid[i].not_existing, 0, "Entry {} should have not_existing=0 after IDR reset", i);
            assert_eq!(cuvid[i].used_for_reference, 0, "Entry {} should have used_for_reference=0 after IDR reset", i);
        }
    }

    #[test]
    fn test_non_reference_frame_not_in_dpb() {
        // Non-reference frames should not appear in CUVID DPB entries
        let mut dpb = NvdecDpbManager::new(16);
        dpb.add_frame(0, 0, true);   // reference
        dpb.add_frame(1, 2, false);  // non-reference
        dpb.add_frame(2, 4, true);   // reference

        let cuvid = dpb.to_cuvid_dpb_entries();

        // Only 2 reference frames should be in CUVID DPB
        let mut ref_count = 0;
        for i in 0..16 {
            if cuvid[i].PicIdx != -1 {
                assert_eq!(cuvid[i].used_for_reference, 3);
                ref_count += 1;
            }
        }
        assert_eq!(ref_count, 2);
    }

    #[test]
    fn test_frame_num_wraparound_mmco_unmark() {
        // Verify MMCO Op 1 (unmark short-term) works correctly across frame_num wraparound
        let mut dpb = NvdecDpbManager::new(16);
        dpb.set_max_frame_num(64);

        // Add frames near wraparound boundary
        dpb.add_frame(60, 120, true);  // frame_num=60
        dpb.add_frame(62, 124, true);  // frame_num=62

        // Current frame_num=3 (wrapped around), unmark picNumX = 64 + 3 - (2+1) = 64
        // Since frame_num wraps at 64, picNumX=64 is out of range, so let's use
        // difference_of_pic_nums_minus1=2: picNumX = 64 + 3 - 3 = 64... that's invalid
        // Let's use difference_of_pic_nums_minus1=1: picNumX = 64 + 3 - 2 = 65... also invalid
        // Better: difference_of_pic_nums_minus1=3: picNumX = 64 + 3 - 4 = 63
        // That should not match any frame.
        // Let's use difference_of_pic_nums_minus1=59: picNumX = 64 + 3 - 60 = 7... no match
        // difference_of_pic_nums_minus1=3: picNumX = 64 + 3 - 4 = 63
        // difference_of_pic_nums_minus1=1: picNumX = 64 + 3 - 2 = 65... no
        // Current=3, diff=5 (minus1=4): picNumX = 64 + 3 - 5 = 62 -> matches frame_num=62!
        let mmco = vec![DecRefPicMarkingEntry {
            memory_management_control_operation: 1, // unmark_short_term
            value: 4, // difference_of_pic_nums_minus1=4, diff=5
        }];
        let slh = make_slice_header(mmco);
        dpb.apply_mmco(3, &slh, false, false);

        assert!(dpb.entries[0].is_valid);  // frame_num=60 still valid
        assert!(!dpb.entries[1].is_valid); // frame_num=62 unmarked
    }

    #[test]
    fn test_mmco_op9_unmark_all_long_term() {
        // Op 9: unmark all long-term pictures with long_term_pic_num >= value
        let mut dpb = NvdecDpbManager::new(16);
        dpb.entries.push(NvdecDpbEntry {
            pic_index: 0,
            frame_num: 0,
            pic_order_cnt: 0,
            is_reference: true,
            is_long_term: true,
            is_valid: true,
            long_term_frame_idx: Some(4),
            seq: 0,
            extracted: false,
        });
        dpb.entries.push(NvdecDpbEntry {
            pic_index: 1,
            frame_num: 1,
            pic_order_cnt: 2,
            is_reference: true,
            is_long_term: true,
            is_valid: true,
            long_term_frame_idx: Some(5),
            seq: 1,
            extracted: false,
        });

        let mmco = vec![DecRefPicMarkingEntry {
            memory_management_control_operation: 9,
            value: 5, // long_term_pic_num = 5
        }];
        let slh = make_slice_header(mmco);
        dpb.apply_mmco(0, &slh, false, false);

        assert!(dpb.entries[0].is_valid);   // lt_idx=4 < 5, kept
        assert!(!dpb.entries[1].is_valid); // lt_idx=5 >= 5, unmarked
    }

    #[test]
    fn test_mmco_op3_set_num_long_term() {
        // Op 3: set_num_long_term — invalidate all long-term pictures with
        // long_term_pic_num >= max_long_term_frame_idx_plus1.
        let mut dpb = NvdecDpbManager::new(16);

        // Create long-term references with long_term_pic_num 0 and 1
        dpb.entries.push(NvdecDpbEntry {
            pic_index: 0,
            frame_num: 0,
            pic_order_cnt: 0,
            is_reference: true,
            is_long_term: true,
            is_valid: true,
            long_term_frame_idx: Some(0),
            seq: 0,
            extracted: false,
        });
        dpb.entries.push(NvdecDpbEntry {
            pic_index: 1,
            frame_num: 1,
            pic_order_cnt: 2,
            is_reference: true,
            is_long_term: true,
            is_valid: true,
            long_term_frame_idx: Some(1),
            seq: 1,
            extracted: false,
        });

        // Op 3 with max_long_term_frame_idx_plus1=1: invalidate lt idx >= 1
        let mmco = vec![DecRefPicMarkingEntry {
            memory_management_control_operation: 3,
            value: 1,
        }];
        let slh = make_slice_header(mmco);
        dpb.apply_mmco(2, &slh, false, false);

        assert!(dpb.entries[0].is_valid);   // lt_idx=0 < 1, kept
        assert!(!dpb.entries[1].is_valid); // lt_idx=1 >= 1, invalidated
    }

    #[test]
    fn test_mmco_op7_mark_future_long_term() {
        // Op 7: mark the long-term picture with long_term_pic_num as a reference
        let mut dpb = NvdecDpbManager::new(16);

        // Long-term picture with long_term_pic_num=3 (already a reference)
        dpb.entries.push(NvdecDpbEntry {
            pic_index: 0,
            frame_num: 0,
            pic_order_cnt: 0,
            is_reference: true,
            is_long_term: true,
            is_valid: true,
            long_term_frame_idx: Some(3),
            seq: 0,
            extracted: false,
        });

        // Long-term picture with long_term_pic_num=7 (not yet a reference)
        dpb.entries.push(NvdecDpbEntry {
            pic_index: 1,
            frame_num: 1,
            pic_order_cnt: 2,
            is_reference: false,
            is_long_term: true,
            is_valid: true,
            long_term_frame_idx: Some(7),
            seq: 1,
            extracted: false,
        });

        // Mark long_term_pic_num=7 as a reference
        let mmco = vec![DecRefPicMarkingEntry {
            memory_management_control_operation: 7,
            value: 7,
        }];
        let slh = make_slice_header(mmco);
        dpb.apply_mmco(2, &slh, false, false);

        assert!(dpb.entries[0].is_valid);
        assert!(dpb.entries[0].is_reference); // long_term_pic_num=3 untouched
        assert!(dpb.entries[1].is_valid);
        assert!(dpb.entries[1].is_reference); // long_term_pic_num=7 now a reference
    }

    #[test]
    fn test_mmco_op3_invalidate_above_max_idx() {
        // Op 3: invalidate all long-term with long_term_pic_num >= value
        let mut dpb = NvdecDpbManager::new(16);

        // Add frames with long_term_pic_num 0, 3, 7
        for (i, lt_idx) in [0u32, 3, 7].into_iter().enumerate() {
            dpb.entries.push(NvdecDpbEntry {
                pic_index: i as i32,
                frame_num: i as u32,
                pic_order_cnt: (i * 2) as i32,
                is_reference: true,
                is_long_term: true,
                is_valid: true,
                long_term_frame_idx: Some(lt_idx),
                seq: i as i32,
                extracted: false,
            });
        }

        // Set max_long_term_frame_idx_plus1 = 4 (invalidate idx >= 4)
        let mmco = vec![DecRefPicMarkingEntry {
            memory_management_control_operation: 3,
            value: 4,
        }];
        let slh = make_slice_header(mmco);
        dpb.apply_mmco(3, &slh, false, false);

        assert!(dpb.entries[0].is_valid);  // lt_idx=0 < 4, kept
        assert!(dpb.entries[1].is_valid);  // lt_idx=3 < 4, kept
        assert!(!dpb.entries[2].is_valid); // lt_idx=7 >= 4, invalidated
    }

    #[test]
    fn test_compute_pic_num_huge_value_no_panic() {
        let mut dpb = NvdecDpbManager::new(16);
        dpb.set_max_frame_num(64);

        // Garbage huge values from an upstream mis-parse must never panic
        // and must stay within [0, max_frame_num).
        let pic_num = dpb.compute_pic_num(5, 0x100_0000);
        assert!(pic_num < 64);

        // u32::MAX wraps diff to 0: picNumX = (5 - 0) mod 64 = 5.
        let pic_num = dpb.compute_pic_num(5, u32::MAX);
        assert_eq!(pic_num, 5);

        // max_frame_num == 0 guard: no division-by-zero, returns 0.
        let mut dpb0 = NvdecDpbManager::new(16);
        dpb0.set_max_frame_num(0);
        assert_eq!(dpb0.compute_pic_num(5, 10), 0);
    }

    #[test]
    fn test_mmco_huge_value_is_noop() {
        let mut dpb = NvdecDpbManager::new(16);
        dpb.add_frame(5, 10, true);
        dpb.add_frame(10, 20, true);

        // Op 1 (unmark_short_term) with a garbage huge
        // difference_of_pic_nums_minus1 must be a graceful no-op:
        // no panic, no eviction.
        let mmco = vec![DecRefPicMarkingEntry {
            memory_management_control_operation: 1,
            value: 1 << 30,
        }];
        let slh = make_slice_header(mmco);
        dpb.apply_mmco(10, &slh, false, false);

        assert!(dpb.entries[0].is_valid);
        assert!(dpb.entries[0].is_reference);
        assert!(dpb.entries[1].is_valid);
        assert!(dpb.entries[1].is_reference);
    }
}

//! Common AV1 DPB state machine + POC (display-order) calculator.
//!
//! Used by all decode backends (Vulkan, NVDEC, VAAPI) so that frame-buffer
//! tracking, slot allocation, reference-list resolution and display-order
//! (POC) assignment have a single implementation.
//!
//! Semantics (verified against the C++ reference `VulkanAV1Decoder.cpp`,
//! the rav1e bitstream writer, and pixel-perfect decode results):
//! - 8 frame buffers, indices 0..7 (names INTRA..ALTREF). Bitstream
//!   `ref_frame_idx` values and `refresh_frame_flags` bits use this indexing:
//!   refresh bit `i` points frame buffer `i` at the current picture after
//!   decode; a reference-name entry `ref_frame_idx[n]` holds the frame
//!   buffer index of the picture referenced by name `LAST + n`.
//! - `refresh_frame_flags` is an 8-bit mask in the bitstream for all frames
//!   except shown KEY and SWITCH frames (which carry no field and implicitly
//!   refresh all 8 buffers).
//! - AV1 has no explicit POC in the bitstream; the display position is
//!   derived from `show_frame` / `show_existing_frame`. The POC of a frame
//!   is its display index (`next_poc`), assigned when the frame becomes
//!   displayable.

use std::collections::VecDeque;

use crate::av1::Av1FrameHeader;

/// Number of AV1 frame buffers (INTRA..ALTREF).
pub const AV1_NUM_FRAME_BUFFERS: usize = 8;
/// Number of reference names in `ref_frame_idx` (LAST..ALTREF).
pub const AV1_NUM_REF_NAMES: usize = 7;

/// Default global motion models: all identity {type=0, wmmat=[0,0,65536,0,0,65536]}.
pub fn default_global_models() -> [(u8, [i32; 6]); AV1_NUM_REF_NAMES] {
    let identity = (0u8, [0i32, 0, 65536, 0, 0, 65536]);
    [identity; AV1_NUM_REF_NAMES]
}

/// State of one AV1 frame buffer (what the C++ reference keeps in
/// `m_pBuffers[i]` plus the parser's per-buffer inheritance state).
#[derive(Debug, Clone)]
pub struct Av1FrameBuffer {
    /// DPB slot holding this buffer's picture; `-1` = empty.
    pub slot: i32,
    /// Order hint of the picture stored here.
    pub order_hint: u32,
    /// Coded dimensions of the picture stored here ((0, 0) if empty).
    pub width: u32,
    pub height: u32,
    /// `SavedOrderHints[ref_name]`: the refreshing frame's OrderHints for
    /// each reference name (index = reference name 0..7; 0=INTRA unused).
    pub saved_order_hints: [u8; AV1_NUM_FRAME_BUFFERS],
    /// Raw signed distance `GetRelativeDist1(refreshing_OH, saved[name])`
    /// per reference name; the Vulkan `RefFrameSignBias` bit `name` is set
    /// when this is <= 0. Index 0 (INTRA) is never set.
    pub ref_frame_sign_bias: [i8; AV1_NUM_FRAME_BUFFERS],
    /// Frame type of the picture stored here.
    pub frame_type: u8,
    pub disable_frame_end_update_cdf: bool,
    pub segmentation_enabled: bool,
    /// Per-buffer global motion models (type, wmmat[6]) for the 7 reference
    /// names — inherited by frames whose primary reference is this buffer.
    pub global_motion: [(u8, [i32; 6]); AV1_NUM_REF_NAMES],
    /// Per-buffer segmentation feature state (inherited when
    /// `segmentation_update_data == 0`).
    pub segment_feature_enabled: [u8; 8],
    pub segment_feature_data: [[i16; 8]; 8],
    /// Per-buffer loop-filter deltas (inherited from the primary reference).
    pub loop_filter_ref_deltas: [i8; 8],
    pub loop_filter_mode_deltas: [i8; 2],
}

impl Default for Av1FrameBuffer {
    fn default() -> Self {
        Self {
            slot: -1,
            order_hint: 0,
            width: 0,
            height: 0,
            saved_order_hints: [0; AV1_NUM_FRAME_BUFFERS],
            ref_frame_sign_bias: [0; AV1_NUM_FRAME_BUFFERS],
            frame_type: 0,
            disable_frame_end_update_cdf: false,
            segmentation_enabled: false,
            global_motion: default_global_models(),
            segment_feature_enabled: [0; 8],
            segment_feature_data: [[0; 8]; 8],
            loop_filter_ref_deltas: [0; 8],
            loop_filter_mode_deltas: [0; 2],
        }
    }
}

/// Common AV1 DPB: 8 frame buffers, backend-agnostic slot bookkeeping
/// (FIFO allocation that never clobbers a live reference) and the POC
/// (display index) counter.
#[derive(Debug)]
pub struct Av1Dpb {
    num_slots: u32,
    pub frame_buffers: [Av1FrameBuffer; AV1_NUM_FRAME_BUFFERS],
    /// FIFO of slots not held by any frame buffer.
    available_slots: VecDeque<u32>,
    /// Number of real frames decoded (decode-order counter).
    decoded_frames: u32,
    /// Display-order counter: the POC of the next displayed frame.
    display_count: u32,
}

impl Av1Dpb {
    /// Create a DPB with `num_slots` backend slots (DPB images / surfaces).
    pub fn new(num_slots: u32) -> Self {
        let mut dpb = Self {
            num_slots,
            frame_buffers: Default::default(),
            available_slots: VecDeque::new(),
            decoded_frames: 0,
            display_count: 0,
        };
        dpb.reset();
        dpb
    }

    /// Reset all state (new stream / SPS change / reconfigure).
    pub fn reset(&mut self) {
        self.frame_buffers = Default::default();
        self.available_slots.clear();
        for s in 0..self.num_slots {
            self.available_slots.push_back(s);
        }
        self.decoded_frames = 0;
        self.display_count = 0;
    }

    /// Number of backend slots.
    pub fn num_slots(&self) -> u32 {
        self.num_slots
    }

    /// Key frame: clear all frame buffers and the slot FIFO (the key frame
    /// will take slot 0). Decode/display counters are preserved.
    pub fn reset_for_keyframe(&mut self) {
        self.frame_buffers = Default::default();
        self.available_slots.clear();
        for s in 0..self.num_slots {
            self.available_slots.push_back(s);
        }
    }

    // ------------------------------------------------------------------
    // Order-hint arithmetic (AV1 spec 7.10 GetRelativeDist1)
    // ------------------------------------------------------------------

    /// AV1 spec 7.10: `GetRelativeDist1(a, b)`.
    pub fn get_relative_dist1(a: i32, b: i32, ohb: u32) -> i32 {
        let bits = ohb + 1;
        let diff = a - b;
        let m = 1 << (bits - 1);
        (diff & (m - 1)) - (diff & m)
    }

    // ------------------------------------------------------------------
    // Reference-list resolution
    // ------------------------------------------------------------------

    /// AV1 spec 7.4.1 reference-list derivation (short signaling).
    ///
    /// Given the LAST/GOLDEN frame-buffer indices from the bitstream and the
    /// current frame's order hint, derive the full 7-entry `ref_frame_idx`
    /// (values = frame buffer indices), using this DPB's per-buffer order
    /// hints (pre-decode state).
    pub fn set_frame_refs(
        &self,
        last_frame_idx: i32,
        golden_frame_idx: i32,
        order_hint: u32,
        ohb: u32,
    ) -> [i32; AV1_NUM_REF_NAMES] {
        let cur_frame_hint = 1_i32 << ohb;

        let mut ref_idx: [i32; AV1_NUM_REF_NAMES] = [-1; AV1_NUM_REF_NAMES];
        let mut used = [false; AV1_NUM_FRAME_BUFFERS];

        ref_idx[0] = last_frame_idx; // LAST
        ref_idx[3] = golden_frame_idx; // GOLDEN
        let li = last_frame_idx as usize;
        if li < AV1_NUM_FRAME_BUFFERS {
            used[li] = true;
        }
        let gi = golden_frame_idx as usize;
        if gi < AV1_NUM_FRAME_BUFFERS {
            used[gi] = true;
        }

        // shiftedOrderHints[i] = curFrameHint + GetRelativeDist1(RefOrderHint[i], OrderHint)
        let shifted: [i32; AV1_NUM_FRAME_BUFFERS] = std::array::from_fn(|i| {
            cur_frame_hint
                + Self::get_relative_dist1(
                    self.frame_buffers[i].order_hint as i32,
                    order_hint as i32,
                    ohb,
                )
        });

        // ALTREF_FRAME (idx 6): unused, hint>=cur, MAX hint
        pick(&mut ref_idx, &mut used, &shifted, cur_frame_hint, 6, true);
        // BWDREF_FRAME (idx 4): unused, hint>=cur, MIN hint
        pick(&mut ref_idx, &mut used, &shifted, cur_frame_hint, 4, false);
        // ALTREF2_FRAME (idx 5): unused, hint>=cur, MIN hint
        pick(&mut ref_idx, &mut used, &shifted, cur_frame_hint, 5, false);
        // Ref_Frame_List = [LAST2(1), LAST3(2), BWDREF(4), ALTREF2(5), ALTREF(6)]:
        // unused, hint<cur, MAX hint
        for name in [1i32, 2, 4, 5, 6] {
            if ref_idx[name as usize] < 0 {
                pick_below(&mut ref_idx, &mut used, &shifted, cur_frame_hint, name as usize);
            }
        }
        // Final: fill remaining with argmin over ALL i of shifted
        let mut fill = 0i32;
        let mut fill_hint = i32::MAX;
        for i in 0..AV1_NUM_FRAME_BUFFERS {
            if shifted[i] < fill_hint {
                fill = i as i32;
                fill_hint = shifted[i];
            }
        }
        for i in 0..AV1_NUM_REF_NAMES {
            if ref_idx[i] < 0 {
                ref_idx[i] = fill;
            }
        }
        ref_idx
    }

    /// DPB slot for each of the 7 reference names (pre-decode state).
    /// `-1` when the referenced frame buffer is empty.
    pub fn reference_name_slots(&self, ref_frame_idx: &[u8; AV1_NUM_REF_NAMES]) -> [i32; AV1_NUM_REF_NAMES] {
        let mut result = [-1i32; AV1_NUM_REF_NAMES];
        for i in 0..AV1_NUM_REF_NAMES {
            let fb = ref_frame_idx[i] as usize;
            if fb < AV1_NUM_FRAME_BUFFERS {
                result[i] = self.frame_buffers[fb].slot;
            }
        }
        result
    }

    // ------------------------------------------------------------------
    // Slot allocation / frame-buffer queries
    // ------------------------------------------------------------------

    /// Allocate the output DPB slot using FIFO semantics. A slot is available
    /// iff it is not currently held by any frame buffer, so the output slot
    /// can never clobber a reference needed by the current or any future
    /// frame (C++ `AllocateSlot`/`FreeSlot` / `QueuePictureForDecode`).
    pub fn allocate_output_slot(&mut self) -> u32 {
        let n = self.num_slots as usize;
        let mut in_use = vec![false; n];
        for fb in &self.frame_buffers {
            if fb.slot >= 0 && (fb.slot as usize) < n {
                in_use[fb.slot as usize] = true;
            }
        }
        self.available_slots
            .retain(|&s| (s as usize) < n && !in_use[s as usize]);
        for s in 0..n {
            if !in_use[s] && !self.available_slots.contains(&(s as u32)) {
                self.available_slots.push_back(s as u32);
            }
        }
        self.available_slots.pop_front().unwrap_or(0)
    }

    /// DPB slot for a frame buffer, `-1` if the buffer is empty.
    pub fn slot_of_frame_buffer(&self, fb: usize) -> i32 {
        self.frame_buffers.get(fb).map(|b| b.slot).unwrap_or(-1)
    }

    /// Order hints of all 8 frame buffers (for [`set_frame_refs`]).
    pub fn frame_buffer_order_hints(&self) -> [u32; AV1_NUM_FRAME_BUFFERS] {
        std::array::from_fn(|i| self.frame_buffers[i].order_hint)
    }

    /// Coded (width, height) of a frame buffer, or (0, 0) if empty.
    pub fn frame_buffer_dims(&self, fb: usize) -> (u32, u32) {
        self.frame_buffers
            .get(fb)
            .map(|b| (b.width, b.height))
            .unwrap_or((0, 0))
    }

    /// Frame buffer index stored in DPB slot, if any.
    pub fn frame_buffer_for_slot(&self, slot: i32) -> Option<usize> {
        self.frame_buffers
            .iter()
            .position(|b| b.slot == slot)
    }

    // ------------------------------------------------------------------
    // Commit (post-decode / parse-time state updates)
    // ------------------------------------------------------------------

    /// Apply this frame's refresh mask after its decode into `slot`.
    ///
    /// For each bit `i` set in `fh.refresh_frame_flags`, frame buffer `i` now
    /// holds the picture at `slot` (with its order hint, coded size and the
    /// per-buffer reference info needed by later frames). Uses the PRE-decode
    /// buffer order hints for the SavedOrderHints/RefFrameSignBias snapshot.
    ///
    /// Note: the parser's per-buffer *content* state (global motion models,
    /// segmentation, loop-filter deltas) is updated separately at parse time
    /// via [`Av1Parser::parse_frame_header`] — both are keyed on the same
    /// `refresh_frame_flags`, so the split is safe.
    pub fn commit_decoded(&mut self, slot: u32, fh: &Av1FrameHeader, ohb: u32) {
        // Snapshot pre-decode state (a refreshed buffer may itself be a
        // reference of the current frame).
        let prev_hints: [u32; AV1_NUM_FRAME_BUFFERS] = self.frame_buffer_order_hints();
        for i in 0..AV1_NUM_FRAME_BUFFERS {
            if fh.refresh_frame_flags >> i & 1 != 0 {
                let fb = &mut self.frame_buffers[i];
                fb.slot = slot as i32;
                fb.order_hint = fh.order_hint;
                fb.width = fh.frame_width;
                fb.height = fh.frame_height;
                fb.frame_type = fh.frame_type;
                fb.disable_frame_end_update_cdf = fh.disable_cdf_update;
                fb.segmentation_enabled = fh.segmentation_enabled;
                // SavedOrderHints[refName] / RefFrameSignBias (C++
                // VulkanAV1Decoder.cpp:390-394): the refreshing frame's
                // OrderHints for each reference name it references.
                for name in 1..AV1_NUM_FRAME_BUFFERS {
                    let ref_fb = fh.ref_frame_idx[name - 1] as usize;
                    let oh = prev_hints.get(ref_fb).copied().unwrap_or(0);
                    fb.saved_order_hints[name] = oh as u8;
                    fb.ref_frame_sign_bias[name] =
                        Self::get_relative_dist1(fh.order_hint as i32, oh as i32, ohb) as i8;
                }
            }
        }
        self.decoded_frames += 1;
    }

    /// Update the per-buffer *content* state (global motion models,
    /// segmentation feature data, loop-filter deltas) for the buffers
    /// refreshed by this frame. Called by the parser after parsing a frame
    /// header so that subsequent frames inherit from the correct buffer.
    pub fn update_content(&mut self, fh: &Av1FrameHeader) {
        for i in 0..AV1_NUM_FRAME_BUFFERS {
            if fh.refresh_frame_flags >> i & 1 != 0 {
                let fb = &mut self.frame_buffers[i];
                fb.global_motion = std::array::from_fn(|j| {
                    (fh.global_motion_type[j], fh.global_motion_params[j])
                });
                fb.segment_feature_enabled = fh.segment_feature_enabled;
                fb.segment_feature_data = fh.segment_feature_data;
                fb.loop_filter_ref_deltas = fh.loop_filter_ref_deltas;
                fb.loop_filter_mode_deltas = fh.loop_filter_mode_deltas;
            }
        }
    }

    /// The `RefFrameSignBias` bitmask for a frame buffer (bit `name` set when
    /// the raw signed distance to that reference name's saved order hint is
    /// <= 0; C++ VulkanAV1Decoder.cpp:331-333).
    pub fn ref_frame_sign_bias_mask(&self, fb: usize) -> u8 {
        let b = match self.frame_buffers.get(fb) {
            Some(b) => b,
            None => return 0,
        };
        let mut mask = 0u8;
        for name in 1..AV1_NUM_FRAME_BUFFERS {
            if b.ref_frame_sign_bias[name] <= 0 {
                mask |= 1 << name;
            }
        }
        mask
    }

    // ------------------------------------------------------------------
    // POC (display order)
    // ------------------------------------------------------------------

    /// POC of the next displayed frame (= its display index).
    pub fn next_poc(&self) -> u32 {
        self.display_count
    }

    /// Note that a frame has been displayed (advances the POC counter).
    pub fn note_displayed(&mut self) {
        self.display_count += 1;
    }

    /// Number of real frames decoded so far.
    pub fn decoded_frames(&self) -> u32 {
        self.decoded_frames
    }
}

/// ALTREF(6)/BWDREF(4)/ALTREF2(5) selection: among unused buffers with
/// shifted hint >= cur, pick MAX (max_hint=true) or MIN hint.
fn pick(
    ref_idx: &mut [i32; AV1_NUM_REF_NAMES],
    used: &mut [bool; AV1_NUM_FRAME_BUFFERS],
    shifted: &[i32; AV1_NUM_FRAME_BUFFERS],
    cur_frame_hint: i32,
    name: usize,
    max_hint: bool,
) {
    let mut best = -1i32;
    let mut best_hint = if max_hint { i32::MIN } else { i32::MAX };
    for i in 0..AV1_NUM_FRAME_BUFFERS {
        if !used[i] && shifted[i] >= cur_frame_hint {
            if best < 0 || (max_hint && shifted[i] >= best_hint) || (!max_hint && shifted[i] < best_hint) {
                best = i as i32;
                best_hint = shifted[i];
            }
        }
    }
    if best >= 0 {
        ref_idx[name] = best;
        used[best as usize] = true;
    }
}

/// LAST2(1)/LAST3(2) (and fallback BWDREF/ALTREF2/ALTREF): among unused
/// buffers with shifted hint < cur, pick MAX hint.
fn pick_below(
    ref_idx: &mut [i32; AV1_NUM_REF_NAMES],
    used: &mut [bool; AV1_NUM_FRAME_BUFFERS],
    shifted: &[i32; AV1_NUM_FRAME_BUFFERS],
    cur_frame_hint: i32,
    name: usize,
) {
    let mut best = -1i32;
    let mut best_hint = i32::MIN;
    for i in 0..AV1_NUM_FRAME_BUFFERS {
        if !used[i] && shifted[i] < cur_frame_hint && shifted[i] >= best_hint {
            best = i as i32;
            best_hint = shifted[i];
        }
    }
    if best >= 0 {
        ref_idx[name] = best;
        used[best as usize] = true;
    }
}

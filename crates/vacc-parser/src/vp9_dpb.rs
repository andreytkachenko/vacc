//! Common VP9 DPB manager shared by the Vulkan, NVDEC, and VAAPI backends.
//!
//! VP9 decode state (spec 7.2):
//!
//! - There are **8 frame buffers** (indices 0-7). The names LAST / GOLDEN /
//!   ALTREF do not denote fixed buffers: each inter frame picks exactly three
//!   of the eight via `ref_frame_idx[0..3]` (LAST, GOLDEN, ALTREF respectively).
//! - A frame's `refresh_frame_flags` (8 bits) says which frame buffers the
//!   frame's picture is stored into. After the frame is decoded, every set bit
//!   `i` means "frame buffer `i` now points at this frame". Key frames use
//!   `0xFF` (all eight).
//! - Display order equals decode order for frames with `show_frame == 1`;
//!   `show_existing_frame` commands re-display a frame buffer without decoding.
//!
//! ## Slot model
//!
//! The backend provides `num_slots` physical DPB slots (decode surfaces /
//! image layers). Frame buffers point at slots, `-1` = empty. This manager is
//! slot-agnostic: each backend maps slot indices to its own resources
//! (Vulkan DPB slot, cuvid `PicIdx`, VA surface-pool index).
//!
//! The output slot for a frame is the **oldest** (by last-use frame index)
//! non-live slot, where *live* = a slot that any of the 8 frame buffers points
//! at. This never overwrites a picture that is (or may still be) referenced,
//! and reproduces NVIDIA's cuvid parser wraparound exactly on streams whose
//! key frames are spaced beyond the surface count (verified pixel-perfect on
//! profile-0 against ffmpeg).
//!
//! Refreshes are applied **immediately** at commit time (spec-correct; the
//! cuvid parser's deferred-apply variant only differs in *which* free surface
//! a following key frame reuses — content is identical either way).

/// Number of VP9 frame buffers.
pub const VP9_NUM_FRAME_BUFFERS: usize = 8;

/// Common VP9 DPB manager (see module docs).
#[derive(Debug)]
pub struct Vp9Dpb {
    /// Number of physical DPB slots the backend allocated (>= 9 recommended so
    /// a free slot always exists while up to 8 are live).
    pub num_slots: u32,
    /// Frame buffer -> slot (`-1` = empty buffer).
    fb_to_slot: [i32; VP9_NUM_FRAME_BUFFERS],
    /// Per slot: decode-order frame index last written there (`-1` = never).
    slot_last_used: Vec<i32>,
    /// Decode-order frame count (used to stamp `slot_last_used`).
    decoded_frames: u32,
}

impl Vp9Dpb {
    pub fn new(num_slots: u32) -> Self {
        assert!(num_slots > 0, "Vp9Dpb needs at least one slot");
        Self {
            num_slots,
            fb_to_slot: [-1; VP9_NUM_FRAME_BUFFERS],
            slot_last_used: vec![-1; num_slots as usize],
            decoded_frames: 0,
        }
    }

    /// Slot index for frame buffer `fb`, or `-1` if the buffer is empty.
    /// Used for `show_existing_frame` (frame_to_show_map_idx).
    pub fn slot_of_frame_buffer(&self, fb: usize) -> i32 {
        self.fb_to_slot.get(fb).copied().unwrap_or(-1)
    }

    /// Slot indices of all 8 frame buffers in buffer order (`-1` = empty).
    /// Backends use this to fill whole reference arrays (e.g. VAAPI's
    /// `reference_frames[8]`).
    pub fn frame_buffer_slots(&self) -> [i32; VP9_NUM_FRAME_BUFFERS] {
        self.fb_to_slot
    }

    /// Map this frame's three active references (LAST, GOLDEN, ALTREF) to DPB
    /// slots. Intra frames (key or intra-only) take no references: `[-1; 3]`.
    pub fn reference_slots(&self, is_intra: bool, ref_frame_idx: &[u8; 3]) -> [i32; 3] {
        if is_intra {
            return [-1; 3];
        }
        let mut out = [-1i32; 3];
        for i in 0..3 {
            out[i] = self.slot_of_frame_buffer(ref_frame_idx[i] as usize);
        }
        out
    }

    /// Choose the slot to decode the next frame into: the **oldest**
    /// (smallest `slot_last_used`) slot that is not live. Ties break toward
    /// the lower slot index. The chosen slot is stamped with the current
    /// decode-order frame index.
    pub fn choose_output_slot(&mut self) -> i32 {
        let n = self.num_slots as usize;

        // Live set (pre-decode): slots pointed at by any frame buffer.
        let mut live = vec![false; n];
        for &s in &self.fb_to_slot {
            if s >= 0 && (s as usize) < n {
                live[s as usize] = true;
            }
        }

        // Scan all slots for the oldest non-live one.
        let mut best: Option<usize> = None;
        for (s, &is_live) in live.iter().enumerate() {
            if is_live {
                continue;
            }
            best = match best {
                None => Some(s),
                Some(b) => {
                    if self.slot_last_used[s] < self.slot_last_used[b]
                        || (self.slot_last_used[s] == self.slot_last_used[b] && s < b)
                    {
                        Some(s)
                    } else {
                        Some(b)
                    }
                }
            };
        }

        let chosen = match best {
            Some(s) => s,
            // Degenerate (all slots live — cannot happen with >= 9 slots):
            // overwrite the slot used longest ago.
            None => {
                let mut oldest = 0usize;
                for s in 1..n {
                    if self.slot_last_used[s] < self.slot_last_used[oldest] {
                        oldest = s;
                    }
                }
                oldest
            }
        };

        self.slot_last_used[chosen] = self.decoded_frames as i32;
        self.decoded_frames += 1;
        chosen as i32
    }

    /// Post-decode commit: apply `refresh_frame_flags` — every set bit `i`
    /// points frame buffer `i` at `slot`. Key frames pass `0xFF`.
    pub fn commit_frame(&mut self, refresh_frame_flags: u8, slot: i32) {
        if slot < 0 || (slot as usize) >= self.num_slots as usize {
            return;
        }
        for i in 0..VP9_NUM_FRAME_BUFFERS {
            if refresh_frame_flags >> i & 1 != 0 {
                self.fb_to_slot[i] = slot;
            }
        }
    }

    /// Drop all state (new stream / reconfigure / reset).
    pub fn reset(&mut self) {
        self.fb_to_slot = [-1; VP9_NUM_FRAME_BUFFERS];
        for s in &mut self.slot_last_used {
            *s = -1;
        }
        self.decoded_frames = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dpb8() -> Vp9Dpb {
        Vp9Dpb::new(16)
    }

    #[test]
    fn fresh_dpb_has_no_references() {
        let d = dpb8();
        assert_eq!(d.reference_slots(false, &[0, 1, 2]), [-1, -1, -1]);
        assert_eq!(d.frame_buffer_slots(), [-1; 8]);
        assert_eq!(d.slot_of_frame_buffer(0), -1);
    }

    #[test]
    fn key_frame_refreshes_all_buffers() {
        let mut d = dpb8();
        let slot = d.choose_output_slot();
        assert_eq!(slot, 0); // first frame -> oldest (never-used) slot 0
        d.commit_frame(0xFF, slot);
        for i in 0..8 {
            assert_eq!(d.slot_of_frame_buffer(i), 0);
        }
    }

    #[test]
    fn inter_frame_last_refresh_only() {
        // Mirrors the profile-0 samples: key (0xFF), then inter frames with
        // refresh=0x01 and refs [0,1,2].
        let mut d = dpb8();
        let s0 = d.choose_output_slot();
        d.commit_frame(0xFF, s0);

        let s1 = d.choose_output_slot();
        assert_ne!(s1, s0, "frame 1 must not reuse a live slot");
        let refs = d.reference_slots(false, &[0, 1, 2]);
        assert_eq!(
            refs,
            [s0, s0, s0],
            "all 3 refs still point at the key frame"
        );
        d.commit_frame(0x01, s1);

        // Now LAST (fb0) -> s1; GOLDEN/ALTREF (fb1/fb2) still -> s0.
        let s2 = d.choose_output_slot();
        assert_ne!(s2, s0);
        assert_ne!(s2, s1);
        let refs = d.reference_slots(false, &[0, 1, 2]);
        assert_eq!(refs, [s1, s0, s0]);
    }

    #[test]
    fn intra_frames_take_no_references() {
        let mut d = dpb8();
        let s0 = d.choose_output_slot();
        d.commit_frame(0xFF, s0);
        assert_eq!(d.reference_slots(true, &[0, 1, 2]), [-1, -1, -1]);
    }

    #[test]
    fn output_slot_never_collides_with_live_set() {
        let mut d = dpb8();
        let s0 = d.choose_output_slot();
        d.commit_frame(0xFF, s0); // all 8 buffers live on s0
        for _ in 1..8 {
            let si = d.choose_output_slot();
            assert_ne!(si, s0);
            // Commit with refresh=0 so the live set stays {s0}.
            d.commit_frame(0x00, si);
        }
        // After 8 more frames every slot except s0 has been used; the next
        // choice must still avoid s0 (the only live slot).
        let s9 = d.choose_output_slot();
        assert_ne!(s9, s0);
    }

    #[test]
    fn wraparound_reuses_oldest_non_live() {
        // 16 slots; simulate the cuvid baseline: frames 0..15 fill slots in
        // order when each new frame refreshes a distinct buffer so the live
        // set grows, then wraparound reuses the oldest non-live slot.
        let mut d = Vp9Dpb::new(16);
        let mut prev = -1i32;
        for i in 0..16u32 {
            let s = d.choose_output_slot();
            if i == 0 {
                assert_eq!(s, 0);
            } else {
                assert_ne!(s, prev);
            }
            prev = s;
            // Refresh buffer i%8 with this frame's slot.
            d.commit_frame(1u8 << (i % 8), s);
        }
        // Frame 16: buffers 0..7 all live (slots 8..15 for the last round,
        // slots 0..7 for the first). Oldest non-live = slot 0? No — slot 0 is
        // live (fb0 refreshed at frame 8). Oldest non-live among the free
        // slots... all 16 are live (each slot holds a distinct buffer content
        // from frames 8..15 + ... ). Actually buffers hold frames 8..15 in
        // slots 8..15 and frames 0..7 in slots 0..7 — wait, refresh i%8 at
        // frame i overwrote buffer i%8 with slot i, so after frame 15:
        // fb0->slot8 ... fb7->slot15. Slots 0..7 are NOT live.
        let s16 = d.choose_output_slot();
        assert_eq!(s16, 0, "oldest non-live slot is 0 (used at frame 0)");
    }

    #[test]
    fn reset_clears_state() {
        let mut d = dpb8();
        let s0 = d.choose_output_slot();
        d.commit_frame(0xFF, s0);
        d.reset();
        assert_eq!(d.frame_buffer_slots(), [-1; 8]);
        assert_eq!(d.choose_output_slot(), 0);
    }

    #[test]
    fn show_existing_uses_buffer_slot() {
        let mut d = dpb8();
        let s0 = d.choose_output_slot();
        d.commit_frame(0xFF, s0);
        let s1 = d.choose_output_slot();
        d.commit_frame(0x02, s1); // fb1 -> s1
        assert_eq!(d.slot_of_frame_buffer(1), s1);
        assert_eq!(d.slot_of_frame_buffer(0), s0);
    }
}

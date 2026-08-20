use vk_video_core::picture::H264Sps;
use vk_video_parser::h264::SliceHeader;

/// H.264 Picture Order Count calculator.
///
/// Implements the decoding process for picture order count per H.264
/// specification Annex D.3.3. Maintains state between frame calculations.
pub struct PocCalculator {
    // POC Type 0 state
    prev_pic_order_cnt_lsb: i32,
    prev_pic_order_cnt_msb: i32,
    // POC Type 1 state (H.264 D.3.3.2)
    prev_frame_num: i32,
    prev_pic_order_cnt: i32,
    last_pic_order_cnt: i32,       // PicOrderCnt of last reference frame
    last_pic_order_cnt_cycle: i32, // PicOrderCntCycle of last reference frame
    num_ref_frames_in_pic_order_cnt_cycle: u32,
    prev_is_reference: bool,
    has_prev_pic: bool,
}

impl PocCalculator {
    /// Create a new POC calculator.
    ///
    /// Initial state per H.264 D.3.3.2 (before first picture):
    /// - prev_frame_num = 0
    /// - prev_pic_order_cnt = 0
    /// - last_pic_order_cnt = 0
    /// - last_pic_order_cnt_cycle = 0
    /// - prev_is_reference = false
    pub fn new() -> Self {
        Self {
            prev_pic_order_cnt_lsb: 0,
            prev_pic_order_cnt_msb: 0,
            prev_frame_num: 0,
            prev_pic_order_cnt: 0,
            last_pic_order_cnt: 0,
            last_pic_order_cnt_cycle: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            prev_is_reference: false,
            has_prev_pic: false,
        }
    }

    /// Reset POC state. Call before processing an IDR picture.
    ///
    /// Per H.264 D.3.3.2, before decoding an IDR picture:
    /// - prev_frame_num = 0
    /// - prev_pic_order_cnt = 0
    /// - last_pic_order_cnt = 0
    /// - last_pic_order_cnt_cycle = 0
    pub fn reset(&mut self) {
        self.prev_pic_order_cnt_lsb = 0;
        self.prev_pic_order_cnt_msb = 0;
        self.prev_frame_num = 0;
        self.prev_pic_order_cnt = 0;
        self.last_pic_order_cnt = 0;
        self.last_pic_order_cnt_cycle = 0;
        self.prev_is_reference = false;
        self.has_prev_pic = false;
    }

    /// Calculate the Picture Order Count for a slice.
    ///
    /// Returns the POC value for the top field (or frame). For bottom fields,
    /// add `sps.offset_for_top_to_bottom_field` to obtain the bottom field POC.
    pub fn calculate(&mut self, sps: &H264Sps, slh: &SliceHeader, is_reference: bool) -> i32 {
        // Update cycle count for POC Type 1
        self.num_ref_frames_in_pic_order_cnt_cycle = sps.num_ref_frames_in_pic_order_cnt_cycle;

        let poc = match sps.pic_order_cnt_type {
            0 => self.calc_type0(sps, slh),
            1 => self.calc_type1(sps, slh, is_reference),
            2 => self.calc_type2(slh, is_reference),
            _ => 0,
        };

        if slh.field_pic_flag && slh.bottom_field {
            poc + sps.offset_for_top_to_bottom_field
        } else {
            poc
        }
    }

    /// POC Type 0: explicit with pic_order_cnt_lsb and MSB wrapping.
    ///
    /// Per H.264 Annex D.3.3.1:
    /// - If lsb decreased by >= Max/2: wrap upward (msb += Max)
    /// - If lsb increased by > Max/2: wrap downward (msb -= Max)
    ///   Note: strictly greater than for the second condition per spec
    fn calc_type0(&mut self, sps: &H264Sps, slh: &SliceHeader) -> i32 {
        let max_pic_order_cnt_lsb = sps.max_pic_order_cnt_lsb as i32;
        let pic_order_cnt_lsb = slh.pic_order_cnt_lsb;

        let pic_order_cnt_msb = if self.has_prev_pic
            && pic_order_cnt_lsb < self.prev_pic_order_cnt_lsb
            && (self.prev_pic_order_cnt_lsb - pic_order_cnt_lsb) >= max_pic_order_cnt_lsb / 2
        {
            // Large decrease → wrapped upward
            self.prev_pic_order_cnt_msb + max_pic_order_cnt_lsb
        } else if self.has_prev_pic
            && pic_order_cnt_lsb > self.prev_pic_order_cnt_lsb
            && (pic_order_cnt_lsb - self.prev_pic_order_cnt_lsb) > max_pic_order_cnt_lsb / 2
        {
            // Large increase → wrapped downward
            // Note: strictly > per H.264 D.3.3.1
            self.prev_pic_order_cnt_msb - max_pic_order_cnt_lsb
        } else {
            self.prev_pic_order_cnt_msb
        };

        self.prev_pic_order_cnt_lsb = pic_order_cnt_lsb;
        self.prev_pic_order_cnt_msb = pic_order_cnt_msb;
        self.has_prev_pic = true;

        pic_order_cnt_msb + pic_order_cnt_lsb
    }

    /// POC Type 1: explicit with delta_pic_order_cnt and offset cycling.
    ///
    /// Per H.264 Annex D.3.3.2. This is the most complex POC type.
    ///
    /// For frame pictures:
    ///   - Reference frames cycle through offset_for_ref_frame[]
    ///   - Non-reference frames use offset_for_non_ref_pic
    ///   - Bottom field POC = top field POC + offset_for_top_to_bottom_field
    ///
    /// For field pictures:
    ///   - Each field increments prev_frame_num
    ///   - POC = prev_frame_num + delta_pic_order_cnt[0]
    fn calc_type1(&mut self, sps: &H264Sps, slh: &SliceHeader, is_reference: bool) -> i32 {
        let is_frame = sps.frame_mbs_only_flag || !slh.field_pic_flag;
        let delta0 = slh.delta_pic_order_cnt[0];

        if is_frame {
            // Per H.264 D.3.3.2: detect frame_num wrap-around and reset cycle.
            // No `> 0` guard — wrap from max-1 to 0 must also be detected.
            if slh.frame_num < self.prev_frame_num as u32 {
                self.last_pic_order_cnt_cycle = 0;
            }

            // Calculate top field POC per H.264 D.3.3.2
            let top_field_order_cnt = if is_reference {
                if self.num_ref_frames_in_pic_order_cnt_cycle == 0 {
                    // When NumRefFramesInPicOrderCntCycle == 0:
                    // CurrFieldOrderCnt[0] = PrevPicOrderCnt + 2
                    self.prev_pic_order_cnt + 2
                } else if self.prev_is_reference {
                    // Previous frame was reference: use its POC + offset
                    self.prev_pic_order_cnt
                        + sps.offset_for_ref_frame[self.last_pic_order_cnt_cycle
                            as usize % self.num_ref_frames_in_pic_order_cnt_cycle as usize]
                } else {
                    // Previous frame was non-reference: use last ref POC + offset
                    self.last_pic_order_cnt
                        + sps.offset_for_ref_frame[self.last_pic_order_cnt_cycle
                            as usize % self.num_ref_frames_in_pic_order_cnt_cycle as usize]
                }
            } else {
                // Non-reference picture
                if self.num_ref_frames_in_pic_order_cnt_cycle == 0 {
                    // When NumRefFramesInPicOrderCntCycle == 0:
                    // CurrFieldOrderCnt[0] = PrevPicOrderCnt + 2 * OffsetForNonRefPic
                    self.prev_pic_order_cnt + 2 * sps.offset_for_non_ref_pic
                } else if self.prev_is_reference {
                    // Per spec D.3.3.2: LastPicOrderCnt + OffsetForNonRefPic
                    self.last_pic_order_cnt + sps.offset_for_non_ref_pic
                } else {
                    self.prev_pic_order_cnt + sps.offset_for_non_ref_pic
                }
            };

            // Update state per H.264 D.3.3.2
            self.prev_frame_num = slh.frame_num as i32;
            self.prev_pic_order_cnt = top_field_order_cnt;

            if is_reference {
                self.last_pic_order_cnt = top_field_order_cnt;
                self.last_pic_order_cnt_cycle =
                    (self.last_pic_order_cnt_cycle + 1)
                        % self.num_ref_frames_in_pic_order_cnt_cycle as i32;
            }
            self.prev_is_reference = is_reference;

            top_field_order_cnt
        } else {
            // Field picture: each field has its own POC
            // Per H.264 D.3.3.2 for field pictures:
            // prev_frame_num is incremented for each field
            self.prev_frame_num += 1;
            let field_poc = self.prev_frame_num + delta0;
            self.prev_pic_order_cnt = field_poc;
            self.prev_is_reference = is_reference;

            field_poc
        }
    }

    /// POC Type 2: implicit from frame_num.
    ///
    /// Per H.264 Annex D.3.3.3:
    /// - Reference frames: POC = frame_num * 2
    /// - Non-reference frame pictures: POC = frame_num * 2 + 1
    fn calc_type2(&mut self, slh: &SliceHeader, is_reference: bool) -> i32 {
        let frame_num = slh.frame_num as i32;
        self.prev_frame_num = frame_num;

        if is_reference {
            frame_num * 2
        } else {
            frame_num * 2 + 1
        }
    }
}

impl Default for PocCalculator {
    fn default() -> Self {
        Self::new()
    }
}

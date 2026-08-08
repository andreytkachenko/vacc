//! Access unit extraction from H.264/H.265 bitstreams.

use vk_video_parser::nal::{
    find_next_start_code, parse_h264_nal_header, parse_h265_nal_header,
    remove_emulation_prevention_bytes,
};

/// An access unit (single frame) extracted from the bitstream.
#[derive(Debug, Clone)]
pub struct AccessUnit {
    /// Bitstream data (slice NALs with start codes, no SPS/PPS)
    pub data: Vec<u8>,
    /// Offsets of each slice within data (pointing to start codes)
    pub slice_offsets: Vec<u32>,
    /// Frame number from first slice header
    pub frame_num: u32,
    /// Picture order count [top_field, bottom_field]
    pub pic_order_cnt: [i32; 2],
    /// Whether this is an IDR frame
    pub is_idr: bool,
    /// Whether this is a reference frame
    pub is_reference: bool,
    /// Slice type (0=I, 1=P, 2=B, 3=SI, 4=SP for H.264)
    pub slice_type: u32,
    /// H.265: NumBitsForShortTermRPSInSlice from slice header
    pub num_bits_for_st_ref_pic_set_in_slice: i32,
    /// H.265: NumDeltaPocsOfRefRpsIdx from slice header
    pub num_delta_pocs_of_ref_rps_idx: i32,
    /// H.265: short_term_ref_pic_set_sps_flag from slice header
    pub short_term_ref_pic_set_sps_flag: bool,
    /// H.265: Computed reference picture POCs from RPS
    pub ref_pocs: Vec<i32>,
    /// H.264: adaptive_ref_pic_marking_mode_flag from slice header (true=MMCO, false=sliding window)
    pub adaptive_ref_pic_marking_mode_flag: bool,
    /// H.264: MMCO commands parsed from slice header
    pub mmco_commands: Vec<H264MmcoCommand>,
}

/// H.264 Memory Management Control Operation (MMCO) command.
/// See H.264 spec 8.2.5.4 for details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H264MmcoCommand {
    /// MMCO 1: Unmark short-term reference with difference_of_pic_nums_minus1
    UnmarkShortTerm { difference_of_pic_nums_minus1: u32 },
    /// MMCO 2: Unmark long-term reference with long_term_frame_idx
    UnmarkLongTerm { long_term_frame_idx: u32 },
    /// MMCO 3: Assign LongTermFrameIdx to short-term reference
    AssignLongTerm { difference_of_pic_nums_minus1: u32, long_term_frame_idx: u32 },
    /// MMCO 4: Set MaxLongTermFrameIdx
    SetMaxLongTermFrameIdx { max_long_term_frame_idx_plus1: u32 },
    /// MMCO 5: Unmark all references
    UnmarkAll,
    /// MMCO 6: Assign LongTermFrameIdx to current picture
    AssignLongTermToCurrent { long_term_frame_idx: u32 },
}

/// Enum to hold either H.264 or H.265 SPS.
#[derive(Debug, Clone)]
pub enum H264OrH265Sps {
    H264(vk_video_core::picture::H264Sps),
    H265(vk_video_core::picture::H265Sps),
}

/// Enum to hold either H.264 or H.265 PPS.
#[derive(Debug, Clone)]
pub enum H264OrH265Pps {
    H264(vk_video_core::picture::H264Pps),
    H265(vk_video_core::picture::H265Pps),
}

/// Minimal bit reader for slice header parsing.
struct SliceBitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> SliceBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bit(&mut self) -> Option<u32> {
        let byte_idx = self.pos / 8;
        if byte_idx >= self.data.len() {
            return None;
        }
        let bit_idx = 7 - (self.pos % 8);
        let bit = ((self.data[byte_idx] >> bit_idx) & 1) as u32;
        self.pos += 1;
        Some(bit)
    }

    fn read_bits(&mut self, n: u32) -> Option<u32> {
        let mut val = 0u32;
        for _ in 0..n {
            val = (val << 1) | self.read_bit()?;
        }
        Some(val)
    }

    fn read_ue(&mut self) -> Option<u32> {
        let mut leading_zeros = 0u32;
        loop {
            let bit = self.read_bit()?;
            if bit == 1 {
                break;
            }
            leading_zeros += 1;
            if leading_zeros >= 32 {
                return None;
            }
        }
        let mut value = 0u32;
        for _ in 0..leading_zeros {
            value = (value << 1) | self.read_bit()?;
        }
        Some((1 << leading_zeros) - 1 + value)
    }

    fn read_se(&mut self) -> Option<i32> {
        let ue = self.read_ue()?;
        if ue & 1 != 0 {
            Some((ue + 1) as i32 / 2)
        } else {
            Some(-((ue as i32) / 2))
        }
    }

    fn pos(&self) -> usize {
        self.pos
    }
}

/// Parse H.264 slice header to extract frame boundary info.
fn parse_h264_slice_header(
    nal_data: &[u8],
    sps: &vk_video_core::picture::H264Sps,
    nal_ref_idc: u8,
    nal_unit_type: u8,
    prev_pic_order_cnt_lsb: i32,
    prev_pic_order_cnt_msb: i32,
    max_pic_order_cnt_lsb: u32,
) -> Option<(u32, u32, i32, i32, [i32; 2], u32, bool, Vec<H264MmcoCommand>)> {
    if nal_data.len() < 4 {
        return None;
    }

    let payload = &nal_data[1..];
    let mut r = SliceBitReader::new(payload);

    let first_mb_in_slice = r.read_ue()?;
    let slice_type = r.read_ue()?;
    let _pps_id = r.read_ue()?;

    let frame_num_bits = sps.log2_max_frame_num_minus4 as u32 + 4;
    let frame_num = r.read_bits(frame_num_bits)?;

    let is_idr = nal_unit_type == 5;

    if is_idr {
        let _idr_pic_id = r.read_ue().unwrap_or(0);
    }

    let pic_order_cnt_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
    let pic_order_cnt_lsb = r.read_bits(pic_order_cnt_lsb_bits)? as i32;

    let pic_order_cnt_msb = if is_idr {
        0
    } else if pic_order_cnt_lsb < prev_pic_order_cnt_lsb
        && (prev_pic_order_cnt_lsb - pic_order_cnt_lsb) >= (max_pic_order_cnt_lsb as i32 / 2)
    {
        prev_pic_order_cnt_msb + max_pic_order_cnt_lsb as i32
    } else if pic_order_cnt_lsb > prev_pic_order_cnt_lsb
        && (pic_order_cnt_lsb - prev_pic_order_cnt_lsb) > (max_pic_order_cnt_lsb as i32 / 2)
    {
        prev_pic_order_cnt_msb - max_pic_order_cnt_lsb as i32
    } else {
        prev_pic_order_cnt_msb
    };

    let pic_order_cnt = [
        pic_order_cnt_msb + pic_order_cnt_lsb,
        pic_order_cnt_msb + pic_order_cnt_lsb,
    ];

    // adaptive_ref_pic_marking_mode_flag appears after pic_order_cnt for reference frames
    let adaptive_ref_pic_marking_mode_flag = if nal_ref_idc > 0 {
        r.read_bit().unwrap_or(0) != 0
    } else {
        false
    };

    // Parse MMCO commands if adaptive_ref_pic_marking_mode_flag is true
    // See H.264 spec 7.3.3 and 8.2.5.4
    let mut mmco_commands = Vec::new();
    if adaptive_ref_pic_marking_mode_flag {
        loop {
            let Some(memory_management_control_operation) = r.read_ue() else {
                break;
            };

            // MMCO 0 is the terminator
            if memory_management_control_operation == 0 {
                break;
            }

            let cmd = match memory_management_control_operation {
                // MMCO 1: Unmark short-term reference
                1 => {
                    let difference_of_pic_nums_minus1 = r.read_ue().unwrap_or(0);
                    H264MmcoCommand::UnmarkShortTerm { difference_of_pic_nums_minus1 }
                }
                // MMCO 2: Unmark long-term reference
                2 => {
                    let long_term_frame_idx = r.read_ue().unwrap_or(0);
                    H264MmcoCommand::UnmarkLongTerm { long_term_frame_idx }
                }
                // MMCO 3: Assign LongTermFrameIdx to short-term reference
                3 => {
                    let difference_of_pic_nums_minus1 = r.read_ue().unwrap_or(0);
                    let long_term_frame_idx = r.read_ue().unwrap_or(0);
                    H264MmcoCommand::AssignLongTerm { difference_of_pic_nums_minus1, long_term_frame_idx }
                }
                // MMCO 4: Set MaxLongTermFrameIdx
                4 => {
                    let long_term_frame_idx = r.read_ue().unwrap_or(0);
                    H264MmcoCommand::SetMaxLongTermFrameIdx { max_long_term_frame_idx_plus1: long_term_frame_idx }
                }
                // MMCO 5: Unmark all references
                5 => H264MmcoCommand::UnmarkAll,
                // MMCO 6: Assign LongTermFrameIdx to current picture
                6 => {
                    let long_term_frame_idx = r.read_ue().unwrap_or(0);
                    H264MmcoCommand::AssignLongTermToCurrent { long_term_frame_idx }
                }
                _ => {
                    // Unknown MMCO, stop parsing
                    break;
                }
            };
            mmco_commands.push(cmd);
        }
    }

    Some((
        first_mb_in_slice,
        frame_num,
        pic_order_cnt_lsb,
        pic_order_cnt_msb,
        pic_order_cnt,
        slice_type,
        adaptive_ref_pic_marking_mode_flag,
        mmco_commands,
    ))
}

/// Parse H.265 slice header to extract frame boundary info.
fn parse_h265_slice_header(
    nal_data: &[u8],
    sps: &vk_video_core::picture::H265Sps,
    pps: &vk_video_core::picture::H265Pps,
    nal_unit_type: u8,
    nuh_temporal_id_plus1: u8,
    prev_pic_order_cnt_lsb: i32,
    prev_pic_order_cnt_msb: i32,
) -> Option<(bool, i32, i32, [i32; 2], bool, bool, u32, i32, i32, bool, Vec<i32>)> {
    if nal_data.len() < 3 {
        return None;
    }

    let payload = remove_emulation_prevention_bytes(&nal_data[2..]);
    let mut r = SliceBitReader::new(&payload);

    let first_slice_segment_in_pic_flag = r.read_bit()? == 1;

    let is_rap = nal_unit_type >= 16 && nal_unit_type <= 23;
    if is_rap {
        let _no_output_of_prior_pics_flag = r.read_bit().unwrap_or(0);
    }

    let _pps_id = r.read_ue().unwrap_or(0);

    if !first_slice_segment_in_pic_flag {
        return None;
    }

    if pps.num_extra_slice_header_bits > 0 {
        let _extra_bits = r.read_bits(pps.num_extra_slice_header_bits as u32).unwrap_or(0);
    }

    let slice_type = r.read_ue().unwrap_or(0);

    if pps.output_flag_present_flag {
        let _pic_output_flag = r.read_bit().unwrap_or(0);
    }

    if sps.separate_colour_plane_flag {
        let _colour_plane_id = r.read_bits(2).unwrap_or(0);
    }

    let is_idr = nal_unit_type == 19 || nal_unit_type == 20;

    let pic_order_cnt_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
    let pic_order_cnt_lsb = if is_idr {
        0i32
    } else {
        r.read_bits(pic_order_cnt_lsb_bits)? as i32
    };

    let pic_order_cnt_msb = if nal_unit_type >= 16 && nal_unit_type <= 20 {
        0
    } else {
        let max_pic_order_cnt_lsb = 1i32 << pic_order_cnt_lsb_bits;
        if pic_order_cnt_lsb < prev_pic_order_cnt_lsb
            && (prev_pic_order_cnt_lsb - pic_order_cnt_lsb) >= (max_pic_order_cnt_lsb / 2)
        {
            prev_pic_order_cnt_msb + max_pic_order_cnt_lsb
        } else if pic_order_cnt_lsb > prev_pic_order_cnt_lsb
            && (pic_order_cnt_lsb - prev_pic_order_cnt_lsb) > (max_pic_order_cnt_lsb / 2)
        {
            prev_pic_order_cnt_msb - max_pic_order_cnt_lsb
        } else {
            prev_pic_order_cnt_msb
        }
    };

    let pic_order_cnt_val = pic_order_cnt_msb + pic_order_cnt_lsb;
    let pic_order_cnt = [pic_order_cnt_val, pic_order_cnt_val];

    let is_reference = if is_rap {
        true
    } else {
        nal_unit_type % 2 == 1
    };

    let mut num_bits_for_st_ref_pic_set_in_slice: i32 = 0;
    let mut num_delta_pocs_of_ref_rps_idx: i32 = 0;
    let mut short_term_ref_pic_set_sps_flag: bool = !is_idr;
    let mut ref_pocs: Vec<i32> = Vec::new();

    if !is_idr {
        short_term_ref_pic_set_sps_flag = r.read_bit().unwrap_or(0) == 1;

        if !short_term_ref_pic_set_sps_flag {
            let bitcnt_before = r.pos();

            let inter_ref_pic_set_prediction_flag = if sps.num_short_term_ref_pic_sets > 0 {
                r.read_bit().unwrap_or(0) == 1
            } else {
                false
            };

            if inter_ref_pic_set_prediction_flag {
                let idx = sps.num_short_term_ref_pic_sets as u32;
                let delta_idx_minus1 = r.read_ue().unwrap_or(0) as u32;
                let r_idx = idx as usize - (delta_idx_minus1 as usize + 1);

                let delta_rps_sign = r.read_bit().unwrap_or(0) == 1;
                let abs_delta_rps_minus1 = r.read_ue().unwrap_or(0) as i32;
                let delta_rps = if delta_rps_sign {
                    -(abs_delta_rps_minus1 + 1)
                } else {
                    abs_delta_rps_minus1 + 1
                };

                if r_idx < sps.short_term_ref_pic_sets.len() {
                    let ref_strps = &sps.short_term_ref_pic_sets[r_idx];
                    num_delta_pocs_of_ref_rps_idx =
                        (ref_strps.num_negative_pics as i32 + ref_strps.num_positive_pics as i32);

                    let num_ref_entries =
                        ref_strps.num_negative_pics as usize + ref_strps.num_positive_pics as usize;
                    let mut used_by_curr_pic_flag = vec![false; num_ref_entries + 1];
                    let mut use_delta_flag = vec![true; num_ref_entries + 1];

                    for j in 0..=num_ref_entries {
                        used_by_curr_pic_flag[j] = r.read_bit().unwrap_or(0) == 1;
                        if !used_by_curr_pic_flag[j] {
                            use_delta_flag[j] = r.read_bit().unwrap_or(0) == 1;
                        } else {
                            use_delta_flag[j] = true;
                        }
                    }

                    let curr_poc = pic_order_cnt_val;

                    let mut ref_poc_s0: Vec<i32> = Vec::new();
                    for i in 0..ref_strps.num_negative_pics as usize {
                        let stored = ref_strps.delta_poc_s0_minus1[i] as i32;
                        let delta_poc = if stored > 32767 {
                            stored - 65536
                        } else {
                            stored
                        };
                        ref_poc_s0.push(curr_poc + delta_poc);
                    }

                    let mut ref_poc_s1: Vec<i32> = Vec::new();
                    for i in 0..ref_strps.num_positive_pics as usize {
                        let delta = ref_strps.delta_poc_s1_minus1[i] as i32;
                        ref_poc_s1.push(curr_poc + delta);
                    }

                    let mut new_num_neg: usize = 0;
                    for j in (0..ref_strps.num_positive_pics as usize).rev() {
                        let new_poc = ref_poc_s1[j] + delta_rps;
                        let entry_idx = ref_strps.num_negative_pics as usize + j;
                        if new_poc < curr_poc && use_delta_flag[entry_idx] {
                            if used_by_curr_pic_flag[entry_idx] {
                                ref_pocs.push(new_poc);
                            }
                            new_num_neg += 1;
                        }
                    }
                    if delta_rps < 0 && use_delta_flag[num_ref_entries] {
                        let new_poc = curr_poc + delta_rps;
                        if used_by_curr_pic_flag[num_ref_entries] {
                            ref_pocs.push(new_poc);
                        }
                        new_num_neg += 1;
                    }
                    for j in 0..ref_strps.num_negative_pics as usize {
                        let new_poc = ref_poc_s0[j] + delta_rps;
                        if new_poc < curr_poc && use_delta_flag[j] {
                            if used_by_curr_pic_flag[j] {
                                ref_pocs.push(new_poc);
                            }
                            new_num_neg += 1;
                        }
                    }

                    for j in (0..ref_strps.num_negative_pics as usize).rev() {
                        let new_poc = ref_poc_s0[j] + delta_rps;
                        if new_poc > curr_poc && use_delta_flag[j] {
                            if used_by_curr_pic_flag[j] {
                                ref_pocs.push(new_poc);
                            }
                        }
                    }
                    if delta_rps > 0 && use_delta_flag[num_ref_entries] {
                        let new_poc = curr_poc + delta_rps;
                        if used_by_curr_pic_flag[num_ref_entries] {
                            ref_pocs.push(new_poc);
                        }
                    }
                    for j in 0..ref_strps.num_positive_pics as usize {
                        let new_poc = ref_poc_s1[j] + delta_rps;
                        let entry_idx = ref_strps.num_negative_pics as usize + j;
                        if new_poc > curr_poc && use_delta_flag[entry_idx] {
                            if used_by_curr_pic_flag[entry_idx] {
                                ref_pocs.push(new_poc);
                            }
                        }
                    }
                }
            } else {
                let num_negative_pics = r.read_ue().unwrap_or(0) as i32;
                let num_positive_pics = r.read_ue().unwrap_or(0) as i32;
                let curr_poc = pic_order_cnt_val;

                let mut cumulative_delta_poc_s0: i32 = 0;
                for i in 0..num_negative_pics {
                    let delta = r.read_ue().unwrap_or(0) as i32;
                    cumulative_delta_poc_s0 += delta + 1;
                    let used = r.read_bit().unwrap_or(0);
                    if used == 1 {
                        let ref_poc = curr_poc - cumulative_delta_poc_s0;
                        ref_pocs.push(ref_poc);
                    }
                }

                let mut cumulative_delta_poc_s1: i32 = 0;
                for i in 0..num_positive_pics {
                    let delta = r.read_ue().unwrap_or(0) as i32;
                    cumulative_delta_poc_s1 += delta + 1;
                    let used = r.read_bit().unwrap_or(0);
                    if used == 1 {
                        let ref_poc = curr_poc + cumulative_delta_poc_s1;
                        ref_pocs.push(ref_poc);
                    }
                }
            }

            let bitcnt_after = r.pos();
            num_bits_for_st_ref_pic_set_in_slice = (bitcnt_after - bitcnt_before) as i32;
        } else {
            let num_short_term_ref_pic_sets = sps.num_short_term_ref_pic_sets as u32;
            let short_term_ref_pic_set_idx = if num_short_term_ref_pic_sets > 1 {
                let strps_idx_bits = (num_short_term_ref_pic_sets as f64).log2().ceil() as u32;
                r.read_bits(strps_idx_bits).unwrap_or(0) as usize
            } else {
                0
            };

            if short_term_ref_pic_set_idx < sps.short_term_ref_pic_sets.len() {
                let strps = &sps.short_term_ref_pic_sets[short_term_ref_pic_set_idx];
                let curr_poc = pic_order_cnt_val;

                for i in 0..strps.num_negative_pics as usize {
                    if (strps.used_by_curr_pic_s0_flag & (1 << i)) != 0 {
                        let stored = strps.delta_poc_s0_minus1[i] as i32;
                        let delta_poc = if stored > 32767 {
                            stored - 65536
                        } else {
                            stored
                        };
                        let ref_poc = curr_poc + delta_poc;
                        ref_pocs.push(ref_poc);
                    }
                }

                for i in 0..strps.num_positive_pics as usize {
                    if (strps.used_by_curr_pic_s1_flag & (1 << i)) != 0 {
                        let ref_poc = curr_poc + strps.delta_poc_s1_minus1[i] as i32;
                        ref_pocs.push(ref_poc);
                    }
                }
            }
        }

        if sps.long_term_ref_pics_present_flag && sps.num_long_term_ref_pics_sps > 0 {
            let num_long_term_sps = r.read_ue().unwrap_or(0);
            let num_long_term_pics = r.read_ue().unwrap_or(0);

            let lt_idx_bits = if sps.num_long_term_ref_pics_sps > 1 {
                (sps.num_long_term_ref_pics_sps as f64).log2().ceil() as u32
            } else {
                0
            };
            let poc_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;

            for i in 0u32..(num_long_term_sps + num_long_term_pics) {
                let mut poc_lsb: i32 = 0;
                if i < num_long_term_sps {
                    if lt_idx_bits > 0 {
                        let lt_idx_sps = r.read_bits(lt_idx_bits).unwrap_or(0);
                        poc_lsb = sps.lt_ref_pic_poc_lsb_sps[lt_idx_sps as usize] as i32;
                    } else {
                        poc_lsb = sps.lt_ref_pic_poc_lsb_sps[0] as i32;
                    }
                } else {
                    poc_lsb = r.read_bits(poc_lsb_bits).unwrap_or(0) as i32;
                    let _used_by_curr_pic_lt_flag = r.read_bit().unwrap_or(0);
                }

                ref_pocs.push(poc_lsb);

                let delta_poc_msb_present_flag = r.read_bit().unwrap_or(0);
                if delta_poc_msb_present_flag == 1 {
                    let _delta_poc_msb_cycle_lt = r.read_ue().unwrap_or(0);
                }
            }
        }

        if sps.sps_temporal_mvp_enabled_flag {
            let _slice_temporal_mvp_enabled_flag = r.read_bit().unwrap_or(0);
        }
    }

    Some((
        first_slice_segment_in_pic_flag,
        pic_order_cnt_lsb,
        pic_order_cnt_msb,
        pic_order_cnt,
        is_idr,
        is_reference,
        slice_type,
        num_bits_for_st_ref_pic_set_in_slice,
        num_delta_pocs_of_ref_rps_idx,
        short_term_ref_pic_set_sps_flag,
        ref_pocs,
    ))
}

/// Codec type for access unit extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    H264,
    H265,
}

/// Extract all access units from the bitstream.
pub fn extract_all_access_units(
    data: &[u8],
    codec: VideoCodec,
    max_frames: usize,
    sps: Option<&H264OrH265Sps>,
    pps: Option<&H264OrH265Pps>,
) -> Vec<AccessUnit> {
    let mut access_units: Vec<AccessUnit> = Vec::new();
    let mut offset = 0;
    let mut current_au_data: Vec<u8> = Vec::new();
    let mut current_slice_offsets: Vec<u32> = Vec::new();
    let mut current_frame_num: u32 = 0;
    let mut current_poc: [i32; 2] = [0, 0];
    let mut current_is_idr: bool = false;
    let mut current_is_reference: bool = true;
    let mut current_slice_type: u32 = 0;
    let mut current_num_bits_for_st_ref_pic_set_in_slice: i32 = 0;
    let mut current_num_delta_pocs_of_ref_rps_idx: i32 = 0;
    let mut current_short_term_ref_pic_set_sps_flag: bool = true;
    let mut current_ref_pocs: Vec<i32> = Vec::new();
    let mut current_adaptive_ref_pic_marking_mode_flag: bool = false;
    let mut current_mmco_commands: Vec<H264MmcoCommand> = Vec::new();
    let mut in_frame = false;

    let mut prev_pic_order_cnt_lsb: i32 = 0;
    let mut prev_pic_order_cnt_msb: i32 = 0;
    let mut max_pic_order_cnt_lsb: u32 = 256;
    let mut prev_frame_num: u32 = 0;

    let h264_sps = match sps {
        Some(H264OrH265Sps::H264(s)) => Some(s),
        _ => None,
    };
    let h264_pps = match pps {
        Some(H264OrH265Pps::H264(p)) => Some(p),
        _ => None,
    };

    let h265_sps = match sps {
        Some(H264OrH265Sps::H265(s)) => Some(s),
        _ => None,
    };
    let h265_pps = match pps {
        Some(H264OrH265Pps::H265(p)) => Some(p),
        _ => None,
    };

    if let Some(sps) = h264_sps {
        max_pic_order_cnt_lsb = 1u32 << (sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4);
    }

    while offset < data.len() && access_units.len() < max_frames {
        let Some((start, code_len)) = find_next_start_code(data, offset) else {
            break;
        };

        let next_start = find_next_start_code(data, start + code_len);
        let end = next_start.map(|(s, _)| s).unwrap_or(data.len());

        let nal_data = &data[start + code_len..end];
        if nal_data.is_empty() {
            offset = end;
            continue;
        }

        let (nal_type, is_irap, is_au_delimiter, is_slice, is_params) = match codec {
            VideoCodec::H264 => {
                if let Some((_, _, t)) = parse_h264_nal_header(nal_data) {
                    let is_idr = t == 5;
                    let is_aud = t == 9;
                    let is_slice_type = matches!(t, 1..=5);
                    let is_params_type = t == 7 || t == 8;
                    (t as usize, is_idr, is_aud, is_slice_type, is_params_type)
                } else {
                    (0, false, false, false, false)
                }
            }
            VideoCodec::H265 => {
                if let Some((_, t, _, _)) = parse_h265_nal_header(nal_data) {
                    let is_irap_type = matches!(t, 16..=23);
                    let is_aud = t == 38;
                    let is_slice_type = matches!(t, 0..=31);
                    let is_params_type = matches!(t, 32..=34);
                    (t as usize, is_irap_type, is_aud, is_slice_type, is_params_type)
                } else {
                    (0, false, false, false, false)
                }
            }
        };

        if is_au_delimiter {
            if in_frame && !current_au_data.is_empty() {
                 access_units.push(AccessUnit {
                      data: current_au_data.clone(),
                      slice_offsets: current_slice_offsets.clone(),
                      frame_num: current_frame_num,
                      pic_order_cnt: current_poc,
                      is_idr: current_is_idr,
                      is_reference: current_is_reference,
                      slice_type: current_slice_type,
                      num_bits_for_st_ref_pic_set_in_slice: current_num_bits_for_st_ref_pic_set_in_slice,
                      num_delta_pocs_of_ref_rps_idx: current_num_delta_pocs_of_ref_rps_idx,
                      short_term_ref_pic_set_sps_flag: current_short_term_ref_pic_set_sps_flag,
                      ref_pocs: current_ref_pocs.clone(),
                      adaptive_ref_pic_marking_mode_flag: current_adaptive_ref_pic_marking_mode_flag,
                      mmco_commands: current_mmco_commands.clone(),
                  });
                  current_au_data.clear();
                  current_slice_offsets.clear();
                  current_mmco_commands.clear();
            }
            offset = end;
            continue;
        }

        if is_params {
            offset = end;
            continue;
        }

        if is_slice {
            let is_new_frame;

            if codec == VideoCodec::H264 {
                if let Some(H264OrH265Sps::H264(sps)) = sps {
                    if let Some((_, ref_idc, nal_unit_type)) = parse_h264_nal_header(nal_data) {
                        if let Some((first_mb, frame_num, poc_lsb, poc_msb, poc, slice_type, adaptive_ref_pic_marking_mode_flag, mmco_commands)) =
                            parse_h264_slice_header(
                                nal_data,
                                sps,
                                ref_idc,
                                nal_unit_type,
                                prev_pic_order_cnt_lsb,
                                prev_pic_order_cnt_msb,
                                max_pic_order_cnt_lsb,
                            )
                        {
                            let is_idr_slice = nal_unit_type == 5;

                            is_new_frame = if !in_frame || current_slice_offsets.is_empty() {
                                true
                            } else if first_mb == 0 {
                                true
                            } else if frame_num != prev_frame_num {
                                true
                            } else if current_is_idr && !is_idr_slice {
                                // Non-IDR slice after IDR slice must be a new frame
                                // (IDR frames cannot contain Non-IDR slices)
                                true
                            } else {
                                false
                            };

                             if is_new_frame {
                                 if in_frame && !current_au_data.is_empty() {
                                      access_units.push(AccessUnit {
                                          data: current_au_data.clone(),
                                          slice_offsets: current_slice_offsets.clone(),
                                          frame_num: current_frame_num,
                                          pic_order_cnt: current_poc,
                                          is_idr: current_is_idr,
                                          is_reference: current_is_reference,
                                          slice_type: current_slice_type,
                                          num_bits_for_st_ref_pic_set_in_slice: current_num_bits_for_st_ref_pic_set_in_slice,
                                          num_delta_pocs_of_ref_rps_idx: current_num_delta_pocs_of_ref_rps_idx,
                                          short_term_ref_pic_set_sps_flag: current_short_term_ref_pic_set_sps_flag,
                                          ref_pocs: current_ref_pocs.clone(),
                                          adaptive_ref_pic_marking_mode_flag: current_adaptive_ref_pic_marking_mode_flag,
                                          mmco_commands: current_mmco_commands.clone(),
                                      });
                                      current_au_data.clear();
                                      current_slice_offsets.clear();
                                      current_mmco_commands.clear();
                                  }

                                   current_is_idr = is_idr_slice;
                                 current_is_reference = ref_idc != 0;
                                 current_frame_num = frame_num;
                                 current_poc = poc;
                                 current_slice_type = slice_type;
                                 current_adaptive_ref_pic_marking_mode_flag = adaptive_ref_pic_marking_mode_flag;
                                 current_mmco_commands = mmco_commands;

                                 if ref_idc != 0 {
                                    prev_pic_order_cnt_lsb = poc_lsb;
                                    prev_pic_order_cnt_msb = poc_msb;
                                    prev_frame_num = frame_num;
                                }

                                if is_idr_slice {
                                    prev_pic_order_cnt_lsb = 0;
                                    prev_pic_order_cnt_msb = 0;
                                }

                                in_frame = true;
                            }
                        } else {
                            is_new_frame = true;
                        }
                    } else {
                        is_new_frame = !in_frame || current_slice_offsets.is_empty();
                    }
                } else {
                    is_new_frame = if !in_frame || current_slice_offsets.is_empty() {
                        true
                    } else if let Some((_, _, nal_type)) = parse_h264_nal_header(nal_data) {
                        let is_idr_slice = nal_type == 5;
                        (current_is_idr && !is_idr_slice) || is_idr_slice
                    } else {
                        false
                    };
                }
            } else {
                if let (Some(h265_sps), Some(h265_pps)) = (h265_sps, h265_pps) {
                    if let Some((_, nal_unit_type, _, nuh_temporal_id_plus1)) =
                        parse_h265_nal_header(nal_data)
                    {
                        if let Some((
                            first_slice_in_pic,
                            poc_lsb,
                            poc_msb,
                            poc,
                            slice_is_idr,
                            slice_is_reference,
                            slice_type,
                            slice_num_bits_strps,
                            slice_num_delta_pocs,
                            slice_short_term_ref_pic_set_sps_flag,
                            slice_ref_pocs,
                        )) = parse_h265_slice_header(
                            nal_data,
                            h265_sps,
                            h265_pps,
                            nal_unit_type,
                            nuh_temporal_id_plus1,
                            prev_pic_order_cnt_lsb,
                            prev_pic_order_cnt_msb,
                        ) {
                            is_new_frame = if !in_frame || current_slice_offsets.is_empty() {
                                true
                            } else if first_slice_in_pic {
                                true
                            } else if is_irap {
                                true
                            } else {
                                false
                            };

                             if is_new_frame {
                                 if in_frame && !current_au_data.is_empty() {
                                      access_units.push(AccessUnit {
                                          data: current_au_data.clone(),
                                          slice_offsets: current_slice_offsets.clone(),
                                          frame_num: current_frame_num,
                                          pic_order_cnt: current_poc,
                                          is_idr: current_is_idr,
                                          is_reference: current_is_reference,
                                          slice_type: current_slice_type,
                                          num_bits_for_st_ref_pic_set_in_slice: current_num_bits_for_st_ref_pic_set_in_slice,
                                          num_delta_pocs_of_ref_rps_idx: current_num_delta_pocs_of_ref_rps_idx,
                                          short_term_ref_pic_set_sps_flag: current_short_term_ref_pic_set_sps_flag,
                                          ref_pocs: current_ref_pocs.clone(),
                                          adaptive_ref_pic_marking_mode_flag: current_adaptive_ref_pic_marking_mode_flag,
                                          mmco_commands: current_mmco_commands.clone(),
                                      });
                                      current_au_data.clear();
                                      current_slice_offsets.clear();
                                      current_mmco_commands.clear();
                                  }

                                  current_is_idr = slice_is_idr;
                                 current_is_reference = slice_is_reference;
                                current_poc = poc;
                                current_slice_type = slice_type;
                                current_num_bits_for_st_ref_pic_set_in_slice = slice_num_bits_strps;
                                current_num_delta_pocs_of_ref_rps_idx = slice_num_delta_pocs;
                                current_short_term_ref_pic_set_sps_flag = slice_short_term_ref_pic_set_sps_flag;
                                current_ref_pocs = slice_ref_pocs;
                                prev_frame_num += 1;
                                current_frame_num = prev_frame_num;

                                let temporal_id = nuh_temporal_id_plus1 - 1;
                                let is_radl_rasl = nal_unit_type == 22 || nal_unit_type == 23;
                                let is_sub_layer_non_ref =
                                    nal_unit_type < 16 && nal_unit_type % 2 == 0;
                                if temporal_id == 0 && !is_radl_rasl && !is_sub_layer_non_ref {
                                    prev_pic_order_cnt_lsb = poc_lsb;
                                    prev_pic_order_cnt_msb = poc_msb;
                                }

                                in_frame = true;
                            }
                        } else {
                            is_new_frame = if !in_frame || current_slice_offsets.is_empty() {
                                true
                            } else if is_irap {
                                true
                            } else {
                                false
                            };
                        }
                    } else {
                        is_new_frame = !in_frame || current_slice_offsets.is_empty();
                    }
                } else {
                    is_new_frame = if !in_frame || current_slice_offsets.is_empty() {
                        true
                    } else if is_irap {
                        true
                    } else {
                        false
                    };
                }
            }

            if is_new_frame && codec != VideoCodec::H264 {
                if in_frame && !current_au_data.is_empty() {
                                         access_units.push(AccessUnit {
                                             data: current_au_data.clone(),
                                             slice_offsets: current_slice_offsets.clone(),
                                             frame_num: current_frame_num,
                                             pic_order_cnt: current_poc,
                                             is_idr: current_is_idr,
                                             is_reference: current_is_reference,
                                             slice_type: current_slice_type,
                                             num_bits_for_st_ref_pic_set_in_slice: current_num_bits_for_st_ref_pic_set_in_slice,
                                             num_delta_pocs_of_ref_rps_idx: current_num_delta_pocs_of_ref_rps_idx,
                                             short_term_ref_pic_set_sps_flag: current_short_term_ref_pic_set_sps_flag,
                                             ref_pocs: current_ref_pocs.clone(),
                                             adaptive_ref_pic_marking_mode_flag: current_adaptive_ref_pic_marking_mode_flag,
                                             mmco_commands: current_mmco_commands.clone(),
                                         });
                     current_au_data.clear();
                     current_slice_offsets.clear();
                     current_mmco_commands.clear();
                }

                if codec == VideoCodec::H264 && sps.is_none() {
                    if let Some((_, ref_idc, nal_type)) = parse_h264_nal_header(nal_data) {
                        current_is_idr = nal_type == 5;
                        current_is_reference = ref_idc != 0;
                        prev_frame_num += 1;
                        current_frame_num = prev_frame_num;
                        current_poc = [0, 0];
                        current_slice_type = if nal_type == 5 {
                            0
                        } else if ref_idc != 0 {
                            1
                        } else {
                            2
                        };
                    }
                } else if codec == VideoCodec::H265 {
                    if let Some((_, nal_type, _, _)) = parse_h265_nal_header(nal_data) {
                        current_is_idr = nal_type == 19 || nal_type == 20;
                        current_is_reference =
                            (nal_type >= 16 && nal_type <= 23) || nal_type % 2 == 1;
                        prev_frame_num += 1;
                        current_frame_num = prev_frame_num;
                        current_slice_type = if current_is_idr {
                            2
                        } else if current_is_reference {
                            1
                        } else {
                            0
                        };
                    }
                }
                in_frame = true;
            }

            let slice_offset = current_au_data.len();
            current_au_data.extend_from_slice(&[0x00, 0x00, 0x01]);
            current_au_data.extend_from_slice(nal_data);
            current_slice_offsets.push(slice_offset as u32);
        }

        offset = end;
    }

    if in_frame && !current_au_data.is_empty() {
        access_units.push(AccessUnit {
            data: current_au_data,
            slice_offsets: current_slice_offsets,
            frame_num: current_frame_num,
            pic_order_cnt: current_poc,
            is_idr: current_is_idr,
            is_reference: current_is_reference,
            slice_type: current_slice_type,
            num_bits_for_st_ref_pic_set_in_slice: current_num_bits_for_st_ref_pic_set_in_slice,
            num_delta_pocs_of_ref_rps_idx: current_num_delta_pocs_of_ref_rps_idx,
            short_term_ref_pic_set_sps_flag: current_short_term_ref_pic_set_sps_flag,
            ref_pocs: current_ref_pocs,
            adaptive_ref_pic_marking_mode_flag: current_adaptive_ref_pic_marking_mode_flag,
            mmco_commands: current_mmco_commands,
        });
    }

    access_units
}

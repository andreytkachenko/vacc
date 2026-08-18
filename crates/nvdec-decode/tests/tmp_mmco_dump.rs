//! Temporary: dump MMCO ops from the sample. DELETE AFTER USE.
use vk_video_parser::{h264::H264Parser, BitstreamPacket, ParseResult, VideoParser};

const PROJECT_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

#[test]
fn dump_mmco() {
    let data = std::fs::read(format!(
        "{}/assets/bframe_test.h264",
        PROJECT_ROOT
    ))
    .unwrap();
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data);
    let mut pic = 0;
    let mut dumped_pps = false;
    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::Slice { slices, .. }) if !slices.is_empty() => {
                if !dumped_pps {
                    if let (Some(sps), Some(pps)) = (parser.active_sps(), parser.active_pps()) {
                        eprintln!(
                            "SPS: log2_fn={} poc_type={} log2_poc={} fmo={} sep_colour={} max_ref={}",
                            sps.log2_max_frame_num_minus4, sps.pic_order_cnt_type,
                            sps.log2_max_pic_order_cnt_lsb_minus4, sps.frame_mbs_only_flag,
                            sps.separate_colour_plane_flag, sps.max_num_ref_frames
                        );
                        eprintln!(
                            "PPS: entropy={} bf_poc={} nrl0={} nrl1={} wpred={} wbipred={} deblock={} red={}",
                            pps.entropy_coding_mode_flag, pps.bottom_field_pic_order_in_frame_present_flag,
                            pps.num_ref_idx_l0_default_active_minus1, pps.num_ref_idx_l1_default_active_minus1,
                            pps.weighted_pred_flag, pps.weighted_bipred_idc,
                            pps.deblocking_filter_control_present_flag, pps.redundant_pic_cnt_present_flag
                        );
                    }
                    dumped_pps = true;
                }
                if let Some(vk_video_parser::SliceHeader::H264(h)) = &slices[0].slice_header {
                    let ops: Vec<String> = h
                        .dec_ref_pic_marking
                        .iter()
                        .map(|e| format!("op{}(v={})", e.memory_management_control_operation, e.value))
                        .collect();
                    let mod0: Vec<String> = h
                        .ref_pic_list_modification_l0
                        .iter()
                        .map(|e| format!("L0[i={} len={} op={} d={}]", e.index, e.length, e.op, e.difference))
                        .collect();
                    let mod1: Vec<String> = h
                        .ref_pic_list_modification_l1
                        .iter()
                        .map(|e| format!("L1[i={} len={} op={} d={}]", e.index, e.length, e.op, e.difference))
                        .collect();
                    eprintln!(
                        "pic {} fn={} poc_lsb={} st={} ref={} nal={} nref0={} nref1={} mods=[{} | {}] ops=[{}]",
                        pic,
                        h.frame_num,
                        h.pic_order_cnt_lsb,
                        h.slice_type,
                        h.nal_ref_idc,
                        h.nal_unit_type,
                        h.num_ref_idx_l0_active_minus1,
                        h.num_ref_idx_l1_active_minus1,
                        mod0.join(","),
                        mod1.join(","),
                        ops.join(",")
                    );
                }
                pic += 1;
                if pic >= 50 {
                    break;
                }
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            _ => {}
        }
    }
}

use vk_video_parser::{h264::H264Parser, BitstreamPacket, ParseResult, VideoParser};

#[test]
fn dbg() {
    let data = std::fs::read(
        "/home/andrey/workspace/rewrite/vk-video-worktree/nvdev/assets/bframe_test.h264",
    )
    .unwrap();

    // Feed the whole file through the parser and print each slice header.
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data.clone());
    let mut pic = 0usize;
    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::Slice { slices, .. }) => {
                if slices.is_empty() {
                    break;
                }
                if let Some(vk_video_parser::SliceHeader::H264(h)) = &slices[0].slice_header {
                    if pic < 12 {
                        println!("PIC {} nal_ref_idc={} nal_unit_type={} slice_type={} frame_num={} poc_lsb={} nr0={} nr1={}",
                            pic, h.nal_ref_idc, h.nal_unit_type, h.slice_type, h.frame_num,
                            h.pic_order_cnt_lsb, h.num_ref_idx_l0_active_minus1, h.num_ref_idx_l1_active_minus1);
                        println!(
                            "  rplm_l0={:?} rplm_l1={:?}",
                            h.ref_pic_list_modification_l0, h.ref_pic_list_modification_l1
                        );
                        println!("  mmco={:?}", h.dec_ref_pic_marking);
                        println!(
                            "  cabac={} qpd={} header_bits={}",
                            h.cabac_init_idc, h.slice_qp_delta, h.header_bit_size
                        );
                    }
                }
                pic += 1;
                if pic > 12 {
                    break;
                }
            }
            Ok(ParseResult::Nothing) | Ok(ParseResult::EndOfStream) => break,
            _ => {}
        }
    }
}

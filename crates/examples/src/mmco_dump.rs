use vk_video_parser::h264::H264Parser;
use vk_video_parser::{BitstreamPacket, ParseResult, SliceHeader, VideoParser};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "assets/test_baseline.h264".into());
    let data = std::fs::read(&path).unwrap();
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data);
    let mut pic = 0;
    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::Slice { slices, .. }) => {
                if slices.is_empty() { break; }
                for sl in &slices {
                    if let Some(SliceHeader::H264(slh)) = &sl.slice_header {
                        let mmco = if slh.dec_ref_pic_marking.is_empty() {
                            "none".to_string()
                        } else {
                            slh.dec_ref_pic_marking
                                .iter()
                                .map(|e| format!("op{}={}", e.memory_management_control_operation, e.value))
                                .collect::<Vec<_>>()
                                .join(",")
                        };
                        println!(
                            "pic{} fn={} st={} nalref={} nalt={} mmco=[{}] noout={} ltref={}",
                            pic, slh.frame_num, slh.slice_type, slh.nal_ref_idc, slh.nal_unit_type,
                            mmco, slh.no_output_of_prior_pics_flag, slh.long_term_reference_flag
                        );
                    }
                }
                pic += 1;
                if pic >= 30 { break; }
            }
            Ok(_) => {}
            Err(e) => { eprintln!("err: {e}"); break; }
        }
    }
}

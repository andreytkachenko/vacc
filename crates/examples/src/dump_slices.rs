//! Debug: parse all slice headers of an H.264 stream and print MMCO ops,
//! ref-pic-list-modification, frame_num, POC lsb, slice type.
use vk_video_core::codec::VideoCodec;
use vk_video_parser::h264::H264Parser;
use vk_video_parser::{BitstreamPacket, DetectedVideoFormat, VideoParser};

fn main() {
    let file_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/bframe_test.h264".to_string());
    let data = std::fs::read(&file_path).expect("Failed to read file");
    println!("Read {} bytes from {}", data.len(), file_path);

    let mut parser = H264Parser::new();
    let format = DetectedVideoFormat::new(VideoCodec::DecodeH264);
    parser.init(&format).expect("Failed to init parser");

    let packet = BitstreamPacket::new(data);
    let mut pic = 0;
    loop {
        match parser.parse(&packet) {
            Ok(vk_video_parser::ParseResult::ParameterSet { .. }) => {}
            Ok(vk_video_parser::ParseResult::Slice { slices, .. }) => {
                for s in &slices {
                    if let Some(vk_video_parser::SliceHeader::H264(slh)) = &s.slice_header {
                        let sl_type = ["P", "B", "I", "SP", "SI"][slh.slice_type as usize];
                        print!(
                            "pic{} fn={} poc_lsb={} type={} ref_idc={} mmco=[",
                            pic, slh.frame_num, slh.pic_order_cnt_lsb, sl_type, slh.nal_ref_idc
                        );
                        for (i, m) in slh.dec_ref_pic_marking.iter().enumerate() {
                            if i > 0 {
                                print!(", ");
                            }
                            print!("{}({})", m.memory_management_control_operation, m.value);
                        }
                        print!("] listmod_l0=[");
                        for (i, m) in slh.ref_pic_list_modification_l0.iter().enumerate() {
                            if i > 0 {
                                print!(", ");
                            }
                            print!("{}({})", m.op, m.difference);
                        }
                        print!("] listmod_l1=[");
                        for (i, m) in slh.ref_pic_list_modification_l1.iter().enumerate() {
                            if i > 0 {
                                print!(", ");
                            }
                            print!("{}({})", m.op, m.difference);
                        }
                        println!(
                            "] nref0={} nref1={}",
                            slh.num_ref_idx_l0_active_minus1, slh.num_ref_idx_l1_active_minus1
                        );
                    }
                }
                pic += slices.len();
            }
            Ok(vk_video_parser::ParseResult::Nothing)
            | Ok(vk_video_parser::ParseResult::EndOfStream) => break,
            Err(e) => {
                eprintln!("parse error: {}", e);
                break;
            }
        }
    }
    println!("total pics: {}", pic);
}

use vk_video_parser::h264::H264Parser;
use vk_video_parser::{BitstreamPacket, ParseResult, VideoParser};

fn main() {
    let data = std::fs::read("assets/bframe_test.h264").unwrap();
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data);
    let mut pic = 0;
    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::Slice { slices, .. }) => {
                if slices.is_empty() {
                    break;
                }
                if pic == 4 {
                    let nal = &slices[0].nal_data;
                    println!("pic4 nal: {:02x?}", &nal[..nal.len().min(40)]);
                    // EPB removal: 00 00 03 -> 00 00
                    let mut rbsp = Vec::new();
                    let mut i = 1; // skip nal header
                    while i < nal.len() {
                        if i + 2 < nal.len() && nal[i] == 0 && nal[i + 1] == 0 && nal[i + 2] == 3 {
                            rbsp.push(0);
                            rbsp.push(0);
                            i += 3;
                        } else {
                            rbsp.push(nal[i]);
                            i += 1;
                        }
                    }
                    println!("pic4 rbsp: {:02x?}", &rbsp[..rbsp.len().min(40)]);
                    break;
                }
                pic += 1;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("err: {e}");
                break;
            }
        }
    }
}

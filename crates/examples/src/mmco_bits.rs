use vk_video_parser::h264::H264Parser;
use vk_video_parser::{BitstreamPacket, ParseResult, SliceHeader, VideoParser};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/test_baseline.h264".into());
    let want_pic: usize = std::env::args()
        .nth(2)
        .map(|s| s.parse().unwrap())
        .unwrap_or(2);
    let data = std::fs::read(&path).unwrap();
    let mut parser = H264Parser::new();
    let packet = BitstreamPacket::new(data);
    let mut pic = 0;
    loop {
        match parser.parse(&packet) {
            Ok(ParseResult::Slice { slices, .. }) => {
                if slices.is_empty() {
                    break;
                }
                if pic == want_pic {
                    for sl in &slices {
                        let nal = &sl.nal_data;
                        println!(
                            "pic{} nal ({} bytes): {:02x?}",
                            pic,
                            nal.len(),
                            &nal[..nal.len().min(48)]
                        );
                        // EPB removal
                        let mut rbsp = Vec::new();
                        let mut i = 1;
                        while i < nal.len() {
                            if i + 2 < nal.len()
                                && nal[i] == 0
                                && nal[i + 1] == 0
                                && nal[i + 2] == 3
                            {
                                rbsp.push(0);
                                rbsp.push(0);
                                i += 3;
                            } else {
                                rbsp.push(nal[i]);
                                i += 1;
                            }
                        }
                        // print as bits
                        let mut bits = String::new();
                        for b in &rbsp {
                            for k in (0..8).rev() {
                                bits.push(if b >> k & 1 == 1 { '1' } else { '0' });
                            }
                        }
                        println!("rbsp bits ({} bits):", bits.len());
                        for chunk in bits.as_str().as_bytes().chunks(48) {
                            println!("  {}", std::str::from_utf8(chunk).unwrap());
                        }
                        if let Some(SliceHeader::H264(slh)) = &sl.slice_header {
                            println!(
                                "parsed: fn={} st={} mmco=[{}]",
                                slh.frame_num,
                                slh.slice_type,
                                slh.dec_ref_pic_marking
                                    .iter()
                                    .map(|e| format!(
                                        "op{}={}",
                                        e.memory_management_control_operation, e.value
                                    ))
                                    .collect::<Vec<_>>()
                                    .join(",")
                            );
                        }
                    }
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

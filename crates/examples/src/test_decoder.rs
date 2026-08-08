use std::io::Write;

use vk_video_parser::{
    bitstream::BitstreamPacket, h264::H264Parser,
    DetectedVideoFormat, ParseResult, VideoParser,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bitstream_path = if args.len() >= 2 { &args[1] } else { "born_trailer.h264" };
    let max_frames: usize = if args.len() >= 3 { args[2].parse().unwrap_or(3) } else { 3 };

    let data = std::fs::read(bitstream_path).expect("Failed to read file");

    // Parse SPS to check profile_idc
    let mut parser = H264Parser::new();
    parser
        .init(&DetectedVideoFormat::new(
            vk_video_core::codec::VideoCodec::DecodeH264,
        ))
        .expect("Failed to init parser");

    let packet = BitstreamPacket::new(data.clone());
    if let Ok(ParseResult::ParameterSet { sps, pps, .. }) = parser.parse(&packet) {
        if let Some(s) = sps {
            if let Some(sps) = s.downcast_ref::<vk_video_core::picture::H264Sps>() {
                println!("SPS profile_idc: {}", sps.profile_idc);
                println!("SPS level_idc: {}", sps.level_idc);
                println!("SPS chroma_format_idc: {}", sps.chroma_format_idc);
                println!("SPS bit_depth_luma_minus8: {}", sps.bit_depth_luma_minus8);
            }
        }
    }

    let mut decoder = match vk_video_vulkan::VideoDecoder::new(data, max_frames) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to create decoder: {}", e);
            std::process::exit(1);
        }
    };

    let decoded_frames = match decoder.decode_all(max_frames) {
        Ok(frames) => frames,
        Err(e) => {
            eprintln!("Decode failed: {}", e);
            std::process::exit(1);
        }
    };

    println!("Decoded {} frames", decoded_frames.len());

    for (i, frame) in decoded_frames.iter().enumerate() {
        let filename = format!("test_vulkan_frame_{:02}.yuv", i);
        let mut file = std::fs::File::create(&filename).expect("Failed to create frame file");
        
        let width = frame.coded_width as usize;
        let height = frame.coded_height as usize;
        let uv_width = width / 2;
        let uv_height = height / 2;

        file.write_all(&frame.pixels.y_plane[..width * height]).unwrap();
        file.write_all(&frame.pixels.u_plane[..uv_width * uv_height]).unwrap();
        file.write_all(&frame.pixels.v_plane[..uv_width * uv_height]).unwrap();

        println!("Wrote {} (POC={}, is_idr={}, is_ref={})", 
                 filename, frame.poc, frame.is_idr, frame.is_reference);
    }
}

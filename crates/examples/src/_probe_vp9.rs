use vk_video_parser::vp9::Vp9Parser;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).unwrap();
    let is_ivf = data.len() >= 32 && &data[0..4] == b"DKIF";
    let mut off = if is_ivf { 32 } else { 0 };
    let mut parser = Vp9Parser::new();
    let mut i = 0usize;
    let mut stats = [0u32; 8]; // show0, show_existing, intra_only, key, inter, seg, tiles>1, mcomp!=1
    while off + 12 <= data.len() && i < 300 {
        let size = u32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize;
        if off + 12 + size > data.len() { break; }
        let payload = &data[off+12..off+12+size];
        match parser.parse_frame(payload) {
            Ok(fd) => {
                if fd.show_existing_frame { stats[1] += 1; }
                if fd.picture_info.flags.show_frame == 0 { stats[0] += 1; }
                if fd.frame_is_intra { if fd.picture_info.frame_type == vk_video_core::picture::Vp9FrameType::Key { stats[3] += 1; } else { stats[2] += 1; } } else { stats[4] += 1; }
                if fd.picture_info.flags.segmentation_enabled != 0 { stats[5] += 1; }
                if fd.num_tiles > 1 { stats[6] += 1; }
                if fd.picture_info.interpolation_filter as u32 != 1 { stats[7] += 1; }
                if i < 3 || fd.picture_info.flags.show_frame == 0 || fd.show_existing_frame {
                    println!("f{}: key={} intra_only={} show={} existing={} refs=[{},{},{}] refresh={:#04x} mcomp={:?} tiles={} w={}x{}",
                        i, fd.picture_info.frame_type == vk_video_core::picture::Vp9FrameType::Key,
                        fd.picture_info.flags.intra_only, fd.picture_info.flags.show_frame,
                        fd.show_existing_frame, fd.ref_frame_idx[0], fd.ref_frame_idx[1], fd.ref_frame_idx[2],
                        fd.picture_info.refresh_frame_flags, fd.picture_info.interpolation_filter,
                        fd.num_tiles, fd.frame_width, fd.frame_height);
                }
            }
            Err(e) => { println!("f{}: PARSE ERR {}", i, e); break; }
        }
        off += 12 + size;
        i += 1;
    }
    println!("total={} show0={} existing={} intra_only={} key={} inter={} seg={} tiles>1={} mcomp!=8tap_smooth={}",
        i, stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6], stats[7]);
}

//! Regression test: the VP9 parser must carry the key frame's color config
//! (bit depth, subsampling) over to inter frames, which do not re-signal it.
//! Without this, backends see 8-bit for every inter frame of a 10/12-bit
//! stream and misconfigure the decoder.
use vacc_core::picture::Vp9FrameType;
use vacc_parser::vp9::Vp9Parser;

fn ivf_packets(data: &[u8]) -> Vec<&[u8]> {
    assert_eq!(&data[0..4], b"DKIF", "expected IVF container");
    let mut out = Vec::new();
    let mut off = 32usize;
    while off + 12 <= data.len() {
        let size = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        if size == 0 || off + 12 + size > data.len() {
            break;
        }
        out.push(&data[off + 12..off + 12 + size]);
        off += 12 + size;
    }
    out
}

#[test]
fn inter_frames_carry_keyframe_color_config() {
    for (name, bit_depth, data) in [
        ("vp9_profile1.ivf", 10u8, include_bytes!("../../../assets/samples/vp9_profile1.ivf") as &[u8]), // profile 2, 10-bit
        ("vp9_profile2.ivf", 12u8, include_bytes!("../../../assets/samples/vp9_profile2.ivf") as &[u8]), // profile 2, 12-bit
    ] {
        let packets = ivf_packets(data);
        assert!(packets.len() >= 10, "{name}: too few packets");

        let mut parser = Vp9Parser::new();
        let mut checked_inter = 0usize;
        for (i, pkt) in packets.iter().take(10).enumerate() {
            let fd = parser.parse_frame(pkt).unwrap_or_else(|e| {
                panic!("{name}: parse frame {i}: {e}")
            });
            if fd.picture_info.frame_type == Vp9FrameType::Key {
                assert_eq!(
                    fd.color_config.bit_depth, bit_depth,
                    "{name}: key frame {i} bit depth"
                );
            } else {
                assert_eq!(
                    fd.color_config.bit_depth, bit_depth,
                    "{name}: inter frame {i} must carry the key frame's bit depth"
                );
                assert_eq!(fd.color_config.subsampling_x, 1);
                assert_eq!(fd.color_config.subsampling_y, 1);
                checked_inter += 1;
            }
        }
        assert!(checked_inter >= 5, "{name}: expected inter frames in first 10");
    }
}

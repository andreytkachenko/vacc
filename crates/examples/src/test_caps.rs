fn main() {
    use nvdec_decode::ffi::{cudaVideoChromaFormat, cudaVideoCodec};
    use nvdec_decode::{init_nvdec, query_decoder_caps};

    init_nvdec().unwrap();
    let caps = query_decoder_caps(
        cudaVideoCodec::cudaVideoCodec_H264,
        cudaVideoChromaFormat::cudaVideoChromaFormat_420,
        0, // 8-bit
    )
    .unwrap();

    println!("H.264 4:2:0 8-bit decoder caps:");
    println!("  bIsSupported: {}", caps.bIsSupported);
    println!("  nNumNVDECs: {}", caps.nNumNVDECs);
    println!("  nOutputFormatMask: 0x{:08x}", caps.nOutputFormatMask);
    println!("  nMaxWidth: {}", caps.nMaxWidth);
    println!("  nMaxHeight: {}", caps.nMaxHeight);

    // Check which formats are supported
    println!("\nSupported output formats:");
    if caps.nOutputFormatMask & (1 << 0) != 0 {
        println!("  Unknown (0)");
    }
    if caps.nOutputFormatMask & (1 << 1) != 0 {
        println!("  NV12 (1)");
    }
    if caps.nOutputFormatMask & (1 << 2) != 0 {
        println!("  P016 (2)");
    }
    if caps.nOutputFormatMask & (1 << 3) != 0 {
        println!("  P016_BIG (3)");
    }
    if caps.nOutputFormatMask & (1 << 4) != 0 {
        println!("  YUV444 (4)");
    }
    if caps.nOutputFormatMask & (1 << 5) != 0 {
        println!("  YUV444_8Bit (5)");
    }
    if caps.nOutputFormatMask & (1 << 6) != 0 {
        println!("  YUV420 (6)");
    }
}

# vacc-libva

Safe Rust bindings over libva, vendored from [cros-libva](https://github.com/chromeos/cros-libva)
(BSD-3-Clause, The Chromium OS Authors) at upstream master (v0.0.13 generation), plus two
local commits required by the vacc VAAPI decoders targeting Intel iHD:

0. **`Display::handle` made public** — exposes the raw `VADisplay` for direct libva FFI use.

1. **HEVC extension buffer wrappers** — `PictureParameterBufferHEVCExtension` /
   `SliceParameterBufferHEVCExtension` over the `*Extension` FFI structs (base + rext, scc
   zeroed), plus `PictureParameter::HEVCExtension` / `SliceParameter::HEVCExtension` enum
   variants. For REXT/SCC profiles (`sps profile_idc >= 4`) the driver expects the full
   extension-size buffer attached as a plain picture/slice parameter type, matching FFmpeg
   `vaapi_hevc.c`.
2. **Zero-initialized decode parameter buffers** — `VAPictureParameterBufferH264`,
   `VASliceParameterBufferH264`, `VASliceParameterBufferHEVC`, `VASliceParameterBufferHEVCRext`
   and `VADecPictureParameterBufferVP9` constructors zero-initialize instead of using struct
   literals: Rust struct literals leave implicit C padding bytes uninitialized, and iHD
   compares the whole mapped buffer byte-wise against FFmpeg's memset-zeroed buffers.

The crate is published to crates.io as `vacc-libva`; its lib target keeps the upstream name
`libva`, so consumers write `use libva::...` exactly as with cros-libva.

## License

BSD-3-Clause (upstream cros-libva). See `LICENSE`.

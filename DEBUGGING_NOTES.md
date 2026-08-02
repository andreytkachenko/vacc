# Vulkan H.264 Decode Debugging Notes

## Problem
Vulkan hardware-accelerated H.264 decoding produces all-gray pixels (Y=128) instead of correct video content.

## Bisecting Analysis

### Stage 1: Bitstream Parsing ✓
- SPS/PPS extraction works correctly
- Verified: profile_idc=66 (Baseline), level_idc=41, width=1920, height=816
- Verified: pic_order_cnt_type=0, max_num_ref_frames=1

### Stage 2: Access Unit Extraction ✓
- **ROOT CAUSE FOUND (Partially)**: Slice offset must point to START CODE, not after it
  - Before fix: offset pointed after start code → all-black (Y=0)
  - After fix: offset points to start code → gray (Y=128)
- Bitstream contains ONLY slice NALs (no SPS/PPS)
- SPS/PPS provided via session parameters (out-of-band)

### Stage 3: Session Creation ✓
- Session created with correct profile chain
- SPS/PPS added via `VideoDecodeH264SessionParametersAddInfoKHR`
- `pStdHeaderVersion` set correctly

### Stage 4: Decode Command Recording ✓
- Picture info: frame_num=0, poc=[0,0], is_intra=1, is_reference=1
- DPB slot: slot_index=-1 in begin coding, slot_index=0 in decode
- Buffer barrier: HOST_WRITE → VIDEO_DECODE_READ
- Image barrier: UNDEFINED → VIDEO_DECODE_DPB

### Stage 5: Readback ✓
- Readback verified working (compared with existing reference output)

## Comparison with Reference Implementation

### Reference (NVIDIA Vulkan-Video-Samples)
- Works correctly (produces correct YUV output)
- Has validation errors:
  1. `srcBufferRange` not aligned to 256 bytes
  2. `pSetupReferenceSlot->pNext` missing DPB slot info
  3. DPB slot index 0 not active in begin coding

### Our Implementation
- More spec-compliant (no validation errors with DPB slot info)
- Produces gray output (Y=128)

## Key Differences Investigated

1. **Bitstream content**: Tried with/without SPS/PPS - same result
2. **Slice offset**: Points to start code (correct)
3. **DPB slot info**: Tried with/without chaining - same result
4. **IdrPicFlag**: Tried with/without setting - same result
5. **Image barrier aspect**: Tried COLOR vs PLANE_0/PLANE_1 - same result
6. **Buffer range alignment**: Already aligned to 256 bytes

## Bisecting Results (Updated)

| Configuration | Result | Conclusion |
|--------------|--------|------------|
| Slice offset after start code | Y=0 (black) | WRONG - decoder can't find slices |
| Slice offset at start code | Y=128 (gray) | CORRECT - decoder finds slices but fails to decode |
| SPS/PPS in session params only | Y=128 (gray) | Session params ARE being used |
| SPS/PPS in bitstream only | Y=0 (black) | Decoder needs session params |
| No SPS/PPS anywhere | Y=0 (black) | Confirms session params required |
| IdrPicFlag=0 vs 1 | Same (gray) | Not the issue |
| max_dpb_slots+1 | Same (gray) | Not the issue |
| DPB slot chained/not chained | Same (gray) | Not the issue |

## Current State

```
Vulkan decode: Y=128 (all gray) - decoder runs but produces neutral gray
FFmpeg ref:    Y min=18, max=232, avg=74.4 - correct video content
Reference:     Y min=18, max=232, avg=74.4 - matches FFmpeg exactly
```

## Key Findings

1. **Slice offset fix is CORRECT** - changed from black to gray
2. **Session parameters ARE being used** - removing SPS/PPS changes output from gray to black
3. **SPS/PPS conversion is PARTIALLY working** - gray instead of black means decoder sees params
4. **Gray (Y=128) = decoder initializes but can't decode slice data**

## Root Cause Analysis

The gray output indicates:
- Decoder successfully initialized with session parameters
- Decode command accepted without errors
- BUT decoder can't extract valid pixel data from bitstream

This suggests a MISMATCH between:
- SPS/PPS values in session parameters
- Picture info in decode command  
- What decoder expects from bitstream

## Remaining Hypotheses

1. **SPS/PPS conversion bug**: Some field in StdVideoH264SequenceParameterSet is wrong
2. **Picture info bug**: frame_num/Poc/slice IDs don't match bitstream
3. **Driver-specific behavior**: NVIDIA driver expects specific patterns
4. **Session params lifecycle**: Reference adds SPS/PPS separately, we add together

## Next Steps

1. **Compare SPS bytes**: Dump exact StdVideoH264SequenceParameterSet from reference vs ours
2. **Try separate SPS/PPS addition**: Add SPS first, then update with PPS (like reference)
3. **Parse frame_num/Poc from bitstream**: Instead of hardcoding 0
4. **Check PPS conversion**: Verify all fields are correct

## Files Modified

- `crates/examples/src/vulkan_decode.rs`: Main decode example with bisecting debug output
- `crates/vk-video-parser/src/h264.rs`: Added debug output for SPS parsing
- `DEBUGGING_NOTES.md`: This file

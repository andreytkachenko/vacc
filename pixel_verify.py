#!/usr/bin/env python3
"""
Pixel verification tool for comparing video decoder outputs.

Usage:
    python3 pixel_verify.py <input_video> [--frames N] [--codec hw|sw]
    
Examples:
    # Compare Vulkan HW vs FFmpeg SW decode
    python3 pixel_verify.py assets/big_buck_bunney.h265 --frames 10
    
    # Compare your decoder output (YUV files) vs FFmpeg reference
    python3 pixel_verify.py --yuv-dir ./my_decoder_output --ref-ffmpeg assets/big_buck_bunney.h265 --frames 10
"""

import subprocess
import sys
import os
import struct
import math
import tempfile
import argparse
from pathlib import Path


def run_ffmpeg(args):
    """Run ffmpeg command and return output."""
    cmd = ["ffmpeg", "-v", "error"] + args
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode, result.stdout, result.stderr


def decode_frames_ffmpeg(input_file, frames, hwaccel=None, prefix=None):
    """Decode frames using ffmpeg and return list of YUV frame data."""
    tmpdir = tempfile.mkdtemp(prefix=prefix or "pixel_verify_")
    
    cmd = ["ffmpeg", "-v", "error"]
    if hwaccel:
        cmd.extend(["-hwaccel", hwaccel])
    cmd.extend([
        "-i", input_file,
        "-frames:v", str(frames),
        "-pix_fmt", "yuv420p",
        "-f", "rawvideo",
        "pipe:"
    ])
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    
    # Parse video dimensions from ffprobe
    probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0", input_file]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    width, height = map(int, probe_result.stdout.strip().split(","))
    
    frame_size = width * height * 3 // 2
    frames_data = []
    for i in range(frames):
        start = i * frame_size
        end = start + frame_size
        frame_data = result.stdout[start:end]
        frames_data.append({
            "index": i,
            "data": frame_data,
            "width": width,
            "height": height
        })
    
    return frames_data, width, height


def yuv_plane(frame_data, width, height, plane):
    """Extract a YUV plane from frame data. plane: 0=Y, 1=U, 2=V."""
    if plane == 0:  # Y plane
        return frame_data[:width * height]
    half_w = width // 2
    half_h = height // 2
    uv_size = half_w * half_h
    if plane == 1:  # U plane
        return frame_data[width * height:width * height + uv_size]
    else:  # V plane
        return frame_data[width * height + uv_size:]


def calculate_mse(plane_a, plane_b):
    """Calculate Mean Squared Error between two planes."""
    if len(plane_a) != len(plane_b):
        raise ValueError(f"Plane size mismatch: {len(plane_a)} vs {len(plane_b)}")
    
    mse = 0.0
    for a, b in zip(plane_a, plane_b):
        diff = int.from_bytes(bytes([a]), 'big') - int.from_bytes(bytes([b]), 'big')
        mse += diff * diff
    
    return mse / len(plane_a)


def calculate_psnr(mse):
    """Calculate PSNR from MSE."""
    if mse == 0:
        return float('inf')
    max_val = 255.0
    return 10.0 * math.log10((max_val * max_val) / mse)


def count_diff_pixels(plane_a, plane_b, threshold=0):
    """Count pixels that differ by more than threshold."""
    count = 0
    total = len(plane_a)
    max_diff = 0
    for a, b in zip(plane_a, plane_b):
        diff = abs(int.from_bytes(bytes([a]), 'big') - int.from_bytes(bytes([b]), 'big'))
        if diff > threshold:
            count += 1
        max_diff = max(max_diff, diff)
    return count, total, max_diff


def compare_frames(frame_a, frame_b, label_a="HW", label_b="SW"):
    """Compare two YUV frames and return detailed statistics."""
    width = frame_a["width"]
    height = frame_a["height"]
    
    stats = {
        "index": frame_a["index"],
        "width": width,
        "height": height,
    }
    
    for plane_name, plane_idx in [("Y", 0), ("U", 1), ("V", 2)]:
        plane_a = yuv_plane(frame_a["data"], width, height, plane_idx)
        plane_b = yuv_plane(frame_b["data"], width, height, plane_idx)
        
        mse = calculate_mse(plane_a, plane_b)
        psnr = calculate_psnr(mse)
        diff_count, total, max_diff = count_diff_pixels(plane_a, plane_b)
        
        stats[f"{plane_name}_mse"] = mse
        stats[f"{plane_name}_psnr"] = psnr
        stats[f"{plane_name}_diff_pixels"] = diff_count
        stats[f"{plane_name}_total_pixels"] = total
        stats[f"{plane_name}_max_diff"] = max_diff
    
    # Overall PSNR (Y plane weighted)
    stats["overall_psnr"] = stats["Y_psnr"]
    stats["identical"] = stats["Y_mse"] == 0 and stats["U_mse"] == 0 and stats["V_mse"] == 0
    
    return stats


def print_comparison(stats):
    """Print comparison statistics in a readable format."""
    print(f"\nFrame {stats['index']:3d} ({stats['width']}x{stats['height']}):")
    
    if stats["identical"]:
        print("  ✓ IDENTICAL - No pixel differences")
        return
    
    print(f"  Overall PSNR: {stats['overall_psnr']:.2f} dB")
    
    for plane in ["Y", "U", "V"]:
        mse = stats[f"{plane}_mse"]
        psnr = stats[f"{plane}_psnr"]
        diff = stats[f"{plane}_diff_pixels"]
        total = stats[f"{plane}_total_pixels"]
        max_d = stats[f"{plane}_max_diff"]
        pct = (diff / total * 100) if total > 0 else 0
        
        print(f"  {plane}: MSE={mse:.4f}, PSNR={psnr:.2f} dB, "
              f"diff={diff}/{total} ({pct:.2f}%), max_diff={max_d}")


def main():
    parser = argparse.ArgumentParser(description="Pixel verification for video decoders")
    parser.add_argument("input", nargs="?", help="Input video file")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to decode")
    parser.add_argument("--hwaccel", choices=["vulkan", "vaapi", "cuda", "qsv"],
                        default="vulkan", help="Hardware acceleration method")
    parser.add_argument("--yuv-dir", help="Directory with YUV files to compare")
    parser.add_argument("--ref-ffmpeg", help="Reference video for FFmpeg decode")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.yuv_dir and args.ref_ffmpeg:
        # Compare custom YUV files vs FFmpeg reference
        print("=== Custom YUV vs FFmpeg Reference ===")
        
        # Get FFmpeg reference frames
        ref_frames, width, height = decode_frames_ffmpeg(
            args.ref_ffmpeg, args.frames, hwaccel=None, prefix="ref_"
        )
        
        # Load custom YUV files
        yuv_dir = Path(args.yuv_dir)
        custom_frames = []
        frame_size = width * height * 3 // 2
        
        for i in range(args.frames):
            # Try common naming patterns
            for pattern in [
                f"frame_{i}.yuv", f"frame_{i:04d}.yuv", f"frame_{i:02d}.yuv",
                f"{i}.yuv", f"{i:04d}.yuv", f"{i:02d}.yuv", f"frame{i}.yuv",
                f"frame{i:04d}.yuv", f"frame{i:02d}.yuv",
            ]:
                yuv_file = yuv_dir / pattern
                if yuv_file.exists():
                    with open(yuv_file, "rb") as f:
                        data = f.read()
                    if len(data) == frame_size:
                        custom_frames.append({
                            "index": i,
                            "data": data,
                            "width": width,
                            "height": height
                        })
                        break
            else:
                print(f"Warning: Frame {i} not found in {yuv_dir}")
        
        if not custom_frames:
            print("Error: No YUV frames found")
            sys.exit(1)
        
        print(f"Comparing {len(custom_frames)} frames...")
        
        results = []
        for i, (custom, ref) in enumerate(zip(custom_frames, ref_frames)):
            stats = compare_frames(custom, ref, label_a="Custom", label_b="FFmpeg SW")
            results.append(stats)
            print_comparison(stats)
        
        print_summary(results)
        
    elif args.input:
        # Compare HW vs SW decode
        print(f"=== Pixel Verification: {args.hwaccel.upper()} HW vs FFmpeg SW ===")
        print(f"Input: {args.input}")
        print(f"Frames: {args.frames}")
        
        # Decode with hardware acceleration
        try:
            hw_frames, width, height = decode_frames_ffmpeg(
                args.input, args.frames, hwaccel=args.hwaccel, prefix="hw_"
            )
            print(f"HW decode: {len(hw_frames)} frames ({width}x{height})")
        except RuntimeError as e:
            print(f"HW decode failed: {e}")
            sys.exit(1)
        
        # Decode with software
        sw_frames, _, _ = decode_frames_ffmpeg(
            args.input, args.frames, hwaccel=None, prefix="sw_"
        )
        print(f"SW decode: {len(sw_frames)} frames")
        
        # Compare
        print("\n--- Comparison Results ---")
        results = []
        for hw, sw in zip(hw_frames, sw_frames):
            stats = compare_frames(hw, sw, label_a="HW", label_b="SW")
            results.append(stats)
            print_comparison(stats)
        
        print_summary(results)
    else:
        parser.print_help()


def print_summary(results):
    """Print summary statistics."""
    identical = sum(1 for r in results if r["identical"])
    total = len(results)
    
    print(f"\n=== Summary ===")
    print(f"Frames tested: {total}")
    print(f"Identical frames: {identical}/{total}")
    
    if identical == total:
        print("✓ ALL FRAMES MATCH - Decoder output is pixel-perfect!")
    else:
        print(f"⚠ {total - identical} frame(s) have differences")
        
        # Average PSNR for non-identical frames
        psnrs = [r["overall_psnr"] for r in results if not r["identical"] and r["overall_psnr"] != float('inf')]
        if psnrs:
            avg_psnr = sum(psnrs) / len(psnrs)
            min_psnr = min(psnrs)
            max_psnr = max(psnrs)
            print(f"PSNR range: {min_psnr:.2f} - {max_psnr:.2f} dB (avg: {avg_psnr:.2f})")


if __name__ == "__main__":
    main()

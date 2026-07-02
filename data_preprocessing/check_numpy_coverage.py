#!/usr/bin/env python3
"""
check_numpy_coverage.py — Analyze MediaPipe hand detection coverage (non-zero rate) from .npy files.
Usage:
  python data_preprocessing/check_numpy_coverage.py <path_to_npy_file_or_directory>
"""

import sys
from pathlib import Path
import numpy as np

# Set stdout/stderr encoding to UTF-8 to prevent encoding errors on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Keypoint layout slices (matching video_npy_converter.py)
POSE_SL  = slice(0,   99)
LHAND_SL = slice(99,  162)
RHAND_SL = slice(162, 225)

def analyze_single_file(file_path: Path) -> dict:
    try:
        data = np.load(str(file_path))
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return None

    if len(data.shape) != 2 or data.shape[1] != 225:
        print(f"Warning: {file_path.name} has unexpected shape {data.shape} (expected T x 225)")
        return None

    T = data.shape[0]
    if T == 0:
        return {
            "filename": file_path.name,
            "total_frames": 0,
            "left_cov": 0.0,
            "right_cov": 0.0,
            "either_cov": 0.0,
            "both_cov": 0.0
        }

    # Slice left and right hand keypoints
    lh_kp = data[:, LHAND_SL]
    rh_kp = data[:, RHAND_SL]

    # A frame is "detected" if it is NOT all zeros
    lh_detected = ~np.all(lh_kp == 0.0, axis=1)
    rh_detected = ~np.all(rh_kp == 0.0, axis=1)

    lh_count = int(np.sum(lh_detected))
    rh_count = int(np.sum(rh_detected))
    either_count = int(np.sum(lh_detected | rh_detected))
    both_count = int(np.sum(lh_detected & rh_detected))

    return {
        "filename": file_path.name,
        "total_frames": T,
        "left_count": lh_count,
        "left_cov": lh_count / T,
        "right_count": rh_count,
        "right_cov": rh_count / T,
        "either_count": either_count,
        "either_cov": either_count / T,
        "both_count": both_count,
        "both_cov": both_count / T
    }

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python data_preprocessing/check_numpy_coverage.py <path_to_npy_file_or_directory>")
        sys.exit(1)

    input_path = Path(sys.argv[1]).resolve()
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        sys.exit(1)

    if input_path.is_file():
        if input_path.suffix.lower() != ".npy":
            print(f"Error: File is not a .npy file: {input_path.name}")
            sys.exit(1)
        res = analyze_single_file(input_path)
        if res:
            print("=" * 60)
            print(f"File Name: {res['filename']}")
            print("-" * 60)
            print(f"Total Frames (T): {res['total_frames']}")
            print(f"Left Hand Detection Rate:  {res['left_count']:4d} / {res['total_frames']} ({res['left_cov']:.2%})")
            print(f"Right Hand Detection Rate: {res['right_count']:4d} / {res['total_frames']} ({res['right_cov']:.2%})")
            print(f"Either Hand Detected:      {res['either_count']:4d} / {res['total_frames']} ({res['either_cov']:.2%})")
            print(f"Both Hands Detected:        {res['both_count']:4d} / {res['total_frames']} ({res['both_cov']:.2%})")
            print("=" * 60)

    elif input_path.is_dir():
        npy_files = sorted(input_path.glob("*.npy"))
        if not npy_files:
            print(f"No .npy files found in directory: {input_path}")
            sys.exit(0)

        print(f"Analyzing directory: {input_path.name} (found {len(npy_files)} .npy files)...")
        results = []
        for f in npy_files:
            res = analyze_single_file(f)
            if res:
                results.append(res)

        if not results:
            print("Failed to successfully analyze any numpy files.")
            sys.exit(0)

        # Calculate averages
        avg_left = np.mean([r["left_cov"] for r in results])
        avg_right = np.mean([r["right_cov"] for r in results])
        avg_either = np.mean([r["either_cov"] for r in results])
        avg_both = np.mean([r["both_cov"] for r in results])
        total_frames = sum([r["total_frames"] for r in results])

        print("\n" + "=" * 60)
        print(f"[Overall Dataset Hand Detection Coverage Summary]")
        print("-" * 60)
        print(f"Total Files Analyzed:       {len(results)}")
        print(f"Total Accumulated Frames:   {total_frames}")
        print(f"Average Left Hand Coverage:  {avg_left:.2%}")
        print(f"Average Right Hand Coverage: {avg_right:.2%}")
        print(f"Average Either Hand Coverage:{avg_either:.2%}")
        print(f"Average Both Hands Coverage: {avg_both:.2%}")
        print("=" * 60)
        
        # Show top 5 worst files (lowest either hand coverage) to help identify bad videos
        print("\n[Top 5 Worst Files with Lowest Coverage] (Recommended for testing on HaMeR Colab)")
        worst_files = sorted(results, key=lambda x: x["either_cov"])[:5]
        for idx, w in enumerate(worst_files, 1):
            print(f"  {idx}. {w['filename']} (Total Frames: {w['total_frames']} | Left: {w['left_cov']:.1%} | Right: {w['right_cov']:.1%})")

if __name__ == "__main__":
    main()

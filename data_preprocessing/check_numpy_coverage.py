#!/usr/bin/env python3
"""
check_numpy_coverage.py — Analyze MediaPipe hand detection coverage (non-zero rate) from .npy files.

Usage:
  python check_numpy_coverage.py <path_to_npy_file_or_directory> [--workers N] [--threshold 0.7] [--export_excel path/to/output.xlsx]

Examples:
  1. Analyze a single file:
     python check_numpy_coverage.py wlasl_numpy/23951.npy

  2. Analyze a directory and print all files (default threshold is 1.0):
     python check_numpy_coverage.py videos_numpy/

  3. Set a custom threshold (e.g., 0.6) and export the results to an Excel file:
     python check_numpy_coverage.py videos_numpy/ --threshold 0.6 --export_excel all_coverage.xlsx
"""

import sys
import argparse
import heapq
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd

# Set stdout/stderr encoding to UTF-8 to prevent encoding errors on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Keypoint layout slices (matching video_npy_converter.py)
# Pose: 0 to 99 (33 * 3)
# Face: 99 to 1533 (478 * 3)
# Left Hand: 1533 to 1596 (21 * 3)
# Right Hand: 1596 to 1659 (21 * 3)
LHAND_SL = slice(1533, 1596)
RHAND_SL = slice(1596, 1659)


def analyze_single_file(file_path: Path) -> Optional[dict]:
    """Load a single .npy file and compute hand-detection coverage stats."""
    try:
        data = np.load(str(file_path))
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return None

    if data.ndim != 2 or data.shape[1] != 1659:
        print(f"Warning: {file_path.name} has unexpected shape {data.shape} (expected T x 1659)")
        return None

    T = data.shape[0]
    if T == 0:
        return {
            "filename": file_path.name,
            "total_frames": 0,
            "left_count": 0, "left_cov": 0.0,
            "right_count": 0, "right_cov": 0.0,
            "either_count": 0, "either_cov": 0.0,
            "both_count": 0, "both_cov": 0.0,
        }

    # Slice left/right hand keypoints once, reuse views (no copies made here)
    lh_kp = data[:, LHAND_SL]
    rh_kp = data[:, RHAND_SL]

    # np.any(x != 0) is equivalent to ~np.all(x == 0) but avoids the extra
    # negation pass over the array — cheaper for large T.
    lh_detected = np.any(lh_kp != 0.0, axis=1)
    rh_detected = np.any(rh_kp != 0.0, axis=1)

    lh_count = int(np.count_nonzero(lh_detected))
    rh_count = int(np.count_nonzero(rh_detected))
    either_count = int(np.count_nonzero(lh_detected | rh_detected))
    both_count = int(np.count_nonzero(lh_detected & rh_detected))

    return {
        "filename": file_path.name,
        "total_frames": T,
        "left_count": lh_count, "left_cov": lh_count / T,
        "right_count": rh_count, "right_cov": rh_count / T,
        "either_count": either_count, "either_cov": either_count / T,
        "both_count": both_count, "both_cov": both_count / T,
    }


def print_single_result(res: dict) -> None:
    print("=" * 60)
    print(f"File Name: {res['filename']}")
    print("-" * 60)
    print(f"Total Frames (T): {res['total_frames']}")
    print(f"Left Hand Detection Rate:  {res['left_count']:4d} / {res['total_frames']} ({res['left_cov']:.2%})")
    print(f"Right Hand Detection Rate: {res['right_count']:4d} / {res['total_frames']} ({res['right_cov']:.2%})")
    print(f"Either Hand Detected:      {res['either_count']:4d} / {res['total_frames']} ({res['either_cov']:.2%})")
    print(f"Both Hands Detected:        {res['both_count']:4d} / {res['total_frames']} ({res['both_cov']:.2%})")
    print("=" * 60)


def analyze_directory(dir_path: Path, workers: int) -> list:
    """Analyze all .npy files in a directory in parallel (I/O-bound np.load benefits from threads)."""
    npy_files = sorted(dir_path.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in directory: {dir_path}")
        return []

    print(f"Analyzing directory: {dir_path.name} (found {len(npy_files)} .npy files, {workers} workers)...")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyze_single_file, f): f for f in npy_files}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    # Restore deterministic file order (thread completion order is non-deterministic)
    results.sort(key=lambda r: r["filename"])
    return results


def print_directory_summary(results: list, threshold: float, export_excel: Optional[str] = None) -> None:
    if not results:
        print("Failed to successfully analyze any numpy files.")
        return

    # Single pass accumulation instead of 4 separate list comprehensions over `results`.
    n = len(results)
    sum_left = sum_right = sum_either = sum_both = 0.0
    total_frames = 0
    for r in results:
        sum_left += r["left_cov"]
        sum_right += r["right_cov"]
        sum_either += r["either_cov"]
        sum_both += r["both_cov"]
        total_frames += r["total_frames"]

    print("\n" + "=" * 60)
    print("[Overall Dataset Hand Detection Coverage Summary]")
    print("-" * 60)
    print(f"Total Files Analyzed:       {n}")
    print(f"Total Accumulated Frames:   {total_frames}")
    print(f"Average Left Hand Coverage:  {sum_left / n:.2%}")
    print(f"Average Right Hand Coverage: {sum_right / n:.2%}")
    print(f"Average Either Hand Coverage:{sum_either / n:.2%}")
    print(f"Average Both Hands Coverage: {sum_both / n:.2%}")
    print("=" * 60)

    threshold_files = [r for r in results if r["either_cov"] <= threshold]
    threshold_files.sort(key=lambda x: x["either_cov"])

    print(f"\n[Files with Either Hand Coverage <= {threshold:.0%}] ({len(threshold_files)} files)")
    if threshold_files:
        print("-" * 75)
        print(f"{'No.':<5} | {'Filename':<15} | {'Frames':<8} | {'Left Cov':<10} | {'Right Cov':<10} | {'Either Cov':<10}")
        print("-" * 75)
        for idx, w in enumerate(threshold_files, 1):
            print(f"{idx:<5} | {w['filename']:<15} | {w['total_frames']:<8} | "
                  f"{w['left_cov']:<10.1%} | {w['right_cov']:<10.1%} | {w['either_cov']:<10.1%}")
        print("-" * 75)
    else:
        print("  All files meet or exceed the coverage threshold!")

    if export_excel:
        df = pd.DataFrame(threshold_files if threshold_files else results)
        # Format the coverage columns as percentages for better readability in Excel if desired, 
        # or leave as floats so users can sort numerically. Leaving as floats is usually better for Excel.
        # Rename columns to be more human readable
        df = df.rename(columns={
            "filename": "Filename",
            "total_frames": "Total Frames",
            "left_count": "Left Detected Frames",
            "left_cov": "Left Coverage",
            "right_count": "Right Detected Frames",
            "right_cov": "Right Coverage",
            "either_count": "Either Detected Frames",
            "either_cov": "Either Coverage",
            "both_count": "Both Detected Frames",
            "both_cov": "Both Coverage"
        })
        # Format as percentages for Excel string (Optional, but let's keep numeric so Excel can sort)
        df.to_excel(export_excel, index=False)
        print(f"\n[Success] Exported {len(threshold_files)} files to {export_excel}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze MediaPipe hand detection coverage from .npy files."
    )
    parser.add_argument("path", type=str, help="Path to a .npy file or a directory of .npy files")
    parser.add_argument("--workers", type=int, default=8, help="Thread pool size for directory mode (default: 8)")
    parser.add_argument("--threshold", type=float, default=1.0, help="Coverage threshold (0.0 to 1.0) to flag poor quality videos (default: 1.0, meaning show all)")
    parser.add_argument("--export_excel", type=str, default=None, help="Path to save the low coverage videos as an .xlsx file")
    args = parser.parse_args()

    input_path = Path(args.path).resolve()
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        sys.exit(1)

    if input_path.is_file():
        if input_path.suffix.lower() != ".npy":
            print(f"Error: File is not a .npy file: {input_path.name}")
            sys.exit(1)
        res = analyze_single_file(input_path)
        if res:
            print_single_result(res)

    elif input_path.is_dir():
        results = analyze_directory(input_path, workers=args.workers)
        print_directory_summary(results, threshold=args.threshold, export_excel=args.export_excel)


if __name__ == "__main__":
    main()
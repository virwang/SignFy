#!/usr/bin/env python3
"""
normalize_npy.py - Normalize 3D keypoints from MediaPipe Holistic output.
It translates each video's sequence so that the median shoulder midpoint is at (0, 0, 0)
and scales all coordinates by the median shoulder width. Undetected points (all zeros) are preserved.
Outputs are saved in separate directories with '_normalized' suffix.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_DIR = SCRIPT_DIR.parent.resolve()

def normalize_keypoints(npy_path: Path, out_path: Path) -> dict:
    """
    Normalizes a single .npy file and saves the result to out_path.
    Returns metadata dict for logging.
    """
    try:
        # Load the keypoints, shape: (T, 225)
        kp = np.load(str(npy_path))
        if kp.ndim != 2 or kp.shape[1] != 225:
            return {
                "status": "error",
                "error": f"Invalid shape: {kp.shape}. Expected (T, 225)."
            }
        
        T = kp.shape[0]
        if T == 0:
            return {
                "status": "error",
                "error": "Empty array (0 frames)."
            }

        # Reshape to (T, 75, 3) for convenience
        pts = kp.reshape(T, 75, 3) # 75 landmarks: 33 pose, 21 left_hand, 21 right_hand
        
        # Extract shoulder points for all frames
        # Left shoulder = index 11, Right shoulder = index 12
        left_shoulders = pts[:, 11, :]  # shape: (T, 3)
        right_shoulders = pts[:, 12, :] # shape: (T, 3)
        
        # Identify frames where both shoulders are detected (not all zeros)
        detected_mask = (~np.all(left_shoulders == 0.0, axis=1)) & (~np.all(right_shoulders == 0.0, axis=1))
        valid_indices = np.where(detected_mask)[0]
        
        valid_count = len(valid_indices)
        
        if valid_count > 0:
            # Calculate mid-shoulder midpoint and width for valid frames
            midpoints = (left_shoulders[valid_indices] + right_shoulders[valid_indices]) / 2.0  # (V, 3)
            widths = np.linalg.norm(left_shoulders[valid_indices] - right_shoulders[valid_indices], axis=1)  # (V,)
            
            # Use median across the video to establish stable scale and origin
            median_mid_shoulder = np.median(midpoints, axis=0)  # (3,)
            median_shoulder_width = float(np.median(widths))
            
            # Avoid division by zero
            if median_shoulder_width < 1e-5:
                median_shoulder_width = 1.0
        else:
            # Fallback values if shoulders are never detected
            median_mid_shoulder = np.zeros(3)
            median_shoulder_width = 0.40  # 0.40m is a typical human shoulder width in meters

        # Perform normalization
        normalized_pts = np.zeros_like(pts)
        
        for t in range(T):
            for i in range(75):
                pt = pts[t, i]
                if np.all(pt == 0.0):
                    # Keep undetected landmarks as 0.0
                    normalized_pts[t, i] = pt
                else:
                    # Translate and scale
                    normalized_pts[t, i] = (pt - median_mid_shoulder) / median_shoulder_width
                    
        # Reshape back to (T, 225)
        normalized_kp = normalized_pts.reshape(T, 225)
        
        # Save the normalized array
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), normalized_kp)
        
        return {
            "status": "success",
            "original_shape": list(kp.shape),
            "normalized_shape": list(normalized_kp.shape),
            "median_shoulder_width": median_shoulder_width,
            "median_mid_shoulder": median_mid_shoulder.tolist(),
            "valid_frames_count": valid_count,
            "total_frames_count": T
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def process_directory(input_dir: Path, output_dir: Path) -> dict:
    """
    Normalizes all .npy files in input_dir and saves them in output_dir.
    """
    print(f"Scanning directory: {input_dir.name} -> {output_dir.name}")
    if not input_dir.exists():
        print(f"Directory {input_dir} does not exist. Skipping.")
        return {}

    npy_files = sorted(list(input_dir.glob("*.npy")))
    print(f"Found {len(npy_files)} .npy files.")
    
    results = {}
    success_count = 0
    
    for idx, npy_path in enumerate(npy_files, 1):
        out_path = output_dir / npy_path.name
        
        res = normalize_keypoints(npy_path, out_path)
        rel_name = f"{input_dir.name}/{npy_path.name}"
        results[rel_name] = res
        
        if res["status"] == "success":
            success_count += 1
            if idx % 50 == 0 or idx == len(npy_files):
                print(f"  Processed {idx}/{len(npy_files)} files successfully.")
        else:
            print(f"  Error processing {npy_path.name}: {res.get('error')}")

    print(f"Finished {input_dir.name}: {success_count}/{len(npy_files)} processed successfully.\n")
    return results

def main():
    # Define directories
    dirs_to_process = [
        (WORKSPACE_DIR / "videos_numpy", WORKSPACE_DIR / "videos_numpy_normalized"),
        (WORKSPACE_DIR / "microsoft_numpy", WORKSPACE_DIR / "microsoft_numpy_normalized")
    ]
    
    all_summary = {}
    
    for src_dir, dst_dir in dirs_to_process:
        dir_results = process_directory(src_dir, dst_dir)
        all_summary.update(dir_results)
        
    # Save the logs/records
    summary_path = SCRIPT_DIR / "normalization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
        
    print(f"Normalization complete. Records saved to {summary_path}")

if __name__ == "__main__":
    main()

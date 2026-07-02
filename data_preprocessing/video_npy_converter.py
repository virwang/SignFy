"""
video_npy_converter.py — Extract 3D world keypoints from .mp4 file(s) into .npy file(s).
Output shape: (T, 225), float32 representing 3D world coordinates in meters.
    [0  : 99 ] pose       (33 × 3)
    [99 : 162] left_hand  (21 × 3)
    [162: 225] right_hand (21 × 3)
"""

import sys
import os
import urllib.request
from pathlib import Path
from typing import Union
import numpy as np

# ============================================================
# Configurations and Model Setup
# ============================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_DIR = SCRIPT_DIR.parent.resolve()

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)
MODEL_PATH = WORKSPACE_DIR / "holistic_landmarker.task"

def _ensure_model_file() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    print(f"Downloading Holistic model task -> {MODEL_PATH.name} ...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    print("Download complete [OK]")
    return MODEL_PATH

def _build_landmarker():
    # pyrefly: ignore [missing-import]
    from mediapipe.tasks.python.core.base_options import BaseOptions
    # pyrefly: ignore [missing-import]
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    # pyrefly: ignore [missing-import]
    from mediapipe.tasks.python.vision.holistic_landmarker import (
        HolisticLandmarker,
        HolisticLandmarkerOptions,
    )

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_ensure_model_file())),
        running_mode=VisionTaskRunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return HolisticLandmarker.create_from_options(options)

def _landmarks_to_flat(landmarks, expected_count: int) -> list:
    if landmarks and len(landmarks) == expected_count:
        flat = []
        for lm in landmarks:
            flat.extend([lm.x, lm.y, lm.z])
        return flat
    return [0.0] * (expected_count * 3)

# ============================================================
# Core Extraction
# ============================================================

def extract_world_landmarks(mp4_path: Path) -> np.ndarray:
    """
    Extracts Pose, Left Hand, and Right Hand world landmarks from a video file.
    Returns a float32 numpy array of shape (T, 225).
    """
    # pyrefly: ignore [missing-import]
    import av
    # pyrefly: ignore [missing-import]
    import mediapipe as mp

    landmarker = _build_landmarker()
    frames_kp = []

    try:
        container = av.open(str(mp4_path))
        video_stream = next(iter(container.streams.video), None)
        if video_stream is None:
            print(f"Error: No video stream found in {mp4_path.name}")
            return np.zeros((0, 225), dtype=np.float32)

        fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
        
        for frame_idx, av_frame in enumerate(container.decode(video=0)):
            img_rgb  = av_frame.to_ndarray(format="rgb24")
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            timestamp = int(frame_idx * 1000 / fps)

            result = landmarker.detect_for_video(mp_image, timestamp)

            kp = (
                _landmarks_to_flat(result.pose_world_landmarks,       33)
                + _landmarks_to_flat(result.left_hand_world_landmarks,  21)
                + _landmarks_to_flat(result.right_hand_world_landmarks, 21)
            )
            frames_kp.append(kp)

        container.close()
    finally:
        landmarker.close()

    if not frames_kp:
        return np.zeros((0, 225), dtype=np.float32)
    return np.array(frames_kp, dtype=np.float32)

# ============================================================
# Directory Routing & Helper APIs
# ============================================================

def get_output_dir(mp4_path: Path) -> Path:
    """
    Determines output directory based on path naming conventions:
    - contains 'microsoft_cut' -> WORKSPACE_DIR / 'microsoft_numpy'
    - contains 'videos_cut' -> WORKSPACE_DIR / 'videos_numpy'
    """
    mp4_path = mp4_path.resolve()
    parts = [p.lower() for p in mp4_path.parts]
    if "microsoft_cut" in parts:
        return WORKSPACE_DIR / "microsoft_numpy"
    elif "videos_cut" in parts:
        return WORKSPACE_DIR / "videos_numpy"
    else:
        # Fallback replacing _cut with _numpy or default out directory
        parent_name = mp4_path.parent.name
        if parent_name.endswith("_cut"):
            return mp4_path.parent.parent / (parent_name[:-4] + "_numpy")
        return mp4_path.parent / "numpy_out"

# ============================================================
# Export APIs
# ============================================================

def convert_video_file(mp4_path: Union[str, Path], npy_path: Union[str, Path] = None) -> bool:
    """
    Converts a single .mp4 file to a world landmarks .npy file.
    """
    mp4_p = Path(mp4_path).resolve()
    if not mp4_p.exists():
        print(f"Error: File not found: {mp4_p}")
        return False
        
    if mp4_p.suffix.lower() != ".mp4":
        print(f"Skipping non-mp4 file: {mp4_p.name}")
        return False

    if not npy_path:
        out_dir = get_output_dir(mp4_p)
        npy_path = out_dir / mp4_p.with_suffix(".npy").name
    else:
        npy_path = Path(npy_path).resolve()

    print(f"Converting: {mp4_p.name} -> {npy_path}")
    
    try:
        kp = extract_world_landmarks(mp4_p)
        if kp.shape[0] < 1:
            print(f"Error: No frames extracted from {mp4_p.name}")
            return False
            
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), kp)
        print(f"Saved: {npy_path.name} (shape: {kp.shape})")
        return True
    except Exception as e:
        print(f"Error converting {mp4_p.name}: {e}")
        return False

def convert_path(input_path: Union[str, Path]) -> bool:
    """
    Accepts a single file path or a directory path.
    Enforces that only .mp4 files are processed.
    Returns True if all conversions succeeded, False if any failed.
    """
    _ensure_model_file()
    
    input_p = Path(input_path).resolve()
    if not input_p.exists():
        print(f"Error: Path does not exist: {input_p}")
        return False

    if input_p.is_file():
        if input_p.suffix.lower() != ".mp4":
            print(f"Error: File is not a .mp4: {input_p.name}")
            return False
        return convert_video_file(input_p)

    elif input_p.is_dir():
        mp4_files = sorted(input_p.glob("*.mp4"))
        if not mp4_files:
            print(f"No .mp4 files found in directory: {input_p}")
            return True
            
        print(f"Found {len(mp4_files)} .mp4 files in {input_p.name}")
        success_count = 0
        for idx, mp4 in enumerate(mp4_files, 1):
            print(f"[{idx}/{len(mp4_files)}] ", end="")
            if convert_video_file(mp4):
                success_count += 1
                
        print(f"\nCompleted: {success_count}/{len(mp4_files)} files successfully converted.")
        return success_count == len(mp4_files)

    return False

# ============================================================
# Main Entry Point
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python video_npy_converter.py <input_mp4_file>")
        print("  python video_npy_converter.py <input_directory>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    try:
        success = convert_path(input_path)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
media_pipline_converter.py — Extract keypoints from .mp4 file into .npy file.
Outputs are saved in the `bone_sign_out` folder.
"""

import sys
import urllib.request
from pathlib import Path
from typing import Union, Tuple

import numpy as np

# ============================================================
# Configuration
# ============================================================

TARGET_DIM = 225

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)
# Model file is placed in the same directory as this script
MODEL_PATH = Path(__file__).parent / "holistic_landmarker.task"
OUTPUT_DIR = Path(__file__).parent / "bone_sign_out"


# ============================================================
# Utility functions
# ============================================================

def _ensure_model_file() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    print(f"First run — downloading Holistic model -> {MODEL_PATH.name} ...")
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


def extract(mp4_path: Path) -> np.ndarray:
    """
    Extract keypoints frame-by-frame and return a float32 array of shape (T, 225).
    A new landmarker instance is created per call (VIDEO mode timestamp constraint).
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
            print("[Error] No video stream found")
            return np.zeros((0, TARGET_DIM), dtype=np.float32)

        fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
        total = video_stream.frames or 0
        print(f"  FPS: {fps:.2f}  |  Total frames: {total or 'unknown'}")

        for frame_idx, av_frame in enumerate(container.decode(video=0)):
            img_rgb  = av_frame.to_ndarray(format="rgb24")
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            timestamp = int(frame_idx * 1000 / fps)

            result = landmarker.detect_for_video(mp_image, timestamp)

            kp = (
                _landmarks_to_flat(result.pose_landmarks,       33)
                + _landmarks_to_flat(result.left_hand_landmarks,  21)
                + _landmarks_to_flat(result.right_hand_landmarks, 21)
            )
            frames_kp.append(kp)

            if (frame_idx + 1) % 100 == 0:
                print(f"  Processed {frame_idx + 1} frames...", end="\r")

        container.close()
    finally:
        landmarker.close()

    if not frames_kp:
        return np.zeros((0, TARGET_DIM), dtype=np.float32)
    return np.array(frames_kp, dtype=np.float32)


# ============================================================
# Core API
# ============================================================

def convert_video(mp4_path: Union[str, Path]) -> Tuple[str, str]:
    """
    Extract keypoints from an .mp4 video file and save them to BOTH .npy and .json files.
    Outputs are stored under the 'bone_sign_out' directory in the project root.
    
    Returns:
        A tuple of (npy_path_str, json_path_str) representing the output file locations.
    """
    mp4_p = Path(mp4_path).resolve()
    if not mp4_p.exists():
        raise FileNotFoundError(f"Input video file not found: {mp4_p}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    npy_path = OUTPUT_DIR / mp4_p.with_suffix(".npy").name

    print(f"\n[media_pipline_converter] Input:  {mp4_p}")
    print(f"[media_pipline_converter] Output NPY:  {npy_path}")
    print("Extracting keypoints...")

    kp = extract(mp4_p)

    if kp.shape[0] < 5:
        raise ValueError(f"Too few frames ({kp.shape[0]}) to process — minimum required is 5 frames.")

    # Save as .npy
    np.save(str(npy_path), kp)
    print(f"[OK] Saved NPY: {npy_path}  shape={kp.shape}")

    return str(npy_path)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python media_pipline_converter.py <input.mp4>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    try:
        npy_path = convert_video(input_path)
        print(f"\nSuccessfully converted! \nNPY: {npy_path}")
    except Exception as e:
        print(f"\n[Error] Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

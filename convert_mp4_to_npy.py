"""
convert_mp4_to_npy.py — Extract keypoints from .mp4 file(s) into .npy file(s)

Usage:
    # Single file
    python convert_mp4_to_npy.py <input.mp4> [output.npy]

    # Batch (directory)
    python convert_mp4_to_npy.py <input_dir> [output_dir]
    # If output_dir is omitted, .npy files are saved alongside the .mp4 files.

Output shape: (T, 225), float32
    [0  : 99 ] pose       (33 × 3)
    [99 : 162] left_hand  (21 × 3)
    [162: 225] right_hand (21 × 3)

Dependencies: pip install av mediapipe>=0.10 numpy
"""

import sys
import urllib.request
from pathlib import Path

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


# ============================================================
# Utility functions
# ============================================================

def _ensure_model_file() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    print(f"First run — downloading Holistic model → {MODEL_PATH.name} ...")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    print("Download complete ✓")
    return MODEL_PATH


def _build_landmarker():
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
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
    import av
    import mediapipe as mp

    landmarker = _build_landmarker()
    frames_kp = []

    try:
        container = av.open(str(mp4_path))
        video_stream = next(iter(container.streams.video), None)
        if video_stream is None:
            print("✗ No video stream found")
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
# Entry point
# ============================================================

def _convert_one(mp4_path: Path, npy_path: Path) -> bool:
    """Convert a single mp4 to npy. Returns True on success."""
    print(f"\nInput:  {mp4_path}")
    print(f"Output: {npy_path}")
    print("Extracting keypoints...")

    kp = extract(mp4_path)

    if kp.shape[0] < 5:
        print(f"✗ Too few frames ({kp.shape[0]}) — skipped")
        return False

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_path), kp)
    print(f"✓ Saved: {npy_path}  shape={kp.shape}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python convert_mp4_to_npy.py <input.mp4> [output.npy]")
        print("  python convert_mp4_to_npy.py <input_dir>  [output_dir]")
        sys.exit(1)

    try:
        import av        # noqa: F401
        import mediapipe # noqa: F401
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run: pip install av mediapipe>=0.10")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"✗ Path not found: {input_path}")
        sys.exit(1)

    # ── Directory batch mode ──────────────────────────────────
    if input_path.is_dir():
        out_arg = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
        if out_arg is not None and not out_arg.is_dir():
            print(f"✗ Output path must be a directory when input is a directory: {out_arg}")
            sys.exit(1)

        mp4_files = sorted(input_path.glob("*.mp4"))
        if not mp4_files:
            print(f"✗ No .mp4 files found in {input_path}")
            sys.exit(1)

        print(f"Found {len(mp4_files)} .mp4 file(s) in {input_path}")
        ok, fail = 0, 0
        for mp4 in mp4_files:
            out_dir = out_arg if out_arg is not None else mp4.parent
            npy = out_dir / mp4.with_suffix(".npy").name
            if _convert_one(mp4, npy):
                ok += 1
            else:
                fail += 1

        print(f"\nDone — {ok} succeeded, {fail} failed")
        if fail:
            sys.exit(1)

    # ── Single file mode ─────────────────────────────────────
    else:
        if input_path.suffix.lower() != ".mp4":
            print(f"✗ Not a .mp4 file: {input_path}")
            sys.exit(1)

        npy_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_path.with_suffix(".npy")
        if not _convert_one(input_path, npy_path):
            sys.exit(1)


if __name__ == "__main__":
    main()

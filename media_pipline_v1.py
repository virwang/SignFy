"""
convert_mp4_to_npy.py — Extract keypoints from .mp4 file(s) into .json file(s)
and output a skeleton video with BONE_ prefix.

Usage:
    # Single file
    python convert_mp4_to_npy.py <input.mp4> [output.json]

    # Batch (directory)
    python convert_mp4_to_npy.py <input_dir> [output_dir]
    # If output_dir is omitted, .json and BONE_*.mp4 files are saved alongside the .mp4 files.

Output shape of landmarks in JSON: [T, 225]
    [0  : 99 ] pose       (33 × 3)
    [99 : 162] left_hand  (21 × 3)
    [162: 225] right_hand (21 × 3)

Dependencies: pip install av mediapipe>=0.10 numpy opencv-python
"""

import sys
import urllib.request
from pathlib import Path
from typing import Union, Optional

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


def _draw_landmarks_on_image(img_bgr, pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks=None):
    # pyrefly: ignore [missing-import]
    import cv2
    h, w, _ = img_bgr.shape

    # Define connections
    POSE_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),
        (24, 26), (26, 28), (28, 30), (30, 32), (32, 28)
    ]
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]

    # Draw face contours (burgundy/dark red: BGR (42, 42, 165))
    if face_landmarks:
        from mediapipe.tasks.python.vision.drawing_styles import _FaceLandmarksConnections
        face_color = (42, 42, 165)
        
        # 1. Face contours (eyes, eyebrows, lips, face oval)
        for conn in _FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
            if conn.start < len(face_landmarks) and conn.end < len(face_landmarks):
                p1 = face_landmarks[conn.start]
                p2 = face_landmarks[conn.end]
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                cv2.line(img_bgr, pt1, pt2, face_color, 1)

        # 2. Nose contours
        for conn in _FaceLandmarksConnections.FACE_LANDMARKS_NOSE:
            if conn.start < len(face_landmarks) and conn.end < len(face_landmarks):
                p1 = face_landmarks[conn.start]
                p2 = face_landmarks[conn.end]
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                cv2.line(img_bgr, pt1, pt2, face_color, 1)

    # Draw pose (green)
    if pose_landmarks:
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                p1 = pose_landmarks[start_idx]
                p2 = pose_landmarks[end_idx]
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                vis1 = getattr(p1, 'visibility', 1.0)
                vis2 = getattr(p2, 'visibility', 1.0)
                if vis1 > 0.5 and vis2 > 0.5:
                    cv2.line(img_bgr, pt1, pt2, (0, 255, 0), 2)
        for lm in pose_landmarks:
            vis = getattr(lm, 'visibility', 1.0)
            if vis > 0.5:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(img_bgr, pt, 3, (0, 0, 255), -1)

    # Draw left hand (cyan)
    if left_hand_landmarks:
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(left_hand_landmarks) and end_idx < len(left_hand_landmarks):
                p1 = left_hand_landmarks[start_idx]
                p2 = left_hand_landmarks[end_idx]
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                cv2.line(img_bgr, pt1, pt2, (255, 255, 0), 2)
        for lm in left_hand_landmarks:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(img_bgr, pt, 3, (0, 0, 255), -1)

    # Draw right hand (yellow)
    if right_hand_landmarks:
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(right_hand_landmarks) and end_idx < len(right_hand_landmarks):
                p1 = right_hand_landmarks[start_idx]
                p2 = right_hand_landmarks[end_idx]
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                cv2.line(img_bgr, pt1, pt2, (0, 255, 255), 2)
        for lm in right_hand_landmarks:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(img_bgr, pt, 3, (0, 0, 255), -1)


def extract(mp4_path: Path, bone_video_path: Optional[Path] = None) -> np.ndarray:
    """
    Extract keypoints frame-by-frame and return a float32 array of shape (T, 225).
    A new landmarker instance is created per call (VIDEO mode timestamp constraint).
    If bone_video_path is specified, writes a skeleton visualization video there.
    """
    # pyrefly: ignore [missing-import]
    import av
    import cv2
    # pyrefly: ignore [missing-import]
    import mediapipe as mp

    landmarker = _build_landmarker()
    frames_kp = []
    out_video = None

    try:
        container = av.open(str(mp4_path))
        video_stream = next(iter(container.streams.video), None)
        if video_stream is None:
            print("[Error] No video stream found")
            return np.zeros((0, TARGET_DIM), dtype=np.float32)

        fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
        total = video_stream.frames or 0
        print(f"  FPS: {fps:.2f}  |  Total frames: {total or 'unknown'}")

        video_initialized = False

        for frame_idx, av_frame in enumerate(container.decode(video=0)):
            img_rgb  = av_frame.to_ndarray(format="rgb24")
            
            # Defer VideoWriter initialization until we have the first frame to get exact size
            if bone_video_path is not None and not video_initialized:
                height, width, _ = img_rgb.shape
                # List of codecs to try: mp4v, avc1, XVID, MJPG
                codecs = [('mp4v', '.mp4'), ('avc1', '.mp4'), ('XVID', '.avi'), ('MJPG', '.mp4')]
                for codec, ext in codecs:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_path = bone_video_path.with_suffix(ext) if ext != '.mp4' else bone_video_path
                    print(f"[Info] Attempting to initialize VideoWriter with codec={codec} to {test_path.name}")
                    out_video = cv2.VideoWriter(str(test_path), fourcc, fps, (width, height))
                    if out_video.isOpened():
                        bone_video_path = test_path
                        print(f"[Info] VideoWriter successfully initialized with codec={codec}")
                        break
                    else:
                        out_video.release()
                        out_video = None
                if out_video is None:
                    print("[Warning] Failed to initialize VideoWriter with any codec. Skeleton video will not be saved.")
                video_initialized = True

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            timestamp = int(frame_idx * 1000 / fps)

            result = landmarker.detect_for_video(mp_image, timestamp)

            kp = (
                _landmarks_to_flat(result.pose_landmarks,       33)
                + _landmarks_to_flat(result.left_hand_landmarks,  21)
                + _landmarks_to_flat(result.right_hand_landmarks, 21)
            )
            frames_kp.append(kp)

            if out_video is not None:
                img_bgr = np.zeros_like(img_rgb)
                _draw_landmarks_on_image(
                    img_bgr,
                    result.pose_landmarks,
                    result.left_hand_landmarks,
                    result.right_hand_landmarks,
                    result.face_landmarks
                )
                out_video.write(img_bgr)

            if (frame_idx + 1) % 100 == 0:
                print(f"  Processed {frame_idx + 1} frames...", end="\r")

        container.close()
    finally:
        landmarker.close()
        if out_video is not None:
            out_video.release()
            print(f"[OK] Saved skeleton video: {bone_video_path}")

    if not frames_kp:
        return np.zeros((0, TARGET_DIM), dtype=np.float32)
    return np.array(frames_kp, dtype=np.float32)


# ============================================================
# Entry point
# ============================================================

def _convert_one(mp4_path: Path, json_path: Path) -> bool:
    """Convert a single mp4 to json. Returns True on success."""
    import json
    
    bone_video_path = json_path.parent / f"BONE_{mp4_path.name}"
    print(f"\nInput:  {mp4_path}")
    print(f"Output JSON:  {json_path}")
    print(f"Output Video: {bone_video_path}")
    print("Extracting keypoints...")

    kp = extract(mp4_path, bone_video_path=bone_video_path)

    if kp.shape[0] < 5:
        print(f"[Error] Too few frames ({kp.shape[0]}) - skipped")
        return False

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(kp.tolist(), f)
    print(f"[OK] Saved JSON: {json_path}  shape={kp.shape}")
    return True


def convert_video_to_json(mp4_path: Union[str, Path], json_path: Optional[Union[str, Path]] = None) -> Optional[str]:
    """
    Extract keypoints from a .mp4 video file and save them to a .json file.
    Also outputs a skeleton video named BONE_<original_name>.mp4.
    If json_path is not provided, saves it in the same directory with a .json extension.
    Returns:
        The absolute path to the saved .json file as a string, or None if failed.
    """
    mp4_p = Path(mp4_path).resolve()
    if json_path is None:
        json_p = mp4_p.with_suffix(".json")
    else:
        json_p = Path(json_path).resolve()

    try:
        # pyrefly: ignore [missing-import]
        import av        # noqa: F401
        # pyrefly: ignore [missing-import]
        import mediapipe # noqa: F401
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Please run: pip install av mediapipe>=0.10", file=sys.stderr)
        return None

    if _convert_one(mp4_p, json_p):
        return str(json_p)
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python convert_mp4_to_npy.py <input.mp4> [output.json]")
        print("  python convert_mp4_to_npy.py <input_dir>  [output_dir]")
        sys.exit(1)

    try:
        # pyrefly: ignore [missing-import]
        import av        # noqa: F401
        # pyrefly: ignore [missing-import]
        import mediapipe # noqa: F401
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run: pip install av mediapipe>=0.10")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"[Error] Path not found: {input_path}")
        sys.exit(1)

    # ── Directory batch mode ──────────────────────────────────
    if input_path.is_dir():
        out_arg = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
        if out_arg is not None and not out_arg.is_dir():
            print(f"[Error] Output path must be a directory when input is a directory: {out_arg}")
            sys.exit(1)

        mp4_files = sorted(input_path.glob("*.mp4"))
        if not mp4_files:
            print(f"[Error] No .mp4 files found in {input_path}")
            sys.exit(1)

        print(f"Found {len(mp4_files)} .mp4 file(s) in {input_path}")
        ok, fail = 0, 0
        for mp4 in mp4_files:
            out_dir = out_arg if out_arg is not None else mp4.parent
            json_file = out_dir / mp4.with_suffix(".json").name
            if _convert_one(mp4, json_file):
                ok += 1
            else:
                fail += 1

        print(f"\nDone — {ok} succeeded, {fail} failed")
        if fail:
            sys.exit(1)

    # ── Single file mode ─────────────────────────────────────
    else:
        if input_path.suffix.lower() != ".mp4":
            print(f"[Error] Not a .mp4 file: {input_path}")
            sys.exit(1)

        json_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_path.with_suffix(".json")
        if not _convert_one(input_path, json_path):
            sys.exit(1)


if __name__ == "__main__":
    main()

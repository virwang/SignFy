#!/usr/bin/env python3
"""
Sign Language Keypoint Player
Reads output/keypoints/*.npy (shape: T×225) and renders each frame
using the MediaPipe Holistic Landmarker skeleton topology with OpenCV.

Labels are loaded from how2sign_verified_all.csv (SENTENCE_NAME → SENTENCE).

Dependencies: pip install opencv-python numpy pandas

Usage:
  python npy_player.py path/to/dir/          # play all .npy files in directory
  python npy_player.py path/to/file.npy      # play a single file (loops)
  python npy_player.py path/to/dir --fps 25  # set playback frame rate

Keyboard controls:
  Space  - pause / resume
  Q/Esc  - quit
  N      - next file
  P      - previous file
  A / ←  - step back one frame (while paused)
  D / →  - step forward one frame (while paused)
  + / =  - speed up
  -      - slow down
  R      - replay current file
"""

import sys
import argparse
import csv
import numpy as np
import cv2
from pathlib import Path

# ── Canvas size ───────────────────────────────────────────────────────────────
CANVAS_W, CANVAS_H = 860, 660

# ── Skeleton connection tables (MediaPipe Holistic Landmarker) ────────────────
#
# Pose: 33 landmarks, indices match holistic_landmarker.task pose_landmarks
POSE_CONNECTIONS = [
    # Face outline
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Left arm (viewer's right)
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 15), (15, 21),
    # Right arm
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22),
    # Shoulders & torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left leg
    (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),
    # Right leg
    (24, 26), (26, 28), (28, 30), (30, 32), (32, 28),
]

# Hand: 21 landmarks, shared by left and right hands
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17),
]

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
C_POSE_CON   = (160, 160, 160)   # pose connections: light grey
C_POSE_LM    = (240, 240, 240)   # pose landmarks: white
C_LHAND_CON  = ( 30, 200, 200)   # left-hand connections: cyan
C_LHAND_LM   = (  0, 255, 255)   # left-hand landmarks: bright cyan
C_RHAND_CON  = ( 30, 120, 255)   # right-hand connections: orange
C_RHAND_LM   = (  0, 165, 255)   # right-hand landmarks: bright orange

# ── Keypoint layout (225-dim, matches how2sign_importer.py TARGET_DIM) ────────
POSE_N   = 33   # 33 × 3 = 99 dims
HAND_N   = 21   # 21 × 3 = 63 dims
POSE_SL  = slice(0,   99)
LHAND_SL = slice(99,  162)
RHAND_SL = slice(162, 225)

# Windows extended key codes (cv2.waitKeyEx)
KEY_LEFT  = 2424832
KEY_RIGHT = 2555904

# Path to the all-splits CSV
BASE_DIR   = Path(__file__).parent
ALL_CSV    = BASE_DIR / "how2sign_verified_all.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton rendering
# ─────────────────────────────────────────────────────────────────────────────

def flat_to_pixels(flat: np.ndarray, n: int, W: int, H: int) -> list:
    """Convert normalised coordinates to pixel coordinate list."""
    pts = []
    for i in range(n):
        x = int(np.clip(flat[i * 3],     0.0, 1.0) * W)
        y = int(np.clip(flat[i * 3 + 1], 0.0, 1.0) * H)
        pts.append((x, y))
    return pts


def draw_skeleton(canvas, flat, n, connections, c_con, c_lm, radius=4):
    H, W = canvas.shape[:2]
    pts = flat_to_pixels(flat, n, W, H)
    for i, j in connections:
        cv2.line(canvas, pts[i], pts[j], c_con, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(canvas, (x, y), radius,     c_lm,      -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius + 1, (0, 0, 0),  1, cv2.LINE_AA)


def is_detected(flat: np.ndarray) -> bool:
    """Return False when all values are zero (landmark not detected)."""
    return not np.all(flat == 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Label loading from how2sign_verified_all.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_label_map(csv_path: Path = ALL_CSV) -> dict[str, str]:
    """
    Build {SENTENCE_NAME: SENTENCE} from how2sign_verified_all.csv.
    Returns an empty dict if the file is missing.
    """
    if not csv_path.exists():
        print(f"  WARNING: label CSV not found: {csv_path}")
        return {}

    label_map: dict[str, str] = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row.get("SENTENCE_NAME", "").strip()
            text = row.get("SENTENCE",      "").strip()
            if name and text and text.lower() != "nan":
                label_map[name] = text

    print(f"  Labels loaded: {len(label_map)} entries from {csv_path.name}")
    return label_map


# ─────────────────────────────────────────────────────────────────────────────
# Frame rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_frame(kp, frame_idx, total, filename, label, paused, fps):
    W, H = CANVAS_W, CANVAS_H
    canvas = np.full((H, W, 3), 30, dtype=np.uint8)

    pose_flat  = kp[POSE_SL]
    lhand_flat = kp[LHAND_SL]
    rhand_flat = kp[RHAND_SL]

    if is_detected(pose_flat):
        draw_skeleton(canvas, pose_flat, POSE_N,
                      POSE_CONNECTIONS, C_POSE_CON, C_POSE_LM)
    if is_detected(lhand_flat):
        draw_skeleton(canvas, lhand_flat, HAND_N,
                      HAND_CONNECTIONS, C_LHAND_CON, C_LHAND_LM)
    if is_detected(rhand_flat):
        draw_skeleton(canvas, rhand_flat, HAND_N,
                      HAND_CONNECTIONS, C_RHAND_CON, C_RHAND_LM)

    # Bottom info bar
    BAR_Y = H - 80
    cv2.rectangle(canvas, (0, BAR_Y), (W, H), (12, 12, 12), -1)

    progress = frame_idx / max(total - 1, 1)
    cv2.rectangle(canvas, (10, BAR_Y + 8), (W - 10, BAR_Y + 18), (55, 55, 55), -1)
    cv2.rectangle(canvas, (10, BAR_Y + 8),
                  (10 + int((W - 20) * progress), BAR_Y + 18), (0, 190, 100), -1)

    state = "[ PAUSED ]" if paused else f">> {fps:.0f} fps"
    cv2.putText(canvas, f"{filename}    {frame_idx+1}/{total}    {state}",
                (10, BAR_Y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (190, 190, 190), 1, cv2.LINE_AA)

    if label:
        cv2.putText(canvas, label, (10, BAR_Y + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 220, 255), 1, cv2.LINE_AA)

    if frame_idx == 0:
        cv2.putText(canvas,
                    "Space=pause  N=next  P=prev  A/D=step  +/-=speed  R=replay  Q=quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (120, 120, 120), 1, cv2.LINE_AA)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Single-file playback
# ─────────────────────────────────────────────────────────────────────────────

def play_file(npy_path: Path, fps: float, label_map: dict, loop: bool = False) -> str:
    kp_seq = np.load(str(npy_path)).astype(np.float32)
    if kp_seq.ndim != 2 or kp_seq.shape[1] != 225:
        print(f"  Skipping (unexpected shape): {npy_path.name}  shape={kp_seq.shape}")
        return "next"

    T      = kp_seq.shape[0]
    label  = label_map.get(npy_path.stem, "")
    frame  = 0
    paused = False
    delay  = max(1, int(1000 / fps))

    WIN = "Sign Language Keypoint Player"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, CANVAS_W, CANVAS_H)

    while True:
        cv2.imshow(WIN, render_frame(
            kp_seq[frame], frame, T, npy_path.name, label, paused, fps))

        key       = cv2.waitKeyEx(1 if paused else delay)
        ascii_key = key & 0xFF

        if ascii_key in (ord('q'), 27):
            cv2.destroyAllWindows()
            return "quit"
        elif ascii_key == ord('n'):
            return "next"
        elif ascii_key == ord('p'):
            return "prev"
        elif ascii_key == ord('r'):
            frame, paused = 0, False
        elif ascii_key == ord(' '):
            paused = not paused
        elif ascii_key == ord('a') or key == KEY_LEFT:
            if paused:
                frame = max(0, frame - 1)
        elif ascii_key == ord('d') or key == KEY_RIGHT:
            if paused:
                frame = min(T - 1, frame + 1)
        elif ascii_key in (ord('+'), ord('=')):
            fps   = min(fps * 1.25, 120.0)
            delay = max(1, int(1000 / fps))
        elif ascii_key == ord('-'):
            fps   = max(fps / 1.25, 1.0)
            delay = max(1, int(1000 / fps))

        if not paused:
            if frame < T - 1:
                frame += 1
            elif loop:
                frame = 0
            else:
                return "next"


# ─────────────────────────────────────────────────────────────────────────────
# File collection & entry point
# ─────────────────────────────────────────────────────────────────────────────

def collect_files(target: Path) -> list:
    if target.is_file() and target.suffix == ".npy":
        return [target]
    if target.is_dir():
        files = sorted(target.glob("*.npy")) or sorted(target.rglob("*.npy"))
        return files
    return []


def main():
    parser = argparse.ArgumentParser(description="Sign Language Keypoint NPY Player")
    parser.add_argument("target", nargs="?",
                        help=".npy file or directory")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    if args.target:
        target = Path(args.target)
    else:
        print("Usage: npy_player.py <file.npy | directory>")
        sys.exit(1)

    files = collect_files(target)
    if not files:
        print(f"No .npy files found at: {target}")
        sys.exit(1)

    print(f"Found {len(files)} file(s)")
    label_map = load_label_map()
    print("Space=pause  N=next  P=prev  A/D=step  +/-=speed  R=replay  Q=quit")

    idx = 0
    while 0 <= idx < len(files):
        f = files[idx]
        sentence = label_map.get(f.stem, "<not found in CSV>")
        print(f"[{idx+1}/{len(files)}] {f.name}")
        print(f"  SENTENCE: {sentence}")
        action = play_file(f, fps=args.fps, label_map=label_map, loop=len(files) == 1)
        if action == "quit":
            break
        if action == "next":
            idx += 1
        else:
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()
    print("Playback finished.")


if __name__ == "__main__":
    main()

import os
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
import time
import warnings
from torch.utils.data._utils.collate import default_collate

warnings.filterwarnings('ignore')

import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.holistic_landmarker import HolisticLandmarker, HolisticLandmarkerOptions

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset

# --- Landmark layout constants -------------------------------------------------
# Upper-body trunk: L/R shoulder, L/R elbow, L/R wrist, L/R hip (MediaPipe Pose indices)
TRUNK_IDX = [11, 12, 13, 14, 15, 16, 23, 24]
FACE_LANDMARK_COUNT = 478   # full MediaPipe face mesh (includes eyebrows, lips/mouth, eyes, etc.)
HAND_KEYPOINT_COUNT = 21    # HaMeR hand keypoint layout (wrist + 4 joints x 5 fingers)


def initialize_hamer():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Loading HaMeR models to {device}...")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes in video

    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(device)
    model.eval()

    # We only need HaMeR. Detector and ViTPose are completely removed.
    return model, model_cfg, device


def get_bbox_from_mp_landmarks(landmarks, img_width, img_height, scale_factor=1.8):
    """
    Calculate a square bounding box (in pixel coords) from MediaPipe normalized
    hand landmarks, padded by scale_factor and clamped to the image bounds.
    """
    pts = np.fromiter(
        (v for lm in landmarks for v in (lm.x * img_width, lm.y * img_height)),
        dtype=np.float64,
        count=len(landmarks) * 2,
    ).reshape(-1, 2)

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    box_size = max(x_max - x_min, y_max - y_min) * scale_factor

    return [
        max(0, cx - box_size / 2),
        max(0, cy - box_size / 2),
        min(img_width, cx + box_size / 2),
        min(img_height, cy + box_size / 2),
    ]


def _ensure_model_file() -> Path:
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
    MODEL_PATH = Path(__file__).parent / "holistic_landmarker.task"
    if MODEL_PATH.exists():
        return MODEL_PATH
    print(f"Downloading Holistic model -> {MODEL_PATH.name} ...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    return MODEL_PATH


def _run_hamer_batch(model, device, batch_tasks, frame_results):
    """
    Run HaMeR inference on one batch of hand crops and write the resulting 3D
    keypoints (re-aligned to the MediaPipe wrist position) back into
    frame_results in place.

    Pulled out into its own function so process_video can flush a batch as
    soon as it fills up, instead of holding every hand-crop for the entire
    video in memory before running a single giant batch at the end.
    """
    if not batch_tasks:
        return

    batch_items = [t['item'] for t in batch_tasks]
    batch = default_collate(batch_items)
    batch = recursive_to(batch, device)

    with torch.inference_mode():
        out = model(batch)

    pred_keypoints = out['pred_keypoints_3d'].cpu().numpy()
    # Converted once per batch (was previously re-converted for every single
    # hand inside the loop below).
    batch_right_np = batch['right'].cpu().numpy()

    for j, task in enumerate(batch_tasks):
        f_idx = task['frame_idx']
        hand_type = task['hand_type']
        kp_3d = pred_keypoints[j]

        # Undo the left/right X-axis flip HaMeR applies internally.
        is_right_hand = batch_right_np[j]
        kp_3d[:, 0] = (2 * is_right_hand - 1) * kp_3d[:, 0]

        f_res = frame_results[f_idx]
        if hand_type == 'left' and f_res['mp_lh_wrist'] is not None:
            offset = f_res['mp_lh_wrist'] - kp_3d[0]
            f_res['lh_arr'] = kp_3d + offset
        elif hand_type == 'right' and f_res['mp_rh_wrist'] is not None:
            offset = f_res['mp_rh_wrist'] - kp_3d[0]
            f_res['rh_arr'] = kp_3d + offset


def process_video(video_path, output_npy_path, hamer_models, batch_size=48):
    model, model_cfg, device = hamer_models

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_ensure_model_file())),
        running_mode=VisionTaskRunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    landmarker = HolisticLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_results = []
    hamer_tasks = []       # hand crops waiting to be sent through HaMeR
    frame_idx = 0
    dropped_frames = 0
    hands_detected = 0
    start_time = time.time()

    print(f"Start processing video: {Path(video_path).name} (Total frames: {total_frames})")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pose_arr = np.zeros((33, 3), dtype=np.float32)
            face_arr = np.zeros((FACE_LANDMARK_COUNT, 3), dtype=np.float32)
            lh_bbox = None
            rh_bbox = None

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp = int(frame_idx * 1000 / fps)

                # 1. MediaPipe Pose (trunk), Face mesh, and Hand bboxes
                result = landmarker.detect_for_video(mp_image, timestamp)

                # Use pose_world_landmarks for metric 3D coordinates.
                if result.pose_world_landmarks:
                    landmarks = result.pose_world_landmarks
                    n = min(len(landmarks), pose_arr.shape[0])
                    pose_arr[:n] = np.array(
                        [[lm.x, lm.y, lm.z] for lm in landmarks[:n]], dtype=np.float32
                    )

                    if result.face_landmarks:
                        raw = result.face_landmarks
                        # Handles both a flat list-of-landmarks and a
                        # list-of-one-list-of-landmarks shape depending on
                        # mediapipe version.
                        face_list = raw if hasattr(raw[0], 'x') else raw[0]
                        n_face = min(len(face_list), FACE_LANDMARK_COUNT)
                        face_arr[:n_face] = np.array(
                            [[lm.x, lm.y, lm.z] for lm in face_list[:n_face]], dtype=np.float32
                        )

                    # Compute bboxes dynamically from MediaPipe hand detection.
                    if result.left_hand_landmarks:
                        lh_bbox = get_bbox_from_mp_landmarks(result.left_hand_landmarks, img_width, img_height)

                    if result.right_hand_landmarks:
                        rh_bbox = get_bbox_from_mp_landmarks(result.right_hand_landmarks, img_width, img_height)

                    # Queue hand crops for batched HaMeR inference.
                    bboxes, is_right, hand_types = [], [], []

                    if lh_bbox is not None:
                        bboxes.append(lh_bbox)
                        is_right.append(0)
                        hand_types.append('left')

                    if rh_bbox is not None:
                        bboxes.append(rh_bbox)
                        is_right.append(1)
                        hand_types.append('right')

                    if bboxes:
                        boxes = np.stack(bboxes)
                        right = np.stack(is_right)
                        dataset = ViTDetDataset(model_cfg, frame, boxes, right, rescale_factor=2.0)
                        for i in range(len(dataset)):
                            hamer_tasks.append({
                                'frame_idx': frame_idx,
                                'hand_type': hand_types[i],
                                'item': dataset[i],
                            })
                            hands_detected += 1

            except Exception as e:
                dropped_frames += 1
                print(f"  [WARN] Frame {frame_idx} failed ({e.__class__.__name__}: {e}); using zero-filled landmarks.")

            frame_results.append({
                'trunk': pose_arr[TRUNK_IDX],
                'face': face_arr,
                'mp_lh_wrist': pose_arr[15] if lh_bbox is not None else None,
                'mp_rh_wrist': pose_arr[16] if rh_bbox is not None else None,
                'lh_arr': np.zeros((HAND_KEYPOINT_COUNT, 3), dtype=np.float32),
                'rh_arr': np.zeros((HAND_KEYPOINT_COUNT, 3), dtype=np.float32),
            })

            frame_idx += 1

            # Flush a batch to the GPU as soon as it's full instead of
            # accumulating every hand-crop for the whole video in RAM.
            if len(hamer_tasks) >= batch_size:
                _run_hamer_batch(model, device, hamer_tasks[:batch_size], frame_results)
                hamer_tasks = hamer_tasks[batch_size:]

            if frame_idx % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  MediaPipe processing: {frame_idx}/{total_frames} frames ({elapsed:.2f}s)")
    finally:
        cap.release()
        landmarker.close()

    if len(frame_results) == 0:
        print("No frames processed.")
        return False

    # 2. Run HaMeR on whatever hand crops are left over (< one full batch).
    if hamer_tasks:
        print(f"  Running HaMeR batch inference on remaining {len(hamer_tasks)} hand(s)...")
        _run_hamer_batch(model, device, hamer_tasks, frame_results)

    if hands_detected:
        print(f"  HaMeR processed {hands_detected} hand crop(s) across {frame_idx} frames.")

    # 3. Assemble final data: trunk + face + left hand + right hand, per frame.
    final_data = [
        np.concatenate([res['trunk'].flatten(), res['face'].flatten(), res['lh_arr'].flatten(), res['rh_arr'].flatten()])
        for res in frame_results
    ]

    if dropped_frames > 0:
        print(f"  [WARN] {dropped_frames}/{frame_idx} frame(s) failed and were zero-filled.")

    frames_data_np = np.stack(final_data)
    os.makedirs(os.path.dirname(os.path.abspath(output_npy_path)), exist_ok=True)
    np.save(output_npy_path, frames_data_np)
    print(f"Saved {output_npy_path} (Shape: {frames_data_np.shape}) in {time.time() - start_time:.2f}s")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ASL video to 3D skeleton .npy")
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_npy", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=48)
    args = parser.parse_args()

    hamer_models = initialize_hamer()
    process_video(args.input_video, args.output_npy, hamer_models, batch_size=args.batch_size)
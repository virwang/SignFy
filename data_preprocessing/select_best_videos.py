#!/usr/bin/env python3
"""
select_best_videos.py

Evaluates candidate videos for each sign language gloss to identify the best video
for MediaPipe processing using a two-stage evaluation pipeline:
1. Clarity Pre-filter (Laplacian variance) on all candidate videos.
2. Stability Evaluation (MediaPipe detection success) on the top M clearest videos.

Outputs a best_asl_videos.json mapping each gloss to exactly one best video object.
"""

import os
import sys
import json
import math
import argparse
import logging
import numpy as np
import cv2

# Globals for worker processes
_landmarker = None
_last_resolution = None

# Configure module-level logger
logger = logging.getLogger("select_best_videos")

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

# Constants for default paths
WORKSPACE_DIR = os.path.dirname(get_script_dir())
MS_VIDEO_DIR = os.path.join(WORKSPACE_DIR, "Microsoft_Videos")
OTHER_VIDEO_DIR = os.path.join(WORKSPACE_DIR, "videos")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
MODEL_PATH = os.path.join(WORKSPACE_DIR, "holistic_landmarker.task")

def ensure_model_file():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    logger.info(f"Downloading Holistic model to {MODEL_PATH}...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    logger.info("Download complete.")
    return MODEL_PATH

def resolve_video_path(video_id, source):
    source = str(source).lower()
    if source == 'microsoft':
        path = os.path.join(MS_VIDEO_DIR, video_id)
    else:
        path = os.path.join(OTHER_VIDEO_DIR, video_id)
        if not path.endswith('.mp4') and not path.endswith('.MP4'):
            path += '.mp4'
            
    if os.path.exists(path):
        return path
        
    # Case-insensitive fallback
    parent_dir = os.path.dirname(path)
    if os.path.isdir(parent_dir):
        base_lower = os.path.basename(path).lower()
        for f in os.listdir(parent_dir):
            if f.lower() == base_lower:
                return os.path.join(parent_dir, f)
                
    return path

def check_clarity_worker(task):
    """
    Stage 1 Worker: Opens the video and computes the average Laplacian variance
    for a sample of frames.
    """
    video_path = task['video_path']
    video_id = task['video_id']
    source = task['source']
    gloss = task['gloss']
    num_samples = task['num_samples']
    
    if not os.path.exists(video_path):
        return {
            'gloss': gloss,
            'video_id': video_id,
            'source': source,
            'video_path': video_path,
            'clarity': -1.0,
            'status': 'Missing',
            'width': 0,
            'height': 0,
            'total_frames': 0,
            'fps': 0.0
        }
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'gloss': gloss,
            'video_id': video_id,
            'source': source,
            'video_path': video_path,
            'clarity': -1.0,
            'status': 'Corrupt',
            'width': 0,
            'height': 0,
            'total_frames': 0,
            'fps': 0.0
        }
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0
        
    if total_frames <= 0:
        # Fallback sequential read
        frames_grayscale = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames_grayscale.append(gray)
        cap.release()
        total_frames = len(frames_grayscale)
        if total_frames == 0:
            return {
                'gloss': gloss,
                'video_id': video_id,
                'source': source,
                'video_path': video_path,
                'clarity': 0.0,
                'status': 'Empty',
                'width': width,
                'height': height,
                'total_frames': total_frames,
                'fps': fps
            }
        indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
        variances = []
        for idx in indices:
            gray = frames_grayscale[idx]
            if gray.shape[1] > 320:
                gray = cv2.resize(gray, (320, int(gray.shape[0] * 320 / gray.shape[1])))
            variances.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        avg_var = sum(variances) / len(variances) if variances else 0.0
    else:
        # Sample using set of indices (direct seek)
        indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
        variances = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.shape[1] > 320:
                gray = cv2.resize(gray, (320, int(gray.shape[0] * 320 / gray.shape[1])))
            var = cv2.Laplacian(gray, cv2.CV_64F).var()
            variances.append(var)
        cap.release()
        
        avg_var = sum(variances) / len(variances) if variances else 0.0
        
    return {
        'gloss': gloss,
        'video_id': video_id,
        'source': source,
        'video_path': video_path,
        'clarity': avg_var,
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'fps': fps,
        'status': 'OK'
    }

def eval_stability_worker(task):
    """
    Stage 2 Worker: Evaluates pose/hand detection stability and extracts metadata
    for the 3D avatar pipeline.
    """
    global _landmarker, _last_resolution
    # pyrefly: ignore [missing-import]
    import cv2
    import numpy as np
    import math
    import os
    # pyrefly: ignore [missing-import]
    import mediapipe as mp
    
    video_path = task['video_path']
    video_id = task['video_id']
    gloss = task['gloss']
    num_samples = task['num_samples']
    width = task['width']
    height = task['height']
    total_frames = task['total_frames']
    model_path_str = task['model_path']
    
    if not os.path.exists(video_path):
        return {
            'gloss': gloss,
            'video_id': video_id,
            'status': 'Missing'
        }
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'gloss': gloss,
            'video_id': video_id,
            'status': 'Corrupt'
        }
        
    # Get sorted unique indices for direct seeking
    indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
    
    sampled_frames_rgb = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sampled_frames_rgb.append(rgb)
    cap.release()
    
    if not sampled_frames_rgb:
        return {
            'gloss': gloss,
            'video_id': video_id,
            'status': 'NoFrames'
        }
        
    current_resolution = (width, height)
    
    # Re-initialize landmarker if resolution changes to avoid MediaPipe Graph errors
    if _landmarker is not None and _last_resolution != current_resolution:
        try:
            _landmarker.close()
        except Exception:
            pass
        _landmarker = None
        
    # Lazy initialization of HolisticLandmarker
    if _landmarker is None:
        try:
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
            from mediapipe.tasks.python.vision.holistic_landmarker import (
                HolisticLandmarker,
                HolisticLandmarkerOptions,
            )
            options = HolisticLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path_str),
                running_mode=VisionTaskRunningMode.IMAGE,
                min_face_detection_confidence=0.5,
                min_pose_detection_confidence=0.5,
                min_hand_landmarks_confidence=0.5,
            )
            _landmarker = HolisticLandmarker.create_from_options(options)
            _last_resolution = current_resolution
        except Exception as e:
            return {
                'gloss': gloss,
                'video_id': video_id,
                'status': 'InitError',
                'error_msg': str(e)
            }
            
    landmarker = _landmarker
    
    try:
        pose_detections = 0
        hand_detections = 0
        
        shoulder_widths = []
        eye_distances = []
        shoulder_angles = []
        
        all_x = []
        all_y = []
        
        for rgb in sampled_frames_rgb:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            
            pose_ok = bool(result.pose_landmarks)
            left_hand_ok = bool(result.left_hand_landmarks)
            right_hand_ok = bool(result.right_hand_landmarks)
            face_ok = bool(result.face_landmarks)
            
            if pose_ok:
                pose_detections += 1
                pose_lms = result.pose_landmarks
                
                # Left shoulder = 11, Right shoulder = 12
                p11 = pose_lms[11]
                p12 = pose_lms[12]
                
                x11, y11 = p11.x * width, p11.y * height
                x12, y12 = p12.x * width, p12.y * height
                
                s_width = math.sqrt((x11 - x12)**2 + (y11 - y12)**2)
                shoulder_widths.append(s_width)
                
                # Shoulder tilt/angle relative to horizontal
                s_angle = math.degrees(math.atan2(y12 - y11, x12 - x11))
                shoulder_angles.append(s_angle)
                
                # Outer eye landmarks: Left eye outer = 3, Right eye outer = 6
                p3 = pose_lms[3]
                p6 = pose_lms[6]
                x3, y3 = p3.x * width, p3.y * height
                x6, y6 = p6.x * width, p6.y * height
                e_dist = math.sqrt((x3 - x6)**2 + (y3 - y6)**2)
                eye_distances.append(e_dist)
                
                all_x.extend([lm.x for lm in pose_lms])
                all_y.extend([lm.y for lm in pose_lms])
                
            if left_hand_ok or right_hand_ok:
                hand_detections += 1
                
            if left_hand_ok:
                all_x.extend([lm.x for lm in result.left_hand_landmarks])
                all_y.extend([lm.y for lm in result.left_hand_landmarks])
                
            if right_hand_ok:
                all_x.extend([lm.x for lm in result.right_hand_landmarks])
                all_y.extend([lm.y for lm in result.right_hand_landmarks])
                
            if face_ok:
                all_x.extend([lm.x for lm in result.face_landmarks])
                all_y.extend([lm.y for lm in result.face_landmarks])
                
        num_actual_samples = len(sampled_frames_rgb)
        pose_rate = pose_detections / num_actual_samples if num_actual_samples > 0 else 0.0
        hand_rate = hand_detections / num_actual_samples if num_actual_samples > 0 else 0.0
        
        # Stability Score = 0.3 * pose_rate + 0.7 * hand_rate
        stability_score = 0.3 * pose_rate + 0.7 * hand_rate
        
        avg_s_width = sum(shoulder_widths) / len(shoulder_widths) if shoulder_widths else 0.0
        avg_e_dist = sum(eye_distances) / len(eye_distances) if eye_distances else 0.0
        avg_s_angle = sum(shoulder_angles) / len(shoulder_angles) if shoulder_angles else 0.0
        
        bbox = {}
        if all_x and all_y:
            bbox = {
                'xmin': min(all_x),
                'ymin': min(all_y),
                'xmax': max(all_x),
                'ymax': max(all_y)
            }
            
        return {
            'gloss': gloss,
            'video_id': video_id,
            'stability_score': stability_score,
            'pose_rate': pose_rate,
            'hand_rate': hand_rate,
            'avg_shoulder_width_px': avg_s_width,
            'avg_eye_distance_px': avg_e_dist,
            'avg_shoulder_angle_deg': avg_s_angle,
            'bbox': bbox,
            'status': 'OK'
        }
    except Exception as e:
        return {
            'gloss': gloss,
            'video_id': video_id,
            'status': 'Error',
            'error_msg': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate ASL videos and select the best one per gloss.")
    parser.add_argument("--input", default=os.path.join(get_script_dir(), "asl_mis_wlasl.json"), help="Path to input mapping JSON")
    parser.add_argument("--output", default=os.path.join(get_script_dir(), "best_asl_videos.json"), help="Path to output JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of glosses to process for testing")
    parser.add_argument("--top_m", type=int, default=3, help="Number of top candidates per gloss to evaluate with MediaPipe")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of frames to sample per video")
    args = parser.parse_args()
    
    # Configure logger
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    log_file = os.path.join(get_script_dir(), "select_best_videos.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("Starting ASL Video Evaluation Pipeline")
    logger.info(f"Configuration: input={args.input}, output={args.output}, limit={args.limit}, top_m={args.top_m}, num_samples={args.num_samples}")

    # Check dependencies and model
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, *args, **kwargs):
            return iterable
            
    # Load input mapping
    if not os.path.exists(args.input):
        logger.error(f"Mapping file {args.input} not found.")
        sys.exit(1)
        
    with open(args.input, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
        
    # Apply limit if specified
    if args.limit is not None:
        mapping_data = mapping_data[:args.limit]
        logger.info(f"Limiting execution to the first {args.limit} glosses.")
        
    # Ensure MediaPipe model is downloaded
    model_path = ensure_model_file()

    # Load progress checkpoint if it exists
    progress_file = os.path.join(get_script_dir(), "select_best_videos_progress.json")
    progress = {'clarity_results': {}, 'stability_results': {}}
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                progress = json.load(f)
            logger.info(f"Loaded progress from {progress_file}. "
                        f"Found {len(progress.get('clarity_results', {}))} clarity results "
                        f"and {len(progress.get('stability_results', {}))} stability results.")
        except Exception as e:
            logger.warning(f"Failed to load progress file {progress_file}: {e}. Starting from scratch.")
    
    # -------------------------------------------------------------
    # STAGE 1: Clarity Evaluation
    # -------------------------------------------------------------
    logger.info("Stage 1: Evaluating image clarity for all candidate videos...")
    stage1_tasks = []
    
    clarity_results = []
    for res in progress.get('clarity_results', {}).values():
        clarity_results.append(res)
        
    for entry in mapping_data:
        gloss = entry.get("gloss", "").upper()
        if not gloss:
            continue
        items = entry.get("item", [])
        for item in items:
            video_id = item.get("video_id")
            source = item.get("source")
            video_path = resolve_video_path(video_id, source)
            
            key = f"{gloss}::{video_id}"
            if key in progress.get('clarity_results', {}):
                continue
                
            task = {
                'gloss': gloss,
                'video_id': video_id,
                'source': source,
                'video_path': video_path,
                'num_samples': args.num_samples
            }
            stage1_tasks.append(task)
            
    logger.info(f"Total candidate videos to evaluate for clarity: {len(stage1_tasks)}")
    
    if stage1_tasks:
        from multiprocessing import Pool
        total_stage1 = len(stage1_tasks)
        log_interval_s1 = max(1, total_stage1 // 20)  # Log every 5%
        
        # Run clarity workers
        with Pool(processes=os.cpu_count()) as pool:
            for idx, res in enumerate(tqdm(pool.imap_unordered(check_clarity_worker, stage1_tasks), total=total_stage1, desc="Clarity Check")):
                if res:
                    clarity_results.append(res)
                    key = f"{res['gloss']}::{res['video_id']}"
                    progress['clarity_results'][key] = res
                    if res['status'] != 'OK':
                        logger.warning(f"Clarity worker reported {res['status']} for {res['gloss']} ({res['video_id']}): {res['video_path']}")
                
                if (idx + 1) % log_interval_s1 == 0 or (idx + 1) == total_stage1:
                    percentage = ((idx + 1) / total_stage1) * 100
                    logger.info(f"Stage 1 progress: {idx + 1}/{total_stage1} videos processed ({percentage:.1f}%)")
                    try:
                        with open(progress_file, "w", encoding="utf-8") as f:
                            json.dump(progress, f, indent=4)
                    except Exception as e:
                        logger.warning(f"Failed to save progress to {progress_file}: {e}")
    else:
        logger.info("All Stage 1 clarity tasks loaded from progress file.")
                
    # Group results by gloss
    gloss_candidates = {}
    for res in clarity_results:
        if res['clarity'] < 0:
            continue  # Skip missing/corrupt
        gloss = res['gloss']
        if gloss not in gloss_candidates:
            gloss_candidates[gloss] = []
        gloss_candidates[gloss].append(res)
        
    stage2_tasks = []
    stability_results = {}
    
    # Load already completed stability results from progress
    for key, res in progress.get('stability_results', {}).items():
        parts = key.split("::", 1)
        if len(parts) == 2:
            g, v_id = parts
            stability_results[(g, v_id)] = res
            
    for gloss, candidates in gloss_candidates.items():
        # Sort by clarity descending
        candidates.sort(key=lambda x: x['clarity'], reverse=True)
        # Keep top M candidates and add parameters
        top_candidates = candidates[:args.top_m]
        for tc in top_candidates:
            key = f"{gloss}::{tc['video_id']}"
            if key in progress.get('stability_results', {}):
                continue
            tc['num_samples'] = args.num_samples
            tc['model_path'] = model_path
            stage2_tasks.append(tc)
        
    # -------------------------------------------------------------
    # STAGE 2: Stability Evaluation
    # -------------------------------------------------------------
    logger.info(f"Stage 2: Evaluating detection stability on top {args.top_m} clearest videos per gloss (Total to run: {len(stage2_tasks)})...")
    
    if stage2_tasks:
        from multiprocessing import Pool
        total_stage2 = len(stage2_tasks)
        log_interval_s2 = max(1, total_stage2 // 20)  # Log every 5%
        
        # Run stability workers using lazy resolution init pool (no need for manual process initializer anymore)
        with Pool(processes=os.cpu_count()) as pool:
            for idx, res in enumerate(tqdm(pool.imap_unordered(eval_stability_worker, stage2_tasks), total=total_stage2, desc="MediaPipe Check")):
                if res:
                    status = res.get('status', 'OK')
                    key = f"{res['gloss']}::{res['video_id']}"
                    if status == 'OK':
                        stability_results[(res['gloss'], res['video_id'])] = res
                        progress['stability_results'][key] = res
                    else:
                        err_msg = res.get('error_msg', 'Unknown error')
                        logger.warning(f"Stability worker reported {status} for {res['gloss']} ({res['video_id']}): {err_msg}")
                        progress['stability_results'][key] = res
                
                if (idx + 1) % log_interval_s2 == 0 or (idx + 1) == total_stage2:
                    percentage = ((idx + 1) / total_stage2) * 100
                    logger.info(f"Stage 2 progress: {idx + 1}/{total_stage2} videos processed ({percentage:.1f}%)")
                    try:
                        with open(progress_file, "w", encoding="utf-8") as f:
                            json.dump(progress, f, indent=4)
                    except Exception as e:
                        logger.warning(f"Failed to save progress to {progress_file}: {e}")
    else:
        logger.info("All Stage 2 stability tasks loaded from progress file.")
                
    # -------------------------------------------------------------
    # Stage 3: Scoring and Final Selection
    # -------------------------------------------------------------
    logger.info("Stage 3: Selecting the best videos and building final mapping...")
    best_videos = {}
    
    for gloss, candidates in gloss_candidates.items():
        for cand in candidates:
            video_id = cand['video_id']
            stability_res = stability_results.get((gloss, video_id))
            
            if stability_res:
                stability_score = stability_res['stability_score']
                # Composite score
                composite_score = stability_score * cand['clarity']
            else:
                composite_score = 0.0
                stability_res = None
                
            cand['composite_score'] = composite_score
            cand['stability_res'] = stability_res
            
        # Select best candidate
        # Primary key: composite_score descending
        # Fallback secondary key: clarity (Laplacian variance) descending
        candidates.sort(key=lambda x: (x.get('composite_score', 0.0), x['clarity']), reverse=True)
        best = candidates[0]
        
        width = best['width']
        height = best['height']
        fps = best['fps']
        total_frames = best['total_frames']
        clarity = best['clarity']
        
        stab = best.get('stability_res')
        if stab:
            pose_rate = stab['pose_rate']
            hand_rate = stab['hand_rate']
            avg_shoulder_width_px = stab['avg_shoulder_width_px']
            avg_eye_distance_px = stab['avg_eye_distance_px']
            avg_shoulder_angle_deg = stab['avg_shoulder_angle_deg']
            bbox = stab['bbox']
            score = best['composite_score']
        else:
            pose_rate = 0.0
            hand_rate = 0.0
            avg_shoulder_width_px = 0.0
            avg_eye_distance_px = 0.0
            avg_shoulder_angle_deg = 0.0
            bbox = {}
            score = 0.0
            
        # Normalize metrics
        distance_scale_factor = avg_shoulder_width_px / width if width > 0 else 0.0
        
        # Determine distance category based on resolution-independent scale factor
        if distance_scale_factor > 0.35:
            distance_category = "close-up"
        elif 0.18 <= distance_scale_factor <= 0.35:
            distance_category = "medium"
        else:
            distance_category = "long"
            
        # Structuring metadata
        metadata = {
            'video_details': {
                'width': width,
                'height': height,
                'fps': round(fps, 2),
                'total_frames': total_frames
            },
            'subject_metrics': {
                'avg_shoulder_width_px': round(avg_shoulder_width_px, 2),
                'avg_eye_distance_px': round(avg_eye_distance_px, 2),
                'distance_scale_factor': round(distance_scale_factor, 4),
                'estimated_distance_category': distance_category,
                'avg_shoulder_angle_deg': round(avg_shoulder_angle_deg, 2),
                'bbox': bbox
            },
            'tracking_quality': {
                'pose_detection_rate': round(pose_rate, 4),
                'hand_detection_rate': round(hand_rate, 4),
                'avg_clarity_score': round(clarity, 2)
            }
        }
        
        best_videos[gloss] = {
            'gloss': gloss,
            'score': round(score, 2) if score > 0 else round(clarity, 2),
            'video_id': best['video_id'],
            'source': best['source'],
            'metadata': metadata
        }
        
    # Write to file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(best_videos, f, indent=4)
        
    logger.info(f"Completed evaluation. Best ASL videos saved to: {args.output}")

if __name__ == "__main__":
    main()

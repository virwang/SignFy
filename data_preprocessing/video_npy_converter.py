import os
import sys
import json
import argparse
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

WORKSPACE_DIR = os.path.dirname(get_script_dir())

def convert_video_file(mp4_path, npy_path=None):
    """
    Reads an MP4 file, extracts all frames, converts them to RGB,
    and saves them as a numpy array (.npy).
    """
    if not os.path.exists(mp4_path):
        print(f"Error: Video file not found: {mp4_path}")
        return False
        
    if not npy_path:
        # Save in the same folder as mp4_path, replacing extension with .npy
        base, _ = os.path.splitext(mp4_path)
        npy_path = base + ".npy"
        
    print(f"Converting video: {mp4_path} -> {npy_path}")
    
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {mp4_path}")
        return False
        
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB (MediaPipe / standard format)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    if not frames:
        print(f"Error: No frames extracted from {mp4_path}")
        return False
        
    frames_np = np.array(frames, dtype=np.uint8)
    
    # Ensure output parent directory exists
    parent_dir = os.path.dirname(npy_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
        
    np.save(npy_path, frames_np)
    print(f"Successfully saved numpy array with shape {frames_np.shape} to {npy_path}")
    return True

def run_batch_conversion(best_videos_json=None):
    if best_videos_json is None:
        best_videos_json = os.path.join(WORKSPACE_DIR, "data_preprocessing", "best_asl_videos.json")

    if not os.path.exists(best_videos_json):
        print(f"Error: JSON file not found: {best_videos_json}")
        return
        
    print(f"Loading best videos from {best_videos_json}...")
    with open(best_videos_json, "r", encoding="utf-8") as f:
        best_videos = json.load(f)
        
    if isinstance(best_videos, dict):
        videos_list = list(best_videos.values())
    else:
        videos_list = best_videos
        
    print(f"Found {len(videos_list)} videos in JSON mapping.")
    
    # Paths (relative to root directory)
    MS_VIDEO_DIR = os.path.join(WORKSPACE_DIR, "Microsoft_Videos")
    OTHER_VIDEO_DIR = os.path.join(WORKSPACE_DIR, "videos")
    MS_OUT_DIR = os.path.join(WORKSPACE_DIR, "microsoft_numpy")
    OTHER_OUT_DIR = os.path.join(WORKSPACE_DIR, "videos_numpy")
    
    # Ensure root output directories exist
    os.makedirs(MS_OUT_DIR, exist_ok=True)
    os.makedirs(OTHER_OUT_DIR, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for item in videos_list:
            video_id = item.get("video_id")
            gloss = item.get("gloss", "unknown").upper()
            source = item.get("source", "other").lower()
            
            if not video_id:
                continue
                
            # Determine source and output directories
            if source == "microsoft":
                source_dir = MS_VIDEO_DIR
                target_base = MS_OUT_DIR
            else:
                source_dir = OTHER_VIDEO_DIR
                target_base = OTHER_OUT_DIR
                
            # Resolve source file path
            video_id_str = str(video_id)
            possible_paths = [
                os.path.join(source_dir, video_id_str),
                os.path.join(source_dir, video_id_str + ".mp4"),
                os.path.join(source_dir, video_id_str + ".MP4")
            ]
            if video_id_str.lower().endswith(".mp4"):
                possible_paths.insert(0, os.path.join(source_dir, video_id_str[:-4]))
                
            src_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    src_path = p
                    break
                    
            if not src_path:
                # Try case insensitive fallback
                if os.path.exists(source_dir):
                    base_lower = video_id_str.lower()
                    for f in os.listdir(source_dir):
                        if f.lower() == base_lower or f.lower() == (base_lower + ".mp4"):
                            src_path = os.path.join(source_dir, f)
                            break
                            
            if not src_path:
                print(f"⚠️ Missing source video for gloss {gloss} (video_id: {video_id})")
                continue
                
            # Clean video_id for numpy filename
            video_id_clean = video_id_str
            if video_id_clean.lower().endswith(".mp4"):
                video_id_clean = video_id_clean[:-4]
                
            # Build target output path under the gloss folder
            dst_dir = os.path.join(target_base, gloss)
            dst_path = os.path.join(dst_dir, f"{video_id_clean}.npy")
            
            futures.append(executor.submit(convert_video_file, src_path, dst_path))
            
        success_count = 0
        failure_count = 0
        for fut in as_completed(futures):
            try:
                if fut.result():
                    success_count += 1
                else:
                    failure_count += 1
            except Exception as e:
                print(f"Exception during conversion: {e}")
                failure_count += 1
                
        print(f"\nBatch conversion finished. {success_count} succeeded, {failure_count} failed.")

def main():
    parser = argparse.ArgumentParser(description="Convert MP4 files to raw frames numpy arrays.")
    parser.add_argument("input", nargs="?", default=None, help="Path to input .mp4 file for single conversion.")
    parser.add_argument("output", nargs="?", default=None, help="Path to output .npy file for single conversion.")
    parser.add_argument("--batch", action="store_true", help="Run batch conversion of best videos from JSON.")
    parser.add_argument("--json", default=None, help="Path to best videos JSON mapping.")
    args = parser.parse_args()
    
    json_path = args.json
    if json_path is None:
        json_path = os.path.join(WORKSPACE_DIR, "data_preprocessing", "best_asl_videos.json")
        
    if args.batch:
        run_batch_conversion(json_path)
    elif args.input:
        convert_video_file(args.input, args.output)
    else:
        # Default behavior if run with no args: run batch mode
        run_batch_conversion(json_path)

if __name__ == "__main__":
    main()

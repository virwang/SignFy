'''
This script is aim to clip all the videos that has frame_start and frame_end information,
Based on the information, clipe the video and save it under the given folder.
The output video name must be the same as the original video name, just with different length.
Example: video name: '00_0001_000001.mp4', then the output video name must be '00_0001_000001.mp4'

Input: 
    video folder: 'videos', 'Microsoft_Videos'
    split json: 'data_preprocessing/best_asl_videos.json'
    output folder: 'videos_cut', 'microsoft_cut'

Output: 
    clipped videos under the given output folder
'''

import os
import sys
import json
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
# pyrefly: ignore [missing-import]
import cv2
import numpy as np

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

WORKSPACE_DIR = os.path.dirname(get_script_dir())

def load_wlasl_mapping(wlasl_path):
    """
    Loads WLASL_v03.json and returns a lookup dictionary of
    video_id -> {frame_start, frame_end, fps}.
    """
    if not os.path.exists(wlasl_path):
        print(f"WARNING: WLASL JSON not found at {wlasl_path}")
        return {}
        
    try:
        with open(wlasl_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping = {}
        for entry in data:
            for inst in entry.get("instances", []):
                vid = str(inst.get("video_id", ""))
                if vid:
                    mapping[vid] = {
                        "frame_start": inst.get("frame_start"),
                        "frame_end": inst.get("frame_end"),
                        "fps": inst.get("fps")
                    }
                    # Also map the extension-free version
                    mapping[os.path.splitext(vid)[0]] = mapping[vid]
        return mapping
    except Exception as e:
        print(f"ERROR: Error reading WLASL JSON: {e}")
        return {}

def clip_video_opencv(src_path, dst_path, frame_start, frame_end):
    """
    Clips video from frame_start to frame_end using OpenCV.
    frame_start is 1-indexed.
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return False, f"Could not open source video: {src_path}"
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1-indexed to 0-indexed conversion
    start_frame = max(0, frame_start - 1)
    
    if frame_end == -1 or frame_end is None:
        end_frame = total_frames
    else:
        end_frame = min(total_frames, frame_end)
        
    if start_frame >= total_frames:
        cap.release()
        return False, f"frame_start ({frame_start}) is beyond total_frames ({total_frames})"
        
    if start_frame >= end_frame:
        cap.release()
        return False, f"frame_start ({frame_start}) >= frame_end ({frame_end})"
        
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    # Use MP4V codec for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        return False, "Could not initialize OpenCV VideoWriter"
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames_written = 0
    curr_frame = start_frame
    while curr_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_written += 1
        curr_frame += 1
        
    cap.release()
    out.release()
    
    if frames_written == 0:
        return False, "Zero frames written"
        
    return True, f"Clipped {frames_written} frames ({frame_start} to {frame_end})"

def clip_video_moviepy(src_path, dst_path, frame_start, frame_end):
    """
    Clips video from frame_start to frame_end using MoviePy as a fallback.
    """
    try:
        # pyrefly: ignore [missing-import]
        from moviepy import VideoFileClip
    except ImportError:
        return False, "MoviePy not installed"
        
    try:
        clip = VideoFileClip(src_path)
        fps = clip.fps if clip.fps else 25.0
        
        t_start = max(0.0, (frame_start - 1) / fps)
        t_end = None
        if frame_end > 0:
            t_end = frame_end / fps
            
        if clip.duration:
            if t_start >= clip.duration:
                t_start = 0.0
            if t_end is not None:
                if t_end > clip.duration or t_end <= t_start:
                    t_end = clip.duration
                    
        if hasattr(clip, "subclipped"):
            clipped = clip.subclipped(t_start, t_end)
        else:
            clipped = clip.subclip(t_start, t_end)
            
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        clipped.write_videofile(
            dst_path,
            codec="libx264",
            audio=False,
            logger=None
        )
        clipped.close()
        clip.close()
        return True, f"Clipped with MoviePy (t_start={t_start:.2f}s, t_end={t_end})"
    except Exception as e:
        return False, f"MoviePy error: {e}"

def process_video_task(src_path, dst_path, frame_start, frame_end, requires_clipping, copy_uncut, engine):
    """
    Worker task: either clips the video or copies it as-is.
    """
    try:
        if requires_clipping:
            # Attempt OpenCV clipping first
            if engine == 'opencv':
                success, msg = clip_video_opencv(src_path, dst_path, frame_start, frame_end)
                if not success:
                    # Fallback to moviepy
                    success, msg = clip_video_moviepy(src_path, dst_path, frame_start, frame_end)
                return "clip", success, msg
            else:
                success, msg = clip_video_moviepy(src_path, dst_path, frame_start, frame_end)
                return "clip", success, msg
        else:
            if copy_uncut:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                return "copy", True, "Copied file directly"
            else:
                return "skip", True, "Skipped uncut video"
    except Exception as e:
        return "error", False, str(e)

def run_clipping(args):
    script_dir = get_script_dir()
    
    # Setup default paths relative to workspace root
    default_video_dirs = [
        os.path.join(WORKSPACE_DIR, "videos"),
        os.path.join(WORKSPACE_DIR, "Microsoft_Videos")
    ]
    default_output_dirs = [
        os.path.join(WORKSPACE_DIR, "videos_cut"),
        os.path.join(WORKSPACE_DIR, "microsoft_cut")
    ]
    default_split_json = os.path.join(script_dir, "best_asl_videos.json")
    default_wlasl_json = os.path.join(script_dir, "WLASL_v03.json")
    
    # Resolve argument variables or fallbacks
    video_dirs = args.video_dirs if args.video_dirs else default_video_dirs
    output_dirs = args.output_dirs if args.output_dirs else default_output_dirs
    split_json = args.split_json if args.split_json else default_split_json
    wlasl_json = args.wlasl_json if args.wlasl_json else default_wlasl_json
    
    other_video_dir = video_dirs[0] if len(video_dirs) > 0 else default_video_dirs[0]
    ms_video_dir = video_dirs[1] if len(video_dirs) > 1 else default_video_dirs[1]
    
    other_cut_dir = output_dirs[0] if len(output_dirs) > 0 else default_output_dirs[0]
    ms_cut_dir = output_dirs[1] if len(output_dirs) > 1 else default_output_dirs[1]
    
    print("--- Video Frame Clipper ---")
    print(f"Loading Split JSON: {split_json}")
    if not os.path.exists(split_json):
        print(f"ERROR: Split JSON not found: {split_json}")
        return
        
    with open(split_json, "r", encoding="utf-8") as f:
        best_videos = json.load(f)
        
    if isinstance(best_videos, dict):
        videos_list = list(best_videos.values())
    else:
        videos_list = best_videos
        
    print(f"Found {len(videos_list)} videos in Split JSON.")
    
    # Load WLASL mapping
    print(f"Loading WLASL metadata from: {wlasl_json}")
    wlasl_mapping = load_wlasl_mapping(wlasl_json)
    print(f"Loaded WLASL mappings for {len(wlasl_mapping)} video entries.")
    
    # Ensure cut directories exist
    os.makedirs(other_cut_dir, exist_ok=True)
    os.makedirs(ms_cut_dir, exist_ok=True)
    
    stats = {"clip_success": 0, "clip_fail": 0, "copy_success": 0, "copy_fail": 0, "skipped": 0, "missing": 0}
    
    print(f"Processing videos using {args.num_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for item in videos_list:
            video_id = item.get("video_id")
            gloss = item.get("gloss", "unknown").upper()
            source = item.get("source", "other").lower()
            
            if not video_id:
                continue
                
            # Determine paths
            if source == "microsoft":
                source_dir = ms_video_dir
                target_dir = ms_cut_dir
            else:
                source_dir = other_video_dir
                target_dir = other_cut_dir
                
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
                # Case insensitive check
                if os.path.exists(source_dir):
                    base_lower = video_id_str.lower()
                    for f in os.listdir(source_dir):
                        if f.lower() == base_lower or f.lower() == (base_lower + ".mp4"):
                            src_path = os.path.join(source_dir, f)
                            break
                            
            if not src_path:
                print(f"WARNING: Missing source file for {gloss} (video_id: {video_id})")
                stats["missing"] += 1
                continue
                
            # Destination path flat inside target_dir
            dst_filename = os.path.basename(src_path)
            dst_path = os.path.join(target_dir, dst_filename)
            
            # Lookup frame_start / frame_end
            frame_start, frame_end = None, None
            requires_clipping = False
            
            if source != "microsoft":
                vid_clean = os.path.splitext(video_id_str)[0]
                wlasl_info = wlasl_mapping.get(video_id_str) or wlasl_mapping.get(vid_clean)
                if wlasl_info:
                    frame_start = wlasl_info.get("frame_start")
                    frame_end = wlasl_info.get("frame_end")
                    if (frame_start is not None and frame_start > 1) or (frame_end is not None and frame_end != -1):
                        requires_clipping = True
                        
            # Submit task
            future = executor.submit(
                process_video_task,
                src_path,
                dst_path,
                frame_start,
                frame_end,
                requires_clipping,
                not args.no_copy_uncut,
                args.engine
            )
            futures[future] = (gloss, video_id_str, requires_clipping)
            
        # Collect results
        total_tasks = len(futures)
        completed = 0
        for future in as_completed(futures):
            gloss, video_id_str, was_clipped = futures[future]
            completed += 1
            try:
                action, success, msg = future.result()
                if action == "clip":
                    if success:
                        stats["clip_success"] += 1
                        print(f"[{completed}/{total_tasks}] CLIPPED {gloss} ({video_id_str}): {msg}")
                    else:
                        stats["clip_fail"] += 1
                        print(f"[{completed}/{total_tasks}] FAILED CLIP {gloss} ({video_id_str}): {msg}")
                elif action == "copy":
                    if success:
                        stats["copy_success"] += 1
                    else:
                        stats["copy_fail"] += 1
                        print(f"[{completed}/{total_tasks}] FAILED COPY {gloss} ({video_id_str}): {msg}")
                elif action == "skip":
                    stats["skipped"] += 1
            except Exception as e:
                print(f"[{completed}/{total_tasks}] EXCEPTION processing {gloss} ({video_id_str}): {e}")
                if was_clipped:
                    stats["clip_fail"] += 1
                else:
                    stats["copy_fail"] += 1
                    
    print("\n--- Processing Summary ---")
    print(f"Total videos processed:  {total_tasks}")
    print(f"Successful clips:        {stats['clip_success']}")
    print(f"Failed clips:            {stats['clip_fail']}")
    print(f"Successful direct copies: {stats['copy_success']}")
    print(f"Failed direct copies:     {stats['copy_fail']}")
    print(f"Skipped uncut:           {stats['skipped']}")
    print(f"Missing source files:    {stats['missing']}")
    print("--------------------------")

def main():
    parser = argparse.ArgumentParser(description="Clip video files based on frame start and end metadata.")
    parser.add_argument("--video_dirs", nargs="+", default=None,
                        help="List of input video folders (e.g. videos Microsoft_Videos)")
    parser.add_argument("--output_dirs", nargs="+", default=None,
                        help="List of output video folders (e.g. videos_cut microsoft_cut)")
    parser.add_argument("--split_json", default=None,
                        help="Path to best_asl_videos.json mapping file")
    parser.add_argument("--wlasl_json", default=None,
                        help="Path to WLASL_v03.json mapping file")
    parser.add_argument("--engine", choices=["opencv", "moviepy"], default="opencv",
                        help="Primary video cutting library (default: opencv)")
    parser.add_argument("--no_copy_uncut", action="store_true",
                        help="Do not copy files that don't need clipping")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of threads for batch conversion (default: 8)")
    args = parser.parse_args()
    
    run_clipping(args)

if __name__ == "__main__":
    main()

'''
This script is aim to read the best_asl_videos.json and move the videos based on the source to the 
target folder.

Input:
    video folder: 'videos', 'Microsoft_Videos'
    split json: 'data_preprocessing/best_asl_videos.json'
    output folder: 'videos_best', 'microsoft_best'

Output: 
    videos_best and microsoft_best folder
Note:
best_asl_videos.json if source is microsoft then copy to microsoft_best, other copy to videos_best。
The purpose is just copy file, not clip video...
Before moving in, check if the video is already exists in the target folder.
If the video name is different in the source folder, try to find it.
If the video is still not found, print the video name and the source folder.
'''

import os
import sys
import shutil
import json

def find_video_file(source_dir, video_id):
    """
    Search for a video file inside source_dir matching video_id.
    Handles variations in extension, case, and dashes/underscores.
    """
    if not os.path.exists(source_dir):
        return None
        
    video_id_str = str(video_id).strip()
    if not video_id_str:
        return None

    # Normalization helper
    def clean_name(name):
        name_lower = name.lower()
        if name_lower.endswith('.mp4'):
            name_lower = name_lower[:-4]
        return name_lower

    target_clean = clean_name(video_id_str)

    # 1. First pass: exact or extension-only matching (case-insensitive)
    for filename in os.listdir(source_dir):
        if clean_name(filename) == target_clean:
            return os.path.join(source_dir, filename)

    # 2. Second pass: If target_clean contains separators (e.g. '-', '_'), try matching parts
    # e.g., if JSON has "5022117525359895-PARK.mp4" but file is "5022117525359895.mp4"
    # or if JSON has "5022117525359895" but file is "5022117525359895-PARK.mp4"
    for filename in os.listdir(source_dir):
        fn_clean = clean_name(filename)
        
        parts_target = [p for p in target_clean.replace('_', '-').split('-') if p]
        parts_fn = [p for p in fn_clean.replace('_', '-').split('-') if p]
        
        if parts_target and parts_fn:
            if parts_target[0] == parts_fn[0]:
                return os.path.join(source_dir, filename)

    # 3. Third pass: substring match as a fallback
    for filename in os.listdir(source_dir):
        fn_clean = clean_name(filename)
        if len(target_clean) >= 5 and (target_clean in fn_clean or fn_clean in target_clean):
            return os.path.join(source_dir, filename)
            
    return None

def main():
    # Hardcoded paths relative to the workspace root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(script_dir)

    split_json = os.path.join(script_dir, "best_asl_videos.json")
    
    # Input video folders
    microsoft_video_dir = os.path.join(workspace_dir, "Microsoft_Videos")
    other_video_dir = os.path.join(workspace_dir, "videos")
    
    # Output folders
    microsoft_best_dir = os.path.join(workspace_dir, "microsoft_best")
    other_best_dir = os.path.join(workspace_dir, "videos_best")

    print(f"Loading best videos mapping from: {split_json}")
    try:
        with open(split_json, "r", encoding="utf-8") as f:
            best_videos = json.load(f)
    except Exception as e:
        print(f"Error: Could not read {split_json}: {e}")
        sys.exit(1)

    # Convert dictionary to list of values if needed
    if isinstance(best_videos, dict):
        videos_list = list(best_videos.values())
    else:
        videos_list = best_videos

    print(f"Loaded {len(videos_list)} video entries.")

    # Create destination folders
    os.makedirs(microsoft_best_dir, exist_ok=True)
    os.makedirs(other_best_dir, exist_ok=True)

    copied_count = 0
    already_exists_count = 0
    missing_count = 0

    for video_info in videos_list:
        if not isinstance(video_info, dict):
            continue
            
        video_id = video_info.get("video_id") or video_info.get("video_name")
        source = video_info.get("source", "other").lower()

        if not video_id:
            continue

        # Determine target and source folders
        if source == "microsoft":
            source_dir = microsoft_video_dir
            output_dir = microsoft_best_dir
        else:
            source_dir = other_video_dir
            output_dir = other_best_dir

        # 1. Before moving, check if the video already exists in the target folder
        existing_target_path = find_video_file(output_dir, video_id)
        if existing_target_path:
            already_exists_count += 1
            continue

        # 2. Try to find the video in the source folder (even if name is different)
        src_path = find_video_file(source_dir, video_id)

        # 3. If still not found, print the video name and the source folder
        if not src_path:
            print(f"Warning: Video '{video_id}' not found in source folder '{source_dir}'")
            missing_count += 1
            continue

        # Destination path preserving the resolved filename
        resolved_filename = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, resolved_filename)

        try:
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {resolved_filename} -> {dst_path}")
            copied_count += 1
        except Exception as e:
            print(f"Error copying {resolved_filename}: {e}")

    print(f"\nDone! Copied: {copied_count}, Already Exists: {already_exists_count}, Missing: {missing_count}")

if __name__ == "__main__":
    main()
    
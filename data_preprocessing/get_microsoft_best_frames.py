#!/usr/bin/env python3
"""
get_microsoft_best_frames.py

Scans the Microsoft_best folder for MP4 videos, extracts their frame count, FPS, and dimensions,
and outputs a JSON file mapping each video to its frame_start (1) and frame_end (total_frames).
This JSON can be consumed by videos_frame_clipper.py to cut the videos.
"""

import os
import sys
import json
import argparse
import cv2

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

WORKSPACE_DIR = os.path.dirname(get_script_dir())

def parse_args():
    parser = argparse.ArgumentParser(description="Scan Microsoft_best videos and extract frame metadata.")
    parser.add_argument("--input_dir", default=os.path.join(WORKSPACE_DIR, "Microsoft_best"),
                        help="Path to the directory containing Microsoft best videos.")
    parser.add_argument("--output_json", default=os.path.join(get_script_dir(), "microsoft_best_frames.json"),
                        help="Path to save the output JSON file.")
    return parser.parse_args()

def extract_gloss_from_filename(filename):
    """
    Extracts the gloss name from the filename.
    Format: [ID]-[GLOSS].mp4 or similar.
    """
    base = os.path.splitext(filename)[0]
    if "-" in base:
        parts = base.split("-", 1)
        # parts[1] is e.g. "CONFUSED" or "CONFUSED 2"
        return parts[1].strip().upper()
    return base.strip().upper()

def get_video_info(video_path):
    """
    Opens the video file and retrieves frame count, FPS, width, and height.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}", file=sys.stderr)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return {
        "fps": fps if fps > 0 else 25.0,
        "width": width,
        "height": height,
        "total_frames": total_frames
    }

def main():
    args = parse_args()
    input_dir = args.input_dir
    output_json = args.output_json

    print("--- Microsoft Best Videos Frame Extractor ---")
    print(f"Scanning directory: {input_dir}")
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    mp4_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]
    print(f"Found {len(mp4_files)} MP4 file(s). Processing...")

    records = []
    processed = 0
    errors = 0

    for filename in sorted(mp4_files):
        video_path = os.path.join(input_dir, filename)
        info = get_video_info(video_path)
        if not info:
            errors += 1
            continue

        gloss = extract_gloss_from_filename(filename)
        
        record = {
            "gloss": gloss,
            "video_id": filename,
            "source": "microsoft",
            "frame_start": 1,
            "frame_end": info["total_frames"],
            "fps": info["fps"],
            "width": info["width"],
            "height": info["height"]
        }
        records.append(record)
        
        processed += 1
        if processed % 50 == 0 or processed == len(mp4_files):
            print(f"Processed {processed}/{len(mp4_files)} files...")

    print(f"Saving metadata to: {output_json}")
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)
        print("Success! JSON file generated.")
    except Exception as e:
        print(f"Error writing output JSON: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Summary: {processed} succeeded, {errors} failed.")

if __name__ == "__main__":
    main()

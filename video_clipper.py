"""Video Clipper and Merger Module

This module provides functionality to read ASL mapping output results,
locate corresponding sign language video clips, and concatenate them
into a single sequence video file.
"""

import os
import sys
import json
import uuid
import argparse
from pathlib import Path
from typing import Union, Dict, List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Define project directories relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MS_VIDEO_DIR = os.path.join(SCRIPT_DIR, "Microsoft_Videos")
OTHER_VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos")


def add_subtitle_to_clip(clip, text: str):
    """Draws a semi-transparent background box with the subtitle text on each frame."""
    height = clip.h
    # Determine dynamic font size based on clip height (approx 8%, min 16px)
    font_size = max(16, int(height * 0.08))
    
    def process_frame(frame):
        # frame is a numpy array of shape (H, W, 3) in RGB
        img = Image.fromarray(frame)
        
        # Try to load a clean font, fallback to default if not found
        font = None
        for font_name in ["arial.ttf", "segoeui.ttf", "calibri.ttf"]:
            try:
                font = ImageFont.truetype(font_name, size=font_size)
                break
            except IOError:
                continue
        if font is None:
            font = ImageFont.load_default()
            
        draw = ImageDraw.Draw(img)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(text, font=font)
            
        x = (img.width - text_width) // 2
        y = img.height - text_height - int(height * 0.1) # positioning at 10% from bottom
        
        # Create a transparent overlay for the background box
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        padding_x = 12
        padding_y = 6
        box_coords = [
            x - padding_x,
            y - padding_y,
            x + text_width + padding_x,
            y + text_height + padding_y
        ]
        
        try:
            overlay_draw.rounded_rectangle(box_coords, radius=4, fill=(0, 0, 0, 150))
        except AttributeError:
            overlay_draw.rectangle(box_coords, fill=(0, 0, 0, 150))
            
        overlay_draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        
        img = Image.alpha_composite(img.convert("RGBA"), overlay)
        return np.array(img.convert("RGB"))
        
    return clip.image_transform(process_frame)


def clip_and_merge_videos(
    video_records: Union[str, Path, Dict, List],
    output_path: Optional[Union[str, Path]] = None
) -> Optional[str]:
    """Clips and merges the found videos in `video_records` into a single .mp4 file.

    Parameters:
    -----------
    video_records: Union[str, Path, Dict, List]
        Can be:
        1. A string or Path pointing to the JSON output file.
        2. A dictionary matching the return value of find_video_records.
        3. A list of gloss objects (e.g., json_output).
    output_path: Optional[Union[str, Path]]
        (Optional) Path to save the merged .mp4. If not provided, it will be
        saved under the sign_out directory with the same name as the JSON file.

    Returns:
    --------
    Optional[str]
        The absolute path of the generated .mp4 file, or None if the merge failed.
    """
    json_path = None
    json_data = None

    # 1. Parse and extract video records data based on the type
    if isinstance(video_records, (str, Path)):
        candidate_path = str(video_records)
        if os.path.isfile(candidate_path):
            json_path = os.path.abspath(candidate_path)
            print(f"[Clipper] Reading JSON file from: {json_path}")
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
            except Exception as e:
                print(f"[Error] Failed to read JSON file from {json_path}: {e}", file=sys.stderr)
                return None
        else:
            # Check if it is a raw JSON string
            try:
                json_data = json.loads(candidate_path)
            except json.JSONDecodeError:
                print(
                    f"[Error] video_records string is neither a valid file path nor valid JSON data: {candidate_path}",
                    file=sys.stderr
                )
                return None
    elif isinstance(video_records, dict):
        json_data = video_records.get("json_output")
        json_path = video_records.get("output_json_path")
        
        # Fallback if find_video_records was called without writing file or structure differs
        if json_data is None:
            matching_glosses = video_records.get("matching_glosses")
            if matching_glosses:
                json_data = []
                for gloss, details in matching_glosses.items():
                    path_lower = str(details.get("video_path", "")).lower()
                    source = "microsoft" if "microsoft" in path_lower else "wlasl"
                    json_data.append({
                        "gloss": gloss,
                        "status": "found",
                        "item": [{
                            "video_id": details.get("video_id"),
                            "source": source
                        }]
                    })
    elif isinstance(video_records, list):
        json_data = video_records
    else:
        print(f"[Error] Unsupported type for video_records: {type(video_records)}", file=sys.stderr)
        return None

    if not json_data:
        print("[Error] No video records data found.", file=sys.stderr)
        return None

    # 2. Iterate through glosses and resolve video file paths
    video_paths = []
    skipped_glosses = []

    for entry in json_data:
        if not isinstance(entry, dict):
            continue
        gloss = entry.get("gloss", "UNKNOWN")
        status = str(entry.get("status", "")).lower()
        items = entry.get("item", [])

        if status != "found" or not items:
            skipped_glosses.append(f"{gloss} (missing)")
            continue

        # Use the first matched video clip for the gloss
        first_item = items[0]
        video_id = first_item.get("video_id")
        source = str(first_item.get("source", "")).lower()

        if not video_id:
            skipped_glosses.append(f"{gloss} (no video_id)")
            continue

        # Resolve paths to Microsoft or WLASL directories
        if source == "microsoft":
            full_path = os.path.abspath(os.path.join(MS_VIDEO_DIR, video_id))
        else:
            video_name = video_id if video_id.lower().endswith(".mp4") else f"{video_id}.mp4"
            full_path = os.path.abspath(os.path.join(OTHER_VIDEO_DIR, video_name))

        if os.path.isfile(full_path):
            video_paths.append((gloss, full_path))
        else:
            print(f"[Warning] Video file not found on disk: {full_path} for gloss '{gloss}'", file=sys.stderr)
            skipped_glosses.append(f"{gloss} (file not found)")

    # 3. Determine the output .mp4 file path
    out_dir = os.path.join(SCRIPT_DIR, "sign_out")
    os.makedirs(out_dir, exist_ok=True)

    if output_path:
        output_mp4_path = os.path.abspath(str(output_path))
    elif json_path:
        # Save under sign_out with the same filename as the JSON file
        json_basename = os.path.basename(json_path)
        name_without_ext = os.path.splitext(json_basename)[0]
        output_mp4_path = os.path.abspath(os.path.join(out_dir, f"{name_without_ext}.mp4"))
    else:
        # Generate default path with random UUID
        output_mp4_path = os.path.abspath(os.path.join(out_dir, f"merged_video_{uuid.uuid4().hex[:8]}.mp4"))

    # Ensure output filename ends with .mp4
    if not output_mp4_path.lower().endswith(".mp4"):
        output_mp4_path += ".mp4"

    if not video_paths:
        print("[Error] No valid video files were found to merge.", file=sys.stderr)
        return None

    print(f"[Clipper] Preparing to merge {len(video_paths)} videos:")
    for gloss, path in video_paths:
        print(f"  - {gloss} => {path}")
    if skipped_glosses:
        print(f"[Clipper] Skipped glosses: {', '.join(skipped_glosses)}")

    # 4. Import MoviePy and concatenate video clips
    try:
        # pyrefly: ignore [missing-import]
        from moviepy import VideoFileClip, concatenate_videoclips
    except ImportError as e:
        print(f"[Error] Failed to import MoviePy. Please make sure it is installed: {e}", file=sys.stderr)
        return None

    opened_clips = []
    merged_clip = None

    try:
        clips_to_merge = []
        for gloss, path in video_paths:
            print(f"[MoviePy] Loading clip: {path} for '{gloss}'")
            clip = VideoFileClip(path)
            opened_clips.append(clip)
            
            # Apply subtitle overlay
            clip_with_subtitle = add_subtitle_to_clip(clip, gloss)
            clips_to_merge.append(clip_with_subtitle)
            opened_clips.append(clip_with_subtitle)

        print("[MoviePy] Concatenating clips using method='compose'...")
        # method="compose" handles different sizes and frame rates cleanly
        merged_clip = concatenate_videoclips(clips_to_merge, method="compose")

        output_dir = os.path.dirname(output_mp4_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"[MoviePy] Writing output video file to: {output_mp4_path}")
        # audio=False is set because ASL sign clips typically do not have audio
        merged_clip.write_videofile(
            output_mp4_path,
            codec="libx264",
            audio=False,
            logger=None
        )
        print(f"[Success] Merging complete! Output video: {output_mp4_path}")
        return output_mp4_path

    except Exception as e:
        print(f"[Error] Failed during clipping and merging execution: {e}", file=sys.stderr)
        return None

    finally:
        # Guarantee all opened files are closed to avoid file locks on Windows
        print("[MoviePy] Releasing video files...")
        for clip in opened_clips:
            try:
                clip.close()
            except Exception:
                pass
        if merged_clip is not None:
            try:
                merged_clip.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Clip and merge individual ASL video files based on a video mapping JSON report."
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to the JSON output report generated by asl_llm_video_mapping.py."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output path for the merged .mp4. Defaults to the sign_out directory with the same filename as the input JSON."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.json):
        print(f"Error: The file '{args.json}' does not exist.", file=sys.stderr)
        sys.exit(1)

    result_path = clip_and_merge_videos(args.json, output_path=args.output)
    if result_path:
        print(f"Video created successfully at: {result_path}")
    else:
        print("Failed to create video.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

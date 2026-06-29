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
# pyrefly: ignore [missing-import]
from moviepy import VideoFileClip, concatenate_videoclips, ColorClip

# Define project directories relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MS_VIDEO_DIR = os.path.join(SCRIPT_DIR, "Microsoft_Videos")
OTHER_VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos")

_wlasl_v3_data = None

def load_wlasl_v3_data():
    global _wlasl_v3_data
    if _wlasl_v3_data is None:
        json_path = os.path.join(SCRIPT_DIR, "data_preprocessing", "WLASL_v03.json")
        if not os.path.isfile(json_path):
            json_path = os.path.join(SCRIPT_DIR, "data_preprocessing", "WLASL_v0.3.json")
            
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    _wlasl_v3_data = json.load(f)
                print(f"[Clipper] Loaded {len(_wlasl_v3_data)} gloss entries from {os.path.basename(json_path)}")
            except Exception as e:
                print(f"[Warning] Failed to load {os.path.basename(json_path)}: {e}", file=sys.stderr)
                _wlasl_v3_data = []
        else:
            print(f"[Warning] WLASL dataset JSON not found at {json_path}", file=sys.stderr)
            _wlasl_v3_data = []
    return _wlasl_v3_data


def resolve_best_videos(json_data):
    wlasl_v3 = load_wlasl_v3_data()
    
    # Build a set of video IDs that belong to signer_id=9 to avoid them in all parts
    blacklisted_video_ids = set()
    for item in wlasl_v3:
        for inst in item.get("instances", []):
            signer = inst.get("signer_id")
            if signer is not None and str(signer) == "9":
                vid = str(inst.get("video_id", ""))
                if vid:
                    blacklisted_video_ids.add(vid)
    
    # 1. Collect all signbank candidates on disk for each gloss
    signbank_candidates = {}
    other_wlasl_candidates = {}
    microsoft_candidates = {}
    
    for entry in json_data:
        if not isinstance(entry, dict):
            continue
        gloss = entry.get("gloss", "UNKNOWN")
        gloss_upper = gloss.upper()
        
        signbank_candidates[gloss_upper] = []
        other_wlasl_candidates[gloss_upper] = []
        microsoft_candidates[gloss_upper] = []
        
        # A. Look in WLASL_v03.json for signbank and other WLASL instances
        v3_entry = None
        for item in wlasl_v3:
            if item.get("gloss", "").upper() == gloss_upper:
                v3_entry = item
                break
                
        if v3_entry:
            for inst in v3_entry.get("instances", []):
                signer = str(inst.get("signer_id", "")) if inst.get("signer_id") is not None else None
                if signer == "9":
                    continue
                
                vid = str(inst.get("video_id", ""))
                src = str(inst.get("source", "")).lower()
                
                # Check file existence in videos directory
                vname = vid if vid.lower().endswith(".mp4") else f"{vid}.mp4"
                vpath = os.path.abspath(os.path.join(OTHER_VIDEO_DIR, vname))
                
                if os.path.isfile(vpath):
                    candidate = {"video_id": vid, "signer_id": signer, "path": vpath, "source": src}
                    if "signbank" in src:
                        signbank_candidates[gloss_upper].append(candidate)
                    else:
                        other_wlasl_candidates[gloss_upper].append(candidate)
                        
        # B. Look in the input json items (for Microsoft and fallback WLASL videos resolved by Stage 2)
        items = entry.get("item", [])
        for item in items:
            vid = str(item.get("video_id", ""))
            src = str(item.get("source", "")).lower()
            
            if src == "microsoft":
                mpath = os.path.abspath(os.path.join(MS_VIDEO_DIR, vid))
                if os.path.isfile(mpath):
                    microsoft_candidates[gloss_upper].append({"video_id": vid, "path": mpath})
            else:
                vname = vid if vid.lower().endswith(".mp4") else f"{vid}.mp4"
                vpath = os.path.abspath(os.path.join(OTHER_VIDEO_DIR, vname))
                if os.path.isfile(vpath):
                    # Skip if the video belongs to signer_id=9
                    vid_no_ext = os.path.splitext(vid)[0]
                    if vid in blacklisted_video_ids or vid_no_ext in blacklisted_video_ids:
                        continue
                        
                    # Only add if not already in our lists
                    if not any(c["video_id"] == vid for c in other_wlasl_candidates[gloss_upper]) and \
                       not any(c["video_id"] == vid for c in signbank_candidates[gloss_upper]):
                        other_wlasl_candidates[gloss_upper].append({
                            "video_id": vid,
                            "signer_id": None,
                            "path": vpath,
                            "source": src
                        })

    # 2. Determine best signer_id for Priority 1 (signers shared across multiple glosses)
    signer_coverage = {}
    for gloss_upper, candidates in signbank_candidates.items():
        seen_signers = set()
        for c in candidates:
            signer = c["signer_id"]
            if signer:
                seen_signers.add(signer)
        for signer in seen_signers:
            signer_coverage[signer] = signer_coverage.get(signer, 0) + 1
            
    best_signer_id = None
    max_coverage = 0
    for signer, count in signer_coverage.items():
        if count > max_coverage:
            max_coverage = count
            best_signer_id = signer
            
    # If the maximum coverage is only 1, we don't have multiple glosses sharing a signer,
    # so we won't enforce best_signer_id (or rather, it doesn't help prioritize sharing).
    if max_coverage <= 1:
        best_signer_id = None

    # 3. Resolve each gloss to a final video or placeholder
    resolved_paths = [] # List of tuples: (gloss, path, is_placeholder)
    
    for entry in json_data:
        if not isinstance(entry, dict):
            continue
        gloss = entry.get("gloss", "UNKNOWN")
        gloss_upper = gloss.upper()
        
        # Priority 1: videos下 & signbank & signer_id 相同的glosses 第一優先
        p1_selected = None
        if best_signer_id:
            for c in signbank_candidates[gloss_upper]:
                if c["signer_id"] == best_signer_id:
                    p1_selected = c
                    break
        
        if p1_selected:
            resolved_paths.append((gloss, p1_selected["path"], False))
            print(f"[Resolver] '{gloss}' resolved via Priority 1 (Signbank, Signer {best_signer_id}): {p1_selected['path']}")
            continue
            
        # Priority 2: videos下 & signbank 的gloss 優先
        if signbank_candidates[gloss_upper]:
            p2_selected = signbank_candidates[gloss_upper][0]
            resolved_paths.append((gloss, p2_selected["path"], False))
            print(f"[Resolver] '{gloss}' resolved via Priority 2 (Signbank): {p2_selected['path']}")
            continue
            
        # Priority 2b (implicit fallback): Any other WLASL source under videos/
        if other_wlasl_candidates[gloss_upper]:
            p2b_selected = other_wlasl_candidates[gloss_upper][0]
            resolved_paths.append((gloss, p2b_selected["path"], False))
            print(f"[Resolver] '{gloss}' resolved via Fallback WLASL: {p2b_selected['path']}")
            continue
            
        # Priority 3: 只有microsoft 有，那就使用microsoft
        if microsoft_candidates[gloss_upper]:
            p3_selected = microsoft_candidates[gloss_upper][0]
            resolved_paths.append((gloss, p3_selected["path"], False))
            print(f"[Resolver] '{gloss}' resolved via Priority 3 (Microsoft): {p3_selected['path']}")
            continue
            
        # Priority 4: 如果有missing 的gloss，顯示在subtitle (We mark as placeholder)
        resolved_paths.append((gloss, None, True))
        print(f"[Resolver] '{gloss}' is missing. Marked for placeholder.")
        
    return resolved_paths


def add_subtitle_to_clip(clip, text: str):
    """Draws a semi-transparent background box with the subtitle text on each frame."""
    height = clip.h
    # Determine dynamic font size based on clip height (approx 8%, min 16px)
    font_size = max(16, int(height * 0.08))
    
    def process_frame(frame):
        # frame is a numpy array of shape (H, W, 3) in RGB
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
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
    
    if not json_data:
        print("[Error] No video records data found.", file=sys.stderr)
        return None

    resolved_paths = resolve_best_videos(json_data)

    out_dir = os.path.join(os.getcwd(), "sign_out")
    os.makedirs(out_dir, exist_ok=True)

    if output_path:
        output_mp4_path = os.path.abspath(str(output_path))
    elif json_path:
        output_mp4_path = os.path.abspath(os.path.join(out_dir, f"{os.path.splitext(os.path.basename(json_path))[0]}.mp4"))
    else:
        output_mp4_path = os.path.abspath(os.path.join(out_dir, f"merged_{uuid.uuid4().hex[:8]}.mp4"))

    if not output_mp4_path.lower().endswith(".mp4"):
        output_mp4_path += ".mp4"

    if not resolved_paths:
        print("[Error] No video records data found to process.", file=sys.stderr)
        return None

    print(f"[Clipper] Preparing to merge {len(resolved_paths)} clips:")
    
    opened_clips = []
    merged_clip = None

    try:
        target_size = (640, 480)
        for gloss, path, is_placeholder in resolved_paths:
            if not is_placeholder and path and os.path.exists(path):
                try:
                    temp_clip = VideoFileClip(path)
                    target_size = (temp_clip.w, temp_clip.h)
                    temp_clip.close()
                    break
                except: continue

        clips_to_merge = []
        for gloss, path, is_placeholder in resolved_paths:
            if is_placeholder:
                clip = ColorClip(size=target_size, color=(0, 0, 0), duration=1.5)
                clip_with_subtitle = add_subtitle_to_clip(clip, f"{gloss} (missing)")
                clips_to_merge.append(clip_with_subtitle)
                opened_clips.append(clip_with_subtitle)
                opened_clips.append(clip)
            elif path and os.path.exists(path):
                clip = VideoFileClip(path)
                opened_clips.append(clip)
                clip_with_subtitle = add_subtitle_to_clip(clip, gloss)
                clips_to_merge.append(clip_with_subtitle)
                opened_clips.append(clip_with_subtitle)

        merged_clip = concatenate_videoclips(clips_to_merge, method="compose")
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

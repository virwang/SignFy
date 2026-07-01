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
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
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


def get_video_frame_range(video_id):
    wlasl_v3 = load_wlasl_v3_data()
    video_id_str = str(video_id)
    video_id_no_ext = os.path.splitext(video_id_str)[0]
    for item in wlasl_v3:
        for inst in item.get("instances", []):
            inst_vid = str(inst.get("video_id", ""))
            inst_vid_no_ext = os.path.splitext(inst_vid)[0]
            if inst_vid == video_id_str or inst_vid_no_ext == video_id_no_ext:
                return inst.get("frame_start"), inst.get("frame_end"), inst.get("fps")
    return None, None, None


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
                if not os.path.isfile(vpath):
                    for fb in ["videos_best", "videos_raw"]:
                        p = os.path.abspath(os.path.join(SCRIPT_DIR, fb, vname))
                        if os.path.isfile(p):
                            vpath = p
                            break
                
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
                if not os.path.isfile(mpath):
                    for fb in ["Microsoft_best", "microsoft_best", "Microsoft_Videos_raw"]:
                        p = os.path.abspath(os.path.join(SCRIPT_DIR, fb, vid))
                        if os.path.isfile(p):
                            mpath = p
                            break
                if os.path.isfile(mpath):
                    microsoft_candidates[gloss_upper].append({"video_id": vid, "path": mpath})
            else:
                vname = vid if vid.lower().endswith(".mp4") else f"{vid}.mp4"
                vpath = os.path.abspath(os.path.join(OTHER_VIDEO_DIR, vname))
                if not os.path.isfile(vpath):
                    for fb in ["videos_best", "videos_raw"]:
                        p = os.path.abspath(os.path.join(SCRIPT_DIR, fb, vname))
                        if os.path.isfile(p):
                            vpath = p
                            break
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
    resolved_paths = [] # List of tuples: (gloss, path, is_placeholder, video_id, source)
    
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
            resolved_paths.append((gloss, p1_selected["path"], False, p1_selected.get("video_id"), p1_selected.get("source")))
            print(f"[Resolver] '{gloss}' resolved via Priority 1 (Signbank, Signer {best_signer_id}): {p1_selected['path']}")
            continue
            
        # Priority 2: use signbank videos in videos/ folder
        if signbank_candidates[gloss_upper]:
            p2_selected = signbank_candidates[gloss_upper][0]
            resolved_paths.append((gloss, p2_selected["path"], False, p2_selected.get("video_id"), p2_selected.get("source")))
            print(f"[Resolver] '{gloss}' resolved via Priority 2 (Signbank): {p2_selected['path']}")
            continue
            
        # Priority 2b (implicit fallback): Any other WLASL source under videos/
        if other_wlasl_candidates[gloss_upper]:
            p2b_selected = other_wlasl_candidates[gloss_upper][0]
            resolved_paths.append((gloss, p2b_selected["path"], False, p2b_selected.get("video_id"), p2b_selected.get("source")))
            print(f"[Resolver] '{gloss}' resolved via Fallback WLASL: {p2b_selected['path']}")
            continue
            
        # Priority 3: use microsoft video when the gloss only exits in microsoft dataset
        if microsoft_candidates[gloss_upper]:
            p3_selected = microsoft_candidates[gloss_upper][0]
            resolved_paths.append((gloss, p3_selected["path"], False, p3_selected.get("video_id"), "microsoft"))
            print(f"[Resolver] '{gloss}' resolved via Priority 3 (Microsoft): {p3_selected['path']}")
            continue
            
        # Priority 4: if the gloss is missing, show at subtitle
        resolved_paths.append((gloss, None, True, None, None))
        print(f"[Resolver] '{gloss}' is missing.")
        
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
        for gloss, path, is_placeholder, _, _ in resolved_paths:
            if not is_placeholder and path and os.path.exists(path):
                try:
                    temp_clip = VideoFileClip(path)
                    target_size = (temp_clip.w, temp_clip.h)
                    temp_clip.close()
                    break
                except: continue

        clips_to_merge = []
        for gloss, path, is_placeholder, video_id, source in resolved_paths:
            if is_placeholder:
                clip = ColorClip(size=target_size, color=(0, 0, 0), duration=1.5)
                clip_with_subtitle = add_subtitle_to_clip(clip, f"{gloss} (missing)")
                clips_to_merge.append(clip_with_subtitle)
                opened_clips.append(clip_with_subtitle)
                opened_clips.append(clip)
            elif path and os.path.exists(path):
                clip = VideoFileClip(path)
                opened_clips.append(clip)
                
                # If the source is not microsoft, we need to read the WLASL dataset JSON
                # to get the frame_start and frame_end, and clip the video.
                if source and source.lower() != "microsoft":
                    frame_start, frame_end, fps = get_video_frame_range(video_id)
                    if frame_start is not None and frame_end is not None:
                        video_fps = fps if (fps and fps > 0) else (clip.fps if clip.fps else 25.0)
                        
                        t_start = 0.0
                        if frame_start > 1:
                            t_start = (frame_start - 1) / video_fps
                        
                        t_end = None
                        if frame_end > 0:
                            t_end = frame_end / video_fps
                            
                        # Ensure time points do not exceed the video's actual duration
                        if clip.duration:
                            if t_start >= clip.duration:
                                t_start = 0.0
                            if t_end is not None:
                                if t_end > clip.duration or t_end <= t_start:
                                    t_end = clip.duration
                                    
                        print(f"[Clipper] Clipping '{gloss}' ({video_id}) from frame {frame_start} to {frame_end} "
                              f"(t_start={t_start:.2f}s, t_end={f'{t_end:.2f}s' if t_end is not None else 'None'}) at {video_fps} fps")
                        try:
                            if hasattr(clip, "subclipped"):
                                clipped_clip = clip.subclipped(t_start, t_end)
                            else:
                                clipped_clip = clip.subclip(t_start, t_end)
                            clip = clipped_clip
                            opened_clips.append(clip)
                        except Exception as subclip_err:
                            print(f"[Warning] Failed to subclip {video_id}: {subclip_err}", file=sys.stderr)
                            
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
            if engine == 'opencv':
                success, msg = clip_video_opencv(src_path, dst_path, frame_start, frame_end)
                if not success:
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


def clip_single_video(input_file, output_dir=None, frame_start=None, frame_end=None, wlasl_json=None, engine='opencv', copy_uncut=True):
    """
    Clips or copies a single video file based on explicit arguments or JSON lookup.
    """
    src_path = input_file
    if not os.path.exists(src_path):
        print(f"ERROR: Input file not found: {src_path}")
        return False
        
    filename = os.path.basename(src_path)
    
    # If frame boundaries are not provided, try to load WLASL JSON
    if frame_start is None or frame_end is None:
        if wlasl_json is None:
            wlasl_json = os.path.join(SCRIPT_DIR, "data_preprocessing", "WLASL_v03.json")
            if not os.path.exists(wlasl_json):
                wlasl_json = os.path.join(SCRIPT_DIR, "data_preprocessing", "WLASL_v0.3.json")
                
        if os.path.exists(wlasl_json):
            print(f"Loading WLASL metadata from: {wlasl_json}")
            wlasl_mapping = load_wlasl_mapping(wlasl_json)
            
            video_id_str = os.path.splitext(filename)[0]
            wlasl_info = wlasl_mapping.get(filename) or wlasl_mapping.get(video_id_str)
            
            if wlasl_info:
                if frame_start is None:
                    frame_start = wlasl_info.get("frame_start")
                if frame_end is None:
                    frame_end = wlasl_info.get("frame_end")
                print(f"Found WLASL frame metadata for {filename}: frame_start={frame_start}, frame_end={frame_end}")
            else:
                print(f"WARNING: WLASL frame metadata not found for {filename}.")
        else:
            print(f"WARNING: WLASL json not found at {wlasl_json}.")
            
    # Set default values if still None
    if frame_start is None:
        frame_start = 1
    if frame_end is None:
        frame_end = -1
        
    # Determine output directory
    if not output_dir:
        parent_name = os.path.basename(os.path.dirname(os.path.abspath(src_path))).lower()
        if "microsoft" in parent_name:
            output_dir = os.path.join(SCRIPT_DIR, "microsoft_cut")
        else:
            output_dir = os.path.join(SCRIPT_DIR, "videos_cut")
        print(f"No output_dir specified. Defaulting to: {output_dir}")
        
    os.makedirs(output_dir, exist_ok=True)
    dst_path = os.path.join(output_dir, filename)
    
    requires_clipping = (frame_start > 1) or (frame_end != -1)
    
    print(f"Processing single video: {src_path} -> {dst_path}")
    print(f"Range: frame_start={frame_start}, frame_end={frame_end} (requires_clipping={requires_clipping})")
    
    action, success, msg = process_video_task(
        src_path,
        dst_path,
        frame_start,
        frame_end,
        requires_clipping,
        copy_uncut,
        engine
    )
    
    if success:
        print(f"SUCCESS: Single file process completed ({action}): {msg}")
        return True
    else:
        print(f"ERROR: Single file process failed: {msg}")
        return False


def batch_clip_videos(video_dirs=None, output_dirs=None, split_json=None, wlasl_json=None, engine='opencv', copy_uncut=True, num_workers=8):
    """
    Clips or copies a batch of videos based on a split JSON mapping and WLASL metadata.
    """
    # Setup default paths
    default_video_dirs = [
        os.path.join(SCRIPT_DIR, "videos_raw"),
        os.path.join(SCRIPT_DIR, "Microsoft_Videos_raw")
    ]
    default_output_dirs = [
        os.path.join(SCRIPT_DIR, "videos_cut"),
        os.path.join(SCRIPT_DIR, "microsoft_cut")
    ]
    if split_json is None:
        split_json = os.path.join(SCRIPT_DIR, "data_preprocessing", "best_asl_videos.json")
    if wlasl_json is None:
        wlasl_json = os.path.join(SCRIPT_DIR, "data_preprocessing", "WLASL_v03.json")
        if not os.path.exists(wlasl_json):
            wlasl_json = os.path.join(SCRIPT_DIR, "data_preprocessing", "WLASL_v0.3.json")
            
    video_dirs = video_dirs if video_dirs else default_video_dirs
    output_dirs = output_dirs if output_dirs else default_output_dirs
    
    other_video_dir = video_dirs[0] if len(video_dirs) > 0 else default_video_dirs[0]
    ms_video_dir = video_dirs[1] if len(video_dirs) > 1 else default_video_dirs[1]
    
    other_cut_dir = output_dirs[0] if len(output_dirs) > 0 else default_output_dirs[0]
    ms_cut_dir = output_dirs[1] if len(output_dirs) > 1 else default_output_dirs[1]
    
    print("--- Batch Video Clipper ---")
    if not os.path.exists(split_json):
        print(f"ERROR: Split JSON not found: {split_json}")
        return False
        
    with open(split_json, "r", encoding="utf-8") as f:
        best_videos = json.load(f)
        
    if isinstance(best_videos, dict):
        videos_list = list(best_videos.values())
    else:
        videos_list = best_videos
        
    print(f"Loaded {len(videos_list)} videos from {os.path.basename(split_json)}.")
    
    print(f"Loading WLASL metadata from: {wlasl_json}")
    wlasl_mapping = load_wlasl_mapping(wlasl_json)
    
    os.makedirs(other_cut_dir, exist_ok=True)
    os.makedirs(ms_cut_dir, exist_ok=True)
    
    stats = {"clip_success": 0, "clip_fail": 0, "copy_success": 0, "copy_fail": 0, "skipped": 0, "missing": 0}
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for item in videos_list:
            video_id = item.get("video_id")
            gloss = item.get("gloss", "unknown").upper()
            source = item.get("source", "other").lower()
            
            if not video_id:
                continue
                
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
                
            dst_filename = os.path.basename(src_path)
            dst_path = os.path.join(target_dir, dst_filename)
            
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
                        
            future = executor.submit(
                process_video_task,
                src_path,
                dst_path,
                frame_start,
                frame_end,
                requires_clipping,
                copy_uncut,
                engine
            )
            futures[future] = (gloss, video_id_str, requires_clipping)
            
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
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Clip and merge individual ASL video files based on a video mapping JSON report."
    )
    parser.add_argument(
        "--json",
        type=str,
        required=False,
        help="Path to the JSON output report generated by asl_llm_video_mapping.py."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output path for the merged .mp4. Defaults to the sign_out directory with the same filename as the input JSON."
    )
    
    # Single-file testing arguments
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to a single input video file to clip."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory to save the clipped single video."
    )
    parser.add_argument(
        "--frame_start",
        type=int,
        default=None,
        help="Start frame for clipping (1-indexed)."
    )
    parser.add_argument(
        "--frame_end",
        type=int,
        default=None,
        help="End frame for clipping."
    )
    parser.add_argument(
        "--wlasl_json",
        type=str,
        default=None,
        help="Path to WLASL_v03.json dataset file."
    )
    parser.add_argument(
        "--engine",
        choices=["opencv", "moviepy"],
        default="moviepy",
        help="Primary engine for video clipping (default: moviepy)"
    )
    parser.add_argument(
        "--no_copy_uncut",
        action="store_true",
        help="Do not copy files that don't need clipping"
    )
    args = parser.parse_args()

    if args.input_file:
        clip_single_video(
            input_file=args.input_file,
            output_dir=args.output_dir,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            wlasl_json=args.wlasl_json,
            engine=args.engine,
            copy_uncut=not args.no_copy_uncut
        )
    elif args.json:
        if not os.path.isfile(args.json):
            print(f"Error: The file '{args.json}' does not exist.", file=sys.stderr)
            sys.exit(1)
        result_path = clip_and_merge_videos(args.json, output_path=args.output)
        if result_path:
            print(f"Video created successfully at: {result_path}")
        else:
            print("Failed to create video.", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

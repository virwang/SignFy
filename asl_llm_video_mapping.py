"""_summary_
        This script performs the following tasks:
        1. Receive the translated english text input and the corresponding ASL gloss output from the Llama model.
        2. Load the dataset and check if the corresponding videos exist in the Microsoft and WLASL video directories.
        3. Generate a comprehensive report in Excel format, categorizing the results into "Found" and "Missing" videos, and providing statistics on the distribution of glosses across sources.
        4. The script is optimized for performance by caching video filenames in memory to avoid repeated disk access, and it handles edge cases such as missing fields in the input JSON gracefully.   
"""
import argparse
import json 
import os
import re
import sys
import uuid

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from typing import List, Dict, Tuple

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPLING_PATH = os.path.join(SCRIPT_DIR, "data_preprocessing", "best_asl_videos.json")
MS_VIDEO_DIR = os.path.join(SCRIPT_DIR, "microsoft_cut")
OTHER_VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos_cut")

# In-memory video file caches
_ms_video_cache = None
_other_video_cache = None

def get_video_caches(ms_dir=MS_VIDEO_DIR, other_dir=OTHER_VIDEO_DIR):
    """
    Retrieves or builds in-memory caches of video filenames (without extension).
    """
    global _ms_video_cache, _other_video_cache
    if _ms_video_cache is None or _other_video_cache is None:
        _ms_video_cache = set()
        _other_video_cache = set()
        
        if os.path.isdir(ms_dir):
            for f in os.listdir(ms_dir):
                if f.lower().endswith(".mp4"):
                    _ms_video_cache.add(os.path.splitext(f)[0])
                    
        if os.path.isdir(other_dir):
            for f in os.listdir(other_dir):
                if f.lower().endswith(".mp4"):
                    _other_video_cache.add(os.path.splitext(f)[0])
                    
        print(f"[Cache] Loaded {len(_ms_video_cache)} Microsoft videos_cut and {len(_other_video_cache)} WLASL videos_cut in memory.")
        
    return _ms_video_cache, _other_video_cache

def load_video_filenames(file_path=None):
    if file_path is None:
        file_path = MAPPLING_PATH
        
    # If path doesn't exist, try resolving relative to SCRIPT_DIR
    if not os.path.isfile(file_path):
        resolved_path = os.path.join(SCRIPT_DIR, file_path)
        if os.path.isfile(resolved_path):
            file_path = resolved_path
        else:
            # Try looking directly inside data_preprocessing
            resolved_path = os.path.join(SCRIPT_DIR, "data_preprocessing", os.path.basename(file_path))
            if os.path.isfile(resolved_path):
                file_path = resolved_path
                
    print(f"Loading mapping data from {file_path}...")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Mapping file {file_path} not found.")
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data

_valid_gloss_cache = None

def get_valid_glosses() -> set:
    """
    Returns a cached set of all uppercase gloss names from the mapping file.
    """
    global _valid_gloss_cache
    if _valid_gloss_cache is None:
        try:
            data = load_video_filenames()
            if isinstance(data, dict):
                _valid_gloss_cache = {
                    str(k).upper() for k in data.keys()
                }
            elif isinstance(data, list):
                _valid_gloss_cache = {
                    entry.get("gloss", "").upper()
                    for entry in data
                    if isinstance(entry, dict) and entry.get("gloss")
                }
            else:
                _valid_gloss_cache = set()
        except Exception as e:
            print(f"[Warning] Failed to load valid gloss cache: {e}", file=sys.stderr)
            _valid_gloss_cache = set()
    return _valid_gloss_cache

def is_gloss_in_db(gloss: str) -> bool:
    """
    Checks if the gloss is present in the best_asl_videos.json mapping file.
    """
    if not gloss:
        return False
    return gloss.strip().upper() in get_valid_glosses()

def _normalize_llm_output(llm_output) -> list:
    """
    Normalizes llm_output to a list of (primary_gloss, fallback_gloss) tuples.
    Supports both list of strings and list of dicts.
    """
    pairs = []
    for item in llm_output:
        if not item:
            continue
        if isinstance(item, dict):
            primary = str(item.get("gloss", "")).strip().upper()
            fallback = str(item.get("fallback", "")).strip().upper()
            if primary:
                pairs.append((primary, fallback or primary))
        else:
            token = str(item).strip().upper()
            if token:
                pairs.append((token, token))
    return pairs

def find_matching_glosses(json_data, llm_output):
    """
    Find matching glosses and return a dictionary mapping each gloss to its first found video path & ID.
    Supports fallback synonyms if primary is missing.
    """
    ms_cache, other_cache = get_video_caches()
    
    gloss_index = {}
    if isinstance(json_data, dict):
        for gloss, entry in json_data.items():
            if not isinstance(entry, dict):
                continue
            gloss_upper = gloss.upper()
            video_id = entry.get('video_id')
            if not video_id:
                continue
            video_id = str(video_id)
            source = str(entry.get('source', '')).lower()
            
            if source == 'microsoft':
                video_name = os.path.splitext(video_id)[0]
                is_found = video_name in ms_cache
            else:
                is_found = video_id in other_cache
                
            entry_copy = entry.copy()
            entry_copy['status'] = 'Found' if is_found else 'Missing'
            if is_found:
                gloss_index[gloss_upper] = [entry_copy]
    else:
        for entry in json_data:
            if not isinstance(entry, dict):
                continue
            gloss = str(entry.get('gloss', '')).upper()
            if not gloss:
                continue
            items = entry.get('item')
            if not isinstance(items, list):
                continue
                
            found_items = []
            for i in items:
                if not isinstance(i, dict):
                    continue
                video_id = i.get('video_id')
                if not video_id:
                    continue
                video_id = str(video_id)
                source = str(i.get('source', '')).lower()
                
                if source == 'microsoft':
                    video_name = os.path.splitext(video_id)[0]
                    is_found = video_name in ms_cache
                else:
                    is_found = video_id in other_cache
                    
                i['status'] = 'Found' if is_found else 'Missing'
                if is_found:
                    found_items.append(i)
                    
            if found_items:
                gloss_index[gloss] = found_items

    matching_glosses = {}
    normalized_pairs = _normalize_llm_output(llm_output)
    
    for primary, fallback in normalized_pairs:
        # Determine which gloss to use (primary, or fallback if primary is missing/not found)
        chosen_gloss = None
        if primary in gloss_index:
            chosen_gloss = primary
        elif fallback in gloss_index:
            chosen_gloss = fallback
            
        if chosen_gloss:
            first_found_item = gloss_index[chosen_gloss][0]
            source = first_found_item.get('source', '').lower()
            video_id = first_found_item.get('video_id', 'N/A')

            if source == 'microsoft':
                video_path = "./microsoft_cut/"
            else:
                video_path = "./videos_cut/"

            # Maintain original primary gloss as key in output mapping for Stage 3 pipeline integration
            matching_glosses[primary] = {
                "video_id": video_id,
                "video_path": video_path
            }

    return matching_glosses

def record_not_found_glosses(json_data, llm_output):
    """
    Retrieve primary glosses in llm_output that have no found videos_cut (neither primary nor fallback found).
    """
    ms_cache, other_cache = get_video_caches()
    found_glosses = set()
    
    if isinstance(json_data, dict):
        for gloss, entry in json_data.items():
            if not isinstance(entry, dict):
                continue
            gloss_upper = gloss.upper()
            video_id = entry.get('video_id')
            if not video_id:
                continue
            video_id = str(video_id)
            source = str(entry.get('source', '')).lower()
            
            if source == 'microsoft':
                video_name = os.path.splitext(video_id)[0]
                is_found = video_name in ms_cache
            else:
                is_found = video_id in other_cache
                
            if is_found:
                found_glosses.add(gloss_upper)
    else:
        for entry in json_data:
            if not isinstance(entry, dict):
                continue
            gloss = str(entry.get('gloss', '')).upper()
            if not gloss:
                continue
            items = entry.get('item')
            if not isinstance(items, list):
                continue
                
            for i in items:
                if not isinstance(i, dict):
                    continue
                video_id = i.get('video_id')
                if not video_id:
                    continue
                video_id = str(video_id)
                source = str(i.get('source', '')).lower()
                
                if source == 'microsoft':
                    video_name = os.path.splitext(video_id)[0]
                    is_found = video_name in ms_cache
                else:
                    is_found = video_id in other_cache
                    
                if is_found:
                    found_glosses.add(gloss)
                    break
                
    normalized_pairs = _normalize_llm_output(llm_output)
    not_found_glosses = []
    for primary, fallback in normalized_pairs:
        # If neither primary nor fallback has any found video, it's not found
        if primary not in found_glosses and fallback not in found_glosses:
            not_found_glosses.append(primary)
            
    return not_found_glosses

def find_video_records(llm_output, english_input=None, output_excel_path="asl_mapping_report.xlsx", output_json_path="gloss_video_mapping_output/asl_mapping_report.json"):
    """
    Performs video matching, caches video filenames in memory, handles missing fields in input JSON gracefully,
    and generates a styled Excel report as well as a JSON report for logging and issue tracking.
    """
    normalized_pairs = _normalize_llm_output(llm_output)

    if output_json_path:
        base_name = os.path.splitext(os.path.basename(output_json_path))[0]
        
        clean_glosses = [g for g, _ in normalized_pairs]
        gloss_part = "_".join(clean_glosses) if clean_glosses else "mapping"
        if len(gloss_part) > 100:
            gloss_part = gloss_part[:100]
            
        if base_name == "asl_mapping_report":
            out_dir = os.path.join(SCRIPT_DIR, "gloss_video_mapping_output")
            filename = f"{gloss_part}_{uuid.uuid4()}.json"
        else:
            out_dir = os.path.dirname(os.path.abspath(output_json_path))
            filename = f"{base_name}_{uuid.uuid4()}.json"
            
        output_json_path = os.path.join(out_dir, filename)

    try:
        json_data = load_video_filenames()
    except Exception as e:
        print(f"Error loading video filenames: {e}", file=sys.stderr)
        json_data = []

    ms_cache, other_cache = get_video_caches()

    # Pre-process the JSON data to update/verify status dynamically using caches
    gloss_index = {}
    if isinstance(json_data, dict):
        for gloss, entry in json_data.items():
            if not isinstance(entry, dict):
                continue
            gloss_upper = gloss.upper()
            video_id = entry.get('video_id')
            if not video_id:
                continue
            video_id = str(video_id)
            source = str(entry.get('source', '')).lower()
            
            if source == 'microsoft':
                video_name = os.path.splitext(video_id)[0]
                is_found = video_name in ms_cache
            else:
                is_found = video_id in other_cache
                
            entry_copy = entry.copy()
            entry_copy['status'] = 'Found' if is_found else 'Missing'
            gloss_index[gloss_upper] = [entry_copy]
    else:
        for entry in json_data:
            if not isinstance(entry, dict):
                continue
            gloss = str(entry.get('gloss', '')).upper()
            if not gloss:
                continue
            items = entry.get('item')
            if not isinstance(items, list):
                continue
                
            for i in items:
                if not isinstance(i, dict):
                    continue
                video_id = i.get('video_id')
                if not video_id:
                    continue
                video_id = str(video_id)
                source = str(i.get('source', '')).lower()
                
                if source == 'microsoft':
                    video_name = os.path.splitext(video_id)[0]
                    is_found = video_name in ms_cache
                else:
                    is_found = video_id in other_cache
                    
                i['status'] = 'Found' if is_found else 'Missing'
                
            gloss_index[gloss] = items

    matching_glosses = {}
    found_videos_list = []
    missing_videos_list = []
    source_stats = {}
    json_output_data = []

    unique_glosses_found_count = 0
    unique_glosses_missing_count = 0

    for primary, fallback in normalized_pairs:
        gloss_upper = primary
        fallback_upper = fallback
        
        # Check if primary exists and has found videos_cut
        primary_has_found = False
        if gloss_upper in gloss_index:
            primary_has_found = any(i.get('status') == 'Found' for i in gloss_index[gloss_upper])
            
        # Determine which gloss/items to map to
        chosen_gloss = gloss_upper
        is_fallback_used = False
        
        if primary_has_found:
            items = gloss_index[gloss_upper]
            chosen_gloss = gloss_upper
        elif fallback_upper in gloss_index and any(i.get('status') == 'Found' for i in gloss_index[fallback_upper]):
            items = gloss_index[fallback_upper]
            chosen_gloss = fallback_upper
            is_fallback_used = True
        elif gloss_upper in gloss_index:
            items = gloss_index[gloss_upper]
            chosen_gloss = gloss_upper
        elif fallback_upper in gloss_index:
            items = gloss_index[fallback_upper]
            chosen_gloss = fallback_upper
            is_fallback_used = True
        else:
            items = None
            chosen_gloss = gloss_upper
            
        gloss_found_any_video = False
        items_list = []
        display_gloss = primary if chosen_gloss == primary else f"{primary} ({fallback})"
        
        if items is not None:
            for item in items:
                video_id = item.get('video_id', 'N/A')
                source = item.get('source', 'unknown').lower()
                status = item.get('status', 'Missing')
                
                if source not in source_stats:
                    source_stats[source] = {"total": 0, "found": 0, "missing": 0}
                source_stats[source]["total"] += 1
                
                if status == 'Found':
                    gloss_found_any_video = True
                    source_stats[source]["found"] += 1
                    
                    if source == 'microsoft':
                        dir_path = "./microsoft_cut/"
                        full_path = os.path.abspath(os.path.join(MS_VIDEO_DIR, video_id))
                    else:
                        dir_path = "./videos_cut/"
                        actual_filename = video_id
                        if not actual_filename.lower().endswith(".mp4"):
                            actual_filename += ".mp4"
                        full_path = os.path.abspath(os.path.join(OTHER_VIDEO_DIR, actual_filename))
                        
                    found_videos_list.append({
                        "Gloss": display_gloss,
                        "Video ID": video_id,
                        "Source": source,
                        "Directory": dir_path,
                        "Full Path": full_path
                    })
                else:
                    source_stats[source]["missing"] += 1
                    if source == 'microsoft':
                        exp_dir = "./microsoft_cut/"
                    else:
                        exp_dir = "./videos_cut/"
                        
                    missing_videos_list.append({
                        "Gloss": display_gloss,
                        "Video ID": video_id,
                        "Source": source,
                        "Expected Directory": exp_dir
                    })
                
                # Append to items_list for JSON output
                items_list.append({
                    "video_id": video_id,
                    "source": source
                })
            
            # Populate matching_glosses with the first found item/video for pipeline Stage 3
            found_items = [i for i in items if i.get('status') == 'Found']
            if found_items:
                first_found_item = found_items[0]
                source = first_found_item.get('source', '').lower()
                video_id = first_found_item.get('video_id', 'N/A')

                if source == 'microsoft':
                    video_path = "./microsoft_cut/"
                else:
                    video_path = "./videos_cut/"

                matching_glosses[primary] = {
                    "video_id": video_id,
                    "video_path": video_path
                }
        else:
            # Gloss not in database at all
            missing_videos_list.append({
                "Gloss": display_gloss,
                "Video ID": "N/A (Gloss not in DB)",
                "Source": "N/A",
                "Expected Directory": "N/A"
            })
            
        if gloss_found_any_video:
            unique_glosses_found_count += 1
        else:
            unique_glosses_missing_count += 1

        # Populate JSON structure: {'gloss':'', 'status':'found'/'missing', 'item':[{'video_id':'','source':''}]}
        json_output_data.append({
            "gloss": primary,
            "fallback": fallback,
            "chosen_gloss": chosen_gloss,
            "fallback_used": is_fallback_used,
            "status": "found" if gloss_found_any_video else "missing",
            "item": items_list
        })

    # DataFrames for Excel
    df_found = pd.DataFrame(found_videos_list)
    if df_found.empty:
        df_found = pd.DataFrame(columns=["Gloss", "Video ID", "Source", "Directory", "Full Path"])
        
    df_missing = pd.DataFrame(missing_videos_list)
    if df_missing.empty:
        df_missing = pd.DataFrame(columns=["Gloss", "Video ID", "Source", "Expected Directory"])

    stats_rows = []
    for src, counts in source_stats.items():
        total = counts["total"]
        found = counts["found"]
        missing = counts["missing"]
        success_rate = (found / total * 100) if total > 0 else 0.0
        stats_rows.append({
            "Source": src,
            "Total Videos": total,
            "Found Videos": found,
            "Missing Videos": missing,
            "Success Rate (%)": f"{success_rate:.2f}%"
        })
    df_stats = pd.DataFrame(stats_rows)
    if df_stats.empty:
        df_stats = pd.DataFrame(columns=["Source", "Total Videos", "Found Videos", "Missing Videos", "Success Rate (%)"])

    summary_data = {
        "Metric": [
            "English Input",
            "ASL Gloss Output",
            "Total Unique Glosses",
            "Unique Glosses Found",
            "Unique Glosses Missing",
            "Total Videos Checked",
            "Total Videos Found",
            "Total Videos Missing"
        ],
        "Value": [
            english_input if english_input else "N/A",
            " ".join(g for g, _ in normalized_pairs),
            len(set(g for g, _ in normalized_pairs)),
            unique_glosses_found_count,
            unique_glosses_missing_count,
            len(found_videos_list) + len([v for v in missing_videos_list if v["Video ID"] != "N/A (Gloss not in DB)"]),
            len(found_videos_list),
            len([v for v in missing_videos_list if v["Video ID"] != "N/A (Gloss not in DB)"])
        ]
    }
    df_summary = pd.DataFrame(summary_data)

    # Save to Excel
    if output_excel_path:
        try:
            out_dir = os.path.dirname(os.path.abspath(output_excel_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                
            with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
                df_summary.to_excel(writer, sheet_name="Summary", index=False)
                df_found.to_excel(writer, sheet_name="Found Videos", index=False)
                df_missing.to_excel(writer, sheet_name="Missing Videos", index=False)
                df_stats.to_excel(writer, sheet_name="Source Statistics", index=False)

            wb = load_workbook(output_excel_path)
            
            # Header Styling: Dark Navy (1F4E78) with white bold text
            header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
            header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
            
            body_font = Font(name="Segoe UI", size=10)
            bold_body_font = Font(name="Segoe UI", size=10, bold=True)
            
            align_center = Alignment(horizontal="center", vertical="center")
            align_left = Alignment(horizontal="left", vertical="center")
            
            thin_border_side = Side(border_style="thin", color="D9D9D9")
            thin_border = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)
            
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                ws.views.sheetView[0].showGridLines = True  # Enable grid lines!
                
                for col_idx in range(1, ws.max_column + 1):
                    cell = ws.cell(row=1, column=col_idx)
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = align_center
                    cell.border = thin_border
                    
                for row_idx in range(2, ws.max_row + 1):
                    for col_idx in range(1, ws.max_column + 1):
                        cell = ws.cell(row=row_idx, column=col_idx)
                        cell.font = body_font
                        cell.border = thin_border
                        
                        if sheet_name == "Summary":
                            if col_idx == 1:
                                cell.alignment = align_left
                                cell.font = bold_body_font
                            else:
                                cell.alignment = align_left
                        elif sheet_name == "Source Statistics":
                            if col_idx == 1:
                                cell.alignment = align_left
                            else:
                                cell.alignment = align_center
                        else:
                            if col_idx in [1, 2, 3]:  # Gloss, Video ID, Source
                                cell.alignment = align_center
                            else:
                                cell.alignment = align_left
                                
                for col in ws.columns:
                    max_len = 0
                    col_letter = col[0].column_letter
                    for cell in col:
                        val = str(cell.value or '')
                        if len(val) > max_len:
                            max_len = len(val)
                    ws.column_dimensions[col_letter].width = max(max_len + 4, 12)
                    
            wb.save(output_excel_path)
            print(f"Comprehensive Excel report saved successfully to: {output_excel_path}")
        except Exception as e:
            print(f"Error generating Excel report: {e}", file=sys.stderr)

    # Save to JSON
    if output_json_path:
        try:
            out_dir = os.path.dirname(os.path.abspath(output_json_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(json_output_data, f, indent=4, ensure_ascii=False)
            print(f"JSON logging report saved successfully to: {output_json_path}")
        except Exception as e:
            print(f"Error generating JSON report: {e}", file=sys.stderr)

    return {
        "matching_glosses": matching_glosses,
        "json_output": json_output_data,
        "output_json_path": output_json_path
    }

def main():
    parser = argparse.ArgumentParser(
        description="Verify and map translated ASL Glosses to dataset video files, generating a comprehensive Excel report."
    )
    parser.add_argument(
        "--english",
        type=str,
        required=True,
        help="The translated English sentence input."
    )
    parser.add_argument(
        "--gloss",
        type=str,
        required=True,
        help="The corresponding ASL gloss output (space-separated tokens)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="asl_mapping_report.xlsx",
        help="Path to save the generated Excel report. Default: asl_mapping_report.xlsx"
    )
    parser.add_argument(
        "--json",
        type=str,
        default="gloss_video_mapping_output/asl_mapping_report.json",
        help="Path to save the generated JSON report. Default: gloss_video_mapping_output/asl_mapping_report.json"
    )
    args = parser.parse_args()

    gloss_array = [g.strip() for g in args.gloss.split(" ") if g.strip()]
    print(f"English Input: {args.english}")
    print(f"ASL Glosses: {gloss_array}")

    results = find_video_records(
        llm_output=gloss_array,
        english_input=args.english,
        output_excel_path=args.output,
        output_json_path=args.json
    )
    
    print("\nMapping Results:")
    for gloss, details in results["matching_glosses"].items():
        print(f"  {gloss} => ID: {details['video_id']}, Path: {details['video_path']}")

if __name__ == "__main__":
    main()

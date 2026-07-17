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
NUMPY_DIR = os.path.join(SCRIPT_DIR, os.getenv("ASL_NUMPY_DIR", "videos_numpy"))

# In-memory video file caches
_numpy_cache = None

def get_numpy_cache(numpy_dir=NUMPY_DIR):
    """
    Retrieves or builds in-memory cache of numpy filenames (without extension).
    """
    global _numpy_cache
    if _numpy_cache is None:
        _numpy_cache = set()
        
        if os.path.isdir(numpy_dir):
            for f in os.listdir(numpy_dir):
                if f.lower().endswith(".npy"):
                    _numpy_cache.add(os.path.splitext(f)[0])
                    
        print(f"[Cache] Loaded {len(_numpy_cache)} numpy arrays from {numpy_dir} in memory.")
        
    return _numpy_cache

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
    Normalizes llm_output to a list of (primary_gloss, list_of_synonyms) tuples.
    Supports both list of strings and list of dicts.
    """
    pairs = []
    for item in llm_output:
        if not item:
            continue
        if isinstance(item, dict):
            primary = str(item.get("gloss", "")).strip().upper()
            if "synonyms" in item:
                syns = item.get("synonyms", [])
                if not isinstance(syns, list):
                    syns = [str(syns).strip().upper()]
                else:
                    syns = [str(s).strip().upper() for s in syns]
            elif "fallback" in item:
                syns = [str(item.get("fallback", "")).strip().upper()]
            else:
                syns = []
            if primary:
                pairs.append((primary, syns))
        else:
            token = str(item).strip().upper()
            if token:
                pairs.append((token, []))
    return pairs

def find_matching_glosses(json_data, llm_output):
    """
    Find matching glosses and return a dictionary mapping each gloss to its first found video path & ID.
    Supports fallback synonyms if primary is missing.
    """
    numpy_cache = get_numpy_cache()
    
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
            
            video_name = os.path.splitext(video_id)[0]
            is_found = video_name in numpy_cache
                
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
                
                video_name = os.path.splitext(video_id)[0]
                is_found = video_name in numpy_cache
                    
                i['status'] = 'Found' if is_found else 'Missing'
                if is_found:
                    found_items.append(i)
                    
            if found_items:
                gloss_index[gloss] = found_items

    matching_glosses = {}
    normalized_pairs = _normalize_llm_output(llm_output)
    
    for primary, synonyms in normalized_pairs:
        # Determine which gloss to use (primary, or fallback if primary is missing/not found)
        chosen_gloss = None
        if primary in gloss_index:
            chosen_gloss = primary
        else:
            for syn in synonyms:
                if syn in gloss_index:
                    chosen_gloss = syn
                    break
            
        if chosen_gloss:
            first_found_item = gloss_index[chosen_gloss][0]
            source = first_found_item.get('source', '').lower()
            video_id = first_found_item.get('video_id', 'N/A')

            video_path = f"./{os.path.basename(NUMPY_DIR)}/"

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
    for primary, synonyms in normalized_pairs:
        # If neither primary nor fallback has any found video, it's not found
        if primary not in found_glosses and not any(syn in found_glosses for syn in synonyms):
            not_found_glosses.append(primary)
            
    return not_found_glosses

def find_video_records(llm_output, english_input=None, output_excel_path="asl_mapping_report.xlsx", output_json_path="gloss_video_mapping_output/asl_mapping_report.json"):
    """
    Performs video matching, caches video filenames in memory, handles missing fields in input JSON gracefully,
    and generates a styled Excel report as well as a JSON report for logging and issue tracking.
    """
    normalized_pairs = _normalize_llm_output(llm_output)

    # JSON output path generation logic has been removed.

    try:
        json_data = load_video_filenames()
    except Exception as e:
        print(f"Error loading video filenames: {e}", file=sys.stderr)
        json_data = []

    numpy_cache = get_numpy_cache()

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
            
            video_name = os.path.splitext(video_id)[0]
            is_found = video_name in numpy_cache
                
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
                
                video_name = os.path.splitext(video_id)[0]
                is_found = video_name in numpy_cache
                    
                i['status'] = 'Found' if is_found else 'Missing'
                
            gloss_index[gloss] = items

    matching_glosses = {}
    found_videos_list = []
    missing_videos_list = []
    source_stats = {}
    json_output_data = []

    unique_glosses_found_count = 0
    unique_glosses_missing_count = 0

    for primary, synonyms in normalized_pairs:
        gloss_upper = primary
        
        # Check if primary exists and has found videos_cut
        primary_has_found = False
        if gloss_upper in gloss_index:
            primary_has_found = any(i.get('status') == 'Found' for i in gloss_index[gloss_upper])
            
        # Determine which gloss/items to map to
        chosen_gloss = gloss_upper
        is_fallback_used = False
        items = None
        
        if primary_has_found:
            items = gloss_index[gloss_upper]
            chosen_gloss = gloss_upper
        else:
            # Check synonyms for a found video
            found_syn = False
            for syn in synonyms:
                if syn in gloss_index and any(i.get('status') == 'Found' for i in gloss_index[syn]):
                    items = gloss_index[syn]
                    chosen_gloss = syn
                    is_fallback_used = True
                    found_syn = True
                    break
            
            # If still not found, just get any items if available
            if not found_syn:
                if gloss_upper in gloss_index:
                    items = gloss_index[gloss_upper]
                    chosen_gloss = gloss_upper
                else:
                    for syn in synonyms:
                        if syn in gloss_index:
                            items = gloss_index[syn]
                            chosen_gloss = syn
                            is_fallback_used = True
                            break
            
        gloss_found_any_video = False
        items_list = []
        display_gloss = primary if chosen_gloss == primary else f"{primary} ({chosen_gloss})"
        
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
                    
                    actual_filename = video_id
                    if not actual_filename.lower().endswith(".npy"):
                        actual_filename = os.path.splitext(actual_filename)[0] + ".npy"
                    
                    dir_path = f"./{os.path.basename(NUMPY_DIR)}/"
                    full_path = os.path.abspath(os.path.join(NUMPY_DIR, actual_filename))
                        
                    found_videos_list.append({
                        "Gloss": display_gloss,
                        "Video ID": video_id,
                        "Source": source,
                        "Directory": dir_path,
                        "Full Path": full_path
                    })
                else:
                    source_stats[source]["missing"] += 1
                    exp_dir = f"./{os.path.basename(NUMPY_DIR)}/"
                        
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

                video_path = f"./{os.path.basename(NUMPY_DIR)}/"

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
            "synonyms": synonyms,
            "chosen_gloss": chosen_gloss,
            "fallback_used": is_fallback_used,
            "status": "found" if gloss_found_any_video else "missing",
            "item": items_list
        })

    # Excel and JSON report saving logic has been removed.

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

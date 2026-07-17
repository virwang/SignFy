import os
import sys
import numpy as np
from typing import Union, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NUMPY_DIR_NAME = os.getenv("ASL_NUMPY_DIR", "videos_numpy")
DEFAULT_NUMPY_DIR = os.path.join(SCRIPT_DIR, DEFAULT_NUMPY_DIR_NAME)
BONE_SIGN_OUT_DIR = os.path.join(SCRIPT_DIR, "bone_sign_out")

def find_npy_file(video_id, search_dir):
    """
    Finds the corresponding .npy file for a given video_id in the search directory.
    """
    if not video_id:
        return None
        
    vid_no_ext = os.path.splitext(video_id)[0]
    
    npy_path = os.path.join(search_dir, f"{vid_no_ext}.npy")
    if os.path.exists(npy_path):
        return npy_path
        
    if os.path.exists(search_dir):
        for f in os.listdir(search_dir):
            if f.startswith(vid_no_ext) and f.endswith(".npy"):
                return os.path.join(search_dir, f)
            
    return None

def merge_numpy_arrays(video_records, numpy_dir=DEFAULT_NUMPY_DIR) -> Optional[str]:
    json_data = None
    if isinstance(video_records, dict):
        json_data = video_records.get("json_output")
    elif isinstance(video_records, list):
        json_data = video_records
        
    if not json_data:
        print("[Error] No video records data found for numpy merging.", file=sys.stderr)
        return None
        
    arrays_to_merge = []
    glosses_used = []
    
    for entry in json_data:
        gloss = entry.get("gloss")
        status = entry.get("status")
        items = entry.get("item", [])
        
        if status == "missing" or not items:
            print(f"[Numpy Merger] Gloss '{gloss}' is missing in videos. Skipping in numpy merge.")
            continue
            
        video_id = items[0].get("video_id")
        
        npy_path = find_npy_file(video_id, numpy_dir)
        if npy_path and os.path.exists(npy_path):
            try:
                arr = np.load(npy_path)
                arrays_to_merge.append(arr)
                glosses_used.append(gloss)
                print(f"[Numpy Merger] Loaded {npy_path} for gloss '{gloss}' (shape: {arr.shape})")
            except Exception as e:
                print(f"[Error] Failed to load {npy_path}: {e}", file=sys.stderr)
        else:
            print(f"[Numpy Merger] Could not find .npy file for video_id: {video_id} in {numpy_dir}")
            
    if not arrays_to_merge:
        print("[Error] No numpy arrays found to merge.", file=sys.stderr)
        return None
        
    # Concatenate along the frames axis (axis=0)
    merged_array = np.concatenate(arrays_to_merge, axis=0)
    
    os.makedirs(BONE_SIGN_OUT_DIR, exist_ok=True)
    
    # "名稱與gloss 順序相同，不允許重複檔名"
    # Join glosses to form the base filename
    base_name = "_".join(glosses_used)
    # Ensure valid filename characters
    base_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in base_name)
    if not base_name:
        base_name = "merged_output"
        
    out_file = os.path.join(BONE_SIGN_OUT_DIR, f"{base_name}.npy")
    
    # Handle duplicate filename
    counter = 1
    while os.path.exists(out_file):
        out_file = os.path.join(BONE_SIGN_OUT_DIR, f"{base_name}_{counter}.npy")
        counter += 1
        
    np.save(out_file, merged_array)
    print(f"[Success] Numpy merging complete! Output file: {out_file} with shape {merged_array.shape}")
    return out_file

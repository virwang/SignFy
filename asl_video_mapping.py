"""_summary_
        This script performs the following tasks:
        1. Receive the translated english text input and the corresponding ASL gloss output from the Llama model.
        2. Load the WLASL dataset and check if the corresponding videos exist in the Microsoft and WLASL video directories.
        3. Generate a comprehensive report in Excel format, categorizing the results into "Found" and "Missing" videos, and providing statistics on the distribution of glosses across sources.
        4. The script is optimized for performance by caching video filenames in memory to avoid repeated disk access, and it handles edge cases such as missing fields in the input JSON gracefully.   
"""
import argparse
import json 
import os
import re
import sys

import urllib.request
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from typing import List, Dict, Tuple

MAPPLING_PATH = "./data_preprocessing/asl_mis_wlasl.json"
    
def find_video_records(gloss_array):
    print(f"Finding video records for ASL glosses: {gloss_array}")
    
    if not os.path.isfile(MAPPLING_PATH):
        raise FileNotFoundError(f"Mapping file {MAPPLING_PATH} not found.")
    if not os.path.isdir("./Microsoft_Videos"):
        raise RuntimeError("Microsoft_Videos directory not found.")
    if not os.path.isdir("./videos"):     
        raise RuntimeError("videos directory not found.")

    
    print(f"Loading mapping data from {MAPPLING_PATH}...")
    
    with open(MAPPLING_PATH, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    results = {}
    
    for entry_data in mapping_data.values():
        source_name = entry_data.get("source")
        video_id = entry_data.get("video_id")
        gloss = entry_data.get("gloss", "").upper()  # Normalize gloss for matching
        
        if source_name == "microsoft":      
            video_dir = "Microsoft_Videos"
        else:
            video_dir = "videos"
        
        if gloss in results:
            results[gloss]["video_id"].append(video_id)
            results[gloss]["source"].append(video_dir)
        else:
            results[gloss] = {
                "video_id": [video_id],
                "source": [video_dir]
            }
    
#     for gloss in gloss_array:
#         # Retrieve metadata if gloss exists in mapping
#         gloss = gloss.upper()
#         print(f"gloss: {gloss}")# Normalize input gloss for matching
        
        
#         if gloss in mapping_data:
#             source_name = mapping_data[gloss].get("source")
#             if source_name == "microsoft":      
#                 video_dir = "Microsoft_Videos"
#             else:
#                 video_dir = "videos"
                            
#             results[gloss] = {
#                 "video_id": mapping_data[gloss].get("video_id"),
#                 "source": video_dir
#             }
#         else:
#             # Handle missing gloss case
#             results[gloss] = {"video_id": None, "source": None}


    
    print(f"Video mapping results: {json.dumps(results)}")        
    return results
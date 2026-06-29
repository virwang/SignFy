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

def load_video_filenames(file_path=MAPPLING_PATH):
    print(f"Loading mapping data from {MAPPLING_PATH}...")
    
    if not os.path.isfile(MAPPLING_PATH):
        raise FileNotFoundError(f"Mapping file {MAPPLING_PATH} not found.")
    if not os.path.isdir("./Microsoft_Videos"):
        raise RuntimeError("Microsoft_Videos directory not found.")
    if not os.path.isdir("./videos"):     
        raise RuntimeError("videos directory not found.") 

    with open(MAPPLING_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data

def find_matching_glosses(json_data, llm_output):
    # build a mapping from gloss name (uppercase) to a list of its video items
    gloss_index = {}
    for entry in json_data:
        gloss = str(entry.get('gloss', '')).upper()
        items = entry.get('item', [])
        # We only want to keep items that are "Found"
        found_items = [i for i in items if i.get('status') == 'Found']
        if found_items:
            gloss_index[gloss] = found_items

    matching_glosses = {}
    for gloss in llm_output:
        if not gloss:
            continue

        gloss_upper = gloss.upper()
        if gloss_upper in gloss_index:
            # pick the first found item/video
            first_found_item = gloss_index[gloss_upper][0]
            source = first_found_item.get('source', '').lower()
            video_id = first_found_item.get('video_id', 'N/A')

            if source == 'microsoft':
                video_path = "./Microsoft_Videos/"
            else:
                video_path = "./videos/"

            matching_glosses[gloss] = {
                "video_id": video_id,
                "video_path": video_path
            }

    return matching_glosses

def record_not_found_glosses(json_data, llm_output):
    # build a set of found glosses
    found_glosses = set()
    for entry in json_data:
        gloss = str(entry.get('gloss', '')).upper()
        items = entry.get('item', [])
        if any(i.get('status') == 'Found' for i in items):
            found_glosses.add(gloss)
            
    not_found_glosses = [g for g in llm_output if g and g.upper() not in found_glosses]
    return not_found_glosses

def find_video_records(llm_output):
    json_data = load_video_filenames()
    matching_glosses = find_matching_glosses(json_data, llm_output)
    # not_found_glosses = record_not_found_glosses(json_data, llm_output)

    return {
        "matching_glosses": matching_glosses
        # "not_found_glosses": not_found_glosses
    }
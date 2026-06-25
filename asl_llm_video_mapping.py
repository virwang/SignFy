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
    matching_glosses = {}
    for gloss in llm_output:
        print(f"Checking gloss: {gloss}")
        
        for data in json_data:  
            data_gloss = data.get('gloss', '').upper()  # Ensure gloss is uppercase for comparison  
            if data_gloss == gloss.upper() and data.get('status') == 'Found':                           
                video_path = data.get('source', 'N/A')                
                if video_path == 'microsoft':
                    video_path = "./Microsoft_Videos/"
                else:
                    video_path = "./videos/"
                print(f"Match found for gloss '{gloss}': video_id={data.get('video_id', 'N/A')}, video_path={video_path}")        
                matching_glosses[gloss] = {
                    "video_id": data.get('video_id', 'N/A'),
                    "video_path": video_path
                }
                
                print(f"Found match for gloss '{gloss}': {matching_glosses[gloss]}")
                break
    
    return matching_glosses

def find_matching_glosses(json_data, llm_output):

    gloss_index = {
        str(item.get('gloss', '')).upper(): item
        for item in json_data
        if item.get('status') == 'Found'
    }

    matching_glosses = {}

    for gloss in llm_output:

        if not gloss:
            continue

        item = gloss_index.get(gloss.upper())

        if item:

            source = item.get('source', '').lower()

            if source == 'Microsoft_Videos':
                video_path = "./Microsoft_Videos/"
            else:
                video_path = "./videos/"

            matching_glosses[gloss] = {
                "video_id": item.get('video_id', 'N/A'),
                "video_path": video_path
            }

    return matching_glosses

def record_not_found_glosses(json_data, llm_output):
    not_found_glosses = [g for g in llm_output if not any(gloss['gloss'] == g.upper() for gloss in json_data) or (gloss := next((g for g in json_data if g['gloss'].upper() == g.upper()), None)) and gloss['status'] != 'Found']
    return not_found_glosses

def find_video_records(llm_output):
    json_data = load_video_filenames()
    matching_glosses = find_matching_glosses(json_data, llm_output)
    # not_found_glosses = record_not_found_glosses(json_data, llm_output)

    return {
        "matching_glosses": matching_glosses
        # "not_found_glosses": not_found_glosses
    }
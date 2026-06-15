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





"""
Script: generate_exists_v2.py
Reads exists.json and writes exists_v2.json containing the same top-level structure
but each instance only keeps the fields: source and video_id.

Usage: python generate_exists_v2.py

Requires: exists.json present in the working directory.
"""

import json
from pathlib import Path

INPUT = Path('exists.json')
OUTPUT = Path('exists_v2.json')

if not INPUT.exists():
    print(f"Error: {INPUT} not found. Run the script that generates exists.json first.")
    raise SystemExit(1)

with INPUT.open('r', encoding='utf-8') as f:
    data = json.load(f)

new_content = []
kept_instances = 0
removed_instances = 0

for entry in data:
    gloss = entry.get('gloss')
    instances = entry.get('instances', [])

    filtered = []
    for inst in instances:
        # Only keep source and video_id (if video_id missing, skip)
        if 'video_id' not in inst:
            removed_instances += 1
            continue
        filtered.append({
            'source': "WLASL_"+inst.get('source'),
            'video_id': inst.get('video_id')
        })
        kept_instances += 1

    if filtered:
        new_content.append({
            'gloss': gloss,
            'instances': filtered
        })

with OUTPUT.open('w', encoding='utf-8') as f:
    json.dump(new_content, f, indent=4, ensure_ascii=False)

print(f"Generated {OUTPUT} with {len(new_content)} gloss entries and {kept_instances} instances kept (skipped {removed_instances}).")

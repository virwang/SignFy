# find out the video files that exist in the raw_videos directory and keep only those instances in the JSON file, then save the result to exists.json. This is to ensure that we only keep the data for which we have valid video files, which is crucial for training and evaluation purposes.

import os
import json

# set up paths and filenames
video_dir = 'videos'
input_json = 'WLASL_v0.3.json'
output_json = 'exists.json'

# generate a set of valid video IDs based on existing files in the video directory (length > 0)
def get_valid_files(directory):
    valid_ids = set()
    if not os.path.exists(directory):
        print(f"Error：Path not found in {directory}")
        return valid_ids
        
    for f in os.listdir(directory):
        file_path = os.path.join(directory, f)
        
        # file exits and is not empty
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
    
            # get the filename without extension (e.g., '65225.mp4' -> '65225')
            valid_ids.add(os.path.splitext(f)[0])
    return valid_ids

valid_ids = get_valid_files(video_dir)
print(f"Scan complete, found {len(valid_ids)} valid video files.")

# read the original JSON file
with open(input_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

new_content = []

for entry in data:
    # filter instances, keep only those whose video_id is in valid_ids
    valid_instances = [
        inst for inst in entry.get('instances', [])
        if str(inst.get('video_id')) in valid_ids
    ]
    
    # keep the original entry structure, but only with valid instances
    if valid_instances:

        # create a new entry with the same structure as the original, but only with valid instances
        new_entry = entry.copy()
        new_entry['instances'] = valid_instances
        new_content.append(new_entry)

# 3. 寫入 exists.json
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(new_content, f, indent=4, ensure_ascii=False)

print(f"Processing complete! {output_json} generated with {len(new_content)} glosses.")
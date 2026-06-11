import json

# Load both files
with open('exists_v2.json', 'r', encoding='utf-8') as f:
    exists_v2 = json.load(f)

with open('microsoft.json', 'r', encoding='utf-8') as f:
    microsoft = json.load(f)

# Dictionary to combine entries by lowercase gloss
# Using structure: gloss -> list of video_ids with their sources
combined = {}

# First add all entries from exists_v2 (flattening the instances)
for entry in exists_v2:
    gloss_lower = entry['gloss'].lower()
    if gloss_lower not in combined:
        combined[gloss_lower] = {
            'gloss': entry['gloss'],
            'entries': []  # Temporary storage for video_ids
        }
    # Flatten instances from exists_v2
    if 'instances' in entry:
        for instance in entry['instances']:
            combined[gloss_lower]['entries'].append({
                'video_id': instance.get('video_id'),
                'source': instance.get('source', 'unknown')
            })

# Then add entries from microsoft.json
for entry in microsoft:
    gloss_lower = entry['gloss'].lower()
    if gloss_lower not in combined:
        combined[gloss_lower] = {
            'gloss': entry['gloss'],
            'entries': []
        }
    # Add microsoft entry with source as 'microsoft'
    combined[gloss_lower]['entries'].append({
        'video_id': entry.get('video_id'),
        'source': 'microsoft'
    })

# Convert to final format following microsoft.json structure
# Final structure will have multiple entries for same gloss with different video_ids
result = []
for gloss_lower, data in combined.items():
    for entry_data in data['entries']:
        result.append({
            'gloss': data['gloss'],
            'video_id': entry_data['video_id'],
            'source': entry_data['source']
        })

# Sort by gloss
result = sorted(result, key=lambda x: x['gloss'].lower())

# Save output
with open('asl_words.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f'Combined entries: {len(result)} total video_id entries')
print(f'Unique glosses: {len(combined)}')
print(f'\nSample entries from result:')
for i, entry in enumerate(result[:5]):
    print(f"  {i+1}. gloss: {entry['gloss']}, video_id: {entry['video_id']}, source: {entry['source']}")


import json

# Check output for microsoft entries
with open('asl_words.json', 'r', encoding='utf-8') as f:
    output = json.load(f)

# Look for evaluate entries with microsoft source
ms_entries = [e for e in output if e['gloss'].lower() == 'evaluate' and e['source'] == 'microsoft']
print(f"Microsoft entries for 'evaluate': {len(ms_entries)}")
for e in ms_entries:
    print(f"  video_id: {e['video_id']}, source: {e['source']}")

# Check if video_id 9972993522717659 is there
entry_found = [e for e in output if e['video_id'] == '9972993522717659']
print(f"\nEntry with video_id 9972993522717659: {entry_found}")

# Count microsoft sources
ms_count = sum(1 for e in output if e['source'] == 'microsoft')
print(f"\nTotal entries with source 'microsoft': {ms_count}")

# Show a few microsoft entries
print(f"\nFirst 5 entries with microsoft source:")
for i, e in enumerate([e for e in output if e['source'] == 'microsoft'][:5]):
    print(f"  {i+1}. gloss: {e['gloss']}, video_id: {e['video_id']}")

import json

# Load original files
with open('exists_v2.json', 'r', encoding='utf-8') as f:
    exists_v2 = json.load(f)

with open('microsoft.json', 'r', encoding='utf-8') as f:
    microsoft = json.load(f)

# Load output
with open('asl_words.json', 'r', encoding='utf-8') as f:
    output = json.load(f)

print("=" * 60)
print("ASL Words Combination Summary")
print("=" * 60)

print(f"\nInput files:")
print(f"  exists_v2.json: {len(exists_v2)} entries")
total_ev_instances = sum(len(e.get('instances', [])) for e in exists_v2)
print(f"    - Total video_ids: {total_ev_instances}")

print(f"  microsoft.json: {len(microsoft)} entries")

print(f"\nOutput file:")
print(f"  asl_words.json: {len(output)} total entries")

# Count by source
sources_count = {}
for e in output:
    src = e['source']
    sources_count[src] = sources_count.get(src, 0) + 1

print(f"\nEntries by source:")
for src, count in sorted(sources_count.items()):
    pct = count / len(output) * 100
    print(f"  - {src}: {count} ({pct:.1f}%)")

# Count unique glosses
unique_glosses = len(set(e['gloss'].lower() for e in output))
print(f"\nUnique glosses (case-insensitive): {unique_glosses}")

# Sample output
print(f"\nSample entries:")
samples = [output[0], output[len(output)//2], output[-1]]
for s in samples:
    print(f"  - gloss: '{s['gloss']}', video_id: {s['video_id']}, source: {s['source']}")

print("\n" + "=" * 60)
print("✓ Combination complete!")
print("=" * 60)

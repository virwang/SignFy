import json

# Load both files
with open('exists_v2.json', 'r', encoding='utf-8') as f:
    exists_v2 = json.load(f)

with open('microsoft.json', 'r', encoding='utf-8') as f:
    microsoft = json.load(f)

# Get glosses from both
ev_glosses = {e['gloss'].lower() for e in exists_v2}
ms_glosses = {e['gloss'].lower() for e in microsoft}

# Find common glosses
common = ev_glosses.intersection(ms_glosses)

# Find an example from each
if common:
    example_gloss = next(iter(common)).upper()
    print(f"Example common gloss: {example_gloss}")
    
    # Check exists_v2
    print(f"\nIn exists_v2.json:")
    for e in exists_v2:
        if e['gloss'].lower() == example_gloss.lower():
            print(f"  gloss: {e['gloss']}")
            if 'instances' in e:
                print(f"  instances: {len(e['instances'])} total")
                for inst in e['instances'][:2]:
                    print(f"    - {inst}")
            break
    
    # Check microsoft
    print(f"\nIn microsoft.json:")
    for e in microsoft:
        if e['gloss'].lower() == example_gloss.lower():
            print(f"  gloss: {e['gloss']}")
            print(f"  video_id: {e['video_id']}")
            break
    
    # Check output
    print(f"\nIn asl_words.json:")
    with open('asl_words.json', 'r', encoding='utf-8') as f:
        output = json.load(f)
    
    count = 0
    for e in output:
        if e['gloss'].lower() == example_gloss.lower():
            print(f"  gloss: {e['gloss']}, video_id: {e['video_id']}, source: {e['source']}")
            count += 1
            if count >= 3:
                print(f"  ... and {sum(1 for x in output if x['gloss'].lower() == example_gloss.lower()) - count} more")
                break

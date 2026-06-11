import json

# Check microsoft.json structure
with open('microsoft.json', 'r', encoding='utf-8') as f:
    ms = json.load(f)

# Count entries with instances
has_instances = sum(1 for e in ms if 'instances' in e and len(e.get('instances', [])) > 0)
no_instances = sum(1 for e in ms if 'instances' not in e or len(e.get('instances', [])) == 0)

print(f'Microsoft.json: {len(ms)} total entries')
print(f'  - With instances: {has_instances}')
print(f'  - Without instances: {no_instances}')

# Check exists_v2.json structure
with open('exists_v2.json', 'r', encoding='utf-8') as f:
    ev = json.load(f)

has_instances = sum(1 for e in ev if 'instances' in e and len(e.get('instances', [])) > 0)
no_instances = sum(1 for e in ev if 'instances' not in e or len(e.get('instances', [])) == 0)

print(f'\nExists_v2.json: {len(ev)} total entries')
print(f'  - With instances: {has_instances}')
print(f'  - Without instances: {no_instances}')

# Check output asl_words.json
with open('asl_words.json', 'r', encoding='utf-8') as f:
    output = json.load(f)

has_instances = sum(1 for e in output if len(e.get('instances', [])) > 0)
no_instances = sum(1 for e in output if len(e.get('instances', [])) == 0)

print(f'\nASL_words.json (output): {len(output)} total entries')
print(f'  - With instances: {has_instances}')
print(f'  - Without instances: {no_instances}')

# Check for matches
ms_glosses_lower = {e['gloss'].lower() for e in ms}
ev_glosses_lower = {e['gloss'].lower() for e in ev}

matches = ms_glosses_lower.intersection(ev_glosses_lower)
only_ms = ms_glosses_lower - ev_glosses_lower
only_ev = ev_glosses_lower - ms_glosses_lower

print(f'\nGloss matching analysis:')
print(f'  - Common glosses (case-insensitive): {len(matches)}')
print(f'  - Only in microsoft.json: {len(only_ms)}')
print(f'  - Only in exists_v2.json: {len(only_ev)}')

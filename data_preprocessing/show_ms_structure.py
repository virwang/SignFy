import json

with open('microsoft.json', 'r', encoding='utf-8') as f:
    ms = json.load(f)

# Get first few entries with their structure
for i, e in enumerate(ms[:10]):
    print(f"Entry {i}: {json.dumps(e, indent=2)}")
    print()

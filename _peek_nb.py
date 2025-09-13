import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'attack/multi_attack.ipynb'
nb = json.load(open(path, 'r', encoding='utf-8'))
codes = [(i, c) for i, c in enumerate(nb.get('cells', [])) if c.get('cell_type') == 'code']
print('Code cells:', len(codes))
for i, c in codes[-5:]:
    print('\n--- idx', i, '---')
    print(''.join(c.get('source', [])))


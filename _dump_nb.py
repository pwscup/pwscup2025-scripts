import json
import sys
p = r'attack/multi_attack.ipynb'
with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if src.strip():
            print('--- CELL ---')
            print(src)

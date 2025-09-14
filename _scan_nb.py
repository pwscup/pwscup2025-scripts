import json, sys
p = r'attack/multi_attack.ipynb'
with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)
count = 0
for idx, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        count += 1
        src = ''.join(cell.get('source', []))
        first = src.splitlines()[0] if src.splitlines() else ''
        print(f'{idx}\t{first[:120]}')
print('TOTAL_CODE', count)

import json
p = r'attack/multi_attack.ipynb'
nb = json.load(open(p, 'r', encoding='utf-8'))
for idx in (12,13):
    cell = nb['cells'][idx]
    print('--- CELL', idx, '---')
    print(''.join(cell.get('source', [])))

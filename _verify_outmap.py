import json
p = r'attack/multi_attack.ipynb'
nb = json.load(open(p, 'r', encoding='utf-8'))
for idx, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type')=='code' and 'attack/new_attackDi_Ci_greedy.py' in ''.join(cell.get('source', [])):
        print('CELL', idx)
        print(''.join(cell['source']))

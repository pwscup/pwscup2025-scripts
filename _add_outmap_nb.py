import json
p = r"attack/multi_attack.ipynb"
with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'attack/new_attackDi_Ci_greedy.py' in src and '--out-map' not in src:
        src_lines = src.splitlines()
        # Insert --out-map before closing bracket of list
        for i, line in enumerate(src_lines):
            if line.strip().startswith(']'):
                insert_at = i
                break
        else:
            insert_at = len(src_lines)
        new_opts = '    "--out-map", "out/C{id:02d}_matchmap_greedy.csv",'
        # Put it just before the closing bracket line
        src_lines.insert(insert_at, new_opts)
        cell['source'] = [l + '\n' for l in src_lines]

with open(p, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('done')

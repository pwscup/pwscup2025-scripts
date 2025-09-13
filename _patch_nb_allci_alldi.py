import json

p = r"attack/multi_attack.ipynb"
nb = json.load(open(p, 'r', encoding='utf-8'))

target_snippet = 'attack/attack_allCi_allDi_hungarian.py'
new_src = '''# AllCi + AllDi (Hungarian): rank by [Hungarian Ci distance + Di |pred - y|], pick top 10,000
AllCi_AllDi_Hungarian = [
    "python", "attack/attack_allCi_allDi_hungarian.py",
    "out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv",
    "out/PWSCUP2025_Pre_Data_for_Attack/C{id:02d}_fix.csv",
    "out/PWSCUP2025_Pre_Data_for_Attack/D{id:02d}.json",
    "--hung-mode", "knn",
    "-k", "300",
    "--w-dist", "1.0",
    "--w-conf", "1.0",
    "--auto-wdist",
    "--topn", "10000",
    "-o", "out/Fij_all_hungarian_{id:02d}.csv",
    "--out-rank", "out/Fij_all_hungarian_{id:02d}_rank.csv",
    "--out-map", "out/C{id:02d}_matchmap_all_hungarian.csv",
]
loop_for_all_teams(AllCi_AllDi_Hungarian)
print(f"AllCi+AllDi (Hungarian) scoring completed")'''

def set_source(cell, text):
    cell['source'] = [line + '\n' for line in text.splitlines()]

found = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if target_snippet in src:
        set_source(cell, new_src)
        found = True
        break

if not found:
    # Append as a new cell if not found
    nb.setdefault('cells', []).append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in new_src.splitlines()],
    })

json.dump(nb, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('patched AllCi+AllDi (Hungarian) cell')


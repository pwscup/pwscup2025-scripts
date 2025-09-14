import json
p = r"attack/multi_attack.ipynb"
with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# helper to update a cell's source string list from a multi-line string
def set_source(cell, text):
    cell['source'] = [line + '\n' for line in text.splitlines()]

# find and update cells
for idx, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'attack/new_attackDi_Ci.py' in src:
        new_src = '''# New Di->Ci scoring attack (union of Di selections, rank by Ci distance and |pred - y|)
# Example knobs: threshold-based (pred=0.5, conf=0.3), k=5, select topn=10000
New_DiCi_scoring = [
    "python", "attack/new_attackDi_Ci.py",
    "out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv",
    "out/PWSCUP2025_Pre_Data_for_Attack/C{id:02d}_fix.csv",
    "out/PWSCUP2025_Pre_Data_for_Attack/D{id:02d}.json",
    "--pred-threshold", "0.5",
    "--conf-threshold", "0.3",
    "--mode", "union",
    "-k", "5",
    "--w-conf", "1.0",
    "--auto-wdist",
    "--topn", "10000",
    "-o", "out/Fij_new_{id:02d}.csv",
    "--out-rank", "out/Fij_new_{id:02d}_rank.csv",
]
loop_for_all_teams(New_DiCi_scoring)
print(f"new Di->Ci scoring attack completed")'''
        set_source(cell, new_src)
    if 'attack/new_attackDi_Ci_greedy.py' in src:
        new_src = '''# New Di->Ci scoring attack (greedy Ci matching; k ignored, ranks expand automatically)
New_DiCi_scoring_greedy = [
    "python", "attack/new_attackDi_Ci_greedy.py",
    "out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv",
    "out/PWSCUP2025_Pre_Data_for_Attack/C{id:02d}_fix.csv",
    "out/PWSCUP2025_Pre_Data_for_Attack/D{id:02d}.json",
    "--pred-threshold", "0.5",
    "--conf-threshold", "0.3",
    "--mode", "union",
    "--w-conf", "1.0",
    "--auto-wdist",
    "--topn", "10000",
    "-o", "out/Fij_new_greedy_{id:02d}.csv",
    "--out-rank", "out/Fij_new_greedy_{id:02d}_rank.csv",
]
loop_for_all_teams(New_DiCi_scoring_greedy)
print(f"new Di->Ci scoring (greedy) completed")'''
        set_source(cell, new_src)

with open(p, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('done')

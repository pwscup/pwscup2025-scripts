import json

p = r"attack/multi_attack.ipynb"

with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_src = '''# AllCi + AllDi (Hungarian): rank by [Hungarian Ci distance + Di |pred - y|], pick top 10,000
AllCi_AllDi_Hungarian = [
    "python", "attack/attack_allCi_allDi_hungarian.py",
    "out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv",
    "out/PWSCUP2025_Pre_Data_for_Attack/C{id:02d}_fix.csv",
    "out/PWSCUP2025_Pre_Data_for_Attack/D{id:02d}.json",
    "--hung-mode", "knn",
    "-k", "300",
    "--topn", "10000",
    "-o", "out/Fij_all_hungarian_{id:02d}.csv",
    "--out-rank", "out/Fij_all_hungarian_{id:02d}_rank.csv",
    "--out-map", "out/C{id:02d}_matchmap_all_hungarian.csv",
]
loop_for_all_teams(AllCi_AllDi_Hungarian)
print(f"AllCi+AllDi (Hungarian) scoring completed")'''

def make_code_cell(src: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.splitlines()]
    }

# Append at the end
nb.setdefault('cells', []).append(make_code_cell(new_src))

with open(p, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('appended new AllCi+AllDi (Hungarian) cell')


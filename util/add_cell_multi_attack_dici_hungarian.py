#!/usr/bin/env python3
import json
from pathlib import Path


NB_PATH = Path("attack/multi_attack.ipynb")


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    # Prepare the new code cell (mirrors style of existing batch blocks)
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "new-dici-hungarian",
        "metadata": {},
        "outputs": [],
        "source": [
            "# New Di->Ci scoring attack (Hungarian Ci matching; rank Di candidates by matched distance and |pred - y|)\n",
            "# Example knobs: threshold-based selection, Hungarian knn mode k=300, select top 10,000\n",
            "New_DiCi_hungarian = [\n",
            "    \"python\", \"attack/attackDi_Ci_hungarian.py\",\n",
            "    \"out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv\",\n",
            "    \"out/PWSCUP2025_Pre_Data_for_Attack/C{id:02d}_fix.csv\",\n",
            "    \"out/PWSCUP2025_Pre_Data_for_Attack/D{id:02d}.json\",\n",
            "    \"--pred-threshold\", \"0.5\",\n",
            "    \"--conf-threshold\", \"0.5\",\n",
            "    \"--mode\", \"union\",\n",
            "    \"--hung-mode\", \"knn\",\n",
            "    \"-k\", \"300\",\n",
            "    \"--w-conf\", \"1.0\",\n",
            "    \"--auto-wdist\",\n",
            "    \"--topn\", \"10000\",\n",
            "    \"-o\", \"out/Fij_new_hung_{id:02d}.csv\",\n",
            "    \"--out-rank\", \"out/Fij_new_hung_{id:02d}_rank.csv\",\n",
            "    \"--out-map\", \"out/C{id:02d}_matchmap_hungarian_used.csv\",\n",
            "]\n",
            "loop_for_all_teams(New_DiCi_hungarian)\n",
            "print(f\"new Di->Ci Hungarian scoring attack completed\")\n",
        ],
    }

    cells.append(cell)
    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[ok] Appended Hungarian Di->Ci batch cell to attack/multi_attack.ipynb")


if __name__ == "__main__":
    main()


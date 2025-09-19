"""
attack_Di.pyのTopConfAttackを使って全チームに攻撃し、
提出可能なzipファイルを作るスクリプト例
"""
import argparse
import os
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np

from attack_Di import TopConfAttack

# チーム数と行数を定義
NUM_TEAMS = 24
NUM_RAWS = int(1e5)

# コマンドライン引数の読み込み
argparser = argparse.ArgumentParser(description="")
argparser.add_argument("i", type=str, help="your team ID")
argparser.add_argument("indir", type=str, help="target directory")
argparser.add_argument("outdir", type=str, help="output directory")
argparser.add_argument("--overwrite", "-o", action='store_true', help='overwrite the existing output directory')
args = argparser.parse_args()

# indirが存在しない場合は終了
if not os.path.isdir(args.indir):
    raise FileNotFoundError(f"{args.indir}は存在しません")

# outdirが存在し、-oがついていない場合は意図しない上書きを避けるために終了
if os.path.exists(args.outdir) and not args.overwrite:
    raise FileExistsError(f"{args.outdir}はすでに存在します。上書きする場合は-oをつけてください。")

# 各チームjに攻撃
Fi = pd.DataFrame()
for j in range(1, NUM_TEAMS+1):
    print(j)
    if str(j) == args.i:
        print(f"The {j}th column was filled with empty because it is your team")
        Fi[f"{str(j).zfill(2)}"] = pd.DataFrame(np.nan, index=range(NUM_RAWS), columns=["inferred"])
        continue

    # path_to_Ci = os.path.join(args.indir, f"D{str(j).zfill(2)}.csv")
    # Ciも使う攻撃の場合は↑のコメントアウトを解除
    path_to_Di = os.path.join(args.indir, f"D{str(j).zfill(2)}.json")
    path_to_Ai = os.path.join(args.indir, f"A{str(j).zfill(2)}.csv")

    try:
        # TopConfAttackで攻撃。ここを書き換えると別の攻撃にできる。
        attacker = TopConfAttack(path_to_Di)
        pred = attacker.infer(path_to_Ai)
    except Exception as e:
        # 攻撃中にエラーが出た場合はダミーデータで代替
        dummy = pd.DataFrame(0, index=range(NUM_RAWS), columns=["inferred"])
        dummy.loc[:1e4, "inferred"] = 1
        pred = dummy.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"The {j}th column was filled with dummy data becase of {e}")

    Fi[f"{str(j).zfill(2)}"] = pred
    # attacker.save_inferred("inferred_membership1.csv")

# outdirが存在しなければ作る
os.makedirs(args.outdir, exist_ok=True)

# 攻撃結果Fiをcsvに書き込み
Fi_csv_path = Path(os.path.join(args.outdir, f"F{args.i.zfill(2)}.csv"))
Fi.to_csv(Fi_csv_path, index=False, header=False)

# id.txtを作る
id_txt_path = Path(os.path.join(args.outdir, "id.txt"))
with open(id_txt_path, 'w') as file:
    file.write(args.i)

# zipファイルを作る
zip_path = os.path.join(args.outdir, f"F{args.i.zfill(2)}.zip")
with zipfile.ZipFile(zip_path, 'w') as zf:
    zf.write(Fi_csv_path, arcname=Fi_csv_path.name)
    zf.write(id_txt_path, arcname=id_txt_path.name)

print(f"submission zip file was successfully saved at {zip_path}")

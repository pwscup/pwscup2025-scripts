"""
attack_Di.pyのTopConfAttackを使って全チームに攻撃し、
提出可能なzipファイルを作るスクリプト例
"""
import argparse
import os, sys
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np

from attack_Di import TopConfAttack
# モジュールの相対参照制限を強制的に回避
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'util'))
from pws_data_format import FFjDataFrame # 提出用CSVのフォーマット確認用

# チーム数と行数を定義
NUM_TEAMS = 24
NUM_RAWS = int(1e5)

# 未提出チーム
UNSUBMITTED_TEAMS = FFjDataFrame.UNSUBMITTED_TEAMS

# コマンドライン引数の読み込み
argparser = argparse.ArgumentParser(description="")
argparser.add_argument("j", type=str, help="your team ID")
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

# 各チームiに攻撃
FFj = pd.DataFrame()
for i in range(1, NUM_TEAMS+1):
    """
    "01", ..., "24"という名前の列を作って、各列に対応するチームに対する攻撃結果を格納
    最終的なFFj.valuesは列のindexが0 ~ 23で一致していないので、混同に注意
    """
    print(i)
    if str(i) == args.j:
        print(f"The {i}th column was filled with empty because it is your team")
        FFj[f"{str(i).zfill(2)}"] = pd.DataFrame("", index=range(NUM_RAWS), columns=["inferred"])
        continue

    if i in UNSUBMITTED_TEAMS:
        print(f"The {i}th column was filled with empty because it is an unsubmitted team")
        FFj[f"{str(i).zfill(2)}"] = pd.DataFrame("", index=range(NUM_RAWS), columns=["inferred"])
        continue

    # path_to_Ci = os.path.join(args.indir, f"D{str(j).zfill(2)}.csv")
    # Ciも使う攻撃の場合は↑のコメントアウトを解除
    path_to_Di = os.path.join(args.indir, f"D{str(i).zfill(2)}.json")
    path_to_Ai = os.path.join(args.indir, f"A{str(i).zfill(2)}.csv")

    try:
        # TopConfAttackで攻撃。ここを書き換えると別の攻撃にできる。
        attacker = TopConfAttack(path_to_Di)
        pred = attacker.infer(path_to_Ai)
    except Exception as e:
        # 攻撃中にエラーが出た場合はダミーデータで代替
        dummy = pd.DataFrame("0", index=range(NUM_RAWS), columns=["inferred"])
        dummy.loc[:1e4-1, "inferred"] = "1"
        pred = dummy.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"The {i}th column was filled with dummy data becase of {e}")

    FFj[f"{str(i).zfill(2)}"] = pred.astype(str)

# 提出用csvのフォーマットを満たしているか確認
FFj = FFjDataFrame(FFj, j=args.j)

# outdirが存在しなければ作る
os.makedirs(args.outdir, exist_ok=True)

# 攻撃結果Fiをcsvに書き込み
FFj_csv_path = Path(os.path.join(args.outdir, f"F{args.j.zfill(2)}.csv"))
FFj.to_csv(FFj_csv_path)

# id.txtを作る
id_txt_path = Path(os.path.join(args.outdir, "id.txt"))
with open(id_txt_path, 'w') as file:
    file.write(args.j)

# zipファイルを作る
zip_path = os.path.join(args.outdir, f"F{args.j.zfill(2)}.zip")
with zipfile.ZipFile(zip_path, 'w') as zf:
    zf.write(FFj_csv_path, arcname=FFj_csv_path.name)
    zf.write(id_txt_path, arcname=id_txt_path.name)

print(f"submission zip file was successfully saved at {zip_path}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import pandas as pd

# モジュールの相対参照制限を強制的に回避
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'util'))
from pws_data_format import BiDataFrame, CiDataFrame

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 randomshuffle_rows.py <input_filename (csv)> <output_filename (csv)>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    try:
        # 文字列として読み込む：空欄も空文字のまま保持 → 数値表記が変わらない
        df = BiDataFrame.read_csv(input_csv)
    except Exception as e:
        print(f"CSV読み込みエラー: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("入力CSVが空です。処理を終了します。")
        sys.exit(0)

    # 行シャッフル（ヘッダーはそのまま）
    df_shuffled = df.sample(frac=1, random_state=None).reset_index(drop=True)

    try:
        # 文字列のまま書き出し → 123 が 123.0 になる問題を防止
        Ci_df = CiDataFrame(df_shuffled)
        Ci_df.to_csv(output_csv)
    except Exception as e:
        print(f"CSV書き込みエラー: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"シャッフル済みCSVを出力しました: {output_csv}")

if __name__ == "__main__":
    main()

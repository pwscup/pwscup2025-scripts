#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
  python3 gen_ans.py Ai.csv Bi.csv -o Zi.csv

- Ai.csv, Bi.csv はどちらもヘッダー付き
- Ai.csv の各データ行が Bi.csv に「同一行（同じ列名順の全値が一致）」として存在すれば 1、なければ 0
- 出力 Zi.csv はヘッダー無しの 1列。行数は Ai.csv のデータ行数（= Ai.csv の総行数 からヘッダー行数（=1）を引いた値）
"""

import argparse
import os
import sys
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Check if each row in Ai.csv exists in Bi.csv (exact match).")
    ap.add_argument("csv1", metavar="Ai.csv", help="ヘッダー付き Ai.csv")
    ap.add_argument("csv2", metavar="Bi.csv", help="ヘッダー付き Bi.csv")
    ap.add_argument("-o", "--output", required=True, help="出力CSV Zi（ヘッダー無しの1列）")
    args = ap.parse_args()

    for p in (args.csv1, args.csv2):
        if not os.path.isfile(p):
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # 文字列として読み込み（空文字もそのまま扱うため keep_default_na=False）
    df1 = pd.read_csv(args.csv1, dtype=str, keep_default_na=False)
    df2 = pd.read_csv(args.csv2, dtype=str, keep_default_na=False)

    # 列の合わせ込み：列名と順序は 1.csv 基準
    if list(df1.columns) != list(df2.columns):
        # input2.csv が input1.csv の列をすべて持っていれば並べ替え／不足があればエラー
        missing = [c for c in df1.columns if c not in df2.columns]
        if missing:
            print(f"Error: input2.csv に次の列がありません: {missing}", file=sys.stderr)
            sys.exit(1)
        df2 = df2[df1.columns]  # 余分な列は無視し、input1.csv の列順に合わせる

    # input2.csv の行集合（タプル化）を作成
    set2 = set(map(tuple, df2.values.tolist()))

    # input1.csv 各行が input2.csv に存在するか判定
    result = [1 if tuple(row) in set2 else 0 for row in df1.values.tolist()]

    # ヘッダー無し・1列で保存
    pd.Series(result, dtype=int).to_csv(args.output, index=False, header=False)

    print(f"Done: wrote {len(result)} lines to {args.output}")

if __name__ == "__main__":
    main()

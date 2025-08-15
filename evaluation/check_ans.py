#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
import argparse

def is_one(val: str) -> bool:
    s = str(val).strip()
    if s == "":
        return False
    # 数値として1と等しいかを判定（"1", "01", "1.0" 等を許容）
    try:
        return float(s) == 1.0
    except ValueError:
        return False

def read_first_column(path: str):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:  # 空行
                yield ""
            else:
                yield row[0]

def main():
    ap = argparse.ArgumentParser(
        description="2つの1列CSVを比較し、両方1の行番号を出力。最後に総数を表示。"
    )
    ap.add_argument("csv1", help="1つ目のCSV（1列想定）")
    ap.add_argument("csv2", help="2つ目のCSV（1列想定）")
    ap.add_argument("--skip-header", action="store_true",
                    help="先頭1行をヘッダーとしてスキップ")
    ap.add_argument("--zero-based", action="store_true",
                    help="行番号を0始まりで出力（既定は1始まり）")
    args = ap.parse_args()

    col1 = list(read_first_column(args.csv1))
    col2 = list(read_first_column(args.csv2))

    # 長さが異なる場合は短い方に合わせる
    n1, n2 = len(col1), len(col2)
    n = min(n1, n2)
    if n1 != n2:
        print(f"[WARN] 長さが異なるため短い方の {n} 行で比較します。", file=sys.stderr)

    start = 1 if not args.zero_based else 0
    offset = 1 if args.skip_header else 0  # ヘッダー分をスキップ

    matched_indices = []
    for i in range(offset, n):
        if is_one(col1[i]) and is_one(col2[i]):
            # 表示用の行番号（1始まり/0始まり）
            idx = i if args.zero_based else i
            # ヘッダーをスキップしても行番号は元の行番号で数える想定
            # 例：skip-header時、2行目が一致なら1始まりで「2」を出す
            print(start + i - (0 if args.zero_based else 0))
            matched_indices.append(i)

    # 総数
    print(f"TOTAL {len(matched_indices)}")

if __name__ == "__main__":
    main()

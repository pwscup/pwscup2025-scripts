#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import os

def main():
    ap = argparse.ArgumentParser(description="num_* は空値→0、他列の空値行は削除")
    ap.add_argument("input_csv", help="入力CSV")
    ap.add_argument("-o", "--output", required=True, help="出力CSV")
    args = ap.parse_args()

    if not os.path.isfile(args.input_csv):
        raise SystemExit(f"File not found: {args.input_csv}")

    # 読み込み（空文字や空白だけのセルも NaN に）
    df = pd.read_csv(args.input_csv, low_memory=False)
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # num_* 列を特定
    num_cols = [c for c in df.columns if c.startswith("num_")]
    other_cols = [c for c in df.columns if c not in num_cols]

    # num_* 列は 0 埋め（数値化してから Int64 に）
    if num_cols:
        for c in num_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            df[c] = s.fillna(0).astype("Int64")

    # それ以外の列に NaN がある行は削除
    if other_cols:
        before = len(df)
        df = df.dropna(subset=other_cols, how="any")
        after = len(df)
        print(f"Removed rows with empties in non-num_* columns: {before - after}")

    # 保存（インデックスは書かない）
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output} (rows={len(df)}, cols={df.shape[1]})")

if __name__ == "__main__":
    main()

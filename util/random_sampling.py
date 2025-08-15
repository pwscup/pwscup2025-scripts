#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="ヘッダー付きCSVからランダムにN行を抽出して出力")
    ap.add_argument("input_csv", help="入力CSV（ヘッダー付き）")
    ap.add_argument("output_csv", help="出力CSV（ヘッダー付き）")
    ap.add_argument("-n", "--n", type=int, default=10000, help="抽出行数（既定: 10000）")
    ap.add_argument("--seed", type=int, default=None, help="乱数シード（省略可）")
    args = ap.parse_args()

    if args.n < 0:
        print("ERROR: N は 0 以上で指定してください。", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.input_csv):
        print(f"ERROR: ファイルが見つかりません: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    # 文字列として読み込み（空文字も保持）
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False, low_memory=False)

    if df.empty:
        # ヘッダーのみ出力
        df.to_csv(args.output_csv, index=False)
        print(f"入力にデータ行が無いため、ヘッダーのみを出力しました: {args.output_csv}")
        return

    n_avail = len(df)
    n_take = min(args.n, n_avail)

    if n_take == 0:
        # 行数0を明示指定された場合
        df.head(0).to_csv(args.output_csv, index=False)
        print(f"0 行を抽出指定のため、ヘッダーのみ出力: {args.output_csv}")
        return

    sampled = df.sample(n=n_take, random_state=args.seed, replace=False)
    # 行順はサンプルのまま。元順を保ちたい場合は sort_index() も可
    sampled.to_csv(args.output_csv, index=False)

    if n_take < args.n:
        print(f"注意: 要求N={args.n} > データ行数={n_avail} のため、{n_take} 行に丸めました。")
    print(f"完了: {args.output_csv} に {n_take} 行を書き出しました（ヘッダー付き）。")

if __name__ == "__main__":
    main()

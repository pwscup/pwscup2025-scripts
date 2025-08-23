import argparse

import pandas as pd

import stats_diff
import LR_asthma_diff
import KW_IND_diff

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("Bi_csv", help="path to Bi.csv")
    ap.add_argument("Ci_csv", help="path to Ci.csv")
    # ap.add_argument("-o", "--out", default=None, help="optional path to save the full diff table (CSV)")
    ap.add_argument("-d", "--detail", action="store_true", help="[optional] despley the details")
    args = ap.parse_args()

    # 基本統計の誤差を算出
    stats_diff_max_abs = stats_diff.eval(args.Bi_csv, args.Ci_csv, print_details=args.detail)
    print(f"stats_diff max_abs: {stats_diff_max_abs}")

    # Logistic Regressionでの誤差を算出
    LR_asthma_diff_max_abs = LR_asthma_diff.eval(args.Bi_csv, args.Ci_csv, print_details=args.detail)
    print(f"LR_asthma_diff max_abs: {LR_asthma_diff_max_abs}")
    
    # KW_IND_diff
    KW_IND_diff_max_abs = KW_IND_diff.eval(args.Bi_csv, args.Ci_csv, print_details=args.detail)
    print(f"KW_IND_diff max_abs: {KW_IND_diff_max_abs}")

    # 重み付き合計
    total = 40 * stats_diff_max_abs + 20 * LR_asthma_diff_max_abs + 20 * KW_IND_diff_max_abs
    print(f"Total loss: {total}")

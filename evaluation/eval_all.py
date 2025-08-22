import argparse

import pandas as pd

import stats_diff
import LR_asthma_diff
import KW_IND_diff


def eval_Ci_utility(path_to_Bi_csv, path_to_Ci_csv, 
                    print_details=False)->float:

    # 基本統計の誤差を算出
    stats_diff_max_abs = stats_diff.eval(path_to_Bi_csv, path_to_Ci_csv, 
                                         print_details=print_details)
    print(f"stats_diff max_abs: {stats_diff_max_abs}")

    # Logistic Regressionでの誤差を算出
    LR_asthma_diff_max_abs = LR_asthma_diff.eval(path_to_Bi_csv, path_to_Ci_csv, 
                                                 print_details=print_details)
    print(f"LR_asthma_diff max_abs: {LR_asthma_diff_max_abs}")
    
    # KW_IND_diff
    KW_IND_diff_max_abs = KW_IND_diff.eval(path_to_Bi_csv, path_to_Ci_csv, 
                                           print_details=print_details)
    print(f"KW_IND_diff max_abs: {KW_IND_diff_max_abs}")

    # 重み付きutility
    Ci_utility = 40 * (1-stats_diff_max_abs) + 20 * (1-LR_asthma_diff_max_abs) + 20 * (1-KW_IND_diff_max_abs)
    print(f"Ci utility: {Ci_utility} / 80")

    return Ci_utility

def eval_Di_utility()->float:
    pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("Bi_csv", help="path to Bi.csv")
    ap.add_argument("Ci_csv", help="path to Ci.csv")
    # ap.add_argument("-o", "--out", default=None, help="optional path to save the full diff table (CSV)")
    ap.add_argument("-d", "--print-details", action="store_true", help="[optional] despley the details", default=False)
    args = ap.parse_args()

    eval_Ci_utility(args.Bi_csv, args.Ci_csv, print_details=args.print_details)
    eval_Di_utility()

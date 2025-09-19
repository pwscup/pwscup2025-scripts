import csv
import argparse
import sys, os

import pandas as pd

import stats_diff
import LR_asthma_diff
import KW_IND_diff

# モジュールの相対参照制限を強制的に回避
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'util'))
from pws_data_format import BiDataFrame, CiDataFrame, FormatError

def is_csv_file(path):
    return path.lower().endswith(".csv")

def eval_Ci_df_utility(Bi_df:pd.DataFrame, Ci_df:pd.DataFrame,
                       print_details:bool=False)->float:
    
    # 基本統計の誤差を算出
    stats_diff_max_abs = stats_diff.eval_diff_max_abs(Bi_df, Ci_df, 
                                                      print_details=print_details)
    print(f"stats_diff max_abs: {stats_diff_max_abs}")

    # Logistic Regressionでの誤差を算出
    LR_asthma_diff_max_abs = LR_asthma_diff.eval_diff_max_abs(Bi_df, Ci_df,
                                                              print_details=print_details)
    print(f"LR_asthma_diff max_abs: {LR_asthma_diff_max_abs}")

    # KW_IND_diff
    KW_IND_diff_max_abs = KW_IND_diff.eval_diff_max_abs(Bi_df, Ci_df, 
                                                        print_details=print_details)
    print(f"KW_IND_diff max_abs: {KW_IND_diff_max_abs}")

    # 重み付きutility
    Ci_utility = 40 * (1-stats_diff_max_abs) + 20 * (1-LR_asthma_diff_max_abs) + 20 * (1-KW_IND_diff_max_abs)
    print(f"Ci utility: {Ci_utility} / 80")
    
def eval_Ci_utility(path_to_Bi_csv:str, path_to_Ci_csv:str, 
                    print_details:bool=False)->float:

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
    ap.add_argument("-f", "--force", action="store_true", help="[optional]Ciにフォーマット違反があっても強制的に採点を試す")
    ap.add_argument("-d", "--print-details", action="store_true", help="[optional] despley the details", default=False)
    args = ap.parse_args()

    if not os.path.isfile(args.Bi_csv):
        raise FileNotFoundError(f"{args.Bi_csv} は存在しません。")
    
    if not os.path.isfile(args.Ci_csv):
        raise FileNotFoundError(f"{args.Ci_csv} は存在しません。")

    if not is_csv_file(args.Bi_csv):
        raise TypeError(f"{args.Bi_csv}はCSVファイルではありません")
    
    if not is_csv_file(args.Ci_csv):
        raise TypeError(f"{args.Ci_csv}はCSVファイルではありません")

    try:
        Bi_df = BiDataFrame.read_csv(args.Bi_csv)
    except ExceptionGroup as eg:
        print(f"採点できません。Biが異常です。")
        raise eg # エラーを吐いて強制終了

    try:
        # Ci_csvをCiと解釈できるか試す。軽微なフォーマット違反は修正する
        Ci_df = CiDataFrame.read_csv(args.Ci_csv)
        eval_Ci_df_utility(Bi_df, Ci_df, print_details=args.print_details)
    except* FormatError as e: # 修正不能なフォーマット違反があった場合
        print(f"Ciのフォーマットに異常があります:")
        # raiseしてしまうと次のif文が実行されない
        for sub_e in e.exceptions:
            print(sub_e)
        
        if args.force:
            print("採点を強行します")
            eval_Ci_utility(args.Bi_csv, args.Ci_csv, print_details=args.print_details)
            # ここでエラーが起きてもどうしようもないのでこれ以上は何もしない
    except* Exception as e: # おそらく発生しない処理
        print(f"{args.Ci_csv}を読み込めませんでした:")
        for sub_e in e.exceptions:
            print(sub_e)

    
    eval_Di_utility()

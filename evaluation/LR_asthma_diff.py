#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LR_asthma_diff.py
- 2つのCSVを入力し、各CSVに対して LR_asthma.py 相当のロジスティック回帰（asthma_flag）を実行。
- 出力表（term, coef, p_value, OR_norm, CI_low_norm, CI_high_norm, VIF_norm）を作成。
- 2表を term の和集合で突き合わせ、(file2 - file1) の差分表を表示（列名は元のまま）。
- 片側にしか無い term は無い側を 0 扱い。NaN も 0 にする。
- AUC は各CSVで算出し、AUC_DIFF を併せて出力。
- 最後に、表全体（coef, p_value, OR_norm, CI_low_norm, CI_high_norm, VIF_norm）の
  絶対差の最大値を MAX_ABS_DIFF として出力。

依存: numpy, pandas, statsmodels, scikit-learn
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd

# モジュールの相対参照制限を強制的に回避
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'analysis'))
import LR_asthma


COLS_ORDER = LR_asthma.COLS_ORDER

def eval_diff_max_abs(df1: pd.DataFrame, df2: pd.DataFrame, 
              target="asthma_flag", test_size=0.2, random_state=42, ensure_terms="ETHNICITY_hispanic",
              out=None, print_details=False):
    """
    DataFrameを2つ受け取って、それらのLRでの差が最大になる変数での差を出力

    Paramters:
    path_to_csv1 (str): CSVファイルその1
    path_to_csv2 (str): CSVファイルその1
    out (str): 各指標の差を保存したい場所を指定。テーブル同士の差が書き込まれる
    print_details (bool): 詳細の結果を表示するかどうか

    Returns:
    最も大きい差 (float)
    """

    diff = eval_diff(df1, df2, target, test_size, random_state, ensure_terms, out, print_details)

    # 最大絶対差（表の全数値列から算出。AUCは含めない）
    max_abs = float(np.nanmax(np.abs(diff[COLS_ORDER[1:]].to_numpy()))) if not diff.empty else 0.0

    return max_abs

def eval_diff(df1: pd.DataFrame, df2: pd.DataFrame, 
              target="asthma_flag", test_size=0.2, random_state=42, ensure_terms="ETHNICITY_hispanic",
              out=None, print_details=False):
    """
    DataFrameを2つ受け取って、それぞれの各変数を目的としたLRを実行して、差を評価

    Paramters:
    kw_table1 (pd.DataFrame): DataFrameその1
    kw_table2 (pd.DataFrame): DataFrameその2
    out (str): 各差を保存したい場所を指定。テーブル同士の差が書き込まれる
    print_details (bool): 詳細の結果を表示するかどうか

    Returns:
    その2 - その1 (pd.DataFrame)
    """
    # 片方ずつ LR 実行
    t1, auc1, terms1 = LR_asthma.run_lr_table(
        df1, target=target, test_size=test_size,
        random_state=random_state, ensure_terms=ensure_terms
    )
    t2, auc2, terms2 = LR_asthma.run_lr_table(
        df2, target=target, test_size=test_size,
        random_state=random_state, ensure_terms=ensure_terms
    )

    # term 和集合で reindex（欠側は0）
    terms_all = sorted(set(terms1).union(set(terms2)))
    t1i = t1.set_index("term").reindex(terms_all).fillna(0.0)
    t2i = t2.set_index("term").reindex(terms_all).fillna(0.0)

    # 差分（file2 - file1）：列名は元と同じ（値が差分）
    diff = (t2i[COLS_ORDER[1:]] - t1i[COLS_ORDER[1:]]).reset_index()
    diff = diff.rename(columns={"index": "term"})
    diff = diff[COLS_ORDER]  # 列順を固定
    # NaN 安全化（念のため）
    for c in COLS_ORDER[1:]:
        diff[c] = pd.to_numeric(diff[c], errors="coerce").fillna(0.0)

    # 出力
    if print_details:
        with pd.option_context("display.max_columns", None,
                               "display.width", None,
                               "display.float_format", lambda x: f"{x:.6g}"):
            print(f"AUC (file1): {auc1:.6f}")
            print(f"AUC (file2): {auc2:.6f}")
            print(f"AUC_DIFF   : {auc2 - auc1:.6f}")

            print("\n=== Logistic regression DIFF (file2 - file1) — const excluded; values are differences ===")
            print(diff.to_string(index=False))

    # 保存（任意）
    if out:
        diff.to_csv(out, index=False)

    return diff

def eval(path_to_csv1:str, path_to_csv2:str,
         target="asthma_flag", test_size=0.2, random_state=42, ensure_terms="ETHNICITY_hispanic",
         out=None, print_details=False):
    """
    CSVファイルへのパスを2つ受け取って、それらのLRでの差が最大になる変数での差を出力

    Paramters:
    path_to_csv1 (str): CSVファイルその1
    path_to_csv2 (str): CSVファイルその2
    out (str): 各指標の差を保存したい場所を指定。テーブル同士の差が書き込まれる
    print_details (bool): 詳細の結果を表示するかどうか

    Returns:
    最も大きい差 (float)
    """
    
    df1 = pd.read_csv(path_to_csv1, dtype=str, keep_default_na=False)
    df2 = pd.read_csv(path_to_csv2, dtype=str, keep_default_na=False)

    max_abs = eval_diff_max_abs(df1, df2, target, test_size, random_state, ensure_terms, out, print_details)

    return max_abs

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LR_asthma 差分: 2つのCSVの 0〜1 指標（coefは実数）を (file2 - file1) で出力。欠側termは0埋め。NaNは0。")
    ap.add_argument("csv1", help="1つ目のCSV（baseline）")
    ap.add_argument("csv2", help="2つ目のCSV（to compare）")
    ap.add_argument("--target", default="asthma_flag", help="binary target column (default: asthma_flag)")
    ap.add_argument("--test-size", type=float, default=0.2, help="holdout ratio (default: 0.2)")
    ap.add_argument("--random-state", type=int, default=42, help="random seed (default: 42)")
    ap.add_argument("--ensure-terms", default="ETHNICITY_hispanic",
                    help="常に出力に含めたい term 名（カンマ区切り）")
    ap.add_argument("-o", "--out", default=None, help="差分表をCSV保存（任意）")
    args = ap.parse_args()

    max_abs = eval(args.csv1, args.csv2, 
                   args.target, args.test_size, args.random_state, 
                   args.ensure_terms, args.out, True)
    print(f"\nMAX_ABS_DIFF {max_abs:.6g}")

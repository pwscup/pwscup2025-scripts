#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KW_IND_diff.py
- 2つのCSVを入力し、各CSVに対して KW_IND.py と同等の 0～1 指標群を算出
  （H_norm, minus_log10_p_norm, epsilon2, eta2_kw, rank_eta2, A_pair_avg, A_pair_sym）
- metric ごとに (file2 - file1) の差分を出力（group_sizesは出さない）
- NaN は 0 に置換
- 末尾に全差分の最大絶対値を出力

オプションは KW_IND.py と同様（age 列名、metrics、custom-bins 等）
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd

# モジュールの相対参照制限を強制的に回避
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'analysis'))
import KW_IND

# ==== 既定値（KW_IND.py と同じ） ====
TARGET_DEFAULT = KW_IND.TARGET_DEFAULT
DEFAULT_BINS = KW_IND.DEFAULT_BINS
METRICS_DEFAULT = KW_IND.METRICS_DEFAULT

DEFAULT_P_NORM = KW_IND.DEFAULT_P_NORM # 'arctan' / 'exp' / 'log1p'
DEFAULT_P_SCALE = KW_IND.DEFAULT_P_SCALE
DEFAULT_P_CAP = KW_IND.DEFAULT_P_CAP

NUM_COLS = KW_IND.NUM_COLS

def eval(path_to_csv1, path_to_csv2, print_details=False):
    # fix optional hyperparameters with their default values
    age_col = TARGET_DEFAULT
    metrics = ",".join(METRICS_DEFAULT)
    custom_bins = ""
    min_per_group = 2
    p_norm = DEFAULT_P_NORM
    p_scale = DEFAULT_P_SCALE
    p_cap = DEFAULT_P_CAP

    t1 = compute_kw_table(
        path_to_csv1, age_col, metrics, custom_bins,
        min_per_group, p_norm, p_scale, p_cap
    )
    t2 = compute_kw_table(
        path_to_csv2, age_col, metrics, custom_bins,
        min_per_group, p_norm, p_scale, p_cap
    )

    # metric の和集合で揃え、欠側は0埋め
    metrics_all = sorted(set(t1["metric"]).union(set(t2["metric"])))
    t1i = t1.set_index("metric").reindex(metrics_all).fillna(0.0)
    t2i = t2.set_index("metric").reindex(metrics_all).fillna(0.0)

    # 差分 (file2 - file1)
    diff = (t2i[NUM_COLS] - t1i[NUM_COLS]).reset_index()
    diff = diff.rename(columns={"index": "metric"}).sort_values("metric", kind="mergesort").reset_index(drop=True)

    # 表示（KW_IND 風ヘッダ、group_sizes なし）
    if print_details:
        with pd.option_context("display.max_columns", None,
                               "display.width", None,
                               "display.float_format", lambda x: f"{x:.6g}"):
            print("\n=== KW_IND Diff (file2 - file1) — H_norm / p_norm / effect sizes (0–1) ===")
            print(diff.to_string(index=False))

    # 全差分の最大絶対値
    max_abs = float(np.nanmax(np.abs(diff[NUM_COLS].to_numpy()))) if not diff.empty else 0.0
    if print_details:
        print(f"\nMAX_ABS_DIFF {max_abs:.6g}")

    # 保存オプション
    # if args.out:
    #     diff.to_csv(args.out, index=False)

    return max_abs

def eval(path_to_csv1:str, path_to_csv2:str,
         age_col=TARGET_DEFAULT, metrics=",".join(METRICS_DEFAULT),
         custom_bins="", min_per_group=2, 
         p_norm=DEFAULT_P_NORM, p_scale=DEFAULT_P_SCALE, 
         p_cap=DEFAULT_P_CAP, out=None, print_details=False) -> float:
    """
    CSVファイルへのパスを2つ受け取って、それらのKW指標のうち最も大きい差を返す

    Paramters:
    path_to_csv1 (str): CSVファイルその1
    path_to_csv2 (str): CSVファイルその2
    out (str): 各指標の差を保存したい場所を指定。テーブル同士の差が書き込まれる
    print_details (bool): 詳細の結果を表示するかどうか

    Returns:
    最も大きいKW指標の差 (float)
    """
    
    # csvファイルをpandasのDataFrameとして読み込み
    df1 = pd.read_csv(path_to_csv1, dtype=str, keep_default_na=False)
    df2 = pd.read_csv(path_to_csv2, dtype=str, keep_default_na=False)

    max_abs = eval_diff_max_abs(df1, df2, age_col, metrics,
         custom_bins, min_per_group, p_norm, p_scale, p_cap, 
         out, print_details)

    return max_abs

def eval_diff_max_abs(df1: pd.DataFrame, df2:pd.DataFrame, 
                      age_col=TARGET_DEFAULT, 
                      metrics=",".join(METRICS_DEFAULT),
                      custom_bins="", min_per_group=2, 
                      p_norm=DEFAULT_P_NORM, p_scale=DEFAULT_P_SCALE, 
                      p_cap=DEFAULT_P_CAP,
                      out=None, print_details=False) -> float:
    """
    データフレームを2つ受け取って、最も大きいKW指標の差を返す

    Paramters:
    df1 (pd.DataFrame): データフレームその1
    df2 (pd.DataFrame): データフレームその2
    out (str): 各指標の差を保存したい場所を指定。各指標の差が書き込まれる
    print_details (bool): 詳細の結果を表示するかどうか

    Returns:
    最も大きいKW指標の差 (float)
    """

    diff = eval_diff(df1, df2, age_col, metrics,
         custom_bins, min_per_group, p_norm, p_scale, p_cap, 
         out, print_details)


    # 全差分の最大絶対値
    max_abs = float(np.nanmax(np.abs(diff[NUM_COLS].to_numpy()))) if not diff.empty else 0.0

    return max_abs

def eval_diff(df1: pd.DataFrame, df2:pd.DataFrame, 
                      age_col=TARGET_DEFAULT, 
                      metrics=",".join(METRICS_DEFAULT),
                      custom_bins="", min_per_group=2, 
                      p_norm=DEFAULT_P_NORM, p_scale=DEFAULT_P_SCALE, 
                      p_cap=DEFAULT_P_CAP,
                      out=None, print_details=False) -> pd.DataFrame:
    """
    データフレームを2つ受け取って、各指標の差を計算する

    Paramters:
    df1 (pd.DataFrame): データフレームその1
    df2 (pd.DataFrame): データフレームその2
    out (str): 各指標の差を保存したい場所を指定。各指標の差が書き込まれる
    print_details (bool): 詳細の結果を表示するかどうか

    Returns:
    テーブルその2の指標 - テーブルその1の指標 (pd.DataFrame)
    """
    # DataFrameからKW指標をまとめたテーブルを算出
    kw_table1 = KW_IND.compute_kw_table(
        df1, age_col, metrics, custom_bins,
        min_per_group, p_norm, p_scale, p_cap
    )
    kw_table2 = KW_IND.compute_kw_table(
        df2, age_col, metrics, custom_bins,
        min_per_group, p_norm, p_scale, p_cap
    )

    # metric の和集合で揃え、欠側は0埋め
    metrics_all = sorted(set(kw_table1["metric"]).union(set(kw_table2["metric"])))
    t1i = kw_table1.set_index("metric").reindex(metrics_all).fillna(0.0)
    t2i = kw_table2.set_index("metric").reindex(metrics_all).fillna(0.0)

    # 差分 (file2 - file1)
    diff = (t2i[NUM_COLS] - t1i[NUM_COLS]).reset_index()
    diff = diff.rename(columns={"index": "metric"}).sort_values("metric", kind="mergesort").reset_index(drop=True)

    # 表示（KW_IND 風ヘッダ、group_sizes なし）
    if print_details:
        with pd.option_context("display.max_columns", None,
                               "display.width", None,
                               "display.float_format", lambda x: f"{x:.6g}"):
            print("\n=== KW_IND Diff (file2 - file1) — H_norm / p_norm / effect sizes (0–1) ===")
            print(diff.to_string(index=False))

    # 保存オプション
    if out:
        diff.to_csv(out, index=False)

    return diff

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="KW_IND 差分: 2つのCSVの 0～1 指標を (file2 - file1) で出力し、最後に最大絶対差を表示（group_sizes非表示）。")
    ap.add_argument("csv1", help="1つ目のCSV（baseline）")
    ap.add_argument("csv2", help="2つ目のCSV（to compare）")
    ap.add_argument("--age-col", default=TARGET_DEFAULT, help="年齢列名（既定: AGE）")
    ap.add_argument("--metrics", default=",".join(METRICS_DEFAULT),
                    help="解析する指標列（カンマ区切り）既定: " + ",".join(METRICS_DEFAULT))
    ap.add_argument("--custom-bins", default="",
                    help=f"臨床カスタム区切り（例: 0,18,45,65,75,200）。未指定なら {DEFAULT_BINS} を使用")
    ap.add_argument("--min-per-group", type=int, default=2, help="各群の最小サンプル数（既定: 2）")
    ap.add_argument("--p-norm", choices=["arctan", "exp", "log1p"], default=DEFAULT_P_NORM,
                    help="minus_log10_p の 0–1 正規化方式（既定: arctan）")
    ap.add_argument("--p-scale", type=float, default=DEFAULT_P_SCALE,
                    help="arctan/exp のスケール（大きいほどゆっくり1に近づく）既定: 10")
    ap.add_argument("--p-cap", type=float, default=DEFAULT_P_CAP,
                    help="log1p 正規化の上限（既定: 300）")
    ap.add_argument("-o", "--out", default=None, help="差分テーブルをCSV保存するパス（任意）")
    args = ap.parse_args()

    # 2つのKWテーブルの差を評価
    max_abs = eval(args.csv1, args.csv2, args.age_col, args.metrics,
                   args.custom_bins, args.min_per_group, 
                   args.p_norm, args.p_scale, args.p_cap,
                   args.out, True)
    print(f"\nMAX_ABS_DIFF {max_abs:.6g}")

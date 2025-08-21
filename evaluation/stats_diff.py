#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stats_diff.py
- 2つのCSVを入力し、各CSVの統計量（0〜1、相関係数は-1〜1）を算出。
- 体裁を揃えた長形式テーブルにして差 (file2 - file1) を出力。
- term が片側にしかない場合は、無い側の値を 0 とみなす。
- NaN は 0 に置換。
- 末尾に全差分の最大絶対値を出力。

出力セクション:
  === DIFF: Normalized (0-1) ===
      term,value_diff
  === DIFF: Correlation (-1..1) ===
      term,value_diff
  MAX_ABS_DIFF <float>

注:
- 数値列の判定: 非空のうち 95% 以上が数値化できれば「数値」扱い。
- 数値統計の 0-1 化: 同一ファイル内で「列×統計量」ベクトルごとに min-max 正規化。
- カテゴリ列は各カテゴリ値の「比率（0〜1）」。
- 相関は Pearson。数値列が2未満の場合はスキップ。
"""

import argparse
import sys
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


# ---------- I/O ----------

def read_csv_all_str(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# ---------- 型推定 ----------

def detect_numeric_columns(df: pd.DataFrame, thresh: float = 0.95) -> List[str]:
    num_cols = []
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        nonblank = s != ""
        if nonblank.sum() == 0:
            continue
        conv = pd.to_numeric(s[nonblank], errors="coerce")
        if conv.notna().mean() >= thresh:
            num_cols.append(c)
    return num_cols


# ---------- 数値統計（0-1化） ----------

def numeric_stats_norm01(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """
    各数値列の mean/std/q25/median/q75 を算出し、
    統計量ごと（mean群, std群, …）に min-max 正規化して 0-1 にする。
    戻りは長形式: columns=['term','value'] （term例: 'NUM:mean:AGE'）
    """
    if not num_cols:
        return pd.DataFrame(columns=["term", "value"])

    # 数値化（空文字→NaN）
    dfn = df.copy()
    for c in num_cols:
        dfn[c] = pd.to_numeric(dfn[c].replace({"": np.nan}), errors="coerce")

    stats = {}
    # 各統計量を dict[col] に
    stats["mean"] = {c: float(dfn[c].mean(skipna=True)) for c in num_cols}
    stats["std"]  = {c: float(dfn[c].std(skipna=True)) for c in num_cols}
    stats["q25"]  = {c: float(dfn[c].quantile(0.25, interpolation="linear")) for c in num_cols}
    stats["q50"]  = {c: float(dfn[c].quantile(0.50, interpolation="linear")) for c in num_cols}
    stats["q75"]  = {c: float(dfn[c].quantile(0.75, interpolation="linear")) for c in num_cols}

    rows = []
    # 統計量ごとに min-max 正規化
    for stat_name, mapping in stats.items():
        vals = pd.Series(mapping, dtype=float)
        # min-max（全NaNへの対処）
        if vals.notna().any():
            vmin = float(vals.min())
            vmax = float(vals.max())
            rng = vmax - vmin
            if rng == 0:
                norm = vals.apply(lambda x: 0.0 if np.isfinite(x) else np.nan)
            else:
                norm = (vals - vmin) / rng
        else:
            norm = pd.Series({k: np.nan for k in mapping.keys()}, dtype=float)

        for col, v in norm.items():
            term = f"NUM:{stat_name}:{col}"
            rows.append((term, 0.0 if pd.isna(v) else float(v)))

    out = pd.DataFrame(rows, columns=["term", "value"])
    out = out.sort_values("term", kind="mergesort").reset_index(drop=True)
    return out


# ---------- カテゴリ統計（比率0-1） ----------

def categorical_ratios(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """
    各カテゴリ列について、各値の比率（0-1）を計算。
    戻りは長形式: term='CAT:<col>=<val>', value=ratio
    """
    rows = []
    n = len(df)
    if n == 0 or not cat_cols:
        return pd.DataFrame(columns=["term", "value"])

    for c in cat_cols:
        s = df[c].astype(str).str.strip()
        counts = s.value_counts(dropna=False)
        for val, cnt in counts.items():
            ratio = float(cnt) / float(n)
            term = f"CAT:{c}={val}"
            rows.append((term, ratio))

    out = pd.DataFrame(rows, columns=["term", "value"])
    out = out.sort_values("term", kind="mergesort").reset_index(drop=True)
    return out


# ---------- 相関（-1〜1） ----------

def pearson_corr_pairs(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """
    数値列間の Pearson 相関（上三角）を長形式で返す。
    term='CORR:<c1>|<c2>', value=r (-1..1)
    """
    if len(num_cols) < 2:
        return pd.DataFrame(columns=["term", "value"])

    dfn = df[num_cols].apply(pd.to_numeric, errors="coerce")
    corr = dfn.corr(method="pearson", min_periods=1)
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            r = corr.loc[c1, c2]
            v = 0.0 if pd.isna(r) else float(r)
            term = f"CORR:{c1}|{c2}"
            rows.append((term, v))
    out = pd.DataFrame(rows, columns=["term", "value"])
    out = out.sort_values("term", kind="mergesort").reset_index(drop=True)
    return out


# ---------- 1ファイル分の長形式テーブル作成 ----------

def build_long_table_for_csv(csv_path: str) -> pd.DataFrame:
    df = read_csv_all_str(csv_path)
    num_cols = detect_numeric_columns(df, thresh=0.95)
    cat_cols = [c for c in df.columns if c not in num_cols]

    t_num = numeric_stats_norm01(df, num_cols)
    t_cat = categorical_ratios(df, cat_cols)
    t_cor = pearson_corr_pairs(df, num_cols)

    # 同じ「stats.py ライク」な長形式 term/value
    all_terms = pd.concat([t_num, t_cat, t_cor], axis=0, ignore_index=True)
    # NaN→0
    all_terms["value"] = all_terms["value"].fillna(0.0).astype(float)
    # term 重複が万一あれば平均で集約（通常は発生しない想定）
    all_terms = all_terms.groupby("term", as_index=False)["value"].mean()
    all_terms = all_terms.sort_values("term", kind="mergesort").reset_index(drop=True)
    return all_terms

def eval(path_to_csv1, path_to_csv2, print_details=False):
    tbl1 = build_long_table_for_csv(path_to_csv1)
    tbl2 = build_long_table_for_csv(path_to_csv2)

    # term全体の和集合を作り、無い方は0で埋めて差分（csv2 - csv1）
    terms_all = sorted(set(tbl1["term"]).union(set(tbl2["term"])))
    d1 = tbl1.set_index("term")["value"].reindex(terms_all).fillna(0.0)
    d2 = tbl2.set_index("term")["value"].reindex(terms_all).fillna(0.0)
    diff = (d2 - d1).astype(float)

    # 表示を2セクションに分ける（Normalized と Correlation）
    diff_df = diff.reset_index()
    diff_df.columns = ["term", "value_diff"]

    # NaN→0（念のため）
    diff_df["value_diff"] = diff_df["value_diff"].fillna(0.0)

    # セクション分割
    diff_norm = diff_df[diff_df["term"].str.startswith(("NUM:", "CAT:"))].copy()
    diff_corr = diff_df[diff_df["term"].str.startswith("CORR:")].copy()

    # 出力（stats.py風）
    if print_details:
        with pd.option_context("display.max_columns", None, "display.width", None, "display.float_format", lambda x: f"{x:.6g}"):
            print("=== DIFF: Normalized (0-1) ===")
            if not diff_norm.empty:
                print(diff_norm.to_string(index=False))
            else:
                print("(no normalized terms)")

            print("\n=== DIFF: Correlation (-1..1) ===")
            if not diff_corr.empty:
                print(diff_corr.to_string(index=False))
            else:
                print("(no correlation terms)")

    # 最大絶対差
    max_abs = float(diff_df["value_diff"].abs().max()) if not diff_df.empty else 0.0
    if print_details:
        print(f"\nMAX_ABS_DIFF {max_abs:.6g}")
    
    return max_abs

    # 保存オプション
    # if path_to_out:
    #     diff_df.to_csv(path_to_out, index=False)

# ---------- 差分と出力 ----------

def main():
    ap = argparse.ArgumentParser(description="Compute stats for two CSVs and print their difference in stats.py-like format (missing terms treated as 0).")
    ap.add_argument("csv1", help="first CSV")
    ap.add_argument("csv2", help="second CSV")
    ap.add_argument("-o", "--out", default=None, help="optional path to save the full diff table (CSV)")
    args = ap.parse_args()

    tbl1 = build_long_table_for_csv(args.csv1)
    tbl2 = build_long_table_for_csv(args.csv2)

    # term全体の和集合を作り、無い方は0で埋めて差分（csv2 - csv1）
    terms_all = sorted(set(tbl1["term"]).union(set(tbl2["term"])))
    d1 = tbl1.set_index("term")["value"].reindex(terms_all).fillna(0.0)
    d2 = tbl2.set_index("term")["value"].reindex(terms_all).fillna(0.0)
    diff = (d2 - d1).astype(float)

    # 表示を2セクションに分ける（Normalized と Correlation）
    diff_df = diff.reset_index()
    diff_df.columns = ["term", "value_diff"]

    # NaN→0（念のため）
    diff_df["value_diff"] = diff_df["value_diff"].fillna(0.0)

    # セクション分割
    diff_norm = diff_df[diff_df["term"].str.startswith(("NUM:", "CAT:"))].copy()
    diff_corr = diff_df[diff_df["term"].str.startswith("CORR:")].copy()

    # 出力（stats.py風）
    with pd.option_context("display.max_columns", None, "display.width", None, "display.float_format", lambda x: f"{x:.6g}"):
        print("=== DIFF: Normalized (0-1) ===")
        if not diff_norm.empty:
            print(diff_norm.to_string(index=False))
        else:
            print("(no normalized terms)")

        print("\n=== DIFF: Correlation (-1..1) ===")
        if not diff_corr.empty:
            print(diff_corr.to_string(index=False))
        else:
            print("(no correlation terms)")

    # 最大絶対差
    max_abs = float(diff_df["value_diff"].abs().max()) if not diff_df.empty else 0.0
    print(f"\nMAX_ABS_DIFF {max_abs:.6g}")

    # 保存オプション
    if args.out:
        diff_df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()

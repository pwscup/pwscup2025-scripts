#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# mia.py — nearest-neighbor marking
# 入力: Ai.csv, Ci.csv（いずれもヘッダーあり）
# 出力: -o で指定（既定 Fij.csv）。1列・ヘッダー無し。長さは Ai の行数-1。
# 仕様:
#  - 共通列だけで比較（列名完全一致の交差）
#  - 数値列: df1 側の min-max で [0,1] 正規化（ゼロ分散は無視）
#  - カテゴリ列: OneHot 化
#  - 最近傍: sklearn NearestNeighbors(metric='manhattan', n_neighbors=1)
#  - Ci の各行が最も近い df1 の行インデックスを集合化し、その位置を 1、他を 0 にして出力

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def read_csv_str(path: str) -> pd.DataFrame:
    # 文字列として読み、空欄は空文字で保持（NaNにしない）
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def infer_numeric_mask(df: pd.DataFrame, cols: List[str], thresh: float = 0.95) -> pd.Series:
    """
    df（=df1）で「数値列っぽいか」を判定。非空のうち数値化できる割合が thresh 以上なら numeric。
    """
    mask = []
    for c in cols:
        s = df[c].astype(str).str.strip()
        nonblank = s != ""
        if nonblank.sum() == 0:
            mask.append(False)
            continue
        conv = pd.to_numeric(s[nonblank], errors="coerce")
        frac = conv.notna().mean()
        mask.append(frac >= thresh)
    return pd.Series(mask, index=cols)


def preprocess_numeric(df1: pd.DataFrame, df2: pd.DataFrame, num_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    数値列: df1 の min-max でスケーリング。df2 も同じパラメータで変換。
    変換不能やゼロ分散列は除外。
    """
    if not num_cols:
        return (pd.DataFrame(index=df1.index), pd.DataFrame(index=df2.index))

    X1n_list, X2n_list, kept = [], [], []
    for c in num_cols:
        # to_numeric（空文字は NaN に）
        a1 = pd.to_numeric(df1[c].replace({"": np.nan}), errors="coerce")
        a2 = pd.to_numeric(df2[c].replace({"": np.nan}), errors="coerce")
        # df1 側に有効値がなければ使わない
        if a1.notna().sum() == 0:
            continue
        vmin = a1.min()
        vmax = a1.max()
        rng = vmax - vmin
        if not np.isfinite(vmin) or not np.isfinite(vmax) or rng == 0:
            # ゼロ分散や無限は除外
            continue
        # スケーリング
        s1 = (a1 - vmin) / rng
        s2 = (a2 - vmin) / rng
        # 欠損は df1 の中央値で補完（スケール後）
        med = float(np.nanmedian(s1.values))
        s1 = s1.fillna(med)
        s2 = s2.fillna(med)
        X1n_list.append(s1.to_frame(name=c))
        X2n_list.append(s2.to_frame(name=c))
        kept.append(c)

    if not kept:
        return (pd.DataFrame(index=df1.index), pd.DataFrame(index=df2.index))

    X1n = pd.concat(X1n_list, axis=1)
    X2n = pd.concat(X2n_list, axis=1)
    return (X1n, X2n)


def preprocess_categorical(df1: pd.DataFrame, df2: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    カテゴリ列: 前後空白除去・小文字化・欠損は "___missing___" に統一 → 連結して one-hot → 分割
    """
    if not cat_cols:
        return (pd.DataFrame(index=df1.index), pd.DataFrame(index=df2.index))

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            s = out[c].astype(str).str.strip().str.lower()
            s = s.replace({"": "___missing___"})
            out[c] = s
        return out

    df1c = clean(df1[cat_cols])
    df2c = clean(df2[cat_cols])

    both = pd.concat([df1c, df2c], axis=0, ignore_index=True)
    dummies = pd.get_dummies(both, dummy_na=False)
    n1 = len(df1c)
    X1c = dummies.iloc[:n1, :].reset_index(drop=True)
    X2c = dummies.iloc[n1:, :].reset_index(drop=True)
    # 列順は同じ（同じ one-hot を共有）
    return (X1c, X2c)


def build_feature_matrices(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    共通列のみ使用し、数値/カテゴリを前処理して結合。float32 の numpy 行列を返す。
    """
    common_cols = [c for c in df1.columns if c in df2.columns]
    if not common_cols:
        # 何も比較できない → ゼロ次元
        return np.empty((len(df1), 0), dtype=np.float32), np.empty((len(df2), 0), dtype=np.float32)

    num_mask = infer_numeric_mask(df1, common_cols)  # df1 を基準に数値列を推定
    num_cols = [c for c in common_cols if num_mask[c]]
    cat_cols = [c for c in common_cols if not num_mask[c]]

    X1n, X2n = preprocess_numeric(df1, df2, num_cols)
    X1c, X2c = preprocess_categorical(df1, df2, cat_cols)

    # 列結合（順不同でOK。距離は同じ次元で計算される）
    X1 = pd.concat([X1n, X1c], axis=1)
    X2 = pd.concat([X2n, X2c], axis=1)

    # 空になった場合もある（全列ゼロ分散等）
    X1 = X1.astype("float32", copy=False)
    X2 = X2.astype("float32", copy=False)
    return X1.values, X2.values


def main():
    ap = argparse.ArgumentParser(description="For each row in Ci, mark its nearest row in Ai (1), others 0. Output: 1-column CSV.")
    ap.add_argument("input1", help="CSV with header (reference; e.g., 100000 rows)")
    ap.add_argument("input2", help="CSV with header (query)")
    ap.add_argument("-o", "--out", default="Fij.csv", help="output CSV path (1 column, no header)")
    args = ap.parse_args()

    # 読み込み（文字列保持）
    df1 = read_csv_str(args.input1)
    df2 = read_csv_str(args.input2)

    # 特徴行列作成
    X1, X2 = build_feature_matrices(df1, df2)

    # 比較次元が 0（共通列なし or すべて除外）の場合は全て 0 を出力
    m = len(df1)
    if X1.shape[1] == 0 or X2.shape[1] == 0 or m == 0:
        pd.Series(np.zeros(m, dtype=int)).to_csv(args.out, index=False, header=False)
        return

    # 最近傍検索（マンハッタン距離：数値[0,1] + one-hot を統一的に扱える）
    nn = NearestNeighbors(n_neighbors=1, metric="manhattan")
    nn.fit(X1)
    idx = nn.kneighbors(X2, n_neighbors=1, return_distance=False).ravel()  # 0-based indices into df1

    # 重複をまとめて 1
    marks = np.zeros(m, dtype=int)
    if idx.size > 0:
        marks[np.unique(idx)] = 1

    # 1列・ヘッダー無しで保存（行数 = input1 の行数）
    pd.Series(marks, dtype=int).to_csv(args.out, index=False, header=False)


if __name__ == "__main__":
    main()

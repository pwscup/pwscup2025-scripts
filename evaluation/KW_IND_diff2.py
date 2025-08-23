#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KW_IND_diff.py (SciPy-free)
- 2つのCSVを読み、AGEを臨床カスタムビンに区切って各数値列について
  Kruskal–WallisのH（タイ補正あり）とepsilon^2をNumPy/Pandasのみで算出。
- p値は扱わない（SciPy不要化のため）。0〜1の比較可能な指標のみ。
- (file2 - file1) の差分テーブルを作り、全数値列の最大絶対差を返す。
"""

import argparse
import numpy as np
import pandas as pd

DEFAULT_AGE_COL = "AGE"
# 年齢の臨床カスタムビン（右閉）：[0-12], [13-17], [18-39], [40-64], [65-79], [80+]
AGE_BINS = [0, 13, 18, 40, 65, 80, np.inf]
AGE_LABELS = ["0-12", "13-17", "18-39", "40-64", "65-79", "80+"]

METRIC_COLS = ["H_norm", "epsilon2"]  # 比較に使う0〜1指標


def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _rank_average_ties(x: np.ndarray) -> np.ndarray:
    """平均順位（ties平均）: pandasのrank(method='average')を利用。"""
    return pd.Series(x).rank(method="average").to_numpy()


def _kruskal_h_numpy(groups: list[np.ndarray]) -> tuple[float, float]:
    """
    Kruskal–Wallis H をNumPy/Pandasで実装（タイ補正あり）。
    返り値: (H_corrected, epsilon2)
    """
    # 有効値のみ
    clean = [g[np.isfinite(g)] for g in groups]
    lens = [len(g) for g in clean]
    k = sum(l > 0 for l in lens)
    if k < 2:
        return 0.0, 0.0

    # 空グループは捨てる
    nonempty = [(g, n) for g, n in zip(clean, lens) if n > 0]
    groups = [g for g, _ in nonempty]
    ni = np.array([n for _, n in nonempty], dtype=float)
    k = len(groups)

    # 連結して順位付け
    x = np.concatenate(groups, axis=0)
    N = float(x.size)
    if N < 2:
        return 0.0, 0.0

    ranks = _rank_average_ties(x)

    # 各群の順位和
    start = 0
    Ri = []
    for n in ni.astype(int):
        Ri.append(ranks[start:start + n].sum())
        start += n
    Ri = np.array(Ri, dtype=float)

    # タイ補正 T
    # 同値のカウント t_j を求め、T = sum(t^3 - t) / (N^3 - N)
    # pandasのrankを使っているので、同値は rank の重複から検出
    # （同じ値の個数カウント）
    vals, counts = np.unique(x, return_counts=True)
    ties = counts[counts > 1]
    if ties.size > 0:
        T = (np.sum(ties**3 - ties)) / (N**3 - N)
    else:
        T = 0.0

    # H
    H = (12.0 / (N * (N + 1.0))) * np.sum((Ri**2) / ni) - 3.0 * (N + 1.0)

    # 補正
    if T < 1.0:
        Hc = H / (1.0 - T)
        Hc = max(Hc, 0.0)
    else:
        Hc = max(H, 0.0)

    # 効果量（epsilon squared）
    df = k - 1.0
    eps2 = (Hc - df) / (N - k) if (N - k) > 0 else 0.0
    eps2 = float(np.clip(eps2, 0.0, 1.0))

    return float(Hc), eps2


def _h_norm(Hc: float, k: int) -> float:
    """
    H を 0〜1 に単調写像（データ非依存）。
    df = k-1 を用い、H が大きいほど 1 に近づく滑らかな写像にする。
    ここではロバストなスクイッシュ:  H_norm = 1 - exp( - Hc / (df + 1) )
    （単調・有界・入力データ規模に依存しない）
    """
    df = max(k - 1, 1)
    return float(1.0 - np.exp(- Hc / (df + 1.0)))


def _age_bins(s_age: pd.Series, age_col: str) -> pd.Series:
    age = _to_numeric_series(s_age)
    bins = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=False, include_lowest=True)
    return bins.astype(str)


def _compute_table(df: pd.DataFrame, age_col: str = DEFAULT_AGE_COL) -> pd.DataFrame:
    if age_col not in df.columns:
        # 大文字小文字ゆるく探す
        cand = next((c for c in df.columns if c.lower() == age_col.lower()), None)
        if cand is None:
            raise SystemExit(f"[ERR] age column '{age_col}' not found.")
        age_col = cand

    # 年齢ビン
    bins = _age_bins(df[age_col], age_col)
    df = df.assign(__AGE_BIN__=bins)

    # 数値列候補（AGEやビン列は除外）
    exclude = {age_col, "__AGE_BIN__"}
    num_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        col = _to_numeric_series(df[c])
        if col.notna().any():
            num_cols.append(c)

    # 各数値列についてH, epsilon2を算出
    rows = []
    for c in sorted(num_cols):
        series = _to_numeric_series(df[c])
        groups = [series[bins == lab].to_numpy() for lab in AGE_LABELS]
        Hc, eps2 = _kruskal_h_numpy(groups)
        k = sum((len(g) > 0) for g in groups)
        rows.append({"term": c, "H_norm": _h_norm(Hc, k), "epsilon2": eps2})

    tbl = pd.DataFrame(rows).sort_values("term", kind="mergesort").reset_index(drop=True)
    return tbl


def eval_diff(csv1: str, csv2: str, age_col: str = DEFAULT_AGE_COL, detail: bool = False) -> pd.DataFrame:
    df1 = pd.read_csv(csv1, dtype=str, keep_default_na=False)
    df2 = pd.read_csv(csv2, dtype=str, keep_default_na=False)

    t1 = _compute_table(df1, age_col=age_col)
    t2 = _compute_table(df2, age_col=age_col)

    terms = sorted(set(t1["term"]).union(set(t2["term"])))
    t1i = t1.set_index("term").reindex(terms).fillna(0.0)
    t2i = t2.set_index("term").reindex(terms).fillna(0.0)

    diff = (t2i[METRIC_COLS] - t1i[METRIC_COLS]).reset_index()
    diff = diff.rename(columns={"index": "term"}).sort_values("term", kind="mergesort").reset_index(drop=True)

    if detail:
        with pd.option_context("display.max_columns", None, "display.width", None,
                               "display.float_format", lambda x: f"{x:.6g}"):
            print("\n=== KW_IND_diff (file2 - file1) ===")
            print(diff.to_string(index=False))
    return diff


def eval_diff_max_abs(csv1: str, csv2: str, age_col: str = DEFAULT_AGE_COL, detail: bool = False) -> float:
    diff = eval_diff(csv1, csv2, age_col=age_col, detail=detail)
    if diff.empty:
        return 0.0
    vals = diff[METRIC_COLS].to_numpy(dtype=float)
    return float(np.nanmax(np.abs(vals)))


def eval(csv1: str, csv2: str, detail: bool = False) -> float:
    return eval_diff_max_abs(csv1, csv2, age_col=DEFAULT_AGE_COL, detail=detail)


def main():
    ap = argparse.ArgumentParser(description="KW_IND_diff (SciPy-free)")
    ap.add_argument("csv1")
    ap.add_argument("csv2")
    ap.add_argument("--age-col", default=DEFAULT_AGE_COL)
    ap.add_argument("-d", "--detail", action="store_true")
    args = ap.parse_args()
    print(eval_diff_max_abs(args.csv1, args.csv2, age_col=args.age_col, detail=args.detail))

if __name__ == "__main__":
    main()

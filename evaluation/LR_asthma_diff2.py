#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LR_asthma_diff.py (statsmodels-free)
- 2つのCSVに対して、ターゲット(既定: 'asthma_flag')のロジスティック回帰を
  scikit-learn の LogisticRegression (lbfgs) で実行（常に sklearn を使用）。
- カテゴリ列(GENDER, RACE, ETHNICITY, AGE_DECADE)は one-hot(drop_first=True)、
  その他は数値化できる列を説明変数に採用。欠損は0埋め。
- 係数βはデータ非依存で正規化:
    coef = (2/pi) * arctan(β / 2)         ∈ [-1, 1]
    coef_norm = (coef + 1) / 2             ∈ [0, 1]
- 2CSVの係数表を term の和集合で突き合わせ、(file2 - file1) の差分表を返す。
- eval(csv1, csv2, detail=False) は差分表の |coef|, |coef_norm| の最大値（float）を返す。
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings

DEFAULT_TARGET = "asthma_flag"
NUM_COLS = ["coef", "coef_norm"]  # 差分対象の数値列（順序固定）
CATEGORICAL_CANDIDATES = ["GENDER", "RACE", "ETHNICITY", "AGE_DECADE"]


# ---------- Helpers ----------
def find_col_case_insensitive(columns, name):
    low = name.lower()
    for c in columns:
        if c.lower() == low:
            return c
    return None

def arctan_squash(beta, scale=2.0):
    """β→[-1,1] (入力非依存)"""
    return (2.0 / np.pi) * np.arctan(np.asarray(beta, dtype=float) / float(scale))


def build_design(df: pd.DataFrame, target_col: str):
    """
    - CATEGORICAL_CANDIDATES を one-hot (drop_first=True)
    - 残りで数値化できる列を特徴量化（ターゲット・カテゴリ原列は除外）
    - 欠損は0埋め、全てfloat化
    - yは0/1に変換。NaNは除外
    return: X(DataFrame), y(Series), feature_names(list[str]), tcol(str)
    """
    tcol = find_col_case_insensitive(df.columns, target_col)
    if tcol is None:
        raise SystemExit(f"[ERR] target column '{target_col}' not found.")

    df_local = df.copy()

    # カテゴリ列取り出し
    cat_map = {}
    for cand in CATEGORICAL_CANDIDATES:
        c = find_col_case_insensitive(df_local.columns, cand)
        if c is not None:
            cat_map[cand] = c

    # one-hot
    X_cat = pd.DataFrame(index=df_local.index)
    for logical_name, real_col in cat_map.items():
        dmy = pd.get_dummies(df_local[real_col].astype(str), prefix=logical_name, drop_first=True)
        X_cat = pd.concat([X_cat, dmy], axis=1)

    # 数値候補（ターゲット・カテゴリ原列は除外）
    exclude_cols = set(cat_map.values()) | {tcol}
    num_candidates = [c for c in df_local.columns if c not in exclude_cols]

    X_num = pd.DataFrame(index=df_local.index)
    for c in num_candidates:
        col_num = pd.to_numeric(df_local[c], errors="coerce")
        if col_num.notna().any():
            X_num[c] = col_num

    # X / y 形成
    X = pd.concat([X_num, X_cat], axis=1)
    y = pd.to_numeric(df_local[tcol], errors="coerce")

    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    X = X.fillna(0.0)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    # yを0/1に（それ以外は除外）
    y = (y > 0.5).astype(int)

    return X, y, list(X.columns), tcol


def fit_logistic_and_table(df: pd.DataFrame, target_col: str = DEFAULT_TARGET,
                           solver: str = "auto", max_iter: int = 5000, C: float = 1.0, tol: float = 1e-4) -> pd.DataFrame:
    X, y, feat, _ = build_design(df, target_col=target_col)

    # 特徴なし or 目的変数が1クラスのみ → 全ゼロで返す（diffは安定）
    if X.shape[1] == 0 or y.nunique() < 2:
        return pd.DataFrame({
            "term": list(feat),
            "coef": np.zeros(len(feat), dtype=float),
            "coef_norm": np.full(len(feat), 0.5, dtype=float),
        }).sort_values("term", kind="mergesort").reset_index(drop=True)

    def _fit_with(sv):
        clf = LogisticRegression(solver=sv, max_iter=max_iter, C=C, tol=tol)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(X.values, y.values.astype(int))
            warned = any(isinstance(w.message, ConvergenceWarning) for w in rec)
        return clf, warned

    beta = None
    if solver in ("auto", "lbfgs"):
        try:
            clf, warned = _fit_with("lbfgs")
            if warned and solver == "auto":
                # フォールバック1: liblinear（2値に強い）
                clf, _ = _fit_with("liblinear")
            beta = clf.coef_.ravel()
        except Exception:
            beta = None

    if beta is None and solver in ("auto", "liblinear"):
        try:
            clf, _ = _fit_with("liblinear")
            beta = clf.coef_.ravel()
        except Exception:
            beta = None

    if beta is None and solver in ("auto", "saga"):
        # フォールバック2: saga（高次元も安定）
        clf, _ = _fit_with("saga")
        beta = clf.coef_.ravel()

    # それでもダメならゼロ返却
    if beta is None:
        beta = np.zeros(X.shape[1], dtype=float)

    coef = arctan_squash(beta, scale=2.0)          # [-1,1]
    coef_norm = 0.5 * (coef + 1.0)                 # [0,1]

    tbl = pd.DataFrame({"term": [str(t) for t in feat], "coef": coef, "coef_norm": coef_norm})
    return tbl.sort_values("term", kind="mergesort").reset_index(drop=True)

def eval_diff(csv1: str, csv2: str, target: str = DEFAULT_TARGET, detail: bool = False) -> pd.DataFrame:
    """
    2つのCSVの係数表の差分 (file2 - file1) を返す。
    欠側termは0埋め、出力は term の辞書順固定。
    """
    df1 = pd.read_csv(csv1, dtype=str, keep_default_na=False)
    df2 = pd.read_csv(csv2, dtype=str, keep_default_na=False)

    t1 = fit_logistic_and_table(df1, target_col=target)
    t2 = fit_logistic_and_table(df2, target_col=target)

    terms_all = sorted(set(t1["term"]).union(set(t2["term"])))
    t1i = t1.set_index("term").reindex(terms_all).fillna({"coef": 0.0, "coef_norm": 0.5})
    t2i = t2.set_index("term").reindex(terms_all).fillna({"coef": 0.0, "coef_norm": 0.5})

    diff = (t2i[NUM_COLS] - t1i[NUM_COLS]).reset_index()
    diff = diff.rename(columns={"index": "term"}).sort_values("term", kind="mergesort").reset_index(drop=True)

    if detail:
        with pd.option_context("display.max_columns", None,
                               "display.width", None,
                               "display.float_format", lambda x: f"{x:.6g}"):
            print("\n=== LR_asthma_diff (file2 - file1) ===")
            print(diff.to_string(index=False))

    return diff


def eval_diff_max_abs(csv1: str, csv2: str, target: str = DEFAULT_TARGET, detail: bool = False) -> float:
    """差分表の全数値列での最大絶対差（float）"""
    diff = eval_diff(csv1, csv2, target=target, detail=detail)
    if diff.empty:
        return 0.0
    vals = diff[NUM_COLS].to_numpy(dtype=float)
    max_abs = float(np.nanmax(np.abs(vals)))
    if detail:
        print(f"\nMAX_ABS_DIFF {max_abs:.6g}")
    return max_abs


def eval(csv1: str, csv2: str, detail: bool = False) -> float:
    """eval_all_func から呼ばれる想定のAPI（最大絶対差を返す）"""
    return eval_diff_max_abs(csv1, csv2, target=DEFAULT_TARGET, detail=detail)


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="LR_asthma_diff (sklearn-only, with solver fallback)")
    ap.add_argument("csv1")
    ap.add_argument("csv2")
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--solver", default="auto", choices=["auto", "lbfgs", "liblinear", "saga"])
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--C", type=float, default=1.0, help="inverse of regularization strength")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("-d", "--detail", action="store_true")
    args = ap.parse_args()

    # 内部で solver/max_iter/C/tol を渡すよう eval* に引数追加していれば反映
    print(eval_diff_max_abs(args.csv1, args.csv2, target=args.target, detail=args.detail))

if __name__ == "__main__":
    main()

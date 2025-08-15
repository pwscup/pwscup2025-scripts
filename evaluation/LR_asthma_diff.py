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
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ==== ユーティリティ（LR_asthma.py と整合） ====

def is_binary_01(s: pd.Series) -> bool:
    if s.empty:
        return False
    ss = s.astype(str).str.strip()
    ss = ss[~ss.str.lower().isin({"nan", "none", ""})]
    if ss.empty:
        return False
    num = pd.to_numeric(ss, errors="coerce")
    if num.isna().any():
        return False
    vals = set(pd.unique(num.dropna().astype(int)))
    return vals.issubset({0, 1}) and len(vals) > 0

def odds_to_unit(x):
    arr = np.asarray(x, dtype=float)
    return arr / (1.0 + arr)

def vif_to_unit(v):
    v = np.asarray(v, dtype=float)
    v = np.where(~np.isfinite(v) | (v < 1.0), 1.0, v)
    return 1.0 - 1.0 / v


# ==== 1 CSV から LR_asthma 互換の表を作る ====

COLS_ORDER = ["term", "coef", "p_value", "OR_norm", "CI_low_norm", "CI_high_norm", "VIF_norm"]

def run_lr_table(csv_path: str,
                 target: str = "asthma_flag",
                 test_size: float = 0.2,
                 random_state: int = 42,
                 ensure_terms: str = "ETHNICITY_hispanic") -> tuple[pd.DataFrame, float, list[str]]:
    """
    csv_path を読み、LR_asthma.py と同様に
    - 前処理（数値 + カテゴリone-hot drop_first）
    - ロジスティック回帰（const あり、出力は const 除外）
    - OR/CI を 0-1 化、VIF を 0-1 化
    - 固定順序の term を作る（方式B: 学習に使った列 ∪ ensure_terms を辞書順）
    を行い、[COLS_ORDER] の表を返す。AUC と最終の term リストも返す。
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    if target not in df.columns:
        raise SystemExit(f"[{csv_path}] target column '{target}' not found.")
    if not is_binary_01(df[target]):
        raise SystemExit(f"[{csv_path}] target column '{target}' must be strictly binary 0/1.")

    y = pd.to_numeric(df[target], errors="coerce").astype("float64")

    X_raw = df.drop(columns=[target]).copy()
    X_num_try = X_raw.apply(pd.to_numeric, errors="coerce")
    num_cols = [c for c in X_num_try.columns if X_num_try[c].notna().sum() > 0]
    X_num = X_num_try[num_cols] if num_cols else pd.DataFrame(index=df.index)

    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    if cat_cols:
        X_cat = pd.get_dummies(
            X_raw[cat_cols].replace({"": np.nan}).fillna("(NA)").astype("category"),
            drop_first=True
        )
    else:
        X_cat = pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_cat], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise SystemExit(f"[{csv_path}] No usable features after preprocessing.")

    med = X.median(numeric_only=True)
    X = X.fillna(med).astype("float64")
    zero_var = X.nunique(dropna=False) <= 1
    if zero_var.all():
        raise SystemExit(f"[{csv_path}] All feature columns have zero variance.")
    if zero_var.any():
        X = X.loc[:, ~zero_var]

    base_terms = sorted(X.columns.tolist())
    X = X.reindex(columns=base_terms)

    mask = y.notna() & y.isin([0.0, 1.0])
    X = X.loc[mask]
    y = y.loc[mask]
    if X.shape[0] < 3 or X.shape[1] == 0:
        raise SystemExit(f"[{csv_path}] Not enough samples or features after cleaning.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y.astype(int)
    )
    X_tr_const = sm.add_constant(X_tr, has_constant="add").astype("float64")
    X_te_const = sm.add_constant(X_te, has_constant="add").astype("float64")

    model = sm.GLM(y_tr.astype("float64"), X_tr_const, family=sm.families.Binomial())
    res = model.fit()

    proba = res.predict(X_te_const)
    auc = roc_auc_score(y_te, proba)

    params = res.params.drop(labels=["const"])
    pvals  = res.pvalues.drop(labels=["const"])
    conf   = res.conf_int().drop(index="const")  # columns [0,1]

    OR      = np.exp(params.values)
    CI_low  = np.exp(conf[0].reindex(params.index).values)
    CI_high = np.exp(conf[1].reindex(params.index).values)

    coef_df = pd.DataFrame({
        "term": params.index,
        "coef": params.values,
        "p_value": pvals.values,
        "OR_norm": odds_to_unit(OR),
        "CI_low_norm": odds_to_unit(CI_low),
        "CI_high_norm": odds_to_unit(CI_high),
    })

    # VIF（const除外）
    vif_rows = []
    cols = list(X_tr_const.columns)  # ['const', ...base_terms...]
    for i, col in enumerate(cols):
        if col == "const":
            continue
        try:
            v = float(variance_inflation_factor(X_tr_const.values, i))
        except Exception:
            v = np.nan
        vif_rows.append((col, v))
    vif_df = pd.DataFrame(vif_rows, columns=["term", "VIF"])
    vif_df["VIF_norm"] = vif_to_unit(vif_df["VIF"].values)
    vif_df = vif_df.drop(columns=["VIF"])

    # 固定スキーマ：ensure-terms を union
    ensure_list = [t.strip() for t in ensure_terms.split(",") if t.strip()]
    final_terms = sorted(set(base_terms).union(ensure_list))

    out = (coef_df.merge(vif_df, on="term", how="outer")
                  .set_index("term")
                  .reindex(final_terms)
                  .reset_index())

    # 欠損は 0 に
    for c in COLS_ORDER[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # 列順
    out = out[COLS_ORDER]
    return out, float(auc), final_terms


# ==== 差分本体 ====

def main():
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

    # 片方ずつ LR 実行
    t1, auc1, terms1 = run_lr_table(
        args.csv1, target=args.target, test_size=args.test_size,
        random_state=args.random_state, ensure_terms=args.ensure_terms
    )
    t2, auc2, terms2 = run_lr_table(
        args.csv2, target=args.target, test_size=args.test_size,
        random_state=args.random_state, ensure_terms=args.ensure_terms
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
    with pd.option_context("display.max_columns", None,
                           "display.width", None,
                           "display.float_format", lambda x: f"{x:.6g}"):
        print(f"AUC (file1): {auc1:.6f}")
        print(f"AUC (file2): {auc2:.6f}")
        print(f"AUC_DIFF   : {auc2 - auc1:.6f}")

        print("\n=== Logistic regression DIFF (file2 - file1) — const excluded; values are differences ===")
        print(diff.to_string(index=False))

    # 最大絶対差（表の全数値列から算出。AUCは含めない）
    max_abs = float(np.nanmax(np.abs(diff[COLS_ORDER[1:]].to_numpy()))) if not diff.empty else 0.0
    print(f"\nMAX_ABS_DIFF {max_abs:.6g}")

    # 保存（任意）
    if args.out:
        diff.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()

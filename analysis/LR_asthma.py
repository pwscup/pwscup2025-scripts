#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logistic Regression for asthma_flag (fixed-schema output, approach B)

- すべての「出力可能な term」を辞書順で並べ、さらに --ensure-terms で
  指定された term を union して最終出力の順序（辞書順）を固定。
- const は出力しない。
- 出力: AUC と単一テーブル
    [term, coef, p_value, OR_norm, CI_low_norm, CI_high_norm, VIF_norm]
- OR_norm, CI_*_norm = odds / (1 + odds)
- VIF_norm = 1 - 1 / max(VIF, 1)    # VIF=1→0（無相関）, VIF→∞→1（強い多重共線性）
- ensure-terms に含まれるが学習行列に存在しない term は NaN で占位。
"""

import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

def main():
    ap = argparse.ArgumentParser(
        description="Logistic regression (asthma_flag) with fixed-schema output and normalized VIF"
    )
    ap.add_argument("csv", help="input CSV")
    ap.add_argument("--target", default="asthma_flag", help="binary target column (default: asthma_flag)")
    ap.add_argument("--test-size", type=float, default=0.2, help="holdout ratio (default: 0.2)")
    ap.add_argument("--random-state", type=int, default=42, help="random seed (default: 42)")
    ap.add_argument(
        "--ensure-terms",
        default="ETHNICITY_hispanic",
        help="カンマ区切りで常に表に載せたい term 名（例: 'ETHNICITY_hispanic,GENDER_M'）"
    )
    args = ap.parse_args()

    # 1) 読み込み＆ターゲット確認
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not found.")
    if not is_binary_01(df[args.target]):
        raise SystemExit(f"Target column '{args.target}' must be strictly binary 0/1.")
    y = pd.to_numeric(df[args.target], errors="coerce").astype("float64")

    # 2) 特徴量作成：数値 + 文字列はワンホット（drop_first=True）
    X_raw = df.drop(columns=[args.target]).copy()
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
        raise SystemExit("No usable features after preprocessing.")

    # 3) 欠損補完・ゼロ分散除去
    med = X.median(numeric_only=True)
    X = X.fillna(med).astype("float64")
    zero_var = X.nunique(dropna=False) <= 1
    if zero_var.all():
        raise SystemExit("All feature columns have zero variance.")
    if zero_var.any():
        X = X.loc[:, ~zero_var]

    # 4) 固定順序ベース（現に学習に使う列）
    base_terms = sorted(X.columns.tolist())
    X = X.reindex(columns=base_terms)

    # y と整合
    mask = y.notna() & y.isin([0.0, 1.0])
    X = X.loc[mask]
    y = y.loc[mask]
    if X.shape[0] < 3 or X.shape[1] == 0:
        raise SystemExit("Not enough samples or features after cleaning.")

    # 5) 学習・評価
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y.astype(int)
    )
    X_tr_const = sm.add_constant(X_tr, has_constant="add").astype("float64")
    X_te_const = sm.add_constant(X_te, has_constant="add").astype("float64")
    model = sm.GLM(y_tr.astype("float64"), X_tr_const, family=sm.families.Binomial())
    res = model.fit()

    proba = res.predict(X_te_const)
    auc = roc_auc_score(y_te, proba)

    # 6) 係数・p・CI → OR系を0-1化（const除外）
    params = res.params.drop(labels=["const"])
    pvals  = res.pvalues.drop(labels=["const"])
    conf   = res.conf_int().drop(index="const")  # columns: [0,1] (low, high)

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

    # 7) VIF（const除外）→ 正規化のみ出力
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

    # 8) ★固定スキーマ（方式B）：ensure-terms を union し、辞書順で最終順序を固定
    ensure_list = []
    if args.ensure_terms:
        ensure_list = [t.strip() for t in args.ensure_terms.split(",") if t.strip()]
    final_terms = sorted(set(base_terms).union(ensure_list))

    # 係数表とVIF_normを term で結合し、final_terms で reindex（存在しないtermはNaNで占位）
    out = (coef_df.merge(vif_df, on="term", how="outer")
                  .set_index("term")
                  .reindex(final_terms)
                  .reset_index())

    # 9) 出力
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print(f"\nAUC (holdout): {auc:.6f}")
    print("\n=== Logistic regression summary (const excluded; fixed-term schema with ensure-terms) ===")
    with pd.option_context('display.float_format', lambda x: f"{x:.6g}"):
        # 列順を固定
        cols_order = ["term", "coef", "p_value", "OR_norm", "CI_low_norm", "CI_high_norm", "VIF_norm"]
        print(out[cols_order].to_string(index=False))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xgbt_train.py  (contest-spec compliant; no post-edit)

前提:
- 入力CSVに ID 列は無い
- 欠損データは無い（欠損補完は行わない）

処理:
- CSV読み込み → 前処理（数値化 + one-hot(drop_first=True) + ゼロ分散列除去 + 列名昇順固定）
- XGBoost（二値分類, early_stopping はコンストラクタ指定）で学習
- Booster に attributes を set_attr で埋め込んでから save_model():
    * feature_names（学習時列名・昇順）の JSON文字列
    * target（目的変数名）
    * xgboost_version（実行バージョン）
- JSONは一切後編集しない（version など内部キーを失わないため）
"""

import argparse
import json
import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

def is_binary_01(s: pd.Series) -> bool:
    if s.empty:
        return False
    ss = s.astype(str).str.strip()
    if (ss == "").any():
        return False
    num = pd.to_numeric(ss, errors="coerce")
    if num.isna().any():
        return False
    vals = set(pd.unique(num.astype(int)))
    return vals.issubset({0, 1}) and len(vals) > 0

def build_X(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        raise SystemExit(f"Target column '{target}' not found in CSV.")
    X_raw = df.drop(columns=[target]).copy()

    X_num_try = X_raw.apply(pd.to_numeric, errors="coerce")
    num_cols = [c for c in X_num_try.columns if X_num_try[c].notna().sum() > 0]
    X_num = X_num_try[num_cols] if num_cols else pd.DataFrame(index=df.index)

    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    X_cat = pd.get_dummies(X_raw[cat_cols].astype("category"), drop_first=True) if cat_cols else pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_cat], axis=1)

    zero_var = X.nunique(dropna=False) <= 1
    if zero_var.all():
        raise SystemExit("All feature columns have zero variance.")
    if zero_var.any():
        X = X.loc[:, ~zero_var]

    X = X.reindex(columns=sorted(X.columns)).astype("float32")
    return X

def main():
    ap = argparse.ArgumentParser(description="Train XGBoost binary classifier and export JSON (no post-edit).")
    ap.add_argument("train_csv", help="training CSV with header")
    ap.add_argument("--model-json", required=True, help="output model JSON path")
    ap.add_argument("--target", default="stroke_flag", help="binary target column (default: stroke_flag)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    ap.add_argument("--test-size", type=float, default=0.1, help="validation ratio (default: 0.1)")
    ap.add_argument("--n-estimators", type=int, default=600, help="n_estimators (default: 600)")
    ap.add_argument("--max-depth", type=int, default=6, help="max_depth (default: 6)")
    ap.add_argument("--learning-rate", type=float, default=0.05, help="learning_rate (default: 0.05)")
    ap.add_argument("--subsample", type=float, default=0.9, help="subsample (default: 0.9)")
    ap.add_argument("--colsample-bytree", type=float, default=0.9, help="colsample_bytree (default: 0.9)")
    ap.add_argument("--early-stopping-rounds", type=int, default=50, help="early stopping rounds (default: 50)")
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv, dtype=str, keep_default_na=False)

    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not found.")
    if not is_binary_01(df[args.target]):
        raise SystemExit(f"Target column '{args.target}' must be strictly binary 0/1.")
    y = pd.to_numeric(df[args.target], errors="coerce").astype(int).values

    X = build_X(df, target=args.target)
    feature_names = list(X.columns)  # 昇順・重複なし

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        early_stopping_rounds=args.early_stopping_rounds,  # ← 警告回避
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = (model.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy (threshold=0.5): {acc:.6f}")

    booster = model.get_booster()
    # JSONを後編集せず、属性を埋めて「そのまま」保存
    booster.set_attr(feature_names=json.dumps(feature_names, ensure_ascii=False))
    booster.set_attr(target=args.target)
    booster.set_attr(xgboost_version=xgb.__version__)
    booster.save_model(args.model_json)

    print(f"Saved model JSON to: {args.model_json}")
    print(f"#features: {len(feature_names)}")

if __name__ == "__main__":
    main()

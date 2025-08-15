#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys, os
from itertools import combinations

# ===== 固定スキーマ（順序も固定） =====
EXPECTED_NUMERIC = [
    "AGE",
    "encounter_count",
    "num_procedures",
    "num_medications",
    "num_immunizations",
    "num_allergies",
    "num_devices",
    "mean_systolic_bp",
    "mean_diastolic_bp",
    "mean_bmi",
    "mean_weight",
]

# 0/1 フラグもカテゴリ扱い
EXPECTED_FLAGS = ["asthma_flag", "stroke_flag", "obesity_flag", "depression_flag"]

EXPECTED_CATEGORICAL_LEVELS = {
    "GENDER": ["M", "F", "O", "U"],
    "RACE": ["asian", "black", "white", "native", "hawaiian", "other", "unknown"],
    "ETHNICITY": ["hispanic", "nonhispanic", "unknown"],
    "AGE_GROUP": ["0–17", "18–44", "45–64", "65–74", "75+"],
    "asthma_flag": [0, 1],
    "stroke_flag": [0, 1],
    "obesity_flag": [0, 1],
    "depression_flag": [0, 1],
}

CAT_COL_ORDER = ["GENDER", "RACE", "ETHNICITY", "AGE_GROUP"] + EXPECTED_FLAGS
# count_ratio を出さない
STATS_ORDER = ["mean", "std", "25%", "50%", "75%"]

# ====== I/O ======
if len(sys.argv) < 2:
    print("Usage: python3 stats.py <input.csv>")
    sys.exit(1)
csv_file = sys.argv[1]
if not os.path.isfile(csv_file):
    print(f"Error: File {csv_file} not found")
    sys.exit(1)

df_raw = pd.read_csv(csv_file, dtype=str)

# ====== 補助関数 ======
def to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def minmax_normalize(v: pd.Series) -> pd.Series:
    x = to_float(v)
    mn, mx = x.min(skipna=True), x.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return x * 0.0
    return (x - mn) / (mx - mn)

def build_age_group(age_series: pd.Series) -> pd.Series:
    lab = EXPECTED_CATEGORICAL_LEVELS["AGE_GROUP"]
    age = to_float(age_series)
    edges = [0, 18, 45, 65, 75, np.inf]
    out = pd.cut(age, bins=edges, right=False, labels=lab, include_lowest=True)
    return out.astype(object)

def norm_gender(x: pd.Series) -> pd.Series:
    s = x.astype(str).str.strip().str.lower()
    out = np.where(s.str.startswith("m"), "M",
          np.where(s.str.startswith("f"), "F",
          np.where(s.eq("o") | s.str.startswith("oth"), "O", "U")))
    return pd.Series(out, index=x.index, dtype=object)

def norm_race(x: pd.Series) -> pd.Series:
    s = x.astype(str).str.strip().str.lower()
    keys = ["asian","black","white","native hawaiian","hawaiian","native","other","unknown","null","none"]
    def map_one(t):
        for k in keys:
            if k in t:
                if k == "native hawaiian" or k == "hawaiian":
                    return "hawaiian"
                if k in ("unknown","null","none"):
                    return "unknown"
                if k == "native":
                    return "native"
                return k
        return "unknown"
    return s.map(map_one).astype(object)

def norm_ethnicity(x: pd.Series) -> pd.Series:
    s = x.astype(str).str.strip().str.lower()
    out = np.where(s.str.contains("non") | s.str.contains("not"), "nonhispanic",
          np.where(s.str.contains("hisp"), "hispanic", "unknown"))
    return pd.Series(out, index=x.index, dtype=object)

def norm_flag(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    return v.where(v.isin([0,1]), np.nan)

# ====== データ整形 ======
df = df_raw.copy()

# 数値列（固定順）
for col in EXPECTED_NUMERIC:
    if col not in df.columns:
        df[col] = np.nan
    df[col] = to_float(df[col])

# フラグ列（0/1）
for col in EXPECTED_FLAGS:
    if col not in df.columns:
        df[col] = np.nan
    df[col] = norm_flag(df[col])

# カテゴリ列の正規化
if "GENDER" not in df.columns:
    df["GENDER"] = np.nan
df["GENDER"] = norm_gender(df["GENDER"])

if "RACE" not in df.columns:
    df["RACE"] = np.nan
df["RACE"] = norm_race(df["RACE"])

if "ETHNICITY" not in df.columns:
    df["ETHNICITY"] = np.nan
df["ETHNICITY"] = norm_ethnicity(df["ETHNICITY"])

# AGE_GROUP の生成（AGE が無い/欠損でもOK）
df["AGE_GROUP"] = build_age_group(df["AGE"]) if "AGE" in df.columns else pd.Series(np.nan, index=df.index)

# ====== 1) 数値列の統計（min-max 正規化, 固定行列） ======
df_norm = df[EXPECTED_NUMERIC].apply(minmax_normalize, axis=0)

desc = df_norm.describe(percentiles=[0.25, 0.5, 0.75])
desc = desc.loc[["mean","std","25%","50%","75%"]]
desc = desc.reindex(columns=EXPECTED_NUMERIC)

print("\n=== 数値列の統計量（min-max正規化後, 0〜1・min/max除外・固定スキーマ） ===")
with pd.option_context("display.max_columns", None, "display.width", None, "display.float_format", lambda x: f"{x:.6g}"):
    print(desc)

# ====== 2) 数値×数値 相関（固定スキーマ） ======
corr = df[EXPECTED_NUMERIC].astype(float).corr()
corr = corr.reindex(index=EXPECTED_NUMERIC, columns=EXPECTED_NUMERIC)

print("\n--- 相関行列（Pearson, 固定スキーマ） ---")
with pd.option_context("display.max_columns", None, "display.width", None, "display.float_format", lambda x: f"{x:.6g}"):
    print(corr)

# ====== 3) カテゴリ列の集計（比率のみ、ラベルは ratio） ======
def categorical_summary_fixed(colname: str) -> pd.DataFrame:
    levels = EXPECTED_CATEGORICAL_LEVELS[colname]
    s = df[colname]
    vc = s.value_counts(dropna=False)
    count = pd.Series({lev: int(vc.get(lev, 0)) for lev in levels}, index=levels)
    ratio = (count / len(df)).astype(float)
    return pd.DataFrame({"ratio": ratio})

print("\n=== カテゴリ列の集計（固定レベル順・比率のみ） ===")
for cat in CAT_COL_ORDER:
    print(f"\n--- {cat} ---")
    tbl = categorical_summary_fixed(cat)
    with pd.option_context("display.max_columns", None, "display.width", None, "display.float_format", lambda x: f"{x:.6g}"):
        print(tbl)

# ====== 4) カテゴリ×数値（min-max 正規化、固定レベル×固定統計：count系は出さない） ======
def group_table_fixed(cat_col: str) -> pd.DataFrame:
    levels = EXPECTED_CATEGORICAL_LEVELS[cat_col]
    s = df[cat_col]  # すでに正規化済み

    pieces = {}
    for num in EXPECTED_NUMERIC:
        col = df_norm[num]
        d = (
            col.groupby(s, dropna=True)
               .describe(percentiles=[0.25, 0.5, 0.75])[["mean","std","25%","50%","75%"]]
               .reindex(index=levels)   # 行順を固定
        )
        d = d.reindex(columns=STATS_ORDER)
        pieces[num] = d

    out = pd.concat(pieces, axis=1)  # MultiIndex 列（数値×統計）
    out = out.reindex(index=levels)
    return out

print("\n=== カテゴリ×数値の要約統計（min-max正規化後, 固定レベル×固定統計） ===")
for cat in CAT_COL_ORDER:
    print(f"\n--- Group by: {cat} ---")
    tbl = group_table_fixed(cat)
    with pd.option_context("display.max_columns", None, "display.width", None, "display.float_format", lambda x: f"{x:.6g}"):
        print(tbl)

# ====== 5) カテゴリ×カテゴリのクロス集計（★比率 0〜1 のみ出力・固定レベル） ======
def crosstab_ratio_fixed(a: str, b: str) -> pd.DataFrame:
    levels_a = EXPECTED_CATEGORICAL_LEVELS[a]
    levels_b = EXPECTED_CATEGORICAL_LEVELS[b]
    s1 = df[a]
    s2 = df[b]
    ct = pd.crosstab(s1, s2, dropna=True)
    ct = ct.reindex(index=levels_a, columns=levels_b, fill_value=0)
    ratio = (ct / len(df)).astype(float)
    return ratio

print("\n=== カテゴリ×カテゴリのクロス集計（全体比 0〜1・固定レベル） ===")
for a, b in combinations(CAT_COL_ORDER, 2):
    print(f"\n--- Crosstab (ratio): {a} × {b} ---")
    ratio = crosstab_ratio_fixed(a, b)
    with pd.option_context("display.max_columns", None, "display.width", None, "display.float_format", lambda x: f"{x:.6g}"):
        print(ratio)

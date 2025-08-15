#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
  python3 columns_range_json.py input.csv [-o output.json]

- 出力(-o/--output)未指定時は input.csv → input.json に保存
- 各列を 数値 / 日付 / カテゴリ に判定して JSON 化
  * 数値: min, max, max_decimal_places
  * 日付(yyyy/m/d, yyyy/mm/dd, yyyy-m-d, yyyy-mm-dd を許容):
      min, max（いずれも yyyy-mm-dd で出力）, format="yyyy-mm-dd"
  * カテゴリ: 全ユニーク値（NaN除外, 出現順保持）
- 仕上がりは {"columns": { ... }} でラップ
"""

import argparse
import json
import os
import re
import numpy as np
import pandas as pd

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Infer column schema from CSV and save as JSON (wrapped by 'columns').")
    ap.add_argument("input_csv", help="入力CSVファイル")
    ap.add_argument("-o", "--output", dest="output_json", default=None,
                    help="出力JSONファイル（省略時は input.csv → input.json）")
    return ap.parse_args()

# -------------- helpers --------------
# yyyy/m/d, yyyy/mm/dd, yyyy-m-d, yyyy-mm-dd を許容
DATE_REGEX = re.compile(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}\s*$")
NUM_TOKEN  = re.compile(r"^-?\d+(?:\.\d+)?$")

def is_date_like(series: pd.Series, min_match_ratio: float = 0.7) -> bool:
    s = series.astype(str)
    s = s[~s.str.lower().isin({"nan", "none", ""})]
    if s.empty:
        return False
    return (s.str.match(DATE_REGEX).mean() >= min_match_ratio) and (len(s) >= 5)

def parse_dates(series: pd.Series) -> pd.Series:
    # pandas に任せてゆるく解釈（スラッシュ/ハイフン両対応）
    return pd.to_datetime(series, errors="coerce", format=None)

def is_numeric_like(series: pd.Series, min_match_ratio: float = 0.7) -> bool:
    s = series.astype(str)
    s = s[~s.str.lower().isin({"nan", "none", ""})].str.replace(",", "", regex=False)
    if s.empty:
        return False
    return (s.str.match(NUM_TOKEN).mean() >= min_match_ratio) and (len(s) >= 5)

def max_decimal_places(series: pd.Series) -> int:
    s = series.astype(str).str.replace(",", "", regex=False)
    s = s[~s.str.lower().isin({"nan", "none", ""})]
    decs = []
    for v in s:
        if NUM_TOKEN.match(v):
            if "." in v:
                decs.append(len(v.split(".", 1)[1]))
            else:
                decs.append(0)
    return int(max(decs) if decs else 0)

def to_float_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def uniq_categories(series: pd.Series):
    vals = series.dropna().tolist()
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

# --------------- main ---------------
def main():
    args = parse_args()
    in_csv = args.input_csv
    if not os.path.isfile(in_csv):
        raise SystemExit(f"File not found: {input_csv}")

    if args.output_json is None:
        base = os.path.splitext(os.path.basename(in_csv))[0]
        args.output_json = os.path.join(os.path.dirname(in_csv), base + ".json")

    # 読み込み（文字列優先、後で判定）
    df = pd.read_csv(in_csv, dtype=str, keep_default_na=True)

    columns_schema = {}
    for col in df.columns:
        s = df[col]

        # 1) 日付？
        if is_date_like(s):
            dt = parse_dates(s)
            if dt.notna().any():
                dmin = dt.min()
                dmax = dt.max()
                columns_schema[col] = {
                    "type": "date",
                    "format": "yyyy-mm-dd",
                    "min": dmin.strftime("%Y-%m-%d"),
                    "max": dmax.strftime("%Y-%m-%d"),
                }
                continue

        # 2) 数値？
        if is_numeric_like(s):
            num = to_float_series(s)
            vmin = np.nanmin(num) if np.isfinite(np.nanmin(num)) else None
            vmax = np.nanmax(num) if np.isfinite(np.nanmax(num)) else None
            columns_schema[col] = {
                "type": "numeric",
                "min": float(vmin) if vmin is not None else None,
                "max": float(vmax) if vmax is not None else None,
                "max_decimal_places": max_decimal_places(s),
            }
            continue

        # 3) カテゴリ（デフォルト）
        columns_schema[col] = {
            "type": "categorical",
            "values": uniq_categories(s),
        }

    out_obj = {"columns": columns_schema}

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"Saved schema to: {args.output_json}")

if __name__ == "__main__":
    main()

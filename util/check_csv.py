#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import re
from decimal import Decimal, InvalidOperation
import pandas as pd
from pathlib import Path

YMD_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def _decimal_places(d: Decimal) -> int:
    """Decimal の小数点以下桁数を返す（指数が負ならその絶対値、非負なら 0）。"""
    exp = d.as_tuple().exponent
    return -exp if exp < 0 else 0

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 check_csv.py <input.csv> <input.json>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2])

    if not csv_path.exists():
        print(f"エラー: CSVファイルがありません: {csv_path}")
        sys.exit(1)
    if not json_path.exists():
        print(f"エラー: JSONファイルがありません: {json_path}")
        sys.exit(1)

    # 文字列で読み込み（欠損は空文字）
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"CSV読み込みエラー: {e}")
        sys.exit(1)

    # JSON 読み込み
    try:
        with json_path.open("r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as e:
        print(f"JSON読み込みエラー: {e}")
        sys.exit(1)

    if "columns" not in spec:
        print("エラー: JSONの形式が不正です（'columns'キーがありません）")
        sys.exit(1)

    all_ok = True
    errors = []

    for col in df.columns:
        if col not in spec["columns"]:
            print(f"警告: JSON に '{col}' 列の定義がありません。スキップします。")
            continue

        col_spec = spec["columns"][col]
        raw_vals = df[col].map(lambda x: x.strip())
        ctype = col_spec.get("type", "")

        if ctype == "number":
            # min/max は必須
            try:
                min_val = Decimal(str(col_spec["min"]))
                max_val = Decimal(str(col_spec["max"]))
            except Exception:
                print(f"エラー: {col} の数値範囲(min/max)がJSONで不正です。")
                sys.exit(1)

            # max_decimal_places は任意
            max_places = col_spec.get("max_decimal_places", None)
            if max_places is not None:
                try:
                    max_places = int(max_places)
                    if max_places < 0:
                        raise ValueError
                except Exception:
                    print(f"エラー: {col} の max_decimal_places が不正です（非負整数で指定）。")
                    sys.exit(1)

            for idx, val in raw_vals.items():
                if val == "":
                    continue
                try:
                    d = Decimal(val)
                except InvalidOperation:
                    all_ok = False
                    errors.append((idx+1, col, val, "数値変換不可"))
                    continue

                # 桁数チェック（指定がある場合）
                if max_places is not None:
                    places = _decimal_places(d)
                    if places > max_places:
                        all_ok = False
                        errors.append(
                            (idx+1, col, val, f"小数点以下{places}桁（許容 {max_places} 桁）")
                        )

                # 範囲チェック
                if d < min_val or d > max_val:
                    all_ok = False
                    errors.append((idx+1, col, val, f"{min_val}〜{max_val}の範囲外"))

        elif ctype == "category":
            allowed = set(col_spec.get("values", []))
            for idx, val in raw_vals.items():
                if val == "":
                    continue
                if val not in allowed:
                    all_ok = False
                    errors.append((idx+1, col, val, f"許可されていない値（{allowed}）"))

        elif ctype == "date":
            # yyyy-mm-dd 固定
            try:
                min_dt = pd.to_datetime(col_spec["min"], format="%Y-%m-%d", errors="raise")
                max_dt = pd.to_datetime(col_spec["max"], format="%Y-%m-%d", errors="raise")
            except Exception as e:
                print(f"エラー: JSON の日付範囲が不正です（{col}: {e}）")
                sys.exit(1)

            for idx, val in raw_vals.items():
                if val == "":
                    continue
                if not YMD_RE.match(val):
                    all_ok = False
                    errors.append((idx+1, col, val, "日付形式違反（yyyy-mm-dd）"))
                    continue
                dt = pd.to_datetime(val, format="%Y-%m-%d", errors="coerce")
                if pd.isna(dt):
                    all_ok = False
                    errors.append((idx+1, col, val, "日付変換不可（yyyy-mm-dd）"))
                    continue
                if dt < min_dt or dt > max_dt:
                    all_ok = False
                    errors.append(
                        (idx+1, col, val, f"{min_dt.strftime('%Y-%m-%d')}〜{max_dt.strftime('%Y-%m-%d')}の範囲外")
                    )

        else:
            print(f"警告: 列 '{col}' のタイプ '{ctype}' は未対応。スキップします。")

    if all_ok:
        print("全ての値が JSON 仕様内です。")
    else:
        print(f"範囲外または不正値が {len(errors)} 件見つかりました。")
        for rownum, col, val, reason in errors:
            print(f"  行{rownum} 列'{col}' 値='{val}' → {reason}")

if __name__ == "__main__":
    main()

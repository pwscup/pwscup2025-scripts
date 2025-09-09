#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_and_fix_csv.py
CSVをJSONスキーマで検査し、カテゴリ不正はエラー終了、
数値の範囲外/不正は「最も近い範囲内の値」へ補正して出力します。

Usage:
  python3 check_and_fix_csv.py input.csv spec.json output.csv [--report fix_report.csv]

JSONスキーマ例:
{
  "columns": {
    "age": {"type": "number", "min": 0, "max": 120, "max_decimal_places": 0},
    "sex": {"type": "category", "values": ["M","F"]},
    "bmi": {"type": "number", "min": 10, "max": 60, "max_decimal_places": 1}
  }
}
"""
import sys
import json
import argparse
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
import pandas as pd

def to_decimal_maybe(val: str):
    """数値らしき文字列から Decimal を試みる（カンマは除去）。失敗時は None。"""
    s = val.strip().replace(",", "")
    if s == "":
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None

def quantize_to_places(d: Decimal, places: int | None) -> Decimal:
    if places is None:
        return d
    if places < 0:
        q = Decimal(10) ** (-places)  # 負の桁指定（通常は使わない想定）
    else:
        q = Decimal(1).scaleb(-places)  # 10^(-places)
    return d.quantize(q, rounding=ROUND_HALF_UP)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_in")
    ap.add_argument("json_spec")
    ap.add_argument("csv_out")
    ap.add_argument("--report", default=None, help="補正内容の明細CSV（任意）")
    args = ap.parse_args()

    csv_path = Path(args.csv_in)
    json_path = Path(args.json_spec)
    out_path = Path(args.csv_out)
    report_path = Path(args.report) if args.report else None

    if not csv_path.exists():
        print(f"エラー: CSVファイルがありません: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not json_path.exists():
        print(f"エラー: JSONファイルがありません: {json_path}", file=sys.stderr)
        sys.exit(1)

    # CSV 読み込み（欠損は空文字として保持）
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"CSV読み込みエラー: {e}", file=sys.stderr)
        sys.exit(1)

    # JSON 読み込み
    try:
        with json_path.open("r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as e:
        print(f"JSON読み込みエラー: {e}", file=sys.stderr)
        sys.exit(1)

    if "columns" not in spec or not isinstance(spec["columns"], dict):
        print("エラー: JSONの形式が不正です（'columns'キーが無い/不正）", file=sys.stderr)
        sys.exit(1)

    # 収集用
    any_cat_error = False
    cat_errors: list[tuple[int, str, str, str]] = []   # (row, col, value, reason)
    num_fixes: list[tuple[int, str, str, str, str]] = []  # (row, col, original, fixed, reason)

    # CSVにある列をスキーマでチェック
    for col in df.columns:
        if col not in spec["columns"]:
            print(f"警告: JSON に '{col}' 列の定義がありません。スキップします。", file=sys.stderr)
            continue

        cdef = spec["columns"][col]
        ctype = str(cdef.get("type", "")).lower()

        # 値の前後空白除去
        series = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

        if ctype == "category":
            allowed = set(cdef.get("values", []))
            for i, raw in series.items():
                if raw == "":
                    continue  # 空は許容（必要に応じて厳格化可）
                if raw not in allowed:
                    any_cat_error = True
                    cat_errors.append((i+1, col, raw, f"未許可の値。許容: {sorted(allowed)}"))

        elif ctype == "number":
            # 必須: min/max、任意: max_decimal_places
            try:
                min_val = Decimal(str(cdef["min"]))
                max_val = Decimal(str(cdef["max"]))
            except Exception:
                print(f"エラー: {col} の数値範囲(min/max)がJSONで不正です。", file=sys.stderr)
                sys.exit(1)
            places = cdef.get("max_decimal_places", None)
            if places is not None:
                try:
                    places = int(places)
                except Exception:
                    print(f"エラー: {col} の max_decimal_places が不正です。", file=sys.stderr)
                    sys.exit(1)

            fixed_vals: list[str] = []
            for i, raw in series.items():
                if raw == "":
                    fixed_vals.append(raw)  # 空はそのまま（必要なら補完ルールへ）
                    continue

                d = to_decimal_maybe(raw)
                reason = None
                if d is None:
                    d = min_val
                    reason = "数値変換不可→minへ補正"

                # 範囲クランプ
                clamped = d
                if clamped < min_val:
                    clamped = min_val
                    reason = "最小値へクランプ" if reason is None else f"{reason};最小値へクランプ"
                elif clamped > max_val:
                    clamped = max_val
                    reason = "最大値へクランプ" if reason is None else f"{reason};最大値へクランプ"

                # 小数桁丸め
                if places is not None:
                    rounded = quantize_to_places(clamped, places)
                    if rounded != clamped:
                        reason = "小数桁丸め" if reason is None else f"{reason};小数桁丸め"
                    clamped = rounded

                fixed_str = format(clamped, 'f')
                fixed_vals.append(fixed_str)

                if reason is not None and fixed_str != raw:
                    num_fixes.append((i+1, col, raw, fixed_str, reason))

            df[col] = fixed_vals

        else:
            print(f"警告: 列 '{col}' のタイプ '{ctype}' は未対応。スキップします。", file=sys.stderr)

    # カテゴリ不正があれば終了（出力は行わない）
    if any_cat_error:
        print(f"エラー: カテゴリの不正が {len(cat_errors)} 件見つかりました。", file=sys.stderr)
        for rownum, col, val, reason in cat_errors[:50]:
            print(f"  行{rownum} 列'{col}' 値='{val}' → {reason}", file=sys.stderr)
        if len(cat_errors) > 50:
            print(f"  ... 省略（合計 {len(cat_errors)} 件）", file=sys.stderr)
        sys.exit(2)

    # 数値補正のレポート
    if num_fixes:
        print(f"数値の補正を {len(num_fixes)} 件行いました（min/max クランプ、丸め等）。", file=sys.stderr)
        if report_path:
            rep_df = pd.DataFrame(num_fixes, columns=["row", "column", "original", "fixed", "reason"])
            rep_df.to_csv(report_path, index=False)
            print(f"補正の明細を {report_path} に保存しました。", file=sys.stderr)
        else:
            for row, col, org, fix, rsn in num_fixes[:20]:
                print(f"  行{row} 列'{col}' '{org}' -> '{fix}' （{rsn}）", file=sys.stderr)
            if len(num_fixes) > 20:
                print(f"  ... 省略（合計 {len(num_fixes)} 件）", file=sys.stderr)
    else:
        print("数値の補正はありません。", file=sys.stderr)

    # 補正後CSVを書き出し
    try:
        df.to_csv(out_path, index=False)
        print(f"補正後のCSVを書き出しました: {out_path}", file=sys.stderr)
    except Exception as e:
        print(f"CSV書き出しエラー: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

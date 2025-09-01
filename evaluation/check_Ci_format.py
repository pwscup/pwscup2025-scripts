from typing import Tuple
import argparse
import json
from decimal import Decimal, InvalidOperation
import re

import pandas as pd


# COLUMNS = []
NUM_LINE = 10000
YMD_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def get_col_specs():
    with open('data/pre_columns_range.json', encoding='utf-8') as f:
        j = json.load(f)
    
    return j["columns"]

COL_SPECS = get_col_specs()

def get_correct_columns():
    """
    正しい列名を取得
    """
    columns = list(COL_SPECS.keys())

    return columns

class CiFormatError(Exception):
    """
    Ci.csvのフォーマットに不正があった場合の例外
    """
    pass

class ColumnsError(CiFormatError):
    """
    Ciの列が不正
    """
    pass

class LineNumError(CiFormatError):
    """
    Ciの行数が不正
    """
    pass

class ColSpecError(CiFormatError):
    """
    ある列の仕様(spec)が不正
    """
    pass

def check_col_names(Ci_df:pd.DataFrame)->Tuple[bool, list]:
    correct_columns = get_correct_columns()
    Ci_columns = list(Ci_df.columns)

    errors = []
    ok = True

    if set(Ci_columns) != set(correct_columns):
        ok = False
        errors.append(ColumnsError(f"期待される列{correct_columns}, 実際の列{Ci_columns}"))

    return ok, errors

def check_line_num(Ci_df:pd.DataFrame)->Tuple[bool, list]:
    num_line = Ci_df.shape[0]

    errors = []
    ok = True

    if num_line != NUM_LINE:
        ok = False
        errors.append(ColumnsError(f"期待される行数は{NUM_LINE}, 実際の行数は{num_line}"))

    return ok, errors

def check_col_specs(Ci_df:pd.DataFrame)->Tuple[bool, list]:
    Ci_columns = list(Ci_df.columns)

    errors = []

    for col, col_spec in COL_SPECS.items():
        # col = col_spec.key()
        if col not in Ci_columns:
            continue

        # col_spec = col_specs[col]
        raw_vals = Ci_df[col].map(lambda x: x.strip())
        col_type = col_spec.get("type", "")

        if col_type == "numeric" or col_type == "number":
            min_val = Decimal(str(col_spec["min"]))
            max_val = Decimal(str(col_spec["max"]))
            for idx, val in raw_vals.items():
                if val == "":
                    continue
                try:
                    d = Decimal(val)
                except InvalidOperation:
                    all_ok = False
                    errors.append(ColSpecError(idx+1, col, val, "数値変換不可"))
                    continue
                if d < min_val or d > max_val:
                    all_ok = False
                    errors.append(ColSpecError(idx+1, col, val, f"{min_val}〜{max_val}の範囲外"))

        elif col_type == "categorical" or col_type == "category":
            allowed = set(col_spec.get("values", []))
            for idx, val in raw_vals.items():
                if val == "":
                    continue
                if val not in allowed:
                    all_ok = False
                    errors.append(ColSpecError(idx+1, col, val, f"許可されていない値（{allowed}）"))

        elif col_type == "date":
            # yyyy-mm-dd 固定
            min_dt = pd.to_datetime(col_spec["min"], format="%Y-%m-%d", errors="raise")
            max_dt = pd.to_datetime(col_spec["max"], format="%Y-%m-%d", errors="raise")

            for idx, val in raw_vals.items():
                if val == "":
                    continue
                if not YMD_RE.match(val):
                    all_ok = False
                    errors.append(ColSpecError(idx+1, col, val, "日付形式違反（yyyy-mm-dd）"))
                    continue
                dt = pd.to_datetime(val, format="%Y-%m-%d", errors="coerce")
                if pd.isna(dt):
                    all_ok = False
                    errors.append(ColSpecError(idx+1, col, val, "日付変換不可（yyyy-mm-dd）"))
                    continue
                if dt < min_dt or dt > max_dt:
                    all_ok = False
                    errors.append(ColSpecError(idx+1, col, val, f"{min_dt.strftime('%Y-%m-%d')}〜{max_dt.strftime('%Y-%m-%d')}の範囲外"))

        else:
            # デバッグ用。pre_columns_range.jsonを編集した場合に、表示される可能性あり
            print(f"警告: 列 '{col}' のタイプ '{col_type}' は未対応。スキップします。")

    if errors:
        ok = False
    else:
        ok = True

    return ok, errors

def check_df_Ci_format(Ci_df:pd.DataFrame)->Tuple[bool, list]:
    errors = []
    
    _, col_name_errors = check_col_names(Ci_df)
    errors.extend(col_name_errors)

    _, line_num_errors = check_line_num(Ci_df)
    errors.extend(line_num_errors)

    _, col_specs_errors = check_col_specs(Ci_df)
    errors.extend(col_specs_errors)
    
    if not errors:
        ok = True
    else:
        ok = False

    return ok, errors

def check_csv_Ci_format(path_to_Ci_csv:str)->Tuple[bool, list]:
    try:
        # csvファイルをpandasのDataFrameとして開くことを試みる
        df_Ci = pd.read_csv(path_to_Ci_csv, dtype=str, keep_default_na=False)
    except Exception as e:
        # この時点で例外が発生した場合、次のテェックが不可能なので終了
        return (False, [e])
    
    return check_df_Ci_format(df_Ci)


if __name__=="__main__":
    """
    使用例
    """
    argparser = argparse.ArgumentParser(description="Ciのフォーマットが必要条件を満たしているか確認する")
    argparser.add_argument("path_to_Ci_csv", help="Ci.csvへのパス")
    args = argparser.parse_args()

    ok, errors = check_csv_Ci_format(args.path_to_Ci_csv)
    if ok:
        print(f"{args.path_to_Ci_csv}は必要条件を満たしています")
    else:
        print(f"{args.path_to_Ci_csv}は必要条件を満たしていません")
        for e in errors:
            print(e)
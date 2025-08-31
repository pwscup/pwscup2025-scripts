from typing import Tuple
import argparse
import json

import pandas as pd


# COLUMNS = []
NUM_LINE = 10000

def get_correct_columns():
    """
    正しい列名を取得
    """
    with open('data/pre_columns_range.json', encoding='utf-8') as f:
        col_ranges = json.load(f)

    columns = list(col_ranges['columns'].keys())

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

def check_df_Ci_format(Ci_df:pd.DataFrame)->Tuple[bool, list]:
    correct_columns = get_correct_columns()
    errors = []

    Ci_columns = list(Ci_df.columns)
    if set(Ci_columns) != set(correct_columns):
        errors.append(ColumnsError(f"期待される列{correct_columns}, 実際の列{Ci_columns}"))

    num_line = Ci_df.shape[0]
    if num_line != NUM_LINE:
        errors.append(ColumnsError(f"期待される行数は{NUM_LINE}, 実際の行数は{num_line}"))
    
    if not errors:
        ok = True
    else:
        ok = False

    return (ok, errors)

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
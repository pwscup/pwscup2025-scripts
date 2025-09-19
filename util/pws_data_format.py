import re
import os
import json
from decimal import Decimal, InvalidOperation

import pandas as pd

from check_and_fix_csv import to_decimal_maybe, quantize_to_places

# COLUMNS = []
YMD_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def get_col_specs():
    # 現在のモジュールファイルのディレクトリを取得
    module_dir = os.path.abspath(os.path.dirname(__file__))

    json_path = os.path.join(module_dir, "..", "data", "columns_range.json")

    with open(json_path, encoding='utf-8') as f:
        j = json.load(f)
    
    return j["columns"]



def get_correct_columns():
    """
    正しい列名を取得
    """
    col_specs = get_col_specs()
    columns = list(col_specs.keys())

    return columns

class FormatError(Exception):
    """
    フォーマットに不正があった場合の例外
    """
    pass

class RowNumError(FormatError):
    """
    行数が不正
    """
    pass

class ColumnsError(FormatError):
    """
    列が不正
    """
    pass

class ColumnsOrderError(ColumnsError):
    """
    列の順番が不正. 修正が容易
    """
    pass

class ColSpecError(FormatError):
    """
    ある列の仕様(spec)が不正
    """
    pass

class CatSpecError(ColSpecError):
    """
    カテゴリー列の仕様が不正。修復不能
    """
    pass

class NumSpecError(ColSpecError):
    """
    数値列の仕様が不正。修復可能
    """
    pass

class BiDataFrame(pd.DataFrame):
    """
    Biのフォーマットを満たすpd.DataFrameをクラスとして定義。
    Biのフォーマットを規定する定数を格納する。
    constructorは書いていないので、スライスなどの操作をすると返り値はpd.DataFrame。
    """
    # Biのフォーマットに関わる定数
    ROW_NUM = 10000 # 正しい行数
    COLUMNS = get_correct_columns() # 正しい列のリスト
    COL_SPECS = get_col_specs() # 各列の仕様
    
    @classmethod
    def check_col_names(cls, df:pd.DataFrame):
        target_columns = list(df.columns)

        errors = []

        only_target = set(target_columns) - set(cls.COLUMNS)
        if only_target:
            errors.append(ColumnsError(f"不正な列が存在します: {only_target}"))

        only_cor = set(cls.COLUMNS) - set(target_columns)
        if only_cor:
            errors.append(ColumnsError(f"列が不足しています: {only_cor}"))

        # 列の過不足がない場合だけ順番をチェック
        if (not only_target) and (not only_cor) and (target_columns != cls.COLUMNS):
            errors.append(ColumnsOrderError(f"列の順番が不正です: 正{cls.COLUMNS}, 誤{target_columns}"))

        if errors:
            raise ExceptionGroup("列に不正を検知", errors)

    @classmethod
    def check_row_num(cls, df:pd.DataFrame):
        row_num = df.shape[0]

        if row_num != cls.ROW_NUM:
            raise RowNumError(f"期待される行数は{cls.ROW_NUM}, 実際の行数は{row_num}")
        
    @classmethod
    def check_col_specs(cls, df:pd.DataFrame):
        target_columns = list(df.columns)

        errors = []

        for col, col_spec in cls.COL_SPECS.items():
            if col not in target_columns:
                # フォーマットをチェックしているデータにあるべき列がない場合はエラー
                raise ColumnsError(f"列がありません: {col}")

            # 値の前後空白除去
            raw_vals = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
            col_type = col_spec.get("type", "")

            if col_type == "number":
                min_val = Decimal(str(col_spec["min"]))
                max_val = Decimal(str(col_spec["max"]))
                for idx, val in raw_vals.items():
                    if val == "":
                        continue
                    try:
                        d = Decimal(val)
                    except InvalidOperation:
                        errors.append(NumSpecError(idx+1, col, val, "数値変換不可"))
                        continue
                    if d < min_val or d > max_val:
                        errors.append(NumSpecError(idx+1, col, val, f"{min_val}〜{max_val}の範囲外"))

            elif col_type == "category":
                allowed = set(col_spec.get("values", []))
                for idx, val in raw_vals.items():
                    if val == "":
                        continue
                    if val not in allowed:
                        errors.append(CatSpecError(idx+1, col, val, f"許可されていない値（{allowed}）"))

            elif col_type == "date":
                # yyyy-mm-dd 固定
                min_dt = pd.to_datetime(col_spec["min"], format="%Y-%m-%d", errors="raise")
                max_dt = pd.to_datetime(col_spec["max"], format="%Y-%m-%d", errors="raise")

                for idx, val in raw_vals.items():
                    if val == "":
                        continue
                    if not YMD_RE.match(val):
                        errors.append(ColSpecError(idx+1, col, val, "日付形式違反（yyyy-mm-dd）"))
                        continue
                    dt = pd.to_datetime(val, format="%Y-%m-%d", errors="coerce")
                    if pd.isna(dt):
                        errors.append(ColSpecError(idx+1, col, val, "日付変換不可（yyyy-mm-dd）"))
                        continue
                    if dt < min_dt or dt > max_dt:
                        errors.append(ColSpecError(idx+1, col, val, f"{min_dt.strftime('%Y-%m-%d')}〜{max_dt.strftime('%Y-%m-%d')}の範囲外"))

            else:
                # デバッグ用。columns_range.jsonを編集した場合に、表示される可能性あり
                print(f"警告: 列 '{col}' のタイプ '{col_type}' は未対応。スキップします。")

            if errors:
                raise ExceptionGroup("仕様に反する列が存在します", errors)

    @classmethod
    def check_format(cls, in_df:pd.DataFrame):
        df = in_df.astype(str)
        errors = []

        try:
            cls.check_col_names(df)
        except ExceptionGroup as eg:
            errors.extend(eg.exceptions)
        except Exception as e:
            errors.append(e)

        try:
            cls.check_row_num(df)
        except Exception as e:
            errors.append(e)
        
        try:
            cls.check_col_specs(df)
        except ExceptionGroup as eg:
            errors.extend(eg.exceptions)
        except Exception as e:
            errors.append(e)

        if errors:
            raise ExceptionGroup("フォーマットに不正が見つかりました", errors)
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.check_format(self)
        

    def to_csv(self, path_to_output):
        super().to_csv(path_to_output, index=False)

    @classmethod
    def read_csv(cls, path_to_csv):
        """
        CSVファイルがフォーマットを満たしていると仮定して読み込む
        """
        # 文字列で読み込み（欠損は空文字）
        # この時点でエラーが出た場合は修正できない
        df = pd.read_csv(path_to_csv, dtype=str, keep_default_na=False)

        # dfをCiと解釈する。できなければexceptionが出て失敗
        df = cls(df)
        
        return df

class CiDataFrame(BiDataFrame):
    """
    Ciのフォーマットを満たすpd.DataFrameをクラスとして定義。
    
    __init__でインスタンスを作るとき、入力されたdfがフォーマットを満たすか確認し、
    満たしていない場合は修正を試みる。
    
    pd.DataFrameを継承しているので、それでできることは大抵できる。
    __init__とto_csvはoverrideしてあるので、挙動が異なる。

    一部の機能だけを呼び出せるように、
    チェック関数はチェック項目ごとに分割した上でクラスメソッドとして定義
    """
    # Ciのフォーマットに関わる変数はBiと同じなので独自には定義しない
    
    @classmethod
    def fix_col_order(cls, df:pd.DataFrame):
        """
        列を正しい順番に並べ替える
        列に過不足がある時に実行するとエラー
        """
        return df.reindex(columns=get_correct_columns())

    @classmethod
    def fix_num_columns(cls, df:pd.DataFrame):
        df = df.astype(str)
        target_columns = df.columns
        num_fixes: list[tuple[int, str, str, str, str]] = []  # (row, col, original, fixed, reason)

        # CSVにある列をスキーマでチェック
        for col, col_spec in cls.COL_SPECS.items():
            if col not in target_columns:
                """
                フォーマットをチェックしているデータにあるべき列がない場合は
                列の確認をスキップ
                ここはチェックが目的ではないので、修正を続行
                """
                continue

            # 値の前後空白除去
            raw_vals = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
            col_type = col_spec.get("type", "")

            if col_type == "number":
                # 必須: min/max、任意: max_decimal_places
                try:
                    min_val = Decimal(str(col_spec["min"]))
                    max_val = Decimal(str(col_spec["max"]))
                except Exception as e:
                    print(f"エラー: {col} の数値範囲(min/max)がJSONで不正です。")
                    raise e
                
                places = col_spec.get("max_decimal_places", None)
                if places is not None:
                    try:
                        places = int(places)
                    except Exception as e:
                        print(f"エラー: {col} の max_decimal_places が不正です。")
                        raise e

                fixed_vals: list[str] = []
                for i, raw in raw_vals.items():
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
        
        # 数値補正のレポート
        if num_fixes:
            print(f"数値の補正を {len(num_fixes)} 件行いました（min/max クランプ、丸め等）。")
            for row, col, org, fix, rsn in num_fixes[:20]:
                print(f"  行{row} 列'{col}' '{org}' -> '{fix}' （{rsn}）")
            if len(num_fixes) > 20:
                print(f"  ... 省略（合計 {len(num_fixes)} 件）")
        else:
            print("数値の補正はありません。")

        return df

    def __init__(self, *args, **kwargs):
        # self = pd.DataFrame.__init__(*args, **kwargs)
        try:
            try:
                # DataFrameをCiと解釈できるかtry
                # self.__class__.check_format(self)
                super().__init__(*args, **kwargs)
            except* ColumnsOrderError:
                # 列の順番を直す
                self = self.__class__.fix_col_order(self)
                print("列の順番を修正しました")
            except* NumSpecError:
                # 数値列の異常を強制的に直す
                self.__class__.fix_num_columns(self)
                print("数値列の異常値を最も近い正常値に修正しました")
        except ExceptionGroup as eg:
            # 修正できないタイプのExceptionが上がってきたら諦める
            print("修正不可能なフォーマット違反がありました:")
            raise eg
        except Exception as e:
            # except*ブロックで予期せぬエラーが発生した時
            print(f"フォーマットの修正を試みましたが失敗しました:")
            raise e

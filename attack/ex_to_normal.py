# %%
import pandas as pd
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Read a CSV (no header), extract 1st column, binarize (>0 ->1), and save as 1-column CSV."
    )
    ap.add_argument("input_csv", help="Input CSV path (no header)")
    ap.add_argument("output_csv", help="Output CSV path (1 column, no header)")
    args = ap.parse_args()

    # %%
    # CSVをヘッダー無しで読み込み
    ex = pd.read_csv(args.input_csv, header=None)

    # %%
    # 1列目のみ抜き出し & コピー
    inferred = ex.iloc[:, 0].copy()

    # %%
    # 0/1に正規化
    inferred[inferred > 0] = 1

    # %%
    # 保存
    inferred.to_csv(args.output_csv, index=False, header=False)

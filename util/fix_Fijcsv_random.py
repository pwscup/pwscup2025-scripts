import pandas as pd
import argparse
import numpy as np

### 1の数が10000を超える場合、1の数を10000に減らす
def fix_attack_csv(path_to_in, path_to_out):
    df = pd.read_csv(path_to_in, header=None)
    count_1 = (df[0]==1).sum()
    if count_1 > 10000:
        # 1のインデックスを取得
        indices_1 = df.index[df[0] == 1].tolist()
        # ランダムに10000個選ぶ
        selected_indices = np.random.choice(indices_1, size=10000, replace=False)
        # 1のインデックスを0に変更
        df.loc[~df.index.isin(selected_indices), 0] = 0
        # 変更した数を出力
        print(f"Fixed: Reduced number of 1s from {count_1} to 10000.")
    else:
        print(f"No fix needed: Number of 1s is {count_1}, which is within the limit.")
    df.to_csv(path_to_out, index=False, header=False)

# コマンドライン引数の処理
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix attack CSV to ensure the number of 1s does not exceed 10000.")
    parser.add_argument("input", help="Path to the input CSV file.")
    parser.add_argument("output", help="Path to the output CSV file.")
    args = parser.parse_args()
    
    fix_attack_csv(args.input, args.output)
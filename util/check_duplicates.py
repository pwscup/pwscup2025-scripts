import sys
import pandas as pd

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 check_duplicates.py <input.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"CSV読み込みエラー: {e}")
        sys.exit(1)

    # 重複行の抽出（全列で比較）
    duplicates = df[df.duplicated(keep=False)]

    if duplicates.empty:
        print("重複行はありませんでした。")
    else:
        print("重複行が見つかりました。")
        print(f"重複行数: {len(duplicates)}")
        print("---- 重複行 ----")
        print(duplicates)
        print("------------------------")

if __name__ == "__main__":
    main()

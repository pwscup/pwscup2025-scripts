from abc import ABC, abstractmethod
import argparse

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from mia import build_feature_matrices

class AttackCiBase(ABC):
    def __init__(self, path_to_Ci_csv):
        self.Ci_df = pd.read_csv(path_to_Ci_csv, dtype=str, 
                            keep_default_na=False)
        self.inferred = None

    @abstractmethod
    def infer(self, path_to_Ai_csv):
        pass

    def save_inferred(self, path_to_output):
        # 1列・ヘッダー無しで保存（行数 = input1 の行数）
        if self.inferred is None:
            print("inferred is None. No file was saved.")
        else:
            self.inferred.to_csv(path_to_output, index=False, header=False)
            print("inferred was successfully saved.")

class AttackCiNN(AttackCiBase):
    """
    mia.pyと本質的に同じ
    """
    def infer(self, path_to_Ai_csv):
        Ai_df = pd.read_csv(path_to_Ai_csv, dtype=str, 
                            keep_default_na=False)
        # 特徴行列作成
        X1, X2 = build_feature_matrices(Ai_df, self.Ci_df)

        # 比較次元が 0（共通列なし or すべて除外）の場合は全て 0 を出力
        m = len(Ai_df)
        if X1.shape[1] == 0 or X2.shape[1] == 0 or m == 0:
            self.inferred = pd.Series(np.zeros(m, dtype=int))
            return

        # 最近傍検索（マンハッタン距離：数値[0,1] + one-hot を統一的に扱える）
        nn = NearestNeighbors(n_neighbors=1, metric="manhattan")
        nn.fit(X1)
        idx = nn.kneighbors(X2, n_neighbors=1, return_distance=False).ravel()  # 0-based indices into df1

        # 重複をまとめて 1
        marks = np.zeros(m, dtype=int)
        if idx.size > 0:
            marks[np.unique(idx)] = 1

        
        self.inferred = pd.DataFrame(marks, dtype=int)

        return self.inferred


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="For each row in Ci, mark its nearest row in Ai (1), others 0. Output: 1-column CSV.")
    ap.add_argument("path_to_Ai_csv", help="CSV with header (reference; e.g., 100000 rows)")
    ap.add_argument("path_to_Ci_csv", help="CSV with header (query)")
    ap.add_argument("-o", "--out", default="Fij.csv", help="output CSV path (1 column, no header)")
    args = ap.parse_args()

    attacker = AttackCiNN(args.path_to_Ci_csv)
    attacker.infer(args.path_to_Ai_csv)
    attacker.save_inferred(args.out)

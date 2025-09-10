from abc import ABC, abstractmethod
import argparse

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from mia import build_feature_matrices

class AttackCiBase(ABC):
    def __init__(self, path_to_Ci_csv, k):
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
    
class AttackCiNN_extended(AttackCiBase):
    """
    AttackCiNN の k-NN 拡張
    返り値(2列):
      col0: Ai 各行が（Ci全行×k個の）近傍集合の中で選ばれた回数（頻度）
      col1: その Ai 行に対して観測された最小距離（float）
    """
    def infer(self, path_to_Ai_csv, k=1):
        Ai_df = pd.read_csv(path_to_Ai_csv, dtype=str, 
                            keep_default_na=False)
        # 特徴行列作成
        X1, X2 = build_feature_matrices(Ai_df, self.Ci_df)

        # 比較次元が 0（共通列なし or すべて除外）の場合は全て 0 を出力
        m = len(Ai_df)
        if X1.shape[1] == 0 or X2.shape[1] == 0 or m == 0:
            out = pd.DataFrame({
                "knn_hits": np.zeros(m, dtype=int),
                "min_dist": np.full(m, np.nan, dtype=float)
            })
            self.inferred = out
            return self.inferred

        # 最近傍検索（マンハッタン距離：数値[0,1] + one-hot を統一的に扱える）
        nn = NearestNeighbors(n_neighbors=k, metric="manhattan")
        nn.fit(X1)
        dists, inds = nn.kneighbors(X2, n_neighbors=k, return_distance=True)  

        # 平坦化（Ci全行×k個）
        idx = inds.ravel() # 0-based indices into df1
        distances = dists.ravel()  # flatten distances

        # 1) 出現回数（頻度）を数える
        hits = np.bincount(idx, minlength=m).astype(int)

        # 2) インデックスごとの最小距離を取る
        min_dist = np.full(m, np.inf, dtype=float)
        # min_dist[idx] = np.minimum(min_dist[idx], distances) は使えないので:
        np.minimum.at(min_dist, idx, distances)
        # 近傍に一度も選ばれていない行は inf → 1000 に
        min_dist[~np.isfinite(min_dist)] = 1000.0

        # 2列の DataFrame に
        self.inferred = pd.DataFrame({
            "knn_hits": hits,
            "min_dist": min_dist
        })

        return self.inferred

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="For each row in Ci, mark its nearest row in Ai (1), others 0. Output: 1-column CSV.")
    ap.add_argument("path_to_Ai_csv", help="CSV with header (reference; e.g., 100000 rows)")
    ap.add_argument("path_to_Ci_csv", help="CSV with header (query)")
    ap.add_argument("-o", "--out", default="Fij.csv", help="output CSV path (1 column, no header)")
    ap.add_argument("-k", "--k", default=1, type=int, help="number of nearest neighbors")
    args = ap.parse_args()

    attacker = AttackCiNN_extended(args.path_to_Ci_csv, args.k)
    attacker.infer(args.path_to_Ai_csv)
    attacker.save_inferred(args.out)        

# %%
from abc import ABC, abstractmethod
import argparse

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from mia import build_feature_matrices

# %%
class AttackCiBase(ABC):
    def __init__(self, path_to_Ci_csv, k):
        self.Ci_df = pd.read_csv(path_to_Ci_csv, dtype=str, 
                            keep_default_na=False)
        self.inferred = None
        self.k = int(k)

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

# %%
# class AttackCiNN(AttackCiBase):
#     """
#     mia.pyと本質的に同じ
#     """
#     def infer(self, path_to_Ai_csv, k):
#         Ai_df = pd.read_csv(path_to_Ai_csv, dtype=str, 
#                             keep_default_na=False)
#         # 特徴行列作成
#         X1, X2 = build_feature_matrices(Ai_df, self.Ci_df)

#         # 比較次元が 0（共通列なし or すべて除外）の場合は全て 0 を出力
#         m = len(Ai_df)
#         if X1.shape[1] == 0 or X2.shape[1] == 0 or m == 0:
#             self.inferred = pd.Series(np.zeros(m, dtype=int))
#             return

#         # 最近傍検索（マンハッタン距離：数値[0,1] + one-hot を統一的に扱える）
#         nn = NearestNeighbors(n_neighbors=1, metric="manhattan")
#         nn.fit(X1)
#         idx = nn.kneighbors(X2, n_neighbors=1, return_distance=False).ravel()  # 0-based indices into df1

#         # 重複をまとめて 1
#         marks = np.zeros(m, dtype=int)
#         if idx.size > 0:
#             marks[np.unique(idx)] = 1

        
#         self.inferred = pd.DataFrame(marks, dtype=int)

#         return self.inferred

# %%
class AttackCiNN(AttackCiBase):
    """
    AttackCiNN の k-NN 拡張
    返り値(2列):
      col0: Ai 各行が（Ci全行×k個の）近傍集合の中で選ばれた回数（頻度）
      col1: その Ai 行に対して観測された最小距離（float）
    """
    def infer(self, path_to_Ai_csv):
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
        nn = NearestNeighbors(n_neighbors=self.k, metric="manhattan")
        nn.fit(X1) # fit on Ci
        dists, inds = nn.kneighbors(X2, n_neighbors=self.k, return_distance=True)
        print(f"dists: {dists}")
        print(f"inds: {inds}")
        print(f"dists.shape: {dists.shape}, inds.shape: {inds.shape}")
        print(dists[0][0], inds[0][0])  # Aiの1行目に対する最近傍1個目の距離とインデックス
        
        # 平坦化（Ci全行×k個）
        distances = dists.ravel()  # flatten distances
        idx = inds.ravel() # 0-based indices into df1
        # idxにはCiに対応する10000件の最近傍インデックス入っている
        # distancesにはその距離が入っている
        print(f"idx: {idx}")
        print(f"distances: {distances}")
        print(f"idx.shape: {idx.shape}, distances.shape: {distances.shape}")

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

# %%
class AttackCiGreedyMatch(AttackCiBase):
    """
    Greedy 1-to-1 matching:
      - Use kNN ranks: r=0 (1st NN), then r=1 (2nd NN), ...
      - Within each rank, assign pairs by ascending distance, skipping already-used Ai and already-assigned Ci.
      - Stops when all Ci are matched (ideally |Ci| == 10000).
    Outputs:
      - self.inferred: 1-column (0/1) length |Ai| -> assigned Ai rows marked 1
      - self.match_table_: DataFrame(ci_idx, ai_idx, distance, rank) with |Ci| rows
    """
    def infer(self, path_to_Ai_csv):
        Ai_df = pd.read_csv(path_to_Ai_csv, dtype=str, keep_default_na=False)

        # 特徴行列（X1: Ai, X2: Ci）
        X1, X2 = build_feature_matrices(Ai_df, self.Ci_df)

        m = len(Ai_df)
        n = len(self.Ci_df)

        # 比較次元が0なら全0と空の対応表
        if X1.shape[1] == 0 or X2.shape[1] == 0 or m == 0 or n == 0:
            self.inferred = pd.DataFrame(np.zeros(m, dtype=int))
            self.match_table_ = pd.DataFrame(columns=["ci_idx", "ai_idx", "distance", "rank"])
            return self.inferred

        # kNN（マンハッタン距離）
        nn = NearestNeighbors(n_neighbors=self.k, metric="manhattan")
        nn.fit(X1)  # fit on Ai
        dists, inds = nn.kneighbors(X2, n_neighbors=self.k, return_distance=True)
        # dists, inds: shape = (n_Ci, k)

        assigned_ci = np.full(n, -1, dtype=int)     # 各Ciに割当てた Ai index（未割当は -1）
        used_ai = np.zeros(m, dtype=bool)           # 既に割当てに使った Ai をブールで管理
        pairs = []                                  # 確定した (ci, ai, dist, rank) を貯める

        # ランク r=0..k-1 の順で処理
        for r in range(self.k):
            # 未割当のCiについて、(distance, ci_idx, ai_idx) を作る
            ci_unassigned = np.flatnonzero(assigned_ci == -1)
            if ci_unassigned.size == 0:
                break

            # 距離・候補を取り出し
            cand_d = dists[ci_unassigned, r]
            cand_ai = inds[ci_unassigned, r]

            # 距離昇順に並べて貪欲に確定
            order = np.argsort(cand_d)
            for t in order:
                ci = ci_unassigned[t]
                ai = cand_ai[t]
                dist = cand_d[t]
                if assigned_ci[ci] == -1 and not used_ai[ai]:
                    assigned_ci[ci] = ai
                    used_ai[ai] = True
                    pairs.append((ci, ai, float(dist), r))

            # ここで次の rank へ（残った未割当Ciを続きで埋めていく）

        # 念のため、未割当が残っていたらエラーメッセージ
        remaining = int((assigned_ci == -1).sum())
        if remaining > 0:
            print(f"[Warn] {remaining} Ci rows are still unmatched after r=0..{self.k-1}. "
                  f"Consider increasing k.")

        # 出力1: Ai 側の0/1ベクトル
        marks = np.zeros(m, dtype=int)
        for ai in assigned_ci:
            if ai >= 0:
                marks[ai] = 1
        self.inferred = pd.DataFrame(marks.astype(int))

        # 出力2: 対応表
        self.match_table_ = pd.DataFrame(pairs, columns=["ci_idx", "ai_idx", "distance", "rank"])

        return self.inferred


# %%
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(description="For each row in Ci, mark its nearest row in Ai (1), others 0. Output: 1-column CSV.")
#     ap.add_argument("path_to_Ai_csv", help="CSV with header (reference; e.g., 100000 rows)")
#     ap.add_argument("path_to_Ci_csv", help="CSV with header (query)")
#     ap.add_argument("-o", "--out", default="Fij.csv", help="output CSV path (1 column, no header)")
#     args = ap.parse_()

#     attacker = AttackCiNN(args.path_to_Ci_csv)
#     attacker.infer(args.path_to_Ai_csv)
#     attacker.save_inferred(args.out)

# %%
# ap = argparse.ArgumentParser(description="For each row in Ci, mark its nearest row in Ai (1), others 0. Output: 1-column CSV.")
# ap.add_argument("path_to_Ai_csv", help="CSV with header (reference; e.g., 100000 rows)")
# ap.add_argument("path_to_Ci_csv", help="CSV with header (query)")
# ap.add_argument("-o", "--out", default="Fij.csv", help="output CSV path (1 column, no header)")
# args = ap.parse_args()

# # %%
# id = "01"
# path_to_Ci_csv = f"C:\\Users\\kikus\\Documents\\統数研\\pwscup2025-scripts\\out\\PWSCUP2025_Pre_Data_for_Attack\\C{id}.csv"
# path_to_Ai_csv = f"C:\\Users\\kikus\\Documents\\統数研\\pwscup2025-scripts\\out\\PWSCUP2025_Pre_Data_for_Attack\\A{id}.csv"
# out = f"C:\\Users\\kikus\\Documents\\統数研\\pwscup2025-scripts\\out\\C{id}_inferred_ex.csv"
# k = 2  # k-NN


# # %%
# id = "22"
# path_to_Ci_csv = f"C:\\Users\\kikus\\Documents\\統数研\\pwscup2025-scripts\\out\\PWSCUP2025_Pre_Data_for_Attack\\C{id}.csv"
# path_to_Ai_csv = f"C:\\Users\\kikus\\Documents\\統数研\\pwscup2025-scripts\\out\\PWSCUP2025_Pre_Data_for_Attack\\A{id}.csv"
# out = f"C:\\Users\\kikus\\Documents\\統数研\\pwscup2025-scripts\\out\\C{id}_inferred_ex_greedy.csv"
# out_map = f"C:\\Users\\kikus\\Documents\\統数研\\pwscup2025-scripts\\out\\C{id}_matchmap_ex_greedy.csv"
# k = 20  # まずは5～10程度を推奨。足りなければ増やす

# # %%
# k = 25  # まずは5～10程度を推奨。足りなければ増やす
# attacker = AttackCiGreedyMatch(path_to_Ci_csv, k)
# _ = attacker.infer(path_to_Ai_csv)
# # 0/1ベクトル（|Ai|）:
# attacker.save_inferred(out)  # 例: C01_inferred.csv
# # 対応表（|Ci|行）:
# attacker.match_table_.to_csv(out_map, index=False)  # 例: C01_matchmap.csv


# %%
# attacker = AttackCiNN(path_to_Ci_csv)
# attacker.infer(path_to_Ai_csv)
# attacker.save_inferred(out)


# %%
# attacker = AttackCiNN(path_to_Ci_csv, k)
# attacker.infer(path_to_Ai_csv)
# attacker.save_inferred(out)

# %%
# import pandas as pd
# filename = logfile_pre_attack_Ci = "C:\\Users\\kikus\\Documents\\統数研\\pwscup2025-scripts\\out\\log_attack_Ci.txt"
# for id_int in range(1, 21):
#     id = f"{id_int:02d}"
#     path_to_Ci_csv = f"C:\\Users\\kikus\\Documents\\統数研\\pwscup2025-scripts\\out\\PWSCUP2025_Pre_Data_for_Attack\\C{id}.csv"
#     path_to_Ai_csv = f"C:\\Users\\kikus\\Documents\\統数研\\pwscup2025-scripts\\out\\PWSCUP2025_Pre_Data_for_Attack\\A{id}.csv"
#     out = f"C:\\Users\\kikus\\Documents\\統数研\\pwscup2025-scripts\\out\\C{id}_inferred.csv"
#     attacker = AttackCiNN(path_to_Ci_csv)
#     attacker.infer(path_to_Ai_csv)
#     attacker.save_inferred(out)
#     count = pd.read_csv(out, header=None).value_counts()
#     with open(filename, 'a') as f:
#         print(f"{id}: {count}", file=f)
    
# id_int = 22
# id = f"{id_int:02d}"
# path_to_Ci_csv = f"C:\\Users\\kikus\\Documents\\統数研\\pwscup2025-scripts\\out\\PWSCUP2025_Pre_Data_for_Attack\\C{id}.csv"
# path_to_Ai_csv = f"C:\\Users\\kikus\\Documents\\統数研\\pwscup2025-scripts\\out\\PWSCUP2025_Pre_Data_for_Attack\\A{id}.csv"
# out = f"C:\\Users\\kikus\\Documents\\統数研\\pwscup2025-scripts\\out\\C{id}_inferred.csv"
# attacker = AttackCiNN(path_to_Ci_csv)
# attacker.infer(path_to_Ai_csv)
# attacker.save_inferred(out)
# count = pd.read_csv(out, header=None).value_counts()
# with open(filename, 'a') as f:
#     print(f"{id}: {count}", file=f)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="k-NN based Ci→Ai inference (NN count/min-dist) or greedy 1-to-1 matching."
    )
    ap.add_argument("path_to_Ci_csv", help="Path to Ci.csv (query set ~10k rows)")
    ap.add_argument("path_to_Ai_csv", help="Path to Ai.csv (reference set ~100k rows)")
    ap.add_argument("-m", "--mode", choices=["nn", "greedy"], default="nn",
                    help="nn: output knn_hits & min_dist per Ai; greedy: 1-to-1 matching using ranks")
    ap.add_argument("-k", "--k", type=int, default=5,
                    help="number of neighbors (k). For greedy, increase if some Ci remain unmatched.")
    ap.add_argument("-o", "--out", default="Fij.csv",
                    help="Output CSV path. For 'nn': 2 columns [knn_hits, min_dist]. For 'greedy': 1 column (0/1).")
    ap.add_argument("--out-map", default=None,
                    help="(greedy only) Optional path to save match table [ci_idx, ai_idx, distance, rank].")
    args = ap.parse_args()

    # インスタンス生成
    if args.mode == "nn":
        attacker = AttackCiNN(args.path_to_Ci_csv, args.k)
    else:
        attacker = AttackCiGreedyMatch(args.path_to_Ci_csv, args.k)

    # 推論
    _ = attacker.infer(args.path_to_Ai_csv)

    # 保存
    attacker.save_inferred(args.out)

    # 追加出力（greedyの対応表）
    if args.mode == "greedy" and getattr(attacker, "match_table_", None) is not None:
        if args.out_map:
            attacker.match_table_.to_csv(args.out_map, index=False)
            print(f"match table was successfully saved to {args.out_map}")

    # ちょい統計
    try:
        if args.mode == "nn":
            df = attacker.inferred
            hit_rows = int((df.iloc[:, 0] > 0).sum())
            total_hits = int(df.iloc[:, 0].sum())
            md_median = float(df.iloc[:, 1].median())
            md_mean = float(df.iloc[:, 1].mean())
            print(f"[stats nn] hit_rows={hit_rows}, total_hits={total_hits}, "
                  f"min_dist_median={md_median:.6g}, min_dist_mean={md_mean:.6g}")
        else:
            s = attacker.inferred.iloc[:, 0]
            print(f"[stats greedy] selected={int(s.sum())}/{len(s)}")
    except Exception as e:
        print(f"[warn] failed to print stats: {e}")

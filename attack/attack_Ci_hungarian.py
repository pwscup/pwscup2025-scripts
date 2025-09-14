from abc import ABC
import argparse
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from mia import build_feature_matrices


class AttackCiHungarian(ABC):
    """
    Hungarian (assignment) based Ci→Ai matching.

    - Build a unified feature space using Ai∩Ci common columns
      (numeric scaled by Ai's min-max, categorical one-hot) via mia.build_feature_matrices.
    - Cost = Manhattan (L1) distance in that space.
    - Solve min-sum assignment (Ci rows to unique Ai rows) with Hungarian.

    Two computation modes:
      - full: compute full pairwise cost (|Ci|×|Ai|) then solve.
      - knn: for each Ci, restrict candidates to top-k Ai neighbors; build a
        reduced dense matrix (|Ci|×U) where U is the number of unique Ai
        referenced by any Ci's top-k, fill others with a large sentinel cost,
        then solve. This is a practical fallback when full is too large.

    Output:
      - 1-column CSV (length = |Ai|) marking matched Ai rows with 1, others 0.
      - Optional mapping table of matched pairs with distances.
    """

    def __init__(self, path_to_Ci_csv: str,
                 mode: str = "auto",
                 k: int = 300,
                 fill_cost: float = 1000.0,
                 max_full_mn: int = 30_000_000,
                 verbose: bool = False):
        self.Ci_df = pd.read_csv(path_to_Ci_csv, dtype=str, keep_default_na=False)
        self.mode = mode
        self.k = int(k)
        self.fill_cost = float(fill_cost)
        self.max_full_mn = int(max_full_mn)
        self.verbose = bool(verbose)

        self.inferred: pd.DataFrame | None = None
        self.match_table_: pd.DataFrame | None = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _features(self, path_to_Ai_csv: str) -> Tuple[np.ndarray, np.ndarray]:
        Ai_df = pd.read_csv(path_to_Ai_csv, dtype=str, keep_default_na=False)
        X1, X2 = build_feature_matrices(Ai_df, self.Ci_df)
        return X1, X2

    def _solve_full(self, X_ai: np.ndarray, X_ci: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, n = X_ci.shape[0], X_ai.shape[0]
        self._log(f"[full] shapes: Ci={m}, Ai={n}, feat={X_ai.shape[1]}")
        # SciPy cdist with cityblock (Manhattan)
        cost = cdist(X_ci, X_ai, metric="cityblock")
        r, c = linear_sum_assignment(cost)
        d = cost[r, c]
        return r, c, d

    def _solve_knn(self, X_ai: np.ndarray, X_ci: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, n = X_ci.shape[0], X_ai.shape[0]
        k = min(max(1, self.k), max(1, n))
        self._log(f"[knn] shapes: Ci={m}, Ai={n}, feat={X_ai.shape[1]}, k={k}")

        # Fit k-NN on Ai; query Ci
        nn = NearestNeighbors(n_neighbors=k, metric="manhattan")
        nn.fit(X_ai)
        dists, inds = nn.kneighbors(X_ci, n_neighbors=k, return_distance=True)

        # Collect unique Ai candidates across all Ci
        uniq_cols = np.unique(inds.ravel())
        col_map: Dict[int, int] = {int(j): i for i, j in enumerate(uniq_cols)}
        U = uniq_cols.size
        if U < m:
            self._log(f"[knn] warning: unique candidate Ai columns U={U} < |Ci|={m}; solution may be partial or high-cost. Consider increasing -k.")

        # Build dense reduced cost matrix (m x U) initialized with fill_cost
        fill = float(self.fill_cost)
        C = np.full((m, U), fill, dtype=np.float64)
        # Fill known neighbor distances
        for i in range(m):
            for t in range(k):
                jj = int(inds[i, t])
                C[i, col_map[jj]] = float(dists[i, t])

        r, c_small = linear_sum_assignment(C)
        d = C[r, c_small]
        # Map reduced column indices back to original Ai indices
        c_orig = np.array([uniq_cols[j] for j in c_small], dtype=int)
        return r, c_orig, d

    def infer(self, path_to_Ai_csv: str):
        X_ai, X_ci = self._features(path_to_Ai_csv)
        m, n = X_ci.shape[0], X_ai.shape[0]

        # If no comparable features or empty inputs, emit zeros
        if X_ai.shape[1] == 0 or X_ci.shape[1] == 0 or m == 0:
            marks = np.zeros(n, dtype=int)
            self.inferred = pd.DataFrame(marks)
            self.match_table_ = pd.DataFrame(columns=["ci_idx", "ai_idx", "distance"])  # empty
            print(f"[AttackCiHungarian] no comparable features or empty inputs; selected=0/{n}")
            return self.inferred

        # Decide mode
        mode = self.mode
        if mode == "auto":
            mode = "full" if (m * n) <= self.max_full_mn else "knn"
            self._log(f"[auto] m*n={m*n} (limit {self.max_full_mn}) -> mode={mode}")

        if mode == "full":
            r, c, d = self._solve_full(X_ai, X_ci)
        elif mode == "knn":
            r, c, d = self._solve_knn(X_ai, X_ci)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Build 0/1 marks over Ai (length n)
        marks = np.zeros(n, dtype=int)
        used_cols = set()
        for rr, cc in zip(r, c):
            if 0 <= cc < n and cc not in used_cols:
                marks[cc] = 1
                used_cols.add(int(cc))

        self.inferred = pd.DataFrame(marks)
        # Save mapping table
        self.match_table_ = pd.DataFrame({
            "ci_idx": r.astype(int),
            "ai_idx": c.astype(int),
            "distance": d.astype(float),
        })

        print(f"[AttackCiHungarian] matched={len(self.match_table_)} of Ci={m}; selected Ai={int(marks.sum())}/{n}")
        if self.mode == "knn" or (self.mode == "auto" and (m * n) > self.max_full_mn):
            # Warn if any selected pair used fill_cost (indicative of insufficient k)
            if (np.asarray(self.match_table_["distance"]) >= (self.fill_cost - 1e-12)).any():
                print("[warn] Some assignments used the fill cost. Increase -k to expand candidate pool.")

        return self.inferred

    def save_inferred(self, path_to_output: str):
        if self.inferred is None:
            print("inferred is None. No file was saved.")
        else:
            self.inferred.to_csv(path_to_output, index=False, header=False)
            print(f"inferred was successfully saved as {path_to_output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ci→Ai Hungarian assignment using unified Manhattan distance (numeric+categorical)")
    ap.add_argument("path_to_Ai_csv", help="CSV with header (reference; e.g., 100000 rows)")
    ap.add_argument("path_to_Ci_csv", help="CSV with header (query; e.g., ~10000 rows)")
    ap.add_argument("-o", "--out", default="Fij.csv", help="output CSV path (1 column, no header)")

    ap.add_argument("-m", "--mode", choices=["auto", "full", "knn"], default="auto",
                    help="auto: full if feasible by size; else knn. full: full pairwise; knn: restrict to top-k per Ci")
    ap.add_argument("-k", "--k", type=int, default=300, help="knn mode: number of nearest Ai candidates per Ci")
    ap.add_argument("--fill-cost", type=float, default=1000.0, help="knn mode: cost for non-candidate pairs")
    ap.add_argument("--max-full-mn", type=int, default=30_000_000, help="max |Ci|×|Ai| to attempt full mode")
    ap.add_argument("--verbose", action="store_true", help="print internal progress logs")
    ap.add_argument("--out-map", default=None, help="optional CSV to save assignment pairs [ci_idx, ai_idx, distance]")

    args = ap.parse_args()

    attacker = AttackCiHungarian(
        path_to_Ci_csv=args.path_to_Ci_csv,
        mode=args.mode,
        k=args.k,
        fill_cost=args.fill_cost,
        max_full_mn=args.max_full_mn,
        verbose=args.verbose,
    )

    attacker.infer(args.path_to_Ai_csv)
    attacker.save_inferred(args.out)
    if args.out_map and attacker.match_table_ is not None:
        attacker.match_table_.to_csv(args.out_map, index=False)
        print(f"match table was successfully saved as {args.out_map}")


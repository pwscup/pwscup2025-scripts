from abc import ABC
import argparse

import numpy as np
import pandas as pd

from mia import build_feature_matrices, infer_numeric_mask


class AttackAllCiAllDiHungarian(ABC):
    """
    Rank all Ai rows by: Hungarian Ci→Ai matched distance + Di |pred - y|.

    - Ci distance: Build a unified feature space (numeric min-max scaled on Ai,
      categorical one-hot) and run Hungarian assignment (min-sum) to map each
      Ci row to a unique Ai row. Use the matched distance per Ai; unmatched Ai
      get a large fill cost.
    - Di distance: From Conf_Attack, take the absolute error vector |pred - y|
      for all Ai rows.
    - Score: score = hungarian_distance + |pred - y| (smaller is better).
      Select top-N rows globally (default: 10000) and output a 0/1 vector.

    Notes:
      - No candidate filtering by Di; this attack considers all Ai rows.
      - Hungarian computation supports 'auto'/'full'/'knn' modes, mirroring
        attack_Ci_hungarian.py.
    """

    def __init__(self, ci_csv, di_json,
                 hung_mode="knn", k=300, fill_cost=1000.0, max_full_mn=30_000_000, verbose=False,
                 topn=10000, w_dist=1.0, w_conf=1.0, auto_wdist=False):
        self.ci_csv = ci_csv
        self.di_json = di_json

        self.hung_mode = hung_mode
        self.k = int(k)
        self.fill_cost = float(fill_cost)
        self.max_full_mn = int(max_full_mn)
        self.verbose = bool(verbose)

        self.topn = int(topn)
        self.w_dist = float(w_dist)   # weight for Ci distance
        self.w_conf = float(w_conf)   # weight for Di |pred - y|
        self.auto_wdist = bool(auto_wdist)

        self.inferred = None
        self.rank_table_ = None
        self.match_table_ = None

    def _features(self, ai_csv):
        Ai_df = pd.read_csv(ai_csv, dtype=str, keep_default_na=False)
        Ci_df = pd.read_csv(self.ci_csv, dtype=str, keep_default_na=False)
        X_ai, X_ci = build_feature_matrices(Ai_df, Ci_df)
        return X_ai, X_ci

    def _solve_full_with_di(self, X_ai: np.ndarray, X_ci: np.ndarray, di_cost: np.ndarray, w_dist_eff: float):
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment

        m, n = X_ci.shape[0], X_ai.shape[0]
        if self.verbose:
            print(f"[full+di] shapes: Ci={m}, Ai={n}, feat={X_ai.shape[1]}")
        base = cdist(X_ci, X_ai, metric="cityblock")  # CiDist
        C = (w_dist_eff * base) + (self.w_conf * di_cost[np.newaxis, :])
        r, c = linear_sum_assignment(C)
        d = C[r, c]
        return r, c, d

    def _solve_knn_with_di(self, X_ai: np.ndarray, X_ci: np.ndarray, di_cost: np.ndarray, w_dist_eff: float):
        from sklearn.neighbors import NearestNeighbors
        from scipy.optimize import linear_sum_assignment

        m, n = X_ci.shape[0], X_ai.shape[0]
        k = min(max(1, self.k), max(1, n))
        if self.verbose:
            print(f"[knn+di] shapes: Ci={m}, Ai={n}, feat={X_ai.shape[1]}, k={k}")

        nn = NearestNeighbors(n_neighbors=k, metric="manhattan")
        nn.fit(X_ai)
        dists, inds = nn.kneighbors(X_ci, n_neighbors=k, return_distance=True)

        uniq_cols = np.unique(inds.ravel())
        col_map = {int(j): i for i, j in enumerate(uniq_cols)}
        U = uniq_cols.size
        if U < m and self.verbose:
            print(f"[knn+di] warning: unique Ai candidates U={U} < |Ci|={m}")

        # Initialize cost with w_dist*fill_cost + w_conf*per-column Di cost
        fill = float(self.fill_cost)
        base_col = di_cost[uniq_cols].astype(float)  # shape (U,)
        C = (self.w_conf * np.tile(base_col, (m, 1)).astype(np.float64))
        C += (w_dist_eff * fill)

        # Fill neighbor distances + Di cost
        for i in range(m):
            for t in range(k):
                jj = int(inds[i, t])
                C[i, col_map[jj]] = (w_dist_eff * float(dists[i, t])) + (self.w_conf * float(di_cost[jj]))

        r, c_small = linear_sum_assignment(C)
        d = C[r, c_small]
        c_orig = np.array([uniq_cols[j] for j in c_small], dtype=int)
        return r, c_orig, d

    def _di_abs_error(self, ai_csv):
        # Use Conf_Attack to compute |pred - y| for all Ai rows
        from attack_Di_ex import Conf_Attack
        conf = Conf_Attack(self.di_json)
        _ = conf.infer(ai_csv)
        return conf.score_.astype(float)

    def _effective_wdist(self, ai_csv):
        if not self.auto_wdist:
            return self.w_dist
        try:
            Ai_df = pd.read_csv(ai_csv, dtype=str, keep_default_na=False)
            Ci_df = pd.read_csv(self.ci_csv, dtype=str, keep_default_na=False)
            common = [c for c in Ai_df.columns if c in Ci_df.columns]
            if not common:
                return self.w_dist
            num_mask = infer_numeric_mask(Ai_df, common)
            n_num = int(num_mask.sum())
            n_cat = int((~num_mask).sum())
            denom = n_num + 2 * n_cat
            if denom > 0:
                w = self.w_dist / float(denom)
                print(f"[auto-wdist] N={n_num}, C={n_cat}, denom={denom}, w_dist_eff={w:.6g}")
                return w
        except Exception as e:
            print(f"[auto-wdist warn] failed to compute feature-based scaling: {e}")
        return self.w_dist

    def infer(self, ai_csv):
        # Build features and Di per-Ai cost
        X_ai, X_ci = self._features(ai_csv)
        m, n = X_ci.shape[0], X_ai.shape[0]

        # If no comparable features or empty inputs, emit zeros
        if X_ai.shape[1] == 0 or X_ci.shape[1] == 0 or m == 0 or n == 0:
            marks = np.zeros(n, dtype=int)
            self.inferred = pd.DataFrame(marks)
            self.match_table_ = pd.DataFrame(columns=["ci_idx", "ai_idx", "distance"])  # empty
            print(f"[AttackAllCiAllDiHungarian] no comparable features or empty inputs; selected=0/{n}")
            return self.inferred

        di_cost = self._di_abs_error(ai_csv).astype(float)
        if di_cost.shape[0] != n:
            raise ValueError("Di cost size mismatches Ai rows")

        # Decide mode
        mode = self.hung_mode
        if mode == "auto":
            mode = "full" if (m * n) <= self.max_full_mn else "knn"
            if self.verbose:
                print(f"[auto] m*n={m*n} (limit {self.max_full_mn}) -> mode={mode}")

        w_dist_eff = self._effective_wdist(ai_csv)
        if mode == "full":
            r, c, d = self._solve_full_with_di(X_ai, X_ci, di_cost, w_dist_eff)
        elif mode == "knn":
            r, c, d = self._solve_knn_with_di(X_ai, X_ci, di_cost, w_dist_eff)
        else:
            raise ValueError(f"Unknown mode: {self.hung_mode}")

        # Build 0/1 marks over Ai (length n); optionally cap to topn by minimal total cost
        pairs = list(zip(r.astype(int), c.astype(int), d.astype(float)))
        # Sort by cost ascending and pick up to topn
        pairs.sort(key=lambda x: (x[2], x[1]))
        use_k = min(self.topn, len(pairs)) if self.topn is not None else len(pairs)
        chosen = pairs[:use_k]

        marks = np.zeros(n, dtype=int)
        for _, ai_idx, _ in chosen:
            if 0 <= ai_idx < n:
                marks[ai_idx] = 1

        self.inferred = pd.DataFrame(marks)
        # per-pair components (recompute Ci L1 distance for clarity)
        ci_comp = []
        di_comp = []
        for ci_idx, ai_idx, _ in pairs:
            d_ci = float(np.abs(X_ci[ci_idx] - X_ai[ai_idx]).sum())
            ci_comp.append(d_ci)
            di_comp.append(float(di_cost[ai_idx]))

        self.match_table_ = pd.DataFrame({
            "ci_idx": np.array([ci for ci, _, _ in pairs], dtype=int),
            "ai_idx": np.array([ai for _, ai, _ in pairs], dtype=int),
            "distance": np.array([cost for _, _, cost in pairs], dtype=float),  # total weighted cost = w_dist*Ci + w_conf*Di
            "ci_dist": np.array(ci_comp, dtype=float),
            "di_cost": np.array(di_comp, dtype=float),
        })

        # Ranked table over matched pairs
        ord_pairs = np.argsort(self.match_table_["distance"].values)
        self.rank_table_ = self.match_table_.iloc[ord_pairs].reset_index(drop=True)

        print(f"[AttackAllCiAllDiHungarian] matched={len(pairs)} (capped={use_k}); selected Ai={int(marks.sum())}/{n}")
        if mode == "knn" and (self.fill_cost is not None):
            # Warn if any cost equals or exceeds fill_cost baseline (indicative of insufficient k)
            if (self.match_table_["distance"] >= (self.fill_cost - 1e-12)).any():
                print("[warn] Some assignments used the fill cost baseline. Increase -k to expand candidate pool.")
        return self.inferred

    def save_inferred(self, path):
        if self.inferred is None:
            print("inferred is None. No file was saved.")
        else:
            self.inferred.to_csv(path, index=False, header=False)
            print(f"inferred was successfully saved as {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AllCi+AllDi Hungarian: min-sum Hungarian on (w_dist*Ci + w_conf*|pred - y|); select top-N Ai")
    ap.add_argument("Ai_csv", help="Path to Ai.csv")
    ap.add_argument("Ci_csv", help="Path to Ci.csv")
    ap.add_argument("Di_json", help="Path to Di model JSON (Booster.save_model)")

    # Hungarian Ci options
    ap.add_argument("--hung-mode", choices=["auto", "full", "knn"], default="knn", help="Hungarian: 'auto' tries full up to size limit; 'knn' restricts candidates per Ci")
    ap.add_argument("-k", "--k", type=int, default=300, help="Hungarian (knn): number of nearest Ai candidates per Ci")
    ap.add_argument("--fill-cost", type=float, default=1000.0, help="Hungarian (knn): cost for non-candidate pairs")
    ap.add_argument("--max-full-mn", type=int, default=30_000_000, help="Hungarian (auto/full): max |Ci|×|Ai| to attempt full matrix")
    ap.add_argument("--verbose", action="store_true", help="Enable Hungarian internal logs")

    # Scoring weights
    ap.add_argument("--w-dist", type=float, default=1.0, help="weight for Ci distance in cost")
    ap.add_argument("--w-conf", type=float, default=1.0, help="weight for Di |pred - y| in cost")
    ap.add_argument("--auto-wdist", action="store_true", help="auto-scale w_dist by 1/(N + 2*C) using Ai/Ci common columns")

    # Output control
    ap.add_argument("--topn", type=int, default=10000, help="number of Ai rows to mark as 1 (default: 10000)")
    ap.add_argument("-o", "--out", default="Fij.csv", help="output CSV path (1 column, no header)")
    ap.add_argument("--out-rank", default=None, help="optional CSV to save ranked candidates with scores")
    ap.add_argument("--out-map", default=None, help="optional CSV to save full Hungarian match table [ci_idx, ai_idx, distance]")

    args = ap.parse_args()

    attacker = AttackAllCiAllDiHungarian(
        ci_csv=args.Ci_csv,
        di_json=args.Di_json,
        hung_mode=args.hung_mode,
        k=args.k,
        fill_cost=args.fill_cost,
        max_full_mn=args.max_full_mn,
        verbose=args.verbose,
        topn=args.topn,
        w_dist=args.w_dist,
        w_conf=args.w_conf,
        auto_wdist=args.auto_wdist,
    )

    attacker.infer(args.Ai_csv)
    attacker.save_inferred(args.out)
    if args.out_rank and attacker.rank_table_ is not None:
        attacker.rank_table_.to_csv(args.out_rank, index=False)
        print(f"rank table was successfully saved as {args.out_rank}")
    if args.out_map and attacker.match_table_ is not None:
        try:
            attacker.match_table_.to_csv(args.out_map, index=False)
            print(f"match table was successfully saved as {args.out_map}")
        except Exception as e:
            print(f"[warn] failed to save Hungarian match table: {e}")

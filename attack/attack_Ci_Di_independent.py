from abc import ABC
import argparse
import pandas as pd
import numpy as np

# Greedy Ci matcher that assigns up to |Ci| unique Ai with distances
from attack_Ci_ex_greedy import AttackCiGreedyMatchAll as AttackCiGreedy
from attack_Di_ex import Conf_Attack
from mia import infer_numeric_mask


class AttackCiDiIndependent(ABC):
    """
    Independent scoring with greedy Ci distances and Di |pred - y|.

    - Ci step (greedy):
        Use greedy 1-to-1 matching (Ci -> Ai) to get a distance per Ai.
        Unmatched Ai get a large default distance (1000.0). Optionally record
        a 0/1 greedy_match vector indicating whether Ai was matched by any Ci.

    - Di step (independent):
        Run Conf_Attack once and take the absolute error vector |pred - y|.
        No candidate filtering; the vector is used as a penalty term.

    - Scoring and selection:
        score = (w_hits * greedy_match) - (w_dist_eff * greedy_dist) - (w_conf * |pred - y|)
        where w_dist_eff = w_dist or, if --auto-wdist, w_dist/(N + 2*C) using
        common numeric (N) and categorical (C) column counts from Ai and Ci.
        Pick top-N rows globally (default: 10000 or user-specified).
    """

    def __init__(self, ci_csv, di_json, w_hits=0.0, w_dist=1.0, w_conf=1.0,
                 topn=10000, auto_wdist=False, k_hint=300):
        self.ci_csv = ci_csv
        self.di_json = di_json
        self.w_hits = float(w_hits)
        self.w_dist = float(w_dist)
        self.w_conf = float(w_conf)
        self.topn = int(topn)
        self.auto_wdist = bool(auto_wdist)
        self.k_hint = int(k_hint)

        self.inferred = None
        self.rank_table_ = None
        self.match_table_ = None

    def _ci_greedy(self, ai_csv):
        # Run greedy matcher; k acts as an initial hint for rank expansion
        ci = AttackCiGreedy(self.ci_csv, k=self.k_hint)
        _ = ci.infer(ai_csv)
        greedy_match = ci.inferred.iloc[:, 0].to_numpy().astype(int)
        greedy_dist = getattr(ci, "distance_per_ai_", None)
        if greedy_dist is None:
            # Backfill distances via match_table_ if not present
            m = greedy_match.size
            greedy_dist = np.full(m, 1000.0, dtype=float)
            mt = getattr(ci, "match_table_", None)
            if mt is not None and len(mt) > 0:
                for _, row in mt.iterrows():
                    ai_idx = int(row["ai_idx"]) if "ai_idx" in row else int(row[1])
                    dist = float(row["distance"]) if "distance" in row else float(row[2])
                    if 0 <= ai_idx < m:
                        greedy_dist[ai_idx] = dist
        # Save map table if any
        self.match_table_ = getattr(ci, "match_table_", None)
        return greedy_match, greedy_dist

    def _di_abs_error(self, ai_csv):
        # Use Conf_Attack to compute |pred - y| as a vector; ignore its selection mask
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
        # Ci greedy distances and match flags
        greedy_match, greedy_dist = self._ci_greedy(ai_csv)
        # Di absolute errors
        conf_abs = self._di_abs_error(ai_csv)

        n = greedy_match.shape[0]
        if conf_abs.shape[0] != n:
            raise ValueError("Size mismatch between Ci distances and Di scores")

        # Score: larger is better
        w_dist_eff = self._effective_wdist(ai_csv)
        score = (self.w_hits * greedy_match) - (w_dist_eff * greedy_dist) - (self.w_conf * conf_abs)

        # Rank all rows by score desc; tie-break by smaller distance then index
        idx = np.arange(n)
        order = np.lexsort((idx, greedy_dist, -score))
        ranked_idx = idx[order]

        k = max(0, min(self.topn, ranked_idx.size))
        sel = np.zeros(n, dtype=int)
        if k > 0:
            sel[ranked_idx[:k]] = 1

        # Keep rank table for inspection
        self.rank_table_ = pd.DataFrame({
            "ai_idx": ranked_idx,
            "score": score[ranked_idx],
            "greedy_match": greedy_match[ranked_idx],
            "greedy_dist": greedy_dist[ranked_idx],
            "conf_abs_err": conf_abs[ranked_idx],
        })

        self.inferred = pd.DataFrame(sel)
        print(f"[AttackCiDiIndependent] selected={int(sel.sum())}/{n} (topn={self.topn})")
        return self.inferred

    def save_inferred(self, path):
        if self.inferred is None:
            print("inferred is None. No file was saved.")
        else:
            self.inferred.to_csv(path, index=False, header=False)
            print(f"inferred was successfully saved as {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Independent ranking: greedy Ci distance + Di |pred - y|; select top-N"
    )
    ap.add_argument("Ai_csv", help="Path to Ai.csv")
    ap.add_argument("Ci_csv", help="Path to Ci.csv")
    ap.add_argument("Di_json", help="Path to Di model JSON (Booster.save_model)")

    # Scoring knobs
    ap.add_argument("--w-hits", type=float, default=0.0, help="weight for greedy_match (0/1) in score")
    ap.add_argument("--w-dist", type=float, default=1.0, help="weight for greedy_dist in score (larger penalizes distance)")
    ap.add_argument("--w-conf", type=float, default=1.0, help="weight for |pred - y| from Di (smaller is better)")
    ap.add_argument("--auto-wdist", action="store_true", help="auto-scale distance weight by 1/(N + 2*C) using Ai/Ci common columns")
    ap.add_argument("--k-hint", type=int, default=300, help="initial k hint for greedy neighbor expansion")

    # Output control
    ap.add_argument("--topn", type=int, default=10000, help="number of Ai rows to mark as 1 (default: 10000)")
    ap.add_argument("-o", "--out", default="Fij.csv", help="output CSV path (1 column, no header)")
    ap.add_argument("--out-rank", default=None, help="optional CSV to save ranked candidates with scores")
    ap.add_argument("--out-map", default=None, help="optional CSV to save greedy match table [ci_idx, ai_idx, distance, rank]")

    args = ap.parse_args()

    attacker = AttackCiDiIndependent(
        ci_csv=args.Ci_csv,
        di_json=args.Di_json,
        w_hits=args.w_hits,
        w_dist=args.w_dist,
        w_conf=args.w_conf,
        topn=args.topn,
        auto_wdist=args.auto_wdist,
        k_hint=args.k_hint,
    )

    attacker.infer(args.Ai_csv)
    attacker.save_inferred(args.out)
    if args.out_rank and attacker.rank_table_ is not None:
        attacker.rank_table_.to_csv(args.out_rank, index=False)
        print(f"rank table was successfully saved as {args.out_rank}")
    if args.out_map and attacker.match_table_ is not None:
        attacker.match_table_.to_csv(args.out_map, index=False)
        print(f"match table was successfully saved as {args.out_map}")


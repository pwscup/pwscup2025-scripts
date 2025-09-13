from abc import ABC
import argparse

import pandas as pd
import numpy as np

# Use extended Di attacks (support topk/ratio) and k-NN Ci stats
from attack_Ci_ex import AttackCiNN as AttackCiKNN
from mia import infer_numeric_mask


class NewAttackDiCi(ABC):
    """
    Score Di-selected candidates using Ci distances, then pick top-N.

    - Di step: select candidates via Pred_Attack and/or Conf_Attack
      using threshold/topk/ratio options (union or intersection).
    - Ci step: compute k-NN stats (hits, min_dist) for Ai vs Ci; rank
      candidates primarily by smaller min_dist (optionally include hits).
    """
    def __init__(self, ci_csv, di_json, k=5, mode="union", w_hits=0.0, w_dist=1.0, w_conf=1.0, topn=1, auto_wdist=False):
        self.ci_csv = ci_csv
        self.di_json = di_json
        self.k = int(k)
        self.mode = mode
        self.w_hits = float(w_hits)
        self.w_dist = float(w_dist)
        # Smaller |pred - y| is better; weight defaults to 1.0
        self.w_conf = float(w_conf)
        self.topn = int(topn)
        self.auto_wdist = bool(auto_wdist)

        self.inferred = None
        self.rank_table_ = None

    def _select_di(self, ai_csv,
                   pred_threshold=0.5, pred_topk=None, pred_pos_ratio=None,
                   conf_threshold=0.1, conf_topk=None, conf_pos_ratio=None):
        # Lazy-import Di attacks to avoid requiring xgboost on --help
        from attack_Di_ex import Pred_Attack, Conf_Attack

        # Pred
        pred = Pred_Attack(self.di_json,
                           threshold=pred_threshold,
                           topk=pred_topk,
                           pos_ratio=pred_pos_ratio)
        _ = pred.infer(ai_csv)
        s_pred = pred.inferred.iloc[:, 0].astype(int).values

        # Conf
        conf = Conf_Attack(self.di_json,
                           threshold=conf_threshold,
                           topk=conf_topk,
                           pos_ratio=conf_pos_ratio)
        _ = conf.infer(ai_csv)
        s_conf = conf.inferred.iloc[:, 0].astype(int).values
        # Capture |pred - y| vector from Conf_Attack (added attribute)
        conf_score = getattr(conf, "score_", None)

        if self.mode == "intersection":
            cand = (s_pred & s_conf).astype(int)
        else:
            cand = (s_pred | s_conf).astype(int)

        return cand, s_pred, s_conf, conf_score

    def _score_with_ci(self, ai_csv, conf_score=None):
        ci = AttackCiKNN(self.ci_csv, self.k)
        df = ci.infer(ai_csv)
        # hits: larger is better; min_dist: smaller is better
        hits = df.iloc[:, 0].to_numpy()
        md = df.iloc[:, 1].to_numpy()
        # Optional auto scaling for distance weight by (N + 2*C)
        w_dist_eff = self.w_dist
        if self.auto_wdist:
            try:
                Ai_df = pd.read_csv(ai_csv, dtype=str, keep_default_na=False)
                Ci_df = pd.read_csv(self.ci_csv, dtype=str, keep_default_na=False)
                common = [c for c in Ai_df.columns if c in Ci_df.columns]
                if common:
                    num_mask = infer_numeric_mask(Ai_df, common)
                    n_num = int(num_mask.sum())
                    n_cat = int((~num_mask).sum())
                    denom = n_num + 2 * n_cat
                    if denom > 0:
                        w_dist_eff = self.w_dist / float(denom)
                        print(f"[auto-wdist] N={n_num}, C={n_cat}, denom={denom}, w_dist_eff={w_dist_eff:.6g}")
            except Exception as e:
                print(f"[auto-wdist warn] failed to compute feature-based scaling: {e}")
        # Convert to score where larger is better
        score = self.w_hits * hits - w_dist_eff * md
        if conf_score is not None:
            # Smaller |pred - y| is better → subtract with weight
            score = score - self.w_conf * conf_score
        return score, hits, md

    def infer(self, ai_csv,
              pred_threshold=0.5, pred_topk=None, pred_pos_ratio=None,
              conf_threshold=0.1, conf_topk=None, conf_pos_ratio=None):
        # Di step
        cand, s_pred, s_conf, conf_score = self._select_di(
            ai_csv,
            pred_threshold=pred_threshold, pred_topk=pred_topk, pred_pos_ratio=pred_pos_ratio,
            conf_threshold=conf_threshold, conf_topk=conf_topk, conf_pos_ratio=conf_pos_ratio,
        )
        n = cand.shape[0]

        # Ci step scoring
        score, hits, md = self._score_with_ci(ai_csv, conf_score=conf_score)

        # Restrict to candidates if any; otherwise fall back to all rows
        if cand.sum() > 0:
            mask = cand.astype(bool)
        else:
            print("[Warn] Di produced no candidates; ranking all rows by Ci distance.")
            mask = np.ones(n, dtype=bool)

        idx = np.arange(n)
        idx_cand = idx[mask]
        # Sort by score desc (i.e., small distance wins when w_dist>0). Tie-break by distance then index
        order = np.lexsort((idx_cand, md[idx_cand], -score[idx_cand]))
        ranked_idx = idx_cand[order]

        k = max(0, min(self.topn, ranked_idx.size))
        sel = np.zeros(n, dtype=int)
        if k > 0:
            sel[ranked_idx[:k]] = 1

        # Save rank table for debugging
        data = {
            "ai_idx": ranked_idx,
            "score": score[ranked_idx],
            "knn_hits": hits[ranked_idx],
            "min_dist": md[ranked_idx],
            "pred_sel": s_pred[ranked_idx],
            "conf_sel": s_conf[ranked_idx],
        }
        if conf_score is not None:
            data["conf_abs_err"] = conf_score[ranked_idx]

        self.rank_table_ = pd.DataFrame(data)

        self.inferred = pd.DataFrame(sel)
        print(f"[NewAttackDiCi] selected={int(sel.sum())}/{n} (topn={self.topn}, candidates={int(mask.sum())})")
        return self.inferred

    def save_inferred(self, path):
        if self.inferred is None:
            print("inferred is None. No file was saved.")
        else:
            self.inferred.to_csv(path, index=False, header=False)
            print(f"inferred was successfully saved as {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score Di-selected candidates by Ci distance and pick top-N members")
    ap.add_argument("Ai_csv", help="Path to Ai.csv")
    ap.add_argument("Ci_csv", help="Path to Ci.csv")
    ap.add_argument("Di_json", help="Path to Di model JSON (Booster.save_model)")

    # Di selection knobs (mirrors attack_Di_ex)
    ap.add_argument("--pred-threshold", type=float, default=0.5, help="Pred threshold (used when no topk/ratio)")
    ap.add_argument("--pred-topk", type=int, default=None, help="Pred: select exactly top-K rows")
    ap.add_argument("--pred-pos-ratio", type=float, default=None, help="Pred: select round(ratio*N) rows")

    ap.add_argument("--conf-threshold", type=float, default=0.1, help="Conf: |p-y| <= threshold (when no topk/ratio)")
    ap.add_argument("--conf-topk", type=int, default=None, help="Conf: select exactly top-K rows with smallest |p-y|")
    ap.add_argument("--conf-pos-ratio", type=float, default=None, help="Conf: select round(ratio*N) rows")

    ap.add_argument("--mode", choices=["union", "intersection"], default="union", help="Combine Pred/Conf candidates")

    # Ci scoring knobs
    ap.add_argument("-k", "--k", type=int, default=5, help="k neighbors for Ci k-NN stats")
    ap.add_argument("--w-hits", type=float, default=0.0, help="weight for knn_hits in score")
    ap.add_argument("--w-dist", type=float, default=1.0, help="weight for min_dist in score (larger penalizes distance)")
    ap.add_argument("--w-conf", type=float, default=1.0, help="weight for |pred - y| from Di (smaller is better)")
    ap.add_argument("--auto-wdist", action="store_true", help="auto-scale distance weight by 1/(N + 2*C) using Ai/Ci common columns")

    # Output control
    ap.add_argument("--topn", type=int, default=1, help="number of Ai rows to mark as 1 (default: 1)")
    ap.add_argument("-o", "--out", default="Fij.csv", help="output CSV path (1 column, no header)")
    ap.add_argument("--out-rank", default=None, help="optional CSV to save ranked candidates with scores")

    args = ap.parse_args()

    attacker = NewAttackDiCi(
        ci_csv=args.Ci_csv,
        di_json=args.Di_json,
        k=args.k,
        mode=args.mode,
        w_hits=args.w_hits,
        w_dist=args.w_dist,
        w_conf=args.w_conf,
        topn=args.topn,
        auto_wdist=args.auto_wdist,
    )

    attacker.infer(
        args.Ai_csv,
        pred_threshold=args.pred_threshold,
        pred_topk=args.pred_topk,
        pred_pos_ratio=args.pred_pos_ratio,
        conf_threshold=args.conf_threshold,
        conf_topk=args.conf_topk,
        conf_pos_ratio=args.conf_pos_ratio,
    )
    attacker.save_inferred(args.out)
    if args.out_rank and attacker.rank_table_ is not None:
        attacker.rank_table_.to_csv(args.out_rank, index=False)
        print(f"rank table was successfully saved as {args.out_rank}")

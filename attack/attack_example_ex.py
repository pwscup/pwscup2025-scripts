from abc import ABC, abstractmethod
import argparse

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from attack_Ci import AttackCiNN
from attack_Di import Conf_Attack, Pred_Attack

## csvから攻撃結果を読み込むバージョン
from abc import ABC, abstractmethod
import argparse
import pandas as pd
import numpy as np

class AttackBase(ABC):
    def __init__(self):
        self.inferred = None

    @abstractmethod
    def infer(self):
        ...

    def save_inferred(self, path_to_output):
        if self.inferred is None:
            print("inferred is None. No file was saved.")
        else:
            self.inferred.to_csv(path_to_output, index=False, header=False)
            print(f"inferred was successfully saved as {path_to_output}")

def _load_ci_binary(path, expected_len=None):
    """
    Ciの出力を読み込む。
    - 2列 (knn_hits, min_dist) の場合: (knn_hits > 0) を1とする
    - 1列の場合: そのまま0/1とみなす
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] == 2:
        ci_bin = (df.iloc[:, 0] > 0).astype(int)
    elif df.shape[1] == 1:
        ci_bin = df.iloc[:, 0].astype(int)
    else:
        raise ValueError(f"Unexpected CI result shape {df.shape} for file: {path}")

    if expected_len is not None and len(ci_bin) != expected_len:
        raise ValueError(f"Length mismatch for Ci: got {len(ci_bin)} vs expected {expected_len}")

    return ci_bin.reset_index(drop=True)

def _load_di_binary(path, expected_len=None):
    """
    Diの出力（Pred/Conf）を読み込む。1列0/1を想定。
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 1:
        raise ValueError(f"Unexpected DI result shape {df.shape} for file: {path}")
    s = df.iloc[:, 0].astype(int)
    if expected_len is not None and len(s) != expected_len:
        raise ValueError(f"Length mismatch for Di: got {len(s)} vs expected {expected_len}")
    return s.reset_index(drop=True)

class MixAttack(AttackBase):
    def __init__(self, path_to_Ci_result, path_to_Di_result_pred, path_to_Di_result_conf, ai_len=None):
        super().__init__()
        self.path_to_Ci_result = path_to_Ci_result
        self.path_to_Di_result_pred = path_to_Di_result_pred
        self.path_to_Di_result_conf = path_to_Di_result_conf
        self.ai_len = ai_len  # 任意：長さ検証用
        self.limit = limit  # 追加

    def infer(self):
        ci_bin = _load_ci_binary(self.path_to_Ci_result, expected_len=self.ai_len)
        exp = len(ci_bin)
        di_pred = _load_di_binary(self.path_to_Di_result_pred, expected_len=exp)
        di_conf = _load_di_binary(self.path_to_Di_result_conf, expected_len=exp)

        # 多数決スコア（0～3）
        votes = ci_bin.astype(int) + di_pred.astype(int) + di_conf.astype(int)

        if self.limit is not None and self.limit > 0:
            # 上位 limit 件を 1 にする、それ以外は 0
            order = np.argsort(-votes.values)  # 大きい順
            sel = np.zeros(exp, dtype=int)
            topk = min(self.limit, exp)
            sel[order[:topk]] = 1
            inferred = sel
            print(f"[MixAttack] top-{topk} selected (limit={self.limit})")
        else:
            # 通常の多数決（>=2）
            inferred = (votes >= 2).astype(int)

        self.inferred = pd.DataFrame(inferred)
        n1 = int(self.inferred.sum().iloc[0])
        print(f"[MixAttack] result: selected={n1}/{exp}")
        return self.inferred

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mix Ci/Di attack votes to infer membership on Ai")
    ap.add_argument("--Ai_csv", default=None, help="(optional) Ai.csv for length validation only")
    ap.add_argument("path_to_Ci_result", help="Ci_inferred.csv (1col or 2cols[khn_hits,min_dist])")
    ap.add_argument("path_to_Di_result_pred", help="inferred_membership1_i.csv (Pred; 1 col)")
    ap.add_argument("path_to_Di_result_conf", help="inferred_membership2_i.csv (Conf; 1 col)")
    ap.add_argument("-o", "--output", help="path_to_output", default="Fij.csv")
    ap.add_argument("-l", "--limit", type=int, default=None, help="limit top-k rows to 1 (e.g., 10000)")  # 追加
    args = ap.parse_args()

    ai_len = None
    if args.Ai_csv:
        try:
            ai_len = sum(1 for _ in open(args.Ai_csv, "r", encoding="utf-8")) - 1
            if ai_len < 0:
                ai_len = None
        except Exception as e:
            print(f"[Warn] Could not read Ai_csv: {e}")

    attacker = MixAttack(args.path_to_Ci_result,
                         args.path_to_Di_result_pred,
                         args.path_to_Di_result_conf,
                         ai_len=ai_len,
                         limit=args.limit)
    attacker.infer()
    attacker.save_inferred(args.output)

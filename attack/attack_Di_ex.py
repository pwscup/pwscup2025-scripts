from abc import ABC
import sys, os
import argparse

import xgboost as xgb
import pandas as pd
import numpy as np


# モジュールの相対参照制限を強制的に回避
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'analysis'))
from xgbt_train import build_X

TARGET = "stroke_flag"
NUM_FEATURES = 21
NUM_CLASSES = 2

"""
Usage examples:
    # Minimum run: Pred=threshold 0.5 (default), Conf=threshold 0.1 (default)
    python attack_Di_ex.py D15.json A15.csv

    # Apply a common threshold of 0.5 to both Pred and Conf
    python attack_Di_ex.py D15.json A15.csv --threshold 0.5

    # Use different thresholds: Pred=0.4, Conf=0.08
    python attack_Di_ex.py D15.json A15.csv --pred-threshold 0.4 --conf-threshold 0.08

    # Pred selects exactly top-10,000 rows, Conf selects 10% of rows
    python attack_Di_ex.py D15.json A15.csv --pred-topk 10000 --conf-pos-ratio 0.10

    # Both Pred and Conf select top-10,000 rows
    python attack_Di_ex.py D15.json A15.csv --pred-topk 10000 --conf-topk 10000

    # Change output file names
    python attack_Di_ex.py D15.json A15.csv --out-pred pred_15.csv --out-conf conf_15.csv

Options:
    model_json            Path to trained XGBoost model in JSON format (e.g., D15.json)
    Ai_csv                Path to Ai data CSV (e.g., A15.csv)

    --threshold T         (Legacy) Apply the same threshold T to both Pred and Conf.
                          Ignored if specific thresholds or topk/ratio are provided.

    --pred-threshold TP   Threshold for Pred_Attack (default=0.5).
                          Only used if --pred-topk/--pred-pos-ratio are not set.
    --pred-topk K         Pred_Attack: select exactly K rows as members (overrides threshold).
    --pred-pos-ratio R    Pred_Attack: select round(R*N) rows as members (when topk not set).

    --conf-threshold TC   Threshold for Conf_Attack (default=0.1, |p - y| <= TC).
                          Only used if --conf-topk/--conf-pos-ratio are not set.
    --conf-topk K         Conf_Attack: select exactly K rows with smallest |p - y|.
    --conf-pos-ratio R    Conf_Attack: select round(R*N) rows (when topk not set).

    --out-pred PATH       Output file for Pred_Attack results
                          (default: inferred_membership1_ex.csv)
    --out-conf PATH       Output file for Conf_Attack results
                          (default: inferred_membership2_ex.csv)

Notes:
    - Priority order within each attack: topk > pos_ratio > threshold.
    - Outputs are 1-column CSV files (0/1 values, no header, no index).
    - Features are automatically reindexed to match model.feature_names;
      missing columns are filled with zeros.
"""

class Attack_Di_Base(ABC):
    def __init__(self, path_to_xgbt_model_json):
        """
        攻撃者の初期化

        path_to_xgbt_model_json: 学習済みのxgboostモデルのjsonファイルへのパス
        """
        # json fileを読み込み
        xgbt_model = xgb.Booster()
        xgbt_model.load_model(path_to_xgbt_model_json)

        self.xgbt_model = xgbt_model

        self.X = None
        self.y = None
        self.inferred = None
    
    def infer(self, path_to_Ai_csv):
        Ai_df = pd.read_csv(path_to_Ai_csv, dtype=str, 
                            keep_default_na=False)
        
        # 説明変数と目的変数に分割
        X = build_X(Ai_df, TARGET)

        # Xのみにある列は削除する, 9/10追記
        columns_only_X = set(X.columns) - set(self.xgbt_model.feature_names)
        if columns_only_X:
            X = X.drop(columns=columns_only_X)

        # xgbt_model.feature_namesのみにある列は0埋め, 9/10追記
        columns_only_feature_names = set(self.xgbt_model.feature_names) - set(X.columns)
        if columns_only_feature_names:
            for col in columns_only_feature_names:
                # 0で埋める
                X[col] = 0

        # Xの列をXGBoostモデルが要求する順番に並び替え, 9/10追記
        X = X.reindex(columns=self.xgbt_model.feature_names)

        X.columns = self.xgbt_model.feature_names
        self.X = X.copy()
        self.y = pd.to_numeric(Ai_df[TARGET], errors="coerce").astype(int).values

        # print(set(self.xgbt_model.feature_names)-set(X.columns.tolist()))

        return None
    
    def save_inferred(self, path_to_output):
        if self.inferred is None:
            print("inferred is None. No file was saved.")
        else:
            self.inferred.to_csv(path_to_output, index=False, header=False)
            print("inferred was successfully saved.")

        
class Pred_Attack(Attack_Di_Base):
    """
    予測が正解した行を優先しつつ、上位K件（または割合）だけを 1 にする版
    - topk:      ちょうどK件を1にする（推奨）
    - pos_ratio: 全体に対する割合で1件数を指定（topk未指定のときのみ有効）
    - threshold: 従来どおりの0.5丸め→正誤フラグ（topk/pos_ratio未指定時にフォールバック）
    """
    def __init__(self, path_to_xgboost_model_json, threshold=0.5, topk=None, pos_ratio=None):
        super().__init__(path_to_xgboost_model_json)
        self.threshold = float(threshold)
        self.topk = topk
        self.pos_ratio = pos_ratio

    def infer(self, path_to_Ai_csv):
        super().infer(path_to_Ai_csv)

        # 予測確率
        p = self.xgbt_model.predict(xgb.DMatrix(self.X))
        y = self.y

        # 従来方式（しきい値で丸め→正誤）へフォールバック
        if self.topk is None and self.pos_ratio is None:
            pred01 = (p >= self.threshold).astype(int)
            inferred = (pred01 == y).astype(int)
            self.inferred = pd.DataFrame(inferred)
            return self.inferred

        # 1) マージン（正解かつ自信が高いほど大）
        margin = 1.0 - np.abs(p - y)  # [0,1]

        # 2) 正解/不正解マスク
        correct_mask = ((p >= self.threshold).astype(int) == y)

        n = len(y)
        # 目標件数Kを決定
        if self.topk is not None:
            K = int(self.topk)
        else:
            r = float(self.pos_ratio)
            r = min(max(r, 0.0), 1.0)
            K = int(round(r * n))
        K = max(0, min(K, n))

        sel = np.zeros(n, dtype=bool)
        if K > 0:
            # 3) まず正解の中から margin 大きい順に min(K, #correct) 件
            idx_correct = np.flatnonzero(correct_mask)
            kc = min(K, idx_correct.size)
            if kc > 0:
                # margin大 → 昇順ではなく降順取りたいので -margin でargpartition
                part_c = np.argpartition(-margin[idx_correct], kc - 1)[:kc]
                sel[idx_correct[part_c]] = True

            # 4) 足りなければ不正解から補完
            rem = K - sel.sum()
            if rem > 0:
                idx_incorrect = np.flatnonzero(~correct_mask & ~sel)
                if idx_incorrect.size > 0:
                    ki = min(rem, idx_incorrect.size)
                    part_i = np.argpartition(-margin[idx_incorrect], ki - 1)[:ki]
                    sel[idx_incorrect[part_i]] = True

        self.inferred = pd.DataFrame(sel.astype(int))
        return self.inferred

class Conf_Attack(Attack_Di_Base):
    """
    モデルが確信を持って正答した行をmemberと推定する
    - threshold: 旧来どおりの |p - y| 閾値方式
    - topk:      |p - y| が小さい順にちょうど K 件 True にする（推奨）
    - pos_ratio: データ数に対する割合で指定（topk未指定のときのみ適用）
    """
    def __init__(self, path_to_xgboost_model_json, threshold=0.1, topk=None, pos_ratio=None, random_tie_break=False, seed=0):
        super().__init__(path_to_xgboost_model_json)
        self.threshold = float(threshold)
        self.topk = topk
        self.pos_ratio = pos_ratio
        self.random_tie_break = random_tie_break
        self.seed = seed

    def infer(self, path_to_Ai_csv):
        super().infer(path_to_Ai_csv)

        # 予測確率（0..1）
        pred = self.xgbt_model.predict(xgb.DMatrix(self.X))
        # スコア = |p - y|
        score = np.abs(pred - self.y)  # ndarray shape (n,)

        n = score.shape[0]

        # --- K件指定の優先ロジック ---
        k = None
        if self.topk is not None:
            k = int(self.topk)
        elif self.pos_ratio is not None:
            # 0..1の割合で上限下限をクリップ
            r = float(self.pos_ratio)
            r = min(max(r, 0.0), 1.0)
            k = int(round(r * n))

        if k is not None:
            k = max(0, min(k, n))  # 範囲ガード
            mask = np.zeros(n, dtype=bool)
            if k > 0:
                # 距離が小さい順にK件を取得（安くて速い）
                # 同値が多い場合でも厳密に「K件」を選ぶため、閾値ではなくインデックスで切る
                idx = np.argpartition(score, k - 1)[:k]

                if self.random_tie_break:
                    # 同値の中の選抜をランダム化したい場合のみ
                    rng = np.random.default_rng(self.seed)
                    # k件ぶんをシャッフル（順位に意味をもたせない）
                    rng.shuffle(idx)

                mask[idx] = True

            # DataFrame(1列)で返す（既存仕様に合わせる）
            self.inferred = pd.DataFrame(mask.astype(int))
            return self.inferred

        # --- 従来のthreshold方式（fallback） ---
        inferred = (score <= self.threshold)
        self.inferred = pd.DataFrame(inferred.astype(int))
        return self.inferred


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run Pred/Conf attacks on Ai with an XGBoost model.")
    ap.add_argument("model_json", help="trained model JSON (Booster.save_model)")
    ap.add_argument("Ai_csv", help="Ai.csv to attack")

    # 後方互換: 共通threshold（指定時のみ両者へ反映）
    ap.add_argument("--threshold", type=float, default=None,
                    help="(legacy) common threshold for both Pred and Conf")

    # Pred用しきい値／件数指定
    ap.add_argument("--pred-threshold", type=float, default=0.5,
                    help="threshold for Pred_Attack (default=0.5)")
    ap.add_argument("--pred-topk", type=int, default=None,
                    help="if set, select exactly top-K rows for Pred_Attack (overrides threshold)")
    ap.add_argument("--pred-pos-ratio", type=float, default=None,
                    help="if set, select given ratio of rows for Pred_Attack (used when topk is None)")

    # Conf用しきい値／件数指定
    ap.add_argument("--conf-threshold", type=float, default=0.1,
                    help="threshold for Conf_Attack (|p - y| <= threshold) (default=0.1)")
    ap.add_argument("--conf-topk", type=int, default=None,
                    help="if set, select exactly top-K rows (smallest |p - y|) for Conf_Attack")
    ap.add_argument("--conf-pos-ratio", type=float, default=None,
                    help="if set, select given ratio of rows for Conf_Attack (when conf-topk is None)")

    # 出力ファイル（必要に応じて変更可能）
    ap.add_argument("--out-pred", default="inferred_membership1_ex.csv",
                    help="output CSV for Pred_Attack (default: inferred_membership1_ex.csv)")
    ap.add_argument("--out-conf", default="inferred_membership2_ex.csv",
                    help="output CSV for Conf_Attack (default: inferred_membership2_ex.csv)")

    args = ap.parse_args()

    # 後方互換: --threshold が与えられたら両者へ適用（topk/ratio指定が無いときに使われる）
    if args.threshold is not None:
        args.pred_threshold = args.threshold
        args.conf_threshold = args.threshold

    # ---- Pred_Attack ----
    attacker = Pred_Attack(
        args.model_json,
        threshold=args.pred_threshold,
        topk=args.pred_topk,
        pos_ratio=args.pred_pos_ratio
    )
    _ = attacker.infer(args.Ai_csv)
    attacker.save_inferred(args.out_pred)

    # ---- Conf_Attack ----
    attacker = Conf_Attack(
        args.model_json,
        threshold=args.conf_threshold,
        topk=args.conf_topk,
        pos_ratio=args.conf_pos_ratio
    )
    _ = attacker.infer(args.Ai_csv)
    attacker.save_inferred(args.out_conf)

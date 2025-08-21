from abc import ABC
import sys, os
import argparse

import xgboost as xgb
import pandas as pd


# モジュールの相対参照制限を強制的に回避
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'analysis'))
from xgbt_train import build_X

TARGET = "stroke_flag"
NUM_FEATURES = 21
NUM_CLASSES = 2


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
    モデルが正答した行をmemberと推定する
    """
    def __init__(self, path_to_xgboost_model_json):
        super().__init__(path_to_xgboost_model_json)

    def infer(self, path_to_Ai_csv):
        super().infer(path_to_Ai_csv)

        pred = self.xgbt_model.predict(xgb.DMatrix(self.X))
        pred[pred<0.5] = 0
        pred[pred>=0.5] = 1
        
        inferred = pd.DataFrame(pred == self.y, dtype=int)
        self.inferred = inferred

        return inferred

class Conf_Attack(Attack_Di_Base):
    """
    モデルが確信を持って正答した行をmemberと推定する
    """
    def __init__(self, path_to_xgboost_model_json, threshold=0.1):
        super().__init__(path_to_xgboost_model_json)
        self.threshold = threshold

    def infer(self, path_to_Ai_csv):
        super().infer(path_to_Ai_csv)

        pred = self.xgbt_model.predict(xgb.DMatrix(self.X))
        confidence = pd.DataFrame(pred-self.y).abs()
        inferred = (confidence <= self.threshold)
        self.inferred = inferred

        return inferred


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("model_json", help="trained model JSON (Booster.save_model)")
    ap.add_argument("Ai_csv", help="Ai.csv to attack")
    args = ap.parse_args()

    attacker = Pred_Attack(args.model_json)
    pred = attacker.infer(args.Ai_csv)
    attacker.save_inferred("inferred_membership1.csv")

    attacker = Conf_Attack(args.model_json)
    pred = attacker.infer(args.Ai_csv)
    attacker.save_inferred("inferred_membership2.csv")

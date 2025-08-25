from abc import ABC, abstractmethod
import sys, os
import json
import argparse

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

# モジュールの相対参照制限を強制的に回避
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'analysis'))
from xgbt_train import build_X


TARGET = "stroke_flag"

def df_to_Xy(df):
    X = build_X(df, TARGET)
    # X = xgb.DMatrix(X)
    y = pd.to_numeric(df[TARGET], errors="coerce").astype(int) # .values

    return X, y

class DiGenBase(ABC):
    def __init__(self, n_estimators=600, max_depth=6, learning_rate=0.05, 
                subsample=0.9, colsample_bytree=0.9, early_stopping_rounds=50, 
                random_state=42, n_jobs=-1):
        # model 初期化
        # See https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
        Di = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",                          # 決定木アルゴリズム
            n_estimators=n_estimators,                   # 決定木の数
            max_depth=max_depth,                         # 決定木の最大深さ
            learning_rate=learning_rate,                 # learning rate(学習率)の設定
            subsample=subsample,                         # 各決定木の学習で使う訓練データの割合
            colsample_bytree=colsample_bytree,           # 各決定木が使用する特徴量の割合
            early_stopping_rounds=early_stopping_rounds, # ← 警告回避
            random_state=random_state,                   # 乱数のseed値
            n_jobs=n_jobs, # CPUコア使用数。3とすれば3コアだけ使用。-1やNoneで全部使用。
        )
        
        # 初期化したモデルをselfに格納
        self.Di = Di
        self.feature_names = None
        
    @abstractmethod
    def fit(self,path_to_Bi_csv, path_to_Ci_csv):
        pass

    def save_Di(self, out:str):
        booster = self.Di.get_booster()
        # JSONを後編集せず、属性を埋めて「そのまま」保存
        booster.set_attr(feature_names=json.dumps(self.feature_names, ensure_ascii=False))
        booster.set_attr(target=TARGET)
        booster.set_attr(xgboost_version=xgb.__version__)
        booster.save_model(out)

    def eval_accuracy(self, path_to_test_data):
        test_data = pd.read_csv(path_to_test_data, dtype=str, keep_default_na=False)
        X_test, y_test = df_to_Xy(test_data)
        y_pred = self.Di.predict(X_test)
        accuracy = (y_pred==y_test).mean()

        return accuracy


class TrainDiFromBiCi(DiGenBase):
    def __init__(self):
        super().__init__()
        # print(None)

    def fit(self, path_to_Bi_csv, path_to_Ci_csv):
        Bi = pd.read_csv(path_to_Bi_csv, dtype=str, keep_default_na=False)
        Ci = pd.read_csv(path_to_Ci_csv, dtype=str, keep_default_na=False)
        
        # BiとCiを結合して訓練データを作る
        raw_data = pd.concat([Bi, Ci], ignore_index=True)

        # 行を半分だけサンプリングして訓練データを作成
        data = raw_data.sample(frac=0.5, ignore_index=True)

        # コンペの要件に合わせて、訓練データをXとyに分割
        X, y = df_to_Xy(data)

        # インデックスを取得
        shuffled_indices = X.index
        # 分割したい割合
        split_point = int(len(y) * 0.9)
        # インデックスを分割
        idx1 = shuffled_indices[:split_point]
        idx2 = shuffled_indices[split_point:]

        X_train = X.loc[idx1].reset_index(drop=True)
        X_val = X.loc[idx2].reset_index(drop=True)
        y_train = y.loc[idx1].reset_index(drop=True)
        y_val = y.loc[idx2].reset_index(drop=True)

        # モデル学習
        self.Di.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.feature_names = list(X.columns)

if __name__ == "__main__":
    # コマンドライン引数の読み込み
    argparser = argparse.ArgumentParser(description="BiとCiからDiを学習するサンプルコードです")
    argparser.add_argument("path_to_Bi_csv", help="Path to Bi to learn")
    argparser.add_argument("path_to_Ci_csv", help="Path to Ci to learn")
    args = argparser.parse_args()

    generator = TrainDiFromBiCi()
    generator.fit(args.path_to_Bi_csv, args.path_to_Ci_csv)
    acc = generator.eval_accuracy(args.path_to_Bi_csv)
    print(f"accuracy: {acc}")
    generator.save_Di("sample_Di.json")
    print("a Di.json example was saved as sample_Di.json")

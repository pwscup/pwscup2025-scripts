# xgboost>=2, scikit-learn
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import joblib, json

# 0) ユーティリティ：B準拠エンコーダ
def fit_schema_encoder(dfB, num_cols, cat_cols):
    enc = OneHotEncoder(handle_unknown='ignore', sparse=True)
    enc.fit(dfB[cat_cols])
    # 数値の処理（任意）：winsorizeやビン境界をここで保存してもよい
    return enc

def transform_with_schema(df, enc, num_cols, cat_cols):
    X_num = df[num_cols].to_numpy(dtype=float)
    X_cat = enc.transform(df[cat_cols])  # sparse
    # 数値＋one-hotの横結合（scipy.sparse で結合推奨）
    from scipy.sparse import csr_matrix, hstack
    X = hstack([csr_matrix(X_num), X_cat]).tocsr()
    return X

# 1) データ読み込み（例）
# dfB: 学習基準（B）
# dfS: 事前学習（S, B準拠に整形済み）
# 目的変数は 'stroke_flag'
# num_cols, cat_cols は B から確定させること
# dfB, dfS, num_cols, cat_cols を用意済みと仮定
yB = dfB['stroke_flag'].astype(int).to_numpy()
yS = dfS['stroke_flag'].astype(int).to_numpy()

encoder = fit_schema_encoder(dfB, num_cols, cat_cols)
XB = transform_with_schema(dfB, encoder, num_cols, cat_cols)
XS = transform_with_schema(dfS, encoder, num_cols, cat_cols)

# 2) 重要度重み（任意）：S→Bのimportance weighting
# 簡易実装：ロジ回帰で domain 判別 → w(x) = P(B|x)/P(S|x)
def compute_domain_weights(XB, XS):
    from scipy.sparse import vstack
    X = vstack([XB, XS])
    y = np.r_[np.ones(XB.shape[0], dtype=int), np.zeros(XS.shape[0], dtype=int)]
    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    clf.fit(X, y)
    pB = clf.predict_proba(XS)[:,1]
    eps = 1e-3
    w = np.clip(pB / np.clip(1-pB, eps, 1), 0.2, 5.0)  # クリップ安定化
    return w

wS = compute_domain_weights(XB, XS)

# 3) 事前学習（S）
#   - Sを学習、Bのvalidで早期終了（Bの一部をvalidに使う）
XBt, XBv, yBt, yBv = train_test_split(XB, yB, test_size=0.2, stratify=yB, random_state=42)

params_pre = dict(
    objective='binary:logistic',
    tree_method='hist',
    eval_metric='logloss',
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    gamma=0.0,
    n_estimators=4000,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=(1-yB.mean())/yB.mean()  # Bの陽性率で近似
)

model_pre = xgb.XGBClassifier(**params_pre)
eval_set = [(XBt, yBt), (XBv, yBv)]
model_pre.fit(
    XS, yS,
    sample_weight=wS,
    eval_set=eval_set,
    verbose=False,
    early_stopping_rounds=200
)

# 4) Bでfine-tune（追加学習）
params_ft = params_pre.copy()
params_ft.update(dict(learning_rate=0.03))  # 小さく
model_ft = xgb.XGBClassifier(**params_ft)

# 事前学習Boosterを渡して追加学習
model_ft.fit(
    XB, yB,
    xgb_model=model_pre.get_booster(),
    eval_set=[(XBt, yBt), (XBv, yBv)],
    verbose=False,
    early_stopping_rounds=100
)

# 5) キャリブレーション（Platt または Isotonic）
#   - Bのholdout (XBv, yBv) でキャリブレータをfit
p_val = model_ft.predict_proba(XBv)[:,1]

# Platt
platt = LogisticRegression(max_iter=200)
platt.fit(p_val.reshape(-1,1), yBv)

# 閾値最適化：Accuracy最大（Bのholdoutで固定）
def best_threshold(p, y):
    ths = np.linspace(0.05, 0.95, 181)
    accs = [(t, accuracy_score(y, (p>=t).astype(int))) for t in ths]
    return max(accs, key=lambda z: z[1])

p_val_cal = platt.predict_proba(p_val.reshape(-1,1))[:,1]
thr, acc = best_threshold(p_val_cal, yBv)

# 6) 擬似XXでの一致数確認（例）
# XG = 5k(B未使用) + 5k(S or A*) を事前に用意しておく想定
# XG, yG = ...
# pG = model_ft.predict_proba(XG)[:,1]
# pG_cal = platt.predict_proba(pG.reshape(-1,1))[:,1]
# yhat = (pG_cal >= thr).astype(int)
# match = (yhat == yG).sum()
# print("Pseudo-XX matches:", match)

# 7) エクスポート（モデル本体＋前処理＋キャリブ＋閾値）
artifact = {
    "model": "xgboost",
    "booster_json": json.loads(model_ft.get_booster().save_raw(raw_format='json')),
    "encoder": joblib.dumps(encoder).decode('latin1'),
    "calibration": {
        "type": "platt",
        "coef_": platt.coef_.tolist(),
        "intercept_": platt.intercept_.tolist()
    },
    "threshold": float(thr),
    "schema": {
        "num_cols": num_cols,
        "cat_cols": cat_cols
    },
    "meta": {"random_state": 42}
}
with open("xgb_stroke_export.json", "w", encoding="utf-8") as f:
    json.dump(artifact, f, ensure_ascii=False)

# `stats.py` : 要約統計量
- CSV 形式の医療データを入力として、単一変数および二変数間の各種要約統計量を出力。
- 使い方の例
  - `python3 stats.py HI_10K.csv`
- 入力：ヘッダー付き CSV ファイル
    - 実行：`python3 stats.py <input.csv>` 
- 出力（標準出力）：
  1. 数値列の統計（min–max 正規化後）：平均、標準偏差、四分位数
  2. 数値×数値の相関行列（Pearson）
  3. 各カテゴリ列の集計値の比率（ratio）
  4. カテゴリ×数値の要約統計（min–max 正規化後）：カテゴリの値毎の数値列の平均、標準偏差、四分位数（Group By）
  5. カテゴリ×カテゴリのクロス集計の値の比率（ratio）
- `AGE` を `\[0–17, 18–44, 45–64, 65–74, 75+\]` の固定ビンで再生成してカテゴリ列 `AGE_GROUP` を新たに作成

# `LR_asthma.py` : 喘息リスク因子のロジスティック回帰
- CSV 形式の医療データを入力として、二値目的変数（既定：`asthma_flag`）に対してロジスティック回帰を適用し、係数や信頼区間由来の指標を0〜1に正規化して出力。多重共線性の強さを示す VIF を正規化した値も出力。出力行の個数・順序を入力データに依存させず一定に保つため、実際にモデルに入らなかった項目も値をNaNとして出力。
- 使い方の例
  - `python3 LR_asthma.py HI_10K.csv`
  - `python3 LR_asthma.py HI_10K.csv --ensure-terms "ETHNICITY_hispanic,GENDER_M,RACE_black"` # 必ず載せたい term を追加
- 入力： ヘッダー付き CSV ファイル。デフォルトの目的変数は 0/1 の asthma_flag（--target で変更可）
    - 実行：`python3 LR_asthma.py <input.csv> \[--target TARGET\] \[--test-size TEST_SIZE\] \[--random-state RANDOM_STATE\] \[--ensure-terms ENSURE_TERMS\]` 
- 出力（標準出力）：
    1. AUC（holdout）：ホールドアウト検証での AUC
    2. 単一テーブル：`term, coef, p_value, OR_norm, CI_low_norm, CI_high_norm, VIF_norm`
        - `term`：説明変数名（const は除外）
        - `coef`：ロジスティック回帰の回帰係数
        - `p_value`：係数の有意性
        - `OR_norm`：オッズ比 OR を OR/(1+OR) に変換した 0〜1 値
        - `CI_low_norm`：95%信頼区間の下限値 CI_low を CI_low/(1+CI_low) に変換した 0～1 値
        - `CI_high_norm`：95%信頼区間の上限値 CI_high を CI_high/(1+CI_high) に変換した 0～1 値
        - `VIF_norm`：1 - 1/max(VIF,1) により 0〜1 化（1 に近いほど多重共線性が強い）
- 解析の流れ
  1. 読み込み・検査：CSV を文字列優先で読み込み、target 列が厳密な 0/1 であることを検証
  2. 前処理：
      - 数値列とカテゴリ列で分離
      - カテゴリ列は get_dummies(drop_first=True) でダミー変数化（多重共線性を抑制）
      - 無限値/全欠損列を除去し、中央値補完、ゼロ分散列の削除
      - 列名の昇順で並べ、以降の表出力の順序を固定
  3. 学習・評価：
      - 学習/検証に分割（層化、既定 80/20）
      - statsmodels の GLM（Binomial）でロジスティック回帰を学習し、検証で AUC を算出
  4. 統計量の作成：
      - 係数・p 値・信頼区間から OR_norm / CI_low_norm / CI_high_norm を計算（いずれも 0〜1）
      - 説明変数間の多重共線性評価として VIF を算出し、VIF_norm のみを出力
  5. 固定スキーマ出力：
      - 実際に学習で使われた列（=基底スキーマ）に、--ensure-terms で指定した列名群を和集合で追加
      - 最終的な term 一覧を辞書順で固定し、存在しない列は NaN で占位して表を生成
          - これにより、行数と並びが常に一定となる

# `KW_IND.py` : 年齢群間における各医療指標の分布差のKruskal-Wallis検定
- CSV形式の医療データを入力として、年齢を臨床的なカスタム区切り（ビン）で群分けし、各医療指標（例：`encounter_count`, `num_medications` など）についてKruskal–Wallis検定を行い、統計量を入力スケールに依存しない 0〜1 指標へ整形して出力。大規模データや極端なp値でも値が飽和しにくいよう、数値安定化と滑らかな正規化を行っている。
- 使い方の例
  - `python3 KW_IND.py HI_10K.csv`
  - `python3 KW_IND.py HI_10K.csv --p-norm arctan --p-scale 30` # より緩やかな飽和（有意が多いときの分離改善）
  - `python3 KW_IND.py HI_10K.csv --custom-bins "0,18,65,200"` # ビンを変更（小児・成人・高齢の3群）
- 目的と特徴
  - ノンパラメトリック検定（Kruskal–Wallis）で、年齢群間の分布差を評価
  - 統計量は 0〜1 の指標を提示（コンテストの都合上）：
      - H_norm（Kruskal-Wallis検定の統計量 H の規格化指標）
      - minus_log10_p_norm（p 値から導いた強さを滑らかに 0〜1 化）
      - 効果量：epsilon2, eta2_kw, rank_eta2, さらに群間ペアの優越確率に基づく A_pair_avg と差の非対称性 A_pair_sym
      - 数値安定化（chi2_logp_safe）により、かなり小さい p 値でも NaN を回避
- 入力： ヘッダー付き CSV ファイル
    - 実行：`python3 KW_IND.py <input.csv> \[--age-col AGE_COL\] \[--metrics METRICS\] \[--custom-bins CUSTOM_BINS\] \[--min-per-group MIN_PER_GROUP\] \[--p-norm {arctan,exp,log1p}\] \[--p-scale P_SCALE\] \[--p-cap P_CAP\]`
        - `--age-col`（既定 AGE）：年齢列名
        - `--metrics`：解析する数値列（カンマ区切り、既定は代表的 5 指標）
        - `--custom-bins`：年齢の区切り（例 0,18,45,65,75,200）。未指定は既定ビン
        - `--min-per-group`：各群の最小サンプル数（既定 2、満たさない群は除外）
        - `--p-norm`：p 由来指標の正規化方法（arctan/exp/log1p、既定 arctan）
        - `--p-scale, --p-cap`：正規化の形状調整用パラメータ
- 出力（標準出力）：
    1. `metric`：指標名
    2. `group_sizes`：利用群のサイズ
    3. `H_norm`：0〜1
    4. `minus_log10_p_norm`：0〜1
    5. `epsilon2`：0〜1
    6. `eta2_kw`：0〜1
    7. `rank_eta2`：0〜1
    8. `A_pair_avg`：0〜1（0.5 中立）
    9. `A_pair_sym`：0〜1（群間差の左右非対称性）
- 解析の流れ
  1. 年齢のビニング：`--custom-bins` で与えた左閉右開区間に年齢を割当て。未指定時は既定の臨床区切りを使用
  2. 前処理：各メトリクス列を数値化し、`min_per_group` 未満の群は解析から外す
  3. Kruskal–Wallis 検定：利用可能な群で H を計算
  4. p 値の数値安定化：H と自由度から `logsf` を基本に、Wilson–Hilferty 近似や正規漸近で**安定な `-log10(p)`**を得る
  5. 正規化：
      - `H_norm`：理論最大 H（完全分離・同点なし）で割って 0〜1 化
      - `minus_log10_p_norm`：`arctan/exp/log1p` の滑らかな飽和関数で 0〜1 に写像
  6. 効果量：
      - `epsilon2`：H の補正に基づく 0〜1 指標
      - `eta2_kw`：Kruskal-Wallis検定に対応した η² 近似
      - `rank_eta2`：全体順位化 → 一元 ANOVA の η²（SSB/SST）
      - `A_pair_avg`/`A_pair_sym`：全 i<j ペアの Vargha–Delaney A をサイズ重みで要約

# `xgbt_train.py` : XGBoostで二値目的変数を学習
- CSV形式の医療データを入力として、説明変数を自動整形してから XGBoost（分類）で二値目的変数を学習し、学習済みモデルをJSONファイルとして保存。
- 使い方の例
  - `python3 xgbt_train.py HI_10K.csv --model-json model.json`
  - `python3 xgbt_train.py HI_10K.csv --model-json model.json --target obesity_flag --seed 123 --n-estimators 600 --max-depth 5 --learning-rate 0.06 --subsample 0.8 --colsample-bytree 0.8 --early-stopping-rounds 40` # 目的変数やハイパーパラメータを変更
- 入力： ヘッダー付き CSV ファイル
    - 実行：`python3 xgbt_train.py <input.csv> --model-json <output.json> \[--target TARGET\] \[--seed SEED\] \[--test-size TEST-SIZE\] \[--n-estimators N-EST\] \[--max-depth MAX-DEPTH\] \[--learning-rate LEARNING-RATE\] \[--subsample SUBSAMPLE\] \[--colsample-bytree COLS-BYTREE\] \[--early-stopping-rounds EARLY-STOPPING-ROUNDS\]
        - `--target`：目的変数（既定 stroke_flag、0/1 必須）
        - `--seed`：乱数シード
        - `--test-size`：検証データ比率（既定値 0.1）
        - `--n-estimators`：学習ハイパーパラメータ（既定値 600）
        - `--max-depth`：学習ハイパーパラメータ（既定値 6）
        - `--learning-rate`：学習ハイパーパラメータ（既定値 0.05）
        - `--subsample`：学習ハイパーパラメータ（既定値 0.9）
        - `--colsample_bytree`：学習ハイパーパラメータ（既定値 0.9）
        - `--early-stopping-rounds`：学習ハイパーパラメータ（既定値 50）
- 前処理
    1. 読み込み：`input.csv` を文字列として読み込み、`--traget`（既定：`stroke_flag`）が0/1の二値になっているかチェック
    2. 目的変数を除外し、説明変数を抽出
    3. 各列を `pd.to_numeric(..., errors="coerce")` で数値化できるかチェックし、数値化できたものを数値列として採用
    4. 残りはカテゴリ列とみなし、`get_dummies(drop_first=True)` でOne-Hot化
    5. ゼロ分散列の除外（全値が同じなど、学習に寄与しない列を削除）
    6. 列名を照準に固定し、`float32` 化 
- データ分割と学習
    - `train_test_split`（層化 `stratify=y`）で 学習/検証 = 90/10（既定）
    - モデルは `xgb.XGBClassifier`：
        - `objective="binary:logistic"`（確率出力）
        - `eval_metric="logloss"`
        - `tree_method="hist"`
        - 主要なハイパーパラメータは引数で変更可能（既定：`n_estimators=600`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`）
    - 早期終了はコンストラクタで `early_stopping_rounds=50` を指定し、`fit` の `eval_set=\[(X_val, y_val)\] を監視
        - 改善が止まると停止
- 学習後の評価と出力（ファイル保存）：
    - 検証セットで `predict_proba` を 0.5 で二値化し、Validation Accuracy を標準出力
    - `model.get_booster()` の結果を JSON で保存（--model-json）
    - attributes にメタ情報を埋め込み：
        - `feature_names`：学習に使った列名リスト（JSON 文字列）
        - `target`：目的変数名
- JSONファイル仕様（最小構成のサンプルファイル `sample_minimum_model.json` 参照）
    - 共通
        - 文字コード：UTF-8
        - 形式：1つのJSONオブジェクト（末尾カンマ不可）
        - トップレベルキー：`version`, `learner`, `attributes`
    - `version`
        - XGBoostのモデル形式バージョン。`save_model` が自動付与 
    - `attributes`
        - `feature_names`：学習に用いた列名のJSON文字列
            - 例：`"\[\"AGE\",\"GENDER_M\",\"num_medications\", ...\]"`
            - 配列に復元すると 長さ ≥ 1 / 文字列のみ / 重複なし / 列名昇順（sorted）
        - `target`: 目的変数名（例：`"stroke_flag"`、0/1 二値）
        - `xgboost_version`: 学習時の xgboost バージョン（例：`"2.0.3"`
    - `learner`
        - `learner.objective.name == "binary:logistic"`
        - `learner.gradient_booster.name == "gbtree"`
        - `learner.gradient_booster.model.trees`: 配列長 ≥ 1
        - `learner.learner_model_param.num_feature`: **数値（文字列可）**で、`len(attributes.feature_names を復元した配列)` と一致
    - 一貫性チェック（`validate_model_json.py` でチェック可能）
        1. `attributes.feature_names` を `json.loads()` で配列に復元できる
        2. 復元配列の長さと `learner.learner_model_param.num_feature` が一致
        3. `attributes.target` は非空文字列
        4. `version` は存在し、`xgbt_pred.py` 実行中の xgboost と比較可能（実行時チェックでモデル側が新しければエラー） 

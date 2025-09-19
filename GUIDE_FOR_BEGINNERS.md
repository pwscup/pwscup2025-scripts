# PWS Cup 2025 初心者向けガイド

## 概要
このガイドでは、PWS Cup 2025の加工フェーズにおけるハンズオン作業を初心者向けに説明します。ハワイ州のサンプルデータと各種サンプルスクリプトを使用して、匿名加工の実践的な体験を行います。

## 使用するファイルセット

### データファイル
- **HI_100K.csv**: データAi相当（100,000件のハワイ州患者データ）
- **HI_10k.csv**: データBi相当（10,000件のハワイ州患者データのサブセット）

### データ項目
両ファイルには以下の項目が含まれています：
- `GENDER`: 性別 (M/F)
- `AGE`: 年齢
- `RACE`: 人種 (asian, black, hawaiian, native, other, white)
- `ETHNICITY`: 民族 (hispanic/nonhispanic)
- `encounter_count`: 診察回数
- `num_procedures`: 処置数
- `num_medications`: 処方薬数
- `num_immunizations`: 予防接種数
- `num_allergies`: アレルギー数
- `num_devices`: デバイス数
- `asthma_flag`: 喘息フラグ (0/1)
- `stroke_flag`: 脳卒中フラグ (0/1)
- `obesity_flag`: 肥満フラグ (0/1)
- `depression_flag`: うつ病フラグ (0/1)
- `mean_systolic_bp`: 収縮期血圧平均
- `mean_diastolic_bp`: 拡張期血圧平均
- `mean_bmi`: BMI平均
- `mean_weight`: 体重平均

## 加工フェーズのハンズオン

### Step 1: 匿名化データ Ci の作成

サンプル匿名加工スクリプト ano.py を使ってHI_10k.csv を匿名加工して、Ci相当データを作成します。

```bash
python anonymization/ano.py data/HI_10k.csv HI_10k_anon.csv
```

#### ano.py による匿名加工の内容
- **カテゴリ項目**: GENDER, RACE, ETHNICITY を確率的に他の値に置換
- **年齢**: ±2歳のランダムノイズを追加（0-120歳でクリップ）
- **整数項目**: encounter_count, num_procedures等に範囲内でランダムノイズを追加
- **フラグ項目**: asthma_flag等を確率的に反転
- **実数項目**: 血圧、BMI、体重にランダムノイズを追加

#### Step 1-1: データ値域の確認

値域確認スクリプト check_csv.py を使い、生成した匿名データが適切な値域内にあることを確認します。

```bash
python util/check_csv.py HI_10k_anon.csv data/columns_range.json.json
```

##### 値域定義ファイル（columns_range.json）の内容
各項目について以下が定義されています：
- **type**: データタイプ（number/category/date）
- **min/max**: 数値項目の最小値・最大値
- **values**: カテゴリ項目の許可される値
- **max_decimal_places**: 小数点以下の最大桁数

#### Step 1-2: 有用性評価の実行

有用性評価スクリプト eval_all.py によるBiとCiの比較を行い、有用性を評価します。

```bash
python evaluation/eval_all.py data/HI_10k.csv HI_10k_anon.csv
```

##### 評価項目
1. **基本統計の差異** (stats_diff): 統計量の最大絶対誤差
2. **ロジスティック回帰での差異** (LR_asthma_diff): 喘息予測モデルの差異
3. **独立性検定での差異** (KW_IND_diff): カテゴリ変数間の独立性の差異

##### スコア計算
```
Ci_utility = 40 × (1 - stats_diff_max_abs) + 
             20 × (1 - LR_asthma_diff_max_abs) + 
             20 × (1 - KW_IND_diff_max_abs)
```
最大80点で評価されます。

### Step 2: 機械学習モデル Di の作成

機械学習モデルを構築します。
以下の2つのアプローチを取るサンプルスクリプトが利用できます。

#### 方法A: Ciのみを使用（標準的な方法）

機械学習モデル学習スクリプト xgbt_train.py を使い、匿名化データ（Ci）のみを使用してモデルを作成します。

```bash
python analysis/xgbt_train.py HI_10k_anon.csv --model-json HI_10k_anon.json
```

#### 方法B: BiとCiの混合データを使用（専用スクリプト）

元データ（Bi）と匿名化データ（Ci）を混合した学習データを作成するサンプルスクリプト gen_Di.py を使います。

```bash
python anonymization/gen_Di.py data/HI_10k.csv HI_10k_anon.csv
```

このスクリプトは自動的に `sample_Di.json` として結果を保存します。

#### 学習内容（共通）
- **アルゴリズム**: XGBoost（勾配ブースティング決定木）
- **目的変数**: stroke_flag（脳卒中フラグ）をデフォルトで使用
- **前処理**: 
  - 数値列とカテゴリ列を自動判定
  - カテゴリ列はone-hot encoding（drop_first=True）
  - ゼロ分散列の除去
  - 列名を昇順で固定

#### 方法Bの特徴
- BiとCiを結合後、50%をランダムサンプリングして学習データを作成
- より豊富なデータでの学習により、モデル性能が向上する可能性
- 反面、Biの一部が学習に使われるため、攻撃耐性が下がる可能性

#### Step 2-1: 学習データでの評価確認

機械学習モデル予測スクリプト xgbt_pred.py を使い、作成したモデルを学習データで評価してみます。

```bash
python analysis/xgbt_pred.py HI_10k_anon.json --target "stroke_flag" --test-csv HI_10k_anon.csv
```

#### Step 2-2: 大本データでの評価（参考）

同様に、大本のデータであるAiに対しても評価を試してみます。

```bash
python analysis/xgbt_pred.py HI_10k_anon.json --target "stroke_flag" --test-csv data/HI_100K.csv
```

**注意**: 実際の競技では、データAi（HI_100K.csv相当）は運営側のみが持つ情報のため、参加者はこの評価を実施できません。ここでは学習目的での参考実行となります。

## 攻撃フェーズのハンズオン

作成した匿名化データ（Ci）と機械学習モデル（Di）に対してメンバーシップ推論攻撃を実行します。

### 攻撃の概要
攻撃者は、元データAi（HI_100K.csv）のうち、どのレコードが学習データBi（HI_10k.csv）に含まれていたかを推測します。

### Step 1: 総合攻撃の実行

サンプル攻撃スクリプト attack_example.py を使い、CiとDiの両方を使った攻撃を実行します。

```bash
python attack/attack_example.py data/HI_100K.csv HI_10k_anon.csv HI_10k_anon.json -o attack_result.csv
```

### Step 2: 攻撃結果の評価

採点用コード check_ans.py を使い、攻撃結果を正解データと比較して攻撃性能を評価します。

```bash
python evaluation/check_ans.py attack_result.csv data/HI_ans.csv
```

### 個別攻撃スクリプト

より詳細な攻撃を試したい場合は、個別のスクリプト (attack_Ci.py および attack_Di.py) も利用できます：

**Ci（匿名化データ）への攻撃:**
```bash
python attack/attack_Ci.py data/HI_100K.csv HI_10k_anon.csv -o attack_ci_result.csv
```

**Di（機械学習モデル）への攻撃:**
```bash
python attack/attack_Di.py data/HI_100K.csv HI_10k_anon.json -o attack_di_result.csv
```

### 攻撃の仕組み
- **Ci攻撃**: 匿名化されたデータと元データの類似度を分析
- **Di攻撃**: 機械学習モデルの予測信頼度を分析
- **総合攻撃**: CiとDi両方の情報を統合して判定

## 実行例

```bash
# Step 1: 匿名化データ Ci の生成
python anonymization/ano.py data/HI_10k.csv HI_10k_anon.csv

# Step 1-1: データ値域の確認
python util/check_csv.py HI_10k_anon.csv data/columns_range.json

# Step 1-2: 有用性評価
python evaluation/eval_all.py data/HI_10k.csv HI_10k_anon.csv

# Step 2: 機械学習モデル Di の作成
# 方法A: Ciのみを使用
python analysis/xgbt_train.py HI_10k_anon.csv --model-json HI_10k_anon.json

# 方法B: BiとCiの混合データを使用（オプション）
python anonymization/gen_Di.py data/HI_10k.csv HI_10k_anon.csv

# Step 2-1: 学習データでの評価確認
python analysis/xgbt_pred.py HI_10k_anon.json --target "stroke_flag" --test-csv HI_10k_anon.csv

# Step 2-2: 大本データでの評価（参考）
python analysis/xgbt_pred.py HI_10k_anon.json --target "stroke_flag" --test-csv data/HI_100K.csv

# 攻撃フェーズのハンズオン
# Step 1: 総合攻撃の実行
python attack/attack_example.py data/HI_100K.csv HI_10k_anon.csv HI_10k_anon.json -o attack_result.csv

# Step 2: 攻撃結果の評価
python evaluation/check_ans.py attack_result.csv data/HI_ans.csv
```

## 期待される出力

### Step 1-1: 値域確認の結果
```
全ての値が JSON 仕様内です。
```

### Step 1-2: 有用性評価の結果例
```
stats_diff max_abs: 0.1234
LR_asthma_diff max_abs: 0.0567
KW_IND_diff max_abs: 0.0890
Ci utility: 65.42 / 80
```

### Step 2: 機械学習モデル学習の結果例

**方法A（xgbt_train.py）の結果:**
```
Validation Accuracy (threshold=0.5): 0.769000
Saved model JSON to: HI_10k_anon.json
#features: 21
```

**方法B（gen_Di.py）の結果:**
```
accuracy: 0.8911
a Di.json example was saved as sample_Di.json
```

### Step 2-1: 学習データ評価の結果例
```
Accuracy (test, threshold=0.500): 0.928000
```

### Step 2-2: 大本データ評価の結果例
```
Accuracy (test, threshold=0.500): 0.890000
```

### 攻撃フェーズ Step 1: 総合攻撃の結果例
```
inferred was successfully saved as attack_result.csv
```

### 攻撃フェーズ Step 2: 攻撃結果評価の結果例
```
49
66
74
77
...（攻撃が成功したと推定される行番号が表示される）...
TOTAL 2257
```
この例では、100,000件中2,257件をメンバーとして正しく特定できたことを示しています。

## ファイル構成

```
pwscup2025-scripts/
├── data/
│   ├── HI_100K.csv          # データAi相当
│   ├── HI_10k.csv           # データBi相当
│   └── columns_range.json   # 値域定義（本戦用）
├── anonymization/
│   ├── ano.py               # 匿名加工スクリプト
│   └── gen_Di.py            # BiとCi混合データでのDi作成スクリプト
├── util/
│   └── check_csv.py         # 値域確認スクリプト
├── evaluation/
│   └── eval_all.py          # 有用性評価スクリプト
├── analysis/
│   ├── xgbt_train.py        # 機械学習モデル学習スクリプト
│   └── xgbt_pred.py         # 機械学習モデル予測スクリプト
├── attack/
│   ├── attack_Ci.py         # Ci攻撃スクリプト
│   ├── attack_Di.py         # Di攻撃スクリプト
│   └── attack_example.py    # 総合攻撃スクリプト
├── HI_10k_anon.csv          # 生成される匿名データ（Ci相当）
├── HI_10k_anon.json         # 生成される機械学習モデル（Di相当）
└── attack_result.csv        # 攻撃結果（推定されたメンバーシップ）
```

## 注意事項

1. 匿名加工時に `--seed` オプションを指定すると再現可能な結果が得られます
2. 各評価指標は0-1の範囲で出力され、小さいほど元データとの差異が少ないことを示します
3. 有用性スコアが高いほど、匿名化後も元データの有用性が保たれていることを示します
4. 攻撃結果の評価では、正解データとの比較により攻撃性能を測定できます

## 実際のコンテストとの違い

このガイドは学習目的のため、実際のPWS Cup 2025とは以下の点で異なります：

### 攻撃対象の違い
- **このガイド**: 自分で作成したAi、Ci、Diを攻撃
- **実際のコンテスト**: 他のチームが作成したAi、Ci、Diを攻撃

### 攻撃精度の要件
- **このガイド**: サンプル攻撃スクリプトは適当な数のユーザをメンバーとして判別
- **実際のコンテスト**: **正確に10,000ユーザ**をメンバーとして予測する必要がある

### 正解データの利用可能性
- **このガイド**: data/HI_ans.csvで攻撃結果を検証可能
- **実際のコンテスト**: 正解は非公開のため、攻撃性能の事前検証は不可能

### データアクセスの制限
- **このガイド**: 元データAi（HI_100K.csv）にアクセス可能
- **実際のコンテスト**: 元データAiは運営側のみが保持

このガイドに沿って作業を進めることで、PWS Cup 2025の加工フェーズと攻撃フェーズの基本的な流れを体験できます。
PWSCUP 2025 Scripts — 実践ガイド（日本語）

本ドキュメントは、リポジトリの開発メモ（usage.txt）を読みやすく再構成した実践ガイドです。匿名化、評価、攻撃（Ci/Di/組み合わせ）の主要ワークフローを、リポジトリ直下で実行する前提でまとめています。

重要: 入力ファイルは `in/`、出力ファイルは `out/` に配置・生成されます。


**前提条件**
- Python パッケージ: `pip install -r requirements.txt`
- OS 想定: usage.txt に合わせて Windows パス表記を使用


**フォルダ構成**
- `in/`: 入力 CSV（例: `B22_*.csv`）を配置
- `out/`: 生成物（例: `C22_*.csv`, `C22_*_shuffled.csv`, サンプル `D22_*.json`、攻撃結果、レポートなど）
- `anonymization/`: 匿名化と精度評価（`ano.py`, `randomshuffle_rows.py`, `gen_Di.py`, `MLacc_files.py`）
- `evaluation/`: 指標計算・採点（`eval_all.py`, `gen_ans.py`, `check_ans.py`）
- `attack/`: 攻撃（Ci/Di/組み合わせ）実装とバリアント、バッチ処理
- `util/`: CSV の検証・修正など各種ユーティリティ
- `analysis/`: モデル検査系（例: `validate_model_json.py`）


**ファイル命名（usage.txt での慣習）**
- `B..`: 入力 CSV（`in/` 配下）。例: `in/B22_3.csv`
- `C..`: 匿名化・加工後 CSV（`out/` 配下）。例: `out/C22_3.csv`, `out/C22_3_shuffled.csv`
- `D..`: 学習済みモデル JSON（例: XGBoost）。例: `out/sample_D22_3.json` や `out/PWSCUP2025_Pre_Data_for_Attack/D15.json`
- `A..`: 攻撃で利用する補助データ。例: `out/PWSCUP2025_Pre_Data_for_Attack/A01.csv`
- `Fij_..`: 会員推定（Membership Inference）の最終出力。各レコード 0/1 など


**クイックスタート（B22_3 を例に）**
1) 匿名化と派生物の生成
   - `python anonymization/ano.py in/B22_3.csv out/C22_3.csv`
   - `python anonymization/randomshuffle_rows.py out/C22_3.csv out/C22_3_shuffled.csv`
   - `python anonymization/gen_Di.py in/B22_3.csv out/C22_3_shuffled.csv out/sample_D22_3.json`

2) 機械学習ベースの精度サマリ出力
   - `python anonymization/MLacc_files.py`
   - 出力: `out/MLacc_files.txt`
   - 備考: この精度は `stroke_flag` の推論誤差のみに依存

3) 匿名化品質の評価（C22_*）
   - `python evaluation/eval_all.py in/B22_1.csv out/C22_1_shuffled.csv`
   - `python evaluation/eval_all.py in/B22_2.csv out/C22_2_shuffled.csv`
   - `python evaluation/eval_all.py in/B22_3.csv out/C22_3_shuffled.csv`
   - 例として `stats_diff (max_abs)`, `LR_asthma_diff (max_abs)`, `KW_IND_diff (max_abs)`, `Ci utility` 等が出力されます


**攻撃の概要**
- Ci 攻撃: Ai と Ci の共通列で距離に基づく近傍マッチング
- Di 攻撃: 予測（Prediction）攻撃と信頼度（Confidence）攻撃の組み合わせ
- 例（距離 + 予測 + 信頼度の結合; 閾値超えで会員判定）:
  - `python attack/attack_example.py -o out/Fij_01.csv out\PWSCUP2025_Pre_Data_for_Attack_01-05\A01.csv out\PWSCUP2025_Pre_Data_for_Attack_01-05\C01.csv out\PWSCUP2025_Pre_Data_for_Attack_01-05\D01.json`
- 注意: `Fij_01.csv` は 100,000 行。素朴設定のままだと 1 の個数が 10,000 を超えやすく、偽陰性（第 II 種の誤り）が多くなる可能性があります

攻撃用データの準備
- `out/PWSCUP2025_Pre_Data_for_Attack/` を作成
- 配布物 `PWSCUP2025_Pre_Data_for_Attack_**` を上記フォルダへ展開


**攻撃バリアントと実行コマンド**
- Ci 攻撃（original / extended / greedy）
  - `python attack\attack_Ci.py out\PWSCUP2025_Pre_Data_for_Attack\A22.csv out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv -o out\C22_inferred.csv`
  - `python attack\attack_Ci_ex.py out\PWSCUP2025_Pre_Data_for_Attack\A22.csv out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv -o out\C22_inferred_ex.csv -k 1`
  - `python attack\attack_Ci_ex_greedy.py out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv -m nn -k 5 -o out\C22_inferred_ex_greedy_k5_nn.csv`
  - `python attack\attack_Ci_ex_greedy.py out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv -m greedy -k 20 -o out\C22_inferred_ex_greedy_k20_greedy.csv --out-map out\C22_matchmap_k20.csv`

- Di 攻撃（original / extended）
  - `python attack\attack_Di.py out\PWSCUP2025_Pre_Data_for_Attack\D22.json out\PWSCUP2025_Pre_Data_for_Attack\A22.csv`
  - 拡張版の主な指定項目:
    - `python attack\attack_Di_ex.py D15.json A15.csv --pred-threshold 0.5 --conf-threshold 0.1 --out-pred out\pred_15.csv --out-conf out\conf_15.csv`
    - Pred_Attack の選択数制御: `--pred-topk K` または `--pred-pos-ratio R`（topk が優先。いずれか指定時は threshold 無視）
    - Conf_Attack の選択数制御: `--conf-topk K` または `--conf-pos-ratio R`（topk が優先。いずれか指定時は threshold 無視）
    - 出力: 0/1 の 1 列 CSV（ヘッダ・インデックスなし）。特徴量は model.feature_names に自動整列（欠損は 0）

- 組み合わせ攻撃（original / 限界数付き）
  - `python attack\attack_example_ex.py --Ai_csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv -o out\Fij_22.csv out\C22_inferred.csv out\inferred_membership1_22_ex.csv out\inferred_membership2_22_ex.csv`
  - 1 の個数を 10,000 に制限: `python attack\attack_example_ex.py --Ai_csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv -o out\Fij_22.csv -l 10000 out\C22_inferred.csv out\inferred_membership1_22_ex.csv out\inferred_membership2_22_ex.csv`

- 新しい Di→Ci スコアリング攻撃
  - 目的: Di（Pred/Conf）で候補を抽出し、Ci の k-NN 距離で順位付けして上位 N を選択
  - 例: `python attack\new_attackDi_Ci.py out\PWSCUP2025_Pre_Data_for_Attack\A22.csv out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv out\PWSCUP2025_Pre_Data_for_Attack\D22.json --pred-topk 10000 --conf-topk 10000 --mode union -k 5 --topn 10000 -o out\Fij_new_22.csv --out-rank out\Fij_new_22_rank.csv`
  - 主なオプション:
    - Pred 選択: `--pred-threshold`, `--pred-topk`, `--pred-pos-ratio`（優先度: topk > ratio > threshold）
    - Conf 選択: `--conf-threshold`, `--conf-topk`, `--conf-pos-ratio`（優先度: topk > ratio > threshold）
    - Pred/Conf の結合: `--mode {union,intersection}`（既定: union）
    - Ci 指標: `-k` 近傍（既定: 5）。スコアは `w_hits * knn_hits - w_dist * min_dist`
      - 重み: `--w-hits`（既定 0.0）, `--w-dist`（既定 1.0）
    - 出力制御: `--topn`（既定: 1）, `-o/--out` 出力 CSV, `--out-rank` 順位表を保存
    - フォールバック: Di 候補が 0 件のとき、Ci 距離のみで全行を順位付け（標準出力に警告を表示）

- バッチ実行
  - `attack\multi_attack.ipynb`, `attack\multi_attack.py`


**開発メモ（usage.txt 抜粋の要旨）**
- Ci 攻撃は 1-NN から k-NN に拡張。出力に「重複カウント」と「最近傍距離の最小値」を含められる
- Di 攻撃は結果を 0/1 ではなく信頼度そのものや上位 n 個などで扱える
- 複合例:
  - Ci の k-NN 情報と Di の信頼度を重み付け結合
  - 論理積（`Ci_attack==1 and Di_attack==1`）
  - 上位 10,000 件のランキング出力
- 新バリアント（`new_attackDi_Ci`）: Di の上位 n を Ci 距離で加点し総合順位化、最上位を出力


**攻撃（Ci）開発時コマンド**
- 拡張 Ci 推定（k 近傍）
  - `python attack\attack_Ci_ex.py out\PWSCUP2025_Pre_Data_for_Attack\A01.csv out\PWSCUP2025_Pre_Data_for_Attack\C01.csv -o out\C01_inferred_ex.csv -k 1`
  - 出力列: (1) Ci による推定回数（k を跨ぐ重複数）、(2) 最近傍までの最小距離
- 重複なし形式へ変換
  - `python attack\ex_to_normal.py out\C01_inferred_ex.csv out\C01_inferred.csv`


**攻撃（Di）拡張 CLI の要点**
- 基本指定:
  - 閾値指定: `--pred-threshold TP`（既定 0.5）, `--conf-threshold TC`（既定 0.1, |p - y| が小さいほど採用）
  - 厳密件数: `--pred-topk K`, `--conf-topk K`
  - 比率指定: `--pred-pos-ratio R`, `--conf-pos-ratio R`（四捨五入で件数化）
- 優先度（各攻撃内）: topk > pos_ratio > threshold
- 出力先: `--out-pred`, `--out-conf`（既定は `inferred_membership1_ex.csv`, `inferred_membership2_ex.csv`）


**品質・整合性ツール**
- Ci CSV を範囲チェックし修正
  - `python util/check_and_fix_csv.py out\PWSCUP2025_Pre_Data_for_Attack\C01.csv data\pre_columns_range.json out\PWSCUP2025_Pre_Data_for_Attack\C01_fix.csv --report fix_report.csv`
  - 複数ファイル版: `python util\multi_check_and_fix.py`
- モデル JSON の検証
  - `python analysis/validate_model_json.py out\PWSCUP2025_Pre_Data_for_Attack\D15.json`
  - 例: OK (#features=27, target=stroke_flag, attrs_src=learner)


**匿名性評価の例**
- 答え生成と採点
  - `python evaluation\gen_ans.py out\PWSCUP2025_Pre_Data_for_Attack\A22.csv in\B22_3.csv -o out\Z22.csv`
  - `python evaluation\check_ans.py out\Z22.csv out\C01_inferred_ex_greedy.csv` → 例: 1008pt（greedy attack）
- サンプル攻撃の実行
  - `python attack/attack_example.py -o out/example_22.csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv out\PWSCUP2025_Pre_Data_for_Attack\D22.json`
  - 注意: サンプル出力は `Fij.csv` の 1 の個数が 10,000 を超える場合があります。必要なら次の簡易修正を利用
  - 簡易修正（ランダム制限）:
    - `python util\fix_Fijcsv_random.py out\example_22.csv out\example_22_fix.csv`
    - 例: 修正後スコア ≈ 1013pt


**ヒント / 実務上の注意**
- `Fij.csv` の 1（会員）件数は 10,000 に制限することを推奨
- Ci の greedy マッチは、各 id を 1 回のみ使う前提で最大 k 近傍から貪欲にペアを組む（会員推定が目的で、完全な対応推定ではない）
- `out/` の `fix_report.csv` など衛生チェック系の成果物を随時確認


**開発アイデア（usage.txt の提案要約）**
- 提案0（簡単/効率）: 元スコアを用い 1 の件数を 10,000 に制限
- 提案1（Ci 中程度）: ハンガリアン法を Ci に導入し、1 を 10,000 に制限
- 提案2（Di 簡単）: Di 攻撃の信頼度閾値を引き上げる
- 提案3（Ci+Di 中程度）: Ci/Di の 1/0 にノイズ注入し、その量で重み付け
- 提案4（Di 難/スクラッチ）: 複数 ML モデルを探索し、対象モデルと整合をとる
- 提案5（Ci 中程度/効率）: 相互最近傍（Ci→Ai と Ai→Ci の双方制約）
- 提案6（Ci 中程度/スクラッチ/効率）: 1-NN を k-NN に一般化


**参考**
- 全体フロー図: `PWSCUP2025flow.pdf`
- 詳細な履歴と全コマンド: `usage.txt`


**注記**
- 本書は `usage.txt` の要点を保ちつつ、読みやすさを優先して再編集した日本語版です

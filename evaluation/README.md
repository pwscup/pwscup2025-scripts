# `eval_all.py`： Ciの有用性評価を実行し、80点満点で採点
- 概要：3分野の有用性評価モジュール(stats_diff.py, LR_asthma_diff.py, KW_IND_diff.py)のeval()を呼び出し、各モジュールでの得点をそれぞれ40、20、20で重みをつけて80点満点で計算
- 書式：`$ python eval_all.py [-h] [-d] Bi_csv Ci_csv`
- 引数：
    - `Bi_csv`: path to Bi.csv
    - `Ci.csv`: path to Ci.csv
    - `-d`(任意): 途中のログを省略せずに出力
    - `-h`(任意): ヘルプを表示
- 入出力例：
    - `$ python3 evaluation/eval_all.py data/HI_10K.csv data/MA_10K.csv`
    - `stats_diff max_abs: 0.5518`
    - `LR_asthma_diff max_abs: 0.8862336002175595`
    - `KW_IND_diff max_abs: 0.033761053244578954`
    - `Ci utility: 39.52810693075723 / 80`
# `stats_diff.py`： Ciの基本統計の評価(Ci有用性評価、分野1)
- 概要：Bi.csv, Ci.csv の様々な基本的統計値の比較して最大の差分を0(最低)〜1（最高）に正規化してコマンドラインに出力
- 書式：`$ python stats_diff.py [-h] [-o OUT] csv1 csv2`
- 引数：
    - `csv1`：Bi.csvへのパス
    - `csv2`：Ci.csvへのパス(BiとCiの順番は逆でも結果は同じ)
    - `-o OUT`(任意)：途中結果(全ての基本的統計の差分の一覧)を保存するファイル(`out.csv`等)
# `LR_asthma_diff.py`：Ciをロジスティック回帰に使用した場合の評価（Ci有用性評価、分野2）
- 概要：Bi.csv, Ci.csvそれぞれを訓練データとして、"asthma_flag"を目的編するとするロジスティック回帰を実行。得られた2つの学習済みモデルの各種指標(coef, p_value, OR_norm, CI_low_norm, CI_high_norm, VIF_norm)の差分を0-1に正規化して、最大の差分をコマンドラインに出力する。
- 書式：`$ python LR_asthma_diff.py [-h] [--target TARGET] [--test-size TEST_SIZE]
                         [--random-state RANDOM_STATE]
                         [--ensure-terms ENSURE_TERMS] [-o OUT]
                         csv1 csv2`
- 引数：
    - `csv1`：Bi.csvへのパス
    - `csv2`：Ci.csvへのパス(BiとCiの順番は逆でも結果は同じ)
    - `-h`(任意): ヘルプを表示
    - `--target TARGET`(任意)：目的変数になる列を指定。Defaultは"asthma_flag"で固定。
    - `--test-size TEST_SIZE`(任意)：テストデータの割合を指定。Defaultは0.2で固定。
    - `--random-state RANDOM_STATE`(任意)：テスト・訓練データの分割に使う乱数のseed値。Defaultでは42で固定。
    - `--ensure-terms ENSURE_TERMS`(任意)：常に出力に含めたい term 名（カンマ区切り）
    - `-o OUT`(任意)：各学習モデルのAUCと途中結果(全ての指標の差分の一覧)を保存するファイル(`out.csv`等)

# `KW_IND_diff.py`： Ci.csvのKruskal–Wallis指標（KW指標、安定p・0～1効果量・Hの正規化）の評価（Ci有用性評価、分野3）
- 概要：Bi.csvとCi.csvそれぞれの列"encounter_count", "num_medications", "num_procedures", "num_immunizations", "num_devices"のKW指標群("H_norm", "minus_log10_p_norm", "epsilon2", "eta2_kw", "rank_eta2", "A_pair_avg", "A_pair_sym")を評価し0-1に正規化した上で最大の差分をコマンドラインに出力
- 書式：`$ python KW_IND_diff.py [-h] [--age-col AGE_COL] [--metrics METRICS]
                      [--custom-bins CUSTOM_BINS] [--min-per-group MIN_PER_GROUP]
                      [--p-norm {arctan,exp,log1p}] [--p-scale P_SCALE]
                      [--p-cap P_CAP] [-o OUT]
                      csv1 csv2`
- 引数：
    - `csv1`：Bi.csvへのパス
    - `csv2`：Ci.csvへのパス(BiとCiの順番は逆でも結果は同じ)
    - `-h`(任意): ヘルプを表示
    - `--age-col AGE_COL`(任意)：年齢列名（既定: AGE
    - `--metrics METRICS`(任意)：解析する指標列（カンマ区切り）既定: encounter_count,num_medications,num_procedures,num_immunizations,num_devices
    -  `--custom-bins CUSTOM_BINS`(任意)：臨床カスタム区切り（例: 0,18,45,65,75,200）。未指定なら `[0, 18, 45, 65, 75, 200]`を使用
    -  `--min-per-group MIN_PER_GROUP`(任意)：各群の最小サンプル数（既定: 2）
    -  `--p-norm`(任意)： {arctan,exp,log1p}のいずれか。minus_log10_p の 0–1 正規化方式（既定: arctan）
    -  `--p-scale P_SCALE`(任意)：arctan/exp のスケール（大きいほどゆっくり1に近づく）既定: 10
    -  `--p-cap P_CAP`(任意)：log1p 正規化の上限（既定: 300）
    -  `-o, --out OUT`(任意)：差分テーブルをCSV保存するパス

# `gen_ans.py`：匿名性評価用のメンバー/非メンバー正解データ生成
- 概要：Ai.csv と Bi.csv を入力し、Bi.csvのレコードと一致するAi.csvの行を1とする1列の CSV ファイル（正解データ）Zi.csv を書き出し。
- 書式：`gen_ans.py [-h] -o OUTPUT Ai.csv Bi.csv`
- 引数：
    - `Ai.csv`：ヘッダー付き Ai.csv
    - `Bi.csv`：ヘッダー付き Bi.csv
    - `-o OUTPUT`：生成された正解データの保存場所
    - `-h`(任意)：ヘルプをコマンドラインに出力
- 使用例：`$ python gen_ans.py A99.csv B99.csv -o Z99.csv`

# `check_ans.py`：匿名性採点用コード
- 概要：Zi.csv と Fij.csv を入力し、ともに値が1となっている行番号（正解）と正解数を出力 

# 開発用Tips
各採点用モジュールにはCSVファイルへのパスを入力するとその分野での得点を返す`eval(path_to_csv1:str, path_to_csv2:str)->float`とpandasのDataFrameから得点を計算する`eval_diff_max_abs(df1:pd.DataFrame, df2:pd.DataFrame) -> float`を用意しました。結果をCSVに書き出さずに何度も採点したい場合はこれらの関数を使ってください。

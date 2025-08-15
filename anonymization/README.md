# `ano.py`：ランダム化処理ベースのサンプル匿名化
- 概要：CSVファイルを入力して、二値の列は確率反転、カテゴリ列は確率的ランダム置換、数値列はノイズ付加を行う。確率やノイズ範囲のパラメータ値はソースコードの値を直接変更できる。
- 入力：ヘッダー付き CSV
    - 書式：`python3 ano.py <input.csv> <output.csv> \[--seed SEED\]`
    - 実行例：`python3 ano.py input.csv output.csv --seed 42`
    - 引数：
        - `--seed`：値を入れると乱数を固定でき再現性ある出力になる（内部で `numpy` と `random` の両方に設定）  
- 出力：ランダム化されたヘッダー付き CSV（レコードの順序は変化しないので注意。レコードの順序のランダムな置き換えは `randomshuffle_rows.py` を使う）
- はじめに行う処理：入力データを読み込み、各列について二値、
- 二値の列の確率反転
    - 対象：`asthma_flag`, `stroke_flag`, `obesity_flag`, `depression_flag`
    - 方法：内部で設定した置換確率に基づき値を反転する
- カテゴリ列の確率的ランダム置換
    - 対象：`GENDER`, `RACE`, `ETHNICITY`
    - 方法：内部で設定した置換確率に基づき値をランダムに置き換える
- 整数値列のノイズ付加
    - 対象：`AGE`, `encounter_count`, `num_procedures`, `num_medications`, `num_immunizations`, `num_allergies`, `num_devices`
    - 方法：整数ノイズの上限（正の値を想定）と下限（負の値を想定）を決め、範囲内のノイズをランダムに生成し、元の値に足す。`AGE` は設定した上限値、下限値でクランプする。その他は入力データから最大値、最小値を求め、その範囲でクランプする。 
- 実数値列のノイズ付加
    - 対象：`mean_systolic_bp`, `mean_diastolic_bp`, `mean_weight`, `mean_bmi`
    - 方法：実数ノイズの上限（正の値を想定）と下限（負の値を想定）を決め、範囲内のノイズをランダムに生成し、元の値に足す。入力データから最大値、最小値を求め、その範囲でクランプする。

# `randomshuffle_rows.py`：レコードの順番をランダムに入れ替える
- 入力：ヘッダー付き CSV
- 出力：入力の各レコードの順番がランダムに入れ替わったヘッダー付き CSV（ヘッダー行はそのまま）

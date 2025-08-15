- `gen_ans.py`：Ai.csv と Bi.csv を入力し、Bi.csvのレコードと一致するAi.csvの行を1とする1列の CSV ファイル（正解データ）Zi.csv を出力
- `check_ans.py`：Zi.csv と Fij.csv を入力し、ともに値が1となっている行番号（正解）と正解数を出力
- `stats_diff.py`：`stats.py` に Bi.csv, Ci.csv を入力として得られる統計値の差分を出力（有用性評価指標）
- `KW_IND_diff.py`：`KW_IND.py` に Bi.csv, Ci.csv を入力として得られる統計値の差分を出力（有用性評価指標）
- `LR_asthma_diff.py`：`LR_asthma.py` に Bi.csv, Ci.csv を入力として得られる統計値の差分を出力（有用性評価指標）


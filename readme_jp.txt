PWSCUP 2025 Scripts 窶・螳溯ｷｵ繧ｬ繧､繝会ｼ域律譛ｬ隱橸ｼ・

譛ｬ繝峨く繝･繝｡繝ｳ繝医・縲√Μ繝昴ず繝医Μ縺ｮ髢狗匱繝｡繝｢・・sage.txt・峨ｒ隱ｭ縺ｿ繧・☆縺丞・讒区・縺励◆螳溯ｷｵ繧ｬ繧､繝峨〒縺吶ょ諺蜷榊喧縲∬ｩ穂ｾ｡縲∵判謦・ｼ・i/Di/邨・∩蜷医ｏ縺幢ｼ峨・荳ｻ隕√Ρ繝ｼ繧ｯ繝輔Ο繝ｼ繧偵√Μ繝昴ず繝医Μ逶ｴ荳九〒螳溯｡後☆繧句燕謠舌〒縺ｾ縺ｨ繧√※縺・∪縺吶・

驥崎ｦ・ 蜈･蜉帙ヵ繧｡繧､繝ｫ縺ｯ `in/`縲∝・蜉帙ヵ繧｡繧､繝ｫ縺ｯ `out/` 縺ｫ驟咲ｽｮ繝ｻ逕滓・縺輔ｌ縺ｾ縺吶・


**蜑肴署譚｡莉ｶ**
- Python 繝代ャ繧ｱ繝ｼ繧ｸ: `pip install -r requirements.txt`
- OS 諠ｳ螳・ usage.txt 縺ｫ蜷医ｏ縺帙※ Windows 繝代せ陦ｨ險倥ｒ菴ｿ逕ｨ


**繝輔か繝ｫ繝讒区・**
- `in/`: 蜈･蜉・CSV・井ｾ・ `B22_*.csv`・峨ｒ驟咲ｽｮ
- `out/`: 逕滓・迚ｩ・井ｾ・ `C22_*.csv`, `C22_*_shuffled.csv`, 繧ｵ繝ｳ繝励Ν `D22_*.json`縲∵判謦・ｵ先棡縲√Ξ繝昴・繝医↑縺ｩ・・
- `anonymization/`: 蛹ｿ蜷榊喧縺ｨ邊ｾ蠎ｦ隧穂ｾ｡・・ano.py`, `randomshuffle_rows.py`, `gen_Di.py`, `MLacc_files.py`・・
- `evaluation/`: 謖・ｨ呵ｨ育ｮ励・謗｡轤ｹ・・eval_all.py`, `gen_ans.py`, `check_ans.py`・・
- `attack/`: 謾ｻ謦・ｼ・i/Di/邨・∩蜷医ｏ縺幢ｼ牙ｮ溯｣・→繝舌Μ繧｢繝ｳ繝医√ヰ繝・メ蜃ｦ逅・
- `util/`: CSV 縺ｮ讀懆ｨｼ繝ｻ菫ｮ豁｣縺ｪ縺ｩ蜷・ｨｮ繝ｦ繝ｼ繝・ぅ繝ｪ繝・ぅ
- `analysis/`: 繝｢繝・Ν讀懈渊邉ｻ・井ｾ・ `validate_model_json.py`・・


**繝輔ぃ繧､繝ｫ蜻ｽ蜷搾ｼ・sage.txt 縺ｧ縺ｮ諷｣鄙抵ｼ・*
- `B..`: 蜈･蜉・CSV・・in/` 驟堺ｸ具ｼ峨ゆｾ・ `in/B22_3.csv`
- `C..`: 蛹ｿ蜷榊喧繝ｻ蜉蟾･蠕・CSV・・out/` 驟堺ｸ具ｼ峨ゆｾ・ `out/C22_3.csv`, `out/C22_3_shuffled.csv`
- `D..`: 蟄ｦ鄙呈ｸ医∩繝｢繝・Ν JSON・井ｾ・ XGBoost・峨ゆｾ・ `out/sample_D22_3.json` 繧・`out/PWSCUP2025_Pre_Data_for_Attack/D15.json`
- `A..`: 謾ｻ謦・〒蛻ｩ逕ｨ縺吶ｋ陬懷勧繝・・繧ｿ縲ゆｾ・ `out/PWSCUP2025_Pre_Data_for_Attack/A01.csv`
- `Fij_..`: 莨壼藤謗ｨ螳夲ｼ・embership Inference・峨・譛邨ょ・蜉帙ょ推繝ｬ繧ｳ繝ｼ繝・0/1 縺ｪ縺ｩ


**繧ｯ繧､繝・け繧ｹ繧ｿ繝ｼ繝茨ｼ・22_3 繧剃ｾ九↓・・*
1) 蛹ｿ蜷榊喧縺ｨ豢ｾ逕溽黄縺ｮ逕滓・
   - `python anonymization/ano.py in/B22_3.csv out/C22_3.csv`
   - `python anonymization/randomshuffle_rows.py out/C22_3.csv out/C22_3_shuffled.csv`
   - `python anonymization/gen_Di.py in/B22_3.csv out/C22_3_shuffled.csv out/sample_D22_3.json`

2) 讖滓｢ｰ蟄ｦ鄙偵・繝ｼ繧ｹ縺ｮ邊ｾ蠎ｦ繧ｵ繝槭Μ蜃ｺ蜉・
   - `python anonymization/MLacc_files.py`
   - 蜃ｺ蜉・ `out/MLacc_files.txt`
   - 蛯呵・ 縺薙・邊ｾ蠎ｦ縺ｯ `stroke_flag` 縺ｮ謗ｨ隲冶ｪ､蟾ｮ縺ｮ縺ｿ縺ｫ萓晏ｭ・

3) 蛹ｿ蜷榊喧蜩∬ｳｪ縺ｮ隧穂ｾ｡・・22_*・・
   - `python evaluation/eval_all.py in/B22_1.csv out/C22_1_shuffled.csv`
   - `python evaluation/eval_all.py in/B22_2.csv out/C22_2_shuffled.csv`
   - `python evaluation/eval_all.py in/B22_3.csv out/C22_3_shuffled.csv`
   - 萓九→縺励※ `stats_diff (max_abs)`, `LR_asthma_diff (max_abs)`, `KW_IND_diff (max_abs)`, `Ci utility` 遲峨′蜃ｺ蜉帙＆繧後∪縺・


**謾ｻ謦・・讎りｦ・*
- Ci 謾ｻ謦・ Ai 縺ｨ Ci 縺ｮ蜈ｱ騾壼・縺ｧ霍晞屬縺ｫ蝓ｺ縺･縺剰ｿ大ｍ繝槭ャ繝√Φ繧ｰ
- Di 謾ｻ謦・ 莠域ｸｬ・・rediction・画判謦・→菫｡鬆ｼ蠎ｦ・・onfidence・画判謦・・邨・∩蜷医ｏ縺・
- 萓具ｼ郁ｷ晞屬 + 莠域ｸｬ + 菫｡鬆ｼ蠎ｦ縺ｮ邨仙粋; 髢ｾ蛟､雜・∴縺ｧ莨壼藤蛻､螳夲ｼ・
  - `python attack/attack_example.py -o out/Fij_01.csv out\PWSCUP2025_Pre_Data_for_Attack_01-05\A01.csv out\PWSCUP2025_Pre_Data_for_Attack_01-05\C01.csv out\PWSCUP2025_Pre_Data_for_Attack_01-05\D01.json`
- 豕ｨ諢・ `Fij_01.csv` 縺ｯ 100,000 陦後らｴ譛ｴ險ｭ螳壹・縺ｾ縺ｾ縺縺ｨ 1 縺ｮ蛟区焚縺・10,000 繧定ｶ・∴繧・☆縺上∝⊃髯ｰ諤ｧ・育ｬｬ II 遞ｮ縺ｮ隱､繧奇ｼ峨′螟壹￥縺ｪ繧句庄閭ｽ諤ｧ縺後≠繧翫∪縺・

謾ｻ謦・畑繝・・繧ｿ縺ｮ貅門ｙ
- `out/PWSCUP2025_Pre_Data_for_Attack/` 繧剃ｽ懈・
- 驟榊ｸ・黄 `PWSCUP2025_Pre_Data_for_Attack_**` 繧剃ｸ願ｨ倥ヵ繧ｩ繝ｫ繝縺ｸ螻暮幕


**謾ｻ謦・ヰ繝ｪ繧｢繝ｳ繝医→螳溯｡後さ繝槭Φ繝・*
- Ci 謾ｻ謦・ｼ・riginal / extended / greedy・・
  - `python attack\attack_Ci.py out\PWSCUP2025_Pre_Data_for_Attack\A22.csv out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv -o out\C22_inferred.csv`
  - `python attack\attack_Ci_ex.py out\PWSCUP2025_Pre_Data_for_Attack\A22.csv out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv -o out\C22_inferred_ex.csv -k 1`
  - `python attack\attack_Ci_ex_greedy.py out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv -m nn -k 5 -o out\C22_inferred_ex_greedy_k5_nn.csv`
  - `python attack\attack_Ci_ex_greedy.py out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv -m greedy -k 20 -o out\C22_inferred_ex_greedy_k20_greedy.csv --out-map out\C22_matchmap_k20.csv`

- Di 謾ｻ謦・ｼ・riginal / extended・・
  - `python attack\attack_Di.py out\PWSCUP2025_Pre_Data_for_Attack\D22.json out\PWSCUP2025_Pre_Data_for_Attack\A22.csv`
  - 諡｡蠑ｵ迚医・荳ｻ縺ｪ謖・ｮ夐・岼:
    - `python attack\attack_Di_ex.py D15.json A15.csv --pred-threshold 0.5 --conf-threshold 0.1 --out-pred out\pred_15.csv --out-conf out\conf_15.csv`
    - Pred_Attack 縺ｮ驕ｸ謚樊焚蛻ｶ蠕｡: `--pred-topk K` 縺ｾ縺溘・ `--pred-pos-ratio R`・・opk 縺悟━蜈医ゅ＞縺壹ｌ縺区欠螳壽凾縺ｯ threshold 辟｡隕厄ｼ・
    - Conf_Attack 縺ｮ驕ｸ謚樊焚蛻ｶ蠕｡: `--conf-topk K` 縺ｾ縺溘・ `--conf-pos-ratio R`・・opk 縺悟━蜈医ゅ＞縺壹ｌ縺区欠螳壽凾縺ｯ threshold 辟｡隕厄ｼ・
    - 蜃ｺ蜉・ 0/1 縺ｮ 1 蛻・CSV・医・繝・ム繝ｻ繧､繝ｳ繝・ャ繧ｯ繧ｹ縺ｪ縺暦ｼ峨ら音蠕ｴ驥上・ model.feature_names 縺ｫ閾ｪ蜍墓紛蛻暦ｼ域ｬ謳阪・ 0・・

- 邨・∩蜷医ｏ縺帶判謦・ｼ・riginal / 髯千阜謨ｰ莉倥″・・
  - `python attack\attack_example_ex.py --Ai_csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv -o out\Fij_22.csv out\C22_inferred.csv out\inferred_membership1_22_ex.csv out\inferred_membership2_22_ex.csv`
  - 1 縺ｮ蛟区焚繧・10,000 縺ｫ蛻ｶ髯・ `python attack\attack_example_ex.py --Ai_csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv -o out\Fij_22.csv -l 10000 out\C22_inferred.csv out\inferred_membership1_22_ex.csv out\inferred_membership2_22_ex.csv`

- 譁ｰ縺励＞ Di竊辰i 繧ｹ繧ｳ繧｢繝ｪ繝ｳ繧ｰ謾ｻ謦・
  - 逶ｮ逧・ Di・・red/Conf・峨〒蛟呵｣懊ｒ謚ｽ蜃ｺ縺励，i 縺ｮ k-NN 霍晞屬縺ｧ鬆・ｽ堺ｻ倥￠縺励※荳贋ｽ・N 繧帝∈謚・
  - 萓・ `python attack\new_attackDi_Ci.py out\PWSCUP2025_Pre_Data_for_Attack\A22.csv out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv out\PWSCUP2025_Pre_Data_for_Attack\D22.json --pred-topk 10000 --conf-topk 10000 --mode union -k 5 --topn 10000 -o out\Fij_new_22.csv --out-rank out\Fij_new_22_rank.csv`
  - 荳ｻ縺ｪ繧ｪ繝励す繝ｧ繝ｳ:
    - Pred 驕ｸ謚・ `--pred-threshold`, `--pred-topk`, `--pred-pos-ratio`・亥━蜈亥ｺｦ: topk > ratio > threshold・・
    - Conf 驕ｸ謚・ `--conf-threshold`, `--conf-topk`, `--conf-pos-ratio`・亥━蜈亥ｺｦ: topk > ratio > threshold・・
    - Pred/Conf 縺ｮ邨仙粋: `--mode {union,intersection}`・域里螳・ union・・
    - Ci 謖・ｨ・ `-k` 霑大ｍ・域里螳・ 5・峨ゅせ繧ｳ繧｢縺ｯ `w_hits * knn_hits - w_dist * min_dist`
      - 驥阪∩: `--w-hits`・域里螳・0.0・・ `--w-dist`・域里螳・1.0・・
    - 蜃ｺ蜉帛宛蠕｡: `--topn`・域里螳・ 1・・ `-o/--out` 蜃ｺ蜉・CSV, `--out-rank` 鬆・ｽ崎｡ｨ繧剃ｿ晏ｭ・
    - 繝輔か繝ｼ繝ｫ繝舌ャ繧ｯ: Di 蛟呵｣懊′ 0 莉ｶ縺ｮ縺ｨ縺阪，i 霍晞屬縺ｮ縺ｿ縺ｧ蜈ｨ陦後ｒ鬆・ｽ堺ｻ倥￠・域ｨ呎ｺ門・蜉帙↓隴ｦ蜻翫ｒ陦ｨ遉ｺ・・

- 繝舌ャ繝∝ｮ溯｡・
  - `attack\multi_attack.ipynb`, `attack\multi_attack.py`


**髢狗匱繝｡繝｢・・sage.txt 謚懃ｲ九・隕∵葎・・*
- Ci 謾ｻ謦・・ 1-NN 縺九ｉ k-NN 縺ｫ諡｡蠑ｵ縲ょ・蜉帙↓縲碁㍾隍・き繧ｦ繝ｳ繝医阪→縲梧怙霑大ｍ霍晞屬縺ｮ譛蟆丞､縲阪ｒ蜷ｫ繧√ｉ繧後ｋ
- Di 謾ｻ謦・・邨先棡繧・0/1 縺ｧ縺ｯ縺ｪ縺丈ｿ｡鬆ｼ蠎ｦ縺昴・繧ゅ・繧・ｸ贋ｽ・n 蛟九↑縺ｩ縺ｧ謇ｱ縺医ｋ
- 隍・粋萓・
  - Ci 縺ｮ k-NN 諠・ｱ縺ｨ Di 縺ｮ菫｡鬆ｼ蠎ｦ繧帝㍾縺ｿ莉倥￠邨仙粋
  - 隲也炊遨搾ｼ・Ci_attack==1 and Di_attack==1`・・
  - 荳贋ｽ・10,000 莉ｶ縺ｮ繝ｩ繝ｳ繧ｭ繝ｳ繧ｰ蜃ｺ蜉・
- 譁ｰ繝舌Μ繧｢繝ｳ繝茨ｼ・new_attackDi_Ci`・・ Di 縺ｮ荳贋ｽ・n 繧・Ci 霍晞屬縺ｧ蜉轤ｹ縺礼ｷ丞粋鬆・ｽ榊喧縲∵怙荳贋ｽ阪ｒ蜃ｺ蜉・


**謾ｻ謦・ｼ・i・蛾幕逋ｺ譎ゅさ繝槭Φ繝・*
- 諡｡蠑ｵ Ci 謗ｨ螳夲ｼ・ 霑大ｍ・・
  - `python attack\attack_Ci_ex.py out\PWSCUP2025_Pre_Data_for_Attack\A01.csv out\PWSCUP2025_Pre_Data_for_Attack\C01.csv -o out\C01_inferred_ex.csv -k 1`
  - 蜃ｺ蜉帛・: (1) Ci 縺ｫ繧医ｋ謗ｨ螳壼屓謨ｰ・・ 繧定ｷｨ縺宣㍾隍・焚・峨・2) 譛霑大ｍ縺ｾ縺ｧ縺ｮ譛蟆剰ｷ晞屬
- 驥崎､・↑縺怜ｽ｢蠑上∈螟画鋤
  - `python attack\ex_to_normal.py out\C01_inferred_ex.csv out\C01_inferred.csv`


**謾ｻ謦・ｼ・i・画僑蠑ｵ CLI 縺ｮ隕∫せ**
- 蝓ｺ譛ｬ謖・ｮ・
  - 髢ｾ蛟､謖・ｮ・ `--pred-threshold TP`・域里螳・0.5・・ `--conf-threshold TC`・域里螳・0.1, |p - y| 縺悟ｰ上＆縺・⊇縺ｩ謗｡逕ｨ・・
  - 蜴ｳ蟇・ｻｶ謨ｰ: `--pred-topk K`, `--conf-topk K`
  - 豈皮紫謖・ｮ・ `--pred-pos-ratio R`, `--conf-pos-ratio R`・亥屁謐ｨ莠泌・縺ｧ莉ｶ謨ｰ蛹厄ｼ・
- 蜆ｪ蜈亥ｺｦ・亥推謾ｻ謦・・・・ topk > pos_ratio > threshold
- 蜃ｺ蜉帛・: `--out-pred`, `--out-conf`・域里螳壹・ `inferred_membership1_ex.csv`, `inferred_membership2_ex.csv`・・


**蜩∬ｳｪ繝ｻ謨ｴ蜷域ｧ繝・・繝ｫ**
- Ci CSV 繧堤ｯ・峇繝√ぉ繝・け縺嶺ｿｮ豁｣
  - `python util/check_and_fix_csv.py out\PWSCUP2025_Pre_Data_for_Attack\C01.csv data\pre_columns_range.json out\PWSCUP2025_Pre_Data_for_Attack\C01_fix.csv --report fix_report.csv`
  - 隍・焚繝輔ぃ繧､繝ｫ迚・ `python util\multi_check_and_fix.py`
- 繝｢繝・Ν JSON 縺ｮ讀懆ｨｼ
  - `python analysis/validate_model_json.py out\PWSCUP2025_Pre_Data_for_Attack\D15.json`
  - 萓・ OK (#features=27, target=stroke_flag, attrs_src=learner)


縲占ｿｽ險倥賎reedy 迚・Di竊辰i 繧ｹ繧ｳ繧｢繝ｪ繝ｳ繧ｰ謾ｻ謦・→豕ｨ諢丈ｺ矩・
- Greedy 迚医せ繧ｳ繧｢繝ｪ繝ｳ繧ｰ: `python attack\new_attackDi_Ci_greedy.py Ai_csv Ci_csv Di_json [options]`
  - 萓・ `python attack\new_attackDi_Ci_greedy.py out\PWSCUP2025_Pre_Data_for_Attack\A22.csv out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv out\PWSCUP2025_Pre_Data_for_Attack\D22.json --pred-topk 10000 --conf-topk 10000 --mode union --topn 10000 -o out\Fij_new_greedy_22.csv --out-rank out\Fij_new_greedy_22_rank.csv`
- Greedy 繝｢繝ｼ繝峨・ k 縺ｫ縺､縺・※: `attack_Ci_ex_greedy.py` 縺ｮ greedy 縺ｯ k 繧貞崋螳壹○縺壹∝ｿ・ｦ√↓蠢懊§縺ｦ繝ｩ繝ｳ繧ｯ・郁ｿ大ｍ鬆・ｽ搾ｼ峨ｒ閾ｪ蜍慕噪縺ｫ諡｡蠑ｵ縺励※蜈ｨ縺ｦ縺ｮ Ci 陦後↓蟇ｾ蠢懊☆繧・Ai 陦後ｒ隕九▽縺代↓縺・″縺ｾ縺呻ｼ・i 繧剃ｽｿ縺・・縺｣縺溷ｴ蜷医・谿九ｋ縺薙→縺後≠繧翫∪縺呻ｼ峨・

縲占ｿｽ險倥題ｷ晞屬縺ｮ閾ｪ蜍輔せ繧ｱ繝ｼ繝ｪ繝ｳ繧ｰ・・-auto-wdist・・
- `new_attackDi_Ci.py` / `new_attackDi_Ci_greedy.py` 縺ｫ `--auto-wdist` 繧定ｿｽ蜉縲・
- Ai/Ci 縺ｮ蜈ｱ騾壼・縺ｮ縺・■縲梧焚蛟､蛻・N縲阪後き繝・ざ繝ｪ蛻・C縲阪ｒ謗ｨ螳壹＠縲∬ｷ晞屬縺ｮ驥阪∩繧・1/(N+2C) 縺ｫ閾ｪ蜍戊ｪｿ謨ｴ縺励※ |pred 竏・y|・・..1・・縺ｨ繧ｹ繧ｱ繝ｼ繝ｫ繧呈純縺医∪縺吶・
- 譌｢螳壼､縺ｮ `--w-dist 1.0` 繧貞渕貅悶↓縲∝ｮ溷柑逧・↓縺ｯ `w_dist/(N+2C)` 縺檎畑縺・ｉ繧後∪縺吶・
**蛹ｿ蜷肴ｧ隧穂ｾ｡縺ｮ萓・*
- 遲斐∴逕滓・縺ｨ謗｡轤ｹ
  - `python evaluation\gen_ans.py out\PWSCUP2025_Pre_Data_for_Attack\A22.csv in\B22_3.csv -o out\Z22.csv`
  - `python evaluation\check_ans.py out\Z22.csv out\C01_inferred_ex_greedy.csv` 竊・萓・ 1008pt・・reedy attack・・
- 繧ｵ繝ｳ繝励Ν謾ｻ謦・・螳溯｡・
  - `python attack/attack_example.py -o out/example_22.csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv out\PWSCUP2025_Pre_Data_for_Attack\D22.json`
  - 豕ｨ諢・ 繧ｵ繝ｳ繝励Ν蜃ｺ蜉帙・ `Fij.csv` 縺ｮ 1 縺ｮ蛟区焚縺・10,000 繧定ｶ・∴繧句ｴ蜷医′縺ゅｊ縺ｾ縺吶ょｿ・ｦ√↑繧画ｬ｡縺ｮ邁｡譏謎ｿｮ豁｣繧貞茜逕ｨ
  - 邁｡譏謎ｿｮ豁｣・医Λ繝ｳ繝繝蛻ｶ髯撰ｼ・
    - `python util\fix_Fijcsv_random.py out\example_22.csv out\example_22_fix.csv`
    - 萓・ 菫ｮ豁｣蠕後せ繧ｳ繧｢ 竕・1013pt


**繝偵Φ繝・/ 螳溷漁荳翫・豕ｨ諢・*
- `Fij.csv` 縺ｮ 1・井ｼ壼藤・我ｻｶ謨ｰ縺ｯ 10,000 縺ｫ蛻ｶ髯舌☆繧九％縺ｨ繧呈耳螂ｨ
- Ci 縺ｮ greedy 繝槭ャ繝√・縲∝推 id 繧・1 蝗槭・縺ｿ菴ｿ縺・燕謠舌〒譛螟ｧ k 霑大ｍ縺九ｉ雋ｪ谺ｲ縺ｫ繝壹い繧堤ｵ・・・井ｼ壼藤謗ｨ螳壹′逶ｮ逧・〒縲∝ｮ悟・縺ｪ蟇ｾ蠢懈耳螳壹〒縺ｯ縺ｪ縺・ｼ・
- `out/` 縺ｮ `fix_report.csv` 縺ｪ縺ｩ陦帷函繝√ぉ繝・け邉ｻ縺ｮ謌先棡迚ｩ繧帝囂譎ら｢ｺ隱・


**髢狗匱繧｢繧､繝・い・・sage.txt 縺ｮ謠先｡郁ｦ∫ｴ・ｼ・*
- 謠先｡・・育ｰ｡蜊・蜉ｹ邇・ｼ・ 蜈・せ繧ｳ繧｢繧堤畑縺・1 縺ｮ莉ｶ謨ｰ繧・10,000 縺ｫ蛻ｶ髯・
- 謠先｡・・・i 荳ｭ遞句ｺｦ・・ 繝上Φ繧ｬ繝ｪ繧｢繝ｳ豕輔ｒ Ci 縺ｫ蟆主・縺励・ 繧・10,000 縺ｫ蛻ｶ髯・
- 謠先｡・・・i 邁｡蜊假ｼ・ Di 謾ｻ謦・・菫｡鬆ｼ蠎ｦ髢ｾ蛟､繧貞ｼ輔″荳翫￡繧・
- 謠先｡・・・i+Di 荳ｭ遞句ｺｦ・・ Ci/Di 縺ｮ 1/0 縺ｫ繝弱う繧ｺ豕ｨ蜈･縺励√◎縺ｮ驥上〒驥阪∩莉倥￠
- 謠先｡・・・i 髮｣/繧ｹ繧ｯ繝ｩ繝・メ・・ 隍・焚 ML 繝｢繝・Ν繧呈爾邏｢縺励∝ｯｾ雎｡繝｢繝・Ν縺ｨ謨ｴ蜷医ｒ縺ｨ繧・
- 謠先｡・・・i 荳ｭ遞句ｺｦ/蜉ｹ邇・ｼ・ 逶ｸ莠呈怙霑大ｍ・・i竊但i 縺ｨ Ai竊辰i 縺ｮ蜿梧婿蛻ｶ邏・ｼ・
- 謠先｡・・・i 荳ｭ遞句ｺｦ/繧ｹ繧ｯ繝ｩ繝・メ/蜉ｹ邇・ｼ・ 1-NN 繧・k-NN 縺ｫ荳闊ｬ蛹・


**蜿り・*
- 蜈ｨ菴薙ヵ繝ｭ繝ｼ蝗ｳ: `PWSCUP2025flow.pdf`
- 隧ｳ邏ｰ縺ｪ螻･豁ｴ縺ｨ蜈ｨ繧ｳ繝槭Φ繝・ `usage.txt`


**豕ｨ險・*
- 譛ｬ譖ｸ縺ｯ `usage.txt` 縺ｮ隕∫せ繧剃ｿ昴■縺､縺､縲∬ｪｭ縺ｿ繧・☆縺輔ｒ蜆ｪ蜈医＠縺ｦ蜀咲ｷｨ髮・＠縺滓律譛ｬ隱樒沿縺ｧ縺・
 
【新規】独立結合（Ci 距離 greedy + Di |pred - y|）のランキング攻撃
縲先眠隕上醍峡遶狗ｵ仙粋謾ｻ謦・ｼ・i 霍晞屬 greedy + Di |pred - y|・・- 讎りｦ・ Ci 蛛ｴ縺ｯ greedy 1蟇ｾ1 繝槭ャ繝√Φ繧ｰ縺ｧ蜷・Ai 縺ｫ霍晞屬繧貞牡繧雁ｽ薙※・域悴蜑ｲ蠖薙・ 1000.0・峨．i 蛛ｴ縺ｯ |pred - y| 繧貞・陦後〒邂怜・縲ゆｸ｡閠・ｒ驥阪∩莉倥￠縺励※蜈ｨ莉ｶ繧偵せ繧ｳ繧｢鬆・↓荳ｦ縺ｹ縲∽ｸ贋ｽ・N 莉ｶ繧・1 縺ｨ縺励∪縺吶・- 蜈･蜉・ `Ai_csv`, `Ci_csv`, `Di_json`・・GBoost Booster JSON・・- 蜃ｺ蜉・
  - `--out`: 1 蛻・CSV・・/1, 繝倥ャ繝辟｡縺暦ｼ・  - 霑ｽ蜉: `--out-rank` 繝ｩ繝ｳ繧ｭ繝ｳ繧ｰ荳隕ｧ, `--out-map` greedy 蟇ｾ蠢懆｡ｨ [ci_idx, ai_idx, distance, rank]
- 繧ｪ繝励す繝ｧ繝ｳ:
  - `--w-hits`, `--w-dist`, `--w-conf`・磯㍾縺ｿ・・  - `--auto-wdist`・郁ｷ晞屬驥阪∩繧・1/(N+2*C) 縺ｧ閾ｪ蜍輔せ繧ｱ繝ｼ繝ｪ繝ｳ繧ｰ・・  - `--k-hint`・・reedy 縺ｮ蛻晄悄霑大ｍ蟷・ょｿ・ｦ√↓蠢懊§縺ｦ閾ｪ蜍墓僑蠑ｵ・・  - `--topn`・亥・蜉・1 縺ｮ莉ｶ謨ｰ縲よ里螳・10000・・- 繧ｹ繧ｳ繧｢蠑・ `w_hits * greedy_match - w_dist_eff * greedy_dist - w_conf * |pred - y|`
- 螳溯｡御ｾ・
  - `python attack\\attack_Ci_Di_independent.py out\\PWSCUP2025_Pre_Data_for_Attack\\A22.csv out\\PWSCUP2025_Pre_Data_for_Attack\\C22_fix.csv out\\PWSCUP2025_Pre_Data_for_Attack\\D22.json --w-conf 1.0 --auto-wdist --k-hint 300 --topn 10000 -o out\\Fij_independent_22.csv --out-rank out\\Fij_independent_22_rank.csv --out-map out\\C22_matchmap_independent.csv`
- 繝舌ャ繝・ `attack/multi_attack.py` 縺ｫ譛ｬ謾ｻ謦・・荳諡ｬ螳溯｡後ｒ霑ｽ蜉貂医∩縲・- 豕ｨ諢・ greedy 縺ｯ繝ｩ繝ｳ繧ｯ繧貞ｿ・ｦ√↓蠢懊§縺ｦ閾ｪ蜍慕噪縺ｫ諡｡蠑ｵ縺励∪縺呻ｼ・i 縺悟ｰｽ縺阪ｋ縺ｨ譛ｪ蜑ｲ蠖薙・ Ci 縺梧ｮ九ｋ蝣ｴ蜷医′縺ゅｊ縺ｾ縺呻ｼ峨・

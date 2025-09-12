PWSCUP 2025 Scripts — Practical Guide

This readme is distilled from usage.txt to provide a clear, step-by-step guide to the repository’s workflows: anonymization, evaluation, and attack pipelines. All commands assume you are in the repository root.

Important: Input files go in `in/`. Outputs are written to `out/`.


**Prerequisites**
- Python: Install packages with `pip install -r requirements.txt`.
- OS: Commands below use Windows-style paths as in usage.txt.


**Folder Overview**
- `in/`: Place input CSVs (e.g., `B22_*.csv`).
- `out/`: Generated outputs (e.g., `C22_*.csv`, `C22_*_shuffled.csv`, sample `D22_*.json`, attacks, reports).
- `anonymization/`: Anonymization and accuracy scripts (e.g., `ano.py`, `randomshuffle_rows.py`, `gen_Di.py`, `MLacc_files.py`).
- `evaluation/`: Utility/quality metrics and scoring tools (e.g., `eval_all.py`, `gen_ans.py`, `check_ans.py`).
- `attack/`: Attack implementations and variants (Ci/Di/combined) and batch helpers.
- `util/`: CSV validation/fixers and misc utilities.
- `analysis/`: Model inspectors (e.g., `validate_model_json.py`).


**File Naming (as used in usage.txt)**
- `B..`: Source/input CSVs (placed in `in/`). Example: `in/B22_3.csv`.
- `C..`: Anonymized/processed CSVs (written to `out/`). Example: `out/C22_3.csv`, `out/C22_3_shuffled.csv`.
- `D..`: Model JSONs (e.g., XGBoost JSON). Example: `out/sample_D22_3.json` or provided `out/PWSCUP2025_Pre_Data_for_Attack/D15.json`.
- `A..`: Auxiliary data used by attacks. Example: `out/PWSCUP2025_Pre_Data_for_Attack/A01.csv`.
- `Fij_..`: Combined attack answers (membership inference results), one row per record.


**Quick Start (Example with B22_3)**
1) Anonymize and prepare outputs
   - `python anonymization/ano.py in/B22_3.csv out/C22_3.csv`
   - `python anonymization/randomshuffle_rows.py out/C22_3.csv out/C22_3_shuffled.csv`
   - `python anonymization/gen_Di.py in/B22_3.csv out/C22_3_shuffled.csv out/sample_D22_3.json`

2) Check ML-based accuracy summary across files
   - `python anonymization/MLacc_files.py`
   - Output: `out/MLacc_files.txt`
   - Note: This accuracy depends only on inference errors of `stroke_flag`.

3) Evaluate anonymization quality metrics (C22_*)
   - `python evaluation/eval_all.py in/B22_1.csv out/C22_1_shuffled.csv`
   - `python evaluation/eval_all.py in/B22_2.csv out/C22_2_shuffled.csv`
   - `python evaluation/eval_all.py in/B22_3.csv out/C22_3_shuffled.csv`
   - Reported metrics include, for example: `stats_diff` (max_abs), `LR_asthma_diff` (max_abs), `KW_IND_diff` (max_abs), and `Ci utility`.


**Attacks Overview**
- Ci attack: Distance-based matching on common columns between Ai and Ci.
- Di attack: Combination of Prediction attack and Confidence attack.
- Example combined attack (distance + prediction + confidence; membership if score > threshold):
  - `python attack/attack_example.py -o out/Fij_01.csv out\PWSCUP2025_Pre_Data_for_Attack_01-05\A01.csv out\PWSCUP2025_Pre_Data_for_Attack_01-05\C01.csv out\PWSCUP2025_Pre_Data_for_Attack_01-05\D01.json`
- Note: `Fij_01.csv` has 100,000 records; naive settings may exceed the limit of 10,000 positives (“1”s). Expect many Type II errors if not limited.

Prepare data for attacks
- Create and populate: `out/PWSCUP2025_Pre_Data_for_Attack/`
- Unzip provided archives `PWSCUP2025_Pre_Data_for_Attack_**` into that folder.


**Attack Variants and Commands**
- Ci attack (original/extended/greedy):
  - `python attack\attack_Ci.py out\PWSCUP2025_Pre_Data_for_Attack\A22.csv out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv -o out\C22_inferred.csv`
  - `python attack\attack_Ci_ex.py out\PWSCUP2025_Pre_Data_for_Attack\A22.csv out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv -o out\C22_inferred_ex.csv -k 1`
  - `python attack\attack_Ci_ex_greedy.py out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv -m nn -k 5 -o out\C22_inferred_ex_greedy_k5_nn.csv`
  - `python attack\attack_Ci_ex_greedy.py out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv -m greedy -k 20 -o out\C22_inferred_ex_greedy_k20_greedy.csv --out-map out\C22_matchmap_k20.csv`

- Di attack (original/extended):
  - `python attack\attack_Di.py out\PWSCUP2025_Pre_Data_for_Attack\D22.json out\PWSCUP2025_Pre_Data_for_Attack\A22.csv`
  - Extended (key options):
    - `python attack\attack_Di_ex.py D15.json A15.csv --pred-threshold 0.5 --conf-threshold 0.1 --out-pred out\pred_15.csv --out-conf out\conf_15.csv`
    - Selection controls (Pred_Attack): `--pred-topk K` or `--pred-pos-ratio R` (topk overrides ratio; both override threshold)
    - Selection controls (Conf_Attack): `--conf-topk K` or `--conf-pos-ratio R` (topk overrides ratio; both override threshold)
    - Outputs: 1-column CSVs with 0/1, no header or index; features auto-reindexed to model feature names (missing columns filled with 0).

- Combined attacks (original/with limit):
  - `python attack\attack_example_ex.py --Ai_csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv -o out\Fij_22.csv out\C22_inferred.csv out\inferred_membership1_22_ex.csv out\inferred_membership2_22_ex.csv`
  - Enforce 10,000 positives: `python attack\attack_example_ex.py --Ai_csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv -o out\Fij_22.csv -l 10000 out\C22_inferred.csv out\inferred_membership1_22_ex.csv out\inferred_membership2_22_ex.csv`

- Batch helpers:
  - Notebooks and script: `attack\multi_attack.ipynb`, `attack\multi_attack.py`


**Development Notes (from usage.txt)**
- Ci attack extended from 1-NN to k-NN; outputs can include counts and minimum distances.
- Di attack outputs can be raw confidence values or top-n confidences.
- Combinations:
  - Weighted fusion of Ci k-NN scores and Di confidences.
  - Logical AND (`Ci_attack==1 and Di_attack==1`).
  - Ranked outputs (e.g., top 10,000 records).
- New variant (`new_attackDi_Ci`): score top-n Di answers weighted by Ci distance; output the top-ranked record.


**Attack (Ci) — Dev Commands**
- Build extended Ci inference with k neighbors:
  - `python attack\attack_Ci_ex.py out\PWSCUP2025_Pre_Data_for_Attack\A01.csv out\PWSCUP2025_Pre_Data_for_Attack\C01.csv -o out\C01_inferred_ex.csv -k 1`
  - Output columns: (1) times inferred by Ci (overlaps across k), (2) min distance to nearest neighbor
- Convert extended to non-overlapping form:
  - `python attack\ex_to_normal.py out\C01_inferred_ex.csv out\C01_inferred.csv`


**Attack (Di) — Extended CLI Summary**
- Basic forms:
  - Use thresholds: `--pred-threshold TP` (default 0.5), `--conf-threshold TC` (default 0.1, selects small |p - y|)
  - Force exact counts: `--pred-topk K`, `--conf-topk K`
  - Proportional selection: `--pred-pos-ratio R`, `--conf-pos-ratio R` (select round(R*N))
- Priority within each attack: topk > pos_ratio > threshold
- Output files: set with `--out-pred PATH`, `--out-conf PATH` (defaults: `inferred_membership1_ex.csv`, `inferred_membership2_ex.csv`)


**Quality/Integrity Tools**
- Validate and fix Ci CSVs against ranges:
  - `python util/check_and_fix_csv.py out\PWSCUP2025_Pre_Data_for_Attack\C01.csv data\pre_columns_range.json out\PWSCUP2025_Pre_Data_for_Attack\C01_fix.csv --report fix_report.csv`
  - Multi-file version: `python util\multi_check_and_fix.py`
- Model JSON sanity check:
  - `python analysis/validate_model_json.py out\PWSCUP2025_Pre_Data_for_Attack\D15.json`
  - Example output: OK (#features=27, target=stroke_flag, attrs_src=learner)


**Anonymity Evaluation Examples**
- Generate expected answers and score:
  - `python evaluation\gen_ans.py out\PWSCUP2025_Pre_Data_for_Attack\A22.csv in\B22_3.csv -o out\Z22.csv`
  - `python evaluation\check_ans.py out\Z22.csv out\C01_inferred_ex_greedy.csv`  → example: 1008pt (greedy attack)
- Sample combined attack run:
  - `python attack/attack_example.py -o out/example_22.csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv out\PWSCUP2025_Pre_Data_for_Attack\C22_fix.csv out\PWSCUP2025_Pre_Data_for_Attack\D22.json`
  - Warning: sample output may exceed the 10,000 “1”s limit for `Fij.csv`. Use the fixer below if needed.
  - Enforce limit randomly (simple fixer):
    - `python util\fix_Fijcsv_random.py out\example_22.csv out\example_22_fix.csv`
    - Example: fixed score ≈ 1013pt


**Heuristics & Tips**
- Limit positives (“1”s) in `Fij.csv` to 10,000 for fairness/comparability.
- For Ci greedy matching, you can take up to k nearest candidates and assign pairs greedily (k=5 example), ensuring each id is used at most once. This is for membership inference, not full identity mapping.
- Keep an eye on `out/` artifacts like `fix_report.csv` for data hygiene.


**Ideas/Proposals From Development History**
- Proposal 0 (easy/efficient): Limit “1”s to 10,000 using the original combined score.
- Proposal 1 (Ci moderate): Limit “1”s to 10,000 using Hungarian assignment in Ci.
- Proposal 2 (Di easy): Raise the confidence threshold in AttackDi.
- Proposal 3 (Ci+Di moderate): Inject noise in Ci/Di 1/0 answers and weight by noise magnitude.
- Proposal 4 (Di hard/scratch): Explore multiple ML models; align with the target model via internal training comparisons.
- Proposal 5 (Ci moderate/efficient): Mutual nearest neighbors (both Ci→Ai and Ai→Ci constraints).
- Proposal 6 (Ci moderate/scratch/efficient): Generalize 1-NN to k-NN.


**References**
- High-level flow diagram: `PWSCUP2025flow.pdf`
- Full development log and raw commands: `usage.txt`


**Notes**
- This readme is a concise, cleaned summary of `usage.txt` to maximize readability while preserving the original intent and commands.

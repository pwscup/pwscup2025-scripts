# %%
import os
import sys
import numpy as np
import subprocess
from shutil import which

# %%
# 0..21 から 21(=index 20) を除くチーム一覧
def get_teams():
    arr = np.arange(1, 23)
    return np.delete(arr, 20)  # 21 をスキップ（コンテスト仕様に合わせる）

# %%
def loop_for_all_teams(
    command_template,
    *,
    dry_run=False,
    strict=True,
    continue_on_error=False,
    cwd=None
):
    """
    command_template: ['python', 'attack/attack_Ci.py', '...{id:02d}...', ...] のようなリスト
    dry_run: True -> 実行せず展開コマンドのみ表示
    strict: True -> {id:02d} が一つも無ければ例外
    continue_on_error: True -> 失敗しても次の team へ。False -> そこで中断
    cwd: サブプロセスの作業ディレクトリ（attack/ ディレクトリの相対パス解決に使える）
    """
    id_indices = [i for i, arg in enumerate(command_template)
                  if isinstance(arg, str) and "{id:02d}" in arg]
    if strict and not id_indices:
        raise ValueError(f"No {{id:02d}} placeholder found in: {command_template}")

    # 'python' を実行ファイルに置換（環境ズレ回避）
    cmd0 = command_template[:]
    if cmd0 and cmd0[0] in ("python", "python3"):
        cmd0[0] = sys.executable

    # 事前: 実行ファイルの存在チェック（python 以外の最初の実体コマンドにも対応）
    exe = cmd0[0]
    if os.path.sep not in exe and which(exe) is None:
        raise RuntimeError(f"Executable not found on PATH: {exe}")

    for team in get_teams():
        cmd = cmd0[:]
        for ind in id_indices:
            cmd[ind] = cmd[ind].format(id=team)

        # 簡易プリフライト: 既知の入力系ファイルっぽい引数を存在確認
        # （.csv, .json かつ -o/--out* ではない位置を対象にする）
        def is_out_flag(i):
            return isinstance(cmd[i-1], str) and (
                cmd[i-1] in ("-o", "--out", "--out-map", "--out-pred", "--out-conf")
                or cmd[i-1].startswith("--out")
            )

        missing_inputs = []
        for i, a in enumerate(cmd):
            if isinstance(a, str) and (a.endswith(".csv") or a.endswith(".json")):
                if not is_out_flag(i):  # 出力ではなく入力と推定
                    apath = a if cwd is None else os.path.join(cwd, a)
                    if not os.path.exists(apath):
                        missing_inputs.append(a)

        print(">>", " ".join(cmd))
        if missing_inputs:
            msg = f"[team {team}] Missing input files: {missing_inputs}"
            if continue_on_error:
                print("!!", msg)
                continue
            else:
                raise FileNotFoundError(msg)

        if dry_run:
            continue

        try:
            # 標準出力・標準エラーを取得して、失敗時に見せる
            completed = subprocess.run(
                cmd, check=True, cwd=cwd,
                capture_output=True, text=True
            )
            if completed.stdout:
                print(completed.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] team {team} command failed with code {e.returncode}")
            if e.stdout:
                print("--- stdout ---")
                print(e.stdout.strip())
            if e.stderr:
                print("--- stderr ---")
                print(e.stderr.strip())
            if not continue_on_error:
                raise

# %%
# 必要なら cwd='プロジェクトのルート' を指定（例: cwd=r'c:\work\pwscup2025'）
# cwd = r"C:\Users\kikus\Documents\統数研\pwscup2025-scripts"  # <- 適宜書き換え
# os.chdir(cwd)

# %%
## Attack of Ci with original samples
Ci_attack_original = ["python", "attack/attack_Ci.py", "out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv", "out/PWSCUP2025_Pre_Data_for_Attack/C{id:02d}_fix.csv", "-o", "out/C{id:02d}_inferred.csv"]
loop_for_all_teams(Ci_attack_original)
print(f"sample Ci-attack completed")

# %%
## Attack of Ci with extended version
Ci_attack_extended = ["python", "attack/attack_Ci_ex.py", "out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv", "out/PWSCUP2025_Pre_Data_for_Attack/C{id:02d}_fix.csv", "-o", "out/C{id:02d}_inferred_ex.csv", "-k", "1"]
loop_for_all_teams(Ci_attack_extended)
print(f"extended Ci-attack completed")

# %%
## Attack of Ci with k-NN version
mode = "nn"
k = 5 # choose Nearest k neighbors
Ci_attack_knn = ["python", "attack/attack_Ci_ex_greedy.py", "out/PWSCUP2025_Pre_Data_for_Attack/C{id:02d}_fix.csv", "out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv", "-m", mode, "-k", str(k), "-o", f"out/C{{id:02d}}_inferred_ex_greedy_k{k}_{mode}.csv"]
loop_for_all_teams(Ci_attack_knn)
print(f"k-NN Ci-attack completed")

# %%
## Attack of Ci with greedy-k-NN version
mode = "greedy"
k = 300 # choose greedy k ranks (need enough big k for computation)
Ci_attack_greedy = ["python", "attack/attack_Ci_ex_greedy.py", "out/PWSCUP2025_Pre_Data_for_Attack/C{id:02d}_fix.csv", "out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv", "-m", mode, "-k", str(k), "-o", f"out/C{{id:02d}}_inferred_ex_greedy_k{k}_{mode}.csv", "--out-map", f"out/C{{id:02d}}_matchmap_k{k}.csv"]
loop_for_all_teams(Ci_attack_greedy)
print(f"greedy-k-NN Ci-attack completed")

# %%
## Attack of Di with original samples
Di_attack_original = ["python", "attack/attack_Di.py", "out/PWSCUP2025_Pre_Data_for_Attack/D{id:02d}.json", "out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv"]
loop_for_all_teams(Di_attack_original)
print(f"sample Di-attack completed")

# %%
## Attack of Di with extended version
# python attack\attack_Di_ex.py out\PWSCUP2025_Pre_Data_for_Attack\D22.json out\PWSCUP2025_Pre_Data_for_Attack\A22.csv [--pred-threshold 0.5] [--pred-topk 10000] [--pred-pos-ratio 0.10] [--conf-threshold 0.1] [--conf-topk 10000] [--conf-pos-ratio 0.10] --out-pred out/inferred_membership1_22_ex.csv --out-conf out/inferred_membership2_22_ex.csv
# if threshold is not specified, default values will be used.
Di_attack_extended = ["python", "attack/attack_Di_ex.py", "out/PWSCUP2025_Pre_Data_for_Attack/D{id:02d}.json", "out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv", "--out-pred", "out/inferred_membership1_{id:02d}_ex.csv", "--out-conf", "out/inferred_membership2_{id:02d}_ex.csv"]
loop_for_all_teams(Di_attack_extended)
print(f"extended Di-attack completed")

# %%
## Original Combination Attack on Ci and Di.
# python attack\attack_Di_ex.py out\PWSCUP2025_Pre_Data_for_Attack\D22.json out\PWSCUP2025_Pre_Data_for_Attack\A22.csv [--pred-threshold 0.5] [--pred-topk 10000] [--pred-pos-ratio 0.10] [--conf-threshold 0.1] [--conf-topk 10000] [--conf-pos-ratio 0.10] --out-pred out/inferred_membership1_22_ex.csv --out-conf out/inferred_membership2_22_ex.csv
# if threshold is not specified, default values will be used.
Combi_attack_original = ["python", "attack/attack_example_ex.py", "--Ai_csv", "out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv", "-o", "out/Fij_{id:02d}.csv", "out/C{id:02d}_inferred.csv", "out/inferred_membership1_{id:02d}_ex.csv", "out/inferred_membership2_{id:02d}_ex.csv"]
loop_for_all_teams(Combi_attack_original)
print(f"original combination attack completed")

# %%
## Extended Combination Attack on Ci and Di.
# python attack\attack_example_ex.py --Ai_csv out\PWSCUP2025_Pre_Data_for_Attack\A22.csv -o out\Fij_22.csv -l 10000 out\C22_inferred.csv out\inferred_membership1_22_ex.csv out\inferred_membership2_22_ex.csv
# if threshold is not specified, default values will be used.
limit = 10000 # limit of number of samples to be inferred

Combi_attack_extended = ["python", "attack/attack_example_ex.py", "--Ai_csv", "out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv", "-o", "out/Fij_{id:02d}.csv", "-l", str(limit), f"out/C{{id:02d}}_inferred_ex_greedy_k{k}_{mode}.csv", "out/inferred_membership1_{id:02d}_ex.csv", "out/inferred_membership2_{id:02d}_ex.csv"]
loop_for_all_teams(Combi_attack_extended)
print(f"extended combination attack completed")


# %%
# New Di->Ci scoring attack (union of Di selections, rank by Ci distance and |pred - y|)
# Example knobs: threshold-based (broader candidates), k=5, w_conf=1.0, select topn=10000
New_DiCi_scoring = [
    "python", "attack/new_attackDi_Ci.py",
    "out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv",
    "out/PWSCUP2025_Pre_Data_for_Attack/C{id:02d}_fix.csv",
    "out/PWSCUP2025_Pre_Data_for_Attack/D{id:02d}.json",
    "--pred-threshold", "0.5",
    "--conf-threshold", "0.3",
    "--mode", "union",
    "-k", "5",
    "--w-conf", "1.0",
    "--auto-wdist",
    "--topn", "10000",
    "-o", "out/Fij_new_{id:02d}.csv",
    "--out-rank", "out/Fij_new_{id:02d}_rank.csv",
]
loop_for_all_teams(New_DiCi_scoring)
print(f"new Di->Ci scoring attack completed")

# %%
# New Di->Ci scoring attack (greedy Ci matching; k ignored, ranks expand automatically)
New_DiCi_scoring_greedy = [
    "python", "attack/new_attackDi_Ci_greedy.py",
    "out/PWSCUP2025_Pre_Data_for_Attack/A{id:02d}.csv",
    "out/PWSCUP2025_Pre_Data_for_Attack/C{id:02d}_fix.csv",
    "out/PWSCUP2025_Pre_Data_for_Attack/D{id:02d}.json",
    "--pred-threshold", "0.5",
    "--conf-threshold", "0.3",
    "--mode", "union",
    "--w-conf", "1.0",
    "--auto-wdist",
    "--topn", "10000",
    "-o", "out/Fij_new_greedy_{id:02d}.csv",
    "--out-rank", "out/Fij_new_greedy_{id:02d}_rank.csv",
    "--out-map", "out/C{id:02d}_matchmap_greedy.csv",
]
loop_for_all_teams(New_DiCi_scoring_greedy)
print(f"new Di->Ci scoring (greedy) completed")

import subprocess

command1 = ["python", "util/check_and_fix_csv.py", "out\\PWSCUP2025_Pre_Data_for_Attack\\C01.csv", "data\\pre_columns_range.json", "out\\PWSCUP2025_Pre_Data_for_Attack\\C01_fix.csv", "--report", "fix_report.csv"]

for team in range(1, 21):
    command1[2] = f"out\\PWSCUP2025_Pre_Data_for_Attack\\C{team:02d}.csv"
    command1[4] = f"out\\PWSCUP2025_Pre_Data_for_Attack\\C{team:02d}_fix.csv"
    subprocess.run(command1)
    print(f"check_and_fix for C{team:02d} completed")


command1 = ["python", "util/check_and_fix_csv.py", "out\\PWSCUP2025_Pre_Data_for_Attack\\C22.csv", "data\\pre_columns_range.json", "out\\PWSCUP2025_Pre_Data_for_Attack\\C22_fix.csv", "--report", "fix_report.csv"]
subprocess.run(command1)
print(f"check_and_fix for C22 completed")
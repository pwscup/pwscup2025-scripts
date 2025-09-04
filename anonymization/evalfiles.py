import subprocess
import re
import glob
def run_and_get_acc(i, run_id):
    cmd = [
        "python", "anonymization/gen_Di.py",
        f"in/B22_{i}.csv",
        f"out/C22_{i}_shuffled.csv",
        f"out/sample_D22_{i}_{run_id}.json"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # accuracy: 0.8973
    match = re.search(r"accuracy:\s*([0-9.]+)", result.stdout)
    if match:
        return float(match.group(1))
    else:
        return None

def main():
    with open("out/evalfiles.txt", "w", encoding="utf-8") as f:
        for i in range(1, 4):
            accs = []
            f.write(f"B22_{i} (in/B22_{i}.csv, out/C22_{i}_shuffled.csv)\n")
            for run_id in range(1, 11):
                acc = run_and_get_acc(i, run_id)
                accs.append(acc)
                f.write(f"  Run {run_id}: acc = {acc}\n")
            avg = sum(accs) / len(accs)
            f.write(f"  Average acc: {avg}\n\n")

if __name__ == "__main__":
    main()
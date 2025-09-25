# tools/pws_anonymize.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
KANON = ROOT / "third_party" / "k-anonymity"
sys.path.insert(0, str(KANON))  # ← 依存パスを通す

# 以降は third_party のスクリプトを呼ぶ
# 例：python third_party/k-anonymity/anonymize-pws.py の main() を import して実行
if __name__ == "__main__":
    import runpy
    script = KANON / "anonymize-pws.py"
    runpy.run_path(str(script), run_name="__main__")

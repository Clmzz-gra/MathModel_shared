# -*- coding: utf-8 -*-
"""修复 verify-S2-* 与 S3-e3-fewshot 的 utils import（函数已移至 utils-S2）。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

scratch = Path(__file__).resolve().parent
files = [
    "verify-S2-v1-baseline.py", "verify-S2-v2-zerobin.py",
    "verify-S2-v3-redundancy.py", "verify-S2-v4-overlap.py",
    "verify-S2-v5-clr.py", "verify-S2-v6-stability.py",
    "S3-e3-fewshot.py",
]
for fn in files:
    p = scratch / fn
    txt = p.read_text(encoding="utf-8")
    if "from utils_S2 import" in txt:
        print(f"{fn}: already fixed, skip")
        continue
    if "from utils import" not in txt:
        print(f"{fn}: no 'from utils import', skip")
        continue
    new = txt.replace("from utils import", "from utils_S2 import", 1)
    p.write_text(new, encoding="utf-8")
    print(f"{fn}: import fixed")
print("DONE")

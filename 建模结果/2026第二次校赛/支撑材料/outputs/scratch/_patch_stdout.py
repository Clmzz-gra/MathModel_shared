# -*- coding: utf-8 -*-
"""为缺 stdout.reconfigure 的探索脚本批量补上（修复 GBK 打印崩溃）。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

scratch = Path(__file__).resolve().parent
files = [
    "S1-model.py", "S2-model.py", "S3-model.py", "S3-e3-fewshot.py",
    "verify-S1-a1.py", "verify-S1-a2.py", "verify-S1-a3.py",
    "verify-S1-a4.py", "verify-S1-a5.py", "verify-S1-a6.py",
    "verify-S2-v1-baseline.py", "verify-S2-v2-zerobin.py",
    "verify-S2-v3-redundancy.py", "verify-S2-v4-overlap.py",
    "verify-S2-v5-clr.py", "verify-S2-v6-stability.py",
    "verify-S3-a1-baseline.py", "verify-S3-a2-overlap.py",
    "verify-S3-a3-domain-auc.py", "verify-S3-a4-hierarchy.py",
    "verify-S3-a5-batch.py",
]
ANCHOR = "import sys\n"
INSERT = "import sys\n\nsys.stdout.reconfigure(encoding=\"utf-8\")\n"
for fn in files:
    p = scratch / fn
    txt = p.read_text(encoding="utf-8")
    if "stdout.reconfigure" in txt:
        print(f"{fn}: already has, skip")
        continue
    # 找 import sys 行（在 docstring 之后）
    if "import sys\n" in txt:
        new = txt.replace(ANCHOR, INSERT, 1)
        p.write_text(new, encoding="utf-8")
        print(f"{fn}: inserted")
    else:
        print(f"{fn}: no 'import sys' anchor, skip")
print("DONE")

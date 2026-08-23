# -*- coding: utf-8 -*-
"""批量给探索图脚本插入 pdf.fonttype=42（消除 Type3 方块字）。"""
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
ANCHOR = 'matplotlib.use("Agg")'
INSERT = (
    'matplotlib.rcParams["pdf.fonttype"] = 42\n'
    'matplotlib.rcParams["ps.fonttype"] = 42'
)
for fn in files:
    p = scratch / fn
    txt = p.read_text(encoding="utf-8")
    if "pdf.fonttype" in txt:
        print(f"{fn}: already has fonttype, skip")
        continue
    if ANCHOR not in txt:
        print(f"{fn}: ANCHOR NOT FOUND, skip")
        continue
    new = txt.replace(ANCHOR, ANCHOR + "\n" + INSERT, 1)
    p.write_text(new, encoding="utf-8")
    print(f"{fn}: inserted OK")
print("DONE")

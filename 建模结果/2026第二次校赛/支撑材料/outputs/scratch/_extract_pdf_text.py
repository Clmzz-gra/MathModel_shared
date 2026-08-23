"""提取 S2 三张正式图 PDF 的文本，核对自明性元素是否齐全。"""
import sys
import fitz
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIG = Path(r"E:\MathModel_pj-2026-sim2-B\outputs\figures")

for name in ["S2-stable-frequency", "S2-tau-sensitivity", "S2-cooccurrence-heatmap"]:
    doc = fitz.open(str(FIG / f"{name}.pdf"))
    print("=" * 60)
    print(name)
    for page in doc:
        txt = page.get_text()
        print(txt)
    print()

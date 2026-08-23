"""检查 S2 PDF 的字体与文本 span，判断中文是否真正渲染。"""
import fitz
from pathlib import Path

FIG = Path(r"E:\MathModel_pj-2026-sim2-B\outputs\figures")

for name in ["S2-stable-frequency", "S2-tau-sensitivity", "S2-cooccurrence-heatmap"]:
    doc = fitz.open(str(FIG / f"{name}.pdf"))
    print("=" * 70)
    print(name)
    page = doc[0]
    # 字体列表
    print("--- fonts ---")
    for f in page.get_fonts():
        print("  ", f)
    # 文本 span（含字体名）
    print("--- spans (text | font | size) ---")
    d = page.get_text("dict")
    for block in d["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                t = span["text"].strip()
                if t:
                    print(f"  [{span['font']}] {t!r}")
    print()

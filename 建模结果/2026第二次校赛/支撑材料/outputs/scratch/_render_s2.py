"""渲染 S2 三张正式图为 PNG 供自审。"""
import fitz
from pathlib import Path

FIG = Path(r"E:\MathModel_pj-2026-sim2-B\outputs\figures")
OUT = Path(r"E:\MathModel_pj-2026-sim2-B\outputs\scratch\_render_s2")
OUT.mkdir(exist_ok=True)

for name in ["S2-stable-frequency", "S2-tau-sensitivity", "S2-cooccurrence-heatmap"]:
    doc = fitz.open(str(FIG / f"{name}.pdf"))
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        pix.save(str(OUT / f"{name}-p{i}.png"))
    print(name, "pages:", len(doc))

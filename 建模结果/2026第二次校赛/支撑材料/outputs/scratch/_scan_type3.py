# -*- coding: utf-8 -*-
"""全量扫描 _explore 与正式图的 Type3 字体残留。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
from pypdf import PdfReader


def type3_names(folder):
    bad = []
    total = 0
    for p in sorted(folder.glob("*.pdf")):
        total += 1
        try:
            r = PdfReader(str(p))
            fonts = set()
            for pg in r.pages:
                f = pg.get("/Resources", {}).get("/Font", {})
                for v in f.values():
                    try:
                        fonts.add(str(v.get_object().get("/Subtype")))
                    except Exception:
                        pass
            if "/Type3" in fonts:
                bad.append(p.name)
        except Exception:
            bad.append(p.name + ":ERR")
    return bad, total


for label, folder in [
    ("_explore", Path(r"E:\MathModel_pj-2026-sim2-B\outputs\figures\_explore")),
    ("figures", Path(r"E:\MathModel_pj-2026-sim2-B\outputs\figures")),
]:
    bad, total = type3_names(folder)
    print(f"{label}: {len(bad)} Type3 of {total} -> {bad if bad else 'ALL FIXED'}")

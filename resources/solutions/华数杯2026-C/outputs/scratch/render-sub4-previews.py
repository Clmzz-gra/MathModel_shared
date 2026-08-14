# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""一次性：将 outputs/figures 下 sub4-*.pdf 渲染为 PNG 预览（scratch/sub4-preview/）。"""
import glob
import os

import fitz

BASE = r"e:\MathModel_pj-2026-C"
OUT = os.path.join(BASE, "outputs", "scratch", "sub4-preview")
os.makedirs(OUT, exist_ok=True)
for p in glob.glob(os.path.join(BASE, "outputs", "figures", "sub4-*.pdf")):
    d = fitz.open(p)
    pg = d[0]
    pix = pg.get_pixmap(dpi=110)
    out = os.path.join(OUT, os.path.basename(p).replace(".pdf", ".png"))
    pix.save(out)
    print(out)

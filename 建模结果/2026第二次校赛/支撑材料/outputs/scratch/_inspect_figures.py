# -*- coding: utf-8 -*-
"""图表审查提取工具：PyMuPDF 提取 20 张正式图的元数据（文本/元素数/颜色）。

用途：主会话图审（模型不支持图像输入，改以 PDF 结构提取 + 文本判读）。
用法：python outputs/scratch/_inspect_figures.py  → 写 outputs/scratch/_fig_summary.txt
"""
import os, fitz

FIGDIR = r"outputs/figures"
OUT = r"outputs/scratch/_fig_summary.txt"

FILES = [
    # S1 疾病预测（5）
    "S1-roc-curve.pdf", "S1-performance-compare.pdf", "S1-adenoma-sensitivity.pdf",
    "S1-feature-importance.pdf", "S1-threshold-analysis.pdf",
    # S2 特征选择（4）
    "S2-stable-frequency.pdf", "S2-tau-sensitivity.pdf", "S2-cooccurrence-heatmap.pdf",
    "S2-cross-disease.pdf",
    # S3 跨疾病（4）
    "S3-strategy-compare.pdf", "S3-decay-attribution.pdf", "S3-migration-direction.pdf",
    "S3-threshold-drift.pdf",
    # 数据特征（5）
    "chart-sample-composition.pdf", "chart-zero-sparsity.pdf", "chart-abundance-distribution.pdf",
    "chart-batch-effect.pdf", "chart-known-biomarker-presence.pdf",
    # 画像（2）
    "pca-scree.pdf", "cluster-tsne.pdf",
]


def summarize(path):
    doc = fitz.open(path)
    lines_all, colors, ndraw = [], set(), 0
    for page in doc:
        txt = page.get_text("text")
        lines_all.extend(l.strip() for l in txt.splitlines() if l.strip())
        for d in page.get_drawings():
            ndraw += 1
            for key in ("fill", "color"):
                if d.get(key):
                    colors.add(tuple(round(c, 2) for c in d[key]))
    return {"pages": doc.page_count, "n_draw": ndraw,
            "texts": lines_all[:60], "colors": sorted(colors)[:20]}


buf = []
for f in FILES:
    p = os.path.join(FIGDIR, f)
    if not os.path.exists(p):
        buf.append(f"== {f}: MISSING\n")
        continue
    try:
        s = summarize(p)
        buf.append(f"== {f} | pages={s['pages']} | n_draw={s['n_draw']} | colors={len(s['colors'])}\n")
        for t in s["texts"][:40]:
            buf.append(f"   T: {t[:90]}\n")
        if s["colors"]:
            buf.append(f"   C: {s['colors'][:12]}\n")
    except Exception as e:
        buf.append(f"== {f}: ERROR {e}\n")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.writelines(buf)
print(f"written {OUT}, {len(buf)} lines")

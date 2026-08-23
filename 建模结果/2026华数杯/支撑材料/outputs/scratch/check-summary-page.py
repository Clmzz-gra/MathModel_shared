"""
目的：
    测量摘要页(第1页)文本占用高度，判断重写后能否再容纳数据处理段

原理：
    PyMuPDF 遍历第1页文本块，取最大 y1（底部边界），与页面高度比较

输入数据：
    - COMP2026-C-final.pdf (编译产物) — 正式版论文

输出：
    控制台诊断信息

对应论文章节：
    摘要页（第一页）
"""
import fitz

doc = fitz.open(r"e:\MathModel_pj-2026-C\solution\final-paper\COMP2026-C-final.pdf")
page = doc[0]
h = page.rect.height
blocks = page.get_text("blocks")
blocks = [b for b in blocks if b[4].strip()]
max_y1 = max(b[3] for b in blocks)
print(f"页面高度: {h:.1f} pt")
print(f"第1页最底部文本块 y1: {max_y1:.1f} pt, 剩余 {h - max_y1:.1f} pt")
print(f"文本块数: {len(blocks)}")
# 打印最后3个文本块的y位置与内容摘要，确认摘要正文是否到底
for b in sorted(blocks, key=lambda x: x[1])[-3:]:
    print(f"  y[{b[1]:.0f}-{b[3]:.0f}] {b[4].strip()[:40]!r}")

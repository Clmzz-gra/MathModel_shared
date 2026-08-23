"""
目的：
    提取《2026年第七届华数杯数学建模竞赛论文格式规范与提交说明.pdf》全文文本，
    供格式合规审查与提交 checklist 整理使用

原理：
    PyMuPDF (fitz) 逐页抽取文本块，按页面顺序拼接为 Markdown，
    保留页号标记便于对照原文核查

输入数据：
    - 2026年第七届华数杯数学建模竞赛论文格式规范与提交说明.pdf (原始) — 官方 PDF

输出：
    - outputs/scratch/format-spec-extract.md — 全文 Markdown 文本

对应论文章节：
    提交规范（官方文件，非论文内容）
"""
import fitz

SRC = r"e:\MathModel_pj-2026-C\solutions\华数杯2026-C\2026年第七届华数杯数学建模竞赛论文格式规范与提交说明.pdf"
OUT = r"e:\MathModel_pj-2026-C\outputs\scratch\format-spec-extract.md"

doc = fitz.open(SRC)
lines = []
for i, page in enumerate(doc):
    lines.append(f"\n\n<!-- ===== 第 {i+1} 页 ===== -->\n")
    lines.append(page.get_text())
text = "".join(lines)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(text)
print(f"pages={doc.page_count}, chars={len(text)}")
print(f"saved: {OUT}")

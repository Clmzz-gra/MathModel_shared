"""
目的：
    提取大雅 AIGC 检测报告 PDF 的文本内容，便于定位 AI 率偏高的章节与段落

原理：
    检测报告为文本型 PDF，用 PyMuPDF (fitz) 逐页提取纯文本，按页写出，
    保留段落结构与百分比标注，供人工比对论文章节定位高 AI 率段落。

输入数据：
    - COMP2026-C-final_检测结果报告.pdf (原始检测报告) — 大雅 AIGC 检测输出

输出：
    - aigc-report.txt — 按页提取的文本（同一目录）

对应论文章节：
    终稿合规核验（AIGC 检测率 < 40%，章程第十六条）
"""
import fitz
import pathlib

report_dir = pathlib.Path(r"e:\MathModel_pj-2026-C\solution\final-paper\COMP2026-C-final_A_大雅AIGC检测报告")
pdf_path = report_dir / "COMP2026-C-final_检测结果报告.pdf"
out_path = report_dir / "aigc-report.txt"

doc = fitz.open(pdf_path)
lines = []
for i, page in enumerate(doc):
    lines.append(f"\n===== 第 {i+1} 页 =====\n")
    lines.append(page.get_text("text"))

out_path.write_text("".join(lines), encoding="utf-8")
print(f"pages: {doc.page_count}, chars: {sum(len(l) for l in lines)}")
print(f"written: {out_path}")

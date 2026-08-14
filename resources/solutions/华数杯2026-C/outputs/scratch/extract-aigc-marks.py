"""
目的：
    从大雅 AIGC 检测报告中提取被标注（疑似 AIGC）的文本片段及页码

原理：
    大雅检测报告把疑似 AIGC 片段以特殊颜色/下划线标注呈现。用 PyMuPDF 提取
    每个文本 span 的字体颜色与下划线标志，筛出非默认黑色的 span 即为疑似标注，
    汇总其所在页码与文本，供定位高 AI 率段落。

输入数据：
    - COMP2026-C-final_检测结果报告.pdf (原始检测报告)

输出：
    - aigc-marked-spans.txt — 疑似标注文本（页码 + span 内容）

对应论文章节：
    终稿合规核验（AIGC 检测率 < 40%，章程第十六条）
"""
import fitz
import pathlib

report_dir = pathlib.Path(r"e:\MathModel_pj-2026-C\solution\final-paper\COMP2026-C-final_A_大雅AIGC检测报告")
pdf_path = report_dir / "COMP2026-C-final_检测结果报告.pdf"
out_path = report_dir / "aigc-marked-spans.txt"

doc = fitz.open(pdf_path)
lines = []
for i, page in enumerate(doc):
    d = page.get_text("dict")
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                color = span.get("color", 0)
                flags = span.get("flags", 0)
                # flags bit 1 = underline
                underlined = bool(flags & 2)
                # 默认正文黑色 0x000000，标注通常为其他颜色（红/橙/蓝）
                is_colored = (color != 0)
                if (is_colored or underlined) and span.get("text", "").strip():
                    txt = span["text"].strip()
                    if len(txt) >= 4:
                        lines.append(
                            f"[p{i+1} color=0x{color:06x} ul={int(underlined)}] {txt}"
                        )

out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"marked spans: {len(lines)}")
print(f"written: {out_path}")

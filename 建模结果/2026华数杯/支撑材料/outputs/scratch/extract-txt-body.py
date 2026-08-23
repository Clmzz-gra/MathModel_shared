"""
目的：
    提取论文纯文本正文版本（含摘要），供 AIGC/查重检测工具使用

原理：
    正文版 PDF（COMP2026-C-正文版.pdf，仅 1-10 章）不含摘要页；
    从正式版 PDF 第 1 页取摘要文本，与正文版全文拼接为单一 txt。

输入数据：
    - COMP2026-C-正文版.pdf (编译产物) — 纯正文 1-10 章
    - COMP2026-C-final.pdf (编译产物) — 第 1 页为摘要页

输出：
    - solution/final-paper/COMP2026-C-正文含摘要.txt — 摘要 + 正文纯文本

对应论文章节：
    摘要页（第一页）与正文 1-10 章
"""
import fitz
import re

BASE = r"e:\MathModel_pj-2026-C\solution\final-paper"
body_pdf = BASE + r"\COMP2026-C-正文版.pdf"
final_pdf = BASE + r"\COMP2026-C-final.pdf"
out_txt = BASE + r"\COMP2026-C-正文含摘要.txt"

PAGE_RE = re.compile(r"^\d+$")  # 页脚纯页码行


def page_text(doc, i):
    lines = []
    for ln in doc[i].get_text().splitlines():
        s = ln.strip()
        if not s or PAGE_RE.match(s):
            continue
        lines.append(s)
    return "\n".join(lines)


# 摘要页：正式版第 1 页
final = fitz.open(final_pdf)
summary = page_text(final, 0)

# 正文：正文版全页
body_doc = fitz.open(body_pdf)
body = "\n\n".join(page_text(body_doc, i) for i in range(body_doc.page_count))

full = summary + "\n\n" + body + "\n"

with open(out_txt, "w", encoding="utf-8") as f:
    f.write(full)

print("输出:", out_txt)
print("总字符数:", len(full))
print("摘要页字符数:", len(summary))
print("正文版字符数:", len(body))
print("正文版页数:", body_doc.page_count)
print("开头:", full[:40].replace("\n", " "))
print("含问题重述:", "问题重述" in full, "| 含参考文献:", "参考文献" in full)

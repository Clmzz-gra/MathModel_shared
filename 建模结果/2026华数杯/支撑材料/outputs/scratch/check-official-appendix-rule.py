"""
目的：
    核对官方两份 PDF 中「附录是否参与查重 / AIGC 检测」的条款原文

原理：
    用 PyMuPDF 提取两份官方文档全文文本，grep 含「附录/查重/AIGC/检测」的
    段落并输出上下文，供人工核对附录是否计入 AIGC 统计。

输入数据：
    - 2026年第七届华数杯数学建模竞赛论文格式规范与提交说明.pdf (官方)
    - “华数杯”大学生数学建模竞赛人工智能工具使用章程.pdf (官方)

输出：
    - 控制台输出命中段落

对应论文章节：
    终稿合规核验（章程第十六条 AIGC < 40%）
"""
import fitz
import pathlib

base = pathlib.Path(r"e:\MathModel_pj-2026-C\solutions\华数杯2026-C")
files = [
    base / "2026年第七届华数杯数学建模竞赛论文格式规范与提交说明.pdf",
    base / "“华数杯”大学生数学建模竞赛人工智能工具使用章程.pdf",
]
keywords = ["附录", "查重", "AIGC", "检测", "人工智能生成", "AI生成", "AI 生成", "占比"]

for f in files:
    print(f"\n########## {f.name} ##########")
    doc = fitz.open(f)
    full = "\n".join(page.get_text("text") for page in doc)
    # 按句分割，保留含关键词的句子及其前后一句
    import re
    sents = re.split(r"(?<=[。；\n])", full)
    for i, s in enumerate(sents):
        if any(k in s for k in keywords):
            ctx = "".join(sents[max(0, i - 1): i + 2]).strip()
            if ctx:
                print(f"  >>> {ctx[:500]}")

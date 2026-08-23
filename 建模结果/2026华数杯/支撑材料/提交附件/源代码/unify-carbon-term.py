"""
目的：
    全文统一术语：将 final-paper 所有 .tex 中"碳排"（非"碳排放"）统一替换为"碳排放"

原理：
    "碳排"是"碳排放"的口语简称，论文中混用约 86 次。用正则负向前瞻 碳排(?!放)
    只匹配"碳排"后跟的不是"放"的片段，避免把已有"碳排放"替换成"碳碳排放"。
    替换次数累计后与替换前 Grep 统计总数（106 含碳排放本身）核对：
    替换后"碳排放"总数应等于替换前"碳排"总数（106），且无"碳排"残留。

输入数据：
    - solution/final-paper/*.tex（论文全部 tex，正文/表格/图注/摘要/附录）

输出：
    - 各文件替换次数与总计（stdout）；文件原地更新

对应论文章节：
    全文术语统一（用户要求的表述修正）
"""

import re
import pathlib

FINAL_PAPER = pathlib.Path(r"e:\MathModel_pj-2026-C\solution\final-paper")

pattern = re.compile(r"碳排(?!放)")

total = 0
for tex in sorted(FINAL_PAPER.glob("*.tex")):
    text = tex.read_text(encoding="utf-8")
    new_text, n = pattern.subn("碳排放", text)
    if n:
        tex.write_text(new_text, encoding="utf-8")
        print(f"{tex.name}: {n}")
        total += n

print(f"TOTAL_REPLACED: {total}")

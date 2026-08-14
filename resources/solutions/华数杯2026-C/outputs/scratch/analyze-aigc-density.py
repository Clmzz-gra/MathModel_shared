"""
目的：
    按检测报告页统计正文各章节的重度标注密度，确定降 AI 率改写优先级

原理：
    大雅报告第 5 页起为论文正文标注版。aigc-marked-body.txt 给出每页标注文本，
    aigc-report.txt 给出每页全部文本。逐页计算「标注字符数 / 总字符数」，
    并按报告页映射到论文章节（报告页为检测系统重排版，章节边界由文本特征定位）。

输入数据：
    - aigc-report.txt (处理后) — 每页全文
    - aigc-marked-body.txt (处理后) — 每页标注文本

输出：
    - 控制台输出每页字符总量、标注量、密度及章节映射

对应论文章节：
    终稿合规核验（AIGC 检测率 < 40%，章程第十六条）
"""
import pathlib
import re

report_dir = pathlib.Path(r"e:\MathModel_pj-2026-C\solution\final-paper\COMP2026-C-final_A_大雅AIGC检测报告")
report_path = report_dir / "aigc-report.txt"
marks_path = report_dir / "aigc-marked-body.txt"

# 每页全文字符数（从 aigc-report.txt）
page_chars = {}
current = None
for line in open(report_path, encoding="utf-8"):
    m = re.match(r"===== 第 (\d+) 页 =====", line)
    if m:
        current = int(m.group(1))
        page_chars[current] = 0
    elif current:
        page_chars[current] += len(line.strip())

# 每页标注字符数（从 aigc-marked-body.txt）
mark_chars = {}
current = None
for line in open(marks_path, encoding="utf-8"):
    m = re.match(r"===== 论文标注版第 (\d+) 页 =====", line)
    if m:
        current = int(m.group(1))
        mark_chars[current] = 0
    elif current:
        mark_chars[current] += len(line.strip())

# 章节边界（按报告页，依据正文顺序推断）
sections = [
    (5, 6, "摘要页/UI"),
    (6, 8, "摘要 + 问题重述 + 数据预处理(起)"),
    (8, 9, "数据预处理 + 假设符号 + S1(起)"),
    (9, 11, "S1 问题一"),
    (11, 13, "S2 问题二"),
    (13, 19, "S3 问题三"),
    (19, 24, "S4 问题四"),
    (24, 26, "敏感性分析"),
    (26, 27, "模型评价"),
    (27, 28, "结论 + AI声明 + 参考文献 + 附录代码清单"),
    (28, 29, "附录 AI 工具使用详情"),
    (29, 31, "附录补充图表"),
]

print(f"{'报告页':<8}{'章节':<38}{'总字符':>8}{'标注':>8}{'密度%':>8}")
total_chars = total_marks = 0
for lo, hi, name in sections:
    tc = sum(page_chars.get(p, 0) for p in range(lo, hi + 1))
    mc = sum(mark_chars.get(p, 0) for p in range(lo, hi + 1))
    total_chars += tc
    total_marks += mc
    dens = mc / tc * 100 if tc else 0
    print(f"p{lo}-{hi:<6}{name:<38}{tc:>8}{mc:>8}{dens:>7.1f}%")
print(f"\n合计(第5页起正文部分): 总 {total_chars} / 标注 {total_marks} / 密度 {total_marks/total_chars*100:.1f}%")

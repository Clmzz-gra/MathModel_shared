"""
目的：
    统计大雅 AIGC 检测报告中论文正文部分（第 5 页起）疑似标注的分布

原理：
    aigc-marked-spans.txt 记录了所有疑似标注 span（页码+颜色+文本）。
    检测报告前 4 页为报告 UI（片段概率列表），第 5 页起为论文正文标注版。
    按页统计标注 span 数与字符数，输出标注最集中的页及对应论文页码范围，
    并抽样输出被标注的正文片段用于人工定位高 AI 率章节。

输入数据：
    - aigc-marked-spans.txt (处理后) — 标注 span 列表

输出：
    - 控制台统计 + aigc-marked-body.txt（正文部分标注文本，按页分组）

对应论文章节：
    终稿合规核验（AIGC 检测率 < 40%，章程第十六条）
"""
import pathlib
import re

report_dir = pathlib.Path(r"e:\MathModel_pj-2026-C\solution\final-paper\COMP2026-C-final_A_大雅AIGC检测报告")
marks_path = report_dir / "aigc-marked-spans.txt"
out_path = report_dir / "aigc-marked-body.txt"

per_page = {}
with open(marks_path, encoding="utf-8") as f:
    for line in f:
        m = re.match(r"\[p(\d+) color=0x[0-9a-f]+ ul=(\d)\]\s*(.*)", line.strip())
        if not m:
            continue
        page = int(m.group(1))
        txt = m.group(3).strip()
        if page >= 5:  # 论文正文标注版
            per_page.setdefault(page, []).append(txt)

out_lines = []
for page in sorted(per_page):
    spans = per_page[page]
    chars = sum(len(t) for t in spans)
    out_lines.append(f"\n===== 论文标注版第 {page} 页 =====（span {len(spans)} 条 / 字符 {chars}）")
    # 合并连续 span 成行（标注文本通常连续）
    joined = "".join(spans)
    out_lines.append(joined)

out_path.write_text("\n".join(out_lines), encoding="utf-8")

print(f"正文标注页范围: {min(per_page)} - {max(per_page)}")
stats = sorted(((sum(len(t) for t in v), k) for k, v in per_page.items()), reverse=True)
print("标注字符数最多的页（Top 15）:")
for chars, page in stats[:15]:
    print(f"  第 {page} 页: {chars} 字符")

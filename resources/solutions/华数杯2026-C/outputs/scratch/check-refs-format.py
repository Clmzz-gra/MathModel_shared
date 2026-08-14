"""
目的：
    检查 final-paper 各章节 tex 的 \label 与 \ref 匹配、图表编号连续性、数字千分位格式

原理：
    正则扫描所有 .tex：收集 label 定义与 ref 引用，对比找出未定义引用与未使用 label；
    检查裸数字逗号千分位（应使用 {,}）；统计 table/figure 计数

输入数据：
    - solution/final-paper/*.tex (标准化)

输出：
    控制台打印检查结果

对应论文章节：
    全文
"""
import re, glob, os

base = r"e:\MathModel_pj-2026-C\solution\final-paper"
texs = [f for f in glob.glob(os.path.join(base, "*.tex"))
        if os.path.basename(f) not in ("COMP2026-C-正文版.tex",)]

labels = {}
refs = {}
tab_count = fig_count = 0

for f in texs:
    src = open(f, encoding="utf-8").read()
    for m in re.finditer(r"\\label\{([^}]+)\}", src):
        labels[m.group(1)] = os.path.basename(f)
    for m in re.finditer(r"\\ref\{([^}]+)\}", src):
        refs.setdefault(m.group(1), set()).add(os.path.basename(f))
    tab_count += len(re.findall(r"\\begin\{table\}", src))
    fig_count += len(re.findall(r"\\begin\{figure\}", src))

print(f"== 检查文件 {len(texs)} 个，table {tab_count}，figure {fig_count} ==")
missing = [r for r in refs if r not in labels]
print("\n[未定义引用 ref 无 label]")
for r in missing:
    print(f"  {r}  引用自 {sorted(refs[r])}")

unused = [l for l in labels if l not in refs]
print("\n[定义但从未引用的 label]")
for l in unused:
    print(f"  {l}  ({labels[l]})")

print("\n[裸数字千分位逗号（如 50,000 应为 50{,}000）]")
pat = re.compile(r"(?<![{A-Za-z_\\-])\d,\d{3}(?!\d)")
for f in texs:
    src = open(f, encoding="utf-8").read()
    for i, line in enumerate(src.splitlines(), 1):
        for m in pat.finditer(line):
            print(f"  {os.path.basename(f)}:{i}  ...{line[max(0,m.start()-25):m.end()+10]}...")

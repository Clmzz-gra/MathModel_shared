"""
目的：
    扫描 final-paper 正文章节，找出「两三个字符独立成句」的破碎短句

原理：
    去掉注释与数学内容后按中文句号/叹号/问号切分，统计每句中文字符数，输出短句候选

输入数据：
    - solution/final-paper/chapter*.tex (标准化)

输出：
    控制台打印短句候选（按句长升序）

对应论文章节：
    全文
"""
import re, glob, os

base = r"e:\MathModel_pj-2026-C\solution\final-paper"
files = sorted(glob.glob(os.path.join(base, "chapter*.tex")))

def clean_line(line):
    # 去注释（% 后内容，保留 % 转义）
    line = re.sub(r"(?<!\\)%.*$", "", line)
    # 去数学 $...$
    line = re.sub(r"\$[^$]*\$", " ", line)
    # 去 \command{...}
    line = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", line)
    # 去 \command
    line = re.sub(r"\\[a-zA-Z]+", " ", line)
    return line

found = []
for f in files:
    src = open(f, encoding="utf-8").read()
    for line in src.splitlines():
        cl = clean_line(line).strip()
        if not cl:
            continue
        # 按句号切分
        parts = re.split(r"(?<=[。！？])", cl)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # 中文字符数
            cn = len(re.findall(r"[\u4e00-\u9fff，、：；]", p))
            if cn <= 10:
                found.append((cn, os.path.basename(f), p))

found.sort()
print(f"共 {len(found)} 个短句候选（≤10 中文字符）\n")
for cn, fn, p in found:
    print(f"[{cn}字] {fn}: {p}")

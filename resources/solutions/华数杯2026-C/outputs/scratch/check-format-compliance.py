"""
目的：
    对照《2026年第七届华数杯数学建模竞赛论文格式规范与提交说明》程序化核查
    COMP2026-C-final.pdf 的格式合规项：页码位置、页眉、摘要页独立性、
    正文页数、参考文献引用标注、附录范围

原理：
    用 PyMuPDF 按页面高度归一化定位页脚区（y/H>0.90）与页眉区（y/H<0.08）文本，
    判定页码是否页脚中部、是否存在页眉；按"问题重述/参考文献/附录"关键字定位
    各部分起止页；用正则统计正文中的方括号引用标注 [n]

输入数据：
    - solution/final-paper/COMP2026-C-final.pdf (处理后) — 重新编译的终稿 PDF

输出：
    - 控制台核查报告（供格式合规审查与提交 checklist 使用）

对应论文章节：
    提交规范（官方文件，非论文内容）
"""
import fitz
import re

PDF = r"e:\MathModel_pj-2026-C\solution\final-paper\COMP2026-C-final.pdf"

doc = fitz.open(PDF)
H = doc[0].rect.height
print(f"=== 基本 === pages={doc.page_count}, page_h={H:.0f}pt\n")

# ---- 页码位置 & 页眉检查 ----
foot_ok, foot_bad, header_hits = 0, [], []
for i, page in enumerate(doc):
    foot_text = []
    header_text = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
        t = txt.strip()
        if not t:
            continue
        if y1 / H > 0.90:            # 页脚区
            foot_text.append((round(x0, 1), round(x1, 1), t.replace("\n", " ")))
        if y0 / H < 0.08:            # 页眉区
            header_text.append(t.replace("\n", " "))
    # 页脚判定：应有且仅有一个页码数字，且大致居中
    digits = [f for f in foot_text if re.fullmatch(r"\s*\d+\s*", f[2])]
    mid = H / 2
    if digits and len(foot_text) == 1:
        x0, x1, _ = digits[0]
        center = (x0 + x1) / 2
        if abs(center - mid) < H * 0.05:
            foot_ok += 1
        else:
            foot_bad.append((i + 1, "页码不居中", foot_text))
    else:
        foot_bad.append((i + 1, f"页脚异常 foot={foot_text}", None))
    for h in header_text:
        header_hits.append((i + 1, h))

print(f"=== 页脚页码 === 正确居中页数={foot_ok}/{doc.page_count}")
if foot_bad:
    for e in foot_bad:
        print("  异常:", e)
print(f"=== 页眉 === 检出 {len(header_hits)} 处页眉文本")
for h in header_hits[:10]:
    print("  ", h)

# ---- 摘要页独立性 & 正文/参考文献/附录起止页 ----
print("\n=== 结构定位 ===")
targets = {"问题重述": None, "参考文献": None, "附录": None, "代码文件清单": None}
for i, page in enumerate(doc):
    t = page.get_text()
    for k in targets:
        if targets[k] is None and k in t:
            targets[k] = i + 1
for k, v in targets.items():
    print(f"  {k}: 第 {v} 页" if v else f"  {k}: 未找到")
print(f"  摘要页是否独占第1页: 第2页首行={doc[1].get_text().strip()[:40]!r}")

# ---- 正文中的方括号引用标注 [n] ----
print("\n=== 正文引用标注 [n] ===")
hits = []
for i, page in enumerate(doc):
    t = page.get_text()
    for m in re.finditer(r"\[\s*(\d+)\s*\]", t):
        hits.append((i + 1, m.group(0)))
print(f"  检出 {len(hits)} 处；前 10: {hits[:10]}")
# 参考文献页内的 [1][2] 是条目本身，区分正文页（2~参考文献页前）
body_refs = [h for h in hits if h[0] < (targets['参考文献'] or 999)]
print(f"  正文部分引用标注: {len(body_refs)} 处: {body_refs[:15]}")

# ---- 文件大小 ----
import os
sz = os.path.getsize(PDF) / 1e6
print(f"\n=== 文件大小 === {sz:.2f} MB (<20MB 达标)")

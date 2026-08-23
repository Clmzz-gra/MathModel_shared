# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
arch-sub1s3.py — S1–S3 跨模型架构图（阶段 3.0 round-1）

目的：绘制 S1/S2/S3/S4 子问题间的数据流与控制流架构图，供跨子问题审查与论文使用。

原理：节点 = 子问题模型；箭头 = 数据/框架/基线传递（实线=已实现依赖，虚线=待 S4 集成）。

输入映射：无外部数据（静态架构信息）。

输出：outputs/figures/00-architecture.pdf + solution/artifacts/charts/ 副本。

论文章节：阶段 3 跨模型架构图。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(9, 6))

nodes = {
    "S1": (0.12, 0.62, "S1 预测+基础调度\n零迁移 MILP\n(已定稿)"),
    "S2": (0.45, 0.62, "S2 碳感知调度\n迁移 + 时间维 MILP\n(已定稿)"),
    "S3": (0.45, 0.22, "S3 储能协同优化\n受限消纳 LP\n(已定稿)"),
    "S4": (0.78, 0.42, "S4 全系统协同\n算-储-电一体化\n(已定稿)"),
}

for key, (x, y, label) in nodes.items():
    box = FancyBboxPatch((x - 0.09, y - 0.09), 0.18, 0.18,
                         boxstyle="round,pad=0.015", linewidth=1.5,
                         edgecolor="#0d47a1", facecolor="#e3f2fd")
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=9, color="#0d47a1")

arrows = [
    ("S1", "S2", "框架复用 + 零迁移基线\n(C0=333.3M/E0=374.28kt)", "solid"),
    ("S1", "S4", "预测序列/调度框架", "solid"),
    ("S2", "S4", "任务-区域分配/碳排分量", "solid"),
    ("S3", "S4", "储能策略/受限消纳口径", "solid"),
    ("S2", "S3", "无数据依赖\n(问题三口径固定)", "dotted"),
]
pos = {k: (v[0], v[1]) for k, v in nodes.items()}
for src, dst, label, style in arrows:
    (x1, y1), (x2, y2) = pos[src], pos[dst]
    if src == "S2" and dst == "S3":
        x1, y1 = x1 - 0.08, y1 - 0.08
        x2, y2 = x2 + 0.05, y2 + 0.08
    elif src == "S1" and dst == "S2":
        x1, y1 = x1 + 0.08, y1 + 0.03
        x2, y2 = x2 - 0.08, y2 + 0.03
    else:
        x1, y1 = x1 + 0.08, y1
        x2, y2 = x2 - 0.08, y2
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                            mutation_scale=14, linewidth=1.4,
                            color="#1a1a1a", linestyle=style)
    ax.add_patch(arrow)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mx, my + 0.03, label, ha="center", va="bottom",
            fontsize=7, color="#555555")

ax.set_xlim(0, 1); ax.set_ylim(0, 0.9)
ax.axis("off")
ax.set_title("2026-C 跨子问题模型架构（S1–S4 全部定稿）", fontsize=12)
plt.tight_layout()
plt.savefig(r"E:\MathModel_pj-2026-C\outputs\figures\00-architecture.pdf")
print("arch diagram saved")

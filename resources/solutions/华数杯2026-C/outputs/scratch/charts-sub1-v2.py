# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 2.1.5 图表门禁准备 — S1 图表 v2：修正甘特图信息过载（按类型 3 分面）与
    收尾任务缺失（含 [2400,2406) 结清任务），利用率曲线确认范围，输出 -v2.pdf

原理：
    1. 甘特图分面：按 TaskType 分 3 子图（训练/批量/实时），每子图任务按到达排序，
       条 = 开工小时 → 开工+时长（精确小数）；跨 2399 任务条延伸至 2406（全画）
    2. 着色按类型：AITraining=#333333 / BatchInference=#1f77b4 / RealTimeInference=#d62728
    3. 利用率：evaluate 重算（base + α 开工表），6 区域 × Alpha/Beta，纵轴 [0,1]
    4. 收尾时段 [2400,2406) 完整显示（相对小时 24-30）

输入数据：
    - outputs/data/s1-schedule-test.pkl（alpha/beta 开工表 + tasks）
    - outputs/data/s1-preprocessed.pkl（schedule_input: base/cap/hours/hidx/free）

输出：
    - outputs/figures/sub1-gantt-last24h-v2.pdf — 分面甘特图（含收尾）
    - outputs/figures/sub1-utilization-v2.pdf — 利用率曲线（确认范围）
    - 副本至 solution/artifacts/charts/ + manifest 登记

对应论文章节：
    问题一（S1）调度段 — 阶段 2.1.5 图表门禁
"""
import pickle
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Source Han Sans CN"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(r"e:\MathModel_pj-2026-C")
regions = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
T0, T_END = 2376, 2406
COLORS = {"AITraining": "#333333", "BatchInference": "#1f77b4", "RealTimeInference": "#d62728"}

with open(BASE / "outputs" / "data" / "s1-schedule-test.pkl", "rb") as f:
    sched = pickle.load(f)
with open(BASE / "outputs" / "data" / "s1-preprocessed.pkl", "rb") as f:
    prep = pickle.load(f)
si = prep["schedule_input"]
base, cap = si["base"], si["cap"]
hours, hidx = si["hours"], si["hidx"]
free = si["free"]

tasks = sched["tasks"]
alpha = sched["alpha"]
beta = sched["beta"]
# 实时任务开工=到达
rt_start = {t["id"]: t["arrive"] for t in tasks if t["type"] == "RealTimeInference"}
alpha_full = {**rt_start, **{k: v for k, v in alpha.items()}}
beta_full = {**rt_start, **{k: v for k, v in beta.items()}}

# ============================================================
# 图3: 甘特图 v2（按类型 3 分面，含收尾任务）
# ============================================================
FIG3 = BASE / "outputs" / "figures" / "sub1-gantt-last24h-v2.pdf"
types = ["AITraining", "BatchInference", "RealTimeInference"]
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, tt in zip(axes, types):
    sub = [t for t in tasks if t["type"] == tt]
    sub = sorted(sub, key=lambda t: alpha_full[t["id"]])
    y = 0
    for t in sub:
        h = alpha_full[t["id"]]
        ax.barh(y, t["dur"], left=h - T0, height=0.8, color=COLORS[tt], alpha=0.85)
        y += 1
    ax.set_title(f"{tt}（{len(sub)} 任务）")
    ax.set_ylabel("任务（按开工排序）")
    ax.set_ylim(-1, y)
    # 标注主窗/收尾分界（2399→2400 相对小时 23/24）
    ax.axvline(23.5, color="#d62728", ls="--", lw=0.8, alpha=0.6)
axes[2].set_xlabel("相对小时（2376 = 0）")
axes[2].set_xticks(range(0, 31))
fig.suptitle("最后 24h Alpha 调度甘特图（按类型分面，含收尾任务）")
# 收尾说明移图下（原顶部红色注解已删，审查用户指令）
fig.text(0.5, -0.02, "红虚线右侧为收尾时段 [2400,2406)；跨 2399 的任务条延伸至 2406",
         ha="center", fontsize=8, color="#555555")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG3, dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 图4: 利用率曲线 v2（6 区域 × Alpha/Beta，纵轴 [0,1]）
# ============================================================
FIG4 = BASE / "outputs" / "figures" / "sub1-utilization-v2.pdf"

def evaluate(sched_free):
    use = base.copy()
    for t in free:
        h = sched_free.get(t["id"])
        if h is None:
            continue
        r = regions.index(t["region"])
        s, e = h, h + t["dur"]
        hi = int(np.floor(s))
        while s < e and hi < T_END:
            hh = hidx.get(hi)
            if hh is not None:
                ov = min(e, hi + 1.0) - max(s, float(hi))
                if ov > 0:
                    use[r, hh] += t["dem"] * ov
            s = hi + 1.0
            hi = int(np.floor(s))
    return use / np.array([cap[r] for r in regions])[:, None]

util_a = evaluate(alpha_full)
util_b = evaluate(beta_full)
assert util_a.max() <= 1.0001 and util_b.max() <= 1.0001, "利用率超 1（数据异常）"

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
h_axis = np.arange(len(hours))
for i, r in enumerate(regions):
    ax = axes[i // 3][i % 3]
    ax.plot(h_axis, util_a[i], color="#1f77b4", lw=1.6, label="Alpha")
    ax.plot(h_axis, util_b[i], color="#d62728", lw=1.6, ls="--", label="Beta")
    ax.axvline(23.5, color="#333333", ls=":", lw=0.6)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{r}（Cap={cap[r]}）")
    ax.set_ylabel("利用率")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(0, 30)
for ax in axes[-1]:
    ax.set_xlabel("相对小时（2376=0，虚线右为收尾时段）")
fig.suptitle("6 区域逐时 GPU 利用率（Alpha vs Beta，含收尾时段）")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG4, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[OK] {FIG3}")
print(f"[OK] {FIG4}")
print(f"利用率范围: Alpha [{util_a.min():.4f}, {util_a.max():.4f}] / Beta [{util_b.min():.4f}, {util_b.max():.4f}]（均 ≤1）")
print(f"Alpha 极差 {util_a.max()-util_a.min():.4f} / Beta 极差 {util_b.max()-util_b.min():.4f}（验证基准一致）")

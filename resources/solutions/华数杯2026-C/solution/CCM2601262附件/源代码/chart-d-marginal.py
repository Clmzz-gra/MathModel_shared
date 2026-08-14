# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 2.1.6 S4 降碳成本拐点图 — D 区（唯一高弹性区）碳约束收紧时的
    成本-碳排 Pareto 曲线 + 逐段边际降碳成本曲线，定位"前段几乎免费、
    末段边际成本骤增"的拐点，量化碳约束的有效降碳区间。

原理：
    只读 s4-d-eps-scan.pkl（阶段 2.1.6 细网格扫描，33 点，先算后画 PR-014）。
    边际成本按"从 ε=1.00 向下收紧"方向计算：
        MC_i = (C(ε_{i-1}) − C(ε_i)) / (E(ε_{i-1}) − E(ε_i))  [元/吨 CO2]
    数据形态（实测）：ε ∈ [0.95, 1.00] 成本仅 +0.8%（边际 ~100-200 元/吨，
    任务时移+储能吃新能源，几乎免费）；ε < 0.94 后边际成本跨越数量级
    （>1000 元/吨，须动用高价购电），形成清晰拐点。
    左面板：成本(M) vs 碳排(kt) — L 形 Pareto 前沿（横轴碳排从低到高，
    即从 ε_min 到 ε=1.00；注意成本轴方向：ε_min 端成本高、ε=1.00 端低）。
    右面板：边际成本(元/吨) vs 累计降碳比例(%) — 线性轴显示前段贴地、
    后段飙升；标注拐点 ε≈0.94 与有效降碳区间（约前 6%）。

输入数据：
    - outputs/data/s4-d-eps-scan.pkl（阶段 2.1.6）— eps_grid/cost/carbon/
      feasible/e0_kt/eps_min
    - 中文指标 → 变量名映射：碳排→carbon(kt), 成本(M元)→cost(M)

输出：
    - outputs/figures/sub4-d-marginal-cost.pdf — 双面板矢量图
    - 控制台边际成本序列核对（PR-014）

对应论文章节：
    问题四（S4）— 碳约束灵敏度 / 降碳成本拐点分析
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === 中文字体与负号（chart-generator 强制前置）===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                   'WenQuanYi Micro Hei', 'Source Han Sans CN']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"
FIG = BASE / "outputs" / "figures"

C_MAIN = "#333333"
C_BLUE = "#4c78a8"
C_RED = "#c14b3a"
C_GRAY = "#999999"


def main():
    d = pickle.load(open(DATA / "s4-d-eps-scan.pkl", "rb"))
    eps = np.array(d["eps_grid"])
    cost = np.array(d["cost"])
    carb = np.array(d["carbon"])
    feas = np.array(d["feasible"], dtype=bool)
    e0 = d["e0_kt"]
    emin = d["eps_min"]

    # 只保留可行点；按 ε 升序（ε_min → 1.00）
    m = feas & ~np.isnan(cost)
    eps, cost, carb = eps[m], cost[m], carb[m]

    # 边际成本：从 ε=1.00 向下收紧方向（ε 升序数组，倒序遍历）
    steps = []
    for i in range(len(eps) - 1, 0, -1):      # 从最松到最紧
        dC = cost[i - 1] - cost[i]            # 收紧一步的成本增加（M 元）
        dE = carb[i] - carb[i - 1]            # 收紧一步的碳排减少（kt）
        if dE > 1e-9:
            steps.append({
                "eps": (eps[i - 1] + eps[i]) / 2,
                "dC": dC, "dE": dE,
                "mc": dC * 1e3 / dE,                       # 元/吨 CO2
                "cum_cut": (carb[i] - carb[i - 1]) / e0 * 100,  # 该步降碳 % E0
            })
    mc = np.array([s["mc"] for s in steps])
    cum_cut = np.array([s["cum_cut"] for s in steps])
    mc_eps = np.array([s["eps"] for s in steps])

    print("=== 边际成本序列（从 ε=1.00 收紧）===")
    print(f"{'ε(步中点)':>9} {'Δ成本M':>8} {'Δ碳kt':>7} {'MC 元/吨':>9}")
    for s in steps:
        print(f"{s['eps']:9.4f} {s['dC']:+8.3f} {s['dE']:+7.2f} {s['mc']:9.0f}")
    print(f"  拐点搜索：MC 首次 >1000 元/吨 的 ε ≈ {mc_eps[np.argmax(mc > 1000)]:.4f}")
    # 有效降碳区间（MC ≤ 1000 元/吨 的累计降碳）
    cheap = mc <= 1000
    cheap_cut = float(np.sum(cum_cut[cheap]))
    print(f"  有效降碳区间（MC≤1000 元/吨）累计 ≈ {cheap_cut:.1f}% of E0")

    # ---- 双面板绘图 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.2))

    # 左：成本-碳排 L 形 Pareto（x 轴碳排，从左 ε_min 到右 ε=1.00）
    k = int(np.argmin(np.abs(eps - 0.94)))
    ax1.plot(carb, cost, color=C_MAIN, lw=1.8, marker="o", ms=3,
             label="成本-碳排曲线")
    # 免费/昂贵分区（图例表达，不再图内文字）
    ax1.fill_between([carb[k], carb[-1]], 130, 200, color=C_BLUE, alpha=0.12,
                     label="免费降碳区（MC≤200 元/吨）")
    ax1.fill_between([carb[0], carb[k]], 130, 200, color=C_RED, alpha=0.12,
                     label="昂贵降碳区（MC>1000 元/吨）")
    # 拐点特殊点标注
    ax1.scatter(carb[k], cost[k], marker="D", s=42, color="#e8a33d",
                edgecolors="#1a1a1a", linewidths=0.8, zorder=5,
                label=f"拐点 $\\varepsilon$≈0.94（{float(carb[k]):.0f} kt）")
    # 端点数值注解（无箭头：红点右侧、蓝点上方）
    ax1.set_ylim(135, 184)
    ax1.set_xlim(float(carb[0]) - 0.3, float(carb[-1]) + 0.3)  # 收紧到红蓝区
    ax1.text(carb[0] + 1.2, cost[0] - 2, f"$\\varepsilon$={emin:.3f}\n"
             f"{float(cost[0]):.0f}M / {float(carb[0]):.0f}kt",
             fontsize=8, color=C_RED, ha="left", va="top")
    ax1.text(carb[-1] - 2.0, cost[-1] + 5, f"$\\varepsilon$=1.00\n"
             f"{float(cost[-1]):.0f}M / {float(carb[-1]):.0f}kt",
             fontsize=8, color=C_BLUE, ha="center", va="bottom")
    ax1.set_xlabel("碳排放（kt，主时域）")
    ax1.set_ylabel("运行成本（M 元）")
    ax1.set_title("(a) D 区运行成本与碳排放权衡曲线（Pareto 前沿）",
                  fontsize=10, pad=30)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False,
               fontsize=8, ncol=2)
    ax1.grid(linestyle=":", linewidth=0.3, alpha=0.4)

    # 右：边际成本 vs 累计降碳比例
    cum_axis = np.cumsum(cum_cut)  # 累计降碳 %
    kk = int(np.argmax(mc > 1000))
    ax2.plot(cum_axis, mc, color=C_RED, lw=1.8, marker="s", ms=3,
             label="边际降碳成本 MC($\\varepsilon$)")
    ax2.axhline(1000, color=C_GRAY, ls="--", lw=0.9, label="1000 元/吨 参考线")
    ax2.scatter(cum_axis[kk], mc[kk], marker="D", s=42, color="#e8a33d",
                edgecolors="#1a1a1a", linewidths=0.8, zorder=5,
                label=f"拐点（累计降碳 {cum_axis[kk]:.1f}%）")
    ax2.set_xlabel("累计降碳比例（%）")
    ax2.set_ylabel("边际降碳成本（元/吨 CO$_2$）")
    ax2.set_title("(b) 边际降碳成本随累计降碳比例的变化", fontsize=10, pad=30)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False,
               fontsize=8, ncol=2)
    ax2.grid(linestyle=":", linewidth=0.3, alpha=0.4)

    fig.suptitle("D 区碳约束收紧下的降碳成本分析"
                 "（$\\varepsilon_{\\min}$=%.4f，基准碳排 %.0f kt）" % (emin, e0),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "sub4-d-marginal-cost.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ 已保存 {FIG / 'sub4-d-marginal-cost.pdf'}")
    print("S4 D-MARGINAL-COST DONE")


if __name__ == "__main__":
    main()

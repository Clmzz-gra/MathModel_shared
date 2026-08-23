# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 2.1.5 S4 图表生成 — 从 s4-results.pkl 绘制问题四可视化
    （场景对比 + 储能/购售电时序 + 基荷覆盖），供图表审查门禁。

原理：
    先算后画（PR-014）：所有数值已在 sub4-model.py 求解落盘
    （s4-results.pkl），本脚本只读结果、统计量打印核对后绘图。
    图表与论文章节映射（math-sub4.tex §7 场景设计 / §6 六指标）：
    - fig1 场景对比汇总：6 场景 × 成本/碳排 双轴柱状 → 论文"不同碳约束、
      电价机制与新能源波动场景下的策略变化"（问题四核心）
    - fig2 六区域成本-碳排分解：S0-main 每区购电成本 + 碳排 → "多区域
      协同"空间结构
    - fig3 储能 SOC + 充放（D 区示例 + 全局）：D 是唯一电量受限区（基荷
      17.3% 填充），SOC 曲线展示储能协同价值
    - fig4 新能源利用率双口径：受限消纳下 (R+Cr)/Avail vs (R+Cr+S)/Avail
      → 与 S3 20.83% 口径衔接（衔接风险 E7）
    - fig5 峰值净购电逐区：max(G-S) → "区域峰值净购电"指标
    - fig6 基荷策略贡献：基荷 GPU-hour 占比 vs 非基荷 → 基荷预填 95.3%
      压缩变量空间的量化（Q3 方案 A/B 对比前置）

输入数据：
    - outputs/data/s4-results.pkl（阶段 2.1）
      main: {sols: {Region: {G/S/R/Cg/Cr/D/E/net/cost/carbon/peak/std/util}}}
      scenarios: {eps-0.95/eps-0.9/price-1.5/renew-1.2/renew-0.8: {同上}}
      e0_s4_kt / s3_carbon_ref_kt / ref_A
    - 中文指标 → 变量名映射：
      购电功率→G, 卖电→S, 新能源直供→R, 新能源充电→Cr, 放电→D,
      SOC→E, 净购电→net=G-S, 成本(M元)→cost_main_M, 碳排(kt)→carbon_kt

输出：
    - outputs/figures/sub4-*.pdf（6 图，矢量，SimHei，去饱和配色）
    - 控制台统计量（min/max/mean/std，PR-014 核对）

对应论文章节：
    问题四（S4）算-储-电协同优化 — 阶段 2.1.5 图表
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
# mathtext 用 Computer Modern 字体集（与 LaTeX 论文正文数学风格一致）
plt.rcParams['mathtext.fontset'] = 'cm'

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"
FIG = BASE / "outputs" / "figures"
FIG.mkdir(exist_ok=True)

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
# 去饱和配色（chart-generator：灰主调 + 辅助色 ≤2）
C_MAIN = "#333333"
C_BLUE = "#1f77b4"
C_RED = "#d62728"
C_GREEN = "#2ca02c"

# 区域标签缩写（图内空间有限）
RLAB = {"RegionA": "A", "RegionB": "B", "RegionC": "C",
        "RegionD": "D", "RegionE": "E", "RegionF": "F"}

SCEN_LABEL = {
    "S0-main": "S0 基准 ($\\varepsilon$=1.00)",
    "eps-0.95": "$\\varepsilon$=0.95",
    "eps-0.9": "$\\varepsilon$=0.90",
    "price-1.5": "峰谷差×1.5",
    "renew-1.2": "新能源+20%",
    "renew-0.8": "新能源-20%",
}


def load():
    with open(DATA / "s4-results.pkl", "rb") as f:
        return pickle.load(f)


def fig1_scenario_compare(d):
    """场景对比：成本+碳排双柱（问题四核心结论图）。
    部分可行场景（如 ε=0.95 仅 D 区可行）不加图中注解，避免干扰。"""
    print("\n=== fig1 场景对比统计 ===")
    names = ["S0-main"] + [k for k in d["scenarios"] if d["scenarios"][k].get("feasible")]
    costs, carbons = [], []
    for n in names:
        agg = d["scenarios"][n] if n != "S0-main" else d["main"]
        if not agg.get("feasible"):
            costs.append(np.nan)
            carbons.append(np.nan)
            continue
        costs.append(agg["cost_main_M"])
        carbons.append(agg["carbon_kt"])
        print(f"  {n}: 成本 {agg['cost_main_M']:.2f}M | 碳 {agg['carbon_kt']:.2f}kt"
              f" | 不可行区 {agg.get('n_infeasible', 0)}/6")

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(names))
    w = 0.35
    b1 = ax1.bar(x - w / 2, costs, w, color=C_BLUE, label="运行成本（M 元）")
    ax1.set_ylabel("运行成本（M 元）", color=C_BLUE)
    ax1.tick_params(axis="y", labelcolor=C_BLUE)
    ax1.set_xticks(x)
    ax1.set_xticklabels([SCEN_LABEL.get(n, n) for n in names], rotation=15, fontsize=8)
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w / 2, carbons, w, color=C_RED, alpha=0.7, label="碳排放（kt）")
    ax2.set_ylabel("碳排放（kt）", color=C_RED)
    ax2.tick_params(axis="y", labelcolor=C_RED)
    for i, (c1, c2) in enumerate(zip(costs, carbons)):
        if not np.isnan(c1):
            ax1.text(i - w / 2, c1, f"{c1:.0f}", ha="center", va="bottom", fontsize=7)
            ax2.text(i + w / 2, c2, f"{c2:.0f}", ha="center", va="bottom", fontsize=7)
    lines = [b1, b2]
    # 图例放图（数据区）上方外侧居中、横向展开，避免遮挡最高柱顶标注（审查用户指令）
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower center",
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8, ncol=2)
    ax1.set_title("S4 场景对比：碳约束 / 电价机制 / 新能源波动下的策略变化\n"
                  "（ε=0.90 全部区域不可行已剔除）",
                  pad=28)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "sub4-scenario-compare.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → 已保存 sub4-scenario-compare.pdf")


def fig2_region_cost_carbon(d):
    """S0-main 六区域成本+碳排分解"""
    print("\n=== fig2 区域成本-碳排分解统计 ===")
    sols = d["main"]["sols"]
    costs = [sols[r]["cost_main_M"] for r in REGIONS]
    carbons = [sols[r]["carbon_kt"] for r in REGIONS]
    for r, c, k in zip(REGIONS, costs, carbons):
        print(f"  {r}: 成本 {c:.2f}M | 碳 {k:.2f}kt")

    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(REGIONS))
    w = 0.35
    b1 = ax1.bar(x - w / 2, costs, w, color=C_BLUE, label="运行成本（M 元）")
    ax1.set_ylabel("运行成本（M 元）", color=C_BLUE)
    ax1.tick_params(axis="y", labelcolor=C_BLUE)
    ax1.set_xticks(x)
    ax1.set_xticklabels([RLAB[r] for r in REGIONS])
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w / 2, carbons, w, color=C_RED, alpha=0.7, label="碳排放（kt）")
    ax2.set_ylabel("碳排放（kt）", color=C_RED)
    ax2.tick_params(axis="y", labelcolor=C_RED)
    for i, (c, k) in enumerate(zip(costs, carbons)):
        ax1.text(i - w / 2, c, f"{c:.0f}", ha="center", va="bottom", fontsize=7)
        ax2.text(i + w / 2, k, f"{k:.0f}", ha="center", va="bottom", fontsize=7)
    lines = [b1, b2]
    # 图例横向展开，置于图框上方、标题下方（用户指令）
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower center",
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8, ncol=2)
    ax1.set_title("S0 主解：六区域运行成本与碳排放分解（A-F 对应 RegionA-F）", pad=30)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "sub4-region-cost-carbon.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → 已保存 sub4-region-cost-carbon.pdf")


def fig3_soc_storage(d):
    """D 区储能 SOC + 充放（D 是唯一电量受限区）+ 全局 SOC 末态"""
    print("\n=== fig3 SOC 统计 ===")
    sols = d["main"]["sols"]
    # 初始 SOC 从 sub4-preprocessed 读取（审查 M1：不能与自身末态恒等比较）
    with open(DATA / "sub4-preprocessed.pkl", "rb") as f:
        dp = pickle.load(f)
    init_soc = dp["storage"]["RegionD"]["InitialSOC_MWh"]
    NT = 2406
    MAIN = 2400
    r = "RegionD"
    s = sols[r]
    E, C, Ds = s["E"], s["Cg"] + s["Cr"], s["D"]
    print(f"  {r} SOC: min {E[:MAIN].min():.1f} | max {E[:MAIN].max():.1f} | "
          f"末态 {E[-1]:.1f} (初始 {init_soc:.1f}) | "
          f"充电 max {C[:MAIN].max():.1f}MW | 放电 max {Ds[:MAIN].max():.1f}MW")
    n_ch = int(np.sum(C[:MAIN] > 0.001))
    n_dis = int(np.sum(Ds[:MAIN] > 0.001))
    print(f"  充电非零小时 {n_ch} | 放电非零小时 {n_dis} | 同刻充放 {s['dual_hours']} | "
          f"终态≥初始: {E[-1] >= init_soc - 1e-6}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    t = np.arange(MAIN)
    # 绘制主时域（0-2399），2400-2405 结清段省略（避免稀疏）
    ax1.plot(t, E[:MAIN], color=C_MAIN, lw=1.5)
    ax1.axhline(init_soc, color=C_RED, ls="--", lw=0.8, label=f"初始 SOC ({init_soc:.0f})")
    ax1.set_ylabel("SOC（MWh）")
    ax1.set_title(f"D 区储能荷电状态与充放电（主时域 0-2399h）")
    ax1.legend(loc="lower left", frameon=False, fontsize=8)
    ax1.grid(linestyle=":", alpha=0.4)
    ax2.plot(t, C[:MAIN], color=C_BLUE, lw=1.5, label="充电功率 $C^g+C^r$（电网+新能源）")
    ax2.plot(t, Ds[:MAIN], color=C_RED, lw=1.5, label="放电功率 $D$")
    ax2.set_ylabel("功率（MW）")
    ax2.set_xlabel("小时")
    ax2.legend(loc="upper right", frameon=False, fontsize=8)
    ax2.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "sub4-soc-storage.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → 已保存 sub4-soc-storage.pdf")


def fig4_utilization(d):
    """新能源利用率双口径（受限消纳）+ 与 S3 基准对比"""
    print("\n=== fig4 利用率统计 ===")
    sols = d["main"]["sols"]
    util_ns = [sols[r]["util_no_sell_pct"] for r in REGIONS]
    util_s = [sols[r]["util_sell_pct"] for r in REGIONS]
    for r, u1, u2 in zip(REGIONS, util_ns, util_s):
        print(f"  {r}: 不含外送 {u1:.1f}% | 含外送 {u2:.1f}%")

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(REGIONS))
    w = 0.32
    b1 = ax.bar(x - w / 2, util_ns, w, color=C_BLUE,
                label="不含外送 $(R+C^r)/\\mathrm{Avail}$")
    b2 = ax.bar(x + w / 2, util_s, w, color=C_GREEN, alpha=0.7,
                label="含外送 $(R+C^r+S)/\\mathrm{Avail}$")
    ax.set_ylabel("新能源利用率（%）")
    ax.set_xticks(x)
    ax.set_xticklabels([RLAB[r] for r in REGIONS])
    for i, (u1, u2) in enumerate(zip(util_ns, util_s)):
        ax.text(i - w / 2, u1 + 0.5, f"{u1:.1f}", ha="center", fontsize=7)
        ax.text(i + w / 2, u2 + 0.5, f"{u2:.1f}", ha="center", fontsize=7)
    # 全局基准线用灰色（审查 L1：辅助色 ≤2），并注明口径（审查 M3）
    ax.axhline(20.83, color="#666666", ls="--", lw=0.8,
               label="S3 全局加权基准 20.8%")
    ax.set_title("S4 新能源利用率双口径（受限消纳）\n"
                 "注：S4 受限口径锚定 S3 基准（逐区与 S3 一致）；全局 20.8% 为加权值")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "sub4-utilization.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → 已保存 sub4-utilization.pdf")


def fig5_peak_net(d):
    """六区域峰值净购电（主时域）"""
    print("\n=== fig5 峰值净购电统计 ===")
    sols = d["main"]["sols"]
    peaks = [sols[r]["peak_MW"] for r in REGIONS]
    stds = [sols[r]["std_MW"] for r in REGIONS]
    for r, p, sd in zip(REGIONS, peaks, stds):
        print(f"  {r}: 峰值 {p:.1f}MW | 净购电 std {sd:.1f}")

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(REGIONS))
    b1 = ax.bar(x, peaks, 0.5, color=C_BLUE, label="峰值净购电 ($\\max\\,G-S$)")
    ax.set_ylabel("峰值净购电（MW）")
    ax.set_xticks(x)
    ax.set_xticklabels([RLAB[r] for r in REGIONS])
    for i, p in enumerate(peaks):
        ax.text(i, p + 5, f"{p:.0f}", ha="center", fontsize=8)
    ax.set_title("S0 主解：六区域峰值净购电（主时域）")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "sub4-peak-net.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → 已保存 sub4-peak-net.pdf")


def fig6_baseload_contribution(d):
    """基荷策略贡献：任务数覆盖 + 0-1 变量压缩量化（审查 M2：口径动态计算）"""
    print("\n=== fig6 基荷贡献统计 ===")
    with open(DATA / "sub4-preprocessed.pkl", "rb") as f:
        dp = pickle.load(f)
    tasks = dp["tasks"]
    n_bl = sum(1 for t in tasks if t.get("baseload", False))
    n_nbl = sum(1 for t in tasks
                if t["type"] != "RealTimeInference" and not t.get("baseload", False))
    n_rt = sum(1 for t in tasks if t["type"] == "RealTimeInference")
    gh_bl = sum(t["gh"] for t in tasks if t.get("baseload", False))
    gh_tot = sum(t["gh"] for t in tasks)
    gh_nbl = gh_tot - gh_bl - sum(t["gh"] for t in tasks
                                  if t["type"] == "RealTimeInference")
    print(f"  基荷任务 {n_bl} ({n_bl/len(tasks):.1%}) | 非基荷 {n_nbl} ({n_nbl/len(tasks):.1%}) | 实时 {n_rt}")
    print(f"  基荷 GPU-hour {gh_bl:,.0f}/{gh_tot:,.0f} ({gh_bl/gh_tot:.1%}) | "
          f"非基荷 GPU-hour 占比 {gh_nbl/gh_tot:.1%}")

    labels = ["基荷预填\n(EDF 固定)", "非基荷\n(进 MILP)", "实时推理\n(到达即开工)"]
    vals = [n_bl, n_nbl, n_rt]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # 对数刻度（v2 调整）：非基荷仅 50 任务（0.1%），线性 y 轴下几乎不可见。
    # 柱底部从 1 起画（log(0) 无定义，且最小任务数为 50 > 1），柱高 ∝ log10(v)。
    bars = ax.bar(labels, vals, 0.5, bottom=1.0, zorder=3,
                  color=[C_GREEN, C_BLUE, C_MAIN])
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.18, f"{v:,}",
                ha="center", va="bottom", fontsize=9, zorder=4)
    ax.set_ylabel("任务数（对数刻度）")
    ax.set_ylim(1, max(vals) * 1.6)
    ax.set_title(f"基荷策略任务覆盖：{len(tasks):,} 任务中 {n_bl/len(tasks):.1%} 由基荷预填固定\n"
                 f"（基荷占 GPU-hour {gh_bl/gh_tot:.1%}，层 2 0-1 变量仅剩 {n_nbl} 任务 ≈ "
                 f"{n_nbl/len(tasks):.2%}）")
    # 网格线置于数据之下（zorder 低于柱子/数值），避免虚线压在柱上
    ax.grid(axis="y", linestyle=":", alpha=0.4, which="both", zorder=0)
    fig.tight_layout()
    fig.savefig(FIG / "sub4-baseload-contribution-v2.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → 已保存 sub4-baseload-contribution-v2.pdf")


def main():
    d = load()
    assert d["main"].get("feasible"), "主解不可行，无法出图"
    print("=" * 72)
    print("S4 阶段 2.1.5 图表生成（先算后画，PR-014）")
    print("=" * 72)
    fig1_scenario_compare(d)
    fig2_region_cost_carbon(d)
    fig3_soc_storage(d)
    fig4_utilization(d)
    fig5_peak_net(d)
    fig6_baseload_contribution(d)
    print("\nS4 CHARTS DONE")


if __name__ == "__main__":
    main()

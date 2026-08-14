# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 2.1.6 S4 图表 v2 — 重做问题四可视化，聚焦三类信息：
    ① 算-储-电协同机理（净购电时序 + 储能充放 + SOC）
    ② 场景权衡（全可行场景 × 指标 热力图）
    ③ 灵敏度分析（碳约束 ε_min / 峰谷价差 / 新能源波动）

原理：
    先算后画（PR-014）：数值全部来自已落盘结果——
    - s4-results.pkl（阶段 2.1 主解）：A/D 区净购电、SOC、充放时序（机理图）
    - s4-sensitivity.pkl（阶段 2.1.6 灵敏度扫描）：ε/峰谷价差/新能源网格 +
      ε_min 二分精确值（灵敏度图）
    设计原则（对齐问题四核心"权衡 + 场景策略变化"）：
    - fig1 机理图：储能削峰填谷的时序证据（东部 A 高载 vs 西部 D 算力中心）
    - fig2 权衡热力图：全可行场景 × 指标"向好变化%"，绿色=变好红色=变差
      （替代原 fig1 场景对比——原图含不可比单区解，误导）
    - fig3 碳约束灵敏度：逐区域 ε_min 与理论降碳空间 (1−ε_min)
    - fig4 峰谷价差灵敏度：成本线性 / 碳排不变 / 储能套利饱和
    - fig5 新能源波动灵敏度：成本碳排线性 / 利用率与储能策略不变
    - 原 fig4（利用率双口径，卖电≈0 无差异）、fig5（峰值净购电=表格）、
      fig6（基荷贡献=标题文字）信息量为 0，废弃归档

输入数据：
    - outputs/data/s4-results.pkl — main.sols.{A,D}: {G,S,Cg,Cr,D,E,net,...}
    - outputs/data/s4-sensitivity.pkl — eps{ε:{agg,per}} / price / renew /
      eps_min{r} / grids
    - 中文指标 → 变量名映射：购电→G, 卖电→S, 新能源充电→Cr, 充电→Cg+Cr,
      放电→D, SOC→E, 净购电→net=G-S, 成本(M元)→cost_main_M,
      碳排(kt)→carbon_kt, 储能充放能→es_MWh

输出：
    - outputs/figures/sub4-mechanism-net.pdf — A/D 区净购电+SOC 时序机理图
    - outputs/figures/sub4-tradeoff-heatmap.pdf — 场景×指标权衡热力图
    - outputs/figures/sub4-pareto-carbon.pdf — ε_min 与降碳空间
    - outputs/figures/sub4-sens-price.pdf — 峰谷价差灵敏度
    - outputs/figures/sub4-sens-renew.pdf — 新能源波动灵敏度
    - 控制台统计量（PR-014 核对）

对应论文章节：
    问题四（S4）算-储-电协同优化 — 场景设计 / 灵敏度分析
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# === 中文字体与负号（chart-generator 强制前置）===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                   'WenQuanYi Micro Hei', 'Source Han Sans CN']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"
FIG = BASE / "outputs" / "figures"
FIG.mkdir(exist_ok=True)

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
RLAB = {"RegionA": "A", "RegionB": "B", "RegionC": "C",
        "RegionD": "D", "RegionE": "E", "RegionF": "F"}
MAIN = 2400

# 去饱和配色（灰主调 + 辅助色 ≤2）
C_MAIN = "#333333"
C_BLUE = "#4c78a8"      # 充电 / 利好
C_RED = "#c14b3a"       # 放电 / 利空
C_GRAY = "#999999"

SCEN_ORDER = ["S0-main", "price-1.5", "renew-1.2", "renew-0.8"]
SCEN_LABEL = {"S0-main": "S0 基准", "price-1.5": "峰谷差×1.5",
              "renew-1.2": "新能源+20%", "renew-0.8": "新能源-20%"}


def load_all():
    with open(DATA / "s4-results.pkl", "rb") as f:
        res = pickle.load(f)
    with open(DATA / "s4-sensitivity.pkl", "rb") as f:
        sens = pickle.load(f)
    return res, sens


def ma(x, w):
    """滚动均值（窗口 w，边界取有效均值）"""
    k = np.ones(w) / w
    y = np.convolve(x, k, mode="same")
    n = np.minimum(np.arange(len(x)) + 1, len(x) - np.arange(len(x)))
    n = np.minimum(n, w)
    return y * w / n


def fig1_mechanism_net(res):
    """A 区（东部高载）与 D 区（西部算力中心）：净购电时序 + 储能充放 + SOC。
    展示算-储-电协同下储能削峰填谷的机理。"""
    print("\n=== fig1 机理图（净购电+储能）统计 ===")
    sols = res["main"]["sols"]
    t = np.arange(MAIN)
    rows = [("RegionA", "A：东部高载区（新能源利用率 8.5%）"),
            ("RegionD", "D：西部算力中心（基荷 17.3%，利用率 24.8%）")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})
    for i, (r, sub) in enumerate(rows):
        s = sols[r]
        net = s["net"][:MAIN]
        ch = (s["Cg"] + s["Cr"])[:MAIN]
        ds = s["D"][:MAIN]
        E = s["E"][:MAIN]
        with open(DATA / "sub4-preprocessed.pkl", "rb") as f:
            init = pickle.load(f)["storage"][r]["InitialSOC_MWh"]
        print(f"  {r}: net 峰谷差 {net.max()-net.min():.0f}MW | "
              f"充电能 {ch.sum()/1e3:.1f}GWh | 放电能 {ds.sum()/1e3:.1f}GWh | "
              f"SOC [{E.min():.0f},{E.max():.0f}] 初始 {init:.0f}")
        # 左列：净购电 + 充放背景（两层填充，充放叠在 net 下方）
        ax = axes[i, 0]
        ax.fill_between(t, 0, ch, color="#3a7bd5", alpha=0.50,
                        label="充电 $C^g+C^r$（电网+新能源）")
        ax.fill_between(t, 0, -ds, color="#e8523e", alpha=0.50,
                        label="放电 $D$")
        ax.plot(t, ma(net, 24), color="#1a1a1a", lw=1.6, alpha=1.0,
                label="净购电 $G-S$（日均）")
        ax.set_ylabel("功率（MW）", fontsize=9)
        ax.set_title(f"{sub}\n净购电峰谷差 {net.max()-net.min():.0f} MW | "
                     f"储能充放能 {ch.sum()/1e3:.0f} GWh", fontsize=10, pad=28)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  frameon=False, fontsize=9, ncol=3)
        ax.grid(linestyle=":", linewidth=0.3, alpha=0.4)
        # 右列：SOC
        ax2 = axes[i, 1]
        ax2.plot(t, E, color=C_RED, lw=1.0, alpha=0.7)
        ax2.plot(t, ma(E, 24), color=C_RED, lw=1.8, label="SOC $E_t$（日均）")
        ax2.axhline(init, color=C_GRAY, ls="--", lw=1.0,
                    label=f"初始 SOC {init:.0f} MWh")
        ax2.set_ylabel("SOC（MWh）", fontsize=9)
        ax2.set_title(f"{r[-1]} 区储能 SOC 全程 {E.min():.0f}–{E.max():.0f} MWh",
                      fontsize=10, pad=28)
        ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   frameon=False, fontsize=9, ncol=2)
        ax2.grid(linestyle=":", linewidth=0.3, alpha=0.4)
        if i == 1:
            ax.set_xlabel("小时")
            ax2.set_xlabel("小时")
    fig.suptitle("算-储-电协同机理：储能削峰填谷与 SOC 轨迹（主时域 0–2399h）",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "sub4-mechanism-net.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → 已保存 sub4-mechanism-net.pdf")


def fig2_tradeoff_heatmap(res, sens):
    """全可行场景 × 指标"向好变化%"热力图（绿色=向好，红色=向坏）。"""
    print("\n=== fig2 权衡热力图统计 ===")
    base = res["main"]
    base_es = sum(float(np.sum(s["Cg"][:MAIN] + s["Cr"][:MAIN]))
                  for s in base["sols"].values())
    base_v = {"cost_main_M": base["cost_main_M"], "carbon_kt": base["carbon_kt"],
              "es_MWh": base_es}
    scen_agg = {"S0-main": base_v,
                "price-1.5": sens["price"][1.5]["agg"],
                "renew-1.2": sens["renew"][1.2]["agg"],
                "renew-0.8": sens["renew"][0.8]["agg"]}
    rows = [("cost_main_M", "运行成本", -1),
            ("carbon_kt", "碳排放", -1),
            ("es_MWh", "储能充放能", 1)]
    data = np.zeros((len(rows), len(SCEN_ORDER)))
    annot = np.empty_like(data, dtype=object)
    for j, sc in enumerate(SCEN_ORDER):
        agg = scen_agg[sc]
        for i, (key, lab, _dir) in enumerate(rows):
            vb, vs = base_v[key], agg[key]
            chg = (vs - vb) / vb * 100
            data[i, j] = chg
            annot[i, j] = f"{chg:+.1f}%"
            print(f"  {sc:<10} {lab}: {vs:.1f} vs 基准 {vb:.1f} → {chg:+.1f}%")

    # 显示矩阵：dir 归一化（正=绿=向好）
    show = data * np.array([r[2] for r in rows])[:, None]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=25)
    im = ax.imshow(show, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(SCEN_ORDER)))
    ax.set_xticklabels([SCEN_LABEL[s] for s in SCEN_ORDER], fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows], fontsize=10)
    for i in range(len(rows)):
        for j in range(len(SCEN_ORDER)):
            ax.text(j, i, annot[i, j], ha="center", va="center",
                    fontsize=10, color="black",
                    fontweight="bold" if abs(data[i, j]) > 5 else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("相对 S0 基准变化（%）", fontsize=9)
    ax.set_title("S4 场景权衡：全可行场景 × 关键指标变化\n"
                 "（峰值净购电 550 MW、时延 28.1 ms、QoS 100%、利用率 20.8% 在各场景下不变）",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "sub4-tradeoff-heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → 已保存 sub4-tradeoff-heatmap.pdf")


def fig3_pareto_carbon(sens):
    """逐区域 ε_min 与理论降碳空间 (1−ε_min)×100%。"""
    print("\n=== fig3 碳约束灵敏度统计 ===")
    emin = sens["eps_min"]
    space = {r: (1 - v) * 100 for r, v in emin.items()}
    for r in REGIONS:
        print(f"  {r}: ε_min={emin[r]:.4f} | 理论降碳空间 {space[r]:.2f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.8))
    x = np.arange(len(REGIONS))
    colors = [C_BLUE if r == "RegionD" else C_RED if r in ("RegionE", "RegionF")
              else C_MAIN for r in REGIONS]
    # 左：ε_min
    ax1.bar(x, [emin[r] for r in REGIONS], 0.55, color=colors)
    ax1.axhline(1.0, color=C_GRAY, ls="--", lw=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([RLAB[r] for r in REGIONS])
    ax1.set_ylim(0.88, 1.01)
    ax1.set_ylabel("最低可行碳约束系数 $\\varepsilon_{\\min}$")
    ax1.set_title("逐区域 $\\varepsilon_{\\min}$：碳约束可行域下限")
    for i, r in enumerate(REGIONS):
        ax1.text(i, emin[r] + 0.004, f"{emin[r]:.3f}", ha="center", fontsize=8)
    ax1.grid(axis="y", linestyle=":", linewidth=0.3, alpha=0.4)
    # 右：理论降碳空间
    ax2.bar(x, [space[r] for r in REGIONS], 0.55, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels([RLAB[r] for r in REGIONS])
    ax2.set_ylabel("理论降碳空间 $1-\\varepsilon_{\\min}$（%）")
    ax2.set_title("降碳空间：东部 ≈1%，西部绿电 ≈2.7%，D 区 ≈9.5%")
    for i, r in enumerate(REGIONS):
        ax2.text(i, space[r] + 0.15, f"{space[r]:.1f}%", ha="center", fontsize=8)
    ax2.grid(axis="y", linestyle=":", linewidth=0.3, alpha=0.4)
    fig.suptitle("碳约束灵敏度（区域异质性）：固定负荷购电碳排刚性 vs 任务/储能弹性",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "sub4-pareto-carbon.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → 已保存 sub4-pareto-carbon.pdf")


def _sens_curve(ax, xs, ys, labels=None, ylab="", unit="", agg_color=None,
                mark=True, yfmt=".1f"):
    """多区域浅线 + 聚合粗线。xs: 参数值列表, ys: {区域→list}。"""
    agg = np.mean(list(ys.values()), axis=0)
    for r in ys:
        ax.plot(xs, ys[r], color=C_GRAY, lw=0.8, alpha=0.45)
    ax.plot(xs, agg, color=agg_color or C_MAIN, lw=2.0,
            marker="o" if mark else None, ms=4)
    if mark:
        for xv, yv in zip(xs, agg):
            ax.text(xv, yv, f"{yv:{yfmt}}", ha="center", va="bottom",
                    fontsize=9, color=agg_color or C_MAIN)
    ax.set_ylabel(ylab)
    ax.grid(linestyle=":", linewidth=0.3, alpha=0.4)


def fig4_sens_price(sens):
    """峰谷价差灵敏度：成本/碳排/储能充放能/峰值。"""
    print("\n=== fig4 峰谷差灵敏度统计 ===")
    xs = sens["grids"]["price"]
    pr = sens["price"]
    cost, carb, es, peak = {}, {}, {}, {}
    for r in REGIONS:
        cost[r] = [pr[ps]["per"][r]["cost_main_M"] for ps in xs]
        carb[r] = [pr[ps]["per"][r]["carbon_kt"] for ps in xs]
        es[r] = [pr[ps]["per"][r]["es_MWh"] / 1e3 for ps in xs]     # GWh
        peak[r] = [pr[ps]["per"][r]["peak_MW"] for ps in xs]
    agg_cost = [pr[ps]["agg"]["cost_main_M"] for ps in xs]
    agg_carb = [pr[ps]["agg"]["carbon_kt"] for ps in xs]
    agg_es = [pr[ps]["agg"]["es_MWh"] / 1e3 for ps in xs]
    agg_peak = [pr[ps]["agg"]["peak_MW"] for ps in xs]
    print(f"  聚合成本: {agg_cost[0]:.0f}→{agg_cost[-1]:.0f}M（{(agg_cost[-1]/agg_cost[0]-1)*100:+.1f}%）")
    print(f"  聚合碳排: {agg_carb[0]:.1f}→{agg_carb[-1]:.1f}kt（Δ{(agg_carb[-1]-agg_carb[0]):+.1f}）")
    print(f"  储能充放能: {agg_es[0]:.1f}→{agg_es[-1]:.1f}GWh（{(agg_es[-1]/agg_es[0]-1)*100:+.1f}%）")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8))
    _sens_curve(axes[0, 0], xs, cost, ylab="运行成本（M 元）", unit="M")
    axes[0, 0].set_title("(a) 成本随峰谷差扩大近线性上升")
    _sens_curve(axes[0, 1], xs, carb, ylab="碳排放（kt）", unit="kt")
    axes[0, 1].set_title("(b) 碳排不受电价机制影响（恒定 1961 kt）")
    _sens_curve(axes[1, 0], xs, es, ylab="储能充放能（GWh）", unit="GWh")
    axes[1, 0].set_title("(c) 储能套利增强后饱和（×1.5 后平台）")
    _sens_curve(axes[1, 1], xs, peak, ylab="峰值净购电（MW）", unit="MW")
    axes[1, 1].set_title("(d) 峰值净购电不变（550 MW）")
    for ax in axes.flat:
        ax.set_xlabel("峰段价格乘数（峰谷价差）")
    fig.suptitle("峰谷价差灵敏度：只放大成本与储能套利，不改变购电结构与碳排",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "sub4-sens-price.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → 已保存 sub4-sens-price.pdf")


def fig5_sens_renew(sens):
    """新能源波动灵敏度：成本/碳排/利用率/储能充放能。"""
    print("\n=== fig5 新能源波动灵敏度统计 ===")
    xs = sens["grids"]["renew"]
    rn = sens["renew"]
    cost, carb, util, es = {}, {}, {}, {}
    for r in REGIONS:
        cost[r] = [rn[rs]["per"][r]["cost_main_M"] for rs in xs]
        carb[r] = [rn[rs]["per"][r]["carbon_kt"] for rs in xs]
        util[r] = [rn[rs]["per"][r]["util_no_sell_pct"] for rs in xs]
        es[r] = [rn[rs]["per"][r]["es_MWh"] / 1e3 for rs in xs]
    agg_cost = [rn[rs]["agg"]["cost_main_M"] for rs in xs]
    agg_carb = [rn[rs]["agg"]["carbon_kt"] for rs in xs]
    agg_util = [rn[rs]["agg"]["util_no_sell_pct"] for rs in xs]
    agg_es = [rn[rs]["agg"]["es_MWh"] / 1e3 for rs in xs]
    slope_cost = (agg_cost[-1] - agg_cost[0]) / (1.2 - 0.8)
    slope_carb = (agg_carb[-1] - agg_carb[0]) / (1.2 - 0.8)
    print(f"  聚合成本: {agg_cost[0]:.0f}→{agg_cost[-1]:.0f}M（斜率 {slope_cost:+.0f}M/10%新能源）")
    print(f"  聚合碳排: {agg_carb[0]:.1f}→{agg_carb[-1]:.1f}kt（斜率 {slope_carb:+.0f}kt/10%新能源）")
    print(f"  利用率: {agg_util[0]:.2f}→{agg_util[-1]:.2f}%（不变） | 储能: {agg_es[0]:.0f}→{agg_es[-1]:.0f}GWh")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8))
    _sens_curve(axes[0, 0], xs, cost, ylab="运行成本（M 元）", unit="M")
    axes[0, 0].set_title("(a) 成本随新能源增加线性下降")
    _sens_curve(axes[0, 1], xs, carb, ylab="碳排放（kt）", unit="kt")
    axes[0, 1].set_title("(b) 碳排随新能源增加线性下降")
    _sens_curve(axes[1, 0], xs, util, ylab="新能源利用率（%）", unit="%",
                agg_color=C_BLUE)
    axes[1, 0].set_title("(c) 利用率不变（受限消纳上限与分母同比缩放）")
    _sens_curve(axes[1, 1], xs, es, ylab="储能充放能（GWh）", unit="GWh",
                agg_color=C_BLUE)
    axes[1, 1].set_title("(d) 储能策略基本不变（电价驱动为主）")
    for ax in axes.flat:
        ax.set_xlabel("新能源出力比例（波动 ±20%）")
    fig.suptitle("新能源波动灵敏度：线性改变成本/碳排，利用率与储能策略稳健",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "sub4-sens-renew.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → 已保存 sub4-sens-renew.pdf")


def main():
    res, sens = load_all()
    assert res["main"].get("feasible"), "主解不可行，无法出图"
    print("=" * 72)
    print("S4 阶段 2.1.6 图表 v2（先算后画，PR-014）")
    print("=" * 72)
    fig1_mechanism_net(res)
    fig2_tradeoff_heatmap(res, sens)
    fig3_pareto_carbon(sens)
    fig4_sens_price(sens)
    fig5_sens_renew(sens)
    print("\nS4 CHARTS V2 DONE")


if __name__ == "__main__":
    main()

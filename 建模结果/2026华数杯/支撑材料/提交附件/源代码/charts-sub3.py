# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
charts-sub3.py — S3 储能协同优化：阶段 2.1 出图 v2（chart-generator skill）

目的：
    生成 S3 主解（受限消纳 ε=1.00）5 张论文图表 v2：SOC/充放、净购电对比（削峰）、
    碳排压降极限（ε_min，替换原 Pareto 图）、四指标对比、区域峰值削峰。
    文字与标注使用 LaTeX 排版：所有数学符号/单位/标注经 matplotlib **mathtext**
    （内置 LaTeX 语法数学排版引擎，无需外部 TeX 安装）渲染为 $\\mathrm{}$/$\\epsilon$ 等；
    中文保留 SimHei（usetex 无法可靠排版中文）。

原理：
    1. 数据全部来自 outputs/data/cache/s3_solutions.pkl（阶段 2.1 求解结果，含 check.eps_min 诊断），
       先算后画（PR-014）：图表脚本不做任何计算，只读取已核对的数值
    2. 削峰对比口径：净购电 = G − S；聚合 = 6 区域逐时加总
    3. 配色去饱和（#333333/#1f77b4/#d62728），线宽 1.5–2pt，SimHei 中文字体
    4. 出图后副本写 solution/artifacts/charts/ 并在 manifest.md 登记（版本 -v2 不覆盖 v1）

输入映射：
    - outputs/data/cache/s3_solutions.pkl（solutions/aggregate/compare/no_storage_c_region/benchmark_region/check.eps_min）
    - outputs/data/s3-preprocessed.pkl（panel：NetGridImport_base_MW 等，仅取对照序列）

输出：
    - outputs/figures/sub3-{soc-charge-discharge,net-import-compare,carbon-floor,four-metrics,peak-shaving}-v2.pdf
    - solution/artifacts/charts/ 同名副本 + solution/artifacts/manifest.md 登记

论文章节：
    问题三 储能协同优化：结果分析（阶段 2.2）
"""
import pickle
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Source Han Sans CN"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(r"E:\MathModel_pj-2026-C-sub3")
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
MAIN = 2400
COL_OPT = "#333333"
COL_NOC = "#1f77b4"
COL_BEN = "#d62728"

with open(BASE / "outputs" / "data" / "cache" / "s3_solutions.pkl", "rb") as f:
    sol = pickle.load(f)
with open(BASE / "outputs" / "data" / "s3-preprocessed.pkl", "rb") as f:
    prep = pickle.load(f)
panel = prep["panel"]

# ---- 计算（仅汇总/对照，全部来自已核对求解结果）----
h = np.arange(MAIN)
net_opt = {r: sol["solutions"][(r, 1.00)]["net"][:MAIN] for r in REGIONS}
net_ben = {r: panel.xs(r)["NetGridImport_base_MW"].values[:MAIN] for r in REGIONS}
gp_noc = {r: np.maximum(panel.xs(r)["Total_Load_MW"].values[:MAIN]
                        - panel.xs(r)["UsedRenewable_MW"].values[:MAIN], 0.0) for r in REGIONS}

opt_cost = sol["aggregate"][1.00]["cost_M"]
opt_carb = sol["aggregate"][1.00]["carbon_kt"]
opt_peak = sol["aggregate"][1.00]["peak_MW"]
opt_std = sol["aggregate"][1.00]["std_MW"]
noc = sol["compare"]["no_storage_c"]
ben = sol["compare"]["benchmark"]
noc_std = float(np.std(sum(gp_noc[r] for r in REGIONS)))
ben_std = float(np.std(sum(net_ben[r] for r in REGIONS)))

print("[校验] 四指标（优化/无储能c/基准）: 成本", opt_cost, noc["cost_M"], ben["cost_M"],
      "| 碳", opt_carb, noc["carbon_kt"], ben["carbon_kt"],
      "| 峰值", opt_peak, noc["peak_MW"], ben["peak_MW"],
      "| std", opt_std, round(noc_std, 2), round(ben_std, 2))


def save(name):
    fig = plt.gcf()
    pdf = BASE / "outputs" / "figures" / name
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(pdf, BASE / "solution" / "artifacts" / "charts" / name)
    print(f"  saved {pdf.name}")
    return pdf.name


# ---- 图 1：6 区域 SOC 曲线 + 充放功率（ε=1.00，全时域 0–2405）----
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
hh = np.arange(2406)
for ax, r in zip(axes.ravel(), REGIONS):
    s = sol["solutions"][(r, 1.00)]
    ax.plot(hh, s["E"], color=COL_OPT, lw=1.8, label=r"$\mathrm{SOC}$（$\mathrm{MWh}$）")
    ax2 = ax.twinx()
    ax2.fill_between(hh, s["Cg"] + s["Cr"], color="#2ca02c", alpha=0.35,
                     label=r"充电（$\mathrm{MW}$）")
    ax2.fill_between(hh, -s["D"], color="#d62728", alpha=0.35,
                     label=r"放电（$\mathrm{MW}$）")
    ax.set_title(r, fontsize=11)
    ax.set_ylabel(r"$\mathrm{SOC}$（$\mathrm{MWh}$）")
    ax2.set_ylabel(r"功率（$\mathrm{MW}$）")
    ax2.set_ylim(-s["D"].max() * 1.1, (s["Cg"] + s["Cr"]).max() * 1.1 + 1)
    if r == "RegionF":
        ax.set_xlabel(r"小时（$0$–$2405$）")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
fig.suptitle(r"6 区域最优储能：$\mathrm{SOC}$ 曲线与充放电功率（$\epsilon=1.00$）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
save("sub3-soc-charge-discharge-v2.pdf")

# ---- 图 2：净购电时序对比（基准 vs 最优，6 区域聚合，削峰可视化）----
fig, ax = plt.subplots(figsize=(13.5, 5))
ax.plot(h, sum(net_ben[r] for r in REGIONS), color=COL_BEN, lw=1.5, label="基准轨迹（参考）")
ax.plot(h, sum(net_opt[r] for r in REGIONS), color=COL_OPT, lw=2.0,
        label=r"优化解（$\epsilon=1.00$）")
ax.set_xlabel(r"小时（主时域 $0$–$2399$）")
ax.set_ylabel(r"净购电功率（$\mathrm{MW}$）")
ax.set_title(r"净购电时序对比：基准 vs 最优（6 区域聚合，削峰可视化）", fontsize=13)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
ax.grid(alpha=0.3, lw=0.3)
fig.subplots_adjust(right=0.82)
save("sub3-net-import-compare-v2.pdf")

# ---- 图 3：碳排可压降极限（ε_min 可视化，替换 Pareto 前沿；建模核验 §10）----
eps_min = sol["check"]["eps_min"]
fig, ax = plt.subplots(figsize=(10, 6.8))
x = np.arange(len(REGIONS))
w = 0.35
bases = [prep["carbon_base_kt"][r] for r in REGIONS]
floors = [eps_min[r]["c_min_kt"] for r in REGIONS]
vmax = max(bases + floors)
ax.bar(x - w / 2, bases, w, label=r"基准碳排（$\epsilon=1.00$）", color=COL_BEN)
ax.bar(x + w / 2, floors, w, label=r"碳排下限 $C_{\min}$", color=COL_OPT)
for xi, (b, f, em) in enumerate(zip(bases, floors,
                                    [eps_min[r]["eps_min"] for r in REGIONS])):
    ax.text(xi + w / 2, f + vmax * 0.02, rf"$\epsilon_{{\min}}$={em:.3f}",
            ha="center", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels(REGIONS, fontsize=10)
ax.set_ylabel(r"碳排（$\mathrm{kt}$）")
ax.set_ylim(0, vmax * 1.14)
ax.set_title(r"区域碳排可压降极限（受限消纳；$\epsilon<\epsilon_{\min}$ 不可行，主解 $\epsilon=1.00$）",
             fontsize=13)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3, lw=0.3)
save("sub3-carbon-floor-v2.pdf")

# ---- 图 4：四指标对比（优化 vs 无储能 c vs 基准，2×2 子图）----
metrics = [
    (r"运行成本（$\mathrm{M}$ 元）", [opt_cost, noc["cost_M"], ben["cost_M"]]),
    (r"碳排（$\mathrm{kt}$）", [opt_carb, noc["carbon_kt"], ben["carbon_kt"]]),
    (r"峰值净购电（$\mathrm{MW}$）", [opt_peak, noc["peak_MW"], ben["peak_MW"]]),
    (r"净购电 $\mathrm{std}$（$\mathrm{MW}$）", [opt_std, noc_std, ben_std]),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
labels = [r"优化 $\epsilon=1.00$", "无储能口径c", "基准(参考)"]
colors = [COL_OPT, COL_NOC, COL_BEN]
for ax, (title, vals) in zip(axes.ravel(), metrics):
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    vmax = max(vals)
    ax.set_ylim(0, vmax * 1.14)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + vmax * 0.02, f"{v:.1f}",
                ha="center", fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="x", labelsize=9)
fig.suptitle(r"问题三 四指标对比（优化解 vs 无储能 vs 基准，主时域）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
save("sub3-four-metrics-v2.pdf")

# ---- 图 5：区域峰值净购电削峰效果（6 区域分组柱状）----
x = np.arange(len(REGIONS))
w = 0.26
opt_peaks = [sol["solutions"][(r, 1.00)]["peak_MW"] for r in REGIONS]
noc_peaks = [sol["no_storage_c_region"][r]["peak_MW"] for r in REGIONS]
ben_peaks = [sol["benchmark_region"][r]["peak_MW"] for r in REGIONS]
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x - w, opt_peaks, w, label=r"优化 $\epsilon=1.00$", color=COL_OPT)
ax.bar(x, noc_peaks, w, label="无储能口径c", color=COL_NOC)
ax.bar(x + w, ben_peaks, w, label="基准(参考)", color=COL_BEN)
ax.set_xticks(x)
ax.set_xticklabels(REGIONS, fontsize=10)
ax.set_ylabel(r"区域峰值净购电（$\mathrm{MW}$）")
ax.set_title(r"区域峰值净购电削峰效果（主时域）", fontsize=13)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3, lw=0.3)
save("sub3-peak-shaving-v2.pdf")

print("ALL S3 CHARTS DONE (charts-sub3.py)")

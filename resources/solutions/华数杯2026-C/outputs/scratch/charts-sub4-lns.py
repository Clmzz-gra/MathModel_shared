# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    绘制 LNS 大规模选时实验的收敛曲线图 sub4-lns-convergence.pdf，
    替换 result-analysis 中占位符，回答"S4 大规模选时可解性"章节的收敛证据。

原理：
    LNS（大邻域搜索）逐轮对 E/F 区子 MILP 求解，cost 序列单调非增（接受准则
    保证只接受改进解）。curve 记录每轮当前最优 cost。图中以 E/F 两区 50 轮
    cost 曲线展示收敛过程，并叠加 EDF 全量基线（水平虚线）作为对照：
      收敛速率 = 前期快速下降、中后期平缓趋近终值；
      终值与 EDF 基线的差距 = LNS 的选时增益（E +7.66M / F +7.09M）。
    W=168h 邻域截断 + B2 起点已吃电价红利，LNS 增益为保守下界（见报告 7.7.2）。

输入数据：
    - outputs/scratch/s4-lns-results.pkl — per_region[RegionE/RegionF]:
      curve (list of dict: it/cost/improvement/time_s), cost_edf, cost_b2,
      cost_final, iterations
    - 中文指标 → 变量名映射：区域成本(M元)→cost, EDF 基线→cost_edf,
      LNS 终值→cost_final, 轮次→it

输出：
    - outputs/figures/sub4-lns-convergence.pdf — E/F 两区 50 轮收敛曲线
    - 控制台统计量（PR-014 先算后画：曲线 min/max/起止值、总改善、末段斜率）

对应论文章节：
    §7.7.2 LNS 大规模选时（36.6M 变量不可解 → 启发式逼近）
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
SCRATCH = BASE / "outputs" / "scratch"
FIG = BASE / "outputs" / "figures"
FIG.mkdir(exist_ok=True)

# 去饱和配色（灰主调 + 辅助色 ≤2）
C_E = "#333333"      # E 区 LNS（主灰）
C_F = "#4c78a8"      # F 区 LNS（辅助蓝）
C_EDF = "#bbbbbb"    # EDF 基线（浅灰虚线）


def load():
    with open(SCRATCH / "s4-lns-results.pkl", "rb") as f:
        d = pickle.load(f)
    return d


def main():
    d = load()
    per = d["per_region"]
    cmp_ = d["compare"]

    # ================= 第一阶段：纯计算 + 统计量打印 =================
    series = {}
    for r, name in [("RegionE", "E"), ("RegionF", "F")]:
        rec = per[r]
        its = [p["it"] for p in rec["curve"]]
        costs = [p["cost"] for p in rec["curve"]]
        imp = [p["improvement"] for p in rec["curve"]]
        series[name] = dict(
            its=its, costs=costs, imp=imp,
            edf=rec["cost_edf"], b2=rec["cost_b2"], final=rec["cost_final"],
            n_iter=rec["iterations"],
        )
        arr = np.asarray(costs)
        # 末段斜率（后 1/4 轮成本变化/轮次）
        tail = arr[len(arr) // 4 * 3:]
        slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
        gain_edf = rec["cost_edf"] - rec["cost_final"]
        print(f"[{name} 区收敛曲线统计量]")
        print(f"  it: {its[0]}..{its[-1]} (共 {len(its)} 轮, params.max_iter="
              f"{d['params']['max_iter']})")
        print(f"  cost: min={arr.min():.3f} max={arr.max():.3f} "
              f"mean={arr.mean():.3f} std={arr.std():.3f}")
        print(f"  start(it0)={arr[0]:.3f} end={arr[-1]:.3f} "
              f"总下降={arr[0]-arr[-1]:.3f}M")
        print(f"  EDF 基线={rec['cost_edf']:.3f} | B2 起点={rec['cost_b2']:.3f} "
              f"| LNS 终值={rec['cost_final']:.3f}")
        print(f"  EDF→LNS 增益={gain_edf:.3f}M ({gain_edf/rec['cost_edf']*100:.2f}%)")
        print(f"  末段(后1/4)斜率={slope:.4f} M/轮 (|slope|<0.02 视为已趋平)")
        print(f"  improvement: min={min(imp):.3f} max={max(imp):.3f} "
              f"mean={np.mean(imp):.3f}")

    print(f"\n[聚合口径] EDF 合计={cmp_['edf_total_M']:.2f}M | B2 合计="
          f"{cmp_['b2_total_M']:.2f}M | LNS 合计={cmp_['lns_total_M']:.2f}M | "
          f"EDF→LNS 增益={cmp_['edf_gain_pct']:.2f}% (达 clean-test 上界 "
          f"{cmp_['upper_bound_pct']}% 的 {cmp_['pct_of_upper']:.0f}%)")

    # ================= 第二阶段：绘图 =================
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    x_all = series["E"]["its"]
    for name, c in [("E", C_E), ("F", C_F)]:
        s = series[name]
        ax.plot(s["its"], s["costs"], "-o", color=c, lw=1.8, ms=4.5,
                label=f"{name} 区 LNS", zorder=3)
        # EDF 基线（水平虚线）
        ax.axhline(s["edf"], color=c, lw=1.0, ls="--", alpha=0.6,
                   label=f"{name} 区 EDF 基线")
        # 终值/B2 起点注解已删：信息由图注（caption）说明，避免压曲线（审查用户指令）

    ax.set_xlabel("LNS 迭代轮次", fontsize=11)
    ax.set_ylabel("区域运行成本 (M 元)", fontsize=11)
    ax.set_title("LNS 大邻域搜索收敛曲线（E/F 区，50 轮）", fontsize=12)
    ax.set_xlim(0, len(x_all) + 8)
    ax.grid(True, which="major", lw=0.3, color="#dddddd", zorder=0)
    # 图例置于图框右侧外侧（右上角对齐），四条纵向单列排列（用户指令）
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              fontsize=9)

    fig.tight_layout()
    out = FIG / "sub4-lns-convergence.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\n→ 已保存 {out}")


if __name__ == "__main__":
    main()

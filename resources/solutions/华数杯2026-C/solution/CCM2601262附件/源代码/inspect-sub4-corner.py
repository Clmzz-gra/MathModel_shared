# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    S4 阶段 2.2 拐角解检测 — 统计主解各区域连续变量
    （购电 G / 卖电 S / 放电 D / 充电 C / SOC E）触碰约束边界的比例。

原理：
    拐角解主导判据：若 >80% 变量在边界上，标记"拐角解主导"。
    G 上界 = MaxGridImport（购电撞顶）、S 下界 0 或上界 SellLimit、
    D 上界 = MaxDischargePower、C = Cg+Cr 上界 = MaxChargePower、
    E 触碰 MinSOC 或 Capacity 视为边界。逐时统计触碰比例（0-2399 主时域）。

输入数据：
    - outputs/data/s4-results.pkl — main.sols.{r}.{G,S,Cg,Cr,D,E}
    - outputs/data/sub4-preprocessed.pkl — power.{r}.{MaxGridImport_MW,
      SellLimit_MW, MaxDischargePower_MW, MaxChargePower_MW,
      MinSOC_MWh, Capacity_MWh}

输出：
    - 控制台统计表（PR-014，仅统计不重算）
"""
import pickle
from pathlib import Path

import numpy as np

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
MAIN = 2400
TOL = 1e-6


def main():
    with open(DATA / "s4-results.pkl", "rb") as f:
        s4 = pickle.load(f)
    with open(DATA / "sub4-preprocessed.pkl", "rb") as f:
        d = pickle.load(f)
    power = d["power"]
    storage = d["storage"]

    main_agg = s4["main"]
    print(f"{'区域':<10}{'G撞顶%':>8}{'G零%':>7}{'S上界%':>8}{'S零%':>7}"
          f"{'D上限%':>8}{'D零%':>7}{'C上限%':>8}{'E边界%':>8}{'边界合计%':>10}")
    ratios = {}
    for r in REGIONS:
        sol = main_agg["sols"][r]
        p = power[r]
        G = sol["G"][:MAIN]
        S = sol["S"][:MAIN]
        D = sol["D"][:MAIN]
        Cg = sol["Cg"][:MAIN]
        Cr = sol["Cr"][:MAIN]
        E = sol["E"][:MAIN]

        st = storage[r]
        g_up = st["MaxGridImport_MW"]
        s_up = st["SellLimit_MW"]
        d_up = st["MaxDischargePower_MW"]
        c_up = st["MaxChargePower_MW"]
        e_lo = st["MinSOC_MWh"]
        e_hi = st["Capacity_MWh"]

        n = MAIN
        g_hit = np.mean(G >= g_up - TOL) * 100
        g_zero = np.mean(G <= TOL) * 100
        s_hit = np.mean(S >= s_up - TOL) * 100
        s_zero = np.mean(S <= TOL) * 100
        d_hit = np.mean(D >= d_up - TOL) * 100
        d_zero = np.mean(D <= TOL) * 100
        c_hit = np.mean((Cg + Cr) >= c_up - TOL) * 100
        e_hit = np.mean((E <= e_lo + TOL) | (E >= e_hi - TOL)) * 100

        # 边界合计：每个变量触碰任一自身边界的平均比例
        n_var = 6
        tot = (g_hit + s_hit + d_hit + c_hit + e_hit
               + np.mean(G <= TOL) * 0 + s_zero * 0 + d_zero * 0) / n_var
        ratios[r] = tot
        print(f"{r:<10}{g_hit:>8.1f}{g_zero:>7.1f}{s_hit:>8.1f}{s_zero:>7.1f}"
              f"{d_hit:>8.1f}{d_zero:>7.1f}{c_hit:>8.1f}{e_hit:>8.1f}{tot:>10.3f}")

    agg = float(np.mean(list(ratios.values())))
    print(f"\n聚合拐角解比例（边界触碰均值）: {agg:.3f}")
    print("判据：>0.80 → '拐角解主导'；0.50-0.80 → 偏高，须注明结构性绑定来源")
    print("边界参数（每区）：")
    for r in REGIONS:
        st = storage[r]
        print(f"  {r}: MaxImport={st['MaxGridImport_MW']:.0f}MW "
              f"SellLimit={st['SellLimit_MW']:.0f} Discharge={st['MaxDischargePower_MW']:.0f} "
              f"Charge={st['MaxChargePower_MW']:.0f} SOC[{st['MinSOC_MWh']:.0f},{st['Capacity_MWh']:.0f}]MWh")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
review-m10-s3nostorage.py — 审查补实验 M10：S3 无储能-自由购电 LP 对照

1. 目的
   分离 S3 "储能价值"（−7.3% 成本，vs 无储能口径 c）中"储能充放调度"与
   "购电时点自由化"两部分的贡献（review-notes M10 / critical-reading 脆弱点 19）。
   Δ₂ = 2387.86 − X（购电时点自由化价值）；Δ₁ = X − 2213.35（储能充放价值）；
   Δ₁ + Δ₂ ≈ 2387.86 − 2213.35 = 174.51 M 元。

2. 原理
   - 无储能模式：D_t = C^g_t = C^r_t = 0，SOC ≡ Initial（终态自然满足）；
     但保留 G 购电（0 ≤ G ≤ MaxGridImport）、S 卖电（0 ≤ S ≤ SellLimit）、
     R 新能源直供（受限消纳 R ≤ UsedRenewable+RenewableCharge，C2 口径不变）
   - 约束（3 变量 × 2406 时点）：
     C1 功率平衡（等式）：G + R = Load + S
     C2 受限消纳（≤）：R ≤ cap_t
     C7 碳 ε=1.00（主时域）：Σ G·CI ≤ 1e3·carbon_base_kt
   - 目标：min Σ(G·Price − S·SellPrice)，t=0..2405（与主解同口径）
   - 求解：scipy.optimize.linprog（HiGHS），逐区域独立，6 次 LP

3. 输入映射
   - outputs/data/s3-preprocessed.pkl（panel/storage/carbon_base_kt）
   - outputs/data/csv/region_time_data/region_time_data.csv（消纳上限列）
   - outputs/data/cache/s3_solutions.pkl（主解聚合 2213.35 / 无储能c 2387.86 对照，
     在 main() 中读取，不硬编码）

4. 输出
   - 控制台：每区域无储能自由购电 LP 成本 X + 聚合值 + Δ₁/Δ₂ 拆分表 + 退化检查

5. 论文章节
   - 问题三（S3）储能协同优化：储能价值归因拆分（阶段 3.1 审查修正）
"""
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

BASE = Path(r"E:\MathModel_pj-2026-C-sub3")
DATA = BASE / "outputs" / "data"
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
NT = 2406
MAIN = 2400


def build_absorb_cap():
    """受限消纳上限 cap_t = UsedRenewable + RenewableCharge（与主解同口径）"""
    rtd = pd.read_csv(DATA / "csv" / "region_time_data" / "region_time_data.csv")
    d = rtd[rtd["Hour"] <= 2405].copy()
    col = d["UsedRenewable_MW"] + d["RenewableCharge_MW"]
    s = pd.Series(col.values, index=pd.MultiIndex.from_arrays([d["Region"], d["Hour"]]))
    return s.sort_index()


def solve_nostorage(panel_r, s, carbon_base_kt, cap_t):
    """单区域无储能-自由购电 LP；返回成本/碳排/峰值/std/状态"""
    price = panel_r["Price_CNY_per_MWh"].values
    sellp = panel_r["SellPrice_CNY_per_MWh"].values
    ci = panel_r["CarbonIntensity_tCO2_per_MWh"].values
    load = panel_r["Total_Load_MW"].values
    max_imp = s["MaxGridImport_MW"]
    sell_lim = s["SellLimit_MW"]

    n_vars = 3 * NT
    off = {"G": 0, "S": NT, "R": 2 * NT}

    # 不等式：C2(NT) + C7(1)
    A_ub = lil_matrix((NT + 1, n_vars), dtype=float)
    b_ub = np.zeros(NT + 1)
    for t in range(NT):
        A_ub[t, off["R"] + t] = 1.0
        b_ub[t] = cap_t[t]
    for t in range(MAIN):
        A_ub[NT, off["G"] + t] = ci[t]
    b_ub[NT] = 1e3 * 1.00 * carbon_base_kt

    # 等式：C1(NT)
    A_eq = lil_matrix((NT, n_vars), dtype=float)
    b_eq = np.zeros(NT)
    for t in range(NT):
        A_eq[t, off["G"] + t] = 1.0
        A_eq[t, off["R"] + t] = 1.0
        A_eq[t, off["S"] + t] = -1.0
        b_eq[t] = load[t]

    c = np.zeros(n_vars)
    c[off["G"]:off["G"] + NT] = price
    c[off["S"]:off["S"] + NT] = -sellp
    bounds = ([(0.0, max_imp)] * NT + [(0.0, sell_lim)] * NT + [(0.0, None)] * NT)

    t0 = time.perf_counter()
    res = linprog(c, A_ub=A_ub.tocsr(), b_ub=b_ub, A_eq=A_eq.tocsr(), b_eq=b_eq,
                  bounds=bounds, method="highs")
    dt = time.perf_counter() - t0
    if res.status != 0:
        return {"feasible": False, "status": res.status, "message": str(res.message), "time_s": dt}

    x = res.x
    G = x[off["G"]:off["G"] + NT]
    S = x[off["S"]:off["S"] + NT]
    R = x[off["R"]:off["R"] + NT]
    net = G - S
    cost_main = float(np.sum(G[:MAIN] * price[:MAIN] - S[:MAIN] * sellp[:MAIN])) / 1e6
    carbon_kt = float(np.sum(G[:MAIN] * ci[:MAIN])) / 1e3
    return {"feasible": True, "status": res.status, "time_s": dt,
            "cost_main_M": cost_main, "carbon_kt": carbon_kt,
            "peak_MW": float(net[:MAIN].max()), "std_MW": float(net[:MAIN].std()),
            "G_mean": float(G.mean()), "S_mean": float(S.mean()), "R_mean": float(R.mean()),
            "pb_resid_MW": float(np.abs(G + R - load - S).max())}


def main():
    with open(DATA / "s3-preprocessed.pkl", "rb") as f:
        d = pickle.load(f)
    panel = d["panel"]
    storage = d["storage"]
    carbon_base = d["carbon_base_kt"]
    cap_t = build_absorb_cap()
    # 对照基准自 s3_solutions.pkl 读取（不硬编码，防上游重算漂移）
    with open(DATA / "cache" / "s3_solutions.pkl", "rb") as f:
        sol = pickle.load(f)
    main_sol_cost = sol["aggregate"][1.00]["cost_M"]
    no_storage_c = sol["compare"]["no_storage_c"]["cost_M"]
    print(f"对照基准（s3_solutions.pkl）: 储能主解 {main_sol_cost:.2f} | 无储能口径c {no_storage_c:.2f} M元")

    print("=" * 78)
    print("M10 无储能-自由购电 LP 对照（分离：储能充放 vs 购电时点自由化）")
    print("=" * 78)
    results = {}
    for r in REGIONS:
        res = solve_nostorage(panel.xs(r), storage[r], carbon_base[r], cap_t.xs(r).values)
        results[r] = res
        if res["feasible"]:
            print(f"  {r}: 成本 {res['cost_main_M']:8.2f} M元 | 碳 {res['carbon_kt']:8.2f} kt | "
                  f"峰值净购电 {res['peak_MW']:7.1f} MW | G均值 {res['G_mean']:6.1f} | "
                  f"S均值 {res['S_mean']:5.2f} | R均值 {res['R_mean']:6.1f} | "
                  f"残差 {res['pb_resid_MW']:.1e} | {res['time_s']:.2f}s")
        else:
            print(f"  {r}: **不可行** status={res['status']}（LP 退化？）")

    X = sum(res["cost_main_M"] for res in results.values() if res["feasible"])
    feasible_r = [r for r in REGIONS if results[r]["feasible"]]
    print("\n" + "=" * 78)
    print("Δ₁/Δ₂ 拆分（聚合，主时域 0–2399）")
    print("=" * 78)
    if len(feasible_r) != len(REGIONS):
        print(f"  ⚠️ 存在不可行区域（{feasible_r}），X 为可行子集口径，"
              f"与全区域基准 {no_storage_c:.2f}/{main_sol_cost:.2f} 不一致 → Δ 拆分不成立，仅报告每区域值")
    print(f"  无储能-自由购电 LP X = {X:9.2f} M元（可行区域 {feasible_r}）")
    print(f"  储能 LP 主解       = {main_sol_cost:9.2f} M元")
    print(f"  无储能固定购电口径c = {no_storage_c:9.2f} M元")
    if len(feasible_r) == len(REGIONS):
        print(f"  Δ₂（购电时点自由化价值） = {no_storage_c - X:8.2f} M元")
        print(f"  Δ₁（储能充放价值）      = {X - main_sol_cost:8.2f} M元")
        print(f"  合计 Δ₁+Δ₂ = {(no_storage_c - X) + (X - main_sol_cost):.2f} M元 "
              f"（应={no_storage_c - main_sol_cost:.2f}）")
    # 退化检查
    print("\n  退化检查：")
    print(f"  可行 {len(feasible_r)}/6 区域；无储能下 S 均值（D/E/F）≈0 说明卖电无套利空间")
    print("  G 仍为变量（购电时点自由）→ 若 X 明显低于 {no_storage_c:.2f} 则购电时点自由化贡献显著".format(
        no_storage_c=no_storage_c))
    print("M10 NOSTORAGE LP DONE (review-m10-s3nostorage.py)")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
sub3-model.py — S3 储能协同优化：LP 求解（阶段 2.1 正式模型）

1. 目的
   实现主方案 A（储能协同优化 LP）：受限消纳口径（B1），主解 ε=1.00 单档
   （D4 修订：ε=0.90/0.95 不可行，碳排作评价指标，见建模核验 §10），最小化运行成本
   （购电 − 卖电收入），输出逐时充放/购售电/新能源分配决策序列与四项指标
   （成本/碳排/区域峰值净购电/负荷波动）对比，附 ε_min/撞顶/B3 诊断，落盘 s3_solutions.pkl。

2. 原理
   - 决策变量（每区域 7×2406）：G 购电 / S 卖电 / R 新能源直供 / Cg 电网充电 /
     Cr 新能源充电 / D 放电 / E SOC（时段末状态，E_{-1}=InitialSOC）
   - 约束（math-sub3.tex §2–§4；C2 按 handoff §11 裁定 1 修改为受限消纳）：
     C1 功率平衡：G + R + D = Load + Cg + S
     C2 受限消纳上限（B1 主口径，默认开启）：R + Cr ≤ UsedRenewable + RenewableCharge
        （= 基准观测的逐时消纳能力；--free 时回退自由消纳 R + Cr ≤ Avail）
     C3 SOC 递推：E_t − E_{t−1} − ηc(Cg+Cr) + D/ηd = 0
     C4 SOC 边界：Min ≤ E ≤ Cap；终态 E_2405 ≥ Initial
     C5 充放功率：Cg+Cr ≤ MaxCharge，D ≤ MaxDischarge
     C6 购售电边界：G ≤ MaxGridImport，S ≤ SellLimit
     C7 碳 ε-约束（主时域 0–2399）：Σ G_t·CI_t ≤ 1e3·ε·carbon_base_kt（kt→tCO₂ 换算）
   - 目标：min Σ (G_t·Price_t − S_t·SellPrice_t)，t=0..2405（含结清段结算）
   - 净购电 = G − S（与 NetGridImport 同义）；同刻充放由目标自动排除（ηc·ηd<1，F4 实证）
   - 求解：scipy.optimize.linprog（HiGHS），稀疏矩阵（lil→csr），逐区域独立
   - 碳 ε 两阶段标定（裁定 2）：先 --eps 1.00 单档回报 C* 与卖电构成；核验后确认 ε=0.90/0.95
     不可行（ε_min>0.95）→ D4 修订：主解 ε=1.00 单档，碳排作评价指标，Pareto 三档设计失效

3. 输入映射
   - outputs/data/s3-preprocessed.pkl（阶段 1.4 产物：panel/storage/carbon_base_kt/epsilon）

4. 输出
   - outputs/data/cache/s3_solutions.pkl（每 (region,ε) 时序 + 指标 + 聚合表 + 自洽性检查）
   - 控制台：18 次 LP 状态、四指标对比、ε Pareto、自洽性三项、拐角解统计、数组统计量（PR-014）

5. 论文章节
   - 问题三 储能协同优化：模型构建与求解（阶段 2.0/2.1）
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

DATA = Path(r"E:\MathModel_pj-2026-C-sub3\outputs\data")
PKL_IN = DATA / "s3-preprocessed.pkl"
PKL_OUT = DATA / "cache" / "s3_solutions.pkl"
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
EPS_FULL = [0.90, 0.95, 1.00]
NT = 2406          # 全时域 0–2405
MAIN = 2400        # 主时域 0–2399
VAR_NAMES = ["G", "S", "R", "Cg", "Cr", "D", "E"]   # 7 变量块顺序
DOL_TOL = 0.01     # 同刻充放判定阈值（与 verify-sub3 一致）
CORNER_TOL = 1e-6  # 拐角解判定容差


def describe(name, arr):
    a = np.asarray(arr, dtype=float)
    print(f"    [{name}] min {a.min():10.3f} | max {a.max():10.3f} | "
          f"mean {a.mean():10.3f} | std {a.std():10.3f}")


def solve_region(region, panel_r, s, carbon_base_kt, eps, absorb_cap, free=False, carbon_mode=False):
    """单区域单 ε 档 LP 求解；返回决策时序 + 指标 + 状态

    absorb_cap: 逐时新能源消纳上限（受限口径 = Used+RenewableCharge；--free 时 = Avail）
    carbon_mode: True 时目标改为最小化主时域碳排、去掉 C7 碳约束 → 得碳排下限 C_min（ε_min 标定用）
    status!=0 时返回占位记录（feasible=False），主循环跳过，不中断全档（建模核验 §10 问题 1）"""
    price = panel_r["Price_CNY_per_MWh"].values
    sellp = panel_r["SellPrice_CNY_per_MWh"].values
    ci = panel_r["CarbonIntensity_tCO2_per_MWh"].values
    load = panel_r["Total_Load_MW"].values
    avail = panel_r["AvailableRenewable_MW"].values

    cap, min_soc, init = s["Capacity_MWh"], s["MinSOC_MWh"], s["InitialSOC_MWh"]
    max_ch, max_dis = s["MaxChargePower_MW"], s["MaxDischargePower_MW"]
    eta_c, eta_d = s["ChargeEfficiency"], s["DischargeEfficiency"]
    sell_lim, max_imp = s["SellLimit_MW"], s["MaxGridImport_MW"]

    n_vars = 7 * NT
    off = {name: k * NT for k, name in enumerate(VAR_NAMES)}
    cap_t = np.asarray(absorb_cap, dtype=float)   # 逐时消纳上限（len=NT）

    # ---- 不等式：C2(2406) + C5充(2406) + [C7碳(1)] + C4终态(1) ----
    n_ub = 2 * NT + (0 if carbon_mode else 1) + 1
    A_ub = lil_matrix((n_ub, n_vars), dtype=float)
    b_ub = np.zeros(n_ub)
    row = 0
    for t in range(NT):  # C2: R + Cr <= cap_t（受限消纳 B1 / 自由消纳 --free）
        A_ub[row, off["R"] + t] = 1.0
        A_ub[row, off["Cr"] + t] = 1.0
        b_ub[row] = cap_t[t]
        row += 1
    for t in range(NT):  # C5 充电: Cg + Cr <= MaxCharge
        A_ub[row, off["Cg"] + t] = 1.0
        A_ub[row, off["Cr"] + t] = 1.0
        b_ub[row] = max_ch
        row += 1
    if not carbon_mode:  # C7 碳约束（主时域，kt→tCO₂ ×1e3）
        for t in range(MAIN):
            A_ub[row, off["G"] + t] = ci[t]
        b_ub[row] = 1e3 * eps * carbon_base_kt
        row += 1
    A_ub[row, off["E"] + NT - 1] = -1.0  # C4 终态: -E_2405 <= -Initial
    b_ub[row] = -init
    row += 1
    assert row == n_ub

    # ---- 等式：C1(2406) + C3(2406) ----
    A_eq = lil_matrix((2 * NT, n_vars), dtype=float)
    b_eq = np.zeros(2 * NT)
    row = 0
    for t in range(NT):  # C1: G + R + D - Cg - S = Load
        A_eq[row, off["G"] + t] = 1.0
        A_eq[row, off["R"] + t] = 1.0
        A_eq[row, off["D"] + t] = 1.0
        A_eq[row, off["Cg"] + t] = -1.0
        A_eq[row, off["S"] + t] = -1.0
        b_eq[row] = load[t]
        row += 1
    for t in range(NT):  # C3: E_t - E_{t-1} - ηc(Cg+Cr) + D/ηd = 0
        A_eq[row, off["E"] + t] = 1.0
        if t >= 1:
            A_eq[row, off["E"] + t - 1] = -1.0
        A_eq[row, off["Cg"] + t] = -eta_c
        A_eq[row, off["Cr"] + t] = -eta_c
        A_eq[row, off["D"] + t] = 1.0 / eta_d
        b_eq[row] = init if t == 0 else 0.0
        row += 1
    assert row == 2 * NT

    # ---- 目标与边界 ----
    c = np.zeros(n_vars)
    if carbon_mode:                     # ε_min 标定：最小化主时域碳排，无卖电/成本激励
        c[off["G"]:off["G"] + MAIN] = ci[:MAIN]
    else:
        c[off["G"]:off["G"] + NT] = price
        c[off["S"]:off["S"] + NT] = -sellp
    bounds = ([(0.0, max_imp)] * NT + [(0.0, sell_lim)] * NT + [(0.0, None)] * NT
              + [(0.0, None)] * NT + [(0.0, None)] * NT + [(0.0, max_dis)] * NT
              + [(min_soc, cap)] * NT)

    t0 = time.perf_counter()
    res = linprog(c, A_ub=A_ub.tocsr(), b_ub=b_ub, A_eq=A_eq.tocsr(), b_eq=b_eq,
                  bounds=bounds, method="highs")
    dt = time.perf_counter() - t0
    if res.status != 0:
        print(f"    [警告] {region} ε={eps}: linprog status={res.status} {res.message}"
              f"（不可行/非最优，跳过，不中断全档）")
        return {"region": region, "eps": eps, "status": res.status,
                "message": str(res.message), "feasible": False, "time_s": dt}

    x = res.x
    G = x[off["G"]:off["G"] + NT]
    S = x[off["S"]:off["S"] + NT]
    R = x[off["R"]:off["R"] + NT]
    Cg = x[off["Cg"]:off["Cg"] + NT]
    Cr = x[off["Cr"]:off["Cr"] + NT]
    D = x[off["D"]:off["D"] + NT]
    E = x[off["E"]:off["E"] + NT]

    net = G - S
    cost_main = float(np.sum(G[:MAIN] * price[:MAIN] - S[:MAIN] * sellp[:MAIN])) / 1e6
    cost_full = float(np.sum(G * price - S * sellp)) / 1e6
    carbon_kt = float(np.sum(G[:MAIN] * ci[:MAIN])) / 1e3
    peak_mw = float(net[:MAIN].max())
    std_mw = float(net[:MAIN].std())
    rng_mw = float(net[:MAIN].max() - net[:MAIN].min())
    soc_end = float(E[-1])                     # E_2405 = E_2406（状态结算）
    dual_h = int(np.sum((Cg + Cr > DOL_TOL) & (D > DOL_TOL)))
    pb_resid = float(np.abs(G + R + D - load - Cg - S).max())   # C1 残差
    # 拐角解统计：触及下/上界（有限上界）的变量比例
    ub_list = np.concatenate([np.full(NT, max_imp), np.full(NT, sell_lim),
                              np.full(NT, np.inf), np.full(NT, np.inf),
                              np.full(NT, np.inf), np.full(NT, max_dis),
                              np.full(NT, cap)])
    lb_list = np.concatenate([np.zeros(NT), np.zeros(NT), np.zeros(NT), np.zeros(NT),
                              np.zeros(NT), np.zeros(NT), np.full(NT, min_soc)])
    x_all = x
    at_lb = x_all <= lb_list + CORNER_TOL
    at_ub = (ub_list != np.inf) & (x_all >= ub_list - CORNER_TOL)
    corner = float((at_lb | at_ub).sum()) / n_vars
    # 新能源利用率（主时域，双口径）
    sum_avail = float(avail[:MAIN].sum())
    util_no_sell = float(np.sum(R[:MAIN] + Cr[:MAIN])) / sum_avail * 100
    util_sell = float(np.sum(R[:MAIN] + Cr[:MAIN] + S[:MAIN])) / sum_avail * 100

    # 卖电构成分解（主时域，GWh）：卖电 vs 购电 vs 新能源充电 vs 直供 vs 放电
    sell_gwh = float(np.sum(S[:MAIN])) / 1e3
    g_gwh = float(np.sum(G[:MAIN])) / 1e3
    rc_gwh = float(np.sum(Cr[:MAIN])) / 1e3
    r_gwh = float(np.sum(R[:MAIN])) / 1e3
    d_gwh = float(np.sum(D[:MAIN])) / 1e3

    return {
        "region": region, "eps": eps, "status": res.status, "time_s": dt, "free": free,
        "feasible": True, "carbon_mode": carbon_mode,
        "G": G, "S": S, "R": R, "Cg": Cg, "Cr": Cr, "D": D, "E": E, "net": net,
        "cost_main_M": cost_main, "cost_full_M": cost_full, "carbon_kt": carbon_kt,
        "peak_MW": peak_mw, "std_MW": std_mw, "range_MW": rng_mw, "soc_end_MWh": soc_end,
        "dual_hours": dual_h, "pb_resid_MW": pb_resid, "corner_frac": corner,
        "util_no_sell_pct": util_no_sell, "util_sell_pct": util_sell,
        "sell_gwh": sell_gwh, "g_gwh": g_gwh, "rc_gwh": rc_gwh,
        "r_gwh": r_gwh, "d_gwh": d_gwh,
    }


def no_storage_c(panel_r):
    """无储能口径 c（诚实下界，F1）：Gp = max(Load − Used, 0)，GridSell=0"""
    price = panel_r["Price_CNY_per_MWh"].values
    ci = panel_r["CarbonIntensity_tCO2_per_MWh"].values
    load = panel_r["Total_Load_MW"].values
    used = panel_r["UsedRenewable_MW"].values
    gp = np.maximum(load - used, 0.0)
    cost = float(np.sum(gp[:MAIN] * price[:MAIN])) / 1e6
    carb = float(np.sum(gp[:MAIN] * ci[:MAIN])) / 1e3
    return dict(cost_M=cost, carbon_kt=carb, peak_MW=float(gp[:MAIN].max()),
                std_MW=float(gp[:MAIN].std()), range_MW=float(gp[:MAIN].max() - gp[:MAIN].min()))


def benchmark(panel_r):
    """基准轨迹（F2，参考）：用 panel 的 G_base / NetGridImport_base"""
    price = panel_r["Price_CNY_per_MWh"].values
    sellp = panel_r["SellPrice_CNY_per_MWh"].values
    ci = panel_r["CarbonIntensity_tCO2_per_MWh"].values
    gb = panel_r["GridPurchase_base_MW"].values
    nb = panel_r["NetGridImport_base_MW"].values   # = G_base − S_base
    sb = gb - nb
    cost = float(np.sum(gb[:MAIN] * price[:MAIN] - sb[:MAIN] * sellp[:MAIN])) / 1e6
    carb = float(np.sum(gb[:MAIN] * ci[:MAIN])) / 1e3
    return dict(cost_M=cost, carbon_kt=carb, peak_MW=float(nb[:MAIN].max()),
                std_MW=float(nb[:MAIN].std()), range_MW=float(nb[:MAIN].max() - nb[:MAIN].min()))


def build_absorb_cap(free=False):
    """逐时新能源消纳上限：受限口径 cap = UsedRenewable + RenewableCharge（裁定 1）；
    --free 时 cap = AvailableRenewable。回读 region_time_data.csv（阶段 0.2 缓存，不改 pkl）。"""
    rtd = pd.read_csv(DATA / "csv" / "region_time_data" / "region_time_data.csv")
    d = rtd[rtd["Hour"] <= 2405].copy()
    col = (d["AvailableRenewable_MW"] if free
           else d["UsedRenewable_MW"] + d["RenewableCharge_MW"])
    s = pd.Series(col.values, index=pd.MultiIndex.from_arrays([d["Region"], d["Hour"]]))
    return s.sort_index()


def main():
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="S3 储能协同优化 LP（阶段 2.1）")
    parser.add_argument("--eps", nargs="+", type=float, default=EPS_FULL,
                        help="碳 ε 档位集（两阶段标定：第一阶段 1.00 单档，建模定档后全档重跑）")
    parser.add_argument("--free", action="store_true",
                        help="自由消纳（原 C2 R+Cr<=Avail，R1 退化口径，默认关闭）")
    args = parser.parse_args()
    eps_list = sorted(set(args.eps))

    with open(PKL_IN, "rb") as f:
        d = pickle.load(f)
    panel = d["panel"]
    storage = d["storage"]
    carbon_base = d["carbon_base_kt"]
    absorb = build_absorb_cap(free=args.free)

    mode = "自由消纳(--free)" if args.free else "受限消纳 B1 (cap=Used+RenewableCharge)"
    print("=" * 78)
    print("S3 阶段 2.1 LP 求解（主方案 A：成本最小化 + 碳 ε-约束）")
    print(f"数据源: {PKL_IN} | 消纳口径: {mode} | ε 档: {eps_list}")
    print("=" * 78)

    # ---- 第一阶段：纯计算（|ε| × 6 次 LP）----
    solutions = {}
    for r in REGIONS:
        panel_r = panel.xs(r)
        cap_r = absorb.xs(r).values
        s = storage[r]
        print(f"\n{r}（Cap {s['Capacity_MWh']:.0f} MWh, ηc {s['ChargeEfficiency']}, "
              f"SellLimit {s['SellLimit_MW']:.0f}, MaxImport {s['MaxGridImport_MW']:.0f}）")
        for eps in eps_list:
            sol = solve_region(r, panel_r, s, carbon_base[r], eps, cap_r, free=args.free)
            solutions[(r, eps)] = sol
            if not sol["feasible"]:
                print(f"  ε={eps:.2f}: **不可行** status={sol['status']}（跳过聚合，全档继续）")
                continue
            print(f"  ε={eps:.2f}: status {sol['status']} | 成本(主/全) {sol['cost_main_M']:9.2f}/"
                  f"{sol['cost_full_M']:9.2f} M元 | 碳 {sol['carbon_kt']:8.2f}/{eps*carbon_base[r]:6.2f} kt"
                  f"(上限) | 峰值净购电 {sol['peak_MW']:7.1f} MW | 净购电std {sol['std_MW']:6.2f} | "
                  f"SOC末 {sol['soc_end_MWh']:7.1f} | 同刻充放 {sol['dual_hours']}h | "
                  f"用时 {sol['time_s']:.2f}s")
            print(f"      [卖电构成·主时域] S {sol['sell_gwh']:6.1f} | G {sol['g_gwh']:6.1f} | "
                  f"Cr {sol['rc_gwh']:5.1f} | R {sol['r_gwh']:6.1f} | D {sol['d_gwh']:5.1f} GWh")
            for vn in VAR_NAMES:   # PR-014 数组统计量（handoff §3.7）
                describe(vn, sol[vn])

    # ---- 聚合与对照 ----
    print("\n" + "=" * 78)
    print("聚合与对照（主时域 0–2399）")
    print("=" * 78)
    aggregate = {}
    infeasible = {}
    for eps in eps_list:
        feas_r = [r for r in REGIONS if solutions[(r, eps)].get("feasible", False)]
        miss = [r for r in REGIONS if r not in feas_r]
        if miss:
            infeasible[eps] = miss
        if not feas_r:
            aggregate[eps] = None
            print(f"  ε={eps:.2f}: 全部区域不可行（无聚合）")
            continue
        costs = sum(solutions[(r, eps)]["cost_main_M"] for r in feas_r)
        carbs = sum(solutions[(r, eps)]["carbon_kt"] for r in feas_r)
        peak = max(solutions[(r, eps)]["peak_MW"] for r in feas_r)
        netsum = sum(solutions[(r, eps)]["net"][:MAIN] for r in feas_r)   # 逐时区域加总
        util_no = sum(solutions[(r, eps)]["util_no_sell_pct"] * 0.01
                      * np.sum(panel.xs(r)["AvailableRenewable_MW"][:MAIN]) for r in feas_r)
        util_sell = sum(solutions[(r, eps)]["util_sell_pct"] * 0.01
                        * np.sum(panel.xs(r)["AvailableRenewable_MW"][:MAIN]) for r in feas_r)
        tot_avail = sum(np.sum(panel.xs(r)["AvailableRenewable_MW"][:MAIN]) for r in feas_r)
        aggregate[eps] = dict(
            cost_M=costs, carbon_kt=carbs, peak_MW=peak, std_MW=float(netsum.std()),
            range_MW=float(netsum.max() - netsum.min()),
            util_no_sell_pct=float(util_no / tot_avail * 100),
            util_sell_pct=float(util_sell / tot_avail * 100),
            corner_frac=float(np.mean([solutions[(r, eps)]["corner_frac"] for r in feas_r])),
        )
        print(f"  ε={eps:.2f}: 成本 {costs:9.2f} M元 | 碳 {carbs:8.2f} kt | "
              f"峰值净购电 {peak:7.1f} MW | 净购电std {aggregate[eps]['std_MW']:6.2f} | "
              f"利用率(不含/含外送) {aggregate[eps]['util_no_sell_pct']:.2f}/{aggregate[eps]['util_sell_pct']:.2f}% | "
              f"拐角比例 {aggregate[eps]['corner_frac']:.1%}"
              + (f" | 缺省区域: {miss}" if miss else ""))

    # 对照（无储能口径 c + 基准轨迹，主时域）
    no_c = {r: no_storage_c(panel.xs(r)) for r in REGIONS}
    ben = {r: benchmark(panel.xs(r)) for r in REGIONS}
    compare = {
        "no_storage_c": dict(cost_M=sum(v["cost_M"] for v in no_c.values()),
                             carbon_kt=sum(v["carbon_kt"] for v in no_c.values()),
                             peak_MW=max(v["peak_MW"] for v in no_c.values())),
        "benchmark": dict(cost_M=sum(v["cost_M"] for v in ben.values()),
                          carbon_kt=sum(v["carbon_kt"] for v in ben.values()),
                          peak_MW=max(v["peak_MW"] for v in ben.values())),
    }
    print("\n  四指标对比（聚合，主时域）")
    print("  | 方案 | 成本 M元 | 碳 kt | 峰值净购电 MW |")
    eps_ref = max(eps_list)
    for tag, rec in ((f"优化 ε={eps_ref:.2f}", aggregate[eps_ref]), ("无储能口径c", compare["no_storage_c"]),
                     ("基准轨迹(参考)", compare["benchmark"])):
        print(f"  | {tag:14s} | {rec['cost_M']:9.2f} | {rec['carbon_kt']:8.2f} | "
              f"{rec['peak_MW']:12.1f} |")

    # ---- 自洽性检查（仅可行解）----
    print("\n自洽性检查")
    feas_pairs = [(r, e) for r in REGIONS for e in eps_list if solutions[(r, e)].get("feasible", False)]
    all_ok_status = all(solutions[p]["status"] == 0 for p in feas_pairs)
    dual_max = max((solutions[p]["dual_hours"] for p in feas_pairs), default=0)
    soc_ok = all(solutions[p]["soc_end_MWh"] >= storage[p[0]]["InitialSOC_MWh"] - 1e-6
                 for p in feas_pairs)
    pb_max = max((solutions[p]["pb_resid_MW"] for p in feas_pairs), default=0.0)
    mono = (all(aggregate[e1]["cost_M"] >= aggregate[e2]["cost_M"] - 1e-6
                for e1, e2 in zip(eps_list[:-1], eps_list[1:])
                if aggregate[e1] is not None and aggregate[e2] is not None)
            if len(eps_list) >= 2 else True)
    print(f"  可行解 {len(feas_pairs)}/{len(eps_list) * 6} 档 | 全 status=0: {all_ok_status} | "
          f"同刻充放 max: {dual_max} h（须=0）")
    print(f"  SOC(2405)≥Initial 全过: {soc_ok} | 功率平衡残差 max: {pb_max:.2e} MW（≤1e-3）")
    print(f"  ε 单调性（可行档间成本非增）: {mono} | 不可行档: "
          f"{infeasible if infeasible else '无'}")

    # ---- 诊断：ε_min（碳排下限）/ 撞顶 / B3（建模核验 §9/§10）----
    print("\n诊断（建模核验 §9/§10）")
    eps_min = {}
    for r in REGIONS:
        solc = solve_region(r, panel.xs(r), storage[r], carbon_base[r], None,
                            absorb.xs(r).values, free=args.free, carbon_mode=True)
        if solc["feasible"]:
            cmin = solc["carbon_kt"]
            eps_min[r] = {"c_min_kt": cmin, "eps_min": cmin / carbon_base[r]}
            print(f"  {r}: C_min {cmin:7.2f} kt | ε_min {cmin / carbon_base[r]:.4f}"
                  f"（建模二分实测 0.9935/0.9939/0.9941/0.9694/0.9575/0.9617）")
        else:
            eps_min[r] = {"c_min_kt": None, "eps_min": None}
            print(f"  {r}: ε_min 标定 LP 不可行（异常）")

    atcap = {}
    if 1.00 in eps_list:
        for r in REGIONS:
            sol = solutions.get((r, 1.00))
            if not sol or not sol.get("feasible", False):
                continue
            G = sol["G"][:MAIN]
            max_imp = storage[r]["MaxGridImport_MW"]
            idx = np.where(G >= max_imp - 1e-2)[0]
            pr = panel.xs(r)["Price_CNY_per_MWh"].values[:MAIN]
            ci_a = panel.xs(r)["CarbonIntensity_tCO2_per_MWh"].values[:MAIN]
            picks = idx[np.argsort(-ci_a[idx])][:3] if len(idx) else []
            atcap[r] = {"hours": int(len(idx)), "ratio": float(len(idx)) / MAIN,
                        "sample": [(int(h), float(pr[h]), float(ci_a[h])) for h in picks]}
            print(f"  {r} ε=1.00: 撞顶 {len(idx)} h ({len(idx) / MAIN:.1%}) | "
                  f"典型(Hour, Price, CI) {atcap[r]['sample']}")

    b3 = sum(float((panel.xs(r)["GridPurchase_base_MW"]
                    * panel.xs(r)["Price_CNY_per_MWh"]).values[:MAIN].sum()) / 1e6
             for r in REGIONS)
    print(f"  B3 校核：基准剔除卖电收入 = Σ(G_base×Price) = {b3:.2f} M元"
          f"（建模基准 2231.50 ±0.5%，相对偏差 {(b3 - 2231.50) / 2231.50 * 100:+.2f}%）")

    check = {
        "mode": "free" if args.free else "restricted_b1",
        "d4_revision": "main_solution_eps_1.00（ε=0.90/0.95 不可行，碳排作评价指标）",
        "all_status_ok": all_ok_status,
        "dual_hours_max": dual_max,
        "soc_end_ok": soc_ok,
        "pb_resid_max_MW": pb_max,
        "eps_monotone": mono,
        "infeasible": infeasible,
        "eps_min": eps_min,
        "at_cap": atcap,
        "b3_cost_M": b3,
    }

    # ---- 落盘 ----
    out = {
        "meta": {"generated": "2026-08-08",
                 "source": "outputs/data/s3-preprocessed.pkl（阶段 1.4）+ region_time_data.csv（消纳上限）",
                 "regions": REGIONS, "eps": eps_list, "mode": mode,
                 "units": {"cost": "M元", "carbon": "kt", "power": "MW", "energy": "MWh"}},
        "solutions": solutions,
        "aggregate": aggregate,
        "compare": compare,
        "no_storage_c_region": no_c,
        "benchmark_region": ben,
        "check": check,
    }
    PKL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PKL_OUT, "wb") as f:
        pickle.dump(out, f)
    print(f"\n已写入 {PKL_OUT}（{PKL_OUT.stat().st_size / 1e3:.1f} KB）")
    print(f"总用时 {time.perf_counter() - t_start:.1f}s")
    print("S3 LP SOLVING DONE (sub3-model.py)")


if __name__ == "__main__":
    main()

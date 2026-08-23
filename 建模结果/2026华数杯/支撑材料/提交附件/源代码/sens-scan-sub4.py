# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 2.1.6 S4 灵敏度扫描 — 对碳约束 ε / 峰谷价差 / 新能源波动做
    参数网格扫描，输出逐区 + 聚合指标，供灵敏度分析与 Pareto 前沿绘图。

原理：
    复用 sub4-model.solve_region（已验证的层 2 每区域协同 MILP）：
    - ε 扫描：碳约束收紧系数 ε ∈ {0.90..1.00}，基准 = S4 自身 free 解
      E0_S4（阶段 2.1 实证口径）。ε 收紧 → 逐区在可行域内降碳 → 成本-碳排
      Pareto 前沿（若 ε_min 接近 1.0 则前沿极短，如实呈现）。
    - price 扫描：峰段价格 × {1.0..2.0}（只放大 price>均值 的峰段，
      审查 S5 口径），碳基准沿用 E0_S4（price 不改变新能源/碳结构，
      碳约束不绑定，直接可行）。
    - renew 扫描：新能源出力（受限消纳上限 + 利用率分母）× {0.8..1.2}，
      每点先用 free 求解得到该水平下 E0_renew 作 ε 基准（renew 波动移动
      碳排水平，相对 S0 基准的 ε 无意义）。
    六指标与主解同口径（主时域 0-2399）：成本/碳排/峰值净购电/时延/
    QoS/利用率（受限消纳双口径）。附储能充放能与卖电小时（灵敏度的
    机理抓手）。

输入数据：
    - outputs/data/sub4-preprocessed.pkl（阶段 1.4）— tasks/power/storage
    - outputs/data/s3-preprocessed.pkl（阶段 1.4 S3）— panel（结清段实际值）
    - outputs/data/c-data-cleaned.pkl（阶段 0.3）— region_time_data（NonAI +
      受限消纳上限）
    - outputs/data/s2-preprocessed.pkl（阶段 1.4 S2）— latency
    - outputs/scratch/sub4-model.py（阶段 2.1）— solve_region（只复用纯函数）
    - 中文指标 → 变量名映射：购电→G, 卖电→S, 新能源直供→R, 新能源充电→Cr,
      放电→D, SOC→E, 净购电→net=G-S, 成本(M元)→cost_main_M, 碳排(kt)→carbon_kt,
      充电功率→Cg+Cr

输出：
    - outputs/data/s4-sensitivity.pkl — 键：
      eps:   {ε: {agg:{...}, per:{r:{cost,carbon,peak,util,feasible,es}}}}
      price: {scale: {agg, per}}
      renew: {scale: {agg, per}}
      e0_s4_kt / 求解耗时
    - 控制台汇总表（PR-014 核对）

对应论文章节：
    问题四（S4）算-储-电协同优化 — §7 场景设计 / 灵敏度分析
"""
import importlib.util
import pickle
import time
from pathlib import Path

import numpy as np

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"
SCRATCH = BASE / "outputs" / "scratch"

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
MAIN = 2400

EPS_GRID = [0.90, 0.92, 0.94, 0.96, 0.98, 1.00]
PRICE_GRID = [1.0, 1.25, 1.5, 1.75, 2.0]
RENEW_GRID = [0.8, 0.9, 1.0, 1.1, 1.2]

# 加载 sub4-model 模块（仅复用 solve_region 纯函数；其模块级 BASE 指向
# worktree 路径，不在此使用）
_spec = importlib.util.spec_from_file_location(
    "sub4model", SCRATCH / "sub4-model.py")
M4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M4)
solve_region = M4.solve_region


def load():
    """与 sub4-model.load() 同口径，仅 BASE 指向当前目录。"""
    with open(DATA / "sub4-preprocessed.pkl", "rb") as f:
        d = pickle.load(f)
    with open(DATA / "s3-preprocessed.pkl", "rb") as f:
        s3 = pickle.load(f)
    with open(DATA / "c-data-cleaned.pkl", "rb") as f:
        cd = pickle.load(f)
    with open(DATA / "s2-preprocessed.pkl", "rb") as f:
        s2 = pickle.load(f)

    panel = s3["panel"]
    ext = {}
    for r in REGIONS:
        pr = panel.xs(r, level="Region")
        ext[r] = {
            "price": pr["Price_CNY_per_MWh"].values,
            "sellp": pr["SellPrice_CNY_per_MWh"].values,
            "carbon": pr["CarbonIntensity_tCO2_per_MWh"].values,
            "renewable": pr["AvailableRenewable_MW"].values,
        }
    rtd = cd["region_time_data"]
    nonai_full, absorb_full = {}, {}
    for r in REGIONS:
        sub = rtd[rtd["Region"] == r].sort_values("Hour")
        nonai_full[r] = sub["NonAI_IT_Load_MW"].values[:2406].astype(float)
        absorb_full[r] = (sub["UsedRenewable_MW"].values[:2406]
                          + sub["RenewableCharge_MW"].values[:2406]).astype(float)
    return d, ext, nonai_full, absorb_full, s2["latency"]


def per_metrics(sol):
    """从单区解提取灵敏度用的逐区指标。"""
    if not sol["feasible"]:
        return {"feasible": False}
    E = sol["E"]
    charge = sol["Cg"] + sol["Cr"]
    return {
        "feasible": True,
        "cost_main_M": sol["cost_main_M"],
        "carbon_kt": sol["carbon_kt"],
        "peak_MW": sol["peak_MW"],
        "std_MW": sol["std_MW"],
        "util_no_sell_pct": sol["util_no_sell_pct"],
        "util_sell_pct": sol["util_sell_pct"],
        "es_MWh": float(np.sum(charge[:MAIN])),           # 储能充放能（充电侧）
        "sell_h": int(np.sum(sol["S"][:MAIN] > 1e-3)),    # 卖电小时数
        "soc_end_MWh": sol["soc_end_MWh"],
    }


def run_all(tasks, power, storage, ext, nonai_full, absorb_full, latency,
            e0, eps=1.0, price_scale=1.0, renew_scale=1.0, free=False):
    """6 区求解 + 聚合。e0: {r: kt} ε 基准（free=True 时忽略）。"""
    sols = {r: solve_region(r, tasks, power, storage, e0, nonai_full,
                            absorb_full, ext, eps, price_scale, renew_scale,
                            free=free)
            for r in REGIONS}
    feas = [s for s in sols.values() if s["feasible"]]
    per = {r: per_metrics(s) for r, s in sols.items()}
    if not feas:
        return {"feasible": False, "per": per, "sols": sols,
                "n_infeasible": 6}
    gm = M4.global_metrics(tasks, latency, sols)
    agg = {
        "feasible": True,
        "cost_main_M": sum(s["cost_main_M"] for s in feas),
        "carbon_kt": sum(s["carbon_kt"] for s in feas),
        "peak_MW": max(s["peak_MW"] for s in feas),
        "std_avg_MW": float(np.mean([s["std_MW"] for s in feas])),
        "util_no_sell_pct": float(np.mean([s["util_no_sell_pct"] for s in feas])),
        "util_sell_pct": float(np.mean([s["util_sell_pct"] for s in feas])),
        "delay_ms": gm["delay_ms"],
        "qos_pct": gm["qos_pct"],
        "es_MWh": sum(p["es_MWh"] for p in per.values() if p["feasible"]),
        "sell_h": sum(p["sell_h"] for p in per.values() if p["feasible"]),
        "n_infeasible": len(sols) - len(feas),
    }
    return {"feasible": True, "agg": agg, "per": per, "sols": sols,
            "n_infeasible": agg["n_infeasible"]}


def main():
    d, ext, nonai_full, absorb_full, latency = load()
    tasks, power, storage = d["tasks"], d["power"], d["storage"]
    print("=" * 72)
    print("S4 灵敏度扫描：ε / 峰谷价差 / 新能源波动")
    print("=" * 72)

    t_start = time.perf_counter()

    # ---- 基准：free 解 E0_S4 ----
    free_res = run_all(tasks, power, storage, ext, nonai_full, absorb_full,
                       latency, {}, eps=1.0, free=True)
    e0 = {r: free_res["sols"][r]["carbon_kt"] for r in REGIONS}
    print(f"E0_S4 (kt): { {r: round(v, 2) for r, v in e0.items()} }")
    print(f"  free 聚合: 成本 {free_res['agg']['cost_main_M']:.2f}M | "
          f"碳 {free_res['agg']['carbon_kt']:.2f}kt")

    # ---- ε 扫描（基准 e0）----
    eps_res = {}
    print("\n--- ε 扫描 ---")
    for ep in EPS_GRID:
        res = run_all(tasks, power, storage, ext, nonai_full, absorb_full,
                      latency, e0, eps=ep)
        eps_res[ep] = res
        if res["feasible"]:
            print(f"  ε={ep}: 成本 {res['agg']['cost_main_M']:8.2f}M | "
                  f"碳 {res['agg']['carbon_kt']:8.2f}kt | "
                  f"不可行 {res['n_infeasible']}/6")
        else:
            print(f"  ε={ep}: 全部不可行")

    # ---- price 扫描（基准 e0，碳不绑定直接可行）----
    price_res = {}
    print("\n--- 峰谷差扫描 ---")
    for ps in PRICE_GRID:
        res = run_all(tasks, power, storage, ext, nonai_full, absorb_full,
                      latency, e0, eps=1.0, price_scale=ps)
        price_res[ps] = res
        if res["feasible"]:
            print(f"  峰谷×{ps:.2f}: 成本 {res['agg']['cost_main_M']:8.2f}M | "
                  f"碳 {res['agg']['carbon_kt']:8.2f}kt | "
                  f"储能 {res['agg']['es_MWh']:8,.0f}MWh")
        else:
            print(f"  峰谷×{ps:.2f}: 不可行")

    # ---- renew 扫描（每点自身 free 作 ε 基准）----
    renew_res = {}
    print("\n--- 新能源波动扫描 ---")
    for rs in RENEW_GRID:
        fa = run_all(tasks, power, storage, ext, nonai_full, absorb_full,
                     latency, {}, eps=1.0, renew_scale=rs, free=True)
        e0r = {r: fa["sols"][r]["carbon_kt"] for r in REGIONS}
        res = run_all(tasks, power, storage, ext, nonai_full, absorb_full,
                      latency, e0r, eps=1.0, renew_scale=rs)
        renew_res[rs] = res
        if res["feasible"]:
            print(f"  renew×{rs:.2f}: 成本 {res['agg']['cost_main_M']:8.2f}M | "
                  f"碳 {res['agg']['carbon_kt']:8.2f}kt | "
                  f"储能 {res['agg']['es_MWh']:8,.0f}MWh")
        else:
            print(f"  renew×{rs:.2f}: 不可行")

    # ---- ε_min 精确下探（二分，精度 ~0.0002；每个区域独立可行域）----
    print("\n--- ε_min 精确下探（二分）---")
    eps_min = {}
    for r in REGIONS:
        lo, hi = 0.85, 1.0
        # 基准自检：1.0 必须可行
        if not solve_region(r, tasks, power, storage, e0, nonai_full,
                            absorb_full, ext, eps=1.0)["feasible"]:
            eps_min[r] = None
            print(f"  {r}: ε=1.0 即不可行（异常）")
            continue
        for _ in range(12):   # 精度 (0.15)/2^12 ≈ 3.7e-5
            mid = (lo + hi) / 2
            sol = solve_region(r, tasks, power, storage, e0, nonai_full,
                               absorb_full, ext, eps=mid)
            if sol["feasible"]:
                hi = mid
            else:
                lo = mid
        eps_min[r] = hi
        print(f"  {r}: ε_min ≈ {hi:.4f}")
    out = {
        "eps": eps_res, "price": price_res, "renew": renew_res,
        "eps_min": eps_min,
        "e0_s4_kt": e0,
        "grids": {"eps": EPS_GRID, "price": PRICE_GRID, "renew": RENEW_GRID},
        "meta": {"generated": "2026-08-09",
                 "source": "sub4-preprocessed.pkl + 层2 协同 MILP（复用 solve_region）"
                           "+ ε 基准 = 各场景自身 free 解（实证修正）"},
    }
    with open(DATA / "s4-sensitivity.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n已写入 {DATA / 's4-sensitivity.pkl'}  "
          f"（总耗时 {time.perf_counter() - t_start:.1f}s）")
    print("S4 SENSITIVITY DONE")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    只读核验《S4 降低基荷实验计划》引用的关键数字：基荷预填比例、
    候选窗口规模、层2 MILP 求解时间/n_x、B1→B2 单位节省，评估计划可行性。

原理：
    直接从现有 pickle 缓存读取统计数据，不重算调度。

输入数据：
    - outputs/data/s2-preprocessed.pkl — tasks/gh/latency
    - outputs/data/sub4-preprocessed.pkl — 基荷/任务结构
    - outputs/data/s4-results.pkl — 层2 求解时间与 n_x
    - outputs/data/s4-baseline-heuristic.pkl — B1/B2 对照

输出：
    - 控制台统计量（PR-014）
"""
import pickle
from pathlib import Path

import numpy as np

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]


def main():
    with open(DATA / "sub4-preprocessed.pkl", "rb") as f:
        d = pickle.load(f)
    tasks = d["tasks"]
    power = d["power"]
    T_END = d["T_END"]

    gh_tot = sum(t["gh"] for t in tasks)
    gh_bl = sum(t["gh"] for t in tasks if t.get("baseload", False))
    n_bl = sum(1 for t in tasks if t.get("baseload", False))
    print(f"[sub4-preprocessed 口径] 任务 {len(tasks)} | 基荷 {n_bl}"
          f" ({n_bl/len(tasks):.1%}) | GH 基荷 {gh_bl:,.0f}/{gh_tot:,.0f}"
          f" ({gh_bl/gh_tot:.1%})")
    print("  基荷 meta 每区:")
    for r in REGIONS:
        m = d["baseload_meta"][r]
        print(f"    {r}: 填充 {m['n_filled']:5d}/{m['n_candidates']:5d}"
              f" | GH {m['gh_filled']:12,.0f} | 理论配额 {m['theoretical_quota_gh']:12,.0f}"
              f" | fill {m['fill_rate_vs_quota']:.1%}")
    print(f"  改派 {d['n_reassigned']} | 仍失败 {d['n_still_failed']}")

    # 候选窗口规模（非基荷任务）
    nbl_by_r = {}
    n_x_by_r = {}
    med_win = {}
    for r in REGIONS:
        nbl = [t for t in tasks
               if t["dest"] == r and t["type"] != "RealTimeInference"
               and not t.get("baseload", False)]
        nbl_by_r[r] = len(nbl)
        ws = []
        for t in nbl:
            lo = int(t["arrive"])
            hi = int(min(t["latest"], T_END) - t["dur"] + 1e-9)
            ws.append(max(hi - lo + 1, 1))
        n_x_by_r[r] = sum(ws)
        med_win[r] = float(np.median(ws)) if ws else 0.0
    print("\n[层2 非基荷任务候选窗口]")
    for r in REGIONS:
        print(f"  {r}: 非基荷 {nbl_by_r[r]:5d} 任务 | 候选变量 n_x = {n_x_by_r[r]:9,d}"
              f" | 窗口 median {med_win[r]:.0f}h")
    print(f"  合计非基荷 {sum(nbl_by_r.values()):,d} | 合计 n_x {sum(n_x_by_r.values()):,d}")

    # 层2 MILP 求解时间
    with open(DATA / "s4-results.pkl", "rb") as f:
        s4 = pickle.load(f)
    main_agg = s4["main"]
    print(f"\n[现状 s4-results main] 成本 {main_agg['cost_main_M']:.2f} M | "
          f"总求解时间 {main_agg['total_time_s']:.1f} s | 碳 {main_agg['carbon_kt']:.1f} kt")
    for r in REGIONS:
        sol = main_agg["sols"][r]
        print(f"  {r}: n_x={sol['n_x']:9,d} n_nbl={sol['n_nbl']:5d} "
              f"time={sol['time_s']:8.1f}s status={sol['status']}")

    # B1/B2 对照
    with open(DATA / "s4-baseline-heuristic.pkl", "rb") as f:
        bh = pickle.load(f)
    b1 = bh["b1"]
    b2 = bh["b2"]
    print(f"\n[基线] B1-EDF 成本 {b1['cost_main_M']:.2f}M | B2-电价 成本 {b2['cost_main_M']:.2f}M")
    n_t = len(tasks)
    print(f"  B1→B2 单位节省: {(b1['cost_main_M']-b2['cost_main_M'])*1e6/n_t:.0f} 元/任务"
          f"（按全 5 万任务）")

    # 若降低基荷至 70% 的释放估算
    gh_by_dest = {}
    for t in tasks:
        if t["type"] != "RealTimeInference" and not t.get("baseload", False):
            continue
        gh_by_dest[t["dest"]] = gh_by_dest.get(t["dest"], 0.0) + t["gh"]
    cap_gh = {r: power[r]["cap"] * 2400 for r in REGIONS}
    print("\n[降低基荷释放量估算（以 70% 区域 GPU 容量为界）]")
    for r in REGIONS:
        bl_r = sum(t["gh"] for t in tasks
                   if t["dest"] == r and t["type"] != "RealTimeInference"
                   and t.get("baseload", False))
        cap_r = cap_gh[r]
        rel = max(bl_r - 0.7 * cap_r, 0.0)
        print(f"  {r}: 基荷GH {bl_r:12,.0f} / 容量 {cap_r:12,.0f}"
              f" ({bl_r/cap_r*100:5.1f}%) | 70% 界释放 {rel:12,.0f} GH"
              f" ({rel/max(bl_r,1)*100:5.1f}% 该区基荷)")


if __name__ == "__main__":
    main()

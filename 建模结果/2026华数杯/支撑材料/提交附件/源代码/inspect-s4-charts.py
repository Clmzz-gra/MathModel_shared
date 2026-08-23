# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 2.1.5 审查辅助 — 打印 s4-results.pkl 数值，评估各图表信息量

原理：
    只读 s4-results.pkl，打印主解/场景/逐区域指标，供判断哪几张图
    "只是柱状表格、缺机理"，进而提出改进建议。

输入数据：
    - outputs/data/s4-results.pkl

输出：
    - 控制台统计表
"""
import pickle
import numpy as np

BASE = r"e:\MathModel_pj-2026-C"
d = pickle.load(open(BASE + r"\outputs\data\s4-results.pkl", "rb"))

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
print("keys:", list(d.keys()))
m = d["main"]

print("\n=== S0-main 聚合 ===")
for k in ["cost_main_M", "carbon_kt", "peak_MW", "std_avg_MW",
          "util_no_sell_pct", "util_sell_pct", "delay_ms", "qos_pct",
          "n_infeasible"]:
    print(f"  {k} = {m[k]}")

print("\n=== 逐区域 (main) ===")
print(f"{'区域':<10}{'成本M':>8}{'碳kt':>8}{'峰值MW':>8}{'std':>6}"
      f"{'util_ns%':>9}{'util_s%':>9}{'超容h':>6}{'同刻充放h':>8}")
for r in REGIONS:
    s = m["sols"][r]
    print(f"{r:<10}{s['cost_main_M']:>8.2f}{s['carbon_kt']:>8.2f}"
          f"{s['peak_MW']:>8.1f}{s['std_MW']:>6.1f}"
          f"{s['util_no_sell_pct']:>9.1f}{s['util_sell_pct']:>9.1f}"
          f"{s['over_hours']:>6}{s['dual_hours']:>8}")

print("\n=== free (E0_S4 基准) ===")
print("  E0_S4:", {r: round(v, 2) for r, v in d["e0_s4_kt"].items()})
print("  S3基准:", {r: round(v, 2) for r, v in d["s3_carbon_ref_kt"].items()})

print("\n=== 场景 ===")
for name, agg in d["scenarios"].items():
    if not agg.get("feasible"):
        print(f"  {name}: 不可行 n_inf={agg.get('n_infeasible','?')}")
    else:
        feas = [r for r, s in agg["sols"].items() if s["feasible"]]
        infeas = [r for r, s in agg["sols"].items() if not s["feasible"]]
        print(f"  {name}: cost={agg['cost_main_M']:.2f}M carbon={agg['carbon_kt']:.2f}kt "
              f"peak={agg['peak_MW']:.1f} util_ns={agg['util_no_sell_pct']:.1f} "
              f"delay={agg['delay_ms']:.1f} qos={agg['qos_pct']:.1f} n_inf={agg['n_infeasible']}")
        print(f"      可行区:{feas} 不可行区:{infeas}")

# 场景间差异是否显著（对 fig1 信息量的判断）
print("\n=== 场景间指标变化（相对 S0-main） ===")
base_cost, base_carb = m["cost_main_M"], m["carbon_kt"]
for name, agg in d["scenarios"].items():
    if agg.get("feasible"):
        print(f"  {name}: Δcost={agg['cost_main_M']-base_cost:+8.2f}M "
              f"Δcarbon={agg['carbon_kt']-base_carb:+8.2f}kt")
    else:
        print(f"  {name}: 不可行")

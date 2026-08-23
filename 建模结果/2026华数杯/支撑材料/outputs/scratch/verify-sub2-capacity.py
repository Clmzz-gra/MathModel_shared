# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 1.1 A 类共享事实补充验证 — S2 复现 F3（容量感知分配收益）与
    F4（区域分解可行性），为方案确认书 A3（容量感知 90% 阈值 + 区域分解）
    提供可追溯数据证据（对应 verify-sub2-20260808.md F3/F4）

原理：
    1. F3 容量感知贪心：每任务候选目的地按成本升序，选"成本低且该区
       GPU-hour ≤ 90%×容量"者；候选全满则退路（选候选内负载率最低区域）
    2. F4 区域分解可行性：分配后每区域 GPU-hour 负载率 = 需求/容量，
       ≤90% 视为有充分裕量做时间维 MILP
    3. 成本/碳 = GPU_hours × power × PUE ×（电价/碳强度）全时域均值，
       与 verify-sub2.py 同口径
    4. 运行末尾与 verify-sub2-20260808.md 基准比对（16.6% / 30.4% / 117 退路）

输入数据：
    - outputs/data/s2-preprocessed.pkl（阶段 1.4 预处理产物）
    - 键 → 变量名映射：tasks → tasks（id/type/source/cand/gh），
      power → power（price/carbon/pue/cap 逐小时 + 容量），
      regions → regions, power_mapping → pm（每类型功率 MW/等效GPU）, T_END → T_END

输出：
    - 控制台：F3 成本/碳降幅 + 退路数、F4 各区域负载率、与报告基准一致性判定

对应论文章节：
    问题二（S2）碳感知任务调度 — 阶段 1.1 A 类共享事实（F3/F4）
"""
import pickle
from pathlib import Path

import numpy as np

BASE = Path(r"e:\MathModel_pj-2026-C")
with open(BASE / "outputs" / "data" / "s2-preprocessed.pkl", "rb") as f:
    prep = pickle.load(f)

tasks = prep["tasks"]
power = prep["power"]
regions = prep["regions"]
pm = prep["power_mapping"]
T_END = prep["T_END"]

# ============================================================
# 参数
# ============================================================
CAP_GH = {r: power[r]["cap"] * 2400 for r in regions}   # GPU-hour 容量
THRESHOLD = 0.90                                          # 90% 利用率阈值（F3 口径）

# 电价/碳均值（全时域，与 verify-sub2.py 一致）
price = {r: power[r]["price"].mean() for r in regions}
carbon = {r: power[r]["carbon"].mean() for r in regions}
pue = {r: power[r]["pue"] for r in regions}


def cost_of(task_type, region, gh):
    return gh * pm[task_type] * pue[region] * price[region]


def co2_of(task_type, region, gh):
    return gh * pm[task_type] * pue[region] * carbon[region]


# ============================================================
# F3：容量感知分配（90% 阈值贪心 + 退路）
# ============================================================
def capacity_aware_assign(threshold=THRESHOLD):
    """容量感知贪心：候选按成本排序，选'成本低且该区 GPU-hour ≤ threshold×容量'者；
    无满足则退路（选负载率最低区域）。"""
    demand = {r: 0.0 for r in regions}
    assign = {r: 0 for r in regions}
    fail = 0
    cost_total, co2_total = 0.0, 0.0
    for t in tasks:
        cand = sorted(t["cand"], key=lambda r: cost_of(t["type"], r, t["gh"]))
        placed = False
        for r in cand:
            if demand[r] + t["gh"] <= CAP_GH[r] * threshold:
                demand[r] += t["gh"]
                assign[r] += 1
                best = r
                placed = True
                break
        if not placed:
            best = min(cand, key=lambda r: demand[r] / CAP_GH[r])
            demand[best] += t["gh"]
            assign[best] += 1
            fail += 1
        cost_total += cost_of(t["type"], best, t["gh"])
        co2_total += co2_of(t["type"], best, t["gh"])
    return demand, assign, fail, cost_total, co2_total


def baseline_cost():
    """S1 基线（零迁移）：任务本地运行成本/碳"""
    c = sum(cost_of(t["type"], t["source"], t["gh"]) for t in tasks)
    k = sum(co2_of(t["type"], t["source"], t["gh"]) for t in tasks)
    return c, k


# ============================================================
# 运行
# ============================================================
c0, k0 = baseline_cost()
demand, assign, fail, c_new, k_new = capacity_aware_assign()

print("=== F3：容量感知分配（90% 阈值贪心）===")
print(f"成本: 基线 {c0/1e6:.1f}M 元 → 容量感知 {c_new/1e6:.1f}M 元 → 降 {(c0-c_new)/c0:.1%}")
print(f"碳排: 基线 {k0/1e3:.1f}kt → 容量感知 {k_new/1e3:.1f}kt → 降 {(k0-k_new)/k0:.1%}")
print(f"触发退路任务: {fail} ({fail/len(tasks):.2%})")
print()

print("=== F4：区域分解可行性（分配后各区域负载）===")
for r in regions:
    util = demand[r] / CAP_GH[r]
    print(f"{r}: 任务 {assign[r]:>6}  GPU-hour {demand[r]:>12,.0f} / {CAP_GH[r]:>12,.0f} = {util:.1%}")
over = [r for r in regions if demand[r] / CAP_GH[r] > 1.0]
print(f"\n超容量区域: {over if over else '无'}")
print(f"验证: 成本降 16.6%、碳降 30.4%、退路 117 → 与 verify-sub2-20260808.md F3 一致 = "
      f"{abs((c0-c_new)/c0-0.166)<0.005 and abs((k0-k_new)/k0-0.304)<0.005 and fail==117}")

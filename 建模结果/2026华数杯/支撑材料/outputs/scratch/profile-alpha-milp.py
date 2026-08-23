# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    Alpha MILP 性能剖析（阶段 1.2 head-to-head 之后）——量化 472s 中
    约束构建 vs 求解的时间占比，测试 HiGHS 收敛速度，为阶段 2 缓存复用提供依据

原理：
    1. 复用缓存 s1_test_tasks.pkl（free/base/cap/hours/hidx），避免重算数据
    2. 分段计时：候选窗构建、开工约束(eq)、容量+U 约束(A1/A2/A3)矩阵构建
    3. 用极小 time_limit 探测 HiGHS 早期可行解质量（仅探测，不缓存、不影响正式结果）
    4. 分析阶段 2 复用点：预处理的 schedule_input 已含候选窗，可跳过重复构建

输入数据：
    - outputs/data/cache/s1_test_tasks.pkl（notebook cell 4 缓存）
    - outputs/data/c-data-cleaned.pkl（cap 引用）

输出：
    - 控制台时间构成报告（供方案确认/阶段 2 决策引用）

对应论文章节：
    问题一（S1）基础算力调度 — 求解性能与缓存复用分析
"""
import pickle
import time
import numpy as np
from pathlib import Path

BASE = Path(r"e:\MathModel_pj-2026-C")
with open(BASE / "outputs" / "data" / "cache" / "s1_test_tasks.pkl", "rb") as f:
    r4 = pickle.load(f)
free = r4["free"]
base = r4["base"]
regions = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
# cap 从清洗数据读取（与 notebook/预处理同源）
with open(BASE / "outputs" / "data" / "c-data-cleaned.pkl", "rb") as f:
    gi = pickle.load(f)["GPU_information"].set_index("Region")
cap = {r: gi.loc[r, "Available_GPU"] for r in regions}
print("cap:", cap)

T0, T_END = 2376, 2406
hours = list(range(T0, T_END))
Hn = len(hours)

# ---------- 1. 候选窗构建计时 ----------
t0 = time.time()
cand = []
xoff = []
col = 0
for t in free:
    lo = max(t["arrive"], T0)
    w = [h for h in hours if lo <= h < min(t["latest"], T_END) - t["dur"] + 1e-9
         and h + t["dur"] <= min(t["latest"], T_END) + 1e-9]
    if not w:
        w = [lo]
    cand.append(w)
    xoff.append(col)
    col += len(w)
t_cand = time.time() - t0
nvar = col + 2
print(f"[1] 候选窗构建: {t_cand:.3f}s | 变量总数 {col}（0-1）+2 连续")

# ---------- 2. 开工约束 (eq) 构建计时 ----------
t0 = time.time()
eq_rows = []
for i, off in enumerate(xoff):
    row = np.zeros(nvar)
    for k in range(len(cand[i])):
        row[off + k] = 1.0
    eq_rows.append(row)
Aeq = np.array(eq_rows)
t_eq = time.time() - t0
print(f"[2] 开工约束 (Σx=1, {len(eq_rows)} 条): {t_eq:.3f}s")

# ---------- 3. 容量 + U 约束矩阵构建计时 ----------
t0 = time.time()
A1, ub1 = [], []
A2, ub2 = [], []
A3, lb3 = [], []
for ri, r in enumerate(regions):
    cr = cap[r]
    for hh in range(Hn):
        row = np.zeros(nvar)
        for i, t in enumerate(free):
            if t["region"] != r:
                continue
            off = xoff[i]
            for k, h in enumerate(cand[i]):
                ov = min(h + t["dur"], hours[hh] + 1.0) - max(float(h), float(hours[hh]))
                if ov > 0:
                    row[off + k] += t["dem"] * ov
        A1.append(row.copy())
        ub1.append(cr - base[ri, hh])
        r2 = row.copy(); r2[col] = -cr; A2.append(r2); ub2.append(-base[ri, hh])
        r3 = row.copy(); r3[col + 1] = -cr; A3.append(r3); lb3.append(-base[ri, hh])
t_cons = time.time() - t0
n_cons = len(A1)
print(f"[3] 容量+U 约束矩阵 ({n_cons} 条/区域×小时): {t_cons:.3f}s")
print(f"    其中 A1 非零元估算: {sum(int(np.count_nonzero(r)) for r in A1)}")

# ---------- 4. HiGHS 求解时间探测（不缓存，仅测量不同 time_limit 的解质量） ----------
from scipy.optimize import milp, LinearConstraint, Bounds
c = np.zeros(nvar)
c[col] = 1.0
c[col + 1] = -1.0
constraints = [
    LinearConstraint(Aeq, np.ones(len(eq_rows)), np.ones(len(eq_rows))),
    LinearConstraint(np.array(A1), -np.inf, np.array(ub1)),
    LinearConstraint(np.array(A2), -np.inf, np.array(ub2)),
    LinearConstraint(np.array(A3), np.array(lb3), np.inf),
]
integrality = np.ones(nvar)
integrality[col:] = 0
bounds = Bounds(np.zeros(nvar), np.ones(nvar))
bounds.ub[col] = bounds.ub[col + 1] = np.inf

print("\n[4] HiGHS 求解时间-质量探测（正式结果以缓存 s1_alpha_milp.pkl 为准，此处不覆盖）:")
for tl in [5, 30, 60]:
    t0 = time.time()
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
               options={"time_limit": tl, "mip_rel_gap": 0.01})
    dt = time.time() - t0
    print(f"  time_limit={tl:>3}s → 实际耗时 {dt:6.1f}s | status={res.status} | 目标={res.fun if res.x is not None else '无解'}")

# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
review-q3-s1costbase.py — 审查补实验 Q3：S1 成本最小化零迁移基线

1. 目的
   S2 报告以 "S1 时移基线"（333.3 M 元 / 374.28 kt）对比 S2 成本（340.1 M 元，+2.0%），
   但该基线目标函数未声明（S1 主模型目标为利用率极差最小化，非成本最小）。
   本实验复用 S1 时间索引 MILP 框架，将目标改为成本最小化，构造
   **同目标（成本最小化）零迁移基线**，重估 +2.0% 对比成立性（review-notes Q3/P5）。

2. 原理
   - 复用 S1 框架（sub1-model.py solve_alpha 同构）：任务恰好开工一次 /
     GPU 容量逐时（重叠精确折算）/ 完成时限窗口（cand 生成逻辑一致）；
     仅目标改为 min Σ 成本（去掉 Umax/Umin 辅助变量）
   - 成本 = Σ_{任务 j, 占用小时 h} dem_j·GPU_Power(type_j)·PUE(region)·Price(region,h)·重叠
   - 碳排 = 同构 × CarbonIntensity(region,h)
   - 零迁移：任务留在 source region（S1 口径即零迁移，只做时移）
   - 实时任务固定于到达时点（rt_fixed）；free 任务（194 训练/184 批量）由 MILP 定时移
   - 求解：scipy.milp（HiGHS），测试窗 2376–2399 到达 538 任务

3. 输入映射
   - outputs/data/s1-preprocessed.pkl（schedule_input: tasks/rt_fixed/free/base/hours/hidx/regions/cap/pue + power_mapping）
   - outputs/data/cache/s1_alpha_milp.pkl（极差目标 α 调度，同窗对照）
   - outputs/data/s3-preprocessed.pkl（panel: 逐时 Price/CarbonIntensity，取 2376–2405）

4. 输出
   - 控制台：成本目标基线（成本/碳排/超容小时）+ vs 极差目标 α 基线 + 口径说明

5. 论文章节
   - 问题二（S2）碳感知任务调度：基线目标函数定案（阶段 3.1 审查修正）
"""
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(r"E:\MathModel_pj-2026-C-sub3")
DATA = BASE / "outputs" / "data"
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
T0, T_END = 2376, 2406


def main():
    with open(DATA / "s1-preprocessed.pkl", "rb") as f:
        prep = pickle.load(f)
    si = prep["schedule_input"]
    pm = prep["power_mapping"]
    tasks, rt_fixed, free = si["tasks"], si["rt_fixed"], si["free"]
    base, hours, hidx = si["base"], si["hours"], si["hidx"]
    cap, pue = si["cap"], si["pue"]
    Hn = len(hours)
    T0loc = hours[0]

    with open(DATA / "cache" / "s1_alpha_milp.pkl", "rb") as f:
        alpha = pickle.load(f)["schedule"]

    # 逐时价格/碳强度（2376–2405，30h），来自 s3 面板
    with open(DATA / "s3-preprocessed.pkl", "rb") as f:
        panel = pickle.load(f)["panel"]
    PRICE = {r: panel.xs(r)["Price_CNY_per_MWh"].values for r in REGIONS}
    CARB = {r: panel.xs(r)["CarbonIntensity_tCO2_per_MWh"].values for r in REGIONS}

    print("=" * 78)
    print("Q3 S1 成本最小化零迁移基线（复用 S1 时间索引 MILP 框架改目标）")
    print(f"测试窗 {T0}–2399 到达 {len(tasks)} 任务: "
          f"{dict(Counter(t['type'] for t in free))} + 实时 {len(rt_fixed)}")
    print("=" * 78)

    def task_cost(t, r, h):
        c = 0.0
        e_end = min(h + t["dur"], T_END)
        s = h; hi = int(np.floor(s))
        while s < e_end and hi < T_END:
            ov = min(e_end, hi + 1.0) - max(s, float(hi))
            if ov > 0:
                c += t["dem"] * pm[t["type"]] * pue[r] * PRICE[r][hi - T0loc] * ov
            s = hi + 1.0; hi = int(np.floor(s))
        return c

    def task_co2(t, r, h):
        e = 0.0
        e_end = min(h + t["dur"], T_END)
        s = h; hi = int(np.floor(s))
        while s < e_end and hi < T_END:
            ov = min(e_end, hi + 1.0) - max(s, float(hi))
            if ov > 0:
                e += t["dem"] * pm[t["type"]] * pue[r] * CARB[r][hi - T0loc] * ov
            s = hi + 1.0; hi = int(np.floor(s))
        return e

    # ---- 候选窗（与 solve_alpha 同构）----
    cand, xoff, col = [], [], 0
    for t in free:
        lo = max(t["arrive"], T0)
        w = [h for h in hours if lo <= h < min(t["latest"], T_END) - t["dur"] + 1e-9
             and h + t["dur"] <= min(t["latest"], T_END) + 1e-9]
        if not w:
            w = [lo]
        cand.append(w)
        xoff.append(col)
        col += len(w)
    nvar = col
    print(f"free 任务 {len(free)}，候选变量 {nvar}")

    # ---- 目标：成本最小 ----
    c = np.zeros(nvar)
    for i, t in enumerate(free):
        off = xoff[i]
        for k, h in enumerate(cand[i]):
            c[off + k] = task_cost(t, t["region"], h)

    # ---- 约束：恰好一次 + 容量逐时 ----
    eq_rows = []
    for i, off in enumerate(xoff):
        row = np.zeros(nvar)
        for k in range(len(cand[i])):
            row[off + k] = 1.0
        eq_rows.append(row)
    A1, ub1 = [], []
    for ri, r in enumerate(REGIONS):
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

    from scipy.optimize import milp, LinearConstraint, Bounds
    constraints = [
        LinearConstraint(np.array(eq_rows), np.ones(len(eq_rows)), np.ones(len(eq_rows))),
        LinearConstraint(np.array(A1), -np.inf, np.array(ub1)),
    ]
    integrality = np.ones(nvar)
    bounds = Bounds(np.zeros(nvar), np.ones(nvar))
    t0 = time.time()
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
               options={"time_limit": 1800, "mip_rel_gap": 0.01})
    dt = time.time() - t0
    print(f"MILP: status={res.status} fun={res.fun/1e6:.4f}M（free 部分）耗时 {dt:.1f}s")
    if res.x is None:
        raise RuntimeError(f"Q3 成本目标 MILP 无解 status={res.status}: {res.message}")
    approx = res.status != 0   # time_limit/迭代截断的可行次优
    if approx:
        print(f"    [警告] 成本目标 MILP 未达最优: status={res.status}（time_limit/迭代截断），"
              f"解为可行次优，Q3 结论按次优近似口径")

    # ---- 解出调度 ----
    sched_cost = {}
    for i, t in enumerate(free):
        off = xoff[i]
        sel = None
        for k, h in enumerate(cand[i]):
            if res.x[off + k] > 0.5:
                sel = h
                break
        sched_cost[t["id"]] = sel if sel is not None else cand[i][0]

    # ---- 评估（全部 538 任务：free 用解出时点，rt 用到达时点）----
    def evaluate(start_of, tag):
        cost = co2 = 0.0
        use = base.copy()
        for t in tasks:
            h = start_of[t["id"]]; r = t["region"]
            cost += task_cost(t, r, h)
            co2 += task_co2(t, r, h)
            if t["type"] == "RealTimeInference":
                continue  # 实时占用已计入 base，避免重复回写
            # 回写 free 任务占用（超容检查）
            e_end = min(h + t["dur"], T_END)
            s = h; hi = int(np.floor(s))
            while s < e_end and hi < T_END:
                ov = min(e_end, hi + 1.0) - max(s, float(hi))
                if ov > 0 and hi - T0loc < Hn:
                    use[REGIONS.index(r), hi - T0loc] += t["dem"] * ov
                s = hi + 1.0; hi = int(np.floor(s))
        caps = np.array([cap[r] for r in REGIONS])[:, None]
        over = int((use > caps + 1e-6).sum())
        print(f"  [{tag}] 成本 {cost/1e6:9.2f} M元 | 碳 {co2/1e3:8.2f} kt | 超容小时 {over}")
        return cost / 1e6, co2 / 1e3, over

    start_rt = {t["id"]: t["arrive"] for t in rt_fixed}
    c_cost, e_cost, o_cost = evaluate({**start_rt, **sched_cost}, "成本目标基线（Q3）")
    start_alpha = {**start_rt, **alpha}
    c_alpha, e_alpha, o_alpha = evaluate(start_alpha, "极差目标 α 基线（同窗对照）")

    print("\n" + "=" * 78)
    print("Q3 结论（测试窗 0–2405 结算口径）")
    print("=" * 78)
    print(f"  成本目标基线   {c_cost:8.2f} M元 / {e_cost:8.2f} kt"
          + ("（次优近似，见上警告）" if approx else ""))
    print(f"  极差目标 α 基线 {c_alpha:8.2f} M元 / {e_alpha:8.2f} kt")
    print(f"  成本目标相对 α: {(c_cost - c_alpha) / c_alpha * 100:+.2f}%（同窗同口径）")
    print("\n  口径说明：本实验为 S1 测试窗（538 任务）窗口尺度；S2 报告的 333.3/340.1 M元"
          " 为全量 50000 任务 2400h 尺度。")
    print("  全量基线目标函数定案（代码核验）：S2 run_baseline → schedule_dest(dest=source, "
          "obj='cost')，")
    print("  即全量 333.3 M元 基线**已是成本最小化**（零迁移可时移）→ S2 +2.0% 为同目标对比，"
          "成立。")
    print("  （成本目标 ≤ 极差目标成本为 LP 保证；两基线差值 = 目标差异导致的时移选择成本）")
    print("Q3 S1 COST-MIN BASELINE DONE (review-q3-s1costbase.py)")


if __name__ == "__main__":
    main()

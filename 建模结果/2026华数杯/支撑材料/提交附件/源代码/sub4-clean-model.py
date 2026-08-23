# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    Clean-test（方案 B）配对选时实验 — 对 preprocess-sub4-clean 产出的
    每档实例做 S_EDF（样本 EDF 固定）与 S_MILP（样本 MILP 全局选时）
    配对对比，按 GH 比例放大到全部释放任务，量化"EDF 预填锁死 vs
    MILP 全局最优"的成本收益边界（回答 R1）。

原理：
    1. 配对隔离（核心）：同一 θ 档实例上，样本任务 M 的选时方式
       是唯一差异——
       - S_EDF：M 按 GH 降序 EDF 真实填入（占用累积，失败任务强制取
         arrive 超容，由 MILP slack 兜底），其余固定占用（实时 + θ 预填
         + 原残差任务变量）不变；
       - S_MILP：M 为 0-1 变量（窗口 = EDF 预填窗口，主时域 h+dur≤2400，
         同域公平），与电力/储能连续变量联合优化。
       两解仅 M 的选时不同 → Δ_r = C_EDF − C_MILP 纯为选时增益。
    2. **主时域同域**：EDF 预填窗口限主时域，MILP 若允许结清段
       （2400-2405）则混入"结清段红利"，非 R1 范畴；故对 M 加
       main_horizon_ids 参数截断窗口。原残差任务（EDF 失败进 MILP）
       保持现状（含结清段），两解一致不贡献 Δ。
    3. 抽样放大：Δ_total = Σ_r Δ_r × (GH_F_r / GH_M_r)，GH 与电量/成本
       线性相关（电量 = GH × 单位GPU功率 × PUE），按 GH 覆盖比例放大
       到释放任务中 EDF 可行子集（F_r）。EDF 不可行释放任务（U_r）排除
       在配对外——它们即便交 MILP 也主要靠超容惩罚，不代表"选时优化"
       收益，单独报告数量。
    4. free 求解（无碳约束）：主解 ε=1.00 实测碳约束不绑定（main==free，
       见 decompose-sub4-cost），ε 不是本实验对象；与 decompose 口径一致。
    5. 成本指标：主时域购售电成本 cost_main_M（同 sub4-model）。

输入数据：
    - outputs/scratch/sub4-clean-ratioXXX.pkl（Clean-test 预处理，θ 档）
      tasks: id/type/dest/baseload/start_h/released/edf_ok/sample/arrive/
             dur/dem/latest/gh/power
      power / storage / quota_aiit_arr / sampling_meta / dryrun
    - outputs/data/s3-preprocessed.pkl — panel 结清段实际值（经 M4.load）
    - outputs/data/c-data-cleaned.pkl — NonAI + 受限消纳（经 M4.load）
    - outputs/data/s2-preprocessed.pkl — latency（经 M4.load）
    - 中文指标 → 变量名映射：
      购电→G, 卖电→S, 新能源直供→R, 电网充电→Cg, 新能源充电→Cr, 放电→D,
      成本(M元)→cost_main_M, 超容松弛→slack, 样本→sample, 释放→released,
      EDF可行→edf_ok, GPU-hour→gh, 候选变量数→n_x

输出：
    - outputs/scratch/s4-clean-results.pkl — 键：
      ratio / delta: {per_region: {Δ_M, gh_ratio}, total_M, pct_of_main}
      s_edf / s_milp: 每区域 {cost_main_M, n_x, time_s, slack_gh_total, status}
      sampling_meta / meta
    - 控制台配对表（每区域 Δ、放大、求解时间，PR-014）

对应论文章节：
    问题四（S4）算-储-电协同优化 — 基荷预填比例敏感性实验（Clean-test）
"""
import importlib.util
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"
SCRATCH = BASE / "outputs" / "scratch"

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
NT = 2406
MAIN = 2400
T_END = 2406
MAIN_COST_M = 2166.64   # 现状主解成本（判定基准，s4-results.pkl 实测）

# 求解参数（同 sub4-model）
MILP_TIME_LIMIT = 600.0
MILP_GAP = 0.01
GPU_SLACK_PENALTY = 1e6
TOL = 1e-6


def earliest_feasible_hour(t, r, occ_gpu, occ_aiit, cap_gpu, quota_arr):
    """同 preprocess-sub4：找最早可行开工小时（EDF，主时域窗口）。"""
    lo = int(t['arrive'])
    hi = int(min(t['latest'], T_END) - t['dur'])
    hi = min(hi, MAIN - int(np.ceil(t['dur'])))
    if hi < lo:
        return None
    for h in range(lo, hi + 1):
        h_end = int(np.ceil(h + t['dur']))
        ok = True
        for hh in range(h, h_end):
            overlap = min(h + t['dur'], hh + 1) - max(h, hh)
            if occ_gpu[hh] + t['dem'] * overlap > cap_gpu + 1e-6:
                ok = False
                break
            if occ_aiit[hh] + t['dem'] * t['power'] * overlap > quota_arr[hh] + 1e-6:
                ok = False
                break
        if ok:
            return h
    return None


def commit_placement(t, h, occ_gpu, occ_aiit):
    h_end = int(np.ceil(h + t['dur']))
    for hh in range(h, h_end):
        overlap = min(h + t['dur'], hh + 1) - max(h, hh)
        occ_gpu[hh] += t['dem'] * overlap
        occ_aiit[hh] += t['dem'] * t['power'] * overlap


def edf_place_samples(r, tasks, sample_ids, power, quota_arr):
    """S_EDF：样本任务按 GH 降序 EDF 真实填入（占用累积）。

    tasks = 区域任务列表（已剔除 released-non-sample）。
    base = 实时 + θ 预填固定占用（与 solve_region.fixed_occupancy 同口径）。
    EDF 失败任务强制取 arrive（GPU 超容 → MILP slack 兜底），记入 n_forced。
    返回 (sched{id:h}, n_forced)。
    """
    occ_gpu = np.zeros(NT, dtype=float)
    occ_aiit = np.zeros(NT, dtype=float)
    for t in tasks:
        if t["id"] in sample_ids:
            continue
        if t["type"] == "RealTimeInference":
            h0, dur = t["arrive"], t["dur"]
        elif t.get("baseload", False) and t["start_h"] is not None:
            h0, dur = t["start_h"], t["dur"]
        else:
            continue
        h_end = int(np.ceil(h0 + dur))
        for hh in range(int(np.floor(h0)), min(h_end, NT)):
            ov = min(h0 + dur, hh + 1.0) - max(h0, float(hh))
            occ_gpu[hh] += t["dem"] * ov
            occ_aiit[hh] += t["dem"] * t["power"] * ov
    samples = sorted([t for t in tasks if t["id"] in sample_ids],
                     key=lambda t: -t["gh"])
    sched = {}
    n_forced = 0
    for t in samples:
        h = earliest_feasible_hour(t, r, occ_gpu, occ_aiit,
                                   power[r]["cap"], quota_arr[r])
        if h is None:
            h = int(t["arrive"])
            n_forced += 1
        commit_placement(t, h, occ_gpu, occ_aiit)
        sched[t["id"]] = h
    return sched, n_forced


def solve_region_clean(r, tasks, power, storage, carbon_ref, nonai_full,
                       absorb_full, ext, eps=1.00, price_scale=1.0,
                       renew_scale=1.0, free=False, main_horizon_ids=None):
    """层2 每区域协同 MILP（复制 sub4-model.solve_region + 主时域窗口参数）。

    main_horizon_ids：这些非基荷任务的候选窗口截断到主时域
    （h+dur ≤ 2400，同 EDF 预填窗口），保证 S_EDF/S_MILP 同域公平。
    """
    s = storage[r]
    p = power[r]
    price = np.concatenate([p["price"], ext[r]["price"][MAIN:NT]])
    sellp = np.concatenate([p["sell"], ext[r]["sellp"][MAIN:NT]])
    ci = np.concatenate([p["carbon"], ext[r]["carbon"][MAIN:NT]])
    pue = p["pue"]
    cap_gpu = p["cap"]
    nonai = nonai_full[r]
    absorb = absorb_full[r] * renew_scale
    avail = np.concatenate([p["renewable"], ext[r]["renewable"][MAIN:NT]]) * renew_scale

    if price_scale > 1.0:
        peak_mask = price > price.mean()
        price = price.copy()
        price[peak_mask] *= price_scale
        sellp = sellp.copy()
        sellp[peak_mask] *= price_scale

    # ---- 固定占用（实时 + 基荷预填；任务列表已剔除 released-non-sample） ----
    fg = np.zeros(NT, dtype=float)
    fa = np.zeros(NT, dtype=float)
    for t in tasks:
        if t["type"] == "RealTimeInference":
            h0, dur = t["arrive"], t["dur"]
        elif t.get("baseload", False) and t["start_h"] is not None:
            h0, dur = t["start_h"], t["dur"]
        else:
            continue
        h_end = int(np.ceil(h0 + dur))
        for hh in range(int(np.floor(h0)), min(h_end, NT)):
            ov = min(h0 + dur, hh + 1.0) - max(h0, float(hh))
            fg[hh] += t["dem"] * ov
            fa[hh] += t["dem"] * t["power"] * ov

    # ---- 非基荷自由任务 + 候选窗口 ----
    nbl = [t for t in tasks
           if t["type"] != "RealTimeInference"
           and not t.get("baseload", False)]
    cands, xoff = [], []
    n_x = 0
    n_forced = 0
    for t in nbl:
        lo = int(t["arrive"])
        hi = int(min(t["latest"], T_END) - t["dur"] + 1e-9)
        if main_horizon_ids is not None and t["id"] in main_horizon_ids:
            hi = min(hi, MAIN - int(np.ceil(t["dur"])))   # 主时域同 EDF 预填
        if hi < lo:
            n_forced += 1
            hi = lo
        w = list(range(lo, hi + 1))
        cands.append(w)
        xoff.append(n_x)
        n_x += len(w)
    n_nbl = len(nbl)

    off_g = n_x + 0 * NT
    off_s = n_x + 1 * NT
    off_r = n_x + 2 * NT
    off_cg = n_x + 3 * NT
    off_cr = n_x + 4 * NT
    off_d = n_x + 5 * NT
    off_e = n_x + 6 * NT
    off_slack = n_x + 7 * NT
    n_vars = n_x + 7 * NT + NT

    c = np.zeros(n_vars)
    c[off_g:off_g + NT] = price
    c[off_s:off_s + NT] = -sellp
    c[off_slack:] = GPU_SLACK_PENALTY

    n_ub = 3 * NT + 2
    if free:
        n_ub -= 1
    A_ub = lil_matrix((n_ub, n_vars), dtype=float)
    b_ub = np.zeros(n_ub)
    row = 0
    for t in range(NT):
        for i, tsk in enumerate(nbl):
            off = xoff[i]
            for k, h in enumerate(cands[i]):
                ov = min(h + tsk["dur"], t + 1.0) - max(h, float(t))
                if ov > 0:
                    A_ub[row, off + k] += tsk["dem"] * ov
        A_ub[row, off_slack + t] = -1.0
        b_ub[row] = cap_gpu - fg[t]
        row += 1
    for t in range(NT):
        A_ub[row, off_r + t] = 1.0
        A_ub[row, off_cr + t] = 1.0
        b_ub[row] = absorb[t]
        row += 1
    for t in range(NT):
        A_ub[row, off_cg + t] = 1.0
        A_ub[row, off_cr + t] = 1.0
        b_ub[row] = s["MaxChargePower_MW"]
        row += 1
    if not free:
        for t in range(MAIN):
            A_ub[row, off_g + t] = ci[t]
        b_ub[row] = 1e3 * eps * carbon_ref[r]
        row += 1
    A_ub[row, off_e + NT - 1] = -1.0
    b_ub[row] = -s["InitialSOC_MWh"]
    row += 1
    assert row == n_ub

    n_eq = n_nbl + NT + NT
    A_eq = lil_matrix((n_eq, n_vars), dtype=float)
    b_eq = np.zeros(n_eq)
    row = 0
    for i, tsk in enumerate(nbl):
        off = xoff[i]
        for k in range(len(cands[i])):
            A_eq[row, off + k] = 1.0
        b_eq[row] = 1.0
        row += 1
    for t in range(NT):
        A_eq[row, off_g + t] = 1.0
        A_eq[row, off_r + t] = 1.0
        A_eq[row, off_d + t] = 1.0
        A_eq[row, off_cg + t] = -1.0
        A_eq[row, off_s + t] = -1.0
        for i, tsk in enumerate(nbl):
            off = xoff[i]
            for k, h in enumerate(cands[i]):
                ov = min(h + tsk["dur"], t + 1.0) - max(h, float(t))
                if ov > 0:
                    A_eq[row, off + k] -= pue * tsk["dem"] * tsk["power"] * ov
        b_eq[row] = pue * (fa[t] + nonai[t])
        row += 1
    eta_c, eta_d = s["ChargeEfficiency"], s["DischargeEfficiency"]
    init = s["InitialSOC_MWh"]
    for t in range(NT):
        A_eq[row, off_e + t] = 1.0
        if t >= 1:
            A_eq[row, off_e + t - 1] = -1.0
        A_eq[row, off_cg + t] = -eta_c
        A_eq[row, off_cr + t] = -eta_c
        A_eq[row, off_d + t] = 1.0 / eta_d
        b_eq[row] = init if t == 0 else 0.0
        row += 1
    assert row == n_eq

    bounds_lb = np.zeros(n_vars)
    bounds_ub = np.full(n_vars, np.inf)
    bounds_ub[:n_x] = 1.0
    bounds_lb[off_e:off_e + NT] = s["MinSOC_MWh"]
    bounds_ub[off_e:off_e + NT] = s["Capacity_MWh"]
    bounds_ub[off_g:off_g + NT] = s["MaxGridImport_MW"]
    bounds_ub[off_s:off_s + NT] = s["SellLimit_MW"]
    bounds_ub[off_d:off_d + NT] = s["MaxDischargePower_MW"]

    constraints = [
        LinearConstraint(A_ub.tocsr(), -np.inf, b_ub),
        LinearConstraint(A_eq.tocsr(), b_eq, b_eq),
    ]
    integrality = np.zeros(n_vars)
    integrality[:n_x] = 1

    t0 = time.perf_counter()
    res = milp(c=c, constraints=constraints, integrality=integrality,
               bounds=Bounds(bounds_lb, bounds_ub),
               options={"time_limit": MILP_TIME_LIMIT, "mip_rel_gap": MILP_GAP,
                        "disp": False})
    dt = time.perf_counter() - t0
    if res.x is None:
        # 审查 B1 修复：infeasible 返回补齐报告所需键（NaN 占位，不触发 KeyError）
        return {"region": r, "status": res.status, "feasible": False,
                "time_s": dt, "n_x": n_x, "n_nbl": n_nbl,
                "cost_main_M": float("nan"), "slack_gh_total": float("nan"),
                "mip_gap": float("nan")}

    x = res.x
    G = x[off_g:off_g + NT]
    S = x[off_s:off_s + NT]
    slack = x[off_slack:]
    cost_main_M = float(np.sum(G[:MAIN] * price[:MAIN]
                               - S[:MAIN] * sellp[:MAIN])) / 1e6
    # LNS 迭代需要：非基荷任务选时（与 sub4-model 原版同口径）
    sched_nbl = {}
    for i, tsk in enumerate(nbl):
        off = xoff[i]
        sel = None
        for k, h in enumerate(cands[i]):
            if x[off + k] > 0.5:
                sel = h
                break
        if sel is None:
            sel = cands[i][0]
        sched_nbl[tsk["id"]] = sel
    return {
        "region": r, "status": res.status, "time_s": dt, "feasible": True,
        "cost_main_M": cost_main_M, "n_x": n_x, "n_nbl": n_nbl,
        "slack_gh_total": float(np.sum(slack)), "n_forced_single": n_forced,
        "mip_gap": float(res.mip_gap) if res.mip_gap is not None
                   else float("nan"),
        "sched_nbl": sched_nbl,
    }


def main():
    import sys
    if len(sys.argv) > 1:
        ratio = float(sys.argv[1])
    else:
        ratio = None
    # 从最新匹配的预处理 pkl 推断 θ（若无参数）
    if ratio is None:
        import re
        cands = sorted(SCRATCH.glob("sub4-clean-ratio*.pkl"))
        assert cands, "未找到 sub4-clean-ratio*.pkl"
        m = re.search(r"ratio(\d{3})", cands[-1].name)
        ratio = int(m.group(1)) / 100.0
    clean_pkl = SCRATCH / f"sub4-clean-ratio{int(round(ratio * 100)):03d}.pkl"
    assert clean_pkl.exists(), f"缺少 {clean_pkl}"

    # 复用 sub4-model 的 load（结清段实际值 / NonAI / 受限消纳 / latency）
    spec = importlib.util.spec_from_file_location(
        "sub4model", SCRATCH / "sub4-model.py")
    M4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M4)
    M4.BASE = BASE          # sub4-model.py 的 BASE 指向旧 worktree，重定向
    M4.DATA = DATA
    _, ext, nonai_full, absorb_full, latency = M4.load()

    with open(clean_pkl, "rb") as f:
        clean = pickle.load(f)
    tasks = clean["tasks"]
    power = clean["power"]
    storage = clean["storage"]
    quota_arr = clean["quota_aiit_arr"]
    sampling_meta = clean["sampling_meta"]

    print("=" * 78)
    print(f"S4 Clean-test 配对实验: θ = {ratio}")
    print("=" * 78)

    delta_total_M = 0.0
    results = {"ratio": ratio, "per_region": {}, "s_edf": {}, "s_milp": {}}
    for r in REGIONS:
        tasks_r = [t for t in tasks if t["dest"] == r]
        samples = [t for t in tasks_r if t.get("sample")]
        ex_ids = {t["id"] for t in tasks_r
                  if t.get("released") and not t.get("sample")}
        tasks_base = [t for t in tasks_r if t["id"] not in ex_ids]

        if samples:
            sample_ids = {t["id"] for t in samples}
            # ---- S_EDF：样本 EDF 固定 ----
            sched, n_forced_edf = edf_place_samples(
                r, tasks_base, sample_ids, power, quota_arr)
            tasks_edf = [dict(t) for t in tasks_base]
            for t in tasks_edf:
                if t["id"] in sched:
                    t["baseload"] = True
                    t["start_h"] = sched[t["id"]]
            sol_edf = solve_region_clean(r, tasks_edf, power, storage, {},
                                         nonai_full, absorb_full, ext,
                                         free=True, main_horizon_ids=None)
            # ---- S_MILP：样本 MILP 变量（主时域窗口） ----
            tasks_milp = [dict(t) for t in tasks_base]   # 样本已 baseload=False
            sol_milp = solve_region_clean(r, tasks_milp, power, storage, {},
                                          nonai_full, absorb_full, ext,
                                          free=True,
                                          main_horizon_ids=sample_ids)
        else:
            # 无样本区域：两解相同（Δ=0），仅解一次报告基线
            sol_edf = solve_region_clean(r, tasks_base, power, storage, {},
                                         nonai_full, absorb_full, ext,
                                         free=True, main_horizon_ids=None)
            sol_milp = dict(sol_edf) if sol_edf["feasible"] else sol_edf
            n_forced_edf = 0
            sched = {}

        ok = sol_edf["feasible"] and sol_milp["feasible"]
        sm = sampling_meta[r]
        gh_ratio = sm["gh_edf_ok"] / sm["gh_sample"] if sm["gh_sample"] > 0 else 0.0
        delta_r = 0.0
        if ok and samples:
            delta_r = sol_edf["cost_main_M"] - sol_milp["cost_main_M"]
        elif not ok:
            print(f"  [警告] {r}: S_EDF/S_MILP 不可行 — 跳过 Δ")
        delta_total_M += delta_r * gh_ratio

        results["per_region"][r] = {
            "n_released": sm["n_released"], "n_edf_ok": sm["n_edf_ok"],
            "n_sample": sm["n_sample"], "gh_ratio": gh_ratio,
            "delta_M": delta_r, "n_forced_edf": n_forced_edf,
            "sample_ids": sorted(sched.keys()) if samples else [],
        }
        results["s_edf"][r] = {k: sol_edf[k] for k in
                               ("cost_main_M", "n_x", "time_s",
                                "slack_gh_total", "status", "feasible",
                                "mip_gap")}
        results["s_milp"][r] = {k: sol_milp[k] for k in
                                ("cost_main_M", "n_x", "time_s",
                                 "slack_gh_total", "status", "feasible",
                                 "mip_gap")}
        tag = "样本" if samples else "   "
        print(f"  {r}[{tag}]: 释放 {sm['n_released']:5d} | 样本 {sm['n_sample']:3d}"
              f" | 放大 {gh_ratio:8.1f}x"
              f" | Δ {delta_r:+8.3f} M | S_EDF 成本 {sol_edf['cost_main_M']:9.2f}"
              f" (n_x {sol_edf['n_x']:6d}, {sol_edf['time_s']:6.1f}s,"
              f" gap {sol_edf.get('mip_gap', float('nan')):.1%})"
              f" | S_MILP {sol_milp['cost_main_M']:9.2f}"
              f" (n_x {sol_milp['n_x']:6d}, {sol_milp['time_s']:6.1f}s,"
              f" gap {sol_milp.get('mip_gap', float('nan')):.1%})"
              f" | EDF超容 {n_forced_edf}")

    delta_pct = delta_total_M / MAIN_COST_M * 100
    print("\n" + "=" * 78)
    print(f"配对 Δ_total（按 GH 放大） = {delta_total_M:+8.3f} M 元"
          f" = {delta_pct:+6.2f}% 主解成本 {MAIN_COST_M} M")
    if delta_pct < 0.5:
        print("判定：EDF 预填次优性可忽略（主解叙事成立，R1 闭环）")
    elif delta_pct < 2.0:
        print("判定：有收益但边际，如实报告")
    else:
        print("判定：EDF 预填确实隐没更优解，与人类重议主解方案")

    out = {
        "ratio": ratio,
        "delta": {"per_region": results["per_region"], "total_M": delta_total_M,
                  "pct_of_main": delta_pct, "main_cost_M": MAIN_COST_M},
        "s_edf": results["s_edf"],
        "s_milp": results["s_milp"],
        "sampling_meta": sampling_meta,
        "meta": {"generated": "2026-08-09",
                 "source": f"sub4-clean-ratio{int(round(ratio*100)):03d}.pkl"
                           " + S_EDF/S_MILP 配对（free 求解，主时域同域）",
                 "note": "Δ 为样本任务 EDF vs MILP 选时增益，按 GH 比例放大到"
                         "EDF 可行释放子集；EDF 不可行释放任务（U_r）不计入配对",
                 "time_limit_s": MILP_TIME_LIMIT, "gap": MILP_GAP},
    }
    out_path = SCRATCH / f"s4-clean-results-ratio{int(round(ratio*100)):03d}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\n已写入 {out_path}")
    print("S4 CLEAN MODEL DONE")


if __name__ == "__main__":
    main()

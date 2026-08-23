# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 2.1 S4 正式模型 — 层 2 每区域协同 MILP（任务时间维调度 + 储能
    + 购售电 + 新能源分配统一优化），主解 ε=1.00 + 场景对比
    （碳约束 ε / 电价机制 / 新能源波动）。

原理：
    1. 半耦合架构（approach-sub4-confirmed）：
       层 1（贪心分配 + 基荷预填）在阶段 1.4 完成，本脚本加载
       sub4-preprocessed.pkl，只做层 2。
    2. 层 2 每区域独立 MILP，变量三类：
       - 非基荷自由任务 x_{j,h} ∈ {0,1}（候选窗口 h ∈ [A_j, min(L_j,2406)−D_j]）
       - 储能/购售电连续变量 7×NT：G(购电), S(卖电), R(新能源直供),
         Cg(电网充电), Cr(新能源充电), D(放电), E(SOC)
       - GPU 超容松弛 s_t ≥ 0（惩罚 M=1e6 元/GPU-hour，对齐 S2 Q4 裁定
         "允许临时超容、超容断言告警"；方案书 §8 R1）
    3. 基荷任务 + 实时任务为固定占用（阶段 1.4 EDF 定 start_h /
       到达即开工），不建变量，只贡献 GPU 占用与 AI IT 功率（固定）。
       非基荷任务在层 2 不受基荷配额 quota_aiit 约束——它们可购电完成，
       与"基荷=绿色算力"叙事不冲突（方案书 §5.2 层 2 约束无配额项）。
    4. 关键耦合约束（C3 功率平衡）：
         G_t + R_t + D_t = (AI_IT_fixed_t + AI_IT_var_t + NonAI_t)·PUE
                          + Cg_t + S_t
       其中 AI_IT_var_t = Σ_jh dem_j·p_k·overlap(j,h,t)·x_jh —— x 与连续
       变量线性耦合，任务用电优先吃新能源（零成本），不足才购电。
    5. 碳约束 ε（主时域 0-2399）：Σ G_t·ci_t ≤ 1e3·ε·C_ref_r。
       **碳基准口径（阶段 2.1 实证修正）**：C_ref 取 S4 **自身无约束解
       的碳排 E0_S4**（先跑 free 场景获得），而非 S3 baseline 基准——
       S4 负荷结构（实时+基荷任务）与 S3 给定负荷不同，E/F 固定负荷
       最小购电碳排已超 S3 基准（E 121.2 vs 84.6 kt），直接套用 S3
       基准导致 ε=1.00 主解不可行。与 S3"ε=1.00=无刻意降碳的碳排"
       语义一致（S3 基准=基准轨迹碳排，S4 基准=无约束最优解碳排）。
       ⚠️ 跨子问题：S2 的 ε 是 AI 任务购电碳排口径、S3/S4 是全设施
       购电碳排口径，两者不可直接比较（math-sub4 §9.6）。
    6. SOC 递推（S3 同构）：E_t = E_{t-1} + ηc(Cg+Cr) − D/ηd，
       E_0 = init + ηc·C_0 − D_0/ηd；终态 E_2405 ≥ init。
    7. 结清段 2400-2405 数据口径（审查 M1 修正）：电价/售电价/碳强度/
       可用新能源取 s3-preprocessed panel 的 2400-2405 **实际值**（S3 同源，
       非 2399 外推——外推会使 E 区电价高估 40%、新能源低估 72%）。
       NonAI 结清段取 c-data-cleaned region_time_data 实际值。
    8. 新能源消纳口径（审查 S1 修正）：**受限消纳（S3 B1 主口径）**——
       消纳上限 = 基准观测 UsedRenewable + RenewableCharge（c-data 逐时，
       含结清段），非 AvailableRenewable。理由：自由消纳下可用新能源
       （500-1100 MW）恒大于负荷（241-652 MW），LP 购电→0、碳排≈0、
       ε=1.00 碳上限不绑定 → 成本为负、场景对比失效（与 S3 R1 退化同因）。
       基荷策略提供消纳**下界**（P25 确定性部分），受限消纳提供**上界**
       （电网物理能力观测值），两者兼容。
    9. 场景参数化：
       - ε ∈ {1.00, 0.95, 0.90}（碳约束；S3 已实证 ε<1.00 主时域不可行
         ε_min≈0.957-0.994，S4 同口径预期类似，如实报告）
       - price_scale（电价机制）：**只放大峰段**（price > 区域均值），
         模拟"峰谷价差扩大"（审查 S5 修正：等比缩放不改变决策）
       - renew_scale（新能源波动 ±20%）：缩放受限消纳上限与利用率分母；
         ⚠️ 基荷预填结果固定（阶段 1.4 已落盘），renew 场景只体现
         "波动后的消纳能力 → 储能/购电权衡"，不体现基荷矩形重算
    10. 六指标（主时域）：成本 / 碳排 / GPU-hour 加权时延 / QoS（按时
        完成率）/ 新能源利用率（双口径）/ 区域峰值净购电。

输入数据：
    - outputs/data/sub4-preprocessed.pkl（阶段 1.4）
      tasks: id/type/source/cand/dest/baseload/start_h/arrive/dur/dem/
             latest/gh/power
      power: 区域 → {price/sell/carbon/renewable:(2400,), pue/cap/...}
      storage: 区域 → 储能参数
      carbon_base_kt / nonai_arr(2400) / p25 / T_END
    - outputs/data/s3-preprocessed.pkl（阶段 1.4 S3）
      panel: index=(Region,Hour) 0-2405，含 Price/SellPrice/CarbonIntensity/
             AvailableRenewable 实际值（结清段）
    - outputs/data/c-data-cleaned.pkl（阶段 0.3）
      region_time_data: Region/Hour/NonAI_IT_Load_MW（0-2406）
    - outputs/data/s2-preprocessed.pkl（阶段 1.4 S2）
      latency: {(from,to): ms}（时延指标）
    - 中文指标 → 变量名映射：
      购电功率→G, 卖电功率→S, 新能源直供→R, 电网充电→Cg,
      新能源充电→Cr, 放电→D, SOC→E, 超容松弛→s
      到达小时→arrive, 时长(h)→dur, GPU需求→dem, 最晚完成→latest,
      GPU-hour→gh, 单位GPU功率(MW)→power, 基荷标记→baseload,
      基荷开工→start_h

输出：
    - outputs/data/s4-results.pkl — 键：
      main: {region_solutions, metrics（六指标）, status}
      scenarios: {eps / price / renew 各场景汇总}
    - 控制台统计量（min/max/mean/std + 场景对比表，PR-014）

对应论文章节：
    问题四（S4）算-储-电协同优化 — 阶段 2.1 代码实现
"""
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
NT = 2406          # 0-2405 全时域（2406 状态结算 = E_2405）
MAIN = 2400        # 主时域 0-2399
T_END = 2406

# 求解参数
MILP_TIME_LIMIT = 600.0
MILP_GAP = 0.01
GPU_SLACK_PENALTY = 1e6    # 超容 1 GPU-hour 惩罚（元），远大于合法调度差异
TOL = 1e-6
DOL_TOL = 1e-3

# 场景默认
EPS_MAIN = 1.00            # 主解
EPS_SCEN = [1.00, 0.95, 0.90]
PRICE_SCEN = [1.0, 1.5]    # 峰谷差缩放（1.5 = 峰段价格 ×1.5）
RENEW_SCEN = [1.0, 1.2, 0.8]

# 方案 A 参照（S2 已知结果，Q3 完整回测留 2.2）
REF_A = {"C_M": 340.1, "E_kt": 251.78, "note": "S2 层1贪心+层2滚动窗成本最小化（无储能/基荷）"}


def load():
    with open(DATA / "sub4-preprocessed.pkl", "rb") as f:
        d = pickle.load(f)
    with open(DATA / "s3-preprocessed.pkl", "rb") as f:
        s3 = pickle.load(f)
    with open(DATA / "c-data-cleaned.pkl", "rb") as f:
        cd = pickle.load(f)
    with open(DATA / "s2-preprocessed.pkl", "rb") as f:
        s2 = pickle.load(f)

    # 结清段实际值（S3 panel：0-2405 电价/售电价/碳强度/新能源）
    panel = s3["panel"]
    ext = {}
    for r in REGIONS:
        pr = panel.xs(r, level="Region")
        ext[r] = {
            "price": pr["Price_CNY_per_MWh"].values,                # (2406,)
            "sellp": pr["SellPrice_CNY_per_MWh"].values,
            "carbon": pr["CarbonIntensity_tCO2_per_MWh"].values,
            "renewable": pr["AvailableRenewable_MW"].values,
        }
    # NonAI 结清段实际值 + 受限消纳上限（c-data region_time_data：0-2406）
    rtd = cd["region_time_data"]
    nonai_full = {}
    absorb_full = {}
    for r in REGIONS:
        sub = rtd[rtd["Region"] == r].sort_values("Hour")
        assert len(sub) == 2407, f"{r} NonAI 行数异常 {len(sub)}"
        nonai_full[r] = sub["NonAI_IT_Load_MW"].values[:NT].astype(float)  # (2406,)
        # 受限消纳（S3 B1 口径）：消纳上限 = UsedRenewable + RenewableCharge
        absorb_full[r] = (sub["UsedRenewable_MW"].values[:NT]
                          + sub["RenewableCharge_MW"].values[:NT]).astype(float)

    return d, ext, nonai_full, absorb_full, s2["latency"]


def fixed_occupancy(r, tasks):
    """区域 r 固定占用（不建变量）：实时任务（到达即开工，跨收尾段）
    + 基荷任务（start_h 固定，主时域内）。
    返回 (fixed_gpu[NT], fixed_aiit[NT])，单位 GPU / MW(IT 侧)。"""
    fg = np.zeros(NT, dtype=float)
    fa = np.zeros(NT, dtype=float)
    for t in tasks:
        if t["dest"] != r:
            continue
        if t["type"] == "RealTimeInference":
            h0, dur = t["arrive"], t["dur"]
        elif t.get("baseload", False) and t["start_h"] is not None:
            h0, dur = t["start_h"], t["dur"]
        else:
            continue  # 非基荷任务 → 变量
        h_end = int(np.ceil(h0 + dur))
        for hh in range(int(np.floor(h0)), min(h_end, NT)):
            ov = min(h0 + dur, hh + 1.0) - max(h0, float(hh))
            fg[hh] += t["dem"] * ov
            fa[hh] += t["dem"] * t["power"] * ov
    return fg, fa


def ai_it_var_from_solution(nbl, cands, xoff, x):
    """由 MILP 解计算非基荷任务的 AI_IT 功率序列（IT 侧，NT 长）。"""
    ait = np.zeros(NT, dtype=float)
    for i, tsk in enumerate(nbl):
        off = xoff[i]
        for k, h in enumerate(cands[i]):
            if x[off + k] > 0.5:
                h_end = int(np.ceil(h + tsk["dur"]))
                for hh in range(h, min(h_end, NT)):
                    ov = min(h + tsk["dur"], hh + 1.0) - max(h, float(hh))
                    ait[hh] += tsk["dem"] * tsk["power"] * ov
                break
    return ait


def solve_region(r, tasks, power, storage, carbon_ref, nonai_full, absorb_full,
                 ext, eps=EPS_MAIN, price_scale=1.0, renew_scale=1.0,
                 free=False):
    """层 2 每区域协同 MILP。carbon_ref: {r: kt} 碳排基准（free=True 时忽略）。
    返回解字典；status!=0 时 feasible=False。"""
    s = storage[r]
    p = power[r]
    # 结清段实际值（2406 长）；主时域用 sub4 power（2400），结清段用 S3 panel
    price = np.concatenate([p["price"], ext[r]["price"][MAIN:NT]])
    sellp = np.concatenate([p["sell"], ext[r]["sellp"][MAIN:NT]])
    ci = np.concatenate([p["carbon"], ext[r]["carbon"][MAIN:NT]])
    pue = p["pue"]
    cap_gpu = p["cap"]
    nonai = nonai_full[r]                              # (2406,) IT 侧
    # 受限消纳上限（S3 B1 口径），renew_scale 缩放模拟新能源波动
    absorb = absorb_full[r] * renew_scale              # (2406,) MW
    # 可用新能源（利用率分母口径，与 S3 一致：利用率 = (R+Cr+S)/ΣAvail）
    avail = np.concatenate([p["renewable"], ext[r]["renewable"][MAIN:NT]]) * renew_scale

    # price_scale：只放大峰段（price > 区域均值），模拟峰谷价差扩大
    if price_scale > 1.0:
        peak_mask = price > price.mean()
        price = price.copy()
        price[peak_mask] *= price_scale
        sellp = sellp.copy()
        sellp[peak_mask] *= price_scale

    # ---- 固定占用（实时 + 基荷）----
    fg, fa = fixed_occupancy(r, tasks)

    # ---- 非基荷自由任务 + 候选窗口 ----
    nbl = [t for t in tasks
           if t["dest"] == r and t["type"] != "RealTimeInference"
           and not t.get("baseload", False)]
    cands, xoff = [], []
    n_x = 0
    n_forced = 0
    for t in nbl:
        lo = int(t["arrive"])
        hi = int(min(t["latest"], T_END) - t["dur"] + 1e-9)  # 浮点保护（对齐 S2）
        if hi < lo:
            n_forced += 1  # 窗口为空：强制单点（当前数据 0 例，防御计数）
            hi = lo
        w = list(range(lo, hi + 1))
        cands.append(w)
        xoff.append(n_x)
        n_x += len(w)
    n_nbl = len(nbl)

    # ---- 变量布局 ----
    # [0, n_x) x 0-1 | [n_x+7*NT) 连续 | GPU 松弛
    off_g = n_x + 0 * NT
    off_s = n_x + 1 * NT
    off_r = n_x + 2 * NT
    off_cg = n_x + 3 * NT
    off_cr = n_x + 4 * NT
    off_d = n_x + 5 * NT
    off_e = n_x + 6 * NT
    off_slack = n_x + 7 * NT
    n_vars = n_x + 7 * NT + NT

    # ---- 目标系数 ----
    c = np.zeros(n_vars)
    c[off_g:off_g + NT] = price
    c[off_s:off_s + NT] = -sellp
    c[off_slack:] = GPU_SLACK_PENALTY

    # ---- 不等式约束：C2 GPU(含松弛) + C4 消纳 + 充电上限 + C7 碳 + 终态 ----
    n_ub = 3 * NT + 2  # free 时无碳约束行，终态保留 → 3*NT+1
    if free:
        n_ub -= 1
    A_ub = lil_matrix((n_ub, n_vars), dtype=float)
    b_ub = np.zeros(n_ub)
    row = 0
    # C2 GPU 容量：Σa·x − s_t ≤ cap − fg（s 提供超容兜底，惩罚在目标）
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
    # C4 新能源消纳（受限，S3 B1）：R + Cr ≤ absorb_t
    for t in range(NT):
        A_ub[row, off_r + t] = 1.0
        A_ub[row, off_cr + t] = 1.0
        b_ub[row] = absorb[t]
        row += 1
    # C5 充电上限：Cg + Cr ≤ MaxCharge
    for t in range(NT):
        A_ub[row, off_cg + t] = 1.0
        A_ub[row, off_cr + t] = 1.0
        b_ub[row] = s["MaxChargePower_MW"]
        row += 1
    if not free:
        # C7 碳约束（主时域）
        for t in range(MAIN):
            A_ub[row, off_g + t] = ci[t]
        b_ub[row] = 1e3 * eps * carbon_ref[r]
        row += 1
    # C8 终态 SOC：−E_{NT−1} ≤ −init
    A_ub[row, off_e + NT - 1] = -1.0
    b_ub[row] = -s["InitialSOC_MWh"]
    row += 1
    assert row == n_ub

    # ---- 等式约束：C1 恰好开工 + C3 功率平衡 + C5 SOC ----
    n_eq = n_nbl + NT + NT
    A_eq = lil_matrix((n_eq, n_vars), dtype=float)
    b_eq = np.zeros(n_eq)
    row = 0
    # C1 恰好开工一次：Σ_k x_{j,k} = 1
    for i, tsk in enumerate(nbl):
        off = xoff[i]
        for k in range(len(cands[i])):
            A_eq[row, off + k] = 1.0
        b_eq[row] = 1.0
        row += 1
    # C3 功率平衡：G + R + D − Cg − S − PUE·AI_IT_var = PUE·(AI_IT_fixed + NonAI)
    #   AI_IT_var 系数 = PUE·dem·p·ov（IT 侧：dem×power×overlap）
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
    # C5 SOC 递推：E_t − E_{t−1} − ηc(Cg+Cr) + D/ηd = 0（t=0 用 init）
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

    # ---- 边界 ----
    bounds_lb = np.zeros(n_vars)
    bounds_ub = np.full(n_vars, np.inf)
    bounds_ub[:n_x] = 1.0                       # x ∈ [0,1]（0-1 变量上界）
    bounds_lb[off_e:off_e + NT] = s["MinSOC_MWh"]
    bounds_ub[off_e:off_e + NT] = s["Capacity_MWh"]
    bounds_ub[off_g:off_g + NT] = s["MaxGridImport_MW"]
    bounds_ub[off_s:off_s + NT] = s["SellLimit_MW"]
    bounds_ub[off_d:off_d + NT] = s["MaxDischargePower_MW"]
    # Cg/Cr 上界：各自 ≤ MaxCharge（松弛为两者之和，已在 A_ub C5 行）

    constraints = [
        LinearConstraint(A_ub.tocsr(), -np.inf, b_ub),
        LinearConstraint(A_eq.tocsr(), b_eq, b_eq),
    ]
    integrality = np.zeros(n_vars)
    integrality[:n_x] = 1

    t0 = time.perf_counter()
    res = milp(c=c, constraints=constraints,
               integrality=integrality,
               bounds=Bounds(bounds_lb, bounds_ub),
               options={"time_limit": MILP_TIME_LIMIT, "mip_rel_gap": MILP_GAP,
                        "disp": False})
    dt = time.perf_counter() - t0
    if res.x is None:
        print(f"    [警告] {r} ε={eps}: milp status={res.status} 无解")
        return {"region": r, "eps": eps, "status": res.status, "feasible": False,
                "time_s": dt, "n_x": n_x, "n_nbl": n_nbl}

    x = res.x
    G = x[off_g:off_g + NT]
    S = x[off_s:off_s + NT]
    R = x[off_r:off_r + NT]
    Cg = x[off_cg:off_cg + NT]
    Cr = x[off_cr:off_cr + NT]
    D = x[off_d:off_d + NT]
    E = x[off_e:off_e + NT]
    slack = x[off_slack:]
    ait_var = ai_it_var_from_solution(nbl, cands, xoff, x)

    # 非基荷任务开工时间
    sched_nbl = {}
    for i, tsk in enumerate(nbl):
        off = xoff[i]
        sel = None
        for k, h in enumerate(cands[i]):
            if x[off + k] > 0.5:
                sel = h
                break
        if sel is None:
            print(f"    [错误] {r} 任务 {tsk['id']} 未调度（x 全 0）")
            sel = cands[i][0]
        sched_nbl[tsk["id"]] = sel

    # ---- 指标（主时域） ----
    net = G - S
    cost_main = float(np.sum(G[:MAIN] * price[:MAIN] - S[:MAIN] * sellp[:MAIN])) / 1e6
    cost_full = float(np.sum(G * price - S * sellp)) / 1e6
    carbon_kt = float(np.sum(G[:MAIN] * ci[:MAIN])) / 1e3
    peak_mw = float(net[:MAIN].max())
    std_mw = float(net[:MAIN].std())
    rng_mw = float(net[:MAIN].max() - net[:MAIN].min())
    soc_end = float(E[-1])
    dual_h = int(np.sum((Cg + Cr > DOL_TOL) & (D > DOL_TOL)))
    over_h = int(np.sum(slack > TOL))
    # 功率平衡残差（含 AI_IT_var 项）
    pb_resid = float(np.abs(G + R + D - pue * (fa + ait_var + nonai) - Cg - S).max())
    # 新能源利用率（主时域，分母=可用新能源累计 ΣAvail，S3 同口径）
    sum_avail = float(avail[:MAIN].sum())
    util_no_sell = float(np.sum(R[:MAIN] + Cr[:MAIN])) / sum_avail * 100
    util_sell = float(np.sum(R[:MAIN] + Cr[:MAIN] + S[:MAIN])) / sum_avail * 100

    return {
        "region": r, "eps": eps, "status": res.status, "time_s": dt, "feasible": True,
        "G": G, "S": S, "R": R, "Cg": Cg, "Cr": Cr, "D": D, "E": E, "slack": slack,
        "net": net, "sched_nbl": sched_nbl, "n_x": n_x, "n_nbl": n_nbl,
        "cost_main_M": cost_main, "cost_full_M": cost_full, "carbon_kt": carbon_kt,
        "peak_MW": peak_mw, "std_MW": std_mw, "range_MW": rng_mw, "soc_end_MWh": soc_end,
        "dual_hours": dual_h, "over_hours": over_h,
        "slack_gh_total": float(np.sum(slack)), "pb_resid_MW": pb_resid,
        "util_no_sell_pct": util_no_sell, "util_sell_pct": util_sell,
        "n_forced_single": n_forced,
    }


def global_metrics(tasks, latency, region_sols):
    """跨区域聚合指标：时延（GPU-hour 加权）、QoS（按时完成率）。
    region_sols: {r: sol}（含 sched_nbl）。返回 dict。"""
    total_gh = 0.0
    w_delay = 0.0
    n_done = 0
    n_tot = len(tasks)
    for t in tasks:
        gh = t["gh"]
        total_gh += gh
        # 实际开工：基荷 start_h / 实时 arrive / 非基荷 sched_nbl
        if t.get("baseload", False) and t["start_h"] is not None:
            h0 = t["start_h"]
        elif t["type"] == "RealTimeInference":
            h0 = t["arrive"]
        else:
            sol = region_sols.get(t["dest"])
            h0 = sol["sched_nbl"].get(t["id"]) if sol and sol["feasible"] else None
        if h0 is not None and h0 + t["dur"] <= t["latest"] + 1e-9:
            n_done += 1
        # 时延 = 来源→目的地时延（层 1 已定 dest）
        w_delay += gh * latency.get((t["source"], t["dest"]), 0.0)
    delay_ms = w_delay / total_gh if total_gh > 0 else 0.0
    qos = n_done / n_tot * 100
    return {"delay_ms": float(delay_ms), "qos_pct": float(qos),
            "n_done": n_done, "n_total": n_tot}


def run_scenario(tasks, power, storage, carbon_ref, nonai_full, absorb_full,
                 ext, latency,
                 eps=EPS_MAIN, price_scale=1.0, renew_scale=1.0, label="",
                 free=False):
    """跑全部 6 区域，聚合指标 + 六指标（含时延/QoS）。
    free=True：无碳约束（求 E0_S4 基准用）；eps 仍记录在 agg 中。"""
    tag = "free" if free else f"ε={eps}"
    print(f"\n--- 场景 {label} ({tag}, price={price_scale}, renew={renew_scale}) ---")
    sols = {}
    for r in REGIONS:
        sol = solve_region(r, tasks, power, storage, carbon_ref,
                           nonai_full, absorb_full, ext, eps, price_scale,
                           renew_scale, free=free)
        sols[r] = sol
        if sol["feasible"]:
            print(f"  {r}: 成本 {sol['cost_main_M']:8.2f} M元 | 碳 {sol['carbon_kt']:8.2f} kt"
                  f" | 峰值 {sol['peak_MW']:6.1f} | std {sol['std_MW']:6.1f}"
                  f" | 超容 {sol['over_hours']}h({sol['slack_gh_total']:,.0f}gh)"
                  f" | 同刻充放 {sol['dual_hours']}h | {sol['time_s']:.1f}s")
        else:
            print(f"  {r}: 不可行 (status={sol['status']})")
    feasible = [s for s in sols.values() if s["feasible"]]
    if not feasible:
        return {"label": label, "feasible": False}
    gm = global_metrics(tasks, latency, sols)
    agg = {
        "label": label, "eps": eps, "price_scale": price_scale,
        "renew_scale": renew_scale, "feasible": True,
        "cost_main_M": sum(s["cost_main_M"] for s in feasible),
        "carbon_kt": sum(s["carbon_kt"] for s in feasible),
        "peak_MW": max(s["peak_MW"] for s in feasible),
        "std_avg_MW": float(np.mean([s["std_MW"] for s in feasible])),
        "util_no_sell_pct": float(np.mean([s["util_no_sell_pct"] for s in feasible])),
        "util_sell_pct": float(np.mean([s["util_sell_pct"] for s in feasible])),
        "delay_ms": gm["delay_ms"],
        "qos_pct": gm["qos_pct"],
        "total_time_s": sum(s["time_s"] for s in feasible),
        "n_infeasible": len(sols) - len(feasible),
        "sols": sols,
    }
    print(f"  聚合: 成本 {agg['cost_main_M']:8.2f} M元 | 碳 {agg['carbon_kt']:8.2f} kt"
          f" | 峰值 {agg['peak_MW']:6.1f} | 时延 {agg['delay_ms']:.1f}ms"
          f" | QoS {agg['qos_pct']:.1f}% | 利用率 {agg['util_no_sell_pct']:.1f}%")
    return agg


def main():
    d, ext, nonai_full, absorb_full, latency = load()
    tasks = d["tasks"]
    power = d["power"]
    storage = d["storage"]
    carbon_base_kt = d["carbon_base_kt"]  # S3 基准（仅参照，非 ε 基准）
    print("=" * 78)
    print("S4 阶段 2.1 层2 协同 MILP（基荷预填 + 储能 + 场景对比）")
    print("=" * 78)

    n_bl = sum(1 for t in tasks if t.get("baseload", False))
    n_nbl = sum(1 for t in tasks
                if t["type"] != "RealTimeInference" and not t.get("baseload", False))
    print(f"任务结构: 基荷 {n_bl} | 非基荷(进MILP) {n_nbl} | 实时固定 "
          f"{sum(1 for t in tasks if t['type']=='RealTimeInference')}")

    # ---- Step 1: free 场景 → E0_S4（碳排基准，阶段 2.1 实证修正） ----
    free_agg = run_scenario(tasks, power, storage, {}, nonai_full, absorb_full,
                            ext, latency, eps=EPS_MAIN, label="S0-free", free=True)
    if free_agg.get("feasible"):
        e0 = {r: free_agg["sols"][r]["carbon_kt"] for r in REGIONS}
        print(f"\nE0_S4 碳排基准（kt）: { {r: round(v, 2) for r, v in e0.items()} }"
              f"\nS3 基准（参照，kt）: { {r: round(v, 2) for r, v in carbon_base_kt.items()} }")
    else:
        print("\n[严重] free 场景存在不可行区域——检查基础约束")
        return

    # ---- Step 2: 主解 ε=1.00（基准 = E0_S4） ----
    main_agg = run_scenario(tasks, power, storage, e0, nonai_full, absorb_full,
                            ext, latency, eps=1.00, label="S0-main")

    # ---- 场景对比（Q2 裁定 4-6 组） ----
    scen = {}
    for eps in EPS_SCEN:
        if eps >= 1.00:
            continue
        scen[f"eps-{eps}"] = run_scenario(tasks, power, storage, e0, nonai_full,
                                          absorb_full, ext, latency,
                                          eps=eps, label=f"eps-{eps}")
    scen["price-1.5"] = run_scenario(tasks, power, storage, e0, nonai_full,
                                     absorb_full, ext, latency,
                                     price_scale=1.5, label="price-1.5")
    # renew 场景：用自己的 free 解作 ε 基准（新能源波动改变消纳能力，
    # 碳排水平整体位移，相对 S0 基准的 ε 无意义——renew-0.8 相对 S0 E0
    # 会全部不可行，不能反映"新能源减少时的策略调整"）
    def renew_scen(rs, label):
        fa = run_scenario(tasks, power, storage, {}, nonai_full, absorb_full,
                          ext, latency, eps=EPS_MAIN, renew_scale=rs,
                          label=label + "-free", free=True)
        if not fa.get("feasible"):
            return fa
        e0r = {rr: fa["sols"][rr]["carbon_kt"] for rr in REGIONS}
        return run_scenario(tasks, power, storage, e0r, nonai_full, absorb_full,
                            ext, latency, eps=1.00, renew_scale=rs, label=label)

    scen["renew-1.2"] = renew_scen(1.2, "renew-1.2")
    scen["renew-0.8"] = renew_scen(0.8, "renew-0.8")

    # ---- 落盘 ----
    out = {
        "main": main_agg,
        "free": free_agg,
        "e0_s4_kt": e0,
        "s3_carbon_ref_kt": carbon_base_kt,
        "scenarios": scen,
        "ref_A": REF_A,
        "meta": {"generated": "2026-08-08",
                 "source": "sub4-preprocessed.pkl（阶段 1.4）+ 层2 协同 MILP"
                           "+ 结清段实际值（S3 panel / c-data）+ 受限消纳（S3 B1）"
                           "+ ε 基准 = S4 自身 free 解（实证修正）"},
    }
    with open(DATA / "s4-results.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n已写入 {DATA / 's4-results.pkl'}")

    # ---- 汇总表 ----
    print("\n" + "=" * 78)
    print("场景对比汇总（主时域六指标）")
    print("=" * 78)
    print(f"{'场景':<12}{'成本(M)':>10}{'碳(kt)':>10}{'峰值':>8}{'时延ms':>8}"
          f"{'QoS%':>7}{'利用率%':>9}{'可行':>5}{'超容gh':>9}")
    for name, agg in [("S0-main", main_agg)] + list(scen.items()):
        if agg.get("feasible"):
            n_ok = sum(1 for s in agg["sols"].values() if s["feasible"])
            tot_slack = sum(s.get("slack_gh_total", 0.0)
                            for s in agg["sols"].values() if s["feasible"])
            print(f"{name:<12}{agg['cost_main_M']:>10.2f}{agg['carbon_kt']:>10.2f}"
                  f"{agg['peak_MW']:>8.1f}{agg['delay_ms']:>8.1f}"
                  f"{agg['qos_pct']:>7.1f}{agg['util_no_sell_pct']:>9.1f}"
                  f"{n_ok:>4}/6{tot_slack:>9,.0f}")
        else:
            print(f"{name:<12}{'不可行':>10}")
    print(f"\n方案 A 参照（S2 已知结果，Q3 完整回测留 2.2）: "
          f"C={REF_A['C_M']}M | E={REF_A['E_kt']}kt | {REF_A['note']}")
    print("S4 MODEL DONE")


if __name__ == "__main__":
    main()

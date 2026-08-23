# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    S4 LNS（大邻域搜索）启发式求解 — 对 θ=0.3 档释放任务集（26,099 任务，
    E/F 区）做全局选时逼近：B2 电价贪心初始解 + 每轮 K=300 任务子 MILP
    迭代改进，量化"启发式 vs clean-test 最优上界（2.92%）"的 gap，
    形成论文收敛证据链（B2 贪心 1.6% → LNS → 最优上界 2.9%）。

原理：
    1. 规模问题：θ=0 全任务全窗口 ≈ 33M 候选变量不可解 → LNS 分解：
       每轮从释放任务池按 GH 分层抽 K 个作 0-1 变量，其余释放任务按当前
       选时固定；与预填/实时（固定）+ 残差任务（变量）共同构成子问题，
       复用 sub4-clean-model.solve_region_clean（free 求解，主时域同域）。
    2. 单调性保证（关键）：子问题变量窗口上界 = max(arrive+W, 当前start_h)
       ——通过把任务 latest 副本缩小为 min(latest, max(arrive+W, cur)+dur)
       实现。故当前解 ∈ 候选可行域 → 候选成本 ≤ 当前成本恒成立；
       接受准则 = 改进量 > TOL 才更新，成本单调不增，收敛于局部最优。
    3. 初始解 B2：电价感知贪心（窗口 = 主时域全窗口 [arrive,
       min(latest,2406)−dur] ∩ h+dur≤2400，按电价升序取首个 GPU+配额
       双可行小时，占用累积，失败取最早超容），dest 固定 E/F（禁改派，
       与 clean-test 一致）。
    4. 对照口径（三个基准，全部主时域购售电 cost_main_M）：
       - cost_edf_full：θ=0.3 全部释放任务 EDF（最早可行）固定 → 与
         clean-test 0.3 档上界（EDF→MILP 2.92%）**直接可比**；
       - cost_b2_full：同集 B2 电价贪心固定（= LNS 初始解）；
       - cost_lns：LNS 迭代终值。gap = (edf − lns)/主解成本。
       注：此三值同为 θ=0.3 结构（预填 30% + 储能/卖电 + 禁改派），
       内部差异纯为选时方式（EDF / B2 贪心 / LNS-MILP）。
    5. 收敛：连续 STALL=30 轮改进 < TOL 或 MAX_ITER=100 轮停止；
       每轮记录成本/改进/耗时 → 收敛曲线（论文素材）。
    6. 性能：子问题 n_x ≈ K×min(W,~1280) ≈ 50k 变量（clean-test 实测
       15s 级）；若单轮 >60s，降 K=200 或 W=120。

输入数据：
    - outputs/scratch/sub4-clean-ratio030.pkl（Clean-test 预处理 θ=0.3）
      tasks: id/type/source/cand/dest/baseload/start_h/released/edf_ok/
             sample/arrive/dur/dem/latest/gh/power
      power（区域逐时电价/售电价/碳/新能源/pue/cap）/
      storage（储能参数）/ quota_aiit_arr（逐时 AI IT 配额）/
      sampling_meta / dryrun
    - outputs/data/s3-preprocessed.pkl、c-data-cleaned.pkl、
      s2-preprocessed.pkl（经 sub4-clean-model 的 load：
      ext 结清段实际值 / nonai_full / absorb_full / latency）
    - 中文指标 → 变量名映射：
      到达→arrive, 时长→dur, GPU需求→dem, 最晚→latest, GPU-hour→gh,
      功率→power, 基荷→baseload, 开工→start_h, 释放→released,
      购电→G, 卖电→S, 成本(M元)→cost_main_M, 邻域窗口→W, 抽样数→K,
      候选变量数→n_x, 最优gap→mip_gap

输出：
    - outputs/scratch/s4-lns-results.pkl — 键：
      per_region: {r: {cost_edf, cost_b2, cost_final, iterations, stalled,
                        curve: [{it, cost, improvement, time_s}],
                        sched_final: {id: h}, edf_sched: {id: h},
                        b2_sched: {id: h}}}
      compare: {edf_total_M, b2_total_M, lns_total_M,
                edf_gain_pct, lns_gain_pct, gap_vs_upper_pct,
                b1_M, b2_M, main_M, upper_bound_pct}
      params: {W, K, MAX_ITER, STALL, TOL, seed}
    - 控制台收敛日志（PR-014）

对应论文章节：
    问题四（S4）算-储-电协同优化 — §7.3 基荷策略合理性检验（Q3 对照）
    与大规模任务调度的 LNS 可解性分析
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
NT = 2406
MAIN = 2400
T_END = 2406

# LNS 参数（超时降档：K→200 / W→120）
K = 300            # 每轮邻域抽样任务数
W = 168            # 邻域窗口截断（h ≤ arrive+W），小时
MAX_ITER = 100     # 最大轮数
STALL = 30         # 连续无改进停止阈值（轮）
TOL = 0.005        # 改进量阈值（M 元/轮）
N_STRATA = 5       # GH 分层数
SEED = 42

# 对照基准（s4-results.pkl / s4-baseline-heuristic.pkl 实测）
MAIN_COST_M = 2166.64
B1_COST_M = 2253.57
B2_COST_M = 2216.74
UPPER_BOUND_PCT = 2.92   # clean-test 0.3 档（EDF→MILP 配对增益，% 主解成本）


def load_modules():
    """加载 sub4-model（取 load）与 sub4-clean-model（取 solve_region_clean），
    均重定向 BASE/DATA（原文件指向已删除的旧 worktree）。审查#1 修复。"""
    spec4 = importlib.util.spec_from_file_location(
        "sub4model", SCRATCH / "sub4-model.py")
    M4 = importlib.util.module_from_spec(spec4)
    spec4.loader.exec_module(M4)
    M4.BASE = BASE
    M4.DATA = DATA
    spec = importlib.util.spec_from_file_location(
        "cleanmodel", SCRATCH / "sub4-clean-model.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    M.BASE = BASE
    M.DATA = DATA
    return M4, M


def price_aware_feasible_hour(t, r, occ_gpu, occ_aiit, cap_gpu, quota_arr,
                              price):
    """窗口内按电价升序找首个 GPU+配额双可行小时（B2 电价感知贪心）。

    窗口 = [arrive, min(latest,2406)−dur] ∩ 主时域（同 EDF 预填）。
    返回小时 h；无可行返回 None。不修改占用表。
    """
    lo = int(t['arrive'])
    hi = int(min(t['latest'], T_END) - t['dur'])
    hi = min(hi, MAIN - int(np.ceil(t['dur'])))
    if hi < lo:
        return None
    order = np.argsort(price[lo:hi + 1], kind="stable")
    for k in order:
        h = lo + int(k)
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
    for hh in range(h, min(h_end, len(occ_gpu))):   # 防御：不越界（审查#4）
        overlap = min(h + t['dur'], hh + 1) - max(h, hh)
        occ_gpu[hh] += t['dem'] * overlap
        occ_aiit[hh] += t['dem'] * t['power'] * overlap


def base_occupancy(r, tasks, NT_local=NT):
    """实时 + 预填基荷固定占用（与 solve_region_clean.fixed_occupancy 同口径）。"""
    occ_gpu = np.zeros(NT_local, dtype=float)
    occ_aiit = np.zeros(NT_local, dtype=float)
    for t in tasks:
        if t["type"] == "RealTimeInference":
            h0, dur = t["arrive"], t["dur"]
        elif t.get("baseload", False) and t["start_h"] is not None:
            h0, dur = t["start_h"], t["dur"]
        else:
            continue
        h_end = int(np.ceil(h0 + dur))
        for hh in range(int(np.floor(h0)), min(h_end, NT_local)):
            ov = min(h0 + dur, hh + 1.0) - max(h0, float(hh))
            occ_gpu[hh] += t["dem"] * ov
            occ_aiit[hh] += t["dem"] * t["power"] * ov
    return occ_gpu, occ_aiit


def greedy_schedule(r, rel_tasks, tasks, power, quota_arr, price, mode):
    """释放任务贪心调度（dest 固定 r，占用累积）。

    mode='edf'：最早可行；mode='price'：电价最低可行（B2）。
    返回 {id: h}；失败任务取 arrive（超容，slack 兜底）。
    """
    occ_gpu, occ_aiit = base_occupancy(r, tasks)
    sched = {}
    n_forced = 0
    for t in sorted(rel_tasks, key=lambda t: -t['gh']):
        if mode == 'edf':
            # 最早可行：与 earliest_feasible_hour 相同逻辑（主时域窗口）
            lo = int(t['arrive'])
            hi = int(min(t['latest'], T_END) - t['dur'])
            hi = min(hi, MAIN - int(np.ceil(t['dur'])))
            h = None
            for cand in range(lo, hi + 1):
                h_end = int(np.ceil(cand + t['dur']))
                ok = True
                for hh in range(cand, h_end):
                    ov = min(cand + t['dur'], hh + 1) - max(cand, hh)
                    if occ_gpu[hh] + t['dem'] * ov > power[r]['cap'] + 1e-6:
                        ok = False
                        break
                    if occ_aiit[hh] + t['dem'] * t['power'] * ov > quota_arr[hh] + 1e-6:
                        ok = False
                        break
                if ok:
                    h = cand
                    break
        else:
            h = price_aware_feasible_hour(t, r, occ_gpu, occ_aiit,
                                          power[r]['cap'], quota_arr, price)
        if h is None:
            h = int(t['arrive'])
            n_forced += 1
        commit_placement(t, h, occ_gpu, occ_aiit)
        sched[t['id']] = h
    return sched, n_forced


def gh_strat_sample(pool, K, n_strata, seed):
    """GH 分位数分层抽样（不重复，确定性种子）。pool: [(id, gh)]。"""
    if len(pool) <= K:
        return [tid for tid, _ in pool]
    ghs = np.array([gh for _, gh in pool])
    qs = np.quantile(ghs, np.linspace(0.0, 1.0, n_strata + 1)[1:-1])
    strata = [[] for _ in range(n_strata)]
    for tid, gh in pool:
        s = int(np.searchsorted(qs, gh, side="right"))
        strata[min(s, n_strata - 1)].append(tid)
    rng = np.random.default_rng(seed)
    out = []
    per = int(np.ceil(K / n_strata))
    for s in strata:
        rng.shuffle(s)
        out.extend(s[:per])
    return out[:K]


def build_subproblem(tasks_r, sched, k_ids, W):
    """构建子问题任务列表（浅拷贝，不污染原数据）。

    - K 任务：baseload=False（变量），latest 副本缩小 → 窗口上界
      = max(arrive+W, 当前start_h)，保证当前解 ∈ 候选可行域；
    - 非 K 释放任务：baseload=True + 当前选时（固定）；
    - 残差（released=False 且 baseload=False）：保持变量；
    - 预填/实时：保持原标记。
    """
    out = []
    for t in tasks_r:
        tc = dict(t)
        if t['id'] in k_ids:
            tc['baseload'] = False
            tc['start_h'] = None
            cur = sched.get(t['id'])
            hi_bound = max(t['arrive'] + W, cur) if cur is not None \
                else t['arrive'] + W
            tc['latest'] = min(t['latest'], hi_bound + t['dur'])
        elif t.get('released', False):
            tc['baseload'] = True
            tc['start_h'] = sched[t['id']]
        out.append(tc)
    return out


def region_cost(M, r, tasks_r, power, storage, nonai_full, absorb_full, ext):
    """求解区域完整成本（残差为变量、其余按任务标记固定）。"""
    sol = M.solve_region_clean(r, tasks_r, power, storage, {},
                               nonai_full, absorb_full, ext,
                               free=True, main_horizon_ids=None)
    return sol


CHECKPOINT = SCRATCH / "s4-lns-checkpoint.pkl"


def save_checkpoint(ckpt):
    """原子写检查点（tmp + replace），供断点续跑与中途分析。"""
    tmp = CHECKPOINT.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    tmp.replace(CHECKPOINT)


def main():
    global K
    import sys
    max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_ITER
    clean_pkl = SCRATCH / "sub4-clean-ratio030.pkl"
    assert clean_pkl.exists(), f"缺少 {clean_pkl}（先跑 preprocess-sub4-clean.py 0.3）"
    M4, M = load_modules()
    _, ext, nonai_full, absorb_full, _ = M4.load()

    with open(clean_pkl, "rb") as f:
        clean = pickle.load(f)
    tasks = clean["tasks"]
    power = clean["power"]
    storage = clean["storage"]
    quota_arr = clean["quota_aiit_arr"]

    print("=" * 78)
    print(f"S4 LNS 启发式求解: K={K} W={W}h max_iter={max_iter} "
          f"STALL={STALL} TOL={TOL}M")
    print("=" * 78)

    ckpt = {}
    if CHECKPOINT.exists():
        with open(CHECKPOINT, "rb") as f:
            ckpt = pickle.load(f)
        print(f"[resume] 加载检查点: {list(ckpt.keys())}")

    results = {"per_region": {}, "params": {"K": K, "W": W, "max_iter": max_iter,
               "STALL": STALL, "TOL": TOL, "seed": SEED,
               "n_strata": N_STRATA}}
    lns_total_M = 0.0
    edf_total_M = 0.0
    b2_total_M = 0.0
    for r in REGIONS:
        tasks_r = [t for t in tasks if t["dest"] == r]
        rel = [t for t in tasks_r if t.get("released", False)]
        if not rel:
            # 无释放任务区域：EDF=B2=LNS=现状（残差变量），仅解一次
            sol = region_cost(M, r, tasks_r, power, storage, nonai_full,
                              absorb_full, ext)
            c = sol["cost_main_M"] if sol["feasible"] else float("nan")
            results["per_region"][r] = {"cost_edf": c, "cost_b2": c,
                                        "cost_final": c, "iterations": 0,
                                        "stalled": 0, "curve": [],
                                        "sched_final": {}, "edf_sched": {},
                                        "b2_sched": {}}
            lns_total_M += c
            edf_total_M += c
            b2_total_M += c
            print(f"  {r}: 无释放任务 | 成本 {c:.2f} M")
            continue

        price = power[r]["price"]           # (2400,) 主时域
        pool = [(t["id"], t["gh"]) for t in rel]

        rc = ckpt.get(r)
        if rc and rc.get("done"):
            # 已完成区域：直接复用缓存（不重算）
            cost_edf = rc["cost_edf"]
            cost_b2 = rc["cost_b2"]
            sched_edf = rc["edf_sched"]
            sched_b2 = rc["b2_sched"]
            nf_edf = rc["n_forced_edf"]
            nf_b2 = rc["n_forced_b2"]
            slack_edf = rc["slack_edf_gh"]
            slack_b2 = rc["slack_b2_gh"]
            sched = rc["sched"]
            current_cost = rc["current_cost"]
            curve = rc["curve"]
            it = rc["it"]
            stalled = rc["stalled"]
            slow_rounds = rc.get("slow_rounds", 0)
            print(f"  {r}: [resume] 已完成（cost {current_cost:.2f} M,"
                  f" {it} 轮, stalled {stalled}）")
        else:
            if rc:
                # 未完成：恢复进度（跳过 EDF/B2 重算，续跑 LNS 循环）
                cost_edf = rc["cost_edf"]
                cost_b2 = rc["cost_b2"]
                sched_edf = rc["edf_sched"]
                sched_b2 = rc["b2_sched"]
                nf_edf = rc["n_forced_edf"]
                nf_b2 = rc["n_forced_b2"]
                slack_edf = rc["slack_edf_gh"]
                slack_b2 = rc["slack_b2_gh"]
                sched = rc["sched"]
                current_cost = rc["current_cost"]
                curve = rc["curve"]
                it = rc["it"]
                stalled = rc["stalled"]
                slow_rounds = rc.get("slow_rounds", 0)
                print(f"  {r}: [resume] 从 it={it} 续跑（cost {current_cost:.2f} M）")
            else:
                # 全新区域：EDF / B2 对照 + LNS 初始化
                t0 = time.perf_counter()
                sched_edf, nf_edf = greedy_schedule(r, rel, tasks_r, power,
                                                    quota_arr[r], price,
                                                    mode="edf")
                tasks_edf = [dict(t) for t in tasks_r]
                for t in tasks_edf:
                    if t["id"] in sched_edf:
                        t["baseload"] = True
                        t["start_h"] = sched_edf[t["id"]]
                sol_edf = region_cost(M, r, tasks_edf, power, storage,
                                      nonai_full, absorb_full, ext)
                cost_edf = sol_edf["cost_main_M"] if sol_edf["feasible"] \
                    else float("nan")
                slack_edf = sol_edf["slack_gh_total"] if sol_edf["feasible"] \
                    else float("nan")
                print(f"  {r}: EDF 全量固定 → 成本 {cost_edf:9.2f} M"
                      f" (超容任务 {nf_edf}, slack {slack_edf:,.0f}gh,"
                      f" {time.perf_counter()-t0:.1f}s)")

                t0 = time.perf_counter()
                sched_b2, nf_b2 = greedy_schedule(r, rel, tasks_r, power,
                                                  quota_arr[r], price,
                                                  mode="price")
                tasks_b2 = [dict(t) for t in tasks_r]
                for t in tasks_b2:
                    if t["id"] in sched_b2:
                        t["baseload"] = True
                        t["start_h"] = sched_b2[t["id"]]
                sol_b2 = region_cost(M, r, tasks_b2, power, storage,
                                     nonai_full, absorb_full, ext)
                cost_b2 = sol_b2["cost_main_M"] if sol_b2["feasible"] \
                    else float("nan")
                slack_b2 = sol_b2["slack_gh_total"] if sol_b2["feasible"] \
                    else float("nan")
                print(f"  {r}: B2 电价贪心 → 成本 {cost_b2:9.2f} M"
                      f" (超容任务 {nf_b2}, slack {slack_b2:,.0f}gh,"
                      f" {time.perf_counter()-t0:.1f}s)")
                sched = dict(sched_b2)
                current_cost = cost_b2
                curve = []
                it = 0
                stalled = 0
                slow_rounds = 0

        # ---- LNS 迭代（全新/续跑共用；每 5 轮落检查点） ----
        t_start = time.perf_counter()
        while it < max_iter and stalled < STALL:
            t0 = time.perf_counter()
            k_ids = set(gh_strat_sample(pool, K, N_STRATA, SEED + it))
            sub_tasks = build_subproblem(tasks_r, sched, k_ids, W)
            sol = M.solve_region_clean(r, sub_tasks, power, storage, {},
                                       nonai_full, absorb_full, ext,
                                       free=True, main_horizon_ids=k_ids)
            dt = time.perf_counter() - t0
            improvement = 0.0
            if sol["feasible"]:
                cand_cost = sol["cost_main_M"]
                improvement = current_cost - cand_cost
                if improvement > TOL:
                    for tid, h in sol["sched_nbl"].items():
                        if tid in k_ids:          # 只更新本轮变量任务
                            sched[tid] = h
                    current_cost = cand_cost
                    stalled = 0
                else:
                    stalled += 1
            else:
                cand_cost = float("nan")
                stalled += 1
            # 自适应降档（审查#5）：连续 3 轮 >90s → K 减半（保下限 150）
            if dt > 90.0:
                slow_rounds += 1
                if slow_rounds >= 3 and K > 150:
                    K = max(150, K // 2)
                    print(f"    [自适应降档] 连续 3 轮 >90s → K={K}")
                    slow_rounds = 0
            else:
                slow_rounds = 0
            curve.append({"it": it, "cost": current_cost,
                          "cand_cost": cand_cost, "improvement": improvement,
                          "time_s": dt, "n_x": sol["n_x"],
                          "status": sol["status"],
                          "slack_gh": sol["slack_gh_total"]
                          if sol["feasible"] else float("nan")})
            it += 1
            if it % 10 == 0 or it == 1:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / it * (max_iter - it) if it > 0 else 0.0
                print(f"    it {it:3d}: cost {current_cost:9.2f} M"
                      f" (Δ{improvement:+.4f}M, {dt:.1f}s)"
                      f" | ETA {eta/60:.0f}min")
            # 检查点（每 5 轮 / 提前停止 / 达上限）
            if it % 5 == 0 or stalled >= STALL or it >= max_iter:
                ckpt[r] = {"b2_sched": sched_b2, "edf_sched": sched_edf,
                           "cost_edf": cost_edf, "cost_b2": cost_b2,
                           "n_forced_edf": nf_edf, "n_forced_b2": nf_b2,
                           "slack_edf_gh": slack_edf, "slack_b2_gh": slack_b2,
                           "sched": sched, "current_cost": current_cost,
                           "it": it, "stalled": stalled,
                           "slow_rounds": slow_rounds, "curve": curve,
                           "done": False}
                save_checkpoint(ckpt)
        ckpt[r]["done"] = True
        save_checkpoint(ckpt)

        results["per_region"][r] = {
            "cost_edf": cost_edf, "cost_b2": cost_b2, "cost_final": current_cost,
            "iterations": it, "stalled": stalled, "curve": curve,
            "sched_final": sched, "edf_sched": sched_edf,
            "b2_sched": sched_b2,             # 审查#2 修复：B2 初始排程独立保存
            "n_forced_edf": nf_edf, "n_forced_b2": nf_b2,
            "slack_edf_gh": slack_edf, "slack_b2_gh": slack_b2,
        }
        edf_total_M += cost_edf
        b2_total_M += cost_b2
        lns_total_M += current_cost
        print(f"  {r}: LNS 终值 → 成本 {current_cost:9.2f} M"
              f" ({it} 轮, stalled {stalled}) | EDF→LNS 节省 {cost_edf-current_cost:+8.3f} M"
              f" | B2→LNS 节省 {cost_b2-current_cost:+8.3f} M")

    # ---- 汇总对比 ----
    edf_gain_pct = (edf_total_M - lns_total_M) / MAIN_COST_M * 100
    b2_gain_pct = (b2_total_M - lns_total_M) / MAIN_COST_M * 100
    gap_vs_upper_pct = (UPPER_BOUND_PCT - edf_gain_pct)
    print("\n" + "=" * 78)
    print("汇总对比（主时域成本，M 元）")
    print("=" * 78)
    print(f"  B1 EDF（无储能卖电，全任务）        : {B1_COST_M:9.2f}")
    print(f"  B2 电价贪心（无储能卖电，全任务）    : {B2_COST_M:9.2f}")
    print(f"  主解（95.3% 预填）                  : {MAIN_COST_M:9.2f}")
    print(f"  θ=0.3 结构 · EDF 全量固定           : {edf_total_M:9.2f}")
    print(f"  θ=0.3 结构 · B2 贪心（LNS 初始）     : {b2_total_M:9.2f}")
    print(f"  θ=0.3 结构 · LNS 终值               : {lns_total_M:9.2f}")
    print(f"\n  EDF→LNS 增益 = {edf_gain_pct:+.2f}% 主解成本"
          f"（clean-test 最优上界 {UPPER_BOUND_PCT}%）")
    print(f"  B2→LNS 增益 = {b2_gain_pct:+.2f}%")
    print(f"  gap vs 最优上界 = {gap_vs_upper_pct:.2f} pct（LNS 达到"
          f" {edf_gain_pct/UPPER_BOUND_PCT*100:.0f}% 上界）")

    out = {
        "per_region": results["per_region"],
        "params": results["params"],
        "compare": {
            "b1_M": B1_COST_M, "b2_M": B2_COST_M, "main_M": MAIN_COST_M,
            "edf_total_M": edf_total_M, "b2_total_M": b2_total_M,
            "lns_total_M": lns_total_M,
            "edf_gain_pct": edf_gain_pct, "b2_gain_pct": b2_gain_pct,
            "upper_bound_pct": UPPER_BOUND_PCT,
            "gap_vs_upper_pct": gap_vs_upper_pct,
            "pct_of_upper": edf_gain_pct / UPPER_BOUND_PCT * 100
            if UPPER_BOUND_PCT else 0.0,
        },
        "meta": {"generated": "2026-08-09",
                 "source": "sub4-clean-ratio030.pkl + LNS（B2 初始 + 子 MILP 迭代，"
                           "free 求解，主时域同域，检查点可续跑）",
                 "note": "EDF/B2/LNS 同为 θ=0.3 结构（预填 30% + 储能/卖电 + 禁改派），"
                         "内部差异纯为选时方式；与 clean-test 0.3 档上界可比"},
    }
    out_path = SCRATCH / "s4-lns-results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\n已写入 {out_path}")
    print("S4 LNS DONE")


if __name__ == "__main__":
    main()

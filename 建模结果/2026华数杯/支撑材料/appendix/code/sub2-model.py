# -*- coding: utf-8 -*-
"""
目的：
    阶段 2.1 S2 正式模型代码 — 碳感知任务调度：模块一 S1 时移基线回测、
    模块二 层1 容量感知目的地分配、模块三 层2 每区 24h 滚动窗时间维 MILP
    + ε-约束三档迭代、模块四 评价指标，并出图 + 代理值核销 + Artifact 登记

原理：
    1. 模块一（Q1 裁定：正式基线=零迁移时移 MILP）：每区域本地任务
       24h 滚动窗时间维 MILP（可时移），得正式基线 C₀/E₀（441.7M/358.0kt
       为朴素口径参考值，不作复现目标）
    2. 模块二：容量感知贪心 — 候选按成本升序，选"demand+gh ≤ 0.9·Cap·2400"
       者；全满则退路（候选内负载率最低，允许临时超容，Q4 裁定，超容断言告警）
    3. 模块三：层 2 每区域滚动窗 MILP（S1 Alpha 同构：恰好一次/容量重叠折算/
       时限；目标改为成本最小 min Σ dem·power·PUE·Price(h)）；ε-约束迭代
       （Q3 裁定：收敛 |E-ηE₀|/ηE₀<0.5% 或 E≤ηE₀，迭代上限 3 轮，批量让渡
       按区域碳强度降序、改派候选内最低碳区至 100% 容量，兜底 ε 放宽 +1%）
    4. 实时推理任务时间维固定开工（到达即开工，20ms SLA），仅迁移目的地可变
    5. 滚动窗语义（Q2 裁定实现化）：任务在 arrive 所在窗求解，候选开工
       h ∈ [arrive, min(arrive+W_EXT, latest-dur)]（跨窗后移平抑到达峰值，
       W_EXT=48h）；容量约束覆盖候选运行全小时（防无约束预订未来容量）。
       本窗排不下的任务顺延下窗（pending）。实时任务时间维固定。
       实验依据：窗内开工（h<w1）时 F 区窗24 需求峰值 3893 vs cap 966 必然
       infeasible；跨窗后移后峰值可平抑，顺序求解=先到先得，区域 GPU-hour
       ≤90% 容量时几乎总能排下
    6. 电价/碳序列扩展至 2406（2400-2406 用 2399 值外推，注释口径）；
       最后一窗 [2376,2406) 容量约束覆盖 30h（收尾末端结清，同 S1）
    7. 成本 = Σ dem·power·PUE·price[hh]·重叠；碳排同理（carbon[hh]）

输入数据：
    - outputs/data/s2-preprocessed.pkl（阶段 1.4 预处理）
    - 键 → 变量名映射：
      tasks: id/type/source/cand/arrive/dur/dem/latest/latency/gh/power
      power: 区域 → {price/sell/carbon/renewable:(2400,) 数组, pue/cap/...: 标量}
      regions/type_maxlat/latency/power_mapping/T_END: 元数据
      network_latency 时延矩阵自 s2-preprocessed.pkl['latency']（迁移时延评价）

输出：
    - outputs/data/s2-results.pkl — 分配 dest/承接量/退路、各 η 调度成本/碳/时延、
      C₀/E₀（正式基线）、收敛迭代记录
    - outputs/figures/sub2-reachability.pdf — 6×6 时延热力图（≤SLA 标记）
    - outputs/figures/sub2-region-load.pdf — 6 区域承接任务/负载率条形图
    - outputs/figures/sub2-epsilon-curve.pdf — ε 敏感性曲线（碳排上限 vs 成本）
    - solution/artifacts/tables/s2-results.tex — 迁移收益对比表 LaTeX 片段
    - 控制台统计量（min/max/mean/std，PR-014）

对应论文章节：
    问题二（S2）碳感知任务调度 — 阶段 2.1 代码实现
"""
import pickle
import time
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import milp, LinearConstraint, Bounds

# === 中文字体与负号（chart-generator 强制前置）===
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Source Han Sans CN"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(r"e:\MathModel_pj-2026-C")
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
H = 2400
T_END = 2406
WIN = 24                # 滚动窗宽（按 arrive 分窗）
W_EXT = 48              # 候选开工小时跨窗上限（后移平抑到达峰值）
THRESHOLD = 0.90        # 容量感知阈值
ETA_LEVELS = [1.0, 0.9, 0.8]
MILP_TIME_LIMIT = 20    # 每窗秒（gap=0.01 近似解 20s 内可达，32 核下充裕）
MILP_GAP = 0.01
NPROC = 8               # 区域并行线程数（CPU 32 核）

# ============================================================
# 数据加载
# ============================================================
with open(BASE / "outputs" / "data" / "s2-preprocessed.pkl", "rb") as f:
    prep = pickle.load(f)
tasks = prep["tasks"]
power = prep["power"]
latency = prep["latency"]
pm = prep["power_mapping"]
type_maxlat = prep["type_maxlat"]

# 电价/碳扩展至 T_END（2400-2406 用 2399 值外推）
CACHE_DIR = BASE / "outputs" / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def extend(arr):
    out = np.full(T_END, arr[-1])
    out[:H] = arr
    return out

PRICE = {r: extend(power[r]["price"]) for r in REGIONS}
CARBON = {r: extend(power[r]["carbon"]) for r in REGIONS}
PUE = {r: power[r]["pue"] for r in REGIONS}
CAP = {r: power[r]["cap"] for r in REGIONS}
CAP_GH = {r: CAP[r] * H for r in REGIONS}

# 碳强度均值（让渡排序用）
CARB_MEAN = {r: power[r]["carbon"].mean() for r in REGIONS}

# 窗口列表：0-2376 每 24h，最后一窗 [2376, 2406) 覆盖收尾末端
WINDOWS = [(w0, w0 + WIN) for w0 in range(0, H, WIN)]
WINDOWS[-1] = (WINDOWS[-1][0], T_END)


def stat(name, arr):
    a = np.asarray(arr, dtype=float)
    print(f"{name}: min={a.min():.3f}, max={a.max():.3f}, mean={a.mean():.3f}, std={a.std():.3f}")
    return a


def add_usage(use, h, dur, dem):
    """把任务 [h, h+dur) 的 GPU-hour 占用写回逐小时 use 数组"""
    s, e = h, h + dur
    hi = int(np.floor(s))
    while s < e and hi < T_END:
        ov = min(e, hi + 1.0) - max(s, float(hi))
        if ov > 0:
            use[hi] += dem * ov
        s = hi + 1.0
        hi = int(np.floor(s))


def task_cost(t, r, h):
    """成本 = Σ_hh dem·power·PUE·price[hh]·重叠"""
    s, e = h, h + t["dur"]
    hi = int(np.floor(s))
    total = 0.0
    while s < e and hi < T_END:
        ov = min(e, hi + 1.0) - max(s, float(hi))
        if ov > 0:
            total += t["dem"] * t["power"] * PUE[r] * PRICE[r][hi] * ov
        s = hi + 1.0
        hi = int(np.floor(s))
    return total


def task_co2(t, r, h):
    s, e = h, h + t["dur"]
    hi = int(np.floor(s))
    total = 0.0
    while s < e and hi < T_END:
        ov = min(e, hi + 1.0) - max(s, float(hi))
        if ov > 0:
            total += t["dem"] * t["power"] * PUE[r] * CARBON[r][hi] * ov
        s = hi + 1.0
        hi = int(np.floor(s))
    return total


def subtract_usage(use, h, dur, dem):
    """从逐小时 use 数组减去任务 [h, h+dur) 的占用（future_occ 计算用）"""
    s, e = h, h + dur
    hi = int(np.floor(s))
    while s < e and hi < T_END:
        ov = min(e, hi + 1.0) - max(s, float(hi))
        if ov > 0:
            use[hi] -= dem * ov
        s = hi + 1.0
        hi = int(np.floor(s))


def greedy_base(region_tasks, r):
    """全局 EDF 贪心基解：实时任务固定 arrive，非实时按
    (latest-arrive-dur, -dem) 排序，从 arrive 扫最早可行小时（可后移）。
    区域 GPU-hour ≤ 容量时几乎总能可行，为逐窗 MILP 提供基解兜底。
    返回 (sched, use, n_forced)"""
    use = np.zeros(T_END)
    sched = {}
    cap = CAP[r]
    n_forced = 0
    order = sorted(region_tasks, key=lambda x: (x["latest"] - x["arrive"] - x["dur"], -x["dem"]))
    for t in order:
        if t["type"] == "RealTimeInference":
            h = int(t["arrive"])  # 实时任务时间维固定（20ms SLA）
        else:
            lo = int(t["arrive"])
            hi = int(min(t["latest"], T_END) - t["dur"] + 1e-9)
            if hi < lo:
                hi = lo
            h = None
            for cand_h in range(lo, hi + 1):
                ok = True
                s, e = cand_h, cand_h + t["dur"]
                hh = int(np.floor(s))
                while s < e and hh < T_END:
                    ov = min(e, hh + 1.0) - max(s, float(hh))
                    if ov > 0 and use[hh] + t["dem"] * ov > cap:
                        ok = False
                        break
                    s = hh + 1.0
                    hh = int(np.floor(s))
                if ok:
                    h = cand_h
                    break
            if h is None:
                h = hi  # 兜底（总量≤容量时不应发生）
                n_forced += 1
        sched[t["id"]] = (h, r)
        add_usage(use, h, t["dur"], t["dem"])
    return sched, use, n_forced


# ============================================================
# 滚动窗时间维 MILP（模块一/模块三共用）
# ============================================================
def rolling_schedule(region_tasks, r, label=""):
    """方案 K3：全局 EDF 贪心基解 + 顺序逐窗 MILP 改进（未来基解预留）。
    - 基解保证全局可行（实时固定 arrive；非实时 EDF 最早可行、可后移）
    - 顺序处理窗 w：约束 ub = cap − use（实时+前序窗最终解） − future_occ
      （本窗及未来窗基解占用）→ 本窗任务只能在未预留容量内安排
    - 数学保证：末窗约束含全部前序最终解 → 全局总占用 ≤ cap
    - 每窗候选跨窗 W_EXT 平抑到达峰值；MILP 无解时基解兜底（Q4 允许）
    返回 (sched, use, cost, co2, n_status, n_fallback, n_forced, dt)"""
    t_start = time.time()
    sched_g, use_g, n_forced = greedy_base(region_tasks, r)
    free_tasks = [t for t in region_tasks if t["type"] != "RealTimeInference"]
    by_arr = sorted(free_tasks, key=lambda t: t["arrive"])
    # 实时任务占用（固定，MILP 不动）
    realtime_use = np.zeros(T_END)
    for t in region_tasks:
        if t["type"] == "RealTimeInference":
            add_usage(realtime_use, sched_g[t["id"]][0], t["dur"], t["dem"])
    use = realtime_use.copy()      # 累积：实时 + 前序窗最终解
    pre_base = np.zeros(T_END)     # 前序窗基解占用累积
    sched = dict(sched_g)
    n_status = {}
    n_fallback = 0
    dt = 0.0
    cap = CAP[r]
    for w0, w1 in WINDOWS:
        jw = [t for t in by_arr if w0 <= t["arrive"] < w1]
        if jw:
            # future_occ = 未来窗基解占用（不含本窗；本窗任务可在自身基解
            # 空间内重排 → 高负载区（E/F 90%）MILP 有解，避免退化贪心）
            future_occ = use_g - realtime_use - pre_base
            for t in jw:
                subtract_usage(future_occ, sched_g[t["id"]][0], t["dur"], t["dem"])
            # 候选窗（跨窗后移：h ≤ min(arrive+W_EXT, latest-dur)）
            cand, xoff, col = [], [], 0
            max_h = w0
            for t in jw:
                lo = int(max(t["arrive"], w0))
                hi = int(min(t["latest"], T_END) - t["dur"] + 1e-9)
                hi = min(lo + W_EXT, hi)
                if hi < lo:
                    hi = lo
                w = list(range(lo, hi + 1))
                cand.append(w)
                xoff.append(col)
                col += len(w)
                max_h = max(max_h, hi)
            # 目标系数（成本）
            c = np.zeros(col)
            for i, t in enumerate(jw):
                off = xoff[i]
                for k, h in enumerate(cand[i]):
                    c[off + k] = task_cost(t, r, h)
            # 恰好一次约束
            eq_rows = []
            for i, off in enumerate(xoff):
                row = np.zeros(col)
                row[off:off + len(cand[i])] = 1.0
                eq_rows.append(row)
            # 容量约束（ub = cap − 前序最终 − 本窗及未来基解预留）
            max_dur = max(t["dur"] for t in jw)
            hhs = list(range(w0, min(int(max_h + max_dur) + 1, T_END)))
            A1, ub1 = [], []
            for hh in hhs:
                row = np.zeros(col)
                for i, t in enumerate(jw):
                    off = xoff[i]
                    for k, h in enumerate(cand[i]):
                        ov = min(h + t["dur"], hh + 1.0) - max(float(h), float(hh))
                        if ov > 0:
                            row[off + k] += t["dem"] * ov
                A1.append(row)
                ub1.append(cap - use[hh] - future_occ[hh])
            constraints = [
                LinearConstraint(np.array(eq_rows), np.ones(len(eq_rows)), np.ones(len(eq_rows))),
                LinearConstraint(np.array(A1), -np.inf, np.array(ub1)),
            ]
            t0 = time.time()
            res = milp(c=c, constraints=constraints, integrality=np.ones(col),
                       bounds=Bounds(np.zeros(col), np.ones(col)),
                       options={"time_limit": MILP_TIME_LIMIT, "mip_rel_gap": MILP_GAP,
                                "disp": False})
            dt += time.time() - t0
            n_status[res.status] = n_status.get(res.status, 0) + 1
            if res.x is None:
                # 基解兜底（前序改进后理论可能，Q4 允许）
                n_fallback += 1
                print(f"⚠️ [MILP {r} 窗{w0}] status={res.status} 无解，{len(jw)} 任务用基解兜底")
            else:
                if res.status != 0:
                    print(f"⚠️ [MILP {r} 窗{w0}] status={res.status}（time-limit），解为近似最优 (gap={MILP_GAP})")
                x = res.x
                for i, t in enumerate(jw):
                    off = xoff[i]
                    sel = None
                    for k, h in enumerate(cand[i]):
                        if x[off + k] > 0.5:
                            sel = h
                            break
                    if sel is None:
                        sel = cand[i][0]
                    sched[t["id"]] = (sel, r)
            # use 累积本窗最终解（基解兜底时用基解位置）
            for t in jw:
                add_usage(use, sched[t["id"]][0], t["dur"], t["dem"])
        # pre_base 累积本窗基解
        for t in jw:
            add_usage(pre_base, sched_g[t["id"]][0], t["dur"], t["dem"])
    # 按最终 sched 重算 cost/co2（use 已按序累积）
    cost = 0.0
    co2 = 0.0
    for t in region_tasks:
        h, rr = sched[t["id"]]
        cost += task_cost(t, rr, h)
        co2 += task_co2(t, rr, h)
    over = int((use > cap).sum())
    if over:
        print(f"⚠️ [容量] {r} 超容量小时 {over}（基解兜底超容，Q4 允许，记录）")
    return sched, use, cost, co2, n_status, n_fallback, n_forced, time.time() - t_start


# ============================================================
# 区域级并行调度（区域间独立，滚动窗内顺序）
# 线程池：HiGHS/numpy 求解期间释放 GIL → 真并行；且无 spawn 限制
#（notebook exec 注入函数也可用，Pool 在 ipykernel 下无法定位 worker）
# ============================================================
def _rs_worker(args):
    region_tasks, r, label = args
    return r, rolling_schedule(region_tasks, r, label)


def schedule_all(by_region, label="", nproc=NPROC):
    """并行调度 6 区域（线程池；全局 PRICE/CARBON 等只读，线程安全）"""
    args = [(by_region[r], r, f"{label}|{r}") for r in REGIONS if by_region[r]]
    results = {}
    if len(args) > 1:
        with ThreadPoolExecutor(max_workers=nproc) as ex:
            futs = [ex.submit(_rs_worker, a) for a in args]
            for f in futs:
                r, res = f.result()
                results[r] = res
    else:
        for a in args:
            r, res = _rs_worker(a)
            results[r] = res
    return results


# ============================================================
# 模块二：层1 容量感知分配（F3/F4 复现）
# ============================================================
def mean_price(r):
    return power[r]["price"].mean()


def cost_of(t, r):
    return t["gh"] * t["power"] * PUE[r] * mean_price(r)


def co2_of(t, r):
    return t["gh"] * t["power"] * PUE[r] * power[r]["carbon"].mean()


def capacity_aware_assign(threshold=THRESHOLD):
    demand = {r: 0.0 for r in REGIONS}
    assign = {r: 0 for r in REGIONS}
    fail = 0
    dest = {}
    cost0 = co2_0 = cost_new = co2_new = 0.0
    for t in tasks:
        cand = sorted(t["cand"], key=lambda r: cost_of(t, r))
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
        dest[t["id"]] = best
        cost0 += cost_of(t, t["source"])
        co2_0 += co2_of(t, t["source"])
        cost_new += cost_of(t, best)
        co2_new += co2_of(t, best)
    return dest, demand, assign, fail, cost0, co2_0, cost_new, co2_new


# ============================================================
# 模块三：层2 时间维 MILP + ε-约束迭代
# ============================================================
def _dest_fp(dest):
    """目的地分配指纹（η 档间复用：各档 iter0 均为容量感知 dest0）。
    含求解参数 + 算法版本 salt：参数/算法变化自动失效 sched 缓存"""
    s = json.dumps([(k, v) for k, v in sorted(dest.items())], sort_keys=True)
    salt = f"K3v2|{MILP_TIME_LIMIT}|{WIN}|{W_EXT}|{THRESHOLD}"
    return hashlib.sha256(f"{salt}|{s}".encode()).hexdigest()[:12]


def schedule_dest(dest, label=""):
    """按 dest 指纹调度并落盘缓存（s2_sched_{fp}.pkl）。
    不同 η 档首轮共享 dest0 调度 → 只算一次，省 ~50% 调度时间"""
    fp = _dest_fp(dest)
    fp_path = BASE / "outputs" / "data" / "cache" / f"s2_sched_{fp}.pkl"
    if fp_path.exists():
        with open(fp_path, "rb") as f:
            data = pickle.load(f)
        print(f"[sched-cache] 命中 dest 指纹 {fp}，跳过调度")
        return data
    by_region = {r: [t for t in tasks if dest[t["id"]] == r] for r in REGIONS}
    results = schedule_all(by_region, label=label)
    total_cost = sum(v[2] for v in results.values())
    total_co2 = sum(v[3] for v in results.values())
    region_meta = {r: {"cost": v[2], "co2": v[3], "n": len(v[0]),
                       "status": v[4], "fallback": v[5], "forced": v[6], "dt": v[7]}
                   for r, v in results.items()}
    data = {"C": total_cost, "E": total_co2, "regions": region_meta}
    with open(fp_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[sched-cache] 已缓存 dest 指纹 {fp}")
    return data


def reassign_round(dest, demand, carb_mean=CARB_MEAN):
    """Q3 让渡一轮：按区域碳强度降序，改派候选内最低碳区（至 100% 容量）"""
    order = sorted(REGIONS, key=lambda r: -carb_mean[r])
    changed = 0
    for r in order:
        for t in tasks:
            if dest[t["id"]] != r:
                continue
            lower = [c for c in t["cand"] if carb_mean[c] < carb_mean[r] - 1e-9]
            if not lower:
                continue
            r_new = min(lower, key=lambda c: carb_mean[c])
            if demand[r_new] + t["gh"] > CAP_GH[r_new]:
                continue  # 目标区已 100%，跳过
            dest[t["id"]] = r_new
            demand[r] -= t["gh"]
            demand[r_new] += t["gh"]
            changed += 1
    return dest, demand, changed


def run_eta(eta, E0):
    """η 档：容量感知分配 → 层2 滚动窗调度 → ε 迭代（≤3 轮）。
    断点保护：每轮迭代写 s2_eta_{eta}_partial.pkl，中途崩溃后重跑自动恢复
    （已算轮次的调度结果由 s2_sched_{fp}.pkl 缓存承接，恢复后秒级续算）"""
    dest, demand, assign, fail, c0, k0, cn, kn = capacity_aware_assign()
    eta_eff = eta
    iters = []
    # 断点恢复
    pp = CACHE_DIR / f"s2_eta_{eta}_partial.pkl"
    if pp.exists():
        with open(pp, "rb") as f:
            snap = pickle.load(f)
        dest = snap["dest"]
        demand = snap["demand"]
        iters = snap["iters"]
        print(f"[断点] 恢复 η={eta}：已完成 {len(iters)} 轮（E={snap['E']/1e3:.2f}kt），从断点续算")
    for it in range(4):  # 初始 + 最多 3 轮让渡
        if it < len(iters):
            continue  # 已完成的轮次
        t_start = time.time()
        sd = schedule_dest(dest, label=f"eta={eta}")
        total_cost, total_co2 = sd["C"], sd["E"]
        region_meta = sd["regions"]
        iters.append({"iter": it, "E": total_co2, "C": total_cost,
                      "converged": total_co2 <= eta * E0 or abs(total_co2 - eta * E0) / (eta * E0) < 0.005,
                      "dt": time.time() - t_start})
        print(f"[ε η={eta}] iter{it}: 成本 {total_cost/1e6:.1f}M 元, 碳 {total_co2/1e3:.2f}kt "
              f"(上限 {eta*E0/1e3:.2f}kt) → {'收敛' if iters[-1]['converged'] else '未收敛'}")
        # 断点保存（每轮结束）
        with open(pp, "wb") as f:
            pickle.dump({"dest": dest, "demand": demand, "iters": iters,
                         "E": total_co2, "C": total_cost}, f)
        if iters[-1]["converged"]:
            break
        if it == 3:
            eta_eff = eta + 0.01  # 兜底：放宽 ε +1%（Q3 裁定）
            print(f"  ⚠️ 3 轮未收敛，ε 放宽至 {eta_eff}（记录超限偏差 {abs(total_co2-eta*E0)/(eta*E0):.1%}）")
            break
        dest, demand, changed = reassign_round(dest, demand)
        print(f"  让渡 {changed} 任务（区域碳强度降序改派低碳候选）")
        if changed == 0:
            dev = abs(total_co2 - eta * E0) / (eta * E0)
            print(f"  ⚠️ 无可让渡任务，提前终止（碳排超限偏差 {dev:.1%}，论文注明；"
                  f"不触发 ε 放宽，实际 ε 保持 {eta}）")
            break
    # 完成：清理断点文件
    if pp.exists():
        pp.unlink()
    return {"eta": eta, "eta_eff": eta_eff, "C": total_cost, "E": total_co2,
            "iters": iters, "dest": dest, "assign": assign, "fail": fail,
            "regions": region_meta}


# ============================================================
# 模块一：S1 时移基线（零迁移，Q1 裁定正式口径）
# ============================================================
def run_baseline():
    """每区域本地任务零迁移滚动窗 MILP（区域并行）→ C₀/E₀（正式基线）。
    基线调度走 schedule_dest 缓存（dest = 零迁移 source），中断后可复用"""
    dest_src = {t["id"]: t["source"] for t in tasks}
    sd = schedule_dest(dest_src, label="baseline")
    return sd["C"], sd["E"], sd["regions"]


# ============================================================
# 模块四：评价指标
# ============================================================
def eval_metrics(dest):
    """平均迁移时延（仅统计迁移任务）"""
    migrated = 0
    ttl = 0.0
    for t in tasks:
        r_new = dest[t["id"]]
        if r_new != t["source"]:
            migrated += 1
            ttl += latency.get((t["source"], r_new), np.nan)
    return ttl / migrated if migrated else 0.0, migrated


# ============================================================
# 出图（chart-generator 规范：SimHei/PDF/去饱和/线宽 1.5-2pt）
# ============================================================
def plot_reachability(fig_dir):
    """6×6 时延热力图 + 3 类型可达布尔子图（≤SLA 标记）"""
    M = np.zeros((6, 6))
    for i, s in enumerate(REGIONS):
        for j, t in enumerate(REGIONS):
            M[i, j] = latency.get((s, t), np.nan)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    ax = axes[0, 0]
    im = ax.imshow(M, cmap="YlGnBu")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                    color="white" if M[i, j] > 45 else "#1a1a1a", fontsize=8)
    ax.set_xticks(range(6)); ax.set_xticklabels([r[6:] for r in REGIONS])
    ax.set_yticks(range(6)); ax.set_yticklabels([r[6:] for r in REGIONS])
    ax.set_title("区域间网络时延 (ms)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    types = ["AITraining", "BatchInference", "RealTimeInference"]
    for k, tt in enumerate(types):
        ax = axes[(k + 1) // 2, (k + 1) % 2]
        ml = type_maxlat[tt]
        R = np.array([[latency.get((s, t), 999) <= ml for t in REGIONS] for s in REGIONS], dtype=int)
        ax.imshow(R, cmap="RdYlGn", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, "Y" if R[i, j] else "N", ha="center", va="center",
                        color="#1a5e20" if R[i, j] else "#7b1a1a", fontsize=9)
        ax.set_xticks(range(6)); ax.set_xticklabels([r[6:] for r in REGIONS])
        ax.set_yticks(range(6)); ax.set_yticklabels([r[6:] for r in REGIONS])
        ax.set_title(f"{tt}（MaxLatency={ml}ms）可达矩阵")
    fig.tight_layout()
    fig.savefig(fig_dir / "sub2-reachability.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[图] sub2-reachability.pdf")


def plot_region_load(assign, demand, fig_dir):
    """6 区域承接任务数（条形）+ 负载率（折线，双轴）"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [r[6:] for r in REGIONS]
    x = np.arange(6)
    ax.bar(x - 0.2, [assign[r] for r in REGIONS], width=0.4, color="#4c72b0", label="承接任务数")
    ax2 = ax.twinx()
    ax2.plot(x + 0.2, [demand[r] / CAP_GH[r] * 100 for r in REGIONS], "o-",
             color="#dd8452", lw=2, label="GPU-hour 负载率")
    ax2.set_ylabel("负载率 (%)")
    ax2.set_ylim(0, 105)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("承接任务数")
    ax.set_title("容量感知分配后各区域承接任务量与负载率")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "sub2-region-load.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[图] sub2-region-load.pdf")


def plot_epsilon(eta_results, C0, E0, fig_dir):
    """ε 敏感性曲线：碳排相对 E₀ vs 成本相对 C₀（3 档 + 基线参考线）"""
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [r["E"] / E0 for r in eta_results]
    ys = [r["C"] / C0 for r in eta_results]
    ax.plot(xs, ys, "o-", color="#4c72b0", lw=2)
    for r in eta_results:
        ax.annotate(f"eta={r['eta']}", (r["E"] / E0, r["C"] / C0),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.axhline(1.0, ls="--", color="#8c8c8c", lw=1.2, label="S1 时移基线成本")
    ax.axvline(1.0, ls=":", color="#8c8c8c", lw=1.2, label="S1 时移基线碳排")
    ax.set_xlabel("碳排 / E0（S1 时移基线碳排归一）")
    ax.set_ylabel("成本 / C0（S1 时移基线成本归一）")
    ax.set_title("碳排放 epsilon-约束敏感性（成本代价曲线）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "sub2-epsilon-curve.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[图] sub2-epsilon-curve.pdf")


def plot_threshold_sensitivity(fig_dir):
    """容量感知阈值敏感性（handoff §5）：80/85/90/95% → 退路数/负载率"""
    ths = [0.80, 0.85, 0.90, 0.95]
    fails = []
    loads = {}
    for th in ths:
        dest, demand, assign, fail, *_ = capacity_aware_assign(threshold=th)
        fails.append(fail)
        loads[th] = {r: demand[r] / CAP_GH[r] * 100 for r in REGIONS}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.bar([f"{t*100:.0f}%" for t in ths], fails, color="#4c72b0", width=0.5)
    ax.set_xlabel("容量感知阈值")
    ax.set_ylabel("退路任务数")
    ax.set_title("阈值 vs 退路任务数")
    ax = axes[1]
    for r in REGIONS:
        ax.plot([t * 100 for t in ths], [loads[t][r] for t in ths], "o-", label=r[6:], lw=1.5)
    ax.set_xlabel("容量感知阈值 (%)")
    ax.set_ylabel("GPU-hour 负载率 (%)")
    ax.set_title("阈值 vs 各区域负载率")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "sub2-threshold-sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[图] sub2-threshold-sensitivity.pdf")


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    t_all = time.time()
    print("=" * 60)
    print("模块一：S1 时移基线回测（零迁移，正式口径）")
    print("=" * 60)
    C0, E0, base_meta = run_baseline()
    print(f"S1 时移基线: 成本 C₀ = {C0/1e6:.1f}M 元, 碳排 E₀ = {E0/1e3:.2f}kt")
    print("（朴素口径参考：441.7M / 358.0kt，Q1 裁定降级，正式基线以本值为准）")

    print("\n" + "=" * 60)
    print("模块二：层1 容量感知分配（F3/F4 复现）")
    print("=" * 60)
    dest0, demand0, assign0, fail0, c0, k0, cn, kn = capacity_aware_assign()
    print(f"退路任务: {fail0} ({fail0/len(tasks):.2%})  [基准: 117]")
    for r in REGIONS:
        print(f"  {r}: 承接 {assign0[r]} 任务, GPU-hour {demand0[r]:,.0f} / {CAP_GH[r]:,.0f} = {demand0[r]/CAP_GH[r]:.1%}")
    print(f"朴素口径成本降幅: {(c0-cn)/c0:.1%}（基准 -16.6%）")
    print(f"朴素口径碳排降幅: {(k0-kn)/k0:.1%}（基准 -30.4%）")

    print("\n" + "=" * 60)
    print("模块三：层2 滚动窗 MILP + ε-约束迭代（η=1.0/0.9/0.8）")
    print("=" * 60)
    eta_results = []
    for eta in ETA_LEVELS:
        print(f"\n--- η={eta} ---")
        res = run_eta(eta, E0)
        eta_results.append(res)

    # 模块四评价（以 η=1.0 调度为例）
    print("\n" + "=" * 60)
    print("模块四：评价指标")
    print("=" * 60)
    r1 = eta_results[0]
    delay, n_mig = eval_metrics(r1["dest"])
    print(f"平均迁移时延: {delay:.1f} ms（迁移任务 {n_mig}/{len(tasks)} = {n_mig/len(tasks):.1%}）")

    # 新能源利用率（S2 只报告间接贡献，正式计算移交 S3/S4）
    print("\n新能源利用率（间接贡献口径，正式计算移交 S3/S4）：")
    for r in REGIONS:
        gh = sum(t["gh"] for t in tasks if r1["dest"][t["id"]] == r)
        ren = power[r]["renewable"].sum()
        print(f"  {r}: 承接 GPU-hour {gh:,.0f}, 可再生可用 {ren:,.0f} MWh")

    # 汇总表
    print("\n" + "=" * 60)
    print("迁移收益对比（S1 基线 vs S2 三档）")
    print("=" * 60)
    rows = [("S1 时移基线", C0 / 1e6, E0 / 1e3, "-")]
    for r in eta_results:
        rows.append((f"S2 η={r['eta']}", r["C"] / 1e6, r["E"] / 1e3, f"{delay:.1f}ms"))
    print(f"{'方案':<14}{'成本(M元)':<12}{'碳排(kt)':<12}{'迁移时延'}")
    for name, c, e, d in rows:
        print(f"{name:<14}{c:<12.1f}{e:<12.2f}{d}")

    # 落盘
    out = {
        "C0": C0, "E0": E0,
        "baseline_regions": base_meta,
        "assign": assign0, "demand": demand0, "fail": fail0,
        "dest_capacity_aware": dest0,
        "eta_results": eta_results,
        "delay_ms": delay, "n_migrated": n_mig,
        "params": {"WIN": WIN, "W_EXT": W_EXT, "THRESHOLD": THRESHOLD,
                   "MILP_TIME_LIMIT": MILP_TIME_LIMIT},
    }
    with open(BASE / "outputs" / "data" / "s2-results.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n[OK] 已写入 outputs/data/s2-results.pkl（总耗时 {(time.time()-t_all)/60:.1f} min）")

    # 出图
    fig_dir = BASE / "outputs" / "figures"
    plot_reachability(fig_dir)
    plot_region_load(assign0, demand0, fig_dir)
    plot_epsilon(eta_results, C0, E0, fig_dir)
    plot_threshold_sensitivity(fig_dir)

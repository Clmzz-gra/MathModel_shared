# -*- coding: utf-8 -*-
"""
⚠️ 已归档（2026-08-07）：本脚本已被增量 notebook 取代 → outputs/notebooks/verify-sub1.ipynb
   使用 run-verify-sub1-notebook.py 执行（含缓存/指纹机制，结果落盘 outputs/data/cache/ 与 s1-schedule-test.pkl）。
   保留本脚本仅供口径追溯；求解请走 notebook。

目的：
    阶段 1.2 辩论后 B 类方案偏好验证 — 子问题 S1 调度段 head-to-head：
    Alpha（时间索引 0-1 MILP 精确解）vs Beta（动态权重贪心+局部改进），
    在测试窗 2376-2399 实际到达的 538 任务上对比求解时间与利用率均衡质量

原理：
    1. 决策变量：每个自由任务（训练/批量）的开工小时 h；实时推理到达即开工（固定）
    2. Alpha MILP（scipy.milp/HiGHS）：
       - x[i,k]∈{0,1} 任务 i 是否在候选小时 k 开工；Σ_k x=1（每任务一次）
       - 逐区域逐小时 GPU 容量：base(r,t) + Σ a·x ≤ Available_GPU(r)
         （a = GPU_Demand×重叠比例，精确小数折算）
       - 目标 min(U_max−U_min)：U_max/U_min 为全局逐时利用率上/下界，
         U_max ≥ (base+Σax)/cap、U_min ≤ (base+Σax)/cap
    3. Beta 贪心：EDF 变体排序（弹性=latest−arrive−dur 升序，平局需求降序）；
       逐任务扫描候选窗，评分 = 放置后全局利用率方差 − 0.1×空余率（动态权重），
       选择使方差最小的可行窗；容量检查逐重叠小时
    4. 指标：利用率极差（全局 max−min）、利用率方差、超容量小时数
    5. 对比口径差异注明：Alpha 直接优化极差，Beta 优化方差+空余修正，非同一目标

输入数据：
    - outputs/data/c-data-cleaned.pkl（阶段 0.3 清洗后）
    - 中文指标 → 变量名映射：
      workload_trace: 任务类型→TaskType, 到达小时→ArrivalHour, GPU需求→GPU_Demand,
        预估时长(分钟)→EstimatedDuration_min, 最晚完成小时→LatestFinishHour,
        来源区域→SourceRegion
      GPU_information: 区域→Region, 可用GPU→Available_GPU

输出：
    - outputs/data/s1-schedule-test.pkl — {alpha/beta: {TaskID: 开工小时}, tasks}
      （供甘特图与阶段 2 使用）

对应论文章节：
    问题一（S1）基础算力调度模型 — 阶段 1.2 方案辩论 B 类验证
"""
import pickle
import time
import numpy as np

with open('outputs/data/c-data-cleaned.pkl', 'rb') as f:
    d = pickle.load(f)
wt = d['workload_trace']
gi = d['GPU_information'].set_index('Region')

regions = ['RegionA', 'RegionB', 'RegionC', 'RegionD', 'RegionE', 'RegionF']
cap = {r: gi.loc[r, 'Available_GPU'] for r in regions}

# ---------- 测试窗任务 ----------
test = wt[wt['ArrivalHour'] >= 2376].copy()
T0 = 2376          # 窗口起始
T_END = 2406       # 任务不得占用 2406
test = test[test['ArrivalHour'] < T_END]
print(f"测试窗任务数: {len(test)}")
print(test['TaskType'].value_counts().to_string())

# 每个任务的 (区域, 到达, 时长h, 需求, 最晚完成)
tasks = []
for _, row in test.iterrows():
    dur = row['EstimatedDuration_min'] / 60.0
    tasks.append({
        'id': row['TaskID'], 'type': row['TaskType'],
        'region': row['SourceRegion'], 'arrive': row['ArrivalHour'],
        'dur': dur, 'dem': row['GPU_Demand'],
        'latest': row['LatestFinishHour'],
    })
# 实时推理固定开工
rt_fixed = [t for t in tasks if t['type'] == 'RealTimeInference']
free = [t for t in tasks if t['type'] != 'RealTimeInference']
print(f"固定(实时): {len(rt_fixed)}  自由(训练+批量): {len(free)}")

# 小时轴 [T0, T_END)
hours = list(range(T0, T_END))
Hn = len(hours)
hidx = {h: i for i, h in enumerate(hours)}

# 实时固定占用的 GPU-hour（base 占用）
base = np.zeros((6, Hn))   # [region, hour]
for t in rt_fixed:
    r = regions.index(t['region'])
    h0 = t['arrive']
    # 按精确重叠 [h0, h0+dur) 落在各整小时的部分
    s, e = h0, h0 + t['dur']
    hi = int(np.floor(s)); hh = hidx.get(hi)
    while hh is not None and s < e and hi < T_END:
        overlap = min(e, hi + 1.0) - max(s, float(hi))
        if overlap > 0:
            base[r, hh] += t['dem'] * overlap
        s = hi + 1.0; hi = int(np.floor(s)); hh = hidx.get(hi)

# 检查 base 是否超容量（实时不可等待，必须可行）
for i, r in enumerate(regions):
    if base[i].max() > cap[r]:
        print(f"⚠️ 实时推理在 {r} 超容量: {base[i].max():.1f} > {cap[r]}")

# ============================================================
# Alpha: 时间索引 0-1 MILP（scipy.milp）
# ============================================================
from scipy.optimize import milp, LinearConstraint, Bounds

def build_alpha(free_tasks, base, cap, hours, hidx, Hn):
    """构建并求解 Alpha MILP。返回 (状态, 开工表, 指标)"""
    n = len(free_tasks)
    nvar = n + 2                    # x 变量 + U_max, U_min
    # 变量索引: x[i] 连续范围, 每个任务的候选开工时段
    cand = []                        # 每个自由任务的可开工小时列表
    xoff = []                        # 每个任务第一个 x 变量的列偏移
    col = 0
    for t in free_tasks:
        lo = max(t['arrive'], T0)
        # 候选开工窗：满足 lo ≤ h 且 h + dur ≤ min(latest, T_END)
        w = [h for h in hours if lo <= h < min(t['latest'], T_END) - t['dur'] + 1e-9]
        w = [h for h in w if h + t['dur'] <= min(t['latest'], T_END) + 1e-9]
        if not w:
            w = [lo]
        cand.append(w)
        xoff.append(col)
        col += len(w)
    nvar = col + 2
    print(f"Alpha 变量数: {nvar} (0-1 整数: {col}, 连续: 2)")

    # 目标: min U_max - U_min  → c[U_max]=+1, c[U_min]=-1, x 系数 0
    c = np.zeros(nvar)
    c[col] = 1.0      # U_max
    c[col + 1] = -1.0 # U_min

    rows = []          # (region, hour) 的线性约束系数行
    # 每任务恰好开工一次: 对每个任务 Σ x = 1
    eq_rows = []
    for i, off in enumerate(xoff):
        row = np.zeros(nvar)
        for k in range(len(cand[i])):
            row[off + k] = 1.0
        eq_rows.append(row)

    # GPU 容量 + U 上下界（逐区域逐小时）
    # 约束1: base + Σ a x ≤ cap   → Σ a x ≤ cap - base
    # 约束2: u = (base+Σa x)/cap ;  u ≥ U_min, u ≤ U_max
    #   → Σ a x - cap*U_max ≤ -base
    #   → Σ a x - cap*U_min ≥ -base
    A1 = []; ub1 = []
    A2 = []; lb2 = []; ub2 = []
    A3 = []; lb3 = []
    for ri, r in enumerate(regions):
        cr = cap[r]
        for hh in range(Hn):
            row = np.zeros(nvar)
            for i, t in enumerate(free_tasks):
                if t['region'] != r:
                    continue
                off = xoff[i]
                for k, h in enumerate(cand[i]):
                    # 任务在 [h, h+dur) 与 [hour, hour+1) 重叠
                    ov = min(h + t['dur'], hours[hh] + 1.0) - max(float(h), float(hours[hh]))
                    if ov > 0:
                        row[off + k] += t['dem'] * ov
            # 容量约束
            A1.append(row.copy())
            ub1.append(cr - base[ri, hh])
            # U 上下界
            r2 = row.copy(); r2[col] = -cr
            A2.append(r2); ub2.append(-base[ri, hh])      # Σa x - cr*U_max ≤ -base
            r3 = row.copy(); r3[col + 1] = -cr
            A3.append(r3); lb3.append(-base[ri, hh])      # Σa x - cr*U_min ≥ -base

    # 组装约束
    Aeq = np.array(eq_rows)
    beq = np.ones(len(eq_rows))
    constraints = [
        LinearConstraint(Aeq, beq, beq),
        LinearConstraint(np.array(A1), -np.inf, np.array(ub1)),
        LinearConstraint(np.array(A2), -np.inf, np.array(ub2)),
        LinearConstraint(np.array(A3), np.array(lb3), np.inf),
    ]
    integrality = np.ones(nvar)
    integrality[col:] = 0           # U 变量连续
    bounds = Bounds(np.zeros(nvar), np.ones(nvar))
    bounds.ub[col] = np.inf         # U_max 无上界
    bounds.ub[col + 1] = np.inf
    # 利用率非负（约束已隐含 U_max ≥ 利用率 ≥ 0，显式下界更安全）
    bounds.lb[col] = 0.0
    bounds.lb[col + 1] = 0.0

    t0 = time.time()
    res = milp(c=c, constraints=constraints, integrality=integrality,
               bounds=bounds, options={'time_limit': 1800, 'mip_rel_gap': 0.01})
    dt = time.time() - t0
    if res.status != 0:
        print(f"⚠️ Alpha 求解状态 {res.status}: {res.message}")
    if res.x is None:
        # 不可行/异常：无解可读，抛出明确错误而非静默崩溃
        raise RuntimeError(f"Alpha MILP 无解（status={res.status}）：{res.message}")
    x = res.x
    # 解出开工表
    schedule = {}
    for i, t in enumerate(free_tasks):
        off = xoff[i]
        sel = None
        for k, h in enumerate(cand[i]):
            if x[off + k] > 0.5:
                sel = h; break
        schedule[t['id']] = sel if sel is not None else cand[i][0]
    return res, schedule, dt, cand, xoff

t0 = time.time()
res, sched_alpha, dt_alpha, cand, xoff = build_alpha(free, base, cap, hours, hidx, Hn)
print(f"\n=== Alpha MILP: 状态={res.status} 耗时={dt_alpha:.1f}s 目标值(Umax-Umin)={res.fun:.3f} ===")

# 指标计算
def evaluate(sched_free, free_tasks, base, cap):
    """输入开工表，计算逐时利用率极差/方差（实时已含在 base 中）"""
    missing = [t['id'] for t in free_tasks if t['id'] not in sched_free]
    if missing:
        print(f"⚠️ evaluate: {len(missing)} 个任务未调度，指标将失真: {missing[:5]}...")
    use = base.copy()
    for t in free_tasks:
        h = sched_free.get(t['id'])
        if h is None:
            continue
        r = regions.index(t['region'])
        s = h
        e = h + t['dur']
        hi = int(np.floor(s))
        while s < e and hi < T_END:
            hh = hidx.get(hi)
            if hh is not None:
                ov = min(e, hi + 1.0) - max(s, float(hi))
                if ov > 0:
                    use[r, hh] += t['dem'] * ov
            s = hi + 1.0; hi = int(np.floor(s))
    util = use / np.array([cap[r] for r in regions])[:, None]
    return util, use

util_a, use_a = evaluate(sched_alpha, free, base, cap)
range_a = util_a.max() - util_a.min()
var_a = util_a.var()
print(f"Alpha: 利用率极差={range_a:.4f} 利用率方差={var_a:.6f} 超容量={int((use_a > np.array([cap[r] for r in regions])[:,None]).sum())}")

# ============================================================
# Beta: 动态权重贪心 + 局部改进
# ============================================================
def beta_greedy(free_tasks, base, cap, hours, hidx, Hn):
    use = base.copy()
    sched = {}
    # EDF 变体排序：弹性 = latest - arrive - dur 升序；平局需求降序
    order = sorted(free_tasks, key=lambda t: (t['latest'] - t['arrive'] - t['dur'], -t['dem']))
    t0 = time.time()
    for t in order:
        r = regions.index(t['region'])
        cr = cap[t['region']]
        # 候选窗口
        w = [h for h in hours if t['arrive'] <= h < min(t['latest'], T_END) - t['dur'] + 1e-9 and h + t['dur'] <= min(t['latest'], T_END) + 1e-9]
        if not w:
            w = [max(t['arrive'], T0)]
        best = None; best_score = None
        for h in w:
            # 检查容量
            ok = True
            s = h; e = h + t['dur']; hi = int(np.floor(s))
            while s < e and hi < T_END:
                hh = hidx.get(hi)
                if hh is not None:
                    ov = min(e, hi + 1.0) - max(s, float(hi))
                    if use[r, hh] + t['dem'] * ov > cr:
                        ok = False; break
                s = hi + 1.0; hi = int(np.floor(s))
            if not ok:
                continue
            # 评分: 放置后利用率方差增量 + 动态权重(空余率倒数)
            tmp = use.copy()
            s = h; e = h + t['dur']; hi = int(np.floor(s))
            while s < e and hi < T_END:
                hh = hidx.get(hi)
                if hh is not None:
                    ov = min(e, hi + 1.0) - max(s, float(hi))
                    tmp[r, hh] += t['dem'] * ov
                s = hi + 1.0; hi = int(np.floor(s))
            util = tmp / np.array([cap[rr] for rr in regions])[:, None]
            rvar = np.var(util)
            # 动态权重：时段空余率低则惩罚
            spare = 0.0
            s = h; e = h + t['dur']; hi = int(np.floor(s))
            while s < e and hi < T_END:
                hh = hidx.get(hi)
                if hh is not None:
                    spare += (cr - tmp[r, hh]) / cr
                s = hi + 1.0; hi = int(np.floor(s))
            score = rvar - 0.1 * spare
            if best_score is None or score < best_score:
                best_score = score; best = h
        if best is None:
            # 无容量可行窗（A 类验证表明测试窗容量充足，理论不触发）：
            # 防御性放最早窗并标记，避免静默产出超容量解
            best = min(w)
            print(f"⚠️ Beta 任务 {t['id']} 无可行窗，已强制放 {best}h（可能超容量）")
        sched[t['id']] = best
        # 更新 use
        s = best; e = best + t['dur']; hi = int(np.floor(s))
        while s < e and hi < T_END:
            hh = hidx.get(hi)
            if hh is not None:
                ov = min(e, hi + 1.0) - max(s, float(hi))
                use[r, hh] += t['dem'] * ov
            s = hi + 1.0; hi = int(np.floor(s))
    dt = time.time() - t0
    return sched, use, dt

sched_beta, use_beta, dt_beta = beta_greedy(free, base, cap, hours, hidx, Hn)
util_b = use_beta / np.array([cap[r] for r in regions])[:, None]
range_b = util_b.max() - util_b.min()
var_b = util_b.var()
print(f"\n=== Beta 贪心: 耗时={dt_beta:.2f}s ===")
print(f"Beta: 利用率极差={range_b:.4f} 利用率方差={var_b:.6f} 超容量={int((use_beta > np.array([cap[r] for r in regions])[:,None]).sum())}")

# ============================================================
# 对比表
# ============================================================
print("\n=== head-to-head 对比 ===")
print(f"{'指标':<20}{'Alpha MILP':<16}{'Beta 贪心':<16}")
print(f"{'求解时间(s)':<20}{dt_alpha:<16.1f}{dt_beta:<16.3f}")
print(f"{'利用率极差':<20}{range_a:<16.4f}{range_b:<16.4f}")
print(f"{'利用率方差':<20}{var_a:<16.6f}{var_b:<16.6f}")
print(f"{'超容量小时':<20}{int((use_a > np.array([cap[r] for r in regions])[:,None]).sum()):<16}{int((use_beta > np.array([cap[r] for r in regions])[:,None]).sum()):<16}")

# 保存调度结果（供甘特图/阶段2使用）
with open('outputs/data/s1-schedule-test.pkl', 'wb') as f:
    pickle.dump({
        'alpha': {**{t['id']: t['arrive'] for t in rt_fixed}, **sched_alpha},
        'beta': {**{t['id']: t['arrive'] for t in rt_fixed}, **sched_beta},
        'tasks': tasks,
    }, f)
print("\n调度结果已保存: outputs/data/s1-schedule-test.pkl")

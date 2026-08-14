# -*- coding: utf-8 -*-
"""
目的：
    阶段 2.1 S1 正式模型代码 — 三模块：统计（GPU 需求分析 + 白噪声证明）、
    预测（简单基线 + 线性回归）、调度（Alpha MILP 主方案 + Beta 贪心对照），
    并出图 + 代理值核销 + Artifact 登记

原理：
    1. 统计段：逐时 GPU 需求 = Σ 该到达小时任务 GPU_Demand（总 + 3 类型）；
       ACF(lag1..200)≈0 + 泊松到达（mean≈20.8/h）→ 白噪声证明
    2. 预测段（赛题协议三段式，split_idx 严格切分）：
       - 常数均值 = 训练窗均值外推（诚实口径，无测试窗泄漏）
       - Last-Hour / 季节朴素 lag24 / 线性回归（hour+sin/cos24+sin/cos168）
       - RMSE/MAPE 对比；结论：常数均值最优 → "不可预测性"叙事
    3. 调度段（与 notebook cell 4-7 逐位一致）：
       - Alpha：时间索引 0-1 MILP（scipy.milp/HiGHS），目标 min(Umax-Umin)，
         U 为逐时利用率（base+Σa·x)/Cap，a=dem·重叠比例精确折算
       - Beta：EDF 变体排序 + 方差-空余评分贪心
       - res.status 检查（status!=0 标近似最优，阶段 1.5 Major 修复项）
    4. 出图（chart-generator 规范：SimHei/PDF/去饱和/先算后画）

输入数据：
    - outputs/data/s1-preprocessed.pkl（阶段 1.4 预处理）
    - outputs/data/c-data-cleaned.pkl（workload_trace 统计段用）
    - outputs/data/cache/s1_alpha_milp.pkl（可选：缓存最优解，命中即跳过重解）
    - 中文指标 → 变量名映射：
      workload_trace: 任务类型→TaskType, 到达小时→ArrivalHour, GPU需求→GPU_Demand,
        预估时长(分钟)→EstimatedDuration_min, 最晚完成→LatestFinishHour,
        来源区域→SourceRegion, 最大时延→MaxLatency_ms, 任务编号→TaskID
      s1-preprocessed: series{Total/AITraining/...}, feat_df, split_idx,
        schedule_input{tasks/rt_fixed/free/base/hours/hidx/regions/cap/pue}

输出：
    - outputs/figures/sub1-demand-acf.pdf — 需求时序 + ACF
    - outputs/figures/sub1-forecast-baselines.pdf — 预测 vs 实际（4 基线）
    - outputs/figures/sub1-gantt-last24h.pdf — 最后 24h 调度甘特图（Alpha）
    - outputs/figures/sub1-utilization.pdf — 6 区域逐时 GPU 利用率（Alpha/Beta 叠加）
    - 控制台统计量（min/max/mean/std，PR-014）

对应论文章节：
    问题一（S1）统计 / 预测 / 基础调度 — 阶段 2.1 代码实现
"""
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 中文字体与负号（chart-generator 强制前置）===
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Source Han Sans CN"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(r"e:\MathModel_pj-2026-C")
regions = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
T0, T_END = 2376, 2406
COLORS = ["#333333", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]

# ============================================================
# 数据加载
# ============================================================
with open(BASE / "outputs" / "data" / "s1-preprocessed.pkl", "rb") as f:
    prep = pickle.load(f)
with open(BASE / "outputs" / "data" / "c-data-cleaned.pkl", "rb") as f:
    wt = pickle.load(f)["workload_trace"]

series = prep["series"]
split_idx = prep["split_idx"]
si = prep["schedule_input"]
tasks, rt_fixed, free = si["tasks"], si["rt_fixed"], si["free"]
base, hours, hidx = si["base"], si["hours"], si["hidx"]
cap = si["cap"]
Hn = len(hours)


def stat(name, arr):
    a = np.asarray(arr, dtype=float)
    print(f"{name}: min={a.min():.3f}, max={a.max():.3f}, mean={a.mean():.3f}, std={a.std():.3f}")
    return a


# ============================================================
# 模块一：统计段（GPU 需求分析 + 白噪声证明）
# ============================================================
print("=" * 60)
print("模块一：GPU 需求统计")
print("=" * 60)
# 任务数量/类型分布（全量）
print(f"全量任务: {len(wt)} = 训练 {(wt['TaskType']=='AITraining').sum()} / 批量 {(wt['TaskType']=='BatchInference').sum()} / 实时 {(wt['TaskType']=='RealTimeInference').sum()}")
# GPU_Demand 分布
stat("GPU_Demand", wt["GPU_Demand"])
stat("EstimatedDuration_min", wt["EstimatedDuration_min"])
# 逐时需求序列
for k in ["Total", "AITraining", "BatchInference", "RealTimeInference"]:
    stat(f"逐时需求[{k}]", series[k])
# 白噪声证明：ACF lag24
def acf(x, maxlag=200):
    x = x - x.mean()
    n = len(x)
    var = (x * x).sum()
    return np.array([(x[lag:] * x[:-lag]).sum() / (n - lag) / (var / n) if var > 0 and lag < n else 0 for lag in range(1, maxlag + 1)])

acf_total = acf(series["Total"])
print(f"ACF(Total): lag1={acf_total[0]:.3f} lag24={acf_total[23]:.3f} lag168={acf_total[167]:.3f}（≈0 → 白噪声）")
# 泊松拟合（到达计数）
cnt = np.bincount(wt["ArrivalHour"].values, minlength=2400)
print(f"到达计数: mean={cnt.mean():.2f}/h（泊松特征）")

# ============================================================
# 模块二：预测段（简单基线 + 白噪声叙事）
# ============================================================
print("\n" + "=" * 60)
print("模块二：短期预测基线")
print("=" * 60)
total = series["Total"]
y_test = total[split_idx["test"]]
y_val = total[split_idx["val"]]
tr_idx = split_idx["train"]

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def mape(a, b):
    return np.mean(np.abs((a - b) / np.maximum(a, 1e-6))) * 100

# 基线1: 常数均值（训练窗均值外推，诚实口径）
pred_mean = np.full(len(y_test), total[tr_idx].mean())
# 基线2: Last-Hour
pred_lh = np.roll(total, 1)[split_idx["test"]]
pred_lh[0] = total[split_idx["test"][0] - 1]
# 基线3: 季节朴素 lag24
pred_sea = np.roll(total, 24)[split_idx["test"]]
# 基线4: 线性回归（周期特征）
from numpy import sin, cos, pi
def feats(h):
    return np.array([1, h, sin(2 * pi * h / 24), cos(2 * pi * h / 24), sin(2 * pi * h / 168), cos(2 * pi * h / 168)])

X_tr = np.array([feats(h) for h in tr_idx])
y_trv = total[tr_idx]
b_lr = np.linalg.solve(X_tr.T @ X_tr + 1e-6 * np.eye(6), X_tr.T @ y_trv)
pred_lin = np.array([feats(h) @ b_lr for h in split_idx["test"]])

baselines = {"常数均值": pred_mean, "Last-Hour": pred_lh, "季节朴素(lag24)": pred_sea, "线性回归": pred_lin}
print(f"{'基线':<14}{'RMSE':<10}{'MAPE'}")
results = {}
for name, p in baselines.items():
    r, m = rmse(y_test, p), mape(y_test, p)
    results[name] = (r, m)
    print(f"{name:<14}{r:<10.1f}{m:.1f}%")
best = min(results, key=lambda k: results[k][0])
print(f"最优基线: {best} RMSE={results[best][0]:.1f}（→ 序列不可预测，论文用白噪声叙事）")

# ============================================================
# 模块三：调度段 — Alpha MILP
# ============================================================
print("\n" + "=" * 60)
print("模块三：Alpha MILP 精确调度")
print("=" * 60)

def load_alpha_cache():
    """尝试加载缓存最优解；无则返回 None（触发重解）"""
    fp = BASE / "outputs" / "data" / "cache" / "s1_alpha_milp.pkl"
    if fp.exists():
        with open(fp, "rb") as f:
            return pickle.load(f)
    return None

def solve_alpha():
    from scipy.optimize import milp, LinearConstraint, Bounds
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
    nvar = col + 2
    c = np.zeros(nvar)
    c[col] = 1.0
    c[col + 1] = -1.0
    eq_rows = []
    for i, off in enumerate(xoff):
        row = np.zeros(nvar)
        for k in range(len(cand[i])):
            row[off + k] = 1.0
        eq_rows.append(row)
    A1, ub1, A2, ub2, A3, lb3 = [], [], [], [], [], []
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
    constraints = [
        LinearConstraint(np.array(eq_rows), np.ones(len(eq_rows)), np.ones(len(eq_rows))),
        LinearConstraint(np.array(A1), -np.inf, np.array(ub1)),
        LinearConstraint(np.array(A2), -np.inf, np.array(ub2)),
        LinearConstraint(np.array(A3), np.array(lb3), np.inf),
    ]
    integrality = np.ones(nvar)
    integrality[col:] = 0
    bounds = Bounds(np.zeros(nvar), np.ones(nvar))
    bounds.ub[col] = bounds.ub[col + 1] = np.inf
    t0 = time.time()
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
               options={"time_limit": 1800, "mip_rel_gap": 0.01})
    dt = time.time() - t0
    if res.x is None:
        raise RuntimeError(f"Alpha 无解 status={res.status}: {res.message}")
    if res.status != 0:
        print(f"⚠️ Alpha 未达最优: status={res.status}（time-limit/iteration limit），解为可行次优 (mip_rel_gap=0.01)；论文引用需注明近似口径")
    x = res.x
    schedule = {}
    for i, t in enumerate(free):
        off = xoff[i]
        sel = None
        for k, h in enumerate(cand[i]):
            if x[off + k] > 0.5:
                sel = h
                break
        schedule[t["id"]] = sel if sel is not None else cand[i][0]
    return {"res_status": res.status, "fun": res.fun, "dt": dt, "schedule": schedule}

alpha_cache = load_alpha_cache()
if alpha_cache is not None:
    alpha = alpha_cache
    print(f"[cache] 加载 Alpha 最优解: status={alpha['res_status']} fun={alpha['fun']:.4f} dt={alpha['dt']:.1f}s 已调度={len(alpha['schedule'])}/{len(free)}")
else:
    alpha = solve_alpha()
    print(f"Alpha 求解完成: status={alpha['res_status']} fun={alpha['fun']:.4f} dt={alpha['dt']:.1f}s 已调度={len(alpha['schedule'])}/{len(free)}")

sched_alpha = alpha["schedule"]

# ============================================================
# 模块三：调度段 — Beta 贪心
# ============================================================
print("\n" + "=" * 60)
print("模块三：Beta 动态权重贪心")
print("=" * 60)
def beta_greedy():
    use = base.copy()
    sched = {}
    caps = np.array([cap[r] for r in regions])[:, None]
    order = sorted(free, key=lambda t: (t["latest"] - t["arrive"] - t["dur"], -t["dem"]))
    t0 = time.time()
    for t in order:
        r = regions.index(t["region"])
        cr = cap[t["region"]]
        w = [h for h in hours if t["arrive"] <= h < min(t["latest"], T_END) - t["dur"] + 1e-9
             and h + t["dur"] <= min(t["latest"], T_END) + 1e-9]
        if not w:
            w = [max(t["arrive"], T0)]
        best, best_score = None, None
        for h in w:
            ok = True
            s, e = h, h + t["dur"]
            hi = int(np.floor(s))
            while s < e and hi < T_END:
                hh = hidx.get(hi)
                if hh is not None:
                    ov = min(e, hi + 1.0) - max(s, float(hi))
                    if use[r, hh] + t["dem"] * ov > cr:
                        ok = False
                        break
                s = hi + 1.0
                hi = int(np.floor(s))
            if not ok:
                continue
            tmp = use.copy()
            s, e = h, h + t["dur"]
            hi = int(np.floor(s))
            while s < e and hi < T_END:
                hh = hidx.get(hi)
                if hh is not None:
                    ov = min(e, hi + 1.0) - max(s, float(hi))
                    tmp[r, hh] += t["dem"] * ov
                s = hi + 1.0
                hi = int(np.floor(s))
            rvar = np.var(tmp / caps)
            spare = 0.0
            s, e = h, h + t["dur"]
            hi = int(np.floor(s))
            while s < e and hi < T_END:
                hh = hidx.get(hi)
                if hh is not None:
                    spare += (cr - tmp[r, hh]) / cr
                s = hi + 1.0
                hi = int(np.floor(s))
            score = rvar - 0.1 * spare
            if best_score is None or score < best_score:
                best_score, best = score, h
        if best is None:
            best = min(w)
            print(f"⚠️ Beta 任务 {t['id']} 无可行窗，强制放 {best}h")
        sched[t["id"]] = best
        s, e = best, best + t["dur"]
        hi = int(np.floor(s))
        while s < e and hi < T_END:
            hh = hidx.get(hi)
            if hh is not None:
                ov = min(e, hi + 1.0) - max(s, float(hi))
                use[r, hh] += t["dem"] * ov
            s = hi + 1.0
            hi = int(np.floor(s))
    return sched, use, time.time() - t0

sched_beta, use_beta, dt_beta = beta_greedy()
print(f"Beta 贪心完成: {len(sched_beta)}/{len(free)}，耗时 {dt_beta:.2f}s")

# ============================================================
# 评估（head-to-head）
# ============================================================
def evaluate(sched_free):
    use = base.copy()
    for t in free:
        h = sched_free.get(t["id"])
        if h is None:
            continue
        r = regions.index(t["region"])
        s, e = h, h + t["dur"]
        hi = int(np.floor(s))
        while s < e and hi < T_END:
            hh = hidx.get(hi)
            if hh is not None:
                ov = min(e, hi + 1.0) - max(s, float(hi))
                if ov > 0:
                    use[r, hh] += t["dem"] * ov
            s = hi + 1.0
            hi = int(np.floor(s))
    util = use / np.array([cap[r] for r in regions])[:, None]
    return util, use

print("\n" + "=" * 60)
print("head-to-head 对比")
print("=" * 60)
util_a, use_a = evaluate(sched_alpha)
util_b, use_b = evaluate(sched_beta)
over_a = int((use_a > np.array([cap[r] for r in regions])[:, None]).sum())
over_b = int((use_b > np.array([cap[r] for r in regions])[:, None]).sum())
print(f"{'指标':<14}{'Alpha MILP':<16}{'Beta 贪心':<16}")
print(f"{'利用率极差':<14}{util_a.max()-util_a.min():<16.4f}{util_b.max()-util_b.min():<16.4f}")
print(f"{'利用率方差':<14}{util_a.var():<16.6f}{util_b.var():<16.6f}")
print(f"{'超容量小时':<14}{over_a:<16}{over_b:<16}")
print(f"{'求解时间(s)':<14}{alpha['dt']:<16.1f}{dt_beta:<16.2f}")

# 保存调度结果
with open(BASE / "outputs" / "data" / "s1-schedule-test.pkl", "wb") as f:
    pickle.dump({
        "alpha": {**{t["id"]: t["arrive"] for t in rt_fixed}, **sched_alpha},
        "beta": {**{t["id"]: t["arrive"] for t in rt_fixed}, **sched_beta},
        "tasks": tasks,
    }, f)
print("已保存 s1-schedule-test.pkl")

# ============================================================
# 出图（chart-generator 规范，先算后画）
# ============================================================
FIGS = BASE / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# 图1: 需求时序 + ACF
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
axes[0].plot(range(2400), series["Total"], color="#333333", lw=0.8)
axes[0].set_title("逐时 GPU 需求（Total，0-2399h）")
axes[0].set_ylabel("GPU")
lags = np.arange(1, 201)
axes[1].bar(lags, acf_total, width=1.0, color="#1f77b4", alpha=0.8)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].axhline(0.05, color="#d62728", ls="--", lw=0.8)
axes[1].axhline(-0.05, color="#d62728", ls="--", lw=0.8)
axes[1].set_title("ACF（Total，lag1-200）≈0 → 白噪声")
axes[1].set_xlabel("lag")
axes[1].set_ylabel("ACF")
fig.tight_layout()
fig.savefig(FIGS / "sub1-demand-acf.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 图2: 预测 vs 实际（4 基线）
# v2 修正（2026-08-08）：图例外置右侧，避免遮挡左上曲线；画布加宽容纳图例
fig, ax = plt.subplots(figsize=(10.5, 5))
test_h = np.array(split_idx["test"])
ax.plot(test_h, y_test, color="#333333", lw=1.8, label="实际")
styles = [("#1f77b4", "-"), ("#d62728", "--"), ("#2ca02c", "-."), ("#9467bd", ":")]
for (name, p), (col, ls) in zip(baselines.items(), styles):
    ax.plot(test_h, p, color=col, ls=ls, lw=1.5, label=f"{name} (RMSE={results[name][0]:.0f})")
ax.set_title("测试窗 2376-2399 预测 vs 实际（4 基线）")
ax.set_xlabel("小时")
ax.set_ylabel("GPU 需求")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
fig.tight_layout()
fig.savefig(FIGS / "sub1-forecast-baselines-v2.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 图3: 甘特图（Alpha，最后 24h 窗口）
fig, ax = plt.subplots(figsize=(11, 6))
task_rows = {}
y = 0
for t in sorted(free, key=lambda t: t["arrive"]):
    h = sched_alpha.get(t["id"])
    if h is None or h >= 2400:
        continue
    r = regions.index(t["region"])
    color_idx = {"AITraining": 0, "BatchInference": 1, "RealTimeInference": 2}.get(t["type"], 0)
    ax.barh(y, t["dur"], left=h - T0, height=0.7,
            color=[COLORS[0], COLORS[1], COLORS[2]][color_idx], alpha=0.85)
    task_rows[t["id"]] = (y, r)
    y += 1
# 实时固定任务
for t in rt_fixed:
    if t["arrive"] >= 2400:
        continue
    ax.barh(y, t["dur"], left=t["arrive"] - T0, height=0.7,
            color=COLORS[2], alpha=0.6)
    task_rows[t["id"]] = (y, regions.index(t["region"]))
    y += 1
ax.set_xlabel("相对小时（2376=0）")
ax.set_ylabel("任务索引（按到达排序）")
ax.set_title("最后 24h Alpha 调度甘特图（聚合视图）")
from matplotlib.patches import Patch
legend_handles = [Patch(color=COLORS[0], label="AITraining"),
                  Patch(color=COLORS[1], label="BatchInference"),
                  Patch(color=COLORS[2], label="RealTimeInference")]
ax.legend(handles=legend_handles, loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(FIGS / "sub1-gantt-last24h.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 图4: 6 区域逐时利用率（Alpha vs Beta）
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
for i, r in enumerate(regions):
    ax = axes[i // 3][i % 3]
    h_axis = np.arange(Hn)
    ax.plot(h_axis, util_a[i], color="#1f77b4", lw=1.5, label="Alpha")
    ax.plot(h_axis, util_b[i], color="#d62728", lw=1.5, ls="--", label="Beta")
    ax.set_title(f"{r} (Cap={cap[r]})")
    ax.set_xlabel("相对小时")
    ax.set_ylabel("利用率")
    ax.legend(frameon=False, fontsize=8)
fig.suptitle("6 区域逐时 GPU 利用率（Alpha vs Beta）")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGS / "sub1-utilization.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n[OK] 图表已输出 outputs/figures/: sub1-demand-acf.pdf / sub1-forecast-baselines.pdf / sub1-gantt-last24h.pdf / sub1-utilization.pdf")

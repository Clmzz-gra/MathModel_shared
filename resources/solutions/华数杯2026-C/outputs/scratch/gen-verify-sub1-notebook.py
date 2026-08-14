# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    生成 S1 增量 Jupyter notebook（verify-sub1.ipynb）——把阶段 1.1/1.2 的
    A 类验证 + B 类 head-to-head 拆成可缓存 cell，重复运行不重算耗时步骤

原理：
    - 用 nbformat 程序化构建 notebook，保证可复现
    - 增量策略：
      1) 数据加载结果缓存到 outputs/data/cache/（pickle），存在即跳过
      2) 耗时步骤（MILP 求解）结果缓存，指纹（任务ID哈希+参数）命中即加载
      3) 快速步骤（基线预测/贪心）每次重跑，保证可追踪
    - 求解指纹 = sha256(自由任务 (TaskID,dem,dur,arrive,latest,region) 元组)

输入数据：
    - 无（脚本生成 notebook 文件）

输出：
    - outputs/notebooks/verify-sub1.ipynb

对应论文章节：
    问题一（S1）预测与基础调度 — 阶段 1.1/1.2 验证
"""
import hashlib
import json
from pathlib import Path

from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

BASE = Path(r"e:\MathModel_pj-2026-C")
OUT_DIR = BASE / "outputs" / "notebooks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 缓存目录与工具 cell（所有 notebook 共享的"缓存即加载"模式）
CACHE_HELPERS = r'''import hashlib, json, pickle, os
from pathlib import Path

BASE = Path(r"e:\MathModel_pj-2026-C")
CACHE = BASE / "outputs" / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

def load_or_compute(name, compute_fn, fingerprint=None, force=False):
    """缓存即加载：指纹命中直接读 pickle，否则计算并缓存。
    name: 缓存文件名（不含扩展名）
    fingerprint: 可迭代对象，参与哈希；None 则视为无指纹版本
    compute_fn: 无参函数，返回要缓存的对象
    """
    key = name + ".pkl"
    # 指纹文件（JSON），与数据解耦
    fp_path = CACHE / (name + ".fp.json")
    if not force and (CACHE / key).exists():
        if fingerprint is None or (fp_path.exists() and fp_path.read_text(encoding="utf-8") == str(hashlib.sha256(json.dumps(fingerprint, ensure_ascii=False).encode()).hexdigest())):
            with open(CACHE / key, "rb") as f:
                obj = pickle.load(f)
            print(f"[cache] 加载 {name}")
            return obj
    obj = compute_fn()
    with open(CACHE / key, "wb") as f:
        pickle.dump(obj, f)
    if fingerprint is not None:
        h = hashlib.sha256(json.dumps(fingerprint, ensure_ascii=False).encode()).hexdigest()
        fp_path.write_text(h, encoding="utf-8")
    print(f"[cache] 计算并缓存 {name}")
    return obj

def clear_cache(name=None):
    """清除单个或全部缓存（force 重跑用）"""
    if name is None:
        for p in CACHE.glob("*"):
            p.unlink()
        return
    for suffix in (".pkl", ".fp.json"):
        p = CACHE / (name + suffix)
        if p.exists():
            p.unlink()'''

CELLS = [
    # ============ 0. 环境与缓存工具 ============
    new_markdown_cell("# S1 增量验证 Notebook\n\n阶段 1.1（A 类共享事实）+ 阶段 1.2（B 类 head-to-head）\n\n- 耗时步骤（数据加载、MILP 求解）自动缓存，重跑跳过\n- 快速步骤（基线、贪心）每次重算保证可追踪\n- 强制重算：改 cell 内 `force=True` 或调 `clear_cache()`"),

    new_code_cell(CACHE_HELPERS),

    # ============ 1. 数据加载（缓存） ============
    new_markdown_cell("## 1. 数据加载（缓存）\n\n从 `c-data-cleaned.pkl` 读入，序列化到缓存，重复运行不重复 IO。"),

    new_code_cell(r'''with open(BASE / "outputs" / "data" / "c-data-cleaned.pkl", "rb") as f:
    import pickle as _pkl
    d = _pkl.load(f)
wt = d['workload_trace']
gi = d['GPU_information'].set_index('Region')
regions = ['RegionA','RegionB','RegionC','RegionD','RegionE','RegionF']
cap = {r: gi.loc[r,'Available_GPU'] for r in regions}
print("workload_trace:", wt.shape, "| GPU_information:", gi.shape)'''),

    # ============ 2. 序列构建 + ACF（缓存） ============
    new_markdown_cell("## 2. 逐时 GPU 需求序列 + ACF 周期诊断（缓存）\n\nA 类共享事实 F1/F7：序列白噪声、训练占容量 80%。"),

    new_code_cell(r'''H = 2400
import numpy as np

def _build_and_acf():
    series = {}
    for t in ['AITraining','BatchInference','RealTimeInference']:
        sub = wt[wt['TaskType']==t]
        ts = np.zeros(H)
        for h, g in zip(sub['ArrivalHour'], sub['GPU_Demand']):
            if h < H: ts[h] += g
        series[t] = ts
    series['Total'] = series['AITraining'] + series['BatchInference'] + series['RealTimeInference']
    # ACF（lag 1-200）
    def acf(x, maxlag=200):
        x = x - x.mean(); n = len(x); var = (x*x).sum()
        return np.array([(x[lag:]*x[:-lag]).sum()/(n-lag)/(var/n) if var>0 and lag<n else 0 for lag in range(1, maxlag+1)])
    out = {'series': series, 'acf': {k: acf(v) for k, v in series.items()}}
    return out

res2 = load_or_compute("s1_series_acf", _build_and_acf)
acf24 = {k: v[23] for k, v in res2['acf'].items()}
print("各序列 lag24 ACF:", {k: round(v,3) for k, v in acf24.items()})
print("（≈0 说明无日周期，近乎白噪声 → 预测难）")'''),

    # ============ 3. 基线预测（快速，不缓存） ============
    new_markdown_cell("## 3. 简单基线预测（每次重算）\n\n赛题协议：0-2351 训练 / 2352-2375 调参 / 0-2375 重训 / 2376-2399 测试。"),

    new_code_cell(r'''from numpy import sin, cos, pi

total = res2['series']['Total']
def rmse(a, b): return np.sqrt(np.mean((a-b)**2))
def mape(a, b): return np.mean(np.abs((a-b)/np.maximum(a, 1e-6)))*100

y_test = total[2376:2400]
# 基线1: Last-Hour
pred_lh = np.roll(total, 1)[2376:2400]; pred_lh[0] = total[2375]
# 基线2: 季节朴素 lag24
pred_sea = np.roll(total, 24)[2376:2400]
# 基线3: 线性回归（周期特征）
def feats(h): return np.array([1, h, sin(2*pi*h/24), cos(2*pi*h/24), sin(2*pi*h/168), cos(2*pi*h/168)])
X_tr = np.array([feats(h) for h in range(2352)]); y_tr = total[:2352]
b = np.linalg.solve(X_tr.T@X_tr + 1e-6*np.eye(6), X_tr.T@y_tr)
pred_lin = np.array([feats(h)@b for h in range(2376, 2400)])
# 基线4: 常数均值
pred_mean = np.full(24, total[:2376].mean())

print(f"{'基线':<12}{'RMSE':<8}{'MAPE'}")
for name, p in [("Last-Hour", pred_lh), ("季节朴素", pred_sea), ("线性回归", pred_lin), ("常数均值", pred_mean)]:
    print(f"{name:<12}{rmse(y_test,p):<8.1f}{mape(y_test,p):.1f}%")'''),

    # ============ 4. 测试窗任务 + 实时 base（缓存） ============
    new_markdown_cell("## 4. 测试窗任务与实时 base 占用（缓存）\n\n零迁移下实时推理到达即开工，固定占用 GPU-hour。"),

    new_code_cell(r'''T0, T_END = 2376, 2406
test = wt[wt['ArrivalHour'] >= 2376].copy()
hours = list(range(T0, T_END)); Hn = len(hours)
hidx = {h: i for i, h in enumerate(hours)}

def _build_tasks():
    tasks = []
    for _, row in test.iterrows():
        tasks.append({'id': row['TaskID'], 'type': row['TaskType'],
                      'region': row['SourceRegion'], 'arrive': row['ArrivalHour'],
                      'dur': row['EstimatedDuration_min']/60.0, 'dem': row['GPU_Demand'],
                      'latest': row['LatestFinishHour']})
    rt_fixed = [t for t in tasks if t['type']=='RealTimeInference']
    free = [t for t in tasks if t['type']!='RealTimeInference']
    base = np.zeros((6, Hn))
    for t in rt_fixed:
        r = regions.index(t['region']); h0 = t['arrive']
        s, e = h0, h0 + t['dur']; hi = int(np.floor(s)); hh = hidx.get(hi)
        while hh is not None and s < e and hi < T_END:
            ov = min(e, hi+1.0) - max(s, float(hi))
            if ov > 0: base[r, hh] += t['dem'] * ov
            s = hi+1.0; hi = int(np.floor(s)); hh = hidx.get(hi)
    return {'tasks': tasks, 'rt_fixed': rt_fixed, 'free': free, 'base': base}

r4 = load_or_compute("s1_test_tasks", _build_tasks)
tasks, rt_fixed, free, base = r4['tasks'], r4['rt_fixed'], r4['free'], r4['base']
print(f"测试窗任务 {len(tasks)}: 固定(实时) {len(rt_fixed)} / 自由(训练+批量) {len(free)}")
# 实时超容量检查
for i, r in enumerate(regions):
    if base[i].max() > cap[r]:
        print(f"⚠️ {r} 实时超容量 {base[i].max():.1f} > {cap[r]}")'''),

    # ============ 5. Alpha MILP（缓存 + 指纹） ============
    new_markdown_cell("## 5. Alpha MILP 精确调度（**缓存**：求解一次，指纹命中即加载）\n\n求解 >10min，是唯一需要持久缓存的步骤。指纹 = 自由任务属性序列哈希。"),

    new_code_cell(r'''from scipy.optimize import milp, LinearConstraint, Bounds

def _solve_alpha():
    n = len(free)
    cand = []; xoff = []; col = 0
    for t in free:
        lo = max(t['arrive'], T0)
        w = [h for h in hours if lo <= h < min(t['latest'], T_END) - t['dur'] + 1e-9 and h + t['dur'] <= min(t['latest'], T_END) + 1e-9]
        if not w: w = [lo]
        cand.append(w); xoff.append(col); col += len(w)
    nvar = col + 2
    c = np.zeros(nvar); c[col] = 1.0; c[col+1] = -1.0
    eq_rows = []
    for i, off in enumerate(xoff):
        row = np.zeros(nvar)
        for k in range(len(cand[i])): row[off+k] = 1.0
        eq_rows.append(row)
    A1, ub1 = [], []; A2, ub2 = [], []; A3, lb3 = [], []
    for ri, r in enumerate(regions):
        cr = cap[r]
        for hh in range(Hn):
            row = np.zeros(nvar)
            for i, t in enumerate(free):
                if t['region'] != r: continue
                off = xoff[i]
                for k, h in enumerate(cand[i]):
                    ov = min(h + t['dur'], hours[hh]+1.0) - max(float(h), float(hours[hh]))
                    if ov > 0: row[off+k] += t['dem'] * ov
            A1.append(row.copy()); ub1.append(cr - base[ri, hh])
            r2 = row.copy(); r2[col] = -cr; A2.append(r2); ub2.append(-base[ri, hh])
            r3 = row.copy(); r3[col+1] = -cr; A3.append(r3); lb3.append(-base[ri, hh])
    constraints = [
        LinearConstraint(np.array(eq_rows), np.ones(len(eq_rows)), np.ones(len(eq_rows))),
        LinearConstraint(np.array(A1), -np.inf, np.array(ub1)),
        LinearConstraint(np.array(A2), -np.inf, np.array(ub2)),
        LinearConstraint(np.array(A3), np.array(lb3), np.inf),
    ]
    integrality = np.ones(nvar); integrality[col:] = 0
    bounds = Bounds(np.zeros(nvar), np.ones(nvar))
    bounds.ub[col] = bounds.ub[col+1] = np.inf
    import time
    t0 = time.time()
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
               options={'time_limit': 1800, 'mip_rel_gap': 0.01})
    dt = time.time() - t0
    if res.x is None:
        raise RuntimeError(f"Alpha 无解 status={res.status}: {res.message}")
    if res.status != 0:
        print(f"⚠️ Alpha 未达最优: status={res.status} (time-limit/iteration limit)，解为可行次优 (mip_rel_gap=0.01)；论文引用需注明近似口径")
    x = res.x
    schedule = {}
    for i, t in enumerate(free):
        off = xoff[i]; sel = None
        for k, h in enumerate(cand[i]):
            if x[off+k] > 0.5: sel = h; break
        schedule[t['id']] = sel if sel is not None else cand[i][0]
    return {'res_status': res.status, 'fun': res.fun, 'dt': dt, 'schedule': schedule}

fp = sorted([(t['id'], t['dem'], t['dur'], t['arrive'], t['latest'], t['region']) for t in free])
r5 = load_or_compute("s1_alpha_milp", _solve_alpha, fingerprint=fp)
print(f"Alpha: status={r5['res_status']} 目标(Umax-Umin)={r5['fun']:.4f} 耗时={r5['dt']:.1f}s 已调度={len(r5['schedule'])}/{len(free)}")'''),

    # ============ 6. Beta 贪心（快速，不缓存） ============
    new_markdown_cell("## 6. Beta 动态权重贪心（每次重算）\n\nEDF 变体排序 + 方差评分，秒级完成。"),

    new_code_cell(r'''def beta_greedy():
    use = base.copy(); sched = {}
    order = sorted(free, key=lambda t: (t['latest'] - t['arrive'] - t['dur'], -t['dem']))
    caps = np.array([cap[r] for r in regions])[:, None]
    for t in order:
        r = regions.index(t['region']); cr = cap[t['region']]
        w = [h for h in hours if t['arrive'] <= h < min(t['latest'], T_END) - t['dur'] + 1e-9 and h + t['dur'] <= min(t['latest'], T_END) + 1e-9]
        if not w: w = [max(t['arrive'], T0)]
        best = None; best_score = None
        for h in w:
            ok = True; s = h; e = h + t['dur']; hi = int(np.floor(s))
            while s < e and hi < T_END:
                hh = hidx.get(hi)
                if hh is not None:
                    ov = min(e, hi+1.0) - max(s, float(hi))
                    if use[r, hh] + t['dem']*ov > cr: ok = False; break
                s = hi+1.0; hi = int(np.floor(s))
            if not ok: continue
            tmp = use.copy(); s = h; e = h + t['dur']; hi = int(np.floor(s))
            while s < e and hi < T_END:
                hh = hidx.get(hi)
                if hh is not None:
                    ov = min(e, hi+1.0) - max(s, float(hi)); tmp[r, hh] += t['dem']*ov
                s = hi+1.0; hi = int(np.floor(s))
            rvar = np.var(tmp / caps)
            spare = 0.0; s = h; e = h + t['dur']; hi = int(np.floor(s))
            while s < e and hi < T_END:
                hh = hidx.get(hi)
                if hh is not None: spare += (cr - tmp[r, hh]) / cr
                s = hi+1.0; hi = int(np.floor(s))
            score = rvar - 0.1*spare
            if best_score is None or score < best_score: best_score, best = score, h
        if best is None:
            best = min(w)
            print(f"⚠️ {t['id']} 无可行窗，强制放 {best}h")
        sched[t['id']] = best
        s = best; e = best + t['dur']; hi = int(np.floor(s))
        while s < e and hi < T_END:
            hh = hidx.get(hi)
            if hh is not None:
                ov = min(e, hi+1.0) - max(s, float(hi)); use[r, hh] += t['dem']*ov
            s = hi+1.0; hi = int(np.floor(s))
    return sched, use

sched_beta, use_beta = beta_greedy()
print(f"Beta 贪心完成: {len(sched_beta)} 任务已调度")'''),

    # ============ 7. 对比评估（每次重算） ============
    new_markdown_cell("## 7. head-to-head 对比\n\n指标：利用率极差/方差/超容量小时。**注意**：Alpha 优化极差，Beta 优化方差+空余，口径不同。"),

    new_code_cell(r'''def evaluate(sched_free, free_tasks):
    missing = [t['id'] for t in free_tasks if t['id'] not in sched_free]
    if missing: print(f"⚠️ {len(missing)} 任务未调度: {missing[:5]}")
    use = base.copy()
    for t in free_tasks:
        h = sched_free.get(t['id'])
        if h is None: continue
        r = regions.index(t['region']); s = h; e = h + t['dur']; hi = int(np.floor(s))
        while s < e and hi < T_END:
            hh = hidx.get(hi)
            if hh is not None:
                ov = min(e, hi+1.0) - max(s, float(hi))
                if ov > 0: use[r, hh] += t['dem']*ov
            s = hi+1.0; hi = int(np.floor(s))
    return use / np.array([cap[r] for r in regions])[:, None], use

def summarize(name, sched):
    util, use = evaluate(sched, free)
    over = int((use > np.array([cap[r] for r in regions])[:, None]).sum())
    print(f"{name:<12} 极差={util.max()-util.min():.4f}  方差={util.var():.6f}  超容量h={over}")
    return util

util_a = summarize("Alpha", r5['schedule'])
util_b = summarize("Beta", sched_beta)

# 调度结果持久化（甘特图/阶段2用）
with open(BASE / "outputs" / "data" / "s1-schedule-test.pkl", "wb") as f:
    import pickle
    pickle.dump({
        'alpha': {**{t['id']: t['arrive'] for t in rt_fixed}, **r5['schedule']},
        'beta':  {**{t['id']: t['arrive'] for t in rt_fixed}, **sched_beta},
        'tasks': tasks,
    }, f)
print("已保存 s1-schedule-test.pkl")'''),
]

nb = new_notebook(cells=CELLS)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

out_path = OUT_DIR / "verify-sub1.ipynb"
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] notebook -> {out_path}")

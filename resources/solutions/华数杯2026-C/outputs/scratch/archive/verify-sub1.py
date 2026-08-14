# -*- coding: utf-8 -*-
"""
⚠️ 已归档（2026-08-07）：A 类验证结果已落盘 solution/model-notes/verify-sub1-20260807.md，
   本脚本保留仅供口径追溯；后续预测/调度分析请走 outputs/notebooks/verify-sub1.ipynb。

S1 A类共享事实验证（阶段 1.1）
第一步：简单基线先行 —— 建立预测性能下界 + 序列周期诊断 + 调度可行性检查
输出：solution/model-notes/verify-sub1-20260807.md 的数据基础
"""
import pickle
import numpy as np
import pandas as pd

# ---------- 加载 ----------
with open('outputs/data/c-data-cleaned.pkl', 'rb') as f:
    d = pickle.load(f)
wt = d['workload_trace']
gi = d['GPU_information'].set_index('Region')

regions = ['RegionA', 'RegionB', 'RegionC', 'RegionD', 'RegionE', 'RegionF']
H = 2400

# ============================================================
# 1. 构建逐时 GPU 需求序列（总 + 分类型）
# ============================================================
def build_demand_series(wt, H=2400, by_type=False):
    """按到达小时聚合 GPU_Demand（决策点2：总序列 + 分类型序列）"""
    if not by_type:
        ts = np.zeros(H)
        for h, g in zip(wt['ArrivalHour'], wt['GPU_Demand']):
            if h < H:
                ts[h] += g
        return {'Total': ts}
    series = {}
    for t in wt['TaskType'].unique():
        sub = wt[wt['TaskType'] == t]
        ts = np.zeros(H)
        for h, g in zip(sub['ArrivalHour'], sub['GPU_Demand']):
            if h < H:
                ts[h] += g
        series[t] = ts
    series['Total'] = series.get('AITraining', np.zeros(H)) + series.get('BatchInference', np.zeros(H)) + series.get('RealTimeInference', np.zeros(H))
    return series

series = build_demand_series(wt, H, by_type=True)
print("=== 序列统计 ===")
for k, v in series.items():
    print(f"{k}: mean={v.mean():.1f} std={v.std():.1f} max={v.max():.0f} 零值占比={(v==0).mean():.2%}")

# ============================================================
# 2. 周期诊断：ACF 峰值（24h / 168h 周期检测）
# ============================================================
def acf(x, maxlag):
    x = x - x.mean()
    n = len(x)
    out = []
    var = (x * x).sum()
    for lag in range(1, maxlag + 1):
        c = (x[lag:] * x[:-lag]).sum() / (n - lag) if (n - lag) > 0 else 0
        out.append(c / (var / n) if var > 0 else 0)
    return np.array(out)

total = series['Total']
acf_vals = acf(total, maxlag=200)
print("\n=== ACF 周期诊断 (Total 序列, lag1-200) ===")
for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
    if lag <= len(acf_vals):
        print(f"lag={lag}: ACF={acf_vals[lag-1]:.3f}")
peak_lags = np.argsort(acf_vals)[-8:] + 1
print("ACF 前 8 强峰值 lag:", sorted(peak_lags.tolist()))

# ============================================================
# 3. 简单基线预测（A类第一步：性能下界）
#    赛题协议：0-2351 训练 / 2352-2375 调参 / 0-2375 重训 / 2376-2399 测试
#    基线：Last-Hour（朴素）、季节朴素（前24h同一小时）、线性回归(小时数+周期哑变量)
# ============================================================
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))) * 100

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

train_end = 2352      # 0-2351 训练
val_end = 2376        # 2352-2375 调参
test_start = 2376     # 2376-2399 测试
y_test = total[test_start:]

print("\n=== 简单基线预测（测试窗 2376-2399, 24点）===")

# 基线1: Last-Hour 朴素
pred_lh = np.roll(total, 1)[test_start:]
pred_lh[:1] = total[test_start-1]
print(f"Last-Hour:      RMSE={rmse(y_test, pred_lh):.1f}  MAPE={mape(y_test, pred_lh):.1f}%")

# 基线2: 季节朴素（同小时前一天, 即 lag=24）
pred_sea = np.roll(total, 24)[test_start:]
print(f"季节朴素(lag24): RMSE={rmse(y_test, pred_sea):.1f}  MAPE={mape(y_test, pred_sea):.1f}%")

# 基线3: 线性回归（小时数 + 24h周期 sin/cos + 168h周期 sin/cos + 类型占比不适用）
from numpy import sin, cos, pi
def feats(h):
    return np.array([1, h, sin(2*pi*h/24), cos(2*pi*h/24), sin(2*pi*h/168), cos(2*pi*h/168)])

X_tr = np.array([feats(h) for h in range(train_end)])
y_tr = total[:train_end]
# 简单线性最小二乘
Xtx = X_tr.T @ X_tr
beta = np.linalg.solve(Xtx + 1e-6*np.eye(Xtx.shape[0]), X_tr.T @ y_tr)
pred_lin = np.array([feats(h) @ beta for h in range(test_start, 2400)])
print(f"线性回归:       RMSE={rmse(y_test, pred_lin):.1f}  MAPE={mape(y_test, pred_lin):.1f}%")
print(f"  系数(截距,h,24sin,24cos,168sin,168cos): {np.round(beta,2).tolist()}")

# ============================================================
# 4. 调度可行性检查（零迁移假设下）
#    实时推理到达即开工：逐时叠加 vs 容量
#    训练/批量：GPU-hour 总量 vs 容量（时间维可时移）
# ============================================================
print("\n=== 调度可行性（零迁移, 决策点1=选项A）===")
# 4a. 实时推理（不可等待）
rt = wt[wt['TaskType'] == 'RealTimeInference']
rt_agg = np.zeros((H, 6))
for _, row in rt.iterrows():
    h0 = row['ArrivalHour']
    dur = row['EstimatedDuration_min'] / 60.0
    h1 = min(H, int(np.floor(h0 + dur)))
    r = regions.index(row['SourceRegion'])
    rt_agg[h0:h1, r] += row['GPU_Demand']
print("实时推理逐时叠加 vs Available_GPU:")
for i, r in enumerate(regions):
    cap = gi.loc[r, 'Available_GPU']
    u = rt_agg[:, i] / cap
    print(f"  {r}: max_util={u.max():.1%}  超容量小时={int((u>1).sum())}")

# 4b. 训练/批量 GPU-hour 总量
print("训练+批量 GPU-hour 需求/容量:")
for r in regions:
    sub = wt[(wt['SourceRegion'] == r) & (wt['TaskType'] != 'RealTimeInference')]
    dem = (sub['GPU_Demand'] * sub['EstimatedDuration_min'] / 60).sum()
    cap = gi.loc[r, 'Available_GPU'] * H
    print(f"  {r}: {dem/cap:.1%}  ({dem:,.0f}/{cap:,.0f})")

# 4c. 最后24h调度窗（2376-2399 到达任务）
test_wt = wt[wt['ArrivalHour'] >= 2376]
print(f"\n=== 测试窗任务 (2376-2399): {len(test_wt)} 个 ===")
print(test_wt['TaskType'].value_counts().to_string())
# 实时推理在测试窗的叠加 vs 容量
rt_test = test_wt[test_wt['TaskType'] == 'RealTimeInference']
for i, r in enumerate(regions):
    sub = rt_test[rt_test['SourceRegion'] == r]
    dem = (sub['GPU_Demand'] * sub['EstimatedDuration_min'] / 60).sum()
    print(f"  {r} 测试窗实时推理 GPU-hour: {dem:.0f}")

# ============================================================
# 5. IT 功率 / 设施功率校核（说明 3：调度结果须满足功率约束）
#    power_mapping: AITraining=0.16, BatchInference=0.10, RealTimeInference=0.08 MW/等效GPU
#    校核逻辑：GPU 容量约束满足 ⟹ 功率约束自动满足的数学证明
#    AI_IT_Load(r,t) = Σ GPU_Demand×Overlap×GPU_Power ≤ GPU_running×0.16 ≤ Available_GPU×0.16
# ============================================================
print("\n=== 功率校核（power_mapping × GPU 容量约束的自动满足性）===")
pm = d['power_mapping'].set_index('TaskType')['GPU_Power_MW_per_EquivalentGPU']
print("功率映射:", {k: float(v) for k, v in pm.items()})
print()
print("| 区域 | Available_GPU | 最坏AI_IT_Load上界(MW) | Max_IT_Power_MW | 余量倍数 | 最坏设施Load上界(MW) | Max_Facility(MW) | 余量倍数 |")
print("|------|--------------|----------------------|-----------------|---------|--------------------|------------------|---------|")
for i, r in enumerate(regions):
    av = gi.loc[r, 'Available_GPU']
    pue = gi.loc[r, 'PUE']
    mip = gi.loc[r, 'Max_IT_Power_MW']
    mfp = gi.loc[r, 'Max_Facility_Power_MW']
    worst_it = av * pm['AITraining']          # 全部GPU跑训练=最高功率
    worst_fac = worst_it * pue
    print(f"| {r} | {av} | {worst_it:.1f} | {mip} | {mip/worst_it:.1f}× | {worst_fac:.1f} | {mfp} | {mfp/worst_fac:.1f}× |")

# 实际校核：实时推理（不可等）+ 训练/批量按到达即时开工（最紧基线）逐时功率
print("\n实际逐时功率校核（实时到达即开工 + 训练/批量到达即开工 = 最紧时间基线）:")
AI_power = np.zeros((H, 6))
for _, row in wt.iterrows():
    h0 = row['ArrivalHour']
    if h0 >= H:
        continue
    dur = row['EstimatedDuration_min'] / 60.0
    h1 = min(H, int(np.floor(h0 + dur)))
    r = regions.index(row['SourceRegion'])
    AI_power[h0:h1, r] += row['GPU_Demand'] * pm[row['TaskType']]
for i, r in enumerate(regions):
    p = AI_power[:, i]
    mip = gi.loc[r, 'Max_IT_Power_MW']
    mfp = gi.loc[r, 'Max_Facility_Power_MW']
    fac = p * gi.loc[r, 'PUE']
    print(f"  {r}: IT功率峰值={p.max():.1f}MW (限{mip}) 超限h={int((p>mip).sum())} | 设施功率峰值={fac.max():.1f}MW (限{mfp}) 超限h={int((fac>mfp).sum())}")

# ============================================================
# 6. 网络时延校核（说明 3：调度结果须满足网络时延约束）
#    零迁移假设下：任务留在来源区域 → 时延=同区5ms
# ============================================================
print("\n=== 时延校核（零迁移 = 同区 5ms）===")
nl = d['network_latency']
for t in wt['TaskType'].unique():
    sub = wt[wt['TaskType'] == t]
    print(f"  {t}: MaxLatency_ms 范围 [{sub['MaxLatency_ms'].min()}, {sub['MaxLatency_ms'].max()}] 同区5ms是否全满足: {(sub['MaxLatency_ms'] >= 5).all()}")

print("\n=== A类验证完成 ===")

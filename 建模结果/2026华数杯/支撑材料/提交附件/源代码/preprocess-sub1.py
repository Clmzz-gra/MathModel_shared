# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 1.4 数据预处理 — 子问题 S1：构建预测目标序列（逐时 GPU 需求）、
    预测特征集（周期 sin/cos + 滞后特征）、赛题协议切分、调度输入
    （测试窗任务结构 + 候选窗口 + 实时 base 占用），输出统一预处理缓存

原理：
    P1. 逐时 GPU 需求 = Σ 该到达小时任务的 GPU_Demand（总 + 分类型）
    P2. 特征：hour + 24h/168h 周期 sin/cos + 滞后 lag1/lag24/lag168 + ma24；
        滞后特征需前 168h 预热 → 有效样本从 h=168 起（工程截断，见 code-review I3）
    P3. 赛题协议切分：0-2351 训练 / 2352-2375 调参 / 0-2375 重训 / 2376-2399 测试
        （有效样本从 168 起，受 lag168 限制）
    P4. 调度输入：测试窗（ArrivalHour ≥ 2376）任务标准结构；
        实时推理到达即开工固定占用 base（跨小时重叠精确折算）；
        自由任务候选窗 = {h: arrive ≤ h 且 h+dur ≤ min(latest,2406)}

输入数据：
    - outputs/data/c-data-cleaned.pkl（阶段 0.3 清洗后）
    - 中文指标 → 变量名映射：
      workload_trace: 任务类型→TaskType, 到达小时→ArrivalHour, GPU需求→GPU_Demand,
        预估时长(分钟)→EstimatedDuration_min, 最晚完成小时→LatestFinishHour,
        最大时延(ms)→MaxLatency_ms, 来源区域→SourceRegion, 任务编号→TaskID
      GPU_information: 区域→Region, 可用GPU→Available_GPU, 能效→PUE,
        IT功率上限(MW)→Max_IT_Power_MW, 设施功率上限(MW)→Max_Facility_Power_MW
      power_mapping: 任务类型→TaskType, 每等效GPU功率(MW)→GPU_Power_MW_per_EquivalentGPU

输出：
    - outputs/data/s1-preprocessed.pkl — {series, feat_df, split_idx, schedule_input,
      power_mapping}（供预测与调度共用）

对应论文章节：
    问题一（S1）预测模型构建与基础算力调度 — 阶段 1.4 数据预处理
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r"e:\MathModel_pj-2026-C")

with open(BASE / "outputs" / "data" / "c-data-cleaned.pkl", "rb") as f:
    d = pickle.load(f)
wt = d['workload_trace']
gi = d['GPU_information'].set_index('Region')

regions = ['RegionA', 'RegionB', 'RegionC', 'RegionD', 'RegionE', 'RegionF']
H = 2400
T0, T_END = 2376, 2406

# ============================================================
# P1. 逐时 GPU 需求聚合（预测目标）
# ============================================================
def build_demand_series(wt, H=2400):
    series = {}
    for t in ['AITraining', 'BatchInference', 'RealTimeInference']:
        sub = wt[wt['TaskType'] == t]
        ts = np.zeros(H)
        for h, g in zip(sub['ArrivalHour'], sub['GPU_Demand']):
            if h < H:
                ts[h] += g
        series[t] = ts
    series['Total'] = series['AITraining'] + series['BatchInference'] + series['RealTimeInference']
    return series

series = build_demand_series(wt, H)

# ============================================================
# P2. 特征工程（预测用）
#   特征集：小时数 + 24h/168h 周期 sin/cos + 滞后特征（lag1/lag24/lag168）
#   滞后特征需前 168h 预热 → 有效样本从 h=168 起
# ============================================================
from numpy import sin, cos, pi

def make_features(total, H=2400, lag_pre=168):
    feats = []
    for h in range(H):
        f = {
            'hour': h,
            'sin24': sin(2 * pi * h / 24), 'cos24': cos(2 * pi * h / 24),
            'sin168': sin(2 * pi * h / 168), 'cos168': cos(2 * pi * h / 168),
        }
        if h >= lag_pre:
            f['lag1'] = total[h - 1]
            f['lag24'] = total[h - 24]
            f['lag168'] = total[h - 168]
            f['ma24'] = total[h - 24:h].mean()   # 前24h滑动均值
        else:
            f['lag1'] = f['lag24'] = f['lag168'] = f['ma24'] = np.nan
        feats.append(f)
    df = pd.DataFrame(feats)
    df['y_total'] = total
    return df

feat_df = make_features(series['Total'])
# 训练起点 168（lag168 可用）
train_mask = (feat_df['hour'] >= 168) & (feat_df['hour'] < 2352)
val_mask = (feat_df['hour'] >= 2352) & (feat_df['hour'] < 2376)
test_mask = (feat_df['hour'] >= 2376) & (feat_df['hour'] < 2400)
feat_df['split'] = np.where(train_mask, 'train',
                  np.where(val_mask, 'val',
                  np.where(test_mask, 'test', 'pre')))
# 重训集：0-2375（有效样本 168-2375）
retrain_mask = (feat_df['hour'] >= 168) & (feat_df['hour'] < 2376)

print("=== P2 特征集 ===")
print(f"样本总数: {len(feat_df)} | 有效(train+val+test): {int(train_mask.sum()+val_mask.sum()+test_mask.sum())}")
print(f"train {int(train_mask.sum())} / val {int(val_mask.sum())} / test {int(test_mask.sum())}")

# ============================================================
# P3. 赛题协议切分（记录索引供各基线/模型使用）
# ============================================================
split_idx = {
    'train': feat_df.index[train_mask].tolist(),
    'val': feat_df.index[val_mask].tolist(),
    'test': feat_df.index[test_mask].tolist(),
    'retrain': feat_df.index[retrain_mask].tolist(),
}
print("=== P3 切分 ===")
for k, v in split_idx.items():
    print(f"{k}: {len(v)} 样本 [{min(v)},{max(v)}]")

# ============================================================
# P4. 调度输入（测试窗 2376-2399 到达任务）
# ============================================================
test = wt[wt['ArrivalHour'] >= T0].copy()
hours = list(range(T0, T_END))
Hn = len(hours)
hidx = {h: i for i, h in enumerate(hours)}

tasks = []
for _, row in test.iterrows():
    tasks.append({
        'id': row['TaskID'], 'type': row['TaskType'],
        'region': row['SourceRegion'], 'arrive': row['ArrivalHour'],
        'dur': row['EstimatedDuration_min'] / 60.0, 'dem': row['GPU_Demand'],
        'latest': row['LatestFinishHour'], 'latency': row['MaxLatency_ms'],
    })
rt_fixed = [t for t in tasks if t['type'] == 'RealTimeInference']
free = [t for t in tasks if t['type'] != 'RealTimeInference']

# 实时 base 占用（到达即开工，跨小时重叠折算）
base = np.zeros((6, Hn))
for t in rt_fixed:
    r = regions.index(t['region']); h0 = t['arrive']
    s, e = h0, h0 + t['dur']; hi = int(np.floor(s)); hh = hidx.get(hi)
    while hh is not None and s < e and hi < T_END:
        ov = min(e, hi + 1.0) - max(s, float(hi))
        if ov > 0:
            base[r, hh] += t['dem'] * ov
        s = hi + 1.0; hi = int(np.floor(s)); hh = hidx.get(hi)

# 自由任务候选窗口（与 notebook cell 5 口径一致）
for t in free:
    w = [h for h in hours if max(t['arrive'], T0) <= h < min(t['latest'], T_END) - t['dur'] + 1e-9
         and h + t['dur'] <= min(t['latest'], T_END) + 1e-9]
    t['cand'] = w if w else [max(t['arrive'], T0)]

# 实时超容量检查
print("\n=== P4 调度输入 ===")
print(f"测试窗任务: {len(tasks)} (固定实时 {len(rt_fixed)} / 自由 {len(free)})")
for i, r in enumerate(regions):
    cap = gi.loc[r, 'Available_GPU']
    if base[i].max() > cap:
        print(f"  ⚠️ {r} 实时超容量 {base[i].max():.1f} > {cap}")
    else:
        print(f"  {r}: 实时峰值占用 {base[i].max():.1f}/{cap}")

# ============================================================
# 落盘
# ============================================================
out = {
    'series': series,
    'feat_df': feat_df,
    'split_idx': split_idx,
    'schedule_input': {
        'tasks': tasks, 'rt_fixed': rt_fixed, 'free': free,
        'base': base, 'hours': hours, 'hidx': hidx,
        'regions': regions,
        'cap': {r: gi.loc[r, 'Available_GPU'] for r in regions},
        'pue': {r: gi.loc[r, 'PUE'] for r in regions},
        'max_it_power': {r: gi.loc[r, 'Max_IT_Power_MW'] for r in regions},
        'max_facility_power': {r: gi.loc[r, 'Max_Facility_Power_MW'] for r in regions},
    },
    'power_mapping': d['power_mapping'].set_index('TaskType')['GPU_Power_MW_per_EquivalentGPU'].to_dict(),
}
with open(BASE / "outputs" / "data" / "s1-preprocessed.pkl", "wb") as f:
    pickle.dump(out, f)
print("\n[OK] 已写入 outputs/data/s1-preprocessed.pkl")

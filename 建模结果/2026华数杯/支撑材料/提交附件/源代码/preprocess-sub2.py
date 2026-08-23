# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 1.4 子问题 S2 数据预处理 — 构建 s2-preprocessed.pkl，
    为碳感知任务调度模型（ε-约束 + 容量感知分配 + 区域分解 MILP）提供
    任务候选目的地、区域电力参数、S1 基线回测输入

原理：
    1. 可行域裁剪：任务可迁移目标 = {r: NetworkLatency(s→r) ≤ MaxLatency_ms}
       （实时 20ms / 批量 80ms / 训练 150ms），生成 6×6 可达矩阵 reach_by_type
       （注：E↔F 实时推理 18ms ≤ 20ms 可互迁，建模已确认允许，2026-08-08）
    2. 任务→候选目的地：每任务记录源区域 + 可行目标集合 + GPU-hour/时延/功率预算，
       供 MILP 决策变量直接消费；GPU-hour = GPU_Demand × EstimatedDuration_min / 60
    3. 区域电力参数表：电价/售电价/碳强度/可再生/容量/功率上限，逐小时（0-2399）
    4. S1 基线：零迁移全时域回测输入（每区域本地任务列表 + GPU 容量），
       供 S1 框架复用时移回测

输入数据：
    - outputs/data/c-data-cleaned.pkl（阶段 0.3 清洗后）
    - 中文指标 → 变量名映射：
      workload_trace: 任务编号→TaskID, 任务类型→TaskType, 来源区域→SourceRegion,
        GPU需求→GPU_Demand, 预估时长(分钟)→EstimatedDuration_min, 到达小时→ArrivalHour,
        最晚完成→LatestFinishHour, 最大时延→MaxLatency_ms
      GPU_information: 区域→Region, 可用GPU→Available_GPU, 能效→PUE,
        最大IT功率→Max_IT_Power_MW, 最大设施功率→Max_Facility_Power_MW
      network_latency: 源→FromRegion, 目标→ToRegion, 时延(ms)→NetworkLatency_ms
      region_time_data: 区域→Region, 小时→Hour, 电价→ElectricityPrice_CNY_per_MWh,
        售电价→SellPrice_CNY_per_MWh, 碳强度→CarbonIntensity_tCO2_per_MWh,
        可再生可用→AvailableRenewable_MW
      power_mapping: 任务类型→TaskType, 每等效GPU功率(MW)→GPU_Power_MW_per_EquivalentGPU

输出：
    - outputs/data/s2-preprocessed.pkl — 键：
      tasks（任务→候选目的地+预算）/ reach_by_type（按类型 6×6 可达矩阵）/
      power（区域逐小时电力参数）/ baseline（零迁移每区域本地任务）/
      type_maxlat / latency / power_mapping / regions / T_END

对应论文章节：
    问题二（S2）碳感知任务调度 — 阶段 1.4 数据预处理
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
nl = d['network_latency']
rt = d['region_time_data']
pm = d['power_mapping'].set_index('TaskType')['GPU_Power_MW_per_EquivalentGPU']

regions = ['RegionA', 'RegionB', 'RegionC', 'RegionD', 'RegionE', 'RegionF']
H = 2400
T_END = 2406

# ============================================================
# P1. 可行域裁剪矩阵（按任务类型 MaxLatency）
# ============================================================
lat = {(row['FromRegion'], row['ToRegion']): row['NetworkLatency_ms'] for _, row in nl.iterrows()}

# 按任务类型算可达性（可达 = NetworkLatency ≤ MaxLatency）
# 注：E↔F 实时推理 18ms ≤ 20ms 可互迁（建模已确认允许，2026-08-08）
type_maxlat = wt.groupby('TaskType')['MaxLatency_ms'].first().to_dict()
reach_by_type = {}
for tt, ml in type_maxlat.items():
    m = np.zeros((6, 6), dtype=bool)
    for i, s in enumerate(regions):
        for j, t in enumerate(regions):
            m[i, j] = lat.get((s, t), 999) <= ml
    reach_by_type[tt] = m
    print(f"{tt} (MaxLatency={ml}ms): {m.sum()} 条可达路径 / 36")

# ============================================================
# P2. 任务→候选目的地（含预算）
# ============================================================
tasks = []
for _, row in wt.iterrows():
    s_idx = regions.index(row['SourceRegion'])
    cand = [regions[j] for j in range(6) if reach_by_type[row['TaskType']][s_idx, j]]
    assert len(cand) > 0, f"任务 {row['TaskID']} 候选目的地为空（源 {row['SourceRegion']}）"
    gh = row['GPU_Demand'] * row['EstimatedDuration_min'] / 60.0
    tasks.append({
        'id': row['TaskID'], 'type': row['TaskType'],
        'source': row['SourceRegion'], 'cand': cand,
        'arrive': row['ArrivalHour'], 'dur': row['EstimatedDuration_min'] / 60.0,
        'dem': row['GPU_Demand'], 'latest': row['LatestFinishHour'],
        'latency': row['MaxLatency_ms'], 'gh': gh,
        'power': pm[row['TaskType']],
    })
print(f"\nP2: 任务数 {len(tasks)}，平均候选目的地 {np.mean([len(t['cand']) for t in tasks]):.2f} 个")
locked = [t for t in tasks if len(t['cand']) == 1]
print(f"锁死任务（仅 1 候选）: {len(locked)} ({len(locked)/len(tasks):.1%})")

# ============================================================
# P3. 区域电力参数表（0-2399 逐小时）
# ============================================================
rt_main = rt[rt['Hour'] < H].copy()
power = {}
for r in regions:
    sub = rt_main[rt_main['Region'] == r]
    assert len(sub) == H and bool(sub['Hour'].is_monotonic_increasing), f"{r} 电力参数行数/顺序异常"
    power[r] = {
        'price': sub['ElectricityPrice_CNY_per_MWh'].values,       # (2400,)
        'sell': sub['SellPrice_CNY_per_MWh'].values,
        'carbon': sub['CarbonIntensity_tCO2_per_MWh'].values,       # (2400,)
        'renewable': sub['AvailableRenewable_MW'].values,
        'pue': gi.loc[r, 'PUE'],
        'cap': gi.loc[r, 'Available_GPU'],
        'max_it_power': gi.loc[r, 'Max_IT_Power_MW'],
        'max_facility': gi.loc[r, 'Max_Facility_Power_MW'],
    }
print("\nP3: 区域电力参数已就绪")
for r in regions:
    print(f"  {r}: 电价[{power[r]['price'].mean():.0f}] 碳[{power[r]['carbon'].mean():.3f}] PUE[{power[r]['pue']}] Cap[{power[r]['cap']}]")

# ============================================================
# P4. S1 基线数据（零迁移全时域回测输入）
# ============================================================
# 基线：任务本地运行（不迁移），时间维可时移
# 提供每区域本地任务列表 + GPU 容量，供 S1 框架回测
baseline = {r: [] for r in regions}
for t in tasks:
    baseline[t['source']].append(t)
for r in regions:
    gh = sum(x['gh'] for x in baseline[r])
    cap_gh = gi.loc[r, 'Available_GPU'] * H
    print(f"P4 基线 {r}: 本地任务 {len(baseline[r])}，GPU-hour {gh:,.0f} / 容量 {cap_gh:,.0f} = {gh/cap_gh:.1%}")

# ============================================================
# 落盘
# ============================================================
out = {
    'tasks': tasks,
    'reach_by_type': reach_by_type,
    'power': power,
    'baseline': baseline,
    'regions': regions,
    'type_maxlat': type_maxlat,
    'latency': lat,
    'power_mapping': pm.to_dict(),
    'T_END': T_END,
}
with open(BASE / "outputs" / "data" / "s2-preprocessed.pkl", "wb") as f:
    pickle.dump(out, f)
print("\n[OK] 已写入 outputs/data/s2-preprocessed.pkl")

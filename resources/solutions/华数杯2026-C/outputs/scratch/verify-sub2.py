# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 1.1 A 类共享事实验证 — 子问题 S2：可行域裁剪核验、迁移收益上界
    （简单基线）、目的地分配规模估算，为碳感知调度模型提供数据证据

原理：
    1. 可行域裁剪：任务可迁移目标 = {r: NetworkLatency(s→r) ≤ MaxLatency_ms}
       （实时 20ms / 批量 80ms / 训练 150ms）
    2. 迁移收益上界：每任务选"成本最小"可行目标（不含 GPU 容量约束），
       对比本地基线 → 成本/碳排降幅上界。⚠️ 这是无容量约束上界，真实约束下不可达
    3. 目的地分配：按成本最优贪心分配 → 暴露容量瓶颈（如 E 超容量 198%），
       论证 S2 必须建模 GPU 容量约束
    4. 成本 = GPU_hours × power × PUE × price；碳排 = GPU_hours × power × PUE × carbon

输入数据：
    - outputs/data/c-data-cleaned.pkl（阶段 0.3 清洗后）
    - 中文指标 → 变量名映射：
      workload_trace: 任务类型→TaskType, 来源区域→SourceRegion, GPU需求→GPU_Demand,
        预估时长(分钟)→EstimatedDuration_min, 最大时延(ms)→MaxLatency_ms
      GPU_information: 区域→Region, 可用GPU→Available_GPU, 能效→PUE
      network_latency: 源→FromRegion, 目标→ToRegion, 时延(ms)→NetworkLatency_ms
      region_time_data: 区域→Region, 小时→Hour, 电价→ElectricityPrice_CNY_per_MWh,
        碳强度→CarbonIntensity_tCO2_per_MWh
      power_mapping: 任务类型→TaskType, 每等效GPU功率(MW)→GPU_Power_MW_per_EquivalentGPU

输出：
    - solution/model-notes/verify-sub2-20260808.md 的数据基础（控制台打印）

对应论文章节：
    问题二（S2）碳感知任务调度 — 阶段 1.1 A 类共享事实
"""
import pickle
import numpy as np
import pandas as pd

with open('outputs/data/c-data-cleaned.pkl', 'rb') as f:
    d = pickle.load(f)
wt = d['workload_trace']
gi = d['GPU_information'].set_index('Region')
nl = d['network_latency']
rt = d['region_time_data']

regions = ['RegionA', 'RegionB', 'RegionC', 'RegionD', 'RegionE', 'RegionF']
H = 2400

# ============================================================
# 1. 可行域裁剪核验（决策点 3）
# ============================================================
lat = {}
for _, row in nl.iterrows():
    lat[(row['FromRegion'], row['ToRegion'])] = row['NetworkLatency_ms']

print("=== 可行域裁剪（MaxLatency 核验）===")
for t in ['AITraining', 'BatchInference', 'RealTimeInference']:
    sub = wt[wt['TaskType'] == t]
    max_lat_vals = sub['MaxLatency_ms'].unique()
    assert len(max_lat_vals) == 1, f"{t} MaxLatency 非唯一: {max_lat_vals}"
    max_lat = max_lat_vals[0]
    # 每个来源区域可迁移到哪些目标
    reachable = {}
    for s in regions:
        ok = [r for r in regions if lat.get((s, r), 999) <= max_lat]
        reachable[s] = ok
    # 统计被锁死的任务比例（只能去 1 个区域）
    locked = 0
    for _, row in sub.iterrows():
        if len(reachable[row['SourceRegion']]) == 1:
            locked += 1
    print(f"{t} (MaxLatency={max_lat}ms): 全图锁死任务 {locked}/{len(sub)} = {locked/len(sub):.1%}")
    print("  可达目标（来源 → [目标区域]）:")
    for s in regions:
        print(f"    {s[6:]}: {[r[6:] for r in reachable[s]]}")

# ============================================================
# 2. 迁移收益上界（简单基线：全部可迁移任务迁往最优区域）
# ============================================================
pm = d['power_mapping'].set_index('TaskType')['GPU_Power_MW_per_EquivalentGPU']

# 每区域电价/碳/PUE（全时域均值，近似计算收益上界）
rt_main = rt[rt['Hour'] < H]
price = rt_main.groupby('Region')['ElectricityPrice_CNY_per_MWh'].mean()
carbon = rt_main.groupby('Region')['CarbonIntensity_tCO2_per_MWh'].mean()
pue = gi['PUE']

def cost_of(task_type, region, gpu_hours):
    """成本 = GPU_hours × power × PUE × price"""
    return gpu_hours * pm[task_type] * pue[region] * price[region]

def co2_of(task_type, region, gpu_hours):
    return gpu_hours * pm[task_type] * pue[region] * carbon[region]

print("\n=== 迁移收益上界（简单基线：迁往成本最优可行区域）===")
# 对每任务算 GPU-hours，找成本最优可行目标
total_cost0 = 0; total_co2_0 = 0
total_cost_opt = 0; total_co2_opt = 0
migrated = 0
for _, row in wt.iterrows():
    t = row['TaskType']; s = row['SourceRegion']
    gh = row['GPU_Demand'] * row['EstimatedDuration_min'] / 60
    max_lat = row['MaxLatency_ms']
    cand = [r for r in regions if lat.get((s, r), 999) <= max_lat]
    # 基线（不迁移）: 本地成本
    c0 = cost_of(t, s, gh); co0 = co2_of(t, s, gh)
    total_cost0 += c0; total_co2_0 += co0
    # 最优: 成本最小可行目标
    best = min(cand, key=lambda r: cost_of(t, r, gh))
    if best != s:
        migrated += 1
    total_cost_opt += cost_of(t, best, gh)
    total_co2_opt += co2_of(t, best, gh)

print(f"总任务: {len(wt)}")
print(f"可迁移任务: {migrated} ({migrated/len(wt):.1%})")
print(f"成本: 基线 {total_cost0/1e6:.1f}M 元 → 最优迁移 {total_cost_opt/1e6:.1f}M 元 → 降 {(total_cost0-total_cost_opt)/total_cost0:.1%}")
print(f"碳排: 基线 {total_co2_0/1e3:.1f}kt → 最优迁移 {total_co2_opt/1e3:.1f}kt → 降 {(total_co2_0-total_co2_opt)/total_co2_0:.1%}")
print("⚠️ 以上为无 GPU 容量约束的上界，且电价/碳用全时域均值近似（不含时间维时移收益）；实际迁移受容量约束（见第 4 节 E 超容量 198%）")

# ============================================================
# 3. 目的地分配规模估算（决策点 1：区域分解可行性）
# ============================================================
print("\n=== 区域分解：目的地分配后各区域任务量 ===")
# 贪心式分配（按成本最优），粗算每区域承接任务数
assign = {r: 0 for r in regions}
for _, row in wt.iterrows():
    t = row['TaskType']; s = row['SourceRegion']
    gh = row['GPU_Demand'] * row['EstimatedDuration_min'] / 60
    cand = [r for r in regions if lat.get((s, r), 999) <= row['MaxLatency_ms']]
    best = min(cand, key=lambda r: cost_of(t, r, gh))
    assign[best] += 1
for r in regions:
    print(f"{r}: 承接 {assign[r]} 任务 ({assign[r]/len(wt):.1%})")

# 4. 每区域 GPU-hour 需求 vs 容量（迁移后是否超容量）
print("\n=== 迁移后各区域 GPU-hour 需求/容量 ===")
demand = {r: 0.0 for r in regions}
for _, row in wt.iterrows():
    t = row['TaskType']; s = row['SourceRegion']
    gh = row['GPU_Demand'] * row['EstimatedDuration_min'] / 60
    cand = [r for r in regions if lat.get((s, r), 999) <= row['MaxLatency_ms']]
    best = min(cand, key=lambda r: cost_of(t, r, gh))
    demand[best] += gh
for r in regions:
    cap = gi.loc[r, 'Available_GPU'] * H
    print(f"{r}: 需求 {demand[r]:,.0f} / 容量 {cap:,.0f} = {demand[r]/cap:.1%}")
print("\n=== A类验证完成 ===")

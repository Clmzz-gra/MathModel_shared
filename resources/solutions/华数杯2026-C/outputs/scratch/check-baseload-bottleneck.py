# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    验证基荷策略（新能源矩形基荷直接匹配延迟容忍任务）的瓶颈到底
    是"新能源电量"还是"GPU 物理容量"——回答讨论中的核心矛盾。

原理：
    1. 对每个区域 r，取可用新能源 AvailableRenewable(r,t) 的统计下界
       （P10/P25/P50 三档分位）作为"矩形基荷"功率（MW）。
    2. 新能源直供 IT 的功率上限 = 基荷 / PUE(r)（新能源先供 IT，按 PUE
       折算设施侧功率；NonAI 固定负荷先占用，剩余才是 AI 任务可用的
       IT 功率）。
    3. AI IT 功率 → 等效 GPU 数：按训练功率 0.16 MW/GPU 折算（下限）与
       批量功率 0.10 MW/GPU 折算（上限），即：
         GPU_from_baseload = AI_IT_from_baseload / GPU_Power(type)
       换算成"每小时基荷能支撑的等效 GPU 数"，再乘 2400h 得 GPU-hour
       上限。
    4. 对比物理 GPU 容量（Available_GPU × 2400）：
       - 若 GPU_from_baseload > Available_GPU → 物理 GPU 是瓶颈（基荷
         策略在容量饱和区无效，多余新能源只能弃/储）；
       - 若 GPU_from_baseload < Available_GPU → 新能源电量是瓶颈（基荷
         策略可提升利用率，GPU 尚有余量）。
    5. 另算"把 E/F 实际 GPU 池子填满所需的新能源 vs 基荷可用量"，
       量化结构性过载（延迟容忍任务量与 E/F 容量比）。

输入数据：
    - outputs/data/c-data-cleaned.pkl（阶段 0.3 清洗后，与
      preprocess-sub2.py 同源同口径）
    - 中文指标 → 变量名映射：
      GPU_information: 区域→Region, 可用GPU→Available_GPU, 能效→PUE,
        最大IT功率→Max_IT_Power_MW, 最大设施功率→Max_Facility_Power_MW
      region_time_data: 区域→Region, 小时→Hour,
        可再生可用→AvailableRenewable_MW, 非AI负荷→NonAI_IT_Load_MW
      power_mapping: 任务类型→TaskType, 每等效GPU功率(MW)→GPU_Power_MW_per_EquivalentGPU
      workload_trace: 任务类型→TaskType, GPU需求→GPU_Demand,
        预估时长(分钟)→EstimatedDuration_min, 来源区域→SourceRegion

输出：
    - 控制台统计量（PR-014）：每区域基荷支撑的等效 GPU vs 物理 GPU 容量、
      瓶颈判定、结构性过载比
    - outputs/data/basecheck.pkl — 各分位下的基荷功率/等效GPU/瓶颈判定

对应论文章节：
    问题四（S4）算-储-电协同 — 基荷匹配策略可行性验证（阶段 2.x 前置）
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r"e:\MathModel_pj-2026-C")

with open(BASE / "outputs" / "data" / "c-data-cleaned.pkl", "rb") as f:
    d = pickle.load(f)

gi = d['GPU_information'].set_index('Region')
rt = d['region_time_data']
pm = d['power_mapping'].set_index('TaskType')['GPU_Power_MW_per_EquivalentGPU']
wt = d['workload_trace']

regions = ['RegionA', 'RegionB', 'RegionC', 'RegionD', 'RegionE', 'RegionF']
H = 2400

# 各类型单位等效 GPU 功率（MW/GPU）
p_train = pm['AITraining']
p_batch = pm['BatchInference']

# 逐区域新能源可用量（主时域 0-2399）
avail = {}
nonai = {}
for r in regions:
    sub = rt[(rt['Region'] == r) & (rt['Hour'] < H)]
    avail[r] = sub['AvailableRenewable_MW'].values
    nonai[r] = sub['NonAI_IT_Load_MW'].values

# 可用新能源的统计口径（控制台打印）
print("=" * 72)
print("一、各区域可用新能源分布（主时域 0-2399，MW）")
print("=" * 72)
print(f"{'Region':<8}{'min':>8}{'P10':>8}{'P25':>8}{'P50':>8}{'mean':>8}{'max':>8}")
avail_stat = {}
for r in regions:
    a = avail[r]
    q = np.percentile(a, [0, 10, 25, 50, 100])
    avail_stat[r] = {'min': q[0], 'P10': q[1], 'P25': q[2], 'P50': q[3],
                     'mean': a.mean(), 'max': q[4]}
    print(f"{r:<8}{q[0]:>8.0f}{q[1]:>8.0f}{q[2]:>8.0f}{q[3]:>8.0f}"
          f"{a.mean():>8.0f}{q[4]:>8.0f}")

print()
print("=" * 72)
print("二、各区域物理 GPU 容量 vs 基荷矩形支撑的等效 GPU")
print("=" * 72)
print(f"{'Region':<8}{'AvailGPU':>9}{'PUE':>6}{'NonAI均值':>10}"
      f"{'基荷(P25)':>10}{'→IT功率':>10}{'→等效GPU':>10}{'瓶颈判定':>14}")

results = []
for r in regions:
    cap_gpu = gi.loc[r, 'Available_GPU']
    pue = gi.loc[r, 'PUE']
    nonai_mean = nonai[r].mean()

    for quant in ['P10', 'P25', 'P50']:
        base_power = avail_stat[r][quant]          # 新能源基荷功率 (MW)
        # 新能源直供 IT 侧功率（设施侧 = IT × PUE → IT = 基荷/PUE）
        it_from_base = base_power / pue            # IT 功率 (MW)
        ai_it = max(it_from_base - nonai_mean, 0)  # 扣掉 NonAI 固定负荷
        # 等效 GPU：按训练功率（最保守）与批量功率（最宽松）上下限
        gpu_lower = ai_it / p_train                # 全训练
        gpu_upper = ai_it / p_batch                # 全批量

        if quant == 'P25':
            print(f"{r:<8}{cap_gpu:>9.0f}{pue:>6.2f}{nonai_mean:>10.1f}"
                  f"{base_power:>10.0f}{it_from_base:>10.1f}"
                  f"{gpu_lower:>10.0f}"
                  f"{'GPU受限' if gpu_lower > cap_gpu else '电量受限':>14}")
            results.append({
                'region': r, 'cap_gpu': cap_gpu, 'pue': float(pue),
                'nonai_mean': nonai_mean,
                'base_power': base_power, 'it_from_base': it_from_base,
                'gpu_eq_train': gpu_lower, 'gpu_eq_batch': gpu_upper,
                'bottleneck': 'GPU' if gpu_lower > cap_gpu else 'ENERGY',
                'cap_gpuh': cap_gpu * H,
                'base_gpuh_train': gpu_lower * H,
                'base_gpuh_batch': gpu_upper * H,
            })

print()
print("=" * 72)
print("三、结构性过载检验：延迟容忍任务量 vs E/F 容量")
print("=" * 72)
# 延迟容忍任务 = AITraining + BatchInference
dt = wt[wt['TaskType'].isin(['AITraining', 'BatchInference'])].copy()
dt['gh'] = dt['GPU_Demand'] * dt['EstimatedDuration_min'] / 60.0
total_dt_gh = dt['gh'].sum()

# E/F 物理 GPU-hour 容量（90% 阈值同 S2 口径）
ef_cap_gh = sum(gi.loc[r, 'Available_GPU'] * H for r in ['RegionE', 'RegionF'])
ef_cap_gh_90 = ef_cap_gh * 0.9
d_cap_gh = gi.loc['RegionD', 'Available_GPU'] * H

print(f"延迟容忍任务（训练+批量）总 GPU-hour : {total_dt_gh:,.0f}")
print(f"E+F 物理 GPU-hour 容量              : {ef_cap_gh:,.0f} "
      f"（90% 阈值 = {ef_cap_gh_90:,.0f}）")
print(f"D   物理 GPU-hour 容量              : {d_cap_gh:,.0f}")
print(f"延迟容忍任务 / (E+F 90%容量)        : {total_dt_gh / ef_cap_gh_90:.2f}×"
      f"  -> {'过载' if total_dt_gh > ef_cap_gh_90 else '未过载'}")
print(f"延迟容忍任务 / (E+F+D 90%容量)      : "
      f"{total_dt_gh / (0.9 * (ef_cap_gh + d_cap_gh)):.2f}×")

print()
print("=" * 72)
print("四、基荷策略可行区间：GPU 容量 ↔ 基荷等效 GPU 的交叉点")
print("=" * 72)
# 反解：要让"基荷等效GPU ≥ 物理GPU"，需要多少新能源基荷功率？
for r in regions:
    cap_gpu = gi.loc[r, 'Available_GPU']
    pue = gi.loc[r, 'PUE']
    nonai_mean = nonai[r].mean()
    # 全训练口径：需要 IT 功率 = cap_gpu × p_train + nonai
    need_it = cap_gpu * p_train + nonai_mean
    need_base = need_it * pue          # 设施侧所需新能源功率
    max_avail = avail_stat[r]['max']
    p25 = avail_stat[r]['P25']
    cover = "✔ 基荷(P25)已够" if p25 >= need_base else (
        "✘ P25不够" + (f"，但max够" if max_avail >= need_base else "，max也不够"))
    print(f"{r:<8}物理GPU {cap_gpu:>5.0f} → 需新能源 {need_base:>7.0f} MW"
          f" | P25={p25:>6.0f} max={max_avail:>6.0f} → {cover}")

# 保存
out = {'avail_stat': avail_stat, 'results': results}
with open(BASE / "outputs" / "data" / "basecheck.pkl", "wb") as f:
    pickle.dump(out, f)
print(f"\n已保存 outputs/data/basecheck.pkl")

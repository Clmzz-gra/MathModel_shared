# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 0.3 观测深挖 — 验证 GPU_Utilization_Percent>100% 的成因、弃电率结构、
    GPU-hour 折算与 GPU_Demand 长尾分布，为数学讲解提供数据证据

原理：
    - GPU 利用率 = Σ(实际占用 GPU)/可调度容量。基准利用率 GPU_Utilization_Percent 与
      按到达即运行聚合的 GPU 需求对比，检验 >100% 行是否由任务聚合超过容量所致
    - 弃电率 = Curtailment/AvailableRenewable，分区域×时段结构分析
    - GPU-hour = GPU_Demand × 运行小时数（重叠折算）

输入数据：
    - outputs/data/c-data-cleaned.pkl（阶段 0.3 清洗后）

输出：
    - 控制台打印分析结果（供讲解使用）

对应论文章节：
    阶段 0.3 基础数据清洗 — 观测讲解
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"e:\MathModel_pj-2026-C")
clean = pd.read_pickle(BASE / "outputs" / "data" / "c-data-cleaned.pkl")
wt = clean["workload_trace"]
rt = clean["region_time_data"]
gi = clean["GPU_information"].set_index("Region")
st = clean["storage_information"].set_index("Region")

print("=" * 70)
print("[1] GPU_Utilization_Percent > 100% 成因验证")
print("=" * 70)

# 按 到达小时×来源区域 聚合 GPU 需求（到达即开工的极端假设）
agg = wt.groupby(["ArrivalHour", "SourceRegion"])["GPU_Demand"].sum().rename("AggGPU_demand")
rt2 = rt.merge(agg, left_on=["Hour", "Region"], right_index=True, how="left").fillna({"AggGPU_demand": 0})
rt2["AggUtil_pct"] = rt2["AggGPU_demand"] / rt2["Region"].astype(str).map(gi["Available_GPU"]) * 100

over = rt2[rt2["GPU_Utilization_Percent"] > 100]
print(f">100% 行数: {len(over)}")
print(f">100% 行中 到达聚合利用率 AggUtil_pct 分布: "
      f"min={over['AggUtil_pct'].min():.1f} max={over['AggUtil_pct'].max():.1f} mean={over['AggUtil_pct'].mean():.1f}")
print(f">100% 行中 GPU_Utilization_Percent 分布: mean={over['GPU_Utilization_Percent'].mean():.1f}")
print(f"全部行 GPU_Utilization vs AggUtil 相关系数: {rt2['GPU_Utilization_Percent'].corr(rt2['AggUtil_pct']):.3f}")
# 查看超100%行的典型行
print("\n典型 >100% 行（E/F）:")
print(over[over["Region"].isin(["RegionE", "RegionF"])][
    ["Hour", "Region", "AITrainingPower_MW", "GPU_Utilization_Percent", "AggUtil_pct", "IT_Load_MW", "NonAI_IT_Load_MW"]
].head(8).to_string(index=False))
# 检查 AITrainingPower 与 利用率的关系：利用率是否由 AITrainingPower/容量导出？
print("\n>100% 行的 AITrainingPower_MW / Max_IT_Power_MW：")
mip = gi["Max_IT_Power_MW"]
print((over["AITrainingPower_MW"] / over["Region"].astype(str).map(mip) * 100).describe().round(1).to_string())

print()
print("=" * 70)
print("[2] 弃电率结构（分区域）")
print("=" * 70)
rt["Curtail_rate"] = rt["Curtailment_MW"] / rt["AvailableRenewable_MW"]
g = rt.groupby("Region").agg(
    avail_mean=("AvailableRenewable_MW", "mean"),
    curt_mean=("Curtailment_MW", "mean"),
    used_mean=("UsedRenewable_MW", "mean"),
    rc_mean=("RenewableCharge_MW", "mean"),
    sell_mean=("GridSell_MW", "mean"),
    curtail_rate=("Curtail_rate", "mean"),
).round(2)
g["利用率口径"] = ((g["used_mean"] + g["rc_mean"] + g["sell_mean"]) / g["avail_mean"] * 100).round(1)
print(g.to_string())

print()
print("=" * 70)
print("[3] GPU-hour 折算与长尾分布")
print("=" * 70)
wt2 = wt.copy()
# 观测口径：向上取整跨小时数（保守上界，用于观测"是否必须时移"）。
# 注：人类已裁定建模口径用精确小数（GPU-hour = GPU_Demand × Duration_min/60），见 domain-knowledge.md 建模口径区
wt2["Duration_h"] = np.ceil(wt2["EstimatedDuration_min"] / 60)  # 向上取整跨小时数（保守）
wt2["GPU_hours"] = wt2["GPU_Demand"] * wt2["Duration_h"]
print(f"总 GPU-hour（按向上取整小时）: {wt2['GPU_hours'].sum():.0f}")
print(f"每小时平均 GPU-hour 需求: {wt2['GPU_hours'].sum() / 2400:.1f}")
print(f"Available_GPU 总量: {gi['Available_GPU'].sum()}")
print(f"GPU-hour/容量比: {wt2['GPU_hours'].sum() / 2400 / gi['Available_GPU'].sum():.2f}（>1 意味必须时移才能容纳）")
print(f"\nGPU_Demand 分位数: P50={wt['GPU_Demand'].quantile(.5):.0f} P90={wt['GPU_Demand'].quantile(.9):.0f} P99={wt['GPU_Demand'].quantile(.99):.0f} max={wt['GPU_Demand'].max()}")
big = wt2[wt2["GPU_Demand"] >= 100]
print(f"GPU_Demand>=100 的大任务: {len(big)} 条 ({len(big)/len(wt2)*100:.2f}%)，合计 GPU 需求 {big['GPU_Demand'].sum()} ({big['GPU_Demand'].sum()/wt2['GPU_Demand'].sum()*100:.1f}%)")
print(f"大任务类型分布: {big['TaskType'].value_counts().to_dict()}")

print()
print("=" * 70)
print("[4] 电价/碳强度差异 → 迁移收益上限")
print("=" * 70)
price_peak = rt[rt["PricePeriod"] == "Peak"].groupby("Region")["ElectricityPrice_CNY_per_MWh"].mean()
price_valley = rt[rt["PricePeriod"] == "Valley"].groupby("Region")["ElectricityPrice_CNY_per_MWh"].mean()
ci = rt.groupby("Region")["CarbonIntensity_tCO2_per_MWh"].mean()
pue = gi["PUE"]
tab = pd.DataFrame({"Peak电价": price_peak, "Valley电价": price_valley, "碳强度": ci, "PUE": pue}).round(3)
print(tab.to_string())

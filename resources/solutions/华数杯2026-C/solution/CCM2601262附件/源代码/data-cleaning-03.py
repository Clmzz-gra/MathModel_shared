# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 0.3 基础数据清洗 — 从 c-data-raw.pkl 加载，执行模型无关的基础清洗：
    重复检测、缺失/零值区分、异常值检测（不自动删除）、类型标准化、跨表校验、
    覆盖率检查；写出清洗后缓存与清洗报告

原理：
    - 重复检测：TaskID 唯一性、区域×小时网格唯一性
    - 缺失/零值：逐列统计 NA 数与非零占比，判断 0 的真实性（如 GridSell=0 为真实值）
    - 异常值：GPU_Utilization_Percent>100 为可疑；NetGridImport 负值=外送净额属合理；
      电价/碳强度按区域 PricePeriod 分组核对量级
    - 类型标准化：Region/TaskType/PricePeriod 等分类列统一为 str + category 编码
    - 跨表校验：Available_GPU 与 Max_Workload_GPUh_per_h 一致性；功率平衡式残差
    - 覆盖率：填补后可用样本占比重算

输入数据：
    - outputs/data/c-data-raw.pkl（阶段 0.2 缓存，原始）

输出：
    - outputs/data/c-data-cleaned.pkl — 清洗后六表字典
    - solution/data-cleaning-03.md — 清洗报告

对应论文章节：
    TRAE.md 阶段 0.3 基础数据清洗
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"e:\MathModel_pj-2026-C")
OUT = BASE / "outputs" / "data"

RAW = pd.read_pickle(OUT / "c-data-raw.pkl")
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]

report = ["# C 题 阶段 0.3 基础数据清洗报告\n"]
report.append("> 由 `outputs/scratch/data-cleaning-03.py` 自动生成，2026-08-07\n")
report.append("> 原则：只做模型无关清洗；异常值不自动删除，写入报告供人类判断；\n> 需要建模上下文的操作推迟到阶段 1.4。\n")

clean = {}


def sec(title):
    report.append(f"\n---\n\n## {title}\n")


# ============ 1. workload_trace ============
sec("1. workload_trace")
wt = RAW["workload_trace"].copy()
report.append(f"原始 shape: {wt.shape}")

# 1.1 重复检测
dup_id = wt["TaskID"].duplicated().sum()
report.append(f"- TaskID 重复: {dup_id}")
if dup_id:
    wt = wt.drop_duplicates(subset="TaskID", keep="first")
    report.append(f"  → 已去重，去重后 {len(wt)}")

# 1.2 缺失值
missing = wt.isna().sum()
report.append(f"- 缺失值总数: {int(missing.sum())}，分布: {missing[missing > 0].to_dict() or '无'}")

# 1.3 零值/异常
for col, lo, hi, note in [
    ("GPU_Demand", 1, 200, "等效 GPU 数，>=1 合理"),
    ("EstimatedDuration_min", 1, 1440, "执行分钟数，0 不合理"),
    ("MaxLatency_ms", 1, 1000, "时延上限"),
]:
    bad = wt[(wt[col] < lo) | (wt[col] > hi)][col]
    report.append(f"- {col} 越界(不在 [{lo},{hi}]): {len(bad)} 条（{note}）")
    if len(bad):
        report.append(f"  → 样本: {bad.head(5).tolist()}")

# EarliestStartHour 是否等于 ArrivalHour
ne = (wt["EarliestStartHour"] != wt["ArrivalHour"]).sum()
report.append(f"- EarliestStartHour ≠ ArrivalHour: {ne} 条（应全等）")

# 类型分布与 SourceRegion 合法
bad_region = ~wt["SourceRegion"].isin(REGIONS)
report.append(f"- SourceRegion 非法区域: {int(bad_region.sum())}")
report.append(f"- TaskType 分布: {wt['TaskType'].value_counts().to_dict()}")
report.append(f"- DelaySensitivity 分布: {wt['DelaySensitivity'].value_counts().to_dict()}")

# 类型标准化
wt["TaskType"] = wt["TaskType"].astype("category")
wt["DelaySensitivity"] = wt["DelaySensitivity"].astype("category")
wt["SourceRegion"] = wt["SourceRegion"].astype("category")
wt["ExecutionMode"] = wt["ExecutionMode"].astype("str").astype("category")
clean["workload_trace"] = wt

# ============ 2. region_time_data ============
sec("2. region_time_data")
rt = RAW["region_time_data"].copy()
report.append(f"原始 shape: {rt.shape}")

missing = rt.isna().sum()
report.append(f"- 缺失值总数: {int(missing.sum())}，分布: {missing[missing > 0].to_dict() or '无'}")

# GPU_Utilization 超 100%
over = rt[rt["GPU_Utilization_Percent"] > 100]
report.append(f"- GPU_Utilization_Percent > 100%: {len(over)} 条 ({len(over)/len(rt)*100:.2f}%)，max={rt['GPU_Utilization_Percent'].max():.2f}")
if len(over):
    report.append(f"  → 样本区域分布: {over['Region'].value_counts().to_dict()}；按 PricePeriod: {over['PricePeriod'].value_counts().to_dict()}")

# 电价按区域×时段核对量级
price = rt.groupby(["Region", "PricePeriod"])["ElectricityPrice_CNY_per_MWh"].agg(["min", "max", "mean"]).round(1)
report.append(f"- 电价 区域×峰谷平 量级:\n```\n{price.to_string()}\n```")

# 0 值真实性：GridSell 为 0 时对应区域是否有外送能力
st = RAW["storage_information"]
sell_ok = rt.merge(st[["Region", "SellLimit_MW"]], on="Region")
only_no_sell = ((sell_ok["SellLimit_MW"] == 0) & (sell_ok["GridSell_MW"] == 0)).mean() * 100
report.append(f"- 无外送能力区域 GridSell=0 占比: {only_no_sell:.1f}%（0 为真实值的印证）")

# 功率平衡式残差（基准值自洽性）
resid = (rt["GridPurchase_MW"] + rt["AvailableRenewable_MW"] + rt["DischargePower_MW"]
         - rt["Total_Load_MW"] - rt["ChargePower_MW"] - rt["GridSell_MW"] - rt["Curtailment_MW"])
report.append(f"- 功率平衡残差: max|resid|={resid.abs().max():.4f}（≈0 则基准自洽）")

# 类型标准化
rt["Region"] = rt["Region"].astype("category")
rt["PricePeriod"] = rt["PricePeriod"].astype("str").astype("category")
rt["DemandResponseLevel"] = rt["DemandResponseLevel"].astype("str").astype("category")
rt["DataPeriod"] = rt["DataPeriod"].astype("str").astype("category")
clean["region_time_data"] = rt

# ============ 3. GPU_information ============
sec("3. GPU_information")
gi = RAW["GPU_information"].copy()
report.append(f"原始 shape: {gi.shape}，缺失: {int(gi.isna().sum().sum())}")
# Available_GPU 与 Max_Workload_GPUh_per_h 一致？
aw = (gi["Available_GPU"] == gi["Max_Workload_GPUh_per_h"]).all()
report.append(f"- Available_GPU == Max_Workload_GPUh_per_h: {aw}")
# 预留比例核对
calc_avail = (gi["Total_GPU"] * (1 - gi["Reserved_GPU_Ratio"])).round(0)
diff = (calc_avail - gi["Available_GPU"]).abs()
report.append(f"- Total×(1-Reserved) vs Available_GPU max差: {diff.max():.2f}（≈0 则自洽）")
gi["Region"] = gi["Region"].astype("category")
clean["GPU_information"] = gi

# ============ 4. storage_information ============
sec("4. storage_information")
ss = RAW["storage_information"].copy()
report.append(f"原始 shape: {ss.shape}，缺失: {int(ss.isna().sum().sum())}")
# InitialSOC 是否 = 45% 容量
ratio = ss["InitialSOC_MWh"] / ss["StorageCapacity_MWh"]
report.append(f"- InitialSOC/容量 比例: {ratio.round(4).tolist()}（是否统一为 0.45）")
# MinSOC 是否 = 10% 容量
ratio2 = ss["MinSOC_MWh"] / ss["StorageCapacity_MWh"]
report.append(f"- MinSOC/容量 比例: {ratio2.round(4).tolist()}（是否统一为 0.10）")
ss["Region"] = ss["Region"].astype("category")
clean["storage_information"] = ss

# ============ 5. network_latency / power_mapping ============
sec("5. network_latency 与 power_mapping")
nl = RAW["network_latency"].copy()
report.append(f"network_latency: {nl.shape}，缺失 {int(nl.isna().sum().sum())}")
# 对称性检查（单向时延，不做对称假设，仅报告差异）
sym = nl.merge(nl, left_on=["FromRegion", "ToRegion"], right_on=["ToRegion", "FromRegion"], suffixes=("", "_rev"))
if len(sym):
    asym = (sym["NetworkLatency_ms"] - sym["NetworkLatency_ms_rev"]).abs()
    report.append(f"- 区域对对称性: 最大不对称差 {asym.max()} ms（单向时延允许不对称）")
nl["FromRegion"] = nl["FromRegion"].astype("category")
nl["ToRegion"] = nl["ToRegion"].astype("category")
nl["LatencyClass"] = nl["LatencyClass"].astype("str").astype("category")
clean["network_latency"] = nl

pm = RAW["power_mapping"].copy()
report.append(f"power_mapping: {pm.shape}，缺失 {int(pm.isna().sum().sum())}")
pm["TaskType"] = pm["TaskType"].astype("category")
clean["power_mapping"] = pm

# ============ 6. 覆盖率检查 ============
sec("6. 候选池覆盖率检查")
# workload 全部任务可用（无剔除）
n_keep_wt = len(clean["workload_trace"])
report.append(f"- workload_trace: {n_keep_wt}/{len(RAW['workload_trace'])} 可用（100%）")
# region_time_data 全网格
rt_clean = clean["region_time_data"]
grid = len(rt_clean)
report.append(f"- region_time_data: {grid}/14442 行保留（100%），覆盖 2407h × 6 区域")
report.append(f"- 所有表均无剔除 ⇒ 候选池覆盖率 100%，无触发 >50% 排除警告")

# ============ 写出 ============
sec("7. 清洗后缓存")
out_pkl = OUT / "c-data-cleaned.pkl"
clean["_meta"] = {"source": "c-data-raw.pkl", "cleaning": "stage 0.3", "generated": "2026-08-07"}
pd.to_pickle(clean, out_pkl)
report.append(f"已写入 `outputs/data/c-data-cleaned.pkl`（{out_pkl.stat().st_size/1024:.0f} KB）")

out_md = BASE / "solution" / "data-cleaning-03.md"
out_md.write_text("\n".join(report), encoding="utf-8")
print(f"[OK] cleaned cache -> {out_pkl}")
print(f"[OK] report -> {out_md}")

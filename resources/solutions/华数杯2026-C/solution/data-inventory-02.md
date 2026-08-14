# C 题 阶段 0.2 数据盘点报告

> 由 `outputs/scratch/data-inventory-02.py` 自动生成，2026-08-07


## GPU_information.xlsx（sheet: GPU中心基础情况）shape=(6, 11)
列：Region, RegionRole, Total_GPU, Max_IT_Power_MW, PUE, Max_Facility_Power_MW, Reserved_GPU_Ratio, Available_GPU, Max_Workload_GPUh_per_h, CapacityLevel, Remarks

## workload_trace.xlsx（sheet: Sheet1）shape=(50000, 11)
列：TaskID, TaskType, ArrivalHour, GPU_Demand, EstimatedDuration_min, DelaySensitivity, SourceRegion, MaxLatency_ms, LatestFinishHour, EarliestStartHour, ExecutionMode

## region_time_data.xlsx（sheet: region_time_data）shape=(14442, 26)
列：Hour, Region, PricePeriod, ElectricityPrice_CNY_per_MWh, SellPrice_CNY_per_MWh, CarbonIntensity_tCO2_per_MWh, AITrainingPower_MW, GPU_Utilization_Percent, AvailableRenewable_MW, UsedRenewable_MW, RenewableCharge_MW, Curtailment_MW, IT_Load_MW, Total_Load_MW, GridPurchase_MW, GridCharge_MW, GridSell_MW, NetGridImport_MW, CarbonEmission_tCO2, DemandResponseLevel, SOC_MWh, ChargePower_MW, DischargePower_MW, Baseline_AI_IT_Load_MW, NonAI_IT_Load_MW, DataPeriod

## power_mapping.xlsx（sheet: 任务功率映射）shape=(3, 3)
列：TaskType, GPU_Power_MW_per_EquivalentGPU, Remarks

## network_latency.xlsx（sheet: network_latency）shape=(36, 4)
列：FromRegion, ToRegion, NetworkLatency_ms, LatencyClass

## storage_information.xlsx（sheet: storage_information）shape=(6, 13)
列：Region, StorageCapacity_MWh, MinSOC_MWh, InitialSOC_MWh, MaxChargePower_MW, MaxDischargePower_MW, ChargeEfficiency, DischargeEfficiency, SellLimit_MW, Remarks, MaxGridImport_MW, MaxGridExport_MW, SOC_State_Convention

## 缓存
已写入 `outputs/data/c-data-raw.pkl`（9486 KB）


---

## 关联键完整性检查

### 1. 区域键覆盖

- workload_trace.SourceRegion: 6 个区域 ✓
- GPU_information.Region: 6 个区域 ✓
- storage_information.Region: 6 个区域 ✓
- network_latency.FromRegion: 6 个区域 ✓
- network_latency.ToRegion: 6 个区域 ✓
- region_time_data.Region: 6 个区域 ✓

### 2. 时域覆盖

- region_time_data.Hour: min=0 max=2406 共 2407 个（期望 0–2406 共 2407）
- workload_trace.ArrivalHour: min=0 max=2399 共 2400 个（期望 0–2399 共 2400）
- region_time_data 缺失小时: 无 ✓
- workload_trace 缺失到达小时: 无 ✓
- region_time_data 区域×小时网格: 实际 14442 / 期望 14442，缺口 0 ✓
- network_latency 区域对: 实际 36 / 期望 36 ✓

### 3. workload 任务分布

- 总任务数: 50000
- 类型分布: {'RealTimeInference': 16724, 'BatchInference': 16717, 'AITraining': 16559}
- 来源区域分布: {'RegionA': 10062, 'RegionD': 9560, 'RegionB': 9200, 'RegionC': 7559, 'RegionE': 6912, 'RegionF': 6707}
- LatestFinishHour: min=1 max=2406，其中 =2406 的占比 66.6%
- GPU_Demand: min=1 max=127 均值 29.5

---

## 变量量纲/单位初查（数值字段量级分布）


### region_time_data 数值字段

| 字段 | min | max | 均值 | 中位数 | 非零比例 |
|------|-----|-----|------|--------|----------|
| ElectricityPrice_CNY_per_MWh | 234.7 | 1096 | 538.8 | 460.4 | 100.0% |
| SellPrice_CNY_per_MWh | 0 | 510.5 | 154.6 | 91.52 | 50.0% |
| CarbonIntensity_tCO2_per_MWh | 0.196 | 0.6876 | 0.4419 | 0.4779 | 100.0% |
| AITrainingPower_MW | 0 | 225.6 | 44.58 | 33.41 | 87.9% |
| GPU_Utilization_Percent | 0 | 135.6 | 37.66 | 35.61 | 99.9% |
| AvailableRenewable_MW | 500 | 1100 | 799.8 | 800 | 100.0% |
| UsedRenewable_MW | 32.95 | 553.9 | 153.2 | 108.7 | 100.0% |
| RenewableCharge_MW | 0 | 106.9 | 14.4 | 0 | 23.0% |
| Curtailment_MW | 0 | 1033 | 537 | 518.4 | 99.9% |
| IT_Load_MW | 189.9 | 513 | 343.8 | 351.4 | 100.0% |
| Total_Load_MW | 241.2 | 651.5 | 451.1 | 462.8 | 100.0% |
| GridPurchase_MW | 0 | 497 | 286.3 | 281 | 91.6% |
| GridCharge_MW | 0 | 62.4 | 7.34 | 0 | 17.3% |
| GridSell_MW | 0 | 220 | 95.23 | 0 | 50.0% |
| NetGridImport_MW | -220 | 497 | 191 | 213.2 | 99.9% |
| CarbonEmission_tCO2 | 0 | 341.7 | 141.6 | 139.7 | 91.6% |
| SOC_MWh | 82 | 900 | 462.3 | 349.6 | 100.0% |
| ChargePower_MW | 0 | 169.3 | 21.74 | 0 | 25.6% |
| DischargePower_MW | 0 | 182 | 18.94 | 0 | 20.8% |
| Baseline_AI_IT_Load_MW | 0 | 232.1 | 51.17 | 40.09 | 99.9% |
| NonAI_IT_Load_MW | 4.566 | 473.4 | 292.6 | 302.9 | 100.0% |

### GPU_information

| 字段 | min | max | 均值 | 中位数 | 非零比例 |
|------|-----|-----|------|--------|----------|
| Total_GPU | 600 | 1600 | 950 | 875 | 100.0% |
| Max_IT_Power_MW | 405 | 720 | 509.2 | 480 | 100.0% |
| PUE | 1.25 | 1.38 | 1.313 | 1.315 | 100.0% |
| Max_Facility_Power_MW | 553.5 | 921.6 | 664.5 | 626.4 | 100.0% |
| Reserved_GPU_Ratio | 0.08 | 0.1 | 0.09 | 0.09 | 100.0% |
| Available_GPU | 540 | 1472 | 867.5 | 798 | 100.0% |
| Max_Workload_GPUh_per_h | 540 | 1472 | 867.5 | 798 | 100.0% |

### storage_information

| 字段 | min | max | 均值 | 中位数 | 非零比例 |
|------|-----|-----|------|--------|----------|
| StorageCapacity_MWh | 300 | 900 | 590 | 585 | 100.0% |
| MinSOC_MWh | 30 | 90 | 59 | 58.5 | 100.0% |
| InitialSOC_MWh | 135 | 405 | 265.7 | 263.8 | 100.0% |
| MaxChargePower_MW | 85 | 260 | 170.8 | 170 | 100.0% |
| MaxDischargePower_MW | 85 | 260 | 170.8 | 170 | 100.0% |
| ChargeEfficiency | 0.93 | 0.94 | 0.935 | 0.935 | 100.0% |
| DischargeEfficiency | 0.92 | 0.93 | 0.925 | 0.925 | 100.0% |
| SellLimit_MW | 0 | 220 | 103.3 | 90 | 50.0% |
| MaxGridImport_MW | 340 | 550 | 466.7 | 510 | 100.0% |
| MaxGridExport_MW | 0 | 220 | 103.3 | 90 | 50.0% |

### workload_trace

| 字段 | min | max | 均值 | 中位数 | 非零比例 |
|------|-----|-----|------|--------|----------|
| GPU_Demand | 1 | 127 | 29.47 | 13 | 100.0% |
| EstimatedDuration_min | 10 | 399 | 204.2 | 204 | 100.0% |
| MaxLatency_ms | 20 | 150 | 83.11 | 80 | 100.0% |
| LatestFinishHour | 1 | 2406 | 2004 | 2406 | 100.0% |

### network_latency

| 字段 | min | max | 均值 | 中位数 | 非零比例 |
|------|-----|-----|------|--------|----------|
| NetworkLatency_ms | 5 | 82 | 42.39 | 41.5 | 100.0% |

### power_mapping

| 字段 | min | max | 均值 | 中位数 | 非零比例 |
|------|-----|-----|------|--------|----------|
| GPU_Power_MW_per_EquivalentGPU | 0.08 | 0.16 | 0.1133 | 0.1 | 100.0% |

### 交叉校验

- Baseline_AI + NonAI 与 IT_Load 之差: max abs = 0.0000（≈0 则口径自洽）
- Total_Load 与 IT_Load×PUE 之差: max abs = 0.0001（≈0 则口径自洽）

---

## 数据清单表（每个子问题：需要的数据 | 已有 | 缺口 | 获取方案）

| 子问题 | 需要的数据 | 已有（附件） | 缺口 | 获取方案 |
|--------|-----------|-------------|------|---------|
| S1 GPU 需求统计+短期预测+基础调度 | 任务轨迹（到达/类型/GPU 需求/时长/SLA）、区域 GPU 容量、时延矩阵 | workload_trace（5 万任务）、GPU_information（6 区域）、network_latency（36 对） | 无外部缺口 | — |
| S2 碳感知任务调度 | S1 数据 + 逐时电价/碳强度/新能源、功率映射 | region_time_data（2407h×6）、power_mapping（3 类） | 无外部缺口 | — |
| S3 储能协同优化 | 给定 IT 负荷（Baseline_AI+NonAI）、储能参数 | region_time_data（Baseline_AI_IT_Load/NonAI_IT_Load/SOC 基准）、storage_information（6 区域） | 无外部缺口 | — |
| S4 算-储-电协同+场景比较 | S2+S3 全部 + 场景输入 | 全部已有 | 场景输入为构造型（非外部数据）：不同碳约束/电价机制/新能源波动情景 | 从现有数据派生场景（如碳价系数、电价平移、风光出力缩放），属建模侧构造，无需外部数据 |

**结论：本题全部数据已由附件提供，无外部数据缺口。** 唯一需要"构造"的是 S4 的场景比较输入，属建模派生而非数据采购。

## 量纲初查值得注意的观测（供阶段 0.3 跟进）

1. **GPU_Utilization_Percent 存在 >100% 值（max=135.6）**——超 100% 的利用率需在清洗时核实（是否基准运行允许超载，还是单位/口径问题）。
2. **Curtailment_MW 弃电量极大**（均值 537 MW，占 AvailableRenewable 均值 800 的 ~67%）——东部 A/B/C 无外送能力（SellLimit=0），新能源富余大量弃电；这为问题三储能与问题四外送提供了优化空间。
3. **SellPrice 非零比例 50%**——恰好对应 D/E/F 三个有外送能力区域，与 SellLimit/GridExport 口径一致。
4. **LatestFinishHour = 2406 占 66.6%**——三分之二任务（训练+大部分批量）以 2406 为最晚完成时点，为弹性调度预留了极大时移空间。
5. **GPU_Demand 长尾**（均值 29.5 / 中位数 13 / 最大 127）——需求分布右偏，预测与调度需注意少数大任务。
6. **EstimatedDuration_min 均值 204 分钟（≈3.4h）**——任务平均持续约 3 小时，跨小时重叠将普遍，GPU-hour 折算重要。
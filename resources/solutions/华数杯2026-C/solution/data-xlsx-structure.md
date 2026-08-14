# C 题 附件 xlsx 数据结构摘要

> 由 `outputs/scratch/inspect-c-data.py` 自动生成，2026-08-07


## GPU_information.xlsx

**sheet 列表**：GPU中心基础情况, 字段说明


### sheet: GPU中心基础情况  （shape=(6, 11)）

- 列: Region (str), RegionRole (str), Total_GPU (int64), Max_IT_Power_MW (int64), PUE (float64), Max_Facility_Power_MW (float64), Reserved_GPU_Ratio (float64), Available_GPU (int64), Max_Workload_GPUh_per_h (int64), CapacityLevel (str), Remarks (str)

前 5 行：
```
Region|RegionRole|Total_GPU|Max_IT_Power_MW|PUE|Max_Facility_Power_MW|Reserved_GPU_Ratio|Available_GPU|Max_Workload_GPUh_per_h|CapacityLevel|Remarks
RegionA|东部高负荷区域/用户侧实时业务中心|700|420|1.35|567.0|0.1|630|630|Medium|靠近用户侧，实时推理需求高，跨区迁移应受时延约束限制。
RegionB|东部高负荷区域/综合业务中心|650|410|1.35|553.5|0.1|585|585|Medium|承接实时推理与批量推理任务，容量略低于RegionA。
RegionC|东部高负荷区域/边缘算力节点|600|405|1.38|558.9|0.1|540|540|Medium-Low|可承担部分时延敏感业务，算力容量相对较小。
RegionD|西部算力中心/低电价大规模训练节点|1600|720|1.28|921.6|0.08|1472|1472|Very High|适合承接AI训练与可延迟批量任务，是主要算力迁移目的地。
RegionE|新能源资源丰富的光伏区域|1100|560|1.25|700.0|0.08|1012|1012|High|光伏出力较高，可结合新能源消纳进行低碳调度。

```

- 已导出: `outputs\data\csv\GPU_information\GPU中心基础情况.csv`

### sheet: 字段说明  （shape=(12, 5)）

- 列: 字段名 (str), 中文名称 (str), 单位 (str), 属性 (str), 含义与使用说明 (str)

前 5 行：
```
字段名|中文名称|单位|属性|含义与使用说明
Region|区域编号|无|输入索引|与region_time_data.xlsx、workload_trace.xlsx中的区域字段保持一致。
RegionRole|区域角色|无|输入参数|描述区域功能定位，如东部高负荷区域、西部算力中心、光伏/风电富集区。
Total_GPU|总GPU等效调度单元|等效GPU单元|输入参数|区域数据中心可用于AI任务调度的总GPU等效规模。
Max_IT_Power_MW|IT侧最大功率|MW|输入约束|数据中心IT设备侧最大可承载功率，用于功率容量约束。
PUE|能源使用效率|无量纲|输入参数|Total_Load_MW = IT_Load_MW × PUE。

```

- 已导出: `outputs\data\csv\GPU_information\字段说明.csv`

## workload_trace.xlsx

**sheet 列表**：Sheet1, 字段说明


### sheet: Sheet1  （shape=(50000, 11)）

- 列: TaskID (int64), TaskType (str), ArrivalHour (int64), GPU_Demand (int64), EstimatedDuration_min (int64), DelaySensitivity (str), SourceRegion (str), MaxLatency_ms (int64), LatestFinishHour (int64), EarliestStartHour (int64), ExecutionMode (str)

前 5 行：
```
TaskID|TaskType|ArrivalHour|GPU_Demand|EstimatedDuration_min|DelaySensitivity|SourceRegion|MaxLatency_ms|LatestFinishHour|EarliestStartHour|ExecutionMode
1|AITraining|456|118|280|Low|RegionD|150|2406|456|NonPreemptive
2|RealTimeInference|1126|3|198|High|RegionB|20|1130|1126|NonPreemptive
3|RealTimeInference|914|5|224|High|RegionB|20|918|914|NonPreemptive
4|RealTimeInference|419|3|97|High|RegionC|20|421|419|NonPreemptive
5|AITraining|2233|115|161|Low|RegionF|150|2406|2233|NonPreemptive

```

- 已导出: `outputs\data\csv\workload_trace\Sheet1.csv`

### sheet: 字段说明  （shape=(11, 5)）

- 列: 字段名 (str), 中文名称 (str), 单位 (str), 属性 (str), 含义与使用说明 (str)

前 5 行：
```
字段名|中文名称|单位|属性|含义与使用说明
TaskID|任务编号|无|输入索引|AI任务唯一编号。
TaskType|任务类型|无|输入参数|AITraining、BatchInference、RealTimeInference。
ArrivalHour|到达小时|h|输入参数|任务到达时段，范围0–2399。
EarliestStartHour|最早开工小时|h|输入约束|等于 ArrivalHour；任务不得在到达前执行。
GPU_Demand|GPU需求量|等效GPU单元|输入参数|任务运行时持续占用的等效GPU数量。

```

- 已导出: `outputs\data\csv\workload_trace\字段说明.csv`

## region_time_data.xlsx

**sheet 列表**：region_time_data, 字段说明


### sheet: region_time_data  （shape=(14442, 26)）

- 列: Hour (int64), Region (str), PricePeriod (str), ElectricityPrice_CNY_per_MWh (float64), SellPrice_CNY_per_MWh (float64), CarbonIntensity_tCO2_per_MWh (float64), AITrainingPower_MW (float64), GPU_Utilization_Percent (float64), AvailableRenewable_MW (float64), UsedRenewable_MW (float64), RenewableCharge_MW (float64), Curtailment_MW (float64), IT_Load_MW (float64), Total_Load_MW (float64), GridPurchase_MW (float64), GridCharge_MW (float64), GridSell_MW (float64), NetGridImport_MW (float64), CarbonEmission_tCO2 (float64), DemandResponseLevel (str), SOC_MWh (float64), ChargePower_MW (float64), DischargePower_MW (float64), Baseline_AI_IT_Load_MW (float64), NonAI_IT_Load_MW (float64), DataPeriod (str)

前 5 行：
```
Hour|Region|PricePeriod|ElectricityPrice_CNY_per_MWh|SellPrice_CNY_per_MWh|CarbonIntensity_tCO2_per_MWh|AITrainingPower_MW|GPU_Utilization_Percent|AvailableRenewable_MW|UsedRenewable_MW|RenewableCharge_MW|Curtailment_MW|IT_Load_MW|Total_Load_MW|GridPurchase_MW|GridCharge_MW|GridSell_MW|NetGridImport_MW|CarbonEmission_tCO2|DemandResponseLevel|SOC_MWh|ChargePower_MW|DischargePower_MW|Baseline_AI_IT_Load_MW|NonAI_IT_Load_MW|DataPeriod
0|RegionA|Valley|468.0|0.0|0.6679|0.0|3.07672|540.19|59.8979|55.0|425.2921|369.74|499.149|457.2511|18.0|0.0|457.2511|305.398|Medium|225.39|73.0|0.0|1.7583333333|367.981667|Main_0_2399
0|RegionB|Valley|448.5|0.0|0.6248|13.4133333333|16.381766|540.19|68.8115|49.5|421.8785|339.81|458.7435|406.132|16.2|0.0|406.132|253.7513|Medium|205.101|65.7|0.0|14.3733333333|325.436667|Main_0_2399
0|RegionC|Valley|435.5|0.0|0.5925|0.0|2.987654|540.19|84.2598|46.75|409.1802|339.21|468.1098|399.15|15.3|0.0|399.15|236.4964|Medium|192.7065|62.05|0.0|1.5506666667|337.659333|Main_0_2399
0|RegionD|Valley|279.5|218.01|0.4525|49.92|21.195652|540.19|161.3846|75.7611|136.37|346.36|443.3408|344.3562|62.4|166.6744|177.6818|155.8212|Low|534.8714|138.1611|0.0|49.92|296.44|Main_0_2399
0|RegionE|Valley|247.0|192.66|0.237|9.76|8.300395|540.19|246.6269|58.7126|105.6827|358.73|448.4125|259.3856|57.6|129.1678|130.2178|61.4744|Low|478.3339|116.3126|0.0|12.06|346.67|Main_0_2399

```

- 已导出: `outputs\data\csv\region_time_data\region_time_data.csv`

### sheet: 字段说明  （shape=(26, 4)）

- 列: 字段 (str), 含义 (str), 单位 (str), 变量属性 (str)

前 5 行：
```
字段|含义|单位|变量属性
Hour|调度时段编号，0–2399，共2400小时。|h|输入索引
Region|区域编号，包括RegionA至RegionF。|无|输入索引
PricePeriod|峰谷平电价时段，Valley/Flat/Peak。|无|输入参数
ElectricityPrice_CNY_per_MWh|从电网购电价格。|元/MWh|输入参数
SellPrice_CNY_per_MWh|新能源富余外送或售电价格。|元/MWh|输入参数

```

- 已导出: `outputs\data\csv\region_time_data\字段说明.csv`

## power_mapping.xlsx

**sheet 列表**：任务功率映射, 计算口径


### sheet: 任务功率映射  （shape=(3, 3)）

- 列: TaskType (str), GPU_Power_MW_per_EquivalentGPU (float64), Remarks (str)

前 5 行：
```
TaskType|GPU_Power_MW_per_EquivalentGPU|Remarks
AITraining|0.16|训练任务的等效 GPU 平均 IT 功率；仅在任务实际占用时计入。
BatchInference|0.1|批量推理任务的等效 GPU 平均 IT 功率；仅在任务实际占用时计入。
RealTimeInference|0.08|实时推理任务的等效 GPU 平均 IT 功率；仅在任务实际占用时计入。

```

- 已导出: `outputs\data\csv\power_mapping\任务功率映射.csv`

### sheet: 计算口径  （shape=(5, 2)）

- 列: 项目 (str), 定稿公式/规则 (str)

前 5 行：
```
项目|定稿公式/规则
AI IT负荷|AI_IT_Load(r,t)=Σ_i[GPU_Demand(i)×Overlap(i,r,t)×GPU_Power(TaskType_i)]。Overlap为任务在时段t的实际占用小时数。
总IT负荷|IT_Load(r,t)=NonAI_IT_Load(r,t)+AI_IT_Load(r,t)。
设施负荷|Total_Load(r,t)=IT_Load(r,t)×PUE(r)。
容量|Σ_i[GPU_Demand(i)×Overlap(i,r,t)]≤Available_GPU(r)×1h。
时域|主时域为0–2399；2400–2406仅用于结清2399小时前到达任务，期间无新任务到达。

```

- 已导出: `outputs\data\csv\power_mapping\计算口径.csv`

## network_latency.xlsx

**sheet 列表**：network_latency, 字段说明, 时延矩阵, 模型说明


### sheet: network_latency  （shape=(36, 4)）

- 列: FromRegion (str), ToRegion (str), NetworkLatency_ms (int64), LatencyClass (str)

前 5 行：
```
FromRegion|ToRegion|NetworkLatency_ms|LatencyClass
RegionA|RegionA|5|Local
RegionA|RegionB|12|Regional
RegionA|RegionC|15|Regional
RegionA|RegionD|65|LongDistance
RegionA|RegionE|78|LongDistance

```

- 已导出: `outputs\data\csv\network_latency\network_latency.csv`

### sheet: 字段说明  （shape=(4, 5)）

- 列: 字段名 (str), 中文名称 (str), 单位 (str), 属性 (str), 含义与使用说明 (str)

前 5 行：
```
字段名|中文名称|单位|属性|含义与使用说明
FromRegion|任务来源区域|无|输入索引|与workload_trace.xlsx中的SourceRegion对应。
ToRegion|任务调度区域|无|输入索引|候选数据中心所在区域。
NetworkLatency_ms|网络时延|ms|输入参数|从任务来源区域到调度区域的单向等效网络时延。
LatencyClass|时延等级|无|说明字段|Local/Regional/InterRegional/LongDistance，用于解释区域迁移距离。

```

- 已导出: `outputs\data\csv\network_latency\字段说明.csv`

### sheet: 时延矩阵  （shape=(6, 7)）

- 列: From\To (str), RegionA (int64), RegionB (int64), RegionC (int64), RegionD (int64), RegionE (int64), RegionF (int64)

前 5 行：
```
From\To|RegionA|RegionB|RegionC|RegionD|RegionE|RegionF
RegionA|5|12|15|65|78|82
RegionB|12|5|14|60|74|79
RegionC|15|14|5|58|70|76
RegionD|65|60|58|5|22|25
RegionE|78|74|70|22|5|18

```

- 已导出: `outputs\data\csv\network_latency\时延矩阵.csv`

### sheet: 模型说明  （shape=(3, 2)）

- 列: 项目 (str), 口径 (str)

前 5 行：
```
项目|口径
网络建模边界|NetworkLatency_ms 为来源区域到执行区域的单向等效时延。
迁移开销|本赛题不考虑迁移数据量、网络带宽、传输能耗和传输费用；仅按时延筛选。
SLA使用规则|实时推理任务到达即开工，且要求 NetworkLatency_ms ≤ MaxLatency_ms；批量/训练任务受 LatestFinishHour 约束。

```

- 已导出: `outputs\data\csv\network_latency\模型说明.csv`

## storage_information.xlsx

**sheet 列表**：storage_information, 字段说明


### sheet: storage_information  （shape=(6, 13)）

- 列: Region (str), StorageCapacity_MWh (int64), MinSOC_MWh (int64), InitialSOC_MWh (float64), MaxChargePower_MW (int64), MaxDischargePower_MW (int64), ChargeEfficiency (float64), DischargeEfficiency (float64), SellLimit_MW (int64), Remarks (str), MaxGridImport_MW (int64), MaxGridExport_MW (int64), SOC_State_Convention (str)

前 5 行：
```
Region|StorageCapacity_MWh|MinSOC_MWh|InitialSOC_MWh|MaxChargePower_MW|MaxDischargePower_MW|ChargeEfficiency|DischargeEfficiency|SellLimit_MW|Remarks|MaxGridImport_MW|MaxGridExport_MW|SOC_State_Convention
RegionA|350|35|157.5|100|100|0.93|0.92|0|不具备新能源外送能力|550|0|SOC_MWh is end-of-hour; InitialSOC_MWh is before Hour 0
RegionB|320|32|144.0|90|90|0.93|0.92|0|不具备新能源外送能力|520|0|SOC_MWh is end-of-hour; InitialSOC_MWh is before Hour 0
RegionC|300|30|135.0|85|85|0.93|0.92|0|不具备新能源外送能力|510|0|SOC_MWh is end-of-hour; InitialSOC_MWh is before Hour 0
RegionD|900|90|405.0|260|260|0.94|0.93|180|具备有限新能源外送能力|510|180|SOC_MWh is end-of-hour; InitialSOC_MWh is before Hour 0
RegionE|820|82|370.0|240|240|0.94|0.93|220|具备有限新能源外送能力|370|220|SOC_MWh is end-of-hour; InitialSOC_MWh is before Hour 0

```

- 已导出: `outputs\data\csv\storage_information\storage_information.csv`

### sheet: 字段说明  （shape=(13, 5)）

- 列: 字段名 (str), 中文名称 (str), 单位 (str), 属性 (str), 含义与使用说明 (str)

前 5 行：
```
字段名|中文名称|单位|属性|含义与使用说明
Region|区域编号|无|输入索引|六个区域编号。
StorageCapacity_MWh|储能容量|MWh|输入约束|区域储能系统最大可存储电量。
MinSOC_MWh|最小SOC|MWh|输入约束|储能安全运行下限。
InitialSOC_MWh|初始SOC|MWh|输入参数|调度初始时刻储能荷电状态。
MaxChargePower_MW|最大充电功率|MW|输入约束|储能每小时最大充电功率。

```

- 已导出: `outputs\data\csv\storage_information\字段说明.csv`
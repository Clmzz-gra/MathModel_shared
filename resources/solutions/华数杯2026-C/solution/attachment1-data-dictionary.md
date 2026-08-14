# C 题 附件 1 — 数据说明与名词解释（整理版）

> 来源：`附件1.docx`（python-docx 提取整理，2026-08-07）

## 一、背景与系统场景说明

- 本赛题设置六个典型区域：
  - **RegionA / RegionB / RegionC**：东部高负荷区域，实时业务需求强、网络时延要求高
  - **RegionD**：西部算力中心，具备较大算力承载能力
  - **RegionE**：光伏资源较丰富区域
  - **RegionF**：风电资源较丰富区域
- 各区域通过广域网络互联，构成跨区域算力调度与能源协同优化场景。
- 系统主运行周期第 0–2399 小时，粒度 1 小时；第 2400–2405 小时结清末端任务；第 2406 小时仅做电力与储能终端状态结算。所有任务最迟时点 2406 前完成，不可抢占、不可拆分、不可中途迁移。

## 二、表 1 — 区域情况说明

| 区域 | 定位 | 含义 |
|------|------|------|
| RegionA | 东部高负荷区域 | 实时业务需求较高，算力压力较大，适合优先保障低时延任务 |
| RegionB | 东部高负荷区域 | 靠近用户侧，适合承接实时推理和部分批量推理任务 |
| RegionC | 东部高负荷区域 | 负荷水平较高，可作为东部任务承接与迁移对比区域 |
| RegionD | 西部算力中心 | 算力容量较大，适合承接可迁移的批量推理和训练任务 |
| RegionE | 光伏资源富集区域 | 新能源出力具有明显日间特征，适合开展低碳调度与新能源消纳分析 |
| RegionF | 风电资源富集区域 | 新能源出力具有波动性，适合开展储能协同与风电消纳分析 |

## 三、表 2 — 三种不同 AI 任务的说明

| TaskType | 中文名称 | DelaySensitivity | 调度含义 |
|----------|---------|------------------|---------|
| RealTimeInference | 实时推理任务 | High | 时延约束强，原则上不宜远距离迁移，优先满足 SLA 和网络时延约束 |
| BatchInference | 批量推理任务 | Medium | 具有一定调度弹性，可在成本、时延和容量之间折中 |
| AITraining | AI 训练任务 | Low | 对实时网络时延不敏感，通常是低电价、低碳和新能源消纳调度的主要调节对象 |

## 四、表 3 — 附件数据表说明（见 problem-statement.md 附件数据文件清单）

## 五、表 4 — GPU_information.xlsx 字段说明

| 字段 | 中文名称 | 单位 | 属性 | 说明 |
|------|---------|------|------|------|
| Region | 区域编号 | 无 | 输入索引 | RegionA 至 RegionF |
| Total_GPU | GPU 总容量 | 等效GPU单元 | 输入参数 | 区域数据中心总体算力资源规模 |
| Available_GPU | 可调度 GPU 容量 | 等效GPU单元 | 输入参数 | 扣除预留比例后的可调度容量，用于容量约束 |
| Max_IT_Power_MW | 最大 IT 功率 | MW | 输入参数 | IT 侧负荷上限 |
| PUE | 能源使用效率 | 无量纲 | 输入参数 | 设施侧总负荷与 IT 负荷之间的转换系数 |
| Max_Facility_Power_MW | 设施侧最大功率 | MW | 输入参数 | 数据中心设施侧最大用电功率 |

## 六、表 5 — workload_trace.xlsx 字段说明

| 字段 | 中文名称 | 单位 | 属性 | 说明 |
|------|---------|------|------|------|
| TaskID | 任务编号 | 无 | 索引 | 任务唯一标识 |
| TaskType | 任务类型 | 无 | 输入参数 | RealTimeInference、BatchInference、AITraining |
| ArrivalHour | 任务到达小时 | h | 输入参数 | 0–2399 |
| SourceRegion | 任务来源区域 | 无 | 输入参数 | 确定任务原始区域和跨区域网络时延 |
| GPU_Demand | GPU 需求量 | 等效GPU单元 | 输入参数 | 任务执行所需的 GPU 资源需求 |
| EstimatedDuration_min | 预计执行时间 | min | 输入参数 | 任务预计持续时间 |
| MaxLatency_ms | 最大网络时延 | ms | 输入参数 | 任务可接受的最大跨区域网络时延 |
| LatestFinishHour | 最晚完成小时 | h | 输入参数 | 主要用于批量推理和训练任务的延迟调度约束 |

> 补充说明：GPU_Demand 表示任务运行时持续占用的等效 GPU 数量，EstimatedDuration_min 表示不可抢占的连续执行时长。每小时容量约束以实际重叠时长折算 GPU-hour。任务迁移后的 AI IT 功率必须按 power_mapping.xlsx 的 GPU_Power_MW_per_EquivalentGPU 计算；区域总 IT 负荷 = NonAI_IT_Load_MW + 调度形成的 AI IT 负荷，设施负荷 = 总 IT 负荷 × PUE。

## 七、表 6 — region_time_data.xlsx 字段说明

| 字段 | 中文名称 | 单位 | 属性 | 说明 |
|------|---------|------|------|------|
| ElectricityPrice_CNY_per_MWh | 购电价格 | 元/MWh | 输入参数 | 计算电网购电成本 |
| SellPrice_CNY_per_MWh | 售电价格 | 元/MWh | 输入参数 | 计算新能源富余外送收益 |
| CarbonIntensity_tCO2_per_MWh | 碳强度 | tCO2/MWh | 输入参数 | 计算购电对应碳排放 |
| AvailableRenewable_MW | 可用新能源出力 | MW | 输入参数 | 区域当前时段可用于消纳、储能或外送的新能源功率 |
| UsedRenewable_MW | 直接消纳新能源 | MW | 基准结果 | 基准运行中直接满足负荷的新能源功率 |
| RenewableCharge_MW | 新能源充电功率 | MW | 基准结果 | 基准运行中用于储能充电的新能源功率 |
| Curtailment_MW | 弃风弃光功率 | MW | 基准结果 | 未被直接消纳、储能或外送利用的新能源功率 |
| IT_Load_MW | IT 侧负荷 | MW | 基准状态 | 数据中心计算设备侧负荷 |
| Total_Load_MW | 设施侧总负荷 | MW | 基准状态 | 按 IT_Load_MW × PUE 计算 |
| GridPurchase_MW | 电网购电功率 | MW | 基准结果 | 包括供负荷购电和 GridCharge_MW |
| GridCharge_MW | 电网充电功率 | MW | 基准结果 | 低谷电价或平价时段从电网购电给储能充电 |
| GridSell_MW | 外送/售电功率 | MW | 基准结果 | 新能源富余外送或售电功率 |
| NetGridImport_MW | 净购电功率 | MW | 基准结果 | GridPurchase_MW − GridSell_MW |
| CarbonEmission_tCO2 | 碳排放量 | tCO2 | 基准结果 | GridPurchase_MW × CarbonIntensity_tCO2_per_MWh |
| SOC_MWh | 储能荷电状态 | MWh | 状态变量 | 储能系统时段末 SOC |
| ChargePower_MW | 储能总充电功率 | MW | 基准结果 | RenewableCharge_MW + GridCharge_MW |
| DischargePower_MW | 储能放电功率 | MW | 基准结果 | 用于削峰填谷和降低电网购电峰值 |

> 补充说明：碳排放量按电网购电功率 × 碳强度 × 1 小时累计。新能源利用率按（直接消纳新能源 + 新能源充电 + 新能源外送）/ 可用新能源累计计算。

## 八、表 7 — network_latency.xlsx 与 storage_information.xlsx 字段说明

| 数据表 | 字段 | 中文名称 | 单位 | 说明 |
|--------|------|---------|------|------|
| network_latency.xlsx | FromRegion | 源区域 | 无 | 任务来源区域或迁移起点 |
| network_latency.xlsx | ToRegion | 目标区域 | 无 | 任务执行区域或迁移终点 |
| network_latency.xlsx | NetworkLatency_ms | 网络时延 | ms | 判断跨区域调度是否满足 MaxLatency_ms 和 SLA 约束 |
| storage_information.xlsx | StorageCapacity_MWh | 储能容量 | MWh | 储能系统最大容量 |
| storage_information.xlsx | MinSOC_MWh | 最小 SOC | MWh | 储能安全运行下限 |
| storage_information.xlsx | MaxChargePower_MW | 最大充电功率 | MW | 储能充电功率上限 |
| storage_information.xlsx | MaxDischargePower_MW | 最大放电功率 | MW | 储能放电功率上限 |
| storage_information.xlsx | ChargeEfficiency / DischargeEfficiency | 充放电效率 | 无量纲 | 用于储能 SOC 动态方程 |
| storage_information.xlsx | SellLimit_MW | 外送上限 | MW | 区域新能源富余外送功率上限 |

> 补充说明：InitialSOC_MWh 是第 0 小时运行前的唯一初始状态；SOC_MWh 是该小时运行后的时段末基准状态。SOC 递推：SOC(t)=SOC(t−1)+ηc·ChargePower(t)−DischargePower(t)/ηd。优化结束时 SOC(2406) 不得低于 InitialSOC_MWh。

## 九、表 8 — 建模边界统一口径

| 说明项 | 统一口径 |
|--------|---------|
| power_mapping.xlsx | 列出三类任务每等效 GPU 的平均 IT 功率，是计算电费、碳排、储能和新能源利用率的唯一功率映射 |
| NonAI_IT_Load_MW | 逐时不可迁移固定 IT 负荷。优化时仅改变任务形成的 AI IT 负荷，不得把原 IT_Load_MW 整体固定 |
| 时域 | 任务到达 0–2399 小时；电力与储能参数 0–2406 小时。2400–2406 期间无新任务，仅结清主时域到达但尚未完成的任务 |
| 购售电边界 | MaxGridImport_MW 和 MaxGridExport_MW 为区域级硬约束；本题不建模跨区域线路潮流 |
| 通信边界 | 仅考虑 network_latency.xlsx 的单向时延。忽略网络带宽、迁移数据量、传输能耗及传输费用 |
| SLA 与完成时限 | 实时任务到达即开工且满足 MaxLatency_ms；批量推理和 AI 训练任务统一可在第 2406 小时前完成 |

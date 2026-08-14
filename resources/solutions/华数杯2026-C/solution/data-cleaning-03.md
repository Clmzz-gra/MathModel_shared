# C 题 阶段 0.3 基础数据清洗报告

> 由 `outputs/scratch/data-cleaning-03.py` 自动生成，2026-08-07

> 原则：只做模型无关清洗；异常值不自动删除，写入报告供人类判断；
> 需要建模上下文的操作推迟到阶段 1.4。


---

## 1. workload_trace

原始 shape: (50000, 11)
- TaskID 重复: 0
- 缺失值总数: 0，分布: 无
- GPU_Demand 越界(不在 [1,200]): 0 条（等效 GPU 数，>=1 合理）
- EstimatedDuration_min 越界(不在 [1,1440]): 0 条（执行分钟数，0 不合理）
- MaxLatency_ms 越界(不在 [1,1000]): 0 条（时延上限）
- EarliestStartHour ≠ ArrivalHour: 0 条（应全等）
- SourceRegion 非法区域: 0
- TaskType 分布: {'RealTimeInference': 16724, 'BatchInference': 16717, 'AITraining': 16559}
- DelaySensitivity 分布: {'High': 16724, 'Medium': 16717, 'Low': 16559}

---

## 2. region_time_data

原始 shape: (14442, 26)
- 缺失值总数: 0，分布: 无
- GPU_Utilization_Percent > 100%: 44 条 (0.30%)，max=135.58
  → 样本区域分布: {'RegionF': 27, 'RegionE': 12, 'RegionC': 3, 'RegionD': 1, 'RegionB': 1}；按 PricePeriod: {'Flat': 21, 'Valley': 13, 'Peak': 10}
- 电价 区域×峰谷平 量级:
```
                       min     max    mean
Region  PricePeriod                       
RegionA Flat         615.9   755.7   708.2
        Peak         992.7  1095.8  1044.3
        Valley       444.6   491.1   468.1
RegionB Flat         590.3   724.2   678.6
        Peak         951.3  1050.2  1000.7
        Valley       426.1   470.6   448.6
RegionC Flat         573.2   703.2   659.0
        Peak         923.7  1019.7   971.7
        Valley       413.7   457.0   435.6
RegionD Flat         367.8   451.3   422.9
        Peak         592.8   654.5   623.7
        Valley       265.5   293.3   279.5
RegionE Flat         325.1   398.8   373.7
        Peak         523.9   578.4   551.1
        Valley       234.6   259.2   247.0
RegionF Flat         342.2   419.8   393.4
        Peak         551.5   608.8   580.1
        Valley       247.0   272.8   260.0
```
- 无外送能力区域 GridSell=0 占比: 50.0%（0 为真实值的印证）
- 功率平衡残差: max|resid|=0.0002（≈0 则基准自洽）

---

## 3. GPU_information

原始 shape: (6, 11)，缺失: 0
- Available_GPU == Max_Workload_GPUh_per_h: True
- Total×(1-Reserved) vs Available_GPU max差: 0.00（≈0 则自洽）

---

## 4. storage_information

原始 shape: (6, 13)，缺失: 0
- InitialSOC/容量 比例: [0.45, 0.45, 0.45, 0.45, 0.4512, 0.45]（是否统一为 0.45）
- MinSOC/容量 比例: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]（是否统一为 0.10）

---

## 5. network_latency 与 power_mapping

network_latency: (36, 4)，缺失 0
- 区域对对称性: 最大不对称差 0 ms（单向时延允许不对称）
power_mapping: (3, 3)，缺失 0

---

## 6. 候选池覆盖率检查

- workload_trace: 50000/50000 可用（100%）
- region_time_data: 14442/14442 行保留（100%），覆盖 2407h × 6 区域
- 所有表均无剔除 ⇒ 候选池覆盖率 100%，无触发 >50% 排除警告

---

## 7. 清洗后缓存

已写入 `outputs/data/c-data-cleaned.pkl`（5480 KB）

---

## 人类决策记录（2026-08-07，阶段 0.3 出口门禁）

| # | 决策点 | 人类决定 | 对后续建模的影响 |
|---|--------|---------|-----------------|
| 1 | GPU_Utilization_Percent >100%（44 行） | **选项 B：保留；论文中引用该基准数字做对比** | 不清洗不删除；建模自算利用率作约束，基准利用率用作结果对比指标 |
| 2 | 任务时长折算方式 | **精确小数**（150 分钟 = 2.5 小时） | 容量约束偏松；GPU-hour = GPU_Demand × EstimatedDuration_min/60 |
| 3 | 实时推理能否东部内迁 | **允许东部内部迁移（A/B/C 之间）**；往返问题已澄清：任务不可中途迁移 + 赛题仅考虑单向时延，无往返概念 | 实时任务可行域 = {来源区域} ∪ {东部其余两区}（时延 12–15ms ≤ 20ms）；不得迁往西部 |
| 4 | 弃电率是否必须改善 | **先记录；降低弃电率将来可单独做优化** | 新能源利用率仍按赛题口径计入问题二~四指标；弃电优化是否独立成目标待方案探索阶段决定 |
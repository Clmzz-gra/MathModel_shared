# C 题 面向算电协同的多目标调度优化研究 — 领域知识库

> 本文件是项目级知识基座，后续所有阶段可单文件检索。
> 结构：一、非文献知识区（基准值/标准实践/关键数值）；二、文献摘要区（结构化条目）；三、数据注释区（阶段 0.3 清洗完成后填充）。

---

## 一、非文献知识区（基准值、标准实践、关键数值）

### 1. 算电协同背景与政策基准

- **政策里程碑**：2026 年政府工作报告首次将"算电协同"纳入国家级新基建工程；2026 年 3 月写入"十五五"规划纲要；5 月国家发改委等四部门印发《关于促进人工智能与能源双向赋能的行动方案》（部署算电协同等十方面任务）。
- **定义**（国家数据局 2026）：算电协同 = 通过数字化技术、智能算法、信息网络，将算力基础设施与电力系统深度融合，推动资源动态匹配与优化配置，实现"以电强算、以算促电"的良性循环。
- **产业格局**：东部（高负荷、低时延、高电价）vs 西部（风光水电富集、低电价、绿电丰富）；"东数西算"八大枢纽节点（京津冀、长三角、粤港澳、成渝、内蒙古、贵州、甘肃、宁夏）。
- **负荷可调度性分级（行业共识，直接对应赛题三类任务）**：
  - 实时推理（latency-sensitive）：不可迁移、不可时移，优先 SLA 与网络时延。
  - 批量推理（medium）：可在成本、时延、容量间折中。
  - AI 训练（delay-tolerant）：可时移 + 可跨区域迁移，是低碳调度与新能源消纳的主要调节对象。
- **工程落地案例**：内蒙古和林格尔算电协同项目（36 万千瓦风光直供、绿电占比 43%）；乌兰察布全国首个"绿电直连+源网荷储一体化"数据中心项目；新疆伊吾"疆算入渝"跨省绿电算力输送。
- **跨三地算电协同智能调度测试（2026，全国首例）**：异构芯片（英伟达/昇腾）跨地域 200 秒内完成迁移、成功率 100%；负荷预测准确率 98%；系统可帮电网削峰填谷。

### 2. 数据中心能耗基准值

- 全球：2025 年数据中心用电约 485 TWh（占全球 ~1.7%）；2030 年预计 ~950 TWh（~3%）。AI 负载 2030 年较当前或增 165%。
- 中国：2025 年全国算力中心总用电量 1700 亿千瓦时（占全社会 1.6%）；2030 年预计突破 7000 亿千瓦时（>5%）。
- **PUE（能源使用效率）**：行业均值 1.1–1.5；设施负荷 = IT 负荷 × PUE。优化目标之一即是利用区域 PUE 差异做地理迁移。
- GPU 功率量级（常识，赛题以 power_mapping.xlsx 为准）：等效 GPU 平均 IT 功率通常数百 W 量级（如 A100 ~400W、H100 ~700W）；迁移后 AI IT 功率必须按赛题统一映射表计算。
- 算力中心配套储能（2025 年底）：我国约 40 GWh，储能在数据中心"削峰填谷、平抑新能源波动"中作用关键。

### 3. 碳感知调度（Carbon-Aware Scheduling, CAS）标准实践

- **核心机制**：将延迟容忍负载从高碳强度时段迁移至低碳强度/新能源富余时段（时移），或从高电价高碳区域迁往低电价低碳区域（地理迁移）。
- **24/7 碳免费计算**（UN 24/7 CFE 倡议）：每小时消耗都与碳免费电力匹配；需可再生能源互补 + 储能 + 负载调度三者组合。
- **负载时移效果基准**：工程案例显示，负荷预测准确率 98% 时，配合自建光伏 + 虚拟电厂绿电，可实现从"用绿电"到"帮绿电消纳"的闭环。
- **价格-碳强度联动**：碳价提升会同时提高总成本与降低排放（每 kg 碳 $0.1/$0.2 时排放降 2.12%/5.37%），说明碳约束是调度策略变化的驱动力——对应赛题问题四的场景比较。

### 4. 储能（BESS）运行基准

- 典型参数：锂电池 2–4 h 储能、$150/kWh 安装成本；充放电效率 85–95%（往返）；泵水储能往返效率 ~80%。
- **SOC 递推标准式**（赛题口径）：
  SOC(t) = SOC(t−1) + ηc·ChargePower(t) − DischargePower(t)/ηd
  （充电乘效率、放电除效率；SOC 上下限 + 充放电功率上限约束）
- **储能价值来源**：削峰填谷降购电峰值（NetGridImport 峰值）、低谷充电/高峰放电套利、提升新能源就地消纳、紧急备用。互斥约束：同一时段不可同时充放电。
- **协同效应**（文献共识）：储能单独使用无法替代负载时移的价值；"储能 + 负载时移"协同才显著降本（共享储能双层博弈研究中成本降 3.14%、储能收益 2605 元）。峰值净购电功率约束趋紧时，储能日价值可翻倍以上。

### 5. 新能源消纳与利用率口径

- **新能源利用率**（赛题口径）=（直接消纳 + 新能源充电 + 新能源外送）/ 可用新能源累计。
- 弃风弃光：未被直接消纳、储能或外送利用的部分；提升利用率 = 减少弃电。
- 功率平衡式（赛题统一口径）：
  GridPurchase + AvailableRenewable + DischargePower = Total_Load + ChargePower + GridSell + Curtailment

### 6. 多目标优化方法基准

| 方法 | 适用场景 | 特点 |
|------|---------|------|
| 加权求和法 | 目标权重先验明确 | 简单、可解释，权重需灵敏度分析 |
| ε-约束法 | SLA 严格限制 | 一个目标作约束，优化其余目标 |
| NSGA-II / MOEA-D | 离线/准实时，需 Pareto 前沿 | 解集、多样性强；大规模收敛慢 |
| MILP 精确解 | 小规模 | 全局最优，做基准对比 |
| 启发式/自适应大邻域（ALNS） | 大规模 | hypervolume 可比 NSGA-II 高 4.2× |
| 强化学习（DRL） | 在线动态环境 | 自适应，需训练数据与收敛保障 |

- 赛题问题二/四为大规模离散（任务迁移 × 开工时段 × 2400 小时）多目标问题 → 需权衡求解规模，常见做法：线性化单目标化（加权/ε-约束）+ MILP，或双层分解（调度 + 能量），或启发式。
- 关键约束：GPU 容量（按 GPU-hour 重叠折算）、IT 功率、设施功率、网络时延 SLA、任务完成时限（2406 前）、购售电边界（MaxGridImport/Export）、SOC 边界。

### 7. GPU 工作负载预测方法基准

- **任务到达/GPU 需求预测**：
  - 经典统计：ARIMA（适用于有周期性的交互负载；对高度波动 GPU 负载效果差，假设平稳性）。
  - 深度时序：LSTM/GRU、Transformer 系（Autoformer、TimesNet）、DLinear；TFEGRU（时频增强 GRU+注意力，在 Google/Alibaba 真实 trace 优于 SOTA）。
  - 组合分解：PRISM（原语库 + 谱细化，刻画多周期性 + 突发性，突峰段误差显著降低）。
  - 经验值：小时级 GPU 需求预测（Ridge 回归）RMSE ≈ 38.5 GPU/h 即优于 last-hour 与季节朴素基线；GPU 负载突发性强，NRMSE 明显差于 CPU/内存（LSTM 复现 ~5–6 vs ~2）。
- **预测驱动调度**：预测感知自动缩放（SageServe：ARIMA 预测 TPS → ILP 联合路由/缩放，GPU 小时节省 25%）；负载预测 → 容量预留 → 主动伸缩。
- **赛题规定**：问题一建议 0–2351 训练、2352–2375 调参验证、0–2375 重训、2376–2399 最终测试（24h 滚动窗口）。

### 8. 关键工程认知（建模注意事项）

- **任务不可抢占、不可拆分、不可中途迁移** → 调度模型为离散决策：每个任务指定执行区域 + 开工时段，持续运行至结束。
- **实时推理到达即开工** → 无开工时段自由度，仅区域分配（且受 MaxLatency_ms 约束）；批量推理与训练可延后，LatestFinishHour 约束 + 2406 硬期限。
- **每小时容量约束按 GPU-hour 实际重叠折算**：同时运行任务 GPU_Demand 之和 ≤ Available_GPU。
- **迁移成本忽略**（赛题说明 7：不建模带宽/迁移数据量/传输能耗/传输费用）→ 迁移决策只受时延 SLA 与容量/电力约束影响。
- **问题三固定任务调度与 IT 负荷**（Baseline_AI_IT_Load_MW + NonAI_IT_Load_MW），只优化储能充放电、购售电、新能源分配。
- **电网侧**：区域级硬约束 MaxGridImport/Export；不建模跨区域潮流。

---

## 二、文献摘要区（结构化条目）

### 碳感知调度 / 算电协同

1. **Carbon Explorer: A Holistic Framework for Designing Carbon Aware Datacenters**
   - 作者/年份：Acun et al., ASPLOS'23（Meta + UPenn + Stanford + Harvard）
   - 核心结论：面向 24/7 碳免费计算，联合优化"可再生能源互补配置 + 储能 + 负载时移"的解决方案空间；用延迟容忍负载（AI 训练等）做碳感知时移。
   - 适用：问题二/四的负载时移与储能协同设计理念。
   - 链接: https://www.cs.cmu.edu/~18742/papers/Acun2023.pdf  DOI: 10.1145/3575693.3575754

2. **Contextual Robust Optimization for AI Data Center Scheduling with Statistical Guarantees**
   - 作者/年份：Yang, Weng & Chen, arXiv 2606.17466 (2026)
   - 核心结论：AI 训练/推理异构建模 + 可再生出力与负载预测误差 → 上下文联合机会约束鲁棒优化；实测运行成本平均降 5.57% 且保持可行性保证。
   - 适用：问题二调度模型在预测误差下的鲁棒性讨论。
   - 链接: https://arxiv.org/html/2606.17466v1

3. **CFWS: DRL-Based Framework for Energy Cost and Carbon Footprint Optimization in Cloud Data Centers**
   - 作者/年份：Zhao, Zhou & Li, IEEE Trans. Sustainable Computing, 2025
   - 核心结论：深度强化学习框架 + 自适应阈值（TCN-MAD）+ VM 跨区域迁移；节省 5.67%–13.22% 褐电、最大化 RES 利用、迁移次数降 86.53%。
   - 适用：问题四大规模动态场景的可选求解思路（DRL）。
   - DOI: 10.1109/TSUSC.2024.3391791

4. **Carbon-aware electricity cost minimization for sustainable data centers (CECM)**
   - 作者/年份：Dou, Qi, Wei & Song, IEEE Trans. Sustainable Computing
   - 核心结论：Lyapunov 在线优化框架调度延迟容忍负载，无需未来信息即可权衡电费与性能；实测电费降 9.26%。
   - 适用：在线/无预测场景的调度下界参考。

5. **Watts vs. Bytes: Turning Data Centers into Grid Assets via Storage–Compute Co-Optimization**
   - 作者/年份：Liu, Shin & Deka (MIT), arXiv 2605.16190 (2026)
   - 核心结论：日前数据中心-储能联合优化（负载调度 + DVFS + BESS 放电），考虑峰值/爬坡互联约束；互联约束趋紧时 BESS 日价值翻倍以上；不可调度负载占比高时成本增 25%+。
   - 适用：问题三（储能协同）与问题四（算-储-电联合）的建模框架参考。
   - 链接: https://arxiv.org/html/2605.16190v1

6. **Bi-Level Stackelberg Game-Based Optimization Model for Shared Energy Storage in Data Centers Considering Computational Flexibility**
   - 作者/年份：Qie et al., Energies 2026, 19(15), 3681
   - 核心结论：双层（储能运营商利润最大化 / 算力中心成本最小化）博弈模型；仅储能不负载迁移仅获储能收益，储能+负载时移协同才显著降本（算力中心成本降 3.14%）。
   - 适用：问题三储能策略效果分析的论证素材。
   - DOI: 10.3390/en19153681

7. **Fine-Grained Scheduling Strategies and Optimization of Wind–Solar–Storage Powered Data Center Microgrids (WSETSS)**
   - 作者/年份：Dong et al., IEEE Access 2025, 13: 215339-215356
   - 核心结论：风光储数据中心微电网多目标协同调度（成本/碳排/用户满意度），13 类时延容忍窗口利用时间柔性；价格导向策略总成本降 2.92%。
   - 适用：问题四场景比较（不同电价/碳约束/新能源场景）的设计参考。
   - DOI: 10.1109/ACCESS.2025.3641756

### 负载预测 / GPU 调度

8. **SageServe: Optimizing LLM Serving on Cloud Data Centers with Forecast Aware Auto-Scaling**
   - 作者/年份：Jaiswal et al. (Microsoft + UIUC 等), arXiv 2502.14617 (2025)
   - 核心结论：ARIMA 预测区域-模型粒度 TPS → ILP 联合路由与缩放；GPU 小时节省 25%、自动缩放浪费降 80%、月成本节省可达 $2.5M。
   - 适用：问题一预测驱动调度的方法参考（预测 → 容量/调度决策）。
   - 链接: https://arxiv.org/html/2502.14617

9. **PRISM: Dynamic Primitive-Based Forecasting for Large-Scale GPU Cluster Workloads**
   - 作者/年份：Wu et al., DAC'26 (西南交大 + Penn State)
   - 核心结论：字典驱动时序分解 + 自适应谱细化的可解释组合预测框架，刻画 GPU 负载高波动、多周期、异构性；突发段误差显著降低。
   - 适用：问题一 GPU 需求预测的先进方法参考（若需要提升预测精度）。
   - DOI: 10.1145/3770743.3804350

10. **TFEGRU: Time-Frequency Enhanced GRU With Attention for Cloud Workload Prediction**
    - 作者/年份：Zhao, Lin et al., IEEE Trans. Services Computing, 2025
    - 核心结论：时频增强块 + 通道独立 + 多头注意力 GRU；在 Google/Alibaba 真实 trace 上优于 SOTA。
    - 适用：问题一预测模型对比候选。
    - DOI: 10.1109/TSC.2024.3517324

11. **Profit-Aware Spot GPU Admission Control (JTIE 2026, Alibaba cluster-trace 2026)**
    - 核心数据点：46.7 万条任务记录、4278 GPU 节点；小时级 GPU 需求预测 Ridge RMSE=38.50 GPU/h，优于 last-hour 与季节朴素基线。
    - 适用：问题一预测基线与误差量级参考。

### 政策 / 行业报告

12. **中国信通院《算力电力协同发展研究报告（2025）》**
    - 核心内容：算电协同内涵、要素与发展阶段；六大关键举措（源荷互动、储荷互动、网荷协同、源网荷储一体化、算力负载调度、绿电绿证交易）；新型电力系统调度由"源随荷动"向"源网荷储多元互动"转变。
    - 适用：问题背景表述与政策术语引用。
    - 链接: https://www.caict.ac.cn/english/research/whitepapers/202509/P020250903602966954403.pdf

13. **行业报道：算电协同从"概念"到"落地"（国家能源局/新华网 2026-07）**
    - 关键数据：Token 日消耗 2024 年初 1000 亿 → 2026 年 3 月突破 140 万亿；单座数据中心用电从兆瓦级升至百兆瓦甚至吉瓦级。
    - 适用：背景论述数据。

---

## 三、数据注释区

> 阶段 0.3 清洗完成后记录（2026-08-07）。清洗详情见 `solution/data-cleaning-03.md`。

### 数据表逐列注释

**workload_trace.xlsx**（50,000 任务，0 剔除，100% 保留）
- TaskID：任务唯一编号，无重复。
- TaskType：RealTimeInference(16724)/BatchInference(16717)/AITraining(16559)，三类均衡。
- ArrivalHour：0–2399 全覆盖，每小时均有任务到达。
- EarliestStartHour：恒等于 ArrivalHour（任务不得提前开工）。
- GPU_Demand：1–127，均值 29.5、中位 13 —— 长尾右偏，少数大任务（>100 GPU）需注意。
- EstimatedDuration_min：10–399，均值 204min(≈3.4h) —— 跨小时重叠普遍，GPU-hour 折算关键。
- MaxLatency_ms：20–150；实时推理 20ms 级，训练 150ms。
- LatestFinishHour：1–2406，=2406 占 66.6%（训练+大部分批量）—— 弹性时移空间大。
- SourceRegion：6 区域全覆盖；A 最多(10062)、F 最少(6707)。
- ExecutionMode：全部 NonPreemptive（与"不可抢占"一致）。

**region_time_data.xlsx**（14,442 = 2407h × 6 区域全网格，0 剔除）
- Hour：0–2406；DataPeriod 区分 Main_0_2399 与末端。
- 电价：Valley 234–491 / Flat 325–756 / Peak 523–1096 元/MWh；西部(D/E/F)整体低于东部(A/B/C)。
- CarbonIntensity：0.196–0.688 tCO2/MWh；东部高(0.59–0.69)、西部低(0.20–0.45)。
- AvailableRenewable：500–1100 MW，各区域各小时均有（非零）。
- Curtailment：均值 537 MW（占可用 ~67%）—— 弃电率极高，东部 A/B/C 无外送能力是主因。
- GPU_Utilization_Percent：**44 条 >100%（max 135.58，集中 RegionE/F）—— 基准异常观测，不做自动删除，供人类判断；后续建模建议自算利用率而非直接引用**。
- Baseline_AI_IT_Load + NonAI_IT_Load = IT_Load（max 差 0.0000）；Total_Load = IT_Load×PUE（max 差 0.0001）—— 口径自洽。
- 功率平衡式残差 max 0.0002 —— 基准自洽。
- SOC_MWh/ChargePower/DischargePower/GridPurchase 等为基准运行结果（问题三比较对象）。

**GPU_information.xlsx**（6 区域）
- Total_GPU：600–1600；D 最大(1600)、C 最小(600)。
- Reserved_GPU_Ratio：0.08–0.1；Available_GPU = Total×(1−Ratio) 精确自洽（max 差 0.00）。
- Available_GPU == Max_Workload_GPUh_per_h（每小时 GPU-hour 上限）。
- PUE：1.25–1.38；E/F 最低(1.25) —— 西部能效优势。

**storage_information.xlsx**（6 区域）
- StorageCapacity：300–900 MWh；InitialSOC = 45% 容量（统一）；MinSOC = 10% 容量（统一）。
- ChargeEfficiency 0.93–0.94、DischargeEfficiency 0.92–0.93（西部略优）。
- SellLimit：A/B/C=0（无外送）、D=180、E=220、F=?（与 MaxGridExport 一致）；无外送区域 GridSell=0 占 50% 印证 0 为真实值。
- MaxGridImport：340–550 MW；MaxGridExport：0–220 MW。

**network_latency.xlsx**（36 对）
- 同区 5ms、东部间 12–15ms、东西部 58–82ms、西部间 5–25ms；单向时延（数据对称，但不依赖对称性）。
- MaxLatency 约束：实时任务 20ms → 不可跨区（东西部 >58ms）；训练 150ms → 可迁至任意区域。

**power_mapping.xlsx**（3 类）
- AITraining 0.16 / BatchInference 0.10 / RealTimeInference 0.08 MW per 等效 GPU。

### 已知异常与处理决策
| 异常 | 观测 | 处理 |
|------|------|------|
| GPU_Utilization_Percent > 100% | 44 条（0.3%），RegionE/F 为主 | **人类决定（2026-08-07）：保留；论文引用为基准对比；建模自算利用率作约束** |
| 弃电率 ~67% | Curtailment 均值 537 MW | 属基准结果，非异常；人类决定：降低弃电率将来可单独做优化（先记录） |
| 无外送区域 GridSell=0 | 50% 行 | 真实值（区域无外送能力），不填补 |

### 建模口径（人类确认，2026-08-07）
- 任务时长折算：**精确小数**（GPU-hour = GPU_Demand × EstimatedDuration_min/60）。
- 实时推理可行域（S1 零迁移口径）：来源区域 ∪ 东部其余两区（A/B/C 间时延 12–15ms ≤ 20ms）；**不得迁往西部**；任务不可中途迁移、仅单向时延 ⇒ 无"往返"概念。
- **⚠️ 2026-08-08 S2 口径修正**：S1 的"不得迁往西部"仅适用于 **S1 零迁移场景**。S2 放开迁移后，实时推理可行域回归**数据驱动裁剪**（决策点 3：完全按 MaxLatency）：A/B/C 互迁（12–15ms）+ **E↔F 互迁（18ms ≤ 20ms）** + D 完全锁死（D→E/F=22/25ms > 20ms）。即实时推理可行域 = A/B/C 三区互迁 ∪ E/F 两区互迁 ∪ 各自本地。

### S1 决策记录（人类确认，2026-08-07；保留全部可选项供回溯）
**决策点 1（迁移边界）→ 选定 A：零跨区迁移（来源区域优先）**
- 数据依据：实时推理本地最大利用率 12.7%（永不超容量）；各区域 GPU-hour 本地负载率 28.9–48.7%；全局负载率 40.1%。
- 保留可选项：B（实时推理东部内迁——时延 12–15ms≤20ms 可行，但数据上无必要）、C（全自由迁移——S1/S2 区分度被稀释）。
- 推论：S1 调度自由度在**时间维**（训练/批量可时移，LatestFinishHour=2406 占 100%）；空间维迁移留给 S2。

**决策点 2（预测粒度）→ 选定双粒度：总 GPU 需求（1 序列）+ 分类型（3 序列）**
- 粗粒度做主模型预测；细粒度做统计描述。不选 18 条面板（6区×3类）：预测不反哺调度，细粒度纯增噪声。

**决策点 3（调度目标）→ 选定可行性 + GPU 利用率均衡双目标**
- 满足 GPU 容量/时延/完成时限约束前提下，最小化区域间利用率差异。

### 排除记录
- 无任何表被整体排除；候选池覆盖率 100%。
- 无领域排除（六个区域均为有效研究对象，无临床/工程环境异质子群体）。

### S1 数据预处理记录（阶段 1.4，2026-08-07）
- 产出：`outputs/data/s1-preprocessed.pkl`（结构见 `solution/model-notes/preprocess-sub1-20260807.md`）
- 预测目标：逐时 GPU 需求（Total + AITraining/BatchInference/RealTimeInference 三类型，2400h）
- 预测切分：train [168,2351] 2184 样本 / val [2352,2375] 24 / retrain [168,2375] 2208 / test [2376,2399] 24（滞后特征预热 168h）
- 预测特征：hour + sin/cos(24h) + sin/cos(168h) + lag1/lag24/lag168 + ma24
- 调度输入：测试窗 538 任务（160 实时固定 + 378 自由）；实时 base 峰值东部 37–41.6 GPU 远低于容量（F3 复验）
- 一致性：与 notebook（`outputs/notebooks/verify-sub1.ipynb`）同源同口径

### S2 决策记录（人类确认，2026-08-08；保留全部可选项供回溯）
**决策点 1（求解规模策略）→ 选定区域分解 + 时间维 MILP（复用 S1 框架），弃用启发式加速**
- 数据依据：零迁移下各区域 GPU 容量独立（F3）；任务迁移后仍只占一个目的地 → 6 区域子问题解耦
- 做法：轮次 1 贪心/EDF 选目的地（按成本+碳评分）→ 轮次 2 每区域独立时间维 MILP（S1 框架复用）→ 轮次 3 精确全时域
- 弃用混合方案理由：S4 带储能，启发式窗内低估储能价值会被 SOC 结转放大为全时域累积偏差
- 保留可选项：滚动窗口 MILP（区域 D 等大区域需要时）、纯启发式（毫秒级但非最优）

**决策点 2（四目标处理）→ 选定 ε-约束法：成本为目标、碳排放上限为约束、时延为硬约束、新能源利用率为评价**
- 理由：成本与碳方向一致（均迁往低电价低碳西部）→ 加权和退化为调 α/β 标量；ε-约束提供政策可解释性（减排 X% 的成本代价曲线）
- 碳排放上限三档敏感性（松/中/严）

**决策点 3（迁移边界）→ 选定完全按 MaxLatency 裁剪可行域 + 论文记录精确限制清单**
- 关键限制：批量推理 A→F=82ms>80ms 禁止（A→E=78 勉强保留）；实时推理西部来源锁死本地、东部仅 A/B/C 内迁；训练全图可迁
- 自动缩小约 10% 变量空间
- **⚠️ 2026-08-08 实时推理精确可行域修正**：数据驱动裁剪下实时推理可达性为——A/B/C 三区互迁（12–15ms）、**E/F 两区互迁（E→F=F→E=18ms ≤ 20ms）**、D 完全锁死（D→E=22ms、D→F=25ms、D→A/B/C=58–65ms 均 >20ms）。E/F 实时推理可互迁（影响 1146 个任务 × 2 候选）；D 锁死是时延矩阵的自然结果（无任何区域可迁往 D）。S1 记录"实时推理不得迁往西部"仅限 S1 零迁移场景。

**决策点 4（S1 基线）→ 选定必须做：S1 全时域零迁移回测作基线，量化迁移纯收益**
- 做法：S1 零迁移 MILP 在 5 万任务全时域跑（区域分解后小区域直接 MILP、大区域滚动窗）
- 对比表：成本/碳排/时延/新能源利用率 4 指标迁移收益

### S2 数据预处理记录（阶段 1.4，2026-08-08）
- 产出：`outputs/data/s2-preprocessed.pkl`（结构见 `solution/model-notes/preprocess-sub2-20260808.md`）
- P1 可行域：训练 36/36 全图可达；批量 34/36（A→F=82ms>80ms 禁）；实时 14/36（488 锁死=西部来源）
- P2 任务候选：平均 4.85 个/任务；锁死 1.0%
- P3 电力参数：逐小时电价/碳/新能源/PUE/Cap（0–2399）
- P4 S1 基线：零迁移负载 28.9–48.7%（迁移收益对比基础）

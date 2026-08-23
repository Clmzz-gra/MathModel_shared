# S3 智能体接管说明（handoff-sub3-agent）

> 交接方：建模智能体（S1/S2 阶段负责人） | 接收方：S3 专职智能体 | 2026-08-08 | 分支：experiment/sub3
> 用途：S3（问题三 储能协同优化）方案探索阶段已推进至 1.1（A 类验证完成），由本说明接管后续全部工作。
> 依据文件（只读，勿改）：`problems/2026-C` 分支已合并的 S1 产物 + 本文件引用的赛题原文/数据缓存。

---

## 0. 任务与并行上下文

| 项 | 内容 |
|----|------|
| 子问题 | S3：储能协同优化模型（问题三） |
| 赛题原文 | 在给定任务调度和逐时负荷的基础上，建立储能协同优化模型，设计储能充放电策略，分析储能系统对**运行成本、碳排放、区域峰值净购电功率和负荷波动程度**的影响 |
| 官方口径 | 说明 4：以 `Baseline_AI_IT_Load + NonAI_IT_Load` 为给定 IT 负荷（设施负荷 = IT×PUE），**仅优化储能充放电、购售电及新能源分配**，不优化任务调度；说明 5：SOC(2406) ≥ InitialSOC 硬约束；说明 6：0–2399 主时域 + 2400–2405 结清 + 2406 状态结算 |
| 并行状态 | S2（experiment/sub2，worktree E:\MathModel_pj-2026-C）正由代码智能体运行 sub2-model.py，**与本 S3 无数据依赖**（问题三口径固定），两者物理隔离互不干扰 |
| 工作区 | **E:\MathModel_pj-2026-C-sub3**（独立 worktree，checkout experiment/sub3，从 problems/2026-C 切出） |
| 验收终点 | S3 内部报告（iter-XX-sub3-storage.tex，STATUS=done）→ 阶段 2.4 合并回 problems/2026-C |

## 1. 数据接口（输入，均为缓存，禁止重解析原始 xlsx）

统一从 `outputs/data/csv/` 读取（阶段 0.2 转缓存产物，已在 problems/2026-C 提交）：

| 文件 | 关键列 | 用途 |
|------|--------|------|
| `csv/region_time_data/region_time_data.csv` | Hour(0-2406)/Region/PricePeriod/ElectricityPrice_CNY_per_MWh/SellPrice_CNY_per_MWh/CarbonIntensity_tCO2_per_MWh/AvailableRenewable_MW/UsedRenewable_MW/RenewableCharge_MW/Curtailment_MW/IT_Load_MW/Total_Load_MW/GridPurchase_MW/GridSell_MW/NetGridImport_MW/CarbonEmission_tCO2/SOC_MWh/ChargePower_MW/DischargePower_MW/Baseline_AI_IT_Load_MW/NonAI_IT_Load_MW | 逐时电价/碳/新能源/负荷/储能基准 |
| `csv/storage_information/storage_information.csv` | StorageCapacity_MWh/MinSOC_MWh/InitialSOC_MWh/MaxChargePower_MW/MaxDischargePower_MW/ChargeEfficiency/DischargeEfficiency/SellLimit_MW/MaxGridImport_MW/MaxGridExport_MW | 储能参数（SOC 为时段末；InitialSOC 为 Hour 0 前） |
| `csv/GPU_information/GPU中心基础情况.csv` | PUE/Max_IT_Power_MW/Max_Facility_Power_MW | 设施负荷折算与功率上限 |

统一口径（赛题附件 1）：
- 功率平衡：`GridPurchase + AvailableRenewable + DischargePower = Total_Load + ChargePower + GridSell + Curtailment`
- SOC 递推：`SOC(t) = SOC(t-1) + eta_c·Charge(t) − Discharge(t)/eta_d`
- 成本 = `Σ(GridPurchase×Price − GridSell×SellPrice)`；碳排 = `Σ GridPurchase×CarbonIntensity`
- 新能源利用率 =（直接消纳 + 新能源充电 + 新能源外送）/ 可用新能源累计

## 2. 已完成工作（截止交接）

### 阶段 1.0 知识检索（✅ 已确认）
- 深度问题重述、题型判定（**优化类**）、四维知识卡激活、AI 贡献标记已落盘：
  - `solution/model-notes/decision-log-sub3.md`（题型判定日志）
  - `.trae/ai-markers/stage-10-knowledge-retrieval-sub3.md`（🔴 激活 MS-018/MS-025/PR-001/WF-001）
- 候选方案预览（A 储能优化 LP / B 基准轨迹分析 / C 双目标 Pareto）已呈报，人类确认后进入 1.1

### 阶段 1.1 方案决策树（进行中：A 类验证已完成，报告与决策树未写）
- A 类验证脚本 `outputs/scratch/verify-sub3.py`（已运行 exit 0，可复现）
- **验证报告 `verify-sub3-20260808.md` 尚未编写**（下一项任务）

## 3. A 类验证共享事实（verify-sub3.py 实测，2026-08-08）

### F1 无储能简单基线（主时域 0–2399，三口径）
| 口径 | 规则 | 成本 | 碳排 | 峰值净购电 |
|------|------|------|------|-----------|
| a 零新能源利用 | GridPurchase=Total_Load（上界） | 3353.91 M 元 | 2885.71 kt | 651.5 MW |
| b 全消纳理想 | GridPurchase=max(Load−Renewable,0) | 0（**现实中不可达**） | 0 | 0 |
| c 直消纳代理 | GridPurchase=Load−UsedRenewable（去掉储能的诚实基线） | 2387.86 M 元 | 2108.00 kt | 479.0 MW |

### F2 基准运行状态（region_time_data 自带轨迹）
- 合计：成本 **1802.34 M 元**、碳排 **2045.36 kt**、峰值净购电 497.0 MW
- **RegionE 成本为负（−25.70 M 元）**——卖电收入超购电成本
- 基准 vs 无储能(口径c)：成本 −24.5%、碳 −3.0%（储能净贡献量级）
- 碳排重算与数据列 `CarbonEmission_tCO2` 完全一致（口径自洽 ✓）

### F3 储能参数自洽性
- Init/容量 = 45%、Min/容量 = 10%（6 区域一致）✓
- SOC 递推残差 max 0.0001（**RegionE Hour0 例外 0.9999**，数据小异常，建模以赛题递推式为准并记录）
- 充/放功率超限 0、同刻充放 0 ✓
- **⚠️ SOC(2406) ≥ Initial 违规**：RegionD 226.0<405、RegionE 217.2<370、RegionF 189.2<382.5；A/B/C 合规。**基准轨迹自身不满足赛题终态约束**——不能直接作为合规对照

### F4 同时充放 = 0 h（6 区域全 0）
→ LP 无需显式互斥约束（成本目标下天然排除），可先建模为纯 LP

### F5 功率平衡自洽性：残差 max 0.0001–0.0002 MW ✓

### F6 新能源利用率基准（直供+充电口径）
- 全局 **20.83%**；A 8.47% / B 9.97% / C 11.79% / D 24.76% / E 35.98% / F 34.02%
- 含 GridSell 口径：D 46.96% / E 60.92% / F 58.33%（A/B/C 无外送不变）
- 弃电率：A/B/C ~88–92%、D 53%、E/F ~39–42%

### F7 新能源消纳可行性（关键建模事实）
- **`AvailableRenewable` 六区域完全同分布**（mean 800、min 500、max 1100 MW）——**共享可用序列**，非区域专属
- 直消纳率（used/avail）东低西高：A 7.6% → F 33.3%；直消纳 ≤ 负荷 **100% 恒成立**
- 全局累计：可用 11.52 TWh、直消纳 2.19 TWh、新能源充电 0.21 TWh、**弃电 79.2%**

### F8 卖电套利事实
- A/B/C 无卖电（SellLimit=0）；**D/E/F 卖电 2400h 全覆盖**：D 426.3 GWh/140.90 M 元、E 478.8 GWh/141.82 M 元、F 466.8 GWh/146.44 M 元；SellPrice 均值 291–330 元/MWh
- 结清段 2400–2405：D/E 仍有卖电（净购电 max 为负），全区域无充放

## 4. 待裁定决策点（阶段 1.1 决策树核心，需人类拍板）

| # | 决策点 | 候选选项 | 影响 |
|---|--------|---------|------|
| D1 | **储能价值对比基准**（基准轨迹违反终态约束） | (a) 基准仅作参考下界并注明违规；(b) 构造"满足 SOC(2406)≥Initial 的对照解"；(c) 以无储能(口径c)为对照下界 | 论文"储能价值"叙事的基准 |
| D2 | **新能源消纳建模**（AvailableRenewable 共享+弃电 79%） | (a) LP 允许自由消纳（≤可用，赛题口径，利用率可大幅提升）；(b) 消纳受限（≤基准直消纳能力，保守）；(c) 消纳上限=min(Avail,Load)+储能充电 | 成本/碳排最优解量级、利用率评价 |
| D3 | **GridSell 卖电建模** | (a) 纳入成本目标（卖电收入，D/E/F 必须）；(b) 仅作评价不计目标 | RegionE 负成本现象、套利激励 |
| D4 | **目标函数形态** | (a) 延续 S2 ε-约束（成本目标+碳约束三档）；(b) 成本+碳加权；(c) 成本目标+碳排评价 | 与 S2/S4 口径一致性 |
| D5 | **负荷波动指标定义** | 净购电 std / 峰谷差 / 负荷 std（F2 已有数据） | 问题三 4 指标之一定义 |
| D6 | **RegionE SOC 递推 Hour0 残差 0.9999** | 数据小异常记录即可 / 需深查 | 建模以赛题递推式为准 |

## 5. 下一步任务清单（按序执行，每阶段结束停等人类确认）

1. **写 A 类验证报告**：`solution/model-notes/verify-sub3-20260808.md`（F1–F8 落盘，含口径与数据事实表）
2. **构建 1.1 决策树**：对比矩阵（7 维度）× 候选方案、决策树（含 D1–D5）、最大风险标注、A/B 类验证划分（B 类推迟 1.2）
3. **出口门禁**：人类确认决策树 → 1.2 辩论 / 直接 1.3
4. **1.3 方案确认书**：`approach-sub3-confirmed.md`（含 @PROXY 登记，参照 S1/S2 格式）+ 代理值替换清单
5. **1.4 数据预处理**：`outputs/data/s3-preprocessed.pkl` + `preprocess-sub3-20260808.md`（LP 输入面板：逐时电价/碳/负荷/新能源/储能参数/边界）
6. **1.5 代码审查**（TRAE-code-review skill）
7. **2.0 数学推导** `math-sub3.tex` → **2.1 代码** `outputs/scratch/sub3-model.py` → **2.2–2.4 报告**

## 6. 规范与约束（强制）

- **管线**：TRAE.md 阶段 0–2 规则；每阶段结束输出「已完成阶段 X，是否确认并进入下一阶段？」
- **git**：git-workflow skill——两段式 commit、阶段提交节点（1.1 决策树定稿、1.3 确认书签署、2.0 推导定稿、2.3 报告初稿）、S3 定稿后合并回 `problems/2026-C`
- **只读文件**：`solution/model-notes/` 下 S1 产物（approach-sub1-confirmed.md 等）、`solution/domain-knowledge.md`（S1 版）、赛题附件；S3 探索产物可自由新增
- **代码规范**：`outputs/scratch/` 下脚本五段式头注释（目的/原理/输入映射/输出/论文章节）；禁止重解析原始 xlsx（一律用 `outputs/data/csv/`）
- **求解器**：scipy（linprog/milp，HiGHS）——与 S1/S2 生态一致；S3 规模 ~7 万连续变量，LP 秒级可解，**无需启发式**
- **区域独立性**：6 区域无跨区潮流（说明 7），储能子问题天然解耦，可逐区独立求解
- **结清段**：2400–2405 仍需功率平衡与储能运行并计入结算；2406 仅 SOC 状态结算
- **新能源利用率**：S2 移交的正式口径在本子问题核销（直供+充电+外送）/可用；GridSell 是否计入"外送"需在确认书中明确（D3 关联）

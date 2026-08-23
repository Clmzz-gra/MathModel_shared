# S3 代码智能体交接说明：LP 求解实现（handoff-sub3-model-agent）

> 交接方：建模智能体 | 接收方：代码智能体 | 2026-08-08 | 分支：experiment/sub3
> 用途：按本说明实现 S3 阶段 2.1 正式模型代码 `outputs/scratch/sub3-model.py`，完成 LP 求解 + 出图 + 代理值核销 + Artifact 登记。
> 建模依据（只读，勿改）：`solution/model-notes/approach-sub3-confirmed.md`（方案确认书）、`solution/model-notes/math-sub3.tex`（数学推导，**以本文档为准**）、`solution/model-notes/verify-sub3-20260808.md`（A 类共享事实）、`solution/model-notes/proxy-replacement-checklist-sub3.md`（代理值清单）。

---

## 0. 交接概览

| 项 | 内容 |
|----|------|
| 子问题 | S3：储能协同优化（问题三） |
| 主方案 | 储能优化 LP：成本目标 + 碳排 ε-约束三档（ε∈{0.90,0.95,1.00}），6 区域独立、0–2405h 全时域 |
| 模型代码 | `outputs/scratch/sub3-model.py`（本交接交付物） |
| 数据输入 | `outputs/data/s3-preprocessed.pkl`（阶段 1.4 产物，直接读取） |
| 求解器 | `scipy.optimize.linprog`（method='highs'） |
| 结果落盘 | `outputs/data/cache/s3_solutions.pkl` + 控制台打印 |

## 1. 数据接口（输入）

读取 `outputs/data/s3-preprocessed.pkl`，6 键结构：

| 键 | 内容 |
|----|------|
| `meta` | generated/source/hours(0–2405)/units/pue |
| `panel` | 长表 index=(Region,Hour)：Price_CNY_per_MWh, SellPrice_CNY_per_MWh, CarbonIntensity_tCO2_per_MWh, Total_Load_MW, AvailableRenewable_MW, UsedRenewable_MW(参考), GridPurchase_base_MW(参考), NetGridImport_base_MW(参考) |
| `storage` | 每区域 10 项：Capacity/MinSOC/InitialSOC/MaxChargePower/MaxDischargePower/ChargeEfficiency/DischargeEfficiency/SellLimit/MaxGridImport/MaxGridExport |
| `carbon_base_kt` | 每区域基准碳排（kt，F2 主时域口径） |
| `epsilon` | [0.90, 0.95, 1.00] |
| `check` | 预处理自检（load_equals_IT_times_PUE 等，勿改） |

**中文指标 → 变量映射**：电价→Price_CNY_per_MWh、卖电价→SellPrice_CNY_per_MWh、碳强度→CarbonIntensity_tCO2_per_MWh、设施负荷→Total_Load_MW、可用新能源→AvailableRenewable_MW、容量→Capacity_MWh、SOC 下限→MinSOC_MWh、初始 SOC→InitialSOC_MWh、充/放电上限→MaxChargePower_MW/MaxDischargePower_MW、充/放电效率→ChargeEfficiency/DischargeEfficiency、卖电上限→SellLimit_MW、购电上限→MaxGridImport_MW。

## 2. 数学模型（math-sub3.tex 提炼，逐区域独立求解）

**索引**：$t=0..2405$；主时域 $t<2400$；2406 状态结算 $E_{2406}=E_{2405}$。

**决策变量（每区域，7×2406=16842 个）**：购电 $G_t$、卖电 $S_t$、新能源直供 $R_t$、电网充电 $C^g_t$、新能源充电 $C^r_t$、放电 $D_t$（均 $\ge 0$）、SOC $E_t$（$MinSOC \le E \le Cap$）。

**约束**：
- （C1）功率平衡（等式，2406 行）：$G_t + R_t + D_t = Load_t + C^g_t + S_t$
- （C2）新能源上限：$R_t + C^r_t \le Avail_t$（2406 行）
- （C3）SOC 递推（等式，2406 行）：$E_t - E_{t-1} - \eta_c(C^g_t+C^r_t) + D_t/\eta_d = 0$，$E_{-1}=InitialSOC$
- （C4）SOC 边界：$E_t \le Cap$、$E_t \ge MinSOC$；**终态 $E_{2405} \ge InitialSOC$**（2406 结算）
- （C5）充放功率：$C^g_t+C^r_t \le MaxChargePower$、$D_t \le MaxDischargePower$
- （C6）购售电边界：$G_t \le MaxGridImport$、$S_t \le SellLimit$（A/B/C=0）
- （C7）**碳 ε 约束（主时域 0–2399）**：$\sum_{t=0}^{2399} G_t \cdot CI_t \le 10^3 \cdot \varepsilon \cdot carbon\_base\_kt_r$（**注意 10³ 单位换算 kt→tCO₂**）

**目标（每区域）**：$\min \sum_{t=0}^{2405} (G_t \cdot Price_t - S_t \cdot SellPrice_t)$

## 3. 实现要点

1. **稀疏矩阵**：约束矩阵用 `scipy.sparse`（lil→csr）构建，7 变量块对角排布：`[G,S,R,Cg,Cr,D,E]` 各 2406 长
2. **逐区域 × ε 档**：6 区域 × 3 档 = 18 次 LP；`linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method='highs')`
3. **c 向量**：G 位置放 `Price`；S 位置放 `−SellPrice`；其余 0
4. **状态检查（强制）**：每解检查 `res.status == 0`，非 0 打印警告并输出该解（标注"非最优"）；打印 `res.fun` 成本
5. **结果提取**：从 `res.x` 切回 7 个时序数组，存每 (region, ε) 结果
6. **五段式头注释**（目的/原理/输入映射/输出/论文章节），原理字段先写清再编码
7. **统计量打印**：每数组 `min/max/mean/std`（PR-014）

## 4. 输出规格（每区域 ε 档 + 聚合）

落盘 `outputs/data/cache/s3_solutions.pkl`：
- 每 (region, ε)：`G/S/R/Cg/Cr/D/E` 时序 + 成本（主时域+全时域）、碳排（主时域）、峰值净购电、净购电 std、峰谷差、SOC(2406)、同刻充放小时数
- 聚合表：四指标对比（优化解 vs 无储能口径 c vs 基准轨迹参考）+ ε 档 Pareto 汇总（6 区域聚合成本/碳排）
- 新能源利用率双口径（含 GridSell 外送 / 不含）
- 拐角解统计：变量触及边界比例（阶段 2.2 用）

## 5. 对照基准（必须复现，偏差 >0.5% 暂停回查，勿自行"修正"）

| 对照 | 值 | 来源 |
|------|-----|------|
| 无储能口径 c：成本 | 2387.86 M 元（=Σ max(Load−Used,0)×Price） | F1 |
| 无储能口径 c：碳排 | 2108.00 kt | F1 |
| 无储能口径 c：峰值净购电 | 479.0 MW | F1 |
| 基准轨迹：成本 | 1802.34 M 元（=Σ(G_base×Price−S_base×SellPrice)） | F2 |
| 基准轨迹：碳排 | 2045.36 kt | F2 |
| 基准轨迹：峰值净购电 | 497.0 MW | F2 |
| 功率平衡残差（LP 解） | ≤1e-3 MW | 自检 |

**LP 解自洽性（必须全过）**：status=0；同刻充放小时数=0；SOC(2406)≥InitialSOC（硬约束）；ε 单调性：成本(0.90) ≥ 成本(0.95) ≥ 成本(1.00)（Pareto 梯度）。

## 6. 代理值核销（阶段 2.1 强制）

| @PROXY | 本任务处理 |
|--------|-----------|
| 新能源消纳上限 = 可用（自由消纳） | ✅ 主口径核销：C2 约束 R+Cr≤Avail 已实现；B1 受限消纳灵敏度列附录（可选第二轮） |
| GridSell 全量视为新能源外送 | ✅ 核销：利用率双口径计算（含外送/不含外送） |
| 负荷波动 = 净购电 std（主）+峰谷差（辅） | ✅ 核销：净购电 std/峰谷差已计算 |
| RegionE Hour0 残差不修 | ✅ 已由预处理核销（无需代码处理） |

## 7. 出图清单（chart-generator skill，输出 outputs/figures/ + manifest 登记）

| 图 | 类型 | 文件名 |
|----|------|--------|
| 6 区域 SOC 曲线 + 充放功率 | 2×3 子图 | `sub3-soc-charge-discharge.pdf` |
| 净购电时序对比（基准 vs 最优，削峰可视化） | 折线 | `sub3-net-import-compare.pdf` |
| ε 档 Pareto 前沿（成本 vs 碳排，6 区域聚合） | 散点+连线 | `sub3-pareto-frontier.pdf` |
| 四指标对比（优化 vs 无储能 vs 基准） | 分组柱状 | `sub3-four-metrics.pdf` |
| 峰值削峰效果（区域峰值净购电 6 区域） | 柱状 | `sub3-peak-shaving.pdf` |

格式硬约束（chart-generator）：PDF 矢量、SimHei 中文字体、去饱和配色、线宽 1.5–2pt、`outputs/figures/` 输出、`solution/artifacts/charts/` 副本 + `solution/artifacts/manifest.md` 登记。

## 8. Artifact 登记 + 附录归档

- `solution/artifacts/manifest.md`：登记所有图 + ≤15 行核心代码片段（`solution/artifacts/code-snippets/`）
- `solution/appendix/code/`：`sub3-model.py` 定稿副本
- `outputs/data/cache/s3_solutions.pkl` 不随附录提交（中间产物）

## 9. 代码自检清单（交付前）

- [ ] 读入 s3-preprocessed.pkl，未重解析 csv/xlsx
- [ ] 18 次 LP 全部 status=0（或明确标注非最优）
- [ ] C1–C7 全部实现且与 math-sub3.tex §2–§4 逐条一致（尤其碳约束 10³ 换算、终态 E_2405≥Initial）
- [ ] 对照基准复现（§5 表，偏差 ≤0.5%）
- [ ] 同刻充放=0、SOC(2406)≥Initial、ε 单调性 三项自洽全过
- [ ] 出图走 chart-generator（SimHei/PDF/去饱和/manifest）
- [ ] 代理值 3 项核销完成
- [ ] 代码头注释五段式完整
- [ ] 未修改 `solution/model-notes/` 任何文件

## 10. 交付后回报

向建模智能体回报：18 次 LP 状态汇总、四指标聚合表（三对照）、ε Pareto 汇总、自洽性三项结果、出图清单状态、任何偏差项。

## 11. 建模裁定增补（2026-08-08，R1 风险触发，务必执行）

主口径 LP 触发确认书 R1 风险（自由消纳退化：G≈0、ε 同解、Pareto 单点）。建模裁定如下：

**裁定 1 — 受限消纳主口径（B1 定稿，替代 C2 自由消纳）**：
- C2 改为（逐时，R+Cr 总限）：
  $$R_{r,t} + C^{r}_{r,t} \le UsedRenewable_{r,t} + RenewableCharge_{r,t},\quad \forall r,t$$
- 面板需补充 `RenewableCharge_MW` 列（region_time_data 现有，预处理未入 panel → **在 sub3-model.py 内直接从 pkl 的 panel 无法取到，需在脚本内回读 csv 或由你从 region_time_data.csv 读取补充**；更稳妥：读 `outputs/data/csv/region_time_data/region_time_data.csv` 取 UsedRenewable 与 RenewableCharge 两列构造上限列，加入解算面板，禁止改 pkl）
- 明确：逐时限制（非全局累计）；Cr 单独不受限（总限内自由重排直供↔充电）；`UsedRenewable`/`RenewableCharge` 用数据列值（基准观测的消纳能力）
- ε=1.00 档（碳约束）与 SOC(2406)≥Initial 等其余约束不变

**裁定 2 — 碳 ε 两阶段标定**：
1. 先跑 **ε=1.00 单档 × 6 区域**（受限消纳），回报：每区域最优成本/碳排 C*（主时域）、峰值净购电、净购电 std、SOC(2406)、**卖电构成分解**（卖电电量 vs 新能源充电量 vs 电网购电量）
2. 依据 C* vs 基准碳排（2045.36 kt）由建模侧定最终档位集（绑定→90/95/100；不绑定→85/90/95），再全档重跑
3. **卖电构成观察**：若 D/E/F 负成本明显且卖电主要由"电网购电→卖电"套利驱动（Price<SellPrice 时段），回报时序明细，建模侧将裁定是否追加"卖电 ≤ 新能源充电+新能源富余"约束

**重跑与出图**：确认后实现 `--b1` 受限消纳参数化（默认开启），全档重跑 + 出图 v2（四指标对比/削峰/Pareto），覆盖 §7 图表清单；若 ε 档位变化导致 Pareto 图语义变更，出图标注最终档位。

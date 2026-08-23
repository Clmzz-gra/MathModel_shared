# 交接：S2 A 类共享事实验证（建模 → 代码）

> handoff_type: `code-agent-verify`（A 类验证交接，非正式建模交接）
> sub: S2（特征选择与生物标志物）
> 日期：2026-08-21 | 阶段：1.1 方案决策树（提前并行）
> 上游：`solution/model-notes/decision-tree-S2.md`（A 类验证清单）
> 数据接口：`outputs/data/B-raw.pkl`（见下"数据接口"节）
> ⚠️ 本交接为**只读探索验证**：不写正式代码、不改共享数据、不产正式图。产物为探索图 + 数值 + 解读，回报 `handoff-S2-model-agent-verify.md`。

---

## 1. 验证目标（A 类共享事实）

验证下列"数据特征/约束可行性/基本假设"事实，供建模对话在 1.2 方案辩论前裁决。**全部基于 `B-raw.pkl` 只读计算**。

| # | 事实问题 | 目的 | 决策影响 |
|:--|:--|:--|:--|
| V1 | 单特征判别力的下界（基线） | 建立"至少单特征可区分"的性能下界，防过度设计 | 特征选择是否"小题大做" |
| V2 | 零值占比分箱对各候选方法（Wilcoxon p/L1 入选/RF 重要性/VIP）选中率的影响 | 量化"零值主导 → 方法失效"边界 | 是否需要先过滤近全零特征；各方法适用域 |
| V3 | 特征间冗余度（Spearman 相关簇） | 高冗余 → Lasso 共线组内任选 / RF 重要性分散 | 主方法（Lasso）是否受共线干扰；是否需先聚类去冗余 |
| V4 | 三数据集特征重叠度 + 已知标志物检出率 | 为"跨疾病标志物对比"与 S3 迁移预判铺底 | 跨疾病共同标志物是否存在 |
| V5 | 原始丰度 vs CLR 的伪相关差异 | 确认定和成分数据是否需要 CLR 前置（PR-006/MS-011 铁律） | 各方法是否必须先 CLR |
| V6 | Lasso bootstrap 频率分布是否呈"稳定簇" | 检验"多轮聚合 → 稳定高频率簇"假设是否成立 | 稳定选择阈值 τ 是否可行 |

---

## 2. 数据接口

- **数据文件**：`outputs/data/B-raw.pkl`
  - 结构（来自 `inventory-B.txt`）：484 行（样本）× 1333 列（2 元数据 + 1331 特征）
  - 元数据列：`dataset_name`、`disease`
  - 特征列：1331 个物种相对丰度特征（名含 `k__p__c__o__f__g__s__` 分类学层级），0-100 量级，每行和 ≈100
- **标签口径**（`solution/problem-statement.md`）：每数据集二分类
  - CRC(Zeller_fecal_colorectal_cancer)：`cancer`=患病，其余（`n` + `small_adenoma`）=健康对照（**small_adenoma 口径待 [B级] 裁定**，见 gate-0.1）
  - IBD(metahit)：`ibd_ulcerative_colitis`/`ibd_crohn_disease`=患病，`n`=健康
  - Obesity(Chatelier_gut_obesity)：`obesity`=患病，`leaness`=健康
- **样本量/类别**：CRC 121(病48/健73)、IBD 110(病25/健85)、Obesity 253(病164/健89)

---

## 3. 验证规格（每项：方法 / 输出 / 解读要求）

> **第一步强制为简单基线 V1**，先建立下界再谈方法。V2-V6 相互独立，可并行。

### V1【基线】单特征 Wilcoxon 判别力下界
- **方法**：对每病，每特征病 vs 健做 Wilcoxon 秩和（p 值）；BH-FDR 校正；同时算零值占比差与非零丰度差。给 FDR 显著特征数、Top 单特征 AUC 量级（用 scikit-learn `roc_auc_score`）。
- **输出**：Top 单特征列表 + 散点/箱线图（病 vs 健，1-2 个代表特征）。
- **解读**：零值占比差 vs 非零丰度差哪个主导"差异"；单特征最高 AUC 约多少（判断特征选择的上限预期）。

### V2 零值占比分箱影响
- **方法**：将每病特征按零值占比分箱（如 0-20/20-40/40-60/60-80/80-100%）；各箱统计：Wilcoxon 显著率、被 L1(带惩罚) 选中率（可用 `LogisticRegression(penalty='l1')` 快速评估）、RF 重要性 top-占比、VIP 占比。
- **输出**：分箱→选中率柱状图/热图；"方法失效零值边界"数值。
- **解读**：哪个零值占比以上方法失效；是否需过滤近全零特征（如去掉零值>95% 的特征后维度降到多少）。

### V3 特征冗余度
- **方法**：抽样（如每病 300 随机特征或全特征）对非零样本算 Spearman 相关矩阵；|ρ|>0.7 连边构建图，数连通簇数与最大簇规模。
- **输出**：相关簇规模分布图 + 最大簇成员示例。
- **解读**：冗余是否严重 → Lasso 是否受共线组任选影响；是否需先聚类去冗余。

### V4 三数据集特征重叠 + 已知标志物
- **方法**：三数据集非零特征集合两两 Jaccard 重叠；domain-knowledge 已知标志物（Fusobacterium nucleatum 属、Faecalibacterium prausnitzii、Bifidobacterium 属等）在各数据集的检出率（非零样本比例）。
- **输出**：维恩图/Jaccard 矩阵；已知标志物检出率表。
- **解读**：共同存在的物种多不多（跨疾病可比性）；已知标志物在哪个数据集强检出。

### V5 CLR 前置必要性
- **方法**：抽 50-100 对特征，比较原始丰度 vs CLR 后（用 sklearn/pseudocount+`scipy` 或自实现 CLR）的 Pearson/Spearman 相关；统计相关方向/强度改变的特征对比例。
- **输出**：原始 vs CLR 相关散点图（每点一对特征）。
- **解读**：定和成分导致的伪相关是否显著 → 是否必须 CLR 前置。

### V6 稳定性频率分布
- **方法**：对每病，B=50-100 次 bootstrap（分层抽样病/健），每次在重采样上拟合 Lasso 稀疏 Logistic，统计每特征入选频率；画频率直方图 + 高频特征 Top 列表。
- **输出**：频率直方图（是否双峰/高频率稳定簇）+ Top 稳定特征。
- **解读**：高频率簇是否真实存在；τ=80% 阈值是否落在自然间断点；若频率平缓需提示建模调阈值。

---

## 4. 待产探索图清单（回报时随附解读）

| 图 | 来源验证 | 文件建议名（`outputs/scratch/`） |
|:--|:--|:--|
| Top 单特征病/健分布（1-2 特征） | V1 | `S2-v1-top-single-feature-dist` |
| 零值分箱→选中率 | V2 | `S2-v2-zerobin-selection-rate` |
| 特征相关簇规模分布 | V3 | `S2-v3-correlation-clusters` |
| 三数据集特征重叠（维恩/Jaccard） | V4 | `S2-v4-dataset-overlap` |
| 已知标志物检出率表 | V4 | `S2-v4-known-biomarker-presence` |
| 原始 vs CLR 相关 | V5 | `S2-v5-clr-correlation` |
| Lasso bootstrap 频率直方图 | V6 | `S2-v6-stability-frequency` |

---

## 5. 约束与回报

- **只读**：不修改 `B-raw.pkl`，不改共享数据；探索脚本命名 `verify-S2-*.py`，输出落 `outputs/scratch/`。
- **性能**：V2/V6 涉及重复拟合，若预计单次 >2 分钟，按 TRAE-规范 C4 交主会话后台执行并心跳；脚本先 ≤2 分钟冒烟自测。
- **诚实标注**：指标区分"全量计算（乐观）"与"折内/CV（诚实）"；V1 基线 AUC 用分层抽样给诚实估计。
- **回报**：写 `handoff-S2-model-agent-verify.md`（每项事实：方法 + 数值 + 探索图 + 解读 ≤3 行），git commit。

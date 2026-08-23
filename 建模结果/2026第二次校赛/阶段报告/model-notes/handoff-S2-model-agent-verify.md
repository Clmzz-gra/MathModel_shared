# 交接：S2 A 类共享事实验证回报（代码 → 建模）

> handoff_type: `model-agent-verify`（A 类验证回报，非正式建模交接）
> sub: S2（特征选择与生物标志物）
> from: 代码对话 | to: 建模对话
> 日期：2026-08-21 | 阶段：1.1 方案决策树（提前并行）
> 上游：`solution/model-notes/handoff-S2-code-agent-verify.md`（验证规格）
> 数据接口：`outputs/data/B-raw.pkl`（只读，未修改）
> 验证脚本：`outputs/scratch/verify-S2-v1~v6-*.py` + `utils.py`（C1 头注释全量）
> 探索图：`outputs/figures/_explore/S2-v*-explore.pdf`（7 张，均不入论文）
> status: ready
> next_action: 建模对话 1.2 方案辩论（基于 A 类验证结论裁决主方法/阈值），说「继续」

---

## 0. 结论速览（供 1.2 方案辩论裁决）

| 事实 | 结论 | 对方案的影响 |
|:--|:--|:--|
| V1 基线 | 单特征 AUC 下界 0.64~0.82；**存在/缺失信号主导判别力** | 特征选择非"小题大做"，但信号语义须拆"存在/缺失 vs 丰度"两路 |
| V2 零值分箱 | 近全零特征（>95% 零值）占 **1067/1331**，过滤后仅 **264** 维 | **必须先过滤近全零特征**；VIP>1 阈值过宽需调 |
| V3 冗余度 | 高相关边仅 21~30 条/300 特征，最大簇规模 4 | **冗余低**，Lasso 共线干扰不严重，无需聚类去冗余 |
| V4 重叠+标志物 | 三数据集 Jaccard 重叠 0.64~0.79；Fusobacterium nucleatum 在 CRC 检出率最高(0.182) | 跨疾病共享物种多，可比性强；已知标志物可作方法有效性锚点 |
| V5 CLR | CLR 使 23~34% 特征对相关**方向翻转** | **定和伪相关显著，必须 CLR 前置**（PR-006/MS-011 铁律成立） |
| V6 稳定性 | 频率≥0.8 仅 **2 特征/病**，长尾连续非强双峰 | τ=0.8 过严；稳定簇小，需调 τ 或改"Top-k 频率排序" |

---

## 1. V1【基线】单特征 Wilcoxon 判别力下界

**方法**：每病每特征病 vs 健 Wilcoxon 秩和 + BH-FDR；单特征 AUC（max(AUC,1-AUC)）；零值占比差 vs 非零丰度差与 AUC 的相关。

| 数据集 | FDR 显著特征数 | Top 单特征 AUC | Top 特征 | 零值占比差 corr(AUC) | 非零丰度差 corr(AUC) |
|:--|:--|:--|:--|:--|:--|
| CRC | 7 / 1331 | **0.758** | Peptostreptococcus_stomatis | 0.885 | 0.674 |
| IBD | 3 / 1331 | **0.815** | Alistipes_finegoldii | 0.893 | 0.632 |
| Obesity | 0 / 1331 | **0.639** | Ruminococcus_flavefaciens | 0.860 | 0.668 |

**图**：`S2-v1-top-single-feature-dist-explore.pdf`（每病最优单特征病/健 log 丰度箱线图）

**解读**：
- 单特征判别力下界 AUC 0.64~0.82，CRC/IBD 有明确单特征信号，Obesity 弱（符合 domain-knowledge 预期 AUC 0.65-0.75）。
- **零值占比差（存在/缺失）主导判别力**（corr 0.86~0.89 > 非零丰度差 0.63~0.67）——92% 零值下"差异"主要来自"这个物种在不在"，而非"丰度高低"。→ 解释层必须拆两路信号（Fisher 存在/缺失 + 非零丰度 Wilcoxon），与 decision-tree §4 风险 2 一致。
- CRC Top 特征 Peptostreptococcus_stomatis 即 domain-knowledge 已知 CRC 标志物，方法有效性正面验证。

---

## 2. V2 零值占比分箱 → 各方法选中率

**方法**：特征按零值占比分 5 箱（0-20/20-40/40-60/60-80/80-100%），各箱统计 Wilcoxon(FDR<0.05)/L1(CLR+标准化 Lasso 非零系数)/RF(top-20)/VIP(>1) 选中率。

| 数据集 | 方法 | 0-20% | 20-40% | 40-60% | 60-80% | 80-100% |
|:--|:--|:--|:--|:--|:--|:--|
| CRC | Wilcoxon | 0.00 | 0.00 | 0.02 | 0.05 | 0.01 |
| | L1 | 0.10 | 0.06 | 0.00 | 0.03 | 0.02 |
| | RF | 0.21 | 0.15 | 0.07 | 0.06 | 0.01 |
| | VIP | 0.55 | 0.33 | 0.47 | 0.40 | 0.46 |
| IBD | Wilcoxon | 0.00 | 0.08 | 0.00 | 0.02 | 0.00 |
| | L1 | 0.04 | 0.12 | 0.00 | 0.08 | 0.01 |
| | RF | 0.23 | 0.12 | 0.03 | 0.08 | 0.00 |
| | VIP | 0.51 | 0.52 | 0.45 | 0.51 | 0.39 |
| Obesity | Wilcoxon | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| | L1 | 0.16 | 0.25 | 0.10 | 0.14 | 0.08 |
| | RF | 0.27 | 0.21 | 0.07 | 0.00 | 0.00 |
| | VIP | 0.10 | 0.25 | 0.14 | 0.12 | 0.04 |

**图**：`S2-v2-zerobin-selection-rate-explore.pdf`

**解读**：
- **L1/RF 明显偏好低零值特征**（0-20% 箱选中率最高，随零值占比升高单调下降）——零值主导会稀释 L1/RF 的判别信号。
- **VIP>1 阈值过宽**：选中率 40~55% 且跨箱平坦，不依赖零值占比——VIP>1 在本数据下会选中近半特征，**阈值需上调**（如 VIP>1.5 或按分位数）。
- **过滤近全零特征收益巨大**：零值占比>95% 的特征 1067 个，过滤后维度 **1331 → 264**（降 80%），且几乎不损失判别信息（这些特征 Wilcoxon 选中率≈0）。

---

## 3. V3 特征冗余度

**方法**：每病抽样 300 低零值特征，Spearman 相关（含零值），|ρ|>0.7 连边，并查集数连通簇。

| 数据集 | 高相关边数 | 连通簇数 | 最大簇规模 | 最大簇成员示例 |
|:--|:--|:--|:--|:--|
| CRC | 21 | 283 | 4 | Saccharomycetaceae/Naumovozyma/Debaryomycetaceae/Eremothecium（酵母类） |
| IBD | 30 | 274 | 4 | Veillonella 属 4 种 |
| Obesity | 21 | 285 | 4 | Veillonella 属 4 种 |

**图**：`S2-v3-correlation-clusters-explore.pdf`

**解读**：
- **冗余度低**：300 特征中仅 21~30 条高相关边，最大簇规模仅 4（且多为同属近缘种，如 Veillonella 属）。
- → **Lasso 共线组内任选问题不严重**，RF 重要性分散风险低，**无需先聚类去冗余**。decision-tree 中"是否需先聚类去冗余"的担忧可排除。

---

## 4. V4 三数据集特征重叠 + 已知标志物检出率

**方法**：三数据集非零特征集合两两 Jaccard；已知标志物按种/属名匹配特征名，检出率=非零样本比例。

**Jaccard 重叠矩阵**：

| | CRC | IBD | Obesity |
|:--|:--|:--|:--|
| CRC | 1.000 | 0.645 | 0.643 |
| IBD | 0.645 | 1.000 | 0.787 |
| Obesity | 0.643 | 0.787 | 1.000 |

**已知标志物检出率**：

| 标志物 | CRC | IBD | Obesity |
|:--|:--|:--|:--|
| Fusobacterium nucleatum | **0.182** | 0.055 | 0.020 |
| Faecalibacterium prausnitzii | 1.000 | 0.991 | 1.000 |
| Bifidobacterium (属) | 0.884 | 0.936 | 0.945 |
| Peptostreptococcus stomatis | **0.273** | 0.055 | 0.016 |
| Parvimonas micra | **0.331** | 0.036 | 0.024 |
| Porphyromonas (属) | 0.157 | 0.100 | 0.095 |
| Bacteroides fragilis | 0.620 | 0.536 | 0.542 |

**图**：`S2-v4-dataset-overlap-explore.pdf`、`S2-v4-known-biomarker-presence-explore.pdf`

**解读**：
- **跨疾病共享物种多**（Jaccard 0.64~0.79），三数据集物种层面高度可比，为 S3 跨疾病迁移铺底（共享特征多 → 迁移有基础，但批次效应仍是主要衰减源）。
- **Fusobacterium nucleatum 在 CRC 检出率最高（0.182）**，且 IBD/Obesity 极低（0.055/0.020）——CRC 特异性标志物，与文献共识一致，可作方法有效性锚点。
- Faecalibacterium prausnitzii 三数据集检出率≈1.0（普遍存在），其"减少"信号需看丰度而非存在/缺失——印证 V1"存在/缺失 vs 丰度两路信号"的必要性。

---

## 5. V5 CLR 前置必要性

**方法**：每病抽样 100 对低零值特征，比较原始丰度 vs CLR 的 Pearson/Spearman 相关，统计方向翻转与强度改变比例。

| 数据集 | Pearson 方向翻转 | Spearman 方向翻转 | Pearson \|Δρ\|>0.3 | Spearman \|Δρ\|>0.3 |
|:--|:--|:--|:--|:--|
| CRC | 0.34 | 0.23 | 0.03 | 0.02 |
| IBD | 0.32 | 0.27 | 0.06 | 0.10 |
| Obesity | 0.34 | 0.33 | 0.05 | 0.10 |

**图**：`S2-v5-clr-correlation-explore.pdf`（原始 vs CLR 相关散点，对角线=无变化）

**解读**：
- **CLR 使 23~34% 特征对相关方向翻转**——定和约束（某物种丰度升高挤压其他物种）在原始丰度上制造了显著伪相关，方向不可信。
- 强度改变（|Δρ|>0.3）比例较低（2~10%），说明 CLR 主要修正**方向**而非**强度**。
- → **必须 CLR 前置**（PR-006/MS-011 铁律成立），尤其对依赖相关/线性结构的 Lasso、PLS-DA；RF 树模型可豁免（不依赖线性相关）。

---

## 6. V6 Lasso bootstrap 频率分布

**方法**：每病 B=50 次分层 bootstrap，CLR+标准化后 Lasso(penalty='l1', C=0.1) 拟合，统计每特征入选频率。

| 数据集 | 入选过(>0)特征数 | 频率≥0.8 | 频率 0.5~0.8 | Top 稳定特征（频率） |
|:--|:--|:--|:--|:--|
| CRC | 174 | 2 | 1 | Peptostreptococcus_stomatis(0.98)、**Fusobacterium_nucleatum(0.82)**、Streptococcus_salivarius(0.50) |
| IBD | 122 | 2 | 2 | Bifidobacterium_bifidum(0.92)、Alistipes_finegoldii(0.90)、Akkermansia_muciniphila(0.60) |
| Obesity | 548 | 2 | 13 | Ruminococcus_flavefaciens(0.92)、Pseudoflavonifractor_capillosus(0.82)、Bacteroides_massiliensis(0.78) |

**图**：`S2-v6-stability-frequency-explore.pdf`（频率直方图，红线 τ=0.8）

**解读**：
- **CRC 稳定特征含两个已知标志物**（Peptostreptococcus_stomatis 0.98、Fusobacterium_nucleatum 0.82）——稳定性选择方法有效，且与 domain-knowledge 交叉核对通过（decision-tree §4 风险 1 的"已知标志物交叉核对"要求满足）。
- **频率分布非强双峰**：频率≥0.8 仅 2 特征/病，其后是连续长尾（0.3~0.6 大量特征）。→ **τ=0.8 过严**，只选出 2 个标志物，不足以支撑"Top 10~20 标志物"交付。
- **建议**：τ 下调至 0.5~0.6（每病可选出 3~15 个），或改"按频率排序取 Top-k"而非硬阈值；Obesity 入选过特征数 548（远多于 CRC/IBD），提示其信号更分散、需更宽松阈值或接受更弱标志物。

---

## 7. 异常信号与待裁定项

1. **Obesity 判别力弱**：0 个 FDR 显著特征，Top AUC 仅 0.639，bootstrap 入选特征 548 个（信号分散）——肥胖微生物信号弱于 CRC/IBD，符合 domain-knowledge 预期，但需在方案中显式降低 Obesity 的预期（标志物可信度分级）。
2. **VIP>1 阈值过宽**：选中 40~55% 特征，需上调阈值（VIP>1.5 或分位数），否则 VIP 失去筛选意义。
3. **τ=0.8 过严**：稳定簇仅 2 特征/病，需下调 τ 或改 Top-k 排序（见 V6 解读）。
4. **small_adenoma 口径**（沿用 gate-0.1 [B级] 待裁定）：本验证按"cancer=患病，n+small_adenoma=健康"执行，未改动；若后续裁定 small_adenoma 归患病，CRC 标签需重跑。

---

## 8. 数据接口与脚本清单

- 数据：`outputs/data/B-raw.pkl`（只读，未修改）
- 脚本：`outputs/scratch/utils.py`（公共工具：标签映射/CLR/BH-FDR/零值占比）
  - `verify-S2-v1-baseline.py`、`verify-S2-v2-zerobin.py`、`verify-S2-v3-redundancy.py`、`verify-S2-v4-overlap.py`、`verify-S2-v5-clr.py`、`verify-S2-v6-stability.py`
- 探索图（7 张，`outputs/figures/_explore/`）：`S2-v1-top-single-feature-dist-explore.pdf`、`S2-v2-zerobin-selection-rate-explore.pdf`、`S2-v3-correlation-clusters-explore.pdf`、`S2-v4-dataset-overlap-explore.pdf`、`S2-v4-known-biomarker-presence-explore.pdf`、`S2-v5-clr-correlation-explore.pdf`、`S2-v6-stability-frequency-explore.pdf`

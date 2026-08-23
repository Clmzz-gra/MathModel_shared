# 交接：S2 建模 → 报告（2.2 结果分析完成）

> handoff_type: `report-agent`
> sub: S2（特征选择与生物标志物）
> stage: 2.2（结果分析，门禁 N 材料）
> from: 建模对话 | to: 报告对话
> 日期：2026-08-21
> source_docs: [`result-analysis-S2.md`, `approach-S2-confirmed.md`, `math-S2.tex`, `handoff-S2-model-agent.md`]
> next_action: 报告对话 2.3 内容段（补结论、口径声明、图表规格），说「继续」
> status: ready

---

## 0. 一句话收束

S2 三病独立筛出稳定生物标志物：**CRC 4 个（全部命中已知标志物，3 个 Fisher 显著）、IBD 4 个（4/4 Fisher 显著）、Obesity 20 个（0 个 FDR 显著，弱信号）**。三病稳定特征集 Jaccard 全 0，完全疾病特异。判别信号几乎全为「存在/缺失」主导（presence），丰度信号缺失。VIP 佐证成立；**RF 佐证层退化不可用**（见口径声明）。

---

## 1. 章节映射

| 论文章节 | 内容来源 | 关键数字/表 |
|:--|:--|:--|
| 方法节「特征选择」 | `math-S2.tex` §3-§6 | 近全零过滤 1331→264、CLR δ=6.5e-06、Lasso+bootstrap τ=0.5 B=100、两路信号 BH-FDR m=1331、共现分析、VIP>1.5 |
| 结果节「每病稳定标志物」 | `result-analysis-S2.md` §2.1 | 表 `S2-stable-biomarkers.tex` |
| 结果节「两路信号」 | §2.2 | presence 主导、Wilcoxon 无稳定标志物显著 |
| 结果节「共现分析」 | §2.3 | CRC 4 边全 cooccur、IBD 6 边（3 cooccur+3 exclude） |
| 结果节「跨疾病对比」 | §2.5 | 表 `S2-cross-disease.tex`（Jaccard 全 0） |
| 稳健性/讨论节 | §5 + §3 | 表 `S2-tau-sensitivity.tex`、C 敏感性、RF 退化说明 |
| 讨论节「协同效应」 | §2.3 边界声明 | 二阶初探 + 边界声明 |

---

## 2. 关键数字（来源可溯到 pkl）

> 全部取自 `outputs/data/S2-results.pkl`。**禁止抄 handoff-S2-model-agent 数字**（其 C 敏感性快查与 pkl 有微差，见口径声明 U4）。

### 2.1 每病稳定标志物数（τ=0.5，B=100）

| 疾病 | 稳定数 | Fisher 显著(m=1331) | Wilcoxon 显著 | 共现边数 | VIP>1.5 数 | Spearman(freq vs VIP) |
|:--|:--|:--|:--|:--|:--|:--|
| CRC | 4 | 4 | 1 | 4 | 28 | 0.539 |
| IBD | 4 | 6 | 1 | 6 | 27 | 0.515 |
| Obesity | 20 | 0 | 0 | 24 | 23 | 0.347 |

### 2.2 CRC 稳定标志物（4 个，全部命中已知）

| 标志物 | 频率 | Fisher q | 方向 | 已知 |
|:--|:--|:--|:--|:--|
| Peptostreptococcus_stomatis | 0.99 | 1.24e-05 | up | ✅ |
| Fusobacterium_nucleatum | 0.94 | 3.94e-05 | up | ✅ |
| Porphyromonas_somerae | 0.62 | 1.72e-02 | up | ✅ |
| Clostridium_hathewayi | 0.52 | 0.60 | up | — |

### 2.3 IBD 稳定标志物（4 个，4/4 Fisher 显著）

| 标志物 | 频率 | Fisher q | 方向 | 已知 |
|:--|:--|:--|:--|:--|
| Alistipes_finegoldii | 0.81 | 6.75e-03 | down | — |
| Bifidobacterium_bifidum | 0.75 | 6.75e-03 | up | ✅（已知属） |
| Akkermansia_muciniphila | 0.55 | 6.75e-03 | down | — |
| Eubacterium_ventriosum | 0.53 | 3.98e-02 | down | — |

### 2.4 Obesity 稳定标志物（20 个，0 个 FDR 显著）

Top：Ruminococcus_flavefaciens(0.89)、Pseudoflavonifractor_capillosus(0.84)、Rothia_mucilaginosa(0.72)、Bacteroides_ovatus(0.65)、Mitsuokella_multacida(0.62)、Megasphaera_elsdenii(0.62)、Ruminococcus_bromii(0.61)…全部 fisher_q>0.05。

### 2.5 共现最强对

- **CRC**：Peptostreptococcus_stomatis ↔ Fusobacterium_nucleatum（Spearman 0.762、Fisher p=8.82e-07、OR=12.86）。
- **IBD**：Bifidobacterium_bifidum 与其余 3 个标志物全部互斥（OR 0.19~0.33）。

### 2.6 跨疾病

Jaccard：CRC_IBD=0.0、CRC_Obesity=0.0、IBD_Obesity=0.0；common_biomarkers 空。

---

## 3. 口径声明（报告必须含）

1. **主口径 C=0.1**（T1 裁决）：CRC/IBD 各 4 个高置信标志物为主口径；C=0.5 的 19/17 扩展清单作补充（附录/讨论），不改变主口径。
2. **两路信号**：稳定标志物几乎全为 presence（存在/缺失）主导，丰度信号缺失——报告明确「判别信号以在不在为主」（H3）。
3. **Wilcoxon FDR q>1 判不显著**（U2）：q>1 数值不展示，仅标「不显著」。
4. **RF 佐证层不可用**（U1）：`rf_importance` 全部 ~1e-17 退化，报告**不引用 RF 数字**，仅以 VIP 作独立复现佐证；RF 退化原因记待裁定项 [B级]。
5. **Obesity 低可信度**（U5）：20 个稳定标志物 0 个 FDR 显著，报告显式标注「弱信号、低可信度」，不夸大。
6. **共现边界声明**（必须含）：小样本下仅对入选标志物做二阶探索，无法全特征交互建模；标志物筛选主口径仍为边际信号。
7. **small_adenoma 口径**（T3）：S2 按题面口径（归健康）执行；S1 主口径未定，若 S1 改口径 S2 CRC 需重跑（A级，待核验）。
8. **诚实标注**：全量 bootstrap 频率（乐观）与 CV 折内频率（诚实）两套数字并列（pkl `full_frequency`/`cv_frequency`）。

---

## 4. 图表清单规格

| 图/表 | 数据源（pkl 字段） | 规格 | 论文位置 |
|:--|:--|:--|:--|
| 表：每病稳定标志物 | `per_disease.*.stable_features`+`two_path_signals` | 已出 `S2-stable-biomarkers.tex` | 结果节 |
| 表：跨疾病 Jaccard | `cross_disease.jaccard_matrix` | 已出 `S2-cross-disease.tex` | 结果节 |
| 表：τ 敏感性 | `meta.tau_counts` | 已出 `S2-tau-sensitivity.tex` | 稳健性节 |
| 图：共现网络（CRC/IBD） | `per_disease.*.cooccurrence.cooccurrence_edges` | 节点=标志物，边=cooccur(实线)/exclude(虚线)，标 OR；CRC 4 边、IBD 6 边 | 讨论节「协同效应」 |
| 图：稳定标志物频率直方图 | `outputs/figures/_explore/S2-2.1-stability-frequency-explore.pdf` | 探索图，正式图按 chart-generator 规范重出 | 结果节 |
| 图：VIP>1.5 特征清单 | `per_disease.*.vip` | 独立复现证据（CRC 28/IBD 27/Obesity 23） | 结果节/附录 |

> 正式出图走「图表两级制」：报告对话定规格 → 代码 Preset 子代理出图（chart-generator 规范）→ 审查代理图规合规审。

---

## 5. AI 标注

- 本问（S2 特征选择与生物标志物）建模、实现、结果分析均由 AI 辅助完成（建模对话 + 代码对话 + 审查子代理）。
- 按 `ai-usage-report` skill 规范，报告正文需含 AI 工具使用声明；AI 贡献标记见 `.trae/ai-markers/`（阶段 3.3 扫描合并）。
- 关键 AI 决策点：C 选择（T1）、VIP>1.5 独立复现（T2）、small_adenoma 口径（T3）、FDR m=1331 全特征校正（人类裁定）、共现分析初探（人类裁定采纳）。

---

## 6. 待报告对话注意

- **RF 佐证层不可用**：不要写「RF 与 Lasso 一致」类表述（rf_overlap≈0：CRC/IBD=0、Obesity=0.05；spearman_rank_rf≈0）。
- **Obesity 弱信号**：正文与结论须显式标注低可信度，避免过度声明。
- **共现边界**：协同效应仅作初探证据放讨论节，不改变主选择口径。
- **数字溯源**：所有数字从 pkl 复制，禁止抄 handoff 快查数字。

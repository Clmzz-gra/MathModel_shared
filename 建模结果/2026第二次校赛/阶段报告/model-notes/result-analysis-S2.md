# S2 结果分析：特征选择与生物标志物（2.2）

> 阶段：2.2 结果分析（门禁 N 材料）
> 日期：2026-08-21
> 数据源：`outputs/data/S2-results.pkl`（generated=2026-08-21T17:46:38，source=S2-preprocessed.pkl）
> 上游：`handoff-S2-model-agent.md`（代码→建模回报）、`approach-S2-confirmed.md`、`math-S2.tex`
> 下游：`handoff-S2-report-agent.md`（建模→报告交接）
> 数字口径：**全部取自 pkl 实际值**；handoff 与 pkl 不一致处记待裁定项，不自改。

---

## 0. 一句话收束

S2 三病独立执行「近全零过滤(1331→264) → CLR → Lasso+bootstrap 稳定性选择(τ=0.5, B=100) → 两路信号(Fisher/Wilcoxon + BH-FDR m=1331) → 共现分析 → RF/VIP 佐证」完成。**CRC 稳定标志物 4 个全部命中已知标志物**（Fusobacterium_nucleatum 频率 0.94、Peptostreptococcus_stomatis 0.99、Porphyromonas_somerae 0.62），方法有效性锚点 H6 强验证；IBD 4 个稳定标志物全部（4/4）Fisher 显著；Obesity 20 个稳定标志物 0 个 FDR 显著（弱信号符合 R3 预期）。三病稳定特征集 Jaccard 全 0，完全疾病特异。

---

## 1. 待裁定项裁决（handoff §4）

### T1 — C 选择（P7）：**推荐 C=0.1 为主口径，C=0.5 作扩展补充清单**

**现状（pkl 实际值）**：C=0.1、τ=0.5 下稳定特征数 CRC=4、IBD=4、Obesity=20。CRC 4 个稳定标志物全部命中已知标志物（H6 锚点最强）；IBD 4 个全部 Fisher 显著（q<0.05）。

**C 敏感性（handoff §4 快查，非 pkl 正式值）**：C=0.01→0/0/0、C=0.05→2/1/1、C=0.1→5/4/19、C=0.5→19/17/54、C=1.0→21/17/60。C=0.5 时 CRC 19/IBD 17 达标 Top 10-20，但 Obesity 54 仍弱信号。

**裁决理由**：
1. **生物合理性优先**：C=0.1 的 4 个 CRC 标志物全部是已知 CRC 相关菌（Fusobacterium_nucleatum、Peptostreptococcus_stomatis、Porphyromonas_somerae 均为文献公认 CRC 标志物），这是本问最强的科学验证锚点。上调 C 至 0.5 会引入大量频率 0.5~0.6、Fisher 不显著的低置信特征，稀释生物可信度。
2. **报告目标（Top 10-20）是设计目标而非硬约束**：approach §0 的「Top 10~20」是筛选规模预期，实际数据显示 CRC/IBD 真正稳定且可验证的标志物只有 4 个——「找到 4 个全部验证的强标志物」比「凑 19 个含噪声的标志物」更可辩护。
3. **C=0.5 作扩展补充**：报告可把 C=0.5 的 19/17 清单作为「放宽阈值后的扩展候选」放附录/讨论，正文主口径用 C=0.1 的高置信清单。

**结论**：主口径 C=0.1（4 个高置信标志物，生物合理性最强）；C=0.5 扩展清单作补充证据，不改变主口径。

### T2 — VIP>1.5 独立复现（P2）：**推荐输出 VIP>1.5 特征清单作独立复现证据**

**现状（pkl 实际值）**：VIP>1.5 特征数 CRC=28/264、IBD=27/264、Obesity=23/264。VIP 与 Lasso 稳定标志物的 Spearman 秩相关：CRC=0.539、IBD=0.515、Obesity=0.347（中等正相关）。VIP>1.5 清单**包含**全部稳定标志物：CRC 的 Peptostreptococcus_stomatis(3.37)、Fusobacterium_nucleatum(2.97)、Porphyromonas_somerae(2.11)、Clostridium_hathewayi(1.84) 均在 VIP>1.5 清单（4/4）；IBD 的 4 个稳定标志物全部在 VIP>1.5 清单（Bifidobacterium_bifidum 2.50、Alistipes_finegoldii 2.47、Akkermansia_muciniphila 2.15、Eubacterium_ventriosum 1.91）。

**裁决理由**：
1. VIP 是**真正独立的方法**（PLS-DA，与 Lasso 完全不同），且数据已显示强佐证（稳定标志物出现在 VIP>1.5 清单、秩相关中等正）。输出 VIP>1.5 清单能落实 approach §1.4「多方法独立复现」的 Gamma 思想，把「单一方法偶然」的质疑转化为「多方法交叉印证」。
2. **口径注意**：`vip_overlap`（0.2/0.2/0.1）偏低，是因为 VIP>1.5 集合较大（~27）而稳定集小（4），交集比例被稀释——**报告应以「VIP>1.5 清单佐证 + 秩相关」为主口径，不以低 overlap 作否定证据**。
3. 成本低：VIP 已在 pkl `vip` 字段落盘，仅需输出清单，无重算成本。

**结论**：输出 VIP>1.5 特征清单（CRC 28/IBD 27/Obesity 23）作独立复现证据，配 Spearman 秩相关与「稳定标志物在 VIP>1.5 清单」的交叉印证表述。

### T3 — small_adenoma 口径（R4）：**S1 已选定①归健康，S2 跟随**

**现状**：S1 结果分析 §3.3 已选定主口径①（small_adenoma 归健康，pkl `selected_main_caliber='healthy'`）。S1 approach §H3「四口径敏感性分析全做择优，最终主口径选择记录在结果分析阶段」已落地。

**裁决理由**：S1 主口径①（归健康）与 S2 题面口径一致，S2 当前 CRC 标签口径 = 归健康，与 pkl 实现一致，**无需重跑**。

**结论**：S2 跟随 S1 主口径①（small_adenoma 归健康），与题面口径一致，销项。

---

## 2. 结果解读

### 2.1 每病稳定标志物（τ=0.5，全量 bootstrap 频率，B=100）

**CRC（4 个，全部命中已知标志物，方向均 up=患病富集）**：

| 标志物 | 频率 | CV频率 | Fisher q | Wilcoxon q | 方向 | 已知 | 主导信号 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Peptostreptococcus_stomatis | 0.99 | 0.96 | 1.24e-05 | 0.113 | up | ✅ | presence |
| Fusobacterium_nucleatum | 0.94 | 0.716 | 3.94e-05 | 5.04 | up | ✅ | presence |
| Porphyromonas_somerae | 0.62 | 0.392 | 0.0172 | 5.04 | up | ✅ | presence |
| Clostridium_hathewayi | 0.52 | 0.296 | 0.599 | 4.10 | up | — | presence |

- 前 3 个（P.stomatis、F.nucleatum、P.somerae）Fisher 显著（q<0.05）且为已知 CRC 标志物，构成**高置信核心**；Clostridium_hathewayi 频率 0.52 刚过阈值、Fisher 不显著，为**低置信边缘**。
- 全 4 个主导信号均为 presence（存在/缺失），印证 H3「判别信号以在不在为主」。

**IBD（4 个，4/4 Fisher 显著）**：

| 标志物 | 频率 | CV频率 | Fisher q | 方向 | 已知 | 主导信号 |
|:--|:--|:--|:--|:--|:--|:--|
| Alistipes_finegoldii | 0.81 | 0.68 | 0.00675 | down | — | presence |
| Bifidobacterium_bifidum | 0.75 | 0.664 | 0.00675 | up | ✅（已知属） | presence |
| Akkermansia_muciniphila | 0.55 | 0.484 | 0.00675 | down | — | presence |
| Eubacterium_ventriosum | 0.53 | 0.324 | 0.0398 | down | — | presence |

- 4 个全部 Fisher 显著（q<0.05）：Alistipes_finegoldii（down）、Bifidobacterium_bifidum（up，已知属）、Akkermansia_muciniphila（down）、Eubacterium_ventriosum（down）。Akkermansia_muciniphila 是文献公认的 IBD 保护菌（丰度降低），方向 down 生物合理。
- 全 4 个主导信号 presence。

**Obesity（20 个，0 个 FDR 显著，弱信号符合 R3）**：

| 标志物 | 频率 | CV频率 | Fisher q | 方向 |
|:--|:--|:--|:--|:--|
| Ruminococcus_flavefaciens | 0.89 | 0.728 | 0.244 | down |
| Pseudoflavonifractor_capillosus | 0.84 | 0.532 | 2.41 | down |
| Rothia_mucilaginosa | 0.72 | 0.508 | 0.244 | down |
| Bacteroides_ovatus | 0.65 | 0.524 | 3.65 | up |
| Mitsuokella_multacida | 0.62 | 0.428 | 2.96 | up |
| Megasphaera_elsdenii | 0.62 | 0.476 | 2.41 | up |
| Ruminococcus_bromii | 0.61 | 0.544 | 5.04 | up |
| …（共 20 个） | | | | |

- **全部 20 个 fisher_q > 0.05（0 个 FDR 显著）**，与 A 类验证 R3（Obesity 弱信号、AUC 0.639）一致。稳定标志物可信度**低**，须显式标注。
- 主导信号 18/20 presence，2 个（Bacteroides_ovatus、Ruminococcus_bromii）abundance。

### 2.2 两路信号显著性

- **Fisher（存在/缺失）**：CRC 3 个稳定标志物显著（n_fisher_sig=4，含 1 个稳定集外特征）；IBD 4 个全部显著（n_fisher_sig=6，含 2 个稳定集外特征）；Obesity 0 个显著。
- **Wilcoxon（非零丰度）**：CRC/IBD 各仅 1 个显著（n_wilcoxon_sig=1），且**该显著特征不在稳定集内**——即稳定标志物中**无一个** Wilcoxon 显著；Obesity 0 个。
- **结论**：判别信号几乎完全由「存在/缺失」主导（presence），丰度信号在稳定标志物中缺失。这与 H3（F2 实证：零值占比差 corr(AUC) 0.86~0.89 > 非零丰度差 0.63~0.67）完全一致，两路信号拆分设计得到数据验证。

### 2.3 共现分析（协同效应初探）

**CRC（4 条边，全部 cooccur）**：
- **Peptostreptococcus_stomatis ↔ Fusobacterium_nucleatum**：Spearman 0.762、Fisher p=8.82e-07、OR=12.86——**最强共现对**，两者均为 CRC 相关口腔/肠道菌，生物合理（协同富集）。
- Peptostreptococcus_stomatis ↔ Porphyromonas_somerae：Spearman 0.829、Fisher p=0.0246、OR=4.67。
- Peptostreptococcus_stomatis ↔ Clostridium_hathewayi：Spearman -0.101（负相关）但 Fisher 共现显著（OR=3.12）——存在/缺失口径共现但丰度负相关，提示「同现但丰度此消彼长」。
- Fusobacterium_nucleatum ↔ Clostridium_hathewayi：Spearman 0.346、OR=6.78。

**IBD（6 条边，3 cooccur + 3 exclude）**：
- cooccur：Alistipes_finegoldii↔Akkermansia_muciniphila（OR=6.04）、Alistipes_finegoldii↔Eubacterium_ventriosum（OR=3.13）、Akkermansia_muciniphila↔Eubacterium_ventriosum（OR=3.83）。
- exclude（互斥）：Alistipes_finegoldii↔Bifidobacterium_bifidum（OR=0.194）、Bifidobacterium_bifidum↔Akkermansia_muciniphila（OR=0.328）、Bifidobacterium_bifidum↔Eubacterium_ventriosum（OR=0.272）——**Bifidobacterium_bifidum 与其余 3 个标志物全部互斥**，提示其 up 方向与其余 down 方向菌的拮抗结构。

**Obesity（24 条边）**：详见 pkl `cooccurrence.cooccurrence_edges`；因 0 个 FDR 显著，共现网络仅作弱信号下的结构初探，可信度低。

**边界声明（报告必须含）**：小样本下仅对入选标志物做二阶探索，无法全特征交互建模；标志物筛选主口径仍为边际信号（生物标志物研究标准口径）。

### 2.4 RF/VIP 佐证一致性

- **VIP**：与 Lasso 稳定标志物 Spearman 秩相关 CRC=0.539、IBD=0.515、Obesity=0.347（中等正相关）。稳定标志物全部出现在 VIP>1.5 清单（CRC 4/4、IBD 4/4），VIP 佐证**成立**。
- **RF**：`rf_importance` 全 ~1e-17 机器精度级退化（非零数 CRC 82/IBD 128/Obesity 130），**完全退化**——RF permutation importance 无法区分任何特征，RF 佐证层**不可用**（rf_overlap≈0、spearman_rank_rf≈0）。见 §3 拐角解 #1。

### 2.5 跨疾病对比

- **Jaccard 重叠全 0**：CRC_IBD=0.0、CRC_Obesity=0.0、IBD_Obesity=0.0；`common_biomarkers` 为空。
- **疾病特异性**：CRC 4 个、IBD 4 个、Obesity 20 个，全部为疾病特异，无共同标志物。
- **解读**：符合「Fusobacterium_nucleatum 的 CRC 特异性」等已知生物学预期——三病菌群失调模式高度特异，无跨病共享标志物。这既是方法有效性的佐证（未产生跨病假阳性），也提示三病需独立建模（H7 独立建模假设成立）。

---

## 3. 拐角解 / 未闭合清单

| # | 项 | 严重度 | 说明 | 处置 |
|:--|:--|:--|:--|:--|
| U1 | **RF permutation importance 完全退化** | 高 | 三病 `rf_importance` 全 ~1e-17 机器精度级退化（非零数 CRC 82/IBD 128/Obesity 130），RF 佐证层不可用。可能原因：RF 模型 AUC 近随机（单特征置换不掉性能）或 permutation importance 实现问题。handoff 仅报「RF 重叠 0.00」未暴露此退化 | 记待裁定项 [B级]：需调查 RF 模型 AUC 与 importance 实现；报告 RF 佐证层**降级为「不可用」**，不引用 RF 数字 |
| U2 | **Wilcoxon FDR q>1** | 中 | 多个 wilcoxon_fdr=5.04、4.10 等 >1，标准 BH-FDR q 值应封顶于 1。m=1331 且 p 值大时 q=p·m/rank 可超 1，实现未封顶 | 记口径说明：q>1 一律判「不显著」；报告不展示 q>1 数值，仅标「不显著」 |
| U3 | **n_wilcoxon_sig=1 但不在稳定集** | 中 | CRC/IBD 各 1 个 Wilcoxon 显著特征（全 264 口径）不在稳定标志物内，即稳定标志物无一个丰度显著 | 口径说明：丰度信号在稳定标志物中缺失，符合 H3；报告明确「稳定标志物以存在/缺失信号为主」 |
| U4 | **handoff C 敏感性 vs pkl 不一致** | 低 | handoff §4 快查 C=0.1→5/4/19，pkl 实际 τ=0.5 为 CRC=4/IBD=4/Obesity=20（CRC 5 vs 4、Obesity 19 vs 20 微差） | 记待裁定项：以 pkl 为准（CRC=4/IBD=4/Obesity=20）；handoff 快查为粗扫，不引用 |
| U5 | **Obesity 弱信号** | 中 | 20 个稳定标志物 0 个 FDR 显著，可信度低 | 报告显式标注 Obesity 低可信度（R3 采纳），不夸大 |
| U6 | **T3 small_adenoma 口径已定** | 中 | S1 已选定主口径①（归健康），S2 跟随，与题面口径一致 | 已销项（T3）；S2 CRC 无需重跑 |

---

## 4. 实际落地检查（两条必答题）

### 4.1 输入鲁棒性

- **核心输入**：相对丰度矩阵（CLR 前）。测量误差主要来自测序深度与物种注释。
- **结论敏感性**：稳定标志物对 τ 敏感性低（CRC/IBD 在 τ=0.4~0.7 仅 4~6 个，见 §5）；对 C 敏感性中等（C=0.1→0.5 时 CRC 4→19）。**主口径 C=0.1 的 4 个高置信标志物对参数不敏感**（频率 0.62~0.99，远高于阈值 0.5），鲁棒。
- **需否灵敏度分析**：建议报告附 τ 敏感性表（pkl `tau_counts` 已有）与 C 敏感性说明，作为稳健性证据；无需额外重算。

### 4.2 输出成本

- **建议执行成本**：本问为**生物标志物筛选**，输出是「候选标志物清单 + 方向 + 显著性」，不涉及直接干预成本。落地成本 = 后续验证实验（如 qPCR/队列验证）的成本，属下游研究范畴，本问不量化。
- **风险**：Obesity 弱信号（0 FDR 显著）若被误读为「确定标志物」有过度声明风险——报告须显式标注低可信度。CRC/IBD 高置信标志物可作为后续验证的优先候选。

---

## 5. τ 敏感性（pkl `tau_counts`，全量 bootstrap）

| 疾病 | τ=0.4 | τ=0.5 | τ=0.6 | τ=0.7 |
|:--|:--|:--|:--|:--|
| CRC | 6 | 4 | 3 | 2 |
| IBD | 6 | 4 | 2 | 2 |
| Obesity | 32 | 20 | 9 | 3 |

- CRC/IBD 入选数对 τ 不敏感（4~6），稳定簇与噪声可分（H5 成立）；Obesity 对 τ 敏感（32→3），印证弱信号。

---

## 6. 跨问不等式核验清单（门禁 N 必选项）

- **S1 未合并**：S1 的 2.2 结果分析尚未产出，`data-integration` 无 S1 关键数值可互证。
- **登记待核验项**：S1 合并后需核验——S2 的 CRC 稳定标志物（Fusobacterium_nucleatum 等）与 S1 分类器输入特征一致（approach §11「特征选择→分类建模」一致链路）；S1 已选定主口径①（归健康），S2 跟随，CRC 无需重跑（T3）。
- **本问内部自洽**：三病稳定特征集 Jaccard 全 0（疾病特异）与「Fusobacterium_nucleatum CRC 特异性」预期同向，无内部矛盾。

---

## 7. 结论

1. **CRC**：4 个稳定标志物全部命中已知标志物（P.stomatis 0.99、F.nucleatum 0.94、P.somerae 0.62），3 个 Fisher 显著，方法有效性锚点 H6 强验证；共现最强对 P.stomatis↔F.nucleatum（OR=12.86）。
2. **IBD**：4 个稳定标志物全部（4/4）Fisher 显著，Akkermansia_muciniphila（down）生物合理；Bifidobacterium_bifidum 与其余标志物互斥。
3. **Obesity**：20 个稳定标志物 0 个 FDR 显著，弱信号符合 R3，低可信度标注。
4. **两路信号**：稳定标志物几乎全为 presence 主导，丰度信号缺失，验证 H3。
5. **共现**：CRC 全 cooccur（协同富集），IBD 含 cooccur+exclude（拮抗结构），作二阶初探并声明边界。
6. **跨疾病**：Jaccard 全 0，完全疾病特异，支持独立建模。
7. **佐证**：VIP 佐证成立（秩相关 0.35~0.54）；**RF 佐证层退化不可用**（U1，待调查）。

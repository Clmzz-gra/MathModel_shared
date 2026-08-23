# 降维画像（B 题宏基因组数据）

> 阶段 0.4 产出 | 方法：PCA（numpy SVD）+ t-SNE + UMAP | 样本 484 × 特征 1331

## 1. 方差解释表（前 10 主成分）

| 主成分 | 方差解释率 | 累积解释率 |
|---|---|---|
| PC1 | 2.96% | 2.96% |
| PC2 | 2.53% | 5.49% |
| PC3 | 2.29% | 7.78% |
| PC4 | 2.18% | 9.96% |
| PC5 | 1.87% | 11.83% |
| PC6 | 1.65% | 13.48% |
| PC7 | 1.60% | 15.08% |
| PC8 | 1.53% | 16.61% |
| PC9 | 1.45% | 18.06% |
| PC10 | 1.45% | 19.51% |

- 前 64 个主成分累积解释率 ≥ 60%（cumR² = 60.00%），用于分群路径预降维。
- 首主成分 PC1 解释率仅 2.96%，**无单一主导结构，方差高度分散**（92% 稀疏成分数据典型特征）。

## 2. 前 5 主成分载荷解读（|载荷| 前 6 物种）

- **PC1**（2.96%）：Atopobium|Atopobium_parvulum (+0.170)、Granulicatella|Granulicatella_unclassified (+0.162)、Veillonella|Veillonella_dispar (+0.136)、Anaerococcus|Anaerococcus_obesiensis (+0.136)、Peptostreptococcus|Peptostreptococcus_anaerobius (+0.134)、Solobacterium|Solobacterium_moorei (+0.132)
- **PC2**（2.53%）：Veillonella|Veillonella_dispar (-0.167)、Parvimonas|Parvimonas_micra (+0.166)、Parvimonas|Parvimonas_unclassified (+0.165)、Treponema|Treponema_denticola (-0.164)、Alloprevotella|Alloprevotella_rava (-0.164)、Catonella|Catonella_morbi (-0.163)
- **PC3**（2.29%）：Peptostreptococcus|Peptostreptococcus_anaerobius (+0.143)、Alloprevotella|Alloprevotella_rava (+0.132)、Treponema|Treponema_denticola (+0.132)、Catonella|Catonella_morbi (+0.132)、Bulleidia|Bulleidia_extructa (+0.131)、Prevotella|Prevotella_denticola (+0.127)
- **PC4**（2.18%）：Lactobacillus|Lactobacillus_animalis (-0.207)、Lactobacillus|Lactobacillus_helveticus (-0.207)、Buchnera|Buchnera_aphidicola (-0.207)、Kingella|Kingella_unclassified (-0.207)、Candidatus_Moranella|Candidatus_Moranella_endobia (-0.207)、Lactobacillus|Lactobacillus_ultunensis (-0.207)
- **PC5**（1.87%）：Blautia|Blautia_producta (+0.190)、Clostridium|Clostridium_hathewayi (+0.183)、Coprobacillus|Coprobacillus_unclassified (+0.181)、Flavonifractor|Flavonifractor_plautii (+0.172)、Clostridium|Clostridium_clostridioforme (+0.169)、Anaerostipes|Anaerostipes_unclassified (+0.152)

> 载荷解读说明：物种级相对丰度经 StandardScaler 标准化后进入 PCA。由于数据 92.2% 零值稀疏，
> 载荷主要反映「某物种在少数样本中高丰度」的稀疏结构，而非连续丰度梯度。属/种名从 7 级
> 分类学列名提取。

## 3. t-SNE vs UMAP 对比

- 全量 484 样本，t-SNE（perplexity=30）与 UMAP 双图对比（见 `outputs/figures/_explore/tsne-umap-compare.pdf`）。
- 两者均呈现**按数据集（dataset_name）强分离**的宏观结构（见 `_explore/group-tsne.pdf`），
  提示存在显著**批次效应**（不同队列测序/生信流程差异），跨队列泛化（S3）需重点处理。
- t-SNE 与 UMAP 的簇结构一致性：宏观三群（对应三数据集）一致，细粒度亚群在两种方法下
  略有差异（t-SNE 更强调局部邻域，UMAP 更保留全局结构）。

## 4. 降维方案建议（供阶段 1 参考）

- 成分数据（相对丰度定和）在欧式空间 PCA 前，1.4 阶段应考虑 **CLR 变换**（伪计数处理零值），
  本画像按 0.4b 口径用 StandardScaler，结果以「稀疏结构 + 批次效应」为主，CLR 后结构可能更清晰。
- 分类学层级聚合（属/门级）可显著降维并降低批次噪声，S3 跨队列预测可利用。

# 领域知识：人类肠道宏基因组与疾病预测

> 阶段 0.0 产出（2026-08-21）| 数据注释区待 0.3 清洗完成后填充

## 非文献知识区

### 三类疾病的已知微生物关联（数据解释锚点）

| 疾病 | 数据集名 | 已知关联微生物（文献共识） | 备注 |
|:--|:--|:--|:--|
| 结直肠癌 Colorectal | Zeller_fecal_colorectal_cancer | *Fusobacterium nucleatum*（具核梭杆菌）为最强共识标志物；*Peptostreptococcus stomatis*、*Parvimonas micra*、*Porphyromonas* 属亦多次报道；肿瘤富集 *Bacteroides fragilis* 毒素型 | Zeller 2014 原始队列即本数据来源之一；宏基因组物种级特征建模 AUC 常见 0.80-0.90 |
| 炎症性肠病 IBD | metahit | IBD 与菌群多样性下降、*Faecalibacterium prausnitzii*（普拉梭菌）减少相关；UC/CD 亚型共享部分信号；10-species 微生物特征外部验证 AUC ~0.85 | metahit 即 MetaHIT 队列（IBD 肠型研究原始来源）；本数据集含 UC 与 CD 两种亚型 |
| 肥胖症 Obesity | Chatelier_gut_obesity | 菌群基因丰富度（gene richness）低与肥胖相关；*Bifidobacterium pseudocatenulatum* 等为潜在靶点；肥胖信号一般弱于 CRC/IBD | Chatelier 2013 原始队列（LEAN/OB 分型）；肥胖预测 AUC 通常 0.65-0.75，低于 CRC/IBD |

### 宏基因组数据的建模惯例

- **相对丰度是成分数据**：每行特征和为常数（或近似），进入欧式空间模型前常做 CLR 变换；稀疏零值需处理（伪计数/乘法替换，见知识库 AL-007）
- **高维小样本**：物种级特征通常数百~数千列 vs 样本数百 → 必须特征选择 + 正则化 + 分层交叉验证
- **跨队列泛化是核心难点**：不同研究存在批次效应（测序平台/DNA 提取/生信流程），跨疾病预测（本 B 题 S3）本质是"最严苛的跨队列测试"，预期 AUC 显著下降
- **物种名到属/门聚合**：分类学层级 k__p__c__o__f__g__s__ 可聚合降低维度与批次噪声（S3 可利用）
- **标签语义**：疾病 vs 健康对照是标准二分类协议；IBD 含两种亚型（UC/CD）均为患病

## 文献摘要区

### 结直肠癌（CRC）

1. **Zeller et al. 2014**（本数据来源）：*Potential of fecal microbiota for early-stage detection of colorectal cancer*，Mol Syst Biol。物种级宏基因组标志物 CRC 检测 AUC ~0.89（含健康对照区分）。DOI: 10.15252/msb.20145645
2. **Wirbel et al. 2019 / Thomas et al. 2019**：跨队列荟萃分析，CRC 微生物标志物跨队列 AUC ~0.70-0.80，证明跨队列泛化可行但衰减。Nature Medicine。
3. **[Pooled analysis of 3,741 stool metagenomes from 18 cohorts](https://preview-www.nature.com/articles/s41591-025-03693-9)**（Nature Medicine 2025）：18 队列 3741 粪便宏基因组跨阶段/株级可重复 CRC 生物标志物——跨队列标志物稳定性的最新证据。
4. **[Gut microbiome-based ML model for early CRC and adenoma screening](https://link.springer.com/article/10.1186/s13099-025-00750-z)**（Gut Pathogens 2025）：宏基因组 ML 用于早期 CRC/腺瘤筛查。

### 炎症性肠病（IBD）

5. **[10-species microbial signature of IBD by ML and external validation](https://link.springer.com/article/10.1186/s13619-025-00246-w)**（Cell Regeneration 2025）：10 菌种 IBD 特征 + 外部验证。
6. **[IBDPred: Enhanced Stacking Ensemble for IBD Prediction](https://ieeexplore.ieee.org/document/11356767)**（IEEE 2025）：基于菌群丰度的 stacking 集成 IBD 预测。

### 肥胖症（Obesity）

7. **[Chatelier et al. 2013**（本数据来源）：*Richness of human gut microbiome correlates with metabolic markers*，Nature。低基因丰富度与肥胖/代谢指标相关。DOI: 10.1038/nature12506
8. **[ML prediction of obesity-associated gut microbiota](https://pmc.ncbi.nlm.nih.gov/articles/PMC11839209/)**（Frontiers 2024/2025）：GMrepo 14028 样本 ML 预测肥胖，Bifidobacterium pseudocatenulatum 为潜在靶点。

### 跨疾病/方法学

9. **[Incorporating metabolic activity, taxonomy and community structure to improve microbiome-based predictive models](https://pesquisa.bvsalud.org/controlecancer/resource/pt/mdl-38214657)**：引入分类学/群落结构提升宿主表型预测——S3 分类学信息利用的方法学依据。

> 详细卡片入库可后续经 paper-parser skill 处理；当前以摘要级引用为准（内部报告引用来源）。

## 数据注释区

> 阶段 0.3 清洗完成后填充（2026-08-21 填充）。数据源：`data.csv`（原始）→ `outputs/data/B-raw.pkl`（0.2 转换）→ `outputs/data/c-data-cleaned.pkl`（0.3 清洗，共享）。

### 原始列编码含义

| 列/字段 | 含义 | 处理 |
|---|---|---|
| `dataset_name` | 疾病数据集名：Zeller_fecal_colorectal_cancer（CRC）/ metahit（IBD）/ Chatelier_gut_obesity（Obesity） | 保留原字符串 |
| `disease` | 疾病标签（见下方映射） | 保留原字符串；S1 预处理时构造二分类标签 |
| 其余 1331 列 | 物种级相对丰度（百分比，每行和≈100），列名 = `k__域\|p__门\|c__纲\|o__目\|f__科\|g__属\|s__种` 7 级分类学层级 | float32 压缩；零值=未检出（真实稀疏，不填补） |

### 标签映射（数据解释文档定义，S1 构造二分类用）

| 数据集 | 患病标签 | 健康对照 |
|---|---|---|
| Zeller（CRC） | `cancer` | `n`、`small_adenoma`（26 例，癌前病变归健康——口径待裁定，见 registry） |
| metahit（IBD） | `ibd_ulcerative_colitis`（21 例）、`ibd_crohn_disease`（4 例） | `n` |
| Chatelier（Obesity） | `obesity` | `leaness` |

### 清洗决策记录

- 0 重复行；0 NaN；无整体领域排除
- 零值不填补（0 = 微生物未检出）；不做任何变换（推迟 1.4）
- 异常模式：无；small_adenoma 26 例保留（敏感性分析归 S1，registry 2026-08-21 登记）
- 原始来源索引：`problems/2026第二次模拟赛赛题/B题 基于宏基因组数据的疾病预测模型研究/data.csv`（485 行含表头）

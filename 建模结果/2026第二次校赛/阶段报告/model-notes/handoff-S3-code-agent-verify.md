# S3 A 类验证交接（handoff_type=code-agent-verify）

> 建模对话（S3）→ 代码对话（S3）| 阶段 1.1 A 类共享事实验证
> 运行模式：auto（门禁 1 呈递人类拍板）
> ⚠️ 本文件是 A 类验证交接（`-verify` 后缀），与正式交接 `handoff-S3-code-agent.md` 不同，禁止同名覆盖。

---

```yaml
---
handoff_type: code-agent-verify
sub: S3
from: 建模对话（S3）
to: 代码对话（S3）
stage: "1.1"
source_docs:
  - solution/model-notes/decision-tree-S3.md   # 决策树：协议/候选策略/风险/A类清单（本交接规格来源）
  - solution/model-notes/knowledge-retrieval-B.md  # S3 节卡片命中
  - solution/problem-statement.md              # S3 题面
  - solution/domain-knowledge.md               # 跨队列泛化预期
  - outputs/data/inventory-B.txt               # B-raw.pkl 结构
status: ready
next_action: 代码对话 1.1 A 类验证：按本交接验证规格跑 5 项实验 + 产出探索图 + 解读，回报 handoff-S3-model-agent-verify.md 后切回建模对话（1.1-1.2）
---
```

---

## 一、任务总览

为 S3 跨疾病预测模型（leave-one-disease-out 泛化评估）执行 **A 类共享事实验证**，产出探索图 + 解读，供建模对话在 1.2 辩论中裁定方案。**A 类实验第一步必须是简单基线**（防过度设计）。本交接**只验证事实，不选方案**（方案由建模/人类裁定）。

## 二、数据接口

- **文件**：`outputs/data/B-raw.pkl`
- **结构**（源自 `outputs/data/inventory-B.txt`）：
  - 484 样本 ×（2 元数据列 + 1331 特征列）
  - 元数据列：`dataset_name`、`disease`
  - 三数据集样本量：Chatelier_gut_obesity=253、Zeller_fecal_colorectal_cancer=121、metahit=110
  - 患病判定口径（与 problem-statement / inventory 一致）：Colorectal 患病=`cancer`（48）；IBD 患病=`ibd_ulcerative_colitis`(21)+`ibd_crohn_disease`(4)（共 25）；Obesity 患病=`obesity`（164）；其余为健康对照（CRC 的 `n`+`small_adenoma` 为健康 73、IBD 的 `n` 为健康 85、Obesity 的 `leaness` 为健康 89）
  - 特征列：物种级相对丰度，0 值占比 92.21%，每行丰度和≈100（**定和成分数据**）
  - 特征名格式：`k__Bacteria|p__Firmicutes|...|s__X`（7 级分类学层级，`k__p__c__o__f__g__s__`，可聚合到属 `g__` / 门 `p__`）
- **三组合（测试集正类占比提示类别不平衡）**：
  - C1：训练 {metahit, Chatelier} → 测试 Zeller（CRC），正类 40%
  - C2：训练 {Zeller, Chatelier} → 测试 metahit（IBD），正类 23%（少数类）
  - C3：训练 {Zeller, metahit} → 测试 Chatelier（Obesity），正类 65%（正类占多数）

## 三、验证规格（5 项，A1 为强制第一步）

### A1【简单基线 · 强制第一步】直接迁移的 S1 模型 leave-one-disease-out
- **目的**：建立跨疾病性能下界，防过度设计。
- **做法**：对每个组合，仅用训练疾病样本拟合**正则化 Logistic 回归（L2）**，CLR 前置（成分数据，需零值处理——可用 AL-007 乘法替换 δ=0.65×检出限，或伪计数），物种级特征；对测试疾病样本预测。
- **预处理注意**：特征筛选/标准化参数**只能在训练集内估计**（防泄漏）；CLR 的几何均值与标准化均值/方差均从训练集计算。
- **输出**：3 组合各报 **AUC**（主）+ **训练集 Youden J 阈值迁移**下的 ACC / 灵敏度 / 特异度 / F1；标注测试集正类占比。另报 3 组合 AUC 均值。
- **解读要求**：AUC 是否接近随机 0.5？哪组合最好/最差？与 domain-knowledge 预期（CRC 跨队列 0.70-0.80）比如何？是否呈"同疾病跨队列 > 跨疾病"的衰减？

### A2【三数据集特征重叠度】
- **目的**：判断"训练与测试疾病共享标志物是否足够支撑迁移"（对应决策树特征交集备选）。
- **做法**：统计三数据集在**物种级 / 属级 / 门级**的共享特征（标志物）数量与占比；物种级=按完整特征名，属级=按 `g__` 前缀聚合，门级=按 `p__` 前缀聚合。
- **输出**：三数据集共享特征维恩图/热图 + 各级共享数量表。
- **解读要求**：跨疾病共享信号占比高还是低？属/门级聚合是否显著提高共享率（即"聚合增加共享信号密度"假设）？

### A3【各疾病信号强度（S1 基线 AUC 参考）】
- **目的**：量化跨疾病 AUC vs 域内 AUC 的衰减量（"泛化代价"）。
- **做法**：若 S1 已产出域内 AUC，直接引用；否则用**同样的正则化 Logistic + CLR** 对每疾病数据集做域内评估（分层 CV 或 LOOCV，小样本诚实评估），得三疾病域内 AUC 参考值。
- **输出**：三疾病域内 AUC 表 + 每组合"跨疾病 AUC − 域内 AUC"衰减表。
- **解读要求**：哪疾病信号强（CRC/IBD 预期强，Obesity 预期弱）？跨疾病衰减是否与"信号强度 × 训练-测试共享信号"相关？

### A4【分类学层级聚合对批次差异的影响初查】
- **目的**：验证决策树主策略"属/门级聚合降维减批次/疾病特异噪声"假设。
- **做法**：在 A1 同一协议下，对比**物种级 vs 属级 vs 门级**聚合特征时的 3 组合跨疾病 AUC。
- **输出**：层级对比柱状图/表（x=聚合层级，y=各组合 AUC）。
- **解读要求**：聚合是否提升跨疾病 AUC？门级是否会因过粗丢失信号而下降？最优聚合层级是哪级？

### A5【批次效应初探（可选）】
- **目的**：看三数据集是否因批次（不同研究/平台）清晰分开，辅助区分"批次差异"与"疾病差异"。
- **做法**：CLR 后三数据集 PCA / t-SNE 投影（WF-004），按 dataset_name 着色。
- **输出**：PCA/t-SNE 散点图。
- **解读要求**：三数据集是否分簇？Obesity 与 CRC/IBD 的距离？批次效应是否显著？

## 四、待产探索图清单（`outputs/figures/_explore/`，探索图不入论文，附 2-3 句解读）

| # | 图名（建议） | 内容 | 对应实验 |
|---|---|---|---|
| 1 | `S3-ldo-baseline-auc.pdf` | 3 组合直接迁移 AUC + 训练阈值迁移指标 | A1 |
| 2 | `S3-feature-overlap-venn.pdf` | 三数据集物种/属/门级共享标志物维恩图或热图 | A2 |
| 3 | `S3-domain-vs-cross-auc.pdf` | 域内 AUC vs 跨疾病 AUC 衰减图 | A3 |
| 4 | `S3-hierarchy-levels-auc.pdf` | 物种/属/门聚合层级下跨疾病 AUC 对比 | A4 |
| 5 | `S3-batch-pca-tsne.pdf` | 三数据集 PCA/t-SNE 分布（可选） | A5 |

> 探索图命名：`S3-{内容}-explore.pdf`（`_explore/` 下）；无版本约束，迭代直接覆盖。每图附 2-3 句解读（数据特征/建模启示/异常信号）。

## 五、回报要求

- 结果与探索图解读写入 **`handoff-S3-model-agent-verify.md`**（handoff_type=model-agent-verify、sub=S3、`-verify` 后缀），含：每项实验结论 + 探索图路径 + 异常信号 + 待裁定项。
- **A 类实验第一步必须是 A1 简单基线**；A1 通过后再跑 A2-A5。
- 轻量验证代码命名 `outputs/scratch/verify-S3-*.py`（带代码头注释）。A 类验证属建模期/探索层，不产正式 pkl（只算不产，结论进 `handoff-...-verify.md`）。
- **长计算提醒（C4）**：484 样本 × 1331 特征 × LOOCV/分层 CV，规模不大，预计单实验 <2 分钟；若某实验超 2 分钟，按 C4 规范交主会话后台执行并轮询，勿在子代理内长跑外部进程。

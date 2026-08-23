# S3 正式交接（handoff_type=code-agent）

> 建模对话（S3）→ 代码对话（S3）| 阶段 1.3 方案确认（门禁 1 材料）
> 运行模式：auto（门禁 1 呈递人类拍板）
> ⚠️ 本文件是**正式交接**（无 `-verify` 后缀），与 A 类验证交接 `handoff-S3-code-agent-verify.md` 不同，禁止同名覆盖。

---

```yaml
---
handoff_type: code-agent
sub: S3
from: 建模对话（S3）
to: 代码对话（S3）
stage: "1.3"
source_docs:
  - solution/model-notes/approach-S3-confirmed.md   # 方案确认书（最终方案/数学框架/预期输出，本交接规格来源）
  - solution/model-notes/decision-tree-S3.md         # 决策树（协议/候选策略/风险）
  - solution/model-notes/debate-S3.md               # 辩论（推荐方案 + 待裁定项处置）
status: ready
next_action: 代码对话 1.4 预处理（与建模 2.0 并行），说「继续」
---
```

---

## 一、任务总览

为 S3 跨疾病预测模型（leave-one-disease-out 泛化评估）执行**数据预处理（1.4）**，为 2.1 正式实现准备 `S3-preprocessed.pkl`。本交接是门禁 1 材料，方案已锁定（见 `approach-S3-confirmed.md`）。

**核心结论（先给，2026-08-21 人类裁定重构）**：S3 的硬交付是「建立跨疾病预测模型」——正式构建并对比**四策略**（A 直接迁移 / B 共享标志物 / C 分类学聚合 / D 部署校正），选出最优者交付；**若四策略 3 组合 AUC 均值全部 < 0.60，触发紧急回退（R1 树模型 → R2 样本合并 → R3 重加权 → R4 对抗式域适应），达可用线（均值 ≥0.65 或提升 ≥0.10）即交付**；穷尽后以完整证据链报告。归因分析是性能探讨的一部分，不是替代「建立模型」的答案。详见 `approach-S3-confirmed.md` §1.1-1.6。

---

## 二、规格

### 2.1 LODO 协议（3 组合定义）

leave-one-disease-out（留一疾病）：每次留一种疾病作测试集，其余两种疾病作训练集。

| 组合 | 训练集（2 疾病） | 测试集（1 疾病） | 测试集正类占比 |
|---|---|---|---|
| C1 | IBD（metahit）+ Obesity（Chatelier） | CRC（Zeller） | 40%（患病 48 / 健康 73） |
| C2 | CRC（Zeller）+ Obesity（Chatelier） | IBD（metahit） | 23%（患病 25 / 健康 85，少数类） |
| C3 | CRC（Zeller）+ IBD（metahit） | Obesity（Chatelier） | 65%（患病 164 / 健康 89，正类占多数） |

**硬约束**：测试疾病在训练阶段**完全不可见**（标签与特征均不参与训练）。

### 2.2 模型规格（与 S1 口径一致）

- **模型**：`LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=2000)`，**近全零过滤后物种级特征（零值占比>95% 剔除，1331→264，三病并集统一口径，与 S1/S2 一致）**。
- **预处理**：近全零过滤 → CLR 变换（零值乘法替换 δ=0.65×检出限=6.5e-6）→ StandardScaler（均值/方差**仅训练集估计**，防泄漏）。
- **随机种子**：`seed=42`（全链路统一，见 `proxy-replacement-checklist-S3.md`）。

### 2.3 阈值迁移（禁测试集重定阈值）

- 训练集 Youden J 最优阈值 $\tau^*$（使灵敏度+特异度−1 最大）**只在训练集估计**，原样搬到测试集。
- **禁止在测试集上重定阈值**（会用到新疾病标签，造成评估泄漏，违背 LODO 语义）。

### 2.4 策略 B：特征交集筛选（共享标志物通用模型）

- 取三数据集共享物种作特征子集（A 类验证在 1331 全集上测得 344 个；**正式实现基于过滤后 264 特征集按特征名交集重算**，数量可能略减；不含测试标签→无泄漏），在训练疾病上重训 §2.2 同款模型，重跑 LODO。
- 属级共享可作对照。

### 2.5 策略 C：分类学聚合（属级/门级 + CLR，回应题面注 2）

- 物种级特征按 `g__`（属）/ `p__`（门）层级聚合（同属/同门丰度求和），CLR 后重训 §2.2 同款模型，重跑 LODO。
- A 类验证 A4 已实测聚合不提升（0.556→0.539→0.528）——正式实现仍完整跑 3 组合，**如实报告**（分类学信息已利用，实证无增益）。

### 2.6 策略 D：部署校正（Platt 校准）

- 在策略 A/B/C 中 AUC 最优者上叠加 Platt 缩放：$P_{\text{cal}}(y=1 \mid \mathbf{x}) = 1/(1 + \exp(A \cdot f(\mathbf{x}) + B))$，参数 $A,B$ **仅训练集估计**（Logistic 回归拟合，`max_iter=2000`），实现时校验 $A<0$（否则警告，防排序反转）。
- 目标**不改 AUC**（单调变换不改变排序），只修复可部署指标（ACC/F1/灵敏度）。

### 2.7 紧急回退协议（触发条件：四策略均值均 <0.60）

- **R1 树模型族**：`RandomForestClassifier(n_estimators=500, random_state=42)` / `XGBClassifier`（物种级过滤后特征，CLR 可选，仅训练集拟合超参）。
- **R2 样本合并通用模型**：2 训练疾病样本合并训练「患病 vs 健康」通用分类器（Logistic L2 或 RF），测试疾病样本完全不可见。
- **R3 密度比重加权**：importance weighting（密度比估计给训练样本加权），转导式边界显式声明（用测试集特征、绝不用测试集标签）。
- **R4 对抗式域适应（DANN，最后手段）**：仅在 R1-R3 全败且时间允许时尝试，严格验证集防泄漏。
- **可用判定**：3 组合 AUC 均值 ≥0.65 或相对策略 A 提升 ≥0.10 → 交付。

### 2.8 评估指标

- **主指标**：AUC（阈值无关，3 组合各报 + 均值）。
- **辅指标**：训练阈值迁移下的 ACC / 灵敏度 / 特异度 / F1（标注测试集类别不平衡）。
- **衰减量**：跨疾病 AUC − 域内 AUC（域内参考：CRC 0.814 / IBD 0.885 / Obesity 0.644）。

---

## 三、数据接口

- **文件**：`outputs/data/c-data-cleaned.pkl`（0.3 清洗后）
- **结构**（源自 `outputs/data/inventory-B.txt`，清洗后以实际为准）：
  - 484 样本 ×（2 元数据列 + 1331 特征列）
  - 元数据列：`dataset_name`、`disease`
  - 三数据集样本量：Chatelier_gut_obesity=253、Zeller_fecal_colorectal_cancer=121、metahit=110
  - 患病判定口径：Colorectal 患病=`cancer`（48）；IBD 患病=`ibd_ulcerative_colitis`(21)+`ibd_crohn_disease`(4)（共 25）；Obesity 患病=`obesity`（164）；其余为健康对照；**small_adenoma（26 例）四口径沿 S1 裁定（归健康/归病变/剔除/单开一类，S1 全做择优选定主口径后本问跟随；未选定前按题面口径归健康，见 R4）**
  - 特征列：物种级相对丰度，0 值占比 92.21%，每行丰度和≈100（**定和成分数据**）；**正式实现先近全零过滤（零值占比>95%，1331→264，三病并集统一口径，与 S1/S2 一致），过滤规则见 S2 `verify-S2-v2-zerobin.py`**
  - 特征名格式：`k__Bacteria|p__Firmicutes|...|s__X`（7 级分类学层级，可聚合到属 `g__` / 门 `p__`）

---

## 四、预期输出（S3-results.pkl 结构）

> 2.1 正式实现后落盘 `outputs/data/S3-results.pkl`，结构如下（1.4 预处理产出 `S3-preprocessed.pkl` 为中间产物）。

```
S3-results.pkl
├── meta: {sub, stage, model, seed, clr_delta, 生成时间, 预算受限标注}
├── strategy_compare:      # 策略对比总表（正式建模核心交付）
│   ├── A_direct: {C1, C2, C3: {auc, acc, sensitivity, specificity, f1, youden_threshold}, mean_auc}
│   ├── B_shared: {C1, C2, C3: {auc, ...}, shared_feature_count, mean_auc}
│   ├── C_hierarchy: {C1, C2, C3: {auc, ...}, level: "genus|phylum", mean_auc}
│   ├── D_calibrated: {base_strategy: str, C1, C2, C3: {auc(不变), cal_acc, cal_sensitivity, cal_specificity, cal_f1, A, B}, mean_auc}
│   └── best_strategy: str       # 交付模型选择
├── fallback:              # 紧急回退（触发时）
│   ├── triggered: bool           # 四策略均值均 <0.60
│   ├── R1_tree: {mean_auc, C1, C2, C3}
│   ├── R2_pooled: {mean_auc, C1, C2, C3}
│   ├── R3_weighted: {mean_auc, C1, C2, C3}
│   ├── R4_dann: {mean_auc, C1, C2, C3}   # 可选
│   ├── usable: bool / delivered_strategy: str
│   └── exhausted_evidence: {...}          # 穷尽证据链
├── domain_auc:           # 域内参考（A3 已得，可引用）
│   └── {CRC, IBD, Obesity}
├── decay_attribution:     # 衰减归因表（三分法）
│   └── {CRC, IBD, Obesity}: {domain_auc, cross_auc, decay, dominant_cause}
├── migration_analysis:    # 深度迁移分析
│   └── {direction_consistent_count, direction_flipped_count, shared_species_list}
└── threshold_drift:       # C3 阈值漂移量化
    └── {train_baseline, test_baseline, boundary_position, diagnosis}
```

---

## 五、参考实现

- A 类验证脚本（`outputs/scratch/`，带 C1 头注释）：
  - `verify_S3_common.py`（共享模块：CLR/StandardScaler/LODO 协议）
  - `verify-S3-a1-baseline.py`（直接迁移基线，主实验参考）
  - `verify-S3-a2-overlap.py`（特征重叠度，子实验 1 参考）
  - `verify-S3-a3-domain-auc.py`（域内 AUC，衰减归因参考）
  - `verify-S3-a4-hierarchy.py`（层级聚合对比）
  - `verify-S3-a5-batch.py`（批次效应初探）
- 探索图（`outputs/figures/_explore/`，不入论文）：`S3-ldo-baseline-auc.pdf`、`S3-feature-overlap-venn.pdf`、`S3-domain-vs-cross-auc.pdf`、`S3-hierarchy-levels-auc.pdf`、`S3-batch-pca-tsne.pdf`

---

## 六、已知风险

| 风险 | 应对 |
|---|---|
| **四策略全部失败（负结果）** | 紧急回退协议（§2.7）：R1-R4 逐级换模型族，达可用线（均值 ≥0.65 或提升 ≥0.10）即交付；穷尽后以完整证据链报告（S3-results.pkl fallback.exhausted_evidence） |
| **IBD 衰减最大（−0.358）** | 域内最强却衰减最大，提示模型学的是 IBD 特异标志物；归因到「疾病特异信号」 |
| **C3 灵敏度极低（0.006）** | 标签语义漂移/概率基线偏移；策略 D Platt 校准修复可部署指标，AUC 不变 |
| **阈值迁移禁测试集重定** | 硬约束，代码审查（1.5/2.1.5）重点检查 |
| **回退候选过拟合（R4 对抗式域适应）** | R4 为最后手段，严格验证集防泄漏；优先 R1-R3 |

---

## 七、No Placeholders

本交接所有规格、数据接口、预期输出、参考实现、风险均已给出具体值或明确来源，无 `TODO`/`TBD`/`待定` 占位符。代理值（seed/Platt 迭代数/重加权参数）见 `proxy-replacement-checklist-S3.md`，均为可执行的具体值。

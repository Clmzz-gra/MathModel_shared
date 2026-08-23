# S3 门禁 1 审查结论（review-1.3-S3）

> **审查代理角色**：建模对话审查代理（modeling Preset，审查/裁决分离协议——本文件为审查结论，主建模只裁决）
> **模型**：deepseek-v4-pro:0813 | **思考强度**：max（最高强度，子代理审查固定）
> **审查对象**：`approach-S3-confirmed.md`（方案确认书）、`handoff-S3-code-agent.md`（正式交接）、`proxy-replacement-checklist-S3.md`（代理值清单）
> **对照材料**：`decision-tree-S3.md`、`debate-S3.md`、`handoff-S3-model-agent-verify.md`、`knowledge/method-graph.md`
> **日期**：2026-08-21 | **阶段**：1.3 方案确认（门禁 1：建模定稿）

## 必读清单已读汇报

- [x] `TRAE-建模.md`（门禁协议：审查代理职责 + 结论文件格式 + 1.3 判定内容）
- [x] `TRAE.md`（门禁 1 判定内容 + 四门禁总览 + 待裁定项分级）
- [x] skill `modeling-decision-tree`（题型判定 + 分类评估路径 + 检验要点）
- [x] skill `math-explainer`（讲解质量约束全量 6 条）
- [x] 管线进度确认：`git log --oneline -12` → 当前 S3 门禁 1 审查（1.3 方案确认书已完成 2402d9f）

---

## 一、判定内容逐项结论

### 1. 方案合理性 —— ✅ 通过

**结论**：接受负结果 + 失败原因三分法归因框架完全基于共享事实，主实验 + 子实验完整，"负结果也是答案"定位有题面依据。

**依据**：
- 三分法归因的三个来源均配定量证据，且全部来自 A 类验证共享事实：批次效应（silhouette 0.070，F5）、疾病特异信号（IBD 衰减 −0.358，F3）、标签语义漂移（C3 灵敏度 0.006，F6）。approach §4.5 操作化定义表逐项给出可计算证据，与 `handoff-S3-model-agent-verify.md` §二 数字一致。
- 主实验（§1.2：LODO 3 组合 + AUC 主评估 + 训练集 Youden J 阈值迁移禁测试集重定）+ 子实验 1（§1.3：特征交集 344 物种）+ 子实验 2（§1.4：Platt 校准）完整，与 debate §5.3 推荐方案（Alpha 骨架 + Beta 两子实验 + Gamma 重加权可选）一致。
- "负结果也是答案"定位有题面依据：`problem-statement.md` 第 25 行注1「利用一种或两种疾病作为训练集，另外一种新的疾病作为测试集」+ 第 49 行隐含约束「跨疾病预测大概率性能显著下降……需要合理的分析与讨论而非只报数字」。approach §0/§10 的收束与题面一致。

**证据路径**：`approach-S3-confirmed.md` §1.1/§1.2/§1.3/§1.4/§4.5/§8；`handoff-S3-model-agent-verify.md` §二 A1/A3/A5；`problem-statement.md` L25/L49。

### 2. 推导自洽 —— ✅ 通过

**结论**：LODO 协议、AUC、Youden J、Platt 校准公式符号定义与读法齐全，口径与 S1 一致。

**依据**：
- §4.1 LODO（留一疾病，3 组合定义 + 测试集正类占比表）、§4.2 AUC（曲线下面积 + 阈值无关理由 + 解读基准）、§4.3 Youden J（$J=\text{TPR}+\text{TNR}-1$ + 阈值迁移规则 + 禁测试集重定）、§4.4 Platt 缩放（$P_{\text{cal}}=1/(1+\exp(Af+B))$ + 参数仅训练集估计 + 不改 AUC 理由）、§4.5 三分法（silhouette 系数定义）——均首次出现给定义与读法，符合 math-explainer 约束 1。
- 口径与 S1 一致：§5 求解方法 step 1 明确 CLR（δ=0.65×检出限=6.5e-6）+ StandardScaler 仅训练集估计；step 2 明确 `LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=2000)`。与 `proxy-replacement-checklist-S3.md` #5/#6/#7（δ=6.5e-6、C=1.0、class_weight=balanced 均标注"沿用 S1 口径"）一致。
- 主/次目标声明（§3）：S3 无主次目标结构，AUC 唯一主指标、辅指标为分解视图，符合 TRAE-建模.md 1.3 第 4 条强制要求。

**证据路径**：`approach-S3-confirmed.md` §3/§4/§5；`proxy-replacement-checklist-S3.md` #5/#6/#7。

### 3. handoff 可执行性 —— ✅ 通过

**结论**：数据接口、预期输出结构、参考实现、已知风险齐全，No Placeholders 满足。

**依据**：
- 数据接口（§三）：`c-data-cleaned.pkl`，484 样本 ×（2 元数据列 + 1331 特征列），元数据列 `dataset_name`/`disease`，三数据集样本量、患病判定口径、特征名格式（7 级分类学层级）均给出。
- 预期输出（§四）：`S3-results.pkl` 结构含 `lodo_main`/`domain_auc`/`decay_attribution`/`subexp1_intersection`/`subexp2_platt`/`migration_analysis`/`threshold_drift`，字段级定义完整。
- 参考实现（§五）：6 个 verify 脚本（`verify_S3_common.py` + `verify-S3-a1~a5-*.py`）+ 5 张探索图，路径明确。
- 已知风险（§六）：负结果预期、IBD 衰减最大（−0.358）、C3 灵敏度 0.006、阈值迁移禁测试集重定，均配应对。
- No Placeholders（§七）：无 `TODO`/`TBD`/`待定`，代理值（seed/Platt 迭代数/重加权参数）指向 `proxy-replacement-checklist-S3.md`，均为可执行具体值。

**证据路径**：`handoff-S3-code-agent.md` §三/§四/§五/§六/§七。

### 4. 待裁定项处置 —— ✅ 通过

**结论**：[A级] 属级聚合替代策略明确采纳、[B级] 概率校准纳入、[B级] 重加权降级可选，处置完整且与 debate 一致。

**依据**：
- [A级] 属级聚合被证伪（F4：物种 0.556→属 0.539→门 0.528）→ approach §9 明确采纳替代策略（转向"接受负结果 + 归因"），属级聚合降级为可解释性/降维手段（§2 H5、§1.1 主框架），含人话一句话/影响判定/拟议裁定/处置意见/回滚路径/级别，符合待裁定项自足模板。
- [B级] 概率校准（Platt）→ §9 明确纳入（§1.4 子实验 2），参数仅训练集估计，不改 AUC 主指标。
- [B级] Gamma 重加权 → §9 降级为可选排除实验（§1.5），对抗式域适应砍掉（F5 批次不主导 + 小样本过拟合）。
- 与 debate §6 B1/B2/B3、§7 待裁定项处置建议一致。

**证据路径**：`approach-S3-confirmed.md` §9/§1.1/§1.4/§1.5/§2；`debate-S3.md` §6/§7。

### 5. 评估泄漏防护 —— ✅ 通过（附 1 项轻微提示）

**结论**：阈值迁移只用训练集、特征选择无标签泄漏、StandardScaler/Platt 参数仅训练集估计，防泄漏措施贯穿。

**依据**：
- 阈值迁移：§4.3 明确 $\tau^*$ 只在训练集估计、禁测试集重定（会用到新疾病标签造成泄漏）；handoff §2.3 同款硬约束，并标注"代码审查（1.5/2.1.5）重点检查"。
- 特征选择：子实验 1 特征交集基于特征名（344 物种），不含测试标签→无泄漏（§1.3、handoff §2.4）。
- 预处理：StandardScaler 均值/方差仅训练集估计（§5 step 1）；Platt 参数 A,B 仅训练集估计（§4.4）；重加权转导式边界显式声明（§1.5）。
- 注：判定内容第 5 条"特征选择在 CV 折内"在 S3 的 LODO 语境下对应"特征交集基于特征名（非标签）"，已满足防泄漏语义（见问题清单 #3 的转导式边界提示）。

**证据路径**：`approach-S3-confirmed.md` §4.3/§4.4/§5/§1.3/§1.5；`handoff-S3-code-agent.md` §2.3/§2.4。

---

## 二、问题清单

| # | 问题 | 严重度 | 证据路径 |
|---|---|---|---|
| 1 | **数据源不一致**：共享事实（AUC 0.556、silhouette 0.070、衰减量 −0.358、344 共享物种）来自 A 类验证的 `B-raw.pkl`，而方案/交接的数据接口为 `c-data-cleaned.pkl`（0.3 清洗后），该文件**当前尚未生成**（`outputs/data/` 仅含 `B-raw.pkl`）。0.3 清洗（去重/填补/领域排除）可能改变样本量与标签，共享事实数字需在 1.4 用清洗后数据复核。缓解：handoff §三 已提示"清洗后以实际为准"、approach §6.1 已提示"正式实现后以 S3-results.pkl 为准"，故不阻断门禁 1，但须在 1.4 显式复核。 | 中等（B级，不阻断） | `handoff-S3-model-agent-verify.md` §一（数据=B-raw.pkl）；`approach-S3-confirmed.md` L4 + `handoff-S3-code-agent.md` §三（数据接口=c-data-cleaned.pkl）；`outputs/data/` 实测仅含 B-raw.pkl |
| 2 | **"AUC 尚可（0.56）"表述与自身解读基准矛盾**：§4.2 明确"<0.60 接近随机"，但 §1.1 第 3 点与 §6.4 写"AUC 尚可（0.56）"。0.56 按 §4.2 基准是"接近随机"而非"尚可"，属表述层不一致，可能引起读者误解。 | 低 | `approach-S3-confirmed.md` §4.2 vs §1.1/§6.4 |
| 3 | **子实验 1 特征交集筛选的"转导式"边界未显式声明**：特征交集（三数据集共享 344 物种）使用了测试集的特征存在信息（非标签），§1.3 论证"不含测试标签→无泄漏"成立，但未像 §1.5 重加权那样显式声明"转导式边界"。建议 §1.3 补一句与 §1.5 一致的显式声明，保持防泄漏口径统一。 | 低 | `approach-S3-confirmed.md` §1.3 vs §1.5 |
| 4 | **Platt 缩放 A>0 假设未显式约束**：§4.4 写"单调递增变换（A>0 时）"，但未约束/校验 A>0。若拟合得 A<0 会反转排序、改变 AUC，违背"不改 AUC"目标。正常模型下 A 自然为正，但建议实现时显式校验。 | 低 | `approach-S3-confirmed.md` §4.4 |

**交叉引用提示（供主建模自检核实，非本审查问题）**：`maintenance/registry.md` L10 登记 `small_adenoma 口径 [B级]` 待裁定项，触发时机为"门禁 1 方案确认前"，当前仍 `active`。其拟议为"主口径按题面保留（small_adenoma 归健康对照）+ S1 做敏感性分析"，与 S3 方案标签口径（small_adenoma=健康对照）一致，故不阻断 S3 门禁 1，但主建模裁决前应确认该待裁定项已按拟议销项或明确 S3 不受其影响。

---

## 三、结论

**通过**（附 4 项问题，均不阻断门禁 1，其中 #1 为中等/B级须在 1.4 预处理时显式复核，其余为低严重度表述/口径一致性提示）。

方案框架（接受负结果 + 三分法归因 + 深度迁移分析）基于共享事实、推导自洽、handoff 可执行、待裁定项处置完整、防泄漏措施贯穿，符合门禁 1 判定内容全部 5 项要求。问题清单无 A 级阻断项，可放行至 1.4 预处理（与建模 2.0 并行）。

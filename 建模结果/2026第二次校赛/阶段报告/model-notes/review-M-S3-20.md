# 门禁 M 审查结论：S3 2.0 数学推导（review-M-S3-20）

> 审查代理角色：建模对话（modeling Preset）| 模型：deepseek-v4-pro:0813
> 审查对象：`solution/model-notes/math-S3.tex`（17 节 493 行）+ 编译 PDF（11 页）
> 审查日期：2026-08-21 | 门禁：M（汇聚门禁，2.0 数学推导部分；1.4 预处理由 coding Preset 另审）
> 运行模式：auto（审查/裁决分离——本文件为审查结论，裁决由主建模执行）

## 必读清单已读汇报

- [x] `TRAE-建模.md`（2.0 数学推导门禁 + 方案讲解质量约束 6 条）
- [x] `TRAE.md`（门禁 M 判定内容 + 门禁协议审查/裁决分离）
- [x] skill `math-explainer`（讲解质量约束 6 条，审查对照）
- [x] 管线进度确认：`git log --oneline -10`——1.4 已完成 `06b19ac`，2.0 已完成 `1b4102e`，当前 S3 门禁 M 审查
- [x] 对照材料：`approach-S3-confirmed.md`、`handoff-S3-code-agent.md`、`proxy-replacement-checklist-S3.md`、`latex/templates/internal-report.cls`

---

## 判定内容逐项结论

### 1. 数学推导自洽

**结论：通过。**

**依据**：逐项对照 `approach-S3-confirmed.md` §4 数学框架，math-S3.tex 的推导与方案确认书完全一致，无代数错误：

- **LODO 协议形式化**（math §2）：$\mathcal{T}_k=\bigcup_{d\neq d_k}\mathcal{S}_d$、$\mathcal{E}_k=\mathcal{S}_{d_k}$，3 组合 C1/C2/C3 的训练/测试集划分与正类占比（40%/23%/65%）与 approach §4.1 逐字一致。
- **四策略公式**：
  - 策略 A（math §4）：$P(y=1|\mathbf{x})=\sigma(\mathbf{w}^\top\mathbf{z}+b)$，L2 目标 $\min\frac12\|\mathbf{w}\|_2^2+C\sum_i w_i[-\cdots]$ 与 sklearn `LogisticRegression(penalty='l2', C=1.0, class_weight='balanced')` 口径一致（sklearn 目标即 $\frac12\|\mathbf{w}\|^2+C\sum\text{loss}$）。
  - 策略 B（math §5）：$\mathcal{F}_{\text{shared}}=\mathcal{F}_{\text{CRC}}\cap\mathcal{F}_{\text{IBD}}\cap\mathcal{F}_{\text{Obesity}}$，转导式边界显式声明（B5 裁定）与 approach §1.3 一致。
  - 策略 C（math §6）：$x_g=\sum_{j:\text{species }j\in\text{genus }g}x_j$ 属/门级聚合，如实报告不提升，与 approach §1.4 一致。
  - 策略 D（math §7）：$P_{\text{cal}}=1/(1+\exp(A\cdot f+B))$，$A>0$ 单调性论证 + B6 裁定校验，与 approach §1.5/§4.4 一致。
- **Youden J 阈值迁移**（math §8）：$J(\tau)=\text{TPR}+\text{TNR}-1=\text{TPR}-\text{FPR}$（代数正确，因 $\text{TNR}=1-\text{FPR}$），$\tau^*=\arg\max_\tau J(\tau)$，禁测试集重定阈值，与 approach §4.3 一致。
- **回退协议形式化**（math §10）：触发条件 $\overline{\text{AUC}}_s<0.60\ \forall s$、R1-R4 候选、可用线 $\ge0.65$ 或提升 $\ge0.10$，与 approach §1.6 一致。
- **三分法归因操作化**（math §11）：三互斥来源 + 定量证据（silhouette 0.070 / 衰减量 −0.358 / C3 灵敏度 0.006），与 approach §4.5 一致。

**代数核验**：CLR $g(\mathbf{x})=(\prod_j x_j)^{1/p}$、AUC $=\int_0^1\text{TPR}(\text{FPR}^{-1}(t))dt=P(f(\mathbf{x}_+)>f(\mathbf{x}_-))$、silhouette $s(i)=(b(i)-a(i))/\max\{a(i),b(i)\}$、密度比 $w(\mathbf{x})=p_{\text{test}}/p_{\text{train}}$ 均正确。

**证据路径**：`solution/model-notes/math-S3.tex` §2/§4-§11；`solution/model-notes/approach-S3-confirmed.md` §4。

### 2. 可实现性

**结论：通过。**

**依据**：每个公式符号定义与读法齐全（§1 符号表覆盖 $\mathcal{D},\mathcal{S}_d,\mathbf{x}_i,y_i,\mathcal{T}_k,\text{clr},g,\delta,\mathbf{w},b,\sigma,f,\tau,A,B,\text{TPR/FPR/TNR},\Delta_d$），无歧义；与 handoff 规格一致——handoff §2.2 给出精确 sklearn 参数（`penalty='l2', C=1.0, class_weight='balanced', max_iter=2000`），§2.3-2.8 给出阈值迁移/策略 B/C/D/回退/评估指标的实现规格，与 math 推导一一对应。代理值（seed=42、Platt max_iter=2000、δ=6.5e-6、C=1.0）均在 `proxy-replacement-checklist-S3.md` 登记为可执行具体值，无占位符。

**证据路径**：`solution/model-notes/math-S3.tex` §1；`solution/model-notes/handoff-S3-code-agent.md` §2；`solution/model-notes/proxy-replacement-checklist-S3.md`。

### 3. 防泄漏形式化

**结论：通过。**

**依据**：math §12 统一表达 $\theta=\theta(\mathcal{T}_k)$（任意估计参数只能是训练集函数），并给出 7 项参数（近全零过滤阈值/CLR δ/StandardScaler 均值方差/Logistic 权重/Youden 阈值/Platt 参数/密度比）的估计来源与泄漏风险表；§12.2 对 R3 密度比的转导式边界给出精确表述 $w(\mathbf{x})=w(\mathbf{x};\{\mathbf{x}_i\}_{i\in\mathcal{T}_k\cup\mathcal{E}_k})$，不含 $\{y_i\}_{i\in\mathcal{E}_k}$。阈值/校准/特征选择仅训练集估计的约束在 §3.4/§8.2/§7.1 分别落实，数学表达完整。

**证据路径**：`solution/model-notes/math-S3.tex` §3.4/§7.1/§8.2/§12。

### 4. 假设检验计划 + 方案讲解质量

**结论：通过。**

**依据**：
- **假设检验计划**（math §15）：H2-H6 逐条给检验方法与预期结果（衰减量排序 + silhouette、方向一致性 + 符号/置换检验、校准前后辅指标对比、物种/属/门 AUC 对比、R1-R4 vs 可用线），不因模型类型跳过；方向一致性分析明确「不裸报描述统计」。
- **方案讲解质量**（math §14 自检 + 逐节核验，对照 math-explainer 6 条）：
  1. 先定义再使用——LODO/CLR/AUC/Youden J/Platt/silhouette/三分法/标签语义漂移/密度比均首次出现给定义与读法（§1 符号表 + 各节定义框）。
  2. 每步有「因为」——AUC 选主指标（阈值无关最诚实）、L2 而非 L1（共线特征更稳定）、聚合不提升（求和抹平方向）、Platt 不改 AUC（单调保排序）均显式给理由。
  3. 多视角——几何（AUC=ROC 下面积、Youden J=45° 线最远点、CLR=对数比偏离）、生物学（疾病特异信号=共享物种方向翻转）、统计（silhouette 近 0=不清晰分簇）。
  4. 核心洞察收束——§16 一句话「解决什么/怎么做/为什么」。
  5. 直觉解释——§2 医生类比、§7 温度计刻度类比、§8 Youden 平衡点类比、§10 医生误诊三原因类比。
  6. 防跳跃——§17 列 6 个需展开概念点。

**证据路径**：`solution/model-notes/math-S3.tex` §14-§17。

### 5. 编译质量

**结论：通过。**

**依据**：xelatex 编译成功，输出 11 页 PDF（`math-S3.pdf`，307666 字节，时间戳 12:11:41 晚于 tex 12:11:33，为最新编译）。log 无 `!` 错误、无 undefined/multiply defined 引用；tex 与 aux 均无 `??` 未解析引用（本文件无 `\ref`/`\label` 交叉引用，故无引用解析风险）。仅 8 处 Overfull/Underfull 排版警告（美观层，不影响内容）。

**证据路径**：`solution/model-notes/math-S3.pdf`、`math-S3.log`、`math-S3.aux`。

---

## 问题清单

| # | 问题 | 严重度 | 证据路径 |
|---|---|---|---|
| 1 | 符号 `w` 三处复用：权重向量 $\mathbf{w}$（§1/§4）、类别权重 $w_i$（§4.2）、密度比权重 $w(\mathbf{x})$（§10.2.1），符号表仅列 $\mathbf{w}$，$w_i$ 与 $w(\mathbf{x})$ 未入符号表。各节上下文定义清晰、handoff §2.2 以 `class_weight='balanced'` 消歧，实现无歧义，但建议符号表补定义以防误读 | 低 | `math-S3.tex` §1/§4.2/§10.2.1 |
| 2 | R3 密度比估计器（KLIEP/uLSIF）与权重裁剪上界未最终确定，登记为代理值 #3（可选回退实验，仅四策略全败时触发，不阻断主结论） | 低 | `math-S3.tex` §10.2.1；`proxy-replacement-checklist-S3.md` #3 |
| 3 | 8 处 Overfull/Underfull 排版警告（表格列宽略超，美观层，不影响内容正确性） | 低 | `math-S3.log` |

> 无 A 级（阻断）问题；无 B 级（并行）问题。上述 3 项均为低严重度，不阻断门禁 M 放行，建议在 2.1 实现或后续修订时顺手处理。

---

## 结论

**通过。**

S3 2.0 数学推导（`math-S3.tex`）在五项判定内容上全部通过：数学推导自洽（与 approach §4 一致、无代数错误）、可实现（符号定义齐全、与 handoff 规格一致）、防泄漏形式化完整（$\theta=\theta(\mathcal{T}_k)$ + 转导式边界精确表述）、假设检验计划 + 方案讲解质量达标（math-explainer 6 条全满足）、编译质量合格（xelatex 通过、无 `??`）。问题清单 3 项均为低严重度，不构成放行障碍。

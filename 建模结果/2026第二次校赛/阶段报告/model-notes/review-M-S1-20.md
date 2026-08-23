# 门禁 M 审查结论：S1 2.0 数学推导（math-S1.tex）

> **审查代理角色**：建模对话（modeling Preset）审查代理
> **模型**：deepseek-v4-pro:0813
> **审查对象**：`solution/model-notes/math-S1.tex`（10 节 322 行）+ 编译 PDF（`math-S1.pdf`，6 页）
> **门禁**：门禁 M（汇聚门禁，2.0 数学推导部分；1.4 预处理部分已由 coding preset 审查通过 `review-M-S1-14.md`）
> **日期**：2026-08-21

---

## 0. 必读清单已读汇报

已完整读取：
- `TRAE.md`（门禁 M 判定内容：预处理与模型匹配 + 数学推导自洽可实现；门禁协议审查/裁决分离；空洞审查检出强制）
- `TRAE-建模.md`（2.0 数学推导门禁 + 方案讲解质量约束 6 条 + 数学自检要求 + 假设检验计划要求）
- `math-explainer` skill（讲解质量约束 6 条：先定义再使用 / 每步因为 / 多视角 / 核心洞察收束 / 直觉解释 / 防跳跃）
- 对照材料：`approach-S1-confirmed.md`（方案确认书）、`handoff-S1-code-agent.md`（规格）、`latex/templates/internal-report.cls`（文档类）

管线进度确认（`git log --oneline -10`）：当前 S1 门禁 M 审查——1.4 已完成并审查通过（`ac1e64b` + `review-M-S1-14.md`），2.0 已完成（`3b2744c`）。`review-M-S1-20.md` 此前不存在，本次完整执行审查。

---

## 1. 判定内容逐项结论

### 1.1 数学推导自洽 —— ✅ 通过

**结论**：CLR 公式、Logistic 损失 + L2 + class_weight 公式与 approach §2 数学框架一致，无代数错误。

**依据**：
- **CLR 零值乘法替换**（`math-S1.tex` §2.2，eq:zeroreplace，L84-89）：$\delta = 0.65\times10^{-5} = 6.5\times10^{-6}$，与 approach §2.2 及 handoff §1.2 完全一致；$0.65\times10^{-5}=6.5\times10^{-6}$ 数值正确。
- **CLR 几何均值中心化**（eq:clr，L96-102）：$\mathrm{clr}(x_{ij})=\ln x_{ij}-\frac{1}{p}\sum_k\ln x_{ik}=\ln\frac{x_{ij}}{g_i}$，$g_i=(\prod_k x_{ik})^{1/p}$，代数恒等正确；中心化正确性在 §8 自检第 4 条显式验证（$\sum_j\mathrm{clr}(x_{ij})=0$ 恒成立）。
- **Logistic 损失**（eq:loss，L125-130）：$-\sum_i[y_i\ln\sigma(z_i)+(1-y_i)\ln(1-\sigma(z_i))]+\frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2$，为标准二分类负对数似然 + L2 惩罚，无符号/求导错误。
- **class_weight**（eq:classweight，L140-145）：$w_c=n/(n_{\text{classes}}\times n_c)$，为 sklearn `balanced` 标准公式；代入 metahit $w_1=110/(2\times25)=2.2$、$w_0=110/(2\times85)=0.647$，比值 $3.4$ 倍，数值正确；归一性在 §8 自检第 5 条验证（$\sum_c n_c w_c=n$）。
- **量纲自洽**（§8 自检第 2 条，L270）：CLR 后特征无量纲（对数比）→ 线性得分无量纲 → sigmoid 概率无量纲 → 损失无量纲，全链路自洽。
- **无「用因变量预测因变量」循环**（§8 自检第 1 条，L269）：自变量为 $\mathrm{clr}(\mathbf{x}_i)$，$y_i$ 仅作监督信号。

**证据路径**：`solution/model-notes/math-S1.tex` §2.2/§3.2/§3.3/§8；对照 `approach-S1-confirmed.md` §2.2-2.4。

---

### 1.2 可实现性 —— ✅ 通过

**结论**：每个公式符号定义与读法齐全（math-explainer 约束 1），与 handoff 规格一致，代码对话可无歧义实现。

**依据**：
- **符号表齐全**（§1.1，L22-48）：$n/p/x_{ij}/\delta/g_i/\mathrm{clr}/\mathbf{w}/b/z_i/\sigma/y_i/\lambda/C/w_c/K/z_j$ 共 15 个符号，每个均含「读法 + 定义」，满足「先定义再使用」。
- **与 handoff 规格逐项一致**：
  - 过滤 264 维（$p_0=1331\to p=264$，阈值 $\tau=0.95$，剔除 1067 特征）——`math-S1.tex` §2.1 与 handoff §1.2 一致；$1331-264=1067$ 正确。
  - $\delta=6.5\times10^{-6}$——一致。
  - $C=1.0$（$\lambda=1.0$）——§3.2 L135 与 handoff §1.3 一致。
  - $K=5$、seed=42——§5.3 与 handoff §1.5 一致。
  - RF 规格（n_estimators=500, max_depth=None, min_samples_leaf=1, random_state=42）——§4 L160 与 handoff §1.4 一致。
- **关键正确性亮点**：`math-S1.tex` 正确区分 $p_0=1331$（原始）与 $p=264$（过滤后），CLR 几何均值中心化用 $p=264$（过滤后维数），与 handoff §1.2「$p=264$（过滤后维数）」一致。（注：approach §2.2 的 CLR 公式沿用其符号表 $p=1331$，属 approach 的轻微不一致，非本审查对象；math-S1.tex 已正确修正。）

**证据路径**：`math-S1.tex` §1.1/§2.1/§3.2/§4/§5.3；对照 `handoff-S1-code-agent.md` §1.2-1.5。

---

### 1.3 假设检验计划 —— ⚠️ 基本通过（1 项 B 级缺口）

**结论**：H1-H5 + B3/B4 中，H1/H2b/H3/H4/H5/B3/B4 均有检验方法，但 **H2（零值处理 δ 乘法替换）缺检验方法行**。

**依据**：
- §10 假设检验计划表（L297-312）逐条核对：
  - H1 成分数据需 CLR → 对比 CLR 前后 AUC（F4）✅ 已证
  - H2b 近全零过滤 → 过滤前后 AUC 对比 ✅ 待验证
  - H3 四口径 → 四口径各跑主模型+对照 ✅ 待验证
  - H4 独立建模 → 三数据集分别建模 ✅ 已证
  - H5 无过拟合 → 全量 vs LOOCV 差距 ≤0.1 ✅ 待验证
  - B3 class_weight → 前后 Recall 对比 ✅ 待验证
  - B4 离群样本 → 剔除 14 样本重训对比 AUC ✅ 待验证
- **缺口**：§7 简化假设表（L256-262）列出 H2「零值处理（92.21% 零值视为低于检出限，乘法替换 δ）」，其合理性列声称「δ 扰动不翻转结论」，但 §10 检验计划表**无 H2 对应行**——H2 的 δ 敏感性检验方法未在检验计划中显式列出（辩论 §3.1 反驳曾论证，但未落到本文件的检验计划表）。

**证据路径**：`math-S1.tex` §7（L256-262）vs §10（L297-312）。

---

### 1.4 方案讲解质量（math-explainer 6 条）—— ✅ 通过

**结论**：6 条约束全部满足。

**依据**（逐条对照）：
1. **先定义再使用**：§1.1 符号表 15 符号均含读法；正文中 $\mathbf{1}[\cdot]$（L61）、simplex（L105）等新概念首次出现即给读法。✅
2. **每步有「因为」**：CLR 动机（L80）、零值替换（L92）、中心化（L105）、sigmoid（L121）、L2（L133）、class_weight（L148）、AUC（L185）、CV（L211）均标注「因为」+ 具体数值/理由。✅
3. **多视角**：CLR 的几何解释（单纯形 → 对数空间超平面投影，L105-106）。✅
4. **核心洞察收束**：§11 收束「解决什么/怎么做/为什么」三问（L316-318）。✅
5. **直觉解释**：§9 整节无公式直觉解释（L291-293）+ 各节内联类比（问卷题、橡皮筋、考试扣分、抽签概率）。✅
6. **防跳跃**：§11 末尾标注 5 个可能需展开的概念点（L320）。✅

**证据路径**：`math-S1.tex` §1.1/§2.2/§3/§5/§9/§11。

---

### 1.5 编译质量 —— ✅ 通过

**结论**：xelatex 编译通过，无 `??` 未解析引用。

**依据**：
- `math-S1.log`：无 `!` 错误行，无 undefined/multiply defined 引用，`LaTeX Warning` 计数 0，`Output written on math-S1.pdf (6 pages)`。
- `math-S1.aux`：全部 `\newlabel`（sec:preprocess/sec:filter/sec:clr/sec:classweight/sec:eval + eq:zerofrac/eq:filter/eq:zeroreplace/eq:clr/eq:logistic/eq:loss/eq:classweight/eq:auc/eq:f1/eq:cv/eq:loocv）均已解析，无 `??`。
- PDF 时间戳（12:10:08）晚于 .tex（12:09:47），为当前 .tex 的编译产物，可复验。

**证据路径**：`solution/model-notes/math-S1.log`、`math-S1.aux`、`math-S1.pdf`。

---

## 2. 问题清单

| # | 问题 | 严重度 | 证据路径 |
|---|------|--------|----------|
| 1 | **H2（零值处理 δ 乘法替换）在假设检验计划表缺检验方法行**：§7 声称「δ 扰动不翻转结论」，但 §10 检验计划表无 H2 对应行，未显式给出 δ 敏感性检验方法（辩论 §3.1 曾论证但未落到本文件检验计划） | B 级（不影响数值正确性，δ 已固定为 AL-007 标准值；检验计划完整性缺口） | `math-S1.tex` §7 L256-262 vs §10 L297-312 |
| 2 | **符号 `z` 重载**：`z_i`（线性得分，L38）与 `z_j`（零值占比，L45）共用字母 `z`，靠下标 i/j 区分，可读性略降 | B 级（表述层，下标已消歧，不影响实现） | `math-S1.tex` §1.1 L38/L45 |
| 3 | **损失函数未显式含 class_weight**：eq:loss（L125-130）为未加权交叉熵形式，而 §3.3 说明 class_weight 放大少数类损失；加权形式（$w_{y_i}$ 乘各项）未在公式中示出，仅靠 prose 说明 | B 级（表述层，handoff §1.3 已明确 `class_weight='balanced'`，不影响实现） | `math-S1.tex` §3.2 L125-130 vs §3.3 L137-150 |

> 三项均为 B 级（可并行、不影响正确性、不影响下游按现有规格实现），无 A 级阻断项。

---

## 3. 结论

**通过** ✅

数学推导自洽（CLR/Logistic 损失/class_weight 公式无代数错误，量纲自洽）、可实现（符号定义与读法齐全，与 handoff 规格逐项一致）、讲解质量达标（math-explainer 6 条全满足）、编译通过（无 `??`）。假设检验计划基本完整，存在 1 项 B 级缺口（H2 缺检验方法行）与 2 项 B 级表述问题，均不阻断，建议在 2.1 实现或结果分析阶段补记 H2 的 δ 敏感性检验（或注明已在辩论 §3.1 验证），其余 B 级项可批量销项。

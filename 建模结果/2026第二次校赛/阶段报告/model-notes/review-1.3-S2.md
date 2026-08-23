# 门禁 1 审查结论：S2 建模方案定稿（review-1.3-S2）

> 审查代理角色：门禁 1 审查代理（modeling Preset，审查/裁决分离协议——本文件为审查结论，主建模只裁决）
> 模型：deepseek-v4-pro:0813 | 思考强度：max
> 日期：2026-08-21 | 阶段：1.3 方案确认（门禁 1：建模定稿）
> 审查对象：`approach-S2-confirmed.md`、`handoff-S2-code-agent.md`、`proxy-replacement-checklist-S2.md`
> 对照材料：`decision-tree-S2.md`、`debate-S2.md`、`handoff-S2-model-agent-verify.md`、`knowledge/method-graph.md`、S1 的 `proxy-replacement-checklist-S1.md` / `approach-S1-confirmed.md`（δ 口径核对）

## 必读清单已读汇报

- [x] `TRAE-建模.md`（门禁协议：审查代理职责 + 结论文件格式 + 方案讲解质量约束）
- [x] `TRAE.md`（门禁 1 判定内容 + 四门禁总览 + handoff No Placeholders 约束）
- [x] skill `modeling-decision-tree`（题型判定 + 检验要点 + 评估标准）
- [x] skill `math-explainer`（讲解质量约束 6 条）
- [x] 管线进度确认：`git log --oneline -12` → 当前 S2 门禁 1 审查（1.3 方案确认书已完成 2881593）

---

## 判定内容逐项结论

### 1. 方案合理性 —— 通过

**结论**：主方法（Lasso+bootstrap 稳定性选择 τ=0.5）+ 近全零过滤（1331→264）+ CLR 前置 + 两路信号解释层（Fisher 存在/缺失 + Wilcoxon 非零丰度）+ 佐证层（RF/VIP>1.5）全部基于共享事实，稳定性策略（CV 折内选择防泄漏）到位。

**依据**：
- 主方法选择直击最大风险「小样本特征选择方差大」，bootstrap 聚合频率把选择不确定性显式量化（`decision-tree-S2.md` §3 推荐理由三选）。
- 近全零过滤（F3：1067/1331 特征 >95% 零值，过滤后 264 维）与 CLR 前置（F6：23~34% 特征对相关方向翻转）均为 A 类验证硬约束，非拍脑袋。
- 两路信号拆分（F2：零值占比差 corr(AUC) 0.86~0.89 > 非零丰度差 0.63~0.67）有数据支撑。
- 佐证层（RF 免 CLR + VIP>1.5）是 Gamma「多方法独立复现」思想的降级落地，`debate-S2.md` §4 已论证其合理性。
- 稳定性策略：`approach-S2-confirmed.md` §1.5 明确「所有特征选择在分层 CV 折内做（防泄漏）+ 报告全量（乐观）vs CV 内稳定频率（诚实）两套数字」，防泄漏到位。

**证据路径**：`approach-S2-confirmed.md` §1、§1.5；`debate-S2.md` §0（F1~F9）、§4；`handoff-S2-model-agent-verify.md` §1~§6。

### 2. 推导自洽 —— 通过（1 处口径歧义，见问题清单 Q2）

**结论**：Lasso 目标函数、bootstrap 频率公式、CLR、Wilcoxon、Fisher 精确检验、BH-FDR 的符号定义与读法齐全，公式正确；δ 口径与 S1 一致（S1 已定 δ=6.5e-06）。

**依据**：
- `approach-S2-confirmed.md` §3.1 符号约定表：$n/p/y_i/x_i/\beta_0/\beta_j/\lambda/\sigma/\hat{p}_i/B/\hat{\pi}_j/\tau/\mathbb{1}/g(x)/\binom{n}{k}$ 全部给定义与读法，满足「先定义再使用」。
- §3.2 Lasso 目标函数 = 负对数似然 + L1 惩罚，公式正确；§3.3 bootstrap 频率 $\hat{\pi}_j=\frac{1}{B}\sum_b\mathbb{1}\{\hat{\beta}_j^{(b)}\neq 0\}$ 正确；§3.4 CLR $\text{clr}(x)_j=\ln x_j-\frac{1}{D}\sum_k\ln x_k$ 正确；§3.5 Wilcoxon $U=\min(U_1,U_2)$ 正确；§3.6 Fisher 超几何概率正确；§3.7 BH-FDR $p_{(k)}\leq\frac{k}{m}\alpha$ 正确。
- δ 口径：S1 已定 δ=6.5e-06（0.65×检出限 1e-05，AL-007），S2 文档写「与 S1 一致」，方向正确（见 Q1 未落盘具体值）。
- 讲解质量约束 6 条：§10 逐条对应（先定义/每步因为/多视角/洞察收束/直觉解释/防跳跃），验收自检通过。

**证据路径**：`approach-S2-confirmed.md` §3、§10；`proxy-replacement-checklist-S1.md` P1~P3；`approach-S1-confirmed.md` §2.2。

### 3. handoff 可执行性 —— 通过（1 处占位符风险，见问题清单 Q1）

**结论**：数据接口、预期输出（S2-results.pkl 结构）、参考实现、已知风险（bootstrap 计算量 C8 并行提示）、No Placeholders 声明齐全，代码对话可无歧义开工。

**依据**：
- 数据接口：`handoff-S2-code-agent.md` §2 明确输入 `c-data-cleaned.pkl`（字段 `dataset_name`/`disease` + 1331 物种特征）+ 三病标签口径，并诚实标注「A 类验证用 B-raw.pkl 未清洗 vs 正式实现用 c-data-cleaned.pkl」的口径差异与复核触发条件。
- 预期输出：§3 给出 `S2-results.pkl` 完整结构（per_disease 稳定特征/两路信号/标志物表/RF/VIP/一致性 + cross_disease + meta），覆盖全部交付项。
- 参考实现：§4 列出 `verify-S2-v1~v6-*.py` + `utils.py`，可复用逻辑。
- 已知风险：§5 明确 bootstrap 计算量（C8 并行提示，超 2 分钟按 C4 交主会话后台）、τ 敏感性、δ 口径。
- No Placeholders：§6 声明所有参数取 proxy 清单当前临时值，不得留 TODO/占位符（但 `clr_delta: "<与S1一致>"` 为占位符字符串，见 Q1）。

**证据路径**：`handoff-S2-code-agent.md` §2~§6。

### 4. 待裁定项处置 —— 通过

**结论**：τ 下调（0.8→0.5）、VIP 上调（>1.5）、Obesity 弱信号诚实标注、small_adenoma 沿 S1 销项，四项处置清晰、依据充分。

**依据**：
- R1 τ 下调：F7（频率≥0.8 仅 2 特征/病，不足以支撑 Top 10~20），采纳下调至 0.5~0.6。
- R2 VIP 上调：F8（VIP>1 选中 40~55%，失去筛选意义），采纳上调至 >1.5 或分位数。
- R3 Obesity 弱信号：F1/F9（0 个 FDR 显著、AUC 0.639、bootstrap 入选 548 特征），采纳标志物可信度分级 + 低可信度标注。
- R4 small_adenoma：沿 S1 敏感性分析结论销项，S2 按「cancer=患病，n+small_adenoma=健康」执行，与 S1 口径一致；若 S1 敏感性显示口径显著改变 AUC，S2 CRC 特征选择同步重跑（记 A 级）。

**证据路径**：`approach-S2-confirmed.md` §9；`debate-S2.md` §7；`handoff-S2-model-agent-verify.md` §7。

### 5. B 类分歧 —— 通过

**结论**：τ 精确值、VIP 阈值幅度、Obesity 处理三项 B 类分歧清晰标记，不阻断 1.3。

**依据**：
- `debate-S2.md` §6 明确标记 B1（τ 0.5 vs 0.6）、B2（VIP 1.5 vs 分位数）、B3（Obesity 诚实标注 vs 降预期），均 B 级，拟议明确。
- `approach-S2-confirmed.md` §9 对应 R1/R2/R3，处置与 B 类分歧一致。
- 说明「B1/B2/B3 均不影响 1.3 方案确认，可并行推进，B 类验证结果在 1.4 预处理前回填」清晰。

**证据路径**：`debate-S2.md` §6；`approach-S2-confirmed.md` §9。

---

## 问题清单

| # | 问题 | 严重度 | 证据路径 |
|:--|:--|:--|:--|
| Q1 | **δ 值未落盘具体值**：S2 的 `proxy-replacement-checklist-S2.md` P6 写「与 S1 一致（乘法替换）」且状态「待定」，但 S1 已定 δ=6.5e-06（非待定）；`handoff-S2-code-agent.md` L105 的 `clr_delta: "<与S1一致>"` 是占位符字符串，违反 No Placeholders 纪律。建议 P6 直接落盘 δ=6.5e-06（0.65×检出限 1e-05），状态改「已定」，handoff 的 clr_delta 字段填具体值。 | B级 | `proxy-replacement-checklist-S2.md` P6；`handoff-S2-code-agent.md` L105；对照 `proxy-replacement-checklist-S1.md` P1~P3 |
| Q2 | **FDR 多重比较次数 m 口径歧义**：`approach-S2-confirmed.md` §1.3/§3.5 说「对入选的稳定特征（Top 10~20）做差异丰度检验」，但 §3.7 与 `proxy-replacement-checklist-S2.md` P3 写「m=1331 次比较」。若只对稳定特征检验，m 应为 ~10-20；若对全部 1331 特征检验，则与「对稳定特征做」矛盾。需在 1.4 实现前澄清：两路信号检验的总体是「全部 1331 特征（仅报告稳定特征）」还是「仅稳定特征」。 | B级 | `approach-S2-confirmed.md` §1.3、§3.5、§3.7；`proxy-replacement-checklist-S2.md` P3 |
| Q3 | **近全零过滤口径「每病独立 vs 三病并集」未完全裁定**：`approach-S2-confirmed.md` §4 步骤 1 写「每病独立计算零值占比，取三病并集或各病独立过滤（见 handoff 规格）」，但 §1.1 与 handoff §1.1 又给单数「保留 264 维」。若每病独立过滤，各病保留特征数可能 ≠264；若三病并集，则共享同一 264 维。需澄清过滤是「三病并集统一 264 维」还是「每病独立过滤（各病维度可能不同）」。 | B级 | `approach-S2-confirmed.md` §1.1、§4 步骤 1；`handoff-S2-code-agent.md` §1.1 |

---

## 结论：通过

- 方案合理性、推导自洽、handoff 可执行性、待裁定项处置、B 类分歧五项判定内容均通过。
- 3 个 B 级问题（Q1 δ 值落盘、Q2 FDR 多重比较口径、Q3 过滤口径）均不影响方案正确性与主方法/解释层/过滤规则，可并行推进，建议在 1.4 预处理实现前由建模对话澄清口径（Q1 直接落盘 δ=6.5e-06 即可销项）。
- 无 A 级（阻断）问题，无空洞审查（逐项结论 + 问题清单齐全）。

**建议**：主建模裁决「通过」，放行门禁 1；Q1~Q3 登记为 B 级待裁定项，随 1.4 实现前回填销项。

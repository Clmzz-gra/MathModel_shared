# 门禁 N 审查：review-N-S3-22.md

> **审查代理角色**：建模 Preset（modeling）| **模型**：deepseek-v4-flash:0731（本会话）| **审查对象**：S3 阶段 2.2 结果分析（门禁 N 的 2.2 部分）
> **日期**：2026-08-21 | **工作目录**：`E:\MathModel_pj-2026-sim2-B-S3`（worktree，分支 `experiment/2026sim2B-S3`）
> **审查范围**：对照门禁 N 判定内容（2.2 结果分析部分）——①结果分析合理性 ②待裁定项裁决合理性 ③handoff-report 完整性 ④数字一致性

---

## 一、必读清单已读汇报

| # | 必读文件 | 已读 | 用途 |
|---|---|---|---|
| 1 | `E:\MathModel_pj\TRAE-建模.md`（2.2 结果分析规范） | ✅ | 门禁 N 判定内容、2.2 步骤（拐角解/未闭合/落地检查/跨问不等式） |
| 2 | `E:\MathModel_pj\TRAE-规范.md`（A/B/C 相关节） | ✅ | 执行规范、产出格式、C1 代码头注释、C4 高耗时脚本 |
| 3 | `solution/model-notes/result-analysis-S3.md` | ✅ | 审查对象（2.2 结果分析产出） |
| 4 | `solution/model-notes/handoff-S3-model-agent.md` | ✅ | 代码→建模回报（结果摘要/待裁定项） |
| 5 | `solution/model-notes/handoff-S3-report-agent.md` | ✅ | 建模→报告交接（章节映射/关键数字/口径声明） |
| 6 | `outputs/data/S3-results.pkl` | ✅ | 只读提取关键字段（`outputs/scratch/review-N-S3-22-extract.py` → `.txt`） |

补充已读（裁决落地核验）：`math-S3.tex`、`approach-S3-confirmed.md`、`handoff-S3-code-agent.md`、`proxy-replacement-checklist-S3.md`（Platt 三处规格修正、R3 方法登记核验）。

---

## 二、判定内容逐项结论

### ① 结果分析合理性 — **通过**

- **结论**：result-analysis-S3.md 全部关键数字均取自 pkl 实测，结论与数据一致，未闭合清单完整（6 条 ≥ 规范要求 3 条），拐角解处理合理。
- **依据**：
  - 四策略 3 组合 AUC 均值全部 <0.60（A 0.5603 / B 0.5572 / C属 0.4639 / C门 0.5134 / D 0.5603）→ 触发紧急回退，与 pkl `fallback.triggered=True` 一致。
  - 回退 R1-R4 全部未达可用线，最优可达 R3_weighted 0.6068 < 0.65，`fallback.usable=False`、`delivered_strategy=None`，负结论有完整证据链支撑。
  - 拐角解：本问为分类评估（非优化），无「优化变量触及约束边界」概念，§三 头部已显式说明，处理合理。
  - 未闭合清单 6 条（R3 方法未登记 / R3 未达可用线 / 域内口径切换 / C3 灵敏度极低 / 共享物种方向接近随机 / R4 小样本过拟合），覆盖主要不确定事项。
  - 实际落地检查两条必答题（输入鲁棒性 + 输出成本）在 §四 均作答。
- **证据路径**：`result-analysis-S3.md` §0/§2.1/§2.2/§三/§四；`S3-results.pkl` `strategy_compare`/`fallback`/`exhausted_evidence`。

### ② 待裁定项裁决合理性 — **通过（Platt 修正已实际落地）**

- **结论**：handoff §四 五项待裁定项裁决均合理，Platt 符号修正已实际修改三处规格（非仅声明）。
- **依据**：
  - **#1 Platt 符号约定**：裁决「A>0→A<0」数学正确（$P_{\text{cal}}=1/(1+\exp(Af+B))$ 下 $\frac{dP}{df}=-\frac{A\exp(Af+B)}{(1+\exp(Af+B))^2}>0 \iff A<0$）。**落地核验**：`math-S3.tex`（L40 `A<0`、L193「A<0 时严格单调递增」、L201/L203「A<0 校验，A≥0 触发警告」）、`approach-S3-confirmed.md` §4.4（L138「单调递增（A<0 时）」、L140「校验 A<0，A≥0 触发警告」）、`handoff-S3-code-agent.md` §2.6（L72「校验 A<0」）三处均已改为 A<0。pkl 佐证：`D_calibrated.<C>.A` 三组合均负（−13.10/−21.85/−18.56）、`platt_w` 均正（13.10/21.85/18.56），与「A<0、w>0」一致。
  - **#2 域内 AUC 口径**：采用 264 特征重算域内（`domain_auc`），与跨疾病 264 特征同口径对比更严谨；A3 的 1331 参考（`domain_auc_reference_A3`）保留作对照。裁决合理。
  - **#3 R2≡A 冗余**：pkl 核验 `R2_pooled.mean_auc` = `A_direct.mean_auc` = 0.5603121010295473，逐组合 AUC 完全一致，数学恒等成立，登记为规格冗余合理。
  - **#4 R3 密度比方法**：确认接受域分类器法（Logistic 区分 train/test，w=exp(logit)×n_train/n_test，裁剪上界 10）。`proxy-replacement-checklist-S3.md` #3 已销项登记实际方法（L31），math-S3.tex §9.3 仍写「KLIEP 或 uLSIF」代理值——已如实记入未闭合清单 #1，报告口径声明已用正确方法。
  - **#5 pkl 结构命名**：按实际键名取数（`C_genus`/`C_phylum` 两键、顶层 `best_strategy=R3_weighted`、`strategy_compare.best_strategy=A_direct`），信息未丢失，裁决合理。
- **证据路径**：`result-analysis-S3.md` §一/§五；`math-S3.tex` L40/193/201/203/444；`approach-S3-confirmed.md` L138/140；`handoff-S3-code-agent.md` L72；`proxy-replacement-checklist-S3.md` L31；`S3-results.pkl` `D_calibrated`/`R2_pooled`/`A_direct`/`best_strategy`。

### ③ handoff-report 完整性 — **通过**

- **结论**：handoff-S3-report-agent.md 含章节映射、关键数字（来源可溯到 pkl 字段）、口径声明、图表清单规格、AI 标注，无占位符。
- **依据**：
  - §一 章节映射：7 行，每行标注内容来源 + pkl 字段。
  - §二 关键数字：四策略/回退/衰减/迁移/阈值漂移 5 张表，每行附 pkl 字段（如 `A_direct.<C>.auc`、`fallback.*`、`decay_attribution.<D>.*`）。
  - §三 口径声明：6 条（特征集 264 口径 / 主指标 AUC / Platt 符号 / R3 密度比方法 / R2≡A / C3 不可部署）。
  - §四 图表清单规格：3 图，每图含数据源 pkl 字段 + 图名 + 论文位置 + 说明。
  - §五 AI 标注：建模/代码/两遍审核 AI 贡献 + `[AI-X-Y]` 编号 + pkl 来源（meta.generated/seed/budget_limited）。
  - §六 No Placeholders：无 `TODO`/`TBD`/`待定`。
- **证据路径**：`handoff-S3-report-agent.md` §一~§六。

### ④ 数字一致性 — **通过**

- **结论**：result-analysis 与 handoff-report 的关键数字与 pkl 实际值全部一致（抽核 30+ 项，无一处不符）。
- **依据**（pkl 实际值 vs 文档值）：
  - 四策略均值：A 0.5603 / B 0.5572 / C属 0.4639 / C门 0.5134 / D 0.5603（pkl `*.mean_auc` 一致）。
  - 回退：R1 0.5092 / R2 0.5603 / R3 0.6068 / R4 0.5947（pkl `fallback.*.mean_auc` 一致）；R3 提升 +0.0465（0.6068−0.5603）。
  - 衰减：CRC −0.2138 / IBD −0.2706 / Obesity −0.1384（pkl `decay_attribution.<D>.decay` 一致）。
  - 迁移：387/369、n_valid 756、51.2%（0.5119）、p=0.5364（pkl `migration_analysis.*` 一致）。
  - 阈值漂移：train 0.3160 / test 0.6482 / Δ+0.3322 / τ*=0.9205 / 96.0% 分位（0.9605）/ 灵敏度 0.0244 / thr05 0.1646（pkl `threshold_drift.*`、`D_calibrated.C3.thr05_sensitivity` 一致）。
  - Platt 参数：A 三组合负、platt_w 三组合正（pkl `D_calibrated.<C>.A/platt_w` 一致）。
  - meta：seed=42、budget_limited=False、generated=2026-08-21T17:33:01（pkl `meta.*` 一致）。
- **证据路径**：`S3-results.pkl` 全字段 vs `result-analysis-S3.md` §2.x vs `handoff-S3-report-agent.md` §二。

---

## 三、问题清单

| # | 问题 | 严重度 | 证据路径 |
|---|---|---|---|
| 1 | **跨问不等式核验清单（2.2 step 8，门禁 N 必选项）未在 result-analysis-S3.md 中体现**。规范要求「与已合并子问题的 data-integration 登记数值做包含关系互证」，本问为分类评估（无面积/概率界等可互证数值），且 S3 为当前 worktree 首个达门禁 N 的子问题（无已合并子问题 data-integration 可对照），故可能「不适用」；但应显式声明「不适用/待后续子问题合并后核验」而非静默省略，避免门禁 N 必选项留白。 | 低 | `result-analysis-S3.md` 全文无「跨问/不等式/互证」；`TRAE-建模.md` 2.2 step 8；本 worktree 无 `solution/data-final/data-integration-*.md` |
| 2 | **R3 密度比方法规格未同步**：math-S3.tex §9.3 仍写「KLIEP 或 uLSIF」代理值，实际实现为域分类器法。虽已如实记入未闭合清单 #1 且 proxy checklist #3 已销项登记，但规格文档与实现仍存在代理值残留，报告期需在口径声明中明确（handoff-report §三.4 已覆盖）。 | 低 | `math-S3.tex` L296；`proxy-replacement-checklist-S3.md` L31；`result-analysis-S3.md` §三 #1 |

> 说明：以上 2 项均为低严重度、不改变结论的观察项，不构成门禁 N 阻断。问题 #1 建议主建模在门禁记录中确认跨问不等式核验的处置（不适用声明或后续补核验）。

---

## 四、结论

**门禁 N（2.2 结果分析部分）：通过。**

- 结果分析基于 pkl 实际数字，结论与数据一致，未闭合清单完整。
- 五项待裁定项裁决合理，Platt 符号修正已实际落地三处规格（math-S3.tex / approach / handoff-code）。
- handoff-report 完整（章节映射/关键数字可溯/口径声明/图表规格/AI 标注/无占位符）。
- 关键数字与 pkl 全部一致（抽核 30+ 项无出入）。
- 2 项低严重度观察项（跨问不等式核验未显式声明、R3 方法规格残留），不阻断，建议主建模在门禁记录中确认处置。

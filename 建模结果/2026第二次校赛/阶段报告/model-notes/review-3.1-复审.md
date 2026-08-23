# review-3.1-复审.md — 阶段 3.1 报告修正复审结论

> 阶段：3.1 报告修正复审（跨问审查 P1-P7 修正 diff 复审）| 日期：2026-08-21 | 运行模式：auto
> 审查代理角色：批判阅读审查代理（modeling preset）| 模型：deepseek-v4-flash:0731
> 审查对象：阶段 3.1 修正后的 S3 报告 / S2 报告 / S3 方案确认书 / S2 结果分析（对照 review-3.0-跨问.md 问题清单 P1-P7）
> 数字口径：**只取 pkl 实际值**（`S1/S2/S3-results.pkl` python 只读提取）

---

## 一、必读清单已读汇报

| # | 必读项 | 状态 |
|:--|:--|:--|
| 1 | `solution/model-notes/review-3.0-跨问.md`（原跨问审查结论，含 P1-P7 问题清单） | ✅ 已读 |
| 2 | `solution/model-notes/gate-3.0.md`（门禁记录，含修正方案） | ✅ 已读 |
| 3 | `solution/internal-reports/iter-02-sub3-cross-disease.tex`（S3 报告，修正后） | ✅ 已读 |
| 4 | `solution/internal-reports/iter-02-sub2-biomarker.tex`（S2 报告，修正后） | ✅ 已读 |
| 5 | `solution/model-notes/approach-S3-confirmed.md`（S3 方案，修正后） | ✅ 已读 |
| 6 | `solution/model-notes/result-analysis-S2.md`（S2 结果分析，修正后） | ✅ 已读 |
| 7 | `outputs/data/S1-results.pkl` / `S2-results.pkl` / `S3-results.pkl`（python 只读提取关键字段） | ✅ 已读 |

---

## 二、P1-P7 逐项复审结论

### P1（高）S3 域内 AUC 与 S1 不一致且未解释 —— ✅ 修正到位

- **修正内容**：S3 报告 §6.1 衰减归因表前新增**口径声明**：「本问预处理含 StandardScaler（均值/方差仅训练集估计），与 S1/S2 的纯 CLR 口径不同，故域内 AUC 与 S1 有差异（Δ −0.010/−0.028/+0.014，CRC/IBD/Obesity）」。
- **与 pkl 一致性**：S3 `domain_auc` = 0.7811/0.8588/0.6638；S1 `L2_CLR.AUC` = 0.7907/0.8871/0.6496。Δ = S3−S1 = −0.0096/−0.0283/+0.0142，四舍五入后 **−0.010/−0.028/+0.014**，与报告声明逐项一致。✅
- **结论**：P1 销项。

### P2（中）S3 报告 silhouette「未单独量化」与 0.070 矛盾 —— ✅ 修正到位

- **修正内容**：S3 报告 §6.1 三分法批次效应项补回「定量证据 silhouette 系数（数据集标签）= 0.070（A 类验证值，近 0，不主导）」。
- **与 pkl 一致性**：0.070 为 A 类验证值（`handoff-S3-model-agent-verify.md` F5），非正式 pkl 字段；与 approach §4.5 / result-analysis-S3 §2.3 的 0.070 一致，报告/方案/结果分析三处统一。✅
- **结论**：P2 销项。

### P3（中）S3 声称「沿用 S1 口径」但多加了 StandardScaler —— ✅ 修正到位

- **修正内容**：
  - S3 报告 §2.1 预处理步骤 3 显式列出「StandardScaler（均值/方差仅训练集估计，防泄漏）」；
  - S3 报告 §6.1 口径声明显式说明「与 S1/S2 的纯 CLR 口径不同」；
  - S3 approach §5 步骤 1 由「与 S1 口径一致」改为「**与 S1 同用 L2+CLR，但额外加 StandardScaler**，防泄漏」。
- **与 pkl 一致性**：StandardScaler 为预处理口径差异，pkl 无直接字段；S3 `meta.model` 记录 L2+CLR，`domain_auc` 与 S1 `L2_CLR.AUC` 的差异（P1 已核）即该口径差异的数值体现。✅
- **结论**：P3 销项。

### P4（低）S2 文档「S1 主口径未选定」时序矛盾 —— ✅ 修正到位

- **修正内容**：
  - `result-analysis-S2.md` T3 标题改为「**S1 已选定①归健康，S2 跟随**」，正文「S1 结果分析 §3.3 已选定主口径①（small_adenoma 归健康，pkl `selected_main_caliber='healthy'`）」；
  - `result-analysis-S2.md` §6 待核验项改为「S1 已选定主口径①（归健康），S2 跟随，CRC 无需重跑（T3）」；
  - S2 报告 §2.4 标签口径改为「S1 已选定①归健康，S2 跟随」。
- **与 pkl 一致性**：S1 `adenoma_sensitivity.selected_main_caliber = 'healthy'`，与「①归健康」一致。✅
- **残留检查**：`未选定` 表述仅残留在 approach-S2/handoff-S1/S2/S3 等**规划/交接文档**（历史性前瞻表述）及 review-3.0/gate-3.0/critical-reading/review-notes 等**审查记录**中，均非 P4 目标文档（result-analysis-S2.md、iter-02-sub2-biomarker.tex），不构成未销项。✅
- **结论**：P4 销项。

### P5（低）S3 approach 引用 S1 L2 0.812（A 类验证值）vs 正式 0.7907 —— ✅ 修正到位

- **修正内容**：S3 approach §1.6 R1 表改为「Zeller 0.846 vs 0.812，**A 类验证值，正式值 0.7907**」。
- **与 pkl 一致性**：S1 Zeller `L2_CLR.AUC` = 0.7907，与标注的正式值一致；0.812 明确标注为 A 类验证值。✅
- **结论**：P5 销项。

### P6（低）S3 approach 衰减归因表用 A3 参考值 0.814/0.885/0.644 —— ✅ 修正到位

- **修正内容**：S3 approach §6.2 表下新增注：「本表域内 AUC 为 **A3 参考（1331 特征）**（`domain_auc_reference_A3`：0.814/0.885/0.644）；正式衰减归因以 264 特征口径为准（`domain_auc`：0.7811/0.8588/0.6638，见报告 §6.1），A3 参考仅作对照，避免混用」。
- **与 pkl 一致性**：S3 `domain_auc_reference_A3` = 0.814/0.885/0.644，`domain_auc` = 0.7811/0.8588/0.6638，两口径均与 pkl 一致，且已显式区分。✅
- **结论**：P6 销项。

### P7（低）S3 approach 灵敏度 0.006 vs 报告 0.024 —— ✅ 修正到位

- **修正内容**：S3 approach 多处（§4.5 三分法表、§6.1 策略 A 表、§6.4 阈值漂移、§8 F6）统一标注「0.006（**A 类验证值，正式实现 0.024**）」。
- **与 pkl 一致性**：S3 `threshold_drift.sensitivity` = 0.0244（正式实现），与标注的正式值 0.024 一致；0.006 明确标注为 A 类验证值。✅
- **结论**：P7 销项。

---

## 三、pkl 数字抽核结果（复审抽核）

| 项 | pkl 实际值 | 修正后文档引用 | 一致 |
|:--|:--|:--|:--:|
| S1 L2 AUC（Zeller/metahit/Chatelier） | 0.7907 / 0.8871 / 0.6496 | 报告 0.791/0.887/0.650 | ✅ |
| S3 domain_auc（CRC/IBD/Obesity） | 0.7811 / 0.8588 / 0.6638 | 报告 §6.1 同 | ✅ |
| S3 domain_auc − S1 L2 AUC（Δ） | −0.0096 / −0.0283 / +0.0142 | 报告 §6.1 声明 −0.010/−0.028/+0.014 | ✅ |
| S3 domain_auc_reference_A3 | 0.814 / 0.885 / 0.644 | approach §6.2 注（A3 参考） | ✅ |
| S3 decay_attribution（域内/跨/衰减） | CRC 0.7811/0.5674/−0.2138；IBD 0.8588/0.5882/−0.2706；Obesity 0.6638/0.5253/−0.1384 | 报告 §6.1 表同 | ✅ |
| S3 四策略 mean_auc（A/B/C属/C门/D） | 0.5603 / 0.5572 / 0.4639 / 0.5134 / 0.5603 | 报告 §3 同 | ✅ |
| S3 回退 R1/R2/R3/R4 | 0.5092 / 0.5603 / 0.6068 / 0.5947 | 报告 §5 同 | ✅ |
| S3 R3 各组合 AUC（C1/C2/C3） | 0.5945 / 0.6489 / 0.5771 | 报告 §4.2 同 | ✅ |
| S3 阈值漂移（基线差/阈值/分位/灵敏度） | +0.332 / 0.9205 / 96.0% / 0.0244 | 报告 §6.3 同 | ✅ |
| S3 迁移方向一致/翻转（n_valid，p） | 387 / 369（756，51.2%，p=0.5364） | 报告 §6.2 同 | ✅ |
| S1 selected_main_caliber | 'healthy' | S2 文档「①归健康」 | ✅ |
| S2 clr_delta / filter_threshold / C_lasso / fdr_m / tau | 6.5e-06 / 0.95 / 0.1 / 1331 / 0.5 | 报告同 | ✅ |
| S2 n_stable（CRC/IBD/Obesity） | 4 / 4 / 20 | 报告同 | ✅ |
| S2 Jaccard | 0.0 / 0.0 / 0.0 | 报告同 | ✅ |

**抽核结论**：修正后 S3 报告/方案、S2 报告/结果分析引用的数字与 pkl 实际值**逐项一致**（四舍五入差异可接受）；P1 的 Δ 值、P5 的正式值 0.7907、P6 的 A3 参考值、P7 的正式灵敏度 0.024 均与 pkl 核对一致。

---

## 四、结论：通过

**阶段 3.1 报告修正复审结论：通过。**

- **P1-P7 全部修正到位**：P1/P3（StandardScaler 与域内 AUC 口径差异显式声明 + approach §5 改述）、P2（silhouette 0.070 补回）、P4（S2 文档「S1 已选定①归健康，S2 跟随」）、P5/P6/P7（S3 approach 引用值标注 A 类验证值 vs 正式值）逐项销项。
- **数字与 pkl 一致**：修正后文档引用的全部关键数字（含 P1 的 Δ、P5 正式值、P6 A3 参考、P7 正式灵敏度）与 pkl 实际值逐项核对一致。
- **无新增矛盾**：修正未引入新的跨问数字不一致或口径漂移。

---

## 五、交接收尾

复审结论已落盘（本文件）。**next_action**：主建模裁决 → 阶段 3.2 全部定稿（S1/S2/S3 报告 + 讲解包定稿，进入终稿组装）。

# 门禁 N 审查：S1 2.2 结果分析（review-N-S1-22）

> **审查代理角色**：建模 Preset（modeling preset）| 模型：deepseek-v4-flash:0731
> **审查对象**：`solution/model-notes/result-analysis-S1.md`（2.2 结果分析产出）
> **审查范围**：门禁 N 判定内容之「2.2 结果分析」部分
> **日期**：2026-08-21 | 运行模式：auto | 分支：`experiment/2026sim2B-S1`

---

## 0. 必读清单已读汇报

已完整读取并遵守：
- `E:\MathModel_pj\TRAE-建模.md`（2.2 结果分析规范，§234-250）
- `E:\MathModel_pj\TRAE-规范.md`（A 执行规范 / B 产出格式 / C 项目强制规范相关节）
- `solution/model-notes/result-analysis-S1.md`（审查对象）
- `solution/model-notes/handoff-S1-model-agent.md`（代码→建模回报，含结果摘要/两遍审核结论/待裁定项 §5）
- `solution/model-notes/handoff-S1-report-agent.md`（建模→报告交接）
- `solution/model-notes/approach-S1-confirmed.md`（方案确认书，核验少数类比例/基线/领域下界口径）
- `outputs/data/S1-results.pkl`（用只读脚本 `outputs/scratch/review-N-S1-22-extract.py` 提取关键字段，数字只取 pkl 实际值）

---

## 1. 判定内容逐项结论

### 1.1 结果分析合理性 —— **通过**

**结论**：result-analysis-S1.md 全部关键数字均取自 `S1-results.pkl` 实际值，结论与数据一致，拐角解/未闭合清单完整。

**依据**（逐项抽核，证据路径 = pkl 字段）：
- 三数据集 L2/RF 性能表（§1）：Zeller L2 AUC 0.7907 / RF 0.8454、metahit L2 0.8871 / RF 0.9035、Chatelier L2 0.6496 / RF 0.6602，与 pkl `<ds>.L2_CLR.AUC` / `<ds>.RF_raw.AUC` 完全一致。
- 基线表（§1）：单特征最佳 AUC 0.7581/0.8153/0.6395、Dummy 多数类 ACC 0.6033/0.7727/0.6482，与 pkl `<ds>.baseline` 一致。
- 与基线增益（§2）：L2 +0.0326/+0.0718/+0.0101、RF +0.0873/+0.0882/+0.0207，由 pkl AUC 减基线计算复核一致。
- 结论与数据一致：RF 优于 L2（Zeller/metahit）✓；metahit RF 少数类 F1/Recall 极低（0.3524/0.240）✓；Chatelier 两模型接近基线（0.6496/0.6602 vs 0.6395）✓；LOOCV 与 5 折 CV 差距 0.0135/0.0123/0.0226 <0.025 ✓。
- 拐角解：本问为分类评估问题（非优化），无「优化变量触及约束边界」拐角解，文档正确说明（§5 头注）。未闭合清单 7 项（≥3 项要求），覆盖 full_AUC n≪p 背景、Chatelier 折级不稳定、metahit RF Recall 崩塌、14 离群样本、small_adenoma 口径、无外部验证、C=1.0 未调参，完整合理。

### 1.2 待裁定项裁决合理性 —— **通过**

**结论**：handoff §5 三项待裁定项裁决均合理、有 pkl 事实支撑、且已落地到 result-analysis 与 handoff-report。

**依据**：
- **① 过拟合判定**（result-analysis §3.1）：采用「CV vs LOOCV」口径判定无过拟合（差距 0.0135/0.0123/0.0226 <0.025），显式说明 full_AUC=1.0 是 n≪p（264 特征 vs 110-253 样本）样本内必然现象，保留 overfit_flag=True 留痕。pkl 事实：full_AUC=1.0、overfit_delta=0.1958/0.1252/0.3730、overfit_flag=True。裁决合理。
- **② Chatelier RF F1/Recall 极低**（result-analysis §3.2）：按规格执行（RF 无 class_weight），AUC 0.6602 为阈值无关诚实指标，F1/Recall 低（0.0944/0.0562）是「无 class_weight + 默认阈值」已知局限，不作为性能结论。pkl 事实：RF confusion [[5,84],[6,158]]，少数类（健康）Recall=5/89=0.0562 复核一致。裁决合理。
- **③ small_adenoma 主口径**（result-analysis §3.3）：推荐维持①（归健康），③/④ 入附录作敏感性，②（归病变）排除。pkl 事实：② L2 AUC 0.6112（掉 0.18）显著劣化；③/④ 0.8022/0.8667 略优于①但 Δ<0.05 不显著；① 样本量最大（121）且为题面主口径。裁决合理，`selected_main_caliber='healthy'` 已落盘一致。
- **落地**：三项裁决均写入 handoff-report §3 口径声明（第 3/4/5 条）与 result-analysis 正文，非仅停留在分析层。

### 1.3 handoff-report 完整性 —— **通过**

**结论**：handoff-S1-report-agent.md 含章节映射、关键数字（来源可溯到 pkl）、口径声明、图表清单规格、AI 标注，无占位符。

**依据**：
- 章节映射：§1 表格将建模内容映射到报告骨架 §1-§6，标注需补内容。
- 关键数字：§2 六张表（性能/基线/增益/LOOCV 过拟合/adenoma 四口径/B 类验证），每表标注 pkl 字段来源（如 `<ds>.L2_CLR / RF_raw / baseline`、`adenoma_sensitivity`、`B3_class_weight`、`B4_outlier_removal`、`soft_voting`）。
- 口径声明：§3 六条（数据口径/主指标/过拟合口径/Chatelier RF 局限/small_adenoma 主口径/跨数据集横向比）。
- 图表清单规格：§4 五张正式图，每张含数据源 pkl 字段 + 图名 + 论文位置 + 内容。
- AI 标注：§5 按 ai-usage-report skill 规范标注。
- 无占位符：grep `TODO|TBD|待补充|待填充|待填|\todo|placeholder|占位` 无命中。

### 1.4 数字一致性 —— **通过**

**结论**：result-analysis 与 handoff-report 的关键数字与 pkl 实际值全部一致。

**依据**（pkl 抽核，证据路径 = pkl 字段）：
- 三数据集 AUC/ACC/F1/Recall：全部与 pkl 一致（见 1.1）。
- LOOCV AUC 0.8042/0.8748/0.6270、overfit_delta 0.1958/0.1252/0.3730、CV vs LOOCV 差距 0.0135/0.0123/0.0226：与 pkl `LOOCV.AUC` / `overfit_delta` 一致。
- adenoma 四口径 0.7907/0.6112/0.8022/0.8022（L2）、0.8454/0.6509/0.8667/0.8667（RF）、n=121/121/95/95：与 pkl `adenoma_sensitivity` 一致。
- B3：metahit Recall 0.52→0.64（Δ+0.12）、AUC 不变 0.8871：与 pkl `B3_class_weight` 一致。
- B4：Zeller L2 0.7907→0.8016（Δ+0.0110）、RF 0.8454→0.8949（Δ+0.0494）、n=14：与 pkl `B4_outlier_removal` 一致。
- B2 集成：Zeller 0.8379 vs RF 0.8454（Δ-0.0076）、metahit 0.8955 vs RF 0.9035（Δ-0.0080）、Chatelier 不触发（soft_voting=None）：与 pkl `soft_voting` 一致。
- Chatelier L2 折级 fold AUC 0.520/0.700/0.697/0.622/0.708（§5 项 2）：与 pkl `Chatelier.L2_CLR.cv_folds` 一致。

---

## 2. 问题清单

| # | 问题 | 严重度 | 证据路径 |
|:--|:--|:--|:--|
| 1 | **2.2 实际落地检查（step 6）未显式成对呈现**：result-analysis 未以「输入鲁棒性 / 输出成本」两条必答题形式呈现；输入鲁棒性仅以未闭合清单项 6（无外部验证）间接覆盖，输出成本（临床假阴/假阳成本）未涉及 | 中 | `TRAE-建模.md` §246（step 6）；`result-analysis-S1.md` §5 项 6 |
| 2 | **2.2 跨问不等式核验（step 8，门禁 N 必选项）未显式呈现**：result-analysis 无跨问不等式核验清单；S1 为首个子问题且 `data-integration-*.md` 尚未建立，暂无已合并子问题可互证，但文档未说明该必选项的处置 | 低 | `TRAE-建模.md` §248（step 8）；`result-analysis-S1.md` 全文；`solution/data-final/` 无 data-integration 文件 |
| 3 | **pkl meta `field_semantics` 对 Chatelier 的「正类=1(少数)」标注不准确**：Chatelier 正类（label 1）实为 obesity（164/253=64.8%，多数类），少数类为健康（label 0，89/253=35.2%）；代码对 F1_minority/Recall_minority 计算正确（Recall=5/89=0.0562 复核一致），但 meta 语义描述与数据不符 | 低 | `S1-results.pkl` `meta.field_semantics`；`Chatelier.L2_CLR.confusion_matrix` [[47,42],[47,117]]；`result-analysis-S1.md` §3.2 |

> 注：问题 1/2 属 2.2 规范必选项的呈现性缺口，不影响已产出的数字正确性与结论合理性，建议主建模在门禁 N 裁决时登记为待裁定项或由报告对话在 §6 结论与局限补足；问题 3 为 pkl meta 文档性瑕疵，不影响分析结论。

---

## 3. 审查结论

**通过**。

- 四项判定内容（结果分析合理性 / 待裁定项裁决合理性 / handoff-report 完整性 / 数字一致性）全部通过。
- 关键数字经 `S1-results.pkl` 只读抽核全部一致，无幻觉填数。
- 三项待裁定项裁决合理且已落地。
- 问题清单 3 项均为低/中严重度呈现性或文档性缺口，不构成阻断；建议主建模裁决时登记问题 1/2 为待裁定项（[B级]）或由报告对话补足。

**next_action**: 主建模自检核实 + 门禁 N 裁决，写 `gate-N.md`（引用本结论文件原文）。

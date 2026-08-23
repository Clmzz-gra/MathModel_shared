# 门禁 N 审查：review-N-S2-22.md（2.2 结果分析）

> 审查代理角色：建模 Preset（modeling preset）
> 审查对象：`solution/model-notes/result-analysis-S2.md`（2.2 结果分析产出）
> 关联审查：`handoff-S2-report-agent.md`（建模→报告交接）、`handoff-S2-model-agent.md`（代码→建模回报）
> 数据源：`outputs/data/S2-results.pkl`（generated=2026-08-21T17:46:38）
> 日期：2026-08-21
> 门禁：门禁 N（2.2 结果分析部分）

---

## 0. 必读清单已读汇报

开工前已 Read 以下文件并遵守其中规则：

- [x] `E:\MathModel_pj\TRAE-建模.md`（2.2 结果分析规范：拐角解检测、未闭合清单、实际落地检查、跨问不等式核验）
- [x] `E:\MathModel_pj\TRAE-规范.md`（A 执行规范 / B 产出格式 / C 项目强制规范相关节）
- [x] `solution/model-notes/result-analysis-S2.md`（审查对象）
- [x] `solution/model-notes/handoff-S2-model-agent.md`（代码→建模回报，含结果摘要/待裁定项）
- [x] `solution/model-notes/handoff-S2-report-agent.md`（建模→报告交接，含章节映射/关键数字/口径声明）
- [x] `outputs/data/S2-results.pkl`（用 python 只读提取关键字段，数字只取 pkl 实际值）

---

## 1. 判定内容逐项结论

### 1.1 结果分析合理性

**结论：部分通过（存在数字口径错误，需修正）**

**依据**：result-analysis-S2.md 整体基于 pkl 实际数字，大部分结论与数据一致（稳定标志物数、频率、Fisher/Wilcoxon q、共现边、VIP 秩相关、Jaccard 全 0 等均与 pkl 核对一致）。但存在 3 处与 pkl 不一致的数字错误（见问题清单 P1/P2/P3），其中 P1（IBD Fisher 显著数）为高严重度，直接影响结论表述。

**拐角解/未闭合清单完整性**：U1-U6 共 6 项（≥3 项要求），覆盖 RF 退化、Wilcoxon q>1、Wilcoxon 显著特征不在稳定集、handoff C 敏感性 vs pkl 不一致、Obesity 弱信号、small_adenoma 口径未定，清单完整。注：本问为特征选择（无优化约束变量），「拐角解」概念不直接适用，清单以未闭合项为主，可接受。

### 1.2 待裁定项裁决合理性

**结论：通过（T1/T2/T3 裁决均合理且落地）**

| 待裁定项 | 裁决 | 合理性 | 落地 |
|:--|:--|:--|:--|
| T1 C 选择 | 主口径 C=0.1（4 个高置信标志物全命中已知），C=0.5 作扩展补充 | 合理：C=0.1 的 4 个 CRC 标志物全部为已知 CRC 相关菌，生物验证锚点最强；「Top 10-20」为设计目标非硬约束 | handoff-report §3.1「主口径 C=0.1」已落地 |
| T2 VIP>1.5 独立复现 | 输出 VIP>1.5 特征清单作独立复现证据 | 合理：VIP 为独立方法（PLS-DA），稳定标志物出现在 VIP top、秩相关中等正（0.35~0.54） | handoff-report §4 图表清单含「VIP>1.5 特征清单」已落地 |
| T3 small_adenoma 口径 | S1 主口径未定，S2 按题面口径（归健康）执行，登记 [A级] 待核验 | 合理：S1 2.2 未产出，按 approach R4 执行题面口径，无需重跑 | handoff-report §3.7「S2 按题面口径执行；若 S1 改口径 S2 CRC 需重跑（A级）」已落地 |

### 1.3 handoff-report 完整性

**结论：通过**

**依据**：handoff-S2-report-agent.md 含：
- 章节映射（§1 表：论文章节→内容来源→关键数字/表）
- 关键数字（§2，来源可溯到 pkl，并显式声明「禁止抄 handoff-S2-model-agent 数字」）
- 口径声明（§3，8 项）
- 图表清单规格（§4 表：数据源 pkl 字段+规格+论文位置）
- AI 标注（§5）
- 无占位符（grep `\todo|TBD|TODO|待补充|待填充|待填|占位` 无命中）

### 1.4 数字一致性

**结论：不通过（存在 3 处与 pkl 不一致的错误）**

**依据**：result-analysis 与 handoff-report 的关键数字大部分与 pkl 一致（稳定标志物数 CRC=4/IBD=4/Obesity=20、频率、Fisher/Wilcoxon q、共现边数/OR、VIP>1.5 数 CRC=28/IBD=27/Obesity=23、Spearman 秩相关 0.539/0.515/0.347、Jaccard 全 0、τ 敏感性表均核对一致）。但存在 3 处与 pkl 不一致的错误（P1/P2/P3），其中 P1 为高严重度。

---

## 2. 问题清单

| # | 问题 | 严重度 | 证据路径 |
|:--|:--|:--|:--|
| P1 | **IBD Fisher 显著数误写为 3，实际 4/4**：result-analysis §0/§2.1/§2.2/§7 与 handoff-report §0/§2.1/§2.3 均称「IBD 4 个中 3 个 Fisher 显著」，但 pkl 显示 Eubacterium_ventriosum fisher_fdr=0.0398<0.05，即 4 个稳定标志物全部 Fisher 显著（n_fisher_sig=6 含 2 个稳定集外特征，与 4 稳定显著自洽） | 高 | `pkl per_disease.IBD.two_path_signals` Eubacterium_ventriosum fisher_fdr=0.03979；对照 `result-analysis-S2.md` §2.1 表（Eubacterium_ventriosum Fisher q=0.0398 却未计入显著） |
| P2 | **Obesity 主导信号数误写**：result-analysis §2.1 称「主导信号 19/20 presence，1 个（Bacteroides_ovatus）abundance」，实际 pkl 为 18/20 presence、2 个 abundance（Bacteroides_ovatus 与 Ruminococcus_bromii） | 中 | `pkl per_disease.Obesity.two_path_signals` dominant_signal Counter={presence:18, abundance:2}；Ruminococcus_bromii dominant_signal='abundance' |
| P3 | **RF「0/264 非零」误写**：result-analysis U1/§2.4 称「rf_importance 全部 ~1e-17（0/264 非零）」，实际 pkl 非零数 CRC=82/IBD=128/Obesity=130（值全 ~1e-17，机器精度级）。结论「RF 退化不可用」方向正确，但「0/264 非零」数字错误 | 中 | `pkl per_disease.*.rf_importance` nonzero 计数（CRC=82/IBD=128/Obesity=130，max≈3.3e-17~4.4e-17） |
| P4 | **「VIP top」定义未明确，CRC 3/4 与 VIP>1.5 口径不一致**：result-analysis §2.4 称「稳定标志物出现在 VIP top（CRC 3/4、IBD 4/4）」，但 4 个 CRC 稳定标志物 VIP 均>1.5（C.hathewayi=1.84>1.5），若「VIP top」=VIP>1.5 应为 4/4；「VIP top」的 N 值未定义 | 低 | `pkl per_disease.CRC.vip` C.hathewayi=1.84>1.5；对照 `result-analysis-S2.md` §2.4 |
| P5 | **handoff-report §6「rf_overlap=0」对 Obesity 不精确**：pkl 中 Obesity rf_overlap=0.05（非 0） | 低 | `pkl per_disease.Obesity.topN_consistency.rf_overlap=0.05`；对照 `handoff-S2-report-agent.md` §6 |

---

## 3. 结论

**不通过（需修正后复审）**

- 待裁定项裁决（T1/T2/T3）合理且落地；handoff-report 完整性通过；未闭合清单 U1-U6 完整。
- 但 result-analysis-S2.md 与 handoff-S2-report-agent.md 存在 3 处与 pkl 不一致的数字错误（P1 高、P2/P3 中），其中 P1（IBD Fisher 显著数 3→4）为高严重度，直接影响结论表述并会传播至报告。
- 修正要求：P1/P2/P3 按 pkl 实际值修正（IBD 4/4 Fisher 显著、Obesity 18/20 presence+2 abundance、RF 非零数 82/128/130 或改述为「全 ~1e-17 机器精度级退化」）；P4/P5 建议明确「VIP top」定义并修正 rf_overlap 表述。修正后复审 diff。

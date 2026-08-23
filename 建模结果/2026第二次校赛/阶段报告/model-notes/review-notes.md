# review-notes.md — 阶段 3.0 跨问审查结论草稿

> 阶段：3.0 跨子问题审查 | 日期：2026-08-21 | 运行模式：auto
> 角色：批判阅读审查代理（modeling preset，模型 deepseek-v4-pro:0813）
> 本文件为跨问审查结论**草稿**，供主建模裁决；最终结论见 `review-3.0-跨问.md`。

---

## 1. 审查范围与必读清单

已读：`TRAE-建模.md`（3.0 规范）、`TRAE-规范.md`（A/B/C）、`domain-knowledge.md`、三方案确认书、三结果分析、三报告定稿、三讲解包、三结果 pkl（python 只读提取）。数字只取 pkl 实际值。

---

## 2. 判定内容逐项结论

| # | 判定项 | 结论 | 依据 | 证据路径 |
|:--|:--|:--|:--|:--|
| 1 | 过滤口径统一性 | ✅ 一致 | 三问均「近全零过滤 1331→264，三病并集」 | S1 报告 §2.2 / S2 报告 §2.2 / S3 报告 §2.1；pkl S2 meta filter_threshold=0.95 |
| 2 | CLR 口径统一性 | ✅ 一致 | 三问均 δ=6.5e-06 乘法替换 + 几何均值中心化 | S1/S2/S3 meta clr_delta=6.5e-06 |
| 3 | small_adenoma 口径 | ⚠️ 结果一致、文档时序矛盾 | S1 主口径①归健康；S2/S3 均归健康，但 S2 文档写「S1 未选定」 | S1 result-analysis §3.3（selected_main_caliber='healthy'）vs S2 result-analysis T3 / S2 报告 §2.2 |
| 4 | 标签映射统一性 | ✅ 一致 | 三问患病=1/健康=0 | S1 报告 §2.2 / S2 报告 §2.2 / S3 报告 §2.2 |
| 5 | 数字一致性（跨问引用） | ❌ 不一致 | S3 域内 AUC 与 S1 域内 AUC 不同且未解释；S3 approach 引用 S1 混用 A 类验证值 | S3 报告 §6.1 vs S1 报告 §4.1；pkl S3 domain_auc vs S1 L2_CLR.AUC |
| 6 | 接口/假设冲突 | ❌ 存在冲突 | S3 声称「沿用 S1 口径」但多加了 StandardScaler；S3 报告 silhouette「未量化」vs approach 0.070 | approach-S3 §5 vs approach-S1 §3；S3 报告 §6.1 vs approach-S3 §4.5 |
| 7 | 报告/讲解包一致性 | ⚠️ 基本一致、少量漂移 | 三问报告与讲解包数字基本一致（四舍五入可接受）；S3 silhouette 矛盾、灵敏度 0.006 vs 0.024 | 各报告 vs 各讲解包 |

---

## 3. 问题清单（问题 | 严重度 | 证据路径 | 修正建议）

| # | 问题 | 严重度 | 证据路径 | 修正建议 |
|:--|:--|:--|:--|:--|
| P1 | S3 域内 AUC（0.7811/0.8588/0.6638）与 S1 域内 AUC（0.7907/0.8871/0.6496）不一致，S3 未解释差异 | **高** | S3 报告 §6.1 衰减归因表 vs S1 报告 §4.1 性能表；pkl `S3-results.pkl domain_auc` vs `S1-results.pkl <ds>.L2_CLR.AUC` | 在 S3 报告 §6.1 显式声明「S3 域内 AUC 因增加 StandardScaler 与 S1 略有差异（0.7811 vs 0.7907 等），衰减归因以 S3 264 特征同口径为准」；或统一 S3 域内 AUC 与 S1 口径（去掉 StandardScaler 重算） |
| P2 | S3 报告 silhouette「未单独量化」与 approach §4.5/result-analysis §2.3 的 0.070 矛盾 | 中 | S3 报告 §6.1 vs approach-S3 §4.5 / result-analysis-S3 §2.3 | S3 报告 §6.1 补回「silhouette 系数（数据集标签）=0.070（近 0，批次效应不主导）」定量证据 |
| P3 | S3 声称「沿用 S1 口径」但多加了 StandardScaler（S1/S2 无） | 中 | approach-S3 §5 vs approach-S1 §3 | S3 报告 §2.1 显式声明「在 S1 口径基础上增加 StandardScaler（仅训练集估计）」，并说明对域内 AUC 的影响 |
| P4 | S2 文档「S1 主口径未选定」与 S1 已选定①归健康（时序矛盾） | 低 | S2 result-analysis T3 / S2 报告 §2.2 vs S1 result-analysis §3.3 | 销项 S2 T3 待裁定项：S1 已选定①归健康 = S2 题面口径，无需重跑；更新 S2 报告 §2.2 表述 |
| P5 | S3 approach 引用 S1 L2 0.812（A 类验证值）vs S1 正式 0.7907 | 低 | approach-S3 §1.6 R1 vs S1 报告 §4.1 | approach-S3 §1.6 将 0.812 改为 0.7907（或标注「A 类验证参考值」） |
| P6 | S3 approach 衰减归因表用 A3 参考值 0.814/0.885/0.644 vs 报告 264 口径 | 低 | approach-S3 §6.2 vs S3 报告 §6.1 | approach-S3 §6.2 表标注「A3 1331 参考值，正式以 264 口径为准」 |
| P7 | S3 approach 灵敏度 0.006 vs 报告 0.024（A 类验证 vs 正式实现，4 倍差） | 低 | approach-S3 §4.5/§6.4 vs S3 报告 §6.3 | 报告说明 A 类验证 0.006 → 正式 0.024 的差异来源（或 approach 标注参考值） |

---

## 4. 结论

**不通过**（存在高严重度跨问数字不一致 P1：S3 域内 AUC 与 S1 域内 AUC 不一致且未解释；以及中严重度 P2/P3 的口径矛盾）。

**裁决建议（供主建模）**：
- P1/P3 为同一根因（S3 静默增加 StandardScaler），建议合并处理：S3 报告显式声明 StandardScaler 与域内 AUC 口径差异，或统一口径重算。
- P2 为报告内容段遗漏，直接补回 0.070 即可。
- P4-P7 为低严重度文档漂移，阶段 3.1 逐条销项。

**修正执行主体**：按 TRAE-建模.md 3.1，内部报告正文修正属内容生产（枝干），应由 report Preset 子代理执行，主建模不代写。

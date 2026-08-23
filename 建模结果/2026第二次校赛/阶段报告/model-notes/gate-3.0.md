# 门禁记录：3.0 跨子问题审查

> 日期：2026-08-21 | 运行模式：auto（goal：推进至讲解包产出，裁决倾向全量实验验证）
> 门禁协议：批判阅读审查代理自写结论文件 → 主建模自检核实 → 裁决（自动按推荐）→ 门禁记录

## 审查结论（引用 review 结论文件原文）

审查结论文件：`solution/model-notes/review-3.0-跨问.md`（含 critical-reading.md + review-notes.md 草稿）

**结论：不通过。** 通过项（过滤口径 1331→264 三病并集、CLR δ=6.5e-06、标签映射患病=1/健康=0、small_adenoma 结果均归健康、三问报告/讲解包数字与 pkl 逐项一致）；不通过项 1 高 + 2 中。

## 问题清单（问题 | 严重度 | 证据路径）

| # | 问题 | 严重度 | 证据路径 |
|:--|:--|:--|:--|
| P1 | S3 域内 AUC（0.7811/0.8588/0.6638）与 S1 域内 AUC（0.7907/0.8871/0.6496）不一致，S3 未解释差异 | **高** | `iter-02-sub3-cross-disease.tex` §6.1 vs `iter-02-sub1-disease-prediction.tex` §4.1 |
| P2 | S3 报告 silhouette「未单独量化」与 approach §4.5/result-analysis §2.3 的 0.070 矛盾 | 中 | `iter-02-sub3-cross-disease.tex` §6.1 vs `approach-S3-confirmed.md` §4.5 |
| P3 | S3 声称「沿用 S1 口径」但多加了 StandardScaler（S1/S2 无） | 中 | `approach-S3-confirmed.md` §5 vs `approach-S1-confirmed.md` §3 |
| P4 | S2 文档「S1 主口径未选定」与 S1 已选定①归健康（时序矛盾） | 低 | `result-analysis-S2.md` T3 vs `result-analysis-S1.md` §3.3 |
| P5 | S3 approach 引用 S1 L2 0.812（A 类验证值）vs S1 正式 0.7907 | 低 | `approach-S3-confirmed.md` §1.6 R1 |
| P6 | S3 approach 衰减归因表用 A3 参考值 0.814/0.885/0.644 vs 报告 264 口径 | 低 | `approach-S3-confirmed.md` §6.2 |
| P7 | S3 approach 灵敏度 0.006 vs 报告 0.024（A 类验证 vs 正式实现） | 低 | `approach-S3-confirmed.md` §4.5/§6.4 |

## 主建模自检核实（裁决前过 1-3 个关键点，非审查）

1. **P1/P3 根因确认**：S3 approach §5 显式写「预处理（与 S1 口径一致）：近全零过滤 → CLR → **StandardScaler**（均值/方差仅训练集估计）」，而 S1/S2 无 StandardScaler。S3 静默增加 StandardScaler 导致域内 AUC 与 S1 不一致，且报告未声明差异。✓ 根因确认
2. **P1 影响面**：S3 域内 AUC 用于内部衰减归因（域内 vs 跨疾病，两者均含 StandardScaler），内部比较有效；问题仅在「声称沿用 S1 口径」不准确 + 未声明与 S1 差异。✓
3. **P2 确认**：S3 approach §4.5 给 silhouette=0.070，报告 §6.1 写「未单独量化」，矛盾。✓

## 裁决（自动模式按推荐）

**门禁 3.0 不通过，进入阶段 3.1 报告修正。** 修正执行主体为 report Preset 子代理（内容生产禁则，主建模不代写）。

**修正方案（按审查推荐）**：
- **P1/P3 合并处理**：S3 报告显式声明「预处理含 StandardScaler（均值/方差仅训练集估计），与 S1/S2 的纯 CLR 口径不同，故域内 AUC 与 S1 有差异（Δ −0.010/−0.028/+0.014）」；S3 approach §5 的「与 S1 口径一致」改为「与 S1 同用 L2+CLR，但额外加 StandardScaler」。
- **P2**：S3 报告补回 silhouette=0.070（或显式声明未量化原因）。
- **P4-P7**：阶段 3.1 逐条销项（S2 文档时序表述、S3 approach 引用值口径标注）。

## 采用方案与下一步

门禁 3.0 不通过 → **阶段 3.1 报告修正**（report Preset 子代理执行 P1-P7 修正）→ 修正后复审 → 阶段 3.2 全部定稿。

## 变更记录

| 时间 | 动作 |
|---|---|
| 2026-08-21 | 批判阅读审查代理产 review-3.0-跨问.md（不通过，1 高 + 2 中 + 4 低） |
| 2026-08-21 | 主建模自检核实 P1/P3 根因（StandardScaler 差异）确认 |
| 2026-08-21 | 本门禁记录创建，裁决进入阶段 3.1 报告修正 |

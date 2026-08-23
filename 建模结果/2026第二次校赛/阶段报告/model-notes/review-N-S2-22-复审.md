# 门禁 N 复审：review-N-S2-22-复审.md（2.2 结果分析修正 diff）

> 审查代理角色：建模 Preset（modeling preset）
> 复审对象：`solution/model-notes/result-analysis-S2.md` + `solution/model-notes/handoff-S2-report-agent.md`（修正后，commit 9119f82）
> 原审查：`review-N-S2-22.md`（问题清单 P1-P5）
> 数据源：`outputs/data/S2-results.pkl`（generated=2026-08-21T17:46:38）
> 日期：2026-08-21
> 门禁：门禁 N（2.2 结果分析部分）复审 diff

---

## 0. 必读清单已读汇报

开工前已 Read 以下文件并遵守其中规则：

- [x] `solution/model-notes/review-N-S2-22.md`（原门禁 N 审查结论，含问题清单 P1-P5）
- [x] `solution/model-notes/result-analysis-S2.md`（修正后对象）
- [x] `solution/model-notes/handoff-S2-report-agent.md`（修正后对象）
- [x] `outputs/data/S2-results.pkl`（用 python 只读提取关键字段，数字只取 pkl 实际值）
- [x] `git show 9119f82`（修正 diff，核对实际改动范围）

---

## 1. P1-P5 逐项复审结论

### P1 — IBD Fisher 显著数 3→4（高）

**修正到位：✅ 通过**

- 修正 diff 将 result-analysis §0/T1/§2.1/§2.2/§7 与 handoff-report §0/§2.3 的「IBD 3 个 Fisher 显著」全部改为「4/4 Fisher 显著」。
- **pkl 抽核**：`per_disease.IBD.two_path_signals` 4 个稳定标志物 fisher_fdr 分别为 0.00675 / 0.00675 / 0.00675 / **0.03979**（Eubacterium_ventriosum），全部 <0.05 → 4/4 显著；`n_fisher_sig=6`（含 2 个稳定集外特征）与文档「n_fisher_sig=6，含 2 个稳定集外特征」一致。
- 修正后数字与 pkl 完全一致。

### P2 — Obesity 主导信号 19/20+1 → 18/20 presence + 2 abundance（中）

**修正到位：✅ 通过**

- 修正 diff 将 result-analysis §2.1「主导信号 19/20 presence，1 个（Bacteroides_ovatus）abundance」改为「18/20 presence，2 个（Bacteroides_ovatus、Ruminococcus_bromii）abundance」。
- **pkl 抽核**：`per_disease.Obesity.two_path_signals` dominant_signal Counter = `{presence:18, abundance:2}`；abundance 特征为 Bacteroides_ovatus 与 Ruminococcus_bromii。
- 修正后数字与 pkl 完全一致。

### P3 — RF 非零 0 → CRC 82/IBD 128/Obesity 130（中）

**修正到位：✅ 通过**

- 修正 diff 将 result-analysis §2.4 与 U1 的「0/264 非零」改为「非零数 CRC 82/IBD 128/Obesity 130，全 ~1e-17 机器精度级退化」。
- **pkl 抽核**：`per_disease.*.rf_importance` 非零计数 CRC=82 / IBD=128 / Obesity=130，max 分别为 3.33e-17 / 4.44e-17 / 4.44e-17（机器精度级）。
- 修正后数字与 pkl 完全一致；「RF 退化不可用」结论方向不变。

### P4 — 统一 VIP>1.5 口径（低）

**修正到位：✅ 通过**

- 修正 diff 将 result-analysis T2/§2.4 中所有「VIP top」表述统一为「VIP>1.5 清单」，并补入 Clostridium_hathewayi(1.84) 使 CRC 为 4/4。
- **pkl 抽核**：`per_disease.CRC.vip` 4 个稳定标志物 VIP 均>1.5（P.stomatis 3.37 / F.nucleatum 2.97 / P.somerae 2.11 / **C.hathewayi 1.84**）→ 4/4；IBD 4 个稳定标志物 VIP 均>1.5 → 4/4；VIP>1.5 计数 CRC=28 / IBD=27 / Obesity=23。
- 修正后口径统一且与 pkl 一致。

### P5 — rf_overlap 表述（Obesity=0.05）（低）

**修正到位：✅ 通过**

- 修正 diff 将 handoff-report §6「rf_overlap=0」改为「rf_overlap≈0：CRC/IBD=0、Obesity=0.05」。
- **pkl 抽核**：`per_disease.*.topN_consistency.rf_overlap` CRC=0.0 / IBD=0.0 / **Obesity=0.05**。
- 修正后表述与 pkl 完全一致。

---

## 2. 新增残留问题（复审发现，非 P1-P5 清单内）

### R1 — result-analysis §2.2「CRC 4 个稳定标志物全部显著」与 pkl 不一致（中）

- **位置**：`result-analysis-S2.md` §2.2「两路信号显著性」首行：`CRC 4 个稳定标志物全部显著（n_fisher_sig=4）`。
- **pkl 抽核**：`per_disease.CRC.two_path_signals` 中 Clostridium_hathewayi fisher_fdr=**0.599>0.05**（不显著），即 4 个稳定 CRC 标志物中**仅 3 个** Fisher 显著。`n_fisher_sig=4` 是全 264 特征口径的显著数，**不等于**稳定集内显著数。
- **内部矛盾**：同文档 §2.1 明确「前 3 个 Fisher 显著…Clostridium_hathewayi…Fisher 不显著」、§7 明确「3 个 Fisher 显著」，与 §2.2「全部显著」自相矛盾。
- **传播风险**：若报告据此写「CRC 4 个全部 Fisher 显著」将出错（实际 3/4）。
- **修正建议**：改为「CRC 3 个稳定标志物显著（n_fisher_sig=4，含 1 个稳定集外特征）」或「CRC 4 个中 3 个 Fisher 显著」。

---

## 3. 结论

**不通过（需二次修正 R1）**

- **P1-P5 全部修正到位**，且修正后数字与 pkl 完全一致（逐项抽核通过）。
- 但复审发现**新增残留问题 R1**：result-analysis §2.2「CRC 4 个稳定标志物全部显著」与 pkl 不一致（C.hathewayi fisher_fdr=0.599>0.05，实际 3/4 显著），且与同文档 §2.1/§7 自相矛盾，属「修正后数字与 pkl 一致」门禁范围内的残留错误，会传播至报告。
- 修正要求：仅需将 §2.2 的 CRC 表述改为「3 个稳定标志物显著（n_fisher_sig=4，含 1 个稳定集外特征）」。修正后复审。

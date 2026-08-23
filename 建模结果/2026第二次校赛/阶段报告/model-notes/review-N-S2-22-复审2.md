# 门禁 N 复审2：review-N-S2-22-复审2.md（R1 修正确认）

> 审查代理角色：建模 Preset（modeling preset，自动模式）
> 复审对象：`solution/model-notes/result-analysis-S2.md`（R1 修正后）
> 上一轮复审：`review-N-S2-22-复审.md`（结论：不通过，需二次修正 R1）
> 数据源：`outputs/data/S2-results.pkl`（generated=2026-08-21T17:46:38）
> 日期：2026-08-21
> 门禁：门禁 N（2.2 结果分析部分）R1 二次复审

---

## 0. 必读清单已读汇报

开工前已 Read 以下文件并遵守其中规则：

- [x] `solution/model-notes/review-N-S2-22-复审.md`（上一轮复审结论，含 R1）
- [x] `solution/model-notes/result-analysis-S2.md`（R1 修正后对象）
- [x] `outputs/data/S2-results.pkl`（用 python 只读提取关键字段，数字只取 pkl 实际值）

---

## 1. R1 复审结论

### R1 — result-analysis §2.2「CRC 4 个稳定标志物全部显著」与 pkl 不一致（中）

**修正到位：✅ 通过**

- **修正后表述**（`result-analysis-S2.md` §2.2 首行）：
  > CRC 3 个稳定标志物显著（n_fisher_sig=4，含 1 个稳定集外特征）
- **pkl 抽核**（`per_disease.CRC.two_path_signals`，4 个稳定标志物 fisher_fdr）：
  - Peptostreptococcus_stomatis：fisher_fdr=1.24e-05 → **显著**
  - Fusobacterium_nucleatum：fisher_fdr=3.94e-05 → **显著**
  - Porphyromonas_somerae：fisher_fdr=0.0172 → **显著**
  - **Clostridium_hathewayi：fisher_fdr=0.5988 > 0.05 → 不显著**
  - → 4 个稳定标志物中**仅 3 个** Fisher 显著（3/4）。
- **n_fisher_sig 口径**：`per_disease.CRC.n_fisher_sig=4` 为全 264 特征口径的显著数，其中 1 个显著特征不在稳定集内（稳定集内仅 3 个显著）——与修正后表述「n_fisher_sig=4，含 1 个稳定集外特征」完全一致。
- **内部一致性**：修正后 §2.1（C.hathewayi Fisher q=0.599 不显著，低置信边缘）、§2.2（3 个稳定标志物显著）、§7（3 个 Fisher 显著）三处口径统一，不再自相矛盾。
- **修正后数字与 pkl 完全一致。**

---

## 2. 结论

**通过 ✅**

- R1 已修正到位：result-analysis §2.2 由「CRC 4 个稳定标志物全部显著」改为「CRC 3 个稳定标志物显著（n_fisher_sig=4，含 1 个稳定集外特征）」。
- pkl 抽核确认：C.hathewayi fisher_fdr=0.5988>0.05 不显著，实际 3/4 显著；n_fisher_sig=4 含 1 个稳定集外特征，与修正后表述完全一致。
- 修正后 §2.1/§2.2/§7 三处 CRC Fisher 显著口径统一，无内部矛盾，可放行至报告交接。

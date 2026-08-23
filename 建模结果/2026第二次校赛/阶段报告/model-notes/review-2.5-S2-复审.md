# 门禁 A·B 复审：review-2.5-S2-复审（2.5 讲解包修正 diff 复审）

> 审查代理：report preset 独立质量检查实例（M3，复审轮）
> 复审对象：`solution/handoff/02-问题二分析与思路.md` + `solution/handoff/M-05-方法讲解-VIP佐证.md`（2.5 讲解包修正后）
> 对照：`solution/model-notes/review-2.5-S2.md`（原门禁 A·B 审查结论，问题清单 P1-P4）、`outputs/data/S2-results.pkl`（只读提取）
> 日期：2026-08-21

---

## 一、必读清单已读汇报

- [x] `solution/model-notes/review-2.5-S2.md`（原门禁 A·B 审查结论，含问题清单 P1-P4）
- [x] `solution/handoff/02-问题二分析与思路.md`（修正后对象）
- [x] `solution/handoff/M-05-方法讲解-VIP佐证.md`（修正后对象）
- [x] `outputs/data/S2-results.pkl`（python 只读提取关键字段，数字取 pkl 实际值）

---

## 二、P1-P4 逐项复审结论

### P1：02 §4.1 标题「全部命中已知」→「3/4 命中已知」

**修正到位 ✓，与 pkl 一致 ✓**

- 修正后 02 line 82 标题为「**CRC（4 个稳定标志物，3 个 Fisher 显著，3/4 命中已知）**」，已由「全部命中已知」改为「3/4 命中已知」。
- 一致性核对：02 line 21「CRC 3/4 命中已知标志物」、line 115 禁写句「仅 3/4 命中」均与标题一致，无残留「全部命中已知」过度承诺。
- pkl 抽核：`per_disease.CRC.biomarker_table[].known_biomarker` = Peptostreptococcus_stomatis True、Fusobacterium_nucleatum True、Porphyromonas_somerae True、**Clostridium_hathewayi False** → 3/4 命中已知，与修正后表述一致。

### P2：「稳定标志物全部在 VIP>1.5 清单」→「CRC/IBD 4/4、Obesity 仅 2/20」

**修正到位 ✓，与 pkl 一致 ✓**

- 02 line 103 已改为「CRC/IBD 的稳定标志物全部出现在 VIP>1.5 清单（4/4），Obesity 仅 2/20——VIP 佐证对 CRC/IBD 成立」。
- M-05 line 61 已改为「CRC/IBD 的稳定标志物全部出现在 VIP>1.5 清单（4/4，Obesity 仅 2/20）」；line 67 结论句 1 已改为「CRC/IBD 的全部稳定标志物均出现在 VIP>1.5 清单中（Obesity 仅 2/20）」。M-05 line 55 原已正确限定 CRC/IBD 4/4。
- pkl 抽核：`per_disease.<D>.vip` 统计 >1.5 与稳定集交集 = CRC 4/4、IBD 4/4、**Obesity 2/20**（Ruminococcus_flavefaciens VIP=1.733、Ruminococcus_bromii VIP=1.824，其余 18 个 VIP<1.5）；`topN_consistency.vip_overlap` = 0.2/0.2/0.1。与修正后表述一致。

### P3：τ 敏感性「4~6 个」→「2~6 个」

**修正到位 ✓，与 pkl 一致 ✓**

- 02 line 138 已改为「本问有 τ 敏感性表（CRC/IBD 在 τ=0.4~0.7 仅 2~6 个）」。
- pkl 抽核：`meta.tau_grid` = [0.4, 0.5, 0.6, 0.7]；`meta.tau_counts` = CRC [6,4,3,2]、IBD [6,4,2,2] → 范围 2~6，与修正后表述一致。

### P4：C=0.5 扩展清单标注「C 敏感性快查值」

**修正到位 ✓**

- 02 line 123 已改为「C=0.5 的 19/17 扩展清单（**C 敏感性快查值**）作补充（附录/讨论），不改变主口径」，已显式标注「C 敏感性快查值」，消除与文件头「数字口径：全部取自 S2-results.pkl」的溯源口径不一致。

---

## 三、pkl 数字抽核结果

| 抽核项 | 讲解包（修正后） | pkl 实际值 | 一致 |
|:--|:--|:--|:--|
| CRC 命中已知数 | 3/4 | `biomarker_table[].known_biomarker` = 3 True / 1 False（Clostridium_hathewayi False） | ✓ |
| VIP>1.5 稳定集交集 | CRC 4/4、IBD 4/4、Obesity 2/20 | `vip`>1.5 交集 = 4/4/2（Obesity 2/20） | ✓ |
| vip_overlap | 0.2/0.2/0.1 | `topN_consistency.vip_overlap` = 0.2/0.2/0.1 | ✓ |
| τ 敏感性范围 | 2~6 | `tau_counts` CRC [6,4,3,2]、IBD [6,4,2,2] | ✓ |
| τ 网格 | 0.4~0.7 | `tau_grid` = [0.4,0.5,0.6,0.7] | ✓ |
| 参数 | τ=0.5, B=100/50, C=0.1, VIP>1.5, m=1331 | `meta` 全部一致 | ✓ |

---

## 四、通过/不通过

**门禁 A·B（2.5 讲解包部分）复审判定：通过。**

- P1-P4 全部修正到位，无残留过度承诺/内部矛盾表述。
- 修正后数字与 `S2-results.pkl` 实际值全部一致（含 P1 命中已知 3/4、P2 Obesity 2/20、P3 τ 范围 2~6、P4 C 敏感性快查值标注）。
- 讲解包可交付队友写作。

---

## 附：复审用只读提取脚本（outputs/scratch/）

- 只读提取 `S2-results.pkl` 的 `per_disease.<D>.biomarker_table[].known_biomarker`、`vip`、`topN_consistency.vip_overlap`、`meta.tau_grid/tau_counts` 等关键字段，未写任何 pkl。

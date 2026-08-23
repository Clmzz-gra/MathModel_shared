# 门禁 A·B 审查：review-2.5-S2（2.5 讲解包独立质量检查）

> 审查代理：report preset 独立质量检查实例（M3）
> 审查对象：`solution/handoff/02-问题二分析与思路.md` + `M-01~M-05-方法讲解-*.md`（2.5 讲解包）
> 对照：`iter-02-sub2-biomarker.tex`（报告定稿）、`writing-material-sub2.tex`（写作素材）、`outputs/data/S2-results.pkl` / `S2-preprocessed.pkl`（只读提取）、`knowledge/expression-library.md`（表述库）
> 日期：2026-08-21

---

## 一、必读清单已读汇报

- [x] `E:\MathModel_pj\TRAE-报告.md`（2.5 讲解包规范：独立质量检查 M3、数字抽核 M4）
- [x] `E:\MathModel_pj\TRAE-规范.md` B 节（产出格式）
- [x] `solution/handoff/02-问题二分析与思路.md`（审查对象）
- [x] `solution/handoff/M-01~M-05-方法讲解-*.md`（审查对象）
- [x] `solution/internal-reports/iter-02-sub2-biomarker.tex`（报告定稿，对照技术准确性）
- [x] `solution/internal-reports/writing-material-sub2.tex`（写作素材，对照数字一致性）
- [x] `outputs/data/S2-results.pkl` + `S2-preprocessed.pkl`（python 只读提取关键字段）
- [x] `knowledge/expression-library.md`（表述库，核对引用 ID）

---

## 二、判定内容逐项结论

### 1. 技术准确性（讲解包数字与报告 §2-§4 及 pkl 一致，抽核 ≥3 个关键数字）

**结论：基本一致，但存在 2 处中高严重度过度承诺/内部矛盾（P1、P2），需修复。**

抽核关键数字（全部与 pkl 实际值一致）：

| 抽核项 | 讲解包 | pkl 实际值 | 一致 |
|:--|:--|:--|:--|
| CRC 稳定标志物全量频率 | 0.99/0.94/0.62/0.52 | `per_disease.CRC.stable_features[].frequency` = 0.99/0.94/0.62/0.52 | ✓ |
| CRC Fisher FDR | 1.24e-05/3.94e-05/0.0172/0.599 | `per_disease.CRC.biomarker_table[].fisher_fdr` = 1.24e-05/3.94e-05/0.0172/0.599 | ✓ |
| IBD 稳定标志物频率 | 0.81/0.75/0.55/0.53 | `per_disease.IBD.stable_features[].frequency` = 0.81/0.75/0.55/0.53 | ✓ |
| Obesity Top3 频率 | 0.89/0.84/0.72 | `per_disease.Obesity.stable_features[].frequency` = 0.89/0.84/0.72 | ✓ |
| 三病稳定特征数 | 4/4/20 | `per_disease.<D>.n_stable` = 4/4/20 | ✓ |
| Fisher/Wilcoxon 显著数 | 4/1, 6/1, 0/0 | `per_disease.<D>.n_fisher_sig/n_wilcoxon_sig` = 4/1, 6/1, 0/0 | ✓ |
| 共现边数 | CRC 4 / IBD 6 / Obesity 24 | `cooccurrence.cooccurrence_edges` 长度 = 4/6/24 | ✓ |
| CRC 最强共现对 | Spearman 0.762、OR=12.86 | `cooccurrence_edges[0]` spearman=0.7618、odds_ratio=12.86 | ✓ |
| 三病 Jaccard | 全 0 | `cross_disease.jaccard_matrix` = 0.0/0.0/0.0 | ✓ |
| VIP>1.5 特征数 | 28/27/23 | `per_disease.<D>.vip` 统计 >1.5 = 28/27/23 | ✓ |
| VIP 秩相关 | 0.539/0.515/0.347 | `topN_consistency.spearman_rank_vip` = 0.539/0.515/0.347 | ✓ |
| 样本数 | 121/110/253 | `S2-preprocessed.pkl per_disease.<D>.n_samples` = 121/110/253 | ✓ |
| 特征数 | 1331→264 | `S2-preprocessed.pkl meta.n_features_before/after` = 1331/264 | ✓ |
| 参数 | τ=0.5, B=100/50, C=0.1, δ=6.5e-06, m=1331, VIP>1.5 | `S2-results.pkl meta` 全部一致 | ✓ |

**发现的问题（详见问题清单）**：
- **P1（中高）**：02 §4.1 标题「CRC（…全部命中已知）」与同表（Clostridium_hathewayi 已知列「—」）、禁写句（line 115「仅 3/4 命中」）、line 21「CRC 3/4 命中已知标志物」矛盾。pkl `biomarker_table[].known` 中 Clostridium_hathewayi 未命中，实际 3/4。
- **P2（中高）**：「稳定标志物全部出现在 VIP>1.5 清单」对 Obesity 不成立（pkl 显示 Obesity 仅 2/20 在 VIP>1.5 清单，`vip_overlap=0.1`）。出现在 02 §4.1 line 103、M-05 §4 line 61、M-05 §4 line 67 结论句 1。

### 2. 队友可读性（受众=未参与建模的队友）

**结论：通过。**

- 术语首次出现给自然语言解释：✓ CLR、Lasso、bootstrap、Fisher、Wilcoxon、BH-FDR、VIP、Spearman、OR、Jaccard 等均有自然语言解释（如「像选靠谱员工」「像问这个店有没有卖某商品」）。
- 迷你算例体现「数字从哪来」：✓ 每个 M-xx 方法文件均有迷你算例（M-01 bootstrap 频率、M-02 CLR、M-03 Fisher、M-04 共现、M-05 VIP），02 主体节也有 bootstrap 频率算例。
- 含可写结论句示例 + 禁写句标注：✓ 02 §4.2 可写结论句、§4.3 禁写句；各 M-xx §4 均有可写/禁写句。
- 阅读顺序指引：✓ 02 文件头有阅读顺序，附阅读地图（附节）。

### 3. 表述库指引（引用的表述库 ID 真实存在）

**结论：通过。**

- 讲解包用「节名 → 条目名」定位（无数字 ID），写作素材用行号（L111-112 / L141-142 / L196-197）。
- 核对 `knowledge/expression-library.md`：
  - L111-112 = 一、问题分析段·引入问题·**必要性论证式** ✓
  - L141-142 = 二、模型引入段·过渡到模型·**方法比较式** ✓
  - L196-197 = 三、结果与分析段·**量化结论+规律解释式** ✓
- 讲解包各 M-xx 引用的「节名 → 条目名」全部真实存在：必要性论证式、方法比较式、直接汇报最优解式、检验验证式、结果+历史基线对比式、算法流程式、数据处理说明、因果论证式、量化结论+规律解释式、缺点写法 ✓

### 4. 数字一致性（讲解包与写作素材保持同一组数字）

**结论：通过。**

讲解包与 `writing-material-sub2.tex` 关键数字一致：CRC 频率 0.99/0.94/0.62/0.52、CRC CV 0.96/0.72/0.39/0.30、IBD 频率 0.81/0.75/0.55/0.53、Obesity Top3 0.89/0.84/0.72、n_stable 4/4/20、n_fisher_sig/n_wilcoxon_sig 4/1, 6/1, 0/0、共现边数 4/6/24、最强共现对 0.76/8.8e-07/12.9、Jaccard 0/0/0、样本数 121/110/253、特征数 1331/264。全部一致。

---

## 三、问题清单

| 编号 | 问题 | 严重度 | 证据路径 |
|:--|:--|:--|:--|
| P1 | 02 §4.1 标题「CRC（4 个稳定标志物，3 个 Fisher 显著，**全部命中已知**）」中「全部命中已知」与同表（Clostridium_hathewayi 已知列「—」）、禁写句（line 115「仅 3/4 命中」）、line 21「CRC 3/4 命中已知标志物」矛盾。应为「3/4 命中已知」。 | 中高 | `02-问题二分析与思路.md` line 82 vs line 21/115；`S2-results.pkl per_disease.CRC.biomarker_table[].known`（Clostridium_hathewayi 未命中） |
| P2 | 「稳定标志物全部出现在 VIP>1.5 清单——VIP 佐证成立」为过度承诺，对 Obesity 不成立（pkl 显示 Obesity 仅 2/20 在 VIP>1.5 清单：Ruminococcus_flavefaciens 1.733、Ruminococcus_bromii 1.824，其余 18 个 VIP<1.5）。正确表述应为「CRC/IBD 的稳定标志物全部出现在 VIP>1.5 清单（4/4），Obesity 仅 2/20」。出现在 02 §4.1 line 103、M-05 §4 line 61、M-05 §4 line 67 结论句 1。 | 中高 | `02-问题二分析与思路.md` line 103；`M-05-方法讲解-VIP佐证.md` line 61/67；`S2-results.pkl per_disease.Obesity.vip` + `topN_consistency.vip_overlap=0.1`（对照 M-05 §3.3 line 55 已正确限定 CRC/IBD 4/4） |
| P3 | 02 §5 表述库指引「本问有 τ 敏感性表（CRC/IBD 在 τ=0.4~0.7 仅 4~6 个）」不准确——pkl `meta.tau_counts` 显示 CRC [6,4,3,2]、IBD [6,4,2,2]，范围 2~6 而非 4~6。 | 低 | `02-问题二分析与思路.md` line 138；`S2-results.pkl meta.tau_counts` |
| P4 | 02 §4.4「C=0.5 的 19/17 扩展清单」为 handoff 的 C 敏感性快查值（非 pkl 正式值），但 02 文件头声明「数字口径：全部取自 S2-results.pkl」，存在溯源口径不一致。建议标注「C 敏感性快查值」。 | 低 | `02-问题二分析与思路.md` line 123 + line 5；`handoff-S2-model-agent.md` line 101、`result-analysis-S2.md` line 24（均标注「非 pkl 正式值」） |

---

## 四、通过/不通过

**门禁 A·B（2.5 讲解包部分）判定：不通过。**

- 技术准确性（判定项 1）存在 2 处中高严重度问题（P1「全部命中已知」内部矛盾、P2「全部稳定标志物在 VIP>1.5 清单」对 Obesity 过度承诺），均属会误导队友写作的过度承诺/事实错误，可能传播至论文。
- 队友可读性（判定项 2）、表述库指引（判定项 3）、数字一致性（判定项 4）均通过。
- 建议：修复 P1、P2（P3、P4 一并修正）后复审；复审通过后讲解包方可交付队友写作。

---

## 附：审查用只读提取脚本（outputs/scratch/）

- `review-2.5-extract.py` / `extract2.py` / `extract3.py` / `extract4.py`：只读提取 S2-results.pkl / S2-preprocessed.pkl 关键字段，未写任何 pkl。

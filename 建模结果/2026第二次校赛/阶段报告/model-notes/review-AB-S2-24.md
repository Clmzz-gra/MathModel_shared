# 门禁 A·B 审查 — S2（2.4 报告定稿部分）

> 审查对象：`solution/internal-reports/iter-02-sub2-biomarker.tex`（STATUS: done）
> 审查代理：门禁 A·B 审查代理（report preset，自动模式派遣）
> 审查日期：2026-08-21
> 审查范围：门禁 A·B 判定内容之「2.4 报告定稿」部分（报告定稿 / 数字一致性 / 可读性 / 图表占位符）
> 角色+模型：审查代理（report preset，flash 档）

---

## 一、必读清单已读汇报

开工前已 Read 以下文件并遵守其中规则：

- `E:\MathModel_pj\TRAE-报告.md` — 已读。重点：阶段 2.4「报告定稿（门禁 A·B 第一阶段）」节（STATUS: done、素材独立文件全字段填满）；「报告取数规则」（正文写人话、溯源进素材节、数字只取 pkl）；「报告自足性规则」（符号表、首次出现处一句话定义、叙事脚手架）；「L4 读者代理试读」（试读代理自写 reader-audit、判定权归试读代理）；「图表占位符+规格」（数据源 pkl+字段+图名+论文位置）。
- `E:\MathModel_pj\TRAE-规范.md` — 已读。重点：B 节产出格式（内部报告 .tex、中文撰写、review-notes 用 Markdown 中文撰写）；A 节执行规范（只读提取、工作目录约定）。
- `solution/internal-reports/iter-02-sub2-biomarker.tex` — 审查对象（359 行，STATUS: done）。
- `solution/internal-reports/writing-material-sub2.tex` — 写作素材独立文件（120 行，VERSION: 2026-08-21-2.4修订）。
- `solution/internal-reports/reader-audit-sub2.md` — L4 试读结论（二轮通过）。
- `solution/model-notes/handoff-S2-report-agent.md` — 建模→报告交接。
- `outputs/data/S2-results.pkl`、`outputs/data/S2-preprocessed.pkl` — 用 python 只读提取关键字段，数字只取 pkl 实际值。

---

## 二、判定内容逐项结论

### 1. 报告定稿 — 通过

| 子项 | 结论 | 依据 | 证据路径 |
|---|---|---|---|
| STATUS=done | 通过 | 报告首行 `% STATUS: done` | `iter-02-sub2-biomarker.tex` L1 |
| 素材独立文件全填 | 通过 | 章节映射/故事线/关键数字表/图表清单/口径与局限/AI 标注六节全部填满，无空字段 | `writing-material-sub2.tex` L16-118 |
| 数字可溯（pkl 字段路径） | 通过 | 关键数字表逐行登记 `<pkl>.<键路径>`（如 `S2-results.pkl per_disease.CRC.stable_features[].frequency`） | `writing-material-sub2.tex` L52-75 |
| 无 TODO | 通过 | 报告 grep `\todo|TBD|TODO|待补充|待填充|待填` 无命中；素材仅 L12 为 `\todo` 宏定义（非占位使用） | grep 结果 |
| 读者代理试读通过 | 通过 | reader-audit 二轮判定「通过」，一轮 8 处未定义符号/7 处未解释代号/4 处 pkl 侵入全部清零 | `reader-audit-sub2.md` L115-119 |

### 2. 数字一致性 — 通过（抽核 ≥3 个关键数字，全部与 pkl 一致）

抽核关键数字（重跑只读提取对照 pkl，结果见下节「pkl 数字抽核结果」）：

- **CRC 稳定特征频率与 CV 折内频率**：报告 0.99/0.94/0.62/0.52 与 0.96/0.72/0.39/0.30 → pkl `stable_features[].frequency/cv_frequency` 一致（0.716→0.72、0.392→0.39、0.296→0.30 四舍五入正确）。
- **CRC Fisher FDR**：报告 1.24e-5/3.94e-5/1.72e-2/5.99e-1 → pkl `biomarker_table[].fisher_fdr` 一致。
- **三病样本数**：报告 CRC 121(48/73)、IBD 110(25/85)、Obesity 253(164/89) → pkl `per_disease.<D>.n_samples/n_pos/n_neg` 一致。
- **特征数**：1331→264 → pkl `meta.n_features_before/after` 一致。
- **共现最强对**：报告 Spearman 0.76 / Fisher p≈8.8e-7 / OR 12.9 → pkl `cooccurrence_edges[0]`（0.7618/8.82e-7/12.86）一致。
- **主导信号**：Obesity 18 presence + 2 abundance（Bacteroides_ovatus、Ruminococcus_bromii）→ pkl `two_path_signals[].dominant_signal` 一致（已核 2 个 abundance 特征名）。
- **已知标志物**：CRC 3/4、IBD 1/4 → pkl `biomarker_table[].known_biomarker` 一致（CRC Clostridium_hathewayi=False，IBD 仅 Bifidobacterium_bifidum=True）。
- **Jaccard 重叠**：三病两两全 0 → pkl `cross_disease.jaccard_matrix` 一致。

### 3. 可读性 — 通过

| 层级 | 结论 | 依据 | 证据路径 |
|---|---|---|---|
| L1 自足性 | 通过 | 文首符号表（§1.4）+ 首次出现处一句话定义（OR、BH、A 类验证、四口径、Alpha、R3、H6、T1、T2 均已补） | `iter-02-sub2-biomarker.tex` L57-87、L99、L135、L145-149、L192、L267、L327 |
| L2 叙事脚手架 | 通过 | 全部 9 个 `\section` 首段均以「本节约一句话：……」开头 | L29、L91、L118、L153、L237、L272、L295、L309、L331 |
| L3 符号表 | 通过 | §1.4 符号表含公式/字母/含义/首次出现四列 | L57-87 |
| L4 无 pkl 字段侵入正文 | 通过 | 正文（abstract/§2.1/§5.2）已改人话，无 `\pkl{...}` 字段路径；仅 §9 图表占位表保留 pkl+字段（该节合法用途） | L20-25、L93-96、L264、L336-354 |
| L5 读者代理试读 | 通过 | reader-audit 二轮通过 | `reader-audit-sub2.md` L115-119 |

### 4. 图表占位符 — 通过

§9 图表占位表每行含三要素：**图名** / **数据源（pkl+字段）** / **论文位置**，共 8 张图/表规格齐全。

| 图名 | 数据源（pkl 字段） | 论文位置 |
|---|---|---|
| 每病稳定特征频率条形图 | `per_disease.<D>.stable_features.{feature,frequency,rank}` | §4.1 |
| 每病标志物汇总表 | `per_disease.<D>.biomarker_table` | §4.2 |
| 两路信号统计表 | `per_disease.<D>.two_path_signals` | §5.1 |
| 共现 Spearman 相关热图 | `per_disease.<D>.cooccurrence.spearman_matrix` | §6.1 |
| 共现/互斥网络 | `per_disease.<D>.cooccurrence.cooccurrence_edges` | §6.2 |
| 佐证一致性（VIP vs Alpha） | `per_disease.<D>.topN_consistency` + `vip` | §3.4 |
| 跨疾病 Jaccard 重叠热图 | `cross_disease.jaccard_matrix` | §7.1 |
| 共同/疾病特异性标志物表 | `cross_disease.common_biomarkers` + `disease_specific` | §7.2 |

证据：`iter-02-sub2-biomarker.tex` L336-354。

---

## 三、pkl 数字抽核结果

重跑只读提取（`S2-results.pkl` / `S2-preprocessed.pkl`），报告/素材关键数字与 pkl 实际值对照：

| 数字 | 报告/素材值 | pkl 实际值 | 一致 |
|---|---|---|---|
| 过滤前/后特征数 | 1331 / 264 | meta.n_features_before=1331 / after=264 | ✓ |
| 三病样本数 | 121/110/253 | per_disease.<D>.n_samples | ✓ |
| 三病患病/健康 | 48/73, 25/85, 164/89 | per_disease.<D>.n_pos/n_neg | ✓ |
| τ / B_full / B_cv / C | 0.5 / 100 / 50 / 0.1 | meta.tau/B_full/B_cv/C_lasso | ✓ |
| CLR δ / FDR m / VIP 阈值 | 6.5e-6 / 1331 / 1.5 | meta.clr_delta/fdr_m/vip_threshold | ✓ |
| 三病稳定数 | 4/4/20 | per_disease.<D>.n_stable | ✓ |
| 三病 Fisher/Wilcoxon 显著数 | 4/1, 6/1, 0/0 | per_disease.<D>.n_fisher_sig/n_wilcoxon_sig | ✓ |
| CRC 全量频率 | 0.99/0.94/0.62/0.52 | stable_features[].frequency | ✓ |
| CRC CV 折内频率 | 0.96/0.72/0.39/0.30 | stable_features[].cv_frequency | ✓ |
| CRC Fisher FDR | 1.24e-5/3.94e-5/1.72e-2/5.99e-1 | biomarker_table[].fisher_fdr | ✓ |
| IBD 全量频率 | 0.81/0.75/0.55/0.53 | stable_features[].frequency | ✓ |
| IBD CV 折内频率 | 0.68/0.66/0.48/0.32 | stable_features[].cv_frequency | ✓ |
| IBD Fisher FDR | 6.75e-3×3 / 3.98e-2 | biomarker_table[].fisher_fdr | ✓ |
| Obesity Top3 频率 | 0.89/0.84/0.72 | stable_features[].frequency | ✓ |
| 主导信号 | CRC/IBD 全 presence；Obesity 18 presence+2 abundance | two_path_signals[].dominant_signal（2 abundance=Bacteroides_ovatus、Ruminococcus_bromii） | ✓ |
| 共现边数 | 4/6/24 | cooccurrence.cooccurrence_edges 长度 | ✓ |
| CRC 最强共现对 | 0.76 / 8.8e-7 / 12.9 | cooccurrence_edges[0]（0.7618/8.82e-7/12.86） | ✓ |
| Jaccard 重叠 | 全 0 | cross_disease.jaccard_matrix | ✓ |
| 共同标志物数 | 0 | cross_disease.common_biomarkers（空） | ✓ |
| 疾病特异标志物数 | 4/4/20 | cross_disease.disease_specific.<D> 长度 | ✓ |
| VIP 排名 Spearman | 0.54/0.52/0.35 | topN_consistency.spearman_rank_vip（0.5386/0.5154/0.3474） | ✓ |
| 已知标志物 | CRC 3/4、IBD 1/4 | biomarker_table[].known_biomarker | ✓ |

> 编译验证：`iter-02-sub2-biomarker.pdf` 7 页、`writing-material-sub2.pdf` 3 页，log 无 `??` 未解析引用、无 undefined 引用。

---

## 四、问题清单

| # | 问题 | 严重度 | 证据路径 |
|---|---|---|---|
| P1 | 写作素材「关键数字表」未登记报告 §4.2 标志物表「已知标志物」列的 pkl 字段路径（`known_biomarker`），该列溯源不完整；且「禁写句」引用字段名写为 `known`，实际字段为 `known_biomarker`。数值本身正确（已核 pkl）。 | 低 / B 级 | `writing-material-sub2.tex` L52-75（关键数字表无 known 行）、L107-108（禁写句） |
| P2 | 交接文档 `handoff-S2-report-agent.md` §0/§2.2 称「CRC 4 个全部命中已知标志物」，与 pkl（Clostridium_hathewayi known_biomarker=False，3/4 命中）及报告（3/4）不一致。报告已正确采用 3/4，属交接文档内部措辞错误，不影响报告正确性。 | 低 / B 级 | `handoff-S2-report-agent.md` L16、L46-53 vs pkl `known_biomarker` |

> 两项均为低严重度 B 级（仅表述/溯源完整性，不影响数字正确性与报告定稿质量），不构成门禁阻断。

---

## 五、通过/不通过 判定

**判定：通过。**

门禁 A·B（2.4 报告定稿部分）四项判定内容全部通过：报告定稿（STATUS=done、素材全填、数字可溯、无 TODO、试读通过）、数字一致性（抽核 ≥3 个关键数字全部与 pkl 一致）、可读性（L1-L5 达标）、图表占位符（含数据源 pkl+字段、图名、论文位置）。问题清单仅 2 项低严重度 B 级（溯源完整性/交接措辞），不阻断定稿。

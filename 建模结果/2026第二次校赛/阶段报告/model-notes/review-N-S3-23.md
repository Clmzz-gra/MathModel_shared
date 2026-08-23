# 门禁 N 审查结论：S3 报告初稿（2.3 部分）

> 审查代理（report preset）| 自动模式 | 日期：2026-08-21
> 审查对象：`solution/internal-reports/iter-02-sub3-cross-disease.tex`（阶段 2.3 内容段初稿）
> 审查范围：门禁 N 判定内容之「2.3 报告初稿」部分
> 角色+模型：审查代理（report preset，deepseek-v4-flash:0731）

---

## 一、必读清单已读汇报

开工前已 Read 以下文件并遵守其中规则：

| # | 文件 | 状态 |
|---|---|---|
| 1 | `E:\MathModel_pj\TRAE-报告.md`（2.3 报告规范） | ✅ 已读 |
| 2 | `E:\MathModel_pj\TRAE-规范.md` B 节（产出格式） | ✅ 已读 |
| 3 | `solution/internal-reports/iter-02-sub3-cross-disease.tex`（报告初稿） | ✅ 已读 |
| 4 | `solution/internal-reports/writing-material-S3.tex`（写作素材独立文件） | ✅ 已读 |
| 5 | `solution/model-notes/handoff-S3-report-agent.md`（建模→报告交接） | ✅ 已读 |
| 6 | `outputs/data/S3-results.pkl`（python 只读提取关键字段） | ✅ 已读 |

---

## 二、判定内容逐项结论

### 判定 1：报告初稿可读性（L1-L5）—— 通过（附 B 级小问题）

| 子项 | 结论 | 依据 | 证据路径 |
|---|---|---|---|
| L1 自足性 | 基本通过 | 文首有符号表 + 跨问/裁定代号表；符号首次出现处有内嵌定义（LODO/AUC/J/P_cal/silhouette 等）；R1-R4 在 §5.2 定义。**小缺口**：S1/S2 子问题引用（§2/§3/§6/§7）未在首次出现处一句话说明指代，仅「S1 口径」在跨问代号表定义 | `iter-02-sub3-cross-disease.tex` L61-73、L105/107/132/260/300 |
| L2 叙事脚手架 | 通过 | 每个 `\section` 首段均以「本节约一句话：……」开头（§1-§7 全部命中）；公式/表格前均有动机句 | `iter-02-sub3-cross-disease.tex` L79/102/126/183/199/223/294 |
| L3 符号表 | 通过 | 文首符号表（公式/字母/含义三列）+ 跨问代号表 | `iter-02-sub3-cross-disease.tex` L44-73 |
| L4 无 pkl 字段侵入正文 | 通过 | 正文无 `\texttt{pkl.字段}` 写法；pkl 字段路径全部位于注释（头部说明 + 图表规格注释） | grep 命中 7 处均在 `%` 注释行 |
| 数字填充完整（无 \todo 残留） | 通过 | 报告正文无 `\todo`/`TBD`/`TODO`/待填 残留 | grep 无命中 |
| 编译通过 | 通过 | xelatex 编译成功，输出 7 页 PDF；无错误/未解析引用/`??`；仅 2 处 Overfull \hbox 排版警告 | `iter-02-sub3-cross-disease.log`（Output written, 7 pages） |

### 判定 2：取数规则 —— 通过（数字正确可溯，溯源登记有 B 级缺口）

- **数字从 pkl 只读提取、与 pkl 一致**：✅ 全部关键数字经 python 只读提取核对一致（详见「四、报告数字抽核结果」）。
- **正文写人话不标 pkl 字段路径**：✅ 正文只写值+含义，pkl 路径全部进注释/素材。
- **溯源进素材文件**：⚠️ 大部分数字已登记进 `writing-material-S3.tex` 关键数字表（含 `<pkl名>.<键路径>`），但 **§4.2 R3 阈值迁移辅指标（C1/C2/C3 的 ACC/灵敏/特异/F1 共 12 个数字）与 §6.3「0.5 概率阈值灵敏度 0.165」未登记**。数字本身已与 pkl 核对正确（`fallback.R3_weighted.C{1,2,3}.{acc,sensitivity,specificity,f1}`、`strategy_compare.D_calibrated.C3.thr05_sensitivity=0.1646`），属溯源登记不完整（B 级），非数字错误。

### 判定 3：写作素材完整性 —— 通过

| 子项 | 结论 | 证据路径 |
|---|---|---|
| 关键数字表（每个数字标注 `<pkl名>.<键路径>`） | ✅ 通过 | `writing-material-S3.tex` L34-78 |
| 图表清单 | ✅ 通过 | `writing-material-S3.tex` L80-96 |
| 可写/禁写句 | ✅ 通过 | `writing-material-S3.tex` L98-107 |
| AI 标注 | ✅ 通过（`[AI-X-Y]` 占位待阶段 3.3 填） | `writing-material-S3.tex` L109-111 |
| 章节映射 / 故事线 / 待裁定项 | ✅ 通过 | `writing-material-S3.tex` L16-32、L113-122 |
| 版本戳 | ✅ 通过（`% VERSION: 2026-08-21-内容段初版`） | `writing-material-S3.tex` L6 |

### 判定 4：图表占位符 —— 通过

报告内 4 处图表占位符（策略对比表、衰减归因表、迁移方向图、阈值漂移图）均含**数据源 pkl+字段、图名、论文位置**规格说明，且与 handoff 图表清单一致。

| 图表 | 数据源 pkl+字段 | 图名 | 论文位置 | 证据路径 |
|---|---|---|---|---|
| 四策略 AUC 对比 | `strategy_compare.{A_direct,B_shared,C_genus,C_phylum,D_calibrated}.*.auc + mean_auc` | 四策略跨疾病 AUC 对比 | 结果与讨论（策略对比节） | `iter-02-sub3-cross-disease.tex` L174-177 |
| 衰减归因表 | `decay_attribution.{CRC,IBD,Obesity}.{domain_auc,cross_auc,decay,dominant_cause}` | 三分法衰减归因表 | 结果与讨论（衰减归因节） | L250-253 |
| 迁移方向图 | `migration_analysis.{direction_consistent_count,direction_flipped_count,shared_species_list}` | 共享标志物跨疾病迁移方向一致性 | 结果与讨论（深度迁移节） | L265-268 |
| 阈值漂移图 | `threshold_drift.{train_baseline,test_baseline,boundary_position,diagnosis}` | C3 阈值漂移诊断 | 结果与讨论（阈值漂移节） | L282-285 |

---

## 三、问题清单

| # | 问题 | 严重度 | 证据路径 |
|---|---|---|---|
| 1 | §4.2 R3 阈值迁移辅指标（C1/C2/C3 的 ACC/灵敏/特异/F1 共 12 个数字）未登记进写作素材关键数字表（数字已与 pkl 核对正确，属溯源登记不完整） | B | `iter-02-sub3-cross-disease.tex` L191 vs `writing-material-S3.tex` L34-78 |
| 2 | §6.3「0.5 概率阈值灵敏度 0.165」未登记进关键数字表（字段存在于 pkl `strategy_compare.D_calibrated.C3.thr05_sensitivity=0.1646`，数字正确） | B | `iter-02-sub3-cross-disease.tex` L277 vs `writing-material-S3.tex` L34-78 |
| 3 | 自足性小缺口：S1/S2 子问题引用（§2/§3/§6/§7）未在首次出现处一句话说明指代（仅「S1 口径」在跨问代号表定义） | B | `iter-02-sub3-cross-disease.tex` L105/107/132/260/300 |
| 4 | 报告无文献引用（`\cite`/`\bibliography` 缺失），TRAE-报告.md 要求「方法文献 1-2 篇」 | B | grep 无 `\cite` 命中 |
| 5 | 排版：2 处 Overfull \hbox（符号表/跨问代号表过宽） | B | `iter-02-sub3-cross-disease.log`（L47-59、L64-73） |
| 6 | 笔误：L113「本约说明为什么用 LODO」应为「本节说明」 | B | `iter-02-sub3-cross-disease.tex` L113 |

> 全部为 B 级（表述/溯源/排版/引用完整性），**无 A 级问题**（无数字错误、无口径不明、无 pkl 字段侵入正文）。按 TRAE-报告.md 待裁定项 A/B 分级，B 级可「按拟议继续并记录，人类批量裁定」。

---

## 四、报告数字抽核结果（python 只读提取对照 pkl）

全部关键数字经 `outputs/scratch/extract-S3-review*.py` 只读提取核对，**与 pkl 一致**：

| 报告数字 | pkl 实际值 | 一致 |
|---|---|---|
| 过滤特征数 264 / 样本 484 | `X_filtered.shape=(484,264)`、`meta.filter_rule=1331->264` | ✅ |
| CLR 常数 6.5e-6 | `meta.clr_delta=6.5e-06` | ✅ |
| C1/C2/C3 正类占比 39.7%/22.7%/64.8% | `A_direct.C{1,2,3}.test_pos_frac=0.3967/0.2273/0.6482` | ✅ |
| 策略 A 三组合 0.5674/0.5882/0.5253、均值 0.5603 | `A_direct.C{1,2,3}.auc`、`mean_auc=0.5603` | ✅ |
| 策略 B 0.5417/0.6080/0.5218、均值 0.5572、共享 252 | `B_shared.*`、`shared_feature_count=252` | ✅ |
| 策略 C 属级 0.3616/0.4861/0.5440、均值 0.4639、106 维 | `C_genus.*`、`n_features=106` | ✅ |
| 策略 C 门级 0.4141/0.5261/0.5999、均值 0.5134、11 维 | `C_phylum.*`、`n_features=11` | ✅ |
| 策略 D 均值 0.5603（=A） | `D_calibrated.mean_auc=0.5603`、`base_strategy=A_direct` | ✅ |
| R1 0.5092 / R2 0.5603 / R3 0.6068 / R4 0.5947 | `fallback.R{1,2,3,4}_*.mean_auc` | ✅ |
| R3 三组合 0.5945/0.6489/0.5771 | `R3_weighted.C{1,2,3}.auc` | ✅ |
| R3 阈值辅指标（C1 ACC 0.6446/灵敏 0.2292/特异 0.9178/F1 0.3385；C2 0.5727/0.6800/0.5412/0.4198；C3 0.4150/0.1585/0.8876/0.2600） | `R3_weighted.C{1,2,3}.{acc,sensitivity,specificity,f1}` | ✅ |
| 域内 AUC 0.7811/0.8588/0.6638 | `domain_auc.{CRC,IBD,Obesity}` | ✅ |
| 衰减量 -0.2138/-0.2706/-0.1384 | `decay_attribution.{CRC,IBD,Obesity}.decay` | ✅ |
| 方向一致 387 / 翻转 369 / 51.2% / p=0.5364 | `migration_analysis.*`（consistent_fraction=0.5119） | ✅ |
| 训练 0.316 vs 测试 0.648、Δ=+0.332 | `threshold_drift.train_baseline=0.3160/test_baseline=0.6482/delta=0.3322` | ✅ |
| Youden τ\*=0.9205、96.0% 分位、灵敏度 0.024 | `threshold_drift.youden_threshold=0.9205/boundary_position=0.9605/sensitivity=0.0244` | ✅ |
| 0.5 概率阈值灵敏度 0.165 | `strategy_compare.D_calibrated.C3.thr05_sensitivity=0.1646` | ✅ |

> 抽核脚本：`outputs/scratch/extract-S3-review.py`、`extract-S3-review2.py`、`extract-S3-review3.py`（只读提取，未写新 pkl、未改脚本）。

---

## 五、审查结论

**判定：通过**（附 B 级问题清单，建议修订时一并处理）

- 报告初稿可读性达标（L1-L5 通过，编译通过，无 \todo 残留）；
- 取数规则合规（数字全部从 pkl 只读提取、与 pkl 一致、正文写人话不标 pkl 路径）；
- 写作素材完整（关键数字表/图表清单/可写禁写句/AI 标注/版本戳齐全）；
- 图表占位符规格完整（数据源 pkl+字段/图名/论文位置）；
- 无 A 级问题（无数字错误、无口径不明、无 pkl 字段侵入正文）。

**需在修订（2.4）处理的问题**：问题 1-2（溯源登记补全）、问题 3（S1/S2 自足性说明）、问题 4（补文献引用）、问题 5（排版）、问题 6（笔误）。均为 B 级，不影响门禁 N 通过判定。

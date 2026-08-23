# 门禁 N 审查：S2 报告初稿（2.3 内容段）

> 审查代理：report preset（自动模式派遣）
> 审查对象：`solution/internal-reports/iter-02-sub2-biomarker.tex`（报告初稿内容段）
> 关联文件：`solution/internal-reports/writing-material-sub2.tex`（写作素材独立文件）
> 数据源：`outputs/data/S2-results.pkl`、`outputs/data/S2-preprocessed.pkl`（只读提取）
> 日期：2026-08-21
> 角色+模型：report 审查代理（deepseek-v4-flash:0731）

---

## 一、必读清单已读汇报

开工前已 Read 以下文件并遵守其规则：

- [x] `E:\MathModel_pj\TRAE-报告.md`（2.3 报告规范：取数规则/自足性/叙事脚手架/素材独立文件/图表占位）
- [x] `E:\MathModel_pj\TRAE-规范.md` B 节（产出格式：内部报告 .tex、中文撰写、review-notes .md）
- [x] `solution/internal-reports/iter-02-sub2-biomarker.tex`（审查对象，报告初稿）
- [x] `solution/internal-reports/writing-material-sub2.tex`（写作素材独立文件）
- [x] `solution/model-notes/handoff-S2-report-agent.md`（建模→报告交接，章节映射/关键数字/口径声明）
- [x] `outputs/data/S2-results.pkl`、`outputs/data/S2-preprocessed.pkl`（python 只读提取关键字段）

已读清单汇报完毕，开始审查。

---

## 二、判定内容逐项结论

### 1. 报告初稿可读性（L1-L5）

**结论：通过（含 1 处需修正的声明）**

- **自足性（L1）**：文首有符号表（§1.4，公式/字母/含义/首次出现四列），每个符号首次出现处一句话内嵌定义（如 §3.1 定义 $\hat{\pi}_j$、$\tau$、$B$）；跨问代号 F2/F3/F6、S1 均在符号表与正文首次出现处一句话说明指代。符合自足性规则。
  - 证据：`iter-02-sub2-biomarker.tex` L58-82（符号表）、L78-79（跨问代号表）。
- **叙事脚手架（L2）**：每个 `\section` 首段以「本节约一句话：……」开头（§1-§9 全部满足）；公式/表格前有动机句。符合。
  - 证据：各 section 首段（L29、L85、L113、L144、L228、L260、L282、L296、L316）。
- **符号表（L3）**：存在且完整（§1.4）。符合。
- **无 pkl 字段侵入正文（L4）**：正文结果数字均写人话（值+含义），未在结果叙述中标 pkl 字段路径；`\pkl{}` 仅出现在：摘要数据源说明、§2.1 输入数据描述、§5.2 知识库引用、§9 图表占位节（该节按规范必须含 pkl+字段）。符合取数规则。
  - 证据：正文结果节（§4-§8）无 `\pkl{per_disease...}` 类字段路径；字段路径集中在 §9 图表占位表（L330-337）。
- **数字填充完整（L5）**：正文无残留 `\todo`/`TBD`/`TODO`/待填充（grep 无命中）。符合。
  - 证据：grep `\\todo|TBD|TODO|待填充|待补充` 于报告正文 → 无匹配。
- **编译通过**：xelatex 编译成功，7 页 PDF，无 error、无 undefined reference、无 `??`。
  - 证据：`iter-02-sub2-biomarker.log`（Output written ... 7 pages，无 `^!`/Error/undefined）。

### 2. 取数规则

**结论：通过（数字抽核全部一致，除 1 处「主导信号全部 presence」声明与 pkl 不符，见问题 P1）**

- 报告数字均从 pkl 只读提取，未抄 handoff 快查数字；正文写人话不标 pkl 字段路径（溯源进素材文件）。
- 数字抽核结果（重跑只读提取对照 pkl，全部一致）：
  - 样本数 CRC 121/48/73、IBD 110/25/85、Obesity 253/164/89 ✓（`S2-preprocessed.pkl per_disease.<D>.n_samples/n_pos/n_neg`）
  - 特征数 1331→264 ✓（`meta.n_features_before/after`）
  - τ=0.5、B=100/50、C=0.1、δ=6.5e-06、FDR m=1331、VIP>1.5 ✓（`S2-results.pkl meta.*`）
  - 稳定特征数 4/4/20 ✓（`per_disease.<D>.n_stable`）
  - CRC 频率 0.99/0.94/0.62/0.52、折内 0.96/0.72/0.39/0.30 ✓
  - IBD 频率 0.81/0.75/0.55/0.53、折内 0.68/0.66/0.48/0.32 ✓
  - CRC Fisher FDR 1.24e-05/3.94e-05/1.72e-02/5.99e-01 ✓；IBD 6.75e-03×3/3.98e-02 ✓
  - 两路显著数 CRC 4/1、IBD 6/1、Obesity 0/0 ✓（`n_fisher_sig/n_wilcoxon_sig`）
  - 共现边数 CRC 4（全 cooccur）、IBD 6（3 cooccur/3 exclude）、Obesity 24 ✓
  - CRC 最强共现对 Spearman 0.76 / Fisher p 8.8e-07 / OR 12.9 ✓；Porphyromonas_somerae 0.83 ✓；IBD Alistipes↔Bifidobacterium OR 0.19 ✓
  - Jaccard 全 0、common 空、disease_specific 4/4/20 ✓
  - 已知标志物命中：CRC 3/4（Clostridium_hathewayi known=False）、IBD 1/4（Bifidobacterium_bifidum）✓
- **唯一不符**：报告 §5.1/§8 声称「三病入选标志物主导信号**全部为 presence**」，但 pkl 中 Obesity 有 2 个特征（Bacteroides_ovatus、Ruminococcus_bromii）`dominant_signal='abundance'`（Obesity 分布 presence 18 / abundance 2）。handoff §0 用「几乎全为 presence」（准确），报告过度声明为「全部」。见问题 P1。

### 3. 写作素材完整性

**结论：通过（含 2 处需同步修正，见 P1/P2）**

- 关键数字表：完整，每个数字标注 `<pkl名>.<键路径>`（如 `S2-results.pkl per_disease.CRC.stable_features[].frequency`），字段路径全部有效（逐一核对 pkl 结构存在）。符合。
  - 证据：`writing-material-sub2.tex` L43-77。
- 图表清单：完整（8 项，含论文用途+建议插入位置）。符合。
  - 证据：L79-97。
- 可写/禁写句：有（L99-111），含「不得写 CRC 4 个全部命中」「不得写 Obesity 显著」「不得写共现为因果」等禁写句。符合。
- AI 标注：有（L113-115），但为 `\todo{[AI-2-x] 等编号}` 占位，待阶段 3.3 填充（draft 阶段可接受，定稿前须填）。
- 版本戳：有（L4 `% VERSION: 2026-08-21-内容段取数...`）。符合。
- **需修正**：可写句「三病入选标志物主导信号均为存在/缺失」与 pkl 不符（同 P1）；未显式声明 RF 佐证层不可用（同 P2）。

### 4. 图表占位符

**结论：通过（含 1 处需修正，见 P2）**

- §9 图表占位表含：图名、数据源（pkl+字段）、论文位置三要素，字段路径全部有效。符合。
  - 证据：`iter-02-sub2-biomarker.tex` L322-340。
- **需修正**：占位表「佐证一致性（RF/VIP vs Alpha Top-N 排名）」数据源含 `rf_importance`，但 RF 佐证层已退化不可用（handoff 口径声明 #4），不应作为正式图数据源。见问题 P2。

---

## 三、问题清单

| # | 问题 | 严重度 | 证据路径 |
|---|------|--------|----------|
| P1 | 报告 §5.1 与 §8 声称「三病入选标志物主导信号**全部为 presence**」，但 pkl 中 Obesity 有 2 个特征（Bacteroides_ovatus、Ruminococcus_bromii）`dominant_signal='abundance'`（Obesity 分布 presence 18 / abundance 2）。handoff §0 用「几乎全为 presence」（准确），报告过度声明为「全部」。素材可写句「三病入选标志物主导信号均为存在/缺失」同错。**A 级（数字/口径错误，需修正）** | 高 | 报告 L234、L245、L302；素材 L104；pkl `per_disease.Obesity.two_path_signals[].dominant_signal`（Bacteroides_ovatus=abundance、Ruminococcus_bromii=abundance）；handoff L16 |
| P2 | RF 佐证层不可用未在报告/素材显式声明：handoff 口径声明 #4 要求「报告不引用 RF 数字，仅以 VIP 作独立复现佐证」，但报告 §3.4 将 RF 列为佐证方法、§9 图表占位「佐证一致性」数据源含 `rf_importance`（RF 已退化 ~1e-17）。**B 级（口径/表述）** | 中 | 报告 L138-141、L335；handoff L84、L117 |
| P3 | §9 标题「供门禁 2 后出图」为骨架段残留措辞，当前已过门禁 2（处于门禁 N），应改为「报告定稿后出图」。**B 级（表述）** | 低 | 报告 L317 |
| P4 | 图名「每病稳定特征频率条形图（Top 10--20）」与 CRC/IBD 实际仅 4 个稳定特征不符（Top 10--20 为骨架段目标，实际入选 4/4/20）。**B 级（命名）** | 低 | 报告 L330；pkl `per_disease.<D>.n_stable`=4/4/20 |
| P5 | 素材文件 表述库参考（L29/34/37）与 AI 标注（L115）为 `\todo{}` 占位未填。draft 阶段可接受，但定稿（STATUS: done）前必须填满。**C 级（待定稿前完成）** | 低 | 素材 L29/34/37/115 |

---

## 四、审查结论

**不通过（需修订后复审）**

报告初稿整体质量高：可读性（自足/脚手架/符号表/无 pkl 侵入正文）、数字填充完整、编译通过、写作素材完整、图表占位符含三要素，数字抽核除 P1 外全部与 pkl 一致。

但存在 **1 处 A 级事实性错误（P1）**：报告与素材将 Obesity 主导信号过度声明为「全部 presence」，而 pkl 明确有 2 个 `abundance` 特征。该声明在报告 §5.1、§8 结论与素材可写句中重复出现，会直接传播到终稿，属「数字/口径错误」A 级，按取数规则须修正（改为「几乎全部为 presence」或按 pkl 精确列出 2 个 abundance 例外）后方可通过。

**修订要求**：
1. 修正 P1（报告 §5.1 表、§5.1 正文、§8 结论、素材可写句），与 pkl 一致；
2. 建议一并处理 P2（显式声明 RF 佐证层不可用、图表占位移除 rf_importance 数据源）；
3. P3/P4/P5 为低严重度，可在修订时顺带处理或登记待定稿前完成。

修订后需复审（重点复核 P1 修正是否与 pkl 一致）。

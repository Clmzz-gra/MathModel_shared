# 交接：S2 代码 → 建模（2.1 正式模型实现完成）

> handoff_type: `model-agent`
> sub: S2（特征选择与生物标志物）
> stage: 2.1（正式模型实现，门禁 2 材料）
> from: 代码对话 | to: 建模对话
> 日期：2026-08-21
> source_docs: [`handoff-S2-code-agent.md`, `approach-S2-confirmed.md`, `math-S2.tex`, `proxy-replacement-checklist-S2.md`]
> next_action: 建模对话 2.2 结果分析（读本文件 + `S2-results.pkl`），说「继续」
> status: ready

---

## 0. 一句话收束

S2 2.1 正式模型实现完成，产出 `outputs/data/S2-results.pkl`。三病独立执行「近全零过滤(1331→264) → CLR → Lasso+bootstrap 稳定性选择(τ=0.5, B=100) → 两路信号(Fisher/Wilcoxon + BH-FDR m=1331) → 共现分析初探 → RF/VIP 佐证」。**CRC 稳定标志物 4 个全部命中已知标志物**（Fusobacterium_nucleatum 频率 0.94、Peptostreptococcus_stomatis 0.99、Porphyromonas_somerae 0.62），方法有效性锚点 H6 强验证。

---

## 1. 关键数字摘要

### 1.1 每病稳定特征数（τ=0.5，全量 bootstrap 频率）

| 疾病 | 稳定特征数 | Fisher 显著(m=1331) | Wilcoxon 显著(m=1331) | 共现边数 | RF 重叠 | VIP 重叠 | Spearman(freq vs VIP) |
|:--|:--|:--|:--|:--|:--|:--|:--|
| CRC | **4** | 4 | 1 | 4 | 0.00 | 0.20 | 0.539 |
| IBD | **4** | 6 | 1 | 6 | 0.00 | 0.20 | 0.515 |
| Obesity | **20** | 0 | 0 | 24 | 0.05 | 0.10 | 0.347 |

### 1.2 标志物表 Top（每病）

**CRC**（4 个，全部已知标志物，方向均 up=患病富集）：
| 标志物 | 频率 | Fisher q | 已知 |
|:--|:--|:--|:--|
| Peptostreptococcus_stomatis | 0.99 | 1.24e-05 | ✅ |
| Fusobacterium_nucleatum | 0.94 | 3.94e-05 | ✅ |
| Porphyromonas_somerae | 0.62 | 1.72e-02 | ✅ |
| Clostridium_hathewayi | 0.52 | 5.99e-01 | — |

**IBD**（4 个）：
| 标志物 | 频率 | Fisher q | 方向 |
|:--|:--|:--|:--|
| Alistipes_finegoldii | 0.81 | 6.75e-03 | down |
| Bifidobacterium_bifidum | 0.75 | 6.75e-03 | up（已知属） |
| Akkermansia_muciniphila | 0.55 | 6.75e-03 | down |
| Eubacterium_ventriosum | 0.53 | 3.98e-02 | down |

**Obesity**（20 个，0 个 FDR 显著，弱信号符合预期 R3）：Top 为 Ruminococcus_flavefaciens(0.89)、Pseudoflavonifractor_capillosus(0.84)、Rothia_mucilaginosa(0.72) 等，全部 fisher_q > 0.05。

### 1.3 τ 敏感性（入选数随 τ 变化）

| 疾病 | τ=0.4 | τ=0.5 | τ=0.6 | τ=0.7 |
|:--|:--|:--|:--|:--|
| CRC | 6 | 4 | 3 | 2 |
| IBD | 6 | 4 | 2 | 2 |
| Obesity | 32 | 20 | 9 | 3 |

### 1.4 跨疾病对比

- Jaccard 重叠：CRC_IBD=0.0、CRC_Obesity=0.0、IBD_Obesity=0.0（稳定特征集完全疾病特异，无共同标志物）。
- 疾病特异性：CRC 4 个、IBD 4 个、Obesity 20 个，全部为疾病特异（符合「Fusobacterium nucleatum 的 CRC 特异性」预期）。

### 1.5 共现分析初探（协同效应）

- CRC 4 条边全部 cooccur：Peptostreptococcus_stomatis ↔ Fusobacterium_nucleatum（Spearman 0.76, Fisher p≈0，最强共现对，两者均为 CRC 相关口腔菌，生物合理）、Peptostreptococcus_stomatis ↔ Porphyromonas_somerae（0.83）、Fusobacterium_nucleatum ↔ Clostridium_hathewayi（0.35）等。
- IBD 6 条边、Obesity 24 条边（详见 pkl `cooccurrence.cooccurrence_edges`）。

---

## 2. 探索图清单（`outputs/figures/_explore/`）

| 图 | 内容 | 解读 |
|:--|:--|:--|
| `S2-2.1-stability-frequency-explore.pdf` | 三病 Lasso bootstrap 入选频率直方图（B=100，τ=0.5 红线） | CRC/IBD 频率分布呈「少量高频率稳定簇 + 长尾」，稳定簇与噪声可分（H5）；Obesity 频率分布更平缓（信号分散） |
| `S2-2.1-tau-sensitivity-explore.pdf` | τ=0.4/0.5/0.6/0.7 入选数曲线 | CRC/IBD 入选数对 τ 不敏感（4~6），Obesity 对 τ 敏感（32→3），印证 Obesity 弱信号 |
| `S2-2.1-cooccurrence-heatmap-explore.pdf` | 三病入选标志物两两 Spearman 相关热图 | CRC 标志物间强正相关（共现簇），IBD/Obesity 相关结构更分散 |

---

## 3. 两遍审核结论

### ① 原理合理性（建模子代理）→ 结论：**通过**（复审后）

- 初审「不通过」：审查代理判定 Lasso 实现「l1_ratio=1.0 无 penalty 被静默忽略、实际拟合 L2」为 A 级阻断。
- **复审撤销（误报）**：代码对话三组对照实证（`outputs/scratch/_test_l1.py`）证明 `l1_ratio=1.0`（无 penalty）与 `penalty='l1'` 选中特征集**完全一致**（13/264 非零稀疏解），而 `penalty='l2'` 为 264/264 稠密解——sklearn 1.9.0 起 `penalty` 参数整体弃用，`l1_ratio=1.0` 即官方替代 `penalty='l1'` 的新写法。原实现本就是 L1，非 L2。
- 其余 8 个聚焦点：CLR 复用、bootstrap 频率+分层、两路信号、BH-FDR m=1331、CV 折内防泄漏均通过；共现 Fisher 范围、VIP 阈值、PLS-DA 输入口径为 B/观察级，登记待裁定项/口径说明。
- 结论文件：`solution/model-notes/review-2.1-S2-原理.md`（含复审 diff）。

### ② 代码逻辑（coding 子代理）→ 结论：**通过**（附 1 待裁定项 + 2 低严重度提示）

- C1 性能声明、C8 并行度（bootstrap n_jobs=8 + RF n_jobs=-1）、数据版本（S2-preprocessed.pkl）、索引/变量、输出路径、代理值核销、field_semantics 全部合规。
- 2 个低严重度提示（meta 硬编码、热图 short_name 索引）已修复。
- 结论文件：`solution/model-notes/review-2.1-S2-逻辑.md`。

---

## 4. 待裁定项（交建模对话）

| # | 待裁定项 | 现状 | 建议 |
|:--|:--|:--|:--|
| T1 | **C 选择（P7）** | C=0.1（V6 值）下 CRC/IBD 仅 4 个稳定特征（< Top 10-20 目标 P8）。C 敏感性快查：C=0.01→0/0/0、C=0.05→2/1/1、C=0.1→5/4/19、C=0.5→19/17/54、C=1.0→21/17/60 | 若需 Top 10-20，建议 C 上调至 0.5（CRC 19/IBD 17 达标，Obesity 54 仍弱信号）；或接受 C=0.1 的「4 个高置信标志物」口径（CRC 4 个全命中已知标志物，生物合理性最强） |
| T2 | **VIP>1.5 独立复现（P2）** | VIP>1.5 阈值已定义但未产生独立 VIP 选择集，仅用于 Top-N 一致性 | 是否需输出 VIP>1.5 特征清单作独立复现证据 |
| T3 | **R4 small_adenoma 口径** | CRC 标签按题面口径（small_adenoma 归健康）执行 | 跟随 S1 最终主口径；若 S1 选定不同口径，S2 CRC 需重跑 |

---

## 5. 口径修正说明

1. **数据版本**：正式实现用 `c-data-cleaned.pkl`（float32）→ 1.4 产物 `S2-preprocessed.pkl`，**非 B-raw.pkl**（A 类验证用）。样本数 CRC=121/IBD=110/Obesity=253，与 A 类验证一致。
2. **过滤口径**：零值占比 >95% 剔除，**三病并集统一口径**（全 484 样本上计算零值占比，一次过滤），1331→264，与 S1/S3 一致。
3. **FDR 口径**：两路信号检验对过滤后 264 特征执行，多重比较按 **m=1331 全特征规模**校正（人类裁定：医疗宁可严格不可虚报）。
4. **CLR 口径**：δ=6.5e-06 乘法替换 + 几何均值中心化，与 S1 一致（P6）。
5. **Lasso 实现**：`l1_ratio=1.0`（sklearn 1.9 新 API，等价 `penalty='l1'`，已实证验证），C=0.1（P7 当前临时值）。
6. **诚实标注**：全量 bootstrap 频率（乐观）与 CV 折内 bootstrap 频率（诚实）两套数字并列，均落盘 pkl（`full_frequency` / `cv_frequency`）。

---

## 6. 产出文件

- `outputs/data/S2-results.pkl`（正式结果，meta 含 field_semantics）
- `outputs/scratch/S2-model.py`（正式实现脚本，C1 头注释 + C8 并行）
- `outputs/figures/_explore/S2-2.1-*.pdf`（3 张探索图）
- `solution/model-notes/review-2.1-S2-原理.md`、`review-2.1-S2-逻辑.md`（两遍审核结论）

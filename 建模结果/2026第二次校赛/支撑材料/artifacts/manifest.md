# manifest.md

> 结果表 Artifact 登记（阶段 2.2 结果分析，2026-08-21）。数据源：`outputs/data/S{N}-results.pkl`。

## S2 结果表（solution/artifacts/tables/）

| 文件 | 内容 | 数据源字段 | 论文位置建议 |
|:--|:--|:--|:--|
| `S2-stable-biomarkers.tex` | 每病稳定标志物表（频率/CV频率/Fisher q/方向/已知/主导信号） | `per_disease.*.stable_features` + `two_path_signals` | 结果节「每病稳定标志物」 |
| `S2-cross-disease.tex` | 三病 Jaccard 重叠矩阵 | `cross_disease.jaccard_matrix` | 结果节「跨疾病对比」 |
| `S2-tau-sensitivity.tex` | τ 敏感性入选数 | `meta.tau_counts` | 稳健性/讨论节 |

## S3 结果表（solution/artifacts/tables/）

| 表 | 文件 | 数据源（pkl 字段） | 论文位置 |
|---|---|---|---|
| S3 四策略对比 | `tables/S3-strategy-compare.tex` | `strategy_compare.<S>.<C>.auc` / `mean_auc` | 结果节 · 四策略对比 |

## 正式图（outputs/figures/，副本 solution/artifacts/charts/，2026-08-21 出图）

> 出图规范：chart-generator skill（零基线/Okabe-Ito 色盲安全/无 3D/无彩虹/数字只取 pkl）。数据源见各图行。
> **✅ 审查状态（2026-08-22 更新）：全部 13 张正式图已通过图审（review-图-2026-sim2-B.md），可作终稿引用版本。** 过程：GPT 更新版（2026-08-22）→ 首审 13 条问题 → pro 修复（6a02fb8/90d4f4e/d60fc6a）→ 复审 13/13 通过。残留可选微调：#D62728 参考线色、IBD 归因"（最强）"叙事标注。

### S1 正式图（数据源 S1-results.pkl + S1-preprocessed.pkl）

| 图 | 内容 | 数据源字段 | 论文位置 |
|:--|:--|:--|:--|
| `S1-roc-curve.pdf` | 三数据集 ROC 曲线 + AUC（L2+RF） | `<ds>.L2_CLR.{AUC,oof_prob}` / `<ds>.RF_raw.{AUC,oof_prob}` + `datasets.<ds>.y` | 结果节·三数据集性能 |
| `S1-performance-compare.pdf` | 三数据集 L2/RF/基线 AUC 对比柱状图 | `<ds>.L2_CLR.AUC` / `<ds>.RF_raw.AUC` / `<ds>.baseline.single_feature_best_AUC` | 结果节·性能对比 |
| `S1-adenoma-sensitivity.pdf` | small_adenoma 四口径敏感性对比 | `adenoma_sensitivity.*.{L2_AUC,RF_AUC,n_samples}` | 结果节·腺瘤敏感性 |
| `S1-feature-importance.pdf` | L2 系数 Top 特征（三数据集） | `<ds>.L2_CLR.coefficients` + `feature_names` | 结果节·特征重要性 |
| `S1-threshold-analysis.pdf` | 阈值-指标曲线（ACC/F1/Recall/Specificity） | `<ds>.L2_CLR.oof_prob` + `datasets.<ds>.{y,minority}` | 结果节·阈值分析 |

### S2 正式图（数据源 S2-results.pkl）

| 图 | 内容 | 数据源字段 | 论文位置 |
|:--|:--|:--|:--|
| `S2-stable-frequency.pdf` | 三病稳定标志物频率直方图（τ=0.5 红线，灰菱形=CV 折内诚实频率） | `per_disease.<D>.stable_features.{feature,frequency,cv_frequency}` | 结果节·每病稳定标志物 |
| `S2-tau-sensitivity.pdf` | τ 敏感性曲线 | `meta.{tau_grid,tau_counts}` | 稳健性/讨论节 |
| `S2-cooccurrence-heatmap.pdf` | 共现 Spearman 相关热图（CRC/IBD） | `per_disease.<D>.cooccurrence.spearman_matrix` | 结果节·共现分析 |
| `S2-cross-disease.pdf` | 三病 Jaccard 重叠矩阵 | `cross_disease.jaccard_matrix` | 结果节·跨疾病对比 |

### S3 正式图（数据源 S3-results.pkl）

| 图 | 内容 | 数据源字段 | 论文位置 |
|:--|:--|:--|:--|
| `S3-strategy-compare.pdf` | 五种策略配置在 3 个 LODO 组合及其均值下的 AUC 对比 | `strategy_compare.<S>.<C>.auc` / `mean_auc` | 结果节·策略配置对比 |
| `S3-decay-attribution.pdf` | 域内 vs 跨疾病 AUC 衰减归因 | `decay_attribution.<D>.{domain_auc,cross_auc,decay,dominant_cause}` | 结果节·衰减归因 |
| `S3-migration-direction.pdf` | 共享物种效应方向一致性（未显著偏离 50% 基准） | `migration_analysis.{direction_consistent_count,direction_flipped_count,consistent_fraction,sign_test_pvalue}` | 结果节·深度迁移 |
| `S3-threshold-drift.pdf` | C3 阈值漂移诊断 | `threshold_drift.{train_baseline,test_baseline,delta_baseline,youden_threshold,boundary_position,sensitivity}` | 结果节·阈值漂移 |

## 待补（报告对话按需出图）

- 共现网络图（CRC/IBD）：数据源 `per_disease.*.cooccurrence.cooccurrence_edges`（节点=标志物，边=cooccur/exclude，标 OR）。
- VIP>1.5 特征清单：数据源 `per_disease.*.vip`（CRC 28/IBD 27/Obesity 23）。

## 数据画像图（阶段 0.4，2026-08-22 登记）

> 数据画像产出（cluster/dim-reduction），GPT 更新版，待图审（未标记，可选入终稿）。

| 图 | 内容 | 论文位置 |
|:--|:--|:--|
| `cluster-tsne.pdf` | 聚类 t-SNE 可视化 | 数据理解节（可选） |
| `pca-scree.pdf` | PCA 碎石图 | 数据理解节（可选） |

## 数据特征图（共享数据层，阶段 0，2026-08-24 登记）

> 出图规范：chart-generator skill（零基线/Okabe-Ito 色盲安全/无 3D/无彩虹/数字只取 pkl）。数据源：`outputs/data/c-data-cleaned.pkl`（484×1333：2 元数据列 + 1331 物种特征列）。共享数据层，不进子问题代号，可入论文「数据概览/数据理解」节。
> 支撑论点：类别不平衡 → 主指标 AUC + 少数类 F1/Recall；近全零过滤 1331→264；CLR 对数比变换动机。

| 图 | 内容 | 关键数值（来源 c-data-cleaned.pkl） | 论文位置 |
|:--|:--|:--|:--|
| `chart-sample-composition.pdf` | 三数据集样本构成堆叠条形图（患病/健康/腺瘤分层） | CRC(Zeller): cancer48/n47/small_adenoma26；IBD(metahit): 患病25(n 85)/crohn4+ulcerative21；Obesity(Chatelier): obesity164/leaness89；患病率 CRC 39.7%/IBD 22.7%/Obesity 64.8% | 数据理解节·类别不平衡 |
| `chart-zero-sparsity.pdf` | 特征零值占比直方图（>95% 红色高亮 + 0.95 阈值线） | 全矩阵零值占比 92.2%；1067/1331 特征零值占比 >95%（剔除），保留 264 | 数据理解节·近全零过滤 1331→264 动机 |
| `chart-abundance-distribution.pdf` | 非零丰度 log10 直方图（标注中位数与范围） | 非零值 min 1e-5 / median 0.0776 (log10≈-1.11) / max 79.96，跨约 7 个数量级 | 数据理解节·CLR 对数比变换动机 |
| `chart-batch-effect.pdf` | 批次效应：PCA/t-SNE 按数据集着色散点（CRC/IBD/Obesity 三色） | PCA 前两主成分解释方差 8.8%+5.3%=14.1%（低，无主导结构）；三数据集在特征空间无清晰分离 → 存在批次/分布差异；支撑：绝对 AUC 不可跨数据集独立比，只比相对基线增益 | 数据理解节·批次效应（支撑"跨数据集只比相对增益"） |
| `chart-known-biomarker-presence.pdf` | 已知标志物在患病/健康组存在率分组柱状图 | CRC 病/健存在率：F.nucleatum 41.7/2.7%、P.stomatis 56.2/8.2%、P.somerae 20.8/0%；IBD：B.bifidum 56.0/11.8%、A.muciniphila 28.0/77.6%；Obesity：B.fragilis 59.1/44.9% | 数据理解节·已知标志物存在性（S2 生物合理性锚点） |

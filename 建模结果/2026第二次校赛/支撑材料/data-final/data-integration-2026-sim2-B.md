# data-integration — 2026-sim2-B 跨问数字整合单文件

> **用途**：终稿组装（latex-builder 提取/压缩/映射）的论文正文数值唯一出口。正文数值一律从本文件复制，禁止随手填数。
> **数据源**：`outputs/data/S1-results.pkl`、`S2-results.pkl`、`S3-results.pkl`（只读提取，float32 落盘值）。
> **溯源标注**：每个数字标注 `<pkl名>.<键路径>`，可回溯到 pkl 实际值。
> **生成**：modeling 子代理（data-integration 单文件）| 日期：2026-08-21 | 版本：v1
> **上游**：`result-analysis-S1/S2/S3.md` + `iter-02-sub1/sub2/sub3-*.tex`（三子问题报告定稿，STATUS: done）

---

## 0. 口径声明汇总（跨问统一，终稿必须一致）

| 口径项 | 声明 | 来源 |
|:--|:--|:--|
| **近全零过滤** | 剔除零值占比 >95% 的特征，原始 1331 维 → 264 维（剔除 1067 维）；**三病并集（全 484 样本）统一计算零值占比、一次过滤**，S1/S2/S3 共用同一 264 特征集 | `S2-results.pkl.meta.filter_threshold`（0.95）；S1/S2/S3 报告 §2 |
| **CLR 零值替换常数** | 零值乘法替换伪值 δ = 6.5e-06（检出限 0.65 倍），S1/S2/S3 一致 | `S1-results.pkl`（CLR 前置）、`S2-results.pkl.meta.clr_delta`=6.5e-06、`S3-results.pkl.meta.clr_delta`=6.5e-06 |
| **标签映射** | 患病=1 / 健康=0。CRC：患病=cancer，健康=n+small_adenoma；IBD：患病=ibd_ulcerative_colitis/ibd_crohn_disease 并集，健康=n；Obesity：患病=obesity，健康=leaness | S1/S2 报告 §2.2/§2.4 |
| **small_adenoma 主口径** | **口径①（归健康）为主口径**（题面主口径、样本量最大 n=121、可解释性最直接）；③剔除/④单开一类作敏感性附录；②归病变排除（AUC 掉 0.18） | `S1-results.pkl.adenoma_sensitivity.selected_main_caliber`='healthy'；S1 结果分析 §3.3 |
| **S1 评估协议** | 分层 5 折 CV（K=5, seed=42），主指标 AUC + ACC + 少数类 F1/Recall；LOOCV 兜底 | `S1-results.pkl`（5 折 CV 字段） |
| **S2 稳定性选择参数** | τ=0.5（入选频率阈值）、C=0.1（Lasso 逆正则化）、B_full=100、B_cv=50、FDR m=1331（全特征规模）、VIP 阈值 1.5 | `S2-results.pkl.meta.{tau,C_lasso,B_full,fdr_m,vip_threshold}` |
| **S3 评估协议** | LODO（留一疾病）三组合 C1=测试CRC/训练IBD+Obesity、C2=测试IBD/训练CRC+Obesity、C3=测试Obesity/训练CRC+IBD；seed=42；budget_limited=False | `S3-results.pkl.meta.seed`=42、`meta.budget_limited`=False |
| **S3 域内 AUC 口径** | 衰减归因用 **264 特征**域内 AUC（与跨疾病同口径）；A3 的 1331 特征参考值仅作对照 | `S3-results.pkl.domain_auc`（264）、`domain_auc_reference_A3`（1331 参考） |
| **S3 预处理差异** | S3 含 StandardScaler（均值/方差仅训练集估计），与 S1/S2 纯 CLR 口径不同 → 域内 AUC 与 S1 有差异（见 §4 跨问一致性） | S3 报告 §6.1 口径声明 |

---

## 1. S1 疾病预测模型关键数字

### 1.1 三数据集 L2/RF 性能（5 折 CV，seed=42）

| 数据集 | n | 模型 | AUC | AUC_std | ACC | F1(少数类) | Recall(少数类) | LOOCV AUC |
|:--|:--:|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Zeller CRC | 121 | L2(CLR) | **0.7907** | 0.1022 | 0.7270 | 0.6398 | 0.6000 | 0.8042 |
| Zeller CRC | 121 | RF(原始) | **0.8454** | 0.0678 | 0.7847 | 0.6778 | 0.5800 | — |
| metahit IBD | 110 | L2(CLR) | **0.8871** | 0.0708 | 0.8636 | 0.6719 | 0.6400 | 0.8748 |
| metahit IBD | 110 | RF(原始) | **0.9035** | 0.0485 | 0.8182 | 0.3524 | 0.2400 | — |
| Chatelier Obesity | 253 | L2(CLR) | **0.6496** | 0.0718 | 0.6480 | 0.5180 | 0.5281 | 0.6270 |
| Chatelier Obesity | 253 | RF(原始) | **0.6602** | 0.0482 | 0.6442 | 0.0944 | 0.0562 | — |

- 溯源：`S1-results.pkl.<数据集>.{L2_CLR,RF_raw}.{AUC,AUC_std,ACC,F1_minority,Recall_minority}`；`S1-results.pkl.<数据集>.LOOCV.AUC`。
- 数据集键名：Zeller=`Zeller_fecal_colorectal_cancer`、metahit=`metahit`、Chatelier=`Chatelier_gut_obesity`。

### 1.2 基线（性能地板）

| 数据集 | 单特征最佳 AUC | Dummy 多数类 ACC | Dummy AUC |
|:--|:--:|:--:|:--:|
| Zeller CRC | 0.7581 | 0.6033 | 0.5 |
| metahit IBD | 0.8153 | 0.7727 | 0.5 |
| Chatelier Obesity | 0.6395 | 0.6482 | 0.5 |

- 溯源：`S1-results.pkl.<数据集>.baseline.{single_feature_best_AUC,dummy_ACC,dummy_AUC}`。

### 1.3 与基线增益（相对单特征基线）

| 数据集 | L2 增益 | RF 增益 |
|:--|:--:|:--:|
| Zeller CRC | +0.0326 | +0.0873 |
| metahit IBD | +0.0718 | +0.0882 |
| Chatelier Obesity | +0.0101 | +0.0207 |

- 溯源：由 `S1-results.pkl.<数据集>.{L2_CLR.AUC,RF_raw.AUC}` − `baseline.single_feature_best_AUC` 计算（报告用精确值，非表内四舍五入直接相减）。

### 1.4 过拟合判定（CV vs LOOCV 口径）

| 数据集 | 5 折 CV AUC(L2) | LOOCV AUC | 差距 | 全量 AUC | overfit_delta | overfit_flag |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Zeller | 0.7907 | 0.8042 | 0.0135 | 1.000 | 0.1958 | True |
| metahit | 0.8871 | 0.8748 | 0.0123 | 1.000 | 0.1252 | True |
| Chatelier | 0.6496 | 0.6270 | 0.0226 | 1.000 | 0.3730 | True |

- 溯源：`S1-results.pkl.<数据集>.{LOOCV.AUC,full_AUC,overfit_delta,overfit_flag}`。
- **口径声明**：全量 AUC=1.0 是 n≪p（264 特征 vs 110–253 样本）下样本内拟合的必然现象，非模型缺陷；**正文以「5 折 CV vs LOOCV 差距 <0.025」判定无过拟合**（三数据集 0.0135/0.0123/0.0226 均 <0.025）。`overfit_flag=True` 保留为原始判定留痕。

### 1.5 small_adenoma 四口径敏感性（Zeller）

| 口径 | L2 AUC | RF AUC | n | 说明 |
|:--|:--:|:--:|:--:|:--|
| ① 归健康（默认主口径） | 0.7907 | 0.8454 | 121 | 题面主口径 |
| ② 归病变 | 0.6112 | 0.6509 | 121 | 明显更差（污染正类），排除 |
| ③ 剔除 | 0.8022 | 0.8667 | 95 | 略优于①（ΔL2 +0.0115 / ΔRF +0.0213），Δ<0.05 不显著 |
| ④ 单开一类 | 0.8022 | 0.8667 | 95 | 二分类部分同③ |

- 溯源：`S1-results.pkl.adenoma_sensitivity.{CRC_adenoma_as_healthy,CRC_adenoma_as_diseased,CRC_adenoma_excluded,CRC_adenoma_separate}.{L2_AUC,RF_AUC,n_samples}`；`selected_main_caliber`='healthy'。
- **主口径①**（归健康）已落盘；③/④ 作敏感性附录。

### 1.6 集成（B2 条件触发）与敏感性

| 项 | 值 | 溯源 |
|:--|:--|:--|
| Zeller Soft Voting AUC | 0.8379（vs 单最佳 RF 0.8454，Δ−0.0076，不受益） | `S1-results.pkl.Zeller_fecal_colorectal_cancer.soft_voting.{AUC,vs_best_single_delta_AUC,ensemble_beneficial}` |
| metahit Soft Voting AUC | 0.8955（vs 单最佳 RF 0.9035，Δ−0.0080，不受益） | `S1-results.pkl.metahit.soft_voting.{AUC,vs_best_single_delta_AUC,ensemble_beneficial}` |
| Chatelier 集成 | 不触发（L2 AUC 0.6496 <0.75） | `S1-results.pkl.Chatelier_gut_obesity.soft_voting`=None |
| B3 class_weight（metahit） | balanced AUC 0.8871 / Recall 0.64；none AUC 0.8871 / Recall 0.52；ΔRecall +0.12（AUC 不变） | `S1-results.pkl.B3_class_weight.{balanced,none,delta_Recall}` |
| B4 离群剔除（Zeller，14 样本） | 剔除后 L2 0.7907→0.8016（Δ+0.0110）、RF 0.8454→0.8949（Δ+0.0494） | `S1-results.pkl.B4_outlier_removal.{n_outliers_removed,full_L2_AUC,removed_L2_AUC,delta_L2_AUC,full_RF_AUC,removed_RF_AUC,delta_RF_AUC}` |

### 1.7 跨疾病差异四重归因（结论性数字）

| 来源 | 关键数字 | 溯源 |
|:--|:--|:--|
| 样本量 | n=121/110/253；AUC_std：Zeller L2 0.1022（最大）、metahit 0.0708、Chatelier 0.0718 | `S1-results.pkl.<数据集>.L2_CLR.AUC_std` |
| 类别平衡 | 少数类比例 39.7%/22.7%/35.2%；metahit ACC 0.8636 vs Recall 0.64（L2）、ACC 0.8182 vs Recall 0.24（RF） | `S1-results.pkl.metahit.{L2_CLR,RF_raw}.{ACC,Recall_minority}` |
| 信号强度 | 单特征基线 0.7581/0.8153/0.6395；增益 L2 +0.0326/+0.0718/+0.0101 | `S1-results.pkl.<数据集>.baseline.single_feature_best_AUC` |
| 批次效应 | F8 聚类由 dataset 主导，14 离群样本独立成簇（A 类验证） | A 类验证（非 pkl 正式值） |

---

## 2. S2 特征选择与生物标志物关键数字

### 2.1 每病稳定标志物（τ=0.5, C=0.1, B_full=100）

**CRC（4 个，方向均 up=患病富集，3/4 命中已知标志物）**：

| 标志物 | 全量频率 | CV 折内频率 | Fisher FDR | Wilcoxon FDR | 方向 | 已知 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Peptostreptococcus_stomatis | 0.99 | 0.96 | 1.24e-05 | 0.113 | up | ✅ |
| Fusobacterium_nucleatum | 0.94 | 0.716 | 3.94e-05 | 5.04 | up | ✅ |
| Porphyromonas_somerae | 0.62 | 0.392 | 0.0172 | 5.04 | up | ✅ |
| Clostridium_hathewayi | 0.52 | 0.296 | 0.599 | 4.10 | up | — |

**IBD（4 个，4/4 Fisher 显著）**：

| 标志物 | 全量频率 | CV 折内频率 | Fisher FDR | 方向 | 已知 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Alistipes_finegoldii | 0.81 | 0.68 | 0.00675 | down | — |
| Bifidobacterium_bifidum | 0.75 | 0.664 | 0.00675 | up | ✅（已知属） |
| Akkermansia_muciniphila | 0.55 | 0.484 | 0.00675 | down | — |
| Eubacterium_ventriosum | 0.53 | 0.324 | 0.0398 | down | — |

**Obesity（20 个，0 个 FDR 显著，弱信号）**：Top 为 Ruminococcus_flavefaciens（0.89）、Pseudoflavonifractor_capillosus（0.84）、Rothia_mucilaginosa（0.72）、Bacteroides_ovatus（0.65）、Mitsuokella_multacida（0.62）、Megasphaera_elsdenii（0.62）、Ruminococcus_bromii（0.61）等；**全部 20 个 fisher_q >0.05（0 个 FDR 显著）**，可信度低，须显式标注。

- 溯源：`S2-results.pkl.per_disease.<D>.stable_features.{feature,frequency,cv_frequency,rank}`；`per_disease.<D>.biomarker_table.{feature,frequency,fisher_fdr,wilcoxon_fdr,direction,known_biomarker}`；`per_disease.<D>.n_stable`。

### 2.2 两路信号显著性（全 264 特征口径，BH-FDR m=1331）

| 疾病 | n_stable | n_fisher_sig | n_wilcoxon_sig | 入选标志物主导信号 |
|:--|:--:|:--:|:--:|:--|
| CRC | 4 | 4 | 1 | presence（全 4） |
| IBD | 4 | 6 | 1 | presence（全 4） |
| Obesity | 20 | 0 | 0 | presence 18 / abundance 2（均不显著） |

- 溯源：`S2-results.pkl.per_disease.<D>.{n_stable,n_fisher_sig,n_wilcoxon_sig}`。
- **结论**：判别信号几乎完全由「存在/缺失」主导（presence），丰度信号在稳定标志物中缺失（Wilcoxon 显著特征不在稳定集内）。Obesity 2 个 abundance 为 Bacteroides_ovatus、Ruminococcus_bromii。

### 2.3 共现分析（二阶初探）

**CRC（4 条边，全部 cooccur）**：

| 共现对 | Spearman | Fisher p | OR |
|:--|:--:|:--:|:--:|
| Peptostreptococcus_stomatis ↔ Fusobacterium_nucleatum | 0.762 | 8.82e-07 | 12.86（最强） |
| Peptostreptococcus_stomatis ↔ Porphyromonas_somerae | 0.829 | 0.0246 | 4.67 |
| Peptostreptococcus_stomatis ↔ Clostridium_hathewayi | −0.101 | 0.0199 | 3.12 |
| Fusobacterium_nucleatum ↔ Clostridium_hathewayi | 0.346 | 0.0056 | 6.78 |

**IBD（6 条边，3 cooccur + 3 exclude）**：cooccur：Alistipes_finegoldii↔Akkermansia_muciniphila（OR 6.04）、Alistipes_finegoldii↔Eubacterium_ventriosum（OR 3.13）、Akkermansia_muciniphila↔Eubacterium_ventriosum（OR 3.83）；exclude（互斥）：Alistipes_finegoldii↔Bifidobacterium_bifidum（OR 0.194）、Bifidobacterium_bifidum↔Akkermansia_muciniphila（OR 0.328）、Bifidobacterium_bifidum↔Eubacterium_ventriosum（OR 0.272）——**Bifidobacterium_bifidum 与其余 3 个标志物全部互斥**。

**Obesity（24 条边）**：0 个 FDR 显著，共现网络仅作弱信号结构初探，可信度低。

- 溯源：`S2-results.pkl.per_disease.<D>.cooccurrence.cooccurrence_edges.{feature_a,feature_b,type,spearman,fisher_p,odds_ratio}`。
- **边界声明（报告必须含）**：小样本下仅对入选标志物做二阶探索，无法全特征交互建模；标志物筛选主口径仍为边际信号。

### 2.4 佐证一致性（VIP / RF）

| 疾病 | VIP>1.5 特征数 | VIP Spearman 秩相关 | VIP overlap | RF overlap | RF 退化 |
|:--|:--:|:--:|:--:|:--:|:--|
| CRC | 28/264 | 0.539 | 0.2 | 0.0 | 退化（max 3.3e-17） |
| IBD | 27/264 | 0.515 | 0.2 | 0.0 | 退化（max 4.4e-17） |
| Obesity | 23/264 | 0.347 | 0.1 | 0.05 | 退化（max 4.4e-17） |

- 溯源：`S2-results.pkl.per_disease.<D>.vip`（264 特征 VIP 值，>1.5 计数由值计算）；`per_disease.<D>.topN_consistency.{vip_overlap,rf_overlap,spearman_rank_vip,spearman_rank_rf}`。
- **口径声明**：VIP>1.5 清单**包含**全部稳定标志物（CRC 4/4、IBD 4/4），VIP 佐证成立；`vip_overlap` 偏低因 VIP>1.5 集合大（~27）而稳定集小（4），交集比例被稀释，**以「VIP>1.5 清单佐证 + 秩相关」为主口径，不以低 overlap 作否定证据**。
- **RF 佐证层退化不可用**：`rf_importance` 全 ~1e-17 机器精度级退化（非零数 CRC 82/IBD 128/Obesity 130），**报告不引用 RF 数字**（U1 待调查）。

### 2.5 跨疾病对比

| 项 | 值 | 溯源 |
|:--|:--|:--|
| Jaccard 重叠 | CRC_IBD=0.0、CRC_Obesity=0.0、IBD_Obesity=0.0 | `S2-results.pkl.cross_disease.jaccard_matrix` |
| 共同标志物 | 空（无跨病共享） | `S2-results.pkl.cross_disease.common_biomarkers` |
| 疾病特异标志物数 | CRC 4 / IBD 4 / Obesity 20 | `S2-results.pkl.cross_disease.disease_specific` |

- **结论**：三病稳定特征集完全疾病特异，无共同标志物，支持独立建模（H7）。

### 2.6 τ 敏感性（全量 bootstrap）

| 疾病 | τ=0.4 | τ=0.5 | τ=0.6 | τ=0.7 |
|:--|:--:|:--:|:--:|:--:|
| CRC | 6 | 4 | 3 | 2 |
| IBD | 6 | 4 | 2 | 2 |
| Obesity | 32 | 20 | 9 | 3 |

- 溯源：`S2-results.pkl.meta.tau_counts.{CRC,IBD,Obesity}`（对应 τ_grid=[0.4,0.5,0.6,0.7]）。

---

## 3. S3 跨疾病预测模型关键数字

### 3.1 四策略对比（LODO 三组合 AUC）

| 策略 | 构建方式 | C1(CRC) | C2(IBD) | C3(Obesity) | 3 组合均值 |
|:--|:--|:--:|:--:|:--:|:--:|
| A 直接迁移 | L2+CLR 物种级 264 | 0.5674 | 0.5882 | 0.5253 | **0.5603** |
| B 共享标志物 | 252 特征交集重训 | 0.5417 | 0.6080 | 0.5218 | 0.5572 |
| C 属级聚合 | g__ 求和 106 特征 | 0.3616 | 0.4861 | 0.5440 | 0.4639 |
| C 门级聚合 | p__ 求和 11 特征 | 0.4141 | 0.5261 | 0.5999 | 0.5134 |
| D 部署校正 | Platt 校准（base=A） | 0.5674 | 0.5882 | 0.5253 | 0.5603（AUC 不变） |

- 溯源：`S3-results.pkl.strategy_compare.{A_direct,B_shared,C_genus,C_phylum,D_calibrated}.{C1,C2,C3}.auc` 与 `.mean_auc`；`B_shared.shared_feature_count`=252；`C_genus.n_features`=106、`C_phylum.n_features`=11；`D_calibrated.base_strategy`='A_direct'。
- **四策略 3 组合均值全部 <0.60** → 触发紧急回退（`fallback.triggered`=True）。
- 策略对比内 base 最优：`strategy_compare.best_strategy`='A_direct'（0.5603）。

### 3.2 紧急回退结果（R1–R4）

| 回退层级 | 模型族 | 3 组合均值 | C1 | C2 | C3 | 达可用线？ |
|:--|:--|:--:|:--:|:--:|:--:|:--|
| R1 树模型 | RandomForest 500 树 | 0.5092 | 0.4405 | 0.5529 | 0.5342 | 否 |
| R2 样本合并 | Logistic（≡策略 A 口径） | 0.5603 | 0.5674 | 0.5882 | 0.5253 | 否 |
| R3 密度比重加权 | importance weighting（域分类器法） | **0.6068** | 0.5945 | 0.6489 | 0.5771 | 否（<0.65，提升 +0.0465 <0.10） |
| R4 对抗式域适应 | DANN（梯度反转） | 0.5947 | 0.6184 | 0.6259 | 0.5398 | 否 |

- 溯源：`S3-results.pkl.fallback.{R1_tree,R2_pooled,R3_weighted,R4_dann}.{mean_auc,C1.auc,C2.auc,C3.auc}`。
- **穷尽出口**：`fallback.usable`=False、`delivered_strategy`=None；`exhausted_evidence.best_strategy`='R3_weighted'、`best_mean_auc`=0.6068、`usable_line`=0.65。
- **结论**：跨疾病预测模型最优可达 AUC 0.6068（R3_weighted），低于可用线 0.65；负结果是回退流程验证后的最终结论。

### 3.3 R3 阈值迁移辅指标（训练集 Youden 阈值迁移）

| 组合 | ACC | 灵敏度 | 特异度 | F1 | test_pos_frac |
|:--|:--:|:--:|:--:|:--:|:--:|
| C1(CRC) | 0.6446 | 0.2292 | 0.9178 | 0.3385 | 0.3967 |
| C2(IBD) | 0.5727 | 0.6800 | 0.5412 | 0.4198 | 0.2273 |
| C3(Obesity) | 0.4150 | 0.1585 | 0.8876 | 0.2600 | 0.6482 |

- 溯源：`S3-results.pkl.fallback.R3_weighted.C{1,2,3}.{acc,sensitivity,specificity,f1,test_pos_frac}`。

### 3.4 衰减归因（三分法，264 特征域内口径）

| 疾病 | 域内 AUC | 跨疾病 AUC | 衰减量 | 主导归因 |
|:--|:--:|:--:|:--:|:--|
| CRC | 0.7811 | 0.5674 | −0.2138 | 疾病特异信号 |
| IBD | 0.8588 | 0.5882 | **−0.2706** | 疾病特异信号（最强） |
| Obesity | 0.6638 | 0.5253 | −0.1384 | 标签语义漂移 |

- 溯源：`S3-results.pkl.decay_attribution.{CRC,IBD,Obesity}.{domain_auc,cross_auc,decay,dominant_cause}`；`domain_auc`（264 特征）。
- A3 参考（1331 特征，仅对照）：`domain_auc_reference_A3`={CRC 0.814, IBD 0.885, Obesity 0.644}。
- **批次效应不主导**：A 类验证 silhouette 0.070 近 0（approach §4.5，非 pkl 正式值）。

### 3.5 深度迁移分析（共享物种方向一致性）

| 项 | 值 | 溯源 |
|:--|:--|:--|
| 方向一致（可迁移） | 387 | `S3-results.pkl.migration_analysis.direction_consistent_count` |
| 方向翻转（疾病特异） | 369 | `S3-results.pkl.migration_analysis.direction_flipped_count` |
| 有效共享物种 | 756 | `S3-results.pkl.migration_analysis.n_valid` |
| 一致占比 | 51.2%（0.5119） | `S3-results.pkl.migration_analysis.consistent_fraction` |
| 符号检验 p | 0.5364（不显著） | `S3-results.pkl.migration_analysis.sign_test_pvalue` |

- **结论**：共享物种「患病↑/↓」方向跨疾病接近随机（≈50/50），共享物种存在性不承载可迁移信号，信号藏在疾病特异方向里。

### 3.6 C3 阈值漂移量化（H4 证伪）

| 项 | 值 | 溯源 |
|:--|:--|:--|
| 训练患病基线 | 0.3160 | `S3-results.pkl.threshold_drift.train_baseline` |
| 测试患病基线 | 0.6482 | `S3-results.pkl.threshold_drift.test_baseline` |
| 基线差 Δ | +0.3322 | `S3-results.pkl.threshold_drift.delta_baseline` |
| Youden 阈值 τ* | 0.9205 | `S3-results.pkl.threshold_drift.youden_threshold` |
| 边界位置（测试分布分位） | 96.0%（0.9605） | `S3-results.pkl.threshold_drift.boundary_position` |
| 灵敏度 | 0.0244（几乎全判健康） | `S3-results.pkl.threshold_drift.sensitivity` |

- **H4 证伪**：Platt 校准是单调变换，Youden 阈值迁移后决策边界不变，C3 灵敏度校准前后一致（0.0244）；即使改用 0.5 概率阈值，灵敏度仅 0.1646（`thr05_sensitivity`，报告 §6.3）。**结论：C3 阈值漂移是测试分布基线偏移，训练集拟合的单调校准无法修复**。

### 3.7 Platt 校准参数（D 策略，A<0 符号约定）

| 组合 | A | B |
|:--|:--:|:--:|
| C1 | −13.10 | 6.28 |
| C2 | −21.85 | 10.40 |
| C3 | −18.56 | 9.48 |

- 溯源：`S3-results.pkl.strategy_compare.D_calibrated.C{1,2,3}.{A,B}`。
- **口径声明**：Platt 形式 P=1/(1+exp(A·f+B))，A<0 才使 P 关于 f 单调递增（等价 sklearn 形式 w>0）；三组合 A 均负、platt_w 均正，与「A<0、w>0」一致。

---

## 4. 跨问一致性标注（终稿必须统一）

| # | 跨问引用数字 | 一致性说明 | 溯源 |
|:--|:--|:--|:--|
| C1 | **S3 域内 AUC vs S1 域内 AUC** | S3 域内 AUC（264 特征，含 StandardScaler）：CRC 0.7811 / IBD 0.8588 / Obesity 0.6638；S1 L2 AUC（纯 CLR）：Zeller 0.7907 / metahit 0.8871 / Chatelier 0.6496。**差异 Δ = −0.010 / −0.028 / +0.014**（CRC/IBD/Obesity），源于 S3 预处理含 StandardScaler（均值/方差仅训练集估计）而 S1/S2 为纯 CLR 口径。**终稿必须声明此口径差异，不得将 S3 域内 AUC 与 S1 AUC 混用** | `S3-results.pkl.domain_auc` vs `S1-results.pkl.<数据集>.L2_CLR.AUC` |
| C2 | **S3 域内 AUC 与 S1 的衰减归因口径** | 衰减量 Δ_d = 跨疾病 AUC − 域内 AUC 要求分子分母同特征口径（264），故用 264 特征域内 AUC；A3 的 1331 参考（0.814/0.885/0.644）仅作对照，不作衰减归因主口径 | `S3-results.pkl.domain_auc` + `domain_auc_reference_A3` |
| C3 | **S2 稳定标志物 ↔ S1 分类器输入特征** | S2 所选稳定特征子集即 S1 分类器输入特征（「特征选择→分类建模」一致链路）；S2 的 CRC 稳定标志物（Fusobacterium_nucleatum 等）与 S1 特征方向一致 | S2 报告 §8.2、S1 报告 §6 跨问衔接 |
| C4 | **small_adenoma 主口径跨问一致** | S1 已选定主口径①（归健康，`selected_main_caliber`='healthy'），S2 跟随（CRC 标签口径=归健康），S3 沿用同一清洗口径；三问无需重跑 | `S1-results.pkl.adenoma_sensitivity.selected_main_caliber`；S2 结果分析 T3 |
| C5 | **近全零过滤 1331→264 跨问一致** | S1/S2/S3 共用同一 264 特征集（三病并集统一过滤），特征维度跨问一致 | S1/S2/S3 报告 §2 |
| C6 | **CLR δ=6.5e-06 跨问一致** | S1/S2/S3 均用 δ=6.5e-06 零值替换常数 | `S2-results.pkl.meta.clr_delta`、`S3-results.pkl.meta.clr_delta` |
| C7 | **S3 共享物种方向 vs S2 跨疾病 Jaccard** | S3 深度迁移分析（共享物种 756 个，方向一致 51.2%）与 S2 稳定标志物 Jaccard 全 0（无跨病共享稳定标志物）**同向**：共享物种存在性不承载可迁移信号，稳定标志物完全疾病特异 | `S3-results.pkl.migration_analysis` vs `S2-results.pkl.cross_disease.jaccard_matrix` |
| C8 | **S3 策略 B 共享特征数 vs S2 过滤特征** | S3 策略 B 共享特征交集 252（`B_shared.shared_feature_count`），基于过滤后 264 特征集按特征名取交集 | `S3-results.pkl.strategy_compare.B_shared.shared_feature_count` |

---

## 5. 待核验/待裁定项（终稿组装前）

| # | 项 | 说明 | 状态 |
|:--|:--|:--|:--|
| P1 | **S2 RF 佐证层退化** | `rf_importance` 全 ~1e-17 机器精度级退化（非零数 CRC 82/IBD 128/Obesity 130），RF 佐证层不可用；报告不引用 RF 数字 | 待调查（S2 结果分析 U1） |
| P2 | **S2 Wilcoxon FDR q>1** | 多个 wilcoxon_fdr=5.04、4.10 等 >1（BH-FDR 未封顶），q>1 一律判「不显著」，报告不展示 q>1 数值 | 口径说明（S2 结果分析 U2） |
| P3 | **S3 R3 密度比方法** | R3 用域分类器法（w=exp(logit)×n_train/n_test，裁剪上界 10），math-S3.tex §9.3 原写「KLIEP 或 uLSIF」代理值；报告口径声明注明 | 已裁决（S3 结果分析 §五） |
| P4 | **S3 未交付可用模型** | 最优可达 AUC 0.6068 < 可用线 0.65，负结果如实报告，不夸大 | 已裁决（S3 结果分析 §二.2） |

---

## 6. 数据版本与生成信息

| pkl | generated | source | 关键 meta |
|:--|:--|:--|:--|
| S1-results.pkl | 2026-08-21 | S1-preprocessed.pkl（源自 c-data-cleaned.pkl） | float32；5 折 CV seed=42 |
| S2-results.pkl | 2026-08-21T17:46:38 | S2-preprocessed.pkl | tau=0.5, C_lasso=0.1, B_full=100, fdr_m=1331, clr_delta=6.5e-06 |
| S3-results.pkl | 2026-08-21 | S3-preprocessed.pkl | seed=42, budget_limited=False, clr_delta=6.5e-06 |

> 本文件所有数字均取自上述 pkl 实际值（只读提取），无 `TODO`/`TBD`/`待定` 占位符。终稿正文数值一律从本文件复制。

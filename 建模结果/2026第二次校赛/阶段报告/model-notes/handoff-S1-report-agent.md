---
handoff_type: report-agent
sub: S1
from: 建模
to: 报告
stage: "2.2"
source_docs:
  - result-analysis-S1.md
  - approach-S1-confirmed.md
  - math-S1.tex
  - handoff-S1-model-agent.md
status: done
next_action: 报告对话 2.3 内容段（补结论、口径声明与图表规格），说「继续」
---

# 交接：S1 建模 → 报告（2.2 结果分析完成）

> **handoff_type**: `report-agent`（建模 → 报告）
> **sub**: S1 疾病预测模型
> **日期**: 2026-08-21 | 运行模式: auto
> **上游**: `result-analysis-S1.md`（本交接的建模内容源）+ `S1-results.pkl`（数字唯一来源）

---

## 0. 必读清单已读汇报

已完整读取：`TRAE-建模.md`（2.2 结果分析规范）、`TRAE-规范.md`（A/B 相关节）、`result-analysis-S1.md`（本交接的建模内容源）、`approach-S1-confirmed.md`、`math-S1.tex`、`handoff-S1-model-agent.md`。报告对话需补读 `TRAE-报告.md` 与 `iter-02-sub1-disease-prediction.tex`（骨架段）。

---

## 1. 章节映射（建模内容 → 报告骨架）

| 报告骨架节 | 建模内容来源（result-analysis-S1.md） | 需补内容 |
|:--|:--|:--|
| §1 本问要解决什么 | §1 三数据集性能对比（总起） | 无新增 |
| §2 数据与预处理 | §0 数据口径 + §6 核对表 | 各数据集样本数/类别比例（pkl 无直接字段，从 approach §4.2 取：n=121/110/253，少数类 39.7%/22.7%/35.2%） |
| §3 模型构建 | §3.1 过拟合裁决（评估协议） | 无新增 |
| §4 结果 | §1 性能表 + §2 基线增益 + §3.3 adenoma 主口径 + §5 未闭合清单 | 特征重要性（pkl `coefficients`/`feature_importances`）、阈值分析（探索图） |
| §5 跨疾病差异归因分析 | §4 四重归因 | 无新增 |
| §6 结论与局限 | §5 未闭合清单 + §3 裁决 | 局限量级证据 |

---

## 2. 关键数字（来源可溯到 pkl，报告正文一律从本表复制，禁止随手填数）

### 2.1 三数据集性能表（pkl `<dataset>.L2_CLR / RF_raw / baseline`）

| 数据集 | n | 模型 | AUC | ACC | F1(少数类) | Recall(少数类) |
|:--|:--:|:--|:--:|:--:|:--:|:--:|
| Zeller CRC | 121 | L2(CLR) | 0.7907 | 0.727 | 0.6398 | 0.600 |
| Zeller CRC | 121 | RF(原始) | 0.8454 | 0.7847 | 0.6778 | 0.580 |
| metahit IBD | 110 | L2(CLR) | 0.8871 | 0.8636 | 0.6719 | 0.640 |
| metahit IBD | 110 | RF(原始) | 0.9035 | 0.8182 | 0.3524 | 0.240 |
| Chatelier Obesity | 253 | L2(CLR) | 0.6496 | 0.648 | 0.5180 | 0.5281 |
| Chatelier Obesity | 253 | RF(原始) | 0.6602 | 0.6442 | 0.0944 | 0.0562 |

### 2.2 基线（pkl `baseline` 字段）

| 数据集 | 单特征最佳 AUC | Dummy 多数类 ACC |
|:--|:--:|:--:|
| Zeller CRC | 0.7581 | 0.6033 |
| metahit IBD | 0.8153 | 0.7727 |
| Chatelier Obesity | 0.6395 | 0.6482 |

### 2.3 相对基线增益（pkl 计算值）

| 数据集 | L2 增益 | RF 增益 |
|:--|:--:|:--:|
| Zeller CRC | +0.0326 | +0.0873 |
| metahit IBD | +0.0718 | +0.0882 |
| Chatelier Obesity | +0.0101 | +0.0207 |

### 2.4 LOOCV 与过拟合（pkl `LOOCV.AUC` / `full_AUC` / `overfit_delta`）

| 数据集 | 5 折 CV AUC | LOOCV AUC | CV vs LOOCV 差距 | full_AUC | overfit_delta |
|:--|:--:|:--:|:--:|:--:|:--:|
| Zeller CRC | 0.7907 | 0.8042 | 0.0135 | 1.0 | 0.1958 |
| metahit IBD | 0.8871 | 0.8748 | 0.0123 | 1.0 | 0.1252 |
| Chatelier Obesity | 0.6496 | 0.6270 | 0.0226 | 1.0 | 0.3730 |

### 2.5 small_adenoma 四口径（pkl `adenoma_sensitivity`）

| 口径 | L2 AUC | RF AUC | n |
|:--|:--:|:--:|:--:|
| ① 归健康（主口径） | 0.7907 | 0.8454 | 121 |
| ② 归病变 | 0.6112 | 0.6509 | 121 |
| ③ 剔除 | 0.8022 | 0.8667 | 95 |
| ④ 单开一类 | 0.8022 | 0.8667 | 95 |

### 2.6 B 类验证（pkl `B3_class_weight` / `B4_outlier_removal` / `soft_voting`）

- **B3 class_weight**：metahit Recall 0.52→0.64（Δ+0.12），AUC 不变（0.8871）。**结论：class_weight 有效，保留**。
- **B4 离群样本**：剔除 14 样本后 Zeller L2 0.7907→0.8016（Δ+0.0110）、RF 0.8454→0.8949（Δ+0.0494）。**结论：小敏感性，RF 略受益，不改变主结论**。
- **B2 集成**：Zeller 触发但集成 0.8379 vs RF 0.8454（Δ-0.0076）；metahit 触发但集成 0.8955 vs RF 0.9035（Δ-0.0080）；Chatelier 不触发（L2 0.6496<0.75）。**结论：不做集成，单最佳（RF）即交付**。

---

## 3. 口径声明（报告必须写入）

1. **数据口径**：正式数字一律以 `S1-results.pkl`（float32，`c-data-cleaned.pkl` → `S1-preprocessed.pkl` 加载）为准；A 类验证（`B-raw.pkl`，float64）仅作参考，不采用。
2. **主指标**：AUC 为阈值无关主指标；F1/Recall(少数类) 为辅；ACC 仅参考（metahit 最不平衡，ACC 虚高）。
3. **过拟合口径**：采用「CV vs LOOCV」判定（差距 0.0135/0.0123/0.0226 <0.025，无过拟合）；**显式说明 full_AUC=1.0 是 n≪p（264 特征 vs 110-253 样本）下样本内自评的必然现象，非模型缺陷**。
4. **Chatelier RF F1/Recall 极低**：RF 无 class_weight + 默认阈值 0.5 对不平衡数据次优，AUC 0.6602 才是诚实指标；F1/Recall 低是已知局限，不作为性能结论。
5. **small_adenoma 主口径**：维持①（归健康），③/④ 入附录作敏感性；②（归病变）排除（AUC 掉 0.18）。
6. **跨数据集横向比**：只比「相对基线的增益」，不比绝对 AUC（批次效应强）。

---

## 4. 图表清单规格（数据源 pkl + 图名 + 论文位置）

> 正式图走独立出图流程（chart-generator 规范），本表为规格。探索图在 `outputs/figures/_explore/`，不进论文。

| # | 图名（正式） | 数据源（pkl 字段） | 论文位置 | 内容 |
|:--|:--|:--|:--|:--|
| 1 | 三数据集 ROC 曲线 | `<ds>.L2_CLR.oof_prob` / `<ds>.RF_raw.oof_prob` + 标签 | §4.1 结果 | 三数据集 L2+RF 的 ROC 曲线 + AUC 标注 |
| 2 | 三数据集性能对比柱状图 | §2.1 性能表 | §4.1 结果 | 各数据集 L2/RF 的 AUC 对比 |
| 3 | small_adenoma 四口径敏感性 | `adenoma_sensitivity` | §4.2 结果 | 四口径 L2/RF AUC 对比 |
| 4 | 特征重要性 Top10 | `<ds>.L2_CLR.coefficients` / `<ds>.RF_raw.feature_importances` | §4.5 结果 | L2 系数 Top10 + RF permutation importance Top10 |
| 5 | 阈值-指标曲线 | `<ds>.L2_CLR.oof_prob` | §4 结果（可选） | Precision/Recall/F1/ACC vs 阈值 |

---

## 5. AI 标注

- 本问建模内容（方案/推导/结果分析）由建模对话产出，报告正文引用时按 `ai-usage-report` skill 规范标注 AI 贡献标记（写入 `.trae/ai-markers/`）。
- 图表由代码/出图流程产出，正式图按 chart-generator 规范标注。
- 报告对话在 §6 结论与局限处声明：小样本 AUC 方差大、Chatelier 弱信号接近领域下界、批次效应限制横向比绝对 AUC。

---

## 6. 交接收尾

S1 2.2 结果分析完成，建模内容已落盘 `result-analysis-S1.md`，本交接提供章节映射、关键数字（来源可溯到 pkl）、口径声明、图表清单规格与 AI 标注。报告对话据此补 §4 结果、§5 归因、§6 结论与局限。

**next_action**: 报告对话 2.3 内容段（补结论、口径声明与图表规格），说「继续」。

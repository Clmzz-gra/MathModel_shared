---
handoff_type: model-agent
sub: S1
from: 代码
to: 建模
stage: "2.1.5"
source_docs:
  - handoff-S1-code-agent.md
  - approach-S1-confirmed.md
  - math-S1.tex
status: done
next_action: 建模对话 2.2 结果分析（门禁 2 人类确认后），说「继续」
---

# 交接：S1 代码 → 建模（2.1 正式模型实现完成）

> **handoff_type**: `model-agent`（代码实现回报，代码 → 建模）
> **sub**: S1 疾病预测模型
> **日期**: 2026-08-21 | 运行模式: auto
> **产物**: `outputs/data/S1-results.pkl` + 4 张探索图 + 两遍审核结论

---

## 0. 必读清单已读汇报

已完整读取：`TRAE-代码.md`（2.1 实现规范 + 代码审核规则）、`TRAE-规范.md`（C1 代码头注释 / C2 技术栈 / C4 高耗时脚本 / C8 代码加速决策树）、`handoff-S1-code-agent.md`（正式实现规格）、`approach-S1-confirmed.md`（方案确认书）、`math-S1.tex`（数学推导）、`proxy-replacement-checklist-S1.md`（代理值清单）。

---

## 1. 结果摘要（三数据集 L2/RF 关键数字，5 折 CV）

| 数据集 | 模型 | AUC | ACC | F1(少数类) | Recall(少数类) | LOOCV AUC |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| Zeller CRC (n=121) | L2(CLR) | **0.791** | 0.727 | 0.640 | 0.600 | 0.804 |
| Zeller CRC | RF(原始) | **0.845** | 0.785 | 0.678 | 0.580 | — |
| metahit IBD (n=110) | L2(CLR) | **0.887** | 0.864 | 0.672 | 0.640 | 0.875 |
| metahit IBD | RF(原始) | **0.904** | 0.818 | 0.352 | 0.240 | — |
| Chatelier Obesity (n=253) | L2(CLR) | **0.650** | 0.648 | 0.518 | 0.528 | 0.627 |
| Chatelier Obesity | RF(原始) | **0.660** | 0.644 | 0.094 | 0.056 | — |

- 基线（性能地板）：单特征最佳 AUC = 0.758 / 0.815 / 0.639；Dummy 多数类 ACC = 0.603 / 0.773 / 0.648。
- 与 A 类验证参考值（approach §4.1）量级一致：Zeller L2 0.81/RF 0.85、metahit L2 0.89/RF 0.88、Chatelier L2 0.64/RF 0.67。
- **Chatelier 弱信号确认**：L2 AUC 0.650 ≈ 单特征基线 0.639，多特征增益仅 +0.011，接近领域下界 0.65-0.75（诚实标注，不包装）。

---

## 2. small_adenoma 四口径敏感性（`adenoma_sensitivity` 字段）

| 口径 | L2 AUC | RF AUC | n | 说明 |
|:--|:--:|:--:|:--:|:--|
| ① 归健康（默认主口径） | 0.791 | 0.845 | 121 | 题面主口径 |
| ② 归病变 | **0.611** | **0.651** | 121 | ⚠️ 明显更差（small_adenoma 混入病变污染正类） |
| ③ 剔除 | **0.802** | **0.867** | 95 | 略优于①（ΔL2 +0.011 / ΔRF +0.022） |
| ④ 单开一类 | 0.802 | 0.867 | 95 | 二分类部分同③（26 例单独第三类，画像见 pkl） |

- **结论**：口径②（归病变）显著劣化（AUC 掉 0.18），应排除；口径③/④（剔除）略优于口径①，但 ΔAUC < 0.05（与 A 类 F6 一致），差异不显著。
- **默认主口径 = ①（healthy）已落盘**；最终主口径由建模/人类从结果择优（性能 + 可解释性），建议在 ① 与 ③ 之间权衡（③ 略优但样本量减至 95）。

---

## 3. B 类验证结论

| # | 项 | 结论 |
|:--|:--|:--|
| B2 | Soft Voting（条件触发） | Zeller 与 metahit 触发（两法 AUC 均≥0.75），但集成 AUC 均**不优于单最佳**：Zeller 0.838 vs RF 0.845（Δ-0.008）、metahit 0.896 vs RF 0.904（Δ-0.008）。Chatelier 不触发（L2 0.650<0.75）。**结论：不做集成，单最佳（RF）即交付** |
| B3 | metahit class_weight | class_weight='balanced' 使少数类 Recall 0.52→**0.64**（Δ+0.12），AUC 不变（0.887）。**结论：class_weight 有效，保留** |
| B4 | 14 离群样本剔除 | 剔除 14 样本（簇1）后 Zeller：L2 AUC 0.791→0.802（Δ+0.011）、RF AUC 0.845→**0.895**（Δ+0.049）。**结论：小敏感性，RF 略受益，不改变主结论** |

---

## 4. 探索图清单（`outputs/figures/_explore/`，不进论文）

| 图 | 内容 |
|:--|:--|
| `S1-roc-curve-explore.pdf` | 三数据集 ROC 曲线（L2+RF，OOF 概率）+ AUC |
| `S1-confusion-matrix-explore.pdf` | 三数据集 L2(CLR) 混淆矩阵热力图 |
| `S1-feature-importance-explore.pdf` | L2 系数 Top10 + RF permutation importance Top10（3×2 网格） |
| `S1-threshold-analysis-explore.pdf` | 阈值-指标曲线（L2 概率分类器，Precision/Recall/F1/ACC） |

---

## 5. 待裁定项（反向交接，需建模/人类裁决）

1. **过拟合判定规则失效（n≪p 必然）**：三数据集 `full_AUC`（样本内）均为 1.000，`overfit_delta = full_AUC - LOOCV_AUC` = 0.196 / 0.125 / 0.373，全部 >0.1 触发「过拟合」判定。但这是 n≪p（264 特征 vs 110-253 样本）下样本内 AUC 恒为 1.0 的必然现象，非模型缺陷。**更诚实的判定**：5 折 CV AUC vs LOOCV AUC 差距仅 0.013 / 0.012 / 0.023（<0.025），模型稳定无过拟合。建议建模在 2.2 采用「CV vs LOOCV」口径，或显式说明 full_AUC=1.0 的 n≪p 背景。
2. **Chatelier RF 少数类 F1/Recall 极低（0.094/0.056）**：RF 无 class_weight（handoff §1.4 规格），默认阈值 0.5 对不平衡数据（少数类=健康 35.2%）次优，RF 几乎全预测多数类（obesity）。AUC 0.660 才是诚实指标（阈值无关）。若需 RF 的 F1/Recall 有意义，需加 class_weight 或调阈值——但 handoff 未要求，暂按规格执行，待建模裁决是否补充。
3. **small_adenoma 主口径选择**：默认①（healthy）已落盘；③/④（剔除）略优但 Δ<0.05。最终主口径待建模/人类择优。

---

## 6. 口径修正说明

- **数据口径**：正式实现用 `c-data-cleaned.pkl`（float32）经 `S1-preprocessed.pkl`（1.4 预处理产物）加载，非 A 类验证的 `B-raw.pkl`（float64）。正式数字以本 pkl 为准。
- **L2 规格修正**：A 类验证 `utils.py` 的 `make_logistic` 用 StandardScaler + 无 class_weight；正式实现按 handoff §1.3 用**无 StandardScaler（CLR 已标准化）+ class_weight='balanced'**。这是规格差异，非 bug。
- **近全零过滤**：1331→264（零值占比>95% 剔除，三病并集统一口径），已在 1.4 预处理完成，2.1 直接加载 264 维。
- **代理值核销**：P1-P14 全部核销（δ=6.5e-06、C=1.0、K=5、seed=42、过滤阈值 0.95 等），无 @PROXY 残留。P6（C=1.0）未调参（默认起点，AUC 已达标，无需内层 CV 调参）。

---

## 7. 两遍审核结论

| 审核 | 结论 | 问题数 | 说明 |
|:--|:--|:--|:--|
| ① 原理合理性（建模子代理） | **通过** | 3 项 B 级 | 公式实现/口径/边界与 math/approach/handoff 高度一致，无 A 级阻断 |
| ② 代码逻辑（coding 子代理） | **通过** | 4 项 B 级 | 数据版本/折索引/少数类方向/混淆矩阵/keep_mask/B4 复现/并行度/C1 头注释/输出路径/代理值核销逐项正确 |

**B 级问题已全部修复（复审 diff）**：
1. LOOCV 过拟合判定 → 补 `overfit_flag` 布尔字段（`full_AUC - LOOCV > 0.1` 显式判定落盘）。
2. B2 集成增益判定 → 补 `ensemble_beneficial` 布尔字段（`delta > 0.02` 显式判定落盘）。
3. B4 缺防御断言 → 补 `assert n_out == 14`（复现验证通过，实际定位 14 个）。
4. C=1.0 不调参留痕 → 头注释「原理」字段补「未调参（AUC 已达标，无需内层 CV 调参）」。
5. B4 minority 硬编码 → 改为 `zeller["minority"]` 动态取。
6. 「与 profile-B.py 完全一致」表述夸大 → 改为「复现 kmeans_pp 实现，k=2 固定，非 K 扫描」。
7. argmin 定位「最小簇」vs「簇1」语义 → 补注释说明（最小簇=簇1=14 样本，argmin 更健壮）。

> 修复后已复跑，`S1-results.pkl` 重新落盘（新增 `overfit_flag`/`ensemble_beneficial` 字段），数值与修复前一致（修复均为非功能性健壮性/留痕改进）。

---

## 8. 交接收尾

S1 2.1 正式模型实现完成，`S1-results.pkl` 已落盘（meta 含 field_semantics），4 张探索图已产出，两遍审核已执行。待门禁 2 人类确认后进入 2.2 结果分析。

**next_action**: 建模对话 2.2 结果分析（门禁 2 人类确认后），说「继续」。

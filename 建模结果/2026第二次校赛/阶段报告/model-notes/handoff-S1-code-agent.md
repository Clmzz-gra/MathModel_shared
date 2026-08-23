---
handoff_type: code-agent
sub: S1
from: 建模
to: 代码
stage: "1.3"
source_docs:
  - approach-S1-confirmed.md
  - decision-tree-S1.md
  - debate-S1.md
status: ready
next_action: 代码对话 1.4 预处理（与建模 2.0 并行），说「继续」
---

# 交接：S1 建模 → 代码（正式实现）

> **handoff_type**: `code-agent`（正式实现交接，建模 → 代码）
> **sub**: S1 疾病预测模型
> **日期**: 2026-08-21 | 运行模式: auto（门禁 1 呈递人类拍板）
> **⚠️ 本文件为正式交接（无 `-verify` 后缀），与 A 类验证交接 `handoff-S1-code-agent-verify.md` 区分，禁止同名覆盖。**

---

## 0. 必读清单已读汇报

已完整读取：`TRAE.md`（管线骨架/交接协议/执行主体分工/门禁）、`TRAE-建模.md`（1.3 方案确认 + 方案讲解质量约束 6 条 + 主/次目标二选一）、`TRAE-规范.md`（C1 代码头注释 / C2 技术栈 / C3 GPU / C4 高耗时脚本 / C8 代码加速决策树）、`math-explainer` skill（讲解质量约束 6 条）、`modeling-decision-tree` skill（分类评估路径 + 检验要点）、`decision-tree-S1.md`、`debate-S1.md`、`handoff-S1-model-agent-verify.md`、`domain-knowledge.md`、`cluster-profile.md`。

---

## 1. 决策变量 / 规格

### 1.1 标签映射（三数据集统一：患病=1 / 健康=0）

| 数据集 | 患病（=1） | 健康（=0） | 少数类（F1/Recall 正类） |
|:--|:--|:--|:--|
| Zeller CRC | `cancer` | `n`（主口径）+ small_adenoma 四口径见下 | 患病（39.7%） |
| metahit IBD | `ibd_ulcerative_colitis` + `ibd_crohn_disease` | `n` | 患病（22.7%） |
| Chatelier Obesity | `obesity` | `leaness` | 健康（35.2%，方向特殊） |

> **small_adenoma 四口径敏感性分析（2026-08-21 人类裁定，全做择优）**：
> - 口径 ① 归健康（题面主口径，默认）：`n` + `small_adenoma` = 0
> - 口径 ② 归病变：`cancer` + `small_adenoma` = 1
> - 口径 ③ 剔除：Zeller 数据剔除 26 例 small_adenoma 后建模（121→95 样本）
> - 口径 ④ 单开一类：small_adenoma 不参与二分类（从 Zeller 训练/测试中排除，单独作为第三类报告其丰度画像）
> - 四种口径各跑一遍主模型（L2+CLR）与对照（RF），输出 AUC/ACC/F1 对比表（`S1-results.pkl` 增 `adenoma_sensitivity` 字段）；最终主口径由建模/人类从结果择优（性能 + 可解释性），未选定前默认口径 ① 落盘。

### 1.2 CLR 口径（仅主模型 L2 需要）

- **前置：近全零过滤（2026-08-21 人类裁定采纳，与 S2 口径统一）**：剔除零值占比 >95% 的特征（1067 个），保留 264 维；三病并集统一过滤（同一 264 特征集）。过滤规则实现见 `outputs/scratch/verify-S2-v2-zerobin.py`（A 类验证脚本，S2 产出）。
- 零值乘法替换：$x_{ij} \leftarrow \max(x_{ij}, \delta)$，$\delta = 0.65 \times 10^{-5} = 6.5\times10^{-6}$。
- 逐行几何均值中心化：$\mathrm{clr}(x_{ij}) = \ln x_{ij} - \frac{1}{p}\sum_{k=1}^{p}\ln x_{ik}$，$p=264$（过滤后维数）。
- 完整公式与符号定义见 `approach-S1-confirmed.md` §2.2。

### 1.3 Logistic(L2) 超参

- `LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000)`。
- C=1.0 为默认起点（λ=1.0）；若需调参，在 {0.01, 0.1, 1.0, 10, 100} 内层 CV 选择（见 proxy 清单 P6）。
- class_weight='balanced'：$w_c = n/(n_{\text{classes}} \times n_c)$，针对 metahit 少数类 Recall=0.400（F5）。

### 1.4 RF 超参（对照模型）

- `RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1, random_state=42)`。
- 输入原始丰度（含 92% 零值，免 CLR，**同样过近全零过滤 264 维**）；特征重要性用 permutation importance。

### 1.5 评估协议

- 分层 5 折 CV（`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`）。
- 主指标 AUC（阈值无关）+ ACC + F1(少数类) + Recall(少数类)。
- LOOCV 兜底：全量 AUC vs LOOCV AUC 差距 >0.1 判过拟合。
- 基线：单特征最佳阈值 + `DummyClassifier(strategy='most_frequent')`。

---

## 2. 数据接口

- **数据源**：`outputs/data/c-data-cleaned.pkl`（共享，阶段 0.3 清洗产出）。
  - 字段：2 元数据列（`dataset_name`、`disease`）+ 1331 物种级相对丰度特征列（float32）。
- **⚠️ 口径差异（必须注意）**：A 类验证（`handoff-S1-model-agent-verify.md`）用的是 `outputs/data/B-raw.pkl`（float64，484×1333）；**正式实现用 `c-data-cleaned.pkl`（float32）**。两者字段一致但 dtype 与清洗状态不同，正式实现以 c-data-cleaned.pkl 为准，A 类验证数字仅作参考，正式数字以 S1-results.pkl 落盘为准。
- 标签映射见 §1.1；CLR 口径见 §1.2。

---

## 3. 预期输出

### 3.1 `outputs/data/S1-results.pkl` 结构

```
{
  "<dataset>": {                    # Zeller / metahit / Chatelier
    "L2_CLR": {
      "AUC": float, "ACC": float, "F1_minority": float, "Recall_minority": float,
      "confusion_matrix": [[TN, FP], [FN, TP]],
      "cv_folds": [{"AUC": float, "ACC": float, ...} x5],   # 每折明细
      "coefficients": array(264),   # 特征系数（CLR 空间，过滤后维度）
      "intercept": float
    },
    "RF_raw": {
      "AUC": float, "ACC": float, "F1_minority": float, "Recall_minority": float,
      "confusion_matrix": [[TN, FP], [FN, TP]],
      "cv_folds": [...],
      "feature_importances": array(264)   # permutation importance（过滤后维度）
    },
    "baseline": {
      "single_feature_best_AUC": float,
      "dummy_ACC": float, "dummy_AUC": float
    },
    "LOOCV": {"AUC": float},         # 兜底诚实估计
    "soft_voting": {                 # 仅两基模型 AUC 均 ≥0.75 时输出（B2 人类裁定）
      "AUC": float, "ACC": float, "F1_minority": float, "Recall_minority": float,
      "vs_best_single_delta_AUC": float
    }
  },
  "adenoma_sensitivity": {           # small_adenoma 四口径（2026-08-21 人类裁定，全做）
    "CRC_adenoma_as_healthy":  {"L2_AUC": float, "RF_AUC": float},   # 口径① 默认主口径
    "CRC_adenoma_as_diseased": {"L2_AUC": float, "RF_AUC": float},   # 口径②
    "CRC_adenoma_excluded":    {"L2_AUC": float, "RF_AUC": float, "n_samples": int},  # 口径③
    "CRC_adenoma_separate":    {"L2_AUC": float, "RF_AUC": float, "adenoma_profile": {...}},  # 口径④
    "selected_main_caliber": "healthy|diseased|excluded|separate"   # 择优选定主口径
  }
}
```

### 3.2 探索图（`outputs/figures/_explore/`，不进论文）

- `S1-roc-curve-explore.pdf`：三数据集 ROC 曲线 + AUC。
- `S1-confusion-matrix-explore.pdf`：三数据集混淆矩阵热力图。
- `S1-feature-importance-explore.pdf`：L2 系数 Top 10 + RF importance Top 10。
- `S1-threshold-analysis-explore.pdf`：阈值-指标曲线（概率输出分类器）。

---

## 4. 参考实现

- A 类验证脚本：`outputs/scratch/verify-S1-a1.py` ~ `verify-S1-a6.py` + `utils.py`（CLR 函数、标签映射、评估协议均已实现，可复用）。
- 方案确认书：`solution/model-notes/approach-S1-confirmed.md`（数学框架 + 求解方法）。
- 代理值清单：`solution/model-notes/proxy-replacement-checklist-S1.md`。

---

## 5. 已知风险

| 风险 | 说明 | 处置 |
|:--|:--|:--|
| **Chatelier 弱信号** | AUC 0.643 ≈ 单特征基线 0.639，多特征增益仅 +0.004 | 诚实标注「接近领域下界 0.65-0.75」，不包装成「合理选择」 |
| **14 个 Zeller 离群样本** | 簇 1 = 14 个 Zeller 样本独立成簇（cancer:7/n:4/small_adenoma:3） | 两轨：纳入主口径 + 剔除敏感性（B4 验证） |
| **metahit class_weight** | 少数类 Recall=0.400，class_weight 提升幅度需对比 | B3 验证：head-to-head 对比 class_weight 前后 Recall |
| **小样本 AUC 方差** | metahit 110 样本（25 患病），AUC 标准误大 | LOOCV 兜底 + 报告置信区间 + 四重来源归因 |

---

## 6. B 类验证项（1.3 后由建模自写 verify 或并入 2.1）

| # | 项 | 验证方法（2026-08-21 人类裁定更新） |
|:--|:--|:--|
| B2 | 集成 Gamma（Soft Voting） | **条件触发**：若 L2(CLR) 与 RF 两方法 AUC 均 ≥ 0.75（「尚可」线）→ 做 Soft Voting（概率平均，L2+RF，可选加 PLS-DA）对比单最佳；否则不做（单最佳即交付） |
| B3 | metahit class_weight 提升 Recall | class_weight='balanced' vs None 的 Recall 对比（验证决定） |
| B4 | 14 离群样本剔除敏感性 | 剔除 14 样本重训 Zeller，对比 AUC（验证决定） |

---

## 7. 交接收尾

本文件为建模 → 代码正式交接（门禁 1 材料），规格已锁定，无 TBD/待补充。代码对话按 §1-§3 实现 1.4 预处理 + 2.1 正式模型。

**next_action**: 代码对话 1.4 预处理（与建模 2.0 并行），说「继续」。

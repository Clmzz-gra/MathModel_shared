# 交接：S1 A 类共享事实验证（建模 → 代码）

> **handoff_type**: `code-agent-verify`（A 类验证，由代码子代理执行）
> **sub**: S1 疾病预测模型
> **日期**: 2026-08-21 | 运行模式: auto（门禁 1 呈递人类拍板）
> **⚠️ 本文件为 `-verify` 后缀交接，与正式建模→代码交接 `handoff-S1-code-agent.md` 区分，禁止同名覆盖。**

---

## 0. 必读规范清单（先读后做）

> 代码子代理开工前必须完整读取以下文件，并按 TRAE-代码.md 规范执行；本交接只给任务参数与验证规格。

1. `E:\MathModel_pj\TRAE.md`（开头：管线骨架 / 三对话交接协议 / 执行主体分工）
2. `E:\MathModel_pj\TRAE-代码.md`（角色全量：A 类验证执行 / 轻量脚本 / 两遍审核 / 高耗时 C4）
3. `E:\MathModel_pj\TRAE-规范.md`（C1 代码头注释 / C2 技术栈 / C3 GPU / C4 高耗时脚本 / C8 代码加速决策树）
4. `solution/problem-statement.md`（S1 节）
5. `solution/model-notes/decision-tree-S1.md`（本问方案决策树，含 A 类验证清单 §5）

---

## 1. 交接背景

S1 需要三数据集各自二分类模型 + 跨疾病性能差异分析。建模侧已做方案决策树（见决策树 §2-3），推荐主模型 **Logistic(L2)+CLR 前置**，备选 PLS-DA / RF，基线单特征阈值+Dummy。A 类验证是**共享事实验证**（数据特征/约束可行性/基本假设），由代码子代理写代码跑实验，产出探索图 + 解读，回报 `handoff-S1-model-agent-verify.md` 给建模侧。

**A 类验证第一步必须是简单基线**（单特征阈值 / Dummy 建立性能下界，防过度设计）。

---

## 2. 数据接口

- **数据缓存**：`E:\MathModel_pj-2026-sim2-B-S1\outputs\data\B-raw.pkl`
  - ⚠️ 注意：`c-data-cleaned.pkl` 尚未就绪（0.3 清洗并行中）——本 A 类验证**基于 B-raw.pkl**，不依赖清洗产物。
- **结构**：`pandas.DataFrame`，484 行 × 1333 列（`pd.read_pickle` 加载）
  - **元数据列 2**：`dataset_name`（三数据集名）、`disease`（疾病标签）
  - **特征列 1331**：物种级相对丰度（列名 = 7 级分类学层级 `k__域|p__门|...|s__种`）
- **三数据集划分（按 dataset_name）**：

| dataset_name | 样本 | 患病标签（disease） | 健康 | 少数类 |
|:--|:--:|:--|:--|:--|
| `Zeller_fecal_colorectal_cancer` | 121 | `cancer`（48） | `n`(47) + `small_adenoma`(26) | 患病 39.7% |
| `metahit` | 110 | `ibd_ulcerative_colitis`(21) + `ibd_crohn_disease`(4) | `n`(85) | 患病 22.7% |
| `Chatelier_gut_obesity` | 253 | `obesity`(164) | `leaness`(89) | **健康 35.2%（方向特殊）** |

- **数据特征**：每行丰度和 ≈ 100（定和成分数据）；92.21% 零值；非零值 min=1e-05, median=0.0776, max=79.96。
- **标签口径**：`dataset_name + disease` 映射为三列二分类标签（患病=1 / 健康=0），按上表。

---

## 3. 验证规格（A 类共享事实，对应决策树 §5 清单）

> 每项验证用**只算不产、一次性、非正式**的轻量脚本 `verify-S1-*.py`（输出到 `outputs/scratch/`），不算正式模型代码。**A 类实验第一步必须是简单基线。**

| # | 验证项 | 目的 | 具体规格 |
|:--|:--|:--|:--|
| 1 | **简单基线下界** | 建立性能地板，防过度设计 | 每数据集：单特征最佳阈值 AUC（对所有特征扫最佳切分）+ Dummy(多数类) 的 ACC/AUC。输出：每数据集基线 AUC |
| 2 | **类别不平衡对 AUC vs ACC 影响** | 确认 AUC 主指标、ACC 是否误导 | 三数据集少数类比例下，比较「少数类 Recall / F1」与 ACC 的解读差异；验证 metahit(22.7%) 是否 ACC 虚高 |
| 3 | **零值 92% 对树 vs 线性模型影响** | 决定 CLR+替换 vs 树模型天然鲁棒 | 原始零值丰度直接喂 RF vs Logistic(L2)+CLR 的 AUC 对比；观察零值处理（乘法替换 vs 原样）是否显著改变性能 |
| 4 | **CLR 必要性** | 定和成分数据下线性模型是否必须 CLR | 原始丰度 vs CLR 变换（零值乘法替换 AL-007 后）进 Logistic(L2) 的 AUC + 特征偏相关对比 |
| 5 | **三数据集类内可分性 / 批次差异** | 为跨疾病性能差异分析铺垫 | PCA/t-SNE 降维可视化三数据集样本，观察簇结构与分离度；输出低维投影图 |
| 6 | **small_adenoma 敏感性** | 口径影响量化（registry B 级） | 剔除 26 例 small_adenoma 重训 Zeller CRC 模型，对比 AUC 差异 |

**技术要点**：
- 评估协议：小样本用**分层 K 折 CV（K=5~10）为主，LOOCV 兜底**；报告 AUC + F1(少数类) + Recall(少数类)，ACC 仅参考。
- 诚实标注：全量 AUC=乐观上界 vs CV AUC=诚实估计，两者差距 > 0.1 判过拟合。
- 少数类定义：按 §2 表，正类=各数据集少数类（obesity 数据集正类=健康）；F1/Recall 按少数类定义。
- 特征变换：CLR 前零值用乘法替换（AL-007，δ=0.65×检出限；检出限取非零最小值 1e-05 近似）。
- 脚本头部按 TRAE-规范.md C1 加头注释（含性能声明）；若某验证超 2 分钟 → 脚本交付后交主会话后台执行（C4）。

---

## 4. 待产探索图清单

> 探索图为**探索用**（非正式论文图），输出到 `outputs/figures/` 或 `outputs/scratch/`，随回报解读。正式图由 2.2 出图环节按 chart-generator 另行执行。

| 图 | 内容 | 用途 |
|:--|:--|:--|
| `explore-S1-baseline.png` | 三数据集简单基线 AUC（单特征阈值 + Dummy）柱状图 | 性能下界 |
| `explore-S1-imbalance.png` | 三数据集少数类比例 + ACC vs 少数类Recall 对比 | 不平衡影响 |
| `explore-S1-tree-vs-linear.png` | RF vs Logistic(L2)+CLR 在零值数据上的 AUC 对比 | 零值/CLR 影响 |
| `explore-S1-clr.png` | 原始 vs CLR 进 Logistic(L2) 的 AUC 对比 | CLR 必要性 |
| `explore-S1-pca-tsne.png` | 三数据集 PCA/t-SNE 低维投影（按 dataset 着色） | 类内可分性/批次差异 |
| `explore-S1-adenoma.png` | 剔除 small_adenoma 前后 CRC AUC 对比 | 口径敏感性 |

---

## 5. 回报要求

代码子代理执行完 A 类验证后，回报 `handoff-S1-model-agent-verify.md`（建模→代码 反向交接），包含：
1. 每项验证的探索图 + 数据解读（结论 + 证据路径）
2. 六项验证的量化结果摘要（AUC 等数字，保留三位有效数字）
3. 对决策树推荐的修正建议（若验证推翻某假设 → 标记）

---

## 6. 待裁定项

- **small_adenoma 口径 [B级]**：剔除敏感性结果（#6）用于销项；若差异显著（AUC 变化 > 0.05）→ 报告注明口径影响。登记见 `maintenance/registry.md`。

---

## 7. 交接收尾

验证脚本 + 探索图 + 回报文档均 commit 后，建模侧读取 `handoff-S1-model-agent-verify.md` 进入 1.2 方案辩论。

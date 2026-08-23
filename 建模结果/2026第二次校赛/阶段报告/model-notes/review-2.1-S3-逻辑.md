# S3 2.1 正式模型代码逻辑审查（两遍审核第②遍）

> 角色：coding 子代理 | 模型：deepseek-v4-pro:0813
> 审查对象：`outputs/scratch/S3-model.py`（S3 跨疾病预测模型 2.1 正式实现）
> 审查类型：代码逻辑（变量/索引/数据版本/输出路径/并行度/C1 性能声明/代理值核销）
> 审查日期：2026-08-21

---

## ① 已读清单汇报

已 Read 并遵守以下规范文件：

- `E:\MathModel_pj\TRAE.md`（管线骨架：核心规则/门禁/交接协议/代号/角色边界）
- `E:\MathModel_pj\TRAE-代码.md`（代码角色规范，重点「代码审核规则」）
- `E:\MathModel_pj\TRAE-规范.md`（A 执行规范 / C1 代码头注释 / C2 技术栈 / C4 高耗时 / C8 代码加速决策树）

已 Read 对照文档全文：

- `handoff-S3-code-agent.md`（§三 数据接口 / §四 预期输出结构）
- `proxy-replacement-checklist-S3.md`（代理值清单，逐项核销）
- `preprocess-S3.py`（1.4 预处理，CLR/聚合口径来源）
- `math-S3.tex`（Platt 符号约定 / R3 密度比方法 / 阈值迁移 / 防泄漏约束）

已核对实际 pkl 数据（数字只取 pkl/代码实际值，禁猜禁造）：

- `S3-preprocessed.pkl`：X_filtered(484×264)、y(484)、shared_features=252、genus_features=106、phylum_features=11、feature_names=264、lodo_combos C1(363/121)/C2(374/110)/C3(231/253)、meta.source=c-data-cleaned.pkl
- `S3-results.pkl`：strategy_compare 键、fallback 键、domain_auc、decay_attribution、migration_analysis、threshold_drift、best_strategy 均已核对

---

## ② 逐项结论（引用行号）

### 1. 数据版本 ✅ 通过

- 代码第 83 行 `DATA_PKL = ROOT / "outputs" / "data" / "S3-preprocessed.pkl"`，头注释第 41 行声明「源自 c-data-cleaned.pkl float32，非 B-raw.pkl」。
- 实际 pkl `meta.source = "outputs/data/c-data-cleaned.pkl"`，**未用 B-raw.pkl**。✅

### 2. 变量/索引 ✅ 通过

- **train_idx/test_idx 切分**：第 175-178 行 `Xtr = X_clr[train_idx]`、`Xte = X_clr[test_idx]`、`ytr = y[train_idx]`、`yte = y[test_idx]`，索引来自 `lodo_combos[c]["train_idx"]/["test_idx"]`（preprocess 预生成，0-based 行号）。✅
- **CLR 逐样本无泄漏**：`clr_transform`（第 102-110 行）逐样本 `log` 后减行均值，无跨样本参数；在切分前对整个数据集做 CLR（第 692 行）不引入泄漏（CLR 是逐行独立变换）。✅
- **StandardScaler 仅训练集 fit**：第 180 行 `scaler = StandardScaler().fit(Xtr)`，第 181-182 行分别 transform 训练/测试。✅ 与 math-S3.tex §3.4 一致。

### 3. 输出路径（S3-results.pkl 结构）⚠️ 基本一致，3 处结构偏差

实际 pkl 顶层键：`meta / strategy_compare / fallback / domain_auc / domain_auc_reference_A3 / decay_attribution / migration_analysis / migration_analysis_species_detail / threshold_drift / best_strategy`。

与 handoff §四 对照：

| handoff §四 字段 | 实现 | 结论 |
|---|---|---|
| strategy_compare.A_direct | ✅ 一致（C1/C2/C3 + mean_auc） | 通过 |
| strategy_compare.B_shared（含 shared_feature_count） | ⚠️ 缺 shared_feature_count | 偏差（见问题 1） |
| strategy_compare.C_hierarchy（level: genus\|phylum） | ⚠️ 拆为 C_genus / C_phylum 两键 | 偏差（见问题 2） |
| strategy_compare.D_calibrated | ✅ 一致（base_strategy + C1/C2/C3 + mean_auc） | 通过 |
| strategy_compare.best_strategy（交付模型选择） | ⚠️ 存 base 最优，非交付模型 | 偏差（见问题 3） |
| fallback（triggered/R1-R4/usable/delivered/exhausted） | ✅ 一致 | 通过 |
| domain_auc | ✅ 一致（另加 domain_auc_reference_A3） | 通过 |
| decay_attribution | ✅ 一致 | 通过 |
| migration_analysis | ✅ 一致（另加 n_valid/consistent_fraction/sign_test_pvalue） | 通过 |
| threshold_drift | ✅ 一致（另加 delta_baseline/youden_threshold/sensitivity） | 通过 |

### 4. 并行度 ✅ 通过（基本合规）

- **ProcessPoolExecutor 真实并行**：第 211 行 `ProcessPoolExecutor(max_workers=max_workers)`，`_fit_eval_worker` 为模块级函数（第 168 行，可 pickle），`if __name__ == "__main__"` 保护（第 918 行）。✅
- **RF n_jobs=1 避免嵌套并行**：第 163 行 `n_jobs=1`。✅
- **随机性隔离**：`SEED=42`（第 91 行），Logistic/RF 均 `random_state=SEED`（第 159/163 行），每组合独立拟合，无跨任务共享可变状态。✅
- **C8 单核红线**：每个策略内 3 组合经 ProcessPoolExecutor 并行，非单核串行。策略 A/B/C 之间串行调用（第 703-707 行），但整体 484×264 小样本、Logistic 毫秒级、头注释声明 <2 分钟，属轻量范畴，不触发单核红线。✅
- ⚠️ 头注释第 34 行声明 `max_workers=min(8, cpu)`，但代码第 208 行硬编码默认 `max_workers=8`，无 `min(8, cpu)` 逻辑（见问题 5）。

### 5. C1 性能声明 ✅ 通过

- 头注释第 33-38 行含「性能」字段，位于前 20 行内（第 33 行），声明了并行策略（ProcessPoolExecutor）、RF n_jobs=1、随机性隔离、轻量预计 <2 分钟。✅ 满足 Stop Hook 机械检查项。

### 6. 代理值核销 ✅ 全部核销

| # | 代理值项 | 代码落实 | 行号 | 结论 |
|---|---|---|---|---|
| 1 | 阈值迁移=训练集 Youden J（禁测试集重定） | `thr = youden_threshold(ytr, train_score)`，测试集 `y_pred = (test_score >= thr)` 不重定 | 191/193 | ✅ |
| 2 | Platt max_iter=2000 | `max_iter=2000` | 258 | ✅ |
| 3 | 重加权密度比估计方法+权重裁剪 | 域分类器法（Logistic 区分 train/test），`clip=10.0` | 317-335 | ✅（方法见问题 4） |
| 4 | seed=42 | `SEED = 42` | 91 | ✅ |
| 5 | δ=6.5e-6 | `CLR_DELTA = 0.65 * 1e-5` | 88-89 | ✅ |
| 6 | C=1.0 | `C=1.0` | 158 | ✅ |
| 7 | class_weight='balanced' | `class_weight="balanced"` | 159 | ✅ |

### 7. 代码头注释完整性 ✅ 通过

目的（第 2-7 行）/ 原理（第 9-31 行）/ 性能（第 33-38 行）/ 输入数据（第 40-44 行，含中文指标↔变量名映射）/ 输出（第 46-50 行）/ 对应论文章节（第 52-53 行）六字段齐全。✅

### 8. 数据命名规则 + meta.field_semantics ✅ 通过

- 数据命名：`S3-preprocessed.pkl` / `S3-results.pkl` 均带代号（子问题专属），共享基础数据 `c-data-cleaned.pkl` 不带代号。✅
- meta.field_semantics：第 854-864 行内嵌 9 个易错配字段语义（auc/sensitivity/decay/boundary_position 等），实际 pkl 已落盘。✅

---

## ③ 问题清单

| # | 问题 | 严重度 | 证据路径 |
|---|---|---|---|
| 1 | **B_shared 缺 shared_feature_count 字段**：`_clean()`（第 710-712 行）只保留 C1/C2/C3 键，丢弃 `run_strategy_B` 返回的 `shared_feature_count`（第 238 行）。handoff §四 要求 `B_shared: {..., shared_feature_count, mean_auc}`。实际 pkl `B_shared keys = ['C1','C2','C3','mean_auc']` | 中（待裁定项 B 级） | S3-model.py:710-712, 238；handoff §四 L113；S3-results.pkl 实测 |
| 2 | **C_hierarchy vs C_genus/C_phylum 结构命名**：handoff §四 写 `C_hierarchy: {..., level: "genus\|phylum", mean_auc}`，实现拆为 `C_genus` / `C_phylum` 两个独立键（第 717-718 行）。信息等价但键名不符 | 中（待裁定项 B 级） | S3-model.py:717-718；handoff §四 L114 |
| 3 | **strategy_compare.best_strategy 语义偏差**：handoff §四 写 `best_strategy: str # 交付模型选择`，实现第 875 行存 `best_base`（A/B/C 中 AUC 最优 base，实测='A_direct'），真正的交付/最优可达在顶层 `payload["best_strategy"]`（实测='R3_weighted'）。下游按 handoff 从 strategy_compare.best_strategy 取「交付模型」会取错 | 中（待裁定项 B 级） | S3-model.py:875, 887；handoff §四 L116；S3-results.pkl 实测 |
| 4 | **R3 密度比估计方法偏差**：math-S3.tex §9.3 写「密度比用 KLIEP 或 uLSIF 估计」，实现用域分类器法（Logistic 区分 train/test，`w=exp(logit)×(n_train/n_test)`，第 317-335 行）。checklist #3 标注「方法未定」，故为方法选择而非硬矛盾，但需下游确认域分类器法可接受 | 低（待裁定项 B 级） | S3-model.py:317-335；math-S3.tex L296；checklist #3 |
| 5 | **性能声明与代码不一致**：头注释第 34 行声明 `max_workers=min(8, cpu)`，代码第 208 行硬编码默认 `max_workers=8`，无 `min(8, cpu)` 逻辑。3 组合任务实际只用 3 worker，无实际影响 | 低 | S3-model.py:34, 208 |
| 6 | **R3 串行未并行（一致性）**：R1/R2 用 `_run_parallel`（ProcessPoolExecutor），R3 用 for 循环串行（第 341-351 行）直接调 `_fit_eval_worker`。R3 仅回退触发时运行且轻量，非单核红线，但与其他回退分支不一致 | 低 | S3-model.py:341-351 |
| 7 | **migration_analysis 用 CLR 空间做方向分析（口径）**：第 527-533 行用 `X_clr`（CLR 后）比较「患病 vs 健康」丰度方向。CLR 逐样本减行均值，不同样本减不同常数，跨样本丰度方向在 CLR 空间可能不完全等价于原始丰度空间。handoff/math 未明确要求用原始还是 CLR 空间 | 低（提示） | S3-model.py:527-533 |
| 8 | **策略 D 死代码**：第 281 行 `y_pred = (cal_test >= 0.5)` 计算后未使用，实际用 `y_pred_tau`（第 285 行，把训练 Youden 分数阈值映射到校准概率阈值）。逻辑正确，仅冗余变量 | 低（提示） | S3-model.py:281, 285 |
| 9 | **Platt w≤0 时 AUC 不变声明失效**：第 288 行 `"auc": m["auc"]` 假设 Platt 单调（w>0），但若 w≤0（第 278-279 行警告），cal_test 的 AUC 会不同于 base AUC，代码仍报 base AUC。异常分支，实际数据 w>0 通常成立 | 低（提示） | S3-model.py:278-279, 288 |

---

## ④ 结论

**通过（有条件）**。

核心逻辑审查全部通过：数据版本正确（S3-preprocessed.pkl 源自 c-data-cleaned.pkl，非 B-raw.pkl）、变量/索引切分正确、CLR 逐样本无泄漏、StandardScaler 仅训练集 fit、ProcessPoolExecutor 真实并行 + RF n_jobs=1 + random_state=42、C1 性能声明存在（前 20 行内）、代理值 7 项全部核销、代码头注释六字段齐全、数据命名规则（共享 vs 专属）正确、meta.field_semantics 内嵌易错配字段语义。

**条件**：3 处 handoff §四 预期输出结构与实现的结构偏差（问题 1/2/3）需下游（建模对话）裁定是否接受——均为字段命名/语义层面偏差，不影响数值正确性，可逆、修正代价低，建议按 B 级待裁定项登记，不阻断 2.1.5 门禁。问题 4（R3 密度比方法）建议一并确认。

**未发现**：数据泄漏、单核红线违反、代理值残留、C1 性能声明缺失等高严重度问题。

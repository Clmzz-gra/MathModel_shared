# S1 正式模型代码原理合理性审查（review-2.1-S1-原理）

> 审查角色：建模子代理（原理合理性审查）| 模型：deepseek-v4-pro:0813 | 思考强度：max
> 阶段：2.1.5 两遍审核（①原理合理性）| 日期：2026-08-21
> 待审代码：`outputs/scratch/S1-model.py`（完整 Read，554 行）
> 对照文档：`approach-S1-confirmed.md`、`math-S1.tex`、`handoff-S1-code-agent.md`、`proxy-replacement-checklist-S1.md`
> 补充核对（B4 复现一致性）：`outputs/scratch/preprocess-S1.py`（1.4 预处理产物生产者）、`outputs/scratch/profile-B.py`（0.4 画像）、`solution/model-notes/cluster-profile.md`

---

## ① 必读清单已读汇报

已完整读取并遵守：
- `TRAE.md`（管线骨架：核心规则/门禁/交接协议/代号/角色边界/执行主体分工）
- `TRAE-建模.md`（建模角色规范：方案讲解质量约束/门禁协议/审查裁决分离）
- `TRAE-规范.md`（A 执行规范 / B 产出格式 / C 项目强制规范 C1-C8）

---

## ② 逐项结论（对照审查聚焦点 1-9）

### 聚焦点 1：CLR 变换 —— ✅ 通过

- **结论**：代码**不重复做 CLR**，直接加载 `X_clr`（`S1-model.py` 第 380 行 `X_clr = d["X_clr"].astype(np.float64)`）。CLR 已在 1.4 预处理完成。
- **依据**：`preprocess-S1.py` 第 77-85 行 `clr_transform` 实现 `clr(x_ij)=ln(max(x_ij,δ))-mean_k(ln(max(x_ik,δ)))`，δ 定义于第 71 行 `DELTA = 0.65 * 1e-05 = 6.5e-06`（乘法替换 + 逐行几何均值中心化）。
- **证据路径**：`math-S1.tex` §2.2（`eq:zeroreplace` δ=6.5e-06、`eq:clr` 逐行几何均值中心化）；`S1-model.py` 第 380 行（直接加载，无重复 CLR）。

### 聚焦点 2：近全零过滤 —— ✅ 通过

- **结论**：代码直接加载 264 维（`X_raw`/`X_clr` 均 264 维），**不重复过滤**。
- **依据**：`preprocess-S1.py` 第 108-116 行，零值占比 >95% 剔除（`ZERO_RATIO_THRESHOLD=0.95`，第 72 行），`assert n_kept == 264`（第 114 行），三病并集统一口径（对全部 484 样本逐特征算零值占比）。
- **证据路径**：`handoff-S1-code-agent.md` §1.2；`math-S1.tex` §2.1（`eq:filter` τ=0.95、p=264）。

### 聚焦点 3：标签映射 —— ✅ 通过

- **结论**：代码直接加载 `y`（第 381 行），**不重复做映射**。
- **依据**：`preprocess-S1.py` 第 53-69 行 `DATASETS` 映射——Zeller(cancer=1, n+small_adenoma=0)、metahit(ibd_ulcerative_colitis+ibd_crohn_disease=1, n=0)、Chatelier(obesity=1, leaness=0, **minority=0** 方向特殊)。代码用 `pos_label=minority` 正确处理 Chatelier 少数类=健康。
- **证据路径**：`handoff-S1-code-agent.md` §1.1（标签映射表）。

### 聚焦点 4：L2 超参 —— ✅ 通过

- **结论**：`make_l2`（第 88-92 行）返回 `LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)`。**无 StandardScaler**（CLR 已标准化；`StandardScaler` 仅用于 B4 复现 0.4 画像，第 238 行）。
- **依据**：与 `handoff-S1-code-agent.md` §1.3 逐项一致。代码额外加 `random_state=42`（与 seed=42 一致，lbfgs 确定性下无副作用，合理）。
- **证据路径**：`handoff-S1-code-agent.md` §1.3；`approach-S1-confirmed.md` §3（求解方法表）。

### 聚焦点 5：RF 超参 —— ✅ 通过

- **结论**：`make_rf`（第 95-99 行）返回 `RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1, random_state=42, n_jobs=-1)`，输入 `X_raw`（第 386 行，免 CLR）。
- **依据**：与 `handoff-S1-code-agent.md` §1.4 一致。`n_jobs=-1` 为性能优化（sklearn 内部线程池并行，与 C8 决策树一致），不改变结果口径。
- **证据路径**：`handoff-S1-code-agent.md` §1.4；`math-S1.tex` §4（对照模型规格表）。

### 聚焦点 6：评估协议 —— ✅ 通过（附 B 级提示）

- **结论**：`cv_evaluate`（第 103-142 行）复用预处理折索引（`StratifiedKFold(5, shuffle, seed=42)`），AUC 主指标 + ACC + F1/Recall(少数类)；LOOCV 兜底 `overfit_delta = full_a - loo_auc`（第 427 行）。
- **依据**：与 `handoff-S1-code-agent.md` §1.5、`math-S1.tex` §5（`eq:cv` K=5 seed=42、`eq:loocv` Δ>0.1）一致。
- **B 级提示**：`>0.1 判过拟合` 的判定动作未在代码中显式实现（仅计算 delta 落盘），见问题清单 #1。

### 聚焦点 7：small_adenoma 四口径 —— ✅ 通过

- **结论**：遍历 `pre["adenoma_calibers"]`（第 459-487 行），各口径跑 L2+RF；口径④额外输出 26 例丰度画像（第 473-485 行）。
- **依据**：四口径在预处理定义（`preprocess-S1.py` 第 157-189 行）：①归健康②归病变③剔除④单开一类，与 `handoff-S1-code-agent.md` §1.1 四口径定义一致。
- **证据路径**：`handoff-S1-code-agent.md` §1.1（四口径表）。

### 聚焦点 8：B2/B3/B4 —— ✅ 通过（附 B 级提示）

- **B2 Soft Voting**：条件触发 `l2["AUC"]>=0.75 and rf["AUC"]>=0.75`（第 433 行），概率平均 `(l2["oof_prob"]+rf["oof_prob"])/2.0`（第 434 行）。与 `handoff-S1-code-agent.md` §6 B2 一致。
- **B3 class_weight 对比**：`make_l2(class_weight="balanced")` vs `make_l2(class_weight=None)` 对比 Recall（第 498-499 行）。与 §6 B3 一致。
- **B4 14 离群样本剔除**：`identify_zeller_outliers`（第 230-254 行）复现 0.4 画像——StandardScaler→SVD→`k_pca=searchsorted(cum_ratio,0.60)+1`（60% 方差→64 PC，与 `cluster-profile.md` 第 3 行「cumR²≥60%，64 PC」一致）→K-Means++(k=2,seed=42)。**与 `profile-B.py` 第 104-167 行逐行一致**（kmeans_pp 实现、seed=42、n_init=10、Lloyd 300 次迭代、inertia 最小选优）。与 §6 B4 一致。
- **B 级提示**：P13 增益阈值 0.02 未实现判定（见 #2）；B4 缺 `assert n_out==14` 复现验证（见 #3）。

### 聚焦点 9：代理值核销（P1-P14）—— ✅ 通过（附 B 级提示）

| 代理值 | 核销状态 | 证据 |
|:--|:--|:--|
| P1 δ=6.5e-06 / P2 检出限=1e-05 / P3 系数=0.65 | ✅ 已核销 | `preprocess-S1.py` 第 71 行 `DELTA=0.65*1e-05` |
| P4 K=5 | ✅ 已核销 | `preprocess-S1.py` 第 73 行；`S1-model.py` 第 519 行 `n_splits=5` |
| P5 seed=42 | ✅ 已核销 | `preprocess-S1.py` 第 74 行；`S1-model.py` 多处 `random_state=42` |
| P6 C=1.0 | ✅ 已核销（未调参，保持默认） | `S1-model.py` 第 88 行 `make_l2(C=1.0)` |
| P7 阈值=0.5 | ✅ 已核销 | `S1-model.py` 第 130 行 `oof_pred=(oof_prob>=0.5)` |
| P8 过拟合阈值=0.1 | ⚠️ delta 已落盘，判定未显式实现 | `S1-model.py` 第 427 行（见 #1） |
| P9 敏感性阈值=0.05 | ✅ 已销项（F6 已做，四口径全做） | 不涉及 |
| P10/P11/P12 RF 超参 | ✅ 已核销 | `S1-model.py` 第 95-99 行 |
| P13 集成增益阈值=0.02 | ⚠️ delta 已落盘，判定未显式实现 | `S1-model.py` 第 443 行（见 #2） |
| P14 过滤阈值=0.95 | ✅ 已核销 | `preprocess-S1.py` 第 72 行 |

- **无 `@PROXY` 字面标记残留**（代码中仅第 12 行注释引用「P6 代理值」，非未核销标记）。

---

## ③ 问题清单

| # | 问题 | 严重度 | 证据路径/节号 |
|:--|:--|:--|:--|
| 1 | LOOCV 过拟合判定（`Δ_overfit > 0.1`）未在代码中显式实现，仅计算 `overfit_delta` 落盘；代码头注释第 19 行声称「判过拟合」但无判定逻辑，判定需在 2.2 结果分析完成 | B 级（表述 vs 实现，delta 已正确计算，不影响数值正确性） | `S1-model.py` 第 19 行（注释）、第 427 行（只算 delta）；`handoff-S1-code-agent.md` §1.5；`math-S1.tex` §5.4 `eq:loocv` |
| 2 | B2 的 P13 集成增益阈值 0.02（增益<0.02 放弃集成）未实现判定，仅计算 `vs_best_single_delta_AUC` 落盘，判定需在结果分析完成 | B 级（同上） | `S1-model.py` 第 443 行；`proxy-replacement-checklist-S1.md` P13 |
| 3 | B4 复现 0.4 画像定位离群样本后，缺 `assert n_out == 14` 验证——若复现偏差导致定位到非 14 个样本，会静默产出错误结果（当前实现与 `profile-B.py` 逐行一致，预期定位 14 个，但缺防御性断言） | B 级（健壮性缺口，不影响当前正确性） | `S1-model.py` 第 511-512 行（`n_out=int(outlier_mask.sum())` 无断言）；`handoff-S1-code-agent.md` §6 B4 |

---

## ④ 结论

**通过**。

代码原理实现与数学推导（`math-S1.tex`）、方案确认书（`approach-S1-confirmed.md`）、正式实现规格（`handoff-S1-code-agent.md`）在公式实现、口径、边界上**高度一致**：CLR/近全零过滤/标签映射均在 1.4 预处理正确完成、2.1 代码正确加载不重复；L2/RF 超参、评估协议、四口径、B2/B3/B4 均忠实于规格；代理值 P1-P14 已核销，无 `@PROXY` 残留。

发现 3 个 **B 级**轻微问题（均为「判定动作留给结果分析」或「缺防御性断言」，不影响数值正确性），无 A 级阻断问题。建议在 2.2 结果分析阶段完成 #1/#2 的判定，并在 B4 处补 `assert n_out == 14` 后复审 diff。

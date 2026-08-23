# review-2.1-S3-原理.md — S3 2.1 正式模型代码原理合理性审查（两遍审核第①遍）

> 角色：建模子代理 | 模型：deepseek-v4-pro:0813
> 审查对象：`outputs/scratch/S3-model.py`（S3 跨疾病预测模型 2.1 正式实现）
> 审查类型：原理合理性（公式实现 / 口径 / 边界是否与数学推导一致）
> 日期：2026-08-21

---

## ① 已读清单汇报

- ✅ `E:\MathModel_pj\TRAE.md`（管线骨架：核心规则/门禁/交接协议/代号/角色边界/两遍审核）
- ✅ `E:\MathModel_pj\TRAE-建模.md`（建模角色规范：方案/推导/结果分析/审查代理职责）
- ✅ `E:\MathModel_pj\TRAE-规范.md`（A 执行规范 / B 产出格式 / C 项目强制规范）
- ✅ 对照文档：`approach-S3-confirmed.md`（§4 数学框架 / §5 求解方法）、`math-S3.tex`（§2-§11）、`handoff-S3-code-agent.md`（§2 规格 / §四 预期输出）
- ✅ 待审代码：`outputs/scratch/S3-model.py`（919 行）
- ✅ 数据实测：`outputs/data/S3-preprocessed.pkl`（LODO 组合 / 共享特征 / 属门维度 / 样本形状，数字只取 pkl 实际值）

---

## ② 逐项结论

### 1. LODO 三组合划分 — ✅ 正确

代码从 pkl 读 `lodo_combos`（`S3-model.py:686`），`COMBO_TO_DISEASE = {"C1":"CRC","C2":"IBD","C3":"Obesity"}`（`:94`）。pkl 实测：

| 组合 | 训练集（2 疾病） | 测试集（1 疾病） | 测试正类占比（实测） |
|---|---|---|---|
| C1 | IBD+Obesity（363） | CRC（121） | 48/121 = 0.397 |
| C2 | CRC+Obesity（374） | IBD（110） | 25/110 = 0.227 |
| C3 | CRC+IBD（231） | Obesity（253） | 164/253 = 0.648 |

与 `approach §4.1` / `math-S3.tex §2.1` 表完全一致（40%/23%/65%）。测试疾病在训练阶段完全不可见（train_idx 不含测试疾病样本，pkl 实测 train diseases 与 test disease 无交集）。

### 2. 模型规格 — ✅ 正确

`_make_model("logistic")`（`:156-160`）：`LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, class_weight="balanced", random_state=42)`，与 `handoff §2.2` 一致（`solver="lbfgs"` 为 L2 默认，未偏离）。CLR δ：`DETECTION_LIMIT=1e-5`、`CLR_DELTA=0.65*1e-5=6.5e-6`（`:88-89`），与 `math §3.3` 一致。StandardScaler 仅训练集估计：`_fit_eval_worker` 中 `scaler=StandardScaler().fit(Xtr)` 后 `transform(Xtr/Xte)`（`:180-182`），符合 `math §3.4` 防泄漏约束。过滤 1331→264 在预处理完成（pkl 实测 `X_filtered` 484×264）。

### 3. Youden 阈值迁移（禁测试集重定）— ✅ 正确（硬约束满足）

`_fit_eval_worker`（`:191-193`）：`thr = youden_threshold(ytr, train_score)` 仅用训练集标签与训练分数估计；`y_pred = (test_score >= thr)` 原样搬测试集。全代码无任何测试集重定阈值路径。`youden_threshold`（`:133-138`）用 `roc_curve` 的 `tpr-fpr` 取 argmax，是标准 Youden J 实现，符合 `math §8.1`。

### 4. 策略 B 共享特征交集 — ✅ 正确

`X_clr_shared = clr_transform(X_filtered[shared_features].to_numpy())`（`:693`），`shared_features` 来自 pkl（实测 252 个，`meta.field_semantics` 注明「过滤后 264 特征中三病交集（存在性=丰度>0）」）。仅用特征存在性（三病交集），无测试标签参与，符合 `handoff §2.4` / `approach §1.3` B5 转导式边界。`run_strategy_B` 记录 `shared_feature_count=252`（`:238`）。

### 5. 策略 C 分类学聚合 — ✅ 正确

`taxonomy_aggregate`（`:113-130`）：按 `g__`/`p__` 段分组，`X.T.groupby(key).sum().T` 同层丰度求和，符合 `math §6.1` 的 $x_g=\sum_{j\in g}x_j$。聚合在 CLR 之前（`:694-697` 先聚合后 `clr_transform`），符合「聚合后 CLR 再重训」口径。pkl 实测属 106 / 门 11，与代码注释一致。

### 6. 策略 D Platt 校准 — ⚠️ 代码正确，但规格存在符号错误（待裁定项）

**核心发现（本审查重点）**：`math-S3.tex §7.2`（`:193`）声称「当 $A>0$ 时，$P_{\text{cal}}$ 是 $f$ 的严格单调递增函数」，`§7.3`（`:203`）与 `approach §4.4`（`:138,140`）、`handoff §2.6`（`:72`）均要求「校验 $A>0$」。

**数学事实**：对 $P=1/(1+\exp(A\cdot f+B))$，$\frac{dP}{df}=-\frac{A\exp(Af+B)}{(1+\exp(Af+B))^2}$，符号为 $-\text{sign}(A)$。故 **$A>0$ 实为单调递减，$A<0$ 才单调递增**。规格的「$A>0$ 单调递增」与「校验 $A>0$ 防排序反转」是**符号错误**——按规格字面执行会强制单调递减（排序反转）。

**代码处理（正确）**：`platt_calibrate`（`:252-265`）用 sklearn 形式 $P=1/(1+\exp(-(w\cdot f+b)))$，故 `A=-w, B=-b`（`:262-263`），并校验 `w>0`（`:278-279`，等价 $A<0$，即分数越高概率越高）。代码头注释（`:22-23`）与 meta（`:851-852`）均显式标注「与 math-S3.tex 的 A>0 符号约定相反，见口径修正」。**代码正确规避了规格符号错误**，但属「handoff 规格与实现矛盾」→ 记待裁定项（见问题清单 #1）。

阈值迁移逻辑正确：`run_strategy_D`（`:280-285`）把训练 Youden 分数阈值 $\tau^*$ 映射到校准概率阈值 $\tau_{\text{prob}}=\sigma(A\tau^*+B)$，再判测试。因 Platt 单调，$\tau_{\text{prob}}$ 恰为概率尺度上的 Youden 最优阈值，等价于在校准概率上重估，符合 `approach §1.5`「迁移 Youden 阈值」。

### 7. 回退 R1-R4 — ✅ 基本正确（若干小偏差）

- **R1**（`:305-308`）：`RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", n_jobs=1)`。`n_estimators=500` 符合 `handoff §2.7`；`class_weight="balanced"` 为规格未指定的补充（与 Logistic 平衡口径一致，合理）；`n_jobs=1` 防嵌套并行（合理）。
- **R2**（`:311-314`）：`_logistic_tasks(..., model_type="logistic")` ≡ 策略 A 口径。**注意**：LODO 协议本身已合并 2 训练疾病，故 R2「样本合并」与策略 A 数学上恒等，回退链中无法提供增量（见问题清单 #3）。
- **R3**（`:317-352`）：密度比 $w(x)=p_{\text{test}}(x)/p_{\text{train}}(x)$ 用域分类器估计，`w = exp(logit) * (n_train/n_test)`（`:331`）正确补了先验比修正，裁剪上界 10（`:332`）。转导式边界正确：域分类器只用 train/test 特征（`X_clr`），域标签为 0/1（非疾病标签），绝不用测试疾病标签，符合 `math §9.2` / `handoff §2.7`。
- **R4**（`:358-451`）：DANN 结构（特征提取器 + 标签分类器 + 梯度反转域判别器）正确；标签损失只用训练标签（`ytr_t`），域损失用 train/test 特征 + 域标签，无测试标签泄漏。小偏差：`alpha` 固定 1.0 无退火、无独立验证集（见问题清单 #5）。

### 8. 衰减归因三分法 — ✅ 基本一致（操作化阈值与域内口径有偏差）

`decay_attribution`（`:479-499`）：`decay = cross_auc - domain_auc`（跨疾病 − 域内），符号与 `math §8.2` 的 $\Delta_d=\text{AUC}_{\text{cross}}-\text{AUC}_{\text{domain}}$ 一致（IBD 得 −0.358）。归因判定：`sens<0.10 → 标签语义漂移`、`|decay|≥0.20 → 疾病特异信号`、否则「疾病特异信号（弱）」；批次效应全局排除（silhouette 0.070 近 0，符合 `approach §4.5` F5）。判定结果可复现 `approach §6.2` 表（CRC/IBD→疾病特异信号，Obesity→标签语义漂移）。偏差：① 操作化阈值 0.10/0.20 为代码自定（规格未给具体阈值）；② 域内 AUC 用 264 特征重算（`domain_auc_264`，`:457-476`），而非 `handoff §2.8` 的 A3 参考（1331 特征 0.814/0.885/0.644），衰减量数值与 `approach §6.2` 表不同（见问题清单 #2）。

### 9. 深度迁移分析方向一致性 + 符号检验 — ✅ 正确

`migration_analysis`（`:502-558`）：对每个共享物种，训练疾病方向 `sign(mean_diseased - mean_healthy)`、测试疾病方向同式，一致/翻转/零三类，配二项符号检验 `binomtest(consistent, n_valid, 0.5)`（`:547`）。与 `approach §1.1`「方向一致/方向翻转 + 符号检验」一致。小偏差：方向在 CLR 空间计算（非原始丰度），见问题清单 #8。

### 10. C3 阈值漂移量化 — ✅ 正确

`threshold_drift`（`:561-591`）：`delta_baseline = test_baseline - train_baseline`（`:571`）即 `math §10.4` 的 $\Delta_{\text{baseline}}=\hat{P}_{\text{test}}(y=1)-\hat{P}_{\text{train}}(y=1)$；`boundary_position = (test_score < thr).mean()`（`:575`）即 Youden 阈值在测试分数分布的位置（判健康占比）。与 `approach §6.4` 一致。

---

## ③ 问题清单

| # | 问题 | 严重度 | 证据路径 |
|---|---|---|---|
| 1 | **规格符号错误（待裁定项）**：`math-S3.tex §7.2/§7.3`、`approach §4.4`、`handoff §2.6` 的「$A>0$ 单调递增 / 校验 $A>0$」与公式 $P=1/(1+\exp(Af+B))$ 不自洽——该公式下 $A>0$ 单调递减、$A<0$ 才单调递增。代码正确实现 $A=-w$、校验 $w>0$（≡$A<0$）并标注「口径修正」，但属规格与实现矛盾，需上游修正规格 | 中（规格错误，代码已正确规避，不影响代码正确性） | `math-S3.tex:193,203`；`approach-S3-confirmed.md:138,140`；`handoff-S3-code-agent.md:72`；`S3-model.py:262-263,278-279` |
| 2 | **域内 AUC 口径偏差（待裁定项）**：代码在 264 特征重算域内 AUC（`domain_auc_264`），而 `handoff §2.8` / `approach §6.2` 用 A3 参考（1331 特征 0.814/0.885/0.644）。衰减量 $\Delta$ 因此与 `approach §6.2` 表数值不同。代码已文档化（meta field_semantics），且 264 口径与跨疾病 264 特征「同口径对比」更严谨，但偏离 handoff 显式参考值 | 低-中（口径偏差，已文档化，需确认采用哪一口径） | `S3-model.py:457-476,821`；`handoff-S3-code-agent.md:87`；`approach-S3-confirmed.md:200-202` |
| 3 | **R2 样本合并 ≡ 策略 A（规格冗余）**：LODO 协议本身已合并 2 训练疾病，R2「样本合并」与策略 A 数学恒等，回退链中 R2 恒等于 A（A<0.60 时 R2 也 <0.60，永达不到 0.65 或 +0.10）。代码忠实实现（注释「≡策略 A 口径」），但规格层面 R2 无法提供增量 | 低（规格冗余，非代码错误） | `S3-model.py:311-314`；`handoff-S3-code-agent.md:78`；`approach-S3-confirmed.md:65` |
| 4 | **R1 RF 加 class_weight='balanced'**：`handoff §2.7` 未指定 RF 的 class_weight，代码补充了 balanced（与 Logistic 口径一致，合理），属轻微规格偏离 | 低 | `S3-model.py:162-164`；`handoff-S3-code-agent.md:77` |
| 5 | **R4 DANN 无独立验证集 + alpha 固定 1.0**：`handoff §2.7` 要求「严格验证集防泄漏」，代码固定 200 epoch 无早停/验证集，alpha 无退火。测试集仅用于域判别（转导式，无标签）与最终评估，无标签泄漏，但缺验证集 | 低 | `S3-model.py:358-451`；`handoff-S3-code-agent.md:80` |
| 6 | **pkl 结构命名偏差**：`handoff §四` 期望 `C_hierarchy`（单键 + `level` 字段），代码产出 `C_genus`/`C_phylum` 两键；`strategy_compare.best_strategy` 存 `best_base`（最优 base 策略名）而非「交付模型」（交付模型在顶层 `best_strategy` 字段）。信息未丢失，但字段语义与 handoff 预期有出入 | 低 | `S3-model.py:714-719,875`；`handoff-S3-code-agent.md:114-116` |
| 7 | **衰减归因操作化阈值自定**：`sens<0.10`、`|decay|≥0.20` 为代码自定阈值（规格未给具体值）；且判定优先级 sens 先于 decay，批次效应被全局排除后实际退化为二分类（标签漂移 vs 疾病特异），与 `approach §4.5`「三互斥来源」的并列表述略有出入 | 低 | `S3-model.py:490-497`；`approach-S3-confirmed.md:148-152` |
| 8 | **迁移分析方向在 CLR 空间计算**：`approach §1.1` 说「丰度方向」，代码在 CLR 变换后计算方向（`X_clr`）。CLR 与原始丰度方向大体一致（CLR 逐样本中心化不改变单物种相对高低），但严格说非同一口径 | 低 | `S3-model.py:527-533`；`approach-S3-confirmed.md:25` |
| 9 | **死代码**：`run_strategy_D` 中 `y_pred = (cal_test >= 0.5)`（`:281`）计算后未使用（实际用 `y_pred_tau`） | 极低 | `S3-model.py:281` |

---

## ④ 结论

**通过（附待裁定项）。**

代码的**原理实现与数学推导一致**：LODO 三组合划分、模型规格（L2/C=1.0/balanced/max_iter=2000/CLR δ=6.5e-6/StandardScaler 仅训练集）、Youden 阈值迁移禁测试集重定（硬约束满足）、策略 B 共享特征交集（252，仅存在性无标签泄漏）、策略 C 分类学聚合（属/门求和）、策略 D Platt 校准（A=-w 处理正确）、回退 R1-R4、衰减归因、深度迁移分析、C3 阈值漂移——10 项检查全部通过或基本通过。

**唯一需上游裁定的实质问题**是 #1：`math-S3.tex §7.2/§7.3`、`approach §4.4`、`handoff §2.6` 的「$A>0$ 单调递增 / 校验 $A>0$」为**规格符号错误**（该公式下 $A>0$ 单调递减、$A<0$ 才单调递增）。代码已正确实现 $A=-w$、校验 $w>0$（≡$A<0$）并显式标注「口径修正」，**代码无需修改**，但需上游（建模对话）修正三处规格的符号约定（$A>0$ → $A<0$，或改公式为 $P=1/(1+\exp(-(Af+B)))$）。

其余问题（#2-#9）均为低严重度的口径/命名/冗余偏差，不阻断放行，建议随待裁定项一并登记。

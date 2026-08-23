# S2 2.1 正式模型代码原理合理性审查（第一遍审核）

> 审查代理角色：建模子代理（原理合理性审查）
> 模型：deepseek-v4-pro:0813
> 审查对象：`outputs/scratch/S2-model.py`（488 行，全文已读）
> 审查类型：原理合理性（公式实现 / 口径 / 边界 vs 数学推导）
> 对照文档：`approach-S2-confirmed.md`、`math-S2.tex`、`handoff-S2-code-agent.md`、`proxy-replacement-checklist-S2.md`
> 日期：2026-08-21

---

## ① 已读清单汇报

开工前已 Read 并遵守以下规范文件：

- ✅ `E:\MathModel_pj\TRAE.md`（管线骨架：核心规则 / 门禁 / 交接协议 / 代号 / 角色边界 / 两遍审核）
- ✅ `E:\MathModel_pj\TRAE-建模.md`（建模角色规范：方案讲解质量约束、2.0 推导、门禁协议）
- ✅ `E:\MathModel_pj\TRAE-规范.md`（A 执行规范、B 产出格式、C1 代码头注释、C8 加速决策树）

对照文档全文已读：`approach-S2-confirmed.md`（§1 最终方案、§3 数学框架）、`math-S2.tex`（§2 预处理、§3 主方法、§4 两路信号、§5 共现、§6 佐证）、`handoff-S2-code-agent.md`（§1 规格）、`proxy-replacement-checklist-S2.md`（P1–P8）。

---

## ② 逐项结论（8 个聚焦点）

### 聚焦点 1：CLR 变换口径 —— ✅ 通过

- **结论**：正确。2.1 直接复用 `X_clr`（来自 `S2-preprocessed.pkl`），**不重复变换**，符合审查要求「CLR 已在 1.4 预处理完成」。
- **依据**：代码 `main()` 中 `X_clr = dd["X_clr"]`（L279）直接取用，全文件无任何 CLR 重算逻辑；`meta["clr_delta"] = 6.5e-06`（L438）与 math §2.2 `\delta=6.5\times10^{-6}`（math L97）及 P6 一致。
- **证据路径**：`S2-model.py` L279、L438；`math-S2.tex` §2.2（L92–117）；`proxy-replacement-checklist-S2.md` P6。

### 聚焦点 2：Lasso 稀疏 Logistic —— ❌ 不通过（A 级阻断）

- **结论**：**严重错误**。代码 L130 为 `LogisticRegression(solver="liblinear", C=C, max_iter=2000, random_state=seed, l1_ratio=1.0)`，**缺少 `penalty` 参数**，penalty 默认取 `'l2'`；而 `l1_ratio` 仅在 `penalty='elasticnet'` 时生效，故 `l1_ratio=1.0` 被**静默忽略**，实际拟合的是 **L2 正则 Logistic，而非 L1（Lasso）**。
- **后果**：L2 只收缩不置零，`np.abs(l1.coef_[0]) > 1e-8`（L132）对几乎所有 264 特征恒为 True，导致每轮 bootstrap 全部特征「入选」，$\hat{\pi}_j\approx 1.0$，稳定性选择**完全失效**（所有特征频率≈1，无法区分稳定簇与噪声长尾，直接推翻 H5 与 §3.3 的整个方法）。
- **正确写法**：`LogisticRegression(penalty='elasticnet', l1_ratio=1.0, solver='saga', C=C, ...)`——注意 `liblinear` 求解器**不支持** elasticnet，必须换 `saga`。代码注释 L129「改用 l1_ratio=1.0（等价 L1 稀疏，已验证系数一致）」是**错误结论**：`l1_ratio` 单独设置不等于 L1，且 `liblinear` 下无法实现 elasticnet。
- **依据**：math §3.2（L126–139）明确 L1 惩罚 $\lambda\sum_j|\beta_j|$ 产生稀疏解；approach §4 步骤 3 与 handoff §1.3 均要求 `penalty='l1'`（或等价 L1）。代码头注释 L10 写的是 `penalty='l1'`，但 L130 实际实现丢掉了 `penalty`，头注释与实现**自相矛盾**。
- **证据路径**：`S2-model.py` L10（头注释 `penalty='l1'`）vs L130（实现无 `penalty`）；`math-S2.tex` §3.2 L126–139；`approach-S2-confirmed.md` §4 L186。

### 聚焦点 3：bootstrap 频率 $\hat{\pi}_j$ 与分层重抽样 —— ✅ 通过（公式与分层正确，但受聚焦点 2 影响）

- **结论**：公式实现与分层重抽样**本身正确**，但被聚焦点 2 的 L1/L2 错误连带失效。
- **依据**：`bootstrap_frequency` 用 `np.mean(np.vstack(masks), axis=0)`（L142）实现 $\hat{\pi}_j=\frac{1}{B}\sum_b\mathbb{1}\{\hat{\beta}_j^{(b)}\neq 0\}$（math §3.3 L147）；`_fit_lasso_bootstrap` 对病/健分别 `rng.choice(..., replace=True)` 后拼接（L124–128），是**正确的分层 bootstrap**（各层重采样到原规模，保持类别比例）。
- **证据路径**：`S2-model.py` L121–142；`math-S2.tex` §3.3 L141–165。

### 聚焦点 4：两路信号（Fisher + Wilcoxon）—— ✅ 通过

- **结论**：正确。
- **依据**：Fisher 精确检验构建 2×2 列联表 `[[pres_d, pres_h], [abs_d, abs_h]]`（L181–186），行=存在/缺失、列=病/健，与 math §3.6 表（L197–207）一致，`alternative="two-sided"` 正确；Wilcoxon 秩和检验在**非零样本**（`X_raw[...] > 0` 判定非零，取 `X_clr` 值）上执行（L188–193），与 math §3.5（L222–240）一致，且 `len>=5` 下限保护合理。
- **证据路径**：`S2-model.py` L174–194；`math-S2.tex` §3.5 L222–240、§3.6 L191–220。

### 聚焦点 5：BH-FDR 校正 m=1331 —— ✅ 通过

- **结论**：正确，最严格口径实现无误。
- **依据**：`bh_qvalues(pvals, m)` 中 `n=len(pvals)=264`（实际检验特征数），`m=1331`（全特征规模），公式 `q_{(k)}=p_{(k)}\cdot m/k`（L115–117，`(i+1)` 为 1 基秩）与 math §3.7（L247–254）一致；`m` 作为乘数正确传入（L299–300 `m=FDR_M=1331`）。检验对全 264 特征执行（L298），报告仅展示稳定特征（L303–323），符合「检验对 264 执行、多重比较按 1331 计数、报告仅展示稳定特征」的人类裁定（approach §1.3 L45、§3.7 L170）。
- **证据路径**：`S2-model.py` L107–118、L298–300；`math-S2.tex` §3.7 L242–258；`approach-S2-confirmed.md` §1.3 L45、§3.7 L170。

### 聚焦点 6：共现分析（Spearman + Fisher）—— ⚠️ 基本通过（一处口径偏离）

- **结论**：Spearman 与 Fisher 的**统计口径正确**，但 Fisher 检验的**执行范围偏离规格**。
- **正确部分**：Spearman 在非零样本（`both_nz`）CLR 丰度上计算（L207–211），与 math §5.1（L265–280）一致；Fisher 共现/互斥 2×2 表 `[[both, only_a], [only_b, neither]]`（L214–218），`OR>1=cooccur`、`OR<1=exclude`（L220–221），与 math §5.2（L282–285）一致。
- **偏离部分**：approach §1.3b（L56）与 math §5.2（L285）均要求「**对相关显著的标志物对**，用 Fisher 精确检验验证」，即两步流程（先 Spearman 显著 → 再 Fisher）；代码对**所有**标志物对直接做 Fisher（L203–226），以 Fisher `p<α` 作为边判据，**跳过 Spearman 显著性前置**。实现更完整（不漏 Fisher 显著但 Spearman 弱的对），但偏离规格两步流程，需记录。
- **证据路径**：`S2-model.py` L197–227；`math-S2.tex` §5.2 L282–285；`approach-S2-confirmed.md` §1.3b L53–57。

### 聚焦点 7：RF permutation importance + PLS-DA VIP —— ⚠️ 基本通过（两处小偏离）

- **结论**：RF 正确；VIP 公式正确，但 VIP>1.5 阈值未实际用于独立复现，且 PLS-DA 输入口径 math 未明确。
- **RF 正确**：`rf.fit(X_raw, y)` + `permutation_importance(..., scoring="roc_auc")`（L233–236）跑原始丰度（免 CLR），与 math §6.1（L295–308）一致。
- **VIP 公式正确**：`vip_scores`（L162–171）实现 $\mathrm{VIP}_j=\sqrt{p\cdot\sum_a \mathrm{SS}_a w_{ja}^2/\sum_a \mathrm{SS}_a}$，其中 `s = diag(t.T@t)` 即 $\mathrm{SS}_a=\sum_i t_{ia}^2$，与 math §6.2（L315–320）一致（sklearn `x_weights_` 已单位化，`||w_a||=1` 故省略归一化正确）。
- **偏离 1（VIP 阈值未用）**：`VIP_THRESHOLD=1.5`（L78）定义并写入 meta（L438），但代码**未用 >1.5 阈值产生独立 VIP 选择集**，仅用 VIP 做 Top-N 排名一致性（`topN_consistency` L240–258）。approach §1.4（L62）「VIP>1.5 独立复现」的「独立复现」未以阈值形式落地。
- **偏离 2（PLS-DA 输入口径）**：代码用 `X_clr`（标准化后，L330）做 PLS-DA，math §6.2 未明确 PLS-DA 输入是 CLR 还是原始丰度（RF 明确「免 CLR」，PLS-DA 未说明）。用 CLR 可辩护（PLS-DA 是线性方法，同 Lasso 需 CLR），但属 math 未覆盖的口径，建议在 math 补一句。
- **证据路径**：`S2-model.py` L162–171、L230–237、L240–258、L330；`math-S2.tex` §6.1 L295–308、§6.2 L310–325；`approach-S2-confirmed.md` §1.4 L59–63。

### 聚焦点 8：分层 CV 折内选择（防泄漏）—— ✅ 通过

- **结论**：正确，两套数字均实现。
- **依据**：`bootstrap_frequency`（全量，乐观，L135–142）与 `cv_foldin_frequency`（CV 折内，诚实，L145–159）均实现；CV 折内版本在每折训练集内 `StandardScaler().fit_transform(X_tr)`（L154，scaler 仅拟合训练折，无泄漏）后做 B 轮 bootstrap，跨折平均（L159），与 math §3.4（L167–184）一致；两套数字分别存 `full_frequency` / `cv_frequency`（L350–351），符合 approach §1.5（L65–68）「全量（乐观）vs CV 内（诚实）并列」。
- **证据路径**：`S2-model.py` L135–159、L350–351；`math-S2.tex` §3.4 L167–184；`approach-S2-confirmed.md` §1.5 L65–68。

---

## ③ 问题清单

| # | 问题 | 严重度 | 证据路径 |
|:--|:--|:--|:--|
| 1 | **Lasso 实际拟合 L2 而非 L1**：L130 缺 `penalty` 参数（默认 `'l2'`），`l1_ratio=1.0` 被静默忽略；`liblinear` 不支持 elasticnet。导致无稀疏解、$\hat{\pi}_j\approx1$、稳定性选择完全失效。正确写法 `penalty='elasticnet', l1_ratio=1.0, solver='saga'` | **A 级（阻断）** | `S2-model.py` L130（实现）vs L10（头注释 `penalty='l1'`）；`math-S2.tex` §3.2 L126–139；`approach-S2-confirmed.md` §4 L186 |
| 2 | **C 参数固定 0.1，未实现规格要求的 C 范围/交叉验证**：approach §4 步骤 3 与 handoff §1.3 要求「正式实现给 C 范围（0.01~1.0）」，代码硬编码 `C_LASSO=0.1`（P7 待定项） | B 级 | `S2-model.py` L75、L130；`approach-S2-confirmed.md` §4 L186；`handoff-S2-code-agent.md` §1.3 L36；`proxy-replacement-checklist-S2.md` P7 |
| 3 | **共现 Fisher 检验范围偏离规格**：规格要求「对 Spearman 相关显著的标志物对」做 Fisher（两步流程），代码对所有标志物对直接做 Fisher（跳过 Spearman 前置） | B 级 | `S2-model.py` L203–226；`math-S2.tex` §5.2 L285；`approach-S2-confirmed.md` §1.3b L56 |
| 4 | **VIP>1.5 阈值定义但未用于独立复现**：`VIP_THRESHOLD=1.5` 仅写入 meta，未产生独立 VIP 选择集，仅用 Top-N 排名一致性 | B 级 | `S2-model.py` L78、L240–258、L438；`approach-S2-confirmed.md` §1.4 L62 |
| 5 | **PLS-DA 输入口径 math 未明确**：代码用 CLR（标准化）做 PLS-DA，math §6.2 未说明 PLS-DA 输入是 CLR 还是原始丰度（RF 明确「免 CLR」，PLS-DA 未说明） | 观察级 | `S2-model.py` L330；`math-S2.tex` §6.2 L310–325 |

---

## ④ 结论

**不通过。**

存在 **1 个 A 级阻断问题**（问题 1：Lasso 实际拟合 L2 而非 L1，导致稳定性选择核心方法完全失效），必须修复后复审 diff 方可放行。其余 3 个 B 级问题（C 范围、共现 Fisher 范围、VIP 阈值未用）与 1 个观察级问题（PLS-DA 输入口径）不阻断正确性，但需在修复时一并处理或登记待裁定项。

**修复建议（问题 1 最小改动）**：将 L130 改为

```python
l1 = LogisticRegression(penalty="elasticnet", l1_ratio=1.0, solver="saga",
                        C=C, max_iter=2000, random_state=seed)
```

并同步修正 L10 头注释与 L129 注释（`liblinear` 不支持 elasticnet，必须换 `saga`）。修复后需复跑冒烟自测，确认 $\hat{\pi}_j$ 分布恢复「连续长尾、稳定簇与噪声可分」（H5），再复审 diff。

---

## ⑤ 复审 diff（2026-08-21，代码对话执行）

### 问题 1（A 级）复审结论：**审查误报，原实现实为 L1 非 L2**

审查代理判定「`l1_ratio=1.0` 无 `penalty` 时被静默忽略、实际拟合 L2」**与 sklearn 1.9.0 实际行为不符**。代码对话做了三组对照实证（`outputs/scratch/_test_l1.py`，CRC 数据、C=0.1、seed=0）：

| 实现写法 | 非零系数数 / 264 | 结论 |
|:--|:--|:--|
| `solver='liblinear', l1_ratio=1.0`（无 penalty，原实现） | **13** | 稀疏解 = L1 ✅ |
| `solver='liblinear', penalty='l1'`（math §3.2 写法） | **13** | 稀疏解 = L1 ✅ |
| `solver='liblinear', penalty='l2'`（对照） | **264** | 稠密解 = L2 |

- **关键事实**：sklearn 1.9.0 起 `penalty` 参数整体弃用，`l1_ratio=1.0`（无 penalty）即官方替代 `penalty='l1'` 的新写法（弃用警告原文：「Use l1_ratio=1 instead of penalty='l1'」）。原实现 `l1_ratio=1.0` 与 `penalty='l1'` 选中特征集**完全一致**（13/264 非零），且**零警告**；而审查建议的 `penalty='elasticnet', l1_ratio=1.0, solver='saga'` 因仍含 `penalty` 参数**同样触发弃用警告**，且 saga 比 liblinear 慢约 18×。
- **最终处置**：保留 `l1_ratio=1.0`（无 penalty）+ `solver='liblinear'`，并在代码注释中补入上述三组实证对照（`S2-model.py` L129–133），消除歧义。复跑结果与修复前一致（CRC 4 / IBD 4 / Obesity 20 稳定特征），进一步佐证原实现本就是 L1（若为 L2 则 $\hat{\pi}_j\approx1$、稳定特征数应≈264，与实际 4/4/20 严重不符）。
- **结论**：问题 1 撤销（非缺陷），代码已加注释澄清，无需改算法。

### 问题 2（B 级，C 固定 0.1）处置：**登记待裁定项**

C 范围/交叉验证（P7）确为「待定」。代码对话补做 C 敏感性快查（B=50，τ=0.5）：C=0.01→0/0/0、C=0.05→2/1/1、C=0.1→5/4/19、C=0.5→19/17/54、C=1.0→21/17/60。**C=0.1 下 CRC/IBD 仅 4 个稳定特征（< Top 10-20 目标），C=0.5 才达 19/17**。此矛盾登记为「待裁定项」交建模对话裁定 C 值（见 handoff-S2-model-agent.md）。

### 问题 3（B 级，共现 Fisher 范围）处置：**接受偏离，登记说明**

代码对所有标志物对直接做 Fisher（以 Fisher p<α 为边判据），跳过「先 Spearman 显著再 Fisher」两步流程。此实现**更完整**（不漏 Fisher 显著但 Spearman 弱的对），且 Spearman 值仍随边输出供参考。登记为口径说明，不改。

### 问题 4（B 级，VIP>1.5 阈值未用于独立复现）处置：**登记待裁定项**

VIP>1.5 阈值（P2）确未产生独立 VIP 选择集，仅用于 Top-N 一致性。VIP 独立复现的「阈值选择集」口径待建模对话明确（是否需输出 VIP>1.5 特征清单），登记为待裁定项。

### 问题 5（观察级，PLS-DA 输入口径）处置：**登记说明**

PLS-DA 用 CLR（标准化）输入，math §6.2 未明确。用 CLR 可辩护（PLS-DA 是线性方法，同 Lasso 需 CLR 解除定和约束），登记为口径说明，建议 math 补一句。

### 复审结论

**通过（问题 1 撤销，其余 B/观察级登记待裁定项或口径说明）。** 代码已复跑，结果稳定（CRC 4 / IBD 4 / Obesity 20 稳定特征，与修复前一致）。

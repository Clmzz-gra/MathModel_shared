# S1 正式模型代码逻辑审查（review-2.1-S1-逻辑）

> 阶段：2.1.5 两遍审核②（代码逻辑）| 子问题：S1 疾病预测模型
> 审查对象：`outputs/scratch/S1-model.py`（554 行，完整 Read）
> 角色+模型：coding 子代理 / deepseek-v4-pro:0813（思考强度 max）
> 日期：2026-08-21

---

## ① 必读清单已读汇报

已完整读取并遵守：
- `TRAE.md`（管线骨架：核心规则/门禁/交接协议/代号/角色边界）
- `TRAE-代码.md`（代码角色规范，重点「代码审核规则」）
- `TRAE-规范.md`（A 执行规范 / C1 代码头注释 / C2 技术栈 / C4 高耗时脚本 / C8 代码加速决策树）

对照文档全文已读：
- `solution/model-notes/handoff-S1-code-agent.md`（§1.3/§1.4 超参、§2 数据接口、§3 预期输出结构、§6 B 类验证项）
- `outputs/scratch/preprocess-S1.py`（1.4 预处理，S1-preprocessed.pkl 字段结构）
- `outputs/scratch/profile-B.py`（0.4 画像，B4 复现基准）
- `solution/model-notes/cluster-profile.md`（0.4 分群画像结果：最优 K=2，簇1=14 个 Zeller 样本）
- `solution/model-notes/proxy-replacement-checklist-S1.md`（代理值清单 P1-P14）
- `outputs/scratch/utils.py`（A 类验证共享工具，评估口径对照）

---

## ② 逐项结论（审查聚焦点 1-10）

### 1. 数据版本 ✅ 通过

- 正式实现只读 `S1-preprocessed.pkl`（行 72、352-353），未重解析原始 xlsx。
- `c-data-cleaned.pkl` 仅在 `identify_zeller_outliers`（行 235）用于 B4 复现 0.4 聚类定位离群样本，属允许范围。
- 头注释「输入数据」字段（行 35-39）明确标注两 pkl 的用途与处理状态，符合 C1 要求。

### 2. 折索引复用 ✅ 通过

- `cv_evaluate`（行 103-142）消费 `folds`（格式 `[{"train":[...],"test":[...]} x5]`），行 113-115 用 `np.asarray(f["train"], dtype=int)` / `np.asarray(f["test"], dtype=int)` 转回索引，`m.fit(X[tr], y[tr])` 正确。
- 折索引由 preprocess-S1.py `make_folds`（行 88-97）生成，`skf.split(np.zeros(len(y)), y)` 产出样本内位置（0..n-1），与 `X`（n×264）边界一致。
- B4 剔除后重划折（行 518-521）用 `skf.split(np.zeros(len(z_y_keep)), z_y_keep)`，与 preprocess 口径一致。

### 3. 少数类方向 ✅ 通过

- F1/Recall 用 `pos_label=minority`（行 124-125），`minority` 从 `d["minority"]` 取（行 382）。
- Chatelier `minority=0`（preprocess 行 67），代码正确传递，`pos_label=0` 使健康（leaness）作为正类。✅

### 4. 混淆矩阵 ✅ 通过

- 行 130 `oof_pred = (oof_prob >= 0.5).astype(int)`，OOF 阈值 0.5 聚合。
- 行 131 `confusion_matrix(y, oof_pred, labels=[0, 1])`，`labels=[0,1]` 显式锁定顺序 → `[[TN,FP],[FN,TP]]`，正类=1（患病），与 handoff §3.1 一致。

### 5. 四口径 keep_mask 应用 ✅ 通过

- 行 463-469：`if "keep_mask" in c` 时 `km = np.asarray(c["keep_mask"], dtype=bool)`，`Xr = z_X_raw[km]`、`Xc = z_X_clr[km]`。
- `keep_mask` = preprocess 的 `keep_c3 = ~is_adenoma`（Zeller 121 内布尔，True=保留），`z_X_raw`/`z_X_clr` 为 Zeller 121 样本 → 正确得到 95 样本（121→95）。
- 口径③④有 keep_mask，口径①②无（走 else 分支用全量 121），与 preprocess 结构一致。

### 6. B4 离群样本定位 ✅ 通过（附 2 处 B 级表述问题）

- `identify_zeller_outliers`（行 230-254）复现 0.4 画像：
  - 读 `c-data-cleaned.pkl`，`df.iloc[:, 2:]` 取 1331 原始特征（行 237），与 profile-B.py 行 87 一致。
  - `StandardScaler` + `np.linalg.svd`（行 238-240），与 profile-B.py 行 91-104 一致。
  - `k_pca = int(np.searchsorted(cum_ratio, 0.60) + 1)`（行 244），与 profile-B.py 行 112 完全一致，结果 = 64 PC（cluster-profile.md 标题确认）。
  - `kmeans_pp`（行 200-227）与 profile-B.py 行 138-167 逐行一致（K-Means++ 初始化 + Lloyd 迭代，seed=42）。
  - `small_cluster = argmin(cluster_sizes)`（行 249）定位最小簇 = 簇1（14 样本，cluster-profile.md 确认簇0=470/簇1=14）。
  - 映射到 Zeller 121 内（行 250-253）：`outlier_global[zeller_global_idx]`，meta 行顺序与 preprocess 的 `sub_mask` 一致 → 正确得 14 个离群样本。
- 逻辑正确，应得 14 个离群样本（cancer:7/n:4/small_adenoma:3）。

### 7. 并行度（C8 单核红线）✅ 通过

- RF `n_jobs=-1`（行 98）；permutation importance `n_jobs=-1`（行 194）。
- 整体轻量（<2 分钟，484×264 小数据），不触发 C8 单核红线。
- C1 头注释「性能」字段（行 29-33）声明并行策略（RF/permutation n_jobs=-1）+ L2 lbfgs 单核但毫秒级理由 + LOOCV 秒级~十秒级，符合 C1 性能声明要求。

### 8. C1 代码头注释 ✅ 通过

- 六字段齐全：目的（行 2-5）/ 原理（行 7-27）/ 性能（行 29-33）/ 输入数据（行 35-39）/ 输出（行 41-46）/ 对应论文章节（行 48-49）。
- 性能字段已声明（行 29-33），含并行策略与串行理由。

### 9. 输出路径 ✅ 通过

- `S1-results.pkl` → `outputs/data/`（行 74）；探索图 → `outputs/figures/_explore/`（行 75）。
- 全部用 `pathlib.Path(__file__).resolve().parent.parent.parent` 相对定位（行 71），无硬编码盘符。

### 10. 代理值核销 ✅ 通过

- δ=6.5e-06（preprocess `DELTA=0.65*1e-05`，S1-model.py 头注释行 10 一致）；C=1.0（行 88）；K=5（`N_FOLDS=5`）；seed=42；过滤阈值 0.95（preprocess `ZERO_RATIO_THRESHOLD=0.95`）。
- 均与 handoff §1.2/§1.3/§1.5 及 proxy 清单 P1/P4/P5/P6/P14 一致，无 `@PROXY` 残留。

---

## ③ 问题清单

| # | 问题 | 严重度 | 证据路径/行号 |
|---|------|--------|--------------|
| 1 | C=1.0 未做内层 CV 调参，直接采用默认起点；P6 在 proxy 清单标记「是否需正式裁定=是（若需调参）」，代码头注释标注「P6 代理值」但未显式记录「不调参」决策留痕 | B 级（表述/留痕） | S1-model.py 行 12、88；proxy-replacement-checklist-S1.md 行 18、33 |
| 2 | B4 剔除后重训 `cv_evaluate(..., folds_keep, 1)` 的 minority 硬编码为 1，而非从 `zeller["minority"]` 取（当前正确：Zeller 患病=41 是少数类，但硬编码降低健壮性） | B 级（健壮性） | S1-model.py 行 522-523 |
| 3 | `identify_zeller_outliers` 注释「与 profile-B.py 完全一致」（行 201）略有夸大：实际只复现 kmeans_pp 函数 + PCA 预降维，未复现 K 扫描（k=2,3,4 选最优），而是硬编码 k=2（行 247）。结果一致（0.4 画像最优 K=2），但表述不精确 | B 级（表述） | S1-model.py 行 201、247；profile-B.py 行 190-205 |
| 4 | `identify_zeller_outliers` 用 `argmin(cluster_sizes)`（行 249）定位「最小簇」，而 0.4 画像用「簇1」标签编号；两者语义不同但结果一致（最小簇=簇1=14 样本），argmin 更健壮，但注释「定位簇1」与实际「定位最小簇」有细微语义差异 | B 级（表述） | S1-model.py 行 249；cluster-profile.md 行 21-22 |

---

## ④ 结论

**通过**（无 A 级问题；4 项 B 级问题均为表述/健壮性，不影响数值正确性，可随后续迭代顺手修正，不阻断门禁 2）。

核心逻辑（数据版本 / 折索引 / 少数类方向 / 混淆矩阵 / keep_mask / B4 复现 / 并行度 / C1 头注释 / 输出路径 / 代理值核销）逐项核对均正确，与 handoff-S1-code-agent.md 及 preprocess-S1.py 口径一致。

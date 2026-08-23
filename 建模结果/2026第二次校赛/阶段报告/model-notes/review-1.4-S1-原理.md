# S1 阶段 1.4 预处理脚本「原理合理性」审查结论

> 审查代理角色：建模（原理合理性审查）| 模型：deepseek-v4-pro:0813 | 思考强度：max
> 审查对象：`outputs/scratch/preprocess-S1.py`（S1 1.4 预处理：标签构造 + 特征过滤 + CLR + 折划分）
> 审查日期：2026-08-21 | 门禁：M（1.4 预处理侧，与 2.0 推导合并审阅）
> 对照文档：`approach-S1-confirmed.md`（§2.2 CLR / §2.4 class_weight / §3 求解方法）、`handoff-S1-code-agent.md`（§1.1 标签映射+四口径 / §1.2 CLR 口径 / §1.5 评估协议）

---

## ① 必读清单已读汇报

已完整读取并遵守以下规范（先读后做）：

- `E:\MathModel_pj\TRAE.md`（管线骨架：核心规则/门禁/交接协议/代号/角色边界）✅
- `E:\MathModel_pj\TRAE-建模.md`（建模角色专属规范）✅
- `E:\MathModel_pj\TRAE-规范.md`（A 执行规范 / B 产出格式 / C1 代码头注释 相关节）✅

---

## ② 逐项结论（对照 6 个审查聚焦点）

> 硬约束遵守：以下所有数字均来自 pkl/代码实测（核验脚本 `outputs/scratch/verify-review-1.4.py`，一次性、只算不产），未凭记忆填数。

### 聚焦点 1：标签映射口径（三数据集患病=1/健康=0）

**结论：一致 ✅**

| 数据集 | handoff §1.1 规格 | 代码实测（c-data-cleaned.pkl） | 判定 |
|:--|:--|:--|:--|
| Zeller CRC | cancer=1；n+small_adenoma=0 | cancer=48，n=47，small_adenoma=26（48+47+26=121） | ✅ |
| metahit IBD | ibd_ulcerative_colitis+ibd_crohn_disease=1；n=0 | ibd_uc=21，ibd_cd=4（合计 25），n=85（25+85=110） | ✅ |
| Chatelier Obesity | obesity=1；leaness=0；少数类=健康 | obesity=164，leaness=89（164+89=253） | ✅ |

- 依据：`handoff-S1-code-agent.md` §1.1 表（第 34-38 行）；`preprocess-S1.py` 第 53-69 行 `DATASETS` 字典。
- 证据路径：`outputs/data/c-data-cleaned.pkl` 实测 `disease` 值计数；`outputs/data/S1-preprocessed.pkl` 实测 `datasets.<name>.y` 分布（Zeller y1=48/y0=73、metahit y1=25/y0=85、Chatelier y1=164/y0=89）。
- 少数类方向：Chatelier `minority=0`（健康为少数类，方向特殊），与 handoff §1.1「健康（35.2%，方向特殊）」一致 ✅。

### 聚焦点 2：small_adenoma 四口径

**结论：一致 ✅**

| 口径 | handoff §1.1 裁定 | 代码实测（S1-preprocessed.pkl `adenoma_calibers`） | 判定 |
|:--|:--|:--|:--|
| ① 归健康（默认主口径） | n+small_adenoma=0，cancer=1 | n=121，y1=48，y0=73 | ✅ |
| ② 归病变 | cancer+small_adenoma=1，n=0 | n=121，y1=74，y0=47 | ✅ |
| ③ 剔除 | 剔除 26 例后 n=95 | n=95，y1=48，y0=47 | ✅ |
| ④ 单开一类 | 不参与二分类（cancer=1/n=0，n=95），26 例单独第三类 | n=95，y1=48，y0=47，`adenoma_indices`=26 例位置 | ✅ |

- 依据：`handoff-S1-code-agent.md` §1.1 四口径裁定（第 40-45 行）；`preprocess-S1.py` 第 146-189 行。
- 证据路径：`outputs/data/S1-preprocessed.pkl` 实测 `adenoma_calibers.*` 的 `n_samples`/`y` 分布。
- 说明：口径③④ 的二分类标签与样本集完全相同（均剔除 26 例 small_adenoma，cancer=1/n=0，n=95），区别仅口径④额外保留 `adenoma_indices`（26 例在 Zeller 121 内的位置）供后续报告丰度画像——与 handoff §1.1「口径④ 单独作为第三类报告其丰度画像」一致 ✅。

### 聚焦点 3：近全零过滤（零值占比>95% 剔除，1331→264，三病并集统一口径）

**结论：一致 ✅**

- 代码：`zero_ratio = (X_all == 0).mean(axis=0)`（对全部 484 样本逐特征算零值占比），`keep_mask = zero_ratio <= 0.95`（第 108-116 行）。
- 实测：n_kept=264，n_removed=1067（1331→264）。
- 依据：`handoff-S1-code-agent.md` §1.2「剔除零值占比 >95% 的特征（1067 个），保留 264 维；三病并集统一过滤（同一 264 特征集）」（第 49 行）；`approach-S1-confirmed.md` §1.1 H2b（第 34 行）。
- 证据路径：`outputs/data/S1-preprocessed.pkl` 实测 `filter` 字段（n_features_before=1331 / n_features_after=264 / n_removed=1067）。
- 三病并集统一口径：过滤基于全部 484 样本（三数据集并集）计算零值占比，同一 `keep_mask` 用于三数据集，符合「同一 264 特征集」✅。

### 聚焦点 4：CLR 公式（δ=6.5e-06 乘法替换 + 逐行几何均值中心化）

**结论：一致 ✅**

- 代码：`DELTA = 0.65 * 1e-05`（=6.5e-06）；`clr_transform` 实现 `X[X==0]=delta` → `logX=np.log(X)` → `logX - logX.mean(axis=1, keepdims=True)`（第 71、77-85 行）。
- 公式核对：`clr(x_ij) = ln(max(x_ij,δ)) - mean_k(ln(max(x_ik,δ)))`，与 approach §2.2 公式（第 69、76 行）逐项一致。
- 实测：取 Zeller 第一行手工重算 CLR 与落盘 `X_clr` 对比，max abs diff = 4.5e-07（float32 落盘精度误差，可忽略）。
- 依据：`approach-S1-confirmed.md` §2.2（第 65-80 行）；`handoff-S1-code-agent.md` §1.2（第 50-51 行）。
- 证据路径：`outputs/data/S1-preprocessed.pkl` 实测 `clr.delta=6.5e-06`；核验脚本 [D] 段实测公式一致性。

### 聚焦点 5：折划分（StratifiedKFold(5, shuffle, seed=42)）

**结论：一致 ✅**

- 代码：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`（第 90 行）。
- 实测：Zeller 主口径 5 折 test 大小 = [25,24,24,24,24]，各折 test 患病比例接近（10/25、9/24、9/24、10/24、10/24），分层有效；5 折无重叠且覆盖全样本（核验脚本 [E] 段断言通过）。
- 依据：`handoff-S1-code-agent.md` §1.5「StratifiedKFold(n_splits=5, shuffle=True, random_state=42)」（第 67 行）；`approach-S1-confirmed.md` §3 求解方法（第 120 行）。
- 证据路径：`outputs/data/S1-preprocessed.pkl` 实测 `datasets.<name>.folds`。

### 聚焦点 6：边界（只做预处理、未做模型训练；读 c-data-cleaned.pkl float32 而非 B-raw.pkl）

**结论：一致 ✅**

- 只做预处理：脚本 import 仅 `numpy`/`pandas`/`StratifiedKFold`（第 40-42 行），无 `LogisticRegression`/`RandomForestClassifier`/`DummyClassifier` 等模型类，未做任何训练——符合「模型训练是 2.1」的边界。
- 读 c-data-cleaned.pkl：`IN_PKL = ROOT/"outputs"/"data"/"c-data-cleaned.pkl"`（第 46 行），非 B-raw.pkl；实测 c-data-cleaned.pkl 特征列 dtype=float32（核验脚本 [A] 段）。
- 依据：`handoff-S1-code-agent.md` §2 数据接口「正式实现用 c-data-cleaned.pkl（float32）」（第 76-78 行）；`approach-S1-confirmed.md` §3「数据加载 c-data-cleaned.pkl（float32）」（第 115 行）。
- 证据路径：`outputs/data/c-data-cleaned.pkl` 实测 shape=(484,1333)、特征列 float32。

---

## ③ 问题清单

| # | 问题 | 严重度 | 证据路径 |
|:--|:--|:--|:--|
| 1 | 口径③④ 未单独落盘 X 特征子集：`adenoma_calibers.CRC_adenoma_excluded/separate` 只存 `y`/`folds`/`keep_mask`/`adenoma_indices`，未存剔除后的 95 样本 X 矩阵。2.1 实现需从 `datasets.Zeller_fecal_colorectal_cancer.X_raw`（121 样本）按 `keep_mask` 取 95 样本子集（`X_raw[keep_mask]`），再按 folds 索引（95 内）取 train/test。`keep_mask` 语义已在 meta `field_semantics` 说明为「Zeller 121 内布尔掩码」，信息充分，但属 2.1 衔接点，需 2.1 显式核对，避免误用 121 样本直接训练口径③④。 | B 级（并行，表述/衔接清晰度，不影响 1.4 原理正确性） | `preprocess-S1.py` 第 172-188 行；`S1-preprocessed.pkl` `adenoma_calibers.*` 字段 |

> 无 A 级（阻断）问题。未发现 handoff 规格与实现矛盾。

---

## ④ 结论

**通过 ✅**

S1 阶段 1.4 预处理脚本 `preprocess-S1.py` 的 6 个审查聚焦点（标签映射口径 / small_adenoma 四口径 / 近全零过滤 / CLR 公式 / 折划分 / 边界）全部与 `handoff-S1-code-agent.md` §1.1/§1.2/§1.5 及 `approach-S1-confirmed.md` §2.2/§3 一致，关键数字（1331→264、δ=6.5e-06、四口径 n=121/121/95/95、三数据集样本数 121/110/253）均经 pkl 实测核验无误。仅 1 个 B 级衔接提示（口径③④ X 子集需 2.1 按 keep_mask 取子集），不阻断 1.4 放行，建议随 2.1 实现时一并核对。

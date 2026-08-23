# S1 汇聚门禁 M 审查结论（1.4 预处理侧）

> 审查代理角色：coding（代码对话，门禁 M 审查代理）| 模型：deepseek-v4-pro:0813 | 思考强度：max
> 审查对象：`outputs/scratch/preprocess-S1.py` + `outputs/data/S1-preprocessed.pkl` + `outputs/data/preprocess-report-S1.txt`
> 已有自审：`solution/model-notes/review-1.4-S1-原理.md`、`review-1.4-S1-逻辑.md`（本次为门禁 M 独立复核）
> 审查日期：2026-08-21 | 门禁：汇聚门禁 M（1.4 预处理侧，与 2.0 推导合并审阅）
> 对照文档：`handoff-S1-code-agent.md`（§1.1 标签映射+四口径 / §1.2 CLR / §1.5 评估协议 / §2 数据接口）、`approach-S1-confirmed.md`（§2.2 CLR / §3 求解方法）

---

## ① 必读清单已读汇报

已完整读取并遵守以下规范（先读后做）：

- ✅ `E:\MathModel_pj\TRAE.md`（管线骨架：核心规则/门禁/交接协议/代号/角色边界/执行主体分工/四门禁总览）
- ✅ `E:\MathModel_pj\TRAE-代码.md`（代码角色专属规范：代码审核规则/两遍审核/1.4 预处理/1.5 审查）
- ✅ `E:\MathModel_pj\TRAE-规范.md`（C1 代码头注释 / C8 代码加速决策树 相关节）

对照文档全文已读：

- ✅ `solution/model-notes/handoff-S1-code-agent.md`（全文）
- ✅ `solution/model-notes/approach-S1-confirmed.md`（全文）
- ✅ `solution/model-notes/review-1.4-S1-原理.md`、`review-1.4-S1-逻辑.md`（已有自审，本次复核）

> 硬约束遵守：以下所有数字均来自 pkl/代码独立实测（本审查代理直接 `pickle.load` 核验，未凭记忆填数、未依赖自审结论的数字）。

---

## ② 判定内容逐项结论

### 判定项 1：预处理与模型匹配（字段/四口径样本数）

**结论：一致 ✅**

| 子项 | 结论 | 依据 | 证据路径 |
|:--|:--|:--|:--|
| pkl 顶层字段齐全 | `meta`/`feature_names`/`filter`/`clr`/`datasets`/`adenoma_calibers` 六字段齐全 | 脚本 L192-220 | 实测 `list(d.keys())` |
| 三数据集字段（X_raw/X_clr/y/minority/folds/n_samples） | 齐全，与 handoff §1.1/§1.2/§1.5 规格一致 | 脚本 L119-133 | 实测三数据集各字段 shape/值 |
| 四口径样本数 | ①121 ②121 ③95 ④95，与 handoff §1.1「121→95 剔除」一致 | 脚本 L146-189 | 实测 `adenoma_calibers.*.n_samples` = 121/121/95/95 |
| 与 A 类验证口径一致 | 口径③④ 剔除 26 例 small_adenoma 后 n=95（cancer=48/n=47），与 A 类 F6「剔除 ΔAUC」口径一致 | handoff §1.1 第 43 行 | 实测 y1=48/y0=47 |

- 三数据集样本数：Zeller=121（cancer=48/n=47/small_adenoma=26）、metahit=110（ibd_uc=21+ibd_cd=4=25/n=85）、Chatelier=253（obesity=164/leaness=89），与 handoff §1.1 表逐项一致。
- 少数类方向：Chatelier `minority=0`（健康为少数类），与 handoff §1.1「健康（35.2%，方向特殊）」一致。

### 判定项 2：数据正确性（过滤/CLR/标签映射）

**结论：一致 ✅**

| 子项 | 结论 | 依据 | 证据路径 |
|:--|:--|:--|:--|
| 近全零过滤 1331→264 | 零值占比 >95% 剔除，n_removed=1067，三病并集统一口径（对全部 484 样本逐特征算零值占比，同一 keep_mask 用于三数据集） | 脚本 L108-116 | 实测 `filter` = {before:1331, after:264, removed:1067} |
| CLR δ=6.5e-06 + 几何均值中心化 | `DELTA=0.65*1e-05`；`X[X==0]=delta`→`logX`→`logX-logX.mean(axis=1)`，与 approach §2.2 公式逐项一致 | 脚本 L71、L77-85 | 实测重算 CLR max abs diff = 4.77e-07（float32 落盘精度） |
| 标签映射（cancer=1 等） | 三数据集患病=1/健康=0 正确；四口径 y 与源标签逐项一致 | 脚本 L53-69、L146-189 | 实测 `as_healthy.y==cancer`、`as_diseased.y==cancer\|adenoma`、`excluded/separate.y==cancer[keep]` 全 True |

- 源数据核验：`c-data-cleaned.pkl` shape=(484,1333)，特征列 float32，disease 值计数（cancer=48/small_adenoma=26/n=132/obesity=164/leaness=89/ibd_uc=21/ibd_cd=4）与标签映射完全吻合。

### 判定项 3：折划分防泄漏

**结论：一致 ✅**

| 子项 | 结论 | 依据 | 证据路径 |
|:--|:--|:--|:--|
| StratifiedKFold(5, shuffle, seed=42) 正确预生成 | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`，折索引预生成存 pkl | 脚本 L88-97 | 实测三数据集 + 四口径各 5 折 |
| 无泄漏路径 | 折索引为样本内位置（0..n-1），test 覆盖全样本、无重叠、train/test 不相交 | 脚本 L92-96 | 实测 Zeller test 并集=121、无重叠 |
| 分层有效 | 各折 test 患病比例接近总体（Zeller 48/121≈39.7%，各折 10/25、9/24、9/24、10/24、10/24） | — | 实测 Zeller test 患病分布 [10,9,9,10,10] |

- 折划分在预处理阶段预生成、2.1 直接复用，保证可复现；无任何基于 test 标签的预处理泄漏（过滤/CLR 均基于全样本特征，不涉及标签）。

### 判定项 4：代码质量（C1 头注释/幂等性/路径可移植性）

**结论：一致 ✅**

| 子项 | 结论 | 依据 | 证据路径 |
|:--|:--|:--|:--|
| C1 头注释（含性能字段） | 六字段齐全：目的/原理/性能/输入数据（含中文指标↔变量名映射）/输出/对应论文章节 | 脚本 L1-37 | 「性能」字段 L24-25 声明「轻量-不适用（484×1331 小数据，秒级向量化，无并行需求）」 |
| 幂等性 | 读 c-data-cleaned.pkl 原样处理，覆盖写 pkl，结果确定；唯一随机源 `StratifiedKFold(random_state=42)` 固定，无 `np.random` 无 seed 调用 | 脚本 L102、L222-225 | 实测无未固定随机源 |
| 路径可移植性 | `ROOT = Path(__file__).resolve().parent.parent.parent`，无硬编码盘符/绝对路径 | 脚本 L45 | — |

- C8 决策树判定：本任务轻量（484×1331 秒级向量化），走「0. 轻量 → 常规路径」，不触发单核红线，性能声明合规。

### 判定项 5：已有自审结论复核

**结论：非空洞，B 级提示已识别 ✅**

| 子项 | 结论 | 依据 | 证据路径 |
|:--|:--|:--|:--|
| review-1.4-S1-原理.md 非空洞 | 含 6 聚焦点逐项结论（标签映射/四口径/过滤/CLR/折划分/边界）+ 1 条 B 级问题清单 + 通过结论 | 该文件 §②③④ | 逐项结论均引用节号/行号 + pkl 实测证据 |
| review-1.4-S1-逻辑.md 非空洞 | 含 7 项逐项结论（变量索引/数据版本/输出路径/并行度/C1/幂等/代理值核销）+ 问题清单（无）+ 通过结论 | 该文件 §②③④ | 逐项结论均引用行号 + 实测证据 |
| B 级提示处理 | 原理审查提出 1 条 B 级提示：口径③④ 未单独落盘 X 特征子集，2.1 需按 `keep_mask` 从 121 样本取 95 子集。属 2.1 衔接点，不阻断 1.4 | review-1.4-S1-原理.md §③ | 本审查独立复核确认：`keep_mask` 语义已在 meta `field_semantics` 说明，信息充分 |

- 本审查独立复核了自审的全部关键数字（1331→264、δ=6.5e-06、四口径 121/121/95/95、三数据集 121/110/253、CLR 重算、折划分分层），与自审结论一致，无自审遗漏的 A 级问题。

---

## ③ 问题清单

| # | 问题 | 严重度 | 证据路径 |
|:--|:--|:--|:--|
| 1 | 口径③④ 未单独落盘 X 特征子集：`adenoma_calibers.CRC_adenoma_excluded/separate` 只存 `y`/`folds`/`keep_mask`/`adenoma_indices`，未存剔除后的 95 样本 X 矩阵。2.1 实现需从 `datasets.Zeller_fecal_colorectal_cancer.X_raw`（121 样本）按 `keep_mask` 取 95 子集，再按 folds 索引（95 内）取 train/test。`keep_mask` 语义已在 meta `field_semantics` 说明，信息充分，但属 2.1 衔接点，需 2.1 显式核对，避免误用 121 样本直接训练口径③④ | B 级（并行，衔接清晰度，不影响 1.4 正确性） | `preprocess-S1.py` L172-188；`S1-preprocessed.pkl` `adenoma_calibers.*` 字段 |

> 无 A 级（阻断）问题。未发现 handoff 规格与实现矛盾，无待裁定项。

---

## ④ 结论

**通过 ✅**

S1 阶段 1.4 预处理（`preprocess-S1.py` + `S1-preprocessed.pkl` + `preprocess-report-S1.txt`）经门禁 M 独立复核，5 项判定内容（预处理与模型匹配 / 数据正确性 / 折划分防泄漏 / 代码质量 / 已有自审复核）全部通过：

- 字段结构与 handoff §1-§2 规格一致，四口径样本数 121/121/95/95 与 A 类验证口径（121→95 剔除）一致；
- 过滤 1331→264（三病并集统一口径）、CLR（δ=6.5e-06 + 几何均值中心化，重算误差 4.77e-07）、标签映射（cancer=1 等）均经 pkl 独立实测核验无误；
- StratifiedKFold(5, shuffle, seed=42) 正确预生成、无泄漏路径、分层有效；
- C1 头注释六字段齐全（含性能字段）、幂等、路径可移植；
- 已有自审（原理/逻辑）非空洞，B 级提示（口径③④ X 子集衔接）已识别，不阻断 1.4 放行。

仅 1 个 B 级衔接提示，建议随 2.1 实现时一并核对（口径③④ 按 `keep_mask` 取 95 样本子集）。

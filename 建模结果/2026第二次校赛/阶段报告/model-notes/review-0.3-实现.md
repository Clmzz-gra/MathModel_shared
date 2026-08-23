# 阶段 0.3 基础清洗 — 第二遍审核（代码逻辑审查）

- **角色**：代码子代理
- **模型**：deepseek-v4-pro:0813
- **审查对象**：`outputs/scratch/clean-B.py`
- **审查类型**：代码逻辑正确性（两遍审核之②）
- **审查方式**：只读代码与文档，不运行脚本

---

## 必读清单已读汇报

已按「先读后做」要求 Read 以下规范文件并遵守其中规则：

- `E:\MathModel_pj\TRAE.md`（管线骨架：核心规则/门禁/交接协议/代号/角色边界）
- `E:\MathModel_pj\TRAE-代码.md`（代码角色专属规范，重点「代码审核规则」）
- `E:\MathModel_pj\TRAE-规范.md`（重点 C1 代码头注释规范、A 执行规范、C8 代码加速决策树）

另核对数据证据文件（数字只取 pkl/报告，禁猜禁造）：

- `outputs/data/inventory-B.txt`（阶段 0.2 数据盘点报告）
- `outputs/data/clean-report-B.txt`（阶段 0.3 清洗报告，脚本已运行产物）

---

## 逐项结论（对照 8 个检查项）

### 1. 变量/索引：META_COLS 与特征列划分

**结论：通过。**

- 脚本第 47 行 `META_COLS = ["dataset_name", "disease"]`，与 inventory-B.txt 第 7 行「总列数 1333（元数据列 2 + 特征列 1331）」一致，元数据列确为 `dataset_name`、`disease` 两列。
- 第 55 行 `n_feat = n_cols - len(META_COLS)` = 1333 − 2 = 1331，与 inventory 特征列数一致。
- 第 56 行 `feat_cols = [c for c in df.columns if c not in META_COLS]` 按列名排除元数据列，推导结果与 META_COLS 口径一致（特征列名均为 `k__域|p__门|...|s__种` 7 级层级格式，见 inventory-B.txt 第 38-65 行，不会与 `dataset_name`/`disease` 冲突）。

**证据路径**：`clean-B.py:47,55-56`；`inventory-B.txt:7,38-65`。

### 2. dtype 转换

**结论：通过（float32 精度可接受）。**

- 第 69 行 `df_clean[feat_cols] = df_clean[feat_cols].astype("float32")` 仅对特征列转 float32，正确。
- 元数据列未做任何转换，保留字符串（脚本无对 `META_COLS` 的 category 转换），符合口径。
- float32 精度评估：相对丰度量纲 0–100（inventory-B.txt 第 33 行非零丰度 min=1e-05、max=79.9617）。float32 约 7 位有效数字，对 max=79.9617 绝对精度约 8e-6（保留约 5–6 位小数），对 min=1e-05 相对精度约 6e-8，均远小于微生物丰度测量本身的噪声（有效数字通常 3–4 位）。**精度损失可接受**；后续 1.4 若做 CLR/log 变换需留意低丰度端放大，但属 1.4 阶段职责，本阶段不涉及。

**证据路径**：`clean-B.py:69`；`clean-report-B.txt:18-20`（dtype 确认 float32/str）；`inventory-B.txt:33`。

### 3. 重复检测

**结论：通过。**

- 第 59 行 `df.duplicated(keep=False)` 默认对**全部列**（dataset_name + disease + 全部 1331 特征）判重，正是「完全重复」口径。
- 第 60 行 `assert n_dup == 0` 在检测到重复时正确中断（抛 AssertionError 并附提示信息）。
- 实测结果：inventory-B.txt 第 67-68 行「完全重复行数: 0」，clean-report-B.txt 第 6 行「完全重复行数: 0（assert 通过）」，一致。

**证据路径**：`clean-B.py:59-60`；`inventory-B.txt:67-68`；`clean-report-B.txt:6`。

### 4. NaN 检测

**结论：通过。**

- 第 63 行 `df.isna().sum().sum()` 对全表（所有行 × 所有列）求和，覆盖全表 NaN 检测。
- 第 64-65 行按列统计 NaN，仅用于报告；脚本无任何填补逻辑，符合「仅报告不填补」口径。
- 实测结果：clean-report-B.txt 第 9-10 行「NaN 总数: 0，无缺失值（符合预期）」。

**证据路径**：`clean-B.py:63-65`；`clean-report-B.txt:9-10`。

### 5. 落盘路径

**结论：通过。**

- 第 43 行 `OUT_PKL = ROOT / "outputs" / "data" / "c-data-cleaned.pkl"`，指向 `outputs/data/`，共享数据不带代号（符合 TRAE.md「数据文件命名体系」共享名规则）。
- 第 44 行 `OUT_REPORT` 同样指向 `outputs/data/`。
- 第 41 行 `ROOT = Path(__file__).resolve().parent.parent.parent` 用 pathlib 相对定位项目根，无硬编码盘符（符合 TRAE-规范.md A 节「路径可移植性」）。
- 实测：glob 确认 `c-data-cleaned.pkl` 与 `clean-report-B.txt` 实际落在 `outputs/data/`，ROOT 推导正确。

**证据路径**：`clean-B.py:41-44`；glob 结果 `outputs/data/c-data-cleaned.pkl`、`outputs/data/clean-report-B.txt`。

### 6. 幂等性

**结论：通过。**

- 第 52 行每次从头 `pd.read_pickle(RAW_PKL)` 读原始数据，第 68 行 `df.copy()` 后转换，第 73 行 `to_pickle` 覆盖写，无累积副作用。
- 全脚本无 `random`/`shuffle`/`seed` 等随机性来源，结果确定。
- 可重复运行（读 raw → 清洗 → 覆盖 pkl），符合幂等要求。

**证据路径**：`clean-B.py:52,68,73`。

### 7. 代码头注释 C1

**结论：通过（映射略显简略，基本达标）。**

- 目的（第 2-4 行）、原理（第 6-20 行，含 8 条清洗口径）、性能（第 22-23 行）、输入数据（第 25-27 行）、输出（第 29-31 行）、对应论文章节（第 33-34 行）六字段齐全。
- 「性能」字段声明「轻量-不适用（484 行 × 1333 列，秒级一次性清洗，无并行需求）」，符合 C1 要求（轻量脚本写"轻量-不适用"）。
- 「输入数据」含中文↔变量名映射：`dataset_name`=疾病数据集名、`disease`=疾病标签、特征列=物种级相对丰度（7 级分类学层级）。映射基本达标，但特征列以「物种级相对丰度」概括（1331 列逐一映射不现实，可接受）。

**证据路径**：`clean-B.py:1-35`。

### 8. 读回验证

**结论：基本充分（可补充 dtype/值验证）。**

- 第 76-78 行读回后验证 `shape` 与 `columns` 一致，覆盖了行数/列数/列名三个关键维度。
- 不足：读回验证未覆盖 **dtype**（读回后是否仍 float32）与 **值**（读回后数值是否一致）。第 81-82 行的 dtype 确认针对内存中的 `df_clean`，而非读回的 `df_back`。属可优化项，非正确性缺陷（pandas `to_pickle` 保留 dtype，转换不引入 NaN/重复）。

**证据路径**：`clean-B.py:76-82`。

---

## 问题清单

| 问题 | 严重度 | 证据路径 |
|------|--------|----------|
| 注释表述歧义：第 40 行「项目根目录 = 本脚本上两级」与代码 `.parent.parent.parent`（从文件算上三级）表述不一致；代码正确（ROOT 指向项目根，产物实际落在 `outputs/data/`），但注释「上两级」字面指文件上两级 = `outputs`，有歧义 | 低 | `clean-B.py:40-41` |
| `feat_cols` 与 `n_feat` 无一致性断言：`feat_cols` 按 `c not in META_COLS` 推导（第 56 行），`n_feat` 按 `n_cols - len(META_COLS)` 计算（第 55 行），两者口径一致的前提是 META_COLS 与实际元数据列完全匹配；若列名拼写错误或数据含额外元数据列会静默不一致，建议加 `assert len(feat_cols) == n_feat` | 低 | `clean-B.py:55-56` |
| 读回验证未覆盖 dtype 与值：第 76-78 行只验证 shape/列名，dtype 确认（第 81-82 行）针对内存 `df_clean` 而非读回 `df_back`，建议补充读回 dtype 与值一致性校验 | 低 | `clean-B.py:76-82` |

---

## 结论

**通过。**

8 个检查项全部通过，代码逻辑正确：META_COLS/特征列划分与数据盘点一致、dtype 转换正确且 float32 精度可接受、重复/NaN 检测口径正确、落盘路径合规（共享名 + pathlib 相对定位）、幂等、C1 头注释齐全、读回验证基本充分。

3 个低严重度建议（注释表述歧义、feat_cols/n_feat 一致性断言、读回 dtype/值验证）均不影响正确性，可后续优化，不阻断放行。

---

## 修复 diff 复审（追加，2026-08-16 规范）

针对问题清单 3 项低严重度建议，已全部修复并复跑通过：

| 修复项 | diff | 复审结论 |
|--------|------|----------|
| 注释表述歧义 | `clean-B.py` L40 注释「上两级」→「上三级」，与 `.parent.parent.parent` 一致 | 通过 |
| feat_cols/n_feat 一致性断言 | `clean-B.py` L56-59 新增 `assert len(feat_cols) == n_feat`，排除法推导与计算口径强制一致 | 通过 |
| 读回验证补 dtype 校验 | `clean-B.py` L76-82 新增 `dtype_mismatch` 列表比对 + `assert not dtype_mismatch`，读回对象 `df_back` 与内存 `df_clean` 逐列 dtype 一致才放行 | 通过 |

**复审结论**：3 项修复 diff 与问题清单一一对应，复跑通过（shape 仍 (484,1333)、特征列 float32、元数据列字符串、重复 0、NaN 0），无回归。**放行**。

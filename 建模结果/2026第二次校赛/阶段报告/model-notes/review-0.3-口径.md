# 阶段 0.3 基础清洗 · 第一遍审核（原理/口径审查）

- **角色**：建模子代理
- **模型**：deepseek-v4-pro:0813
- **审查对象**：`outputs/scratch/clean-B.py`
- **审查类型**：原理/口径审查（对照主建模裁定的 8 条清洗策略）

---

## 必读清单已读汇报

已 Read 并遵守以下文件：

- `E:\MathModel_pj\TRAE.md`（管线骨架：核心规则/门禁/交接协议/代号/角色边界）
- `E:\MathModel_pj\TRAE-建模.md`（建模角色专属规范，重点 0.3 节）
- `E:\MathModel_pj\TRAE-规范.md`（A 执行规范 / B 产出格式 / C1 代码头注释）
- `E:\MathModel_pj-2026-sim2-B\outputs\scratch\clean-B.py`（审查对象）
- `E:\MathModel_pj-2026-sim2-B\solution\domain-knowledge.md`（领域知识）
- `E:\MathModel_pj-2026-sim2-B\outputs\data\inventory-B.txt`（数据盘点报告）

---

## 逐项结论（对照 8 条清洗策略）

### 策略 1：重复行检测 + assert 无重复 + 结果打印

- **结论**：一致。
- **依据**：脚本 L58-60 用 `df.duplicated(keep=False)` 检测完全重复行（默认覆盖全部列，即 dataset_name + disease + 全部特征全同），`assert n_dup == 0` 断言无重复；L91 将重复行数打印进报告。与策略「盘点确认 0 重复，脚本仍做一次检测并断言，结果打印」完全一致。
- **证据路径**：`clean-B.py` L58-60、L91；`inventory-B.txt` L67-68（盘点已确认完全重复行数 0）。

### 策略 2：零值不填补、不删除

- **结论**：一致。
- **依据**：脚本全文无任何零值填补或删除逻辑，零值原样保留，符合「0 = 微生物未检出，真实稀疏值，不填补、不删除」。
- **证据路径**：`clean-B.py` 全文（无零值处理代码）；`domain-knowledge.md` L17（相对丰度为成分数据、稀疏零值需处理但推迟到建模）；`inventory-B.txt` L28-30（0 值占比 92.21%，高稀疏成分数据）。

### 策略 3：缺失值仅检测报告、不自动填补

- **结论**：一致。
- **依据**：脚本 L63-65 检测 NaN 总数与含 NaN 列；L95-101 若 `nan_total == 0` 打印「无缺失值」，否则打印异常信号与逐列 NaN 数，全程无填补操作。符合「检测 NaN——若存在仅打印报告（异常信号），不自动填补（预期 0）」。
- **证据路径**：`clean-B.py` L63-65、L95-101。

### 策略 4：特征列转 float32，元数据列保留字符串

- **结论**：一致。
- **依据**：脚本 L46-47 定义 `META_COLS = ["dataset_name", "disease"]`；L68-69 仅对 `feat_cols`（非元数据列）执行 `astype("float32")`，元数据列不做任何转换，保留原字符串（object）。符合「特征列转 float32；dataset_name、disease 保留原字符串（不转 category）」。
- **证据路径**：`clean-B.py` L46-47、L68-69、L81-82（dtype 确认逻辑）。

### 策略 5：不做任何变换

- **结论**：一致。
- **依据**：脚本全文无 CLR / 标准化 / 降维 / 特征筛选 / 标签构造逻辑，仅做盘点 + 类型标准化。头注释 L7-8 亦声明「不做任何统计变换，全部推迟到 1.4 预处理，遵循模型无关原则」。
- **证据路径**：`clean-B.py` L7-8（头注释）、全文（无变换代码）。

### 策略 6：无整体排除，small_adenoma 保留原样

- **结论**：一致。
- **依据**：脚本全文无任何行过滤/子群体排除逻辑，small_adenoma 样本原样保留，标签不动。符合「无整体排除；small_adenoma 保留原样（归健康对照，敏感性分析在 S1 做）」。
- **证据路径**：`clean-B.py` 全文（无过滤代码）；`inventory-B.txt` L19（Zeller 数据集中 small_adenoma 26 例归健康）。

### 策略 7：落盘 c-data-cleaned.pkl（共享数据，不带代号）

- **结论**：一致。
- **依据**：脚本 L43 `OUT_PKL = ROOT / "outputs" / "data" / "c-data-cleaned.pkl"`，文件名不带子问题代号，符合共享数据命名规范。
- **证据路径**：`clean-B.py` L43。

### 策略 8：清洗报告打印 stdout + 写 clean-report-B.txt

- **结论**：一致。
- **依据**：脚本 L44 `OUT_REPORT = ROOT / "outputs" / "data" / "clean-report-B.txt"`；L123 `print(report)` 打印 stdout，L124 `OUT_REPORT.write_text(report, encoding="utf-8")` 落盘。报告内容覆盖重复检测、NaN 报告、shape 前后对比、dtype 确认、清洗口径声明。
- **证据路径**：`clean-B.py` L44、L85-124。

---

## 问题清单

| 问题 | 严重度 | 证据路径 |
|------|--------|----------|
| `feat_cols` 用列名排除法（`c not in META_COLS`）而非位置切片确定特征列，若特征列名与元数据列名 `dataset_name`/`disease` 冲突会被误排除 | 低（实际特征列名为 `k__…` 分类学层级格式，不会与元数据列名冲突，属防御性提示） | `clean-B.py` L56 |
| 落盘读回验证（L76-78）仅校验 shape 与列名，未校验读回后 dtype 仍为 float32；dtype 确认（L81-82）基于内存对象 `df_clean` 而非读回对象 `df_back` | 低（pandas `to_pickle` 保留 dtype，实际无风险，属完整性提示） | `clean-B.py` L76-82 |

> 上述两项均为低严重度提示，不影响清洗口径正确性，不构成放行障碍。

---

## 结论

**通过**。

清洗脚本 `clean-B.py` 的清洗逻辑与主建模裁定的 8 条清洗策略**完全一致**，无自行扩展、无遗漏。零值/缺失值/类型标准化/变换/领域排除/落盘/输出各口径均严格对齐，且代码头注释（C1）与路径可移植性（A 节）合规。仅存在两项低严重度防御性提示，不阻断放行。

---

## 修复 diff 复审（追加，2026-08-16 规范）

针对问题清单中「读回验证未覆盖 dtype」一项（另一项 feat_cols 列名排除法为纯防御性提示，实际列名不冲突，无需改动），已做如下修复并复跑通过：

| 修复项 | diff | 复审结论 |
|--------|------|----------|
| 读回验证补 dtype 校验 | `clean-B.py` L76-82 新增 `dtype_mismatch` 列表比对 + `assert not dtype_mismatch`，读回对象 `df_back` 与内存 `df_clean` 逐列 dtype 一致才放行 | 通过（复跑无异常，dtype 校验生效） |

**复审结论**：修复 diff 与问题清单对应，复跑通过，口径未变（仍为 8 条策略一致）。**放行**。

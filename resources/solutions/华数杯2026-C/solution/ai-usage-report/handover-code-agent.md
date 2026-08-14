# 代码智能体交接：AI 辅助编程声明补注

> 依据：《华数杯章程》第十三条。提交截止：2026-08-10 20:00。
> 已有产出：论文 [COMP2026-C-final.pdf](../final-paper/COMP2026-C-final.pdf)（41 页）、AI 使用详情 [AI工具使用详情.pdf](AI工具使用详情.pdf)（4 页）。

## 任务：为全部 .py 脚本追加 AI 辅助声明

### 目标文件

`outputs/scratch/` 下 **54 个 .py 文件**，完整列表见下方。每个文件顶部追加以下注释块：

```python
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。
```

### 插入规则

1. 在文件顶部的 `""" ... """` 文档字符串**之上**（文件名/首行注释之下），插入一个空行 + 上述 2 行注释
2. 若文件无文档字符串，直接放在文件最顶部
3. 若文件已有相同声明（含 `AI工具辅助` 字样），跳过
4. **不修改文件任何其他内容**（代码、正文注释、文档字符串均保持原样）

### 示例

```python
"""
目的：
    阶段 2.1 S4 主模型 — 层 2 每区域独立 0-1 MILP 求解

原理：
    ...
"""
```

→ 变为 →

```python
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 2.1 S4 主模型 — 层 2 每区域独立 0-1 MILP 求解

原理：
    ...
"""
```

### 完整文件清单（54 个）

```
outputs/scratch/
├── arch-sub1s3.py
├── baseline-sub4-heuristic.py
├── chart-d-marginal.py
├── charts-sub1-v2.py
├── charts-sub1-whitenoise.py
├── charts-sub3.py
├── charts-sub4-lns.py
├── charts-sub4-v2.py
├── charts-sub4.py
├── check-baseload-bottleneck.py
├── check-d-marginal.py
├── check-latency.py
├── data-cleaning-03.py
├── data-inventory-02.py
├── data-profiling-04.py
├── decompose-sub4-cost.py
├── dryrun-sub4-ratio.py
├── extract-c-problem-text.py
├── gen-sub2-model-notebook.py
├── gen-verify-sub1-notebook.py
├── inspect-c-data.py
├── inspect-lns-progress.py
├── inspect-s4-charts.py
├── inspect-sub4-corner.py
├── inspect-sub4-cost.py
├── inspect-sub4-reduction.py
├── inspect-sub4-sensitivity.py
├── lns-sub4.py
├── preprocess-sub1.py
├── preprocess-sub2.py
├── preprocess-sub3.py
├── preprocess-sub4-clean.py
├── preprocess-sub4.py
├── probe-observations.py
├── profile-alpha-milp.py
├── render-sub4-previews.py
├── review-m10-s3nostorage.py
├── review-q3-s1costbase.py
├── run-verify-sub1-notebook.py
├── run-verify-sub2-model-notebook.py
├── scan-d-corners.py
├── scan-sub2-epsilon.py
├── sens-scan-sub4.py
├── sensitivity-minutil-sub2.py
├── sub1-model.py
├── sub2-model.py
├── sub3-model.py
├── sub4-clean-model.py
├── sub4-model.py
├── verify-sub1-b2.py
├── verify-sub2-capacity.py
├── verify-sub2.py
├── verify-sub3.py
├── verify-sub4-issues.py
```

### 验证

完成后，随机抽查 5 个文件确认声明格式正确（两行注释、工具名称、版本号、机构、日期完整）。

### 可选附加任务

- `outputs/notebooks/*.ipynb` 中如有 Python 代码 cell，在其第一个 cell 顶部补相同声明
- `archive/` 子目录下的旧版代码不处理

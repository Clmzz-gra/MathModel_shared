# 门禁 M 审查结论：S2 1.4 预处理

> 审查代理角色：代码对话（coding Preset）· 汇聚门禁 M 审查代理（1.4 预处理侧）
> 模型：deepseek-v4-pro:0813
> 审查日期：2026-08-21
> 审查对象：`outputs/scratch/preprocess-S2.py`、`outputs/data/S2-preprocessed.pkl`、`outputs/data/preprocess-report-S2.txt`
> 对照材料：`solution/model-notes/handoff-S2-code-agent.md`、`solution/model-notes/approach-S2-confirmed.md`
> 审查方式：只读审查（未修改任何审查对象与对照材料）

---

## 必读清单已读汇报

开工前已完整 Read 以下规范文件并遵守其中规则：

- `TRAE-代码.md`（代码审查规则：两遍审核/审查包构造/1.4 预处理阶段定义）
- `TRAE.md`（门禁 M 判定内容、审查/裁决分离协议、四门禁总览）
- `TRAE-规范.md` **C1 代码头注释**、**C8 代码加速决策树** 节

管线进度确认：`git log --oneline -10` 显示 1.4 已完成（`37a9b6b`）、2.0 已完成（`44612e0`），当前处于汇聚门禁 M 审查点。`review-M-S2-14.md` 此前不存在，本次为完整执行。

---

## 判定内容逐项结论

### 1. 预处理与模型匹配 —— 通过

**结论**：`S2-preprocessed.pkl` 字段结构与 handoff 规格一致，标签/264 特征集/CLR/特征名七级元数据/折索引五类要素齐全。

**依据**：
- pkl 顶层键为 `meta` / `feature_names` / `feature_taxonomy` / `per_disease`，`per_disease` 含 CRC/IBD/Obesity 三病。
- `meta` 含 `filter_threshold=0.95`、`clr_delta=6.5e-06`、`n_features_before=1331`、`n_features_after=264`、`cv={n_splits:5, shuffle:True, seed:42}`、`field_semantics`（内嵌 y/X_raw/X_clr/cv_folds/feature_taxonomy 字段语义，符合 TRAE-代码.md 2.1 第 4 条「pkl 落盘 meta 内嵌语义」精神）。
- `feature_names` 长度 264；`feature_taxonomy` 七级（k/p/c/o/f/g/s）各 264 项；每病含 `X_raw`/`X_clr`/`y`/`cv_folds`/`n_samples`/`n_pos`/`n_neg`。

**证据路径**：`outputs/data/S2-preprocessed.pkl`（实测结构）、`outputs/scratch/preprocess-S2.py` L161-184。

### 2. 数据正确性 —— 通过（含 1 项 B 级口径歧义，见问题清单）

**结论**：过滤 1331→264、CLR（δ=6.5e-06）、标签映射三项均正确。

**依据**：
- **过滤**：实测「全 484 样本一次过滤（零值占比>0.95 剔除）」得 264/1331，与 `preprocess-report-S2.txt`（剔除 1067、保留 264）及 approach §1.2 预期「1331→264」一致。
- **CLR**：用 δ=6.5e-06 独立重算，与 pkl 内 `X_clr` 逐元素最大差 = 0.0（完全一致）；三病 CLR 行均值 ≈ 0（-2.6e-15 ~ -3.0e-15），符合几何均值中心化性质。δ 取 6.5e-06 与 proxy-replacement-checklist-S2.md P6（与 S1 一致）相符，未误用 utils.py 的 `minpos/2` 口径。
- **标签映射**：独立重算三病标签与 pkl 完全一致（CRC pos=48/neg=73、IBD pos=25/neg=85、Obesity pos=164/neg=89）。CRC 的 `small_adenoma`（26 例）按题面口径归健康（`healthy=["n","small_adenoma"]`），符合 R4「S1 主口径未定前按题面口径归健康」。

**证据路径**：`outputs/data/preprocess-report-S2.txt`、`outputs/scratch/preprocess-S2.py` L54-77、L80-86、`solution/model-notes/proxy-replacement-checklist-S2.md` P6、`solution/model-notes/approach-S2-confirmed.md` §9 R4。

### 3. 折划分防泄漏 —— 通过

**结论**：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` 实现正确，分层性达标、无泄漏。

**依据**：
- 三病各 5 折，每折 train/test 索引无重叠（实测 overlap=0）。
- 分层性：各折 test 患病比例与整体一致（CRC 0.375~0.417 vs 整体 0.397；IBD 0.227 vs 0.227；Obesity 0.64~0.66 vs 0.648）。
- 折索引落盘 `cv_folds`，供 2.1 折内 Lasso 防泄漏复用，符合 handoff §1.4「分层 CV 折内选择（防泄漏）」要求。

**证据路径**：`outputs/scratch/preprocess-S2.py` L139-140、`outputs/data/S2-preprocessed.pkl`（实测折划分）。

### 4. 代码质量 —— 通过

**结论**：C1 头注释完整（含性能字段）、幂等、路径可移植。

**依据**：
- **C1 头注释**：含「目的/原理/性能/输入数据/输出/对应论文章节」六字段齐全；「性能」字段声明「轻量-不适用（纯数据搬运 + 向量化 CLR + 5 折划分，秒级，无并行需求；bootstrap/Lasso 重计算留给 2.1）」，符合 C8 决策树「0 轻量」路径，无单核红线问题。
- **幂等性**：脚本读 `c-data-cleaned.pkl`、写 `S2-preprocessed.pkl` 与报告，无随机性依赖（仅 meta.generated 时间戳随运行变化，不影响数据），可重复运行。
- **路径可移植性**：`ROOT = Path(__file__).resolve().parent.parent.parent` 相对定位项目根，无硬编码盘符/绝对路径，符合 TRAE-规范.md A 节「路径可移植性」。

**证据路径**：`outputs/scratch/preprocess-S2.py` L1-37（头注释）、L48-51（路径）。

### 5. 特征名元数据完整性 —— 通过

**结论**：264 特征的 k__p__c__o__f__g__s__ 七级拆分字段齐全且拆分正确。

**依据**：
- `feature_taxonomy` 七级（k/p/c/o/f/g/s）各 264 项，长度与 `feature_names` 一致。
- 实测 264 特征名全部含 `k__` 前缀（0 个缺失）；逐特征逐级重拆分与 pkl 内 `feature_taxonomy` 比对，不一致数为 0。
- 样例 `k__Archaea|p__Euryarchaeota|...|s__Methanobrevibacter_smithii` 七级拆分正确，可供属级聚合与标志物解读。

**证据路径**：`outputs/scratch/preprocess-S2.py` L89-99、L115-119、`outputs/data/S2-preprocessed.pkl`（实测拆分比对）。

---

## 问题清单

| # | 问题 | 严重度 | 证据路径 |
|---|------|--------|---------|
| 1 | **过滤口径表述歧义**：handoff §1.1 写「每病独立计算零值占比」，代码实现为「三病并集（全 484 样本一次过滤）」。两者结果不同（全样本一次过滤=264；每病独立取并集=315）。代码口径与 approach §1.2 预期「1331→264」及 A 类验证 V2（`verify-S2-v2-zerobin.py` L137-141 同为全样本一次过滤）一致，故**不影响数值正确性**，但 handoff §1.1 与 approach §4 的「每病独立」表述存在歧义，建议建模对话澄清 handoff 表述（统一为「三病并集统一口径」）。 | B 级（表述歧义，不影响正确性） | `solution/model-notes/handoff-S2-code-agent.md` §1.1、`solution/model-notes/approach-S2-confirmed.md` §4 步骤1、`outputs/scratch/preprocess-S2.py` L107-112 |

---

## 结论

**通过。**

S2 1.4 预处理五项判定内容全部通过：pkl 字段与 handoff 规格匹配、过滤/CLR/标签映射数据正确、StratifiedKFold 折划分防泄漏正确、代码质量（C1 头注释/幂等/路径可移植）达标、特征名七级元数据完整。唯一问题为 B 级过滤口径表述歧义（handoff §1.1「每病独立」与实现「三病并集」字面不符），代码实现与 approach 预期及 V2 验证一致，不影响正确性，建议建模对话在门禁 M 裁决时一并澄清 handoff 表述。

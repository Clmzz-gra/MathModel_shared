# 门禁 M 审查结论：S3 1.4 预处理（review-M-S3-14）

> 审查对象：`outputs/scratch/preprocess-S3.py`、`outputs/data/S3-preprocessed.pkl`、`outputs/data/preprocess-report-S3.txt`
> 对照材料：`solution/model-notes/handoff-S3-code-agent.md`（规格）、`solution/model-notes/approach-S3-confirmed.md`（方案）
> 审查角色：代码对话（coding Preset）审查代理 | 模型：deepseek-v4-pro:0813 | 思考强度：max
> 审查日期：2026-08-21 | 门禁：汇聚门禁 M（1.4 预处理侧，与 2.0 推导合并呈递）

## 必读清单已读汇报

已完整 Read 以下规范文件并遵守其规则：
- `TRAE.md`（管线骨架：门禁 M 判定内容 / 审查裁决分离 / 空洞审查检出 / 代号）
- `TRAE-代码.md`（代码审查规则 / 1.4 预处理 / 1.5 方案代码审查）
- `TRAE-规范.md` C1（代码头注释规范）、C8（代码加速决策树）、A（执行规范）

管线进度确认：`git log --oneline -10` 显示 1.4 已完成（`06b19ac`）、2.0 已完成（`1b4102e`）、2.3 报告骨架段（`16110f3`）。当前处于汇聚门禁 M 审查点。`review-M-S3-14.md` 此前不存在，本次完整执行。

---

## 判定内容逐项结论

### 1. 预处理与模型匹配

**结论：通过。**

- **依据**：`S3-preprocessed.pkl` 顶层字段与 handoff §三「数据接口」+ §2.2「模型规格」完全对齐。核验脚本（`outputs/scratch/_review_M_verify.py`，只读）逐字段确认：
  - `X_filtered`：(484, 264) 过滤后物种级丰度（原始，未 CLR）
  - `y`：(484,) 二分类标签；`dataset_name`/`disease`：(484,) 元数据
  - `feature_names`：264 过滤后特征名；`feature_taxonomy`：264 特征名 → 七级分类学拆分（k/p/c/o/f/g/s）
  - `lodo_combos`：C1/C2/C3 三组合样本索引（train_idx/test_idx/train_datasets/test_dataset/test_disease）
  - `shared_features`：252 共享特征交集；`genus_features`：106 属级聚合特征名；`phylum_features`：11 门级聚合特征名
  - `meta`：sub=S3、stage=1.4、source=c-data-cleaned.pkl、seed=42、clr_delta=6.5e-6、detection_limit=1e-5、filter_rule、label_rule、field_semantics（内嵌字段语义，符合 TRAE-代码.md 2.1 第 4 条「pkl 落盘 meta 内嵌语义」精神）
- **证据路径**：`outputs/data/S3-preprocessed.pkl`（核验输出见本审查执行日志）；`outputs/scratch/preprocess-S3.py` L210-222（payload 组装）、L186-208（meta 组装）。

### 2. LODO 划分正确性

**结论：通过。**

- **依据**：三组合样本索引与 handoff §2.1「LODO 协议」口径完全一致，且「测试疾病完全不可见」硬约束成立：
  - C1：训练 363（metahit 110 + Chatelier 253）/ 测试 121（Zeller/CRC）✓
  - C2：训练 374（Zeller 121 + Chatelier 253）/ 测试 110（metahit/IBD）✓
  - C3：训练 231（Zeller 121 + metahit 110）/ 测试 253（Chatelier/Obesity）✓
  - 三组合均核验：训练集不含测试疾病样本（`train_has_test=False`）、训练/测试索引无重叠、训练+测试覆盖全部 484 样本。
- **证据路径**：`outputs/scratch/preprocess-S3.py` L72-76（COMBOS 定义）、L157-167（索引生成）；`outputs/data/preprocess-report-S3.txt` L8-11（样本数报告）；核验脚本 L55-75（逐组合核验输出）。

### 3. 数据正确性

**结论：通过。**

- **依据**：
  - **过滤 1331→264**：核验脚本对照源数据 `c-data-cleaned.pkl` 重算零值占比>95% 剔除，得 264，与 pkl `feature_names` 完全一致（`set 相等=True`）。三病并集统一口径（对全部 484 样本计算零值占比），与 handoff §三「特征列」及 S1/S2 口径一致。
  - **CLR δ=6.5e-6**：`clr_delta = 0.65 × 检出限(1e-5) = 6.5e-6`，与 handoff §2.2 一致。CLR 为逐样本变换（零值乘法替换 δ → log → 减行均值），无跨样本参数，不引入训练/测试泄漏（`preprocess-S3.py` L87-101）。
  - **标签映射**：患病=1（cancer/ibd_ulcerative_colitis/ibd_crohn_disease/obesity），健康=0（n/small_adenoma/leaness）。核验 `y` 与题面口径一致（`np.array_equal=True`），患病总数 237 = 48+21+4+164 ✓。**small_adenoma（26 例）按题面口径归健康**，与 handoff §三「患病判定口径」R4 一致（未选定 S1 主口径前按题面归健康）。
- **证据路径**：`outputs/scratch/preprocess-S3.py` L57-59（δ 定义）、L69（POSITIVE_LABELS）、L82-84（binary_label）、L147-151（过滤）；核验脚本 L49-53（标签核验）、L77-90（过滤/CLR 核验）。

### 4. 策略支撑数据

**结论：通过。**

- **依据**：
  - **属级聚合 106 属**：`taxonomy_aggregate(X_filtered, "genus")` 按 `g__` 段聚合（同属丰度求和），核验重算得 106，与 pkl `genus_features` 一致（`set 相等=True`）。
  - **门级聚合 11 门**：按 `p__` 段聚合，核验重算得 11，与 pkl `phylum_features` 一致。
  - **共享特征交集 252**：在过滤后 264 特征集内按特征名交集重算（存在=平均丰度>0），核验重算得 252，与 pkl `shared_features` 一致。与 handoff §2.4 一致（「正式实现基于过滤后 264 特征集按特征名交集重算，数量可能略减」——A 类验证在 1331 全集测得 344，正式实现基于 264 重算得 252，符合预期）。
  - **口径与 A 类验证一致**：共享交集仅用特征存在性（`present_features` 用 `X[dataset_mask].mean(axis=0) > 0`，绝不用测试集标签），符合 B5 裁定「转导式边界」（用测试集特征、绝不用测试集标签）。
- **证据路径**：`outputs/scratch/preprocess-S3.py` L104-121（taxonomy_aggregate）、L136-139（present_features）、L169-180（共享交集/聚合）；核验脚本 L92-119（重算核验）。

### 5. 代码质量

**结论：通过。**

- **依据**：
  - **C1 头注释完整**：`preprocess-S3.py` L1-44 含全部强制字段——目的（L2-6）、原理（L8-27，含标签映射/过滤/CLR/聚合/共享交集/LODO 的公式与逻辑推导）、**性能（L29-30：「轻量-不适用（484×1331 小数据，纯向量化过滤/聚合/集合运算，秒级，无并行需求）」）**、输入数据（L32-36，含「中文指标↔代码变量名」映射：dataset_name/disease/1331 物种级特征）、输出（L38-40）、对应论文章节（L42-43）。
  - **C8 决策树合规**：脚本声明「轻量-不适用」，符合 C8 Q0（484×1331 小数据、纯向量化、秒级）→ 常规路径，不触发单核红线。性能声明存在，满足 C1/C8 机械检查项。
  - **幂等性**：脚本从 `c-data-cleaned.pkl` 确定性读取，无随机状态（seed=42 仅记录于 meta，不参与计算），重跑产出相同 pkl/report。
  - **路径可移植性**：`ROOT = Path(__file__).resolve().parent.parent.parent`（L52），无硬编码盘符/绝对路径，符合 TRAE-规范.md A「路径可移植性」。
- **证据路径**：`outputs/scratch/preprocess-S3.py` L1-44（头注释）、L52-55（路径）、L142-265（main 幂等逻辑）。

---

## 问题清单

| # | 问题 | 严重度 | 证据路径 |
|---|---|---|---|
| 1 | （观察项，非缺陷）pkl 存储的 `X_filtered` 为**未 CLR 的原始过滤后丰度**，CLR 变换函数 `clr_transform` 在 1.4 中仅定义未调用，实际 CLR + StandardScaler 推迟到 2.1 按折内（训练集）执行。这与 handoff §2.2「预处理：过滤→CLR→StandardScaler」的表述存在**阶段归属差异**，但设计合理（CLR 逐样本无泄漏、StandardScaler 必须折内估计，二者在 2.1 折结构已知处统一执行更安全），meta.note 已显式说明「2.1 通过 importlib 导入复用」。**不阻断**，仅提示 2.1 实现时须确认 CLR/StandardScaler 折内口径。 | 低（观察项） | `preprocess-S3.py` L87-101（函数定义）、L212（X_filtered 注释「原始，未 CLR」）、L197-199（meta.note）；`handoff-S3-code-agent.md` L52 |
| 2 | （观察项，非缺陷）过滤后 264 特征的行丰度和 min=40.86（原 1331 特征行和≈100），因近全零特征被剔除后部分样本保留丰度下降。CLR 为尺度不变变换（减行均值），不影响结果，非 bug。 | 低（观察项） | 核验脚本 L127-129（行和 min/max/mean=40.86/100.00/98.78） |

> 无高严重度问题。上述两条均为「观察项」，不构成正确性缺陷，不阻断门禁放行。

---

## 结论：**通过**

S3 1.4 预处理产出（`preprocess-S3.py` + `S3-preprocessed.pkl` + `preprocess-report-S3.txt`）与 handoff 规格、approach 方案完全一致：字段结构、LODO 三组合划分（363/110、374/121、231/253，测试疾病完全不可见）、过滤 1331→264、CLR δ=6.5e-6、标签映射（small_adenoma 归健康）、属级 106/门级 11/共享交集 252 策略支撑数据、C1 头注释（含性能字段）与 C8 决策树合规、幂等性与路径可移植性均核验通过。无高严重度问题，仅两条低严重度观察项（CLR 阶段归属、过滤后行和下降），不阻断放行。

**建议**：门禁 M 的 1.4 预处理侧放行，待 2.0 推导侧（modeling preset 审查代理）结论合并后由主建模裁决。

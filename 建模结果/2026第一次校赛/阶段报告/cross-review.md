# 阶段 3.0 跨子问题审查报告

> 2026-07-28 | 审查人: AI Agent
> 审查范围: math-sub1~5.tex, approach-sub1~5-confirmed.md, c2_q1q4_model.py, c2_q2q3_model.py, c2_q5_model.py, c3_generate_charts.py

---

## 一、审查结论总览

| 类别 | 发现数 | 阻塞 | 建议修改 | 记录即可 |
|------|:------:|:----:|:--------:|:--------:|
| 符号体系不一致 | 2 | 0 | 1 | 1 |
| 推导与代码不符 | 2 | 2 | 0 | 0 |
| 跨问题衔接 | 0 | 0 | 0 | 0 |
| 假设一致性 | 通过 | — | — | — |

**总体评价**: 发现 2 个关键性代码-推导不一致问题需修复，其余通过。

---

## 二、关键问题（需修复）

### 🔴 问题 1：标准分按"列"计算而非按"评委"计算

**影响范围**: Q1 全部结果、Q5 全部结果

**问题描述**:

数学推导 ([math-sub1.tex](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/solution/model-notes/math-sub1.tex)) 定义标准分为按评委计算：

$$
z_{j,i} = \frac{x_{j,i} - \bar{x}_j}{\sigma_j}
$$

其中 $\bar{x}_j$ 为评委 $j$ 评阅**所有论文**的平均分，$\sigma_j$ 为其标准差。

但代码实现按**列位置**计算：

```python
# c2_q1q4_model.py 第 34-39 行
for j in range(1, 5):
    sc = tdata[f'打分{j}'].values       # 取第j列全部打分
    mu, sigma = sc.mean(), sc.std()     # 该列均值和标准差
    z_df[f'z{j}'] = (sc - mu) / sigma   # 按列z-score
```

两处代码均存在此问题：
- [c2_q1q4_model.py L34-39](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/outputs/scratch/c2_q1q4_model.py#L34-L39)
- [c2_q5_model.py L28-32](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/outputs/scratch/c2_q5_model.py#L28-L32)
- [c3_generate_charts.py L29-32](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/outputs/scratch/c3_generate_charts.py#L29-L32)

**为何错误**: 评委 ID 在 4 个列中分散分布（如专家A01 可能在论文1的评委1、论文2的评委3位置），按列混合了不同评委的评分。均值 $\mu$ 是"该列所有评委的混合平均"，不是"该评委个人的平均"。

**影响程度**: Spearman $\rho$ 值、ROC AUC、筛选命中率等 Q1 全部数值结果，以及 Q5 敏感性/加权对比结果均会变化。

**修复方案**:
1. 按评委 ID 分组计算每组的 $\mu_j$ 和 $\sigma_j$
2. 对每篇论文的 4 个评委分别查各自的 $\mu_j$ 和 $\sigma_j$ 做 z-score
3. 取 4 个 z-score 的均值作为网评标准分
4. 重新运行 Q1、Q5 代码，更新图表

---

### 🔴 问题 2：信度维度用 Spearman 替代了 ICC

**影响范围**: Q2 信度指标、Q3 TOPSIS 得分、所有涉及评委排名的结果

**问题描述**:

数学推导 ([math-sub2.tex L25-L36](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/solution/model-notes/math-sub2.tex#L25-L36)) 定义信度为成对 ICC(2,1) 均值：

$$
\text{ICC}_{jk} = \frac{\sigma^2_{\text{between}}}{\sigma^2_{\text{between}} + \sigma^2_{\text{within}}},\quad
R_j^{\text{rel}} = \frac{1}{K-1}\sum_{k\neq j}\text{ICC}_{jk}
$$

但 [c2_q2q3_model.py L57-67](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/outputs/scratch/c2_q2q3_model.py#L57-L67) 使用的是成对 Spearman 相关系数：

```python
r, _ = spearmanr(common['打分_j'], common['打分_o'])
pair_rhos.append(r)
reliability = np.mean(pair_rhos)
```

Spearman 只衡量秩一致性（涨跌方向），不衡量绝对评分一致性。例如：评委 A 打 [70,80,90]，评委 B 打 [60,70,80]——Spearman=1.0（完美秩相关），但实际评分低 10 分。ICC 会同时捕捉这种系统偏移。

**影响程度**: 所有评委的信度得分值被系统性高估（因为 Spearman 只要求方向一致），进而影响 TOPSIS 排序和聚类分层结果。Q3 熵权分布也会变化。

**修复方案**:
1. 实现成对 ICC(2,1) 计算（使用 `pingouin.intraclass_corr` 或手写公式）
2. 将信度从 Spearman 均值替换为 ICC 均值
3. 重新运行 Q2/Q3 代码，更新所有 Q2/Q3/Q4/Q5 图表

---

## 三、中等问题（建议修改）

### 🟡 问题 3：符号 $z$ 在 Q1 与 Q3 中含义冲突

| 子问题 | 符号 | 含义 |
|--------|------|------|
| Q1 | $z_{j,i}$ | 标准分（z-score），评委 j 对论文 i |
| Q3 | $z_{ij}$ | 归一化指标值，评委 i 在维度 j |

论文撰写时若同时引用 Q1 和 Q3 的公式，读者容易混淆。

**建议**: Q3 中将归一化矩阵记为 $\mathbf{R} = [r_{ij}]$（rating）或 $\mathbf{N} = [n_{ij}]$（normalized indicator）。

---

### 🟡 问题 4：下标惯例不一致

| 子问题 | i 的含义 | j 的含义 |
|--------|----------|----------|
| Q1, Q2 | 论文索引 | **评委索引** |
| Q3 | **评委索引** | 维度索引 |
| Q4 | 组内观察值索引 | — |
| Q5 | — | **评委索引**（$w_j$） |

Q1/Q2 用 $j$ 表示评委，Q3 用 $i$ 表示评委，读者可能困惑"$i$ 和 $j$ 到底哪个是评委"。

**建议**: 统一为 $i$ = 评委索引，$j$ = 维度/论文索引。

---

## 四、通过项（记录即可）

### ✅ 假设一致性：全部通过

| 假设 | Q1 | Q2 | Q3 | Q4 | Q5 | 一致？ |
|------|:--:|:--:|:--:|:--:|:--:|:------:|
| 最终奖项 = 真实水平代理 | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| 4 评委 per paper | ✓ | — | — | — | ✓ | ✓ |
| 五题评委互斥（不跨题） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 标准分消除尺度差异 | ✓ | — | ✓ | — | ✓ | ✓ |
| PF-005 效应量报告规范 | ✓ | ✓ | — | ✓ | — | ✓ |

### ✅ 数据流衔接：全部通过

- **Q2 → Q3**: Q2 输出 $\mathbf{Z}$（四维度归一化矩阵）→ Q3 输入。代码中 `jdf[feature_cols]` 的 Min-Max 归一化正确实现。
- **Q3 → Q4**: Q3 输出 TOPSIS 得分 → Q4 用 KW+Dunn 检验。代码通过 `q3-judge-scores.pkl` 传递，衔接正确。
- **Q3 → Q5**: Q3 输出 TOPSIS 得分 → Q5 用于评委加权。代码中 `topic_q3['TOPSIS得分']` 读取正确。
- **Q1 → Q5**: Q1 的 Spearman $\rho$ 作为 Q5 的"现状基准"。逻辑链一致。

### ✅ 方法确认书与代码一致：通过

各 approach-sub*-confirmed.md 描述的方法均与对应代码实现一致（除问题 1、2 外）。

### ✅ 已知局限透明标注：通过

- Q4 坦诚"题目效应与评委池效应无法分离"——在 approach-sub4-confirmed.md 和 math-sub4.tex 中均有记录。
- Q2 标注 ICC 对小样本不稳定——approach-sub2-confirmed.md 中记录。
- 异常评委标记策略（保留不排除）在 Q3 中正确贯彻。

---

## 五、修复优先级

| 顺序 | 问题 | 修复内容 | 重跑范围 |
|:----:|------|----------|:--------:|
| **1** | 问题 2: 信度→ICC | 修改 `c2_q2q3_model.py` L57-67 | Q2→Q3→Q4 全部图表 |
| **2** | 问题 1: z-score按评委 | 修改 `c2_q1q4_model.py` L34-39 + `c2_q5_model.py` L28-32 + `c3_generate_charts.py` L29-32 | Q1、Q5 全部图表 |
| 3 | 问题 3+4: 符号统一 | 修改 math-sub3.tex 符号 | 论文撰写阶段处理 |

> **注意**: 问题 1 和 2 修复后，Q5 的硬编码数值（c3_generate_charts.py L320-321, L333-334, L348）需用新计算结果替换。

---

## 六、审查后状态

- [x] 问题 1 修复（z-score 按评委）— c2_q1q4_model.py + c2_q5_model.py + c3_generate_charts.py
- [x] 问题 2 修复（信度改用 ICC）— c2_q2q3_model.py
- [x] 全部图表重新生成 + 多模态审查 — 20/20 通过
- [x] q3-cluster-scatter 图例边框修复 + 重审通过
- [x] pca-scree、cluster-tsne 补充多模态审查通过
- [ ] 问题 3+4 在论文中处理（非阻塞）

**状态**: 阶段 3.0 完成。修复后代码审查通过，全部 20 张图表已通过多模态审查（"通过"级别）。

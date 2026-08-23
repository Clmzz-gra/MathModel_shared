# 代码深度审查报告

> 2026-07-28 | 审查焦点：代理值一致性、方案书匹配、方法正确性

---

## 审查范围

| 文件 | 对应问题 | 行数 |
|------|---------|------|
| [c2_q1q4_model.py](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/outputs/scratch/c2_q1q4_model.py) | Q1, Q4 | 134 |
| [c2_q2q3_model.py](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/outputs/scratch/c2_q2q3_model.py) | Q2, Q3 | 271 |
| [c2_q5_model.py](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/outputs/scratch/c2_q5_model.py) | Q5 | 120 |
| [c3_generate_charts.py](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/outputs/scratch/c3_generate_charts.py) | 图表 | 376 |
| [iter-02-comprehensive.tex](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/solution/internal-reports/iter-02-comprehensive.tex) | 内部报告 | 206 |

---

## 一、关键问题（需立即修复）

### 🔴 问题 1：效度只计算入围论文，排除淘汰论文

**位置**: [c2_q2q3_model.py L100-102](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/outputs/scratch/c2_q2q3_model.py#L100-L102)

**问题描述**：

数学推导 ([math-sub2.tex L44-49](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/solution/model-notes/math-sub2.tex#L44-L49)) 明确定义奖项等级为 4 级：未入围=0、三等=1、二等=2、一等=3：

```
将论文按评分排序得秩次，按奖项等级（未入围=0、三等=1、二等=2、一等=3）排序
```

但代码只对 `奖项数值.notna()` 的论文计算 Spearman：

```python
graded = j_scores[j_scores['奖项数值'].notna()]  # 只含入围论文（约42%）
rho, _ = spearmanr(graded['打分'], graded['奖项数值'])
```

**后果**：
- 淘汰论文（58%）被完全排除在信度计算外
- 一位评委可能对所有淘汰论文打高分——这应该降低效度——但不会被检测到
- 效度只衡量"评委在入围论文内部的排序能力"，而非"评委能否区分入围vs淘汰"

**修复**：奖项数值对淘汰论文填充为 0：
```python
j_scores['奖项数值'] = j_scores['奖项数值'].fillna(0)
```

**影响范围**：Q2 效度指标 → Q3 TOPSIS 得分 → Q4/Q5 全部下游结果

---

### 🔴 问题 2：内部报告数值全部过时

**位置**: [iter-02-comprehensive.tex](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/solution/internal-reports/iter-02-comprehensive.tex)

**问题描述**：

综合内部报告编写于 ICC 修复和 z-score 修复**之前**，所有数值均基于旧的（错误的）计算方法。修复后全部数值已变化：

| 指标 | 报告值 | 当前值 | 差异 |
|------|--------|--------|------|
| Q1 全题 ρ | 0.737 | 0.797 | +0.060 |
| Q1 命中率 | 71.3% | 74.8% | +3.5pp |
| Q1 假阴性率 | 7.8% | 3.3% | -4.5pp |
| Q1 ROC AUC | 0.893 | 0.932 | +0.039 |
| Q3 熵权分布 (5题) | 全部过时 | 全部变化 | — |
| Q3 K-means 分层结构 | 全部过时 | 全部变化 | — |
| Q3 稳健性 r | 0.857-0.964 | 0.851-0.971 | 轻微 |
| Q4 H 统计量 | 12.984 | 11.710 | -1.274 |
| Q4 η² | 0.047 | 0.040 | -0.007 |
| Q4 B-E p_adj | 0.011 | 0.030 | +0.019 |
| Q5 敏感性 ρ 列 | 过时 | 已更新 | — |
| Q5 加权 ρ 列 | 过时 | 已更新 | — |

**此外**，第 60 行仍写"以成对 Spearman 相关系数均值量化"——但代码已改为 ICC(2,1)。

**修复**：用当前运行结果全面刷新报告。

---

### 🔴 问题 3：Q5 敏感性分析用秩混合替代分值混合

**位置**: [c2_q5_model.py L54-58](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/outputs/scratch/c2_q5_model.py#L54-L58)

**问题描述**：

数学推导 ([math-sub5.tex L31-33](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/solution/model-notes/math-sub5.tex#L31-L33)) 使用分值混合：

$$S_{\text{final}}(\alpha) = \alpha \cdot \bar{z}_{\text{web}} + (1-\alpha) \cdot \bar{z}_{\text{central}}$$

代码用秩混合替代：

```python
ideal_rank = q5df.groupby('题目')['奖项数值'].transform(lambda x: x.rank())
web_rank = q5df.groupby('题目')['z_mean'].transform(lambda x: x.rank())
final_rank = alpha * web_rank + (1-alpha) * ideal_rank
rho, _ = spearmanr(final_rank, q5df['奖项数值'])
```

**为何不同**：
- 奖项数值只有 3 个取值 (1,2,3)，大量并列。排名法将并列映射为均值秩
- 分值混合保留间距信息（如 z=1.5 vs z=0.3 差距 > z=0.8 vs z=0.7），秩混合丢失此信息
- 用 Spearman 评估秩混合结果：Spearman 本身基于秩，用秩输入再用 Spearman 输出是冗余的——中间的 α 变化对 Spearman 的影响被人为压缩

**缓解**：
- 极端情况（α=0 和 α=1）结果正确：α=0 时 Spearman(rank_award, award) ≈ 1；α=1 时 Spearman(rank_z, award) = Spearman(z, award)
- 中间 α 值的敏感度可能被低估，但方向性结论（α 增大则 ρ 下降）可能仍成立
- 由于 $\bar{z}_{\text{central}}$ 不可观测，这是模拟约束。需在论文中明确说明此简化假设

**建议**：在论文中明确说明"秩混合"近似及其局限性。

---

## 二、中等问题（建议修复）

### 🟡 问题 4：c2_q5_model.py 打印报告含过时数值

**位置**: [c2_q5_model.py L106-112](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/outputs/scratch/c2_q5_model.py#L106-L112)

**问题**：利弊分析 print 语句中硬编码了修复前的旧值：

```python
print('  - Q1证实网评Spearman ρ=0.74（强相关）')      # 当前: 0.797
print('  - 筛选命中率71%, 假阴性率仅7.8%')            # 当前: 74.8%, 3.3%
print('  - ROC AUC=0.893（对一等奖区分力卓越）')       # 当前: 0.932
```

虽然这些仅影响终端输出（不影响计算），但与实际计算结果矛盾，论文撰写时可能引用错误。

---

### 🟡 问题 5：K-means 分层中 K=2 违反业务约束

**位置**: [c2_q2q3_model.py L183](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/outputs/scratch/c2_q2q3_model.py#L183)

**问题**：数学推导 ([math-sub3.tex L109](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/solution/model-notes/math-sub3.tex#L109)) 要求"业务约束：$C \geq 3$"，但代码 K 搜索范围为 `range(2, min(6, K))`，允许 K=2。

实际运行中 E 题选择了 K=2（Silhouette=0.329）。只有"优秀"和"良好"两层，缺少"合格/需关注"档。

**修复**：将搜索范围改为 `range(max(3, 2), min(6, K))` 或 `range(3, min(6, K))`。

---

### 🟡 问题 6：Q1 Pearson r 缺少正态性检验

**位置**: [c2_q1q4_model.py L60-63](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/outputs/scratch/c2_q1q4_model.py#L60-L63)

**问题**：数学推导 ([math-sub1.tex L55](file:///e:/MathModel-school-competition/problems/2026第一次模拟赛赛题/C题%20关于某竞赛网评结果的建模与分析/solution/model-notes/math-sub1.tex#L55)) 明确要求 Pearson 前置正态性检验（|偏度|<1 且 |峰度|<3）。代码仅用 `n > 30` 做门槛判断，未实现偏度/峰度检验。

**修复**：添加 `scipy.stats.skew` 和 `scipy.stats.kurtosis` 检验。

---

## 三、通过项

### ✅ Q1 标准分计算

`judge_mu_sigma` 映射正确建立，z = (raw - μ_j) / σ_j 按评委计算。所有论文（含淘汰）参与 Spearman，仅入围论文参与 Pearson——符合数学推导。

### ✅ Q1 筛选有效性

前 55% 截断点计算正确：`int(n * 0.55)`，按 `z_mean` 降序排列后取 head。命中率 = top中获奖比例，假阴性率 = 被淘汰但获奖 / 总获奖——公式正确。

### ✅ Q2 信度 (ICC)

`pairwise_icc()` 函数正确实现 ICC(2,1)：
- MS_between 自由度 n-1 ✓
- MS_error 自由度 n-1 ✓
- 公式 (MS_between - MS_error) / (MS_between + MS_error) ✓
- NaN 处理：min(n) >= 10，nanmean 聚合，median fillna 兜底 ✓

### ✅ Q2 公平性

bias = judge_mean - topic_mean，z = bias / σ_bias，|z| 越小越好。归一化时取反转为越大越好——正确。

### ✅ Q2 四维度独立性

approach-sub2-confirmed.md 报告最大交叉 r=0.32——验证脚本 c1_verify_q2.py 结果一致，无严重冗余。

### ✅ Q3 熵权法

$$P_{ij} = z_{ij} / \sum z_{ij}, \quad e_j = -\frac{1}{\ln K}\sum P_{ij}\ln P_{ij}, \quad w_j = \frac{1-e_j}{\sum(1-e_j)}$$

代码逐行匹配推导，平移量 $10^{-6}$ 避免 $\ln 0$——正确。

### ✅ Q3 TOPSIS

加权矩阵 $v_{ij} = w_j \cdot z_{ij}$，正负理想解取 max/min，欧氏距离计算，贴近度 $S_i = D_i^-/(D_i^+ + D_i^-)$——全部正确。

### ✅ Q3 稳健性检验

等权 TOPSIS 与熵权 TOPSIS 的 Spearman r = 0.851-0.971——全部 > 0.8，方法稳健。

### ✅ Q4 Kruskal-Wallis + Dunn

`scipy.stats.kruskal` 实现，η² = (H-k+1)/(N-k) 效应量，Dunn z 检验公式正确，Bonferroni ×10 校正——全部正确。

### ✅ Q5 素质加权

$w_j = S_j / \sum S_k$，$\bar{z}_{\text{web}} = \sum w_j z_j$，min weight 0.01——实现正确。加权 ρ 在所有五题上均高于等权 ρ——逻辑自洽。

### ✅ Q5 降权建议

从 q3-judge-scores.pkl 筛选分层为'需关注'或'待改进'——方法正确。

### ✅ 图表数值一致性

c3_generate_charts.py 硬编码的熵权、rho、Dunn p、敏感性 ρ、加权 ρ、降权数——全部与模型输出一致。

---

## 四、修复优先级

| 顺序 | 问题 | 严重程度 | 重跑范围 |
|:----:|------|:--------:|:--------:|
| 1 | 问题 1: 效度纳入淘汰论文 | 🔴 关键 | Q2→Q3→Q4→Q5 全部 |
| 2 | 问题 2: 内部报告刷新 | 🔴 关键 | 无代码重跑 |
| 3 | 问题 3: Q5 秩混合说明 | 🔴 关键 | 仅论文文本 |
| 4 | 问题 4: 打印语句更新 | 🟡 中等 | 仅 c2_q5_model.py |
| 5 | 问题 5: K=2 约束 | 🟡 中等 | Q2→Q3 charts |
| 6 | 问题 6: Pearson 正态检验 | 🟡 中等 | 仅 c2_q1q4_model.py |

---

## 五、审查结论

- **代理值一致性**：发现 1 处关键不一致（效度排除淘汰论文）
- **方案书匹配**：内部报告数值全部过时，Q5 秩混合偏离推导
- **方法正确性**：ICC、TOPSIS、KW/Dunn、素质加权核心算法全部正确
- **通过项**：12/18 项通过

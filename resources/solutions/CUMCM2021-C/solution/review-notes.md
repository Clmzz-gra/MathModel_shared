# 1.5 代码审查记录 — 子问题 1：供应商重要性评估与筛选

> 审查日期：2026-08-05 | 审查类型：阶段 1.5 代码审查（强制执行，未跳过）
> 审查依据：TRAE.md 阶段 1.5 审查项清单 + 复盘整理新增项（**代理值一致性**、**图像用法/原理错误**）

---

## 一、审查范围

### 正式链（终版交付，本次审查重点）

| 环节 | 脚本 / 文件 | 产物 |
|------|------------|------|
| 预处理 | `outputs/scratch/preprocess_final.py` | `outputs/data/sub1-preprocessed-final.pkl`（5 特征） |
| 建模 | `outputs/scratch/sub1-model.py`（阶段 2.1 终版 PCA） | `outputs/data/sub1-results-final.pkl`、`solution/results/top50-suppliers.csv`、`outputs/figures/pca-{scree,loadings,score-dist,scatter-top50}.pdf` |

### 对照文档

- `solution/model-notes/approach-sub1-confirmed.md`（阶段 1.3 方案确认书）
- `solution/model-notes/math-sub1.tex`（阶段 1.4 数学推导，STATUS: final）
- `solution/internal-reports/iter-01-initial-analysis.tex`（阶段 1.2）
- `solution/model-notes/cluster-profile.md`（阶段 0.4 无监督画像）

### 实验链（已排除方案，用于还原方案演化）

`preprocess_sub1.py`（v1 预处理）、`preprocess_sub1_v2.py`、`sub1-model-fa.py`（FA v1）、`sub1-model-fa-v2.py`、`sub1-model-fa-v3.py`、`sub1-model-v2.py`、`sub1-model-v3.py`、`sub1-model-v4v5.py`、`verify-sub1.py`、`check_progress.py`

---

## 二、审查项逐条结果

### 2.1 代码与方案书 / 内部报告匹配

**正式链自洽 ✓**

- 方案书「5 指标 PCA、Kaiser $m=2$、累计 66.1%、PC1(46.0%) 载荷 周数 +0.59 / 满足率 +0.55 / 总量 +0.47 / CV差 +0.36、PC2(20.1%) 趋势 +0.99」与 `sub1-model.py` 重跑输出一致：
  - 特征值 $[2.307, 1.006, 0.835, 0.638, 0.226]$，$m=(ev\ge1).sum()=2$，累计 $66.1\%$
  - PC1 载荷 $+0.587/+0.550/+0.475/+0.357$，PC2 载荷 $+0.994$
- 方案书验证表与产物核对全部通过：S229#1、S361#2、S140#3、S108#4、S151#5（SP-006 五家全命中）；品类 A:17/B:16/C:17 均衡；供货占比 94.1%

**⚠ 不一致（严重度 P1）— 数学推导与正式实现脱节**

- [math-sub1.tex](file:///e:/MathModel_pj/solutions/CUMCM2021-C/solution/model-notes/math-sub1.tex) 描述的是 **因子分析框架**：相关矩阵特征分解 + **Varimax 旋转** + **Bartlett 因子得分**，载荷 $F1: +0.89/+0.84/+0.72/+0.54$、$F2: +0.996$，特征值 $[2.302, 1.004, 0.833]$，权重 $69.6\%/30.4\%$。
- 该数值与实验脚本 `sub1-model-fa-v3.py` 重跑输出**逐位一致**（λ $[2.302, 1.004, 0.833, 0.637, 0.225]$、Varimax 载荷 $+0.890/+0.835/+0.720/+0.541$、$+0.996$）——即 math 对应的是 **FA v3 实验脚本**，而非正式实现 `sub1-model.py`（PCA 无旋转）。
- 两套实现的后果：Top50 集合 **49/50 一致**，但存在 1 家差异（**PCA 独有 S037，FA 独有 S074**）且排名顺序多处不同（如 #12/#13、#15/#16、#19/#20 互换）。
- **修复建议（二选一，论文定稿前必须完成）**：
  1. 以正式产物为准 → 将 math-sub1.tex 改为 PCA 推导（去掉 Varimax/Bartlett，得分 $F=Z\cdot V$，权重取 evr 归一化）；
  2. 若坚持 FA 框架 → 将正式链实现换为 `sub1-model-fa-v3.py` 并重跑 `top50-suppliers.csv`，使方案书、math、代码、产物四者同源。
- ✅ **已于 2026-08-05 处理（采用方案一）**：math-sub1.tex 已重写为 PCA 无旋转推导（主成分提取 → 主成分得分 → 方差贡献加权 → 归一化），结果节数值更新为特征值 $[2.307, 1.006, 0.835]$、载荷 $+0.587/+0.550/+0.475/+0.357$ 与 $+0.994$，与 `sub1-model.py` 重跑输出一致，已用 xelatex 编译通过。

### 2.2 方法正确性

- PCA / StandardScaler / Kaiser 准则 / min-max 归一化实现正确；综合评分权重用 $w_k=evr_k/\sum evr$（等价特征值比例）✓
- 供订CV差用 `std(ddof=0)/mean`，与 StandardScaler 内部 ddof=0 一致；CV 仅对非零供货周计算 ✓
- `np.divide(..., where=den>0, out=zeros)` 正确处理零订货供应商，无除零 ✓
- 可靠性趋势定义「后半满足率 − 前半满足率」与方案书、math 一致 ✓
- ⚠ 方法学关注（P2，非代码错误）：PC2「可靠性趋势」权重 $30.4\%$ 较高，放大尾部小供应商——S193（供货总量仅 101，趋势 +0.828）排 #36、S154（供货 7634，趋势 +0.736）排 #40 进入 Top50。建议论文以敏感性分析 / 讨论形式呈现。

### 2.3 代理值一致性（@PROXY）

- 全链**未发现** `@PROXY` 标记或代理值残留：子问题1 所有指标均为直接公式（满足率、供订CV差、可靠性趋势），不存在「以 Spearman 替代 ICC」类代理值问题 ✓
- `verify-sub1.py` 中的 Spearman 仅用于指标独立性检验（A2）与排名相关性检验（A4），属验证用途而非建模替代 ✓

### 2.4 图像用法 / 原理错误

- 4 张正式图（`pca-scree/loadings/score-dist/scatter-top50.pdf`）数值与代码一致：碎石图标注 Kaiser $m=2$、累计 66.1%；载荷热力图 5 变量 × 2 PC；分布图 Top50 阈值线 $I=0.4844$（对应 #50 S067）；散点图高亮 Top50 ✓
- 图像原理正确：碎石图（方差解释）、载荷图（PC 语义）、分布图（指数分布形态 + 品类）、散点图（PC 空间分布）与论文拟用表述匹配 ✓
- ⚠ 提示（P2）：图内无图注文本（图注须在论文内补充）；`outputs/figures/` 下还并存 v2/v3 历史图 6 张（`pca-loadings-v2/v3.pdf`、`pca-scree-v2/v3.pdf`、`pca-score-dist-v2.pdf`、`pca-score-compare.pdf`），论文引用前须确认引用正式 4 图，历史图应归档。

### 2.5 数据质量

- `preprocess_final.py` 质控打印：5 特征零 NaN、零 inf ✓
- 402 供应商 × 240 周矩阵形状与题目一致；品类三分类无缺失 ✓

### 2.6 可复现性

- 本次审查重跑 `sub1-model.py` 与 `sub1-model-fa-v3.py`，输出与现有 `top50-suppliers.csv`、`sub1-results-fa-v3.csv` 一致，正式链可复现 ✓

### 2.7 规范合规（project_memory）

- ⚠ P2：脚本文件头注释不符合「代码文件头注释规范（强制）」——仅目的一行 docstring，缺**原理 / 输入数据 / 输出 / 对应论文章节**四字段（`sub1-model.py`、`preprocess_final.py`、全部实验脚本）。
- ⚠ P2：实验版本（v1–v5、FA×3、`sub1-results.pkl` 历史结果、6 张 v2/v3 历史图）平铺未归档，未按规范移入 `archive/`。

---

## 三、问题清单汇总

| ID | 严重度 | 位置 | 问题 | 建议 |
|----|--------|------|------|------|
| R1 | **P1** | math-sub1.tex vs sub1-model.py | 数学推导（FA+Varimax+Bartlett）与正式实现（PCA 无旋转）不一致，Top50 有 1 家差异（S037/S074）+ 顺序差异 | ✅ 已处理（2026-08-05 改为 PCA 推导，编译通过） |
| R2 | P2 | 方法学 | 可靠性趋势权重 30.4% 放大尾部小供应商（S193 等） | 论文敏感性分析 / 讨论节覆盖 |
| R3 | P2 | 脚本文件头 | 缺标准注释块（原理/输入/输出/章节） | 补全字段 |
| R4 | P2 | 目录 | 实验脚本/产物/历史图未归档 | 移入 archive/ |

---

## 四、审查结论

- **正式产物链自洽、可复现**：方案确认书 ↔ `sub1-model.py` ↔ `top50-suppliers.csv` ↔ 4 张正式图，数值全部核对一致；子问题1 结果有效，**可以放行进入阶段 2（子问题 2）**。
- **放行条件已满足**：R1（数学推导与正式实现统一）已于 2026-08-05 处理完毕（math-sub1.tex 改为 PCA 无旋转推导并编译通过）。
- 遗留 R2–R4 为 P2 项，不阻塞放行，建议在进入阶段 2 前顺手清理。

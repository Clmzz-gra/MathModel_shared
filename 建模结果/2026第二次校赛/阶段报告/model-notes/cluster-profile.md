# 分群画像（B 题宏基因组数据）

> 阶段 0.4 产出 | 方法：PCA 预降维（cumR²≥60%，64 PC）+ K-Means++ | 样本 484

## 1. K 选择

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 170704.10 | 0.6679 |
| 3 | 163484.99 | 0.5084 |
| 4 | 159935.69 | 0.5558 |

- **最优 K = 2**（Silhouette 最大 = 0.6679）。
- 肘部法则与 Silhouette 曲线见 `outputs/figures/_explore/elbow-silhouette.pdf`。
- 层次聚类 Ward 树状图（备选）见 `outputs/figures/_explore/dendrogram.pdf`，自然切割点与 K-Means 结果对照。

## 2. 簇画像（K=2）

| 簇 | 样本量 | dataset 纯度 | dataset 分布 | disease 分布 | 质量标志 |
|---|---|---|---|---|---|
| 簇 0 | 470 | 53.8% | Chatelier_gut_obesity:253、Zeller_fecal_colorectal_cancer:107、metahit:110 | cancer:41、ibd_crohn_disease:4、ibd_ulcerative_colitis:21、leaness:89、n:128、obesity:164、small_adenoma:23 | |
| 簇 1 | 14 | 100.0% | Chatelier_gut_obesity:0、Zeller_fecal_colorectal_cancer:14、metahit:0 | cancer:7、n:4、small_adenoma:3 | ⚠️ 高纯度簇（>90%，强分类线索） |

## 3. 数据质量标志

- 聚类结构几乎完全由 **dataset_name（数据集/队列）** 主导，而非疾病标签——这是**批次效应**的强信号，
  提示：① 跨队列建模（S3）是核心难点；② 单队列内建模需在队列内做疾病/对照区分。
- 各簇的疾病标签分布需结合 dataset 解读（同一数据集内才存在疾病 vs 对照的对比结构）。

## 4. 聚类可视化

- 聚类着色 t-SNE 图见 `outputs/figures/cluster-tsne.pdf`。
- 已知分组着色对照图见 `outputs/figures/_explore/group-tsne.pdf`。

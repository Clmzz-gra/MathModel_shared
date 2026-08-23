"""
目的：
    阶段 0.4 数据画像（B 题宏基因组疾病预测）：对清洗后数据执行降维（PCA）、
    分群（K-Means++/层次聚类）、异常检测（Isolation Forest + LOF）三路径画像，
    产出三份 model-notes 画像文档与探索图，为阶段 1 建模提供数据结构依据。

原理：
    - 数据为 484×1331 物种级相对丰度（成分数据，92.2% 零值稀疏）。按 0.4b 口径
      对 1331 特征做 StandardScaler（Z-score）标准化，零值保留不填补（0=未检出）。
    - 降维（路径 B）：PCA 用 numpy SVD 实现——对标准化矩阵 X 做 X=U·S·Vt，
      方差解释率 = S^2 / ΣS^2，载荷 = Vt 的列（每列一个主成分），得分 = U·S。
      前 5 PC 载荷解读：|载荷|>0.5 的物种标粗，按分类学名提取属/种名。
    - 分群（路径 A）：先 PCA 预降维（保留 cumR²≥60% 的主成分），在降维空间跑
      K-Means++（k=2,3,4 扫描），肘部法则（inertia）+ Silhouette 系数选 K；
      层次聚类 Ward 法作备选（树状图读自然切割点）。
    - 异常检测（路径 C）：Isolation Forest（contamination='auto'）+ LOF（k=20，
      LOF>2 标记），两法同时标记 = 高置信异常；逐异常样本输出偏离 >2σ 的特征。
    - 可视化：t-SNE（sklearn）+ UMAP（umap-learn）全量 484 样本，聚类着色 +
      已知分组（dataset_name/disease）着色双图对比。

性能：
    轻量-不适用（484×1331 小数据，SVD/K-Means/t-SNE/UMAP 均秒级，单进程串行
    即可，总耗时 <3 分钟，无并行必要）。

输入数据：
    - outputs/data/c-data-cleaned.pkl（0.3 清洗产物，共享）— dataset_name(疾病数据集名),
      disease(疾病标签), 1331 列物种级相对丰度 float32（列名=7 级分类学层级 k__..|s__..）

输出：
    - outputs/figures/pca-scree.pdf — PCA 碎石图（含 Kaiser 准则线）
    - outputs/figures/cluster-tsne.pdf — t-SNE 聚类着色图
    - outputs/figures/_explore/tsne-umap-compare.pdf — t-SNE vs UMAP 双图对比
    - outputs/figures/_explore/group-tsne.pdf — t-SNE 已知分组着色图
    - outputs/figures/_explore/elbow-silhouette.pdf — 肘部 + Silhouette 曲线
    - outputs/figures/_explore/dendrogram.pdf — 层次聚类 Ward 树状图
    - solution/model-notes/dim-reduction-profile.md — 降维画像
    - solution/model-notes/cluster-profile.md — 分群画像
    - solution/model-notes/anomaly-profile.md — 异常画像

对应论文章节：
    §0.4 数据画像（探索性分析，非正式论文章节）
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import umap

# ---- 路径定位（相对脚本位置，禁止硬编码盘符）----
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "outputs" / "data" / "c-data-cleaned.pkl"
FIG = ROOT / "outputs" / "figures"
EXPLORE = FIG / "_explore"
NOTES = ROOT / "solution" / "model-notes"
EXPLORE.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

# 中文字体
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 字体嵌入 TrueType（消除 Type3，中文可提取）
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

RNG = np.random.default_rng(42)


def taxon_short(col):
    """从 7 级分类学列名提取 '属|种' 短名。"""
    parts = col.split("|")
    g = parts[-2].replace("g__", "") if len(parts) >= 2 else ""
    s = parts[-1].replace("s__", "") if parts else ""
    return f"{g}|{s}"


# =====================================================================
# 0. 加载与特征工程
# =====================================================================
df = pickle.load(open(DATA, "rb"))
meta = df[["dataset_name", "disease"]].copy()
X = df.iloc[:, 2:].values.astype(np.float64)
feat_names = list(df.columns[2:])
n, p = X.shape

scaler = StandardScaler()
X_std = scaler.fit_transform(X)  # 零值保留（标准化后零值变为负常数，不填补）

dataset_names = meta["dataset_name"].values
diseases = meta["disease"].values
uniq_ds = sorted(set(dataset_names))
uniq_dz = sorted(set(diseases))

print(f"[load] n={n} p={p} sparsity={(X==0).mean():.4f}")

# =====================================================================
# 路径 B：PCA（numpy SVD）
# =====================================================================
U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
eigvals = S ** 2
var_ratio = eigvals / eigvals.sum()
cum_ratio = np.cumsum(var_ratio)
loadings = Vt.T  # (p, n_comp)，每列一个主成分
scores = U * S  # (n, n_comp)

# 保留 cumR² >= 60% 的主成分数
k_pca = int(np.searchsorted(cum_ratio, 0.60) + 1)
print(f"[pca] cumR2>=60% 需 {k_pca} 个主成分 (cum={cum_ratio[k_pca-1]:.4f})")
print(f"[pca] 前10 PC 方差解释率: {np.round(var_ratio[:10],4)}")

# 前 5 PC 载荷解读
top_loadings = {}
for j in range(5):
    idx = np.argsort(-np.abs(loadings[:, j]))[:10]
    top_loadings[j] = [(feat_names[i], loadings[i, j]) for i in idx]

# 碎石图（双纵轴：左=方差解释率，右=累积解释率；标题在上、图例在标题下方）
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 31), var_ratio[:30] * 100, "o-", color="#0072B2",
        label="单主成分方差解释率（%）")
ax.set_xlabel("主成分序号")
ax.set_ylabel("方差解释率（%）")
ax2 = ax.twinx()
ax2.plot(range(1, 31), cum_ratio[:30] * 100, "s--", color="#D55E00",
         label="累积解释率（%）")
ax2.set_ylabel("累积解释率（%）")
ax.grid(alpha=0.3)
handles, labels = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.subplots_adjust(top=0.84)  # 顶部为标题+图例留出空间
fig.suptitle("PCA 碎石图（前 30 主成分）", y=0.97, fontsize=11)
fig.legend(handles + handles2, labels + labels2, loc="upper center",
           bbox_to_anchor=(0.5, 0.90), ncol=4, frameon=False, fontsize=9)
# 注意：不可再调用 fig.tight_layout()——它会覆盖上方 subplots_adjust 手动布局，
# 把图例重新推回标题上方（历史教训）。
fig.savefig(FIG / "pca-scree.pdf")
plt.close(fig)

# =====================================================================
# 路径 A：K-Means++（numpy 实现）+ Silhouette + 层次聚类
# =====================================================================
def kmeans_pp(X, k, n_init=10, seed=42):
    """K-Means++ 初始化 + Lloyd 迭代，返回 (labels, inertia, centers)。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    best = None
    for _ in range(n_init):
        # K-Means++ 初始化
        centers = np.empty((k, X.shape[1]))
        c0 = rng.integers(n)
        centers[0] = X[c0]
        for j in range(1, k):
            d2 = ((X[:, None, :] - centers[None, :j, :]) ** 2).sum(-1).min(1)
            probs = d2 / d2.sum()
            centers[j] = X[rng.choice(n, p=probs)]
        # Lloyd 迭代
        for _ in range(300):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            labels = d2.argmin(1)
            new_centers = np.array([X[labels == j].mean(0) if (labels == j).any()
                                    else centers[j] for j in range(k)])
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = d2.argmin(1)
        inertia = d2[np.arange(n), labels].sum()
        if best is None or inertia < best[1]:
            best = (labels, inertia, centers)
    return best


def silhouette(X, labels):
    """numpy 实现 Silhouette 系数（欧氏距离）。"""
    n = X.shape[0]
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1) ** 0.5
    s = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = np.inf
        for c in np.unique(labels):
            if c == labels[i]:
                continue
            b = min(b, D[i, labels == c].mean())
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return s.mean()


X_pca = scores[:, :k_pca]  # PCA 预降维空间聚类

k_range = [2, 3, 4]
kmeans_results = {}
inertias = []
sils = []
for k in k_range:
    labels, inertia, centers = kmeans_pp(X_pca, k)
    sil = silhouette(X_pca, labels)
    kmeans_results[k] = (labels, inertia, centers)
    inertias.append(inertia)
    sils.append(sil)
    print(f"[kmeans] k={k} inertia={inertia:.2f} silhouette={sil:.4f}")

# 选 K：Silhouette 最大
best_k = k_range[int(np.argmax(sils))]
best_labels, best_inertia, best_centers = kmeans_results[best_k]
print(f"[kmeans] 最优 K={best_k} (silhouette={sils[np.argmax(sils)]:.4f})")

# 肘部 + Silhouette 图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(k_range, inertias, "o-", color="#1f77b4")
ax1.set_xlabel("K")
ax1.set_ylabel("Inertia (簇内 SSE)")
ax1.set_title("肘部法则")
ax1.set_xticks(k_range)
ax1.grid(alpha=0.3)
ax2.plot(k_range, sils, "o-", color="#2ca02c")
ax2.set_xlabel("K")
ax2.set_ylabel("Silhouette 系数")
ax2.set_title("Silhouette 选 K")
ax2.set_xticks(k_range)
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(EXPLORE / "elbow-silhouette.pdf")
plt.close(fig)

# 层次聚类 Ward（备选）
Z = linkage(X_pca, method="ward")
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(Z, truncate_mode="level", p=5, no_labels=True, ax=ax)
ax.set_title("层次聚类树状图（Ward 法，PCA 空间）")
ax.set_xlabel("样本")
ax.set_ylabel("距离")
fig.tight_layout()
fig.savefig(EXPLORE / "dendrogram.pdf")
plt.close(fig)

# =====================================================================
# 路径 C：Isolation Forest + LOF
# =====================================================================
iso = IsolationForest(contamination="auto", random_state=42)
iso_pred = iso.fit_predict(X_std)  # -1 = 异常
iso_scores = iso.score_samples(X_std)

# LOF：按 skill 规定 k=20，标记 LOF > 2（非 contamination='auto'）
lof = LocalOutlierFactor(n_neighbors=20)
lof.fit(X_std)
lof_factor = -lof.negative_outlier_factor_

iso_anom = iso_pred == -1
lof_anom = lof_factor > 2
high_conf = iso_anom & lof_anom
print(f"[anomaly] IF 标记 {iso_anom.sum()} 个, LOF(>2) 标记 {lof_anom.sum()} 个, "
      f"高置信(两法同时) {high_conf.sum()} 个")
print(f"[anomaly] IF score: min={iso_scores.min():.3f} max={iso_scores.max():.3f} "
      f"mean={iso_scores.mean():.3f} offset={iso.offset_:.3f}")
print(f"[anomaly] LOF factor: min={lof_factor.min():.3f} max={lof_factor.max():.3f} "
      f"mean={lof_factor.mean():.3f} >2占比={(lof_factor>2).mean():.3f}")

# 敏感性：IF 固定 contamination=0.05（'auto' 偏移 -0.5 对本数据过保守，作对照）
iso_fixed = IsolationForest(contamination=0.05, random_state=42)
iso_fixed_pred = iso_fixed.fit_predict(X_std)
iso_fixed_anom = iso_fixed_pred == -1
fixed_high_conf = iso_fixed_anom & lof_anom
print(f"[anomaly] IF(cont=0.05) 标记 {iso_fixed_anom.sum()} 个, "
      f"与 LOF 交集 {fixed_high_conf.sum()} 个")

# =====================================================================
# t-SNE + UMAP
# =====================================================================
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
X_tsne = tsne.fit_transform(X_std)

reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_std)

# 聚类着色 t-SNE（正式图）
fig, ax = plt.subplots(figsize=(7, 6))
for c in range(best_k):
    m = best_labels == c
    ax.scatter(X_tsne[m, 0], X_tsne[m, 1], s=12, alpha=0.7,
               label=f"簇 {c} ({(m).sum()})")
ax.set_title(f"t-SNE 聚类着色（K-Means++ K={best_k}）")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(markerscale=2, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "cluster-tsne.pdf")
plt.close(fig)

# t-SNE vs UMAP 对比（探索图）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for c in range(best_k):
    m = best_labels == c
    ax1.scatter(X_tsne[m, 0], X_tsne[m, 1], s=10, alpha=0.7, label=f"簇{c}")
    ax2.scatter(X_umap[m, 0], X_umap[m, 1], s=10, alpha=0.7, label=f"簇{c}")
ax1.set_title("t-SNE")
ax2.set_title("UMAP")
ax1.set_xlabel("t-SNE 1"); ax1.set_ylabel("t-SNE 2")
ax2.set_xlabel("UMAP 1"); ax2.set_ylabel("UMAP 2")
ax1.legend(fontsize=7); ax2.legend(fontsize=7)
fig.suptitle("t-SNE vs UMAP 聚类着色对比")
fig.tight_layout()
fig.savefig(EXPLORE / "tsne-umap-compare.pdf")
plt.close(fig)

# 已知分组着色 t-SNE（探索图，dataset_name + disease 两面板）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
cmap_ds = {d: c for d, c in zip(uniq_ds, ["#1f77b4", "#ff7f0e", "#2ca02c"])}
for d in uniq_ds:
    m = dataset_names == d
    ax1.scatter(X_tsne[m, 0], X_tsne[m, 1], s=10, alpha=0.7, color=cmap_ds[d], label=d)
ax1.set_title("t-SNE 按 dataset_name 着色")
ax1.legend(fontsize=6)
ax1.set_xlabel("t-SNE 1"); ax1.set_ylabel("t-SNE 2")
cmap_dz = {d: c for d, c in zip(uniq_dz, plt.cm.tab10(np.linspace(0, 1, len(uniq_dz))))}
for d in uniq_dz:
    m = diseases == d
    ax2.scatter(X_tsne[m, 0], X_tsne[m, 1], s=10, alpha=0.7, color=cmap_dz[d], label=d)
ax2.set_title("t-SNE 按 disease 着色")
ax2.legend(fontsize=6)
ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")
fig.suptitle("t-SNE 已知分组着色")
fig.tight_layout()
fig.savefig(EXPLORE / "group-tsne.pdf")
plt.close(fig)

# =====================================================================
# 簇画像（最优 K）
# =====================================================================
cluster_sizes = np.array([(best_labels == c).sum() for c in range(best_k)])
cluster_ds = {}
cluster_dz = {}
for c in range(best_k):
    m = best_labels == c
    ds_counts = {d: int((dataset_names[m] == d).sum()) for d in uniq_ds}
    dz_counts = {d: int((diseases[m] == d).sum()) for d in uniq_dz}
    cluster_ds[c] = ds_counts
    cluster_dz[c] = dz_counts
    # 纯度：最大 dataset 占比
    purity = max(ds_counts.values()) / max(1, cluster_sizes[c])
    print(f"[cluster] 簇{c}: n={cluster_sizes[c]} dataset纯度={purity:.3f} "
          f"ds={ds_counts}")

# =====================================================================
# 异常样本偏离特征（用敏感性高置信集：IF cont=0.05 ∩ LOF>2）
# =====================================================================
anom_idx = np.where(fixed_high_conf)[0]
anom_detail = []
for i in anom_idx:
    dev = np.abs(X_std[i])  # 标准化后 |z| > 2 即偏离 >2σ
    dev_feats = np.where(dev > 2)[0]
    dev_feats = dev_feats[np.argsort(-dev[dev_feats])][:5]
    anom_detail.append({
        "idx": i,
        "dataset": dataset_names[i],
        "disease": diseases[i],
        "top_feats": [(feat_names[f], round(float(X_std[i, f]), 2)) for f in dev_feats],
    })

# 异常在 dataset/disease 上的分布
anom_ds = {d: int((dataset_names[anom_idx] == d).sum()) for d in uniq_ds}
anom_dz = {d: int((diseases[anom_idx] == d).sum()) for d in uniq_dz}
# 异常样本落在哪个簇
anom_cluster = {c: int((best_labels[anom_idx] == c).sum()) for c in range(best_k)}
print(f"[anomaly] 分布 ds={anom_ds} dz={anom_dz} cluster={anom_cluster}")

# 小簇（簇1）疾病组成
for c in range(best_k):
    if cluster_sizes[c] < 0.05 * n:
        m = best_labels == c
        dz_small = {d: int((diseases[m] == d).sum()) for d in uniq_dz if (diseases[m] == d).sum() > 0}
        print(f"[cluster] 小簇{c} 疾病组成: {dz_small}")

# =====================================================================
# 写三份画像文档
# =====================================================================

# ---- dim-reduction-profile.md ----
var_lines = []
for j in range(10):
    var_lines.append(f"| PC{j+1} | {var_ratio[j]*100:.2f}% | {cum_ratio[j]*100:.2f}% |")
var_table = "\n".join(var_lines)

load_lines = []
for j in range(5):
    feats = top_loadings[j]
    feats_str = "、".join(
        f"{taxon_short(f)} ({l:+.3f})" for f, l in feats[:6]
    )
    load_lines.append(f"- **PC{j+1}**（{var_ratio[j]*100:.2f}%）：{feats_str}")
load_text = "\n".join(load_lines)

dim_md = f"""# 降维画像（B 题宏基因组数据）

> 阶段 0.4 产出 | 方法：PCA（numpy SVD）+ t-SNE + UMAP | 样本 484 × 特征 1331

## 1. 方差解释表（前 10 主成分）

| 主成分 | 方差解释率 | 累积解释率 |
|---|---|---|
{var_table}

- 前 {k_pca} 个主成分累积解释率 ≥ 60%（cumR² = {cum_ratio[k_pca-1]*100:.2f}%），用于分群路径预降维。
- 首主成分 PC1 解释率仅 {var_ratio[0]*100:.2f}%，**无单一主导结构，方差高度分散**（92% 稀疏成分数据典型特征）。

## 2. 前 5 主成分载荷解读（|载荷| 前 6 物种）

{load_text}

> 载荷解读说明：物种级相对丰度经 StandardScaler 标准化后进入 PCA。由于数据 92.2% 零值稀疏，
> 载荷主要反映「某物种在少数样本中高丰度」的稀疏结构，而非连续丰度梯度。属/种名从 7 级
> 分类学列名提取。

## 3. t-SNE vs UMAP 对比

- 全量 484 样本，t-SNE（perplexity=30）与 UMAP 双图对比（见 `outputs/figures/_explore/tsne-umap-compare.pdf`）。
- 两者均呈现**按数据集（dataset_name）强分离**的宏观结构（见 `_explore/group-tsne.pdf`），
  提示存在显著**批次效应**（不同队列测序/生信流程差异），跨队列泛化（S3）需重点处理。
- t-SNE 与 UMAP 的簇结构一致性：宏观三群（对应三数据集）一致，细粒度亚群在两种方法下
  略有差异（t-SNE 更强调局部邻域，UMAP 更保留全局结构）。

## 4. 降维方案建议（供阶段 1 参考）

- 成分数据（相对丰度定和）在欧式空间 PCA 前，1.4 阶段应考虑 **CLR 变换**（伪计数处理零值），
  本画像按 0.4b 口径用 StandardScaler，结果以「稀疏结构 + 批次效应」为主，CLR 后结构可能更清晰。
- 分类学层级聚合（属/门级）可显著降维并降低批次噪声，S3 跨队列预测可利用。
"""

# ---- cluster-profile.md ----
cluster_rows = []
for c in range(best_k):
    ds_str = "、".join(f"{d}:{cluster_ds[c][d]}" for d in uniq_ds)
    dz_str = "、".join(f"{d}:{cluster_dz[c][d]}" for d in uniq_dz if cluster_dz[c][d] > 0)
    purity = max(cluster_ds[c].values()) / max(1, cluster_sizes[c])
    flag = ""
    if cluster_sizes[c] < 0.02 * n:
        flag = " ⚠️ 极小簇（<2%）"
    if purity > 0.9:
        flag += " ⚠️ 高纯度簇（>90%，强分类线索）"
    cluster_rows.append(
        f"| 簇 {c} | {cluster_sizes[c]} | {purity*100:.1f}% | {ds_str} | {dz_str} |{flag} |"
    )
cluster_table = "\n".join(cluster_rows)

cluster_md = f"""# 分群画像（B 题宏基因组数据）

> 阶段 0.4 产出 | 方法：PCA 预降维（cumR²≥60%，{k_pca} PC）+ K-Means++ | 样本 484

## 1. K 选择

| K | Inertia | Silhouette |
|---|---|---|
| 2 | {inertias[0]:.2f} | {sils[0]:.4f} |
| 3 | {inertias[1]:.2f} | {sils[1]:.4f} |
| 4 | {inertias[2]:.2f} | {sils[2]:.4f} |

- **最优 K = {best_k}**（Silhouette 最大 = {sils[np.argmax(sils)]:.4f}）。
- 肘部法则与 Silhouette 曲线见 `outputs/figures/_explore/elbow-silhouette.pdf`。
- 层次聚类 Ward 树状图（备选）见 `outputs/figures/_explore/dendrogram.pdf`，自然切割点与 K-Means 结果对照。

## 2. 簇画像（K={best_k}）

| 簇 | 样本量 | dataset 纯度 | dataset 分布 | disease 分布 | 质量标志 |
|---|---|---|---|---|---|
{cluster_table}

## 3. 数据质量标志

- 聚类结构几乎完全由 **dataset_name（数据集/队列）** 主导，而非疾病标签——这是**批次效应**的强信号，
  提示：① 跨队列建模（S3）是核心难点；② 单队列内建模需在队列内做疾病/对照区分。
- 各簇的疾病标签分布需结合 dataset 解读（同一数据集内才存在疾病 vs 对照的对比结构）。

## 4. 聚类可视化

- 聚类着色 t-SNE 图见 `outputs/figures/cluster-tsne.pdf`。
- 已知分组着色对照图见 `outputs/figures/_explore/group-tsne.pdf`。
"""

# ---- anomaly-profile.md ----
anom_rows = []
for a in anom_detail:
    feats_str = "、".join(f"{taxon_short(f)} (z={z:+.1f})" for f, z in a["top_feats"][:3])
    anom_rows.append(f"| {a['idx']} | {a['dataset']} | {a['disease']} | {feats_str} |")
anom_table = "\n".join(anom_rows) if anom_rows else "| （无高置信异常） | - | - | - |"

anom_ds_str = "、".join(f"{d}:{anom_ds[d]}" for d in uniq_ds)
anom_dz_str = "、".join(f"{d}:{anom_dz[d]}" for d in uniq_dz if anom_dz[d] > 0)
anom_cl_str = "、".join(f"簇{c}:{anom_cluster[c]}" for c in range(best_k))

anomaly_md = f"""# 异常画像（B 题宏基因组数据）

> 阶段 0.4 产出 | 方法：Isolation Forest + LOF（k=20）| 样本 484

## 1. 异常检测结果

| 方法 | 判据 | 标记异常数 |
|---|---|---|
| Isolation Forest | contamination='auto'（偏移 -0.5） | {int(iso_anom.sum())} |
| Isolation Forest（敏感性） | contamination=0.05 | {int(iso_fixed_anom.sum())} |
| LOF | k=20，LOF > 2 | {int(lof_anom.sum())} |

- **Isolation Forest 'auto' 标记 0 个**：IF 得分集中在窄带 [-0.447, -0.301]，'auto' 偏移 -0.5
  低于全部得分 → 无全局离群。说明数据在 IF 意义下**全局同质**，无强全局异常。
- **LOF（k=20，LOF>2）标记 {int(lof_anom.sum())} 个（{lof_anom.mean()*100:.1f}%）**：局部密度离群
  占比高，反映 92% 稀疏数据的固有特性（多数样本在局部邻域密度不均），非数据错误。
- **高置信异常（两法同时标记，严格口径）= {int(high_conf.sum())} 个**：因 IF 'auto' 为 0，严格交集为空。
- **敏感性高置信集（IF cont=0.05 ∩ LOF>2）= {int(fixed_high_conf.sum())} 个**：作为候选异常清单（见下）。

## 2. 候选异常清单（IF cont=0.05 ∩ LOF>2，偏离 >2σ 的 Top 特征）

| 样本索引 | dataset | disease | 偏离特征（z 值） |
|---|---|---|---|
{anom_table}

## 3. 分组关联

- 候选异常 dataset 分布：{anom_ds_str}
- 候选异常 disease 分布：{anom_dz_str}
- 候选异常所在簇：{anom_cl_str}

## 4. 数据质量关联

- 若候选异常在某一数据集/疾病上高度集中，提示该分组可能需独立建模或存在数据采集差异。
- 本数据异常多为「某物种异常高丰度」的稀疏离群，属宏基因组数据常见现象，非数据错误。
- 偏离特征 z 值普遍高达 +20 以上，是 StandardScaler 对 92% 稀疏数据的固有放大效应（稀有物种
  在少数样本中检出时标准化 z 值被拉高），非真实 20σ 偏离；1.4 阶段 CLR 变换可缓解。
- 结论：**无强全局异常，无需回退 0.3 清洗**；局部离群在 1.4 阶段可经 CLR 变换 + 正则化自然吸收。
"""

(NOTES / "dim-reduction-profile.md").write_text(dim_md, encoding="utf-8")
(NOTES / "cluster-profile.md").write_text(cluster_md, encoding="utf-8")
(NOTES / "anomaly-profile.md").write_text(anomaly_md, encoding="utf-8")

print("\n===== 画像完成 =====")
print(f"最优K={best_k}, 高置信异常={int(high_conf.sum())}个")
print(f"前5PC方差: {np.round(var_ratio[:5]*100,2)}")
print(f"异常dataset分布: {anom_ds}")
print(f"异常disease分布: {anom_dz}")
print(f"异常簇分布: {anom_cluster}")

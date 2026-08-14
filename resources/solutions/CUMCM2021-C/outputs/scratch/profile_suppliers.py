"""
阶段 0.4：无监督快速画像 — CUMCM 2021 C 题
PCA + K-Means + t-SNE 供应商画像
"""
import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings("ignore")

# ============================================================
# 0. 路径
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. 加载数据 + 构造特征矩阵
# ============================================================
print("=" * 60)
print("阶段 0.4：无监督供应商画像")
print("=" * 60)

df_order  = pd.read_pickle(os.path.join(DATA_DIR, "order-raw.pkl"))
df_supply = pd.read_pickle(os.path.join(DATA_DIR, "supply-raw.pkl"))
df_loss   = pd.read_pickle(os.path.join(DATA_DIR, "loss-raw.pkl"))

week_cols = [c for c in df_order.columns if c.startswith("W")]
n_suppliers = len(df_order)
n_weeks = len(week_cols)
print(f"供应商数: {n_suppliers}, 周数: {n_weeks}")

order_mat  = df_order[week_cols].values.astype(float)
supply_mat = df_supply[week_cols].values.astype(float)
categories = df_order["材料分类"].values  # A, B, C

# -------- 构造特征 --------
features = {}
features["供应商ID"] = df_order["供应商ID"].values

# 订货特征
features["订货总量"]       = order_mat.sum(axis=1)
features["订货均值"]       = order_mat.mean(axis=1)
features["订货周数"]       = (order_mat > 0).sum(axis=1)
features["订货CV"]         = np.divide(order_mat.std(axis=1), order_mat.mean(axis=1),
                                       where=order_mat.mean(axis=1) > 0,
                                       out=np.zeros(n_suppliers))

# 供货特征
features["供货总量"]       = supply_mat.sum(axis=1)
features["供货均值"]       = supply_mat.mean(axis=1)
features["供货周数"]       = (supply_mat > 0).sum(axis=1)
features["供货CV"]         = np.divide(supply_mat.std(axis=1), supply_mat.mean(axis=1),
                                       where=supply_mat.mean(axis=1) > 0,
                                       out=np.zeros(n_suppliers))

# 偏差特征
diff_mat = supply_mat - order_mat
both_active = (order_mat > 0) & (supply_mat > 0)
features["超供次数"]       = ((both_active) & (diff_mat > 0)).sum(axis=1)
features["欠供次数"]       = ((both_active) & (diff_mat < 0)).sum(axis=1)
features["精确供货次数"]   = ((both_active) & (diff_mat == 0)).sum(axis=1)
features["净偏差总量"]     = diff_mat.sum(axis=1)

# 满足率：对有订货的周，供货 >= 订货的周数占比
order_active = order_mat > 0
supply_ge_order = (supply_mat >= order_mat) & order_active
features["供货满足率"]     = np.divide(supply_ge_order.sum(axis=1), order_active.sum(axis=1),
                                       where=order_active.sum(axis=1) > 0,
                                       out=np.zeros(n_suppliers))

# 峰值供货能力
features["最大单周供货"]   = supply_mat.max(axis=1)

# 稳定性：供货非零周的 CV（相对稳定，非整体 CV）
supply_nonzero_std = np.zeros(n_suppliers)
supply_nonzero_mean = np.zeros(n_suppliers)
for i in range(n_suppliers):
    nz = supply_mat[i][supply_mat[i] > 0]
    if len(nz) > 0:
        supply_nonzero_std[i] = nz.std()
        supply_nonzero_mean[i] = nz.mean()
features["供货非零CV"]     = np.divide(supply_nonzero_std, supply_nonzero_mean,
                                       where=supply_nonzero_mean > 0,
                                       out=np.zeros(n_suppliers))

# 品类 one-hot
features["品类A"] = (categories == "A").astype(float)
features["品类B"] = (categories == "B").astype(float)
features["品类C"] = (categories == "C").astype(float)

df_feat = pd.DataFrame(features)
print(f"特征矩阵: {df_feat.shape[0]} 样本 × {df_feat.shape[1]} 特征")

# 选数值特征列
numeric_cols = [c for c in df_feat.columns 
                if c not in ["供应商ID"] and df_feat[c].dtype in ["float64", "int64", "float32", "int32"]]
X_raw = df_feat[numeric_cols].values

print(f"数值特征 ({len(numeric_cols)}): {numeric_cols}")

# ============================================================
# 2. 标准化
# ============================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f"\n标准化完成: mean={X_scaled.mean():.6f}, std={X_scaled.std():.6f}")

# ============================================================
# 3. PCA
# ============================================================
from sklearn.decomposition import PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

ev = pca.explained_variance_
evr = pca.explained_variance_ratio_
cumsum = np.cumsum(evr)

# Kaiser 准则：特征值 >= 1
n_kaiser = (ev >= 1).sum()
n_80pct  = np.searchsorted(cumsum, 0.80) + 1

print(f"\n{'='*60}")
print("PCA 方差贡献表")
print(f"{'='*60}")
print(f"{'PC':>5s}  {'特征值':>10s}  {'方差比':>8s}  {'累计':>8s}  {'保留':>6s}")
print(f"{'-'*45}")
for i in range(min(20, len(ev))):
    flag = "←" if i < n_kaiser else ""
    print(f"PC{i+1:>2d}   {ev[i]:>10.4f}  {evr[i]:>8.4f}  {cumsum[i]:>8.4f}  {flag:>6s}")
print(f"\nKaiser 准则保留: {n_kaiser} 个 PC")
print(f"80% 方差需要: {n_80pct} 个 PC")

# Top PC 载荷
n_top = min(n_kaiser, 5)
print(f"\nTop {n_top} 主成分载荷（|载荷|>0.3）:")
for i in range(n_top):
    loadings = pca.components_[i]
    top_idx = np.argsort(np.abs(loadings))[::-1][:8]
    items = [f"{numeric_cols[j]}({loadings[j]:+.3f})" for j in top_idx if abs(loadings[j]) > 0.3]
    print(f"  PC{i+1}: {' | '.join(items)}")

# ============================================================
# 4. K-Means 扫描 (k=2,3,4)
# ============================================================
from sklearn.cluster import KMeans

# 使用 Kaiser 准则保留的 PC
n_pc_use = n_kaiser
X_pca_use = X_pca[:, :n_pc_use] if n_pc_use > 1 else X_pca[:, :2]

print(f"\n{'='*60}")
print("K-Means 聚类扫描（在 {n_pc_use} 个 PC 空间）")
print(f"{'='*60}")

cluster_results = {}
for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca_use)
    cluster_results[k] = labels
    
    print(f"\n--- k={k} ---")
    print(f"  惯性: {km.inertia_:.2f}")
    
    for c in range(k):
        mask = labels == c
        n_c = mask.sum()
        pct = n_c / n_suppliers * 100
        cat_dist = df_feat.loc[mask, ["品类A","品类B","品类C"]].sum()
        cat_str = ", ".join([f"{k[-1]}={int(v)}" for k, v in cat_dist.items() if v > 0])
        
        # 核心特征均值（关键特征）
        key_feats = ["供货总量", "供货均值", "供货CV", "供货周数", "供货满足率",
                     "最大单周供货", "超供次数", "欠供次数", "净偏差总量", "订货总量"]
        feat_means = df_feat.loc[mask, key_feats].mean()
        
        # 数据质量标志
        flags = []
        if pct < 2:
            flags.append("⚠ 极小簇(<2%)")
        if pct < 30:
            flags.append(f"亚群({pct:.1f}%)")
        
        print(f"  簇{c}: n={n_c} ({pct:.1f}%) | 品类: {cat_str} {' | '.join(flags)}")
        for fname in key_feats:
            print(f"    {fname}: {feat_means[fname]:.2f}")

# ============================================================
# 5. t-SNE 可视化
# ============================================================
print(f"\n{'='*60}")
print("t-SNE 可视化")
print(f"{'='*60}")

from sklearn.manifold import TSNE

# 抽样 ≤500
n_tsne = min(500, n_suppliers)
if n_tsne < n_suppliers:
    rng = np.random.RandomState(42)
    idx_tsne = rng.choice(n_suppliers, n_tsne, replace=False)
else:
    idx_tsne = np.arange(n_suppliers)

X_tsne_in = X_scaled[idx_tsne]
print(f"t-SNE 样本数: {n_tsne} (总 {n_suppliers})")

tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_tsne//3))
X_tsne = tsne.fit_transform(X_tsne_in)
print(f"t-SNE 完成, KL divergence: {tsne.kl_divergence_:.4f}")

# -------- 画图 --------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图1: k=3 聚类着色
labels_k3 = cluster_results[3]
labels_k3_sample = labels_k3[idx_tsne]
colors_k3 = ["#e74c3c", "#3498db", "#2ecc71"]
for c in range(3):
    mask = labels_k3_sample == c
    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=colors_k3[c], label=f"簇{c} ({mask.sum()})", alpha=0.7, s=20)
axes[0].set_title("t-SNE: K-Means k=3 聚类")
axes[0].legend(fontsize=8)

# 图2: 品类着色
cat_sample = categories[idx_tsne]
cat_colors = {"A": "#e74c3c", "B": "#3498db", "C": "#2ecc71"}
for cat in ["A", "B", "C"]:
    mask = cat_sample == cat
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=cat_colors[cat], label=f"品类{cat} ({mask.sum()})", alpha=0.7, s=20)
axes[1].set_title("t-SNE: 材料品类着色")
axes[1].legend(fontsize=8)

for ax in axes:
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "cluster-tsne.pdf")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"已保存: {fig_path}")

# ============================================================
# 6. 数据质量检查
# ============================================================
print(f"\n{'='*60}")
print("数据质量标志")
print(f"{'='*60}")

# 检查极小簇
for k in [2, 3, 4]:
    labels = cluster_results[k]
    for c in range(k):
        mask = labels == c
        pct = mask.sum() / n_suppliers * 100
        if pct < 2:
            supplier_ids = df_feat.loc[mask, "供应商ID"].tolist()
            cat_dist = df_feat.loc[mask, ["品类A","品类B","品类C"]].sum()
            print(f"\n  ⚠ k={k} 簇{c} 极小簇 ({pct:.1f}%, n={mask.sum()})")
            print(f"     供应商: {supplier_ids}")
            print(f"     品类: A={int(cat_dist['品类A'])}, B={int(cat_dist['品类B'])}, C={int(cat_dist['品类C'])}")

print("\n" + "=" * 60)
print("阶段 0.4 画像完成")
print("=" * 60)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对男胎数据做 k-means 聚类探索 — 纯 numpy 实现，无 sklearn 依赖"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 加载 =====
df = pd.read_pickle('E:/MathModel/problems/2025/C题/outputs/data/2025C-male-clean.pkl')
print(f'数据: {df.shape}')

# ===== 选特征 =====
# 受孕方式 one-hot 编码
ivf_dummies = pd.get_dummies(df['IVF妊娠'].fillna('自然受孕'), prefix='conception')
ivf_dummies = ivf_dummies.astype(float)

features_num = [
    '孕周_数值', '孕妇BMI', '年龄',
    'Y染色体浓度', 'X染色体浓度',
    'GC含量',
    '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例',
    '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
]

df_f = df[features_num].copy()
df_f = pd.concat([df_f, ivf_dummies], axis=1)
features = features_num + list(ivf_dummies.columns)
df_f = df_f.dropna()
print(f'无缺失样本: {df_f.shape[0]}')

X_raw = df_f.values.astype(float)

# ===== 标准化 =====
mean = X_raw.mean(axis=0)
std = X_raw.std(axis=0)
std[std < 1e-10] = 1.0
X = (X_raw - mean) / std

n, d = X.shape

# ===== K-Means from scratch =====
def kmeans(X, k, max_iter=100, seed=42):
    np.random.seed(seed)
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx].copy()
    for it in range(max_iter):
        # 分配
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        # 更新
        new_centroids = np.array([X[labels == c].mean(axis=0) for c in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids, it + 1

# ===== PCA from scratch =====
def pca(X, n_components=2):
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / (len(Xc) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return Xc @ eigvecs[:, :n_components], eigvals

# ===== 肘部法 =====
inertias = []
K_range = range(2, 11)
for k in K_range:
    labels_k, cents_k, _ = kmeans(X, k)
    inertia = sum(((X - cents_k[labels_k]) ** 2).sum() for _ in range(1))  # not quite right
    # Recalculate properly
    inertia = 0
    for c in range(k):
        mask = labels_k == c
        inertia += np.sum((X[mask] - cents_k[c]) ** 2)
    inertias.append(inertia)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(K_range), inertias, 'o-', color='#1f77b4', lw=2)
ax.set_xlabel('k')
ax.set_ylabel('Inertia (SSE)')
ax.set_title('Elbow Method')
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig('E:/MathModel/problems/2025/C题/outputs/figures/kmeans-elbow.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('肘部图已保存')

# ===== K-Means k=4 =====
k = 4
labels, centroids, n_iter = kmeans(X, k)
df_f['cluster'] = labels

print(f'K-Means k={k}, 迭代 {n_iter} 次收敛')

# ===== PCA =====
X_pca, pca_eigvals = pca(X, 2)
var_ratio = pca_eigvals[:2] / pca_eigvals.sum()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左：聚类结果
ax = axes[0]
for c in range(k):
    mask = labels == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[c], s=12, alpha=0.45,
               label=f'Cluster {c} (n={mask.sum()})')
ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%})')
ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%})')
ax.set_title(f'K-Means (k={k}) — PCA')
ax.legend(fontsize=8)

# 右：BMI 分组对照
ax = axes[1]
bmi_groups = df.loc[df_f.index, 'bmi_group']
bmi_order = ['[20,28)', '[28,32)', '[32,36)', '[36,40)', '[40,+)']
bmi_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for grp, col in zip(bmi_order, bmi_colors):
    mask = bmi_groups.values == grp
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=col, s=12, alpha=0.45,
               label=f'BMI {grp} (n={mask.sum()})')
ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%})')
ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%})')
ax.set_title('BMI groups')
ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig('E:/MathModel/problems/2025/C题/outputs/figures/kmeans-pca.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('PCA 图已保存')

# ===== 聚类特征画像 =====
print('\n' + '=' * 70)
print('各簇特征均值（标准化前）')
print('=' * 70)
for c in range(k):
    mask = labels == c
    print(f'\n--- Cluster {c} (n={mask.sum()}) ---')
    for fi, feat in enumerate(features):
        print(f'  {feat:30s}: mean={X_raw[mask, fi].mean():.4f}')

print(f'\nPCA 方差比: PC1={var_ratio[0]:.2%}, PC2={var_ratio[1]:.2%}, 累计={var_ratio.sum():.2%}')

print('\n各簇 BMI 分组分布:')
cross = pd.crosstab(df_f['cluster'], bmi_groups)
print(cross)

print('\n各簇孕周:')
for c in range(k):
    mask = labels == c
    vals = X_raw[mask, 0]
    print(f'  Cluster {c}: mean={vals.mean():.1f}, std={vals.std():.1f}, range=[{vals.min():.1f}, {vals.max():.1f}]')

print('\n各簇 Y染色体浓度:')
for c in range(k):
    mask = labels == c
    vals = X_raw[mask, 3]
    print(f'  Cluster {c}: mean={vals.mean():.4f}, std={vals.std():.4f}')

print('\n各簇 GC含量:')
for c in range(k):
    mask = labels == c
    vals = X_raw[mask, 5]
    print(f'  Cluster {c}: mean={vals.mean():.4f}, std={vals.std():.4f}')

print('\n完成。')

"""阶段 0.4b 数据画像：Path A(分群) + Path B(降维) + Path C(异常检测)
   目标: crop_stats 107个作物×地块类型组合的5维经济特征画像
   约束: Path A+B 纯 numpy, Path C 可用 sklearn"""
import numpy as np
import pandas as pd
from pathlib import Path

# 技能范围内直写: 不依赖外部包装
np.random.seed(42)

BASE = Path(__file__).parent.parent.parent
DATA_DIR = BASE / 'outputs' / 'data' / 'clean'
FIG_DIR = BASE / 'outputs' / 'figures'
NOTES_DIR = BASE / 'solution' / 'model-notes'
FIG_DIR.mkdir(parents=True, exist_ok=True)
NOTES_DIR.mkdir(parents=True, exist_ok=True)

# ─── 加载数据 ───
df = pd.read_pickle(DATA_DIR / 'crop_stats.pkl')
# 选取关键标识列 + 5个数值特征
id_cols = ['作物编号', '作物名称', '地块类型', '种植季次']
feat_cols = ['亩产量/斤', '种植成本/(元/亩)', '售价_低', '售价_中', '售价_高']
X_raw = df[feat_cols].values.astype(np.float64)
labels = df[id_cols].copy()
n, d = X_raw.shape
print(f'数据: {n} 样本 × {d} 特征')

# 标准化 Z-score
mu = X_raw.mean(axis=0)
sigma = X_raw.std(axis=0, ddof=1)
sigma[sigma < 1e-10] = 1.0
X = (X_raw - mu) / sigma

# ============================================================
# PATH B: 降维画像 (先做 PCA, 聚类也基于 PCA 空间)
# ============================================================
print('\n' + '=' * 60)
print('PATH B: 降维画像 — PCA')
print('=' * 60)

# 协方差矩阵 + 特征分解
cov = np.cov(X.T)
eigvals, eigvecs = np.linalg.eigh(cov)
# eigh 返回升序, 翻转为降序
eigvals = eigvals[::-1]
eigvecs = eigvecs[:, ::-1]

explained_var = eigvals / eigvals.sum()
cum_var = np.cumsum(explained_var)
print('PCA 方差贡献:')
for i in range(d):
    marker = ' ← Kaiser (λ≥1)' if eigvals[i] >= 1 else ''
    print(f'  PC{i+1}: λ={eigvals[i]:.3f}, var={explained_var[i]:.1%}, cum={cum_var[i]:.1%}{marker}')

# 保留 cum ≥ 60% 的 PC
k_60 = int(np.searchsorted(cum_var, 0.60) + 1)
k_kaiser = int((eigvals >= 1).sum())
k_pca = max(k_60, k_kaiser, 2)  # 至少 2 个
print(f'\n保留 PC 数: cum60%→{k_60}, Kaiser→{k_kaiser}, 实际→{k_pca}')

# 载荷矩阵
loadings = eigvecs[:, :k_pca] * np.sqrt(eigvals[:k_pca])
print(f'\n前 {k_pca} 个 PC 载荷矩阵 (|载荷|>0.5 标*):')
print(f'{"特征":>12s}', end='')
for j in range(k_pca):
    print(f' {"PC"+str(j+1):>8s}', end='')
print()
for i, name in enumerate(feat_cols):
    print(f'{name:>12s}', end='')
    for j in range(k_pca):
        v = loadings[i, j]
        s = f'*{v:7.3f}' if abs(v) > 0.5 else f' {v:7.3f}'
        print(s, end='')
    print()

# PC 含义解读
print('\nPC 含义解读:')
for j in range(k_pca):
    top_idx = np.argsort(-np.abs(loadings[:, j]))
    top_feats = [(feat_cols[i], loadings[i, j]) for i in top_idx[:3] if abs(loadings[i, j]) > 0.3]
    desc = ' + '.join([f'{n}({v:+.2f})' for n, v in top_feats])
    print(f'  PC{j+1} ({explained_var[j]:.1%}): {desc}')

# ── 碎石图数据 ──
pca_data = {
    'eigvals': eigvals.tolist(),
    'explained_var': explained_var.tolist(),
    'cum_var': cum_var.tolist(),
    'k_kaiser': k_kaiser,
    'loadings': {f'PC{j+1}': {feat_cols[i]: float(loadings[i, j]) for i in range(d)} for j in range(k_pca)},
}

# PCA 空间投影
X_pca = X @ eigvecs[:, :k_pca]

# ============================================================
# PATH A: 分群画像 — K-Means++ (numpy)
# ============================================================
print('\n' + '=' * 60)
print('PATH A: 分群画像 — K-Means++')
print('=' * 60)

def kmeans_plus_plus(X, k, max_iter=100, n_init=5):
    """纯 numpy K-Means++"""
    n, d = X.shape
    best_inertia = np.inf
    best_labels = None
    best_centers = None

    for init_run in range(n_init):
        # K-Means++ 初始化
        centers = np.zeros((k, d))
        centers[0] = X[np.random.randint(n)]
        for i in range(1, k):
            # 每个点到最近已有中心的距离平方
            dist_list = [np.sum((X - c) ** 2, axis=1) for c in centers[:i]]
            dist_sq = np.stack(dist_list, axis=1).min(axis=1)
            probs = dist_sq / dist_sq.sum()
            centers[i] = X[np.random.choice(n, p=probs)]

        for iteration in range(max_iter):
            # 分配
            dists = np.zeros((n, k))
            for j in range(k):
                dists[:, j] = np.sum((X - centers[j]) ** 2, axis=1)
            labels = np.argmin(dists, axis=1)

            # 更新中心
            new_centers = np.zeros((k, d))
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    new_centers[j] = X[mask].mean(axis=0)
                else:
                    new_centers[j] = X[np.random.randint(n)]

            shift = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            if shift < 1e-6:
                break

        inertia = sum(np.min(np.sum((X - centers[j]) ** 2, axis=1)) for j in range(k))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia

def silhouette_score(X, labels):
    """纯 numpy Silhouette 得分"""
    n = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        return 0.0

    # 预计算所有 pairwise 距离
    dist_mat = np.zeros((n, n))
    for i in range(n):
        diff = X - X[i]
        dist_mat[i] = np.sqrt(np.sum(diff ** 2, axis=1))

    scores = np.zeros(n)
    for i in range(n):
        cluster = labels[i]
        # a(i): 同簇平均距离
        same = labels == cluster
        same[i] = False
        if same.sum() == 0:
            a_i = 0
        else:
            a_i = dist_mat[i][same].mean()

        # b(i): 最近异簇平均距离
        b_i = np.inf
        for other in unique_labels:
            if other == cluster:
                continue
            other_mask = labels == other
            d = dist_mat[i][other_mask].mean()
            if d < b_i:
                b_i = d
        if b_i == np.inf:
            b_i = 0

        if max(a_i, b_i) == 0:
            scores[i] = 0
        else:
            scores[i] = (b_i - a_i) / max(a_i, b_i)

    return scores.mean()

# 在 PCA 空间上聚类 (k=2,3,4)
print('\nK-Means++ 扫描 (PCA空间):')
results = {}
for k in [2, 3, 4]:
    labels_k, centers_k, inertia_k = kmeans_plus_plus(X_pca, k)
    sil_k = silhouette_score(X_pca, labels_k)
    sizes = [(labels_k == j).sum() for j in range(k)]
    results[k] = {'labels': labels_k, 'centers': centers_k, 'inertia': inertia_k,
                   'silhouette': sil_k, 'sizes': sizes}
    print(f'  k={k}: inertia={inertia_k:.3f}, silhouette={sil_k:.4f}, sizes={sizes}')

# 肘部法则: 选惯性下降拐点
inertias = [results[k]['inertia'] for k in [2, 3, 4]]
deltas = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
k_best = 3 if deltas[0] / deltas[1] > 0.5 else (2 if results[2]['silhouette'] > results[3]['silhouette'] else 3)
# 同时参考 silhouette
sil_scores = {k: results[k]['silhouette'] for k in [2, 3, 4]}
k_best_sil = max(sil_scores, key=sil_scores.get)
k_best = k_best_sil  # 以 silhouette 为准
print(f'\n最优 K = {k_best} (silhouette={results[k_best]["silhouette"]:.4f})')

best = results[k_best]
cluster_labels = best['labels']

# 簇画像 (在原空间解读)
print(f'\n--- {k_best} 簇画像 (原始空间) ---')
for j in range(k_best):
    mask = cluster_labels == j
    cluster_df = df.iloc[mask]
    size = mask.sum()
    print(f'\n  簇 {j+1}: n={size} ({100*size/n:.1f}%)')
    # 各特征均值
    for ci, col in enumerate(feat_cols):
        cmean = X_raw[mask, ci].mean()
        overall_mean = mu[ci]
        direction = '↑' if cmean > overall_mean else '↓'
        print(f'    {col}: {cmean:.1f} (全局{overall_mean:.1f}) {direction}')
    # 作物类型分布
    type_dist = cluster_df['作物名称'].value_counts().head(5)
    print(f'    Top 作物: {dict(type_dist)}')
    # 地块类型分布
    land_dist = cluster_df['地块类型'].value_counts().to_dict()
    print(f'    地块类型: {land_dist}')

# 数据质量标志
for j in range(k_best):
    mask = cluster_labels == j
    size = mask.sum()
    pct = 100 * size / n
    if pct < 2:
        print(f'\n⚠️ 簇 {j+1} 规模极小 ({pct:.1f}%) → 潜在排除候选')
    cluster_df = df.iloc[mask]
    # 检查纯度
    for col in ['作物名称', '地块类型']:
        top_pct = cluster_df[col].value_counts(normalize=True).iloc[0]
        if top_pct > 0.9:
            print(f'🔍 簇 {j+1} 在 {col} 上纯度 {top_pct:.0%} → 强分类线索')

# ── t-SNE 可视化 ──
print('\n--- t-SNE 可视化 ---')
# 抽样 ≤500
if n > 500:
    idx_sample = np.random.choice(n, 500, replace=False)
else:
    idx_sample = np.arange(n)
X_sample = X[idx_sample]
labels_sample = cluster_labels[idx_sample]

# 纯 numpy t-SNE (简化版, Barnes-Hut 太复杂, 用标准 t-SNE)
# 为简单: 使用 sklearn (Path C 允许)
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=min(30, len(idx_sample)//3),
            random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_sample)
print(f'  t-SNE 完成: {X_tsne.shape}')

# 保存 t-SNE 数据
tsne_data = {
    'x': X_tsne[:, 0].tolist(),
    'y': X_tsne[:, 1].tolist(),
    'cluster': labels_sample.tolist(),
    'names': df.iloc[idx_sample]['作物名称'].tolist(),
    'land_types': df.iloc[idx_sample]['地块类型'].tolist(),
}

# ============================================================
# PATH C: 异常检测
# ============================================================
print('\n' + '=' * 60)
print('PATH C: 异常检测')
print('=' * 60)

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Isolation Forest
iso = IsolationForest(contamination='auto', random_state=42)
iso_labels = iso.fit_predict(X)  # 1=正常, -1=异常
iso_scores = iso.score_samples(X)  # 越低越异常
n_iso = (iso_labels == -1).sum()
print(f'  Isolation Forest: {n_iso} 个异常 ({100*n_iso/n:.1f}%)')

# LOF (n < 500 → 追加)
lof = LocalOutlierFactor(n_neighbors=min(20, n//2), contamination='auto')
lof_labels = lof.fit_predict(X)  # 1=正常, -1=异常
lof_scores = -lof.negative_outlier_factor_  # 越高越异常
n_lof = (lof_labels == -1).sum()
print(f'  LOF: {n_lof} 个异常 ({100*n_lof/n:.1f}%)')

# 高置信异常 (两个方法同时标记)
high_conf = (iso_labels == -1) & (lof_labels == -1)
n_high = high_conf.sum()
print(f'  高置信异常 (两法一致): {n_high} 个')

if n_high > 0:
    print(f'\n  高置信异常作物:')
    for idx in np.where(high_conf)[0]:
        row = df.iloc[idx]
        # 偏离方向
        z = (X_raw[idx] - mu) / sigma
        dev_dirs = []
        for ci, col in enumerate(feat_cols):
            if abs(z[ci]) > 2:
                dev_dirs.append(f'{col}: {X_raw[idx, ci]:.1f} (z={z[ci]:+.1f})')
        print(f'    {row["作物名称"]}({row["地块类型"]}): iso={iso_scores[idx]:.3f}, lof={lof_scores[idx]:.3f}')
        if dev_dirs:
            print(f'      偏离>2σ: {"; ".join(dev_dirs)}')

# ── 异常与聚类交叉 ──
print(f'\n  异常样本在聚类中的分布:')
for j in range(k_best):
    mask_c = cluster_labels == j
    n_anom = (mask_c & high_conf).sum()
    if n_anom > 0:
        print(f'    簇 {j+1}: {n_anom} 个')

anomaly_data = {
    'iso_n': int(n_iso), 'lof_n': int(n_lof), 'high_conf_n': int(n_high),
    'high_conf_indices': [int(i) for i in np.where(high_conf)[0]],
    'high_conf_names': df.iloc[high_conf]['作物名称'].tolist(),
    'high_conf_land': df.iloc[high_conf]['地块类型'].tolist(),
    'cluster_dist': {int(j): int((cluster_labels == j).sum()) for j in range(k_best)},
}

# ============================================================
# 保存所有结果
# ============================================================
import json
results_pack = {
    'pca': pca_data,
    'clustering': {
        'k_best': int(k_best),
        'silhouette': float(results[k_best]['silhouette']),
        'all_k': {str(k): {'silhouette': float(v['silhouette']), 'inertia': float(v['inertia']),
                            'sizes': [int(s) for s in v['sizes']]} for k, v in results.items()},
    },
    'tsne': tsne_data,
    'anomaly': anomaly_data,
    'feat_cols': feat_cols,
    'n_samples': n, 'n_features': d,
}

# JSON 用于后续出图脚本
with open(BASE / 'outputs' / 'data' / 'profiling_results.json', 'w', encoding='utf-8') as f:
    json.dump(results_pack, f, ensure_ascii=False, indent=2)
print(f'\n画像结果已保存: profiling_results.json')

print('\n全部画像路径执行完成。')

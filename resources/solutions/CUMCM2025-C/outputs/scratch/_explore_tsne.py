#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""t-SNE 可视化 — 纯 numpy, 抽样加速"""
import numpy as np, pandas as pd, sys, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 加载与特征工程 =====
df = pd.read_pickle('E:/MathModel/problems/2025/C题/outputs/data/2025C-male-clean.pkl')
ivf_dummies = pd.get_dummies(df['IVF妊娠'].fillna('自然受孕'), prefix='conception').astype(float)
features_num = [
    '孕周_数值', '孕妇BMI', '年龄', 'Y染色体浓度', 'X染色体浓度', 'GC含量',
    '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例',
    '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
]
df_f = pd.concat([df[features_num], ivf_dummies], axis=1).dropna()
features = features_num + list(ivf_dummies.columns)
X_raw = df_f.values.astype(float)
X = (X_raw - X_raw.mean(0)) / X_raw.std(0).clip(1e-10)
print(f'全部: {X.shape}')

# 抽样（保留全部 IVF/IUI + 随机采样自然受孕）
ivf_mask = ivf_dummies['conception_IVF（试管婴儿）'].values == 1
iui_mask = ivf_dummies['conception_IUI（人工授精）'].values == 1
rare_mask = ivf_mask | iui_mask
common_idx = np.where(~rare_mask)[0]
rare_idx = np.where(rare_mask)[0]
np.random.seed(42)
sampled_common = np.random.choice(common_idx, min(500, len(common_idx)), replace=False)
sample_idx = np.sort(np.concatenate([rare_idx, sampled_common]))
X_sample = X[sample_idx]
print(f'抽样: {X_sample.shape} (含 IVF={ivf_mask.sum()}, IUI={iui_mask.sum()})')

# ===== k-means =====
def kmeans_simple(X, k=4, seed=42):
    np.random.seed(seed)
    idx = np.random.choice(len(X), k, replace=False)
    C = X[idx].copy()
    for _ in range(100):
        dists = np.sum((X[:,None,:]-C[None,:,:])**2, axis=2)
        labels = np.argmin(dists, axis=1)
        newC = np.array([X[labels==c].mean(0) for c in range(k)])
        if np.allclose(C, newC): break
        C = newC
    return labels

labels = kmeans_simple(X_sample)
print(f'K-Means: {[(labels==c).sum() for c in range(4)]}')

# ===== t-SNE =====
def tsne(X, perplexity=30, n_iter=500, lr=200, seed=42):
    np.random.seed(seed)
    n, d = X.shape
    sum_X = np.sum(X**2, axis=1)
    D = sum_X[:,None] + sum_X[None,:] - 2 * X @ X.T
    D = np.maximum(D, 0)
    
    P = np.zeros((n, n))
    target_h = np.log(perplexity)
    for i in range(n):
        Di = np.delete(D[i], i)
        lo, hi = 1e-10, 1e10
        for _ in range(50):
            s = (lo + hi) / 2
            Pi = np.exp(-Di / (2*s*s))
            sp = Pi.sum()
            if sp < 1e-10: lo = s; continue
            Pi /= sp
            h = -np.sum(Pi * np.log(Pi.clip(1e-15)))
            if abs(h - target_h) < 1e-5: break
            if h > target_h: hi = s
            else: lo = s
        idx = list(range(i)) + list(range(i+1, n))
        P[i, idx] = Pi
    P = (P + P.T) / (2*n)
    P = P.clip(1e-15)
    
    Y = np.random.randn(n, 2) * 1e-4
    dY = np.zeros_like(Y); iY = np.zeros_like(Y)
    gains = np.ones_like(Y)
    
    for it in range(n_iter):
        sum_Y = np.sum(Y**2, axis=1)
        num = 1.0 / (1.0 + sum_Y[:,None] + sum_Y[None,:] - 2*Y@Y.T)
        np.fill_diagonal(num, 0)
        Q = num / num.sum()
        Q = Q.clip(1e-15)
        PQ = P - Q
        
        for i in range(n):
            dY[i] = 4.0 * np.sum((PQ[i] * num[i])[:,None] * (Y[i]-Y), axis=0)
        
        gains = np.where(np.sign(dY) != np.sign(iY), gains*0.8, gains*1.2)
        gains = gains.clip(0.01, None)
        momentum = 0.5 if it < 250 else 0.8
        iY = momentum * iY - lr * gains * dY
        Y += iY
        Y -= Y.mean(0)
        
        if (it+1) % 200 == 0:
            print(f'  iter {it+1}/{n_iter}, KL={np.sum(P*np.log(P/Q)):.4f}', flush=True)
    return Y

print('t-SNE...')
t0 = time.time()
Y = tsne(X_sample, perplexity=30, n_iter=500)
print(f'完成 ({time.time()-t0:.1f}s)')

# ===== 绘图 =====
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for c in range(4):
    m = labels == c
    ax.scatter(Y[m,0], Y[m,1], c=colors[c], s=10, alpha=0.45,
               label=f'Cluster {c} (n={m.sum()})')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE — K-Means Clusters')
ax.legend(fontsize=8)

ax = axes[1]
bmi_groups = df.loc[df_f.index[sample_idx], 'bmi_group']
bmi_order = ['[20,28)','[28,32)','[32,36)','[36,40)','[40,+)']
bmi_colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
for grp, col in zip(bmi_order, bmi_colors):
    m = bmi_groups.values == grp
    ax.scatter(Y[m,0], Y[m,1], c=col, s=10, alpha=0.45,
               label=f'BMI {grp} (n={m.sum()})')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE — BMI Groups')
ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig('E:/MathModel/problems/2025/C题/outputs/figures/kmeans-tsne.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('已保存: outputs/figures/kmeans-tsne.pdf')

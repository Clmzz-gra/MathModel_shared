#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PCA(7) → K-Means(k=2) → t-SNE 可视化"""
import numpy as np, pandas as pd, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 加载 & 标准化 =====
df = pd.read_pickle('E:/MathModel/problems/2025/C题/outputs/data/2025C-male-clean.pkl')
ivf_dummies = pd.get_dummies(df['IVF妊娠'].fillna('自然受孕'), prefix='c').astype(float)
fnum = ['孕周_数值','孕妇BMI','年龄','Y染色体浓度','X染色体浓度','GC含量',
        '在参考基因组上比对的比例','重复读段的比例','被过滤掉读段数的比例',
        '13号染色体的Z值','18号染色体的Z值','21号染色体的Z值','X染色体的Z值']
df_f = pd.concat([df[fnum], ivf_dummies], axis=1).dropna()
X_raw = df_f.values.astype(float)
X = (X_raw - X_raw.mean(0)) / X_raw.std(0).clip(1e-10)
print(f'样本: {X.shape}')

# ===== PCA → 7 =====
Xc = X - X.mean(0)
cov = Xc.T @ Xc / (len(Xc) - 1)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
X_pc7 = Xc @ eigvecs[:, order[:7]]
print(f'PCA 7: {X_pc7.shape}, 累计方差={eigvals[order][:7].sum()/eigvals.sum()*100:.1f}%')

# ===== K-Means k=2 =====
def kmeans(X, k, seed=42):
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

labels = kmeans(X_pc7, k=2)
for c in [0,1]:
    print(f'Cluster {c}: n={(labels==c).sum()}')

# ===== 抽样 → t-SNE =====
ivf_mask = ivf_dummies.iloc[:, 1].values == 1  # IVF
iui_mask = ivf_dummies.iloc[:, 0].values == 1  # IUI
rare = ivf_mask | iui_mask
common = np.where(~rare)[0]
rare_idx = np.where(rare)[0]
np.random.seed(42)
sampled_common = np.random.choice(common, min(500, len(common)), replace=False)
sidx = np.sort(np.concatenate([rare_idx, sampled_common]))
X_sample = X_pc7[sidx]
labels_sample = labels[sidx]
print(f't-SNE 抽样: {X_sample.shape}')

def tsne(X, perplexity=30, n_iter=500, lr=200, seed=42):
    np.random.seed(seed)
    n = len(X)
    sum_X = np.sum(X**2, axis=1)
    D = sum_X[:,None] + sum_X[None,:] - 2 * X @ X.T
    D = np.maximum(D, 0)
    P = np.zeros((n,n))
    target_h = np.log(perplexity)
    for i in range(n):
        Di = np.delete(D[i], i)
        lo, hi = 1e-10, 1e10
        for _ in range(50):
            s = (lo+hi)/2
            Pi = np.exp(-Di/(2*s*s)); sp = Pi.sum()
            if sp < 1e-10: lo=s; continue
            Pi /= sp
            h = -np.sum(Pi*np.log(Pi.clip(1e-15)))
            if abs(h-target_h)<1e-5: break
            if h>target_h: hi=s
            else: lo=s
        idx = list(range(i))+list(range(i+1,n))
        P[i,idx] = Pi
    P = (P+P.T)/(2*n); P = P.clip(1e-15)
    Y = np.random.randn(n,2)*1e-4
    dY=np.zeros_like(Y); iY=np.zeros_like(Y); gains=np.ones_like(Y)
    for it in range(n_iter):
        sum_Y = np.sum(Y**2,axis=1)
        num = 1.0/(1.0+sum_Y[:,None]+sum_Y[None,:]-2*Y@Y.T)
        np.fill_diagonal(num,0); Q=num/num.sum(); Q=Q.clip(1e-15)
        PQ = P-Q
        for i in range(n):
            dY[i] = 4.0*np.sum((PQ[i]*num[i])[:,None]*(Y[i]-Y),axis=0)
        gains = np.where(np.sign(dY)!=np.sign(iY),gains*.8,gains*1.2).clip(0.01)
        momentum = 0.5 if it<250 else 0.8
        iY = momentum*iY - lr*gains*dY; Y += iY; Y -= Y.mean(0)
        if (it+1)%200==0: print(f'  iter {it+1}/{n_iter}, KL={np.sum(P*np.log(P/Q)):.4f}')
    return Y

print('t-SNE...')
t0 = time.time()
Y = tsne(X_sample, n_iter=500)
print(f'完成 ({time.time()-t0:.1f}s)')

# ===== 绘图 =====
colors2 = ['#d62728', '#1f77b4']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左：t-SNE 聚类
ax = axes[0]
for c in [0,1]:
    m = labels_sample == c
    ax.scatter(Y[m,0], Y[m,1], c=colors2[c], s=14, alpha=0.45,
               label=f'Cluster {c} (n={m.sum()})')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.set_title('PCA(7) + K-Means (k=2) — t-SNE')
ax.legend(fontsize=9)

# 右：BMI 分组对照
ax = axes[1]
bmi_groups = df.loc[df_f.index[sidx], 'bmi_group']
bmi_order = ['[20,28)','[28,32)','[32,36)','[36,40)','[40,+)']
bmi_colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
for grp, col in zip(bmi_order, bmi_colors):
    m = bmi_groups.values == grp
    ax.scatter(Y[m,0], Y[m,1], c=col, s=14, alpha=0.4,
               label=f'BMI {grp} (n={m.sum()})')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE — BMI Groups')
ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig('E:/MathModel/problems/2025/C题/outputs/figures/kmeans-pca7-k2-tsne.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('已保存: kmeans-pca7-k2-tsne.pdf')

# ===== 簇特征对比 =====
print('\n' + '='*60)
print('两簇特征均值对比（标准化前原尺度）')
print('='*60)
for c in [0,1]:
    m = labels == c
    print(f'\n--- Cluster {c} (n={m.sum()}) ---')
    for fi, feat in enumerate(fnum):
        print(f'  {feat:30s}: {X_raw[m, fi].mean():.4f}')

# IVF/IUI 分布
print('\n各簇受孕方式:')
for c in [0,1]:
    m = labels == c
    ivf_in = df.loc[df_f.index[m], 'IVF妊娠'].value_counts().to_dict()
    print(f'  Cluster {c}: {ivf_in}')

# 孕周 & Y浓度
for label_name in ['Y染色体浓度', 'X染色体浓度']:
    ci = list(fnum).index(label_name)
    print(f'\n{label_name}:')
    for c in [0,1]:
        m = labels == c
        v = X_raw[m, ci]
        print(f'  Cluster {c}: mean={v.mean():.4f}, std={v.std():.4f}, range=[{v.min():.4f},{v.max():.4f}]')

print('\n孕周:')
ci = 0
for c in [0,1]:
    m = labels == c
    v = X_raw[m, ci]
    print(f'  Cluster {c}: mean={v.mean():.1f}, std={v.std():.1f}, range=[{v.min():.1f},{v.max():.1f}]')

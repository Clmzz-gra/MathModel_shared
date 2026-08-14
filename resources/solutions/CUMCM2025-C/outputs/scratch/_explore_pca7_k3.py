#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PCA(7) → K-Means(k=3) → t-SNE"""
import numpy as np, pandas as pd, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_pickle('E:/MathModel/problems/2025/C题/outputs/data/2025C-male-clean.pkl')
ivf_dummies = pd.get_dummies(df['IVF妊娠'].fillna('自然受孕'), prefix='c').astype(float)
fnum = ['孕周_数值','孕妇BMI','年龄','Y染色体浓度','X染色体浓度','GC含量',
        '在参考基因组上比对的比例','重复读段的比例','被过滤掉读段数的比例',
        '13号染色体的Z值','18号染色体的Z值','21号染色体的Z值','X染色体的Z值']
df_f = pd.concat([df[fnum], ivf_dummies], axis=1).dropna()
X = (df_f.values.astype(float) - df_f.values.mean(0)) / df_f.values.std(0).clip(1e-10)
Xc = X - X.mean(0)
cov = Xc.T @ Xc / (len(Xc)-1)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
X7 = Xc @ eigvecs[:, order[:7]]

# k=3
np.random.seed(42)
C = X7[np.random.choice(len(X7), 3, replace=False)].copy()
for _ in range(100):
    dists = np.sum((X7[:,None,:]-C[None,:,:])**2, axis=2)
    labels = np.argmin(dists, axis=1)
    newC = np.array([X7[labels==c].mean(0) for c in range(3)])
    if np.allclose(C, newC): break
    C = newC

# 抽样 t-SNE
ivf_mask = ivf_dummies.iloc[:,1].values==1
iui_mask = ivf_dummies.iloc[:,0].values==1
rare = ivf_mask | iui_mask
common = np.where(~rare)[0]
np.random.seed(42)
sidx = np.sort(np.concatenate([np.where(rare)[0], np.random.choice(common, min(500,len(common)), replace=False)]))
Xs, ls = X7[sidx], labels[sidx]
print(f't-SNE 抽样: {Xs.shape}, 簇分布: {np.bincount(ls)}')

def tsne(X, perplexity=30, n_iter=500, lr=200, seed=42):
    np.random.seed(seed); n=len(X)
    sum_X = np.sum(X**2,axis=1); D = sum_X[:,None]+sum_X[None,:]-2*X@X.T; D=np.maximum(D,0)
    P=np.zeros((n,n)); th=np.log(perplexity)
    for i in range(n):
        Di=np.delete(D[i],i); lo,hi=1e-10,1e10
        for _ in range(50):
            s=(lo+hi)/2; Pi=np.exp(-Di/(2*s*s)); sp=Pi.sum()
            if sp<1e-10: lo=s; continue
            Pi/=sp; h=-np.sum(Pi*np.log(Pi.clip(1e-15)))
            if abs(h-th)<1e-5: break
            if h>th: hi=s
            else: lo=s
        idx=list(range(i))+list(range(i+1,n)); P[i,idx]=Pi
    P=(P+P.T)/(2*n); P=P.clip(1e-15)
    Y=np.random.randn(n,2)*1e-4; dY=np.zeros_like(Y); iY=np.zeros_like(Y); gains=np.ones_like(Y)
    for it in range(n_iter):
        sum_Y=np.sum(Y**2,axis=1); num=1.0/(1.0+sum_Y[:,None]+sum_Y[None,:]-2*Y@Y.T)
        np.fill_diagonal(num,0); Q=num/num.sum(); Q=Q.clip(1e-15); PQ=P-Q
        for i in range(n): dY[i]=4.0*np.sum((PQ[i]*num[i])[:,None]*(Y[i]-Y),axis=0)
        gains=np.where(np.sign(dY)!=np.sign(iY),gains*.8,gains*1.2).clip(0.01)
        momentum=0.5 if it<250 else 0.8
        iY=momentum*iY-lr*gains*dY; Y+=iY; Y-=Y.mean(0)
        if (it+1)%200==0: print(f'  iter {it+1}/{n_iter}, KL={np.sum(P*np.log(P/Q)):.4f}')
    return Y

print('t-SNE...'); t0=time.time()
Y = tsne(Xs, n_iter=500)
print(f'完成 ({time.time()-t0:.1f}s)')

# 绘图
colors3 = ['#d62728','#ff7f0e','#1f77b4']
fig, axes = plt.subplots(1,2,figsize=(14,6))
ax=axes[0]
for c in range(3):
    m=ls==c; ax.scatter(Y[m,0],Y[m,1],c=colors3[c],s=14,alpha=0.45,label=f'Cluster {c} (n={m.sum()})')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.set_title('PCA(7) + K-Means (k=3) — t-SNE'); ax.legend(fontsize=9)

ax=axes[1]
bmi_groups=df.loc[df_f.index[sidx],'bmi_group']
for grp,col in zip(['[20,28)','[28,32)','[32,36)','[36,40)','[40,+)'],['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']):
    m=bmi_groups.values==grp; ax.scatter(Y[m,0],Y[m,1],c=col,s=14,alpha=0.4,label=f'BMI {grp} (n={m.sum()})')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE — BMI Groups'); ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig('E:/MathModel/problems/2025/C题/outputs/figures/kmeans-pca7-k3-tsne.pdf',dpi=150,bbox_inches='tight')
plt.close()
print('已保存: kmeans-pca7-k3-tsne.pdf')

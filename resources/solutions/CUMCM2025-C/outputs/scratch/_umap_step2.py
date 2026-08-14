# -*- coding: utf-8 -*-
"""Step 2: Clustering + Visualization from saved UMAP embeddings"""
import sys, time, os
os.environ['NUMBA_DISABLE_JIT'] = '1'
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200

DATA = "E:/MathModel/problems/2025/C题/2025C题测试/outputs/data"
FIGS = "E:/MathModel/problems/2025/C题/2025C题测试/outputs/figures"

print("Loading embeddings...")
Xm = np.load(f"{DATA}/umap_male.npy")
meta_m = pd.read_pickle(f"{DATA}/umap_male_meta.pkl")
Xf = np.load(f"{DATA}/umap_female.npy")
meta_f = pd.read_pickle(f"{DATA}/umap_female_meta.pkl")
print(f"Male: {Xm.shape}, Female: {Xf.shape}")

# ====== Clustering ======
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

print("\n--- Male Clustering ---")
h_m = HDBSCAN(min_cluster_size=8, min_samples=5, metric='euclidean')
lh_m = h_m.fit_predict(Xm)
n_cls_m = len(set(lh_m)) - (1 if -1 in lh_m else 0)
n_noise_m = (lh_m == -1).sum()
print(f"HDBSCAN: {n_cls_m} clusters + {n_noise_m} noise ({n_noise_m/len(lh_m):.1%})")

km_m = KMeans(n_clusters=4, random_state=42, n_init=20)
lk_m = km_m.fit_predict(Xm)
sil_km_m = silhouette_score(Xm, lk_m)
nn_m = lh_m != -1
sil_h_m = silhouette_score(Xm[nn_m], lh_m[nn_m]) if nn_m.sum() > 1 and n_cls_m >= 2 else None
print(f"Silhouette: HDB={sil_h_m:.4f}" if sil_h_m else "Sil: HDB=N/A", f"| KMeans={sil_km_m:.4f}")

print("\n--- Female Clustering ---")
h_f = HDBSCAN(min_cluster_size=8, min_samples=5, metric='euclidean')
lh_f = h_f.fit_predict(Xf)
n_cls_f = len(set(lh_f)) - (1 if -1 in lh_f else 0)
n_noise_f = (lh_f == -1).sum()
print(f"HDBSCAN: {n_cls_f} clusters + {n_noise_f} noise ({n_noise_f/len(lh_f):.1%})")

km_f = KMeans(n_clusters=4, random_state=42, n_init=20)
lk_f = km_f.fit_predict(Xf)
sil_km_f = silhouette_score(Xf, lk_f)
nn_f = lh_f != -1
sil_h_f = silhouette_score(Xf[nn_f], lh_f[nn_f]) if nn_f.sum() > 1 and n_cls_f >= 2 else None
print(f"Silhouette: HDB={sil_h_f:.4f}" if sil_h_f else "Sil: HDB=N/A", f"| KMeans={sil_km_f:.4f}")

ab_f = meta_f['AB_异常'].values.astype(int)
ari_h = adjusted_rand_score(ab_f, lh_f)
nmi_h = normalized_mutual_info_score(ab_f, lh_f)
ari_k = adjusted_rand_score(ab_f, lk_f)
nmi_k = normalized_mutual_info_score(ab_f, lk_f)
print(f"vs AB: HDB ARI={ari_h:.4f} NMI={nmi_h:.4f} | KM ARI={ari_k:.4f} NMI={nmi_k:.4f}")

# ====== Visualization ======
print("\n--- Drawing ---")
cmap10 = plt.cm.tab10
BMI_CATS = ['[20,28)','[28,32)','[32,36)','[36,40)','[40,+)']
BMI_PAL = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
bmi_m = meta_m['bmi_group'].values

fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

# Row 0: Male UMAP
# [0,0] HDBSCAN
ax = fig.add_subplot(gs[0,0])
for c in sorted(set(lh_m)):
    m = lh_m==c; clr = '#888' if c==-1 else cmap10(c%10)
    lbl = f'Noise({m.sum()})' if c==-1 else f'C{c}({m.sum()})'
    ax.scatter(Xm[m,0],Xm[m,1], c=clr, s=14, alpha=0.5, label=lbl)
ax.set_title(f'Male HDBSCAN ({n_cls_m}+noise)', fontsize=10)
ax.legend(fontsize=6, loc='lower left', ncol=3); ax.set_xticks([]); ax.set_yticks([])

# [0,1] BMI
ax = fig.add_subplot(gs[0,1])
for grp,clr in zip(BMI_CATS, BMI_PAL):
    m = bmi_m==grp; ax.scatter(Xm[m,0],Xm[m,1], c=clr, s=14, alpha=0.5, label=grp)
ax.set_title('Male BMI Groups', fontsize=10)
ax.legend(fontsize=6, ncol=3); ax.set_xticks([]); ax.set_yticks([])

# [0,2] K-Means
ax = fig.add_subplot(gs[0,2])
for c in range(4):
    m = lk_m==c; ax.scatter(Xm[m,0],Xm[m,1], c=cmap10(c), s=12, alpha=0.45, label=f'K{c}({m.sum()})')
ax.set_title(f'Male K-Means k=4 (Sil={sil_km_m:.3f})', fontsize=10)
ax.legend(fontsize=6); ax.set_xticks([]); ax.set_yticks([])

# [0,3] Summary
ax = fig.add_subplot(gs[0,3]); ax.axis('off')
lines_m = [
    "===== Male Results =====",
    f"n={Xm.shape[0]} (dedup), d={Xm.shape[1]}",
    f"HDBSCAN: {n_cls_m} clusters + {n_noise_m} noise ({n_noise_m/len(lh_m):.1%})",
    f"Silhouette: HDB={sil_h_m:.4f}" if sil_h_m else "Silhouette: HDB=N/A",
    f"K-Means k=4 Silhouette: {sil_km_m:.4f}",
    f"",
    f"UMAP: n_neighbors=15, min_dist=0.1",
    f"Features: Y conc, BMI, GW, GC, Z-scores, tech QC",
]
for i,l in enumerate(lines_m):
    ax.text(0.05, 0.95-i*0.07, l, transform=ax.transAxes, fontsize=7.5, fontfamily='monospace')

# Row 1: Female UMAP
ax = fig.add_subplot(gs[1,0])
for c in sorted(set(lh_f)):
    m = lh_f==c; clr = '#888' if c==-1 else cmap10(c%10)
    lbl = f'Noise({m.sum()})' if c==-1 else f'C{c}({m.sum()})'
    ax.scatter(Xf[m,0],Xf[m,1], c=clr, s=18, alpha=0.55, label=lbl)
ax.set_title(f'Female HDBSCAN ({n_cls_f}+noise)', fontsize=10)
ax.legend(fontsize=6, loc='lower left', ncol=2); ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(gs[1,1])
for val,clr,lbl in [(0,'#2ca02c','Normal'),(1,'#d62728','Abnormal')]:
    m = ab_f==val; ax.scatter(Xf[m,0],Xf[m,1], c=clr, s=18, alpha=0.55, label=lbl)
ax.set_title('Female AB Labels (ground truth)', fontsize=10)
ax.legend(fontsize=7); ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(gs[1,2])
for c in range(4):
    m = lk_f==c; ax.scatter(Xf[m,0],Xf[m,1], c=cmap10(c), s=15, alpha=0.5, label=f'K{c}({m.sum()})')
ab_m = ab_f==1; ax.scatter(Xf[ab_m,0],Xf[ab_m,1], facecolors='none', edgecolors='red', s=50, lw=1.5, label=f'AB=1({ab_m.sum()})')
ax.set_title(f'Female K-Means k=4 (Sil={sil_km_f:.3f})', fontsize=10)
ax.legend(fontsize=6, ncol=2); ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(gs[1,3]); ax.axis('off')
lines_f = [
    "===== Female Results =====",
    f"n={Xf.shape[0]} (dedup), abnormal={ab_f.sum()} ({ab_f.sum()/len(ab_f):.1%})",
    f"HDBSCAN: {n_cls_f} clusters + {n_noise_f} noise ({n_noise_f/len(lh_f):.1%})",
    f"Silhouette: HDB={sil_h_f:.4f}" if sil_h_f else "Silhouette: HDB=N/A",
    f"K-Means k=4 Silhouette: {sil_km_f:.4f}",
    f"vs AB: HDB ARI={ari_h:.4f} NMI={nmi_h:.4f}",
    f"        KM  ARI={ari_k:.4f} NMI={nmi_k:.4f}",
    f"",
    f"Features: X conc, BMI, GW, GC, Z-scores, tech QC",
]
for i,l in enumerate(lines_f):
    ax.text(0.05, 0.95-i*0.07, l, transform=ax.transAxes, fontsize=7.5, fontfamily='monospace')

# Row 2: Confusion matrices
# [2,0] Male cluster vs BMI
ax = fig.add_subplot(gs[2,0])
cross = pd.crosstab(pd.Series(lh_m).replace(-1,'Noise'), pd.Series(bmi_m))
cpct = cross.div(cross.sum(1), axis=0)
im = ax.imshow(cpct.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(cpct.columns))); ax.set_xticklabels(cpct.columns, fontsize=7, rotation=45, ha='right')
ax.set_yticks(range(len(cpct.index))); ax.set_yticklabels(cpct.index, fontsize=7)
for i in range(cpct.shape[0]):
    for j in range(cpct.shape[1]):
        ax.text(j,i,f'{cpct.iloc[i,j]:.2f}', ha='center',va='center',fontsize=6,
                color='white' if cpct.iloc[i,j]>0.6 else 'black')
ax.set_title('Male: Cluster -> BMI (row-%)', fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046)

# [2,1] Female cluster vs AB
ax = fig.add_subplot(gs[2,1])
cross_f = pd.crosstab(pd.Series(lh_f).replace(-1,'Noise'),
                       pd.Series(ab_f).replace({0:'Normal',1:'Abnormal'}))
cpct_f = cross_f.div(cross_f.sum(1), axis=0)
im = ax.imshow(cpct_f.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(cpct_f.columns))); ax.set_xticklabels(cpct_f.columns, fontsize=8)
ax.set_yticks(range(len(cpct_f.index))); ax.set_yticklabels(cpct_f.index, fontsize=7)
for i in range(cpct_f.shape[0]):
    for j in range(cpct_f.shape[1]):
        ax.text(j,i,f'{cpct_f.iloc[i,j]:.2f}', ha='center',va='center',fontsize=7,
                color='white' if cpct_f.iloc[i,j]>0.6 else 'black')
ax.set_title(f'Female: Cluster -> AB (ARI={ari_h:.3f})', fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046)

# [2,2] Outlier scores
ax = fig.add_subplot(gs[2,2])
scores = h_f.outlier_scores_
for val,clr,lbl in [(0,'#2ca02c','Normal'),(1,'#d62728','Abnormal')]:
    m = ab_f==val; ax.scatter(np.arange(len(scores))[m], scores[m], c=clr, s=10, alpha=0.4, label=lbl)
for q,ls in [(0.50,'--'),(0.90,'-.')]:
    ax.axhline(np.percentile(scores,q*100), color='gray', ls=ls, lw=0.8)
ax.set_xlabel('Sample Index', fontsize=8); ax.set_ylabel('Outlier Score', fontsize=8)
ax.set_title('Female HDBSCAN Outlier Score', fontsize=10); ax.legend(fontsize=7)

# [2,3] Male Outlier scores
ax = fig.add_subplot(gs[2,3])
scores_m = h_m.outlier_scores_
for grp,clr in zip(BMI_CATS, BMI_PAL):
    m = bmi_m==grp; ax.scatter(np.arange(len(scores_m))[m], scores_m[m], c=clr, s=8, alpha=0.35, label=grp)
for q,ls in [(0.50,'--'),(0.90,'-.')]:
    ax.axhline(np.percentile(scores_m,q*100), color='gray', ls=ls, lw=0.8)
ax.set_xlabel('Sample Index', fontsize=8); ax.set_ylabel('Outlier Score', fontsize=8)
ax.set_title('Male HDBSCAN Outlier Score (by BMI)', fontsize=10); ax.legend(fontsize=6, ncol=3)

fig.suptitle('UMAP Clustering Analysis — 2025 C Problem (NIPT)', fontsize=14, fontweight='bold', y=0.99)
out = f"{FIGS}/umap-clustering-full.pdf"
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print("\n=== COMPLETE ===")

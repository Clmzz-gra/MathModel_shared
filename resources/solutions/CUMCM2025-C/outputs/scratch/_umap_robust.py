# -*- coding: utf-8 -*-
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')

LOG = "E:/MathModel/problems/2025/C题/2025C题测试/outputs/scratch/umap_log.txt"
def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(msg, flush=True)

log("=== UMAP Test Start ===")

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

log("Loading data...")
dfm = pd.read_pickle("E:/MathModel/problems/2025/C题/2025C题测试/outputs/data/2025C-male-clean.pkl")
df1 = dfm.sort_values(["孕妇代码", "孕周_数值"]).drop_duplicates("孕妇代码", keep="first")
log(f"Dedup: {df1.shape[0]}")

cols = ['孕周_数值', '孕妇BMI', '年龄', 'Y染色体浓度', 'X染色体浓度',
        'GC含量', '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例',
        '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
ivf = pd.get_dummies(df1['IVF妊娠'].fillna('自然受孕'), prefix='IVF').astype(float)
feat = pd.concat([df1[cols].reset_index(drop=True), ivf.reset_index(drop=True)], axis=1).dropna().astype(float)
X = StandardScaler().fit_transform(feat.values)
log(f"Features: {X.shape}")

log("UMAP...")
from umap import UMAP
u = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=42)
t0 = time.time()
Xu = u.fit_transform(X)
log(f"UMAP done: {Xu.shape}, {time.time()-t0:.1f}s")

log("HDBSCAN...")
from hdbscan import HDBSCAN
h = HDBSCAN(min_cluster_size=8, min_samples=5)
lh = h.fit_predict(Xu)
log(f"HDBSCAN: {len(set(lh))} clusters, noise={(lh==-1).sum()}")

# Female
log("Loading female...")
dff = pd.read_pickle("E:/MathModel/problems/2025/C题/2025C题测试/outputs/data/2025C-female-clean.pkl")
df_f1 = dff.sort_values(["孕妇代码", "孕周_数值"]).drop_duplicates("孕妇代码", keep="first")
log(f"Female dedup: {df_f1.shape[0]}")

fcols = ['孕周_数值', '孕妇BMI', '年龄', 'X染色体浓度',
         'GC含量', '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例',
         '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
fivf = pd.get_dummies(df_f1['IVF妊娠'].fillna('自然受孕'), prefix='IVF').astype(float)
ffeat = pd.concat([df_f1[fcols].reset_index(drop=True), fivf.reset_index(drop=True)], axis=1).dropna().astype(float)
Xf = StandardScaler().fit_transform(ffeat.values)
log(f"Female features: {Xf.shape}")

log("Female UMAP...")
Xf_u = u.fit_transform(Xf)
log(f"Female UMAP done: {Xf_u.shape}")

log("Female HDBSCAN...")
h_f = HDBSCAN(min_cluster_size=8, min_samples=5)
lh_f = h_f.fit_predict(Xf_u)
ab_f = df_f1.loc[ffeat.index, 'AB_异常'].values.astype(int)
log(f"Female HDBSCAN: {len(set(lh_f))} clusters, noise={(lh_f==-1).sum()}, AB=1: {ab_f.sum()}")

# Cross-tab
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(ab_f, lh_f)
log(f"AB vs HDBSCAN ARI: {ari:.4f}")

# KMeans baseline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
km = KMeans(n_clusters=4, random_state=42, n_init=20)
lk_m = km.fit_predict(Xu)
lk_f = km.fit_predict(Xf_u)
sil_m = silhouette_score(Xu, lk_m)
sil_f = silhouette_score(Xf_u, lk_f)
log(f"KMeans Sil: male={sil_m:.4f}, female={sil_f:.4f}")

# Save plots
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

bmi_m = df1.loc[ffeat.index if False else feat.index, 'bmi_group'].values

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
bmi_cats = ['[20,28)','[28,32)','[32,36)','[36,40)','[40,+)']
bmi_pal = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

# Male plots
for c in sorted(set(lh)):
    m = lh==c; clr = '#888' if c==-1 else plt.cm.tab10(c%10)
    lbl = f'Noise({m.sum()})' if c==-1 else f'C{c}({m.sum()})'
    axes[0,0].scatter(Xu[m,0], Xu[m,1], c=clr, s=15, alpha=0.5, label=lbl)
axes[0,0].set_title(f'Male HDBSCAN ({len(set(lh))-1}+noise)'); axes[0,0].legend(fontsize=6)

for grp,clr in zip(bmi_cats, bmi_pal):
    m = bmi_m==grp; axes[0,1].scatter(Xu[m,0], Xu[m,1], c=clr, s=15, alpha=0.5, label=grp)
axes[0,1].set_title('Male BMI Groups'); axes[0,1].legend(fontsize=6)

sc = axes[0,2].scatter(Xu[:,0], Xu[:,1], c=feat['Y染色体浓度'].values, cmap='viridis', s=12, alpha=0.5)
plt.colorbar(sc, ax=axes[0,2]).set_label('Y Conc')
axes[0,2].set_title('Male Y Concentration')

# Female plots
for c in sorted(set(lh_f)):
    m = lh_f==c; clr = '#888' if c==-1 else plt.cm.tab10(c%10)
    lbl = f'Noise({m.sum()})' if c==-1 else f'C{c}({m.sum()})'
    axes[1,0].scatter(Xf_u[m,0], Xf_u[m,1], c=clr, s=18, alpha=0.55, label=lbl)
axes[1,0].set_title(f'Female HDBSCAN ({len(set(lh_f))-1}+noise, ARI={ari:.3f})'); axes[1,0].legend(fontsize=6)

for val,clr,lbl in [(0,'#2ca02c','Normal'),(1,'#d62728','Abnormal')]:
    m = ab_f==val; axes[1,1].scatter(Xf_u[m,0], Xf_u[m,1], c=clr, s=18, alpha=0.55, label=lbl)
axes[1,1].set_title('Female AB Labels'); axes[1,1].legend(fontsize=7)

sc = axes[1,2].scatter(Xf_u[:,0], Xf_u[:,1], c=ffeat['X染色体浓度'].values, cmap='RdYlBu_r', s=14, alpha=0.55)
plt.colorbar(sc, ax=axes[1,2]).set_label('X Conc')
axes[1,2].set_title('Female X Concentration')

fig.suptitle('UMAP Clustering — 2025 C Problem', fontsize=14, fontweight='bold')
out_path = "E:/MathModel/problems/2025/C题/2025C题测试/outputs/figures/umap-clustering.pdf"
fig.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
log(f"Saved: {out_path}")
log("=== COMPLETE ===")

# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
print("TEST START")

import numpy as np, pandas as pd
print("Imports OK")

# Load
dfm = pd.read_pickle("E:/MathModel/problems/2025/C题/2025C题测试/outputs/data/2025C-male-clean.pkl")
print(f"Male loaded: {dfm.shape}")

# Dedup
df1 = dfm.sort_values(["孕妇代码", "孕周_数值"]).drop_duplicates("孕妇代码", keep="first")
print(f"Dedup: {len(df1)}")

# Features
cols = ['孕周_数值', '孕妇BMI', '年龄', 'Y染色体浓度', 'X染色体浓度',
        'GC含量', '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例',
        '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
ivf = pd.get_dummies(df1['IVF妊娠'].fillna('自然受孕'), prefix='IVF').astype(float)
feat = pd.concat([df1[cols].reset_index(drop=True), ivf.reset_index(drop=True)], axis=1).dropna().astype(float)
print(f"Features: {feat.shape}")

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(feat.values)
print(f"Scaled: {X.shape}")

from umap import UMAP
print("Starting UMAP...")
u = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=42, verbose=True)
Xu = u.fit_transform(X)
print(f"UMAP done: {Xu.shape}")

from hdbscan import HDBSCAN
h = HDBSCAN(min_cluster_size=8, min_samples=5)
lh = h.fit_predict(Xu)
print(f"HDBSCAN: {len(set(lh))} labels, noise={(lh==-1).sum()}")

# Save simple plot
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

bmi_groups = df1.loc[feat.index, 'bmi_group'].values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
bmi_colors = {'[20,28)':'#1f77b4','[28,32)':'#ff7f0e','[32,36)':'#2ca02c','[36,40)':'#d62728','[40,+)':'#9467bd'}

for c in sorted(set(lh)):
    m = lh==c; clr = '#888' if c==-1 else plt.cm.tab10(c%10)
    lbl = f'Noise({m.sum()})' if c==-1 else f'C{c}({m.sum()})'
    ax1.scatter(Xu[m,0], Xu[m,1], c=clr, s=15, alpha=0.5, label=lbl)
ax1.set_title('HDBSCAN'); ax1.legend(fontsize=7)

for grp,clr in bmi_colors.items():
    m = bmi_groups==grp; ax2.scatter(Xu[m,0], Xu[m,1], c=clr, s=15, alpha=0.5, label=grp)
ax2.set_title('BMI Groups'); ax2.legend(fontsize=7)

out = "E:/MathModel/problems/2025/C题/2025C题测试/outputs/figures/umap-male-simple.pdf"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print("DONE")

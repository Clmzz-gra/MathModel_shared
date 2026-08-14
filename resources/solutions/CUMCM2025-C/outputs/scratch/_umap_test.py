# -*- coding: utf-8 -*-
import sys, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

print("Load...")
dfm = pd.read_pickle("E:/MathModel/problems/2025/C题/2025C题测试/outputs/data/2025C-male-clean.pkl")
df1 = dfm.sort_values(["孕妇代码", "孕周_数值"]).drop_duplicates("孕妇代码", keep="first")

cols = ['孕周_数值', '孕妇BMI', '年龄', 'Y染色体浓度', 'X染色体浓度',
        'GC含量', '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例',
        '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
ivf = pd.get_dummies(df1['IVF妊娠'].fillna('自然受孕'), prefix='IVF').astype(float)
feat = pd.concat([df1[cols].reset_index(drop=True), ivf.reset_index(drop=True)], axis=1).dropna().astype(float)
X = StandardScaler().fit_transform(feat.values)

print(f"X: {X.shape}")

# Test UMAP with error catch
print("Testing UMAP...")
try:
    from umap import UMAP
    u = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=42)
    t0 = time.time()
    Xu = u.fit_transform(X)
    t1 = time.time()
    print(f"UMAP done: {Xu.shape}, time={t1-t0:.1f}s")
    print(f"Xu range: [{Xu[:,0].min():.3f},{Xu[:,0].max():.3f}] x [{Xu[:,1].min():.3f},{Xu[:,1].max():.3f}]")
except Exception as e:
    print(f"UMAP ERROR: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()

# Test HDBSCAN
print("Testing HDBSCAN...")
try:
    from hdbscan import HDBSCAN
    h = HDBSCAN(min_cluster_size=8, min_samples=5)
    lh = h.fit_predict(Xu)
    print(f"HDBSCAN: {len(set(lh))} unique labels, noise={(lh==-1).sum()}")
except Exception as e:
    print(f"HDBSCAN ERROR: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()

print("ALL DONE")

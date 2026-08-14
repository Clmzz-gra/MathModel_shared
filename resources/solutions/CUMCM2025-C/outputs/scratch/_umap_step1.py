# -*- coding: utf-8 -*-
"""Step 1: UMAP embedding only, save to .npy"""
import sys, time, os
os.environ['NUMBA_DISABLE_JIT'] = '1'
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

DATA = "E:/MathModel/problems/2025/C题/2025C题测试/outputs/data"

def process(name, path):
    print(f"\n=== {name} ===")
    df = pd.read_pickle(path)
    df1 = df.sort_values(["孕妇代码","孕周_数值"]).drop_duplicates("孕妇代码", keep="first")
    print(f"  Dedup: {df1.shape[0]}")

    if name == "male":
        cols = ['孕周_数值','孕妇BMI','年龄','Y染色体浓度','X染色体浓度',
                'GC含量','在参考基因组上比对的比例','重复读段的比例','被过滤掉读段数的比例',
                '13号染色体的Z值','18号染色体的Z值','21号染色体的Z值','X染色体的Z值']
    else:
        cols = ['孕周_数值','孕妇BMI','年龄','X染色体浓度',
                'GC含量','在参考基因组上比对的比例','重复读段的比例','被过滤掉读段数的比例',
                '13号染色体的Z值','18号染色体的Z值','21号染色体的Z值','X染色体的Z值']

    ivf = pd.get_dummies(df1['IVF妊娠'].fillna('自然受孕'), prefix='IVF').astype(float)
    feat = pd.concat([df1[cols].reset_index(drop=True), ivf.reset_index(drop=True)], axis=1)
    feat = feat.dropna().astype(float)
    df_kept = df1.iloc[feat.index].copy()  # feat index = original df1 positions
    print(f"  After dropna: {feat.shape[0]} (dropped {df1.shape[0]-feat.shape[0]})")

    X = StandardScaler().fit_transform(feat.values)
    print(f"  Features: {X.shape}")

    from umap import UMAP
    u = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=42, verbose=True)
    t0 = time.time()
    Xu = u.fit_transform(X)
    print(f"  UMAP: {time.time()-t0:.1f}s")

    np.save(f"{DATA}/umap_{name}.npy", Xu)
    meta = df_kept[['孕妇代码','bmi_group']].copy()
    if name == "female":
        meta['AB_异常'] = df_kept['AB_异常'].values
    meta.to_pickle(f"{DATA}/umap_{name}_meta.pkl")
    print(f"  Saved: umap_{name}.npy, umap_{name}_meta.pkl")
    print(f"  Xu range: [{Xu[:,0].min():.3f},{Xu[:,0].max():.3f}] [{Xu[:,1].min():.3f},{Xu[:,1].max():.3f}]")
    return Xu, meta

Xm, meta_m = process("male", f"{DATA}/2025C-male-clean.pkl")
Xf, meta_f = process("female", f"{DATA}/2025C-female-clean.pkl")
print("\n=== STEP 1 COMPLETE ===")

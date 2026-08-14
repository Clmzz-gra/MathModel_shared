# -*- coding: utf-8 -*-
import sys, time
sys.stdout.reconfigure(encoding='utf-8')

LOG = "E:/MathModel/problems/2025/C题/2025C题测试/outputs/scratch/umap_log2.txt"
def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(msg, flush=True)

log("=== UMAP test with numba disabled ===")

# Disable numba JIT
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'
log("NUMBA_DISABLE_JIT=1 set")

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

log("Loading...")
dfm = pd.read_pickle("E:/MathModel/problems/2025/C题/2025C题测试/outputs/data/2025C-male-clean.pkl")
df1 = dfm.sort_values(["孕妇代码", "孕周_数值"]).drop_duplicates("孕妇代码", keep="first")
cols = ['孕周_数值', '孕妇BMI', '年龄', 'Y染色体浓度', 'X染色体浓度',
        'GC含量', '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例',
        '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
ivf = pd.get_dummies(df1['IVF妊娠'].fillna('自然受孕'), prefix='IVF').astype(float)
feat = pd.concat([df1[cols].reset_index(drop=True), ivf.reset_index(drop=True)], axis=1).dropna().astype(float)
X = StandardScaler().fit_transform(feat.values)
log(f"X: {X.shape}")

# Try UMAP
log("UMAP fit (no JIT, may be slow)...")
try:
    from umap import UMAP
    u = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=42, verbose=True)
    t0 = time.time()
    Xu = u.fit_transform(X)
    t1 = time.time()
    log(f"UMAP done: {Xu.shape}, {t1-t0:.1f}s")
    log(f"Range: [{Xu[:,0].min():.3f},{Xu[:,0].max():.3f}] [{Xu[:,1].min():.3f},{Xu[:,1].max():.3f}]")
except Exception as e:
    log(f"UMAP ERROR: {type(e).__name__}: {e}")
    import traceback; log(traceback.format_exc())

log("=== END ===")

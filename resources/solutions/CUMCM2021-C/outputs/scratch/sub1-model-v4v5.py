"""
v4: 去掉品类 one-hot（5 特征）
v5: 降权品类 one-hot（品类列 scale=0.5）
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCRATCH_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRATCH_DIR, "..", "data")
FIG_DIR  = os.path.join(SCRATCH_DIR, "..", "figures")
RESULT_DIR = os.path.join(SCRATCH_DIR, "..", "..", "solution", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

df_v2 = pd.read_pickle(os.path.join(DATA_DIR, "sub1-preprocessed-v2.pkl"))
ids = df_v2["供应商ID"].values
categories = df_v2["品类"].values
n = len(df_v2)

expected = ["S229","S361","S140","S108","S151"]

def run_pca(name, X, num_cols):
    """标准 PCA 流程，返回 I, rank, top50_idx, m, evr, loadings"""
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    pca = PCA()
    Y = pca.fit_transform(Z)
    ev, evr = pca.explained_variance_, pca.explained_variance_ratio_
    m = (ev >= 1.0).sum()
    w = evr[:m] / evr[:m].sum()
    I_tmp = np.sum(Y[:, :m] * w, axis=1)
    I = (I_tmp - I_tmp.min()) / (I_tmp.max() - I_tmp.min())
    rank = np.argsort(-I)
    top50 = rank[:50]
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"特征: {num_cols}")
    print(f"Kaiser PC: m={m}, 累计={np.cumsum(evr)[m-1]*100:.1f}%")
    
    loadings = pca.components_[:m]
    for k in range(m):
        top_j = np.argsort(np.abs(loadings[k]))[::-1][:4]
        items = [f"{num_cols[j]}({loadings[k,j]:+.3f})" for j in top_j]
        print(f"  PC{k+1} ({evr[k]*100:.1f}%): {' | '.join(items)}")
    
    print(f"\n{'排名':>4s} {'ID':>6s} {'品':>2s} {'I':>8s} {'供货总量':>10s}")
    print(f"{'-'*36}")
    for r, idx in enumerate(top50[:12]):
        print(f"{r+1:>4d}  {ids[idx]:>6s}  {categories[idx]:>2s}  {I[idx]:>8.4f}  {df_v2['供货总量'].iloc[idx]:>10.0f}")
    
    print(f"\n品类: A={(categories[top50]=='A').sum()}, B={(categories[top50]=='B').sum()}, C={(categories[top50]=='C').sum()}")
    print(f"供货占比: {df_v2['供货总量'].iloc[top50].sum()/df_v2['供货总量'].sum()*100:.1f}%")
    print(f"SP-006 验证:")
    for sid in expected:
        pos = np.where(ids == sid)[0][0]
        r = np.where(rank == pos)[0][0] + 1
        print(f"  {sid}: #{r}, I={I[pos]:.4f}")
    
    return I, rank, top50

# ============================================================
# v4: 无品类 one-hot
# ============================================================
num_cols_v4 = ["供货总量","供货周数","供货满足率","供订CV差","可靠性趋势"]
X_v4 = df_v2[num_cols_v4].values.astype(float)
I_v4, rank_v4, top50_v4 = run_pca("v4: 无品类 one-hot", X_v4, num_cols_v4)

# 保存
df_v4 = df_v2.copy()
df_v4["安全指数_I"] = I_v4; df_v4["排名"] = np.argsort(np.argsort(-I_v4)) + 1
df_v4.to_pickle(os.path.join(DATA_DIR, "sub1-results-v4.pkl"))
df_v4.iloc[top50_v4][["供应商ID","品类","供货总量","供货满足率","安全指数_I","排名"]].to_csv(
    os.path.join(RESULT_DIR, "top50-suppliers-v4.csv"), index=False, encoding="utf-8-sig")

# ============================================================
# v5: 降权品类 one-hot（scale=0.5，非 1.0）
# ============================================================
num_cols_v5 = ["供货总量","供货周数","供货满足率","供订CV差","可靠性趋势","品类A","品类B","品类C"]
X_v5_raw = df_v2[num_cols_v5].values.astype(float)
X_v5 = X_v5_raw.copy()

# 手工标准化：前5列 scale=1.0，后3列 scale=0.5
scaler_full = StandardScaler()
scaler_full.fit(X_v5_raw)
mean_full = scaler_full.mean_
std_full = scaler_full.scale_

# 修改后3列 std = 原始std * 2（即 scale=0.5）
std_adj = std_full.copy()
std_adj[-3:] = std_full[-3:] * 2.0
X_v5 = (X_v5_raw - mean_full) / std_adj

I_v5, rank_v5, top50_v5 = run_pca("v5: 降权品类 one-hot (scale=0.5)", X_v5, num_cols_v5)

# 保存
df_v5 = df_v2.copy()
df_v5["安全指数_I"] = I_v5; df_v5["排名"] = np.argsort(np.argsort(-I_v5)) + 1
df_v5.to_pickle(os.path.join(DATA_DIR, "sub1-results-v5.pkl"))
df_v5.iloc[top50_v5][["供应商ID","品类","供货总量","供货满足率","安全指数_I","排名"]].to_csv(
    os.path.join(RESULT_DIR, "top50-suppliers-v5.csv"), index=False, encoding="utf-8-sig")

# ============================================================
# 汇总对比
# ============================================================
# 加载 v3
df_v3 = pd.read_pickle(os.path.join(DATA_DIR, "sub1-results-v3.pkl"))
I_v3 = df_v3["安全指数_I"].values

versions = {
    "v2 (全品类one-hot)":  I_v3,  # rename for clarity in table
    "v3 (去品类C)":        I_v3,
    "v4 (无品类)":         I_v4,
    "v5 (降权0.5)":        I_v5,
}

print(f"\n{'='*60}")
print(f"四版本 Top 50 对比")
print(f"{'='*60}")

# 找出各版本的 Top 50 集合
top50_sets = {}
for name in ["v3","v4","v5"]:
    df = pd.read_pickle(os.path.join(DATA_DIR, f"sub1-results-{name}.pkl"))
    top50_sets[name] = set(ids[df["排名"].values <= 50])

# 重叠
print(f"\nTop 50 重叠:")
for a, b in [("v3","v4"),("v3","v5"),("v4","v5")]:
    overlap = len(top50_sets[a] & top50_sets[b])
    print(f"  {a} ∩ {b}: {overlap}/50")

# 品类分布对比
print(f"\n品类分布:")
print(f"{'版本':>12s}  {'A':>4s}  {'B':>4s}  {'C':>4s}  {'供货%':>6s}  {'SP-006命中':>8s}")
for name in ["v3","v4","v5"]:
    df = pd.read_pickle(os.path.join(DATA_DIR, f"sub1-results-{name}.pkl"))
    top = df[df["排名"] <= 50]
    supply_pct = top["供货总量"].sum() / df["供货总量"].sum() * 100
    hits = sum(1 for sid in expected if sid in set(top["供应商ID"]))
    cats = top["品类"].values
    print(f"{name:>12s}  {(cats=='A').sum():>4d}  {(cats=='B').sum():>4d}  {(cats=='C').sum():>4d}  {supply_pct:>5.1f}%  {hits}/5")

print("\n" + "=" * 60)
print("v4 / v5 完成")
print("=" * 60)

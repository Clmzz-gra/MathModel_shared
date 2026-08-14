"""
阶段 2.1 v3 PCA模型 — 子问题 1
变更: 品类 one-hot 砍掉品类C（去共线性），保留品类A+B，C为参考基准
输入：sub1-preprocessed-v3.pkl
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

# ============================================================
# v3 预处理：从 v2 数据中重构，砍掉品类C
# ============================================================
df_v2 = pd.read_pickle(os.path.join(DATA_DIR, "sub1-preprocessed-v2.pkl"))
num_cols = ["供货总量","供货周数","供货满足率","供订CV差","可靠性趋势","品类A","品类B"]

X = df_v2[num_cols].values.astype(float)
ids = df_v2["供应商ID"].values
categories = df_v2["品类"].values
n, p = X.shape

# 保存 v3
df_v3 = df_v2.copy()
df_v3 = df_v3[["供应商ID","品类"] + num_cols]
df_v3.to_pickle(os.path.join(DATA_DIR, "sub1-preprocessed-v3.pkl"))
np.save(os.path.join(DATA_DIR, "sub1-features-v3.npy"), X)

print(f"v3 特征: {num_cols}")
print(f"品类A={sum(X[:,-2])}, 品类B={sum(X[:,-1])}, 品类C(基准)={n-int(sum(X[:,-2]))-int(sum(X[:,-1]))}")

# ============================================================
# PCA
# ============================================================
scaler = StandardScaler()
Z = scaler.fit_transform(X)

pca = PCA()
Y = pca.fit_transform(Z)

ev   = pca.explained_variance_
evr  = pca.explained_variance_ratio_
cumsum = np.cumsum(evr)
m = (ev >= 1.0).sum()

print(f"\nKaiser PC: m={m}, 累计={cumsum[m-1]*100:.1f}%")

loadings = pca.components_[:m]
print(f"\nPC 载荷:")
for k in range(m):
    top_idx = np.argsort(np.abs(loadings[k]))[::-1][:5]
    items = [f"{num_cols[j]}({loadings[k,j]:+.3f})" for j in top_idx]
    print(f"  PC{k+1} ({evr[k]*100:.1f}%): {' | '.join(items)}")

# ============================================================
# 评分
# ============================================================
w = evr[:m] / evr[:m].sum()
I_tmp = np.sum(Y[:, :m] * w, axis=1)
I = (I_tmp - I_tmp.min()) / (I_tmp.max() - I_tmp.min())

rank = np.argsort(-I)
top50_idx = rank[:50]

print(f"\n{'='*60}")
print(f"Top 50 供应商 (v3: 去品类C one-hot)")
print(f"{'='*60}")
print(f"{'排名':>4s}  {'ID':>6s}  {'品':>2s}  {'I':>8s}  {'供货总量':>10s}  {'满足率':>6s}  {'趋势':>6s}")
print(f"{'-'*52}")
for r, idx in enumerate(top50_idx):
    print(f"{r+1:>4d}  {ids[idx]:>6s}  {categories[idx]:>2s}  {I[idx]:>8.4f}  "
          f"{df_v3['供货总量'].iloc[idx]:>10.0f}  {df_v3['供货满足率'].iloc[idx]:>6.3f}  "
          f"{df_v3['可靠性趋势'].iloc[idx]:>6.3f}")

# ============================================================
# 统计 + 交叉验证
# ============================================================
print(f"\n--- 品类分布 ---")
for cat in ["A","B","C"]:
    cnt = (categories[top50_idx] == cat).sum()
    print(f"  {cat}: {cnt} 家")

print(f"\n--- 供货占比 ---")
pct = df_v3["供货总量"].iloc[top50_idx].sum() / df_v3["供货总量"].sum() * 100
print(f"  Top 50: {pct:.1f}%")

print(f"\n--- SP-006 交叉验证 ---")
expected = ["S229","S361","S140","S108","S151"]
for sid in expected:
    pos = np.where(ids == sid)[0][0]
    r = np.where(rank == pos)[0][0] + 1
    print(f"  {sid}: #{r}, I={I[pos]:.4f}")

# ============================================================
# 保存
# ============================================================
df_out = df_v3.copy()
df_out["安全指数_I"] = I
df_out["排名"] = np.argsort(np.argsort(-I)) + 1
df_out.to_pickle(os.path.join(DATA_DIR, "sub1-results-v3.pkl"))

top50_df = df_out.iloc[top50_idx][["供应商ID","品类","供货总量","供货满足率","供订CV差","可靠性趋势","安全指数_I","排名"]]
top50_df.to_csv(os.path.join(RESULT_DIR, "top50-suppliers-v3.csv"), index=False, encoding="utf-8-sig")

# ============================================================
# 图表: 碎石图
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(range(1, p+1), evr, color="#3498db", alpha=0.7)
ax1.set_xlabel("主成分"); ax1.set_ylabel("方差贡献率", color="#3498db")
ax1.axhline(y=1/p, color="gray", linestyle="--", label=f"1/{p}")
ax1.axvline(x=m+0.5, color="#e74c3c", linestyle="--", label=f"Kaiser m={m}")
ax2 = ax1.twinx()
ax2.plot(range(1, p+1), cumsum*100, "o-", color="#e74c3c", linewidth=1.5, markersize=5)
ax2.set_ylabel("累计%", color="#e74c3c")
for i, c in enumerate(cumsum):
    ax2.annotate(f"{c*100:.0f}%", (i+1, c*100), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=7, color="#e74c3c")
ax1.tick_params(axis="y", labelcolor="#3498db"); ax2.tick_params(axis="y", labelcolor="#e74c3c")
ax1.legend(loc="center right", fontsize=8)
plt.title(f"PCA 碎石图 v3 (m={m}, 累计{cumsum[m-1]*100:.1f}%)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca-scree-v3.pdf"), dpi=150)
plt.close()

# 载荷热力图
fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(loadings, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
ax.set_xticks(range(p))
ax.set_xticklabels([c.replace("供货","").replace("品类","") for c in num_cols], fontsize=8)
ax.set_yticks(range(m))
ax.set_yticklabels([f"PC{k+1} ({evr[k]*100:.0f}%)" for k in range(m)], fontsize=9)
for i in range(m):
    for j in range(p):
        ax.text(j, i, f"{loadings[i,j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(loadings[i,j])>0.5 else "black")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.title("PCA 载荷矩阵 v3")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca-loadings-v3.pdf"), dpi=150)
plt.close()

print(f"\n图表: pca-scree-v3.pdf, pca-loadings-v3.pdf")
print(f"结果: sub1-results-v3.pkl, top50-suppliers-v3.csv")
print("=" * 60)
print("v3 完成")
print("=" * 60)

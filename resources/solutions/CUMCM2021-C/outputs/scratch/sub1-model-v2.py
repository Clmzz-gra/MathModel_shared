"""
阶段 2.1 v2 PCA模型 — 子问题 1
变更：可靠性趋势替代进步因子，统一评分不分类型
输入：sub1-preprocessed-v2.pkl
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================
# 0. 路径
# ============================================================
SCRATCH_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRATCH_DIR, "..", "data")
FIG_DIR  = os.path.join(SCRATCH_DIR, "..", "figures")
RESULT_DIR = os.path.join(SCRATCH_DIR, "..", "..", "solution", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# 1. 加载 v2 数据
# ============================================================
df = pd.read_pickle(os.path.join(DATA_DIR, "sub1-preprocessed-v2.pkl"))
num_cols = ["供货总量","供货周数","供货满足率","供订CV差","可靠性趋势","品类A","品类B","品类C"]
X = df[num_cols].values.astype(float)
ids = df["供应商ID"].values
categories = df["品类"].values
n, p = X.shape
print(f"加载: {n} 供应商 × {p} 特征")
print(f"特征: {num_cols}")

# ============================================================
# 2. 标准化 + PCA
# ============================================================
scaler = StandardScaler()
Z = scaler.fit_transform(X)

pca = PCA()
Y = pca.fit_transform(Z)

ev    = pca.explained_variance_
evr   = pca.explained_variance_ratio_
cumsum = np.cumsum(evr)
m = (ev >= 1.0).sum()

print(f"\nKaiser 保留 PC: m={m}, 累计方差={cumsum[m-1]*100:.1f}%")

# PC 载荷
loadings = pca.components_[:m]
print(f"\nPC 载荷:")
for k in range(m):
    top_idx = np.argsort(np.abs(loadings[k]))[::-1][:4]
    items = [f"{num_cols[j]}({loadings[k,j]:+.3f})" for j in top_idx]
    print(f"  PC{k+1} ({evr[k]*100:.1f}%): {' | '.join(items)}")

# ============================================================
# 3. 综合评分
# ============================================================
w = evr[:m] / evr[:m].sum()
I_tmp = np.sum(Y[:, :m] * w, axis=1)
I_min, I_max = I_tmp.min(), I_tmp.max()
I = (I_tmp - I_min) / (I_max - I_min)

# ============================================================
# 4. 排序 + Top 50
# ============================================================
rank = np.argsort(-I)
top50_idx = rank[:50]

print(f"\n{'='*60}")
print(f"Top 50 供应商")
print(f"{'='*60}")
print(f"{'排名':>4s}  {'ID':>6s}  {'品类':>4s}  {'I':>8s}  {'供货总量':>10s}  {'满足率':>6s}  {'可靠性趋势':>8s}")
print(f"{'-'*58}")
for r, idx in enumerate(top50_idx):
    print(f"{r+1:>4d}  {ids[idx]:>6s}  {categories[idx]:>4s}  {I[idx]:>8.4f}  "
          f"{df['供货总量'].iloc[idx]:>10.0f}  {df['供货满足率'].iloc[idx]:>6.3f}  "
          f"{df['可靠性趋势'].iloc[idx]:>8.3f}")

# ============================================================
# 5. 统计
# ============================================================
print(f"\n--- Top 50 品类 ---")
for cat in ["A","B","C"]:
    cnt = (categories[top50_idx] == cat).sum()
    print(f"  {cat}: {cnt} 家")

print(f"\n--- 供货占比 ---")
top50_total = df["供货总量"].iloc[top50_idx].sum()
total_all = df["供货总量"].sum()
print(f"  Top 50 占总量: {top50_total/total_all*100:.1f}%")

print(f"\n--- 交叉验证 (SP-006) ---")
top50_ids = set(ids[top50_idx])
expected_top5 = ["S229","S361","S140","S108","S151"]
for sid in expected_top5:
    pos = np.where(ids == sid)[0][0]
    print(f"  {sid}: 排名 {np.where(rank==pos)[0][0]+1}, I={I[pos]:.4f}")

# ============================================================
# 6. 保存
# ============================================================
df_out = df.copy()
df_out["安全指数_I"] = I
df_out["排名"] = np.argsort(np.argsort(-I)) + 1
df_out.to_pickle(os.path.join(DATA_DIR, "sub1-results-v2.pkl"))

top50_df = df_out.iloc[top50_idx][["供应商ID","品类","供货总量","供货满足率","供订CV差","可靠性趋势","安全指数_I","排名"]]
top50_df.to_csv(os.path.join(RESULT_DIR, "top50-suppliers-v2.csv"), index=False, encoding="utf-8-sig")
print(f"\n已保存: sub1-results-v2.pkl, top50-suppliers-v2.csv")

# ============================================================
# 7. 图表
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 图1: 碎石图
fig, ax1 = plt.subplots(figsize=(10, 5))
bars = ax1.bar(range(1, p+1), evr, color="#3498db", alpha=0.7)
ax1.set_xlabel("主成分", fontsize=11)
ax1.set_ylabel("方差贡献率", fontsize=11, color="#3498db")
ax1.axhline(y=1/p, color="gray", linestyle="--", linewidth=0.8, label=f"平均线 (1/{p})")
ax1.axvline(x=m+0.5, color="#e74c3c", linestyle="--", linewidth=1, label=f"Kaiser (m={m})")
ax2 = ax1.twinx()
ax2.plot(range(1, p+1), cumsum*100, "o-", color="#e74c3c", linewidth=1.5, markersize=5)
ax2.set_ylabel("累计贡献率 (%)", fontsize=11, color="#e74c3c")
for i, c in enumerate(cumsum):
    ax2.annotate(f"{c*100:.0f}%", (i+1, c*100), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=7, color="#e74c3c")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc="center right", fontsize=8)
ax1.tick_params(axis="y", labelcolor="#3498db")
ax2.tick_params(axis="y", labelcolor="#e74c3c")
plt.title(f"PCA 碎石图 v2 (m={m}, 累计 {cumsum[m-1]*100:.1f}%)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca-scree-v2.pdf"), dpi=150)
plt.close()

# 图2: 载荷热力图
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(loadings, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
ax.set_xticks(range(p))
ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(m))
ax.set_yticklabels([f"PC{k+1} ({evr[k]*100:.1f}%)" for k in range(m)], fontsize=9)
for i in range(m):
    for j in range(p):
        ax.text(j, i, f"{loadings[i,j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(loadings[i,j]) > 0.5 else "black")
plt.colorbar(im, ax=ax, shrink=0.8, label="载荷")
plt.title("PCA 主成分载荷 v2", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca-loadings-v2.pdf"), dpi=150)
plt.close()

# 图3: 安全指数分布 + v1 对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# v2
colors_cat = {"A": "#e74c3c", "B": "#3498db", "C": "#2ecc71"}
for cat in ["A", "B", "C"]:
    mask = categories == cat
    ax1.hist(I[mask], bins=30, alpha=0.6, color=colors_cat[cat], label=f"品类{cat}", edgecolor="white")
ax1.axvline(x=I[top50_idx[-1]], color="black", linestyle="--", linewidth=1)
ax1.set_title("v2: 可靠性趋势", fontsize=11)
ax1.set_xlabel("I")
ax1.legend(fontsize=7)

# v1 (load if exists)
v1_path = os.path.join(DATA_DIR, "sub1-results.pkl")
if os.path.exists(v1_path):
    df_v1 = pd.read_pickle(v1_path)
    I_v1 = df_v1["安全指数_I"].values
    ax2.hist(I_v1, bins=30, alpha=0.7, color="gray", edgecolor="white")
    ax2.set_title("v1: 进步因子 (对比)", fontsize=11)
    ax2.set_xlabel("I")
else:
    ax2.text(0.5, 0.5, "v1 数据未找到", ha="center", va="center", transform=ax2.transAxes)

plt.suptitle("安全指数分布 v1 vs v2", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca-score-compare.pdf"), dpi=150)
plt.close()

# 图4: PC1 vs PC2 散点
fig, ax = plt.subplots(figsize=(9, 7))
not_top50 = np.ones(n, dtype=bool)
not_top50[top50_idx] = False
ax.scatter(Y[not_top50, 0], Y[not_top50, 1], c="lightgray", s=15, alpha=0.4)
sc = ax.scatter(Y[top50_idx, 0], Y[top50_idx, 1], c=I[top50_idx], s=40,
                cmap="RdYlGn", edgecolors="black", linewidth=0.5)
plt.colorbar(sc, ax=ax, label="I", shrink=0.8)
for idx in top50_idx[:5]:
    ax.annotate(ids[idx], (Y[idx,0], Y[idx,1]), textcoords="offset points",
                xytext=(5, 5), fontsize=7, color="darkred")
ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", fontsize=11)
ax.set_title("PC1 vs PC2 (Top 50 高亮) v2", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca-scatter-top50-v2.pdf"), dpi=150)
plt.close()

print(f"\n图表已保存: pca-scree-v2.pdf, pca-loadings-v2.pdf, pca-score-compare.pdf, pca-scatter-top50-v2.pdf")
print("=" * 60)
print("阶段 2.1 v2 PCA 模型完成")
print("=" * 60)

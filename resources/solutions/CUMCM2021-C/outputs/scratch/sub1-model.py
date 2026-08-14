"""
目的：
    阶段 2.1 终版（子问题 1 正式实现）：基于 5 指标 PCA 构建供应商重要性安全指数 I，
    筛选出最重要的 50 家供应商，并输出 4 张正式图（碎石图/载荷图/分布图/散点图）。

原理：
    1. 标准化：Z_ij = (X_ij - μ_j) / σ_j（逐列，与 sklearn StandardScaler 一致，ddof=0）。
    2. 主成分提取：相关矩阵 R = corr(Z)，特征分解 R·v_k = λ_k·v_k，λ 降序排列；
       Kaiser 准则保留 λ_k ≥ 1 的 m 个主成分（本问题 m=2，累计方差 66.1%）。
    3. 主成分得分：Y_ik = Z_i · v_k（即 sklearn PCA.transform，不做 Varimax 旋转）。
    4. 综合评分：以保留主成分的归一化方差贡献为权重 w_k = λ_k / Σ_{j=1..m} λ_j，
       I_tmp^(i) = Σ_k w_k · Y_ik。
    5. 归一化：I = (I_tmp - min) / (max - min) ∈ [0,1]，降序排列取前 50。
    与 math-sub1.tex §3（主成分分析模型）一一对应。

输入数据：
    - sub1-preprocessed-final.pkl (处理后) — 供应商ID, 品类, 供货总量, 供货周数,
      供货满足率, 供订CV差, 可靠性趋势（由 preprocess_final.py 生成）

输出：
    - sub1-results-final.pkl — 402 家安全指数_I + 排名（正式结果）
    - solution/results/top50-suppliers.csv — Top 50 供应商（正式交付）
    - outputs/figures/pca-scree.pdf / pca-loadings.pdf / pca-score-dist.pdf /
      pca-scatter-top50.pdf — 4 张正式图

对应论文章节：
    论文「供应商重要性评价与筛选」（子问题 1，章节号论文写作时定稿）
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
# 1. 加载
# ============================================================
df = pd.read_pickle(os.path.join(DATA_DIR, "sub1-preprocessed-final.pkl"))
num_cols = ["供货总量","供货周数","供货满足率","供订CV差","可靠性趋势"]
X = df[num_cols].values.astype(float)
ids = df["供应商ID"].values
cats = df["品类"].values
n, p = X.shape
print(f"加载: {n}×{p}, 特征: {num_cols}")

# ============================================================
# 2. PCA
# ============================================================
scaler = StandardScaler()
Z = scaler.fit_transform(X)
pca = PCA()
Y = pca.fit_transform(Z)
ev, evr = pca.explained_variance_, pca.explained_variance_ratio_
cumsum = np.cumsum(evr)
m = (ev >= 1.0).sum()

print(f"\nKaiser PC: m={m}, 累计={cumsum[m-1]*100:.1f}%")
loadings = pca.components_[:m]
for k in range(m):
    items = " | ".join([f"{num_cols[j]}({loadings[k,j]:+.3f})" for j in np.argsort(-np.abs(loadings[k]))])
    print(f"  PC{k+1} ({evr[k]*100:.1f}%): {items}")

# ============================================================
# 3. 安全指数
# ============================================================
w = evr[:m] / evr[:m].sum()
I_tmp = np.sum(Y[:, :m] * w, axis=1)
I = (I_tmp - I_tmp.min()) / (I_tmp.max() - I_tmp.min())
rank = np.argsort(-I)
top50 = rank[:50]

print(f"\n{'='*60}")
print(f"Top 50 供应商")
print(f"{'='*60}")
print(f"{'排名':>4s} {'ID':>6s} {'品':>2s} {'I':>8s} {'供货总量':>10s} {'满足率':>6s} {'趋势':>6s}")
print(f"{'-'*52}")
for r, i in enumerate(top50[:50]):
    print(f"{r+1:>4d}  {ids[i]:>6s}  {cats[i]:>2s}  {I[i]:>8.4f}  "
          f"{df['供货总量'].iloc[i]:>10.0f}  {df['供货满足率'].iloc[i]:>6.3f}  "
          f"{df['可靠性趋势'].iloc[i]:>6.3f}")

# ============================================================
# 4. 统计
# ============================================================
print(f"\n--- 品类 ---")
for c in ["A","B","C"]: print(f"  {c}: {(cats[top50]==c).sum()}")
print(f"--- 供货占比: {df['供货总量'].iloc[top50].sum()/df['供货总量'].sum()*100:.1f}% ---")

print(f"--- SP-006 交叉验证 ---")
for sid in ["S229","S361","S140","S108","S151"]:
    p = np.where(ids==sid)[0][0]; r = np.where(rank==p)[0][0]+1
    print(f"  {sid}: #{r}, I={I[p]:.4f}")

# ============================================================
# 5. 保存
# ============================================================
df_out = df.copy()
df_out["安全指数_I"] = I; df_out["排名"] = np.argsort(np.argsort(-I)) + 1
df_out.to_pickle(os.path.join(DATA_DIR, "sub1-results-final.pkl"))
top50_df = df_out.iloc[top50][["供应商ID","品类","供货总量","供货满足率","供订CV差","可靠性趋势","安全指数_I","排名"]]
top50_df.to_csv(os.path.join(RESULT_DIR, "top50-suppliers.csv"), index=False, encoding="utf-8-sig")
print(f"\n已保存: sub1-results-final.pkl, top50-suppliers.csv")

# ============================================================
# 6. 图表
# ============================================================
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei","Microsoft YaHei","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 碎石图
pca_dim = len(evr)
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.bar(range(1,pca_dim+1), evr, color="#3498db", alpha=0.7)
ax1.set_xlabel("主成分"); ax1.set_ylabel("方差贡献率", color="#3498db")
ax1.axhline(y=1/pca_dim, color="gray", ls="--", lw=0.8, label=f"平均 1/{pca_dim}")
ax1.axvline(x=m+0.5, color="#e74c3c", ls="--", lw=1, label=f"Kaiser m={m}")
ax2 = ax1.twinx()
ax2.plot(range(1,pca_dim+1), cumsum*100, "o-", color="#e74c3c", lw=1.5, ms=5)
ax2.set_ylabel("累计贡献率 %", color="#e74c3c")
for i,c in enumerate(cumsum): ax2.annotate(f"{c*100:.0f}%",(i+1,c*100),textcoords="offset points",xytext=(0,8),ha="center",fontsize=7,color="#e74c3c")
ax1.tick_params(axis="y",labelcolor="#3498db"); ax2.tick_params(axis="y",labelcolor="#e74c3c")
ax1.legend(loc="center right",fontsize=8)
plt.title(f"PCA 碎石图 (Kaiser m={m}, 累计 {cumsum[m-1]*100:.1f}%)", fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"pca-scree.pdf"),dpi=150); plt.close()

# 载荷热力图
fig,ax=plt.subplots(figsize=(9,4))
im=ax.imshow(loadings,cmap="RdBu_r",aspect="auto",vmin=-1,vmax=1)
ax.set_xticks(range(pca_dim)); ax.set_xticklabels(num_cols,fontsize=9)
ax.set_yticks(range(m)); ax.set_yticklabels([f"PC{k+1} ({evr[k]*100:.0f}%)" for k in range(m)],fontsize=9)
for i in range(m):
    for j in range(pca_dim):
        ax.text(j,i,f"{loadings[i,j]:.2f}",ha="center",va="center",fontsize=8,
                color="white" if abs(loadings[i,j])>0.5 else "black")
plt.colorbar(im,ax=ax,shrink=0.8,label="载荷")
plt.title("PCA 主成分载荷矩阵",fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"pca-loadings.pdf"),dpi=150); plt.close()

# 安全指数分布
fig,ax=plt.subplots(figsize=(9,5))
colors={"A":"#e74c3c","B":"#3498db","C":"#2ecc71"}
for c in ["A","B","C"]:
    mask=cats==c; ax.hist(I[mask],bins=30,alpha=0.6,color=colors[c],label=f"品类{c}",edgecolor="white")
ax.axvline(x=I[top50[-1]],color="black",ls="--",lw=1,label=f"Top50阈值 I={I[top50[-1]]:.4f}")
ax.set_xlabel("安全指数 I"); ax.set_ylabel("供应商数"); ax.legend(fontsize=8)
ax.set_title("供应商重要性安全指数分布",fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"pca-score-dist.pdf"),dpi=150); plt.close()

# PC1 vs PC2 散点
fig,ax=plt.subplots(figsize=(9,7))
mask=np.ones(n,bool); mask[top50]=False
ax.scatter(Y[mask,0],Y[mask,1],c="lightgray",s=12,alpha=0.3)
sc=ax.scatter(Y[top50,0],Y[top50,1],c=I[top50],s=35,cmap="RdYlGn",edgecolors="black",lw=0.3)
plt.colorbar(sc,ax=ax,label="I",shrink=0.8)
for i in top50[:5]: ax.annotate(ids[i],(Y[i,0],Y[i,1]),textcoords="offset points",xytext=(5,5),fontsize=7,color="darkred")
ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
ax.set_title("PC1 vs PC2 (Top 50 高亮)",fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"pca-scatter-top50.pdf"),dpi=150); plt.close()

print(f"图表已保存: pca-scree.pdf, pca-loadings.pdf, pca-score-dist.pdf, pca-scatter-top50.pdf")
print("=" * 60)
print("终版 PCA 模型完成")
print("=" * 60)

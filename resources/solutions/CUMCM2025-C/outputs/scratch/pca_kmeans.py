"""PCA + K-Means 聚类分析 — 2025 C题数据集（含清洗）"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
for f in fm.fontManager.ttflist:
    if 'SimHei' in f.name or 'Microsoft YaHei' in f.name:
        plt.rcParams['font.sans-serif'] = [f.name]
        plt.rcParams['axes.unicode_minus'] = False
        break
else:
    import os
    for fp in ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
            plt.rcParams['axes.unicode_minus'] = False
            break
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE

# ── 读入原始数据 ──────────────────────────────────────────
df = pd.read_excel(r'e:\MathModel\problems\2025\C题\附件.xlsx')
df.columns = df.columns.str.strip()

# ═══════════════════════════════════════════════════════════
# 数据清洗
# ═══════════════════════════════════════════════════════════
n0 = len(df)
print(f"原始样本数: {n0}")

# 1. 移除任一染色体 Z 值 |Z| > 5 的行（极端离群，主导 PCA 方向）
z_cols = [c for c in df.columns if 'Z值' in c and '染色' in c]
mask_z = (df[z_cols].abs() <= 5).all(axis=1)
n_z_removed = (~mask_z).sum()
df = df[mask_z].copy()
print(f"  移除 |Z|>5 的极端样本: {n_z_removed} 行")

# 2. 移除 X 染色体浓度 < 0 的行（物理不可能）
mask_xc = df['X染色体浓度'] >= 0
n_xc_removed = (~mask_xc).sum()
df = df[mask_xc].copy()
print(f"  移除 X染色体浓度<0 的样本: {n_xc_removed} 行")

# 3. 移除 BMI 录入异常行（A163 最后一笔 BMI=45.7 为错误值）
mask_bmi = ~((df['孕妇代码'] == 'A163') & (df['孕妇BMI'] > 44))
n_bmi_removed = (~mask_bmi).sum()
df = df[mask_bmi].copy()
print(f"  移除 BMI 录入异常: {n_bmi_removed} 行")

print(f"清洗后样本数: {len(df)} (移除 {n0 - len(df)} 行)")

# ── 选择数值型列 ──────────────────────────────────────────
num_cols = [
    '年龄', '身高', '体重', '孕妇BMI', '检测抽血次数',
    '原始读段数', '在参考基因组上比对的比例', '重复读段的比例', '唯一比对的读段数',
    'GC含量',
    '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值',
    'X染色体的Z值', 'Y染色体的Z值',
    'Y染色体浓度', 'X染色体浓度',
    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
    '被过滤掉读段数的比例', '生产次数',
]
df_num = df[num_cols].copy()

# 处理缺失值（中位数填充）
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(df_num)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print(f"样本数: {X_scaled.shape[0]}, 特征数: {X_scaled.shape[1]}")

# ── PCA：选到累积解释率 ≥ 60% ──────────────────────────
pca = PCA()
pca.fit(X_scaled)
cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components = int(np.searchsorted(cumsum, 0.60) + 1)
print(f"达到60%解释率需要的主成分数: {n_components}")
print(f"实际累积解释率: {cumsum[n_components - 1]:.4f}")

for i in range(n_components):
    print(f"  PC{i + 1}: {pca.explained_variance_ratio_[i]:.4f} (cum: {cumsum[i]:.4f})")

X_pca = pca.transform(X_scaled)[:, :n_components]

# ── K-Means（在全部 n_components 维 PCA 空间聚类）───────
def plot_one_projection(ax, X_pca, labels, centers_2d, pc_x, pc_y, evr):
    colors = plt.cm.tab10(np.arange(len(np.unique(labels))))
    for ci in np.unique(labels):
        mask = labels == ci
        ax.scatter(X_pca[mask, pc_x], X_pca[mask, pc_y],
                   c=[colors[ci]], s=8, alpha=0.5, edgecolors='none')
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
               c='black', marker='X', s=80, linewidths=1.2)
    ax.set_xlabel(f'PC{pc_x + 1} ({evr[pc_x]:.1%})', fontsize=8)
    ax.set_ylabel(f'PC{pc_y + 1} ({evr[pc_y]:.1%})', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)


n_plot_pcs = min(5, n_components)

for k in [2, 3]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    counts = np.bincount(labels)
    for i, c in enumerate(counts):
        print(f"  k={k} | 簇{i + 1}: {c} 样本 ({c / len(labels):.1%})")

    fig, axes = plt.subplots(n_plot_pcs - 1, n_plot_pcs - 1, figsize=(14, 12))
    fig.suptitle(
        f'K-Means 聚类投影矩阵 [清洗后] (k={k}, {n_components} PCs, cumR^2={cumsum[n_components - 1]:.1%})',
        fontsize=15, fontweight='bold')

    for i in range(n_plot_pcs - 1):
        for j in range(n_plot_pcs - 1):
            pc_x, pc_y = j, i + 1
            if pc_y <= pc_x:
                axes[i, j].set_visible(False)
            else:
                centers_proj = km.cluster_centers_[:, [pc_x, pc_y]]
                plot_one_projection(axes[i, j], X_pca, labels, centers_proj,
                                    pc_x, pc_y, pca.explained_variance_ratio_)

    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=plt.cm.tab10(ci), markersize=8,
                          label=f'簇{ci + 1} ({counts[ci]})')
               for ci in range(k)]
    fig.legend(handles=handles, loc='lower right', fontsize=9,
               title=f'k={k} 聚类结果', title_fontsize=10)
    plt.tight_layout()
    out_path = rf'e:\MathModel\problems\2025\C题\pca_kmeans_k{k}_pairs_cleaned.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"  k={k} 投影矩阵(清洗后)已保存至: {out_path}")
    plt.close(fig)

# ── t-SNE ──────────────────────────────────────────────
print("\n=== t-SNE 可视化 (清洗后) ===")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
# 先 fit_transform 一次，k=2 和 k=3 共享同一嵌入
X_tsne_all = tsne.fit_transform(X_pca)

fig_tsne, axes_tsne = plt.subplots(1, 2, figsize=(14, 6))

for ax, k in zip(axes_tsne, [2, 3]):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    colors = plt.cm.tab10(np.arange(k))
    for ci in range(k):
        mask = labels == ci
        ax.scatter(X_tsne_all[mask, 0], X_tsne_all[mask, 1],
                   c=[colors[ci]], label=f'簇{ci + 1}', alpha=0.6, s=12,
                   edgecolors='none')
    ax.set_title(f't-SNE + K-Means [清洗后] (k={k})', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 维度 1')
    ax.set_ylabel('t-SNE 维度 2')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

fig_tsne.suptitle(
    f'2025 C题 (清洗后, n={len(df)}) — t-SNE 降维可视化 ({n_components}D PCA → 2D t-SNE, cumR^2={cumsum[n_components - 1]:.1%})',
    fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
tsne_path = r'e:\MathModel\problems\2025\C题\pca_tsne_clusters_cleaned.png'
fig_tsne.savefig(tsne_path, dpi=200, bbox_inches='tight')
print(f"t-SNE 聚类图(清洗后)已保存至: {tsne_path}")
plt.close(fig_tsne)

# ── 碎石图 ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
x = range(1, len(cumsum) + 1)
ax2.bar(x, pca.explained_variance_ratio_, alpha=0.7, label='单个解释率')
ax2.plot(x, cumsum, 'ro-', markersize=4, label='累积解释率')
ax2.axhline(y=0.60, color='green', linestyle='--', label='60% 阈值')
ax2.axvline(x=n_components, color='green', linestyle=':', alpha=0.7)
ax2.set_xlabel('主成分')
ax2.set_ylabel('解释方差比例')
ax2.set_title(f'PCA 方差解释率 (清洗后, n={len(df)})')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()

scree_path = r'e:\MathModel\problems\2025\C题\pca_scree_cleaned.png'
fig2.savefig(scree_path, dpi=200, bbox_inches='tight')
print(f"碎石图(清洗后)已保存至: {scree_path}")

# ── 对比总结 ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("清洗前后对比")
print("=" * 55)
print(f"  样本数: 1082 → {len(df)} (移除 {1082 - len(df)} 行)")
print(f"  Z值>5移除: {n_z_removed} | X浓度<0移除: {n_xc_removed} | BMI异常: {n_bmi_removed}")
print(f"  PCA主成分数: 7 → {n_components}")
print(f"  累积解释率: 0.6123 → {cumsum[n_components - 1]:.4f}")

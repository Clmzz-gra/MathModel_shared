"""阶段 0.4b 出图: t-SNE聚类图 + PCA碎石图"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import json
from pathlib import Path

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = Path(__file__).parent.parent.parent
FIG_DIR = BASE / 'outputs' / 'figures'

with open(BASE / 'outputs' / 'data' / 'profiling_results.json', 'r', encoding='utf-8') as f:
    R = json.load(f)

# ============================================================
# 图1: t-SNE 聚类着色 + 地块类型对照
# ============================================================
tsne = R['tsne']
clusters = np.array(tsne['cluster'])
names = tsne['names']
land_types = tsne['land_types']
k_best = R['clustering']['k_best']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左: 聚类着色
colors_cluster = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0'][:k_best]
for j in range(k_best):
    mask = clusters == j
    axes[0].scatter(np.array(tsne['x'])[mask], np.array(tsne['y'])[mask],
                    c=colors_cluster[j], label=f'簇{j+1} ({mask.sum()})', s=40, alpha=0.8)
axes[0].set_title(f'K-Means++ 聚类 (k={k_best}, Silhouette={R["clustering"]["silhouette"]:.3f})')
axes[0].legend(loc='best')
axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')

# 右: 地块类型着色
type_colors = {'平旱地': '#2196F3', '梯田': '#4CAF50', '山坡地': '#FF9800',
               '水浇地': '#9C27B0', '普通大棚': '#F44336', '智慧大棚': '#00BCD4'}
for lt in sorted(set(land_types)):
    mask = np.array(land_types) == lt
    c = type_colors.get(lt.strip(), '#999999')
    axes[1].scatter(np.array(tsne['x'])[mask], np.array(tsne['y'])[mask],
                    c=c, label=lt.strip(), s=40, alpha=0.8)
axes[1].set_title('地块类型着色对照')
axes[1].legend(loc='best', fontsize=8)
axes[1].set_xlabel('t-SNE 1'); axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
fig.savefig(FIG_DIR / 'cluster-tsne.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('已保存: cluster-tsne.pdf')

# ============================================================
# 图2: PCA 碎石图
# ============================================================
pca = R['pca']
eigvals = np.array(pca['eigvals'])
explained_var = np.array(pca['explained_var'])
cum_var = np.array(pca['cum_var'])
k_kaiser = pca['k_kaiser']

fig, ax1 = plt.subplots(figsize=(8, 5))

bars = ax1.bar(range(1, len(eigvals)+1), explained_var*100, color='#2196F3', alpha=0.7, label='方差解释率')
ax1.set_xlabel('主成分')
ax1.set_ylabel('方差解释率 (%)', color='#2196F3')
ax1.tick_params(axis='y', labelcolor='#2196F3')

# 累积折线
ax2 = ax1.twinx()
ax2.plot(range(1, len(eigvals)+1), cum_var*100, 'o-', color='#FF5722', linewidth=2, label='累积解释率')
ax2.set_ylabel('累积解释率 (%)', color='#FF5722')
ax2.tick_params(axis='y', labelcolor='#FF5722')

# Kaiser 线
ax1.axvline(x=k_kaiser+0.5, color='gray', linestyle='--', alpha=0.5, label=f'Kaiser (λ≥1, k={k_kaiser})')
ax1.axhline(y=1, color='green', linestyle=':', alpha=0.5, label='λ=1')

# 标注
for i, (ev, cv) in enumerate(zip(eigvals, cum_var)):
    ax1.annotate(f'λ={ev:.2f}\n{cv*100:.0f}%', (i+1, explained_var[i]*100),
                 textcoords="offset points", xytext=(0, 12), ha='center', fontsize=8)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='center right')

ax1.set_title('PCA 碎石图 — 作物经济特征')
ax1.set_xticks(range(1, len(eigvals)+1))
plt.tight_layout()
fig.savefig(FIG_DIR / 'pca-scree.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('已保存: pca-scree.pdf')

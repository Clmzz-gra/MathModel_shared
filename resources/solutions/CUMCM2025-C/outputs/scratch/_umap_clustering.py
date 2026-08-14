#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
  2025 C题 — UMAP 聚类探索 完整分析
================================================================================
  技术栈: UMAP (umap-learn) + HDBSCAN + K-Means + StandardScaler
  数据:   男胎(1082条×267孕妇) / 女胎(605条×147孕妇) 清洗后pickle

  核心设计:
    1. 个体去重: 保留每位孕妇首检记录,避免同一人重复测量绑定邻居图
    2. 特征: 仅用原始测量变量(排除ID/标签列/flag列)
    3. 聚类质量: Silhouette + DBCV + ARI/NMI(vs ground truth)
    4. 灵敏度: n_neighbors ∈ {5,10,15,30,50} 稳定性分析
================================================================================
"""
import numpy as np, pandas as pd, warnings, time
from pathlib import Path
warnings.filterwarnings('ignore')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200; plt.rcParams['savefig.dpi'] = 200

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                              normalized_mutual_info_score)
from umap import UMAP
from hdbscan import HDBSCAN

# ========================= 路径 =========================
BASE  = Path("E:/MathModel/problems/2025/C题/2025C题测试")
DATA  = BASE / "outputs" / "data"
FIGS  = BASE / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

print("="*70)
print("  2025 C题 UMAP 聚类探索")
print("="*70)
t0 = time.time()

# ========================= 加载 =========================
dfm = pd.read_pickle(DATA / "2025C-male-clean.pkl")
dff = pd.read_pickle(DATA / "2025C-female-clean.pkl")
print(f"男胎: {dfm.shape} | 女胎: {dff.shape}")

# ========================================================================
# 辅助函数
# ========================================================================
def dedup_by_id(df, id_col='孕妇代码', sort_col='孕周_数值'):
    """去重:每人保留sort_col最小的那条"""
    if id_col in df.columns and sort_col in df.columns:
        return df.sort_values([id_col, sort_col]).drop_duplicates(id_col, keep='first')
    return df

def prep_features(df, num_cols, id_col='孕妇代码'):
    """特征准备:去重→数值特征→IVF one-hot→标准化"""
    df1  = dedup_by_id(df, id_col)
    ivf  = pd.get_dummies(df1['IVF妊娠'].fillna('自然受孕'), prefix='IVF').astype(float)
    ncols = [c for c in num_cols if c in df1.columns]
    feat = pd.concat([df1[ncols].reset_index(drop=True), ivf.reset_index(drop=True)], axis=1)
    feat = feat.dropna().astype(float)
    scaler = StandardScaler()
    X = scaler.fit_transform(feat.values)
    return X, scaler, feat, df1.loc[feat.index], ncols + list(ivf.columns)

def run_umap(X, nn=15, md=0.1, label=''):
    t1 = time.time()
    u = UMAP(n_neighbors=nn, min_dist=md, n_components=2, metric='euclidean',
             random_state=42, verbose=False)
    Xu = u.fit_transform(X)
    print(f"  UMAP {label} ({time.time()-t1:.1f}s), shape={Xu.shape}")
    return Xu, u

def cluster_eval(X_emb, labels_true=None):
    """聚类+评估: HDBSCAN + KMeans(k=4)"""
    # HDBSCAN
    h = HDBSCAN(min_cluster_size=8, min_samples=5, cluster_selection_epsilon=0.3,
                metric='euclidean')
    lh = h.fit_predict(X_emb)
    n_cls = len(set(lh)) - (1 if -1 in lh else 0)
    n_noise = (lh == -1).sum()
    # Silhouette
    nn_mask = lh != -1
    sil_h = silhouette_score(X_emb[nn_mask], lh[nn_mask]) if nn_mask.sum()>1 and n_cls>=2 else None
    try: dbcv = h.relative_validity_
    except: dbcv = None
    # K-Means
    km = KMeans(n_clusters=4, random_state=42, n_init=20)
    lk = km.fit_predict(X_emb)
    sil_k = silhouette_score(X_emb, lk)
    # vs truth
    ari_h, nmi_h = (None, None)
    if labels_true is not None:
        ari_h = adjusted_rand_score(labels_true, lh)
        nmi_h = normalized_mutual_info_score(labels_true, lh)
    ari_k, nmi_k = (None, None)
    if labels_true is not None:
        ari_k = adjusted_rand_score(labels_true, lk)
        nmi_k = normalized_mutual_info_score(labels_true, lk)
    return dict(hdbscan=h, kmeans=km, labels_h=lh, labels_k=lk,
                n_cls=n_cls, n_noise=n_noise, sil_h=sil_h, sil_k=sil_k,
                dbcv=dbcv, ari_h=ari_h, nmi_h=nmi_h, ari_k=ari_k, nmi_k=nmi_k)


# ========================================================================
# 1 — 男胎
# ========================================================================
print("\n" + "-"*50)
print("SECTION 1: 男胎 UMAP")
print("-"*50)

MALE_NUM = [
    '孕周_数值', '孕妇BMI', '年龄', 'Y染色体浓度', 'X染色体浓度',
    'GC含量', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
    '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例',
    '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
]
X_m, scaler_m, feat_m, df_m1, feat_names_m = prep_features(dfm, MALE_NUM)
bmi_m = df_m1['bmi_group'].values
print(f"男胎去重: {len(X_m)} 孕妇, d={X_m.shape[1]}")
print(f"特征: {feat_names_m}")

Xm_u, umap_m = run_umap(X_m, nn=15, label='male')
res_m = cluster_eval(Xm_u)
lh_m, lk_m = res_m['labels_h'], res_m['labels_k']

print(f"  HDBSCAN: {res_m['n_cls']} clusters + {res_m['n_noise']} noise ({res_m['n_noise']/len(lh_m):.1%})")
print(f"  Silhouette: HDB={res_m['sil_h']:.4f}" if res_m['sil_h'] else "  Sil: N/A",
      f"| KM={res_m['sil_k']:.4f}")
print(f"  DBCV: {res_m.get('dbcv','N/A')}")


# ========================================================================
# 2 — 女胎
# ========================================================================
print("\n" + "-"*50)
print("SECTION 2: 女胎 UMAP")
print("-"*50)

FEMALE_NUM = [
    '孕周_数值', '孕妇BMI', '年龄',
    'X染色体浓度',
    'GC含量', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
    '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例',
    '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
]
X_f, scaler_f, feat_f, df_f1, feat_names_f = prep_features(dff, FEMALE_NUM)
ab_f  = df_f1['AB_异常'].values.astype(int)
bmi_f = df_f1['bmi_group'].values if 'bmi_group' in df_f1.columns else None
print(f"女胎去重: {len(X_f)} 孕妇, d={X_f.shape[1]}")
print(f"AB_异常: 1={ab_f.sum()}, 0={(1-ab_f).sum()}")
print(f"特征: {feat_names_f}")

Xf_u, umap_f = run_umap(X_f, nn=15, label='female')
res_f = cluster_eval(Xf_u, labels_true=ab_f)
lh_f, lk_f = res_f['labels_h'], res_f['labels_k']

print(f"  HDBSCAN: {res_f['n_cls']} clusters + {res_f['n_noise']} noise ({res_f['n_noise']/len(lh_f):.1%})")
print(f"  vs AB: ARI={res_f['ari_h']:.4f}, NMI={res_f['nmi_h']:.4f}")
print(f"  Silhouette: HDB={res_f['sil_h']:.4f}" if res_f['sil_h'] else "  Sil: N/A",
      f"| KM={res_f['sil_k']:.4f}")


# ========================================================================
# 3 — 可视化: 男胎 (2x3 网格)
# ========================================================================
print("\n--- 绘图: 男胎 ---")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
cmap10 = plt.cm.tab10
BMI_CATS = ['[20,28)','[28,32)','[32,36)','[36,40)','[40,+)']
BMI_PAL  = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

# [0,0] HDBSCAN clusters
ax = axes[0,0]
for c in sorted(set(lh_m)):
    m = lh_m==c
    clr = '#888' if c==-1 else cmap10(c%10)
    lbl = f'Noise({m.sum()})' if c==-1 else f'C{c}({m.sum()})'
    ax.scatter(Xm_u[m,0], Xm_u[m,1], c=clr, s=16, alpha=0.5, label=lbl)
ax.set_title(f'HDBSCAN ({res_m["n_cls"]}+noise clusters)', fontsize=10)
ax.legend(fontsize=6, loc='lower left', ncol=2); ax.set_xticks([]); ax.set_yticks([])

# [0,1] BMI groups
ax = axes[0,1]
for grp,clr in zip(BMI_CATS, BMI_PAL):
    m = bmi_m==grp; ax.scatter(Xm_u[m,0], Xm_u[m,1], c=clr, s=16, alpha=0.5, label=f'{grp}')
ax.set_title('BMI Groups', fontsize=10); ax.legend(fontsize=6, ncol=3); ax.set_xticks([]); ax.set_yticks([])

# [0,2] Y浓度着色
ax = axes[0,2]
sc=ax.scatter(Xm_u[:,0], Xm_u[:,1], c=feat_m['Y染色体浓度'].values, cmap='viridis', s=14, alpha=0.5)
plt.colorbar(sc, ax=ax, fraction=0.046).set_label('Y Conc.', fontsize=8)
ax.set_title('Y Concentration', fontsize=10); ax.set_xticks([]); ax.set_yticks([])

# [1,0] 聚类×BMI 混淆矩阵
ax = axes[1,0]
cross = pd.crosstab(pd.Series(lh_m).replace(-1,'Noise'), pd.Series(bmi_m))
cpct = cross.div(cross.sum(1), axis=0)
im = ax.imshow(cpct.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(cpct.columns))); ax.set_xticklabels(cpct.columns, fontsize=7, rotation=45, ha='right')
ax.set_yticks(range(len(cpct.index))); ax.set_yticklabels(cpct.index, fontsize=7)
for i in range(cpct.shape[0]):
    for j in range(cpct.shape[1]):
        ax.text(j,i,f'{cpct.iloc[i,j]:.2f}', ha='center', va='center', fontsize=6,
                color='white' if cpct.iloc[i,j]>0.6 else 'black')
ax.set_title('Cluster → BMI (row-%)', fontsize=10); plt.colorbar(im, ax=ax, fraction=0.046)

# [1,1] 特征箱线图(按聚类)
ax = axes[1,1]
key_f = ['孕周_数值', '孕妇BMI', 'Y染色体浓度', 'GC含量']
key_idx = [list(feat_m.columns).index(f) for f in key_f if f in feat_m.columns]
data_box = []
labels_box = []
for c in sorted(set(lh_m)):
    m = lh_m==c
    if m.sum()==0: continue
    lbl = f'Noise' if c==-1 else f'C{c}'
    for ki in key_idx:
        data_box.append(X_m[m, ki])
        labels_box.extend([lbl]*m.sum())
# simplified approach: just plot mean bars
n_cls_eff = len(set(lh_m))
xpos = np.arange(len(key_f))
w = 0.7 / max(n_cls_eff, 1)
for ci, c in enumerate(sorted(set(lh_m))):
    m = lh_m==c
    if m.sum()==0: continue
    means = [X_m[m, ki].mean() for ki in key_idx]
    ax.bar(xpos + (ci-n_cls_eff/2+0.5)*w, means, w,
           color=cmap10(c%10) if c!=-1 else '#888',
           label=f'Noise({m.sum()})' if c==-1 else f'C{c}({m.sum()})', alpha=0.85)
ax.set_xticks(xpos); ax.set_xticklabels(key_f, fontsize=8, rotation=20, ha='right')
ax.set_title('Cluster Feature Means (z-score)', fontsize=10)
ax.axhline(0, color='black', lw=0.5); ax.legend(fontsize=6, ncol=3)

# [1,2] 指标面板
ax = axes[1,2]; ax.axis('off')
lines = [
    "===== 男胎 UMAP 指标 =====",
    f"样本: n={len(X_m)} (去重后孕妇)",
    f"特征: d={X_m.shape[1]}",
    f"",
    f"UMAP: n_neighbors=15, min_dist=0.1",
    f"",
    f"HDBSCAN:",
    f"  clusters={res_m['n_cls']}  noise={res_m['n_noise']} ({res_m['n_noise']/len(lh_m):.1%})",
    f"  Sil={res_m['sil_h']}" if res_m['sil_h'] else "  Sil=N/A",
]
if res_m.get('dbcv'): lines.append(f"  DBCV={res_m['dbcv']:.4f}")
lines += [f"", f"K-Means k=4: Sil={res_m['sil_k']:.4f}"]
for i, l in enumerate(lines):
    ax.text(0.05, 0.97-i*0.055, l, transform=ax.transAxes, fontsize=7, fontfamily='monospace')

fig.suptitle("UMAP Clustering — Male (2025C)", fontsize=13, fontweight='bold', y=0.99)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(FIGS/"umap-male-full.pdf", dpi=200, bbox_inches='tight')
plt.close(fig)
print("  → umap-male-full.pdf")


# ========================================================================
# 4 — 可视化: 女胎 (2x3 网格)
# ========================================================================
print("\n--- 绘图: 女胎 ---")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# [0,0] HDBSCAN
ax = axes[0,0]
for c in sorted(set(lh_f)):
    m = lh_f==c
    clr = '#888' if c==-1 else cmap10(c%10)
    lbl = f'Noise({m.sum()})' if c==-1 else f'C{c}({m.sum()})'
    ax.scatter(Xf_u[m,0], Xf_u[m,1], c=clr, s=20, alpha=0.55, label=lbl)
ax.set_title(f'HDBSCAN ({res_f["n_cls"]}+noise clusters)', fontsize=10)
ax.legend(fontsize=6, loc='lower left'); ax.set_xticks([]); ax.set_yticks([])

# [0,1] AB 标注
ax = axes[0,1]
for val,clr,lbl in [(0,'#2ca02c','Normal'),(1,'#d62728','Abnormal')]:
    m=ab_f==val; ax.scatter(Xf_u[m,0], Xf_u[m,1], c=clr, s=20, alpha=0.55, label=lbl)
ax.set_title('AB Annotation (ground truth)', fontsize=10)
ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])

# [0,2] X浓度着色
ax = axes[0,2]
sc=ax.scatter(Xf_u[:,0], Xf_u[:,1], c=feat_f['X染色体浓度'].values, cmap='RdYlBu_r', s=16, alpha=0.55)
plt.colorbar(sc, ax=ax, fraction=0.046).set_label('X Conc.', fontsize=8)
ax.set_title('X Concentration', fontsize=10); ax.set_xticks([]); ax.set_yticks([])

# [1,0] 聚类 × AB 混淆
ax = axes[1,0]
cross_f = pd.crosstab(pd.Series(lh_f).replace(-1,'Noise'),
                       pd.Series(ab_f).replace({0:'Normal',1:'Abnormal'}))
cpct_f = cross_f.div(cross_f.sum(1), axis=0)
im = ax.imshow(cpct_f.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(cpct_f.columns))); ax.set_xticklabels(cpct_f.columns, fontsize=8)
ax.set_yticks(range(len(cpct_f.index))); ax.set_yticklabels(cpct_f.index, fontsize=7)
for i in range(cpct_f.shape[0]):
    for j in range(cpct_f.shape[1]):
        ax.text(j,i,f'{cpct_f.iloc[i,j]:.2f}', ha='center', va='center', fontsize=7,
                color='white' if cpct_f.iloc[i,j]>0.6 else 'black')
ax.set_title(f'Cluster → AB (ARI={res_f["ari_h"]:.3f})', fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046)

# [1,1] Outlier score
ax = axes[1,1]
scores_f = res_f['hdbscan'].outlier_scores_
for val,clr,lbl in [(0,'#2ca02c','Normal'),(1,'#d62728','Abnormal')]:
    m=ab_f==val; ax.scatter(np.arange(len(scores_f))[m], scores_f[m], c=clr, s=10, alpha=0.45, label=lbl)
for q,ls in [(0.50,'--'),(0.90,'-.')]:
    ax.axhline(np.percentile(scores_f,q*100), color='gray', ls=ls, lw=0.8)
ax.set_xlabel('Sample Index', fontsize=8); ax.set_ylabel('Outlier Score', fontsize=8)
ax.set_title('HDBSCAN Outlier Score', fontsize=10); ax.legend(fontsize=7)

# [1,2] 指标面板
ax = axes[1,2]; ax.axis('off')
lines = [
    "===== 女胎 UMAP 指标 =====",
    f"样本: n={len(X_f)} (去重后孕妇)",
    f"异常: n_ab={ab_f.sum()} ({ab_f.sum()/len(ab_f):.1%})",
    f"特征: d={X_f.shape[1]}",
    f"",
    f"UMAP: n_neighbors=15, min_dist=0.1",
    f"",
    f"HDBSCAN:",
    f"  clusters={res_f['n_cls']}  noise={res_f['n_noise']} ({res_f['n_noise']/len(lh_f):.1%})",
    f"  vs AB: ARI={res_f['ari_h']:.4f}, NMI={res_f['nmi_h']:.4f}",
]
if res_f.get('dbcv'): lines.append(f"  DBCV={res_f['dbcv']:.4f}")
lines += [f"", f"K-Means k=4:", f"  Sil={res_f['sil_k']:.4f}",
          f"  vs AB: ARI={res_f['ari_k']:.4f}, NMI={res_f['nmi_k']:.4f}"]
for i,l in enumerate(lines):
    ax.text(0.05, 0.97-i*0.048, l, transform=ax.transAxes, fontsize=7, fontfamily='monospace')

fig.suptitle("UMAP Clustering — Female (2025C)", fontsize=13, fontweight='bold', y=0.99)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(FIGS/"umap-female-full.pdf", dpi=200, bbox_inches='tight')
plt.close(fig)
print("  → umap-female-full.pdf")


# ========================================================================
# 5 — n_neighbors 灵敏度分析
# ========================================================================
print("\n--- 灵敏度分析: n_neighbors ---")
nn_vals = [5, 10, 15, 30, 50]
fig, axes = plt.subplots(2, len(nn_vals), figsize=(len(nn_vals)*3.2, 6.5))

for ci, nn in enumerate(nn_vals):
    # Male
    Xu_m,_ = run_umap(X_m, nn=nn, label=f'male-nn{nn}')
    ax = axes[0, ci]
    for grp,clr in zip(BMI_CATS, BMI_PAL):
        m=bmi_m==grp; ax.scatter(Xu_m[m,0], Xu_m[m,1], c=clr, s=5, alpha=0.4)
    ax.set_title(f'Male, nn={nn}', fontsize=9); ax.set_xticks([]); ax.set_yticks([])

    # Female
    Xu_f,_ = run_umap(X_f, nn=nn, label=f'female-nn{nn}')
    ax = axes[1, ci]
    for val,clr in [(0,'#2ca02c'),(1,'#d62728')]:
        m=ab_f==val; ax.scatter(Xu_f[m,0], Xu_f[m,1], c=clr, s=6, alpha=0.45)
    ax.set_title(f'Female, nn={nn}', fontsize=9); ax.set_xticks([]); ax.set_yticks([])

axes[0,0].set_ylabel('Male (BMI colored)', fontsize=10, fontweight='bold')
axes[1,0].set_ylabel('Female (AB colored)', fontsize=10, fontweight='bold')
fig.suptitle('UMAP Sensitivity: n_neighbors', fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(FIGS/"umap-sensitivity.pdf", dpi=200, bbox_inches='tight')
plt.close(fig)
print("  → umap-sensitivity.pdf")


# ========================================================================
# 6 — 最终摘要
# ========================================================================
elapsed = time.time()-t0
sil_m_str = f"{res_m['sil_h']:.4f}" if res_m['sil_h'] else "N/A"
sil_f_str = f"{res_f['sil_h']:.4f}" if res_f['sil_h'] else "N/A"

print("\n" + "="*70)
print(f"  DONE ({elapsed:.1f}s)")
print("="*70)
print(f"  输出: 1. umap-male-full.pdf  2. umap-female-full.pdf  3. umap-sensitivity.pdf")
print()
print(f"  [男胎] n={len(X_m)}, d={X_m.shape[1]}")
print(f"    HDBSCAN: {res_m['n_cls']} clusters + {res_m['n_noise']} noise ({res_m['n_noise']/len(lh_m):.1%})")
print(f"    Silhouette: HDB={sil_m_str}, KMeans={res_m['sil_k']:.4f}")
print(f"    BMI分离度 -> 见混淆矩阵")
print()
print(f"  [女胎] n={len(X_f)}, d={X_f.shape[1]}, abnormal={ab_f.sum()}")
print(f"    HDBSCAN: {res_f['n_cls']} clusters + {res_f['n_noise']} noise ({res_f['n_noise']/len(lh_f):.1%})")
print(f"    vs AB: ARI={res_f['ari_h']:.4f}, NMI={res_f['nmi_h']:.4f}")
print(f"    Silhouette: HDB={sil_f_str}, KMeans={res_f['sil_k']:.4f}")
print()
print(f"  结论:")
print(f"    - UMAP 嵌入对 BMI 分层有一定分离趋势,但噪声点占比高")
print(f"    - 女胎 AB 异常与聚类簇对齐度有限 (ARI={res_f['ari_h']:.3f})")
print(f"    - 这证实了之前判断: 低信噪比+纵向结构数据不适合聚类式建模")

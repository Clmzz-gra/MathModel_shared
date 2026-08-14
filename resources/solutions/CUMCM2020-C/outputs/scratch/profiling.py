"""
阶段 0.4b — 数据画像：特征工程 + 5 条画像路径
路径 A(分群) + B(降维) 纯 numpy，路径 C/D/E 用 sklearn
"""
import pandas as pd
import numpy as np
import os, sys, json, warnings
warnings.filterwarnings('ignore')

out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(out_dir)
data_dir = os.path.join(out_dir, 'data')
fig_dir = os.path.join(out_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

REF_DATE = pd.Timestamp('2020-02-21')  # 数据截止日期

# ══════════════════════════════════════════════
# 1. 加载清洗数据 + 特征工程
# ══════════════════════════════════════════════
print("=" * 60)
print("特征工程：发票级 → 企业级")
print("=" * 60)

df1_info = pd.read_parquet(os.path.join(data_dir, 'f1_企业信息_clean.parquet'))
df1_in   = pd.read_parquet(os.path.join(data_dir, 'f1_进项_clean.parquet'))
df1_out  = pd.read_parquet(os.path.join(data_dir, 'f1_销项_clean.parquet'))
df2_info = pd.read_parquet(os.path.join(data_dir, 'f2_企业信息_clean.parquet'))
df2_in   = pd.read_parquet(os.path.join(data_dir, 'f2_进项_clean.parquet'))
df2_out  = pd.read_parquet(os.path.join(data_dir, 'f2_销项_clean.parquet'))

def build_features(info_df, in_df, out_df, label):
    """从发票数据构建企业级特征矩阵"""
    eids = info_df['企业代号'].values
    features = []
    
    for eid in eids:
        inv_in  = in_df[in_df['企业代号'] == eid]
        inv_out = out_df[out_df['企业代号'] == eid]
        
        # ── 规模维度 ──
        total_in   = inv_in['金额'].sum()
        total_out  = inv_out['金额'].sum()
        gross_profit = total_out - total_in
        n_in       = len(inv_in)
        n_out      = len(inv_out)
        
        # ── 月度活跃 ──
        in_dates   = pd.DatetimeIndex(inv_in['开票日期'])
        out_dates  = pd.DatetimeIndex(inv_out['开票日期'])
        all_dates  = in_dates.append(out_dates)
        active_months = all_dates.to_period('M').nunique() if len(all_dates) > 0 else 0
        
        # ── 月度 CV ──
        if len(inv_in) > 0:
            monthly_in  = inv_in.groupby(pd.DatetimeIndex(inv_in['开票日期']).to_period('M'))['金额'].sum()
            cv_in = monthly_in.std() / monthly_in.mean() if monthly_in.mean() > 0 else 0
        else:
            cv_in = 0
        if len(inv_out) > 0:
            monthly_out = inv_out.groupby(pd.DatetimeIndex(inv_out['开票日期']).to_period('M'))['金额'].sum()
            cv_out = monthly_out.std() / monthly_out.mean() if monthly_out.mean() > 0 else 0
        else:
            cv_out = 0
        
        # ── 网络维度 ──
        n_suppliers  = inv_in['销方单位代号'].nunique()
        n_customers  = inv_out['购方单位代号'].nunique()
        
        # ── 均价 ──
        avg_in_price  = inv_in['金额'].mean() if n_in > 0 else 0
        avg_out_price = inv_out['金额'].mean() if n_out > 0 else 0
        
        # ── R/F/M ──
        last_date = all_dates.max() if len(all_dates) > 0 else REF_DATE
        recency = (REF_DATE - last_date).days
        frequency = n_in + n_out
        monetary = total_in + total_out
        
        # ── 利润率 ──
        profit_margin = gross_profit / total_out if total_out > 0 else 0
        
        features.append([
            total_in, total_out, gross_profit, n_in, n_out,
            active_months, cv_in, cv_out,
            n_suppliers, n_customers,
            avg_in_price, avg_out_price,
            recency, frequency, monetary, profit_margin
        ])
    
    feats = np.array(features)
    col_names = [
        'total_in', 'total_out', 'gross_profit', 'n_in', 'n_out',
        'active_months', 'cv_in', 'cv_out',
        'n_suppliers', 'n_customers',
        'avg_in_price', 'avg_out_price',
        'recency_days', 'frequency', 'monetary', 'profit_margin'
    ]
    df_feat = pd.DataFrame(feats, columns=col_names, index=eids)
    
    # 过滤全零行（无任何有效发票的企业）
    mask = (df_feat[['total_in', 'total_out']].sum(axis=1) > 0)
    print(f"  {label}: {len(eids)} 企业 → {mask.sum()} 有交易企业")
    
    return df_feat[mask]

# 附件1: 排除D级企业(已在info中标记)
df1_info_valid = df1_info[df1_info['排除原因'] == '']
df1_feat = build_features(df1_info_valid, df1_in, df1_out, '附件1(非D)')
df2_feat = build_features(df2_info, df2_in, df2_out, '附件2')

# 合并用于无监督画像
df_all = pd.concat([df1_feat, df2_feat])
all_eids = df_all.index.tolist()
n_total = len(df_all)
print(f"\n总计: {n_total} 家企业用于无监督画像")

# 标签信息(仅附件1有)
df1_info_v = df1_info_valid[df1_info_valid['企业代号'].isin(df_all.index)]
default_label = df1_info_v.set_index('企业代号')['是否违约'].map({'是': 1, '否': 0})
credit_label = df1_info_v.set_index('企业代号')['信誉评级']

# 标准化
from sklearn.preprocessing import StandardScaler
feat_cols = df_all.columns.tolist()
X_raw = df_all[feat_cols].values
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
# 处理零方差列导致的 NaN
nan_mask = np.isnan(X).any(axis=0)
if nan_mask.any():
    print(f"  ⚠ {nan_mask.sum()} 列含 NaN (零方差列), 填充为0")
    X[:, nan_mask] = np.nan_to_num(X[:, nan_mask], nan=0.0)
X = np.nan_to_num(X, nan=0.0)  # 全局安全
print(f"特征矩阵: {X.shape}, 含NaN={np.isnan(X).any()}")

# ══════════════════════════════════════════════
# 路径 A：分群画像 (纯 numpy K-Means++)
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("路径 A：分群画像 (K-Means++, k=2,3,4)")
print("=" * 60)

def kmeans_plus_plus(X, k, max_iter=300, tol=1e-4, seed=42):
    """纯 numpy K-Means++"""
    rng = np.random.RandomState(seed)
    n, d = X.shape
    # 初始化
    centers = np.zeros((k, d))
    centers[0] = X[rng.randint(n)]
    for j in range(1, k):
        dist_sq = np.min([np.sum((X - c)**2, axis=1) for c in centers[:j]], axis=0)
        total = dist_sq.sum()
        if total == 0 or np.isnan(total):
            probs = np.ones(n) / n  # 回退均匀分布
        else:
            probs = dist_sq / total
        centers[j] = X[np.random.choice(n, p=probs)]
    
    for it in range(max_iter):
        # E-step: assign clusters
        dist = np.array([np.sum((X - c)**2, axis=1) for c in centers])  # (k, n)
        labels = np.argmin(dist, axis=0)
        # M-step: update centers
        new_centers = np.array([X[labels == i].mean(axis=0) if (labels == i).sum() > 0 
                                 else X[rng.randint(n)] for i in range(k)])
        shift = np.sum((new_centers - centers)**2)
        centers = new_centers
        if shift < tol:
            break
    return labels, centers, it + 1

def silhouette_score(X, labels):
    """纯 numpy Silhouette"""
    n = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        return 0.0
    scores = np.zeros(n)
    for i in range(n):
        same_cluster = labels == labels[i]
        other_clusters = ~same_cluster
        if same_cluster.sum() <= 1:
            scores[i] = 0  # 孤立样本
            continue
        a_i = np.mean(np.sum((X[i] - X[same_cluster])**2, axis=1))
        b_vals = [np.mean(np.sum((X[i] - X[labels == l])**2, axis=1)) 
                   for l in unique_labels if l != labels[i]]
        b_i = np.min(b_vals) if b_vals else a_i
        scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
    return float(np.nan_to_num(np.mean(scores), nan=0.0))

cluster_results = {}
for k in [2, 3, 4]:
    labels, centers, iters = kmeans_plus_plus(X, k)
    sil = silhouette_score(X, labels)
    cluster_results[k] = {'labels': labels, 'centers': centers, 'silhouette': sil, 'iters': iters}
    counts = np.bincount(labels)
    print(f"\nk={k}: silhouette={sil:.4f}, 迭代={iters}次")
    for i, cnt in enumerate(counts):
        pct = cnt / n_total * 100
        print(f"  簇{i}: {cnt} 家 ({pct:.1f}%)")
        # 核心特征均值(逆标准化)
        cluster_mean = X_raw[labels == i].mean(axis=0)
        top3 = np.argsort(np.abs(cluster_mean - X_raw.mean(axis=0)))[-3:][::-1]
        top_info = ", ".join([f"{feat_cols[j]}={cluster_mean[j]:.0f}" for j in top3])
        print(f"    突出特征: {top_info}")

# 选最优K (最高silhouette)
best_k = max(cluster_results, key=lambda k: cluster_results[k]['silhouette'])
best_labels = cluster_results[best_k]['labels']
print(f"\n最优 K = {best_k} (silhouette={cluster_results[best_k]['silhouette']:.4f})")

# 数据质量标志：检查小簇
for i, cnt in enumerate(np.bincount(best_labels)):
    pct = cnt / n_total * 100
    if pct < 2:
        print(f"  ⚠ 簇{i}仅{cnt}家({pct:.1f}%) — 潜在排除候选")

# ── t-SNE 可视化 ──
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sample_n = min(500, n_total)
rng_tsne = np.random.RandomState(42)
indices = rng_tsne.choice(n_total, sample_n, replace=False)
X_sample = X[indices]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_sample)

# 图1: 聚类着色
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i in range(best_k):
    mask = best_labels[indices] == i
    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=colors[i], 
                     label=f'簇{i} ({np.bincount(best_labels)[i]}家)', alpha=0.7, s=30)
axes[0].set_title(f'K-Means k={best_k} (t-SNE)', fontsize=13)
axes[0].legend(fontsize=9)

# 图2: 违约状态着色(附件1企业)
is_f1 = np.array([eid in default_label.index for eid in np.array(all_eids)[indices]])
default_colors = np.array(['gray'] * sample_n, dtype=object)
f1_default = np.array([default_label.get(e, -1) for e in np.array(all_eids)[indices]])
default_colors[(is_f1) & (f1_default == 0)] = '#2ca02c'  # 未违约
default_colors[(is_f1) & (f1_default == 1)] = '#d62728'  # 违约
axes[1].scatter(X_tsne[is_f1, 0], X_tsne[is_f1, 1], c=default_colors[is_f1], alpha=0.7, s=30)
# 附件2灰色
f2_mask = np.array([eid not in default_label.index for eid in np.array(all_eids)[indices]])
axes[1].scatter(X_tsne[f2_mask, 0], X_tsne[f2_mask, 1], c='lightgray', alpha=0.3, s=15, label='附件2(无标签)')
axes[1].scatter([], [], c='#2ca02c', label='未违约')
axes[1].scatter([], [], c='#d62728', label='违约')
axes[1].set_title('违约标签对照 (t-SNE)', fontsize=13)
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'cluster-tsne.pdf'), dpi=200, bbox_inches='tight')
plt.close()
print("  t-SNE 图已保存: outputs/figures/cluster-tsne.pdf")

# ══════════════════════════════════════════════
# 路径 B：降维画像 (纯 numpy PCA)
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("路径 B：降维画像 (PCA)")
print("=" * 60)

def pca_numpy(X):
    """纯 numpy PCA"""
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs

eigvals, eigvecs = pca_numpy(X)
var_ratio = eigvals / eigvals.sum()
cum_ratio = np.cumsum(var_ratio)
print(f"特征值≥1: {(eigvals >= 1).sum()} 个 PC")
print(f"前5 PC 方差解释: {[f'{r:.1%}' for r in var_ratio[:5]]}")
print(f"cumR²≥60%: 需要 {(cum_ratio >= 0.6).argmax() + 1} 个 PC")

# 载荷矩阵（前5 PC）
loadings = eigvecs[:, :5] * np.sqrt(eigvals[:5])
print("\n前5 PC 载荷 (|载荷|>0.3 标注*):")
for i in range(5):
    ordering = np.argsort(np.abs(loadings[:, i]))[::-1]
    top_feats = [f"{feat_cols[j]}({loadings[j,i]:+.3f})" for j in ordering[:5] if abs(loadings[j,i]) > 0.3]
    print(f"  PC{i+1} ({var_ratio[i]:.1%}): {' | '.join(top_feats)}")

# 碎石图
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, 17), var_ratio, alpha=0.7, color='steelblue', label='单独方差比')
ax.plot(range(1, 17), cum_ratio, 'ro-', markersize=5, label='累积')
ax.axhline(y=1/len(feat_cols), color='gray', linestyle='--', alpha=0.7, label='Kaiser 线 (均值)')
ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='60% 累积')
for i, r in enumerate(var_ratio[:5]):
    ax.text(i+1, r + 0.01, f'{r:.1%}', ha='center', fontsize=8)
ax.set_xlabel('主成分', fontsize=12)
ax.set_ylabel('方差解释率', fontsize=12)
ax.set_title('PCA 碎石图', fontsize=14)
ax.legend(fontsize=9)
ax.set_xticks(range(1, 17))
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'pca-scree.pdf'), dpi=200, bbox_inches='tight')
plt.close()
print("  碎石图已保存: outputs/figures/pca-scree.pdf")

# 降维结果解释
pc1_load = loadings[:, 0]
pc2_load = loadings[:, 1]
pc1_top = feat_cols[np.argmax(np.abs(pc1_load))]
pc2_top = feat_cols[np.argmax(np.abs(pc2_load))]
print(f"\n  PC1 主要载荷: {pc1_top} ({pc1_load[np.argmax(np.abs(pc1_load))]:+.3f})")
print(f"  PC2 主要载荷: {pc2_top} ({pc2_load[np.argmax(np.abs(pc2_load))]:+.3f})")

# ══════════════════════════════════════════════
# 路径 C：异常检测 (Isolation Forest)
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("路径 C：异常检测 (Isolation Forest + LOF)")
print("=" * 60)

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

iso = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
iso_labels = iso.fit_predict(X)
iso_anomaly = iso_labels == -1
n_iso = iso_anomaly.sum()

print(f"Isolation Forest: {n_iso} 个异常 ({n_iso/n_total*100:.1f}%)")

# LOF (n<500)
if n_total < 500:
    lof = LocalOutlierFactor(n_neighbors=20, contamination='auto', novelty=False)
    lof_factors = -lof.fit_predict(X)  # LOF returns inlier=1, outlier=-1
    lof_anomaly = lof_factors == -1
    n_lof = lof_anomaly.sum()
    both = (iso_anomaly & lof_anomaly).sum()
    print(f"LOF: {n_lof} 个异常 ({n_lof/n_total*100:.1f}%)")
    print(f"两方法一致: {both} 个高置信异常")

# 异常画像
anomaly_indices = np.where(iso_anomaly)[0]
print(f"\n高置信异常企业 ({n_iso} 家) — 偏离>2σ的特征:")
for idx in anomaly_indices[:5]:  # 前5个
    eid = all_eids[idx]
    deviations = np.abs(X[idx])
    top_dev = np.argsort(deviations)[-3:][::-1]
    dev_info = ", ".join([f"{feat_cols[j]}={X_raw[idx,j]:.0f} (z={X[idx,j]:+.1f})" for j in top_dev])
    in_f1 = "附件1" if eid in default_label.index else "附件2"
    print(f"  {eid} ({in_f1}): {dev_info}")

# ══════════════════════════════════════════════
# 路径 D：关联发现 (交易对手共现)
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("路径 D：关联发现 (交易对手共现模式)")
print("=" * 60)

# 将每个企业的交易对手作为"购物篮"
def get_counterparty_baskets(in_df, out_df):
    """返回每个企业的交易对手集合列表"""
    baskets = []
    for eid in in_df['企业代号'].unique():
        suppliers = set(in_df[in_df['企业代号'] == eid]['销方单位代号'])
        customers = set(out_df[out_df['企业代号'] == eid]['购方单位代号'])
        all_partners = suppliers | customers
        if len(all_partners) > 0:
            baskets.append(list(all_partners))
    return baskets

baskets1 = get_counterparty_baskets(df1_in, df1_out)
baskets2 = get_counterparty_baskets(df2_in, df2_out)
all_baskets = baskets1 + baskets2
print(f"事务数: {len(all_baskets)} 家企业, 去重交易对手: {len(set([x for b in all_baskets for x in b]))}")

# 简单频繁项集：统计交易对手出现频率 + 共现矩阵
from collections import Counter
cp_counter = Counter()
for b in all_baskets:
    cp_counter.update(set(b))
top_cp = cp_counter.most_common(20)
print(f"\nTop 10 频繁交易对手:")
for cp, cnt in top_cp[:10]:
    print(f"  {cp}: {cnt} 家企业 ({cnt/len(all_baskets)*100:.1f}%)")

# 共现对（简化版关联规则）
pair_counter = Counter()
for b in all_baskets:
    s = set(b)
    # 统计频繁对手间的共现
    top_set = {cp for cp, _ in top_cp[:15]}
    common = list(s & top_set)
    for i in range(len(common)):
        for j in range(i+1, len(common)):
            pair = tuple(sorted([common[i], common[j]]))
            pair_counter[pair] += 1

print(f"\nTop 5 频繁共现交易对手对:")
for pair, cnt in pair_counter.most_common(5):
    support = cnt / len(all_baskets)
    print(f"  {pair[0]} ↔ {pair[1]}: {cnt} 家 (support={support:.2%})")

# ══════════════════════════════════════════════
# 路径 E：规则画像 (RFM 打分)
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("路径 E：规则画像 (RFM 5分制)")
print("=" * 60)

# R=recency(越小越好), F=frequency(越大越好), M=monetary(越大越好)
r_col = df_all['recency_days'].values
f_col = df_all['frequency'].values
m_col = df_all['monetary'].values

def score_5bin(vals, reverse=False):
    """5分制打分：前20%=5分，后20%=1分"""
    pcts = np.percentile(vals, [20, 40, 60, 80])
    if reverse:
        pcts = pcts[::-1]
    scores = np.ones(len(vals))
    for thresh in pcts:
        if reverse:
            scores += (vals <= thresh).astype(int)
        else:
            scores += (vals >= thresh).astype(int)
    return np.clip(scores, 1, 5)

r_score = score_5bin(r_col, reverse=True)
f_score = score_5bin(f_col)
m_score = score_5bin(m_col)

# 综合得分
rfm_total = r_score + f_score + m_score
print(f"R 分: {np.bincount(r_score.astype(int))[1:]}")
print(f"F 分: {np.bincount(f_score.astype(int))[1:]}")
print(f"M 分: {np.bincount(m_score.astype(int))[1:]}")

# 分档
p33, p66 = np.percentile(rfm_total, [33, 67])
tiers = np.zeros(n_total, dtype=object)
tiers[rfm_total >= p66] = '高价值'
tiers[(rfm_total >= p33) & (rfm_total < p66)] = '中等'
tiers[rfm_total < p33] = '低价值'

print(f"\nRFM 分档:")
for tier in ['高价值', '中等', '低价值']:
    mask = tiers == tier
    cnt = mask.sum()
    if cnt > 0:
        avg_r = r_col[mask].mean()
        avg_f = f_col[mask].mean()
        avg_m = m_col[mask].mean()
        # 附件1违约率
        eids_tier = np.array(all_eids)[mask]
        in_f1 = [e for e in eids_tier if e in default_label.index]
        if in_f1:
            def_rate = default_label[in_f1].mean()
        else:
            def_rate = 0
        print(f"  {tier}: {cnt} 家 ({cnt/n_total*100:.1f}%), "
              f"R均值={avg_r:.0f}天, F均值={avg_f:.0f}, M均值={avg_m:.0f}, "
              f"附件1违约率={def_rate:.1%}")

# ══════════════════════════════════════════════
# 汇总输出
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("画像完成。各路径产出:")
print("=" * 60)
print(f"  A: K-Means k={best_k}, silhouette={cluster_results[best_k]['silhouette']:.4f}")
print(f"  B: PCA, {(eigvals >= 1).sum()} 个PC特征值≥1, 前2 PC累计={cum_ratio[1]:.1%}")
print(f"  C: Isolation Forest 异常 {n_iso} 家 ({n_iso/n_total*100:.1f}%)")
print(f"  D: 交易对手共现, {len(top_cp)} 个频繁对手, {len(pair_counter)} 对共现")
print(f"  E: RFM, 高价值{tiers[tiers=='高价值'].sum()}家, 中等{tiers[tiers=='中等'].sum()}家, 低价值{tiers[tiers=='低价值'].sum()}家")

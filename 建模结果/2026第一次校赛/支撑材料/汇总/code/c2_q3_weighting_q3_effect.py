"""
各赋权法在 Q3 中的实际效果对比
- 逐题 Top-5 / Bottom-5 排名
- 聚类分层变化（评委在不同方法下所属分层是否一致）
- 排名变动分布（相比熵权，每位评委排名上升/下降了多少）
- 方法间分层一致性矩阵
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings, os
warnings.filterwarnings('ignore')

PROBLEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROBLEM_DIR, 'data')
EXPLORE_DIR = os.path.join(PROBLEM_DIR, '..', 'solution', 'explore-results', 'weighting')

# 读取已计算好的全部方法得分 + 原始维度数据
scores_df = pd.read_csv(os.path.join(EXPLORE_DIR, 'all-scores-all-methods.csv'), encoding='utf-8-sig')
orig_df = pd.read_csv(os.path.join(PROBLEM_DIR, '..', 'solution', 'artifacts', 'tables', 'q3-judge-scores.csv'), encoding='utf-8-sig')
FEATURE_COLS = ['信度', '效度', '公平性', '区分力']
# 合并维度列
full = scores_df.merge(orig_df[['题目', '评委ID'] + FEATURE_COLS], on=['题目', '评委ID'], how='left')
METHOD_NAMES = ['熵权', 'CRITIC', 'CV', '标准差', 'PCA', 'GRA']
TOPICS = ['A', 'B', 'C', 'D', 'E']

# =====================================================================
# 1. 逐题各方法 Top-5 与 Bottom-5 排名对比
# =====================================================================
print('=' * 80)
print('  Q3 实际排名效果：各方法 Top-5 与 Bottom-5')
print('=' * 80)

for topic in TOPICS:
    sub = full[full['题目'] == topic].copy()
    K = len(sub)
    
    print(f'\n{"─"*70}')
    print(f'  题目 {topic} ({K}位评委)')
    print(f'{"─"*70}')
    
    # Top-5
    print(f'\n  【Top-5 排名对比】')
    header = f'  {"名次":<4} {"熵权":<12} {"CRITIC":<12} {"CV":<12} {"标准差":<12} {"PCA":<12} {"GRA":<12}'
    print(header)
    print(f'  {"─"*68}')
    for rank in range(1, 6):
        row = f'  #{rank:<3}'
        for m in METHOD_NAMES:
            rank_col = f'排名_{m}'
            score_col = f'TOPSIS_{m}'
            # 找该排名对应的评委
            match = sub[sub[rank_col] == rank]
            if len(match) > 0:
                jid = match.iloc[0]['评委ID']
                score = match.iloc[0][score_col]
                row += f' {jid}({score:.3f}) '
            else:
                row += f' {"—":<12}'
        print(row)
    
    # Bottom-5
    print(f'\n  【Bottom-5 排名对比】')
    print(header)
    print(f'  {"─"*68}')
    for rank in range(K, K-5, -1):
        row = f'  #{rank:<3}'
        for m in METHOD_NAMES:
            rank_col = f'排名_{m}'
            score_col = f'TOPSIS_{m}'
            match = sub[sub[rank_col] == rank]
            if len(match) > 0:
                jid = match.iloc[0]['评委ID']
                score = match.iloc[0][score_col]
                row += f' {jid}({score:.3f}) '
            else:
                row += f' {"—":<12}'
        print(row)

# =====================================================================
# 2. 排名变动分布（各方法 vs 熵权）
# =====================================================================
print(f'\n\n{"="*80}')
print(f'  排名变动分布：各方法相对熵权的排名偏差')
print(f'{"="*80}')

for method in METHOD_NAMES:
    if method == '熵权':
        continue
    print(f'\n  {method} vs 熵权 的排名偏差:')
    print(f'  {"题目":<6} {"均值±std":>12} {"P50":>6} {"P95":>6} {"最大上移":>8} {"最大下移":>8} {"ρ":>7}')
    print(f'  {"─"*55}')
    
    all_diffs = []
    for topic in TOPICS:
        sub = full[full['题目'] == topic]
        rank_e = sub['排名_熵权'].values
        rank_m = sub[f'排名_{method}'].values
        diff = rank_e - rank_m  # positive = 熵权排名更高, negative = 该方法排名更高
        
        rho, _ = spearmanr(rank_e, rank_m)
        
        print(f'  {topic:<6} {diff.mean():>+7.1f}±{diff.std():>5.1f} '
              f'{np.median(diff):>+6.0f} {np.percentile(np.abs(diff), 95):>6.0f} '
              f'{diff.max():>+8.0f} {diff.min():>+8.0f} {rho:>7.3f}')
        all_diffs.extend(diff.tolist())
    
    all_diffs = np.array(all_diffs)
    print(f'  {"─"*55}')
    print(f'  {"合计":<6} {all_diffs.mean():>+7.1f}±{all_diffs.std():>5.1f} '
          f'{np.median(all_diffs):>+6.0f} {np.percentile(np.abs(all_diffs), 95):>6.0f} '
          f'{all_diffs.max():>+8.0f} {all_diffs.min():>+8.0f}')


# =====================================================================
# 3. 聚类分层一致性
# =====================================================================
print(f'\n\n{"="*80}')
print(f'  聚类分层一致性：不同赋权法下的评委分层是否一致')
print(f'{"="*80}')

# 重新对各方法做 K-means++ 聚类
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

LABELS_AVAIL = ['优秀', '良好', '合格', '需关注', '待改进', '偏低', '偏低2']

def cluster_and_label(jdf, method_name, X_norm, K):
    """对一种赋权法做 K-means++ 聚类并返回分层标签"""
    S = jdf[f'TOPSIS_{method_name}'].values
    features = np.column_stack([S, X_norm])
    
    silhouettes = {}
    for k in range(3, min(6, K)):
        km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels = km.fit_predict(features)
        silhouettes[k] = silhouette_score(features, labels)
    best_k = max(silhouettes, key=silhouettes.get)
    
    km_final = KMeans(n_clusters=best_k, init='k-means++', random_state=42, n_init=10)
    cluster_ids = km_final.fit_predict(features)
    
    # 按 TOPSIS 均值排序命名
    temp = pd.DataFrame({'cid': cluster_ids, 'S': S})
    cluster_order = temp.groupby('cid')['S'].mean().sort_values(ascending=False)
    label_map = {}
    for rank, (cid, _) in enumerate(cluster_order.items()):
        label_map[cid] = LABELS_AVAIL[rank]
    return pd.Series([label_map[c] for c in cluster_ids], index=jdf.index), best_k

# 逐题计算分层
all_clusters = {}  # topic -> {method -> cluster_series}
for topic in TOPICS:
    sub = full[full['题目'] == topic].copy()
    K = len(sub)
    
    # 归一化 X
    X = sub[FEATURE_COLS].values.copy()
    for j in range(4):
        xmin, xmax = X[:,j].min(), X[:,j].max()
        if xmax > xmin:
            X[:,j] = (X[:,j] - xmin) / (xmax - xmin)
    X += 1e-6
    
    all_clusters[topic] = {}
    for m in METHOD_NAMES:
        labels, k = cluster_and_label(sub, m, X, K)
        all_clusters[topic][m] = labels

# 分层一致性：各方法 vs 熵权
print(f'\n  各方法与熵权的分层一致率（评委被分入相同等级的百分比）:')
print(f'  {"题目":<6}', end='')
for m in METHOD_NAMES:
    if m != '熵权':
        print(f' {m:>8}', end='')
print()
print(f'  {"─"*50}')

for topic in TOPICS:
    sub = full[full['题目'] == topic]
    K = len(sub)
    base_labels = all_clusters[topic]['熵权']
    print(f'  {topic:<6} (K={K:>2})', end='')
    for m in METHOD_NAMES:
        if m == '熵权':
            continue
        other_labels = all_clusters[topic][m]
        agree = (base_labels == other_labels).sum()
        print(f' {agree}/{K}({agree/K*100:.0f}%)', end='')
    print()

# 分层转移矩阵（全部题目池化）
print(f'\n  分层转移矩阵（vs 熵权，全部题目池化）:')
for m in METHOD_NAMES:
    if m == '熵权':
        continue
    print(f'\n  ── {m} vs 熵权 ──')
    all_base = []
    all_other = []
    for topic in TOPICS:
        all_base.extend(all_clusters[topic]['熵权'])
        all_other.extend(all_clusters[topic][m])
    
    from sklearn.metrics import confusion_matrix
    unique_labels = sorted(set(all_base + all_other), 
                           key=lambda x: LABELS_AVAIL.index(x) if x in LABELS_AVAIL else 99)
    cm = confusion_matrix(all_base, all_other, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=[f'熵权_{l}' for l in unique_labels],
                         columns=[f'{m}_{l}' for l in unique_labels])
    print(cm_df.to_string())
    agree_total = sum(all_base[i] == all_other[i] for i in range(len(all_base)))
    print(f'  总一致率: {agree_total}/{len(all_base)} ({agree_total/len(all_base)*100:.1f}%)')


# =====================================================================
# 4. 对 Q3 核心结论的影响
# =====================================================================
print(f'\n\n{"="*80}')
print(f'  对 Q3 核心结论的影响')
print(f'{"="*80}')

# Q3 的结论：A类评委（优秀/良好）与 D/E 类评委存在显著差异
# 检查：不同赋权法下，"优秀"和"需关注"两端的评委重叠度

print(f'\n  【极端分层的一致性】')
for topic in TOPICS:
    base_top = all_clusters[topic]['熵权']
    top_judges = set(full[full['题目'] == topic].loc[base_top[base_top == '优秀'].index, '评委ID'])
    bottom_judges = set(full[full['题目'] == topic].loc[base_top[base_top.isin(['需关注', '待改进', '偏低'])].index, '评委ID'])
    
    print(f'\n  {topic}题:')
    print(f'    熵权"优秀": {top_judges}')
    print(f'    熵权"需关注/待改进": {bottom_judges}')
    for m in METHOD_NAMES:
        if m == '熵权':
            continue
        other = all_clusters[topic][m]
        other_top = set(full[full['题目'] == topic].loc[other[other == '优秀'].index, '评委ID'])
        other_bottom = set(full[full['题目'] == topic].loc[other[other.isin(['需关注', '待改进', '偏低'])].index, '评委ID'])
        top_overlap = len(top_judges & other_top)
        bottom_overlap = len(bottom_judges & other_bottom)
        print(f'    {m}: 优秀重叠{top_overlap}/{len(top_judges)} | 需关注重叠{bottom_overlap}/{len(bottom_judges)}')

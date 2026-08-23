"""
阶段 2.1: Q2+Q3 建模代码
- Q2: 计算四维度指标（信度/效度/公平性/区分力）
- Q3: 熵权+TOPSIS+K-means++ 综合评价
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
FIG_DIR = os.path.join(PROBLEM_DIR, 'figures')
OUT_DIR = os.path.join(PROBLEM_DIR, '..', 'solution', 'artifacts', 'tables')
os.makedirs(OUT_DIR, exist_ok=True)

def pairwise_icc(ratings_a, ratings_b):
    """计算两位评分者之间的 ICC(2,1)（双因素随机效应，单一评分者）
    
    ICC(2,1) = (MS_between - MS_error) / (MS_between + MS_error)
    
    参数:
        ratings_a, ratings_b: 两个等长数组, 共同评阅的论文评分
    返回:
        ICC(2,1) 值, 范围 [0, 1]
    """
    n = len(ratings_a)
    if n < 3:
        return np.nan
    grand_mean = (np.mean(ratings_a) + np.mean(ratings_b)) / 2
    paper_means = (np.array(ratings_a) + np.array(ratings_b)) / 2
    SS_between = 2 * np.sum((paper_means - grand_mean) ** 2)
    MS_between = SS_between / (n - 1)
    
    rater_a_mean = np.mean(ratings_a)
    rater_b_mean = np.mean(ratings_b)
    res_a = np.array(ratings_a) - paper_means - rater_a_mean + grand_mean
    res_b = np.array(ratings_b) - paper_means - rater_b_mean + grand_mean
    SS_error = np.sum(res_a ** 2) + np.sum(res_b ** 2)
    MS_error = SS_error / (n - 1)
    
    if MS_between + MS_error == 0:
        return 0.0
    icc = (MS_between - MS_error) / (MS_between + MS_error)
    return max(icc, -1.0)  # 下界保护

# ===== 加载数据 =====
df = pd.read_pickle(os.path.join(DATA_DIR, 'combined-clean.pkl'))

# 展开评委列
records = []
for j in range(1, 5):
    tmp = df[['题目', '阅卷号', '成绩', f'评委{j}', f'打分{j}']].copy()
    tmp.columns = ['题目', '阅卷号', '成绩', '评委ID', '打分']
    records.append(tmp)
long_df = pd.concat(records, ignore_index=True)

# 奖项数值化（淘汰论文=0, 三等=1, 二等=2, 一等=3）
award_map = {'一等奖': 3, '二等奖': 2, '三等奖': 1}
long_df['奖项数值'] = long_df['成绩'].astype(object).map(award_map).fillna(0).astype(int)

TOPICS = ['A', 'B', 'C', 'D', 'E']
all_results = []

for topic in TOPICS:
    print(f'\n{"="*50}')
    print(f'题目 {topic}')
    print(f'{"="*50}')
    
    tdata = long_df[long_df['题目'] == topic].copy()
    judges = sorted(tdata['评委ID'].unique())
    K = len(judges)
    print(f'评委数: {K}')
    
    # ===== Q2: 四维度计算 =====
    judge_metrics = []
    
    for jid in judges:
        j_scores = tdata[tdata['评委ID'] == jid]
        n = len(j_scores)
        scores = j_scores['打分'].values
        
        # 维度1: 信度 — 成对 ICC(2,1) 均值
        pair_iccs = []
        for other_jid in judges:
            if other_jid == jid:
                continue
            other_scores = tdata[tdata['评委ID'] == other_jid]
            common = pd.merge(j_scores[['阅卷号','打分']], other_scores[['阅卷号','打分']],
                             on='阅卷号', suffixes=('_j','_o'))
            if len(common) >= 10:
                icc_val = pairwise_icc(common['打分_j'].values, common['打分_o'].values)
                pair_iccs.append(icc_val)
        reliability = np.nanmean(pair_iccs) if pair_iccs else np.nan
        
        # 维度2: 效度 — 与奖项的Spearman ρ (含淘汰论文, 奖项=0)
        rho, _ = spearmanr(j_scores['打分'], j_scores['奖项数值']) if len(j_scores) >= 10 else (np.nan, None)
        
        # 维度3: 公平性 — |bias z-score|
        topic_mean = tdata['打分'].mean()
        bias = scores.mean() - topic_mean
        # z-score within topic
        all_biases = [tdata[tdata['评委ID']==j]['打分'].mean() - topic_mean for j in judges]
        bias_std = np.std(all_biases)
        fairness = abs(bias) / bias_std if bias_std > 0 else 0
        
        # 维度4: 区分力 — 评分标准差
        discrimination = scores.std()
        
        judge_metrics.append({
            '题目': topic, '评委ID': jid, '阅卷量': n,
            '信度': reliability, '效度': rho,
            '公平性': fairness, '区分力': discrimination,
            '评分均值': scores.mean(),
            '评分标准差': discrimination,
            '获奖率': (j_scores['奖项数值'] > 0).sum() / n,
        })
    
    jdf = pd.DataFrame(judge_metrics)
    
    # 填充NaN (信度/效度)
    jdf['信度'] = jdf['信度'].fillna(jdf['信度'].median())
    jdf['效度'] = jdf['效度'].fillna(0)
    
    # ===== 归一化 (正向化，全部越大越好) =====
    feature_cols = ['信度', '效度', '公平性', '区分力']
    # 公平性越小越好 → 取反
    jdf['公平性_raw'] = jdf['公平性']
    jdf['公平性'] = -jdf['公平性']
    
    X_norm = np.zeros((K, 4))
    for j, col in enumerate(feature_cols):
        x = jdf[col].values
        xmin, xmax = x.min(), x.max()
        if xmax > xmin:
            X_norm[:, j] = (x - xmin) / (xmax - xmin)
        else:
            X_norm[:, j] = 0.5
    X_norm += 1e-6
    
    # ===== 熵权法 =====
    P = X_norm / X_norm.sum(axis=0, keepdims=True)
    ent = -np.sum(P * np.log(P), axis=0) / np.log(K)
    d = 1 - ent
    w = d / d.sum()
    
    print(f'熵权: 信度={w[0]:.3f}, 效度={w[1]:.3f}, 公平性={w[2]:.3f}, 区分力={w[3]:.3f}')
    
    # ===== TOPSIS =====
    W_norm = X_norm * w
    Z_plus = np.max(W_norm, axis=0)
    Z_minus = np.min(W_norm, axis=0)
    D_plus = np.sqrt(np.sum((W_norm - Z_plus)**2, axis=1))
    D_minus = np.sqrt(np.sum((W_norm - Z_minus)**2, axis=1))
    S = D_minus / (D_plus + D_minus)
    jdf['TOPSIS得分'] = S
    
    # 排名
    jdf['排名'] = jdf['TOPSIS得分'].rank(ascending=False).astype(int)
    
    # ===== 等权TOPSIS (稳健性对照) =====
    w_eq = np.ones(4) / 4
    W_eq = X_norm * w_eq
    Dp_eq = np.sqrt(np.sum((W_eq - np.max(W_eq, axis=0))**2, axis=1))
    Dm_eq = np.sqrt(np.sum((W_eq - np.min(W_eq, axis=0))**2, axis=1))
    S_eq = Dm_eq / (Dp_eq + Dm_eq)
    jdf['TOPSIS等权'] = S_eq
    jdf['排名等权'] = jdf['TOPSIS等权'].rank(ascending=False).astype(int)
    rho_robust, _ = spearmanr(S, S_eq)
    print(f'熵权vs等权 Spearman: {rho_robust:.3f}')
    
    # ===== K-means++ 分层 =====
    features_for_km = np.column_stack([S, X_norm])  # TOPSIS得分+四维度
    silhouettes = {}
    inertias = {}
    for k in range(3, min(6, K)):  # 业务约束: K≥3
        km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels = km.fit_predict(features_for_km)
        silhouettes[k] = silhouette_score(features_for_km, labels)
        inertias[k] = km.inertia_
    
    best_k = max(silhouettes, key=silhouettes.get)
    km_final = KMeans(n_clusters=best_k, init='k-means++', random_state=42, n_init=10)
    jdf['聚类'] = km_final.fit_predict(features_for_km)
    jdf['Silhouette'] = silhouettes[best_k]
    
    # 聚类标签命名
    cluster_means = jdf.groupby('聚类')['TOPSIS得分'].mean().sort_values(ascending=False)
    label_map = {}
    labels_avail = ['优秀', '良好', '合格', '需关注', '待改进', '偏低', '偏低2']
    for rank, (cid, _) in enumerate(cluster_means.items()):
        label_map[cid] = labels_avail[rank]
    jdf['分层'] = jdf['聚类'].map(label_map)
    
    print(f'聚类K={best_k}, Silhouette={silhouettes[best_k]:.3f}, 分层: {dict(label_map)}')
    for lid, group in jdf.groupby('分层'):
        print(f'  {lid}: {len(group)}位, TOPSIS均值={group["TOPSIS得分"].mean():.3f}')
    
    # ===== 各维度组内排名 =====
    for col in feature_cols:
        jdf[f'{col}_排名'] = jdf[col].rank(ascending=False).astype(int)
    
    all_results.append(jdf)

# ===== 合并全题结果 =====
full = pd.concat(all_results, ignore_index=True)

# ===== TOP 5 / BOTTOM 5 =====
print(f'\n{"="*50}')
print('全题汇总')
for topic in TOPICS:
    sub = full[full['题目'] == topic].sort_values('排名')
    print(f'\n{topic}题 TOP3:')
    for _, r in sub.head(3).iterrows():
        print(f'  第{r["排名"]}名 {r["评委ID"]}: S={r["TOPSIS得分"]:.3f} ({r["分层"]})')
    print(f'{topic}题 BOTTOM3:')
    for _, r in sub.tail(3).iterrows():
        print(f'  第{r["排名"]}名 {r["评委ID"]}: S={r["TOPSIS得分"]:.3f} ({r["分层"]})')

# ===== 保存结果 =====
full.to_csv(os.path.join(OUT_DIR, 'q3-judge-scores.csv'), index=False, encoding='utf-8-sig')
full.to_pickle(os.path.join(DATA_DIR, 'q3-judge-scores.pkl'))
print(f'\n结果已保存: q3-judge-scores.csv ({len(full)}行)')

# ===== 生成LaTeX结果表 =====
for topic in TOPICS:
    sub = full[full['题目'] == topic].sort_values('排名')
    tex_lines = []
    tex_lines.append(r'\begin{table}[h]')
    tex_lines.append(r'\centering')
    tex_lines.append(r'\caption{' + f'{topic}题评委综合素质排序（熵权+TOPSIS）' + r'}')
    tex_lines.append(r'\begin{tabular}{cccccc}')
    tex_lines.append(r'\hline')
    tex_lines.append(r'排名 & 评委ID & TOPSIS & 信度 & 效度 & 分层 \\')
    tex_lines.append(r'\hline')
    for _, r in sub.head(10).iterrows():
        tex_lines.append(f'{r["排名"]} & {r["评委ID"]} & {r["TOPSIS得分"]:.3f} & {r["信度"]:.3f} & {r["效度"]:.3f} & {r["分层"]} \\\\')
    tex_lines.append(r'\hline')
    tex_lines.append(r'\end{tabular}')
    tex_lines.append(r'\end{table}')
    with open(os.path.join(OUT_DIR, f'q3-ranking-{topic}.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex_lines))

# ===== 汇总统计 =====
print(f'\n===== 五题TOPSIS汇总 =====')
for topic in TOPICS:
    sub = full[full['题目'] == topic]
    print(f'{topic}: mean={sub["TOPSIS得分"].mean():.3f}, std={sub["TOPSIS得分"].std():.3f}, '
          f'min={sub["TOPSIS得分"].min():.3f}, max={sub["TOPSIS得分"].max():.3f}, K={len(sub)}')

print(f'\n===== 熵权分布 =====')
for topic in TOPICS:
    sub = full[full['题目'] == topic]
    X = sub[feature_cols].values.copy()
    for j in range(4):
        col = feature_cols[j]
        xmin, xmax = X[:,j].min(), X[:,j].max()
        if xmax > xmin:
            X[:,j] = (X[:,j] - xmin) / (xmax - xmin)
    X += 1e-6
    P = X / X.sum(axis=0, keepdims=True)
    ent = -np.sum(P * np.log(P), axis=0) / np.log(len(sub))
    w_t = (1 - ent) / (1 - ent).sum()
    print(f'{topic}: 信度={w_t[0]:.3f}, 效度={w_t[1]:.3f}, 公平性={w_t[2]:.3f}, 区分力={w_t[3]:.3f}')

"""
CV法（变异系数法）+ TOPSIS + K-means++ 评委综合评价
=====================================================================
替代赋权法探索：以变异系数法替代熵权法，其余流程不变。

CV法权重: w_j = CV_j / Σ CV_j, 其中 CV_j = σ_j / μ_j
意义: 各维度的相对离散程度越大，权重越高。
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
OUT_DIR = os.path.join(PROBLEM_DIR, '..', 'solution', 'explore-results', 'cv-method')
os.makedirs(OUT_DIR, exist_ok=True)


def pairwise_icc(ratings_a, ratings_b):
    """ICC(2,1) 双因素随机效应"""
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
    return max(icc, -1.0)


def weight_cv(X_norm):
    """变异系数法赋权"""
    mu = np.mean(X_norm, axis=0)
    sigma = np.std(X_norm, axis=0, ddof=1)
    cv = np.abs(sigma / (mu + 1e-8))
    return cv / cv.sum()


def topsis_score(X_norm, w):
    """TOPSIS 贴近度"""
    W = X_norm * w
    Zp = np.max(W, axis=0)
    Zm = np.min(W, axis=0)
    Dp = np.sqrt(np.sum((W - Zp)**2, axis=1))
    Dm = np.sqrt(np.sum((W - Zm)**2, axis=1))
    return Dm / (Dp + Dm)


# ===== 加载数据 =====
df = pd.read_pickle(os.path.join(DATA_DIR, 'combined-clean.pkl'))
records = []
for j in range(1, 5):
    tmp = df[['题目', '阅卷号', '成绩', f'评委{j}', f'打分{j}']].copy()
    tmp.columns = ['题目', '阅卷号', '成绩', '评委ID', '打分']
    records.append(tmp)
long_df = pd.concat(records, ignore_index=True)

award_map = {'一等奖': 3, '二等奖': 2, '三等奖': 1}
long_df['奖项数值'] = long_df['成绩'].astype(object).map(award_map).fillna(0).astype(int)

TOPICS = ['A', 'B', 'C', 'D', 'E']
FEATURE_COLS = ['信度', '效度', '公平性', '区分力']
all_results = []

for topic in TOPICS:
    print(f'\n{"="*50}')
    print(f'  题目 {topic} (CV法)')
    print(f'{"="*50}')

    tdata = long_df[long_df['题目'] == topic].copy()
    judges = sorted(tdata['评委ID'].unique())
    K = len(judges)
    print(f'  评委数: {K}')

    # === Q2: 四维度计算 ===
    judge_metrics = []
    for jid in judges:
        j_scores = tdata[tdata['评委ID'] == jid]
        n = len(j_scores)
        scores = j_scores['打分'].values

        # 信度
        pair_iccs = []
        for other_jid in judges:
            if other_jid == jid:
                continue
            other_scores = tdata[tdata['评委ID'] == other_jid]
            common = pd.merge(
                j_scores[['阅卷号', '打分']], other_scores[['阅卷号', '打分']],
                on='阅卷号', suffixes=('_j', '_o'))
            if len(common) >= 10:
                icc_val = pairwise_icc(common['打分_j'].values, common['打分_o'].values)
                pair_iccs.append(icc_val)
        reliability = np.nanmean(pair_iccs) if pair_iccs else np.nan

        # 效度
        rho, _ = spearmanr(j_scores['打分'], j_scores['奖项数值']) if len(j_scores) >= 10 else (np.nan, None)

        # 公平性
        topic_mean = tdata['打分'].mean()
        bias = scores.mean() - topic_mean
        all_biases = [tdata[tdata['评委ID'] == j]['打分'].mean() - topic_mean for j in judges]
        bias_std = np.std(all_biases)
        fairness = abs(bias) / bias_std if bias_std > 0 else 0

        # 区分力
        discrimination = scores.std()

        judge_metrics.append({
            '题目': topic, '评委ID': jid, '阅卷量': n,
            '信度': reliability, '效度': rho,
            '公平性': fairness, '区分力': discrimination,
            '评分均值': scores.mean(), '评分标准差': discrimination,
            '获奖率': (j_scores['奖项数值'] > 0).sum() / n,
        })

    jdf = pd.DataFrame(judge_metrics)
    jdf['信度'] = jdf['信度'].fillna(jdf['信度'].median())
    jdf['效度'] = jdf['效度'].fillna(0)
    jdf['公平性_raw'] = jdf['公平性']
    jdf['公平性'] = -jdf['公平性']  # 正向化

    # 归一化
    X_norm = np.zeros((K, 4))
    for j, col in enumerate(FEATURE_COLS):
        x = jdf[col].values
        xmin, xmax = x.min(), x.max()
        if xmax > xmin:
            X_norm[:, j] = (x - xmin) / (xmax - xmin)
        else:
            X_norm[:, j] = 0.5
    X_norm += 1e-6

    # === CV法赋权 ===
    w = weight_cv(X_norm)
    print(f'  CV权重: 信度={w[0]:.3f}, 效度={w[1]:.3f}, 公平性={w[2]:.3f}, 区分力={w[3]:.3f}')

    # === TOPSIS ===
    S = topsis_score(X_norm, w)
    jdf['TOPSIS得分'] = S
    jdf['排名'] = pd.Series(S).rank(ascending=False).astype(int)

    # === 等权TOPSIS (稳健性对照) ===
    w_eq = np.ones(4) / 4
    S_eq = topsis_score(X_norm, w_eq)
    jdf['TOPSIS等权'] = S_eq
    jdf['排名等权'] = pd.Series(S_eq).rank(ascending=False).astype(int)
    rho_robust, _ = spearmanr(S, S_eq)
    print(f'  CV vs 等权 Spearman: {rho_robust:.3f}')

    # === K-means++ 分层 ===
    features_for_km = np.column_stack([S, X_norm])
    silhouettes = {}
    for k in range(3, min(6, K)):
        km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels = km.fit_predict(features_for_km)
        silhouettes[k] = silhouette_score(features_for_km, labels)

    best_k = max(silhouettes, key=silhouettes.get)
    km_final = KMeans(n_clusters=best_k, init='k-means++', random_state=42, n_init=10)
    jdf['聚类'] = km_final.fit_predict(features_for_km)
    jdf['Silhouette'] = silhouettes[best_k]

    # 分层标签命名
    cluster_means = jdf.groupby('聚类')['TOPSIS得分'].mean().sort_values(ascending=False)
    labels_avail = ['优秀', '良好', '合格', '需关注', '待改进', '偏低', '偏低2']
    label_map = {}
    for rank, (cid, _) in enumerate(cluster_means.items()):
        label_map[cid] = labels_avail[rank]
    jdf['分层'] = jdf['聚类'].map(label_map)

    print(f'  聚类K={best_k}, Silhouette={silhouettes[best_k]:.3f}, 分层: {dict(label_map)}')
    for lid, group in jdf.groupby('分层'):
        print(f'    {lid}: {len(group)}位, TOPSIS均值={group["TOPSIS得分"].mean():.3f}')

    # 维度组内排名
    for col in FEATURE_COLS:
        jdf[f'{col}_排名'] = jdf[col].rank(ascending=False).astype(int)

    all_results.append(jdf)

# ===== 合并 =====
full = pd.concat(all_results, ignore_index=True)

# ===== 保存 =====
full.to_csv(os.path.join(OUT_DIR, 'q3-judge-scores-cv.csv'), index=False, encoding='utf-8-sig')
full.to_pickle(os.path.join(OUT_DIR, 'q3-judge-scores-cv.pkl'))

# ===== 打印摘要 =====
print(f'\n{"="*50}')
print(f'  CV法 全题摘要')
print(f'{"="*50}')
for topic in TOPICS:
    sub = full[full['题目'] == topic]
    print(f'\n  {topic}题 TOPSIS Top-5:')
    for _, r in sub.sort_values('排名').head(5).iterrows():
        print(f'    #{r["排名"]} {r["评委ID"]}: S={r["TOPSIS得分"]:.3f} ({r["分层"]})')

print(f'\n  五题 TOPSIS 统计:')
for topic in TOPICS:
    sub = full[full['题目'] == topic]
    print(f'  {topic}: mean={sub["TOPSIS得分"].mean():.3f}, std={sub["TOPSIS得分"].std():.3f}, '
          f'min={sub["TOPSIS得分"].min():.3f}, max={sub["TOPSIS得分"].max():.3f}')

# ===== LaTeX 排名表 =====
for topic in TOPICS:
    sub = full[full['题目'] == topic].sort_values('排名')
    tex = [r'\begin{table}[h]', r'\centering',
           r'\caption{' + f'{topic}题评委综合素质排序（CV法+TOPSIS）' + r'}',
           r'\begin{tabular}{cccccc}', r'\hline',
           r'排名 & 评委ID & TOPSIS & 信度 & 效度 & 分层 \\', r'\hline']
    for _, r in sub.head(10).iterrows():
        tex.append(f'{r["排名"]} & {r["评委ID"]} & {r["TOPSIS得分"]:.3f} & '
                   f'{r["信度"]:.3f} & {r["效度"]:.3f} & {r["分层"]} \\\\')
    tex.append(r'\hline')
    tex.append(r'\end{tabular}')
    tex.append(r'\end{table}')
    with open(os.path.join(OUT_DIR, f'q3-ranking-cv-{topic}.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex))

print(f'\n结果已保存至: {OUT_DIR}')

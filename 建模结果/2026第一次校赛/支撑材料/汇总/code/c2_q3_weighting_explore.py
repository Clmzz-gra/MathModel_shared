"""
替代赋权法探索：对比 6 种赋权方法在 Q3 评委综合评价中的表现
=====================================================================
方法清单:
  1. 熵权法 (Entropy)         — 当前 baseline
  2. CRITIC 法                 — 标准差 × 独立性
  3. 变异系数法 (CV)            — σ/μ 归一化
  4. 标准差法 (Std)             — 纯 σ 归一化
  5. PCA 法                    — 第一主成分载荷
  6. 灰色关联度法 (GRA)          — 关联度均值赋权

对比维度:
  - 权重分布 (各维度权重值)
  - TOPSIS 排名一致性 (Spearman ρ 矩阵)
  - 排名变动幅度 (与熵权的 RMS 排名差)
  - 聚类结果敏感性
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings, os
warnings.filterwarnings('ignore')

PROBLEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROBLEM_DIR, 'data')
EXPLORE_DIR = os.path.join(PROBLEM_DIR, '..', 'solution', 'explore-results', 'weighting')
os.makedirs(EXPLORE_DIR, exist_ok=True)

# =====================================================================
# 数据准备（复用 c2_q2q3_model.py 的 Q2 四维度计算）
# =====================================================================
def pairwise_icc(ratings_a, ratings_b):
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


# =====================================================================
# 赋权方法实现
# =====================================================================

def weight_entropy(X_norm):
    """熵权法 (baseline)"""
    K = X_norm.shape[0]
    P = X_norm / X_norm.sum(axis=0, keepdims=True)
    ent = -np.sum(P * np.log(P), axis=0) / np.log(K)
    d = 1 - ent
    return d / d.sum()


def weight_critic(X_norm):
    """
    CRITIC 法: C_j = σ_j * Σ_i(1 - r_ij)
    权重 = C_j / Σ C_j
    
    X_norm: (K, 4) 归一化矩阵
    参考: Diakoulaki et al. (1995)
    """
    stds = np.std(X_norm, axis=0, ddof=1)
    n_features = X_norm.shape[1]
    conflict = np.zeros(n_features)
    corr = np.corrcoef(X_norm.T)
    for j in range(n_features):
        conflict[j] = np.sum(1.0 - np.abs(corr[j, :]))
    C = stds * conflict
    return C / C.sum()


def weight_cv(X_norm):
    """变异系数法: w_j = CV_j / Σ CV_j, CV_j = σ_j / μ_j"""
    mu = np.mean(X_norm, axis=0)
    sigma = np.std(X_norm, axis=0, ddof=1)
    cv = np.abs(sigma / (mu + 1e-8))
    return cv / cv.sum()


def weight_std(X_norm):
    """标准差法: w_j = σ_j / Σ σ_j"""
    sigma = np.std(X_norm, axis=0, ddof=1)
    return sigma / sigma.sum()


def weight_pca(X_norm):
    """
    PCA 赋权: w_j ∝ |loading_j of PC1|
    PC1 贡献率最高, 其载荷反映各维度对主方向的贡献
    """
    pca = PCA(n_components=1)
    pca.fit(X_norm)
    loadings = np.abs(pca.components_[0])
    return loadings / loadings.sum()


def weight_gra(X_norm):
    """
    灰色关联度赋权:
    1. 以每一维度为参考序列, 计算其余维度的关联度
    2. 各维度的关联度均值作为该维度的区分能力指标
    3. 归一化得权重
    """
    K, m = X_norm.shape
    # 对各维度分别作为参考序列
    rel_degree = np.zeros(m)
    for j in range(m):
        ref = X_norm[:, j:j+1]
        comp = X_norm  # 全部维度作为比较
        # 计算灰色关联系数
        diff = np.abs(comp - ref)
        min_diff = diff.min()
        max_diff = diff.max()
        if max_diff == min_diff:
            rel_degree[j] = 1.0
            continue
        rho = 0.5  # 分辨系数
        xi = (min_diff + rho * max_diff) / (diff + rho * max_diff)
        # 该维度对全部维度的平均关联度
        rel_degree[j] = np.mean(xi[:, j])
    # 关联度越大 → 该维度越"典型" → 权重越大
    return rel_degree / rel_degree.sum()


def topsis_score(X_norm, w):
    """TOPSIS: 返回贴近度 S_i"""
    W = X_norm * w
    Zp = np.max(W, axis=0)
    Zm = np.min(W, axis=0)
    Dp = np.sqrt(np.sum((W - Zp)**2, axis=1))
    Dm = np.sqrt(np.sum((W - Zm)**2, axis=1))
    return Dm / (Dp + Dm)


METHODS = {
    '熵权': weight_entropy,
    'CRITIC': weight_critic,
    'CV': weight_cv,
    '标准差': weight_std,
    'PCA': weight_pca,
    'GRA': weight_gra,
}


# =====================================================================
# 主循环：逐题计算
# =====================================================================
all_weights = {}      # topic → {method → [w1,w2,w3,w4]}
all_scores = {}       # topic → {method → (scores, ranks)}
all_dataframes = {}   # topic → full judge df

for topic in TOPICS:
    print(f'\n{"="*60}')
    print(f'  题目 {topic}')
    print(f'{"="*60}')
    
    # --- Q2 四维度计算 (同原流程) ---
    tdata = long_df[long_df['题目'] == topic].copy()
    judges = sorted(tdata['评委ID'].unique())
    K = len(judges)
    
    judge_metrics = []
    for jid in judges:
        j_scores = tdata[tdata['评委ID'] == jid]
        n = len(j_scores)
        scores = j_scores['打分'].values
        
        pair_iccs = []
        for other_jid in judges:
            if other_jid == jid:
                continue
            other_scores = tdata[tdata['评委ID'] == other_jid]
            common = pd.merge(
                j_scores[['阅卷号', '打分']], other_scores[['阅卷号', '打分']],
                on='阅卷号', suffixes=('_j', '_o')
            )
            if len(common) >= 10:
                icc_val = pairwise_icc(common['打分_j'].values, common['打分_o'].values)
                pair_iccs.append(icc_val)
        reliability = np.nanmean(pair_iccs) if pair_iccs else np.nan
        
        rho, _ = spearmanr(j_scores['打分'], j_scores['奖项数值']) if len(j_scores) >= 10 else (np.nan, None)
        
        topic_mean = tdata['打分'].mean()
        bias = scores.mean() - topic_mean
        all_biases = [tdata[tdata['评委ID'] == j]['打分'].mean() - topic_mean for j in judges]
        bias_std = np.std(all_biases)
        fairness = abs(bias) / bias_std if bias_std > 0 else 0
        
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
    
    # --- 计算各方法权重和TOPSIS得分 ---
    w_topic = {}
    s_topic = {}
    
    for name, weight_fn in METHODS.items():
        w = weight_fn(X_norm)
        w_topic[name] = w
        S = topsis_score(X_norm, w)
        ranks = pd.Series(S).rank(ascending=False).astype(int).values
        s_topic[name] = (S, ranks)
        
        # 存入 DataFrame 便于后续分析
        jdf[f'TOPSIS_{name}'] = S
        jdf[f'排名_{name}'] = ranks
    
    all_weights[topic] = w_topic
    all_scores[topic] = s_topic
    all_dataframes[topic] = jdf.copy()
    
    # --- 打印权重对比 ---
    print(f'\n  权重对比 (K={K}位评委):')
    print(f'  {"方法":<8} {"信度":>8} {"效度":>8} {"公平性":>8} {"区分力":>8}')
    print(f'  {"-"*40}')
    for name, w in w_topic.items():
        marker = ' ◀' if name == '熵权' else ''
        print(f'  {name+marker:<8} {w[0]:>8.3f} {w[1]:>8.3f} {w[2]:>8.3f} {w[3]:>8.3f}')


# =====================================================================
# 跨题汇总分析
# =====================================================================

print(f'\n\n{"="*60}')
print(f'  跨题汇总')
print(f'{"="*60}')

# 1. 各方法平均权重（五题均值 ± std）
print(f'\n  各方法五题平均权重:')
print(f'  {"方法":<8} {"信度":>14} {"效度":>14} {"公平性":>14} {"区分力":>14}')
print(f'  {"-"*56}')
for name in METHODS:
    ws = np.array([all_weights[t][name] for t in TOPICS])
    mean_w = ws.mean(axis=0)
    std_w = ws.std(axis=0)
    print(f'  {name:<8} {mean_w[0]:>6.3f}±{std_w[0]:.3f}  {mean_w[1]:>6.3f}±{std_w[1]:.3f}  '
          f'{mean_w[2]:>6.3f}±{std_w[2]:.3f}  {mean_w[3]:>6.3f}±{std_w[3]:.3f}')

# 2. 排名相关性矩阵（所有题目所有评委池化）
print(f'\n  排名 Spearman 相关性矩阵（五题池化）:')
method_names = list(METHODS.keys())
all_ranks = {name: [] for name in method_names}
for topic in TOPICS:
    for name in method_names:
        all_ranks[name].extend(list(all_scores[topic][name][1]))

rho_matrix = np.zeros((len(method_names), len(method_names)))
for i, n1 in enumerate(method_names):
    for j, n2 in enumerate(method_names):
        rho_matrix[i, j], _ = spearmanr(all_ranks[n1], all_ranks[n2])

print(f'  {"":>8} ' + ' '.join([f'{n:>8}' for n in method_names]))
for i, n1 in enumerate(method_names):
    print(f'  {n1:<8} ' + ' '.join([f'{rho_matrix[i,j]:>8.3f}' for j in range(len(method_names))]))

# 3. 与熵权的排名偏差（RMS）
print(f'\n  各方法与熵权的 RMS 排名差:')
for topic in TOPICS:
    K = len(all_dataframes[topic])
    baseline_ranks = all_scores[topic]['熵权'][1]
    print(f'  {topic}题 (K={K}):', end=' ')
    for name in method_names:
        if name == '熵权':
            continue
        other_ranks = all_scores[topic][name][1]
        rms_diff = np.sqrt(np.mean((baseline_ranks - other_ranks)**2))
        same_pct = np.mean(baseline_ranks == other_ranks) * 100
        print(f'{name}={rms_diff:.1f}(一致{same_pct:.0f}%)', end='  ')
    print()

# 4. Top-3 一致性
print(f'\n  各方法 Top-3 评委与熵权法的一致性:')
for topic in TOPICS:
    baseline_top3 = set(all_dataframes[topic].nlargest(3, 'TOPSIS_熵权')['评委ID'])
    print(f'  {topic}题 熵权Top3: {baseline_top3}')
    for name in method_names:
        if name == '熵权':
            continue
        other_top3 = set(all_dataframes[topic].nlargest(3, f'TOPSIS_{name}')['评委ID'])
        overlap = baseline_top3 & other_top3
        print(f'    {name}: {other_top3}  重叠{len(overlap)}/3')
    print()


# =====================================================================
# K-means 聚类敏感性：不同赋权下的聚类结果对比
# =====================================================================
print(f'{"="*60}')
print(f'  K-means 聚类敏感性分析')
print(f'{"="*60}')

cluster_comparison = []
for method_name in method_names:
    for topic in TOPICS:
        jdf = all_dataframes[topic]
        K = len(jdf)
        S_col = f'TOPSIS_{method_name}'
        S = jdf[S_col].values
        X = jdf[FEATURE_COLS].values.copy()
        for j in range(4):
            xmin, xmax = X[:,j].min(), X[:,j].max()
            if xmax > xmin:
                X[:,j] = (X[:,j] - xmin) / (xmax - xmin)
        X += 1e-6
        
        features = np.column_stack([S, X])
        
        # Silhouette 选 K (同原流程)
        silhouettes = {}
        for k in range(3, min(6, K)):
            km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            labels = km.fit_predict(features)
            silhouettes[k] = silhouette_score(features, labels)
        best_k = max(silhouettes, key=silhouettes.get)
        best_sil = silhouettes[best_k]
        
        cluster_comparison.append({
            '题目': topic, '方法': method_name, 'K': best_k,
            'Silhouette': best_sil, '评委数': K,
        })

cluster_df = pd.DataFrame(cluster_comparison)
print(f'\n  各方法逐题聚类效果:')
print(cluster_df.pivot(index='题目', columns='方法', values=['K', 'Silhouette']).to_string())

# 汇总
print(f'\n  各方法五题平均 Silhouette:')
for name in method_names:
    sub = cluster_df[cluster_df['方法'] == name]
    print(f'  {name}: Sil={sub["Silhouette"].mean():.3f} (±{sub["Silhouette"].std():.3f}), '
          f'平均K={sub["K"].mean():.1f}')

# =====================================================================
# 权重稳健性：跨题波动分析
# =====================================================================
print(f'\n{"="*60}')
print(f'  权重跨题稳定性')
print(f'{"="*60}')
print(f'  (CV越小 → 跨题越稳定)')
print(f'  {"方法":<8} {"信度CV":>10} {"效度CV":>10} {"公平性CV":>10} {"区分力CV":>10}')
print(f'  {"-"*48}')
for name in method_names:
    ws = np.array([all_weights[t][name] for t in TOPICS])
    cv_each = ws.std(axis=0, ddof=1) / (ws.mean(axis=0) + 1e-8)
    print(f'  {name:<8} {cv_each[0]:>10.3f} {cv_each[1]:>10.3f} {cv_each[2]:>10.3f} {cv_each[3]:>10.3f}')


# =====================================================================
# 保存结果
# =====================================================================
# 权重汇总表
weights_summary = []
for topic in TOPICS:
    for name in method_names:
        w = all_weights[topic][name]
        weights_summary.append({
            '题目': topic, '方法': name,
            '信度': w[0], '效度': w[1],
            '公平性': w[2], '区分力': w[3],
        })
wdf = pd.DataFrame(weights_summary)
wdf.to_csv(os.path.join(EXPLORE_DIR, 'weights-comparison.csv'), index=False, encoding='utf-8-sig')

# 排名相关系数矩阵
rho_df = pd.DataFrame(rho_matrix, index=method_names, columns=method_names)
rho_df.to_csv(os.path.join(EXPLORE_DIR, 'ranking-spearman-matrix.csv'), encoding='utf-8-sig')

# 聚类敏感性
cluster_df.to_csv(os.path.join(EXPLORE_DIR, 'cluster-sensitivity.csv'), index=False, encoding='utf-8-sig')

# 各题完整评委得分（含全部方法）
full_output = []
for topic in TOPICS:
    full_output.append(all_dataframes[topic])
full_df = pd.concat(full_output, ignore_index=True)
score_cols = ['题目','评委ID'] + [f'TOPSIS_{n}' for n in method_names] + [f'排名_{n}' for n in method_names]
full_df[score_cols].to_csv(os.path.join(EXPLORE_DIR, 'all-scores-all-methods.csv'), index=False, encoding='utf-8-sig')

print(f'\n结果已保存至: {EXPLORE_DIR}')
print(f'  - weights-comparison.csv')
print(f'  - ranking-spearman-matrix.csv')
print(f'  - cluster-sensitivity.csv')
print(f'  - all-scores-all-methods.csv')

"""
阶段 2.1: Q1 相关性分析 + Q4 差异分析
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, kruskal, skew, kurtosis
import os, warnings
warnings.filterwarnings('ignore')

PROBLEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROBLEM_DIR, 'data')
OUT_DIR = os.path.join(PROBLEM_DIR, '..', 'solution', 'artifacts', 'tables')

# ===== Q1: 相关性分析 =====
df = pd.read_pickle(os.path.join(DATA_DIR, 'combined-clean.pkl'))
award_map = {'一等奖': 3, '二等奖': 2, '三等奖': 1}

# 计算每篇论文的网评标准分 — 按评委ID做z-score
# 第一步: 建立评委ID -> (均值, 标准差) 映射
judge_stats = {}
for j in range(1, 5):
    for _, row in df.iterrows():
        jid = row[f'评委{j}']
        score = row[f'打分{j}']
        if jid not in judge_stats:
            judge_stats[jid] = []
        judge_stats[jid].append(score)
judge_mu_sigma = {jid: (np.mean(scores), np.std(scores)) for jid, scores in judge_stats.items()}

# 第二步: 每篇论文按4位评委各自的统计量做z-score, 取均值
records = []
for idx, row in df.iterrows():
    z_scores = []
    for j in range(1, 5):
        jid = row[f'评委{j}']
        mu, sigma = judge_mu_sigma[jid]
        raw = row[f'打分{j}']
        z = (raw - mu) / sigma if sigma > 0 else 0
        z_scores.append(z)
    z_mean = np.mean(z_scores)
    award_val = award_map.get(row['成绩'], 0)
    records.append({'题目': row['题目'], '阅卷号': row['阅卷号'],
                   '网评标准分均值': z_mean, '奖项数值': award_val,
                   '是否获奖': pd.notna(row['成绩']), '是否一等': row['成绩'] == '一等奖'})

q1df = pd.DataFrame(records)

print('===== Q1: 网评与终评相关性 =====')

# Spearman
for topic in ['A','B','C','D','E','全题']:
    if topic == '全题':
        sub = q1df
    else:
        sub = q1df[q1df['题目'] == topic]
    rho, p = spearmanr(sub['网评标准分均值'], sub['奖项数值'])
    n = len(sub)
    # 仅入围论文的Pearson (需正态检验: |偏度|<1, |峰度|<3)
    sub_in = sub[sub['是否获奖']]
    if len(sub_in) > 30:
        sk = skew(sub_in['网评标准分均值'])
        ku = kurtosis(sub_in['网评标准分均值'])
        if abs(sk) < 1 and abs(ku) < 3:
            r, p_pearson = pearsonr(sub_in['网评标准分均值'], sub_in['奖项数值'])
        else:
            r, p_pearson = np.nan, np.nan
    else:
        r, p_pearson = np.nan, np.nan
    print(f'{topic}: Spearman ρ={rho:.3f} (n={n}), Pearson r={r:.3f} (n={len(sub_in)})')

# 筛选命中率
print(f'\n===== 筛选命中率 =====')
for topic in ['A','B','C','D','E','全题']:
    sub = q1df if topic == '全题' else q1df[q1df['题目'] == topic]
    sub = sub.sort_values('网评标准分均值', ascending=False)
    n_total = len(sub)
    cutoff = int(n_total * 0.55)
    top = sub.head(cutoff)
    bottom = sub.tail(n_total - cutoff)
    hit_rate = top['是否获奖'].sum() / len(top)
    false_neg = bottom['是否获奖'].sum() / (bottom['是否获奖'].sum() + top['是否获奖'].sum()) if sub['是否获奖'].sum() > 0 else 0
    print(f'{topic}: 命中率={hit_rate:.2%}, 假阴性率={false_neg:.2%}, 前55%获奖{top["是否获奖"].sum()}/{len(top)}')

# ROC AUC (预测一等奖)
from sklearn.metrics import roc_auc_score
print(f'\n===== ROC AUC (预测一等奖) =====')
for topic in ['A','B','C','D','E','全题']:
    sub = q1df if topic == '全题' else q1df[q1df['题目'] == topic]
    if sub['是否一等'].sum() > 5:
        auc = roc_auc_score(sub['是否一等'], sub['网评标准分均值'])
        print(f'{topic}: AUC={auc:.3f}')

# ===== Q4: 差异分析 =====
print(f'\n{"="*50}')
print('===== Q4: 题目间差异分析 =====')

q3df = pd.read_pickle(os.path.join(DATA_DIR, 'q3-judge-scores.pkl'))

# Kruskal-Wallis
groups = [q3df[q3df['题目'] == t]['TOPSIS得分'].values for t in ['A','B','C','D','E']]
H, p_kw = kruskal(*groups)
N = len(q3df)
k = 5
eta2 = (H - k + 1) / (N - k)
print(f'Kruskal-Wallis: H={H:.3f}, p={p_kw:.4f}, η²={eta2:.3f}')

# Dunn 事后检验
from scipy.stats import rankdata
import itertools
combined_ranks = rankdata(q3df['TOPSIS得分'].values)
rank_map = dict(zip(q3df.index, combined_ranks))

print(f'\nDunn事后检验 (Bonferroni校正):')
for t1, t2 in itertools.combinations(['A','B','C','D','E'], 2):
    r1 = q3df[q3df['题目'] == t1].index.map(rank_map).values
    r2 = q3df[q3df['题目'] == t2].index.map(rank_map).values
    n1, n2 = len(r1), len(r2)
    R1_mean, R2_mean = r1.mean(), r2.mean()
    SE = np.sqrt(N*(N+1)/12 * (1/n1 + 1/n2))
    z = (R1_mean - R2_mean) / SE
    from scipy.stats import norm
    p_raw = 2 * (1 - norm.cdf(abs(z)))
    p_adj = min(p_raw * 10, 1.0)
    sig = '***' if p_adj < 0.001 else ('**' if p_adj < 0.01 else ('*' if p_adj < 0.05 else 'ns'))
    print(f'  {t1}-{t2}: z={z:+.3f}, p_raw={p_raw:.4f}, p_adj={p_adj:.4f} {sig}')

# 各维度分别检验
print(f'\n各维度差异检验:')
for col in ['信度', '效度', '公平性_raw', '区分力']:
    groups_d = [q3df[q3df['题目'] == t][col].dropna().values for t in ['A','B','C','D','E']]
    H_d, p_d = kruskal(*groups_d)
    print(f'  {col}: H={H_d:.3f}, p={p_d:.4f}')

print(f'\n===== 各题均值对比 =====')
for topic in ['A','B','C','D','E']:
    sub = q3df[q3df['题目'] == topic]
    print(f'{topic}(n={len(sub)}): TOPSIS={sub["TOPSIS得分"].mean():.3f}±{sub["TOPSIS得分"].std():.3f}, '
          f'信度={sub["信度"].mean():.3f}, 效度={sub["效度"].mean():.3f}, '
          f'公平性={sub["公平性_raw"].mean():.3f}, 区分力={sub["区分力"].mean():.1f}')

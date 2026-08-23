"""验证 α=0 时 ideal_rank 的 Spearman ρ 应该接近 1"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import os

PROBLEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROBLEM_DIR, 'data')

df = pd.read_pickle(os.path.join(DATA_DIR, 'combined-clean.pkl'))
award_map = {'一等奖': 3, '二等奖': 2, '三等奖': 1}

judge_stats = {}
for j in range(1, 5):
    for _, row in df.iterrows():
        jid = row[f'评委{j}']
        score = row[f'打分{j}']
        if jid not in judge_stats:
            judge_stats[jid] = []
        judge_stats[jid].append(score)
judge_mu_sigma = {jid: (np.mean(scores), np.std(scores)) for jid, scores in judge_stats.items()}

records = []
for topic in ['A','B','C','D','E']:
    tdata = df[df['题目'] == topic]
    for idx, row in tdata.iterrows():
        if pd.isna(row['成绩']):
            continue
        z_scores = []
        for j in range(1, 5):
            jid = row[f'评委{j}']
            mu, sigma = judge_mu_sigma[jid]
            raw = row[f'打分{j}']
            z = (raw - mu) / sigma if sigma > 0 else 0
            z_scores.append(z)
        z_mean = np.mean(z_scores)
        records.append({'题目': topic, '阅卷号': row['阅卷号'],
                       'z_mean': z_mean, '奖项数值': award_map[row['成绩']]})
q5df = pd.DataFrame(records)

# α=0: ideal_rank = rank(奖项数值) within each topic
ideal_rank = q5df.groupby('题目')['奖项数值'].transform(lambda x: x.rank())

print('=== α=0 时 Spearman ρ (ideal_rank vs 奖项数值) ===')
# 跨题整体（当前论文做法）
rho_all, _ = spearmanr(ideal_rank, q5df['奖项数值'])
print(f'跨题整体: ρ = {rho_all:.3f}')

print()
print('逐题内计算:')
for topic in ['A','B','C','D','E']:
    mask = q5df['题目'] == topic
    rho_t, _ = spearmanr(ideal_rank[mask], q5df.loc[mask, '奖项数值'])
    n = mask.sum()
    print(f'  {topic}题内 (n={n}): ρ = {rho_t:.6f}')

print()
y_vals = q5df['奖项数值']
print(f'奖项数值 unique: {sorted(y_vals.unique())}')
vc = y_vals.value_counts().sort_index()
print(f'奖项数值分布: 1={vc.get(1,0)}, 2={vc.get(2,0)}, 3={vc.get(3,0)}')

# 验证: 跨题稀释的机制
print()
print('=== 跨题稀释机制 ===')
for topic in ['A','B','C','D','E']:
    mask = q5df['题目'] == topic
    r = ideal_rank[mask]
    print(f'{topic}题 ideal_rank 范围: [{r.min():.1f}, {r.max():.1f}], '
          f'3等奖rank≈{r[q5df.loc[mask,"奖项数值"]==3].mean():.1f}, '
          f'1等奖rank≈{r[q5df.loc[mask,"奖项数值"]==1].mean():.1f}')

# 正确做法: 按题内 ρ 的样本量加权平均
print()
print('=== 正确做法: 题内ρ的样本量加权平均 ===')
weighted_rho = 0
total_n = 0
for topic in ['A','B','C','D','E']:
    mask = q5df['题目'] == topic
    n = mask.sum()
    rho_t, _ = spearmanr(ideal_rank[mask], q5df.loc[mask, '奖项数值'])
    weighted_rho += n * rho_t
    total_n += n
weighted_rho /= total_n
print(f'加权平均 ρ = {weighted_rho:.6f}')
print(f'当前论文跨题 ρ = {rho_all:.3f}')
print(f'差异 = {weighted_rho - rho_all:.3f}')

"""
阶段 2.1: Q5 网评利用策略
- 权重敏感性模拟
- 评委素质加权
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os, warnings
warnings.filterwarnings('ignore')

PROBLEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROBLEM_DIR, 'data')

df = pd.read_pickle(os.path.join(DATA_DIR, 'combined-clean.pkl'))
q3df = pd.read_pickle(os.path.join(DATA_DIR, 'q3-judge-scores.pkl'))
award_map = {'一等奖': 3, '二等奖': 2, '三等奖': 1}

# 建立评委ID -> (均值, 标准差) 映射 (同Q1修复)
judge_stats = {}
for j in range(1, 5):
    for _, row in df.iterrows():
        jid = row[f'评委{j}']
        score = row[f'打分{j}']
        if jid not in judge_stats:
            judge_stats[jid] = []
        judge_stats[jid].append(score)
judge_mu_sigma = {jid: (np.mean(scores), np.std(scores)) for jid, scores in judge_stats.items()}

# ===== 1. 权重敏感性模拟 =====
# 假设集中评审评分不可见, 用最终奖项反向推断
print('===== 权重敏感性 =====')
# 对入围论文: 按评委z-score计算网评标准分均值
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

# 扫α — 逐题计算后样本量加权平均（避免题内排名跨题不可比导致ρ被压低）
print(f'{"α":>6}  {"A":>8}  {"B":>8}  {"C":>8}  {"D":>8}  {"E":>8}  {"加权均值":>10}')
alphas = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
alpha_results = {}
for alpha in alphas:
    ideal_rank = q5df.groupby('题目')['奖项数值'].transform(lambda x: x.rank())
    web_rank = q5df.groupby('题目')['z_mean'].transform(lambda x: x.rank())
    final_rank = alpha * web_rank + (1-alpha) * ideal_rank
    rho_by_topic = {}
    total_n = 0
    weighted_sum = 0
    for topic in ['A','B','C','D','E']:
        mask = q5df['题目'] == topic
        n_t = mask.sum()
        rho_t, _ = spearmanr(final_rank[mask], q5df.loc[mask, '奖项数值'])
        rho_by_topic[topic] = rho_t
        weighted_sum += n_t * rho_t
        total_n += n_t
    rho_weighted = weighted_sum / total_n
    alpha_results[alpha] = {'by_topic': rho_by_topic, 'weighted': rho_weighted}
    rhos = '  '.join([f'{rho_by_topic[t]:>8.3f}' for t in ['A','B','C','D','E']])
    print(f'{alpha:>6.2f}  {rhos}  {rho_weighted:>10.3f}')

# ===== 2. 评委素质加权 =====
print(f'\n===== 评委素质加权效果 =====')
# 对每篇论文: 等权z_mean vs 加权z_mean
q5df_w = df[df['成绩'].notna()].copy()
q5df_w['奖项数值'] = q5df_w['成绩'].map(award_map)

for topic in ['A','B','C','D','E']:
    topic_q3 = q3df[q3df['题目'] == topic].set_index('评委ID')
    tdata = q5df_w[q5df_w['题目'] == topic].copy()
    
    # 标准分 — 按评委ID做z-score
    z_cols = {}
    for j in range(1, 5):
        z_cols[f'z{j}'] = np.zeros(len(tdata))
    for i, (idx, row) in enumerate(tdata.iterrows()):
        for j in range(1, 5):
            jid = row[f'评委{j}']
            mu, sigma = judge_mu_sigma[jid]
            raw = row[f'打分{j}']
            z_cols[f'z{j}'][i] = (raw - mu) / sigma if sigma > 0 else 0
    for j in range(1, 5):
        tdata[f'z{j}'] = z_cols[f'z{j}']
    
    # 等权均值
    tdata['z_eq'] = tdata[[f'z{j}' for j in range(1,5)]].mean(axis=1)
    
    # 素质加权: 每篇论文根据4位评委的TOPSIS得分赋权
    z_weighted = np.zeros(len(tdata))
    for i, (idx, row) in enumerate(tdata.iterrows()):
        ws = []
        for j in range(1,5):
            jid = row[f'评委{j}']
            s = topic_q3.loc[jid, 'TOPSIS得分'] if jid in topic_q3.index else 0.5
            ws.append(max(s, 0.01))  # 避免零权重
        ws = np.array(ws) / sum(ws)
        z_weighted[i] = sum(ws[j] * row[f'z{j+1}'] for j in range(4))
    tdata['z_weighted'] = z_weighted
    
    rho_eq, _ = spearmanr(tdata['z_eq'], tdata['奖项数值'])
    rho_w, _ = spearmanr(tdata['z_weighted'], tdata['奖项数值'])
    print(f'  {topic}: 等权ρ={rho_eq:.3f}, 加权ρ={rho_w:.3f}, Δ={rho_w-rho_eq:+.3f}')

# ===== 3. 利弊总结 =====
print(f'\n===== 利弊分析 =====')
print('利:')
print(f'  - Q1证实网评Spearman ρ=0.797（强相关）')
print(f'  - 筛选命中率74.8%, 假阴性率仅3.3%')
print(f'  - ROC AUC=0.932（对一等奖区分力卓越）')
print('弊:')
# 统计需关注/待改进评委占比
total_judges = len(q3df)
low_judges = q3df[q3df['分层'].isin(['需关注','待改进'])]
pct_low = len(low_judges) / total_judges * 100
print(f'  - Q3发现{pct_low:.1f}%评委需关注/待改进')
print('  - 等权均值让低素质评委噪声混入总成绩')
# C题效度
c_validity = q3df[q3df['题目']=='C']['效度'].mean()
print(f'  - C题效度仅{c_validity:.2f}, 网评在此题信息增量有限')

# ===== 4. 建议阈值 =====
print(f'\n===== 改进建议 =====')
# 多少评委在"需关注"以下？
for topic in ['A','B','C','D','E']:
    sub = q3df[q3df['题目'] == topic]
    low = sub[sub['分层'].isin(['需关注','待改进'])]
    print(f'  {topic}: 建议降权的评委({len(low)}位): {", ".join(low["评委ID"].values)}')

# ===== 5. 加权后的权重敏感性 (Fix 1: α 交叉验证) =====
print(f'\n===== 加权后权重敏感性 (Fix 1) =====')
# 独立计算每篇入围论文的素质加权 z_mean（与段2逻辑一致）
weighted_records = []
for topic in ['A','B','C','D','E']:
    topic_q3 = q3df[q3df['题目'] == topic].set_index('评委ID')
    tdata = q5df_w[q5df_w['题目'] == topic]
    for idx, row in tdata.iterrows():
        ws = []
        zs = []
        for j in range(1, 5):
            jid = row[f'评委{j}']
            mu, sigma = judge_mu_sigma[jid]
            raw = row[f'打分{j}']
            z = (raw - mu) / sigma if sigma > 0 else 0
            zs.append(z)
            s = topic_q3.loc[jid, 'TOPSIS得分'] if jid in topic_q3.index else 0.5
            ws.append(max(s, 0.01))
        ws = np.array(ws) / sum(ws)
        z_w = sum(ws[j] * zs[j] for j in range(4))
        weighted_records.append({
            '题目': topic, '阅卷号': row['阅卷号'],
            'z_weighted': z_w, '奖项数值': row['奖项数值']
        })
q5df_w_all = pd.DataFrame(weighted_records)

print(f'{"α":>6}  {"ρ_等权":>8}  {"ρ_加权":>8}  {"Δ":>8}')
for alpha in [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
    # 等权版本: 复用段1的逐题加权均值
    rho_eq = alpha_results[alpha]['weighted']

    # 加权版本: 同样逐题计算后加权平均
    ideal_rank_w = q5df_w_all.groupby('题目')['奖项数值'].transform(lambda x: x.rank())
    web_rank_w = q5df_w_all.groupby('题目')['z_weighted'].transform(lambda x: x.rank())
    final_rank_w = alpha * web_rank_w + (1-alpha) * ideal_rank_w
    w_total_n = 0
    w_weighted_sum = 0
    for topic in ['A','B','C','D','E']:
        mask = q5df_w_all['题目'] == topic
        n_t = mask.sum()
        rho_t, _ = spearmanr(final_rank_w[mask], q5df_w_all.loc[mask, '奖项数值'])
        w_weighted_sum += n_t * rho_t
        w_total_n += n_t
    rho_w = w_weighted_sum / w_total_n

    print(f'{alpha:>6.2f}  {rho_eq:>8.3f}  {rho_w:>8.3f}  {rho_w-rho_eq:>+8.3f}')

# ===== 6. 两层次加权：评委级折扣 × 论文内归一化 (Fix 2) =====
print(f'\n===== 两层次加权 (评委级折扣 + 论文内归一化) (Fix 2) =====')

# 构建低素质评委折扣因子
low_judges = q3df[q3df['分层'].isin(['需关注', '待改进'])]
judge_discount = {}
for _, row in low_judges.iterrows():
    jid = row['评委ID']
    topic = row['题目']
    judge_discount[(jid, topic)] = max(row['TOPSIS得分'], 0.01)
# 正常评委折扣 = 1.0 (未在 dict 中)

for topic in ['A','B','C','D','E']:
    topic_q3 = q3df[q3df['题目'] == topic].set_index('评委ID')
    tdata = q5df_w[q5df_w['题目'] == topic].copy()

    # z-scores
    for j in range(1, 5):
        tdata[f'z{j}'] = np.zeros(len(tdata))
    for i, (idx, row) in enumerate(tdata.iterrows()):
        for j in range(1, 5):
            jid = row[f'评委{j}']
            mu, sigma = judge_mu_sigma[jid]
            raw = row[f'打分{j}']
            tdata.loc[idx, f'z{j}'] = (raw - mu) / sigma if sigma > 0 else 0

    # 等权
    tdata['z_eq'] = tdata[[f'z{j}' for j in range(1,5)]].mean(axis=1)

    # 纯论文内加权 (与段2逻辑一致, TOPSIS归一化)
    z_paper_weighted = np.zeros(len(tdata))
    for i, (idx, row) in enumerate(tdata.iterrows()):
        ws = []
        for j in range(1, 5):
            jid = row[f'评委{j}']
            s = topic_q3.loc[jid, 'TOPSIS得分'] if jid in topic_q3.index else 0.5
            ws.append(max(s, 0.01))
        ws = np.array(ws) / sum(ws)
        z_paper_weighted[i] = sum(ws[j] * row[f'z{j+1}'] for j in range(4))
    tdata['z_paper_weighted'] = z_paper_weighted

    # 两层次加权: 评委级折扣 × 论文内归一化
    z_two_level = np.zeros(len(tdata))
    for i, (idx, row) in enumerate(tdata.iterrows()):
        ws = []
        for j in range(1, 5):
            jid = row[f'评委{j}']
            # 评委级折扣 (低素质评委折扣=TOPSIS, 正常=1.0)
            discount = judge_discount.get((jid, topic), 1.0)
            # 论文内权重因子 (TOPSIS得分)
            s = topic_q3.loc[jid, 'TOPSIS得分'] if jid in topic_q3.index else 0.5
            w = discount * max(s, 0.01)
            ws.append(w)
        ws = np.array(ws) / sum(ws)
        z_two_level[i] = sum(ws[j] * row[f'z{j+1}'] for j in range(4))
    tdata['z_two_level'] = z_two_level

    rho_eq, _ = spearmanr(tdata['z_eq'], tdata['奖项数值'])
    rho_pw, _ = spearmanr(tdata['z_paper_weighted'], tdata['奖项数值'])
    rho_2l, _ = spearmanr(tdata['z_two_level'], tdata['奖项数值'])
    print(f'  {topic}: 等权ρ={rho_eq:.3f}, 论文内加权ρ={rho_pw:.3f}, '
          f'两层加权ρ={rho_2l:.3f} (Δ两层vs论文内={rho_2l-rho_pw:+.3f})')

# ===== 7. 多指标验证: 命中率 + FNR (Fix 3) =====
print(f'\n===== 加权后的筛选有效性 (Fix 3) =====')

# 对全样本重新计算加权 z_mean (含未入围论文)
all_z_weighted = []
for idx, row in df.iterrows():
    topic = row['题目']
    topic_q3 = q3df[q3df['题目'] == topic].set_index('评委ID')
    ws = []
    zs = []
    for j in range(1, 5):
        jid = row[f'评委{j}']
        mu, sigma = judge_mu_sigma[jid]
        raw = row[f'打分{j}']
        z = (raw - mu) / sigma if sigma > 0 else 0
        zs.append(z)
        s = topic_q3.loc[jid, 'TOPSIS得分'] if jid in topic_q3.index else 0.5
        ws.append(max(s, 0.01))
    ws = np.array(ws) / sum(ws)
    z_w = sum(ws[j] * zs[j] for j in range(4))
    all_z_weighted.append(z_w)

df['z_weighted_all'] = all_z_weighted

for topic in ['A','B','C','D','E','全题']:
    sub = df if topic == '全题' else df[df['题目'] == topic]

    # 等权版本
    z_eq_list = []
    for idx, row in sub.iterrows():
        zs = []
        for j in range(1, 5):
            jid = row[f'评委{j}']
            mu, sigma = judge_mu_sigma[jid]
            raw = row[f'打分{j}']
            zs.append((raw - mu) / sigma if sigma > 0 else 0)
        z_eq_list.append(np.mean(zs))
    sub = sub.copy()
    sub['z_eq'] = z_eq_list
    sub_eq = sub.sort_values('z_eq', ascending=False)
    n = len(sub_eq)
    cutoff = int(n * 0.55)
    top_eq = sub_eq.head(cutoff)
    bot_eq = sub_eq.tail(n - cutoff)
    hit_eq = top_eq['成绩'].notna().sum() / len(top_eq)
    total_award = sub_eq['成绩'].notna().sum()
    fnr_eq = bot_eq['成绩'].notna().sum() / total_award if total_award > 0 else 0

    # 加权版本
    sub_w = sub.sort_values('z_weighted_all', ascending=False)
    top_w = sub_w.head(cutoff)
    bot_w = sub_w.tail(n - cutoff)
    hit_w = top_w['成绩'].notna().sum() / len(top_w)
    total_award_w = sub_w['成绩'].notna().sum()
    fnr_w = bot_w['成绩'].notna().sum() / total_award_w if total_award_w > 0 else 0

    print(f'  {topic}: 命中率 等权{hit_eq:.2%}→加权{hit_w:.2%} '
          f'| FNR 等权{fnr_eq:.2%}→加权{fnr_w:.2%}')

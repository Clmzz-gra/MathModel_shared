"""
全文数值一致性校验
=====================
从模型脚本重算关键数值，与论文 main.tex 中的报告值对比。
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, kruskal, skew, kurtosis
from sklearn.metrics import roc_auc_score
import os, re, warnings
warnings.filterwarnings('ignore')

PROBLEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROBLEM_DIR, 'data')
PAPER_PATH = os.path.join(PROBLEM_DIR, '..', 'solution', 'paper', 'main.tex')

# ===================== 数据加载 =====================
df = pd.read_pickle(os.path.join(DATA_DIR, 'combined-clean.pkl'))
q3df = pd.read_pickle(os.path.join(DATA_DIR, 'q3-judge-scores.pkl'))
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

# ===================== 提取论文中的数值 =====================
with open(PAPER_PATH, 'r', encoding='utf-8') as f:
    tex = f.read()

def extract_number(pattern, text, group=1):
    """从文本中提取匹配正则的数值"""
    m = re.search(pattern, text)
    return float(m.group(group)) if m else None

print('=' * 60)
print('全文数值一致性校验')
print('=' * 60)

errors = []

# ===================== Q1 校验 =====================
print('\n--- Q1: 相关性分析 ---')

# 计算网评标准分
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

# Spearman ρ
for topic in ['A','B','C','D','E','全题']:
    sub = q1df if topic == '全题' else q1df[q1df['题目'] == topic]
    rho_calc, _ = spearmanr(sub['网评标准分均值'], sub['奖项数值'])
    rho_paper = extract_number(rf'{topic}[^0-9]*?Spearman.*?\\rho.*?(\d+\.\d+)', tex) if topic != '全题' else extract_number(r'全题.*?Spearman.*?\\rho.*?(\d+\.\d+)', tex)
    if rho_paper is None:
        rho_paper = extract_number(r'\\textbf\{全题\}.*?(\d+\.\d+)', tex)
    if rho_paper is None:
        # Try to find from the table context
        pass
    if rho_calc != 0:
        print(f'  {topic} Spearman ρ: 计算={rho_calc:.3f}')

# Pearson r (入围论文)
print('\n  Pearson r (入围论文):')
for topic in ['A','B','C','D','E','全题']:
    sub = q1df if topic == '全题' else q1df[q1df['题目'] == topic]
    sub_in = sub[sub['是否获奖']]
    if len(sub_in) > 30:
        sk = skew(sub_in['网评标准分均值'])
        ku = kurtosis(sub_in['网评标准分均值'])
        if abs(sk) < 1 and abs(ku) < 3:
            r_calc, _ = pearsonr(sub_in['网评标准分均值'], sub_in['奖项数值'])
            print(f'  {topic}: r={r_calc:.3f} (偏度={sk:.2f}, 峰度={ku:.2f})')
        else:
            print(f'  {topic}: 未通过正态性检验 (偏度={sk:.2f}, 峰度={ku:.2f}) — 论文不报告')
    else:
        print(f'  {topic}: 样本不足 ({len(sub_in)})')

# 命中率/FNR
print('\n  筛选有效性:')
for topic in ['A','B','C','D','E','全题']:
    sub = q1df if topic == '全题' else q1df[q1df['题目'] == topic]
    sub = sub.sort_values('网评标准分均值', ascending=False)
    n_total = len(sub)
    cutoff = int(n_total * 0.55)
    top = sub.head(cutoff)
    bottom = sub.tail(n_total - cutoff)
    hit_rate = top['是否获奖'].sum() / len(top)
    total_award = sub['是否获奖'].sum()
    false_neg = bottom['是否获奖'].sum() / total_award if total_award > 0 else 0
    print(f'  {topic}: 命中率={hit_rate:.2%}, FNR={false_neg:.2%}, '
          f'n={n_total}, cutoff={cutoff}, 获奖总数={total_award}')

# ROC AUC
print('\n  ROC AUC (预测一等奖):')
for topic in ['A','B','C','D','E','全题']:
    sub = q1df if topic == '全题' else q1df[q1df['题目'] == topic]
    if sub['是否一等'].sum() > 5:
        auc = roc_auc_score(sub['是否一等'], sub['网评标准分均值'])
        print(f'  {topic}: AUC={auc:.3f}')

# 验证论文报告的命中率 74.8% 和 FNR 3.3%
sub_all = q1df.sort_values('网评标准分均值', ascending=False)
n_total = len(sub_all)
cutoff = int(n_total * 0.55)
top_all = sub_all.head(cutoff)
bottom_all = sub_all.tail(n_total - cutoff)
all_hit = top_all['是否获奖'].sum() / len(top_all)
all_award = sub_all['是否获奖'].sum()
all_fnr = bottom_all['是否获奖'].sum() / all_award
print(f'\n  全题汇总: 命中率={all_hit:.2%}, FNR={all_fnr:.2%}')
print(f'  论文报告: 命中率=74.8%, FNR=3.3%')
if abs(all_hit - 0.748) > 0.001:
    print(f'  *** 命中率不一致! ***')
if abs(all_fnr - 0.033) > 0.001:
    print(f'  *** FNR 不一致! ***')

# ===================== Q2/Q3 校验 =====================
print('\n--- Q2/Q3: 指标与综合评价 ---')

# 四维度独立性
from scipy.stats import spearmanr as spr
dims = ['信度', '效度', '公平性_raw', '区分力']
max_r = 0
fairness_rs = []
for i in range(4):
    for j in range(i+1, 4):
        r_d, _ = spr(q3df[dims[i]], q3df[dims[j]])
        if abs(r_d) > abs(max_r):
            max_r = r_d
        if dims[i] == '公平性_raw' or dims[j] == '公平性_raw':
            fairness_rs.append(abs(r_d))
print(f'  四维度最大交叉 r = {max_r:.3f} (论文报告 0.32)')
print(f'  公平性与其他维度最大 |r| = {max(fairness_rs):.3f} (论文报告 <0.12)')

# 熵权
from collections import defaultdict
entropy_weights = defaultdict(dict)
for topic in ['A','B','C','D','E']:
    sub = q3df[q3df['题目'] == topic]
    z = sub[['信度','效度','公平性','区分力']].values
    # Min-Max normalize
    z_norm = (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0) + 1e-10)
    # entropy
    K = len(z_norm)
    p = z_norm / z_norm.sum(axis=0)
    p = np.clip(p, 1e-10, 1)
    e = -np.sum(p * np.log(p), axis=0) / np.log(K)
    d = 1 - e
    w = d / d.sum()
    for j, dim in enumerate(['信度','效度','公平性','区分力']):
        entropy_weights[topic][dim] = w[j]

print('\n  熵权分布:')
for topic in ['A','B','C','D','E']:
    ws = entropy_weights[topic]
    print(f'  {topic}: 信度={ws["信度"]:.3f}, 效度={ws["效度"]:.3f}, '
          f'公平性={ws["公平性"]:.3f}, 区分力={ws["区分力"]:.3f}')

# 区分力权重均值
disc_mean = np.mean([entropy_weights[t]['区分力'] for t in ['A','B','C','D','E']])
print(f'  区分力权重均值 = {disc_mean:.3f} (论文报告 0.386)')

# TOPSIS 得分均值
print('\n  TOPSIS 得分均值:')
for topic in ['A','B','C','D','E']:
    sub = q3df[q3df['题目'] == topic]
    mean_s = sub['TOPSIS得分'].mean()
    std_s = sub['TOPSIS得分'].std()
    print(f'  {topic}: TOPSIS均值={mean_s:.3f}, 标准差={std_s:.3f}')

# ===================== Q4 校验 =====================
print('\n--- Q4: 差异分析 ---')
groups = [q3df[q3df['题目'] == t]['TOPSIS得分'].values for t in ['A','B','C','D','E']]
H, p_kw = kruskal(*groups)
N = len(q3df)
k = 5
eta2 = (H - k + 1) / (N - k)
print(f'  KW: H={H:.3f}, p={p_kw:.4f}, η²={eta2:.3f}')
print(f'  论文报告: H=8.450, p=0.076, η²=0.023')

# 子维度 KW
for col in ['信度', '效度', '公平性_raw', '区分力']:
    groups_d = [q3df[q3df['题目'] == t][col].dropna().values for t in ['A','B','C','D','E']]
    H_d, p_d = kruskal(*groups_d)
    print(f'  {col}: H={H_d:.3f}, p={p_d:.4f}')

# ===================== Q5 校验 =====================
print('\n--- Q5: 权重敏感性 ---')
# α sweep
q5df_in = df[df['成绩'].notna()].copy()
q5df_in['奖项数值'] = q5df_in['成绩'].map(award_map)
q5_records = []
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
        q5_records.append({'题目': topic, '阅卷号': row['阅卷号'],
                          'z_mean': np.mean(z_scores), '奖项数值': award_map[row['成绩']]})
q5df_s = pd.DataFrame(q5_records)

print('  α扫描:')
for alpha in [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
    ideal_rank = q5df_s.groupby('题目')['奖项数值'].transform(lambda x: x.rank())
    web_rank = q5df_s.groupby('题目')['z_mean'].transform(lambda x: x.rank())
    final_rank = alpha * web_rank + (1-alpha) * ideal_rank
    rho, _ = spearmanr(final_rank, q5df_s['奖项数值'])
    print(f'  α={alpha:.2f}: ρ={rho:.3f}')

# 素质加权
print('\n  素质加权效果 (入围论文内部):')
for topic in ['A','B','C','D','E']:
    topic_q3 = q3df[q3df['题目'] == topic].set_index('评委ID')
    tdata = q5df_in[q5df_in['题目'] == topic].copy()
    # z-scores
    for j in range(1, 5):
        tdata[f'z{j}'] = np.zeros(len(tdata))
    for i, (idx, row) in enumerate(tdata.iterrows()):
        for j in range(1, 5):
            jid = row[f'评委{j}']
            mu, sigma = judge_mu_sigma[jid]
            raw = row[f'打分{j}']
            tdata.loc[idx, f'z{j}'] = (raw - mu) / sigma if sigma > 0 else 0
    tdata['z_eq'] = tdata[[f'z{j}' for j in range(1,5)]].mean(axis=1)
    z_w = np.zeros(len(tdata))
    for i, (idx, row) in enumerate(tdata.iterrows()):
        ws = []
        for j in range(1,5):
            jid = row[f'评委{j}']
            s = topic_q3.loc[jid, 'TOPSIS得分'] if jid in topic_q3.index else 0.5
            ws.append(max(s, 0.01))
        ws = np.array(ws) / sum(ws)
        z_w[i] = sum(ws[j] * row[f'z{j+1}'] for j in range(4))
    rho_eq, _ = spearmanr(tdata['z_eq'], tdata['奖项数值'])
    rho_w, _ = spearmanr(z_w, tdata['奖项数值'])
    print(f'  {topic}: 等权ρ={rho_eq:.3f}, 加权ρ={rho_w:.3f}, Δ={rho_w-rho_eq:+.3f}')

# ===================== 综合判断 =====================
print('\n' + '=' * 60)
print('校验完成。请对照上述打印值与论文报告值逐项核实。')
print('注：因浮点精度和实现细节差异，允许 ±0.001 的浮动。')

"""
疑点5分析: 逐题检查区分力最高的评委，其公平性是否系统性地低?
关键问题: "区分力高→公平性低"是普适规律还是C题独有的现象?
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

df = pd.read_csv(r'e:\MathModel-school-competition\problems\2026第一次模拟赛赛题\C题 关于某竞赛网评结果的建模与分析\solution\artifacts\tables\q3-judge-scores.csv')

print('='*80)
print('疑点5分析: 逐题检查区分力最高评委的公平性表现')
print('='*80)

for topic in ['A','B','C','D','E']:
    sub = df[df['题目'] == topic].copy()
    top3 = sub.nlargest(3, '区分力')
    r, p = spearmanr(sub['区分力'], sub['公平性_raw'])
    n = len(sub)
    
    print(f'\n--- {topic}题 (评委数={n}) ---')
    print(f'  区分力 vs 公平性_raw Spearman = {r:.3f} (p={p:.3f})')
    print(f'  Top3区分力评委:')
    for _, row in top3.iterrows():
        fair_rank = int(row['公平性_排名'])
        disc_rank = int(row['区分力_排名'])
        print(f'    {row["评委ID"]}: 区分力={row["区分力"]:.1f}(#{disc_rank}/{n}), '
              f'fairness_raw={row["公平性_raw"]:.3f}(#{fair_rank}/{n}), '
              f'avg_score={row["评分均值"]:.1f}, layer={row["分层"]}')
    
    top5 = sub.nlargest(5, '区分力')
    all_fair = sub['公平性_raw'].mean()
    top5_fair = top5['公平性_raw'].mean()
    print(f'  Top5区分力 fairness_raw均值={top5_fair:.3f} vs 全题均值={all_fair:.3f}')

print('\n' + '='*80)
print('跨题目: 区分力No.1评委在各自题目内的公平性排名')
print('='*80)
for topic in ['A','B','C','D','E']:
    sub = df[df['题目'] == topic]
    n = len(sub)
    top1 = sub.nlargest(1, '区分力').iloc[0]
    fair_rank = int(top1['公平性_排名'])
    mean_score = sub['评分均值'].mean()
    direction = 'below_mean(strict)' if top1['评分均值'] < mean_score else 'above_mean(lenient)'
    print(f'  {topic}: #{1}disc={top1["评委ID"]}, fairness=#{fair_rank}/{n}, '
          f'raw={top1["公平性_raw"]:.3f}, bias={top1["评分均值"]:.1f} vs topic_mean={mean_score:.1f} ({direction})')

print('\n' + '='*80)
print('关键对比: C题 vs D题 区分力Top3的公平性细节')
print('='*80)
for topic in ['C', 'D']:
    sub = df[df['题目'] == topic]
    t_mean = sub['评分均值'].mean()
    print(f'\n{topic}题全题评分均值: {t_mean:.1f}')
    top3 = sub.nlargest(3, '区分力')
    for _, row in top3.iterrows():
        bias = row['评分均值'] - t_mean
        print(f'  {row["评委ID"]}: disc={row["区分力"]:.1f}, avg={row["评分均值"]:.1f} (bias={bias:+.1f}), '
              f'fair_raw={row["公平性_raw"]:.3f}, layer={row["分层"]}')

print('\n' + '='*80)
print('结论判断')
print('='*80)
print("""
C题区分力Top3评委:
  - C01 (disc=22.2, avg=37.4 vs topic_mean≈53.8, bias=-16.4)
  - C19 (disc=19.2, avg=61.5 vs topic_mean≈53.8, bias=+7.7)
  - C10 (disc=17.0, avg=54.3 vs topic_mean≈53.8, bias=+0.5)
  
  其中C01和C19偏离均值极大, 这是C题"优秀层公平性低"的直接原因.
  
  C题的Spearman r(区分力, fairness_raw) = ? 
""")

"""
阶段 0.3：基础数据清洗（模型无关）
从 combined-raw.pkl 加载 → 清洗 → 写入 combined-clean.pkl
"""
import pandas as pd
import numpy as np
import os

PROBLEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROBLEM_DIR, 'data')

# ===== 1. 加载缓存 =====
df = pd.read_pickle(os.path.join(DATA_DIR, 'combined-raw.pkl'))
print(f'加载数据: {len(df)} 行')

# ===== 2. 重复记录检测 =====
dup_keys = df.duplicated(subset=['阅卷号'], keep=False)
dup_rows = df.duplicated(keep=False)
print(f'\n===== 重复记录检测 =====')
print(f'阅卷号重复: {dup_keys.sum()}')
print(f'完全重复行: {dup_rows.sum()}')

# ===== 3. 零值与缺失值区分 =====
print(f'\n===== 零值与低分分析 =====')
score_cols = ['打分1', '打分2', '打分3', '打分4']
for col in score_cols:
    zeros = (df[col] == 0).sum()
    ones = (df[col] == 1).sum()
    if zeros > 0:
        print(f'{col}: 0分={zeros}条, 1分={ones}条')
# 全部评分
all_scores = pd.concat([df[c] for c in score_cols])
print(f'总评分: min={all_scores.min()}, P01={all_scores.quantile(0.01):.0f}, P99={all_scores.quantile(0.99):.0f}, max={all_scores.max()}')

# ===== 4. 异常值检测（不做自动删除） =====
print(f'\n===== 异常值检测 =====')
for topic in ['A','B','C','D','E']:
    sub = df[df['题目'] == topic]
    scores_t = pd.concat([sub[c] for c in score_cols])
    q1, q3 = scores_t.quantile(0.25), scores_t.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    outliers = scores_t[(scores_t < lower) | (scores_t > upper)]
    topic_papers = len(sub)
    outlier_papers = len(sub[(sub[score_cols] < lower).any(axis=1) | (sub[score_cols] > upper).any(axis=1)])
    print(f'{topic}: IQR边界=[{lower:.0f}, {upper:.0f}], 异常评分{len(outliers)}条({len(outliers)/len(scores_t)*100:.1f}%), 涉及{outlier_papers}篇论文')

# 按评委检测异常（评分方差过小 = 中心化趋势）
print(f'\n===== 评委评分方差检测 =====')
judge_stats = []
for topic in ['A','B','C','D','E']:
    sub = df[df['题目'] == topic]
    for j in range(1,5):
        jc, sc = f'评委{j}', f'打分{j}'
        for jid in sub[jc].unique():
            jscores = sub[sub[jc] == jid][sc]
            judge_stats.append({'评委': jid, '题目': topic, '篇数': len(jscores), '均值': jscores.mean(), '标准差': jscores.std()})
jdf = pd.DataFrame(judge_stats)
# 标准差过小（<8，正常约16-17）
low_var = jdf[jdf['标准差'] < 8]
if len(low_var) > 0:
    print(f'标准差<8的评委数: {len(low_var)} (疑似中心化趋势)')
    for _, r in low_var.iterrows():
        print(f'  {r["评委"]}({r["题目"]}): 篇数={r["篇数"]}, 均值={r["均值"]:.1f}, 标准差={r["标准差"]:.2f}')

# 标准差异常大
high_var = jdf[jdf['标准差'] > 25]
if len(high_var) > 0:
    print(f'标准差>25的评委数: {len(high_var)} (评分极分散)')
    for _, r in high_var.iterrows():
        print(f'  {r["评委"]}({r["题目"]}): 篇数={r["篇数"]}, 均值={r["均值"]:.1f}, 标准差={r["标准差"]:.2f}')

# ===== 5. 类型标准化 =====
print(f'\n===== 类型标准化 =====')
# 成绩列 → category
before = df['成绩'].dtype
df['成绩'] = df['成绩'].astype('category')
print(f'成绩列: {before} → category, 类别: {list(df["成绩"].cat.categories)}')

# 题目列 → category
df['题目'] = df['题目'].astype('category')
print(f'题目列: category, 类别: {list(df["题目"].cat.categories)}')

# 评分列确保int
for col in score_cols:
    df[col] = df[col].astype(int)

# ===== 6. 候选池覆盖率 =====
print(f'\n===== 候选池覆盖率 =====')
# 检查阅卷量异常的评委
low_count = jdf[jdf['篇数'] < 30]
print(f'阅卷量<30篇的评委: {len(low_count)} 位')
for _, r in low_count.iterrows():
    print(f'  {r["评委"]}({r["题目"]}): {r["篇数"]}篇')

# 总可分析样本
total_judges = len(jdf)
usable = len(jdf[jdf['篇数'] >= 30])
print(f'\n可用评委: {usable}/{total_judges} ({usable/total_judges*100:.1f}%)')
print(f'阈值: 阅卷量≥30篇 (保证标准分σ估计基本稳定)')

# ===== 7. 保存清洗后数据 =====
df.to_pickle(os.path.join(DATA_DIR, 'combined-clean.pkl'))
print(f'\n清洗后数据 → combined-clean.pkl ({len(df)} 行)')

# ===== 8. 评委统计保存 =====
jdf.to_pickle(os.path.join(DATA_DIR, 'judge-stats.pkl'))
print(f'评委统计 → judge-stats.pkl ({len(jdf)} 行)')

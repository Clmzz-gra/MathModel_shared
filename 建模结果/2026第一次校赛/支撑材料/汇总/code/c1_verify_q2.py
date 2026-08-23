import pandas as pd, numpy as np
from scipy.stats import spearmanr

PROBLEM_DIR = r'e:\MathModel-school-competition\problems\2026第一次模拟赛赛题\C题 关于某竞赛网评结果的建模与分析'
df = pd.read_pickle(PROBLEM_DIR + r'\outputs\data\judge-profile.pkl')

# 1. 区分度 vs 准确性的独立性
r1, _ = spearmanr(df['评分标准差'], df['与奖项相关性'].fillna(0))
r2, _ = spearmanr(df['评分标准差'], df['获奖率'])
r3, _ = spearmanr(df['变异系数'], df['与奖项相关性'].fillna(0))
print(f'=== 区分度 vs 准确性独立性 ===')
print(f'评分标准差 vs 与奖项相关性: Spearman r={r1:.3f}')
print(f'评分标准差 vs 获奖率: Spearman r={r2:.3f}')
print(f'变异系数 vs 与奖项相关性: Spearman r={r3:.3f}')

# 2. 公平性 vs 区分力独立性
bias = df['偏差(bias)']
r4, _ = spearmanr(abs(bias), df['评分标准差'])
r5, _ = spearmanr(bias, df['与奖项相关性'].fillna(0))
r6, _ = spearmanr(abs(bias), df['获奖率'])
r7, _ = spearmanr(df['评分标准差'], bias)
print(f'\n=== 公平性 vs 其他维度独立性 ===')
print(f'|偏差| vs 评分标准差: Spearman r={r4:.3f}')
print(f'偏差 vs 与奖项相关性: Spearman r={r5:.3f}')
print(f'|偏差| vs 获奖率: Spearman r={r6:.3f}')
print(f'评分标准差 vs 偏差: Spearman r={r7:.3f}')

# 3. 获奖率 vs 评分水平
r8, _ = spearmanr(df['评分均值'], df['获奖率'])
print(f'\n=== 评分水平 vs 获奖率 ===')
print(f'评分均值 vs 获奖率: Spearman r={r8:.3f}')

# 4. 四维度完整独立性矩阵
dims = {
    '信度': df['评分标准差'],  # 代理：离散度越低越一致
    '效度': df['与奖项相关性'].fillna(0),
    '公平性': abs(bias),
    '区分力': df['评分标准差']  # 同信度的反面
}
print(f'\n=== 四维度 Spearman 相关矩阵 ===')
keys = list(dims.keys())
for i, k1 in enumerate(keys):
    for j, k2 in enumerate(keys):
        if i < j:
            r, _ = spearmanr(dims[k1], dims[k2])
            print(f'  {k1:5s} × {k2:5s}: r={r:.3f}')

"""
阶段 1.1-Q3 A类验证：熵权法 + TOPSIS 可行性
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

PROBLEM_DIR = r'e:\MathModel-school-competition\problems\2026第一次模拟赛赛题\C题 关于某竞赛网评结果的建模与分析'
df = pd.read_pickle(PROBLEM_DIR + r'\outputs\data\judge-profile.pkl')

# ===== 构建四维度指标 =====
# 信度：用"评分标准差"的反面做近似（仅在验证阶段，最终版本应计算真实ICC）
# 效度：与奖项的Spearman ρ
# 公平性：偏差的绝对值
# 区分力：评分标准差

indicators = pd.DataFrame({
    '信度_neg_std': -df['评分标准差'],  # 负标准差（越大越好=离散小）
    '效度': df['与奖项相关性'].fillna(0),
    '公平性': -abs(df['偏差(bias)']),   # 负偏差（偏差绝对值越小越好）
    '区分力': df['评分标准差'],         # 标准差（区分力越大越好）
    '题目': df['题目'].values,
}, index=df.index)

# ===== 各题独立熵权 =====
for topic in ['A','B','C','D','E']:
    sub = indicators[indicators['题目'] == topic].copy()
    feature_cols = ['信度_neg_std', '效度', '公平性', '区分力']
    X = sub[feature_cols].values
    n = X.shape[0]
    
    # Min-Max 正向归一化
    X_norm = np.zeros_like(X)
    for j in range(X.shape[1]):
        xmin, xmax = X[:,j].min(), X[:,j].max()
        if xmax != xmin:
            X_norm[:,j] = (X[:,j] - xmin) / (xmax - xmin)
        else:
            X_norm[:,j] = 0.5
    
    # 非负平移
    X_norm = X_norm + 1e-6
    
    # 概率
    P = X_norm / X_norm.sum(axis=0, keepdims=True)
    
    # 信息熵
    k = 1.0 / np.log(n)
    e = -k * np.sum(P * np.log(P), axis=0)
    
    # 熵权
    d = 1 - e
    w = d / d.sum()
    
    # TOPSIS
    Z_plus = np.max(X_norm, axis=0)
    Z_minus = np.min(X_norm, axis=0)
    W_norm = X_norm * w
    D_plus = np.sqrt(np.sum((W_norm - Z_plus*w)**2, axis=1))
    D_minus = np.sqrt(np.sum((W_norm - Z_minus*w)**2, axis=1))
    S = D_minus / (D_plus + D_minus)
    
    sub['TOPSIS得分'] = S
    indicators.loc[sub.index, 'TOPSIS得分'] = S
    
    print(f'{topic}题 ({n}位评委):')
    print(f'  熵权: 信度={w[0]:.3f}, 效度={w[1]:.3f}, 公平性={w[2]:.3f}, 区分力={w[3]:.3f}')
    print(f'  TOPSIS得分: mean={S.mean():.3f}, std={S.std():.3f}, min={S.min():.3f}, max={S.max():.3f}')
    
    # 得分与各维度的相关性
    for k, col in enumerate(feature_cols):
        r, _ = spearmanr(S, sub[col])
        print(f'  得分 vs {col}: r={r:.3f}')
    print()

# ===== 简单基线：等权TOPSIS =====
print('===== 等权TOPSIS（简单基线） =====')
for topic in ['A','B','C','D','E']:
    sub = indicators[indicators['题目'] == topic].copy()
    X = sub[feature_cols].values
    X_norm = np.zeros_like(X)
    for j in range(X.shape[1]):
        xmin, xmax = X[:,j].min(), X[:,j].max()
        X_norm[:,j] = (X[:,j] - xmin) / (xmax - xmin) if xmax != xmin else 0.5
    X_norm += 1e-6
    
    w = np.ones(4) / 4
    Z_plus = np.max(X_norm, axis=0)
    Z_minus = np.min(X_norm, axis=0)
    W_norm = X_norm * w
    D_plus = np.sqrt(np.sum((W_norm - Z_plus*w)**2, axis=1))
    D_minus = np.sqrt(np.sum((W_norm - Z_minus*w)**2, axis=1))
    S_eq = D_minus / (D_plus + D_minus)
    
    # 熵权排序 vs 等权排序的一致性
    S_ent = indicators.loc[sub.index, 'TOPSIS得分'].values
    r_rank, _ = spearmanr(S_ent, S_eq)
    print(f'{topic}: 熵权vs等权 Spearman r={r_rank:.3f}')

# ===== 验证结论 =====
print(f'\n===== 结论 =====')
all_ent = indicators['TOPSIS得分'].dropna()
all_weights = []
for topic in ['A','B','C','D','E']:
    sub = indicators[indicators['题目'] == topic]
    X = sub[feature_cols].values
    X_norm = np.zeros_like(X)
    for j in range(X.shape[1]):
        xmin, xmax = X[:,j].min(), X[:,j].max()
        X_norm[:,j] = (X[:,j] - xmin) / (xmax - xmin) if xmax != xmin else 0.5
    X_norm += 1e-6
    P = X_norm / X_norm.sum(axis=0, keepdims=True)
    k = 1.0 / np.log(len(sub))
    e = -k * np.sum(P * np.log(P), axis=0)
    d = 1 - e
    w = d / d.sum()
    all_weights.append(w)
avg_w = np.mean(all_weights, axis=0)
print(f'五题平均熵权: 信度={avg_w[0]:.3f}, 效度={avg_w[1]:.3f}, 公平性={avg_w[2]:.3f}, 区分力={avg_w[3]:.3f}')

#!/usr/bin/env python
"""输出 PCA 方差贡献表"""
import numpy as np, pandas as pd

df = pd.read_pickle('E:/MathModel/problems/2025/C题/outputs/data/2025C-male-clean.pkl')
ivf_dummies = pd.get_dummies(df['IVF妊娠'].fillna('自然受孕'), prefix='c').astype(float)
fnum = ['孕周_数值','孕妇BMI','年龄','Y染色体浓度','X染色体浓度','GC含量',
        '在参考基因组上比对的比例','重复读段的比例','被过滤掉读段数的比例',
        '13号染色体的Z值','18号染色体的Z值','21号染色体的Z值','X染色体的Z值']
df_f = pd.concat([df[fnum], ivf_dummies], axis=1).dropna()
X = (df_f.values.astype(float) - df_f.values.mean(0)) / df_f.values.std(0).clip(1e-10)
Xc = X - X.mean(0)
cov = Xc.T @ Xc / (len(Xc)-1)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
var_ratio = eigvals / eigvals.sum()
cum_ratio = np.cumsum(var_ratio)

print(f'样本: {len(X)}, 特征: {len(eigvals)}')
print(f'总方差: {eigvals.sum():.4f}')
print()
print(f'{"PC":>4s}  {"特征值":>8s}  {"方差比":>7s}  {"累计":>7s}')
print('-' * 35)
for i in range(len(eigvals)):
    print(f'{i+1:>4d}  {eigvals[i]:>8.4f}  {var_ratio[i]:>6.2%}  {cum_ratio[i]:>6.2%}')

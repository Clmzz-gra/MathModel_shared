#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型对比：OLS vs RF (same features, same CV)"""
import pandas as pd, numpy as np, os, warnings; warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

# ===== 加载 =====
cache_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\data'
male = pd.read_pickle(os.path.join(cache_dir, '2025C-male-clean.pkl'))

def pgw(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    for sep in ['w+','W+']:
        if sep in s:
            p = s.split(sep); return float(p[0])+float(p[1])/7.0
    return float(s.replace('w','').replace('W',''))

male['gw'] = male['检测孕周'].apply(pgw)
male['gw_c'] = male['gw'] - male.groupby('孕妇代码')['gw'].transform('mean')
male['bmi_c'] = male['孕妇BMI'] - male['孕妇BMI'].mean()
male['y_log'] = np.log(male['Y染色体浓度'])

tech_vars = ['原始读段数', '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例']
for v in tech_vars:
    male[v+'_z'] = (male[v] - male[v].mean()) / male[v].std()

month_cols = []
for m in sorted(male['检测日期_std'].dt.to_period('M').astype(str).unique()):
    if m != '2023-01':
        c = 'm_'+m; male[c] = (male['检测日期_std'].dt.to_period('M').astype(str)==m).astype(float)
        month_cols.append(c)

tech_z_cols = [v+'_z' for v in tech_vars]
feature_cols = ['gw_c','bmi_c'] + tech_z_cols + month_cols
X = male[feature_cols].values
y = male['y_log'].values
groups = male['孕妇代码'].values
n = len(y)

logo = LeaveOneGroupOut()
splits = list(logo.split(np.arange(n), groups=groups))

log_lines = []
log_lines.append(f'数据: {n} 条, {male["孕妇代码"].nunique()} 人, {len(feature_cols)} 个特征')
log_lines.append(f'OLCV splits: {len(splits)}')
log_lines.append('')

# OLS
preds_ols = np.zeros(n)
for tr, te in splits:
    X_tr = X[tr]; y_tr = y[tr]
    beta, _, _, _ = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), X_tr]), y_tr, rcond=None)
    preds_ols[te] = np.column_stack([np.ones(len(te)), X[te]]) @ beta
r2_ols = r2_score(y, preds_ols)
rmse_ols = np.sqrt(mean_squared_error(y, preds_ols))
log_lines.append(f'OLS            CV R2={r2_ols:.4f}  RMSE={rmse_ols:.4f}')
print(f'OLS CV R2={r2_ols:.4f}')

# RF
preds_rf = np.zeros(n)
for idx, (tr, te) in enumerate(splits):
    if idx % 20 == 0:
        print(f'  RF fold {idx}/{len(splits)}...')
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=10,
                                random_state=42, n_jobs=-1)
    rf.fit(X[tr], y[tr])
    preds_rf[te] = rf.predict(X[te])
r2_rf = r2_score(y, preds_rf)
rmse_rf = np.sqrt(mean_squared_error(y, preds_rf))
log_lines.append(f'RF(200树)       CV R2={r2_rf:.4f}  RMSE={rmse_rf:.4f}')
print(f'RF  CV R2={r2_rf:.4f}')

# 汇总
log_lines.append('')
log_lines.append(f'{"模型":18s} {"CV R2":>8s} {"RMSE":>8s} {"vs OLS":>8s}')
log_lines.append('-'*48)
for name, r2, rmse in [('OLS', r2_ols, rmse_ols), ('RandomForest', r2_rf, rmse_rf)]:
    delta = r2 - r2_ols
    log_lines.append(f'{name:18s} {r2:8.4f} {rmse:8.4f} {delta:+8.4f}')
    print(f'{name:18s} {r2:8.4f} {rmse:8.4f} {delta:+8.4f}')

with open('E:\\MathModel\\problems\\2025\\C题\\outputs\\scratch\\compare-log.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
print('Done.')

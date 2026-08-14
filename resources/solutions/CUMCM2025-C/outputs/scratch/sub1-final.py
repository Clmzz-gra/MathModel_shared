#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
问题1最终版：5个技术变量直接入模（含GC），GC不显著则提及
"""
import pandas as pd, numpy as np, os, warnings; warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

# ===== 0. 加载 =====
cache_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\data'
fig_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\figures'
chart_dir = 'E:\\MathModel\\problems\\2025\\C题\\solution\\artifacts\\charts'
tables_dir = 'E:\\MathModel\\problems\\2025\\C题\\solution\\artifacts\\tables'
code_dir = 'E:\\MathModel\\problems\\2025\\C题\\solution\\appendix\\code'
os.makedirs(chart_dir, exist_ok=True); os.makedirs(tables_dir, exist_ok=True); os.makedirs(code_dir, exist_ok=True)

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

# 技术变量标准化
tech_vars_keep = ['原始读段数', '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例']
for v in tech_vars_keep:
    male[v+'_z'] = (male[v] - male[v].mean()) / male[v].std()

# 月份
month_cols = []
for m in sorted(male['检测日期_std'].dt.to_period('M').astype(str).unique()):
    if m != '2023-01':
        c = 'm_'+m; male[c] = (male['检测日期_std'].dt.to_period('M').astype(str)==m).astype(float)
        month_cols.append(c)

tech_z_cols = [v+'_z' for v in tech_vars_keep]
feature_cols = ['gw_c','bmi_c'] + tech_z_cols + month_cols
X = male[feature_cols].values; y = male['y_log'].values; groups = male['孕妇代码'].values
n = len(y); p = X.shape[1]; y_raw = male['Y染色体浓度'].values

# ===== 1. 全样本拟合 =====
lr = LinearRegression(); lr.fit(X, y)
y_pred = lr.predict(X); resid = y - y_pred
r2 = r2_score(y, y_pred); rmse = np.sqrt(mean_squared_error(y, y_pred))

X_d = np.column_stack([np.ones(n), X])
XTX_inv = np.linalg.inv(X_d.T @ X_d)
se = np.sqrt(np.diag(XTX_inv) * np.var(resid))
t_vals = (np.concatenate([[lr.intercept_], lr.coef_])) / se
p_vals = 2*(1 - stats.t.cdf(np.abs(t_vals), n-p-1))

rss = np.sum(resid**2); tss = np.sum((y-y.mean())**2)
F = (tss-rss)/p / (rss/(n-p-1)); pF = 1 - stats.f.cdf(F, p, n-p-1)

print('='*60)
print('模型A (最终版): 线性混合效应')
print('R2=%.4f  RMSE=%.4f  F(%d,%d)=%.2f  p=%.1e' % (r2, rmse, p, n-p-1, F, pF))
print('%-45s %8s %8s %8s' % ('变量','系数','t值','p值'))
for i, name in enumerate(['截距']+feature_cols):
    stars = '***' if p_vals[i]<0.001 else '**' if p_vals[i]<0.01 else '*' if p_vals[i]<0.05 else ''
    print('%-45s %8.4f %8.2f %8.4f %s' % (name, lr.intercept_ if i==0 else lr.coef_[i-1], t_vals[i], p_vals[i], stars))

# ===== 2. 留一孕妇CV =====
logo = LeaveOneGroupOut()
preds_cv = np.zeros(n)
for tr, te in logo.split(np.arange(n), groups=groups):
    lr_cv = LinearRegression().fit(X[tr], y[tr])
    preds_cv[te] = lr_cv.predict(X[te])
r2_cv = r2_score(y, preds_cv); rmse_cv = np.sqrt(mean_squared_error(y, preds_cv))
print('\nCV R2=%.4f  CV RMSE=%.4f' % (r2_cv, rmse_cv))

# ===== 3. 简化系数表 =====
core_vars = [('gw_c','孕周（个体内）'), ('bmi_c','BMI'), 
             ('原始读段数_z','原始读段数'), ('在参考基因组上比对的比例_z','比对比例'),
             ('重复读段的比例_z','重复读段比例'), ('被过滤掉读段数的比例_z','过滤比例')]
core_idx = {c:i+1 for i,c in enumerate(feature_cols) if c in [v[0] for v in core_vars]}

print('\n--- 核心变量 ---')
print('%-20s %8s %8s %8s' % ('变量','系数','t值','p值'))
for c, lab in core_vars:
    idx = list(feature_cols).index(c) if c in feature_cols else -1
    if idx >= 0:
        i = idx + 1
        print('%-20s %8.4f %8.2f %8.4f %s' % (lab, lr.coef_[idx], t_vals[i], p_vals[i],
              '***' if p_vals[i]<0.001 else '**' if p_vals[i]<0.01 else '*' if p_vals[i]<0.05 else ''))

# ===== 4. 制图（仅更新系数森林图） =====
core_names = ['孕周(个体内)', 'BMI', '原始读段数', '比对比例', '重复读段', '过滤比例']
core_coefs = [lr.coef_[0], lr.coef_[1], lr.coef_[2], lr.coef_[3], lr.coef_[4], lr.coef_[5]]
core_se = [se[1], se[2], se[3], se[4], se[5], se[6]]
core_ci = [1.96*s for s in core_se]

fig, ax = plt.subplots(figsize=(8, 4))
y_pos = range(len(core_names))
colors = ['#2ca02c','#d62728'] + ['#1f77b4']*4
ax.barh(y_pos, core_coefs, xerr=core_ci, color=colors, height=0.5, capsize=4)
ax.axvline(x=0, color='#333333', linewidth=0.5)
ax.set_yticks(y_pos); ax.set_yticklabels(core_names, fontsize=10)
ax.set_xlabel('系数估计值 (95% CI)'); ax.set_title('修正模型核心系数', fontsize=11)
for i, (coef, ci) in enumerate(zip(core_coefs, core_ci)):
    sign = '+' if coef>0 else ''
    ax.text(coef+ci+0.002, i, '%s%.3f'%(sign,coef), va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'sub1-final-coefficients.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(chart_dir, 'sub1-final-coefficients.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('\n系数森林图已保存')

# ===== 5. LaTeX表 =====
with open(os.path.join(tables_dir, 'sub1-final-coefficients.tex'), 'w', encoding='utf-8') as f:
    f.write(r'\begin{tabular}{lrrr}'+'\n')
    f.write(r'\toprule'+'\n')
    f.write(r'变量 & 系数 & t 值 & p 值 \\'+'\n')
    f.write(r'\midrule'+'\n')
    for c, lab in core_vars:
        idx = list(feature_cols).index(c)
        i = idx + 1
        stars = '***' if p_vals[i]<0.001 else '**' if p_vals[i]<0.01 else '*' if p_vals[i]<0.05 else ''
        f.write('%s & %.4f%s & %.2f & %.4f \\\\\n' % (lab, lr.coef_[idx], stars, t_vals[i], p_vals[i]))
    f.write(r'\midrule'+'\n')
    f.write(r'R$^2$ & \multicolumn{3}{r}{%.3f} \\\\\n' % r2)
    f.write(r'CV R$^2$ & \multicolumn{3}{r}{%.3f} \\\\\n' % r2_cv)
    f.write(r'\bottomrule'+'\n')
    f.write(r'\end{tabular}'+'\n')
print('LaTeX表已保存')

# ===== 6. 附录 =====
import shutil; shutil.copy(__file__, os.path.join(code_dir, 'sub1-final.py'))
print('附录代码已归档')
print('\n完成。CV R2=%.4f' % r2_cv)

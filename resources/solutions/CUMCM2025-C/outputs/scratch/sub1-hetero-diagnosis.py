#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
异方差诊断 + WLS 对比
目标：找到喇叭形残差的根源，检验加权最小二乘能否改善
"""
import pandas as pd, numpy as np, os, warnings; warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error

# ===== 0. 加载 =====
cache_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\data'
fig_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\figures'

male = pd.read_pickle(os.path.join(cache_dir, '2025C-male-clean.pkl'))

# 解析孕周
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
male['y_raw'] = male['Y染色体浓度']

# 技术变量标准化
tech_vars_keep = ['原始读段数', '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例']
for v in tech_vars_keep:
    male[v+'_z'] = (male[v] - male[v].mean()) / male[v].std()

# 月份哑变量
month_cols = []
for m in sorted(male['检测日期_std'].dt.to_period('M').astype(str).unique()):
    if m != '2023-01':
        c = 'm_'+m; male[c] = (male['检测日期_std'].dt.to_period('M').astype(str)==m).astype(float)
        month_cols.append(c)

tech_z_cols = [v+'_z' for v in tech_vars_keep]
feature_cols = ['gw_c','bmi_c'] + tech_z_cols + month_cols
X = male[feature_cols].values
y = male['y_log'].values
groups = male['孕妇代码'].values
n = len(y)

# ===== 1. OLS 基准 =====
lr = LinearRegression(); lr.fit(X, y)
y_pred = lr.predict(X)
resid = y - y_pred
abs_resid = np.abs(resid)
r2_ols = r2_score(y, y_pred)
print(f'OLS: R2={r2_ols:.4f}, RMSE={np.sqrt(mean_squared_error(y, y_pred)):.4f}')

# ===== 2. 异方差源头诊断 =====
print('\n=== 异方差源头诊断 ===')

# 假说二: 残差方差 vs 原始读段数 (测序深度)
# 读段数越少 → 计数噪声越大 → 残差越大
reads = male['原始读段数'].values
# 按读段数分桶看残差标准差
bins = np.percentile(reads, [0, 25, 50, 75, 100])
for i in range(len(bins)-1):
    mask = (reads >= bins[i]) & (reads < bins[i+1])
    if mask.sum() > 10:
        print(f'  读段数 [{bins[i]:.0f},{bins[i+1]:.0f}): '
              f'n={mask.sum()}, 残差SD={np.std(resid[mask]):.4f}, '
              f'拟合均值={np.mean(y_pred[mask]):.4f}')

# 假说二: 残差方差 vs Y浓度本身
# Y浓度低 → 泊松噪声大
y_raw = male['y_raw'].values
y_bins = np.array([0, 0.03, 0.05, 0.08, 0.50])
for i in range(len(y_bins)-1):
    mask = (y_raw >= y_bins[i]) & (y_raw < y_bins[i+1])
    if mask.sum() > 10:
        print(f'  Y浓度 [{y_bins[i]:.2f},{y_bins[i+1]:.2f}): '
              f'n={mask.sum()}, 残差SD={np.std(resid[mask]):.4f}, '
              f'读段数均值={np.mean(reads[mask]):.0f}')

# 假说一: log变换后残差 vs 拟合值
# 直接看图，确认喇叭形方向
print(f'\n  拟合值范围: [{np.min(y_pred):.3f}, {np.max(y_pred):.3f}]')
print(f'  残差SD(低拟合值1/3): {np.std(resid[y_pred < np.percentile(y_pred, 33)]):.4f}')
print(f'  残差SD(中拟合值1/3): {np.std(resid[(y_pred >= np.percentile(y_pred, 33)) & (y_pred < np.percentile(y_pred, 67))]):.4f}')
print(f'  残差SD(高拟合值1/3): {np.std(resid[y_pred >= np.percentile(y_pred, 67)]):.4f}')

# 假说三: 残差 vs 月份（批次效应是否集中在某些月份）
print('\n  按月份残差SD:')
for mc in month_cols:
    mask = male[mc] == 1
    if mask.sum() > 10:
        print(f'    {mc}: n={mask.sum()}, 残差SD={np.std(resid[mask]):.4f}')

# ===== 3. WLS =====
# 策略: 用拟合值的倒数、读段数的倒数、或Y浓度的倒数作为权重
print('\n=== WLS 尝试 ===')

# 权重方案1: 基于拟合值分组估计方差
# 对拟合值排序后取滑动窗口标准差，用1/σ²作为权重
order = np.argsort(y_pred)
y_pred_sorted = y_pred[order]
resid_sorted = resid[order]
window = max(50, n // 20)
var_est = np.zeros(n)
for i in range(n):
    lo = max(0, i - window//2)
    hi = min(n, i + window//2)
    var_est[order[i]] = np.var(resid_sorted[lo:hi]) + 1e-6
weights1 = 1.0 / var_est

# 权重方案2: 逆读段数 (泊松噪声 ∝ 1/√N)
weights2 = reads / reads.mean()  # 读段数多 → 权重大

# 权重方案3: 混合 (读段数 × 1/拟合值分组方差)
weights3 = weights1 * weights2

# 分别拟合WLS
def fit_wls(X, y, weights):
    """用sklearn的LinearRegression做加权最小二乘"""
    w_sqrt = np.sqrt(weights)
    X_w = X * w_sqrt[:, np.newaxis]
    y_w = y * w_sqrt
    lr = LinearRegression().fit(X_w, y_w)
    # 预测和残差仍在原始尺度
    y_pred = lr.predict(X)
    resid = y - y_pred
    r2 = r2_score(y, y_pred)
    return lr, y_pred, resid, r2

print('\n方案对比:')
print(f'{"方案":20s} {"R2(全样本)":>10s} {"残差SD":>10s} {"残差SD/低区":>12s} {"残差SD/高区":>12s}')
print('-'*70)

# OLS基准
mask_low = y_pred < np.percentile(y_pred, 33)
mask_high = y_pred >= np.percentile(y_pred, 67)
sd_low_ols = np.std(resid[mask_low])
sd_high_ols = np.std(resid[mask_high])
print(f'{"OLS(基准)":20s} {r2_ols:10.4f} {np.std(resid):10.4f} {sd_low_ols:12.4f} {sd_high_ols:12.4f}')

# WLS方案1: 拟合值分组方差倒数
lr1, yp1, r1, r2_1 = fit_wls(X, y, weights1)
sd_low_1 = np.std(r1[yp1 < np.percentile(yp1, 33)])
sd_high_1 = np.std(r1[yp1 >= np.percentile(yp1, 67)])
print(f'{"WLS(拟合值分组)":20s} {r2_1:10.4f} {np.std(r1):10.4f} {sd_low_1:12.4f} {sd_high_1:12.4f}')

# WLS方案2: 逆读段数
lr2, yp2, r2w, r2_2 = fit_wls(X, y, weights2)
sd_low_2 = np.std(r2w[yp2 < np.percentile(yp2, 33)])
sd_high_2 = np.std(r2w[yp2 >= np.percentile(yp2, 67)])
print(f'{"WLS(读段数)":20s} {r2_2:10.4f} {np.std(r2w):10.4f} {sd_low_2:12.4f} {sd_high_2:12.4f}')

# WLS方案3: 混合
lr3, yp3, r3, r2_3 = fit_wls(X, y, weights3)
sd_low_3 = np.std(r3[yp3 < np.percentile(yp3, 33)])
sd_high_3 = np.std(r3[yp3 >= np.percentile(yp3, 67)])
print(f'{"WLS(混合)":20s} {r2_3:10.4f} {np.std(r3):10.4f} {sd_low_3:12.4f} {sd_high_3:12.4f}')

# ===== 4. 汇报系数稳定性 =====
print('\n核心系数稳定性:')
print(f'{"变量":20s} {"OLS":>10s} {"WLS(分组)":>10s} {"WLS(读段)":>10s} {"WLS(混合)":>10s}')
print('-'*60)
for i, name in enumerate(feature_cols):
    if i < 6:  # 核心6个
        print(f'{name:20s} {lr.coef_[i]:10.4f} {lr1.coef_[i]:10.4f} {lr2.coef_[i]:10.4f} {lr3.coef_[i]:10.4f}')

# ===== 5. CV对比 =====
print('\nCV对比 (留一孕妇):')
logo = LeaveOneGroupOut()

def cv_eval(X, y, weights):
    preds = np.zeros(n)
    for tr, te in logo.split(np.arange(n), groups=groups):
        w_tr = weights[tr]
        w_sq = np.sqrt(w_tr)
        Xw = X[tr] * w_sq[:, np.newaxis]
        yw = y[tr] * w_sq
        lr_cv = LinearRegression().fit(Xw, yw)
        preds[te] = lr_cv.predict(X[te])
    return r2_score(y, preds)

r2_cv_ols = cv_eval(X, y, np.ones(n))
r2_cv_wls = cv_eval(X, y, weights2)  # 用读段数方案
print(f'  OLS CV R2: {r2_cv_ols:.4f}')
print(f'  WLS(读段数) CV R2: {r2_cv_wls:.4f}')

# ===== 6. 对比诊断图 =====
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 第一行: 残差 vs Y浓度 (散点)
for col_idx, (yp, r, title) in enumerate([
    (y_pred, resid, 'OLS'),
    (yp1, r1, 'WLS(拟合值分组方差)'),
    (yp2, r2w, 'WLS(读段数)')
]):
    ax = axes[0, col_idx]
    ax.scatter(yp, r, s=5, alpha=0.35, color='#1f77b4', edgecolors='none')
    ax.axhline(y=0, color='#d62728', linestyle='--', linewidth=0.8)
    ax.set_xlabel('拟合值 (log-Y)'); ax.set_ylabel('残差')
    ax.set_title(f'{title}', fontsize=10)

# 第二行: 残差 vs 读段数 (散点)
for col_idx, (yp, r, title) in enumerate([
    (reads, resid, 'OLS'),
    (reads, r1, 'WLS(拟合值分组方差)'),
    (reads, r2w, 'WLS(读段数)')
]):
    ax = axes[1, col_idx]
    ax.scatter(reads, r, s=5, alpha=0.35, color='#2ca02c', edgecolors='none')
    ax.axhline(y=0, color='#d62728', linestyle='--', linewidth=0.8)
    ax.set_xlabel('原始读段数'); ax.set_ylabel('残差')
    ax.set_title(f'{title} vs 读段数', fontsize=10)

plt.suptitle('异方差诊断：OLS vs WLS', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{fig_dir}/sub1-hetero-diagnosis.pdf', dpi=150, bbox_inches='tight')
plt.close()
print(f'\n诊断图已保存: {fig_dir}/sub1-hetero-diagnosis.pdf')

# ===== 7. 最佳方案的残差Q-Q图 =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# OLS Q-Q
ax = axes[0]
stats.probplot(resid, dist='norm', plot=ax)
ax.get_lines()[0].set_color('#333333'); ax.get_lines()[1].set_color('#d62728')
ax.set_title('OLS 残差 Q-Q', fontsize=11)

# WLS(读段数) Q-Q
ax = axes[1]
stats.probplot(r2w, dist='norm', plot=ax)
ax.get_lines()[0].set_color('#333333'); ax.get_lines()[1].set_color('#d62728')
ax.set_title('WLS(读段数) 残差 Q-Q', fontsize=11)

plt.suptitle('Q-Q图对比', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{fig_dir}/sub1-hetero-qq-compare.pdf', dpi=150, bbox_inches='tight')
plt.close()
print(f'QQ对比图已保存: {fig_dir}/sub1-hetero-qq-compare.pdf')

print('\n=== 完成 ===')

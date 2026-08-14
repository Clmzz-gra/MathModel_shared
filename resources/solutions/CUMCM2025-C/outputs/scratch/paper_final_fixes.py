#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
终稿补全：Bootstrap中介分析CI + ROC曲线标注阈值点 + 改进残差图
快速版 — 预计算患者数据用于高效Bootstrap
"""
import numpy as np
import pandas as pd
import os, sys, time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from scipy import stats

cache_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\data'
fig_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\figures'
chart_dir = 'E:\\MathModel\\problems\\2025\\C题\\solution\\artifacts\\charts'
fig_body_dir = 'E:\\MathModel\\problems\\2025\\C题\\solution\\artifacts\\figures-body'
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(chart_dir, exist_ok=True)
os.makedirs(fig_body_dir, exist_ok=True)

# ============================================================
# 第1部分：Bootstrap中介分析置信区间（优化版）
# ============================================================
print('=' * 60)
print('1. Bootstrap 中介分析置信区间')
print('=' * 60)

data = pd.read_pickle(os.path.join(cache_dir, '2025C-sub1-preprocessed.pkl'))
month_cols = [c for c in data.columns if c.startswith('m_202')]
X_A_cols = ['gw_c', 'bmi_c', 'pc1'] + month_cols
X_B_cols = X_A_cols + ['x_c']

# 预计算: 以孕妇为单位分组，存储每个孕妇的X_A, X_B, y
patient_ids = data['孕妇代码'].unique()
n_patients = len(patient_ids)
print(f'患者数: {n_patients}')

# 预计算每个患者的行索引和数据块
patient_blocks = {}  # pid -> (X_A, X_B, y, n_rows)
for pid in patient_ids:
    mask = data['孕妇代码'] == pid
    patient_blocks[pid] = (
        data.loc[mask, X_A_cols].values,
        data.loc[mask, X_B_cols].values,
        data.loc[mask, 'y_log'].values,
        mask.sum()
    )

np.random.seed(42)
B = 1000

med_ratios = np.full(B, np.nan)
b1_A_boot = np.full(B, np.nan)

t0 = time.time()
for b in range(B):
    if b % 250 == 0:
        print(f'  Bootstrap: {b}/{B} ({time.time()-t0:.1f}s)')
    
    # Bootstrap重采样患者
    sampled_ids = np.random.choice(patient_ids, size=n_patients, replace=True)
    
    # 聚合重采样数据
    X_A_list, X_B_list, y_list = [], [], []
    for pid in sampled_ids:
        blk = patient_blocks[pid]
        X_A_list.append(blk[0])
        X_B_list.append(blk[1])
        y_list.append(blk[2])
    
    X_A_b = np.vstack(X_A_list)
    X_B_b = np.vstack(X_B_list)
    y_b = np.concatenate(y_list)
    
    try:
        # 模型A
        X_A_d = np.column_stack([np.ones(len(X_A_b)), X_A_b])
        beta_A = np.linalg.lstsq(X_A_d, y_b, rcond=None)[0]
        b1_A = beta_A[1]  # 孕周系数
        
        # 模型B
        X_B_d = np.column_stack([np.ones(len(X_B_b)), X_B_b])
        beta_B = np.linalg.lstsq(X_B_d, y_b, rcond=None)[0]
        b1_B = beta_B[1]
        
        b1_A_boot[b] = b1_A
        if abs(b1_A) > 1e-6:
            med_ratios[b] = (b1_A - b1_B) / b1_A * 100
    except:
        pass

valid = ~np.isnan(med_ratios)
med_ratios = med_ratios[valid]
b1_A_boot = b1_A_boot[valid]
print(f'Bootstrap完成: {len(med_ratios)}/{B} 有效 ({time.time()-t0:.1f}s)')

ci_low = np.percentile(med_ratios, 2.5)
ci_high = np.percentile(med_ratios, 97.5)
med_median = np.median(med_ratios)

print(f'\n中介比例 Bootstrap (B={len(med_ratios)}):')
print(f'  点估计: {med_median:.1f}%')
print(f'  95% CI: [{ci_low:.1f}%, {ci_high:.1f}%]')
print(f'  模型A孕周系数: {np.median(b1_A_boot):.4f} (95% CI: [{np.percentile(b1_A_boot, 2.5):.4f}, {np.percentile(b1_A_boot, 97.5):.4f}])')

# Bootstrap分布图
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(med_ratios, bins=50, color='#1f77b4', alpha=0.7, edgecolor='white', linewidth=0.3)
ax.axvline(x=med_median, color='#d62728', linewidth=2, label=f'中位数={med_median:.1f}%')
ax.axvline(x=ci_low, color='#333333', linewidth=1.5, linestyle='--', label=f'95% CI: [{ci_low:.1f}%, {ci_high:.1f}%]')
ax.axvline(x=ci_high, color='#333333', linewidth=1.5, linestyle='--')
ax.set_xlabel('中介比例 (%)'); ax.set_ylabel('频次')
ax.set_title(f'Bootstrap 中介比例分布 (B={len(med_ratios)})')
ax.legend(fontsize=9)
plt.tight_layout()
for p in [fig_dir, chart_dir]:
    fig.savefig(os.path.join(p, 'sub1-mediation-bootstrap.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('Bootstrap分布图已保存')

# ============================================================
# 第2部分：重绘ROC曲线（标注阈值点）
# ============================================================
print('\n' + '=' * 60)
print('2. 重绘 ROC 曲线（标注 Youden 和 TPR>=90% 阈值点）')
print('=' * 60)

df = pd.read_pickle(os.path.join(cache_dir, '2025C-sub4-preprocessed.pkl'))
df_clean = pd.read_pickle(os.path.join(cache_dir, '2025C-female-clean.pkl'))

gc_map = {'13号染色体的GC含量': 'GC_13', '18号染色体的GC含量': 'GC_18', '21号染色体的GC含量': 'GC_21'}
for orig, new in gc_map.items():
    df[new] = df_clean[orig].values
df['X_conc'] = df_clean['X染色体浓度'].values

z_cols = ['Z13_corrected', 'Z18_corrected', 'Z21_corrected', 'ZX_corrected']
Z = df[z_cols].values
ab = df['AB_异常'].values

n = len(Z)
Z_diff = np.zeros((n, 4))
for k in range(4):
    other = [j for j in range(4) if j != k]
    Z_diff[:, k] = Z[:, k] - np.median(Z[:, other], axis=1)

GC_18_13 = df['GC_18'].values - df['GC_13'].values
GC_21_13 = df['GC_21'].values - df['GC_13'].values
GC_18_21 = df['GC_18'].values - df['GC_21'].values

bmi, age, gw = df['孕妇BMI'].values, df['年龄'].values, df['孕周_数值'].values
filter_rate = df['被过滤掉读段数的比例'].values
dup_rate = df['重复读段的比例'].values
map_rate = df['在参考基因组上比对的比例'].values
X_conc = df['X_conc'].values

feature_sets = {
    13: np.column_stack([Z[:,0], Z_diff[:,0], df['GC_13'].values, GC_18_13, GC_21_13,
        filter_rate, dup_rate, map_rate, X_conc, bmi, age, gw]),
    18: np.column_stack([Z[:,1], Z_diff[:,1], df['GC_18'].values, GC_18_13, GC_18_21,
        filter_rate, dup_rate, map_rate, X_conc, bmi, age, gw, bmi*Z[:,1], age*Z[:,1]]),
    21: np.column_stack([Z[:,2], Z_diff[:,2], df['GC_21'].values, GC_21_13, GC_18_21,
        filter_rate, dup_rate, map_rate, X_conc, bmi, age, gw]),
}

# 去重
date_map = df_clean[['孕妇代码', '检测日期_std']].drop_duplicates('孕妇代码')
df['孕妇代码'] = df_clean['孕妇代码'].values
df['检测日期_std'] = date_map.set_index('孕妇代码').loc[df['孕妇代码'], '检测日期_std'].values

mask_labeled = (ab == 0) | (ab == 1)
labeled = df[mask_labeled].sort_values('检测日期_std').drop_duplicates('孕妇代码', keep='first')
dedup_idx = labeled.index.values
y_true = labeled['AB_异常'].values.astype(float)

ab_types = df_clean.loc[dedup_idx, '染色体的非整倍体'].values
def has_type(s, t):
    return not pd.isna(s) and s != '' and t in str(s)

is_t13 = np.array([has_type(s, 'T13') for s in ab_types])
is_t18 = np.array([has_type(s, 'T18') for s in ab_types])
is_t21 = np.array([has_type(s, 'T21') for s in ab_types])

def fisher_lda_fit(X, y):
    X0, X1 = X[y==0], X[y==1]
    mu0, mu1 = X0.mean(0), X1.mean(0)
    S0 = np.cov(X0, rowvar=False) if len(X0)>1 else np.eye(X.shape[1])
    S1 = np.cov(X1, rowvar=False) if len(X1)>1 else np.eye(X.shape[1])
    Sw = ((len(X0)-1)*S0 + (len(X1)-1)*S1) / (len(X)-2)
    reg = 0.01 * np.trace(Sw) / Sw.shape[0]
    Sw_inv = np.linalg.pinv(Sw + reg*np.eye(Sw.shape[0]))
    w = Sw_inv @ (mu1 - mu0)
    return w

chr_targets = {13: is_t13, 18: is_t18, 21: is_t21}
all_scores = np.zeros((len(dedup_idx), 3))

for k in [13, 18, 21]:
    X_k = feature_sets[k][dedup_idx]
    y_k = chr_targets[k].astype(float)
    if y_k.sum() < 2: continue
    X_mean, X_std = X_k.mean(0), X_k.std(0)
    X_std[X_std<1e-10] = 1.0
    X_scaled = (X_k - X_mean) / X_std
    w = fisher_lda_fit(X_scaled, y_k)
    all_scores[:, {13:0, 18:1, 21:2}[k]] = X_scaled @ w

y_score = np.max(all_scores, axis=1)

# ROC计算
order = np.argsort(-y_score)
y_sort, s_sort = y_true[order], y_score[order]
n_pos, n_neg = y_true.sum(), len(y_true)-y_true.sum()

tpr_list, fpr_list, th_list = [0], [0], [s_sort[0]+1]
tp = fp = 0; i = 0
while i < len(y_sort):
    j = i
    while j < len(y_sort) and s_sort[j] == s_sort[i]:
        if y_sort[j]==1: tp+=1
        else: fp+=1
        j += 1
    tpr_list.append(tp/n_pos); fpr_list.append(fp/n_neg); th_list.append(s_sort[i])
    i = j
fpr = np.array(fpr_list); tpr = np.array(tpr_list); thresholds = np.array(th_list)

roc_auc = np.trapezoid(tpr, fpr)
youden = tpr - fpr
idx_best = np.argmax(youden)

idx_tpr90 = np.where(tpr >= 0.90)[0]
has_tpr90 = len(idx_tpr90) > 0
if has_tpr90: idx_tpr90 = idx_tpr90[0]

print(f'AUC={roc_auc:.3f}')
print(f'Youden: FPR={fpr[idx_best]:.3f}, TPR={tpr[idx_best]:.3f}')
if has_tpr90:
    print(f'TPR>=90%: FPR={fpr[idx_tpr90]:.3f}, TPR={tpr[idx_tpr90]:.3f}, 特异度={1-fpr[idx_tpr90]:.3f}')

# 绘制ROC
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Fisher LDA (AUC={roc_auc:.3f})')
ax.plot([0,1], [0,1], 'k--', linewidth=0.8, alpha=0.5, label='随机分类器')

ax.scatter(fpr[idx_best], tpr[idx_best], c='#d62728', s=100, marker='o',
           zorder=5, edgecolors='white', linewidth=1.5,
           label=f'Youden最优 (TPR={tpr[idx_best]:.1%}, FPR={fpr[idx_best]:.1%})')

if has_tpr90:
    ax.scatter(fpr[idx_tpr90], tpr[idx_tpr90], c='#ff7f0e', s=120, marker='s',
               zorder=5, edgecolors='white', linewidth=1.5,
               label=f'高灵敏度 (TPR={tpr[idx_tpr90]:.1%}, FPR={fpr[idx_tpr90]:.1%})')

ax.axhline(y=tpr[idx_best], color='#d62728', linestyle=':', linewidth=0.8, alpha=0.3)
ax.axvline(x=fpr[idx_best], color='#d62728', linestyle=':', linewidth=0.8, alpha=0.3)

ax.set_xlabel('假阳性率 (FPR)', fontsize=11)
ax.set_ylabel('真阳性率 (TPR / 召回率)', fontsize=11)
ax.set_title('Fisher LDA 组合评分的 ROC 曲线', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.2)
plt.tight_layout()

for p in [fig_dir, chart_dir, fig_body_dir]:
    fig.savefig(os.path.join(p, 'sub4-roc-curve.pdf'), dpi=200, bbox_inches='tight')
plt.close()
print('ROC曲线已保存（含阈值标注）')

# ============================================================
# 第3部分：改进残差诊断图
# ============================================================
print('\n' + '=' * 60)
print('3. 重绘残差诊断图（增强透明度）')
print('=' * 60)

from statsmodels.nonparametric.smoothers_lowess import lowess

data2 = pd.read_pickle(os.path.join(cache_dir, '2025C-sub1-preprocessed.pkl'))
month_cols2 = [c for c in data2.columns if c.startswith('m_202')]
X_A_cols2 = ['gw_c', 'bmi_c', 'pc1'] + month_cols2
X_A2 = data2[X_A_cols2].values
y2 = data2['y_log'].values

X_A2_d = np.column_stack([np.ones(len(X_A2)), X_A2])
beta_A2 = np.linalg.lstsq(X_A2_d, y2, rcond=None)[0]
y_pred_A2 = X_A2_d @ beta_A2
resid_A2 = y2 - y_pred_A2

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
stats.probplot(resid_A2, dist='norm', plot=ax)
ax.get_lines()[0].set_color('#333333'); ax.get_lines()[1].set_color('#d62728')
ax.set_title('Residual Q-Q Plot', fontsize=12, fontweight='bold')

ax = axes[1]
ax.scatter(y_pred_A2, resid_A2, s=8, alpha=0.2, color='#1f77b4', edgecolors='none')

sorted_idx = np.argsort(y_pred_A2)
loess_fit = lowess(resid_A2[sorted_idx], y_pred_A2[sorted_idx], frac=0.3, return_sorted=True)
ax.plot(loess_fit[:,0], loess_fit[:,1], 'r-', linewidth=2, alpha=0.8, label='loess smooth')

ax.axhline(y=0, color='#333333', linestyle='--', linewidth=0.8)
sd_r = np.std(resid_A2)
ax.axhline(y=2*sd_r, color='#999999', linestyle=':', linewidth=0.8, alpha=0.5)
ax.axhline(y=-2*sd_r, color='#999999', linestyle=':', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Fitted (log-Y)'); ax.set_ylabel('Residual')
ax.set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
for p in [fig_dir, chart_dir, fig_body_dir]:
    fig.savefig(os.path.join(p, 'sub1-residual-diagnostics.pdf'), dpi=200, bbox_inches='tight')
plt.close()
print('残差诊断图已保存（增强版）')

# ============================================================
# 汇总
# ============================================================
print('\n' + '=' * 60)
print('全部完成！')
print('=' * 60)
print(f'中介比例: {med_median:.1f}% (95% CI: [{ci_low:.1f}%, {ci_high:.1f}%])')
print(f'ROC AUC: {roc_auc:.3f}')
print(f'图表已保存到: {fig_dir}, {chart_dir}, {fig_body_dir}')

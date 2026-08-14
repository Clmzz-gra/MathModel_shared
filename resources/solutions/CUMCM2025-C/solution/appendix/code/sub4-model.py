#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
问题4完整实现：女胎异常评分模型（Fisher退化 → 马氏距离异常检测）
三层架构：GC校正 → 马氏距离评分 → ROC阈值校准
纯 numpy/scipy 实现，无 sklearn 依赖
"""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats

# ===== 路径 =====
cache_dir  = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\data'
fig_dir    = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\figures'
chart_dir  = 'E:\\MathModel\\problems\\2025\\C题\\solution\\artifacts\\charts'
tables_dir = 'E:\\MathModel\\problems\\2025\\C题\\solution\\artifacts\\tables'
code_dir   = 'E:\\MathModel\\problems\\2025\\C题\\solution\\appendix\\code'
for d in [fig_dir, chart_dir, tables_dir, code_dir]: os.makedirs(d, exist_ok=True)

# ===== 0. 加载 =====
df = pd.read_pickle(os.path.join(cache_dir, '2025C-sub4-preprocessed.pkl'))
print(f'加载数据: {df.shape[0]} 行 x {df.shape[1]} 列')

z_cols = ['Z13_corrected','Z18_corrected','Z21_corrected','ZX_corrected']
z_labels = ['13号','18号','21号','X']
Z = df[z_cols].values
ab = df['AB_异常'].values
n_total = len(Z)

mask_normal = ab == 0
n_normal = mask_normal.sum()
Z_normal = Z[mask_normal]

mask_abnormal = ab == 1
n_abnormal = mask_abnormal.sum()
Z_abnormal = Z[mask_abnormal]

print(f'正常: {n_normal}  异常(标注): {n_abnormal}')

# ===== 1. 稳健协方差估计（迭代裁剪法，无需sklearn） =====
print('\n' + '='*60)
print('1. 稳健协方差估计（迭代裁剪）')
print('='*60)

def trimmed_covariance(X, trim_frac=0.25, max_iter=5):
    """迭代裁剪协方差：去除马氏距离最大的 trim_frac 样本后重估"""
    mu = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False)
    for it in range(max_iter):
        Sinv = np.linalg.inv(Sigma)
        diff = X - mu
        D = np.sum(diff @ Sinv * diff, axis=1)
        cutoff = np.quantile(D, 1 - trim_frac)
        keep = D <= cutoff
        if keep.sum() < 0.5 * len(X):
            break  # 不要裁太多
        X_trim = X[keep]
        mu_new = X_trim.mean(axis=0)
        Sigma_new = np.cov(X_trim, rowvar=False)
        if np.max(np.abs(mu_new - mu)) < 1e-6:
            mu, Sigma = mu_new, Sigma_new
            break
        mu, Sigma = mu_new, Sigma_new
    return mu, Sigma, it+1

# 经典协方差
mu_classic = Z_normal.mean(axis=0)
Sigma_classic = np.cov(Z_normal, rowvar=False)

# 迭代裁剪
mu_robust, Sigma_robust, n_iter = trimmed_covariance(Z_normal, trim_frac=0.25)
mu_mcd = mu_robust
Sigma_mcd = Sigma_robust

print(f'经典协方差 det = {np.linalg.det(Sigma_classic):.4f}')
print(f'裁剪协方差 det = {np.linalg.det(Sigma_robust):.4f}  (迭代{n_iter}轮)')
print(f'det 比 (Robust/Classic) = {np.linalg.det(Sigma_robust)/np.linalg.det(Sigma_classic):.3f}')

print('\n相关系数对比:')
print(f'{"对":12s} {"经典":>7s} {"Robust":>7s} {"差":>7s}')
for i in range(4):
    for j in range(i+1, 4):
        r_cl = Sigma_classic[i,j]/np.sqrt(Sigma_classic[i,i]*Sigma_classic[j,j])
        r_ro = Sigma_robust[i,j]/np.sqrt(Sigma_robust[i,i]*Sigma_robust[j,j])
        print(f'Z{z_labels[i]}-Z{z_labels[j]:3s} {r_cl:7.3f} {r_ro:7.3f} {r_ro-r_cl:+7.3f}')

# ===== 2. 马氏距离 =====
print('\n' + '='*60)
print('2. 马氏距离评分')
print('='*60)

Sigma_inv = np.linalg.inv(Sigma_robust)

def mahalanobis(Z_arr, mu, Sinv):
    diff = Z_arr - mu
    return np.sum(diff @ Sinv * diff, axis=1)

S_all = mahalanobis(Z, mu_robust, Sigma_inv)
S_normal = S_all[mask_normal]
S_abnormal = S_all[mask_abnormal]

print(f'正常 S: mean={S_normal.mean():.2f} std={S_normal.std():.2f} median={np.median(S_normal):.2f}')
print(f'异常 S: mean={S_abnormal.mean():.2f} std={S_abnormal.std():.2f} median={np.median(S_abnormal):.2f}')
ks_stat, ks_p = stats.ks_2samp(S_normal, S_abnormal)
print(f'KS检验: D={ks_stat:.3f} p={ks_p:.4f}')

# ===== 3. ROC 分析（手动实现） =====
print('\n' + '='*60)
print('3. ROC 分析')
print('='*60)

# 标注样本按孕妇去重（需要日期信息，从 clean 数据补）
df_clean = pd.read_pickle(os.path.join(cache_dir, '2025C-female-clean.pkl'))
date_map = df_clean[['孕妇代码','检测日期_std']].drop_duplicates('孕妇代码')
mask_labeled = mask_abnormal | mask_normal
labeled = df[mask_labeled].copy()
labeled['S'] = S_all[mask_labeled]
labeled = labeled.merge(date_map, on='孕妇代码', how='left')
labeled_sorted = labeled.sort_values('检测日期_std')
labeled_dedup = labeled_sorted.drop_duplicates('孕妇代码', keep='first')
print(f'去重前: {len(labeled)}  去重后: {len(labeled_dedup)}')
print(f'  异常: {(labeled_dedup["AB_异常"]==1).sum()}  正常: {(labeled_dedup["AB_异常"]==0).sum()}')

y_true = labeled_dedup['AB_异常'].values.astype(float)
y_score = labeled_dedup['S'].values

# 手动 ROC
def compute_roc(y_true, y_score):
    order = np.argsort(-y_score)
    y_sort = y_true[order]
    s_sort = y_score[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    tpr_list, fpr_list, thresh_list = [0], [0], [s_sort[0] + 1]
    tp = fp = 0
    i = 0
    while i < len(y_sort):
        j = i
        while j < len(y_sort) and s_sort[j] == s_sort[i]:
            if y_sort[j] == 1: tp += 1
            else: fp += 1
            j += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
        thresh_list.append(s_sort[i])
        i = j
    return np.array(fpr_list), np.array(tpr_list), np.array(thresh_list)

fpr, tpr, thresholds = compute_roc(y_true, y_score)

# AUC（梯形法则）
roc_auc = np.trapezoid(tpr, fpr)

# Youden
youden = tpr - fpr
idx_best = np.argmax(youden)
tau_star = thresholds[idx_best]
print(f'AUC = {roc_auc:.4f}')
print(f'τ* = {tau_star:.2f} (Youden={youden[idx_best]:.3f})')
print(f'  TPR={tpr[idx_best]:.3f}  FPR={fpr[idx_best]:.3f}')

# 偏召回阈值
idx_tpr90 = np.where(tpr >= 0.9)[0]
if len(idx_tpr90) > 0:
    idx_tpr90 = idx_tpr90[0]
    tau_lower = thresholds[idx_tpr90]
else:
    tau_lower = thresholds[-1]
print(f'τ_lower = {tau_lower:.2f} (TPR={tpr[idx_tpr90]:.3f})')

# ===== 4. 混淆矩阵 =====
print('\n' + '='*60)
print('4. 混淆矩阵')
print('='*60)

y_pred = (y_score >= tau_star).astype(float)
tp = ((y_true == 1) & (y_pred == 1)).sum()
fp = ((y_true == 0) & (y_pred == 1)).sum()
fn = ((y_true == 1) & (y_pred == 0)).sum()
tn = ((y_true == 0) & (y_pred == 0)).sum()
acc = (tp+tn)/(tp+tn+fp+fn)
prec = tp/(tp+fp) if (tp+fp)>0 else 0
rec = tp/(tp+fn) if (tp+fn)>0 else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0

print(f'TP={tp} FP={fp} FN={fn} TN={tn}')
print(f'准确率={acc:.3f} 精确率={prec:.3f} 召回率={rec:.3f} F1={f1:.3f}')

# ===== 5. 染色体边际贡献 =====
print('\n' + '='*60)
print('5. 染色体边际贡献')
print('='*60)

sigma2 = np.diag(Sigma_robust)
all_marginals = np.zeros((n_total, 4))
for i in range(n_total):
    all_marginals[i] = (Z[i] - mu_robust)**2 / sigma2

high_risk_idx = np.where(S_all >= tau_star)[0]
print(f'高风险样本 (S >= τ*): {len(high_risk_idx)}')
if len(high_risk_idx) > 0:
    dominant = np.argmax(all_marginals[high_risk_idx], axis=1)
    for k in range(4):
        count = (dominant == k).sum()
        print(f'  {z_labels[k]}号: {count} ({count/len(dominant)*100:.1f}%)')

# ===== 6. 三分类 =====
print('\n' + '='*60)
print('6. 三分类')
print('='*60)

theta_min = 1.0
high_mask = (S_all >= tau_star) & (np.max(all_marginals, axis=1) > theta_min)
uncertain_mask = (S_all >= tau_lower) & (S_all < tau_star)
high_but_no_peak = (S_all >= tau_star) & ~high_mask
uncertain_mask = uncertain_mask | high_but_no_peak
low_mask = S_all < tau_lower

print(f'高风险异常: {high_mask.sum()} ({high_mask.sum()/n_total*100:.1f}%)')
print(f'不确定/需复检: {uncertain_mask.sum()} ({uncertain_mask.sum()/n_total*100:.1f}%)')
print(f'低风险: {low_mask.sum()} ({low_mask.sum()/n_total*100:.1f}%)')

# ===== 7. 41条样本 =====
print('\n' + '='*60)
print('7. 41条Z异常无标签样本')
print('='*60)
mismatch_idx = df_clean[df_clean['flag_z_ab_mismatch'] == 1].index
S_mismatch = S_all[mismatch_idx]
print(f'S评分: mean={S_mismatch.mean():.2f} std={S_mismatch.std():.2f}')
in_high = (S_mismatch >= tau_star).sum()
in_unc = ((S_mismatch >= tau_lower) & (S_mismatch < tau_star)).sum()
in_low = (S_mismatch < tau_lower).sum()
print(f'高风险:{in_high}  不确定:{in_unc}  低风险:{in_low}')

# ===== 制图 =====
# 图1: GC校正
print('\n--- 制图: GC校正效果 ---')
fig, axes = plt.subplots(4, 2, figsize=(12, 14))
z_raw_cols = ['13号染色体的Z值','18号染色体的Z值','21号染色体的Z值','X染色体的Z值']
gc = df['GC含量'].values
for k in range(4):
    ax = axes[k, 0]
    ax.scatter(gc[mask_normal], df[z_raw_cols[k]].values[mask_normal], s=6, alpha=0.4, color='#2ca02c', label='正常')
    ax.scatter(gc[mask_abnormal], df[z_raw_cols[k]].values[mask_abnormal], s=10, alpha=0.7, color='#d62728', label='异常')
    ax.set_ylabel(f'{z_labels[k]}号Z值'); ax.set_title(f'{z_labels[k]}号 — 校正前')
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    if k == 0: ax.legend(fontsize=7)
    ax = axes[k, 1]
    ax.scatter(gc[mask_normal], Z[mask_normal, k], s=6, alpha=0.4, color='#2ca02c')
    ax.scatter(gc[mask_abnormal], Z[mask_abnormal, k], s=10, alpha=0.7, color='#d62728')
    ax.set_ylabel(f'{z_labels[k]}号Z值(校正后)'); ax.set_title(f'{z_labels[k]}号 — 校正后')
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
plt.suptitle('GC Bias 校正效果', fontsize=13, y=1.01)
plt.tight_layout()
for d in [fig_dir, chart_dir]:
    plt.savefig(os.path.join(d, 'sub4-gc-correction.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('已保存')

# 图2: 评分分布
print('--- 制图: 评分分布 ---')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
bins = np.linspace(0, max(S_all.max(), 25), 50)
ax.hist(S_normal, bins=bins, alpha=0.6, color='#2ca02c', label=f'正常(n={n_normal})')
ax.hist(S_abnormal, bins=bins, alpha=0.6, color='#d62728', label=f'异常(n={n_abnormal})')
ax.axvline(x=tau_star, color='#333', ls='--', lw=1.5, label=f'$\tau^*$={tau_star:.1f}')
ax.axvline(x=tau_lower, color='#999', ls=':', lw=1, label=f'$\\tau_L$={tau_lower:.1f}')
ax.set_xlabel('马氏距离评分 S'); ax.set_ylabel('频数'); ax.set_title('评分分布')
ax.legend(fontsize=8)
ax = axes[1]
ax.boxplot([S_normal, S_abnormal], labels=['正常','异常'], patch_artist=True,
           boxprops=dict(facecolor='#1f77b4', alpha=0.5))
ax.set_ylabel('S'); ax.set_title('评分箱线图')
plt.tight_layout()
for d in [fig_dir, chart_dir]:
    plt.savefig(os.path.join(d, 'sub4-score-distribution.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('已保存')

# 图3: ROC
print('--- 制图: ROC曲线 ---')
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC={roc_auc:.3f})')
ax.plot([0,1],[0,1], 'k--', lw=0.8, alpha=0.5, label='随机')
ax.scatter([fpr[idx_best]], [tpr[idx_best]], color='#d62728', s=80, zorder=5,
           label=f'$\\tau^*$={tau_star:.1f} (Youden)')
ax.scatter([fpr[idx_tpr90]], [tpr[idx_tpr90]], color='#ff7f0e', s=60, zorder=5,
           label=f'$\\tau_L$={tau_lower:.1f} (TPR>=0.9)', marker='s')
ax.set_xlabel('假阳性率'); ax.set_ylabel('真阳性率')
ax.set_title(f'ROC曲线 (去重标注集, n={len(y_true)})')
ax.legend(fontsize=8, loc='lower right'); ax.set_xlim(0,1); ax.set_ylim(0,1.05)
plt.tight_layout()
for d in [fig_dir, chart_dir]:
    plt.savefig(os.path.join(d, 'sub4-roc-curve.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('已保存')

# 图4: 边际贡献
print('--- 制图: 边际贡献 ---')
if len(high_risk_idx) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(30, len(high_risk_idx))
    top_idx = high_risk_idx[np.argsort(-S_all[high_risk_idx])[:top_n]]
    x = np.arange(top_n); width = 0.18
    colors_bar = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    for k in range(4):
        ax.bar(x+k*width, all_marginals[top_idx,k], width, color=colors_bar[k], alpha=0.8, label=z_labels[k])
    ax.set_xticks(x+1.5*width)
    ax.set_xticklabels([f'#{df.iloc[i]["孕妇代码"]}' for i in top_idx], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('边际贡献'); ax.set_title('高分样本染色体边际贡献 (Top 30)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    for d in [fig_dir, chart_dir]:
        plt.savefig(os.path.join(d, 'sub4-marginal-contribution.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print('已保存')

# 图5: 标注偏差
print('--- 制图: 标注偏差 ---')
bias_features = ['孕妇BMI','年龄','体重']
bias_labels = ['BMI','年龄(岁)','体重(kg)']
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for j, (feat, lbl) in enumerate(zip(bias_features, bias_labels)):
    ax = axes[j]
    ax.boxplot([df[feat].values, df.loc[mask_abnormal, feat].values, df.loc[mask_normal, feat].values],
               labels=['全量(605)',f'异常({n_abnormal})',f'正常({n_normal})'])
    ax.set_ylabel(lbl); ax.set_title(lbl)
plt.suptitle('标注样本偏差分析', fontsize=12, y=1.02)
plt.tight_layout()
for d in [fig_dir, chart_dir]:
    plt.savefig(os.path.join(d, 'sub4-label-bias.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('已保存')

# 图6: 最终分类
print('--- 制图: 最终分类 ---')
fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(Z[low_mask,1], Z[low_mask,3], s=10, alpha=0.4, color='#2ca02c', label='低风险')
ax.scatter(Z[uncertain_mask,1], Z[uncertain_mask,3], s=12, alpha=0.5, color='#ff7f0e', label='不确定')
ax.scatter(Z[high_mask,1], Z[high_mask,3], s=14, alpha=0.7, color='#d62728', label='高风险')
ax.scatter(Z[mask_abnormal,1], Z[mask_abnormal,3], s=45, edgecolors='black',
           facecolors='none', lw=1.5, label='标注异常')
ax.set_xlabel('18号Z值(校正后)'); ax.set_ylabel('X Z值(校正后)')
ax.set_title('最终分类 (18号 vs X)'); ax.legend(fontsize=8)
plt.tight_layout()
for d in [fig_dir, chart_dir]:
    plt.savefig(os.path.join(d, 'sub4-final-classification.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('已保存')

# ===== LaTeX表 =====
with open(os.path.join(tables_dir, 'sub4-performance.tex'), 'w', encoding='utf-8') as f:
    f.write('\\begin{tabular}{lr}\n\\toprule\n')
    f.write('指标 & 值 \\\\\n\\midrule\n')
    f.write(f'AUC & {roc_auc:.3f} \\\\\n')
    f.write(f'最优阈值 $\\tau^*$ & {tau_star:.1f} \\\\\n')
    f.write(f'准确率 & {acc:.1%} \\\\\n')
    f.write(f'精确率 & {prec:.1%} \\\\\n')
    f.write(f'召回率 & {rec:.1%} \\\\\n')
    f.write(f'F1 & {f1:.3f} \\\\\n')
    f.write('\\midrule\n')
    f.write(f'TP & {tp} \\\\\nFP & {fp} \\\\\nFN & {fn} \\\\\nTN & {tn} \\\\\n')
    f.write('\\midrule\n')
    hp = high_mask.sum()/n_total*100
    up = uncertain_mask.sum()/n_total*100
    lp = low_mask.sum()/n_total*100
    f.write(f'高风险 & {high_mask.sum()} ({hp:.0f}\\%) \\\\\n')
    f.write(f'不确定 & {uncertain_mask.sum()} ({up:.0f}\\%) \\\\\n')
    f.write(f'低风险 & {low_mask.sum()} ({lp:.0f}\\%) \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')
print('已保存: sub4-performance.tex')

with open(os.path.join(tables_dir, 'sub4-covariance.tex'), 'w', encoding='utf-8') as f:
    f.write('\\begin{tabular}{lrrrr}\n\\toprule\n')
    f.write('& 13号 & 18号 & 21号 & X \\\\\n\\midrule\n')
    for k, name in enumerate(z_labels):
        row = name
        for j in range(4):
            if j == k: row += f' & {Sigma_robust[k,j]:.4f}'
            else:
                r = Sigma_robust[k,j]/np.sqrt(Sigma_robust[k,k]*Sigma_robust[j,j])
                row += f' & {r:.3f}'
        row += ' \\\\\n'; f.write(row)
    f.write('\\bottomrule\n\\end{tabular}\n')
print('已保存: sub4-covariance.tex')

import shutil; shutil.copy(__file__, os.path.join(code_dir, 'sub4-model.py'))

print('\n' + '='*60)
print('问题4 完成')
print('='*60)
print(f'AUC={roc_auc:.3f}  τ*={tau_star:.1f}  召回率={rec:.1%}  精确率={prec:.1%}')
print(f'高风险:{high_mask.sum()}  不确定:{uncertain_mask.sum()}  低风险:{low_mask.sum()}')
print(f'41条Z异常无标签 -> 高风险:{in_high}  不确定:{in_unc}  低风险:{in_low}')

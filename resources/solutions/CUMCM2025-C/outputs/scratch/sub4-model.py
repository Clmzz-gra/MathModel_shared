#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
问题4 Fisher LDA 实现（按报告 iter-04-sub4.tex 的描述重建）
三层架构：GC校正 → Fisher LDA按染色体拆分 → 高灵敏度阈值校准
"""
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

cache_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\data'

# ===== 0. 加载与合并 =====
df = pd.read_pickle(os.path.join(cache_dir, '2025C-sub4-preprocessed.pkl'))
df_clean = pd.read_pickle(os.path.join(cache_dir, '2025C-female-clean.pkl'))

# 合并染色体特异性 GC 含量
gc_cols_map = {
    '13号染色体的GC含量': 'GC_13',
    '18号染色体的GC含量': 'GC_18',
    '21号染色体的GC含量': 'GC_21',
}
for orig, new in gc_cols_map.items():
    df[new] = df_clean[orig].values

# X染色体浓度从 clean 数据取
df['X_conc'] = df_clean['X染色体浓度'].values

print(f'加载数据: {df.shape[0]} 行 x {df.shape[1]} 列')

# Z值（已校正）
z_cols = ['Z13_corrected', 'Z18_corrected', 'Z21_corrected', 'ZX_corrected']
z_labels = ['13', '18', '21', 'X']
Z = df[z_cols].values

# 标注
ab = df['AB_异常'].values
n_total = len(Z)
mask_normal = ab == 0
mask_abnormal = ab == 1
print(f'正常: {mask_normal.sum()}  异常: {mask_abnormal.sum()}')

# ===== 1. 特征工程（按报告描述） =====
n = n_total

# Z值对比特征: diff_k = z̃(k) - median({z̃(j)}_{j≠k})
Z_diff = np.zeros((n, 4))
for k in range(4):
    other = [j for j in range(4) if j != k]
    Z_diff[:, k] = Z[:, k] - np.median(Z[:, other], axis=1)

# GC跨染色体差异
GC_18_13 = df['GC_18'].values - df['GC_13'].values
GC_21_13 = df['GC_21'].values - df['GC_13'].values
GC_18_21 = df['GC_18'].values - df['GC_21'].values

# 交互项
bmi = df['孕妇BMI'].values
age = df['年龄'].values
gw = df['孕周_数值'].values
BMI_x_Z18 = bmi * Z[:, 1]
Age_x_Z18 = age * Z[:, 1]

# 测序质控指标
filter_rate = df['被过滤掉读段数的比例'].values
dup_rate = df['重复读段的比例'].values
map_rate = df['在参考基因组上比对的比例'].values

# X染色体浓度
X_conc = df['X_conc'].values

# 构建综合特征矩阵（报告表4：含Z值、GC、QC、孕妇特征、派生特征）
# 对每条染色体构建特定特征子集
feature_sets = {
    # 13号染色体特征
    13: np.column_stack([
        Z[:, 0],           # Z13_corrected
        Z_diff[:, 0],      # Z13 vs others contrast
        df['GC_13'].values,
        GC_18_13,
        GC_21_13,
        filter_rate,
        dup_rate,
        map_rate,
        X_conc,
        bmi,
        age,
        gw,
    ]),
    # 18号染色体特征
    18: np.column_stack([
        Z[:, 1],           # Z18_corrected
        Z_diff[:, 1],      # Z18 vs others contrast
        df['GC_18'].values,
        GC_18_13,
        GC_18_21,
        filter_rate,
        dup_rate,
        map_rate,
        X_conc,
        bmi,
        age,
        gw,
        BMI_x_Z18,
        Age_x_Z18,
    ]),
    # 21号染色体特征
    21: np.column_stack([
        Z[:, 2],           # Z21_corrected
        Z_diff[:, 2],      # Z21 vs others contrast
        df['GC_21'].values,
        GC_21_13,
        GC_18_21,
        filter_rate,
        dup_rate,
        map_rate,
        X_conc,
        bmi,
        age,
        gw,
    ]),
}

feature_names = {
    13: ['Z13_c', 'Z13_contrast', 'GC_13', 'GC_18-13', 'GC_21-13',
         'filter_rate', 'dup_rate', 'map_rate', 'X_conc', 'BMI', 'age', 'gw'],
    18: ['Z18_c', 'Z18_contrast', 'GC_18', 'GC_18-13', 'GC_18-21',
         'filter_rate', 'dup_rate', 'map_rate', 'X_conc', 'BMI', 'age', 'gw',
         'BMI*Z18', 'Age*Z18'],
    21: ['Z21_c', 'Z21_contrast', 'GC_21', 'GC_21-13', 'GC_18-21',
         'filter_rate', 'dup_rate', 'map_rate', 'X_conc', 'BMI', 'age', 'gw'],
}

# ===== 2. 按孕妇去重（取首次检测） =====
date_map = df_clean[['孕妇代码', '检测日期_std']].drop_duplicates('孕妇代码')
df['孕妇代码'] = df_clean['孕妇代码'].values
df['检测日期_std'] = date_map.set_index('孕妇代码').loc[df['孕妇代码'], '检测日期_std'].values

# 标注样本去重
mask_labeled = mask_normal | mask_abnormal
labeled = df[mask_labeled].copy()
labeled_sorted = labeled.sort_values('检测日期_std')
labeled_dedup = labeled_sorted.drop_duplicates('孕妇代码', keep='first')

print(f'\n去重前: {len(labeled)}  去重后: {len(labeled_dedup)}')
print(f'  异常: {(labeled_dedup["AB_异常"]==1).sum()}  正常: {(labeled_dedup["AB_异常"]==0).sum()}')

# 获取去重后的索引映射到原始数组
dedup_idx = labeled_dedup.index.values
y_true = labeled_dedup['AB_异常'].values.astype(float)

n_dedup = len(dedup_idx)
print(f'去重后样本数: {n_dedup}')

# ===== 3. 确定每条染色体异常的样本（用于训练） =====
# 从标注信息确定异常类型
ab_types = df_clean.loc[dedup_idx, '染色体的非整倍体'].values
print('\n异常类型分布:')
ab_types_str = np.array([str(t) if not pd.isna(t) else 'nan' for t in ab_types])
for t in np.unique(ab_types_str):
    print(f'  {t}: {(ab_types_str==t).sum()}')

# 按染色体定义异常标签
# T13异常: 含T13, T13T18, T13T21
# T18异常: 含T18, T13T18, T18T21
# T21异常: 含T21, T13T21, T18T21
def has_type(ab_str, target):
    if pd.isna(ab_str) or ab_str == '':
        return False
    return target in str(ab_str)

is_t13 = np.array([has_type(s, 'T13') for s in ab_types])
is_t18 = np.array([has_type(s, 'T18') for s in ab_types])
is_t21 = np.array([has_type(s, 'T21') for s in ab_types])

print(f'\nT13异常: {is_t13.sum()}  T18异常: {is_t18.sum()}  T21异常: {is_t21.sum()}')

# ===== 4. Fisher LDA 实现 =====
def fisher_lda_fit(X, y):
    """Fisher LDA: w = Sw^{-1} (mu1 - mu0)"""
    X0 = X[y == 0]
    X1 = X[y == 1]
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    
    # 类内散度 Sw = (n0-1)*S0 + (n1-1)*S1
    S0 = np.cov(X0, rowvar=False) if len(X0) > 1 else np.eye(X.shape[1])
    S1 = np.cov(X1, rowvar=False) if len(X1) > 1 else np.eye(X.shape[1])
    Sw = (len(X0) - 1) * S0 + (len(X1) - 1) * S1
    Sw = Sw / (len(X) - 2)  # normalize
    
    # 正则化（处理小样本+奇异矩阵）
    reg = 0.01 * np.trace(Sw) / Sw.shape[0]
    Sw_reg = Sw + reg * np.eye(Sw.shape[0])
    
    try:
        Sw_inv = np.linalg.inv(Sw_reg)
    except np.linalg.LinAlgError:
        Sw_inv = np.linalg.pinv(Sw_reg)
    
    w = Sw_inv @ (mu1 - mu0)
    # 阈值取两类均值中点
    s0 = X0 @ w
    s1 = X1 @ w
    threshold = (s0.mean() + s1.mean()) / 2
    
    return w, threshold

def fisher_lda_score(X, w):
    return X @ w

# ===== 5. 为每条染色体训练 Fisher LDA =====
print('\n' + '=' * 70)
print('5. Fisher LDA 按染色体训练')
print('=' * 70)

chr_targets = {13: is_t13, 18: is_t18, 21: is_t21}
lda_models = {}
all_scores = np.zeros((n_dedup, 3))

for k in [13, 18, 21]:
    X_k = feature_sets[k][dedup_idx]
    y_k = chr_targets[k].astype(float)
    n_pos = y_k.sum()
    
    print(f'\n--- 染色体{k}号 ---')
    print(f'  正常: {(y_k==0).sum()}  异常: {n_pos}')
    
    if n_pos < 2:
        print(f'  异常样本不足，跳过')
        continue
    
    # Z-score标准化
    X_mean = X_k.mean(axis=0)
    X_std = X_k.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_scaled = (X_k - X_mean) / X_std
    
    w, thresh = fisher_lda_fit(X_scaled, y_k)
    scores = fisher_lda_score(X_scaled, w)
    
    # 查看单特征 Cohen's d
    print(f'  特征 Cohen\'s d:')
    for j, name in enumerate(feature_names[k]):
        d = (X_k[y_k==1, j].mean() - X_k[y_k==0, j].mean()) / \
            np.sqrt((X_k[y_k==1, j].var() + X_k[y_k==0, j].var()) / 2) if X_k[y_k==1, j].var() + X_k[y_k==0, j].var() > 0 else 0
        if abs(d) > 0.3:
            print(f'    {name:20s}: d={d:+.3f}')
    
    lda_models[k] = {'w': w, 'threshold': thresh, 'X_mean': X_mean, 'X_std': X_std}
    all_scores[:, {13: 0, 18: 1, 21: 2}[k]] = scores

# 综合评分 = max(s13, s18, s21) — 按报告公式
y_score = np.max(all_scores, axis=1)

# ===== 6. ROC 分析 =====
print('\n' + '=' * 70)
print('6. ROC 分析（组合评分 max(s13, s18, s21)）')
print('=' * 70)

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
roc_auc = np.trapezoid(tpr, fpr)
print(f'AUC = {roc_auc:.4f}')

# Youden
youden = tpr - fpr
idx_best = np.argmax(youden)
tau_star = thresholds[idx_best]
print(f'τ* = {tau_star:.4f} (Youden={youden[idx_best]:.4f})')
print(f'  TPR={tpr[idx_best]:.4f}  FPR={fpr[idx_best]:.4f}')

# 高灵敏度阈值（TPR >= 90%，取最小阈值 = 最严格满足条件的）
idx_tpr90 = np.where(tpr >= 0.90)[0]
if len(idx_tpr90) > 0:
    idx_tpr90 = idx_tpr90[0]  # 取第一个满足的（最高阈值，最严格）
    tau_lower = thresholds[idx_tpr90]
    tpr_tau_lower = tpr[idx_tpr90]
    fpr_tau_lower = fpr[idx_tpr90]
    print(f'\n高灵敏度阈值 (TPR>=90%):')
    print(f'  τ = {tau_lower:.4f}')
    print(f'  TPR={tpr_tau_lower:.4f}  FPR={fpr_tau_lower:.4f}')
    print(f'  特异度={1-fpr_tau_lower:.4f}')
else:
    print(f'\n无法达到 TPR>=90% (最大TPR={tpr[-1]:.4f})')
    tau_lower = thresholds[-1]

# 打印关键ROC点
print('\n关键ROC点:')
for target_tpr in [0.50, 0.75, 0.90, 1.00]:
    candidates = np.where(tpr >= target_tpr)[0]
    if len(candidates) > 0:
        idx = candidates[0]
        print(f'  TPR>={target_tpr:.0%}: τ={thresholds[idx]:.4f}, TPR={tpr[idx]:.4f}, FPR={fpr[idx]:.4f}, 特异度={1-fpr[idx]:.4f}')

# ===== 7. 混淆矩阵 =====
print('\n' + '=' * 70)
print('7. 混淆矩阵 (Youden)')
print('=' * 70)

y_pred = (y_score >= tau_star).astype(float)
tp = ((y_true == 1) & (y_pred == 1)).sum()
fp = ((y_true == 0) & (y_pred == 1)).sum()
fn = ((y_true == 1) & (y_pred == 0)).sum()
tn = ((y_true == 0) & (y_pred == 0)).sum()

print(f'         预测异常  预测正常')
print(f'实际异常    {tp:3d}        {fn:3d}')
print(f'实际正常    {fp:3d}        {tn:3d}')
print(f'\n召回率={tp/(tp+fn):.4f}  特异度={tn/(tn+fp):.4f}  精确率={tp/(tp+fp):.4f}')
print(f'漏检={fn}人')

# ===== 8. 高灵敏度混淆矩阵 =====
print('\n' + '=' * 70)
print('8. 混淆矩阵 (高灵敏度 TPR>=90%)')
print('=' * 70)

y_pred_hi = (y_score >= tau_lower).astype(float)
tp_hi = ((y_true == 1) & (y_pred_hi == 1)).sum()
fp_hi = ((y_true == 0) & (y_pred_hi == 1)).sum()
fn_hi = ((y_true == 1) & (y_pred_hi == 0)).sum()
tn_hi = ((y_true == 0) & (y_pred_hi == 0)).sum()

print(f'         预测异常  预测正常')
print(f'实际异常    {tp_hi:3d}        {fn_hi:3d}')
print(f'实际正常    {fp_hi:3d}        {tn_hi:3d}')
print(f'\n召回率={tp_hi/(tp_hi+fn_hi):.4f}  特异度={tn_hi/(tn_hi+fp_hi):.4f}  精确率={tp_hi/(tp_hi+fp_hi):.4f}')
print(f'假阳性率={fp_hi/(fp_hi+tn_hi):.4f}')
print(f'漏检={fn_hi}人')

# ===== 9. 对比报告声称值 =====
print('\n' + '=' * 70)
print('9. 对比报告声称值')
print('=' * 70)
print(f'{"指标":20s} {"本次Fisher LDA":>15s} {"报告声称":>15s} {"偏差":>10s}')
print('-' * 60)
print(f'{"AUC":20s} {roc_auc:15.4f} {"0.585":>15s} {roc_auc-0.585:+10.4f}')
print(f'{"Youden召回率":20s} {tp/(tp+fn):14.1%} {"50.0%":>15s} {tp/(tp+fn)-0.5:+10.1%}')
print(f'{"Youden特异度":20s} {tn/(tn+fp):14.1%} {"96.3%":>15s} {tn/(tn+fp)-0.963:+10.1%}')
print(f'{"高灵敏召回率":20s} {tp_hi/(tp_hi+fn_hi):14.1%} {"91.7%":>15s} {tp_hi/(tp_hi+fn_hi)-0.917:+10.1%}')
print(f'{"高灵敏特异度":20s} {tn_hi/(tn_hi+fp_hi):14.1%} {"23.0%":>15s} {tn_hi/(tn_hi+fp_hi)-0.23:+10.1%}')

# ===== 10. 按异常类型分解 =====
print('\n' + '=' * 70)
print('10. 按异常类型检出情况 (高灵敏度)')
print('=' * 70)
for group_name, group_mask in [('T18 (含组合)', is_t18), ('T13 (含组合)', is_t13), ('T21 (含组合)', is_t21)]:
    group_idx = np.where(group_mask)[0]
    if len(group_idx) > 0:
        detected = y_pred_hi[group_idx].sum()
        print(f'  {group_name:15s}: {len(group_idx)}人, 检出{int(detected)}人, 漏检{len(group_idx)-int(detected)}人')

print('\n完成。')

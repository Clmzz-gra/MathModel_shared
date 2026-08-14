#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A类共享事实验证：熵权法 vs Soft-Voting vs Fisher LDA（子问题4）
纯 numpy/pandas/scipy，不依赖 sklearn
"""
import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from scipy import stats

cache_dir = r'E:\MathModel\problems\2025\C题\2025C题测试\outputs\data'

# ============================================================
# 0. 加载数据
# ============================================================
print("=" * 70)
print("0. 数据加载与去重")
print("=" * 70)

sub4 = pd.read_pickle(os.path.join(cache_dir, '2025C-sub4-preprocessed.pkl'))
clean = pd.read_pickle(os.path.join(cache_dir, '2025C-female-clean.pkl'))

# 合并染色体特异性 GC 含量
gc_cols_map = {
    '13号染色体的GC含量': 'GC_13',
    '18号染色体的GC含量': 'GC_18',
    '21号染色体的GC含量': 'GC_21',
}
for orig, new in gc_cols_map.items():
    sub4[new] = clean[orig].values

sub4['X_conc'] = clean['X染色体浓度'].values
sub4['孕妇代码'] = clean['孕妇代码'].values

# 按孕妇去重（取最早检测）
date_map = clean[['孕妇代码', '检测日期_std']].drop_duplicates('孕妇代码')
sub4['检测日期_std'] = date_map.set_index('孕妇代码').loc[sub4['孕妇代码'], '检测日期_std'].values

mask_labeled = (sub4['AB_异常'] == 0) | (sub4['AB_异常'] == 1)
labeled = sub4[mask_labeled].copy()
labeled_sorted = labeled.sort_values('检测日期_std')
labeled_dedup = labeled_sorted.drop_duplicates('孕妇代码', keep='first')

dedup_idx = labeled_dedup.index.values
y_true = labeled_dedup['AB_异常'].values.astype(float)

print(f"去重后: {len(labeled_dedup)} 人, 正常: {(y_true==0).sum()}, 异常: {(y_true==1).sum()}")

# ============================================================
# 1. 熵权法 (Entropy Weight Method)
# ============================================================
print("\n" + "=" * 70)
print("1. 熵权法")
print("=" * 70)

def entropy_weight(X):
    """
    熵权法：计算每个特征的客观权重
    X: (n_samples, n_features)，值越大越异常（正向指标）
    返回: weights (n_features,)
    """
    n, m = X.shape
    # Min-Max 归一化到 [0.001, 1] 避免 log(0)
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range < 1e-10] = 1.0
    P = (X - x_min) / x_range
    P = np.clip(P, 0.001, 1.0)  # 避免 0

    # 归一化概率
    P_sum = P.sum(axis=0)
    P_norm = P / P_sum

    # 信息熵 e_j = -k * Σ p_ij * ln(p_ij), k = 1/ln(n)
    k = 1.0 / np.log(n)
    e = -k * np.sum(P_norm * np.log(P_norm), axis=0)

    # 差异系数 d_j = 1 - e_j
    d = 1 - e

    # 权重 w_j = d_j / Σ d_j
    w = d / d.sum()

    return w, e, d

# 构建特征矩阵：用全体 605 条计算熵权（不依赖标注）
# 特征：Z_corrected(4) + GC(3) + 质控(3) + X_conc + BMI + age + gw
feature_cols_all = [
    'Z13_corrected', 'Z18_corrected', 'Z21_corrected', 'ZX_corrected',
    'GC_13', 'GC_18', 'GC_21',
    '被过滤掉读段数的比例', '重复读段的比例', '在参考基因组上比对的比例',
    'X_conc', '孕妇BMI', '年龄', '孕周_数值',
]
feature_labels = [
    'Z13_c', 'Z18_c', 'Z21_c', 'ZX_c',
    'GC_13', 'GC_18', 'GC_21',
    'filter_rate', 'dup_rate', 'map_rate',
    'X_conc', 'BMI', 'age', 'gw',
]

X_all = sub4[feature_cols_all].values.astype(float)
# 对部分特征取绝对值（Z值、filter_rate、dup_rate — 偏离0越远越异常）
abs_cols = [0, 1, 2, 3, 7, 8]  # Z13-ZX, filter_rate, dup_rate
for c in abs_cols:
    X_all[:, c] = np.abs(X_all[:, c])

# fill NaN
X_all = np.nan_to_num(X_all, nan=0.0)

w, e, d = entropy_weight(X_all)

print(f"\n{'特征':<18s} {'信息熵':>8s} {'差异系数':>8s} {'权重':>8s}")
print("-" * 50)
for i, label in enumerate(feature_labels):
    print(f"{label:<18s} {e[i]:8.4f} {d[i]:8.4f} {w[i]:8.4f}")

# 计算综合评分
S_entropy_all = X_all @ w

# 在标注样本上评估
S_entropy = S_entropy_all[dedup_idx]

# 分离正异常
s_normal = S_entropy[y_true == 0]
s_abnormal = S_entropy[y_true == 1]
print(f"\n综合评分 (标注样本): 正常 mean={s_normal.mean():.4f} std={s_normal.std():.4f}")
print(f"                    异常 mean={s_abnormal.mean():.4f} std={s_abnormal.std():.4f}")

# Cohen's d
def cohens_d(x1, x2):
    m1, m2 = np.mean(x1), np.mean(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((len(x1)-1)*v1 + (len(x2)-1)*v2) / (len(x1)+len(x2)-2))
    if pooled_std < 1e-12:
        return 0.0
    return (m2 - m1) / pooled_std

d_entropy = cohens_d(s_normal, s_abnormal)
print(f"Cohen's d = {d_entropy:.4f}")

# Welch t-test
t_stat, t_p = stats.ttest_ind(s_abnormal, s_normal, equal_var=False)
print(f"Welch t-test: t={t_stat:.4f}, p={t_p:.6f}")

# ROC AUC
def roc_auc_manual(y_true, y_score):
    order = np.argsort(-y_score)
    y_sort = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    tpr = np.cumsum(y_sort) / n_pos
    fpr = np.cumsum(1 - y_sort) / n_neg
    return np.trapezoid(tpr, fpr)

auc_entropy = roc_auc_manual(y_true, S_entropy)
print(f"AUC = {auc_entropy:.4f}")

# ============================================================
# 2. Soft-Voting 集成
# ============================================================
print("\n" + "=" * 70)
print("2. Soft-Voting 集成")
print("=" * 70)

# 异常类型分布
ab_types = clean.loc[dedup_idx, '染色体的非整倍体'].values
def has_type(ab_str, target):
    if pd.isna(ab_str) or ab_str == '':
        return False
    return target in str(ab_str)

is_t13 = np.array([has_type(s, 'T13') for s in ab_types])
is_t18 = np.array([has_type(s, 'T18') for s in ab_types])
is_t21 = np.array([has_type(s, 'T21') for s in ab_types])
print(f"T13: {is_t13.sum()}, T18: {is_t18.sum()}, T21: {is_t21.sum()}")

def fisher_lda_fit(X, y):
    """Fisher LDA with Tikhonov regularization"""
    X0 = X[y == 0]; X1 = X[y == 1]
    mu0 = X0.mean(axis=0); mu1 = X1.mean(axis=0)
    S0 = np.cov(X0, rowvar=False) if len(X0) > 1 else np.eye(X.shape[1])
    S1 = np.cov(X1, rowvar=False) if len(X1) > 1 else np.eye(X.shape[1])
    Sw = ((len(X0)-1)*S0 + (len(X1)-1)*S1) / (len(X)-2)
    reg = 0.01 * np.trace(Sw) / Sw.shape[0]
    Sw_reg = Sw + reg * np.eye(Sw.shape[0])
    try:
        Sw_inv = np.linalg.inv(Sw_reg)
    except:
        Sw_inv = np.linalg.pinv(Sw_reg)
    w = Sw_inv @ (mu1 - mu0)
    s0 = X0 @ w; s1 = X1 @ w
    threshold = (s0.mean() + s1.mean()) / 2
    return w, threshold

# 为每条染色体构建 Fisher LDA 子分类器
# 特征：该染色体Z_corrected + GC + 质控 + X_conc + BMI
chr_config = {
    13: {'z_idx': 0, 'gc_col': 'GC_13', 'y': is_t13},
    18: {'z_idx': 1, 'gc_col': 'GC_18', 'y': is_t18},
    21: {'z_idx': 2, 'gc_col': 'GC_21', 'y': is_t21},
}

# 构建全量特征矩阵用于投影
Z_vals = sub4[['Z13_corrected', 'Z18_corrected', 'Z21_corrected', 'ZX_corrected']].values
GC_vals = sub4[['GC_13', 'GC_18', 'GC_21']].values
qc_vals = sub4[['被过滤掉读段数的比例', '重复读段的比例', '在参考基因组上比对的比例']].values
X_conc_vals = sub4['X_conc'].values.reshape(-1, 1)
bmi_vals = sub4['孕妇BMI'].values.reshape(-1, 1)
age_vals = sub4['年龄'].values.reshape(-1, 1)
gw_vals = sub4['孕周_数值'].values.reshape(-1, 1)

chr_scores_full = np.zeros((len(sub4), 3))  # scores for all 605
chr_w = {}

for idx, k in enumerate([13, 18, 21]):
    cfg = chr_config[k]
    # 构建特征矩阵
    X_feat = np.column_stack([
        Z_vals[:, cfg['z_idx']],       # Z_corrected for this chr
        GC_vals[:, idx],                # GC for this chr
        qc_vals[:, 0],                  # filter_rate
        qc_vals[:, 1],                  # dup_rate
        qc_vals[:, 2],                  # map_rate
        X_conc_vals.ravel(),
        bmi_vals.ravel(),
        age_vals.ravel(),
        gw_vals.ravel(),
    ])

    # 标注子集
    X_labeled = X_feat[dedup_idx]
    y_chr = cfg['y'].astype(float)
    n_pos = y_chr.sum()

    print(f"\n染色体{k}号子分类器: 正样本={int(n_pos)}")

    if n_pos < 2:
        print(f"  正样本不足，跳过")
        continue

    # Z-score 标准化
    X_mean = X_labeled.mean(axis=0)
    X_std = X_labeled.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_scaled = (X_labeled - X_mean) / X_std

    w_vec, thresh = fisher_lda_fit(X_scaled, y_chr)
    chr_w[k] = {'w': w_vec, 'threshold': thresh, 'X_mean': X_mean, 'X_std': X_std}

    # 全量投影
    X_full_scaled = (X_feat - X_mean) / X_std
    chr_scores_full[:, idx] = X_full_scaled @ w_vec

    # 评估
    scores_chr = chr_scores_full[dedup_idx, idx]
    auc_chr = roc_auc_manual(y_true, scores_chr)
    print(f"  子分类器 AUC (vs 全局异常标签) = {auc_chr:.4f}")

# Soft-voting: 取各染色体分数的加权平均/最大
chr_scores = chr_scores_full[dedup_idx]

# 计算各子分类器的权重（基于信息量/区分度）
valid_chrs = sorted([k for k in [13, 18, 21] if k in chr_w])
n_valid = len(valid_chrs)

if n_valid >= 2:
    # 方案A: 等权 soft-voting
    soft_vote_mean = np.mean(chr_scores[:, :n_valid], axis=1)
    auc_sv_mean = roc_auc_manual(y_true, soft_vote_mean)
    print(f"\nSoft-Voting (等权平均): AUC = {auc_sv_mean:.4f}")

    # 方案B: 加权 soft-voting（用各分类器的投影权重范数作为权重）
    w_norms = np.array([np.linalg.norm(chr_w[k]['w']) for k in sorted(valid_chrs)])
    w_norms = w_norms / w_norms.sum()
    soft_vote_weighted = np.sum(chr_scores[:, :n_valid] * w_norms, axis=1)
    auc_sv_weighted = roc_auc_manual(y_true, soft_vote_weighted)
    print(f"Soft-Voting (加权):      AUC = {auc_sv_weighted:.4f}")

    # 方案C: max voting
    soft_vote_max = np.max(chr_scores[:, :n_valid], axis=1)
    auc_sv_max = roc_auc_manual(y_true, soft_vote_max)
    print(f"Soft-Voting (max):       AUC = {auc_sv_max:.4f}")

    sv_best = max([(auc_sv_mean, 'mean', soft_vote_mean),
                   (auc_sv_weighted, 'weighted', soft_vote_weighted),
                   (auc_sv_max, 'max', soft_vote_max)],
                  key=lambda x: x[0])
    sv_best_auc, sv_best_name, sv_best_scores = sv_best
    print(f"\n最优 Soft-Voting: {sv_best_name}, AUC = {sv_best_auc:.4f}")
elif n_valid == 1:
    sv_best_scores = chr_scores[:, 0]
    sv_best_auc = roc_auc_manual(y_true, sv_best_scores)
    sv_best_name = f'chr{valid_chrs[0]}_only'
    print(f"\n仅1个子分类器可用: AUC = {sv_best_auc:.4f}")
else:
    sv_best_scores = np.zeros(len(y_true))
    sv_best_auc = 0.5
    sv_best_name = 'N/A'

# ============================================================
# 3. Fisher LDA 基线（复现）
# ============================================================
print("\n" + "=" * 70)
print("3. Fisher LDA 基线（复现 sub4-model.py）")
print("=" * 70)

# 特征工程（与 sub4-model.py 一致）
Z_sub4 = sub4[['Z13_corrected', 'Z18_corrected', 'Z21_corrected', 'ZX_corrected']].values

# Z对比特征
Z_diff = np.zeros((len(sub4), 4))
for k_idx in range(4):
    other = [j for j in range(4) if j != k_idx]
    Z_diff[:, k_idx] = Z_sub4[:, k_idx] - np.median(Z_sub4[:, other], axis=1)

GC_all = sub4[['GC_13', 'GC_18', 'GC_21']].values
GC_18_13 = GC_all[:, 1] - GC_all[:, 0]
GC_21_13 = GC_all[:, 2] - GC_all[:, 0]
GC_18_21 = GC_all[:, 1] - GC_all[:, 2]

bmi_arr = sub4['孕妇BMI'].values
age_arr = sub4['年龄'].values
gw_arr = sub4['孕周_数值'].values
filter_arr = sub4['被过滤掉读段数的比例'].values
dup_arr = sub4['重复读段的比例'].values
map_arr = sub4['在参考基因组上比对的比例'].values
xc_arr = sub4['X_conc'].values

# 按染色体构建特征
feature_sets_lda = {
    13: np.column_stack([
        Z_sub4[:, 0], Z_diff[:, 0], GC_all[:, 0],
        GC_18_13, GC_21_13,
        filter_arr, dup_arr, map_arr, xc_arr,
        bmi_arr, age_arr, gw_arr,
    ]),
    18: np.column_stack([
        Z_sub4[:, 1], Z_diff[:, 1], GC_all[:, 1],
        GC_18_13, GC_18_21,
        filter_arr, dup_arr, map_arr, xc_arr,
        bmi_arr, age_arr, gw_arr,
        bmi_arr * Z_sub4[:, 1], age_arr * Z_sub4[:, 1],
    ]),
    21: np.column_stack([
        Z_sub4[:, 2], Z_diff[:, 2], GC_all[:, 2],
        GC_21_13, GC_18_21,
        filter_arr, dup_arr, map_arr, xc_arr,
        bmi_arr, age_arr, gw_arr,
    ]),
}

fish_scores_full = np.zeros((len(sub4), 3))
for idx, k in enumerate([13, 18, 21]):
    X_k = feature_sets_lda[k]
    y_k = chr_config[k]['y'].astype(float)
    n_pos = y_k.sum()

    if n_pos < 2:
        continue

    X_k_labeled = X_k[dedup_idx]
    X_mean_k = X_k_labeled.mean(axis=0)
    X_std_k = X_k_labeled.std(axis=0)
    X_std_k[X_std_k < 1e-10] = 1.0
    X_scaled_k = (X_k_labeled - X_mean_k) / X_std_k

    w_k, th_k = fisher_lda_fit(X_scaled_k, y_k)
    X_full_scaled_k = (X_k - X_mean_k) / X_std_k
    fish_scores_full[:, idx] = X_full_scaled_k @ w_k

fish_scores = fish_scores_full[dedup_idx]
fish_max = np.max(fish_scores, axis=1)
auc_fish = roc_auc_manual(y_true, fish_max)
print(f"Fisher LDA (max): AUC = {auc_fish:.4f}")

# ============================================================
# 4. 综合对比
# ============================================================
print("\n" + "=" * 70)
print("4. 综合对比")
print("=" * 70)

def compute_roc_point(y_true, y_score, target_tpr=None):
    """计算指定点"""
    order = np.argsort(-y_score)
    y_sort = y_true[order]
    s_sort = y_score[order]
    n_pos = y_true.sum(); n_neg = len(y_true) - n_pos

    # 完整 ROC
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

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    th_arr = np.array(thresh_list)

    auc = np.trapezoid(tpr_arr, fpr_arr)

    # Youden
    youden = tpr_arr - fpr_arr
    youden_idx = np.argmax(youden)
    tau_youden = th_arr[youden_idx]
    tpr_youden = tpr_arr[youden_idx]
    fpr_youden = fpr_arr[youden_idx]

    # TPR >= 90%
    idx90 = np.where(tpr_arr >= 0.90)[0]
    if len(idx90) > 0:
        idx90 = idx90[0]
        tau_hi = th_arr[idx90]
        tpr_hi = tpr_arr[idx90]
        fpr_hi = fpr_arr[idx90]
    else:
        tau_hi = th_arr[-1]
        tpr_hi = tpr_arr[-1]
        fpr_hi = fpr_arr[-1]

    return auc, tau_youden, tpr_youden, fpr_youden, tau_hi, tpr_hi, fpr_hi

methods = [
    ('熵权法', S_entropy),
    ('Fisher LDA (现有)', fish_max),
]

if sv_best_name != 'N/A':
    methods.append((f'Soft-Voting ({sv_best_name})', sv_best_scores))

print(f"\n{'方法':<25s} {'AUC':>8s} {'Youden τ':>10s} {'TPR':>8s} {'FPR':>8s} {'高灵敏τ':>10s} {'TPR90':>8s} {'FPR90':>8s}")
print("-" * 95)

for name, scores in methods:
    auc, tau_y, tpr_y, fpr_y, tau_h, tpr_h, fpr_h = compute_roc_point(y_true, scores)
    print(f"{name:<25s} {auc:8.4f} {tau_y:10.4f} {tpr_y:7.1%} {fpr_y:7.1%} {tau_h:10.4f} {tpr_h:7.1%} {fpr_h:7.1%}")

# ============================================================
# 5. 熵权法 + Fisher LDA 融合探索
# ============================================================
print("\n" + "=" * 70)
print("5. 融合方案：熵权加权 + Fisher LDA 组合")
print("=" * 70)

# 方案A: 熵权法和Fisher LDA分数的加权平均
# 归一化两个分数
def minmax_scale(x):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-10:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

s_ent_norm = minmax_scale(S_entropy)
s_fish_norm = minmax_scale(fish_max)

# 尝试不同融合权重
for alpha in [0.3, 0.5, 0.7]:
    s_fused = alpha * s_ent_norm + (1 - alpha) * s_fish_norm
    auc_f = roc_auc_manual(y_true, s_fused)
    print(f"  α*熵权 + (1-α)*Fisher, α={alpha:.1f}: AUC = {auc_f:.4f}")

# 方案B: max of both
s_fused_max = np.maximum(s_ent_norm, s_fish_norm)
auc_fmax = roc_auc_manual(y_true, s_fused_max)
print(f"  max(熵权, Fisher): AUC = {auc_fmax:.4f}")

# ============================================================
# 6. 异常检出情况
# ============================================================
print("\n" + "=" * 70)
print("6. 12例异常样本的各方法评分")
print("=" * 70)

ab_idx = np.where(y_true == 1)[0]
ab_codes = labeled_dedup.iloc[ab_idx]['孕妇代码'].values
ab_types_disp = [str(clean.loc[clean['孕妇代码'] == c, '染色体的非整倍体'].values[0])
                 if len(clean.loc[clean['孕妇代码'] == c, '染色体的非整倍体'].values) > 0
                 and not pd.isna(clean.loc[clean['孕妇代码'] == c, '染色体的非整倍体'].values[0])
                 else '?' for c in ab_codes]

print(f"\n{'孕妇代码':<10} {'类型':<10} {'熵权评分':>10} {'Fisher评分':>10}", end='')
if sv_best_name != 'N/A':
    print(f" {'SV评分':>10}", end='')
print()

print("-" * (45 + (15 if sv_best_name != 'N/A' else 0)))

for i, idx in enumerate(ab_idx):
    print(f"{ab_codes[i]:<10} {ab_types_disp[i]:<10} {S_entropy[idx]:10.4f} {fish_max[idx]:10.4f}", end='')
    if sv_best_name != 'N/A':
        print(f" {sv_best_scores[idx]:10.4f}", end='')
    print()

# 熵权法对异常的区分度
print(f"\n熵权法异常检出: {(S_entropy[ab_idx] > np.median(S_entropy[y_true==0])).sum()}/{len(ab_idx)} 高于正常中位数")

print("\n" + "=" * 70)
print("A类验证完成")
print("=" * 70)

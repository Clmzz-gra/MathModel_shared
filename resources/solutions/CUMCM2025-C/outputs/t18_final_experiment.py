"""
T18 最终判别实验
纯 numpy/scipy/pandas，只读不写，全部打印。
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 数据加载与准备
# ============================================================
print("=" * 90)
print("0. 数据加载与准备")
print("=" * 90)

prep = pd.read_pickle(r'e:\MathModel\problems\2025\C题\outputs\data\2025C-sub4-preprocessed.pkl')
clean = pd.read_pickle(r'e:\MathModel\problems\2025\C题\outputs\data\2025C-female-clean.pkl')

print(f"preprocessed shape: {prep.shape}")
print(f"clean shape: {clean.shape}")

# 去重：按孕妇代码，保留第一条
prep = prep.drop_duplicates(subset='孕妇代码', keep='first').copy()
clean = clean.drop_duplicates(subset='孕妇代码', keep='first').copy()

print(f"去重后 preprocessed: {prep.shape[0]} rows")
print(f"去重后 clean: {clean.shape[0]} rows")

# 合并：以 preprocessed 为基础，left join clean 的额外列
extra_cols = [
    'X染色体浓度',
    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
    '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
    'GC含量', '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例',
    '孕妇BMI', '年龄', '体重', '身高', '孕周_数值', '染色体的非整倍体', 'AB_异常'
]
# 只取 clean 中存在的列
extra_cols = [c for c in extra_cols if c in clean.columns]
clean_sub = clean[['孕妇代码'] + extra_cols].copy()

# 为 clean 的重名列加后缀，避免冲突
clean_sub = clean_sub.rename(columns={
    '13号染色体的Z值': '13号染色体的Z值_raw_clean',
    '18号染色体的Z值': '18号染色体的Z值_raw_clean',
    '21号染色体的Z值': '21号染色体的Z值_raw_clean',
    'X染色体的Z值': 'X染色体的Z值_raw_clean',
})

df = prep.merge(clean_sub, on='孕妇代码', how='left', suffixes=('', '_y'))
print(f"合并后: {df.shape[0]} rows, {df.shape[1]} cols")

# 筛选 label
df['is_T18'] = df['染色体的非整倍体_y'].str.contains('T18', na=False)
df['is_normal'] = df['AB_异常_y'] == 0

t18 = df[df['is_T18']]
normal = df[df['is_normal']]

print(f"T18样本: {t18.shape[0]}")
print(f"正常样本: {normal.shape[0]}")

if t18.shape[0] == 0:
    print("ERROR: No T18 samples found!")
    exit(1)

# ============================================================
# 辅助函数
# ============================================================

def cohens_d(g1, g2):
    """Cohen's d: (mean1 - mean2) / pooled_std"""
    g1 = np.array(g1, dtype=float)
    g2 = np.array(g2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return np.nan, np.nan, np.nan, np.nan
    m1, m2 = np.mean(g1), np.mean(g2)
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan, m1, m2, np.nan
    d = (m1 - m2) / pooled_std
    # Welch t-test
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
    return d, m1, m2, p_val

def print_feature_table(results_dict, title):
    """打印特征筛选结果表格"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    # 按|d|降序
    sorted_items = sorted(results_dict.items(), key=lambda x: abs(x[1][0]), reverse=True)
    print(f"{'特征名':<45s} {'|Cohen d|':>10s} {'Cohen d':>10s} {'T18均值':>12s} {'正常均值':>12s} {'p值':>12s}")
    print("-" * 100)
    for name, (d, m1, m2, p) in sorted_items:
        if np.isnan(d):
            continue
        print(f"{name:<45s} {abs(d):>10.4f} {d:>10.4f} {m1:>12.4f} {m2:>12.4f} {p:>12.2e}")

# ============================================================
# 实验1：穷举单特征筛选
# ============================================================
print("\n" + "=" * 90)
print("实验1：穷举单特征筛选")
print("=" * 90)

# 确保 T18 和 normal 对齐
t18_vals = df[df['is_T18']]
normal_vals = df[df['is_normal']]

results1 = {}

# from preprocessed:
prep_features = [
    'Z13_corrected', 'Z18_corrected', 'Z21_corrected', 'ZX_corrected',
    'read_depth_log', 'dup_rate', 'map_rate', 'filter_rate',
]
for f in prep_features:
    if f in df.columns:
        d, m1, m2, p = cohens_d(t18_vals[f], normal_vals[f])
        results1[f] = (d, m1, m2, p)

# 原始读段数先log
if '原始读段数' in df.columns:
    log_reads = np.log(df['原始读段数'])
    d, m1, m2, p = cohens_d(log_reads[t18_vals.index], log_reads[normal_vals.index])
    results1['原始读段数_log'] = (d, m1, m2, p)

# from clean (merged):
clean_features = [
    '13号染色体的Z值_raw_clean', '18号染色体的Z值_raw_clean',
    '21号染色体的Z值_raw_clean', 'X染色体的Z值_raw_clean',
    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
    'GC含量_y',  # 总体 GC含量 from clean
    'X染色体浓度',
    '在参考基因组上比对的比例_y', '重复读段的比例_y', '被过滤掉读段数的比例_y',
    '孕妇BMI_y', '年龄_y', '体重_y', '身高_y', '孕周_数值_y',
]
for f in clean_features:
    if f in df.columns:
        d, m1, m2, p = cohens_d(t18_vals[f], normal_vals[f])
        # 简化名称
        display_name = f.replace('_y', '').replace('_raw_clean', '')
        results1[display_name] = (d, m1, m2, p)

print_feature_table(results1, "实验1：单特征 Cohen's d（T18 vs 正常）")

# ============================================================
# 实验2：特征工程
# ============================================================
print("\n" + "=" * 90)
print("实验2：特征工程尝试")
print("=" * 90)

results2 = {}

# 构造新特征
_all = df.copy()

# Z18_raw / ZX_raw
if '18号染色体的Z值_raw_clean' in df.columns and 'X染色体的Z值_raw_clean' in df.columns:
    _all['Z18raw_div_ZXraw'] = df['18号染色体的Z值_raw_clean'] / df['X染色体的Z值_raw_clean']
    d, m1, m2, p = cohens_d(_all.loc[t18.index, 'Z18raw_div_ZXraw'], _all.loc[normal.index, 'Z18raw_div_ZXraw'])
    results2['Z18_raw / ZX_raw'] = (d, m1, m2, p)

# Z18_corrected / ZX_corrected
if 'Z18_corrected' in df.columns and 'ZX_corrected' in df.columns:
    _all['Z18corr_div_ZXcorr'] = df['Z18_corrected'] / df['ZX_corrected']
    d, m1, m2, p = cohens_d(_all.loc[t18.index, 'Z18corr_div_ZXcorr'], _all.loc[normal.index, 'Z18corr_div_ZXcorr'])
    results2['Z18_corrected / ZX_corrected'] = (d, m1, m2, p)

# Z18_raw - median(Z13_raw, Z21_raw, ZX_raw)
if all(c in df.columns for c in ['18号染色体的Z值_raw_clean', '13号染色体的Z值_raw_clean', '21号染色体的Z值_raw_clean', 'X染色体的Z值_raw_clean']):
    median_other_raw = np.median([
        df['13号染色体的Z值_raw_clean'].values,
        df['21号染色体的Z值_raw_clean'].values,
        df['X染色体的Z值_raw_clean'].values,
    ], axis=0)
    _all['Z18_raw_minus_median_other'] = df['18号染色体的Z值_raw_clean'].values - median_other_raw
    d, m1, m2, p = cohens_d(_all.loc[t18.index, 'Z18_raw_minus_median_other'], _all.loc[normal.index, 'Z18_raw_minus_median_other'])
    results2['Z18_raw - median(other_raw)'] = (d, m1, m2, p)

# Z18_corrected - median(Z13_corrected, Z21_corrected, ZX_corrected)
if all(c in df.columns for c in ['Z18_corrected', 'Z13_corrected', 'Z21_corrected', 'ZX_corrected']):
    median_other_corr = np.median([
        df['Z13_corrected'].values,
        df['Z21_corrected'].values,
        df['ZX_corrected'].values,
    ], axis=0)
    _all['Z18_corr_minus_median_other'] = df['Z18_corrected'].values - median_other_corr
    d, m1, m2, p = cohens_d(_all.loc[t18.index, 'Z18_corr_minus_median_other'], _all.loc[normal.index, 'Z18_corr_minus_median_other'])
    results2['Z18_corrected - median(other_corr)'] = (d, m1, m2, p)

# 18号GC含量 / 总体GC含量
if '18号染色体的GC含量' in df.columns and 'GC含量_y' in df.columns:
    _all['GC18_div_GCtotal'] = df['18号染色体的GC含量'] / df['GC含量_y']
    d, m1, m2, p = cohens_d(_all.loc[t18.index, 'GC18_div_GCtotal'], _all.loc[normal.index, 'GC18_div_GCtotal'])
    results2['18号GC含量 / 总体GC含量'] = (d, m1, m2, p)

# 18号GC含量 - 13号GC含量
if '18号染色体的GC含量' in df.columns and '13号染色体的GC含量' in df.columns:
    _all['GC18_minus_GC13'] = df['18号染色体的GC含量'] - df['13号染色体的GC含量']
    d, m1, m2, p = cohens_d(_all.loc[t18.index, 'GC18_minus_GC13'], _all.loc[normal.index, 'GC18_minus_GC13'])
    results2['18号GC - 13号GC'] = (d, m1, m2, p)

# 18号GC含量 - 21号GC含量
if '18号染色体的GC含量' in df.columns and '21号染色体的GC含量' in df.columns:
    _all['GC18_minus_GC21'] = df['18号染色体的GC含量'] - df['21号染色体的GC含量']
    d, m1, m2, p = cohens_d(_all.loc[t18.index, 'GC18_minus_GC21'], _all.loc[normal.index, 'GC18_minus_GC21'])
    results2['18号GC - 21号GC'] = (d, m1, m2, p)

# BMI × Z18_corrected
if '孕妇BMI_y' in df.columns and 'Z18_corrected' in df.columns:
    _all['BMI_x_Z18corr'] = df['孕妇BMI_y'] * df['Z18_corrected']
    d, m1, m2, p = cohens_d(_all.loc[t18.index, 'BMI_x_Z18corr'], _all.loc[normal.index, 'BMI_x_Z18corr'])
    results2['BMI × Z18_corrected'] = (d, m1, m2, p)

# 年龄 × Z18_corrected
if '年龄_y' in df.columns and 'Z18_corrected' in df.columns:
    _all['Age_x_Z18corr'] = df['年龄_y'] * df['Z18_corrected']
    d, m1, m2, p = cohens_d(_all.loc[t18.index, 'Age_x_Z18corr'], _all.loc[normal.index, 'Age_x_Z18corr'])
    results2['年龄 × Z18_corrected'] = (d, m1, m2, p)

print_feature_table(results2, "实验2：特征工程 Cohen's d（T18 vs 正常）")

# ============================================================
# 实验3：最好的单特征做简单规则
# ============================================================
print("\n" + "=" * 90)
print("实验3：最佳单特征简单规则")
print("=" * 90)

# 合并实验1和2的结果，找到|d|最大的
all_results = {}
all_results.update(results1)
all_results.update(results2)

# 过滤nan
all_results = {k: v for k, v in all_results.items() if not np.isnan(v[0])}

if not all_results:
    print("ERROR: No valid features with Cohen's d!")
    exit(1)

# 按|d|降序
sorted_all = sorted(all_results.items(), key=lambda x: abs(x[1][0]), reverse=True)

print(f"\nTop 10 特征 (按|d|降序):")
print(f"{'Rank':<6s} {'特征名':<50s} {'|d|':>10s}")
print("-" * 70)
for i, (name, (d, _, _, _)) in enumerate(sorted_all[:10]):
    print(f"{i+1:<6d} {name:<50s} {abs(d):>10.4f}")

# 用最佳特征做规则
best_name, (best_d, best_m1, best_m2, best_p) = sorted_all[0]
print(f"\n最佳特征: [{best_name}], |d| = {abs(best_d):.4f}")

# 构建反向映射：display_name -> (source_df_name, column_name_in_source)
# source_df_name: 'df' or '_all'
feature_col_map = {}
for f in ['Z13_corrected', 'Z18_corrected', 'Z21_corrected', 'ZX_corrected',
          'read_depth_log', 'dup_rate', 'map_rate', 'filter_rate']:
    feature_col_map[f] = ('df', f)
feature_col_map['原始读段数_log'] = ('df_log', None)
for f in clean_features:
    if f in df.columns:
        display_name = f.replace('_y', '').replace('_raw_clean', '')
        feature_col_map[display_name] = ('df', f)
# engineered features map
eng_map = {
    'Z18_raw / ZX_raw': 'Z18raw_div_ZXraw',
    'Z18_corrected / ZX_corrected': 'Z18corr_div_ZXcorr',
    'Z18_raw - median(other_raw)': 'Z18_raw_minus_median_other',
    'Z18_corrected - median(other_corr)': 'Z18_corr_minus_median_other',
    '18号GC含量 / 总体GC含量': 'GC18_div_GCtotal',
    '18号GC - 13号GC': 'GC18_minus_GC13',
    '18号GC - 21号GC': 'GC18_minus_GC21',
    'BMI × Z18_corrected': 'BMI_x_Z18corr',
    '年龄 × Z18_corrected': 'Age_x_Z18corr',
}
for disp, internal in eng_map.items():
    if internal in _all.columns:
        feature_col_map[disp] = ('_all', internal)

t18_idx = t18.index
normal_idx = normal.index

def get_feature_vals(name):
    """Return (t18_values, normal_values) for a feature display name."""
    if name == '原始读段数_log':
        log_reads = np.log(df['原始读段数'].values)
        return log_reads[t18_idx], log_reads[normal_idx]
    if name in _all.columns:
        return _all.loc[t18_idx, name].values, _all.loc[normal_idx, name].values
    if name in df.columns:
        return df.loc[t18_idx, name].values, df.loc[normal_idx, name].values
    if name in feature_col_map:
        src, col = feature_col_map[name]
        if src == 'df' and col in df.columns:
            return df.loc[t18_idx, col].values, df.loc[normal_idx, col].values
        if src == '_all' and col in _all.columns:
            return _all.loc[t18_idx, col].values, _all.loc[normal_idx, col].values
    return None, None

best_t18, best_normal = get_feature_vals(best_name)
if best_t18 is None:
    print(f"ERROR: Cannot find column for feature [{best_name}]")
    exit(1)

best_t18 = np.array(best_t18, dtype=float)
best_normal = np.array(best_normal, dtype=float)
best_t18 = best_t18[~np.isnan(best_t18)]
best_normal = best_normal[~np.isnan(best_normal)]

print(f"  T18组均值: {np.mean(best_t18):.4f}, 正常组均值: {np.mean(best_normal):.4f}")
print(f"  T18组std:  {np.std(best_t18, ddof=1):.4f}, 正常组std:  {np.std(best_normal, ddof=1):.4f}")

# 判断 T18 偏高还是偏低
if best_m1 > best_m2:
    direction = "T18偏高"
    # 规则: 大于分位数 -> 判T18
    rules = [
        (f"{best_name} > 正常组95分位", lambda x: x > np.percentile(best_normal, 95)),
        (f"{best_name} > 正常组90分位", lambda x: x > np.percentile(best_normal, 90)),
        (f"{best_name} > 正常组75分位", lambda x: x > np.percentile(best_normal, 75)),
    ]
else:
    direction = "T18偏低"
    rules = [
        (f"{best_name} < 正常组5分位", lambda x: x < np.percentile(best_normal, 5)),
        (f"{best_name} < 正常组10分位", lambda x: x < np.percentile(best_normal, 10)),
        (f"{best_name} < 正常组25分位", lambda x: x < np.percentile(best_normal, 25)),
    ]

print(f"  方向: {direction}")

# 所有样本（T18 + 正常）
all_labels = np.concatenate([np.ones(len(best_t18)), np.zeros(len(best_normal))])
all_values = np.concatenate([best_t18, best_normal])

print(f"\n{'规则':<45s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'TN':>5s} {'召回率':>8s} {'特异度':>8s} {'精确率':>8s}")
print("-" * 95)

for rule_name, rule_fn in rules:
    pred = (rule_fn(all_values)).astype(int)
    tp = np.sum((pred == 1) & (all_labels == 1))
    fp = np.sum((pred == 1) & (all_labels == 0))
    fn = np.sum((pred == 0) & (all_labels == 1))
    tn = np.sum((pred == 0) & (all_labels == 0))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"{rule_name:<45s} {tp:>5d} {fp:>5d} {fn:>5d} {tn:>5d} {recall:>8.4f} {specificity:>8.4f} {precision:>8.4f}")

# ============================================================
# 实验4：双特征联合
# ============================================================
print("\n" + "=" * 90)
print("实验4：双特征联合分析")
print("=" * 90)

# 取|d|最大的2个特征
top2_names = [sorted_all[0][0], sorted_all[1][0]]
print(f"Top2特征: {top2_names}")

f1_vals_all, _ = get_feature_vals(top2_names[0])
f2_vals_all, _ = get_feature_vals(top2_names[1])
# Need the full array for both
def get_full_vals(name):
    if name == '原始读段数_log':
        return np.log(df['原始读段数'].values)
    if name in _all.columns:
        return _all[name].values
    if name in df.columns:
        return df[name].values
    if name in feature_col_map:
        src, col = feature_col_map[name]
        if src == 'df' and col in df.columns:
            return df[col].values
        if src == '_all' and col in _all.columns:
            return _all[col].values
    return None

f1_vals = get_full_vals(top2_names[0])
f2_vals = get_full_vals(top2_names[1])

if f1_vals is not None and f2_vals is not None:
    f1_t18 = f1_vals[t18_idx].astype(float)
    f2_t18 = f2_vals[t18_idx].astype(float)
    f1_normal = f1_vals[normal_idx].astype(float)
    f2_normal = f2_vals[normal_idx].astype(float)

    # 去除nan
    mask_t18 = ~np.isnan(f1_t18) & ~np.isnan(f2_t18)
    mask_norm = ~np.isnan(f1_normal) & ~np.isnan(f2_normal)
    f1_t18 = f1_t18[mask_t18]
    f2_t18 = f2_t18[mask_t18]
    f1_normal = f1_normal[mask_norm]
    f2_normal = f2_normal[mask_norm]

    print(f"\n  T18组 (n={len(f1_t18)}):")
    print(f"    {top2_names[0]}: 均值={np.mean(f1_t18):.4f}, std={np.std(f1_t18, ddof=1):.4f}")
    print(f"    {top2_names[1]}: 均值={np.mean(f2_t18):.4f}, std={np.std(f2_t18, ddof=1):.4f}")
    cov_t18 = np.cov(f1_t18, f2_t18)
    print(f"    协方差矩阵:\n      [[{cov_t18[0,0]:.4f}, {cov_t18[0,1]:.4f}],\n       [{cov_t18[1,0]:.4f}, {cov_t18[1,1]:.4f}]]")

    print(f"\n  正常组 (n={len(f1_normal)}):")
    print(f"    {top2_names[0]}: 均值={np.mean(f1_normal):.4f}, std={np.std(f1_normal, ddof=1):.4f}")
    print(f"    {top2_names[1]}: 均值={np.mean(f2_normal):.4f}, std={np.std(f2_normal, ddof=1):.4f}")
    cov_norm = np.cov(f1_normal, f2_normal)
    print(f"    协方差矩阵:\n      [[{cov_norm[0,0]:.4f}, {cov_norm[0,1]:.4f}],\n       [{cov_norm[1,0]:.4f}, {cov_norm[1,1]:.4f}]]")

    all_f1 = np.concatenate([f1_t18, f1_normal])
    all_f2 = np.concatenate([f2_t18, f2_normal])
    all_labels = np.concatenate([np.ones(len(f1_t18)), np.zeros(len(f1_normal))])

    def eval_rule(rule_name, pred):
        tp = np.sum((pred == 1) & (all_labels == 1))
        fp = np.sum((pred == 1) & (all_labels == 0))
        fn = np.sum((pred == 0) & (all_labels == 1))
        tn = np.sum((pred == 0) & (all_labels == 0))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"  {rule_name}: TP={tp}, FP={fp}, FN={fn}, TN={tn}, "
              f"召回={recall:.4f}, 特异度={specificity:.4f}, 精确率={precision:.4f}")

    # 判断方向：T18 比正常组高还是低
    dir1 = "higher" if np.mean(f1_t18) > np.mean(f1_normal) else "lower"
    dir2 = "higher" if np.mean(f2_t18) > np.mean(f2_normal) else "lower"
    print(f"\n  方向: {top2_names[0]}: T18 {dir1} ({np.mean(f1_t18):.4f} vs {np.mean(f1_normal):.4f})")
    print(f"        {top2_names[1]}: T18 {dir2} ({np.mean(f2_t18):.4f} vs {np.mean(f2_normal):.4f})")

    def make_rule_op(direction, percentile):
        """返回比较函数"""
        if direction == "higher":
            return lambda vals, thresh: vals > thresh
        else:
            return lambda vals, thresh: vals < thresh

    op1_90 = make_rule_op(dir1, 90)
    op2_90 = make_rule_op(dir2, 90)
    op1_95 = make_rule_op(dir1, 95)
    op2_95 = make_rule_op(dir2, 95)

    thresh1_90 = np.percentile(f1_normal, 90)
    thresh1_95 = np.percentile(f1_normal, 95)
    thresh2_90 = np.percentile(f2_normal, 90)
    thresh2_95 = np.percentile(f2_normal, 95)

    # 两个特征都超过正常组90分位（方向感知）
    both_90 = (op1_90(all_f1, thresh1_90) & op2_90(all_f2, thresh2_90)).astype(int)
    direction_text = ">" if dir1 == "higher" else "<"
    eval_rule(f"两特征都超出正常组90分位(方向感知)", both_90)

    # 任一特征超过正常组95分位
    any_95 = (op1_95(all_f1, thresh1_95) | op2_95(all_f2, thresh2_95)).astype(int)
    eval_rule(f"任一特征超出正常组95分位(方向感知)", any_95)

    both_95 = (op1_95(all_f1, thresh1_95) & op2_95(all_f2, thresh2_95)).astype(int)
    eval_rule(f"两特征都超出正常组95分位(方向感知)", both_95)

    any_90 = (op1_90(all_f1, thresh1_90) | op2_90(all_f2, thresh2_90)).astype(int)
    eval_rule(f"任一特征超出正常组90分位(方向感知)", any_90)

    # 标准化后加权和
    f1_all_std = (all_f1 - np.mean(all_f1)) / np.std(all_f1, ddof=1)
    f2_all_std = (all_f2 - np.mean(all_f2)) / np.std(all_f2, ddof=1)
    combined = f1_all_std + f2_all_std
    combined_normal = combined[len(f1_t18):]
    thresh_comb_90 = np.percentile(combined_normal, 90)
    thresh_comb_95 = np.percentile(combined_normal, 95)
    # 判断combined方向
    comb_dir = "higher" if np.mean(combined[:len(f1_t18)]) > np.mean(combined_normal) else "lower"
    op_comb_90 = make_rule_op(comb_dir, 90)(combined, thresh_comb_90) if comb_dir == "higher" else make_rule_op(comb_dir, 90)(combined, thresh_comb_90) if comb_dir == "lower" else None
    eval_rule(f"标准化和 > 正常组90分位", (combined > thresh_comb_90).astype(int))
    eval_rule(f"标准化和 > 正常组95分位", (combined > thresh_comb_95).astype(int))
    if comb_dir == "lower":
        eval_rule(f"标准化和 < 正常组10分位", (combined < np.percentile(combined_normal, 10)).astype(int))
        eval_rule(f"标准化和 < 正常组5分位", (combined < np.percentile(combined_normal, 5)).astype(int))

else:
    print("ERROR: Cannot extract top2 feature values!")

# ============================================================
# 实验5：总结
# ============================================================
print("\n" + "=" * 90)
print("实验5：总结")
print("=" * 90)

max_abs_d = max(abs(v[0]) for v in all_results.values())
best_feats = [name for name, (d, _, _, _) in sorted_all if abs(d) == max_abs_d]

print(f"\n所有特征中最大 |d| = {max_abs_d:.4f}")
print(f"达到最大|d|的特征: {best_feats}")

if max_abs_d < 0.3:
    print("\n" + "!" * 60)
    print("!  结论：T18在此数据集上无可检测的单一特征信号。")
    print("!  所有特征的 |Cohen's d| < 0.3，")
    print("!  无法用任何单一特征可靠地区分T18与正常样本。")
    print("!" * 60)
elif max_abs_d < 0.5:
    print(f"\n[WARN] 最大 |d| = {max_abs_d:.4f}，介于0.3-0.5之间，信号较弱。")
else:
    print(f"\n[OK] 最大 |d| = {max_abs_d:.4f} > 0.5，存在可检测信号！")
    print(f"  最佳检出方案基于: {best_feats[0]}")

# 列出全部按|d|排序
print(f"\n全部特征按|d|降序（合并实验1+2）:")
print(f"{'特征名':<50s} {'|d|':>8s}")
print("-" * 62)
for name, (d, _, _, _) in sorted_all:
    print(f"{name:<50s} {abs(d):>8.4f}")

print("\n实验完成。")

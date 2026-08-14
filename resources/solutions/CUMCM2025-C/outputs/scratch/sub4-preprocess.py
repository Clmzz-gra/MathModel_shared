#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
问题4 数据预处理：GC Bias 校正、正态性检验、特征工程
"""
import pandas as pd, numpy as np, os, warnings; warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

# ===== 路径 =====
data_dir = r'E:\MathModel\problems\2025\C题\outputs\data'
fig_dir  = r'E:\MathModel\problems\2025\C题\outputs\figures'
chart_dir = r'E:\MathModel\problems\2025\C题\solution\artifacts\charts'
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(chart_dir, exist_ok=True)

input_path  = os.path.join(data_dir, '2025C-female-clean.pkl')
output_path = os.path.join(data_dir, '2025C-sub4-preprocessed.pkl')

# ===== 加载数据 =====
df = pd.read_pickle(input_path)
print(f"加载数据: {df.shape[0]} 行, {df.shape[1]} 列")

# ============================================================
# 步骤1：标注样本独立性检查
# ============================================================
print("\n" + "="*60)
print("步骤1：标注样本独立性检查")
print("="*60)

abnormal = df[df['AB_异常'] == 1]
dup_counts = abnormal['孕妇代码'].value_counts()
dups = dup_counts[dup_counts > 1]
if len(dups) > 0:
    print(f"!!! 发现 {len(dups)} 个重复孕妇代码（AB_异常==1）:")
    for code, cnt in dups.items():
        print(f"  孕妇代码={code}, 出现次数={cnt}")
else:
    print("[OK] 67 条异常样本中，孕妇代码无重复 — 样本独立。")

# ============================================================
# 步骤2：GC Bias 校正（LOESS, span=0.75）
# ============================================================
print("\n" + "="*60)
print("步骤2：GC Bias 校正")
print("="*60)

# 染色体 -> (Z列名, 简称)
chrs = {
    '13': ('13号染色体的Z值', 'Z13'),
    '18': ('18号染色体的Z值', 'Z18'),
    '21': ('21号染色体的Z值', 'Z21'),
    'X':  ('X染色体的Z值',   'ZX'),
}
gc_col = 'GC含量'  # 总体 GC 含量

# 存储校正结果
z_corrected = {}

print("\nLOESS 拟合摘要 (span=0.75, Z ~ 总体GC含量):")
for ch, (z_col, z_short) in chrs.items():
    x = df[gc_col].values
    y = df[z_col].values
    mask = np.isfinite(x) & np.isfinite(y)
    x_fit, y_fit = x[mask], y[mask]

    # LOESS 拟合
    loess_result = lowess(y_fit, x_fit, frac=0.75, return_sorted=True)
    x_sorted = loess_result[:, 0]
    y_pred_sorted = loess_result[:, 1]

    # R²
    ss_res = np.sum((y_fit - np.interp(x_fit, x_sorted, y_pred_sorted))**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
    r2 = 1 - ss_res / ss_tot

    # 预测所有样本的 GC bias
    y_pred_all = np.interp(df[gc_col].values, x_sorted, y_pred_sorted)
    corrected = df[z_col].values - y_pred_all
    z_corrected[z_short] = corrected

    print(f"  染色体{ch} ({z_short}): R^2={r2:.5f}")

# 校正前后均值对比
print("\n正常样本(AB_异常==0)校正前后 Z 值均值对比:")
normal = df[df['AB_异常'] == 0]
print(f"  {'染色体':<8} {'校正前均值':>12} {'校正后均值':>12} {'|校正后-0|':>12}")
print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
for ch, (z_col, z_short) in chrs.items():
    before_mean = normal[z_col].mean()
    after_mean = np.mean(z_corrected[z_short][normal.index])
    print(f"  {ch:<8} {before_mean:>12.5f} {after_mean:>12.5f} {abs(after_mean):>12.5f}")

# ===== 画 2×2 对比图 =====
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, (ch, (z_col, z_short)) in enumerate(chrs.items()):
    row = i // 2
    col_left = (i % 2) * 2
    col_right = col_left + 1

    # 左：校正前
    ax = axes[row][col_left]
    ax.scatter(normal[gc_col], normal[z_col], c='green', alpha=0.4, s=10, label='正常')
    ax.scatter(abnormal[gc_col], abnormal[z_col], c='red', alpha=0.6, s=12, label='异常')
    ax.set_xlabel('总体GC含量')
    ax.set_ylabel(f'{z_short} (校正前)')
    ax.set_title(f'染色体{ch} — 校正前')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=7)

    # 右：校正后
    ax = axes[row][col_right]
    ax.scatter(normal[gc_col], z_corrected[z_short][normal.index],
               c='green', alpha=0.4, s=10, label='正常')
    ax.scatter(abnormal[gc_col], z_corrected[z_short][abnormal.index],
               c='red', alpha=0.6, s=12, label='异常')
    ax.set_xlabel('总体GC含量')
    ax.set_ylabel(f'{z_short} (校正后)')
    ax.set_title(f'染色体{ch} — 校正后')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=7)

fig.suptitle('GC Bias 校正前后对比 (LOESS, span=0.75)', fontsize=14, fontweight='bold')
plt.tight_layout()
gc_fig_pdf = os.path.join(fig_dir, 'sub4-gc-correction.pdf')
gc_fig_chart = os.path.join(chart_dir, 'sub4-gc-correction.pdf')
fig.savefig(gc_fig_pdf, dpi=200, bbox_inches='tight')
fig.savefig(gc_fig_chart, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\nGC校正对比图已保存: {gc_fig_pdf}")
print(f"                  {gc_fig_chart}")

# ============================================================
# 步骤3：Z 值正态性检查
# ============================================================
print("\n" + "="*60)
print("步骤3：Z 值正态性检查（正常样本, GC校正后）")
print("="*60)

print(f"\n{'染色体':<8} {'偏度':>10} {'峰度':>10} {'D-A-P p值':>14}")
print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*14}")
normality_results = {}
for ch, (z_col, z_short) in chrs.items():
    vals = z_corrected[z_short][normal.index]
    skew = stats.skew(vals)
    kurt = stats.kurtosis(vals, fisher=True)  # excess kurtosis
    # D'Agostino-Pearson
    da_stat, da_p = stats.normaltest(vals)
    normality_results[z_short] = {'skew': skew, 'kurt': kurt, 'da_p': da_p}
    sig = '***' if da_p < 0.001 else ('**' if da_p < 0.01 else ('*' if da_p < 0.05 else ''))
    print(f"  {ch:<8} {skew:>10.4f} {kurt:>10.4f} {da_p:>14.6f} {sig}")

# ===== Q-Q 图 =====
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, (ch, (z_col, z_short)) in enumerate(chrs.items()):
    row, col = i // 2, i % 2
    ax = axes[row][col]
    vals = z_corrected[z_short][normal.index]
    stats.probplot(vals, dist="norm", plot=ax)
    ax.set_title(f'染色体{ch} ({z_short})')
    nres = normality_results[z_short]
    ax.text(0.05, 0.95,
            f'偏度={nres["skew"]:.3f}\n峰度={nres["kurt"]:.3f}\nD-A-P p={nres["da_p"]:.4f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('GC校正后 Z 值 Q-Q 图 (正常样本)', fontsize=14, fontweight='bold')
plt.tight_layout()
qq_pdf = os.path.join(fig_dir, 'sub4-normal-qq.pdf')
qq_chart = os.path.join(chart_dir, 'sub4-normal-qq.pdf')
fig.savefig(qq_pdf, dpi=200, bbox_inches='tight')
fig.savefig(qq_chart, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\nQ-Q 图已保存: {qq_pdf}")
print(f"              {qq_chart}")

# ============================================================
# 步骤4：多元正态性检查 (Mardia)
# ============================================================
print("\n" + "="*60)
print("步骤4：多元正态性检查（Mardia 检验）")
print("="*60)

# 构建 4 维向量 (Z13, Z18, Z21, ZX)
Z_mat = np.column_stack([
    z_corrected['Z13'][normal.index],
    z_corrected['Z18'][normal.index],
    z_corrected['Z21'][normal.index],
    z_corrected['ZX'][normal.index]
])
n, p = Z_mat.shape

# 中心化
Z_centered = Z_mat - Z_mat.mean(axis=0)
# 协方差矩阵的逆
S_inv = np.linalg.inv(np.cov(Z_mat, rowvar=False))

# 计算成对马氏距离
# D_{ij}^2 = (Z_i - Z_j)^T S^{-1} (Z_i - Z_j)
# 更高效的方式：计算 g 统计量
# Mardia skewness: b_{1,p} = (1/n^2) * sum_i sum_j [(z_i - z_bar)^T S^{-1} (z_j - z_bar)]^3
# Mardia kurtosis: b_{2,p} = (1/n) * sum_i [(z_i - z_bar)^T S^{-1} (z_i - z_bar)]^2

# Mahalanobis distance for each observation
mahal = np.sum((Z_centered @ S_inv) * Z_centered, axis=1)

# 计算 pairwise: g_{ij} = (z_i - z_bar)^T S^{-1} (z_j - z_bar)
# = Z_centered_i @ S_inv @ Z_centered_j
g_ij = Z_centered @ S_inv @ Z_centered.T

# Mardia skewness
b1p = np.sum(g_ij ** 3) / (n ** 2)
# Mardia kurtosis
b2p = np.mean(mahal ** 2)

# Test statistics
# Skewness: n * b1p / 6 ~ chi2(p(p+1)(p+2)/6)
df_skew = p * (p + 1) * (p + 2) // 6
skew_stat = n * b1p / 6
skew_p = 1 - stats.chi2.cdf(skew_stat, df_skew)

# Kurtosis: (b2p - p(p+2)) / sqrt(8 * p * (p+2) / n) ~ N(0,1)
kurt_mean = p * (p + 2)
kurt_se = np.sqrt(8 * p * (p + 2) / n)
kurt_z = (b2p - kurt_mean) / kurt_se
kurt_p = 2 * (1 - stats.norm.cdf(abs(kurt_z)))

print(f"\n  样本量 n={n}, 维度 p={p}")
print(f"  Mardia 偏度: b1p={b1p:.5f}, 统计量={skew_stat:.3f}, df={df_skew}, p={skew_p:.6f}")
print(f"  Mardia 峰度: b2p={b2p:.5f}, Z={kurt_z:.4f}, p={kurt_p:.6f}")
if skew_p < 0.05 and kurt_p < 0.05:
    print("  [!!] 偏度和峰度均显著 — 多元正态性不满足。")
elif skew_p < 0.05:
    print("  [!!] 偏度显著 — 分布有偏。")
elif kurt_p < 0.05:
    print("  [!!] 峰度显著 — 分布有尖峰/平峰。")
else:
    print("  [OK] 偏度和峰度均不显著 — 不能拒绝多元正态性。")

# ============================================================
# 步骤5：特征工程 — 测序质量信号
# ============================================================
print("\n" + "="*60)
print("步骤5：特征工程 — 测序质量信号")
print("="*60)

# 创建新列
df['read_depth_log'] = np.log(df['原始读段数'])
df['dup_rate'] = df['重复读段的比例']
df['map_rate'] = df['在参考基因组上比对的比例']
df['filter_rate'] = df['被过滤掉读段数的比例']

# 添加校正后 Z 值到 df
for z_short, vals in z_corrected.items():
    df[f'{z_short}_corrected'] = vals

quality_cols = ['read_depth_log', 'dup_rate', 'map_rate', 'filter_rate']
z_corrected_cols = ['Z13_corrected', 'Z18_corrected', 'Z21_corrected', 'ZX_corrected']

print("\n测序质量信号与 Z_corrected 的 Spearman 相关:")
print(f"  {'质量信号':<20}", end='')
for zc in z_corrected_cols:
    print(f" {zc:>14}", end='')
print()
print(f"  {'-'*20}", end='')
for _ in z_corrected_cols:
    print(f" {'-'*14}", end='')
print()

for qc in quality_cols:
    print(f"  {qc:<20}", end='')
    for zc in z_corrected_cols:
        r, pv = stats.spearmanr(df[qc], df[zc], nan_policy='omit')
        sig = '*' if pv < 0.05 else ''
        print(f" {r:>8.4f}{sig:<6}", end='')
    print()

print("\n如果相关较弱（|r|<0.1），则无需质量加权。")

# ============================================================
# 步骤6：写入预处理后数据
# ============================================================
print("\n" + "="*60)
print("步骤6：写入预处理后数据")
print("="*60)

keep_cols = [
    # 标识列
    '序号', '孕妇代码',
    # 原始特征列
    '年龄', '身高', '体重', '孕妇BMI', '孕周_数值', 'IVF_编码',
    '怀孕次数_num', '生产次数', '检测抽血次数',
    # 原始 GC 和测序列
    'GC含量', '原始读段数', '在参考基因组上比对的比例',
    '重复读段的比例', '被过滤掉读段数的比例',
    # 新增 GC 校正后 Z 值
    'Z13_corrected', 'Z18_corrected', 'Z21_corrected', 'ZX_corrected',
    # 新增测序质量信号
    'read_depth_log', 'dup_rate', 'map_rate', 'filter_rate',
    # 标签列
    '染色体的非整倍体', 'AB_异常',
]

# 确保所有列都存在
missing = [c for c in keep_cols if c not in df.columns]
if missing:
    raise ValueError(f"缺失列: {missing}")

out_df = df[keep_cols].copy()

# 同时保存原始 Z 值以供参考
for z_col_name in ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']:
    if z_col_name not in out_df.columns:
        out_df[z_col_name] = df[z_col_name]

out_df.to_pickle(output_path)
print(f"\n预处理完成!")
print(f"  行数: {out_df.shape[0]}")
print(f"  列数: {out_df.shape[1]}")
print(f"  新增列: Z13_corrected, Z18_corrected, Z21_corrected, ZX_corrected, "
      f"read_depth_log, dup_rate, map_rate, filter_rate")
print(f"  保留原始 Z 值列: 13号染色体的Z值, 18号染色体的Z值, 21号染色体的Z值, X染色体的Z值")
print(f"  输出文件: {output_path}")

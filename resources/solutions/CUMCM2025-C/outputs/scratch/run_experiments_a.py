# -*- coding: utf-8 -*-
"""
A类验证实验（共享事实）
不修改任何现有文件，只输出结果到控制台。
图表保存为 PDF。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr

# ============================================================
# 手动实现混淆矩阵和评估指标（不需要 sklearn）
# ============================================================
def conf_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = conf_matrix(y_true, y_pred)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    # AUC = (TPR + TNR) / 2 for binary classifier
    tpr = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)
    auc = (tpr + tnr) / 2.0
    return tp, tn, fp, fn, acc, prec, rec, f1, auc

# ============================================================
# 全局设置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = r'e:\MathModel\problems\2025\C题\outputs\figures'
ARTIFACT_DIR = r'e:\MathModel\problems\2025\C题\solution\artifacts\charts'
DATA_PATH = r'e:\MathModel\problems\2025\C题\outputs\data\2025C-female-clean.pkl'

# 读取数据
df = pd.read_pickle(DATA_PATH)
print("=" * 80)
print("数据加载: 2025C-female-clean.pkl")
print(f"总样本数: {len(df)}")
print(f"AB_异常==1: {(df['AB_异常']==1).sum()}")
print(f"AB_异常==0: {(df['AB_异常']==0).sum()}")
print(f"flag_z_ab_mismatch==1: {(df['flag_z_ab_mismatch']==1).sum()}")
print()

# 关键列名
Z21 = '21号染色体的Z值'
Z18 = '18号染色体的Z值'
Z13 = '13号染色体的Z值'
ZX  = 'X染色体的Z值'
AB  = 'AB_异常'
TYPE = '染色体的非整倍体'

# ============================================================
# 实验1：Z值对各类异常的区分度（单变量分析）
# ============================================================
print("=" * 80)
print("实验1：Z值对各类异常的区分度（单变量分析）")
print("=" * 80)

ab_df = df[df[AB] == 1].copy()
print(f"有标签异常样本数: {len(ab_df)}")
print(f"异常类型分布: {ab_df[TYPE].value_counts().to_dict()}")
print()

# 准备箱线图数据
type_order = ['T21', 'T18', 'T13', 'T13T18', 'T18T21', 'T13T21']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
z_vars = [Z21, Z18, Z13, ZX]
titles = ['21号Z值 按异常类型分组', '18号Z值 按异常类型分组',
          '13号Z值 按异常类型分组', 'X染色体Z值 按异常类型分组']

for idx, (ax, z_var, title) in enumerate(zip(axes.flat, z_vars, titles)):
    data_groups = []
    labels = []
    for t in type_order:
        vals = ab_df[ab_df[TYPE] == t][z_var].dropna()
        if len(vals) > 0:
            data_groups.append(vals.values)
            labels.append(t)
    bp = ax.boxplot(data_groups, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_title(title)
    ax.set_ylabel('Z值')
    ax.axhline(y=3, color='red', linestyle='--', alpha=0.6, label='Z=3')
    ax.axhline(y=-3, color='red', linestyle='--', alpha=0.6)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Z值对各类异常的区分度（67条有标签异常样本）', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f'{FIG_DIR}/sub4-Z-by-type.pdf', dpi=150, bbox_inches='tight')
fig.savefig(f'{ARTIFACT_DIR}/sub4-Z-by-type.pdf', dpi=150, bbox_inches='tight')
plt.close(fig)
print("图表已保存: sub4-Z-by-type.pdf")
print()

# 表格：按类型输出Z值min/max/mean
print("=" * 60)
print("Z值统计表（min / max / mean）")
print("=" * 60)

normal_df = df[df[AB] == 0]

def print_z_stats(label, subset):
    print(f"\n{'─' * 50}")
    print(f"  {label}  (n={len(subset)})")
    print(f"{'─' * 50}")
    for z_col in [Z21, Z18, Z13]:
        vals = subset[z_col].dropna()
        print(f"  {z_col}: min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}")

print_z_stats('正常（AB_异常==0）', normal_df)
for t in ['T21', 'T18', 'T13', 'T13T18', 'T18T21', 'T13T21']:
    subset = ab_df[ab_df[TYPE] == t]
    if len(subset) > 0:
        print_z_stats(f'{t}型', subset)

print()

# ============================================================
# 实验2：Z值 > 3 规则的基准性能
# ============================================================
print("=" * 80)
print("实验2：Z值 > 3 规则的基准性能")
print("=" * 80)

# 在67条有标签样本上验证
labeled = df[df[AB].isin([0, 1])].copy()
# 规则：任一染色体Z值 > 3
labeled['pred'] = ((labeled[Z13] > 3) | (labeled[Z18] > 3) | (labeled[Z21] > 3)).astype(int)
y_true = labeled[AB].values
y_pred = labeled['pred'].values

tp, tn, fp, fn, acc, prec, rec, f1, auc = compute_metrics(y_true, y_pred)

print(f"混淆矩阵 (N={len(labeled)}):")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"  准确率: {acc:.4f}")
print(f"  精确率: {prec:.4f}")
print(f"  召回率: {rec:.4f}")
print(f"  F1分数: {f1:.4f}")
print(f"  AUC:    {auc:.4f}")
print()

# 按异常类型细分
print("按异常类型细分（Z>3规则检出率）:")
print("-" * 50)
for t in ['T21', 'T18', 'T13', 'T13T18', 'T18T21', 'T13T21']:
    subset = labeled[labeled[TYPE] == t]
    if len(subset) > 0:
        detected = subset['pred'].sum()
        pct = detected / len(subset) * 100
        print(f"  {t}型 (n={len(subset)}): 检出 {int(detected)} 例 ({pct:.1f}%)")
        # 详细列出哪些Z值超标
        z13_hi = (subset[Z13] > 3).sum()
        z18_hi = (subset[Z18] > 3).sum()
        z21_hi = (subset[Z21] > 3).sum()
        print(f"    其中 Z13>3:{int(z13_hi)}, Z18>3:{int(z18_hi)}, Z21>3:{int(z21_hi)}")

# 正常样本误判
normal_labeled = labeled[labeled[AB] == 0]
fp_normal = normal_labeled['pred'].sum()
print(f"\n  正常样本 (n={len(normal_labeled)}): 误判 {int(fp_normal)} 例")
print()

# ============================================================
# 实验3：41条"Z异常但无标签"样本分析
# ============================================================
print("=" * 80)
print("实验3：41条 flag_z_ab_mismatch==1 样本分析")
print("=" * 80)

mismatch = df[df['flag_z_ab_mismatch'] == 1].copy()
print(f"样本数: {len(mismatch)}")
print()

# 哪些Z值超标
print("Z值超标情况:")
print("-" * 40)
for z_col, label in [(Z13, '13号'), (Z18, '18号'), (Z21, '21号')]:
    hi_mask = mismatch[z_col] > 3
    n_hi = hi_mask.sum()
    if n_hi > 0:
        vals_hi = mismatch.loc[hi_mask, z_col]
        print(f"  {label}Z > 3: {n_hi} 条, min={vals_hi.min():.4f}, max={vals_hi.max():.4f}, mean={vals_hi.mean():.4f}")

lo_mask = mismatch[z_col] < -3
n_lo = lo_mask.sum()
for z_col, label in [(Z13, '13号'), (Z18, '18号'), (Z21, '21号')]:
    lo_mask = mismatch[z_col] < -3
    n_lo = lo_mask.sum()
    if n_lo > 0:
        vals_lo = mismatch.loc[lo_mask, z_col]
        print(f"  {label}Z < -3: {n_lo} 条, min={vals_lo.min():.4f}, max={vals_lo.max():.4f}, mean={vals_lo.mean():.4f}")

print()

# 具体每条的超标详情
print("每条样本Z值超标详情:")
print("-" * 60)
for idx, row in mismatch.iterrows():
    flags = []
    if row[Z13] > 3: flags.append(f"13号Z={row[Z13]:.3f}")
    if row[Z18] > 3: flags.append(f"18号Z={row[Z18]:.3f}")
    if row[Z21] > 3: flags.append(f"21号Z={row[Z21]:.3f}")
    if row[Z13] < -3: flags.append(f"13号Z={row[Z13]:.3f}(低)")
    if row[Z18] < -3: flags.append(f"18号Z={row[Z18]:.3f}(低)")
    if row[Z21] < -3: flags.append(f"21号Z={row[Z21]:.3f}(低)")
    print(f"  序号{row['序号']} ({row['孕妇代码']}): {', '.join(flags)}")

print()

# Z值分布对比密度图
print("绘制分布对比图...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 提取有标签异常样本的Z值
ab_labeled = df[df[AB] == 1]

for i, (z_col, title) in enumerate([(Z13, '13号Z值分布对比'),
                                      (Z18, '18号Z值分布对比'),
                                      (Z21, '21号Z值分布对比')]):
    ax = axes[i]
    # 有标签异常
    ab_vals = ab_labeled[z_col].dropna()
    # 无标签Z异常
    mm_vals = mismatch[z_col].dropna()

    ab_vals.plot.kde(ax=ax, label=f'有标签异常 (n={len(ab_vals)})', color='red', linewidth=2)
    mm_vals.plot.kde(ax=ax, label=f'无标签Z异常 (n={len(mm_vals)})', color='blue', linewidth=2)
    ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5, label='Z=3')
    ax.set_title(title)
    ax.set_xlabel('Z值')
    ax.set_ylabel('密度')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle('有标签异常 vs 无标签Z异常 的Z值分布对比', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f'{FIG_DIR}/sub4-mismatch-density.pdf', dpi=150, bbox_inches='tight')
fig.savefig(f'{ARTIFACT_DIR}/sub4-mismatch-density.pdf', dpi=150, bbox_inches='tight')
plt.close(fig)
print("图表已保存: sub4-mismatch-density.pdf")
print()

# ============================================================
# 实验4：特征相关性
# ============================================================
print("=" * 80)
print("实验4：特征相关性（Spearman）")
print("=" * 80)

# 选取变量
corr_vars = [
    ZX, Z13, Z18, Z21,
    'X染色体浓度',
    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
    '原始读段数', '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例',
    '孕妇BMI', '年龄'
]

# 检查所有变量是否存在
corr_df = df[corr_vars].copy()
print(f"参与相关分析变量数: {len(corr_vars)}")
print(f"有效样本数: {len(corr_df.dropna())}")
print()

# 计算Spearman相关矩阵
n_vars = len(corr_vars)
corr_matrix = np.zeros((n_vars, n_vars))
pval_matrix = np.zeros((n_vars, n_vars))

for i in range(n_vars):
    for j in range(n_vars):
        if i == j:
            corr_matrix[i, j] = 1.0
            pval_matrix[i, j] = 0.0
        else:
            mask = corr_df[corr_vars[i]].notna() & corr_df[corr_vars[j]].notna()
            if mask.sum() > 2:
                r, p = spearmanr(corr_df.loc[mask, corr_vars[i]], corr_df.loc[mask, corr_vars[j]])
                corr_matrix[i, j] = r
                pval_matrix[i, j] = p
            else:
                corr_matrix[i, j] = np.nan
                pval_matrix[i, j] = np.nan

# 短标签
short_labels = [
    'X_Z', '13_Z', '18_Z', '21_Z',
    'X浓度',
    '13_GC', '18_GC', '21_GC',
    '读段数', '比对率', '重复率', '过滤率',
    'BMI', '年龄'
]

corr_df_out = pd.DataFrame(corr_matrix, index=short_labels, columns=short_labels)

# 输出 |ρ| > 0.3 的显著相关对
print("显著相关对 (|ρ| > 0.3):")
print("-" * 60)
sig_pairs = []
for i in range(n_vars):
    for j in range(i+1, n_vars):
        if abs(corr_matrix[i, j]) > 0.3:
            sig_pairs.append((short_labels[i], short_labels[j], corr_matrix[i, j], pval_matrix[i, j]))
sig_pairs.sort(key=lambda x: -abs(x[2]))
for a, b, r, p in sig_pairs:
    stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"  {a:8s} vs {b:8s}: ρ = {r:+.4f}  (p = {p:.4f}) {stars}")

print(f"\n共 {len(sig_pairs)} 对显著相关")

# 打印完整相关矩阵
print("\n完整Spearman相关矩阵:")
print("-" * 80)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)
print(corr_df_out.to_string())
print()

# 绘制热力图
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True

im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Spearman ρ')

ax.set_xticks(range(n_vars))
ax.set_yticks(range(n_vars))
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(short_labels, fontsize=9)
ax.set_title('特征相关性热力图 (Spearman, N=605)', fontsize=14)

# 在格子上标注数值
for i in range(n_vars):
    for j in range(n_vars):
        if not np.isnan(corr_matrix[i, j]):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

fig.tight_layout()
fig.savefig(f'{FIG_DIR}/sub4-correlation-heatmap.pdf', dpi=150, bbox_inches='tight')
fig.savefig(f'{ARTIFACT_DIR}/sub4-correlation-heatmap.pdf', dpi=150, bbox_inches='tight')
plt.close(fig)
print("图表已保存: sub4-correlation-heatmap.pdf")
print()

# ============================================================
print("=" * 80)
print("全部实验完成。")
print(f"图表输出: {FIG_DIR}/")
print(f"图表输出: {ARTIFACT_DIR}/")
print("=" * 80)

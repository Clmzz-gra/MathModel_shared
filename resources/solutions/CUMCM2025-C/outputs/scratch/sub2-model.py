#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""问题2完整实现：BMI分组 + 最佳NIPT时点"""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

cache_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\data'
fig_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\figures'
chart_dir = 'E:\\MathModel\\problems\\2025\\C题\\solution\\artifacts\\charts'
tables_dir = 'E:\\MathModel\\problems\\2025\\C题\\solution\\artifacts\\tables'
os.makedirs(chart_dir, exist_ok=True); os.makedirs(tables_dir, exist_ok=True)

male = pd.read_pickle(os.path.join(cache_dir, '2025C-male-clean.pkl'))
def pgw(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    for sep in ['w+','W+']:
        if sep in s:
            p = s.split(sep); return float(p[0])+float(p[1])/7.0
    return float(s.replace('w','').replace('W',''))
male['gw'] = male['检测孕周'].apply(pgw)

# === 分组 ===
bins = [0, 28, 32, 36, 40, 100]
labels = ['[20,28)', '[28,32)', '[32,36)', '[36,40)', '40+']
male['bmi_group'] = pd.cut(male['孕妇BMI'], bins=bins, labels=labels, right=False)

# === 真实达标率 ===
weeks = np.arange(10, 26)
pass_rates = {}
for lb in labels:
    rates = []
    for w in weeks:
        mask = (male['bmi_group'] == lb) & (male['gw'] >= w-0.5) & (male['gw'] < w+0.5)
        if mask.sum() >= 3:
            rates.append((male.loc[mask, 'Y染色体浓度'] >= 0.04).mean())
        else:
            rates.append(np.nan)
    pass_rates[lb] = np.array(rates)

# 各BMI组代表值
bmi_mid = {'[20,28)': 24, '[28,32)': 30, '[32,36)': 34, '[36,40)': 38, '40+': 43}

# 最优时点：达标率>=p0的最早孕周（默认p0=0.5）
def best_time(rate_arr, p0=0.5):
    for i, r in enumerate(rate_arr):
        if not np.isnan(r) and r >= p0:
            return weeks[i], r
    return 25, rate_arr[~np.isnan(rate_arr)][-1] if np.any(~np.isnan(rate_arr)) else np.nan

print('=' * 70)
print('问题2: BMI分组与最佳NIPT时点')
print('=' * 70)
print()
print('策略: 早测低代价 + 阶梯风险。p0=0.5（不达标仅需补测，代价低）')
print('硬约束: t* ≤ 25周（预留治疗窗口）')
print()

# === 表: 各BMI组 + p0灵敏度 ===
print(f'{"BMI组":12s} {"n":>5s} {"p0=0.5":>8s} {"概率":>6s} {"p0=0.6":>8s} {"概率":>6s} {"p0=0.7":>8s} {"概率":>6s} {"+0.5%误差":>8s}')
print('-'*80)
results = []
for lb in labels:
    n = (male['bmi_group'] == lb).shape[0]
    rs = pass_rates[lb]
    t05, p05 = best_time(rs, 0.5)
    t06, p06 = best_time(rs, 0.6) if np.any(~np.isnan(rs)) else (25, np.nan)
    t07, p07 = best_time(rs, 0.7) if np.any(~np.isnan(rs)) else (25, np.nan)
    # +0.5% 误差: 阈值从4%变为4.5%
    rates_shift = []
    for w in weeks:
        mask = (male['bmi_group'] == lb) & (male['gw'] >= w-0.5) & (male['gw'] < w+0.5)
        if mask.sum() >= 3:
            rates_shift.append((male.loc[mask, 'Y染色体浓度'] >= 0.045).mean())
        else:
            rates_shift.append(np.nan)
    t_shift, _ = best_time(np.array(rates_shift), 0.5)
    shift_str = f'{t_shift:.0f}周(+{t_shift-t05:.0f})' if t_shift > t05 else f'{t_shift:.0f}周(不变)'
    print(f'{lb:12s} {n:5d} {t05:8.0f}周 {p05:6.0%} {t06:8.0f}周 {p06:6.0%} {t07:8.0f}周 {p07:6.0%} {shift_str:>8s}')
    results.append((lb, n, t05, p05, t06, p06, t07, p07, t_shift))

# === 图1: 达标率曲线 ===
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for lb, c in zip(labels, colors):
    rs = pass_rates[lb]
    valid = ~np.isnan(rs)
    if np.any(valid):
        ax.plot(weeks[valid], rs[valid]*100, 'o-', color=c, linewidth=1.5, markersize=4, label=lb)
        # p0=0.5最佳时点竖线
        t05, _ = best_time(rs, 0.5)
        ax.axvline(x=t05, color=c, linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(y=50, color='#333333', linestyle=':', linewidth=0.8, alpha=0.5, label='p0=0.5')
ax.axhline(y=70, color='#999999', linestyle=':', linewidth=0.5, alpha=0.3, label='p0=0.7')
ax.set_xlabel('孕周'); ax.set_ylabel('Y≥4% 比例(%)'); ax.set_title('各BMI组Y浓度达标率随孕周变化')
ax.legend(fontsize=8, loc='lower right'); ax.set_xlim(10, 25); ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'sub2-pass-rate-curves.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(chart_dir, 'sub2-pass-rate-curves.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('\n图1已保存: 达标率曲线')

# === 图2: 首次达标时间箱线图 (各BMI组) ===
fp = male[male['Y染色体浓度'] >= 0.04].groupby('孕妇代码')['gw'].min().reset_index()
fp.columns = ['code', 'first_pass']
bmi_info = male[['孕妇代码', 'bmi_group', '孕妇BMI']].drop_duplicates('孕妇代码')
fp = fp.merge(bmi_info, left_on='code', right_on='孕妇代码')

fig, ax = plt.subplots(figsize=(10, 6))
data = [fp[fp['bmi_group'] == lb]['first_pass'].values for lb in labels]
bp = ax.boxplot(data, labels=labels, patch_artist=True)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.7)
# 标注中位数
for i, lb in enumerate(labels):
    med = fp[fp['bmi_group'] == lb]['first_pass'].median()
    n = len(fp[fp['bmi_group'] == lb])
    ax.text(i+1, med+0.3, f'{med:.1f}周\n(n={n})', ha='center', fontsize=8)
ax.axhline(y=12, color='#2ca02c', linestyle='--', linewidth=0.8, alpha=0.6, label='低风险期(≤12周)')
ax.set_xlabel('BMI分组'); ax.set_ylabel('首次达标孕周(周)')
ax.set_title('各组首次Y≥4%达标孕周分布')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'sub2-first-pass-boxplot.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(chart_dir, 'sub2-first-pass-boxplot.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('图2已保存: 首次达标时间箱线图')

# === 图3: p0灵敏度热力图 ===
fig, ax = plt.subplots(figsize=(10, 4))
p0_vals = np.arange(0.3, 0.9, 0.05)
heat = np.zeros((len(labels), len(p0_vals)))
for i, lb in enumerate(labels):
    rs = pass_rates[lb]
    for j, p0 in enumerate(p0_vals):
        t_opt, _ = best_time(rs, p0)
        heat[i, j] = t_opt
im = ax.imshow(heat, cmap='RdYlGn_r', aspect='auto', vmin=10, vmax=25)
for i in range(len(labels)):
    for j in range(len(p0_vals)):
        ax.text(j, i, f'{heat[i,j]:.0f}', ha='center', va='center', fontsize=9,
                color='white' if heat[i,j] > 18 else 'black')
ax.set_xticks(range(len(p0_vals)))
ax.set_xticklabels([f'{p:.0%}' for p in p0_vals], fontsize=8)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('p0 (可接受把握)'); ax.set_title('最佳时点灵敏度矩阵 (p0=0.3~0.85)')
plt.colorbar(im, ax=ax, label='最佳时点(周)', shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'sub2-p0-sensitivity.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(chart_dir, 'sub2-p0-sensitivity.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('图3已保存: p0灵敏度热力图')

# === LaTeX表 ===
with open(os.path.join(tables_dir, 'sub2-optimal-timing.tex'), 'w', encoding='utf-8') as f:
    f.write(r'\begin{tabular}{lrrrrr}'+'\n')
    f.write(r'\toprule'+'\n')
    f.write(r'BMI分组 & 样本量 & p0=0.5时点 & 达标概率 & p0=0.7时点 & 达标概率 \\'+'\n')
    f.write(r'\midrule'+'\n')
    for (lb, n, t05, p05, t06, p06, t07, p07, t_shift) in results:
        f.write(r'%s & %d & %d周 & %.0f\%% & %d周 & %.0f\%% \\' % (lb, n, t05, p05*100, t07, p07*100)+'\n')
    f.write(r'\midrule'+'\n')
    f.write(r'\multicolumn{6}{l}{\small 早测低代价策略：p0=0.5（不达标仅需补测，代价为零）}'+'\n')
    f.write(r'\end{tabular}'+'\n')
print('LaTeX表已保存')

# === 摘要输出 ===
print('\n' + '='*70)
print('最终建议')
print('='*70)
for (lb, n, t05, p05, t06, p06, t07, p07, t_shift) in results:
    note = ''
    if t05 == 12: note = ' ← 早测, 不达标补测即可'
    elif t05 >= 14: note = ' ← 稍等2-6周以确保达标'
    if lb == '40+': note += ' [数据稀疏, n=%d]' % n
    print('%s: %d周 (把握%.0f%%)%s' % (lb, t05, p05*100, note))

print('\n完成')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批判7处置: 直接计数 + Wilson CI + 样本量标注"""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

cache_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\data'
chart_dir = 'E:\\MathModel\\problems\\2025\\C题\\solution\\artifacts\\charts'
fig_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\figures'

male = pd.read_pickle(os.path.join(cache_dir, '2025C-male-clean.pkl'))
male['gw'] = male['孕周_数值']

bins = [0, 28, 32, 36, 40, 100]
labels = ['[20,28)', '[28,32)', '[32,36)', '[36,40)', '40+']
male['bmi_group'] = pd.cut(male['孕妇BMI'], bins=bins, labels=labels, right=False)

# === Wilson score CI ===
def wilson_ci(k, n, alpha=0.05):
    """Wilson score confidence interval for binomial proportion"""
    if n == 0: return np.nan, np.nan, np.nan
    z = stats.norm.ppf(1 - alpha/2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denom
    return p_hat, max(0, center - margin), min(1, center + margin)

weeks = np.arange(10, 26)

# === 逐格计算 ===
cell_data = {}  # lb -> list of (week, n, pass_n, rate, ci_low, ci_high)
for lb in labels:
    cells = []
    for w in weeks:
        mask = (male['bmi_group'] == lb) & (male['gw'] >= w-0.5) & (male['gw'] < w+0.5)
        n = mask.sum()
        k = (male.loc[mask, 'Y染色体浓度'] >= 0.04).sum() if n > 0 else 0
        rate, ci_low, ci_high = wilson_ci(k, n)
        cells.append((w, n, k, rate, ci_low, ci_high))
    cell_data[lb] = cells

# === 最佳时点（使用Wilson中心估计）===
def best_time_wilson(cells, p0=0.5):
    for w, n, k, rate, ci_low, ci_high in cells:
        if n >= 3 and not np.isnan(rate) and rate >= p0:
            return w, rate
    return 25, np.nan

# === 打印摘要 ===
print("=" * 130)
print("批判7处置: 直接计数 + Wilson 95% CI + 样本量标注")
print("=" * 130)

p0_list = [0.4, 0.5, 0.6, 0.7]
header = f"{'BMI组':12s} {'总n':>5s} "
for p0 in p0_list:
    header += f" {'p0='+str(p0):>18s}"
print(header)
print("-" * 130)

for lb in labels:
    cells = cell_data[lb]
    total_n = sum(c[1] for c in cells)
    row = f"{lb:12s} {total_n:5d} "
    for p0 in p0_list:
        t, r = best_time_wilson(cells, p0)
        # find CI at this week
        ci_str = ""
        for w, n, k, rate, ci_low, ci_high in cells:
            if w == t:
                ci_str = f"[{ci_low:.0%},{ci_high:.0%}]"
                break
        if not np.isnan(r):
            row += f" {t:3.0f}周@{r:.0%} {ci_str}"
        else:
            row += f" {'---':>18s}"
    print(row)

# === 详细: 每格数据 ===
print("\n" + "=" * 130)
print("逐格数据（n≥3的格子，按BMI组排列）")
print("=" * 130)
for lb in labels:
    cells = cell_data[lb]
    print(f"\n{lb}:")
    print(f"  {'孕周':>5s}  {'n':>4s}  {'达标n':>5s}  {'达标率':>7s}  {'95% Wilson CI':>18s}  {'CI宽度':>8s}  {'可靠?':>5s}")
    print(f"  {'-'*70}")
    for w, n, k, rate, ci_low, ci_high in cells:
        if n >= 1 and not np.isnan(rate):
            ci_str = f"[{ci_low:.0%}, {ci_high:.0%}]"
            ci_width = ci_high - ci_low
            reliable = "OK" if n >= 10 else (".." if n >= 3 else "XX")
            print(f"  {w:5.0f}周  {n:4d}  {k:5d}  {rate:7.0%}  {ci_str:>18s}  {ci_width:7.0%}  {reliable:>5s}")

# === 关键发现：推荐时点处的支撑 ===
print("\n" + "=" * 130)
print("推荐时点处的样本支撑（p0=0.5）")
print("=" * 130)
print(f"{'BMI组':12s} {'推荐时点':>8s} {'n':>5s} {'达标n':>6s} {'达标率':>8s} {'95% CI':>20s} {'CI能排除<50%?':>15s}")
print("-" * 90)
for lb in labels:
    cells = cell_data[lb]
    t, _ = best_time_wilson(cells, 0.5)
    for w, n, k, rate, ci_low, ci_high in cells:
        if w == t:
            ci_str = f"[{ci_low:.0%}, {ci_high:.0%}]"
            above_50 = "是" if ci_low > 0.5 else ("边缘" if ci_low > 0.4 else "否")
            print(f"{lb:12s} {t:6.0f}周   {n:5d} {k:6d} {rate:8.0%} {ci_str:>20s} {above_50:>15s}")
            break

# === 图1: 各组曲线 + Wilson CI + 样本量标注 ===
fig, axes = plt.subplots(3, 2, figsize=(18, 18))
axes = axes.flatten()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for idx, lb in enumerate(labels):
    ax = axes[idx]
    cells = cell_data[lb]
    total_n = sum(c[1] for c in cells)

    weeks_arr = np.array([c[0] for c in cells])
    rates_arr = np.array([c[3] for c in cells])
    ci_low_arr = np.array([c[4] for c in cells])
    ci_high_arr = np.array([c[5] for c in cells])
    ns_arr = np.array([c[1] for c in cells])

    # 只画 n>=3 的点
    valid = ns_arr >= 3

    if np.any(valid):
        # CI 误差棒
        ax.errorbar(weeks_arr[valid], rates_arr[valid]*100,
                   yerr=[(rates_arr[valid] - ci_low_arr[valid])*100,
                         (ci_high_arr[valid] - rates_arr[valid])*100],
                   fmt='none', ecolor=colors[idx], alpha=0.3, capsize=3, linewidth=1)

        # 散点，大小正比于 log(n)
        sizes = np.array([max(15, min(80, 10 + 5*np.log(n))) for n in ns_arr])
        ax.scatter(weeks_arr[valid], rates_arr[valid]*100, s=sizes[valid],
                  alpha=0.8, color=colors[idx], zorder=5, edgecolors='white', linewidth=0.5)

        # 在 n<10 的点上标注 n
        for i in np.where(valid)[0]:
            if ns_arr[i] < 10:
                ax.annotate(f'n={ns_arr[i]}', (weeks_arr[i], rates_arr[i]*100 + 5),
                           fontsize=7, ha='center', color='#555555',
                           bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

    # n<3 的点（空圈）
    invalid = ns_arr < 3
    if np.any(invalid):
        ax.scatter(weeks_arr[invalid], [50]*np.sum(invalid),
                  s=15, alpha=0.3, color='gray', marker='x',
                  label=f'n<3 ({np.sum(invalid)}格)')

    # p0=0.5 线
    ax.axhline(y=50, color='#333333', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(y=70, color='#999999', linestyle=':', linewidth=0.5, alpha=0.3)

    # 最佳时点标记
    t05, r05 = best_time_wilson(cells, 0.5)
    if not np.isnan(r05):
        ax.axvline(x=t05, color=colors[idx], linestyle='--', linewidth=1.2, alpha=0.7)
        ax.annotate(f'{t05:.0f}周', xy=(t05, 60), fontsize=11, color=colors[idx], ha='center',
                   fontweight='bold')

    # 标注各组的推荐信息
    t07, r07 = best_time_wilson(cells, 0.7)
    info_str = f'n={total_n} | p0=0.5: {t05:.0f}周 | p0=0.7: {t07:.0f}周'
    ax.text(0.02, 0.96, info_str, transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('孕周', fontsize=11)
    ax.set_ylabel('Y>=4% 达标率 (%)', fontsize=11)
    ax.set_title(f'{lb}', fontsize=13, fontweight='bold')
    ax.set_xlim(9.5, 25.5); ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

# 隐藏多余子图
for idx in range(len(labels), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('各BMI组达标率 — 直接计数 + Wilson 95%CI + 样本量标注\n'
            '点大小∝log(n), n<10标注数字, 误差棒=95% Wilson CI',
            fontsize=15, y=0.99, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'sub2-wilson-ci.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(chart_dir, 'sub2-wilson-ci.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('\n图1已保存: sub2-wilson-ci.pdf (分格)')

# === 图2: 论文用合并大图 ===
fig, ax = plt.subplots(figsize=(13, 8))

for idx, lb in enumerate(labels):
    cells = cell_data[lb]
    weeks_arr = np.array([c[0] for c in cells])
    rates_arr = np.array([c[3] for c in cells])
    ci_low_arr = np.array([c[4] for c in cells])
    ci_high_arr = np.array([c[5] for c in cells])
    ns_arr = np.array([c[1] for c in cells])
    valid = ns_arr >= 3

    if np.any(valid):
        # CI band
        ax.fill_between(weeks_arr[valid], ci_low_arr[valid]*100, ci_high_arr[valid]*100,
                       alpha=0.12, color=colors[idx])
        # Line connecting points
        ax.plot(weeks_arr[valid], rates_arr[valid]*100, 'o-', color=colors[idx],
               linewidth=1.8, markersize=7, label=f'{lb} (n={sum(ns_arr)})', zorder=5-idx)

        # 最佳时点
        t05, _ = best_time_wilson(cells, 0.5)
        if not np.isnan(t05):
            ax.axvline(x=t05, color=colors[idx], linestyle='--', linewidth=0.8, alpha=0.4)
            ax.text(t05+0.2, 97-idx*7, f'{t05:.0f}周', fontsize=8, color=colors[idx])

ax.axhline(y=50, color='#333333', linestyle=':', linewidth=0.8, alpha=0.5, label='p0=0.5')
ax.axhline(y=70, color='#999999', linestyle=':', linewidth=0.5, alpha=0.3, label='p0=0.7')
ax.set_xlabel('孕周', fontsize=13)
ax.set_ylabel('Y>=4% 达标率 (%)', fontsize=13)
ax.set_title('各BMI组达标率曲线 (直接计数 + Wilson 95% CI)', fontsize=14)
ax.set_xlim(9.5, 25.5); ax.set_ylim(-5, 105)
ax.legend(fontsize=9, loc='lower right', ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'sub2-wilson-all.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(chart_dir, 'sub2-wilson-all.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('图2已保存: sub2-wilson-all.pdf (合并)')

# === 图3: 每格样本量热力图 ===
fig, ax = plt.subplots(figsize=(14, 5))
heat_n = np.zeros((len(labels), len(weeks)))
for i, lb in enumerate(labels):
    cells = cell_data[lb]
    for j, (w, n, k, rate, ci_low, ci_high) in enumerate(cells):
        heat_n[i, j] = n

# log scale for color
heat_display = np.log1p(heat_n)
im = ax.imshow(heat_display, cmap='YlOrRd', aspect='auto')
for i in range(len(labels)):
    for j in range(len(weeks)):
        n = heat_n[i, j]
        if n > 0:
            color = 'white' if n >= 20 else 'black'
            ax.text(j, i, f'{n:.0f}', ha='center', va='center', fontsize=8, color=color)
        else:
            ax.text(j, i, '0', ha='center', va='center', fontsize=7, color='#CCCCCC')

ax.set_xticks(range(len(weeks)))
ax.set_xticklabels([f'{w:.0f}' for w in weeks], fontsize=7)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('孕周', fontsize=12)
ax.set_title('各BMI组×孕周的样本量分布 (热力图)', fontsize=13)
plt.colorbar(im, ax=ax, label='log(n+1)', shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'sub2-cell-sizes-heatmap.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(chart_dir, 'sub2-cell-sizes-heatmap.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('图3已保存: sub2-cell-sizes-heatmap.pdf')

# === 对比表：Wilson方法 vs 原始直接计数 ===
print("\n" + "=" * 130)
print("方法对比: 原始直接计数 vs 直接计数+Wilson CI")
print("=" * 130)
print(f"{'BMI组':12s} {'总n':>5s} {'原始p0=0.5':>12s} {'原始p0=0.7':>12s} {'Wilson p0=0.5':>16s} {'Wilson p0=0.7':>16s} {'一致性':>8s}")
print("-" * 100)

# 原始直接计数（复用之前的逻辑）
dc_rates = {}
for lb in labels:
    rates = []
    for w in weeks:
        mask = (male['bmi_group'] == lb) & (male['gw'] >= w-0.5) & (male['gw'] < w+0.5)
        n = mask.sum()
        if n >= 3:
            rates.append((male.loc[mask, 'Y染色体浓度'] >= 0.04).mean())
        else:
            rates.append(np.nan)
    dc_rates[lb] = np.array(rates)

def best_time_dc(rate_arr, p0=0.5):
    for i, r in enumerate(rate_arr):
        if not np.isnan(r) and r >= p0:
            return weeks[i], r
    return 25, rate_arr[~np.isnan(rate_arr)][-1] if np.any(~np.isnan(rate_arr)) else np.nan

for lb in labels:
    cells = cell_data[lb]
    total_n = sum(c[1] for c in cells)
    dc_t05, dc_r05 = best_time_dc(dc_rates[lb], 0.5)
    dc_t07, dc_r07 = best_time_dc(dc_rates[lb], 0.7)
    wc_t05, wc_r05 = best_time_wilson(cells, 0.5)
    wc_t07, wc_r07 = best_time_wilson(cells, 0.7)

    # get CI strings for Wilson
    wc_t05_ci = ""
    for w, n, k, rate, ci_low, ci_high in cells:
        if w == wc_t05:
            wc_t05_ci = f"[{ci_low:.0%},{ci_high:.0%}]"
            break
    wc_t07_ci = ""
    for w, n, k, rate, ci_low, ci_high in cells:
        if w == wc_t07:
            wc_t07_ci = f"[{ci_low:.0%},{ci_high:.0%}]"
            break

    consistent = ""
    if dc_t05 == wc_t05:
        consistent += "p0=0.5一致"
    else:
        consistent += f"p05差{abs(dc_t05-wc_t05):.0f}w"

    print(f"{lb:12s} {total_n:5d} {dc_t05:6.0f}周@{dc_r05:.0%}  {dc_t07:6.0f}周@{dc_r07:.0%}  "
          f"{wc_t05:6.0f}周@{wc_r05:.0%} {wc_t05_ci}  {wc_t07:6.0f}周@{wc_r07:.0%} {wc_t07_ci}  {consistent:>8s}")

print("\n完成")

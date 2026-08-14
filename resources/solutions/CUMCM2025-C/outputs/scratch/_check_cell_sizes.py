#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""量化批判7: 各BMI组×孕周格子的样本量分布"""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings('ignore')

cache_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\data'
male = pd.read_pickle(os.path.join(cache_dir, '2025C-male-clean.pkl'))

def pgw(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    for sep in ['w+','W+']:
        if sep in s:
            p = s.split(sep); return float(p[0])+float(p[1])/7.0
    return float(s.replace('w','').replace('W',''))

male['gw'] = male['检测孕周'].apply(pgw)

bins = [0, 28, 32, 36, 40, 100]
labels = ['[20,28)', '[28,32)', '[32,36)', '[36,40)', '40+']
male['bmi_group'] = pd.cut(male['孕妇BMI'], bins=bins, labels=labels, right=False)

# === 逐格统计 ===
weeks = np.arange(10, 26)
print("=" * 120)
print("各BMI组×孕周格子的样本量与达标率")
print("=" * 120)

for lb in labels:
    print(f"\n{'-' * 80}")
    print(f"BMI组: {lb}  |  总记录: {(male['bmi_group']==lb).sum()}  |  总人数: {male[male['bmi_group']==lb]['孕妇代码'].nunique()}")
    print(f"{'孕周':>6s}  {'n':>5s}  {'达标n':>6s}  {'达标率':>8s}  {'Y均值':>8s}  {'Y中位':>8s}  {'稳定?':>6s}")
    print('-' * 60)
    for w in weeks:
        mask = (male['bmi_group'] == lb) & (male['gw'] >= w-0.5) & (male['gw'] < w+0.5)
        n = mask.sum()
        pass_n = (male.loc[mask, 'Y染色体浓度'] >= 0.04).sum()
        rate = pass_n / n if n > 0 else np.nan
        y_mean = male.loc[mask, 'Y染色体浓度'].mean() if n > 0 else np.nan
        y_med = male.loc[mask, 'Y染色体浓度'].median() if n > 0 else np.nan
        stable = "OK" if n >= 10 else (".." if n >= 3 else "XX")
        rate_str = f"{rate:.0%}" if not np.isnan(rate) else "N/A"
        y_mean_str = f"{y_mean:.3f}" if not np.isnan(y_mean) else "N/A"
        y_med_str = f"{y_med:.3f}" if not np.isnan(y_med) else "N/A"
        print(f"  {w:3.0f}周  {n:5d}  {pass_n:6d}  {rate_str:>8s}  {y_mean_str:>8s}  {y_med_str:>8s}  {stable:>6s}")

# === 汇总: 各组各周的样本量分布 ===
print("\n\n" + "=" * 120)
print("样本量分布摘要")
print("=" * 120)
print(f"{'BMI组':12s}  {'总n':>5s}  {'最小格n':>8s}  {'最大格n':>8s}  {'中位格n':>8s}  {'n<3格数':>8s}  {'n<5格数':>8s}  {'n<10格数':>8s}  {'总格数':>6s}")
print("-" * 90)
for lb in labels:
    cell_sizes = []
    for w in weeks:
        mask = (male['bmi_group'] == lb) & (male['gw'] >= w-0.5) & (male['gw'] < w+0.5)
        cell_sizes.append(mask.sum())
    cell_sizes = np.array(cell_sizes)
    total = (male['bmi_group'] == lb).sum()
    print(f"{lb:12s}  {total:5d}  {cell_sizes.min():8d}  {cell_sizes.max():8d}  {np.median(cell_sizes):8.0f}  {(cell_sizes < 3).sum():8d}  {(cell_sizes < 5).sum():8d}  {(cell_sizes < 10).sum():8d}  {len(cell_sizes):6d}")

# === 关键: 每个BMI组的推荐时点处的格子样本量 ===
print("\n\n" + "=" * 120)
print("推荐时点处的格子样本量（最关键的格子）")
print("=" * 120)
print(f"{'BMI组':12s}  {'p0=0.5时点':>10s}  {'该格n':>6s}  {'该格达标n':>8s}  {'该格达标率':>10s}  {'上一格达标率':>10s}  {'下一格达标率':>10s}")
print("-" * 80)

for lb in labels:
    rs = []
    for w in weeks:
        mask = (male['bmi_group'] == lb) & (male['gw'] >= w-0.5) & (male['gw'] < w+0.5)
        n = mask.sum()
        if n >= 3:
            rs.append((male.loc[mask, 'Y染色体浓度'] >= 0.04).mean())
        else:
            rs.append(np.nan)
    rs = np.array(rs)

    # find best time at p0=0.5
    t05 = None
    for i, r in enumerate(rs):
        if not np.isnan(r) and r >= 0.5:
            t05 = weeks[i]
            break
    if t05 is None:
        t05 = weeks[~np.isnan(rs)][-1] if np.any(~np.isnan(rs)) else 25

    # get cell sizes at t05-1, t05, t05+1
    idx = np.where(weeks == t05)[0][0] if t05 in weeks else -1
    for offset in [-1, 0, 1]:
        w_target = t05 + offset
        if w_target < 10 or w_target > 25: continue
        mask_target = (male['bmi_group'] == lb) & (male['gw'] >= w_target-0.5) & (male['gw'] < w_target+0.5)
        n_target = mask_target.sum()
        pass_n_target = (male.loc[mask_target, 'Y染色体浓度'] >= 0.04).sum()
        rate_target = pass_n_target / n_target if n_target > 0 else np.nan

        label_w = f"{'时点' if offset == 0 else ('-1周' if offset == -1 else '+1周')}"
        if offset == 0:
            print(f"{lb:12s}  {t05:6.0f}周(时点)  {n_target:6d}  {pass_n_target:8d}  {rate_target:10.0%}")
        else:
            print(f"{'':12s}  {w_target:6.0f}周({label_w})  {n_target:6d}  {pass_n_target:8d}  {rate_target:10.0%}")

print("\n完成")

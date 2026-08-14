#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批判7验证: Logistic回归估计达标率 vs 直接计数"""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

cache_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\data'
chart_dir = 'E:\\MathModel\\problems\\2025\\C题\\solution\\artifacts\\charts'
fig_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\figures'

male = pd.read_pickle(os.path.join(cache_dir, '2025C-male-clean.pkl'))

# 直接使用已清理的孕周_数值列
male['gw'] = male['孕周_数值']

bins = [0, 28, 32, 36, 40, 100]
labels = ['[20,28)', '[28,32)', '[32,36)', '[36,40)', '40+']
male['bmi_group'] = pd.cut(male['孕妇BMI'], bins=bins, labels=labels, right=False)
male['y_pass'] = (male['Y染色体浓度'] >= 0.04).astype(int)

# === 1. 直接计数（现有方法）===
weeks = np.arange(10, 26)
dc_rates = {}
for lb in labels:
    rates = []
    counts = []
    for w in weeks:
        mask = (male['bmi_group'] == lb) & (male['gw'] >= w-0.5) & (male['gw'] < w+0.5)
        n = mask.sum()
        if n >= 3:
            rates.append((male.loc[mask, 'Y染色体浓度'] >= 0.04).mean())
            counts.append(n)
        else:
            rates.append(np.nan)
            counts.append(n)
    dc_rates[lb] = np.array(rates)

def best_time_dc(rate_arr, p0=0.5):
    for i, r in enumerate(rate_arr):
        if not np.isnan(r) and r >= p0:
            return weeks[i], r
    return 25, rate_arr[~np.isnan(rate_arr)][-1] if np.any(~np.isnan(rate_arr)) else np.nan

# === 2. Logistic回归 ===
logit_results = {}
for lb in labels:
    g = male[male['bmi_group'] == lb]
    n_total = len(g)
    n_individuals = g['孕妇代码'].nunique()
    X = g[['gw']].values
    y = g['y_pass'].values

    if len(g) < 5:
        logit_results[lb] = {'status': 'skip', 'n': n_total, 'n_ind': n_individuals}
        continue

    # statsmodels Logit
    X_sm = sm.add_constant(X)
    try:
        model = sm.Logit(y, X_sm)
        result = model.fit(disp=0)
        params = result.params
        coef = float(params[1])
        intercept = float(params[0])
    except Exception as e:
        print(f"  [{lb}] Logit failed: {e}")
        logit_results[lb] = {'status': 'skip', 'n': n_total, 'n_ind': n_individuals}
        continue

    # Predict on fine grid
    gw_grid = np.linspace(10, 25, 200)
    X_pred = sm.add_constant(gw_grid)
    prob_grid = result.predict(X_pred)

    # Integer week predictions
    gw_int = np.arange(10, 26).astype(float)
    X_int = sm.add_constant(gw_int)
    prob_int = result.predict(X_int)

    # Hosmer-Lemeshow test (decile-based)
    proba = result.predict(X_sm)
    n_bins = min(10, max(3, n_total // 10))
    try:
        bins_hl = np.percentile(proba, np.linspace(0, 100, n_bins+1))
        hl_stat_val = 0
        for i in range(n_bins):
            if i < n_bins - 1:
                in_bin = (proba >= bins_hl[i]) & (proba < bins_hl[i+1])
            else:
                in_bin = (proba >= bins_hl[i]) & (proba <= bins_hl[i+1])
            n_g = in_bin.sum()
            o_g = y[in_bin].sum()
            e_g = proba[in_bin].sum()
            if e_g > 0 and (n_g - e_g) > 0:
                hl_stat_val += (o_g - e_g)**2 / (e_g * (1 - e_g/n_g))
        hl_p = 1 - stats.chi2.cdf(hl_stat_val, n_bins - 2)
    except:
        hl_stat_val, hl_p = np.nan, np.nan

    # Best integer week for each p0
    best_int = {}
    for p0 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]:
        idx = np.where(prob_int >= p0)[0]
        if len(idx) > 0:
            best_int[p0] = (gw_int[idx[0]], prob_int[idx[0]])
        else:
            best_int[p0] = (25.0, prob_int[-1])

    # Bootstrap CI
    bs_probs = []
    n_bs = 500
    rng = np.random.RandomState(42)
    for _ in range(n_bs):
        idx_bs = rng.choice(n_total, size=n_total, replace=True)
        X_bs, y_bs = X_sm[idx_bs], y[idx_bs]
        try:
            m_bs = sm.Logit(y_bs, X_bs)
            r_bs = m_bs.fit(disp=0)
            bs_probs.append(r_bs.predict(X_pred))
        except:
            pass
    if len(bs_probs) > 10:
        bs_probs = np.array(bs_probs)
        ci_lower = np.percentile(bs_probs, 2.5, axis=0)
        ci_upper = np.percentile(bs_probs, 97.5, axis=0)
    else:
        ci_lower, ci_upper = None, None

    logit_results[lb] = {
        'status': 'ok',
        'n': n_total, 'n_ind': n_individuals,
        'coef': coef, 'intercept': intercept,
        'hl_stat': hl_stat_val, 'hl_p': hl_p,
        'gw_grid': gw_grid, 'prob_grid': prob_grid,
        'gw_int': gw_int, 'prob_int': prob_int,
        'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'best_int': best_int
    }

# === 3. 打印对比 ===
print("=" * 120)
print("Logistic回归 vs 直接计数 — 最佳时点对比")
print("=" * 120)

p0_list = [0.4, 0.5, 0.6, 0.7]
header = f"{'BMI组':12s} {'n':>5s} {'方法':12s}"
for p0 in p0_list:
    header += f" {'p0='+str(p0):>10s}"
header += f"  {'HL p':>8s}"
print(header)
print("-" * 120)

for lb in labels:
    r = logit_results[lb]
    if r['status'] == 'skip':
        print(f"{lb:12s} {r['n']:5d} {'Logistic:':12s} (样本不足，跳过)")
        continue
    # DC results
    rs = dc_rates[lb]
    dc_str = f"{lb:12s} {r['n']:5d} {'直接计数:':12s}"
    for p0 in p0_list:
        t, prob = best_time_dc(rs, p0)
        dc_str += f" {t:4.0f}周@{prob:.0%}"
    print(dc_str)

    # Logistic results (integer week)
    lg_str = f"{'':12s} {'':5s} {'Logistic:':12s}"
    for p0 in p0_list:
        t, prob = r['best_int'][p0]
        lg_str += f" {t:4.0f}周@{prob:.0%}"
    lg_str += f"  p={r['hl_p']:.3f}" if not np.isnan(r['hl_p']) else f"  {'N/A':>8s}"
    print(lg_str)
    print()

# === 4. 图: 达标率曲线（Logistic + CI + 原始计数点） ===
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes = axes.flatten()

for idx, lb in enumerate(labels):
    ax = axes[idx]
    r = logit_results[lb]

    # 原始直接计数点
    rs = dc_rates[lb]
    valid = ~np.isnan(rs)
    if np.any(valid):
        cell_ns = np.array([((male['bmi_group']==lb) & (male['gw']>=w-0.5) & (male['gw']<w+0.5)).sum() for w in weeks])
        sizes = np.array([20 + 5*np.log(n) if n > 0 else 10 for n in cell_ns])
        ax.scatter(weeks[valid], rs[valid]*100, s=sizes[valid], alpha=0.6,
                  color='#1f77b4', zorder=5, label=f'直接计数 (n={r["n"]})')

    if r['status'] == 'ok':
        gw = r['gw_grid']
        prob = r['prob_grid']
        ax.plot(gw, prob*100, color='#d62728', linewidth=2, label='Logistic回归')
        # CI band
        if r['ci_lower'] is not None:
            ax.fill_between(gw, r['ci_lower']*100, r['ci_upper']*100,
                          alpha=0.15, color='#d62728', label='95% CI (Bootstrap)')
        # p0=0.5 line
        ax.axhline(y=50, color='#333333', linestyle=':', linewidth=0.8, alpha=0.5, label='p0=0.5')
        # optimal time marker
        t05, p05 = r['best_int'][0.5]
        ax.axvline(x=t05, color='#d62728', linestyle='--', linewidth=1, alpha=0.7)
        ax.annotate(f'{t05:.0f}周', xy=(t05, 55), fontsize=10, color='#d62728', ha='center')

        # 标注回归参数
        ax.text(0.98, 0.05,
                f'logit(P) = {r["intercept"]:.3f} + {r["coef"]:.3f}*week\n'
                f'HL test: p={r["hl_p"]:.3f}',
                transform=ax.transAxes, fontsize=8, va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.set_xlabel('孕周', fontsize=11)
    ax.set_ylabel('Y>=4% 达标概率 (%)', fontsize=11)
    ax.set_title(f'{lb} (n={r["n"]}, {r.get("n_ind", r.get("n_ind","?"))}人)', fontsize=12)
    ax.set_xlim(9.5, 25.5); ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

# 删除多余的子图
for idx in range(len(labels), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Logistic回归 vs 直接计数 — 各BMI组达标概率曲线', fontsize=14, y=0.99)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'sub2-logistic-vs-direct.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(chart_dir, 'sub2-logistic-vs-direct.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('\n图已保存: sub2-logistic-vs-direct.pdf')

# === 5. 单独一张对比大图（用于论文） ===
fig, ax = plt.subplots(figsize=(12, 7))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for idx, lb in enumerate(labels):
    r = logit_results[lb]
    if r['status'] == 'skip': continue
    gw = r['gw_grid']
    prob = r['prob_grid']
    ax.plot(gw, prob*100, color=colors[idx], linewidth=2, label=lb)
    # CI band (lighter)
    if r['ci_lower'] is not None:
        ax.fill_between(gw, r['ci_lower']*100, r['ci_upper']*100,
                       alpha=0.1, color=colors[idx])
    # optimal time
    t05, _ = r['best_int'][0.5]
    ax.axvline(x=t05, color=colors[idx], linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(t05+0.2, 95-idx*8, f'{t05:.0f}周', fontsize=8, color=colors[idx])

ax.axhline(y=50, color='#333333', linestyle=':', linewidth=0.8, alpha=0.5, label='p0=0.5')
ax.set_xlabel('孕周', fontsize=12)
ax.set_ylabel('Y>=4% 达标概率 (%)', fontsize=12)
ax.set_title('Logistic回归估计的各BMI组达标概率曲线 (95% Bootstrap CI)', fontsize=13)
ax.set_xlim(9.5, 25.5); ax.set_ylim(-5, 105)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'sub2-logistic-all-groups.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(chart_dir, 'sub2-logistic-all-groups.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('图已保存: sub2-logistic-all-groups.pdf')

# === 6. 总结表：Logistic回归的推荐时点 ===
print("\n\n" + "=" * 120)
print("最终推荐时点（Logistic回归，取整周）")
print("=" * 120)
print(f"{'BMI组':12s} {'n':>5s} {'p0=0.5':>10s} {'达标概率':>10s} {'p0=0.7':>10s} {'达标概率':>10s} {'HL p':>8s}")
print("-" * 80)
for lb in labels:
    r = logit_results[lb]
    if r['status'] == 'skip':
        print(f"{lb:12s} {r['n']:5d} (样本不足)") ; continue
    t05, p05 = r['best_int'][0.5]
    t07, p07 = r['best_int'][0.7]
    hl_p = f"{r['hl_p']:.3f}" if not np.isnan(r['hl_p']) else "N/A"
    print(f"{lb:12s} {r['n']:5d} {t05:6.0f}周    {p05:10.0%} {t07:6.0f}周    {p07:10.0%} {hl_p:>8s}")

print("\n完成")

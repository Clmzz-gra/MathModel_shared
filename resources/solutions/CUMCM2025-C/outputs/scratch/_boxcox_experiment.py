"""
Box-Cox 变换实验：对 2025 C 题 Y 染色体浓度数据搜索最优 lambda
使用 scipy.stats.boxcox 内置实现
"""
import numpy as np
import pandas as pd
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 加载数据 ==========
with open(r'E:\MathModel\problems\2025\C题\2025C题测试\outputs\data\2025C-male-clean.pkl', 'rb') as f:
    df = pickle.load(f)

y_conc = df['Y染色体浓度'].values
print(f"Y 浓度: n={len(y_conc)}, min={y_conc.min():.4f}, max={y_conc.max():.4f}")
print(f"均值={y_conc.mean():.4f}, 中位数={np.median(y_conc):.4f}")
print(f"偏度={stats.skew(y_conc):.3f}, 超额峰度={stats.kurtosis(y_conc):.3f}")

# ========== 2. Box-Cox 对数似然曲线 + MLE 估计 ==========
y_conc = df['Y染色体浓度'].values
n = len(y_conc)
log_y = np.log(y_conc)
sum_log_y = np.sum(log_y)

# 用 scipy 求 MLE
lam_hat = stats.boxcox_normmax(y_conc, method='mle')
print(f"\n========== Box-Cox 结果 ==========")
print(f"scipy MLE: 最优 lambda_hat = {lam_hat:.4f}")

# 网格搜索似然曲线
lambdas = np.linspace(-2, 2, 401)
log_likelihoods = []

for lam in lambdas:
    if abs(lam) < 1e-8:
        yt = log_y
    else:
        yt = (y_conc**lam - 1) / lam
    var_hat = np.var(yt, ddof=1)
    ll = -n/2 * np.log(var_hat) + (lam - 1) * sum_log_y
    log_likelihoods.append(ll)

log_likelihoods = np.array(log_likelihoods)
max_idx = np.argmax(log_likelihoods)
ll_max = log_likelihoods[max_idx]
lam_from_grid = lambdas[max_idx]

# Profile likelihood 95% CI
threshold = ll_max - 1.92
ci_mask = log_likelihoods >= threshold
ci_indices = np.where(ci_mask)[0]
lam_ci_low = lambdas[ci_indices[0]]
lam_ci_high = lambdas[ci_indices[-1]]

print(f"网格搜索峰值 lambda = {lam_from_grid:.4f}")
print(f"95% CI for lambda: [{lam_ci_low:.3f}, {lam_ci_high:.3f}]")
print(f"CI 包含 lambda=0? {'是 (对数变换在CI内)' if lam_ci_low <= 0 <= lam_ci_high else '否'}")
print(f"CI 包含 lambda=0.5 (平方根)? {'是' if lam_ci_low <= 0.5 <= lam_ci_high else '否'}")
print(f"CI 包含 lambda=1 (无变换)? {'是' if lam_ci_low <= 1 <= lam_ci_high else '否'}")

# ========== 3. 各 lambda 的分布对比 ==========
print(f"\n========== 不同 lambda 的变换效果 ==========")
test_lams = [-1.0, -0.5, 0.0, lam_hat, 0.5, 1.0]
test_lams = sorted(set(round(x, 4) for x in test_lams))
for lam in test_lams:
    if abs(lam) < 1e-8:
        yt = np.log(y_conc)
    else:
        yt = (y_conc**lam - 1) / lam
    sk = stats.skew(yt)
    ku = stats.kurtosis(yt)
    _, sw_p = stats.shapiro(yt)
    print(f"lambda={lam:6.2f}: 偏度={sk:8.3f}, 超额峰度={ku:7.3f}, Shapiro p={sw_p:.6f}")

# ========== 4. 绘图 ==========
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ---- 左图: 对数似然曲线 ----
ax = axes[0]
ax.plot(lambdas, log_likelihoods, 'b-', linewidth=1.5)
ax.axvline(1, color='orange', linestyle='--', alpha=0.5, label='lambda=1 (原始)')
ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='lambda=0 (ln)')
ax.axvline(lam_hat, color='red', linestyle=':', linewidth=2, label=f'最优={lam_hat:.3f}')
ax.fill_between(lambdas, threshold, ll_max,
                where=(lambdas >= lam_ci_low) & (lambdas <= lam_ci_high),
                color='red', alpha=0.15, label='95% CI')
ax.axhline(threshold, color='red', linestyle=':', alpha=0.5)
ax.set_xlabel('lambda')
ax.set_ylabel('对数似然')
ax.set_title('Box-Cox 对数似然曲线')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- 中图: 不同 lambda 的分布 vs N(0,1) ----
ax = axes[1]
plot_lam = sorted([-0.5, 0.0, lam_hat, 0.5, 1.0])
colors = ['#E74C3C', '#2ECC71', '#9B59B6', '#3498DB', '#F39C12']
x_range = np.linspace(-5, 3, 200)

for lam, c in zip(plot_lam, colors[:len(plot_lam)]):
    if abs(lam) < 1e-8:
        yt = np.log(y_conc)
    else:
        yt = (y_conc**lam - 1) / lam
    yt_z = (yt - yt.mean()) / yt.std()
    kde = stats.gaussian_kde(yt_z)
    ax.plot(x_range, kde(x_range), color=c, linewidth=1.5,
            label=f'lambda={lam:.2f}')

ax.plot(x_range, stats.norm.pdf(x_range), 'k--', linewidth=1.5,
        label='N(0,1)', alpha=0.7)
ax.set_xlabel('标准化变换值')
ax.set_ylabel('密度')
ax.set_title('变换后分布对比')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- 右图: 原始 vs 最优变换的 Q-Q 图 ----
ax = axes[2]
# 原始数据的 Q-Q 图
stats.probplot(y_conc, dist='norm', plot=None)
# 绘制最优变换后的 Q-Q 图
yt_opt = stats.boxcox(y_conc, lam_hat)
stats.probplot(yt_opt, dist='norm', plot=None)
# 手动绘制
from scipy.stats import probplot
# 原始
os1, os2 = probplot(y_conc, dist='norm', fit=True)
ax.scatter(os1[0], os1[1], c='#F39C12', s=12, alpha=0.5, label='原始数据')
# 最优变换
os1_opt, os2_opt = probplot(yt_opt, dist='norm', fit=True)
ax.scatter(os1_opt[0], os1_opt[1], c='#2ECC71', s=12, alpha=0.5, label=f'Box-Cox (lambda={lam_hat:.2f})')
# 参考线
ax.plot(os1[0], os1[0] * os2[0] + os2[1], 'orange', linestyle='--', alpha=0.3)
ax.plot(os1_opt[0], os1_opt[0] * os2_opt[0] + os2_opt[1], 'green', linestyle='--', alpha=0.3)
ax.set_xlabel('理论分位数')
ax.set_ylabel('样本分位数')
ax.set_title('Q-Q 图对比')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'E:\MathModel\problems\2025\C题\2025C题测试\outputs\figures\boxcox_result_v2.png',
            dpi=150, bbox_inches='tight')
print(f"\n图表已保存到 outputs/figures/boxcox_result_v2.png")
plt.close()

# ========== 5. 额外分析 ==========
# 5a. 对数变换与最优变换的偏差
print(f"\n========== 附加分析 ==========")
print(f"对数变换 (lambda=0) 与 Box-Cox 最优 (lambda={lam_hat:.4f}) 的偏差")
# 两种变换下的偏度
sk_ln = stats.skew(np.log(y_conc))
sk_bc = stats.skew(yt_opt)
print(f"  对数变换偏度={sk_ln:.3f}, Box-Cox 偏度={sk_bc:.3f}")

# 5b. 按孕周分组的 ln(Y) 的方差比
print(f"\n按孕周分组的 ln(Y) 标准差:")
df_temp = df[['Y染色体浓度', '孕周_数值']].dropna().copy()
df_temp['孕周组'] = pd.cut(df_temp['孕周_数值'], bins=[0, 12, 16, 20, 30],
                           labels=['<=12周', '13-16周', '17-20周', '>20周'])
for name, grp in df_temp.groupby('孕周组'):
    y = grp['Y染色体浓度'].values
    ly = np.log(y)
    print(f"  {name}: n={len(y):4d}, 原始 SD={y.std():.4f}, ln(Y) SD={ly.std():.4f}, 均值={y.mean():.4f}")

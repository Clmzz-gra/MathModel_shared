#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
问题1完整代码实现
双模型分层策略 + 中介效应分析
"""
import pandas as pd, numpy as np, os, warnings; warnings.filterwarnings('ignore')

# ===== matplotlib 前置配置 =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

# ===== 0. 加载预处理数据 =====
cache_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\data'
fig_dir = 'E:\\MathModel\\problems\\2025\\C题\\outputs\\figures'
chart_dir = 'E:\\MathModel\\problems\\2025\\C题\\solution\\artifacts\\charts'
code_dir = 'E:\\MathModel\\problems\\2025\\C题\\solution\\appendix\\code'
os.makedirs(fig_dir, exist_ok=True); os.makedirs(chart_dir, exist_ok=True); os.makedirs(code_dir, exist_ok=True)

data = pd.read_pickle(os.path.join(cache_dir, '2025C-sub1-preprocessed.pkl'))
n_obs = len(data); n_patients = data['孕妇代码'].nunique()
print(f'加载数据: {n_obs} 条, {n_patients} 人')

# 模型特征
month_cols = [c for c in data.columns if c.startswith('m_202')]
X_A_cols = ['gw_c', 'bmi_c', 'pc1'] + month_cols
X_B_cols = X_A_cols + ['x_c']
y = data['y_log'].values
Y_raw = data['Y染色体浓度'].values
groups = data['孕妇代码'].values

# ===== 1. 模型A: 全样本拟合 + 留一孕妇CV =====
print('\n=== 模型A: 主模型 ===')
X_A = data[X_A_cols].values

lr_A = LinearRegression(); lr_A.fit(X_A, y)
y_pred_A = lr_A.predict(X_A)
r2_A = r2_score(y, y_pred_A)
rmse_A = np.sqrt(mean_squared_error(y, y_pred_A))

# 显著性
n_params_A = X_A.shape[1]
X_design = np.column_stack([np.ones(n_obs), X_A])
XTX_inv = np.linalg.inv(X_design.T @ X_design)
resid_A = y - y_pred_A
se_A = np.sqrt(np.diag(XTX_inv) * np.var(resid_A))
t_A = np.concatenate([[lr_A.intercept_], lr_A.coef_]) / se_A
p_A = 2 * (1 - stats.t.cdf(np.abs(t_A), n_obs - n_params_A - 1))

# F检验
rss_A = np.sum(resid_A**2); tss_A = np.sum((y - y.mean())**2)
F_A = (tss_A - rss_A) / n_params_A / (rss_A / (n_obs - n_params_A - 1))
p_F_A = 1 - stats.f.cdf(F_A, n_params_A, n_obs - n_params_A - 1)

print(f'  R²={r2_A:.4f}, RMSE={rmse_A:.4f}')
print(f'  F({n_params_A},{n_obs-n_params_A-1})={F_A:.2f}, p={p_F_A:.2e}')
print(f'  {"参数":20s} {"系数":>8s} {"t值":>8s} {"p值":>10s}')
for i, name in enumerate(['截距'] + X_A_cols):
    label = name[:20]
    print(f'  {label:20s} {lr_A.intercept_ if i==0 else lr_A.coef_[i-1]:8.4f} {t_A[i]:8.2f} {p_A[i]:10.4f}')

# 留一孕妇CV
logo = LeaveOneGroupOut()
preds_cv = np.zeros(n_obs)
for tr_idx, te_idx in logo.split(np.arange(n_obs), groups=groups):
    lr = LinearRegression().fit(X_A[tr_idx], y[tr_idx])
    preds_cv[te_idx] = lr.predict(X_A[te_idx])
r2_cv_A = r2_score(y, preds_cv)
rmse_cv_A = np.sqrt(mean_squared_error(y, preds_cv))
print(f'  CV R²={r2_cv_A:.4f}, CV RMSE={rmse_cv_A:.4f}')

# ===== 2. 模型B: 中介分析 =====
print('\n=== 模型B: 中介分析 ===')
X_B = data[X_B_cols].values
lr_B = LinearRegression(); lr_B.fit(X_B, y)
y_pred_B = lr_B.predict(X_B)
r2_B = r2_score(y, y_pred_B)

print(f'  R²={r2_B:.4f}')
print(f'  {"系数":20s} {"模型A":>8s} {"模型B":>8s} {"变化":>8s}')
for i, name in enumerate(['截距'] + X_A_cols):
    b_A = lr_A.intercept_ if i==0 else lr_A.coef_[i-1]
    b_B = lr_B.intercept_ if i==0 else lr_B.coef_[i-1]
    if i == 0:
        print(f'  {name:20s} {b_A:8.4f} {b_B:8.4f} {"—":>8s}')
    else:
        change = b_B - b_A
        print(f'  {name:20s} {b_A:8.4f} {b_B:8.4f} {change:+8.4f}')
# X系数
i_x = list(lr_B.coef_).index(lr_B.coef_[len(X_A_cols)]) if len(X_B_cols) > len(X_A_cols) else -1
bx = lr_B.coef_[-1]
print(f'  {"X染色体浓度":20s} {"—":>8s} {bx:8.4f} {"—":>8s}')

# 中介比例
b1_A = lr_A.coef_[0]  # 孕周系数
b1_B = lr_B.coef_[0]
med_ratio = (b1_A - b1_B) / b1_A * 100
print(f'\n  中介比例(孕周): {med_ratio:.1f}%')

# ===== 3. 制图 =====

# --- 图1: 相关性热力图 ---
print('\n--- 制图1: 相关性热力图 ---')
corr_vars_in = ['Y染色体浓度', 'gw', '孕妇BMI', 'X染色体浓度']
corr_labels = ['Y浓度', '孕周', 'BMI', 'X浓度']
corr_data = pd.DataFrame({l: data[v] for l,v in zip(corr_labels, corr_vars_in)}).corr(method='spearman')

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_data.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        ax.text(j, i, f'{corr_data.values[i,j]:.2f}', ha='center', va='center',
                fontsize=9, color='white' if abs(corr_data.values[i,j]) > 0.5 else 'black')
ax.set_xticks(range(len(corr_labels))); ax.set_yticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(corr_labels, fontsize=9)
ax.set_title('Spearman相关矩阵', fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(f'{fig_dir}/sub1-correlation-heatmap.pdf', dpi=150, bbox_inches='tight')
plt.savefig(f'{chart_dir}/sub1-correlation-heatmap.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('  已保存')

# --- 图2: 残差Q-Q图 ---
print('--- 制图2: 残差Q-Q图 ---')
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# 左: Q-Q
ax = axes[0]
stats.probplot(resid_A, dist='norm', plot=ax)
ax.get_lines()[0].set_color('#333333'); ax.get_lines()[1].set_color('#d62728')
ax.set_title('残差Q-Q图', fontsize=11)

# 右: 残差vs拟合值
ax = axes[1]
ax.scatter(y_pred_A, resid_A, s=8, alpha=0.4, color='#1f77b4', edgecolors='none')
ax.axhline(y=0, color='#333333', linestyle='--', linewidth=0.5)
ax.set_xlabel('拟合值 (log-Y)'); ax.set_ylabel('残差'); ax.set_title('残差 vs 拟合值')

plt.tight_layout()
plt.savefig(f'{fig_dir}/sub1-residual-diagnostics.pdf', dpi=150, bbox_inches='tight')
plt.savefig(f'{chart_dir}/sub1-residual-diagnostics.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('  已保存')

# --- 图3: 个体轨迹示例 ---
print('--- 制图3: 个体轨迹示例 ---')
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
# 选9个代表: 递增、波动、递减各3
patient_info = data.groupby('孕妇代码').agg(
    n=('gw','count'), Y_first=('Y染色体浓度','first'), Y_last=('Y染色体浓度','last'),
    gw_first=('gw','first'), gw_last=('gw','last'), BMI=('孕妇BMI','first'))
patient_info['slope'] = (patient_info['Y_last'] - patient_info['Y_first']) / (patient_info['gw_last'] - patient_info['gw_first'])
patient_info = patient_info[patient_info['n'] >= 4].dropna()
up = patient_info.nlargest(3, 'slope')
down = patient_info.nsmallest(3, 'slope')
mid = patient_info.iloc[(patient_info['slope'] - patient_info['slope'].median()).abs().argsort()[:3]]
codes = list(up.index) + list(mid.index) + list(down.index)
for idx, code in enumerate(codes):
    ax = axes[idx // 3, idx % 3]
    sub = data[data['孕妇代码'] == code].sort_values('gw')
    ax.plot(sub['gw'], sub['Y染色体浓度']*100, 'o-', color='#1f77b4', linewidth=1.5, markersize=5)
    ax.axhline(y=4, color='#d62728', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_title(f'{code} (BMI={sub["孕妇BMI"].iloc[0]:.1f})', fontsize=9)
    ax.set_xlabel('孕周'); ax.set_ylabel('Y浓度(%)')
plt.suptitle('个体Y浓度轨迹示例', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{fig_dir}/sub1-individual-trajectories.pdf', dpi=150, bbox_inches='tight')
plt.savefig(f'{chart_dir}/sub1-individual-trajectories.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('  已保存')

# --- 图4: 模型系数森林图 ---
print('--- 制图4: 系数森林图 ---')
core_names = ['孕周(个体内)', 'BMI', '技术PC1']
core_coefs = lr_A.coef_[:3]
core_se = se_A[1:4]
core_ci = 1.96 * core_se

fig, ax = plt.subplots(figsize=(8, 3))
y_pos = range(len(core_names))
ax.barh(y_pos, core_coefs, xerr=core_ci, color=['#2ca02c','#d62728','#1f77b4'], height=0.5, capsize=4)
ax.axvline(x=0, color='#333333', linewidth=0.5)
ax.set_yticks(y_pos); ax.set_yticklabels(core_names, fontsize=10)
ax.set_xlabel('系数估计值 (95% CI)'); ax.set_title('模型A核心系数', fontsize=11)
# 标注
for i, (coef, ci) in enumerate(zip(core_coefs, core_ci)):
    sign = '+' if coef > 0 else ''
    ax.text(coef + ci + 0.002, i, f'{sign}{coef:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{fig_dir}/sub1-coefficient-forest.pdf', dpi=150, bbox_inches='tight')
plt.savefig(f'{chart_dir}/sub1-coefficient-forest.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('  已保存')

# --- 图5: 中介效应示意图 ---
print('--- 制图5: 中介效应示意图 ---')
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

# 节点
nodes = {'gw': (1, 4.5), 'bmi': (1, 1.5), 'dna': (5, 4.5), 'Y': (9, 5), 'Y_lower': (9, 3)}
for name, (x, y) in nodes.items():
    circle = plt.Circle((x, y), 0.4, color='#1f77b4' if name in ['gw','bmi'] else '#d62728' if name == 'Y' else '#2ca02c', alpha=0.2)
    ax.add_patch(circle)
    labels = {'gw': '孕周↑', 'bmi': 'BMI↑', 'dna': '胎儿DNA\n总浓度↑', 'Y': 'Y浓度↑'}
    if name in labels:
        ax.text(x, y, labels[name], ha='center', va='center', fontsize=10 if name != 'dna' else 8, fontweight='bold')

# 箭头
arrows = [
    ((1.4, 4.5), (4.6, 4.5), '+63%', '#2ca02c'),
    ((1.4, 4.3), (8.6, 5.3), '+37%(直接)', '#1f77b4'),
    ((1.4, 1.5), (4.6, 1.5), '', '#d62728'),
    ((1.4, 1.7), (8.6, 3.3), '-100%(稀释)', '#d62728'),
    ((5.4, 4.5), (8.6, 5.0), 'ρ=0.47', '#333333'),
]
for (x1,y1),(x2,y2),label,color in arrows:
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=color, lw=2))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+0.1, label, ha='center', fontsize=9, color=color)

# 标注
ax.text(1, 6, '自变量', ha='center', fontsize=11, fontweight='bold')
ax.text(5, 6, '中介变量', ha='center', fontsize=11, fontweight='bold')
ax.text(9, 6, '因变量', ha='center', fontsize=11, fontweight='bold')
ax.text(5, 0.5, '孕周 → Y浓度的中介效应路径', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{fig_dir}/sub1-mediation-diagram.pdf', dpi=150, bbox_inches='tight')
plt.savefig(f'{chart_dir}/sub1-mediation-diagram.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('  已保存')

# --- 图6: 月份批次效应 ---
print('--- 制图6: 月份批次效应 ---')
month_data = data.groupby('月份').agg(
    Y_mean=('Y染色体浓度','mean'), Y_std=('Y染色体浓度','std'),
    n=('Y染色体浓度','count'), 达标率=('Y染色体浓度',lambda x:(x>=0.04).mean()))
month_data = month_data.sort_index()

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
# 上: Y均值 + 标准误
ax = axes[0]
x_pos = range(len(month_data))
se = month_data['Y_std'] / np.sqrt(month_data['n'])
ax.bar(x_pos, month_data['Y_mean']*100, yerr=se*100, color='#1f77b4', alpha=0.7, capsize=3)
ax.axhline(y=data['Y染色体浓度'].mean()*100, color='#333333', linestyle='--', linewidth=0.8)
ax.set_ylabel('Y浓度均值(%)'); ax.set_title('各月份Y染色体浓度均值(±1 SE)')
# 下: 达标率
ax = axes[1]
ax.bar(x_pos, month_data['达标率']*100, color='#d62728', alpha=0.7)
ax.axhline(y=data['Y染色体浓度'].ge(0.04).mean()*100, color='#333333', linestyle='--', linewidth=0.8)
ax.set_ylabel('达标率(%)'); ax.set_title('各月份Y≥4%达标比例')
ax.set_xticks(x_pos); ax.set_xticklabels([str(m) for m in month_data.index], rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(f'{fig_dir}/sub1-monthly-batch-effect.pdf', dpi=150, bbox_inches='tight')
plt.savefig(f'{chart_dir}/sub1-monthly-batch-effect.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('  已保存')

# --- 图7: 首次达标时间 vs BMI ---
print('--- 制图7: 首次达标时间 vs BMI ---')
first_pass = data[data['Y染色体浓度']>=0.04].groupby('孕妇代码')['gw'].min().reset_index()
first_pass.columns = ['孕妇代码', '首次达标孕周']
bmi_info = data[['孕妇代码','孕妇BMI']].drop_duplicates('孕妇代码')
fp = first_pass.merge(bmi_info, on='孕妇代码')

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(fp['孕妇BMI'], fp['首次达标孕周'], alpha=0.5, s=15, color='#1f77b4', edgecolors='none')
z = np.polyfit(fp['孕妇BMI'], fp['首次达标孕周'], 1)
bmi_range = np.linspace(fp['孕妇BMI'].min(), fp['孕妇BMI'].max(), 50)
ax.plot(bmi_range, np.poly1d(z)(bmi_range), 'r-', linewidth=2, label=f'趋势(BMI每+1→晚{z[0]:.2f}周)')
# 分组中位数
buckets = [(20,28),(28,32),(32,36),(36,40),(40,65)]
for lo,hi in buckets:
    m = fp[(fp['孕妇BMI']>=lo)&(fp['孕妇BMI']<hi)]
    if len(m)>=2:
        ax.scatter([(lo+hi)/2], [m['首次达标孕周'].median()], color='k', s=60, marker='D', zorder=5)
ax.axhline(y=12, color='#2ca02c', linestyle='--', alpha=0.6, label='低风险期(≤12周)')
ax.set_xlabel('BMI'); ax.set_ylabel('首次达标孕周(周)'); ax.set_title('首次Y≥4%达标孕周 vs BMI')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{fig_dir}/sub1-first-pass-vs-bmi.pdf', dpi=150, bbox_inches='tight')
plt.savefig(f'{chart_dir}/sub1-first-pass-vs-bmi.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('  已保存')

# ===== 4. 写入结果表 =====
print('\n=== 写入结果表 ===')
tables_dir = 'E:\\MathModel\\problems\\2025\\C题\\solution\\artifacts\\tables'
os.makedirs(tables_dir, exist_ok=True)

# 表1: 模型系数
with open(f'{tables_dir}/sub1-model-coefficients.tex', 'w', encoding='utf-8') as f:
    f.write(r'\begin{tabular}{lrrrr}' + '\n')
    f.write(r'\toprule' + '\n')
    f.write(r'变量 & 系数 & t值 & p值 & 95\%CI \\' + '\n')
    f.write(r'\midrule' + '\n')
    names_a = ['孕周(个体内)', 'BMI', '技术PC1']
    for i, name in enumerate(names_a):
        lo = lr_A.coef_[i] - 1.96*se_A[i+1]
        hi = lr_A.coef_[i] + 1.96*se_A[i+1]
        stars = '***' if p_A[i+1]<0.001 else '**' if p_A[i+1]<0.01 else '*' if p_A[i+1]<0.05 else ''
        f.write(f'{name} & {lr_A.coef_[i]:.4f}{stars} & {t_A[i+1]:.2f} & {p_A[i+1]:.4f} & [{lo:.4f}, {hi:.4f}] \\\\\n')
    f.write(r'\midrule' + '\n')
    f.write(r'$R^2$ & \multicolumn{4}{r}{' + f'{r2_A:.3f}' + '} \\\\\n')
    f.write(r'CV $R^2$ & \multicolumn{4}{r}{' + f'{r2_cv_A:.3f}' + '} \\\\\n')
    f.write(r'\bottomrule' + '\n')
    f.write(r'\end{tabular}' + '\n')
print(f'  表1已写入')

# 表2: 中介分析
with open(f'{tables_dir}/sub1-mediation-table.tex', 'w', encoding='utf-8') as f:
    f.write(r'\begin{tabular}{lrrr}' + '\n')
    f.write(r'\toprule' + '\n')
    f.write(r'效应 & 不含X(模型A) & 含X(模型B) & 变化 \\' + '\n')
    f.write(r'\midrule' + '\n')
    f.write(f'孕周系数 & {lr_A.coef_[0]:.4f} & {lr_B.coef_[0]:.4f} & {lr_B.coef_[0]-lr_A.coef_[0]:+.4f} \\\\\n')
    f.write(f'BMI系数 & {lr_A.coef_[1]:.4f} & {lr_B.coef_[1]:.4f} & {lr_B.coef_[1]-lr_A.coef_[1]:+.4f} \\\\\n')
    f.write(f'X浓度系数 & — & {lr_B.coef_[-1]:.4f} & — \\\\\n')
    f.write(r'\midrule' + '\n')
    ratio_str = f'{med_ratio:.1f}\\%'
    f.write(f'中介比例(孕周) & \multicolumn{{3}}{{r}}{{{ratio_str}}} \\\\\n')
    f.write(r'\bottomrule' + '\n')
    f.write(r'\end{tabular}' + '\n')
print(f'  表2已写入')

# ===== 5. Artifact登记 =====
manifest_path = 'E:\\MathModel\\problems\\2025\\C题\\solution\\artifacts\\manifest.md'
with open(manifest_path, 'w', encoding='utf-8') as f:
    f.write('# Artifact 登记清单（问题1）\n\n')
    f.write('## 图表 (Charts)\n\n')
    f.write('| 文件名 | 内容描述 | AI建议位置 | 人类决定 |\n')
    f.write('|--------|---------|-----------|----------|\n')
    f.write('| sub1-correlation-heatmap.pdf | Spearman相关矩阵热力图 | 正文 | |\n')
    f.write('| sub1-residual-diagnostics.pdf | 残差Q-Q图 + 残差vs拟合值 | 正文 | |\n')
    f.write('| sub1-individual-trajectories.pdf | 9例个体Y浓度轨迹 | 正文 | |\n')
    f.write('| sub1-coefficient-forest.pdf | 模型A核心系数估计(含95%CI) | 正文 | |\n')
    f.write('| sub1-mediation-diagram.pdf | 孕周→Y的中介效应路径图 | 正文 | |\n')
    f.write('| sub1-monthly-batch-effect.pdf | 月份批次效应(Y均值+达标率) | 附录 | |\n')
    f.write('| sub1-first-pass-vs-bmi.pdf | 首次达标孕周 vs BMI | 正文 | |\n')
    f.write('\n## 代码片段 (Code Snippets)\n\n')
    f.write('| 文件名 | 内容描述 | 用途 | 人类决定 |\n')
    f.write('|--------|---------|------|----------|\n')
    f.write('| sub1-core-regression.py | 模型A核心回归逻辑(≤15行) | 论文展示核心算法 | |\n')
    f.write('\n## 结果表 (Tables)\n\n')
    f.write('| 文件名 | 内容描述 | AI建议位置 | 人类决定 |\n')
    f.write('|--------|---------|-----------|----------|\n')
    f.write('| sub1-model-coefficients.tex | 模型A系数估计及p值 | 正文 | |\n')
    f.write('| sub1-mediation-table.tex | 中介分析对比表 | 正文 | |\n')
print(f'  manifest已更新')

# ===== 6. 附录代码归档 =====
import shutil; shutil.copy(__file__, os.path.join(code_dir, 'sub1-model.py'))
print(f'\n  附录代码已归档: {code_dir}\\sub1-model.py')

# ===== 7. 核心代码片段 =====
snippet = """
import pandas as pd; from sklearn.linear_model import LinearRegression
data = pd.read_pickle('2025C-sub1-preprocessed.pkl')
month_cols = [c for c in data.columns if c.startswith('m_202')]
X = data[['gw_c','bmi_c','pc1'] + month_cols].values; y = data['y_log'].values
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
R2, RMSE = sklearn.metrics.r2_score(y, y_pred), np.sqrt(np.mean((y-y_pred)**2))
"""
with open(f'{chart_dir}/../code-snippets/sub1-core-regression.py', 'w') as f:
    f.write(snippet)

print('\n' + '='*50)
print('问题1代码实现完成')
print(f'  图表: {fig_dir}/ (7张)')
print(f'  结果表: {tables_dir}/ (2张)')
print(f'  附录代码: {code_dir}/')
print(f'  Artifact清单: {manifest_path}')

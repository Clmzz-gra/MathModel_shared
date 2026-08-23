"""
生成 Q5 α 扫描相关图表：
  - q5-weight-sensitivity.pdf : 正文用 — 修正后加权均值曲线（替换旧图）
  - q5-alpha-by-topic.pdf    : 附录用 — 五题逐题曲线 + 加权均值

依赖 c2_q5_model.py 的 alpha_results（需先跑 c2_q5_model.py）
"""
import sys, os
_SCRATCH_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRATCH_DIR)))))
sys.path.insert(0, _SCRATCH_DIR)
sys.path.insert(0, _PROJECT_ROOT)
from c2_q5_model import alpha_results, alphas, q5df

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chart_utils import setup_mpl, save_figure, resolve_dirs

setup_mpl()
FIG_DIR, CHART_DIR = resolve_dirs(__file__)

TOPICS = ['A', 'B', 'C', 'D', 'E']
TOPIC_COLORS = {
    'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#d62728',
    'D': '#2ca02c', 'E': '#9467bd'
}

# ============================================================
# 图1（正文）：修正后加权均值曲线 + 平坦平台标注
# ============================================================
weighted_rhos = [alpha_results[a]['weighted'] for a in alphas]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(alphas, weighted_rhos, 'ko-', linewidth=2, markersize=8, label='加权均值 ρ')

# 平坦平台高亮
ax.axvspan(0.10, 0.30, alpha=0.08, color='green')
ax.annotate('平坦平台\nρ≈0.883-0.884',
            xy=(0.175, 0.883), fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.6))

# 当前 α=0.25 竖线
ax.axvline(x=0.25, color='#d62728', linestyle='--', alpha=0.5)
ax.annotate(f'当前 α=0.25\nρ=0.883', xy=(0.25, 0.883),
            xytext=(0.32, 0.91), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#d62728'))

# 标注 α=0 基准
ax.annotate('α=0（纯集中评审）\nρ=1.000', xy=(0.0, 1.0),
            xytext=(0.05, 0.96), fontsize=8, ha='left',
            arrowprops=dict(arrowstyle='->', color='gray'),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.6))

ax.set_xlabel('网评权重 α')
ax.set_ylabel('Spearman ρ（题内样本量加权均值）')
ax.set_title('网评权重敏感性分析')
ax.set_ylim(0.82, 1.02)
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)

save_figure(fig, 'q5-weight-sensitivity', fig_dir=FIG_DIR, chart_dir=CHART_DIR,
            context='子问题5：网评权重敏感性曲线（修正后，分题加权均值）', review=True)

# ============================================================
# 图2（附录）：五题逐题曲线 + 加权均值
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

for topic in TOPICS:
    topic_rhos = [alpha_results[a]['by_topic'][topic] for a in alphas]
    ax.plot(alphas, topic_rhos, 'o-', linewidth=1.5, markersize=5,
            color=TOPIC_COLORS[topic], label=f'{topic}题', alpha=0.85)

# 加权均值（粗线强调）
ax.plot(alphas, weighted_rhos, 'ks-', linewidth=2.5, markersize=8,
        label='加权均值', zorder=10)

# 平坦平台
ax.axvspan(0.10, 0.30, alpha=0.06, color='green')

# α=0.25
ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

# 标注 C 题在 α=0.40+ 的偏离
c_rhos = [alpha_results[a]['by_topic']['C'] for a in alphas]
ax.annotate(f'C题 α=0.50: ρ=0.777',
            xy=(0.50, 0.777), xytext=(0.42, 0.755),
            fontsize=8, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728', alpha=0.6))

ax.set_xlabel('网评权重 α')
ax.set_ylabel('Spearman ρ（题内计算）')
ax.set_title('各题网评权重敏感性曲线 (α=0~0.5)')
ax.set_ylim(0.75, 1.02)
ax.legend(loc='lower left', ncol=3, fontsize=8)
ax.grid(True, alpha=0.3)

save_figure(fig, 'q5-alpha-by-topic', fig_dir=FIG_DIR, chart_dir=CHART_DIR,
            context='附录：五题逐题α敏感性曲线 + 加权均值', review=True)

print('Done: q5-weight-sensitivity.pdf + q5-alpha-by-topic.pdf')

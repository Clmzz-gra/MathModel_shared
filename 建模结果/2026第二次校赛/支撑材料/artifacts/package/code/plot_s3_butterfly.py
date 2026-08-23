"""
目的：
    重画 S3 迁移方向一致性图为「蝴蝶图」（diverging bar）——替代原「点+参考线」图，
    用中心零点两侧对称横条直接传达「方向一致 vs 方向翻转 ≈ 50/50（接近随机）」。

原理：
    - 数据只读 S3-results.pkl migration_analysis.{direction_consistent_count=387,
      direction_flipped_count=369, n_valid=756, consistent_fraction=0.5119,
      sign_test_pvalue=0.5364}，禁止写死占位。
    - 蝴蝶图（diverging bar）：横轴 = 标志物数量，中心 0 竖虚线为对称轴；
      左侧横条 = 方向一致（387，Okabe-Ito 蓝 #0072B2），右侧横条 = 方向翻转
      （369，Okabe-Ito 橙 #E69F00）。两条条从中心 0 向左右延伸，几乎等长，
      视觉直接传达「≈50/50 接近随机」；50% 参考由中心线本身表达，无需额外参考线。
    - 条端标注数值：左端「387 一致（51.2%）」、右端「369 翻转（48.8%）」；
      中心 0 加小刻度标签；中部小字标注符号检验 p 值（未显著偏离随机）。
    - 零基线硬约束：两侧条均从中心 0 起（barh 左条用负值、右条用正值，天然零基线）。

性能：
    轻量-不适用（一次性小数据出图，秒级，无并行需求）。

输入数据：
    - S3-results.pkl (结果) — migration_analysis.{direction_consistent_count,
      direction_flipped_count, n_valid, consistent_fraction, sign_test_pvalue}

输出：
    - outputs/figures/S3-migration-direction.pdf（覆盖同名文件，不改名）
    - 副本写入 solution/artifacts/charts/S3-migration-direction.pdf

对应论文章节：
    §6.2 深度迁移分析（S3 内部报告）
"""
import pickle
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 中文字体与负号（Windows 必备）===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Source Han Sans CN']
plt.rcParams['axes.unicode_minus'] = False

# === 字体嵌入 TrueType（修复中文在 PDF 中丢失）===
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# === 自动布局 ===
plt.rcParams['figure.constrained_layout.use'] = True

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "outputs" / "data"
FIG = ROOT / "outputs" / "figures"
CHART = ROOT / "solution" / "artifacts" / "charts"

# Okabe-Ito 色盲安全配色（去饱和）
C_BLUE = "#0072B2"    # 方向一致
C_ORANGE = "#E69F00"  # 方向翻转
C_GRAY = "#999999"
C_BLACK = "#333333"


def load(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


# ============================================================
# 1. 只读提取 pkl 关键字段（禁止写死占位）
# ============================================================
s3 = load("S3-results.pkl")
ma = s3["migration_analysis"]

n_cons = int(ma["direction_consistent_count"])   # 387
n_flip = int(ma["direction_flipped_count"])      # 369
n_total = int(ma["n_valid"])                     # 756
frac = float(ma["consistent_fraction"])         # 0.5119
pval = float(ma["sign_test_pvalue"])             # 0.5364

frac_pct = frac * 100.0
flip_pct = 100.0 - frac_pct

# 核销打印（供人工核对）
print("=" * 60)
print("蝴蝶图数据核销（来自 S3-results.pkl）")
print(f"  方向一致: {n_cons} ({frac_pct:.1f}%)")
print(f"  方向翻转: {n_flip} ({flip_pct:.1f}%)")
print(f"  n_valid:  {n_total}")
print(f"  符号检验 p = {pval:.4f}")
print("=" * 60)

# ============================================================
# 2. 蝴蝶图（diverging bar）
# ============================================================
# 横轴范围：对称，留出条端标注空间
xmax = max(n_cons, n_flip) * 1.28
fig, ax = plt.subplots(figsize=(8.6, 2.6))

# 中心 0 竖虚线（对称轴，同时表达 50% 参考）
ax.axvline(0, color=C_GRAY, lw=1.2, ls="--", zorder=2)

# 两条条从中心 0 向左右延伸（左负右正，天然零基线）
y = 0
ax.barh(y, -n_cons, height=0.55, color=C_BLUE, zorder=3)
ax.barh(y, n_flip, height=0.55, color=C_ORANGE, zorder=3)

# 条端数值标注
ax.text(-n_cons, y, f"{n_cons} 一致（{frac_pct:.1f}%）",
        ha="right", va="center", fontsize=11, color=C_BLUE, fontweight="bold")
ax.text(n_flip, y, f"{n_flip} 翻转（{flip_pct:.1f}%）",
        ha="left", va="center", fontsize=11, color=C_ORANGE, fontweight="bold")

# 中心 0 小刻度标签
ax.text(0, y - 0.62, "0", ha="center", va="top", fontsize=9, color=C_BLACK)

# p 值小字标注（中部）
ax.text(0, y + 0.62, f"符号检验 p={pval:.3f}（未显著偏离随机）",
        ha="center", va="bottom", fontsize=9, color=C_BLACK)

# 坐标轴清理：隐藏 y 轴、上下/右脊线，横轴上下对称
ax.set_xlim(-xmax, xmax)
ax.set_ylim(-1.0, 1.0)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# 横轴刻度：对称（含中心 0）
tick_step = 200
ticks = list(range(-int(xmax // tick_step) * tick_step,
                   int(xmax // tick_step) * tick_step + 1, tick_step))
ax.set_xticks(ticks)
ax.set_xlabel("标志物数量", fontsize=10)

ax.set_title("共享标志物跨疾病方向一致性（387 vs 369，接近随机）", fontsize=11)

fig.savefig(FIG / "S3-migration-direction.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
shutil.copy(FIG / "S3-migration-direction.pdf", CHART / "S3-migration-direction.pdf")
print("S3-migration-direction.pdf (butterfly) done")

"""
目的：
    重画 S3 两张论文正式图（更优雅方案）：
    (1) S3-migration-direction.pdf —— 点+参考线图（替代 100% 堆叠色块），
        单点标在 51.2% 位置 + 50% 灰色虚参考线，传达"方向一致性接近随机"；
    (2) S3-threshold-drift.pdf —— KDE 分布叠加图（替代色块示意），
        训练/测试患病概率分布叠加 + Youden 阈值竖虚线 + 96.0% 判健康阴影。

原理：
    - 图 1 数据只读 S3-results.pkl migration_analysis.{direction_consistent_count=387,
      direction_flipped_count=369, consistent_fraction=0.5119, sign_test_pvalue=0.5364}，
      n_valid=756。点+参考线：单点标 51.2%，50% 处灰虚参考线，p 值小字标注。
    - 图 2 数据：threshold_drift.{train_baseline=0.316, test_baseline=0.648,
      delta_baseline=+0.332, youden_threshold=0.9205, boundary_position=0.9605,
      sensitivity=0.024} + C3 组合训练/测试预测概率分布（S3-results.pkl 未存 oof 概率，
      按 S3 正式口径从 S3-preprocessed.pkl 重算）。
    - C3 组合（LODO 协议）：测试 Obesity（正类占比 64.8%），训练 CRC+IBD（正类占比 31.6%）。
      口径与 S3-model.py 完全一致：近全零过滤 264 特征 → CLR（δ=6.5e-6，逐样本）→
      StandardScaler（仅训练集 fit）→ LogisticRegression(penalty='l2', C=1.0,
      class_weight='balanced', max_iter=2000, random_state=42)，predict_proba[:,1]。
    - 阈值漂移叙事：训练患病基线 0.316 < 测试 0.648（Δ=+0.332，测试分布右移），
      训练集拟合的 Youden 阈值 τ*=0.9205 落在测试分布 96.0% 分位（96.0% 测试样本
      概率 < τ* 被判健康），灵敏度仅 0.024。

性能：
    轻量-不适用（一次性小数据出图 + 单次 Logistic 拟合，秒级，无并行需求）。

输入数据：
    - S3-results.pkl (结果) — migration_analysis.{direction_consistent_count,
      direction_flipped_count, consistent_fraction, sign_test_pvalue} /
      threshold_drift.{train_baseline, test_baseline, delta_baseline,
      youden_threshold, boundary_position, sensitivity}
    - S3-preprocessed.pkl (处理后) — X_filtered(484×264), y(484), lodo_combos.C3.{train_idx,test_idx}

输出：
    - outputs/figures/S3-migration-direction.pdf / S3-threshold-drift.pdf
    - 副本写入 solution/artifacts/charts/

对应论文章节：
    §6.2 深度迁移分析 / §6.3 阈值漂移量化（S3 内部报告）
"""
import pickle
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler

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
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GRAY = "#999999"
C_BLACK = "#333333"
C_RED = "#D62728"

DETECTION_LIMIT = 1e-5
CLR_DELTA = 0.65 * DETECTION_LIMIT  # 6.5e-6
SEED = 42


def clr_transform(X: np.ndarray) -> np.ndarray:
    """CLR 变换（逐样本）：零值乘法替换 δ → log → 逐样本减行均值。与 S3-model.py 一致。"""
    arr = np.asarray(X, dtype=float)
    arr = np.where(arr == 0.0, CLR_DELTA, arr)
    logX = np.log(arr)
    return logX - logX.mean(axis=1, keepdims=True)


def load(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(FIG / name, CHART / name)
    print(f"{name} done")


# ============================================================
# 1. 读取 S3-results.pkl 关键字段
# ============================================================
s3 = load("S3-results.pkl")
ma = s3["migration_analysis"]
td = s3["threshold_drift"]

n_cons = int(ma["direction_consistent_count"])
n_flip = int(ma["direction_flipped_count"])
n_total = int(ma["n_valid"])
frac = float(ma["consistent_fraction"])
pval = float(ma["sign_test_pvalue"])

train_b = float(td["train_baseline"])
test_b = float(td["test_baseline"])
delta = float(td["delta_baseline"])
tau = float(td["youden_threshold"])
bpos = float(td["boundary_position"])
sens = float(td["sensitivity"])

# ============================================================
# 2. 按 S3 正式口径重算 C3 组合训练/测试预测概率分布
# ============================================================
pre = load("S3-preprocessed.pkl")
X_filtered = pre["X_filtered"]
y = np.asarray(pre["y"], dtype=int)
lodo_combos = pre["lodo_combos"]

X_clr = clr_transform(X_filtered.to_numpy())
train_idx = lodo_combos["C3"]["train_idx"]
test_idx = lodo_combos["C3"]["test_idx"]
Xtr = X_clr[train_idx]
Xte = X_clr[test_idx]
ytr = y[train_idx]
yte = y[test_idx]

scaler = StandardScaler().fit(Xtr)
clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
                         class_weight="balanced", random_state=SEED)
clf.fit(scaler.transform(Xtr), ytr)
train_score = clf.predict_proba(scaler.transform(Xtr))[:, 1]
test_score = clf.predict_proba(scaler.transform(Xte))[:, 1]

# 核销：重算值与 pkl 一致
fpr, tpr, thresholds = roc_curve(ytr, train_score)
j = tpr - fpr
thr_recalc = float(thresholds[int(np.argmax(j))])
y_pred = (test_score >= thr_recalc).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, y_pred).ravel()
sens_recalc = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
boundary_recalc = float((test_score < thr_recalc).mean())

print("=" * 60)
print("核销：重算 C3 预测概率 vs pkl 值")
print(f"  train_baseline: 重算 {ytr.mean():.4f} vs pkl {train_b:.4f}")
print(f"  test_baseline:  重算 {yte.mean():.4f} vs pkl {test_b:.4f}")
print(f"  youden_threshold: 重算 {thr_recalc:.4f} vs pkl {tau:.4f}")
print(f"  boundary_position: 重算 {boundary_recalc:.4f} vs pkl {bpos:.4f}")
print(f"  sensitivity: 重算 {sens_recalc:.4f} vs pkl {sens:.4f}")
print(f"  train_score: mean={train_score.mean():.4f} min={train_score.min():.4f} max={train_score.max():.4f}")
print(f"  test_score:  mean={test_score.mean():.4f} min={test_score.min():.4f} max={test_score.max():.4f}")
print("=" * 60)

# ============================================================
# 图 1：迁移方向一致性（点 + 参考线）
# ============================================================
frac_pct = frac * 100.0
fig, ax = plt.subplots(figsize=(8.6, 2.7))
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# 50% 灰色虚参考线
ax.axvline(50, color=C_GRAY, lw=1.2, ls="--", zorder=1)
ax.text(50, 0.88, "随机 50%", color=C_GRAY, ha="center", va="bottom", fontsize=9)

# 蓝色圆点标在 51.2%（大点、白色描边）
ax.scatter(frac_pct, 0.5, s=260, color=C_BLUE, edgecolor="white", linewidth=2.2, zorder=3)
ax.annotate(f"{n_cons}/{n_total} = {frac_pct:.1f}%",
            xy=(frac_pct, 0.5), xytext=(frac_pct + 6, 0.5),
            ha="left", va="center", fontsize=10, color=C_BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C_BLUE, lw=1.0))

# p 值小字标注（右上角）
ax.text(99, 0.88, f"符号检验 p={pval:.3f}（未显著偏离随机）",
        ha="right", va="bottom", fontsize=8.5, color=C_BLACK)

ax.set_xlabel("方向一致占比（%）", fontsize=10)
ax.set_title("共享标志物跨疾病方向一致性（接近随机，p=0.536）", fontsize=11)
save(fig, "S3-migration-direction.pdf")

# ============================================================
# 图 2：C3 阈值漂移（KDE 分布叠加）
# ============================================================
xs = np.linspace(0, 1, 600)
kde_train = gaussian_kde(train_score)
kde_test = gaussian_kde(test_score)
d_train = kde_train(xs)
d_test = kde_test(xs)

# 正类/负类预测概率均值（用于标注"标签语义漂移"）
train_pos_mean = float(train_score[ytr == 1].mean())
test_pos_mean = float(test_score[yte == 1].mean())
print(f"  正类预测概率均值: train={train_pos_mean:.4f} test={test_pos_mean:.4f}")

fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.plot(xs, d_train, color=C_BLUE, lw=2.0, label="训练集（CRC+IBD）", zorder=3)
ax.fill_between(xs, d_train, color=C_BLUE, alpha=0.12, zorder=1)
ax.plot(xs, d_test, color=C_ORANGE, lw=2.0, label="测试集（Obesity）", zorder=3)
ax.fill_between(xs, d_test, color=C_ORANGE, alpha=0.12, zorder=1)

# Youden 阈值竖虚线
ax.axvline(tau, color=C_RED, lw=1.6, ls="--", zorder=2)
ax.text(tau, ax.get_ylim()[1] * 0.98, f"Youden 阈值 τ*={tau:.4f}",
        color=C_RED, ha="right", va="top", fontsize=9)

# 阈值左侧浅色阴影（96.0% 测试样本被判健康）
ax.axvspan(0, tau, color=C_GRAY, alpha=0.07, zorder=0)
ax.text(tau * 0.5, ax.get_ylim()[1] * 0.90,
        f"{bpos*100:.1f}% 测试样本在此侧\n（被判健康）",
        ha="center", va="top", fontsize=9, color=C_BLACK)

# 标注正类预测概率（训练双峰 vs 测试塌缩）
ax.annotate(f"训练正类 ~{train_pos_mean:.2f}", xy=(train_pos_mean, kde_train(train_pos_mean)[0]),
            xytext=(train_pos_mean - 0.16, kde_train(train_pos_mean)[0] * 1.25),
            ha="center", fontsize=9, color=C_BLUE,
            arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.9))
ax.annotate(f"测试正类 ~{test_pos_mean:.2f}", xy=(test_pos_mean, kde_test(test_pos_mean)[0]),
            xytext=(test_pos_mean + 0.14, kde_test(test_pos_mean)[0] * 1.25),
            ha="center", fontsize=9, color=C_ORANGE,
            arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=0.9))

# 图注：标签语义漂移（患病基线 vs 正类预测概率）
ax.text(0.02, 0.03,
        f"标签语义漂移：患病基线 {train_b:.3f}→{test_b:.3f}（Δ={delta:+.3f}），"
        f"但正类预测概率 {train_pos_mean:.2f}→{test_pos_mean:.2f}（塌缩）→ 灵敏度 {sens:.3f}",
        transform=ax.transAxes, fontsize=8.5, color=C_BLACK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="none"))

ax.set_xlim(0, 1)
ax.set_xlabel("预测患病概率", fontsize=10)
ax.set_ylabel("KDE 密度", fontsize=10)
ax.set_title("C3 阈值漂移：测试概率分布左移导致阈值失准", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save(fig, "S3-threshold-drift.pdf")

print("ALL S3 REDRAW FIGURES DONE")

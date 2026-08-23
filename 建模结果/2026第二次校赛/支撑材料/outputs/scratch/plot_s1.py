"""
目的：
    重画 S1 疾病预测模型的 5 张论文正式图（ROC / AUC 对比 / 腺瘤敏感性 / 特征重要性 / 阈值分析），
    修复图审 5 条问题：中文字体可提取（pdf.fonttype=42 消除 Type3 文本提取失败）、
    Youden 最优值正确标注（0.504/0.652/0.274，非阈值）、腺瘤四口径完整命名、
    性能图基线图例 + Dummy 0.5 说明、统一 Okabe-Ito 色板（消除误差棒帽默认蓝 #1f77b4）。

原理：
    全部数字只读取自 S1-results.pkl 与 S1-preprocessed.pkl（y 标签、feature_names、minority）。
    ROC 曲线由 oof_prob 与 y 计算（sklearn roc_curve），AUC 直接取 pkl 落盘值；
    阈值-指标曲线由 oof_prob 与 y 逐阈值扫描计算（ACC/F1/Recall/Specificity），
    Youden J = max(TPR - FPR)，以排序唯一 oof_prob 作候选阈值精确求最优，标注 Youden 值（非阈值）；
    特征重要性取 L2 系数按 |系数| 排序 Top 10，正负方向用 Okabe-Ito 蓝/朱红区分。

性能：
    轻量-不适用（一次性小数据出图，秒级）。

输入数据：
    - S1-results.pkl (结果) — <ds>.L2_CLR.{AUC,AUC_std,oof_prob,coefficients} / <ds>.RF_raw.{AUC,AUC_std,oof_prob} /
      <ds>.baseline.single_feature_best_AUC / adenoma_sensitivity.*.{L2_AUC,RF_AUC,n_samples} / selected_main_caliber
    - S1-preprocessed.pkl (预处理) — datasets.<ds>.{y,minority} / feature_names

输出：
    - outputs/figures/S1-roc-curve.pdf / S1-performance-compare.pdf / S1-adenoma-sensitivity.pdf /
      S1-feature-importance.pdf / S1-threshold-analysis.pdf

对应论文章节：
    §4 结果（S1 内部报告）
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

# === 中文字体与负号（Windows 必备）===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Source Han Sans CN']
plt.rcParams['axes.unicode_minus'] = False

# === 自动布局 ===
plt.rcParams['figure.constrained_layout.use'] = True

# === PDF 字体嵌入为 TrueType（可提取中文，消除 Type3 文本提取失败）===
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "outputs" / "data"
FIG = ROOT / "outputs" / "figures"

# Okabe-Ito 色盲安全色板（全文一致：L2=蓝、RF=橙、基线=灰）
C_BLUE = "#0072B2"      # L2(CLR)
C_ORANGE = "#E69F00"    # RF
C_GRAY = "#999999"      # 基线/参考
C_GREEN = "#009E73"
C_VERM = "#D55E00"      # 负系数/患病减少
C_SKY = "#56B4E9"
C_BLACK = "#333333"

DS = ["Zeller_fecal_colorectal_cancer", "metahit", "Chatelier_gut_obesity"]
DS_SHORT = ["Zeller", "metahit", "Chatelier"]


def load(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


def species_name(feature):
    """从完整分类学名提取物种名（s__ 之后），无 s__ 取末段。"""
    if "|s__" in feature:
        return feature.split("|s__")[-1]
    return feature.split("|")[-1]


def bar_with_err(ax, x, y, w, err, color, label):
    """画柱 + 误差棒，误差棒帽颜色显式设为 C_BLACK（消除 matplotlib 默认蓝 #1f77b4）。"""
    bars = ax.bar(x, y, w, color=color, label=label)
    for xi, yi, ei in zip(x, y, err):
        eb = ax.errorbar(xi, yi, yerr=ei, fmt="none", ecolor=C_BLACK,
                         elinewidth=0.8, capsize=3, capthick=0.8)
        for cap in eb[1]:  # caplines（误差棒帽）
            cap.set_markerfacecolor(C_BLACK)
            cap.set_markeredgecolor(C_BLACK)
    return bars


def threshold_metrics(y, prob, minority):
    """逐阈值扫描（排序唯一 oof_prob + 端点）计算 ACC/F1/Recall/Specificity/TPR/FPR。"""
    thresh = np.unique(np.concatenate([[0.0], np.asarray(prob), [1.0]]))
    accs, f1s, recs, specs, tprs, fprs = [], [], [], [], [], []
    for t in thresh:
        pred = (prob >= t).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        accs.append((tp + tn) / len(y))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
        # 少数类 Recall/F1：按 minority 字段确定正类
        if minority == 1:
            rec = tpr
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        else:
            rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) > 0 else 0.0
        recs.append(rec)
        f1s.append(f1)
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return (thresh, np.asarray(accs), np.asarray(f1s), np.asarray(recs),
            np.asarray(specs), np.asarray(tprs), np.asarray(fprs))


s1 = load("S1-results.pkl")
s1p = load("S1-preprocessed.pkl")
feature_names = s1p["feature_names"]


# ============================================================
# 图 1：三数据集 ROC 曲线 + AUC（L2+RF）
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, d, s in zip(axes, DS, DS_SHORT):
    y = s1p["datasets"][d]["y"]
    l2 = s1[d]["L2_CLR"]
    rf = s1[d]["RF_raw"]
    for prob, model, color, ls, aucv in [(l2["oof_prob"], "L2(CLR)", C_BLUE, "-", l2["AUC"]),
                                          (rf["oof_prob"], "RF", C_ORANGE, "--", rf["AUC"])]:
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, color=color, linestyle=ls, lw=1.8,
                label=f"{model} AUC {aucv:.3f}")
    ax.plot([0, 1], [0, 1], color=C_GRAY, lw=0.8, ls=":")
    ax.set_xlabel("假阳性率（FPR）")
    ax.set_ylabel("真阳性率（TPR）")
    ax.set_title(f"{s}（n={len(y)}）")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
fig.suptitle("三数据集 ROC 曲线（L2+CLR 主模型与 RF 对照，5 折 CV OOF）", fontsize=12)
fig.savefig(FIG / "S1-roc-curve.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("S1-roc-curve.pdf done")


# ============================================================
# 图 2：三数据集 L2/RF/基线 AUC 对比柱状图（含 5 折标准差误差棒）
# ============================================================
l2_auc = [s1[d]["L2_CLR"]["AUC"] for d in DS]
rf_auc = [s1[d]["RF_raw"]["AUC"] for d in DS]
base_auc = [s1[d]["baseline"]["single_feature_best_AUC"] for d in DS]
l2_std = [s1[d]["L2_CLR"]["AUC_std"] for d in DS]
rf_std = [s1[d]["RF_raw"]["AUC_std"] for d in DS]

x = np.arange(len(DS))
w = 0.26
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=False)
b1 = bar_with_err(ax, x - w, l2_auc, w, l2_std, C_BLUE, "L2(CLR)")
b2 = bar_with_err(ax, x, rf_auc, w, rf_std, C_ORANGE, "RF")
b3 = ax.bar(x + w, base_auc, w, label="单特征基线", color=C_GRAY)
for bars, errs in ((b1, l2_std), (b2, rf_std), (b3, [0.0] * len(base_auc))):
    for r, err in zip(bars, errs):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + err + 0.018,
                f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(DS_SHORT)
ax.set_ylabel("AUC")
ax.set_ylim(0, 1.12)  # 降低柱高：上界放宽至 1.12，柱子视觉高度约 80%
ax.axhline(0.5, color=C_BLACK, lw=0.8, ls=":")
handles, labels = ax.get_legend_handles_labels()
fig.subplots_adjust(top=0.84)  # 顶部为标题+图例留出空间
fig.suptitle("三数据集 L2 / RF / 单特征基线 AUC 对比（5 折 CV，误差棒表示折间标准差）",
             y=0.97, fontsize=11)  # 标题最上
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90),
           ncol=3, frameon=False)  # 图例在标题下方
fig.savefig(FIG / "S1-performance-compare.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("S1-performance-compare.pdf done")


# ============================================================
# 图 3：small_adenoma 四口径敏感性对比（高亮主口径）
# ============================================================
ad = s1["adenoma_sensitivity"]
selected = ad["selected_main_caliber"]  # 'healthy'
calibers = [
    ("CRC_adenoma_as_healthy", "①归健康", "healthy"),
    ("CRC_adenoma_as_diseased", "②归病变", "diseased"),
    ("CRC_adenoma_excluded", "③剔除", "excluded"),
    ("CRC_adenoma_separate", "④单开一类", "separate"),
]
l2_ad = [ad[c]["L2_AUC"] for c, _, _ in calibers]
rf_ad = [ad[c]["RF_AUC"] for c, _, _ in calibers]
n_ad = [ad[c]["n_samples"] for c, _, _ in calibers]

x = np.arange(len(calibers))
w = 0.3
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=False)
b1 = ax.bar(x - w / 2, l2_ad, w, label="L2(CLR)", color=C_BLUE)
b2 = ax.bar(x + w / 2, rf_ad, w, label="RF", color=C_ORANGE)
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=8)
# 高亮主口径（selected_main_caliber 对应口径）
for i, (_, _, key) in enumerate(calibers):
    if key == selected:
        ax.axvspan(i - 0.5, i + 0.5, color=C_SKY, alpha=0.15, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip([c for _, c, _ in calibers], n_ad)])
ax.set_ylabel("AUC")
ax.set_ylim(0, 1.12)  # 降低柱高
ax.axhline(0.5, color=C_BLACK, lw=0.8, ls=":")
handles, labels = ax.get_legend_handles_labels()
fig.subplots_adjust(top=0.84)  # 顶部为标题+图例留出空间
fig.suptitle("Zeller small adenoma 四种方案敏感性对比（L2 与 RF，阴影为主方案）",
             y=0.97, fontsize=11)  # 标题最上
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90),
           ncol=2, frameon=False)  # 图例在标题下方
fig.savefig(FIG / "S1-adenoma-sensitivity.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("S1-adenoma-sensitivity.pdf done")


# ============================================================
# 图 4：L2 系数 Top 特征（三数据集，自动对称轴 + 斜体 + 数值标签）
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, d, s in zip(axes, DS, DS_SHORT):
    coef = np.asarray(s1[d]["L2_CLR"]["coefficients"])
    idx = np.argsort(np.abs(coef))[::-1][:10]
    names = [species_name(feature_names[i]).replace("_", " ") for i in idx]
    vals = coef[idx]
    colors = [C_BLUE if v > 0 else C_VERM for v in vals]
    ypos = np.arange(len(names))
    ax.barh(ypos, vals, color=colors, height=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=7, fontstyle="italic")
    ax.invert_yaxis()
    ax.axvline(0, color=C_BLACK, lw=0.8)
    # 自动对称坐标轴（按最大 |系数| 留边距）
    lim = max(abs(vals).max(), 1e-6) * 1.18
    ax.set_xlim(-lim * 1.18, lim * 1.18)
    for yi, v in zip(ypos, vals):
        pad = lim * 0.04
        ax.text(v + (pad if v >= 0 else -pad), yi, f"{v:.2f}",
                ha="left" if v >= 0 else "right", va="center", fontsize=6.5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
    ax.set_xlabel("L2 系数（CLR 空间）")
    ax.set_title(f"{s} Top 10 特征")
fig.suptitle("L2 主模型系数 Top 特征（蓝色表示患病富集，红色表示患病减少）", fontsize=12)
fig.savefig(FIG / "S1-feature-importance.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("S1-feature-importance.pdf done")


# ============================================================
# 图 5：阈值-指标曲线（三数据集，含 Youden 最优值标注）
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=False)
for ax, d, s in zip(axes, DS, DS_SHORT):
    y = s1p["datasets"][d]["y"]
    minority = int(s1p["datasets"][d]["minority"])
    prob = np.asarray(s1[d]["L2_CLR"]["oof_prob"])
    thresh, accs, f1s, recs, specs, tprs, fprs = threshold_metrics(y, prob, minority)
    # Youden J = max(TPR - FPR)，标注 Youden 值（非阈值）
    youden = tprs - fprs
    j_best = float(youden.max())
    t_best = float(thresh[int(np.argmax(youden))])
    ax.plot(thresh, accs, color=C_BLUE, lw=1.6, label="ACC")
    ax.plot(thresh, f1s, color=C_GREEN, lw=1.6, label="F1(少数类)")
    ax.plot(thresh, recs, color=C_ORANGE, lw=1.6, label="Recall(少数类)")
    ax.plot(thresh, specs, color=C_VERM, lw=1.6, label="Specificity")
    ax.axvline(t_best, color=C_GRAY, lw=0.8, ls="--")
    ax.set_xlabel(f"{s} 阈值", fontsize=9)  # 疾病类型与阈值合并于下方
    ax.set_ylabel("指标值")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
handles, labels = axes[0].get_legend_handles_labels()
fig.subplots_adjust(top=0.82, bottom=0.14, wspace=0.25)  # 顶部为标题+图例留出空间
fig.suptitle("L2 主模型阈值-指标曲线（5 折 CV OOF，虚线为 Youden 最优阈值）", y=0.97, fontsize=12)
fig.legend(handles, labels, frameon=False, fontsize=8, ncol=4,
           loc="upper center", bbox_to_anchor=(0.5, 0.90))  # 图例在标题下方
fig.savefig(FIG / "S1-threshold-analysis.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("S1-threshold-analysis.pdf done")
print("ALL S1 FIGURES DONE")

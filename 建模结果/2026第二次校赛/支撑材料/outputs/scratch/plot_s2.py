"""
目的：
    生成 S2 特征选择与生物标志物的 4 张论文正式图（稳定频率/τ 敏感性/共现热图/跨疾病 Jaccard）。

原理：
    全部数字只读取自 S2-results.pkl，禁止写死占位数值。
    - 稳定频率：per_disease.<D>.stable_features.{feature,frequency}，横向条形图，τ=0.5 红色虚线为入选线，
      并叠加 CV 折内频率（诚实口径）灰色菱形标记，直观呈现「全量乐观 vs 折内诚实」的选择方差。
    - τ 敏感性：meta.tau_grid（[0.4,0.5,0.6,0.7]）与 meta.tau_counts（每病入选数），折线图，零基线。
    - 共现热图：per_disease.<D>.cooccurrence.spearman_matrix（特征对字典，None=无法计算）构建对称矩阵，
      None 置 NaN 并以灰色「N/A」标注；正相关蓝、负相关橙（Okabe-Ito 去饱和），范围对称 [-1,1]。
    - 跨疾病 Jaccard：cross_disease.jaccard_matrix（三病两两 0.0），3×3 矩阵，对角线=1（自重叠）。

性能：
    轻量-不适用（一次性小数据出图，秒级）。

输入数据：
    - S2-results.pkl (结果) — per_disease.<D>.stable_features.{feature,frequency,cv_frequency} /
      meta.{tau_grid,tau_counts} / per_disease.<D>.cooccurrence.spearman_matrix /
      cross_disease.jaccard_matrix

输出：
    - outputs/figures/S2-stable-frequency.pdf / S2-tau-sensitivity.pdf / S2-cooccurrence-heatmap.pdf /
      S2-cross-disease.pdf

对应论文章节：
    §4/§6/§7（S2 内部报告）
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# === 中文字体与负号（Windows 必备）===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Source Han Sans CN']
plt.rcParams['axes.unicode_minus'] = False

# === 字体嵌入 TrueType（修复中文在 PDF 中丢失：Type3 仅 256 槽无法编码 CJK）===
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# === 自动布局 ===
plt.rcParams['figure.constrained_layout.use'] = True

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "outputs" / "data"
FIG = ROOT / "outputs" / "figures"

# Okabe-Ito 色盲安全配色（去饱和）
C_BLUE = "#0072B2"    # CRC
C_GREEN = "#009E73"   # IBD
C_VERM = "#D55E00"    # Obesity
C_ORANGE = "#E69F00"
C_GRAY = "#999999"
C_BLACK = "#333333"
C_RED = "#D62728"

DISEASES = ["CRC", "IBD", "Obesity"]
DISEASE_CN = {"CRC": "结直肠癌 CRC", "IBD": "炎症性肠病 IBD", "Obesity": "肥胖症 Obesity"}
DISEASE_COLOR = {"CRC": C_BLUE, "IBD": C_GREEN, "Obesity": C_VERM}

# 正负相关对称发散色板（Okabe-Ito 蓝/橙，白=0）
CMAP_DIVERGING = LinearSegmentedColormap.from_list("okabe_div", [C_VERM, "#FFFFFF", C_BLUE])


def load(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


def species_name(feature):
    if "|s__" in feature:
        return feature.split("|s__")[-1]
    return feature.split("|")[-1]


s2 = load("S2-results.pkl")


# ============================================================
# 图 1：三病稳定标志物频率直方图（τ=0.5 红线）
# ============================================================
# 布局：上排 CRC/IBD（各 4 个），下排 Obesity（20 个，跨两列，更高）
fig = plt.figure(figsize=(12, 11.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 3.0])
ax_crc = fig.add_subplot(gs[0, 0])
ax_ibd = fig.add_subplot(gs[0, 1])
ax_ob = fig.add_subplot(gs[1, :])

for ax, d in [(ax_crc, "CRC"), (ax_ibd, "IBD"), (ax_ob, "Obesity")]:
    sf = s2["per_disease"][d]["stable_features"]
    names = [species_name(f["feature"]).replace("_", " ") for f in sf]
    freqs = [f["frequency"] for f in sf]
    cv_freqs = [f["cv_frequency"] for f in sf]
    ypos = np.arange(len(names))
    ax.barh(ypos, freqs, color=DISEASE_COLOR[d], height=0.62, zorder=3)
    # CV 折内频率（诚实口径）灰色菱形标记
    ax.scatter(cv_freqs, ypos, marker="D", s=28, color=C_GRAY, zorder=4,
               label="CV 折内频率（诚实）")
    # 数值标签（全量频率）
    for y, v in zip(ypos, freqs):
        ax.text(v + 0.015, y, f"{v:.2f}", va="center", ha="left", fontsize=7.5, color=C_BLACK)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.5, color=C_RED, lw=1.3, ls="--", zorder=2)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("全量 bootstrap 频率 $\\hat{\\pi}_j$", fontsize=9)
    ax.set_title(f"{DISEASE_CN[d]}（{len(names)} 个稳定标志物）", fontsize=10)
    ax.grid(axis="x", color="#dddddd", lw=0.3, zorder=0)
    ax.set_axisbelow(True)

# 图例（放 CRC 面板）
ax_crc.legend(loc="lower right", frameon=False, fontsize=7.5)
fig.suptitle("三病稳定标志物全量选择频率（τ=0.5 入选线，灰菱形=CV 折内诚实频率）", fontsize=12)
fig.savefig(FIG / "S2-stable-frequency.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("S2-stable-frequency.pdf done")


# ============================================================
# 图 2：τ 敏感性曲线（不同线型 + 图形节点，一到两种颜色，上方图例）
# ============================================================
tau_grid = s2["meta"]["tau_grid"]
tau_counts = s2["meta"]["tau_counts"]
fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=False)
# 三种疾病：同一主色（C_BLUE）但不同线型 + 不同图形节点区分
line_styles = ["-", "--", "-."]
markers = ["o", "s", "^"]
for i, d in enumerate(DISEASES):
    ax.plot(tau_grid, tau_counts[d], ls=line_styles[i], marker=markers[i],
            lw=1.8, color=C_BLUE, label=DISEASE_CN[d], zorder=3)
ax.axvline(0.5, color=C_RED, lw=1.3, ls="--", zorder=2, label="τ=0.5 入选阈值")
ax.set_xlabel("入选频率阈值 τ")
ax.set_ylabel("入选稳定特征数")
ax.set_xticks(tau_grid)
ax.set_ylim(0, max(max(v) for v in tau_counts.values()) * 1.12)  # 零基线
ax.grid(color="#dddddd", lw=0.3, zorder=0)
ax.set_axisbelow(True)
handles, labels = ax.get_legend_handles_labels()
fig.subplots_adjust(top=0.84)  # 顶部为标题+图例留出空间
fig.suptitle("τ 敏感性：入选稳定特征数随阈值变化", y=0.97, fontsize=11)  # 标题最上
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90),
           ncol=4, frameon=False)  # 图例在标题下方
fig.savefig(FIG / "S2-tau-sensitivity.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("S2-tau-sensitivity.pdf done")


# ============================================================
# 图 3：共现 Spearman 相关热图（CRC / IBD）
# ============================================================
def build_matrix(spearman_dict):
    """由特征对字典构建对称相关矩阵；None 值置 NaN（无法计算）。"""
    feats = []
    for (a, b) in spearman_dict.keys():
        for f in (a, b):
            if f not in feats:
                feats.append(f)
    n = len(feats)
    M = np.full((n, n), np.nan)
    np.fill_diagonal(M, 1.0)
    for (a, b), v in spearman_dict.items():
        i, j = feats.index(a), feats.index(b)
        if v is not None:
            M[i, j] = v
            M[j, i] = v
    return feats, M


fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, d in zip(axes, ["CRC", "IBD"]):
    sm = s2["per_disease"][d]["cooccurrence"]["spearman_matrix"]
    feats, M = build_matrix(sm)
    labels = [species_name(f).replace("_", " ") for f in feats]
    # 对称矩阵取下三角：上三角（j > i）置 NaN 不显示
    tri = np.triu(np.ones(M.shape), k=1).astype(bool)
    Mt = M.copy()
    Mt[tri] = np.nan
    masked = np.ma.masked_invalid(Mt)
    # interpolation="nearest" 避免 NaN 边界 antialiasing 产生黑色轮廓线
    im = ax.imshow(masked, cmap=CMAP_DIVERGING, vmin=-1, vmax=1,
                   interpolation="nearest")
    im.cmap.set_bad("#ffffff")  # NaN/遮罩区域纯白，不留黑边
    # 下三角内真实 N/A（无法计算）以浅灰底显示；上三角留白不显示灰框
    real_na = np.isnan(M) & ~tri
    ax.imshow(real_na, cmap=LinearSegmentedColormap.from_list("na", ["none", "#e8e8e8"]),
              vmin=0, vmax=1, zorder=1, interpolation="nearest")
    for spine in ax.spines.values():
        spine.set_visible(False)  # 去除热图黑色外边框
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if j > i:
                continue  # 上三角不标注
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=6.5, color=C_GRAY)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.5 else C_BLACK)
    ax.set_title(f"{DISEASE_CN[d]} 入选标志物 Spearman 相关", fontsize=10)
fig.colorbar(im, ax=axes, shrink=0.85, label="Spearman ρ")
fig.suptitle("入选标志物共现 Spearman 相关热图（非零样本、CLR 后丰度；下三角，N/A=无法计算）", fontsize=12)
fig.savefig(FIG / "S2-cooccurrence-heatmap.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("S2-cooccurrence-heatmap.pdf done")


# ============================================================
# 图 4：三病 Jaccard 重叠矩阵
# ============================================================
jm = s2["cross_disease"]["jaccard_matrix"]
J = np.eye(3)
J[0, 1] = J[1, 0] = jm["CRC_IBD"]
J[0, 2] = J[2, 0] = jm["CRC_Obesity"]
J[1, 2] = J[2, 1] = jm["IBD_Obesity"]
labels = ["CRC", "IBD", "Obesity"]
fig, ax = plt.subplots(figsize=(5.5, 5.0))
# 对称矩阵取下三角：上三角（j > i）置 NaN 不显示
tri = np.triu(np.ones((3, 3)), k=1).astype(bool)
Jt = J.copy()
Jt[tri] = np.nan
# interpolation="nearest" 避免 NaN 边界 antialiasing 产生黑色轮廓线
im = ax.imshow(np.ma.masked_invalid(Jt), cmap="Blues", vmin=0, vmax=1,
               interpolation="nearest")
im.cmap.set_bad("#ffffff")  # NaN/遮罩区域纯白
for s in ax.spines.values():
    s.set_visible(False)  # 去除热图黑色外边框
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
for i in range(3):
    for j in range(3):
        if j > i:
            continue  # 上三角不标注
        ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", fontsize=11,
                color="white" if J[i, j] > 0.5 else C_BLACK)
ax.set_title("三病稳定标志物 Jaccard 重叠矩阵（下三角）")
fig.colorbar(im, ax=ax, shrink=0.8, label="Jaccard 系数")
ax.set_xlabel("两两 Jaccard 均为 0；对角线 1 表示自重叠", fontsize=8.5, labelpad=10)
fig.savefig(FIG / "S2-cross-disease.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("S2-cross-disease.pdf done")
print("ALL S2 FIGURES DONE")

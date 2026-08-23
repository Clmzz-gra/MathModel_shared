"""
目的：
    绘制 3 张数据特征图（共享数据层，不进子问题代号）：
      图1 chart-sample-composition.pdf  三数据集样本构成（堆叠条形图）
      图2 chart-zero-sparsity.pdf      特征零值占比分布（直方图）
      图3 chart-abundance-distribution.pdf  非零丰度分布（对数直方图）
    支撑：类别不平衡→主指标 AUC + 少数类 F1/Recall；近全零过滤 1331→264；CLR 对数比变换动机。

原理：
    图1：dataset_name × disease 计数，按疾病状态分层（患病/健康/腺瘤）堆叠条形图。
         CRC=Zeller（患病 cancer48 / 腺瘤 small_adenoma26 / 健康 n47），
         IBD=metahit（患病 ibd_crohn4+ibd_ulcerative21=25 / 健康 n85），
         Obesity=Chatelier（患病 obesity164 / 健康 leaness89）。
         各数据集患病占比标注于柱顶。
    图2：1331 特征各自的零值占比直方图（分箱 0.05），零值占比>0.95 的 1067 个特征红色高亮
         +0.95 阈值虚线，标注「剔除 1067 / 保留 264」。
    图3：非零值取 log10 后直方图，标注中位数 log10(0.0776)≈-1.11 与范围（1e-5~79.96，跨7数量级）。

性能：
    轻量-不适用（单次读取小数据 pkl 484×1333，毫秒级绘图，无并行必要）。

输入数据：
    - outputs/data/c-data-cleaned.pkl (共享清洗后) — dataset_name, disease, 1331 物种特征列

输出：
    - outputs/figures/chart-sample-composition.pdf
    - outputs/figures/chart-zero-sparsity.pdf
    - outputs/figures/chart-abundance-distribution.pdf

对应论文章节：
    §2 数据概览（共享数据特征图，不进子任务代号）
"""
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 强制前置：中文字体 / 负号 / 矢量字体 / 自动布局（import 后立即）===
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Source Han Sans CN"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.constrained_layout.use"] = True

ROOT = Path(__file__).resolve().parents[2]  # worktree 根
FIG = ROOT / "outputs" / "figures"

# === Okabe-Ito 色盲安全色板（色盲安全 / 非彩虹 / 禁3D）===
C_DISEASE = "#D55E00"  # vermillion 患病
C_HEALTH = "#009E73"   # bluish green 健康
C_ADENOMA = "#0072B2"  # blue 腺瘤
C_GREY = "#333333"

# ============ 第一阶段：纯计算 ============
df = pd.read_pickle(ROOT / "outputs" / "data" / "c-data-cleaned.pkl")
meta = ["dataset_name", "disease"]
feat_cols = [c for c in df.columns if c not in meta]

# --- 图1 数据：dataset_name × disease 计数，映射为状态分层 ---
# 状态映射（按疾病状态分色：患病/健康/腺瘤）
state_map = {
    "cancer": "患病",
    "small_adenoma": "腺瘤",
    "ibd_crohn_disease": "患病",
    "ibd_ulcerative_colitis": "患病",
    "obesity": "患病",
    "n": "健康",
    "leaness": "健康",
}
df["_state"] = df["disease"].map(state_map)
state_ct = df.groupby(["dataset_name", "_state"]).size()

# 定义三个数据集的中文名与展示顺序
ds_order = [
    ("Zeller_fecal_colorectal_cancer", "CRC"),
    ("metahit", "IBD"),
    ("Chatelier_gut_obesity", "Obesity"),
]
layer_labels = ["患病", "健康", "腺瘤"]

# 每数据集各层样本数（缺失层记 0）
bar_data = {}      # ds_disp -> {layer: count}
for raw, disp in ds_order:
    sub = state_ct.get(raw, pd.Series(dtype=int))
    bar_data[disp] = {l: int(sub.get(l, 0)) for l in layer_labels}
total_per_ds = {disp: sum(bar_data[disp].values()) for _, disp in ds_order}
disease_per_ds = {disp: bar_data[disp]["患病"] for _, disp in ds_order}
adenoma_per_ds = {disp: bar_data[disp]["腺瘤"] for _, disp in ds_order}

print("=== 图1 各数据集样本构成 ===")
for _, disp in ds_order:
    print(f"{disp}: 总={total_per_ds[disp]}, 患病={disease_per_ds[disp]}, 健康={bar_data[disp]['健康']}, "
          f"腺瘤={adenoma_per_ds[disp]}, 患病率={disease_per_ds[disp]/total_per_ds[disp]:.3f}")

# --- 图2 数据：各特征零值占比 ---
mat = df[feat_cols].to_numpy(dtype=float)
zero_per_feat = np.mean(mat == 0, axis=0)
overall_zero_ratio = np.mean(mat == 0)
n_gt95 = int(np.sum(zero_per_feat > 0.95))
n_keep = int(np.sum(zero_per_feat <= 0.95))

# --- 图3 数据：非零值 ---
nz = mat[mat != 0]
log_nz = np.log10(nz)
med_log = np.log10(np.median(nz))

print("=== 图2 零值占比 ===")
print(f"全矩阵零值占比={overall_zero_ratio:.4f}, 特征>95%零值={n_gt95}, 保留(<=95%)={n_keep}")
print("=== 图3 非零值 ===")
print(f"非零值min={nz.min():.2e}, median={np.median(nz):.4f}(log10={med_log:.2f}), max={nz.max():.4f}, "
      f"log10范围[{log_nz.min():.2f},{log_nz.max():.2f}], 非零值个数={nz.size}")

# ============ 第二阶段：绘图 ============
# ---------- 图1：堆叠条形图 ----------
fig, ax = plt.subplots(figsize=(6.5, 5))
xpos = np.arange(len(ds_order))
layer_colors = {"患病": C_DISEASE, "健康": C_HEALTH, "腺瘤": C_ADENOMA}
bottoms = np.zeros(len(ds_order))
x = xpos
for layer in layer_labels:
    vals = np.array([bar_data[disp][layer] for _, disp in ds_order])
    ax.bar(x, vals, bottom=bottoms, color=layer_colors[layer], edgecolor="white",
           linewidth=0.6, width=0.6, label=layer)
    # 段内标注样本数
    for xi, v, b in zip(x, vals, bottoms):
        if v > 0:
            ax.text(xi, b + v / 2, str(int(v)), ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
    bottoms += vals
# 柱顶标注患病率
for xi, (_, disp) in enumerate(ds_order):
    total = total_per_ds[disp]
    if total > 0:
        ratio = disease_per_ds[disp] / total
        # IBD 图例标注患病率（含 ibd_crohn+ibd_ulcerative）
        sub_ann = "metahit" if disp == "IBD" else ("CRC 患病率" if disp == "CRC" else "Obesity 患病率")
        ax.text(xi, bottoms[xi] + max(bottoms) * 0.02, f"{sub_ann} {ratio:.1%}",
                ha="center", va="bottom", fontsize=9, color=C_GREY)
ax.set_xticks(x)
ax.set_xticklabels([d for _, d in ds_order], fontsize=11)
ax.set_ylabel("样本数", fontsize=11)
ax.set_ylim(0, max(bottoms) * 1.18)
ax.set_yticks(range(0, int(max(bottoms) * 1.1) + 1, 50))
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.3)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("三数据集样本构成与类别分布", fontsize=13)
ax.set_axisbelow(True)
fig.savefig(FIG / "chart-sample-composition.pdf", dpi=300)
plt.close(fig)

# ---------- 图2：特征零值占比直方图（>0.95 红色高亮） ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.arange(0, 1.001, 0.05)
counts, edges = np.histogram(zero_per_feat, bins=bins)
colors = [C_DISEASE if e >= 0.95 else "#4C72B0" for e in edges[:-1]]
bars = ax.bar(edges[:-1], counts, width=0.05, color=colors, edgecolor="white", linewidth=0.5,
              align="edge", label="特征数")
# 关键柱（>0.95 剔除区）柱顶标数字
for b, c, e in zip(bars, counts, edges[:-1]):
    if e >= 0.95 and c > 0:
        ax.text(b.get_x() + b.get_width() / 2, c + 8, f"{c}", ha="center", fontsize=8)
ax.axvline(0.95, color="#111111", linestyle="--", linewidth=1.2)
ax.set_xlabel("特征零值占比", fontsize=11)
ax.set_ylabel("特征数", fontsize=11)
ax.set_title("物种特征零值稀疏性（近全零过滤动机）", fontsize=12)
ax.set_xticks(np.arange(0, 1.01, 0.1))
ax.set_xlim(0, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.3)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
fig.savefig(FIG / "chart-zero-sparsity.pdf", dpi=300)
plt.close(fig)

# ---------- 图3：非零丰度对数直方图 ----------
fig, ax = plt.subplots(figsize=(7, 5))
n_bins = 50
counts, edges = np.histogram(log_nz, bins=n_bins)
ax.bar(edges[:-1], counts, width=(edges[1]-edges[0]), color="#4C72B0",
       edgecolor="white", linewidth=0.4)
# 中位数标注（虚线入图例）
ax.axvline(med_log, color=C_DISEASE, linestyle="--", linewidth=1.4,
           label=f"中位数 {np.median(nz):.4f}")
ax.set_xlabel("log10 丰度", fontsize=11)
ax.set_ylabel("非零值计数", fontsize=11)
ax.set_title("非零丰度分布（对数坐标，CLR 动机）", fontsize=12)
ax.set_ylim(0, max(counts) * 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.3)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=10)
fig.savefig(FIG / "chart-abundance-distribution.pdf", dpi=300)
plt.close(fig)

print("\n=== 出图完成 ===")
for f in ["chart-sample-composition.pdf", "chart-zero-sparsity.pdf", "chart-abundance-distribution.pdf"]:
    print((FIG / f).resolve())

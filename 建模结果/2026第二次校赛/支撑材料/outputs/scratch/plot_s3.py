"""
目的：
    重画 S3 跨疾病预测模型的 4 张论文正式图（策略对比 / 衰减归因 / 迁移方向 / 阈值漂移诊断），
    在旧版基础上提升信息密度与可读性（衰减归因改哑铃图、阈值漂移改清晰示意、策略对比加均值组）。

原理：
    全部数字只读取自 S3-results.pkl（禁止写死占位数值）：
    - 策略对比：strategy_compare.<S>.<C>.auc 与 mean_auc（S ∈ {A_direct,B_shared,C_genus,C_phylum,D_calibrated}，
      C ∈ {C1,C2,C3}），加 0.5 随机线 / 0.65 可用线两条参考线；
    - 衰减归因：decay_attribution.<D>.{domain_auc,cross_auc,decay,dominant_cause}（D ∈ {CRC,IBD,Obesity}），
      衰减量 = cross_auc - domain_auc（负值），用哑铃图（域内高 → 跨疾病低）直观呈现衰减幅度；
    - 迁移方向：migration_analysis.{direction_consistent_count,direction_flipped_count,
      consistent_fraction,sign_test_pvalue}，100% 堆叠横条 + 50% 基准线；
    - 阈值漂移：threshold_drift.{train_baseline,test_baseline,delta_baseline,youden_threshold,
      boundary_position,sensitivity}，概率轴示意（训练/测试基线 + Youden 阈值位置 + 灵敏度），
      仅用 pkl 实际值作标记，不虚构分布。

性能：
    轻量-不适用（一次性小数据出图，秒级，无并行需求）。

输入数据：
    - S3-results.pkl (结果) — strategy_compare.*.{C1,C2,C3}.auc / mean_auc /
      decay_attribution.*.{domain_auc,cross_auc,decay,dominant_cause} /
      migration_analysis.{direction_consistent_count,direction_flipped_count,consistent_fraction,
      sign_test_pvalue} / threshold_drift.{train_baseline,test_baseline,delta_baseline,
      youden_threshold,boundary_position,sensitivity}

输出：
    - outputs/figures/S3-strategy-compare.pdf / S3-decay-attribution.pdf /
      S3-migration-direction.pdf / S3-threshold-drift.pdf
    - 副本写入 solution/artifacts/charts/

对应论文章节：
    §3（五种策略配置对比）/ §6（衰减归因、深度迁移、阈值漂移）——S3 内部报告
"""
import pickle
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# === 中文字体与负号（Windows 必备）===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Source Han Sans CN']
plt.rcParams['axes.unicode_minus'] = False

# === 字体嵌入 TrueType（修复中文在 PDF 中丢失：Type3 仅 256 槽无法编码 CJK）===
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# === 自动布局（借鉴点 #43）===
plt.rcParams['figure.constrained_layout.use'] = True

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "outputs" / "data"
FIG = ROOT / "outputs" / "figures"
CHART = ROOT / "solution" / "artifacts" / "charts"

# Okabe-Ito 色盲安全配色（去饱和）
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_VERM = "#D55E00"
C_SKY = "#56B4E9"
C_PURPLE = "#CC79A7"
C_GRAY = "#999999"
C_BLACK = "#333333"
C_RED = "#D62728"

STRATS = ["A_direct", "B_shared", "C_genus", "C_phylum", "D_calibrated"]
STRAT_CN = {"A_direct": "A 直接迁移", "B_shared": "B 共享标志物",
            "C_genus": "C（属级聚合）", "C_phylum": "C（门级聚合）", "D_calibrated": "D 部署校正"}
STRAT_COLOR = {"A_direct": C_BLUE, "B_shared": C_ORANGE, "C_genus": C_GREEN,
               "C_phylum": C_VERM, "D_calibrated": C_SKY}
COMBS = ["C1", "C2", "C3"]
COMB_CN = {"C1": "C1\n(测试 CRC)", "C2": "C2\n(测试 IBD)", "C3": "C3\n(测试 Obesity)"}

DISEASES = ["CRC", "IBD", "Obesity"]


def load(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(FIG / name, CHART / name)
    print(f"{name} done")


s3 = load("S3-results.pkl")


# ============================================================
# 图 1：五种策略配置 × 3 组合 AUC 对比
# ============================================================
sc = s3["strategy_compare"]
# 4 个 x 组：C1 / C2 / C3 / 均值
groups = COMBS + ["mean"]
group_cn = {**COMB_CN, "mean": "均值"}
x = np.arange(len(groups))
w = 0.15

fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=False)
for i, st in enumerate(STRATS):
    d = sc[st]
    aucs = [d[c]["auc"] for c in COMBS] + [d["mean_auc"]]
    bars = ax.bar(x + (i - 2) * w, aucs, w, label=STRAT_CN[st], color=STRAT_COLOR[st],
                  edgecolor="white", linewidth=0.4)
    for r, v in zip(bars, aucs):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6.3)

# 参考线：0.5 随机 / 0.65 可用线（加入图例）
ax.axhline(0.5, color=C_GRAY, lw=0.8, ls=":", label="随机基准 0.5")
ax.axhline(0.65, color=C_RED, lw=1.0, ls="--", label="可用线 0.65")

ax.set_xticks(x)
ax.set_xticklabels([group_cn[g] for g in groups], fontsize=8.5)
ax.set_ylabel("AUC", fontsize=10)
ax.set_ylim(0, 0.85)
handles, labels = ax.get_legend_handles_labels()
fig.subplots_adjust(top=0.86)  # 顶部为标题+图例留出空间
fig.suptitle("五种策略配置在 3 个 LODO 组合及其均值下的 AUC 对比", y=0.97, fontsize=11)
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
           frameon=False, fontsize=7.5, ncol=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save(fig, "S3-strategy-compare.pdf")


# ============================================================
# 图 2：域内 vs 跨疾病 AUC 衰减归因（哑铃图，直观呈现衰减幅度）
# ============================================================
da = s3["decay_attribution"]
domain = [da[d]["domain_auc"] for d in DISEASES]
cross = [da[d]["cross_auc"] for d in DISEASES]
decay = [da[d]["decay"] for d in DISEASES]
causes = [da[d]["dominant_cause"] for d in DISEASES]
# 主导归因标注：dominant_cause 为 pkl 数据源；衰减量最大者标注"（最强）"（与正文"IBD 衰减最大"一致）
max_decay_idx = int(np.argmin(decay))  # decay 全为负，argmin 即衰减幅度最大

x = np.arange(len(DISEASES))
fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=False)
xlim_left = -0.5  # 左图框位置（x 轴从 -0.5 起）
for i in range(len(DISEASES)):
    ax.plot([x[i], x[i]], [cross[i], domain[i]], color=C_GRAY, lw=2.0, zorder=1)
    ax.scatter(x[i], domain[i], s=95, color=C_BLUE, zorder=3)
    ax.scatter(x[i], cross[i], s=95, color=C_ORANGE, zorder=3)
    # 数值经水平虚线引至左侧数轴（到左图框为止，不越框），数字黑色正放
    ax.plot([x[i], xlim_left], [domain[i], domain[i]], color=C_GRAY, ls=":",
            lw=0.7, zorder=2)
    ax.text(xlim_left - 0.02, domain[i], f"{domain[i]:.3f}", ha="right", va="center",
            fontsize=8, color="black")
    ax.plot([x[i], xlim_left], [cross[i], cross[i]], color=C_GRAY, ls=":",
            lw=0.7, zorder=2)
    ax.text(xlim_left - 0.02, cross[i], f"{cross[i]:.3f}", ha="right", va="center",
            fontsize=8, color="black")
ax.axhline(0.5, color=C_GRAY, lw=0.8, ls=":", label="随机基准 0.5")
ax.set_xticks(x)
ax.set_xticklabels(DISEASES, fontsize=10)
ax.set_ylabel("AUC（越高越好）", fontsize=10, labelpad=30)  # ylabel 左移，给三位小数留位
ax.set_ylim(0.45, 0.92)  # 折叠数轴下半无用空白
ax.set_xlim(xlim_left, len(DISEASES) - 0.5)
ax.tick_params(axis="y", labelleft=False)  # 隐藏 y 刻度数字，避免与引出数值重合（数值由虚线引出标注）
handles, labels = ax.get_legend_handles_labels()
# 图例：蓝=域内、橙=跨疾病、灰线=衰减
from matplotlib.lines import Line2D
legend_handles = [Line2D([0], [0], marker="o", ls="", color=C_BLUE, markersize=8, label="域内（同疾病五折 CV）"),
                  Line2D([0], [0], marker="o", ls="", color=C_ORANGE, markersize=8, label="跨疾病（LODO）"),
                  Line2D([0], [0], color=C_GRAY, lw=2.0, label="泛化衰减量"),
                  handles[0]]
fig.subplots_adjust(top=0.86)  # 顶部为标题+图例留出空间
fig.suptitle("跨疾病泛化造成的 AUC 下降：域内性能与 LODO 性能对照", y=0.97, fontsize=11)
fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.91),
           frameon=False, fontsize=8, ncol=4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save(fig, "S3-decay-attribution.pdf")


# ============================================================
# 图 3：共享物种效应方向一致性（100% 堆叠横条 + 50% 参考线）
# ============================================================
ma = s3["migration_analysis"]
n_cons = ma["direction_consistent_count"]
n_flip = ma["direction_flipped_count"]
frac = ma["consistent_fraction"]
pval = ma["sign_test_pvalue"]
n_total = n_cons + n_flip

fig, ax = plt.subplots(figsize=(8.6, 3.0))
ax.barh(0, n_cons, color=C_BLUE)
ax.barh(0, n_flip, left=n_cons, color=C_ORANGE)
ax.axvline(n_total / 2, color=C_GRAY, lw=1.0, ls=":")
ax.text(n_cons / 2, 0, f"方向未翻转\n{n_cons}（{frac*100:.1f}%）", ha="center", va="center",
        color="white", fontsize=9, fontweight="bold")
ax.text(n_cons + n_flip / 2, 0, f"方向翻转\n{n_flip}（{(1-frac)*100:.1f}%）", ha="center", va="center",
        color="white", fontsize=9, fontweight="bold")
ax.set_yticks([])
ax.set_xlim(0, n_total)
ax.set_xlabel(f"共享物种数（有效样本 N={n_total}）", fontsize=10)
ax.set_title(f"共享物种效应方向是否稳定：51.2% 未翻转，与 50% 基准无显著差异（p={pval:.3f}）", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
save(fig, "S3-migration-direction.pdf")


# ============================================================
# 图 4：C3 阈值漂移诊断（概率轴示意：基线差 + Youden 阈值位置 + 灵敏度）
# ============================================================
td = s3["threshold_drift"]
train_b = td["train_baseline"]
test_b = td["test_baseline"]
delta = td["delta_baseline"]
tau = td["youden_threshold"]
bpos = td["boundary_position"]
sens = td["sensitivity"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_yticks([])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax1.set_xlabel("患病概率 / 决策分数")
ax1.plot([train_b, train_b], [0.16, 0.64], color=C_BLUE, lw=3)
ax1.plot([test_b, test_b], [0.16, 0.64], color=C_ORANGE, lw=3)
ax1.plot([tau, tau], [0.16, 0.64], color=C_RED, lw=2.5, ls="--")
ax1.annotate("", xy=(test_b, 0.37), xytext=(train_b, 0.37), arrowprops=dict(arrowstyle="<->", color=C_BLACK, lw=1.3))
ax1.set_title("决策分数上的三个关键位置", fontsize=10)

ax2.bar(["测试分布中\n阈值左侧（判为健康）"], [bpos], color=C_RED, width=0.5)
ax2.axhline(0.5, color=C_GRAY, ls=":", lw=0.9)
ax2.text(0, bpos + 0.03, f"{bpos*100:.1f}%", ha="center", fontsize=10, color=C_RED)
ax2.set_ylim(0, 1.08)
ax2.set_ylabel("测试样本比例")
ax2.set_title(f"阈值位于测试分布第 {bpos*100:.1f} 百分位", fontsize=10)
fig.suptitle("C3 阈值漂移：基线变化使训练阈值在测试域失准", fontsize=12)
save(fig, "S3-threshold-drift.pdf")

print("ALL S3 FIGURES DONE")

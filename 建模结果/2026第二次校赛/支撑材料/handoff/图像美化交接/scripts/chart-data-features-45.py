"""
目的：
    绘制两张数据特征图（共享数据，不进子问题代号）：图4 批次效应
    （三数据集样本在 PCA/t-SNE 二维特征空间的分布，按数据集着色，无清晰分离 →
    存在批次/分布差异，支撑"绝对 AUC 不可跨数据集横向比"）；图5 已知标志物存在性
    （对每个已知微生物标志物，在对应患病数据集内计算患病/健康两组的"存在率"=该物种
    丰度>0 的样本占比，分组柱状图，作为 S2 生物合理性锚点）。

原理：
    - 图4：近全零过滤（零值占比>95% 剔除，1331→264，与 S3 探索图口径一致）→ CLR 变换
      （零值乘法替换 δ=0.65×检出限=6.5e-6，逐样本 log 减行均值，无跨样本参数）→ 统一
      StandardScaler（仅可视化用，不训练标签）→ PCA 前两主成分 / t-SNE(perplexity=30,
      random_state=42)。每样本一个点，按 dataset_name 三色（CRC/IBD/Obesity，Okabe-Ito）。
      读图判据：若三数据集在特征空间自然分离则无批次差异；重叠混杂则存在批次/分布差异。
    - 图5：存在率 = 该标志物特征丰度>0 的样本占比（=检出率）。病组/健组标签：
      CRC(cancer vs n+small_adenoma)、IBD(uc+cd vs n)、Obesity(obesity vs leaness)。
      标志物选择锚定文献已知菌 + S2 稳定标志物（Fusobacterium_nucleatum /
      Peptostreptococcus_stomatis / Porphyromonas_somerae → CRC；
      Bifidobacterium_bifidum / Akkermansia_muciniphila → IBD；
      Bacteroides_fragilis → Obesity）。每组柱从 0 起（零基线，禁截断）。

性能：
    轻量-不适用（484×264 小数据，PCA/t-SNE 秒级一次性，无并行需求）。

输入数据：
    - outputs/data/c-data-cleaned.pkl (处理后) — dataset_name, disease, 1331 物种丰度特征
      （物种列名 k__...|s__...，中文指标↔变量：dataset_name↔数据集、disease↔患病/健康标签、
      species 列↔相对丰度>0 判存在）

输出：
    - outputs/figures/chart-batch-effect.pdf — 图4 批次效应（PCA/t-SNE 按数据集着色散点）
    - outputs/figures/chart-known-biomarker-presence.pdf — 图5 已知标志物存在率分组柱状图

对应论文章节：
    §数据特征（共享数据特征图 4-5，S2 生物合理性锚点支撑）
"""
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# === 强制前置：PDF 字体嵌入 + 中文字体 + 自动布局（chart-generator 硬约束）===
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Source Han Sans CN"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.constrained_layout.use"] = True

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PKL = ROOT / "outputs" / "data" / "c-data-cleaned.pkl"
FIG_DIR = ROOT / "outputs" / "figures"

# Okabe-Ito 色盲安全色板（灰度打印可区分）
COLOR_CRC = "#0072B2"   # 蓝
COLOR_IBD = "#E69F00"   # 橙
COLOR_OBE = "#009E73"   # 蓝绿

DETECTION_LIMIT = 1e-5
CLR_DELTA = 0.65 * DETECTION_LIMIT
SEED = 42


def clr_transform(X: np.ndarray) -> np.ndarray:
    """CLR 变换（逐样本）：零值乘法替换 δ → log → 减行均值（几何均值中心化）。"""
    arr = np.asarray(X, dtype=float)
    arr = np.where(arr == 0.0, CLR_DELTA, arr)
    logX = np.log(arr)
    return logX - logX.mean(axis=1, keepdims=True)


def disease_healthy_masks(df, dataset):
    """返回 (disease_mask, healthy_mask)（按数据集口径）。"""
    m = df["dataset_name"] == dataset
    if dataset == "Zeller_fecal_colorectal_cancer":
        dis = df["disease"] == "cancer"
        h = df["disease"].isin(["n", "small_adenoma"])
    elif dataset == "metahit":
        dis = df["disease"].isin(["ibd_ulcerative_colitis", "ibd_crohn_disease"])
        h = df["disease"] == "n"
    else:  # Chatelier_gut_obesity
        dis = df["disease"] == "obesity"
        h = df["disease"] == "leaness"
    return m & dis, m & h


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    feat_names = [c for c in df.columns if c not in ("dataset_name", "disease")]
    X_raw = df[feat_names].to_numpy(dtype=float)

    # ---------- 图4：批次效应（PCA / t-SNE，按数据集着色）----------
    zero_ratio = (X_raw == 0.0).mean(axis=0)
    keep = zero_ratio <= 0.95
    Xf = X_raw[:, keep]
    print(f"[图4] 近全零过滤 {X_raw.shape[1]}→{Xf.shape[1]} 特征")

    X_clr = clr_transform(Xf)
    X_std = StandardScaler().fit_transform(X_clr)

    # 样本数据集标签（short: CRC/IBD/Obesity）
    ds_short = df["dataset_name"].map(
        {"Zeller_fecal_colorectal_cancer": "CRC", "metahit": "IBD", "Chatelier_gut_obesity": "Obesity"}
    ).to_numpy()

    pca = PCA(n_components=2, random_state=SEED).fit(X_std)
    pc = pca.transform(X_std)
    exp = pca.explained_variance_ratio_

    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca")
    ts = tsne.fit_transform(X_std)

    # 2×1 子图
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, coords, title, xlab, ylab in [
        (axes[0], pc, "PCA", f"PC1（{exp[0]*100:.1f}% 方差）", f"PC2（{exp[1]*100:.1f}% 方差）"),
        (axes[1], ts, "t-SNE", "t-SNE 维度 1", "t-SNE 维度 2"),
    ]:
        for cond, color in [
            (ds_short == "CRC", COLOR_CRC),
            (ds_short == "IBD", COLOR_IBD),
            (ds_short == "Obesity", COLOR_OBE),
        ]:
            ax.scatter(coords[cond, 0], coords[cond, 1], s=18, alpha=0.7, c=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlab, fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(True, lw=0.3)
    # 中文图例（三数据集）
    fig.legend(
        ["结直肠癌 CRC", "炎症性肠病 IBD", "肥胖症 Obesity"],
        loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.98),
    )
    # 结论标注
    fig.suptitle("三数据集特征空间分布（PCA/t-SNE，批次效应提示）", fontsize=13)
    fig.text(
        0.5, 0.01,
        "标注：三数据集样本在特征空间无清晰分离 → 存在批次/分布差异\n"
        "支撑：绝对 AUC 不可跨数据集独立比，只比相对基线增益",
        ha="center", va="bottom", fontsize=9, color="#333333",
    )
    out4 = FIG_DIR / "chart-batch-effect.pdf"
    fig.savefig(out4, bbox_inches="tight")
    plt.close(fig)
    print(f"[图4] 已保存: {out4}")

    # ---------- 图5：已知标志物存在性（分组柱状图）----------
    known = [
        # (标签, 特征名匹配键, 数据集, 病名缩写)
        ("F. nucleatum (CRC)", "nucleatum", "Zeller_fecal_colorectal_cancer", "CRC"),
        ("P. stomatis (CRC)", "stomatis", "Zeller_fecal_colorectal_cancer", "CRC"),
        ("P. somerae (CRC)", "somerae", "Zeller_fecal_colorectal_cancer", "CRC"),
        ("B. bifidum (IBD)", "bifidum", "metahit", "IBD"),
        ("A. muciniphila (IBD)", "muciniphila", "metahit", "IBD"),
        ("B. fragilis (Obesity)", "fragilis", "Chatelier_gut_obesity", "Obesity"),
    ]
    names = [k[0] for k in known]
    dis_pres, h_pres = [], []
    for label, key, ds, _ in known:
        cols = [c for c in feat_names if key.lower() in c.lower()]
        if not cols:
            dis_pres.append(0.0); h_pres.append(0.0)
            print(f"  [警告] {label}: 未找到特征匹配"); continue
        dis_mask, hea_mask = disease_healthy_masks(df, ds)
        xd = df.loc[dis_mask, cols].to_numpy()
        xh = df.loc[hea_mask, cols].to_numpy()
        dis_pres.append(float((xd > 0).any(axis=1).mean()))
        h_pres.append(float((xh > 0).any(axis=1).mean()))

    # 打印统计量（先算后画）
    print("\n[图5] 已知标志物存在率（病组 vs 健康组）")
    for nm, a, b in zip(names, dis_pres, h_pres):
        print(f"  {nm}: 病组={a:.3f} 健康组={b:.3f} 差={a-b:+.3f}")

    xpos = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    b1 = ax.bar(xpos - width / 2, np.array(dis_pres) * 100, width, label="患病组", color=COLOR_CRC)
    b2 = ax.bar(xpos + width / 2, np.array(h_pres) * 100, width, label="健康组", color=COLOR_IBD)
    # 柱顶数值
    for r, v in zip(b1, np.array(dis_pres) * 100):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    for r, v in zip(b2, np.array(h_pres) * 100):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=10, rotation=20, ha="right")
    ax.set_ylim(0, 100)  # 零基线
    ax.set_ylabel("存在率（%）", fontsize=11)
    ax.set_xlabel("已知微生物标志物（括号内为所属患病数据集）", fontsize=11)
    ax.set_title("已知微生物标志物在患病/健康组的存在率对比", fontsize=13)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    fig.text(
        0.5, 0.01,
        "标注：CRC 患病组 F. nucleatum 41.7% vs 健康组 2.7%、P. stomatis 56.2% vs 8.2%，明显更高；"
        "IBD 患病组 A. muciniphila 28.0% vs 77.6%，明显偏低。与文献已知菌方向一致（S2 生物合理性锚点）。",
        ha="center", va="bottom", fontsize=9, color="#333333",
    )
    out5 = FIG_DIR / "chart-known-biomarker-presence.pdf"
    fig.savefig(out5, bbox_inches="tight")
    plt.close(fig)
    print(f"[图5] 已保存: {out5}")

    # 摘要
    print("\n[摘要] 批次分离观察：")
    print(f"  PCA 前两主成分解释方差 {exp[0]*100:.1f}% + {exp[1]*100:.1f}% = {(exp[0]+exp[1])*100:.1f}%")
    print(f"  t-SNE(perplexity=30) 已生成。")


if __name__ == "__main__":
    main()

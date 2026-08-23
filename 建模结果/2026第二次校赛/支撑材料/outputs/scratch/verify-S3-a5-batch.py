"""
目的：
    S3 A 类验证 A5【批次效应初探】：CLR 后对三数据集做 PCA 与 t-SNE 投影，按 dataset_name
    着色，判断三数据集是否因批次（不同研究/平台）清晰分开，辅助区分"批次差异"与"疾病差异"。

原理：
    - CLR 变换（逐样本，解除定和约束）后，用 PCA（线性，前 2 主成分）与 t-SNE（非线性，
      perplexity=30，seed=42）投影到 2 维。
    - 按 dataset_name 着色：若三数据集在投影中清晰分簇 → 批次效应强，跨疾病预测挑战大；
      若按疾病（患病/健康）而非数据集分簇 → 疾病信号主导，批次弱。
    - 同时按 disease 二分类标签着色作对照，看簇结构由"数据集"还是"疾病"主导。

性能：
    轻量-不适用（484 样本 PCA/t-SNE，秒级）。

输入数据：
    - B-raw.pkl (处理后) — dataset_name, disease, 1331 物种级相对丰度特征

输出：
    - outputs/figures/_explore/S3-batch-pca-tsne.pdf — PCA + t-SNE 散点（按数据集/疾病着色）
    - stdout — 各主成分解释方差比

对应论文章节：
    §S3 跨疾病预测模型（A 类验证 A5，探索图不入论文）
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_S3_common import (  # noqa: E402
    load_data, clr_transform, binary_label, FIG_DIR, DATASET_DISEASE,
)

FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df, feature_cols = load_data()
    X = df[feature_cols].astype(float)
    Xc = clr_transform(X)
    ds = df["dataset_name"].map(DATASET_DISEASE)
    y = binary_label(df["disease"])

    # PCA
    pca = PCA(n_components=2, random_state=42)
    Zp = pca.fit_transform(Xc)
    print("=" * 70)
    print("A5 批次效应初探：PCA / t-SNE")
    print("=" * 70)
    print(f"PCA 前 2 主成分解释方差比: {pca.explained_variance_ratio_[0]:.4f}, "
          f"{pca.explained_variance_ratio_[1]:.4f}")

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    Zt = tsne.fit_transform(Xc)

    # 图：2×2（PCA 按数据集 / PCA 按疾病 / t-SNE 按数据集 / t-SNE 按疾病）
    ds_colors = {"CRC": "#4C72B0", "IBD": "#DD8452", "Obesity": "#55A868"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    # PCA 按数据集
    ax = axes[0, 0]
    for d in ["CRC", "IBD", "Obesity"]:
        m = ds == d
        ax.scatter(Zp[m, 0], Zp[m, 1], s=12, alpha=0.7, label=d, color=ds_colors[d])
    ax.set_title("PCA（按数据集着色）")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(fontsize=8)
    # PCA 按疾病
    ax = axes[0, 1]
    for lab, c, name in [(0, "#999999", "健康"), (1, "#C44E52", "患病")]:
        m = y == lab
        ax.scatter(Zp[m, 0], Zp[m, 1], s=12, alpha=0.7, label=name, color=c)
    ax.set_title("PCA（按患病/健康着色）")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(fontsize=8)
    # t-SNE 按数据集
    ax = axes[1, 0]
    for d in ["CRC", "IBD", "Obesity"]:
        m = ds == d
        ax.scatter(Zt[m, 0], Zt[m, 1], s=12, alpha=0.7, label=d, color=ds_colors[d])
    ax.set_title("t-SNE（按数据集着色）")
    ax.legend(fontsize=8)
    # t-SNE 按疾病
    ax = axes[1, 1]
    for lab, c, name in [(0, "#999999", "健康"), (1, "#C44E52", "患病")]:
        m = y == lab
        ax.scatter(Zt[m, 0], Zt[m, 1], s=12, alpha=0.7, label=name, color=c)
    ax.set_title("t-SNE（按患病/健康着色）")
    ax.legend(fontsize=8)
    fig.suptitle("S3 A5 批次效应初探：三数据集分布", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / "S3-batch-pca-tsne.pdf"
    fig.savefig(out)
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()

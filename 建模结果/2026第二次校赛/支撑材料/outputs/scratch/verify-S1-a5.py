"""
目的：
    S1 A 类验证 #5：三数据集类内可分性 / 批次差异（PCA + t-SNE 降维可视化）。

原理：
    对全量 484 样本做 CLR 变换（解除定和偏相关）后，PCA（线性，前 2 主成分）与 t-SNE（非线性，perplexity=30）
    降维到 2D，分别按 dataset（观察批次效应/簇结构）与 disease（观察患病 vs 健康分离度）着色。
    若三数据集在低维空间明显分簇 → 批次效应强，跨疾病差异分析须按批次归因；若同数据集内患病/健康重叠 → 信号弱。

性能：
    轻量-不适用（484×1331 的 PCA 秒级；t-SNE 十秒级）。

输入数据：
    - B-raw.pkl（原始）— dataset_name, disease, 1331 物种丰度

输出：
    - outputs/figures/_explore/S1-pca-tsne-explore.pdf — 2×2 降维投影（dataset/disease 着色）
    - stdout — PCA 前 2 主成分解释方差比

对应论文章节：
    §1.1 A 类验证（探索，不入论文）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import utils


def main():
    df = utils.load_data()
    names = list(utils.DATASETS.keys())
    short = {
        "Zeller_fecal_colorectal_cancer": "Zeller CRC",
        "metahit": "metahit IBD",
        "Chatelier_gut_obesity": "Chatelier Obesity",
    }
    # 全量 CLR
    Xall = df.drop(columns=["dataset_name", "disease"]).values.astype(np.float64)
    Xc = utils.clr_transform(Xall)
    ds = df["dataset_name"].values
    dis = df["disease"].values

    pca = PCA(n_components=2, random_state=42)
    Zp = pca.fit_transform(Xc)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
    Zt = tsne.fit_transform(Xc)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    # disease 二值（患病=1）
    ybin = np.zeros(len(df), dtype=int)
    for name in names:
        cfg = utils.DATASETS[name]
        m = ds == name
        ybin[m] = dis[m].isin(cfg["positive"]).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ds_colors = {"Zeller_fecal_colorectal_cancer": "#4C72B0", "metahit": "#55A868", "Chatelier_gut_obesity": "#C44E52"}
    for ax, Z, title in [(axes[0, 0], Zp, "PCA (by dataset)"), (axes[0, 1], Zt, "t-SNE (by dataset)")]:
        for name in names:
            m = ds == name
            ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.6, label=short[name], color=ds_colors[name])
        ax.set_title(title)
        ax.legend(fontsize=7, markerscale=2)
    for ax, Z, title in [(axes[1, 0], Zp, "PCA (by disease)"), (axes[1, 1], Zt, "t-SNE (by disease)")]:
        ax.scatter(Z[ybin == 0, 0], Z[ybin == 0, 1], s=8, alpha=0.5, label="healthy", color="#55A868")
        ax.scatter(Z[ybin == 1, 0], Z[ybin == 1, 1], s=8, alpha=0.5, label="disease", color="#C44E52")
        ax.set_title(title)
        ax.legend(fontsize=7, markerscale=2)
    fig.suptitle("S1 separability / batch effect (CLR-transformed)", fontsize=12)
    fig.tight_layout()
    out = utils.ensure_fig_dir() / "S1-pca-tsne-explore.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

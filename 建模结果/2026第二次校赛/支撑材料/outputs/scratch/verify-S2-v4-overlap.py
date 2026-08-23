"""
目的：
    V4 三数据集特征重叠 + 已知标志物检出率：三数据集非零特征集合两两 Jaccard 重叠，
    已知标志物（Fusobacterium nucleatum 等）在各数据集的检出率（非零样本比例）。

原理：
    - 非零特征集合 = 该数据集内至少一个样本非零的特征（即"存在"于该数据集的物种）。
    - Jaccard(A,B) = |A∩B| / |A∪B|，衡量两数据集共享物种比例。
    - 已知标志物按种名/属名在特征名（s__ 段）中匹配：
      Fusobacterium nucleatum、Faecalibacterium prausnitzii、Bifidobacterium(属)、
      Peptostreptococcus stomatis、Parvimonas micra、Porphyromonas(属)、Bacteroides fragilis。
    - 检出率 = 该标志物特征非零样本数 / 该数据集样本数。

性能：
    轻量-不适用（集合运算 + 字符串匹配，秒级，无并行需求）。

输入数据：
    - B-raw.pkl (原始) — dataset_name, disease, 1331 物种相对丰度特征

输出：
    - outputs/figures/_explore/S2-v4-dataset-overlap-explore.pdf — Jaccard 重叠热图
    - outputs/figures/_explore/S2-v4-known-biomarker-presence-explore.pdf — 已知标志物检出率热图
    - stdout — Jaccard 矩阵 + 已知标志物检出率表

对应论文章节：
    §1.1 A 类验证 V4（三数据集重叠 + 已知标志物）
"""
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

from utils_S2 import DATASETS, FIG_DIR, get_X, get_feature_names, load_df

KNOWN = {
    "Fusobacterium nucleatum": "nucleatum",
    "Faecalibacterium prausnitzii": "prausnitzii",
    "Bifidobacterium (属)": "Bifidobacterium",
    "Peptostreptococcus stomatis": "stomatis",
    "Parvimonas micra": "micra",
    "Porphyromonas (属)": "Porphyromonas",
    "Bacteroides fragilis": "fragilis",
}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_df()
    feat_names = get_feature_names(df)

    # 1. 非零特征集合
    nonzero_sets = {}
    for dataset in DATASETS:
        X = get_X(df, dataset)
        nz = (X > 0).any(axis=0)
        nonzero_sets[dataset] = set(np.where(nz)[0])

    # 2. Jaccard 矩阵
    names = list(DATASETS.keys())
    shorts = [DATASETS[d]["short"] for d in names]
    jac = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            a, b = nonzero_sets[names[i]], nonzero_sets[names[j]]
            jac[i, j] = len(a & b) / len(a | b)

    # 3. 已知标志物检出率
    det = np.zeros((len(KNOWN), 3))
    for ki, (label, key) in enumerate(KNOWN.items()):
        # 匹配特征名（s__ 段含 key，或整名含 key）
        matched = [n for n in feat_names if key.lower() in n.lower()]
        for di, dataset in enumerate(names):
            X = get_X(df, dataset)
            if matched:
                cols = [feat_names.index(m) for m in matched]
                det[ki, di] = (X[:, cols] > 0).any(axis=1).mean()
            else:
                det[ki, di] = 0.0

    # 图1：Jaccard 热图
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(jac, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(shorts)
    ax.set_yticklabels(shorts)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{jac[i, j]:.2f}", ha="center", va="center", color="white")
    ax.set_title("三数据集非零特征 Jaccard 重叠")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out1 = FIG_DIR / "S2-v4-dataset-overlap-explore.pdf"
    fig.savefig(out1)
    plt.close(fig)

    # 图2：已知标志物检出率热图
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(det, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(len(KNOWN)))
    ax.set_xticklabels(shorts)
    ax.set_yticklabels(list(KNOWN.keys()))
    for i in range(len(KNOWN)):
        for j in range(3):
            ax.text(j, i, f"{det[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title("已知标志物检出率（非零样本比例）")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out2 = FIG_DIR / "S2-v4-known-biomarker-presence-explore.pdf"
    fig.savefig(out2)
    plt.close(fig)

    # 打印
    print("=" * 70)
    print("V4 三数据集特征重叠 + 已知标志物检出率")
    print("=" * 70)
    print("\n[Jaccard 重叠矩阵]")
    print("        " + "  ".join(f"{s:>8s}" for s in shorts))
    for i in range(3):
        print(f"{shorts[i]:8s}" + "  ".join(f"{jac[i, j]:8.3f}" for j in range(3)))
    print("\n[已知标志物检出率]")
    print(f"{'标志物':30s}" + "  ".join(f"{s:>8s}" for s in shorts))
    for ki, label in enumerate(KNOWN):
        print(f"{label:30s}" + "  ".join(f"{det[ki, j]:8.3f}" for j in range(3)))
    print(f"\n[图] {out1}")
    print(f"[图] {out2}")


if __name__ == "__main__":
    main()

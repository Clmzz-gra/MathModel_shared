"""
目的：
    V3 特征冗余度：抽样特征计算 Spearman 相关矩阵，|ρ|>0.7 连边构建图，
    数连通簇数与最大簇规模，判断 Lasso 共线组内任选 / RF 重要性分散的严重程度。

原理：
    - 每病抽样 300 特征（不足则全取），对全样本（含零值）算 Spearman 相关矩阵——
      稀疏成分数据中零值共现模式本身就是主要冗余来源，故含零值更贴近真实冗余。
    - |ρ|>0.7 连边（无向图），用并查集（union-find）数连通分量，统计簇规模分布。
    - 高冗余（大簇多）→ Lasso 在共线组内任选一个、RF 重要性被分散，需先聚类去冗余。

性能：
    轻量-不适用（300×300 Spearman 相关 × 3 病，秒级，无并行需求）。

输入数据：
    - B-raw.pkl (原始) — dataset_name, disease, 1331 物种相对丰度特征

输出：
    - outputs/figures/_explore/S2-v3-correlation-clusters-explore.pdf — 相关簇规模分布
    - stdout — 连通簇数 + 最大簇规模 + 最大簇成员示例

对应论文章节：
    §1.1 A 类验证 V3（特征冗余度）
"""
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils_S2 import DATASETS, FIG_DIR, get_X, get_feature_names, load_df

N_SAMPLE = 300
RHO_TH = 0.7


def union_find(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_df()
    feat_names = get_feature_names(df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    summary = {}

    for ax, (dataset, cfg) in zip(axes, DATASETS.items()):
        X = get_X(df, dataset)
        n_feat = X.shape[1]
        # 抽样特征（优先非零占比较高的，保证相关可算）
        zf = (X == 0).mean(axis=0)
        order = np.argsort(zf)  # 低零值占比在前
        idx = order[: min(N_SAMPLE, n_feat)]
        Xs = X[:, idx]

        # Spearman 相关矩阵（含零值）
        corr = pd.DataFrame(Xs).corr(method="spearman").to_numpy().copy()
        np.fill_diagonal(corr, 0.0)
        edges = [(a, b) for a in range(len(idx)) for b in range(a + 1, len(idx)) if abs(corr[a, b]) > RHO_TH]
        roots = union_find(len(idx), edges)
        from collections import Counter

        sizes = sorted(Counter(roots).values(), reverse=True)
        n_clusters = len(sizes)
        max_cluster = sizes[0] if sizes else 0
        n_edges = len(edges)

        # 最大簇成员示例
        max_root = max(Counter(roots), key=Counter(roots).get)
        members = [feat_names[idx[i]].split("|")[-1] for i in range(len(idx)) if roots[i] == max_root][:5]

        summary[cfg["short"]] = {
            "n_sampled": len(idx),
            "n_edges": n_edges,
            "n_clusters": n_clusters,
            "max_cluster": max_cluster,
            "members": members,
        }

        # 图：簇规模分布
        ax.bar(range(len(sizes)), sizes)
        ax.set_xlabel("簇序号（按规模降序）")
        ax.set_ylabel("簇规模")
        ax.set_title(f"{cfg['short']}  (|ρ|>{RHO_TH} 边数={n_edges})")

    fig.suptitle("V3 特征相关簇规模分布（Spearman |ρ|>0.7）", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "S2-v3-correlation-clusters-explore.pdf"
    fig.savefig(out)
    plt.close(fig)

    print("=" * 70)
    print("V3 特征冗余度（Spearman |ρ|>0.7 连通簇）")
    print("=" * 70)
    for short, s in summary.items():
        print(f"\n[{short}] 抽样 {s['n_sampled']} 特征")
        print(f"  高相关边数: {s['n_edges']}")
        print(f"  连通簇数: {s['n_clusters']}，最大簇规模: {s['max_cluster']}")
        print(f"  最大簇成员示例: {', '.join(s['members'])}")
    print(f"\n[图] {out}")


if __name__ == "__main__":
    main()

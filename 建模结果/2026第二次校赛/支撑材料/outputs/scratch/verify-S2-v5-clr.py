"""
目的：
    V5 CLR 前置必要性：抽样特征对，比较原始丰度 vs CLR 后的 Pearson/Spearman 相关，
    统计相关方向/强度改变的特征对比例，判断定和成分伪相关是否需 CLR 前置。

原理：
    - 成分数据（行和≈100）存在定和约束：某特征丰度升高必然挤压其他特征，
      导致原始丰度间出现"伪负相关"。CLR（log 比几何均值）消除该约束。
    - 抽样 100 对特征，分别算原始丰度与 CLR 后的 Pearson 相关、Spearman 相关。
    - 统计：方向翻转（符号改变）比例、|Δρ|>0.3 的强度改变比例。
      若大量特征对相关被 CLR 显著改变 → 定和伪相关显著，需 CLR 前置。

性能：
    轻量-不适用（100 对 × 2 变换 × 2 相关，秒级，无并行需求）。

输入数据：
    - B-raw.pkl (原始) — dataset_name, disease, 1331 物种相对丰度特征

输出：
    - outputs/figures/_explore/S2-v5-clr-correlation-explore.pdf — 原始 vs CLR 相关散点图
    - stdout — 方向翻转比例 + 强度改变比例

对应论文章节：
    §1.1 A 类验证 V5（CLR 前置必要性）
"""
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from utils_S2 import DATASETS, FIG_DIR, clr, get_X, load_df

N_PAIRS = 100
SEED = 0


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_df()
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    summary = {}

    for ax, (dataset, cfg) in zip(axes, DATASETS.items()):
        X = get_X(df, dataset)
        Xc = clr(X)
        n_feat = X.shape[1]

        # 抽样特征对（优先非零占比较高的，保证相关可算）
        zf = (X == 0).mean(axis=0)
        order = np.argsort(zf)[:200]  # 低零值占比前 200
        pairs = rng.choice(len(order), size=(N_PAIRS, 2), replace=True)
        pairs = order[pairs]

        raw_pear, clr_pear = [], []
        raw_spear, clr_spear = [], []
        for a, b in pairs:
            raw_pear.append(pearsonr(X[:, a], X[:, b])[0])
            clr_pear.append(pearsonr(Xc[:, a], Xc[:, b])[0])
            raw_spear.append(spearmanr(X[:, a], X[:, b])[0])
            clr_spear.append(spearmanr(Xc[:, a], Xc[:, b])[0])
        raw_pear = np.array(raw_pear)
        clr_pear = np.array(clr_pear)
        raw_spear = np.array(raw_spear)
        clr_spear = np.array(clr_spear)

        # 方向翻转 + 强度改变
        flip_pear = np.mean(np.sign(raw_pear) != np.sign(clr_pear))
        flip_spear = np.mean(np.sign(raw_spear) != np.sign(clr_spear))
        big_pear = np.mean(np.abs(raw_pear - clr_pear) > 0.3)
        big_spear = np.mean(np.abs(raw_spear - clr_spear) > 0.3)

        summary[cfg["short"]] = {
            "flip_pear": flip_pear,
            "flip_spear": flip_spear,
            "big_pear": big_pear,
            "big_spear": big_spear,
        }

        # 图：原始 vs CLR Pearson 相关散点
        ax.scatter(raw_pear, clr_pear, s=8, alpha=0.6, label="Pearson")
        ax.scatter(raw_spear, clr_spear, s=8, alpha=0.6, marker="^", label="Spearman")
        lim = [-1, 1]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("原始丰度相关")
        ax.set_ylabel("CLR 后相关")
        ax.set_title(cfg["short"])
        ax.legend(fontsize=8)

    fig.suptitle("V5 原始丰度 vs CLR 相关（对角线=无变化）", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "S2-v5-clr-correlation-explore.pdf"
    fig.savefig(out)
    plt.close(fig)

    print("=" * 70)
    print("V5 CLR 前置必要性（原始 vs CLR 相关差异）")
    print("=" * 70)
    for short, s in summary.items():
        print(f"\n[{short}]")
        print(f"  Pearson 方向翻转比例: {s['flip_pear']:.2f}")
        print(f"  Spearman 方向翻转比例: {s['flip_spear']:.2f}")
        print(f"  Pearson |Δρ|>0.3 比例: {s['big_pear']:.2f}")
        print(f"  Spearman |Δρ|>0.3 比例: {s['big_spear']:.2f}")
    print(f"\n[图] {out}")


if __name__ == "__main__":
    main()

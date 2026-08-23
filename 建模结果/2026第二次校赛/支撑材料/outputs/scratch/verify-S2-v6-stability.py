"""
目的：
    V6 Lasso bootstrap 频率分布：每病 B=50 次分层 bootstrap，每次在重采样上拟合
    Lasso 稀疏 Logistic，统计每特征入选频率，画频率直方图，检验"高频率稳定簇"是否真实存在。

原理：
    - 稳定性选择：小样本下单次 L1 选择方差大，多轮 bootstrap 聚合出现频率，
      频率 ≥ τ（如 80%）的特征为稳定标志物。
    - 每轮：CLR + 标准化（全量数据上做一次，bootstrap 只重采样行）→ Lasso(penalty='l1',
      C 固定) 拟合 → 非零系数特征记为"入选"。
    - 频率直方图若呈双峰（大量 0 频率 + 少量高频率簇）→ 稳定簇真实存在，τ 落在自然间断点；
      若频率平缓连续 → 无稳定簇，需调 τ 或换方法。

性能：
    轻量-不适用（3 病 × 50 轮 Lasso，每轮 ~0.1s，总计 ~15s，无并行需求）。

输入数据：
    - B-raw.pkl (原始) — dataset_name, disease, 1331 物种相对丰度特征

输出：
    - outputs/figures/_explore/S2-v6-stability-frequency-explore.pdf — 频率直方图
    - stdout — 高频特征 Top 列表 + 频率分布特征

对应论文章节：
    §1.1 A 类验证 V6（Lasso bootstrap 稳定性）
"""
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils_S2 import DATASETS, FIG_DIR, clr, get_X, get_feature_names, get_label, load_df

B = 50
C = 0.1
SEED = 0


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_df()
    feat_names = get_feature_names(df)
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    summary = {}

    for ax, (dataset, cfg) in zip(axes, DATASETS.items()):
        X = get_X(df, dataset)
        y = get_label(df, dataset)
        n_feat = X.shape[1]

        # CLR + 标准化（全量一次）
        Xc = clr(X)
        Xs = StandardScaler().fit_transform(Xc)

        # 分层 bootstrap
        idx_d = np.where(y == 1)[0]
        idx_h = np.where(y == 0)[0]
        freq = np.zeros(n_feat)
        for b in range(B):
            bd = rng.choice(idx_d, size=len(idx_d), replace=True)
            bh = rng.choice(idx_h, size=len(idx_h), replace=True)
            idx = np.concatenate([bd, bh])
            l1 = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000, random_state=b)
            l1.fit(Xs[idx], y[idx])
            freq += (np.abs(l1.coef_[0]) > 1e-8).astype(float)
        freq /= B

        # 统计
        n_ever = int((freq > 0).sum())
        n_high = int((freq >= 0.8).sum())
        n_mid = int(((freq >= 0.5) & (freq < 0.8)).sum())
        top_idx = np.argsort(freq)[::-1][:10]
        top_feats = [(feat_names[i].split("|")[-1], freq[i]) for i in top_idx]

        summary[cfg["short"]] = {
            "n_ever": n_ever,
            "n_high80": n_high,
            "n_mid50": n_mid,
            "top_feats": top_feats,
        }

        # 图：频率直方图
        ax.hist(freq, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
        ax.axvline(0.8, color="red", ls="--", lw=1, label="τ=0.8")
        ax.set_xlabel("入选频率")
        ax.set_ylabel("特征数")
        ax.set_title(f"{cfg['short']}  (入选过 {n_ever} 特征)")
        ax.legend(fontsize=8)

    fig.suptitle("V6 Lasso bootstrap 入选频率分布（B=50）", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "S2-v6-stability-frequency-explore.pdf"
    fig.savefig(out)
    plt.close(fig)

    print("=" * 70)
    print("V6 Lasso bootstrap 入选频率分布（B=50, C=0.1）")
    print("=" * 70)
    for short, s in summary.items():
        print(f"\n[{short}]")
        print(f"  入选过(频率>0)特征数: {s['n_ever']}")
        print(f"  频率≥0.8 稳定特征数: {s['n_high80']}")
        print(f"  频率 0.5~0.8 特征数: {s['n_mid50']}")
        print("  Top10 稳定特征:")
        for name, f in s["top_feats"]:
            print(f"    {name:40s} {f:.2f}")
    print(f"\n[图] {out}")


if __name__ == "__main__":
    main()

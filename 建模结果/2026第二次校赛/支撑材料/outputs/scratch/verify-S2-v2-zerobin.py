"""
目的：
    V2 零值占比分箱影响：将每病特征按零值占比分箱（0-20/20-40/40-60/60-80/80-100%），
    统计各候选方法（Wilcoxon p / L1 入选 / RF 重要性 / VIP）每箱选中率，
    量化"零值主导 → 方法失效"边界。

原理：
    - 零值占比分箱：每特征零值样本比例，落入 5 个箱。
    - 各方法选中率 = 该箱内被方法选中的特征数 / 该箱内特征总数：
      · Wilcoxon：BH-FDR<0.05 显著；
      · L1：CLR+标准化后 Lasso LogisticRegression(penalty='l1') 非零系数；
      · RF：RandomForest 重要性 top-20；
      · VIP：PLS-DA(2 成分) VIP>1。
    - 若高零值箱选中率骤降 → 该零值占比以上方法失效，需先过滤近全零特征。

性能：
    轻量-不适用（3 病 × 各 1 次 L1/RF/PLS 拟合 + 1331 次 Wilcoxon，秒级，无并行需求）。

输入数据：
    - B-raw.pkl (原始) — dataset_name, disease, 1331 物种相对丰度特征

输出：
    - outputs/figures/_explore/S2-v2-zerobin-selection-rate-explore.pdf — 分箱→选中率柱状图
    - stdout — 各方法失效零值边界 + 过滤近全零特征后维度

对应论文章节：
    §1.1 A 类验证 V2（零值占比分箱）
"""
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils_S2 import DATASETS, FIG_DIR, bh_fdr, clr, get_X, get_label, load_df, zero_fraction

BINS = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
BIN_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]


def vip_scores(X_std, y):
    pls = PLSRegression(n_components=2)
    pls.fit(X_std, y)
    t = pls.x_scores_
    w = pls.x_weights_
    p, h = w.shape
    s = np.diag(t.T @ t).reshape(h, -1)
    total_s = np.sum(s)
    return np.sqrt(p * (s.T @ (w**2).T) / total_s).flatten()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_df()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    methods = ["Wilcoxon", "L1", "RF", "VIP"]
    all_rates = {}

    for ax, (dataset, cfg) in zip(axes, DATASETS.items()):
        X = get_X(df, dataset)
        y = get_label(df, dataset)
        n_feat = X.shape[1]
        zf = zero_fraction(X)

        # 各方法选中标记
        # Wilcoxon
        pvals = np.empty(n_feat)
        for j in range(n_feat):
            x_d = X[y == 1, j]
            x_h = X[y == 0, j]
            pvals[j] = 1.0 if np.all(X[:, j] == X[0, j]) else mannwhitneyu(x_d, x_h).pvalue
        sel_wil = bh_fdr(pvals)

        # L1（CLR + 标准化）
        Xc = clr(X)
        Xs = StandardScaler().fit_transform(Xc)
        l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=2000, random_state=0)
        l1.fit(Xs, y)
        sel_l1 = np.abs(l1.coef_[0]) > 1e-8

        # RF 重要性 top-20
        rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        rf.fit(X, y)
        top20 = np.argsort(rf.feature_importances_)[::-1][:20]
        sel_rf = np.zeros(n_feat, dtype=bool)
        sel_rf[top20] = True

        # VIP>1
        vip = vip_scores(Xs, y)
        sel_vip = vip > 1.0

        sels = {"Wilcoxon": sel_wil, "L1": sel_l1, "RF": sel_rf, "VIP": sel_vip}

        # 分箱选中率
        rates = {m: [] for m in methods}
        for lo, hi in BINS:
            mask = (zf >= lo) & (zf < hi)
            n_bin = int(mask.sum())
            for m in methods:
                rate = sels[m][mask].sum() / n_bin if n_bin > 0 else 0.0
                rates[m].append(rate)
        all_rates[cfg["short"]] = rates

        # 图
        xpos = np.arange(len(BIN_LABELS))
        width = 0.2
        for i, m in enumerate(methods):
            ax.bar(xpos + (i - 1.5) * width, rates[m], width, label=m)
        ax.set_xticks(xpos)
        ax.set_xticklabels(BIN_LABELS)
        ax.set_xlabel("零值占比分箱")
        ax.set_ylabel("选中率")
        ax.set_title(cfg["short"])
        ax.legend(fontsize=8)

    fig.suptitle("V2 零值占比分箱 → 各方法选中率", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "S2-v2-zerobin-selection-rate-explore.pdf"
    fig.savefig(out)
    plt.close(fig)

    # 打印 + 过滤近全零特征后维度
    print("=" * 70)
    print("V2 零值占比分箱 → 各方法选中率")
    print("=" * 70)
    for short, rates in all_rates.items():
        print(f"\n[{short}]")
        for m in methods:
            print(f"  {m:8s}: " + "  ".join(f"{r:.2f}" for r in rates[m]))
    # 过滤近全零（零值>95%）后维度
    df2 = df
    X_all = df2[[c for c in df2.columns if c not in ("dataset_name", "disease")]].astype(float).to_numpy()
    zf_all = zero_fraction(X_all)
    n_keep = int((zf_all <= 0.95).sum())
    print(f"\n[过滤] 零值占比>95% 的特征数: {int((zf_all > 0.95).sum())}，过滤后维度: {n_keep} / 1331")
    print(f"[图] {out}")


if __name__ == "__main__":
    main()

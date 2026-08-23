"""
目的：
    V1 基线验证：单特征 Wilcoxon 判别力下界——每病每特征病 vs 健 Wilcoxon + BH-FDR，
    计算 Top 单特征 AUC 量级，并区分"零值占比差"与"非零丰度差"哪个主导判别力。

原理：
    - Wilcoxon 秩和检验（scipy.stats.mannwhitneyu）对每特征做病 vs 健差异检验，BH-FDR 校正 1331 次比较。
    - 判别力下界：单特征 AUC（roc_auc_score，取 max(AUC,1-AUC) 消除方向影响）。
      单特征无拟合参数，全量 AUC 即诚实估计（无乐观偏差），故直接报全量 AUC。
    - 信号拆分：零值占比差 = 病组零值占比 - 健组零值占比（存在/缺失信号）；
      非零丰度差 = 对非零值做 Wilcoxon 的 -log10(p)（丰度高/低信号）。
      对 Top 特征统计两者量级，判断哪个主导判别力。

性能：
    轻量-不适用（1331 特征 × 3 病，Wilcoxon 循环 + 单特征 AUC，秒级，无并行需求）。

输入数据：
    - B-raw.pkl (原始) — dataset_name, disease, 1331 物种相对丰度特征

输出：
    - outputs/figures/_explore/S2-v1-top-single-feature-dist-explore.pdf — Top 单特征病/健分布
    - stdout — Top 单特征列表 + AUC 量级 + 信号主导判定

对应论文章节：
    §1.1 A 类验证 V1（基线）
"""
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from utils_S2 import DATASETS, FIG_DIR, bh_fdr, get_X, get_feature_names, get_label, load_df


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_df()
    feat_names = get_feature_names(df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    summary = {}

    for ax, (dataset, cfg) in zip(axes, DATASETS.items()):
        X = get_X(df, dataset)
        y = get_label(df, dataset)
        n_feat = X.shape[1]

        # 1. 每特征 Wilcoxon + FDR
        pvals = np.empty(n_feat)
        for j in range(n_feat):
            x_d = X[y == 1, j]
            x_h = X[y == 0, j]
            if np.all(X[:, j] == X[0, j]):
                pvals[j] = 1.0
            else:
                pvals[j] = mannwhitneyu(x_d, x_h, alternative="two-sided").pvalue
        sig = bh_fdr(pvals)
        n_sig = int(sig.sum())

        # 2. 单特征 AUC（max(AUC,1-AUC)）
        aucs = np.empty(n_feat)
        for j in range(n_feat):
            x = X[:, j]
            if np.all(x == x[0]):
                aucs[j] = 0.5
            else:
                a = roc_auc_score(y, x)
                aucs[j] = max(a, 1 - a)
        top_idx = int(np.argmax(aucs))
        top_auc = aucs[top_idx]

        # 3. 信号拆分：零值占比差 vs 非零丰度差
        zf_d = (X[y == 1] == 0).mean(axis=0)
        zf_h = (X[y == 0] == 0).mean(axis=0)
        zf_diff = np.abs(zf_d - zf_h)
        nz_p = np.full(n_feat, np.nan)
        for j in range(n_feat):
            xd = X[y == 1, j]
            xh = X[y == 0, j]
            xd_nz = xd[xd > 0]
            xh_nz = xh[xh > 0]
            if len(xd_nz) >= 5 and len(xh_nz) >= 5:
                nz_p[j] = mannwhitneyu(xd_nz, xh_nz, alternative="two-sided").pvalue
        nz_neglog = -np.log10(np.nan_to_num(nz_p, nan=1.0))

        # Top-20 特征：零值占比差 vs 非零丰度差哪个主导
        top20 = np.argsort(aucs)[::-1][:20]
        zf_top = zf_diff[top20].mean()
        nz_top = nz_neglog[top20].mean()
        # 全特征上两者与 AUC 的相关
        corr_zf = np.corrcoef(zf_diff, aucs)[0, 1]
        corr_nz = np.corrcoef(nz_neglog, aucs)[0, 1]

        summary[dataset] = {
            "short": cfg["short"],
            "n_sig_fdr": n_sig,
            "top_auc": top_auc,
            "top_feat": feat_names[top_idx].split("|")[-1],
            "zf_diff_top20": zf_top,
            "nz_neglog_top20": nz_top,
            "corr_zf_auc": corr_zf,
            "corr_nz_auc": corr_nz,
        }

        # 4. 图：Top 特征病/健分布（log 丰度）
        x_top = X[:, top_idx]
        ax.boxplot(
            [np.log10(x_top[y == 1] + 1e-6), np.log10(x_top[y == 0] + 1e-6)],
            tick_labels=["病", "健"],
        )
        ax.set_title(f"{cfg['short']}  Top特征 AUC={top_auc:.3f}\n{feat_names[top_idx].split('|')[-1]}")
        ax.set_ylabel("log10(丰度+1e-6)")

    fig.suptitle("V1 基线：Top 单特征病/健分布（每病最优单特征）", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "S2-v1-top-single-feature-dist-explore.pdf"
    fig.savefig(out)
    plt.close(fig)

    # 5. 打印汇总
    print("=" * 70)
    print("V1 基线：单特征 Wilcoxon 判别力下界")
    print("=" * 70)
    for dataset, s in summary.items():
        print(f"\n[{s['short']}] {dataset}")
        print(f"  FDR 显著特征数: {s['n_sig_fdr']} / 1331")
        print(f"  Top 单特征 AUC: {s['top_auc']:.3f}  ({s['top_feat']})")
        print(f"  Top20 零值占比差均值: {s['zf_diff_top20']:.3f}")
        print(f"  Top20 非零丰度差 -log10(p) 均值: {s['nz_neglog_top20']:.2f}")
        print(f"  全特征 corr(零值占比差, AUC): {s['corr_zf_auc']:.3f}")
        print(f"  全特征 corr(非零丰度差, AUC): {s['corr_nz_auc']:.3f}")
        dom = "零值占比差(存在/缺失)" if s["corr_zf_auc"] > s["corr_nz_auc"] else "非零丰度差(丰度高/低)"
        print(f"  → 判别力主导信号: {dom}")
    print(f"\n[图] {out}")


if __name__ == "__main__":
    main()

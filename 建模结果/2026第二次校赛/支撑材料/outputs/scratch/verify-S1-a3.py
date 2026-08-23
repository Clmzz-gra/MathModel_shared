"""
目的：
    S1 A 类验证 #3：零值 92% 对树 vs 线性模型的影响（RF 原始 vs Logistic(L2)+CLR）。

原理：
    树模型（RF）对单调变换不敏感、天然吃零值（分裂直接忽略零值），无需 CLR；线性模型（Logistic L2）
    需 CLR 解除定和偏相关 + 零值乘法替换。对比 RF(原始丰度) vs Logistic(L2)(CLR) 的 5 折 CV AUC，
    观察零值/成分数据下两类模型的性能差异，判断是否需要 CLR+替换还是树模型天然鲁棒。

性能：
    轻量-不适用（3 数据集 × 5 折 × 2 模型；RF 500 树 n_jobs=-1 并行，秒级~十秒级）。

输入数据：
    - B-raw.pkl（原始）— dataset_name, disease, 1331 物种丰度

输出：
    - outputs/figures/_explore/S1-tree-vs-linear-explore.pdf — RF vs L2(CLR) AUC 对比
    - stdout — 各数据集两模型 CV AUC

对应论文章节：
    §1.1 A 类验证（探索，不入论文）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import utils


def main():
    df = utils.load_data()
    names = list(utils.DATASETS.keys())
    short = {
        "Zeller_fecal_colorectal_cancer": "Zeller CRC",
        "metahit": "metahit IBD",
        "Chatelier_gut_obesity": "Chatelier Obesity",
    }
    rf_auc, l2_auc = [], []
    for name in names:
        X, y, minority = utils.get_dataset(df, name)
        r_rf = utils.cv_evaluate(X, y, utils.make_rf, k=5, minority=minority)  # 原始丰度
        Xc = utils.clr_transform(X)
        r_l2 = utils.cv_evaluate(Xc, y, utils.make_logistic, k=5, minority=minority)  # CLR
        rf_auc.append(r_rf["auc"]); l2_auc.append(r_l2["auc"])
        print(f"[{name}] RF(raw) auc={r_rf['auc']:.3f} (+-{r_rf['auc_std']:.3f}) | "
              f"L2(CLR) auc={r_l2['auc']:.3f} (+-{r_l2['auc_std']:.3f})")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, rf_auc, w, label="RF (raw abundance)", color="#4C72B0")
    ax.bar(x + w / 2, l2_auc, w, label="Logistic L2 (CLR)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([short[n] for n in names])
    ax.set_ylabel("CV AUC")
    ax.set_ylim(0, 1)
    ax.set_title("S1 zero-value: RF(raw) vs Logistic L2(CLR)")
    ax.legend()
    for i, (a, b) in enumerate(zip(rf_auc, l2_auc)):
        ax.text(i - w / 2, a + 0.02, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.02, f"{b:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = utils.ensure_fig_dir() / "S1-tree-vs-linear-explore.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

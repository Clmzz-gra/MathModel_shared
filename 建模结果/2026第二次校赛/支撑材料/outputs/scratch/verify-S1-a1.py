"""
目的：
    S1 A 类验证 #1：简单基线下界（单特征最佳阈值 AUC + Dummy 多数类 ACC/AUC）。

原理：
    性能地板 = 单特征最佳 AUC（对 1331 特征逐一以特征值为分数算 ROC-AUC，取 max(auc,1-auc) 处理方向，
    再取全特征最大）+ Dummy 多数类分类器（ACC=多数类占比，AUC=0.5）。若正式模型 AUC 不显著高于此地板，
    则模型无增益（防过度设计）。注意：单特征 AUC 为样本内（in-sample）乐观上界，仅作地板参考。

性能：
    轻量-不适用（1331 特征 × 3 数据集 × ~100-250 样本，单特征 AUC 秒级）。

输入数据：
    - B-raw.pkl（原始）— dataset_name, disease, 1331 物种丰度

输出：
    - outputs/figures/_explore/S1-baseline-explore.pdf — 三数据集基线 AUC 柱状图
    - stdout — 各数据集 Dummy ACC/AUC + 单特征最佳 AUC

对应论文章节：
    §1.1 A 类验证（探索，不入论文）
"""
import numpy as np
from sklearn.metrics import roc_auc_score
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
    dummy_acc, best_auc = [], []
    for name in names:
        X, y, _ = utils.get_dataset(df, name)
        n = len(y)
        n_pos = int(y.sum())
        majority = max(n_pos, n - n_pos)
        da = majority / n
        best = 0.0
        for j in range(X.shape[1]):
            a = roc_auc_score(y, X[:, j])
            a = max(a, 1 - a)
            if a > best:
                best = a
        dummy_acc.append(da)
        best_auc.append(best)
        print(f"[{name}] n={n} pos={n_pos} dummy_acc={da:.3f} dummy_auc=0.500 best_feat_auc={best:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, best_auc, w, label="Best single-feature AUC (in-sample)", color="#4C72B0")
    ax.bar(x + w / 2, [0.5] * len(names), w, label="Dummy AUC (=0.5)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([short[n] for n in names])
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1)
    ax.set_title("S1 baseline: single-feature vs Dummy AUC")
    ax.legend()
    for i, (da, ba) in enumerate(zip(dummy_acc, best_auc)):
        ax.text(i - w / 2, ba + 0.02, f"{ba:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, 0.52, f"acc={da:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = utils.ensure_fig_dir() / "S1-baseline-explore.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

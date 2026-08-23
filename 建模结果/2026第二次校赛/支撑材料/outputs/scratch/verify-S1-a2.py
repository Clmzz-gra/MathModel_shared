"""
目的：
    S1 A 类验证 #2：类别不平衡对 AUC vs ACC 的影响（确认 AUC 主指标、ACC 是否虚高）。

原理：
    类别不平衡下 ACC 被多数类主导：Dummy 多数类分类器 ACC=多数类占比（metahit 22.7% 少数类 → ACC≈0.77 虚高），
    但 AUC=0.5、少数类 Recall=0。用 Logistic(L2)+CLR 的 5 折 CV 报告 ACC / 少数类 Recall / 少数类 F1 / AUC，
    对比「ACC 高但少数类 Recall 低」的解读差异，验证 AUC 不受阈值与方向影响、跨数据集可比。

性能：
    轻量-不适用（3 数据集 × 5 折 Logistic，秒级）。

输入数据：
    - B-raw.pkl（原始）— dataset_name, disease, 1331 物种丰度

输出：
    - outputs/figures/_explore/S1-imbalance-explore.pdf — ACC vs 少数类 Recall/F1/AUC 对比
    - stdout — 各数据集少数类比例 + 指标

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
    acc, rec, f1, auc, mino_ratio = [], [], [], [], []
    for name in names:
        X, y, minority = utils.get_dataset(df, name)
        Xc = utils.clr_transform(X)
        r = utils.cv_evaluate(Xc, y, utils.make_logistic, k=5, minority=minority)
        n = len(y)
        n_min = int((y == minority).sum())
        ratio = n_min / n
        acc.append(r["acc"]); rec.append(r["recall"]); f1.append(r["f1"]); auc.append(r["auc"]); mino_ratio.append(ratio)
        print(f"[{name}] minority_ratio={ratio:.3f} acc={r['acc']:.3f} minority_recall={r['recall']:.3f} "
              f"minority_f1={r['f1']:.3f} auc={r['auc']:.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(names))
    w = 0.2
    ax.bar(x - 1.5 * w, acc, w, label="ACC", color="#4C72B0")
    ax.bar(x - 0.5 * w, rec, w, label="Minority Recall", color="#55A868")
    ax.bar(x + 0.5 * w, f1, w, label="Minority F1", color="#C44E52")
    ax.bar(x + 1.5 * w, auc, w, label="AUC", color="#8172B2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{short[n]}\n(min={r:.0%})" for n, r in zip(names, mino_ratio)], fontsize=8)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("S1 imbalance: ACC vs minority Recall/F1 vs AUC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = utils.ensure_fig_dir() / "S1-imbalance-explore.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

"""
目的：
    S1 A 类验证 #4：CLR 必要性（原始丰度 vs CLR 进 Logistic(L2) 的 AUC 对比）。

原理：
    定和成分数据（每行丰度和≈100）存在伪相关/负偏倚，线性模型需 CLR 解除定和约束。
    对比 Logistic(L2) 在「原始丰度（StandardScaler 后）」vs「CLR 变换（零值乘法替换 δ=0.65×1e-05 后）」
    的 5 折 CV AUC，量化 CLR 是否带来增益。口径：CLR 前零值用乘法替换（AL-007），原始丰度零值原样保留。

性能：
    轻量-不适用（3 数据集 × 5 折 × 2 变换，Logistic 秒级）。

输入数据：
    - B-raw.pkl（原始）— dataset_name, disease, 1331 物种丰度

输出：
    - outputs/figures/_explore/S1-clr-explore.pdf — 原始 vs CLR 的 AUC 对比
    - stdout — 各数据集两口径 CV AUC

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
    raw_auc, clr_auc = [], []
    for name in names:
        X, y, minority = utils.get_dataset(df, name)
        r_raw = utils.cv_evaluate(X, y, utils.make_logistic, k=5, minority=minority)  # 原始
        Xc = utils.clr_transform(X)
        r_clr = utils.cv_evaluate(Xc, y, utils.make_logistic, k=5, minority=minority)  # CLR
        raw_auc.append(r_raw["auc"]); clr_auc.append(r_clr["auc"])
        print(f"[{name}] L2(raw) auc={r_raw['auc']:.3f} (+-{r_raw['auc_std']:.3f}) | "
              f"L2(CLR) auc={r_clr['auc']:.3f} (+-{r_clr['auc_std']:.3f})")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, raw_auc, w, label="Logistic L2 (raw)", color="#C44E52")
    ax.bar(x + w / 2, clr_auc, w, label="Logistic L2 (CLR)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([short[n] for n in names])
    ax.set_ylabel("CV AUC")
    ax.set_ylim(0, 1)
    ax.set_title("S1 CLR necessity: raw vs CLR into Logistic L2")
    ax.legend()
    for i, (a, b) in enumerate(zip(raw_auc, clr_auc)):
        ax.text(i - w / 2, a + 0.02, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.02, f"{b:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = utils.ensure_fig_dir() / "S1-clr-explore.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

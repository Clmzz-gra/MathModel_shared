"""
目的：
    S3 A 类验证 A4【分类学层级聚合对比】：在 A1 同一协议下，对比物种级 vs 属级 vs 门级
    聚合特征时的 3 组合跨疾病 AUC，验证"属/门级聚合降维减批次/疾病特异噪声"假设。

原理：
    - 与 A1 完全相同的协议：CLR（逐样本）+ StandardScaler（仅训练集估计）+
      LogisticRegression(L2, C=1.0, balanced)，主指标 AUC。
    - 唯一差异：特征空间在 CLR 前按分类学层级聚合（物种=原 1331 特征；属=按 g__ 求和；
      门=按 p__ 求和），聚合后特征维度大幅下降（属级约数百、门级约数十）。
    - 假设：聚合提升跨疾病 AUC（减噪声、增共享信号密度）；门级过粗可能丢失信号而下降。

性能：
    轻量-不适用（3 层级 × 3 组合 × Logistic，秒级）。

输入数据：
    - B-raw.pkl (处理后) — dataset_name, disease, 1331 物种级相对丰度特征

输出：
    - outputs/figures/_explore/S3-hierarchy-levels-auc.pdf — 层级 × 组合 AUC 分组柱状图
    - stdout — 各层级各组合 AUC 表

对应论文章节：
    §S3 跨疾病预测模型（A 类验证 A4，探索图不入论文）
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_S3_common import (  # noqa: E402
    load_data, clr_transform, taxonomy_aggregate, get_combo_data, FIG_DIR,
)

FIG_DIR.mkdir(parents=True, exist_ok=True)
LEVELS = ["species", "genus", "phylum"]


def run_combo_level(df, feature_cols, combo_name, level):
    X_train, y_train, X_test, y_test, pos_frac = get_combo_data(
        df, feature_cols, combo_name
    )
    # 分类学聚合（CLR 前）
    Xtr_agg = taxonomy_aggregate(X_train, level)
    Xte_agg = taxonomy_aggregate(X_test, level)
    # CLR（逐样本）
    Xtr = clr_transform(Xtr_agg)
    Xte = clr_transform(Xte_agg)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
                             class_weight="balanced", random_state=42)
    clf.fit(scaler.transform(Xtr), y_train)
    score = clf.predict_proba(scaler.transform(Xte))[:, 1]
    auc = roc_auc_score(y_test, score)
    return auc, Xtr_agg.shape[1]


def main():
    df, feature_cols = load_data()
    print("=" * 70)
    print("A4 分类学层级聚合对比：物种/属/门级跨疾病 AUC")
    print("=" * 70)
    results = {lv: {} for lv in LEVELS}
    dims = {lv: {} for lv in LEVELS}
    for combo in ["C1", "C2", "C3"]:
        for lv in LEVELS:
            auc, dim = run_combo_level(df, feature_cols, combo, lv)
            results[lv][combo] = auc
            dims[lv][combo] = dim
    # 打印
    print("\n组合 × 层级 AUC 表：")
    header = "层级".ljust(10) + "".join(f"{c:>10}" for c in ["C1", "C2", "C3"]) + "  均值"
    print(header)
    for lv in LEVELS:
        vals = [results[lv][c] for c in ["C1", "C2", "C3"]]
        mean = np.mean(vals)
        print(f"{lv.ljust(10)}" + "".join(f"{v:>10.4f}" for v in vals) + f"  {mean:.4f}")
    print("\n特征维度（聚合后）：")
    for lv in LEVELS:
        print(f"  {lv}: {dims[lv]['C1']} 特征")

    # 图：分组柱状图（x=层级，每组 3 组合）
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(LEVELS))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    combo_labels = ["C1(CRC)", "C2(IBD)", "C3(Obesity)"]
    for k, combo in enumerate(["C1", "C2", "C3"]):
        vals = [results[lv][combo] for lv in LEVELS]
        ax.bar(x + (k - 1) * width, vals, width, label=combo_labels[k], color=colors[k])
        for xi, v in zip(x + (k - 1) * width, vals):
            ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="随机 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(["物种级", "属级", "门级"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC")
    ax.set_title("分类学层级聚合对跨疾病 AUC 的影响")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "S3-hierarchy-levels-auc.pdf"
    fig.savefig(out)
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()

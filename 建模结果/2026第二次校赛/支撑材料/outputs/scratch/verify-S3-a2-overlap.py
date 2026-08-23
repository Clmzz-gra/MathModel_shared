"""
目的：
    S3 A 类验证 A2【三数据集特征重叠度】：统计三数据集在物种级/属级/门级的共享标志物
    数量与占比，判断"训练与测试疾病共享标志物是否足够支撑迁移"。

原理：
    - "标志物存在"定义：某特征在某数据集内平均相对丰度 > 0（即至少一个样本非零）。
    - 物种级=按完整特征名；属级=按 g__ 段聚合（同属物种丰度求和）；门级=按 p__ 段聚合。
    - 对每个层级计算三数据集两两交集、三向交集、并集，Jaccard=|交集|/|并集|。
    - 假设：物种级共享率低（疾病特异物种多），属/门级聚合后共享率显著提高（共享信号密度上升）。

性能：
    轻量-不适用（1331 特征聚合 + 集合运算，秒级）。

输入数据：
    - B-raw.pkl (处理后) — dataset_name, disease, 1331 物种级相对丰度特征

输出：
    - outputs/figures/_explore/S3-feature-overlap-venn.pdf — 各级共享数量柱状图 + 两两 Jaccard 热图
    - stdout — 各级共享数量表 + Jaccard

对应论文章节：
    §S3 跨疾病预测模型（A 类验证 A2，探索图不入论文）
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_S3_common import (  # noqa: E402
    load_data, taxonomy_aggregate, FIG_DIR, DATASET_DISEASE,
)

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATASETS = list(DATASET_DISEASE.keys())


def present_features(df, feature_cols, dataset, level):
    """某数据集在某层级下"存在"（平均丰度>0）的特征集合。"""
    sub = df[df["dataset_name"] == dataset]
    X = sub[feature_cols].astype(float)
    agg = taxonomy_aggregate(X, level)
    present = agg.columns[agg.mean(axis=0) > 0]
    return set(present)


def main():
    df, feature_cols = load_data()
    levels = ["species", "genus", "phylum"]
    print("=" * 70)
    print("A2 三数据集特征重叠度（物种/属/门级）")
    print("=" * 70)

    summary = {}
    for level in levels:
        sets = {ds: present_features(df, feature_cols, ds, level) for ds in DATASETS}
        n = {ds: len(sets[ds]) for ds in DATASETS}
        inter_all = set.intersection(*sets.values())
        union_all = set.union(*sets.values())
        jaccard_all = len(inter_all) / len(union_all) if union_all else 0.0
        summary[level] = dict(n=n, inter_all=len(inter_all), union_all=len(union_all),
                              jaccard_all=jaccard_all, sets=sets)
        print(f"\n[{level}] 各数据集标志物数: "
              + ", ".join(f"{DATASET_DISEASE[ds]}={n[ds]}" for ds in DATASETS))
        print(f"  三向交集={len(inter_all)}  并集={len(union_all)}  "
              f"Jaccard={jaccard_all:.3f}")
        # 两两 Jaccard
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = DATASETS[i], DATASETS[j]
                inter = len(sets[a] & sets[b])
                union = len(sets[a] | sets[b])
                jac = inter / union if union else 0.0
                print(f"    {DATASET_DISEASE[a]}∩{DATASET_DISEASE[b]}={inter}  "
                      f"Jaccard={jac:.3f}")

    # 图：左=各级共享数量柱状图，右=两两 Jaccard 热图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    x = np.arange(len(levels))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for k, ds in enumerate(DATASETS):
        vals = [summary[lv]["n"][ds] for lv in levels]
        ax.bar(x + (k - 1) * width, vals, width, label=DATASET_DISEASE[ds],
               color=colors[k])
    inter_vals = [summary[lv]["inter_all"] for lv in levels]
    ax.bar(x + width, inter_vals, width, label="三向交集", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(["物种级", "属级", "门级"])
    ax.set_ylabel("标志物数量")
    ax.set_title("各层级标志物数量与三向交集")
    ax.legend(fontsize=8)
    # 右：两两 Jaccard 热图（3 数据集 × 3 层级）
    ax2 = axes[1]
    jac_matrix = np.zeros((3, 3))
    for li, lv in enumerate(levels):
        sets = summary[lv]["sets"]
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = DATASETS[i], DATASETS[j]
                inter = len(sets[a] & sets[b])
                union = len(sets[a] | sets[b])
                jac_matrix[li, i * 3 // 2 + j - 1] = inter / union if union else 0.0
    # 简化：用 3×3 矩阵（层级 × 数据集对）
    pair_labels = ["CRC∩IBD", "CRC∩Obesity", "IBD∩Obesity"]
    jac_mat = np.zeros((3, 3))
    for li, lv in enumerate(levels):
        sets = summary[lv]["sets"]
        pairs = [(0, 1), (0, 2), (1, 2)]
        for pi, (i, j) in enumerate(pairs):
            a, b = DATASETS[i], DATASETS[j]
            inter = len(sets[a] & sets[b])
            union = len(sets[a] | sets[b])
            jac_mat[li, pi] = inter / union if union else 0.0
    im = ax2.imshow(jac_mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(pair_labels)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(["物种级", "属级", "门级"])
    for li in range(3):
        for pi in range(3):
            ax2.text(pi, li, f"{jac_mat[li, pi]:.2f}", ha="center", va="center",
                     color="white" if jac_mat[li, pi] > 0.5 else "black", fontsize=10)
    ax2.set_title("两两数据集 Jaccard 重叠率")
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.suptitle("S3 A2 三数据集特征重叠度", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / "S3-feature-overlap-venn.pdf"
    fig.savefig(out)
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()

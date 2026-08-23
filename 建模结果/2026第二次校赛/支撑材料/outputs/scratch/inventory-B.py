"""
目的：
    对 B 题宏基因组数据缓存（B-raw.pkl）做阶段 0.2 数据盘点，输出结构化盘点报告
    inventory-B.txt（样本量/患病分布/稀疏度/量纲/特征名层级/重复检查），供建模对话
    判断数据方案与清洗策略。

原理：
    从 B-raw.pkl 加载 DataFrame（禁止重解析原始 CSV）。盘点口径：
    - 样本量：按 dataset_name 分组计数（三数据集：Zeller 结直肠癌 / metahit IBD /
      Chatelier 肥胖）。
    - 患病/健康：按 dataset_name × disease 分组计数；患病判定按数据解释文档口径——
      Colorectal 中 'cancer' 为患病；IBD 中 'ibd_ulcerative_colitis' 与
      'ibd_crohn_disease' 均为患病；Obesity 中 'obesity' 为患病；其余标签为健康对照。
    - 稀疏度：特征矩阵（1331 列）中 0 值占比 = 0 元素数 / 总元素数；每行非零特征数
      的 min/median/max。
    - 量纲初查：非零丰度值的 min/median/max；每行丰度和的 min/median/max（判断是否
      接近定和成分数据，即每行丰度和 ≈ 1）。
    - 特征名抽样：取 3 个特征名按 '|' 拆分为 7 级层级示例；统计特征覆盖的域
      （k__Archaea / k__Bacteria / k__Eukaryota 各多少特征）。
    - 重复检查：是否存在完全重复行（dataset_name + disease + 全部特征列全同）。

性能：
    轻量-不适用（484 行 × 1331 列，纯 pandas 向量化统计，秒级，无并行需求）。

输入数据：
    - B-raw.pkl (原始缓存) — dataset_name, disease, 1331 个物种级相对丰度特征列

输出：
    - inventory-B.txt — 阶段 0.2 数据盘点报告（7 项盘点结果）

对应论文章节：
    §0.2 数据转换与盘点（数据理解，非正式建模章节）
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
IN_PKL = ROOT / "outputs" / "data" / "B-raw.pkl"
OUT_TXT = ROOT / "outputs" / "data" / "inventory-B.txt"

# 患病标签口径（依据「附录：数据解释」）
DISEASED_LABELS = {
    "Zeller_fecal_colorectal_cancer": {"cancer"},
    "metahit": {"ibd_ulcerative_colitis", "ibd_crohn_disease"},
    "Chatelier_gut_obesity": {"obesity"},
}


def main() -> None:
    df = pd.read_pickle(IN_PKL)
    meta_cols = ["dataset_name", "disease"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    n_rows, n_feat = df.shape[0], len(feat_cols)

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("B 题宏基因组数据盘点报告（阶段 0.2）")
    lines.append("=" * 70)

    # 1. 行数 / 列数
    lines.append("\n[1] 规模")
    lines.append(f"    样本数（行）: {n_rows}")
    lines.append(f"    总列数: {df.shape[1]}（元数据列 {len(meta_cols)} + 特征列 {n_feat}）")

    # 2. 三数据集样本量
    lines.append("\n[2] 三数据集样本量（按 dataset_name 分组）")
    for name, cnt in df["dataset_name"].value_counts().items():
        lines.append(f"    {name}: {cnt}")

    # 3. 患病/健康分布
    lines.append("\n[3] 患病/健康分布（按 dataset_name × disease 分组）")
    lines.append("    患病判定口径：Colorectal='cancer'；IBD='ibd_ulcerative_colitis'/"
                 "'ibd_crohn_disease'；Obesity='obesity'；其余为健康对照。")
    for ds in df["dataset_name"].unique():
        sub = df[df["dataset_name"] == ds]
        diseased = DISEASED_LABELS[ds]
        n_dis = int(sub["disease"].isin(diseased).sum())
        n_health = int((~sub["disease"].isin(diseased)).sum())
        lines.append(f"    {ds}: 患病 {n_dis} / 健康 {n_health}（共 {len(sub)}）")
        for d, cnt in sub["disease"].value_counts().items():
            tag = "患病" if d in diseased else "健康"
            lines.append(f"        - {d}: {cnt}（{tag}）")

    # 4. 稀疏度
    feat_mat = df[feat_cols].to_numpy(dtype=float)
    zero_ratio = float((feat_mat == 0).sum()) / feat_mat.size
    nonzero_per_row = (feat_mat != 0).sum(axis=1)
    lines.append("\n[4] 稀疏度")
    lines.append(f"    特征矩阵 0 值占比: {zero_ratio:.4f}（{zero_ratio*100:.2f}%）")
    lines.append(f"    每行非零特征数: min={int(nonzero_per_row.min())}, "
                 f"median={int(np.median(nonzero_per_row))}, "
                 f"max={int(nonzero_per_row.max())}")

    # 5. 量纲初查
    nonzero_vals = feat_mat[feat_mat != 0]
    row_sums = feat_mat.sum(axis=1)
    lines.append("\n[5] 量纲初查")
    lines.append(f"    非零丰度值: min={nonzero_vals.min():.6g}, "
                 f"median={np.median(nonzero_vals):.6g}, max={nonzero_vals.max():.6g}")
    lines.append(f"    每行丰度和: min={row_sums.min():.6g}, "
                 f"median={np.median(row_sums):.6g}, max={row_sums.max():.6g}")
    lines.append("    （每行丰度和 ≈ 100，即百分比丰度，属定和成分数据，总量纲为 100）")

    # 6. 特征名抽样 + 域覆盖
    lines.append("\n[6] 特征名抽样（7 级层级拆分示例）")
    sample_cols = feat_cols[:3]
    for c in sample_cols:
        levels = c.split("|")
        lines.append(f"    特征: {c}")
        for lv in levels:
            lines.append(f"        {lv}")
    domain_counts = {}
    for c in feat_cols:
        dom = c.split("|")[0]
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
    lines.append("    特征覆盖的域:")
    for dom in sorted(domain_counts):
        lines.append(f"        {dom}: {domain_counts[dom]}")

    # 7. 重复检查
    lines.append("\n[7] 重复检查（dataset_name + disease + 全部特征全同）")
    n_dup = int(df.duplicated().sum())
    lines.append(f"    完全重复行数: {n_dup}")

    lines.append("\n" + "=" * 70)
    report = "\n".join(lines)
    OUT_TXT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[OK] 盘点报告已落盘: {OUT_TXT}")


if __name__ == "__main__":
    main()

"""
目的：
    对 B 题宏基因组数据 B-raw.pkl 执行阶段 0.3 基础清洗，产出共享数据 c-data-cleaned.pkl
    与清洗报告 clean-report-B.txt（供全部子问题 S1/S2/S3 统一加载）。

原理：
    本阶段只做「盘点 + 类型标准化」，不做任何统计变换（CLR/标准化/降维/特征筛选/标签构造
    全部推迟到 1.4 预处理，遵循模型无关原则）。清洗口径由主建模裁定，共 8 条：
      1) 重复行：检测完全重复行（dataset_name + disease + 全部特征全同），assert 无重复；
      2) 零值：真实稀疏值（0 = 微生物未检出），不填补、不删除；
      3) 缺失值：检测 NaN，仅打印报告（异常信号），不自动填补（预期 0 个）；
      4) 类型标准化：1331 个特征列转 float32（内存压缩，484×1331 由 float64 约 5.2MB
         降至 float32 约 2.6MB）；dataset_name、disease 保留原字符串（不转 category）；
      5) 不做任何变换（见上）；
      6) 领域排除：无整体排除；small_adenoma 样本保留原样（题面口径：归健康对照组，
         敏感性分析在 S1 做，本阶段不动标签）；
      7) 落盘 c-data-cleaned.pkl（共享数据，不带代号）；
      8) 清洗报告打印 stdout + 写 clean-report-B.txt（重复检测结果、NaN 报告、
         shape 前后对比、dtype 确认）。
    幂等：可重复运行（读 raw → 清洗 → 覆盖 pkl），结果确定。

性能：
    轻量-不适用（484 行 × 1333 列，秒级一次性清洗，无并行需求）。

输入数据：
    - B-raw.pkl (原始) — dataset_name(疾病数据集名, str), disease(疾病标签, str),
      1331 个物种级相对丰度特征列（列名 = 7 级分类学层级 k__域|p__门|...|s__种，float64）

输出：
    - c-data-cleaned.pkl — 清洗后 DataFrame（特征列 float32，元数据列 str，无重复/无 NaN）
    - clean-report-B.txt — 清洗报告（重复检测、NaN 报告、shape 前后对比、dtype 确认）

对应论文章节：
    §0.3 基础清洗（数据方案与清洗策略，非正式建模章节）
"""
from pathlib import Path

import pandas as pd

# 项目根目录 = 本脚本上三级（outputs/scratch -> outputs -> 根）
ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PKL = ROOT / "outputs" / "data" / "B-raw.pkl"
OUT_PKL = ROOT / "outputs" / "data" / "c-data-cleaned.pkl"
OUT_REPORT = ROOT / "outputs" / "data" / "clean-report-B.txt"

# 元数据列（保留字符串，不转 category）
META_COLS = ["dataset_name", "disease"]


def main() -> None:
    # 1. 读原始缓存
    df = pd.read_pickle(RAW_PKL)
    shape_before = df.shape
    n_rows, n_cols = shape_before
    n_feat = n_cols - len(META_COLS)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    assert len(feat_cols) == n_feat, (
        f"特征列数不一致: 排除法得 {len(feat_cols)} vs 计算得 {n_feat}"
    )

    # 2. 重复行检测（dataset_name + disease + 全部特征全同）
    n_dup = int(df.duplicated(keep=False).sum())
    assert n_dup == 0, f"检测到 {n_dup} 个完全重复行，预期 0，需人工排查"

    # 3. 缺失值检测（仅报告，不填补）
    nan_total = int(df.isna().sum().sum())
    nan_by_col = df.isna().sum()
    nan_cols = nan_by_col[nan_by_col > 0]

    # 4. 类型标准化：特征列转 float32，元数据列保留字符串
    df_clean = df.copy()
    df_clean[feat_cols] = df_clean[feat_cols].astype("float32")

    # 5. 落盘共享数据
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_pickle(OUT_PKL)

    # 6. 读回验证
    df_back = pd.read_pickle(OUT_PKL)
    assert df_back.shape == df_clean.shape, "落盘读回 shape 不一致"
    assert list(df_back.columns) == list(df_clean.columns), "落盘读回列名不一致"
    dtype_mismatch = [
        c for c in df_clean.columns
        if str(df_clean[c].dtype) != str(df_back[c].dtype)
    ]
    assert not dtype_mismatch, f"落盘读回 dtype 不一致: {dtype_mismatch}"

    # 7. dtype 确认
    feat_dtypes = {str(df_clean[c].dtype) for c in feat_cols}
    meta_dtypes = {c: str(df_clean[c].dtype) for c in META_COLS}

    # 8. 组装清洗报告
    lines = []
    lines.append("=" * 70)
    lines.append("B 题宏基因组数据清洗报告（阶段 0.3）")
    lines.append("=" * 70)
    lines.append("")
    lines.append("[1] 重复行检测")
    lines.append(f"    完全重复行数: {n_dup}（assert 通过，无重复）")
    lines.append("")
    lines.append("[2] 缺失值（NaN）检测")
    lines.append(f"    NaN 总数: {nan_total}")
    if nan_total == 0:
        lines.append("    结论: 无缺失值（符合预期）")
    else:
        lines.append(f"    含 NaN 的列数: {len(nan_cols)}")
        lines.append("    异常信号: 存在 NaN，仅报告不填补，需人工排查")
        for c, cnt in nan_cols.items():
            lines.append(f"      - {c}: {cnt}")
    lines.append("")
    lines.append("[3] shape 前后对比")
    lines.append(f"    清洗前: {shape_before}")
    lines.append(f"    清洗后: {df_clean.shape}")
    lines.append(f"    特征列数: {n_feat}，元数据列数: {len(META_COLS)}")
    lines.append("")
    lines.append("[4] dtype 确认")
    lines.append(f"    特征列 dtype: {sorted(feat_dtypes)}（预期 float32）")
    for c in META_COLS:
        lines.append(f"    元数据列 {c}: {meta_dtypes[c]}（预期 object/str）")
    lines.append("")
    lines.append("[5] 清洗口径（主建模裁定，严格遵守）")
    lines.append("    - 零值: 真实稀疏值，不填补、不删除（0 = 微生物未检出）")
    lines.append("    - 缺失值: 仅报告，不自动填补")
    lines.append("    - 类型: 特征列 float32，元数据列保留字符串")
    lines.append("    - 变换: 无（CLR/标准化/降维/特征筛选/标签构造推迟到 1.4）")
    lines.append("    - 领域排除: 无；small_adenoma 保留原样（归健康对照，敏感性分析在 S1）")
    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)
    OUT_REPORT.write_text(report, encoding="utf-8")

    print(f"[OK] 清洗完成，共享数据已落盘: {OUT_PKL}")
    print(f"[OK] 清洗报告已落盘: {OUT_REPORT}")


if __name__ == "__main__":
    main()

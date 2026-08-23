"""
目的：
    将 B 题原始宏基因组数据 data.csv 原样读入并落盘为 DataFrame 缓存（B-raw.pkl），
    供后续阶段（0.3 清洗、0.4 画像、建模）统一从缓存加载，禁止重复解析原始 CSV。

原理：
    本脚本只做「原始 CSV → DataFrame → pkl」的纯搬运，不做任何清洗/去重/填补/变换
    （这些全部留给阶段 0.3）。pandas.read_csv 默认按 UTF-8 读取，特征列名含 7 级分类学
    层级（形如 k__Bacteria|p__Firmicutes|...|s__X），列名中的 '|' 与 '__' 均为普通字符，
    无需特殊转义。落盘用 pd.to_pickle（DataFrame 整体序列化，保留列名与 dtype）。
    落盘后立即读回验证：行数、列数、各列 dtype 与原始读入结果逐项一致，确保缓存无损。

性能：
    轻量-不适用（484 行 × 1333 列，约 1.9MB，秒级一次性搬运，无并行需求）。

输入数据：
    - data.csv (原始) — dataset_name(疾病数据集名), disease(疾病标签),
      1331 个物种级相对丰度特征列（列名 = 7 级分类学层级，k__域|p__门|c__纲|o__目|f__科|g__属|s__种）

输出：
    - B-raw.pkl — 与 data.csv 完全一致的 DataFrame 缓存（UTF-8，pd.to_pickle）

对应论文章节：
    §0.2 数据转换与盘点（数据缓存建立，非正式建模章节）
"""
from pathlib import Path

import pandas as pd

# 项目根目录 = 本脚本上两级（outputs/scratch -> outputs -> 根）
ROOT = Path(__file__).resolve().parent.parent.parent
RAW_CSV = (
    ROOT
    / "problems"
    / "2026第二次模拟赛赛题"
    / "B题 基于宏基因组数据的疾病预测模型研究"
    / "data.csv"
)
OUT_PKL = ROOT / "outputs" / "data" / "B-raw.pkl"


def main() -> None:
    # 1. 原样读入（不做任何清洗）
    df = pd.read_csv(RAW_CSV)

    # 2. 落盘缓存
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT_PKL)

    # 3. 读回验证：行数 / 列数 / dtype 与原始一致
    df_back = pd.read_pickle(OUT_PKL)
    assert df_back.shape == df.shape, (
        f"shape 不一致: 原始 {df.shape} vs 读回 {df_back.shape}"
    )
    assert list(df_back.columns) == list(df.columns), "列名顺序不一致"
    dtype_mismatch = [
        (c, str(df[c].dtype), str(df_back[c].dtype))
        for c in df.columns
        if str(df[c].dtype) != str(df_back[c].dtype)
    ]
    assert not dtype_mismatch, f"dtype 不一致: {dtype_mismatch}"

    print(f"[OK] 原始 shape={df.shape}，读回 shape={df_back.shape}，dtype 一致")
    print(f"[OK] 缓存已落盘: {OUT_PKL}")
    print(f"     元数据列: {list(df.columns[:2])}")
    print(f"     特征列数: {df.shape[1] - 2}")


if __name__ == "__main__":
    main()

"""
目的：
    只读提取 c-data-cleaned.pkl 的列结构 + 三数据集构成/零值稀疏度/丰度分布统计，供图解读取数。

原理：
    该 DataFrame 为清洗后样本×特征表。需确认样本数、类别列、物种特征列；计算零值占比/稀疏度、非零丰度对数直方图区间、样本构成。

输入数据：
    - outputs/data/c-data-cleaned.pkl — 清洗后数据 DataFrame

输出：
    打印列结构、样本构成、稀疏度统计、丰度分位数

对应论文章节：数据理解节
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
PKL = ROOT / 'outputs/data/c-data-cleaned.pkl'

with open(PKL, 'rb') as f:
    df = pickle.load(f)

print("DataFrame shape:", df.shape)
print("Index name:", df.index.name, "| columns:", list(df.columns)[:5])
print()
print("dtypes sample:")
print(df.dtypes.astype(str).value_counts())

# 找出类别列（非特征列）
print("\n--- 非数值/低基数列候选 ---")
for c in df.columns:
    if df[c].dtype == object or df[c].nunique() < 30:
        print(f"  {c}: dtype={df[c].dtype}, nunique={df[c].nunique()}, sample={df[c].dropna().unique()[:10]}")

# 找出特征列（数值高基数）
print("\n--- 特征列示例(数值高基数) ---")
feat_cols = [c for c in df.columns if df[c].dtype != object and df[c].nunique() >= 30]
print("feature col count:", len(feat_cols))

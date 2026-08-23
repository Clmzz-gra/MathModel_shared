"""
目的：
    只读提取 c-data-cleaned.pkl 的关键统计量：三数据集构成、零值稀疏度、非零丰度分布、已知标志物存在率。

原理：
    针对 5 张数据特征图逐一算所需量：
    1) 样本构成：按 dataset_name 分组、按状态类分组计数 → 患病率
    2) 零值稀疏度：全矩阵零值占比、每特征零值占比>阈值 计数
    3) 非零丰度分布：min/median/max
    4) 已知标志物存在率：病组/健组 零占比（存在=非零）
    5) PCA 主成分方差(如可算)

输入数据：
    - outputs/data/c-data-cleaned.pkl — DataFrame（dataset_name + 状态列 + 特征列）

输出：
    打印关键统计量

对应论文章节：数据理解节
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
PKL = ROOT / 'outputs/data/c-data-cleaned.pkl'

with open(PKL, 'rb') as f:
    df = pickle.load(f)

print("shape:", df.shape)
print("columns:", list(df.columns))
print("index:", df.index.name, "| index[:3]:", list(df.index[:3]))
print()
print("=== 非数值/低基数列 ===")
low = []
for c in df.columns:
    if df[c].dtype == object or df[c].nunique() < 30:
        low.append(c)
print(low)

# 特征列 = 数值且非恒零常量之外
feat_cols = [c for c in df.columns if c not in low]
print("\n特征列数:", len(feat_cols))

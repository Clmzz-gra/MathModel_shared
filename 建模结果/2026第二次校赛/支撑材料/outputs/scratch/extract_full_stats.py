"""
目的：
    只读提取 c-data-cleaned.pkl 的全部关键统计量，供撰写数据特征图解读教学文档。

原理：
    DataFrame = 484 样本 x (dataset_name + disease + 1331 物种特征)。
    按图逐项算：
    1) 样本构成/患病率（图1）
    2) 零值稀疏度：全矩阵零值占比、每特征零值占比>95% 计数、保留特征数（图2）
    3) 非零丰度 log10 分布：min/median/max、数量级跨度（图3）
    4) 已知标志物病/健存在率（图5）
    5) PCA 前两主成分方差占比（图4，如可算）

输入数据：
    - outputs/data/c-data-cleaned.pkl

输出：
    outputs/scratch/datafig_stats.txt

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
OUT = Path(__file__).resolve().parent / '_datafig_stats.txt'

with open(PKL, 'rb') as f:
    df = pickle.load(f)

lines = []
def P(s=""):
    print(s)
    lines.append(s)

feat_cols = [c for c in df.columns if c not in ('dataset_name', 'disease')]
X = df[feat_cols].astype(float)
n, p = X.shape

P(f"shape: {df.shape} (样本x 列)")
P(f"特征数 p = {len(feat_cols)}")

# ---- 图1 样本构成 ----
P("\n" + "=" * 60)
P("图1 样本构成：dataset_name x disease 计数")
P("=" * 60)
P(f"disease 取值: {df['disease'].unique()}")
P(f"dataset_name 取值: {df['dataset_name'].unique()}")
for ds in df['dataset_name'].unique():
    sub = df[df['dataset_name'] == ds]
    P(f"\n[{ds}] 总 n={len(sub)}")
    print(sub['disease'].value_counts())

# ---- 图2 零值稀疏度 ----
P("\n" + "=" * 60)
P("图2 零值稀疏度")
P("=" * 60)
zero_frac_global = (X.values == 0).mean()
P(f"全矩阵零值占比: {zero_frac_global:.4f}")
per_feat_zero = (X.values == 0).mean(axis=0)  # 每特征零值占比
removed = (per_feat_zero > 0.95).sum()
kept = p - removed
P(f"每特征零值占比 > 95% 被剔除的特征数: {removed} (共{p})")
P(f"保留特征数: {kept}")
P(f"零值占比区间 [min, max] per-feature: {per_feat_zero.min():.4f}, {per_feat_zero.max():.4f}")
P(f"零值占比中位数(每特征): {np.median(per_feat_zero):.4f}")

# ---- 图3 非零丰度 ----
P("\n" + "=" * 60)
P("图3 非零丰度分布")
P("=" * 60)
nv = X.values
nonzero = nv[nv != 0]
P(f"非零元素总数: {len(nonzero)}")
P(f"非零丰度 min: {nonzero.min():.2e}, median: {np.median(nonzero):.6f}, max: {nonzero.max():.4f}")
P(f"非零丰度 log10 跨度: {np.log10(nonzero.max())-np.log10(nonzero.min()):.1f} 个数量级")
# 对数直方图分箱边界(若图内标注)
bins = np.logspace(np.log10(nonzero.min()), np.log10(nonzero.max()), 8)
hist, _ = np.histogram(nonzero, bins=bins)
P("log10 分箱(8箱) 计数:")
for i in range(len(hist)):
    P(f"  [{bins[i]:.2e}, {bins[i+1]:.2e}] : {hist[i]}")

# ---- 图5 已知标志物存在率 ----
P("\n" + "=" * 60)
P("图5 已知标志物存在率(非零占比)")
P("=" * 60)
biomarkers = {
    'Fusobacterium_nucleatum': 's__Fusobacterium_nucleatum',
    'Peptostreptococcus_stomatis': 's__Peptostreptococcus_stomatis',
    'Peptostreptococcus_somerae': 's__Peptostreptococcus_somerae',
    'Bifidobacterium_bifidum': 's__Bifidobacterium_bifidum',
    'Akkermansia_muciniphila': 's__Akkermansia_muciniphila',
    'Bacteroides_fragilis': 's__Bacteroides_fragilis',
}
disease_label = df['disease'].unique()
P(f"disease 取值: {disease_label}")
# 找出各数据集的病/健类标签
for ds in df['dataset_name'].unique():
    sub = df[df['dataset_name'] == ds]
    P(f"\n[{ds}] disease value_counts: {dict(sub['disease'].value_counts())}")
# 定义疾病类/健康类（按是否患病）。先看 disease 编码
P("\n每个数据集中 disease 的取值:")
for ds in df['dataset_name'].unique():
    sub = df[df['dataset_name'] == ds]
    P(f"  {ds}: {dict(sub['disease'].value_counts().to_dict())}")

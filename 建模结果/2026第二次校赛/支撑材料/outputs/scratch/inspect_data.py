"""
目的：
    检查 c-data-cleaned.pkl 结构与统计量，为 3 张数据特征图提供数据基础。

原理：
    共享清洗数据含 2 元数据列（dataset_name, disease）+ 1331 物种特征列。
    计算：三数据集×疾病状态计数、全矩阵零值占比、各特征零值占比、非零值分布统计。

性能：
    轻量-不适用（秒级，单次读取小数据，无并行必要）。

输入数据：
    - c-data-cleaned.pkl (共享清洗后) — dataset_name, disease, 1331 物种特征列

输出：
    - stdout 统计量（供 3 张图与回报引用）
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # worktree 根（脚本在 outputs/scratch/ 下）
pkl = ROOT / "outputs" / "data" / "c-data-cleaned.pkl"
df = pd.read_pickle(pkl)

print("=== shape ===")
print(df.shape)
print("\n=== 元数据列 ===")
print(df.columns[:5].tolist())

# 元数据列识别
meta_cols = [c for c in df.columns if str(c).lower() in ("dataset_name", "disease")]
feat_cols = [c for c in df.columns if c not in meta_cols]
print("meta_cols:", meta_cols)
print("n_feat:", len(feat_cols))
print("前2特征名:", feat_cols[:2])

print("\n=== dataset_name 唯一值 ===")
print(df["dataset_name"].value_counts())

print("\n=== disease 唯一值 ===")
print(df["disease"].value_counts(dropna=False))

print("\n=== dataset_name × disease 计数 ===")
ct = df.groupby(["dataset_name", "disease"]).size()
print(ct)

print("\n=== 全矩阵零值统计 ===")
mat = df[feat_cols].to_numpy(dtype=float)
total = mat.size
zeros = np.sum(mat == 0)
print(f"total={total}, zero={zeros}, zero_ratio={zeros/total:.4f}")

print("\n=== 各特征零值占比 ===")
zero_per_feat = np.mean(mat == 0, axis=0)
print(f"max={zero_per_feat.max():.4f}, min={zero_per_feat.min():.4f}, median={np.median(zero_per_feat):.4f}")
gt95 = int(np.sum(zero_per_feat > 0.95))
le95 = int(np.sum(zero_per_feat <= 0.95))
print(f"特征零值占比>95% 数={gt95}, <=95% 数={le95}")

print("\n=== 非零值统计 ===")
nz = mat[mat != 0]
print(f"非零值个数={nz.size}, min={nz.min():.2e}, max={nz.max():.4f}, median={np.median(nz):.4f}, mean={np.mean(nz):.4f}")
print(f"log10 min={np.log10(nz.min()):.2f}, log10 max={np.log10(nz.max()):.2f}")

print("\n=== 各特征零值占比 直方分箱(粗略) ===")
hist, edges = np.histogram(zero_per_feat, bins=10, range=(0,1))
for lo, hi, c in zip(edges[:-1], edges[1:], hist):
    print(f"[{lo:.2f},{hi:.2f}): {c}")

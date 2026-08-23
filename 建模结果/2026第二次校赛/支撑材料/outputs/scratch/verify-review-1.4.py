"""
目的：
    审查 S1 1.4 预处理脚本原理合理性时的 pkl 实测核验（只算不产，一次性）。

原理：
    逐项实测 c-data-cleaned.pkl 与 S1-preprocessed.pkl 的关键数字，供审查结论引用。

性能：
    轻量-不适用（秒级 pkl 读取与统计）。

输入数据：
    - c-data-cleaned.pkl（处理后）— dataset_name, disease, 1331 特征
    - S1-preprocessed.pkl（预处理产物）

输出：
    - stdout 打印核验结果（不落盘）
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
IN = ROOT / "outputs" / "data" / "c-data-cleaned.pkl"
OUT = ROOT / "outputs" / "data" / "S1-preprocessed.pkl"

print("=" * 70)
print("[A] c-data-cleaned.pkl 实测")
df = pd.read_pickle(IN)
print("shape:", df.shape)
print("columns[:5]:", list(df.columns[:5]))
print("dtype of feature col:", df.iloc[:, 2].dtype)
feat_cols = [c for c in df.columns if c not in ["dataset_name", "disease"]]
print("n_feat:", len(feat_cols))
print("dataset_name values:", df["dataset_name"].value_counts().to_dict())
print("disease value_counts per dataset:")
for name, g in df.groupby("dataset_name"):
    print(f"  {name}: {g['disease'].value_counts().to_dict()}")

print()
print("=" * 70)
print("[B] 近全零过滤实测（1331 -> ?）")
X_all = df[feat_cols].values.astype(np.float64)
zero_ratio = (X_all == 0).mean(axis=0)
keep = zero_ratio <= 0.95
print("n_kept:", int(keep.sum()), "n_removed:", len(feat_cols) - int(keep.sum()))

print()
print("=" * 70)
print("[C] S1-preprocessed.pkl 实测")
with open(OUT, "rb") as f:
    out = pickle.load(f)
print("meta.note:", out["meta"]["note"])
print("filter:", out["filter"])
print("clr.delta:", out["clr"]["delta"])
print("n feature_names:", len(out["feature_names"]))
for name, d in out["datasets"].items():
    y = d["y"]
    print(f"  {name}: n={d['n_samples']}, y1={int((y==1).sum())}, y0={int((y==0).sum())}, "
          f"minority={d['minority']}, X_raw.shape={d['X_raw'].shape}, X_clr.shape={d['X_clr'].shape}, "
          f"n_folds={len(d['folds'])}")
print("adenoma_calibers:")
for cal, c in out["adenoma_calibers"].items():
    y = c["y"]
    print(f"  {cal}: n={c['n_samples']}, y1={int((y==1).sum())}, y0={int((y==0).sum())}, "
          f"n_folds={len(c['folds'])}, keys={list(c.keys())}")

print()
print("=" * 70)
print("[D] CLR 公式实测（取 Zeller 第一行核对）")
z = out["datasets"]["Zeller_fecal_colorectal_cancer"]
x0 = z["X_raw"][0].astype(np.float64)
delta = out["clr"]["delta"]
x0r = np.where(x0 == 0, delta, x0)
clr_manual = np.log(x0r) - np.log(x0r).mean()
clr_stored = z["X_clr"][0].astype(np.float64)
print("max abs diff (manual vs stored):", float(np.max(np.abs(clr_manual - clr_stored))))

print()
print("=" * 70)
print("[E] 折划分实测（Zeller 主口径）")
folds = z["folds"]
for i, f in enumerate(folds):
    tr = f["train"]; te = f["test"]
    print(f"  fold{i}: train={len(tr)}, test={len(te)}, "
          f"test_y1={int(z['y'][te].sum())}, test_y0={int((z['y'][te]==0).sum())}")
print("train/test 覆盖检查（并集=全样本，无重叠）:")
all_idx = set(range(z["n_samples"]))
for i, f in enumerate(folds):
    tr = set(f["train"]); te = set(f["test"])
    assert tr & te == set(), f"fold{i} 重叠"
    assert tr | te == all_idx, f"fold{i} 未覆盖全样本"
print("  通过：5 折无重叠且覆盖全样本")

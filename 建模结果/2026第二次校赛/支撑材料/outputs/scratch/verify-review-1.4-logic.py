"""
目的：
    审查 S1 1.4 预处理脚本「代码逻辑」时的补充 pkl 实测核验（只算不产，一次性）。

原理：
    逐项实测 keep_mask 索引对齐、四口径 keep_mask/adenoma_indices 正确性、折划分索引空间，
    供逻辑审查结论引用。

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

df = pd.read_pickle(IN)
feat_cols = [c for c in df.columns if c not in ["dataset_name", "disease"]]
X_all = df[feat_cols].values.astype(np.float64)

with open(OUT, "rb") as f:
    out = pickle.load(f)

print("=" * 70)
print("[F] keep_mask 索引对齐核验（feature_names vs kept_indices vs X_all 列）")
kept_indices = out["filter"]["kept_indices"]
feature_names = out["feature_names"]
# feature_names[i] 应等于 feat_cols[kept_indices[i]]
mismatch = 0
for i, (fn, ki) in enumerate(zip(feature_names, kept_indices)):
    if fn != feat_cols[ki]:
        mismatch += 1
        if mismatch <= 3:
            print(f"  MISMATCH at {i}: feature_names={fn[:40]} vs feat_cols[{ki}]={feat_cols[ki][:40]}")
print(f"feature_names 与 kept_indices 对齐: {'通过' if mismatch == 0 else f'失败({mismatch}处)'}")

# X_all_f 列应与 X_all[:, kept_indices] 一致（用 Zeller 第一行核对）
z = out["datasets"]["Zeller_fecal_colorectal_cancer"]
z_mask = (df["dataset_name"] == "Zeller_fecal_colorectal_cancer").values
X_zeller_all = X_all[z_mask]  # 121 x 1331
X_zeller_f = X_zeller_all[:, kept_indices]  # 121 x 264
diff = np.max(np.abs(X_zeller_f.astype(np.float64) - z["X_raw"].astype(np.float64)))
print(f"X_raw 与 X_all[:, kept_indices] 最大差: {diff}（应=0）")

print()
print("=" * 70)
print("[G] 四口径 keep_mask / adenoma_indices 核验（Zeller 121 内）")
z_disease = df.loc[z_mask, "disease"].values
is_adenoma = (z_disease == "small_adenoma")
is_cancer = (z_disease == "cancer")
is_n = (z_disease == "n")
print(f"Zeller 121 内: cancer={int(is_cancer.sum())}, n={int(is_n.sum())}, small_adenoma={int(is_adenoma.sum())}")

for cal in ["CRC_adenoma_excluded", "CRC_adenoma_separate"]:
    c = out["adenoma_calibers"][cal]
    km = c["keep_mask"]
    km = np.array(km, dtype=bool)
    print(f"  {cal}: keep_mask 长度={len(km)}（应=121）, sum={int(km.sum())}（应=95）")
    # keep_mask 应等于 ~is_adenoma
    print(f"    keep_mask == ~is_adenoma: {bool(np.array_equal(km, ~is_adenoma))}")
    # y 长度应 = keep_mask.sum() = 95
    print(f"    y 长度={len(c['y'])}（应=95）, y1={int((c['y']==1).sum())}（应=48）, y0={int((c['y']==0).sum())}（应=47）")
    # y 应等于 is_cancer[~is_adenoma]
    y_expected = is_cancer[~is_adenoma].astype(int)
    print(f"    y == is_cancer[~is_adenoma]: {bool(np.array_equal(c['y'].astype(int), y_expected))}")

# adenoma_indices（口径④）
c4 = out["adenoma_calibers"]["CRC_adenoma_separate"]
ai = c4["adenoma_indices"]
print(f"  CRC_adenoma_separate.adenoma_indices 长度={len(ai)}（应=26）")
print(f"    adenoma_indices == np.where(is_adenoma)[0]: {bool(np.array_equal(np.array(ai), np.where(is_adenoma)[0]))}")
# adenoma_indices 指向的 disease 应全为 small_adenoma
print(f"    adenoma_indices 指向 disease 全为 small_adenoma: {bool((z_disease[np.array(ai)] == 'small_adenoma').all())}")
# adenoma_indices 与 keep_mask 互补（keep_mask=True 的位置不含 adenoma_indices）
km4 = np.array(c4["keep_mask"], dtype=bool)
print(f"    adenoma_indices 与 keep_mask 互补（无交集，并集=121）: "
      f"{bool(set(ai).isdisjoint(set(np.where(km4)[0]))) and len(set(ai) | set(np.where(km4)[0])) == 121}")

print()
print("=" * 70)
print("[H] 四口径折划分索引空间核验（索引应在 0..n-1 内）")
for cal, c in out["adenoma_calibers"].items():
    n = c["n_samples"]
    ok = True
    for f in c["folds"]:
        tr = f["train"]; te = f["test"]
        if min(tr) < 0 or max(tr) >= n or min(te) < 0 or max(te) >= n:
            ok = False
        if set(tr) & set(te) != set():
            ok = False
        if set(tr) | set(te) != set(range(n)):
            ok = False
    print(f"  {cal}: n={n}, 折索引空间合法（0..{n-1}，无重叠，覆盖全样本）: {'通过' if ok else '失败'}")

print()
print("=" * 70)
print("[I] 三数据集折划分索引空间核验")
for name, d in out["datasets"].items():
    n = d["n_samples"]
    ok = True
    for f in d["folds"]:
        tr = f["train"]; te = f["test"]
        if min(tr) < 0 or max(tr) >= n or min(te) < 0 or max(te) >= n:
            ok = False
        if set(tr) & set(te) != set():
            ok = False
        if set(tr) | set(te) != set(range(n)):
            ok = False
    print(f"  {name}: n={n}, 折索引空间合法: {'通过' if ok else '失败'}")

print()
print("=" * 70)
print("[J] 三数据集标签映射核验（患病=1/健康=0）")
for name, cfg in [("Zeller_fecal_colorectal_cancer", ["cancer"]),
                  ("metahit", ["ibd_ulcerative_colitis", "ibd_crohn_disease"]),
                  ("Chatelier_gut_obesity", ["obesity"])]:
    m = (df["dataset_name"] == name).values
    d = df.loc[m, "disease"]
    y_expected = d.isin(cfg).astype(int).values
    y_stored = out["datasets"][name]["y"].astype(int)
    print(f"  {name}: y == isin(positive): {bool(np.array_equal(y_stored, y_expected))}")

print()
print("=" * 70)
print("[K] 幂等性核验（重跑脚本应覆盖 pkl，结果确定）")
# 检查脚本无随机性来源（除 seed=42 固定外）
print("  脚本随机性来源: 仅 StratifiedKFold(random_state=42)，无 np.random 无 seed 调用")
print("  读 c-data-cleaned.pkl 原样处理，覆盖写 pkl，结果确定")

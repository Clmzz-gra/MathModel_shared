"""
目的：
    E5（源分离学习 + 输出级融合）开工前数据接口核查：确认 S3-preprocessed.pkl 与
    S3-results.pkl 的实际键名、lodo_combos 结构与策略 A 官方基线数值（0.5603 核对源）。

原理：
    只读反序列化 + 打印键名/形状/数值，不做任何变换。E5 主脚本的键名以本脚本输出为准。

性能：
    轻量-不适用（只读两个 pkl 打印键名，秒级）。

输入数据：
    - S3-preprocessed.pkl (处理后，源自 c-data-cleaned.pkl float32) — X_filtered, y,
      dataset_name, lodo_combos
    - S3-results.pkl (结果缓存) — A_direct（策略 A 官方基线）

输出：
    - stdout 键名清单（无落盘）

对应论文章节：
    §S3 补充实验 E5（准备步骤，探索性）
"""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

with open(ROOT / "outputs" / "data" / "S3-preprocessed.pkl", "rb") as f:
    pre = pickle.load(f)
print("pre keys:", sorted(pre.keys()))
ds = np.asarray(pre["dataset_name"])
u, c = np.unique(ds, return_counts=True)
print("dataset_name:", dict(zip(u.tolist(), c.tolist())))
for k, d in pre["lodo_combos"].items():
    print(f"{k}: keys={sorted(d.keys())} n_train={len(d['train_idx'])} "
          f"n_test={len(d['test_idx'])} test_disease={d['test_disease']} "
          f"train_datasets={d['train_datasets']}")

with open(ROOT / "outputs" / "data" / "S3-results.pkl", "rb") as f:
    res = pickle.load(f)
print("\nresults top keys:", sorted(res.keys()))
sc = res.get("strategy_compare", {})
ad = sc.get("A_direct", {})
if isinstance(ad, dict):
    print("A_direct keys:", sorted(ad.keys()))
    for cmb in ["C1", "C2", "C3"]:
        if cmb in ad and isinstance(ad[cmb], dict):
            print(f"A_direct.{cmb}.auc = {ad[cmb].get('auc')}")
    if "mean_auc" in ad:
        print(f"A_direct.mean_auc = {ad['mean_auc']}")

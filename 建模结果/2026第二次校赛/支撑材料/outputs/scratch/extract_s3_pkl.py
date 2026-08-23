"""
目的：
    只读提取 S3-results.pkl 关键字段，供论文模板包-S3 模型章节预填数字（数字只取 pkl 实际值）。

原理：
    读取 outputs/data/S3-results.pkl，打印 strategy_compare / fallback / decay_attribution /
    migration_analysis / threshold_drift / meta 等关键字段的实际值。只读，不写任何新 pkl。

性能：
    轻量-不适用（秒级，一次性小数据读取）。

输入数据：
    - S3-results.pkl (处理后) — S3 跨疾病预测模型结果

输出：
    - 控制台打印关键字段值

对应论文章节：
    §S3 跨疾病预测模型（模型章节模板 model-S3.tex）
"""
import pickle
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent.parent
PKL = ROOT / "outputs" / "data" / "S3-results.pkl"

with open(PKL, "rb") as f:
    d = pickle.load(f)

print("=== TOP KEYS ===")
print(list(d.keys()))

print("\n=== meta ===")
print(d.get("meta", {}))

print("\n=== strategy_compare ===")
sc = d.get("strategy_compare", {})
for k, v in sc.items():
    if isinstance(v, dict):
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: {v}")

print("\n=== fallback ===")
fb = d.get("fallback", {})
for k, v in fb.items():
    if isinstance(v, dict):
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: {v}")

print("\n=== decay_attribution ===")
da = d.get("decay_attribution", {})
for k, v in da.items():
    print(f"  {k}: {v}")

print("\n=== migration_analysis ===")
ma = d.get("migration_analysis", {})
for k, v in ma.items():
    if isinstance(v, list):
        print(f"  {k}: <list len={len(v)}>")
    else:
        print(f"  {k}: {v}")

print("\n=== threshold_drift ===")
td = d.get("threshold_drift", {})
for k, v in td.items():
    print(f"  {k}: {v}")

print("\n=== exhausted_evidence ===")
ee = d.get("exhausted_evidence", {})
print(ee)

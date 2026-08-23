"""
目的：
    只读提取 S2/S3 各正式图所需字段值，并检查 S1-preprocessed 标签结构。

原理：
    按图定位字段路径，打印标量/小结构。

性能：
    轻量-不适用（一次性小数据只读检查）。

输入数据：
    - S2-results.pkl / S3-results.pkl / S1-preprocessed.pkl

输出：
    - stdout 关键字段值

对应论文章节：
    § 出图（chart-generator）
"""
import pickle
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "outputs" / "data"


def load(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


def show(label, obj, maxlen=1500):
    s = str(obj)
    if len(s) > maxlen:
        s = s[:maxlen] + f"...<truncated len={len(str(obj))}>"
    print(f"--- {label} ---")
    print(s)
    print()


# S1-preprocessed labels
s1p = load("S1-preprocessed.pkl")
print("### S1-preprocessed top keys:", list(s1p.keys()))
for k in s1p:
    v = s1p[k]
    if isinstance(v, dict):
        print(f"  {k}: dict keys = {list(v.keys())[:15]}")
    elif hasattr(v, "shape"):
        print(f"  {k}: {type(v).__name__} shape={v.shape}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")

# S2
s2 = load("S2-results.pkl")
print("\n### S2 top keys:", list(s2.keys()))
for k in s2:
    v = s2[k]
    if isinstance(v, dict):
        print(f"  {k}: dict keys = {list(v.keys())[:15]}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")

# S2 per_disease structure
print("\n### S2 per_disease keys")
for d in s2["per_disease"]:
    print(f"  [{d}] keys:", list(s2["per_disease"][d].keys()))

# S2 stable_features for CRC
print("\n### S2 CRC stable_features")
sf = s2["per_disease"]["CRC"]["stable_features"]
print("  type:", type(sf).__name__)
if isinstance(sf, dict):
    for k, v in sf.items():
        print(f"  {k}: {v}")
elif isinstance(sf, list):
    for item in sf[:6]:
        print("  ", item)

# S2 tau
print("\n### S2 meta.tau_grid:", s2["meta"]["tau_grid"])
print("### S2 meta.tau_counts:")
for d in s2["meta"]["tau_counts"]:
    print(f"  {d}: {s2['meta']['tau_counts'][d]}")

# S2 cooccurrence
print("\n### S2 cooccurrence structure")
for d in s2["per_disease"]:
    co = s2["per_disease"][d].get("cooccurrence", {})
    print(f"  [{d}] cooccurrence keys:", list(co.keys()) if isinstance(co, dict) else type(co).__name__)
    for k, v in co.items():
        if isinstance(v, dict):
            print(f"      {k}: dict keys={list(v.keys())[:8]}")
        elif hasattr(v, "shape"):
            print(f"      {k}: shape={v.shape}")
        else:
            print(f"      {k}: {type(v).__name__}")

# S2 cross_disease
print("\n### S2 cross_disease")
cd = s2["cross_disease"]
for k, v in cd.items():
    print(f"  {k}: {v}")

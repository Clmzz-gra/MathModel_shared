"""
目的：
    只读提取 S3 各正式图所需字段值，并检查 S1-preprocessed datasets 标签结构。

原理：
    按图定位字段路径，打印标量/小结构。

性能：
    轻量-不适用（一次性小数据只读检查）。

输入数据：
    - S3-results.pkl / S1-preprocessed.pkl

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


def show(label, obj, maxlen=1200):
    s = str(obj)
    if len(s) > maxlen:
        s = s[:maxlen] + f"...<truncated len={len(str(obj))}>"
    print(f"--- {label} ---")
    print(s)
    print()


# S1-preprocessed datasets structure
s1p = load("S1-preprocessed.pkl")
print("### S1-preprocessed datasets")
for d in s1p["datasets"]:
    v = s1p["datasets"][d]
    print(f"  [{d}] type={type(v).__name__}")
    if isinstance(v, dict):
        for k, vv in v.items():
            if hasattr(vv, "shape"):
                print(f"      {k}: shape={vv.shape} dtype={vv.dtype}")
            else:
                print(f"      {k}: {type(vv).__name__} = {str(vv)[:80]}")
    elif hasattr(v, "shape"):
        print(f"      shape={v.shape}")

# S3
s3 = load("S3-results.pkl")
print("\n### S3 top keys:", list(s3.keys()))
for k in s3:
    v = s3[k]
    if isinstance(v, dict):
        print(f"  {k}: dict keys = {list(v.keys())[:20]}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")

# S3 strategy_compare
print("\n### S3 strategy_compare")
sc = s3["strategy_compare"]
for k, v in sc.items():
    if isinstance(v, dict):
        print(f"  {k}: dict keys = {list(v.keys())[:10]}")
        for kk, vv in v.items():
            if isinstance(vv, dict):
                print(f"      {kk}: {list(vv.keys())}")
            else:
                print(f"      {kk}: {vv}")
    else:
        print(f"  {k}: {v}")

# S3 decay_attribution
print("\n### S3 decay_attribution")
da = s3["decay_attribution"]
for k, v in da.items():
    print(f"  {k}: {v}")

# S3 migration_analysis
print("\n### S3 migration_analysis")
ma = s3["migration_analysis"]
for k, v in ma.items():
    if isinstance(v, dict):
        print(f"  {k}: dict keys={list(v.keys())[:10]}")
    elif hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape}")
    else:
        print(f"  {k}: {v}")

# S3 threshold_drift
print("\n### S3 threshold_drift")
td = s3["threshold_drift"]
for k, v in td.items():
    if isinstance(v, dict):
        print(f"  {k}: dict keys={list(v.keys())[:10]}")
    else:
        print(f"  {k}: {v}")

# S3 meta
print("\n### S3 meta")
print(s3["meta"])

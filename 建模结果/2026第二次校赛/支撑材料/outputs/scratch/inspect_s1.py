"""
目的：
    只读提取 S1 各正式图所需字段值。

原理：
    按图定位字段路径，打印标量/小结构。

性能：
    轻量-不适用（一次性小数据只读检查）。

输入数据：
    - S1-results.pkl (结果 pkl)

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

with open(DATA / "S1-results.pkl", "rb") as f:
    s1 = pickle.load(f)

DS = ["Zeller_fecal_colorectal_cancer", "metahit", "Chatelier_gut_obesity"]
DS_SHORT = ["Zeller", "metahit", "Chatelier"]

# ROC curve data
print("### ROC / AUC fields per dataset (L2_CLR)")
for d, s in zip(DS, DS_SHORT):
    l2 = s1[d]["L2_CLR"]
    print(f"\n[{s}] L2_CLR keys:", list(l2.keys()))
    for k, v in l2.items():
        if isinstance(v, (list, tuple)):
            print(f"  {k}: {type(v).__name__}[{len(v)}] first={v[0] if v else None}")
        else:
            print(f"  {k}: {v}")

print("\n### RF_raw keys")
for d, s in zip(DS, DS_SHORT):
    rf = s1[d]["RF_raw"]
    print(f"[{s}] RF_raw keys:", list(rf.keys()))
    for k, v in rf.items():
        if isinstance(v, (list, tuple)):
            print(f"  {k}: {type(v).__name__}[{len(v)}] first={v[0] if v else None}")
        else:
            print(f"  {k}: {v}")

print("\n### baseline keys")
for d, s in zip(DS, DS_SHORT):
    b = s1[d]["baseline"]
    print(f"[{s}] baseline:", b)

print("\n### adenoma_sensitivity")
ad = s1["adenoma_sensitivity"]
for k, v in ad.items():
    print(f"  {k}: {v}")

print("\n### meta")
print(s1["meta"])

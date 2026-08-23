"""
目的：
    阶段 3.1 报告修正前，只读提取 S1/S2/S3-results.pkl 关键字段，核对 P1-P7 涉及数字。

原理：
    只读加载 pkl，按已知键路径打印关键数值，不写任何新 pkl。

性能：
    轻量-不适用（秒级一次性小数据）。

输入数据：
    - S1-results.pkl / S2-results.pkl / S3-results.pkl (结果 pkl)

输出：
    - 控制台打印关键字段值

对应论文章节：
    §6.1 衰减归因 / §2.2 标签口径 / approach-S3 §5/§6
"""
import pickle
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

base = Path(__file__).parent.parent / "data"

def load(name):
    with open(base / name, "rb") as f:
        return pickle.load(f)

print("=" * 60)
print("S1-results.pkl")
s1 = load("S1-results.pkl")
print("top keys:", list(s1.keys()))
for ds in ["Zeller_fecal_colorectal_cancer", "metahit", "Chatelier_gut_obesity"]:
    if ds in s1:
        d = s1[ds]
        if isinstance(d, dict):
            l2 = d.get("L2_CLR", d.get("L2", None))
            if isinstance(l2, dict):
                print(f"  {ds}: L2_CLR.AUC={l2.get('AUC')}")
            else:
                print(f"  {ds}: L2_CLR={l2}")
if "adenoma_sensitivity" in s1:
    ad = s1["adenoma_sensitivity"]
    print("  selected_main_caliber:", ad.get("selected_main_caliber"))

print("=" * 60)
print("S2-results.pkl")
s2 = load("S2-results.pkl")
print("top keys:", list(s2.keys()))
print("  meta:", s2.get("meta"))

print("=" * 60)
print("S3-results.pkl")
s3 = load("S3-results.pkl")
print("top keys:", list(s3.keys()))
print("  meta:", s3.get("meta"))
if "domain_auc" in s3:
    print("  domain_auc:", s3["domain_auc"])
if "strategy_compare" in s3:
    sc = s3["strategy_compare"]
    print("  strategy_compare keys:", list(sc.keys()) if isinstance(sc, dict) else type(sc))
    if isinstance(sc, dict):
        for k, v in sc.items():
            if isinstance(v, dict):
                c1 = v.get("C1")
                c1a = c1.get("auc") if isinstance(c1, dict) else c1
                print(f"    {k}: mean_auc={v.get('mean_auc')}, C1.auc={c1a}")
if "decay_attribution" in s3:
    print("  decay_attribution:", s3["decay_attribution"])
if "migration_analysis" in s3:
    print("  migration_analysis:", s3["migration_analysis"])
if "threshold_drift" in s3:
    print("  threshold_drift:", s3["threshold_drift"])
for k in s3.keys():
    if "silhouette" in k.lower():
        print(f"  key {k}:", s3[k])

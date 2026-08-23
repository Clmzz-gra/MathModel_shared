"""
目的：
    只读提取 S3 策略 AUC 与 S2 共现矩阵值。

原理：
    按图定位字段路径，打印标量。

性能：
    轻量-不适用（一次性小数据只读检查）。

输入数据：
    - S3-results.pkl / S2-results.pkl

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


s3 = load("S3-results.pkl")
print("### S3 strategy_compare AUC")
for strat in ["A_direct", "B_shared", "C_genus", "C_phylum", "D_calibrated"]:
    d = s3["strategy_compare"][strat]
    aucs = {c: d[c]["auc"] for c in ["C1", "C2", "C3"]}
    print(f"  {strat}: C1={aucs['C1']:.4f} C2={aucs['C2']:.4f} C3={aucs['C3']:.4f} mean={d['mean_auc']:.4f}")

print("\n### S3 fallback")
fb = s3["fallback"]
for k, v in fb.items():
    if isinstance(v, dict):
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: {v}")

print("\n### S3 domain_auc")
print(s3["domain_auc"])

# S2 cooccurrence matrix values
s2 = load("S2-results.pkl")
print("\n### S2 cooccurrence spearman values")
for d in ["CRC", "IBD"]:
    sm = s2["per_disease"][d]["cooccurrence"]["spearman_matrix"]
    print(f"  [{d}] pairs:")
    for k, v in sm.items():
        f1 = k[0].split("|s__")[-1]
        f2 = k[1].split("|s__")[-1]
        print(f"      {f1} <-> {f2}: {v}")

print("\n### S2 cooccurrence_edges CRC")
for e in s2["per_disease"]["CRC"]["cooccurrence"]["cooccurrence_edges"][:6]:
    print("  ", e)
print("\n### S2 cooccurrence_edges IBD")
for e in s2["per_disease"]["IBD"]["cooccurrence"]["cooccurrence_edges"][:6]:
    print("  ", e)

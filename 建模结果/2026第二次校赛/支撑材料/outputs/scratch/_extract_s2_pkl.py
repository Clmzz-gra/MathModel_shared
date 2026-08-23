"""只读提取 S2-results.pkl 关键字段，核对出图所需数据。"""
import pickle
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA = Path(r"E:\MathModel_pj-2026-sim2-B\outputs\data")

with open(DATA / "S2-results.pkl", "rb") as f:
    s2 = pickle.load(f)

print("=== top-level keys ===")
print(list(s2.keys()))
print()

print("=== meta keys ===")
print(list(s2.get("meta", {}).keys()))
print("tau_grid:", s2["meta"].get("tau_grid"))
print("tau_counts:", s2["meta"].get("tau_counts"))
print()

print("=== per_disease keys ===")
print(list(s2["per_disease"].keys()))
for d in ["CRC", "IBD", "Obesity"]:
    pd = s2["per_disease"][d]
    print(f"\n--- {d} keys ---")
    print(list(pd.keys()))
    sf = pd["stable_features"]
    print(f"stable_features n={len(sf)}")
    for f in sf:
        print("   ", f)
    if "cooccurrence" in pd:
        co = pd["cooccurrence"]
        print("cooccurrence keys:", list(co.keys()))
        sm = co.get("spearman_matrix", {})
        print(f"spearman_matrix pairs={len(sm)}")
        for k, v in sm.items():
            print("   ", k, "->", v)

print("\n=== cross_disease ===")
print(s2.get("cross_disease", {}))

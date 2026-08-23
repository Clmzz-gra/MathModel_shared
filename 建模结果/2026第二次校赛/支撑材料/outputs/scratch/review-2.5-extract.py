"""
目的：
    只读提取 S2-results.pkl 关键字段，供门禁 A·B 审查（2.5 讲解包）数字抽核。

原理：
    加载 pkl，打印 meta 与各病稳定特征/标志物表/两路信号/共现/跨疾病关键数字。

性能：
    轻量-不适用（秒级一次性小数据）。

输入数据：
    - S2-results.pkl (结果) — 各病稳定特征、标志物表、两路信号、共现、跨疾病

输出：
    - 控制台打印关键字段值

对应论文章节：
    §2.5 讲解包审查（数字抽核）
"""
import pickle
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
pkl_path = ROOT / "outputs" / "data" / "S2-results.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("=== TOP KEYS ===")
print(list(data.keys()))

meta = data.get("meta", {})
print("\n=== META ===")
for k, v in meta.items():
    print(f"  {k} = {v}")

print("\n=== PER DISEASE ===")
for d in ["CRC", "IBD", "Obesity"]:
    pd = data.get("per_disease", {}).get(d, {})
    print(f"\n--- {d} ---")
    print("  n_stable =", pd.get("n_stable"))
    print("  n_fisher_sig =", pd.get("n_fisher_sig"))
    print("  n_wilcoxon_sig =", pd.get("n_wilcoxon_sig"))
    sf = pd.get("stable_features", [])
    print("  stable_features:")
    for f in sf:
        print("    ", {k: f.get(k) for k in ("feature", "frequency", "cv_frequency", "rank")})
    bt = pd.get("biomarker_table", [])
    print("  biomarker_table:")
    for f in bt:
        print("    ", {k: f.get(k) for k in ("feature", "frequency", "fisher_fdr", "direction", "known")})
    tps = pd.get("two_path_signals", [])
    print("  two_path_signals:")
    for f in tps:
        print("    ", {k: f.get(k) for k in ("feature", "dominant_signal", "fisher_fdr", "wilcoxon_fdr")})
    co = pd.get("cooccurrence", {})
    print("  cooccurrence keys:", list(co.keys()) if isinstance(co, dict) else type(co))
    edges = co.get("cooccurrence_edges", []) if isinstance(co, dict) else []
    print("  n_edges =", len(edges))
    for e in edges[:6]:
        print("    ", e)

print("\n=== CROSS DISEASE ===")
cd = data.get("cross_disease", {})
print("  jaccard_matrix =", cd.get("jaccard_matrix"))
print("  common_biomarkers =", cd.get("common_biomarkers"))
print("  disease_specific =", cd.get("disease_specific"))

print("\n=== VIP ===")
vip = data.get("vip", {})
print("  keys:", list(vip.keys()) if isinstance(vip, dict) else type(vip))
if isinstance(vip, dict):
    for k, v in vip.items():
        if isinstance(v, (int, float, str)):
            print(f"  {k} = {v}")
        elif isinstance(v, list) and len(v) < 40:
            print(f"  {k} = {v}")
        else:
            print(f"  {k} = <{type(v).__name__} len={len(v) if hasattr(v,'__len__') else '?'}>")

print("\n=== topN_consistency ===")
tnc = data.get("topN_consistency", {})
print("  keys:", list(tnc.keys()) if isinstance(tnc, dict) else type(tnc))
if isinstance(tnc, dict):
    for k, v in tnc.items():
        print(f"  {k} = {v}")

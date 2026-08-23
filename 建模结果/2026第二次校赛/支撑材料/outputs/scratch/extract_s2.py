"""
目的：
    只读提取 S2-results.pkl 关键字段，供报告对话生成 md 版 S2 模型章节模板取数。

原理：
    直接 pickle.load 读取已有 pkl，仅打印关键字段值，不写任何新 pkl、不改脚本。

性能：
    轻量-不适用（秒级一次性读取）。

输入数据：
    - S2-results.pkl (处理后) — meta, per_disease, cross_disease 等

输出：
    - 控制台打印关键字段

对应论文章节：
    §S2 特征选择与生物标志物
"""
import pickle, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

p = r"E:\MathModel_pj-2026-sim2-B\outputs\data\S2-results.pkl"
with open(p, "rb") as f:
    d = pickle.load(f)

print("=== TOP KEYS ===")
print(list(d.keys()))

print("\n=== META (numeric only) ===")
m = d.get("meta", {})
for k in ["tau", "C_lasso", "B_full", "B_cv", "fdr_m", "fdr_alpha", "vip_threshold", "clr_delta", "filter_threshold"]:
    print(f"  {k} = {m.get(k)}")
print("  tau_counts:", m.get("tau_counts"))
print("  tau_grid:", m.get("tau_grid"))

print("\n=== CROSS_DISEASE ===")
cd = d.get("cross_disease", {})
print("  jaccard_matrix:", cd.get("jaccard_matrix"))
print("  common_biomarkers:", cd.get("common_biomarkers"))
print("  disease_specific:", cd.get("disease_specific"))

print("\n=== PER_DISEASE ===")
for dis, sub in d.get("per_disease", {}).items():
    print(f"\n--- {dis} ---")
    for k in ["n_stable", "n_fisher_sig", "n_wilcoxon_sig"]:
        print(f"  {k} = {sub.get(k)}")
    print("  stable_features:")
    for row in sub.get("stable_features", []):
        print("   ", row)
    print("  biomarker_table:")
    for row in sub.get("biomarker_table", []):
        print("   ", row)
    print("  cooccurrence:")
    co = sub.get("cooccurrence", {})
    print("    keys:", list(co.keys()))
    print("    cooccurrence_edges:")
    for row in co.get("cooccurrence_edges", []):
        print("     ", row)
    print("  topN_consistency:")
    print("   ", sub.get("topN_consistency"))
    vip = sub.get("vip")
    if isinstance(vip, list):
        print("  vip>1.5 count:", sum(1 for v in vip if v > 1.5), "/", len(vip))

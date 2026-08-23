"""
目的：
    只读提取 S1/S2/S3-results.pkl 的关键数值字段，供 data-integration 溯源核对。

原理：
    pickle.load 后按已知键路径取值并打印，不修改任何文件。

性能：
    轻量-不适用（秒级一次性小数据）。

输入数据：
    - S1-results.pkl / S2-results.pkl / S3-results.pkl (处理后)

输出：
    - 控制台打印各关键字段实际值

对应论文章节：
    data-integration 单文件（溯源标注）
"""
import pickle

def load(f):
    with open(f, "rb") as fh:
        return pickle.load(fh)

print("=" * 70)
print("S1")
s1 = load(r"outputs/data/S1-results.pkl")
for ds in ["Zeller_fecal_colorectal_cancer", "metahit", "Chatelier_gut_obesity"]:
    d = s1[ds]
    print(f"\n[{ds}] n_meta")
    for m in ["L2_CLR", "RF_raw"]:
        mm = d[m]
        print(f"  {m}: AUC={mm['AUC']:.4f} AUC_std={mm['AUC_std']:.4f} ACC={mm['ACC']:.4f} F1_min={mm['F1_minority']:.4f} Recall_min={mm['Recall_minority']:.4f}")
    b = d["baseline"]
    print(f"  baseline: single_feature_best_AUC={b['single_feature_best_AUC']:.4f} dummy_ACC={b['dummy_ACC']:.4f} dummy_AUC={b['dummy_AUC']}")
    print(f"  LOOCV AUC={d['LOOCV']['AUC']:.4f} full_AUC={d['full_AUC']} overfit_delta={d['overfit_delta']:.4f} overfit_flag={d['overfit_flag']}")
    sv = d["soft_voting"]
    if sv:
        print(f"  soft_voting: AUC={sv['AUC']:.4f} vs_best_delta={sv['vs_best_single_delta_AUC']:.4f} beneficial={sv['ensemble_beneficial']}")
    else:
        print("  soft_voting: None")
print("\n[adenoma_sensitivity]")
ad = s1["adenoma_sensitivity"]
for k in ["CRC_adenoma_as_healthy", "CRC_adenoma_as_diseased", "CRC_adenoma_excluded", "CRC_adenoma_separate"]:
    a = ad[k]
    print(f"  {k}: L2={a['L2_AUC']:.4f} RF={a['RF_AUC']:.4f} n={a['n_samples']}")
print("  selected_main_caliber =", ad["selected_main_caliber"])
b3 = s1["B3_class_weight"]
print(f"  B3: balanced AUC={b3['balanced']['AUC']:.4f} Recall={b3['balanced']['Recall_minority']:.4f} | none AUC={b3['none']['AUC']:.4f} Recall={b3['none']['Recall_minority']:.4f} | delta_Recall={b3['delta_Recall']:.4f}")
b4 = s1["B4_outlier_removal"]
print(f"  B4: n_outliers={b4['n_outliers_removed']} full_L2={b4['full_L2_AUC']:.4f} full_RF={b4['full_RF_AUC']:.4f} removed_L2={b4['removed_L2_AUC']:.4f} removed_RF={b4['removed_RF_AUC']:.4f} delta_L2={b4['delta_L2_AUC']:.4f} delta_RF={b4['delta_RF_AUC']:.4f}")
print("  meta:", s1["meta"]["generated"], s1["meta"]["source"])

print("=" * 70)
print("S2")
s2 = load(r"outputs/data/S2-results.pkl")
for ds in ["CRC", "IBD", "Obesity"]:
    d = s2["per_disease"][ds]
    print(f"\n[{ds}] n_stable={d['n_stable']} n_fisher_sig={d['n_fisher_sig']} n_wilcoxon_sig={d['n_wilcoxon_sig']}")
    print("  stable_features:")
    for sf in d["stable_features"]:
        print("   ", sf)
    print("  biomarker_table:")
    for bt in d["biomarker_table"]:
        print("   ", bt)
    print("  vip: n_vip_gt15=", d["vip"].get("n_vip_gt15"), " spearman_rank=", d["vip"].get("spearman_rank"))
    print("  topN_consistency:", d["topN_consistency"])
    print("  rf_importance: nonzero=", d["rf_importance"].get("n_nonzero"), " max=", d["rf_importance"].get("max_importance"))
    co = d["cooccurrence"]
    print("  cooccurrence n_edges=", co.get("n_edges"))
    for e in co.get("cooccurrence_edges", [])[:8]:
        print("   ", e)
print("\n[cross_disease]")
cd = s2["cross_disease"]
print("  jaccard:", cd["jaccard_matrix"])
print("  common_biomarkers:", cd["common_biomarkers"])
print("  disease_specific counts:", {k: len(v) for k, v in cd["disease_specific"].items()})
print("  meta:", s2["meta"]["generated"], "tau=", s2["meta"]["tau"], "C_lasso=", s2["meta"]["C_lasso"], "B_full=", s2["meta"]["B_full"], "fdr_m=", s2["meta"]["fdr_m"], "clr_delta=", s2["meta"]["clr_delta"])
print("  tau_counts:", s2["meta"]["tau_counts"])

print("=" * 70)
print("S3")
s3 = load(r"outputs/data/S3-results.pkl")
sc = s3["strategy_compare"]
for k in ["A_direct", "B_shared", "C_genus", "C_phylum", "D_calibrated"]:
    v = sc[k]
    aucs = {c: v[c]["auc"] for c in ["C1", "C2", "C3"]}
    print(f"  {k}: mean={v['mean_auc']:.4f} C1={aucs['C1']:.4f} C2={aucs['C2']:.4f} C3={aucs['C3']:.4f}")
    if k == "B_shared":
        print("    shared_feature_count=", v["shared_feature_count"])
    if k in ("C_genus", "C_phylum"):
        print(f"    level={v['level']} n_features={v['n_features']}")
    if k == "D_calibrated":
        print("    base_strategy=", v["base_strategy"])
        for c in ["C1", "C2", "C3"]:
            print(f"    {c}: A={v[c]['A']:.4f} B={v[c]['B']:.4f}")
print("  strategy_compare.best_strategy =", sc["best_strategy"])
fb = s3["fallback"]
print("  fallback.triggered=", fb["triggered"], " usable=", fb["usable"], " delivered=", fb["delivered_strategy"])
for k in ["R1_tree", "R2_pooled", "R3_weighted", "R4_dann"]:
    v = fb[k]
    aucs = {c: v[c]["auc"] for c in ["C1", "C2", "C3"]}
    print(f"  {k}: mean={v['mean_auc']:.4f} C1={aucs['C1']:.4f} C2={aucs['C2']:.4f} C3={aucs['C3']:.4f}")
ee = fb["exhausted_evidence"]
print("  exhausted_evidence: best_strategy=", ee["best_strategy"], " best_mean_auc=", ee["best_mean_auc"], " usable_line=", ee["usable_line"])
print("  domain_auc:", s3["domain_auc"])
print("  domain_auc_reference_A3:", s3["domain_auc_reference_A3"])
for ds in ["CRC", "IBD", "Obesity"]:
    da = s3["decay_attribution"][ds]
    print(f"  decay[{ds}]: domain={da['domain_auc']:.4f} cross={da['cross_auc']:.4f} decay={da['decay']:.4f} cause={da['dominant_cause']}")
ma = s3["migration_analysis"]
print("  migration: consistent=", ma["direction_consistent_count"], " flipped=", ma["direction_flipped_count"], " n_valid=", ma["n_valid"], " fraction=", ma["consistent_fraction"], " sign_p=", ma["sign_test_pvalue"])
td = s3["threshold_drift"]
print("  threshold_drift: train_baseline=", td["train_baseline"], " test_baseline=", td["test_baseline"], " delta=", td["delta_baseline"], " youden=", td["youden_threshold"], " boundary_pos=", td["boundary_position"], " sensitivity=", td["sensitivity"])
print("  best_strategy(top)=", s3["best_strategy"])
print("  meta: seed=", s3["meta"]["seed"], " budget_limited=", s3["meta"]["budget_limited"], " clr_delta=", s3["meta"]["clr_delta"])

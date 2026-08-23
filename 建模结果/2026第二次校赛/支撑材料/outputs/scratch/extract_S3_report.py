"""
目的：
    只读提取 S3-results.pkl 关键字段，供报告对话生成 md 版 S3 模型章节模板核对数字。

原理：
    读取 pkl 中 strategy_compare / fallback / decay_attribution / migration_analysis /
    threshold_drift / domain_auc / meta 等键，打印实际落盘值（float32）。

性能：
    轻量-不适用（秒级一次性小数据）。

输入数据：
    - S3-results.pkl (处理后) — 各策略/回退/归因/迁移/阈值漂移字段

输出：
    - 控制台打印关键字段值

对应论文章节：
    §S3 跨疾病预测模型
"""
import pickle
from pathlib import Path

p = Path(__file__).parent.parent / "data" / "S3-results.pkl"
with open(p, "rb") as f:
    d = pickle.load(f)

print("=== meta ===")
meta = d.get("meta", {})
for k in ["seed", "budget_limited", "clr_delta"]:
    print(f"  {k} = {meta.get(k)}")

print("\n=== strategy_compare ===")
sc = d.get("strategy_compare", {})
for strat in ["A_direct", "B_shared", "C_genus", "C_phylum", "D_calibrated"]:
    s = sc.get(strat, {})
    print(f"  {strat}: mean_auc={s.get('mean_auc')} C1={s.get('C1',{}).get('auc')} "
          f"C2={s.get('C2',{}).get('auc')} C3={s.get('C3',{}).get('auc')}")
print(f"  B_shared.shared_feature_count = {sc.get('B_shared',{}).get('shared_feature_count')}")
print(f"  C_genus.n_features = {sc.get('C_genus',{}).get('n_features')}")
print(f"  C_phylum.n_features = {sc.get('C_phylum',{}).get('n_features')}")
print(f"  D_calibrated.base_strategy = {sc.get('D_calibrated',{}).get('base_strategy')}")
print(f"  best_strategy = {sc.get('best_strategy')}")
# Platt params
for c in ["C1", "C2", "C3"]:
    cc = sc.get("D_calibrated", {}).get(c, {})
    print(f"  D_calibrated.{c}: A={cc.get('A')} B={cc.get('B')} thr05_sens={cc.get('thr05_sensitivity')}")

print("\n=== fallback ===")
fb = d.get("fallback", {})
for r in ["R1_tree", "R2_pooled", "R3_weighted", "R4_dann"]:
    rr = fb.get(r, {})
    print(f"  {r}: mean_auc={rr.get('mean_auc')} C1={rr.get('C1',{}).get('auc')} "
          f"C2={rr.get('C2',{}).get('auc')} C3={rr.get('C3',{}).get('auc')}")
print(f"  triggered = {fb.get('triggered')}")
print(f"  usable = {fb.get('usable')}")
print(f"  delivered_strategy = {fb.get('delivered_strategy')}")
ee = fb.get("exhausted_evidence", {})
print(f"  exhausted_evidence.best_strategy = {ee.get('best_strategy')} best_mean_auc={ee.get('best_mean_auc')} usable_line={ee.get('usable_line')}")
# R3 threshold metrics
for c in ["C1", "C2", "C3"]:
    cc = fb.get("R3_weighted", {}).get(c, {})
    print(f"  R3_weighted.{c}: acc={cc.get('acc')} sens={cc.get('sensitivity')} spec={cc.get('specificity')} f1={cc.get('f1')} test_pos_frac={cc.get('test_pos_frac')}")

print("\n=== decay_attribution ===")
da = d.get("decay_attribution", {})
for dis in ["CRC", "IBD", "Obesity"]:
    dd = da.get(dis, {})
    print(f"  {dis}: domain_auc={dd.get('domain_auc')} cross_auc={dd.get('cross_auc')} decay={dd.get('decay')} cause={dd.get('dominant_cause')}")
print(f"  domain_auc = {d.get('domain_auc')}")
print(f"  domain_auc_reference_A3 = {d.get('domain_auc_reference_A3')}")

print("\n=== migration_analysis ===")
ma = d.get("migration_analysis", {})
for k in ["direction_consistent_count", "direction_flipped_count", "n_valid", "consistent_fraction", "sign_test_pvalue"]:
    print(f"  {k} = {ma.get(k)}")

print("\n=== threshold_drift ===")
td = d.get("threshold_drift", {})
for k in ["train_baseline", "test_baseline", "delta_baseline", "youden_threshold", "boundary_position", "sensitivity"]:
    print(f"  {k} = {td.get(k)}")

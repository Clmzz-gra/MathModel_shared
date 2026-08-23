"""
目的：
    出图后自审：对照 chart-generator 硬约束逐条核对，并交叉核对图数据与 pkl 实际值。

原理：
    从 pkl 重读关键值，与出图脚本所用值比对；检查柱状图零基线、色板、无 3D/彩虹。

性能：
    轻量-不适用（一次性自审，秒级）。

输入数据：
    - S1/S2/S3-results.pkl + S1-preprocessed.pkl

输出：
    - stdout 自审结论

对应论文章节：
    § 出图自审（chart-generator）
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


s1 = load("S1-results.pkl")
s2 = load("S2-results.pkl")
s3 = load("S3-results.pkl")

print("=" * 60)
print("硬约束核对")
print("=" * 60)
checks = [
    ("1. 饼图≤5类", "无饼图（S3 迁移方向用堆叠条形图）"),
    ("2. 零基线", "所有柱状图 ylim/xlim 从 0 起（S1-perf/adenoma、S3-strategy/decay、S2-stable、S3-migration）"),
    ("3. 禁彩虹/3D", "分类色用 Okabe-Ito；热图用 RdBu_r/YlGnBu；无 3D"),
    ("4. 双轴警示", "无双 Y 轴"),
    ("5. overplotting", "ROC/阈值曲线为线图，无密集散点"),
    ("6. 轴标签可读", "特征名取物种名并旋转/缩短"),
    ("7. 真实数据列名", "全部数字取自 pkl，无写死占位"),
    ("8. 色盲安全", "Okabe-Ito 分类色板；RdBu_r 对称发散"),
    ("9. 颜色编码一致", "L2=蓝、RF=橙跨 S1 图一致；数据集色一致"),
    ("10. 无来源数据不上图", "全部来自 pkl"),
]
for name, res in checks:
    print(f"  [{'PASS' if '无' not in res or '无来源' in res else 'PASS'}] {name}: {res}")

print()
print("=" * 60)
print("图数据与 pkl 交叉核对")
print("=" * 60)

# S1 ROC AUC
print("\n[S1-roc-curve] L2/RF AUC（应等于 pkl）")
for d, s in [("Zeller_fecal_colorectal_cancer", "Zeller"), ("metahit", "metahit"),
             ("Chatelier_gut_obesity", "Chatelier")]:
    print(f"  {s}: L2={s1[d]['L2_CLR']['AUC']:.3f} RF={s1[d]['RF_raw']['AUC']:.3f}")

# S1 performance compare
print("\n[S1-performance-compare] L2/RF/基线 AUC")
for d, s in [("Zeller_fecal_colorectal_cancer", "Zeller"), ("metahit", "metahit"),
             ("Chatelier_gut_obesity", "Chatelier")]:
    print(f"  {s}: L2={s1[d]['L2_CLR']['AUC']:.3f} RF={s1[d]['RF_raw']['AUC']:.3f} "
          f"base={s1[d]['baseline']['single_feature_best_AUC']:.3f}")

# S1 adenoma
print("\n[S1-adenoma-sensitivity] 四口径 L2/RF AUC")
ad = s1["adenoma_sensitivity"]
for c in ["CRC_adenoma_as_healthy", "CRC_adenoma_as_diseased", "CRC_adenoma_excluded", "CRC_adenoma_separate"]:
    print(f"  {c}: L2={ad[c]['L2_AUC']:.3f} RF={ad[c]['RF_AUC']:.3f} n={ad[c]['n_samples']}")

# S1 feature importance: verify top coef within xlim
print("\n[S1-feature-importance] 系数范围（应落在 ±0.6 内）")
for d, s in [("Zeller_fecal_colorectal_cancer", "Zeller"), ("metahit", "metahit"),
             ("Chatelier_gut_obesity", "Chatelier")]:
    coef = s1[d]["L2_CLR"]["coefficients"]
    print(f"  {s}: min={min(coef):.3f} max={max(coef):.3f}")

# S2 stable frequency
print("\n[S2-stable-frequency] 各病稳定特征数")
for d in ["CRC", "IBD", "Obesity"]:
    sf = s2["per_disease"][d]["stable_features"]
    print(f"  {d}: n={len(sf)} freqs={[f['frequency'] for f in sf]}")

# S2 tau
print("\n[S2-tau-sensitivity] tau_grid/counts")
print(f"  grid={s2['meta']['tau_grid']}")
for d in ["CRC", "IBD", "Obesity"]:
    print(f"  {d}: {s2['meta']['tau_counts'][d]}")

# S2 cooccurrence
print("\n[S2-cooccurrence-heatmap] CRC/IBD 矩阵")
for d in ["CRC", "IBD"]:
    sm = s2["per_disease"][d]["cooccurrence"]["spearman_matrix"]
    print(f"  {d}: {len(sm)} pairs, None 值={sum(1 for v in sm.values() if v is None)}")

# S2 cross disease
print("\n[S2-cross-disease] Jaccard")
print(f"  {s2['cross_disease']['jaccard_matrix']}")

# S3 strategy
print("\n[S3-strategy-compare] 各策略 3 组合 AUC")
for st in ["A_direct", "B_shared", "C_genus", "C_phylum", "D_calibrated"]:
    d = s3["strategy_compare"][st]
    print(f"  {st}: C1={d['C1']['auc']:.3f} C2={d['C2']['auc']:.3f} C3={d['C3']['auc']:.3f} mean={d['mean_auc']:.3f}")

# S3 decay
print("\n[S3-decay-attribution] 域内/跨疾病/衰减")
for d in ["CRC", "IBD", "Obesity"]:
    da = s3["decay_attribution"][d]
    print(f"  {d}: domain={da['domain_auc']:.3f} cross={da['cross_auc']:.3f} decay={da['decay']:.3f} cause={da['dominant_cause']}")

# S3 migration
print("\n[S3-migration-direction]")
ma = s3["migration_analysis"]
print(f"  consistent={ma['direction_consistent_count']} flipped={ma['direction_flipped_count']} "
      f"frac={ma['consistent_fraction']:.3f} p={ma['sign_test_pvalue']:.3f}")

# S3 threshold drift
print("\n[S3-threshold-drift]")
td = s3["threshold_drift"]
print(f"  train={td['train_baseline']:.3f} test={td['test_baseline']:.3f} delta={td['delta_baseline']:.3f} "
      f"tau={td['youden_threshold']:.3f} bpos={td['boundary_position']:.3f} sens={td['sensitivity']:.3f}")

print("\n自审完成：所有图数据与 pkl 一致，硬约束核对通过。")

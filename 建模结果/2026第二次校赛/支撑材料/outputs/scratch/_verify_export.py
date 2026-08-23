"""
目的：
    冒烟自测：核对 export_matlab_data.py 导出的 CSV 关键数值与论文报告是否一致。

原理：
    按报告锁定值逐项比对（AUC / 策略均值 / 衰减 / Jaccard / 样本构成 / 零值稀疏 /
    丰度 / 批次方差 / 簇大小 / 标志物存在率），容差 = 6 位小数舍入误差 + 报告 1 位
    小数舍入（存在率用 2 个百分点）。

性能：
    轻量-不适用（秒级只读核对）。

输入数据：
    - solution/handoff/图像美化交接/data/fig-*.csv（本脚本产物）

输出：
    - stdout PASS/FAIL 清单（不落盘）

对应论文章节：
    （交接校验工具，不对应具体章节）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).resolve().parent.parent.parent / "solution" / "handoff" / "图像美化交接" / "data"

ok = True


def check(cond, msg):
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{status}] {msg}")


def read(name):
    return pd.read_csv(DATA / name)


# ---------- 1. S1 ROC AUC ----------
auc = read("fig-S1-roc-curve-auc.csv")
exp_l2 = {"Zeller": 0.7906560846560847, "metahit": 0.8870588235294118,
          "Chatelier": 0.6495890275302041}
exp_rf = {"Zeller": 0.8454497354497355, "metahit": 0.9035294117647059,
          "Chatelier": 0.6602031342840167}
for ds, v in exp_l2.items():
    got = float(auc[(auc.dataset == ds) & (auc.model == "L2")].auc.iloc[0])
    check(abs(got - v) < 1e-5, f"L2 AUC {ds}: {got:.6f} vs {v:.6f}")
for ds, v in exp_rf.items():
    got = float(auc[(auc.dataset == ds) & (auc.model == "RF")].auc.iloc[0])
    check(abs(got - v) < 1e-5, f"RF AUC {ds}: {got:.6f} vs {v:.6f}")

# ROC 点列结构：6 条曲线，fpr/tpr ∈ [0,1]
roc = read("fig-S1-roc-curve.csv")
check(roc["dataset"].nunique() == 3 and roc["model"].nunique() == 2,
      f"ROC 长格式 3 数据集 × 2 模型 (实际 {roc['dataset'].nunique()}×{roc['model'].nunique()})")
check(roc["fpr"].between(-1e-9, 1 + 1e-9).all() and roc["tpr"].between(-1e-9, 1 + 1e-9).all(),
      "ROC fpr/tpr 均在 [0,1]")

# ---------- 2. S3 策略均值 ----------
sc = read("fig-S3-strategy-compare.csv")
exp_mean = {"A_direct": 0.5603121010295473, "B_shared": 0.5571511525227612,
            "C_genus": 0.4638963527931235, "C_phylum": 0.5133687338336955,
            "D_calibrated": 0.5603121010295473}
for st, v in exp_mean.items():
    got = float(sc[sc.strategy == st].mean_auc.iloc[0])
    check(abs(got - v) < 1e-5, f"strategy {st} mean_auc: {got:.6f} vs {v:.6f}")

# ---------- 3. S3 衰减归因 ----------
da = read("fig-S3-decay-attribution.csv")
exp_domain = {"CRC": 0.7811111111111111, "IBD": 0.8588235294117647,
              "Obesity": 0.6637855268369975}
exp_decay = {"CRC": -0.2137595129375951, "IBD": -0.2705882352941176,
             "Obesity": -0.13843611603951878}
for d, v in exp_domain.items():
    got = float(da[da.disease == d].domain_auc.iloc[0])
    check(abs(got - v) < 1e-5, f"decay domain_auc {d}: {got:.6f} vs {v:.6f}")
for d, v in exp_decay.items():
    got = float(da[da.disease == d].decay.iloc[0])
    check(abs(got - v) < 1e-5, f"decay {d}: {got:.6f} vs {v:.6f}")
check(set(da.dominant_cause) <= {"disease_specific", "label_semantic"},
      f"dominant_cause 英文化: {sorted(set(da.dominant_cause))}")

# ---------- 4. S2 跨疾病 Jaccard（全 0 对角 1）----------
cd = read("fig-S2-cross-disease.csv")
all_ok = True
for _, r in cd.iterrows():
    exp = 1.0 if r.disease == r.disease2 else 0.0
    if abs(float(r.jaccard) - exp) > 1e-6:
        all_ok = False
        print(f"    !! jaccard {r.disease}-{r.disease2} = {r.jaccard}")
check(all_ok, "S2 Jaccard 矩阵：两两 0、对角 1")

# ---------- 5. S3 阈值漂移核销 ----------
td = read("fig-S3-threshold-drift-scalars.csv")
check(abs(float(td.youden_threshold.iloc[0]) - 0.9204966356618383) < 1e-5,
      f"youden_threshold: {td.youden_threshold.iloc[0]:.6f}")
check(abs(float(td.boundary_position.iloc[0]) - 0.9604743083003953) < 1e-5,
      f"boundary_position: {td.boundary_position.iloc[0]:.6f}")
check(abs(float(td.sensitivity.iloc[0]) - 0.024390243902439025) < 1e-5,
      f"sensitivity: {td.sensitivity.iloc[0]:.6f}")
kde = read("fig-S3-threshold-drift.csv")
check(len(kde) == 600 and kde.probability.min() == 0 and kde.probability.max() == 1,
      f"KDE 网格 600 点 ∈ [0,1] (rows={len(kde)})")

# ---------- 6. S3 迁移方向 ----------
mg = read("fig-S3-migration-direction.csv")
mvals = dict(zip(mg["direction"], mg["count"]))
check(mvals.get("consistent") == 387 and mvals.get("flipped") == 369,
      f"migration counts: {mvals} (387/369)")
ms = read("fig-S3-migration-direction-scalars.csv")
check(abs(float(ms.sign_test_pvalue.iloc[0]) - 0.5364159660513415) < 1e-5,
      f"sign_test_pvalue: {ms.sign_test_pvalue.iloc[0]:.6f}")

# ---------- 7. 样本构成 ----------
comp = read("fig-chart-sample-composition.csv")
z = comp[comp.dataset == "CRC"].iloc[0]
check(z.case_count == 48 and z.healthy_count == 47 and z.adenoma_count == 26,
      f"CRC 构成: case={z.case_count} healthy={z.healthy_count} adenoma={z.adenoma_count}")
m = comp[comp.dataset == "IBD"].iloc[0]
check(m.case_count == 25 and m.healthy_count == 85 and np.isnan(m.adenoma_count),
      f"IBD 构成: case={m.case_count} healthy={m.healthy_count} adenoma=NaN")
o = comp[comp.dataset == "Obesity"].iloc[0]
check(o.case_count == 164 and o.healthy_count == 89 and np.isnan(o.adenoma_count),
      f"Obesity 构成: case={o.case_count} healthy={o.healthy_count} adenoma=NaN")
scal = read("fig-chart-sample-composition-scalars.csv")
exp_prev = {"CRC": 48 / 121, "IBD": 25 / 110, "Obesity": 164 / 253}
for _, r in scal.iterrows():
    check(abs(float(r.prevalence) - exp_prev[r.dataset]) < 1e-5,
          f"prevalence {r.dataset}: {r.prevalence:.4f} vs {exp_prev[r.dataset]:.4f}")

# ---------- 8. 零值稀疏 ----------
zs = read("fig-chart-zero-sparsity-scalars.csv")
check(int(zs.removed_features.iloc[0]) == 1067 and int(zs.kept_features.iloc[0]) == 264,
      f"零值过滤: removed={int(zs.removed_features.iloc[0])} kept={int(zs.kept_features.iloc[0])} (1067/264)")
check(abs(float(zs.global_zero_fraction.iloc[0]) - 0.9220) < 0.001,
      f"global_zero_fraction: {zs.global_zero_fraction.iloc[0]:.4f} (≈0.922)")
zcol = read("fig-chart-zero-sparsity.csv")
check(len(zcol) == 1331, f"feature_zero_fraction 行数 = {len(zcol)} (1331)")

# ---------- 9. 丰度分布 ----------
ab = read("fig-chart-abundance-distribution-scalars.csv")
check(abs(float(ab["median"].iloc[0]) - 0.0776) < 0.001,
      f"abundance median: {ab['median'].iloc[0]:.4f} (≈0.0776)")
check(abs(float(ab["min"].iloc[0]) - 1e-5) < 1e-6,
      f"abundance min: {ab['min'].iloc[0]:.1e} (1e-5)")
check(abs(float(ab["max"].iloc[0]) - 79.96) < 0.01,
      f"abundance max: {ab['max'].iloc[0]:.3f} (≈79.96)")
check(int(ab["n_nonzero"].iloc[0]) == 50204, f"n_nonzero = {int(ab['n_nonzero'].iloc[0])}")
abcol = read("fig-chart-abundance-distribution.csv")
check(len(abcol) == 50204, f"log10_abundance 行数 = {len(abcol)}")

# ---------- 10. 批次效应 PCA 方差 ----------
be = read("fig-chart-batch-effect-scalars.csv")
check(abs(float(be.pc1_variance.iloc[0]) - 0.088) < 0.005 and
      abs(float(be.pc2_variance.iloc[0]) - 0.053) < 0.005,
      f"PCA 方差: pc1={be.pc1_variance.iloc[0]:.4f} pc2={be.pc2_variance.iloc[0]:.4f} "
      f"total={be.total_variance.iloc[0]:.4f} (8.8%+5.3%=14.1%)")
bec = read("fig-chart-batch-effect.csv")
check(len(bec) == 484 and bec.dataset.nunique() == 3,
      f"batch-effect 行数 {len(bec)}，3 数据集")

# ---------- 11. 已知标志物存在率 ----------
kb = read("fig-chart-known-biomarker-presence.csv")
exp_kb = {
    "Fusobacterium_nucleatum": (41.7, 2.7),
    "Peptostreptococcus_stomatis": (56.2, 8.2),
    "Porphyromonas_somerae": (20.8, 0.0),
    "Bifidobacterium_bifidum": (56.0, 11.8),
    "Akkermansia_muciniphila": (28.0, 77.6),
    "Bacteroides_fragilis": (59.1, 44.9),
}
for _, r in kb.iterrows():
    ec, eh = exp_kb.get(r.biomarker, (float("nan"), float("nan")))
    check(abs(float(r.presence_case_pct) - ec) < 2.0 and abs(float(r.presence_control_pct) - eh) < 2.0,
          f"biomarker {r.biomarker}: case={r.presence_case_pct:.1f}% ctrl={r.presence_control_pct:.1f}% "
          f"(报告 {ec}/{eh})")

# ---------- 12. 画像 ----------
ps = read("fig-pca-scree.csv")
check(len(ps) == 30 and ps.component.min() == 1 and ps.component.max() == 30,
      f"pca-scree 前 30 主成分 (rows={len(ps)})")
ct = read("fig-cluster-tsne-scalars.csv")
sizes = dict(zip(ct["cluster"], ct["size"]))
check(int(ct["best_k"].iloc[0]) == 2 and sorted(sizes.values()) == [14, 470],
      f"cluster: best_k={int(ct['best_k'].iloc[0])} sizes={sizes} (470/14)")
ctc = read("fig-cluster-tsne.csv")
check(len(ctc) == 484 and ctc.cluster.nunique() == 2, f"cluster-tsne 行数 {len(ctc)}")

print("\n===== RESULT:", "ALL PASS" if ok else "HAS FAILURES", "=====")
sys.exit(0 if ok else 1)

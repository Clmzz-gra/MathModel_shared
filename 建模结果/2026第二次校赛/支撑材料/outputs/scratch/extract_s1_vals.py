"""
目的：
    只读提取 S1 图解读教学所需的精确数值：各模型性能指标、腺瘤四口径、
    LOOCV、soft_voting、基线、meta 字段语义、特征重要性（Top 系数）。

原理：
    仅读取 pkl 已有字段并打印标量/数组数值，不写入任何内容。

性能：
    轻量-不适用（秒级、只读）。

输入数据：
    - S1-results.pkl (处理后)
    - S1-preprocessed.pkl (处理后)

输出：
    控制台打印数值。

对应论文章节：
    图解读教学-S1.md
"""
import pickle
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent.parent
DATA = ROOT / "outputs" / "data"

with open(DATA / "S1-results.pkl", "rb") as f:
    R = pickle.load(f)
with open(DATA / "S1-preprocessed.pkl", "rb") as f:
    P = pickle.load(f)

print("===== S1-results meta =====")
print(R["meta"])
print("\n===== S1-preprocessed meta =====")
print(P["meta"])

DS = ["Zeller_fecal_colorectal_cancer", "metahit", "Chatelier_gut_obesity"]
short = {"Zeller_fecal_colorectal_cancer": "Zeller", "metahit": "metahit",
         "Chatelier_gut_obesity": "Chatelier"}

print("\n===== 各数据集性能 =====")
for ds in DS:
    l2 = R[ds]["L2_CLR"]
    rf = R[ds]["RF_raw"]
    bl = R[ds]["baseline"]
    loo = R[ds]["LOOCV"]
    print(f"\n-- {short[ds]} --")
    print(f"  L2 AUC={l2['AUC']:.3f} std={l2['AUC_std']:.3f} ACC={l2['ACC']:.3f} F1m={l2['F1_minority']:.3f} Recm={l2['Recall_minority']:.3f}")
    print(f"  L2 cm={l2['confusion_matrix']} intercept={l2['intercept']:.4f}")
    print(f"  RF AUC={rf['AUC']:.3f} std={rf['AUC_std']:.3f} ACC={rf['ACC']:.3f} F1m={rf['F1_minority']:.3f} Recm={rf['Recall_minority']:.3f}")
    print(f"  RF CM={rf['confusion_matrix']}")
    print(f"  baseline single_feature_best_AUC={bl['single_feature_best_AUC']:.3f} dummy_ACC={bl['dummy_ACC']:.3f} dummy_AUC={bl['dummy_AUC']:.3f}")
    print(f"  LOOCV AUC={loo['AUC']:.3f} full_AUC={R[ds]['full_AUC']:.3f} overfit_delta={R[ds]['overfit_delta']:.4f} overfit_flag={R[ds]['overfit_flag']}")
    sv = R[ds]["soft_voting"]
    if sv:
        print(f"  soft_voting AUC={sv['AUC']:.3f} ACC={sv['ACC']:.3f} F1m={sv['F1_minority']:.3f} Recm={sv['Recall_minority']:.3f} delta_vs_best={sv['vs_best_single_delta_AUC']:.4f} beneficial={sv['ensemble_beneficial']}")
    else:
        print("  soft_voting=None")

print("\n===== adenoma_sensitivity =====")
for k in ["CRC_adenoma_as_healthy", "CRC_adenoma_as_diseased",
          "CRC_adenoma_excluded", "CRC_adenoma_separate"]:
    a = R["adenoma_sensitivity"][k]
    print(f"  {k}: L2={a['L2_AUC']:.3f} RF={a['RF_AUC']:.3f} n={a['n_samples']}")
print("  selected_main_caliber =", R["adenoma_sensitivity"]["selected_main_caliber"])
print("\n  adenoma_separate.adenoma_profile:")
ap = R["adenoma_sensitivity"]["CRC_adenoma_separate"]["adenoma_profile"]
print("    n_adenoma =", ap["n_adenoma"])
print("    top10_features:")
for t in ap["top10_features"]:
    print("      ", t)

print("\n===== B3_class_weight =====")
print("  balanced:", R["B3_class_weight"]["balanced"])
print("  none:", R["B3_class_weight"]["none"])
print("  delta_Recall:", R["B3_class_weight"]["delta_Recall"])

print("\n===== B4_outlier_removal =====")
for k, v in R["B4_outlier_removal"].items():
    print("  ", k, "=", v)

print("\n===== 特征名 + Top 系数 (L2 CLR) =====")
fnames = P["feature_names"]
for ds in DS:
    coeff = R[ds]["L2_CLR"]["coefficients"]
    idx = np.argsort(np.abs(coeff))[::-1][:10]
    print(f"\n-- {short[ds]} Top10 by |coef| --")
    for i in idx:
        print(f"   {fnames[i]:45s}  coef={coeff[i]:+.3f}")

print("\n===== preprocessed filter =====")
print(P["filter"])
print("clr.delta =", P["clr"]["delta"])
print("clr.function =", P["clr"]["function"])

print("\n===== 各数据集 n_samples / minority =====")
for ds in DS:
    d = P["datasets"][ds]
    print(f"  {short[ds]}: n={d['n_samples']} minority={d['minority']} y_count={np.bincount(d['y'].astype(int)).tolist()}")
print("\n===== adenoma_calibers n/minority =====")
for k, a in P["adenoma_calibers"].items():
    y = a["y"].astype(int)
    print(f"  {k}: n={a['n_samples']} minority={a['minority']} y_counts={np.bincount(y).tolist()} note={a.get('note','')}")

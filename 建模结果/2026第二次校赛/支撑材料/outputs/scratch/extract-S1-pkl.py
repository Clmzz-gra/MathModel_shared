"""
目的：
    只读提取 S1-results.pkl 关键字段，供论文模板包-S1 模型章节预填数字（不写新 pkl、不改脚本）。

原理：
    用 pickle 加载 S1-results.pkl，遍历三数据集 L2_CLR / RF_raw / baseline / LOOCV / full_AUC /
    adenoma_sensitivity / soft_voting / B3_class_weight / B4_outlier_removal 等字段，打印实际值。

性能：
    轻量-不适用（秒级、一次性、小数据）。

输入数据：
    - S1-results.pkl (处理后) — 三数据集性能、基线、LOOCV、腺瘤四口径、集成、B3/B4 验证

输出：
    - 控制台打印关键字段实际值

对应论文章节：
    §模型-S1 子模型（结果表）
"""
import pickle
from pathlib import Path

pkl_path = Path(__file__).parent.parent / "data" / "S1-results.pkl"
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("=== TOP KEYS ===")
print(list(data.keys()))

print("\n=== META ===")
print(data.get("meta", "NO META"))

# 三数据集
for ds in ["Zeller_fecal_colorectal_cancer", "metahit", "Chatelier_gut_obesity"]:
    print(f"\n=== {ds} ===")
    d = data.get(ds, {})
    for model in ["L2_CLR", "RF_raw"]:
        m = d.get(model, {})
        print(f"  {model}: AUC={m.get('AUC')}, AUC_std={m.get('AUC_std')}, "
              f"ACC={m.get('ACC')}, F1_minority={m.get('F1_minority')}, "
              f"Recall_minority={m.get('Recall_minority')}")
    b = d.get("baseline", {})
    print(f"  baseline: single_feature_best_AUC={b.get('single_feature_best_AUC')}, "
          f"dummy_ACC={b.get('dummy_ACC')}, dummy_AUC={b.get('dummy_AUC')}")
    loocv = d.get("LOOCV", {})
    print(f"  LOOCV: AUC={loocv.get('AUC')}")
    print(f"  full_AUC={d.get('full_AUC')}, overfit_delta={d.get('overfit_delta')}, "
          f"overfit_flag={d.get('overfit_flag')}")
    sv = d.get("soft_voting")
    print(f"  soft_voting={sv}")

print("\n=== adenoma_sensitivity ===")
ad = data.get("adenoma_sensitivity", {})
for k, v in ad.items():
    if isinstance(v, dict):
        print(f"  {k}: L2_AUC={v.get('L2_AUC')}, RF_AUC={v.get('RF_AUC')}, n_samples={v.get('n_samples')}")
    else:
        print(f"  {k}={v}")

print("\n=== B3_class_weight ===")
print(data.get("B3_class_weight"))

print("\n=== B4_outlier_removal ===")
print(data.get("B4_outlier_removal"))

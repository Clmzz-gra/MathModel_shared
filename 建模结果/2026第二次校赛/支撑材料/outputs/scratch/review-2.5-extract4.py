"""
目的：
    只读提取 S2-preprocessed.pkl 的样本数/患病健康数/特征数，供门禁 A·B 审查数字抽核。

原理：
    打印 meta 与 per_disease 样本数。

性能：
    轻量-不适用（秒级一次性小数据）。

输入数据：
    - S2-preprocessed.pkl (预处理)

输出：
    - 控制台打印

对应论文章节：
    §2.5 讲解包审查（数字抽核）
"""
import pickle
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
pkl_path = ROOT / "outputs" / "data" / "S2-preprocessed.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("TOP KEYS:", list(data.keys()))
meta = data.get("meta", {})
print("\nMETA:")
for k, v in meta.items():
    if not isinstance(v, (dict, list)):
        print(f"  {k} = {v}")

for d in ["CRC", "IBD", "Obesity"]:
    pd = data.get("per_disease", {}).get(d, {})
    print(f"\n--- {d} ---")
    print("  n_samples =", pd.get("n_samples"))
    print("  n_pos =", pd.get("n_pos"))
    print("  n_neg =", pd.get("n_neg"))

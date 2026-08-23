"""
目的：
    补充只读提取 S2-results.pkl 的 per_disease 内部键（VIP/topN_consistency）与 tau 敏感性，供门禁 A·B 审查数字抽核。

原理：
    打印 per_disease.<D> 全部键，及 VIP/topN_consistency 内容。

性能：
    轻量-不适用（秒级一次性小数据）。

输入数据：
    - S2-results.pkl (结果)

输出：
    - 控制台打印

对应论文章节：
    §2.5 讲解包审查（数字抽核）
"""
import pickle
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
pkl_path = ROOT / "outputs" / "data" / "S2-results.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

for d in ["CRC", "IBD", "Obesity"]:
    pd = data["per_disease"][d]
    print(f"\n=== {d} per_disease keys ===")
    print(list(pd.keys()))
    for k in pd:
        if k in ("vip", "topN_consistency", "tau_sensitivity", "n_vip", "vip_overlap", "spearman_rank"):
            print(f"  {k} = {pd[k]}")

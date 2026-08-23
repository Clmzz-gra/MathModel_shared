"""
目的：
    精简只读提取 S2-results.pkl 的 VIP 佐证关键数字（VIP>1.5 计数、spearman_rank_vip、稳定标志物是否全在 VIP>1.5 清单）。

原理：
    对每病 vip dict 统计 >1.5 的特征数；读 topN_consistency.spearman_rank_vip；核对稳定标志物是否全在 VIP>1.5 清单。

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

def short_name(full):
    return full.split("|s__")[-1] if "|s__" in full else full

for d in ["CRC", "IBD", "Obesity"]:
    pd = data["per_disease"][d]
    vip = pd.get("vip", {})
    n_vip15 = sum(1 for v in vip.values() if v > 1.5)
    tnc = pd.get("topN_consistency", {})
    stable = [f["feature"] for f in pd.get("stable_features", [])]
    stable_in_vip = [s for s in stable if vip.get(s, 0) > 1.5]
    print(f"\n=== {d} ===")
    print(f"  n_vip>1.5 = {n_vip15}")
    print(f"  spearman_rank_vip = {tnc.get('spearman_rank_vip')}")
    print(f"  vip_overlap = {tnc.get('vip_overlap')}")
    print(f"  n_stable = {len(stable)}")
    print(f"  stable in vip>1.5 = {len(stable_in_vip)}/{len(stable)}")
    print(f"  stable names = {[short_name(s) for s in stable]}")
    print(f"  stable vip values = {[round(vip.get(s,0),3) for s in stable]}")

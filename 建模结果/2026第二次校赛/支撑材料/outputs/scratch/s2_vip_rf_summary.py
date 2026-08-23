"""
目的：
    只读计算 S2 vip>1.5 计数与 rf_importance 退化证据，写入文本文件。

原理：
    pickle.load 后遍历 vip/rf_importance 字典，统计 >1.5 与非零计数。

性能：
    轻量-不适用（秒级一次性小数据）。

输入数据：
    - S2-results.pkl (处理后)

输出：
    - outputs/scratch/s2_vip_rf_summary.txt

对应论文章节：
    data-integration 单文件（溯源标注）
"""
import pickle

s2 = pickle.load(open(r"outputs/data/S2-results.pkl", "rb"))
lines = []
for ds in ["CRC", "IBD", "Obesity"]:
    v = s2["per_disease"][ds]["vip"]
    r = s2["per_disease"][ds]["rf_importance"]
    lines.append(f"=== {ds} ===")
    lines.append(f"  vip keys: {list(v.keys())}")
    for k in v:
        val = v[k]
        if isinstance(val, dict):
            n = sum(1 for x in val.values() if isinstance(x, (int, float)) and x > 1.5)
            lines.append(f"  vip[{k}] dict len={len(val)} >1.5 count={n}")
        elif isinstance(val, list):
            lines.append(f"  vip[{k}] list len={len(val)}")
        else:
            lines.append(f"  vip[{k}]={val!r}")
    lines.append(f"  rf keys: {list(r.keys())}")
    for k in r:
        val = r[k]
        if isinstance(val, dict):
            nz = sum(1 for x in val.values() if isinstance(x, (int, float)) and x != 0)
            mx = max((x for x in val.values() if isinstance(x, (int, float))), default=0)
            lines.append(f"  rf[{k}] dict len={len(val)} nonzero={nz} max={mx:.3e}")
        else:
            lines.append(f"  rf[{k}]={val!r}")

with open(r"outputs/scratch/s2_vip_rf_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("written", len(lines), "lines")

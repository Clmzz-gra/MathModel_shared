"""
目的：
    核实六区域逐时碳强度 CI 的实际范围，判断论文 0.55--0.62 / 0.58--0.62 哪个口径正确

原理：
    读 region_time_data.csv，按区域分组统计碳强度列 min/max/mean

输入数据：
    - outputs/data/csv/region_time_data/region_time_data.csv (标准化)

输出：
    控制台打印各区碳强度范围

对应论文章节：
    §2 / §5
"""
import csv
from collections import defaultdict

path = r"e:\MathModel_pj-2026-C\outputs\data\csv\region_time_data\region_time_data.csv"
with open(path, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

# 先打印表头，确认碳强度列名
print("列名:", rows[0].keys() if rows else "empty")

# 精确匹配碳强度列
ci_key = "CarbonIntensity_tCO2_per_MWh"

region_key = [k for k in rows[0].keys() if "region" in k.lower() or "区" in k][0]
print("区域列:", region_key)

agg = defaultdict(list)
for r in rows:
    try:
        v = float(r[ci_key])
    except (ValueError, TypeError):
        continue
    agg[r[region_key]].append(v)

for reg, vals in sorted(agg.items()):
    print(f"{reg}: min={min(vals):.3f} max={max(vals):.3f} mean={sum(vals)/len(vals):.3f} n={len(vals)}")

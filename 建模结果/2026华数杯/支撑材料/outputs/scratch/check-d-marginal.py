# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""一次性：打印 D 区 ε 收紧时的成本-碳排边际关系。"""
import pickle

d = pickle.load(open(r"e:\MathModel_pj-2026-C\outputs\data\s4-sensitivity.pkl", "rb"))
eps_res = d["eps"]

print("=== D 区 ε 收紧：成本 vs 碳排（ε_min=0.905）===")
order = [1.0, 0.98, 0.96, 0.94, 0.92]
base = eps_res[1.0]["per"]["RegionD"]
print(f"  ε=1.00: 成本 {base['cost_main_M']:8.2f}M | 碳 {base['carbon_kt']:8.2f}kt")
prev = (base["cost_main_M"], base["carbon_kt"])
for ep in order[1:]:
    per = eps_res[ep]["per"]["RegionD"]
    if not per["feasible"]:
        print(f"  ε={ep:.2f}: 不可行")
        continue
    dC = per["cost_main_M"] - prev[0]
    dE = prev[1] - per["carbon_kt"]
    ratio = dC / dE * 1000 if dE > 0 else float("nan")
    print(f"  ε={ep:.2f}: 成本 {per['cost_main_M']:8.2f}M | 碳 {per['carbon_kt']:8.2f}kt"
          f" | Δ成本 {dC:+7.2f}M | Δ碳 {dE:+6.1f}kt"
          f" | 边际成本 {ratio:6.0f} 元/kg碳")
    prev = (per["cost_main_M"], per["carbon_kt"])

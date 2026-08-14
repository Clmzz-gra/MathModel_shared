# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    只读核验《S4 降低基荷实验计划》成本分解口径：储能/卖电贡献、
    层1 分配的区域不均衡（C 区为何 0 任务）。

原理：
    从 s4-cost-decomposition.pkl 与 s4-results.pkl 读聚合数字。

输入数据：
    - outputs/data/s4-cost-decomposition.pkl
    - outputs/data/s4-results.pkl — main 各区域 S/R/Cg/Cr/D/G 序列
    - outputs/data/s2-preprocessed.pkl — power 电价/卖电价（统计用）

输出：
    - 控制台统计量（PR-014）
"""
import pickle
from pathlib import Path

import numpy as np

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
MAIN = 2400


def main():
    try:
        with open(DATA / "s4-cost-decomposition.pkl", "rb") as f:
            dec = pickle.load(f)
        print("[s4-cost-decomposition.pkl 键]", list(dec.keys()))
        print(dec)
    except FileNotFoundError:
        print("s4-cost-decomposition.pkl 不存在")

    with open(DATA / "s4-results.pkl", "rb") as f:
        s4 = pickle.load(f)
    main_agg = s4["main"]
    print("\n[主解各区域 卖电/储能/购电 汇总（主时域）]")
    tot_sell_mwh = 0.0
    tot_sell_rev = 0.0
    tot_disch = 0.0
    for r in REGIONS:
        sol = main_agg["sols"][r]
        S = sol["S"][:MAIN]
        D = sol["D"][:MAIN]
        Cg = sol["Cg"][:MAIN]
        Cr = sol["Cr"][:MAIN]
        # 卖电电价取 s2 power（主时域）
        print(f"  {r}: 卖电 {S.sum():10,.0f} MWh | 放电 {D.sum():9,.0f} MWh"
              f" | 充电 电网 {Cg.sum():8,.0f} + 新能源 {Cr.sum():8,.0f} MWh"
              f" | 成本 {sol['cost_main_M']:8.2f} M")
        tot_sell_mwh += S.sum()
        tot_disch += D.sum()

    print(f"\n  六区合计卖电 {tot_sell_mwh:,.0f} MWh | 放电 {tot_disch:,.0f} MWh")
    print(f"  主解成本 {main_agg['cost_main_M']:.2f}M | B1 基线 2253.57M | Δ = "
          f"{main_agg['cost_main_M']-2253.57:+.2f}M")

    # 层1 分配：C 区为什么 0 候选
    with open(DATA / "sub4-preprocessed.pkl", "rb") as f:
        d = pickle.load(f)
    tasks = d["tasks"]
    power = d["power"]
    from collections import Counter
    dest_cnt = Counter(t["dest"] for t in tasks)
    print("\n[层1 分配结果]")
    for r in REGIONS:
        print(f"  {r}: {dest_cnt[r]:6d} 任务 | 电价均值 {np.mean(power[r]['price']):.0f}"
              f" | PUE {power[r]['pue']:.2f} | cap {power[r]['cap']:,.0f}")
    print(f"  实时任务占比: {sum(1 for t in tasks if t['type']=='RealTimeInference')/len(tasks):.1%}")
    src_cnt = Counter(t["source"] for t in tasks)
    print("  任务来源分布:", dict(src_cnt))


if __name__ == "__main__":
    main()

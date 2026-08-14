# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    只读打印 s4-sensitivity.pkl / s4-d-eps-scan.pkl 全部档位结果，
    供 2.2 结果分析（敏感性/场景）引用数值。

原理：
    纯统计输出，不重算模型。

输入数据：
    - outputs/data/s4-sensitivity.pkl — eps/price/renew 三组扫描
    - outputs/data/s4-d-eps-scan.pkl — D 区 ε 可压降扫描

输出：
    - 控制台统计表（PR-014）
"""
import pickle
from pathlib import Path

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"


def main():
    for name in ["s4-sensitivity.pkl", "s4-d-eps-scan.pkl"]:
        p = DATA / name
        if not p.exists():
            print(f"\n[{name}] 不存在")
            continue
        d = pickle.load(open(p, "rb"))
        print(f"\n[{name}] keys = {list(d.keys())}")
        if name == "s4-sensitivity.pkl":
            for k in ["eps", "price", "renew"]:
                if k not in d:
                    continue
                print(f"--- {k} 扫描 ---")
                for scale, res in d[k].items():
                    agg = res.get("agg", {})
                    if res.get("feasible") is False or not agg.get("feasible", True):
                        print(f"  {k}x{scale}: 不可行")
                        continue
                    print(
                        f"  {k}x{scale}: cost={agg.get('cost_main_M'):8.2f}M "
                        f"carbon={agg.get('carbon_kt'):8.2f}kt "
                        f"peak={agg.get('peak_MW'):6.1f} "
                        f"util_ns={agg.get('util_no_sell_pct'):5.2f}% "
                        f"n_inf={agg.get('n_infeasible')}"
                    )
        else:
            # s4-d-eps-scan
            for r, arr in d.items():
                if isinstance(arr, dict):
                    print(f"  {r}: {arr}")
                else:
                    print(f"  {r}: {arr}")


if __name__ == "__main__":
    main()

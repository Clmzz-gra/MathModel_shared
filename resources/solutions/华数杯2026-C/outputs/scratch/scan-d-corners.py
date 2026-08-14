# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 2.1.6 S4 边际成本拐点扫描 — 对 D 区（唯一高弹性区）在
    ε ∈ [ε_min, 1.00] 细网格扫描，生成成本-碳排曲线，定位
    "前段几乎免费、末段边际成本骤增"的拐点（供拐点可视化）。

原理：
    D 区层 2 协同 MILP（复用 sub4-model.solve_region）：
    - ε 从 ε_min≈0.9053 到 1.00 以 0.003 步长扫描（约 33 点，单点 0.3s）
    - 每点记录运行成本 cost_main_M 与购电碳排 carbon_kt（主时域口径）
    - 逐段边际成本 = ΔCost/ΔCarbon（元/kg），前段应为 100-300 元/kg
      （任务时移+储能吃新能源，几乎免费），接近 ε_min 段应骤增
      （须动用昂贵手段），形成拐点
    - 碳基准 = S4 自身 free 解 E0_S4（阶段 2.1 实证口径）
    - 受限消纳（S3 B1）、结清段实际值（S3 panel / c-data）

输入数据：
    - outputs/data/sub4-preprocessed.pkl（阶段 1.4）— tasks/power/storage
    - outputs/data/s3-preprocessed.pkl（阶段 1.4 S3）— panel（结清段实际值）
    - outputs/data/c-data-cleaned.pkl（阶段 0.3）— region_time_data
    - outputs/data/s2-preprocessed.pkl（阶段 1.4 S2）— latency
    - outputs/scratch/sub4-model.py（阶段 2.1）— solve_region
    - 中文指标 → 变量名映射：购电→G, 碳排→carbon_kt, 成本(M元)→cost_main_M

输出：
    - outputs/data/s4-d-eps-scan.pkl — {eps_grid, cost[], carbon[], feasible[]}
    - 控制台逐点成本/碳排/边际成本（PR-014）

对应论文章节：
    问题四（S4）— 碳约束灵敏度 / 降碳成本拐点分析
"""
import importlib.util
import pickle
from pathlib import Path

import numpy as np

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"
SCRATCH = BASE / "outputs" / "scratch"

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]

_spec = importlib.util.spec_from_file_location(
    "sub4model", SCRATCH / "sub4-model.py")
M4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M4)
solve_region = M4.solve_region


def load():
    with open(DATA / "sub4-preprocessed.pkl", "rb") as f:
        d = pickle.load(f)
    with open(DATA / "s3-preprocessed.pkl", "rb") as f:
        s3 = pickle.load(f)
    with open(DATA / "c-data-cleaned.pkl", "rb") as f:
        cd = pickle.load(f)
    with open(DATA / "s2-preprocessed.pkl", "rb") as f:
        s2 = pickle.load(f)

    panel = s3["panel"]
    ext = {}
    for r in REGIONS:
        pr = panel.xs(r, level="Region")
        ext[r] = {
            "price": pr["Price_CNY_per_MWh"].values,
            "sellp": pr["SellPrice_CNY_per_MWh"].values,
            "carbon": pr["CarbonIntensity_tCO2_per_MWh"].values,
            "renewable": pr["AvailableRenewable_MW"].values,
        }
    rtd = cd["region_time_data"]
    nonai_full, absorb_full = {}, {}
    for r in REGIONS:
        sub = rtd[rtd["Region"] == r].sort_values("Hour")
        nonai_full[r] = sub["NonAI_IT_Load_MW"].values[:2406].astype(float)
        absorb_full[r] = (sub["UsedRenewable_MW"].values[:2406]
                          + sub["RenewableCharge_MW"].values[:2406]).astype(float)
    return d, ext, nonai_full, absorb_full, s2["latency"]


def main():
    d, ext, nonai_full, absorb_full, latency = load()
    tasks, power, storage = d["tasks"], d["power"], d["storage"]

    # ε_min（来自 s4-sensitivity.pkl 精确二分结果）
    sens = pickle.load(open(DATA / "s4-sensitivity.pkl", "rb"))
    emin_d = sens["eps_min"]["RegionD"]
    e0 = sens["e0_s4_kt"]  # E0_S4 碳基准

    # 细网格：ε_min → 1.00，步长 0.003
    eps_grid = np.round(np.arange(emin_d, 1.001, 0.003), 5)
    if eps_grid[-1] != 1.0:
        eps_grid = np.append(eps_grid, 1.0)
    print(f"D 区 ε 扫描网格: {len(eps_grid)} 点, [{eps_grid[0]:.4f}, {eps_grid[-1]:.2f}]")

    costs, carbons, feas = [], [], []
    prev = None
    print(f"{'ε':>7} {'成本M':>9} {'碳kt':>8} {'Δ成本M':>8} {'Δ碳kt':>7} {'边际 元/kg':>10}")
    for ep in eps_grid:
        sol = solve_region("RegionD", tasks, power, storage, e0,
                           nonai_full, absorb_full, ext, eps=float(ep))
        ok = sol["feasible"]
        feas.append(ok)
        if ok:
            c, k = sol["cost_main_M"], sol["carbon_kt"]
            costs.append(c)
            carbons.append(k)
            if prev is not None:
                dC, dE = c - prev[0], prev[1] - k
                marg = dC / dE * 1000 if dE > 1e-9 else float("nan")
                print(f"{float(ep):7.4f} {c:9.2f} {k:8.2f} {dC:+8.2f} {dE:+7.2f} {marg:10.0f}")
            else:
                print(f"{float(ep):7.4f} {c:9.2f} {k:8.2f}")
            prev = (c, k)
        else:
            costs.append(np.nan)
            carbons.append(np.nan)
            print(f"{float(ep):7.4f} 不可行")

    out = {
        "eps_grid": eps_grid.tolist(), "cost": costs, "carbon": carbons,
        "feasible": feas, "e0_kt": e0["RegionD"], "eps_min": emin_d,
        "meta": {"generated": "2026-08-09",
                 "source": "solve_region D 区 ε 细网格扫描（步长 0.003）"},
    }
    with open(DATA / "s4-d-eps-scan.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n已写入 {DATA / 's4-d-eps-scan.pkl'}")
    print("S4 D-EPS-SCAN DONE")


if __name__ == "__main__":
    main()

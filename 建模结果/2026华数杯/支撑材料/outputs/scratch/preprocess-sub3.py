# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
preprocess-sub3.py — S3 储能协同优化：数据预处理（阶段 1.4，LP 输入面板）

1. 目的
   为 S3（问题三：储能协同优化）阶段 2.1 LP 求解构造统一输入面板
   outputs/data/s3-preprocessed.pkl：6 区域 × 0–2405h 全时域逐时电价/卖电价/碳强度/
   设施负荷/可用新能源 + 储能参数 + 区域购售电边界 + 碳基准（ε 档分母）+ 口径自检结果。

2. 原理
   - 时域口径（确认书 A9）：0–2399 主时域 + 2400–2405 结清段均入模；
     2406 仅状态结算（E_2406 = E_2405），不建变量、不需单独行。
   - 负荷核验：Total_Load_MW == (Baseline_AI_IT_Load + NonAI_IT_Load) × PUE（A10）
   - 碳基准：CarbonBase_r = Σ(GridPurchase×CarbonIntensity)（主时域 0–2399，/1e3 转 kt），
     与数据列 CarbonEmission 及 F2 值核对（口径自洽）
   - 功率平衡（赛题统一口径，全时域 0–2405）：
     GridPurchase + AvailableRenewable + DischargePower
       = Total_Load + ChargePower + GridSell + Curtailment
   - SOC 递推（SOC 为时段末状态，InitialSOC 为 Hour 0 前，全时域 0–2406 复核）：
     E(t) = E(t-1) + ηc·C(t) − D(t)/ηd，E(−1) = InitialSOC
     RegionE Hour0 残差 ≈0.9999 MWh 为数据异常，仅记录不修改（A6 / @PROXY 核销）

3. 输入映射
   - outputs/data/csv/region_time_data/region_time_data.csv（0–2406h × 6 区逐时电力/负荷/储能基准）
   - outputs/data/csv/storage_information/storage_information.csv（储能参数与购售电边界）
   - outputs/data/csv/GPU_information/GPU中心基础情况.csv（PUE，负荷折算核验用）

4. 输出
   - outputs/data/s3-preprocessed.pkl（6 键：meta/panel/storage/carbon_base_kt/epsilon/check）
   - 控制台自检打印（设计 §5 基准核对 + 各数组 min/max/mean/std 统计量，PR-014）

5. 论文章节
   - 问题三 储能协同优化：LP 输入面板与统一口径核验（阶段 1.4）
"""
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"E:\MathModel_pj-2026-C-sub3\outputs\data\csv")
OUT = Path(r"E:\MathModel_pj-2026-C-sub3\outputs\data\s3-preprocessed.pkl")
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]

# 验证基准（设计 §5 / handoff §4）：仅核对用，不覆盖计算值
F2_CARBON_KT = {"RegionA": 581.27, "RegionB": 519.70, "RegionC": 483.82,
                "RegionD": 262.96, "RegionE": 84.56, "RegionF": 113.06}
REF_TOTAL_CARBON_KT = 2045.36
REF_COST_M = 1802.34        # 基准成本合计（主时域 0–2399）
REF_PEAK_NET_MW = 497.0     # 峰值净购电（RegionA）
REF_RESID_MAX_MW = 0.0002   # 功率平衡残差上界
TOL_BASELINE = 5e-3         # 与基准偏差阈值（>0.5% 暂停回查，勿自行修正）
TOL_LOAD_MW = 1e-3          # 负荷核验容差（建模裁定 §7.1：由 1e-6 放宽；纯 4 位小数舍入噪声，结构核验语义）

EPSILON = [0.90, 0.95, 1.00]  # 碳排 ε 档（@PROXY，阶段 2.0 核销）


def load():
    rtd = pd.read_csv(DATA / "region_time_data" / "region_time_data.csv")
    st = pd.read_csv(DATA / "storage_information" / "storage_information.csv")
    gpu = pd.read_csv(DATA / "GPU_information" / "GPU中心基础情况.csv")
    return rtd, st, gpu


def describe(name, arr):
    a = np.asarray(arr, dtype=float)
    print(f"    [{name}] min {a.min():10.4f} | max {a.max():10.4f} | "
          f"mean {a.mean():10.4f} | std {a.std():10.4f}")


def check_load(rtd, pue):
    """负荷核验：|Total_Load − (Baseline_AI_IT_Load + NonAI_IT_Load)×PUE| ≤ TOL_LOAD_MW（全时域 0–2406）

    容差 TOL_LOAD_MW=1e-3 MW（建模裁定 §7.1）：IT_Load 与 Baseline+NonAI 精确相等（max|Δ|≈3e-7）；
    Total_Load 为 4 位小数存储，×PUE 后存在 ~5e-5 MW 舍入（相对 ~1e-7）。原阈值 1e-6 触发 1066 行
    假违规，纯属舍入噪声；按结构核验语义放宽后应全部通过。不修正数据。"""
    calc = (rtd["Baseline_AI_IT_Load_MW"] + rtd["NonAI_IT_Load_MW"]) * rtd["Region"].map(pue)
    diff = (rtd["Total_Load_MW"] - calc).abs()
    ok = bool((diff <= TOL_LOAD_MW).all())
    n_bad = int((diff > TOL_LOAD_MW).sum())
    if not ok:
        bad = rtd[diff > TOL_LOAD_MW]
        n_2406 = int((bad["Hour"] == 2406).sum())
        print(f"    [负荷核验] 违规 {n_bad} 行（其中 Hour=2406 {n_2406} 行），max|Δ| = {diff.max():.2e} MW")
    return ok, float(diff.max())


def carbon_base(rtd):
    """碳基准（主时域 0–2399）与数据列 CarbonEmission 累计交叉核验"""
    d = rtd[rtd["Hour"] < 2400]
    base, col_sum = {}, {}
    for r in REGIONS:
        sub = d[d["Region"] == r]
        base[r] = float((sub["GridPurchase_MW"] * sub["CarbonIntensity_tCO2_per_MWh"]).sum()) / 1e3
        col_sum[r] = float(sub["CarbonEmission_tCO2"].sum()) / 1e3
    return base, col_sum, float(sum(base.values())), float(sum(col_sum.values()))


def power_balance_max_resid(rtd):
    """功率平衡复算（全时域 0–2405）"""
    d = rtd[rtd["Hour"] <= 2405]
    lhs = d["GridPurchase_MW"] + d["AvailableRenewable_MW"] + d["DischargePower_MW"]
    rhs = d["Total_Load_MW"] + d["ChargePower_MW"] + d["GridSell_MW"] + d["Curtailment_MW"]
    return float((lhs - rhs).abs().max())


def soc_recursion(rtd, st):
    """SOC 递推复算（全时域 0–2406，E(−1)=InitialSOC）；RegionE Hour0 异常记录不修改

    双口径并列：
    - 链式（handoff §3.5 口径）：E(t) 由上一**递推值**累推，残差为整段轨迹一致性
      （2406 步舍入累积，量级 ~0.001–0.005 MWh）→ 写入 check.soc_recur_resid_max
    - 逐时（F3 口径）：t≥1 用数据 SOC(t−1) 作前值时点，残差为单步自洽性（~0.0001 MWh）
    """
    resid_chain, resid_step = {}, {}
    note = ""
    for _, s in st.iterrows():
        r = s["Region"]
        d = rtd[rtd["Region"] == r].sort_values("Hour")
        soc = d["SOC_MWh"].values
        ch = d["ChargePower_MW"].values
        dis = d["DischargePower_MW"].values
        eta_c, eta_d = float(s["ChargeEfficiency"]), float(s["DischargeEfficiency"])
        init = float(s["InitialSOC_MWh"])
        # 链式递推（E(−1)=InitialSOC）
        pred_c = np.empty_like(soc)
        prev = init
        for t in range(len(soc)):
            pred_c[t] = prev + eta_c * ch[t] - dis[t] / eta_d
            prev = pred_c[t]
        resid_chain[r] = float(np.abs(soc - pred_c).max())
        resid0 = float(np.abs(soc[0] - pred_c[0]))  # Hour0 链式残差（RegionE 异常点）
        # 逐时口径（t=0 用 InitialSOC，t≥1 用数据前值时点，F3 同款）
        pred_s = np.empty_like(soc)
        pred_s[0] = init + eta_c * ch[0] - dis[0] / eta_d
        for t in range(1, len(soc)):
            pred_s[t] = soc[t - 1] + eta_c * ch[t] - dis[t] / eta_d
        resid_step[r] = float(np.abs(soc - pred_s).max())
        if r == "RegionE" and resid0 > 0.5:
            note = (f"RegionE Hour{int(d['Hour'].values[0])} SOC 递推残差 {resid0:.4f} MWh："
                    f"数据 SOC(0)={soc[0]:.3f} vs 递推 InitialSOC({init:.1f})+ηc·C(0)−D(0)/ηd={pred_c[0]:.3f}；"
                    f"建模以赛题递推式为约束，数据异常仅记录不修改（@PROXY 核销）")
    return resid_chain, resid_step, note


def base_cost_peak(rtd):
    """基准成本合计与峰值净购电（主时域 0–2399，设计 §5 核对项）"""
    d = rtd[rtd["Hour"] < 2400]
    cost = 0.0
    peak = 0.0
    for r in REGIONS:
        sub = d[d["Region"] == r]
        cost += float((sub["GridPurchase_MW"] * sub["ElectricityPrice_CNY_per_MWh"]
                       - sub["GridSell_MW"] * sub["SellPrice_CNY_per_MWh"]).sum()) / 1e6
        peak = max(peak, float(sub["NetGridImport_MW"].max()))
    return cost, peak


def build_panel(rtd):
    """长表面板：Hour 0–2405 全时域，index=(Region, Hour)，列映射见设计 §4"""
    d = rtd[rtd["Hour"] <= 2405].copy()
    panel = d[["Region", "Hour", "ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh",
               "CarbonIntensity_tCO2_per_MWh", "Total_Load_MW", "AvailableRenewable_MW",
               "UsedRenewable_MW", "GridPurchase_MW", "NetGridImport_MW"]].rename(columns={
                   "ElectricityPrice_CNY_per_MWh": "Price_CNY_per_MWh",
                   "GridPurchase_MW": "GridPurchase_base_MW",
                   "NetGridImport_MW": "NetGridImport_base_MW"})
    return panel.sort_values(["Region", "Hour"]).set_index(["Region", "Hour"])


def build_storage(st):
    """储能参数与购售电边界（10 项/区域）"""
    storage = {}
    for _, s in st.iterrows():
        r = s["Region"]
        storage[r] = {
            "Capacity_MWh": float(s["StorageCapacity_MWh"]),
            "MinSOC_MWh": float(s["MinSOC_MWh"]),
            "InitialSOC_MWh": float(s["InitialSOC_MWh"]),
            "MaxChargePower_MW": float(s["MaxChargePower_MW"]),
            "MaxDischargePower_MW": float(s["MaxDischargePower_MW"]),
            "ChargeEfficiency": float(s["ChargeEfficiency"]),
            "DischargeEfficiency": float(s["DischargeEfficiency"]),
            "SellLimit_MW": float(s["SellLimit_MW"]),
            "MaxGridImport_MW": float(s["MaxGridImport_MW"]),
            "MaxGridExport_MW": float(s["MaxGridExport_MW"]),
        }
    return storage


def check_selllimit(st):
    """卖电边界核对：SellLimit vs MaxGridExport（建模裁定 §7.3：六区全一致，无待确认项）"""
    for _, s in st.iterrows():
        r = s["Region"]
        sl, mg = float(s["SellLimit_MW"]), float(s["MaxGridExport_MW"])
        same = abs(sl - mg) < 1e-6
        mark = "OK" if same else "!! 不一致（数据异常，需人工核查）"
        print(f"    {r}: SellLimit {sl:g} MW {'==' if same else '≠'} MaxGridExport {mg:g} MW [{mark}]")


def main():
    rtd, st, gpu = load()
    pue = dict(zip(gpu["Region"], gpu["PUE"]))
    print("=" * 78)
    print("S3 阶段 1.4 数据预处理 — LP 输入面板（design: preprocess-sub3-20260808.md）")
    print("=" * 78)

    # 1) 负荷核验
    print("\n1) 负荷核验 Total_Load == (Baseline_AI_IT_Load + NonAI_IT_Load) × PUE（全时域 0–2406）")
    load_ok, load_diff = check_load(rtd, pue)
    print(f"    → load_equals_IT_times_PUE = {load_ok}（max|Δ| = {load_diff:.2e} MW）")

    # 2) 碳基准
    print("\n2) 碳基准 CarbonBase_r = Σ(GridPurchase×CarbonIntensity)（主时域 0–2399）")
    base, col_sum, tot, tot_col = carbon_base(rtd)
    for r in REGIONS:
        print(f"    {r}: 计算 {base[r]:8.2f} kt | 数据列 {col_sum[r]:8.2f} kt | F2 {F2_CARBON_KT[r]:8.2f} kt")
    print(f"    合计 计算 {tot:8.2f} kt | 数据列 {tot_col:8.2f} kt | F2 {REF_TOTAL_CARBON_KT:8.2f} kt")
    carbon_col_ok = all(abs(base[r] - col_sum[r]) / col_sum[r] < 1e-4 for r in REGIONS)
    carbon_f2_ok = all(abs(base[r] - F2_CARBON_KT[r]) / F2_CARBON_KT[r] < TOL_BASELINE for r in REGIONS)
    print(f"    → 与 CarbonEmission 列一致: {carbon_col_ok} | 与 F2 一致: {carbon_f2_ok}")

    # 3) 功率平衡
    resid_pb = power_balance_max_resid(rtd)
    print(f"\n3) 功率平衡残差 max（全时域 0–2405）= {resid_pb:.4f} MW（基准 ≤{REF_RESID_MAX_MW}）")
    pb_ok = resid_pb <= REF_RESID_MAX_MW + 1e-9

    # 4) SOC 递推
    print("\n4) SOC 递推复算（全时域 0–2406，E(−1)=InitialSOC；链式口径 vs F3 逐时口径）")
    resid_chain, resid_step, note = soc_recursion(rtd, st)
    for r in REGIONS:
        print(f"    {r}: 链式残差 max {resid_chain[r]:.4f} MWh | F3 逐时口径 {resid_step[r]:.4f} MWh")
    if note:
        print(f"    [RegionE Hour0 异常记录] {note}")

    # 5) 基准成本 / 峰值净购电核对（主时域 0–2399）
    cost_m, peak_mw = base_cost_peak(rtd)
    cost_ok = abs(cost_m - REF_COST_M) / REF_COST_M < TOL_BASELINE
    peak_ok = abs(peak_mw - REF_PEAK_NET_MW) / REF_PEAK_NET_MW < TOL_BASELINE
    print(f"\n5) 基准成本核对（主时域 0–2399）: 计算 {cost_m:.2f} M元 vs 基准 {REF_COST_M:.2f} → {cost_ok}")
    print(f"   峰值净购电核对: 计算 {peak_mw:.1f} MW vs 基准 {REF_PEAK_NET_MW:.1f} → {peak_ok}")

    # 6) 卖电边界核对
    print("\n6) 卖电边界 SellLimit vs MaxGridExport（建模裁定 §7.3：六区全一致）")
    check_selllimit(st)

    # 7) 面板与储能参数
    panel = build_panel(rtd)
    storage = build_storage(st)
    print(f"\n7) 面板 panel: 行数 {panel.shape[0]}（=6 区域 × {panel.shape[0] // 6} 时点，Hour 0–2405）| "
          f"列数 {panel.shape[1]} | NaN 数 {int(panel.isna().sum().sum())}")
    for c in panel.columns:
        describe(c, panel[c].values)
    print("   储能参数（每区域 10 项）:")
    for r in REGIONS:
        s = storage[r]
        print(f"    {r}: Cap {s['Capacity_MWh']:4.0f} | Init {s['InitialSOC_MWh']:6.1f} "
              f"({s['InitialSOC_MWh'] / s['Capacity_MWh'] * 100:4.1f}%) | Min {s['MinSOC_MWh']:5.1f} "
              f"({s['MinSOC_MWh'] / s['Capacity_MWh'] * 100:4.1f}%) | "
              f"ηc {s['ChargeEfficiency']:.2f}/ηd {s['DischargeEfficiency']:.2f} | "
              f"充放 {s['MaxChargePower_MW']:3.0f}/{s['MaxDischargePower_MW']:3.0f} MW | "
              f"SellLimit {s['SellLimit_MW']:3.0f} | MaxImport {s['MaxGridImport_MW']:3.0f} MW")

    # 8) 汇总与落盘
    print("\n8) 自检汇总（check 字典）")
    checks = {
        "load_equals_IT_times_PUE": load_ok,
        "power_balance_resid_max_MW": resid_pb,
        "carbon_base_matches_F2": carbon_col_ok and carbon_f2_ok,
        "soc_recur_resid_max": max(resid_chain.values()),
        "regionE_hour0_note": note,
    }
    for k, v in checks.items():
        print(f"    {k} = {v}")
    baselines_ok = load_ok and carbon_f2_ok and pb_ok and cost_ok and peak_ok
    print(f"    基准核对总判定（负荷+碳/F2+功率平衡+成本+峰值）: {'PASS' if baselines_ok else 'FAIL（>0.5% 偏差，暂停回查，勿自行修正）'}")

    out = {
        "meta": {
            "generated": "2026-08-08",
            "source": "outputs/data/csv/{region_time_data,storage_information,GPU_information}（阶段 0.2 缓存）",
            "hours": list(range(2406)),  # 0–2405 全时域；2406 状态结算 = E_2405，不建变量
            "units": {"power": "MW", "energy": "MWh", "price": "CNY/MWh",
                      "carbon": "tCO2/MWh", "cost": "CNY", "carbon_total": "tCO2"},
            "pue": pue,  # 记录用（Load 已含 PUE 折算，LP 不再乘）
        },
        "panel": panel,
        "storage": storage,
        "carbon_base_kt": base,
        "epsilon": EPSILON,
        "check": checks,
    }
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    print(f"\n已写入 {OUT}（{OUT.stat().st_size / 1e3:.1f} KB）")
    print(f"重跑时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("S3 PREPROCESSING DONE (preprocess-sub3.py)")


if __name__ == "__main__":
    main()

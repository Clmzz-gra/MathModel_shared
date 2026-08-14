# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
verify-sub3.py — S3 储能协同优化：A 类共享事实验证（阶段 1.1）

1. 目的
   为 S3（问题三：储能协同优化）方案决策提供与方案选择无关的客观数据事实：
   F1 无储能简单基线（性能下界，A 类第一步强制）、F2 基准运行状态 4 指标、
   F3 储能参数自洽性、F4 同时充放事实、F5 功率平衡自洽性、F6 新能源利用率基准口径。

2. 原理
   - 功率平衡（赛题统一口径）：
     GridPurchase + AvailableRenewable + DischargePower = Total_Load + ChargePower + GridSell + Curtailment
   - SOC 递推（SOC 为时段末状态，InitialSOC 为 Hour 0 前）：
     SOC(t) = SOC(t-1) + eta_c*ChargePower(t) - DischargePower(t)/eta_d
   - 成本 = Sum(GridPurchase*Price - GridSell*SellPrice)；碳排 = Sum(GridPurchase*CarbonIntensity)
   - 无储能基线（无充电/放电，三口径）：
     口径a 零新能源利用: GridPurchase = Total_Load
     口径b 新能源全可消纳: GridPurchase = max(Total_Load - AvailableRenewable, 0)
     口径c 基准直消纳能力: GridPurchase = max(Total_Load - UsedRenewable, 0)

3. 输入映射
   - outputs/data/csv/region_time_data/region_time_data.csv（0-2406h x 6 区逐时电力/负荷/储能基准）
   - outputs/data/csv/storage_information/storage_information.csv（储能参数）
   - outputs/data/csv/GPU_information/GPU中心基础情况.csv（PUE，设施负荷折算参考）

4. 输出
   - 控制台逐项打印 F1-F6 结果（min/max/mean/std 统计量）；不落盘——阶段 1.1 仅共享事实

5. 论文章节
   - 问题三 储能协同优化：储能价值评估的对照基准与统一口径核验
"""
import numpy as np
import pandas as pd

DATA = r"E:\MathModel_pj-2026-C-sub3\outputs\data\csv"
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]


def load():
    rtd = pd.read_csv(f"{DATA}/region_time_data/region_time_data.csv")
    st = pd.read_csv(f"{DATA}/storage_information/storage_information.csv")
    gpu = pd.read_csv(f"{DATA}/GPU_information/GPU中心基础情况.csv")
    return rtd, st, gpu


def cost_carbon(g, price, sell_price, carbon, gs):
    """成本(百万元)与碳排(kt)累计；g/gridpurchase, gs/gridsell"""
    cost_m = float(np.sum(g * price - gs * sell_price)) / 1e6
    carb_kt = float(np.sum(g * carbon)) / 1e3
    return cost_m, carb_kt


def main():
    rtd, st, gpu = load()

    # 主时域 0-2399；结清段 2400-2405 另行报告
    main_df = rtd[rtd["Hour"] < 2400].copy()
    tail_df = rtd[rtd["Hour"] >= 2400].copy()

    print("=" * 78)
    print("F1 无储能简单基线（三口径；无储能 = 无充电/放电）")
    print("    口径a 零新能源利用: GridPurchase=Total_Load（成本上界）")
    print("    口径b 新能源全可消纳: GridPurchase=max(Total_Load-Renewable,0)（理想下界，现实中弃电率~79%不可达）")
    print("    口径c 基准直消纳能力: GridPurchase=Total_Load-UsedRenewable（保留基准直消纳、去掉储能的诚实基线）")
    print("=" * 78)
    f1 = {"a": {}, "b": {}, "c": {}}
    for r in REGIONS:
        d = main_df[main_df["Region"] == r]
        tl = d["Total_Load_MW"].values
        ren = d["AvailableRenewable_MW"].values
        used = d["UsedRenewable_MW"].values
        pr = d["ElectricityPrice_CNY_per_MWh"].values
        ci = d["CarbonIntensity_tCO2_per_MWh"].values
        sp = d["SellPrice_CNY_per_MWh"].values
        for k, gp in zip(("a", "b", "c"),
                         (tl, np.maximum(tl - ren, 0.0), np.maximum(tl - used, 0.0))):
            cost_m, carb_kt = cost_carbon(gp, pr, sp, ci, np.zeros_like(gp))
            f1[k][r] = dict(cost=cost_m, carb=carb_kt,
                            peak=float(gp.max()), std=float(gp.std()))
        print(f"  {r}: 成本 a {f1['a'][r]['cost']:8.2f} / b {f1['b'][r]['cost']:8.2f} / "
              f"c {f1['c'][r]['cost']:8.2f} M元 | "
              f"碳排 c {f1['c'][r]['carb']:7.2f} kt | 峰值净购电 c {f1['c'][r]['peak']:7.1f} MW")
    for k, tag in (("a", "零利用"), ("b", "全消纳理想"), ("c", "直消纳代理")):
        print(f"  合计[{tag}]: 成本 {sum(v['cost'] for v in f1[k].values()):.2f} M元 | "
              f"碳排 {sum(v['carb'] for v in f1[k].values()):.2f} kt | "
              f"峰值净购电 {max(v['peak'] for v in f1[k].values()):.1f} MW")

    print()
    print("=" * 78)
    print("F2 基准运行状态 4 指标（region_time_data 自带基准轨迹）")
    print("=" * 78)
    f2 = {}
    for r in REGIONS:
        d = main_df[main_df["Region"] == r]
        cost_m, carb_kt = cost_carbon(
            d["GridPurchase_MW"].values, d["ElectricityPrice_CNY_per_MWh"].values,
            d["SellPrice_CNY_per_MWh"].values,
            d["CarbonIntensity_tCO2_per_MWh"].values, d["GridSell_MW"].values)
        # 碳排列交叉核验
        carb_col = float(d["CarbonEmission_tCO2"].sum()) / 1e3
        f2[r] = dict(cost=cost_m, carb=carb_kt, carb_col=carb_col,
                     peak=float(d["NetGridImport_MW"].max()),
                     std=float(d["NetGridImport_MW"].std()),
                     std_load=float(d["Total_Load_MW"].std()))
        print(f"  {r}: 成本 {cost_m:8.2f} M元 | 碳排 {carb_kt:7.2f} kt(列 {carb_col:7.2f}) | "
              f"峰值净购电 {d['NetGridImport_MW'].max():7.1f} MW | "
              f"净购电 std {d['NetGridImport_MW'].std():6.2f} | 负荷 std {d['Total_Load_MW'].std():6.2f} MW")
    print(f"  合计: 成本 {sum(v['cost'] for v in f2.values()):.2f} M元 | "
          f"碳排 {sum(v['carb'] for v in f2.values()):.2f} kt | "
          f"峰值净购电 {max(v['peak'] for v in f2.values()):.1f} MW")
    # 基准 vs 无储能（口径c：诚实基线）
    base_cost = sum(v["cost"] for v in f2.values()); no_cost = sum(v["cost"] for v in f1["c"].values())
    base_carb = sum(v["carb"] for v in f2.values()); no_carb = sum(v["carb"] for v in f1["c"].values())
    print(f"  [F1c->F2] 基准相对无储能(直消纳代理): 成本 {(base_cost-no_cost)/no_cost*100:+.1f}% | "
          f"碳排 {(base_carb-no_carb)/no_carb*100:+.1f}%")

    print()
    print("=" * 78)
    print("F3 储能参数自洽性（storage_information + 基准 SOC 递推复核）")
    print("=" * 78)
    for _, s in st.iterrows():
        r = s["Region"]
        init_ratio = s["InitialSOC_MWh"] / s["StorageCapacity_MWh"]
        min_ratio = s["MinSOC_MWh"] / s["StorageCapacity_MWh"]
        # SOC 递推复核（基准轨迹）
        d = rtd[rtd["Region"] == r].sort_values("Hour")
        soc = d["SOC_MWh"].values
        ch = d["ChargePower_MW"].values
        dis = d["DischargePower_MW"].values
        eta_c, eta_d = s["ChargeEfficiency"], s["DischargeEfficiency"]
        pred = np.empty_like(soc)
        pred[0] = s["InitialSOC_MWh"] + eta_c * ch[0] - dis[0] / eta_d
        for t in range(1, len(soc)):
            pred[t] = soc[t - 1] + eta_c * ch[t] - dis[t] / eta_d
        resid = np.abs(soc - pred)
        # 同时充放 & 超限检查
        dual = np.sum((ch > 0.01) & (dis > 0.01))
        over_ch = np.sum(ch > s["MaxChargePower_MW"] + 1e-6)
        over_dis = np.sum(dis > s["MaxDischargePower_MW"] + 1e-6)
        # SOC(2406) vs Initial 合规（赛题说明 5：SOC(2406) >= InitialSOC）
        soc_end = soc[-1]
        ok = "OK" if soc_end >= s["InitialSOC_MWh"] - 1e-6 else "!!!违规: 低于Initial"
        print(f"  {r}: 容量 {s['StorageCapacity_MWh']:4.0f} MWh | Init/容量 {init_ratio*100:5.1f}% | "
              f"Min/容量 {min_ratio*100:4.1f}% | SOC递推残差max {resid.max():8.4f} | "
              f"同刻充放 {dual:4d} h | 充/放超限 {over_ch}/{over_dis} | "
              f"SOC(2406) {soc_end:7.1f} vs Initial {s['InitialSOC_MWh']:7.1f} [{ok}]")
        if r == "RegionE" and resid.max() > 0.5:
            t_bad = int(np.argmax(resid))
            # t=0 时无 t-1 状态，Hour 0 前状态即为 InitialSOC（避免 soc[-1] 负索引回绕）
            soc_prev = s["InitialSOC_MWh"] if t_bad == 0 else soc[t_bad - 1]
            print(f"      RegionE 残差最大时点 Hour={int(d['Hour'].values[t_bad])} "
                  f"resid={resid[t_bad]:.4f} SOC(t-1)={soc_prev:.3f} SOC(t)={soc[t_bad]:.3f} "
                  f"ch={ch[t_bad]:.3f} dis={dis[t_bad]:.3f}")

    print()
    print("=" * 78)
    print("F4 同时充放事实（基准轨迹 Charge>0 且 Discharge>0 的小时）")
    print("=" * 78)
    for r in REGIONS:
        d = main_df[main_df["Region"] == r]
        dual = np.sum((d["ChargePower_MW"] > 0.01) & (d["DischargePower_MW"] > 0.01))
        n = len(d)
        print(f"  {r}: {dual:4d} h / {n} h ({dual/n*100:.2f}%)")
    dual_total = sum(np.sum((main_df[main_df["Region"] == r]["ChargePower_MW"] > 0.01) &
                            (main_df[main_df["Region"] == r]["DischargePower_MW"] > 0.01))
                     for r in REGIONS)
    print(f"  合计: {dual_total} h")

    print()
    print("=" * 78)
    print("F5 功率平衡自洽性（GridPurchase+Renewable+Discharge = Load+Charge+Sell+Curtailment）")
    print("=" * 78)
    for r in REGIONS:
        d = rtd[rtd["Region"] == r]
        lhs = d["GridPurchase_MW"] + d["AvailableRenewable_MW"] + d["DischargePower_MW"]
        rhs = d["Total_Load_MW"] + d["ChargePower_MW"] + d["GridSell_MW"] + d["Curtailment_MW"]
        resid = (lhs - rhs).abs()
        print(f"  {r}: 残差 max {resid.max():8.4f} | mean {resid.mean():8.4f} MW")

    print()
    print("=" * 78)
    print("F6 新能源利用率基准口径（主时域 0-2399）")
    print("=" * 78)
    for r in REGIONS:
        d = main_df[main_df["Region"] == r]
        avail = d["AvailableRenewable_MW"].sum()
        used = d["UsedRenewable_MW"].sum()
        rch = d["RenewableCharge_MW"].sum()
        curt = d["Curtailment_MW"].sum()
        gs = d["GridSell_MW"].sum()
        util_dc = (used + rch) / avail * 100          # 消纳率口径（直供+充电）
        util_gs = (used + rch + gs) / avail * 100     # 若 GridSell 计入新能源外送
        print(f"  {r}: 利用率(直供+充电) {util_dc:5.2f}% | 含GridSell {util_gs:5.2f}% | "
              f"弃电率 {curt/avail*100:5.2f}%")
    util_all = (main_df["UsedRenewable_MW"].sum() + main_df["RenewableCharge_MW"].sum()) / \
               main_df["AvailableRenewable_MW"].sum() * 100
    print(f"  全局: 利用率(直供+充电) {util_all:.2f}%")

    print()
    print("=" * 78)
    print("F7 新能源消纳可行性（可用 vs 消纳关系，判断消纳上限建模依据）")
    print("=" * 78)
    for r in REGIONS:
        d = main_df[main_df["Region"] == r]
        avail = d["AvailableRenewable_MW"]
        used = d["UsedRenewable_MW"]
        rch = d["RenewableCharge_MW"]
        tot = d["Total_Load_MW"]
        # 可用新能源的分布特征
        print(f"  {r}: 可用 均值 {avail.mean():7.1f} min {avail.min():7.1f} max {avail.max():7.1f} | "
              f"直消纳率(used/avail) 均值 {(used/avail).mean()*100:5.1f}% | "
              f"直消纳<=负荷占比 {(used <= tot).mean()*100:5.1f}% | "
              f"总利用(used+rch)/avail 均值 {((used+rch)/avail).mean()*100:5.1f}%")
    # 全局：若 LP 允许自由消纳（≤可用），理论可多消纳多少
    tot_avail = main_df["AvailableRenewable_MW"].sum()
    tot_used = main_df["UsedRenewable_MW"].sum()
    tot_rch = main_df["RenewableCharge_MW"].sum()
    print(f"  全局累计: 可用 {tot_avail/1e6:.2f} GWh | 直消纳 {tot_used/1e6:.2f} GWh | "
          f"新能源充电 {tot_rch/1e6:.2f} GWh | 弃电 {1-(tot_used+tot_rch)/tot_avail:.1%}")

    print()
    print("=" * 78)
    print("F8 卖电套利事实（GridSell 活跃区域与收益，RegionE 成本为负探因）")
    print("=" * 78)
    for r in REGIONS:
        d = main_df[main_df["Region"] == r]
        gs = d["GridSell_MW"]
        sp = d["SellPrice_CNY_per_MWh"]
        rev = float((gs * sp).sum()) / 1e6
        n_sell = int((gs > 0.01).sum())
        print(f"  {r}: 卖电 {n_sell:4d} h | 电量 {gs.sum()/1e3:8.1f} GWh | "
              f"收入 {rev:8.2f} M元 | SellPrice 均值 {sp.mean():6.2f} 元/MWh")

    # 结清段 2400-2406 储能运行快照（问题三需覆盖；2406 仅状态结算）
    print()
    print("=" * 78)
    print("附: 结清段 2400-2406 基准运行（含 2406 状态结算时点；主时域 0-2399 之外）")
    print("=" * 78)
    for r in REGIONS:
        d = tail_df[tail_df["Region"] == r]
        if len(d) == 0:
            continue
        print(f"  {r}: 净购电max {d['NetGridImport_MW'].max():7.1f} | "
              f"SOC末 {d['SOC_MWh'].values[-1]:7.1f} | 充 {d['ChargePower_MW'].sum():7.1f} | "
              f"放 {d['DischargePower_MW'].sum():7.1f} MWh")

    print()
    print("ALL A-CLASS VERIFICATION DONE (verify-sub3.py)")


if __name__ == "__main__":
    main()

# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 0.2 数据盘点 — 将 6 个附件 xlsx 数据 sheet 缓存为 pickle，执行关联键完整性检查
    与变量量纲/单位初查，输出数据清单表

原理：
    使用 pandas.read_excel 加载数据 sheet，pickle 序列化缓存到 outputs/data/。
    关联键检查：workload_trace.SourceRegion / GPU_information.Region /
    storage_information.Region 与 region_time_data.Region 的一致性；Hour 时域覆盖检查。
    量纲初查：对每个数值列输出 min/max/mean/median/非零比例，检测量级异常。

输入数据：
    - 附件数据/*.xlsx 的 6 个数据 sheet（原始）
    - outputs/data/csv/ 下已导出的 CSV（回溯备查）

输出：
    - outputs/data/c-data-raw.pkl — 六表 pickle 字典
    - solution/data-inventory-02.md — 数据清单表 + 关联键覆盖报告 + 量纲初查报告

对应论文章节：
    TRAE.md 阶段 0.2 数据盘点
"""
from pathlib import Path

import pandas as pd

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA_DIR = BASE / "problems" / "2026年第七届华数杯数学建模竞赛赛题" / "C题 面向算电协同的多目标调度优化研究" / "附件数据"
OUT = BASE / "outputs" / "data"
OUT.mkdir(parents=True, exist_ok=True)

SHEETS = {
    "GPU_information.xlsx": "GPU中心基础情况",
    "workload_trace.xlsx": "Sheet1",
    "region_time_data.xlsx": "region_time_data",
    "power_mapping.xlsx": "任务功率映射",
    "network_latency.xlsx": "network_latency",
    "storage_information.xlsx": "storage_information",
}

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
NUMERIC_HOURS = list(range(0, 2407))


def main():
    report = ["# C 题 阶段 0.2 数据盘点报告\n"]
    report.append("> 由 `outputs/scratch/data-inventory-02.py` 自动生成，2026-08-07\n")

    dfs = {}
    for fname, sheet in SHEETS.items():
        df = pd.read_excel(DATA_DIR / fname, sheet_name=sheet)
        dfs[fname.replace(".xlsx", "")] = df
        report.append(f"\n## {fname}（sheet: {sheet}）shape={df.shape}")
        report.append(f"列：{', '.join(map(str, df.columns))}")

    # 缓存
    cache_path = OUT / "c-data-raw.pkl"
    dfs["_meta"] = {"sheets": SHEETS, "generated": "2026-08-07"}
    pd.to_pickle(dfs, cache_path)
    report.append(f"\n## 缓存\n已写入 `outputs/data/c-data-raw.pkl`（{cache_path.stat().st_size/1024:.0f} KB）\n")

    # ---- 关联键完整性检查 ----
    report.append("\n---\n\n## 关联键完整性检查\n")

    rt = dfs["region_time_data"]
    wt = dfs["workload_trace"]
    gi = dfs["GPU_information"]
    st = dfs["storage_information"]
    nl = dfs["network_latency"]

    # 1. 区域键覆盖
    report.append("### 1. 区域键覆盖\n")
    checks = [
        ("workload_trace.SourceRegion", sorted(wt["SourceRegion"].unique())),
        ("GPU_information.Region", sorted(gi["Region"].unique())),
        ("storage_information.Region", sorted(st["Region"].unique())),
        ("network_latency.FromRegion", sorted(nl["FromRegion"].unique())),
        ("network_latency.ToRegion", sorted(nl["ToRegion"].unique())),
        ("region_time_data.Region", sorted(rt["Region"].unique())),
    ]
    for name, vals in checks:
        missing = [r for r in REGIONS if r not in vals]
        extra = [v for v in vals if v not in REGIONS]
        status = "✓" if not missing and not extra else "✗"
        report.append(f"- {name}: {len(vals)} 个区域 {status}"
                      + (f"，缺失 {missing}" if missing else "")
                      + (f"，多余 {extra}" if extra else ""))

    # 2. 时域覆盖
    report.append("\n### 2. 时域覆盖\n")
    hours_rt = sorted(rt["Hour"].unique())
    arr_hours = sorted(wt["ArrivalHour"].unique())
    report.append(f"- region_time_data.Hour: min={hours_rt[0]} max={hours_rt[-1]} 共 {len(hours_rt)} 个（期望 0–2406 共 2407）")
    report.append(f"- workload_trace.ArrivalHour: min={arr_hours[0]} max={arr_hours[-1]} 共 {len(arr_hours)} 个（期望 0–2399 共 2400）")
    rt_missing = [h for h in NUMERIC_HOURS if h not in hours_rt]
    arr_missing = [h for h in range(0, 2400) if h not in arr_hours]
    report.append(f"- region_time_data 缺失小时: {rt_missing if rt_missing else '无 ✓'}")
    report.append(f"- workload_trace 缺失到达小时: {arr_missing if arr_missing else '无 ✓'}")

    # 3. 区域×小时完整网格
    rt_grid = set(zip(rt["Hour"], rt["Region"]))
    expect = {(h, r) for h in NUMERIC_HOURS for r in REGIONS}
    gap = expect - rt_grid
    report.append(f"- region_time_data 区域×小时网格: 实际 {len(rt_grid)} / 期望 {len(expect)}，缺口 {len(gap)}"
                  + (f"（示例: {sorted(gap)[:3]}）" if gap else " ✓"))

    # 4. 时延矩阵完整性（6×6 全对）
    pairs = set(zip(nl["FromRegion"], nl["ToRegion"]))
    all_pairs = {(a, b) for a in REGIONS for b in REGIONS}
    miss_pairs = all_pairs - pairs
    report.append(f"- network_latency 区域对: 实际 {len(pairs)} / 期望 {len(all_pairs)}"
                  + (f"，缺失 {sorted(miss_pairs)}" if miss_pairs else " ✓"))

    # 5. workload 各区域任务数与 LatestFinish 覆盖
    report.append(f"\n### 3. workload 任务分布\n")
    report.append(f"- 总任务数: {len(wt)}")
    report.append(f"- 类型分布: {wt['TaskType'].value_counts().to_dict()}")
    report.append(f"- 来源区域分布: {wt['SourceRegion'].value_counts().to_dict()}")
    lf = wt["LatestFinishHour"]
    report.append(f"- LatestFinishHour: min={lf.min()} max={lf.max()}，其中 =2406 的占比 {(lf == 2406).mean()*100:.1f}%")
    report.append(f"- GPU_Demand: min={wt['GPU_Demand'].min()} max={wt['GPU_Demand'].max()} 均值 {wt['GPU_Demand'].mean():.1f}")

    # ---- 量纲/单位初查 ----
    report.append("\n---\n\n## 变量量纲/单位初查（数值字段量级分布）\n")

    def describe(df, cols, label):
        report.append(f"\n### {label}\n")
        report.append("| 字段 | min | max | 均值 | 中位数 | 非零比例 |")
        report.append("|------|-----|-----|------|--------|----------|")
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() == 0:
                report.append(f"| {c} | — | — | — | — | 全空 |")
                continue
            report.append(f"| {c} | {s.min():.4g} | {s.max():.4g} | {s.mean():.4g} | {s.median():.4g} | {(s != 0).mean()*100:.1f}% |")

    describe(rt, [c for c in rt.columns if c not in ("Hour", "Region", "PricePeriod", "DemandResponseLevel", "DataPeriod")], "region_time_data 数值字段")
    describe(gi, ["Total_GPU", "Max_IT_Power_MW", "PUE", "Max_Facility_Power_MW", "Reserved_GPU_Ratio", "Available_GPU", "Max_Workload_GPUh_per_h"], "GPU_information")
    describe(st, ["StorageCapacity_MWh", "MinSOC_MWh", "InitialSOC_MWh", "MaxChargePower_MW", "MaxDischargePower_MW", "ChargeEfficiency", "DischargeEfficiency", "SellLimit_MW", "MaxGridImport_MW", "MaxGridExport_MW"], "storage_information")
    describe(wt, ["GPU_Demand", "EstimatedDuration_min", "MaxLatency_ms", "LatestFinishHour"], "workload_trace")
    describe(nl, ["NetworkLatency_ms"], "network_latency")
    describe(dfs["power_mapping"], ["GPU_Power_MW_per_EquivalentGPU"], "power_mapping")

    # 关键交叉校验：region_time_data 中 Baseline_AI_IT_Load 与 IT_Load 关系
    it_check = (rt["Baseline_AI_IT_Load_MW"] + rt["NonAI_IT_Load_MW"]).round(6) - rt["IT_Load_MW"]
    report.append(f"\n### 交叉校验\n")
    report.append(f"- Baseline_AI + NonAI 与 IT_Load 之差: max abs = {it_check.abs().max():.4f}（≈0 则口径自洽）")
    report.append(f"- Total_Load 与 IT_Load×PUE 之差: max abs = {(rt['Total_Load_MW'] - rt['IT_Load_MW']*gi.set_index('Region').loc[rt['Region'], 'PUE'].values).abs().max():.4f}（≈0 则口径自洽）")

    out_md = BASE / "solution" / "data-inventory-02.md"
    out_md.write_text("\n".join(report), encoding="utf-8")
    print(f"[OK] cache -> {cache_path}")
    print(f"[OK] report -> {out_md}")


if __name__ == "__main__":
    main()

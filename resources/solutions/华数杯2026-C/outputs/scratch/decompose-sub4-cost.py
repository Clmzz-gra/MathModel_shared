# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    S4 成本节省归因分解 — 将主解（基荷预填 + 层2 MILP + 储能 + 卖电）
    相对 B1 EDF 贪心基线的 86.9M（4.0%）成本差，拆解到具体策略：
    储能时间搬运 / 卖电收益 / MILP 任务调度选时。

原理：
    1. 复用已验证的层 2 每区独立 MILP（sub4-model.solve_region）跑
       逐项关闭实验，控制变量：
         A_main          = 主解（储能+卖电+MILP 全开）
         A_no_storage    = 禁储能（MaxCharge=MaxDischarge=Capacity=0）
         A_no_sell       = 禁卖电（SellLimit=0）
         A_no_both       = 双禁（储能+卖电全关）
       B1_EDF           = 纯启发式基线（无储能/无卖电/无 MILP，已有结果）
    2. 关闭实验全部用 free=True（无碳约束）求解——主解 ε=1.00 实测
       碳约束不绑定（main==free），与主解可比；每个场景自洽求其最优。
    3. 归因分解（成本差可加性检验）：
       储能贡献     = A_main − A_no_storage
       卖电贡献     = A_main − A_no_sell
       双禁增量     = A_main − A_no_both（对照单禁之和，检验可加性）
       调度贡献     = A_no_both − B1_EDF（同为无储能无卖电，差异纯来自
                       MILP 任务选时 + 基荷 EDF 固定 vs 基线 EDF 启发）
       总节省       = A_main − B1_EDF = 储能 + 卖电 + 调度 + 交叉项
    4. 输出每场景分区成本 + 聚合分解表（PR-014 核对）。

输入数据：
    - outputs/data/sub4-preprocessed.pkl（阶段 1.4）— tasks/power/storage
    - outputs/data/s3-preprocessed.pkl（阶段 1.4 S3）— panel 结清段实际值
    - outputs/data/c-data-cleaned.pkl（阶段 0.3）— region_time_data
    - outputs/data/s2-preprocessed.pkl（阶段 1.4 S2）— latency
    - outputs/data/s4-results.pkl（阶段 2.1）— 主解聚合（对照基准）
    - outputs/data/s4-baseline-heuristic.pkl — B1/B2 基线聚合
    - 中文指标 → 变量名映射：
      购电→G, 卖电→S, 新能源直供→R, 电网充电→Cg, 新能源充电→Cr,
      放电→D, SOC→E, 成本(M元)→cost_main_M, 碳(kt)→carbon_kt

输出：
    - outputs/data/s4-cost-decomposition.pkl — 键：
      scenarios: {A_main/A_no_storage/A_no_sell/A_no_both: agg}
      decomposition: {storage/sell/both/scheduling/total: M元 + %}
      meta: 口径说明
    - 控制台分解表

对应论文章节：
    问题四（S4）算-储-电协同优化 — §7.3 基荷策略合理性检验（Q3 对照）
"""
import copy
import importlib.util
import pickle
from pathlib import Path

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"
SCRATCH = BASE / "outputs" / "scratch"

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]

# 加载 sub4-model 模块（复用 solve_region / run_scenario 纯函数）
_spec = importlib.util.spec_from_file_location(
    "sub4model", SCRATCH / "sub4-model.py")
M4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M4)
# sub4-model.py 的 BASE 指向已删除的旧 worktree（e:\MathModel_pj-2026-C-sub3），
# 重定向到当前 worktree 的数据目录
M4.BASE = BASE
M4.DATA = DATA
solve_region = M4.solve_region
run_scenario = M4.run_scenario

# 加载数据（sub4-model.load 返回 (d, ext, nonai_full, absorb_full, latency)）
_load = M4.load


def scenario_storage(storage, no_storage=False, no_sell=False):
    """按开关返回深拷贝的 storage 参数。"""
    s = copy.deepcopy(storage)
    for r in REGIONS:
        if no_storage:
            s[r]["MaxChargePower_MW"] = 0.0
            s[r]["MaxDischargePower_MW"] = 0.0
            s[r]["Capacity_MWh"] = 0.0
            s[r]["MinSOC_MWh"] = 0.0
            s[r]["InitialSOC_MWh"] = 0.0
        if no_sell:
            s[r]["SellLimit_MW"] = 0.0
    return s


def main():
    d, ext, nonai_full, absorb_full, latency = _load()
    tasks = d["tasks"]
    power = d["power"]
    storage = d["storage"]

    # 主解聚合（对照组，ε=1.00 实测 == free）
    with open(DATA / "s4-results.pkl", "rb") as f:
        main_agg = pickle.load(f)["main"]
    # B1 EDF 基线聚合
    with open(DATA / "s4-baseline-heuristic.pkl", "rb") as f:
        b1_agg = pickle.load(f)["b1"]

    print("=" * 78)
    print("S4 成本归因分解 — 逐项关闭实验（free 求解，与主解可比）")
    print("=" * 78)
    print(f"主解 A_main   : 成本 {main_agg['cost_main_M']:.2f} M")
    print(f"B1 EDF 基线   : 成本 {b1_agg['cost_main_M']:.2f} M")

    scen = {}
    configs = [
        ("A_main", False, False),
        ("A_no_storage", True, False),
        ("A_no_sell", False, True),
        ("A_no_both", True, True),
    ]
    for name, no_storage, no_sell in configs:
        s_mod = scenario_storage(storage, no_storage, no_sell)
        print(f"\n--- {name} (no_storage={no_storage}, no_sell={no_sell}) ---")
        agg = run_scenario(tasks, power, s_mod, {}, nonai_full, absorb_full,
                           ext, latency, eps=1.00, label=name, free=True)
        scen[name] = agg
        print(f"  {name}: 成本 {agg['cost_main_M']:.2f} M | 碳 {agg['carbon_kt']:.2f} kt"
              f" | 峰值 {agg['peak_MW']:.1f} | 利用率 {agg['util_no_sell_pct']:.1f}%")

    # ---- 归因分解 ----
    c_main = scen["A_main"]["cost_main_M"]
    c_ns = scen["A_no_storage"]["cost_main_M"]
    c_nsl = scen["A_no_sell"]["cost_main_M"]
    c_nb = scen["A_no_both"]["cost_main_M"]
    c_b1 = b1_agg["cost_main_M"]

    d_storage = c_main - c_ns       # 储能贡献（含其引起的最优任务选时）
    d_sell = c_main - c_nsl         # 卖电贡献
    d_both = c_main - c_nb          # 双禁总贡献（与单禁之和比较）
    d_sched = c_nb - c_b1           # 调度贡献（MILP+基荷 vs 纯启发，无储能卖电）
    d_total = c_main - c_b1

    print("\n" + "=" * 78)
    print("成本归因分解（M元，正=主解节省）")
    print("=" * 78)
    rows = [
        ("储能贡献 (A_main − A_no_storage)", d_storage),
        ("卖电贡献 (A_main − A_no_sell)", d_sell),
        ("双禁合计 (A_main − A_no_both)", d_both),
        ("调度贡献 (A_no_both − B1)", d_sched),
        ("总节省   (A_main − B1)", d_total),
    ]
    for name, v in rows:
        print(f"{name:<38}{v:>10.2f}  M  ({v/c_b1*100:>5.2f}%)")

    out = {
        "scenarios": {k: v for k, v in scen.items()},
        "main_A": main_agg,
        "b1": b1_agg,
        "decomposition": {
            "storage_M": d_storage,
            "sell_M": d_sell,
            "both_M": d_both,
            "scheduling_M": d_sched,
            "total_M": d_total,
            "pct_base": c_b1,
        },
        "meta": {
            "generated": "2026-08-09",
            "source": "sub4-model.run_scenario（free 求解，与主解 ε=1.00 可比）"
                      "+ 逐项关闭储能/卖电",
            "note": "调度贡献 = A_no_both(无储能无卖电的 MILP 主解) − B1 EDF 基线，"
                    "纯来自任务调度选时差异（基荷 EDF 固定 + 4.7% MILP vs 全 EDF）",
        },
    }
    with open(DATA / "s4-cost-decomposition.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n已写入 {DATA / 's4-cost-decomposition.pkl'}")
    print("S4 COST DECOMPOSITION DONE")


if __name__ == "__main__":
    main()

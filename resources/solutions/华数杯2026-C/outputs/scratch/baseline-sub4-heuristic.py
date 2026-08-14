# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    S4 基荷策略合理性检验（Q3 对照）— 快速启发式调度基线：
    去掉基荷预填与层 2 MILP，用"电价感知贪心 + 新能源直供 + 购电补足"
    的纯启发式求解六指标，与主解（基荷预填 + 层2 MILP）对比，
    量化 EDF 贪心 vs 全局最优的偏差量级，判断 S4 方法是否合理。

原理：
    1. 问题：主解把 95.3% GPU-hour 交给层 1 EDF 贪心固定（基荷预填），
       层 2 MILP 只剩 ~5% 变量空间——MILP 的"优化"成分被稀释，需检验
       EDF 固定开工是否隐没了更优解。
    2. 对照设计（控制变量）：
       - 层 1 分配：复用主解 dest（口径一致，不改变分配）
       - 实时任务：到达即开工（同主解，物理约束不可优化）
       - 延迟容忍任务：全部由启发式调度，**不建 MILP 0-1 变量**——
         B1: EDF 最早可行开工（无电价意识，纯贪婪）
         B2: 电价最低可行开工（电价意识，绿色/低价时段优先）
       - 电力侧：新能源受限消纳（S3 B1 口径：UsedRenewable+RenewableCharge）
         优先直供，不足购电；**基线无储能、无卖电**——这是"纯启发式
         下界"的设计：主解除 EDF 贪心外还拥有储能时间搬运（六区 Cr/D
         实测全部活跃）与 D 区卖电收益（实测 24832 MWh），这些渠道
         基线全部缺失。故基线是**下界代理**，Δ=(B−A)/A 包含
         "EDF 贪心 + 储能 + 卖电 + 4.7% MILP" 的混合增量，**不能**把
         Δ 全部归因于基荷预填；干净检验（方案 A：无预填全任务 MILP）
         为 Q3 后续工作（见 approach-sub4 §7-5 / R3）。
       - 启发式不可行（GPU 容量冲突）时：任务在 [arrive, latest−dur] 内
         选第一个可行小时，宁超时不可行也记录（量化 QoS 下界）
    3. 六指标口径与主解完全一致（主时域 0-2399）：
       cost = Σ(G·price)/1e6（M元，无卖电收益项）
       carbon = Σ(G·ci)/1e3（kt）
       peak = max(net[:MAIN])，net = G − S（S=0）
       delay = GPU-hour 加权时延（source→dest，同 global_metrics）
       qos = 按时完成率（h0+dur ≤ latest）
       util = Σ(R+Cr+S)/Σ(Avail)（受限消纳利用率，S3 同口径）
    4. 输出主解 A 与基线 B1/B2 的对比表（成本/碳/峰值/时延/QoS/利用率），
       Δ = (B − A)/A 标注每个指标偏差，作为"基荷+MILP 增量价值"的证据。

输入数据：
    - outputs/data/sub4-preprocessed.pkl（阶段 1.4）
      tasks: id/type/source/cand/dest/baseload/start_h/arrive/dur/dem/
             latest/gh/power（dest 复用主解分配）
      power: 区域 → {price/sell/carbon/renewable:(2400,), pue/cap/...}
      storage: 区域 → 储能参数（基线不用，仅存档）
    - outputs/data/s3-preprocessed.pkl（阶段 1.4 S3）— panel 结清段实际值
    - outputs/data/c-data-cleaned.pkl（阶段 0.3）— region_time_data
      （NonAI 负荷 + 受限消纳上限）
    - outputs/data/s2-preprocessed.pkl（阶段 1.4 S2）— latency
    - outputs/data/s4-results.pkl（阶段 2.1）— 主解 main 聚合指标（对照组）
    - 中文指标 → 变量名映射：
      购电功率→G, 卖电功率→S, 新能源直供→R, 到达小时→arrive, 时长→dur,
      GPU需求→dem, 最晚完成→latest, GPU-hour→gh, 单位GPU功率(MW)→power,
      可用GPU→cap, 能效→pue, 受限消纳→absorb, 非AI负荷→NonAI

输出：
    - outputs/data/s4-baseline-heuristic.pkl — 键：
      b1: {agg(六指标), per: {r: {cost,carbon,peak,qos,delay,util,...}}}
      b2: {agg, per}
      compare: {A: 主解聚合, B1/B2: 基线聚合, delta_pct: 相对偏差}
      meta: 生成时间/口径说明
    - 控制台对比表（PR-014）

对应论文章节：
    问题四（S4）算-储-电协同优化 — §7.3 基荷策略合理性检验（Q3 对照）
"""
import pickle
from pathlib import Path

import numpy as np

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
NT = 2406          # 0-2405 全时域
MAIN = 2400        # 主时域 0-2399
TOL = 1e-6


def load():
    with open(DATA / "sub4-preprocessed.pkl", "rb") as f:
        d = pickle.load(f)
    with open(DATA / "s3-preprocessed.pkl", "rb") as f:
        s3 = pickle.load(f)
    with open(DATA / "c-data-cleaned.pkl", "rb") as f:
        cd = pickle.load(f)
    with open(DATA / "s2-preprocessed.pkl", "rb") as f:
        s2 = pickle.load(f)
    with open(DATA / "s4-results.pkl", "rb") as f:
        main = pickle.load(f)["main"]

    panel = s3["panel"]
    ext = {}
    for r in REGIONS:
        pr = panel.xs(r, level="Region")
        ext[r] = {
            "price": pr["Price_CNY_per_MWh"].values,          # (2406,)
            "carbon": pr["CarbonIntensity_tCO2_per_MWh"].values,
            "renewable": pr["AvailableRenewable_MW"].values,
        }
    rtd = cd["region_time_data"]
    nonai_full = {}
    absorb_full = {}
    for r in REGIONS:
        sub = rtd[rtd["Region"] == r].sort_values("Hour")
        nonai_full[r] = sub["NonAI_IT_Load_MW"].values[:NT].astype(float)
        absorb_full[r] = (sub["UsedRenewable_MW"].values[:NT]
                          + sub["RenewableCharge_MW"].values[:NT]).astype(float)
    return d, ext, nonai_full, absorb_full, s2["latency"], main


def schedule_greedy(r, tasks, power, nonai, mode, price):
    """启发式调度：返回 (aiit[NT], sched{id:h}, n_late) 。

    mode='edf'：窗口内最早可行开工（无电价意识）
    mode='price'：窗口内电价最低可行开工（电价意识，按价格排序早停）
    约束：GPU 容量（dem×overlap 累计 ≤ cap）。
    不可行任务：取窗口第一个小时（记录超时）。
    price：全时域电价数组（主时域实际值 + 结清段实际值，长度 NT）
    """
    p = power[r]
    cap_gpu = p["cap"]
    occ = np.zeros(NT, dtype=float)
    # 实时任务固定占用（到达即开工，跨收尾段）
    for t in tasks:
        if t["type"] == "RealTimeInference" and t["dest"] == r:
            h0, dur = t["arrive"], t["dur"]
            h_end = int(np.ceil(h0 + dur))
            for hh in range(int(np.floor(h0)), min(h_end, NT)):
                ov = min(h0 + dur, hh + 1.0) - max(h0, float(hh))
                occ[hh] += t["dem"] * ov

    def _feasible(h, dur, dem):
        """任务在整点小时 h 开工是否 GPU 可行（向量化 overlap 检查）。"""
        h_end = int(np.ceil(h + dur))
        hh = np.arange(h, h_end)
        ov = np.minimum(h + dur, hh + 1.0) - np.maximum(h, hh)
        return bool(np.all(occ[hh] + dem * ov <= cap_gpu + TOL))

    cands = [t for t in tasks
             if t["dest"] == r and t["type"] != "RealTimeInference"]
    sched = {}
    n_late = 0
    for t in sorted(cands, key=lambda t: -t["gh"]):
        dur = t["dur"]
        dem = t["dem"]
        lo = int(t["arrive"])
        # +1e-9 浮点保护（对齐 sub4-model L236）：dur 为 float，latest-dur
        # 可能恰为 2399.9999…，无保护则窗口少 1 个候选小时
        hi = int(min(t["latest"], NT) - dur + 1e-9)
        if hi < lo:
            hi = lo
        best = None
        if mode == "edf":
            for h in range(lo, hi + 1):
                if _feasible(h, dur, dem):
                    best = h
                    break
        else:  # price：按电价升序尝试，取第一个可行（电价感知贪心）
            order = np.argsort(price[lo:hi + 1], kind="stable")
            for k in order:
                h = lo + int(k)
                if _feasible(h, dur, dem):
                    best = h
                    break
        if best is None:
            best = lo  # 无可行 → 最早开工（可能超容量，GPU 占用照实记录）
            n_late += 1
        h = best
        h_end = int(np.ceil(h + dur))
        hh = np.arange(h, h_end)
        ov = np.minimum(h + dur, hh + 1.0) - np.maximum(h, hh)
        occ[hh] += dem * ov
        sched[t["id"]] = h
    # AI_IT 功率序列（IT 侧）
    aiit = np.zeros(NT, dtype=float)
    for t in cands:
        h = sched[t["id"]]
        h_end = int(np.ceil(h + t["dur"]))
        for hh in range(h, min(h_end, NT)):
            ov = min(h + t["dur"], hh + 1.0) - max(h, float(hh))
            aiit[hh] += t["dem"] * t["power"] * ov
    # 实时任务功率
    for t in tasks:
        if t["type"] == "RealTimeInference" and t["dest"] == r:
            h0, dur = t["arrive"], t["dur"]
            h_end = int(np.ceil(h0 + dur))
            for hh in range(int(np.floor(h0)), min(h_end, NT)):
                ov = min(h0 + dur, hh + 1.0) - max(h0, float(hh))
                aiit[hh] += t["dem"] * t["power"] * ov
    return aiit, sched, n_late


def metrics_region(r, aiit, nonai, absorb, avail, price, ci, pue, sched,
                   tasks, latency, n_late):
    """无储能启发式区域的指标（主时域 0-2399）。"""
    pue_fac = pue * (aiit + nonai)          # 设施侧需求
    R = np.minimum(pue_fac[:MAIN], absorb[:MAIN])  # 新能源直供（受限消纳）
    G = np.maximum(pue_fac[:MAIN] - R, 0.0)        # 购电补足
    cost = float(np.sum(G * price[:MAIN])) / 1e6
    carbon = float(np.sum(G * ci[:MAIN])) / 1e3
    peak = float(G.max())
    util = float(np.sum(R)) / float(avail[:MAIN].sum()) * 100
    return {"cost_main_M": cost, "carbon_kt": carbon, "peak_MW": peak,
            "util_no_sell_pct": util, "n_late": n_late}


def global_metrics(tasks, latency, region_sched):
    """跨区域：GPU-hour 加权时延 + QoS（按时完成率）。

    实时任务：到达即开工（h0=arrive，与主解 global_metrics 同口径）；
    延迟容忍任务：从 region_sched 取启发式开工小时。
    """
    total_gh = 0.0
    w_delay = 0.0
    n_done = 0
    for t in tasks:
        gh = t["gh"]
        total_gh += gh
        if t["type"] == "RealTimeInference":
            h0 = t["arrive"]
        else:
            h0 = region_sched.get(t["id"])
        if h0 is not None and h0 + t["dur"] <= t["latest"] + 1e-9:
            n_done += 1
        w_delay += gh * latency.get((t["source"], t["dest"]), 0.0)
    delay_ms = w_delay / total_gh if total_gh > 0 else 0.0
    qos = n_done / len(tasks) * 100
    return {"delay_ms": float(delay_ms), "qos_pct": float(qos)}


def run_baseline(d, ext, nonai_full, absorb_full, latency, mode):
    """跑全部 6 区域启发式，聚合六指标。"""
    tasks = d["tasks"]
    power = d["power"]
    region_sched = {}
    per = {}
    for r in REGIONS:
        p = power[r]
        # 全时域价格（主时域实际值 + 结清段实际值），长度 NT，供 B2 选时
        price = np.concatenate([p["price"], ext[r]["price"][MAIN:NT]])
        aiit, sched, n_late = schedule_greedy(r, tasks, power, nonai_full[r],
                                              mode, price)
        region_sched.update(sched)
        # 结清段碳强度/新能源拼接（指标只用主时域；价格已在上方拼接）
        ci = np.concatenate([p["carbon"], ext[r]["carbon"][MAIN:NT]])
        avail = np.concatenate([p["renewable"], ext[r]["renewable"][MAIN:NT]])
        per[r] = metrics_region(r, aiit, nonai_full[r], absorb_full[r], avail,
                                price, ci, p["pue"], sched, tasks, latency, n_late)
        print(f"  {r}: 成本 {per[r]['cost_main_M']:8.2f} M | 碳 {per[r]['carbon_kt']:8.2f}"
              f" kt | 峰值 {per[r]['peak_MW']:6.1f} | 利用率 {per[r]['util_no_sell_pct']:5.1f}%"
              f" | 无可行取lo {n_late}")
    gm = global_metrics(tasks, latency, region_sched)
    agg = {
        "mode": mode,
        "cost_main_M": sum(v["cost_main_M"] for v in per.values()),
        "carbon_kt": sum(v["carbon_kt"] for v in per.values()),
        "peak_MW": max(v["peak_MW"] for v in per.values()),
        "util_no_sell_pct": float(np.mean([v["util_no_sell_pct"] for v in per.values()])),
        "delay_ms": gm["delay_ms"],
        "qos_pct": gm["qos_pct"],
        "n_late": sum(v["n_late"] for v in per.values()),
    }
    return agg, per, region_sched


def fmt_agg(a):
    return (f"成本{a['cost_main_M']:.2f}M 碳{a['carbon_kt']:.2f}kt "
            f"峰值{a['peak_MW']:.1f}MW 时延{a['delay_ms']:.1f}ms "
            f"QoS{a['qos_pct']:.1f}% 利用率{a['util_no_sell_pct']:.1f}%")


def main():
    d, ext, nonai_full, absorb_full, latency, main_agg = load()
    print("=" * 78)
    print("S4 Q3 基荷策略合理性检验 — 启发式基线（无基荷预填 / 无 MILP）")
    print("=" * 78)
    tasks = d["tasks"]
    print(f"任务数 {len(tasks)} | 主解聚合: {fmt_agg(main_agg)}")

    results = {}
    for mode, label in [("edf", "B1-EDF"), ("price", "B2-电价贪心")]:
        print(f"\n--- {label} 启发式调度 ---")
        agg, per, sched = run_baseline(d, ext, nonai_full, absorb_full,
                                       latency, mode)
        results[label] = {"agg": agg, "per": per, "sched": sched}
        print(f"  聚合: {fmt_agg(agg)} | 无可行取lo {agg['n_late']}")

    # ---- 对比表（主解 A vs 基线 B） ----
    print("\n" + "=" * 78)
    print("对比表（主时域六指标）")
    print("=" * 78)
    hdr = (f"{'指标':<12}{'A 主解':>14}{'B1 EDF':>14}{'B2 电价':>14}"
           f"{'ΔB1%':>10}{'ΔB2%':>10}")
    print(hdr)
    rows = [
        ("成本(M)", "cost_main_M", lambda v: v),
        ("碳(kt)", "carbon_kt", lambda v: v),
        ("峰值(MW)", "peak_MW", lambda v: v),
        ("时延(ms)", "delay_ms", lambda v: v),
        ("QoS(%)", "qos_pct", lambda v: v),
        ("利用率(%)", "util_no_sell_pct", lambda v: v),
    ]
    for name, key, _ in rows:
        av = main_agg[key]
        b1 = results["B1-EDF"]["agg"][key]
        b2 = results["B2-电价贪心"]["agg"][key]
        d1 = (b1 - av) / av * 100 if av else float("nan")
        d2 = (b2 - av) / av * 100 if av else float("nan")
        print(f"{name:<12}{av:>14.2f}{b1:>14.2f}{b2:>14.2f}{d1:>10.1f}{d2:>10.1f}")

    out = {
        "main_A": main_agg,
        "b1": results["B1-EDF"]["agg"],
        "b1_per": results["B1-EDF"]["per"],
        "b2": results["B2-电价贪心"]["agg"],
        "b2_per": results["B2-电价贪心"]["per"],
        "meta": {
            "generated": "2026-08-09",
            "source": "sub4-preprocessed（dest 复用主解）+ 无基荷/无MILP 启发式"
                      "（EDF 与电价感知两档）+ 受限消纳（S3 B1）",
            "note": "基线无储能、无卖电（纯启发式下界）：主解六区 Cr/D 实测全部活跃、"
                    "D 区卖电 24832 MWh，Δ 包含 EDF 贪心+储能+卖电+MILP 混合增量，"
                    "不可全部归因基荷预填；任务窗口同主解 MILP 非基荷候选"
                    "（可进 2400-2405 结清段，主解基荷任务被 preprocess 限制在主时域）",
        },
    }
    with open(DATA / "s4-baseline-heuristic.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n已写入 {DATA / 's4-baseline-heuristic.pkl'}")
    print("S4 BASELINE DONE")


if __name__ == "__main__":
    main()

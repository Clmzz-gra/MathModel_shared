# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    建模裁定（sensitivity-minutil-sub2-20260808.md）— S2 区域最低利用率灵敏度分析：
    A/B 保底承接 0/5/10/15% 容量时，层 1 分配变化 + 层 2 正式 MILP 调度成本，
    产出"最低利用率 vs 总成本"曲线，佐证 A/B 空置是成本最优边界而非模型缺陷

原理：
    1. 保底预分配（外部实现，不改 sub2-model.py）：候选含 A/B 的任务按
       "回迁增量" delta = cost(t,A/B) - min_cost(cand) 升序，优先把
       "强制启用 A/B 代价最小"的任务分给 A/B，直到各达 min_util×CAP_GH
       （GPU-hour 口径，建模裁定 2）
    2. 剩余任务走与 capacity_aware_assign 完全同构的容量感知贪心
       （90% 阈值 + 退路）——0% 档输出必须与现有 dest0 逐任务一致
       （验证点：退路 117、承接量 C/D/E/F）
    3. 层 2：schedule_dest 复用（dest 指纹缓存；0% 档命中现有
       s2_sched_a080c48e47f1 秒级，5/10/15% 各 1 次新调度 ~16min）
    4. 断点保护：每档完成即写 s2-minutil-results.pkl，崩溃重跑跳过已完成档

输入数据：
    - outputs/data/s2-preprocessed.pkl（经 sub2-model 加载）
    - sub2-model.py：cost_of/mean_price/CAP_GH/THRESHOLD/schedule_dest/
      plot 基础设施（importlib 加载）
    - outputs/data/s2-results.pkl（C0/E0 基线参照，画相对成本用）

输出：
    - outputs/data/s2-minutil-results.pkl — 各档 {min_util, C, E, assign, fail}
    - outputs/figures/sub2-minutil-curve.pdf — 最低利用率 vs 总成本曲线（0/5/10/15%，
      标注成本增量，0% 档标注"成本最优边界"）
    - 控制台各档成本/碳/承接统计

对应论文章节：
    问题二（S2）碳感知任务调度 — 敏感性分析（区域最低利用率 vs 成本，建模裁定 5 叙事）
"""
import argparse
import importlib.util
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"e:\MathModel_pj-2026-C")
MODEL_PY = BASE / "outputs" / "scratch" / "sub2-model.py"

# 中文字体（chart-generator 规范）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

MINUTILS = [0.0, 0.05, 0.10, 0.15]
RESULTS_PKL = BASE / "outputs" / "data" / "s2-minutil-results.pkl"


def load_model():
    """sub2-model.py 含连字符无法直接 import，用 importlib 按路径加载"""
    spec = importlib.util.spec_from_file_location("sub2_model", MODEL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load_model()


def assign_minutil(min_util):
    """保底预分配 + 容量感知贪心（与 sub2-model.capacity_aware_assign 同构）。
    0% 档输出必须与 capacity_aware_assign 逐任务一致（口径验证）。
    返回 (dest, demand, assign, fail, cost0, co2_0, cost_new, co2_new)"""
    tgt = {r: min_util * m.CAP_GH[r] for r in ("RegionA", "RegionB")}
    demand = {r: 0.0 for r in m.REGIONS}
    assign = {r: 0 for r in m.REGIONS}
    dest = {}
    fail = 0
    cost0 = co2_0 = cost_new = co2_new = 0.0
    # ---- 保底预分配（min_util > 0 时生效）----
    if min_util > 0:
        pool = []
        for t in m.tasks:
            has_a = "RegionA" in t["cand"]
            has_b = "RegionB" in t["cand"]
            if not (has_a or has_b):
                continue
            base = min(m.cost_of(t, c) for c in t["cand"])
            dA = m.cost_of(t, "RegionA") - base if has_a else float("inf")
            dB = m.cost_of(t, "RegionB") - base if has_b else float("inf")
            pool.append((min(dA, dB), t, dA, dB))
        pool.sort(key=lambda x: x[0])  # 回迁增量最小优先
        for _, t, dA, dB in pool:
            if demand["RegionA"] >= tgt["RegionA"] and demand["RegionB"] >= tgt["RegionB"]:
                break
            opts = []
            if "RegionA" in t["cand"] and demand["RegionA"] < tgt["RegionA"]:
                opts.append(("RegionA", dA))
            if "RegionB" in t["cand"] and demand["RegionB"] < tgt["RegionB"]:
                opts.append(("RegionB", dB))
            if not opts:
                continue
            r = min(opts, key=lambda x: x[1])[0]
            dest[t["id"]] = r
            demand[r] += t["gh"]
            assign[r] += 1
    # ---- 剩余任务容量感知贪心（与 capacity_aware_assign 同构）----
    for t in m.tasks:
        if t["id"] in dest:
            r = dest[t["id"]]
            cost0 += m.cost_of(t, t["source"])
            co2_0 += m.co2_of(t, t["source"])
            cost_new += m.cost_of(t, r)
            co2_new += m.co2_of(t, r)
            continue
        cand = sorted(t["cand"], key=lambda r: m.cost_of(t, r))
        placed = False
        for r in cand:
            if demand[r] + t["gh"] <= m.CAP_GH[r] * m.THRESHOLD:
                demand[r] += t["gh"]
                assign[r] += 1
                best = r
                placed = True
                break
        if not placed:
            best = min(cand, key=lambda r: demand[r] / m.CAP_GH[r])
            demand[best] += t["gh"]
            assign[best] += 1
            fail += 1
        dest[t["id"]] = best
        cost0 += m.cost_of(t, t["source"])
        co2_0 += m.co2_of(t, t["source"])
        cost_new += m.cost_of(t, best)
        co2_new += m.co2_of(t, best)
    return dest, demand, assign, fail, cost0, co2_0, cost_new, co2_new


def run_minutil(min_util, out):
    """单档：保底分配 → 层2 调度（缓存）→ 记录。断点：完成即落盘"""
    for r in out:
        if abs(r["min_util"] - min_util) < 1e-9:
            print(f"[跳过] min_util={min_util:.0%} 已跑（C={r['C']/1e6:.1f}M）")
            return out
    t0 = time.time()
    dest, demand, assign, fail, c0, k0, cn, kn = assign_minutil(min_util)
    print(f"[min_util={min_util:.0%}] 分配完成：退路 {fail}，A/B 承接 "
          f"{assign['RegionA']}/{assign['RegionB']} 任务，"
          f"A/B GPU-hour {demand['RegionA']:,.0f}/{demand['RegionB']:,.0f}")
    sd = m.schedule_dest(dest, label=f"minutil={min_util:.0%}")
    out.append({"min_util": min_util, "C": sd["C"], "E": sd["E"],
                "assign": assign, "demand": demand, "fail": fail,
                "dt": time.time() - t0})
    with open(RESULTS_PKL, "wb") as f:
        pickle.dump(out, f)
    print(f"[min_util={min_util:.0%}] 成本 {sd['C']/1e6:.1f}M 元, "
          f"碳 {sd['E']/1e3:.2f}kt（耗时 {time.time()-t0:.0f}s）")
    return out


def plot_minutil_curve(out, C0, fig_dir):
    """最低利用率 vs 总成本曲线（0% 标注成本最优边界；各档标成本增量）。
    标注布局防遮挡：三个增量标注统一放点左上侧（深色，非红），0% 单行标
    右侧；图注说明"标注 = 相对 0% 基线的成本增量"，含义清晰"""
    pts = sorted(out, key=lambda r: r["min_util"])
    xs = [r["min_util"] * 100 for r in pts]
    ys = [r["C"] / 1e6 for r in pts]
    base = ys[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, "o-", color="#4c72b0", lw=2, ms=7)
    incs = [(y - base) / base * 100 for y in ys[1:]]
    # 0%/+0.1%/+0.9% 三个标注：0% 放点正上方居中，其余两个放点左上侧
    for i, (x, y, inc) in enumerate(zip(xs[:-1], ys[:-1], [None] + incs[:-1])):
        lab = f"+{inc:.1f}%" if inc is not None else "0% 空置"
        xy, ha, va = ((-0, 12), "center", "bottom") if i == 0 else ((-12, 10), "right", "bottom")
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=xy, ha=ha, va=va,
                    fontsize=9, color="#1a1a1a")
    # +2.4%（15% 点）：放点正左（垂直居中，避开标题区）
    ax.annotate(f"+{incs[-1]:.1f}%", (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(-14, 0), ha="right", va="center",
                fontsize=9, color="#1a1a1a")
    ax.set_xlabel("A/B 最低利用率 (%)")
    ax.set_ylabel("总成本 (M 元)")
    ax.set_title("区域最低利用率 vs 总成本（A/B 保底承接灵敏度）")
    ax.grid(alpha=0.3)
    ax.margins(0.08)
    fig.tight_layout()
    # 图注：标注含义 + 空置合理性说明（不贴数据点，防遮挡）
    fig.text(0.5, -0.02,
             "标注为相对 0%（空置）基线的总成本增量；A/B 空置为成本最优边界，"
             "强制启用至 15% 仅使总成本 +2.4%",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(fig_dir / "sub2-minutil-curve.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[图] sub2-minutil-curve.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["0", "5", "10", "15", "all"], default="all")
    args = ap.parse_args()
    want = [0.0, 0.05, 0.10, 0.15] if args.stage == "all" else [float(args.stage) / 100]
    out = pickle.load(open(RESULTS_PKL, "rb")) if RESULTS_PKL.exists() else []
    # 0% 口径验证：与现有 capacity_aware_assign 一致（退路 117 / 承接量）
    if any(abs(r["min_util"]) < 1e-9 for r in out):
        pass
    else:
        d0, dm0, a0, f0, *_ = assign_minutil(0.0)
        d_ref, dm_ref, a_ref, f_ref, *_ = m.capacity_aware_assign()
        same_dest = all(d0[t["id"]] == d_ref[t["id"]] for t in m.tasks)
        print(f"[0% 口径验证] dest 逐任务一致={same_dest}, 退路 {f0} vs {f_ref}, "
              f"承接 {a0} vs {a_ref}")
        if not same_dest or f0 != f_ref or a0 != a_ref:
            raise SystemExit("❌ 0% 档与 capacity_aware_assign 不一致，中止")
    for mu in want:
        out = run_minutil(mu, out)
    with open(BASE / "outputs" / "data" / "s2-results.pkl", "rb") as f:
        s2 = pickle.load(f)
    plot_minutil_curve(out, s2["C0"], BASE / "outputs" / "figures")
    with open(RESULTS_PKL, "wb") as f:
        pickle.dump(out, f)
    print("\n[OK] minutil 灵敏度完成，s2-minutil-results.pkl + 曲线图已产出")


if __name__ == "__main__":
    main()

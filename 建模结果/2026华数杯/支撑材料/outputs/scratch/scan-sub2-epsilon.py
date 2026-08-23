# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    任务 A（handoff §7 人类裁定）— ε 紧档位补充扫描：E_min 碳最小化单目标、
    秒级宽松档、紧档位成本-碳权衡曲线，并更新 s2-results.pkl / 出图 / 表

原理：
    1. E_min：dest_carbon_min（候选内最低碳区 + 100% 容量约束）→ 层2
       obj="co2"（碳排最小单目标）调度 → 精确碳排理论下界
    2. 档位扫描：run_eps(eps_kt) 复用 run_eta 同构迭代（成本最小 + 碳上限
       ε-约束；收敛 E≤eps 或偏差<0.5%；让渡 reassign_round 碳强度降序批量；
       3 轮上限、兜底 ε+1%）。首轮命中 dest0 指纹缓存（秒级）；
       让渡后新 dest 调度由 s2_sched_{fp}.pkl 缓存承接 → 各档让渡链
       （确定性 reassign）天然共享，越跑越快
    3. 档位过滤：eps ≥ E_min 才跑（低于下界标注不可达跳过，防 3 轮兜底浪费）
    4. 断点保护：run_eps 每轮写 s2_eps_{eps}_partial.pkl，崩溃重跑自动续

输入数据：
    - outputs/data/s2-preprocessed.pkl（阶段 1.4，经 sub2-model import）
    - outputs/data/s2-results.pkl（原 η 三档结果，追加 eps_results/emin）
    - outputs/data/cache/s2_sched_*.pkl（dest 指纹缓存，宽松档命中复用）

输出：
    - outputs/data/s2-results.pkl — 追加 emin（E_min/C_at_Emin）+ eps_results
      （各紧档 eps_kt/C/E/迭代记录）
    - outputs/figures/sub2-epsilon-curve.pdf — 多档曲线（含 E_min 点）
    - 控制台各档成本/碳/收敛状态

对应论文章节：
    问题二（S2）碳感知任务调度 — ε-约束敏感性分析（任务 A 补充扫描）
"""
import argparse
import importlib.util
import pickle
import sys
import time
from pathlib import Path

BASE = Path(r"e:\MathModel_pj-2026-C")
MODEL_PY = BASE / "outputs" / "scratch" / "sub2-model.py"


def load_model():
    """文件名 sub2-model.py 含连字符无法直接 import，用 importlib 按路径加载。
    顶层数据加载（读 s2-preprocessed.pkl）在 exec_module 时执行一次"""
    spec = importlib.util.spec_from_file_location("sub2_model", MODEL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load_model()

# 档位候选（§7 人类裁定）：350/320/290/270/251/240/230/220 kt → 内部 kg
EPS_CANDIDATES = [350e3, 320e3, 290e3, 270e3, 251e3, 240e3, 230e3, 220e3]
RESULTS_PKL = BASE / "outputs" / "data" / "s2-results.pkl"


def load_results():
    with open(RESULTS_PKL, "rb") as f:
        return pickle.load(f)


def save_results(out):
    with open(RESULTS_PKL, "wb") as f:
        pickle.dump(out, f)


def run_emin(out):
    """E_min：碳最小化分配 + obj=co2 调度（指纹独立缓存，可续）"""
    if "emin" in out:
        print(f"[E_min 已存在] E_min = {out['emin']['E']/1e3:.2f} kt（跳过）")
        return out
    t0 = time.time()
    dest, demand, assign, fail = m.dest_carbon_min()
    print(f"碳最小化分配：退路 {fail}，承接 {assign}")
    sd = m.schedule_dest(dest, label="emin", obj="co2")
    emin = {"E": sd["E"], "C": sd["C"], "fail": fail, "assign": assign,
            "dt": time.time() - t0}
    out["emin"] = emin
    save_results(out)
    print(f"[E_min] 碳排下界 E_min = {emin['E']/1e3:.2f} kt, "
          f"对应成本 {emin['C']/1e6:.1f} M 元（耗时 {emin['dt']:.0f}s）")
    return out


def run_levels(out, eps_list, tag):
    """跑指定档位列表（kg 值；run_eps 断点 + sched 缓存可续）"""
    E0 = out["E0"]
    if "eps_results" not in out:
        out["eps_results"] = []
    done = {r["eps_kg"] for r in out["eps_results"]}
    for eps in eps_list:
        if eps in done:
            print(f"[跳过] ε={eps/1e3:.0f}kt 已跑（E={next(r for r in out['eps_results'] if r['eps_kg']==eps)['E']/1e3:.2f}kt）")
            continue
        res = m.run_eps(eps, E0, label=tag)
        out["eps_results"].append(res)
        save_results(out)
        print(f"[{tag}] ε={eps/1e3:.0f}kt → 成本 {res['C']/1e6:.1f}M, 碳 {res['E']/1e3:.2f}kt "
              f"({len(res['iters'])} 轮, {'收敛' if res['iters'][-1]['converged'] else '兜底'})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["emin", "quick", "tight", "all"],
                    default="all", help="emin: 仅 E_min；quick: 宽松档（缓存命中）；"
                                        "tight: 紧档位（新调度）；all: 全部")
    args = ap.parse_args()
    out = load_results()
    C0, E0 = out["C0"], out["E0"]
    print(f"现有结果：C0={C0/1e6:.1f}M, E0={E0/1e3:.2f}kt, "
          f"η 档 {len(out['eta_results'])} 个")

    if args.stage in ("emin", "all"):
        out = run_emin(out)

    if args.stage in ("quick", "all"):
        # 宽松档：ε ≥ 自由解（251.78）→ 首轮收敛，dest0 缓存命中秒级
        quick = [e for e in EPS_CANDIDATES if e >= out["eta_results"][0]["E"]]
        out = run_levels(out, quick, "quick")

    if args.stage in ("tight", "all"):
        # 紧档位需 E_min 过滤（低于下界不可达）；未算则先跑
        if "emin" not in out:
            out = run_emin(out)
        E_min = out["emin"]["E"]
        tight = [e for e in EPS_CANDIDATES
                 if e < out["eta_results"][0]["E"] and e >= E_min - 1e-9]
        skipped = [e for e in EPS_CANDIDATES if e < E_min - 1e-9]
        if skipped:
            print(f"[跳过] 档位 {skipped} < E_min={E_min/1e3:.2f}kt（不可达，标注论文）")
        out = run_levels(out, tight, "tight")

    # 出图（多档曲线 + E_min 点；E_min 未跑时仅画已有档）
    emin_pt = {"E": out["emin"]["E"], "C": out["emin"]["C"]} if "emin" in out else None
    m.plot_epsilon(out["eta_results"], C0, E0, BASE / "outputs" / "figures",
                   extra=out.get("eps_results", []), emin=emin_pt)
    save_results(out)
    print("\n[OK] scan 完成，s2-results.pkl 已更新 + 多档图已出")


if __name__ == "__main__":
    main()

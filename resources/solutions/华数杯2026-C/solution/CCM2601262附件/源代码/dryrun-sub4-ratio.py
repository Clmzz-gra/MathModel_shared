# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    《S4 降低基荷实验计划》可行性 dry-run — 模拟 BASELOAD_GH_RATIO=0.7
    档的真实效果：释放任务数/释放 GH/改派去向/最终进 MILP 的任务数与
    候选窗口规模，不覆盖现状 sub4-preprocessed.pkl（落盘到 scratch）。

原理：
    1. 完整复现 preprocess-sub4.py 的基荷预填主循环（EDF + 配额），
       仅新增停止条件：gh_filled ≥ BASELOAD_GH_RATIO × cap_gh[r] 时，
       剩余候选全部释放（baseload=False, start_h=None, 进 failed_all）。
    2. 释放任务走既有第二轮改派通道（候选内负载率最低且未满 90%，
       EDF 试填），仍失败者进 MILP —— 与现状闭环一致。
    3. 统计最终进 MILP 任务数、候选窗口 n_x 总和，预判层2 求解规模
       （现状 n_x=223 / 1.1s，见 s4-results.pkl）。

输入数据：
    - outputs/data/s2-preprocessed.pkl（同 preprocess-sub4）
    - outputs/data/s3-preprocessed.pkl（同 preprocess-sub4）
    - outputs/data/c-data-cleaned.pkl（同 preprocess-sub4）

输出：
    - outputs/scratch/sub4-preprocessed-ratio070.pkl — 模拟版（不覆盖现状）
    - 控制台统计量（释放/改派/进MILP，PR-014）

对应论文章节：
    问题四（S4）— 基荷预填比例敏感性实验（计划 dry-run）
"""
import pickle
from pathlib import Path

import numpy as np

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"
OUT = BASE / "outputs" / "scratch"

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
H = 2400
T_END = 2406
P25_QUANTILE = 25
BASELOAD_THRESHOLD = 0.9
BASELOAD_GH_RATIO = 0.7   # 实验参数：基荷预填占区域 GPU 容量的比例上限


def earliest_feasible_hour(t, r, occ_gpu, occ_aiit, cap_gpu, quota_arr):
    """同 preprocess-sub4：找最早可行开工小时（EDF）。"""
    lo = int(t['arrive'])
    hi = int(min(t['latest'], T_END) - t['dur'])
    hi = min(hi, H - int(np.ceil(t['dur'])))
    if hi < lo:
        return None
    for h in range(lo, hi + 1):
        h_end = int(np.ceil(h + t['dur']))
        ok = True
        for hh in range(h, h_end):
            overlap = min(h + t['dur'], hh + 1) - max(h, hh)
            if occ_gpu[hh] + t['dem'] * overlap > cap_gpu + 1e-6:
                ok = False
                break
            if occ_aiit[hh] + t['dem'] * t['power'] * overlap > quota_arr[hh] + 1e-6:
                ok = False
                break
        if ok:
            return h
    return None


def commit_placement(t, h, occ_gpu, occ_aiit):
    h_end = int(np.ceil(h + t['dur']))
    for hh in range(h, h_end):
        overlap = min(h + t['dur'], hh + 1) - max(h, hh)
        occ_gpu[hh] += t['dem'] * overlap
        occ_aiit[hh] += t['dem'] * t['power'] * overlap


def main():
    print("=" * 78)
    print(f"S4 基荷比例 dry-run: BASELOAD_GH_RATIO = {BASELOAD_GH_RATIO}")
    print("=" * 78)

    with open(DATA / "s2-preprocessed.pkl", "rb") as f:
        s2 = pickle.load(f)
    with open(DATA / "s3-preprocessed.pkl", "rb") as f:
        s3 = pickle.load(f)
    with open(DATA / "c-data-cleaned.pkl", "rb") as f:
        cd = pickle.load(f)

    tasks = s2['tasks']
    power = s2['power']
    storage = s3['storage']
    carbon_base_kt = s3['carbon_base_kt']
    type_maxlat = s2['type_maxlat']
    p_train = s2['power_mapping']['AITraining']

    rtd = cd['region_time_data']
    nonai_arr = {r: rtd[(rtd['Region'] == r) & (rtd['Hour'] < H)]
                 .sort_values('Hour')['NonAI_IT_Load_MW'].values.astype(float)
                 for r in REGIONS}

    # ---- 层1 贪心分配（与原版逐字一致） ----
    assigned_gh = {r: 0.0 for r in REGIONS}
    cap_gh = {r: power[r]['cap'] * H for r in REGIONS}
    mean_price = {r: float(np.mean(power[r]['price'])) for r in REGIONS}
    n_fallback = 0
    for t in tasks:
        cand_sorted = sorted(t['cand'], key=lambda r: power[r]['pue'] * mean_price[r])
        dest = None
        for r in cand_sorted:
            if assigned_gh[r] + t['gh'] <= BASELOAD_THRESHOLD * cap_gh[r]:
                dest = r
                break
        if dest is None:
            dest = min(cand_sorted, key=lambda r: assigned_gh[r] / cap_gh[r])
            n_fallback += 1
        t['dest'] = dest
        assigned_gh[dest] += t['gh']

    # ---- 配额 ----
    p25 = {}
    quota_arr = {}
    quota_mean = {}
    for r in REGIONS:
        a = power[r]['renewable']
        p25[r] = float(np.percentile(a, P25_QUANTILE))
        quota_arr[r] = np.maximum(p25[r] / power[r]['pue'] - nonai_arr[r], 0.0)
        quota_mean[r] = float(quota_arr[r].mean())

    occ_gpu = {r: np.zeros(H, dtype=float) for r in REGIONS}
    occ_aiit = {r: np.zeros(H, dtype=float) for r in REGIONS}
    for t in tasks:
        if t['type'] == 'RealTimeInference':
            r = t['dest']
            h0 = t['arrive']
            dur = t['dur']
            for hh in range(int(np.floor(h0)), int(np.ceil(h0 + dur))):
                if hh < H:
                    overlap = min(h0 + dur, hh + 1) - max(h0, hh)
                    occ_gpu[r][hh] += t['dem'] * overlap
                    occ_aiit[r][hh] += t['dem'] * t['power'] * overlap

    cand_r = {r: sorted(
        [t for t in tasks if t['dest'] == r and t['type'] != 'RealTimeInference'],
        key=lambda t: -t['gh']) for r in REGIONS}

    baseload_meta = {}
    total_reassigned = 0
    n_released = 0
    gh_released = 0.0
    released_tasks = []   # 记录释放任务（clean-test 窗口统计用）
    failed_all = []
    for r in REGIONS:
        cap_gpu = power[r]['cap']
        cap_gh_r = cap_gpu * H
        slack_gpu = float(cap_gh_r - occ_gpu[r].sum())
        slack_aiit = float(quota_arr[r].sum() - occ_aiit[r].sum())
        n_fill = 0
        gh_filled = 0.0
        for i, t in enumerate(cand_r[r]):
            # 【dry-run 新增】按 GPU-hour 比例的停止条件（计划 §3.1）
            if gh_filled >= BASELOAD_GH_RATIO * cap_gh_r:
                for t_rest in cand_r[r][i:]:
                    t_rest['baseload'] = False
                    t_rest['start_h'] = None
                    failed_all.append(t_rest)
                    released_tasks.append(t_rest)
                n_released += len(cand_r[r]) - i
                gh_released += sum(t2['gh'] for t2 in cand_r[r][i:])
                break
            if slack_gpu < t['gh'] or slack_aiit < t['gh'] * t['power']:
                t['baseload'] = False
                t['start_h'] = None
                failed_all.append(t)
                continue
            h = earliest_feasible_hour(t, r, occ_gpu[r], occ_aiit[r],
                                       cap_gpu, quota_arr[r])
            if h is not None:
                commit_placement(t, h, occ_gpu[r], occ_aiit[r])
                slack_gpu -= t['gh']
                slack_aiit -= t['gh'] * t['power']
                t['baseload'] = True
                t['start_h'] = h
                n_fill += 1
                gh_filled += t['gh']
            else:
                t['baseload'] = False
                t['start_h'] = None
                failed_all.append(t)
        theory_gh = quota_mean[r] / p_train * H
        baseload_meta[r] = {
            'n_candidates': len(cand_r[r]), 'n_filled': n_fill,
            'n_failed': len(cand_r[r]) - n_fill, 'gh_filled': gh_filled,
            'theoretical_quota_gh': theory_gh,
            'fill_rate_vs_quota': gh_filled / theory_gh if theory_gh > 0 else 0.0,
        }
        pct = n_fill / len(cand_r[r]) * 100 if cand_r[r] else 0.0
        print(f"  {r}: 候选 {len(cand_r[r]):5d} | 基荷填充 {n_fill:5d} ({pct:4.1f}%)"
              f" | GH {gh_filled:12,.0f} / 70%界 {0.7*cap_gh_r:12,.0f}")

    # ---- 第二轮改派（与原版逐字一致） ----
    print(f"\n第二轮改派：{len(failed_all)} 个不可行/释放任务")
    for t in failed_all:
        old_dest = t['dest']
        alt = sorted(
            [r for r in t['cand']
             if r != old_dest and assigned_gh[r] + t['gh'] <= BASELOAD_THRESHOLD * cap_gh[r]],
            key=lambda r: assigned_gh[r] / cap_gh[r])
        moved = False
        for r in alt:
            h = earliest_feasible_hour(t, r, occ_gpu[r], occ_aiit[r],
                                       power[r]['cap'], quota_arr[r])
            if h is not None:
                commit_placement(t, h, occ_gpu[r], occ_aiit[r])
                assigned_gh[old_dest] -= t['gh']
                assigned_gh[r] += t['gh']
                t['dest'] = r
                t['baseload'] = True
                t['start_h'] = h
                moved = True
                total_reassigned += 1
                baseload_meta[r]['n_filled'] += 1
                baseload_meta[r]['gh_filled'] += t['gh']
                baseload_meta[old_dest]['n_failed'] -= 1
                break
        if not moved:
            t['baseload'] = False
            t['start_h'] = None

    still_failed = [t for t in failed_all if not t['baseload']]
    print(f"改派成功 {total_reassigned} | 仍失败 {len(still_failed)}（进 MILP）")

    # ---- 进 MILP 任务候选窗口规模 ----
    n_x_tot = 0
    for t in still_failed:
        lo = int(t['arrive'])
        hi = int(min(t['latest'], T_END) - t['dur'] + 1e-9)
        w = max(hi - lo + 1, 1)
        n_x_tot += w
    print(f"进 MILP 任务候选变量 n_x 合计 = {n_x_tot:,d}"
          f"（现状 223 / 求解 1.1s）")

    # ---- clean-test 口径：若释放任务跳过改派直进 MILP ----
    n_x_clean = 0
    ws_clean = []
    for t in released_tasks:
        lo = int(t['arrive'])
        hi = int(min(t['latest'], T_END) - t['dur'] + 1e-9)
        w = max(hi - lo + 1, 1)
        n_x_clean += w
        ws_clean.append(w)
    print(f"\n[clean-test 口径] 释放任务 {len(released_tasks)} 直进 MILP："
          f"n_x 合计 = {n_x_clean:,d}（{n_x_clean/len(released_tasks):.0f} 变量/任务，"
          f"median 窗口 {float(np.median(ws_clean)):.0f}h）"
          f"—— 现状 n_x=223/1.1s 的参考放大倍率 {n_x_clean/223:.0f}x")

    # 释放任务 GH 分布（尾部小任务确认）
    ghs = np.array([t['gh'] for t in released_tasks])
    print(f"释放任务 GH: median {np.median(ghs):.0f} | p90 {np.percentile(ghs,90):.0f}"
          f" | max {ghs.max():.0f} | 累计 {ghs.sum():,.0f}")
    durs = np.array([t['dur'] for t in released_tasks])
    print(f"释放任务 dur: median {np.median(durs):.0f}h | p90 {np.percentile(durs,90):.0f}h")
    arrivals = np.array([t['arrive'] for t in released_tasks])
    print(f"释放任务 arrive: median {np.median(arrivals):.0f} | p90 {np.percentile(arrivals,90):.0f}")

    # ---- 释放任务的改派去向 ----
    from collections import Counter
    dest_after = Counter(t['dest'] for t in failed_all if t.get('baseload', False))
    print("\n释放+失败任务改派后去向（成功者）:", dict(dest_after))
    src_before = Counter()
    for t in failed_all:
        src_before[t['dest']] += 1
    print("  改派前 dest 分布:", dict(src_before))

    # ---- 汇总统计 ----
    gh_bl = sum(t['gh'] for t in tasks if t.get('baseload', False))
    gh_tot = sum(t['gh'] for t in tasks)
    n_bl = sum(1 for t in tasks if t.get('baseload', False))
    print(f"\n[0.7 档汇总] 基荷 {n_bl}/{len(tasks)} ({n_bl/len(tasks):.1%})"
          f" | GH {gh_bl:,.0f}/{gh_tot:,.0f} ({gh_bl/gh_tot:.1%})")
    print(f"释放任务 {n_released} | 释放 GH {gh_released:,.0f}"
          f" ({gh_released/gh_tot:.1%} 总 GH) | 改派成功 {total_reassigned}"
          f" | 进 MILP {len(still_failed)}")

    # ---- 落盘（scratch，不覆盖现状） ----
    out = {
        "tasks": tasks,
        "baseload_meta": baseload_meta,
        "power": power,
        "storage": storage,
        "carbon_base_kt": carbon_base_kt,
        "nonai_arr": nonai_arr,
        "regions": REGIONS,
        "T_END": T_END,
        "type_maxlat": type_maxlat,
        "p25": p25,
        "quota_aiit_mean": quota_mean,
        "quota_aiit_arr": quota_arr,
        "n_reassigned": total_reassigned,
        "n_still_failed": len(still_failed),
        "baseload_threshold": BASELOAD_THRESHOLD,
        "dryrun": {
            "baseload_gh_ratio": BASELOAD_GH_RATIO,
            "n_released": n_released,
            "gh_released": gh_released,
            "n_bl": n_bl,
            "gh_bl": gh_bl,
            "n_x_to_milp": n_x_tot,
        },
        "meta": {"generated": "2026-08-09",
                 "source": "preprocess-sub4 复制 + BASELOAD_GH_RATIO 停止条件（dry-run）"},
    }
    out_path = OUT / f"sub4-preprocessed-ratio{int(BASELOAD_GH_RATIO*100):03d}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\n已写入 {out_path}（未覆盖现状 sub4-preprocessed.pkl）")
    print("S4 RATIO DRYRUN DONE")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    Clean-test（方案 B）预处理 — 按 BASELOAD_GH_RATIO 截断基荷预填并
    **禁用改派通道**：释放任务留原区（dest 不变）、标记 released；在
    "θ 预填 + 实时"占用上做 EDF 可行子集预检（F_r/U_r）；对 F_r 按
    GH 分位数分层抽样样本集 M_r。产出 sub4-clean-ratioXXX.pkl，
    供 sub4-clean-model.py 做 S_EDF / S_MILP 配对选时实验（回答 R1）。

原理：
    1. 层1 贪心分配与现状逐字一致（复用 s2 口径，dest 不变）；
       基荷预填 EDF 在 gh_filled ≥ BASELOAD_GH_RATIO × cap_gh[r] 时停止，
       停止点之后候选全部释放（baseload=False, start_h=None）。
    2. **禁改派**（clean 关键）：释放任务不进 failed_all / 第二轮改派，
       留在原区 → 隔离变量 = 仅"选时方式"（EDF vs MILP），区域电价结构、
       储能、卖电、层1 分配与现状完全一致。
    3. EDF 可行子集预检：释放任务在"θ 预填 + 实时"占用上逐个独立 EDF
       检查（窗口 = 预填窗口 [arrive, min(latest,2406)−dur] ∩ 主时域
       h+dur≤2400）→ 可行者进 F_r（样本池），不可行者进 U_r（实验外）。
       注：F_r 自动排除 arrive ≥ 2400−dur 的任务（主时域窗口为空者），
       保证样本在 S_MILP 的主时域窗口非空。
    4. GH 分层抽样：F_r 内按 GH 分位数 5 层等量抽样（每层 ceil(40/5)=8，
       随机种子固定，不重复）→ 样本集 M_r；放大估计按 GH 比例
       Δ_total = Σ_r Δ_r × (GH_F_r / GH_M_r)（GH 与电量/成本线性相关）。
    5. 释放任务特征（dry-run 实测）：GH median 44 / dur 3h / arrive
       median 1201h / 窗口 median 1280h——均为短小任务，样本 GPU 占用
       ~1,760 GH（40×44）相对区域容量 ~2.4M GH 可忽略，配对对比干净。

输入数据：
    - outputs/data/s2-preprocessed.pkl（阶段 1.4 S2，同源同口径）
      tasks: id/type/source/cand/arrive/dur/dem/latest/latency/gh/power
      power: 区域 → {price/sell/carbon/renewable:(2400,), pue/cap/...}
    - outputs/data/s3-preprocessed.pkl（阶段 1.4 S3）— storage/carbon_base
    - outputs/data/c-data-cleaned.pkl（阶段 0.3）— region_time_data(NonAI)
    - 中文指标 → 变量名映射：
      到达小时→arrive, 时长(h)→dur, GPU需求→dem, 最晚完成→latest,
      GPU-hour→gh, 单位GPU功率(MW)→power, 可用GPU→cap(Available_GPU),
      能效→pue, 非AI负荷→NonAI_IT_Load_MW, 基荷比例→BASELOAD_GH_RATIO,
      释放标记→released, EDF可行→edf_ok, 样本标记→sample

输出：
    - outputs/scratch/sub4-clean-ratio{θ*100:03d}.pkl — 键：
      tasks（含 released/edf_ok/sample/baseload/start_h/dest）/ baseload_meta /
      power / storage / carbon_base_kt / nonai_arr / quota_aiit_arr / p25 /
      regions / T_END / sampling_meta（每区 F_r/U_r/M_r 规模与 GH 覆盖）/
      dryrun（释放统计）/ meta
    - 控制台统计量（min/max/mean/std + 释放/抽样汇总，PR-014）

对应论文章节：
    问题四（S4）算-储-电协同优化 — 基荷预填比例敏感性实验（Clean-test）
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
P25_QUANTILE = 25         # Q1 裁定：P25 为基荷主解
BASELOAD_THRESHOLD = 0.9  # 层1 容量阈值（同 S2/S4 决策点）
BASELOAD_GH_RATIO = 0.7   # 实验参数：基荷预填占区域 GPU 容量的比例上限
N_SAMPLE_PER_REGION = 40  # 每区域样本任务数（超时降档建议 25）
N_STRATA = 5              # GH 分位数层数
SEED = 42                 # 抽样随机种子（可复现）


def earliest_feasible_hour(t, r, occ_gpu, occ_aiit, cap_gpu, quota_arr):
    """在区域 r 为任务 t 找最早可行开工小时（EDF，同 preprocess-sub4）。

    窗口 = [arrive, min(latest,2406)−dur] ∩ 主时域（h+dur ≤ 2400）。
    可行 = 运行全程每小时的 GPU 占用 ≤ cap 且 AI IT 功率 ≤ 逐时配额。
    返回开工小时 h；无可行小时返回 None。不修改任何占用表。
    """
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
    """将任务 t 在小时 h 的占用写入占用表（GPU + AI IT）。"""
    h_end = int(np.ceil(h + t['dur']))
    for hh in range(h, h_end):
        overlap = min(h + t['dur'], hh + 1) - max(h, hh)
        occ_gpu[hh] += t['dem'] * overlap
        occ_aiit[hh] += t['dem'] * t['power'] * overlap


def strat_sample_by_gh(ids_gh, n_total, n_strata, seed):
    """按 GH 分位数分层等量抽样（不重复，种子固定）。

    ids_gh: [(task_id, gh), ...]；返回抽样 task_id 列表（≤ n_total）。
    层边界 = GH 分位数 [0.2,0.4,0.6,0.8]；每层取 ceil(n_total/n_strata) 个。
    """
    if len(ids_gh) <= n_total:
        return [tid for tid, _ in ids_gh]
    ghs = np.array([gh for _, gh in ids_gh])
    qs = np.quantile(ghs, np.linspace(0.0, 1.0, n_strata + 1)[1:-1])
    strata = [[] for _ in range(n_strata)]
    for tid, gh in ids_gh:
        s = int(np.searchsorted(qs, gh, side="right"))
        strata[min(s, n_strata - 1)].append(tid)
    rng = np.random.default_rng(seed)
    out = []
    per = int(np.ceil(n_total / n_strata))
    for s in strata:
        rng.shuffle(s)
        out.extend(s[:per])
    return out[:n_total]


def main():
    import sys
    # θ 参数化：python preprocess-sub4-clean.py 0.7（默认 0.7）
    global BASELOAD_GH_RATIO
    if len(sys.argv) > 1:
        BASELOAD_GH_RATIO = float(sys.argv[1])
    assert 0.0 < BASELOAD_GH_RATIO < 1.0
    print("=" * 78)
    print(f"S4 Clean-test 预处理: BASELOAD_GH_RATIO = {BASELOAD_GH_RATIO}"
          f" | 样本/区域 = {N_SAMPLE_PER_REGION}")
    print("=" * 78)

    # ---------- 1) 加载 S2/S3 缓存 ----------
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

    # ---------- 2) NonAI 负荷 ----------
    rtd = cd['region_time_data']
    nonai_arr = {}
    for r in REGIONS:
        sub = rtd[(rtd['Region'] == r) & (rtd['Hour'] < H)].sort_values('Hour')
        nonai_arr[r] = sub['NonAI_IT_Load_MW'].values.astype(float)

    # ---------- 3) 层1 贪心分配（与现状逐字一致） ----------
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

    # ---------- 4) 配额（P25 矩形 − NonAI） ----------
    p25 = {}
    quota_arr = {}
    quota_mean = {}
    for r in REGIONS:
        a = power[r]['renewable']
        p25[r] = float(np.percentile(a, P25_QUANTILE))
        quota_arr[r] = np.maximum(p25[r] / power[r]['pue'] - nonai_arr[r], 0.0)
        quota_mean[r] = float(quota_arr[r].mean())

    # ---------- 5) 基荷预填（θ 停止，禁改派） ----------
    occ_gpu = {r: np.zeros(H, dtype=float) for r in REGIONS}
    occ_aiit = {r: np.zeros(H, dtype=float) for r in REGIONS}
    for t in tasks:
        if t['type'] == 'RealTimeInference':
            r = t['dest']
            h0, dur = t['arrive'], t['dur']
            for hh in range(int(np.floor(h0)), int(np.ceil(h0 + dur))):
                if hh < H:
                    overlap = min(h0 + dur, hh + 1) - max(h0, hh)
                    occ_gpu[r][hh] += t['dem'] * overlap
                    occ_aiit[r][hh] += t['dem'] * t['power'] * overlap

    cand_r = {r: sorted(
        [t for t in tasks if t['dest'] == r and t['type'] != 'RealTimeInference'],
        key=lambda t: -t['gh']) for r in REGIONS}

    baseload_meta = {}
    release_meta = {}
    for r in REGIONS:
        cap_gpu = power[r]['cap']
        cap_gh_r = cap_gpu * H
        slack_gpu = float(cap_gh_r - occ_gpu[r].sum())
        slack_aiit = float(quota_arr[r].sum() - occ_aiit[r].sum())
        n_fill = 0
        gh_filled = 0.0
        n_released = 0
        gh_released = 0.0
        for i, t in enumerate(cand_r[r]):
            if gh_filled >= BASELOAD_GH_RATIO * cap_gh_r:
                # 【clean】停止：剩余候选全部释放，禁改派，留原区
                for t_rest in cand_r[r][i:]:
                    t_rest['released'] = True
                    t_rest['baseload'] = False
                    t_rest['start_h'] = None
                n_released = len(cand_r[r]) - i
                gh_released = sum(t2['gh'] for t2 in cand_r[r][i:])
                break
            if slack_gpu < t['gh'] or slack_aiit < t['gh'] * t['power']:
                t['released'] = False
                t['baseload'] = False
                t['start_h'] = None
                continue
            h = earliest_feasible_hour(t, r, occ_gpu[r], occ_aiit[r],
                                       cap_gpu, quota_arr[r])
            if h is not None:
                commit_placement(t, h, occ_gpu[r], occ_aiit[r])
                slack_gpu -= t['gh']
                slack_aiit -= t['gh'] * t['power']
                t['released'] = False
                t['baseload'] = True
                t['start_h'] = h
                n_fill += 1
                gh_filled += t['gh']
            else:
                t['released'] = False
                t['baseload'] = False
                t['start_h'] = None
        theory_gh = quota_mean[r] / p_train * H
        baseload_meta[r] = {
            'n_candidates': len(cand_r[r]), 'n_filled': n_fill,
            'n_failed': len(cand_r[r]) - n_fill - n_released,
            'gh_filled': gh_filled,
            'theoretical_quota_gh': theory_gh,
            'fill_rate_vs_quota': gh_filled / theory_gh if theory_gh > 0 else 0.0,
            'ratio_cap_gh': BASELOAD_GH_RATIO * cap_gh_r,
        }
        release_meta[r] = {'n_released': n_released, 'gh_released': gh_released}
        pct = n_fill / len(cand_r[r]) * 100 if cand_r[r] else 0.0
        print(f"  {r}: 候选 {len(cand_r[r]):5d} | 基荷填充 {n_fill:5d} ({pct:4.1f}%)"
              f" | GH {gh_filled:12,.0f} / θ界 {BASELOAD_GH_RATIO*cap_gh_r:12,.0f}"
              f" | 释放 {n_released:5d} (GH {gh_released:12,.0f})")

    # ---------- 6) 释放任务 EDF 可行子集预检（在 θ 预填+实时占用上独立检查） ----------
    sampling_meta = {}
    sample_ids = set()
    for r in REGIONS:
        rel = [t for t in tasks if t.get('released') and t['dest'] == r]
        if not rel:
            sampling_meta[r] = {'n_released': 0, 'n_edf_ok': 0, 'n_sample': 0,
                                'gh_edf_ok': 0.0, 'gh_sample': 0.0}
            continue
        # 独立预检：不互相占位，仅检查在"θ 预填+实时"占用上是否有可行小时
        n_ok = 0
        gh_ok = 0.0
        for t in rel:
            h = earliest_feasible_hour(t, r, occ_gpu[r], occ_aiit[r],
                                       power[r]['cap'], quota_arr[r])
            t['edf_ok'] = h is not None
            if h is not None:
                n_ok += 1
                gh_ok += t['gh']
        # GH 分层抽样（仅从可行子集 F_r 抽）
        pool = [(t['id'], t['gh']) for t in rel if t.get('edf_ok')]
        chosen = strat_sample_by_gh(pool, N_SAMPLE_PER_REGION, N_STRATA, SEED)
        chosen_set = set(chosen)
        gh_sample = sum(t['gh'] for t in rel if t['id'] in chosen_set)
        for t in rel:
            t['sample'] = t['id'] in chosen_set
        sample_ids.update(chosen_set)
        sampling_meta[r] = {
            'n_released': len(rel), 'gh_released': release_meta[r]['gh_released'],
            'n_edf_ok': n_ok, 'n_edf_fail': len(rel) - n_ok,
            'gh_edf_ok': gh_ok, 'gh_edf_fail': release_meta[r]['gh_released'] - gh_ok,
            'n_sample': len(chosen), 'gh_sample': gh_sample,
            'sample_ratio_gh': gh_ok / gh_sample if gh_sample > 0 else 0.0,
        }
        print(f"  {r}: 释放 {len(rel):5d} | EDF可行 {n_ok:5d} (GH {gh_ok:12,.0f})"
              f" | 抽样 {len(chosen):3d} (GH {gh_sample:8,.0f})"
              f" | 放大倍数 {sampling_meta[r]['sample_ratio_gh']:.1f}x")

    # ---------- 7) 落盘（scratch，不覆盖现状） ----------
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
        "sampling_meta": sampling_meta,
        "sample_ids": sorted(sample_ids),
        "baseload_threshold": BASELOAD_THRESHOLD,
        "dryrun": {
            "baseload_gh_ratio": BASELOAD_GH_RATIO,
            "n_sample_per_region": N_SAMPLE_PER_REGION,
            "n_strata": N_STRATA,
            "seed": SEED,
            "release": release_meta,
        },
        "meta": {"generated": "2026-08-09",
                 "source": "s2-preprocessed + s3-preprocessed + c-data-cleaned"
                           "（口径零漂移复用）+ θ 停止 + 禁改派 + GH 分层抽样"},
    }
    out_path = OUT / f"sub4-clean-ratio{int(BASELOAD_GH_RATIO * 100):03d}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    # 全局统计
    n_bl = sum(1 for t in tasks if t.get('baseload', False))
    gh_bl = sum(t['gh'] for t in tasks if t.get('baseload', False))
    gh_tot = sum(t['gh'] for t in tasks)
    n_rel = sum(1 for t in tasks if t.get('released', False))
    n_smp = len(sample_ids)
    print(f"\n[θ={BASELOAD_GH_RATIO} 汇总] 基荷 {n_bl}/{len(tasks)} ({n_bl/len(tasks):.1%})"
          f" | GH {gh_bl:,.0f}/{gh_tot:,.0f} ({gh_bl/gh_tot:.1%})"
          f" | 释放 {n_rel} | 样本 {n_smp}")
    print(f"已写入 {out_path}（未覆盖现状 sub4-preprocessed.pkl）")

    # 收敛断言
    assert n_fallback / len(tasks) < 0.01, "退路比例异常（>1%），暂停"
    for r in REGIONS:
        m = baseload_meta[r]
        assert m['fill_rate_vs_quota'] <= 1.0 + 1e-6, f"{r} 基荷填充超理论配额"
        sm = sampling_meta[r]
        assert sm['n_sample'] <= N_SAMPLE_PER_REGION
        assert sm['gh_sample'] <= sm['gh_edf_ok'] + 1e-9, \
            f"{r} 样本 GH 超可行子集 GH"
    print("S4 CLEAN PREPROCESSING DONE")


if __name__ == "__main__":
    main()

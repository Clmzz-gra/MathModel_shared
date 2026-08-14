# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 1.4 S4 数据预处理 — 构建 sub4-preprocessed.pkl，
    为算-储-电协同优化（半耦合：层1 贪心分配 + 层2 每区独立 MILP 含储能）
    提供任务分配、基荷预填、电力/储能参数。

原理：
    1. 数据源复用（口径零漂移）：
       - 任务/时延裁剪/电力参数 ← s2-preprocessed.pkl（阶段 1.4 S2，同源同口径）
       - 储能参数/碳基准       ← s3-preprocessed.pkl（阶段 1.4 S3）
       - NonAI 负荷            ← c-data-cleaned.pkl（阶段 0.3，region_time_data）
    2. 层 1 贪心分配（复用 S2 capacity_aware_assign，口径一致）：
       数据行序遍历；候选按 PUE(r)×Price_mean(r) 升序（gh·power 为任务常量，
       与 S2 cost_of 排序等价）；选 GPU-hour ≤ 0.9·Cap_r·2400 者；
       全满退路选负载率最低。实时任务同样分配目的地（到达即开工）。
    3. 基荷预填（方案书 §6.2，Q1 裁定 P25=588 MW 主解）：
       - 区域 r 的逐时 AI IT 配额（IT 侧，MW/小时）=
         max(P25_r/PUE_r − NonAI_t, 0)，逐时 NonAI（审查 M-3 修正，
         均值口径高峰小时会突破"基荷=绿色算力"物理保证）
       - 候选 = 分配到 r 的延迟容忍任务（训练+批量），按 GPU-hour 降序
       - EDF 逐个尝试：窗口 [arrive, min(latest,2406)−dur] ∩ 主时域
         （h+dur ≤ 2400 ⟺ h ≤ 2400−ceil(dur)，跨收尾段任务交 MILP）
       - 可行 = 该任务运行全程 GPU 占用 ≤ Cap_r 且 AI IT ≤ 逐时配额
         （初始占用含实时任务固定开工的 GPU + AI IT）
       - 填入成功 → baseload=True + start_h 固定（不进 MILP 变量空间）
       - EDF 失败 → 第二轮改派：候选内 GPU-hour 负载率最低且未满 90%
         的区域，再尝试 EDF；成功则更新 dest 并同步扣减旧区域账本
         （M-A/M-B 修正）；仍失败 → 进 MILP（层2 需显式处理，实测
         残留个位数任务）
    4. 性能守卫（审查 M-4）：维护区域 GPU/AI-IT 总余量 slack 标量，
       每任务先 O(1) 预判"总余量不足"直接跳过（精确必要条件，不误杀）；
       跳过任务统一走改派通道（M-E 修正）。
    5. 基荷任务 GPU-hour 覆盖率量化（方案书 §3 结论的任务粒度验证）。

输入数据：
    - outputs/data/s2-preprocessed.pkl（阶段 1.4 S2，同源同口径）
      tasks: id/type/source/cand/arrive/dur/dem/latest/latency/gh/power
      power: 区域 → {price/sell/carbon/renewable:(2400,), pue/cap/...}
      power_mapping: 任务类型 → GPU_Power_MW_per_EquivalentGPU
    - outputs/data/s3-preprocessed.pkl（阶段 1.4 S3）
      storage: 区域 → {Capacity/MinSOC/InitialSOC/MaxCharge/MaxDischarge/
               ChargeEfficiency/DischargeEfficiency/SellLimit/MaxGridImport}
      carbon_base_kt: 区域基准碳排（主时域 0-2399 口径）
    - outputs/data/c-data-cleaned.pkl（阶段 0.3）
      region_time_data: Region/Hour/NonAI_IT_Load_MW
    - 中文指标 → 变量名映射：
      到达小时→arrive, 时长(h)→dur, GPU需求→dem, 最晚完成→latest,
      最大时延→latency, GPU-hour→gh, 单位GPU功率(MW)→power,
      可用GPU→cap(Available_GPU), 能效→pue, 非AI负荷→NonAI_IT_Load_MW

输出：
    - outputs/data/sub4-preprocessed.pkl — 键：
      tasks（50000，含 dest/baseload/start_h）/ baseload_meta（配额/填充/改派）/
      power（区域逐时电力参数，复用）/ storage（储能参数，复用）/
      carbon_base_kt / nonai_mean / nonai_arr / regions / T_END /
      type_maxlat / p25 / quota_aiit_mean / quota_aiit_arr /
      n_reassigned / n_still_failed / baseload_threshold / meta
    - 控制台统计量（min/max/mean/std + 基荷覆盖汇总，PR-014）

对应论文章节：
    问题四（S4）算-储-电协同优化 — 阶段 1.4 数据预处理
"""
import pickle
from pathlib import Path

import numpy as np

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA = BASE / "outputs" / "data"

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
H = 2400
T_END = 2406
P25_QUANTILE = 25         # Q1 裁定：P25 为基荷主解
BASELOAD_THRESHOLD = 0.9  # 层 1 容量阈值（同 S2 决策点）


def earliest_feasible_hour(t, r, occ_gpu, occ_aiit, cap_gpu, quota_arr):
    """在区域 r 为任务 t 找最早可行开工小时（EDF）。

    窗口 = [arrive, min(latest,2406)−dur] ∩ 主时域（h+dur ≤ 2400）。
    可行 = 运行全程每小时的 GPU 占用 ≤ cap 且 AI IT 功率 ≤ 逐时配额。
    返回开工小时 h；无可行小时返回 None。不修改任何占用表。
    """
    lo = int(t['arrive'])
    hi = int(min(t['latest'], T_END) - t['dur'])
    hi = min(hi, H - int(np.ceil(t['dur'])))  # h+dur ≤ 2400 ⟺ h ≤ 2400−ceil(dur)
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


def main():
    print("=" * 78)
    print("S4 阶段 1.4 数据预处理 — 半耦合算-储-电协同输入面板")
    print("=" * 78)

    # ---------- 1) 加载 S2/S3 预处理缓存 ----------
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
    p_train = s2['power_mapping']['AITraining']  # 训练 GPU 功率（理论配额口径，L-1 修正）
    print(f"加载: tasks {len(tasks)} | power 键 {list(power.keys())} | "
          f"storage 键 {list(storage.keys())} | p_train={p_train}")

    # ---------- 2) NonAI 负荷（IT 侧，逐时 + 均值） ----------
    rtd = cd['region_time_data']
    nonai_arr, nonai_mean = {}, {}
    for r in REGIONS:
        sub = rtd[(rtd['Region'] == r) & (rtd['Hour'] < H)].sort_values('Hour')
        nonai_arr[r] = sub['NonAI_IT_Load_MW'].values.astype(float)
        nonai_mean[r] = float(nonai_arr[r].mean())
        print(f"NonAI[{r}] min {nonai_arr[r].min():.1f} | max {nonai_arr[r].max():.1f}"
              f" | mean {nonai_mean[r]:.1f} MW")

    # ---------- 3) 层 1：容量感知贪心分配（复用 S2 逻辑，口径一致） ----------
    print("\n--- 层 1 贪心分配（PUE×均价升序 + 90% 阈值） ---")
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
        if dest is None:  # 退路：候选内负载率最低（同 S2）
            dest = min(cand_sorted, key=lambda r: assigned_gh[r] / cap_gh[r])
            n_fallback += 1
        t['dest'] = dest
        assigned_gh[dest] += t['gh']
    print(f"分配完成 | 退路任务 {n_fallback} ({n_fallback/len(tasks):.2%}) [S2 基准: 117]")
    for r in REGIONS:
        n_r = sum(1 for t in tasks if t['dest'] == r)
        gh_r = sum(t['gh'] for t in tasks if t['dest'] == r)
        print(f"  {r}: 承接 {n_r:6d} 任务 | GPU-hour {gh_r:12,.0f} / {cap_gh[r]:,.0f}"
              f" = {gh_r/cap_gh[r]:6.1%}")

    # ---------- 4) 基荷预填（P25 矩形 + EDF + 第二轮改派） ----------
    print("\n--- 基荷预填（P25 矩形 + EDF + 失败改派，Q1 裁定） ---")
    p25 = {}
    quota_arr = {}     # 逐时 AI IT 配额（IT 侧 MW/小时，M-3 修正）
    quota_mean = {}    # 均值口径（meta 报告用）
    for r in REGIONS:
        a = power[r]['renewable']
        p25[r] = float(np.percentile(a, P25_QUANTILE))
        # 逐时配额 = max(P25/PUE − NonAI_t, 0)：基荷矩形减去逐时 NonAI 后的 AI 余量
        quota_arr[r] = np.maximum(p25[r] / power[r]['pue'] - nonai_arr[r], 0.0)
        quota_mean[r] = float(quota_arr[r].mean())
        print(f"{r}: P25 = {p25[r]:.0f} MW | AI IT 配额 mean = {quota_mean[r]:.1f}"
              f" (min {quota_arr[r].min():.1f} / max {quota_arr[r].max():.1f}) MW"
              f" | 等效 GPU(训练) mean = {quota_mean[r]/p_train:.0f}")

    # 每区域占用表（实时任务固定开工为基础）
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

    # 候选：分配到 r 的延迟容忍任务，GPU-hour 降序
    cand_r = {r: sorted(
        [t for t in tasks if t['dest'] == r and t['type'] != 'RealTimeInference'],
        key=lambda t: -t['gh']) for r in REGIONS}

    baseload_meta = {}
    total_reassigned = 0
    failed_all = []  # 本区域不可行任务（含守卫跳过），统一走第二轮改派通道
    for r in REGIONS:
        cap_gpu = power[r]['cap']
        slack_gpu = float(cap_gpu * H - occ_gpu[r].sum())       # GPU-hour 总余量
        slack_aiit = float(quota_arr[r].sum() - occ_aiit[r].sum())  # AI-IT 总余量
        n_fill = 0
        gh_filled = 0.0
        for t in cand_r[r]:
            # 性能守卫（M-4）：总余量不足则必然无可行小时（精确必要条件）；
            # 跳过任务统一进 failed_all 走改派通道（M-E 修正）
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
            'n_candidates': len(cand_r[r]),
            'n_filled': n_fill,
            'n_failed': len(cand_r[r]) - n_fill,
            'gh_filled': gh_filled,
            'theoretical_quota_gh': theory_gh,
            'fill_rate_vs_quota': gh_filled / theory_gh if theory_gh > 0 else 0.0,
            'quota_aiit_mean_mw': quota_mean[r],
        }
        pct = n_fill / len(cand_r[r]) * 100 if cand_r[r] else 0.0
        print(f"  {r}: 候选 {len(cand_r[r]):5d} | 基荷填充 {n_fill:5d} ({pct:4.1f}%)"
              f" | GPU-hour {gh_filled:12,.0f} / 理论配额 {theory_gh:12,.0f}"
              f" = {gh_filled/max(theory_gh,1):.1%}")

    # 第二轮改派：本区域不可行任务 → 候选内 GPU-hour 负载率最低且未满 90% 的区域
    print(f"\n第二轮改派：{len(failed_all)} 个不可行任务，按负载率最低改派")
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
                # 预算账本：旧区域扣减、新区域累加（M-B 修正）
                assigned_gh[old_dest] -= t['gh']
                assigned_gh[r] += t['gh']
                t['dest'] = r
                t['baseload'] = True
                t['start_h'] = h
                moved = True
                total_reassigned += 1
                # meta 统计：新区域 n_filled 累加、旧区域 n_failed 扣减（M-A 修正）
                baseload_meta[r]['n_filled'] += 1
                baseload_meta[r]['gh_filled'] += t['gh']
                baseload_meta[old_dest]['n_failed'] -= 1
                break
        if not moved:
            t['baseload'] = False
            t['start_h'] = None
    # 改派后统一重算 fill_rate_vs_quota（M-A：与最终 gh_filled 同步）
    for r in REGIONS:
        m = baseload_meta[r]
        m['fill_rate_vs_quota'] = m['gh_filled'] / m['theoretical_quota_gh'] \
            if m['theoretical_quota_gh'] > 0 else 0.0
        pct = m['n_filled'] / m['n_candidates'] * 100 if m['n_candidates'] else 0.0
        print(f"  {r}: 填充 {m['n_filled']:5d}/{m['n_candidates']:5d} ({pct:4.1f}%)"
              f" | GPU-hour {m['gh_filled']:12,.0f} = {m['fill_rate_vs_quota']:.1%} 配额")
    still_failed = [t for t in failed_all if not t['baseload']]
    print(f"改派成功 {total_reassigned} | 仍失败 {len(still_failed)}"
          f"（进 MILP，层2 需显式处理）")

    # 基荷任务全局统计
    n_bl = sum(1 for t in tasks if t.get('baseload', False))
    gh_bl = sum(t['gh'] for t in tasks if t.get('baseload', False))
    gh_tot = sum(t['gh'] for t in tasks)
    print(f"\n基荷任务全局: {n_bl}/{len(tasks)} ({n_bl/len(tasks):.1%}) | "
          f"GPU-hour {gh_bl:,.0f}/{gh_tot:,.0f} ({gh_bl/gh_tot:.1%})")

    # ---------- 5) 落盘 ----------
    out = {
        "tasks": tasks,
        "baseload_meta": baseload_meta,
        "power": power,
        "storage": storage,
        "carbon_base_kt": carbon_base_kt,
        "nonai_mean": nonai_mean,
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
        "meta": {
            "generated": "2026-08-08",
            "source": "s2-preprocessed + s3-preprocessed + c-data-cleaned（口径零漂移复用）",
            "p25_quantile": P25_QUANTILE,
        },
    }
    with open(DATA / "sub4-preprocessed.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n已写入 {DATA / 'sub4-preprocessed.pkl'}")

    # 收敛断言（供审查）
    assert n_fallback / len(tasks) < 0.01, "退路比例异常（>1%），暂停"
    for r in REGIONS:
        assert baseload_meta[r]['fill_rate_vs_quota'] <= 1.0 + 1e-6, \
            f"{r} 基荷填充超理论配额"
    print("S4 PREPROCESSING DONE")


if __name__ == "__main__":
    main()

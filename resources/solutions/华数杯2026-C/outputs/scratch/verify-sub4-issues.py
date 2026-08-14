# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 3.1 批判阅读修正核验（只读，不重算主解）——回答 M14（基荷配额 vs 受限
    消纳上限）与 M17（LNS 终值超容状态）两项待核验问题

原理：
    M14：基荷矩形取 P25(Avail)=588 MW，逐时 AI IT 配额 Quota = P25/PUE - NonAI。
         受限消纳上限 = UsedRenewable + RenewableCharge（基准观测，S3 B1 口径）。
         若 Quota 超过消纳上限，"基荷任务绿色消纳"承诺落空（需购电补足）。
         统计各区域 Quota > 消纳上限 的小时占比与超额量级。
    M17：LNS 每轮子 MILP 含超容松弛（slack_gh），检查 s4-lns-results.pkl
         per_region[E/F].curve 末项的 slack_gh，判断 LNS 终值是否仍超容。

输入数据：
    - sub4-preprocessed.pkl (处理后) — 键: quota_aiit_arr（dict {区域: (2400,) MW}）
    - c-data-cleaned.pkl (处理后) — 键: region_time_data（UsedRenewable_MW,
      RenewableCharge_MW 列，受限消纳上限 = 两者之和）
    - s4-lns-results.pkl (处理后) — per_region[E/F].curve（末项含 slack_gh）
    - 中文指标 → 变量名映射：受限消纳上限→absorb_limit, 逐时配额→quota_arr,
      P25 基荷→p25, 非 AI 负荷→NonAI_IT_Load_MW, 直消→UsedRenewable_MW,
      新能源充电→RenewableCharge_MW

输出：
    - 控制台核验统计（M14：超额小时占比/量级；M17：LNS 终值 slack）
    - 供报告声明引用（不产出新数据文件）

对应论文章节：
    §4 模型 / §8.2 LNS（阶段 3.1 修正依据）
"""
import pickle
from pathlib import Path

import numpy as np

DATA = Path("e:/MathModel_pj-2026-C/outputs/data")
SCRATCH = Path("e:/MathModel_pj-2026-C/outputs/scratch")
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
MAIN = 2400

print("=" * 70)
print("M14 核验：基荷 AI IT 配额 vs 受限消纳上限（Quota ≤ Used+Rch 是否成立）")
print("=" * 70)

with open(DATA / "sub4-preprocessed.pkl", "rb") as f:
    sp = pickle.load(f)
with open(DATA / "c-data-cleaned.pkl", "rb") as f:
    cd = pickle.load(f)

quota_arr = sp["quota_aiit_arr"]
print(f"quota_aiit_arr 类型: {type(quota_arr)}, 键: {list(quota_arr.keys())[:2] if isinstance(quota_arr, dict) else 'n/a'}")

rtd = cd["region_time_data"]
absorb = {}
for r in REGIONS:
    sub = rtd[(rtd["Region"] == r) & (rtd["Hour"] < MAIN)].sort_values("Hour")
    absorb[r] = (sub["UsedRenewable_MW"].values[:MAIN]
                 + sub["RenewableCharge_MW"].values[:MAIN]).astype(float)

print(f"\n{'区域':<8}{'Quota mean(MW)':>16}{'Quota max(MW)':>16}"
      f"{'消纳上限 mean':>16}{'超额小时%':>12}{'超额中位(MW)':>14}")
for r in REGIONS:
    q = np.asarray(quota_arr[r])[:MAIN]
    ab = absorb[r]
    excess_h = int(np.sum(q > ab + 1e-6))
    excess_med = float(np.median(q[q > ab + 1e-6])) if excess_h else 0.0
    print(f"{r:<8}{q.mean():>16.1f}{q.max():>16.1f}{ab.mean():>16.1f}"
          f"{excess_h / MAIN * 100:>11.1f}%{excess_med:>14.1f}")

# 全局口径：按 GPU-hour 加权看基荷任务中"Quota > 消纳上限"小时占比
print("\n[解读] 若超额小时占比高，基荷任务该时段绿电供给不足，需购电补足，"
      "'绿色基荷'表述应弱化（M14）。")

print("\n" + "=" * 70)
print("M17 核验：LNS 终值超容状态（s4-lns-results.pkl per_region curve 末项）")
print("=" * 70)

with open(SCRATCH / "s4-lns-results.pkl", "rb") as f:
    lns = pickle.load(f)

print(f"lns 键: {list(lns.keys())}")
for r in ["RegionE", "RegionF"]:
    pr = lns["per_region"][r]
    curve = pr["curve"]
    last = curve[-1]
    print(f"\n{r}: EDF {pr['cost_edf']:.2f}M | B2 {pr['cost_b2']:.2f}M | "
          f"LNS 终值 {pr['cost_final']:.2f}M")
    print(f"  末轮(it={last['it']}): cost {last['cost']:.2f}M | "
          f"slack_gh {last['slack_gh']:.1f} | status {last['status']} | "
          f"n_x {last['n_x']}")
    print(f"  EDF/B2 基准超容: {pr.get('slack_edf_gh', float('nan')):,.0f} / "
          f"{pr.get('slack_b2_gh', float('nan')):,.0f} GH")

if "compare" in lns:
    print(f"\ncompare: {lns['compare']}")
print("\n[解读] LNS 终值 slack_gh>0 说明该构造仍含超容（θ=0.3 结构），"
      "与主解比较需标注口径（M17）。")

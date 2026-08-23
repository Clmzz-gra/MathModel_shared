# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    生成 S2 模型增量验证 notebook（outputs/notebooks/verify-sub2-model.ipynb），
    将 sub2-model.py 主流程拆为可分段运行 cell，耗时步骤（基线/ε 三档）
    带指纹缓存，二次运行秒级命中

原理：
    1. Cell 1 用 exec 加载 sub2-model.py 全部函数定义（单一代码来源，不重复维护）
    2. 缓存 helper load_or_compute：指纹 = 版本号 + 参数（WIN/W_EXT/阈值/tl/E0），
       命中则跳过计算（落盘 outputs/data/cache/s2_*.pkl）
    3. cell 划分：自检（秒）→ 基线（~20min，缓存）→ 分配（秒）→
       η=1.0/0.9/0.8（各 ~20-40min，缓存）→ 评价/出图/落盘（秒）
    4. 区域级 4 进程并行已内置于 sub2-model.py 的 schedule_all

输入数据：
    - outputs/scratch/sub2-model.py（函数唯一来源）
    - outputs/data/s2-preprocessed.pkl（预处理产物）

输出：
    - outputs/notebooks/verify-sub2-model.ipynb

对应论文章节：
    问题二（S2）碳感知任务调度 — 阶段 2.1 代码实现（增量验证版）
"""
import nbformat as nbf
from pathlib import Path

BASE = Path(r"e:\MathModel_pj-2026-C")

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# S2 模型增量验证（verify-sub2-model）

阶段 2.1 代码实现的分段运行版。耗时步骤（S1 时移基线 / ε 三档）带指纹缓存，**二次运行秒级命中**。

- 求解器：方案 K3（全局 EDF 贪心基解 + 顺序逐窗 MILP 改进 + 未来基解预留，全局容量 ≤ cap 数学保证）
- 参数：`WIN=24 / W_EXT=48 / 阈值 0.90 / time_limit=20s / gap=0.01`（8 线程区域并行）
- 区域级并行：线程池 `schedule_all`（HiGHS 释放 GIL 真并行，兼容 notebook exec）
- 代码唯一来源：[sub2-model.py](../../outputs/scratch/sub2-model.py)，本 notebook 仅分段执行

**运行方式**：① 逐 cell 手动运行（推荐，可中途核对）；② 或 `python outputs/scratch/run-verify-sub2-model-notebook.py` 全量执行。""" ))

cells.append(nbf.v4.new_code_cell(
"""# ===== Cell 1：加载函数定义 + 缓存 helper（秒级）=====
import hashlib, json, pickle, time
from pathlib import Path

BASE = Path(r"e:\\MathModel_pj-2026-C")

# 加载 sub2-model.py 全部函数定义与全局数据（不执行 __main__ 主流程）
_src = (BASE / "outputs" / "scratch" / "sub2-model.py").read_text(encoding="utf-8")
exec(compile(_src.split("if __name__")[0], "sub2-model", "exec"))

CACHE_DIR = BASE / "outputs" / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

VER = 4  # 缓存版本：K3v2（future_occ 不含本窗基解）+ 区域并行 + W_EXT=48/tl=20

def load_or_compute(name, compute_fn, params=None):
    fp = CACHE_DIR / f"s2_{name}.pkl"
    sig = {"ver": VER, "params": params or {}}
    key = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:12]
    if fp.exists():
        with open(fp, "rb") as f:
            data = pickle.load(f)
        if data.get("_key") == key:
            print(f"[cache] 命中 {name} (key={key})，跳过计算")
            return data
    print(f"[cache] 未命中 {name}，开始计算（耗时步骤，区域并行）...")
    t0 = time.time()
    data = compute_fn()
    data["_key"] = key
    with open(fp, "wb") as f:
        pickle.dump(data, f)
    print(f"[cache] {name} 已缓存 (key={key})，耗时 {time.time()-t0:.0f}s")
    return data

print(f"函数与数据加载完成：{len(tasks)} 任务，{len(REGIONS)} 区域")"""))

cells.append(nbf.v4.new_code_cell(
"""# ===== Cell 2：数据自检（秒级）=====
import numpy as np
from collections import Counter

tc = Counter(t["type"] for t in tasks)
durs = np.array([t["dur"] for t in tasks])
dems = np.array([t["dem"] for t in tasks])
n_cand = np.array([len(t["cand"]) for t in tasks])
print(f"任务数: {len(tasks)} = 训练 {tc['AITraining']} / 批量 {tc['BatchInference']} / 实时 {tc['RealTimeInference']}")
print(f"dur(h): min={durs.min():.2f} max={durs.max():.2f} mean={durs.mean():.2f}")
print(f"dem: min={dems.min()} max={dems.max()} mean={dems.mean():.1f}")
print(f"候选目的地: mean={n_cand.mean():.2f}（锁死 1 候选 {int((n_cand==1).sum())}）")

assert len(tasks) == 50000
assert tc["RealTimeInference"] == 16724
assert (durs <= W_EXT).all(), "dur 超过跨窗上限"
print("自检通过 ✅")"""))

cells.append(nbf.v4.new_code_cell(
"""# ===== Cell 3：模块一 S1 时移基线（Q1 正式口径，缓存 ~20min）=====
def _bl():
    C0, E0, meta = run_baseline()
    return {"C0": C0, "E0": E0, "meta": meta}

bl = load_or_compute("baseline", _bl,
                     params={"WIN": WIN, "W_EXT": W_EXT, "THRESHOLD": THRESHOLD})
C0, E0 = bl["C0"], bl["E0"]
print(f"S1 时移基线: 成本 C₀ = {C0/1e6:.1f}M 元, 碳排 E₀ = {E0/1e3:.2f}kt")
print("（朴素口径参考 441.7M/358.0kt 已降级，Q1 裁定正式基线以本值为准）")
for r, m in bl["meta"].items():
    print(f"  {r}: cost={m['cost']/1e6:.1f}M, co2={m['co2']/1e3:.2f}kt, "
          f"status={m['status']}, fallback={m['fallback']}, forced={m['forced']}, dt={m['dt']:.0f}s")"""))

cells.append(nbf.v4.new_code_cell(
"""# ===== Cell 4：模块二 层1 容量感知分配（F3/F4 复现，秒级）=====
dest0, demand0, assign0, fail0, c0, k0, cn, kn = capacity_aware_assign()
print(f"退路任务: {fail0} ({fail0/len(tasks):.2%})  [基准: 117]")
for r in REGIONS:
    print(f"  {r}: 承接 {assign0[r]} 任务, GPU-hour {demand0[r]:,.0f} / {CAP_GH[r]:,.0f} = {demand0[r]/CAP_GH[r]:.1%}")
print(f"朴素口径成本降幅: {(c0-cn)/c0:.1%}（基准 -16.6%）")
print(f"朴素口径碳排降幅: {(k0-kn)/k0:.1%}（基准 -30.4%）")
assert fail0 == 117, "退路任务数应与 F3 基准一致" """))

for eta in [1.0, 0.9, 0.8]:
    vn = f"r{int(eta * 10)}"  # r10 / r9 / r8
    code = (
        f"# ===== Cell 5.{int(eta*10)}：模块三 η={eta} ε-约束调度（缓存 ~20-40min）=====\n"
        f"def _eta():\n    return run_eta({eta}, E0)\n\n"
        f"{vn} = load_or_compute(\"eta_{eta}\", _eta,\n"
        f"                     params={{\"eta\": {eta}, \"E0\": round(E0, 1),\n"
        f"                               \"WIN\": WIN, \"W_EXT\": W_EXT, \"THRESHOLD\": THRESHOLD}})\n"
        f"print(f\"η={eta}: 成本 {{{vn}['C']/1e6:.1f}}M 元, 碳排 {{{vn}['E']/1e3:.2f}}kt, \"\n"
        f"      f\"收敛迭代 {{len({vn}['iters'])}} 轮, 实际 ε={{{vn}['eta_eff']}}\")\n"
        f"for it in {vn}['iters']:\n"
        f"    print(f\"  iter{{it['iter']}}: C={{it['C']/1e6:.1f}}M, E={{it['E']/1e3:.2f}}kt, \"\n"
        f"          f\"收敛={{it['converged']}}, dt={{it.get('dt', 0)/60:.1f}}min\")\n"
    )
    cells.append(nbf.v4.new_code_cell(code))

cells.append(nbf.v4.new_code_cell(
"""# ===== Cell 8：模块四 评价 + 汇总对比表（秒级）=====
eta_results = [r10, r9, r8]
r1 = eta_results[0]
delay, n_mig = eval_metrics(r1["dest"])
print(f"平均迁移时延: {delay:.1f} ms（迁移任务 {n_mig}/{len(tasks)} = {n_mig/len(tasks):.1%}）")

print("\\n迁移收益对比（S1 时移基线 vs S2 三档）")
rows = [("S1 时移基线", C0 / 1e6, E0 / 1e3, "-")]
for r in eta_results:
    rows.append((f"S2 η={r['eta']}", r["C"] / 1e6, r["E"] / 1e3, f"{delay:.1f}ms"))
print(f"{'方案':<14}{'成本(M元)':<12}{'碳排(kt)':<12}{'迁移时延'}")
for name, c, e, d in rows:
    print(f"{name:<14}{c:<12.1f}{e:<12.2f}{d}")"""))

cells.append(nbf.v4.new_code_cell(
"""# ===== Cell 9：出图 + 落盘（秒级）=====
fig_dir = BASE / "outputs" / "figures"
plot_reachability(fig_dir)
plot_region_load(assign0, demand0, fig_dir)
# 读已有 pkl 中的任务 A 产物（emin/eps_results），多档图不因本 cell 退化
_s2 = {}
_p = BASE / "outputs" / "data" / "s2-results.pkl"
if _p.exists():
    with open(_p, "rb") as f:
        _s2 = pickle.load(f)
extra = _s2.get("eps_results", [])
emin_pt = {"E": _s2["emin"]["E"], "C": _s2["emin"]["C"]} if "emin" in _s2 else None
plot_epsilon(eta_results, C0, E0, fig_dir, extra=extra, emin=emin_pt)
plot_threshold_sensitivity(fig_dir)

out = {
    "C0": C0, "E0": E0,
    "baseline_regions": bl["meta"],
    "assign": assign0, "demand": demand0, "fail": fail0,
    "dest_capacity_aware": dest0,
    "eta_results": eta_results,
    "delay_ms": delay, "n_migrated": n_mig,
    "params": {"WIN": WIN, "W_EXT": W_EXT, "THRESHOLD": THRESHOLD,
               "MILP_TIME_LIMIT": MILP_TIME_LIMIT},
}
# 保留任务 A 的 emin/eps_results（本 cell 只更新本阶段字段，防覆盖丢失）
if "emin" in _s2:
    out["emin"] = _s2["emin"]
if "eps_results" in _s2:
    out["eps_results"] = _s2["eps_results"]
with open(BASE / "outputs" / "data" / "s2-results.pkl", "wb") as f:
    pickle.dump(out, f)
print("[OK] 已写入 outputs/data/s2-results.pkl + 4 张图（含多档曲线）")"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 重跑说明

- **二次运行**：Cell 1 秒级加载，Cell 3/5.x 命中缓存秒级，仅 Cell 1-2 重新执行
- **参数变更**：改 `sub2-model.py` 的 `WIN/W_EXT/THRESHOLD/MILP_TIME_LIMIT` 后，
  需将 Cell 1 的 `VER` +1（缓存指纹失效自动重算）
- **缓存位置**：`outputs/data/cache/s2_baseline.pkl`、`s2_eta_1.0/0.9/0.8.pkl`"""))

nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}

out_path = BASE / "outputs" / "notebooks" / "verify-sub2-model.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out_path)
print(f"[OK] 已生成 {out_path}（{len(cells)} cells）")

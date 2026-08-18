"""批量实验 + 参数对比。"""

import itertools
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.algorithm import ParticleSwarmOptimizer
from core.test_functions import get_function


def run_batch_experiment(func_name: str, custom_expr: str, dim: int,
                          param_grid: dict, n_runs: int = 3) -> list:
    func_info = get_function(func_name)

    if custom_expr:
        local_ns = {"np": np}
        fn = lambda x, expr=custom_expr, ns=local_ns: float(eval(expr, ns, {"x": x}))  # noqa
        bounds = np.array([[-10.0, 10.0]] * dim)
    else:
        fn = func_info["fn"]
        bounds = func_info["bounds"]
        if bounds.shape[0] < dim:
            bounds = np.tile(bounds[0], (dim, 1))

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    param_dicts = [{keys[i]: v[i] for i in range(len(keys))} for v in combinations]

    results = []
    total = len(param_dicts)
    progress = st.progress(0, text=f"批量实验: {total} 组 x {n_runs} 次...")

    for idx, p in enumerate(param_dicts):
        pso_params = {
            "objective_fn": fn, "bounds": bounds,
            "swarm_size": p.get("swarm_size", 50),
            "max_iterations": p.get("max_iterations", 300),
            "inertia": p.get("inertia", 0.7),
            "cognitive": p.get("cognitive", 1.5),
            "social": p.get("social", 1.5),
            "topology_name": p.get("topology", "Global"),
            "boundary": p.get("boundary", "clip"),
            "v_clamp_ratio": p.get("v_clamp_ratio", 0.2),
            "constriction": p.get("constriction", False),
            "seed": p.get("seed", 42),
        }

        run_results = []
        for run_idx in range(n_runs):
            if "seed" in pso_params:
                pso_params["seed"] = pso_params["seed"] + run_idx + idx * 1000
            pso = ParticleSwarmOptimizer(**pso_params)
            result = pso.run()
            run_results.append(result)

        best_energies = [r.best_energy for r in run_results]
        results.append({
            "params": p, "runs": run_results,
            "mean": np.mean(best_energies), "std": np.std(best_energies),
            "best": np.min(best_energies), "worst": np.max(best_energies),
            "avg_time": np.mean([r.total_time for r in run_results]),
            "avg_iters": np.mean([r.total_iterations for r in run_results]),
        })
        progress.progress((idx + 1) / total,
                          text=f"完成: {idx + 1}/{total} 组参数")

    progress.empty()
    return results


def render_experiment_results(experiment_results: list):
    if not experiment_results:
        st.info("请先运行批量实验。")
        return

    st.subheader("实验对比结果")
    import pandas as pd
    rows = []
    for i, r in enumerate(experiment_results):
        p = r["params"]
        rows.append({
            "#": i + 1,
            "拓扑": p.get("topology", ""),
            "惯性w": p.get("inertia", ""),
            "c₁": p.get("cognitive", ""),
            "c₂": p.get("social", ""),
            "粒子数": p.get("swarm_size", ""),
            "最优值均值": f"{r['mean']:.6g}",
            "标准差": f"{r['std']:.4g}",
            "最佳": f"{r['best']:.6g}",
            "耗时": f"{r['avg_time']:.1f}s",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Boxplot
    fig = go.Figure()
    for i, r in enumerate(experiment_results):
        energies = [run.best_energy for run in r["runs"]]
        label = f"#{i+1}: {r['params'].get('topology','?')} w={r['params'].get('inertia','?')}"
        fig.add_trace(go.Box(y=energies, name=label[:25]))
    fig.update_layout(title="各组参数最优能量分布", yaxis_title="目标函数值",
                      height=400, margin=dict(l=50, r=20, t=40, b=80))
    st.plotly_chart(fig, use_container_width=True)

    # 收敛曲线叠加
    st.subheader("收敛曲线对比")
    fig2 = go.Figure()
    for i, r in enumerate(experiment_results):
        hist = r["runs"][0].history
        if hist:
            iters = [h.iteration for h in hist]
            bests = [-h.best_fitness for h in hist]
            fig2.add_trace(go.Scatter(
                x=iters, y=bests, mode="lines", name=f"#{i+1}", line=dict(width=2)))
    fig2.update_layout(title="各组最优值的收敛过程", xaxis_title="迭代次数",
                       yaxis_title="最优能量", height=400,
                       margin=dict(l=50, r=20, t=40, b=40))
    st.plotly_chart(fig2, use_container_width=True)

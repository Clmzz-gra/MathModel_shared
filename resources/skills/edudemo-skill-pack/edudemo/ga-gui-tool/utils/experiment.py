"""批量实验 + 参数对比。"""

import itertools
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.algorithm import GeneticAlgorithm
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
    progress = st.progress(0, text=f"批量实验: {total} 组 × {n_runs} 次...")

    for idx, p in enumerate(param_dicts):
        ga_params = {
            "objective_fn": fn, "bounds": bounds,
            "pop_size": p.get("pop_size", 100),
            "max_generations": p.get("max_generations", 200),
            "selection_name": p.get("selection", "Tournament"),
            "crossover_name": p.get("crossover", "Uniform"),
            "mutation_name": p.get("mutation", "Gaussian"),
            "crossover_rate": p.get("crossover_rate", 0.8),
            "mutation_rate": p.get("mutation_rate", 0.1),
            "elite_count": p.get("elite_count", 2),
            "seed": p.get("seed", 42),
        }

        run_results = []
        for run_idx in range(n_runs):
            if "seed" in ga_params:
                ga_params["seed"] = ga_params["seed"] + run_idx + idx * 1000
            ga = GeneticAlgorithm(**ga_params)
            result = ga.run()
            run_results.append(result)

        best_energies = [r.best_energy for r in run_results]
        results.append({
            "params": p, "runs": run_results,
            "mean": np.mean(best_energies), "std": np.std(best_energies),
            "best": np.min(best_energies), "worst": np.max(best_energies),
            "avg_time": np.mean([r.total_time for r in run_results]),
            "avg_gens": np.mean([r.total_generations for r in run_results]),
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
            "选择": p.get("selection", ""),
            "交叉": p.get("crossover", ""),
            "变异": p.get("mutation", ""),
            "种群大小": p.get("pop_size", ""),
            "交叉率": p.get("crossover_rate", ""),
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
        label = f"#{i+1}: {r['params'].get('selection','?')}/{r['params'].get('crossover','?')}"
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
            gens = [h.generation for h in hist]
            bests = [-h.best_fitness for h in hist]
            fig2.add_trace(go.Scatter(
                x=gens, y=bests, mode="lines", name=f"#{i+1}", line=dict(width=2)))
    fig2.update_layout(title="各组最优值的收敛过程", xaxis_title="代数",
                       yaxis_title="最优能量", height=400,
                       margin=dict(l=50, r=20, t=40, b=40))
    st.plotly_chart(fig2, use_container_width=True)

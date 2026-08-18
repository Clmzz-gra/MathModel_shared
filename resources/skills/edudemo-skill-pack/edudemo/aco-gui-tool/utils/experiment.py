"""批量实验 + 参数对比（TSP 场景）。"""

import itertools
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.algorithm import AntColonyOptimizer
from core.tsp_problems import get_instance, build_distance_matrix


def run_batch_experiment(pattern: str, n_cities: int, city_seed: int,
                          param_grid: dict, n_runs: int = 3) -> list:
    cities = get_instance(pattern, n_cities, seed=city_seed)
    dist_matrix = build_distance_matrix(cities)

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    param_dicts = [{keys[i]: v[i] for i in range(len(keys))} for v in combinations]

    results = []
    total = len(param_dicts)
    progress = st.progress(0, text=f"批量实验: {total} 组 x {n_runs} 次...")

    for idx, p in enumerate(param_dicts):
        aco_params = {
            "distance_matrix": dist_matrix, "cities": cities,
            "n_ants": p.get("n_ants", 30),
            "max_iterations": p.get("max_iterations", 200),
            "alpha": p.get("alpha", 1.0),
            "beta": p.get("beta", 2.0),
            "rho": p.get("rho", 0.5),
            "q0": p.get("q0", 0.0),
            "elitist_weight": p.get("elitist_weight", 0.0),
            "mmas": p.get("mmas", False),
            "seed": p.get("seed", 42),
        }

        run_results = []
        for run_idx in range(n_runs):
            if "seed" in aco_params:
                aco_params["seed"] = aco_params["seed"] + run_idx + idx * 1000
            aco = AntColonyOptimizer(**aco_params)
            result = aco.run()
            run_results.append(result)

        best_dists = [r.best_distance for r in run_results]
        results.append({
            "params": p, "runs": run_results,
            "mean": np.mean(best_dists), "std": np.std(best_dists),
            "best": np.min(best_dists), "worst": np.max(best_dists),
            "avg_time": np.mean([r.total_time for r in run_results]),
            "avg_iters": np.mean([r.total_iterations for r in run_results]),
        })
        progress.progress((idx + 1) / total,
                          text=f"完成: {idx + 1}/{total} 组参数")

    progress.empty()
    return {"results": results, "cities": cities, "dist_matrix": dist_matrix}


def render_experiment_results(experiment_data: dict):
    results = experiment_data.get("results", [])
    cities = experiment_data.get("cities")

    if not results:
        st.info("请先运行批量实验。")
        return

    st.subheader("实验对比结果")
    import pandas as pd
    rows = []
    for i, r in enumerate(results):
        p = r["params"]
        variant = "AS"
        if p.get("q0", 0) > 0:
            variant = "ACS"
        if p.get("mmas", False):
            variant = "MMAS"
        if p.get("elitist_weight", 0) > 0:
            variant = "Elitist"

        rows.append({
            "#": i + 1,
            "变体": variant,
            "α": p.get("alpha", ""),
            "β": p.get("beta", ""),
            "ρ": p.get("rho", ""),
            "最优均值": f"{r['mean']:.4f}",
            "标准差": f"{r['std']:.4f}",
            "最佳": f"{r['best']:.4f}",
            "耗时": f"{r['avg_time']:.1f}s",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Boxplot
    fig = go.Figure()
    for i, r in enumerate(results):
        dists = [run.best_distance for run in r["runs"]]
        p = r["params"]
        variant = "AS"
        if p.get("q0", 0) > 0:
            variant = "ACS"
        if p.get("mmas", False):
            variant = "MMAS"
        label = f"#{i+1}: {variant} α={p.get('alpha','?')}"
        fig.add_trace(go.Box(y=dists, name=label[:25]))
    fig.update_layout(title="各组参数最优路径长度分布", yaxis_title="路径长度",
                      height=400, margin=dict(l=50, r=20, t=40, b=80))
    st.plotly_chart(fig, use_container_width=True)

    # 收敛曲线
    st.subheader("收敛曲线对比")
    fig2 = go.Figure()
    for i, r in enumerate(results):
        hist = r["runs"][0].history
        if hist:
            iters = [h.iteration for h in hist]
            bests = [h.best_distance for h in hist]
            fig2.add_trace(go.Scatter(
                x=iters, y=bests, mode="lines", name=f"#{i+1}", line=dict(width=2)))
    fig2.update_layout(title="各组最优路径的收敛过程", xaxis_title="迭代次数",
                       yaxis_title="最优路径长度", height=400,
                       margin=dict(l=50, r=20, t=40, b=40))
    st.plotly_chart(fig2, use_container_width=True)

"""批量实验 + 参数对比工具。"""

import time
import itertools

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.annealing import SimulatedAnnealing
from core.test_functions import get_function


def run_batch_experiment(
    func_name: str,
    custom_expr: str,
    dim: int,
    param_grid: dict,
    n_runs: int = 3,
) -> list:
    """对参数网格中的每种组合运行 n_runs 次 SA。

    Returns:
        [{"params": {...}, "runs": [SAResult, ...], "mean": float, "std": float, "best": float, ...}, ...]
    """
    func_info = get_function(func_name)

    # 构建目标函数
    if custom_expr:
        local_ns = {"np": np}
        fn = lambda x, expr=custom_expr, ns=local_ns: float(eval(expr, ns, {"x": x}))  # noqa
        bounds = np.array([[-10.0, 10.0]] * dim)
    else:
        fn = func_info["fn"]
        bounds = func_info["bounds"]
        # 对高维函数复制 bounds
        if bounds.shape[0] == 2 and dim > 2:
            bounds = np.tile(bounds, (dim, 1))

    # 生成所有参数组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    param_dicts = [{keys[i]: v[i] for i in range(len(keys))} for v in combinations]

    results = []
    total_combos = len(param_dicts)
    progress = st.progress(0, text=f"开始批量实验: {total_combos} 组参数 × {n_runs} 次运行...")

    for idx, p in enumerate(param_dicts):
        sa_params = {
            "objective_fn": fn,
            "bounds": bounds,
            "T0": p.get("T0", 1000),
            "T_end": p.get("T_end", 0.01),
            "max_iter": p.get("max_iter", 5000),
            "cooling_schedule": p.get("cooling", "Geometric"),
            "cooling_params": {"alpha": p.get("alpha", 0.95)},
            "neighborhood": p.get("neighborhood", "Gaussian"),
            "acceptance": p.get("acceptance", "Metropolis"),
            "seed": p.get("seed", 42),
        }

        run_results = []
        for run_idx in range(n_runs):
            if "seed" in sa_params:
                sa_params["seed"] = sa_params["seed"] + run_idx + idx * 1000
            sa = SimulatedAnnealing(**sa_params)
            sa.initialize()
            result = sa.run()
            run_results.append(result)

        best_energies = [r.best_energy for r in run_results]
        results.append({
            "params": p,
            "runs": run_results,
            "mean": np.mean(best_energies),
            "std": np.std(best_energies),
            "best": np.min(best_energies),
            "worst": np.max(best_energies),
            "avg_time": np.mean([r.total_time for r in run_results]),
            "avg_iter": np.mean([r.total_iterations for r in run_results]),
        })

        progress.progress((idx + 1) / total_combos,
                          text=f"完成: {idx + 1}/{total_combos} 组参数")

    progress.empty()
    return results


def render_experiment_results(experiment_results: list):
    """渲染批量实验结果表格和对比图。"""
    if not experiment_results:
        st.info("请先运行批量实验。")
        return

    st.subheader("实验对比结果")

    # 构建表格
    rows = []
    for i, r in enumerate(experiment_results):
        p = r["params"]
        row = {
            "#": i + 1,
            "冷却策略": p.get("cooling", "Geometric"),
            "邻域": p.get("neighborhood", "Gaussian"),
            "接受准则": p.get("acceptance", "Metropolis"),
            "T₀": p.get("T0", ""),
            "α": p.get("alpha", ""),
            "最优值 (均值)": f"{r['mean']:.6g}",
            "标准差": f"{r['std']:.4g}",
            "最优值 (最佳)": f"{r['best']:.6g}",
            "平均耗时": f"{r['avg_time']:.1f}s",
        }
        rows.append(row)

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Boxplot 对比
    fig = go.Figure()
    for i, r in enumerate(experiment_results):
        energies = [run.best_energy for run in r["runs"]]
        label = f"#{i+1}: {r['params'].get('cooling','?')}/{r['params'].get('neighborhood','?')}"
        fig.add_trace(go.Box(y=energies, name=label[:25]))

    fig.update_layout(
        title="各组参数最优能量分布",
        yaxis_title="目标函数值",
        height=400,
        margin=dict(l=50, r=20, t=40, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 收敛曲线叠加 (仅第一组参数的代表性运行)
    st.subheader("收敛曲线对比")
    fig2 = go.Figure()
    for i, r in enumerate(experiment_results):
        hist = r["runs"][0].history
        if hist:
            iters = [h.iteration for h in hist]
            bests = [h.best_energy for h in hist]
            label = f"#{i+1}"
            fig2.add_trace(go.Scatter(
                x=iters, y=bests, mode="lines", name=label,
                line=dict(width=2),
            ))
    fig2.update_layout(
        title="各组最优能量的收敛过程",
        xaxis_title="迭代步数",
        yaxis_title="最优能量",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

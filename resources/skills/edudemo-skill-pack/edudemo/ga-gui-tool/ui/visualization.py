"""过程可视化 Tab — 适应度曲线、多样性、种群分布。"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_fitness_plot(history: list) -> go.Figure:
    """最优/平均/最差适应度 vs 代数。"""
    if not history:
        return go.Figure()

    gens = [h.generation for h in history]
    best = [h.best_fitness for h in history]
    mean = [h.mean_fitness for h in history]
    worst = [h.worst_fitness for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gens, y=worst, mode="lines", name="最差",
        line=dict(color="#e74c3c", width=1, dash="dot"), opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=mean, mode="lines", name="平均",
        line=dict(color="#f39c12", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=best, mode="lines", name="最优",
        line=dict(color="#27ae60", width=2.5),
    ))
    fig.update_layout(
        title="种群适应度进化曲线",
        xaxis_title="代数",
        yaxis_title="适应度 (fitness = -f(x))",
        hovermode="x unified",
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_diversity_plot(history: list) -> go.Figure:
    """种群多样性变化。"""
    if not history:
        return go.Figure()

    gens = [h.generation for h in history]
    div = [h.diversity for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gens, y=div, mode="lines", name="多样性",
        line=dict(color="#9b59b6", width=2),
        fill="tozeroy", fillcolor="rgba(155,89,182,0.1)",
    ))
    fig.update_layout(
        title="种群多样性变化",
        xaxis_title="代数",
        yaxis_title="多样性 (标准差/搜索范围)",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def render_population_scatter_2d(history: list, fn, bounds, func_name: str, gen_idx: int = -1) -> go.Figure:
    """当前代种群在 2D 搜索空间的散点分布 + 等高线背景。"""
    if not history:
        return go.Figure()

    entry = history[gen_idx]
    pop = entry.population
    fits = entry.fitnesses

    x_min, x_max = bounds[0, 0], bounds[0, 1]
    y_min, y_max = bounds[1, 0], bounds[1, 1]

    nx, ny = 50, 50
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(nx):
        for j in range(ny):
            Z[j, i] = fn(np.array([X[j, i], Y[j, i]]))

    fig = go.Figure()
    # 等高线填充
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z,
        colorscale="Greys",
        contours=dict(coloring="heatmap", showlabels=False),
        showscale=False,
        opacity=0.3,
        name="函数表面",
    ))
    # 种群散点
    fig.add_trace(go.Scatter(
        x=pop[:, 0], y=pop[:, 1], mode="markers",
        marker=dict(
            size=8,
            color=fits,
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="适应度", x=1.02),
            line=dict(width=0.5, color="black"),
        ),
        name=f"第 {entry.generation} 代种群",
    ))
    # 最优个体
    best_idx = np.argmax(fits)
    fig.add_trace(go.Scatter(
        x=[pop[best_idx, 0]], y=[pop[best_idx, 1]],
        mode="markers",
        marker=dict(color="red", size=14, symbol="star", line=dict(width=1, color="white")),
        name="最优个体",
    ))

    fig.update_layout(
        title=f"种群分布 — {func_name} (第 {entry.generation} 代)",
        xaxis_title="x₁", yaxis_title="x₂",
        height=450,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


def render_progress_bar(current_gen: int, max_gen: int, best_fitness: float,
                         mean_fitness: float, elapsed: float, is_paused: bool):
    """渲染进度条和状态信息。"""
    progress = min(current_gen / max_gen, 1.0)
    st.progress(progress, text=f"进度: {current_gen}/{max_gen} 代  ({progress*100:.1f}%)")

    cols = st.columns(4)
    cols[0].metric("最优适应度", f"{best_fitness:.4g}")
    cols[1].metric("平均适应度", f"{mean_fitness:.4g}")
    cols[2].metric("已用时间", f"{elapsed:.1f}s")
    cols[3].metric("状态", "⏸️ 已暂停" if is_paused else "▶️ 运行中")

"""过程可视化 Tab — 适应度曲线、多样性、粒子分布。"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_fitness_plot(history: list) -> go.Figure:
    """最优/平均/最差适应度 vs 迭代。"""
    if not history:
        return go.Figure()

    iters = [h.iteration for h in history]
    best = [h.best_fitness for h in history]
    mean = [h.mean_fitness for h in history]
    worst = [h.worst_fitness for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=worst, mode="lines", name="最差",
        line=dict(color="#e74c3c", width=1, dash="dot"), opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=iters, y=mean, mode="lines", name="平均",
        line=dict(color="#f39c12", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=iters, y=best, mode="lines", name="最优",
        line=dict(color="#27ae60", width=2.5),
    ))
    fig.update_layout(
        title="粒子群适应度进化曲线",
        xaxis_title="迭代次数",
        yaxis_title="适应度 (fitness = -f(x))",
        hovermode="x unified",
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_diversity_plot(history: list) -> go.Figure:
    """粒子群多样性与平均速度变化。"""
    if not history:
        return go.Figure()

    iters = [h.iteration for h in history]
    div = [h.diversity for h in history]
    avg_v = [h.avg_velocity for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=div, mode="lines", name="多样性",
        line=dict(color="#9b59b6", width=2),
        fill="tozeroy", fillcolor="rgba(155,89,182,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=iters, y=avg_v, mode="lines", name="平均速度",
        line=dict(color="#e67e22", width=1.5, dash="dash"),
        yaxis="y2",
    ))
    fig.update_layout(
        title="粒子群多样性 & 平均速度",
        xaxis_title="迭代次数",
        yaxis_title="多样性",
        yaxis2=dict(title="平均速度", overlaying="y", side="right"),
        height=300,
        margin=dict(l=50, r=50, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_particle_scatter_2d(history: list, fn, bounds, func_name: str, gen_idx: int = -1) -> go.Figure:
    """当前代粒子在 2D 搜索空间的分布 + 等高线 + 速度箭头。"""
    if not history:
        return go.Figure()

    entry = history[gen_idx]
    pos = entry.positions
    vel = entry.velocities
    fits = entry.fitnesses

    x_min, x_max = bounds[0, 0], bounds[0, 1]
    y_min, y_max = bounds[1, 0], bounds[1, 1]

    # 等高线
    nx, ny = 50, 50
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(nx):
        for j in range(ny):
            Z[j, i] = fn(np.array([X[j, i], Y[j, i]]))

    fig = go.Figure()

    # 等高线
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z,
        colorscale="Greys",
        contours=dict(coloring="heatmap", showlabels=False),
        showscale=False,
        opacity=0.3,
        name="函数表面",
    ))

    # 粒子位置
    fig.add_trace(go.Scatter(
        x=pos[:, 0], y=pos[:, 1], mode="markers",
        marker=dict(
            size=8,
            color=fits,
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="适应度", x=1.02),
            line=dict(width=0.5, color="black"),
        ),
        name=f"第 {entry.iteration} 代粒子",
    ))

    # 速度箭头（采样，太多会乱）
    n_arrows = min(len(pos), 20)
    arrow_idx = np.random.choice(len(pos), n_arrows, replace=False)
    vel_scale = 0.5 * (x_max - x_min) / (np.max(np.abs(vel)) + 1e-10)
    for idx in arrow_idx:
        fig.add_annotation(
            x=pos[idx, 0] + vel[idx, 0] * vel_scale * 0.3,
            y=pos[idx, 1] + vel[idx, 1] * vel_scale * 0.3,
            ax=pos[idx, 0], ay=pos[idx, 1],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowcolor="rgba(52,152,219,0.5)", arrowwidth=1,
            text="", opacity=0.5,
        )

    # 全局最优
    best_idx = np.argmax(fits)
    fig.add_trace(go.Scatter(
        x=[pos[best_idx, 0]], y=[pos[best_idx, 1]],
        mode="markers",
        marker=dict(color="red", size=14, symbol="star", line=dict(width=1, color="white")),
        name="全局最优",
    ))

    fig.update_layout(
        title=f"粒子分布 — {func_name} (第 {entry.iteration} 代)",
        xaxis_title="x₁", yaxis_title="x₂",
        height=450,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


def render_progress_bar(current_iter: int, max_iter: int, best_fitness: float,
                         mean_fitness: float, elapsed: float, is_paused: bool):
    """渲染进度条和状态信息。"""
    progress = min(current_iter / max_iter, 1.0)
    st.progress(progress, text=f"进度: {current_iter}/{max_iter} 迭代  ({progress*100:.1f}%)")

    cols = st.columns(4)
    cols[0].metric("最优适应度", f"{best_fitness:.4g}")
    cols[1].metric("平均适应度", f"{mean_fitness:.4g}")
    cols[2].metric("已用时间", f"{elapsed:.1f}s")
    cols[3].metric("状态", "⏸️ 已暂停" if is_paused else "▶️ 运行中")

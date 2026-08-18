"""过程可视化 Tab — TSP 路径图、收敛曲线、信息素热力图。"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_tour_plot(cities: np.ndarray, tour: np.ndarray, best_distance: float,
                      title: str = "最优路径") -> go.Figure:
    """绘制 TSP 城市分布 + 最优路径。"""
    fig = go.Figure()

    # 城市点
    n = len(cities)
    fig.add_trace(go.Scatter(
        x=cities[:, 0], y=cities[:, 1],
        mode="markers+text",
        marker=dict(size=10, color="#2c3e50", line=dict(width=1, color="white")),
        text=[str(i) for i in range(n)],
        textposition="top center",
        textfont=dict(size=9, color="#7f8c8d"),
        name=f"{n} 城市",
    ))

    # 路径线
    if tour is not None and len(tour) > 0:
        # 确保路径是城市索引
        if tour.dtype == np.int64 or tour.dtype == np.int32 or tour.dtype == int:
            tour_coords = cities[tour]
        else:
            tour_coords = tour

        fig.add_trace(go.Scatter(
            x=tour_coords[:, 0], y=tour_coords[:, 1],
            mode="lines+markers",
            line=dict(color="#e74c3c", width=1.5),
            marker=dict(size=3, color="#e74c3c"),
            name=f"最优路径 ({best_distance:.3f})",
        ))

        # 起点标记
        start = cities[tour[0]] if tour.dtype in [np.int64, np.int32, int] else tour[0]
        fig.add_trace(go.Scatter(
            x=[start[0]], y=[start[1]],
            mode="markers",
            marker=dict(color="#27ae60", size=16, symbol="star",
                         line=dict(width=1.5, color="white")),
            name="起点",
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="x", range=[-0.05, 1.05], showgrid=False, zeroline=False),
        yaxis=dict(title="y", range=[-0.05, 1.05], showgrid=False, zeroline=False),
        height=450,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


def render_convergence_plot(history: list) -> go.Figure:
    """最优/平均路径长度 vs 迭代。"""
    if not history:
        return go.Figure()

    iters = [h.iteration for h in history]
    best = [h.best_distance for h in history]
    mean = [h.mean_distance for h in history]
    worst = [h.worst_distance for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=worst, mode="lines", name="最差",
        line=dict(color="#e74c3c", width=1, dash="dot"), opacity=0.4,
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
        title="路径长度收敛曲线",
        xaxis_title="迭代次数",
        yaxis_title="路径长度",
        hovermode="x unified",
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_diversity_plot(history: list) -> go.Figure:
    """路径多样性 & 信息素浓度变化。"""
    if not history:
        return go.Figure()

    iters = [h.iteration for h in history]
    diversity = [h.diversity for h in history]
    avg_pheromone = [float(np.mean(h.pheromone_matrix[h.pheromone_matrix > 0]))
                     if np.any(h.pheromone_matrix > 0) else 0
                     for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=diversity, mode="lines", name="路径多样性",
        line=dict(color="#9b59b6", width=2),
        fill="tozeroy", fillcolor="rgba(155,89,182,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=iters, y=avg_pheromone, mode="lines", name="平均信息素",
        line=dict(color="#e67e22", width=1.5, dash="dash"),
        yaxis="y2",
    ))
    fig.update_layout(
        title="路径多样性 & 平均信息素浓度",
        xaxis_title="迭代次数",
        yaxis_title="多样性 (唯一路径/总蚂蚁)",
        yaxis2=dict(title="平均信息素", overlaying="y", side="right"),
        height=300,
        margin=dict(l=50, r=50, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_pheromone_heatmap(pheromone: np.ndarray, n: int = None) -> go.Figure:
    """信息素矩阵热力图（取样显示，太大的矩阵全显示看不清）。"""
    if pheromone is None:
        return go.Figure()

    if n is None:
        n = pheromone.shape[0]

    # 如果城市太多，抽样显示
    if n > 50:
        step = n // 50
        sample = pheromone[::step, ::step]
    else:
        sample = pheromone

    fig = go.Figure(data=go.Heatmap(
        z=sample,
        colorscale="YlOrRd",
        showscale=True,
        colorbar=dict(title="信息素浓度", x=1.02),
    ))
    fig.update_layout(
        title="信息素矩阵 (颜色越深 → 信息素越多 → 路径越受偏爱)",
        xaxis_title="城市 i",
        yaxis_title="城市 j",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def render_progress_bar(current_iter: int, max_iter: int, best_distance: float,
                         mean_distance: float, elapsed: float, is_paused: bool):
    """渲染进度条和状态信息。"""
    progress = min(current_iter / max_iter, 1.0)
    st.progress(progress, text=f"进度: {current_iter}/{max_iter} 迭代  ({progress*100:.1f}%)")

    cols = st.columns(4)
    cols[0].metric("最优路径", f"{best_distance:.3f}")
    cols[1].metric("平均路径", f"{mean_distance:.3f}")
    cols[2].metric("已用时间", f"{elapsed:.1f}s")
    cols[3].metric("状态", "⏸️ 已暂停" if is_paused else "▶️ 运行中")

"""过程可视化 Tab — 实时收敛图、温度曲线、搜索轨迹等。"""

import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def render_convergence_plot(history: list) -> go.Figure:
    """收敛曲线：当前能量 + 最优能量 vs 迭代步数。"""
    if not history:
        return go.Figure()

    iters = [h.iteration for h in history]
    current_e = [h.current_energy for h in history]
    best_e = [h.best_energy for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=current_e, mode="lines", name="当前解能量",
        line=dict(color="#aaaacc", width=0.8), opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=iters, y=best_e, mode="lines", name="最优解能量",
        line=dict(color="#3366cc", width=2.5)
    ))
    fig.update_layout(
        title="收敛曲线",
        xaxis_title="迭代步数",
        yaxis_title="目标函数值",
        hovermode="x unified",
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    use_log = "log" if max(best_e) > 100 else "linear"
    fig.update_layout(yaxis=dict(type=use_log))
    return fig


def render_temperature_plot(history: list) -> go.Figure:
    """温度下降曲线。"""
    if not history:
        return go.Figure()

    iters = [h.iteration for h in history]
    temps = [h.temperature for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=temps, mode="lines", name="温度",
        line=dict(color="#e74c3c", width=2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.1)"
    ))
    fig.update_layout(
        title="温度下降曲线",
        xaxis_title="迭代步数",
        yaxis_title="温度 T",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    fig.update_layout(yaxis=dict(type="log"))
    return fig


def render_acceptance_plot(history: list) -> go.Figure:
    """接受率随时间变化。"""
    if not history:
        return go.Figure()

    iters = [h.iteration for h in history]
    rates = [h.acceptance_rate for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=rates, mode="lines", name="累计接受率",
        line=dict(color="#27ae60", width=2),
        fill="tozeroy", fillcolor="rgba(39,174,96,0.1)"
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="50%")
    fig.update_layout(
        title="接受率变化",
        xaxis_title="迭代步数",
        yaxis_title="累计接受率",
        yaxis_range=[0, 1],
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def render_search_path_2d(history: list, fn, bounds, func_name: str) -> go.Figure:
    """2D 等高线 + 搜索路径。"""
    if not history:
        return go.Figure()

    x_min, x_max = bounds[0, 0], bounds[0, 1]
    y_min, y_max = bounds[1, 0], bounds[1, 1]

    # 生成网格（降低分辨率加速渲染）
    nx, ny = 50, 50
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(nx):
        for j in range(ny):
            Z[j, i] = fn(np.array([X[j, i], Y[j, i]]))

    # 路径
    path_x = [h.current_x[0] for h in history]
    path_y = [h.current_x[1] for h in history]

    best = min(history, key=lambda h: h.best_energy)

    fig = go.Figure()

    # 等高线（填充模式，无白色空隙）
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z,
        colorscale="Viridis",
        contours=dict(
            coloring="heatmap",
            showlabels=True,
            labelfont=dict(size=9, color="#333333"),
        ),
        colorbar=dict(title=""),
        showscale=True,
        name="函数表面",
    ))

    # 搜索路径
    fig.add_trace(go.Scatter(
        x=path_x, y=path_y, mode="lines",
        line=dict(color="red", width=1.5), name="搜索路径", opacity=0.4,
    ))
    # 起点
    fig.add_trace(go.Scatter(
        x=[path_x[0]], y=[path_y[0]], mode="markers",
        marker=dict(color="#e74c3c", size=10, symbol="circle"),
        name="起点",
    ))
    # 终点
    fig.add_trace(go.Scatter(
        x=[best.current_x[0]], y=[best.current_x[1]], mode="markers",
        marker=dict(color="#2ecc71", size=12, symbol="star"),
        name="最优解",
    ))

    fig.update_layout(
        title=f"2D 搜索路径 — {func_name}",
        xaxis_title="x₁",
        yaxis_title="x₂",
        height=450,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        coloraxis_colorbar=dict(x=1.02),
    )
    return fig


def render_search_path_3d(history: list, fn, bounds, dim: int, func_name: str) -> go.Figure:
    """3D 表面 + 搜索轨迹。取前两维显示，其余维固定为最优值。"""
    if not history or dim < 2:
        return go.Figure()

    best = min(history, key=lambda h: h.best_energy)
    fixed_vals = best.best_x.copy()

    x_min, x_max = bounds[0, 0], bounds[0, 1]
    y_min, y_max = bounds[1, 0], bounds[1, 1]

    nx, ny = 40, 40
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(nx):
        for j in range(ny):
            pt = fixed_vals.copy()
            pt[0] = X[j, i]
            pt[1] = Y[j, i]
            Z[j, i] = fn(pt)

    path_x = [h.current_x[0] for h in history]
    path_y = [h.current_x[1] for h in history]
    path_z = [h.current_energy for h in history]

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=Z,
        colorscale="Viridis",
        opacity=0.7,
        showscale=False,
        name="函数表面",
    ))
    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode="lines",
        line=dict(color="red", width=3),
        name="搜索轨迹",
    ))
    fig.add_trace(go.Scatter3d(
        x=[best.best_x[0]], y=[best.best_x[1]], z=[best.best_energy],
        mode="markers",
        marker=dict(color="#00ff00", size=8, symbol="diamond"),
        name="最优解",
    ))

    fig.update_layout(
        title=f"3D 搜索轨迹 — {func_name}",
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="x₁", yaxis_title="x₂", zaxis_title="f(x)"
        ),
    )
    return fig


def render_progress_bar(current_iter: int, max_iter: int, temperature: float,
                         best_energy: float, elapsed: float, is_paused: bool):
    """渲染进度条和状态信息。"""
    progress = min(current_iter / max_iter, 1.0)
    st.progress(progress, text=f"进度: {current_iter}/{max_iter}  ({progress*100:.1f}%)")

    cols = st.columns(4)
    cols[0].metric("当前温度", f"{temperature:.4g}")
    cols[1].metric("最优能量", f"{best_energy:.6g}")
    cols[2].metric("已用时间", f"{elapsed:.1f}s")
    cols[3].metric("状态", "⏸️ 已暂停" if is_paused else "▶️ 运行中")

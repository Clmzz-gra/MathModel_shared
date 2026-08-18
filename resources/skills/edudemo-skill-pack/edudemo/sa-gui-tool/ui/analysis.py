"""数值分析 Tab — 最终结果卡片、收敛统计、解质量评估。"""

import io
import json
import csv

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_result_cards(result, func_info: dict, params: dict):
    """最终结果摘要卡片。"""
    st.subheader("优化结果")

    gap = abs(result.best_energy - func_info["global_min"]) if func_info else None

    cols = st.columns(4)
    cols[0].metric(
        "最优值 f(x*)",
        f"{result.best_energy:.8g}",
        delta=f"距全局最优: {gap:.3g}" if gap is not None else None,
        delta_color="inverse",
    )
    cols[1].metric("总迭代", f"{result.total_iterations}")
    cols[2].metric("总耗时", f"{result.total_time:.2f}s")
    cols[3].metric("终止温度", f"{result.final_temperature:.6g}")

    if gap is not None and gap < 1e-6:
        st.success("已找到全局最优解。", icon="🎯")
    elif gap is not None and gap < 0.1:
        st.info("接近全局最优解。")
    elif gap is not None:
        st.warning("离全局最优解还有距离，可尝试增加迭代次数或调整参数。")

    # 最优解详细
    st.markdown("**最优解 x*:**")
    x_str = ", ".join([f"x_{{{i+1}}} = {v:.6g}" for i, v in enumerate(result.best_x)])
    st.latex(rf"[{x_str}]")


def render_convergence_stats(history: list, func_info: dict):
    """收敛统计：到达各误差限所需的步数。"""
    if not history or not func_info:
        return

    global_min = func_info["global_min"]
    best_energies = np.array([h.best_energy for h in history])
    gap = np.abs(best_energies - global_min)

    st.subheader("收敛统计")

    thresholds = [
        ("10% 误差", 0.1),
        ("1% 误差", 0.01),
        ("0.1% 误差", 0.001),
        ("0.01% 误差", 0.0001),
    ]
    results = []
    base_range = func_info["bounds"][:, 1] - func_info["bounds"][:, 0]
    energy_scale = np.mean(base_range) if gap[0] < 1e-10 else abs(gap[0])

    for label, ratio in thresholds:
        threshold = max(energy_scale * ratio, 1e-10)
        idx = np.argmax(gap <= threshold)
        if gap[idx] <= threshold:
            results.append({
                "阈值": label,
                "首次到达步数": idx,
                "占总迭代比例": f"{idx / len(history) * 100:.1f}%",
                "对应温度": f"{history[idx].temperature:.4g}" if idx < len(history) else "N/A",
            })
        else:
            results.append({
                "阈值": label,
                "首次到达步数": "未到达",
                "占总迭代比例": "-",
                "对应温度": "-",
            })

    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_acceptance_breakdown(history: list, n_segments: int = 5):
    """分温度段接受率统计。"""
    if not history:
        return

    st.subheader("分温度段接受率")

    n = len(history)
    if n < n_segments:
        return

    seg_size = n // n_segments
    segments = []
    for i in range(n_segments):
        start = i * seg_size
        end = start + seg_size if i < n_segments - 1 else n
        seg_history = history[start:end]
        accepted_count = sum(1 for h in seg_history if h.accepted)
        seg_rate = accepted_count / len(seg_history) if seg_history else 0
        avg_temp = np.mean([h.temperature for h in seg_history])
        segments.append({
            "段": f"第 {i+1} 段",
            "迭代范围": f"{start}-{end}",
            "平均温度": f"{avg_temp:.4g}",
            "接受率": f"{seg_rate:.2%}",
            "接受/总数": f"{accepted_count}/{len(seg_history)}",
        })

    st.dataframe(pd.DataFrame(segments), use_container_width=True, hide_index=True)

    # 柱状图
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[s["段"] for s in segments],
        y=[float(s["接受率"].strip("%")) / 100 for s in segments],
        marker=dict(
            color=[float(s["平均温度"]) for s in segments],
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="平均温度"),
        ),
    ))
    fig.update_layout(
        title="各温度段接受率",
        yaxis=dict(title="接受率", tickformat=".0%", range=[0, 1]),
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_multi_run_stats(all_results: list):
    """多次运行的统计比较。"""
    if not all_results:
        return

    st.subheader("多次运行统计")

    bests = [r.best_energy for r in all_results]
    times = [r.total_time for r in all_results]
    iters = [r.total_iterations for r in all_results]

    stats = {
        "统计量": ["最优值", "均值", "最差值", "标准差"],
        "最优能量": [
            f"{min(bests):.6g}",
            f"{np.mean(bests):.6g}",
            f"{max(bests):.6g}",
            f"{np.std(bests):.6g}",
        ],
        "耗时 (s)": [
            f"{min(times):.2f}",
            f"{np.mean(times):.2f}",
            f"{max(times):.2f}",
            f"{np.std(times):.2f}",
        ],
        "迭代数": [
            f"{min(iters)}",
            f"{np.mean(iters):.0f}",
            f"{max(iters)}",
            f"{np.std(iters):.0f}",
        ],
    }
    st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

    # boxplot
    fig = go.Figure()
    fig.add_trace(go.Box(y=bests, name="最优能量", marker_color="#3366cc"))
    fig.update_layout(
        title="多次运行最优能量分布",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_export_section(result, history: list, func_name: str):
    """导出按钮：CSV / JSON 下载。"""
    st.subheader("导出结果")

    col1, col2 = st.columns(2)

    with col1:
        # CSV
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["iteration", "temperature", "current_energy", "best_energy",
                          "accepted", "acceptance_rate", "elapsed"])
        for h in history:
            writer.writerow([
                h.iteration, h.temperature, h.current_energy, h.best_energy,
                h.accepted, h.acceptance_rate, h.elapsed,
            ])
        st.download_button(
            "下载 CSV", csv_buf.getvalue(),
            file_name=f"sa-{func_name}-history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        # JSON
        json_data = {
            "function": func_name,
            "best_energy": result.best_energy,
            "best_x": result.best_x.tolist(),
            "total_iterations": result.total_iterations,
            "total_time": result.total_time,
            "final_temperature": result.final_temperature,
            "total_acceptance_rate": result.total_acceptance_rate,
            "history_summary": {
                "iterations": [h.iteration for h in history],
                "temperatures": [h.temperature for h in history],
                "best_energies": [h.best_energy for h in history],
            },
        }
        st.download_button(
            "下载 JSON", json.dumps(json_data, indent=2, ensure_ascii=False),
            file_name=f"sa-{func_name}-result.json",
            mime="application/json",
            use_container_width=True,
        )

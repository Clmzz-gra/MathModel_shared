"""数值分析 Tab — 结果卡片、收敛统计、路径分布、导出。"""

import io, json, csv
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_result_cards(result, cities: np.ndarray):
    st.subheader("优化结果")

    cols = st.columns(4)
    cols[0].metric("最优路径长度", f"{result.best_distance:.6f}")
    cols[1].metric("迭代次数", f"{result.total_iterations}")
    cols[2].metric("总评估次数", f"{result.total_evaluations}")
    cols[3].metric("总耗时", f"{result.total_time:.2f}s")

    # 路径改善幅度
    if len(result.history) > 1:
        initial_best = result.history[0].best_distance
        improvement = (initial_best - result.best_distance) / initial_best * 100
        st.info(f"路径缩短了 {improvement:.1f}%（{initial_best:.3f} → {result.best_distance:.3f}）")

    n = len(cities)
    st.markdown(f"**最优路径 ({n} 城市):**")
    tour_str = " → ".join([str(c) for c in result.best_tour[:10]])
    if len(result.best_tour) > 10:
        tour_str += f" → ... → {result.best_tour[-1]}"
        tour_str += f"（{len(result.best_tour)} 步）"
    st.markdown(f"`{tour_str}`")


def render_convergence_stats(history: list):
    """收敛迭代统计。"""
    if not history:
        return

    st.subheader("收敛统计")
    distances = np.array([h.best_distance for h in history])
    final = distances[-1]

    thresholds = [0.2, 0.1, 0.05, 0.02]
    first = np.max(distances) - np.min(distances)
    scale = first if first > 1e-10 else 1.0
    results = []
    for t in thresholds:
        threshold_val = final + scale * t
        idx_arr = np.where(distances <= threshold_val)[0]
        if len(idx_arr) > 0:
            results.append({
                "相对误差": f"{t*100:.0f}%",
                "到达迭代": idx_arr[0],
                "总迭代占比": f"{idx_arr[0]/len(history)*100:.0f}%",
            })
        else:
            results.append({"相对误差": f"{t*100:.0f}%", "到达迭代": "未到达", "总迭代占比": "-"})

    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)


def render_tour_stats(history: list):
    """最终迭代的路径统计。"""
    if not history:
        return
    last = history[-1]
    dists = last.all_distances

    st.subheader("最终迭代路径统计")
    cols = st.columns(4)
    cols[0].metric("最优距离", f"{np.min(dists):.3f}")
    cols[1].metric("平均距离", f"{np.mean(dists):.3f}")
    cols[2].metric("标准差", f"{np.std(dists):.3f}")
    cols[3].metric("多样性", f"{last.diversity:.2f}")

    # 路径长度分布
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=dists, nbinsx=20, marker_color="#3498db",
                               name="路径长度分布"))
    fig.update_layout(
        title="最终迭代路径长度分布",
        xaxis_title="路径长度",
        yaxis_title="频数（蚂蚁数）",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_multi_run_stats(all_results: list):
    """多次运行统计。"""
    if not all_results:
        return

    st.subheader("多次运行统计")
    bests = [r.best_distance for r in all_results]
    times = [r.total_time for r in all_results]

    stats = {
        "统计量": ["最优值", "均值", "最差值", "标准差"],
        "路径长度": [f"{min(bests):.4f}", f"{np.mean(bests):.4f}",
                    f"{max(bests):.4f}", f"{np.std(bests):.4f}"],
        "耗时(s)": [f"{min(times):.2f}", f"{np.mean(times):.2f}",
                    f"{max(times):.2f}", f"{np.std(times):.2f}"],
    }
    st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Box(y=bests, name="最优路径", marker_color="#27ae60"))
    fig.update_layout(title="多次运行最优路径长度分布", height=300,
                      margin=dict(l=50, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


def render_export_section(result, history: list, cities: np.ndarray):
    st.subheader("导出结果")
    col1, col2 = st.columns(2)

    with col1:
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["iteration", "best_distance", "mean_distance", "worst_distance",
                          "diversity", "elapsed"])
        for h in history:
            writer.writerow([h.iteration, h.best_distance, h.mean_distance,
                             h.worst_distance, h.diversity, h.elapsed])
        st.download_button("下载 CSV", csv_buf.getvalue(),
                           file_name="aco-tsp-history.csv",
                           mime="text/csv", use_container_width=True)

    with col2:
        json_data = {
            "best_distance": result.best_distance,
            "best_tour": result.best_tour.tolist(),
            "cities": cities.tolist(),
            "total_iterations": result.total_iterations,
            "total_evaluations": result.total_evaluations,
            "total_time": result.total_time,
        }
        st.download_button("下载 JSON", json.dumps(json_data, indent=2, ensure_ascii=False),
                           file_name="aco-tsp-result.json",
                           mime="application/json", use_container_width=True)

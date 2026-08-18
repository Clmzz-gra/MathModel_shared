"""数值分析 Tab — 结果卡片、收敛统计、基因分布、导出。"""

import io, json, csv
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_result_cards(result, func_info: dict, params: dict):
    st.subheader("优化结果")

    gap = abs(result.best_energy - func_info["global_min"]) if func_info else None

    cols = st.columns(4)
    cols[0].metric(
        "最优值 f(x*)", f"{result.best_energy:.8g}",
        delta=f"距全局最优: {gap:.3g}" if gap is not None else None,
        delta_color="inverse",
    )
    cols[1].metric("进化代数", f"{result.total_generations}")
    cols[2].metric("总评估次数", f"{result.total_evaluations}")
    cols[3].metric("总耗时", f"{result.total_time:.2f}s")

    if gap is not None and gap < 1e-6:
        st.success("已找到全局最优解。", icon="🎯")
    elif gap is not None and gap < 0.1:
        st.info("接近全局最优解。")
    elif gap is not None:
        st.warning("还有距离，可尝试增大种群或代数。")

    st.markdown("**最优解 x*:**")
    x_str = ", ".join([f"x_{{{i+1}}} = {v:.6g}" for i, v in enumerate(result.best_x)])
    st.latex(rf"[{x_str}]")


def render_convergence_stats(history: list):
    """收敛代数统计。"""
    if not history:
        return

    st.subheader("收敛统计")
    best_energies = np.array([-h.best_fitness for h in history])  # 转回最小化
    final_best = best_energies[-1]

    thresholds = [0.1, 0.05, 0.01, 0.001]
    first = np.max(best_energies) - np.min(best_energies)
    scale = first if first > 1e-10 else 1.0
    results = []
    for t in thresholds:
        threshold_val = final_best + scale * t
        idx_arr = np.where(best_energies <= threshold_val)[0]
        if len(idx_arr) > 0:
            results.append({
                "相对误差": f"{t*100:.1f}%",
                "到达代数": idx_arr[0],
                "总代数占比": f"{idx_arr[0]/len(history)*100:.0f}%",
            })
        else:
            results.append({"相对误差": f"{t*100:.1f}%", "到达代数": "未到达", "总代数占比": "-"})

    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)


def render_population_stats(history: list):
    """最终种群统计。"""
    if not history:
        return
    last = history[-1]
    fits = last.fitnesses
    energies = -fits

    st.subheader("最终种群统计")
    cols = st.columns(4)
    cols[0].metric("最优能量", f"{np.min(energies):.6g}")
    cols[1].metric("平均能量", f"{np.mean(energies):.6g}")
    cols[2].metric("标准差", f"{np.std(energies):.4g}")
    cols[3].metric("多样性", f"{last.diversity:.4g}")

    # 适应度分布直方图
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=energies, nbinsx=20, marker_color="#3498db",
                               name="能量分布"))
    fig.update_layout(
        title="最终种群能量分布",
        xaxis_title="目标函数值 f(x)",
        yaxis_title="频数",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_multi_run_stats(all_results: list):
    """多次运行统计。"""
    if not all_results:
        return

    st.subheader("多次运行统计")
    bests = [r.best_energy for r in all_results]
    times = [r.total_time for r in all_results]

    stats = {
        "统计量": ["最优值", "均值", "最差值", "标准差"],
        "最优能量": [f"{min(bests):.6g}", f"{np.mean(bests):.6g}",
                    f"{max(bests):.6g}", f"{np.std(bests):.6g}"],
        "耗时(s)": [f"{min(times):.2f}", f"{np.mean(times):.2f}",
                    f"{max(times):.2f}", f"{np.std(times):.2f}"],
    }
    st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Box(y=bests, name="最优能量", marker_color="#27ae60"))
    fig.update_layout(title="多次运行最优能量分布", height=300,
                      margin=dict(l=50, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


def render_export_section(result, history: list, func_name: str):
    st.subheader("导出结果")
    col1, col2 = st.columns(2)

    with col1:
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["generation", "best_energy", "mean_energy", "worst_energy",
                          "best_fitness", "diversity", "elapsed"])
        for h in history:
            writer.writerow([h.generation, -h.best_fitness, -h.mean_fitness,
                             -h.worst_fitness, h.best_fitness, h.diversity, h.elapsed])
        st.download_button("下载 CSV", csv_buf.getvalue(),
                           file_name=f"ga-{func_name}-history.csv",
                           mime="text/csv", use_container_width=True)

    with col2:
        json_data = {
            "function": func_name,
            "best_energy": result.best_energy,
            "best_x": result.best_x.tolist(),
            "total_generations": result.total_generations,
            "total_evaluations": result.total_evaluations,
            "total_time": result.total_time,
        }
        st.download_button("下载 JSON", json.dumps(json_data, indent=2, ensure_ascii=False),
                           file_name=f"ga-{func_name}-result.json",
                           mime="application/json", use_container_width=True)
